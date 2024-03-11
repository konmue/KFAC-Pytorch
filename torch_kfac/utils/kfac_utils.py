from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def try_contiguous(x):
    if not x.is_contiguous():
        x = x.contiguous()

    return x


def _extract_patches(x, kernel_size, stride, padding):
    """
    :param x: The input feature maps.  (batch_size, in_c, h, w)
    :param kernel_size: the kernel size of the conv filter (tuple of two elements)
    :param stride: the stride of conv operation  (tuple of two elements)
    :param padding: number of paddings. be a tuple of two elements
    :return: (batch_size, out_h, out_w, in_c*kh*kw)
    """
    if padding[0] + padding[1] > 0:
        x = F.pad(
            x, (padding[1], padding[1], padding[0], padding[0])
        ).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(x.size(0), x.size(1), x.size(2), x.size(3) * x.size(4) * x.size(5))
    return x


def update_running_stat(aa, m_aa, stat_decay):
    # using inplace operation to save memory!
    m_aa *= stat_decay / (1 - stat_decay)
    m_aa += aa
    m_aa *= 1 - stat_decay


class ComputeMatGrad:
    @classmethod
    def __call__(cls, input, grad_output, layer):
        if isinstance(layer, nn.Linear):
            grad = cls.linear(input, grad_output, layer)
        elif isinstance(layer, nn.Conv2d):
            grad = cls.conv2d(input, grad_output, layer)
        else:
            raise NotImplementedError
        return grad

    @staticmethod
    def linear(input, grad_output, layer):
        """
        :param input: batch_size * input_dim
        :param grad_output: batch_size * output_dim
        :param layer: [nn.module] output_dim * input_dim
        :return: batch_size * output_dim * (input_dim + [1 if with bias])
        """
        with torch.no_grad():
            if layer.bias is not None:
                input = torch.cat([input, input.new(input.size(0), 1).fill_(1)], 1)
            input = input.unsqueeze(1)
            grad_output = grad_output.unsqueeze(2)
            grad = torch.bmm(grad_output, input)
        return grad

    @staticmethod
    def conv2d(input, grad_output, layer):
        """
        :param input: batch_size * in_c * in_h * in_w
        :param grad_output: batch_size * out_c * h * w
        :param layer: nn.module batch_size * out_c * (in_c*k_h*k_w + [1 if with bias])
        :return:
        """
        with torch.no_grad():
            input = _extract_patches(
                input, layer.kernel_size, layer.stride, layer.padding
            )
            input = input.view(-1, input.size(-1))  # b * hw * in_c*kh*kw
            grad_output = grad_output.transpose(1, 2).transpose(2, 3)
            grad_output = try_contiguous(grad_output).view(
                grad_output.size(0), -1, grad_output.size(-1)
            )
            # b * hw * out_c
            if layer.bias is not None:
                input = torch.cat([input, input.new(input.size(0), 1).fill_(1)], 1)
            input = input.view(
                grad_output.size(0), -1, input.size(-1)
            )  # b * hw * in_c*kh*kw
            grad = torch.einsum("abm,abn->amn", (grad_output, input))
        return grad


class ComputeCovA:
    @classmethod
    def compute_cov_a(cls, a, layer):
        return cls.__call__(a, layer)

    @classmethod
    def __call__(cls, a, layer):
        if isinstance(layer, nn.Linear):
            cov_a = cls.linear(a, layer)
        elif isinstance(layer, nn.Conv2d):
            cov_a = cls.conv2d(a, layer)
        else:
            # FIXME(CW): for extension to other layers.
            # raise NotImplementedError
            cov_a = None

        return cov_a

    @staticmethod
    def conv2d(a, layer):
        batch_size = a.size(0)
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding)
        spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        a = a / spatial_size
        # FIXME(CW): do we need to divide the output feature map's size?
        return a.t() @ (a / batch_size)

    @staticmethod
    def linear(a, layer):
        # a: batch_size * in_dim
        batch_size = a.size(0)
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        return a.t() @ (a / batch_size)


class ComputeCovG:
    @classmethod
    def compute_cov_g(cls, g, layer, batch_averaged=False):
        """
        :param g: gradient
        :param layer: the corresponding layer
        :param batch_averaged: if the gradient is already averaged with the batch size?
        :return:
        """
        # batch_size = g.size(0)
        return cls.__call__(g, layer, batch_averaged)

    @classmethod
    def __call__(cls, g, layer, batch_averaged):
        if isinstance(layer, nn.Conv2d):
            cov_g = cls.conv2d(g, layer, batch_averaged)
        elif isinstance(layer, nn.Linear):
            cov_g = cls.linear(g, layer, batch_averaged)
        else:
            cov_g = None

        return cov_g

    @staticmethod
    def conv2d(g, layer, batch_averaged):
        # g: batch_size * n_filters * out_h * out_w
        # n_filters is actually the output dimension (analogous to Linear layer)
        spatial_size = g.size(2) * g.size(3)
        batch_size = g.shape[0]
        g = g.transpose(1, 2).transpose(2, 3)
        g = try_contiguous(g)
        g = g.view(-1, g.size(-1))

        if batch_averaged:
            g = g * batch_size
        g = g * spatial_size
        cov_g = g.t() @ (g / g.size(0))

        return cov_g

    @staticmethod
    def linear(g, layer, batch_averaged):
        # g: batch_size * out_dim
        batch_size = g.size(0)
        assert g.ndim == 2

        if batch_averaged:
            cov_g = g.t() @ (g * batch_size)
        else:
            cov_g = g.t() @ (g / batch_size)
        return cov_g


@dataclass
class ScaleMemory:
    stat_decay: float = 0.99

    def initialize_like(self, x: Tensor) -> None:
        self.average = torch.ones_like(x)

    def update(self, x: Tensor) -> None:
        self.average *= self.stat_decay / (1 - self.stat_decay)
        self.average += x
        self.average *= 1 - self.stat_decay


@dataclass
class KFACMemory:
    """
    In the original RNN KFAC, they just define:
        V_0 := E[a_s a_t^T] s.t. s = t
    - they don't specify what t to use
    - which I guess means you average over all t's

    Thus, this code uses:
        T * V_0 <- (1/ batch_size) \sum_i \sum_t (a^i)_t (a^i)_t^T
    i.e., approximate V_0 by averaging over time!
        -> actually summing as we need T * V_0

    TODO: Is this the correct approach?
    """

    n_steps: int
    stat_decay: float = 0.99
    name: Optional[str] = None
    _counter: int = 0
    sum_over_time: bool = False

    @cached_property
    def time_scale(self) -> float:
        return 1 / (self.n_steps**0.5)

    @property
    def n_samples(self):
        return self._counter // self.n_steps

    def initialize(self, x: Tensor) -> None:
        self.average, self._running_sum = x, torch.zeros_like(x)

    def _update_average(self, new_mean: Tensor) -> None:
        self.average *= self.stat_decay / (1 - self.stat_decay)
        self.average += new_mean
        self.average *= 1 - self.stat_decay

    def in_step_update(self, x: Tensor) -> None:
        self._running_sum += x
        self._counter += 1

    def after_step_update(self) -> None:
        if self.n_samples > 0:
            if self.sum_over_time:
                self._update_average(self._running_sum / self.n_samples)

            # NOTE: This should be the correct approach
            else:
                fac = self.time_scale * (1 / self.n_samples)
                self._update_average(self._running_sum * fac)

        self._running_sum = torch.zeros_like(self._running_sum)
        self._counter = 0


class PositiveDefiniteError(Exception):
    def __init__(self, vg_sum, updates):
        m = "vg_sum should be positive"
        message = f"{m}, got {vg_sum}.\n"

        for i, (k, v) in enumerate(updates.items()):
            for _v in v:
                if _v.isnan().any():
                    n_nans = _v.isnan().sum()
                    m = f"Got nan update with {n_nans} nans for {i}th param group {k}."
                    message += f"{m}\n"
        self.message = message
        super().__init__(self.message)


class NumpyFiFo:
    def __init__(self, max_lenght: int) -> None:
        self.max_lenght = max_lenght
        self.data = np.zeros(max_lenght)
        self.filled = False
        self.index = 0

    def append(self, value: float) -> None:
        self.data[self.index] = value
        self.index += 1
        if self.index == self.max_lenght:
            self.index = 0
            self.filled = True

    def __call__(self) -> np.ndarray:
        return self.data

    def diff(self) -> np.ndarray:
        if self.filled:
            return np.diff(np.roll(self.data, -self.index))
        else:
            return np.diff(self.data[: self.index])


@dataclass
class ExponentiallyDecayingFloat:
    initial_value: float
    decay: float = 0.99
    update_every: int = 1
    min_value: Optional[float] = None

    def __post_init__(self) -> None:
        self.value, self._count = self.initial_value, 0

    def step(self) -> None:
        if self._count % self.update_every == 0:
            self.value = max(self.decay * self.value, self.min_value or 0.0)
        self._count += 1


@dataclass
class TrustRegionSize:
    max_value_params: dict
    min_value: float
    no_trust_threshold: float = 0.25
    no_trust_downscale: float = 0.25
    max_trust_threshold: float = 0.75
    max_trust_upscale: float = 2.0
    update_every: int = 1
    memory_size: int = 10
    mean_of_ratios: bool = False
    # probably makes sense to be much slower in adjusting the regions than
    # in classical optimization

    def __post_init__(self) -> None:
        self.max_value = ExponentiallyDecayingFloat(**self.max_value_params)
        if self.max_value.min_value is None:
            self.max_value.min_value = self.min_value
        else:
            self.max_value.min_value = max(self.min_value, self.max_value.min_value)
        self.value = self.max_value.value  # initialize with max_value

    def clamp(self) -> None:
        self.value = max(self.min_value, self.value)
        self.value = min(self.max_value.value, self.value)

    def step(
        self, actual_improvement: np.ndarray, promised_improvement: np.ndarray
    ) -> None:

        if not self.mean_of_ratios:
            improvement_ratios = actual_improvement / promised_improvement
            improvement_ratio = improvement_ratios.mean()
        else:
            improvement_ratio = actual_improvement.mean() / promised_improvement.mean()

        if improvement_ratio < self.no_trust_threshold:
            self.value *= self.no_trust_downscale

        elif improvement_ratio > self.max_trust_threshold:
            self.value *= self.max_trust_upscale

        self.clamp()
