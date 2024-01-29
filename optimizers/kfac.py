import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.optim as optim

from utils.kfac_utils import ComputeCovA, ComputeCovG, update_running_stat


@dataclass
class KFACMemory:
    n_steps: int
    stat_decay: float
    name: Optional[str] = None

    def __post_init__(self):
        self.tensors = [None] * self.n_steps
        self._counter = 0

    def update_counter(self):
        c = self._counter + 1
        c = 0 if c == self.n_steps else c
        self._counter = c

    def initialize(self, tensor):
        self.tensors[self._counter] = tensor

    def update(self, tensor):
        update_running_stat(tensor, self.tensors[self._counter], self.stat_decay)
        self.update_counter()

    def mean(self):
        return torch.stack(self.tensors).mean(0)


@dataclass
class LowKFACMemory:
    stat_decay: float
    name: Optional[str] = None
    _counter: int = 0

    def initialize(self, tensor):
        self.average, self.running_sum = tensor, tensor

    def in_step_update(self, tensor):
        self.running_sum += tensor
        self._counter += 1

    def after_step_update(self):
        self.running_sum /= self._counter
        self._counter = 0
        update_running_stat(self.running_sum, self.average, self.stat_decay)


def compute_gHg(g, H):
    ...


class KFACOptimizer(optim.Optimizer):
    def __init__(
        self,
        model,
        n_model_steps: int,
        lr=0.001,
        momentum=0.9,
        stat_decay=0.95,
        damping=0.001,
        kl_clip=0.001,
        weight_decay=0,
        TCov=10,
        TInv=100,
        batch_averaged=True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(
            lr=lr, momentum=momentum, damping=damping, weight_decay=weight_decay
        )
        # TODO (CW): KFAC optimizer now only support model as input
        super(KFACOptimizer, self).__init__(model.parameters(), defaults)
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()

        self.batch_averaged = batch_averaged

        self.known_modules = {"Linear", "Conv2d"}

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0
        self.first_module = None

        # TODO: this could be more efficient if I treat the time and
        # TODO: batch dimension equally.
        # self.m_aa, self.m_gg = defaultdict(list), defaultdict(list)
        make_aa_memory = lambda: LowKFACMemory(stat_decay=stat_decay, name="aa")
        make_gg_memory = lambda: LowKFACMemory(stat_decay=stat_decay, name="gg")
        self.m_aa, self.m_gg = defaultdict(make_aa_memory), defaultdict(make_gg_memory)

        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}

        # self.Q_a, self.Q_g = defaultdict(list), defaultdict(list)
        # self.d_a, self.d_g = defaultdict(list), defaultdict(list)

        self.stat_decay = stat_decay

        self.kl_clip = kl_clip
        self.TCov = TCov
        self.TInv = TInv

        self.H = None

    def _save_input(self, module, input):
        if self.first_module is None:
            self.first_module = hash(module)
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            aa = self.CovAHandler(input[0].data, module)
            if self.steps == 0:
                self.m_aa[module].initialize(torch.diag(aa.new(aa.size(0)).fill_(1)))
            self.m_aa[module].in_step_update(aa)

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.acc_stats and self.steps % self.TCov == 0:
            gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
            if self.steps == 0:
                self.m_gg[module].initialize(torch.diag(gg.new(gg.size(0)).fill_(1)))
            self.m_gg[module].in_step_update(gg)

    def _prepare_model(self):
        count = 0
        print(self.model)
        print("=> We keep following layers in KFAC. ")
        for module in self.model.modules():
            classname = module.__class__.__name__
            # print('=> We keep following layers in KFAC. <=')
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_full_backward_hook(self._save_grad_output)
                print("(%s): %s" % (count, module))
                count += 1

    def _update_inv(self, m):
        """Do eigen decomposition for computing inverse of the ~ fisher.
        :param m: The layer
        :return: no returns.
        """
        eps = 1e-10  # for numerical stability

        self.m_aa[m].after_step_update()
        self.m_gg[m].after_step_update()
        self.d_a[m], self.Q_a[m] = torch.linalg.eigh(self.m_aa[m].average)
        self.d_g[m], self.Q_g[m] = torch.linalg.eigh(self.m_gg[m].average)

        self.d_a[m].mul_((self.d_a[m] > eps).float())
        self.d_g[m].mul_((self.d_g[m] > eps).float())

    @staticmethod
    def _get_matrix_form_grad(m, classname):
        """
        :param m: the layer
        :param classname: the class name of the layer
        :return: a matrix form of the gradient. it should be a [output_dim, input_dim] matrix.
        """
        if classname == "Conv2d":
            p_grad_mat = m.weight.grad.data.view(
                m.weight.grad.data.size(0), -1
            )  # n_filters * (in_c * kw * kh)
        else:
            p_grad_mat = m.weight.grad.data
        if m.bias is not None:
            p_grad_mat = torch.cat([p_grad_mat, m.bias.grad.data.view(-1, 1)], 1)
        return p_grad_mat

    def _get_natural_grad(self, m, p_grad_mat, damping):
        """
        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        # p_grad_mat is of output_dim * input_dim
        # inv((ss')) p_grad_mat inv(aa') = [ Q_g (1/R_g) Q_g^T ] @ p_grad_mat @ [Q_a (1/R_a) Q_a^T]
        v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
        v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + damping)
        v = self.Q_g[m] @ v2 @ self.Q_a[m].t()
        if m.bias is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]

        return v

    def _kl_clip_and_update_grad(self, updates, lr):
        # do kl clip
        vg_sum = 0
        for m in self.modules:
            v = updates[m]
            vg_sum += (v[0] * m.weight.grad.data * lr**2).sum().item()
            if m.bias is not None:
                vg_sum += (v[1] * m.bias.grad.data * lr**2).sum().item()
        nu = min(1.0, math.sqrt(self.kl_clip / vg_sum))

        for m in self.modules:
            v = updates[m]
            m.weight.grad.data.copy_(v[0])
            m.weight.grad.data.mul_(nu)
            if m.bias is not None:
                m.bias.grad.data.copy_(v[1])
                m.bias.grad.data.mul_(nu)

    def _step(self, closure):
        # FIXME (CW): Modified based on SGD (removed nestrov and dampening in momentum.)
        # FIXME (CW): 1. no nesterov, 2. buf.mul_(momentum).add_(1 <del> - dampening </del>, d_p)
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0 and self.steps >= 20 * self.TCov:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1.0)
                    d_p = buf

                p.data.add_(d_p, alpha=-group["lr"])

    def step(self, closure=None):
        # FIXME(CW): temporal fix for compatibility with Official LR scheduler.
        group = self.param_groups[0]
        lr = group["lr"]
        damping = group["damping"]
        updates = {}
        for m in self.modules:
            classname = m.__class__.__name__
            if self.steps % self.TInv == 0:
                self._update_inv(m)
            p_grad_mat = self._get_matrix_form_grad(m, classname)
            v = self._get_natural_grad(m, p_grad_mat, damping)
            updates[m] = v
        self._kl_clip_and_update_grad(updates, lr)

        self._step(closure)
        self.steps += 1
        self.t = -1
