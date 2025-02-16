"""
The code here is based on:
https://github.com/alecwangcq/KFAC-Pytorch/blob/master/optimizers/kfac.py
https://raw.githubusercontent.com/ntselepidis/KFAC-Pytorch/master/optimizers/kfac.py
"""

import copy
import math
from collections import defaultdict
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_kfac.utils.kfac_utils import (
    ComputeCovA,
    ComputeCovG,
    ExponentiallyDecayingFloat,
    KFACMemory,
    NumpyFiFo,
    PositiveDefiniteError,
    ScaleMemory,
    TrustRegionSize,
)


def cat_weight_bias(weight: Tensor, bias: Optional[Tensor] = None):
    if bias is not None:
        return torch.cat([weight, bias.view(-1, 1)], 1)
    return weight


def optional_clamp(x: torch.Tensor, min=None, max=None) -> torch.Tensor:
    if min is None and max is None:
        return x
    return torch.clamp(x, min=min, max=max)


def get_matrix_form_grad(m):
    """
    :param m: the layer
    :return: a matrix form of the gradient. it should be a [output_dim, input_dim] matrix.
    """
    classname = m.__class__.__name__
    if classname == "Conv2d":
        p_grad_mat = m.weight.grad.view(
            m.weight.grad.size(0), -1
        )  # n_filters * (in_c * kw * kh)
    else:
        p_grad_mat = m.weight.grad
    if m.bias is not None:
        p_grad_mat = cat_weight_bias(p_grad_mat, m.bias.grad)
    return p_grad_mat


class NewKFACOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        model,
        n_steps: int,
        kl_clip: Union[ExponentiallyDecayingFloat, TrustRegionSize],
        lr=0.001,
        momentum=0.9,
        stat_decay_factors=0.95,
        stat_decay_scale=0.9,
        damping=0.001,
        eta_max: float = 1.0,
        weight_decay=0,
        TCov=10,
        TInv=100,
        batch_averaged=True,
        solver="symeig",
        log_every: Optional[int] = None,
        min_damping: Optional[float] = None,
        grad_clip_val: float = 0.0,
        ekfac: bool = False,
        ekfac_factorwise_damping: bool = True,
        kronecker_factors_clip_val: Optional[float] = None,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        # Note: These defaults are never changed during the optimization.
        # In particular: The damping is not updated!
        # In the paper: Damping updated based on some LM-type rule.
        defaults = dict(
            lr=lr,
            momentum=momentum,
            damping=damping,
            weight_decay=weight_decay,
        )

        # TODO (CW): KFAC optimizer now only support model as input
        super(NewKFACOptimizer, self).__init__(model.parameters(), defaults)

        self.ekfac = ekfac
        self.ekfac_factorwise_damping = ekfac_factorwise_damping
        if self.ekfac:
            assert solver == "symeig", "EKFAC only supports symeig solver"
            make_scale_memory = lambda: ScaleMemory(stat_decay_scale)
            self.m_scale = defaultdict(make_scale_memory)

        self.model = model
        self.known_modules = {"Linear", "Conv2d"}
        self.modules = []
        self._register_modules()

        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.batch_averaged = batch_averaged

        self.TCov, self.TInv = TCov, TInv
        self.steps = 0

        self.kl_clip = kl_clip
        self.eta_max = eta_max

        # one-level KFAC vars
        self.solver = solver

        self.stat_decay, self.initial_stat_decay = (
            stat_decay_factors,
            stat_decay_factors,
        )
        make_aa_memory = lambda: KFACMemory(n_steps, stat_decay_factors, name="aa")
        make_gg_memory = lambda: KFACMemory(n_steps, stat_decay_factors, name="gg")
        self.m_aa, self.m_gg = defaultdict(make_aa_memory), defaultdict(make_gg_memory)

        if self.solver == "symeig":
            self.Q_a, self.Q_g = {}, {}  # eigenvectors of A and G
            if not self.ekfac or self.ekfac_factorwise_damping:
                self.d_a, self.d_g = {}, {}  # eigenvalues of A and G
        else:
            self.Inv_a, self.Inv_g = {}, {}

        self.time_scaling = 1 / n_steps

        self.min_damping = min_damping

        self.grad_clip_val = grad_clip_val
        self.clip_gradients = grad_clip_val > 0.0
        self.kronecker_factors_clip_val = kronecker_factors_clip_val

        if isinstance(self.kl_clip, TrustRegionSize):
            self.exp_improvement_memory = NumpyFiFo(self.kl_clip.memory_size)

        self.log_every = log_every
        self.reset_logs()

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            with torch.no_grad():
                aa = self.CovAHandler(input[0], module)
            # Initialize buffers
            if self.steps == 0:
                # TODO: initialize with zeros or diag?!
                # self.m_aa[module] = torch.zeros_like(aa)
                self.m_aa[module].initialize(torch.diag(aa.new(aa.size(0)).fill_(1)))
                # self.m_aa[module].initialize(torch.zeros_like(aa))
            self.m_aa[module].in_step_update(aa)

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.acc_stats and self.steps % self.TCov == 0:
            g = grad_output[0][self._sample_idx : self._sample_idx + 1]
            gg = self.CovGHandler(
                g,
                module,
                batch_averaged=False,
                clip_val=self.kronecker_factors_clip_val,
            )
            # Initialize buffers
            # TODO: initialize with zeros or diag?!
            if self.steps == 0:
                # self.m_gg[module] = torch.zeros_like(gg)
                self.m_gg[module].initialize(torch.diag(gg.new(gg.size(0)).fill_(1)))
                # self.m_gg[module].initialize(torch.zeros_like(gg))
            self.m_gg[module].in_step_update(gg)
            if self.logging_mode:
                g_max = self._logs["scalars"].get("g_max", None)
                g_max = g.max().item() if g_max is None else max(g_max, g.max().item())
                self._logs["scalars"]["g_max"] = g_max
                self._logs["gg_norm"][module].append(gg.norm().item())

    def _register_modules(self):
        self.id_to_layer_map = {}
        count = 0
        print(self.model)
        print("=> We keep following layers in KFAC. ")
        for name, module in self.model.named_modules():
            classname = module.__class__.__name__
            # print('=> We keep following layers in KFAC. <=')
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_full_backward_hook(self._save_grad_output)
                print("(%s): %s" % (count, module))
                count += 1
                self.id_to_layer_map[id(module)] = name

                if self.ekfac:
                    if module.bias is not None:
                        w = cat_weight_bias(module.weight.data, module.bias.data)
                    else:
                        w = module.weight.data
                    self.m_scale[module].initialize_like(w)

    def _update_inv(self, m):
        """Do eigen decomposition or approximate factorization for computing inverse of the ~ fisher.
        :param m: The layer
        :return: no returns.
        """
        self.m_aa[m].after_step_update()
        self.m_gg[m].after_step_update()

        m_aa = self.m_aa[m].average
        m_gg = self.m_gg[m].average

        if self.solver == "symeig":
            eig_vals_a, self.Q_a[m] = torch.linalg.eigh(m_aa)
            eig_vals_g, self.Q_g[m] = torch.linalg.eigh(m_gg)

            if not self.ekfac or self.ekfac_factorwise_damping:
                eps = 1e-10  # for numerical stability
                self.d_a[m], self.d_g[m] = eig_vals_a, eig_vals_g
                self.d_a[m].mul_((self.d_a[m] > eps).float())
                self.d_g[m].mul_((self.d_g[m] > eps).float())
        else:
            group = self.param_groups[0]
            damping = group["damping"]
            numer = m_aa.trace() * m_gg.shape[0]
            denom = m_gg.trace() * (m_aa.shape[0] + 1)
            pi_sq = numer / denom
            assert numer > 0, f"trace(A) should be positive, got {numer}"
            assert denom > 0, f"trace(G) should be positive, got {denom}"
            damping_a = optional_clamp((damping * pi_sq) ** 0.5, min=self.min_damping)
            damping_g = optional_clamp((damping / pi_sq) ** 0.5, min=self.min_damping)
            diag_a = m_aa.new_full((m_aa.shape[0],), damping_a)
            diag_g = m_gg.new_full((m_gg.shape[0],), damping_g)
            self.Inv_a[m] = (m_aa + torch.diag(diag_a)).inverse()
            self.Inv_g[m] = (m_gg + torch.diag(diag_g)).inverse()

    def update_scale(self) -> None:
        """
        Note: does not get called within this optimizer.
        To be called from outside.

        Advantage to reference implementation: Do not need to reconstruct the
        gradient based on the hooks, but can just access it via .grad attributes.

        Assumes single batch!
        """
        if self.steps == 0:
            return
        for m in self.modules:
            if m.bias is not None:  # gradient matrix becomes d_out x (d_in + 1)
                g = torch.cat([m.weight.grad.data, m.bias.grad.data.view(-1, 1)], 1)
            else:
                g = m.weight.grad.data

            # Projecting into the KFE via: Q_g @ mat(g) @ Q_a^T
            # and then squaring entrywise (this follows from deriviation in appendix)
            scale = (self.Q_g[m] @ g @ self.Q_a[m].t()) ** 2
            self.m_scale[m].update(scale)

    def _get_natural_grad(self, m, p_grad_mat, damping):
        """
        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        # p_grad_mat is of output_dim * input_dim
        # inv((ss')) p_grad_mat inv(aa') = [ Q_g (1/R_g) Q_g^T ] @ p_grad_mat @ [Q_a (1/R_a) Q_a^T]
        if self.solver == "symeig":
            v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]

            if self.ekfac:  # rescaling in the Kronecker factored Eigenspace (KFE)

                # The following damping mechanism is from Appendix C of EKFAC paper
                # TODO
                # Understand the logic behind this damping based on
                # the eigenvals of the Kronecker factors.
                if self.ekfac_factorwise_damping:
                    eps = damping
                    n_out, n_in = v1.shape
                    damp_a = math.sqrt(eps) * self.d_a[m].repeat_interleave(n_out)
                    damp_g = math.sqrt(eps) * self.d_g[m].repeat_interleave(n_in)
                    damping = eps + damp_a.reshape_as(v1) + damp_g.reshape_as(v1)

                    if self.logging_mode:
                        self._logs["scalars"]["mean_efac_damp"] = damping.mean()

                v2 = v1 / (self.m_scale[m].average + damping)
            else:
                v2 = v1 / (
                    self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + damping
                )

            v = self.Q_g[m] @ v2 @ self.Q_a[m].t()
        else:
            # This uses the identity:
            # (A^{-1} \otimes B^{-1}) vec(V) = vec(B^{-1} V A^{-1}^\top)
            v = self.Inv_g[m] @ p_grad_mat @ self.Inv_a[m]

        if m.bias is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.size())
            v[1] = v[1].view(m.bias.grad.size())
        else:
            v = [v.view(m.weight.grad.size())]

        if self.logging_mode:
            if m.bias is not None:
                conditioned_p_grad_mat = torch.cat([v[0], v[1].unsqueeze(-1)], dim=-1)
            else:
                conditioned_p_grad_mat = v[0]

            conditioned_p_grad_mat = conditioned_p_grad_mat.flatten()
            cos_sim = F.cosine_similarity(
                p_grad_mat.flatten(), conditioned_p_grad_mat, 0
            )
            self._logs["cos_sim_grad_natural_grad"][m].append(cos_sim)
            self._logs["norm_grad_over_norm_natural_grad"][m].append(
                p_grad_mat.norm() / conditioned_p_grad_mat.norm()
            )

        return v

    def _kl_clip(self, updates, lr):
        # This is the KL clip suggested in
        # distributed second-order optimization using kronecker-factored approximations
        # see eq.14 and the approximation below it

        # For a step direction v compute v^\top G.
        # Because G is block-diagonal, compute v^\top G one block at a time and sum.

        # ! In the original implementation: used lr**2
        # ! dont't get why; note: not just matter of definition; lr also used later)
        # ! We are not computing v^\top F v (then lr**2 would make sense); but v^\top G
        lr_fac = lr  # originally: lr**2

        vg_sum = 0
        for m in self.modules:
            v = updates[m]
            vg_sum += (v[0] * m.weight.grad * lr_fac).sum()
            if m.bias is not None:
                vg_sum += (v[1] * m.bias.grad * lr_fac).sum()

        if not vg_sum > 0:
            raise PositiveDefiniteError(vg_sum, updates)

        # Eq. 14 in Ba et al. (2016)
        eta = min(self.eta_max, (self.kl_clip.value / vg_sum) ** 0.5)

        if isinstance(self.kl_clip, TrustRegionSize):
            self.exp_improvement_memory.append(1.5 * eta * vg_sum)

        if self.logging_mode:
            self._logs["scalars"]["eta"] = eta
            self._logs["scalars"]["vg_sum"] = vg_sum

        return eta

    def _update_grads(self, updates, eta):
        # Set gradient to the previously computed updates * stepsize eta
        for m in self.modules:
            v = updates[m]
            m.weight.grad.copy_(v[0])
            m.weight.grad.mul_(eta)
            if m.bias is not None:
                m.bias.grad.copy_(v[1])
                m.bias.grad.mul_(eta)

    @torch.no_grad()
    def _step(self, closure):
        # FIXME (CW): Modified based on SGD (removed nestrov and dampening in momentum.)
        # FIXME (CW): 1. no nesterov, 2. buf.mul_(momentum).add_(1 <del> - dampening </del>, d_p)
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:  # and self.steps >= 20 * self.TCov:
                    d_p.add_(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                # p = p - lr * d_p
                p.add_(d_p, alpha=-group["lr"])

    def step(
        self,
        closure=None,
        linesearch_objective=None,  # Note: linesearch only for logging!
    ):
        group = self.param_groups[0]
        lr, damping = group["lr"], group["damping"]

        # track the gd updates when doing the line search logging
        if linesearch_objective is not None:
            gd_updates = {}

        updates = {}
        for m in self.modules:

            if self.steps % self.TInv == 0:
                self._update_inv(m)

            p_grad_mat = get_matrix_form_grad(m)
            v = self._get_natural_grad(m, p_grad_mat, damping)

            # TODO: check if I ever want to use this -> else remove
            if self.clip_gradients:  # clip the natural gradients
                v = [ng.clamp_(-self.grad_clip_val, self.grad_clip_val) for ng in v]

            updates[m] = v

            if linesearch_objective is not None:
                gd_updates[m] = p_grad_mat

        eta = self._kl_clip(updates, lr)  # stepsize based on trust region size

        if linesearch_objective is not None:
            max_gd_lr = self.linesearch(
                linesearch_objective, gd_updates, is_mat_form=True
            )

            # this maps trust region size to step size
            ssmap = lambda x: self._kl_clip(updates, x * 100)  # start with lr = 100

            max_ngd_tr = self.linesearch(linesearch_objective, updates, ssmap)

            self._logs["scalars"]["max_gd_lr"] = max_gd_lr
            self._logs["scalars"]["max_tr_size"] = max_ngd_tr
            self._logs["scalars"]["max_ngd_lr"] = ssmap(max_ngd_tr)

        self._update_grads(updates, eta)
        self._step(closure)

        self.steps += 1
        self.stat_decay = min(
            1.0 - 1.0 / (self.steps // self.TCov + 1), self.initial_stat_decay
        )

        if isinstance(self.kl_clip, ExponentiallyDecayingFloat):
            self.kl_clip.step()
        elif isinstance(self.kl_clip.max_value, ExponentiallyDecayingFloat):
            self.kl_clip.max_value.step()

    @property
    def logging_mode(self) -> bool:
        if self.log_every is None:
            return False
        return self.steps % self.log_every == 0

    def get_logs(self) -> dict:
        logs = self._logs
        self.reset_logs()
        return logs

    def reset_logs(self):
        self._logs = defaultdict(lambda: defaultdict(list))

    @torch.no_grad()
    def linesearch(
        self,
        objective,
        updates,
        search_scalar_to_step_map=None,
        max_iter=10,
        is_mat_form=False,
    ):

        base_eval = objective(self.model)

        def eval(scalar):
            if search_scalar_to_step_map is not None:
                scalar = search_scalar_to_step_map(scalar)

            model_copy = copy.deepcopy(self.model)
            j = 0
            for m in model_copy.modules():
                if isinstance(m, torch.nn.Linear):
                    _m = self.modules[j]
                    v = updates[_m]
                    if is_mat_form and m.bias is not None:
                        v = [v[:, :-1], v[:, -1:]]
                        v[0] = v[0].view(m.weight.data.size())
                        v[1] = v[1].view(m.bias.data.size())
                    m.weight.add_(v[0], alpha=-scalar)
                    if m.bias is not None:
                        m.bias.add_(v[1], alpha=-scalar)
                    j += 1
            return objective(model_copy)

        x = 1.0
        for i in range(max_iter):
            current_eval = eval(x)
            if current_eval > base_eval:  # goal: minimize
                x /= 10
            else:
                break

        if i == max_iter - 1:
            return 0.0

        # now finer search
        best_x = x
        low, high = x, 10 * x
        for _ in range(5):
            mid = (high + low) / 2
            mid_eval = eval(mid)
            if mid_eval > current_eval:
                low = mid
                best_x = mid
            else:
                high = mid

        return best_x
