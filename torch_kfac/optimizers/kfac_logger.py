from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch_kfac.utils.kfac_utils import ComputeCovA, ComputeCovG


@dataclass
class PerTimeStepMemory:
    name: Optional[str] = None
    _counter: int = 0

    def __post_init__(self):
        self.samples = []

    def in_step_update(self, x: Tensor) -> None:
        self.samples.append(x)
        self._counter += 1


class KFACLoggerAAGG:
    def __init__(self):
        """
        This Logger loggs aa and gg for each time step seperately.
        """

        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()

        make_aa_memory = lambda: PerTimeStepMemory(name="aa")
        make_gg_memory = lambda: PerTimeStepMemory(name="gg")
        self.m_aa, self.m_gg = defaultdict(make_aa_memory), defaultdict(make_gg_memory)

    def clear_memory(self):
        for k in self.m_aa.keys():
            self.m_aa[k].samples = []
            self.m_gg[k].samples = []

    def register_hooks(self, model):
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.register_forward_pre_hook(self._save_input)
                module.register_full_backward_hook(self._save_grad_output)

    @torch.no_grad()
    def _save_input(self, module, input):
        aa = self.CovAHandler(input[0], module)
        self.m_aa[module].in_step_update(aa)

    def _save_grad_output(self, module, grad_input, grad_output):
        # batch average does not matter when using single input
        gg = self.CovGHandler(grad_output[0], module, batch_averaged=True)
        self.m_gg[module].in_step_update(gg)


class KFACLoggerAG:
    """
    This logger logs a and g for each time step seperately.
    """

    def __init__(self):
        make_a_memory = lambda: PerTimeStepMemory(name="a")
        make_g_memory = lambda: PerTimeStepMemory(name="g")
        self.m_a, self.m_g = defaultdict(make_a_memory), defaultdict(make_g_memory)

    def clear_memory(self):
        for k in self.m_a.keys():
            self.m_a[k].samples = []
            self.m_g[k].samples = []

    def register_hooks(self, model):
        handles = []
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                handle_fwd = module.register_forward_pre_hook(self._save_input)
                handle_bwd = module.register_full_backward_hook(self._save_grad_output)
                handles.extend([handle_fwd, handle_bwd])
        return handles

    @torch.no_grad()
    def _save_input(self, module, a):
        a = a[0]
        if module.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        self.m_a[module].in_step_update(a)

    def _save_grad_output(self, module, grad_input, grad_output):
        self.m_g[module].in_step_update(grad_output[0])
