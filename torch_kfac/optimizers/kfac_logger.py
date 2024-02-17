from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch_kfac.utils.kfac_utils import ComputeCovA, ComputeCovG


@dataclass
class PerTimeStepMemory:
    n_steps: int
    name: Optional[str] = None
    _counter: int = 0

    def __post_init__(self):
        self.samples = []

    def in_step_update(self, x: Tensor) -> None:
        self.samples.append(x)
        self._counter += 1


class KFACLogger:
    def __init__(self, n_steps: int):

        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()

        make_aa_memory = lambda: PerTimeStepMemory(n_steps, name="aa")
        make_gg_memory = lambda: PerTimeStepMemory(n_steps, name="gg")
        self.m_aa, self.m_gg = defaultdict(make_aa_memory), defaultdict(make_gg_memory)

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
