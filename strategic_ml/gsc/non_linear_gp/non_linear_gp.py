"""_summary
"""

# External imports
import torch
from torch import nn
from typing import Optional

# Internal imports
from strategic_ml.cost_functions.cost_function import _CostFunction
from strategic_ml.gsc.generalized_strategic_delta import _GSC


class _NonLinearGP(_GSC):
    def __init__(
        self, strategic_model: nn.Module, cost: _CostFunction, cost_weight: float = 1, optimizer: torch.optim.Optimizer, 
    ) -> None:
        super().__init__(strategic_model, cost, cost_weight)
    
