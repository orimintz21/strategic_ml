# External imports
import torch
from torch import nn
from typing import Dict, Any

# Internal imports
from strategic_ml.cost_functions.cost_function import _CostFunction
from strategic_ml.gsc.non_linear_gp import _NonLinearGP


class NonLinearStrategicDelta(_NonLinearGP):
    def __init__(
        self,
        cost: _CostFunction,
        strategic_model: nn.Module,
        cost_weight: float = 1.0,
        *args,
        training_params: Dict[str, Any],
    ) -> None:
        super(NonLinearStrategicDelta, self).__init__(
            cost, strategic_model, cost_weight, training_params=training_params
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The data

        Returns:
            torch.Tensor: the modified data
        """
        # array of ones with the number of rows of x
        ones = torch.ones((x.shape[0], 1))
        return super().find_x_prime(x, ones)
