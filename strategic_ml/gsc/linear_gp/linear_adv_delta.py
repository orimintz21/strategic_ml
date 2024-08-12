"""file: linear_adv_delta.py
This is the LinearAdvDelta model.
The LinearAdvDelta model calculates the delta based on the GP formula.
The strategic delta calculates the movement of a user that tries to trick the
model to get a negative outcome (this is the reason that z==-y).
"""

# External imports
import torch
from torch import nn
from typing import Optional

# Internal imports
from strategic_ml.cost_functions.cost_function import _CostFunction
from strategic_ml.gsc.linear_gp.linear_gp import _LinearGP


class LinearAdvDelta(_LinearGP):
    def __init__(
        self,
        cost: _CostFunction,
        strategic_model: nn.Module,
        cost_weight: float = 1.0,
        epsilon: float = 0.01,
    ) -> None:
        super(LinearAdvDelta, self).__init__(
            cost, strategic_model, cost_weight, epsilon
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor 
    ) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): The data
            y (torch.Tensor): The label.

        Returns:
            torch.Tensor: the modified data
        """
        return super().find_x_prime(x, -y)
