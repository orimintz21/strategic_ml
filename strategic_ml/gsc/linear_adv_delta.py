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
from strategic_ml.gsc import _LinearGP


class LinearAdvDelta(_LinearGP):
    def __init__(
        self,
        cost: _CostFunction,
        strategic_model: nn.Module,
        cost_weight: float = 1.0,
        models_temp: float = 1.0,
        z_temp: float = 1.0,
        margin_temp: float = 1.0,
    ) -> None:
        super(LinearAdvDelta, self).__init__(
            cost, strategic_model, cost_weight, models_temp, z_temp, margin_temp
        )

    def forward(
        self, x: torch.Tensor, z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): The data
            z (torch.Tensor): In this case we use -y as z. Use y as the input. Defaults to None.

        Returns:
            torch.Tensor: the modified data
        """
        assert z is not None
        return super().forward(x, -z)
