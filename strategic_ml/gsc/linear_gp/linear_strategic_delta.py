"""file: linear_strategic_delta.py
This is the LinearStrategicDelta model.
The LinearStrategicDelta model calculates the delta based on the GP formula.
Note that in this case the delta is a constant value of 1.
The strategic delta calculates the movement of a strategic user, which tries
to get a positive outcome from the model (this is the reason that z==1).
For more information see paper "Generalized Strategic Data Augmentation" and
"Strategic Classification Made Practical".
"""

# External imports
import torch
from torch import nn
from typing import Optional

# Internal imports
from strategic_ml.cost_functions.cost_function import _CostFunction
from strategic_ml.gsc.linear_gp.linear_gp import _LinearGP


class LinearStrategicDelta(_LinearGP):
    def __init__(
        self,
        cost: _CostFunction,
        strategic_model: nn.Module,
        cost_weight: float = 1.0,
        epsilon: float = 0.01,
    ) -> None:
        super(LinearStrategicDelta, self).__init__(
            cost, strategic_model, cost_weight, epsilon
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """This is the forward method of the LinearStrategicDelta model.
        This function calculates the delta based on the GP formula.
        Note that in this case the delta is a constant value of 1.
        For more information see paper "Generalized Strategic Data Augmentation".

        Args:
            x (torch.Tensor): The data

        Returns:
            torch.Tensor: the modified data
        """
        # array of ones with the number of rows of x
        ones = torch.ones((x.shape[0], 1))
        return super().find_x_prime(x, ones)

    def get_z(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return torch.ones((batch_size, 1))

    def get_minimal_distance(self, x: torch.Tensor) -> torch.Tensor:
        ones = torch.ones((x.shape[0], 1))
        return super()._get_minimal_distance(x, ones)
