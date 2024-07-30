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
from typing import Optional

# Internal imports
from strategic_ml.cost_functions.cost_function import _CostFunction
from strategic_ml.gsc import _LinearGP
from strategic_ml.models.strategic_model import _StrategicModel


class LinearStrategicDelta(_LinearGP):
    def __init__(
        self,
        cost: _CostFunction,
        strategic_model: _StrategicModel,
        cost_weight: float = 1.0,
        models_temp: float = 1,
        z_temp: float = 1,
        margin_temp: float = 1,
    ) -> None:
        super(LinearStrategicDelta, self).__init__(
            cost, strategic_model, cost_weight, models_temp, z_temp, margin_temp
        )

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """This is the forward method of the LinearStrategicDelta model.
        This function calculates the delta based on the GP formula.
        Note that in this case the delta is a constant value of 1.
        For more information see paper "Generalized Strategic Data Augmentation".

        Args:
            x (torch.Tensor): The data
            z (torch.Tensor): No need for that argument,
            used for polymorphism. Defaults to None.

        Returns:
            torch.Tensor: the modified data
        """
        # array of ones with the number of rows of x
        ones = torch.ones((x.shape[0], 1))
        return super().forward(x, ones)
