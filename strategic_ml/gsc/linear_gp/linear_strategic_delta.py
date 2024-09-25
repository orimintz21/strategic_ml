# External imports
import torch
from torch import nn
from typing import Optional

# Internal imports
from strategic_ml.cost_functions.cost_function import _CostFunction
from strategic_ml.gsc.linear_gp.linear_gp import _LinearGP


class LinearStrategicDelta(_LinearGP):
    """
    Implements the LinearStrategicDelta for strategic classification with linear models.

    This model assumes that the strategic users aim to maximize their prediction with minimal cost.
    The delta is calculated using a closed-form solution for linear models with an L2 or weighted L2 
    cost function.

    Attributes:
        cost (_CostFunction): The cost function used in the delta computation.
        strategic_model (nn.Module): The linear model used for the strategic classification.
        cost_weight (float): The weight of the cost function in the strategic calculation.
        epsilon (float): A small adjustment added to ensure correct model predictions.
    """

    def __init__(
        self,
        cost: _CostFunction,
        strategic_model: nn.Module,
        cost_weight: float = 1.0,
        epsilon: float = 0.01,
    ) -> None:
        """
        Initializes the LinearStrategicDelta model.

        Args:
            cost (_CostFunction): The cost function used in the delta computation.
            strategic_model (nn.Module): The linear model used for the strategic classification.
            cost_weight (float, optional): The weight of the cost function. Defaults to 1.0.
            epsilon (float, optional): A small adjustment added to ensure correct model predictions. Defaults to 0.01.
        """
        super(LinearStrategicDelta, self).__init__(
            cost, strategic_model, cost_weight, epsilon
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Computes the delta for strategic users aiming to maximize their prediction.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The strategically modified data `x'`.
        """
        # array of ones with the number of rows of x
        ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
        return super().find_x_prime(x, ones)

    def get_z(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Returns the metadata `z` for the strategic users, which is `1` in this case.

        Args:
            x (torch.Tensor): The input data.
            y (torch.Tensor): The true label.

        Returns:
            torch.Tensor: A tensor of ones, representing the desired prediction `z`.
        """
        batch_size = x.shape[0]
        return torch.ones((batch_size, 1), dtype=x.dtype, device=x.device)

    def get_minimal_distance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the minimal distance (cost) required for strategic users to 
        achieve a positive outcome.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The minimal cost required for a positive outcome.
        """
        ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
        return super()._get_minimal_distance(x, ones)
