# External imports
import torch
from torch import nn

# Internal imports
from strategic_ml.cost_functions.cost_function import _CostFunction
from strategic_ml.gsc.linear_gp.linear_gp import _LinearGP
from strategic_ml.models import LinearModel


class LinearAdvDelta(_LinearGP):
    """
    Implements the LinearAdversarialDelta for strategic classification with linear models.

    This model assumes that the strategic users aim to get an incorrect prediction (i.e., 
    the opposite of their true label) with minimal cost. The delta is calculated using a 
    closed-form solution for linear models with an L2 or weighted L2 cost function.

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
        Initializes the LinearAdvDelta model.

        Args:
            cost (_CostFunction): The cost function used in the delta computation.
            strategic_model (nn.Module): The linear model used for the strategic classification.
            cost_weight (float, optional): The weight of the cost function. Defaults to 1.0.
            epsilon (float, optional): A small adjustment added to ensure correct model predictions. Defaults to 0.01.
        """
        super(LinearAdvDelta, self).__init__(
            cost, strategic_model, cost_weight, epsilon
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the delta for strategic users aiming to get an incorrect prediction.

        Args:
            x (torch.Tensor): The input data.
            y (torch.Tensor): The true label.

        Returns:
            torch.Tensor: The strategically modified data `x'`.
        """
        assert isinstance(self.strategic_model, LinearModel)
        device = self.strategic_model.model.weight.device
        x = x.to(device)
        y = y.to(device).to(self.strategic_model.model.weight.dtype)

        return super().find_x_prime(x, -y)

    def get_z(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Returns the metadata `z` for the cost function, which is `-y` in this case.

        Args:
            x (torch.Tensor): The input data.
            y (torch.Tensor): The true label.

        Returns:
            torch.Tensor: The metadata `z`, which is `-y`.
        """
        device = self.strategic_model.model.weight.device
        y = y.to(device).to(self.strategic_model.model.weight.dtype)
        return -y
