# External imports
import torch
from torch import nn
from typing import Tuple, Dict, Any

# Internal imports
from strategic_ml.cost_functions.cost_function import _CostFunction
from strategic_ml.gsc.linear_gp.linear_gp import _LinearGP


class LinearAdvDelta(_LinearGP):
    """
    This is the LinearAdvDelta model. This model assumes that the model
    is linear and that the cost function is L2 or weighted L2.
    The reason for this assumption is that we can calculate the delta in a
    closed form for linear models and not via GD or any other optimization
    algorithm. Therefore we do not need to train a delta model.
    In this case, the strategic users tries to get the wrong prediction
    with the minimal cost. The delta is calculated by the following formula:
    x_prime = argmax_{x' in X}(1{model(x') = -y} - r/2 * (cost(x,x')))
    For more information, see _LinearGP class, the paper
    "Generalized Strategic Classification and the Case of Aligned Incentives".

    Parent Class: _LinearGP
    """

    def __init__(
        self,
        cost: _CostFunction,
        strategic_model: nn.Module,
        cost_weight: float = 1.0,
        epsilon: float = 0.01,
    ) -> None:
        """Initializer for the LinearStrategicDelta model.

        Args:
            cost (_CostFunction): The cost function of the delta, we assume that the cost is L2 or weighted L2.
            strategic_model (nn.Module): The strategic model that the delta is calculated on, we assume that the model is linear.
            cost_weight (float, optional): The weight of the cost function. Defaults to 1.0.
            epsilon (float): move to the negative/positive direction of the model
            to make sure that the model will predict the label correctly. The
            delta does it by adding the (epsilon * w/||w||). Defaults to 0.01.
        """
        super(LinearAdvDelta, self).__init__(
            cost, strategic_model, cost_weight, epsilon
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        The forward method of the LinearAdvDelta model. This method calculates the delta
        for the strategic users.
        Note that the delta is calculated by a closed form and not by an optimization algorithm.
        It uses the find_x_prime method from the parent class with the label -y.
        Args:
            x (torch.Tensor): The data.
            y (torch.Tensor): The label.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: The delta and the logs.
        """
        return super().find_x_prime(x, -y)

    def get_z(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the z value for the cost function. In this case, the z value is -y.

        Args:
            x (torch.Tensor): The data.
            y (torch.Tensor): The label.

        Returns:
            torch.Tensor: The z value, which is -y.
        """
        return -y
