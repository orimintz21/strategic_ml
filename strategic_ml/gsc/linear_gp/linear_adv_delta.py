"""
This module implements the LinearAdvDelta model for strategic classification.

The LinearAdvDelta model assumes that both the model is linear and the cost function 
is L2 or weighted L2. It calculates the delta in closed form for linear models without 
the need for gradient descent or any other optimization algorithm.

In this model, the strategic user aims to receive an incorrect prediction with minimal cost.

For more information, see the _LinearGP class and the paper:
- "Generalized Strategic Classification and the Case of Aligned Incentives"
"""

# External imports
import torch
from torch import nn

# Internal imports
from strategic_ml.cost_functions.cost_function import _CostFunction
from strategic_ml.gsc.linear_gp.linear_gp import _LinearGP


class LinearAdvDelta(_LinearGP):
    """
    This is the LinearAdvDelta model. This model assumes that the model
    is linear and that the cost function is L2 or weighted L2.

    The reason for this assumption is that we can calculate the delta in a
    closed form for linear models and not via gradient descent (GD) or any other 
    optimization algorithm. Therefore, we do not need to train a delta model.

    In this case, the strategic user tries to get the wrong prediction with the minimal cost.
    The delta is calculated by the following formula:
    x_prime = argmax_{x' in X}(1{model(x') = -y} - r/2 * (cost(x,x')))

    For more information, see the _LinearGP class and the paper
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
        """
        Initializes the LinearAdvDelta model.

        Args:
            cost (_CostFunction): The cost function for the delta, assumed to be L2 or weighted L2.
            strategic_model (nn.Module): The strategic model (assumed linear) on which the delta is calculated.
            cost_weight (float, optional): The weight of the cost function. Defaults to 1.0.
            epsilon (float, optional): A small value used to adjust the model's prediction direction 
                                       to ensure correct label prediction. Defaults to 0.01.
        """
        super(LinearAdvDelta, self).__init__(
            cost, strategic_model, cost_weight, epsilon
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the delta for the strategic user. 

        The delta is calculated in closed form (not through optimization). 
        This method uses the `find_x_prime` method from the parent class, with the label -y.

        Args:
            x (torch.Tensor): The input data.
            y (torch.Tensor): The label.

        Returns:
            torch.Tensor: The calculated delta for the strategic user.
        """
        return super().find_x_prime(x, -y)

    def get_z(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Returns the z value for the cost function, which in this case is -y.

        Args:
            x (torch.Tensor): The input data.
            y (torch.Tensor): The label.

        Returns:
            torch.Tensor: The z value, which is -y.
        """
        return -y
