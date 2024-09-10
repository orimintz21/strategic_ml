"""
This module implements the LinearStrategicDelta model for strategic classification.

The LinearStrategicDelta model calculates the delta in closed form for linear models
using L2 or weighted L2 cost functions. This delta is calculated without the need 
for gradient descent or other optimization algorithms. 

For more information, see the _LinearGP class and the papers:
- "Strategic Classification Made Practical"
- "Generalized Strategic Classification and the Case of Aligned Incentives"
"""

# External imports
import torch
from torch import nn
from typing import Optional

# Internal imports
from strategic_ml.cost_functions.cost_function import _CostFunction
from strategic_ml.gsc.linear_gp.linear_gp import _LinearGP


class LinearStrategicDelta(_LinearGP):
    """
    The LinearStrategicDelta model calculates the delta for strategic users assuming 
    a linear model and L2 or weighted L2 cost functions. The delta is computed in closed form,
    ensuring that the strategic users maximize the model's prediction with minimal cost.

    The delta is calculated using the formula:
    x_prime = argmax_{x' in X}(1{model(x') = 1} - r/2 * (cost(x, x')))
    
    Parent Class:
        _LinearGP
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
            cost (_CostFunction): The cost function for calculating the delta (L2 or weighted L2).
            strategic_model (nn.Module): The strategic model (assumed linear) that the delta is calculated on.
            cost_weight (float, optional): The weight of the cost function. Defaults to 1.0.
            epsilon (float, optional): A small value added to ensure correct label prediction by adjusting 
                                       the projection direction. Defaults to 0.01.
        """
        super(LinearStrategicDelta, self).__init__(
            cost, strategic_model, cost_weight, epsilon
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Calculates the delta for strategic users. The delta is computed in closed form 
        (not via optimization) and ensures the users maximize model predictions at minimal cost.

        This method uses the `find_x_prime` method from the parent class (_LinearGP) with the label 1.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The calculated delta for the strategic users.
        """
        # array of ones with the number of rows of x
        ones = torch.ones((x.shape[0], 1))
        return super().find_x_prime(x, ones)

    def get_z(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Returns the z value for the strategic users, which is always 1 for this model.

        Args:
            x (torch.Tensor): The input data.
            y (torch.Tensor): The labels.

        Returns:
            torch.Tensor: A tensor of ones with the number of samples in x.
        """
        batch_size = x.shape[0]
        return torch.ones((batch_size, 1))

    def get_minimal_distance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the minimal distance (cost) that strategic users must pay to 
        change the model's prediction to positive. 

        This may differ from the distance between x and x_prime, as this function is 
        not bounded by the model, and the users can "pay" more to achieve the desired prediction.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The minimal cost for the strategic users to change the model prediction.
        """
        ones = torch.ones((x.shape[0], 1))
        return super()._get_minimal_distance(x, ones)
