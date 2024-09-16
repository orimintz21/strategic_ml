# External imports
import torch
from torch import nn
from typing import Optional

# Internal imports
from strategic_ml.cost_functions.cost_function import _CostFunction
from strategic_ml.gsc.linear_gp.linear_gp import _LinearGP


class LinearStrategicDelta(_LinearGP):
    """
    This is the LinearStrategicDelta model. This model assumes that the model
    is linear and that the cost function is L2 or weighted L2.
    The reason for this assumption is that we can calculate the delta in a
    closed form for linear models and not via GD or any other optimization
    algorithm. Therefore we do not need to train a delta model.
    In this case, the strategic users tries to maximize the prediction of the
    model with the minimal cost. The delta is calculated by the following formula:
    x_prime = argmax_{x' in X}(1{model(x') = 1} - r/2 * (cost(x,x')))
    For more information, see _LinearGP class, the paper "Strategic Classification Made Practical"
    , and the paper "Generalized Strategic Classification and the Case of Aligned Incentives".

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
        super(LinearStrategicDelta, self).__init__(
            cost, strategic_model, cost_weight, epsilon
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        The forward method of the LinearStrategicDelta model. This method calculates the delta
        for the strategic users.
        Note that the delta is calculated by a closed form and not by an optimization algorithm.
        It uses the find_x_prime method from the parent class with the label 1.
        Args:
            x (torch.Tensor): The data.
        """
        # array of ones with the number of rows of x
        ones = torch.ones((x.shape[0], 1))
        # ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
        return super().find_x_prime(x, ones)

    def get_z(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """This method returns the z value for the strategic users which is 1.

        Args:
            x (torch.Tensor): The data.
            y (torch.Tensor): The labels.

        Returns:
            torch.Tensor: An array of ones with the number of x samples.
        """
        batch_size = x.shape[0]
        return torch.ones((batch_size, 1))

    def get_minimal_distance(self, x: torch.Tensor) -> torch.Tensor:
        """This method returns the minimal distance of the strategic users.
        Which means the minimal cost that the strategic users can get pay
        to change the model prediction to positive.
        Note that this may be different from the distance between x and
        x_prime, because in this function we are not bounded by the model, so
        we can 'pay' more than the review of changing the model prediction.

        Args:
            x (torch.Tensor): The data.

        Returns:
            torch.Tensor: The minimal distance of the strategic users.
        """
        ones = torch.ones((x.shape[0], 1))
        return super()._get_minimal_distance(x, ones)
