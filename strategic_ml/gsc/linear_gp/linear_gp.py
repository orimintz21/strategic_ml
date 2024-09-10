"""
This module implements the _LinearGP class for Generalized Strategic Classification.
The GP is a strategic delta that can be calculated in closed form for linear models.
It handles both L2 and weighted L2 cost functions, projecting inputs to minimize costs.
"""

# External imports
import torch
from torch import nn
from typing import Optional

# Internal imports
from strategic_ml.gsc.generalized_strategic_delta import _GSC
from strategic_ml.models import LinearStrategicModel
from strategic_ml.cost_functions import (
    _CostFunction,
    CostNormL2,
    CostWeightedLoss,
)


class _LinearGP(_GSC):
    """
    The _LinearGP class calculates the strategic delta using a closed-form formula.
    The delta is calculated based on the projection of the input on the linear model, 
    minimizing the cost to achieve a desired prediction.

    The GP is computed as:
        x_prime = x - ((w.T @ x + b) / (w.T @ w)) * w  (for L2 cost)
        x_prime = x - ((w.T @ x + b) / (w.T @ inverse(Lambda) @ w)) * inverse(Lambda) @ w  (for weighted L2 cost)

    Parent Class:
        _GSC
    """

    def __init__(
        self,
        cost: _CostFunction,
        strategic_model: nn.Module,
        cost_weight: float = 1.0,
        epsilon: float = 0.01,
    ) -> None:
        """
        Initializes the _LinearGP model, assuming the model is linear. 
        The GP can be calculated in closed form without training a delta model.

        Args:
            cost (_CostFunction): The cost function of the delta.
            strategic_model (nn.Module): The linear model on which the delta is calculated.
            cost_weight (float, optional): The weight of the cost function. Defaults to 1.0.
            epsilon (float, optional): A small value to ensure correct prediction by 
                                       adjusting the projection direction. Defaults to 0.01.
        """
        super(_LinearGP, self).__init__(
            strategic_model=strategic_model, cost=cost, cost_weight=cost_weight
        )
        self._assert_model()
        self._assert_cost()
        self.epsilon: float = epsilon

    def find_x_prime(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Calculates x' (the projected input) using the GP formula.

        Args:
            x (torch.Tensor): The input data.
            z (torch.Tensor): Meta data used in GP (often the label).

        Returns:
            torch.Tensor: The calculated x'.
        """
        self._validate_input(x, z)

        # Get the weights and bias of the model
        assert isinstance(
            self.strategic_model, LinearStrategicModel
        ), "The model should be a LinearStrategicModel, but got {}".format(
            type(self.strategic_model)
        )
        weights, bias = self.strategic_model.get_weight_and_bias()
        x_prime: Optional[torch.Tensor] = None

        for x_sample, z_sample in zip(x, z):
            x_sample = x_sample.view(1, -1)
            projection: torch.Tensor = self._calculate_projection(
                x_sample, weights, bias, z_sample
            )

            assert (
                torch.sign(self.strategic_model(projection)) == z_sample
            ), "The projection is wrong, {}".format(self.strategic_model(projection))

            worth_to_move: bool = self._worth_to_move(x_sample, projection)

            if (torch.sign(self.strategic_model(x_sample)) == z_sample) or (
                not worth_to_move
            ):
                what_to_add = x_sample
            else:
                what_to_add = projection

            if x_prime is None:
                x_prime = what_to_add
            else:
                x_prime = torch.cat((x_prime, what_to_add))

        assert x_prime is not None, "The x_prime is None after the loop"
        return x_prime

    def get_z(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Abstract method that returns the meta data for the GP.
        Should be implemented by the child class.

        Args:
            x (torch.Tensor): The input data.
            y (torch.Tensor): The true labels.

        Raises:
            NotImplementedError: This method should be implemented by the child class.

        Returns:
            torch.Tensor: The meta data for the GP.
        """
        raise NotImplementedError("This is an abstract method")

    def _get_minimal_distance(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Calculates the minimal cost for the strategic agent to obtain a positive outcome.

        Args:
            x (torch.Tensor): The input data.
            z (torch.Tensor): The meta data for the GP.

        Returns:
            torch.Tensor: The minimal cost for each data sample.
        """
        self._validate_input(x, z)
        # Get the weights and bias of the model
        assert isinstance(
            self.strategic_model, LinearStrategicModel
        ), "The model should be a LinearStrategicModel, but got {}".format(
            type(self.strategic_model)
        )
        weights, bias = self.strategic_model.get_weight_and_bias()
        costs: Optional[torch.Tensor] = None

        for x_sample, z_sample in zip(x, z):
            x_sample: torch.Tensor = x_sample.view(1, -1)
            if torch.sign(self.strategic_model(x_sample)) == z_sample:
                costs_of_moment = torch.tensor([0])
            else:
                projection: torch.Tensor = self._calculate_projection(
                    x_sample, weights, bias, z_sample
                )

                assert (
                    torch.sign(self.strategic_model(projection)) == z_sample
                ), "The projection is wrong, {}".format(
                    self.strategic_model(projection)
                )

                costs_of_moment: torch.Tensor = self.cost(x_sample, projection)
            if costs is None:
                costs = costs_of_moment
            else:
                costs = torch.cat((costs, costs_of_moment))

        assert costs is not None, "The costs is None after the loop"
        return costs

    def _assert_cost(self) -> None:
        """
        Asserts that the cost function is either CostNormL2 or CostWeightedLoss.
        """
        assert isinstance(self.cost, CostNormL2) or isinstance(
            self.cost, CostWeightedLoss
        ), "The cost should be a  CostNormL2 or CostWeightedLoss"

    def _validate_input(self, x: torch.Tensor, z: torch.Tensor) -> None:
        """
        Validates the input of the linear GP.

        Args:
            x (torch.Tensor): The input data.
            z (torch.Tensor): The meta data for the GP.

        Raises:
            AssertionError: If the input or the dimensions are invalid.
        """
        # Check the input
        self._assert_cost()
        self._assert_model()
        batch_size: int = x.size(0)
        assert z.size() == torch.Size(
            [batch_size, 1]
        ), "z should be of size [batch_size, 1], but got {}".format(z.size())

    def _assert_model(self) -> None:
        """
        Asserts that the strategic model is a LinearStrategicModel.
        """
        assert isinstance(
            self.strategic_model, LinearStrategicModel
        ), "The strategic model should be a StrategicModel"

    def _calculate_projection(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        z: torch.Tensor = torch.tensor([1]),
        norm_waits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculates the projection of the data onto the model.

        Args:
            x (torch.Tensor): The input data.
            w (torch.Tensor): The model weights.
            b (torch.Tensor): The model bias.
            z (torch.Tensor, optional): The target value for projection. Defaults to 1.
            norm_waits (Optional[torch.Tensor]): The weight matrix for weighted norms. Defaults to None.

        Returns:
            torch.Tensor: The projection onto the model.
        """

        assert z.size() == torch.Size(
            [1]
        ), "z should be of size [1], but got {}".format(z.size())

        if norm_waits is None:
            norm_w = torch.matmul(w, w.T)
            projection = x - ((torch.matmul(x, w.T) + b) / (torch.matmul(w, w.T))) * w

        else:
            inverse_norm_waits: torch.Tensor = torch.inverse(norm_waits)
            norm_w = torch.matmul(w, torch.matmul(inverse_norm_waits, w.T))
            projection = x - ((torch.matmul(x, w.T) + b) / norm_w) * torch.matmul(
                inverse_norm_waits, w.T
            )

        unit_w = w / norm_w
        projection_with_safety = projection + z * self.epsilon * unit_w
        return projection_with_safety

    def _worth_to_move(self, x: torch.Tensor, projection: torch.Tensor) -> bool:
        """
        Determines whether it is worth moving the data point to the projection.

        Args:
            x (torch.Tensor): The input data.
            projection (torch.Tensor): The projected data.

        Returns:
            bool: True if it is worth moving the data point, False otherwise.
        """
        cost_of_moment = self.cost_weight * self.cost(x, projection)
        return bool(cost_of_moment < 2)
