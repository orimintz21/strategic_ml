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
    The GP is a strategic delta that is calculated by the following formula:
    delta_h(x,z) = argmax_{x' in X}(1{model(x') = z} - r/2 * (cost(x,x')))

    We can see that if model(x) == z then x is the best x' that we can choose.
    If for every x' in X, such that model(x') != z, we get cost(x,x') > 2/r, then
    the GP will be x (it is not worth to change x).

    If we assume that the model is linear and the cost is norm2, or weighted
    norm, we see that if model(x) != z and the margin from the model is smaller than
    the wait of the cost using the norm to calculate the margin, then the GP will be
    the projection of x on the model.

    Then the GP is calculated by the following formula:
    x_prime = x - ((w.T @ x + b) / (w.T @ w)) * w
    if we use the norm2 cost function.
    and
    x_prime = x - ((w.T @ x + b) / (w.T @ inverse(Lambda) @ w)) * inverse(Lambda) @ w
    if we use the weighted norm cost function with Lambda as the weight matrix.

    """

    def __init__(
        self,
        cost: _CostFunction,
        strategic_model: nn.Module,
        cost_weight: float = 1.0,
        epsilon: float = 0.01,
    ) -> None:
        """This is the LinearGP model. This model assumes that the model is linear.
        The reason for this assumption is that we can calculate the GP in a closed form
        for linear models.
        Therefore we do not need to train a delta model.

        Args:
            cost (_CostFunction): The cost function of the delta.
            strategic_model (nn.Module): the strategic that the delta is calculated on.
            cost_weight (int): The weight of the cost function.
            epsilon (float): move to the negative/positive direction of the model
            to make sure that the model will predict the label correctly. The
            delta does it by adding the (epsilon * w/||w||). Defaults to 0.01.
        """
        super(_LinearGP, self).__init__(
            strategic_model=strategic_model, cost=cost, cost_weight=cost_weight
        )
        self._assert_model()
        self._assert_cost()
        self.epsilon: float = epsilon

    def find_x_prime(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """This function calculates x' based on the GP formula.

        Args:
            x (torch.Tensor): The input of the model.
            z (torch.Tensor): Meta data for the GP.

        Raises:
            AssertionError: If the cost or the model is None.
            NotImplementedError: If the model is not a Linear model (This should not happen)

        Returns:
            torch.Tensor: x' the GP.
        """
        self._validate_input(x, z)

        # Get the weights and bias of the model
        assert isinstance(
            self.strategic_model, LinearStrategicModel
        ), "The model should be a LinearStrategicModel, but got {}".format(
            type(self.strategic_model)
        )
        weights, bias = self.strategic_model.get_weight_and_bias()

        # Ensure dtype consistency
        x = x.to(weights.dtype)
        z = z.to(weights.dtype)

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
        """This function returns the meta data for the GP.
        It should be implemented by the child class.

        Args:
            x (torch.Tensor): The input of the model.
            y (torch.Tensor): The true labels of the model.

        Raises:
            NotImplementedError: Should be implemented by the child class.

        Returns:
            torch.Tensor: The meta data for the GP.
        """
        raise NotImplementedError("This is an abstract method")

    def _get_minimal_distance(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """This function calculates the minimal cost that the strategic agent needs to do
        to get a positive outcome from the model.

        Args:
            x (torch.Tensor): The data.
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
                costs_of_moment = torch.tensor([0.0])
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

            costs_of_moment = costs_of_moment.view(
                -1
            )  # Ensure 1D tensor for concatenation

            if costs is None:
                costs = costs_of_moment
            else:
                costs = torch.cat((costs, costs_of_moment))

        if costs is None:
            # Return a zero tensor if no costs were calculated
            return torch.zeros(x.shape[0], dtype=torch.float64)

        # assert costs is not None, "The costs is None after the loop"
        return costs

    def _assert_cost(self) -> None:
        """
        This function asserts that the cost is a CostNormL2 or CostWeightedLoss
        """
        assert isinstance(self.cost, CostNormL2) or isinstance(
            self.cost, CostWeightedLoss
        ), "The cost should be a  CostNormL2 or CostWeightedLoss"

    def _validate_input(self, x: torch.Tensor, z: torch.Tensor) -> None:
        """This function validates the input of the linear gp.

        Args:
            x (torch.Tensor): The data
            z (torch.Tensor): The meta data for the GP.
        """
        # Check the input
        self._assert_cost()
        self._assert_model()
        batch_size: int = x.size(0)
        assert z.size() == torch.Size(
            [batch_size, 1]
        ), "z should be of size [batch_size, 1], but got {}".format(z.size())

        w, b = self.strategic_model.get_weight_and_bias()
        # Ensure all tensors have consistent dtype
        assert (x.dtype == b.dtype) , "All tensors should have the same dtype but we got x: {0}, b: {1}".format(x.dtype, b.dtype)
        assert (x.dtype  == w.dtype), "All tensors should have the same dtype but we got x: {0}, w: {2}".format(x.dtype, w.dtype)
        assert (x.dtype == z.dtype), "All tensors should have the same dtype but we got x: {0}, z: {3}".format(x.dtype, z.dtype)

    def _assert_model(self) -> None:
        """This function asserts that the strategic model is a LinearStrategicModel"""
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
        """This method calculates the projection on the model.

        Args:
            x (torch.Tensor): The data
            w (torch.Tensor): The weights of the model
            b (torch.Tensor): The bias of the model
            norm_waits (torch.Tensor): The weights of the cost function

        Returns:
            torch.Tensor: The projection on the model
        """

        assert z.size() == torch.Size(
            [1]
        ), "z should be of size [1], but got {}".format(z.size())

        if norm_waits is None:
            norm_w = torch.matmul(w, w.T)
            projection = x - ((torch.matmul(x, w.T) + b) / (norm_w)) * w

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
        """This function checks if it is worth to move the data point to the projection.
        It is worth to move the data point if the cost of the moment  times its
        weight is smaller than 2.

        Args:
            x (torch.Tensor): The data
            projection (torch.Tensor): The projection of the data on the model.

        Returns:
            bool: If it is worth to move the data point to the projection.
        """
        cost_of_moment = self.cost_weight * self.cost(x, projection)
        return bool(cost_of_moment < 2)
