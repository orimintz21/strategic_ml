# External imports
import torch
from torch import nn
from typing import Optional

# Internal imports
from strategic_ml.gsc.generalized_strategic_delta import _GSC
from strategic_ml.models import LinearModel
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
            self.strategic_model, LinearModel
        ), "The model should be a LinearModel."
        weights, bias = self.strategic_model.get_weight_and_bias()
        assert weights.device == x.device, "x and w should be on the same device"

        # Ensure dtype and device consistency
        dtype = weights.dtype
        x = x.to(dtype=dtype)
        z = z.to(dtype=dtype)

        # Calculate projections for all samples
        projections = self._calculate_projection(x, weights, bias, z)

        # Ensure that projections produce the correct predictions
        projection_outputs = self.strategic_model(projections)
        projection_signs = torch.sign(projection_outputs)
        assert (
            projection_signs == z
        ).all(), "Projections do not yield correct predictions."

        # Determine if it's worth moving to the projection
        cost_of_movement = (
            self.cost(x, projections) * self.cost_weight
        )  # Shape: [batch_size]
        worth_to_move = cost_of_movement < 2  # Shape: [batch_size]

        # Determine if the original predictions are already correct
        original_outputs = self.strategic_model(x)  # Shape: [batch_size, 1]
        original_signs = torch.sign(original_outputs)
        original_correct = (original_signs == z).squeeze(1)  # Shape: [batch_size]

        # Ensure both are BoolTensors
        worth_to_move = worth_to_move.bool()
        original_correct = original_correct.bool()

        # Compute the condition tensor
        condition = original_correct | (~worth_to_move)  # Shape: [batch_size]

        # Expand the condition tensor to match x's shape
        condition = condition.unsqueeze(1).expand(
            -1, x.size(1)
        )  # Shape: [batch_size, feature_dim]

        x_prime = torch.where(condition, x, projections)
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
        weights, bias = self.strategic_model.get_weight_and_bias()

        # Ensure dtype and device consistency
        assert x.device == weights.device, "x and w should be on the same device"
        x = x.to(weights.dtype)
        z = z.to(weights.dtype)

        # Calculate model outputs
        model_outputs = self.strategic_model(x)  # Shape: [batch_size, 1]
        original_correct = torch.sign(model_outputs).squeeze(1) == z.squeeze(
            1
        )  # Shape: [batch_size]

        # Calculate projections
        projections = self._calculate_projection(
            x, weights, bias, z
        )  # Shape: [batch_size, feature_dim]

        # Calculate costs
        cost_of_movement = self.cost(x, projections)

        # Set cost to zero where original predictions are correct
        costs = torch.where(
            original_correct, torch.zeros_like(cost_of_movement), cost_of_movement
        )

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

        assert x.device == z.device, "x and z should be on the same device"

        w, b = self.strategic_model.get_weight_and_bias()
        # Ensure all tensors have consistent dtype
        assert (
            x.dtype == b.dtype
        ), "All tensors should have the same dtype but we got x: {0}, b: {1}".format(
            x.dtype, b.dtype
        )
        assert (
            x.dtype == w.dtype
        ), "All tensors should have the same dtype but we got x: {0}, w: {2}".format(
            x.dtype, w.dtype
        )
        assert (
            x.dtype == z.dtype
        ), "All tensors should have the same dtype but we got x: {0}, z: {3}".format(
            x.dtype, z.dtype
        )
        assert x.device == w.device, "x and w should be on the same device"

    def _assert_model(self) -> None:
        """This function asserts that the strategic model is a LinearModel"""
        assert isinstance(
            self.strategic_model, LinearModel
        ), "The strategic model should be a StrategicModel"

    def _calculate_projection(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        z: torch.Tensor,
        norm_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Ensure z is of shape [batch_size, 1]
        if z.dim() == 1:
            z = z.unsqueeze(1)  # Shape: [batch_size, 1]

        if norm_weights is None:
            # Calculate norm_w as scalar
            norm_w = torch.norm(w, p=2) ** 2  # Scalar

            # Compute numerator and denominator separately
            numerator = x @ w.T + b  # Shape: [batch_size, 1]
            denominator = norm_w  # Scalar

            # Compute projection
            projection = x - (numerator / denominator) * w  # Broadcasting over batch
        else:
            # Handle weighted norm case
            inverse_norm_weights = torch.inverse(norm_weights).to(x.device)
            norm_w = w @ inverse_norm_weights @ w.T  # Scalar

            numerator = x @ w.T + b  # Shape: [batch_size, 1]
            denominator = norm_w  # Scalar

            projection = (
                x - (numerator / denominator) * (inverse_norm_weights @ w.T).T
            )  # Shape: [batch_size, feature_dim]

        # Add epsilon adjustment
        unit_w = w / torch.norm(w, p=2)  # Shape: [1, feature_dim]
        projection_with_safety = (
            projection + z * self.epsilon * unit_w
        )  # Broadcasting over batch

        return projection_with_safety
