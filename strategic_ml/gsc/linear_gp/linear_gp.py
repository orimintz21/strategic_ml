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
    Abstract base class for linear Generalized Strategic Classification (GSC) models.

    This class implements the strategic delta computation for linear models, assuming
    the cost function is either L2 or a weighted L2 norm. The delta is computed using
    a closed-form solution based on the projection of the input data onto the decision
    boundary of the model.

    The GP is calculated by the following formula:
    delta_h(x, z) = argmax_{x' in X}(1{model(x') = z} - r/2 * (cost(x, x')))

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
        Initializes the _LinearGP class.

        Args:
            cost (_CostFunction): The cost function used in the delta computation.
            strategic_model (nn.Module): The linear model used for the strategic classification.
            cost_weight (float, optional): The weight of the cost function. Defaults to 1.0.
            epsilon (float, optional): A small adjustment added to ensure correct model predictions. Defaults to 0.01.
        """
        super(_LinearGP, self).__init__(
            strategic_model=strategic_model, cost=cost, cost_weight=cost_weight
        )
        self._assert_model()
        self._assert_cost()
        self.epsilon: float = epsilon

    def find_x_prime(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the modified input `x'` based on the linear GP formula.

        Args:
            x (torch.Tensor): The input data.
            z (torch.Tensor): Metadata for the GP, typically the desired label.

        Returns:
            torch.Tensor: The strategically modified data `x'`.
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
        """
        Returns the metadata `z` for the GP, which should be implemented by subclasses.

        Args:
            x (torch.Tensor): The input data.
            y (torch.Tensor): The true labels.

        Returns:
            torch.Tensor: The metadata `z` for the GP.
        """
        raise NotImplementedError("This is an abstract method")

    def _get_minimal_distance(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Calculates the minimal cost required for the strategic agent to be classified as z.

        Args:
            x (torch.Tensor): The input data.
            z (torch.Tensor): Metadata for the GP.

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
        Asserts that the cost function is either L2 or a weighted L2 norm.
        """
        assert isinstance(self.cost, CostNormL2) or isinstance(
            self.cost, CostWeightedLoss
        ), "The cost should be a  CostNormL2 or CostWeightedLoss"

    def _validate_input(self, x: torch.Tensor, z: torch.Tensor) -> None:
        """
        Validates the input data and metadata for the linear GP.

        Args:
            x (torch.Tensor): The input data.
            z (torch.Tensor): Metadata for the GP.
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
        """
        Asserts that the strategic model is a LinearModel.
        """        
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
        """
        Calculates the projection of `x` onto the decision boundary of the linear model.

        Args:
            x (torch.Tensor): The input data.
            w (torch.Tensor): The weights of the linear model.
            b (torch.Tensor): The bias of the linear model.
            z (torch.Tensor): Metadata for the GP.
            norm_weights (Optional[torch.Tensor]): Weights for the weighted norm calculation, if applicable.

        Returns:
            torch.Tensor: The projection of `x` onto the decision boundary.
        """
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
