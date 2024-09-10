"""
This module implements the Strategic Hinge Loss (s-hinge) function for strategic classification.

The Strategic Hinge Loss is a modified version of the standard hinge loss that accounts for 
strategic behavior in classification settings. It anticipates strategic modifications that 
agents might apply to their features to achieve better classification outcomes. 

The loss function assumes the following:
- The model is linear.
- The delta is a Linear Delta (from _LinearGP).
- The cost function is the L2 norm.

The loss function is defined as:
    L(x, z, y; w, b) = max(0, 1 - y(w^T x + b) - 2 * cost_weight * z * y * (||w||_2 + ||b||_2))

For more information, see the article:
- "Generalized Strategic Classification and the Case of Aligned Incentives"
"""

# External imports
import torch
from torch import nn

# Internal imports
from strategic_ml.loss_functions import _Loss
from strategic_ml.gsc import _LinearGP
from strategic_ml.models import LinearStrategicModel


class StrategicHingeLoss(_Loss):
    """
    The Strategic Hinge Loss (s-hinge) is a modified hinge loss function that incorporates 
    strategic behavior in classification. It assumes that the model is linear, the delta is 
    calculated using the _LinearGP framework, and the cost function is L2 norm-based.

    The s-hinge loss is defined as:
    L(x, z, y; w, b) = max(0, 1 - y(w^T x + b) - 2 * cost_weight * z * y * (||w||_2 + ||b||_2))

    This loss maintains differentiability, allowing for optimization.

    Parent Class:
        _Loss
    """

    def __init__(
        self,
        model: LinearStrategicModel,
        delta: _LinearGP,
        regularization_lambda: float = 0.01,
    ) -> None:
        """
        Initializes the Strategic Hinge Loss class.

        Args:
            model (LinearStrategicModel): The linear strategic model.
            delta (_LinearGP): The delta model used to compute the strategic modifications.
            regularization_lambda (float, optional): The regularization parameter to control model complexity. Defaults to 0.01.
        """
        super(StrategicHingeLoss, self).__init__(model, regularization_lambda)
        assert isinstance(
            model, LinearStrategicModel
        ), f"model should be an instance of LinearStrategicModel, but it is {type(model)}"
        assert isinstance(
            delta, _LinearGP
        ), f"delta should be an instance of linear gp , but it is {type(delta)}"
        self.delta = delta

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass to compute the strategic hinge loss.

        The delta is used to account for strategic modifications in the input features, and the 
        loss is calculated by considering the L2 norm of the model weights and bias.

        Args:
            x (torch.Tensor): Input features.
            y (torch.Tensor): True labels.

        Returns:
            torch.Tensor: The computed strategic hinge loss.
        """
        assert isinstance(
            self.model, LinearStrategicModel
        ), f"model should be an instance of LinearStrategicModel, but it is {type(self.model)}"
        z = self.delta.get_z(x, y)
        assert (
            z.shape[0] == x.shape[0]
        ), f"z should have the same number of samples as x, but z has {z.shape[0]} samples and x has {x.shape[0]} samples"
        w, b = self.model.get_weight_and_bias_ref()
        cost_weight = self.delta.get_cost_weight()

        linear_output = torch.matmul(x, w.T) + b

        norm = torch.linalg.norm(w, ord=2) + torch.linalg.norm(b, ord=2)

        additional_term = 2 * cost_weight * z * y * norm

        loss = torch.clamp(1 - y * linear_output - additional_term, min=0)

        return loss.mean()
