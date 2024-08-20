""" file: strategic_hinge_loss.py
This module implements the Strategic Hinge Loss (s-hinge), a modified version 
of the standard hinge loss function designed to account for strategic behavior 
in classification settings. The s-hinge loss anticipates and incorporates the 
strategic modifications that agents might apply to their features to achieve 
better classification outcomes.

It maintains a differentiable form, allowing for optimization.

See more: "Generalized Strategic Classification and the Case of Aligned Incentives" Article
"""

# External imports
import torch
from torch import nn

# Internal imports
from strategic_ml.loss_functions import _Loss
from strategic_ml.gsc import _LinearGP
from strategic_ml.models import LinearStrategicModel


class StrategicHingeLoss(_Loss):
    def __init__(
        self,
        model: LinearStrategicModel,
        delta: _LinearGP,
        regularization_lambda: float = 0.01,
    ) -> None:
        """
        Initialize the Strategic Hinge Loss class.

        :param model: The strategic model.
        :param regularization_lambda: Regularization parameter.
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
        Forward pass to compute the strategic hinge loss.

        :param x: Input features as a torch.tensor.
        :param y: True labels as a torch.tensor.
        :return: Computed loss as a torch.tensor.
        """
        assert isinstance(
            self.model, LinearStrategicModel
        ), f"model should be an instance of LinearStrategicModel, but it is {type(self.model)}"
        z = self.delta.get_z(x, y)
        assert (
            z.shape[0] == x.shape[0]
        ), f"z should have the same number of samples as x, but z has {z.shape[0]} samples and x has {x.shape[0]} samples"
        w, b = self.model.get_weights_and_bias_ref()
        cost_weight = self.delta.get_cost_weight()
        linear_output = torch.matmul(x, w.T) + b

        norm = torch.linalg.norm(w, ord=2) + torch.linalg.norm(b, ord=2)

        additional_term = 2 * cost_weight * z * y * norm

        loss = torch.clamp(1 - y * linear_output - additional_term, min=0)

        return loss.mean()
