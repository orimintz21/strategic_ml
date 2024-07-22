""" file: strategic_hinge_loss.py
This module implements the Strategic Hinge Loss (s-hinge), a modified version 
of the standard hinge loss function designed to account for strategic behavior 
in classification settings. The s-hinge loss anticipates and incorporates the 
strategic modifications that agents might apply to their features to achieve 
better classification outcomes.

It maintains a differentiable form, allowing for optimization.

See more: "Generalized Strategic Classification and the Case of Aligned Incentives" Article
"""

import torch
from torch import nn
from strategic_ml.gsc.generalized_strategic_delta import _GSC
from loss import _Loss


class StrategicHingeLoss(_Loss):
    def __init__(self, model: nn.Module, regularization_lambda: float = 0.01):
        """
        Initialize the Strategic Hinge Loss class.

        :param model: The strategic model.
        :param regularization_lambda: Regularization parameter.
        """
        super(StrategicHingeLoss, self).__init__(model, regularization_lambda)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the strategic hinge loss.

        :param x: Input features as a torch.tensor.
        :param y: True labels as a torch.tensor.
        :return: Computed loss as a torch.tensor.
        """
        raise NotImplementedError()

    def compute_loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the strategic hinge loss for the given inputs and labels.

        :param X: Input features.
        :param y: True labels.
        :return: Computed loss as a float.
        """
        raise NotImplementedError()

    def compute_gradient(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the strategic hinge loss.

        :param X: Input features.
        :param y: True labels.
        :return: Computed gradient as a NumPy array.
        """
        raise NotImplementedError()
