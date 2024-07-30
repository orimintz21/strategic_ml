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


class StrategicHingeLoss(_Loss):
    def __init__(self, model: nn.Module, regularization_lambda: float = 0.01) -> None:
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
