""" This module contains the abstract class _Loss, which is the base class for all loss functions in the strategic_ml package.
When creating a new loss function, you should inherit from this class and implement the forward method.
"""

import torch
import torch.nn as nn
from strategic_ml.gsc.generalized_strategic_delta import _GSC


class _Loss(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        regularization_lambda: float = 0.01,
    ) -> None:
        """
        Initialize the base loss class.

        :param model: The strategic model.
        :param regularization_lambda: Regularization parameter.
        """
        super(_Loss, self).__init__()
        self.model: nn.Module = model
        self.regularization_lambda = regularization_lambda

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass and calculate the loss.

        :param X: Input features.
        :param y: True labels.
        :return: Loss value.
        """
        raise NotImplementedError()

    @property
    def regularization_lambda(self) -> float:
        return self.regularization_lambda

    @regularization_lambda.setter
    def regularization_lambda(self, value: float) -> None:
        self.regularization_lambda = value

    @property
    def model(self) -> nn.Module:
        return self.model

    @model.setter
    def model(self, value: nn.Module) -> None:
        self.model = value
