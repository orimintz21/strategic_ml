""" This module contains the abstract class _Loss, which is the base class for all loss functions in the strategic_ml package.
When creating a new loss function, you should inherit from this class and implement the forward method.
"""

import torch
import torch.nn as nn
from strategic_ml.gsc.generalized_strategic_delta import GSC
from typing import Optional


class _Loss(nn.Module):
    def __init__(
        self, model, delta: GSC, regularization_lambda: float = 0.01
    ):  # TODO Add the SC base model, and add typying
        """
        Initialize the base loss class.

        :param delta: Function to modify features strategically.
        :param regularization_lambda: Regularization parameter.
        :param model: Initial model parameters.
        """
        raise NotImplementedError()

    def compute_loss(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute the loss for given inputs and labels.

        :param X: Input features.
        :param y: True labels.
        :return: Loss value.
        """

        raise NotImplementedError()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass.

        :param X: Input features.
        :return: Predicted scores.
        """
        if self.w is None:
            raise ValueError("Model is not initialized. Please set the model weights.")
        return np.dot(X, self.w)

    def compute_gradient(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the loss function.

        :param X: Input features.
        :param y: True labels.
        :return: Gradient value.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    @property
    def delta(self) -> GSC:
        return self._delta

    @delta.setter
    def delta(self, value: GSC) -> None:
        self._delta = value

    @property
    def regularization_lambda(self) -> float:
        return self._regularization_lambda

    @regularization_lambda.setter
    def regularization_lambda(self, value: float) -> None:
        self._regularization_lambda = value

    @property
    def model(self) -> Optional[torch.Tensor]:
        return self._w

    @model.setter
    def model(self, value: Optional[torch.Tensor]) -> None:
        self._w = value
