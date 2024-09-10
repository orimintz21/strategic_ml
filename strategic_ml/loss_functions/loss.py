"""
This module contains the abstract class _Loss, which serves as the base class for all loss functions in the strategic_ml package.

When creating a new loss function, you should inherit from this class and implement the forward method.
"""

# External imports
import torch
import torch.nn as nn


class _Loss(nn.Module):
    """
    The _Loss class is an abstract base class for all loss functions in the strategic_ml package.

    All loss functions should inherit from this class and implement the forward method.
    The class handles a regularization parameter, which can be adjusted, and stores a reference 
    to the strategic model used in the loss calculation.

    Parent Class:
        nn.Module
    """
    def __init__(
        self,
        model: nn.Module,
        regularization_lambda: float = 0.01,
    ) -> None:
        """
        Initializes the base loss class.

        Args:
            model (nn.Module): The strategic model to be used for loss computation.
            regularization_lambda (float, optional): Regularization parameter to control model complexity. Defaults to 0.01.
        """
        super(_Loss, self).__init__()
        self.model: nn.Module = model
        self.regularization_lambda: float = regularization_lambda

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to compute the forward pass and calculate the loss. Must be implemented by subclasses.

        Args:
            x (torch.Tensor): Input features.
            y (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Loss value.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError()

    @property
    def get_regularization_lambda(self) -> float:
        """
        Getter for the regularization lambda.

        Returns:
            float: The regularization lambda value.
        """
        return self.regularization_lambda

    def set_regularization_lambda(self, value: float) -> None:
        """
        Setter for the regularization lambda.

        Args:
            value (float): The regularization lambda to set.
        """
        self.regularization_lambda = value

    @property
    def get_model(self) -> nn.Module:
        """
        Getter for the strategic model.

        Returns:
            nn.Module: The strategic model used for loss computation.
        """
        return self.model

    def set_model(self, value: nn.Module) -> None:
        """
        Setter for the strategic model.

        Args:
            value (nn.Module): The strategic model to set.
        """
        self.model = value
