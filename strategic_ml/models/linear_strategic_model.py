"""
This module implements the LinearStrategicModel for binary classification in strategic settings.

The LinearStrategicModel calculates the strategic delta and regularization to modify the input data 
before passing it to the model. The model is linear, and the delta can be computed in closed form, 
allowing for efficient computation without requiring gradient-based optimization techniques.
"""

# External imports
import torch
from torch import nn
from typing import Any, Tuple, Optional

# Internal imports


class LinearStrategicModel(nn.Module):
    """
    The LinearStrategicModel class defines a binary classification model used in strategic settings.

    The model calculates strategic modifications to the input data using a closed-form calculation 
    for the strategic delta and regularization. This model is specifically designed for binary classification 
    tasks and uses a linear model structure.

    Parent Class:
        nn.Module
    """

    def __init__(
        self,
        in_features: int,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initializes the LinearStrategicModel class.

        Args:
            in_features (int): Number of input features for the model.
            weight (Optional[torch.Tensor], optional): Predefined weight tensor. If not provided, defaults to None.
            bias (Optional[torch.Tensor], optional): Predefined bias tensor. If not provided, defaults to None.
        """
        super(LinearStrategicModel, self).__init__()
        self.model: torch.nn.Linear = torch.nn.Linear(
            in_features=in_features, out_features=1, bias=True
        )
        if weight is not None and bias is not None:
            self.set_weight_and_bias(weight, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor of the model after applying the linear transformation.
        """
        return self.model(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calls the forward method of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor of the model.
        """
        return self.forward(x)

    def train(self, mode: bool = True) -> "LinearStrategicModel":
        """
        Sets the model to training mode.

        Args:
            mode (bool, optional): Whether to set the model to training mode. Defaults to True.

        Returns:
            LinearStrategicModel: The model in training mode.
        """
        return super().train(mode)

    def get_weight_and_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the weights and bias of the model.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The weights and bias of the model.
        """
        return self.model.weight.detach(), self.model.bias.detach()

    def set_weight_and_bias(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        """
        Sets the weight and bias of the model.

        Args:
            weight (torch.Tensor): The new weights to set.
            bias (torch.Tensor): The new bias to set.
        """
        # Check the input
        assert (
            weight.shape[1] == self.model.weight.shape[1]
        ), "The number of features should be the same, the input was {}".format(
            weight.shape[1]
        )
        assert (
            weight.shape[0] == 1
        ), "This is a binary classification model, the number of outputs should be 1 instead of {}".format(
            weight.shape[0]
        )
        assert (
            bias.shape[0] == 1
        ), "This is a binary classification model, the number of outputs should be 1 instead of {}".format(
            bias.shape[0]
        )
        # Set the weights and bias
        with torch.no_grad():
            self.model.weight.copy_(weight)
            self.model.bias.copy_(bias)

    def get_weight_and_bias_ref(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the weights and bias of the model by reference.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The weights and bias of the model by reference.
        """
        return self.model.weight, self.model.bias
