# External imports
import torch
from torch import nn
from typing import Any, Tuple, Optional

# Internal imports


class LinearModel(nn.Module):
    """
    Implements a linear binary classification model. This model calculates the relevant
    strategic delta and regularization terms before modifying the input data and passing
    it to the model. The linear nature of this model allows for closed-form calculations
    of the strategic delta.

    Attributes:
        model (torch.nn.Linear): The linear layer for binary classification.
    """

    def __init__(
        self,
        in_features: int,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        dtype: Any = torch.float32,
    ) -> None:
        """
        Initializes the LinearModel class.

        Args:
            in_features (int): Number of input features.
            weight (Optional[torch.Tensor]): Initial weights for the model. Defaults to None.
            bias (Optional[torch.Tensor]): Initial bias for the model. Defaults to None.
            dtype (Any): Data type for the model. Defaults to torch.float32.
        """
        super(LinearModel, self).__init__()
        self.model: torch.nn.Linear = torch.nn.Linear(
            in_features=in_features, out_features=1, bias=True, dtype=dtype
        )
        if weight is not None and bias is not None:
            self.set_weight_and_bias(weight, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the linear model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output of the linear model.
        """
        assert (
            x.device == self.model.weight.device
        ), f"Input tensor is on a different device than the model. {x.device} != {self.model.weight.device}"
        return self.model(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calls the forward method.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output of the linear model.
        """
        return self.forward(x)

    def train(self, mode: bool = True) -> "LinearModel":
        """
        Sets the model in training mode.

        Args:
            mode (bool, optional): Whether to set training mode (True) or evaluation mode (False). Defaults to True.

        Returns:
            LinearModel: The model itself in the specified mode.
        """
        return super().train(mode)

    def get_weight_and_bias(
        self, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the weights and bias of the model.

        Args:
            device (Optional[torch.device]): Device to move the tensors to. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The weights and bias of the model.
        """
        weight = self.model.weight.detach()
        bias = self.model.bias.detach()
        if device is not None:
            weight = weight.to(device)
            bias = bias.to(device)

        return weight, bias

    def set_weight_and_bias(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        """
        Sets the weights and bias of the model.

        Args:
            weight (torch.Tensor): The new weights for the model.
            bias (torch.Tensor): The new bias for the model.
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
            weight.dtype == self.model.weight.dtype
        ), "The weight dtype should be the same"
        assert (
            bias.shape[0] == 1
        ), "This is a binary classification model, the number of outputs should be 1 instead of {}".format(
            bias.shape[0]
        )
        assert bias.dtype == self.model.bias.dtype, "The bias dtype should be the same"
        # Move weight and bias to the model's device
        device = self.model.weight.device
        weight = weight.to(device)
        bias = bias.to(device)

        # Set the weights and bias
        with torch.no_grad():
            self.model.weight.copy_(weight)
            self.model.bias.copy_(bias)

    def get_weight_and_bias_ref(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns references to the weights and bias of the model.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: References to the weights and bias of the model.
        """
        return self.model.weight, self.model.bias

    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the model's parameters are located.

        Returns:
            torch.device: The device of the model's parameters.
        """
        return next(self.parameters()).device
