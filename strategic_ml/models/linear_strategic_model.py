""" linear_strategic_model.py
This is the linear strategic model.
The linear strategic model calculates the relent delta and the strategic regularization
and uses them to modify the input data before passing it to the model.

We implement the LinearStrategicModel class because when we use a linear model
we can calculate the strategic delta in a closed form.
"""

# External imports
import torch
from torch import nn
from typing import Any, Tuple, Optional

# Internal imports


class LinearStrategicModel(nn.Module):
    def __init__(
        self,
        in_features: int,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Constructor for the LinearStrategicModel class.
        This is a binary classification model therefore the output features is 1.
        """
        super(LinearStrategicModel, self).__init__()
        self.model: torch.nn.Linear = torch.nn.Linear(
            in_features=in_features, out_features=1, bias=True
        )
        if weight is not None and bias is not None:
            self.set_weights_and_bias(weight, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def train(self, mode: bool = True) -> "LinearStrategicModel":
        return super().train(mode)

    def get_weights_and_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The get_weights_and_bias method returns the weights and bias of the model

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the weights and bias of the model
        """
        return self.model.weight.detach(), self.model.bias.detach()

    def set_weights_and_bias(self, weighs: torch.Tensor, bias: torch.Tensor) -> None:
        """
        The set_weights_and_bias method sets the weights and bias of the model

        Args:
            weighs (torch.Tensor): the new weighs
            bias (torch.Tensor): the new bias
        """
        # Check the input
        assert (
            weighs.shape[1] == self.model.weight.shape[1]
        ), "The number of features should be the same, the input was {}".format(
            weighs.shape[1]
        )
        assert (
            weighs.shape[0] == 1
        ), "This is a binary classification model, the number of outputs should be 1 instead of {}".format(
            weighs.shape[0]
        )
        assert (
            bias.shape[0] == 1
        ), "This is a binary classification model, the number of outputs should be 1 instead of {}".format(
            bias.shape[0]
        )
        # Set the weights and bias
        with torch.no_grad():
            self.model.weight.copy_(weighs)
            self.model.bias.copy_(bias)

    def get_weights_and_bias_ref(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The get_weights_and_bias method returns the weights and bias of the model

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the weights and bias of the model
        """
        return self.model.weight, self.model.bias
