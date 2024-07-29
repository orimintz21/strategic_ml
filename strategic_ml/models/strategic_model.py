"""
:file
This file contains the basic structure for strategic model.
When creating a new model, you must inherit from this model to use the features
in the library.

This is an interface, meaning that you are not intended to construct this class
without implementing the relevant functions.

The purpose of the strategic model is to be an wrapper for the model, so when
calling the forward method, the model will also take into account the strategic
delta, the cost function and the strategic regularization.
"""

# External imports
import torch
from torch import nn

# Internal imports
from strategic_ml.gsc.generalized_strategic_delta import _GSC


class _StrategicModel(nn.Module):
    def __init__(
        self,
        delta: _GSC,
        model: nn.Module,
    ) -> None:
        """
        Constructor for the StrategicModel class.
        """
        super(_StrategicModel, self).__init__()
        self.delta: _GSC = delta
        self.model: nn.Module = model

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """This is the forward method of the StrategicModel class.
        The strategic model is a wrapper for the model, so when calling the
        forward method, the model will find x' using the delta, and then
        calculate the output of the model using x'.

        Args:
            x (torch.Tensor): the input data
            *args: additional arguments
            **kwargs: additional keyword arguments

        Returns:
            torch.Tensor: the output of the model
        """
        assert self.delta is not None and self.model is not None

        x_prime: torch.Tensor = self.delta.forward(x, *args, **kwargs)
        out: torch.Tensor = self.model(x_prime)
        return out

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """This is the call method of the StrategicModel class.
        The call method is a wrapper for the forward method.

        Args:
            x (torch.Tensor): the input data
            *args: additional arguments
            **kwargs: additional keyword arguments

        Returns:
            torch.Tensor: the output of the model
        """
        return self.forward(x, *args, **kwargs)

    def forward_no_delta(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """This is the forward method for the underlying model.
        In this function we do not apply the delta.

        Args:
            x (torch.Tensor):the input data
            *args: additional arguments
            **kwargs: additional keyword arguments

        Returns:
            torch.Tensor: the output of the model
        """
        return self.model(x)

    def get_delta(self) -> _GSC:
        """Getter for the delta.

        Returns:
            _GSC: the delta
        """
        return self.delta

    def set_delta(self, delta: _GSC) -> None:
        """Setter for the delta.

        Args:
            delta (_GSC): the delta to set
        """
        self.delta

    def get_underlying_model(self) -> nn.Module:
        """Getter for the model.

        Returns:
            nn.Module: the model
        """
        return self.model

    def set_underlying_model(self, model: nn.Module) -> None:
        """Setter for the model.

        Args:
            model (nn.Module): the model to set
        """
        self.model = model
