"""
This module contains the _StrategicRegularization class, which serves as the base class for all
strategic regularization methods in the strategic_ml package.

Strategic regularization differs from traditional regularization methods like L1 and L2. It introduces 
a regularization term to the loss function that encourages the model to account for the effect of the model 
on strategic agents. This concept is described in the paper "Strategic Classification Made Practical."

Parent Class:
    nn.Module
"""

import torch
from torch import nn


class _StrategicRegularization(nn.Module):
    """
    The _StrategicRegularization class serves as the base class for all strategic regularization methods.

    Strategic regularization adds a term to the loss function to encourage the model to account for 
    the behavior of strategic agents when making predictions. This base class should be inherited 
    and the `forward` method should be implemented by subclasses.

    Parent Class:
        nn.Module
    """
    def __init__(self) -> None:
        """
        Initializes the _StrategicRegularization class.
        """
        super(_StrategicRegularization, self).__init__()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass for strategic regularization.

        This method must be implemented by subclasses to define the specific regularization behavior.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.

        Returns:
            torch.Tensor: A 1x1 tensor representing the regularization term.
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        Calls the forward method of the regularization.

        Args:
            *args: Additional arguments for the regularization method.
            **kwargs: Additional keyword arguments for the regularization method.

        Returns:
            torch.Tensor: A tensor representing the regularization term.
        """
        return self.forward(*args, **kwargs)
