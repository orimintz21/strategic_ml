# External imports
import torch
from torch import nn


class _StrategicRegularization(nn.Module):
    """
    Abstract base class for strategic regularization methods in the strategic_ml package.

    Strategic regularization refers to adding a term to the loss function that accounts
    for the effect of the model on strategic agents, rather than traditional regularization
    methods like L1 or L2. This concept is discussed in the paper "Strategic Classification
    Made Practical."

    This class should be inherited to create custom strategic regularization methods.
    """

    def __init__(self) -> None:
        """
        Initializes the _StrategicRegularization class.
        """
        super(_StrategicRegularization, self).__init__()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward method for computing the strategic regularization term.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.

        Returns:
            torch.Tensor: A scalar tensor representing the regularization term.
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        Calls the forward method to compute the strategic regularization term.

        Returns:
            torch.Tensor: A scalar tensor representing the regularization term.
        """
        return self.forward(*args, **kwargs)
