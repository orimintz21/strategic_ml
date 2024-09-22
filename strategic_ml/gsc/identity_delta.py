# External Imports
import torch
from torch import nn
from strategic_ml.gsc import _GSC
from typing import Any, Optional
# Internal Imports
from strategic_ml.cost_functions import _CostFunction



class IdentityDelta(_GSC):
    """
    A No Operation (Identity) Delta class that implements the _GSC interface.
    This class returns the input as it is, without applying any strategic
    manipulations or modifications.
    """

    def __init__(self, cost: Optional[_CostFunction], strategic_model: nn.Module):
        """
        Initialize the IdentityDelta class.

        Args:
            cost (nn.Module): Optional cost function, not used in this implementation.
            strategic_model (nn.Module): the model, it is not used but is kept to keep similar interface.
        """
        if cost is None:
            cost = _CostFunction()
        super(IdentityDelta, self).__init__(cost=cost, strategic_model=strategic_model)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward method that returns the input as it is.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The same input tensor without any modifications.
        """
        return x

    def compute_deltas(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method to compute the delta, which is zero in this case.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: A zero tensor of the same shape as x.
        """
        return torch.zeros_like(x)

    def get_z(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Return a zero tensor for the strategic modification.

        Args:
            x (torch.Tensor): The input tensor.
            y (torch.Tensor): The labels.

        Returns:
            torch.Tensor: A zero tensor of the same shape as x.
        """
        return torch.zeros_like(x)

    def get_minimal_distance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return a zero tensor for the minimal distance, since there is no modification.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: A zero tensor of the same shape as x.
        """
        return torch.zeros_like(x)
