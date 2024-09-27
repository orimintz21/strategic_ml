# External Imports
import torch
from torch import nn
from strategic_ml.gsc import _GSC
from typing import Any, Optional

# Internal Imports
from strategic_ml.cost_functions import _CostFunction


class IdentityDelta(_GSC):
    """
    Implements a No-Operation (Identity) Delta for GSC models.

    This class provides an identity transformation, meaning the input data `x`
    is returned unchanged. It serves as a baseline or a no-op scenario where
    no strategic modifications are applied.
    """

    def __init__(self, cost: Optional[_CostFunction], strategic_model: nn.Module):
        """
        Initializes the IdentityDelta class.

        Args:
            cost (_CostFunction): An optional cost function, not used in this implementation.
            strategic_model (nn.Module): The strategic model, included for interface consistency.
        """
        if cost is None:
            cost = _CostFunction()
        super(IdentityDelta, self).__init__(cost=cost, strategic_model=strategic_model)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Returns the input data `x` unchanged.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The same input data `x`.
        """
        return x

    def get_z(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Returns a zero tensor as the strategic modification.

        Args:
            x (torch.Tensor): The input data.
            y (torch.Tensor): The associated labels.

        Returns:
            torch.Tensor: A zero tensor of the same shape as `x`.
        """
        return torch.zeros_like(x)

    def get_minimal_distance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a zero tensor as the minimal distance since no modification is applied.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: A zero tensor of the same shape as `x`.
        """
        return torch.zeros_like(x)
