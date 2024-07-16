""" This module contains the abstract class CostFunction, which is the base class for all cost functions in the strategic_ml package.
When creating a new cost function, you should inherit from this class and implement the forward method.
"""

import torch
from torch import nn


class _CostFunction(nn.Module):
    def __init__(self) -> None:
        """Constructor for the _CostFunction class."""
        super(_CostFunction, self).__init__()

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """Calculates the cost function.

        Args:
            x (torch.Tensor): the original data
            x_prime (torch.Tensor): modified data

        Raises:
            NotImplementedError: you should implement this method in your subclass

        Returns:
            torch.Tensor: the cost of moving from x to x_prime
        """
        raise NotImplementedError()

    def __call__(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """Calls the forward method.

        Args:
            x (torch.Tensor): the original data
            x_prime (torch.Tensor):  modified data

        Returns:
            torch.Tensor: the cost of moving from x to x_prime
        """
        return self.forward(x, x_prime)
