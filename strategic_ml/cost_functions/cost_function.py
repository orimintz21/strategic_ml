"""
This module contains the abstract class _CostFunction, which is the base class for all cost functions in the strategic_ml package.
When creating a new cost function, you should inherit from this class and implement the forward method.
"""

import torch
from torch import nn
from typing import Optional, Union, List, Tuple


class _CostFunction(nn.Module):
    def __init__(self, dim: Optional[Union[int, List[int], Tuple[int]]] = None) -> None:
        """
        Initializes the _CostFunction class.

        Args:
            dim (Optional[Union[int, List[int], Tuple[int]]]): Dimension(s) for the cost function.
        """
        super(_CostFunction, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """
        Calculates the cost function. This method should be implemented by subclasses.

        Args:
            x (torch.Tensor): The original data.
            x_prime (torch.Tensor): The modified data.

        Returns:
            torch.Tensor: The cost of moving from x to x_prime.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError()

    def __call__(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """
        Calls the forward method to calculate the cost.

        Args:
            x (torch.Tensor): The original data.
            x_prime (torch.Tensor): The modified data.

        Returns:
            torch.Tensor: The cost of moving from x to x_prime.
        """
        return self.forward(x, x_prime)

    def get_dim(self) -> Optional[Union[int, List[int], Tuple[int]]]:
        """
        Returns the dimension(s) of the cost function.

        Returns:
            Optional[Union[int, List[int], Tuple[int]]]: The dimension(s) of the cost function.
        """
        return self.dim

    def set_dim(self, dim: Optional[Union[int, List[int], Tuple[int]]]) -> None:
        """
        Sets the dimension(s) of the cost function.

        Args:
            dim (Optional[Union[int, List[int], Tuple[int]]]): The dimension(s) of the cost function.
        """
        self.dim = dim
