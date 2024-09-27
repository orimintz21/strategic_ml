# External imports
import torch
from torch import nn
from typing import Optional, Union, List, Tuple


class _CostFunction(nn.Module):
    """
    Abstract base class for cost functions in the strategic_ml package.

    This class serves as a base for implementing various cost functions that
    calculate the cost of transitioning from an original data point `x` to a
    modified data point `x_prime`. Subclasses should implement the `forward`
    method to define specific cost calculations.

    Attributes:
        dim (Optional[Union[int, List[int], Tuple[int]]]): Dimensions over which the cost is calculated.
    """

    def __init__(self, dim: Optional[Union[int, List[int], Tuple[int]]] = None) -> None:
        """This module contains the abstract class CostFunction, which is the base class for all cost functions in the strategic_ml package.
        When creating a new cost function, you should inherit from this class and implement the forward method.
        """
        super(_CostFunction, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """
        Calculates the cost of transitioning from `x` to `x_prime`.

        Args:
            x (torch.Tensor): The original data point.
            x_prime (torch.Tensor): The modified data point.

        Raises:
            NotImplementedError: Must be implemented in subclasses.

        Returns:
            torch.Tensor: The cost of moving from `x` to `x_prime`.
        """
        raise NotImplementedError()

    def __call__(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """
        Calls the forward method to calculate the cost.

        Args:
            x (torch.Tensor): The original data point.
            x_prime (torch.Tensor): The modified data point.

        Returns:
            torch.Tensor: The cost of moving from `x` to `x_prime`.
        """
        return self.forward(x, x_prime)

    def get_dim(self) -> Optional[Union[int, List[int], Tuple[int]]]:
        """
        Returns the dimensions over which the cost function operates.

        Returns:
            Optional[Union[int, List[int], Tuple[int]]]: The dimension(s) of the cost function.
        """
        return self.dim

    def set_dim(self, dim: Optional[Union[int, List[int], Tuple[int]]]) -> None:
        """
        Sets the dimensions over which the cost function operates.

        Args:
            dim (Optional[Union[int, List[int], Tuple[int]]]): The dimension(s) to set.
        """
        self.dim = dim
