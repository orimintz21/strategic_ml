"""
This module implements basic norm cost functions.
All cost functions inherit from the _CostFunction class, so they can be used
in the strategic_ml package.
"""

import torch
from typing import Optional, Union, List, Tuple

from strategic_ml.cost_functions.cost_function import _CostFunction


class CostNormL2(_CostFunction):
    def __init__(self, dim: Optional[Union[int, List[int], Tuple[int]]] = None) -> None:
        """
        Initializes the L2 norm cost function.

        Args:
            dim (Optional[Union[int, List[int], Tuple[int]]]): Dimension(s) along which to calculate the norm.
        """
        super(CostNormL2, self).__init__(dim=dim)

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """
        Calculates the L2 norm cost function.

        Args:
            x (torch.Tensor): The original data.
            x_prime (torch.Tensor): The modified data.

        Returns:
            torch.Tensor: The L2 cost of moving from x to x_prime.
        """
        assert x.size() == x_prime.size(), f"{x.size()} != {x_prime.size()}"
        return torch.linalg.norm(x - x_prime, dim=self.dim, ord=2)


class CostMeanSquaredError(_CostFunction):
    def __init__(self, dim: Optional[Union[int, List[int], Tuple[int]]] = None) -> None:
        """
        Initializes the mean squared error cost function.

        Args:
            dim (Optional[Union[int, List[int], Tuple[int]]]): Dimension(s) along which to calculate the mean squared error.
        """
        super(CostMeanSquaredError, self).__init__(dim=dim)

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """
        Calculates the mean squared error cost function.

        Args:
            x (torch.Tensor): The original data.
            x_prime (torch.Tensor): The modified data.

        Returns:
            torch.Tensor: The mean squared error cost of moving from x to x_prime.
        """
        assert x.size() == x_prime.size()
        if self.dim is None:
            return torch.mean((x - x_prime) ** 2)

        return torch.mean((x - x_prime) ** 2, dim=self.dim)


class CostWeightedLoss(_CostFunction):
    def __init__(
        self,
        weights: torch.Tensor,
        dim: Optional[Union[int, List[int], Tuple[int]]] = None,
    ) -> None:
        """
        Initializes the weighted loss cost function based on the mean squared error.

        Args:
            weights (torch.Tensor): The weights to apply to the loss.
            dim (Optional[Union[int, List[int], Tuple[int]]]): Dimension(s) along which to calculate the weighted loss.
        """
        super(CostWeightedLoss, self).__init__(dim=dim)
        self.weights: torch.Tensor = weights

    def set_weights(self, weights: torch.Tensor) -> None:
        """
        Sets the weights for the weighted loss cost function.

        Args:
            weights (torch.Tensor): The weights to apply to the loss.
        """
        self.weights = weights

    @property
    def get_weights(self) -> torch.Tensor:
        """
        Gets the weights for the weighted loss cost function.

        Returns:
            torch.Tensor: the weights to apply to the loss
        """
        return self.weights

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """
        Calculates the weighted loss cost function.

        Args:
            x (torch.Tensor): the original data
            x_prime (torch.Tensor): modified data

        Returns:
            torch.Tensor: the weighted loss of moving from x to x_prime
        """
        assert x.size() == x_prime.size()
        distance = x - x_prime
        if self.dim == 1:
            return torch.sqrt(torch.einsum("ij,ij->", distance, self.weights))

        return torch.sqrt(distance @ self.weights @ distance.T)


class CostNormL1(_CostFunction):
    def __init__(self, dim: Optional[Union[int, List[int], Tuple[int]]] = None) -> None:
        """
        Initializes the L1 norm cost function.

        Args:
            dim (Optional[Union[int, List[int], Tuple[int]]]): Dimension(s) along which to calculate the norm.
        """
        super(CostNormL1, self).__init__(dim=dim)

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """
        Calculates the L1 norm cost function.

        Args:
            x (torch.Tensor): The original data.
            x_prime (torch.Tensor): The modified data.

        Returns:
            torch.Tensor: The L1 cost of moving from x to x_prime.
        """
        assert x.size() == x_prime.size()
        return torch.linalg.norm(x - x_prime, ord=1, dim=self.dim)


class CostNormLInf(_CostFunction):
    def __init__(self, dim: Optional[Union[int, List[int], Tuple[int]]] = None) -> None:
        """
        Initializes the LInf norm cost function.

        Args:
            dim (Optional[Union[int, List[int], Tuple[int]]]): Dimension(s) along which to calculate the norm.
        """
        super(CostNormLInf, self).__init__(dim=dim)

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """
        Calculates the LInf norm cost function.

        Args:
            x (torch.Tensor): The original data.
            x_prime (torch.Tensor): The modified data.

        Returns:
            torch.Tensor: The LInf cost of moving from x to x_prime.
        """
        assert x.size() == x_prime.size()
        return torch.linalg.norm(x - x_prime, ord=float("inf"), dim=self.dim)
