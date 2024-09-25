# External imports
import torch
from typing import Optional, Union, List, Tuple

# Internal imports
from strategic_ml.cost_functions.cost_function import _CostFunction


class CostNormL2(_CostFunction):
    """
    Implements the L2 norm cost function.

    Calculates the Euclidean distance between the original data `x` and the 
    modified data `x_prime`, representing the cost of moving from `x` to `x_prime`.
    """

    def __init__(self, dim: Optional[Union[int, List[int], Tuple[int]]] = None) -> None:
        """
        Initializes the L2 norm cost function.

        Args:
            dim (Optional[Union[int, List[int], Tuple[int]]]): Dimensions over which the L2 norm is calculated. 
                If None, the norm is calculated over all dimensions.
        """
        super(CostNormL2, self).__init__(dim=dim)

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """
        Calculates the L2 norm cost function.

        Args:
            x (torch.Tensor): The original data point.
            x_prime (torch.Tensor): The modified data point.

        Returns:
            torch.Tensor: The L2 cost of moving from `x` to `x_prime`.
        """
        assert x.size() == x_prime.size(), f"{x.size()} != {x_prime.size()}"
        assert x.device == x_prime.device, "Input tensors are on different devices."
        diff = x - x_prime

        return torch.linalg.norm(diff, dim=self.dim, ord=2)


class CostMeanSquaredError(_CostFunction):
    """
    Implements the mean squared error (MSE) cost function.

    Calculates the average squared difference between the original data `x`
    and the modified data `x_prime`, representing the cost of moving from `x` 
    to `x_prime`.
    """
        
    def __init__(self, dim: Optional[Union[int, List[int], Tuple[int]]] = None) -> None:
        """
        Initializes the mean squared error cost function.

        Args:
            dim (Optional[Union[int, List[int], Tuple[int]]]): Dimensions over which the MSE is calculated. 
                If None, the error is averaged over all dimensions.
        """
        super(CostMeanSquaredError, self).__init__(dim=dim)

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """
        Calculates the mean squared error cost function.

        Args:
            x (torch.Tensor): The original data point.
            x_prime (torch.Tensor): The modified data point.

        Returns:
            torch.Tensor: The mean squared error cost of moving from `x` to `x_prime`.
        """
        assert x.size() == x_prime.size()
        assert x.device == x_prime.device, "Input tensors are on different devices."
        if self.dim is None:
            return torch.mean((x - x_prime) ** 2)

        return torch.mean((x - x_prime) ** 2, dim=self.dim)


class CostWeightedLoss(_CostFunction):
    """
    Implements a weighted loss cost function based on the mean squared error (MSE).

    Applies weights to each feature in the calculation of the squared difference 
    between the original data `x` and the modified data `x_prime`.
    """
    def __init__(
        self,
        weights: torch.Tensor,
        dim: Optional[Union[int, List[int], Tuple[int]]] = None,
    ) -> None:
        """
        Initializes the weighted loss cost function.

        Args:
            weights (torch.Tensor): The weights to apply to the loss.
            dim (Optional[Union[int, List[int], Tuple[int]]] Optional): Dimensions over which the loss is calculated. 
                If None, the loss is calculated over all dimensions.
        """
        super(CostWeightedLoss, self).__init__(dim=dim)
        self.register_buffer("weights", weights)

    def set_weights(self, weights: torch.Tensor) -> None:
        """
        Sets the weights for the weighted loss cost function.

        Args:
            weights (torch.Tensor): The weights to apply to the loss.
        """
        self.register_buffer("weights", weights)

    @property
    def get_weights(self) -> torch.Tensor:
        """
        Returns the weights used in the loss calculation.

        Returns:
            torch.Tensor: The weights applied to the loss.
        """
        return self.weights

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """
        Calculates the weighted loss cost function.

        Args:
            x (torch.Tensor): The original data point
            x_prime (torch.Tensor): The modified data point

        Returns:
            torch.Tensor: The weighted loss of moving from `x` to `x_prime`, shape [batch_size].
        """
        assert x.size() == x_prime.size(), "Input tensors must have the same shape."
        assert x.device == x_prime.device, "Input tensors are on different devices."
        if self.weights.device != x.device:
            self.weights = self.weights.to(x.device)
        distance = x - x_prime  # Shape: [batch_size, features] or [features]
        squared_distance = distance**2  # Element-wise square
        weighted_squared_distance = (
            squared_distance * self.weights
        )  # Element-wise multiplication
        if distance.dim() == 1:
            # Sum over all features
            cost = torch.sqrt(weighted_squared_distance.sum())
        else:
            # Sum over the specified dimension
            cost = torch.sqrt(weighted_squared_distance.sum(dim=self.dim))
        return cost


class CostNormL1(_CostFunction):
    """
    Implements the LInf norm cost function.

    Calculates the Chebyshev distance (LInf norm) between the original data `x`
    and the modified data `x_prime`, representing the cost of moving from `x` 
    to `x_prime`.
    """
    def __init__(self, dim: Optional[Union[int, List[int], Tuple[int]]] = None) -> None:
        """
        Initializes the LInf norm cost function.

        Args:
            dim (Optional[Union[int, List[int], Tuple[int]]] Optional): Dimensions over which the LInf norm is calculated. 
                If None, the norm is calculated over all dimensions.
        """
        super(CostNormL1, self).__init__(dim=dim)

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """
        Calculates the LInf norm cost function.

        Args:
            x (torch.Tensor): The original data point.
            x_prime (torch.Tensor): The modified data point.

        Returns:
            torch.Tensor: The LInf cost of moving from `x` to `x_prime`.
        """
        assert x.size() == x_prime.size()
        assert x.device == x_prime.device, "Input tensors are on different devices."
        return torch.linalg.norm(x - x_prime, ord=1, dim=self.dim)


class CostNormLInf(_CostFunction):
    def __init__(self, dim: Optional[Union[int, List[int], Tuple[int]]] = None) -> None:
        """
        Constructor for the LInf norm cost function.
        """
        super(CostNormLInf, self).__init__(dim=dim)

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """Calculates the LInf norm cost function.

        Args:
            x (torch.Tensor): the original data
            x_prime (torch.Tensor): the modified data

        Returns:
            torch.Tensor: the LInf cost of moving from x to x_prime
        """
        assert x.size() == x_prime.size()
        assert x.device == x_prime.device, "Input tensors are on different devices."
        return torch.linalg.norm(x - x_prime, ord=float("inf"), dim=self.dim)
