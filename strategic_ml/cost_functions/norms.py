# External imports
import torch
from typing import Optional, Union, List, Tuple

# Internal imports
from strategic_ml.cost_functions.cost_function import _CostFunction


class CostNormL2(_CostFunction):
    """
    In this file we implement basic norm cost functions.
    All cost functions inherit from the CostFunction class so they can be used
    in the strategic_ml package.

    Parent class: _CostFunction
    """

    def __init__(self, dim: Optional[Union[int, List[int], Tuple[int]]] = None) -> None:
        """
        Constructor for the L2 norm cost function.
        """
        super(CostNormL2, self).__init__(dim=dim)

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """Calculates the L2 norm cost function.

        Args:
            x (torch.Tensor): the original data
            x_prime (torch.Tensor): modified data

        Returns:
            torch.Tensor: the L2 cost of moving from x to x_prime
        """
        assert x.size() == x_prime.size(), f"{x.size()} != {x_prime.size()}"
        assert x.device == x_prime.device, "Input tensors are on different devices."
        diff = x - x_prime

        return torch.linalg.norm(diff, dim=self.dim, ord=2)


class CostMeanSquaredError(_CostFunction):
    def __init__(self, dim: Optional[Union[int, List[int], Tuple[int]]] = None) -> None:
        """
        Constructor for the mean squared error cost function.
        """
        super(CostMeanSquaredError, self).__init__(dim=dim)

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """Calculates the mean squared error cost function.

        Args:
            x (torch.Tensor): the original data
            x_prime (torch.Tensor): modified data

        Returns:
            torch.Tensor: the mean squared error cost of moving from x to x_prime
        """
        assert x.size() == x_prime.size()
        assert x.device == x_prime.device, "Input tensors are on different devices."
        if self.dim is None:
            return torch.mean((x - x_prime) ** 2)

        return torch.mean((x - x_prime) ** 2, dim=self.dim)


class CostWeightedLoss(_CostFunction):
    def __init__(
        self,
        weights: torch.Tensor,
        dim: Optional[Union[int, List[int], Tuple[int]]] = None,
    ) -> None:
        """Constructor for the weighted loss cost function.
        Based on the mean squared error cost function.

        Args:
            weights (torch.Tensor): the weights to apply to the loss
        """
        super(CostWeightedLoss, self).__init__(dim=dim)
        self.register_buffer("weights", weights)

    def set_weights(self, weights: torch.Tensor) -> None:
        """Sets the weights for the weighted loss cost function.

        Args:
            weights (torch.Tensor): the weights to apply to the loss
        """
        self.register_buffer("weights", weights)

    @property
    def get_weights(self) -> torch.Tensor:
        """Gets the weights for the weighted loss cost function.

        Returns:
            torch.Tensor: the weights to apply to the loss
        """
        return self.weights

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """Calculates the weighted loss cost function.

        Args:
            x (torch.Tensor): the original data
            x_prime (torch.Tensor): modified data

        Returns:
            torch.Tensor: the weighted loss of moving from x to x_prime
        """
        assert x.size() == x_prime.size()
        assert x.device == x_prime.device, "Input tensors are on different devices."
        if self.weights.device != x.device:
            self.weights = self.weights.to(x.device)
        distance = x - x_prime
        if self.dim == 1:
            return torch.sqrt(torch.einsum("ij,ij->", distance, self.weights))

        return torch.sqrt(distance @ self.weights @ distance.T)


class CostNormL1(_CostFunction):
    def __init__(self, dim: Optional[Union[int, List[int], Tuple[int]]] = None) -> None:
        """
        Constructor for the L1 norm cost function.
        """
        super(CostNormL1, self).__init__(dim=dim)

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """Calculates the L1 norm cost function.

        Args:
            x (torch.Tensor): the original data
            x_prime (torch.Tensor): the modified data

        Returns:
            torch.Tensor: _description_
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
