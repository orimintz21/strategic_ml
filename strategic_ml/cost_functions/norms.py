""":file: cost_functions/norms.py
In this file we implement basic norm cost functions.
All cost functions inherit from the CostFunction class so they can be used
in the strategic_ml package.
"""

import torch

from strategic_ml.cost_functions.cost_function import _CostFunction


class CostNormL2(_CostFunction):
    def __init__(self) -> None:
        """
        Constructor for the L2 norm cost function.
        """
        super(CostNormL2, self).__init__()

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """Calculates the L2 norm cost function.

        Args:
            x (torch.Tensor): the original data
            x_prime (torch.Tensor): modified data

        Returns:
            torch.Tensor: the L2 cost of moving from x to x_prime
        """
        assert x.size() == x_prime.size(), f"{x.size()} != {x_prime.size()}"
        return torch.linalg.norm(x - x_prime, ord=2)


class CostMeanSquaredError(_CostFunction):
    def __init__(self) -> None:
        """
        Constructor for the mean squared error cost function.
        """
        super(_CostFunction, self).__init__()

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """Calculates the mean squared error cost function.

        Args:
            x (torch.Tensor): the original data
            x_prime (torch.Tensor): modified data

        Returns:
            torch.Tensor: the mean squared error cost of moving from x to x_prime
        """
        assert x.size() == x_prime.size()
        return torch.mean((x - x_prime) ** 2)


class CostWeightedLoss(_CostFunction):
    def __init__(self, weights: torch.Tensor) -> None:
        """Constructor for the weighted loss cost function.
        Based on the mean squared error cost function.

        Args:
            weights (torch.Tensor): the weights to apply to the loss
        """
        super(_CostFunction, self).__init__()
        self.weights: torch.Tensor = weights

    def set_weights(self, weights: torch.Tensor) -> None:
        """Sets the weights for the weighted loss cost function.

        Args:
            weights (torch.Tensor): the weights to apply to the loss
        """
        self.weights = weights

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
        return torch.sum(self.weights * (x - x_prime) ** 2)


class CostNormL1(_CostFunction):
    def __init__(self) -> None:
        """
        Constructor for the L1 norm cost function.
        """
        super(_CostFunction, self).__init__()

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """Calculates the L1 norm cost function.

        Args:
            x (torch.Tensor): the original data
            x_prime (torch.Tensor): the modified data

        Returns:
            torch.Tensor: _description_
        """
        assert x.size() == x_prime.size()
        return torch.linalg.norm(x - x_prime, ord=1)


class CostNormLInf(_CostFunction):
    def __init__(self) -> None:
        """
        Constructor for the LInf norm cost function.
        """
        super(_CostFunction, self).__init__()

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """Calculates the LInf norm cost function.

        Args:
            x (torch.Tensor): the original data
            x_prime (torch.Tensor): the modified data

        Returns:
            torch.Tensor: the LInf cost of moving from x to x_prime
        """
        assert x.size() == x_prime.size()
        return torch.linalg.norm(x - x_prime, ord=float("inf"))
