""":file: generalized_strategic_delta.py
This is the interface for all of GSC models.
When creating a new model in this framework you will need to inherit from
it.
A GSC - stands for generalized strategic is based on the paper:
'Generalized Strategic Classification and the Case of Aligned Incentives'

The general idea of the GSC is that a GSC gets a model, and change the input
x to x' based on the GSC's type.
"""

# External imports
import torch
from torch import nn
from typing import Any, Optional

# Internal imports
from strategic_ml.cost_functions import _CostFunction


# Implementation
class _GSC(nn.Module):
    """
    Abstract base class for Generalized Strategic Classification (GSC) models.

    This class defines the interface for GSC models that modify input data `x`
    to a new value `x'` based on strategic considerations. Subclasses should
    implement specific strategies for altering the input data.

    Attributes:
        strategic_model (nn.Module): The model used to calculate the strategic delta.
        cost (_CostFunction): The cost function applied in the strategic setting.
        cost_weight (float): The weight of the cost in the strategic calculation.
    """
    def __init__(
        self,
        strategic_model: nn.Module,
        cost: _CostFunction,
        cost_weight: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes the _GSC class.

        Args:
            strategic_model (nn.Module): The model for strategic classification.
            cost (_CostFunction): The cost function applied in the GSC.
            cost_weight (float): The weight of the cost function in the GSC.
            delta_model (Optional[nn.Module]): The model used to calculate the delta, if applicable.
        """
        super(_GSC, self).__init__()

        self.strategic_model: nn.Module = strategic_model
        self.cost: _CostFunction = cost
        self.cost_weight: float = cost_weight

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        The forward method for applying the strategic modifications.

        Args:
            x (torch.Tensor): The input data.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: Must be implemented in subclasses.

        Returns:
            torch.Tensor: The strategically modified data `x'`.
        """
        raise NotImplementedError()

    def find_x_prime(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Finds the modified input `x'` based on the strategic model and cost function.

        Args:
            x (torch.Tensor): The input data.
            z (torch.Tensor): The label associated with the input.

        Raises:
            NotImplementedError: Must be implemented in subclasses.

        Returns:
            torch.Tensor: The strategically modified data `x'`.
        """
        raise NotImplementedError()

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Calls the forward method to apply the strategic modifications.

        Args:
            x (torch.Tensor): The input data.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The strategically modified data `x'`.
        """
        return self.forward(x, *args, **kwargs)

    def train_delta_model(self, x: torch.Tensor, *args, **kwargs) -> None:
        """
        Trains the delta model, if applicable, as part of the strategic model training.
        Override this method in subclasses to implement delta model training.

        Args:
            x (torch.Tensor): The input data.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        """
        pass

    def get_strategic_model(self) -> nn.Module:
        """
        Returns the strategic model.

        Returns:
            nn.Module: The strategic model.
        """
        return self.strategic_model

    def set_strategic_model(self, model: nn.Module) -> None:
        """
        Sets the strategic model.

        Args:
            model (nn.Module): The strategic model to be set.
        """
        self.strategic_model = model

    def get_cost(self) -> _CostFunction:
        """
        Returns the cost function.

        Returns:
            _CostFunction: The cost function used in the GSC.
        """
        return self.cost

    def set_cost(self, cost: _CostFunction) -> None:
        """
        Sets the cost function.

        Args:
            cost (_CostFunction): The cost function to be set.
        """
        self.cost = cost

    def get_cost_weight(self) -> float:
        """
        Returns the cost weight.

        Returns:
            float: The weight of the cost function in the GSC.
        """
        return self.cost_weight

    def set_cost_weight(self, cost_weight: float) -> None:
        """
        Sets the cost weight.

        Args:
            cost_weight (float): The weight of the cost function in the GSC.
        """
        self.cost_weight = cost_weight
