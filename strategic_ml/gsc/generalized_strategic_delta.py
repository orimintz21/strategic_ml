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
from typing import Any, Optional, Dict, Tuple

# Internal imports
from strategic_ml.cost_functions import _CostFunction


# Implementation
class _GSC:
    def __init__(
        self,
        strategic_model: nn.Module,
        cost: _CostFunction,
        cost_weight: float = 1.0,
        delta_model: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ) -> None:
        """Constructor for the _GSC class.
        When creating a new GSC model, you should inherit from this class.
        The strategic_model is the model that the GSC will use to calculate the strategic
        delta. Note that the model could be different from the main model
        (i.e. the one that holds this GSC).
        The args and kwargs are for the delta model training.


        Args:
            strategic_model (nn.Module): the model for the GSC.
            cost (_CostFunction): the cost function for the GSC.
            delta_model [nn.Module]: the delta model for the GSC
            this is the model that will be used to calculate the delta.
            Not all of the GSC models will use a delta model.
        """
        super(_GSC, self).__init__()

        self.strategic_model: nn.Module = strategic_model
        self.cost: _CostFunction = cost
        self.cost_weight: float = cost_weight

        if delta_model is not None:
            self.delta_model: nn.Module = delta_model

    def forward(
        self, x: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """The forward method of the GSC class.

        Args:
            x (torch.Tensor): the data
            *args: additional arguments
            **kwargs: additional keyword arguments

        Raises:
            NotImplementedError: This is an interface, you should implement this method in your subclass

        Returns:
            torch.Tensor: x' - the modified data
            dict: additional information
        """
        raise NotImplementedError()

    def find_x_prime(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """The find_x_prime method of the GSC class.

        Args:
            x (torch.Tensor): The data
            z (torch.Tensor): Metadata

        Raises:
            NotImplementedError: This is an interface, you should implement this method in your subclass

        Returns:
            torch.Tensor: x_prime
            Dict[str, Any]: additional information
        """
        raise NotImplementedError()

    def __call__(
        self, x: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """The call method of the GSC class.

        Args:
            x (torch.Tensor): the data
            *args: additional arguments
            **kwargs: additional keyword arguments

        Returns:
            torch.Tensor: x' - the modified data
            Dict[str, Any]: additional information
        """
        return self.forward(x, *args, **kwargs)

    def get_strategic_model(self) -> nn.Module:
        """Getter for the strategic model.

        Returns:
            nn.Module: the model
        """
        return self.strategic_model

    def set_strategic_model(self, model: nn.Module) -> None:
        """Setter for the strategic model.

        Args:
            model (nn.Module): the model
        """
        self.strategic_model = model

    def get_cost(self) -> _CostFunction:
        """Getter for the cost function.

        Returns:
            _CostFunction: the cost function
        """
        return self.cost

    def set_cost(self, cost: _CostFunction) -> None:
        """Setter for the cost function.

        Args:
            cost (_CostFunction): the cost function
        """
        self.cost = cost

    def get_cost_weight(self) -> float:
        """Getter for the cost weight.

        Returns:
            float: the cost weight
        """
        return self.cost_weight

    def set_cost_weight(self, cost_weight: float) -> None:
        """Setter for the cost weight.

        Args:
            cost_weight (float): the cost weight
        """
        self.cost_weight = cost_weight
