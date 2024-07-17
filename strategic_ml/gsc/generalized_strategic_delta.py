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
from typing import Optional

# Internal imports
from strategic_ml.models import _StrategicModel
from cost_functions import _CostFunction


# Implementation
class _GSC(nn.Module):
    def __init__(
        self,
        model: Optional[_StrategicModel] = None,
        cost: Optional[_CostFunction] = None,
    ) -> None:
        """Constructor for the _GSC class.
        When creating a new GSC model, you should inherit from this class.
        The model is the model that the GSC will use to calculate the strategic
        delta. Note that the model could be different from the main model
        (i.e. the one that holds this GSC).

        Args:
            model (_StrategicModel): the model for the GSC.
            cost (_CostFunction): the cost function for the GSC.
        """
        super(_GSC, self).__init__()
        if model is not None:
            self.model = model

        if cost is not None:
            self.cost = cost

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """The forward method of the GSC class.

        Args:
            x (torch.Tensor): the data
            *args: additional arguments
            **kwargs: additional keyword arguments

        Raises:
            NotImplementedError: This is an interface, you should implement this method in your subclass

        Returns:
            torch.Tensor: x' - the modified data
        """

        raise NotImplementedError()

    @property
    def model(self) -> _StrategicModel:
        """Getter for the model.

        Returns:
            _StrategicModel: the model
        """
        return self.model

    @model.setter
    def model(self, model: _StrategicModel) -> None:
        """Setter for the model.

        Args:
            model (_StrategicModel): the model
        """
        self.model = model

    @property
    def cost(self) -> _CostFunction:
        """Getter for the cost function.

        Returns:
            _CostFunction: the cost function
        """
        return self.cost

    @cost.setter
    def cost(self, cost: _CostFunction) -> None:
        """Setter for the cost function.

        Args:
            cost (_CostFunction): the cost function
        """
        self.cost = cost
