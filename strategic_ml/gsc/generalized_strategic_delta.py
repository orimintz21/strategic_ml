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
        strategic_model: Optional[_StrategicModel] = None,
        cost: Optional[_CostFunction] = None,
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
            strategic_model (_StrategicModel): the model for the GSC.
            cost (_CostFunction): the cost function for the GSC.
            delta_model (Optional[nn.Module]): the delta model for the GSC
            this is the model that will be used to calculate the delta.
            Not all of the GSC models will use a delta model.
        """
        super(_GSC, self).__init__()
        if strategic_model is not None:
            self.strategic_model: _StrategicModel = strategic_model

        if cost is not None:
            self.cost: _CostFunction = cost

        if delta_model is not None:
            self.delta_model: nn.Module = delta_model

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

    def train_delta_model(self, x: torch.Tensor, *args, **kwargs) -> None:
        """This is the method that will train the delta model. It is
        part of the training of the strategic model. Some models will

        Args:
            x (torch.Tensor): The data
            *args: additional arguments
            **kwargs: additional keyword arguments
        """
        raise NotImplementedError()

    def get_strategic_model(self) -> _StrategicModel:
        """Getter for the strategic model.

        Returns:
            _StrategicModel: the model
        """
        return self.strategic_model

    def set_strategic_model(self, model: _StrategicModel) -> None:
        """Setter for the strategic model.

        Args:
            model (_StrategicModel): the model
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

    def set_arguments_for_training(self, *args, **kwargs) -> None:
        """This method will set the arguments for the training of the delta model.
        For example, optimizer, learning rate, etc.

        Args:
            *args: additional arguments
            **kwargs: additional keyword arguments
        """
        raise NotImplementedError()
