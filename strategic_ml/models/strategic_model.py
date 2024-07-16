"""
:file
This file contains the basic structure for strategic model.
When creating a new model, you must inherit from this model to use the features
in the library.

This is an interface, meaning that you are not intended to construct this class
without implementing the relevant functions.

The purpose of the strategic model is to be an wrapper for the model, so when
calling the forward method, the model will also take into account the strategic
delta, the cost function and the strategic regularization.
"""

# External imports
import torch
from torch import nn
from typing import Optional

# Internal imports
from strategic_ml.gsc.generalized_strategic_delta import _GSC
from strategic_ml.cost_functions.cost_function import _CostFunction
from strategic_ml.strategic_regularization import _StrategicRegularization


class _StrategicModel(nn.Module):
    def __init__(
        self,
        delta: Optional[_GSC] = None,
        cost: Optional[_CostFunction] = None,
        s_reg: Optional[_StrategicRegularization] = None,
    ) -> None:
        """
        Constructor for the StrategicModel class.
        """
        super(_StrategicModel, self).__init__()
        if delta is not None:
            self.delta = delta
        if cost is not None:
            self.cost = cost
        if s_reg is not None:
            self.s_reg = s_reg

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """This is the forward method of the StrategicModel class.
        The strategic model is a wrapper for the model, so when calling the forward
        method, the model will also take into account the strategic delta, the cost
        function and the strategic regularization.

        Args:
            x (torch.Tensor): the input data

        Raises:
            NotImplementedError: this is an interface, you should implement this method in your subclass

        Returns:
            torch.Tensor: the output of the model
        """

        raise NotImplementedError()

    @property
    def delta(self) -> _GSC:
        """Getter for the delta.

        Returns:
            _GSC: the delta
        """
        return self.delta

    @delta.setter
    def delta(self, delta: _GSC) -> None:
        """Setter for the delta.

        Args:
            delta (_GSC): the delta to set
        """
        self.delta

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
            cost (_CostFunction): the cost function to set
        """
        self.cost

    @property
    def s_reg(self) -> _StrategicRegularization:
        """Getter for the strategic regularization.

        Returns:
            _StrategicRegularization: the strategic regularization
        """
        return self.s_reg

    @s_reg.setter
    def s_reg(self, s_reg: _StrategicRegularization) -> None:
        """Setter for the strategic regularization.

        Args:
            s_reg (_StrategicRegularization): the strategic regularization to set
        """
        self.s_reg
