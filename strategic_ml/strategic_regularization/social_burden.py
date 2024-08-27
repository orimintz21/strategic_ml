# external imports
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

# internal imports
from strategic_ml.gsc.linear_gp import LinearStrategicDelta
from strategic_ml.strategic_regularization.strategic_regularization import (
    _StrategicRegularization,
)


class SocialBurden(_StrategicRegularization):
    """
    SocialBurden class.
    This is the implementation of the Social Burden regularization method that
    is described in the paper "Strategic Classification Made Practical".
    Social Burden is a strategic regularization method that tries to minimize
    the amount of effort that the strategic agents that are labeled as positive
    need to do in order to get a positive outcome from the model.
    Note that the Social Burden regularization method only works with linear
    strategic deltas and a convex cost function.

    Parent Class: _StrategicRegularization
    """

    def __init__(
        self,
        linear_delta: Optional[LinearStrategicDelta] = None,
    ) -> None:
        """
        Constructor for the SocialBurden class.
        :param linear_delta: The linear strategic delta that the Social Burden, if not provided
        it should be provided in the forward method.
        """
        super(SocialBurden, self).__init__()

        self.linear_delta = linear_delta

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        linear_delta: Optional[LinearStrategicDelta] = None,
    ) -> torch.Tensor:
        """This is the forward method of the SocialBurden class.
        This function calculates the Social Burden regularization term.
        The Social Burden regularization term is calculated by the following formula:
        Social Burden = sum_{xi in X if yi == 1}(distance(x)) where distance(x) is the minimal cost
        that the strategic agent needs to do in order to get a positive outcome from the model.

        Args:
            x (torch.Tensor): The input of the model.
            y (torch.Tensor): The true labels of the model.
            linear_delta (Optional[LinearStrategicDelta], optional): The strategic delta
            if None is provided the forward method will use the delta
            that it got at initialization. Defaults to None.

        Returns:
            torch.Tensor: The Social Burden regularization term.
        """
        assert (
            x.shape[0] == y.shape[0]
        ), "x, x_prime, and y must have the same batch size"
        assert y.shape[1] == 1, "y must be a 1D tensor"

        positive_label = y == 1
        positive_label = positive_label.squeeze()
        x_positive = x[positive_label]

        if linear_delta is not None:
            distance = linear_delta.get_minimal_distance(x_positive)
        else:
            assert self.linear_delta is not None, "linear_delta must be provided"
            distance = self.linear_delta.get_minimal_distance(x_positive)

        total_reg = torch.sum(distance)

        return total_reg
