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
    Social Burden 

    Parent Class: _StrategicRegularization
    """
    def __init__(
        self,
        linear_delta: Optional[LinearStrategicDelta] = None,
    ) -> None:
        """
        Constructor for the SocialBurden class.
        """
        super(SocialBurden, self).__init__()

        self.linear_delta = linear_delta

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        linear_delta: Optional[LinearStrategicDelta] = None,
    ) -> torch.Tensor:
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
