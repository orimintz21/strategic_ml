"""
This module implements the SocialBurden class for strategic regularization.

The Social Burden regularization method aims to minimize the effort that strategic agents labeled as positive 
must exert to achieve a positive outcome from the model. The method operates in settings where linear strategic 
deltas are used with convex cost functions. This regularization method was introduced in the paper 
"Strategic Classification Made Practical".

Parent Class:
    _StrategicRegularization
"""

# external imports
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

# internal imports
from strategic_ml.gsc.linear_gp import LinearStrategicDelta
from strategic_ml.regularization.strategic_regularization import (
    _StrategicRegularization,
)


class SocialBurden(_StrategicRegularization):
    """
    The SocialBurden class implements the Social Burden regularization method.

    Social Burden is a strategic regularization method that aims to minimize the amount of 
    effort required by strategic agents labeled as positive to achieve a positive outcome from the model. 
    The method works with linear strategic deltas and convex cost functions.

    Parent Class:
        _StrategicRegularization
    """

    def __init__(
        self,
        linear_delta: Optional[LinearStrategicDelta] = None,
    ) -> None:
        """
        Initializes the SocialBurden class.

        Args:
            linear_delta (Optional[LinearStrategicDelta], optional): The linear strategic delta used for regularization. 
                                                                     If not provided during initialization, it must be passed 
                                                                     in the forward method. Defaults to None.
        """
        super(SocialBurden, self).__init__()

        self.linear_delta = linear_delta

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        linear_delta: Optional[LinearStrategicDelta] = None,
    ) -> torch.Tensor:
        """
        Calculates the Social Burden regularization term.

        The Social Burden regularization term is computed as the sum of the distances that strategic agents labeled as positive 
        must traverse in feature space to achieve a positive outcome from the model. The distance is calculated using the 
        strategic delta.

        Formula:
        Social Burden = sum_{xi in X if yi == 1}(distance(x))

        Args:
            x (torch.Tensor): The input data to the model.
            y (torch.Tensor): The true labels corresponding to the input data.
            linear_delta (Optional[LinearStrategicDelta], optional): The linear strategic delta used for the calculation. 
                                                                     If not provided, the one passed during initialization is used. 
                                                                     Defaults to None.

        Returns:
            torch.Tensor: The calculated Social Burden regularization term.
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
