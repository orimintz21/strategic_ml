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
    Implements the Social Burden regularization method, as described in the paper
    "Strategic Classification Made Practical."

    Social Burden minimizes the effort required by strategic agents labeled as positive
    to achieve a favorable outcome. This method only works with linear strategic deltas
    and convex cost functions.

    The formula for the Social Burden term is:
    Social Burden = sum_{xi in X if yi == 1}(distance(x)) where distance(x) is the minimal cost
    that the strategic agent needs to do in order to get a positive outcome from the model.

    Attributes:
        linear_delta (Optional[LinearStrategicDelta]): The linear strategic delta that the Social Burden uses.
    """

    def __init__(
        self,
        linear_delta: Optional[LinearStrategicDelta] = None,
    ) -> None:
        """
        Initializes the SocialBurden class.

        Args:
            linear_delta (Optional[LinearStrategicDelta]): The linear strategic delta. If not provided, it should be provided in the forward method.
        """
        super(SocialBurden, self).__init__()

        self.linear_delta = linear_delta

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        linear_delta: Optional[LinearStrategicDelta] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes the Social Burden regularization term.

        The formula is:
        Social Burden = sum_{xi in X, yi == 1}(distance(xi)), where distance(x) is the minimal
        cost required by the strategic agent to achieve a positive outcome.

        Args:
            x (torch.Tensor): Input data.
            y (torch.Tensor): True labels.
            linear_delta (Optional[LinearStrategicDelta]): The linear strategic delta. If None, the delta provided at initialization is used.

        Returns:
            torch.Tensor: The Social Burden regularization term.
        """
        assert (
            x.shape[0] == y.shape[0]
        ), "x, x_prime, and y must have the same batch size"

        assert y.shape[1] == 1, "y must be a 1D tensor"
        y = y.to(device=x.device, dtype=x.dtype)

        positive_label = (y == 1).squeeze()
        x_positive = x[positive_label]
        if x_positive.numel() == 0:
            return torch.tensor(0.0).to(device=x.device, dtype=x.dtype)

        if linear_delta is not None:
            distance = linear_delta.get_minimal_distance(x_positive)
        else:
            assert self.linear_delta is not None, "linear_delta must be provided"
            distance = self.linear_delta.get_minimal_distance(x_positive)

        total_reg = torch.sum(distance)

        return total_reg
