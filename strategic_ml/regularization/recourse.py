"""
This module implements the Recourse class for strategic regularization.

The Recourse class calculates a soft version of the recourse term for strategic agents. 
Recourse refers to the capacity of a user who is denied a service to restore approval through 
low-cost feature modification. The recourse term is calculated using a sigmoid approximation.

The formula for the recourse term is:
    Recourse = sum_{x in X}(sigmoid(-f(x)*temp)*sigmoid(-f(delta(x))*temp))

For more details, see the paper:
- "Strategic Classification Made Practical"

Parent Class:
    _StrategicRegularization
"""

# External Imports
import torch
from torch import nn
from typing import Optional

# Internal Imports
from strategic_ml.regularization.strategic_regularization import (
    _StrategicRegularization,
)


class Recourse(_StrategicRegularization):
    """
    The Recourse class calculates the recourse term for strategic agents.

    The recourse term refers to the ability of users who are denied a service to restore approval 
    through low-cost feature modification. It uses a sigmoid approximation to compute a soft 
    version of the term:
    Recourse = sum_{x in X}(sigmoid(-f(x)*temp)*sigmoid(-f(delta(x))*temp))

    Parent Class:
        _StrategicRegularization
    """

    def __init__(
        self, sigmoid_temp: float = 1.0, model: Optional[nn.Module] = None
    ) -> None:
        """
        Initializes the Recourse class.

        Args:
            sigmoid_temp (float, optional): The temperature for the sigmoid functions. Defaults to 1.0.
            model (Optional[nn.Module], optional): The model used for recourse calculation. If not provided during initialization, 
                                                   it must be passed in the forward method.
        """
        super(Recourse, self).__init__()

        self.sigmoid_temp = sigmoid_temp
        if model is not None:
            self.model = model

    def forward(
        self,
        x: torch.Tensor,
        delta_predictions: torch.Tensor,
        model: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculates the recourse term for strategic agents.

        The recourse term is calculated as:
        Recourse = sum_{x in X}(sigmoid(-f(x)*temp)*sigmoid(-f(delta(x))*temp))

        Args:
            x (torch.Tensor): The input tensor (features).
            delta_predictions (torch.Tensor): The predictions of the model on the modified (delta) inputs.
            model (Optional[nn.Module], optional): The model used for recourse calculation. If not provided, 
                                                   the model passed during initialization is used.

        Returns:
            torch.Tensor: The computed recourse term for the batch.
        """
        assert (
            x.shape[0] == delta_predictions.shape[0]
        ), "x and delta_predictions must have the same batch size"

        if model is not None:
            x_predictions: torch.Tensor = model(x)
        else:
            assert hasattr(
                self, "model"
            ), "model must be passed to the forward method or the Recourse object must have a model"
            x_predictions = self.model(x)

        x_neg_predictions_sig: torch.Tensor = torch.sigmoid(
            -x_predictions * self.sigmoid_temp
        )
        delta_neg_predictions_sig: torch.Tensor = torch.sigmoid(
            -delta_predictions * self.sigmoid_temp
        )
        product: torch.Tensor = x_neg_predictions_sig * delta_neg_predictions_sig
        recourse: torch.Tensor = torch.sum(product)
        return recourse
