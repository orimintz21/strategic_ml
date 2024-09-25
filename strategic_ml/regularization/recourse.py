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
    Implements the Recourse regularization method, which quantifies the ability of strategic agents 
    to modify their features in order to obtain a favorable outcome. The recourse term is a soft 
    version of the sum of indicators for negative predictions before and after modification.

    The formula for the recourse term is:
    Recourse = sum_{x in X}(sigmoid(-f(x) * temp) * sigmoid(-f(delta(x)) * temp))

    Attributes:
        sigmoid_temp (float): The temperature for the sigmoids.
    """
    def __init__(
        self, sigmoid_temp: float = 1.0, model: Optional[nn.Module] = None
    ) -> None:
        """
        Initializes the Recourse class.

        Args:
            sigmoid_temp (float): Temperature for the sigmoid function. Defaults to 1.0.
            model (Optional[nn.Module]): The model to be used for predictions. If None, it must be provided in the forward method.
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
        Computes the recourse term for strategic agents.

        Args:
            x (torch.Tensor): Input data.
            delta_predictions (torch.Tensor): Predictions of the model after strategic modification.
            model (Optional[nn.Module]): The model to be used for predictions. If None, the model provided at initialization is used.

        Returns:
            torch.Tensor: The recourse term.
        """
        assert (
            x.shape[0] == delta_predictions.shape[0]
        ), "x and delta_predictions must have the same batch size"
        assert (
            delta_predictions.device == x.device
        ), "x and delta_predictions must be on the same device"

        if model is not None:
            x_predictions: torch.Tensor = model(x)
        else:
            assert hasattr(
                self, "model"
            ), "model must be passed to the forward method or the Recourse object must have a model"
            x_predictions = self.model(x)

        assert (
            x_predictions.device == x.device
        ), "x and x_predictions must be on the same device"
        assert (
            x.dtype == x_predictions.dtype
        ), "x and x_predictions must have the same dtype"
        x_neg_predictions_sig: torch.Tensor = torch.sigmoid(
            -x_predictions * self.sigmoid_temp
        )
        delta_neg_predictions_sig: torch.Tensor = torch.sigmoid(
            -delta_predictions * self.sigmoid_temp
        )
        product: torch.Tensor = x_neg_predictions_sig * delta_neg_predictions_sig
        recourse: torch.Tensor = torch.sum(product)
        return recourse
