# External Imports
import torch
from torch import nn
from typing import Optional

# Internal Imports
from strategic_ml.strategic_regularization.strategic_regularization import (
    _StrategicRegularization,
)


class Recourse(_StrategicRegularization):
    """This is the Recourse class.
    This class calculates the recourse term of the strategic agents.
    The recourse term is a soft version of sum_{x in X}(1{f(x)!=1 and f(delta(x))!=1})
    Recourse refers to the capacity of a user who is denied a service to restore
    approval through reasonable action (in our case, low-cost feature modification)
    The formula for the recourse term is:
    Recourse = sum_{x in X}(sigmoid(-f(x)*temp)*sigmoid(-f(delta(x))*temp))
    For more information see paper "Strategic Classification Made Practical".

    Parent Class: _StrategicRegularization
    """

    def __init__(
        self, sigmoid_temp: float = 1.0, model: Optional[nn.Module] = None
    ) -> None:
        """Initializer for the Recourse class.

        Args:
            sigmoid_temp (float, optional): The temperature for the sigmoids. Defaults to 1.0.
            model: The model that we use, if not provided it should be provided
            in the forward method.
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
    ) -> torch.Tensor:
        """This is the forward method of the Recourse class.
        This function calculates the recourse term of the strategic agents.
        The recourse term is a soft version of sum_{x in X}(1{f(x)!=1 and f(delta(x))!=1})
        The formula for the recourse term is:
        Recourse = sum_{x in X}(sigmoid(-f(x)*temp)*sigmoid(-f(delta(x))*temp))

        Args:
            x (torch.Tensor): The input of the model.
            delta_predictions (torch.Tensor): The predictions of the model on the delta of x.
            model (Optional[nn.Module], optional): The model that we use, if None
            is provided, the Recourse class will use the one that was provided
            at initialization. Defaults to None.

        Returns:
            torch.Tensor: _description_
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
