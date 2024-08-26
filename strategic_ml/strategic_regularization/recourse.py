# External Imports
import torch
from torch import nn
from typing import Optional

# Internal Imports
from strategic_ml.strategic_regularization.strategic_regularization import (
    _StrategicRegularization,
)


class Recourse(_StrategicRegularization):
    def __init__(
        self, sigmoid_temp: float = 1.0, model: Optional[nn.Module] = None
    ) -> None:
        """
        Constructor for the Recourse class.
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
        assert (
            x.shape[0] == delta_predictions.shape[0]
        ), "x and delta_predictions must have the same batch size"

        if model is not None:
            x_predictions = model(x)
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
