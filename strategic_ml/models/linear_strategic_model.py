""" linear_strategic_model.py
This is the linear strategic model.
The linear strategic model calculates the relent delta and the strategic regularization
and uses them to modify the input data before passing it to the model.

We implement the LinearStrategicModel class because when we use a linear model
we can calculate the strategic delta in a closed form.
"""

# External imports
import torch
from torch import nn
from typing import Tuple

# Internal imports


class LinearStrategicModel(nn.Module):
    def __init__(
        self,
        in_features: int,
    ) -> None:
        """
        Constructor for the LinearStrategicModel class.
        This is a binary classification model therefore the output features is 1.
        """
        super(LinearStrategicModel, self).__init__()
        self.model: torch.nn.Linear = torch.nn.Linear(
            in_features=in_features, out_features=1, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def train(self, mode: bool = True) -> "LinearStrategicModel":
        return super().train(mode)

    def get_weights_and_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The get_weights_and_bias method returns the weights and bias of the model

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the weights and bias of the model
        """
        return self.model.weight, self.model.bias
