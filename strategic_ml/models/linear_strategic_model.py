""" linear_strategic_model.py
This is the linear strategic model.
The linear strategic model calculates the relent delta and the strategic regularization
and uses them to modify the input data before passing it to the model.

We implement the LinearStrategicModel class because when we use a linear model
we can calculate the strategic delta in a closed form.
"""

# External imports
import torch
from typing import Optional, Tuple

# Internal imports
from strategic_ml.models import _StrategicModel
from strategic_ml.gsc import _GSC


class LinearStrategicModel(_StrategicModel):
    def __init__(
        self,
        in_features: int,
        delta: Optional[_GSC] = None,
    ) -> None:
        """
        Constructor for the LinearStrategicModel class.
        """
        model: torch.nn.Linear = torch.nn.Linear(
            in_features=in_features, out_features=1, bias=True
        )
        super(LinearStrategicModel, self).__init__(delta=delta, model=model)

    def get_weights_and_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The get_weights_and_bias method returns the weights and bias of the model

        Raises:
            NotImplementedError: if the model is not a Linear model
            (This should not happen)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the weights and bias of the model
        """
        if isinstance(self.model, torch.nn.Linear):
            return self.model.weight, self.model.bias

        raise NotImplementedError("The model is not a Linear model")
