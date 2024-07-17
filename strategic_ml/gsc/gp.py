""":file: gp.py

The GP is a strategic delta that is calculated by the following formula:
delta_h(x,z) = argmax_{x' in X}(1{model(x') = z} - 1/2*(cost(x,x')))

We can see that the GP is non-differentiable (and even non-continues), 
and therefore, we cannot use it directly in the optimization process.
"""

# External imports
from typing import Optional
import torch

# Internal imports
from strategic_ml.gsc.generalized_strategic_delta import _GSC
from strategic_ml.models import _StrategicModel
from strategic_ml.cost_functions import _CostFunction


class _GP(_GSC):

    def __init__(
        self,
        model: Optional[_StrategicModel] = None,
        cost: Optional[_CostFunction] = None,
    ) -> None:
        """Constructor for the _GP class.

        Args:
            model (Optional[_StrategicModel], optional): the model. Defaults to None.
            cost (Optional[_CostFunction], optional): the cost function. Defaults to None.
        """
        super(_GP, self).__init__(model=model, cost=cost)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """The forward method of the GP class calculates the x' - the modified data.

        We use the z tensor to generalize all of the possible GP deltas.

        Args:
            x (torch.Tensor): the data
            z (torch.Tensor): z is a tensor that generalize all of the possible
                GP deltas, for example, in the strategic case, z is equal to 1
                and in the adversarial case, z is equal to -y.

        Raises:
            NotImplementedError: this is an interface, you should implement this method in your subclass

        Returns:
            torch.Tensor: x' - the modified data
        """
        raise NotImplementedError()
