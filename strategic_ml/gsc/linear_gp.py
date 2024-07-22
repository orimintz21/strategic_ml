""":file: gp.py

The GP is a strategic delta that is calculated by the following formula:
delta_h(x,z) = argmax_{x' in X}(1{model(x') = z} - 1/2*(cost(x,x')))

We can see that if model(x) == z then x is the best x' that we can choose.
If for every x' in X, such that model(x') != z, we get cost(x,x') > 1, then
the GP will be x (it is not worth to change x).

If we assume that the model is linear and the cost is norm2, MSE, or weighted
norm, we see that if model(x) != z and the margin from the model is smaller then 
1 using the norm to calculate the margin, then the GP will be the projection of 
x on the model.

Then the GP is calculated by the following formula:
x_prime = x - ((w.T @ x + b) / (w.T @ w)) * w
if we use the norm2 cost function.
and
x_prime = x - ((w.T @ x + b) / (w.T @ inverse(Lambda) @ w)) * inverse(Lambda) @ w
if we use the weighted norm cost function with Lambda as the weight matrix.

The problem is that the condition that the margin should be smaller than 1 is not
differentiable and that model(x) != z is not differentiable as well.
Therefore we need to use the sigmoid function to approximate the condition and
the margin.

Notions:
h == model
z == label of the GP (see "Generalized Strategic Classification and the Case of Aligned Incentives")
x == input
x_prime == the GP
w == the weights of the model
b == the bias of the model
cost == the cost function

The GP is calculated by the following formula:
The first condition is that the model should be different from the label:
We use cond1 = 1 - tanh(t2 * ( sigmoid(h(x) * t1) - z))
t1 and t2 are the temperature of the soft sign functions.

The second condition is that the margin should be smaller than 1:
We use a sigmoid function with the temperature of t3.
cond2 = 1 - tanh(t3 * (1 - margin(w,b,x))

Then the GP is calculated by the following formula:
x_prime = conditions * projection + (1 - conditions) * x
"""

# External imports
import torch
from typing import Optional, Callable

# Internal imports
from strategic_ml.gsc.generalized_strategic_delta import _GSC
from strategic_ml.models import _StrategicModel
from strategic_ml.cost_functions import (
    _CostFunction,
    CostNormL2,
    CostWeightedLoss,
)


class _LinearGP(_GSC):
    def __init__(
        self,
        strategic_model: Optional[_StrategicModel] = None,
        cost: Optional[_CostFunction] = None,
        models_temp: float = 1.0,
        z_temp: float = 1.0,
        margin_temp: float = 1.0,
    ) -> None:
        """This is the LinearGP model. This model assumes that the model is linear.
        The reason for this assumption is that we can calculate the GP in a closed form
        for linear models.
        Therefore we do not need to train a delta model.

        The GP is calculated by the following formula:
        delta_h(x,z) := argmax_{x' in X}(sigmoid{model(x') == z} - 1/2*(cost(x,x')))

        Args:
            strategic_model (Optional[_StrategicModel], optional): the strategic
            that the delta is calculated on. Defaults to None.
            cost (Optional[_CostFunction], optional): The cost function of the
            delta. Defaults to None.
            temperature (float): The temperature of the sigmoid function.
        """
        super(_LinearGP, self).__init__(strategic_model=strategic_model, cost=cost)
        if strategic_model is not None:
            # assert isinstance(self.strategic_model, LinearStrategicModel), "The strategic model should be a StrategicModel"
            pass

        if cost is not None:
            self._assert_cost()

        self.models_temp: float = models_temp
        self.z_temp: float = z_temp
        self.margin_temp: float = margin_temp

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """The forward method of the LinearGP model.
        This function calculates x' based on the GP formula.
        For more information see the file prolog.


        Args:
            x (torch.Tensor): The input of the model.
            z (torch.Tensor): Meta data for the GP.

        Returns:
            torch.Tensor: x' the GP.
        """

        self._assert_cost()
        assert self.strategic_model is not None, "The strategic model is None"
        assert self.cost is not None, "The cost function is None"
        assert z.size() == torch.Size(
            [x.size(0), 1]
        ), "z should be of size [batch_size, 1]"
        assert x.size(0) == z.size(0), "x and z should have the same batch size"

    def _assert_cost(self) -> None:
        assert self.cost is not None, "The cost function is None"
        assert isinstance(self.cost, CostNormL2) or isinstance(
            self.cost, CostWeightedLoss
        ), "The cost should be a  CostNormL2 or CostWeightedLoss"

    def _assert_model(self) -> None:
        assert self.strategic_model is not None, "The strategic model is None"
        # assert isinstance(self.strategic_model, LinearStrategicModel), "The strategic model should be a StrategicModel"
