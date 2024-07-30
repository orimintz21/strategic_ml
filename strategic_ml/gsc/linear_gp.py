""":file: gp.py

The GP is a strategic delta that is calculated by the following formula:
delta_h(x,z) = argmax_{x' in X}(1{model(x') = z} - 1/2*(cost(x,x')))

We can see that if model(x) == z then x is the best x' that we can choose.
If for every x' in X, such that model(x') != z, we get cost(x,x') > 1, then
the GP will be x (it is not worth to change x).

If we assume that the model is linear and the cost is norm2, or weighted
norm, we see that if model(x) != z and the margin from the model is smaller than 
the wait of the cost using the norm to calculate the margin, then the GP will be 
the projection of x on the model.

Then the GP is calculated by the following formula:
x_prime = x - ((w.T @ x + b) / (w.T @ w)) * w
if we use the norm2 cost function.
and
x_prime = x - ((w.T @ x + b) / (w.T @ inverse(Lambda) @ w)) * inverse(Lambda) @ w
if we use the weighted norm cost function with Lambda as the weight matrix.

The problem is that the condition that the margin should be smaller than weight 
of the cost is not differentiable and that model(x) != z is not differentiable as well.
Therefore we need to use the sigmoid function to approximate the condition and
the margin.

Notions:
h == model
z == label of the GP (see "Generalized Strategic Classification and the Case of Aligned Incentives")
x == input
x_prime == the GP
m == the weight of the cost
w == the weights of the model
b == the bias of the model
cost == the cost function

The GP is calculated by the following formula:
The first condition is that the model should be different from the label:
1{z != sign(h(x))}
We use cond1 = m - sigmoid(t2 * (tanh(h(x) * t1) * z))
t1 and t2 are the temperature of the soft sign functions.

The second condition is that the margin should be smaller than 1:
We use a sigmoid function with the temperature of t3.
if margin(w,b,x)>1 then cond2 = 0
cond2 = sigmoid(t3 * (1 - margin(w,b,x))

Then the GP is calculated by the following formula:
x_prime = conditions * projection + (1 - conditions) * x
"""

# External imports
import torch
from typing import Optional

# Internal imports
from strategic_ml.gsc.generalized_strategic_delta import _GSC
from strategic_ml.models import _StrategicModel
from strategic_ml.models import LinearStrategicModel
from strategic_ml.cost_functions import (
    _CostFunction,
    CostNormL2,
    CostWeightedLoss,
)


class _LinearGP(_GSC):
    def __init__(
        self,
        cost: _CostFunction,
        strategic_model: _StrategicModel,
        cost_weight: float = 1.0,
        models_temp: float = 1.0,
        z_temp: float = 1.0,
        margin_temp: float = 1.0,
    ) -> None:
        """This is the LinearGP model. This model assumes that the model is linear.
        The reason for this assumption is that we can calculate the GP in a closed form
        for linear models.
        Therefore we do not need to train a delta model.

        The GP is calculated by the following formula:
        delta_h(x,z) := argmax_{x' in X}(sigmoid{model(x') == z} - 1/2*weight*(cost(x,x')))

        Args:
            strategic_model (_StrategicModel): the strategic
            that the delta is calculated on.
            cost (_CostFunction): The cost function of the
            delta. Defaults to None.
            cost_weight (int): The weight of the cost function.
            temperature (float): The temperature of the sigmoid function.
        """
        super(_LinearGP, self).__init__(
            strategic_model=strategic_model, cost=cost, cost_weight=cost_weight
        )
        self._assert_model()
        self._assert_cost()
        self.models_temp: float = models_temp
        self.z_temp: float = z_temp
        self.margin_temp: float = margin_temp

    def forward(
        self, x: torch.Tensor, z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """The forward method of the LinearGP model.
        This function calculates x' based on the GP formula.
        For more information see the file prolog.


        Args:
            x (torch.Tensor): The input of the model.
            z (torch.Tensor): Meta data for the GP.

        Raises:
            AssertionError: If the cost or the model is None.
            NotImplementedError: If the model is not a Linear model (This should not happen)

        Returns:
            torch.Tensor: x' the GP.
        """

        self._assert_cost()
        self._assert_model()
        assert z is not None, "The label of the GP is None"
        assert z.size() == torch.Size(
            [x.size(0), 1]
        ), "z should be of size [batch_size, 1]"

        if isinstance(self.strategic_model, LinearStrategicModel):
            # check again if the model is linear to avid linting errors
            weights, bias = self.strategic_model.get_weights_and_bias()
        else:
            raise NotImplementedError("The model is not a Linear model")

        if isinstance(self.cost, CostNormL2):
            margin: torch.Tensor = self._calculate_margin(x, weights, bias)
        elif isinstance(self.cost, CostWeightedLoss):
            norm_waits: torch.Tensor = self.cost.get_weights
            assert norm_waits is not None, "The weights of the cost function are None"
            margin = self._calculate_margin(x, weights, bias, norm_waits)
        else:
            raise ValueError("The cost function is not supported")

        assert margin.size() == torch.Size(
            [x.size(0), 1]
        ), "The margin should be of size [batch_size, 1]"

        model_prediction_tanh: torch.Tensor = torch.tanh(
            self.models_temp * (torch.matmul(x, weights) + bias)
        )
        z_neq_prediction: torch.Tensor = 1 - torch.sigmoid(
            model_prediction_tanh * self.z_temp * z
        )

        assert len(z_neq_prediction.size()) == 1, "The margin should be one dimensional"
        assert z_neq_prediction.size(0) == 1, "The margin should be of size 1"

        margin_smaller_than_1: torch.Tensor = torch.sigmoid(
            self.margin_temp * (1 - margin)
        )

        assert (
            len(margin_smaller_than_1.size()) == 1
        ), "The margin should be one dimensional"
        assert margin_smaller_than_1.size(0) == 1, "The margin should be of size 1"

        conditions: torch.Tensor = z_neq_prediction * margin_smaller_than_1

        return (
            conditions * self._calculate_projection(x, weights, bias)
            + (1 - conditions) * x
        )

    def _assert_cost(self) -> None:
        assert isinstance(self.cost, CostNormL2) or isinstance(
            self.cost, CostWeightedLoss
        ), "The cost should be a  CostNormL2 or CostWeightedLoss"

        if isinstance(self.cost, CostWeightedLoss):
            raise NotImplementedError(
                "The weighted loss cost function is not supported yet (TODO!!)"
            )

    def _assert_model(self) -> None:
        assert isinstance(
            self.strategic_model, LinearStrategicModel
        ), "The strategic model should be a StrategicModel"

    def _calculate_projection(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        norm_waits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """This method calculates the projection on the model.

        Args:
            x (torch.Tensor): The data
            w (torch.Tensor): The weights of the model
            b (torch.Tensor): The bias of the model
            norm_waits (torch.Tensor): The weights of the cost function

        Returns:
            torch.Tensor: The projection on the model
        """
        if norm_waits is None:
            return x - ((torch.matmul(x, w) + b) / (torch.matmul(w, w))) * w

        else:
            inverse_norm_waits: torch.Tensor = torch.inverse(norm_waits)
            return x - (
                (torch.matmul(x, w) + b)
                / (torch.matmul(w, torch.matmul(inverse_norm_waits, w)))
            ) * torch.matmul(inverse_norm_waits, w)

    def _calculate_margin(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        norm_waits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """This method calculates the projection on the model.

        Args:
            x (torch.Tensor): The data
            w (torch.Tensor): The weights of the model
            b (torch.Tensor): The bias of the model
            norm_waits (torch.Tensor): The weights of the cost function

        Returns:
            torch.Tensor: The margin from the model
        """
        if norm_waits is None:
            # The equation is |w.T @ x + b| / ||w||
            return torch.abs(torch.matmul(x, w) + b) / torch.sqrt((torch.matmul(w, w)))

        else:
            # The equation is |w.T @ x + b| / ||w||_{waits}
            inverse_norm_waits: torch.Tensor = torch.inverse(norm_waits)
            return torch.abs(torch.matmul(x, w) + b) / (
                torch.sqrt(torch.matmul(w, torch.matmul(inverse_norm_waits, w)))
            )
