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
z == label of the GP (see "Generalized Strategic Classification and the Case of Aligned Incentives")
x == input
x_prime == the GP
m == the weight of the cost
w == the weights of the model
b == the bias of the model
h_{w,b} == model
cost == the cost function

The GP is calculated by the following formula:
The first condition is that the model should be different from the label:
1{z != sign(h(x))}
We use cond1 = 1 - sigmoid(t2 * (tanh(h(x) * t1) * z))
t1 and t2 are the temperature of the soft sign functions.

The second condition is that the cost of the movement should be smaller than
the profit from the movement. Because the maximum profit is 2 and the distance
is the margin, the condition is the following:
We use a sigmoid function with the temperature of t3.
if margin(w,b,x)*m>2 then cond2 = 0
cond2 = sigmoid(t3 * (2 - (m * margin(w,b,x)))

Then the GP is calculated by the following formula:
x_prime = conditions * projection + (1 - conditions) * x
"""

# External imports
import torch
from torch import nn
from typing import Optional

# Internal imports
from strategic_ml.gsc.generalized_strategic_delta import _GSC
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
        strategic_model: nn.Module,
        cost_weight: float = 1.0,
        epsilon: float = 0.01,
    ) -> None:
        """This is the LinearGP model. This model assumes that the model is linear.
        The reason for this assumption is that we can calculate the GP in a closed form
        for linear models.
        Therefore we do not need to train a delta model.

        The GP is calculated by the following formula:
        delta_h(x,z) := argmax_{x' in X}(sigmoid{model(x') == z} - 1/2*weight*(cost(x,x')))

        Args:
            strategic_model (nn.Module): the strategic
            that the delta is calculated on.
            cost (_CostFunction): The cost function of the
            delta. Defaults to None.
            cost_weight (int): The weight of the cost function.
            epsilon (float): move to the negative/positive
            direction of the model to make sure that the model will predict the
            label correctly. Defaults to 0.01.
        """
        super(_LinearGP, self).__init__(
            strategic_model=strategic_model, cost=cost, cost_weight=cost_weight
        )
        self._assert_model()
        self._assert_cost()
        self.epsilon: float = epsilon

    def find_x_prime(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """This function calculates x' based on the GP formula.
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
        self._validate_input(x, z)

        # Get the weights and bias of the model
        assert isinstance(
            self.strategic_model, LinearStrategicModel
        ), "The model should be a LinearStrategicModel, but got {}".format(
            type(self.strategic_model)
        )
        weights, bias = self.strategic_model.get_weights_and_bias()
        x_prime: Optional[torch.Tensor] = None

        for x_sample, z_sample in zip(x, z):
            x_sample = x_sample.view(1, -1)
            projection: torch.Tensor = self._calculate_projection(
                x_sample, weights, bias, z_sample
            )

            assert (
                torch.sign(self.strategic_model(projection)) == z_sample
            ), "The projection is wrong, {}".format(self.strategic_model(projection))

            worth_to_move: bool = self._worth_to_move(x_sample, projection)

            if (torch.sign(self.strategic_model(x_sample)) == z_sample) or (
                not worth_to_move
            ):
                what_to_add = x_sample
            else:
                what_to_add = projection

            if x_prime is None:
                x_prime = what_to_add
            else:
                x_prime = torch.cat((x_prime, what_to_add))

        assert x_prime is not None, "The x_prime is None after the loop"
        return x_prime

    def get_z(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This is an abstract method")

    def _get_minimal_distance(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        self._validate_input(x, z)
        # Get the weights and bias of the model
        assert isinstance(
            self.strategic_model, LinearStrategicModel
        ), "The model should be a LinearStrategicModel, but got {}".format(
            type(self.strategic_model)
        )
        weights, bias = self.strategic_model.get_weights_and_bias()
        costs: Optional[torch.Tensor] = None

        for x_sample, z_sample in zip(x, z):
            x_sample: torch.Tensor = x_sample.view(1, -1)
            if torch.sign(self.strategic_model(x_sample)) == z_sample:
                costs_of_moment = torch.tensor([0])
            else:
                projection: torch.Tensor = self._calculate_projection(
                    x_sample, weights, bias, z_sample
                )

                assert (
                    torch.sign(self.strategic_model(projection)) == z_sample
                ), "The projection is wrong, {}".format(
                    self.strategic_model(projection)
                )

                costs_of_moment: torch.Tensor = self.cost(x_sample, projection)
            if costs is None:
                costs = costs_of_moment
            else:
                costs = torch.cat((costs, costs_of_moment))

        assert costs is not None, "The costs is None after the loop"
        return costs

    def _assert_cost(self) -> None:
        assert isinstance(self.cost, CostNormL2) or isinstance(
            self.cost, CostWeightedLoss
        ), "The cost should be a  CostNormL2 or CostWeightedLoss"

    def _validate_input(self, x: torch.Tensor, z: torch.Tensor) -> None:
        # Check the input
        self._assert_cost()
        self._assert_model()
        batch_size: int = x.size(0)
        assert z.size() == torch.Size(
            [batch_size, 1]
        ), "z should be of size [batch_size, 1], but got {}".format(z.size())

        # Get the weights and bias of the model
        assert isinstance(
            self.strategic_model, LinearStrategicModel
        ), "The model should be a LinearStrategicModel, but got {}".format(
            type(self.strategic_model)
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
        z: torch.Tensor = torch.tensor([1]),
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

        assert z.size() == torch.Size(
            [1]
        ), "z should be of size [1], but got {}".format(z.size())

        if norm_waits is None:
            norm_w = torch.matmul(w, w.T)
            projection = x - ((torch.matmul(x, w.T) + b) / (torch.matmul(w, w.T))) * w

        else:
            inverse_norm_waits: torch.Tensor = torch.inverse(norm_waits)
            norm_w = torch.matmul(w, torch.matmul(inverse_norm_waits, w.T))
            projection = x - ((torch.matmul(x, w.T) + b) / norm_w) * torch.matmul(
                inverse_norm_waits, w.T
            )

        unit_w = w / norm_w
        projection_with_safety = projection + z * self.epsilon * unit_w
        return projection_with_safety

    def _worth_to_move(self, x: torch.Tensor, projection: torch.Tensor) -> bool:
        cost_of_moment = self.cost_weight * self.cost(x, projection)
        return bool(cost_of_moment < 2)
