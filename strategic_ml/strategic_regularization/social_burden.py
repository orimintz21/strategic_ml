# external imports
import torch
from torch import nn
import torch.nn.functional as F

# internal imports
from strategic_ml import _CostFunction
from strategic_ml.strategic_regularization.strategic_regularization import (
    _StrategicRegularization,
)


class SocialBurden(_StrategicRegularization):
    def __init__(
        self,
        cost_fn: _CostFunction,
        poss_pred_temp: float = 1.0e3,
        poss_pred_mask: float = 1.0e2,
    ) -> None:
        """
        Constructor for the SocialBurden class.
        """
        super(SocialBurden, self).__init__()

        self.cost_fn = cost_fn
        self.poss_pred_temp = poss_pred_temp
        self.poss_pred_mask = poss_pred_mask

    def forward(
        self,
        x: torch.Tensor,
        x_prime: torch.Tensor,
        y: torch.Tensor,
        predictions: torch.Tensor,
    ) -> torch.Tensor:

        positive_label = y == 1
        positive_label = positive_label.squeeze()

        x_positive = x[positive_label]
        x_prime_pos = x_prime[positive_label]
        positive_label_predictions = predictions[positive_label]

        pred_positive_mask = torch.sigmoid(
            positive_label_predictions * self.poss_pred_temp
        ).T

        # Calculate the cost of the positive examples
        costs = self.cost_fn(x_positive, x_prime_pos)

        # Select the cost of the positive examples
        mask_costs = costs + (1 - pred_positive_mask) * self.poss_pred_mask

        softmin_costs = F.softmin(mask_costs, dim=1)

        min_costs = (softmin_costs * mask_costs).sum(dim=1)

        total_reg = min_costs.sum()

        return total_reg
