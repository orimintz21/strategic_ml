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
    def __init__(self, cost_fn: _CostFunction) -> None:
        """
        Constructor for the SocialBurden class.
        """
        super(SocialBurden, self).__init__()

        self.cost_fn = cost_fn

    def forward(
        self,
        x: torch.Tensor,
        x_prime: torch.Tensor,
        y: torch.Tensor,
        predictions: torch.Tensor,
    ) -> torch.Tensor:

        positive_label = y == 1

        x_positive = x[positive_label]
        x_prime_pos = x[positive_label]
        positive_label_predictions = predictions[positive_label]

        pred_positive_mask = (positive_label_predictions > 0).float()

        costs = self.cost_fn(x_positive, x_prime_pos)

        mask_costs = costs + (1 - pred_positive_mask) * 1e6

        softmin_costs = F.softmin(mask_costs)

        min_costs = (softmin_costs * mask_costs).sum(dim=-1)

        total_reg = min_costs.sum()

        return total_reg
