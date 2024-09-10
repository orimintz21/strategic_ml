"""
This module implements the LinearNoisyLabelDelta model.

The LinearNoisyLabelDelta model calculates the delta based on the GP formula.
In this model, z represents the noisy label, and there are two types of noise:
1. The label is flipped with a probability `p_bernoulli`.
2. Pure noise {-1, 1} is generated with a probability `p_bernoulli`.

The delta represents the movement of a strategic user trying to receive the noisy label from the model.
"""

# External imports
import torch
from torch import nn
from typing import Optional

# Internal imports
from strategic_ml.cost_functions.cost_function import _CostFunction
from strategic_ml.gsc.linear_gp.linear_gp import _LinearGP


class LinearNoisyLabelDelta(_LinearGP):
    """
    The LinearNoisyLabelDelta model calculates the delta based on the GP formula.
    In this model, z represents the noisy label, and there are two types of noise:
    1. The label is flipped with a probability `p_bernoulli`.
    2. Pure noise {-1, 1} is generated with a probability `p_bernoulli`.

    Args:
        cost (_CostFunction): The cost function to calculate the delta.
        strategic_model (nn.Module): The strategic model, assumed to be linear.
        cost_weight (float, optional): The weight of the cost function. Defaults to 1.0.
        p_bernoulli (float, optional): The probability of Bernoulli distribution for noise generation. Defaults to 0.5.
        use_label (bool, optional): If True, flip the label with `p_bernoulli`; otherwise, use pure noise. Defaults to True.
        epsilon (float, optional): A small value to ensure that the model predicts correctly by adjusting the projection. Defaults to 0.01.
    """

    def __init__(
        self,
        cost: _CostFunction,
        strategic_model: nn.Module,
        cost_weight: float = 1.0,
        p_bernoulli: float = 0.5,
        use_label: bool = True,
        epsilon: float = 0.01,
    ) -> None:
        """
        Initializes the LinearNoisyLabelDelta model.

        Args:
            cost (_CostFunction): The cost function used for calculating the delta.
            strategic_model (nn.Module): The strategic model (assumed linear) on which the delta is calculated.
            cost_weight (float, optional): The weight of the cost function. Defaults to 1.0.
            p_bernoulli (float, optional): The probability of Bernoulli distribution for flipping labels or generating pure noise. Defaults to 0.5.
            use_label (bool, optional): Determines whether to flip labels with Bernoulli probability or generate pure noise. Defaults to True.
            epsilon (float, optional): A small adjustment added to the projection direction to ensure correct predictions. Defaults to 0.01.
        """
        super(LinearNoisyLabelDelta, self).__init__(
            cost, strategic_model, cost_weight, epsilon
        )
        self.p_bernoulli: float = p_bernoulli
        self.use_label: bool = use_label

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculates the delta for the noisy label model.

        If `use_label` is True, the label `y` is flipped with a probability `p_bernoulli`.
        If `use_label` is False, pure noise is generated with the Bernoulli distribution.

        Args:
            x (torch.Tensor): The input data.
            y (Optional[torch.Tensor]): If provided, represents the label. If None, Bernoulli noise is applied. Defaults to None.

        Returns:
            torch.Tensor: The modified data with delta applied.
        """
        bernoulli_tensor = self._create_bernoulli_tensor(x.shape[0], self.p_bernoulli)
        if self.use_label:
            assert y is not None, "y should not be None"
            z = y * bernoulli_tensor
        else:
            z = bernoulli_tensor

        return super().find_x_prime(x, z)

    @staticmethod
    def _create_bernoulli_tensor(batch_size: int, p_bernoulli: float) -> torch.Tensor:
        """
        Creates a tensor of Bernoulli samples to simulate noise.

        Args:
            batch_size (int): The number of samples in the batch.
            p_bernoulli (float): The probability of Bernoulli distribution for noise generation.

        Returns:
            torch.Tensor: A tensor of Bernoulli samples, where 0 is mapped to -1 and 1 stays as 1.
        """

        # Create a tensor of probabilities with the same value p_bernoulli
        probs: torch.Tensor = torch.full((batch_size, 1), p_bernoulli)

        # Sample from the Bernoulli distribution
        bernoulli_tensor: torch.Tensor = torch.bernoulli(probs)

        # Map 0 to -1 and 1 stays 1
        modified_tensor: torch.Tensor = bernoulli_tensor * 2 - 1

        return modified_tensor
