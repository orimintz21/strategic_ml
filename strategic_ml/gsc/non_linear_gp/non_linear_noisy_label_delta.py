"""
This module implements the NonLinearNoisyLabelDelta model for adversarial users in non-linear models.

The NonLinearNoisyLabelDelta model calculates the delta using a non-linear GP formula where the labels can be
flipped with a probability `p_bernoulli`, or pure noise can be generated based on the Bernoulli distribution.
This allows for modeling adversarial behavior in scenarios where labels might be noisy.

For more information, see the _NonLinearGP class.
"""

# External imports
import torch
from torch import nn
from typing import Optional, Any, Dict

# Internal imports
from strategic_ml.cost_functions.cost_function import _CostFunction
from strategic_ml.gsc.non_linear_gp import _NonLinearGP


class NonLinearNoisyLabelDelta(_NonLinearGP):
    """
    The NonLinearNoisyLabelDelta model calculates delta for adversarial users where the labels
    are noisy. The noise is introduced either by flipping the label with a probability `p_bernoulli`
    or by generating pure noise based on the Bernoulli distribution.

    Parent Class:
        _NonLinearGP
    """

    def __init__(
        self,
        cost: _CostFunction,
        strategic_model: nn.Module,
        cost_weight: float = 1.0,
        p_bernoulli: float = 0.5,
        use_label: bool = True,
        *args,
        training_params: Dict[str, Any],
    ) -> None:
        """
        Initializes the NonLinearNoisyLabelDelta model.

        Args:
            cost (_CostFunction): The cost function used to calculate the delta.
            strategic_model (nn.Module): The non-linear strategic model on which the delta is calculated.
            cost_weight (float, optional): The weight of the cost function. Defaults to 1.0.
            p_bernoulli (float, optional): The probability of the Bernoulli distribution for label flipping or noise generation. Defaults to 0.5.
            use_label (bool, optional): If True, flip the label with `p_bernoulli`; otherwise, generate pure noise. Defaults to True.
            training_params (Dict[str, Any]): A dictionary containing the training parameters.
        """
        super(NonLinearNoisyLabelDelta, self).__init__(
            cost, strategic_model, cost_weight, training_params=training_params
        )
        self.p_bernoulli: float = p_bernoulli
        self.use_label: bool = use_label

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculates the delta for the adversarial users, with optional noisy labels.

        If `use_label` is True, the label `y` is flipped with a probability `p_bernoulli`.
        If `use_label` is False, pure noise is generated using the Bernoulli distribution.

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
        Creates a tensor of Bernoulli samples to simulate noisy labels or pure noise.

        Args:
            batch_size (int): The number of samples in the batch.
            p_bernoulli (float): The probability of Bernoulli distribution for label flipping or noise generation.

        Returns:
            torch.Tensor: A tensor of Bernoulli samples, where 0 is mapped to -1 and 1 remains as 1.
        """

        # Create a tensor of probabilities with the same value p_bernoulli
        probs: torch.Tensor = torch.full((batch_size, 1), p_bernoulli)

        # Sample from the Bernoulli distribution
        bernoulli_tensor: torch.Tensor = torch.bernoulli(probs)

        # Map 0 to -1 and 1 stays 1
        modified_tensor: torch.Tensor = bernoulli_tensor * 2 - 1

        return modified_tensor
