# External imports
import torch
from torch import nn
from typing import Optional, Any, Dict

# Internal imports
from strategic_ml.cost_functions.cost_function import _CostFunction
from strategic_ml.gsc.non_linear_gp import _NonLinearGP


class NonLinearNoisyLabelDelta(_NonLinearGP):
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
        super(NonLinearNoisyLabelDelta, self).__init__(
            cost, strategic_model, cost_weight, training_params=training_params
        )
        self.p_bernoulli: float = p_bernoulli
        self.use_label: bool = use_label

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): The data
            y (torch.Tensor): y or None. If it is y, then the we use change each
            label with a probability p_bernoulli. If it is None, then we use
            a vector of Bernoulli samples with probability p_bernoulli. Defaults to None.

        Returns:
            torch.Tensor: the modified data
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
        """Create a tensor of Bernoulli samples.

        Args:
            batch_size (int): the number of samples
            p_bernoulli (float): the probability of the Bernoulli distribution

        Returns:
            torch.Tensor: the tensor of Bernoulli samples
        """

        # Create a tensor of probabilities with the same value p_bernoulli
        probs: torch.Tensor = torch.full((batch_size, 1), p_bernoulli)

        # Sample from the Bernoulli distribution
        bernoulli_tensor: torch.Tensor = torch.bernoulli(probs)

        # Map 0 to -1 and 1 stays 1
        modified_tensor: torch.Tensor = bernoulli_tensor * 2 - 1

        return modified_tensor
