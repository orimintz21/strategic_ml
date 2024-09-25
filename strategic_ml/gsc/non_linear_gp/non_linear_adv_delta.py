# External imports
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any

# Internal imports
from strategic_ml.cost_functions.cost_function import _CostFunction
from strategic_ml.gsc.non_linear_gp import _NonLinearGP


class NonLinearAdvDelta(_NonLinearGP):
    """
    NonLinearAdvDelta is a model that calculates the delta for adversarial users in non-linear settings.
    The adversarial users aim to receive incorrect predictions with minimal cost.
    The delta is calculated based on the following formula:
    x_prime = argmax_{x' in X}(1{model(x') = -y} - r/2 * (cost(x,x')))
    """

    def __init__(
        self,
        cost: _CostFunction,
        strategic_model: nn.Module,
        cost_weight: float = 1.0,
        save_dir: str = ".",
        *args,
        training_params: Dict[str, Any],
    ) -> None:
        """
        Initializes the NonLinearAdvDelta model.

        Args:
            cost (_CostFunction): The cost function of the delta.
            strategic_model (nn.Module): The model used for strategic classification.
            cost_weight (float, optional): The weight of the cost function. Defaults to 1.
            save_dir (str, optional): Directory to save the computed x_prime values. Defaults to ".".
            training_params (Dict[str, Any]): Dictionary containing training parameters.
        """
        super(NonLinearAdvDelta, self).__init__(
            cost,
            strategic_model,
            cost_weight,
            save_dir=save_dir,
            training_params=training_params,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the delta for adversarial users by optimizing the input data.

        Args:
            x (torch.Tensor): Input data.
            y (torch.Tensor): Target labels.

        Returns:
            torch.Tensor: The optimized x_prime tensor.
        """
        y = y.to(x.device)
        z = -y
        return super().find_x_prime(x, z)

    def _gen_z_fn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Generates the z values for adversarial users, where z = -y.

        Args:
            x (torch.Tensor): Input data.
            y (torch.Tensor): Target labels.

        Returns:
            torch.Tensor: The generated z values (-y).
        """
        y = y.to(x.device)
        return -y
