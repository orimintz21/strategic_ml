# External imports
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any

# Internal imports
from strategic_ml.cost_functions.cost_function import _CostFunction
from strategic_ml.gsc.non_linear_gp import _NonLinearGP


class NonLinearStrategicDelta(_NonLinearGP):
    """
    NonLinearStrategicDelta is a model that calculates the delta for strategic users in non-linear settings.
    Strategic users aim to maximize the model's prediction with minimal cost.
    The delta is calculated based on the following formula:
    x_prime = argmax_{x' in X}(1{model(x') = 1} - r/2 * (cost(x,x')))
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
        Initializes the NonLinearStrategicDelta model.

        Args:
            cost (_CostFunction): The cost function of the delta.
            strategic_model (nn.Module): The model used for strategic classification.
            cost_weight (float, optional): The weight of the cost function. Defaults to 1.
            save_dir (str, optional): Directory to save the computed x_prime values. Defaults to ".".
            training_params (Dict[str, Any]): Dictionary containing training parameters.
        """
        super(NonLinearStrategicDelta, self).__init__(
            cost,
            strategic_model,
            cost_weight,
            save_dir=save_dir,
            training_params=training_params,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the delta for strategic users by optimizing the input data.

        Args:
            x (torch.Tensor): Input data.
            y (torch.Tensor): Target labels.

        Returns:
            torch.Tensor: The optimized x_prime tensor.
        """
        # array of ones with the number of rows of x
        z = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
        return super().find_x_prime(x, z)

    def _gen_z_fn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Generates the z values for strategic users, where z = 1.

        Args:
            x (torch.Tensor): Input data.
            y (torch.Tensor): Target labels.

        Returns:
            torch.Tensor: The generated z values (1).
        """
        return torch.ones_like(y)
