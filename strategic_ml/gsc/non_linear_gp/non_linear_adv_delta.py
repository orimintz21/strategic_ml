"""
This module implements the NonLinearAdvDelta model for adversarial users in non-linear models.

The NonLinearAdvDelta model calculates delta based on the non-linear GP formula without any assumptions on the model or cost function.
The delta is optimized alternately with the model by using the gradient of the model to find the optimal x'. 
Large datasets are handled by saving the calculated x' values to disk and loading them as needed.

For more information, see the _NonLinearGP class.
"""

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
    The NonLinearAdvDelta model calculates delta for adversarial users in non-linear models.
    This model does not impose any assumptions on the model or cost function.

    The delta is calculated using the non-linear GP formula:
    x_prime = argmax_{x' in X}(1{model(x') = -y} - r/2 * (cost(x,x')))
    (i.e. z = -y).

    Instead of optimizing both the model and delta in the same epoch, the delta is 
    optimized alternately with the model using gradients. 

    For large datasets, x_prime values are computed and saved to disk to be loaded when needed.

    Parent Class:
        _NonLinearGP
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
            cost (_CostFunction): The cost function used to calculate the delta.
            strategic_model (nn.Module): The strategic model (non-linear) on which the delta is calculated.
            cost_weight (float, optional): The weight of the cost function. Defaults to 1.0.
            save_dir (str): Directory to save the computed x_prime values to disk.
            training_params (Dict[str, Any]): Dictionary containing the training parameters.
        
        Training parameters include:
            - optimizer_class (type): The optimizer class to be used (default: SGD).
            - optimizer_params (Dict[str, Any]): Parameters for the optimizer (default: {"lr": 0.01}).
            - scheduler_class (type, optional): The scheduler class for optimization (optional).
            - scheduler_params (Dict[str, Any]): Parameters for the scheduler (default: {}).
            - early_stopping (int, optional): Number of epochs for early stopping (default: -1, no early stopping).
            - num_epochs (int, optional): Number of epochs for optimization (default: 100).
            - temp (float, optional): Temperature for the model's tanh function (default: 1.0).
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
        Calculates the delta for adversarial users using the find_x_prime method from the parent class.

        If a precomputed x_prime is not available, it will calculate it and return the delta.
        Use the train method to calculate x_prime and save it to disk.

        Args:
            x (torch.Tensor): The input data.
            y (torch.Tensor): The label.

        Returns:
            torch.Tensor: x_prime, the calculated delta.
        """
        return super().find_x_prime(x, -y)

    def _gen_z_fn(self, data: torch.Tensor) -> torch.Tensor:
        """
        Generates the z value for the non-linear GP formula. In this case, z = -y.

        Args:
            data (torch.Tensor): A tuple containing the input data and labels.

        Returns:
            torch.Tensor: The negated label (-y).
        """
        _, y = data
        return -y
