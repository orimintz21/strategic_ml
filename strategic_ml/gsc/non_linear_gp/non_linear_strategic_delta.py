"""
This module implements the NonLinearStrategicDelta model for strategic users in non-linear models.

The NonLinearStrategicDelta model calculates delta based on the non-linear GP formula without any assumptions 
on the model or cost function. The delta is optimized alternately with the model to ensure strategic users 
achieve positive predictions at minimal cost. For large datasets, the delta values (x_prime) are written to disk 
and loaded as needed.

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


class NonLinearStrategicDelta(_NonLinearGP):
    """
    The NonLinearStrategicDelta model calculates delta for strategic users based on 
    the non-linear GP formula. This model does not assume anything about the model 
    or the cost function.

    The delta is calculated using the formula:
    x_prime = argmax_{x' in X}(1{model(x') = 1} - r/2 * (cost(x, x')))
    (i.e. z = 1).

    The goal of the strategic user is to achieve a positive prediction from the model 
    while minimizing the associated cost. The optimization is performed alternately 
    between the delta and the model, allowing for efficient computation, even for 
    large datasets, where x_prime is saved to disk.

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
        Initializes the NonLinearStrategicDelta model for strategic classification.

        Args:
            cost (_CostFunction): The cost function for delta calculation.
            strategic_model (nn.Module): The non-linear strategic model for calculating the delta.
            cost_weight (float, optional): The weight of the cost function. Defaults to 1.0.
            save_dir (str): Directory where computed x_prime values are saved.
            training_params (Dict[str, Any]): A dictionary containing the training parameters.

        Training parameters include:
            - optimizer_class (type): The optimizer class for optimization (default: SGD).
            - optimizer_params (Dict[str, Any]): Parameters for the optimizer (default: {"lr": 0.01}).
            - scheduler_class (type, optional): The scheduler class for optimization.
            - scheduler_params (Dict[str, Any]): Parameters for the scheduler (default: {}).
            - early_stopping (int, optional): Number of epochs for early stopping (default: -1, no early stopping).
            - num_epochs (int, optional): Number of epochs for optimization (default: 100).
            - temp (float, optional): Temperature for the model's tanh function (default: 1.0).
        """
        super(NonLinearStrategicDelta, self).__init__(
            cost,
            strategic_model,
            cost_weight,
            save_dir=save_dir,
            training_params=training_params,
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Calculates the delta for strategic users using the find_x_prime method 
        from the parent class. If precomputed x_prime is not available, it will 
        calculate and return the delta.

        Use the train method to calculate x_prime and save it to disk.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: x_prime, the calculated delta.
        """
        # array of ones with the number of rows of x
        ones = torch.ones((x.shape[0], 1))
        return super().find_x_prime(x, ones)

    def _gen_z_fn(self, data: torch.Tensor) -> torch.Tensor:
        """
        Generates the z values for the GP formula. In this case, z = 1.

        Args:
            data (torch.Tensor): A tuple containing the input data and labels.

        Returns:
            torch.Tensor: A tensor of ones with the same shape as the labels.
        """
        _, y = data
        return torch.ones_like(y)
