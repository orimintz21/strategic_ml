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
    This is the NonLinearAdvDelta model. This model does not have any
    assumptions on the model or the cost function. The model uses the non-linear
    gp model to calculate the delta for the adversarial users.
    The delta is calculated by the following formula:
    x_prime = argmax_{x' in X}(1{model(x') = -y} - r/2 * (cost(x,x')))
    (i.e. z = -y)
    By using the gradient of the model, we can find the x' that will be close to
    the optimal x'.
    We don't want to run the optimization for epoch of the model, so we optimize
    the delta and the model alternately. Note that the number of samples
    could be large, so we need to write x' to the disk and load it when needed.
    For more information, see _NonLinearGP class.

    Parent Class: _NonLinearGP
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
        """Initializer for the  class.

        Args:
            cost (_CostFunction): The cost function of the delta.
            strategic_model (nn.Module): The strategic model that the delta is calculated on.
            cost_weight (float, optional): The weight of the cost function. Defaults to 1.
            save_dir (str): Directory to save the computed x_prime values
            training_params (Dict[str, Any]): A dictionary that contains the training parameters.

            expected keys:
                - optimizer_class: The optimizer class that will be used for the optimization. (default: SGD)
                - optimizer_params: The parameters for the optimizer. (default: {"lr": 0.01})
                - scheduler_class: The scheduler class that will be used for the optimization. (optional)
                - scheduler_params: The parameters for the scheduler. (default: {})
                - early_stopping: The number of epochs to wait before stopping the optimization. (default: -1, i.e. no early stopping)
                - num_epochs: The number of epochs for the optimization. (default: 100)
                - temp: The temperature for the tanh function for the model. (default: 1.0)
        """
        super(NonLinearAdvDelta, self).__init__(
            cost,
            strategic_model,
            cost_weight,
            save_dir=save_dir,
            training_params=training_params,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """This function calculates the delta for the adversarial users.
        It uses the find_x_prime method from the parent class with the label -y.
        If precomputed x_prime is not available, it will calculate it.
        Use the train method to calculate x_prime and save it to the disk.


        Args:
            x (torch.Tensor): The data.
            y (torch.Tensor): The label.

        Returns:
            torch.Tensor: x_prime, the delta.
        """
        y = y.to(x.device)
        z = -y
        return super().find_x_prime(x, z)

    def _gen_z_fn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = y.to(x.device)
        return -y
