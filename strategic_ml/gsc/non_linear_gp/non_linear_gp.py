"""
This module implements the _NonLinearGP model for non-linear strategic classification.

The NonLinearGP model calculates delta using a non-linear GP formula. It alternates optimization between 
the delta and the model to minimize cost and maximize adversarial outcomes. The model supports the storage 
of precomputed delta values (x_prime) to disk to handle large datasets efficiently. 

The delta is calculated using the formula:
    delta_h(x,z) = argmax_{x' in X}(1{model(x') = z} - r/2 * (cost(x, x')))

Parent Class:
    _GSC
""" 

# External imports
import os
import torch
from torch import nn
import torch.optim as optim
from typing import Optional, Dict, Any
import logging
from torch.utils.data import DataLoader

# Internal imports
from strategic_ml.cost_functions.cost_function import _CostFunction
from strategic_ml.gsc.generalized_strategic_delta import _GSC


class _NonLinearGP(_GSC):
    """
    The NonLinearGP model calculates delta using a non-linear GP formula for strategic classification.

    The delta is optimized alternately with the model. For large datasets, precomputed delta values (x_prime) 
    can be saved to disk and loaded as needed, ensuring scalability. 

    Parent Class:
        _GSC
    """

    def __init__(
        self,
        cost: _CostFunction,
        strategic_model: nn.Module,
        cost_weight: float = 1,
        save_dir: str = ".",
        *args,
        training_params: Dict[str, Any],
    ) -> None:
        """
        Initializes the NonLinearGP model for strategic classification.

        Args:
            cost (_CostFunction): The cost function for delta calculation.
            strategic_model (nn.Module): The non-linear strategic model for calculating the delta.
            cost_weight (float, optional): The weight of the cost function. Defaults to 1.0.
            save_dir (str): Directory where computed delta values (x_prime) will be saved. Defaults to current directory.
            training_params (Dict[str, Any]): A dictionary containing the training parameters.

        Training parameters include:
            - optimizer_class (type): The optimizer class for optimization (default: SGD).
            - optimizer_params (Dict[str, Any]): Parameters for the optimizer (default: {"lr": 0.01}).
            - scheduler_class (type, optional): The scheduler class for optimization.
            - scheduler_params (Dict[str, Any]): Parameters for the scheduler (default: {}).
            - early_stopping (int, optional): Number of epochs to wait for early stopping (default: -1, i.e., no early stopping).
            - num_epochs (int, optional): Number of epochs for optimization (default: 100).
            - temp (float, optional): The temperature for the tanh function (default: 1.0).
        """
        super().__init__(strategic_model, cost, cost_weight)
        self.training_params: Dict[str, Any] = training_params

        logging.basicConfig(level=logging.DEBUG)
        self.save_dir: str = save_dir

        self.set_training_params()
        self.current_x_prime: Optional[torch.Tensor] = None

    def train(
        self,
        data: DataLoader,
    ) -> None:
        """
        Trains the model by finding x_prime for all data in the DataLoader and saves the results to disk.

        Args:
            data (DataLoader): DataLoader containing x and y tensors.
        """
        os.makedirs(self.save_dir, exist_ok=True)

        for batch_idx, data_batch in enumerate(data):
            z_batch: torch.Tensor = self._gen_z_fn(data_batch)
            x_batch, _ = data_batch
            x_prime: torch.Tensor = self.find_x_prime(x_batch, z_batch)
            save_path: str = os.path.join(
                self.save_dir, f"x_prime_batch_{batch_idx}.pt"
            )
            torch.save(x_prime, save_path)
            logging.debug(f"Saved x_prime for batch {batch_idx} to {save_path}")

    def load_x_prime(self, batch_idx: int) -> torch.Tensor:
        """
        Loads precomputed x_prime values from disk for a specific batch.

        Args:
            batch_idx (int): Index of the batch to load.

        Returns:
            torch.Tensor: Loaded x_prime tensor.
        """

        save_path: str = os.path.join(self.save_dir, f"x_prime_batch_{batch_idx}.pt")
        if os.path.exists(save_path):
            x_prime: torch.Tensor = torch.load(save_path)
            logging.debug(f"Loaded x_prime for batch {batch_idx} from {save_path}")
            return x_prime
        else:
            raise FileNotFoundError(f"No saved x_prime found at {save_path}")

    def find_x_prime(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates or loads the precomputed delta (x_prime). If precomputed x_prime is not available, 
        computes x_prime using the GP formula and gradient descent (GD).

        Args:
            x (torch.Tensor): Input tensor.
            z (torch.Tensor): Metadata tensor.

        Returns:
            torch.Tensor: Calculated x_prime tensor.
        """

        batch_size: int = x.size(0)
        assert (
            z.size(0) == batch_size
        ), f"z should have the same size as x, but it is {z.size(0)} and {batch_size}"

        x_prime: torch.Tensor = x.clone().detach().requires_grad_(True)
        optimizer: optim.Optimizer = self.optimizer_class(
            [x_prime], **self.optimizer_params
        )
        scheduler: Optional[Any] = None
        if self.has_scheduler:
            scheduler = self.scheduler_class(optimizer, **self.scheduler_params)

        best_loss: torch.Tensor = torch.full(
            (batch_size,), float("inf"), device=x.device
        )
        best_x_prime: torch.Tensor = x_prime.clone().detach()
        patience_counter: torch.Tensor = torch.zeros(
            batch_size, dtype=torch.int, device=x.device
        )

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()

            logits: torch.Tensor = self.strategic_model.forward(x_prime) * z.view(-1, 1)
            output: torch.Tensor = torch.tanh(logits * self.temp).squeeze()
            movement_cost: torch.Tensor = self.cost(x, x_prime)
            if batch_size != 1:
                movement_cost = movement_cost.squeeze()

            loss: torch.Tensor = output - self.cost_weight * movement_cost
            loss = -loss  # Since we're maximizing, we minimize the negative loss

            with torch.no_grad():
                improved: torch.Tensor = loss < best_loss
                best_loss[improved] = loss[improved]
                best_x_prime[improved] = x_prime[improved].clone().detach()

                if self.early_stopping != -1:
                    patience_counter[~improved] += 1
                    patience_counter[improved] = 0

            loss.mean().backward()  # We sum the loss to avoid interaction between samples
            optimizer.step()
            if scheduler:
                scheduler.step()

            if (
                self.early_stopping != -1
                and patience_counter.max().item() >= self.early_stopping
            ):
                break

        x_prime = best_x_prime
        self.current_x_prime = x_prime

        return x_prime

    def _set_optimizer_params(self) -> None:
        """
        Set the optimizer class and parameters for the optimization.
        """
        self.optimizer_class = self.training_params.get("optimizer_class", optim.SGD)
        self.optimizer_params = self.training_params.get(
            "optimizer_params", {"lr": 0.01}
        )

    def _set_scheduler_params(self) -> None:
        """
        Set the scheduler class and parameters for the optimization.
        """
        assert self.optimizer_class is not None, "call _set_optimizer_params first"

        self.has_scheduler = False
        if "scheduler_class" in self.training_params:
            self.has_scheduler = True
            self.scheduler_class = self.training_params["scheduler_class"]
            self.scheduler_params = self.training_params.get("scheduler_params", {})

    def _set_early_stopping(self) -> None:
        """
        Set the early stopping for the optimization.
        """
        self.early_stopping: int = self.training_params.get("early_stopping", -1)

    def set_training_params(self) -> None:
        """
        Set the training parameters for the optimization.
        """
        assert self.training_params is not None, "training_params should not be None"
        self.temp: float = self.training_params.get("temp", 1.0)

        self.num_epochs: int = self.training_params.get("num_epochs", 100)

        self._set_optimizer_params()

        self._set_scheduler_params()

        self._set_early_stopping()

    def update_training_params(self, training_params: Dict[str, Any]) -> None:
        """"Updates the training parameters for optimization.

        Args:
            training_params (Dict[str, Any]): Updated dictionary of training parameters.
        """
        self.training_params.update(training_params)
        self.set_training_params()

    def _gen_z_fn(self, data: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to generate the z values for the optimization
        """
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )
