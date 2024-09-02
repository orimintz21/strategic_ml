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
    """The NonLinearGP is a strategic delta that is calculated by the following formula:
    delta_h(x,z) = argmax_{x' in X}(1{model(x') = z} - r/2 * (cost(x,x')))
    By using the gradient of the model, we can find the x' that will be close to
    the optimal x'.
    We don't want to run the optimization for epoch of the model, so we optimize
    the delta and the model alternately. Note that the number of samples
    could be large, so we need to write x' to the disk and load it when needed.

    Parent Class: _GSC
    """

    def __init__(
        self,
        cost: _CostFunction,
        strategic_model: nn.Module,
        cost_weight: float = 1,
        *args,
        training_params: Dict[str, Any],
    ) -> None:
        """Initializer for the NonLinearGP class.

        Args:
            cost (_CostFunction): The cost function of the delta.
            strategic_model (nn.Module): The strategic model that the delta is calculated on.
            cost_weight (float, optional): The weight of the cost function. Defaults to 1.
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
        super().__init__(strategic_model, cost, cost_weight)
        self.training_params: Dict[str, Any] = training_params

        logging.basicConfig(level=logging.INFO)

        self.set_training_params()
        self.current_x_prime: Optional[torch.Tensor] = None

    def train(
        self,
        x_loader: DataLoader,
        z_loader: DataLoader,
        save_dir: str,
    ) -> None:
        """
        Train the model by finding x_prime for all data in x_loader and z_loader,
        and save the results to disk.

        :param x_loader: DataLoader for x
        :param z_loader: DataLoader for z
        :param save_dir: Directory to save the computed x_prime values
        """
        os.makedirs(save_dir, exist_ok=True)

        for batch_idx, (x_batch, z_batch) in enumerate(zip(x_loader, z_loader)):
            x_prime: torch.Tensor = self.find_x_prime(x_batch, z_batch)
            save_path: str = os.path.join(save_dir, f"x_prime_batch_{batch_idx}.pt")
            torch.save(x_prime, save_path)
            logging.info(f"Saved x_prime for batch {batch_idx} to {save_path}")

    def load_x_prime(self, batch_idx: int, save_dir: str) -> torch.Tensor:
        """
        Load precomputed x_prime values from disk for a specific batch.

        :param batch_idx: Index of the batch to load
        :param save_dir: Directory where the x_prime values are saved
        :return: Loaded x_prime tensor
        """
        save_path: str = os.path.join(save_dir, f"x_prime_batch_{batch_idx}.pt")
        if os.path.exists(save_path):
            x_prime: torch.Tensor = torch.load(save_path)
            logging.info(f"Loaded x_prime for batch {batch_idx} from {save_path}")
            return x_prime
        else:
            raise FileNotFoundError(f"No saved x_prime found at {save_path}")

    def find_x_prime(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        batch_idx: Optional[int] = None,
        save_dir: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Modified find_x_prime to load precomputed x_prime if available.
        If precomputed x_prime is not available, compute x_prime using the GP formula
        and a GD method that was specified in the training dictionary.

        :param x: Input tensor
        :param z: Metadata tensor
        :param batch_idx: Index of the batch to load precomputed x_prime (if provided)
        :param save_dir: Directory to load precomputed x_prime (if provided)
        :return: x_prime tensor
        """
        if batch_idx is not None and save_dir is not None:
            try:
                x_prime = self.load_x_prime(batch_idx, save_dir)
                self.current_x_prime = x_prime
                return x_prime
            except FileNotFoundError:
                logging.info("No precomputed x_prime found. Computing x_prime...")

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
            movement_cost: torch.Tensor = self.cost(x, x_prime).squeeze()
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
        """Set the optimizer class and parameters for the optimization."""
        self.optimizer_class = self.training_params.get("optimizer_class", optim.SGD)
        self.optimizer_params = self.training_params.get(
            "optimizer_params", {"lr": 0.01}
        )

    def _set_scheduler_params(self) -> None:
        """Set the scheduler class and parameters for the optimization."""
        assert self.optimizer_class is not None, "call _set_optimizer_params first"

        self.has_scheduler = False
        if "scheduler_class" in self.training_params:
            self.has_scheduler = True
            self.scheduler_class = self.training_params["scheduler_class"]
            self.scheduler_params = self.training_params.get("scheduler_params", {})

    def _set_early_stopping(self) -> None:
        """Set the early stopping for the optimization."""
        self.early_stopping: int = self.training_params.get("early_stopping", -1)

    def set_training_params(self) -> None:
        """Set the training parameters for the optimization."""
        assert self.training_params is not None, "training_params should not be None"
        self.temp: float = self.training_params.get("temp", 1.0)

        self.num_epochs: int = self.training_params.get("num_epochs", 100)

        self._set_optimizer_params()

        self._set_scheduler_params()

        self._set_early_stopping()

    def update_training_params(self, training_params: Dict[str, Any]) -> None:
        """Update the training parameters for the optimization.

        Args:
            training_params (Dict[str, Any]):
            - optimizer_class: The optimizer class that will be used for the optimization. (default: SGD)
            - optimizer_params: The parameters for the optimizer. (default: {"lr": 0.01})
            - scheduler_class: The scheduler class that will be used for the optimization. (optional)
            - scheduler_params: The parameters for the scheduler. (default: {})
            - early_stopping: The number of epochs to wait before stopping the optimization. (default: -1, i.e. no early stopping)
            - num_epochs: The number of epochs for the optimization. (default: 100)
            - temp: The temperature for the tanh function for the model. (default: 1.0)
        """
        self.training_params.update(training_params)
        self.set_training_params()
