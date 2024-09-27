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
    The NonLinearGP is a strategic delta model that calculates the delta based on the following formula:
    delta_h(x,z) = argmax_{x' in X}(1{model(x') = z} - r/2 * (cost(x,x')))

    This class uses gradient-based optimization to find the optimal x' that minimizes the cost.
    The optimization process alternates between updating the model and the delta, and large datasets
    require saving x' to disk for later retrieval.

    """

    def __init__(
        self,
        cost: _CostFunction,
        strategic_model: nn.Module,
        cost_weight: float = 1,
        save_dir: str = ".",
        logging_level: int = logging.INFO,
        *args,
        training_params: Dict[str, Any],
    ) -> None:
        """
        Initializes the NonLinearGP model.

        Args:
            cost (_CostFunction): The cost function of the delta.
            strategic_model (nn.Module): The model used for strategic classification.
            cost_weight (float, optional): The weight of the cost function. Defaults to 1.
            save_dir (str, optional): Directory to save the computed x_prime values. Defaults to ".".
            logging_level (int, optional): Logging level for the model. Defaults to logging.INFO.
            training_params (Dict[str, Any]): Dictionary containing training parameters:
                - optimizer_class: The optimizer class (default: SGD)
                - optimizer_params: Parameters for the optimizer (default: {"lr": 0.01})
                - scheduler_class: (optional) The scheduler class for learning rate adjustment
                - scheduler_params: Parameters for the scheduler (default: {})
                - early_stopping: Epochs to wait before early stopping (default: -1, no early stopping)
                - num_epochs: Number of epochs (default: 100)
                - temp: Temperature for the tanh function (default: 1.0)
        """
        super().__init__(strategic_model, cost, cost_weight)
        self.training_params: Dict[str, Any] = training_params

        logging.basicConfig(level=logging_level)
        self.save_dir: str = save_dir

        self.set_training_params()
        self.current_x_prime: Optional[torch.Tensor] = None

    def set_mapping(
        self,
        data: DataLoader,
        set_name: str = "",
    ) -> None:
        """
        Trains the model by computing x_prime for all data and saving the results to disk.

        Args:
            data (DataLoader): DataLoader containing input data.
            set_name (str, optional): Identifier for the dataset. Defaults to "".
        """
        # Add the set_name to the save_dir
        for batch_idx, data_batch in enumerate(data):
            x, y = data_batch
            self.set_mapping_for_batch(x, y, batch_idx, set_name)

    def set_mapping_for_batch(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        batch_idx: int,
        set_name: str = "",
    ) -> None:
        """
        Computes x_prime for a specific batch and saves the results to disk.

        Args:
            x_batch (torch.Tensor): Input data for the batch.
            y_batch (torch.Tensor): Labels for the batch.
            batch_idx (int): Index of the batch.
            set_name (str, optional): Identifier for the dataset. Defaults to "".
        """
        save_dir: str = os.path.join(self.save_dir, set_name)
        os.makedirs(save_dir, exist_ok=True)

        z_batch: torch.Tensor = self._gen_z_fn(x_batch, y_batch)
        x_prime: torch.Tensor = self.find_x_prime(x_batch, z_batch)
        save_path: str = os.path.join(save_dir, f"x_prime_batch_{batch_idx}.pt")
        torch.save(x_prime, save_path)
        logging.debug(f"Saved x_prime for batch {batch_idx} to {save_path}")

    def load_x_prime(
        self, batch_idx: int, set_name: str = "", device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Loads precomputed x_prime values from disk for a specific batch.

        Args:
            batch_idx (int): Index of the batch to load.
            set_name (str, optional): Identifier for the dataset. Defaults to "".
            device (torch.device, optional): Device to load the tensor to. Defaults to GPU if available, else CPU.

        Returns:
            torch.Tensor: The loaded x_prime tensor.
        """
        save_dir: str = os.path.join(self.save_dir, set_name)

        save_path: str = os.path.join(save_dir, f"x_prime_batch_{batch_idx}.pt")
        if device is None:
            # Load to gpu if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if os.path.exists(save_path):
            x_prime: torch.Tensor = torch.load(
                save_path, weights_only=True, map_location=device
            )
            logging.debug(f"Loaded x_prime for batch {batch_idx} from {save_path}")
            return x_prime
        else:
            raise FileNotFoundError(f"No saved x_prime found at {save_path}")

    def find_x_prime(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Finds the optimal x' based on the given x and z using gradient-based optimization.

        Args:
            x (torch.Tensor): Input data.
            z (torch.Tensor): Labels or target outputs.

        Returns:
            torch.Tensor: The optimized x_prime tensor.
        """

        assert x.device == z.device, "x and z should be on the same device"
        assert x.size(0) == z.size(0), "x and z should have the same batch size"
        device = x.device
        dtype = x.dtype
        with torch.enable_grad():
            batch_size: int = x.size(0)
            x_prime: torch.Tensor = x.clone().detach().to(x.device).requires_grad_(True)
            optimizer: optim.Optimizer = self.optimizer_class(
                [x_prime], **self.optimizer_params
            )
            scheduler: Optional[Any] = None
            if self.has_scheduler:
                scheduler = self.scheduler_class(optimizer, **self.scheduler_params)

            best_loss: torch.Tensor = torch.full(
                (batch_size,), float("inf"), device=device, dtype=dtype
            )
            best_x_prime: torch.Tensor = x_prime.clone().detach().to(device)
            patience_counter: torch.Tensor = torch.zeros(
                batch_size, dtype=torch.int, device=device
            )

            for epoch in range(self.num_epochs):
                optimizer.zero_grad()

                logits: torch.Tensor = self.strategic_model.forward(x_prime) * z.view(
                    -1, 1
                )
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

                loss.mean().backward()
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
        Sets the training parameters for the optimization process.
        """
        assert self.training_params is not None, "training_params should not be None"
        self.temp: float = self.training_params.get("temp", 1.0)

        self.num_epochs: int = self.training_params.get("num_epochs", 100)

        self._set_optimizer_params()

        self._set_scheduler_params()

        self._set_early_stopping()

    def update_training_params(self, training_params: Dict[str, Any]) -> None:
        """
        Updates the training parameters for the optimization.

        Args:
            training_params (Dict[str, Any]): Dictionary containing updated training parameters.
        """
        self.training_params.update(training_params)
        self.set_training_params()

    def _gen_z_fn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to generate the z values for the optimization.

        Returns:
            torch.Tensor: Generated z values for the optimization.
        """
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )
