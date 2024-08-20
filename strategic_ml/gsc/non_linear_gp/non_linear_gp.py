"""_summary
"""

# External imports
import torch
from torch import nn
import torch.optim as optim
from typing import Optional, Dict, Any
import logging

# Internal imports
from strategic_ml.cost_functions.cost_function import _CostFunction
from strategic_ml.gsc.generalized_strategic_delta import _GSC


class _NonLinearGP(_GSC):
    def __init__(
        self,
        cost: _CostFunction,
        strategic_model: nn.Module,
        cost_weight: float = 1,
        *args,
        training_params: Dict[str, Any],
    ) -> None:
        super().__init__(strategic_model, cost, cost_weight)
        self.training_params: Dict[str, Any] = training_params

        logging.basicConfig(level=logging.INFO)

        self.set_training_params()

    def find_x_prime(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:

        batch_size: int = x.size(0)
        assert (
            z.size(0) == batch_size
        ), f"z should have the same size as x, but it is {z.size(0)} and {batch_size}"

        x_prime = x.clone().detach().requires_grad_(True)
        optimizer = self.optimizer_class([x_prime], **self.optimizer_params)
        scheduler = None
        if self.has_scheduler:
            scheduler = self.scheduler_class(optimizer, **self.scheduler_params)

        best_loss = torch.full((batch_size,), float("inf"), device=x.device)
        best_x_prime = x_prime.clone().detach()
        patience_counter = torch.zeros(batch_size, dtype=torch.int, device=x.device)

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()

            logits = self.strategic_model.forward(x_prime) * z.view(-1, 1)
            output = torch.tanh(logits * self.temp).squeeze()
            movement_cost = self.cost(x, x_prime).squeeze()
            loss = output - self.cost_weight * movement_cost
            loss = -loss  # Since we're maximizing, we minimize the negative loss

            with torch.no_grad():
                improved = loss < best_loss
                best_loss[improved] = loss[improved]
                best_x_prime[improved] = x_prime[improved].clone().detach()

                if self.early_stopping != -1:
                    patience_counter[~improved] += 1
                    patience_counter[improved] = 0

            loss.sum().backward()  # We sum the loss to avoid interaction between samples
            optimizer.step()
            if scheduler:
                scheduler.step()

            if self.early_stopping != -1 and patience_counter.max().item() >= self.early_stopping:
                break

        x_prime = best_x_prime
        return x_prime

    def _set_optimizer_params(self) -> None:
        self.optimizer_class = self.training_params.get("optimizer_class", optim.SGD)
        self.optimizer_params = self.training_params.get(
            "optimizer_params", {"lr": 0.01}
        )

    def _set_scheduler_params(self) -> None:
        assert self.optimizer_class is not None, "call _set_optimizer_params first"

        self.has_scheduler = False
        if "scheduler_class" in self.training_params:
            self.has_scheduler = True
            self.scheduler_class = self.training_params["scheduler_class"]
            self.scheduler_params = self.training_params.get("scheduler_params", {})

    def _set_early_stopping(self) -> None:
        self.early_stopping: int = self.training_params.get("early_stopping", -1)

    def set_training_params(self) -> None:
        assert self.training_params is not None, "training_params should not be None"
        self.temp: float = self.training_params.get("temp", 1.0)

        self.num_epochs: int = self.training_params.get("num_epochs", 100)

        self._set_optimizer_params()

        self._set_scheduler_params()

        self._set_early_stopping()

    def update_training_params(self, training_params: Dict[str, Any]) -> None:
        self.training_params.update(training_params)
        self.set_training_params()
