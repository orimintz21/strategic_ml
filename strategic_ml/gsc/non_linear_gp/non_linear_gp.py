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

        x_prime: Optional[torch.Tensor] = None
        for x_sample, z_sample in zip(x, z):
            x_sample = x_sample
            assert z_sample in [-1, 1], "z should be a label but it is {}".format(
                z_sample
            )

            x_prime_sample = x_sample.clone().detach().requires_grad_(True)
            optimizer = self.optimizer_class(
                self.strategic_model.parameters(), **self.optimizer_params
            )
            scheduler = None
            if self.has_scheduler:
                scheduler = self.scheduler_class(optimizer, **self.scheduler_params)

            best_loss: float = float("inf")
            patience_counter: int = 0

            for epoch in range(self.num_epochs):
                optimizer.zero_grad()
                logits = self.strategic_model.forward(x_prime_sample) * z_sample
                output = torch.tanh(logits * self.temp)
                movement_cost = self.cost(x_sample, x_prime_sample)
                loss = output - self.cost_weight * movement_cost
                loss = -loss

                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0

                else:
                    if self.early_stopping == -1:
                        continue

                    patience_counter += 1
                    if patience_counter >= self.early_stopping:
                        logging.info(
                            "Early stopping triggered. epoch: {}".format(epoch)
                        )
                        break

            if x_prime is None:
                x_prime = x_prime_sample.unsqueeze(0)
            else:
                x_prime = torch.cat([x_prime, x_prime_sample.unsqueeze(0)], dim=0)

        assert (
            x_prime is not None
        ), "x_prime should not be None in the end of the function"

        return x_prime

    def _set_optimizer_params(self) -> None:
        self.optimizer_class = getattr(
            optim, self.training_params.get("optimizer", "Adam")
        )
        self.optimizer_params = self.training_params.get(
            "optimizer_params", {"lr": 0.01}
        )

    def _set_scheduler_params(self) -> None:
        assert self.optimizer_class is not None, "call _set_optimizer_params first"

        self.has_scheduler = False
        if "scheduler" in self.training_params:
            self.has_scheduler = True
            self.scheduler_class = getattr(
                optim.lr_scheduler, self.training_params["scheduler"]
            )
            self.scheduler_params = self.training_params.get("scheduler_params", {})

    def _set_early_stopping(self) -> None:
        self.early_stopping: int = self.training_params.get("early_stopping", -1)

    def set_training_params(self) -> None:
        assert self.training_params is not None, "training_params should not be None"
        assert "temp" in self.training_params, "temp should be in the training_params"
        self.temp: float = self.training_params["temp"]

        self.num_epochs: int = self.training_params.get("num_epochs", 100)

        self._set_optimizer_params()

        self._set_scheduler_params()

        self._set_early_stopping()

    def update_training_params(self, training_params: Dict[str, Any]) -> None:
        self.training_params.update(training_params)
        self.set_training_params()
