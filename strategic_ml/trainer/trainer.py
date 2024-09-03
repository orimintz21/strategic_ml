# External imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from typing import Optional, TYPE_CHECKING, Dict, Any, Union
import logging

# Internal imports
from strategic_ml.gsc import _LinearGP, _NonLinearGP, _GSC
from strategic_ml.regularization import _StrategicRegularization
from strategic_ml.loss_functions import StrategicHingeLoss


class StrategicModelSuit(pl.LightningModule):
    def __init__(
        self,
        *args,
        model: nn.Module,
        delta: _GSC,
        loss_fn: nn.Module,
        regularization: Optional[_StrategicRegularization] = None,
        train_loader: DataLoader,
        validation_loader: Optional[DataLoader] = None,
        test_loader: DataLoader,
        test_model: Optional[nn.Module] = None,
        test_delta: Optional[_GSC] = None,
        logging_level: int = logging.INFO,
        training_params: Dict[str, Any],
        **kwargs,
    ) -> None:
        super(StrategicModelSuit, self).__init__()
        self.model = model
        self.delta = delta
        self.loss_fn = loss_fn
        self.regularization = regularization
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.test_model = test_model
        self.test_delta = test_delta
        self.training_params = training_params
        logging.basicConfig(level=logging_level)

        if isinstance(self.model, _NonLinearGP):
            if "train_delta_every" not in kwargs:
                logging.warning(
                    """Using NonLinearGP without train_delta_every 
                                parameter, this means that the delta will use GD 
                                in every epoch."""
                )

                self.train_delta_every: Optional[int] = None
            else:
                # We know that the train_delta_every is in kwargs
                self.train_delta_every: Optional[int] = kwargs["train_delta_every"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        tensorboard_logs: Dict[str, Any] = {}
        if isinstance(self.loss_fn, StrategicHingeLoss):
            assert (
                self.regularization is None
            ), "Regularization is not supported for StrategicHingeLoss"
            loss: torch.Tensor = self.loss_fn.forward(x=x, y=y)
            delta_logs: Dict[str, Any] = {"delta_used": False}

        else:
            use_training_delta: bool = (isinstance(self.delta, _NonLinearGP)) and (
                self.train_delta_every is not None
            )

            if use_training_delta:
                assert isinstance(self.delta, _NonLinearGP)
                assert self.train_delta_every is not None
                if self.current_epoch % self.train_delta_every == 0:
                    self.delta.train(
                        self.train_dataloader()
                    )
                else:
                    training_logs = "Delta not trained"
                x_prime: torch.Tensor = self.delta.load_x_prime(batch_idx=batch_idx)
                delta_logs: Dict[str, Any] = {
                    "delta_loaded": True,
                }
            else:
                x_prime  = self.delta.forward(x)

            predictions = self.forward(x_prime)
            loss = self.loss_fn(predictions, y)

        tensorboard_logs["train_loss"] = loss
        self.log("train_loss", loss, on_step=True, on_epoch=True)

        output_dict = {"loss": loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        optimizer_class = self.training_params.get("optimizer_class", optim.SGD)
        optimizer_params = self.training_params.get("optimizer_params", {"lr": 0.01})
        optimizer = optimizer_class(self.model.parameters(), **optimizer_params)

        if "scheduler_class" in self.training_params:
            scheduler_class = self.training_params["scheduler_class"]
            scheduler_params = self.training_params.get("scheduler_params", {})
            scheduler = scheduler_class(optimizer, **scheduler_params)
            return [optimizer], [scheduler]

        return optimizer

    def train_dataloader(self) -> Any:
        return self.train_loader

    def val_dataloader(self) -> Any:
        return self.validation_loader

    def test_dataloader(self) -> Any:
        return self.test_loader

    def on_train_start(self):
        logging.info("Training is starting.")

    def on_epoch_start(self):
        logging.info("Starting a new epoch.")
