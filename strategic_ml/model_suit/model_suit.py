# External imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Optional, TYPE_CHECKING, Dict, Any, Tuple
import logging
from enum import Enum

# Internal imports
from strategic_ml.gsc import _LinearGP, _NonLinearGP, _GSC
from strategic_ml.regularization import _StrategicRegularization
from strategic_ml.loss_functions import StrategicHingeLoss


class ModelSuit(pl.LightningModule):
    def __init__(
        self,
        *args,
        model: nn.Module,
        delta: _GSC,
        loss_fn: nn.Module,
        regularization: Optional[_StrategicRegularization] = None,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        test_loader: DataLoader,
        test_delta: Optional[_GSC] = None,
        logging_level: int = logging.INFO,
        training_params: Dict[str, Any],
        **kwargs,
    ) -> None:
        super(ModelSuit, self).__init__()
        self.model = model
        self.delta = delta
        self.loss_fn = loss_fn
        self.regularization = regularization
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
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

    class _Mode(Enum):
        TRAIN = "train"
        VALIDATION = "validation"
        TEST = "test"

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, _ = self._calculate_loss_and_predictions(
            x,
            y,
            batch_idx,
            mode=self._Mode.TRAIN,
        )

        loss = loss.mean()
        zero_one_loss = (torch.sign(self.forward(x)) != y).sum().item() / len(y)
        # Log metrics
        self.log('train_loss_epoch', loss, on_epoch=True, prog_bar=True)
        self.log('zero_one_loss_epoch', zero_one_loss, on_epoch=True, prog_bar=True)

        # print(f"Logging train loss: {loss.item()}")

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        val_loss, predictions = self._calculate_loss_and_predictions(
            x,
            y,
            batch_idx,
            mode=self._Mode.VALIDATION,
        )
        assert predictions is not None

        # Log the validation loss
        self.log("val_loss", val_loss, on_step=True, on_epoch=True)
        
        zero_one_loss = (torch.sign(predictions) != y).sum().item() / len(y)
        
        # Log the zero-one loss (accuracy metric)
        self.log("val_zero_one_loss", zero_one_loss, on_step=True, on_epoch=True)

        # print(f"Logging val loss: {val_loss.item()}, zero-one loss: {zero_one_loss}")

        return {"val_loss": val_loss, "zero_one_loss": zero_one_loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        if self.test_delta is not None:
            # In the dark
            x_prime = self.test_delta.forward(x, y)
            predictions = self.forward(x_prime)
            test_loss = self.loss_fn(predictions, y)
        else:
            test_loss, predictions = self._calculate_loss_and_predictions(
                x,
                y,
                batch_idx,
                mode=self._Mode.TEST,
            )
            assert predictions is not None

        # Log the test loss
        self.log("test_loss", test_loss, on_step=True, on_epoch=True)
        
        zero_one_loss = (torch.sign(predictions) != y).sum().item() / len(y)
        
        # Log the zero-one loss for the test set
        self.log("test_zero_one_loss", zero_one_loss, on_step=True, on_epoch=True)

        return {"test_loss": test_loss, "zero_one_loss": zero_one_loss}

    def _calculate_loss_and_predictions(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_idx: int,
        mode: _Mode,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(self.loss_fn, StrategicHingeLoss):
            assert (
                self.regularization is None
            ), "Regularization is not supported for StrategicHingeLoss"
            loss: torch.Tensor = self.loss_fn.forward(x=x, y=y)
            delta_logs: Dict[str, Any] = {"delta_used": False}
            if mode != self._Mode.TRAIN:
                # We know that the model is a linear model (this is an assumption for the StrategicHingeLoss)
                x_prime = self.delta.forward(x, y)
                # Used for the zero one loss
                predictions = self.forward(x)
            else:
                predictions = None

        else:
            use_training_delta: bool = (isinstance(self.delta, _NonLinearGP)) and (
                self.train_delta_every is not None
            )

            if use_training_delta and (not mode == self._Mode.TEST):
                assert isinstance(self.delta, _NonLinearGP)
                assert self.train_delta_every is not None
                if self.current_epoch % self.train_delta_every == 0:
                    if mode == self._Mode.TRAIN:
                        self.delta.train(self.train_dataloader())
                    elif mode == self._Mode.VALIDATION:
                        self.delta.train(self.val_dataloader())
                    else:
                        raise ValueError("Do not train the delta in test mode")

                x_prime: torch.Tensor = self.delta.load_x_prime(batch_idx=batch_idx)

            else:
                if mode == self._Mode.TEST and self.test_delta is not None:
                    x_prime = self.test_delta.forward(x, y)
                else:
                    x_prime = self.delta.forward(x, y)

            predictions = self.forward(x_prime)
            loss = self.loss_fn(predictions, y)
            if self.regularization is not None and mode == self._Mode.TRAIN:
                assert not isinstance(self.delta, _NonLinearGP)
                assert not isinstance(self.loss_fn, StrategicHingeLoss)
                regularization_term = self.regularization(
                    x=x,
                    delta_predictions=predictions,
                    model=self.model,
                    y=y,
                    linear_delta=self.delta,
                )
                loss += regularization_term

        return loss, predictions

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


    def train_dataloader(self) -> DataLoader:
        return self.train_loader


    def val_dataloader(self) -> DataLoader:
        return self.validation_loader


    def test_dataloader(self) -> DataLoader:
        return self.test_loader