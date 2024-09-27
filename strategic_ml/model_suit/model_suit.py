# External imports
from enum import Enum
from typing import Optional, Dict, Any, Tuple, List, Union
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Internal imports
from strategic_ml.gsc import _LinearGP, _NonLinearGP, _GSC, IdentityDelta
from strategic_ml.models import LinearModel, _LinearRegularization
from strategic_ml.regularization import _StrategicRegularization
from strategic_ml.loss_functions import StrategicHingeLoss


class ModelSuit(pl.LightningModule):
    """
    A PyTorch Lightning module that integrates various components of the strategic_ml library to train, validate,
    and test machine learning models within the context of strategic classification. This module handles the
    training process, including strategic deltas, loss functions, regularization, and logging.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        delta: _GSC,
        loss_fn: nn.Module,
        regularization: Optional[_StrategicRegularization] = None,
        regularization_weight: float = 0.0,
        linear_regularization: Optional[List[_LinearRegularization]] = None,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        test_loader: DataLoader,
        test_delta: Optional[_GSC] = None,
        train_delta_every: Optional[int] = None,
        training_params: Dict[str, Any],
    ) -> None:
        """
        Initializes the ModelSuit class with the specified model, delta, loss function, regularization,
        and other configurations.

        Args:
            model (nn.Module): The main model to be trained, validated, and tested.
            delta (_GSC): The strategic delta model, responsible for modifying the input data based on strategic behavior.
            loss_fn (nn.Module): The loss function to optimize. If this is not
            a StrategicHingeLoss, left first argument of the loss function
            should be the predictions and the second should be the true labels.
            regularization (Optional[_StrategicRegularization]): A strategic regularization method. Default is None.
            regularization_weight (float): The weight of the strategic regularization. Default is 0.
            linear_regularization (Optional[List[_LinearRegularization]]): A list of linear regularization methods. Default is None.
            train_loader (DataLoader): DataLoader for the training data.
            validation_loader (DataLoader): DataLoader for the validation data.
            test_loader (DataLoader): DataLoader for the test data.
            test_delta (Optional[_GSC]): An optional strategic delta for testing purposes. Default is None.
            train_delta_every (Optional[int]): Frequency of training the delta during the training process. Default is None.
            training_params (Dict[str, Any]): A dictionary of parameters for configuring the training process.
                If you want to use the default values, pass an empty dictionary. Parameters include:
                - optimizer_class (optim.Optimizer): The optimizer class to use for training. Default is optim.SGD.
                - optimizer_params (Dict[str, Any]): The parameters for the optimizer. Default is {"lr": 0.01}.
                - scheduler_class (Optional[Any]): The learning rate scheduler class. Default is None (no scheduler).
                - scheduler_params (Dict[str, Any]): The parameters for the learning rate scheduler (if applicable).
        """
        super(ModelSuit, self).__init__()
        self.model = model
        self.delta = delta
        self.loss_fn = loss_fn
        self.regularization = regularization
        self.regularization_weight = regularization_weight
        self.linear_regularization = linear_regularization
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.test_delta = test_delta
        self.training_params = training_params
        self.train_delta_every: Optional[int] = train_delta_every
        if isinstance(self.delta, _NonLinearGP) and self.train_delta_every is None:
            logging.warning(
                """Using NonLinearGP without train_delta_every 
                            parameter, this means that the delta will use GD 
                            in every epoch."""
            )
            self.train_delta_every = 1

        if isinstance(self.delta, _LinearGP):
            assert isinstance(self.model, LinearModel)
            if self.train_delta_every is not None:
                logging.warning(
                    "Linear delta is used, there is no need for train_delta_every parameter"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the model.
        """
        return self.model(x)

    class _Mode(Enum):
        """
        Enum class representing the mode of operation: TRAIN, VALIDATION, or TEST.
        """

        TRAIN = "train"
        VALIDATION = "validation"
        TEST = "test"

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step, including loss calculation, logging, and metrics computation.

        Args:
            batch: A batch of data from the training DataLoader.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The computed loss for the current batch.
        """
        x, y = batch
        loss, predictions = self._calculate_loss_and_predictions(
            x=x, y=y, batch_idx=batch_idx, mode=self._Mode.TRAIN
        )

        loss = loss.mean()
        zero_one_loss = (torch.sign(self.forward(x)) != y).sum().item() / len(y)

        # Log metrics
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_zero_one_loss",
            zero_one_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """
        Performs a single validation step, including loss calculation, logging, and metrics computation.

        Args:
            batch: A batch of data from the validation DataLoader.
            batch_idx (int): The index of the current batch.

        Returns:
            dict: A dictionary containing the validation loss and zero-one loss.
        """
        x, y = batch
        val_loss, predictions = self._calculate_loss_and_predictions(
            x=x, y=y, batch_idx=batch_idx, mode=self._Mode.VALIDATION
        )
        assert predictions is not None

        zero_one_loss = (torch.sign(predictions) != y).sum().item() / len(y)

        self.log(
            "val_loss",
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_zero_one_loss",
            zero_one_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """
        Performs a single test step, including loss calculation, logging, and metrics computation.

        Note: If the delta that is used in the test phase is a NonLinearGP,
        you need to train the delta for the test set using the train_delta_for_test method.

        Args:
            batch: A batch of data from the test DataLoader.
            batch_idx (int): The index of the current batch.

        Returns:
            dict: A dictionary containing the test loss and zero-one loss.
        """
        # Enable gradient computation during test_step
        with torch.enable_grad():
            # Log the structure of the batch
            logging.debug(f"Test batch contents: {len(batch)} elements")

            x, y = batch
            device = x.device
            if self.test_delta is not None:
                # We are testing an In The Dark scenario
                if isinstance(self.test_delta, _NonLinearGP):
                    x_prime = self.test_delta.load_x_prime(
                        batch_idx=batch_idx,
                        device=device,
                        set_name=self._Mode.TEST.value,
                    )
                elif isinstance(self.test_delta, IdentityDelta):
                    x_prime = x
                else:
                    x_prime = self.test_delta.forward(x, y).to(device)

                assert (
                    x_prime.device == device
                ), f"x_prime is on {x_prime.device} device, but should be on {device} device"
                predictions = self.model.forward(x_prime)
                test_loss = self.loss_fn(predictions, y)

            else:
                test_loss, predictions = self._calculate_loss_and_predictions(
                    x=x, y=y, batch_idx=batch_idx, mode=self._Mode.TEST
                )

            assert predictions is not None
            zero_one_loss = (torch.sign(predictions) != y).sum().item() / len(y)

            # Log the test loss
            self.log("test_loss", test_loss, on_step=True, on_epoch=True, logger=True)

            # Log the zero-one loss for the test set
            self.log(
                "test_zero_one_loss",
                zero_one_loss,
                on_step=True,
                on_epoch=True,
                logger=True,
            )

    def _calculate_loss_and_predictions(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_idx: int,
        mode: _Mode,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Calculates the loss and predictions for a given batch.

        Args:
            x (torch.Tensor): The input data.
            y (torch.Tensor): The true labels.
            batch_idx (int): The index of the current batch.
            mode (_Mode): The mode of operation (TRAIN, VALIDATION, TEST).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: The loss and predictions.
        """

        device = x.device

        assert (
            y.device == device
        ), f"y ({y.device}) should be on the same device as x ({device})"

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
            if isinstance(self.delta, _NonLinearGP):
                # Now we have two option, or we use the precomputed x_prime or we calculate it
                # and then use it.
                assert self.train_delta_every is not None
                if (mode != self._Mode.TEST) and self.train_delta_every == 1:
                    x_prime = self.delta.forward(x, y)

                else:
                    if (self.current_epoch % self.train_delta_every == 0) and (
                        mode != self._Mode.TEST
                    ):
                        self.delta.set_mapping_for_batch(
                            x_batch=x,
                            y_batch=y,
                            set_name=mode.value,
                            batch_idx=batch_idx,
                        )

                    x_prime: torch.Tensor = self.delta.load_x_prime(
                        batch_idx=batch_idx, set_name=mode.value, device=x.device
                    )

            elif isinstance(self.delta, IdentityDelta):
                x_prime = x

            else:
                x_prime = self.delta.forward(x, y)

            assert (
                x_prime.device == device
            ), f"x_prime should be on the same device as the model, but x_prime is on {x_prime.device} device and the x is on {device} device"
            predictions = self.model.forward(x_prime)

            loss = self.loss_fn(predictions, y)

            if (
                self.regularization is not None
                and mode == self._Mode.TRAIN
                and self.regularization_weight > 0
                and not isinstance(self.delta, IdentityDelta)
            ):
                assert not isinstance(self.loss_fn, StrategicHingeLoss)
                cost = self.delta.get_cost().forward(x, x_prime)

                regularization_term = self.regularization(
                    x=x,
                    delta_predictions=predictions,
                    model=self.model,
                    y=y,
                    linear_delta=self.delta,
                    cost=cost,
                )
                loss = loss + self.regularization_weight * regularization_term

        linear_regularization_term: torch.Tensor = torch.tensor(0.0, device=self.device)

        if self.linear_regularization is not None:
            assert isinstance(self.model, LinearModel)
            for reg in self.linear_regularization:
                linear_regularization_term = linear_regularization_term + reg(
                    self.model
                )

        loss = loss + linear_regularization_term

        return loss, predictions

    def train_delta_for_test(self, dataloader: Optional[DataLoader] = None) -> None:
        """
        Trains the delta model specifically for the test set.
        Use this method when you want to test the model with a non-linear delta.
        If you are using a linear delta, you do not need to train the delta for the test set.

        If you don't use this function before testing and you are using a non-linear
        delta, the test will fail.

        Args:
            dataloader (Optional[DataLoader]): The DataLoader for the test data. If None, the test DataLoader is used.

        Returns:
            None
        """
        dataloader = dataloader if dataloader is not None else self.test_dataloader()

        if self.test_delta is None:
            if not isinstance(self.delta, _NonLinearGP):
                return

            self.delta.train()
            self.delta.set_mapping(data=dataloader, set_name=self._Mode.TEST.value)

        else:
            if not isinstance(self.test_delta, _NonLinearGP):
                return

            self.test_delta.train()
            self.test_delta.set_mapping(data=dataloader, set_name=self._Mode.TEST.value)

    def configure_optimizers(self):
        """
        Configures the optimizers and learning rate schedulers for the training process.

        Returns:
            Union[Optimizer, Tuple[List[Optimizer], List[Any]]]: The optimizer(s) and optionally the scheduler(s).
        """
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
        """
        Returns the DataLoader for the training data.

        Returns:
            DataLoader: The training DataLoader.
        """
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the validation data.

        Returns:
            DataLoader: The validation DataLoader.
        """
        return self.validation_loader

    def test_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the test data.

        Returns:
            DataLoader: The test DataLoader.
        """
        return self.test_loader

    def get_dataloader(self, mode: _Mode) -> DataLoader:
        """
        Returns the appropriate DataLoader based on the mode.

        Args:
            mode (_Mode): The mode of operation (TRAIN, VALIDATION, TEST).

        Returns:
            DataLoader: The corresponding DataLoader.
        """
        if mode == self._Mode.TRAIN:
            return self.train_dataloader()
        elif mode == self._Mode.VALIDATION:
            return self.val_dataloader()
        elif mode == self._Mode.TEST:
            return self.test_dataloader()
        else:
            raise ValueError("Mode not supported")
