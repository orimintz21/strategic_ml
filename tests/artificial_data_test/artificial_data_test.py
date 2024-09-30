# External imports
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from typing import Optional, Dict, Any, Tuple
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import unittest
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping

# internal imports
from strategic_ml import (
    ModelSuit,
    LinearModel,
    _GSC,
    LinearStrategicDelta,
    NonLinearStrategicDelta,
    CostNormL2,
    SocialBurden,
    visualize_data_and_delta_2D,
    visualize_data_and_delta_1D,
    IdentityDelta,
    LinearL2Regularization,
    StrategicHingeLoss,
)

from .gen_data_loader import gen_custom_normal_data

DUMMY_RUN = False
GPUS = 1
ACCELERATOR = "auto"
print("DUMMY_RUN: ", DUMMY_RUN)
print(f"CUDA Available: {torch.cuda.is_available()}")


class BCEWithLogitsLossPNOne(nn.Module):
    def __init__(self):
        super(BCEWithLogitsLossPNOne, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        target = (target + 1) / 2
        input = (input + 1) / 2
        return self.loss(input, target)


DELTA_TRAINING_PARAMS: Dict[str, Any] = {
    "num_epochs": 100,
    "optimizer_class": optim.Adam,
    "optimizer_params": {
        "lr": 0.01,
    },
    "early_stopping": 60,
    "temp": 20,
}

NON_LINEAR_TRAINING_PARAMS: Dict[str, Any] = {
    "optimizer_class": optim.Adam,
    "optimizer_params": {
        "lr": 0.001,
    },
    "num_epochs": 100,
    "early_stopping": 60,
}

LINEAR_TRAINING_PARAMS: Dict[str, Any] = {
    "optimizer_class": optim.Adam,
    "optimizer_params": {
        "lr": 0.01,
    },
}


VERBOSE = True
SEED = 0


def reset_seed():
    torch.manual_seed(SEED)


torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
    train_size = 500000
    val_size = 10000
    test_size = 10000

    blobs_dist = 15
    blobs_std = 4
    blobs_x2_std = 3
    pos_noise_frac = 0.1
    neg_noise_frac = 0.1
    num_epochs = 10
else:
    train_size = 5000
    val_size = 1000
    test_size = 1000

    blobs_dist = 11
    blobs_std = 1.5
    blobs_x2_std = 1.5
    pos_noise_frac = 0.0
    neg_noise_frac = 0.0
    num_epochs = 100

DEFAULT_POS_MEAN_2D = np.array([blobs_dist / 2 + 10, 0])
DEFAULT_POS_STD_2D = np.array([blobs_std, blobs_x2_std])
DEFAULT_NEG_MEAN_2D = np.array([-blobs_dist / 2 + 10, 0])
DEFAULT_NEG_STD_2D = np.array([blobs_std, blobs_x2_std])
DEFAULT_POS_MEAN_1D = blobs_dist / 2 + 10
DEFAULT_POS_STD_1D = blobs_std
DEFAULT_NEG_MEAN_1D = -blobs_dist / 2 + 10
DEFAULT_NEG_STD_1D = blobs_std


def print_if_verbose(message: str) -> None:
    global VERBOSE
    if VERBOSE:
        print(message)


def visualize_train_and_test_2D(
    model: Optional[LinearModel],
    data_loader_train,
    data_loader_test,
    delta: _GSC,
    grid_size=100,
    display_percentage_train=1.0,
    display_percentage_test=1.0,
    prefix="",
):
    visualize_data_and_delta_2D(
        model,
        data_loader_train,
        delta,
        grid_size=grid_size,
        display_percentage=display_percentage_train,
        prefix="train" + prefix,
    )
    visualize_data_and_delta_2D(
        model,
        data_loader_test,
        delta,
        grid_size=grid_size,
        display_percentage=display_percentage_test,
        prefix="test" + prefix,
    )


class NonLinearModel(torch.nn.Module):
    def __init__(self, x_dim: int) -> None:
        super(NonLinearModel, self).__init__()
        self.linear1 = torch.nn.Linear(x_dim, 5)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(5, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out


class TestModelSuit(unittest.TestCase):

    def setUp(self):
        self.train_dataLoader = gen_custom_normal_data(
            num_samples=train_size,
            x_dim=2,
            pos_mean=DEFAULT_POS_MEAN_2D,
            pos_std=DEFAULT_POS_STD_2D,
            neg_mean=DEFAULT_NEG_MEAN_2D,
            neg_std=DEFAULT_NEG_STD_2D,
            pos_noise_frac=pos_noise_frac,
            neg_noise_frac=neg_noise_frac,
        )

        self.val_dataLoader = gen_custom_normal_data(
            num_samples=val_size,
            x_dim=2,
            pos_mean=DEFAULT_POS_MEAN_2D,
            pos_std=DEFAULT_POS_STD_2D,
            neg_mean=DEFAULT_NEG_MEAN_2D,
            neg_std=DEFAULT_NEG_STD_2D,
            pos_noise_frac=pos_noise_frac,
            neg_noise_frac=neg_noise_frac,
        )

        self.test_dataLoader = gen_custom_normal_data(
            num_samples=test_size,
            x_dim=2,
            pos_mean=DEFAULT_POS_MEAN_2D,
            pos_std=DEFAULT_POS_STD_2D,
            neg_mean=DEFAULT_NEG_MEAN_2D,
            neg_std=DEFAULT_NEG_STD_2D,
            pos_noise_frac=pos_noise_frac,
            neg_noise_frac=neg_noise_frac,
        )

        self.train_dataLoader_one_dim = gen_custom_normal_data(
            num_samples=train_size,
            x_dim=1,
            pos_mean=DEFAULT_POS_MEAN_1D,
            pos_std=DEFAULT_POS_STD_1D,
            neg_mean=DEFAULT_NEG_MEAN_1D,
            neg_std=DEFAULT_NEG_STD_1D,
            pos_noise_frac=pos_noise_frac,
            neg_noise_frac=neg_noise_frac,
        )

        self.val_dataLoader_one_dim = gen_custom_normal_data(
            num_samples=val_size,
            x_dim=1,
            pos_mean=DEFAULT_POS_MEAN_1D,
            pos_std=DEFAULT_POS_STD_1D,
            neg_mean=DEFAULT_NEG_MEAN_1D,
            neg_std=DEFAULT_NEG_STD_1D,
            pos_noise_frac=pos_noise_frac,
            neg_noise_frac=neg_noise_frac,
        )

        self.test_dataLoader_one_dim = gen_custom_normal_data(
            num_samples=test_size,
            x_dim=1,
            pos_mean=DEFAULT_POS_MEAN_1D,
            pos_std=DEFAULT_POS_STD_1D,
            neg_mean=DEFAULT_NEG_MEAN_1D,
            neg_std=DEFAULT_NEG_STD_1D,
            pos_noise_frac=pos_noise_frac,
            neg_noise_frac=neg_noise_frac,
        )

        self.loss_fn = BCEWithLogitsLossPNOne()
        self.linear_model = LinearModel(2)
        self.non_linear_model = NonLinearModel(2)
        self.cost = CostNormL2(dim=1)
        self.linear_delta = LinearStrategicDelta(
            cost=self.cost, strategic_model=self.linear_model
        )
        self.regulation = SocialBurden(self.linear_delta)

        self.linear_test_suite = ModelSuit(
            model=self.linear_model,
            delta=self.linear_delta,
            loss_fn=self.loss_fn,
            train_loader=self.train_dataLoader,
            validation_loader=self.val_dataLoader,
            test_loader=self.test_dataLoader,
            training_params=LINEAR_TRAINING_PARAMS,
        )

        self.non_linear_delta = NonLinearStrategicDelta(
            cost=self.cost,
            cost_weight=0.5,
            strategic_model=self.non_linear_model,
            training_params=DELTA_TRAINING_PARAMS,
            save_dir="./tests/artificial_data_set/delta_data",
        )

    # def test_linear_model(self):
    #     # Pass the logger to the Trainer

    #     max_epochs = 1 if DUMMY_RUN else num_epochs
    #     trainer = pl.Trainer(
    #         max_epochs=max_epochs,
    #         logger=CSVLogger("logs/", name="my_experiment"),
    #         log_every_n_steps=1,  # Ensure logging at each step
    #         devices=GPUS,
    #         accelerator=ACCELERATOR,
    #     )

    #     trainer.fit(self.linear_test_suite)
    #     trainer.test(self.linear_test_suite)

    #     visualize_train_and_test_2D(
    #         self.linear_model,
    #         self.train_dataLoader,
    #         self.test_dataLoader,
    #         self.linear_delta,
    #         display_percentage_train=0.05,
    #         display_percentage_test=0.5,
    #         prefix="linear",
    #     )

    # def test_identity_delta(self):
    #     linear_model = LinearModel(2)
    #     delta = IdentityDelta(cost=None, strategic_model=linear_model)

    #     identity_model = ModelSuit(
    #         model=linear_model,
    #         delta=delta,
    #         loss_fn=self.loss_fn,
    #         train_loader=self.train_dataLoader,
    #         validation_loader=self.val_dataLoader,
    #         test_loader=self.test_dataLoader,
    #         training_params=LINEAR_TRAINING_PARAMS,
    #     )

    #     # Pass the logger to the Trainer
    #     max_epochs = 1 if DUMMY_RUN else num_epochs
    #     trainer = pl.Trainer(
    #         max_epochs=max_epochs,
    #         logger=CSVLogger("logs/", name="my_experiment"),
    #         log_every_n_steps=1,  # Ensure logging at each step
    #         devices=GPUS,
    #         accelerator=ACCELERATOR,
    #     )
    #     trainer.test(identity_model)
    #     visualize_data_and_delta_2D(
    #         linear_model,
    #         self.train_dataLoader,
    #         delta,
    #         display_percentage=1,
    #         prefix="no_train",
    #     )
    #     visualize_data_and_delta_2D(
    #         linear_model,
    #         self.test_dataLoader,
    #         delta,
    #         display_percentage=1,
    #         prefix="test_no_train",
    #     )
    #     trainer.fit(identity_model)
    #     trainer.test(identity_model)
    #     visualize_data_and_delta_2D(
    #         linear_model,
    #         self.train_dataLoader,
    #         delta,
    #         display_percentage=1,
    #         prefix="train_identity",
    #     )
    #     visualize_data_and_delta_2D(
    #         linear_model,
    #         self.test_dataLoader,
    #         delta,
    #         display_percentage=1,
    #         prefix="test_identity",
    #     )
    #     # visualize the results with a strategic delta
    #     delta = LinearStrategicDelta(cost=self.cost, strategic_model=linear_model)
    #     identity_model.delta = delta
    #     trainer.test(identity_model)
    #     visualize_data_and_delta_2D(
    #         linear_model,
    #         self.train_dataLoader,
    #         delta,
    #         display_percentage=1,
    #         prefix="train_identity_model_with_strategic_delta",
    #     )
    #     visualize_data_and_delta_2D(
    #         linear_model,
    #         self.test_dataLoader,
    #         delta,
    #         display_percentage=1,
    #         prefix="test_identity_model_with_strategic_delta",
    #     )

    def test_linear_regularization_with_s_hinge(self):

        linear_model = LinearModel(
            2, weight=torch.Tensor([[1.0, 0.0]]), bias=torch.Tensor([-10.0])
        )
        linear_delta = LinearStrategicDelta(
            cost=self.cost, strategic_model=linear_model
        )

        loss_fn = StrategicHingeLoss(linear_model, linear_delta)
        linear_regularization = LinearL2Regularization(0.0003)
        training_params = {
            "optimizer_class": optim.Adam,
            "optimizer_params": {
                "lr": 0.005,
            },
        }

        model_suit = ModelSuit(
            model=linear_model,
            delta=linear_delta,
            loss_fn=loss_fn,
            linear_regularization=[linear_regularization],
            train_loader=self.train_dataLoader,
            validation_loader=self.val_dataLoader,
            test_loader=self.test_dataLoader,
            training_params=training_params,
        )

        # Pass the logger to the Trainer
        max_epochs = 1 if DUMMY_RUN else num_epochs
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=CSVLogger("logs/", name="my_experiment"),
            log_every_n_steps=1,  # Ensure logging at each step
            devices=GPUS,
            accelerator=ACCELERATOR,
            callbacks=[EarlyStopping(monitor="val_zero_one_loss_epoch", patience=16)],
        )

        trainer.test(model_suit)
        visualize_train_and_test_2D(
            model=linear_model,
            data_loader_train=self.train_dataLoader,
            data_loader_test=self.test_dataLoader,
            delta=linear_delta,
            display_percentage_train=0.05,
            display_percentage_test=0.5,
            prefix="temp",
        )
        trainer.fit(model_suit)
        trainer.test(model_suit)
        visualize_train_and_test_2D(
            model=linear_model,
            data_loader_train=self.train_dataLoader,
            data_loader_test=self.test_dataLoader,
            delta=linear_delta,
            display_percentage_train=0.05,
            display_percentage_test=0.5,
            prefix="train_s_hinge",
        )

    # def test_linear_one_dim(self):
    #     linear_model = LinearModel(1)
    #     delta = LinearStrategicDelta(cost=self.cost, strategic_model=linear_model)

    #     model_suit = ModelSuit(
    #         model=linear_model,
    #         delta=delta,
    #         loss_fn=self.loss_fn,
    #         train_loader=self.train_dataLoader_one_dim,
    #         validation_loader=self.val_dataLoader_one_dim,
    #         test_loader=self.test_dataLoader_one_dim,
    #         training_params=LINEAR_TRAINING_PARAMS,
    #     )

    #     max_epochs = 1 if DUMMY_RUN else num_epochs
    #     # Pass the logger to the Trainer
    #     trainer = pl.Trainer(
    #         max_epochs=max_epochs,
    #         logger=CSVLogger("logs/", name="my_experiment"),
    #         log_every_n_steps=1,  # Ensure logging at each step
    #         devices=GPUS,
    #         accelerator=ACCELERATOR,
    #     )

    #     trainer.fit(model_suit)
    #     trainer.test(model_suit)
    #     visualize_data_and_delta_1D(
    #         linear_model,
    #         self.train_dataLoader_one_dim,
    #         delta,
    #         display_percentage=0.05,
    #         prefix="train_one_dim",
    #     )
    #     visualize_data_and_delta_1D(
    #         linear_model,
    #         self.test_dataLoader_one_dim,
    #         delta,
    #         display_percentage=0.5,
    #         prefix="test_one_dim",
    #     )

    # def test_non_linear_model(self):
    #     identity_delta = IdentityDelta(cost=None, strategic_model=self.non_linear_model)

    #     non_linear_train_suite = ModelSuit(
    #         model=self.non_linear_model,
    #         delta=identity_delta,
    #         loss_fn=self.loss_fn,
    #         train_loader=self.train_dataLoader,
    #         validation_loader=self.val_dataLoader,
    #         test_loader=self.test_dataLoader,
    #         training_params=NON_LINEAR_TRAINING_PARAMS,
    #         train_delta_every=1,
    #     )

    #     # Train the model without delta
    #     max_epochs = 1 if DUMMY_RUN else 10
    #     if torch.cuda.is_available():
    #         max_epochs = 3
    #     trainer = pl.Trainer(
    #         max_epochs=max_epochs,
    #         logger=CSVLogger("logs/", name="my_experiment"),
    #         log_every_n_steps=1,  # Ensure logging at each step
    #         devices=GPUS,
    #         accelerator=ACCELERATOR,
    #     )

    #     trainer.fit(non_linear_train_suite)

    #     visualize_data_and_delta_2D(
    #         None,
    #         self.train_dataLoader,
    #         identity_delta,
    #         display_percentage=0.05,
    #         prefix="non_linear_delta_pre_delta_train",
    #     )
    #     visualize_data_and_delta_2D(
    #         None,
    #         self.test_dataLoader,
    #         identity_delta,
    #         display_percentage=0.05,
    #         prefix="non_linear_delta_pre_delta_test",
    #     )
    #     non_linear_train_suite.delta = self.non_linear_delta

    #     # Train the model with delta
    #     max_epochs = 1 if DUMMY_RUN else num_epochs
    #     trainer = pl.Trainer(
    #         max_epochs=max_epochs,
    #         logger=CSVLogger("logs/", name="my_experiment"),
    #         log_every_n_steps=1,  # Ensure logging at each step
    #         devices=GPUS,
    #         accelerator=ACCELERATOR,
    #     )
    #     trainer.fit(non_linear_train_suite)
    #     non_linear_train_suite.train_delta_for_test()

    #     trainer.test(non_linear_train_suite)

    #     # visualize the results
    #     visualize_data_and_delta_2D(
    #         None,
    #         self.train_dataLoader,
    #         self.non_linear_delta,
    #         display_percentage=0.05,
    #         prefix="non_linear_delta_train",
    #     )
    #     visualize_data_and_delta_2D(
    #         None,
    #         self.test_dataLoader,
    #         self.non_linear_delta,
    #         display_percentage=0.5,
    #         prefix="non_linear_delta_test",
    #     )

    # def test_non_linear_model_one_dim(self):
    #     identity_delta = IdentityDelta(cost=None, strategic_model=self.non_linear_model)
    #     non_linear_model = NonLinearModel(1)
    #     delta = NonLinearStrategicDelta(
    #         cost=self.cost,
    #         cost_weight=0.5,
    #         strategic_model=non_linear_model,
    #         training_params=DELTA_TRAINING_PARAMS,
    #         save_dir="./tests/artificial_data_test/delta_data",
    #     )

    #     non_linear_train_suite = ModelSuit(
    #         model=non_linear_model,
    #         delta=identity_delta,
    #         loss_fn=self.loss_fn,
    #         train_loader=self.train_dataLoader_one_dim,
    #         validation_loader=self.val_dataLoader_one_dim,
    #         test_loader=self.test_dataLoader_one_dim,
    #         training_params=NON_LINEAR_TRAINING_PARAMS,
    #         train_delta_every=1,
    #     )

    #     # Train the model without delta
    #     max_epochs = 1 if DUMMY_RUN else 10
    #     if torch.cuda.is_available():
    #         max_epochs = 3
    #     trainer = pl.Trainer(
    #         max_epochs=max_epochs,
    #         logger=CSVLogger("logs/", name="my_experiment"),
    #         log_every_n_steps=1,  # Ensure logging at each step
    #         devices=GPUS,
    #         accelerator=ACCELERATOR,
    #     )

    #     trainer.fit(non_linear_train_suite)

    #     visualize_data_and_delta_1D(
    #         None,
    #         self.train_dataLoader_one_dim,
    #         identity_delta,
    #         display_percentage=0.05,
    #         prefix="non_linear_delta_pre_delta_train_one_dim",
    #     )
    #     visualize_data_and_delta_1D(
    #         None,
    #         self.test_dataLoader_one_dim,
    #         identity_delta,
    #         display_percentage=0.5,
    #         prefix="non_linear_delta_pre_delta_test_one_dim",
    #     )
    #     non_linear_train_suite.delta = delta

    #     # Train the model with delta
    #     max_epochs = 1 if DUMMY_RUN else num_epochs
    #     trainer = pl.Trainer(
    #         max_epochs=max_epochs,
    #         logger=CSVLogger("logs/", name="my_experiment"),
    #         log_every_n_steps=1,  # Ensure logging at each step
    #         devices=GPUS,
    #         accelerator=ACCELERATOR,
    #     )
    #     trainer.fit(non_linear_train_suite)
    #     non_linear_train_suite.train_delta_for_test()

    #     trainer.test(non_linear_train_suite)

    #     # visualize the results
    #     visualize_data_and_delta_1D(
    #         None,
    #         self.train_dataLoader_one_dim,
    #         delta,
    #         display_percentage=0.05,
    #         prefix="non_linear_delta_train_one_dim",
    #     )
    #     visualize_data_and_delta_1D(
    #         None,
    #         self.test_dataLoader_one_dim,
    #         delta,
    #         display_percentage=0.5,
    #         prefix="non_linear_delta_test_one_dim",
    #     )

    # def test_linear_model_in_the_dark(self):
    #     # Initialize a LinearModel with random weights
    #     model_train = LinearModel(in_features=2)
    #     model_test = LinearModel(in_features=2)
    #     delta_train = LinearStrategicDelta(cost=self.cost, strategic_model=model_train)
    #     test_delta = LinearStrategicDelta(cost=self.cost, strategic_model=model_test)
    #     train_dataLoader_in_the_dark = gen_custom_normal_data(
    #         train_size // 3,
    #         2,
    #         np.array([blobs_dist / 2 + 10, 0]),
    #         np.array([blobs_std, blobs_x2_std]),
    #         np.array([-blobs_dist / 2 + 10, 0]),
    #         np.array([blobs_std, blobs_x2_std]),
    #         pos_noise_frac=pos_noise_frac,
    #         neg_noise_frac=neg_noise_frac,
    #     )

    #     val_dataLoader_in_the_dark = gen_custom_normal_data(
    #         val_size,
    #         2,
    #         np.array([blobs_dist / 2 + 10, 0]),
    #         np.array([blobs_std, blobs_x2_std]),
    #         np.array([-blobs_dist / 2 + 10, 0]),
    #         np.array([blobs_std, blobs_x2_std]),
    #         pos_noise_frac=pos_noise_frac,
    #         neg_noise_frac=neg_noise_frac,
    #     )

    #     in_the_dark_module_suite = ModelSuit(
    #         model=model_test,
    #         delta=test_delta,
    #         loss_fn=self.loss_fn,
    #         train_loader=train_dataLoader_in_the_dark,
    #         validation_loader=val_dataLoader_in_the_dark,
    #         test_loader=val_dataLoader_in_the_dark,
    #         training_params=LINEAR_TRAINING_PARAMS,
    #     )
    #     max_epochs = 1 if DUMMY_RUN else num_epochs

    #     in_the_dark_trainer = pl.Trainer(
    #         max_epochs=max_epochs,
    #         logger=CSVLogger("logs/", name="my_experiment"),
    #         log_every_n_steps=1,  # Ensure logging at each step
    #         devices=GPUS,
    #         accelerator=ACCELERATOR,
    #     )
    #     in_the_dark_trainer.fit(in_the_dark_module_suite)
    #     in_the_dark_trainer.test(in_the_dark_module_suite)
    #     visualize_train_and_test_2D(
    #         model_test,
    #         train_dataLoader_in_the_dark,
    #         val_dataLoader_in_the_dark,
    #         test_delta,
    #         display_percentage_train=0.05,
    #         display_percentage_test=0.5,
    #         prefix="in_the_dark_dummy_model",
    #     )

    #     model_suit = ModelSuit(
    #         model=model_train,
    #         delta=delta_train,
    #         loss_fn=self.loss_fn,
    #         train_loader=self.train_dataLoader,
    #         validation_loader=self.val_dataLoader,
    #         test_loader=self.test_dataLoader,
    #         test_delta=test_delta,
    #         training_params=LINEAR_TRAINING_PARAMS,
    #     )
    #     max_epochs = 1 if DUMMY_RUN else num_epochs
    #     trainer = pl.Trainer(
    #         max_epochs=max_epochs,
    #         logger=CSVLogger("logs/", name="my_experiment"),
    #         log_every_n_steps=1,  # Ensure logging at each step
    #         devices=GPUS,
    #         accelerator=ACCELERATOR,
    #     )
    #     trainer.fit(model_suit)
    #     trainer.test(model_suit)

    #     visualize_train_and_test_2D(
    #         model_train,
    #         self.train_dataLoader,
    #         self.test_dataLoader,
    #         test_delta,
    #         display_percentage_train=0.05,
    #         display_percentage_test=0.5,
    #         prefix="in_the_dark_model",
    #     )

    # def test_non_linear_model_in_the_dark(self):
    #     # Initialize a LinearModel with random weights
    #     model_train = NonLinearModel(x_dim=2)
    #     model_test = NonLinearModel(x_dim=2)
    #     delta_train = NonLinearStrategicDelta(
    #         cost=self.cost,
    #         strategic_model=model_train,
    #         training_params=DELTA_TRAINING_PARAMS,
    #     )
    #     delta_test = NonLinearStrategicDelta(
    #         cost=self.cost,
    #         strategic_model=model_train,
    #         training_params=DELTA_TRAINING_PARAMS,
    #     )
    #     train_dataLoader_in_the_dark = gen_custom_normal_data(
    #         train_size // 3,
    #         2,
    #         np.array([blobs_dist / 2 + 10, 0]),
    #         np.array([blobs_std, blobs_x2_std]),
    #         np.array([-blobs_dist / 2 + 10, 0]),
    #         np.array([blobs_std, blobs_x2_std]),
    #         pos_noise_frac=pos_noise_frac,
    #         neg_noise_frac=neg_noise_frac,
    #     )

    #     val_dataLoader_in_the_dark = gen_custom_normal_data(
    #         val_size,
    #         2,
    #         np.array([blobs_dist / 2 + 10, 0]),
    #         np.array([blobs_std, blobs_x2_std]),
    #         np.array([-blobs_dist / 2 + 10, 0]),
    #         np.array([blobs_std, blobs_x2_std]),
    #         pos_noise_frac=pos_noise_frac,
    #         neg_noise_frac=neg_noise_frac,
    #     )

    #     in_the_dark_module_suite = ModelSuit(
    #         model=model_test,
    #         delta=delta_test,
    #         loss_fn=self.loss_fn,
    #         train_loader=train_dataLoader_in_the_dark,
    #         validation_loader=val_dataLoader_in_the_dark,
    #         test_loader=val_dataLoader_in_the_dark,
    #         training_params=LINEAR_TRAINING_PARAMS,
    #     )

    #     max_epochs = 1 if DUMMY_RUN else num_epochs

    #     in_the_dark_trainer = pl.Trainer(
    #         max_epochs=max_epochs,
    #         logger=CSVLogger("logs/", name="my_experiment"),
    #         log_every_n_steps=1,  # Ensure logging at each step
    #         devices=GPUS,
    #         accelerator=ACCELERATOR,
    #     )
    #     # set the dummy to create the delta
    #     in_the_dark_trainer.fit(in_the_dark_module_suite)
    #     in_the_dark_module_suite.train_delta_for_test()
    #     in_the_dark_trainer.test(in_the_dark_module_suite)

    #     visualize_train_and_test_2D(
    #         None,
    #         train_dataLoader_in_the_dark,
    #         val_dataLoader_in_the_dark,
    #         delta_test,
    #         display_percentage_train=0.05,
    #         display_percentage_test=0.5,
    #         prefix="in_the_dark_dummy_model_non_linear",
    #     )

    #     # create the model itself

    #     model_suit = ModelSuit(
    #         model=model_train,
    #         delta=delta_train,
    #         loss_fn=self.loss_fn,
    #         train_loader=self.train_dataLoader,
    #         validation_loader=self.val_dataLoader,
    #         test_loader=self.test_dataLoader,
    #         test_delta=delta_test,
    #         training_params=LINEAR_TRAINING_PARAMS,
    #     )

    #     max_epochs = 1 if DUMMY_RUN else 100

    #     trainer = pl.Trainer(
    #         max_epochs=max_epochs,
    #         logger=CSVLogger("logs/", name="my_experiment"),
    #         log_every_n_steps=1,  # Ensure logging at each step
    #         devices=GPUS,
    #         accelerator=ACCELERATOR,
    #     )

    #     trainer.fit(model_suit)
    #     model_suit.train_delta_for_test()
    #     trainer.test(model_suit)

    #     visualize_train_and_test_2D(
    #         None,
    #         self.train_dataLoader,
    #         self.test_dataLoader,
    #         delta_test,
    #         display_percentage_train=0.05,
    #         display_percentage_test=0.5,
    #         prefix="in_the_dark_model_non_linear",
    #     )


if __name__ == "__main__":
    unittest.main()
