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
import pandas as pd
import matplotlib.pyplot as plt
from random import sample

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
    L2Regularization,
    StrategicHingeLoss,
)


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
        "lr": 0.001,
    },
}


VERBOSE = True
SEED = 0


def reset_seed():
    torch.manual_seed(SEED)


torch.set_default_dtype(torch.float64)
train_size = 5000
val_size = 1000
test_size = 1000

blobs_dist = 11
blobs_std = 1.5
blobs_x2_std = 1.5
pos_noise_frac = 0.0
neg_noise_frac = 0.0


def gen_custom_normal_data(
    num_samples,
    x_dim,
    pos_mean,
    pos_std,
    neg_mean,
    neg_std,
    pos_noise_frac=0.01,
    neg_noise_frac=0.01,
):
    reset_seed()
    pos_samples_num = num_samples // 2
    neg_samples_num = num_samples - pos_samples_num
    posX = (
        torch.randn((pos_samples_num, x_dim), dtype=torch.float64) * pos_std + pos_mean
    )
    negX = (
        torch.randn((neg_samples_num, x_dim), dtype=torch.float64) * neg_std + neg_mean
    )

    X = torch.cat((posX, negX), 0).to(torch.float64)  # Ensure X is float64

    Y = torch.unsqueeze(
        torch.cat(
            (
                torch.from_numpy(
                    np.random.choice(
                        [1, -1], len(posX), p=[1 - pos_noise_frac, pos_noise_frac]
                    )
                )
                .float()
                .to(torch.float64),  # Convert to float64
                torch.from_numpy(
                    np.random.choice(
                        [-1, 1], len(posX), p=[1 - neg_noise_frac, neg_noise_frac]
                    )
                )
                .float()
                .to(torch.float64),  # Convert to float64
            ),
            0,
        ),
        1,
    )

    dataset = TensorDataset(X, Y)
    return DataLoader(dataset, batch_size=100, shuffle=False, num_workers=5)


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
            train_size,
            2,
            np.array([blobs_dist / 2 + 10, 0]),
            np.array([blobs_std, blobs_x2_std]),
            np.array([-blobs_dist / 2 + 10, 0]),
            np.array([blobs_std, blobs_x2_std]),
            pos_noise_frac=pos_noise_frac,
            neg_noise_frac=neg_noise_frac,
        )

        self.val_dataLoader = gen_custom_normal_data(
            val_size,
            2,
            np.array([blobs_dist / 2 + 10, 0]),
            np.array([blobs_std, blobs_x2_std]),
            np.array([-blobs_dist / 2 + 10, 0]),
            np.array([blobs_std, blobs_x2_std]),
            pos_noise_frac=pos_noise_frac,
            neg_noise_frac=neg_noise_frac,
        )

        self.test_dataLoader = gen_custom_normal_data(
            test_size,
            2,
            np.array([blobs_dist / 2 + 10, 0]),
            np.array([blobs_std, blobs_x2_std]),
            np.array([-blobs_dist / 2 + 10, 0]),
            np.array([blobs_std, blobs_x2_std]),
            pos_noise_frac=pos_noise_frac,
            neg_noise_frac=neg_noise_frac,
        )

        self.train_dataLoader_one_dim = gen_custom_normal_data(
            train_size,
            1,
            blobs_dist / 2 + 10,
            blobs_std,
            -blobs_dist / 2 + 10,
            blobs_std,
            pos_noise_frac=pos_noise_frac,
            neg_noise_frac=neg_noise_frac,
        )

        self.val_dataLoader_one_dim = gen_custom_normal_data(
            val_size,
            1,
            blobs_dist / 2 + 10,
            blobs_std,
            -blobs_dist / 2 + 10,
            blobs_std,
            pos_noise_frac=pos_noise_frac,
            neg_noise_frac=neg_noise_frac,
        )

        self.test_dataLoader_one_dim = gen_custom_normal_data(
            test_size,
            1,
            blobs_dist / 2 + 10,
            blobs_std,
            -blobs_dist / 2 + 10,
            blobs_std,
            pos_noise_frac=pos_noise_frac,
            neg_noise_frac=neg_noise_frac,
        )

        self.loss_fn = nn.BCEWithLogitsLoss()
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
            save_dir="./tests/model_suit/delta_data",
        )

    def test_linear_model(self):
        # Pass the logger to the Trainer
        trainer = pl.Trainer(
            max_epochs=100,
            logger=CSVLogger("logs/", name="my_experiment"),
            log_every_n_steps=1,  # Ensure logging at each step
        )

        trainer.fit(self.linear_test_suite)
        trainer.test(self.linear_test_suite)

        # visualize the results
        if isinstance(self.linear_model, LinearModel):
            visualize_data_and_delta_2D(
                self.linear_model,
                self.train_dataLoader,
                self.linear_delta,
                display_percentage=0.05,
                prefix="train",
            )
            visualize_data_and_delta_2D(
                self.linear_model,
                self.test_dataLoader,
                self.linear_delta,
                display_percentage=0.5,
                prefix="test",
            )
        else:
            print(
                f"self.linear_model is not an instance of LinearModel, it's {type(self.linear_model)}"
            )

    def test_identity_delta(self):
        linear_model = LinearModel(2)
        delta = IdentityDelta(cost=None, strategic_model=linear_model)

        identity_model = ModelSuit(
            model=linear_model,
            delta=delta,
            loss_fn=self.loss_fn,
            train_loader=self.train_dataLoader,
            validation_loader=self.val_dataLoader,
            test_loader=self.test_dataLoader,
            training_params=LINEAR_TRAINING_PARAMS,
        )

        # Pass the logger to the Trainer
        trainer = pl.Trainer(
            max_epochs=100,
            logger=CSVLogger("logs/", name="my_experiment"),
            log_every_n_steps=1,  # Ensure logging at each step
        )

        trainer.fit(identity_model)
        trainer.test(identity_model)
        visualize_data_and_delta_2D(
            linear_model,
            self.train_dataLoader,
            delta,
            display_percentage=1,
            prefix="train_identity",
        )
        visualize_data_and_delta_2D(
            linear_model,
            self.test_dataLoader,
            delta,
            display_percentage=1,
            prefix="test_identity",
        )
        # visualize the results with a strategic delta
        delta = LinearStrategicDelta(cost=self.cost, strategic_model=linear_model)
        identity_model.delta = delta
        trainer.test(identity_model)
        visualize_data_and_delta_2D(
            linear_model,
            self.train_dataLoader,
            delta,
            display_percentage=1,
            prefix="train_identity_model_with_strategic_delta",
        )
        visualize_data_and_delta_2D(
            linear_model,
            self.test_dataLoader,
            delta,
            display_percentage=1,
            prefix="test_identity_model_with_strategic_delta",
        )

    def test_linear_regularization_with_s_hinge(self):
        linear_model = LinearModel(2)
        linear_delta = LinearStrategicDelta(
            cost=self.cost, strategic_model=linear_model
        )

        loss_fn = StrategicHingeLoss(linear_model, linear_delta)
        linear_regularization = L2Regularization(0.01)

        model_suit = ModelSuit(
            model=self.linear_model,
            delta=self.linear_delta,
            loss_fn=loss_fn,
            linear_regularization=[linear_regularization],
            train_loader=self.train_dataLoader,
            validation_loader=self.val_dataLoader,
            test_loader=self.test_dataLoader,
            training_params=LINEAR_TRAINING_PARAMS,
        )

        # Pass the logger to the Trainer
        trainer = pl.Trainer(
            max_epochs=100,
            logger=CSVLogger("logs/", name="my_experiment"),
            log_every_n_steps=1,  # Ensure logging at each step
        )

        trainer.fit(model_suit)
        trainer.test(model_suit)
        visualize_data_and_delta_2D(
            self.linear_model,
            self.train_dataLoader,
            self.linear_delta,
            display_percentage=1,
            prefix="train_s_hinge",
        )
        visualize_data_and_delta_2D(
            self.linear_model,
            self.test_dataLoader,
            self.linear_delta,
            display_percentage=1,
            prefix="test_s_hinge",
        )

    def test_linear_one_dim(self):
        linear_model = LinearModel(1)
        delta = LinearStrategicDelta(cost=self.cost, strategic_model=linear_model)

        model_suit = ModelSuit(
            model=linear_model,
            delta=delta,
            loss_fn=self.loss_fn,
            train_loader=self.train_dataLoader_one_dim,
            validation_loader=self.val_dataLoader_one_dim,
            test_loader=self.test_dataLoader_one_dim,
            training_params=LINEAR_TRAINING_PARAMS,
        )
        logger = pl.loggers.CSVLogger("logs/", name="my_experiment")

        # Pass the logger to the Trainer
        trainer = pl.Trainer(
            max_epochs=100,
            logger=CSVLogger("logs/", name="my_experiment"),
            log_every_n_steps=1,  # Ensure logging at each step
        )

        trainer.fit(model_suit)
        trainer.test(model_suit)
        visualize_data_and_delta_1D(
            linear_model,
            self.train_dataLoader_one_dim,
            delta,
            display_percentage=0.5,
            prefix="train_one_dim",
        )
        visualize_data_and_delta_1D(
            linear_model,
            self.test_dataLoader_one_dim,
            delta,
            display_percentage=0.5,
            prefix="test_one_dim",
        )

    def test_non_linear_model(self):
        logger = pl.loggers.CSVLogger("logs/", name="my_experiment")
        identity_delta = IdentityDelta(cost=None, strategic_model=self.non_linear_model)

        non_linear_train_suite = ModelSuit(
            model=self.non_linear_model,
            delta=identity_delta,
            loss_fn=self.loss_fn,
            train_loader=self.train_dataLoader,
            validation_loader=self.val_dataLoader,
            test_loader=self.test_dataLoader,
            training_params=NON_LINEAR_TRAINING_PARAMS,
            train_delta_every=1,
        )

        # Train the model without delta
        trainer = pl.Trainer(
            max_epochs=10,
            logger=CSVLogger("logs/", name="my_experiment"),
            log_every_n_steps=1,  # Ensure logging at each step
        )

        trainer.fit(non_linear_train_suite)

        visualize_data_and_delta_2D(
            None,
            self.train_dataLoader,
            identity_delta,
            display_percentage=0.05,
            prefix="non_linear_delta_pre_delta_train",
        )
        visualize_data_and_delta_2D(
            None,
            self.test_dataLoader,
            identity_delta,
            display_percentage=0.5,
            prefix="non_linear_delta_pre_delta_test",
        )
        non_linear_train_suite.delta = self.non_linear_delta

        # Train the model with delta
        trainer = pl.Trainer(
            max_epochs=100,
            logger=CSVLogger("logs/", name="my_experiment"),
            log_every_n_steps=1,  # Ensure logging at each step
        )
        trainer.fit(non_linear_train_suite)
        non_linear_train_suite.train_delta_for_test()

        trainer.test(non_linear_train_suite)

        # visualize the results
        visualize_data_and_delta_2D(
            None,
            self.train_dataLoader,
            self.non_linear_delta,
            display_percentage=0.05,
            prefix="non_linear_delta_train",
        )
        visualize_data_and_delta_2D(
            None,
            self.test_dataLoader,
            self.non_linear_delta,
            display_percentage=0.5,
            prefix="non_linear_delta_test",
        )

    def test_non_linear_model_one_dim(self):
        logger = pl.loggers.CSVLogger("logs/", name="my_experiment")
        identity_delta = IdentityDelta(cost=None, strategic_model=self.non_linear_model)
        non_linear_model = NonLinearModel(1)
        delta = NonLinearStrategicDelta(
            cost=self.cost,
            cost_weight=0.5,
            strategic_model=non_linear_model,
            training_params=DELTA_TRAINING_PARAMS,
            save_dir="./tests/model_suit/delta_data",
        )

        non_linear_train_suite = ModelSuit(
            model=non_linear_model,
            delta=identity_delta,
            loss_fn=self.loss_fn,
            train_loader=self.train_dataLoader_one_dim,
            validation_loader=self.val_dataLoader_one_dim,
            test_loader=self.test_dataLoader_one_dim,
            training_params=NON_LINEAR_TRAINING_PARAMS,
            train_delta_every=1,
        )

        # Train the model without delta
        trainer = pl.Trainer(
            max_epochs=10,
            logger=CSVLogger("logs/", name="my_experiment"),
            log_every_n_steps=1,  # Ensure logging at each step
        )

        trainer.fit(non_linear_train_suite)

        visualize_data_and_delta_1D(
            None,
            self.train_dataLoader_one_dim,
            identity_delta,
            display_percentage=0.05,
            prefix="non_linear_delta_pre_delta_train_one_dim",
        )
        visualize_data_and_delta_1D(
            None,
            self.test_dataLoader_one_dim,
            identity_delta,
            display_percentage=0.5,
            prefix="non_linear_delta_pre_delta_test_one_dim",
        )
        non_linear_train_suite.delta = delta

        # Train the model with delta
        trainer = pl.Trainer(
            max_epochs=100,
            logger=CSVLogger("logs/", name="my_experiment"),
            log_every_n_steps=1,  # Ensure logging at each step
        )
        trainer.fit(non_linear_train_suite)
        non_linear_train_suite.train_delta_for_test()

        trainer.test(non_linear_train_suite)

        # visualize the results
        visualize_data_and_delta_1D(
            None,
            self.train_dataLoader_one_dim,
            delta,
            display_percentage=0.05,
            prefix="non_linear_delta_train_one_dim",
        )
        visualize_data_and_delta_1D(
            None,
            self.test_dataLoader_one_dim,
            delta,
            display_percentage=0.5,
            prefix="non_linear_delta_test_one_dim",
        )

    def test_linear_model_in_the_dark(self):
        # Initialize a LinearModel with random weights
        model_train = LinearModel(in_features=2)
        model_test = LinearModel(in_features=2)
        delta_train = LinearStrategicDelta(cost=self.cost, strategic_model=model_train)
        delta_test = LinearStrategicDelta(cost=self.cost, strategic_model=model_test)
        train_dataLoader_in_the_dark = gen_custom_normal_data(
            train_size // 3,
            2,
            np.array([blobs_dist / 2 + 10, 0]),
            np.array([blobs_std, blobs_x2_std]),
            np.array([-blobs_dist / 2 + 10, 0]),
            np.array([blobs_std, blobs_x2_std]),
            pos_noise_frac=pos_noise_frac,
            neg_noise_frac=neg_noise_frac,
        )

        val_dataLoader_in_the_dark = gen_custom_normal_data(
            val_size,
            2,
            np.array([blobs_dist / 2 + 10, 0]),
            np.array([blobs_std, blobs_x2_std]),
            np.array([-blobs_dist / 2 + 10, 0]),
            np.array([blobs_std, blobs_x2_std]),
            pos_noise_frac=pos_noise_frac,
            neg_noise_frac=neg_noise_frac,
        )

        in_the_dark_module_suite = ModelSuit(
            model=model_test,
            delta=delta_test,
            loss_fn=self.loss_fn,
            train_loader=train_dataLoader_in_the_dark,
            validation_loader=val_dataLoader_in_the_dark,
            test_loader=val_dataLoader_in_the_dark,
            training_params=LINEAR_TRAINING_PARAMS,
        )

        in_the_dark_trainer = pl.Trainer(
            max_epochs=100,
            logger=CSVLogger("logs/", name="my_experiment"),
            log_every_n_steps=1,  # Ensure logging at each step
        )
        in_the_dark_trainer.fit(in_the_dark_module_suite)
        in_the_dark_trainer.test(in_the_dark_module_suite)
        visualize_train_and_test_2D(
            model_test,
            train_dataLoader_in_the_dark,
            val_dataLoader_in_the_dark,
            delta_test,
            display_percentage_train=0.5,
            display_percentage_test=0.5,
            prefix="in_the_dark_dummy_model",
        )

        model_suit = ModelSuit(
            model=model_train,
            delta=delta_train,
            loss_fn=self.loss_fn,
            train_loader=self.train_dataLoader,
            validation_loader=self.val_dataLoader,
            test_loader=self.test_dataLoader,
            delta_test=delta_test,
            training_params=LINEAR_TRAINING_PARAMS,
        )

        trainer = pl.Trainer(
            max_epochs=100,
            logger=CSVLogger("logs/", name="my_experiment"),
            log_every_n_steps=1,  # Ensure logging at each step
        )
        trainer.fit(model_suit)
        trainer.test(model_suit)

        visualize_train_and_test_2D(
            model_train,
            self.train_dataLoader,
            self.test_dataLoader,
            delta_test,
            display_percentage_train=0.5,
            display_percentage_test=0.5,
            prefix="in_the_dark_model",
        )

    def test_non_linear_model_in_the_dark(self):
        # Initialize a LinearModel with random weights
        model_train = NonLinearModel(x_dim=2)
        model_test = NonLinearModel(x_dim=2)
        delta_train = NonLinearStrategicDelta(
            cost=self.cost,
            strategic_model=model_train,
            training_params=DELTA_TRAINING_PARAMS,
        )
        delta_test = NonLinearStrategicDelta(
            cost=self.cost,
            strategic_model=model_train,
            training_params=DELTA_TRAINING_PARAMS,
        )
        train_dataLoader_in_the_dark = gen_custom_normal_data(
            train_size // 3,
            2,
            np.array([blobs_dist / 2 + 10, 0]),
            np.array([blobs_std, blobs_x2_std]),
            np.array([-blobs_dist / 2 + 10, 0]),
            np.array([blobs_std, blobs_x2_std]),
            pos_noise_frac=pos_noise_frac,
            neg_noise_frac=neg_noise_frac,
        )

        val_dataLoader_in_the_dark = gen_custom_normal_data(
            val_size,
            2,
            np.array([blobs_dist / 2 + 10, 0]),
            np.array([blobs_std, blobs_x2_std]),
            np.array([-blobs_dist / 2 + 10, 0]),
            np.array([blobs_std, blobs_x2_std]),
            pos_noise_frac=pos_noise_frac,
            neg_noise_frac=neg_noise_frac,
        )

        in_the_dark_module_suite = ModelSuit(
            model=model_test,
            delta=delta_test,
            loss_fn=self.loss_fn,
            train_loader=train_dataLoader_in_the_dark,
            validation_loader=val_dataLoader_in_the_dark,
            test_loader=val_dataLoader_in_the_dark,
            training_params=LINEAR_TRAINING_PARAMS,
        )

        in_the_dark_trainer = pl.Trainer(
            max_epochs=100,
            logger=CSVLogger("logs/", name="my_experiment"),
            log_every_n_steps=1,  # Ensure logging at each step
        )
        # set the dummy to create the delta
        in_the_dark_trainer.fit(in_the_dark_module_suite)
        in_the_dark_module_suite.train_delta_for_test()
        in_the_dark_trainer.test(in_the_dark_module_suite)

        visualize_train_and_test_2D(
            None,
            train_dataLoader_in_the_dark,
            val_dataLoader_in_the_dark,
            delta_test,
            display_percentage_train=0.5,
            display_percentage_test=0.5,
            prefix="in_the_dark_dummy_model_non_linear",
        )

        # create the model itself

        model_suit = ModelSuit(
            model=model_train,
            delta=delta_train,
            loss_fn=self.loss_fn,
            train_loader=self.train_dataLoader,
            validation_loader=self.val_dataLoader,
            test_loader=self.test_dataLoader,
            delta_test=delta_test,
            training_params=LINEAR_TRAINING_PARAMS,
        )

        trainer = pl.Trainer(
            max_epochs=100,
            logger=CSVLogger("logs/", name="my_experiment"),
            log_every_n_steps=1,  # Ensure logging at each step
        )

        trainer.fit(model_suit)
        model_suit.train_delta_for_test()
        trainer.test(model_suit)

        visualize_train_and_test_2D(
            None,
            self.train_dataLoader,
            self.test_dataLoader,
            delta_test,
            display_percentage_train=0.5,
            display_percentage_test=0.5,
            prefix="in_the_dark_model_non_linear",
        )


if __name__ == "__main__":
    unittest.main()
