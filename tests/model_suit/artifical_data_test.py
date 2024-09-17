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
    LinearStrategicModel,
    LinearStrategicDelta,
    NonLinearStrategicDelta,
    CostNormL2,
    SocialBurden,
    visualize_linear_classifier_2D,
)


DELTA_TRAINING_PARAMS: Dict[str, Any] = {
    "num_epochs": 500,
    "optimizer_class": optim.SGD,
    "optimizer_params": {
        "lr": 1,
    },
    "scheduler_class": optim.lr_scheduler.StepLR,
    "scheduler_params": {
        "step_size": 100,
        "gamma": 0.5,
    },
    "early_stopping": 60,
    "temp": 20,
}

NON_LINEAR_TRAINING_PARAMS: Dict[str, Any] = {}

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
test_size = 500

x_dim = 2
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
    return DataLoader(dataset, batch_size=100, shuffle=False)


def print_if_verbose(message: str) -> None:
    global VERBOSE
    if VERBOSE:
        print(message)


class NonLinearModel(torch.nn.Module):
    def __init__(self, x_dim: int) -> None:
        super(NonLinearModel, self).__init__()
        self.linear1 = torch.nn.Linear(x_dim, 10)
        self.linear2 = torch.nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class TestModelSuit(unittest.TestCase):
    def setUp(self):
        self.train_dataset = gen_custom_normal_data(
            train_size,
            x_dim,
            np.array([blobs_dist / 2 + 10, 0]),
            np.array([blobs_std, blobs_x2_std]),
            np.array([-blobs_dist / 2 + 10, 0]),
            np.array([blobs_std, blobs_x2_std]),
            pos_noise_frac=pos_noise_frac,
            neg_noise_frac=neg_noise_frac,
        )

        self.val_dataset = gen_custom_normal_data(
            val_size,
            x_dim,
            np.array([blobs_dist / 2 + 10, 0]),
            np.array([blobs_std, blobs_x2_std]),
            np.array([-blobs_dist / 2 + 10, 0]),
            np.array([blobs_std, blobs_x2_std]),
            pos_noise_frac=pos_noise_frac,
            neg_noise_frac=neg_noise_frac,
        )

        self.test_dataset = gen_custom_normal_data(
            test_size,
            x_dim,
            np.array([blobs_dist / 2 + 10, 0]),
            np.array([blobs_std, blobs_x2_std]),
            np.array([-blobs_dist / 2 + 10, 0]),
            np.array([blobs_std, blobs_x2_std]),
            pos_noise_frac=pos_noise_frac,
            neg_noise_frac=neg_noise_frac,
        )

        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.linear_model = LinearStrategicModel(x_dim)
        self.non_linear_model = NonLinearModel(x_dim)
        self.cost = CostNormL2(dim=1)
        self.linear_delta = LinearStrategicDelta(
            cost=self.cost, strategic_model=self.linear_model
        )
        self.regulation = SocialBurden(self.linear_delta)

        self.linear_test_suite = ModelSuit(
            model=self.linear_model,
            delta=self.linear_delta,
            loss_fn=self.loss_fn,
            # regularization=self.regulation,
            train_loader=self.train_dataset,
            validation_loader=self.val_dataset,
            test_loader=self.test_dataset,
            training_params=LINEAR_TRAINING_PARAMS,
        )

        self.non_linear_delta = NonLinearStrategicDelta(
            cost=self.cost,
            strategic_model=self.non_linear_model,
            training_params=DELTA_TRAINING_PARAMS,
        )

        self.non_linear_train_suite = ModelSuit(
            model=self.non_linear_model,
            delta=self.non_linear_delta,
            loss_fn=self.loss_fn,
            train_loader=self.train_dataset,
            validation_loader=self.val_dataset,
            test_loader=self.test_dataset,
            training_params=NON_LINEAR_TRAINING_PARAMS,
        )

    def test_linear_model(self):

        logger = pl.loggers.CSVLogger("logs/", name="my_experiment")

        # Pass the logger to the Trainer
        trainer = pl.Trainer(
            max_epochs=10,
            logger=CSVLogger("logs/", name="my_experiment"),
            log_every_n_steps=1,  # Ensure logging at each step
        )

        trainer.fit(self.linear_test_suite)
        trainer.test(self.linear_test_suite)
        # visualize the results
        # self.visualize_results(trainer)

        # After training the linear model:
        if isinstance(self.linear_model, LinearStrategicModel):
            visualize_linear_classifier_2D(
                self.linear_model,
                self.train_dataset,
                self.linear_delta,
                display_percentage=0.05,
                prefix="train",
            )
            visualize_linear_classifier_2D(
                self.linear_model,
                self.test_dataset,
                self.linear_delta,
                display_percentage=0.05,
                prefix="test",
            )
        else:
            print(
                f"self.linear_model is not an instance of LinearStrategicModel, it's {type(self.linear_model)}"
            )


if __name__ == "__main__":
    unittest.main()
