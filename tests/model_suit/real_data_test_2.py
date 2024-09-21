import os
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import unittest
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import logging
from typing import Dict, Any
import matplotlib.pyplot as plt

# Imports for your models
from strategic_ml import (
    ModelSuit,
    LinearModel,
    LinearStrategicDelta,
    CostNormL2,
)
from strategic_ml.loss_functions import StrategicHingeLoss


# Helper function to load the dataset
def gen_coats_dataset(seed=0):
    def one_hot_to_discrete(ranges, features):
        new_features = np.zeros((features.shape[0], len(ranges) + 1))
        ranges = [0] + ranges + [features.shape[1]]
        for i in range(len(ranges) - 1):
            columns = features[:, ranges[i] : ranges[i + 1]]
            new_features[:, i] = np.array(
                [np.where(one_hot == 1)[0][0] for one_hot in columns]
            )
        return new_features

    path = "tests/data"

    # Indices for features
    USER_RANGES = [2, 8, 11]
    ITEM_RANGES = [2, 18, 31]

    with open(
        path + "/user_item_features/user_features.ascii", encoding="utf-8-sig"
    ) as f:
        user_features = np.loadtxt(f)
        user_features = one_hot_to_discrete(USER_RANGES, user_features)

    with open(
        path + "/user_item_features/item_features.ascii", encoding="utf-8-sig"
    ) as f:
        item_features = np.loadtxt(f)
        item_features = one_hot_to_discrete(ITEM_RANGES, item_features)

    with open(path + "/train.ascii", encoding="utf-8-sig") as f:
        train = np.loadtxt(f)

    with open(path + "/test.ascii", encoding="utf-8-sig") as f:
        test = np.loadtxt(f)

    num_users, x_dim = user_features.shape
    num_items, z_dim = item_features.shape

    train_indices = (train != 0).nonzero()
    test_indices = (test != 0).nonzero()

    n_items_known = len(train_indices[0]) // num_users
    n_items_unknown = len(test_indices[0]) // num_users

    X = user_features
    Z_known = np.zeros((num_users, n_items_known, z_dim))
    Z_unknown = np.zeros((num_users, n_items_unknown, z_dim))
    Y_known = np.zeros((num_users, n_items_known))
    Y_unknown = np.zeros((num_users, n_items_unknown))

    for i, (user, item) in enumerate(zip(*train_indices)):
        Z_known[user, i % n_items_known] = item_features[item]
        Y_known[user, i % n_items_known] = train[user, item]

    for i, (user, item) in enumerate(zip(*test_indices)):
        Z_unknown[user, i % n_items_unknown] = item_features[item]
        Y_unknown[user, i % n_items_unknown] = test[user, item]

    # Binarize the ratings
    Y_known[Y_known < 3] = -1
    Y_known[Y_known >= 3] = 1
    Y_unknown[Y_unknown < 3] = -1
    Y_unknown[Y_unknown >= 3] = 1

    return (
        torch.from_numpy(X),
        torch.from_numpy(Z_known),
        torch.from_numpy(Z_unknown),
        torch.from_numpy(Y_known),
        torch.from_numpy(Y_unknown),
    )


def split_data(dataset, test_percentage):
    X, Z_known, Z_unknown, Y_known, Y_unknown = dataset
    num_test = int(len(X) * test_percentage)
    return (
        (
            X[num_test:],
            Z_known[num_test:],
            Z_unknown[num_test:],
            Y_known[num_test:],
            Y_unknown[num_test:],
        ),
        (
            X[:num_test],
            Z_known[:num_test],
            Z_unknown[:num_test:],
            Y_known[:num_test],
            Y_unknown[:num_test],
        ),
    )


LINEAR_TRAINING_PARAMS: Dict[str, Any] = {
    "optimizer_class": optim.SGD,
    "optimizer_params": {
        "lr": 1e-3,
    },
}


class TestModelSuit(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        logging.info("Loading dataset")

        # Load dataset
        dataset = gen_coats_dataset(seed=0)
        logging.info(
            f"Dataset loaded with shapes: "
            f"X: {dataset[0].shape}, Z_known: {dataset[1].shape}, Z_unknown: {dataset[2].shape}, "
            f"Y_known: {dataset[3].shape}, Y_unknown: {dataset[4].shape}"
        )

        # Split into train/validation/test sets
        train_dataset, val_and_test_dataset = split_data(dataset, test_percentage=0.4)
        val_dataset, test_dataset = split_data(
            val_and_test_dataset, test_percentage=0.5
        )

        # Convert all tensors to float
        X_train, Z_known_train, Z_unknown_train, Y_known_train, Y_unknown_train = (
            train_dataset[0].float(),
            train_dataset[1].float(),
            train_dataset[2].float(),
            train_dataset[3].float(),
            train_dataset[4].float(),
        )

        X_val, Z_known_val, Z_unknown_val, Y_known_val, Y_unknown_val = (
            val_dataset[0].float(),
            val_dataset[1].float(),
            val_dataset[2].float(),
            val_dataset[3].float(),
            val_dataset[4].float(),
        )

        X_test, Z_known_test, Z_unknown_test, Y_known_test, Y_unknown_test = (
            test_dataset[0].float(),
            test_dataset[1].float(),
            test_dataset[2].float(),
            test_dataset[3].float(),
            test_dataset[4].float(),
        )

        # Log the shapes of split datasets
        logging.info(
            f"Train dataset shapes: X: {X_train.shape}, Z_known: {Z_known_train.shape}, "
            f"Z_unknown: {Z_unknown_train.shape}, Y_known: {Y_known_train.shape}, Y_unknown: {Y_unknown_train.shape}"
        )
        logging.info(
            f"Validation dataset shapes: X: {X_val.shape}, Z_known: {Z_known_val.shape}, "
            f"Z_unknown: {Z_unknown_val.shape}, Y_known: {Y_known_val.shape}, Y_unknown: {Y_unknown_val.shape}"
        )
        logging.info(
            f"Test dataset shapes: X: {X_test.shape}, Z_known: {Z_known_test.shape}, "
            f"Z_unknown: {Z_unknown_test.shape}, Y_known: {Y_known_test.shape}, Y_unknown: {Y_unknown_test.shape}"
        )

        # Use DataLoader for train/validation/test sets
        self.train_dataset = DataLoader(
            TensorDataset(
                X_train, Z_known_train, Z_unknown_train, Y_known_train, Y_unknown_train
            ),
            batch_size=16,
            shuffle=True,
        )
        self.val_dataset = DataLoader(
            TensorDataset(
                X_val, Z_known_val, Z_unknown_val, Y_known_val, Y_unknown_val
            ),
            batch_size=16,
            shuffle=False,
        )
        self.test_dataset = DataLoader(
            TensorDataset(
                X_test, Z_known_test, Z_unknown_test, Y_known_test, Y_unknown_test
            ),
            batch_size=16,
            shuffle=False,
        )

        # Define models, loss function, and regularization
        self.linear_model = LinearModel(X_train.shape[1])
        self.cost = CostNormL2(dim=1)
        self.linear_delta = LinearStrategicDelta(
            cost=self.cost, strategic_model=self.linear_model
        )
        self.loss_fn = nn.MSELoss()  # Replacing StrategicHingeLoss for testing

        # Initialize ModelSuit
        self.linear_test_suite = ModelSuit(
            model=self.linear_model,
            delta=self.linear_delta,
            loss_fn=self.loss_fn,
            train_loader=self.train_dataset,
            validation_loader=self.val_dataset,
            test_loader=self.test_dataset,
            training_params=LINEAR_TRAINING_PARAMS,
        )

    def test_linear_model(self):
        # CSV logger
        logger = CSVLogger("logs/", name="real_data_experiment")

        # Initialize trainer
        trainer = pl.Trainer(max_epochs=100, logger=logger, log_every_n_steps=1)

        # Log before starting the training
        logging.info("Starting training")

        # Train and test the model
        trainer.fit(self.linear_test_suite)
        trainer.test(self.linear_test_suite)

        # Log after training
        logging.info("Training complete")

        # Visualize results
        self.visualize_results(trainer)

    # def visualize_results(self, trainer):
    #     # Read CSV log file
    #     csv_log_path = trainer.logger.experiment.metrics_file_path
    #     metrics_df = pd.read_csv(csv_log_path)
    #     logging.info("Metrics dataframe content:\n" + metrics_df.to_string())

    #     # Plotting logic
    #     plt.figure(figsize=(10, 5))
    #     metrics_df.plot()
    #     plt.title("Training and Validation Metrics")
    #     plt.savefig('training_results_real_data.png')
    #     logging.info("Training results saved to 'training_results_real_data.png'")

    # def visualize_results(self, trainer):
    #     # Read CSV log file
    #     csv_log_path = trainer.logger.experiment.metrics_file_path
    #     metrics_df = pd.read_csv(csv_log_path)
    #     print("Metrics dataframe content:\n", metrics_df)

    #     # Print available columns for debugging
    #     print("Available metrics:", metrics_df.columns)

    #     # Check for train loss column by name
    #     train_loss_column = None
    #     zero_one_loss_column = None

    #     for col in metrics_df.columns:
    #         if "train_loss" in col:
    #             train_loss_column = col
    #         if "zero_one_loss" in col:
    #             zero_one_loss_column = col

    #     # Extract logged metrics
    #     if train_loss_column:
    #         train_loss = metrics_df[train_loss_column].dropna().values
    #     else:
    #         print("train_loss column not found")
    #         train_loss = []

    #     if zero_one_loss_column:
    #         zero_one_loss = metrics_df[zero_one_loss_column].dropna().values
    #     else:
    #         print("zero_one_loss column not found")
    #         zero_one_loss = []

    #     # Plot the metrics
    #     epochs = range(len(train_loss))

    #     plt.figure(figsize=(10, 5))

    #     # Plot training loss
    #     plt.subplot(1, 2, 1)
    #     if len(train_loss) > 0:
    #         plt.plot(epochs, train_loss, label="Train Loss")
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Loss")
    #     plt.title("Training Loss Across Epochs")
    #     plt.legend()

    #     # Plot zero-one loss
    #     plt.subplot(1, 2, 2)
    #     if len(zero_one_loss) > 0:
    #         plt.plot(epochs, zero_one_loss, label="Zero-One Loss")
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Zero-One Loss")
    #     plt.title("Zero-One Loss Across Epochs")
    #     plt.legend()

    #     # Save the plot
    #     plt.savefig('training_results_real_data.png')
    #     print("Training results saved to 'training_results_real_data.png'")

    def visualize_results(self, trainer):
        # Read CSV log file
        csv_log_path = trainer.logger.experiment.metrics_file_path
        metrics_df = pd.read_csv(csv_log_path)
        print("Metrics dataframe content:\n", metrics_df)

        # Print available columns for debugging
        print("Available metrics:", metrics_df.columns)

        # Check for relevant metric columns by name
        train_loss_column = None
        zero_one_loss_column = None
        val_loss_column = None
        test_loss_column = None

        for col in metrics_df.columns:
            if "train_loss" in col:
                train_loss_column = col
            if "zero_one_loss" in col:
                zero_one_loss_column = col
            if "val_loss" in col:
                val_loss_column = col
            if "test_loss" in col:
                test_loss_column = col

        # Extract logged metrics
        train_loss = (
            metrics_df[train_loss_column].dropna().values if train_loss_column else []
        )
        zero_one_loss = (
            metrics_df[zero_one_loss_column].dropna().values
            if zero_one_loss_column
            else []
        )
        val_loss = (
            metrics_df[val_loss_column].dropna().values if val_loss_column else []
        )
        test_loss = (
            metrics_df[test_loss_column].dropna().values if test_loss_column else []
        )

        # Ensure epochs matches the metric size
        epochs = range(len(train_loss))  # Adjust to match the length of train_loss
        val_epochs = range(len(val_loss))  # Adjust to match the length of val_loss
        test_epochs = range(len(test_loss))  # Adjust to match the length of test_loss

        plt.figure(figsize=(15, 10))

        # Plot training loss
        plt.subplot(2, 2, 1)
        if len(train_loss) > 0:
            plt.plot(epochs, train_loss, label="Train Loss", color="blue")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Across Epochs")
        plt.legend()

        # Plot zero-one loss
        plt.subplot(2, 2, 2)
        if len(zero_one_loss) > 0:
            plt.plot(epochs, zero_one_loss, label="Zero-One Loss", color="green")
        plt.xlabel("Epochs")
        plt.ylabel("Zero-One Loss")
        plt.title("Zero-One Loss Across Epochs")
        plt.legend()

        # Plot validation loss
        plt.subplot(2, 2, 3)
        if len(val_loss) > 0:
            plt.plot(val_epochs, val_loss, label="Validation Loss", color="orange")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Validation Loss Across Epochs")
        plt.legend()

        # Plot test loss
        plt.subplot(2, 2, 4)
        if len(test_loss) > 0:
            plt.plot(test_epochs, test_loss, label="Test Loss", color="red")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Test Loss Across Epochs")
        plt.legend()

        # Save the plot
        plt.tight_layout()
        plt.savefig("training_results_real_data_extended.png")
        print("Training results saved to 'training_results_real_data_extended.png'")


if __name__ == "__main__":
    unittest.main()
