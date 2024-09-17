# External imports
import os
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from typing import Dict, Any
import pytorch_lightning as pl
import unittest
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from strategic_ml.loss_functions import StrategicHingeLoss


# internal imports
from strategic_ml import (
    ModelSuit,
    LinearStrategicModel,
    LinearStrategicDelta,
    CostNormL2,
    SocialBurden,
)


# Helper functions to load real data
def load_real_data():
    # Paths for your data files

    features_dir = "tests/data/user_item_features"
    data_dir = "tests/data/"
    user_features_path = os.path.join(features_dir, "user_features.ascii")
    item_features_path = os.path.join(features_dir, "item_features.ascii")
    train_path = os.path.join(data_dir, "train.ascii")
    test_path = os.path.join(data_dir, "test.ascii")

    # Load the real data from .ascii files
    user_features = np.loadtxt(user_features_path)
    item_features = np.loadtxt(item_features_path)
    train_data = np.loadtxt(train_path)
    test_data = np.loadtxt(test_path)

    return user_features, item_features, train_data, test_data


# Parameters for training
DELTA_TRAINING_PARAMS: Dict[str, Any] = {
    "num_epochs": 500,
    "optimizer_class": optim.Adam,
    "optimizer_params": {
        "lr": 0.000001,
    },
    "scheduler_class": optim.lr_scheduler.StepLR,
    "scheduler_params": {
        "step_size": 10,
        "gamma": 0.1,
    },
    "early_stopping": 60,
    "temp": 20,
}

LINEAR_TRAINING_PARAMS: Dict[str, Any] = {
    "optimizer_class": optim.SGD,
    "optimizer_params": {
        "lr": 1e-3,
    },
}


def shuffle(dataset, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, Z_known, Y_known, Y_unknown = dataset  # Expecting 4 elements
    perm_users = torch.randperm(len(X))  # Shuffle users (first dimension)

    # Apply user shuffle to all data components
    X, Z_known, Y_known, Y_unknown = (
        X[perm_users],
        Z_known[perm_users],
        Y_known[perm_users],
        Y_unknown[perm_users],
    )

    # Shuffle items for each user
    for i in range(len(X)):
        num_known_items = Z_known[i].size(
            0
        )  # Get the number of known items for the current user

        # Ensure Y_known matches the number of known items in Z_known
        if Y_known[i].size(0) != num_known_items:
            raise ValueError(
                f"Mismatch between Z_known and Y_known sizes for user {i}: Z_known has {num_known_items} items, but Y_known has {Y_known[i].size(0)} items."
            )

        perm_items = torch.randperm(
            num_known_items
        )  # Generate a permutation matching the number of known items
        Z_known[i] = Z_known[i][perm_items]  # Shuffle items in Z_known for this user
        Y_known[i] = Y_known[i][perm_items]  # Shuffle items in Y_known for this user

    return (X, Z_known, Y_known, Y_unknown)


# Adjust split_data function accordingly
def split_data(dataset, test_percentage):
    X, Z_known, Y_known, Y_unknown = shuffle(dataset)
    num_test = int(len(X) * test_percentage)
    return (
        (X[num_test:], Z_known[num_test:], Y_known[num_test:], Y_unknown[num_test:]),
        (X[:num_test], Z_known[:num_test], Y_known[:num_test], Y_unknown[:num_test]),
    )


class TestModelSuit(unittest.TestCase):
    # def setUp(self):
    #     # Load real dataset
    #     user_features, item_features, train_data, test_data = load_real_data()

    #     # Convert data to PyTorch tensors
    #     X_train = torch.from_numpy(user_features).float()  # Shape: [290, 14] (users)
    #     Z_train = torch.from_numpy(item_features).float()  # Shape: [300, 33] (items)
    #     Y_train = torch.from_numpy(train_data).float()  # Shape: [290, 300] (ratings)

    #     # Flatten the interaction matrix to create user-item pairs
    #     num_users, num_items = Y_train.shape
    #     user_indices, item_indices = torch.meshgrid(torch.arange(num_users), torch.arange(num_items))
    #     user_indices = user_indices.flatten()
    #     item_indices = item_indices.flatten()

    #     # Create the combined dataset by pairing each user with an item and their rating
    #     X_train_expanded = X_train[user_indices]  # Shape: [290*300, 14] (expanded users)
    #     Z_train_expanded = Z_train[item_indices]  # Shape: [290*300, 33] (expanded items)
    #     Y_train_flattened = Y_train.flatten()  # Shape: [290*300] (flattened ratings)

    #     # Combine user features, item features, and ratings into one dataset
    #     full_train_dataset = TensorDataset(X_train_expanded, Z_train_expanded, Y_train_flattened)

    #     # Split part of the training data into validation data
    #     validation_split = 0.2
    #     train_size = int((1 - validation_split) * len(full_train_dataset))
    #     val_size = len(full_train_dataset) - train_size

    #     train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

    #     # Use the DataLoader for both train and validation sets
    #     self.train_dataset = DataLoader(train_dataset, batch_size=16, shuffle=True)
    #     self.val_dataset = DataLoader(val_dataset, batch_size=16, shuffle=False)
    #     self.test_dataset = DataLoader(TensorDataset(X_train_expanded, Z_train_expanded, Y_train_flattened), batch_size=16, shuffle=False)

    #     # Define loss function and models
    #     self.loss_fn = nn.MSELoss()
    #     self.linear_model = LinearStrategicModel(X_train.shape[1])  # Assuming X_train has shape (n_users, features)
    #     self.cost = CostNormL2()
    #     self.linear_delta = LinearStrategicDelta(cost=self.cost, strategic_model=self.linear_model)
    #     # self.regulation = SocialBurden(self.linear_delta)

    #     # Update with validation_loader
    #     self.linear_test_suite = ModelSuit(
    #         model=self.linear_model,
    #         delta=self.linear_delta,
    #         loss_fn=self.loss_fn,
    #         # regularization=self.regulation,
    #         train_loader=self.train_dataset,
    #         validation_loader=self.val_dataset,  # Add this argument
    #         test_loader=self.test_dataset,
    #         training_params=LINEAR_TRAINING_PARAMS,
    #     )
    def setUp(self):
        # Load and shuffle the dataset
        user_features, item_features, train_data, test_data = load_real_data()

        # Split into train, val, test datasets
        full_dataset = (
            torch.from_numpy(user_features).float(),
            torch.from_numpy(item_features).float(),
            torch.from_numpy(train_data).float(),
            torch.from_numpy(test_data).float(),
        )

        train_dataset, val_and_test_dataset = split_data(
            full_dataset, test_percentage=0.4
        )
        val_dataset, test_dataset = split_data(
            val_and_test_dataset, test_percentage=0.5
        )

        # Update training parameters based on project needs
        self.train_dataset = DataLoader(train_dataset, batch_size=16, shuffle=True)
        self.val_dataset = DataLoader(val_dataset, batch_size=16, shuffle=False)
        self.test_dataset = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Model Setup
        self.model = LinearStrategicModel(train_dataset[0].shape[1])
        self.cost = CostNormL2()
        self.delta = LinearStrategicDelta(cost=self.cost, strategic_model=self.model)
        self.loss_fn = StrategicHingeLoss(model=self.model, delta=self.delta)

        # Define the ModelSuit for strategic model training
        self.suit = ModelSuit(
            model=self.model,
            delta=self.delta,
            loss_fn=self.loss_fn,
            train_loader=self.train_dataset,
            validation_loader=self.val_dataset,
            test_loader=self.test_dataset,
            training_params=LINEAR_TRAINING_PARAMS,
        )

        print(
            f"X_train shape: {X_train.shape}, Z_train shape: {Z_train.shape}, Y_train shape: {Y_train.shape}"
        )

    def test_linear_model(self):
        # CSV logger
        logger = CSVLogger("logs/", name="real_data_experiment")

        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=10, logger=logger, log_every_n_steps=1, gradient_clip_val=0.5
        )

        # Train and test the model
        trainer.fit(self.linear_test_suite)
        trainer.test(self.linear_test_suite)

        # Visualize results
        self.visualize_results(trainer)

    def visualize_results(self, trainer):
        # Read CSV log file
        csv_log_path = trainer.logger.experiment.metrics_file_path
        metrics_df = pd.read_csv(csv_log_path)
        print("Metrics dataframe content:\n", metrics_df)

        # Print available columns for debugging
        print("Available metrics:", metrics_df.columns)

        # Check for train loss column by name
        train_loss_column = None
        zero_one_loss_column = None

        for col in metrics_df.columns:
            if "train_loss" in col:
                train_loss_column = col
            if "zero_one_loss" in col:
                zero_one_loss_column = col

        # Extract logged metrics
        if train_loss_column:
            train_loss = metrics_df[train_loss_column].dropna().values
        else:
            print("train_loss column not found")
            train_loss = []

        if zero_one_loss_column:
            zero_one_loss = metrics_df[zero_one_loss_column].dropna().values
        else:
            print("zero_one_loss column not found")
            zero_one_loss = []

        # Plot the metrics
        epochs = range(len(train_loss))

        plt.figure(figsize=(10, 5))

        # Plot training loss
        plt.subplot(1, 2, 1)
        if len(train_loss) > 0:
            plt.plot(epochs, train_loss, label="Train Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Across Epochs")
        plt.legend()

        # Plot zero-one loss
        plt.subplot(1, 2, 2)
        if len(zero_one_loss) > 0:
            plt.plot(epochs, zero_one_loss, label="Zero-One Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Zero-One Loss")
        plt.title("Zero-One Loss Across Epochs")
        plt.legend()

        # Save the plot
        plt.savefig("training_results_real_data.png")
        print("Training results saved to 'training_results_real_data.png'")


if __name__ == "__main__":
    unittest.main()
