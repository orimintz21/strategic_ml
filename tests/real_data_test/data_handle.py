# External imports
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

# Internal imports


def get_data_set(
    data_path: str,
    seed: int,
    test_frac: float,
    val_frac_from_train: float,
    dtype: torch.dtype,
) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    from sklearn.preprocessing import RobustScaler
    import numpy as np
    import math

    data = pd.read_csv(data_path)
    rob_scaler = RobustScaler()
    data["scaled_amount"] = rob_scaler.fit_transform(
        data["Amount"].values.reshape(-1, 1)
    )
    # We will not use the time column
    data.drop(["Time", "Amount"], axis=1, inplace=True)
    scaled_amount = data["scaled_amount"]
    data.drop(["scaled_amount"], axis=1, inplace=True)
    data.insert(0, "scaled_amount", scaled_amount)

    # We use -1,+1 as labels and not 0,1
    data["Class"].replace({1: -1, 0: 1}, inplace=True)

    data = data.sample(frac=1, random_state=seed)

    fraud_data = data[data["Class"] == -1]
    non_fraud_data = data[data["Class"] == 1][: len(fraud_data)]
    normal_dist_data = pd.concat([fraud_data, non_fraud_data])
    data = normal_dist_data.sample(frac=1, random_state=seed)

    Y = data["Class"].values
    X = data.drop("Class", axis=1).values

    X_dim = len(X[0])
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    X /= math.sqrt(X_dim)

    # We will now split the data into train, val and test
    test_size = int(test_frac * len(data))
    X_train = X[:-test_size]
    y_train = Y[:-test_size]
    X_test = X[-test_size:]
    y_test = Y[-test_size:]
    X_val = X_train[-int(val_frac_from_train * len(X_train)) :]
    y_val = y_train[-int(val_frac_from_train * len(X_train)) :]
    X_train = X_train[: -int(val_frac_from_train * len(X_train))]
    y_train = y_train[: -int(val_frac_from_train * len(y_train))]

    # We will now convert the data into torch tensors
    X_train = torch.tensor(X_train, dtype=dtype)
    y_train = torch.tensor(y_train, dtype=dtype).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=dtype)
    y_val = torch.tensor(y_val, dtype=dtype).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=dtype)
    y_test = torch.tensor(y_test, dtype=dtype).unsqueeze(1)

    # We will now create the dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    return train_dataset, val_dataset, test_dataset


def load_data(
    data_path: str,
    seed: int,
    test_frac: float,
    val_frac_from_train: float,
    batch_size_train: int,
    batch_size_val: int,
    batch_size_test: int,
    dtype: torch.dtype,
    train_num_workers: int = 0,
    val_num_workers: int = 0,
    test_num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_dataset, val_dataset, test_dataset = get_data_set(
        data_path,
        seed,
        test_frac,
        val_frac_from_train,
        dtype,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=False,
        num_workers=train_num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=val_num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=test_num_workers,
    )

    return train_loader, val_loader, test_loader
