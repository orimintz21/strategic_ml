import numpy as np
import torch
from torch import nn
import torch.optim as optim
from typing import Union
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import unittest
from pytorch_lightning.loggers import CSVLogger

SEED = 0


def reset_seed():
    torch.manual_seed(SEED)


def gen_custom_normal_data(
    num_samples: int,
    x_dim: int,
    pos_mean: Union[np.ndarray, float],
    pos_std: Union[np.ndarray, float],
    neg_mean: Union[np.ndarray, float],
    neg_std: Union[np.ndarray, float],
    pos_noise_frac: float = 0.01,
    neg_noise_frac: float = 0.01,
) -> DataLoader:
    reset_seed()
    pos_samples_num = num_samples // 2
    neg_samples_num = num_samples - pos_samples_num
    posX = (
        torch.randn((pos_samples_num, x_dim), dtype=torch.float32) * pos_std + pos_mean
    )
    negX = (
        torch.randn((neg_samples_num, x_dim), dtype=torch.float32) * neg_std + neg_mean
    )

    X = torch.cat((posX, negX), 0).to(torch.float32)  # Ensure X is float32

    Y = torch.unsqueeze(
        torch.cat(
            (
                torch.from_numpy(
                    np.random.choice(
                        [1, -1], len(posX), p=[1 - pos_noise_frac, pos_noise_frac]
                    )
                )
                .float()
                .to(torch.float32),  # Convert to float32
                torch.from_numpy(
                    np.random.choice(
                        [-1, 1], len(posX), p=[1 - neg_noise_frac, neg_noise_frac]
                    )
                )
                .float()
                .to(torch.float32),  # Convert to float32
            ),
            0,
        ),
        1,
    )

    if torch.cuda.is_available():
        batch_size = 1000
    else:
        batch_size = 100
    dataset = TensorDataset(X, Y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
