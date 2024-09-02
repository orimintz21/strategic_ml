from typing import Any, Dict
import torch
import torch.nn as nn
import torch.optim as optim
import unittest

from strategic_ml import (
    CostNormL2,
    LinearStrategicModel,
    NonLinearStrategicDelta,
    NonLinearAdvDelta,
    LinearStrategicDelta,
    LinearAdvDelta,
)

VERBOSE: bool = True


TRAINING_PARAMS_SIMPLE: Dict[str, Any] = {
    "num_epochs": 500,
    "optimizer_class": optim.SGD,
    "optimizer_params": {
        "lr": 0.1,
    },
    "early_stopping": 30,
    "temp": 0.3,
}
TRAINING_PARAMS: Dict[str, Any] = {
    "num_epochs": 1500,
    "optimizer_class": optim.SGD,
    "optimizer_params": {
        "lr": 1.0,
    },
    "scheduler_class": optim.lr_scheduler.StepLR,
    "scheduler_params": {
        "step_size": 100,
        "gamma": 0.5,
    },
    "early_stopping": 300,
    "temp": 5,
}


def print_if_verbose(message: str) -> None:
    global VERBOSE
    if VERBOSE:
        print(message)


def create_strategic_separable_data():
    # Set the random seed for reproducibility
    torch.manual_seed(0)

    # Generate the first half of the points with the first index less than -5
    x1 = torch.cat((torch.randn(5, 1) - 10, torch.randn(5, 1)), dim=1)

    # Generate the second half of the points with the first index greater than 5
    x2 = torch.cat((torch.randn(5, 1) + 10, torch.randn(5, 1)), dim=1)

    # Concatenate both parts to create the dataset
    x = torch.cat((x1, x2), dim=0)

    # Create labels: 1 for the first half, -1 for the second half
    y1 = torch.ones(5, 1)
    y2 = -torch.ones(5, 1)
    y = torch.cat((y1, y2), dim=0)
    return x, y


def create_adv_need_movement():
    x_p = torch.Tensor([[1, -1], [1, 1]])
    y_p = torch.Tensor([[1], [1]])
    x_n = torch.Tensor([[-1, 10]])
    y_n = torch.Tensor([[-1]])

    x = torch.cat((x_p, x_n), dim=0)
    y = torch.cat((y_p, y_n), dim=0)
    return x, y


def create_strategic_need_movement():
    """This function creates a dataset where the strategic model needs to move
    to correctly classify the points if we have a strategic user

    Returns:
        _type_: _description_
    """
    x_p = torch.Tensor([[1, -1], [1, 1]])
    y_p = torch.Tensor([[1], [1]])
    x_n = torch.Tensor([[-1, -1], [-1, 1]])
    y_n = torch.Tensor([[-1], [-1]])

    x = torch.cat((x_p, x_n), dim=0)
    y = torch.cat((y_p, y_n), dim=0)
    return x, y


class NonLinearModel(nn.Module):
    def __init__(self, in_features: int) -> None:
        super(NonLinearModel, self).__init__()
        self.fc1 = nn.Linear(in_features, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class TestLinearStrategicDelta(unittest.TestCase):

if __name__ == "__main__":
    unittest.main()
