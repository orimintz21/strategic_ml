import os
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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


TRAINING_PARAMS: Dict[str, Any] = {
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


def print_if_verbose(message: str) -> None:
    global VERBOSE
    if VERBOSE:
        print(message)


def create_adv() -> DataLoader:
    """This function creates a dataset where the a perfect score can be achieved
    with this decision boundary (for adv user):
    weight = [3.5, -3] and bias ~= 9
    """

    x_p = torch.Tensor([[1, -1], [1, 1]])
    y_p = torch.Tensor([[1], [1]])
    x_n = torch.Tensor([[-1, 10]])
    y_n = torch.Tensor([[-1]])

    x = torch.cat((x_p, x_n), dim=0)
    y = torch.cat((y_p, y_n), dim=0)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=2, shuffle=False)


def create_strategic() -> DataLoader:
    """This function creates a dataset where the strategic model needs to move
    to correctly classify the points if we have a strategic user

    The best decision boundary is weight = [1, 0] and bias ~= -1.3
    """
    x_p = torch.Tensor([[1, -1], [1, 1]])
    y_p = torch.Tensor([[1], [1]])
    x_n = torch.Tensor([[-1, -1], [-1, 1]])
    y_n = torch.Tensor([[-1], [-1]])

    x = torch.cat((x_p, x_n), dim=0)
    y = torch.cat((y_p, y_n), dim=0)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=2, shuffle=False)


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
    def setUp(self) -> None:
        self.save_dir = "tests/non_linear_data/delta"
        self.adv_loader = create_adv()
        self.strategic_loader = create_strategic()
        self.perf_strategic_linear = LinearStrategicModel(
            in_features=2, weight=torch.Tensor([[1, 0]]), bias=torch.Tensor([-1.03])
        )
        self.perf_adv_linear = LinearStrategicModel(
            in_features=2, weight=torch.Tensor([[3.5, -3]]), bias=torch.Tensor([9])
        )
        self.non_linear_model = NonLinearModel(in_features=2)
        self.cost = CostNormL2(dim=1)
        self.cost_weight = 1.0

    def tearDown(self) -> None:
        super().tearDown()
        self.perf_strategic_linear.set_weight_and_bias(
            torch.Tensor([[1, 0]]), torch.Tensor([-1.03])
        )
        self.perf_adv_linear.set_weight_and_bias(
            torch.Tensor([[3.5, -3]]), torch.Tensor([9])
        )
        self.non_linear_model = NonLinearModel(in_features=2)

    def test_non_linear_strategic_delta(self) -> None:
        save_dir = os.path.join(self.save_dir, "test_non_linear_strategic_delta")
        strategic_delta = NonLinearStrategicDelta(
            self.cost,
            self.perf_strategic_linear,
            cost_weight=self.cost_weight,
            save_dir=save_dir,
            training_params=TRAINING_PARAMS,
        )
        strategic_delta_linear = LinearStrategicDelta(
            self.cost,
            self.perf_strategic_linear,
            cost_weight=self.cost_weight,
        )

        strategic_delta.train(self.strategic_loader)
        for batch_idx, data in enumerate(self.strategic_loader):
            x_batch, y_batch = data
            x_prime_batch = strategic_delta.load_x_prime(batch_idx)
            for x, y, x_prime in zip(x_batch, y_batch, x_prime_batch):
                x = x.unsqueeze(0)
                y = y.unsqueeze(0)
                x_prime = x_prime.unsqueeze(0)
                print_if_verbose(
                    f"Strategic: x: {x}, y: {y}, x_prime: {x_prime}, cost: {self.cost(x, x_prime)}, prediction: {self.perf_strategic_linear(x_prime)}"
                )
                # We assume that the non-linear model is able to find good points
                self.assertEqual(torch.sign(self.perf_strategic_linear(x_prime)), y)
                # We assume that the non-linear delta is close to the linear delta
                self.assertTrue(
                    torch.allclose(strategic_delta_linear(x), x_prime, atol=0.1)
                )

    def test_non_linear_adv_delta(self) -> None:
        save_dir = os.path.join(self.save_dir, "test_non_linear_adv_delta")
        adv_delta = NonLinearAdvDelta(
            self.cost,
            self.perf_adv_linear,
            cost_weight=self.cost_weight,
            save_dir=save_dir,
            training_params=TRAINING_PARAMS,
        )
        adv_delta_linear = LinearAdvDelta(
            self.cost,
            self.perf_adv_linear,
            cost_weight=self.cost_weight,
        )

        adv_delta.train(self.adv_loader)
        for batch_idx, data in enumerate(self.adv_loader):
            x_batch, y_batch = data
            x_prime_batch = adv_delta.load_x_prime(batch_idx)
            for x, y, x_prime in zip(x_batch, y_batch, x_prime_batch):
                x = x.unsqueeze(0)
                y = y.unsqueeze(0)
                x_prime = x_prime.unsqueeze(0)
                print_if_verbose(
                    f"Adv: x: {x}, y: {y}, x_prime: {x_prime}, cost: {self.cost(x, x_prime)}, prediction: {self.perf_adv_linear(x_prime)}"
                )
                # We assume that the non-linear model is able to find good points
                self.assertEqual(torch.sign(self.perf_adv_linear(x_prime)), y)
                # We assume that the non-linear delta is close to the linear delta
                self.assertTrue(torch.allclose(adv_delta_linear(x, y), x_prime, atol=0.1))
    
    def test_non_linear_with_non_linear_model(self) -> None:
        save_dir = os.path.join(self.save_dir, "test_non_linear_with_non_linear_model")
        strategic_delta = NonLinearStrategicDelta(
            self.cost,
            self.non_linear_model,
            cost_weight=self.cost_weight,
            save_dir=save_dir,
            training_params=TRAINING_PARAMS,
        )
        NUM_OF_EPOCHS = 400
        TRAIN_DELTA_EVERY = 20
        optimizer = torch.optim.Adam(self.non_linear_model.parameters(), lr=0.01)
        loss_fn = torch.nn.BCEWithLogitsLoss()


        for epoch in range(NUM_OF_EPOCHS):
            if epoch % TRAIN_DELTA_EVERY == 0:
                strategic_delta.train(self.strategic_loader)

            for batch_idx, data in enumerate(self.strategic_loader):
                x_batch, y_batch = data
                x_prime_batch = strategic_delta.load_x_prime(batch_idx)
                optimizer.zero_grad()
                prediction = self.non_linear_model(x_prime_batch)
                loss = loss_fn(prediction, y_batch)
                loss.backward()
                optimizer.step()
                print_if_verbose(f"Epoch {epoch}, batch {batch_idx}, loss {loss}")


if __name__ == "__main__":
    unittest.main()
