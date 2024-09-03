import torch
import unittest
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset

from strategic_ml.cost_functions.norms import CostNormL2
from strategic_ml.models.linear_strategic_model import LinearStrategicModel
from strategic_ml.gsc.linear_gp.linear_strategic_delta import LinearStrategicDelta
from strategic_ml.loss_functions.stratigic_hinge_loss import StrategicHingeLoss
from strategic_ml.trainer.strategic_trainer import create_trainer

VERBOSE: bool = True


def print_if_verbose(message: str) -> None:
    global VERBOSE
    if VERBOSE:
        print(message)


def create_strategic_need_movement():
    x_p = torch.Tensor([[1, -1], [1, 1]])
    y_p = torch.Tensor([[1], [1]])
    x_n = torch.Tensor([[-1, -1], [-1, 1]])
    y_n = torch.Tensor([[-1], [-1]])

    x = torch.cat((x_p, x_n), dim=0)
    y = torch.cat((y_p, y_n), dim=0)
    return x, y


class TestStrategicClassificationModuleIntegration(unittest.TestCase):

    def setUp(self):
        self.x, self.y = create_strategic_need_movement()

        # Wrap the dataset in a DataLoader
        self.dataset = TensorDataset(self.x, self.y)
        self.dataloader = DataLoader(
            self.dataset, batch_size=len(self.dataset), shuffle=False
        )

        # Instantiate the strategic model, cost function, delta, and loss function
        self.strategic_model = LinearStrategicModel(in_features=2)
        self.cost = CostNormL2(dim=1)
        self.strategic_delta = LinearStrategicDelta(
            cost=self.cost,
            strategic_model=self.strategic_model,
            cost_weight=1.0,
        )
        self.loss_fn = StrategicHingeLoss(
            model=self.strategic_model,
            delta=self.strategic_delta,
        )

        # Define the training parameters
        self.training_params = {
            "lr": 4.0,
            "max_epochs": 1501,
            "devices": 1,
            "precision": 32,
            "accelerator": "cpu",
            "logger": False,  # Disable logging for testing
        }

        # Create the trainer using the StrategicClassificationModule
        self.trainer, self.model = create_trainer(
            model=self.strategic_model,
            strategic_regularization=None,
            loss_fn=self.loss_fn,
            gsc=self.strategic_delta,
            training_params=self.training_params,
        )

    def test_strategic_separable_needs_movement_hinge_loss(self) -> None:
        # Train the model using the PyTorch Lightning Trainer
        self.trainer.fit(self.model, self.dataloader)

        print("The strategic model has been trained")

        # Check the model's predictions after training
        successful = 0
        for x, y in zip(self.x, self.y):
            x = x.unsqueeze(0)
            x_prime_test = self.strategic_delta.forward(x)
            print_if_verbose(
                f"""
                x = {x},
                delta = {x_prime_test},
                cost = {self.cost(x, x_prime_test)},
                y = {y},
                x pred {(self.strategic_model(x))},
                x_prime = {(self.strategic_model(x_prime_test))}
                """
            )
            if torch.sign(self.strategic_model(x_prime_test)) == y:
                successful += 1

        # Assert that all predictions are correct
        self.assertEqual(successful, len(self.x), "Not all predictions are correct.")


if __name__ == "__main__":
    unittest.main()
