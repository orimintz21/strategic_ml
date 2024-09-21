import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
import unittest
from strategic_ml.model_suit.strategic_trainer import create_trainer
from strategic_ml.model_suit.strategic_classification_module import (
    StrategicClassificationModule,
)
from strategic_ml.model_suit.strategic_callbacks import StrategicAdjustmentCallback
from strategic_ml.models.linear_strategic_model import LinearModel


# Define a simple linear model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(2, 1)  # 2 input features, 1 output

    def forward(self, x):
        return self.linear(x)


# Dummy strategic regularization (for testing)
class DummyStrategicRegularization(torch.nn.Module):
    def forward(self, inputs, x_prime, targets, outputs):
        return torch.tensor(0.0)  # No regularization


# Dummy GSC (Generalized Strategic Delta)
class DummyGSC(torch.nn.Module):
    def forward(self, inputs, targets):
        return inputs  # No modification


class TestStrategicTrainer(unittest.TestCase):
    def setUp(self):
        # Create a small dataset with 4 points
        data_points = torch.tensor(
            [[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=torch.float32
        )
        labels = torch.tensor([1, -1, 1, -1], dtype=torch.float32).unsqueeze(1)
        self.dataset = TensorDataset(data_points, labels)
        self.dataloader = DataLoader(self.dataset, batch_size=2, shuffle=True)

        # Define the training parameters for CPU
        self.training_params = {
            "lr": 0.01,
            "max_epochs": 10,
            "devices": 1,  # Set to 1 to ensure CPU usage
            "precision": 32,
            "accelerator": "cpu",
        }

        # Instantiate the model, strategic regularization, loss function, and GSC
        # self.model = SimpleModel()
        self.model = LinearModel(in_features=2)
        self.strategic_regularization = DummyStrategicRegularization()
        self.loss_fn = torch.nn.MSELoss()  # Using MSE for simplicity
        self.gsc = DummyGSC()

        # Create the trainer using the function from strategic_trainer.py
        self.trainer, self.model = create_trainer(
            model=self.model,
            strategic_regularization=self.strategic_regularization,
            loss_fn=self.loss_fn,
            gsc=self.gsc,
            training_params=self.training_params,
            callbacks=[StrategicAdjustmentCallback()],  # Example callback
        )

    def test_trainer_initialization(self):
        """Test that the trainer is created successfully."""
        self.assertIsInstance(self.trainer, pl.Trainer)

    def test_trainer_run(self):
        """Test the trainer on the small dataset."""
        try:
            self.trainer.fit(self.model, self.dataloader)
            print("Trainer created and ran successfully on the small dataset.")
        except Exception as e:
            self.fail(f"Trainer encountered an error: {e}")


if __name__ == "__main__":
    unittest.main()
