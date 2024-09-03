import unittest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from strategic_ml.cost_functions.norms import CostNormL2
from strategic_ml.gsc.linear_strategic_delta import LinearStrategicDelta
from strategic_ml.model_suit.linear_trainer import LinearTrainer
from strategic_ml.models.linear_strategic_model import LinearStrategicModel


class TestLinearTrainer(unittest.TestCase):

    def setUp(self):
        # Create a simple dataset
        X = torch.tensor([[0.5, 0.5], [0.6, 0.6], [0.7, 0.7]], dtype=torch.float32)
        y = torch.tensor([[1], [-1], [-1]], dtype=torch.float32)

        dataset = TensorDataset(X, y)
        self.train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
        self.val_loader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Instantiate the real model and other components for the trainer
        model = LinearStrategicModel(in_features=2)
        loss = (
            torch.nn.BCEWithLogitsLoss()
        )  # Binary Cross-Entropy Loss for binary classification
        cost = CostNormL2()  # Use a real cost function
        gsc = LinearStrategicDelta(
            cost=cost, strategic_model=model
        )  # Use the LinearStrategicDelta instance

        # Training parameters
        training_params = {
            "optimizer": torch.optim.SGD,
            "learning_rate": 0.01,
            "optimizer_params": {},
            "scheduler": None,
            "scheduler_params": {},
            "early_stopping": {
                "patience": 5,
                "min_delta": 0.001,
                "monitor": "val_loss",
                "restore_best_weights": True,
            },
        }

        # Instantiate the trainer with the real model and strategic delta
        self.trainer = LinearTrainer(
            model=model,
            strategic_regularization=None,  # Assuming no strategic regularization for simplicity
            loss=loss,
            cost=cost,
            gsc=gsc,
            device="cpu",
            training_params=training_params,
        )

    def test_train_batch(self):
        # Test if train_batch runs without errors
        for batch in self.train_loader:
            self.trainer.train_batch(batch)
        self.assertTrue(True)  # If no exceptions were raised, the test passes

    def test_fit(self):
        # Test if fit runs for a few epochs without errors
        self.trainer.fit(self.train_loader, self.val_loader, epochs=2)
        self.assertTrue(True)  # If no exceptions were raised, the test passes

    def test_evaluate(self):
        # Test if evaluate runs without errors
        metrics = self.trainer.evaluate(self.val_loader)
        self.assertIsInstance(metrics, dict)  # The result should be a dictionary


if __name__ == "__main__":
    unittest.main()
