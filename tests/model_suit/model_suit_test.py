import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
from strategic_ml.model_suit import ModelSuit
from strategic_ml.loss_functions import StrategicHingeLoss
from strategic_ml.gsc.linear_gp import LinearStrategicDelta
from strategic_ml.models.linear_strategic_model import LinearStrategicModel
from strategic_ml.cost_functions.norms import CostNormL2
from strategic_ml.regularization.social_burden import SocialBurden


class TestModelSuit(unittest.TestCase):

    def setUp(self):
        # Create a small dataset with 4 points
        data_points = torch.tensor(
            [[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=torch.float32
        )
        labels = torch.tensor([1, -1, 1, -1], dtype=torch.float32).unsqueeze(1)
        self.dataset = TensorDataset(data_points, labels)
        self.dataloader = DataLoader(self.dataset, batch_size=2, shuffle=True)

        # Instantiate the real model, cost function, delta, and loss function
        self.model = LinearStrategicModel(in_features=2)
        self.cost = CostNormL2(dim=1)
        self.delta = LinearStrategicDelta(cost=self.cost, strategic_model=self.model)
        self.loss_fn = StrategicHingeLoss(model=self.model, delta=self.delta)

        # Define the training parameters
        self.training_params = {
            "optimizer_class": torch.optim.SGD,
            "optimizer_params": {"lr": 0.01},
            "max_epochs": 10,
        }

        # Initialize the ModelSuit instance
        self.model_suit = ModelSuit(
            model=self.model,
            delta=self.delta,
            loss_fn=self.loss_fn,
            regularization=None,
            train_loader=self.dataloader,
            validation_loader=self.dataloader,
            test_loader=self.dataloader,
            training_params=self.training_params,
        )

    def test_initialization(self):
        # Test if the ModelSuit object is initialized correctly
        self.assertIsInstance(self.model_suit, ModelSuit)
        self.assertEqual(self.model_suit.model, self.model)
        self.assertEqual(self.model_suit.delta, self.delta)
        self.assertEqual(self.model_suit.loss_fn, self.loss_fn)
        self.assertEqual(self.model_suit.train_loader, self.dataloader)

    def test_forward(self):
        # Test the forward pass with real data
        x = torch.randn(4, 2)  # 4 samples with 2 features each
        output = self.model_suit.forward(x)

        # Assert that the output is a tensor
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(
            output.shape, (4, 1)
        )  # The output should have 4 samples, 1 output each

    def test_training_step(self):
        # Take a batch from the dataloader
        batch = next(iter(self.dataloader))

        # Run a training step and check that the loss is calculated
        loss = self.model_suit.training_step(batch, batch_idx=0)
        self.assertIsInstance(loss, torch.Tensor)

    def test_validation_step(self):
        # Take a batch from the dataloader
        batch = next(iter(self.dataloader))

        # Run a validation step and check that the output contains val_loss and zero_one_loss
        output = self.model_suit.validation_step(batch, batch_idx=0)
        self.assertIn("val_loss", output)
        self.assertIn("zero_one_loss", output)

    def test_test_step(self):
        # Take a batch from the dataloader
        batch = next(iter(self.dataloader))

        # Run a test step and check that the output contains test_loss and zero_one_loss
        output = self.model_suit.test_step(batch, batch_idx=0)
        self.assertIn("test_loss", output)
        self.assertIn("zero_one_loss", output)

    def test_configure_optimizers(self):
        # Call the configure_optimizers method and ensure it returns a valid optimizer
        optimizer = self.model_suit.configure_optimizers()
        self.assertIsInstance(optimizer, torch.optim.SGD)

    def test_assert_regularization_with_strategic_hinge_loss(self):
        regularization = SocialBurden()
        # Initialize the ModelSuit instance with StrategicHingeLoss and regularization
        model_suit = ModelSuit(
            model=self.model,
            delta=self.delta,
            loss_fn=self.loss_fn,
            regularization=regularization,  # This should trigger the assertion
            train_loader=self.dataloader,
            validation_loader=self.dataloader,
            test_loader=self.dataloader,
            training_params=self.training_params,
        )

        # Take a batch from the dataloader
        batch = next(iter(self.dataloader))

        # Call training_step to trigger the assertion
        with self.assertRaises(AssertionError) as context:
            model_suit.training_step(batch, batch_idx=0)

        # Check if the error message matches the expected assertion message
        self.assertEqual(
            str(context.exception),
            "Regularization is not supported for StrategicHingeLoss",
        )


if __name__ == "__main__":
    unittest.main()
