import torch
import unittest
from torch.utils.data import DataLoader, TensorDataset
from strategic_ml.model_suit import ModelSuit
from strategic_ml.cost_functions.norms import CostNormL2
from strategic_ml.models.linear_strategic_model import LinearModel
from strategic_ml.gsc.linear_gp.linear_strategic_delta import LinearStrategicDelta
from strategic_ml.loss_functions.stratigic_hinge_loss import StrategicHingeLoss

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


class TestModelSuitIntegration(unittest.TestCase):

    def setUp(self):
        # Create dataset for testing
        self.x, self.y = create_strategic_need_movement()

        # Wrap the dataset in a DataLoader
        self.dataset = TensorDataset(self.x, self.y)
        self.dataloader = DataLoader(
            self.dataset, batch_size=len(self.dataset), shuffle=False
        )

        # Instantiate the strategic model, cost function, delta, and loss function
        self.strategic_model = LinearModel(in_features=2)
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
            "optimizer_class": torch.optim.SGD,
            "optimizer_params": {"lr": 0.01},  # Lower learning rate
            "max_epochs": 50,  # Increase the number of epochs
        }

        # Initialize ModelSuit
        self.model_suit = ModelSuit(
            model=self.strategic_model,
            delta=self.strategic_delta,
            loss_fn=self.loss_fn,
            regularization=None,
            train_loader=self.dataloader,
            validation_loader=self.dataloader,
            test_loader=self.dataloader,
            training_params=self.training_params,
        )

    def test_strategic_separable_needs_movement_hinge_loss(self) -> None:
        # Train the model using ModelSuit
        for epoch in range(self.training_params["max_epochs"]):
            for batch in self.dataloader:
                self.model_suit.training_step(batch, batch_idx=epoch)

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
