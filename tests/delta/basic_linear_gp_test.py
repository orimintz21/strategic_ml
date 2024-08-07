import torch
import unittest

from strategic_ml import (
    CostNormL2,
    LinearStrategicModel,
    LinearStrategicDelta,
    LinearAdvDelta,
    LinearNoisyLabelDelta
)


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

def create_strategic_noisy_data():
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

    # Create noisy labels
    bernoulli_tensor = torch.bernoulli(0.5 * torch.ones(y.shape))
    y_noisy = y * bernoulli_tensor
    return x, y_noisy


class TestLinearStrategicDelta(unittest.TestCase):

    def test_demo(self) -> None:
        """This test just checks that the flow of the code is correct.
        It does not check the correctness of the code.
        """
        self.x, self.y = create_strategic_separable_data()
        # Create a strategic model
        strategic_model = LinearStrategicModel(in_features=2)

        # Create a cost function
        cost = CostNormL2()

        # Create a strategic delta
        strategic_delta: LinearStrategicDelta = LinearStrategicDelta(
            cost, strategic_model
        )

        # Train the strategic model
        optimizer = torch.optim.SGD(strategic_model.parameters(), lr=0.001)
        loss = torch.nn.MSELoss()

        strategic_model.train()
        for _ in range(200):
            with torch.no_grad():
                delta_move: torch.Tensor = strategic_delta(self.x)
            x_prime = delta_move
            optimizer.zero_grad()
            prediction = strategic_model(x_prime)
            output = loss(prediction, self.y)
            output.backward()
            optimizer.step()
        print("The strategic model has been trained")

        # validate the the distance between the two points is less than 1
        for x, y in zip(self.x, self.y):
            x = x.unsqueeze(0)
            x_prime_test = strategic_delta.forward(x)
            print(cost(x, x_prime_test))
            self.assertEqual(torch.sign(strategic_model(x)), y)
            self.assertTrue(cost(x, x_prime_test) < 1)

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
