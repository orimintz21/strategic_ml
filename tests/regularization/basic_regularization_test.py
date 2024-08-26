import torch
import unittest

from strategic_ml import (
    SocialBurden,
    CostNormL2,
    LinearStrategicDelta,
    LinearStrategicModel,
)


class TestSocialBurden(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.cost = CostNormL2(dim=1)
        self.model = LinearStrategicModel(
            2, weight=torch.Tensor([[2, 0]]), bias=torch.Tensor([-2.5])
        )
        self.delta = LinearStrategicDelta(cost=self.cost, strategic_model=self.model)
        self.social_burden = SocialBurden(cost_fn=self.cost)

    def test_social_burden_all_true(self):
        # Create the data

        x_p = torch.Tensor([[1, -1], [1, 1]])
        y_p = torch.Tensor([[1], [1]])
        x_n = torch.Tensor([[-1, -1], [-1, 1]])
        y_n = torch.Tensor([[-1], [-1]])

        x = torch.cat((x_p, x_n), dim=0)
        y = torch.cat((y_p, y_n), dim=0)

        x_prime = self.delta(x)
        predictions = self.model(x_prime)
        # We expect that we will get the max cost of the true samples, which is 0.255 (due to epsilon)
        social_burden_value = self.social_burden(x, x_prime, y, predictions)

        self.assertAlmostEqual(social_burden_value.item(), 0.255, delta=0.05)

    def test_social_burden_false_label(self):
        # Create the data

        x_p = torch.Tensor([[1, -1], [1, 1]])
        y_p = torch.Tensor([[1], [1]])
        x_n = torch.Tensor([[-1, -1], [-1, 1], [1, 1]])
        y_n = torch.Tensor([[-1], [-1], [-1]])

        x = torch.cat((x_p, x_n), dim=0)
        y = torch.cat((y_p, y_n), dim=0)

        x_prime = self.delta(x)
        predictions = self.model(x_prime)
        # We expect that the false samples will not effect the social burden
        social_burden_value = self.social_burden(x, x_prime, y, predictions)

        self.assertAlmostEqual(social_burden_value.item(), 0.255, delta=0.05)

    def test_social_burden_false_prediction(self):
        # Create the data

        x_p = torch.Tensor([[1, -1], [1, 1], [-1, -1]])
        y_p = torch.Tensor([[1], [1], [1]])
        x_n = torch.Tensor([[-1, -1], [-1, 1]])
        y_n = torch.Tensor([[-1], [-1]])

        x = torch.cat((x_p, x_n), dim=0)
        y = torch.cat((y_p, y_n), dim=0)

        x_prime = self.delta(x)
        predictions = self.model(x_prime)
        # We expect that the example with the false label will not effect the social burden
        social_burden_value = self.social_burden(x, x_prime, y, predictions)

        self.assertAlmostEqual(social_burden_value.item(), 0.255, delta=0.05)

    def test_social_burden_training(self) -> None:

        # Create the data

        x_p = torch.Tensor([[1, -1], [1, 1], [1, 3]])
        y_p = torch.Tensor([[1], [1], [1]])
        x_n = torch.Tensor([[-1, -1], [-1, 1]])
        y_n = torch.Tensor([[-1], [-1]])

        x = torch.cat((x_p, x_n), dim=0)
        y = torch.cat((y_p, y_n), dim=0)

        loss_fn = torch.nn.BCEWithLogitsLoss()
        # train the model
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        for _ in range(100):
            optimizer.zero_grad()
            x_prime = self.delta(x)
            predictions = self.model(x_prime)
            regularization = self.social_burden(x, x_prime, y, predictions)
            loss = loss_fn(predictions, y)
            loss_with_reg = loss_fn(predictions, y) + 0.1 * regularization
            loss_with_reg.backward()
            optimizer.step()

        x_prime_final = self.delta(x)
        predictions_final = self.model(x_prime_final)
        print(self.model.get_weights_and_bias())
        print(f"X: {x}")
        print(f"Y: {y}")
        print(f"Prediction: {predictions_final}")
        print(f"Prediction sign: {torch.sign(predictions_final)}")
        print(f"Delta: {x_prime_final}")
        print(f"Cost: {self.cost(x, x_prime_final)}")
        print(
            f"Social Burden: {self.social_burden(x, x_prime_final, y, predictions_final)}"
        )


if __name__ == "__main__":
    unittest.main()
