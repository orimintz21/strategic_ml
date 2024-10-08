import torch
import unittest
from typing import Tuple

from strategic_ml import (
    SocialBurden,
    Recourse,
    ExpectedUtility,
    CostNormL2,
    LinearStrategicDelta,
    LinearModel,
)

VERBOSE = False


def print_if_verbose(message) -> None:
    global VERBOSE
    if VERBOSE:
        print(message)


class TestRegularization(unittest.TestCase):
    def create_data_all_true(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x_p = torch.Tensor([[1, -1], [1, 1]])
        y_p = torch.Tensor([[1], [1]])
        x_n = torch.Tensor([[-1, -1], [-1, 1]])
        y_n = torch.Tensor([[-1], [-1]])

        x = torch.cat((x_p, x_n), dim=0)
        y = torch.cat((y_p, y_n), dim=0)
        return x, y

    def create_data_false_label(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x_p = torch.Tensor([[1, -1], [1, 1]])
        y_p = torch.Tensor([[1], [1]])
        x_n = torch.Tensor([[-1, -1], [-1, 1], [1, 1]])
        y_n = torch.Tensor([[-1], [-1], [-1]])

        x = torch.cat((x_p, x_n), dim=0)
        y = torch.cat((y_p, y_n), dim=0)
        return x, y

    def create_data_false_prediction(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x_p = torch.Tensor([[1, -1], [1, 1], [-1, -1]])
        y_p = torch.Tensor([[1], [1], [1]])
        x_n = torch.Tensor([[-1, -1], [-1, 1]])
        y_n = torch.Tensor([[-1], [-1]])

        x = torch.cat((x_p, x_n), dim=0)
        y = torch.cat((y_p, y_n), dim=0)
        return x, y

    def create_data_false_label_false_prediction(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_p = torch.Tensor([[1, -1], [1, 1]])
        y_p = torch.Tensor([[1], [1]])
        x_n = torch.Tensor([[-1, -1], [-1, 1], [2, 2]])
        y_n = torch.Tensor([[-1], [-1], [-1]])

        x = torch.cat((x_p, x_n), dim=0)
        y = torch.cat((y_p, y_n), dim=0)
        return x, y

    def create_data_positive_that_does_not_move(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_p = torch.Tensor([[1, -1], [1, 1], [2, 2]])
        y_p = torch.Tensor([[1], [1], [1]])
        x_n = torch.Tensor([[-1, -1], [-1, 1]])
        y_n = torch.Tensor([[-1], [-1]])

        x = torch.cat((x_p, x_n), dim=0)
        y = torch.cat((y_p, y_n), dim=0)
        return x, y

    def create_data_false_label_moves_to_true(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_p = torch.Tensor([[1, -1], [1, 1]])
        y_p = torch.Tensor([[1], [1], [1]])
        x_n = torch.Tensor([[-1, -1], [-1, 1], [0, 0]])
        y_n = torch.Tensor([[-1], [-1], [-1]])

        x = torch.cat((x_p, x_n), dim=0)
        y = torch.cat((y_p, y_n), dim=0)
        return x, y

    def setUp(self) -> None:
        super().setUp()
        self.cost = CostNormL2(dim=1)
        self.model = LinearModel(
            2, weight=torch.Tensor([[2, 0]]), bias=torch.Tensor([-2.5])
        )
        self.delta = LinearStrategicDelta(cost=self.cost, strategic_model=self.model)
        self.social_burden = SocialBurden(linear_delta=self.delta)
        self.recourse = Recourse(model=self.model, sigmoid_temp=1e5)
        self.expected_utility: ExpectedUtility = ExpectedUtility(tanh_temp=1e5)

    def tearDown(self) -> None:
        super().tearDown()
        self.model.set_weight_and_bias(torch.Tensor([[2, 0]]), torch.Tensor([-2.5]))

    def test_social_burden_all_true(self):
        # Create the data
        x, y = self.create_data_all_true()

        x_prime = self.delta(x)
        predictions = self.model(x_prime)
        # We expect that we will get the max cost of the true samples, which is 0.255 (due to epsilon)
        social_burden_value = self.social_burden(x, y)

        self.assertAlmostEqual(social_burden_value.item(), 0.510, delta=0.05)

    def test_social_burden_false_label(self):
        # Create the data
        x, y = self.create_data_false_label()
        x_prime = self.delta(x)
        predictions = self.model(x_prime)
        # We expect that the false samples will not effect the social burden
        social_burden_value = self.social_burden(x, y)

        self.assertAlmostEqual(social_burden_value.item(), 0.510, delta=0.05)

    def test_social_burden_false_prediction(self):
        # Create the data
        x, y = self.create_data_false_prediction()

        x_prime = self.delta(x)
        predictions = self.model(x_prime)
        """
        We have 5 examples in the data,
        2 are true and are classified as false (without movement) and as true (with movement), that are 0.255 far from the decision boundary,
        1 is true and is classified as false (without movement) and as false (with movement), that is 2.255 far from the decision boundary,
        2 are false and are classified as false (without movement) and as false (with movement),
        """
        social_burden_value = self.social_burden(x, y)

        self.assertAlmostEqual(social_burden_value.item(), 2.765, delta=0.05)

    def test_social_burden_data_positive_that_does_not_move(self):
        # Create the data
        x, y = self.create_data_positive_that_does_not_move()

        x_prime = self.delta(x)
        predictions = self.model(x_prime)
        """
        We have 5 examples in the data,
        2 are true and are classified as false (without movement) and as true (with movement), that are 0.255 far from the decision boundary,
        1 is true and is classified as true (without movement) and as true (with movement), so it does not effect the social burden,
        2 are false and are classified as false (without movement) and as false (with movement),
        """
        social_burden_value = self.social_burden(x, y)

        self.assertAlmostEqual(social_burden_value.item(), 0.510, delta=0.05)

    def test_social_burden_training(self) -> None:

        # Create the data
        x, y = self.create_data_false_prediction()
        loss_fn = torch.nn.BCEWithLogitsLoss()
        # train the model
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        for _ in range(100):
            optimizer.zero_grad()
            x_prime = self.delta(x)
            predictions = self.model(x_prime)
            regularization = self.social_burden(x, y)
            loss = loss_fn(predictions, y)
            loss_with_reg = loss + 0.1 * regularization
            loss_with_reg.backward()
            optimizer.step()

        x_prime_final = self.delta(x)
        predictions_final = self.model(x_prime_final)
        print_if_verbose(self.model.get_weight_and_bias())
        print_if_verbose(f"X: {x}")
        print_if_verbose(f"Y: {y}")
        print_if_verbose(f"Prediction: {predictions_final}")
        print_if_verbose(f"Prediction sign: {torch.sign(predictions_final)}")
        print_if_verbose(f"Delta: {x_prime_final}")
        print_if_verbose(f"Cost: {self.cost(x, x_prime_final)}")
        print_if_verbose(f"Social Burden: {self.social_burden(x ,y)}")

    def test_recourse_all_true(self):
        # Create the data
        x, y = self.create_data_all_true()

        x_prime = self.delta(x)
        predictions = self.model(x_prime)
        recourse_value = self.recourse(x, predictions)

        """
        There are 4 examples in the data, 
        2 are true and are classified as false (without movement) and as true (with movement),
        2 are false and are classified as false (without movement) and as false (with movement), 
        So we have 2 examples that are classified as false and stays false.
        """
        self.assertAlmostEqual(recourse_value.item(), 2.0, delta=0.05)

    def test_recourse_false_label(self):
        # Create the data
        x, y = self.create_data_false_label()
        x_prime = self.delta(x)
        predictions = self.model(x_prime)
        recourse_value = self.recourse(x, predictions)

        """
        There are 5 examples in the data, 
        2 are true and are classified as false (without movement) and as true (with movement),
        1 is false and is classified as false (without movement) and as true (with movement),
        2 are false and are classified as false (without movement) and as false (with movement), 
        So we have 2 examples that are classified as false and stays false.
        """
        self.assertAlmostEqual(recourse_value.item(), 2.0, delta=0.05)

    def test_recourse_false_prediction(self):
        # Create the data
        x, y = self.create_data_false_prediction()

        x_prime = self.delta(x)
        predictions = self.model(x_prime)
        # We expect that the example with the false label will not effect the social burden
        recourse_value = self.recourse(x, predictions)

        """
        There are 5 examples in the data, 
        2 are true and are classified as false (without movement) and as true (with movement),
        1 is true and is classified as false (without movement) and as false (with movement),
        2 are false and are classified as false (without movement) and as false (with movement), 
        So we have 2 examples that are classified as false and stays false.
        """
        self.assertAlmostEqual(recourse_value.item(), 3.0, delta=0.05)

    def test_recourse_false_label_false_prediction(self):
        # Create the data
        x, y = self.create_data_false_label_false_prediction()

        x_prime = self.delta(x)
        predictions = self.model(x_prime)
        # We expect that the example with the false label will not effect the social burden
        recourse_value = self.recourse(x, predictions)

        """
        There are 5 examples in the data, 
        2 are true and are classified as false (without movement) and as true (with movement),
        1 is false and is classified as true (without movement) and as true (with movement),
        2 are false and are classified as false (without movement) and as false (with movement), 
        So we have 2 examples that are classified as false and stays false.
        """
        self.assertAlmostEqual(recourse_value.item(), 2, delta=0.05)

    def test_recourse_false_labe_moves_to_true(self):
        # Create the data
        x, y = self.create_data_false_label_moves_to_true()

        x_prime = self.delta(x)
        predictions = self.model(x_prime)
        # We expect that the example with the false label will not effect the social burden
        recourse_value = self.recourse(x, predictions)

        """
        There are 5 examples in the data, 
        2 are true and are classified as false (without movement) and as true (with movement),
        1 is false and is classified as false (without movement) and as true (with movement),
        2 are false and are classified as false (without movement) and as false (with movement), 
        So we have 2 examples that are classified as false and stays false.
        """
        self.assertAlmostEqual(recourse_value.item(), 2, delta=0.05)

    def test_recourse_training(self) -> None:

        # Create the data
        x, y = self.create_data_false_prediction()
        loss_fn = torch.nn.BCEWithLogitsLoss()
        # train the model
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        for _ in range(100):
            optimizer.zero_grad()
            x_prime = self.delta(x)
            predictions = self.model(x_prime)
            regularization = self.recourse(x, predictions)
            loss = loss_fn(predictions, y)
            loss_with_reg = loss + 0.1 * regularization
            loss_with_reg.backward()
            optimizer.step()

        x_prime_final = self.delta(x)
        predictions_final = self.model(x_prime_final)
        print_if_verbose(self.model.get_weight_and_bias())
        print_if_verbose(f"X: {x}")
        print_if_verbose(f"Y: {y}")
        print_if_verbose(f"Prediction: {predictions_final}")
        print_if_verbose(f"Prediction sign: {torch.sign(predictions_final)}")
        print_if_verbose(f"Delta: {x_prime_final}")
        print_if_verbose(f"Cost: {self.cost(x, x_prime_final)}")
        print_if_verbose(f"Recourse: {self.recourse(x, predictions_final)}")

    def test_expected_utility_all_true(self):
        # Create the data
        x, y = self.create_data_all_true()

        x_prime = self.delta(x)
        predictions = self.model(x_prime)
        costs = self.cost(x, x_prime).unsqueeze(1)
        expected_utility_value = self.expected_utility(
            delta_predictions=predictions, cost=costs
        )

        """
        There are 4 examples in the data, 
        2 are true and are classified as false (without movement) and as true (with movement),
        and they moved 0.255 from the decision boundary
        2 are false and are classified as false (without movement) and as false (with movement), 
        and they did not move
        therefore the expected utility is -(2(1-0.255) + 2(-1))/4 = 0.1275 
        """
        self.assertAlmostEqual(expected_utility_value.item(), 0.1275, delta=0.05)


if __name__ == "__main__":
    unittest.main()
