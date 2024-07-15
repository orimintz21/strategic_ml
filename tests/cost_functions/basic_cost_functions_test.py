import torch
import unittest
from strategic_ml.cost_functions.norms import (
    CostNormL2,
    CostMeanSquaredError,
    CostWeightedLoss,
    CostNormL1,
    CostNormLInf,
)


class TestBasicCostFunctions(unittest.TestCase):
    def setUp(self):
        self.x = torch.rand(10, 10)
        self.x_prime = torch.rand(10, 10)
        self.weights = torch.rand(10, 10)

    def test_CostNormL2(self):
        l2 = CostNormL2()
        output = l2.forward(self.x, self.x_prime)
        expected_output = torch.linalg.norm(self.x - self.x_prime, ord=2)
        self.assertTrue(torch.isclose(output, expected_output))

    def test_CostMeanSquaredError(self):
        mse = CostMeanSquaredError()
        output = mse.forward(self.x, self.x_prime)
        expected_output = torch.mean((self.x - self.x_prime) ** 2)
        self.assertTrue(torch.isclose(output, expected_output))

    def test_CostWeightedLoss(self):
        weighted_loss = CostWeightedLoss(self.weights)
        output = weighted_loss.forward(self.x, self.x_prime)
        expected_output = torch.sum(self.weights * (self.x - self.x_prime) ** 2)
        self.assertTrue(torch.isclose(output, expected_output))

    def test_CostNormL1(self):
        l1 = CostNormL1()
        output = l1.forward(self.x, self.x_prime)
        expected_output = torch.linalg.norm(self.x - self.x_prime, ord=1)
        self.assertTrue(torch.isclose(output, expected_output))

    def test_CostNormLInf(self):
        linf = CostNormLInf()
        output = linf.forward(self.x, self.x_prime)
        expected_output = torch.linalg.norm(self.x - self.x_prime, ord=float("inf"))
        self.assertTrue(torch.isclose(output, expected_output))


class TestL2Cost(unittest.TestCase):
    def test_compute_cost(self):
        cost_function = CostNormL2()
        x = torch.Tensor([1, 2, 3])
        x_prime = torch.Tensor([4, 5, 6])
        expected_cost = ((1 - 4) ** 2 + (2 - 5) ** 2 + (3 - 6) ** 2) ** 0.5
        self.assertEqual(cost_function(x, x_prime), expected_cost)


class TestL1Cost(unittest.TestCase):
    def test_compute_cost(self):
        cost_function: CostNormL1 = CostNormL1()
        x = torch.Tensor([1, 2, 3])
        x_prime = torch.Tensor([4, 5, 6])
        expected_cost = abs(1 - 4) + abs(2 - 5) + abs(3 - 6)
        cost_value = cost_function(x, x_prime)
        self.assertEqual(cost_value, expected_cost)


if __name__ == "__main__":
    unittest.main()
