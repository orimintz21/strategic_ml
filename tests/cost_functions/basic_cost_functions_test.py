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
        print(output)
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


if __name__ == "__main__":
    unittest.main()
