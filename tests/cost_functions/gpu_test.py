import torch
import unittest

from strategic_ml.cost_functions import (
    CostNormL2,
    CostMeanSquaredError,
    CostWeightedLoss,
    CostNormL1,
    CostNormLInf,
)


def test_cost_function_on_device(self, cost_function, device):
    x = torch.randn(10, requires_grad=True).to(device)
    x_prime = torch.randn(10, requires_grad=True).to(device)
    cost = cost_function(x, x_prime)
    cost.backward()


class TestGPUOnCost(unittest.TestCase):
    def test_CostNormL2(self):
        test_cost_function_on_device(self, CostNormL2(), torch.device("cuda"))
        test_cost_function_on_device(self, CostNormL2(), torch.device("cpu"))

    def test_CostMeanSquaredError(self):
        test_cost_function_on_device(self, CostMeanSquaredError(), torch.device("cuda"))
        test_cost_function_on_device(self, CostMeanSquaredError(), torch.device("cpu"))

    def test_CostWeightedLoss(self):
        test_cost_function_on_device(
            self, CostWeightedLoss(torch.randn(10)), torch.device("cuda")
        )
        test_cost_function_on_device(
            self, CostWeightedLoss(torch.randn(10)), torch.device("cpu")
        )

    def test_CostNormL1(self):
        test_cost_function_on_device(self, CostNormL1(), torch.device("cuda"))
        test_cost_function_on_device(self, CostNormL1(), torch.device("cpu"))

    def test_CostNormLInf(self):
        test_cost_function_on_device(self, CostNormLInf(), torch.device("cuda"))
        test_cost_function_on_device(self, CostNormLInf(), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
