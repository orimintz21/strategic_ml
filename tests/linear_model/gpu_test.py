# External Imports
import torch
import unittest

# Internal Imports
from strategic_ml.models import LinearModel, L2Regularization, L1Regularization


class TestGPUOnModel(unittest.TestCase):
    def test_model_on_device(self):
        device = torch.device("cuda")
        model = LinearModel(in_features=10).to(device)
        x = torch.randn(5, 10).to(device)
        output = model(x)
        # Assert output shape and device
        assert output.device == device
        # Test regularization
        l2_reg = L2Regularization(lambda_=0.1)
        reg_term = l2_reg(model)
        assert reg_term.device == device

    def test_L1Regularization_on_device(self):
        device = torch.device("cuda")
        model = LinearModel(in_features=10).to(device)
        x = torch.randn(5, 10).to(device)
        output = model(x)
        # Assert output shape and device
        assert output.device == device
        # Test regularization
        l1_reg = L1Regularization(lambda_=0.1)
        reg_term = l1_reg(model)
        assert reg_term.device == device


if __name__ == "__main__":
    unittest.main()
