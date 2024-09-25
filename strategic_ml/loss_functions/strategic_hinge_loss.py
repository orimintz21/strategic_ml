# External imports
import torch
from torch import nn

# Internal imports
from strategic_ml.gsc import _LinearGP
from strategic_ml.models import LinearModel


class StrategicHingeLoss(nn.Module):
    """
    Implements the Strategic Hinge Loss (s-hinge), a modified version of the standard hinge loss 
    function that accounts for strategic behavior in classification settings. The s-hinge loss 
    anticipates and incorporates the strategic modifications that agents might apply to their 
    features to achieve better classification outcomes.

    The loss function maintains a differentiable form, allowing for optimization. The s-hinge loss 
    assumes that the model is linear, the delta is a Linear Delta, and the cost function is the L2 norm.

    The s-hinge loss is defined as:
    L(x, z, y; w, b) = max(0, 1 - y * (w^T * x + b) - 2 * cost_weight * z * y * (||w||_2 + ||b||_2))

    Reference: "Generalized Strategic Classification and the Case of Aligned Incentives"
    """
    def __init__(
        self,
        model: LinearModel,
        delta: _LinearGP,
    ) -> None:
        """
        Initializes the Strategic Hinge Loss class.

        Args:
            model (LinearModel): The linear model used in the strategic classification.
            delta (_LinearGP): The Linear Delta that accounts for strategic modifications.
        """
        super(StrategicHingeLoss, self).__init__()
        self.model = model
        assert isinstance(
            model, LinearModel
        ), f"model should be an instance of LinearModel, but it is {type(model)}"
        assert isinstance(
            delta, _LinearGP
        ), f"delta should be an instance of linear gp , but it is {type(delta)}"
        self.delta = delta

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the strategic hinge loss for a batch of input features and true labels.

        Args:
            x (torch.Tensor): Input features.
            y (torch.Tensor): True labels.

        Returns:
            torch.Tensor: The computed loss.
        """
        device = x.device
        dtype = x.dtype
        assert isinstance(
            self.model, LinearModel
        ), f"model should be an instance of LinearModel, but it is {type(self.model)}"
        z = self.delta.get_z(x, y)
        assert (
            z.dtype == x.dtype
        ), f"z should have the same dtype as x, but z has {z.dtype} dtype and x has {x.dtype} dtype"
        assert (
            z.device == x.device
        ), f"z should have the same device as x, but z has {z.device} device and x has {x.device} device"
        assert (
            z.shape[0] == x.shape[0]
        ), f"z should have the same number of samples as x, but z has {z.shape[0]} samples and x has {x.shape[0]} samples"
        w, b = self.model.get_weight_and_bias_ref()
        w = w.to(device=device, dtype=dtype)
        b = b.to(device=device, dtype=dtype)

        cost_weight = self.delta.get_cost_weight()

        linear_output = torch.matmul(x, w.T) + b
        w_norm = torch.linalg.norm(w, ord=2, dtype=dtype)
        b_norm = torch.abs(b).squeeze()  # b is a scalar

        norm = w_norm + b_norm

        additional_term = 2 * cost_weight * z * y * norm

        loss = torch.clamp(1 - y * linear_output - additional_term, min=0)

        return loss.mean()
