import torch
from torch import nn
from strategic_ml.gsc.generalized_strategic_delta import GSC
from loss import _Loss


class StrategicHingeLoss(_Loss):
    def __init__(
        self, model, delta: GSC, regularization_lambda: float = 0.01
    ):  # TODO Add the SC base model, and add typying
        """
        Initialize the Strategic Hinge Loss class.

        :param delta: Function to modify features strategically.
        :param regularization_lambda: Regularization parameter.
        :param model: Initial model parameters.
        """
        super().__init__(delta, regularization_lambda, model)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, y_tilde: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass to compute the strategic hinge loss.

        :param x: Input features as a torch.tensor.
        :param y: True labels as a torch.tensor.
        :param y_tilde: Modified labels as a torch.tensor.
        :param w: Model weights as a torch.tensor.
        :return: Computed loss as a torch.tensor.
        """
        raise NotImplementedError()

    def compute_loss(self, X: torch.torch.Tensor, y) -> float:
        """
        Compute the strategic hinge loss for the given inputs and labels.

        :param X: Input features as a NumPy array.
        :param y: True labels as a NumPy array.
        :return: Computed loss as a float.
        """
        raise NotImplementedError()

    def compute_gradient(
        self, X: torch.torch.Tensor, y: torch.torch.Tensor
    ) -> torch.torch.Tensor:
        """
        Compute the gradient of the strategic hinge loss.

        :param X: Input features as a NumPy array.
        :param y: True labels as a NumPy array.
        :return: Computed gradient as a NumPy array.
        """
        raise NotImplementedError()
