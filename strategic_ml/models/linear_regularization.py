# External Imports
from abc import ABC, abstractmethod
import torch

# Internal Imports
from strategic_ml.models.linear_model import LinearModel


class _LinearRegularization(ABC):
    """
    Abstract base class for linear regularization methods applied to linear models.
    This class should be inherited to create custom regularization methods for linear models.
    """

    @abstractmethod
    def __call__(self, linear_model: LinearModel) -> torch.Tensor:
        """
        Applies the regularization to the given linear model.

        Args:
            linear_model (LinearModel): The linear model to apply the regularization to.

        Returns:
            torch.Tensor: The regularization term.
        """
        pass


class LinearL2Regularization(_LinearRegularization):
    """
    Implements L2 regularization for a linear model. The L2 regularization term is calculated as:
    L2 Regularization = lambda * (||weight||_2 + ||bias||_2)

    Attributes:
        lambda_ (float): The lambda parameter for L2 regularization.
    """

    def __init__(self, lambda_: float) -> None:
        """
        Initializes the L2Regularization class.

        Args:
            lambda_ (float): The lambda parameter for L2 regularization.
        """
        self.lambda_: float = lambda_

    def __call__(self, linear_model: LinearModel) -> torch.Tensor:
        """
        Applies L2 regularization to the given linear model.

        Args:
            linear_model (LinearModel): The linear model to apply the regularization to.

        Returns:
            torch.Tensor: The L2 regularization term.
        """
        weight, bias = linear_model.get_weight_and_bias_ref()

        device = weight.device
        assert device == bias.device, "Weight and bias must be on the same device"

        return self.lambda_ * (torch.norm(weight, p=2) + torch.norm(bias, p=2))


class LinearL1Regularization(_LinearRegularization):
    """
    Implements L1 regularization for a linear model. The L1 regularization term is calculated as:
    L1 Regularization = lambda * (||weight||_1 + ||bias||_1)

    Attributes:
        lambda_ (float): The lambda parameter for L1 regularization.
    """

    def __init__(self, lambda_: float) -> None:
        """
        Initializes the L1Regularization class.

        Args:
            lambda_ (float): The lambda parameter for L1 regularization.
        """
        self.lambda_ = lambda_

    def __call__(self, linear_model: LinearModel) -> torch.Tensor:
        """
        Applies L1 regularization to the given linear model.

        Args:
            linear_model (LinearModel): The linear model to apply the regularization to.

        Returns:
            torch.Tensor: The L1 regularization term.
        """
        weight, bias = linear_model.get_weight_and_bias_ref()

        device = weight.device
        assert device == bias.device, "Weight and bias must be on the same device"

        return self.lambda_ * (torch.norm(weight, p=1) + torch.norm(bias, p=1))


class LinearElasticNetRegularization(_LinearRegularization):
    """
    Implements Elastic Net regularization for a linear model, which combines L1 and L2 regularization.
    The Elastic Net regularization term is calculated as:
    Elastic Net Regularization = lambda * (l1_loss + l2_loss)
    where l1_loss = (1 - alpha) * (||weight||_1 + ||bias||_1)
    and l2_loss = alpha * (||weight||_2 + ||bias||_2)

    Attributes:
        lambda_ (float): The lambda parameter for Elastic Net regularization.
        alpha (float): The parameter for combining L1 and L2 regularization. Must be between 0 and 1.

    """

    def __init__(self, lambda_: float, alpha: float) -> None:
        """
        Initializes the ElasticNetRegularization class.

        Args:
            lambda_ (float): Parameter for the total regularization strength.
            alpha (float): Parameter for combining L1 and L2 regularization. Must be between 0 and 1.
        """
        self.lambda_ = lambda_
        self.alpha = alpha
        assert 0 <= alpha <= 1, "alpha must be between 0 and 1"

    def __call__(self, linear_model: LinearModel) -> torch.Tensor:
        """
        Applies Elastic Net regularization to the given linear model.

        Args:
            linear_model (LinearModel): The linear model to apply the regularization to.

        Returns:
            torch.Tensor: The Elastic Net regularization term.
        """
        # Get the weight and bias of the model
        weight, bias = linear_model.get_weight_and_bias_ref()

        device = weight.device
        assert device == bias.device, "Weight and bias must be on the same device"

        # Calculate the L1 and L2 regularization terms
        l1_loss: torch.Tensor = (1 - self.alpha) * (
            torch.norm(weight, p=1) + torch.norm(bias, p=1)
        )
        l2_loss: torch.Tensor = self.alpha * (
            torch.norm(weight, p=2) + torch.norm(bias, p=2)
        )

        # Return the combined ElasticNet loss
        return self.lambda_ * (l1_loss + l2_loss)


class LinearInfRegularization(_LinearRegularization):
    """
    Implements Inf regularization for a linear model. The Inf regularization term is calculated as:
    Inf Regularization = lambda * (||weight||_inf + ||bias||_inf)


    Attributes:
        lambda_ (float): The lambda parameter for Inf regularization.
    """

    def __init__(self, lambda_: float) -> None:
        """
        Initializes the InfRegularization class.

        Args:
            lambda_ (float): The lambda parameter for Inf regularization.
        """

        self.lambda_ = lambda_

    def __call__(self, linear_model: LinearModel) -> torch.Tensor:
        """
        Applies Inf regularization to the given linear model.

        Args:
            linear_model (LinearModel): The linear model to apply the regularization to.

        Returns:
            torch.Tensor: The Inf regularization term.
        """
        weight, bias = linear_model.get_weight_and_bias_ref()

        device = weight.device
        assert device == bias.device, "Weight and bias must be on the same device"

        return self.lambda_ * (
            torch.norm(weight, p=float("inf")) + torch.norm(bias, p=float("inf"))
        )
