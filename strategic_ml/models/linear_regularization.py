# External Imports
from abc import ABC, abstractmethod
import torch

# Internal Imports
from strategic_ml.models.linear_model import LinearModel


class _LinearRegularization(ABC):
    """This is the _LinearRegularization class.
    This class is an abstract class for linear regularization.
    The linear regularization is a regularization that is applied only to the
    linear model. Use this class in the model suit.
    To create more linear regularization's, inherit from this class and implement
    the __call__ method.

    Parent Class: ABC
    """

    @abstractmethod
    def __call__(self, linear_model: LinearModel) -> torch.Tensor:
        """This is the __call__ method of the _LinearRegularization class.

        Args:
            linear_model (LinearModel): the linear model that we want to apply the regularization to.

        Returns:
            torch.Tensor: the regularization term.
        """
        pass


class L2Regularization(_LinearRegularization):
    """
    This is the L2Regularization class.
    This class calculates the L2 regularization term for the linear model.
    The L2 regularization term is calculated by the following formula:
    L2 Regularization = lambda*(||weight||_2 + ||bias||_2)

    Parent Class: _LinearRegularization
    """

    def __init__(self, lambda_: float) -> None:
        """Initializer for the L2Regularization class.

        Args:
            lambda_ (float): The lambda parameter for the L2 regularization.
        """
        self.lambda_ = lambda_

    def __call__(self, linear_model: LinearModel) -> torch.Tensor:
        """This is the __call__ method of the L2Regularization class.

        Args:
            linear_model (LinearModel): the linear model that we want to apply the regularization to.

        Returns:
            torch.Tensor: the regularization term.
        """
        weight, bias = linear_model.get_weight_and_bias_ref()
        return self.lambda_ * (torch.norm(weight, p=2) + torch.norm(bias, p=2))


class L1Regularization(_LinearRegularization):
    """This is the L1Regularization class.
    This class calculates the L1 regularization term for the linear model.
    The L1 regularization term is calculated by the following formula:
    L1 Regularization = lambda*(||weight||_1 + ||bias||_1)

    Parent Class: _LinearRegularization
    """

    def __init__(self, lambda_: float) -> None:
        """Initializer for the L1Regularization class.

        Args:
            lambda_ (float): the lambda parameter for the L1 regularization.
        """
        self.lambda_ = lambda_

    def __call__(self, linear_model: LinearModel) -> torch.Tensor:
        """This is the __call__ method of the L1Regularization class.

        Args:
            linear_model (LinearModel): the linear model that we want to apply
            the regularization to.

        Returns:
            torch.Tensor: the regularization term.
        """
        weight, bias = linear_model.get_weight_and_bias_ref()
        return self.lambda_ * (torch.norm(weight, p=1) + torch.norm(bias, p=1))


class ElasticNetRegularization(_LinearRegularization):
    """This is the ElasticNetRegularization class.
    This class calculates the Elastic Net regularization term for the linear model.
    The Elastic Net regularization term is calculated by the following formula:
    Elastic Net Regularization = lambda*(l1_loss + l2_loss)
    where l1_loss = (1 - alpha)*(||weight||_1 + ||bias||_1)
    and l2_loss = alpha*(||weight||_2 + ||bias||_2)

    Parent Class: _LinearRegularization
    """

    def __init__(self, lambda_: float, alpha: float) -> None:
        """Initializer for the ElasticNetRegularization class.

        Args:
            lambda_ (float): Parameter for the total value
            alpha (float): Parameter for combining L1 and L2 regularization
            must be between 0 and 1
        """
        self.lambda_ = lambda_
        self.alpha = alpha
        assert 0 <= alpha <= 1, "alpha must be between 0 and 1"

    def __call__(self, linear_model: LinearModel) -> torch.Tensor:
        """This is the __call__ method of the ElasticNetRegularization class.

        Args:
            linear_model (LinearModel): the linear model that we want to apply
            the regularization to.

        Returns:
            torch.Tensor: the regularization term.
        """
        # Get the weight and bias of the model
        weight, bias = linear_model.get_weight_and_bias_ref()

        # Calculate the L1 and L2 regularization terms
        l1_loss: torch.Tensor = (1 - self.alpha) * (
            torch.norm(weight, p=1) + torch.norm(bias, p=1)
        )
        l2_loss: torch.Tensor = self.alpha * (
            torch.norm(weight, p=2) + torch.norm(bias, p=2)
        )

        # Return the combined ElasticNet loss
        return self.lambda_ * (l1_loss + l2_loss)


class InfRegularization(_LinearRegularization):
    """This is the InfRegularization class.
    This class calculates the Inf regularization term for the linear model.
    The Inf regularization term is calculated by the following formula:
    Inf Regularization = lambda*(||weight||_inf + ||bias||_inf)

    Parent Class: _LinearRegularization
    """

    def __init__(self, lambda_: float) -> None:
        """Initializer for the InfRegularization class.

        Args:
            lambda_ (float): the lambda parameter for the Inf regularization.
        """

        self.lambda_ = lambda_

    def __call__(self, linear_model: LinearModel) -> torch.Tensor:
        """This is the __call__ method of the InfRegularization class.

        Args:
            linear_model (LinearModel): the linear model that we want to apply
            the regularization to.

        Returns:
            torch.Tensor: the regularization term.
        """
        weight, bias = linear_model.get_weight_and_bias_ref()
        return self.lambda_ * (
            torch.norm(weight, p=float("inf")) + torch.norm(bias, p=float("inf"))
        )
