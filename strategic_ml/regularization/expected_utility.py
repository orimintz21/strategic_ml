# external imports
import torch

# internal imports
from strategic_ml.regularization.strategic_regularization import (
    _StrategicRegularization,
)


class ExpectedUtility(_StrategicRegularization):
    """
    ExpectedUtility class.
    This is the implementation of the Expected Utility regularization method that
    is described in the paper "Strategic Classification Made Practical".
    Expected Utility is a strategic regularization method that tries to maximize
    the expected utility of the strategic agents, meaning the difference between
    the predictions of the model and the cost of the strategic agents,
    i.e., the difference between the predictions after the movement of the
    strategic agents and the cost of the movement.

    Parent Class: _StrategicRegularization
    """

    def __init__(
        self,
        tanh_temp: float = 1.0,
    ) -> None:
        """
        Constructor for the ExpectedUtility class.
        :param tanh_temp: The temperature for the tanh function that normalizes the predictions.
        """
        super(ExpectedUtility, self).__init__()
        assert tanh_temp > 0, "The temperature for the tanh should be positive"
        self.tanh_temp = tanh_temp

    def forward(
        self,
        delta_predictions: torch.Tensor,
        cost: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        This is the forward method of the ExpectedUtility class.
        This function calculates the Expected Utility regularization term.
        The Expected Utility regularization term is calculated by the following
        formula:
        Expected Utility = -mean(prediction - cost)
        where the prediction is the predictions of the model after the movement
        of the strategic agents and is normalized using tanh.

        Args:
            delta_predictions (torch.Tensor): The predictions of the model on the delta of x.
            cost (torch.Tensor): The cost of the strategic agents (i.e., the cost of the movement).

        Returns:
            torch.Tensor: The Expected Utility regularization term.
        """

        # they should have the same batch size
        assert (
            delta_predictions.shape[0] == cost.shape[0]
        ), "delta_predictions and cost must have the same batch size"
        assert (
            delta_predictions.shape[1] == 1
        ), "delta_predictions must have only one output"
        assert cost.shape[1] == 1, "cost must have only one output"

        # The predictions of the model aren't in scale of -1 to 1, so we need
        # to normalize them using tanh
        delta_predictions_tanh: torch.Tensor = torch.tanh(
            delta_predictions * self.tanh_temp
        )
        utility: torch.Tensor = delta_predictions_tanh - cost

        # The expected utility is the negative of the mean of the predictions
        return -utility.mean()
