# external imports
import torch

# internal imports
from strategic_ml.regularization.strategic_regularization import (
    _StrategicRegularization,
)


class ExpectedUtility(_StrategicRegularization):
    """
    Implements the Expected Utility regularization method, as described in the paper
    "Strategic Classification Made Practical."

    Expected Utility is a strategic regularization method that maximizes the expected
    utility of strategic agents by considering the difference between the model's
    predictions and the cost incurred by agents when they alter their features.
    The formula for the Expected Utility term is:
    Expected Utility = -mean(tanh(f(x) * temp) - cost)

    Attributes:
        tanh_temp (float): The temperature for the tanh function that normalizes the predictions.

    """

    def __init__(
        self,
        tanh_temp: float = 1.0,
    ) -> None:
        """
        Initializes the ExpectedUtility class.

        Args:
            tanh_temp (float): Temperature for the tanh function used to normalize predictions. Must be positive.
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
        Computes the Expected Utility regularization term.

        Args:
            delta_predictions (torch.Tensor): Predictions of the model after strategic modification.
            cost (torch.Tensor): Cost of the strategic agents (i.e., the cost of modification).

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

        # they should be on the same device
        assert (
            delta_predictions.device == cost.device
        ), "delta_predictions and cost must be on the same device"
        # they shouldn't be empty
        assert delta_predictions.shape[0] > 0, "delta_predictions should not be empty"

        # The predictions of the model aren't in scale of -1 to 1, so we need
        # to normalize them using tanh
        delta_predictions_tanh: torch.Tensor = torch.tanh(
            delta_predictions * self.tanh_temp
        )
        utility: torch.Tensor = delta_predictions_tanh - cost

        # The expected utility is the negative of the mean of the predictions
        return -utility.mean()
