""":file: strategic_regularization.py
This file contains the regularization module, which is the base class for all
the strategic regularization methods in the strategic_ml package.
Strategic regularization does not means the regular regularization (i.e., L1, L2, etc.).
It is a concept that is described in the paper "Strategic Classification Made Practical".
The meaning of strategic regularization is to add a regularization term to the loss function
that will encourage the model to take to account the effect of the model on 
the strategic agents.
"""

import torch
from torch import nn


class _StrategicRegularization(nn.Module):
    def __init__(self) -> None:
        """
        Constructor for the _StrategicRegularization class.
        """
        super(_StrategicRegularization, self).__init__()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """The forward method of the strategic regularization. Each strategic


        Raises:
            NotImplementedError:

        Returns:
            torch.Tensor: a 1x1 tensor
        """
        raise NotImplementedError()
