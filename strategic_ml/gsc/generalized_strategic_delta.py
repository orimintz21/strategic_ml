""":file: generalized_strategic_delta.py
This is the interface for all of GSC models.
When creating a new model in this framework you will need to inherit from
it.
A GSC - stands for generalized strategic is based on the paper:
'Generalized Strategic Classification and the Case of Aligned Incentives'

The general idea of the GSC is that a GSC gets a model, and change the input
x to x' based on the GSC's type.
"""

# Imports
import torch
from torch import nn
from typing import Optional
from strategic_ml.models import _Model



# Implementation
class _GSC(nn.Module):
    def __init__(self, model: _Model) -> None:
        super(_GSC, self).__init__()
        self.model : _Model= model

    def forward(self, x: torch.Tensor, y: torch.Tensor, y_tilde: Optional[torch.Tensor] = None) -> torch.Tensor:


