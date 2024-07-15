"""_summary_

Raises:
    NotImplementedError: _description_

Returns:
    _type_: _description_
"""

import torch
from torch import nn


class CostFunction(nn.Module):
    def __init__(self) -> None:
        super(CostFunction, self).__init__()

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def __call__(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        return self.forward(x, x_prime)
