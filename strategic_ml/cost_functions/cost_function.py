
import torch
from torch import nn

class CostFunction(nn.Module):
    def __init__(self) -> None:
        super(CostFunction, self).__init__()
    
    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        
        raise NotImplementedError()