""":file: cost_functions/norms.py

"""

import torch
 
from .cost_function import CostFunction

class CostNormL2(CostFunction):
    def __init__(self) -> None:
        super(CostNormL2, self).__init__()
    
    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        return torch.norm(x-x_prime, p='fro')
    

    
