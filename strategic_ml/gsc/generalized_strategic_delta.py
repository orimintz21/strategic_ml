""":file: generalized_strategic_delta.py
"""

# Imports
import torch
from torch import nn

# Implementation
class GSC(nn.Module):
    """
    This is the interface for all of GSC models.
    When creating a new model in this framework you will need to inherit from
    it.
    A GSC - stands for generalized strategic is based on the paper:
    'Generalized Strategic Classification and the Case of Aligned Incentives'

    The general idea of the GSC is that a GSC gets a model, and change the input
    x to x' based on the GSC's type.
    """
    
    def __init__(self) -> None:
        super(GSC, self).__init__()
    


    def set_cost()-> None:
        pass

norm = torch.norm(3,'fro', 3)
