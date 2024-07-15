"""
:file: __init__.py
This is the init file for the strategic cost.
"""

from .cost_function import CostFunction
from .norms import (
    CostNormL2,
    CostNormL1,
    CostMeanSquaredError,
    CostWeightedLoss,
    CostNormLInf,
)

__all__ = [
    "CostFunction",
    "CostNormL2",
    "CostNormL1",
    "CostMeanSquaredError",
    "CostWeightedLoss",
    "CostNormLInf",
]
