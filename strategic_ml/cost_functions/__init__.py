"""
:file: __init__.py
This is the init file for the strategic cost.
"""

from .cost_function import _CostFunction
from .norms import (
    CostNormL2,
    CostNormL1,
    CostMeanSquaredError,
    CostWeightedLoss,
    CostNormLInf,
)

__all__ = [
    "_CostFunction",
    "CostNormL2",
    "CostNormL1",
    "CostMeanSquaredError",
    "CostWeightedLoss",
    "CostNormLInf",
]
