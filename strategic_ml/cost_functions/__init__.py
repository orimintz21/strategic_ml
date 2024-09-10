"""
This module initializes the strategic cost functions package.
It provides various norm cost functions, all of which inherit from the _CostFunction class.
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
