"""
:file: __init__.py
This is the init file for the strategic_ml package.
"""

# Package metadata
__version__ = "0.1.0"
__author__ = "Ori Mintz & Kfir Cohen"

# Main modules and functions
from .cost_functions import (
    CostFunction,
    CostNormL2,
    CostNormL1,
    CostMeanSquaredError,
    CostWeightedLoss,
    CostNormLInf,
)

# Functionality of the package
# TODO: Add functionality here
__all__: list = [
    "CostFunction",
    "CostNormL2",
    "CostNormL1",
    "CostMeanSquaredError",
    "CostWeightedLoss",
    "CostNormLInf",
]  # List of modules and functions to be imported when using 'from strategic_ml import *'
