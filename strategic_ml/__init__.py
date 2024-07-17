"""
:file: __init__.py
This is the init file for the strategic_ml package.
"""

# Package metadata
__version__ = "0.1.0"
__author__ = "Ori Mintz & Kfir Cohen"

# Main modules and functions
from .cost_functions import (
    _CostFunction,
    CostNormL2,
    CostNormL1,
    CostMeanSquaredError,
    CostWeightedLoss,
    CostNormLInf,
)
from .strategic_regularization import _StrategicRegularization

from .loss_functions import _Loss

from .gsc import _GSC, _GP

from .models import _StrategicModel


# Functionality of the package
__all__: list = [
    # Costs
    "_CostFunction",
    "CostNormL2",
    "CostNormL1",
    "CostMeanSquaredError",
    "CostWeightedLoss",
    "CostNormLInf",
    # Strategic Regularization
    "_StrategicRegularization",
    # Loss functions
    "_Loss",
    # GSC
    "_GSC",
    "_GP",
    # Models
    "_StrategicModel",
]  # List of modules and functions to be imported when using 'from strategic_ml import *'
