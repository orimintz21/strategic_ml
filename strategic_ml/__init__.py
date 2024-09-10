"""
This is the initialization module for the strategic_ml package.

The strategic_ml package provides a set of tools and models for strategic classification, 
including cost functions, regularization methods, loss functions, and model classes. 
The package is designed for working with both linear and non-linear strategic models.

Authors:
    Ori Mintz & Kfir Cohen

Version:
    0.1.0
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

from .regularization import _StrategicRegularization, SocialBurden, Recourse

from .loss_functions import _Loss, StrategicHingeLoss

from .gsc import (
    _GSC,
    _LinearGP,
    LinearAdvDelta,
    LinearStrategicDelta,
    LinearNoisyLabelDelta,
    _NonLinearGP,
    NonLinearStrategicDelta,
    NonLinearNoisyLabelDelta,
    NonLinearAdvDelta,
)

from .models import LinearStrategicModel

from .model_suit import ModelSuit


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
    "SocialBurden",
    "Recourse",
    # Loss functions
    "_Loss",
    "StrategicHingeLoss",
    # GSC
    "_GSC",
    "_LinearGP",
    "LinearAdvDelta",
    "LinearStrategicDelta",
    "LinearNoisyLabelDelta",
    "_NonLinearGP",
    "NonLinearStrategicDelta",
    "NonLinearNoisyLabelDelta",
    "NonLinearAdvDelta",
    # Models
    "LinearStrategicModel",
    # Model Suit
    "ModelSuit",
]  # List of modules and functions to be imported when using 'from strategic_ml import *'
