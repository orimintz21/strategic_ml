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

from .regularization import (
    _StrategicRegularization,
    SocialBurden,
    Recourse,
    ExpectedUtility,
)

from .loss_functions import StrategicHingeLoss

from .gsc import (
    _GSC,
    IdentityDelta,
    _LinearGP,
    LinearAdvDelta,
    LinearStrategicDelta,
    LinearNoisyLabelDelta,
    _NonLinearGP,
    NonLinearStrategicDelta,
    NonLinearNoisyLabelDelta,
    NonLinearAdvDelta,
)

from .models import (
    LinearModel,
    L2Regularization,
    L1Regularization,
    ElasticNetRegularization,
    InfRegularization,
    _LinearRegularization,
)

from .model_suit import ModelSuit

from .utils import visualize_linear_classifier_2D

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
    "ExpectedUtility",
    # Loss functions
    "StrategicHingeLoss",
    # GSC
    "_GSC",
    "IdentityDelta",
    "_LinearGP",
    "LinearAdvDelta",
    "LinearStrategicDelta",
    "LinearNoisyLabelDelta",
    "_NonLinearGP",
    "NonLinearStrategicDelta",
    "NonLinearNoisyLabelDelta",
    "NonLinearAdvDelta",
    # Linear Model
    "LinearModel",
    "L2Regularization",
    "L1Regularization",
    "ElasticNetRegularization",
    "InfRegularization",
    "_LinearRegularization",
    # Model Suit
    "ModelSuit",
    # Utils
    "visualize_linear_classifier_2D",
]  # List of modules and functions to be imported when using 'from strategic_ml import *'
