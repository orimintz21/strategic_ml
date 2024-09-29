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


from .models import (
    LinearModel,
    _LinearRegularization,
    LinearL2Regularization,
    LinearL1Regularization,
    LinearElasticNetRegularization,
    LinearInfRegularization,
)

from .gsc import (
    _GSC,
    IdentityDelta,
    _LinearGP,
    LinearAdvDelta,
    LinearStrategicDelta,
    _NonLinearGP,
    NonLinearStrategicDelta,
    NonLinearAdvDelta,
)

from .loss_functions import StrategicHingeLoss

from .regularization import (
    _StrategicRegularization,
    SocialBurden,
    Recourse,
    ExpectedUtility,
)

from .model_suit import ModelSuit

from .utils import visualize_data_and_delta_2D, visualize_data_and_delta_1D

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
    "_NonLinearGP",
    "NonLinearStrategicDelta",
    "NonLinearAdvDelta",
    # Linear Model
    "LinearModel",
    "LinearL2Regularization",
    "LinearL1Regularization",
    "LinearElasticNetRegularization",
    "LinearInfRegularization",
    "_LinearRegularization",
    # Model Suit
    "ModelSuit",
    # Utils
    "visualize_data_and_delta_2D",
    "visualize_data_and_delta_1D",
]  # List of modules and functions to be imported when using 'from strategic_ml import *'
