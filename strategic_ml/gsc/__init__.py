"""
This module initializes the Generalized Strategic Classification (GSC) package.

It provides various delta strategies for both linear and non-linear models, including adversarial, strategic, 
and noisy label scenarios. The deltas are calculated using the GP formula, and the models 
are designed to handle large datasets efficiently.

Classes provided:
    - _GSC: The base class for all generalized strategic classification models.
    - _LinearGP, _NonLinearGP: Classes for calculating deltas in linear and non-linear models, respectively.
    - LinearAdvDelta, LinearStrategicDelta, LinearNoisyLabelDelta: Delta models for linear strategic users.
    - NonLinearAdvDelta, NonLinearStrategicDelta, NonLinearNoisyLabelDelta: Delta models for non-linear strategic users.
"""

from strategic_ml.gsc.generalized_strategic_delta import _GSC
from strategic_ml.gsc.linear_gp import (
    _LinearGP,
    LinearAdvDelta,
    LinearStrategicDelta,
    LinearNoisyLabelDelta,
)
from strategic_ml.gsc.non_linear_gp import (
    _NonLinearGP,
    NonLinearStrategicDelta,
    NonLinearNoisyLabelDelta,
    NonLinearAdvDelta,
)

__all__ = [
    "_GSC",
    "_LinearGP",
    "LinearAdvDelta",
    "LinearStrategicDelta",
    "LinearNoisyLabelDelta",
    "_NonLinearGP",
    "NonLinearStrategicDelta",
    "NonLinearNoisyLabelDelta",
    "NonLinearAdvDelta",
]
