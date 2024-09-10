"""
This module initializes the strategic regularization package.

The package provides various regularization techniques designed for strategic classification, 
including social burden regularization and recourse. All regularization methods inherit from 
the _StrategicRegularization class.
"""

from strategic_ml.regularization.strategic_regularization import (
    _StrategicRegularization,
)
from strategic_ml.regularization.social_burden import SocialBurden
from strategic_ml.regularization.recourse import Recourse

__all__ = ["_StrategicRegularization", "SocialBurden", "Recourse"]
