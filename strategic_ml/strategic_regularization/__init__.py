""" file: __init__.py
This is the init file for the strategic regularization.
"""

from strategic_ml.strategic_regularization.strategic_regularization import (
    _StrategicRegularization,
)
from strategic_ml.strategic_regularization.social_burden import SocialBurden

__all__ = [
    "_StrategicRegularization",
    "SocialBurden",
]
