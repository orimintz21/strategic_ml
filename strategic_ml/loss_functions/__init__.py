"""
This module initializes the strategic loss functions package.

It provides loss functions designed for strategic classification, all of which inherit from the _Loss class.
Currently, the package includes the StrategicHingeLoss function, which is tailored for hinge loss in strategic scenarios.
"""

from .loss import _Loss
from .stratigic_hinge_loss import StrategicHingeLoss

__all__ = [
    "_Loss",
    "StrategicHingeLoss",
]
