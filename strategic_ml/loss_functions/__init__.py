"""
:file: __init__.py
This is the init file for the strategic cost.
"""

from .loss import _Loss
from .stratigic_hinge_loss import StrategicHingeLoss

__all__ = [
    "_Loss",
    "StrategicHingeLoss",
]
