"""
This module initializes the non-linear Generalized Strategic Classification (GSC) components.
It provides various delta strategies for non-linear models, all of which inherit from the _NonLinearGP class.
"""

from strategic_ml.gsc.non_linear_gp.non_linear_gp import _NonLinearGP
from strategic_ml.gsc.non_linear_gp.non_linear_strategic_delta import (
    NonLinearStrategicDelta,
)
from strategic_ml.gsc.non_linear_gp.non_linear_noisy_label_delta import (
    NonLinearNoisyLabelDelta,
)
from strategic_ml.gsc.non_linear_gp.non_linear_adv_delta import NonLinearAdvDelta


__all__ = [
    "_NonLinearGP",
    "NonLinearStrategicDelta",
    "NonLinearNoisyLabelDelta",
    "NonLinearAdvDelta",
]
