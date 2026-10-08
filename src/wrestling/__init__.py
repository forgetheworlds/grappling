"""Wrestling environment package (Phase 4).

Modules:
  backdet : back-to-mat detector (deliverable 10)
  env     : two-G1 standing-wrestling environment with exchange loop (deliverable 11)
"""

from .backdet import (
    ROBOTS,
    BackDetConfig,
    BackFeatures,
    BackToMatDetector,
    back_features,
    confirmed_mask,
    first_confirmed_index,
)
from .env import (
    CTRL_HZ,
    MAT_RADIUS,
    N_JOINTS,
    QPOS_SLICE,
    STEP_DT,
    ReferenceReplay,
    WrestlingEnv,
    default_observation,
    load_wrestling_model,
)

__all__ = [
    "ROBOTS",
    "BackDetConfig",
    "BackFeatures",
    "BackToMatDetector",
    "back_features",
    "confirmed_mask",
    "first_confirmed_index",
    "CTRL_HZ",
    "MAT_RADIUS",
    "N_JOINTS",
    "QPOS_SLICE",
    "STEP_DT",
    "ReferenceReplay",
    "WrestlingEnv",
    "default_observation",
    "load_wrestling_model",
]
