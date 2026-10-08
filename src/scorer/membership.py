"""Smooth fuzzy membership functions for technique predicates.

Every membership returns [0, 1], is 1.0 on a "core" region and falls to 0
outside a "soft" region via a C1 cubic smoothstep (3t^2 - 2t^3) — no hard
cliffs, so small deviations degrade the score gracefully (adaptation room)
while large relational violations kill it. Thresholds live in config.py and
are calibrated on the phase-2 reference data (reports/2026-10-08/scorer.md).
"""

from __future__ import annotations

import numpy as np


def _smoothstep(t: float) -> float:
    """C1 ramp 0->1 on t in [0, 1]."""
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return t * t * (3.0 - 2.0 * t)


def band(x: float, lo: float, hi: float, lo_soft: float, hi_soft: float) -> float:
    """1 on [lo, hi]; smoothstep decay to 0 at lo_soft (below) / hi_soft."""
    if x < lo:
        return 1.0 - _smoothstep((lo - x) / max(lo - lo_soft, 1e-9))
    if x > hi:
        return 1.0 - _smoothstep((x - hi) / max(hi_soft - hi, 1e-9))
    return 1.0


def at_most(x: float, core: float, soft: float) -> float:
    """1 for x <= core; smoothstep decay to 0 at soft (>= core)."""
    if x <= core:
        return 1.0
    return 1.0 - _smoothstep((x - core) / max(soft - core, 1e-9))


def at_least(x: float, core: float, soft: float) -> float:
    """1 for x >= core; smoothstep decay to 0 at soft (<= core)."""
    if x >= core:
        return 1.0
    return 1.0 - _smoothstep((core - x) / max(core - soft, 1e-9))


def ang_near(x: float, center: float, core: float, soft: float) -> float:
    """Circular band: 1 within +-/-'core' of center (radians), 0 at 'soft'.

    Core distance is computed on the wrapped difference, so angles near
    +/-pi (e.g. face-off headings crossing the seam) behave continuously.
    """
    d = abs((x - center + np.pi) % (2.0 * np.pi) - np.pi)
    if d <= core:
        return 1.0
    return 1.0 - _smoothstep((d - core) / max(soft - core, 1e-9))
