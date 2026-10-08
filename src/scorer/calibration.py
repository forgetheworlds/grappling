"""Threshold-calibration rule for scorer predicates (documented + testable).

The rule is fixed; the *numbers* are data. For every predicate of every
(technique, phase) band the reference trace supplies the sample ``x`` of the
predicate's feature over exactly the frames of that band in
``data/refs/<TECHNIQUE>.npz``; the rule turns that sample into the fuzzy
membership parameters:

- ``band``      core = [p10, p90]; soft = core extended by
                ``SPREAD_FRAC * (p90 - p10) + floor``.
- ``at_most``   core = p90; soft = core + ``SPREAD_FRAC * (p90 - p10) + floor``
                (feature must stay low: leg-back, lock distance, sprawl depth).
- ``at_least``  core = p10; soft = core - same extension (feature must stay
                high: knee/ankle height, free-leg distance, lock asymmetry).
- ``ang_near``  center = circular mean; core = p90 of |wrapped deviation|;
                soft = core + max(``SPREAD_FRAC * core``, floor).

Why this shape: the reference itself must score ~1.0 (the [p10, p90] core plus
a soft ramp that still covers the reference tails), while anything outside the
demonstrated relationship decays smoothly through the C1 smoothstep — graded
room to adapt under resistance, no cliffs. ``floor`` keeps a phase whose
reference is nearly constant from becoming a razor band; per-feature floors are
0.10 m for distances and 0.15 rad for angles.

Montage caveat: for SPRAWL the per-edge keyframe boundaries are blurred by the
solver's junction alignment (see reports/2026-10-08/scorer.md), so its bands sit
on the defender's own feature trajectories; every other technique's band
boundaries fall on reconstructed keyframe times from the npz meta stretches.
"""

from __future__ import annotations

import numpy as np

#: core percentile (percentile used for the "at least"/"at most" cores too)
CORE_PCT = 10.0
#: soft ramp width as a fraction of the core spread (p90 - p10)
SPREAD_FRAC = 0.25
#: minimum soft ramp width, meters (distances) / radians (angles)
FLOOR_M = 0.10
FLOOR_RAD = 0.15

#: features measured in radians (everything else is meters or meters/derived)
ANGLE_FEATURES = frozenset({"bearing", "rel_yaw", "pitch_s", "roll_s",
                            "pitch_o", "roll_o"})

#: technique whose phase bands follow feature trajectories, not keyframetimes
MONTAGE_TECHNIQUES = frozenset({"SPRAWL"})

#: boundary tolerance (s) when checking bands against reconstructed keyframes
BAND_TOL_S = 0.03


def floor_for(feature: str) -> float:
    """Minimum soft-ramp width for a feature (rad for angles, else m)."""
    return FLOOR_RAD if feature in ANGLE_FEATURES else FLOOR_M


def circular_mean(x: np.ndarray) -> float:
    """Mean angle of a sample in (-pi, pi] via the unit-vector mean."""
    return float(np.arctan2(np.sin(x).mean(), np.cos(x).mean()))


def fit_params(kind: str, feature: str, x: np.ndarray) -> tuple[float, ...]:
    """Fuzzy parameters for one predicate from its reference sample ``x``.

    ``kind`` is one of band / at_most / at_least / ang_near (see module doc).
    Raises ValueError for an unknown kind or an empty sample.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError(f"empty reference sample for {feature!r}")
    a = CORE_PCT
    lo, hi = (float(v) for v in np.percentile(x, [a, 100.0 - a]))
    spread = hi - lo
    ext = SPREAD_FRAC * spread + floor_for(feature)
    if kind == "band":
        return (lo, hi, lo - ext, hi + ext)
    if kind == "at_most":
        return (hi, hi + ext)
    if kind == "at_least":
        return (lo, lo - ext)
    if kind == "ang_near":
        core = float(np.percentile(np.abs((x - circular_mean(x) + np.pi)
                                          % (2.0 * np.pi) - np.pi), 100.0 - a))
        return (circular_mean(x), core, core + max(SPREAD_FRAC * core, floor_for(feature)))
    raise ValueError(f"unknown membership kind {kind!r}")


def params_to_list(params: tuple[float, ...], ndigits: int = 4) -> list[float]:
    """Round params for stable JSON output."""
    return [round(float(v), ndigits) for v in params]


if __name__ == "__main__":  # self-check
    x = np.linspace(0.0, 1.0, 101)
    lo, hi, ls, hs = fit_params("band", "sep", x)
    assert abs(lo - 0.1) < 1e-9 and abs(hi - 0.9) < 1e-9, (lo, hi)
    assert abs(ls - (0.1 - 0.25 * 0.8 - 0.10)) < 1e-9, ls
    core_most, soft_most = fit_params("at_most", "lock_r", x)
    assert core_most == hi and soft_most > core_most > 0.0
    core_least, soft_least = fit_params("at_least", "knee_z_s", x)
    assert core_least == lo and soft_least < core_least
    ang = np.linspace(-0.2, 0.2, 41)
    c3, core3, soft3 = fit_params("ang_near", "bearing", ang)
    assert abs(c3) < 1e-9 and core3 > 0.0 and soft3 > core3
    try:
        fit_params("nope", "sep", x)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown kind accepted")
    print("calibration rule self-check OK:",
          params_to_list(fit_params("band", "sep", x)),
          params_to_list(fit_params("ang_near", "bearing", ang)))
