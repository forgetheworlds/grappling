"""Per-phase stabilizer gains (phase-3 teacher).

One GainSet per phase kind (see phases.py). All are dimensionless per-tick
fractions of a clamped error at the 50 Hz control rate, so a gain of 0.2
removes ~1 - 0.8^5 = 67% of the error in 100 ms.

Fields
  k_com        CoM horizontal error gain (ankle/hip strategy, candidate 1)
  cp_tau       capture-point lookahead (s); 0 disables velocity damping
               (candidate 2)
  com_clamp    max commanded CoM shift per tick (m)
  k_anchor     planted-foot world-anchor gain (candidate 3; differential
               6-DoF leg IK via the same MuJoCo jacobians)
  anchor_clamp max per-tick foot position correction (m)
  k_up         torso uprightness gain; upright_blend blends the reference
               up-vector toward world +z (0 = follow reference lean)
  k_z, ki_z    pelvis-height P and (leaky) I gains vs the reference height
               (candidate 4); ki_int_clamp bounds the integrator (m)
  max_offset   per-tick cap on |joint offset| (rad), split legs / upper
  max_rate     cap on offset change per control tick (rad)

Gains were tuned on seed 0 of all seven techniques (see the iteration log in
reports/2026-10-08/teacher.md); TECHNIQUE_OVERRIDES holds the departures that
survived measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class GainSet:
    k_com: float = 0.0
    cp_tau: float = 0.28
    com_clamp: float = 0.10
    k_anchor: float = 0.0
    anchor_clamp: float = 0.05
    k_up: float = 0.0
    upright_blend: float = 0.0
    k_z: float = 0.0
    ki_z: float = 0.0
    ki_int_clamp: float = 0.06
    max_leg_offset: float = 0.45
    max_upp_offset: float = 0.15
    max_rate: float = 0.25


#: defaults per phase kind
GAIN_TABLE: dict[str, GainSet] = {
    "STAND": GainSet(k_com=0.22, k_anchor=0.30, k_up=0.30, upright_blend=0.7,
                     k_z=0.25, ki_z=0.4),
    "RISE":  GainSet(k_com=0.15, k_anchor=0.25, k_up=0.20, upright_blend=0.5,
                     k_z=0.20, ki_z=0.3),
    "LOW":   GainSet(k_com=0.12, k_anchor=0.25, k_up=0.15, upright_blend=0.35,
                     k_z=0.15, ki_z=0.25),
    "DIVE":  GainSet(k_com=0.05, k_anchor=0.12, k_up=0.05, upright_blend=0.10,
                     k_z=0.05, ki_z=0.0),
    "PRONE": GainSet(),
}


def gain_for(technique: str, kind: str) -> GainSet:
    """GainSet for (technique, phase kind) after overrides."""
    base = replace(GAIN_TABLE[kind])
    return replace(base, **TECHNIQUE_OVERRIDES.get(technique, {}).get(kind, {}))


#: measured per-technique departures from GAIN_TABLE
TECHNIQUE_OVERRIDES: dict[str, dict[str, dict]] = {
    # SPRAWL defender rises/sprawls repeatedly: stay soft so the intended
    # posture changes are not fought by the height PI.
    "SPRAWL": {"RISE": {"k_z": 0.10, "ki_z": 0.15, "k_anchor": 0.20}},
    # STAND_UP pushes from the mat through long RISE phases: stronger anchors
    # stop foot slip during the technical stand-up.
    "STAND_UP": {"RISE": {"k_anchor": 0.35, "k_com": 0.18, "ki_z": 0.4}},
    # BODY_LOCK clinch: heavy contact forces push CoM around; max authority.
    "BODY_LOCK": {"STAND": {"k_com": 0.26, "k_anchor": 0.35}},
}
