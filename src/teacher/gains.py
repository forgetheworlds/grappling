"""Per-phase stabilizer gains for the phase-3 teacher.

The single world-frame task error ``e`` (see controller.py) is in metres, so
the gains have direct physical meaning:

  k_pitch, k_roll   fraction of the authority-matched direct (ankle/knee/hip)
                    correction: 1.0 commands the joint offsets whose measured
                    authority would displace the base by ``e``
  k_yaw             heading (base yaw) feedback gain: 1.0 cancels the yaw
                    error to its own reference over one 0.25 s burst
  yaw_max           clamp on the hip_yaw offset (rad).  The measured static
                    authority is ~0.35 rad of base yaw per rad of hip_yaw, but
                    enabling the channel in the closed loop measured *worse*
                    (balance disturbed: STANCE stay-up 0.63 -> 0.43 while the
                    yaw error still saturated), so the shipped table leaves
                    k_yaw = 0; the channel stays for ablation
  kd                capture-point lookahead (s): 0 disables velocity feedback
  com_clamp         clamp on the CoM-balance part of ``e`` (m)
  task_clamp        clamp on the total ``e`` fed to the channels (m)
  ki, int_max       task integrator gain (1/s) and clamp (m); ki=0 disables
  k_anchor          planted-foot IK gain (1 = full correction per tick)
  anchor_clamp      per-site IK error clamp (m)
  k_flat            sole flattening: 0 = keep the reference pose's foot pitch,
                    1 = drive both sole sites to the model's rest height
  k_z, ki_z         pelvis-height P/I gains -> foot-target z offset
  z_clamp           clamp on the foot-target z offset (m)
  k_up              torso-upright gain (waist channel)
  upright_blend     0 = follow the reference lean, 1 = drive the torso to +z
  max_leg/max_upp   per-tick clamps on the leg / upper-body joint offsets (rad)
  max_rate          max offset change per control tick (rad)

Gains were tuned by measurement on all seven techniques (iteration log in
reports/2026-10-08/teacher.md).  ``TECHNIQUE_OVERRIDES`` holds the departures
that survived that measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class GainSet:
    k_pitch: float = 0.0
    k_roll: float = 0.0
    k_yaw: float = 0.0
    yaw_max: float = 0.25
    kd: float = 0.30
    com_clamp: float = 0.08
    task_clamp: float = 0.25
    ki: float = 0.6
    int_max: float = 0.12
    k_anchor: float = 0.6
    anchor_clamp: float = 0.08
    k_flat: float = 1.0
    k_z: float = 0.40
    ki_z: float = 0.40
    z_clamp: float = 0.05
    k_up: float = 0.20
    upright_blend: float = 0.40
    #: posture governor: balance error (m) where the blend starts / is full
    govern_e0: float = 0.05
    govern_e1: float = 0.14
    #: blend rise / fall per control tick (rad of authority per tick)
    govern_up: float = 0.08
    govern_down: float = 0.010
    #: which joints the governor may move: legs+waist (True) or everything
    govern_legs_only: bool = True
    max_leg: float = 0.40
    max_upp: float = 0.15
    max_rate: float = 0.25


#: defaults per phase kind (phases.py labels)
GAIN_TABLE: dict[str, GainSet] = {
    "STAND": GainSet(k_pitch=1.00, k_roll=0.50, k_yaw=0.0, ki=0.6, k_anchor=0.60,
                     k_flat=1.0, k_z=0.40, ki_z=0.40,
                     k_up=0.20, upright_blend=0.40,
                     govern_e0=0.05, govern_e1=0.14),
    "RISE":  GainSet(k_pitch=0.70, k_roll=0.40, k_yaw=0.0, ki=0.5, k_anchor=0.50,
                     k_flat=0.8, k_z=0.30, ki_z=0.35,
                     k_up=0.15, upright_blend=0.25,
                     govern_e0=0.07, govern_e1=0.20),
    "LOW":   GainSet(k_pitch=0.70, k_roll=0.40, k_yaw=0.0, ki=0.4, k_anchor=0.50,
                     k_flat=0.7, k_z=0.30, ki_z=0.30,
                     k_up=0.12, upright_blend=0.20,
                     govern_e0=0.08, govern_e1=0.22),
    "DIVE":  GainSet(k_pitch=0.40, k_roll=0.30, k_yaw=0.0, ki=0.0, k_anchor=0.40,
                     k_flat=0.4, k_z=0.15, ki_z=0.10,
                     k_up=0.06, upright_blend=0.10,
                     govern_e0=0.12, govern_e1=0.30, govern_up=0.04),
    "PRONE": GainSet(k_pitch=0.0, k_roll=0.0, ki=0.0, k_anchor=0.0,
                     k_flat=0.0, k_z=0.0, ki_z=0.0, k_up=0.0,
                     upright_blend=0.0, govern_e0=9.0, govern_e1=9.1),
}


def gain_for(technique: str, kind: str) -> GainSet:
    """GainSet for (technique, phase kind) after the measured overrides."""
    base = replace(GAIN_TABLE[kind])
    return replace(base, **TECHNIQUE_OVERRIDES.get(technique, {}).get(kind, {}))


#: measured per-technique departures from GAIN_TABLE
TECHNIQUE_OVERRIDES: dict[str, dict[str, dict]] = {}
