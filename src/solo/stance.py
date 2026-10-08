"""Stance parameterisation: (width, height, lead leg) -> 29 joint targets.

Measured geometry (solo scene, position-free forward kinematics; this host
2026-10-08):

* ``stand`` keyframe: foot sites at y = +/-0.1185 m -> **stance width 0.237 m**
  (site separations), pelvis z 0.790 m, sole z 0.033 m.
* hip_roll sweep (both legs, sole-levelling ankle_roll = -hip_roll):
  y-separation 0.237 + ~1.22 * hip_roll m; roll 0.1 -> 0.360 m, 0.3 -> 0.600 m.
* knee flex (hip = knee/2, ankle = -knee/2): foot rises off the ground by
  0.035 m at knee 0.3, 0.132 m at 0.6, 0.328 m at 1.0 -- i.e. a crouch to
  pelvis height h needs the base to drop by roughly that rise.

``stance_targets`` returns the ctrl targets for a symmetric stance with the
requested width/height; the *physical* holdability of a given target is a
property of the closed loop, not of this mapping -- the eval harness measures it
(no crouch below ~0.72 m is holdable by pure position servos; see
``reports/2026-10-08/solo_env.md`` for the measured table).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scene import N_JOINTS, load_solo_model, stand_frame

#: measured stand geometry
STAND_WIDTH = 0.237
STAND_HEIGHT = 0.790
_Y_PER_ROLL = 1.22
#: measured foot rise (m) vs knee flex (rad) -- monotone interpolation table
_KNEE_RISE = np.array([0.0, 0.3, 0.6, 1.0, 1.4])
_FOOT_RISE = np.array([0.0, 0.035, 0.132, 0.328, 0.545])
#: foot flex window used to keep soles approximately level
MAX_ANKLE = 0.52


@dataclass(frozen=True)
class StanceGeometry:
    width: float
    height: float
    knee: float
    hip_roll: float
    hip_pitch: float
    ankle_pitch: float


def solve_stance(width: float = STAND_WIDTH, height: float = STAND_HEIGHT) -> StanceGeometry:
    """Joint angles realising (width, height) up to the measured model limits."""
    rise = max(0.0, STAND_HEIGHT - float(height))
    knee = float(np.interp(rise, _FOOT_RISE, _KNEE_RISE))
    hip_roll = (float(width) - STAND_WIDTH) / _Y_PER_ROLL
    return StanceGeometry(
        width=float(width), height=float(height), knee=knee, hip_roll=hip_roll,
        hip_pitch=knee / 2.0,
        ankle_pitch=float(np.clip(-knee / 2.0, -0.87, MAX_ANKLE)),
    )


def stance_targets(width: float = STAND_WIDTH, height: float = STAND_HEIGHT,
                   model=None) -> np.ndarray:
    """(29,) ctrl targets for the symmetric stance (lead leg is a *tactical*
    label for skills; the geometric stance here stays symmetric, matching the
    ``stand`` keyframe whose legs are symmetric)."""
    m = model if model is not None else load_solo_model()
    _, base = stand_frame(m)
    g = solve_stance(width, height)
    c = base.copy()
    for side, roll_sign in (("left", +1.0), ("right", -1.0)):
        off = 0 if side == "left" else 6
        c[off + 0] = g.hip_pitch          # hip_pitch
        c[off + 1] = roll_sign * g.hip_roll  # hip_roll
        c[off + 3] = g.knee               # knee
        c[off + 4] = g.ankle_pitch        # ankle_pitch
        c[off + 5] = -roll_sign * g.hip_roll  # ankle_roll levels the sole
    lo, hi = (np.asarray(m.actuator_ctrlrange[:, 0], np.float64),
              np.asarray(m.actuator_ctrlrange[:, 1], np.float64))
    return np.clip(c, lo, hi)


def stance_qpos(width: float = STAND_WIDTH, height: float = STAND_HEIGHT,
                model=None) -> np.ndarray:
    """(36,) qpos for the stance: stand keyframe with the solved leg angles
    and the base lowered by the measured foot rise (feet stay on the mat)."""
    m = model if model is not None else load_solo_model()
    q, _ = stand_frame(m)
    g = solve_stance(width, height)
    q = q.copy()
    rise = float(np.interp(g.knee, _KNEE_RISE, _FOOT_RISE))
    q[2] = STAND_HEIGHT - rise
    q[7 + 0] = q[7 + 6] = g.hip_pitch
    q[7 + 1] = g.hip_roll
    q[7 + 7] = -g.hip_roll
    q[7 + 3] = q[7 + 9] = g.knee
    q[7 + 4] = q[7 + 10] = g.ankle_pitch
    q[7 + 5] = -g.hip_roll
    q[7 + 11] = g.hip_roll
    return q


def measured_width_y(targets: np.ndarray, model=None) -> float:
    """Foot-site y separation (m) that ``targets`` produce kinematically."""
    import mujoco

    m = model if model is not None else load_solo_model()
    d = mujoco.MjData(m)
    q, _ = stand_frame(m)
    d.qpos[:] = q
    d.qpos[7:36] = np.asarray(targets, dtype=np.float64)
    mujoco.mj_forward(m, d)
    lf = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "a_left_foot")]
    rf = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "a_right_foot")]
    return float(lf[1] - rf[1])


if __name__ == "__main__":  # self-check
    m = load_solo_model()
    for width, height in ((STAND_WIDTH, STAND_HEIGHT), (0.30, 0.79), (0.42, 0.79),
                          (0.30, 0.72), (0.24, 0.65)):
        t = stance_targets(width, height, m)
        assert t.shape == (N_JOINTS,)
        w = measured_width_y(t, m)
        print(f"stance w={width:.3f} h={height:.2f} -> knee={solve_stance(width, height).knee:.2f} "
              f"roll={solve_stance(width, height).hip_roll:.3f} measured width={w:.3f}")
    print("solo.stance self-check OK")
