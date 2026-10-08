"""One measured "is this a good stance?" predicate, shared by the reward and gates.

Operator directive (2026-10-08): *every* behaviour we train must return to a good
stance, so stance validity must be a single predicate used by the reward and by
a terminal-state criterion in every task gate -- not a T1-only property.

Every threshold is derived from measured data on this host; none is invented.

Provenance of each channel
--------------------------
* ``upright_min`` (**0.97**) and ``pelvis_tol_m`` (**0.06 m**) and
  ``speed_max_mps`` (**0.15 m/s**): ``solo.metrics.recovery_time`` defaults --
  the project's own measured "stable stance" definition (upright + stance height
  + base speed, held 0.2 s), already used by ``eval._recovered_episode`` and the
  T1 gate note.
* ``require_both_feet`` (**True**): measured held stand has both feet in contact
  99.8 % of frames (``data/solo/metrics/locomotion_t2gate_stand_hold_s0.jsonl``,
  2400 frames).
* ``tilt_deg_max`` (**5.0**): the same reference trace reaches tilt p95 1.53 deg
  and max 3.58 deg; ``solo.fall`` records the measured held stand at 0.2 deg.
  5.0 deg gives headroom over the measured max.  (The fall rule's own
  ``tilt_ground_deg`` = 60 deg is the "on the mat" boundary, far looser.)
* ``hull_margin_min_m`` (**0.02**): the project's stated stance requirement --
  ``docs/QUALITY_RUBRIC.md`` A7 "CoM inside the support polygon with >= 2 cm
  margin" and the drill ``lift_gate`` >= 0.02 m.  Measured reference: the
  stand_hold rollout holds a minimum signed hull margin of **0.0247 m**
  (2400 frames, this host), and the L1 stance spec reports ``com_margin_m`` =
  **0.0861 m** (``data/drill/stance_report.json``).
* ``pose_tol_rad`` (**0.10**): measured reference ``max|q - q_stand|`` over the
  same 2400-frame stand_hold rollout is **0.0396 rad** (mean 0.0056).
* hull geometry: convex hull of the 8 sole contact spheres
  (``robots/g1/g1.xml``, r = 5 mm; heel x = -0.05, toe x = +0.12, y = +-0.025/0.03),
  projected to the mat -- the physical CoP region for point contacts.

Channels the *recorded* per-step rows cannot supply (``hull_margin_m``,
``pose_err_rad``) are measured live from the env at the terminal step; the
predicate also accepts them directly so the reward can pass its own values.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

#: radius of the sole contact spheres in robots/g1/g1.xml
SOLE_RADIUS = 0.005

#: measured reference stand height (solo scene); see solo.stance.STAND_HEIGHT
STAND_HEIGHT = 0.790


@dataclass(frozen=True)
class StanceThresholds:
    """Measured thresholds for :func:`stance_valid` (provenance in the module docstring)."""

    hull_margin_min_m: float = 0.02
    tilt_deg_max: float = 5.0
    pelvis_tol_m: float = 0.06
    speed_max_mps: float = 0.15
    pose_tol_rad: float = 0.10
    upright_min: float = 0.97
    require_both_feet: bool = True

    def as_dict(self) -> dict:
        return {
            "hull_margin_min_m": self.hull_margin_min_m,
            "tilt_deg_max": self.tilt_deg_max,
            "pelvis_tol_m": self.pelvis_tol_m,
            "speed_max_mps": self.speed_max_mps,
            "pose_tol_rad": self.pose_tol_rad,
            "upright_min": self.upright_min,
            "require_both_feet": self.require_both_feet,
        }


#: the default (shared) thresholds -- reward and gates MUST use this object
STANCE = StanceThresholds()


def stance_valid(*, hull_margin_m: float | None = None, tilt_deg: float | None = None,
                 pelvis_z: float | None = None, stand_height: float | None = None,
                 speed: float | None = None, pose_err_rad: float | None = None,
                 contact_l: bool | None = None, contact_r: bool | None = None,
                 upright: float | None = None,
                 thr: StanceThresholds = STANCE) -> tuple[bool, list[str]]:
    """Is this state a valid stance?  Returns ``(ok, reasons)``.

    ``reasons`` lists every failed channel (``["ok"]`` when all pass).  A channel
    passed as ``None`` is *unavailable* and is skipped -- callers should supply
    everything they can; the tests pin which channels are required for a gate.
    """
    bad: list[str] = []
    if hull_margin_m is not None and float(hull_margin_m) < thr.hull_margin_min_m:
        bad.append(f"hull_margin {hull_margin_m:.4f} < {thr.hull_margin_min_m}")
    if tilt_deg is not None and float(tilt_deg) > thr.tilt_deg_max:
        bad.append(f"tilt {tilt_deg:.2f} > {thr.tilt_deg_max}")
    if upright is not None and float(upright) < thr.upright_min:
        bad.append(f"upright {upright:.4f} < {thr.upright_min}")
    if pelvis_z is not None and stand_height is not None:
        if abs(float(pelvis_z) - float(stand_height)) > thr.pelvis_tol_m:
            bad.append(f"pelvis_z {pelvis_z:.4f} off {stand_height:.3f} "
                       f"by more than {thr.pelvis_tol_m}")
    if speed is not None and float(speed) > thr.speed_max_mps:
        bad.append(f"speed {speed:.4f} > {thr.speed_max_mps}")
    if pose_err_rad is not None and float(pose_err_rad) > thr.pose_tol_rad:
        bad.append(f"pose_err {pose_err_rad:.4f} > {thr.pose_tol_rad}")
    if thr.require_both_feet:
        if contact_l is not None and not bool(contact_l):
            bad.append("left foot not in contact")
        if contact_r is not None and not bool(contact_r):
            bad.append("right foot not in contact")
    return (not bad), (bad or ["ok"])


# --------------------------------------------------------------------------- #
# geometry / env extraction
# --------------------------------------------------------------------------- #
_SOLE_CACHE: dict[int, tuple[int, ...]] = {}


def sole_geom_ids(model) -> tuple[int, ...]:
    """Geom ids of the 8 sole contact spheres (cached per model)."""
    key = id(model)
    ids = _SOLE_CACHE.get(key)
    if ids is None:
        import mujoco

        ids = tuple(g for g in range(model.ngeom)
                    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
                    and abs(float(model.geom_size[g][0]) - SOLE_RADIUS) < 1e-9)
        _SOLE_CACHE[key] = ids
    return ids


def convex_hull_xy(points: np.ndarray) -> np.ndarray:
    """Convex hull (monotone chain) of an (n,2) point array, CCW, no scipy."""
    pts = sorted({(float(x), float(y)) for x, y in np.asarray(points, float)})
    if len(pts) <= 2:
        return np.asarray(pts, float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], float)


def hull_margin_m(hull_xy: np.ndarray, com_xy: np.ndarray) -> float:
    """Signed distance from ``com_xy`` to the hull boundary, positive inside.

    For point contacts the CoP region *is* the convex hull of the contact points,
    so this is the physical support margin.
    """
    h = np.asarray(hull_xy, float)
    c = np.asarray(com_xy, float)
    if len(h) < 3:
        return float("nan")
    best = np.inf
    for i in range(len(h)):
        a, b = h[i], h[(i + 1) % len(h)]
        e = b - a
        n = np.array([-e[1], e[0]])
        ln = float(np.linalg.norm(n))
        if ln < 1e-12:
            continue
        n = n / ln
        mid = 0.5 * (a + b)
        if float(np.dot(n, mid - c)) < 0.0:   # orient outward
            n = -n
        best = min(best, -float(np.dot(n, c - mid)))   # + inside
    return float(best)


def support_hull(model, data) -> np.ndarray:
    """Convex hull (xy) of the projected sole contact spheres at this state."""
    ids = sole_geom_ids(model)
    pts = np.asarray([data.geom_xpos[g][:2] for g in ids], float)
    return convex_hull_xy(pts)


def stance_state(env) -> dict:
    """Extract the stance-validity channels from a live :class:`solo.env.SoloEnv`.

    ``hull_margin_m`` and ``pose_err_rad`` are measured here (they are not
    recorded per step); everything else comes from the same quantities the
    recorder logs.
    """
    from .fall import contact_state

    model, data = env.model, env.data
    com = np.asarray(data.subtree_com[0], float)
    hull = support_hull(model, data)
    contacts = contact_state(model, data)
    up_z = float(np.asarray(data.xmat[env._torso_bid]).reshape(3, 3)[2, 2])
    q_stand = np.asarray(getattr(env, "_q_stand", np.zeros(36)), float)
    return {
        "hull_margin_m": hull_margin_m(hull, com[:2]),
        "tilt_deg": float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0)))),
        "upright": up_z,
        "pelvis_z": float(data.xpos[env._pelvis_bid][2]),
        "stand_height": float(getattr(env, "_q_stand", np.zeros(36))[2]),
        "speed": float(np.linalg.norm(np.asarray(data.qvel[0:3], float))),
        "pose_err_rad": float(np.abs(np.asarray(data.qpos[7:36], float)
                                     - q_stand[7:36]).max()),
        "contact_l": bool(contacts.left_foot),
        "contact_r": bool(contacts.right_foot),
    }


def ends_in_valid_stance(env, thr: StanceThresholds = STANCE) -> tuple[bool, list[str]]:
    """Validate the env's CURRENT (terminal) state; ``(ok, reasons)``."""
    return stance_valid(**stance_state(env), thr=thr)
