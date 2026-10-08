"""Literature-sourced balance recipe for the T1 solo task (flags OFF by default).

Two primary sources are transferred here; both were read in full for this file
(citations in ``reports/2026-10-08/lit_balance.md``):

* **A** -- van Marum et al., *Revisiting Reward Design and Evaluation for Robust
  Humanoid Standing and Walking*, arXiv:2404.19173.  Same control interface as
  ours (joint-space PD setpoints, 50 Hz policy, episodic, early termination on
  fall, PPO **extended with a mirror loss**).  Their measured finding: rewarding
  double foot contact for standing is *harmful* -- it penalises the recovery
  steps that must break contact, and makes walk->stand pick the nearest stance
  instead of the most stable one (their "Feet contact" term is the *constant* 1
  for a standing command, "giving no preference for foot contact").
* **B** -- Yang et al., *Learning Whole-body Motor Skills for Humanoids*,
  arXiv:2002.02991.  One policy learns ankle/hip/foot-tilt/stepping recovery with
  the upper body locked, heading-invariant observations, pushes repeated during
  an episode drawn from ``[0.5x, 2x]`` the analytic non-stepping ceiling
  ``J = m * dCOP * sqrt(g / z_c)``, and exp-quadratic rewards on torso pose, CoM
  position (target: **the centre of the support polygon**, "to provide maximum
  disturbance compensation"), CoM velocity (target from the capture point) and
  even left/right ground-reaction-force distribution, plus contact and power
  penalties.  Trained on one 4-core desktop CPU, converged in ~2 days.

What lives here (everything default-OFF, so v5's configuration is untouched):

1. :func:`measure_ceiling` -- the analytic non-stepping ceiling computed from the
   *actual* G1 model in the ``a_stand`` keyframe (mass, CoM height, support hull,
   per-direction ``dCOP`` and ``J``).  The single scalar "~13 N*s" that
   ``solo/pushes.py`` and the T1 gate quoted is wrong as a scalar: the ceiling is
   direction-dependent (heel-ward pushes are the binding ones).
2. :func:`joint_mask` -- the action-space restriction (freeze the arms, wrists
   and hip/waist yaw at the ``a_stand`` ctrl targets; the residual acts on the
   balance-relevant joints only).  Built from the model + keyframe, applied by
   ``SoloEnv.resolve_action`` so training and evaluation share one mask.
3. :class:`LitPushConfig` / :func:`lit_push_schedule` -- the disturbance
   distribution: source-B interval pushes and source-A per-frame Bernoulli
   pushes, magnitudes as a fraction of the *directional* ceiling.
4. :func:`gamma_half_life_s` -- the gamma half-life table (source B derives
   gamma from a ~0.5 s half-life at 25 Hz; ours is 2.77 s at 50 Hz).

Not transferred here: source A's per-term weights live in ``solo.reward``
(``RewardWeights`` + ``LIT_BALANCE_TERMS``); source A's mirror loss lives in
``solo.mirror``.

Measurement conventions (do not duplicate): the support polygon is the convex
hull of the **sole contact spheres** of every loaded foot (the four 5 mm
collision spheres per foot, read from the model -- ``robots/g1/g1.xml`` geoms
15/16/17/18 and 30/31/32/33), and the support centre is that hull's *area
centroid* (for the rectangular double-support footprint this is also the
max-min-margin point, i.e. the "maximum disturbance compensation" point of
source B, because the hull is symmetric in ``y`` and the sagittal extremes are
``-0.05``/``+0.12`` m about the true centre).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from drill.kin import hull2d, polygon_margin

from .pushes import PushSchedule, PushSpec
from .scene import (N_JOINTS, PELVIS_BODY, STEP_DT, body_id, load_solo_model,
                    stand_frame)

#: joints the balance residual is allowed to move (source B's ankle/hip/knee
#: recovery set; source A's rp-orientation + base-height terms are the signals
#: that shape it).  Names are the model names with the ``a_`` prefix and the
#: ``_joint`` suffix stripped.
LIT_ACTIVE_JOINTS: tuple[str, ...] = (
    "left_hip_pitch", "left_hip_roll", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_pitch", "waist_roll",
)

#: why each frozen group is safe to freeze at the ``a_stand`` keyframe ctrl
#: (the keyframe is the S1 "verified-stable" stand and the reset distribution's
#: mean; every value below is that keyframe's own value, so freezing changes
#: nothing at reset).
LIT_FROZEN_GROUPS: dict[str, tuple[tuple[str, ...], str]] = {
    "arms": (("left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
              "left_elbow", "right_shoulder_pitch", "right_shoulder_roll",
              "right_shoulder_yaw", "right_elbow"),
             "keyframe holds shoulder_pitch 0.20 rad, |shoulder_roll| 0.20 rad, "
             "elbow 1.28 rad (arms out, forearms level); source B locks the "
             "upper body and still learns ankle/hip/stepping recovery"),
    "wrists": (("left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
                "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"),
               "keyframe ctrl is exactly 0.0 and the wrist actuators are the "
               "weakest in the robot (+/-5 N*m): they cannot contribute to "
               "balance"),
    "hip_yaw": (("left_hip_yaw", "right_hip_yaw"),
                "keyframe 0.0; leg yaw only steers the foot in double support "
                "and the T1 task commands wz = 0"),
    "waist_yaw": (("waist_yaw",),
                  "keyframe 0.0; torso yaw is not a balance degree of freedom "
                  "(the gate measures roll/pitch uprightness and CoM offset)"),
}

#: source A's per-frame push probability and its impulse band, expressed in our
#: units: they sample a 200-800 N force lasting a single 20 ms control step,
#: i.e. 4-16 N*s of impulse.  Kept as documentation/report evidence; the flags
#: default to OFF and the band used by the lit modes is a *fraction of the
#: measured directional ceiling* (see :class:`LitPushConfig`).
SOURCE_A_PUSH_PROB = 0.01
SOURCE_A_PUSH_IMPULSE_Ns = (4.0, 16.0)


# --------------------------------------------------------------------- model
def joint_names(model: mujoco.MjModel | None = None) -> tuple[str, ...]:
    """The 29 actuated joint names in actuator (action) order, unprefixed."""
    m = load_solo_model() if model is None else model
    out = []
    for i in range(m.nu):
        jid = m.actuator_trnid[i, 0]
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(jid))
        if name is None:  # pragma: no cover - every g1 joint is named
            raise ValueError(f"actuator {i} drives an unnamed joint")
        out.append(name.replace("a_", "", 1).replace("_joint", "", 1))
    return tuple(out)


def _swap_side(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[len("left_"):]
    if name.startswith("right_"):
        return "left_" + name[len("right_"):]
    return name


def mirror_index_sign(model: mujoco.MjModel | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """(index, sign) of the sagittal mirror map on the 29 joint vectors.

    ``mirror(x)[i] = x[index[i]] * sign[i]``, i.e. a state vector is mirrored by
    swapping left/right joints and negating roll/yaw joints (a reflection across
    the sagittal plane reverses the handedness of a rotation about ``x`` or
    ``z``).  Verified geometrically in ``tests/solo/test_lit_reward.py``: a
    mirrored qpos puts every left body exactly where the mirrored right body is.

    The map is an involution (``mirror(mirror(x)) == x``) and is its own
    inverse-permutation, so the same call mirrors observations and actions.
    """
    names = joint_names(model)
    idx = np.empty(N_JOINTS, dtype=int)
    sign = np.empty(N_JOINTS, dtype=float)
    for i, name in enumerate(names):
        try:
            idx[i] = names.index(_swap_side(name))
        except ValueError as exc:  # pragma: no cover - all sides exist
            raise ValueError(f"no mirror partner for joint {name!r}") from exc
        sign[i] = -1.0 if name.endswith(("_roll", "_yaw")) else 1.0
    assert sorted(idx.tolist()) == list(range(N_JOINTS)), "mirror map is not a bijection"
    return idx, sign


def sole_geom_ids(model: mujoco.MjModel | None = None) -> dict[str, np.ndarray]:
    """Body -> the ids of its four sole contact spheres (model order)."""
    m = load_solo_model() if model is None else model
    out: dict[str, np.ndarray] = {}
    for side, body in (("left", "a_left_ankle_roll_link"),
                       ("right", "a_right_ankle_roll_link")):
        bid = body_id(m, body)
        geoms = [g for g in range(m.body_geomadr[bid],
                                  m.body_geomadr[bid] + m.body_geomnum[bid])
                 if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
                 and m.geom_contype[g] > 0]
        if len(geoms) != 4:  # pragma: no cover - model contract
            raise ValueError(f"{body}: expected 4 sole spheres, found {len(geoms)}")
        out[side] = np.asarray(geoms, dtype=int)
    return out


def sole_points_world(model: mujoco.MjModel, data: mujoco.MjData,
                      geom_ids: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """(2, 4, 3) world positions of the sole contact spheres of both feet."""
    ids = sole_geom_ids(model) if geom_ids is None else geom_ids
    return np.asarray([data.geom_xpos[ids["left"]],
                       data.geom_xpos[ids["right"]]], dtype=np.float64)


# ------------------------------------------------------------------- 2-D geo
def hull_centroid(hull: np.ndarray) -> np.ndarray:
    """Area centroid of a convex polygon (2,); vertex mean if degenerate."""
    p = np.asarray(hull, dtype=float).reshape(-1, 2)
    if len(p) == 0:
        raise ValueError("empty hull")
    if len(p) < 3:
        return p.mean(axis=0)
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = 0.5 * float(cross.sum())
    if abs(area) < 1e-12:
        return p.mean(axis=0)
    cx = float(((x + x1) * cross).sum()) / (6.0 * area)
    cy = float(((y + y1) * cross).sum()) / (6.0 * area)
    return np.array([cx, cy])


def support_hull(points: np.ndarray) -> np.ndarray:
    """Convex hull (n, 2) of world-projected sole points; () when empty."""
    p = np.asarray(points, dtype=float).reshape(-1, 3)[:, :2]
    if len(p) == 0:
        return np.zeros((0, 2))
    return hull2d(p)


def support_centre(sole_points: np.ndarray, loaded: tuple[bool, bool]) -> np.ndarray:
    """Area centroid of the hull of the *loaded* feet's sole points.

    ``sole_points`` is (2, 4, 3) world (one row per foot, as
    :func:`sole_points_world` returns); ``loaded`` are the per-foot contact
    flags.  Returns ``zeros(2, 1)``-shaped ``(2,)`` of ``nan`` when nothing is
    loaded -- callers decide what an airborne foot configuration means.
    """
    pts = np.asarray(sole_points, dtype=float).reshape(2, -1, 3)
    sel = np.concatenate([pts[i] for i in range(2) if loaded[i]]) \
        if any(loaded) else np.zeros((0, 3))
    if len(sel) < 3:
        return np.full(2, np.nan)
    return hull_centroid(support_hull(sel))


def directional_margin(point: np.ndarray, hull: np.ndarray,
                       direction: np.ndarray) -> float:
    """Distance (m) from ``point`` to the hull boundary along ``direction``.

    The ray-cast result of the capture-point rejection test: how far the CoM can
    still travel in the push direction before it leaves the support polygon.
    Returns 0.0 when the point is already outside or the ray misses.
    """
    p = np.asarray(point, dtype=float).reshape(2)
    u = np.asarray(direction, dtype=float).reshape(2)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    h = np.asarray(hull, dtype=float).reshape(-1, 2)
    if len(h) < 3:
        return 0.0
    best = math.inf
    for i in range(len(h)):
        a, b = h[i], h[(i + 1) % len(h)]
        e = b - a
        den = u[0] * (-e[1]) + u[1] * e[0]
        if abs(den) < 1e-12:
            continue
        w = a - p
        t = (w[0] * (-e[1]) + w[1] * e[0]) / den
        s = (u[0] * w[1] - u[1] * w[0]) / den
        if t > 1e-9 and -1e-9 <= s <= 1.0 + 1e-9:
            best = min(best, float(t))
    return 0.0 if not math.isfinite(best) else best


# ------------------------------------------------------------------ ceiling
#: the direction labels used by :class:`SupportCeiling` (push *direction* = the
#: world yaw of the applied force, the same convention as ``PushSpec.direction``
#: and the eval battery)
CEILING_LABELS: tuple[tuple[str, float], ...] = (
    ("push_+x", 0.0), ("push_+y", 0.5 * math.pi), ("push_-x", math.pi),
    ("push_-y", 1.5 * math.pi),
)


@dataclass(frozen=True)
class SupportCeiling:
    """Analytic non-stepping ceiling ``J = m * dCOP * sqrt(g / z_c)``.

    ``dCOP`` is the distance from the CoM's ground projection to the support
    hull boundary along the push direction, so the ceiling is *direction
    dependent* (a scalar is not a ceiling -- the weakest direction is the one
    that matters).
    """

    mass: float
    z_com: float
    gravity: float
    com_xy: np.ndarray
    hull: np.ndarray
    labels: tuple[tuple[str, float], ...] = CEILING_LABELS

    # --- per-direction ---------------------------------------------------
    def dcop(self, yaw: float) -> float:
        return directional_margin(self.com_xy, self.hull,
                                  (math.cos(float(yaw)), math.sin(float(yaw))))

    def j_reject(self, yaw: float) -> float:
        """Largest impulse (N*s) absorbable in direction ``yaw`` without stepping."""
        return self.mass * self.dcop(yaw) * math.sqrt(self.gravity / self.z_com)

    def table(self, n_directions: int = 8) -> list[dict]:
        rows = []
        for k in range(int(n_directions)):
            yaw = 2.0 * math.pi * k / int(n_directions)
            rows.append({"yaw_deg": round(math.degrees(yaw), 2),
                         "dcop_m": round(self.dcop(yaw), 4),
                         "j_reject_ns": round(self.j_reject(yaw), 3)})
        return rows

    def as_dict(self) -> dict:
        named = {name: {"dcop_m": round(self.dcop(yaw), 4),
                        "j_reject_ns": round(self.j_reject(yaw), 3)}
                 for name, yaw in self.labels}
        table = self.table(8)
        js = [row["j_reject_ns"] for row in table]
        sagittal = named["push_+x"]["j_reject_ns"]
        lateral = named["push_+y"]["j_reject_ns"]
        hull = np.asarray(self.hull, dtype=float)
        return {
            "mass_kg": round(float(self.mass), 6),
            "z_com_m": round(float(self.z_com), 6),
            "gravity": round(float(self.gravity), 4),
            "com_xy": [round(float(x), 5) for x in np.asarray(self.com_xy).reshape(2)],
            "hull": [[round(float(x), 5) for x in row] for row in hull],
            "hull_depth_m": (round(float(hull[:, 0].max() - hull[:, 0].min()), 5)
                             if len(hull) else None),
            "hull_width_m": (round(float(hull[:, 1].max() - hull[:, 1].min()), 5)
                             if len(hull) else None),
            "support_centre_xy": [round(float(x), 5)
                                  for x in hull_centroid(hull)] if len(hull) >= 3
            else None,
            "directions": named,
            "j_reject_sagittal_ns": sagittal,
            "j_reject_lateral_ns": lateral,
            "j_reject_min_ns": round(float(min(js)), 3),
            "j_reject_max_ns": round(float(max(js)), 3),
            "j_reject_table_8dir": table,
        }


def ceiling_from_hull(mass: float, z_com: float, com_xy, hull, gravity: float = 9.81
                      ) -> SupportCeiling:
    """Pure constructor (tests hand-construct hulls and check the arithmetic)."""
    return SupportCeiling(mass=float(mass), z_com=float(z_com),
                          gravity=float(gravity),
                          com_xy=np.asarray(com_xy, dtype=float).reshape(2),
                          hull=np.asarray(hull, dtype=float).reshape(-1, 2))


def measure_ceiling(model: mujoco.MjModel | None = None) -> SupportCeiling:
    """Measure the ceiling from the actual model in the ``a_stand`` keyframe."""
    from .pushes import clear_applied  # local: keeps this module import-light

    m = load_solo_model() if model is None else model
    data = mujoco.MjData(m)
    q, ctrl = stand_frame(m)
    data.qpos[:] = q
    data.ctrl[:] = ctrl
    clear_applied(data)
    mujoco.mj_forward(m, data)
    mass = float(np.asarray(m.body_mass, dtype=float).sum())
    com = np.asarray(data.subtree_com[body_id(m, PELVIS_BODY)], dtype=float)
    hull = support_hull(sole_points_world(m, data))
    return ceiling_from_hull(mass, float(com[2]), com[:2], hull,
                             gravity=float(-m.opt.gravity[2]))


# --------------------------------------------------------------------- stance
@dataclass(frozen=True)
class StanceTol:
    """Thresholds of the stance-validity predicate.

    **Shared-predicate note:** T1GateCal was to define this predicate; it has NOT
    landed (``solo.eval`` has no ``stance_valid``/``stable`` symbol), so this is
    the *first* definition and must be unified with theirs, not duplicated.  The
    numbers below are taken from definitions that already exist in the repo
    wherever they do, so the two cannot silently disagree: ``up_z >= 0.97`` and
    ``|pelvis_z - stand| <= 0.06`` and ``base speed < 0.15 m/s`` are the gate's
    own ``_recovered_episode`` conditions (``solo/eval.py``); the margin, flatness
    and pose checks are the ones Main listed for T1GateCal's version.
    """

    margin_m: float = 0.0            # CoM must be strictly inside the hull
    up_z_min: float = 0.97           # eval.py's upright condition
    height_tol_m: float = 0.06       # env's RECOVERY_TOL
    speed_max: float = 0.15          # eval.py's stability speed
    tilt_max_deg: float = 14.0       # ~ up_z 0.97
    pose_dev_max: float = 0.35       # mean |q - q_stand| over the 29 joints (rad)
    flat_tol_m: float = 0.012        # sole sphere height spread on a planted foot


def stance_valid(*, pelvis_z: float, stand_height: float, up_z: float, speed: float,
                 com_xy, hull, foot_contact, sole_points, joint_dev: float,
                 tol: StanceTol = StanceTol()) -> bool:
    """Is the robot in a *valid stance* right now?  One definition, one place.

    All of: the CoM strictly inside the support hull (positive margin), both feet
    loaded, both feet flat on the mat (their sole spheres within ``flat_tol_m`` of
    the lowest one), pelvis height within ``height_tol_m`` of the stance height,
    torso upright (``up_z >= up_z_min``), base speed below ``speed_max``, and the
    joint pose within ``pose_dev_max`` of the stand keyframe.  Every input is a
    measured quantity the env already has.
    """
    from .scene import N_JOINTS as _N  # local: keeps the module import-light

    del _N
    if float(up_z) < float(tol.up_z_min):
        return False
    if abs(float(pelvis_z) - float(stand_height)) > float(tol.height_tol_m):
        return False
    if float(speed) >= float(tol.speed_max):
        return False
    if abs(float(joint_dev)) > float(tol.pose_dev_max):
        return False
    if not (bool(foot_contact[0]) and bool(foot_contact[1])):
        return False
    pts = np.asarray(sole_points, dtype=float).reshape(2, -1, 3)
    for i in range(2):
        z = pts[i][:, 2]
        if float(z.max() - z.min()) > float(tol.flat_tol_m):
            return False
    if hull is None or len(np.asarray(hull)) < 3:
        return False
    return directional_margin(com_xy, hull, np.array([1.0, 0.0])) > float(tol.margin_m) \
        and directional_margin(com_xy, hull, np.array([0.0, 1.0])) > float(tol.margin_m) \
        and directional_margin(com_xy, hull, np.array([-1.0, 0.0])) > float(tol.margin_m) \
        and directional_margin(com_xy, hull, np.array([0.0, -1.0])) > float(tol.margin_m)


def com_margin(com_xy, hull) -> float:
    """Signed CoM margin (m) vs the hull.

    Positive inside: the worst of the four cardinal directions (how far the CoM
    can still travel before leaving the hull -- the scalar the stance predicate
    thresholds).  Negative outside: the penetration depth (distance to the hull
    boundary, via :func:`drill.kin.polygon_margin`), so a return-to-stance term
    can price *leaving* the hull instead of saturating at 0.  A degenerate hull
    (< 3 points) has no interior and reports ``-inf``.
    """
    h = np.asarray(hull, dtype=float).reshape(-1, 2)
    if len(h) < 3:
        return float("-inf")
    p = np.asarray(com_xy, dtype=float).reshape(2)
    m = float(min(directional_margin(p, h, d) for d in
                  ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))))
    if m > 0.0:
        return m
    return float(polygon_margin(p, h))


#: a legitimate recovery may take this long; the no-recovery termination cannot
#: fire inside it, and a state that is recovering (rising pelvis, upright enough)
#: never counts as "ended badly" (see ``SoloEnv._update_stance_return``)
STANCE_GRACE_S = 1.0


# --------------------------------------------------------------------- mask
@dataclass(frozen=True)
class JointMask:
    """Freeze ``frozen`` joints at ``targets``; leave ``active`` to the policy.

    Applied by ``SoloEnv.resolve_action`` (the single choke point every
    training *and* evaluation action passes through), so a masked run and its
    evaluation cannot diverge: both write the keyframe ctrl on the frozen
    joints to the last bit.
    """

    names: tuple[str, ...]
    active: tuple[int, ...]
    frozen: tuple[int, ...]
    targets: tuple[float, ...]          # ctrl target per frozen joint (keyframe)
    groups: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def apply(self, ctrl: np.ndarray) -> np.ndarray:
        out = np.asarray(ctrl, dtype=float).reshape(N_JOINTS).copy()
        if self.frozen:
            out[list(self.frozen)] = np.asarray(self.targets, dtype=float)
        return out

    def as_dict(self) -> dict:
        return {
            "n_active": len(self.active),
            "n_frozen": len(self.frozen),
            "active": [self.names[i] for i in self.active],
            "frozen": [self.names[i] for i in self.frozen],
            "frozen_targets": {self.names[i]: round(float(t), 6)
                               for i, t in zip(self.frozen, self.targets)},
            "groups": {k: list(v) for k, v in self.groups.items()},
        }


def joint_mask(model: mujoco.MjModel | None = None,
               active_names: tuple[str, ...] = LIT_ACTIVE_JOINTS) -> JointMask:
    """Build the balance mask: ``active_names`` move, everything else is frozen
    at its ``a_stand`` keyframe ctrl value.

    Single source of truth for training and evaluation (both call this with the
    model; the trainer stores it on the env and ``env.config()`` echoes it).
    """
    m = load_solo_model() if model is None else model
    names = joint_names(m)
    active_set = set(active_names)
    unknown = sorted(active_set - set(names))
    if unknown:
        raise ValueError(f"unknown joints in the balance mask: {unknown}")
    _, ctrl = stand_frame(m)
    active = tuple(i for i, n in enumerate(names) if n in active_set)
    frozen = tuple(i for i, n in enumerate(names) if n not in active_set)
    groups = {g: tuple(n for n in members if n in names)
              for g, (members, _why) in LIT_FROZEN_GROUPS.items()}
    return JointMask(names=names, active=active, frozen=frozen,
                     targets=tuple(float(ctrl[i]) for i in frozen), groups=groups)


# --------------------------------------------------------------------- push
@dataclass(frozen=True)
class LitPushConfig:
    """Disturbance distribution for the lit balance modes (source A or B).

    ``mode="interval"`` (source B): pushes repeated during the episode, spaced
    ``interval`` seconds apart (optionally jittered), drawn from
    ``[min_frac, max_frac] x J(direction)``.  ``mode="bernoulli"`` (source A): a
    per-control-step probability ``prob`` of a single-timestep push.

    ``magnitude = uniform(min_frac, max_frac) * J(direction)`` where
    ``J(direction)`` is the *directional* ceiling from
    :func:`measure_ceiling` -- so "2x the ceiling" means 2x in that push's own
    direction, which is what makes the weak (heel-ward) directions trainable
    instead of unreachable.  ``cap`` clamps the absolute magnitude (e.g. 12 N*s
    to stay inside the T1 non-stepping band).
    """

    mode: str = "interval"
    interval: float = 5.0                 # source B: pushes every ~5 s
    interval_jitter: float = 0.0          # +/- s
    prob: float = SOURCE_A_PUSH_PROB      # source A: 1%/frame
    min_frac: float = 0.5                 # source B: [0.5x, 2x] the ceiling
    max_frac: float = 2.0
    directions: int = 8                   # evenly spaced world yaws (battery-compatible)
    height: float = 0.95                  # m, chest push height (measured holdable)
    t_first: float = 1.0                  # s, no pushes before this
    cap: float | None = None              # N*s absolute clamp (None = uncapped)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("interval", "bernoulli"):
            raise ValueError(f"unknown push mode {self.mode!r}")
        if self.mode == "interval" and self.interval <= STEP_DT:
            raise ValueError("interval mode needs interval > one control step")
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError("prob must be in [0, 1]")
        if not 0.0 <= self.min_frac <= self.max_frac:
            raise ValueError("need 0 <= min_frac <= max_frac")
        if self.directions < 1:
            raise ValueError("directions must be >= 1")
        if self.cap is not None and self.cap <= 0:
            raise ValueError("cap must be positive or None")
        if self.t_first < 0:
            raise ValueError("t_first must be >= 0")

    def as_dict(self) -> dict:
        return {
            "mode": self.mode, "interval": self.interval,
            "interval_jitter": self.interval_jitter, "prob": self.prob,
            "min_frac": self.min_frac, "max_frac": self.max_frac,
            "directions": self.directions, "height": self.height,
            "t_first": self.t_first, "cap": self.cap, "seed": self.seed,
        }


def _grid(t: float) -> float:
    """Quantize a time to the control-step grid (exact push windows)."""
    return round(float(t) / STEP_DT) * STEP_DT


@dataclass(frozen=True)
class LitPushCurriculum:
    """``schedule_for(steps, seed)`` adapter for the literature push mode.

    The single-env trainer and ``rl.vec_solo`` both install pushes through a
    ``schedule_for(steps, episode_seed)`` object (see ``solo.curriculum``), so
    this wraps :func:`lit_push_schedule` in the same shape.  ``steps`` is
    deliberately ignored: the literature distribution does **not** ramp (that is
    the point -- it trains the tested band from step 0, unlike the v5 curriculum
    whose ramp never reached the gate's magnitudes).
    """

    cfg: LitPushConfig
    ceiling: SupportCeiling
    horizon: float

    def schedule_for(self, steps: int, episode_seed: int | None = None
                     ) -> PushSchedule:
        del steps  # stationary distribution by design
        return lit_push_schedule(self.cfg, self.ceiling, self.horizon,
                                 0 if episode_seed is None else int(episode_seed))

    def as_dict(self) -> dict:
        return {"lit_push": self.cfg.as_dict(), "horizon": self.horizon,
                "ceiling": self.ceiling.as_dict()}


def lit_push_schedule(cfg: LitPushConfig, ceiling: SupportCeiling,
                      horizon: float, episode_seed: int) -> PushSchedule:
    """Deterministic schedule for one training episode.

    Deterministic in ``(cfg.seed, episode_seed, horizon)`` so a run is
    reproducible and a test can sample the distribution through the same path
    the trainer uses.
    """
    rng = np.random.default_rng(7_919_003 + int(cfg.seed) * 1_000_003
                                + int(episode_seed))
    specs: list[PushSpec] = []

    def one(t: float) -> PushSpec:
        k = int(rng.integers(0, int(cfg.directions)))
        ang = 2.0 * math.pi * k / int(cfg.directions)
        j_dir = ceiling.j_reject(ang)
        mag = float(rng.uniform(cfg.min_frac, cfg.max_frac)) * j_dir
        if cfg.cap is not None:
            mag = min(mag, float(cfg.cap))
        return PushSpec(t=_grid(t), impulse=mag, direction=ang,
                        height=float(cfg.height), duration=STEP_DT,
                        label=f"lit_{cfg.mode}_d{k}")

    if cfg.mode == "interval":
        t = float(cfg.t_first)
        while t < horizon - STEP_DT + 1e-9:
            jitter = (cfg.interval_jitter
                      * float(rng.uniform(-1.0, 1.0)) if cfg.interval_jitter else 0.0)
            specs.append(one(t + jitter))
            t += float(cfg.interval)
    else:
        steps = int(math.floor((horizon - cfg.t_first) / STEP_DT))
        if cfg.prob > 0.0:
            draws = rng.random(max(0, steps)) < float(cfg.prob)
            for k in np.nonzero(draws)[0]:
                specs.append(one(float(cfg.t_first) + int(k) * STEP_DT))
    return PushSchedule(specs)


# -------------------------------------------------------------------- gamma
def gamma_half_life_s(gamma: float, hz: float) -> float:
    """Future-reward half-life (s): ``ln(0.5)/ln(gamma)`` steps at ``hz``."""
    if not 0.0 < float(gamma) < 1.0:
        raise ValueError("gamma must be in (0, 1)")
    steps = math.log(0.5) / math.log(float(gamma))
    return float(steps) / float(hz)


def gamma_table() -> list[dict]:
    """The comparison source B's derivation forces (half-life in seconds).

    Source B derives gamma = 0.95 from a ~0.5 s future-reward half-life at
    25 Hz; source A trains 16 s episodes at 50 Hz.  Our balance task is judged
    over an 8 s episode and 2 s of post-push stance, so a 0.5 s half-life would
    discount most of the reward the gate measures.
    """
    rows = [
        {"label": "source B (0.95 @ 25 Hz)", "gamma": 0.95, "hz": 25.0},
        {"label": "ours (balance default)", "gamma": 0.995, "hz": 50.0},
        {"label": "ours (locomotion)", "gamma": 0.997, "hz": 50.0},
        {"label": "0.99 @ 50 Hz", "gamma": 0.99, "hz": 50.0},
        {"label": "0.98 @ 50 Hz", "gamma": 0.98, "hz": 50.0},
        {"label": "source-B gamma @ ours", "gamma": 0.95, "hz": 50.0},
    ]
    for r in rows:
        g, hz = float(r["gamma"]), float(r["hz"])
        r["half_life_steps"] = round(math.log(0.5) / math.log(g), 3)
        r["half_life_s"] = round(gamma_half_life_s(g, hz), 4)
        r["horizon_steps_1_over_1mg"] = round(1.0 / (1.0 - g), 2)
        r["horizon_s_1_over_1mg"] = round(1.0 / (1.0 - g) / hz, 4)
    return rows


if __name__ == "__main__":  # self-check
    model = load_solo_model()
    ceil = measure_ceiling(model)
    rows = ceiling_rows = ceil.table(8)
    print("ceiling:", {k: v for k, v in ceil.as_dict().items()
                       if k not in ("hull", "j_reject_table_8dir", "directions")})
    for r in rows:
        print("  ", r)
    assert abs(ceil.mass - 33.3411) < 1e-3, ceil.mass
    assert abs(ceil.z_com - 0.6919) < 1e-3, ceil.z_com
    assert 6.0 < ceil.as_dict()["j_reject_min_ns"] < 7.5
    assert 20.0 < ceil.as_dict()["j_reject_max_ns"] < 21.5

    mask = joint_mask(model)
    print("mask:", {"active": len(mask.active), "frozen": len(mask.frozen)})
    assert len(mask.active) == 12 and len(mask.frozen) == 17
    frozen_ctrl = [(mask.names[i], round(t, 4)) for i, t in zip(mask.frozen, mask.targets)]
    print("frozen targets:", frozen_ctrl)

    cfg = LitPushConfig(mode="interval", seed=0)
    sch = lit_push_schedule(cfg, ceil, horizon=8.0, episode_seed=3)
    print("interval pushes:", [(round(p.t, 2), round(p.impulse, 2),
                                round(math.degrees(p.direction))) for p in sch.pushes])
    cfg_b = LitPushConfig(mode="bernoulli", prob=0.01, seed=0)
    schb = lit_push_schedule(cfg_b, ceil, horizon=8.0, episode_seed=3)
    print("bernoulli pushes:", len(schb), "expected ~",
          round(0.01 * (8.0 - 1.0) / STEP_DT, 2))
    adapt = LitPushCurriculum(cfg=cfg, ceiling=ceil, horizon=8.0)
    assert adapt.schedule_for(0, 3).as_list() == sch.as_list()
    assert adapt.schedule_for(999_999, 3).as_list() == sch.as_list()  # no ramp
    for r in gamma_table():
        print("  gamma:", r)
    print("solo.lit self-check OK")
