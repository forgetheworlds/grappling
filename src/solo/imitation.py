"""DeepMimic-style imitation terms for the solo drill (RL refinement stage).

Two families, both individually testable, both bounded in ``[0, 1]`` and
combined as a weighted sum (the DeepMimic reward shape
``r = sum_i w_i * exp(-err_i^2 / scale_i^2)``):

* **site / landmark family (PRIMARY, recommended for refinement).**  The
  refinement objective should be driven by the 19 retarget landmark sites
  (:data:`retarget.landmarks.SOLVED_SITES`; the solo scene attaches them with
  the ``a_`` prefix -- ``a_core``, ``a_left_knee``, ...).  Imitating the
  *kinematic joint projection* pins the policy to a two-stage
  video->IK->imitate pipeline, whose geometric bias was measured on this repo
  (``docs``/reports: 6 of 7 retargeted references topple under naive PD
  execution, CoM margins -0.40..+0.07 m).  Optimising site positions with the
  physics in the loop lets the policy realise the *same movement* in a
  dynamically valid way.  Per-site weights reuse the retarget solve's class
  weights (:data:`retarget.landmarks.WEIGHT_PRESETS` ``"default"``:
  HIGH 4.0 / MED 1.5 / LOW 0.5 -- hands are the noisiest landmarks and the
  ankle/toe/heel landmarks oscillate even when planted, so equal weighting
  would chase detector noise).
* **joint family (SECONDARY, looser).**  Joint pose + velocity terms encode the
  pose *shape*; they stay in the reward at a smaller weight so the policy is
  not pinned to the infeasible projection.

Plus a root-position term, a whole-body-CoM term, and an
early-termination-on-deviation predicate (:func:`deviation_terminated`).

Reference (target) values are the shipped retargeted tracks
(``data/refs_video/*.npz`` / ``data/refs/*.npz``, ``qpos_a`` (T,36) at 50 Hz):
already 2.5 Hz zero-phase low-passed, continuous-rotation-vector base-filtered
and foot-contact-anchored by ``derived/tools/retarget_video.py``.  Site targets
are forward kinematics of *that same qpos* at the 19 landmark sites, so the
site and joint families describe one reference (their difference is exactly
the solve's ``landmark_rms`` residual, 3-5 cm weighted).

Frame convention: every comparison is in ONE world frame.  Reset the robot with
the reference's own start pose (``SoloEnv.reset(pose=ref.qpos[0])``) so the
frames coincide; no re-anchoring is applied here.

Call shape for the RL wiring (``LitImplement`` owns ``solo/reward.py``)::

    from solo.imitation import (ImitationTargets, ImitationWeights,
                                imitation_reward, deviation_terminated)

    targets = ImitationTargets.from_reference("data/refs_video/shot_entry_full.npz")
    weights = ImitationWeights()            # DEFAULT_WEIGHTS, site-primary
    ...
    # once per control step (k = the episode's reference frame index at the
    # *start* of the step; the action drives k -> k+1):
    current = ImitationState.from_env(env)          # or imitation_state(env)
    target  = targets.at(k)
    terms   = imitation_reward(current, target, weights)   # {"site", "joint_pose",
                                                           #  "joint_vel", "root",
                                                           #  "com", "total"}
    if deviation_terminated(current, target, weights):
        terminate(reason="imitation_deviation")

The per-term entries of ``terms`` are ALREADY multiplied by their weights;
``terms["total"]`` is the reward scalar to add (DeepMimic adds the sum of the
weighted kernels to the task reward).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .scene import N_JOINTS, PREFIX, load_solo_model

REPO = Path(__file__).resolve().parents[2]

#: reference directories searched by :meth:`ImitationTargets.from_reference`
REFERENCE_DIRS: tuple[Path, ...] = (REPO / "data" / "refs_video", REPO / "data" / "refs")


def _landmark_names() -> tuple[str, ...]:
    """The 19 retarget landmark site names (unprefixed), in solver order."""
    from retarget.landmarks import SOLVED_SITES

    return tuple(SOLVED_SITES)


def site_weight_table(preset: str = "default") -> dict[str, dict]:
    """Per-site name -> {"class", "weight"} using the retarget class weights."""
    from retarget.landmarks import (GM_JOINT_TO_SITE, WEIGHT_CLASSES,
                                    WEIGHT_PRESETS, SOLVED_SITES)

    pw = WEIGHT_PRESETS[preset]
    out: dict[str, dict] = {}
    for joint, cls in WEIGHT_CLASSES.items():
        site = GM_JOINT_TO_SITE[joint]
        if site is None:
            continue
        cur = out.get(site)
        w = float(pw[cls])
        if cur is None or w > cur["weight"]:
            out[site] = {"class": cls, "weight": w}
    assert len(out) == len(SOLVED_SITES), (len(out), len(SOLVED_SITES))
    return out


def site_weights(preset: str = "default") -> np.ndarray:
    """(19,) weight per landmark site, aligned with :func:`_landmark_names`."""
    from retarget.landmarks import site_weights as _sw

    return np.asarray(_sw(preset), dtype=np.float64)


#: site name -> index in :func:`_landmark_names`
SITE_INDEX: dict[str, int] = {n: i for i, n in enumerate(_landmark_names())}

_WEIGHT_TABLE = None


def weight_table(preset: str = "default") -> dict[str, dict]:
    global _WEIGHT_TABLE
    if _WEIGHT_TABLE is None:
        _WEIGHT_TABLE = {}
    if preset not in _WEIGHT_TABLE:
        _WEIGHT_TABLE[preset] = site_weight_table(preset)
    return _WEIGHT_TABLE[preset]


# --------------------------------------------------------------------- state
@dataclass
class ImitationState:
    """One timestep of the imitation comparison (all arrays float64 copies)."""

    joints: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINTS))
    joint_vel: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINTS))
    root_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    root_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    #: unprefixed landmark site name -> (3,) world position (defaults to the
    #: zeroed full set, the same idiom as solo.obs.ObsContext; the constructors
    #: ``from_env``/``ImitationTargets.at`` always fill real values)
    site_pos: dict[str, np.ndarray] = field(
        default_factory=lambda: {n: np.zeros(3) for n in _landmark_names()})
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.joints = np.asarray(self.joints, dtype=np.float64).reshape(N_JOINTS)
        self.joint_vel = np.asarray(self.joint_vel, dtype=np.float64).reshape(N_JOINTS)
        self.root_pos = np.asarray(self.root_pos, dtype=np.float64).reshape(3)
        self.root_quat = np.asarray(self.root_quat, dtype=np.float64).reshape(4)
        self.com = np.asarray(self.com, dtype=np.float64).reshape(3)
        self.site_pos = {k: np.asarray(v, dtype=np.float64).reshape(3)
                         for k, v in self.site_pos.items()}

    # ------------------------------------------------------------------ read
    @classmethod
    def from_env(cls, env, model=None) -> "ImitationState":
        """Current state of a :class:`solo.env.SoloEnv` (no expert correction)."""
        return state_from_env(env, model=model)

    def site_vector(self, names: tuple[str, ...] | None = None) -> np.ndarray:
        """(19, 3) site positions in solver order (missing sites raise)."""
        names = _landmark_names() if names is None else names
        missing = [n for n in names if n not in self.site_pos]
        if missing:
            raise ValueError(f"ImitationState is missing sites: {missing}")
        return np.stack([self.site_pos[n] for n in names])


def sites_from_qpos(qpos: np.ndarray, model=None) -> dict[str, np.ndarray]:
    """FK of one G1 qpos (36,) -> 19 landmark site positions (solo scene model).

    The solo model already carries the retarget landmark sites with the ``a_``
    prefix (:mod:`solo.scene` composes them), so the reference and the simulated
    robot use the identical site set.
    """
    global _FK_CACHE
    m = model if model is not None else load_solo_model()
    if _FK_CACHE is None or _FK_CACHE[0] is not m:
        ids = []
        for n in _landmark_names():
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, PREFIX + n)
            if sid < 0:
                raise ValueError(f"model has no site {PREFIX + n!r}; "
                                 f"the solo scene must attach the retarget landmarks")
            ids.append(sid)
        _FK_CACHE = (m, mujoco.MjData(m), ids)
    m2, d, ids = _FK_CACHE
    d.qpos[:] = np.asarray(qpos, dtype=np.float64).reshape(m2.nq)
    d.qvel[:] = 0.0
    mujoco.mj_kinematics(m2, d)
    mujoco.mj_comPos(m2, d)
    out = {n: d.site_xpos[i].copy() for n, i in zip(_landmark_names(), ids)}
    return out


_FK_CACHE = None


def com_from_qpos(qpos: np.ndarray, model=None) -> np.ndarray:
    """Whole-robot CoM (3,) for one qpos (pelvis-subtree CoM of the solo model)."""
    m = model if model is not None else load_solo_model()
    d = mujoco.MjData(m)
    d.qpos[:] = np.asarray(qpos, dtype=np.float64).reshape(m.nq)
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PREFIX + "pelvis")
    if bid < 0:
        raise ValueError("model has no a_pelvis body")
    return np.asarray(d.subtree_com[bid], dtype=np.float64).copy()


def state_from_env(env, model=None) -> ImitationState:
    """Build an :class:`ImitationState` from a live ``SoloEnv`` (mujoco data)."""
    m = env.model if model is None else model
    d = env.data
    joint_q = slice(7, 36)
    joint_v = slice(6, 35)
    sites = {}
    for n in _landmark_names():
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, PREFIX + n)
        if sid < 0:
            raise ValueError(f"model has no site {PREFIX + n!r}")
        sites[n] = np.asarray(d.site_xpos[sid], dtype=np.float64).copy()
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PREFIX + "pelvis")
    return ImitationState(
        joints=np.asarray(d.qpos[joint_q], np.float64),
        joint_vel=np.asarray(d.qvel[joint_v], np.float64),
        root_pos=np.asarray(d.qpos[:3], np.float64),
        root_quat=np.asarray(d.qpos[3:7], np.float64),
        site_pos=sites,
        com=np.asarray(d.subtree_com[bid], np.float64),
    )


# ------------------------------------------------------------------- targets
def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _rotation_vectors(quat: np.ndarray) -> np.ndarray:
    """Continuous rotation-vector series from a w-first quaternion track.

    Same construction the retarget base filter uses (cumulative quaternion
    deltas; sign-unwrapped), so the derivative below is the base angular
    velocity consistent with the shipped references.
    """
    from scipy.spatial.transform import Rotation

    q = np.asarray(quat, dtype=np.float64).copy()
    for i in range(1, len(q)):
        if float(np.dot(q[i], q[i - 1])) < 0.0:
            q[i] = -q[i]
    out = np.zeros((len(q), 3))
    out[0] = Rotation.from_quat([q[0][1], q[0][2], q[0][3], q[0][0]]).as_rotvec()
    for i in range(1, len(q)):
        d = _quat_mul(np.array([q[i - 1][0], -q[i - 1][1], -q[i - 1][2], -q[i - 1][3]]),
                      q[i])
        if d[0] < 0.0:
            d = -d
        out[i] = out[i - 1] + Rotation.from_quat(
            [d[1], d[2], d[3], d[0]]).as_rotvec()
    return out


def reference_velocities(qpos: np.ndarray, t: np.ndarray):
    """Finite-difference velocities of a reference qpos track.

    Returns ``(lin_world (T,3), ang_world (T,3), joints (T,29))``.  The linear
    and angular parts are world-frame; the joint part is the hinge velocity.
    Rotation uses the continuous-rotation-vector series (no +/-pi flips) and
    ``np.gradient`` (second-order interior, first-order at the ends).
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    lin = np.gradient(qpos[:, :3], t, axis=0)
    rv = _rotation_vectors(qpos[:, 3:7])
    ang = np.gradient(rv, t, axis=0)
    joints = np.gradient(qpos[:, 7:36], t, axis=0)
    return lin, ang, joints


def resolve_reference(source) -> Path:
    """Resolve a reference name or path to an existing .npz."""
    p = Path(source)
    if p.exists():
        return p
    for d in REFERENCE_DIRS:
        for cand in (d / f"{source}.npz", d / str(source)):
            if cand.exists():
                return cand
    raise FileNotFoundError(f"reference {source!r} not found in {REFERENCE_DIRS}")


def _quat_to_R(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@dataclass
class ImitationTargets:
    """Per-frame reference targets of one retargeted npz (qpos_a)."""

    name: str
    source: str
    t: np.ndarray            # (T,) seconds, 50 Hz
    joints: np.ndarray       # (T,29)
    joint_vel: np.ndarray    # (T,29)
    root_pos: np.ndarray     # (T,3)
    root_quat: np.ndarray    # (T,4) w,x,y,z
    base_linvel_local: np.ndarray   # (T,3) pelvis-frame, matches actor obs
    base_angvel_local: np.ndarray   # (T,3) pelvis-frame, matches actor obs
    sites: np.ndarray        # (T,19,3) FK of the reference qpos at SOLVED_SITES
    com: np.ndarray          # (T,3)

    @classmethod
    def from_reference(cls, source, model=None) -> "ImitationTargets":
        path = resolve_reference(source)
        z = np.load(path, allow_pickle=True)
        qpos = np.asarray(z["qpos_a"], dtype=np.float64)
        t = np.asarray(z["t"], dtype=np.float64)
        meta = {}
        if "meta" in z:
            try:
                meta = json.loads(str(z["meta"]))
            except (TypeError, ValueError):
                meta = {}
        lin, ang, jvel = reference_velocities(qpos, t)
        sites = np.zeros((len(qpos), len(_landmark_names()), 3))
        com = np.zeros((len(qpos), 3))
        for i in range(len(qpos)):
            s = sites_from_qpos(qpos[i], model=model)
            for n, p in s.items():
                sites[i, SITE_INDEX[n]] = p
            com[i] = com_from_qpos(qpos[i], model=model)
        # base-frame velocities exactly as solo.obs/actor observes them
        bl = np.zeros_like(lin)
        ba = np.zeros_like(ang)
        for i in range(len(qpos)):
            R = _quat_to_R(qpos[i, 3:7])
            bl[i] = R.T @ lin[i]
            ba[i] = R.T @ ang[i]
        return cls(name=path.stem, source=str(path), t=t, joints=qpos[:, 7:36].copy(),
                   joint_vel=jvel, root_pos=qpos[:, :3].copy(),
                   root_quat=qpos[:, 3:7].copy(), base_linvel_local=bl,
                   base_angvel_local=ba, sites=sites, com=com)

    def __len__(self) -> int:
        return int(self.t.size)

    def at(self, i: int) -> ImitationState:
        """Target state at frame ``i`` (clamped to the track)."""
        i = int(np.clip(i, 0, len(self) - 1))
        return ImitationState(
            joints=self.joints[i], joint_vel=self.joint_vel[i],
            root_pos=self.root_pos[i], root_quat=self.root_quat[i],
            site_pos={n: self.sites[i, SITE_INDEX[n]] for n in _landmark_names()},
            com=self.com[i])

    def index_at(self, time: float) -> int:
        """Frame index at simulation time ``t`` (clamped)."""
        return int(np.clip(np.searchsorted(self.t, float(time)), 0, len(self) - 1))


# ------------------------------------------------------------------- weights
@dataclass(frozen=True)
class ImitationWeights:
    """Named, weighted terms + kernel scales + deviation thresholds.

    ``term_i = exp(-err_i^2 / scale_i^2)`` -- each ``scale_i`` is the RMS error
    at which that term has dropped to ``1/e`` (0.368).  Term weights multiply
    the kernels; the site family carries the largest weight (PRIMARY).
    """

    #: term weights (site = PRIMARY; joint family = SECONDARY, looser)
    site: float = 1.00
    joint_pose: float = 0.25
    joint_vel: float = 0.10
    root: float = 0.30
    com: float = 0.20
    #: kernel scales (RMS error at term = 1/e)
    site_scale_m: float = 0.05
    joint_pose_scale_rad: float = 0.35
    joint_vel_scale_rps: float = 1.50
    root_scale_m: float = 0.10
    com_scale_m: float = 0.08
    #: early-termination-on-deviation thresholds (strictly greater fires)
    term_joint_rad: float = 0.60          # mean |joint - joint_ref|
    term_root_xy_m: float = 0.30          # root horizontal offset
    term_pelvis_drop_m: float = 0.35      # root below the reference
    term_site_m: float = 0.12             # weighted RMS landmark error

    def as_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


DEFAULT_WEIGHTS = ImitationWeights()

#: named presets for ablations (all documented in the report)
PRESETS: dict[str, ImitationWeights] = {
    #: recommended for the refinement stage: movement geometry (sites) primary,
    #: pose shape secondary, root/CoM as stabilisers
    "site_primary": ImitationWeights(),
    #: joint-space only (DeepMimic classic at this joint space) -- kept for
    #: comparison; this is the family that inherits the IK geometric bias
    "joint_only": ImitationWeights(site=0.0, joint_pose=1.0, joint_vel=0.30,
                                   root=0.30, com=0.20),
    #: site-space only (no joint term at all)
    "site_only": ImitationWeights(joint_pose=0.0, joint_vel=0.0),
}


# -------------------------------------------------------------------- errors
def joint_pose_error(cur: ImitationState, ref: ImitationState) -> float:
    """Mean |joint - joint_ref| over the 29 hinges (rad)."""
    return float(np.mean(np.abs(cur.joints - ref.joints)))


def joint_pose_mse(cur: ImitationState, ref: ImitationState) -> float:
    """Mean squared joint error (rad^2)."""
    return float(np.mean((cur.joints - ref.joints) ** 2))


def joint_velocity_mse(cur: ImitationState, ref: ImitationState) -> float:
    """Mean squared joint-velocity error (rad^2/s^2)."""
    return float(np.mean((cur.joint_vel - ref.joint_vel) ** 2))


def site_mse(cur: ImitationState, ref: ImitationState) -> float:
    """Class-weighted mean squared landmark-site error (m^2)."""
    names = _landmark_names()
    w = site_weights()
    dp = cur.site_vector(names) - ref.site_vector(names)
    return float(np.sum(w[:, None] * dp ** 2) / np.sum(w))


def site_error_m(cur: ImitationState, ref: ImitationState) -> float:
    """Class-weighted RMS landmark-site error (m) -- the diagnostic number."""
    return float(np.sqrt(site_mse(cur, ref)))


def root_error_m(cur: ImitationState, ref: ImitationState) -> float:
    """3-D root (pelvis) position error (m)."""
    return float(np.linalg.norm(cur.root_pos - ref.root_pos))


def root_xy_error_m(cur: ImitationState, ref: ImitationState) -> float:
    """Horizontal root error (m)."""
    return float(np.linalg.norm((cur.root_pos - ref.root_pos)[:2]))


def com_error_m(cur: ImitationState, ref: ImitationState) -> float:
    """3-D whole-body CoM error (m)."""
    return float(np.linalg.norm(cur.com - ref.com))


# -------------------------------------------------------------------- terms
def joint_pose_term(cur: ImitationState, ref: ImitationState,
                    weights: ImitationWeights = DEFAULT_WEIGHTS) -> float:
    return float(np.exp(-joint_pose_mse(cur, ref) / weights.joint_pose_scale_rad ** 2))


def joint_velocity_term(cur: ImitationState, ref: ImitationState,
                        weights: ImitationWeights = DEFAULT_WEIGHTS) -> float:
    return float(np.exp(-joint_velocity_mse(cur, ref) / weights.joint_vel_scale_rps ** 2))


def site_term(cur: ImitationState, ref: ImitationState,
              weights: ImitationWeights = DEFAULT_WEIGHTS) -> float:
    """PRIMARY term: exp(-weighted landmark MSE / site_scale^2)."""
    return float(np.exp(-site_mse(cur, ref) / weights.site_scale_m ** 2))


def root_term(cur: ImitationState, ref: ImitationState,
              weights: ImitationWeights = DEFAULT_WEIGHTS) -> float:
    return float(np.exp(-root_error_m(cur, ref) ** 2 / weights.root_scale_m ** 2))


def com_term(cur: ImitationState, ref: ImitationState,
             weights: ImitationWeights = DEFAULT_WEIGHTS) -> float:
    return float(np.exp(-com_error_m(cur, ref) ** 2 / weights.com_scale_m ** 2))


def imitation_reward(cur: ImitationState, ref: ImitationState,
                     weights: ImitationWeights = DEFAULT_WEIGHTS) -> dict[str, float]:
    """Weighted DeepMimic-style terms; ``total`` is the scalar to add.

    Exact match -> every term 1.0 and ``total`` = the sum of the weights.
    """
    terms = {
        "site": weights.site * site_term(cur, ref, weights),
        "joint_pose": weights.joint_pose * joint_pose_term(cur, ref, weights),
        "joint_vel": weights.joint_vel * joint_velocity_term(cur, ref, weights),
        "root": weights.root * root_term(cur, ref, weights),
        "com": weights.com * com_term(cur, ref, weights),
    }
    terms["total"] = float(sum(terms.values()))
    return terms


def imitation_errors(cur: ImitationState, ref: ImitationState) -> dict[str, float]:
    """Raw (unweighted) errors for logging/verification."""
    return {
        "joint_rad": joint_pose_error(cur, ref),
        "joint_vel_rms": float(np.sqrt(joint_velocity_mse(cur, ref))),
        "site_m": site_error_m(cur, ref),
        "root_m": root_error_m(cur, ref),
        "root_xy_m": root_xy_error_m(cur, ref),
        "com_m": com_error_m(cur, ref),
    }


# ---------------------------------------------------------------- termination
def deviation_reason(cur: ImitationState, ref: ImitationState,
                     weights: ImitationWeights = DEFAULT_WEIGHTS) -> str | None:
    """First deviation threshold exceeded (strict ``>``), or ``None``.

    Order: joint pose, root xy, pelvis drop, site error.
    """
    if joint_pose_error(cur, ref) > weights.term_joint_rad:
        return "joint"
    if root_xy_error_m(cur, ref) > weights.term_root_xy_m:
        return "root_xy"
    if float(ref.root_pos[2] - cur.root_pos[2]) > weights.term_pelvis_drop_m:
        return "pelvis_drop"
    if site_error_m(cur, ref) > weights.term_site_m:
        return "site"
    return None


def deviation_terminated(cur: ImitationState, ref: ImitationState,
                         weights: ImitationWeights = DEFAULT_WEIGHTS) -> bool:
    """Early-termination-on-deviation predicate (see :func:`deviation_reason`)."""
    return deviation_reason(cur, ref, weights) is not None


def step_terms(env, targets: ImitationTargets, frame: int,
               weights: ImitationWeights = DEFAULT_WEIGHTS):
    """Convenience for the RL loop: (terms, deviation_reason) for one step."""
    cur = state_from_env(env)
    ref = targets.at(frame)
    return imitation_reward(cur, ref, weights), deviation_reason(cur, ref, weights)


if __name__ == "__main__":  # self-check
    import numpy as np

    # hand-constructed: exact match is maximal
    w = DEFAULT_WEIGHTS
    s = ImitationState(joints=np.full(N_JOINTS, 0.3))
    t0 = ImitationState(joints=np.full(N_JOINTS, 0.3))
    terms = imitation_reward(s, t0, w)
    assert abs(terms["total"] - (w.site + w.joint_pose + w.joint_vel + w.root + w.com)) < 1e-9
    # one joint off by delta -> the specific decrement
    t1 = ImitationState(joints=np.full(N_JOINTS, 0.3))
    delta = 0.3
    t1.joints = t1.joints.copy()
    t1.joints[0] += delta
    expect = np.exp(-(delta ** 2 / N_JOINTS) / w.joint_pose_scale_rad ** 2)
    assert abs(joint_pose_term(t1, t0, w) - expect) < 1e-12
    # deviation predicate: not below, fires above
    t2 = ImitationState(joints=np.full(N_JOINTS, 0.3 + w.term_joint_rad - 1e-9))
    assert not deviation_terminated(t2, t0, w)
    t3 = ImitationState(joints=np.full(N_JOINTS, 0.3 + w.term_joint_rad + 1e-9))
    assert deviation_terminated(t3, t0, w)
    # site weights reuse the retarget classes
    tbl = weight_table()
    assert tbl["core"]["weight"] == 4.0 and tbl["left_wrist"]["weight"] == 1.5
    assert tbl["left_toe"]["weight"] == 0.5
    # reference targets load + FK
    tg = ImitationTargets.from_reference("STANCE")
    st = tg.at(0)
    assert np.allclose(st.site_pos["core"], st.root_pos, atol=1e-12)
    assert len(st.site_pos) == 19
    print("solo.imitation self-check OK:", {"frames": len(tg),
                                            "sites": len(st.site_pos),
                                            "site_weights": site_weights().tolist()})
