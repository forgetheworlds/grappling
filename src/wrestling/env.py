"""Two-G1 standing-wrestling environment (deliverable 11, MISSION Phase 3 rules).

Scene
-----
``robots/wrestling_scene.xml``: one floor mat + two Unitree G1 humanoids,
``a_`` (qpos[0:36]) and ``b_`` (qpos[36:72]), 58 position actuators
(29 per robot, ``ctrl`` in rad, one per hinge joint; ``ctrlrange`` = joint
range).  Physics at 500 Hz (``opt.timestep=0.002``); the environment exposes a
**50 Hz control step** = 10 physics substeps.

Exchange / match rules (implemented here, see ``docs/MISSION.md`` Phase 3)
-------------------------------------------------------------------------
* A match is 180 s of simulated time (``match_clock``); exchanges run
  back-to-back inside it.
* Every exchange starts from a standing stance (the STANCE reference start
  pose by default, randomized per seed -- see :meth:`WrestlingEnv.reset`);
  when one wrestler's back is clearly taken to the mat, the other scores and
  both reset standing.
* Back-to-mat is decided by :class:`wrestling.backdet.BackToMatDetector`
  (dorsal torso mat contact + torso tilt + low pelvis, sustained); a single
  collision bit never ends an exchange.  Knees/hands/sprawl are legal.
* Simultaneous back events: the first trigger wins; if both robots trigger
  within ``AMBIGUITY_WINDOW_S`` (0.10 s) the exchange is *ambiguous*: no
  score, logged for review (MISSION).
* Anti-run-away: the competition area is a circle of radius ``mat_radius``
  (default :data:`MAT_RADIUS` = 1.5 m) centered at the world origin -- the
  scene's floor is an unbounded plane, so the mat extent is declared here.
  A pelvis leaving the area is an out-of-bounds **event** (with 5 cm re-entry
  hysteresis); each event is penalized and ``OOB_EVENTS_TO_FORFEIT`` (3)
  events in one exchange forfeit the exchange.
* Exchange timeout: 20 s of sim time without a back event -> draw, reset.

Controllers
-----------
``env.step(action=None)`` uses the two pluggable controller callables
(``controllers=(ctrl_a, ctrl_b)``) when ``action`` is None.  A controller is
``callable(env, data) -> array(29,) | None`` and may read ``env.exchange_time``
/ ``env.exchange_index`` / ``env.time``.  Two scripted controllers ship here:

* :class:`ReferenceReplay` -- PD-tracks a ``data/refs/<TECHNIQUE>.npz`` trace
  with the model's position servos (``ctrl = q_ref + fb*(q_ref - q)``).
* :class:`StandHold` -- static hold of the model's verified-stable ``stand``
  keyframe targets (used by tests: it is the one pose that stays up >20 s
  open-loop).

Measured (2026-10-08, this host): the retargeted STANCE crouch is **not**
statically holdable by pure joint targets -- without balance control it
topples backward after ~1.4 s (see ``reports/2026-10-08/wrestling_env.md``).
That is by design out of scope for phase 2/4 (keeping a humanoid balanced in
wrestling postures is the phase-3 teacher's job); the environment therefore
starts from that stance and exposes the imbalance as part of the task.

Observation / action / reward contract (consumed by Phase 5 PPO)
----------------------------------------------------------------
* action: ``float32 (2, 29)`` or ``(58,)`` joint position targets in rad,
  clipped to ``ctrlrange``.  ``env.action_dim == 58``.
* observation: two ``float32 (84,)`` vectors (one per robot) from
  ``obs_fn(model, data)`` -- defaults to :func:`default_observation`
  (layout documented there and via :func:`obs_layout`).  Swappable:
  ``env.obs_fn = my_fn``; the final obs spec is a Phase-5 decision
  (deliverable 14).
* reward: ``(r_a, r_b)`` per control step, produced by the pluggable hook
  ``reward_fn(env, event) -> (float, float)`` (default
  :func:`default_reward`).  Events in an exchange: ``{"kind": "oob", ...}``
  (-0.25 to the offender), ``{"kind": "exchange_end", ...}`` (+1 winner,
  -1 loser, 0 draw/ambiguous).  No per-step shaping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import mujoco
import numpy as np

from .backdet import ROBOTS, BackDetConfig, BackToMatDetector

REPO = Path(__file__).resolve().parents[2]
SCENE_XML = REPO / "robots" / "wrestling_scene.xml"
REF_DIR = REPO / "data" / "refs"

CONTROL_HZ = 50
STEP_DT = 1.0 / CONTROL_HZ
MODEL_DT = 0.002
SUBSTEPS = 10
N_JOINTS = 29
QPOS_SLICE = {"a": slice(0, 36), "b": slice(36, 72)}
QVEL_SLICE = {"a": slice(0, 35), "b": slice(35, 70)}
ACT_SLICE = {"a": slice(0, 29), "b": slice(29, 58)}
PREFIX = {"a": "a_", "b": "b_"}

#: declared competition area (m), see module docstring
MAT_RADIUS = 1.5
#: re-entry hysteresis for OOB events (m)
OOB_REENTRY_HYSTERESIS = 0.05
OOB_EVENTS_TO_FORFEIT = 3
AMBIGUITY_WINDOW_S = 0.10
EXCHANGE_TIMEOUT = 20.0
MATCH_CLOCK = 180.0
#: per-exchange RNG stream: seed_i = base_seed + i * EXCHANGE_SEED_STRIDE
EXCHANGE_SEED_STRIDE = 10007
#: start-position randomization (MISSION Phase 3: distance/angle/joint noise)
START_DISTANCE_JITTER = 0.15   # m, +-
START_ANGLE_JITTER_DEG = 10.0  # deg, +-
START_JOINT_NOISE = 0.03       # rad, +-

OBS_DIM = 84
_OBS_LAYOUT = (
    ("self_base_pos", (0, 3)),
    ("self_base_quat", (3, 7)),
    ("self_joints", (7, 36)),
    ("self_base_linvel", (36, 39)),
    ("self_base_angvel", (39, 42)),
    ("self_joint_vel", (42, 71)),
    ("self_torso_up", (71, 74)),
    ("self_foot_contact", (74, 76)),
    ("opp_rel_pos", (76, 79)),
    ("opp_rel_vel", (79, 82)),
    ("opp_rel_heading", (82, 84)),
)


def obs_layout() -> tuple:
    """(name, (start, stop)) per observation block; identical for both robots."""
    return _OBS_LAYOUT


_model_cache: mujoco.MjModel | None = None


def load_wrestling_model() -> mujoco.MjModel:
    """Compiled two-robot scene (nq=72, nu=58), cached per process."""
    global _model_cache
    if _model_cache is None:
        m = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        assert m.nq == 72 and m.nu == 58, (m.nq, m.nu)
        _model_cache = m
    return _model_cache


def reference_trace(technique: str) -> dict:
    """Load ``data/refs/<technique>.npz`` (read-only, cached)."""
    cache = getattr(reference_trace, "_cache", None)
    if cache is None:
        cache = {}
        reference_trace._cache = cache
    if technique not in cache:
        z = np.load(REF_DIR / f"{technique}.npz", allow_pickle=False)
        cache[technique] = {
            "qpos_a": np.asarray(z["qpos_a"], dtype=np.float64),
            "qpos_b": np.asarray(z["qpos_b"], dtype=np.float64),
            "t": np.asarray(z["t"], dtype=np.float64),
            "meta": json.loads(str(z["meta"])),
        }
    return cache[technique]


def stance_start_qpos() -> tuple[np.ndarray, np.ndarray]:
    """(qpos_a, qpos_b) of the first STANCE reference frame."""
    tr = reference_trace("STANCE")
    return tr["qpos_a"][0].copy(), tr["qpos_b"][0].copy()


def keyframe_pair_qpos(model: mujoco.MjModel, name: str = "both_stand") -> tuple[np.ndarray, np.ndarray]:
    """(qpos_a, qpos_b) from a model keyframe holding all 72 qpos values."""
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if kid < 0:
        raise KeyError(f"keyframe {name!r} not in model")
    q = np.asarray(model.key_qpos[kid], dtype=np.float64)
    return q[0:36].copy(), q[36:72].copy()


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of (w,x,y,z) quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _yaw_quat(angle: float) -> np.ndarray:
    return np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])


def _foot_geom_ids(model, prefix: str) -> tuple[frozenset, frozenset]:
    """(left, right) collision geom ids of the ankle_roll bodies (foot spheres + shells)."""
    sets = []
    for side in ("left", "right"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{side}_ankle_roll_link")
        if bid < 0:
            raise KeyError(f"{prefix}{side}_ankle_roll_link not in model")
        sets.append(frozenset(g for g in range(model.ngeom)
                              if int(model.geom_bodyid[g]) == bid))
    return sets[0], sets[1]


def default_observation(model: mujoco.MjModel, data: mujoco.MjData):
    """Interim obs scaffold (Phase-5 spec is deliverable 14): ``(84,)`` float32
    per robot, world frame for self, self frame for the opponent.

    Per robot ``r`` with opponent ``o`` (identity block from :func:`obs_layout`):

    ====================  ====================================================
    ``self_base_pos``     pelvis xyz in world (3)
    ``self_base_quat``    pelvis quaternion w,x,y,z in world (4)
    ``self_joints``       29 hinge positions in model order (rad)
    ``self_base_linvel``  pelvis linear velocity in world (3, m/s)
    ``self_base_angvel``  pelvis angular velocity in world (3, rad/s)
    ``self_joint_vel``    29 hinge velocities (rad/s)
    ``self_torso_up``     torso_link local +z in world (3)
    ``self_foot_contact`` any floor contact of left/right foot geoms (2, 0/1)
    ``opp_rel_pos``       opponent pelvis minus self pelvis in self frame (3)
    ``opp_rel_vel``       opponent minus self pelvis velocity in self frame (3)
    ``opp_rel_heading``   sin/cos of (opponent yaw - self yaw) (2)
    ====================  ====================================================
    """
    from .backdet import FLOOR_GEOM, body_maps  # local import: shared helper

    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM)
    feats = {}
    for r in ROBOTS:
        q = data.qpos[QPOS_SLICE[r]]
        v = data.qvel[QVEL_SLICE[r]]
        maps = body_maps(model, r)
        pid, tid = maps.pelvis_bid, maps.torso_bid
        R = data.xmat[pid].reshape(3, 3)
        torso_up = data.xmat[tid].reshape(3, 3)[:, 2].copy()
        left_geoms, right_geoms = _foot_geom_ids(model, PREFIX[r])
        contact = np.zeros(2)
        for c in range(data.ncon):
            con = data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 == floor and g2 in left_geoms or g2 == floor and g1 in left_geoms:
                contact[0] = 1.0
            elif g1 == floor and g2 in right_geoms or g2 == floor and g1 in right_geoms:
                contact[1] = 1.0
        feats[r] = {
            "pos": np.asarray(data.xpos[pid], dtype=np.float64).copy(),
            "quat": np.asarray(data.xquat[pid], dtype=np.float64).copy(),
            "joints": np.asarray(q[7:36], dtype=np.float64).copy(),
            "linvel": np.asarray(v[0:3], dtype=np.float64).copy(),
            "angvel": np.asarray(v[3:6], dtype=np.float64).copy(),
            "joint_vel": np.asarray(v[6:35], dtype=np.float64).copy(),
            "torso_up": torso_up,
            "foot": contact,
            "R": R,
        }
    out = []
    for r in ROBOTS:
        o = "b" if r == "a" else "a"
        f, g = feats[r], feats[o]
        R = f["R"]
        dpos = R.T @ (g["pos"] - f["pos"])
        dvel = R.T @ (g["linvel"] - f["linvel"])
        yaw_r = np.arctan2(R[1, 0], R[0, 0])
        Ro = g["R"]
        yaw_o = np.arctan2(Ro[1, 0], Ro[0, 0])
        rel = yaw_o - yaw_r
        out.append(np.concatenate([
            f["pos"], f["quat"], f["joints"], f["linvel"], f["angvel"],
            f["joint_vel"], f["torso_up"], f["foot"], dpos, dvel,
            [np.sin(rel), np.cos(rel)],
        ]).astype(np.float32))
    return out[0], out[1]


def default_reward(env: "WrestlingEnv", event: dict) -> tuple[float, float]:
    """Minimal game-structure reward: back-to-mat wins exchanges, OOB costs.

    * ``exchange_end``: +1 / -1 to winner / loser; 0 for draw or ambiguous.
    * ``oob``: -0.25 to the robot that left the area.
    """
    if event["kind"] == "oob":
        return tuple(-0.25 if p == event["robot"] else 0.0 for p in ROBOTS)
    if event["kind"] == "exchange_end":
        winner = event.get("winner")
        if winner in ROBOTS:
            return tuple(1.0 if p == winner else -1.0 for p in ROBOTS)
        return (0.0, 0.0)
    return (0.0, 0.0)


def resolve_back_events(triggers: dict, ambiguity_window: float = AMBIGUITY_WINDOW_S):
    """Winner / ambiguity from the per-robot back-trigger times (s, or None).

    Returns ``(winner, ambiguous)``: the first trigger wins; if both robots
    triggered within ``ambiguity_window`` the exchange is ambiguous (no score);
    ``(None, False)`` when nothing triggered.
    """
    ta, tb = triggers.get("a"), triggers.get("b")
    if ta is None and tb is None:
        return None, False
    if ta is not None and tb is not None:
        if abs(ta - tb) <= ambiguity_window:
            return None, True
        # the earlier trigger is the wrestler whose back was taken
        return ("b" if ta < tb else "a"), False
    # only one back was taken: the other wrestler wins
    return ("b" if ta is not None else "a"), False


class ReferenceReplay:
    """Replay a ``data/refs`` joint trajectory through the position servos.

    ``ctrl = q_ref(t) + feedback_gain * (q_ref(t) - q_actual)`` where
    ``t = env.exchange_time`` (restarts at every exchange).  After the trace
    ends the final frame is held (``hold_last=True``) or the trace loops.

    This is the scripted stand-in used to test the environment before the
    phase-3 teacher exists.  Measured caveat: open-loop traces are not
    balance-controlled (STANCE topples at ~1.4 s); the teacher is responsible
    for standing control.
    """

    def __init__(self, technique: str, robot: str, feedback_gain: float = 0.0,
                 hold_last: bool = True, loop: bool = False):
        if robot not in ROBOTS:
            raise ValueError(robot)
        tr = reference_trace(technique)
        self.technique = technique
        self.robot = robot
        self.feedback_gain = float(feedback_gain)
        self.hold_last = bool(hold_last)
        self.loop = bool(loop)
        self.q_ref = tr["qpos_a"] if robot == "a" else tr["qpos_b"]
        self.t_ref = tr["t"]
        self.duration = float(self.t_ref[-1])

    def _sample(self, t: float) -> np.ndarray:
        if self.loop and self.duration > 0:
            t = t % self.duration
        elif t >= self.t_ref[-1]:
            t = self.t_ref[-1]
        i = int(np.clip(np.searchsorted(self.t_ref, t), 1, len(self.t_ref) - 1))
        w = (t - self.t_ref[i - 1]) / (self.t_ref[i] - self.t_ref[i - 1])
        return self.q_ref[i - 1, 7:36] * (1 - w) + self.q_ref[i, 7:36] * w

    def __call__(self, env: "WrestlingEnv", data: mujoco.MjData) -> np.ndarray:
        q = self._sample(float(env.exchange_time))
        if self.feedback_gain:
            q = q + self.feedback_gain * (q - np.asarray(data.qpos[QPOS_SLICE[self.robot]][7:36]))
        return q


class StandHold:
    """Hold the model's verified-stable ``stand`` keyframe targets (one robot).

    ``ctrl`` = the robot's ``a_stand`` / ``b_stand`` keyframe ctrl values.
    Verified: starting from the ``both_stand`` keyframe qpos the pair stands
    with <0.2 mm drift for >20 s (``scripts``/tests measure this).
    """

    def __init__(self, robot: str):
        if robot not in ROBOTS:
            raise ValueError(robot)
        self.robot = robot
        model = load_wrestling_model()
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, f"{robot}_stand")
        if kid < 0:
            raise KeyError(f"keyframe {robot}_stand not in model")
        base = 0 if robot == "a" else 29
        self.target = np.asarray(model.key_ctrl[kid], dtype=np.float64)[base:base + 29].copy()

    def __call__(self, env: "WrestlingEnv", data: mujoco.MjData) -> np.ndarray:
        return self.target


@dataclass
class ExchangeRecord:
    """One exchange of the match (logged for review / Phase-7 evaluation)."""

    index: int
    start_time: float          # match clock at exchange start (s)
    end_time: float            # match clock at exchange end (s)
    duration: float            # s
    winner: str | None
    loser: str | None
    cause: str                 # 'back' | 'oob' | 'timeout' | 'match_end'
    ambiguous: bool
    oob_events: dict
    back_triggers: dict        # simulator time of the first back trigger per robot
    end_back: dict             # detector snapshot at the end (tilt/pelvis/contact)

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "start_time": round(float(self.start_time), 4),
            "end_time": round(float(self.end_time), 4),
            "duration": round(float(self.duration), 4),
            "winner": self.winner,
            "loser": self.loser,
            "cause": self.cause,
            "ambiguous": bool(self.ambiguous),
            "oob_events": dict(self.oob_events),
            "back_triggers": {k: (None if v is None else round(float(v), 4))
                              for k, v in self.back_triggers.items()},
            "end_back": self.end_back,
        }


class WrestlingEnv:
    """Standing-wrestling environment; see module docstring for the contract."""

    def __init__(
        self,
        model: mujoco.MjModel | None = None,
        controllers: tuple | None = None,
        seed: int = 0,
        *,
        exchange_timeout: float = EXCHANGE_TIMEOUT,
        match_clock: float = MATCH_CLOCK,
        mat_radius: float = MAT_RADIUS,
        backdet_cfg: BackDetConfig | None = None,
        obs_fn: Callable | None = None,
        reward_fn: Callable | None = None,
        substeps: int = SUBSTEPS,
    ):
        self.model = model if model is not None else load_wrestling_model()
        assert abs(self.model.opt.timestep - MODEL_DT) < 1e-9, self.model.opt.timestep
        self.n_substeps = int(substeps)
        assert abs(self.n_substeps * self.model.opt.timestep - STEP_DT) < 1e-9
        self.data = mujoco.MjData(self.model)
        self.exchange_timeout = float(exchange_timeout)
        self.match_clock = float(match_clock)
        self.mat_radius = float(mat_radius)
        self.backdet = BackToMatDetector(backdet_cfg or BackDetConfig())
        self.obs_fn = obs_fn or default_observation
        self.reward_fn = reward_fn or default_reward
        self.controllers: tuple = tuple(controllers) if controllers else (None, None)
        self.base_seed = int(seed)
        self.exchange_log: list[ExchangeRecord] = []
        self._pelvis_bid = {r: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                                 f"{PREFIX[r]}pelvis") for r in ROBOTS}
        lo = np.asarray(self.model.actuator_ctrlrange[:, 0], dtype=np.float64)
        hi = np.asarray(self.model.actuator_ctrlrange[:, 1], dtype=np.float64)
        self._joint_limits = ((lo[:N_JOINTS].copy(), hi[:N_JOINTS].copy()),
                              (lo[N_JOINTS:].copy(), hi[N_JOINTS:].copy()))
        self.reset(seed=self.base_seed)

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int | None = None, pose: tuple | None = None):
        """Start a fresh match at the standing start; returns ``(obs_a, obs_b)``.

        ``seed`` (default: the constructor seed) drives, in this fixed order:
        pair yaw ``U(-10, 10) deg`` about the pair midpoint, then pelvis
        separation ``d0 + U(-0.15, 0.15) m``, then joint noise
        ``U(-0.03, 0.03) rad`` for robot ``a`` (model joint order) then robot
        ``b``.  Same seed -> identical qpos.

        ``pose=(qpos_a, qpos_b)`` (36,) overrides the randomized stance
        entirely (test/calibration hook: e.g. a reference's own first frame).
        """
        if seed is not None:
            self.base_seed = int(seed)
        mujoco.mj_resetData(self.model, self.data)
        qa, qb = pose if pose is not None else self._randomized_stance(self._exchange_seed(0))
        self.data.qpos[QPOS_SLICE["a"]] = qa
        self.data.qpos[QPOS_SLICE["b"]] = qb
        self.data.qvel[:] = 0.0
        # default action: hold the start pose joints (used when action is None
        # and no controllers are set)
        self.data.ctrl[:] = np.concatenate([qa[7:36], qb[7:36]])
        mujoco.mj_forward(self.model, self.data)
        self._hold_targets = (qa[7:36].copy(), qb[7:36].copy())
        self.exchange_index = 0
        self.exchange_start = float(self.data.time)
        self.exchange_log = []
        self._match_start = float(self.data.time)
        self._oob = {r: 0 for r in ROBOTS}
        self._outside = {r: False for r in ROBOTS}
        self.backdet.reset()
        self.match_over = False
        self.last_exchange: ExchangeRecord | None = None
        return self._obs()

    def _exchange_seed(self, index: int) -> int:
        return self.base_seed + EXCHANGE_SEED_STRIDE * index

    def _randomized_stance(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        qa, qb = stance_start_qpos()
        mid = 0.5 * (qa[0:3] + qb[0:3])
        d_vec = qb[0:2] - qa[0:2]
        d0 = float(np.linalg.norm(d_vec))
        yaw = np.deg2rad(rng.uniform(-START_ANGLE_JITTER_DEG, START_ANGLE_JITTER_DEG))
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s], [s, c]])
        dist = d0 + rng.uniform(-START_DISTANCE_JITTER, START_DISTANCE_JITTER)
        axis = Rz @ (d_vec / d0)
        noise_a = rng.uniform(-START_JOINT_NOISE, START_JOINT_NOISE, N_JOINTS)
        noise_b = rng.uniform(-START_JOINT_NOISE, START_JOINT_NOISE, N_JOINTS)
        out = []
        for (q, side, noise), limit in zip(
                ((qa, -1.0, noise_a), (qb, +1.0, noise_b)), self._joint_limits):
            pos = q.copy()
            for i, p in enumerate((0, 1)):
                pos[p] = mid[p] + (Rz @ (q[0:2] - mid[0:2]))[i]
            pos[0:2] += side * (dist - d0) / 2.0 * axis
            pos[3:7] = _quat_mul(_yaw_quat(yaw), q[3:7])
            pos[7:36] = np.clip(q[7:36] + noise, limit[0], limit[1])
            out.append(pos)
        return out[0], out[1]

    # ---------------------------------------------------------------- stepping
    @property
    def time(self) -> float:
        """Simulator time (s)."""
        return float(self.data.time)

    @property
    def match_time(self) -> float:
        return float(self.data.time) - self._match_start

    @property
    def exchange_time(self) -> float:
        return float(self.data.time) - self.exchange_start

    @property
    def action_dim(self) -> int:
        return 2 * N_JOINTS

    def set_controllers(self, ctrl_a, ctrl_b) -> None:
        """Install/replace the two controller callables ``(env, data) -> (29,)``."""
        self.controllers = (ctrl_a, ctrl_b)

    def step(self, action: Sequence | None = None):
        """Advance one 50 Hz control step (10 physics substeps).

        Returns ``(obs, reward, terminated, truncated, info)`` with
        ``obs = (obs_a, obs_b)`` float32 (84,) each, ``reward = (r_a, r_b)``
        floats, ``terminated`` True when the match clock has expired and
        ``truncated`` always False (the match clock is the episode's natural
        end).  ``info`` carries the exchange clock, OOB counts, the detector
        snapshot and the exchange record if one ended this step.
        """
        u = self._resolve_action(action)
        ctrl_lo = self.model.actuator_ctrlrange[:, 0]
        ctrl_hi = self.model.actuator_ctrlrange[:, 1]
        self.data.ctrl[:] = np.clip(u, ctrl_lo, ctrl_hi)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        rewards = [0.0, 0.0]
        events: list[dict] = []

        self.backdet.update(self.model, self.data, float(self.data.time))
        for robot in ROBOTS:
            pos = self.data.xpos[self._pelvis_bid[robot]]
            radius = float(np.hypot(pos[0], pos[1]))
            if self._outside[robot]:
                if radius < self.mat_radius - OOB_REENTRY_HYSTERESIS:
                    self._outside[robot] = False
            elif radius > self.mat_radius:
                self._outside[robot] = True
                self._oob[robot] += 1
                events.append({"kind": "oob", "robot": robot, "count": self._oob[robot]})

        record = self._maybe_end_exchange()
        if record is not None:
            events.append({"kind": "exchange_end", "winner": record.winner,
                           "loser": record.loser, "cause": record.cause,
                           "ambiguous": record.ambiguous, "record": record})
        for ev in events:
            ra, rb = self.reward_fn(self, ev)
            rewards[0] += float(ra)
            rewards[1] += float(rb)

        info = {
            "t": self.time,
            "match_time": self.match_time,
            "exchange_index": self.exchange_index,
            "exchange_time": self.exchange_time,
            "oob": dict(self._oob),
            "back": self.backdet.status(),
            "exchange_ended": None if record is None else record.as_dict(),
            "match_over": self.match_over,
        }
        return self._obs(), (rewards[0], rewards[1]), bool(self.match_over), False, info

    def _resolve_action(self, action) -> np.ndarray:
        if action is None:
            acts = []
            for i, robot in enumerate(ROBOTS):
                ctrl = self.controllers[i]
                if ctrl is None:
                    acts.append(self._hold_targets[i])
                else:
                    out = ctrl(self, self.data)
                    acts.append(self._hold_targets[i] if out is None else np.asarray(out, dtype=np.float64))
            return np.concatenate([np.asarray(a, dtype=np.float64).reshape(N_JOINTS) for a in acts])
        a = np.asarray(action, dtype=np.float64)
        if a.shape == (2, N_JOINTS):
            return a.reshape(2 * N_JOINTS)
        if a.shape == (2 * N_JOINTS,):
            return a
        raise ValueError(f"action shape {a.shape}: expected (2, 29) or (58,)")

    # ----------------------------------------------------------- exchange loop
    def _maybe_end_exchange(self) -> ExchangeRecord | None:
        triggers = self.backdet.triggers
        cause: str | None = None
        winner: str | None = None
        ambiguous = False

        if any(t is not None for t in triggers.values()):
            winner, ambiguous = resolve_back_events(triggers)
            cause = "back"
        if cause is None:
            for robot in ROBOTS:
                if self._oob[robot] >= OOB_EVENTS_TO_FORFEIT:
                    winner = "b" if robot == "a" else "a"
                    cause = "oob"
                    break
        if cause is None and self.exchange_time >= self.exchange_timeout - 1e-9:
            cause = "timeout"
        if cause is None and self.match_time >= self.match_clock - 1e-9:
            cause = "match_end"

        if cause is None:
            return None

        loser = None
        if winner is not None:
            loser = "b" if winner == "a" else "a"
        end_back = {
            r: {
                "trigger_t": None if triggers[r] is None else round(float(triggers[r]), 4),
                "dorsal": bool(self.backdet.last_features[r].dorsal_contact),
                "tilt_deg": round(float(self.backdet.last_features[r].tilt_deg), 2),
                "pelvis_z": round(float(self.backdet.last_features[r].pelvis_z), 4),
            }
            for r in ROBOTS
        }
        record = ExchangeRecord(
            index=self.exchange_index,
            start_time=self.exchange_start - self._match_start,
            end_time=self.match_time,
            duration=self.exchange_time,
            winner=winner,
            loser=loser,
            cause=cause,
            ambiguous=ambiguous,
            oob_events=dict(self._oob),
            back_triggers={r: triggers[r] for r in ROBOTS},
            end_back=end_back,
        )
        self.exchange_log.append(record)
        self.last_exchange = record

        if cause == "match_end" or self.match_time >= self.match_clock - 1e-9:
            self.match_over = True
        else:
            self._start_exchange()
        return record

    def _start_exchange(self) -> None:
        self.exchange_index += 1
        qa, qb = self._randomized_stance(self._exchange_seed(self.exchange_index))
        self.data.qpos[QPOS_SLICE["a"]] = qa
        self.data.qpos[QPOS_SLICE["b"]] = qb
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = np.concatenate([qa[7:36], qb[7:36]])
        mujoco.mj_forward(self.model, self.data)
        self._hold_targets = (qa[7:36].copy(), qb[7:36].copy())
        self.exchange_start = float(self.data.time)
        self._oob = {r: 0 for r in ROBOTS}
        self._outside = {r: False for r in ROBOTS}
        self.backdet.reset()

    # ------------------------------------------------------------------ extras
    def _obs(self):
        obs_a, obs_b = self.obs_fn(self.model, self.data)
        return obs_a, obs_b

    def back_snapshot(self) -> dict:
        """Detector status for both robots (analysis/debug)."""
        return self.backdet.status()


if __name__ == "__main__":  # self-check
    m = load_wrestling_model()
    env = WrestlingEnv(model=m, seed=7)
    qa = env.data.qpos[QPOS_SLICE["a"]].copy()
    env2 = WrestlingEnv(model=m, seed=7)
    assert np.allclose(qa, env2.data.qpos[QPOS_SLICE["a"]]), "reset not deterministic"
    obs, reward, term, trunc, info = env.step()
    assert obs[0].shape == (OBS_DIM,) and obs[0].dtype == np.float32
    print("wrestling env self-check OK:", {"obs_dim": int(obs[0].size),
                                           "action_dim": env.action_dim,
                                           "match_clock": env.match_clock})
