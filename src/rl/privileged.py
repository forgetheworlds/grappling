"""Privileged (critic-only) state: full simulator state, exact contacts, and a
graded back-exposure proxy.

MISSION ("PPO critic"): the critic may use privileged state -- complete robot
states, precise contact information, exact opponent velocities, distance to
back-exposure terminal state -- while the actor must not.  This module is the
*only* place privileged features are produced; it never feeds the actor path
and :class:`CriticObsBuilder` concatenates actor obs with privileged obs
behind an explicit layout whose first block is byte-identical to the actor obs.

Layout (``PRIV_LAYOUT``), per control step, float32, dim :data:`PRIV_DIM`
(162 = 142 state + 12 contact + 8 back-exposure):

================  =====================================================
``qpos_a``        robot a free joint + 29 hinges, model order (36)
``qpos_b``        robot b (36)
``qvel_a``        robot a base + 29 joints (35)
``qvel_b``        robot b (35)
``contact_a``     floor-contact flags [feet, knees, hands, torso, pelvis]
                  + opponent-contact count (6)
``contact_b``     same for robot b (6)
``back_a``        [tilt/90, pelvis_z, dorsal_contact, exposure] (4)
``back_b``        same for robot b (4)
================  =====================================================

``exposure`` in ``[0, 1]`` is the graded "distance to back exposure" proxy:
``clip((tilt - 45)/45) * clip((0.35 - pelvis_z)/0.35)`` using the calibrated
:class:`wrestling.backdet.BackDetConfig` thresholds -- 0.0 in clearly safe
postures, 1.0 once both detector thresholds are (met or) exceeded.

Contact categories are computed here from model body names (independent of
``wrestling.backdet``'s diagnostics): feet = ``ankle_roll_link`` (the env's
own foot-contact definition), knees = ``knee_link``, hands = ``wrist_*_link``,
torso = ``torso_link``, pelvis = ``pelvis``.
"""

from __future__ import annotations

import mujoco
import numpy as np

from wrestling.backdet import BackDetConfig, back_features

from .obs import ROBOTS

#: per-robot contact flags
CONTACT_KINDS: tuple[str, ...] = ("feet", "knees", "hands", "torso", "pelvis")
CONTACT_DIM = len(CONTACT_KINDS) + 1  # + opponent-contact count
BACK_DIM = 4

_FLOOR_GEOM = "floor"


def back_exposure(tilt_deg: float, pelvis_z: float, cfg: BackDetConfig | None = None) -> float:
    """Graded back-exposure proxy in ``[0, 1]`` (detector-threshold consistent)."""
    c = cfg or BackDetConfig()
    tilt_term = float(np.clip((tilt_deg - c.tilt_threshold_deg) / c.tilt_threshold_deg, 0.0, 1.0))
    low_term = float(np.clip((c.pelvis_z_threshold - pelvis_z) / c.pelvis_z_threshold, 0.0, 1.0))
    return tilt_term * low_term


PRIV_LAYOUT: tuple[tuple[str, int, int], ...] = (
    ("qpos_a", 0, 36),
    ("qpos_b", 36, 72),
    ("qvel_a", 72, 107),
    ("qvel_b", 107, 142),
    ("contact_a", 142, 142 + CONTACT_DIM),
    ("contact_b", 142 + CONTACT_DIM, 142 + 2 * CONTACT_DIM),
    ("back_a", 142 + 2 * CONTACT_DIM, 142 + 2 * CONTACT_DIM + BACK_DIM),
    ("back_b", 142 + 2 * CONTACT_DIM + BACK_DIM, 142 + 2 * CONTACT_DIM + 2 * BACK_DIM),
)
PRIV_DIM = PRIV_LAYOUT[-1][2]
#: qpos_a(36) + qpos_b(36) + qvel_a(35) + qvel_b(35)
_STATE_DIM = 142
assert PRIV_LAYOUT[4][1] == _STATE_DIM, "layout/state-block mismatch"

#: model-order slices per robot (mirror of wrestling.env, pinned by tests)
_QPOS_SLICE = {"a": slice(0, 36), "b": slice(36, 72)}
_QVEL_SLICE = {"a": slice(0, 35), "b": slice(35, 70)}


def priv_layout() -> tuple[tuple[str, int, int], ...]:
    """((name, start, stop), ...) of the privileged block."""
    return PRIV_LAYOUT


def unpack(priv: np.ndarray) -> dict:
    """Named views/copies of a ``(PRIV_DIM,)`` privileged vector (debug/eval)."""
    p = np.asarray(priv, dtype=np.float32).reshape(PRIV_DIM)
    out = {}
    for name, start, stop in PRIV_LAYOUT:
        out[name] = p[start:stop]
    return out


class ContactCategorizer:
    """Geom-id sets per robot/contact category, resolved once per model."""

    def __init__(self, model: mujoco.MjModel):
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, _FLOOR_GEOM)
        if floor < 0:
            raise KeyError(f"geom {_FLOOR_GEOM!r} not in model")
        self.floor_geom = int(floor)
        self._cats: dict[str, dict[str, frozenset]] = {}
        self._all_geoms: dict[str, frozenset] = {}
        for robot in ROBOTS:
            cats: dict[str, set] = {k: set() for k in CONTACT_KINDS}
            all_geoms: set = set()
            for g in range(model.ngeom):
                body = int(model.geom_bodyid[g])
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
                if not name.startswith(f"{robot}_"):
                    continue
                all_geoms.add(g)
                if "ankle_roll_link" in name:
                    cats["feet"].add(g)
                if "knee_link" in name:
                    cats["knees"].add(g)
                if "wrist_" in name:
                    cats["hands"].add(g)
                if name == f"{robot}_torso_link":
                    cats["torso"].add(g)
                if name == f"{robot}_pelvis":
                    cats["pelvis"].add(g)
            self._cats[robot] = {k: frozenset(v) for k, v in cats.items()}
            self._all_geoms[robot] = frozenset(all_geoms)

    def floor_flags(self, data: mujoco.MjData, robot: str) -> np.ndarray:
        """``(5,)`` float32 floor-contact flags in :data:`CONTACT_KINDS` order."""
        flags = np.zeros(len(CONTACT_KINDS), dtype=np.float32)
        cats = self._cats[robot]
        for c in range(data.ncon):
            con = data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 == self.floor_geom:
                other = g2
            elif g2 == self.floor_geom:
                other = g1
            else:
                continue
            for i, kind in enumerate(CONTACT_KINDS):
                if other in cats[kind]:
                    flags[i] = 1.0
        return flags

    def opponent_contact_count(self, data: mujoco.MjData, robot: str) -> float:
        """Number of contacts between ``robot`` and the other robot's geoms."""
        other = "b" if robot == "a" else "a"
        mine, theirs = self._all_geoms[robot], self._all_geoms[other]
        n = 0
        for c in range(data.ncon):
            con = data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if (g1 in mine and g2 in theirs) or (g1 in theirs and g2 in mine):
                n += 1
        return float(n)


class PrivilegedObsBuilder:
    """Builds the ``(PRIV_DIM,)`` privileged vector from ``MjData``."""

    def __init__(self, model: mujoco.MjModel, det_cfg: BackDetConfig | None = None):
        self.categorizer = ContactCategorizer(model)
        self.det_cfg = det_cfg or BackDetConfig()
        self.dim = PRIV_DIM

    def build(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        out = np.empty(PRIV_DIM, dtype=np.float32)
        # state block: qpos_a | qpos_b | qvel_a | qvel_b (source slices are the
        # env's per-robot qpos/qvel offsets; destinations are the layout offsets)
        state = np.concatenate([
            data.qpos[_QPOS_SLICE["a"]], data.qpos[_QPOS_SLICE["b"]],
            data.qvel[_QVEL_SLICE["a"]], data.qvel[_QVEL_SLICE["b"]],
        ])
        if state.shape != (_STATE_DIM,):  # pragma: no cover - layout guard
            raise AssertionError(f"state block {state.shape} != ({_STATE_DIM},)")
        out[:_STATE_DIM] = state
        for robot in ROBOTS:
            i = _STATE_DIM if robot == "a" else _STATE_DIM + CONTACT_DIM
            out[i:i + len(CONTACT_KINDS)] = self.categorizer.floor_flags(data, robot)
            out[i + len(CONTACT_KINDS)] = self.categorizer.opponent_contact_count(data, robot)
        for robot in ROBOTS:
            i = _STATE_DIM + 2 * CONTACT_DIM + (0 if robot == "a" else BACK_DIM)
            f = back_features(model, data, robot, self.det_cfg)
            out[i] = f.tilt_deg / 90.0
            out[i + 1] = f.pelvis_z
            out[i + 2] = 1.0 if f.dorsal_contact else 0.0
            out[i + 3] = back_exposure(f.tilt_deg, f.pelvis_z, self.det_cfg)
        return out

    def describe(self) -> dict:
        """Human-readable layout + resolved category sizes (report/debug)."""
        return {
            "priv_dim": PRIV_DIM,
            "layout": [(n, s, e) for n, s, e in PRIV_LAYOUT],
            "contact_kinds": CONTACT_KINDS,
            "geom_counts": {
                r: {k: len(v) for k, v in self.categorizer._cats[r].items()}
                for r in ROBOTS
            },
        }


class CriticObsBuilder:
    """Concatenation ``critic_obs = [actor_obs | privileged]``.

    The first ``actor_obs_dim`` entries are exactly the actor observation --
    pinned by ``test_critic_obs_is_actor_plus_privileged``.
    """

    def __init__(self, actor_dim: int, priv: PrivilegedObsBuilder | None = None):
        self.actor_dim = int(actor_dim)
        self.priv = priv
        self.dim = self.actor_dim + (PRIV_DIM if priv is None else priv.dim)

    def build(self, actor_obs: np.ndarray, privileged: np.ndarray) -> np.ndarray:
        a = np.asarray(actor_obs, dtype=np.float32).reshape(self.actor_dim)
        p = np.asarray(privileged, dtype=np.float32).reshape(self.dim - self.actor_dim)
        return np.concatenate([a, p]).astype(np.float32)

    def layout(self):
        return (("actor_obs", (0, self.actor_dim)),
                ("privileged", (self.actor_dim, self.dim)))


if __name__ == "__main__":  # self-check
    from wrestling.env import QPOS_SLICE, QVEL_SLICE, WrestlingEnv, load_wrestling_model

    env = WrestlingEnv(seed=3, match_clock=2.0, exchange_timeout=2.0)
    env.reset(seed=3)
    env.step()
    model = env.model
    b = PrivilegedObsBuilder(model)
    p = b.build(model, env.data)
    assert p.shape == (PRIV_DIM,) and p.dtype == np.float32
    # state block must mirror the live simulator state exactly
    assert np.allclose(p[0:36], env.data.qpos[QPOS_SLICE["a"]])
    assert np.allclose(p[36:72], env.data.qpos[QPOS_SLICE["b"]])
    assert np.allclose(p[72:107], env.data.qvel[QVEL_SLICE["a"]])
    assert np.allclose(p[107:142], env.data.qvel[QVEL_SLICE["b"]])
    d = b.describe()
    assert all(v > 0 for v in d["geom_counts"]["a"].values()), d["geom_counts"]
    cb = CriticObsBuilder(92, b)
    c = cb.build(np.zeros(92, np.float32), p)
    assert c.shape == (92 + PRIV_DIM,)
    print("rl.privileged self-check OK:", {"priv_dim": PRIV_DIM,
                                           "critic_dim": int(c.size),
                                           "geom_counts_a": d["geom_counts"]["a"]})
