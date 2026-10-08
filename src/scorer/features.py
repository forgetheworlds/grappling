"""Relational-geometry feature extraction for the technique-validity scorer.

Given the 36-dof qpos of the commanded robot ("self") and of its opponent
("opp"), plus the two-G1 wrestling scene model, compute a fixed dictionary of
scalar RELATIONAL features (distances, angles, height relationships) that the
technique predicates are defined over. Features are deliberately expressed in
the SELF yaw frame (forward = pelvis heading) and as relative quantities, so a
technique scores the same when the pair is translated/rotated anywhere in the
arena, or when the robots drift — the scorer judges RELATIONSHIPS, not
Cartesian replay (MISSION.md "Preserve wrestling geometry").

Kinematics: one ``mj_kinematics`` call per featurize() (~8 us on this ARM
core for the 61-body scene model). No dynamics, no contacts, no rendering.
Self is placed in the scene's ``a_`` slot and opp in ``b_``; since every
feature is relative this is only a kinematic container.

Site names follow src/retarget/scene.py (prefix a_/b_, 19 retargeting
landmarks per robot; contract in notes.md "Interface contracts").
"""

from __future__ import annotations

import numpy as np
import mujoco

#: scalar feature names, in canonical order (FEAT_ORDER[i] = value i)
FEAT_ORDER: tuple[str, ...] = (
    # pair separation / approach
    "sep",            # ||core_S - core_O|| (m)
    "dx",             # forward distance core_O - core_S in self yaw frame (m)
    "dz",             # core_O z - core_S z (m)
    "bearing",        # bearing of opp from self in self yaw frame (rad)
    "rel_yaw",        # wrap(yaw_O - yaw_S): 0 = same heading, pi = facing (rad)
    # torso orientation
    "pitch_s",        # self torso forward-lean from core->neck (rad, + = lean fwd)
    "roll_s",         # self torso side-lean (rad)
    "pitch_o",
    "roll_o",
    # head relationships
    "head_rel",       # self head z - opp core z (m); head-on-chest ~ small +
    "head_chest",     # ||head_S - chest_O||, chest = mid(neck, core) (m)
    "chest_on",       # ||neck_S - neck_O||: chest-to-chest pressure (m)
    # level change / legs
    "knee_z_s",       # self MIN knee height (m)
    "knee_z_o",       # opp min knee height (m)
    "ankle_z_s",      # self mean ankle height (m)
    "ankle_z_o",
    "base_w_s",       # self stance width: horiz ||ankle_L - ankle_R|| (m)
    "base_w_o",
    "leg_back_s",     # self mean ankle x rel own core in OWN yaw frame (m, - = trailing)
    # arm lock points (opponent legs/torso/head)
    "lock_l",         # min over self wrists of dist to opp LEFT leg (knee or mid-thigh)
    "lock_r",         # same, opp RIGHT leg
    "lock_asym",      # |lock_l - lock_r| (m)
    "wrap",           # min over self wrists of dist to opp chest (body lock)
    "head_snap",      # min over self wrists of dist to opp head/neck midpoint
)
_SITES = ("core", "neck", "head", "left_hip", "right_hip", "left_knee",
          "right_knee", "left_ankle", "right_ankle", "left_shoulder",
          "right_shoulder", "left_wrist", "right_wrist")
_N = len(FEAT_ORDER)
_IDX = {n: i for i, n in enumerate(FEAT_ORDER)}

_SITES_S = ("core", "neck", "head", "left_hip", "right_hip", "left_knee",
            "right_knee", "left_ankle", "right_ankle", "left_shoulder",
            "right_shoulder")
_SITES_O = _SITES_S

_QW, _QX, _QY, _QZ = 3, 4, 5, 6


def _yaw_from_quat(q: np.ndarray) -> float:
    """Yaw (rad) of a (w,x,y,z) quaternion (Z-up world, body->world)."""
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y),
                            1.0 - 2.0 * (y * y + z * z)))


def _wrap(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


class Featurizer:
    """Extracts FEAT_ORDER relational features from a pair of G1 qpos.

    One instance owns its own ``MjData`` bound to the passed scene model and
    is NOT thread-safe (single MjData reuse). Kinematics only — no constraint
    solve — so calling from inside a training loop never perturbs a
    concurrently-stepped MjData of the caller.
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self._data = mujoco.MjData(model)
        # site id block: [core, neck, head, hip_l, hip_r, knee_l, knee_r,
        #                 ank_l, ank_r, sho_l, sho_r, wrist_l, wrist_r]
        self._sid_s = [model.site("a_" + s).id for s in _SITES]
        self._sid_o = [model.site("b_" + s).id for s in _SITES]
        self._a = slice(*_robot_qpos_range(model, "a_"))
        self._b = slice(*_robot_qpos_range(model, "b_"))

    def features(self, self_qpos: np.ndarray, opp_qpos: np.ndarray) -> np.ndarray:
        """Return the (len(FEAT_ORDER),) float64 feature vector.

        ``self_qpos``/``opp_qpos`` are (36,) G1 qpos vectors (base 7 + 29
        hinges, layout per notes.md); quaternions need not be normalized.
        """
        d = self._data
        q = d.qpos
        q[self._a] = self_qpos
        q[self._b] = opp_qpos
        # normalize both base quats (perturbed inputs may drift)
        for sl in (self._a, self._b):
            qua = q[sl][_QW:_QZ + 1]
            n = np.sqrt(np.dot(qua, qua))
            if n > 0.0:
                q[sl][_QW:_QZ + 1] = qua / n
        mujoco.mj_kinematics(self.model, d)
        xs = d.site_xpos[self._sid_s]     # (13,3) self landmark positions
        xo = d.site_xpos[self._sid_o]     # (13,3) opp landmark positions

        yaw_s = _yaw_from_quat(q[self._a][_QW:_QZ + 1])
        yaw_o = _yaw_from_quat(q[self._b][_QW:_QZ + 1])
        c, s = np.cos(yaw_s), np.sin(yaw_s)
        fwd = np.array([c, s])
        side = np.array([-s, c])

        def flat(v3: np.ndarray) -> np.ndarray:
            """(3,) world vector -> (2,) [forward, lateral] in self yaw frame."""
            return np.array([v3[:2] @ fwd, v3[:2] @ side])

        v = np.empty(_N)
        i = _IDX

        # --- separation / approach ---
        rel = xo[0] - xs[0]                       # core_O - core_S world
        v[i["sep"]] = np.linalg.norm(rel)
        f_rel = flat(rel)
        v[i["dx"]] = f_rel[0]
        v[i["dz"]] = rel[2]
        v[i["bearing"]] = np.arctan2(f_rel[1], f_rel[0])
        v[i["rel_yaw"]] = _wrap(yaw_o - yaw_s)

        # --- torso orientation: core->neck lean, per robot in own yaw frame ---
        for xs_, yaw_, tag in ((xs, yaw_s, "s"), (xo, yaw_o, "o")):
            up = xs_[1] - xs_[0]                  # pelvis -> neck
            cc, ss = np.cos(yaw_), np.sin(yaw_)
            fx = up[0] * cc + up[1] * ss          # forward component
            fy = -up[0] * ss + up[1] * cc         # lateral component
            v[i[f"pitch_{tag}"]] = np.arctan2(fx, up[2])
            v[i[f"roll_{tag}"]] = np.arctan2(fy, up[2])

        # --- head relationships ---
        v[i["head_rel"]] = xs[2, 2] - xo[0, 2]
        chest_o = 0.5 * (xo[1] + xo[0])
        v[i["head_chest"]] = np.linalg.norm(xs[2] - chest_o)
        v[i["chest_on"]] = np.linalg.norm(xs[1] - xo[1])

        # --- level change / legs ---
        v[i["knee_z_s"]] = min(xs[5, 2], xs[6, 2])
        v[i["knee_z_o"]] = min(xo[5, 2], xo[6, 2])
        v[i["ankle_z_s"]] = 0.5 * (xs[7, 2] + xs[8, 2])
        v[i["ankle_z_o"]] = 0.5 * (xo[7, 2] + xo[8, 2])
        v[i["base_w_s"]] = np.linalg.norm(flat(xs[7] - xs[8]))
        v[i["base_w_o"]] = np.linalg.norm(flat(xo[7] - xo[8]))
        # self ankles relative to own core in own yaw frame
        cc, ss = np.cos(yaw_s), np.sin(yaw_s)
        trail = 0.0
        for k in (7, 8):
            w = xs[k] - xs[0]
            trail += w[0] * cc + w[1] * ss
        v[i["leg_back_s"]] = 0.5 * trail

        # --- arm lock points ---
        wrists = (xs[11], xs[12])

        def _lock(side_k: int, hip_k: int) -> float:
            """Min wrist distance to one opp leg: knee or mid-thigh."""
            knee = xo[side_k]
            thigh = 0.5 * (xo[hip_k] + xo[side_k])
            return min(min(np.linalg.norm(w - knee), np.linalg.norm(w - thigh))
                       for w in wrists)

        v[i["lock_l"]] = _lock(5, 3)
        v[i["lock_r"]] = _lock(6, 4)
        v[i["lock_asym"]] = abs(v[i["lock_l"]] - v[i["lock_r"]])
        v[i["wrap"]] = min(np.linalg.norm(w - chest_o) for w in wrists)
        head_o = 0.5 * (xo[2] + xo[1])
        v[i["head_snap"]] = min(np.linalg.norm(w - head_o) for w in wrists)
        return v


def _robot_qpos_range(model: mujoco.MjModel, prefix: str) -> tuple[int, int]:
    """(start, end) qpos indices of one robot from its free-joint address."""
    jid = model.joint(prefix + "floating_base_joint").id
    start = model.jnt_qposadr[jid]
    return int(start), int(start + 7 + 29)


def feature_dict(vec: np.ndarray) -> dict[str, float]:
    """Feature vector -> named dict (debugging / reports)."""
    return {n: float(x) for n, x in zip(FEAT_ORDER, vec)}
