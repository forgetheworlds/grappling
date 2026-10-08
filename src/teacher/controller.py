"""Stabilized teacher controller (phase 3).

Executes a phase-2 reference on the 29 position actuators per robot while
adding balance feedback on top of the reference joint targets — closing the
documented phase-2 gap (pure joint-PD topples: pelvis zErr 0.29-0.83 m).

Architecture (per robot, 50 Hz, one weighted least-squares solve):

  ctrl = clip(q_ref(t) + dq, ctrlrange)

where dq solves a stacked first-order task Jacobian built from the *simulated*
state (MuJoCo's own kinematics — no hand-coded axes/signs):

  1. CoM XY over the support center  (ankle/hip strategy + capture-point
     lookahead: e = com + v*tau - support, candidate 1+2)
  2. planted-foot world anchoring at the reference foot positions
     (differential 6-DoF leg IK: per-tick corrections are <5 cm, so the
     first-order solve IS the small analytic IK; iterating it to convergence
     would only matter for large corrections, which the clamps forbid)
  3. torso uprightness: rotate the torso up-vector toward a blend of the
     reference up-vector and world +z (follows intentional lean)
  4. pelvis height P+leaky-I vs the reference height (squat support)

against a regularizer that keeps dq small and keeps the arms tracking.
Gains are scheduled per phase (phases.py) — the stabilizer is fully off in
PRONE phases (takedown finishes, sprawled defender, stand-up start) so
intended ground postures are not fought.

Everything is closed-form and stateless except two leaky integrators + the
previous offset (rate limiting): identical inputs give identical outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .gains import GainSet, gain_for
from .phases import label_frames

DT_CTRL = 0.02          # 50 Hz control
FOOT_Z_TOUCH = 0.045     # site height counting as touching the mat

#: task weights in the stacked LS (com / anchor site / upright / height)
W_COM = 3.0
W_ANCHOR = 2.0
W_UP = 0.8
W_Z = 1.0
#: regularization: lambda per joint group (legs keep balance authority,
#: the upper body stays close to the reference)
REG_LAMBDA = 0.05
_REG_GROUP = np.array([1.0] * 12 + [3.0] * 3 + [6.0] * 14)


@dataclass
class TeacherFlags:
    """Ablation switches — each maps to one stabilizer candidate."""
    use_com: bool = True        # 1: ankle/hip CoM feedback
    use_cp: bool = True         # 2: capture-point velocity damping
    use_anchor: bool = True     # 3: planted-foot anchoring (leg IK)
    use_up: bool = True         # torso uprightness
    use_z: bool = True          # 4: pelvis-height P+I
    use_phases: bool = True     # gain scheduling by reference phase


ALL_OFF = TeacherFlags(use_com=False, use_cp=False, use_anchor=False,
                       use_up=False, use_z=False, use_phases=False)


class _RobotCtx:
    """Per-robot precomputed ids, reference tables and integrator state."""

    def __init__(self, model: mujoco.MjModel, prefix: str, qpos_ref: np.ndarray,
                 t_ref: np.ndarray):
        self.prefix = prefix
        n = mujoco.mj_name2id
        obj = mujoco.mjtObj
        self.site = {s: n(model, obj.mjOBJ_SITE, f"{prefix}{s}")
                     for s in ("core", "left_toe", "left_heel", "right_toe",
                               "right_heel")}
        self.foot_sites = (("left_toe", "left_heel"), ("right_toe", "right_heel"))
        self.pelvis = n(model, obj.mjOBJ_BODY, f"{prefix}pelvis")
        self.torso = n(model, obj.mjOBJ_BODY, f"{prefix}torso_link")
        jid0 = n(model, obj.mjOBJ_JOINT, f"{prefix}left_hip_pitch_joint")
        self.dofs = slice(model.jnt_dofadr[jid0],
                          model.jnt_dofadr[jid0] + 29)
        self.qpos0 = model.jnt_qposadr[jid0]
        self.labels = label_frames(t_ref, qpos_ref[:, 2])
        self._precompute_ref(model, qpos_ref)

    def _precompute_ref(self, model, qpos_ref):
        d = mujoco.MjData(model)
        T = len(qpos_ref)
        self.ref_pelvis_z = qpos_ref[:, 2].copy()
        self.ref_qpos = qpos_ref.copy()
        self.ref_feet = np.empty((T, 2, 2, 3))       # foot x (toe, heel) xyz
        self.ref_up = np.empty((T, 3))
        self.ref_support = np.empty((T, 2))
        self.ref_touch = np.empty((T, 2, 2), dtype=bool)  # site touching mat
        for k in range(T):
            d.qpos[:] = 0.0
            sl = slice(0, 36) if self.prefix == "a_" else slice(36, 72)
            d.qpos[sl] = qpos_ref[k]
            mujoco.mj_kinematics(model, d)
            for f, (toe, heel) in enumerate(self.foot_sites):
                self.ref_feet[k, f, 0] = d.site_xpos[self.site[toe]]
                self.ref_feet[k, f, 1] = d.site_xpos[self.site[heel]]
            self.ref_up[k] = d.xmat[self.torso].reshape(3, 3) @ (0, 0, 1)
            pts = []
            for f in range(2):
                for s in range(2):
                    touch = self.ref_feet[k, f, s, 2] < FOOT_Z_TOUCH
                    self.ref_touch[k, f, s] = touch
                    if touch:
                        pts.append(self.ref_feet[k, f, s, :2])
            # ball-of-foot references keep only the toe down; the support is
            # the touching sites (falls back to the mid-foot line)
            self.ref_support[k] = (np.mean(pts, axis=0) if pts else
                                   self.ref_feet[k].mean(axis=(0, 1))[:2])


class TeacherController:
    """Stabilized joint-target teacher for both robots."""

    def __init__(self, model: mujoco.MjModel, qpos_a: np.ndarray,
                 qpos_b: np.ndarray, t_ref: np.ndarray, technique: str = "",
                 flags: TeacherFlags | None = None):
        self.model = model
        self.t_ref = np.asarray(t_ref, float)
        self.technique = technique
        self.flags = flags or TeacherFlags()
        self.ctx = {"a_": _RobotCtx(model, "a_", qpos_a, self.t_ref),
                    "b_": _RobotCtx(model, "b_", qpos_b, self.t_ref)}
        self.ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = model.actuator_ctrlrange[:, 1].copy()
        # per-robot controller state
        self._dq = {p: np.zeros(29) for p in self.ctx}
        self._int_z = {p: 0.0 for p in self.ctx}
        self._prev_com = {p: None for p in self.ctx}
        self._t_prev = 0.0
        self._jac_buf = np.zeros((3, model.nv))
        self._jacr_buf = np.zeros((3, model.nv))

    # -- reference interpolation ------------------------------------------------

    def _frame(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.t_ref, t), 0,
                           len(self.t_ref) - 1))

    def _q_ref(self, ctx: _RobotCtx, t: float) -> np.ndarray:
        i = int(np.clip(np.searchsorted(self.t_ref, t), 1, len(self.t_ref) - 1))
        w = (t - self.t_ref[i - 1]) / (self.t_ref[i] - self.t_ref[i - 1])
        w = float(np.clip(w, 0.0, 1.0))
        return (ctx.ref_qpos[i - 1, 7:36] * (1 - w)
                + ctx.ref_qpos[i, 7:36] * w)

    # -- one control tick --------------------------------------------------

    def control(self, data: mujoco.MjData, t: float) -> np.ndarray:
        """Position targets for all 58 actuators at simulation time t."""
        dt = DT_CTRL
        ctrl = np.empty(58)
        for k, (prefix, ctx) in enumerate(self.ctx.items()):
            ctrl[29 * k:29 * (k + 1)] = self._control_robot(
                data, prefix, ctx, t, dt)
        self._t_prev = t
        return ctrl

    def _control_robot(self, data, prefix, ctx: _RobotCtx, t: float,
                       dt: float) -> np.ndarray:
        model = self.model
        frame = self._frame(t)
        kind = str(ctx.labels[frame]) if self.flags.use_phases else "STAND"
        g: GainSet = gain_for(self.technique, kind)
        f = self.flags
        dofs = ctx.dofs

        rows: list[tuple[np.ndarray, np.ndarray, float]] = []
        com = data.subtree_com[ctx.pelvis]
        v_com = np.zeros(2) if self._prev_com[prefix] is None else (
            (com[:2] - self._prev_com[prefix]) / dt)
        self._prev_com[prefix] = com[:2].copy()

        # 1+2: CoM over support with capture-point lookahead
        if f.use_com and g.k_com > 0:
            support = self._support_xy(data, ctx)
            if f.use_cp and g.cp_tau > 0:
                h = max(com[2] - 0.05, 0.3)
                tau = min(g.cp_tau, 0.6 * np.sqrt(h / G))
                e = com[:2] + v_com * tau - support
            else:
                e = com[:2] - support
            des = -g.k_com * np.clip(e, -g.com_clamp, g.com_clamp)
            mujoco.mj_jacSubtreeCom(model, data, self._jac_buf, ctx.pelvis)
            rows.append((self._jac_buf[:2, dofs].copy(), des, W_COM))

        # 3: planted-foot anchoring (differential leg IK): fully anchor the
        # reference-touching sites (xyz); the raised partner site of a
        # touching foot keeps xy only so the sim foot may settle flatter and
        # widen the support behind the CoM.
        if f.use_anchor and g.k_anchor > 0:
            for foot in range(2):
                touch = ctx.ref_touch[frame, foot]
                if not touch.any():
                    continue
                for s in range(2):
                    sid = ctx.site[ctx.foot_sites[foot][s]]
                    tgt = ctx.ref_feet[frame, foot, s]
                    err = np.clip(tgt - data.site_xpos[sid],
                                  -g.anchor_clamp, g.anchor_clamp)
                    mujoco.mj_jacSite(model, data, self._jac_buf, None, sid)
                    if touch[s]:
                        rows.append((self._jac_buf[:, dofs].copy(),
                                     g.k_anchor * err, W_ANCHOR))
                    else:
                        rows.append((self._jac_buf[:2, dofs].copy(),
                                     g.k_anchor * err[:2], 0.5 * W_ANCHOR))

        # torso uprightness (follow reference lean, partially upright)
        if f.use_up and g.k_up > 0:
            up_sim = data.xmat[ctx.torso].reshape(3, 3) @ (0, 0, 1)
            blend = g.upright_blend
            up_tgt = (1 - blend) * ctx.ref_up[frame] + blend * np.array(
                [0.0, 0.0, 1.0])
            n = np.linalg.norm(up_tgt)
            if n > 1e-6:
                up_tgt /= n
                err = np.cross(up_sim, up_tgt)     # rotation vector sim->tgt
                mujoco.mj_jacBodyCom(model, data, self._jac_buf,
                                     self._jacr_buf, ctx.torso)
                rows.append((self._jacr_buf[:2, dofs].copy(),
                             g.k_up * err[:2], W_UP))

        # 4: pelvis height P + leaky I vs reference
        if f.use_z and (g.k_z > 0 or g.ki_z > 0):
            z_ref, z_sim = ctx.ref_pelvis_z[frame], float(
                data.site_xpos[ctx.site["core"]][2])
            self._int_z[prefix] = float(np.clip(
                self._int_z[prefix] + (z_ref - z_sim) * dt,
                -g.ki_int_clamp, g.ki_int_clamp))
            des = float(np.clip(g.k_z * (z_ref - z_sim)
                                + g.ki_z * self._int_z[prefix], -0.03, 0.03))
            mujoco.mj_jacSite(model, data, self._jac_buf, None, ctx.site["core"])
            rows.append((self._jac_buf[2, dofs].reshape(1, -1).copy(),
                         np.array([des]), W_Z))
        if kind == "PRONE":
            self._int_z[prefix] = 0.0

        # weighted regularized least squares for the 29 joint offsets
        if rows:
            A = np.vstack([w * J for J, _, w in rows])
            b = np.concatenate([w * d for _, d, w in rows])
            A = np.vstack([A, np.sqrt(REG_LAMBDA * _REG_GROUP)[None, :]])
            b = np.concatenate([b, np.zeros(1)])
            dq, *_ = np.linalg.lstsq(A, b, rcond=None)
        else:
            dq = np.zeros(29)

        # clamps: magnitude by group, then rate
        lim = np.full(29, g.max_upp_offset)
        lim[:12] = g.max_leg_offset
        dq = np.clip(dq, -lim, lim)
        dq = self._dq[prefix] + np.clip(dq - self._dq[prefix],
                                        -g.max_rate, g.max_rate)
        self._dq[prefix] = dq

        q_ref = self._q_ref(ctx, t)
        lo = self.ctrl_lo[0:29] if prefix == "a_" else self.ctrl_lo[29:58]
        hi = self.ctrl_hi[0:29] if prefix == "a_" else self.ctrl_hi[29:58]
        return np.clip(q_ref + dq, lo, hi)

    def _support_xy(self, data, ctx: _RobotCtx) -> np.ndarray:
        """Support center from the sites currently touching the mat."""
        pts = []
        for foot in range(2):
            for s in ctx.foot_sites[foot]:
                p = data.site_xpos[ctx.site[s]]
                if p[2] < FOOT_Z_TOUCH:
                    pts.append(p[:2])
        if pts:
            return np.mean(pts, axis=0)
        return ctx.ref_support[self._frame(data.time)]


class BaselinePD:
    """Pure reference tracking (phase-2 controller) for ablation baselines."""

    def __init__(self, model, qpos_a, qpos_b, t_ref, **_):
        self.t_ref = np.asarray(t_ref, float)
        self.q = [np.asarray(qpos_a)[:, 7:36], np.asarray(qpos_b)[:, 7:36]]
        self.ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = model.actuator_ctrlrange[:, 1].copy()

    def control(self, data, t: float) -> np.ndarray:
        i = int(np.clip(np.searchsorted(self.t_ref, t), 1, len(self.t_ref) - 1))
        w = float(np.clip((t - self.t_ref[i - 1])
                          / (self.t_ref[i] - self.t_ref[i - 1]), 0.0, 1.0))
        ctrl = np.empty(58)
        for k in range(2):
            seg = slice(29 * k, 29 * (k + 1))
            q = self.q[k][i - 1] * (1 - w) + self.q[k][i] * w
            ctrl[seg] = np.clip(q, self.ctrl_lo[seg], self.ctrl_hi[seg])
        return ctrl
