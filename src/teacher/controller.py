"""Stabilized teacher controller (phase 3).

Executes a phase-2 reference on the 29 position actuators per robot while
adding balance feedback on top of the reference joint targets — closing the
documented phase-2 gap (pure joint-PD topples: pelvis zErr 0.29-0.83 m).

Per robot, per 50 Hz tick::

    ctrl = clip(q_eff(t) + dq_direct + dq_ls, ctrlrange)
    q_eff(t) = q_ref(t) + trim_scale(z_ref) * trim        (candidate 5)

``trim`` (trims.py) is a small, measured, per-technique/per-robot reference
pose adjustment that makes the schematic posture statically holdable — the
STANCE crouch otherwise needs 36 Nm of ankle torque (of +-50) to hold and
topples in <1.5 s under any feedback (measured; see the report).  It is a
reference edit, not feedback, and it is scaled to zero for ground work
(reference pelvis < 0.42 m).

``dq_direct`` is the measured-authority balance feedback (stabilizers.AUTH):
a symmetric ankle/knee/hip correction that cancels the capture-point error of
the robot's CoM over its own support, resolved in the robot's heading frame,
plus a slow task integrator that lets the controller *hold* a nonzero
sustained offset when a posture still needs one.

``dq_ls`` is a regularized least-squares differential leg IK on the foot
sites, used for the channels that are not expressible as direct joint
offsets: planted-foot world anchoring, sole flattening onto the mat, and the
pelvis-height (squat) task; plus a soft torso-uprightness row (waist channel).

Design constraints that shaped this (all measured, see stabilizers.py and the
report):
  * the base is free-floating, so any world-frame task on a body that lives on
    the base (CoM, pelvis site) has zero Jacobian authority through joints —
    balance must go through the feet (ankle strategy) or the swing chain;
  * one leg's offset alone is resisted by the other leg in double support, so
    the balance channels are applied symmetrically to the contacting legs;
  * the direct channels saturate (ankle +0.2 rad moves the base only 5.4 cm:
    the CoP reaches the foot edge), and a *foot-target* bias (the "push the
    foot backwards to push the body forwards" hip strategy) measured only
    ~0.05 m/m of authority through the servo+contact equilibrium — it was
    implemented, measured and removed (see the report);
  * the retargeted STANCE keeps its heels 6-7 cm up (site geometry) and the
    pose's CoM sits behind the effective support — flattening both sole sites
    onto the mat and the measured reference trim together give the stance a
    real support polygon.

Everything is deterministic: the only state is the height integrator, the
task integrator and the previous offset (rate limiting).  Identical inputs
give identical outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco

from . import stabilizers as st
from .gains import GainSet, gain_for
from .phases import label_frames
from .trims import trim_for, trim_scale, trim_vector

DT_CTRL = 0.02           # 50 Hz control
G = 9.81                 # m/s^2 (capture-point time constant)
FOOT_Z_TOUCH = 0.045     # site height counting as touching the mat
COM_VEL_LP = 0.35        # CoM velocity low-pass weight (per tick)


@dataclass
class TeacherFlags:
    """Ablation switches — each maps to one stabilizer candidate."""
    use_com: bool = True        # 1: ankle/hip CoM-over-support (capture point)
    use_cp: bool = True         # 2: capture-point velocity damping
    use_roll: bool = True       # lateral half of candidate 1
    use_anchor: bool = True     # 3: planted-foot world anchoring (leg IK)
    use_flat: bool = True       # 3b: flatten planted soles onto the mat
    use_up: bool = True         # torso uprightness (waist channel)
    use_z: bool = True          # 4: pelvis-height P+I
    use_int: bool = True        # task integrator (sustained offset)
    use_trim: bool = True       # 5: measured reference-pose trim
    use_govern: bool = True     # 5b: posture governor toward the stable stand
    use_phases: bool = True     # gain scheduling by reference phase


ALL_OFF = TeacherFlags(use_com=False, use_cp=False, use_roll=False,
                       use_anchor=False, use_flat=False, use_up=False,
                       use_z=False, use_int=False, use_trim=False,
                       use_govern=False, use_phases=False)
NO_GOVERN = TeacherFlags(use_govern=False)


class _RobotCtx:
    """Per-robot precomputed ids, reference tables and integrator state.

    ``qpos_ref`` is stored *with the trim already added* to the 29 joints: the
    context's reference tables (foot sites, torso up, support, pelvis height)
    are therefore the tables of the effective reference the controller tracks.
    """

    def __init__(self, model: mujoco.MjModel, prefix: str, qpos_ref: np.ndarray,
                 t_ref: np.ndarray, trim: np.ndarray | None = None,
                 dz_trim: float = 0.0):
        self.prefix = prefix
        n = mujoco.mj_name2id
        obj = mujoco.mjtObj
        self.site = {s: n(model, obj.mjOBJ_SITE, f"{prefix}{s}")
                     for s in ("core", "left_toe", "left_heel", "right_toe",
                               "right_heel")}
        self.foot_sites = (("left_toe", "left_heel"), ("right_toe", "right_heel"))
        self.pelvis = n(model, obj.mjOBJ_BODY, f"{prefix}pelvis")
        self.torso = n(model, obj.mjOBJ_BODY, f"{prefix}torso_link")
        self.base_qpos = int(model.jnt_qposadr[n(
            model, obj.mjOBJ_JOINT, f"{prefix}floating_base_joint")])
        self.qpos_slice = slice(self.base_qpos, self.base_qpos + 36)
        self.base_dof = int(model.jnt_dofadr[n(
            model, obj.mjOBJ_JOINT, f"{prefix}floating_base_joint")])
        #: explicit support foot (0/1) during a weight transfer; None = both
        self.support_foot: int | None = None
        #: True while a stepping primitive owns the support (governor gated off)
        self.step_mode: bool = False
        jid0 = n(model, obj.mjOBJ_JOINT, f"{prefix}left_hip_pitch_joint")
        self.dofs = np.arange(model.jnt_dofadr[jid0], model.jnt_dofadr[jid0] + 29)
        qpos_ref = np.array(qpos_ref, float)
        if trim is not None:
            qpos_ref[:, 7:36] = qpos_ref[:, 7:36] + trim
        self.trim = np.zeros(29) if trim is None else np.asarray(trim, float)
        self.trim_dz = float(dz_trim)
        self.t_ref = np.asarray(t_ref, float)
        self.labels = label_frames(self.t_ref, qpos_ref[:, 2])
        self.touch_z = FOOT_Z_TOUCH
        self._precompute_ref(model, qpos_ref)

    def _precompute_ref(self, model, qpos_ref):
        T = len(qpos_ref)
        self.ref_qpos = np.asarray(qpos_ref, float).copy()
        self.ref_pelvis_z = np.empty(T)
        #: reference base heading (yaw, rad) per frame
        self.ref_yaw = np.empty(T)
        self.ref_scale = np.empty(T)
        self.ref_feet = np.empty((T, 2, 2, 3))       # foot x (toe, heel) xyz
        self.ref_up = np.empty((T, 3))
        self.ref_support = np.empty((T, 2))
        self.ref_touch = np.empty((T, 2, 2), dtype=bool)  # site touching mat
        self._fk = mujoco.MjData(model)              # row-wise FK scratch
        self.set_rows(model, self.ref_qpos, np.arange(T))
        # rear foot (0=left, 1=right): the foot whose mean forward coordinate
        # is smallest, i.e. the one behind the line of action — the leg the
        # operator-preferred stance repair extends backwards
        yaw0 = float(np.arctan2(np.sin(self.ref_yaw).mean(),
                                np.cos(self.ref_yaw).mean()))
        axis = np.array([np.cos(yaw0), np.sin(yaw0)])
        self.rear_foot = int(np.argmin(
            [self.ref_feet[:, f, :, :2].mean(axis=(0, 1)) @ axis
             for f in range(2)]))

    def set_rows(self, model, qpos_rows: np.ndarray, rows: np.ndarray,
                 feet: np.ndarray | None = None,
                 touch: np.ndarray | None = None) -> None:
        """(Re)compute every reference-derived table row for frames ``rows``.

        Shared by construction and the live update.  ``qpos_rows`` are the
        full (n, 36) reference rows of those frames — already including the
        trim, i.e. the effective reference the controller tracks.  Refreshes
        the joint table, pelvis height, base heading, trim scale, foot-site
        positions, torso up-vector, touching sites and the support centre,
        and keeps the phase labels in sync with the pelvis-height track.
        Nothing else is touched, so all feedback state survives the update.

        ``feet`` (n, 2, 2, 3) and ``touch`` (n, 2, 2) override the FK-derived
        world foot-site targets and touching flags: this is how a stepping
        scheduler *pins* a planted foot in the world (so the leg IK holds it
        while the body moves) or declares a swing foot airborne.
        """
        rows = np.asarray(rows, dtype=np.intp)
        q = np.asarray(qpos_rows, float)
        self.ref_qpos[rows] = q
        self.ref_pelvis_z[rows] = q[:, 2] + self.trim_dz
        qw, qx, qy, qz = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
        self.ref_yaw[rows] = np.arctan2(2.0 * (qw * qz + qx * qy),
                                        1.0 - 2.0 * (qy * qy + qz * qz))
        self.ref_scale[rows] = [trim_scale(float(z)) for z in q[:, 2]]
        d = self._fk
        sl = self.qpos_slice
        feet_arr = None if feet is None else np.asarray(feet, float)
        touch_arr = None if touch is None else np.asarray(touch, bool)
        for j, (k, row) in enumerate(zip(rows, q)):
            d.qpos[:] = 0.0                           # only this robot's FK
            d.qpos[sl] = row
            mujoco.mj_kinematics(model, d)
            if feet_arr is not None:
                self.ref_feet[k] = feet_arr[j]
            else:
                for f, (toe, heel) in enumerate(self.foot_sites):
                    self.ref_feet[k, f, 0] = d.site_xpos[self.site[toe]]
                    self.ref_feet[k, f, 1] = d.site_xpos[self.site[heel]]
            self.ref_up[k] = d.xmat[self.torso].reshape(3, 3) @ (0, 0, 1)
            pts = []
            for f in range(2):
                for s in range(2):
                    touch_f = (touch_arr[j, f, s] if touch_arr is not None else
                               self.ref_feet[k, f, s, 2] < FOOT_Z_TOUCH)
                    self.ref_touch[k, f, s] = bool(touch_f)
                    if touch_f:
                        pts.append(self.ref_feet[k, f, s, :2])
            # ball-of-foot references keep only the toe down; the support is
            # the touching sites (falls back to the mid-foot line)
            self.ref_support[k] = (np.mean(pts, axis=0) if pts else
                                   self.ref_feet[k].mean(axis=(0, 1))[:2])
        self.labels = label_frames(self.t_ref, self.ref_pelvis_z)

    def reset_ref(self, model, qpos_rows: np.ndarray, t_ref: np.ndarray) -> None:
        """Re-initialise the reference tables to a fresh horizon (drill reset).

        A drill `reset()` must drop the previously *grown* tables, otherwise the
        time grid and the row arrays disagree in length (the labels' gradient
        then raises, and stale rows leak into the new run).
        """
        q = np.asarray(qpos_rows, float)
        T = len(q)
        self.t_ref = np.asarray(t_ref, float)
        assert len(self.t_ref) == T, (len(self.t_ref), T)
        self.ref_pelvis_z = np.empty(T)
        self.ref_yaw = np.empty(T)
        self.ref_scale = np.empty(T)
        self.ref_feet = np.empty((T, 2, 2, 3))
        self.ref_up = np.empty((T, 3))
        self.ref_support = np.empty((T, 2))
        self.ref_touch = np.empty((T, 2, 2), dtype=bool)
        self.ref_qpos = np.empty((T, 36))
        self.set_rows(model, q, np.arange(T))

    def grow_ref(self, n_rows: int) -> None:
        """Extend the reference tables to ``n_rows`` frames (rolling timeline).

        Rows past the current horizon are filled with the last row; the drill
        overwrites them before they are read.  This is what removes the fixed
        table cap (the previous design clamped after 17.98 s and could not run
        a 60-90 s drill).
        """
        T = len(self.ref_qpos)
        if n_rows <= T:
            return
        add = n_rows - T
        pad = np.tile(self.ref_qpos[-1], (add, 1))
        self.ref_qpos = np.vstack([self.ref_qpos, pad])
        self.ref_pelvis_z = np.concatenate(
            [self.ref_pelvis_z, np.repeat(self.ref_pelvis_z[-1], add)])
        self.ref_yaw = np.concatenate(
            [self.ref_yaw, np.repeat(self.ref_yaw[-1], add)])
        self.ref_scale = np.concatenate(
            [self.ref_scale, np.repeat(self.ref_scale[-1], add)])
        self.ref_feet = np.concatenate(
            [self.ref_feet, np.tile(self.ref_feet[-1], (add, 1, 1, 1))])
        self.ref_up = np.concatenate(
            [self.ref_up, np.tile(self.ref_up[-1], (add, 1))])
        self.ref_support = np.concatenate(
            [self.ref_support, np.tile(self.ref_support[-1], (add, 1))])
        self.ref_touch = np.concatenate(
            [self.ref_touch, np.tile(self.ref_touch[-1], (add, 1, 1))])
        self.t_ref = np.concatenate(
            [self.t_ref, self.t_ref[-1] + DT_CTRL * np.arange(1, add + 1)])


def _rest_sole_z(model, prefix: str = "a_") -> list[list[float]]:
    """Sole site heights of one robot's own standing keyframe (m)."""
    d = mujoco.MjData(model)
    kid = -1
    for name in (f"{prefix}stand", "both_stand", "stand", "home"):
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
        if kid >= 0:
            break
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(model, d, kid)
    else:                                     # no keyframe: the robot's qpos
        q = _keyframe_qpos(model, prefix)
        d.qpos[:] = 0.0
        d.qpos[0:36] = q if prefix != "b_" else 0.0
    mujoco.mj_forward(model, d)
    foot_sites = (("left_toe", "left_heel"), ("right_toe", "right_heel"))
    return [[float(d.site_xpos[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}{s}")][2]) for s in foot]
        for foot in foot_sites]


def _keyframe_qpos(model, prefix: str) -> np.ndarray:
    """The model's own standing keyframe qpos for one robot (36,).

    Works in both scenes: the two-robot scene has ``both_stand`` (72), the
    single-G1 scene has ``stand`` (36).
    """
    kid = -1
    for name in (f"{prefix}stand", "both_stand", "stand", "home"):
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
        if kid >= 0:
            break
    q = np.array(model.key_qpos[kid if kid >= 0 else 0], float)
    if len(q) == 36:
        return q
    return q[0:36] if prefix != "b_" else q[36:72]


def _actuator_ids(model, prefix: str) -> np.ndarray:
    """Actuator ids of the 29 hinges of one robot, resolved by name."""
    jid0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                             f"{prefix}left_hip_pitch_joint")
    out = []
    for j in range(jid0, jid0 + 29):
        ids = np.where(model.actuator_trnid[:, 0] == j)[0]
        assert len(ids) == 1, f"actuator for joint {j}"
        out.append(int(ids[0]))
    return np.array(out, dtype=int)


class RobotTeacher:
    """Stabilised joint-target teacher for **one explicit robot context**.

    All model lookups are by name with the robot's prefix (``""`` in the
    single-G1 scene, ``a_``/``b_`` in the paired scene), the qpos/dof slices
    come from the model's own free joint, and the 29 actuator ids are
    resolved by name — no hard-coded 36/72 or 29/58 offsets.  ``control()``
    returns exactly that robot's 29 position targets and reads only its own
    state and its own reference.

    Feedback state (integrators, CoM filter, rate-limit memory, governor
    blend) lives here and survives reference updates.
    """

    def __init__(self, model: mujoco.MjModel, prefix: str, qpos_ref: np.ndarray,
                 t_ref: np.ndarray, technique: str = "",
                 flags: TeacherFlags | None = None, dbg: dict | None = None):
        self.model = model
        self.prefix = prefix
        self.flags = flags or TeacherFlags()
        self.technique = technique
        spec = trim_for(technique, prefix) if self.flags.use_trim else {}
        # two-pass: the rear foot is read off the reference, then the context
        # is rebuilt with the full (rear-aware) trim applied, so the
        # precomputed tables (feet, up-vector, support) match the effective
        # reference the controller tracks
        t_ref = np.asarray(t_ref, float)
        probe = _RobotCtx(model, prefix, qpos_ref, t_ref, trim=None,
                          dz_trim=float(spec.get("dz", 0.0)))
        trim = trim_vector(spec, rear=probe.rear_foot)
        self.ctx = _RobotCtx(model, prefix, qpos_ref, t_ref, trim=trim,
                             dz_trim=float(spec.get("dz", 0.0)))
        self.ctx.rest_z = _rest_sole_z(model, prefix)
        self.safe_q = _keyframe_qpos(model, prefix)[7:36].copy()
        self.act_ids = _actuator_ids(model, prefix)
        self.ctrl_lo = model.actuator_ctrlrange[self.act_ids, 0].copy()
        self.ctrl_hi = model.actuator_ctrlrange[self.act_ids, 1].copy()
        #: feedback state (preserved across reference updates; not reset)
        self._dq = np.zeros(29)
        self._int_z = 0.0
        self._int = np.zeros(2)
        self._com = None
        self._v = np.zeros(2)
        self._alpha = 0.0
        self._jac = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        #: last-tick introspection (kind, e, e_raw, e_int, dz, alpha) — read-out
        self.dbg: dict = dbg if dbg is not None else {}

    # -- reference ---------------------------------------------------------

    @property
    def t_ref(self) -> np.ndarray:
        """Reference time grid (owned by the context; grows with it)."""
        return self.ctx.t_ref

    @t_ref.setter
    def t_ref(self, value) -> None:
        self.ctx.t_ref = np.asarray(value, float)

    def _frame(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.t_ref, t), 0,
                           len(self.t_ref) - 1))

    def _q_ref(self, t: float) -> np.ndarray:
        ctx = self.ctx
        i = int(np.clip(np.searchsorted(self.t_ref, t), 1, len(self.t_ref) - 1))
        w = (t - self.t_ref[i - 1]) / (self.t_ref[i] - self.t_ref[i - 1])
        w = float(np.clip(w, 0.0, 1.0))
        q = (ctx.ref_qpos[i - 1, 7:36] * (1 - w)
             + ctx.ref_qpos[i, 7:36] * w)
        # the trim is scaled by the reference height (ground work: no trim)
        s = ctx.ref_scale[i - 1] * (1 - w) + ctx.ref_scale[i] * w
        return q - (1.0 - s) * ctx.trim

    def update_reference(self, rows: np.ndarray, qpos: np.ndarray,
                         feet: np.ndarray | None = None,
                         touch: np.ndarray | None = None) -> None:
        """Live reference update: refresh this robot's targets for ``rows``.

        Refreshes **every** reference-derived target the controller reads —
        joint table, foot-site targets, pelvis height, base heading, trim
        scale, torso up-vector, support centre, touching sites and the phase
        labels — from the new (trim-including) reference rows.  ``feet`` /
        ``touch`` may pin explicit world foot targets (stepping).  All
        feedback state is preserved: this edits the reference, not a reset.
        """
        self.ctx.set_rows(self.model, qpos, rows, feet=feet, touch=touch)

    def phase_kind(self, t: float) -> str:
        """Reference phase label at time ``t`` (for gain scheduling/reporting)."""
        return (str(self.ctx.labels[self._frame(t)]) if self.flags.use_phases
                else "STAND")

    # -- one control tick --------------------------------------------------

    def control(self, data: mujoco.MjData, t: float) -> np.ndarray:
        """The 29 position targets of this robot at simulation time ``t``."""
        f = self.flags
        model = self.model
        ctx = self.ctx
        frame = self._frame(t)
        kind = self.phase_kind(t)
        """PRONE interlock: the label comes from the *reference* (its intended
        ground work), so it must not switch the stabilizer off while the
        simulated robot is still standing — a lagging execution would be left
        uncontrolled.  Only a low simulated pelvis (< 0.55 m) confirms PRONE.
        """
        if kind == "PRONE" and float(data.qpos[ctx.base_qpos + 2]) > 0.55:
            kind = "LOW"
        g: GainSet = gain_for(self.technique, kind)

        # -- measurements (own state + own reference only) ------------------
        com = data.subtree_com[ctx.pelvis].copy()
        if self._com is None or not f.use_cp:
            v = np.zeros(2)
        else:
            raw = (com[:2] - self._com) / DT_CTRL
            v = (1 - COM_VEL_LP) * self._v + COM_VEL_LP * raw
        self._com = com[:2].copy()
        self._v = v
        _, foot_contact = st.foot_contact(data, ctx)
        support = st.support_center(data, ctx, frame)
        base = data.qpos[ctx.base_qpos:ctx.base_qpos + 3].copy()
        yaw = st.heading_yaw(data, ctx)

        # -- task error (world frame): CoM velocity-extrapolated over support
        e_raw = np.zeros(2)
        if f.use_com or f.use_roll:
            e = st.capture_error(com, v, support, g.kd, g.com_clamp)
            # the governor must activate on the *unclipped* error: the clipped
            # one saturates at sqrt(2)*com_clamp (~0.11 m) and could never
            # reach the emergency thresholds (measured defect)
            if f.use_govern:
                e_raw = st.capture_error(com, v, support, g.kd, None)
        else:
            e = np.zeros(2)

        # -- task integrator: hold a sustained offset ----------------------
        if f.use_int and kind != "PRONE" and g.ki > 0.0:
            self._int = np.clip(self._int + g.ki * DT_CTRL * e,
                                -g.int_max, g.int_max)
        elif kind == "PRONE":
            self._int *= 0.98
        e_int = self._int if f.use_int else np.zeros(2)
        e_dir = np.clip(e + e_int, -g.task_clamp, g.task_clamp)
        e_yaw = float(np.arctan2(np.sin(yaw - ctx.ref_yaw[frame]),
                                 np.cos(yaw - ctx.ref_yaw[frame])))

        # -- candidate 1+2: direct ankle/knee/hip balance channels ----------
        dq_direct = np.zeros(29)
        if f.use_com or f.use_roll:
            if not f.use_com:
                e_dir = np.array([0.0, e_dir[1]])
            if not f.use_roll:
                e_dir = np.array([e_dir[0], 0.0])
            dq_direct = st.channel_offsets(st.rot_z(-yaw) @ e_dir, foot_contact,
                                          {"k_pitch": g.k_pitch,
                                           "k_roll": g.k_roll})
            dq_direct += st.yaw_offsets(e_yaw, foot_contact, g.k_yaw, g.yaw_max)

        # -- candidate 3/3b/4 + upright: leg IK rows ------------------------
        z_err = float(base[2] - ctx.ref_pelvis_z[frame])
        self._int_z = float(np.clip(
            self._int_z + z_err * DT_CTRL, -0.08, 0.08)) if f.use_z else 0.0
        dz = 0.0
        if f.use_z and kind != "PRONE":
            dz = float(np.clip(g.k_z * z_err + g.ki_z * self._int_z,
                               -g.z_clamp, g.z_clamp))
        else:
            self._int_z = 0.0
        rows, des = [], []
        if f.use_anchor or f.use_flat or f.use_z:
            st.foot_rows(model, data, ctx, frame, dz, {
                "k_anchor": g.k_anchor if (f.use_anchor or f.use_flat) else 0.0,
                "anchor_clamp": g.anchor_clamp,
                "k_flat": g.k_flat if f.use_flat else 0.0}, self._jac, rows, des)
        if f.use_up:
            st.upright_rows(model, data, ctx, frame, {
                "k_up": g.k_up, "upright_blend": g.upright_blend},
                self._jac, self._jacr, rows, des)
        dq_ls = st.solve_offsets(rows, des, REG_LAMBDA)

        # -- candidate 5b: posture governor ---------------------------------
        # Emergency blend of the commanded posture toward the model's own
        # proven-stable standing keyframe, weighted by the *unclipped*
        # capture-point error and decaying slowly when the robot is safe
        # again.  It is the last resort for postures the trim cannot fix.
        #
        # Gated off while a stepping primitive owns the support: a *commanded*
        # weight transfer moves the CoM away from the mid-stance support by
        # design, which looks exactly like a large balance error; blending the
        # reference back to the two-feet-down stand keyframe mid-step fights
        # the transfer (measured: alpha 1.0 through the whole load phase and a
        # lateral topple).  The step's own guards/timeout own the abort.
        alpha_t = 0.0
        if f.use_govern and not getattr(ctx, "step_mode", False) \
                and g.govern_e1 > g.govern_e0:
            emag = float(np.hypot(*e_raw))
            alpha_t = float(np.clip((emag - g.govern_e0)
                                    / (g.govern_e1 - g.govern_e0), 0.0, 1.0))
        if kind == "PRONE":
            alpha_t = 0.0
        step = g.govern_up if alpha_t > self._alpha else g.govern_down
        self._alpha += float(np.clip(alpha_t - self._alpha, -step, step))
        self._alpha = float(np.clip(self._alpha, 0.0, 1.0))
        alpha = self._alpha

        # -- clamps, rate limit, servo limits ------------------------------
        dq = dq_direct + dq_ls
        lim = np.full(29, g.max_upp)
        lim[:12] = g.max_leg
        dq = np.clip(dq, -lim, lim)
        dq = self._dq + np.clip(dq - self._dq, -g.max_rate, g.max_rate)
        self._dq = dq
        q_track = self._q_ref(t)
        if alpha > 0.0:
            n = 15 if g.govern_legs_only else 29
            q_track[:n] = (1 - alpha) * q_track[:n] + alpha * self.safe_q[:n]
        self.dbg.update({
            "kind": kind, "e": e.copy(), "e_raw": np.asarray(e_raw).copy(),
            "e_int": np.asarray(e_int).copy(),
            "dz": dz, "z_err": z_err, "alpha": alpha, "e_yaw": e_yaw,
            "q_track": q_track,
        })
        return np.clip(q_track + dq, self.ctrl_lo, self.ctrl_hi)

    # -- introspection (tests, reporting) ---------------------------------

    def last_offsets(self) -> np.ndarray:
        """Joint offsets applied on the most recent tick."""
        return self._dq.copy()

    def reference_q(self, t: float) -> np.ndarray:
        """The reference joint row the executor tracks at time ``t`` (29,)."""
        return self._q_ref(t)


class TeacherController:
    """Two-robot composition of :class:`RobotTeacher` (acceptance path).

    ``control(data, t)`` returns the 58 position targets, written into each
    robot's own actuator ids (robot A first).  Kept for the phase-3 reference
    replay and acceptance runs; the solo milestone uses ``RobotTeacher``
    directly (or through ``teacher.skills.SkillController``).
    """

    def __init__(self, model: mujoco.MjModel, qpos_a: np.ndarray,
                 qpos_b: np.ndarray, t_ref: np.ndarray, technique: str = "",
                 flags: TeacherFlags | None = None):
        self.model = model
        self.t_ref = np.asarray(t_ref, float)
        self.technique = technique
        self.flags = flags or TeacherFlags()
        self._dbg: dict[str, dict] = {"a_": {}, "b_": {}}
        self.robots = {
            "a_": RobotTeacher(model, "a_", qpos_a, self.t_ref, technique,
                               self.flags, dbg=self._dbg["a_"]),
            "b_": RobotTeacher(model, "b_", qpos_b, self.t_ref, technique,
                               self.flags, dbg=self._dbg["b_"]),
        }
        self.ctx = {p: r.ctx for p, r in self.robots.items()}
        self.nu = model.nu

    def control(self, data: mujoco.MjData, t: float) -> np.ndarray:
        ctrl = np.empty(self.nu)
        for r in self.robots.values():
            ctrl[r.act_ids] = r.control(data, t)
        return ctrl

    def update_reference(self, rows: np.ndarray, qpos_a: np.ndarray | None = None,
                         qpos_b: np.ndarray | None = None,
                         feet_a: np.ndarray | None = None,
                         feet_b: np.ndarray | None = None,
                         touch_a: np.ndarray | None = None,
                         touch_b: np.ndarray | None = None) -> None:
        """Live reference update for one or both robots (see RobotTeacher)."""
        for prefix, qpos, feet, touch in (("a_", qpos_a, feet_a, touch_a),
                                          ("b_", qpos_b, feet_b, touch_b)):
            if qpos is not None:
                self.robots[prefix].update_reference(rows, qpos, feet, touch)

    def phase_kind(self, prefix: str, t: float) -> str:
        """Reference phase label of ``prefix`` at time ``t`` (for reporting)."""
        return self.robots[prefix].phase_kind(t)

    def last_offsets(self, prefix: str) -> np.ndarray:
        """Joint offsets applied on the most recent tick for ``prefix``."""
        return self.robots[prefix].last_offsets()


#: regularization weight of the leg-IK least squares (rad^-1 scaling of rows)
REG_LAMBDA = 0.05


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
