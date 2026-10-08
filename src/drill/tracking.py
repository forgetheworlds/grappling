"""Reference-trajectory tracking mode: drive the plan from a retargeted track.

The drill up to L2 is built from a *constructed* stance and a scripted sequence.
This module adds the other direction: a track recorded from the operator's own
video, retargeted to the G1 joint layout (``data/refs_video/<name>.npz``, keys
``qpos_a (T, 36) f64 @ 50 Hz``, ``t``, ``technique``, ``meta``; the retargeting
summary is ``data/refs_video/retarget_summary.json``) drives the robot's *plan*
-- base pose and FK sole targets per frame -- with the same balance law on top.

What is honest about the result, and what is not:

* the tracks are retargeted from **monocular video** and are **not dynamically
  validated** (6.9 % of the reference robot-frames are even statically
  holdable; ``reports/2026-10-08/support_envelope.md``).  Tracking is therefore
  accepted and reported as **partial**;
* tracking authority is **capped** exactly where the reference would take the
  robot outside its support: the reference's clock slows (and stops) when the
  measured CoM margin falls, and the plan's base is kept within
  ``TrackParams.base_follow_max`` of the *body*, so the leg IK keeps solving for
  a base the robot is actually in (a dishonest reference is the failure the
  stepping primitive paid for -- see ``drill.stepping``);
* the deviation is the finding: where the robot cannot do what the reference
  asks, the report says *reference demands X, robot did Y*.

Nothing here is learned: the reference is a trajectory, the controller is the
same measured-feedback law the drill uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import kin as K
from .balance import DrillPlan, FootTarget
from .controller import FeasibleDrill

REPO = Path(__file__).resolve().parents[2]
TRACK_DIR = REPO / "data" / "refs_video"

#: the order the tracks are attempted in (report order, easiest first)
TRACK_ORDER = ("stance_widen_step", "stalk_shuffle", "circle_step",
               "level_change_full", "shot_entry_full", "knee_sprawl_entry",
               "knee_sprawl_hold", "shot_recover")

#: joints weighted below 1.0 in the tracking error (the gesture channels); the
#: legs and waist carry the motion, the wrists are morphology-bound noise
JOINT_WEIGHTS = np.array([1.0] * 12 + [1.0] * 3 + [0.4, 0.4, 0.4, 0.6, 0.3, 0.3, 0.3] * 2)

#: sites included in the site-tracking error (names of the drill scene)
SITE_NAMES = ("left_knee", "right_knee", "left_wrist", "right_wrist", "head",
              "left_toe", "right_toe")


@dataclass
class TrackParams:
    """Tracking-mode limits (all measured or explicitly chosen).

    ``com_floor`` is the support-envelope floor the *reference* is clamped to:
    the reference robot-frames are overwhelmingly not statically holdable
    (``reports/2026-10-08/support_envelope.md``: 6.9 % of frames), so a faithful
    replay topples within a second.  The clamp translates the whole reference
    pose (base *and* foot targets -- translation preserves the shape) until its
    CoM sits ``com_floor`` inside the *measured* support hull, and the size of
    that translation is the reported deviation.
    """

    ref_rate: float = 1.0           # reference clock rate at full authority
    ref_rate_min: float = 0.0       # the clock may stop (that *is* the cap)
    margin_gate_lo: float = 0.000   # measured CoM margin where authority starts closing
    margin_gate_hi: float = 0.030   # ... and where it is fully open
    base_follow_max: float = 0.050  # plan base kept within this of the body (m)
    base_follow_rate: float = 0.12  # m/s slew of that correction
    foot_rate: float = 0.45         # m/s cap on a planned swing-foot target move
    blend_down: float = 0.04        # authority slew per tick (50 Hz)
    slow_tau: float = 0.25          # s: first-order blend of z / roll / upper body
    com_floor: float = 0.015        # planned CoM margin the clamp keeps (m)


@dataclass
class Track:
    """One retargeted reference track (the npz as written by the retargeter)."""

    name: str
    qpos: np.ndarray                # (T, 36) f64
    t: np.ndarray                   # (T,)
    technique: str = ""
    meta: dict = field(default_factory=dict)
    path: Path | None = None

    @classmethod
    def load(cls, name: str | Path) -> "Track":
        path = Path(name)
        if not path.exists():
            path = TRACK_DIR / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"no track {name!r} in {TRACK_DIR}")
        z = np.load(path, allow_pickle=True)
        return cls(name=path.stem, qpos=np.asarray(z["qpos_a"], float),
                   t=np.asarray(z["t"], float),
                   technique=str(z["technique"]) if "technique" in z else "",
                   meta=(z["meta"].item() if "meta" in z else {}),
                   path=path)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) else 0.0

    def at(self, tau: float) -> np.ndarray:
        """Reference qpos (36,) linearly interpolated at track time ``tau`` (s)."""
        tau = float(np.clip(tau, self.t[0], self.t[-1]))
        k = int(np.searchsorted(self.t, tau, side="right") - 1)
        k = max(0, min(k, len(self.t) - 2)) if len(self.t) > 1 else 0
        t0, t1 = float(self.t[k]), float(self.t[min(k + 1, len(self.t) - 1)])
        u = 0.0 if t1 <= t0 else (tau - t0) / (t1 - t0)
        return (1.0 - u) * self.qpos[k] + u * self.qpos[min(k + 1, len(self.qpos) - 1)]

    def start_qpos(self, model=None, ids=None, floor: float = 0.015) -> np.ndarray:
        """The episode's single initial state: frame 0, leaned into support.

        A retargeted frame 0 is usually *not* holdable (measured frame-0 CoM
        margins run from -0.40 m to +0.07 m).  The pose is therefore leaned: the
        pelvis is moved by the clamp vector and the legs are re-solved by IK so
        the feet keep their recorded world positions.  That is a *reference
        adjustment* of exactly the kind ``support_envelope.md`` recommends for
        the standing postures -- the deviation is the lean, it is reported, and
        the alternative (a rigid translation) provably cannot change a
        CoM-versus-support relation.
        """
        q = np.array(self.qpos[0], float)
        if model is None or ids is None:
            return q
        import mujoco

        d = mujoco.MjData(model)
        d.qpos[:] = q
        mujoco.mj_forward(model, d)
        hull = ids.support_polygon(d, ids.foot_contact(d))
        com = ids.com_xy(d)
        m = float(K.polygon_margin(com, hull)) if len(hull) >= 3 else 0.0
        if len(hull) >= 3 and m < floor:
            delta = TrackController._clamp_shift(com, hull, floor)
            feet = {s: np.array(d.xpos[ids.foot_body[s]], float) for s in K.SIDES}
            yaws = {s: _foot_yaw(ids, d, s) for s in K.SIDES}
            zs = {s: float(np.asarray(d.site_xpos[ids.sole_sites[s]], float)[:, 2].min())
                  for s in K.SIDES}
            planted = ids.foot_contact(d)
            q[:2] += delta
            d.qpos[:] = q
            mujoco.mj_forward(model, d)
            for s in K.SIDES:
                if not planted[K.SIDES.index(s)]:
                    continue                      # keep a swinging foot as recorded
                tg = K.foot_targets(ids, s, feet[s][:2], yaws[s],
                                    sole_z=max(zs[s], K.SOLE_REST_Z))
                qadr = ids.leg_qadr[s]
                q[qadr] = K.leg_ik(model, d, ids, s, tg, q[qadr], iters=40)
                d.qpos[:] = q
                mujoco.mj_forward(model, d)
        return q


def _foot_yaw(ids: K.RobotIds, d, side: str) -> float:
    """World heading of one foot's sole long axis (rad)."""
    pts = np.asarray(d.site_xpos[ids.sole_sites[side]], float)
    heel = pts[list(K.HEEL_ROWS)].mean(axis=0)
    toe = pts[list(K.TOE_ROWS)].mean(axis=0)
    return float(np.arctan2(toe[1] - heel[1], toe[0] - heel[0]))


class TrackController(FeasibleDrill):
    """``FeasibleDrill`` with its plan driven by a reference track.

    The scripted controller's own plan builder is replaced: every tick the
    reference's frame (interpolated at the authority-controlled clock) becomes
    the plan -- pelvis pose, per-foot sole targets from the reference's FK,
    upper body from the reference's joints -- and the measured balance law,
    foot anchoring and safety layers run unchanged on top.
    """

    kind = "scripted reference tracking (TrackController)"
    name = "track"

    def __init__(self, stance, ids, track: str | Path, seed: int = 0,
                 params: TrackParams | None = None, **kw):
        import dataclasses

        from .balance import BalanceParams

        # the drill's safety blend pulls the reference toward the *built stance*,
        # which would drag the feet wholesale while tracking a reference foot in
        # the air; the mode's own authority cap is the fallback instead
        bp = dataclasses.replace(BalanceParams(), govern_e0=9.9, govern_e1=10.0)
        super().__init__(stance, ids, rung="L3", seed=seed, params=bp, **kw)
        self.track = Track.load(track) if not isinstance(track, Track) else track
        self.tp = params or TrackParams()
        self.tau = 0.0                  # reference clock (s of track time)
        self.authority = 1.0
        self.stats = {"base_err": [], "e_joint": [], "e_site": [], "coast": 0,
                      "frozen_s": 0.0, "rate_mean": 1.0}

    # -- interface ---------------------------------------------------------
    def reset(self, model, data) -> None:
        super().reset(model, data)
        # Planar alignment of the reference to the robot's initial state.  The
        # tracks are retargeted from monocular video in their own world frame,
        # and their frame-0 poses are mostly *not* holdable (support_envelope.md:
        # the frame-0 CoM margin runs from -0.40 m to +0.07 m, and two tracks
        # start with no foot on the mat).  What is tracked is therefore the
        # reference's *motion*: its base is rotated/translated so that frame 0
        # coincides with the robot's start, and the joints (and hence the FK
        # sole targets) are used as recorded.
        ref0 = np.asarray(self.track.qpos[0], float)
        self._ref0 = ref0
        self._align_xy = np.asarray(data.qpos[:2], float) - ref0[:2]
        self._align_dyaw = K.quat_yaw(data.qpos[3:7]) - K.quat_yaw(ref0[3:7])
        self._align_z = float(data.qpos[2]) - float(ref0[2])
        R = K.z_rot(self._align_dyaw)
        self._align_rot = R
        self.tau = float(self.track.t[0])
        self.authority = 1.0
        self.stats = {"base_err": [], "e_joint": [], "e_site": [], "coast": 0,
                      "frozen_s": 0.0, "rate_mean": 1.0}

    # -- the plan ----------------------------------------------------------
    def _update_plan(self, model, data, cmd, dt: float) -> None:
        tp = self.tp
        # 1) tracking authority: the measured CoM margin decides how fast the
        #    reference may advance.  A reference that walks the CoM out of the
        #    support has its clock slowed -- and stopped -- instead of being
        #    followed off the feet.
        margin = float(getattr(self.law, "info", {}).get("margin", 0.05))
        a_t = float(np.clip((margin - tp.margin_gate_lo)
                            / max(1e-6, tp.margin_gate_hi - tp.margin_gate_lo), 0.0, 1.0))
        self.authority += float(np.clip(a_t - self.authority, -tp.blend_down, tp.blend_down))
        rate = tp.ref_rate_min + (tp.ref_rate - tp.ref_rate_min) * self.authority
        self.stats["rate_mean"] = (0.98 * self.stats["rate_mean"] + 0.02 * rate)
        if rate < 0.6:
            self.stats["frozen_s"] += dt
        self.tau = min(self.tau + dt * rate, float(self.track.t[-1]))
        # 2) the reference frame -> the plan, in the robot's aligned frame.  The
        #    reference's own joints become the servo targets directly (a
        #    retargeted track *is* a joint trajectory): re-solving them through
        #    the cartesian IK only adds a transient and a contradiction.
        ref = self._aligned(self.track.at(self.tau))
        q_ref = np.array(ref[7:36], float)
        plan_ref = self._plan_from_reference(model, ref)
        # 3) support-envelope clamp: lean the reference pose until its CoM sits
        #    inside the *measured* support (see TrackParams).  The lean is the
        #    deviation this mode costs and the first thing the report prints.
        com_ref = self._reference_com_xy(model, ref)
        hull = self._support_hull(data)
        m = float(K.polygon_margin(com_ref, hull)) if len(hull) >= 3 else 0.0
        self.stats.setdefault("com_margin_ref", []).append(m)
        if len(hull) >= 3 and m < tp.com_floor:
            delta = self._clamp_shift(com_ref, hull, tp.com_floor)
            self.stats.setdefault("clamp", []).append(float(np.linalg.norm(delta)))
            self.stats["clamped_ticks"] = self.stats.get("clamped_ticks", 0) + 1
            # the *base* moves, the foot targets do not: the correction is a
            # lean (moving the feet with the base would cancel it -- a rigid
            # translation leaves the CoM-vs-support relation unchanged).  The
            # reference's *joints* must be leaned with it, or the commanded
            # posture would keep putting the CoM back where the clamp took it
            # from: the legs are re-solved by IK with the feet held where the
            # reference has them.
            plan_ref.base_xyz[:2] = plan_ref.base_xyz[:2] + delta
            q_ref = self._leaned_legs(model, plan_ref, q_ref)
        self.solver.direct = q_ref
        # 4) base honesty: the plan's base may not leave the body by more than
        #    base_follow_max -- the deviation this cap costs is reported
        body = np.asarray(data.qpos[:2], float)
        d = np.asarray(plan_ref.base_xyz[:2], float) - body
        n = float(np.linalg.norm(d))
        self.stats["base_err"].append(n)
        if n > tp.base_follow_max:
            plan_ref.base_xyz[:2] = body + d * (tp.base_follow_max / n)
            self.stats["coast"] += 1
        # 5) slew the plant into the reference *by the authority*: at full
        #    authority the reference leads; as it closes, the robot holds what
        #    it has instead of following a pose it cannot hold (the reference's
        #    own poses are mostly outside the support -- support_envelope.md)
        a = self.authority
        step = self._slew_xyz(dt) * a
        d = plan_ref.base_xyz - self.plan.base_xyz
        self.plan.base_xyz += np.clip(d, -step, step)
        self.plan.base_yaw = float(self.plan.base_yaw + a * np.clip(
            ((plan_ref.base_yaw - self.plan.base_yaw + np.pi) % (2 * np.pi)) - np.pi,
            -0.6 * dt, 0.6 * dt))
        k = float(np.clip(dt / max(1e-3, tp.slow_tau), 0.0, 0.5)) * a
        for s in K.SIDES:
            ft, fr = self.plan.feet[s], plan_ref.feet[s]
            ft.origin_xy += np.clip(fr.origin_xy - ft.origin_xy,
                                    -tp.foot_rate * dt * a, tp.foot_rate * dt * a)
            ft.sole_z = float(ft.sole_z + k * (fr.sole_z - ft.sole_z))
            ft.yaw = float(ft.yaw + a * np.clip(fr.yaw - ft.yaw, -1.5 * dt, 1.5 * dt))
            ft.roll = float(ft.roll + k * (fr.roll - ft.roll))
            # planted follows the *plan's* own sole height, so the anchoring
            # semantics can never jump ahead of the blended target
            ft.planted = bool(ft.sole_z - K.SOLE_REST_Z < 0.012)
        self.plan.upper = self.plan.upper + np.clip(
            k * (plan_ref.upper - self.plan.upper), -0.08, 0.08)
        self.plan.label = f"TRACK {self.track.name} t={self.tau:.2f}s"
        # 6) report the tracking error against the same reference frame
        self._measure_tracking(model, data, ref)

    def _leaned_legs(self, model, plan_ref: DrillPlan, q_ref: np.ndarray) -> np.ndarray:
        """Re-solve the reference's legs for its leaned base (feet held)."""
        import mujoco

        d = self._ref_data()
        d.qpos[:7] = np.concatenate([plan_ref.base_xyz,
                                     K.yaw_quat(plan_ref.base_yaw)])
        d.qpos[7:36] = q_ref
        mujoco.mj_forward(model, d)
        q = np.array(q_ref, float)
        for s in K.SIDES:
            tg = plan_ref.feet[s].points(self.ids, s)
            adr = self.ids.leg_qadr[s]
            q[adr - 7] = K.leg_ik(model, d, self.ids, s, tg, q[adr - 7], iters=6)
            d.qpos[adr] = q[adr - 7]
            mujoco.mj_forward(model, d)
        return q

    def _support_hull(self, data) -> np.ndarray:
        """The *measured* support hull of the feet currently on the mat."""
        loaded = self.ids.foot_contact(data)
        return self.ids.support_polygon(data, loaded)

    def _reference_com_xy(self, model, ref: np.ndarray) -> np.ndarray:
        d = self._ref_data2()
        d.qpos[:] = ref
        K.scratch_kinematics(model, d)
        return self.ids.com_xy(d)

    @staticmethod
    def _clamp_shift(com: np.ndarray, hull: np.ndarray, floor: float) -> np.ndarray:
        """Translation that puts ``com`` at margin ``floor`` inside ``hull``."""
        p = np.asarray(com, float)
        best, bd = None, np.inf
        for i in range(len(hull)):
            a, b = hull[i], hull[(i + 1) % len(hull)]
            e = b - a
            t = 0.0 if e @ e < 1e-12 else float(np.clip((p - a) @ e / (e @ e), 0.0, 1.0))
            q = a + t * e
            d = float(np.linalg.norm(p - q))
            if d < bd:
                best, bd = q, d
        if best is None:
            return np.zeros(2)
        inward = best - p
        n = float(np.linalg.norm(inward))
        target = best + (floor - 0.0) * (inward / n) if n > 1e-9 else best
        return np.asarray(target - p, float)

    def _aligned(self, ref: np.ndarray) -> np.ndarray:
        """Reference frame -> the robot's world (planar alignment, see reset)."""
        out = np.array(ref, float)
        out[:2] = self._align_rot @ (np.asarray(ref[:2], float) - self._ref0[:2]) \
            + self._ref0[:2] + self._align_xy
        out[2] = float(ref[2]) + self._align_z
        out[3:7] = K.yaw_quat(K.quat_yaw(np.asarray(ref[3:7], float)) + self._align_dyaw)
        return out

    def _slew_xyz(self, dt: float) -> np.ndarray:
        return np.array([0.30 * dt, 0.30 * dt, 0.20 * dt])

    def _plan_from_reference(self, model, ref: np.ndarray) -> DrillPlan:
        """Reference qpos (36,) -> DrillPlan (world frame = the reference's)."""
        feet = {}
        d = self._ref_data()
        d.qpos[:] = ref
        K.scratch_kinematics(model, d)
        for s in K.SIDES:
            pts = np.asarray(d.site_xpos[self.ids.sole_sites[s]], float)
            heel = pts[list(K.HEEL_ROWS)].mean(axis=0)
            toe = pts[list(K.TOE_ROWS)].mean(axis=0)
            yaw = float(np.arctan2(toe[1] - heel[1], toe[0] - heel[0]))
            z = float(pts[:, 2].min())
            planted = bool(z - K.SOLE_REST_Z < K.TOUCH_Z)
            # sole-plane tilt about the foot's long axis, the same convention as
            # kin.foot_targets(roll=...): fit z = k * (lateral offset)
            lat = np.array([-np.sin(yaw), np.cos(yaw)])
            off = (pts[:, :2] - pts[:, :2].mean(axis=0)) @ lat
            zc = pts[:, 2] - pts[:, 2].mean()
            den = float(off @ off)
            roll = float(np.clip((off @ zc) / den, -0.6, 0.6)) if den > 1e-12 else 0.0
            feet[s] = FootTarget(np.array(d.xpos[self.ids.foot_body[s]][:2], float),
                                 yaw, sole_z=float(max(z, K.SOLE_REST_Z)),
                                 planted=planted, roll=roll)
        return DrillPlan(base_xyz=np.array(ref[:3], float),
                         base_yaw=K.quat_yaw(ref[3:7]),
                         feet=feet, upper=np.array(ref[19:36], float),
                         label=f"TRACK {self.track.name}")

    # -- the reported tracking error --------------------------------------
    def _measure_tracking(self, model, data, ref: np.ndarray) -> None:
        """Weighted joint + site error against the current reference frame.

        The base is excluded from the joint error (it is free -- the reference's
        own base may be unreachable) and the sites are compared in the
        *pelvis* frame, so the numbers say how well the *body* is being
        reproduced, not how well a global position is.
        """
        q = np.asarray(data.qpos[7:36], float)
        r = np.asarray(ref[7:36], float)
        self.stats["e_joint"].append(float(np.abs(JOINT_WEIGHTS * (q - r)).mean()))
        e = self._ref_data()
        e.qpos[:] = data.qpos
        K.scratch_kinematics(model, e)
        base = np.array(data.qpos[:2], float)
        yaw = K.quat_yaw(data.qpos[3:7])
        ref_d = self._ref_data2()
        ref_d.qpos[:] = ref
        K.scratch_kinematics(model, ref_d)
        errs = []
        for name in SITE_NAMES:
            sid = self.ids.site[name]
            p = K.z_rot(-yaw) @ (np.asarray(e.site_xpos[sid], float)[:2] - base)
            pr = (K.z_rot(-K.quat_yaw(ref[3:7]))
                  @ (np.asarray(ref_d.site_xpos[sid], float)[:2] - np.asarray(ref[:2], float)))
            errs.append(float(np.linalg.norm(p - pr)))
        self.stats["e_site"].append(float(np.mean(errs)) if errs else 0.0)

    def _ref_data2(self):
        if not hasattr(self, "_rd2") or self._rd2 is None:
            import mujoco

            self._rd2 = mujoco.MjData(self.ids.model)
        return self._rd2

    # -- the reporting block ----------------------------------------------
    def tracking_report(self) -> dict:
        """Aggregates for the evidence bundle (means + p95, never a mean alone)."""
        ej = np.asarray(self.stats["e_joint"], float)
        es = np.asarray(self.stats["e_site"], float)
        be = np.asarray(self.stats["base_err"], float)
        return {
            "track": self.track.name, "technique": self.track.technique,
            "track_duration_s": round(self.track.duration, 2),
            "tracked_s": round(float(self.tau - self.track.t[0]), 2),
            "tracked_frac": round(float((self.tau - self.track.t[0])
                                        / max(1e-6, self.track.duration)), 4),
            "e_joint_mean_rad": round(float(ej.mean()), 4) if len(ej) else None,
            "e_joint_p95_rad": round(float(np.percentile(ej, 95)), 4) if len(ej) else None,
            "e_site_mean_m": round(float(es.mean()), 4) if len(es) else None,
            "e_site_p95_m": round(float(np.percentile(es, 95)), 4) if len(es) else None,
            "base_follow_err_mean_m": round(float(be.mean()), 4) if len(be) else None,
            "base_follow_err_max_m": round(float(be.max()), 4) if len(be) else None,
            "coast_ticks": int(self.stats["coast"]),
            "com_margin_ref_mean_m": round(float(np.mean(self.stats["com_margin_ref"])), 4)
            if self.stats.get("com_margin_ref") else None,
            "com_margin_ref_min_m": round(float(np.min(self.stats["com_margin_ref"])), 4)
            if self.stats.get("com_margin_ref") else None,
            "clamped_ticks": int(self.stats.get("clamped_ticks", 0)),
            "clamp_shift_mean_m": round(float(np.mean(self.stats["clamp"])), 4)
            if self.stats.get("clamp") else 0.0,
            "clamp_shift_max_m": round(float(np.max(self.stats["clamp"])), 4)
            if self.stats.get("clamp") else 0.0,
            "authority_frozen_s": round(float(self.stats["frozen_s"]), 2),
            "ref_rate_mean": round(float(self.stats["rate_mean"]), 3),
        }
