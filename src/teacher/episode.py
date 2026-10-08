"""Teacher episode rollout + evaluation metrics (phase 3).

``run_episode`` rolls the teacher (or any controller exposing
``control(data, t) -> ctrl58``) from the reference's own initial state plus a
seeded perturbation, at 50 Hz control on the 2 ms sim.

Metrics (per task spec):
  stay_up_frac   per robot: among frames whose REFERENCE pelvis is >= 0.45 m,
                 the fraction where sim pelvis >= 0.45 m AND torso tilt < 30
                 deg. Frames where the reference itself is down (takedown
                 finishes, sprawled defender, stand-up start) never enter the
                 denominator, so prone-by-design segments are excluded.
  final_landmark_dist
                 weighted mean landmark distance (retarget preset 'default'
                 weights 4/1.5/0.5 over the 19 sites) between the simulated
                 pose at the reference end and the reference end pose.
  ground/inter/self penetration
                 max depth over the rollout (same conventions as phase 2's
                 validate_refs).
  sprawl_end_*   SPRAWL end-posture checks (defender prone, attacker on top).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from retarget.landmarks import SOLVED_SITES, site_weights
from retarget.scene import robot_slice

CTRL_DT = 0.02           # 50 Hz
STAND_Z = 0.45           # stay-up pelvis threshold (task spec)
TILT_MAX_DEG = 30.0      # stay-up tilt threshold (task spec)
HOLD_TAIL = 0.3          # s of extra execution after the reference ends
SAMPLE_EVERY = 5         # metric rows every 5 ctrl ticks (10 Hz)


def _perturb_initial(qpos_a0, qpos_b0, seed: int):
    """Seeded initial-state perturbation (joint noise + base wobble)."""
    rng = np.random.default_rng(seed)

    def pert(q0):
        q = q0.copy()
        q[7:36] += rng.normal(0.0, 0.01, 29)        # joints +-0.01 rad
        q[:2] += rng.normal(0.0, 0.005, 2)          # base xy +-5 mm
        q[2] += float(rng.normal(0.0, 0.004))     # base z +-4 mm
        ax = rng.normal(0.0, 0.01, 3)               # small base tilt
        ang = np.linalg.norm(ax)
        if ang > 1e-9:
            dq = np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * ax / ang])
            w0 = q[3:7] / np.linalg.norm(q[3:7])
            w1 = _quat_mul(dq, w0)
            q[3:7] = w1 / np.linalg.norm(w1)
        return q

    qa, qb = pert(qpos_a0), pert(qpos_b0)
    v = np.zeros(70)
    v[6:35] = rng.normal(0.0, 0.05, 29)             # A joint vel
    v[41:70] = rng.normal(0.0, 0.05, 29)            # B joint vel
    v[0:2] = rng.normal(0.0, 0.02, 2)
    v[35:37] = rng.normal(0.0, 0.02, 2)
    return qa, qb, v


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


@dataclass
class EpisodeResult:
    rows: np.ndarray                     # (N, 10): t z_a z_b tilt_a tilt_b
                                          # pen_a pen_b inter self
    end_qpos: np.ndarray                 # (72,) pose at reference end
    ctrl_hist: list = field(default_factory=list)

    # -- derived metrics -------------------------------------------------

    def stay_up_frac(self, t_ref, qpos_a, qpos_b) -> dict:
        t, za, zb = self.rows[:, 0], self.rows[:, 1], self.rows[:, 2]
        ta, tb = self.rows[:, 3], self.rows[:, 4]
        in_ref = t <= t_ref[-1]
        out = {}
        for name, z, tilt, q in (("a", za, ta, qpos_a), ("b", zb, tb, qpos_b)):
            ref_z = np.interp(t, t_ref, q[:, 2])
            elig = (ref_z >= STAND_Z) & in_ref
            ok = elig & (z >= STAND_Z) & (tilt < TILT_MAX_DEG)
            out[name] = float(ok.sum() / elig.sum()) if elig.any() else 1.0
        out["min"] = min(out["a"], out["b"])
        return out

    def penetration(self) -> dict:
        return {"ground": float(max(self.rows[:, 5].max(), self.rows[:, 6].max())),
                "inter": float(self.rows[:, 7].max()),
                "self": float(self.rows[:, 8].max())}


def _tilt_deg(model, data, prefix):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                            f"{prefix}torso_link")
    up = data.xmat[bid].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))


def run_episode(model, controller, qpos_a, qpos_b, t_ref, seed: int = 0,
                capture_ctrl: bool = False) -> EpisodeResult:
    """One seeded rollout; returns rows + end pose (+ctrl history)."""
    data = mujoco.MjData(model)
    sa, sb = robot_slice(model, "a_"), robot_slice(model, "b_")
    qa0, qb0, v0 = _perturb_initial(qpos_a[0], qpos_b[0], seed)
    data.qpos[sa], data.qpos[sb] = qa0, qb0
    data.qvel[:] = v0
    mujoco.mj_forward(model, data)

    foot_sites = {p: np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                                 f"{p}{n}")
                               for n in ("left_toe", "right_toe", "left_heel",
                                         "right_heel", "left_ankle",
                                         "right_ankle")])
                  for p in ("a_", "b_")}
    body_of_geom = np.array([int(model.geom_bodyid[g])
                             for g in range(model.ngeom)])
    robot_of_body = np.array([
        0 if model.body(int(b)).name.startswith("a_")
        else 1 if model.body(int(b)).name.startswith("b_") else -1
        for b in range(model.nbody)])
    parent = model.body_parentid

    dt = model.opt.timestep
    steps_per_ctrl = int(round(CTRL_DT / dt))
    t_end = float(t_ref[-1]) + HOLD_TAIL
    n_steps = int(np.ceil(t_end / dt))
    rows = []
    ctrl_hist = [] if capture_ctrl else None
    end_qpos = np.zeros(72)
    captured_end = False
    for step in range(n_steps):
        t = step * dt
        if step % steps_per_ctrl == 0:
            ctrl = controller.control(data, t)
            if ctrl_hist is not None:
                ctrl_hist.append((float(t), ctrl.copy()))
            data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        if not captured_end and data.time >= t_ref[-1]:
            end_qpos[:] = data.qpos
            captured_end = True
        if step % (steps_per_ctrl * SAMPLE_EVERY):
            continue
        inter = self_d = 0.0
        for c in range(data.ncon):
            con = data.contact[c]
            b1, b2 = body_of_geom[int(con.geom1)], body_of_geom[int(con.geom2)]
            r1, r2 = robot_of_body[b1], robot_of_body[b2]
            depth = float(-con.dist)
            if r1 >= 0 and r2 >= 0 and r1 != r2:
                inter = max(inter, depth)
            elif r1 == r2 and r1 >= 0:
                if b1 != b2 and parent[b1] != b2 and parent[b2] != b1:
                    self_d = max(self_d, depth)
        pen = {p: float(max(0.0, -data.site_xpos[ids][:, 2].min()))
               for p, ids in foot_sites.items()}
        rows.append((float(data.time),
                     float(data.qpos[sa][2]), float(data.qpos[sb][2]),
                     _tilt_deg(model, data, "a_"), _tilt_deg(model, data, "b_"),
                     pen["a_"], pen["b_"], inter, self_d, 0.0))
    if not captured_end:
        end_qpos[:] = data.qpos
    return EpisodeResult(np.array(rows), end_qpos, ctrl_hist or [])


def final_landmark_dist(model, end_qpos, qpos_a_end, qpos_b_end) -> dict:
    """Weighted mean landmark distance (m) of the end pose vs reference end."""
    w = site_weights("default")
    site_ids = {p: [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                      f"{p}{s}") for s in SOLVED_SITES]
                for p in ("a_", "b_")}
    data = mujoco.MjData(model)
    data.qpos[:] = end_qpos
    mujoco.mj_kinematics(model, data)
    ref = mujoco.MjData(model)
    ref.qpos[0:36], ref.qpos[36:72] = qpos_a_end, qpos_b_end
    mujoco.mj_kinematics(model, ref)
    out = {}
    for name, p in (("a", "a_"), ("b", "b_")):
        sim = data.site_xpos[site_ids[p]]
        want = ref.site_xpos[site_ids[p]]
        err = np.linalg.norm(sim - want, axis=1)
        out[name] = float(np.sqrt((w * err ** 2).sum() / w.sum()))
    out["max"] = float(max(out["a"], out["b"]))
    return out


def sprawl_end_posture(model, end_qpos, qpos_b_ref_end) -> dict:
    """SPRAWL end checks: defender prone, attacker sprawled on top of it."""
    z_a, z_b = float(end_qpos[2]), float(end_qpos[38])
    data = mujoco.MjData(model)
    data.qpos[:] = end_qpos
    mujoco.mj_kinematics(model, data)
    core_a = data.site_xpos[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "a_core")]
    core_b = data.site_xpos[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "b_core")]
    return {
        "defender_prone": bool(z_b < STAND_Z),
        "attacker_low": bool(z_a < STAND_Z + 0.10),
        "attacker_above_defender": bool(core_a[2] > core_b[2]),
        "defender_pelvis_z": z_b, "attacker_pelvis_z": z_a,
        "core_sep": float(np.linalg.norm(core_a - core_b)),
    }
