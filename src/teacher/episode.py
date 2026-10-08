"""Teacher episode rollout + evaluation metrics (phase 3).

``run_episode`` rolls the teacher (or any controller exposing
``control(data, t) -> ctrl58``) from the reference's own initial state plus a
seeded perturbation, at 50 Hz control on the 2 ms sim.

Metrics (per task spec):

  stay_up_frac   among frames whose REFERENCE pelvis is >= 0.45 m, the
                 fraction where the sim pelvis is >= 0.45 m AND the sim torso
                 tilt is inside budget.  The budget is
                 ``max(30 deg, ref tilt + 15 deg)`` because the references
                 themselves lean: e.g. the DOUBLE_LEG shooter's torso is
                 30-70 deg from vertical for the whole shot and 100 % of its
                 eligible frames exceed 30 deg, so a literal 30 deg test
                 would reject a perfect tracking of the intended posture.
                 The literal fraction is reported too (``literaltilt``) for
                 comparability with the phase-2 numbers.
  final_landmark_dist
                 weighted RMS landmark distance (retarget preset 'default'
                 weights 4/1.5/0.5 over the 19 sites) between the simulated
                 pose at the reference end and the reference end pose.
  penetration    max depth over the rollout (ground / inter-robot / self),
                 same conventions as phase 2's validate_refs.
  travel         sim base displacement / reference base displacement, per
                 robot; 1.0 = the base followed the reference's world motion.
  sprawl_end_*   SPRAWL end-posture checks (defender prone, attacker atop).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from retarget.landmarks import SOLVED_SITES, site_weights
from retarget.scene import robot_slice

CTRL_DT = 0.02           # 50 Hz
STAND_Z = 0.45           # stay-up pelvis threshold (task spec)
TILT_MAX_DEG = 30.0      # stay-up tilt threshold (task spec)
TILT_SLACK_DEG = 15.0    # allowance over the reference's own tilt
HOLD_TAIL = 0.3          # s of extra execution after the reference ends
SAMPLE_EVERY = 5         # metric rows every 5 ctrl ticks (10 Hz)
VIDEO_FPS = 30           # captured frames per second for the review videos


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _perturb_initial(qpos_a0, qpos_b0, seed: int, joint_sigma: float = 0.01):
    """Seeded initial-state perturbation (joint noise + base wobble)."""
    rng = np.random.default_rng(seed)

    def pert(q0):
        q = q0.copy()
        q[7:36] += rng.normal(0.0, joint_sigma, 29)   # joints +-0.01 rad
        q[:2] += rng.normal(0.0, 0.005, 2)            # base xy +-5 mm
        q[2] += float(rng.normal(0.0, 0.004))         # base z +-4 mm
        ax = rng.normal(0.0, 0.01, 3)                 # small base tilt
        ang = np.linalg.norm(ax)
        if ang > 1e-9:
            dq = np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * ax / ang])
            w0 = q[3:7] / np.linalg.norm(q[3:7])
            w1 = _quat_mul(dq, w0)
            q[3:7] = w1 / np.linalg.norm(w1)
        return q

    qa, qb = pert(qpos_a0), pert(qpos_b0)
    v = np.zeros(70)
    v[6:35] = rng.normal(0.0, 0.05, 29)               # A joint vel
    v[41:70] = rng.normal(0.0, 0.05, 29)              # B joint vel
    v[0:2] = rng.normal(0.0, 0.02, 2)
    v[35:37] = rng.normal(0.0, 0.02, 2)
    return qa, qb, v


def _tilt_deg(model, data, prefix):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                            f"{prefix}torso_link")
    up = data.xmat[bid].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))


def reference_track(model, qpos_a, qpos_b):
    """Per-frame reference pelvis heights, torso tilts and base x positions."""
    d = mujoco.MjData(model)
    T = len(qpos_a)
    out = np.empty((T, 6))
    for k in range(T):
        d.qpos[0:36], d.qpos[36:72] = qpos_a[k], qpos_b[k]
        mujoco.mj_kinematics(model, d)
        out[k] = (qpos_a[k, 2], qpos_b[k, 2], _tilt_deg(model, d, "a_"),
                  _tilt_deg(model, d, "b_"), qpos_a[k, 0], qpos_b[k, 0])
    return out


@dataclass
class EpisodeResult:
    rows: np.ndarray                     # (N, 10): t z_a z_b tilt_a tilt_b
                                          # pen_a pen_b inter self 0
    end_qpos: np.ndarray                 # (72,) pose at reference end
    ref: np.ndarray                      # (T, 6) reference track
    base_x: dict = field(default_factory=dict)   # sim base x traces at 10 Hz
    frames: list = field(default_factory=list)   # (t, qpos72) for video
    traj: np.ndarray | None = None       # (N, 2, 36) qpos at 10 Hz (scorer)

    # -- derived metrics -------------------------------------------------

    def stay_up_frac(self, t_ref, tilt_slack: float = TILT_SLACK_DEG) -> dict:
        """Reference-relative stay-up fractions (see module docstring)."""
        t = self.rows[:, 0]
        ref = self.ref
        tref = np.linspace(0.0, float(t_ref[-1]), len(ref))
        in_ref = t <= t_ref[-1] + 1e-9
        out = {}
        for name, z, tilt, iz, it in (("a", self.rows[:, 1], self.rows[:, 3], 0, 2),
                                      ("b", self.rows[:, 2], self.rows[:, 4], 1, 3)):
            ref_z = np.interp(t, tref, ref[:, iz])
            ref_tilt = np.interp(t, tref, ref[:, it])
            elig = (ref_z >= STAND_Z) & in_ref
            budget = np.maximum(TILT_MAX_DEG, ref_tilt + tilt_slack)
            up = z >= STAND_Z
            n = max(int(elig.sum()), 1)
            out[name] = float((elig & up & (tilt <= budget)).sum() / n)
            out[name + "_pelvis"] = float((elig & up).sum() / n)
            out[name + "_literal"] = float((elig & up & (tilt < TILT_MAX_DEG)).sum() / n)
        out["min"] = min(out["a"], out["b"])
        return out

    def penetration(self) -> dict:
        return {"ground": float(max(self.rows[:, 5].max(), self.rows[:, 6].max())),
                "inter": float(self.rows[:, 7].max()),
                "self": float(self.rows[:, 8].max())}

    def travel_ratio(self, t_ref) -> dict:
        """Sim / reference base displacement over the reference horizon."""
        tref = np.linspace(0.0, float(t_ref[-1]), len(self.ref))
        out = {}
        for name, iz, sim in (("a", 4, self.base_x.get("a")),
                              ("b", 5, self.base_x.get("b"))):
            if not sim:
                continue
            ts = np.array([p[0] for p in sim])
            xs = np.array([p[1] for p in sim])
            m = ts <= t_ref[-1]
            if not m.any():
                continue
            d_ref = float(self.ref[-1, iz] - self.ref[0, iz])
            d_sim = float(xs[m][-1] - xs[m][0])
            out[name] = (float(d_sim / d_ref) if abs(d_ref) > 0.05
                         else float("nan"))
            out[name + "_disp"] = d_sim
            out[name + "_ref_disp"] = d_ref
        return out


def run_episode(model, controller, qpos_a, qpos_b, t_ref, seed: int = 0,
                capture_frames: bool = False) -> EpisodeResult:
    """One seeded rollout; returns rows + end pose (+ 30 fps frames)."""
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
    body_of_geom = np.array([int(model.geom_bodyid[g]) for g in range(model.ngeom)])
    robot_of_body = np.array([
        0 if model.body(int(b)).name.startswith("a_")
        else 1 if model.body(int(b)).name.startswith("b_") else -1
        for b in range(model.nbody)])
    parent = model.body_parentid

    dt = model.opt.timestep
    steps_per_ctrl = int(round(CTRL_DT / dt))
    steps_per_frame = int(round((1.0 / VIDEO_FPS) / dt)) if capture_frames else 0
    t_end = float(t_ref[-1]) + HOLD_TAIL
    n_steps = int(np.ceil(t_end / dt))
    rows, frames = [], []
    traj = []
    base_x = {"a": [], "b": []}
    end_qpos = np.zeros(72)
    captured_end = False
    for step in range(n_steps):
        t = step * dt
        if step % steps_per_ctrl == 0:
            data.ctrl[:] = controller.control(data, t)
        mujoco.mj_step(model, data)
        if capture_frames and step % steps_per_frame == 0:
            frames.append((float(data.time), data.qpos.copy()))
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
        traj.append((data.qpos[sa].copy(), data.qpos[sb].copy()))
        base_x["a"].append((float(data.time), float(data.qpos[sa][0])))
        base_x["b"].append((float(data.time), float(data.qpos[sb][0])))
    if not captured_end:
        end_qpos[:] = data.qpos
    ref = reference_track(model, qpos_a, qpos_b)
    return EpisodeResult(np.array(rows), end_qpos, ref, base_x, frames,
                         np.array(traj))


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


def landmark_track_dist(model, end_qpos, qpos_a_end, qpos_b_end) -> dict:
    """Alias kept explicit: weighted landmark RMS (same as phase-2 metric)."""
    return final_landmark_dist(model, end_qpos, qpos_a_end, qpos_b_end)


def sprawl_end_posture(model, end_qpos) -> dict:
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


def scored_trajectories(res, t_ref, technique: str) -> tuple:
    """(self_traj, opp_traj) for the scorer: the executor robot first.

    Uses only frames up to the end of the reference (the hold tail is an
    execution artefact, not part of the technique).  Lazy scorer import keeps
    the teacher package free of a hard dependency on the judge.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scorer.config import executor_of  # noqa: E402  (lazy: judge only)
    if res.traj is None:
        raise ValueError("episode has no trajectory (run_episode does capture it)")
    m = res.rows[:, 0] <= t_ref[-1] + 1e-9
    tr = res.traj[m]                       # (N, 2, 36)
    if executor_of(technique) == "A":
        return tr[:, 0], tr[:, 1]
    return tr[:, 1], tr[:, 0]


def score_episode(res, t_ref, technique: str, model) -> dict:
    """Technique-scorer result for one episode: per-phase means + overall.

    Lazy in-function import (same pattern as src/rl/reward.ScorerAdapter): the
    teacher trains/executes without the judge; the evaluation script opts in.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scorer.scorer import TechniqueScorer  # noqa: E402
    self_traj, opp_traj = scored_trajectories(res, t_ref, technique)
    scorer = _scorer_for(model)
    return scorer.score_trace(technique, np.ascontiguousarray(self_traj),
                              np.ascontiguousarray(opp_traj))


_SCORERS: dict = {}


def _scorer_for(model):
    """Cached TechniqueScorer per model (one MjData per process, like scorer.scorer)."""
    key = id(model)
    if key not in _SCORERS:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scorer.scorer import TechniqueScorer  # noqa: E402
        _SCORERS[key] = TechniqueScorer(model)
    return _SCORERS[key]
