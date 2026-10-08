"""Continuous-run harness: ONE reset, no internal resets, full trace.

The episode contract (``docs/SOLO_DRILL.md`` §4 and the external review):

* exactly **one** state initialisation at t=0 (a keyframe plus optional seeded
  noise) -- there is no exchange-reset path and no teleport: a fall ends the
  run and is recorded;
* 50 Hz control (20 ms) over 500 Hz physics (2 ms), i.e. 10 ``mj_step`` per
  control tick; the tick counters are part of the trace;
* disturbances are injected as ``data.xfrc_applied`` on the torso for a bounded
  window and are **zeroed on every other physics step** (the known trap:
  a stale applied force silently changes the dynamics of the whole run);
* the trace is the artifact: qpos/qvel/ctrl/contacts/margins/commands/pushes at
  control rate, so metrics and video are computed from the same recording and
  frames are never fabricated.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from . import kin as K
from . import scene as scene_mod

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data" / "drill"

#: torso push application point (the pelvis body, at its CoM height)
PUSH_BODY = "torso_link"


@dataclass
class PushSpec:
    """One disturbance: a body-frame force (N) applied for a bounded window."""

    t: float                       # start time (s)
    dur: float = 0.12              # duration (s)
    fx: float = 0.0                # body-frame force (N), +x = the robot's forward
    fy: float = 0.0
    fz: float = 0.0
    label: str = "push"

    def force(self) -> np.ndarray:
        return np.array([self.fx, self.fy, self.fz], float)


@dataclass
class RunConfig:
    controller: str = "feasible"          # feasible | stance_pd | teacher
    rung: str = "L0"
    seconds: float = 20.0
    seed: int = 0
    start: str = "stand"                  # stand | stance | entry
    pushes: list = field(default_factory=list)
    noise: float = 0.0                    # initial joint noise sigma (rad)
    tag: str = "run"
    render_every: int = 0                 # >0: save qpos at that control-step stride
    max_wall_s: float = 900.0

    def key(self) -> str:
        return f"{self.tag}_{self.controller}_{self.rung}_seed{self.seed}"


@dataclass
class RunResult:
    config: RunConfig
    trace: dict
    events: list
    metrics: dict
    wall_s: float
    aborted: str | None = None

    provenance: dict = field(default_factory=dict)

    def save(self, directory: Path | str = DATA_DIR) -> dict:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        stem = self.config.key()
        npz = d / f"{stem}.npz"
        np.savez_compressed(npz, **self.trace)
        js = d / f"{stem}.json"
        js.write_text(json.dumps({"config": asdict(self.config),
                                  "metrics": self.metrics,
                                  "events": self.events,
                                  "wall_s": self.wall_s,
                                  "aborted": self.aborted,
                                  "provenance": provenance(self.config)},
                                 indent=1, default=str))
        return {"npz": str(npz), "json": str(js)}


def provenance(cfg: "RunConfig") -> dict:
    """The evidence-protocol provenance block (docs/EVIDENCE_PROTOCOL.md)."""
    import hashlib
    import subprocess
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                cwd=REPO, capture_output=True, text=True,
                                timeout=10).stdout.strip() or "unknown"
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                                    capture_output=True, text=True,
                                    timeout=10).stdout.strip())
    except Exception:                                    # pragma: no cover
        commit, dirty = "unknown", None
    blob = json.dumps(asdict(cfg), sort_keys=True, default=str).encode()
    return {"git_commit": commit, "worktree_dirty": dirty,
            "controller": cfg.controller, "rung": cfg.rung, "seed": cfg.seed,
            "config_hash": hashlib.sha256(blob).hexdigest()[:16],
            "control_hz": 50, "physics_dt_s": 0.002, "physics_substeps_per_tick": 10,
            "scene": "robots/g1/g1.xml + landmark/sole sites (single robot, MjSpec)",
            "reset_count": 0, "one_reset": True,
            "sim_wall_s": None,
            "reproduce": (f"MUJOCO_GL=egl python scripts/solo_drill_render.py run "
                          f"--controller {cfg.controller} --rung {cfg.rung} "
                          f"--seconds {cfg.seconds:g} --seed {cfg.seed} "
                          f"--start {cfg.start} --out-tag {cfg.tag}")}


# --------------------------------------------------------------------- fall check
class FallDetector:
    """Terminates on a real fall: low pelvis, large tilt, or dorsal contact.

    Deliberately *conservative about the vocabulary* (review §6): knees, a low
    crouch, hands on the mat and a foot in the air are NOT falls.  What counts
    is the whole body going down and staying down (pelvis below 0.42 m), the
    torso leaving the vertical by more than 55 deg, or the torso's own up-axis
    pointing below the horizon while the pelvis is low (dorsal).
    """

    def __init__(self, persistence_s: float = 0.16, pelvis_z: float = 0.42,
                 tilt_deg: float = 55.0):
        self.persistence = int(round(persistence_s / 0.02))
        self.pelvis_z = float(pelvis_z)
        self.tilt_deg = float(tilt_deg)
        self._n = 0
        self.reason = ""

    def update(self, ids: K.RobotIds, data: mujoco.MjData) -> bool:
        z = ids.pelvis_z(data)
        tilt = ids.torso_tilt_deg(data)
        up_z = float(ids.torso_up(data)[2])
        bad = (z < self.pelvis_z) or (tilt > self.tilt_deg) or (up_z < -0.25)
        self._n = self._n + 1 if bad else 0
        if self._n >= self.persistence:
            self.reason = (f"fall: pelvis_z={z:.3f} tilt={tilt:.1f} up_z={up_z:.2f} "
                           f"sustained {self._n} ticks")
            return True
        return False


# -------------------------------------------------------------------------- runner
def build_controller(cfg: RunConfig, model, ids, stance):
    """Controller factory (import kept local so the ladder works without teacher)."""
    from . import controller as controller_mod

    if cfg.controller == "feasible":
        return controller_mod.FeasibleDrill(stance, ids, rung=cfg.rung, seed=cfg.seed)
    if cfg.controller == "stance_pd":
        return controller_mod.StancePD(stance, ids)
    if cfg.controller == "teacher":
        from .teacher_adapter import TeacherAdapter

        return TeacherAdapter(stance, ids, seed=cfg.seed)
    raise ValueError(f"unknown controller {cfg.controller!r}")


def start_qpos(model, cfg: RunConfig, stance) -> np.ndarray:
    """The single initial state of the run (never re-entered later)."""
    if cfg.start == "stand":
        return scene_mod.keyframe(model, "stand").copy()
    if cfg.start == "stance":
        return stance.qpos.copy()
    raise ValueError(f"unknown start {cfg.start!r}")


def run(cfg: RunConfig, stance=None, model=None, scheduler=None,
        verbose: bool = True) -> RunResult:
    """Simulate one continuous episode; returns the trace, events, metrics."""
    from . import metrics as metrics_mod
    from . import scheduler as scheduler_mod

    t0 = time.time()
    ids_model = model or scene_mod.load_model(None if stance is None else stance.qpos)
    ids = K.RobotIds.build(ids_model)
    if stance is None:
        from . import posture as posture_mod

        stance = posture_mod.build_stance(ids_model, posture_mod.StanceSpec(), ids)
    ctrl = build_controller(cfg, ids_model, ids, stance)
    sched = scheduler or scheduler_mod.SkillScheduler(cfg.rung, seed=cfg.seed)

    data = mujoco.MjData(ids_model)
    rng = np.random.default_rng(cfg.seed)
    data.qpos[:] = start_qpos(ids_model, cfg, stance)
    if cfg.noise > 0:
        data.qpos[7:36] += rng.normal(0.0, cfg.noise, 29)
        data.qpos[:2] += rng.normal(0.0, 0.002, 2)
    data.qvel[:] = 0.0
    data.xfrc_applied[:] = 0.0
    data.ctrl[:] = np.clip(data.qpos[7:36], ids.ctrl_lo, ids.ctrl_hi)
    mujoco.mj_forward(ids_model, data)
    ctrl.reset(ids_model, data)
    if hasattr(sched, "_stance_z"):
        sched._stance_z = float(getattr(stance.spec, "pelvis_z", 0.74))
    sched.reset(ids_model, data)

    push_body = mujoco.mj_name2id(ids_model, mujoco.mjtObj.mjOBJ_BODY, PUSH_BODY)
    det = FallDetector()
    n_ctrl = int(round(cfg.seconds / 0.02))
    sub = int(round(0.02 / ids_model.opt.timestep))

    tr = {k: [] for k in ("t", "qpos", "qvel", "ctrl", "com", "margin", "tilt_deg",
                          "contact", "clearance", "skill_id", "cmd", "push",
                          "com_ref", "foot_xy", "plan_planted", "ik_err",
                          "e_track", "safety_alpha", "sole_pts", "knee_z",
                          "pen_min", "sat_frac", "push_force", "foot_load",
                          "hands_local", "head_z")
          }
    events: list = []
    push_state = []
    for ps in cfg.pushes:
        push_state.append({"spec": ps, "n": 0})
    aborted = None
    n_sub_phys = 0
    for k in range(n_ctrl):
        t = k * 0.02
        cmd = sched.tick(ids, data, ctrl, t)
        data.ctrl[:] = ctrl.act(ids_model, data, cmd)
        # -- disturbances: bounded windows, zeroed on every other substep ----
        active = np.zeros(3)
        for st in push_state:
            lo, hi = st["spec"].t, st["spec"].t + st["spec"].dur
            if lo <= t < hi:
                active = active + st["spec"].force()
                if st["n"] == 0:
                    events.append({"t": round(t, 3), "event": "push_start",
                                   "label": st["spec"].label,
                                   "force": st["spec"].force().tolist()})
                st["n"] += 1
            elif t >= hi and st["n"] > 0 and not st.get("done"):
                st["done"] = True
                events.append({"t": round(t, 3), "event": "push_end",
                               "label": st["spec"].label})
        pen, sat = 0.0, 0.0
        for _ in range(sub):
            data.xfrc_applied[:] = 0.0                 # the trap: always zero first
            if active.any():
                data.xfrc_applied[push_body, :3] = active
            mujoco.mj_step(ids_model, data)
            n_sub_phys += 1
            if data.ncon:
                pen = min(pen, float(np.min(data.contact.dist)))
            f = np.abs(data.actuator_force)
            fr = ids_model.actuator_forcerange[:, 1]
            ok = fr > 0                      # 0 = unlimited in MuJoCo
            if ok.any():
                sat = max(sat, float(np.max(f[ok] / fr[ok])))
        # -- record ----------------------------------------------------------
        info = getattr(ctrl.law, "info", {}) if hasattr(ctrl, "law") else {}
        tr["t"].append(t)
        tr["qpos"].append(data.qpos.copy())
        tr["qvel"].append(data.qvel.copy())
        tr["ctrl"].append(data.ctrl.copy())
        tr["com"].append(ids.com(data))
        tr["margin"].append(float(info.get("margin", ids.com_margin(
            data, ids.foot_contact(data)))))
        tr["tilt_deg"].append(ids.torso_tilt_deg(data))
        tr["contact"].append(ids.foot_contact(data))
        tr["clearance"].append(ids.foot_clearance(data))
        tr["sole_pts"].append(np.array([ids.sole_points(data, s) for s in K.SIDES]))
        tr["foot_load"].append(ids.foot_load(data))
        tr["hands_local"].append(np.array(
            [[*ids.local_xy(data.site_xpos[ids.site[f"{s}_wrist"]][:2], data),
              float(data.site_xpos[ids.site[f"{s}_wrist"]][2])] for s in K.SIDES]))
        tr["head_z"].append(float(data.site_xpos[ids.site["head"]][2]))
        tr["knee_z"].append(np.array([float(data.site_xpos[ids.site[f"{s}_knee"]][2])
                                      for s in K.SIDES]))
        tr["pen_min"].append(float(pen))
        tr["sat_frac"].append(float(sat))
        tr["skill_id"].append(_SKILL_IX.get(cmd.skill, -1))
        tr["cmd"].append([cmd.vx, cmd.vy, cmd.wz, cmd.stance_height,
                          0.0 if cmd.lead_leg == "left" else 1.0])
        tr["push"].append(active.copy())
        tr["push_force"].append(float(np.linalg.norm(active)))
        tr["com_ref"].append(np.asarray(info.get("com_ref", ids.com_xy(data)), float))
        tr["e_track"].append(np.asarray(info.get("e_track", np.zeros(2)), float))
        tr["safety_alpha"].append(float(info.get("alpha", 0.0)))
        tr["ik_err"].append(float(max(getattr(ctrl.solver, "ik_err", {0: 0.0}).values()))
                            if hasattr(ctrl, "solver") else 0.0)
        tr["foot_xy"].append(np.array([ids.sole_center(data, s)[:2] for s in K.SIDES]))
        tr["plan_planted"].append([bool(ctrl.plan.feet[s].planted)
                                   for s in K.SIDES] if getattr(ctrl, "plan", None)
                                   else [True, True])
        for e in getattr(sched, "drain_events", lambda: [])():
            events.append({"t": round(t, 3), **e})
        if hasattr(ctrl, "events"):
            while len(ctrl.events) > getattr(ctrl, "_ev_published", 0):
                e = ctrl.events[getattr(ctrl, "_ev_published", 0)]
                events.append({"t": round(float(e.get("t", t)), 3), **e})
                ctrl._ev_published = getattr(ctrl, "_ev_published", 0) + 1
        # -- termination -----------------------------------------------------
        if det.update(ids, data):
            aborted = det.reason
            events.append({"t": round(t, 3), "event": "fall", "detail": det.reason})
            break
        if verbose and (k + 1) % 250 == 0:
            print(f"    t={t:5.1f}s z={data.qpos[2]:.3f} margin={tr['margin'][-1]:+.3f} "
                  f"skill={cmd.skill} alpha={tr['safety_alpha'][-1]:.2f}")
        if time.time() - t0 > cfg.max_wall_s:
            aborted = f"wall-clock cap {cfg.max_wall_s:.0f}s at t={t:.1f}s"
            break

    trace = {k: np.asarray(v) for k, v in tr.items()}
    trace["meta"] = np.array([{"physics_substeps": sub,
                               "model_timestep": float(ids_model.opt.timestep),
                               "physics_steps": n_sub_phys, "control_ticks": len(trace["t"]),
                               "one_reset": True, "controller": cfg.controller,
                               "rung": cfg.rung, "seed": cfg.seed,
                               "start": cfg.start}], dtype=object)
    metrics = metrics_mod.summarize(trace, events, cfg)
    res = RunResult(cfg, trace, events, metrics, time.time() - t0, aborted)
    prov = provenance(cfg)
    prov["sim_wall_s"] = round(res.wall_s, 1)
    res.provenance = prov
    return res


_SKILL_IX = {s: i for i, s in enumerate(
    ("STANCE", "ENTRY", "SHUFFLE_F", "SHUFFLE_B", "SHUFFLE_L", "SHUFFLE_R",
     "APPROACH", "RETREAT", "CIRCLE_L", "CIRCLE_R", "LEVEL_CHANGE",
     "SHOT_GESTURE", "RECOVER"))}


if __name__ == "__main__":                              # self-check
    cfg = RunConfig(controller="feasible", rung="L0", seconds=6.0, tag="smoke")
    res = run(cfg)
    print(json.dumps(res.metrics, indent=1)[:1200])
    print("saved:", res.save())
