"""Per-step and aggregated metrics for the solo drill.

Metrics (per control step, all computed from ``mjData``; no simulator-private
state leaks into observations):

===================  ==========================================================
field                definition
===================  ==========================================================
``t``                sim time (s)
``vel_err``          ||(v_xy_local - cmd.(vx,vy))|| (m/s)
``vx_err``           v_xy_local.x - cmd.vx (signed, m/s)
``vy_err``           v_xy_local.y - cmd.vy (signed, m/s)
``yaw_err``          |wz_local - cmd.wz| (rad/s)
``upright``          torso_link local +z . world +z in [-1, 1]
``tilt_deg``         arccos(upright) in degrees
``pelvis_z``         pelvis height (m)
``stance_err``       pelvis_z - cmd.stance_height (m)
``slip``             max horizontal foot-site speed among *loaded* feet (m/s)
``slip_travel``      sum over *loaded* feet of |xy displacement this step| (m)
``body_step``        |pelvis xy displacement this step| (m)
``cmd_vx``/``cmd_vy``/``cmd_wz``  the filtered command in force this step
``contact_l/r``      floor contact of each foot (0/1)
``knee_contact``     knee mat contact (0/1; diagnostics, never terminal)
``hand_contact``     hand/wrist mat contact (0/1)
``torso_contact``    torso/pelvis mat contact (0/1)
``dorsal_contact``   dorsal torso/pelvis contact (solo.detector feature, 0/1)
``act_delta``        mean |ctrl_t - ctrl_{t-1}| over 29 actuators (rad)
``sat_frac``         fraction of actuators at >= 95% of the joint force limit
``limit_prox``       max over joints of clip(1 - margin/0.1, 0, 1)
``reward``           total task reward this step
===================  ==========================================================

Aggregation adds episode-level numbers (``fall``, ``dorsal``, ``n_steps``,
``fall_rate`` over a run, ``recovery_time_s`` after a push).  Traces are written
as JSONL (one row per control step) and a summary JSON is emitted next to them;
``data/solo/metrics/`` is the canonical output directory.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .scene import STEP_DT

METRICS_DIR = Path(__file__).resolve().parents[2] / "data" / "solo" / "metrics"

#: per-step fields written to JSONL (order fixed for readability)
METRIC_FIELDS: tuple[str, ...] = (
    "t", "vel_err", "vx_err", "vy_err", "yaw_err", "upright", "tilt_deg",
    "pelvis_z", "stance_err",
    "slip", "slip_travel", "body_step", "cmd_vx", "cmd_vy", "cmd_wz",
    "speed", "com_offset", "steps_taken",
    "contact_l", "contact_r", "knee_contact", "hand_contact",
    "torso_contact", "dorsal_contact", "act_delta", "sat_frac", "limit_prox",
    "hand_err", "reward",
)

#: joints whose force limit is used for the saturation metric
_SAT_LIMIT_FRACTION = 0.95


def _round(x, n=6):
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    v = float(x)
    return None if math.isnan(v) else round(v, n)


# --------------------------------------------------------------- per-step math
def local_xy(vel_world: np.ndarray, yaw: float) -> np.ndarray:
    """World xy velocity -> robot heading frame (vx forward, vy left)."""
    c, s = math.cos(yaw), math.sin(yaw)
    v = np.asarray(vel_world, dtype=np.float64)
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])


def foot_slip(model: mujoco.MjModel, data: mujoco.MjData,
              foot_sites: tuple[int, int], contacts: tuple[bool, bool]) -> float:
    """Max horizontal speed (m/s) of the foot sites that are in floor contact."""
    out = 0.0
    vel = np.zeros(6)
    for sid, loaded in zip(foot_sites, contacts):
        if not loaded:
            continue
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, int(sid),
                                 vel, 0)
        out = max(out, float(math.hypot(vel[3], vel[4])))
    return out


def action_smoothness(prev: np.ndarray | None, cur: np.ndarray) -> float:
    """Mean |ctrl_t - ctrl_{t-1}| over the 29 actuators (0.0 on the first step)."""
    if prev is None:
        return 0.0
    return float(np.abs(np.asarray(cur, np.float64)
                        - np.asarray(prev, np.float64)).mean())


def saturation_fraction(model: mujoco.MjModel, data: mujoco.MjData,
                        fraction: float = _SAT_LIMIT_FRACTION) -> float:
    """Fraction of the 29 actuators whose force is at >= ``fraction`` of the
    joint actuator-force limit (``jnt_actfrcrange``; measured, not assumed)."""
    lim = np.asarray(model.jnt_actfrcrange[1:30], dtype=np.float64)
    lim = np.maximum(np.abs(lim[:, 0]), np.abs(lim[:, 1]))
    force = np.abs(np.asarray(data.actuator_force, dtype=np.float64))[:29]
    sat = force >= fraction * lim
    return float(np.mean(sat))


def joint_limit_proximity(model: mujoco.MjModel, data: mujoco.MjData,
                          margin: float = 0.1) -> tuple[float, int]:
    """(max proximity in [0,1], count of joints closer than ``margin``) using
    hinge positions vs the model joint ranges."""
    q = np.asarray(data.qpos[7:36], dtype=np.float64)
    lo = np.asarray(model.jnt_range[1:30, 0], dtype=np.float64)
    hi = np.asarray(model.jnt_range[1:30, 1], dtype=np.float64)
    m = np.minimum(q - lo, hi - q)
    prox = np.clip(1.0 - m / margin, 0.0, 1.0)
    return float(prox.max()), int(np.sum(m < margin))


def recovery_time(rows: list[dict], push_end_t: float, *, upright_min: float = 0.97,
                  pelvis_tol: float = 0.06, pelvis_nominal: float = 0.79,
                  hold_s: float = 0.2, vel_max: float | None = 0.15) -> float | None:
    """Seconds from ``push_end_t`` until the robot is *stable*: upright, at
    stance height and (when ``vel_max`` is given) with base speed below it,
    held for ``hold_s`` continuously (None if it never stabilises)."""
    need = max(1, int(round(hold_s / STEP_DT)))
    run = 0
    for row in rows:
        t = float(row.get("t", 0.0))
        if t < push_end_t:
            continue
        ok = (float(row.get("upright", 0.0)) >= upright_min
              and abs(float(row.get("pelvis_z", 0.0)) - pelvis_nominal) <= pelvis_tol)
        if ok and vel_max is not None:
            ok = float(row.get("speed", 0.0)) <= float(vel_max)
        run = run + 1 if ok else 0
        if run >= need:
            return float(t - push_end_t - (need - 1) * STEP_DT)
    return None


def com_offset_max(rows: list[dict], start_t: float) -> float | None:
    """Max CoM-to-support-centre offset (m) at/after ``start_t``.

    Honest proxy: ``||CoM_xy - mean(loaded foot sites)_xy||`` in the pelvis
    frame -- *not* a convex-hull support margin (no hull is implemented yet).
    """
    vals = [float(r["com_offset"]) for r in rows
            if float(r.get("t", 0.0)) >= start_t and r.get("com_offset") is not None]
    return max(vals) if vals else None


# ----------------------------------------------------------------- aggregation
def _stat(values: list[float]) -> dict:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return {"mean": None, "p95": None, "min": None, "max": None, "n": 0}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "mean": round(float(a.mean()), 6),
        "p95": round(float(np.percentile(a, 95)), 6),
        "min": round(float(a.min()), 6),
        "max": round(float(a.max()), 6),
        "n": int(a.size),
    }


def summarize_rows(rows: list[dict], fields: tuple[str, ...] = METRIC_FIELDS) -> dict:
    """Mean/p95/min/max per metric + the term means from ``row["terms"]``."""
    out = {f: _stat([r.get(f) for r in rows]) for f in fields}
    term_names: set[str] = set()
    for r in rows:
        term_names.update((r.get("terms") or {}).keys())
    terms = {name: _stat([(r.get("terms") or {}).get(name) for r in rows])
             for name in sorted(term_names)}
    return {"n_steps": len(rows), "metrics": out, "terms": terms}


def write_jsonl(path: str | Path, rows: list[dict]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    return p


def write_json(path: str | Path, payload: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return p


@dataclass
class MetricsRecorder:
    """Per-episode recorder: ``add`` each control step, then write/summarise."""

    episode: int = 0
    task: str = ""
    controller: str = ""
    seed: int = 0
    meta: dict = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)

    def add(self, **fields) -> dict:
        row: dict = {"episode": int(self.episode), "step": len(self.rows)}
        for f in METRIC_FIELDS:
            row[f] = _round(fields.get(f))
        terms = fields.get("terms")
        if terms:
            row["terms"] = {k: _round(v) for k, v in terms.items()}
        self.rows.append(row)
        return row

    def to_jsonl(self, path: str | Path) -> Path:
        return write_jsonl(path, self.rows)

    def summary(self, extra: dict | None = None) -> dict:
        s = {
            "episode": int(self.episode),
            "task": self.task,
            "controller": self.controller,
            "seed": int(self.seed),
            "meta": dict(self.meta),
        }
        s.update(summarize_rows(self.rows))
        if extra:
            s.update(extra)
        return s


if __name__ == "__main__":  # self-check
    rows = [{"t": i * STEP_DT, "upright": 0.999, "pelvis_z": 0.79, "terms": {"upright": 0.5}} for i in range(50)]
    s = summarize_rows(rows)
    assert s["n_steps"] == 50 and s["metrics"]["upright"]["mean"] == 0.999
    assert s["terms"]["upright"]["mean"] == 0.5
    assert recovery_time(rows, 0.0) is not None
    assert recovery_time([{**r, "upright": 0.1} for r in rows], 0.0) is None
    rec = MetricsRecorder(episode=7, task="balance", controller="zero")
    for r in rows:
        rec.add(**r)
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = rec.to_jsonl(Path(td) / "x.jsonl")
        assert sum(1 for _ in p.open()) == 50
    print("solo.metrics self-check OK:", {"steps": s["n_steps"],
                                          "recovery": recovery_time(rows, 0.0)})
