#!/usr/bin/env python3
"""Measure the per-technique reference pose trims for the phase-3 teacher.

The retargeted references are schematic GrappleMap poses scaled onto the G1;
some of those poses are not static equilibria for this morphology (the STANCE
crouch needs 36 Nm of ankle torque of a +-50 Nm limit just to hold and topples
under any feedback — see the report).  This script searches a *small constant
joint-space trim* per technique and per robot that makes the posture holdable,
using the physics as the oracle:

    fitness(trim) = mean over seeds of stay_up_frac of the trimmed robot
    (the other robot keeps its own current trim; both run the full teacher)

Coordinate search over four sagittal parameters (hip_pitch, knee, ankle_pitch
applied to both legs, plus waist_pitch), two passes, step 0.1 rad, keeping a
candidate only when it improves the mean stay-up fraction.  The settled
pelvis-height correction ``dz`` of the winning trim is measured in the same
rollout and stored with it.

Output: data/teacher_trims.json (consumed by src/teacher/trims.py).
Run from the repo root:
    MUJOCO_GL=egl .venv/bin/python scripts/tune_teacher.py [TECH ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import mujoco  # noqa: E402

from retarget.scene import load_scene_model  # noqa: E402
import teacher.trims as trims  # noqa: E402
from teacher import TeacherController, run_episode  # noqa: E402
from teacher.controller import TeacherFlags  # noqa: E402

TECHNIQUES = ["DOUBLE_LEG", "SINGLE_LEG", "BODY_LOCK", "SNAPDOWN", "SPRAWL",
              "STAND_UP", "STANCE"]
REF_DIR = REPO / "data" / "refs"
OUT_PATH = REPO / "data" / "teacher_trims.json"
PARAMS = ("hip", "knee", "ankle", "waist")
STEP = 0.10          # rad per coordinate step
PASSES = 2
STAY_UP_GATE = 0.80  # acceptance gates the search optimises toward
SCORER_GATE = 0.85


def load_ref(tech):
    npz = np.load(REF_DIR / f"{tech}.npz", allow_pickle=True)
    return npz["qpos_a"], npz["qpos_b"], npz["t"]


def build(model, tech, qa, qb, t_ref):
    """Teacher with the *current* global trim table (mutated by the search)."""
    return TeacherController(model, qa, qb, t_ref, technique=tech,
                             flags=TeacherFlags())


def fitness(stay: float, scorer: float) -> tuple:
    """Gate-aware objective: progress toward BOTH gates, then the raw values.

    Both the stay-up gate (0.80) and the similarity gate (0.85) are hard
    acceptance criteria, so a candidate that fixes the failing one must win —
    a plain "stay-up first" rule happily trades the similarity gate away (it
    did: STANCE moved to scorer 0.60 at stay 1.0), and a plain "scorer first"
    rule trades the stay-up away.  Capping each term at its gate makes the two
    commensurate, and the raw pair breaks ties.
    """
    return (min(stay, STAY_UP_GATE) / STAY_UP_GATE
            + min(scorer, SCORER_GATE) / SCORER_GATE, stay, scorer)


def evaluate(model, tech, qa, qb, t_ref, seeds, trims_by_prefix):
    """Mean stay-up per robot + mean technique-scorer value of the traces.

    The scorer is the operator's similarity gate, so the search optimises
    lexicographically: first reach stay_up >= 0.80 (the acceptance gate),
    then maximise the scorer while staying there.
    """
    from teacher.episode import score_episode
    vals = {"a_": [], "b_": []}
    scores = []
    trims.POSE_TRIM[tech] = trims_by_prefix
    for seed in seeds:
        ctrl = build(model, tech, qa, qb, t_ref)
        res = run_episode(model, ctrl, qa, qb, t_ref, seed=seed)
        sf = res.stay_up_frac(t_ref)
        vals["a_"].append(sf["a"])
        vals["b_"].append(sf["b"])
        try:
            scores.append(score_episode(res, t_ref, tech, model)["mean"])
        except Exception:
            scores.append(float("nan"))
    return ({"a_": float(np.mean(vals["a_"])), "b_": float(np.mean(vals["b_"]))},
            float(np.nanmean(scores)))


def tune(model, tech, qa, qb, t_ref, seeds):
    """Coordinate-search trims for both robots of one technique."""
    table = {p: dict(trims.POSE_TRIM.get(tech, {}).get(p, {})) for p in ("a_", "b_")}
    best, best_score = evaluate(model, tech, qa, qb, t_ref, seeds, table)
    print(f"[{tech}] start stay={({p: round(v, 3) for p, v in best.items()})} "
          f"scorer={best_score:.3f} trim={table}", flush=True)
    for pass_i in range(PASSES):
        for prefix in ("a_", "b_"):
            for param in PARAMS:
                improved = True
                while improved:
                    improved = False
                    for delta in (STEP, -STEP):
                        cand = {p: dict(s) for p, s in table.items()}
                        cand[prefix][param] = round(cand[prefix].get(param, 0.0) + delta, 3)
                        vals, sc = evaluate(model, tech, qa, qb, t_ref, seeds, cand)
                        key_new = fitness(vals[prefix], sc)
                        key_old = fitness(best[prefix], best_score)
                        if key_new > key_old:
                            table, best, best_score = cand, vals, sc
                            improved = True
                            print(f"[{tech}] pass{pass_i} {prefix} {param}{delta:+.2f} "
                                  f"-> stay={({p: round(v, 3) for p, v in best.items()})} "
                                  f"scorer={best_score:.3f}", flush=True)
                            break
    # settled height correction: mean pelvis z of the winning rollout minus ref
    trims.POSE_TRIM[tech] = table
    dz = {}
    for prefix in ("a_", "b_"):
        ctrl = build(model, tech, qa, qb, t_ref)
        res = run_episode(model, ctrl, qa, qb, t_ref, seed=seeds[0])
        m = (res.rows[:, 0] >= 0.5 * t_ref[-1]) & (res.rows[:, 0] <= t_ref[-1])
        z_sim = float(res.rows[m, 1 if prefix == "a_" else 2].mean()) if m.any() else np.nan
        z_ref = float(res.ref[:, 0 if prefix == "a_" else 1].mean())
        dz[prefix] = round(z_sim - z_ref, 3) if np.isfinite(z_sim) else 0.0
    for prefix in ("a_", "b_"):
        if abs(dz[prefix]) > 0.002:
            table[prefix]["dz"] = dz[prefix]
    sc = {}
    for prefix in ("a_", "b_"):
        ctrl = build(model, tech, qa, qb, t_ref)
        res = run_episode(model, ctrl, qa, qb, t_ref, seed=seeds[0])
        try:
            from teacher.episode import score_episode
            sc[prefix] = round(score_episode(res, t_ref, tech, model)["mean"], 3)
        except Exception as exc:                       # pragma: no cover
            sc[prefix] = str(exc)
    print(f"[{tech}] final trims {table} dz={dz} stay={best} scorer={sc}",
          flush=True)
    return table, best, dz, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("techniques", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    techs = args.techniques or TECHNIQUES
    model = load_scene_model()
    seeds = list(range(args.seeds))
    out = {"step_rad": STEP, "passes": PASSES, "seeds": seeds,
           "trims": {}, "stay_up": {}, "scorer": {}}
    if OUT_PATH.exists():            # partial runs merge, never clobber
        try:
            out = json.loads(OUT_PATH.read_text())
            out.update({"step_rad": STEP, "passes": PASSES, "seeds": seeds})
        except Exception:
            pass
    t0 = time.time()
    for tech in techs:
        qa, qb, t_ref = load_ref(tech)
        table, best, dz, sc = tune(model, tech, qa, qb, t_ref, seeds)
        out["trims"][tech] = table
        out["stay_up"][tech] = best
        out.setdefault("scorer", {})[tech] = sc
    OUT_PATH.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"wrote {OUT_PATH} ({time.time()-t0:.0f}s)")
    print(json.dumps(out["trims"], indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
