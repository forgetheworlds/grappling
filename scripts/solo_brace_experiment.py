#!/usr/bin/env python
"""Scripted brace experiment: which pushes does a small step recover that a
static stance does not?  (Capturability, operator question 2026-10-08.)

Three strategies on the drill scene (single G1, 50 Hz control / 500 Hz physics):

  static      ``stance_pd``        pure position hold, no feedback, no step
  ankle/hip   ``feasible`` rung L1 balance law ON, stepping not allowed
  brace step  ``feasible`` rung L2 balance law ON + ``FootStepper``

For every impulse in ``IMPULSES`` (N*s, applied as a +x body force over
``PUSH_DUR`` s at t=1.0 s from the stand keyframe, one reset per run, no
re-entry) the script reports:

  fall?   final pelvis/tilt/margin   max CP excursion (m)   max required brace
  step (m)   achieved foot-centre travel (m)

Definitions used here (all from recorded trace columns):

* capture point ``cp = com_xy + v_xy * sqrt(z_c/g)`` -- ``v`` is a 5-tick
  central difference of the recorded CoM path (the trace carries no CoM
  velocity column; the drill's ``margin`` column is the CoM margin, not the CP);
* required brace step = ``solo.stability.brace_step_length(com, cp, hull)`` on
  the hull of the *touching* feet's sole points at that tick;
* "fall" = the runner's own ``FallDetector`` fired (pelvis < 0.42 m, tilt > 55
  deg or dorsal, sustained 0.16 s);
* "recovered" = survived AND ends inside the stance predicate (pelvis >= 0.55,
  tilt <= 20 deg, both feet touching, CoM margin > 0).

Run:  MUJOCO_GL=egl .venv/bin/python scripts/solo_brace_experiment.py [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from drill import kin as K  # noqa: E402
from drill import posture as posture_mod  # noqa: E402
from drill import runner as runner_mod  # noqa: E402
from drill import scene as scene_mod  # noqa: E402
from solo.stability import MIN_CP_HEIGHT, brace_step_length, hull2d  # noqa: E402

#: impulses tested (N*s) and the push duration used to realise them (s)
IMPULSES: tuple[float, ...] = (5.0, 8.0, 12.0, 16.0, 20.0, 25.0)
PUSH_DUR = 0.12
PUSH_T = 1.0
SECONDS = 6.0
#: strategies: (label, controller, rung)
STRATEGIES: tuple[tuple[str, str, str], ...] = (
    ("no-feedback", "stance_pd", "L0"),
    ("static", "feasible", "L0"),
    ("ankle/hip", "feasible", "L1"),
    ("brace step", "feasible", "L2"),
)


def capture_point_series(trace: dict) -> np.ndarray:
    """(T, 2) capture point from the recorded CoM path (central difference)."""
    com = np.asarray(trace["com"], float)
    dt = 0.02
    v = np.zeros_like(com[:, :2])
    v[2:-2] = (com[4:, :2] - com[:-4, :2]) / (4.0 * dt)
    v[1] = (com[2, :2] - com[0, :2]) / (2.0 * dt)
    v[-2] = (com[-1, :2] - com[-3, :2]) / (2.0 * dt)
    z = np.maximum(com[:, 2], MIN_CP_HEIGHT)
    return com[:, :2] + v * np.sqrt(z / 9.81)[:, None]


def hull_series(trace: dict) -> list[np.ndarray]:
    """Per-tick support hull of the touching feet's sole points (n, 2)."""
    out = []
    for pts, touch in zip(np.asarray(trace["sole_pts"], float),
                          np.asarray(trace["contact"], bool)):
        sel = [pts[i][:, :2] for i in range(2) if touch[i]]
        out.append(hull2d(np.vstack(sel)) if sel else np.zeros((0, 2)))
    return out


def analyse(res: runner_mod.RunResult) -> dict:
    trace = res.trace
    com = np.asarray(trace["com"], float)
    cp = capture_point_series(trace)
    hulls = hull_series(trace)
    brace = np.array([brace_step_length(com[i, :2], cp[i], hulls[i])
                      if len(hulls[i]) >= 3 else np.nan
                      for i in range(len(com))])
    foot_xy = np.asarray(trace["foot_xy"], float)
    travel = float(np.max(np.linalg.norm(foot_xy - foot_xy[0], axis=2)))
    margin = np.asarray(trace["margin"], float)
    tilt = np.asarray(trace["tilt_deg"], float)
    contact = np.asarray(trace["contact"], bool)
    z = np.asarray(trace["qpos"], float)[:, 2]
    t = np.asarray(trace["t"], float)
    after = t >= PUSH_T                     # everything from the push onward
    fall = bool(res.aborted is not None and str(res.aborted).startswith("fall"))
    died_before_push = bool(fall and not after.any())
    win = np.where(after)[0]
    if win.size == 0:                       # fell before the push arrived
        win = np.arange(len(t))
    last = -1
    end_ok = bool(
        not fall and z[last] >= 0.55 and tilt[last] <= 20.0
        and contact[last].all() and margin[last] > 0.0)
    return {
        "fall": fall,
        "died_before_push": died_before_push,
        "aborted": res.aborted,
        "end_ok": end_ok,
        "pelvis_end_m": round(float(z[last]), 3),
        "tilt_end_deg": round(float(tilt[last]), 1),
        "margin_end_m": round(float(margin[last]), 4),
        "cp_excursion_max_m": round(float(np.nanmax(
            np.linalg.norm(cp[win] - com[win, :2], axis=1))), 4),
        "cp_margin_min_m": round(float(np.nanmin(
            [np.nan if len(hulls[i]) < 3 else _signed(cp[i], hulls[i])
             for i in win])), 4),
        "brace_step_max_m": round(float(np.nanmax(brace[win])), 4),
        "foot_travel_max_m": round(travel, 4),
        "steps_taken": int(sum(1 for e in res.events if e.get("event") == "step_done")),
    }


def _signed(p, hull) -> float:
    """Signed distance from p to the hull boundary (+, inside) -- local copy."""
    d = math.inf
    h = np.asarray(hull, float)
    for a, b in zip(h, np.roll(h, -1, axis=0)):
        e = b - a
        L = float(np.linalg.norm(e))
        if L < 1e-12:
            continue
        d = min(d, (e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])) / L)
    return float(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--impulses", default=None, help="comma list (N*s)")
    args = ap.parse_args()
    impulses = ([float(x) for x in args.impulses.split(",")] if args.impulses
                else list(IMPULSES))

    model = scene_mod.load_model()
    ids = K.RobotIds.build(model)
    stance = posture_mod.build_stance(model, posture_mod.StanceSpec(), ids)

    rows = []
    for label, controller, rung in STRATEGIES:
        for imp in impulses:
            force = float(imp) / PUSH_DUR
            cfg = runner_mod.RunConfig(
                controller=controller, rung=rung, seconds=SECONDS, seed=0,
                start="stand", tag=f"brace_{label.replace(' ', '')}_{imp:g}",
                pushes=[runner_mod.PushSpec(t=PUSH_T, dur=PUSH_DUR, fx=force,
                                            label=f"m{imp:g}")])
            res = runner_mod.run(cfg, stance=stance, model=model, verbose=False)
            r = analyse(res)
            r.update({"strategy": label, "impulse_ns": float(imp),
                      "wall_s": round(res.wall_s, 1)})
            rows.append(r)
            print(f"[{label:10s} {imp:5.1f} N*s] fall={str(r['fall']):5s} "
                  f"end_ok={str(r['end_ok']):5s} pelvis={r['pelvis_end_m']:.3f} "
                  f"tilt={r['tilt_end_deg']:5.1f} margin={r['margin_end_m']:+.3f} "
                  f"cpExc={r['cp_excursion_max_m']:.3f} brace={r['brace_step_max_m']:.3f} "
                  f"travel={r['foot_travel_max_m']:.3f} steps={r['steps_taken']}")

    # ---- summary table + self-check ------------------------------------
    print("\n=== recovery vs impulse (fall = the runner's detector fired) ===")
    hdr = f"{'impulse':>8s} " + " ".join(f"{s[0]:>10s}" for s in STRATEGIES)
    print(hdr)
    for imp in impulses:
        cells = []
        for label, _, _ in STRATEGIES:
            row = next(r for r in rows if r["strategy"] == label
                       and r["impulse_ns"] == float(imp))
            cells.append("FALL" if row["fall"] else
                         ("ok" if row["end_ok"] else "survived"))
        print(f"{imp:8.1f} " + " ".join(f"{c:>10s}" for c in cells))

    static = [r for r in rows if r["strategy"] == "static"]
    brace = [r for r in rows if r["strategy"] == "brace step"]
    ankle = [r for r in rows if r["strategy"] == "ankle/hip"]
    assert len(rows) == len(STRATEGIES) * len(impulses)
    # structural checks (do not assert an outcome we have not measured)
    assert all(r["brace_step_max_m"] >= 0.0 for r in rows)
    assert any(r["fall"] for r in rows), "no arm ever fell: check the setup"
    hardest = max(impulses)
    st_h = next(r for r in static if r["impulse_ns"] == hardest)
    br_h = next(r for r in brace if r["impulse_ns"] == hardest)
    an_h = next(r for r in ankle if r["impulse_ns"] == hardest)
    print(f"\nself-check: static falls at {hardest:g} N*s: {st_h['fall']}; "
          f"ankle/hip: {an_h['fall']}; brace step: {br_h['fall']} "
          f"(foot travel {br_h['foot_travel_max_m']:.3f} m, "
          f"{br_h['steps_taken']} step(s))")
    assert st_h["fall"] or an_h["fall"] or br_h["fall"], "no arm fell at the top impulse"

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=1))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
