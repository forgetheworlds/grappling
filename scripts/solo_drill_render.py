#!/usr/bin/env python3
"""Solo drill: run, verify and render the continuous-drill evidence.

Usage (from the repo root, with ``.venv`` active)::

    # 1) simulate (one reset, no resets inside the run) + metrics JSON
    python scripts/solo_drill_render.py run --rung L1 --seconds 90 --start stance \
        --out-tag L1_90

    # 2) render a cached trajectory at the acceptance format (no re-simulation)
    python scripts/solo_drill_render.py render --npz data/drill/L1_90_feasible_L1_seed0.npz \
        --out videos/solo_drill/final_continuous_drill.mp4 --title "continuous solo drill"

    # 3) everything the milestone asks for, in order, with the suite check
    python scripts/solo_drill_render.py suite --headline data/drill/L1_90_feasible_L1_seed0.npz

The physics is never re-run by ``render``: frames come from the saved trace, so
a clip and its metrics JSON always describe the same episode.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

VIDEO = REPO / "videos" / "solo_drill"
DATA = REPO / "data" / "drill"


def _meta_for(npz: Path, label: str, title: str, footer: str = "") -> dict:
    """HUD meta: controller kind, seed, rung, element phases, push labels."""
    from drill import video as video_mod
    trace = video_mod.load_trace(npz)
    js = npz.with_suffix(".json")
    cfg, events = {}, []
    if js.exists():
        blob = json.loads(js.read_text())
        cfg, events = blob.get("config", {}), blob.get("events", [])
    phases = {}
    for ev in events:
        if ev.get("event") == "element_done":
            t1 = float(ev["t"])
            t0 = t1 - float(ev.get("elapsed") or 0.0)
            for k in range(int(t0 * 10), int(t1 * 10) + 1):
                phases[round(k / 10.0, 1)] = ev.get("phase", "")
    pushes = [e for e in events if e.get("event") == "push_start"]
    kind = {"feasible": "scripted feedback (FeasibleDrill)",
            "stance_pd": "scripted, NO feedback (PD baseline)",
            "teacher": "teacher feedback (adapter)"}.get(cfg.get("controller", ""), "")
    return {"title": title, "controller_kind": kind, "seed": cfg.get("seed", 0),
            "rung": cfg.get("rung", ""), "phases": phases,
            "push_labels": ",".join(p.get("label", "") for p in pushes),
            "footer": footer or f"config: {npz.name} (physics staged from this trace)"}


def cmd_run(a) -> int:
    from drill.lock import sim_lock
    from drill.runner import PushSpec, RunConfig, run
    pushes = [PushSpec(t=float(t), dur=0.12, fx=float(fx), label=f"push{fx:+.0f}N")
              for t, fx in (a.push or [])]
    cfg = RunConfig(controller=a.controller, rung=a.rung, seconds=a.seconds,
                    seed=a.seed, start=a.start, pushes=pushes, tag=a.out_tag)
    with sim_lock(f"run {cfg.key()}") if a.seconds > 60 else _null():
        res = run(cfg, verbose=True)
    paths = res.save(DATA)
    print(json.dumps({"paths": paths, "aborted": res.aborted,
                      "falls": res.metrics["falls"],
                      "longest_s": res.metrics["longest_continuous_s"],
                      "steps": res.metrics["steps_completed"],
                      "elements": res.metrics["elements_done"]}, indent=1))
    return 1 if res.metrics["falls"] else 0


def cmd_render(a) -> int:
    from drill import video as video_mod
    npz = Path(a.npz)
    out = Path(a.out)
    meta = _meta_for(npz, a.label, a.title, a.footer)
    sheet = tuple(float(v) for v in a.sheet) if a.sheet else tuple(
        npz and [])
    r = video_mod.render_trace(video_mod.load_trace(npz), out, meta=meta,
                               t0=a.t0, t1=a.t1, speed=a.speed, scale=a.scale,
                               sheet_times=sheet or (a.t0 + 0.4,
                                                     (a.t0 + (a.t1 or 5)) / 2,
                                                     (a.t1 or 5) - 0.2),
                               caption=a.caption, label=a.label)
    print(json.dumps(r, indent=1))
    return 0


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def cmd_report(a) -> int:
    from drill import rubric as rubric_mod
    js = Path(a.json)
    blob = json.loads(js.read_text())
    trace_path = js.with_suffix(".npz")
    r = rubric_mod.assess(trace_path, json.loads(trace_path.with_suffix(".json").read_text()))
    out = DATA / f"rubric_{js.stem}.json"
    out.write_text(json.dumps(r, indent=1))
    print(rubric_mod.render_table(r))
    print(f"\nwrote {out}")
    return 0


def cmd_suite(a) -> int:
    """Run + render the whole deliverable set (one heavy process at a time)."""
    import numpy as np
    from drill.lock import sim_lock
    from drill.runner import PushSpec, RunConfig, run
    from drill import video as video_mod

    tag = "FINAL"
    jobs = [
        # (tag, RunConfig, renders)
        (f"{tag}_L1_90", dict(controller="feasible", rung="L1", seconds=90.0,
                              start="stance", tag=f"{tag}_L1_90"), [
            ("final_continuous_drill.mp4", "continuous solo drill (headline)",
             0.0, None, ("the headline clip: one unbroken episode, no resets, "
                         "rung L1 (stance hold + weight shift + level change)")),
        ]),
        (f"{tag}_L1_push", dict(controller="feasible", rung="L1", seconds=90.0,
                                start="stance", tag=f"{tag}_L1_push",
                                pushes=[PushSpec(t=15.0, dur=0.12, fx=-20.0, label="push-20N"),
                                        PushSpec(t=38.0, dur=0.12, fy=20.0, label="push+20N"),
                                        PushSpec(t=62.0, dur=0.12, fx=20.0, label="push+20N")]), [
            ("final_disturbances.mp4", "continuous solo drill with pushes",
             0.0, 60.0, "3 x 20 N pushes (all recovered): CoM, margin and the push windows are overlaid"),
        ]),
        (f"{tag}_L0_60", dict(controller="feasible", rung="L0", seconds=60.0,
                              start="stance", tag=f"{tag}_L0_60"), [
            ("L0_hold_30s.mp4", "rung L0: stance hold + posture modulation",
             0.0, 30.0, "both feet planted throughout: the rung below the ship candidate"),
        ]),
        ("BASE_PD", dict(controller="stance_pd", rung="L1", seconds=12.0,
                         start="stance", tag="BASE_PD"), [
            ("01_baseline_stancepd.mp4", "baseline: StancePD (no feedback)",
             0.0, None, "the same stance pose under pure position control: it topples"),
        ]),
        ("FAIL_PUSH90", dict(controller="feasible", rung="L1", seconds=25.0,
                             start="stance", tag="FAIL_PUSH90",
                             pushes=[PushSpec(t=8.0, dur=0.12, fx=-90.0, label="push-90N")]), [
            ("99_failure_push90N.mp4", "FAILURE: 90 N push",
             6.0, 13.0, "above the measured recovery limit: 20 N recovers, 35 N is marginal, 90 N topples"),
        ]),
        ("FAIL_ENTRY_L2", dict(controller="feasible", rung="L2", seconds=30.0,
                               start="stand", tag="FAIL_ENTRY_L2"), [
            ("99_failure_entry_L2.mp4", "FAILURE: stand -> stance entry (L2 stepping)",
             1.0, 12.0, "the entry walk: steps complete in isolation but the sequence is not clean yet"),
        ]),
        ("SLOWMO_L1", dict(controller="feasible", rung="L1", seconds=20.0,
                           start="stance", tag="SLOWMO_L1"), [
            ("02_slowmo_level_change_quarter_speed.mp4",
             "slow motion 0.25x: level change down / hold / rise", 3.0, 6.0,
             "0.25x of the level change: descent, hold, rise (rubric H3)"),
        ]),
    ]
    results = {}
    with sim_lock("drill suite (runs + renders)"):
        for run_tag, cfg_kw, renders in jobs:
            cfg = RunConfig(**cfg_kw)
            res = run(cfg, verbose=True)
            paths = res.save(DATA)
            results[run_tag] = {"paths": paths, "falls": res.metrics["falls"],
                                "duration_s": res.metrics["duration_s"],
                                "elements": res.metrics["elements_done"],
                                "steps": res.metrics["steps_completed"]}
            print(f"[suite] {run_tag}: falls={res.metrics['falls']} "
                  f"steps={res.metrics['steps_completed']}", flush=True)
            for name, title, t0, t1, caption in renders:
                npz = Path(paths["npz"])
                meta = _meta_for(npz, name.split("_")[0], title)
                meta["footer"] = (f"config {npz.name} | one reset, no in-run resets | "
                                  f"{caption}")
                r = video_mod.render_trace(video_mod.load_trace(npz), VIDEO / name,
                                           meta=meta, t0=t0, t1=t1,
                                           sheet_times=tuple(np.linspace(
                                               t0 + 0.5, float(
                                                   (t1 if t1 else np.load(npz)["t"][-1]))
                                               - 0.5, 3)),
                                           caption=caption)
                print(f"[suite] rendered {r['mp4']} ({r['frames']} frames)", flush=True)
                if name == "final_continuous_drill.mp4" and not cfg.pushes:
                    # the headline episode IS the nominal one (no pushes were
                    # injected): the second required name is a copy, not a
                    # second render of the same 90 s
                    import shutil
                    shutil.copy(VIDEO / name, VIDEO / "final_nominal.mp4")
                    shutil.copy(VIDEO / name.replace(".mp4", "_sheet.png"),
                                VIDEO / "final_nominal_sheet.png")
                    print("[suite] final_nominal.mp4 = copy of the headline (same episode)",
                          flush=True)
    (DATA / "suite_summary.json").write_text(json.dumps(results, indent=1, default=str))
    print(json.dumps(results, indent=1, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="simulate one continuous episode")
    r.add_argument("--controller", default="feasible")
    r.add_argument("--rung", default="L1")
    r.add_argument("--seconds", type=float, default=90.0)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--start", default="stance", choices=("stand", "stance"))
    r.add_argument("--push", action="append", nargs=2, metavar=("T", "FX"))
    r.add_argument("--out-tag", default="drill")
    r.set_defaults(func=cmd_run)

    d = sub.add_parser("render", help="render a cached trajectory")
    d.add_argument("--npz", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--title", default="continuous solo drill")
    d.add_argument("--label", default="")
    d.add_argument("--footer", default="")
    d.add_argument("--t0", type=float, default=0.0)
    d.add_argument("--t1", type=float, default=None)
    d.add_argument("--speed", type=float, default=1.0)
    d.add_argument("--scale", type=float, default=1.0)
    d.add_argument("--sheet", nargs="*")
    d.add_argument("--caption", default="")
    d.set_defaults(func=cmd_render)

    p = sub.add_parser("report", help="rubric self-assessment for a run")
    p.add_argument("--json", required=True)
    p.set_defaults(func=cmd_report)

    s_ = sub.add_parser("suite", help="run + render the whole deliverable set")
    s_.set_defaults(func=cmd_suite)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
