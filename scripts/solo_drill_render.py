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


def verify_clip(path: Path, expect_s: float, expect_frames: int,
                fps: int = 30) -> dict:
    """ffprobe duration/frame count + a non-black/non-static frame check.

    A clip is only evidence if it decodes, lasts what the metrics say, and its
    pixels actually change (a frozen or black render is not a video).
    """
    import subprocess
    import numpy as np
    import imageio.v2 as imageio
    out = {"path": str(path), "ok": True, "problems": []}
    try:
        pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                             "-show_entries", "stream=nb_frames,duration,codec_name,"
                             "pix_fmt,width,height,r_frame_rate", "-of", "json",
                             str(path)], capture_output=True, text=True, timeout=120)
        st = json.loads(pr.stdout)["streams"][0]
        out.update({k: st.get(k) for k in ("codec_name", "pix_fmt", "width", "height",
                                           "r_frame_rate", "nb_frames", "duration")})
        if st.get("codec_name") != "h264":
            out["problems"].append(f"codec {st.get('codec_name')} != h264")
        if st.get("pix_fmt") != "yuv420p":
            out["problems"].append(f"pix_fmt {st.get('pix_fmt')} != yuv420p")
        dur = float(st.get("duration") or 0.0)
        if abs(dur - expect_s) > 0.5:
            out["problems"].append(f"duration {dur:.2f}s vs expected {expect_s:.2f}s")
        nf = int(st.get("nb_frames") or 0)
        if abs(nf - expect_frames) > 3:
            out["problems"].append(f"frames {nf} vs expected {expect_frames}")
    except Exception as exc:                              # pragma: no cover
        out["problems"].append(f"ffprobe failed: {exc}")
    try:
        rd = imageio.get_reader(str(path))
        meta = rd.get_meta_data()
        n = int(meta.get("nframes") or expect_frames)
        idx = [int(n * f) for f in (0.1, 0.4, 0.7, 0.95) if 0 <= int(n * f) < n]
        imgs = [np.asarray(rd.get_data(i), dtype=float) for i in idx]
        rd.close()
        luma = [float(im.mean()) for im in imgs]
        out["mean_luma"] = [round(v, 1) for v in luma]
        if min(luma) < 8:
            out["problems"].append(f"near-black frame (min luma {min(luma):.1f})")
        diffs = [float(np.abs(imgs[i + 1] - imgs[i]).mean())
                 for i in range(len(imgs) - 1)]
        out["frame_diff"] = [round(v, 3) for v in diffs]
        if diffs and max(diffs) < 0.5:
            out["problems"].append(f"static video (max diff {max(diffs):.3f})")
    except Exception as exc:                              # pragma: no cover
        out["problems"].append(f"decode failed: {exc}")
    out["ok"] = not out["problems"]
    return out


def cmd_suite(a) -> int:
    """Run + render the whole deliverable set (one heavy process at a time)."""
    import numpy as np
    from drill.lock import sim_lock
    from drill.runner import PushSpec, RunConfig, run
    from drill import rubric as rubric_mod
    from drill import video as video_mod

    tag = "FINAL"
    jobs = [
        # (tag, RunConfig, renders)
        (f"{tag}_L1_90", dict(controller="feasible", rung="L1", seconds=90.0,
                              start="stance", tag=f"{tag}_L1_90"), [
            ("final_L1_90s.mp4", "rung L1 - stance hold + weight shift + level change (90 s)",
             0.0, None, ("the rung-L1 clip: one unbroken episode, no resets, "
                         "stance hold + weight shift + level change"), 1.0),
        ]),
        ("BASE_PD", dict(controller="stance_pd", rung="L1", seconds=12.0,
                         start="stance", tag="BASE_PD"), [
            ("01_baseline_stancepd.mp4", "baseline: StancePD (no feedback)",
             0.0, None, "the same stance pose under pure position control: it topples", 1.0),
        ]),
        ("FAIL_ENTRY_L2", dict(controller="feasible", rung="L2", seconds=30.0,
                               start="stand", tag="FAIL_ENTRY_L2"), [
            ("99_failure_entry_L2.mp4", "FAILURE: stand -> stance entry (L2 stepping)",
             1.0, 12.0, "the entry walk: steps complete in isolation but the sequence is not clean yet", 1.0),
        ]),
        (f"{tag}_L1_push", dict(controller="feasible", rung="L1", seconds=90.0,
                                start="stance", tag=f"{tag}_L1_push",
                                pushes=[PushSpec(t=15.0, dur=0.12, fx=-20.0, label="push-20N"),
                                        PushSpec(t=38.0, dur=0.12, fy=20.0, label="push+20N"),
                                        PushSpec(t=62.0, dur=0.12, fx=20.0, label="push+20N")]), [
            ("L1_90s_with_pushes.mp4", "rung L1 with pushes (3 x 20 N, all recovered)",
             0.0, 60.0, "3 x 20 N pushes (all recovered): CoM, margin and the push windows are overlaid", 1.0),
        ]),
        (f"{tag}_L0_60", dict(controller="feasible", rung="L0", seconds=60.0,
                              start="stance", tag=f"{tag}_L0_60"), [
            ("L0_hold_30s.mp4", "rung L0: stance hold + posture modulation",
             0.0, 30.0, "both feet planted throughout: the rung below the ship candidate", 0.5),
        ]),
        ("FAIL_PUSH90", dict(controller="feasible", rung="L1", seconds=25.0,
                             start="stance", tag="FAIL_PUSH90",
                             pushes=[PushSpec(t=8.0, dur=0.12, fx=-90.0, label="push-90N")]), [
            ("99_failure_push90N.mp4", "FAILURE: 90 N push",
             6.0, 13.0, "above the measured recovery limit: 20 N recovers, 35 N is marginal, 90 N topples", 0.5),
        ]),
        ("SLOWMO_L1", dict(controller="feasible", rung="L1", seconds=20.0,
                           start="stance", tag="SLOWMO_L1"), [
            ("02_slowmo_level_change_quarter_speed.mp4",
             "slow motion 0.25x: level change down / hold / rise", 3.0, 6.0,
             "0.25x of the level change: descent, hold, rise (rubric H3)", 1.0),
        ]),
    ]
    results = {}
    # 1) SIMULATION pass: takes the heavy-process lock, one run at a time, and
    #    releases it as soon as the traces are saved (~5 min total).  Rendering
    #    never blocks another agent's simulation: it only reads cached traces.
    with sim_lock("drill suite: 6 short runs"):
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
    # 2) RENDER pass: LOCK-FREE (no physics is stepped here).  Priority order:
    #    the rung clip first, then the baseline and one failure clip, then the
    #    rest -- the last group at diagnostic scale, named *_diag.
    for run_tag, cfg_kw, renders in jobs:
        cfg = RunConfig(**cfg_kw)
        paths = results[run_tag]["paths"]
        for name, title, t0, t1, caption, scale in renders:
            if scale < 1.0:
                name = name.replace(".mp4", "_diag.mp4")
            npz = Path(paths["npz"])
            meta = _meta_for(npz, name.split("_")[0], title)
            meta["footer"] = (f"config {npz.name} | one reset, no in-run resets | {caption}")
            r = video_mod.render_trace(video_mod.load_trace(npz), VIDEO / name,
                                       meta=meta, t0=t0, t1=t1, scale=scale,
                                       sheet_times=tuple(np.linspace(
                                           t0 + 0.5, float(
                                               (t1 if t1 else np.load(npz)["t"][-1])) - 0.5, 3)),
                                       caption=caption)
            ver = verify_clip(Path(r["mp4"]), expect_s=(t1 or float(
                np.load(npz)["t"][-1])) - t0, expect_frames=r["frames"])
            print(f"[suite] rendered {r['mp4']} ({r['frames']} frames) verify={ver}",
                  flush=True)
            results[run_tag].setdefault("renders", []).append({"name": name, "verify": ver})
            run_blob = json.loads((Path(paths["json"])).read_text())
            rub = rubric_mod.assess(paths["npz"], run_blob)
            bundle = {"video": str(VIDEO / name),
                      "contact_sheet": r.get("sheet"), "trace_npz": paths["npz"],
                      "run_json": paths["json"], "config": run_blob.get("config"),
                      "provenance": run_blob.get("provenance"),
                      "metrics": run_blob.get("metrics"), "rubric": rub,
                      "rubric_table": rubric_mod.render_table(rub),
                      "label": title, "caption": caption,
                      "clip_verification": ver,
                      "reproduce": run_blob.get("provenance", {}).get("reproduce"),
                      "render_command": (f"MUJOCO_GL=egl python scripts/solo_drill_render.py "
                                         f"render --npz {paths['npz']} --out videos/solo_drill/{name}")}
            bdir = REPO / "data" / "solo_drill"
            bdir.mkdir(parents=True, exist_ok=True)
            (bdir / (Path(name).stem + ".json")).write_text(
                json.dumps(bundle, indent=1, default=str))
            if name == "final_L1_90s.mp4" and not cfg.pushes:
                print("[suite] rung clip named final_L1_90s.mp4; the acceptance name "
                      "final_continuous_drill.mp4 is intentionally NOT created "
                      "(no full-drill clip exists)", flush=True)
    (DATA / "suite_summary.json").write_text(json.dumps(results, indent=1, default=str))
    print(json.dumps(results, indent=1, default=str))
    return 0


# --------------------------------------------------------------- L2 evidence
#: the base the stepping rungs are demonstrated in.  The drill stance's own
#: width is *not* steppable on this robot: measured, lifting a foot in a
#: 0.495 m stance needs ~0.21-0.25 m of lateral CoM travel against a measured
#: authority of ~0.14 m (see reports/2026-10-08/drill_l2.md).  The deviation is
#: reported with every clip; the drill stance is unchanged and still the L1
#: hold base.
def _stepping_base(model, ids):
    from drill import posture, stepping as stepping_mod

    return posture.build_stance(model, stepping_mod.stepping_base_spec(), ids)


def _l1_plus_step_scheduler(seed: int = 0, timeout: float = 14.0):
    """The L1 programme with ONE step embedded per cycle (the L2 gate)."""
    import numpy as np
    from drill import scheduler as sch

    rng = np.random.default_rng(seed)
    sched = sch.SkillScheduler("L2", seed=seed)
    sched.elements = sch.program("L1", rng) + [
        sch.Element("x_step", "SHUFFLE_F", steps=1, timeout=timeout, guard="flat",
                    params={"vx": 0.07}),
        sch.Element("x_hold", "STANCE", hold_s=0.6, timeout=6.0, guard="stance")]
    return sched


def cmd_l2suite(a) -> int:
    """Run + render the L2 (stepping) evidence set: one heap-free pass.

    Stages (reports/2026-10-08/drill_l2.md): the isolated first step from the
    stand, the entry walk, one step per L1 cycle, a 2-3 step shuffle -- each in
    the base the primitive can actually use -- plus the *refusal* evidence in
    the drill stance itself (the geometry that names why).
    """
    import numpy as np

    from drill import posture, scene
    from drill import stepping as stepping_mod
    from drill import rubric as rubric_mod
    from drill import video as video_mod
    from drill.lock import sim_lock
    from drill.runner import RunConfig, run

    DATA.mkdir(parents=True, exist_ok=True)
    (REPO / "data" / "solo_drill").mkdir(parents=True, exist_ok=True)
    model = scene.load_model()
    ids = scene_mod_ids(model)
    nbase = _stepping_base(model, ids)
    spec = posture.build_stance(model, posture.StanceSpec(), ids)
    jobs = [
        # (tag, RunConfig, stance, scheduler, renders)
        ("L2_ENTRY_SPEC", dict(controller="feasible", rung="L2", seconds=14.0,
                               start="stand", tag="L2_ENTRY_SPEC"), spec, None, [
            ("L2_isolated_step.mp4", "L2: the first step out of the stand (drill stance base)",
             0.4, 7.0, "one isolated step: margin-gated lift, world-tracked swing, "
             "load-gated landing; the rest of the entry is refused on geometry", 1.0)]),
        ("L2_ENTRY_BASE", dict(controller="feasible", rung="L2", seconds=25.0,
                               start="stand", tag="L2_ENTRY_BASE"), nbase, None, [
            ("final_L2_entry_walk.mp4", "L2: stand -> stepping base, walked",
             0.4, 16.0, "the entry walk in the base the primitive can use "
             "(0.25 m wide, 0.06 m deep)", 1.0)]),
        ("L2_CYCLE", dict(controller="feasible", rung="L2", seconds=34.0,
                          start="stance", tag="L2_CYCLE"), nbase,
         _l1_plus_step_scheduler(0), [
            ("L2_cycle_step_diag.mp4", "L2: one step per L1 programme cycle",
             8.0, 18.0, "level change / rise with one gate-checked step per cycle", 0.5)]),
        ("L2_SHUFFLE", dict(controller="feasible", rung="L2", seconds=34.0,
                            start="stance", tag="L2_SHUFFLE"), nbase, None, [
            ("L2_shuffle_3steps_diag.mp4", "L2: short shuffle (2-3 steps)",
             4.0, 18.0, "consecutive single-foot repositions with the settle between", 0.5)]),
        ("L2_STANCE_REFUSED", dict(controller="feasible", rung="L2", seconds=10.0,
                                   start="stance", tag="L2_STANCE_REFUSED"), spec, None, [
            ("L2_stance_step_refused_diag.mp4",
             "L2: the drill stance refuses the step (geometry)", 1.0, 8.0,
             "the same primitive in the 0.495 m stance: every lift needs more CoM "
             "travel than the robot has; it refuses instead of toppling", 0.5)]),
    ]
    results = {}
    with sim_lock("L2 suite"):
        for tag, cfg_kw, stance, sched, _renders in jobs:
            cfg = RunConfig(**cfg_kw)
            res = run(cfg, stance=stance, model=model, scheduler=sched, verbose=False)
            paths = res.save(DATA)
            m = res.metrics
            results[tag] = {
                "paths": paths, "falls": m["falls"], "longest_s": m["longest_continuous_s"],
                "steps": m["steps_completed"], "elements": m["elements_done"],
                "timeouts": m["timeouts"],
                "slip_m": round(float(m["slip"]["max_load_drift_m"]), 4),
                "margin_min": round(float(np.min(res.trace["margin"])), 4),
                "ik_max": round(float(m["ik_err_max"]), 4),
                "refusals": len([e for e in res.events
                                 if e.get("event") == "step_refused"]),
                "required_com_travel": sorted({round(float(e["required_com_travel_m"]), 3)
                                               for e in res.events
                                               if e.get("event") == "step_refused"}),
            }
            print(f"[L2] {tag}: falls {m['falls']} steps {m['steps_completed']} "
                  f"margin_min {results[tag]['margin_min']:+.4f} "
                  f"refusals {results[tag]['refusals']}", flush=True)
    # render pass (lock-free: cached traces only)
    for tag, cfg_kw, stance, sched, renders in jobs:
        cfg = RunConfig(**cfg_kw)
        paths = results[tag]["paths"]
        for name, title, t0, t1, caption, scale in renders:
            npz = Path(paths["npz"])
            meta = _meta_for(npz, name.split("_")[0], title)
            meta["footer"] = f"config {npz.name} | one reset | {caption}"
            r = video_mod.render_trace(video_mod.load_trace(npz), VIDEO / name,
                                       meta=meta, t0=t0, t1=t1, scale=scale,
                                       sheet_times=tuple(np.linspace(t0 + 0.4, t1 - 0.4, 3)),
                                       caption=caption)
            ver = verify_clip(Path(r["mp4"]), expect_s=t1 - t0, expect_frames=r["frames"])
            blob = json.loads(Path(paths["json"]).read_text())
            rub = rubric_mod.assess(paths["npz"], blob)
            bundle = {"video": str(VIDEO / name), "contact_sheet": r.get("sheet"),
                      "trace_npz": paths["npz"], "run_json": paths["json"],
                      "config": blob.get("config"), "provenance": blob.get("provenance"),
                      "metrics": blob.get("metrics"), "rubric": rub,
                      "rubric_table": rubric_mod.render_table(rub),
                      "label": title, "caption": caption,
                      "clip_verification": ver,
                      "l2_evidence_notes": results[tag],
                      "reproduce": (f"MUJOCO_GL=egl python scripts/solo_drill_render.py "
                                    f"run --controller feasible --rung {cfg.rung} "
                                    f"--seconds {cfg.seconds:g} --start {cfg.start} "
                                    f"--out-tag {cfg.tag}")}
            (REPO / "data" / "solo_drill" / (Path(name).stem + ".json")).write_text(
                json.dumps(bundle, indent=1, default=str))
            print(f"[L2] rendered {r['mp4']} verify={ver['ok']} "
                  f"problems={ver['problems']}", flush=True)
    (DATA / "l2_suite_summary.json").write_text(json.dumps(results, indent=1, default=str))
    print(json.dumps(results, indent=1, default=str))
    return 0


def scene_mod_ids(model):
    from drill import kin as K

    return K.RobotIds.build(model)


# ------------------------------------------------------- reference tracking
def cmd_tracks(a) -> int:
    """Run every retargeted reference track and write the per-track report."""
    import numpy as np

    from drill import kin as K, posture, scene
    from drill import rubric as rubric_mod
    from drill import video as video_mod
    from drill.lock import sim_lock
    from drill.runner import RunConfig, run
    from drill.tracking import TRACK_ORDER, Track

    DATA.mkdir(parents=True, exist_ok=True)
    (REPO / "data" / "solo_drill").mkdir(parents=True, exist_ok=True)
    model = scene.load_model()
    ids = K.RobotIds.build(model)
    stance = posture.build_stance(model, posture.StanceSpec(), ids)
    names = [n for n in TRACK_ORDER if (REPO / "data" / "refs_video" / f"{n}.npz").exists()]
    summary = {}
    with sim_lock("track suite"):
        for name in names:
            trk = Track.load(name)
            cfg = RunConfig(controller="track", rung="L3", track=name,
                            seconds=trk.duration + 1.0, seed=0, start="track",
                            tag=f"TRACK_{name}")
            res = run(cfg, stance=stance, model=model, verbose=False,
                      scheduler=_TrackScheduler(name))
            paths = res.save(DATA)
            m = res.metrics
            qpos = np.asarray(res.trace["qpos"], float)
            apeak = float(max(np.abs(qpos[:, ids.leg_qadr[s][5]]).max() for s in K.SIDES))
            summary[name] = {
                "paths": paths,
                "duration_s": round(trk.duration, 2),
                "falls": m["falls"], "fall_t": m["fall_times"],
                "longest_s": round(float(m["longest_continuous_s"]), 2),
                "margin_min": round(float(np.min(res.trace["margin"])), 4),
                "ankle_roll_peak": round(apeak, 4),
                "ankle_roll_limit": round(float(ids.leg_limits["left"][5, 1]), 4),
                "slip_m": round(float(m["slip"]["max_load_drift_m"]), 4),
                "ik_max": round(float(m["ik_err_max"]), 4),
                "rubric": rubric_mod.assess(paths["npz"], json.loads(
                    Path(paths["json"]).read_text())),
                "tracking": m.get("reference_tracking"),
            }
            t = summary[name]["tracking"] or {}
            print(f"[TRK] {name}: falls {m['falls']} longest "
                  f"{summary[name]['longest_s']}s tracked {t.get('tracked_frac')} "
                  f"e_joint {t.get('e_joint_mean_rad')} e_site {t.get('e_site_mean_m')} "
                  f"clamp {t.get('clamp_shift_max_m')} "
                  f"margin_min {summary[name]['margin_min']:+.3f}", flush=True)
    # clips for the tracks that ran longest (diagnostic scale: render budget)
    order = sorted(names, key=lambda n: -summary[n]["longest_s"])[:3]
    for name in order:
        paths = summary[name]["paths"]
        npz = Path(paths["npz"])
        trk = Track.load(name)
        t_end = max(1.5, summary[name]["longest_s"] + 0.4)
        meta = _meta_for(npz, "TRK", f"reference tracking: {name}")
        meta["footer"] = (f"track {name} (retargeted, not dynamically validated) | "
                          f"cached trace {npz.name}")
        r = video_mod.render_trace(video_mod.load_trace(npz),
                                   VIDEO / f"L3_track_{name}_diag.mp4", meta=meta,
                                   t0=0.0, t1=t_end, scale=0.5,
                                   sheet_times=(0.2, t_end / 2, max(0.3, t_end - 0.2)),
                                   caption=("reference trajectory tracking: the plan is "
                                            "driven by the retargeted track; partial "
                                            "tracking is expected and reported"))
        ver = verify_clip(Path(r["mp4"]), expect_s=t_end, expect_frames=r["frames"])
        blob = json.loads(Path(paths["json"]).read_text())
        bundle = {"video": str(VIDEO / f"L3_track_{name}_diag.mp4"),
                  "contact_sheet": r.get("sheet"), "trace_npz": paths["npz"],
                  "run_json": paths["json"], "config": blob.get("config"),
                  "provenance": blob.get("provenance"), "metrics": blob.get("metrics"),
                  "reference_tracking": blob["metrics"].get("reference_tracking"),
                  "clip_verification": ver,
                  "reference_note": ("retargeted from monocular video (mediapipe); "
                                     "not dynamically validated -- tracking is "
                                     "reported as partial"),
                  "reproduce": (f"MUJOCO_GL=egl python scripts/solo_drill_render.py "
                                f"run --controller track --track {name} "
                                f"--seconds {trk.duration + 1.0:g} --start track "
                                f"--out-tag TRACK_{name}")}
        (REPO / "data" / "solo_drill" / f"L3_track_{name}.json").write_text(
            json.dumps(bundle, indent=1, default=str))
        print(f"[TRK] rendered {r['mp4']} verify={ver['ok']}", flush=True)
    (DATA / "tracks_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "paths"}
                      for k, v in summary.items()}, indent=1, default=str))
    return 0


class _TrackScheduler:
    """One TRACK command for the whole episode (the mode's own driver)."""

    def __init__(self, name: str):
        self.name = name

    def reset(self, model, data) -> None:
        pass

    def tick(self, ids, data, ctrl, t):
        from drill.controller import DrillCommand

        return DrillCommand("TRACK", phase=f"track {self.name}")

    def drain_events(self):
        return []


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

    l2 = sub.add_parser("l2suite", help="run + render the L2 stepping evidence set")
    l2.set_defaults(func=cmd_l2suite)

    tk = sub.add_parser("tracks", help="run every retargeted reference track")
    tk.set_defaults(func=cmd_tracks)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
