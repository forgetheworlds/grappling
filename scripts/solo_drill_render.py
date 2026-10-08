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

FPS = 30


def _final_name(name: str, scale: float) -> str:
    """The artifact name a render spec produces.

    Diagnostic-scale renders (``scale < 1.0``) carry a ``_diag`` suffix so a
    reader cannot mistake one for a full-scale artifact.  This is the ONLY
    place the suffix is applied: bundles and the index both name the file this
    returns, so they cannot disagree (the 2026-10-08 audit found two index
    paths naming the pre-rename file because the rename happened after the
    table had been hand-copied).
    """
    if scale < 1.0 and not name.endswith("_diag.mp4"):
        return name.replace(".mp4", "_diag.mp4")
    return name


#: Every clip the pipeline renders and the exact window/speed it renders:
#: SINGLE SOURCE OF TRUTH for what each artifact must contain.
#: ``scripts/evidence_check.py`` recomputes each artifact's expected duration
#: and frame count from this table plus the trace *independently of the
#: renderer*, so a hand-typed claim (or a post-hoc rename) that drifts from
#: the artifact is caught mechanically.  ``speed < 1.0`` slows the clip:
#: rendered frames = round((min(t1, trace_end) - t0) * fps / speed) + 1.
CLIP_SPECS: dict[str, dict] = {
    # --- deliverable suite (cmd_suite) -----------------------------------
    "final_L1_90s.mp4": dict(
        title="rung L1 - stance hold + weight shift + level change (90 s)",
        t0=0.0, t1=None, scale=1.0, speed=1.0,
        caption="the rung-L1 clip: one unbroken episode, no resets, "
                "stance hold + weight shift + level change"),
    "01_baseline_stancepd.mp4": dict(
        title="baseline: StancePD (no feedback) - 12 s hold, NO fall",
        t0=0.0, t1=None, scale=1.0, speed=1.0,
        # CAPTION CORRECTED 2026-10-08 (claim-audit F1): this job renders the
        # 12 s BASE_PD_stance_pd_L1_seed0 trace, which does NOT fall.  The old
        # caption ("it topples") was false here; the run that topples is the
        # separate BASELINE_PD_stance_pd_L1_seed0, wired to its own job below.
        caption="the same stance pose under pure position control: stays up for "
                "12 s (does NOT topple - the toppling baseline is the separate "
                "run rendered as 01_baseline_pd_topples.mp4)"),
    "01_baseline_pd_topples.mp4": dict(
        title="baseline: StancePD (no feedback) - TOPPLES",
        t0=0.0, t1=None, scale=1.0, speed=1.0,
        caption="baseline: StancePD (no feedback) - the SAME stance pose as the "
                "feasible drill; with no feedback the robot TOPPLES at 8.4 s "
                "(fall detector fires; pelvis z 0.736 -> 0.299 m)"),
    "99_failure_entry_L2.mp4": dict(
        title="FAILURE: stand -> stance entry (L2 stepping)",
        t0=1.0, t1=12.0, scale=1.0, speed=1.0,
        caption="FAILURE: the entry walk falls at t=2.82 s - the clip ends at "
                "the episode's own end (not a truncated render); the isolated "
                "step completes, the sequence does not"),
    "L1_90s_with_pushes.mp4": dict(
        title="rung L1 with pushes (t=15 s and t=38 s recovered; t=62 s outside this clip)",
        t0=0.0, t1=60.0, scale=1.0, speed=1.0,
        # CLAIM CORRECTED 2026-10-08 (claim-audit mode D): the third push is at
        # t=62 s, outside this deliberate t=0-60 s window, so the clip can only
        # show two of the three pushes.  The claim now says exactly that; the
        # run's metrics record all 3/3 recovered.
        caption="push windows overlaid: two of the three 20 N pushes (t=15 s, "
                "t=38 s) are inside this 0-60 s clip and recover; the third "
                "(t=62 s) is outside it - the run's metrics record 3/3"),
    "L0_hold_30s.mp4": dict(
        title="rung L0: stance hold + posture modulation (t=0-30 s of the 60 s run)",
        t0=0.0, t1=30.0, scale=0.5, speed=1.0,
        caption="both feet planted throughout: the rung below the ship candidate"),
    "99_failure_push90N.mp4": dict(
        title="FAILURE: 90 N push",
        t0=6.0, t1=13.0, scale=0.5, speed=1.0,
        caption="above the measured recovery limit: 20 N recovers, 35 N is "
                "marginal, 90 N topples"),
    "02_slowmo_level_change_quarter_speed.mp4": dict(
        # WINDOW + SPEED CORRECTED 2026-10-08 (claim-audit modes C+D): the old
        # spec window (3-6 s) held only the level-change HOLD (descent ends at
        # t~1.8 s, rise starts at t~6 s) and no speed was passed, so the clip
        # rendered at 1.0x.  The full level-change element (crouch 0-6.02 s +
        # rise to 8.06 s, from the run's element_done events) is now rendered at
        # the 0.25x its name claims: 8.06 s / 0.25 = 32.2 s of video.
        title="slow motion 0.25x: level change down / hold / rise",
        t0=0.0, t1=8.06, scale=1.0, speed=0.25,
        caption="0.25x of the level-change element (t=0-8.06 s: descent to "
                "~1.8 s, hold, rise to ~8 s) - rubric H3"),
    # --- L2 stepping suite (cmd_l2suite) ---------------------------------
    "L2_isolated_step.mp4": dict(
        title="L2: the first step out of the stand (drill stance base)",
        t0=0.4, t1=7.0, scale=1.0, speed=1.0,
        caption="one isolated step: margin-gated lift, world-tracked swing, "
                "load-gated landing; the rest of the entry is refused on geometry"),
    "final_L2_entry_walk.mp4": dict(
        title="L2: stand -> stepping base, walked",
        t0=0.4, t1=16.0, scale=1.0, speed=1.0,
        # WIDTH CORRECTED 2026-10-08: the stepping base is half_width=0.105 m
        # (0.21 m wide, sole separation 0.209-0.215 m in the traces), not the
        # 0.25 m the old caption claimed (drill/stepping.stepping_base_spec).
        caption="the entry walk in the base the primitive can use "
                "(0.21 m wide, 0.06 m deep)"),
    "L2_cycle_step_diag.mp4": dict(
        title="L2: one step per L1 programme cycle",
        t0=8.0, t1=18.0, scale=0.5, speed=1.0,
        caption="level change / rise with one gate-checked step per cycle "
                "(the run falls at t=22.56 s, outside this 8-18 s window)"),
    "L2_shuffle_3steps_diag.mp4": dict(
        # STEP CLAIM CORRECTED 2026-10-08: the run completes 4 steps, not the
        # "2-3" the old label claimed (metrics.steps_completed).
        title="L2: short shuffle (4 consecutive steps)",
        t0=4.0, t1=18.0, scale=0.5, speed=1.0,
        caption="consecutive single-foot repositions with the settle between"),
    "L2_stance_step_refused_diag.mp4": dict(
        title="L2: the drill stance refuses the step (geometry)",
        t0=1.0, t1=8.0, scale=0.5, speed=1.0,
        caption="the same primitive in the 0.495 m stance: every lift needs more "
                "CoM travel than the robot has; it refuses instead of toppling"),
}


def clip_spec(name: str) -> dict:
    """The render spec for ``name`` (a copy, with ``name`` attached)."""
    try:
        return {"name": name, **CLIP_SPECS[name]}
    except KeyError:
        raise SystemExit(f"no clip spec for {name!r} - add it to CLIP_SPECS")


def _trace_end(npz: Path | str) -> float:
    import numpy as np

    with np.load(npz, allow_pickle=True) as z:
        return float(np.asarray(z["t"], float)[-1])


def window_expectations(npz: Path | str, t0: float = 0.0,
                        t1: float | None = None, speed: float = 1.0,
                        fps: int = FPS) -> dict:
    """What a clip rendered from ``npz`` over ``[t0, t1]`` at ``speed`` MUST measure.

    Mirrors ``drill.video.sample_frames``: the render truncates at the trace's
    own end (an episode that falls at 2.82 s cannot produce 11 s of video), and
    ``speed < 1`` interpolates ``1/speed`` more frames over the same physics
    window while the writer still plays them at ``fps``, so the clip's duration
    is ``(min(t1, trace_end) - t0) / speed``.  These are computed from the
    trace + the render window alone, never read back from the artifact.
    """
    t_end = _trace_end(npz)
    t1_raw = None if t1 is None else float(t1)
    t1_eff = t_end if t1_raw is None else min(t1_raw, t_end)
    rate = max(1, int(round(fps / max(1e-6, float(speed)))))
    n = max(2, int(round((t1_eff - t0) * rate)) + 1)
    return {"t0": float(t0), "t1": t1_raw, "t1_effective": round(t1_eff, 4),
            "trace_end_s": round(t_end, 4), "speed": float(speed),
            "fps": int(fps), "expect_frames": n,
            "expect_s": round(n / float(fps), 3)}


def _render_command(npz: Path | str, name: str, cspec: dict) -> str:
    """The exact command that reproduces a clip (window/speed INCLUDED).

    The old bundles recorded a command without the window, which would
    reproduce a different artifact than the one shipped (claim-audit mode D).
    """
    parts = ["MUJOCO_GL=egl python scripts/solo_drill_render.py render",
             f"--npz {npz}", f"--out videos/solo_drill/{name}",
             f"--t0 {cspec['t0']:g}"]
    if cspec["t1"] is not None:
        parts.append(f"--t1 {cspec['t1']:g}")
    parts += [f"--speed {cspec['speed']:g}", f"--scale {cspec['scale']:g}",
              f"--title {cspec['title']!r}", f"--caption {cspec['caption']!r}"]
    return " ".join(parts)


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
    import numpy as np

    from drill import video as video_mod
    npz = Path(a.npz)
    out = Path(a.out)
    exp = window_expectations(npz, a.t0, a.t1, a.speed)
    meta = _meta_for(npz, a.label, a.title, a.footer)
    sheet = tuple(float(v) for v in a.sheet) if a.sheet else tuple(
        np.linspace(a.t0 + 0.4, float(exp["t1_effective"]) - 0.4, 3))
    target = out
    if a.safe:
        # SAFE WRITE (orchestrator protocol): a killed render must never leave
        # a partial file occupying the artifact name
        target = out.with_name(out.stem + ".partial" + out.suffix)
    r = video_mod.render_trace(video_mod.load_trace(npz), target, meta=meta,
                               t0=a.t0, t1=a.t1, speed=a.speed, scale=a.scale,
                               sheet_times=sheet,
                               caption=a.caption, label=a.label)
    ver = verify_clip(Path(r["mp4"]), expect_s=exp["expect_s"],
                      expect_frames=exp["expect_frames"], speed=a.speed)
    print(json.dumps({**r, "verify": ver}, indent=1))
    if not ver["ok"]:
        print(f"[render] EVIDENCE GATE FAILED for {r['mp4']}: "
              f"{json.dumps(ver['problems'])}", flush=True)
        return 1
    if a.safe:
        Path(r["mp4"]).replace(out)
        if r.get("sheet"):
            Path(r["sheet"]).replace(out.with_name(out.stem + "_sheet.png"))
        print(f"[render] SAFE WRITE OK: verified then renamed -> {out}", flush=True)
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


def verify_clip(path: Path, expect_s: float | None = None,
                expect_frames: int | None = None, fps: int = 30,
                speed: float = 1.0) -> dict:
    """ffprobe facts, format, window match, and a non-black/non-static frame check.

    A clip is only evidence if it decodes, is the delivered format
    (h264 / yuv420p / ``fps``), matches the window it was rendered from
    (duration and frame count -- pass ``None`` only when no trace exists, e.g.
    an unbundled clip gets the format and pixel checks alone), and its pixels
    actually change (a frozen or black render is not a video).

    ``expect_s``/``expect_frames`` come from the trace + render window via
    :func:`window_expectations`; they are never read back from ``path``, so a
    truncated or mis-windowed render fails.  A verifier-side exception is
    reported separately (``error``) instead of being conflated with a media
    problem -- the 2026-10-08 audit found 12/13 bundles recording
    "decode failed: cannot convert float infinity to integer", which was an
    ``int(inf)`` bug in THIS function (imageio reports ``nframes: inf`` for
    ffmpeg-written mp4s; fixed in 103dcee), not a defect in any clip.
    """
    import subprocess
    import numpy as np
    import imageio.v2 as imageio

    out = {"path": str(path), "ok": True, "problems": [],
           "expect": {"seconds": expect_s, "frames": expect_frames, "fps": fps,
                      "speed": speed}}
    probed_frames = None
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
        rate, fps_act = str(st.get("r_frame_rate") or ""), None
        try:
            num, _, den = rate.partition("/")
            fps_act = float(num) / float(den or 1)
        except ValueError:
            out["problems"].append(f"unreadable frame rate {rate!r}")
        if fps_act is not None and abs(fps_act - fps) > 0.1:
            # the writer-fps bug class: a clip written at fps/speed plays back
            # at 1.0x no matter what the name claims (fixed 2026-10-08)
            out["problems"].append(f"frame rate {fps_act:g} != {fps}")
        dur = float(st.get("duration") or 0.0)
        if expect_s is not None and abs(dur - expect_s) > 0.5:
            out["problems"].append(f"duration {dur:.3f}s vs expected {expect_s:.3f}s")
        nb = st.get("nb_frames")
        try:
            probed_frames = int(nb)
        except (TypeError, ValueError):
            probed_frames = None
        if (expect_frames is not None and probed_frames is not None
                and abs(probed_frames - expect_frames) > 3):
            out["problems"].append(f"frames {probed_frames} vs expected {expect_frames}")
    except Exception as exc:
        out["error"] = {"where": "ffprobe", "type": type(exc).__name__,
                        "message": str(exc)}
        out["problems"].append(f"ffprobe failed ({type(exc).__name__}): {exc}")
    try:
        rd = imageio.get_reader(str(path))
        meta = rd.get_meta_data()
        n = None
        nf = meta.get("nframes")
        if nf is not None:
            try:
                n = int(nf) if np.isfinite(float(nf)) else None
            except (TypeError, ValueError, OverflowError):
                n = None
        if not n:
            n = probed_frames or expect_frames
        if not n:
            md = meta.get("duration")
            n = int(float(md) * fps) if md not in (None, "") else 0
        n = int(n or 0)
        if n >= 2:
            idx = sorted({min(n - 1, max(0, int(n * f)))
                          for f in (0.1, 0.4, 0.7, 0.95)})
            imgs = [np.asarray(rd.get_data(i), dtype=float) for i in idx]
        else:
            imgs = [np.asarray(rd.get_data(0), dtype=float)]
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
    except Exception as exc:
        out["error"] = {"where": "decode", "type": type(exc).__name__,
                        "message": str(exc)}
        out["problems"].append(f"decode failed ({type(exc).__name__}): {exc}")
    out["ok"] = not out["problems"]
    out["measured"] = {
        "duration_s": out.get("duration"), "frames": probed_frames,
        "fps": out.get("r_frame_rate"), "size": f"{out.get('width')}x{out.get('height')}",
        "mean_luma_min": min(out.get("mean_luma") or [None]) if out.get("mean_luma") else None,
        "frame_diff_max": max(out.get("frame_diff") or [None]) if out.get("frame_diff") else None,
    }
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
        # (tag, RunConfig, renders); each render is a CLIP_SPECS entry -- the
        # single source of truth for window/speed/title/caption
        (f"{tag}_L1_90", dict(controller="feasible", rung="L1", seconds=90.0,
                              start="stance", tag=f"{tag}_L1_90"), [
            clip_spec("final_L1_90s.mp4"),
        ]),
        ("BASE_PD", dict(controller="stance_pd", rung="L1", seconds=12.0,
                         start="stance", tag="BASE_PD"), [
            clip_spec("01_baseline_stancepd.mp4"),
        ]),
        # The topple contrast runs on its own tag: wiring it to BASE_PD was the
        # F1 hand-entry error (the two tags differ only by "BASE_").
        ("BASELINE_PD", dict(controller="stance_pd", rung="L1", seconds=10.0,
                             start="stance", tag="BASELINE_PD"), [
            clip_spec("01_baseline_pd_topples.mp4"),
        ]),
        ("FAIL_ENTRY_L2", dict(controller="feasible", rung="L2", seconds=30.0,
                               start="stand", tag="FAIL_ENTRY_L2"), [
            clip_spec("99_failure_entry_L2.mp4"),
        ]),
        (f"{tag}_L1_push", dict(controller="feasible", rung="L1", seconds=90.0,
                                start="stance", tag=f"{tag}_L1_push",
                                pushes=[PushSpec(t=15.0, dur=0.12, fx=-20.0, label="push-20N"),
                                        PushSpec(t=38.0, dur=0.12, fy=20.0, label="push+20N"),
                                        PushSpec(t=62.0, dur=0.12, fx=20.0, label="push+20N")]), [
            clip_spec("L1_90s_with_pushes.mp4"),
        ]),
        (f"{tag}_L0_60", dict(controller="feasible", rung="L0", seconds=60.0,
                              start="stance", tag=f"{tag}_L0_60"), [
            clip_spec("L0_hold_30s.mp4"),
        ]),
        ("FAIL_PUSH90", dict(controller="feasible", rung="L1", seconds=25.0,
                             start="stance", tag="FAIL_PUSH90",
                             pushes=[PushSpec(t=8.0, dur=0.12, fx=-90.0, label="push-90N")]), [
            clip_spec("99_failure_push90N.mp4"),
        ]),
        ("SLOWMO_L1", dict(controller="feasible", rung="L1", seconds=20.0,
                           start="stance", tag="SLOWMO_L1"), [
            clip_spec("02_slowmo_level_change_quarter_speed.mp4"),
        ]),
    ]
    results = {}
    # 1) SIMULATION pass: takes the heavy-process lock, one run at a time, and
    #    releases it as soon as the traces are saved (~5 min total).  Rendering
    #    never blocks another agent's simulation: it only reads cached traces.
    with sim_lock("drill suite: 7 short runs"):
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
    #    rest -- the last group at diagnostic scale, named *_diag by _final_name.
    gate_failures = []
    for run_tag, cfg_kw, renders in jobs:
        cfg = RunConfig(**cfg_kw)
        paths = results[run_tag]["paths"]
        for cspec in renders:
            name = _final_name(cspec["name"], cspec["scale"])
            npz = Path(paths["npz"])
            exp = window_expectations(npz, cspec["t0"], cspec["t1"], cspec["speed"])
            meta = _meta_for(npz, name.split("_")[0], cspec["title"])
            meta["footer"] = (f"config {npz.name} | one reset, no in-run resets | "
                              f"{cspec['caption']}")
            r = video_mod.render_trace(video_mod.load_trace(npz), VIDEO / name,
                                       meta=meta, t0=cspec["t0"], t1=cspec["t1"],
                                       speed=cspec["speed"], scale=cspec["scale"],
                                       sheet_times=tuple(np.linspace(
                                           cspec["t0"] + 0.5,
                                           float(exp["t1_effective"]) - 0.5, 3)),
                                       caption=cspec["caption"])
            ver = verify_clip(Path(r["mp4"]), expect_s=exp["expect_s"],
                              expect_frames=exp["expect_frames"],
                              speed=cspec["speed"])
            print(f"[suite] rendered {r['mp4']} ({r['frames']} frames) "
                  f"verify={json.dumps(ver['measured'])} ok={ver['ok']} "
                  f"problems={ver['problems']}", flush=True)
            if not ver["ok"]:
                gate_failures.append({"clip": name, "problems": ver["problems"]})
            results[run_tag].setdefault("renders", []).append({"name": name, "verify": ver})
            run_blob = json.loads((Path(paths["json"])).read_text())
            rub = rubric_mod.assess(paths["npz"], run_blob)
            bundle = {"video": str(VIDEO / name),
                      "contact_sheet": r.get("sheet"), "trace_npz": paths["npz"],
                      "run_json": paths["json"], "config": run_blob.get("config"),
                      "provenance": run_blob.get("provenance"),
                      "metrics": run_blob.get("metrics"), "rubric": rub,
                      "rubric_table": rubric_mod.render_table(rub),
                      "label": cspec["title"], "caption": cspec["caption"],
                      "clip_verification": ver,
                      "render": {"t0": cspec["t0"], "t1": cspec["t1"],
                                 "speed": cspec["speed"], "scale": cspec["scale"],
                                 "fps": FPS, "frames": r["frames"],
                                 "window": exp},
                      "evidence": bool(ver["ok"]),
                      "reproduce": run_blob.get("provenance", {}).get("reproduce"),
                      "render_command": _render_command(paths["npz"], name, cspec)}
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
    # EVIDENCE GATE: a clip whose own verification fails is written with
    # "evidence": false and the suite exits nonzero -- a silently-failing check
    # can never again coexist with a "verified" claim in the index.
    if gate_failures:
        print("[suite] EVIDENCE GATE FAILED (clip is NOT evidence): "
              f"{json.dumps(gate_failures)}", flush=True)
        return 1
    print(f"[suite] evidence gate: all {sum(len(j[2]) for j in jobs)} clips verified",
          flush=True)
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
        # (tag, RunConfig, stance, scheduler, renders); each render is a
        # CLIP_SPECS entry (single source of truth for window/speed/text)
        ("L2_ENTRY_SPEC", dict(controller="feasible", rung="L2", seconds=14.0,
                               start="stand", tag="L2_ENTRY_SPEC"), spec, None, [
            clip_spec("L2_isolated_step.mp4")]),
        ("L2_ENTRY_BASE", dict(controller="feasible", rung="L2", seconds=25.0,
                               start="stand", tag="L2_ENTRY_BASE"), nbase, None, [
            clip_spec("final_L2_entry_walk.mp4")]),
        ("L2_CYCLE", dict(controller="feasible", rung="L2", seconds=34.0,
                          start="stance", tag="L2_CYCLE"), nbase,
         _l1_plus_step_scheduler(0), [
            clip_spec("L2_cycle_step_diag.mp4")]),
        ("L2_SHUFFLE", dict(controller="feasible", rung="L2", seconds=34.0,
                            start="stance", tag="L2_SHUFFLE"), nbase, None, [
            clip_spec("L2_shuffle_3steps_diag.mp4")]),
        ("L2_STANCE_REFUSED", dict(controller="feasible", rung="L2", seconds=10.0,
                                   start="stance", tag="L2_STANCE_REFUSED"), spec, None, [
            clip_spec("L2_stance_step_refused_diag.mp4")]),
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
    gate_failures = []
    for tag, cfg_kw, stance, sched, renders in jobs:
        cfg = RunConfig(**cfg_kw)
        paths = results[tag]["paths"]
        for cspec in renders:
            name = _final_name(cspec["name"], cspec["scale"])
            npz = Path(paths["npz"])
            exp = window_expectations(npz, cspec["t0"], cspec["t1"], cspec["speed"])
            meta = _meta_for(npz, name.split("_")[0], cspec["title"])
            meta["footer"] = f"config {npz.name} | one reset | {cspec['caption']}"
            r = video_mod.render_trace(video_mod.load_trace(npz), VIDEO / name,
                                       meta=meta, t0=cspec["t0"], t1=cspec["t1"],
                                       speed=cspec["speed"], scale=cspec["scale"],
                                       sheet_times=tuple(np.linspace(
                                           cspec["t0"] + 0.4,
                                           float(exp["t1_effective"]) - 0.4, 3)),
                                       caption=cspec["caption"])
            ver = verify_clip(Path(r["mp4"]), expect_s=exp["expect_s"],
                              expect_frames=exp["expect_frames"],
                              speed=cspec["speed"])
            if not ver["ok"]:
                gate_failures.append({"clip": name, "problems": ver["problems"]})
            blob = json.loads(Path(paths["json"]).read_text())
            rub = rubric_mod.assess(paths["npz"], blob)
            bundle = {"video": str(VIDEO / name), "contact_sheet": r.get("sheet"),
                      "trace_npz": paths["npz"], "run_json": paths["json"],
                      "config": blob.get("config"), "provenance": blob.get("provenance"),
                      "metrics": blob.get("metrics"), "rubric": rub,
                      "rubric_table": rubric_mod.render_table(rub),
                      "label": cspec["title"], "caption": cspec["caption"],
                      "clip_verification": ver,
                      "render": {"t0": cspec["t0"], "t1": cspec["t1"],
                                 "speed": cspec["speed"], "scale": cspec["scale"],
                                 "fps": FPS, "frames": r["frames"],
                                 "window": exp},
                      "evidence": bool(ver["ok"]),
                      "l2_evidence_notes": results[tag],
                      "reproduce": (f"MUJOCO_GL=egl python scripts/solo_drill_render.py "
                                    f"run --controller feasible --rung {cfg.rung} "
                                    f"--seconds {cfg.seconds:g} --start {cfg.start} "
                                    f"--out-tag {cfg.tag}")}
            (REPO / "data" / "solo_drill" / (Path(name).stem + ".json")).write_text(
                json.dumps(bundle, indent=1, default=str))
            print(f"[L2] rendered {r['mp4']} verify={json.dumps(ver['measured'])} "
                  f"ok={ver['ok']} problems={ver['problems']}", flush=True)
    (DATA / "l2_suite_summary.json").write_text(json.dumps(results, indent=1, default=str))
    print(json.dumps(results, indent=1, default=str))
    if gate_failures:
        print("[L2] EVIDENCE GATE FAILED (clip is NOT evidence): "
              f"{json.dumps(gate_failures)}", flush=True)
        return 1
    print(f"[L2] evidence gate: all {sum(len(j[4]) for j in jobs)} clips verified",
          flush=True)
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
        exp_tr = window_expectations(npz, 0.0, t_end)
        ver = verify_clip(Path(r["mp4"]), expect_s=exp_tr["expect_s"],
                          expect_frames=exp_tr["expect_frames"])
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


# ------------------------------------------------------------ deliverable clip
DELIVER_SPEC = dict(width=0.30, depth=0.10, reach_cap=0.25, settle_tol=0.026,
                    lean_gain=0.12, v_gate=0.04, pivot_max=0.0)

#: the deliverable clip's claim text (single source: the renderer writes it into
#: the HUD and the bundle; evidence_check reads it back to check the claims)
DELIVER_LABEL = "L2 motion: continuous stepping in the drill stance"
DELIVER_CAPTION = ("rung L2: stance + repeated steps (stalk / backpedal / lateral "
                   "shuffle); every step is gate-checked on the measured CoM margin")
DELIVER_SLOWMO_CAPTION = "0.25x: one step cycle (shift, lift, swing, plant, settle)"


def deliver_case(model, ids, seconds=95.0, spec_kw=None, tag="L2_MOTION",
                 save=True):
    """Run the deliverable motion programme (one continuous episode)."""
    from drill import motion

    spec = motion.MotionSpec(**(spec_kw or DELIVER_SPEC))
    sched = motion.DrillProgramSched(hold_s=0.3, block_hold_s=3.0)
    res = motion.run_case(model, ids, spec, seconds, sched=sched, tag=tag,
                          save_dir=DATA if save else None)
    return spec, sched, res


def cmd_deliver(a) -> int:
    """Run + render the deliverable motion clip with its full evidence bundle."""
    import numpy as np

    from drill import kin as K, motion, rubric as rubric_mod, scene
    from drill import video as video_mod
    from drill.lock import sim_lock

    model = scene.load_model()
    ids = K.RobotIds.build(model)
    kw = dict(DELIVER_SPEC)
    for item in (a.set or []):
        k, v = item.split("=")
        kw[k] = float(v)
    if a.npz:
        # render from a cached trace (the measured way: a clip is rendered from
        # the episode it is claimed to describe, never re-simulated inside a
        # render)
        from drill import motion as motion_mod

        npz_in = Path(a.npz)
        blob_in = json.loads(npz_in.with_suffix(".json").read_text())
        trace_in = video_mod.load_trace(npz_in)
        summary = motion_mod.motion_span(trace_in)
        summary.update({"falls": blob_in["metrics"]["falls"],
                        "steps": blob_in["metrics"]["steps_completed"],
                        "longest_s": blob_in["metrics"]["longest_continuous_s"],
                        "margin_min": round(float(np.min(np.asarray(
                            trace_in["margin"], float))), 4)})
        res = {"paths": {"npz": str(npz_in), "json": str(npz_in.with_suffix(".json"))},
               "summary": summary, "res": type("R", (), {"trace": trace_in})()}
        a.seconds = float(np.asarray(trace_in["t"])[-1])
        if not (a.stance_w > 0):
            print("[deliver] WARNING: rendering a cached trace without --stance-w: "
                  "the bundle will record the DELIVER_SPEC width "
                  f"{kw['width']:.2f} m, which need not be the run's own spec "
                  "(the 2026-10-08 audit: final_L2_motion.json recorded 0.30 m for "
                  "the M_E28f run whose stance is 0.28 m). Pass the run's width.",
                  flush=True)
    else:
        with sim_lock("deliverable motion run"):
            spec, sched, res = deliver_case(model, ids, seconds=a.seconds, spec_kw=kw)
    paths = res["paths"]
    m = res["summary"]
    print("[deliver] steps", m["steps"], "falls", m["falls"],
          "margin_min", m.get("margin_min"), flush=True)
    # phase-advance evidence: the scheduler's own phase string over the run
    trace = res["res"].trace
    t = np.asarray(trace["t"], float)
    cmd_arr = np.asarray(trace["cmd"], float)
    skill = np.asarray(trace["skill_id"], int)
    changes = int(np.sum(np.diff(skill) != 0))
    print("[deliver] skill changes", changes, flush=True)
    npz = Path(paths["npz"])
    blob = json.loads(Path(paths["json"]).read_text())
    step_done = [e for e in blob["events"] if e.get("event") == "step_done"]
    meta = _meta_for(npz, "L2", "L2 motion: continuous stepping in the drill stance")
    sw = a.stance_w if a.stance_w > 0 else kw["width"]
    sd = a.stance_d if a.stance_d > 0 else kw["depth"]
    meta["stance"] = f"{sw:.2f} x {sd:.2f} m (reference 0.49 m: narrowed for steppability)"
    meta["step_times"] = [round(float(e["t"]), 2) for e in step_done]
    meta["footer"] = (f"run: {Path(paths['npz']).name} | stance {sw:.2f} x {sd:.2f} m | "
                      f"scripted feedback, one reset, no in-run resets | "
                      f"steps {len(step_done)}, falls {blob['metrics']['falls']}")
    summary_from_trace = motion.motion_span(npz and video_mod.load_trace(npz))
    t_first = float(step_done[0]["t"]) - 1.2 if step_done else 4.0
    # SAFE WRITE (orchestrator protocol): render to a *.partial.mp4, verify it,
    # and only then rename over the final name -- a killed render must never
    # leave a partial file occupying the artifact name
    out_name = "final_L2_motion.partial.mp4" if a.safe else "final_L2_motion.mp4"
    label = DELIVER_LABEL
    caption = DELIVER_CAPTION
    r = video_mod.render_trace(video_mod.load_trace(npz), VIDEO / out_name,
                               meta=meta, t0=0.0, t1=a.seconds,
                               label=label,
                               sheet_times=tuple(np.linspace(3.0, a.seconds - 3.0, 3)),
                               caption=caption)
    exp = window_expectations(npz, 0.0, a.seconds, 1.0)
    ver = verify_clip(Path(r["mp4"]), expect_s=exp["expect_s"],
                      expect_frames=exp["expect_frames"])
    print("[deliver] main clip", json.dumps(ver), flush=True)
    if a.safe:
        if not ver["ok"]:
            print("[deliver] SAFE WRITE ABORTED: verification failed", json.dumps(ver))
            return 1
        final = VIDEO / "final_L2_motion.mp4"
        Path(r["mp4"]).replace(final)
        r["mp4"] = str(final)
        if r.get("sheet"):
            Path(r["sheet"]).replace(VIDEO / "final_L2_motion_sheet.png")
            r["sheet"] = str(VIDEO / "final_L2_motion_sheet.png")
        print(f"[deliver] SAFE WRITE OK: verified then renamed -> {final} "
              f"({ver.get('nb_frames')} frames, {ver.get('duration')} s, "
              f"{ver.get('width')}x{ver.get('height')}, {ver.get('codec_name')}/"
              f"{ver.get('pix_fmt')})", flush=True)
    # 0.25x slow motion of the first complete step cycle
    slowmo_caption = DELIVER_SLOWMO_CAPTION
    r2 = video_mod.render_trace(video_mod.load_trace(npz),
                                VIDEO / "L2_motion_slowmo_step_quarter.mp4",
                                meta=meta, t0=t_first, t1=t_first + 3.4,
                                speed=0.25, scale=0.5, label=label,
                                sheet_times=(t_first + 0.2, t_first + 1.7, t_first + 3.2),
                                caption=slowmo_caption)
    exp2 = window_expectations(npz, t_first, t_first + 3.4, 0.25)
    ver2 = verify_clip(Path(r2["mp4"]), expect_s=exp2["expect_s"],
                       expect_frames=exp2["expect_frames"], speed=0.25)
    print("[deliver] slowmo", json.dumps(ver2), flush=True)
    rub = rubric_mod.assess(paths["npz"], blob)
    bundle = {"video": str(VIDEO / "final_L2_motion.mp4"), "contact_sheet": r.get("sheet"),
              "slowmo": str(VIDEO / "L2_motion_slowmo_step_quarter.mp4"),
              "trace_npz": paths["npz"], "run_json": paths["json"],
              "config": blob.get("config"), "provenance": blob.get("provenance"),
              "metrics": blob.get("metrics"), "summary": m,
              "skill_changes": changes, "step_events": step_done,
              "rubric": rub, "rubric_table": rubric_mod.render_table(rub),
              "label": label, "caption": caption,
              "clip_verification": ver, "slowmo_verification": ver2,
              "render": {"t0": 0.0, "t1": a.seconds, "speed": 1.0, "scale": 1.0,
                         "fps": FPS, "frames": r["frames"], "window": exp,
                         "stance_w": sw, "stance_d": sd},
              "slowmo_render": {"t0": t_first, "t1": t_first + 3.4, "speed": 0.25,
                                "scale": 0.5, "fps": FPS, "frames": r2["frames"],
                                "window": exp2, "caption": slowmo_caption},
              "evidence": bool(ver["ok"] and ver2["ok"]),
              "stance_note": (f"stance {sw:.2f} x {sd:.2f} m: a "
                              "documented narrowing from the operator's 0.49 m "
                              "reference -- the crossing CoM travel a lift needs "
                              "grows as half the width, and the measured lateral "
                              "authority caps it (reports/2026-10-08/drill_motion.md)"),
              "reproduce": (f"MUJOCO_GL=egl python scripts/solo_drill_render.py "
                            f"deliver --seconds {a.seconds:g}")}
    (REPO / "data" / "solo_drill").mkdir(parents=True, exist_ok=True)
    # merge into the existing bundle rather than replace it: the hand-assembled
    # evidence (per-step loaded creep, safety-governor analysis, margin
    # reconciliation) must survive the render pass
    bpath = REPO / "data" / "solo_drill" / "final_L2_motion.json"
    if bpath.exists():
        try:
            prev = json.loads(bpath.read_text())
            prev.update(bundle)
            bundle = prev
        except Exception:                               # pragma: no cover
            pass
    bpath.write_text(json.dumps(bundle, indent=1, default=str))
    print(json.dumps({"steps": m["steps"], "falls": m["falls"],
                      "margin_min": m["margin_min"], "cadence": m["cadence_s_per_step"],
                      "verify": ver["ok"], "verify_slowmo": ver2["ok"],
                      "skill_changes": changes}, indent=1))
    if not (ver["ok"] and ver2["ok"]):
        print("[deliver] EVIDENCE GATE FAILED (clip is NOT evidence): "
              f"{json.dumps({'main': ver['problems'], 'slowmo': ver2['problems']})}",
              flush=True)
        return 1
    return 0


# --------------------------------------------------------- motion study (D1-D3)
def cmd_motion(a) -> int:
    """The width/mechanism/cadence study and the deliverable-motion clip.

    Stages:
      widths     -- the decision table: widths x {required, achievable, cadence}
      mech       -- D2 authority mechanisms, one at a time and composed
      cadence    -- the shift-speed / settle / drop sweep
      deliver    -- run the chosen programme and render final_L2_motion.mp4
      tracks     -- D4 reference-track repair probe (stance_widen_step, stalk_shuffle)
    """
    import numpy as np

    from drill import kin as K, motion, scene
    from drill.lock import sim_lock

    stage = a.stage
    model = scene.load_model()
    ids = K.RobotIds.build(model)
    DATA.mkdir(parents=True, exist_ok=True)
    base = motion.MotionSpec(width=a.width, depth=a.depth, reach_cap=a.cap,
                             shift_speed=a.speed, drop_max=a.drop,
                             settle_tol=a.settle_tol, lean_gain=a.lean,
                             support_roll=a.support_roll,
                             v_gate=a.v_gate, pivot_max=a.pivot_max,
                             settle=not a.no_settle,
                             centre_tol=(a.centre_tol if a.centre_tol is not None
                                         else 0.030))
    out = {}
    if stage == "widths":
        widths = [float(v) for v in a.widths.split(",")]
        with sim_lock(f"motion widths x{len(widths)}"):
            rows = motion.stage_width_table(
                model, ids, widths=widths, depth=a.depth, seconds=a.seconds,
                reach_cap=a.cap, base=base, sched=motion.ShuffleSched(vy=a.vy))
        (DATA / "motion_widths.json").write_text(json.dumps(rows, indent=1, default=str))
        out = {"rows": rows}
    elif stage == "mech":
        variants = {
            "m0_baseline": base,
            "m1_lean": replace_spec(base, lean_gain=0.16),
            "m2_support_roll": replace_spec(base, support_roll=True),
            "m3_lean_roll": replace_spec(base, lean_gain=0.16, support_roll=True),
            "m4_speed": replace_spec(base, shift_speed=0.055),
            "m5_all": replace_spec(base, lean_gain=0.16, support_roll=True,
                                   shift_speed=0.055, drop_max=0.055,
                                   settle_tol=0.026, drop_gain=0.26),
        }
        with sim_lock(f"motion mechanisms x{len(variants)}"):
            out = motion.stage_variants(model, ids, variants, seconds=a.seconds,
                                        sched=motion.ShuffleSched(vy=a.vy))
        (DATA / "motion_mech.json").write_text(json.dumps(out, indent=1, default=str))
    elif stage == "cadence":
        variants = {}
        for sp in (0.030, 0.045, 0.060, 0.080):
            variants[f"speed{int(sp*1000):03d}"] = replace_spec(
                base, shift_speed=sp, centre_tol=(a.centre_tol or 0.30),
                settle=False)
        for stl in (0.020, 0.026, 0.035):
            variants[f"settle{int(stl*1000):03d}"] = replace_spec(base,
                                                                  settle_tol=stl)
        with sim_lock(f"motion cadence x{len(variants)}"):
            out = motion.stage_variants(model, ids, variants, seconds=a.seconds,
                                        sched=motion.ShuffleSched(vy=a.vy))
        (DATA / "motion_cadence.json").write_text(json.dumps(out, indent=1, default=str))
    elif stage == "singles":
        out = singles(model, ids, base, a)
    else:
        raise SystemExit(f"unknown motion stage {stage!r}")
    print(json.dumps(out, indent=1, default=str)[:4000])
    return 0


def replace_spec(base, **kw):
    from dataclasses import replace

    return replace(base, **kw)


def singles(model, ids, base, a):
    """Single-case runs used to iterate on one parameter (tag -> saved trace)."""
    import json as _json

    import numpy as np

    from drill import motion
    from drill.lock import sim_lock

    cases = {}
    for item in (a.case or []):
        # name:key=val,key=val  (values are floats or 0/1 for flags)
        name, _, body = item.partition(":")
        kw = {}
        for kv in filter(None, body.split(",")):
            k, v = kv.split("=")
            k = {"w": "width", "d": "depth", "cap": "reach_cap",
                 "speed": "shift_speed"}.get(k, k)
            kw[k] = float(v)
        cases[name] = replace_spec(base, **kw)
    seconds = a.seconds
    saved = {}
    with sim_lock(f"motion singles x{len(cases)}"):
        for name, spec in cases.items():
            if a.deliverable or a.sched == "deliverable":
                sched = motion.DeliverableSched(block_s=a.block, start_hold=0.5)
            elif a.sched == "hold":
                sched = motion.StepHoldSched(vx=a.vx, vy=a.vy, hold_s=a.hold_s,
                                             axis=a.axis)
            elif a.sched == "program":
                sched = motion.DrillProgramSched(hold_s=a.hold_s,
                                                 block_hold_s=(a.block if a.block > 1.0 else 3.0))
            elif a.sched == "tap":
                sched = motion.TapSched(vx=a.vx, hold_s=a.hold_s,
                                        switch_s=a.switch_s)
            else:
                sched = motion.ShuffleSched(vy=a.vy, vx=a.vx, axis=a.axis,
                                            block=a.block)
            res = motion.run_case(model, ids, spec, seconds, sched=sched,
                                  tag=f"M_{name}",
                                  save_dir=DATA if a.save else None)
            saved[name] = {**res["summary"],
                           "events": [e for e in res["events"]
                                      if e.get("event") in (
                                          "shift_done", "step_done", "step_aborted",
                                          "step_refused", "fall", "emergency_plant",
                                          "settle_timeout", "recovered")][:400],
                           "paths": res.get("paths")}
            print(f"[{name}] fall {saved[name]['falls']} steps {saved[name]['steps']} "
                  f"aborts {saved[name]['aborts']} cadence "
                  f"{saved[name]['cadence_s_per_step']} margin_min "
                  f"{saved[name]['margin_min']}", flush=True)
    (DATA / "motion_singles.json").write_text(_json.dumps(saved, indent=1, default=str))
    return saved


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
    d.add_argument("--safe", action="store_true",
                   help="render to *.partial.mp4, verify, then atomically rename")
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

    m = sub.add_parser("motion", help="width/mechanism/cadence study + motion clip")
    m.add_argument("--stage", default="widths",
                   choices=("widths", "mech", "cadence", "singles", "tracks",
                            "deliver"))
    m.add_argument("--widths", default="0.21,0.28,0.35,0.42,0.495")
    m.add_argument("--width", type=float, default=0.28)
    m.add_argument("--depth", type=float, default=0.14)
    m.add_argument("--seconds", type=float, default=30.0)
    m.add_argument("--cap", type=float, default=0.40)
    m.add_argument("--speed", type=float, default=0.030)
    m.add_argument("--drop", type=float, default=0.045)
    m.add_argument("--settle-tol", type=float, default=0.020)
    m.add_argument("--lean", type=float, default=0.0)
    m.add_argument("--support-roll", action="store_true")
    m.add_argument("--v-gate", type=float, default=0.0)
    m.add_argument("--pivot-max", type=float, default=0.50)
    m.add_argument("--no-settle", action="store_true")
    m.add_argument("--centre-tol", type=float, default=None)
    m.add_argument("--vy", type=float, default=0.09)
    m.add_argument("--vx", type=float, default=0.09)
    m.add_argument("--axis", default="y", choices=("x", "y"))
    m.add_argument("--sched", default="shuffle",
                   choices=("shuffle", "hold", "tap", "program", "deliverable"))
    m.add_argument("--switch-s", type=float, default=0.0)
    m.add_argument("--hold-s", type=float, default=1.2)
    m.add_argument("--block", type=float, default=9.0)
    m.add_argument("--case", action="append")
    m.add_argument("--save", action="store_true")
    m.add_argument("--deliverable", action="store_true")
    m.set_defaults(func=cmd_motion)

    dv = sub.add_parser("deliver", help="run + render the deliverable motion clip")
    dv.add_argument("--seconds", type=float, default=95.0)
    dv.add_argument("--set", action="append")
    dv.add_argument("--npz", default="")
    dv.add_argument("--stance-w", type=float, default=0.0)
    dv.add_argument("--stance-d", type=float, default=0.0)
    dv.add_argument("--safe", action="store_true",
                    help="render to *.partial.mp4, verify, then atomically rename")
    dv.set_defaults(func=cmd_deliver)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
