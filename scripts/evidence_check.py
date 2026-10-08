#!/usr/bin/env python3
"""Evidence integrity: verify clips, check label claims, gate the index.

The durable fix for the 2026-10-08 clip audit.  Three rules, all mechanical:

1. **Verification.**  ``verify_clip`` (from ``solo_drill_render``) is re-run on
   every bundle with the window/speed recovered from the renderer's own spec
   table (or the bundle's recorded ``render`` block), so the expected duration
   and frame count come from the TRACE + WINDOW, never from the artifact.
2. **Gating.**  A clip is presented as evidence only when its verification
   passes AND its numeric label claims check out.  Anything else is listed as
   ``verification-failed`` / ``claims-failed`` with the reason.
3. **Label integrity.**  Every numeric claim a clip's label/caption makes
   (duration, step count, stance width, push count/force, speed factor), plus
   the fall words and the event-coverage of the render window, is compared with
   the bundle + trace.  A mismatch FAILS.

Commands::

    python scripts/evidence_check.py table            # machine-generated table
    python scripts/evidence_check.py claims           # label integrity only
    python scripts/evidence_check.py refresh [--write]  # re-verify (+ update bundles)
    python scripts/evidence_check.py gate             # exit 1 unless all ok

Regenerate the report table with ``python scripts/evidence_check.py table``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import solo_drill_render as sdr  # noqa: E402  (verify_clip, window_expectations, CLIP_SPECS)

BUNDLE_DIR = REPO / "data" / "solo_drill"
VIDEO_DIR = REPO / "videos" / "solo_drill"

TODAY = "2026-10-08"

#: how close a claim must be to the measured value
DUR_TOL_S = 0.5
STANCE_W_TOL_M = 0.03
STANCE_D_TOL_M = 0.06
FORCE_TOL_N = 1.0
#: the measured lateral sole separation runs ~0-22 mm above the commanded spec
#: in the runs on disk, so the width tolerance is 30 mm: a legitimate
#: spec-vs-measure gap passes, a >=30 mm drift (e.g. the entry-walk caption's
#: "0.25 m wide" against the 0.21 m base, 38 mm) fails.


# --------------------------------------------------------------- primitives
def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def probe(path: Path) -> dict:
    """ffprobe facts for one clip (duration / frames / size / fps / codec)."""
    out = {"path": str(path)}
    try:
        pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                             "-show_entries", "stream=nb_frames,duration,codec_name,"
                             "pix_fmt,width,height,r_frame_rate", "-of", "json",
                             str(path)], capture_output=True, text=True, timeout=120)
        st = json.loads(pr.stdout)["streams"][0]
        out.update({k: st.get(k) for k in ("nb_frames", "duration", "codec_name",
                                           "pix_fmt", "width", "height", "r_frame_rate")})
    except Exception as exc:                                   # pragma: no cover
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def clip_duration_s(path: Path) -> float | None:
    return _num(probe(path).get("duration"))


def trace_end(npz: Path | str) -> float | None:
    if not npz or not Path(npz).exists():
        return None
    return sdr._trace_end(npz)


def load_bundle(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _bundle_files() -> list[Path]:
    """The per-clip bundles (any JSON in the bundle dir that names a video)."""
    out = []
    for p in sorted(BUNDLE_DIR.glob("*.json")):
        try:
            if load_bundle(p).get("video"):
                out.append(p)
        except (OSError, ValueError):                       # pragma: no cover
            continue
    return out


def spec_for(name: str) -> dict | None:
    """CLIP_SPECS entry for a clip name, as rendered (``_diag`` names included)."""
    if name in sdr.CLIP_SPECS:
        return sdr.CLIP_SPECS[name]
    for k, v in sdr.CLIP_SPECS.items():
        if sdr._final_name(k, v["scale"]) == name:
            return v
    return None


def recover_window(name: str, bundle: dict) -> dict | None:
    """The render intent for a clip: (t0, t1, speed) + where it was recovered from.

    Order: the bundle's own recorded ``render`` block > the renderer's spec
    table (CLIP_SPECS -- the single source of truth) > the deliver/ad-hoc
    fallbacks.  A recovered window that does not match the artifact makes
    verification FAIL, so a bad transcription cannot pass silently.
    """
    if Path(str(bundle.get("video") or "")).name != name:
        # a secondary artifact of the bundle (the deliverable's slow-motion
        # clip): its own recorded block, else the step-event fallback
        sr = bundle.get("slowmo_render") or {}
        if sr:
            return {"t0": sr.get("t0"), "t1": sr.get("t1"),
                    "speed": sr.get("speed", 1.0), "source": "bundle.slowmo_render"}
        if name == "L2_motion_slowmo_step_quarter.mp4":
            steps = [_num(e.get("t")) for e in (bundle.get("step_events") or [])
                     if e.get("t") is not None]
            if steps:
                return {"t0": steps[0] - 1.2, "t1": steps[0] + 2.2, "speed": 0.25,
                        "source": "deliver slowmo (first step t from step_events)"}
        return None
    r = bundle.get("render") or {}
    if r:
        return {"t0": r.get("t0", 0.0), "t1": r.get("t1"),
                "speed": r.get("speed", 1.0), "source": "bundle.render"}
    if spec_for(name) is not None:
        s = spec_for(name)
        return {"t0": s["t0"], "t1": s["t1"], "speed": s["speed"],
                "source": "CLIP_SPECS (renderer spec table)"}
    if name == "final_L2_motion.mp4":
        return {"t0": 0.0, "t1": None, "speed": 1.0,
                "source": "deliver (0 .. trace end)"}
    if bundle.get("trace_npz"):
        return {"t0": 0.0, "t1": None, "speed": 1.0,
                "source": "assumed 0 .. trace end (validated against the artifact)"}
    return None


def verify_bundle(name: str, bundle: dict, path: Path) -> dict:
    """Re-run the fixed verifier with trace-derived expectations."""
    win = recover_window(name, bundle)
    npz = bundle.get("trace_npz")
    if win is None or not (npz and Path(npz).exists()):
        ver = sdr.verify_clip(path)                     # format + pixels only
        ver["window"] = None
        ver["window_source"] = "no trace bundle: format/pixel checks only"
        return ver
    exp = sdr.window_expectations(npz, win["t0"], win["t1"], win["speed"])
    ver = sdr.verify_clip(path, expect_s=exp["expect_s"],
                          expect_frames=exp["expect_frames"],
                          speed=win["speed"])
    ver["window"] = {**win, "t0": exp["t0"], "t1": exp["t1"],
                     "t1_effective": exp["t1_effective"],
                     "trace_end_s": exp["trace_end_s"],
                     "clip_window_s": round(exp["t1_effective"] - exp["t0"], 3)}
    ver["window_source"] = win["source"]
    return ver


# ------------------------------------------------------------------- facts
def facts_from_bundle(bundle: dict, clip_s: float | None = None,
                      clip_path: Path | str | None = None) -> dict:
    """Everything a label's numeric claims can be checked against."""
    m = bundle.get("metrics") or {}
    rub = bundle.get("rubric") or {}
    a = ((rub.get("sections") or {}).get("A_stance") or {})
    pushes = m.get("pushes") or {}
    rec = [r for r in (pushes.get("recoveries") or []) if isinstance(r, dict)]
    cfg_pushes = ((bundle.get("config") or {}).get("pushes")) or []
    summary = bundle.get("summary") or {}
    f = {
        "clip_s": clip_s,
        "run_s": _num(m.get("duration_s")),
        "falls": m.get("falls"),
        "steps": m.get("steps_completed"),
        "steps_summary": summary.get("steps"),
        "n_pushes": pushes.get("n_pushes"),
        "pushes_recovered": pushes.get("recovered"),
        "push_times": [_num(r.get("t")) for r in rec if r.get("t") is not None],
        "push_forces": [max((abs(_num(v)) or 0.0) for v in (r.get("force") or []))
                        for r in rec if r.get("force")],
        "stance_w_measured": (a.get("A1_width") or {}).get("value"),
        "stance_d_measured": (a.get("A2_base_depth") or {}).get("value"),
        "trace_end_s": trace_end(bundle.get("trace_npz")),
        "trace_npz": bundle.get("trace_npz"),
        "clip_path": str(clip_path) if clip_path else bundle.get("video"),
    }
    if not f["push_times"] and cfg_pushes:
        f["push_times"] = [_num(p.get("t")) for p in cfg_pushes]
        f["push_forces"] = [max(abs(_num(p.get(k)) or 0.0) for k in ("fx", "fy", "fz"))
                            for p in cfg_pushes]
        f["n_pushes"] = f["n_pushes"] or len(cfg_pushes)
    return f


def _res(kind: str, claim: str, value, ok: bool | None, measured, why: str = "") -> dict:
    status = "ok" if ok else ("fail" if ok is False else "skip")
    return {"kind": kind, "claim": claim.strip(), "value": value,
            "measured": measured, "status": status, "why": why}


_NUM = r"\d+(?:\.\d+)?"
#: markers that make a number an EVENT TIME rather than a duration
_EVENT_PREFIX = re.compile(r"(t\s*=\s*|at\s+|after\s+|since\s+|until\s+|~\s*|burns?|by\s+t\s*=)")

_SIMPLE_DUR = re.compile(rf"(?<![\w.=])({_NUM})\s*(?:s\b|sec\b|secs\b|seconds?\b)", re.I)
_WINDOW_RANGE = re.compile(rf"({_NUM})\s*(?:-|\u2013|to)\s*({_NUM})\s*s\b")
_PUSH = re.compile(rf"(\d+)\s*[x\u00d7]\s*({_NUM})\s*N\b")
_FORCE = re.compile(rf"({_NUM})\s*N\b")
_STEP_RANGE = re.compile(rf"(\d+)\s*(?:-|\u2013|to)\s*(\d+)\s*steps?\b", re.I)
_STEP_COUNT = re.compile(rf"(\d+)\s*(?:gate-checked\s+|visible\s+|consecutive\s+|x\s+)?steps?\b|steps?\s*[=:]?\s*(\d+)\b", re.I)
_STANCE_XD = re.compile(rf"({_NUM})\s*x\s*({_NUM})\s*m\b")
_STANCE_WIDE = re.compile(rf"({_NUM})\s*m\s*(?:-|\s)*wide", re.I)
_STANCE_DEEP = re.compile(rf"({_NUM})\s*m\s+deep", re.I)
_STANCE_BARE = re.compile(rf"({_NUM})\s*m\s+stance")
_SPEED = re.compile(rf"({_NUM})\s*x\b")
_NEG_FALL = re.compile(r"(does\s+not\s+topple|not\s+topple|no\s+fall|stays?\s+up|instead\s+of\s+toppl|without\s+toppl|refuses?\s+instead)", re.I)
_POS_FALL = re.compile(r"(topples?|toppling|falls?\b)", re.I)
_PLANTED = re.compile(r"(both\s+feet\s+planted|feet\s+planted\s+throughout)", re.I)


def _first_fall_word(text: str):
    """The first fall word that is not a NEGATED count ("0 falls", "no falls")."""
    for m in _POS_FALL.finditer(text or ""):
        before = (text or "")[max(0, m.start() - 5):m.start()]
        if re.search(r"(\d|\bno|\bn)\s*$", before):
            continue
        return m
    return None


def check_text_claims(text: str, facts: dict, window: dict | None = None) -> list[dict]:
    """Every numeric claim in ``text`` vs the bundle/trace facts.

    ``window`` = the recovered render intent ({t0, t1, t1_effective, speed,
    clip_window_s, trace_end_s}) or None.  A claim whose fact source is absent
    is ``skip`` (with the reason); a claim that contradicts the facts is
    ``fail``.
    """
    out: list[dict] = []
    text = text or ""
    masked = [False] * len(text)
    spans: list[tuple[int, int]] = []

    def _take(m) -> None:
        spans.append(m.span())

    # 1) push count+force ("3 x 20 N") -- must come before steps/duration
    for m in _PUSH.finditer(text):
        n, force = int(m.group(1)), float(m.group(2))
        if facts.get("n_pushes") is None:
            out.append(_res("push_count", m.group(0), n, None, None,
                            "run records no push count"))
        else:
            ok = int(facts["n_pushes"]) == n
            out.append(_res("push_count", m.group(0), n, ok, facts["n_pushes"],
                            "" if ok else f"run has {facts['n_pushes']} pushes"))
        fmax = max(facts.get("push_forces") or [0.0])
        ok = abs(fmax - force) <= FORCE_TOL_N
        out.append(_res("push_force", m.group(0), force, ok, fmax,
                        "" if ok else f"max |push force| measured {fmax:g} N"))
        _take(m)
    # 2) bare forces ("90 N", "20 N recovers") -- cross-run values are skipped
    for m in _FORCE.finditer(text):
        if any(s <= m.start() < e for s, e in spans):
            continue
        force = float(m.group(1))
        known = facts.get("push_forces") or []
        ok = any(abs(k - force) <= FORCE_TOL_N for k in known)
        out.append(_res("push_force", m.group(0), force,
                        True if ok else None, known or None,
                        "" if ok else "no such push force in this run "
                                      "(cross-run claim: not checkable from this bundle)"))
        _take(m)
    # 3) step ranges ("2-3 steps")
    for m in _STEP_RANGE.finditer(text):
        lo, hi = int(m.group(1)), int(m.group(2))
        steps = facts.get("steps")
        if steps is None:
            out.append(_res("steps_range", m.group(0), [lo, hi], None, None,
                            "run records no step count"))
        else:
            ok = lo <= int(steps) <= hi
            out.append(_res("steps_range", m.group(0), [lo, hi], ok, steps,
                            "" if ok else f"trace has {steps} steps, outside [{lo},{hi}]"))
        _take(m)
    # 4) step counts ("6 gate-checked steps", "steps 5")
    for m in _STEP_COUNT.finditer(text):
        if any(s <= m.start() < e for s, e in spans):
            continue
        n = int(m.group(1) or m.group(2))
        steps = facts.get("steps")
        if steps is None:
            out.append(_res("steps", m.group(0), n, None, None, "run records no step count"))
        else:
            ok = int(steps) == n
            out.append(_res("steps", m.group(0), n, ok, steps,
                            "" if ok else f"trace has {steps} steps"))
        _take(m)
    # 5) window ranges ("t=0-8.06 s", "0-30 s")
    for m in _WINDOW_RANGE.finditer(text):
        if any(s <= m.start() < e for s, e in spans):
            continue
        a0, a1 = float(m.group(1)), float(m.group(2))
        before = text[max(0, m.start() - 4):m.start()]
        if window is None:
            out.append(_res("window", m.group(0), [a0, a1], None, None,
                            "render window not recorded for this clip"))
            _take(m)
            continue
        t0 = window.get("t0", 0.0)
        t1e = window.get("t1_effective")
        ok = (t0 is not None and a0 >= t0 - 0.05
              and (t1e is None or a1 <= t1e + 0.05))
        out.append(_res("window", ("t=" if before.endswith("t=") else "") + m.group(0),
                        [a0, a1], ok, [t0, t1e],
                        "" if ok else f"clip covers [{t0}, {t1e}], claim [{a0}, {a1}]"))
        _take(m)
    # 6) durations ("90 s", "12 s hold") -- event times excluded
    for m in _SIMPLE_DUR.finditer(text):
        if any(s <= m.start() < e for s, e in spans):
            continue
        before = text[max(0, m.start(1) - 10):m.start(1)]
        if _EVENT_PREFIX.search(before):
            continue
        val = float(m.group(1))
        cands = {"clip_s": facts.get("clip_s"), "run_s": facts.get("run_s")}
        if window is not None and window.get("clip_window_s") is not None:
            cands["window_s"] = window["clip_window_s"]
        known = {k: v for k, v in cands.items() if v is not None}
        if not known:
            out.append(_res("duration", m.group(0), val, None, None, "no duration facts"))
            continue
        hit = [k for k, v in known.items() if abs(val - v) <= DUR_TOL_S]
        out.append(_res("duration", m.group(0), val, bool(hit), known,
                        "" if hit else f"{val:g} s matches none of {known}"))
        _take(m)
    # 7) stance width/depth ("0.21 m wide, 0.06 m deep", "0.28 x 0.10 m")
    for rx, kind in ((_STANCE_XD, "stance_w"), (_STANCE_WIDE, "stance_w"),
                     (_STANCE_BARE, "stance_w"), (_STANCE_DEEP, "stance_d")):
        for m in rx.finditer(text):
            if any(s <= m.start() < e for s, e in spans):
                continue
            val = float(m.group(1))
            meas = facts.get("stance_w_measured" if kind == "stance_w"
                             else "stance_d_measured")
            if meas is None:
                out.append(_res(kind, m.group(0), val, None, None,
                                "no stance measurement in the bundle"))
            else:
                tol = STANCE_W_TOL_M if kind == "stance_w" else STANCE_D_TOL_M
                ok = abs(val - float(meas)) <= tol
                out.append(_res(kind, m.group(0), val, ok, meas,
                                "" if ok else f"measured {'lateral' if kind == 'stance_w' else 'fore-aft'} "
                                              f"separation {float(meas):.3f} m"))
            _take(m)
            if rx is _STANCE_XD and m.group(2):
                d = float(m.group(2))
                meas_d = facts.get("stance_d_measured")
                if meas_d is not None:
                    ok = abs(d - float(meas_d)) <= STANCE_D_TOL_M
                    out.append(_res("stance_d", m.group(0), d, ok, meas_d,
                                    "" if ok else f"measured fore-aft separation {float(meas_d):.3f} m"))
                    _take(m)
    # 8) slow-motion factor ("0.25x") vs the measured duration ratio
    for m in _SPEED.finditer(text):
        if any(s <= m.start() < e for s, e in spans):
            continue
        val = float(m.group(1))
        if not 0 < val < 1:
            continue
        ratio_meas, ratio_exp = None, 1.0 / val
        if window is not None and window.get("clip_window_s"):
            ratio_meas = (facts.get("clip_s") or 0.0) / window["clip_window_s"]
        if ratio_meas is None:
            out.append(_res("speed", m.group(0), val, None, None,
                            "render window not recorded: duration ratio unverifiable"))
        else:
            ok = abs(ratio_meas - ratio_exp) <= max(0.1, 0.05 * ratio_exp)
            out.append(_res("speed", m.group(0), val, ok,
                            {"measured_ratio": round(ratio_meas, 3),
                             "expected_ratio": round(ratio_exp, 3)},
                            "" if ok else f"clip runs at {ratio_meas:.2f}x real time "
                                          f"(ratio {ratio_meas:.2f}), claim needs "
                                          f"{ratio_exp:.2f}"))
        _take(m)
    # 9) fall words: "topples" => falls >= 1; "does NOT topple" => falls == 0
    if _NEG_FALL.search(text):
        falls = facts.get("falls")
        ok = (falls == 0) if falls is not None else None
        out.append(_res("falls", _NEG_FALL.search(text).group(0), 0, ok, falls,
                        "" if ok else f"run has falls={falls}"))
    else:
        pm = _first_fall_word(text)
        if pm is not None:
            falls = facts.get("falls")
            if falls is None:
                out.append(_res("falls", pm.group(0), ">=1", None, None,
                                "run records no fall count"))
            else:
                ok = int(falls) >= 1
                out.append(_res("falls", pm.group(0), ">=1", ok, falls,
                                "" if ok else f"the run does NOT fall (falls={falls})"))
    # 10) "both feet planted throughout" -> both foot loads > 20 N in the window
    pm = _PLANTED.search(text)
    if pm:
        loads = _foot_loads(facts, window)
        if loads is None:
            out.append(_res("planted", pm.group(0), True, None, None,
                            "no trace available: load check skipped"))
        else:
            frac, n = loads
            ok = frac >= 0.95
            out.append(_res("planted", pm.group(0), True, ok,
                            {"both_feet_loaded_frac": round(frac, 4), "frames": n},
                            "" if ok else f"both feet loaded in only {frac:.3f} of frames"))
    return out


def _foot_loads(facts: dict, window: dict | None):
    """Fraction of window frames with BOTH feet loaded (>20 N), from the trace."""
    npz = facts.get("trace_npz")
    if not npz or not Path(npz).exists():
        return None
    import numpy as np

    with np.load(npz, allow_pickle=True) as z:
        if "foot_load" not in z.files:
            return None
        t = np.asarray(z["t"], float)
        fl = np.asarray(z["foot_load"], float)
    t0 = (window or {}).get("t0", 0.0) or 0.0
    t1 = (window or {}).get("t1_effective", t[-1])
    sel = (t >= t0 - 1e-9) & (t <= (t1 if t1 is not None else t[-1]) + 1e-9)
    if not sel.any():
        return None
    both = np.minimum(fl[sel, 0], fl[sel, 1])
    return float((both > 20.0).mean()), int(sel.sum())


def check_event_coverage(text: str, facts: dict, window: dict | None) -> list[dict]:
    """Every push the run performed must be inside the window OR acknowledged.

    The audited defect: `L1_90s_with_pushes` renders t=0-60 s of a run whose
    third push is at t=62 s while the caption claims all three are shown.
    """
    out = []
    if window is None or not facts.get("push_times"):
        return out
    t0 = window.get("t0", 0.0) or 0.0
    t1 = window.get("t1_effective", (window.get("t1") if window.get("t1") is not None
                                     else facts.get("trace_end_s")))
    for pt in facts["push_times"]:
        if pt is None or (t0 - 0.05 <= pt <= (t1 + 0.05 if t1 is not None else pt)):
            continue
        acknowledged = bool(re.search(rf"{pt:g}\s*s|{pt:g}", text))
        out.append(_res("event_coverage", f"push at t={pt:g} s", [t0, t1],
                        True if acknowledged else False,
                        {"window": [t0, t1]},
                        "outside the window but acknowledged in the text"
                        if acknowledged else
                        f"push at t={pt:g} s lies outside the render window "
                        f"[{t0}, {t1}] and the text does not acknowledge it"))
    return out


def bundle_text(bundle: dict) -> str:
    """The human-readable claim surface of a bundle (label + caption + notes)."""
    return " | ".join(str(bundle.get(k)) for k in
                      ("label", "caption", "stance_note") if bundle.get(k))


# --------------------------------------------------------------- verdicts
def evidence_verdict(ver: dict, claims: list[dict],
                     extra_ver: list | None = None) -> dict:
    """THE GATE: is this clip presentable as evidence?"""
    claim_fails = [c for c in claims if c["status"] == "fail"]
    for label, v in (extra_ver or []):
        if v is not None and not v.get("ok", False):
            return {"ok": False, "status": "verification-failed",
                    "reason": f"{label}: "
                              + "; ".join(v.get("problems") or ["verification failed"])}
    if ver is not None and not ver.get("ok", False):
        return {"ok": False, "status": "verification-failed",
                "reason": "; ".join(ver.get("problems") or ["verification failed"])}
    if claim_fails:
        return {"ok": False, "status": "claims-failed",
                "reason": "; ".join(f"{c['kind']}: {c['claim']} -> {c['why']}"
                                    for c in claim_fails)}
    return {"ok": True, "status": "evidence",
            "reason": "; ".join(c["why"] for c in claims if c["status"] == "skip")
                      or "verification and all numeric claims pass"}


def _rel(p: Path | str) -> str:
    try:
        return str(Path(p).relative_to(REPO))
    except ValueError:
        return str(p)


def check_bundle(bundle_path: Path, verify: bool = True) -> dict:
    """Full integrity record for one bundle: verification + claims + verdict.

    A deliverable bundle can cover two videos (the main clip and its slow-mo
    companion); both are verified and both contribute to the verdict.
    """
    bundle = load_bundle(bundle_path)
    name = Path(str(bundle.get("video", bundle_path.stem))).name
    path = Path(bundle.get("video") or "")
    if not path.is_absolute():
        path = REPO / path
    clip_s = clip_duration_s(path) if path.exists() else None
    facts = facts_from_bundle(bundle, clip_s=clip_s, clip_path=path)
    win = recover_window(name, bundle)
    if verify:
        ver = verify_bundle(name, bundle, path) if path.exists() else {
            "ok": False, "problems": [f"video file missing: {path}"], "window": None}
    else:
        ver = bundle.get("clip_verification") or {"ok": False,
                                                  "problems": ["no verification recorded"]}
    # the claim window must carry the artifact's effective window (clip_window_s)
    claim_win = (ver.get("window") if isinstance(ver, dict) else None) or win
    claims = check_text_claims(bundle_text(bundle), facts, claim_win)
    claims += check_event_coverage(bundle_text(bundle), facts, claim_win)
    # internal consistency: a bundle that records both step counts must agree
    if facts.get("steps_summary") is not None and facts.get("steps") is not None \
            and facts["steps_summary"] != facts["steps"]:
        claims.append(_res("steps_internal", "summary.steps vs metrics.steps_completed",
                           facts["steps_summary"], False, facts["steps"],
                           "the bundle contradicts itself"))
    # the bundle's secondary artifact (slow-mo companion), if any
    extra_ver, extra_clips = [], []
    slow = bundle.get("slowmo")
    if slow:
        spath = Path(str(slow))
        spath = spath if spath.is_absolute() else REPO / spath
        sname = spath.name
        sver = (verify_bundle(sname, bundle, spath) if (verify and spath.exists())
                else (bundle.get("slowmo_verification")
                      or {"ok": False, "problems": ["no slowmo verification recorded"]}))
        sfacts = {**facts, "clip_s": clip_duration_s(spath) if spath.exists() else None}
        stext = (bundle.get("slowmo_render") or {}).get("caption") or ""
        swin = sver.get("window") if isinstance(sver, dict) else None
        sclaims = check_text_claims(stext, sfacts, swin) \
            + check_event_coverage(stext, sfacts, swin)
        for c in sclaims:
            c["clip"] = sname
        claims += sclaims
        extra_ver.append((sname, sver))
        extra_clips.append({"clip": sname, "path": _rel(spath), "label": stext,
                            "verification": sver, "claims": sclaims})
    verdict = evidence_verdict(ver, claims, extra_ver)
    return {"bundle": _rel(bundle_path), "clip": name,
            "path": _rel(path),
            "label": bundle.get("label"), "caption": bundle.get("caption"),
            "window": (ver.get("window") if isinstance(ver, dict) else None),
            "window_source": (ver.get("window_source") if isinstance(ver, dict) else None),
            "verification": ver, "claims": claims, "verdict": verdict,
            "secondary": extra_clips,
            "facts": {k: v for k, v in facts.items() if k != "trace_npz"}}


# ------------------------------------------------------------------ table
def _unbundled_videos() -> list[Path]:
    """Every video on disk with no per-clip bundle (indexed or not)."""
    bundled_names = set()
    for b in _bundle_files():
        blob = load_bundle(b)
        if blob.get("video"):
            bundled_names.add(Path(blob["video"]).name)
        if blob.get("slowmo"):
            bundled_names.add(Path(blob["slowmo"]).name)
    dirs = [VIDEO_DIR, VIDEO_DIR / "baselines", REPO / "videos" / "refs",
            REPO / "videos" / "teacher", REPO / "videos" / "env"]
    found = []
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.mp4")):
            if p.name in bundled_names or ".partial" in p.name:
                continue
            found.append(p)
    return found


def _row(clip, bundle, label, window, measured, verified, status, reason,
         claims) -> dict:
    return {"clip": clip, "bundle": bundle, "claims": label, "window": window or {},
            "measured": measured, "verified": bool(verified), "status": status,
            "reason": reason,
            "claim_fails": [f"{c['kind']}:{c['claim']}" for c in claims
                            if c["status"] == "fail"],
            "unverifiable_claims": [f"{c['kind']}: {c['why']}" for c in claims
                                    if c["status"] == "skip"]}


def _rows_for(rec: dict) -> list[dict]:
    """One row per artifact a bundle covers (main clip + slow-mo companion)."""
    ver = rec["verification"] or {}
    primary = [c for c in rec["claims"] if c.get("clip") in (None, rec["clip"])]
    rows = [_row(rec["path"], rec["bundle"], rec["label"], rec["window"],
                 {**(ver.get("measured") or {}),
                  "run_s": rec["facts"].get("run_s"),
                  "falls": rec["facts"].get("falls"),
                  "steps": rec["facts"].get("steps"),
                  "pushes": rec["facts"].get("n_pushes")},
                 ver.get("ok"), rec["verdict"]["status"], rec["verdict"]["reason"],
                 primary)]
    for sec in rec.get("secondary") or []:
        sver = sec["verification"] or {}
        fails = [c for c in sec["claims"] if c["status"] == "fail"]
        status = ("evidence" if (sver.get("ok") and not fails)
                  else ("verification-failed" if not sver.get("ok") else "claims-failed"))
        reason = ("; ".join(sver.get("problems") or [])
                  or "; ".join(f"{c['kind']}: {c['claim']} -> {c['why']}" for c in fails)
                  or "verification and all numeric claims pass")
        rows.append(_row(sec["path"], rec["bundle"], sec.get("label"), sver.get("window"),
                         sver.get("measured") or {}, sver.get("ok"), status, reason,
                         sec["claims"]))
    return rows


def build_table(include_unbundled: bool = True) -> dict:
    rows = []
    for b in _bundle_files():
        rec = check_bundle(b, verify=True)
        rows += _rows_for(rec)
    unbundled = []
    if include_unbundled:
        for p in _unbundled_videos():
            ver = sdr.verify_clip(p)                    # format + pixels only
            unbundled.append({
                "clip": str(p.relative_to(REPO)), "bundle": None,
                "measured": ver.get("measured") or {},
                "verified": bool(ver.get("ok")),
                "status": "unbundled (format+pixel only)",
                "reason": "no per-clip bundle/trace: no claim corroboration; "
                          + ("format/pixel checks pass" if ver.get("ok")
                             else f"format/pixel problems: {ver.get('problems')}"),
            })
    return {"generated": TODAY, "gate": gate(rows), "rows": rows, "unbundled": unbundled}


def gate(rows: list[dict]) -> dict:
    bad = [r["clip"] for r in rows if r["status"] != "evidence"]
    return {"ok": not bad, "not_evidence": bad,
            "n_evidence": sum(1 for r in rows if r["status"] == "evidence"),
            "n_rows": len(rows)}


def render_table(doc: dict) -> str:
    lines = ["# Evidence status (machine-generated by scripts/evidence_check.py table)",
             "",
             f"Generated: {doc['generated']} · gate: "
             f"{'PASS' if doc['gate']['ok'] else 'FAIL'} "
             f"({doc['gate']['n_evidence']}/{doc['gate']['n_rows']} clips are evidence)",
             "",
             "| clip | claims (label) | window | measured | verified | status | reason |",
             "|---|---|---|---|---|---|---|"]
    for r in doc["rows"]:
        w = r["window"]
        w_s = (f"t={w.get('t0')}..{w.get('t1_effective')} "
               f"speed {w.get('speed')}x" if w else "—")
        m = r["measured"]
        m_s = (f"{m.get('duration_s')} s / {m.get('frames')} f / {m.get('size')} "
               f"| run {r['measured'].get('run_s')} s, falls {r['measured'].get('falls')}, "
               f"steps {r['measured'].get('steps')}")
        lines.append(f"| `{r['clip']}` | {str(r['claims'])[:90]} | {w_s} | {m_s} | "
                     f"{'yes' if r['verified'] else 'NO'} | {r['status']} | "
                     f"{r['reason'][:160]} |")
    if doc["unbundled"]:
        lines += ["", "## Unbundled clips (no bundle/trace: format+pixel checks only)", "",
                  "| clip | measured | format+pixels | reason |", "|---|---|---|---|"]
        for r in doc["unbundled"]:
            m = r["measured"]
            m_s = f"{m.get('duration_s')} s / {m.get('frames')} f / {m.get('size')}"
            lines.append(f"| `{r['clip']}` | {m_s} | "
                         f"{'pass' if r['verified'] else 'FAIL'} | {r['reason']} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------- refresh
def refresh_bundle(bundle_path: Path, write: bool = False) -> dict:
    """Re-verify + refresh stale label/render fields from the spec table."""
    bundle = load_bundle(bundle_path)
    name = Path(str(bundle.get("video", ""))).name
    path = Path(bundle.get("video") or "")
    if not path.is_absolute():
        path = REPO / path
    before = check_bundle(bundle_path, verify=True)
    spec = spec_for(name)
    changes = []
    if spec is not None:
        if bundle.get("label") != spec["title"] or bundle.get("caption") != spec["caption"]:
            bundle.setdefault("label_history", []).append(
                {"label": bundle.get("label"), "caption": bundle.get("caption"),
                 "replaced": f"{TODAY} by EvidenceFix"})
            bundle["label"] = spec["title"]
            bundle["caption"] = spec["caption"]
            changes.append("label+caption refreshed from CLIP_SPECS")
            bundle["label_correction"] = (
                f"{TODAY}: the label/caption above were refreshed from the renderer's "
                "spec table (CLIP_SPECS in scripts/solo_drill_render.py); the previous "
                "text is kept in label_history. Note: the artifact's burned-in HUD text "
                "was rendered from the OLD text and cannot be changed without a "
                "re-render.")
    elif name == "final_L2_motion.mp4" and not bundle.get("label"):
        # the deliver bundle predates label/caption recording; the renderer's
        # constants are the single source for the claim text
        bundle["label"] = sdr.DELIVER_LABEL
        bundle["caption"] = sdr.DELIVER_CAPTION
        changes.append("label+caption backfilled from DELIVER_* constants")
    if name == "final_L2_motion.mp4" and "0.30 x 0.10" in str(bundle.get("stance_note", "")):
        # the deliver render ran without --stance-w, so the footer printed the
        # DELIVER_SPEC default; the shipped clip is the M_E28f run (0.28 x 0.10)
        bundle["stance_note"] = str(bundle["stance_note"]).replace(
            "stance 0.30 x 0.10 m", "stance 0.28 x 0.10 m")
        bundle["stance_note_correction"] = (
            f"{TODAY}: stance corrected 0.30 -> 0.28 m. The 0.30 value was the "
            "deliver-stage default printed because the render ran without "
            "--stance-w; the shipped clip is M_E28f_feasible_L3_seed0, the "
            "0.28 x 0.10 m run (reports/2026-10-08/drill_motion.md sec 5.1(5)); "
            "the trace's measured mean lateral sole separation is 0.289 m "
            "(rubric A1).")
        changes.append("stance_note corrected to the run's own 0.28 m")
    win = recover_window(name, bundle)
    if win and not (bundle.get("render") or {}):
        bundle["render"] = {"t0": win["t0"], "t1": win["t1"], "speed": win["speed"],
                            "scale": spec["scale"] if spec else 1.0, "fps": sdr.FPS,
                            "frames": (probe(path).get("nb_frames") if path.exists()
                                       else None),
                            "window_source": win["source"],
                            "backfilled": f"{TODAY} (window not recorded at render time; "
                                          f"recovered from {win['source']}; corroborated "
                                          "by the artifact duration)"}
        changes.append("render block backfilled")
    slow = bundle.get("slowmo")
    if slow and not (bundle.get("slowmo_render") or {}):
        sname = Path(str(slow)).name
        swin = recover_window(sname, bundle)
        if swin:
            spath = Path(str(slow))
            sver = (verify_bundle(sname, bundle, spath) if spath.exists() else None)
            fit = (sver or {}).get("window") or swin
            bundle["slowmo_render"] = {
                "t0": fit.get("t0"), "t1": fit.get("t1"),
                "speed": fit.get("speed", 0.25), "scale": 0.5, "fps": sdr.FPS,
                "frames": fit.get("expect_frames"), "window": fit,
                "caption": sdr.DELIVER_SLOWMO_CAPTION,
                "window_source": swin["source"],
                "backfilled": f"{TODAY} (window not recorded at render time; recovered "
                              f"from {swin['source']}; corroborated by the artifact "
                              "duration)"}
            changes.append("slowmo render block backfilled")
    ver = before["verification"]
    old = bundle.get("clip_verification") or {}
    if old.get("ok") is not True or old.get("problems"):
        hist = bundle.setdefault("clip_verification_history", [])
        if old:
            hist.append({"recorded_before": f"{TODAY}",
                         "ok": old.get("ok"), "problems": old.get("problems"),
                         "note": "the pre-fix verifier bug (int(inf) on imageio's "
                                 "nframes=inf) recorded false failures; the guard "
                                 "landed in commit 103dcee"})
        bundle["clip_verification"] = ver
        changes.append("clip_verification re-run")
    if spec and bundle.get("render_command") and "--t0" not in str(bundle["render_command"]):
        # the old command omitted the window: running it would reproduce a
        # DIFFERENT artifact than the one shipped
        bundle["render_command"] = sdr._render_command(bundle.get("trace_npz"), name, spec)
        changes.append("render_command corrected (window/speed included)")
    claims_skips = [c for c in before["claims"] if c["status"] == "skip"]
    # candidate verdict after the refresh (claims re-checked on the new text)
    after = dict(before)
    if changes:
        tmp = REPO / "data" / "_tmp_evidence_refresh.json"
        tmp.write_text(json.dumps(bundle, indent=1, default=str))
        after = check_bundle(tmp, verify=True)
        tmp.unlink(missing_ok=True)
        bundle["evidence"] = after["verdict"]["status"] == "evidence"
        bundle["claim_check"] = {
            "checked": f"{TODAY}",
            "status": after["verdict"]["status"],
            "fails": [f"{c['kind']}: {c['claim']} ({c['why']})"
                      for c in after["claims"] if c["status"] == "fail"],
            "unverifiable": [f"{c['kind']}: {c['why']}" for c in claims_skips],
        }
    if write and changes:
        bundle_path.write_text(json.dumps(bundle, indent=1, default=str))
    return {"bundle": str(bundle_path.relative_to(REPO)), "clip": name,
            "changes": changes, "before": before["verdict"],
            "after": after["verdict"], "claims": after["claims"]}


# -------------------------------------------------------------------- CLI
def cmd_table(a) -> int:
    doc = build_table(include_unbundled=not a.bundled_only)
    text = render_table(doc)
    print(text)
    if a.json:
        Path(a.json).write_text(json.dumps(doc, indent=1, default=str))
        print(f"wrote {a.json}")
    return 0 if doc["gate"]["ok"] or a.allow_failures else 1


def cmd_claims(a) -> int:
    bad = 0
    for b in _bundle_files():
        rec = check_bundle(b, verify=False)
        fails = [c for c in rec["claims"] if c["status"] == "fail"]
        skips = [c for c in rec["claims"] if c["status"] == "skip"]
        bad += bool(fails)
        print(f"== {rec['clip']}")
        for c in rec["claims"]:
            print(f"   [{c['status']:4s}] {c['kind']:14s} {c['claim'][:44]:44s} "
                  f"measured={json.dumps(c['measured'], default=str)[:70]}"
                  + (f"  <- {c['why']}" if c["why"] else ""))
        if not rec["claims"]:
            print("   (no numeric claims)")
        if skips:
            print(f"   {len(skips)} claim(s) not mechanically checkable: "
                  + "; ".join(sorted({c["kind"] for c in skips})))
    print(f"\n{bad} bundle(s) with failing claims")
    return 1 if bad and not a.allow_failures else 0


def cmd_refresh(a) -> int:
    bad = 0
    for b in _bundle_files():
        res = refresh_bundle(b, write=a.write)
        flag = "" if res["after"]["status"] == "evidence" else f"  <-- {res['after']['status']}"
        print(f"== {res['clip']}{flag}")
        if res["changes"]:
            print(f"   changes: {', '.join(res['changes'])}"
                  + ("  (written)" if a.write else "  (dry run)"))
        print(f"   before: {res['before']['status']}: {res['before']['reason'][:110]}")
        print(f"   after : {res['after']['status']}: {res['after']['reason'][:110]}")
        bad += res["after"]["status"] != "evidence"
    print(f"\n{bad} bundle(s) not evidence after refresh")
    return 1 if bad and not a.allow_failures else 0


def cmd_gate(a) -> int:
    doc = build_table(include_unbundled=False)
    ok = doc["gate"]["ok"]
    print(json.dumps({"gate": doc["gate"]}, indent=1))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("table", help="machine-generated evidence status table")
    t.add_argument("--json", default="")
    t.add_argument("--bundled-only", action="store_true")
    t.add_argument("--allow-failures", action="store_true")
    t.set_defaults(func=cmd_table)

    c = sub.add_parser("claims", help="label-integrity check only")
    c.add_argument("--allow-failures", action="store_true")
    c.set_defaults(func=cmd_claims)

    r = sub.add_parser("refresh", help="re-verify + refresh stale bundle fields")
    r.add_argument("--write", action="store_true")
    r.add_argument("--allow-failures", action="store_true")
    r.set_defaults(func=cmd_refresh)

    g = sub.add_parser("gate", help="exit nonzero unless every bundled clip is evidence")
    g.set_defaults(func=cmd_gate)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
