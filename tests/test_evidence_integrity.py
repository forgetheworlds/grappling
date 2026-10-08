"""Evidence-integrity tests: the 2026-10-08 clip audit, pinned as regressions.

Each test reconstructs an audited defect and asserts the NEW checks FAIL on it
(report: ``reports/2026-10-08/evidence_pipeline.md``; prior audit:
``reports/2026-10-08/clip_wiring_audit.md``).  The defects pinned here:

* **D1** 12/13 bundles recorded ``clip_verification.ok=false`` ("cannot convert
  float infinity to integer" -- the pre-``103dcee`` verifier bug) while the
  index claimed the clips were verified -> ``test_gate_refuses_unverified_clip``
* **D2** the index claimed 95 s / 6 steps / 0.30 m for ``final_L2_motion.mp4``
  while the shipped M_E28f trace has 69.98 s / 5 steps / 0.289 m
  -> ``test_label_step_and_duration_claims_fail``
* **D3** ``02_slowmo_level_change_quarter_speed.mp4`` claims 0.25x but was
  rendered at 1.0x (3.033 s for a 3.0 s window)
  -> ``test_slowmo_duration_ratio_must_match_the_claimed_speed``
* **D4** ``final_L2_entry_walk.mp4``'s caption claims a "0.25 m wide" base
  while the stepping base is 0.21 m (measured 0.212 m)
  -> ``test_stance_width_claim_fails``
* **D5** ``L2_shuffle_3steps_diag``'s label claimed "2-3 steps" while the trace
  completes 4 -> ``test_step_range_claim_fails``
* **D6** the F1 caption claimed "it topples" for a run with ``falls=0``
  -> ``test_topple_claim_vs_trace_falls``
* **D7** the t=0-60 s push clip cannot show the t=62 s push while its caption
  claims all three -> ``test_push_outside_window_must_be_acknowledged``
* **D8** three complete clips were recorded as "truncated mid-write" because
  the old expectation was (t1 - t0) and ignored the trace's own end
  -> ``test_window_expectation_clamps_at_trace_end``
* **D9** the verifier never ran to completion (the imageio ``nframes=inf``
  path), so no clip was machine-corroborated
  -> ``test_verify_clip_completes_and_fails_static_or_black``
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import evidence_check as ec  # noqa: E402
import solo_drill_render as sdr  # noqa: E402


# ------------------------------------------------------------------ helpers
def facts(**kw) -> dict:
    base = dict(clip_s=None, run_s=None, falls=None, steps=None, steps_summary=None,
                n_pushes=None, pushes_recovered=None, push_times=[], push_forces=[],
                stance_w_measured=None, stance_d_measured=None, trace_end_s=None,
                trace_npz=None, clip_path=None)
    base.update(kw)
    return base


def window(t0: float, t1: float | None, speed: float = 1.0,
           trace_end: float | None = None) -> dict:
    t1e = min(t1, trace_end) if (t1 is not None and trace_end is not None) else t1
    return {"t0": t0, "t1": t1, "t1_effective": t1e, "speed": speed,
            "trace_end_s": trace_end,
            "clip_window_s": round(t1e - t0, 3) if t1e is not None else None}


def fails(checks: list[dict]) -> set[str]:
    return {c["kind"] for c in checks if c["status"] == "fail"}


def write_mp4(path: Path, n: int = 31, fps: int = 30, moving: bool = True) -> Path:
    import imageio.v2 as imageio

    w = imageio.get_writer(str(path), fps=fps, quality=8,
                           macro_block_size=None, mode="I")
    try:
        for i in range(n):
            frame = np.full((48, 64, 3), 120, np.uint8)
            if moving:
                frame[12:36, 4 + (i % 24):16 + (i % 24)] = 240
            w.append_data(frame)
    finally:
        w.close()
    return path


# ------------------------------------------------------- D1: the gate
def test_gate_refuses_unverified_clip():
    """D1: a bundle with ok:false must NOT be presentable as evidence."""
    stale = {"ok": False,
             "problems": ["decode failed: cannot convert float infinity to integer"]}
    verdict = ec.evidence_verdict(stale, [])
    assert verdict["ok"] is False
    assert verdict["status"] == "verification-failed"
    assert "infinity" in verdict["reason"]
    # ... and a passing verification with a failed claim is refused too
    claim = ec._res("steps", "6 steps", 6, False, 5, "trace has 5 steps")
    assert ec.evidence_verdict({"ok": True, "problems": []}, [claim])["status"] == \
        "claims-failed"
    # positive control: the same clip passes once both halves pass
    assert ec.evidence_verdict({"ok": True, "problems": []}, [])["status"] == "evidence"


# ------------------------------------------- D2/D3/D4/D5/D6/D7: the claims
def test_label_step_and_duration_claims_fail():
    """D2: the superseded-run numbers (95 s / 6 steps) must fail; 0.30 m is a
    documented blind spot of the 30 mm stance tolerance."""
    f = facts(clip_s=70.0, run_s=69.98, falls=0, steps=5, steps_summary=5,
              stance_w_measured=0.289, stance_d_measured=0.135)
    good = ec.check_text_claims(
        "70.000 s / 2100 frames, 0 falls, 5 steps in a 0.28 m-wide stance", f)
    assert not fails(good), good
    bad = ec.check_text_claims(
        "95 s, 0 falls, 6 gate-checked steps in a 0.30 m-wide stance", f)
    assert {"duration", "steps"} <= fails(bad), bad
    # honest coverage: 0.30 vs measured 0.289 is inside the 30 mm tolerance, so
    # the stance check alone does not catch this drift (the report says so)
    assert "stance_w" not in fails(bad)


def test_slowmo_duration_ratio_must_match_the_claimed_speed():
    """D3: 0.25x claimed, 1.0x rendered -> fail; 4x ratio -> ok."""
    bad = ec.check_text_claims("slow motion 0.25x: level change down / hold / rise",
                               facts(clip_s=3.033), window(3.0, 6.0, speed=1.0))
    assert "speed" in fails(bad), bad
    assert "ratio" in next(c["why"] for c in bad if c["kind"] == "speed")
    good = ec.check_text_claims("slow motion 0.25x",
                                facts(clip_s=13.633),
                                window(3.0, 6.4, speed=0.25))
    assert "speed" not in fails(good), good


def test_stance_width_claim_fails():
    """D4: the entry-walk caption's 0.25 m base against the measured 0.212 m."""
    f = facts(stance_w_measured=0.212, stance_d_measured=0.067)
    bad = ec.check_text_claims(
        "the entry walk in the base the primitive can use (0.25 m wide, 0.06 m deep)", f)
    assert "stance_w" in fails(bad), bad
    good = ec.check_text_claims(
        "the entry walk in the base the primitive can use (0.21 m wide, 0.06 m deep)", f)
    assert not fails(good), good


def test_step_range_claim_fails():
    """D5: "2-3 steps" against a run that completes 4."""
    bad = ec.check_text_claims("L2: short shuffle (2-3 steps)", facts(steps=4))
    assert "steps_range" in fails(bad), bad
    good = ec.check_text_claims("L2: short shuffle (4 consecutive steps)",
                                facts(steps=4))
    assert not fails(good), good


def test_topple_claim_vs_trace_falls():
    """D6 (F1): "it topples" against falls=0 fails; the corrected text passes."""
    bad = ec.check_text_claims(
        "the same stance pose under pure position control: it topples", facts(falls=0))
    assert "falls" in fails(bad), bad
    good = ec.check_text_claims(
        "stays up for 12 s (does NOT topple - the toppling baseline is the "
        "separate run)", facts(falls=0, clip_s=12.0, run_s=11.98))
    assert not fails(good), good
    real = ec.check_text_claims(
        "with no feedback the robot TOPPLES at 8.4 s", facts(falls=1, clip_s=8.4))
    assert not fails(real), real


def test_push_outside_window_must_be_acknowledged():
    """D7: the t=62 s push cannot be shown in a t=0-60 s clip unless the text says so."""
    f = facts(clip_s=60.033, run_s=89.98, trace_end_s=89.98, n_pushes=3,
              pushes_recovered=3, push_times=[15.0, 38.0, 62.0],
              push_forces=[20.0, 20.0, 20.0])
    w = window(0.0, 60.0, trace_end=89.98)
    old = ec.check_event_coverage(
        "3 x 20 N pushes (all recovered): CoM, margin and the push windows are overlaid",
        f, w)
    assert [c for c in old if c["status"] == "fail"], old
    new = ec.check_event_coverage(
        "two of the three 20 N pushes (t=15 s, t=38 s) are inside this 0-60 s clip "
        "and recover; the third (t=62 s) is outside it - the run's metrics record 3/3",
        f, w)
    assert not fails(new), new           # acknowledged -> not a failure


# ------------------------------------------------ D8: window expectations
def test_window_expectation_clamps_at_trace_end(tmp_path):
    """D8: expectations come from the trace+window; the naive (t1-t0) said three
    complete clips were 'truncated mid-write'."""
    tr = tmp_path / "trace.npz"
    np.savez(tr, t=np.array([0.0, 1.0, 2.82]))
    exp = sdr.window_expectations(tr, 1.0, 12.0)        # episode ends at 2.82 s
    assert exp["expect_frames"] == 56                   # 99_failure_entry_L2's artifact
    assert exp["expect_s"] == pytest.approx(1.867, abs=0.002)
    assert (12.0 - 1.0) - exp["expect_s"] > 9           # what the old check used

    tr2 = tmp_path / "trace2.npz"
    np.savez(tr2, t=np.array([0.0, 89.98]))
    exp2 = sdr.window_expectations(tr2, 0.0, 60.0)
    assert (exp2["expect_frames"], exp2["expect_s"]) == (1801, 60.033)

    tr3 = tmp_path / "trace3.npz"
    np.savez(tr3, t=np.array([0.0, 19.98]))
    exp3 = sdr.window_expectations(tr3, 0.0, 8.06, speed=0.25)
    # 0.25x: the sampling rate carries the slowdown, frames = window/speed*fps
    assert (exp3["expect_frames"], exp3["expect_s"]) == (968, 32.267)


# --------------------------------------------- D9: the verifier itself
def test_verify_clip_completes_and_fails_static_or_black(tmp_path):
    """D9: the pixel half must RUN (imageio reports nframes=inf for these mp4s)
    and must flag static and black renders, wrong fps and a wrong duration."""
    import imageio.v2 as imageio

    moving = write_mp4(tmp_path / "moving.mp4", n=31, fps=30, moving=True)
    ver = sdr.verify_clip(moving, expect_s=31 / 30, expect_frames=31)
    assert ver["ok"], ver["problems"]
    assert ver["mean_luma"] and len(ver["frame_diff"]) >= 1   # the check RAN
    assert ver["measured"]["frames"] == 31
    nf = imageio.get_reader(str(moving)).get_meta_data().get("nframes")
    assert nf is None or not np.isfinite(float(nf)) or float(nf) == 31.0

    static = write_mp4(tmp_path / "static.mp4", n=12, fps=30, moving=False)
    ver_static = sdr.verify_clip(static)                 # no expectations: pixels only
    assert not ver_static["ok"]
    assert any("static" in p for p in ver_static["problems"]), ver_static

    black = tmp_path / "black.mp4"
    w = imageio.get_writer(str(black), fps=30, quality=8, macro_block_size=None,
                           mode="I")
    try:
        for _ in range(12):
            w.append_data(np.zeros((48, 64, 3), np.uint8))
    finally:
        w.close()
    ver_black = sdr.verify_clip(black)
    assert any("near-black" in p for p in ver_black["problems"]), ver_black

    wrong_fps = write_mp4(tmp_path / "slow.mp4", n=6, fps=10)
    ver_fps = sdr.verify_clip(wrong_fps)                 # fps defaults to 30
    assert any("frame rate" in p for p in ver_fps["problems"]), ver_fps

    ver_dur = sdr.verify_clip(moving, expect_s=99.0, expect_frames=31)
    assert any("duration" in p for p in ver_dur["problems"]), ver_dur
    ver_fr = sdr.verify_clip(moving, expect_s=31 / 30, expect_frames=999)
    assert any("frames" in p for p in ver_fr["problems"]), ver_fr


# ------------------------------------------- end-to-end: bundle verdict
def test_visuals_index_paths_exist():
    """The `_diag` rename broke two index paths (the audit's mode B).  Every
    path in the VISUALS row-10b brace list must exist, and the renderer's
    naming rule must map specs to the names the index uses."""
    text = (REPO / "docs" / "VISUALS.md").read_text()
    line = next((l for l in text.splitlines() if "10b" in l), None)
    if not line or not re.search(r"videos/solo_drill/\{[^}]+\}", line):
        pytest.skip("VISUALS 10b index shape changed")
    block = re.search(r"videos/solo_drill/\{([^}]+)\}", line).group(1)
    missing = []
    for name in block.split(","):
        p = REPO / "videos" / "solo_drill" / f"{name.strip()}.mp4"
        if not p.exists():
            missing.append(str(p.relative_to(REPO)))
    assert not missing, missing
    # the renderer's naming rule is the single source the index must follow
    assert sdr._final_name("L0_hold_30s.mp4", 0.5) == "L0_hold_30s_diag.mp4"
    assert sdr._final_name("L2_cycle_step_diag.mp4", 0.5) == "L2_cycle_step_diag.mp4"
    assert ec.spec_for("L0_hold_30s_diag.mp4") is not None
    assert ec.spec_for("99_failure_push90N_diag.mp4") is not None


def test_check_bundle_gates_a_claim_mismatch(tmp_path):
    """The whole chain on a synthetic bundle: verified clip + false step claim
    is NOT evidence; the same bundle with a true claim is."""
    mp4 = write_mp4(tmp_path / "clip.mp4", n=31, fps=30, moving=True)
    tr = tmp_path / "trace.npz"
    np.savez(tr, t=np.array([0.0, 1.0]))
    base = {"video": str(mp4), "trace_npz": str(tr),
            "metrics": {"falls": 0, "steps_completed": 5, "duration_s": 1.0},
            "config": {"pushes": []},
            "clip_verification": {"ok": False,
                                  "problems": ["decode failed: cannot convert "
                                               "float infinity to integer"]}}
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**base, "label": "6 gate-checked steps"}))
    rec = ec.check_bundle(bad, verify=True)
    assert rec["verification"]["ok"] is True             # media is fine now
    assert rec["verdict"]["status"] == "claims-failed"   # the claim is not
    assert "steps" in fails(rec["claims"])
    good = tmp_path / "good.json"
    good.write_text(json.dumps({**base, "label": "5 gate-checked steps"}))
    rec2 = ec.check_bundle(good, verify=True)
    assert rec2["verdict"]["status"] == "evidence", rec2["verdict"]
