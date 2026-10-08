"""Discrimination tests for the S5 executability checker (``solo.exec_check``).

The checker decides whether a captured penetration-step demo is *executable*:
no fall, the lead foot plants then the knee reaches the mat then the robot
rises, the torso tracks the reference, the CoM stays inside the support, the
knee actually reaches the reference depth, no joint saturation, no foot slide
while loaded, the reference's travel is retained, and the demo ENDS in a valid
stance.

These tests are synthetic (no sim, no model).  The POSITIVE control is a
hand-constructed trace that satisfies all nine criteria; each rejection case
perturbs exactly one thing and asserts the matching criterion fires — so a
checker that rejects everything (or accepts everything) fails here.

Report: ``reports/2026-10-08/s5_demo_prep.md``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from solo.exec_check import (  # noqa: E402
    SEG_BLEND, SEG_ENTRY, SEG_HOLD, SEG_RECOVER, SEG_STAND, ExecSpec, RefTracks,
    check_executability)

N = 300
DT = 0.02
T = np.arange(N) * DT
SOLE_REST_Z = 0.0031


def _segments() -> np.ndarray:
    seg = np.full(N, SEG_ENTRY, np.int8)
    seg[80:90] = SEG_BLEND
    seg[90:220] = SEG_RECOVER
    seg[220:240] = SEG_STAND
    seg[240:] = SEG_HOLD
    return seg


def _ref_time(seg: np.ndarray) -> np.ndarray:
    """The demo's reference clock: the entry runs on the reference's own time,
    the recover continues it (the blend/hold have no reference)."""
    rt = np.full(N, np.nan)
    rt[seg == SEG_ENTRY] = T[seg == SEG_ENTRY]
    rt[seg == SEG_RECOVER] = 1.60 + (T[seg == SEG_RECOVER] - 1.80)
    return rt


def _spec(seg: np.ndarray, rt: np.ndarray) -> ExecSpec:
    tracks = RefTracks(
        grid_t=T.copy(), ref_time=T.copy(), segment=seg.copy(),
        phase=np.full(N, "LOW"), pitch_deg=np.full(N, 45.0),
        margin=np.full(N, 0.02), contact=np.ones((N, 2), bool),
        knee_z=np.tile(np.array([0.30, 0.40]), (N, 1)),
        pelvis_z=np.full(N, 0.60), base_xy=np.zeros((N, 2)),
        knee_site_xy=np.zeros((N, 2, 2)))
    return ExecSpec(
        technique="shot_entry", dt=DT, lead_side="right", lead_plant_t=0.10,
        knee_side="left", knee_min_z=0.30, knee_min_t=0.30,
        knee_contact_z=0.35, knee_contact_t=0.30, travel_forward_m=0.0,
        rise_pelvis_z=0.55, rise_knee_z=0.20, stance_pelvis_z=0.79,
        stance_pelvis_lo=0.70, stance_pelvis_hi=0.80, stance_tilt_max_deg=6.0,
        stance_margin_min=0.02, stance_speed_max=0.04, stance_flat_tol_m=0.01,
        sole_rest_z=SOLE_REST_Z, tracks=tracks)


def _foot(cx: float, cy: float, z: float) -> np.ndarray:
    """One foot's four sole-sphere centres (a real footprint, not a point)."""
    return np.array([[cx - 0.05, cy - 0.09, z], [cx + 0.05, cy - 0.09, z],
                     [cx + 0.05, cy + 0.09, z], [cx - 0.05, cy + 0.09, z]])


def _good_trace() -> dict:
    seg = _segments()
    rt = _ref_time(seg)
    knee_z = np.tile(np.array([0.40, 0.40]), (N, 1))
    # the penetration knee (left) descends to the reference depth at t=0.30 s
    knee_z[15:220, 0] = 0.30
    knee_z[220:, 0] = 0.40
    contact_foot = np.ones((N, 2), bool)
    contact_foot[:5, 1] = False              # the lead (right) foot is airborne
    pelvis_z = np.full(N, 0.60)
    pelvis_z[120:] = np.linspace(0.60, 0.79, N - 120)
    tilt = np.full(N, 25.0)
    tilt[120:] = np.linspace(25.0, 0.5, N - 120)
    sole = np.zeros((N, 2, 4, 3))
    sole[:, 0] = _foot(0.0, 0.0, SOLE_REST_Z)        # stance foot, under the CoM
    sole[:, 1] = _foot(0.0, -0.25, SOLE_REST_Z)      # lead foot, away from the CoM
    sole[:5, 1, :, 2] = 0.05                         # the airborne lead foot
    return {
        "t": T.copy(), "ref_time": rt, "segment": seg,
        "qpos": np.zeros((N, 36)), "qvel": np.zeros((N, 35)),
        "act_force": np.zeros((N, 29)),
        "jnt_range": np.tile([-1.0, 1.0], (29, 1)), "act_limit": np.ones(29),
        "contact_foot": contact_foot,
        "contact_knee": np.zeros((N, 2), bool),
        "contact_hand": np.zeros((N, 2), bool),
        "grounded": np.zeros(N, bool), "sole_pts": sole, "knee_z": knee_z,
        "com": np.tile([0.0, 0.0, 0.69], (N, 1)),
        "pelvis_z": pelvis_z, "tilt_deg": tilt, "pitch_deg": np.full(N, 45.0),
        "margin": np.full(N, 0.02),
    }


def _failed(rep) -> set[str]:
    return {c.name for c in rep.criteria if not c.ok}


def test_known_good_trace_passes_every_criterion():
    seg = _segments()
    rep = check_executability(_good_trace(), _spec(seg, _ref_time(seg)))
    assert rep.ok, rep.lines()
    assert _failed(rep) == set()


def test_fall_is_rejected():
    seg = _segments()
    tr = _good_trace()
    tr["grounded"][-30:] = True              # on the mat at the end
    rep = check_executability(tr, _spec(seg, _ref_time(seg)))
    assert not rep.ok
    assert "no_fall" in _failed(rep)


def test_foot_slide_is_rejected():
    seg = _segments()
    tr = _good_trace()
    # 30 mm of loaded drift on the stance foot (stays under the CoM, so the
    # criterion's "loaded" test keeps arming while it slides)
    tr["sole_pts"][150:170, 0, :, 0] += np.linspace(0.0, 0.03, 20)[:, None]
    rep = check_executability(tr, _spec(seg, _ref_time(seg)))
    assert not rep.ok
    assert "no_foot_slide" in _failed(rep)


def test_torso_collapse_is_rejected():
    seg = _segments()
    tr = _good_trace()
    tr["pitch_deg"][20:60] = 90.0            # collapses away from the 45 deg ref
    rep = check_executability(tr, _spec(seg, _ref_time(seg)))
    assert not rep.ok
    assert "torso_pitch" in _failed(rep)


def test_knee_depth_miss_is_rejected():
    seg = _segments()
    tr = _good_trace()
    tr["knee_z"][:, 0] = 0.45                # never reaches the reference depth
    rep = check_executability(tr, _spec(seg, _ref_time(seg)))
    assert not rep.ok
    assert "knee_depth" in _failed(rep)


def test_missing_lead_plant_is_rejected():
    seg = _segments()
    tr = _good_trace()
    tr["contact_foot"][:, 1] = True          # no airborne->plant edge at all
    rep = check_executability(tr, _spec(seg, _ref_time(seg)))
    assert not rep.ok
    assert "contact_sequence" in _failed(rep)


def test_terminal_stance_is_required():
    seg = _segments()
    tr = _good_trace()
    tr["pelvis_z"][-20:] = 0.55              # ends crouched, not in stance
    tr["tilt_deg"][-20:] = 30.0
    rep = check_executability(tr, _spec(seg, _ref_time(seg)))
    assert not rep.ok
    assert "terminal_stance" in _failed(rep)


def test_captured_demo_exports_as_a_loadable_reference(tmp_path):
    """The captured demo must BE the refinement's reference, not need a conversion.

    The refinement stage (``scripts/solo_imitation_train.py``) conditions on a
    ``solo.bc.ReferenceTrack``; the capture used to write only ``trace.npz``,
    whose key names ``load_reference`` does not read -- so the whole leg would
    have failed only after a two-hour capture.  This pins the contract: the
    exported arrays load through ``load_reference`` with the trace's own qpos/t
    and the provenance survives.
    """
    from solo.bc import load_reference
    from solo.demo import reference_arrays

    seg = _segments()
    tr = _good_trace()
    tr["qpos"] = np.zeros((N, 36), np.float64)
    tr["qpos"][:, 2] = tr["pelvis_z"]
    tr["t"] = T
    meta = {"source": "solo_demo_capture:cem", "name": "t", "episode": 0,
            "seed": 0, "n_ticks": N, "terminated_t": None}
    out = reference_arrays(tr, meta)
    assert set(out) == {"qpos_a", "t", "meta"}
    p = tmp_path / "reference.npz"
    np.savez_compressed(p, **out)
    track = load_reference(p)
    assert track.qpos.shape == (N, 36)
    assert np.allclose(track.qpos[:, 2], tr["pelvis_z"])
    assert np.allclose(track.t, T)
    assert track.meta["source"] == "solo_demo_capture:cem"
    assert track.meta["n_ticks"] == N
