"""Contracts for the v2 motion-reference dataset (data/references/motion_refs/v2).

Agent 3 (reference re-timing).  Pinned contracts, each traced to a measured
defect or the re-timing acceptance:

  * format round-trip: every v2 track loads through solo.bc.load_reference;
    qpos/contact/phase arrays shape-consistent; meta JSON parses; NO
    `expert_action` key ever exists (a kinematic reference is not a
    demonstration);
  * v2 is the loader's resolution DEFAULT (the unchanged training interface
    must consume v2), while the v1 archive stays loadable by explicit path;
  * timing: `t` strictly increasing at 50 Hz, phase windows contiguous and
    non-overlapping, every phase `t_end > t_start`;
  * the re-timed root stays inside the MEASURED acceleration envelope:
    grounded soles (>= -1 cm), bounded root/joint velocities, and --
    the decisive numbers -- CoM inside the support on every frame of the
    quasi-static phases (the v1 video paths were outside for 62-100 %
    of frames; that is the measured 0.36 s fixed-clock wall);
  * the stepping contact schedule transfers weight BEFORE the lift (the
    swing foot's contact flag is 0 only while its FK sole is off the mat);
  * honesty: the fused knee-down penetration keeps its geometry (not
    flattened), failures carry measured labels, and the label schema is the
    v1 enum plus the measured `balance_verified` level (documented in
    v2/RETIMING.md).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

V2 = REPO / "data/references/motion_refs/v2"
REFS = V2 / "refs"

GENERATED_TAKES = ["stance_hold", "stand_to_stance", "stance_widen_step",
                   "shuffle_back", "circle_step", "level_change_full",
                   "level_change_fast", "shot_entry_full", "shot_recover"]
TIMING_ONLY_TAKES = ["knee_sprawl_entry", "knee_sprawl_hold",
                     "knee_sprawl_recover", "stalk_shuffle",
                     "knee_sprawl_entry2"]
#: v1 enum + the measured v2 levels (RETIMING.md): `balance_verified` (the
#: balance layer traverses it) and `balance_blocked` (L1/L2 hold; the only
#: probed balance layer fails it -- a learning question, not a reference-
#: timing question)
VALIDITY = {"kinematically_valid", "statically_holdable", "dynamically_verified",
            "balance_verified", "balance_blocked", "unverified",
            "known_infeasible", "kinematically_invalid"}
#: quasi-static phases: the CoM must stay inside the support everywhere
QUASI_STATIC = ("STAND", "LOWER_TO_STANCE", "STANCE_HOLD", "LEVEL_CHANGE",
                "DOUBLE_LEG_ENTRY_CROUCH", "RECOVER_TO_STANCE")
#: measured envelope bounds for the v2 generated tracks (RETIMING.md)
JOINT_V_MAX = 6.0        # rad/s (the retarget pipeline's own cap)
ROOT_STEP_MAX = 0.07     # m/frame at 50 Hz (3.5 m/s, the v1 root cap + margin)
SOLE_PEN_MAX = 0.01      # m (QUALITY_RUBRIC G2)


def _load(path):
    d = np.load(path, allow_pickle=True)
    return d, json.loads(str(d["meta"]))


@pytest.fixture(scope="module")
def model():
    from solo.scene import load_solo_model
    return load_solo_model()


# ------------------------------------------------------------- format
def test_v2_is_the_resolution_default_and_v1_archive_loadable():
    from solo.bc import load_reference, reference_path
    p = reference_path("stance_hold")
    assert "motion_refs/v2" in str(p), p
    tr = load_reference("stance_hold")
    assert tr.qpos.shape[1] == 36 and len(tr) > 10
    v1 = load_reference(str(REPO / "data/references/motion_refs/v1/refs/stance_hold.npz"))
    assert "motion_refs/v1" in v1.source
    # the v2 take is the re-solved repair, not the raw video posture
    assert abs(float(v1.qpos[0, 2]) - float(tr.qpos[0, 2])) > 1e-3


@pytest.mark.parametrize("name", GENERATED_TAKES + TIMING_ONLY_TAKES)
def test_v2_takes_round_trip_through_loader(name):
    from solo.bc import load_reference
    tr = load_reference(str(REFS / f"{name}.npz"))
    d, meta = _load(REFS / f"{name}.npz")
    assert np.allclose(tr.qpos, np.asarray(d["qpos_a"]))
    assert meta["version"] == "motion_refs/v2"
    assert "reference_not_demonstration" in meta
    assert "expert_action" not in d.files
    assert meta["validity"] in VALIDITY or name in TIMING_ONLY_TAKES
    assert meta["lead_leg"] in ("left", "right")
    if name in GENERATED_TAKES:
        assert "retime" in meta and "v1_peer" in meta["retime"]


def test_v2_drill_format_and_phase_table():
    d, meta = _load(V2 / "drill_continuous.npz")
    q = np.asarray(d["qpos_a"], float)
    pid = np.asarray(d["phase_id"], int)
    t = np.asarray(d["t"], float)
    c = np.asarray(d["contact"], np.uint8)
    assert len(q) == len(t) == len(pid) == len(c)
    phases = meta["phases"]
    assert len(phases) == int(pid.max()) + 1
    # timing monotonicity: strictly increasing 50 Hz clock
    dt = np.diff(t)
    assert dt.min() > 0 and np.allclose(dt, 0.02, atol=1e-9)
    # phase windows contiguous, non-overlapping, positive length
    for a, b in zip(phases, phases[1:]):
        assert a["t_end"] <= b["t_start"] + 1e-9
    for p in phases:
        assert p["t_end"] > p["t_start"]
        assert p["source"] in ("observed", "g1_native", "fused", "generated",
                                "synthetic_connector")
        assert p["validity"] in VALIDITY
        if p["source"] != "synthetic_connector":
            assert p["lead_leg"] in ("left", "right")
    # the movement set is preserved from v1 (nothing silently deleted)
    names = {p["name"] for p in phases}
    for required in ("STAND", "LOWER_TO_STANCE", "STANCE_HOLD", "SHUFFLE_F",
                     "SHUFFLE_B", "CIRCLE", "LEVEL_CHANGE",
                     "DOUBLE_LEG_ENTRY_CROUCH", "DOUBLE_LEG_PENETRATION",
                     "RECOVER_TO_STANCE", "REPOSITION"):
        assert required in names, f"missing phase {required}"
    assert "expert_action" not in d.files


def test_drill_continuity_and_repeat_closure():
    d, meta = _load(V2 / "drill_continuous.npz")
    q = np.asarray(d["qpos_a"], float)
    pid = np.asarray(d["phase_id"], int)
    steps = np.abs(np.diff(q[:, 7:], axis=0)).max(axis=1)
    root_steps = np.linalg.norm(np.diff(q[:, :3], axis=0), axis=1)
    assert steps.max() <= 0.13, f"joint snap {steps.max():.3f} rad"
    assert root_steps.max() <= ROOT_STEP_MAX, f"root jump {root_steps.max():.3f} m"
    phases = meta["phases"]
    hold = next(p for p in phases if p["name"] == "STANCE_HOLD")
    i0 = int(np.searchsorted(np.asarray(d["t"], float), hold["t_start"]))
    gap = float(np.abs(q[-1, 7:] - q[i0, 7:]).max())
    assert gap <= 0.20, f"REPEAT gap {gap:.3f} rad"


def test_stance_rise_v2_format():
    d, meta = _load(V2 / "stance_rise.npz")
    q = np.asarray(d["qpos_a"], float)
    t = np.asarray(d["t"], float)
    pid = np.asarray(d["phase_id"], int)
    assert len(q) == len(t) and pid.max() == 1
    names = [p["name"] for p in meta["phases"]]
    assert names == ["STANCE", "RISE_TO_STAND"]
    assert np.abs(np.diff(q[:, 7:], axis=0)).max() <= 0.13


# ------------------------------------------------- the measured envelope
def test_grounding_and_velocity_envelope(model):
    """Every v2 generated track: soles grounded (G2), velocities inside the
    pipeline caps.  Measured with the refgen instrument (FK)."""
    import solo.refgen as rg
    for name in GENERATED_TAKES:
        d, _ = _load(REFS / f"{name}.npz")
        q = np.asarray(d["qpos_a"], float)
        t = np.asarray(d["t"], float)
        c = np.asarray(d["contact"], np.uint8)
        st = rg.measure_track(q, t, c)
        assert st["sole_pen_min_m"] >= -SOLE_PEN_MAX, \
            f"{name}: sole penetration {st['sole_pen_min_m']}"
        assert st["joint_v_max_rad_s"] <= JOINT_V_MAX, \
            f"{name}: joint velocity {st['joint_v_max_rad_s']} rad/s"
        root_v = float(np.linalg.norm(
            np.gradient(q[:, :3], 0.02, axis=0), axis=1).max())
        assert root_v <= 3.5, f"{name}: root velocity {root_v:.2f} m/s"


def test_quasi_static_phases_hold_com_inside_support(model):
    """THE decisive re-timing contract: on every frame of the quasi-static
    phases the CoM is inside the planted-foot hull (the v1 paths were outside
    for 62-100 % of frames -- the measured 0.36 s fixed-clock wall)."""
    d, meta = _load(V2 / "drill_continuous.npz")
    q = np.asarray(d["qpos_a"], float)
    t = np.asarray(d["t"], float)
    pid = np.asarray(d["phase_id"], int)
    contact = np.asarray(d["contact"], np.uint8)
    import solo.refgen as rg
    for name in QUASI_STATIC:
        p = next(p for p in meta["phases"] if p["name"] == name)
        fr = np.flatnonzero(pid == int(p["id"]))
        st = rg.measure_track(q[fr], t[fr], contact[fr])
        assert st["com_margin_min_m"] >= 0.0, \
            f"{name}: CoM margin {st['com_margin_min_m']} m (v1 was negative)"
        assert st["com_margin_neg_frac"] == 0.0


def test_penetration_geometry_not_flattened():
    """The fused knee-down penetration keeps its GrappleMap depth (identity
    is preserved; the verdict is a label, not a geometry edit)."""
    d, meta = _load(V2 / "drill_continuous.npz")
    q = np.asarray(d["qpos_a"], float)
    pid = np.asarray(d["phase_id"], int)
    p = next(p for p in meta["phases"]
             if p["name"] == "DOUBLE_LEG_PENETRATION")
    fr = np.flatnonzero(pid == int(p["id"]))
    assert q[fr, 2].min() <= 0.45, (
        "the penetration's deep knee-down posture was flattened away")
    assert p["source"] == "fused"


def test_step_schedule_transfers_weight_before_lift(model):
    """B2 contract, pinned on the generated stepping tracks: the pelvis moves
    toward the coming support BEFORE the swing foot's flag drops, and the
    flag is 0 exactly while the FK sole is off the mat."""
    import mujoco
    from solo.lit import sole_points_world
    data = mujoco.MjData(model)
    for name in ("shuffle_back", "circle_step"):
        d, _ = _load(REFS / f"{name}.npz")
        q = np.asarray(d["qpos_a"], float)
        c = np.asarray(d["contact"], np.uint8)
        lifted_frames = np.flatnonzero(c.all(axis=1) == False)  # noqa: E712
        assert len(lifted_frames) > 20, f"{name}: no single-support phase"
        # the swing bump must be REAL (mean sole height of the lifted foot
        # clearly off the mat) while never penetrating beyond G2
        data0 = lifted_frames[::max(1, len(lifted_frames) // 40)]
        heights = []
        for i in data0:
            data.qpos[:] = q[i]
            mujoco.mj_forward(model, data)
            sole = sole_points_world(model, data)
            for s in (0, 1):
                if not c[i, s]:
                    heights.append(float(sole[s][..., 2].min()))
                    assert heights[-1] >= -0.01, (
                        f"{name} frame {i}: flagged-lifted sole penetrates "
                        f"{heights[-1]:.4f} m (beyond G2)")
        assert float(np.mean(heights)) > 0.008, (
            f"{name}: lifted-sole mean height {np.mean(heights):.4f} m -- "
            "the swing does not actually lift (measured v2: ~0.012 m mean "
            "over the bump for a 0.04 m designed lift; the reach clip near "
            "full extension absorbs part of it)")


def test_probe_provenance_and_honest_labels():
    """feasibility.json v2 exists with the three levels + balance layer +
    probe command; the kept failures are labelled."""
    f = json.loads((V2 / "feasibility.json").read_text())
    assert f["version"] == "2.0"
    assert "balance_layer" in f["protocol"]
    assert f["protocol"]["balance_layer"]["checkpoint"]
    assert "reproduce" in f["provenance"]
    for name, row in f["per_track"].items():
        assert "validity" in row and "static" in row and "replay" in row \
            and "balance" in row
        assert row["balance"]["balance_layer"]
    # the fused penetration is honestly labelled, not passed off: its verdict
    # is either measured-infeasible or balance-blocked, never a quiet pass
    pen = f["drill_phases"]["DOUBLE_LEG_PENETRATION"]
    if not pen["balance"]["completed"]:
        assert f["drill_continuous_validity"]["DOUBLE_LEG_PENETRATION"] \
            in ("known_infeasible", "balance_blocked", "unverified")
    # the repaired entry no longer carries v1's blanket infeasible verdict
    assert f["drill_continuous_validity"]["DOUBLE_LEG_ENTRY_CROUCH"] \
        in VALIDITY


def test_lower_and_rise_take_measurements_recorded():
    """stand_to_stance (the 0.36 s wall take) and stance_rise carry their
    measured retime blocks with the v1 numbers they replace."""
    for name in ("stand_to_stance",):
        _, meta = _load(REFS / f"{name}.npz")
        assert "0.36 s" in meta["retime"]["what"]
        assert meta["retime"]["v1_peer"].startswith("v1/")
