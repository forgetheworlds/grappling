"""Contracts for the v1 motion-reference dataset (data/references/motion_refs).

Pinned failure modes (each traces to a measured defect or an acceptance point):
  * grounding: frames flagged planted must have their planted sole ON the mat
    (the v0 defect was +/-3..11 cm hover/penetration -- reference_fidelity §4);
  * continuity: composed references are C0 with bounded per-step deltas (no
    teleport, no joint snap);
  * format: load_reference() consumes every v1 track; phase/contact arrays are
    shape-consistent; no `expert_action` key ever exists (a kinematic reference
    is not a demonstration);
  * honesty: validity fields exist, and the known-infeasible shot crouch is
    LABELLED, not hidden;
  * provenance: the clip index carries measured timestamps inside the video.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

V1 = REPO / "data/references/motion_refs/v1"
REFS = V1 / "refs"

ALL_TAKES = ["stance_hold", "stance_widen_step", "stand_to_stance", "stance_to_stand",
             "shuffle_back", "level_change_full", "level_change_fast",
             "shot_entry_full", "shot_recover", "knee_sprawl_entry",
             "knee_sprawl_hold", "knee_sprawl_entry2", "knee_sprawl_recover",
             "stalk_shuffle", "circle_step"]

VALIDITY = {"kinematically_valid", "statically_holdable", "dynamically_verified",
            "unverified", "known_infeasible", "kinematically_invalid"}


def _load(path):
    d = np.load(path, allow_pickle=True)
    return d, json.loads(str(d["meta"]))


@pytest.fixture(scope="module")
def model():
    from solo.scene import load_solo_model
    return load_solo_model()


def test_all_v1_tracks_load_through_bc_loader():
    from solo.bc import load_reference
    for name in ALL_TAKES + ["drill_continuous", "stance_rise"]:
        tr = load_reference(name)
        assert tr.qpos.shape[1] == 36
        assert tr.qpos.shape[0] == len(tr.t) > 10


@pytest.mark.parametrize("name", ALL_TAKES)
def test_take_format_and_grounding(name, model):
    d, meta = _load(REFS / f"{name}.npz")
    assert "contact" in d, "v1 format requires the contact array"
    q, t, c = (np.asarray(d["qpos_a"], float), np.asarray(d["t"], float),
               np.asarray(d["contact"], bool))
    assert q.shape[0] == len(t) == c.shape[0]
    assert meta["grounding"]["method"] in ("per_frame_contact",
                                            "constant_floor_no_contact_detected")
    # grounding contract: at 50 Hz, a frame whose flags say planted must have
    # that foot's lowest sole within 2 cm of the mat (FK)
    import mujoco
    from solo.lit import sole_points_world
    data = mujoco.MjData(model)
    step = max(1, len(q) // 150)
    devs = []
    for i in range(0, len(q), step):
        if not c[i].any():
            continue
        data.qpos[:] = q[i]
        mujoco.mj_forward(model, data)
        sole = sole_points_world(model, data)
        for s in (0, 1):
            if c[i, s]:
                devs.append(abs(float(sole[s][..., 2].min())))
    med, worst = float(np.median(devs)), float(np.max(devs))
    # measured contract (v1): planted soles median within 2 cm everywhere;
    # worst within 8 cm on drill takes (fast crouch phases retain a bounded
    # residual).  Ground-posture takes (knee_sprawl_*, context-only, excluded
    # from the drill) carry only the median bar: their knees are ON the mat
    # and the soles are not the contact surface.
    assert med <= 0.02, f"{name}: planted sole median {med*100:.1f} cm off the mat"
    if name.startswith("knee_sprawl"):
        return
    assert worst <= 0.08, f"{name}: planted sole worst {worst*100:.1f} cm off the mat"


def test_drill_continuity_and_phases():
    d, meta = _load(V1 / "drill_continuous.npz")
    q = np.asarray(d["qpos_a"], float)
    pid = np.asarray(d["phase_id"], int)
    assert len(pid) == len(q) and pid.min() == 0
    phases = meta["phases"]
    assert len(phases) == int(pid.max()) + 1
    # every phase name unique except connectors; sources labelled
    for p in phases:
        assert p["source"] in ("observed", "g1_native", "fused",
                                "synthetic_connector")
        assert p["validity"] in VALIDITY
        assert p["t_end"] > p["t_start"]
    # C0 with bounded steps (V_MAX 6 rad/s * 0.02 s; root 3 m/s * 0.02 s)
    joint_steps = np.abs(np.diff(q[:, 7:], axis=0)).max(axis=1)
    root_steps = np.linalg.norm(np.diff(q[:, :3], axis=0), axis=1)
    assert joint_steps.max() <= 0.13, f"joint snap {joint_steps.max():.3f} rad"
    assert root_steps.max() <= 0.07, f"root jump {root_steps.max():.3f} m"
    # root displacement accumulates in the world; the cycle CLOSES (ends near
    # STANCE_HOLD's start) so net displacement is small by design -- assert on
    # total path length and max excursion instead
    steps = np.linalg.norm(np.diff(q[:, :2], axis=0), axis=1)
    path_len = float(steps.sum())
    excursion = float(np.max(np.linalg.norm(q[:, :2] - q[0, :2], axis=1)))
    assert path_len >= 1.0, f"root path length only {path_len:.3f} m"
    assert excursion >= 0.30, f"max root excursion only {excursion:.3f} m"
    # REPEAT affordance: final posture matches STANCE_HOLD start
    hold = next(p for p in phases if p["name"] == "STANCE_HOLD")
    i0 = int(np.searchsorted(np.asarray(d["t"], float), hold["t_start"]))
    gap = float(np.abs(q[-1, 7:] - q[i0, 7:]).max())
    assert gap <= 0.20, f"REPEAT gap {gap:.3f} rad"


def test_required_movement_set_present():
    d, meta = _load(V1 / "drill_continuous.npz")
    names = [p["name"] for p in meta["phases"]]
    for required in ("STAND", "LOWER_TO_STANCE", "STANCE_HOLD", "SHUFFLE_F",
                     "SHUFFLE_B", "CIRCLE", "LEVEL_CHANGE", "DOUBLE_LEG_ENTRY_CROUCH",
                     "DOUBLE_LEG_PENETRATION",
                     "RECOVER_TO_STANCE", "REPOSITION"):
        assert required in names, f"missing phase {required}"
    # synthetic connectors marked distinctly from observed motion
    assert any(p["source"] == "synthetic_connector" for p in meta["phases"])
    assert any(p["source"] == "observed" for p in meta["phases"])
    # lead-leg designation present on non-connector phases
    for p in meta["phases"]:
        if p["source"] != "synthetic_connector":
            assert p["lead_leg"] in ("left", "right")


def test_known_infeasible_is_labelled_not_hidden():
    d, meta = _load(REFS / "shot_entry_full.npz")
    assert meta["validity"] == "known_infeasible"
    assert "1.24 s" in meta["validity_reason"]
    drill_meta = _load(V1 / "drill_continuous.npz")[1]
    entry = next(p for p in drill_meta["phases"] if p["name"] == "DOUBLE_LEG_ENTRY_CROUCH")
    assert entry["validity"] == "known_infeasible"
    feas = json.loads((REPO / "data/references/motion_refs/feasibility.json").read_text())
    assert feas["tracks"]["shot_entry_full"]["validity"] == "known_infeasible"
    assert feas["tracks"]["drill_continuous"]["validity"] in VALIDITY


def test_no_expert_action_field_anywhere():
    for path in list(REFS.glob("*.npz")) + [V1 / "drill_continuous.npz",
                                             V1 / "stance_to_stand.npz"]:
        d = np.load(path, allow_pickle=True)
        assert "expert_action" not in d.files, path
        meta = json.loads(str(d["meta"]))
        assert "reference_not_demonstration" in meta or "source_npz" in meta


def test_clip_index_measured_and_in_range():
    idx = json.loads((REPO / "data/references/motion_refs/clip_index.json").read_text())
    assert idx["sources"][0]["video_id"] == "gBAhX5t-GW4"
    assert idx["sources"][1]["video_id"] == "tyU-QaV8MnI"
    assert idx["provenance_decision"]["primary"].startswith("gBAhX5t-GW4")
    for c in idx["clips"]:
        ts = c["video_timestamps"]
        assert 0 <= ts["t0_s"] < ts["t1_s"] <= 824.0
        assert ts["duration_s"] == pytest.approx(ts["t1_s"] - ts["t0_s"], abs=0.2)
        e = c["events_measured"]
        for key in ("pelvis_min", "pelvis_max", "knee_min", "feet"):
            assert key in e
        assert c["quality"]["frames"] >= 8


def test_probe_results_are_honest():
    path = REPO / "videos/motion_refs/dynamic_probe_metrics.json"
    if not path.exists():
        pytest.fail("dynamic probe metrics missing -- run scripts/probe_drill_dynamic.py")
    res = json.loads(path.read_text())
    assert res["per_track"]["shot_entry_full"]["ok"] is False, (
        "the shot-entry crouch topples the robot; a PASS here would be a "
        "measurement error (cem_capture.md measured 1.24 s)")
    assert "protocol" in res and "provenance" in res
    assert res["protocol"]["claim"].startswith("dynamic INDICATOR only")


def test_fusion_spec_attributes_sources():
    spec = json.loads((REPO / "data/references/motion_refs/fusion_spec.json").read_text())
    feats = {f["feature"]: f for f in spec["features"]}
    needed = {"stance_width", "stance_crouch_height", "torso_lean",
              "level_change_depth", "shot_structure", "recovery",
              "g1_repair_axes"}
    assert needed <= set(feats)
    for f in spec["features"]:
        assert set(f["sources"]) <= {"video", "grapplemap", "repair", "operator"}
        assert f["confidence"] in ("high", "medium", "low")
        assert f["conflict_rule"]
    # the shot structure must cite GrappleMap for spatial constraints
    assert "grapplemap" in feats["shot_structure"]["sources"]


def test_grounding_on_synthetic_stance_and_step():
    """The grounding contract, pinned on synthetic tracks (no video needed):
    a flat stance grounds both soles at z=0; a step keeps the stance foot
    planted at zero and the swing foot un-planted; the translation anchor is
    the planted foot."""
    sys.path.insert(0, str(REPO / "data/references/yt_gBAhX5t-GW4/derived/tools"))
    import retarget_video as rv
    from grapplemap import JOINTS as J
    ji = {n: J.index(n) for n in J}
    F = 30
    track = np.zeros((F, len(J), 3))
    track[:, ji["Core"], 1] = -0.9          # hips 0.9 above the feet (y-up)
    track[:, ji["LeftToe"], 1] = track[:, ji["LeftHeel"], 1] = -1.0
    track[:, ji["RightToe"], 1] = track[:, ji["RightHeel"], 1] = -1.0
    track[:, ji["LeftAnkle"], 1] = track[:, ji["RightAnkle"], 1] = -0.92
    off, contact, info = rv.ground_per_frame(track.copy(), 15.0)
    assert info["method"] == "per_frame_contact"
    # the 2-frame dwell leaves frame 0 unplanted by construction (edge effect)
    assert contact[1:, 0].all() and contact[1:, 1].all(), "flat stance = both planted"
    grounded = track[..., 1] - (info["floor_plane_m"] + off)[:, None]
    assert np.allclose(grounded[:, ji["LeftToe"]], 0.0, atol=1e-9)
    assert np.allclose(grounded[:, ji["RightHeel"]], 0.0, atol=1e-9)

    # step: the right foot lifts 12 cm mid-take
    track2 = track.copy()
    track2[10:20, ji["RightToe"], 1] = -0.88
    track2[10:20, ji["RightHeel"], 1] = -0.88
    track2[10:20, ji["RightAnkle"], 1] = -0.80
    off2, contact2, info2 = rv.ground_per_frame(track2.copy(), 15.0)
    assert contact2[1:, 0].all(), "stance foot stays planted through the step"
    assert not contact2[10:20, 1].any(), "swing foot is not planted"
    assert info2["contact_frac"] > 0.5

    # translation: left foot anchored, right foot slides forward 0.2 m ->
    # the pelvis does NOT translate with the swing foot
    track3 = track.copy()
    track3[:, ji["RightAnkle"], 0] = np.linspace(0.0, 0.2, F)
    track3[:, ji["RightToe"], 0] = np.linspace(0.0, 0.2, F)
    track3[:, ji["RightHeel"], 0] = np.linspace(0.0, 0.2, F)
    pos, anchor = rv.reconstruct_translation(track3, 15.0, contact=contact)
    assert not anchor.any(), "left is the only planted foot -> left anchors"
    assert np.allclose(pos[:, 0], 0.0, atol=1e-9), "pelvis static while stepping"
