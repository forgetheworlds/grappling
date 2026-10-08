"""BC pose-prior + imitation-term tests (S6 precursor).

Every test asserts a NUMBER (or an exact structural identity) so it fails if the
thing it pins breaks:

* the dataset's state->action mapping on a hand-constructed snippet,
* the per-reference TIME split (disjoint frame ranges, proven),
* the BC policy's held-out joint error vs BOTH constant baselines, on the
  shipped ``data/solo/bc`` artifacts,
* the imitation terms (exact match -> maximal; a known offset -> the specific
  decrement) and the deviation predicate thresholds,
* the actor phase field's population status (constant for the corpus; live for
  the shot task) and the measured (pose, velocity) -> action ambiguity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from solo import bc, imitation  # noqa: E402
from solo.commands import Command  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BC_DIR = REPO / "data" / "solo" / "bc"

#: acceptance margin: the BC policy must reach AT MOST this fraction of the
#: stand-keyframe constant baseline (measured margin is reported in
#: reports/2026-10-08/imitation.md)
MARGIN_VS_STAND = 0.50
MARGIN_VS_MEAN = 0.50


@pytest.fixture(scope="module")
def artifacts():
    metrics = BC_DIR / "bc_metrics.json"
    ckpt = BC_DIR / "bc_policy.pt"
    data = BC_DIR / "dataset.npz"
    for p in (metrics, ckpt, data):
        assert p.exists(), f"missing artifact {p}; run scripts/solo_bc_train.py"
    return {
        "metrics": json.loads(metrics.read_text()),
        "policy": bc.BCPolicy.load(ckpt),
        "dataset": bc.load_dataset(data),
    }


# --------------------------------------------------------------- dataset
def test_dataset_mapping_is_exact_on_hand_snippet():
    """ctrl_k = clip(q_ref[k+1]); u_k = (ctrl_k - mid)/half, asserted exactly."""
    q = np.zeros((3, 36))
    q[:, 2] = 0.7
    q[:, 3] = 1.0                      # w-first identity quaternion
    q[1:, 7] = [0.4, 0.9]              # joint 0: 0.0 -> 0.4 -> 0.9
    q[1:, 8] = -0.5
    q[2, 9] = 1e3                      # joint 2 outside ctrlrange -> must clip
    t = np.array([0.0, 0.02, 0.04])
    model = bc.load_solo_model()
    from solo.scene import ctrl_range, stand_frame

    lo, hi = ctrl_range(model)
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    pr = bc.pairs_from_qpos(q, t, name="hand", model=model)
    assert len(pr) == 2                      # T - lookahead(1)
    # the action of pair 0 targets FRAME 1 (lookahead), not frame 0
    expected_ctrl = np.clip(q[1, 7:36], lo, hi)
    assert np.array_equal(pr.ctrl[0], expected_ctrl)
    assert pr.ctrl[0][0] == pytest.approx(0.4)
    # the clipped joint lives in the action whose target is frame 2 (pair 1)
    assert pr.ctrl[1][2] == pytest.approx(hi[2])
    assert pr.clipped == 1
    assert not np.array_equal(pr.ctrl[0], np.clip(q[0, 7:36], lo, hi))
    assert np.array_equal(pr.unit[0], (expected_ctrl - mid) / half)
    # observation fields derivable from the reference alone
    q_stand, _ = stand_frame(model)
    assert np.allclose(pr.obs[0, 9:38], np.float32(q[0, 7:36] - q_stand[7:36]))
    # joint velocity: one-sided first difference at the track start
    assert pr.obs[0, 38] == pytest.approx((q[1, 7] - q[0, 7]) / 0.02, abs=1e-5)
    # prev_action of pair 1 is the unit action of pair 0
    assert np.allclose(pr.obs[1, 67:96], pr.unit[0], atol=1e-6)
    assert np.allclose(pr.obs[0, 67:96], 0.0)


def test_time_split_has_zero_overlap_and_is_contiguous():
    """Per-reference train/val frame ranges are disjoint (proven on indices)."""
    ds = bc.build_dataset(("STANCE", "DOUBLE_LEG"), train_frac=0.7)
    ov = bc.split_overlap(ds)
    assert ov["overlapping"] == []
    assert ov["train_frames"] == int(ds.train_mask.sum())
    assert ov["val_frames"] == int(ds.val_mask.sum())
    for name, s in ds.split.items():
        n = s["n"]
        assert s["train"] == [0, s["train"][1]]
        assert s["val"] == [s["train"][1], n]
        assert s["train"][1] == int(n * 0.7)
        assert s["val"][1] == n
    # every pair belongs to exactly one side, and frames are unique per ref
    key = np.stack([ds.ref_id, ds.frame], axis=1)
    assert len({tuple(k) for k in key}) == len(ds)
    assert np.all(ds.train_mask ^ ds.val_mask)


def test_shipped_dataset_split_artifact_is_disjoint(artifacts):
    ds = artifacts["dataset"]
    ov = bc.split_overlap(ds)
    assert ov["overlapping"] == []
    meta = artifacts["metrics"]["corpus"]
    assert meta["overlap_pairs"] == 0
    assert meta["n"] == len(ds)
    assert meta["actor_slice"] == [0, 96]


# ------------------------------------------------------------------ policy
def test_bc_beats_both_constant_baselines_on_held_out(artifacts):
    """Held-out joint error below the stand AND mean-pose constants by margin."""
    ds, policy = artifacts["dataset"], artifacts["policy"]
    val = bc.validation_report(policy, ds)
    m = artifacts["metrics"]["validation"]
    bc_err = val["bc"]["mean_rad"]
    stand = val["stand_baseline_rad"]["mean_rad"]
    mean = val["mean_baseline_rad"]["mean_rad"]
    # artifact consistency: the report and the live recomputation agree
    assert bc_err == pytest.approx(m["bc"]["mean_rad"], abs=1e-6)
    assert stand == pytest.approx(m["stand_baseline_rad"]["mean_rad"], abs=1e-6)
    # movement, not a static pose: margins stated as constants at the top
    assert bc_err <= MARGIN_VS_STAND * stand, (bc_err, stand)
    assert bc_err <= MARGIN_VS_MEAN * mean, (bc_err, mean)
    # and the BC error is small in absolute terms
    assert bc_err < 0.10
    # per-joint: BC must not be materially worse than the stand baseline on ANY
    # joint (the wrists are static in both: baseline error 0.0, BC ~1e-4 rad,
    # so a strict < would be an unfair 0.01%-level test)
    bc_j = np.array(val["bc"]["per_joint_rad"])
    stand_j = np.array(val["stand_baseline_rad"]["per_joint_rad"])
    worse = [bc.joint_names()[i] for i in np.where(bc_j > stand_j + 0.01)[0]]
    assert not worse, worse


def test_policy_contract_matches_dataset_metadata(artifacts):
    ds, policy = artifacts["dataset"], artifacts["policy"]
    assert np.allclose(policy.mid, ds.mid) and np.allclose(policy.half, ds.half)
    assert policy.meta["mapping"] == ds.as_meta()["mapping"]
    assert policy.actor.obs_dim == 96 and policy.actor.act_dim == 29
    # unit output is bounded and ctrl stays inside ctrlrange
    u = policy.unit(ds.obs[ds.val_mask])
    assert np.max(np.abs(u)) <= 1.0
    ctrl = policy.ctrl(ds.obs[ds.val_mask])
    assert np.all(ctrl >= ds.lo - 1e-9) and np.all(ctrl <= ds.hi + 1e-9)


# --------------------------------------------------------------- imitation
def test_imitation_joint_pose_term_exact_and_offset():
    w = imitation.DEFAULT_WEIGHTS
    ref = imitation.ImitationState(joints=np.full(29, 0.3))
    exact = imitation.ImitationState(joints=np.full(29, 0.3))
    assert imitation.joint_pose_term(exact, ref, w) == pytest.approx(1.0)
    assert imitation.imitation_reward(exact, ref, w)["total"] == pytest.approx(
        w.site + w.joint_pose + w.joint_vel + w.root + w.com)
    # a known offset on ONE joint -> the specific decrement
    delta = 0.3
    off = imitation.ImitationState(joints=np.full(29, 0.3))
    off.joints = off.joints.copy()
    off.joints[0] += delta
    expect = np.exp(-(delta ** 2 / 29) / w.joint_pose_scale_rad ** 2)
    assert imitation.joint_pose_term(off, ref, w) == pytest.approx(expect, rel=1e-12)
    assert imitation.joint_pose_error(off, ref) == pytest.approx(delta / 29, rel=1e-12)


def test_imitation_site_and_root_terms_offsets():
    w = imitation.DEFAULT_WEIGHTS
    ref = imitation.ImitationState()
    # a HIGH site offset by delta -> exp(-(w_i/sum(w) * delta^2)/scale^2)
    delta = 0.05
    off = imitation.ImitationState()
    off.site_pos["core"] = off.site_pos["core"] + np.array([delta, 0.0, 0.0])
    sw = imitation.site_weights()
    idx = imitation.SITE_INDEX["core"]
    expect = np.exp(-((sw[idx] * delta ** 2) / sw.sum()) / w.site_scale_m ** 2)
    assert imitation.site_term(off, ref, w) == pytest.approx(expect, rel=1e-12)
    assert imitation.weight_table()["core"]["class"] == "HIGH"
    # root offset by 0.1 m -> exp(-(0.1^2)/0.1^2) = 1/e
    off = imitation.ImitationState(root_pos=np.array([0.1, 0.0, 0.0]))
    assert imitation.root_term(off, ref, w) == pytest.approx(np.exp(-1.0), rel=1e-12)
    # com offset by 0.08 m -> also 1/e
    off = imitation.ImitationState(com=np.array([0.08, 0.0, 0.0]))
    assert imitation.com_term(off, ref, w) == pytest.approx(np.exp(-1.0), rel=1e-12)
    # exact match is maximal for every family
    terms = imitation.imitation_reward(imitation.ImitationState(),
                                       imitation.ImitationState(), w)
    assert terms["site"] == pytest.approx(w.site)
    assert terms["root"] == pytest.approx(w.root)
    assert terms["com"] == pytest.approx(w.com)


def test_site_weight_classes_reuse_the_retarget_preset():
    from retarget.landmarks import WEIGHT_PRESETS

    preset = WEIGHT_PRESETS["default"]
    tbl = imitation.weight_table()
    assert tbl["core"]["weight"] == preset["HIGH"]
    assert tbl["left_knee"]["weight"] == preset["HIGH"]
    assert tbl["left_wrist"]["weight"] == preset["MED"]
    assert tbl["left_toe"]["weight"] == preset["LOW"]
    # the noisiest landmarks (hands) are down-weighted vs the stable ones
    assert tbl["left_wrist"]["weight"] < tbl["core"]["weight"]
    assert tbl["left_toe"]["weight"] < tbl["left_wrist"]["weight"]
    assert len(imitation.site_weights()) == 19


def test_deviation_predicate_thresholds():
    """Strictly-greater semantics, checked exactly AT and just above each
    threshold.  The thresholds are replaced by exactly representable values
    (powers of two) so ``err == threshold`` is bit-exact and the boundary test
    does not depend on rounding inside a mean."""
    from dataclasses import replace

    w = replace(imitation.DEFAULT_WEIGHTS, term_joint_rad=0.5,
                term_root_xy_m=0.25, term_pelvis_drop_m=0.5, term_site_m=0.5)
    ref = imitation.ImitationState()

    def joints(err):
        return imitation.ImitationState(joints=np.full(29, err))

    # joint pose: exactly AT the threshold does not fire; just above does
    assert imitation.joint_pose_error(joints(w.term_joint_rad), ref) == w.term_joint_rad
    assert not imitation.deviation_terminated(joints(w.term_joint_rad), ref, w)
    s = joints(w.term_joint_rad * (1 + 1e-6))
    assert imitation.deviation_terminated(s, ref, w)
    assert imitation.deviation_reason(s, ref, w) == "joint"
    # root xy threshold
    r = imitation.ImitationState(root_pos=np.array([w.term_root_xy_m, 0.0, 0.0]))
    assert not imitation.deviation_terminated(r, ref, w)
    r = imitation.ImitationState(
        root_pos=np.array([w.term_root_xy_m * (1 + 1e-6), 0.0, 0.0]))
    assert imitation.deviation_reason(r, ref, w) == "root_xy"
    # pelvis drop threshold
    ref_high = imitation.ImitationState(root_pos=np.array([0.0, 0.0, 0.8]))
    d = imitation.ImitationState(
        root_pos=np.array([0.0, 0.0, 0.8 - w.term_pelvis_drop_m]))
    assert not imitation.deviation_terminated(d, ref_high, w)
    d = imitation.ImitationState(
        root_pos=np.array([0.0, 0.0, 0.8 - w.term_pelvis_drop_m * (1 + 1e-6)]))
    assert imitation.deviation_reason(d, ref_high, w) == "pelvis_drop"
    # site rms threshold: every site offset by exactly the threshold
    ref0 = imitation.ImitationState()
    s = imitation.ImitationState()
    for name in imitation.SITE_INDEX:
        s.site_pos[name] = np.array([w.term_site_m, 0.0, 0.0])
    assert imitation.site_error_m(s, ref0) == pytest.approx(w.term_site_m, rel=0, abs=0)
    assert not imitation.deviation_terminated(s, ref0, w)
    for name in imitation.SITE_INDEX:
        s.site_pos[name] = np.array([w.term_site_m * (1 + 1e-6), 0.0, 0.0])
    assert imitation.deviation_reason(s, ref0, w) == "site"


# ------------------------------------------------------- phase / ambiguity
def test_phase_field_is_constant_for_the_corpus(artifacts):
    """The actor phase field (offset 114) is NOT populated for these refs."""
    ds = artifacts["dataset"]
    rep = bc.corpus_phase_report(ds)
    assert all(v["constant"] for v in rep.values())
    assert all(v["n_unique"] == 1 for v in rep.values())
    assert all(v["min"] == 0.0 and v["max"] == 0.0 for v in rep.values())
    # nothing in the 96-dim input carries a clock either
    assert ds.phase.shape == (len(ds),)


def test_phase_field_is_live_when_the_shot_skill_is_commanded():
    """The same field DOES advance (0 -> 1) under the shot task -- so the
    minimal fix for the corpus is to drive cmd.skill/phase, not to invent a new
    observation."""
    from solo.env import SoloEnv

    env = SoloEnv(task="shot", seed=0, record_metrics=False)
    env.reset(seed=0)
    ctrl = env._ctrl_stand.copy()
    phases = []
    for _ in range(140):                      # 2.8 s: 0.5 s lead-in + 1.2 s clock
        obs, _r, term, trunc, _info = env.step(ctrl)
        phases.append(float(obs["actor"][114]))
        if term or trunc:
            break
    ph = np.array(phases)
    assert float(ph[0]) < 1e-6                # starts at zero
    assert np.all(np.diff(ph) >= -1e-7)       # monotone non-decreasing
    assert ph[-1] > 0.9                       # reaches the top of the clock
    assert len(np.unique(ph)) > 50            # a real clock, not a constant


def test_obs_to_action_ambiguity_is_measured_and_bounded(artifacts):
    """Pose+velocity alone: nearest non-local frame's action spread (measured)."""
    ds = artifacts["dataset"]
    amb = bc.ambiguity_report(ds)
    stored = artifacts["metrics"]["ambiguity"]
    nn = amb["val_to_train_nn"]
    assert nn["n"] == stored["val_to_train_nn"]["n"]
    assert nn["act_err_median_rad"] == pytest.approx(
        stored["val_to_train_nn"]["act_err_median_rad"], abs=1e-9)
    # the ambiguity floor is a real number, not zero...
    assert nn["act_err_median_rad"] > 1e-3
    # ...and it is bounded: the median indistinguishable neighbour demands less
    # than 0.20 rad, and <10% of held-out frames have a >0.25 rad twin.
    assert nn["act_err_median_rad"] < 0.20
    assert nn["frac_gt_material"] < 0.10
    assert 0.0 <= nn["frac_nearest_is_same_ref"] <= 1.0


def test_corpus_covers_the_video_references():
    """The 12 operator-video tracks are the primary corpus (plus GrappleMap)."""
    ds = bc.build_dataset(("shot_entry_full", "STANCE"), train_frac=0.7)
    assert ds.names[0] == "shot_entry_full"
    meta = json.loads((BC_DIR / "bc_metrics.json").read_text())
    used = set(meta["corpus"]["used"])
    missing = [n for n in bc.VIDEO_REFS if n not in used]
    assert not missing, missing
    assert ds.split["shot_entry_full"]["n"] == 410
    assert ds.split["STANCE"]["n"] == 117
