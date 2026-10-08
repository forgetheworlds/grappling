"""Tests for the GrappleMap -> G1 retargeting pipeline (src/retarget).

Fast subset: unit-level mapping/algebra/world tests + one small end-to-end
build (STANCE) + npz roundtrip. Heavy per-technique builds live in
scripts/build_refs.py / scripts/validate_refs.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grapplemap import JOINTS, Reorientation  # noqa: E402
from retarget import landmarks as lm  # noqa: E402
from retarget.gmframe import (PairTransform, assemble_chain,  # noqa: E402
                              from_reorientation, yaw_align)
from retarget.landmarks import (GM_JOINT_TO_SITE, SOLVED_SITES,  # noqa: E402
                                gm_weights, site_weights)
from retarget.solve import DT_50HZ, G1Kinematics, resample_50hz  # noqa: E402
from retarget.techniques import (TECHNIQUES, build_technique,  # noqa: E402
                                 build_technique_targets, get_graph)
from retarget.world import place_world


# ---- landmark mapping ------------------------------------------------------

def test_mapping_table_counts():
    assert len(JOINTS) == 23
    mapped = [j for j, s in GM_JOINT_TO_SITE.items() if s is not None]
    dropped = [j for j, s in GM_JOINT_TO_SITE.items() if s is None]
    assert len(mapped) == 21 and dropped == ["LeftFingers", "RightFingers"]
    assert GM_JOINT_TO_SITE["LeftHand"] == "left_wrist"  # Hand shares wrist
    assert len(SOLVED_SITES) == 19
    assert len(lm.SITE_SPECS) == 19


def test_weight_presets_ordered():
    for preset in ("default", "strict"):
        w = gm_weights(preset)
        hi = w[JOINTS.index("Core")]
        med = w[JOINTS.index("LeftElbow")]
        lo = w[JOINTS.index("LeftToe")]
        assert 0 < lo < med < hi
        assert w[JOINTS.index("LeftFingers")] == 0.0
    assert gm_weights("strict")[0] > gm_weights("default")[0]
    sw = site_weights("default")
    assert sw[SOLVED_SITES.index("core")] == pytest.approx(4.0)
    assert sw[SOLVED_SITES.index("left_wrist")] == pytest.approx(1.5)  # Hand=Wrist max


def test_g1_geometry_sites_exist():
    geo = lm.g1_reference_geometry()
    segs = geo["segments"]
    assert 0.30 < segs["thigh"] < 0.40
    assert 0.25 < segs["shank"] < 0.35
    assert 0.02 < segs["neck_head"] < 0.15
    assert geo["pelvis_z"] == pytest.approx(0.79, abs=0.01)


# ---- transform algebra -----------------------------------------------------

def test_pair_transform_matches_parser_and_algebra():
    rng = np.random.default_rng(0)
    pose = rng.normal(size=(2, 23, 3))
    for _ in range(100):
        r = Reorientation(float(rng.uniform(-3, 3)),
                          tuple(rng.uniform(-1, 1, 3)),
                          bool(rng.integers(2)), bool(rng.integers(2)))
        p = from_reorientation(r)
        assert np.allclose(p.apply(pose), r.apply(pose), atol=1e-12)
        assert np.allclose(p.inverse().apply(p.apply(pose)), pose, atol=1e-10)
        q = Reorientation(float(rng.uniform(-3, 3)),
                          tuple(rng.uniform(-1, 1, 3)), False, False)
        pq = from_reorientation(q)
        comp = p.compose(pq)
        assert np.allclose(comp.apply(pose), p.apply(pq.apply(pose)), atol=1e-10)


def test_yaw_align_exact_on_rigid_motion():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(2, 23, 3))
    R = Reorientation(1.234, (0.3, -0.2, 0.7), False, False)
    t, rms = yaw_align(R.apply(pts).reshape(-1, 3), pts.reshape(-1, 3))
    assert rms < 1e-9
    back = t.apply(R.apply(pts))
    assert np.allclose(back, pts, atol=1e-8)


# ---- chain assembly --------------------------------------------------------

def test_double_leg_chain_exact_node_chaining():
    g = get_graph()
    chain = assemble_chain(g, [1144, 1147, 1141, 1140])
    assert len(chain.frames) == 2 + 3 + 3 + 3 - 3  # shared nodes drop dupes
    assert np.all(np.diff(chain.times) > 0)
    assert all(j["kind"] == "shared_node" and j["rms"] == 0.0
               for j in chain.junctions)


def test_technique_targets_shapes_and_floor():
    for tech in TECHNIQUES:
        tt = build_technique_targets(tech)
        assert tt.targets_a.shape == tt.targets_b.shape
        assert tt.targets_a.shape[1] == 23 and len(tt.times) == len(tt.targets_a)
        assert np.all(np.diff(tt.times) > 0)
        # mat floor respected (gm joint radius semantics)
        assert tt.targets_a[:, :, 1].min() >= 0.02 - 1e-9
        assert tt.targets_b[:, :, 1].min() >= 0.02 - 1e-9


def test_sprawl_roles_and_splice():
    tt = build_technique_targets("SPRAWL")
    assert tt.edges == [1147, 1141, 381, 382, 383]
    assert 0.0 < tt.info["splice"]["alignment_rms"] < 0.5
    # attacker starts lower (shot) than the standing sprawler (gm height = y)
    assert tt.targets_a[0, JOINTS.index("Core"), 1] < \
        tt.targets_b[0, JOINTS.index("Core"), 1] - 0.1


# ---- world placement -------------------------------------------------------

def test_place_world_ground_facing_and_scale():
    tt = build_technique_targets("STANCE")
    pl = place_world(tt.targets_a, tt.targets_b)
    core = JOINTS.index("Core")
    # ground preserved: z = scale * gm height >= ~0
    zmin = min(pl.targets_a[:, :, 2].min(), pl.targets_b[:, :, 2].min())
    assert 0.0 <= zmin < 0.05
    # attacker-side robot A on -x, robot B on +x (time-averaged cores)
    assert (pl.targets_a[:, core, 0].mean()
            < pl.targets_b[:, core, 0].mean() - 0.2)
    # scale ~ 1.32 m G1 vs ~1.7 m GrappleMap human
    assert 0.65 < pl.scale < 0.90
    assert abs(pl.scale_a - pl.scale_b) < 0.05


def test_place_world_is_chirality_preserving():
    """(x,y,z)_gm -> (x,-z,y)_mj: a right-handed arrangement must not mirror.

    Facing -x_gm with left landmarks at +z_gm must map to facing -x_mj with
    left at -y_mj (proper rotation), not +y_mj.
    """
    track = np.zeros((1, 23, 3))
    track[0, JOINTS.index("Core")] = (0.0, 1.0, 0.0)
    track[0, JOINTS.index("Neck")] = (-0.2, 1.4, 0.0)   # lean -x => facing -x
    track[0, JOINTS.index("LeftHip")] = (0.0, 1.0, 0.2)   # left at +z
    track[0, JOINTS.index("RightHip")] = (0.0, 1.0, -0.2)
    track_b = track.copy()
    track_b[0, :, 0] += 1.0
    pl = place_world(track, track_b)
    lh, rh = JOINTS.index("LeftHip"), JOINTS.index("RightHip")
    # gm +z must land at mj -y (proper rotation), so left-hip y < right-hip y
    assert pl.targets_a[0, lh, 1] < pl.targets_a[0, rh, 1]


# ---- end-to-end (small) + npz roundtrip ------------------------------------

def test_stance_end_to_end_and_npz_roundtrip(tmp_path):
    result = build_technique("STANCE")
    qa, qb, t = result["qpos_a"], result["qpos_b"], result["t"]
    kin = G1Kinematics()
    assert qa.shape == qb.shape == (len(t), 36)
    assert np.allclose(np.diff(t), DT_50HZ, atol=1e-9)
    for q in (qa, qb):
        assert np.abs(np.linalg.norm(q[:, 3:7], axis=1) - 1).max() < 1e-9
        assert (q[:, 7:36] >= kin.joint_lo - 1e-9).all()
        assert (q[:, 7:36] <= kin.joint_hi + 1e-9).all()
    assert result["landmark_rms"] < 0.15

    path = tmp_path / "STANCE.npz"
    np.savez_compressed(
        path, qpos_a=qa, qpos_b=qb, t=t, technique=np.array("STANCE"),
        edges=np.array(result["edges"]), landmark_rms=result["landmark_rms"],
        meta=json.dumps(result["meta"]))
    z = np.load(path, allow_pickle=False)
    assert str(z["technique"]) == "STANCE"
    assert z["qpos_a"].shape == (len(t), 36)
    assert json.loads(str(z["meta"]))["dt"] == DT_50HZ
    assert np.allclose(z["qpos_a"], qa)


def test_resample_limits_enforced():
    rng = np.random.default_rng(3)
    f = 12
    t_kf = np.linspace(0, 1.0, f)
    qa = np.zeros((f, 36))
    qa[:, 2] = 0.79
    qa[:, 10] = rng.uniform(-1, 1, f) * 2  # wild knee
    qb = np.zeros((f, 36))
    qb[:, 2] = 0.79
    qb[:, 3] = 1.0
    t_grid, ga, gb, stretch = resample_50hz(t_kf, qa, qb)
    assert np.allclose(np.diff(t_grid), DT_50HZ, atol=1e-9)
    assert stretch > 1.0
    v = np.gradient(ga[:, 10], t_grid)
    a = np.gradient(v, t_grid)
    from retarget.solve import A_MAX, V_MAX
    assert np.abs(v).max() <= V_MAX + 1e-6
    assert np.abs(a).max() <= A_MAX * 1.5  # gradient smoothing of C1 curve


# ---- scene -----------------------------------------------------------------

def test_scene_model_contract():
    import mujoco
    from retarget.scene import load_scene_model, robot_slice
    m = load_scene_model()
    assert m.nq == 72 and m.nu == 58
    for i in range(m.nu):  # actuator i drives joint of the same name
        jid = int(m.actuator_trnid[i, 0])
        assert m.joint(jid).name == m.actuator(i).name
    for prefix, start in (("a_", 0), ("b_", 36)):
        sl = robot_slice(m, prefix)
        assert sl.start == start and sl.stop == start + 36
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE,
                                 f"{prefix}core") >= 0
    assert m.opt.timestep == pytest.approx(0.002)
