"""Detector unit tests: back-to-mat (dorsal) semantics.

These four tests are the verbatim test bodies of ``tests/test_wrestling.py``'s
"detector units" section, moved here on 2026-10-08 when the two-robot pipeline
(``src/wrestling``, its env and its tests) was deleted and the detector itself
moved to :mod:`solo.detector`.  Only the import header and the ``model``
fixture changed: the module path (the module moved under the mandate) and the
model source (``wrestling.env.load_wrestling_model`` compiled
``robots/wrestling_scene.xml``; this file compiles that same XML directly, and
asserts the same ``nq``/``nu``).  Every assertion, threshold and feature row is
unchanged -- these are the tests that pin the ledger-verified detector
(sensitivity/specificity 1.000 on the scripted falls,
``reports/2026-10-08/wrestling_env.md`` §1) and the calibrated operating point
in ``data/backdet_calibration.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import mujoco  # noqa: E402

from solo.detector import (FLOOR_GEOM, ROBOTS, BackDetConfig,  # noqa: E402
                           BackFeatures, BackToMatDetector, back_features,
                           body_maps, confirmed_mask, first_confirmed_index)
from solo.scene import STEP_DT  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
#: the scene the detector was calibrated on (retained: ``robots/**`` is untouched)
SCENE_XML = REPO / "robots" / "wrestling_scene.xml"
REF_DIR = REPO / "data" / "refs"
#: model-order qpos slices of that scene (7 base + 29 joints per robot)
QPOS_SLICE = {"a": slice(0, 36), "b": slice(36, 72)}


@pytest.fixture(scope="module")
def model():
    """The calibration scene model (same XML, nq=72 / nu=58)."""
    m = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    assert (m.nq, m.nu) == (72, 58), (m.nq, m.nu)
    return m


def reference_trace(technique: str) -> dict:
    """``data/refs/<technique>.npz`` with the keys the calibration replays.

    Local copy of the loader that lived in ``wrestling.env`` (deleted
    2026-10-08): the frames the limb-contact regression replays are the same
    reference data read with the same dtypes.
    """
    with np.load(REF_DIR / f"{technique}.npz", allow_pickle=False) as z:
        return {
            "qpos_a": np.asarray(z["qpos_a"], dtype=np.float64),
            "qpos_b": np.asarray(z["qpos_b"], dtype=np.float64),
            "t": np.asarray(z["t"], dtype=np.float64),
        }


# -------------------------------------------------------------- detector units
def _feat(t=0.0, dorsal=False, front=False, tilt=0.0, pelz=0.8, limb=False, n=0):
    return BackFeatures(t=t, dorsal_contact=dorsal, front_contact=front,
                        contact_count=n, contact_x_min=-0.05 if dorsal else np.nan,
                        contact_x_max=0.05 if front else np.nan, tilt_deg=tilt,
                        pelvis_z=pelz, limb_contact=limb)


def test_backdet_persistence_and_negatives():
    cfg = BackDetConfig()
    det = BackToMatDetector(cfg)
    # knees/hands + front contact + upright: never a trigger
    for i in range(50):
        new = det.update_features({r: _feat(t=i * STEP_DT, front=True, tilt=80.0,
                                            pelz=0.3, limb=True, n=3) for r in ROBOTS},
                                  i * STEP_DT)
        assert not new
    # brief dorsal contact shorter than the confirmation period: no trigger
    det.reset()
    for i in range(5):
        assert not det.update_features(
            {r: _feat(t=i * STEP_DT, dorsal=True, tilt=80.0, pelz=0.3, n=2) for r in ROBOTS},
            i * STEP_DT)
    # sustained dorsal contact: triggers exactly once, after the confirmation
    det.reset()
    t0 = 10.0
    fired = []
    for i in range(40):
        t = t0 + i * STEP_DT
        new = det.update_features(
            {r: _feat(t=t, dorsal=True, tilt=80.0, pelz=0.3, n=2) for r in ROBOTS}, t)
        if new:
            fired.append((t, tuple(sorted(new))))
    assert fired == [(pytest.approx(t0 + cfg.confirm_s), ("a", "b"))], fired
    # latched: no re-trigger while the state continues
    assert det.triggers["a"] == pytest.approx(t0 + cfg.confirm_s)
    # a robot that leaves the state before confirmation does not trigger
    det.reset()
    for i in range(int(cfg.confirm_s / STEP_DT) - 1):
        det.update_features({r: _feat(t=i * STEP_DT, dorsal=True, tilt=80.0, pelz=0.3)
                             for r in ROBOTS}, i * STEP_DT)
    det.update_features({r: _feat(t=1.0, dorsal=False, tilt=10.0, pelz=0.8) for r in ROBOTS}, 1.0)
    assert det.triggers == {"a": None, "b": None}


def test_backdet_limb_contact_side_infixed_bodies(model):
    """limb_contact diagnostic must see the scene's real limb bodies.

    Regression: limb ids were built by prefix concatenation (``a_knee_link``),
    but the scene infixes left/right (``a_left_knee_link`` …), so ``limb_bids``
    was always empty and ``limb_contact`` always False even with limbs on the
    mat. The flag is diagnostic-only: it never gates the back-to-mat rule.
    """
    from solo.detector import LIMB_BODIES
    for robot in ROBOTS:
        limb_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
                      for b in body_maps(model, robot).limb_bids}
        assert limb_names, robot
        assert all(n.startswith(f"{robot}_") for n in limb_names)
        assert all(n.endswith(LIMB_BODIES) for n in limb_names)
        # both sides of every limb resolved (knee + both wrists + elbow, L/R)
        assert len(limb_names) == 2 * len(LIMB_BODIES)

    def floor_bodies(data):
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM)
        out = set()
        for c in range(data.ncon):
            con = data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 == floor or g2 == floor:
                other = g2 if g1 == floor else g1
                out.add(mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[other])))
        return out

    def frame_data(trace, i):
        data = mujoco.MjData(model)
        data.qpos[QPOS_SLICE["a"]] = trace["qpos_a"][i]
        data.qpos[QPOS_SLICE["b"]] = trace["qpos_b"][i]
        mujoco.mj_forward(model, data)
        return data

    # positive: STAND_UP ground-start frame 6. STAND_UP.npz was rebuilt on the
    # shared-node chain t1125->t380->t540 (grounded, 244 frames; the airborne
    # montage it replaced is kept as STAND_UP.airborne.npz) and now plants the
    # RIGHT wrist at the ground start (the airborne reference planted the
    # left). The side is incidental to this regression -- the resolution
    # assertions above already pin both sides -- so require a *resolved*
    # executor wrist body, not a hardcoded side, and keep the knee on the mat.
    data = frame_data(reference_trace("STAND_UP"), 6)
    on_mat = floor_bodies(data)
    assert any(n.startswith("a_") and n.endswith("wrist_yaw_link")
               for n in on_mat), sorted(on_mat)
    assert "b_left_knee_link" in on_mat, sorted(on_mat)
    det = BackToMatDetector(BackDetConfig())
    for robot in ROBOTS:
        f = back_features(model, data, robot)
        assert f.limb_contact is True, robot
        # diagnostic-only: limb contact never satisfies the rule
        assert not det.condition(f), robot

    # negative: standing frame — an ankle may touch the mat, no limb does
    data = frame_data(reference_trace("STANCE"), 0)
    on_mat = floor_bodies(data)
    assert on_mat and not any(n.endswith(LIMB_BODIES) for n in on_mat)
    for robot in ROBOTS:
        f = back_features(model, data, robot)
        assert f.limb_contact is False, robot
        assert f.pelvis_z > 0.55, robot  # clearly standing


def test_backdet_batch_matches_streaming():
    """The vectorized calibration rule must reproduce the streaming detector."""
    rng = np.random.default_rng(0)
    n = 400
    t = np.arange(n) * STEP_DT
    feats = [_feat(t=t[i], dorsal=bool(rng.random() < 0.4),
                   tilt=float(rng.uniform(0, 130)), pelz=float(rng.uniform(0.05, 0.9)))
             for i in range(n)]
    for cfg in (BackDetConfig(),
                BackDetConfig(tilt_threshold_deg=70, confirm_s=0.1),
                BackDetConfig(pelvis_z_threshold=0.2, confirm_s=0.5)):
        det = BackToMatDetector(cfg)
        streaming = []
        for f in feats:
            if det.update_features({"a": f, "b": f}, f.t).get("a"):
                streaming.append(f.t)
        cond = np.array([det.condition(f) for f in feats])
        idx = first_confirmed_index(cond, t, cfg.confirm_s)
        batch = [] if idx is None else [t[idx]]
        assert streaming == pytest.approx(batch)
        # the batch mask marks the same state and is monotone after the trigger
        mask = confirmed_mask(cond, t, cfg.confirm_s)
        if idx is not None:
            assert mask[idx:].sum() >= 1


def test_calibration_json_matches_default_config():
    path = REPO / "data" / "backdet_calibration.json"
    payload = json.loads(path.read_text())
    assert payload["config"] == BackDetConfig().as_dict()
    assert payload["metrics"]["sens"] >= 0.95 and payload["metrics"]["spec"] >= 0.95
    assert payload["margins"]["tilt_margin_pos_deg"] > 0
    assert payload["margins"]["tilt_margin_neg_deg"] > 0
    assert payload["margins"]["pelz_margin_pos_m"] > 0
    assert payload["evidence"]["dorsal_normal_agreement"] >= 0.95
