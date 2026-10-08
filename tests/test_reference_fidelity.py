"""Number-asserting tests for the reference-fidelity instrument
(``scripts/measure_reference_fidelity.py``), whose measurements back
``reports/2026-10-08/reference_fidelity.md``.

These pin the parts a wrong answer would silently corrupt:

* the 2-D geometry helpers (the hull margin sign is what "inside the support
  polygon" means everywhere downstream);
* the contact-run detector (the contact/timing table is built from it);
* the terminal-posture predicate, whose POSITIVE control is our own stance
  keyframe (flat feet, CoM inside the hull) and whose NEGATIVE control is the
  same keyframe with the root pitched 80 deg.  Without the positive control a
  predicate that rejects *everything* would look like a finding.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

_spec = importlib.util.spec_from_file_location(
    "measure_reference_fidelity", REPO / "scripts/measure_reference_fidelity.py")
mrf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mrf)


def test_cross2_and_hull_margin_on_a_known_square():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert mrf.cross2(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 1.0
    assert mrf.hull_margin(square, np.array([0.5, 0.5])) == pytest.approx(0.5)
    assert mrf.hull_margin(square, np.array([1.5, 0.5])) == pytest.approx(-0.5)
    assert np.isnan(mrf.hull_margin(np.array([[0.0, 0.0], [1.0, 0.0]]), np.zeros(2)))


def test_convex_hull_drops_interior_points():
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], float)
    h = mrf.convex_hull_2d(pts)
    assert len(h) == 4
    assert not np.any(np.all(np.isclose(h, [0.5, 0.5]), axis=1))


def test_contact_runs_and_events_on_a_hand_series():
    t = np.arange(8) * 0.1
    flags = np.array([0, 1, 1, 1, 0, 1, 1, 1], bool)
    runs = mrf.contact_runs(flags, t, min_s=0.0)
    assert [(a, b) for a, b, _ in runs] == [(1, 3), (5, 7)]
    assert [d for _, _, d in runs] == [pytest.approx(0.2), pytest.approx(0.2)]
    # min_s drops both (duration 0.2 s < 0.25 s)
    assert mrf.contact_runs(flags, t, min_s=0.25) == []
    starts, ends = mrf.events(flags, t, min_s=0.2)
    assert starts == [pytest.approx(0.1), pytest.approx(0.5)]
    assert ends == [pytest.approx(0.3), pytest.approx(0.7)]


@pytest.fixture(scope="module")
def fk_stance():
    fk = mrf.G1FK()
    stance_pelvis = mrf.load_stance_pelvis(fk)
    from solo.scene import load_solo_model
    from solo.stance import stance_qpos

    return fk, stance_pelvis, stance_qpos(model=load_solo_model())


def test_terminal_stance_accepts_our_keyframe(fk_stance):
    """Positive control: our own stance is flat-footed, CoM inside, upright."""
    fk, sp, q = fk_stance
    r = mrf.terminal_stance(fk, np.tile(q, (5, 1)), sp)
    assert r["feet_flat_both"] is True
    assert r["com_inside"] is True
    assert r["com_margin_m"] == pytest.approx(0.0533, abs=1e-3)
    assert r["torso_tilt_deg"] == pytest.approx(0.69, abs=0.05)
    assert r["pelvis_z_end"] == pytest.approx(0.79, abs=1e-6)
    assert r["ok"] is True


def test_terminal_stance_rejects_a_collapsed_root(fk_stance):
    """Negative control: the same keyframe pitched 80 deg forward must fail."""
    fk, sp, q = fk_stance
    q2 = q.copy()
    a = np.radians(80.0) / 2.0
    q2[3:7] = [np.cos(a), 0.0, np.sin(a), 0.0]   # quat (w, x, y, z), pitch about +y
    r = mrf.terminal_stance(fk, np.tile(q2, (5, 1)), sp)
    assert r["tilt_ok"] is False
    assert r["torso_tilt_deg"] == pytest.approx(79.3, abs=0.1)
    assert r["feet_flat_both"] is False
    assert r["ok"] is False
