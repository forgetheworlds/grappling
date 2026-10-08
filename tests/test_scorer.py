"""Tests for the technique-validity scorer (deliverable 7).

Covers the acceptance bar from the task brief plus the contracts the training
loops will rely on:

* every reference technique scores >= 0.8 mean on ITS OWN trace in EVERY phase
  (calibration quality), and every technique's own trace beats every other
  technique's trace (discrimination);
* the DOUBLE_LEG trace scored as SINGLE_LEG averages < 0.5;
* perturbation monotonicity: increasing joint noise, root tilt and (for
  progressive techniques) phase shift monotonically decrease the score;
* speed: < 1 ms per ``score()`` call on this 4-core ARM host;
* relational invariance: a rigid transform of the whole pair leaves the score
  unchanged (the judge sees relationships, not absolute pose);
* architecture: ``src/scorer`` is separate from any value function / critic.
"""

from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import mujoco  # noqa: E402

from scorer import (FEAT_ORDER, PARAMS_PATH, TECHNIQUES, ScoreResult,  # noqa: E402
                    TechniqueScorer, executor_of, score)
from scorer import calibration as cal  # noqa: E402
from scorer.config import TECHNIQUE_CONFIG  # noqa: E402
from scorer.membership import ang_near, at_least, at_most, band  # noqa: E402
from scorer.spec import TECHNIQUE_SPEC  # noqa: E402

REF_DIR = REPO / "data" / "refs"
SCENE = REPO / "robots" / "wrestling_scene.xml"
SELF_MIN = 0.8
CROSS_MAX_DOUBLE_AS_SINGLE = 0.5
#: techniques whose bands advance monotonically through the movement (a phase
#: shift means "the body is late/early"); SPRAWL is a cyclic montage of three
#: sprawl/recover cycles and STANCE is a single held phase, so a shift there
#: lands back inside an equivalent band by construction (see reports/
#: 2026-10-08/scorer.md).
PROGRESSIVE = ("DOUBLE_LEG", "SINGLE_LEG", "BODY_LOCK", "SNAPDOWN", "STAND_UP")


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(SCENE))


@pytest.fixture(scope="module")
def scorer(model):
    return TechniqueScorer(model)


@pytest.fixture(scope="module")
def refs():
    """{tech: (self_qpos, opp_qpos)} with the reference executor as ``self``."""
    out = {}
    for tech in TECHNIQUES:
        d = np.load(REF_DIR / f"{tech}.npz", allow_pickle=False)
        a, b = d["qpos_a"], d["qpos_b"]
        out[tech] = (a, b) if executor_of(tech) == "A" else (b, a)
    return out


def _trace_score(scorer, tech, st, ot):
    """Mean score over a trace, frame i judged at its own progress phase."""
    T = len(st)
    totals = [scorer.score(tech, scorer.phase_at_frac(tech, i / max(T - 1, 1)),
                           st[i], ot[i]).total for i in range(T)]
    return float(np.mean(totals))


def _add_joint_noise(st, sigma, seed=0):
    rng = np.random.default_rng(seed)
    out = st.copy()
    out[:, 7:36] += rng.normal(0.0, sigma, size=(len(st), 29))
    return out


def _tilt_root(st, angle):
    """Pitch the floating base by ``angle`` rad (delta quat about world +x)."""
    out = st.copy()
    w, x, y, z = out[:, 3], out[:, 4], out[:, 5], out[:, 6]
    ca, sa = np.cos(angle / 2.0), np.sin(angle / 2.0)
    out[:, 3] = ca * w - sa * x
    out[:, 4] = ca * x + sa * w
    out[:, 5] = ca * y - sa * z
    out[:, 6] = ca * z + sa * y
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    return out


def _shift_phase(scorer, tech, st, ot, delta):
    """Score every frame as if the body were ``delta`` (progress) late."""
    T = len(st)
    totals = [scorer.score(tech, scorer.phase_at_frac(tech, min(1.0, i / max(T - 1, 1) + delta)),
                           st[i], ot[i]).total for i in range(T)]
    return float(np.mean(totals))


def _yaw_translate(st, yaw, dx, dy):
    """Rotate the whole robot about world +z and translate it in the plane."""
    out = st.copy()
    c, s = np.cos(yaw), np.sin(yaw)
    out[:, 0] = c * st[:, 0] - s * st[:, 1] + dx
    out[:, 1] = s * st[:, 0] + c * st[:, 1] + dy
    aw, az = np.cos(yaw / 2.0), np.sin(yaw / 2.0)   # pre-multiply by a yaw quat
    w, x, y, z = st[:, 3], st[:, 4], st[:, 5], st[:, 6]
    out[:, 3] = aw * w - az * z
    out[:, 4] = aw * x - az * y
    out[:, 5] = aw * y + az * x
    out[:, 6] = aw * z + az * w
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    return out


# ---- API contract ----------------------------------------------------------

def test_score_result_contract(scorer, refs):
    st, ot = refs["DOUBLE_LEG"]
    r = scorer.score("DOUBLE_LEG", 1, st[40], ot[40])
    assert isinstance(r, ScoreResult)
    assert r.technique == "DOUBLE_LEG" and r.phase == 1
    assert r.phase_name == TECHNIQUE_CONFIG["DOUBLE_LEG"]["phases"][1][0]
    assert 0.0 <= r.total <= 1.0
    assert r.terms, "per-predicate breakdown must not be empty"
    assert set(r.features) == set(FEAT_ORDER)
    wsum = sum(w for _mu, w in r.terms.values())
    assert wsum > 0.0
    for mu, w in r.terms.values():
        assert 0.0 <= mu <= 1.0 and w > 0.0
    rebuilt = sum(w * mu for mu, w in r.terms.values()) / wsum
    assert abs(rebuilt - r.total) < 1e-12


def test_public_score_wrapper(scorer, model, refs):
    st, ot = refs["SPRAWL"]
    for i in (0, 50, 200):
        direct = scorer.score("SPRAWL", scorer.phase_at_frac("SPRAWL", i / (len(st) - 1)),
                              st[i], ot[i])
        wrapped = score("SPRAWL", direct.phase, st[i], ot[i], model)
        assert wrapped.total == direct.total
        assert wrapped.terms == direct.terms


def test_bad_arguments_rejected(scorer, refs):
    st, ot = refs["STANCE"]
    with pytest.raises(KeyError):
        scorer.score("NOT_A_TECHNIQUE", 0, st[0], ot[0])
    with pytest.raises(IndexError):
        scorer.score("SNAPDOWN", 2, st[0], ot[0])
    with pytest.raises(IndexError):
        scorer.score("SNAPDOWN", -1, st[0], ot[0])
    with pytest.raises(ValueError):
        scorer.score_trace("STANCE", st, ot[:-1])


def test_phase_bands_partition_the_timeline(scorer):
    for tech in TECHNIQUES:
        bands = TECHNIQUE_CONFIG[tech]["phases"]
        assert bands[0][1] == 0.0
        for (_n0, _a0, b0), (_n1, a1, _b1) in zip(bands, bands[1:]):
            assert b0 == a1, f"{tech}: phase bands must be contiguous"
        for frac in np.linspace(0.0, 1.0, 25):
            ph = scorer.phase_at_frac(tech, frac)
            assert 0 <= ph < len(bands)
        assert scorer.phase_at_frac(tech, 1.0) == len(bands) - 1


def test_score_trace_aggregation_matches_per_frame(scorer, refs):
    for tech in ("DOUBLE_LEG", "SPRAWL"):
        st, ot = refs[tech]
        res = scorer.score_trace(tech, st, ot)
        manual = [scorer.score(tech, scorer.phase_at_frac(tech, i / (len(st) - 1)),
                               st[i], ot[i]).total for i in range(len(st))]
        assert abs(res["mean"] - float(np.mean(manual))) < 1e-12
        assert sum(c for _n, _v, c in res["per_phase"]) == len(st)
        assert res["executor"] == executor_of(tech)


# ---- calibration quality (acceptance bar) ---------------------------------

def test_every_reference_scores_high_on_its_own_trace(scorer, refs):
    for tech in TECHNIQUES:
        st, ot = refs[tech]
        res = scorer.score_trace(tech, st, ot)
        per_phase = {n: v for n, v, _c in res["per_phase"]}
        assert all(np.isfinite(v) for v in per_phase.values())
        worst = min(per_phase.values())
        assert worst >= SELF_MIN, f"{tech}: phase {per_phase} below {SELF_MIN}"
        assert res["mean"] >= 0.9, f"{tech}: own-trace mean {res['mean']:.3f}"


def test_double_leg_trace_scores_low_as_single_leg(scorer, refs):
    st, ot = refs["DOUBLE_LEG"]
    mean = scorer.score_trace("SINGLE_LEG", st, ot)["mean"]
    assert mean < CROSS_MAX_DOUBLE_AS_SINGLE, mean


def test_own_trace_beats_every_cross_score(scorer, refs):
    """The judge discriminates: diagonal dominance over the cross matrix."""
    means, cross = {}, {}
    for src in TECHNIQUES:
        st, ot = refs[src]
        for dst in TECHNIQUES:
            m = scorer.score_trace(dst, st, ot)["mean"]
            (means if src == dst else cross)[(src, dst) if src != dst else src] = m
    for tech in TECHNIQUES:
        best_cross = max(v for (s, _d), v in cross.items() if s == tech)
        assert means[tech] > best_cross + 0.1, (
            f"{tech}: self {means[tech]:.3f} vs best cross {best_cross:.3f}")


def test_calibration_file_matches_the_fitting_rule(scorer, refs):
    """Re-fit every predicate from the reference traces and compare."""
    table = json.loads(PARAMS_PATH.read_text())["params"]
    feat_of = {n: i for i, n in enumerate(FEAT_ORDER)}
    for tech in TECHNIQUES:
        st, ot = refs[tech]
        T = len(st)
        F = np.array([scorer._feat.features(st[i], ot[i]) for i in range(T)])
        ph = np.array([scorer.phase_at_frac(tech, i / max(T - 1, 1)) for i in range(T)])
        for p, tmpl in TECHNIQUE_SPEC[tech]["templates"].items():
            sel = F[ph == p]
            assert sel.shape[0] > 0
            for feat, kind, _w in tmpl:
                want = np.asarray(cal.fit_params(kind, feat, sel[:, feat_of[feat]]))
                got = np.asarray(table[tech][str(p)][f"{feat}:{kind}"])
                assert got.shape == want.shape
                assert np.allclose(got, want, atol=1e-4), (tech, p, feat, kind)


def test_calibration_covers_exactly_the_templates():
    table = json.loads(PARAMS_PATH.read_text())["params"]
    for tech, spec in TECHNIQUE_SPEC.items():
        for p, tmpl in spec["templates"].items():
            keys = {f"{f}:{k}" for f, k, _w in tmpl}
            assert set(table[tech][str(p)]) == keys, (tech, p)


# ---- membership functions --------------------------------------------------

def test_band_membership_shape():
    assert band(0.5, 0.2, 0.8, 0.05, 0.95) == 1.0
    assert band(0.05, 0.2, 0.8, 0.05, 0.95) == 0.0
    assert band(0.95, 0.2, 0.8, 0.05, 0.95) == 0.0
    below = [band(x, 0.2, 0.8, 0.0, 1.0) for x in np.linspace(0.0, 0.2, 20)]
    above = [band(x, 0.2, 0.8, 0.0, 1.0) for x in np.linspace(0.8, 1.0, 20)]
    assert all(np.diff(below) >= 0.0) and all(np.diff(above) <= 0.0)


def test_one_sided_memberships_monotone():
    xs = np.linspace(0.0, 1.0, 50)
    most = [at_most(x, 0.3, 0.7) for x in xs]
    least = [at_least(x, 0.7, 0.3) for x in xs]
    assert all(np.diff(most) <= 0.0) and all(np.diff(least) >= 0.0)
    assert at_most(0.1, 0.3, 0.7) == 1.0 and at_most(0.7, 0.3, 0.7) == 0.0
    assert at_least(0.9, 0.7, 0.3) == 1.0 and at_least(0.3, 0.7, 0.3) == 0.0


def test_ang_near_wraps_around_pi():
    assert ang_near(0.0, 3.14159, 0.35, 1.0) == 1.0 or ang_near(-3.1, 3.14159, 0.35, 1.0) > 0.9
    assert ang_near(np.pi, np.pi, 0.2, 0.6) == 1.0
    assert ang_near(-np.pi + 0.05, np.pi, 0.2, 0.6) == 1.0  # seam is continuous
    assert ang_near(0.0, np.pi, 0.2, 0.6) == 0.0
    xs = np.linspace(np.pi, np.pi + 0.6, 25)
    vals = [ang_near(x, np.pi, 0.2, 0.6) for x in xs]
    assert all(np.diff(vals) <= 0.0)


# ---- perturbation monotonicity --------------------------------------------

def test_joint_noise_monotonically_reduces_score(scorer, refs):
    sigmas = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4)
    for tech in TECHNIQUES:
        st, ot = refs[tech]
        curve = [_trace_score(scorer, tech, _add_joint_noise(st, s), ot)
                 for s in sigmas]
        assert all(a >= b - 1e-9 for a, b in zip(curve, curve[1:])), (tech, curve)
        assert curve[-1] < curve[0] - 0.1, (tech, curve)


#: per-step tolerance for the tilt / phase-shift curves. A reference frame whose
#: at_least/at_most predicate sits in its soft ramp (the calibration core is a
#: p10/p90 band, so ~10% of reference frames do) can be nudged *into* the core by
#: a small perturbation, so the per-step decrease is asserted up to 5e-3 of the
#: score range while the end-to-end drop must be large.
STEP_TOL = 5e-3


def test_root_tilt_monotonically_reduces_score(scorer, refs):
    angles = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)
    for tech in TECHNIQUES:
        st, ot = refs[tech]
        curve = [_trace_score(scorer, tech, _tilt_root(st, a), ot) for a in angles]
        assert all(a >= b - STEP_TOL for a, b in zip(curve, curve[1:])), (tech, curve)
        assert curve[-1] < curve[0] - 0.1, (tech, curve)


def test_phase_shift_monotonically_reduces_score(scorer, refs):
    deltas = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
    for tech in PROGRESSIVE:
        st, ot = refs[tech]
        curve = [_shift_phase(scorer, tech, st, ot, d) for d in deltas]
        assert all(a >= b - STEP_TOL for a, b in zip(curve, curve[1:])), (tech, curve)
        # SNAPDOWN has only two bands, so the shift saturates early; a >=0.05
        # end-to-end drop still shows the phase menu is doing real work.
        assert curve[-1] < curve[0] - 0.05, (tech, curve)


# ---- invariance, speed, separation ---------------------------------------

def test_rigid_joint_transform_of_the_pair_leaves_score_unchanged(scorer, refs):
    for tech in ("DOUBLE_LEG", "SPRAWL"):
        st, ot = refs[tech]
        base = scorer.score_trace(tech, st, ot)["mean"]
        moved = scorer.score_trace(
            tech, _yaw_translate(st, 0.7, 0.3, -1.2),
            _yaw_translate(ot, 0.7, 0.3, -1.2))["mean"]
        assert abs(base - moved) < 1e-6, (tech, base, moved)


def test_score_is_under_one_millisecond(scorer, refs):
    st, ot = refs["SINGLE_LEG"]
    for i in range(50):  # warm up
        scorer.score("SINGLE_LEG", 1, st[i], ot[i])
    n = 300
    t0 = time.perf_counter()
    for i in range(n):
        scorer.score("SINGLE_LEG", i % 4, st[i % len(st)], ot[i % len(st)])
    per_call = (time.perf_counter() - t0) / n
    assert per_call < 1e-3, f"{per_call * 1e6:.0f} us per call"


def _imports(path: Path, module_scope_only: bool = False) -> set[str]:
    tree = ast.parse(path.read_text())
    nodes = tree.body if module_scope_only else list(ast.walk(tree))
    names: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            if node.level:  # relative import inside the package
                names.add("." * node.level + (node.module or ""))
    return names


def test_scorer_is_separate_from_value_functions():
    """Separation contract: the judge is not part of the actor-critic.

    Two directions are checked:
    * ``src/scorer`` never imports RL / value / reward / torch code;
    * the value-function modules (critic net, privileged critic input, PPO
      value loss, actor observation) never import the scorer at any scope, and
      no ``src/rl`` module imports it at module scope. The single sanctioned
      user is ``rl/reward.py``'s form-signal adapter, which must stay a lazy
      in-function import (MISSION: technique conformity is a shaping signal,
      never the return estimator).
    """
    forbidden = {"torch", "rl", "wrestling", "reward"}
    scorer_files = sorted((REPO / "src" / "scorer").glob("*.py"))
    assert scorer_files
    for f in scorer_files:
        mods = _imports(f)
        assert not {m.split(".")[0] for m in mods} & forbidden, (f.name, mods)
        assert not [m for m in mods if m.strip(".").split(".")[0] in ("rl", "reward")], \
            (f.name, mods)

    value_dir = REPO / "src" / "rl"
    if not value_dir.is_dir():
        pytest.skip("no src/rl package present")
    #: modules implementing the value function / its inputs
    critic_modules = ("net.py", "privileged.py", "ppo.py", "obs.py")
    for name in critic_modules:
        f = value_dir / name
        if f.exists():
            assert "scorer" not in _imports(f), f"{name} must not import the scorer"
    for f in sorted(value_dir.glob("*.py")):
        mods = {m.split(".")[0] for m in _imports(f, module_scope_only=True)}
        assert "scorer" not in mods, f"{f.name} imports the scorer at module scope"
        if f.name != "reward.py":
            assert "scorer" not in {m.split(".")[0] for m in _imports(f)}, \
                f"only rl/reward.py may use the scorer (offender: {f.name})"
