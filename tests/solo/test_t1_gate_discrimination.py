"""Discrimination test for the T1 balance gate (``solo.eval.GATES["balance"]``).

**The defect this file pins.**  The shipped T1 gate cannot be passed by *any*
existing controller, including the scripted ``stand_hold`` reference it was
supposedly calibrated from (see ``reports/2026-10-08/t1_gate_calibration.md``):

* ``mean_upright >= 0.95`` is set *above* the reference's own measured hold.
  The stored 40-episode, in-train-only battery (4/6/8/10/12 N*s x 8 dirs) has
  ``balance_stand_hold_s0.summary.json`` -> ``mean_upright = 0.922786`` (the
  "0.923" the gate's note cites), and the shipped 48-episode battery
  (4/8/12/16/20/25 N*s x 3 heights) has ``0.849118``.  ``0.95`` clears neither.
* ``max_recoverable_impulse_heldout >= 16`` is a *held-out-magnitude* bar, so it
  is identically 0.0 (vacuous) on the non-stepping subset, and ``fall_rate_heldout``
  computed over the 16/20/25 N*s magnitudes demands stepping recovery that is
  T3/T2 capability -- and it is *silently vacuous* once the battery is capped.

**The split this file asserts** (per-capability rule):

* **T1 certifies NON-STEPPING dynamic balance** on the in-band battery
  (magnitudes ``<= TRAIN_MAX_IMPULSE`` = 12 N*s), with the uprightness bar set
  *below* the measured reference hold.
* **Held-out is redefined as held-out *conditions*, not above-cap magnitudes.**
  Training pushes are fixed at one height (``curriculum.py``: ``height=0.95`` m);
  the held-out set is the off-training application heights {0.79, 1.10} m inside
  the in-band battery, which is non-empty and disjoint from the training
  conditions.
* the **stepping-recovery bars move to the stance/footwork stage**; nothing in
  T1 may claim active recovery (a perfectly static rigid stand satisfies the old
  ``recovery_success_rate`` / ``max_recoverable_impulse*`` criteria).

Behavioural: it drives the real ``TaskGate.verdict`` / ``solo.eval.aggregate``
over the *stored* per-episode summaries, and asserts certified / not-certified
outcomes plus the structural invariants below.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# The repo has no installed package / conftest; test files put src/ on the path
# themselves (same convention as tests/solo/test_config_plumbing.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from solo.eval import GATES, Criterion, TaskGate, aggregate  # noqa: E402
from solo.pushes import TRAIN_MAX_IMPULSE  # noqa: E402
from solo.stance_valid import STAND_HEIGHT, stance_valid  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
METRICS_DIR = REPO / "data" / "solo" / "metrics"

# --------------------------------------------------------------------------- #
# training push distribution (src/solo/curriculum.py:36-80, PushCurriculum)
# --------------------------------------------------------------------------- #
#: magnitudes unlocked over training (``magnitudes=(2,4,6,8,10,12)``) -- a
#: superset of the in-band battery magnitudes {4, 8, 12}, so magnitude is NOT a
#: held-out axis inside the band.
TRAIN_MAGNITUDES = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
#: world yaws trained at full progress (``max_directions=8``) -- all 8, so
#: direction is NOT a held-out axis for a fully-trained policy.
TRAIN_DIRECTIONS = 8
#: EVERY training push is applied at this height (``PushCurriculum.height=0.95``;
#: the docstring: "application height fixed at height (0.95 m chest) -- height
#: variation stays an evaluation axis").  This is the one disjoint axis.
TRAIN_PUSH_HEIGHT = 0.95
HELDOUT_HEIGHTS = (0.79, 1.10)
_HELDOUT_H = tuple(round(h, 2) for h in HELDOUT_HEIGHTS)

#: Metrics that *claim active recovery*.  A perfectly static rigid stand passes
#: them on small pushes (it survives without stepping), so they must live in the
#: stepping/footwork gate, never in the non-stepping T1 balance gate.
#: ``fall_rate_heldout`` is NOT here: it is a stability metric on held-out
#: *conditions*, not a recovery claim.
ACTIVE_RECOVERY = frozenset({
    "recovery_success_rate",
    "max_recoverable_impulse",
    "max_recoverable_impulse_heldout",
})

#: Frozen copy of the as-written T1 gate (src/solo/eval.py GATES["balance"] as of
#: commit 25069de).  Kept as a literal so the defect stays pinned even after the
#: gate is migrated.
AS_WRITTEN = TaskGate(
    "T1_balance",
    (Criterion("fall_rate", "<=", 0.05, "falls across the full battery"),
     Criterion("max_recoverable_impulse_heldout", ">=", 16.0,
               "provenance: StandHold measured 0.0 held-out, and 16 N*s is beyond "
               "the analytic non-stepping ceiling (~13 N*s)"),
     Criterion("mean_upright", ">=", 0.95, "provenance: StandHold measured 0.923"),
     Criterion("time_to_stability_mean", "<=", 1.0, "baseline table"),
     Criterion("com_offset_max", "<=", 0.20, "baseline table"),
     Criterion("recovery_success_rate", ">=", 0.9,
               "stabilised to stance after each push"),
     Criterion("fall_rate_heldout", "<=", 0.10, "held-out magnitudes only")),
    note="as-written; provisional placeholder",
)

#: Corrected T1 gate (PROPOSED, not yet applied -- see the report's PENDING-OWNER
#: diff).  In-band magnitudes only; uprightness bar 0.84 sits BELOW both the
#: reference hold measured on the in-band subset (0.913239) and the stored
#: full-battery hold (0.849118); held-out is by CONDITION (off-training height);
#: 16/20/25 N*s are reported-only and never gate; no active-recovery metric.
PROPOSED = TaskGate(
    "T1_balance",
    (Criterion("fall_rate", "<=", 0.05,
               "falls across the IN-BAND battery (magnitudes <= 12 N*s, all heights)"),
     Criterion("mean_upright", ">=", 0.84,
               "below StandHold's in-band hold 0.913239 and the stored "
               "full-battery hold 0.849118; never above the reference"),
     Criterion("time_to_stability_mean", "<=", 1.0, "s; baseline-table bar retained"),
     Criterion("com_offset_max", "<=", 0.20, "m; baseline-table bar retained"),
     Criterion("fall_rate_heldout", "<=", 0.10,
               "held-out CONDITIONS = off-training application heights "
               "{0.79,1.10} m inside the in-band battery (training pushes are all "
               "at 0.95 m, curriculum.py); true fall rate, matching fall_rate's "
               "predicate; StandHold measured 0.0625 over 16 eps; the 16-25 N*s "
               "magnitudes are REPORTED-ONLY and do not gate"),
     Criterion("survivor_valid_stance_rate", ">=", 1.0,
               "TERMINAL state: every non-terminated episode must END inside a "
               "valid stance (solo.stance_valid); StandHold measured 1.0 (11/11 "
               "survivors) on the in-band battery")),
    note="T1 = non-stepping dynamic balance, magnitudes <= TRAIN_MAX_IMPULSE; "
         "held-out = off-training heights; stepping recovery moved to T3",
)

#: Reference holds the uprightness bar must stay below, with provenance.
STORED_FULL_BATTERY_HOLD = 0.849118   # t1_gate_baselines.json -> controllers.stand_hold.mean_upright
SUBSET_REFERENCE_HOLD = 0.913239      # same controller, in-band (magnitudes <= 12 N*s)

#: Where the stepping-recovery bars move to (T3 stance/footwork stage).
PROPOSED_STEPPING = TaskGate(
    "T3_stance_stepping",
    (Criterion("max_recoverable_impulse", ">=", 16.0,
               "MUST be re-derived from a stepping-capable reference (v5/T3)"),
     Criterion("max_recoverable_impulse_heldout", ">=", 16.0,
               "MUST be re-derived from a stepping-capable reference (v5/T3); "
               "set verbatim from the defect only as a placeholder"),
     Criterion("fall_rate_heldout", "<=", 0.10, "held-out magnitudes only"),
     Criterion("recovery_success_rate", ">=", 0.9,
               "stabilised to stance after a held-out push")),
    note="stepping/footwork stage",
)


# --------------------------------------------------------------------------- #
# stored data loading
# --------------------------------------------------------------------------- #
def _reconstruct_metrics(episode: dict) -> dict:
    """Inject the ``metrics`` sub-dict that :func:`solo.eval.aggregate` reads.

    The stored ``.summary.json`` episodes keep a flattened ``metric_means``
    instead of the live ``metrics`` mapping; rebuilding the few keys the
    aggregate touches lets the **production** aggregator run unchanged.
    """
    mm = episode.get("metric_means", {})
    keys = ("upright", "tilt_deg", "pelvis_z", "vel_err", "yaw_err", "stance_err",
            "slip", "act_delta", "sat_frac", "limit_prox", "reward")
    metrics = {k: {"mean": mm.get(k)} for k in keys}
    metrics["upright"] = {"mean": mm.get("upright"), "min": mm.get("upright")}
    out = dict(episode)
    out["metrics"] = metrics
    return out


def _aggregate(episodes: list[dict]) -> dict:
    eps = [_reconstruct_metrics(e) for e in episodes]
    return aggregate(eps,
                     steps_total=sum(int(e["steps"]) for e in eps),
                     wall_total=sum(float(e["wall_s"]) for e in eps))


def _episodes(path: Path) -> list[dict]:
    if not path.exists():
        pytest.skip(f"stored baseline not present: {path.relative_to(REPO)}")
    return json.loads(path.read_text())["episodes"]


def _t1gate(name: str) -> list[dict]:
    return _episodes(METRICS_DIR / f"balance_t1gate_{name}_s0.summary.json")


def _in_band(episodes: list[dict]) -> list[dict]:
    """T1 scope: magnitudes inside the non-stepping envelope (<= 12 N*s)."""
    return [e for e in episodes if float(e["pushes"][0]["impulse"]) <= TRAIN_MAX_IMPULSE]


def _height(episode: dict) -> float:
    return round(float(episode["pushes"][0]["height"]), 2)


def _mark_heldout_conditions(episodes: list[dict]) -> list[dict]:
    """Re-tag ``heldout`` by CONDITION (off-training height), not magnitude.

    Mirrors the shipped ``battery_pushes``: held out iff above the training cap
    OR off the single training application height.  Within the in-band battery
    that is exactly the off-training heights, non-empty and disjoint from the
    training conditions.
    """
    out = []
    for e in episodes:
        e2 = dict(e)
        imp = float(e["pushes"][0]["impulse"])
        e2["heldout"] = (imp > TRAIN_MAX_IMPULSE
                         or abs(_height(e) - TRAIN_PUSH_HEIGHT) > 1e-9)
        out.append(e2)
    return out


def _heldout_subset(episodes: list[dict]) -> list[dict]:
    return [e for e in _in_band(episodes) if _height(e) in _HELDOUT_H]


def _with_terminal_stance(episodes: list[dict]) -> list[dict]:
    """Add ``ends_in_valid_stance`` to stored episodes via the shared predicate.

    The gate-battery summaries predate the terminal-stance criterion, so the
    per-episode flag is rebuilt here from the two terminal channels the stored
    summary carries (``final_upright``, ``final_pelvis_z``); the predicate skips
    the channels the stored data cannot supply (hull / pose / contacts).
    """
    out = []
    for e in episodes:
        e2 = dict(e)
        ok, _ = stance_valid(upright=e.get("final_upright"),
                             pelvis_z=e.get("final_pelvis_z"),
                             stand_height=STAND_HEIGHT)
        e2.setdefault("ends_in_valid_stance", bool(ok))
        out.append(e2)
    return out


def _t1_summary(episodes: list[dict]) -> dict:
    """Aggregate the in-band battery with the condition-based held-out metric.

    ``aggregate``'s ``fall_rate_heldout`` reads the episodes tagged ``heldout``
    (now the off-training heights) with the same predicate as ``fall_rate``.
    """
    band = _with_terminal_stance(_mark_heldout_conditions(_in_band(episodes)))
    return _aggregate(band)


def _t1_conditions(episodes: list[dict]) -> set:
    return {(float(e["pushes"][0]["impulse"]), round(float(e["pushes"][0]["direction_deg"]), 2),
             _height(e)) for e in episodes}


def _fmt(reasons: list[str]) -> str:
    return "\n      ".join(reasons)


# --------------------------------------------------------------------------- #
# training distribution vs held-out conditions
# --------------------------------------------------------------------------- #
def test_training_push_distribution_is_single_height():
    """Establish the training distribution the held-out set must dodge.

    Source: ``src/solo/curriculum.py:36-80`` (``PushCurriculum``): magnitudes
    (2,4,6,8,10,12) N*s unlocked progressively, up to 8 world yaws, application
    height FIXED at 0.95 m.  The trainer only overrides start/warmup/seed
    (``src/solo/train.py:134-137``), never the height.
    """
    from solo.curriculum import DEFAULT_CURRICULUM as c

    print(f"[training] magnitudes={c.magnitudes} max_directions={c.max_directions} "
          f"height={c.height}")
    assert c.height == TRAIN_PUSH_HEIGHT
    assert float(max(c.magnitudes)) <= TRAIN_MAX_IMPULSE
    # in-band battery magnitudes are all inside the training magnitude set
    band_mags = {float(e["pushes"][0]["impulse"]) for e in _in_band(_t1gate("stand_hold"))}
    assert band_mags <= set(TRAIN_MAGNITUDES)
    # ... and direction is fully covered at the end of the curriculum, so the
    # only permanently disjoint axis is the application height
    assert c.max_directions == TRAIN_DIRECTIONS


def test_heldout_conditions_are_nonempty_and_disjoint_from_training():
    eps = _t1gate("stand_hold")
    in_band = _in_band(eps)
    trained = [e for e in in_band if _height(e) == TRAIN_PUSH_HEIGHT]
    heldout = [e for e in in_band if _height(e) in [round(h, 2) for h in HELDOUT_HEIGHTS]]
    print(f"[split] in-band={len(in_band)} trained-height={len(trained)} "
          f"heldout-height={len(heldout)}")
    assert len(heldout) > 0, "held-out set must be non-empty (vacuous pass guard)"
    assert not (_t1_conditions(trained) & _t1_conditions(heldout)), (
        "held-out conditions must be disjoint from the training conditions")
    # every held-out episode is off the single training height
    assert all(_height(e) in [round(h, 2) for h in HELDOUT_HEIGHTS] for e in heldout)
    # battery shape: 8 episodes at each of the three heights
    counts = {h: sum(1 for e in in_band if _height(e) == h) for h in (0.79, 0.95, 1.10)}
    assert counts == {0.79: 8, 0.95: 8, 1.10: 8}


def test_gate_heldout_metric_is_computed_over_the_heldout_subset():
    """The held-out metric must read the held-out subset, not everything.

    Two things are pinned: (1) the metric is a *true fall rate* over the held-out
    subset -- the pre-fix ``aggregate`` used ``termination is not None`` and gave
    0.5625 for the same set, so 0.0625 pins the aligned predicate (a revert
    breaks this test); (2) with every held-out episode terminating, the metric
    must go to 1.0 and fail the criterion, so a degenerate held-out set cannot
    pass.
    """
    eps = _t1gate("stand_hold")
    healthy = _t1_summary(eps)
    assert healthy["fall_rate_heldout"] == pytest.approx(0.0625, abs=1e-6)
    print(f"[heldout-metric] true-fall-rate={healthy['fall_rate_heldout']} "
          f"(pre-fix any-termination value was 0.5625)")
    assert healthy["fall_rate_heldout"] != pytest.approx(0.5625, abs=1e-6), (
        "fall_rate_heldout must not use the any-termination predicate")

    poisoned = []
    for e in eps:
        e = dict(e)
        if _height(e) in _HELDOUT_H and float(e["pushes"][0]["impulse"]) <= TRAIN_MAX_IMPULSE:
            e["termination"] = "fall"
            e["stable"] = False
            e["recovered"] = False
            e["time_to_stability_s"] = None
        poisoned.append(e)
    bad = _t1_summary(poisoned)
    print(f"[heldout-metric] poisoned={bad['fall_rate_heldout']}")
    assert bad["fall_rate_heldout"] == pytest.approx(1.0, abs=1e-6)
    ok, reasons = PROPOSED.verdict(bad)
    assert not ok
    assert any(r.startswith("FAIL") and r.split()[1] == "fall_rate_heldout" for r in reasons)


# --------------------------------------------------------------------------- #
# (a) the gate is not vacuous: the reference passes the corrected T1 gate
# --------------------------------------------------------------------------- #
def test_reference_hold_is_measured():
    band = _in_band(_t1gate("stand_hold"))
    assert len(band) == 24
    assert {e["pushes"][0]["impulse"] for e in band} == {4.0, 8.0, 12.0}
    hold = _aggregate(band)["mean_upright"]
    print(f"[provenance] StandHold mean_upright on the in-band subset = {hold}")
    assert hold == pytest.approx(SUBSET_REFERENCE_HOLD, abs=1e-6)


def test_as_written_gate_rejects_the_reference_it_was_calibrated_from():
    eps = _t1gate("stand_hold")
    for label, summary in (("full battery", _aggregate(eps)),
                           ("in-band subset", _aggregate(_in_band(eps)))):
        ok, reasons = AS_WRITTEN.verdict(summary)
        print(f"[as-written / {label}] certified={ok}\n      {_fmt(reasons)}")
        assert not ok, f"as-written gate unexpectedly certified the reference on {label}"
    blockers = {r.split()[1] for r in AS_WRITTEN.verdict(_aggregate(_in_band(eps)))[1]
                if r.startswith("FAIL")}
    assert "mean_upright" in blockers
    assert AS_WRITTEN.criteria[2].threshold > SUBSET_REFERENCE_HOLD  # 0.95 > reference


def test_proposed_gate_certifies_the_reference_on_the_nonstepping_subset():
    summary = _t1_summary(_t1gate("stand_hold"))
    ok, reasons = PROPOSED.verdict(summary)
    print(f"[proposed / in-band] certified={ok}\n      {_fmt(reasons)}")
    assert ok, f"proposed T1 gate must certify the reference:\n{_fmt(reasons)}"
    # ... and does not certify the reference on the above-envelope (16-25 N*s) row
    beyond = [e for e in _t1gate("stand_hold") if e["heldout"]]
    assert not PROPOSED.verdict(_aggregate(beyond))[0]


# --------------------------------------------------------------------------- #
# (b) trivial controllers fail
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("controller", ["zero_action", "random_init_policy"])
def test_trivial_controllers_fail_the_proposed_gate(controller):
    summary = _t1_summary(_t1gate(controller))
    ok, reasons = PROPOSED.verdict(summary)
    print(f"[proposed / {controller}] certified={ok}\n      {_fmt(reasons)}")
    assert not ok, f"trivial controller {controller} must not pass T1"


def test_v5_checkpoint_does_not_pass_the_proposed_gate():
    eps = _episodes(METRICS_DIR / "balance_t1_monitor_100352_s0.summary.json")
    summary = _t1_summary(eps)
    ok, reasons = PROPOSED.verdict(summary)
    print(f"[proposed / v5@100k] certified={ok} upright={summary['mean_upright']} "
          f"fall_rate={summary['fall_rate']} heldout_fall={summary['fall_rate_heldout']}")
    print(f"      {_fmt(reasons)}")
    assert not ok, "v5 must not pass by construction; thresholds were not tuned for it"


# --------------------------------------------------------------------------- #
# (c) no criterion may credit a static controller with active recovery
# --------------------------------------------------------------------------- #
def test_t1_has_no_active_recovery_metric_and_the_bars_moved_to_stepping():
    t1_metrics = {c.metric for c in PROPOSED.criteria}
    as_written_metrics = {c.metric for c in AS_WRITTEN.criteria}
    print(f"[criteria] proposed T1 = {sorted(t1_metrics)}")
    print(f"[criteria] as-written T1 = {sorted(as_written_metrics)}")
    assert ACTIVE_RECOVERY & as_written_metrics
    assert not (ACTIVE_RECOVERY & t1_metrics), (
        "no metric in the non-stepping T1 gate may claim active recovery")
    stepping_metrics = {c.metric for c in PROPOSED_STEPPING.criteria}
    assert ACTIVE_RECOVERY <= stepping_metrics


def test_static_survivor_is_not_credited_with_recovery():
    """A never-falls / never-moves / perfectly-upright controller.

    It legitimately passes T1 (non-stepping balance *is* what a rigid stand
    does), but no T1 criterion may report it as "recovered" -- and it must fail
    the stepping gate, where recovery is actually required.
    """
    static = {"fall_rate": 0.0, "mean_upright": 0.999, "com_offset_max": 0.01,
              "time_to_stability_mean": 0.1, "fall_rate_heldout": 0.0,
              "survivor_valid_stance_rate": 1.0,
              # what a static controller would report if recovery were claimed:
              "recovery_success_rate": 1.0, "max_recoverable_impulse_heldout": 16.0}
    ok, reasons = PROPOSED.verdict(static)
    print(f"[proposed / static survivor] certified={ok}\n      {_fmt(reasons)}")
    assert ok  # stability-only pass, not a recovery credit
    assert not (ACTIVE_RECOVERY & {c.metric for c in PROPOSED.criteria})
    step_ok, _ = PROPOSED_STEPPING.verdict(static)
    assert not step_ok, "a purely static survivor must not pass the stepping gate"


# --------------------------------------------------------------------------- #
# (d) the unattainable-by-construction defect cannot return
# --------------------------------------------------------------------------- #
def test_uprightness_bar_is_below_the_measured_reference_hold():
    ref_hold = _aggregate(_in_band(_t1gate("stand_hold")))["mean_upright"]
    bar = next(c for c in PROPOSED.criteria if c.metric == "mean_upright")
    print(f"[boundary] proposed mean_upright bar={bar.threshold} "
          f"in_band_hold={ref_hold} stored_full_battery_hold={STORED_FULL_BATTERY_HOLD}")
    assert ref_hold == pytest.approx(SUBSET_REFERENCE_HOLD, abs=1e-6)
    assert bar.threshold <= ref_hold, (
        "uprightness bar raised above the in-band reference hold -> unattainable by construction")
    assert bar.threshold <= STORED_FULL_BATTERY_HOLD, (
        "uprightness bar raised above the stored full-battery reference hold (0.849118)")


def test_collapsed_but_non_terminating_controller_fails_on_uprightness():
    ref_hold = _aggregate(_in_band(_t1gate("stand_hold")))["mean_upright"]
    frozen_ok = {"fall_rate": 0.0, "mean_upright": ref_hold, "com_offset_max": 0.12,
                 "time_to_stability_mean": 0.12, "fall_rate_heldout": 0.0,
                 "survivor_valid_stance_rate": 1.0}
    collapsed = {"fall_rate": 0.0, "mean_upright": 0.30, "com_offset_max": 0.15,
                 "time_to_stability_mean": 0.50, "fall_rate_heldout": 0.0,
                 "survivor_valid_stance_rate": 0.0}
    assert PROPOSED.verdict(frozen_ok)[0] is True
    ok, reasons = PROPOSED.verdict(collapsed)
    print(f"[proposed / collapsed] certified={ok}\n      {_fmt(reasons)}")
    assert not ok
    assert any(r.startswith("FAIL") and r.split()[1] == "mean_upright" for r in reasons)


def test_uprightness_bar_is_not_vacuous():
    bar = next(c for c in PROPOSED.criteria if c.metric == "mean_upright").threshold
    for controller in ("zero_action", "random_init_policy"):
        summary = _t1_summary(_t1gate(controller))
        assert summary["mean_upright"] < bar
        assert not PROPOSED.verdict(summary)[0]


# --------------------------------------------------------------------------- #
# beyond-envelope magnitudes are reported, never gated
# --------------------------------------------------------------------------- #
def test_beyond_envelope_magnitudes_are_reported_but_not_gated():
    """16/20/25 N*s stay visible (reported) but must not gate T1.

    Held-out is defined by CONDITION (height), not by magnitude: the T1 gate
    must contain no magnitude-held-out criterion.
    """
    eps = _t1gate("stand_hold")
    beyond = [e for e in eps if e["heldout"]]  # magnitude-based flag from the battery
    assert {e["pushes"][0]["impulse"] for e in beyond} == {16.0, 20.0, 25.0}
    assert not any(c.metric.startswith("max_recoverable") for c in PROPOSED.criteria)
    reported = _aggregate(beyond)
    print(f"[reported, non-gating / beyond envelope] stand_hold mean_upright="
          f"{reported['mean_upright']} fall_rate={reported['fall_rate']} "
          f"recovery_success_rate={reported['recovery_success_rate']}")
    assert reported["n_episodes"] == 24
    # the held-out axis that gates T1 is height, not magnitude
    ho = next(c for c in PROPOSED.criteria if c.metric == "fall_rate_heldout")
    assert "height" in ho.note.lower()
    assert PROPOSED.verdict(_t1_summary(eps))[0] is True


# --------------------------------------------------------------------------- #
# derived non-stepping ceiling (from OUR model), vs the stale ~13 N*s claim
# --------------------------------------------------------------------------- #
def _per_yaw_nonstepping_ceiling() -> list[tuple[int, float, float]]:
    """J_reject per push yaw, derived from the shipped model + measured stance.

    ``J_reject ~= m * dCOP * sqrt(g / z_c)`` (Yang et al. 2020, eq. 6), where
    ``dCOP`` is the distance from the CoM ground projection to the support-hull
    border along the push direction.  Hull = convex hull of the 8 sole contact
    spheres in the ``a_stand`` pose; ``m``/``z_c`` from the compiled model.
    """
    import mujoco
    from scipy.spatial import ConvexHull
    from solo.scene import load_solo_model, stand_frame

    model = load_solo_model()
    mass = float(sum(float(model.body_mass[i]) for i in range(model.nbody)))
    data = mujoco.MjData(model)
    data.qpos[:] = stand_frame(model)[0]
    mujoco.mj_forward(model, data)
    sole = [g for g in range(model.ngeom)
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
            and float(model.geom_size[g][0]) == pytest.approx(0.005)]
    pts = np.array([data.geom_xpos[g][:2] for g in sole])
    com = np.array(data.subtree_com[0])
    z_c = float(com[2])
    hull = pts[ConvexHull(pts).vertices]
    out = []
    for j in range(8):
        th = 2.0 * np.pi * j / 8.0
        d = np.array([np.cos(th), np.sin(th)])
        d_cop = np.inf
        for i in range(len(hull)):
            a, b = hull[i], hull[(i + 1) % len(hull)]
            e = b - a
            M = np.array([[d[0], -e[0]], [d[1], -e[1]]])
            if abs(np.linalg.det(M)) < 1e-12:
                continue
            t, s = np.linalg.solve(M, a - com[:2])
            if t > 1e-9 and -1e-9 <= s <= 1 + 1e-9:
                d_cop = min(d_cop, float(t))
        out.append((j, d_cop, mass * d_cop * np.sqrt(9.81 / z_c)))
    return out


def test_nonstepping_ceiling_is_direction_dependent_not_the_stale_13n():
    """The gate's '~13 N*s' scalar ceiling is wrong; pin the derived range.

    Derived per push yaw from the shipped model (m=33.34 kg, z_c=0.692 m, sole
    hull), the rejection ceiling spans ~6.7-20.7 N*s.  Consequences the analysis
    depends on: 16 N*s is INSIDE the envelope laterally (so the impulse bar is
    not "beyond the ceiling" as the note claims), and even the 12 N*s training
    cap already exceeds the weakest-direction ceiling.  The magnitude cap is
    therefore about matching the TRAINING distribution, not a physics limit.
    """
    rows = _per_yaw_nonstepping_ceiling()
    js = [J for _, _, J in rows]
    print("[ceiling] per-yaw J_reject (N*s): "
          + ", ".join(f"{j_}:{J:.1f}" for j_, _, J in rows))
    print(f"[ceiling] range {min(js):.2f}..{max(js):.2f} N*s")
    assert max(js) >= 16.0, "16 N*s must be inside the envelope in some direction"
    assert min(js) < 12.0, "training cap 12 N*s already exceeds the weakest direction"
    assert min(js) < 13.0 < max(js), (
        "the stale '~13 N*s' scalar is not the ceiling in any direction")


# --------------------------------------------------------------------------- #
# migration guard: the shipped gate must be one of the two known states
# --------------------------------------------------------------------------- #
def _spec(gate: TaskGate) -> tuple:
    return tuple(sorted((c.metric, c.op, float(c.threshold)) for c in gate.criteria))


def test_shipped_balance_gate_is_exactly_as_written_or_exactly_proposed():
    spec = _spec(GATES["balance"])
    print(f"[shipped] T1 {spec}")
    assert spec in (_spec(AS_WRITTEN), _spec(PROPOSED)), (
        "shipped T1 gate is neither the pinned as-written defect nor the proposed "
        "correction -- thresholds drifted without a calibration record")
