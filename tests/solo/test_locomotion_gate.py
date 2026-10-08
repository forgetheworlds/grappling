"""T2 (locomotion) gate discrimination tests: the gate must *decide*, not run.

The point of these tests is that a controller that does not step/translate
correctly is rejected **for the right reason**, on physically meaningful
outcomes measured on commands held out from training.  No reward value is a
criterion here (asserted below).

Cases (each asserts a verdict and prints the deciding metric):

1. STAND-STILL fails: StandHold/ZeroAction, commanded to move, fail tracking --
   while standing upright, alive and never terminating (i.e. they cannot pass by
   "being upright and alive").
2. FALL-FORWARD fails: the canonical exploit (lean forward, topple, be carried
   at the commanded speed) *does* displace the robot, and fails uprightness /
   the fall criteria.
3. SLIDE fails: displacement produced by dragging loaded feet fails the
   slip-while-loaded criterion, while a load-transfer mechanism (the body is
   moved only while both feet are airborne) passes that same criterion at
   comparable displacement -- so the criterion separates the two mechanisms.
   The test also *measures* that the point-sampled ``slip`` metric cannot see a
   slow drag (it reads the same as standing), which is why the gate's
   slip criterion is the position-based ``slip_ratio``.
4. The REFERENCE passes: StandHold under the tiny (non-held-out) command set is
   certified by every criterion -- the gate is not vacuously impossible.  Its
   passing depends on the command magnitude (the same controller under
   held-out commands fails tracking/distance), which is stated, not hidden.
5. Command-set integrity: every held-out command is outside the training
   command domain by a real margin, the locomotion *preset* cannot sample it,
   and the reversal (oscillation) schedule is held out too.

CPU-light by construction: a handful of 3-4 s episodes per case, fixed seeds,
no training, no lock.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from solo.baselines import (AirborneTransferController,  # noqa: E402
                            FallForwardController, PlantedFootDragController,
                            StandHoldController, ZeroActionController)
from solo.commands import (T2_TRAIN_RANGES, Command, CommandRanges)  # noqa: E402
from solo.env import TASKS, SoloEnv  # noqa: E402
from solo.eval import (GATES, HELDOUT_COMMANDS, T2_EPISODE_S,  # noqa: E402
                       T2_HELDOUT_MARGIN, T2_SETTLE_S, T2_THRESHOLDS,
                       evaluate, heldout_command_outside, heldout_command_plan,
                       is_heldout_command, t2_oscillation_schedule,
                       tiny_command_plan)
from solo.metrics import METRIC_FIELDS  # noqa: E402

GATE = GATES["locomotion"]
EPISODE_S = 4.0            # short episodes: settle + a few seconds of command
FORWARD = HELDOUT_COMMANDS[0]     # vx=+0.45 (fastest held-out forward command)
LATERAL = HELDOUT_COMMANDS[2]     # vy=+0.18 (held-out lateral shuffle)


def _agg(report: dict) -> dict:
    return report["aggregate"]


def run_controller(factory, plan, *, name: str) -> dict:
    """Evaluate ``factory`` on ``plan`` (episode count == plan length)."""
    rep = evaluate(factory, task="locomotion", seed0=0, episodes=len(plan),
                   command_plan=plan, gate=GATE, max_episode_s=EPISODE_S,
                   verbose=False, name=name)
    assert len(rep["episodes"]) == len(plan), (name, len(rep["episodes"]), len(plan))
    return rep


def failing_criteria(report: dict) -> list[str]:
    return [r for r in report["reasons"] if r.startswith("FAIL")]


def criterion_line(report: dict, metric: str) -> str:
    """The verdict line for ``metric`` (PASS/FAIL + measured value)."""
    for line in report["reasons"]:
        parts = line.split()
        if len(parts) > 1 and parts[1] == metric:
            return line
    raise KeyError(f"{metric} is not a criterion of {report['gate']['name']}")


# ---------------------------------------------------------------- 1. stand still
def test_stand_still_commanded_to_move_fails_tracking():
    """Standing still must FAIL the tracking criteria -- upright and alive is not
    enough.  StandHold is the strongest stand-still case: it never terminates
    (fall_rate 0) and stays upright, so only the tracking/distance criteria can
    reject it."""
    rep = run_controller(lambda env, seed: StandHoldController(),
                         [heldout_command_plan()[0], heldout_command_plan()[2]],
                         name="test_stand_hold")
    agg = _agg(rep)
    vx_line = criterion_line(rep, "vx_err_abs_mean")
    dist_line = criterion_line(rep, "dist_err_mean")
    print(f"\n[stand-still] verdict={rep['verdict']} "
          f"mean_upright={agg['mean_upright']} fall_rate={agg['fall_rate']} "
          f"travelled={agg['travelled_m_mean']} m "
          f"commanded={agg['commanded_m_mean']} m")
    print(f"[stand-still] deciding: {vx_line} | {dist_line}")

    assert rep["verdict"] == "not_certified"
    assert agg["fall_rate"] == 0.0 and agg["dorsal_rate"] == 0.0, agg
    assert agg["mean_upright"] >= 0.99, agg          # it is upright the whole time
    assert agg["vx_err_abs_mean"] > T2_THRESHOLDS["vx_err_abs_mean"], agg
    assert vx_line.startswith("FAIL"), vx_line
    assert agg["dist_err_mean"] > T2_THRESHOLDS["dist_err_mean"], agg
    assert dist_line.startswith("FAIL"), dist_line
    assert agg["travelled_m_mean"] < 0.5 * agg["commanded_m_mean"], agg

    # the same controller under a tiny, non-held-out command *is* certified --
    # so the tracking threshold is not simply unreachable
    tiny = run_controller(lambda env, seed: StandHoldController(), tiny_command_plan(),
                          name="test_stand_hold_tiny")
    print(f"[stand-still] same controller, tiny commands: verdict={tiny['verdict']} "
          f"vx_err={_agg(tiny)['vx_err_abs_mean']} "
          f"dist_err={_agg(tiny)['dist_err_mean']}")
    assert tiny["certified"] is True

    # zero-action fails too (it topples mid-ctrlrange, so it fails harder)
    zero = run_controller(lambda env, seed: ZeroActionController(), [heldout_command_plan()[0]],
                          name="test_zero_action")
    zagg = _agg(zero)
    print(f"[stand-still] zero-action: verdict={zero['verdict']} "
          f"vx_err={zagg['vx_err_abs_mean']} upright={zagg['mean_upright']}")
    assert zero["verdict"] == "not_certified"
    assert zagg["vx_err_abs_mean"] > T2_THRESHOLDS["vx_err_abs_mean"], zagg


# ------------------------------------------------------------- 2. fall forward
def test_fall_forward_displaces_and_fails_uprightness_or_fall():
    """The canonical exploit: pitch forward in the commanded direction and get
    carried at the commanded speed.  It *does* produce displacement -- the gate
    must reject it on uprightness and/or the fall criteria, not on displacement
    alone."""
    rep = run_controller(lambda env, seed: FallForwardController(), [heldout_command_plan()[0]],
                         name="test_fall_forward")
    agg = _agg(rep)
    up_line = criterion_line(rep, "mean_upright")
    fall_line = criterion_line(rep, "fall_rate")
    print(f"\n[fall-forward] verdict={rep['verdict']} "
          f"travelled={agg['travelled_m_mean']} m "
          f"commanded={agg['commanded_m_mean']} m "
          f"mean_upright={agg['mean_upright']} "
          f"fall_rate={agg['fall_rate']} dorsal_rate={agg['dorsal_rate']} "
          f"vx_err={agg['vx_err_abs_mean']} slip={agg['slip_mean']}")
    print(f"[fall-forward] deciding: {up_line} | {fall_line}")

    assert rep["verdict"] == "not_certified"
    # the exploit is real: it moves in the commanded direction
    assert agg["travelled_m_mean"] > 0.30, (
        "the fall-forward probe must actually displace the robot, otherwise it "
        f"is not the exploit the gate has to reject (travelled "
        f"{agg['travelled_m_mean']})")
    # ... and is rejected on the physical quality criteria
    assert up_line.startswith("FAIL") or fall_line.startswith("FAIL"), rep["reasons"]
    assert agg["mean_upright"] < T2_THRESHOLDS["mean_upright"], agg
    assert agg["fall_rate"] > T2_THRESHOLDS["fall_rate"], agg


# -------------------------------------------------------------------- 3. slide
def test_planted_foot_slide_fails_slip_and_is_separated_from_load_transfer():
    """A controller that produces displacement by dragging its loaded feet must
    fail the slip-while-loaded criterion; a controller that moves the body only
    while both feet are airborne (a load-transfer/step mechanism) passes that
    same criterion at comparable displacement.

    The printed numbers also show *why* the criterion is position-based: the
    point-sampled ``slip`` field reads the same for the drag as for a standing
    robot, because MuJoCo's friction constraint zeroes the relative foot
    velocity inside a substep while the positions still integrate.
    """
    plan = [heldout_command_plan()[0]]
    drag = run_controller(lambda env, seed: PlantedFootDragController(speed=0.25), plan,
                          name="test_slide_drag")
    step = run_controller(lambda env, seed: AirborneTransferController(), plan,
                          name="test_slide_transfer")
    d, s = _agg(drag), _agg(step)
    ratio_line = criterion_line(drag, "slip_ratio_mean")
    print(f"\n[slide] drag  : verdict={drag['verdict']} "
          f"slip_ratio={d['slip_ratio_mean']} (slip_travel={d['slip_travel_mean']} m "
          f"per {d['travelled_m_mean']} m travelled) slip={d['slip_mean']} "
          f"upright={d['mean_upright']} fall={d['fall_rate']}/{d['dorsal_rate']} "
          f"steps={d['steps_total_mean']} loaded_frac={d['loaded_step_frac_mean']}")
    print(f"[slide] transfer: verdict={step['verdict']} "
          f"slip_ratio={s['slip_ratio_mean']} (slip_travel={s['slip_travel_mean']} m "
          f"per {s['travelled_m_mean']} m travelled) slip={s['slip_mean']} "
          f"upright={s['mean_upright']} fall={s['fall_rate']}/{s['dorsal_rate']} "
          f"steps={s['steps_total_mean']} loaded_frac={s['loaded_step_frac_mean']}")
    print(f"[slide] deciding: {ratio_line}")

    # (a) the slide is rejected *on the slip criterion*
    assert drag["verdict"] == "not_certified"
    assert ratio_line.startswith("FAIL"), ratio_line
    assert d["slip_ratio_mean"] > T2_THRESHOLDS["slip_ratio_mean"], d
    # ... and it is not a fall, so only the slip/tracking/distance criteria can
    # reject it (this is the criterion that must do the work)
    assert d["fall_rate"] == 0.0 and d["dorsal_rate"] == 0.0, d
    assert d["mean_upright"] >= 0.99, d
    # it never lifts a foot: a "planted foot sliding", not a step
    assert d["steps_total_mean"] == 0, d
    assert d["loaded_step_frac_mean"] == 1.0, d

    # (b) the load-transfer mechanism passes the *same* criterion at a
    # comparable displacement -> the criterion separates the mechanisms
    assert s["travelled_m_mean"] >= 0.5 * d["travelled_m_mean"], (s, d)
    assert s["slip_ratio_mean"] <= T2_THRESHOLDS["slip_ratio_mean"], s
    assert criterion_line(step, "slip_ratio_mean").startswith("PASS")
    assert s["steps_total_mean"] > 0, s        # feet really leave the floor
    assert s["loaded_step_frac_mean"] < 1.0, s

    # (c) the honest gap: the point-sampled slip metric cannot separate them
    assert d["slip_mean"] <= T2_THRESHOLDS["slip_mean"], (
        "if this ever fires the point-sampled slip metric became sufficient; "
        f"re-derive the gate (drag slip {d['slip_mean']})")
    print(f"[slide] note: point-sampled slip does NOT separate them "
          f"(drag {d['slip_mean']} <= {T2_THRESHOLDS['slip_mean']} but "
          f"transfer {s['slip_mean']}); the gate therefore uses slip_ratio "
          f"(loaded-foot travel per metre travelled)")


# ---------------------------------------------------------------- 4. reference
def test_reference_controller_is_certified():
    """StandHold under the tiny command set must pass *every* criterion: a gate
    nothing can pass is vacuous.  Thresholds are not weakened for this: the
    reference passes with the thresholds the held-out baselines are rejected
    by, and it only passes because the commands are small."""
    rep = run_controller(lambda env, seed: StandHoldController(), tiny_command_plan(),
                         name="test_reference_tiny")
    agg = _agg(rep)
    print(f"\n[reference] verdict={rep['verdict']} upright={agg['mean_upright']} "
          f"fall={agg['fall_rate']} slip={agg['slip_mean']} "
          f"slip_ratio={agg['slip_ratio_mean']} vx_err={agg['vx_err_abs_mean']} "
          f"vy_err={agg['vy_err_abs_mean']} yaw_err={agg['yaw_err_abs_mean']} "
          f"dist_err={agg['dist_err_mean']} "
          f"travelled={agg['travelled_m_mean']} commanded={agg['commanded_m_mean']}")
    for line in rep["reasons"]:
        print(f"[reference]   {line}")

    assert rep["certified"] is True, rep["reasons"]
    assert not failing_criteria(rep), rep["reasons"]
    # every criterion in the gate is actually decided by a measured number
    for c in GATE.criteria:
        assert agg.get(c.metric) is not None, (c.metric, agg)
    # the tiny commands are genuinely small: the reference is not tracking at
    # speed, it is standing still where standing still is correct
    assert agg["commanded_m_mean"] < 0.25, agg

    # the *same* controller on held-out commands fails the same thresholds
    held = run_controller(lambda env, seed: StandHoldController(),
                          [heldout_command_plan()[0]], name="test_reference_heldout")
    hagg = _agg(held)
    print(f"[reference] same controller, held-out command: "
          f"verdict={held['verdict']} vx_err={hagg['vx_err_abs_mean']} "
          f"dist_err={hagg['dist_err_mean']} upright={hagg['mean_upright']}")
    assert held["verdict"] == "not_certified"
    assert hagg["vx_err_abs_mean"] > T2_THRESHOLDS["vx_err_abs_mean"], hagg


# --------------------------------------------------- 5. held-out set integrity
def test_heldout_commands_are_disjoint_from_the_training_domain():
    """Held-out means outside the training command domain -- a matter of ranges,
    not of seeds.  Also asserts the locomotion task preset cannot sample them,
    and that the reversal schedule is held out as well."""
    feasible = CommandRanges()
    for cmd in HELDOUT_COMMANDS:
        assert feasible.contains(cmd), cmd.as_dict()      # physically feasible
        assert not T2_TRAIN_RANGES.contains(cmd), cmd.as_dict()
        margins = heldout_command_outside(cmd)
        worst = max(margins.values())
        assert worst >= T2_HELDOUT_MARGIN - 1e-9, (cmd.as_dict(), margins)
        assert is_heldout_command(cmd), cmd.as_dict()
    print(f"\n[integrity] {len(HELDOUT_COMMANDS)} held-out commands, all outside "
          f"T2_TRAIN_RANGES(vx={T2_TRAIN_RANGES.vx}, vy={T2_TRAIN_RANGES.vy}, "
          f"wz={T2_TRAIN_RANGES.wz}) by >= {T2_HELDOUT_MARGIN} on >=1 axis, "
          f"max margin {max(max(heldout_command_outside(c).values()) for c in HELDOUT_COMMANDS)}")
    print("[integrity] margins: " + ", ".join(
        f"({c.vx:+.2f},{c.vy:+.2f},{c.wz:+.2f})->"
        f"{ {k: round(v, 2) for k, v in heldout_command_outside(c).items()} }"
        for c in HELDOUT_COMMANDS))

    # the env's locomotion preset is the training domain, so no sampler in the
    # default task can draw a held-out command
    preset = TASKS["locomotion"].command_sampler
    assert preset is not None
    assert preset.ranges == T2_TRAIN_RANGES, preset.ranges.as_dict()
    sampler = type(preset)(ranges=T2_TRAIN_RANGES, skills=preset.skills, seed=1)
    for _ in range(500):
        c = sampler.sample()
        assert T2_TRAIN_RANGES.contains(c), c.as_dict()
        assert not is_heldout_command(c), c.as_dict()
    # ... and the *feasible* range would cover the held-out set: that is the
    # overlap the preset change exists to remove
    covered = [c.as_dict() for c in HELDOUT_COMMANDS if feasible.contains(c)]
    assert len(covered) == len(HELDOUT_COMMANDS)
    assert not T2_TRAIN_RANGES.contains(Command(vx=0.45)) and \
        feasible.contains(Command(vx=0.45))

    # the reversal (command-oscillation) schedule is held out too
    osc = t2_oscillation_schedule()
    targets = [seg.command for seg in osc.segments]
    assert len(targets) >= 4, targets
    assert all(is_heldout_command(t) for t in targets), [t.as_dict() for t in targets]
    signs = {t.vx > 0 for t in targets}
    assert signs == {True, False}, signs
    print(f"[integrity] reversal schedule: {len(targets)} segments, "
          f"vx targets {sorted({round(t.vx, 2) for t in targets})}")


# ------------------------------------------------------------ gate invariants
def test_gate_uses_no_reward_value_and_every_metric_exists():
    """No criterion may reference a reward quantity, every criterion metric must
    be a declared gate metric and must be produced per step or per episode."""
    metrics = [c.metric for c in GATE.criteria]
    assert not [m for m in metrics if "reward" in m], metrics
    assert set(metrics) == {
        "vx_err_abs_mean", "vy_err_abs_mean", "yaw_err_abs_mean", "mean_upright",
        "fall_rate", "dorsal_rate", "slip_mean", "slip_ratio_mean",
        "dist_err_mean"}, metrics
    # the tracking/slip quantities the gate reads are recorded per control step
    for field in ("vx_err", "vy_err", "yaw_err", "slip", "slip_travel",
                  "body_step", "cmd_vx", "cmd_vy", "upright"):
        assert field in METRIC_FIELDS, field
    assert GATE.provisional is False
    assert T2_SETTLE_S > 0.0
    episode_s = T2_EPISODE_S
    assert episode_s > T2_SETTLE_S


def test_env_records_the_motion_fields_for_a_locomotion_episode():
    """The gate's numbers must come from the recorded trace, measured on the
    live env (not from a synthetic dict)."""
    from solo.eval import run_episode

    env = SoloEnv(task="locomotion", seed=0)
    summary = run_episode(env, StandHoldController(), 0,
                          command=heldout_command_plan()[0], max_steps=150)
    rows = summary["__rows"]
    assert rows, "no trace rows recorded"
    for field in ("vx_err", "vy_err", "yaw_err", "slip", "slip_travel",
                  "body_step", "cmd_vx", "cmd_vy"):
        assert any(r.get(field) is not None for r in rows[1:]), field
    for key in ("travelled_m", "commanded_m", "dist_err_m", "slip_travel_m",
                "slip_ratio", "vx_err_abs_mean", "vy_err_abs_mean",
                "yaw_err_abs_mean", "loaded_step_frac"):
        assert key in summary, key
    assert summary["commanded_m"] > 0.0
    assert summary["slip_ratio"] >= 0.0
    print(f"\n[env] locomotion episode: travelled={summary['travelled_m']} m "
          f"commanded={summary['commanded_m']} m "
          f"dist_err={summary['dist_err_m']} m slip_ratio={summary['slip_ratio']} "
          f"steps_settled={summary['n_steps_settled']} (of {len(rows)})")
