#!/usr/bin/env python3
"""S1 solo-drill harness: smoke checks, evidence runs, baselines and probes.

Run from the repo root with the project venv:

    MUJOCO_GL=egl .venv/bin/python scripts/solo_env_smoke.py smoke
    MUJOCO_GL=egl .venv/bin/python scripts/solo_env_smoke.py holdability
    MUJOCO_GL=egl .venv/bin/python scripts/solo_env_smoke.py probes
    MUJOCO_GL=egl .venv/bin/python scripts/solo_env_smoke.py baselines
    MUJOCO_GL=egl .venv/bin/python scripts/solo_env_smoke.py all      # everything

What each subcommand does
-------------------------
``smoke``        env/model contract + a short stand-hold rollout (seconds).
``holdability``  reset-distribution holdability (16 seeds), commanded stance
                 width extremes, and measured throughput.
``throughput``   measured control steps/s on this host.
``probes``       the four exploit probes (+ random-init) through the eval
                 harness, with clips under ``videos/solo_drill/baselines/``.
``baselines``    T1 (balance, push battery) and T2 (locomotion) baselines for
                 the random-init policy and StandHold, same harness, clips.
``all``          smoke + holdability + probes + baselines (holds the sim lock).

Every run writes metrics under ``data/solo/metrics/`` and prints its own
verification (AGENTS.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solo import baselines as bl  # noqa: E402
from solo.baselines import (FallForwardController, RandomInitPolicyController,  # noqa: E402
                            SquatController, StandHoldController,
                            ZeroActionController, probe_specs)
from solo.commands import Command, CommandSchedule, Skill  # noqa: E402
from solo.env import RESET_TILT_DEG, SoloEnv  # noqa: E402
from solo.eval import (GATES, TRAIN_MAX_IMPULSE, battery_pushes,  # noqa: E402
                       evaluate, take_clips)
from solo.lock import SimLock  # noqa: E402
from solo.metrics import METRICS_DIR, write_json  # noqa: E402
from solo.scene import N_JOINTS, load_solo_model, stand_frame  # noqa: E402
from solo.stance import stance_qpos, stance_targets  # noqa: E402
from solo.video import ClipRecorder, probe_media  # noqa: E402

VIDEO_DIR = REPO / "videos" / "solo_drill" / "baselines"
CONTROLLERS = {
    "stand_hold": lambda: StandHoldController(),
    "zero_action": lambda: ZeroActionController(),
    "fall_forward": lambda: FallForwardController(),
    "squat_repeat": lambda: SquatController(),
    "random_init_policy": None,  # seeded per episode
}


def _controller_factory(name: str):
    if name == "random_init_policy":
        return lambda env, seed: RandomInitPolicyController(seed=seed)
    if name not in CONTROLLERS:
        raise SystemExit(f"unknown controller {name!r}; choose from {sorted(CONTROLLERS)}")
    return lambda env, seed: CONTROLLERS[name]()


# --------------------------------------------------------------------- smoke
def cmd_smoke(args) -> int:
    model = load_solo_model()
    env = SoloEnv(seed=args.seed)
    facts = env.config()["model"]
    assert facts["nq"] == 36 and facts["nu"] == 29
    assert abs(facts["timestep"] - 0.002) < 1e-12
    obs = env.reset(seed=args.seed)
    assert obs["actor"].shape == (115,) and obs["privileged"].shape == (43,)
    assert obs["critic"].shape == (158,)
    hold = env._ctrl_stand.copy()
    pz0 = float(env.data.qpos[2])
    q_a = env.data.qpos.copy()
    env.reset(seed=args.seed)
    assert np.array_equal(q_a, env.data.qpos), "reset not deterministic for a fixed seed"
    r_total = 0.0
    for k in range(250):  # 5 s
        obs, r, term, trunc, info = env.step(hold)
        r_total += r
        assert not term, info["termination"]
    pz1 = float(env.data.qpos[2])
    print("smoke:", {"model": {k: facts[k] for k in ("nq", "nv", "nu", "timestep")},
                     "control_dt": facts["control_dt"], "substeps": facts["substeps"],
                     "obs": {k: int(v.size) for k, v in obs.items()},
                     "stand_pelvis_z": [round(pz0, 4), round(pz1, 4)],
                     "reward_sum_5s": round(r_total, 3)})
    assert abs(pz1 - pz0) < 0.02, (pz0, pz1)
    print("VERIFIED: model contract + 5 s stand hold, no termination")
    return 0


# --------------------------------------------------------------- holdability
def cmd_holdability(args) -> int:
    model = load_solo_model()
    env = SoloEnv(seed=0)
    env.horizon = 1e9
    failures = []
    min_pz, max_tilt = 9.0, 0.0
    max_drift = 0.0
    for seed in range(args.seeds):
        env.reset(seed=seed)
        hold = env._ctrl_stand.copy()
        xy0 = env._pelvis_xy().copy()
        for k in range(100):  # 2 s
            obs, r, term, trunc, info = env.step(hold)
            min_pz = min(min_pz, info["metrics"]["pelvis_z"])
            max_tilt = max(max_tilt, info["metrics"]["tilt_deg"])
            max_drift = max(max_drift, float(np.linalg.norm(env._pelvis_xy() - xy0)))
            if term:
                failures.append({"seed": seed, "cause": info["termination"], "step": k})
                break
    print(f"reset holdability: {args.seeds - len(failures)}/{args.seeds} seeds held 2 s "
          f"(joint noise +/-0.03 rad, xy +/-0.02 m, yaw +/-10 deg, tilt +/-{RESET_TILT_DEG} deg) "
          f"| min pelvis_z {min_pz:.4f} | max tilt {max_tilt:.2f} deg "
          f"| max pelvis xy drift {max_drift:.4f} m")
    assert not failures, failures

    stance_rows = []
    for width in (0.23, 0.30, 0.42):
        env2 = SoloEnv(seed=0, jitter=False)
        env2.horizon = 1e9
        q = stance_qpos(width, 0.79, model)
        tgt = stance_targets(width, 0.79, model)
        env2.reset(seed=0, pose=q, jitter=False)
        pz_min, tilt_max, y0 = 9.0, 0.0, float(q[1])
        cause = None
        for k in range(100):
            obs, r, term, trunc, info = env2.step(tgt)
            pz_min = min(pz_min, info["metrics"]["pelvis_z"])
            tilt_max = max(tilt_max, info["metrics"]["tilt_deg"])
            if term:
                cause = info["termination"]
                break
        dy = abs(float(env2.data.qpos[1]) - y0)
        stance_rows.append({"width": width, "termination": cause,
                            "min_pelvis_z": round(pz_min, 4),
                            "max_tilt_deg": round(tilt_max, 3),
                            "lateral_drift_m": round(dy, 4)})
        print(f"stance width {width:.2f} m (h 0.79): term={cause} "
              f"min pelvis_z={pz_min:.4f} max tilt={tilt_max:.2f} deg drift={dy:.4f} m")
    assert all(r["termination"] is None for r in stance_rows), stance_rows

    steps = 1000
    env3 = SoloEnv(seed=1)
    env3.horizon = 1e9
    hold = env3._ctrl_stand.copy()
    t0 = time.perf_counter()
    for _ in range(steps):
        env3.step(hold)
    dt = time.perf_counter() - t0
    print(f"throughput: {steps / dt:.1f} control steps/s ({steps * 0.02 / dt:.2f}x realtime, "
          f"{dt * 1000 / steps:.2f} ms/step)")
    write_json(METRICS_DIR / "holdability.json", {
        "seeds": args.seeds, "failures": failures,
        "min_pelvis_z": min_pz, "max_tilt_deg": max_tilt,
        "max_pelvis_xy_drift_m": round(max_drift, 5), "stances": stance_rows,
        "throughput_steps_per_s": round(steps / dt, 2),
    })
    print("VERIFIED: reset distribution holds; stance widths stable; throughput measured")
    return 0


# --------------------------------------------------------------- throughput
def cmd_throughput(args) -> int:
    env = SoloEnv(seed=0)
    env.horizon = 1e9
    hold = env._ctrl_stand.copy()
    n = int(args.steps)
    t0 = time.perf_counter()
    for _ in range(n):
        env.step(hold)
    dt = time.perf_counter() - t0
    print(json.dumps({"steps": n, "wall_s": round(dt, 3),
                      "steps_per_s": round(n / dt, 2),
                      "realtime_x": round(n * 0.02 / dt, 2)}))
    return 0


def _clip_factory(title: str, subtitle: str, max_frames: int, only_episode: int | None):
    def make(i: int, seed: int, env: SoloEnv):
        if only_episode is not None and i != only_episode:
            return None
        return ClipRecorder(title, f"{subtitle} | seed={seed}",
                            max_frames=max_frames)
    return make


def _render(clip, model, stem: str, verdict_line: str, caption: str) -> dict:
    mp4 = clip.render(model, VIDEO_DIR / f"{stem}.mp4",
                      extra_lines=(verdict_line,))
    sheet = clip.contact_sheet(model, VIDEO_DIR / f"{stem}_sheet.png",
                               caption=caption)
    media = {"video": mp4, "sheet": sheet, "ffprobe": probe_media(mp4["path"])}
    print(f"  media {stem}: {media['ffprobe']}")
    return media


def _verdict_line(report: dict) -> str:
    agg = report["aggregate"]
    return (f"RUN VERDICT: {report['verdict'].upper()} | fall_rate={agg['fall_rate']:.2f} "
            f"upright={agg['mean_upright'] if agg['mean_upright'] is not None else float('nan'):.3f} "
            f"max_J={agg['max_recoverable_impulse']:.1f} N*s")


# ------------------------------------------------------------------ probes
def _run_probe(spec, *, video: bool, out_dir: Path, clip_seconds: float) -> dict:
    model = load_solo_model()
    gate = GATES[spec.task]
    if spec.task == "balance":
        pushes = battery_pushes(magnitudes=(4.0, 8.0, 12.0), directions=8, seed=11,
                                heights=(0.95,))
        clip_ep = len(pushes) - 1
        kwargs = dict(push_plan=pushes, name=f"probe_{spec.name}")
        title = f"S1 exploit probe - {spec.name}"
        subtitle = spec.description
    elif spec.task == "locomotion":
        kwargs = dict(episodes=6, command_plan=[spec.command] * 6, name=spec.name)
        clip_ep = 0
        title = f"S1 exploit probe - {spec.name}"
        subtitle = f"{spec.description} | cmd vx={spec.command.at(0).vx:+.2f}"
    else:  # stance
        kwargs = dict(episodes=6, command_plan=[spec.command] * 6, name=spec.name)
        clip_ep = 0
        title = f"S1 exploit probe - {spec.name}"
        subtitle = (f"{spec.description} | cmd stance_h="
                    f"{spec.command.at(0).stance_height:.2f}")
    factory = _clip_factory(title, subtitle, int(clip_seconds * 30), clip_ep) if video else None
    report = evaluate(spec.controller_factory, task=spec.task, seed0=0, gate=gate,
                      out_dir=out_dir, clip_factory=factory, verbose=False,
                      max_episode_s=4.0, **kwargs)
    clips = take_clips(report)
    if clips:
        verdict = _verdict_line(report)
        report["media"] = _render(clips[0], model, f"probe_{spec.name}",
                                  f"episode verdict: {verdict}",
                                  f"{spec.description} - outcome: "
                                  f"{'not certified' if not report['certified'] else 'certified'}")
    print(f"probe {spec.name:20s} task={spec.task:10s} verdict={report['verdict']:14s} "
          f"fall_rate={report['aggregate']['fall_rate']:.2f} "
          f"vel_err={report['aggregate']['vel_err_mean']} "
          f"upright={report['aggregate']['mean_upright']}")
    return report


def cmd_probes(args) -> int:
    out_dir = METRICS_DIR
    reports = {}
    for spec in probe_specs():
        reports[spec.name] = {k: v for k, v in
                              _run_probe(spec, video=not args.no_video,
                                         out_dir=out_dir,
                                         clip_seconds=args.clip_seconds).items()
                              if k != "__clips"}
    write_json(out_dir / "probes_summary.json", reports)
    for name, rep in reports.items():
        assert rep["verdict"] == "not_certified", (name, rep["verdict"])
    print("VERIFIED: no exploit probe is certified by the harness "
          f"({len(reports)} probes)")
    return 0


# --------------------------------------------------------------- baselines
def cmd_baselines(args) -> int:
    model = load_solo_model()
    out_dir = METRICS_DIR
    summary = {}
    t1_pushes = battery_pushes(magnitudes=(4.0, 6.0, 8.0, 10.0, 12.0), directions=8,
                               seed=7, heights=(0.95,))
    # --- T1 balance battery
    for name in ("stand_hold", "random_init_policy"):
        clip_ep = len(t1_pushes) - 1
        factory = (None if args.no_video else
                   _clip_factory(f"S1 baseline T1 balance - {name}",
                                 "push battery (J=4..12 N*s, 8 directions)",
                                 int(args.clip_seconds * 30), clip_ep))
        rep = evaluate(_controller_factory(name), task="balance", seed0=0,
                       push_plan=t1_pushes, gate=GATES["balance"], out_dir=out_dir,
                       clip_factory=factory, verbose=False, max_episode_s=4.0,
                       name=name)
        clips = take_clips(rep)
        if clips:
            rep["media"] = _render(clips[0], model, f"t1_{name}",
                                   _verdict_line(rep) + " | episode: "
                                   f"J={t1_pushes[clip_ep].impulse:.0f} N*s "
                                   f"dir={np.degrees(t1_pushes[clip_ep].direction):.0f} deg",
                                   f"T1 balance battery baseline - {name} - "
                                   f"{'certified' if rep['certified'] else 'not certified'}")
        summary[f"T1_{name}"] = {k: v for k, v in rep.items() if k != "__clips"}
        print(f"T1 {name:20s} verdict={rep['verdict']:14s} "
              f"fall_rate={rep['aggregate']['fall_rate']:.2f} "
              f"max_J={rep['aggregate']['max_recoverable_impulse']:.1f} "
              f"recovery={rep['aggregate']['recovery_success_rate']:.2f} "
              f"upright={rep['aggregate']['mean_upright']}")
    # --- T2 locomotion
    for name in ("stand_hold", "random_init_policy"):
        factory = (None if args.no_video else
                   _clip_factory(f"S1 baseline T2 locomotion - {name}",
                                 "seeded velocity commands",
                                 int(args.clip_seconds * 30), 0))
        rep = evaluate(_controller_factory(name), task="locomotion", seed0=0,
                       episodes=args.t2_episodes, gate=GATES["locomotion"],
                       out_dir=out_dir, clip_factory=factory, verbose=False,
                       max_episode_s=6.0, name=name)
        clips = take_clips(rep)
        if clips:
            rep["media"] = _render(clips[0], model, f"t2_{name}",
                                   _verdict_line(rep),
                                   f"T2 locomotion baseline - {name} - "
                                   f"{'certified' if rep['certified'] else 'not certified'}")
        summary[f"T2_{name}"] = {k: v for k, v in rep.items() if k != "__clips"}
        print(f"T2 {name:20s} verdict={rep['verdict']:14s} "
              f"vel_err={rep['aggregate']['vel_err_mean']} "
              f"yaw_err={rep['aggregate']['yaw_err_mean']} "
              f"upright={rep['aggregate']['mean_upright']} "
              f"fall_rate={rep['aggregate']['fall_rate']:.2f}")
    write_json(out_dir / "baselines_summary.json", summary)
    for key, rep in summary.items():
        assert rep["verdict"] == "not_certified", (key, rep["verdict"])
    print("VERIFIED: random-init and StandHold baselines are NOT certified on T1/T2 "
          "(they are the pre-training reference contrast)")
    return 0


def cmd_battery(args) -> int:
    pushes = battery_pushes(magnitudes=tuple(args.magnitudes), directions=args.directions,
                            seed=args.seed)
    rep = evaluate(_controller_factory(args.controller), task="balance", seed0=0,
                   push_plan=pushes, gate=GATES["balance"], out_dir=METRICS_DIR,
                   verbose=False, max_episode_s=4.0, name=args.controller)
    take_clips(rep)
    agg = rep["aggregate"]
    print(json.dumps({"controller": args.controller, "verdict": rep["verdict"],
                      "n_pushes": len(pushes), "fall_rate": agg["fall_rate"],
                      "max_recoverable_impulse": agg["max_recoverable_impulse"],
                      "recovery_success_rate": agg["recovery_success_rate"],
                      "steps_per_s": agg["steps_per_s"]}, indent=2))
    return 0



# ------------------------------------------------------------------- monitor
def cmd_monitor(args) -> int:
    """Evaluate a training checkpoint on the T1 gate battery (never race the writer).

    Copies the checkpoint first, builds the policy from its own config, runs the
    same 48-push battery the gate uses, and writes a comparison row against the
    baseline table (data/solo/metrics/t1_gate_baselines.json).
    """
    import shutil
    import torch

    from rl.checkpoint import apply_checkpoint, load_checkpoint
    from rl.net import ActorCritic, NetConfig
    from solo.baselines import PolicyController
    from solo.obs import ACTOR_DIM, CRITIC_DIM

    src = Path(args.checkpoint)
    if not src.exists():
        raise SystemExit(f"checkpoint {src} does not exist yet")
    tmp = Path("/tmp") / f"monitor_{src.name}"
    shutil.copy2(src, tmp)                      # never read the live file
    ckpt = load_checkpoint(str(tmp))
    steps = int((ckpt.get("state") or {}).get("steps_done", -1))
    tc = (ckpt.get("config") or {}).get("train", {})
    hidden = tuple(tc.get("hidden", (256, 256)))
    from solo.train import resolve_action_mode

    try:
        mode, residual_scale = resolve_action_mode(
            ckpt, None if args.action_mode == "auto" else args.action_mode)
    except ValueError as exc:
        raise SystemExit(f"[monitor] {exc}")
    net = ActorCritic(ACTOR_DIM, CRITIC_DIM, act_dim=N_JOINTS,
                      cfg=NetConfig(hidden=hidden, action_mode=mode,
                                    residual_scale=residual_scale))
    apply_checkpoint(ckpt, policy=net)
    net.eval()
    print(f"[monitor] checkpoint steps={steps} action_mode={mode} "
          f"residual_scale={residual_scale} (from checkpoint config)")

    pushes = battery_pushes(magnitudes=tuple(args.magnitudes), directions=args.directions,
                            heights=(0.79, 0.95, 1.10), seed=0)
    rep = evaluate(lambda env, seed: PolicyController(net, name=f"t1_monitor_{steps}",
                                                      stochastic=False),
                   task="balance", seed0=0, push_plan=pushes,
                   gate=GATES["balance"], out_dir=METRICS_DIR, verbose=False,
                   max_episode_s=4.0, name=f"t1_monitor_{steps}",
                   env_kwargs={"action_mode": mode, "residual_scale": residual_scale})
    take_clips(rep)
    agg = rep["aggregate"]
    base = {}
    bpath = METRICS_DIR / "t1_gate_baselines.json"
    if bpath.exists():
        base = json.loads(bpath.read_text()).get("controllers", {})
    row = {"steps": steps, "action_mode": mode,
           "residual_scale": residual_scale, "verdict": rep["verdict"],
           "reasons": rep["reasons"], "aggregate": agg,
           "baselines": {k: {m: v.get(m) for m in
                             ("fall_rate", "fall_rate_heldout",
                              "max_recoverable_impulse_heldout", "mean_upright",
                              "time_to_stability_mean", "com_offset_max",
                              "recovery_success_rate")}
                         for k, v in base.items()}}
    write_json(METRICS_DIR / f"t1_v2_monitor_{steps}.json", row)
    print(f"monitor steps={steps} verdict={rep['verdict']} "
          f"fall={agg['fall_rate']:.3f}/{agg['fall_rate_heldout']:.3f}(held) "
          f"upright={agg['mean_upright']} recovery={agg['recovery_success_rate']} "
          f"maxJ_held={agg['max_recoverable_impulse_heldout']} "
          f"t_stab={agg['time_to_stability_mean']} com_max={agg['com_offset_max']}")
    for b_name, b_row in base.items():
        print(f"  baseline {b_name:20s} fall={b_row['fall_rate']:.3f}/"
              f"{b_row['fall_rate_heldout']:.3f}(held) upright={b_row['mean_upright']} "
              f"recovery={b_row['recovery_success_rate']} "
              f"maxJ_held={b_row['max_recoverable_impulse_heldout']}")
    return 0

# ------------------------------------------------------------------- dispatch
def cmd_t1gate(args) -> int:
    """Extended T1 battery on the baseline trio -> the gate provenance table."""
    pushes = battery_pushes(magnitudes=tuple(args.magnitudes), directions=args.directions,
                            heights=(0.79, 0.95, 1.10), seed=args.seed)
    print(f"T1 gate battery: {len(pushes)} pushes = {len(args.magnitudes)} magnitudes "
          f"x {args.directions} directions x 3 heights "
          f"(held-out: impulses > {TRAIN_MAX_IMPULSE} N*s)")
    table = {}
    for name in ("zero_action", "stand_hold", "random_init_policy"):
        rep = evaluate(_controller_factory(name), task="balance", seed0=0,
                       push_plan=pushes, gate=GATES["balance"], out_dir=METRICS_DIR,
                       verbose=False, max_episode_s=4.0, name=f"t1gate_{name}")
        take_clips(rep)
        agg = rep["aggregate"]
        table[name] = {
            "verdict": rep["verdict"],
            "fall_rate": agg["fall_rate"],
            "fall_rate_heldout": agg["fall_rate_heldout"],
            "max_recoverable_impulse": agg["max_recoverable_impulse"],
            "max_recoverable_impulse_heldout": agg["max_recoverable_impulse_heldout"],
            "mean_upright": agg["mean_upright"],
            "time_to_stability_mean": agg["time_to_stability_mean"],
            "time_to_stability_rate": agg["time_to_stability_rate"],
            "com_offset_max": agg["com_offset_max"],
            "steps_after_push_mean": agg["steps_after_push_mean"],
            "recovery_success_rate": agg["recovery_success_rate"],
            "steps_per_s": agg["steps_per_s"],
        }
        print(f"T1gate {name:20s} verdict={rep['verdict']:14s} "
              f"falls={agg['fall_rate']:.2f}/{agg['fall_rate_heldout']:.2f}(held) "
              f"maxJ={agg['max_recoverable_impulse']:.1f}"
              f"/{agg['max_recoverable_impulse_heldout']:.1f}(held) "
              f"upright={agg['mean_upright']} "
              f"t_stab={agg['time_to_stability_mean']} "
              f"com_max={agg['com_offset_max']} steps={agg['steps_after_push_mean']}")
    write_json(METRICS_DIR / "t1_gate_baselines.json",
               {"battery": {"n_pushes": len(pushes),
                            "magnitudes": list(args.magnitudes),
                            "directions": args.directions,
                            "heights": [0.79, 0.95, 1.10],
                            "train_max_impulse": TRAIN_MAX_IMPULSE,
                            "seed": args.seed},
                "gate": GATES["balance"].as_dict(),
                "controllers": table})
    for name, row in table.items():
        assert row["verdict"] == "not_certified", (name, row["verdict"])
    print("VERIFIED: baseline trio is not certified by the extended T1 gate")
    return 0


def cmd_all(args) -> int:
    rc = cmd_smoke(args)
    rc |= cmd_holdability(args)
    rc |= cmd_probes(args)
    rc |= cmd_baselines(args)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("smoke", "holdability", "throughput",
                                        "probes", "baselines", "battery", "t1gate",
                                        "monitor", "all"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=16, help="holdability seeds")
    ap.add_argument("--steps", type=int, default=1000, help="throughput steps")
    ap.add_argument("--clip-seconds", type=float, default=4.0)
    ap.add_argument("--t2-episodes", type=int, default=8)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--controller", default="stand_hold",
                    help="battery subcommand: stand_hold|zero_action|random_init_policy|...")
    ap.add_argument("--magnitudes", type=float, nargs="+",
                    default=(4.0, 8.0, 12.0, 16.0, 20.0, 25.0))
    ap.add_argument("--directions", type=int, default=8)
    ap.add_argument("--checkpoint", default="checkpoints/solo/t1_balance_v2.pt")
    ap.add_argument("--action-mode", choices=("auto", "absolute", "residual"),
                    default="auto",
                    help="monitor: auto = read it from the checkpoint config "
                         "(refuses to guess if the checkpoint predates the field)")
    ap.add_argument("--lock-wait", type=float, default=900.0)
    ap.add_argument("--no-lock", action="store_true",
                    help="skip the advisory sim lock (only for short runs)")
    args = ap.parse_args()
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    fn = {"smoke": cmd_smoke, "holdability": cmd_holdability,
          "throughput": cmd_throughput, "probes": cmd_probes,
          "baselines": cmd_baselines, "battery": cmd_battery, "t1gate": cmd_t1gate,
          "monitor": cmd_monitor, "all": cmd_all}[args.command]
    needs_lock = args.command in ("probes", "baselines", "battery", "t1gate") \
        and not args.no_lock
    if needs_lock:
        with SimLock(owner=f"solo_env_smoke {args.command}", wait_s=args.lock_wait):
            return fn(args)
    return fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
