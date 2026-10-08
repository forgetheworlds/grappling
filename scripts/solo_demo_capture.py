#!/usr/bin/env python3
"""S5 demo capture entry point (single-G1 penetration-step demonstrations).

Runs the stabilized teacher (``teacher.controller.RobotTeacher``) on the
operator's retargeted shot-entry track + its recovery, at 50 Hz control with
ten 2 ms physics substeps per tick, and writes per-tick (observation, action,
reference progress, contacts, CoM/support metrics) demo datasets under
``data/solo/demos/``.  ``--dry-run`` reports the plan (reference id, ticks,
files, estimate) without simulating.

Examples
--------
  # dry run: what a full capture would do (no simulator)
  PYTHONPATH=src .venv/bin/python scripts/solo_demo_capture.py --dry-run --measure

  # the real capture (8 perturbed episodes; ~see --dry-run estimate)
  PYTHONPATH=src .venv/bin/python scripts/solo_demo_capture.py \\
      --ref data/refs_video/shot_entry_full.npz \\
      --recover-ref data/refs_video/shot_recover.npz \\
      --role a --blend-s 0.8 --hold-s 0.5 --seed 0 --episodes 8 \\
      --perturb-sigma 0.01 --out data/solo/demos --name shot_entry_full_a_p0.01_s0_n8 \\
      --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solo.demo import (CEM_ACTIVE_DIMS, DEFAULT_ENTRY_REF,  # noqa: E402
                       DEFAULT_RATE_TICKS_PER_S, DEFAULT_RECOVER_REF,
                       DEMO_ROOT, CaptureSpec, plan_capture, run_capture)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Capture penetration-step demos (source-agnostic: CEM "
                    "dynamic retargeting / learned policy / teacher fallback).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--source", default="cem", choices=["cem", "policy", "teacher"],
                    help="demo source: 'cem'=sampling-based dynamic retargeting "
                         "(primary), 'policy'=an external learned controller, "
                         "'teacher'=the stabilised teacher (fallback only)")
    ap.add_argument("--ref", default=str(DEFAULT_ENTRY_REF),
                    help="retargeted shot-entry reference npz (qpos_a)")
    ap.add_argument("--recover-ref", default=str(DEFAULT_RECOVER_REF),
                    help="retargeted recovery reference npz (the rise)")
    ap.add_argument("--role", default="a", choices=["a"],
                    help="reference role (A = the operator)")
    ap.add_argument("--upto", type=float, default=None,
                    help="cut the entry at this reference time (s)")
    ap.add_argument("--blend-s", type=float, default=0.8,
                    help="min-jerk blend from the entry end to the recovery start")
    ap.add_argument("--recover-s", type=float, default=None,
                    help="cut the recovery track at this time (s)")
    ap.add_argument("--stand-s", type=float, default=1.2,
                    help="min-jerk rise from the recovery end onto the terminal "
                         "stance (operator rule: every demo ends standing)")
    ap.add_argument("--stance-width", type=float, default=0.30,
                    help="terminal-stance width (m; measured holdable stance)")
    ap.add_argument("--stance-height", type=float, default=0.79,
                    help="terminal-stance pelvis height (m; solo.stance)")
    ap.add_argument("--hold-s", type=float, default=0.5,
                    help="final hold of the terminal stance (s)")
    ap.add_argument("--start-offset", type=float, default=0.0,
                    help="start the episode at this reference time (s)")
    ap.add_argument("--perturb-sigma", type=float, default=0.0,
                    help="seeded initial-state joint noise (rad; base xy/yaw "
                         "wobble scales with it)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--out", default=str(DEMO_ROOT))
    ap.add_argument("--name", default=None,
                    help="dataset directory name (default: derived from config)")
    # --- CEM source ---------------------------------------------------------
    ap.add_argument("--cem-samples", type=int, default=450,
                    help="rollouts per CEM iteration (rule: >= 10x the window "
                         "search dimension)")
    ap.add_argument("--cem-iters", type=int, default=10,
                    help="max CEM iterations (early stop on stall)")
    ap.add_argument("--cem-knot-ticks", type=int, default=25,
                    help="ticks between residual knots")
    ap.add_argument("--cem-window", type=int, default=50,
                    help="receding-horizon window (ticks; 0 = whole episode)")
    ap.add_argument("--cem-stride", type=int, default=25,
                    help="ticks executed per plan before replanning")
    ap.add_argument("--cem-mask", default="legs_waist",
                    choices=sorted(CEM_ACTIVE_DIMS),
                    help="residual joint mask (search dimension = knots x active)")
    ap.add_argument("--cem-early-stop", type=float, default=0.02,
                    help="relative improvement per iteration to keep searching")
    ap.add_argument("--policy-module", default="",
                    help="source=policy: path/to/module.py:attr exposing "
                         "policy(actor_obs(115,)) -> action(29,)")
    ap.add_argument("--touch-repair", choices=["none", "lead"], default="none",
                    help="teacher fallback only: declare the lead foot planted "
                         "from its reference plant onward")
    ap.add_argument("--action-mode", choices=["absolute", "residual"],
                    default="absolute", help="env action contract to record")
    ap.add_argument("--residual-scale", type=float, default=0.5)
    ap.add_argument("--rate", type=float, default=DEFAULT_RATE_TICKS_PER_S,
                    help="measured ticks/s, used for the dry-run estimate")
    ap.add_argument("--measure", action="store_true",
                    help="with --dry-run: also measure the reference landmarks "
                         "(FK only, no simulation)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit without simulating")
    ap.add_argument("--check", dest="check", action="store_true", default=True,
                    help="run the executability checker on each trace")
    ap.add_argument("--no-check", dest="check", action="store_false")
    return ap


def _spec(args) -> CaptureSpec:
    return CaptureSpec(
        ref=args.ref, recover_ref=args.recover_ref, role=args.role,
        upto_s=args.upto, blend_s=args.blend_s, recover_s=args.recover_s,
        stand_s=args.stand_s, stance_width=args.stance_width,
        stance_height=args.stance_height, hold_s=args.hold_s,
        start_offset_s=args.start_offset,
        perturb_sigma=args.perturb_sigma, seed=args.seed,
        episodes=args.episodes, out_root=args.out, name=args.name,
        touch_repair=args.touch_repair, source=args.source,
        policy_module=args.policy_module,
        cem_samples=args.cem_samples, cem_iters=args.cem_iters,
        cem_knot_ticks=args.cem_knot_ticks, cem_window=args.cem_window,
        cem_stride=args.cem_stride, cem_mask=args.cem_mask,
        cem_early_stop=args.cem_early_stop, cem_seed=args.seed,
        action_mode=args.action_mode,
        residual_scale=args.residual_scale, check=args.check,
        rate_ticks_per_s=args.rate)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = _spec(args)

    if args.dry_run:
        plan = plan_capture(spec)
        if args.measure:
            from solo.demo import (_Ids, build_repaired_track, load_track,
                                   measure_spec, stance_qpos_for)
            from solo.scene import load_solo_model
            model = load_solo_model()
            entry = load_track(spec.ref)
            recover = load_track(spec.recover_ref)
            stand = stance_qpos_for(spec) if spec.stand_s > 0 else None
            repaired = build_repaired_track(entry, recover, blend_s=spec.blend_s,
                                            hold_s=spec.hold_s,
                                            recover_s=spec.recover_s,
                                            start_offset_s=spec.start_offset_s,
                                            upto_s=spec.upto_s, stand_qpos=stand,
                                            stand_s=spec.stand_s)
            exec_spec, _ = measure_spec(model, entry, recover, repaired,
                                        _Ids(model))
            plan["measured_reference"] = {
                "lead_side": exec_spec.lead_side,
                "lead_plant_t": exec_spec.lead_plant_t,
                "knee_side": exec_spec.knee_side,
                "knee_min_z": round(exec_spec.knee_min_z, 4),
                "knee_min_t": round(exec_spec.knee_min_t, 3),
                "knee_contact_z": round(exec_spec.knee_contact_z, 4),
                "knee_contact_t": exec_spec.knee_contact_t,
                "travel_forward_m": round(exec_spec.travel_forward_m, 4),
                "travel_lateral_m": round(exec_spec.travel_lateral_m, 4),
                "travel_net_m": round(exec_spec.travel_net_m, 4),
                "rise_pelvis_z": round(exec_spec.rise_pelvis_z, 4),
                "rise_knee_z": round(exec_spec.rise_knee_z, 4),
                "phase_segments": [list(x) for x in exec_spec.phase_segments],
            }
        print(json.dumps(plan, indent=1))
        return 0

    out = run_capture(spec, verbose=True)
    failed = [e for e in out["episodes"] if e["check_ok"] is False]
    print(json.dumps({"dir": out["dir"], "episodes": out["episodes"],
                      "check_failed": len(failed)}, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
