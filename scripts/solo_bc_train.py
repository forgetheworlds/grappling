#!/usr/bin/env python
"""Build the BC corpus, fit the pose prior, evaluate it, roll it out open-loop.

    .venv/bin/python scripts/solo_bc_train.py            # full run (~1-2 min CPU)
    .venv/bin/python scripts/solo_bc_train.py --no-rollout --epochs 100

Artifacts written under ``data/solo/bc/``:

* ``dataset.npz``  -- obs (N,96) / ctrl / unit / ref_id / frame / phase + masks
* ``bc_policy.pt`` -- the trained actor + the frozen action/obs contract
* ``bc_metrics.json`` -- split, train/val curves, per-joint errors vs both
  constant baselines, phase report, ambiguity report, open-loop verdicts

NO RL training is launched here; this is the supervised pose prior only.
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

from solo.bc import (BC_DIR, BCTrainConfig, DEFAULT_REFS, GRAPPLEMAP_REFS,  # noqa: E402
                     VIDEO_REFS, ambiguity_report, build_dataset,
                     corpus_phase_report, joint_names, load_reference,
                     load_solo_model, rollout, save_dataset, split_overlap,
                     train_bc, train_report, validation_report)
from solo.imitation import DEFAULT_WEIGHTS, PRESETS, site_weight_table  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", nargs="*", default=list(DEFAULT_REFS))
    ap.add_argument("--out", type=Path, default=BC_DIR)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-rollout", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    model = load_solo_model()
    ds = build_dataset(tuple(args.refs), train_frac=args.train_frac, model=model)
    ov = split_overlap(ds)
    print(f"[bc] corpus: {len(ds)} pairs "
          f"({int(ds.train_mask.sum())} train / {int(ds.val_mask.sum())} val) "
          f"from {len(ds.names)} references  [{time.time() - t0:.1f}s]")
    print(f"[bc] split overlap: {len(ov['overlapping'])} (ref, frame) pairs "
          f"in both masks")
    assert not ov["overlapping"], ov["overlapping"][:5]

    phase = corpus_phase_report(ds)
    amb = ambiguity_report(ds)
    print(f"[bc] actor phase (offset 114): constant="
          f"{all(v['constant'] for v in phase.values())} "
          f"(value {next(iter(phase.values()))['min']}, skill {ds.skill})")
    nn = amb["val_to_train_nn"]
    print(f"[bc] obs->action ambiguity (1-NN in pose+velocity space): "
          f"val->train median {nn['act_err_median_rad']:.4f} rad, "
          f"p90 {nn['act_err_p90_rad']:.4f}, "
          f"{100 * nn['frac_gt_material']:.1f}% > {amb['material_rad']} rad")

    cfg = BCTrainConfig(hidden=tuple(args.hidden), epochs=args.epochs,
                        weight_decay=args.weight_decay, seed=args.seed,
                        torch_threads=args.threads)
    t1 = time.time()
    result = train_bc(ds, cfg, verbose=not args.quiet)
    policy = result["policy"]
    print(f"[bc] trained {args.epochs} epochs in {time.time() - t1:.1f}s; "
          f"best val MAE {result['best']['val_mae_rad']:.4f} rad "
          f"(epoch {result['best']['epoch']})")

    val = validation_report(policy, ds)
    trn = train_report(policy, ds)
    m = val["margins"]
    print(f"[bc] held-out joint MAE {val['bc']['mean_rad']:.4f} rad "
          f"vs stand {val['stand_baseline_rad']['mean_rad']:.4f} "
          f"(x{m['vs_stand_factor']:.2f}, -{100 * m['reduction_vs_stand']:.1f}%) "
          f"vs mean-pose {val['mean_baseline_rad']['mean_rad']:.4f} "
          f"(x{m['vs_mean_factor']:.2f})")
    print(f"[bc] train MAE {trn['mean_rad']:.4f} rad  (train/val gap "
          f"{val['bc']['mean_rad'] - trn['mean_rad']:+.4f})")

    names = joint_names(model)
    print("[bc] per-joint val MAE (rad):")
    for j, name in enumerate(names):
        print(f"     {name:32s} bc {val['bc']['per_joint_rad'][j]:.4f}  "
              f"stand {val['stand_baseline_rad']['per_joint_rad'][j]:.4f}  "
              f"mean {val['mean_baseline_rad']['per_joint_rad'][j]:.4f}")

    # persist the fit BEFORE the rollouts: a crash in the (slower) rollout loop
    # must never lose a 20-minute training run
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_dataset(ds, out / "dataset.npz")
    policy.meta.update({"corpus": list(ds.names), "split": ds.split,
                        "validation": val, "train": trn,
                        "ambiguity": amb["val_to_train_nn"],
                        "phase_constant": all(v["constant"] for v in phase.values()),
                        "mapping": ds.as_meta()["mapping"]})
    policy.save(out / "bc_policy.pt")

    def _write_metrics(rollouts: list) -> None:
        metrics = {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "corpus": {"video_refs": list(VIDEO_REFS),
                       "grapplemap_refs": list(GRAPPLEMAP_REFS),
                       "used": list(ds.names), "n": int(len(ds)),
                       "n_train": int(ds.train_mask.sum()),
                       "n_val": int(ds.val_mask.sum()),
                       "train_frac": ds.train_frac, "split": ds.split,
                       "overlap_pairs": len(ov["overlapping"]),
                       "actor_slice": [0, 96],
                       "mapping": ds.as_meta()["mapping"]},
            "config": cfg.as_dict(),
            "history": result["history"],
            "best": result["best"],
            "train": trn,
            "validation": val,
            "phase_report": phase,
            "ambiguity": amb,
            "rollouts": rollouts,
            "imitation": {"weights": DEFAULT_WEIGHTS.as_dict(),
                          "presets": {k: v.as_dict() for k, v in PRESETS.items()},
                          "site_weight_table": site_weight_table()},
        }
        (out / "bc_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    _write_metrics([])
    print(f"[bc] checkpoint + dataset + metrics written under {out}")

    rollouts = []
    if not args.no_rollout:
        t2 = time.time()
        for src in ds.names:
            v = rollout(policy, src, model=model)
            rollouts.append({k: v[k] for k in v if k != "rows"})
            if v["fell"]:
                outcome = f"FELL@{v['fall_time']:.2f}s ({v['termination']})"
            elif v["collapsed"]:
                outcome = f"sagged (pelvis {v['pelvis_z_min']:.2f} m)"
            else:
                outcome = "stayed up"
            print(f"[bc] open-loop {v['name']:22s} {v['steps']:4d}/"
                  f"{v['reference_steps']:4d} steps {outcome:34s} "
                  f"joint_err first-half {v['joint_err_first_half']:.4f} -> "
                  f"last-half {v['joint_err_last_half']:.4f} rad")
        print(f"[bc] rollouts in {time.time() - t2:.1f}s")
        _write_metrics(rollouts)

    print(f"[bc] artifacts -> {out} (dataset.npz, bc_policy.pt, bc_metrics.json) "
          f"[total {time.time() - t0:.1f}s]")
    print("[bc] RL refinement stage: NOT launched (see reports/2026-10-08/"
          "imitation.md for the exact command line)")


if __name__ == "__main__":
    main()
