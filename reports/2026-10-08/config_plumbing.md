# Config plumbing — CLI -> trainer -> env seam: two fixes + the missing flag

**Agent:** ConfigFix · **Date:** 2026-10-08
**Files touched:** `src/solo/train.py`, `scripts/solo_env_smoke.py`,
`tests/solo/test_config_plumbing.py`
**Tests:** `MUJOCO_GL=egl .venv/bin/python -m pytest tests/solo/ -q` -> **8 passed**.
**Service:** `solo-t1-v5` (pid 3196830) untouched — still logging (step 180224 /
iteration 88 at hand-off); source edits are safe because the process already
imported these modules.

## 1. INSTANCE 1 (highest risk) — resume silently rebuilt the env from defaults

`_run_train` constructed `SoloTrainer(cfg)` from the argparse-derived `cfg`
before `load()`, and `SoloTrainer.load()` never reconciles `action_mode`, so
resuming a residual checkpoint without re-passing `--action-mode residual`
trained through the **absolute** mapping. The checkpoint's own
`config.train` is now authoritative, with an explicit CLI flag as override —
via the existing `train.resolve_action_mode`, no second resolver.

Before:

```python
def _run_train(cfg: TrainConfig, args) -> int:
    if args.resume:
        trainer = SoloTrainer(cfg)          # env built from argparse defaults
        trainer.load(args.resume)           # load() never reconciles action_mode
        stats = trainer.run()
        if cfg.out:
            trainer.save()
    else:
        stats = train(cfg, on_sigint_save=not args.no_sigint_save)
```

After:

```python
def _run_train(cfg: TrainConfig, args) -> int:
    if args.resume:
        from dataclasses import replace
        from rl.checkpoint import load_checkpoint

        ckpt = load_checkpoint(args.resume)
        try:
            mode, scale = resolve_action_mode(ckpt, args.action_mode)
        except ValueError as exc:
            raise SystemExit(f"[solo.train] --resume {args.resume}: {exc}")
        if args.residual_scale is not None:      # explicit CLI override
            scale = float(args.residual_scale)
        if (mode, scale) != (cfg.action_mode, cfg.residual_scale):
            print(f"[solo.train] resume: checkpoint config wins over CLI defaults "
                  f"(action_mode={mode}, residual_scale={scale})")
        cfg = replace(cfg, action_mode=mode, residual_scale=scale)
        trainer = SoloTrainer(cfg)
        trainer.load(args.resume)
        stats = trainer.run()
        if cfg.out:
            trainer.save()
    else:
        stats = train(cfg, on_sigint_save=not args.no_sigint_save)
```

To make "CLI flag not passed" detectable, `--action-mode` now defaults to
`None` (effective default `"absolute"`); `--residual-scale` likewise defaults
to `None` (effective default `0.5`). A one-line note was added to the module
docstring's resume flow.

## 2. INSTANCE 2 — the monitor read a checkpoint key that never exists

`scripts/solo_env_smoke.py:376` read `ckpt.get('cfg')` for `hidden`.
`rl/checkpoint.py:68` saves the config under `config`, so `hidden` always fell
back to `(256, 256)` and a non-default-hidden checkpoint raised a strict-load
size mismatch in `apply_checkpoint` instead of evaluating.

```diff
-    tc = (ckpt.get("cfg") or {}).get("train", {})
+    tc = (ckpt.get("config") or {}).get("train", {})
```

**Other `['cfg']` / `.get('cfg')` reads on a loaded checkpoint: NONE.** Grepped
`src/ scripts/ tests/` for `["cfg"]` and `.get("cfg")`; the only hit was the
line above. `scripts/eval_ppo.py:54` (`ckpt.get("config")`) and
`train.resolve_action_mode` (`ckpt.get("config")`) were already correct.

## 3. ADDED FLAG — `--residual-scale` (and two more one-liners)

`residual_scale` was a `TrainConfig` field applied to the env and saved, but had
no CLI argument, so `--action-mode residual` was locked to 0.5.

```python
ap.add_argument("--residual-scale", type=float, default=None,
                help="residual action magnitude (default 0.5); on --resume the "
                     "checkpoint's saved scale wins unless this is passed")
```

Wired into the cfg construction:
`residual_scale=(0.5 if args.residual_scale is None else args.residual_scale)`.

The two other un-plumbed `TrainConfig` fields were one-liners each, so both got
flags and were wired into the cfg:

```python
ap.add_argument("--push-seed", type=int, default=0,
                help="seed of the ramped training push schedule")
ap.add_argument("--torch-threads", type=int, default=1,
                help="torch intra-op thread count")
...
push_seed=args.push_seed, torch_threads=args.torch_threads,
```

## 4. Tests (extended `tests/solo/test_config_plumbing.py`)

`pytest tests/solo/ -q` -> **8 passed in ~15 s** (CPU-light, no sim lock; the
three CLI tests pass `--lock off` and stub `SoloTrainer.run`).

| test | covers |
|---|---|
| `test_env_side_config_reaches_the_live_env` | (existing) env-side values reach live env |
| `test_residual_mapping_reaches_step_single_and_mode_aware` | (existing) single-mapping |
| `test_checkpoint_round_trip_preserves_env_config` | (existing) save/load + the `ckpt['cfg'] is None` pin |
| `test_resume_uses_checkpoint_action_mode_not_cli_defaults` | (a) resume reconciles from checkpoint |
| `test_resume_cli_still_overrides_checkpoint` | explicit CLI still wins on resume |
| `test_monitor_reads_config_key_and_rebuilds_net_at_saved_width` | (b) eval reads `config`, rebuilds at saved width |
| `test_residual_scale_flag_reaches_the_live_env` | (c) `--residual-scale 0.37` reaches live env |
| `test_push_seed_and_torch_threads_flags_reach_cfg` | new one-liner flags |

### Pre-fix failure evidence

**Bug 1** — ran the resume test with the `_run_train` fix reverted in place
(fixed file restored immediately after). The assert that fails is
`assert tr.env.action_mode == MODE`:

```
>       assert tr.env.action_mode == MODE
E       AssertionError: assert 'absolute' == 'residual'
tests/solo/test_config_plumbing.py:326: AssertionError
Captured stdout call:
  [solo.train] startup check: mode=residual max|ctrl - base_action| = 0.002305 rad
  [solo.train] startup check: mode=absolute max|ctrl - base_action| = 1.397425 rad
```

The rebuilt trainer printed `mode=absolute max|ctrl-base| = 1.397425 rad`
(the absolute mapping writes the ctrlrange midpoint); after the fix the same
path reports `mode=residual max|ctrl-base| = 0.002305 rad` and the test asserts
`startup_ctrl_diff < 0.02`.

**Bug 2** — with a `(64, 32)`-hidden checkpoint, the pre-fix key read yields the
default and the strict load fails (so the checkpoint could not be evaluated):

```
pre-fix key read -> (256, 256) | fixed key read -> (64, 32)
pre-fix eval fails: Error(s) in loading state_dict for ActorCritic:
```

The test pins both halves: the fixed `config` read recovers `(64, 32)` and a
strict `apply_checkpoint` succeeds; the pre-fix `cfg` read falls back to
`(256, 256)` and `pytest.raises(RuntimeError)` is asserted.

## 5. Other `cfg` key reads

None other — a repo-wide grep for `['cfg']` / `.get('cfg')` found only the
single monitor line fixed in §2.
