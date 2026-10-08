"""MuJoCo Playground (MJX) probe on CPU: G1JoystickFlatTerrain throughput.

Playground is the Google reference stack for G1 locomotion learning (MJX/brax/PPO).
This measures the env step rate on THIS aarch64 CPU box, batched, and reports the
env's obs/action sizes. Load-stamped like probe_mjx2.

Usage: <libtest venv>/bin/python probe_playground.py [--batch 256] [--steps 300]
Appends to data/library_audit/mjx_throughput.json (key "playground").
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import time

import numpy as np

REPO = pathlib.Path("/home/ubuntu/grappling")
OUT = REPO / "data/library_audit/playground_probe.json"


def env_stamp() -> dict:
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"),
           "loadavg": [round(x, 2) for x in os.getloadavg()],
           "affinity_cores": len(os.sched_getaffinity(0))}
    try:
        p = subprocess.run(["pgrep", "-af", "solo.train"], capture_output=True, text=True)
        out["training_procs"] = p.stdout.strip().splitlines() or []
    except Exception as e:  # noqa: BLE001
        out["training_procs"] = [f"pgrep failed: {e}"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--env", default="G1JoystickFlatTerrain")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from mujoco_playground import registry

    record: dict = {"env": args.env, "start": env_stamp()}
    env = registry.load(args.env)
    record["single_env"] = {
        "action_size": int(env.action_size),
        "obs_keys": sorted(env.observation_size.keys()) if isinstance(env.observation_size, dict)
                     else int(env.observation_size),
    }
    try:
        rec = env.mj_model  # may not exist
        record["mj_model"] = {"nq": int(rec.nq), "nv": int(rec.nv), "nu": int(rec.nu)}
    except Exception:  # noqa: BLE001
        pass

    key = jax.random.PRNGKey(0)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    # single env
    state = reset(key)
    state = step(state, jnp.zeros(env.action_size))
    jax.block_until_ready(state)
    before = env_stamp()
    t0 = time.perf_counter()
    for _ in range(50):
        state = step(state, jnp.zeros(env.action_size))
    jax.block_until_ready(state)
    dt = time.perf_counter() - t0
    record["single_env"]["env_steps_per_s"] = 50 / dt
    record["single_env"]["wall_s"] = dt
    record["single_env"]["loadavg_before"] = before["loadavg"]
    print(json.dumps({"stage": "playground_single", **record["single_env"]}), flush=True)

    # batched
    B = args.batch
    n = args.steps
    keys = jax.random.split(key, B)
    states = jax.jit(jax.vmap(env.reset))(keys)
    actions = jnp.zeros((n, B, env.action_size))

    def rollout(state, act):
        return jax.lax.scan(env.step, state, act)

    fn = jax.jit(jax.vmap(rollout, in_axes=(0, 1)))
    r = fn(states, actions)
    jax.block_until_ready(r)
    before = env_stamp()
    t0 = time.perf_counter()
    r = fn(states, actions)
    jax.block_until_ready(r)
    dt = time.perf_counter() - t0
    after = env_stamp()
    record["batched"] = {"batch": B, "steps": n, "wall_s": dt,
                         "env_steps_per_s": B * n / dt,
                         "loadavg_before": before["loadavg"],
                         "loadavg_after": after["loadavg"]}
    print(json.dumps({"stage": "playground_batched", **record["batched"]}), flush=True)
    record["end"] = env_stamp()

    OUT.write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
