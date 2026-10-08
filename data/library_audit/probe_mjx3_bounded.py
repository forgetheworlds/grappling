"""BOUNDED MJX probe: hard wall-clock caps, incremental flushed output.

Run under `timeout <N>`; every stage prints one JSON line immediately so a kill
still leaves evidence. Caps: numpy 50 control steps; MJX batch 1 and 64 with
n_control=100 each. No unbounded loops, no unbounded compile wait (the outer
timeout is the cap).

Usage: timeout 150 <libtest venv>/bin/python probe_mjx3_bounded.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path("/home/ubuntu/grappling")
sys.path.insert(0, str(REPO / "src"))
OUT_DIR = REPO / "data/library_audit"
XML = OUT_DIR / "solo_scene_mjx.xml"
RESULTS = OUT_DIR / "mjx_bounded_probe.jsonl"

import mujoco  # noqa: E402
import mujoco.mjx as mjx  # noqa: E402


def stamp() -> dict:
    return {"loadavg": [round(x, 2) for x in os.getloadavg()],
            "cores": len(os.sched_getaffinity(0))}


def emit(obj: dict) -> None:
    obj["stamp"] = stamp()
    line = json.dumps(obj)
    print(line, flush=True)
    with RESULTS.open("a") as f:
        f.write(line + "\n")


def main() -> int:
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)  # a_stand
    emit({"stage": "model", "nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu),
          "nmocap": int(m.nmocap)})

    # numpy: 50 control steps x 10 substeps
    t0 = time.perf_counter()
    for _ in range(50):
        d.ctrl[:] = m.key_ctrl[0]
        for _ in range(10):
            mujoco.mj_step(m, d)
    dt = time.perf_counter() - t0
    emit({"stage": "numpy_50_control_steps", "wall_s": dt,
          "control_steps_per_s": 50 / dt})

    import jax
    import jax.numpy as jnp

    mx = mjx.put_model(m)
    d0 = mjx.put_data(m, d)

    def control_step(mx, dd, act):
        dd = dd.replace(ctrl=act)

        def sub(dd, _):
            return mjx.step(mx, dd), None

        dd, _ = jax.lax.scan(sub, dd, None, length=10)
        return dd

    for B, n in ((1, 100), (64, 100)):
        acts = jnp.broadcast_to(jnp.asarray(m.key_ctrl[0], np.float32), (n, B, m.nu))
        db = jax.tree.map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), d0)

        def rollout(dd, a):
            return jax.lax.scan(lambda dd, a: (control_step(mx, dd, a), None)[0], dd, a)

        fn = jax.jit(jax.vmap(rollout, in_axes=(0, 1)))
        t0 = time.perf_counter()
        r = fn(db, acts)  # first call = compile + execute
        jax.block_until_ready(r)
        compile_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        r = fn(db, acts)
        jax.block_until_ready(r)
        run_s = time.perf_counter() - t0
        emit({"stage": "mjx", "batch": B, "control_steps": n,
              "compile_plus_first_wall_s": compile_s, "run_wall_s": run_s,
              "env_steps_per_s": B * n / run_s})
    return 0


if __name__ == "__main__":
    sys.exit(main())
