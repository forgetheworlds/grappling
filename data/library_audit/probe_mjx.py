"""MJX-CPU throughput probe on the project's G1 model (aarch64, no GPU).

Measures, on the same 29-dof G1 MJCF:
  1. numpy/mujoco 3.15  : control steps/s  (1 control step = 10 physics substeps, our env's contract)
  2. MJX (jax 0.11.2 CPU): physics steps/s at batch 1/64/256, then control-step equivalents
  3. MJX + minimal obs/reward loop (jax lax.scan, batch 256) approximating a full RL env step

Usage: <libtest venv>/bin/python probe_mjx.py [--xml robots/g1/g1.xml]
Prints JSON to stdout and writes data/library_audit/mjx_throughput.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]


def timed(fn, repeat: int = 3):
    best = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best


def numpy_baseline(xml: str, control_steps: int = 3000) -> dict:
    import mujoco

    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    def run():
        for _ in range(control_steps):
            for _ in range(10):
                mujoco.mj_step(m, d)

    run()  # warmup
    dt = timed(run)
    return {
        "model": {"nq": m.nq, "nv": m.nv, "nu": m.nu, "timestep": m.opt.timestep},
        "control_steps": control_steps,
        "wall_s": dt,
        "control_steps_per_s": control_steps / dt,
        "physics_steps_per_s": 10.0 * control_steps / dt,
    }


def mjx_raw(xml: str, batches=(1, 64, 256), n_steps: int = 4000) -> dict:
    import jax
    import jax.numpy as jnp
    import mujoco
    import mujoco.mjx as mjx

    m = mujoco.MjModel.from_xml_path(xml)
    mx = mjx.put_model(m)
    d0 = mjx.make_data(m)

    out = {"jax_version": jax.__version__, "devices": [str(x) for x in jax.devices()],
           "batches": {}}

    def rollout(mx, d, n):
        def body(d, _):
            return mjx.step(mx, d), None

        d, _ = jax.lax.scan(body, d, None, length=n)
        return d

    for B in batches:
        db = jax.tree.map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), d0)
        stepper = jax.jit(rollout)
        r = stepper(mx, db, n_steps)  # compile + warmup
        jax.block_until_ready(r)
        t0 = time.perf_counter()
        r = stepper(mx, db, n_steps)
        jax.block_until_ready(r)
        dt = time.perf_counter() - t0
        phys = B * n_steps / dt
        out["batches"][str(B)] = {
            "physics_steps_per_s": phys,
            "control_steps_per_s": phys / 10.0,
            "wall_s": dt,
        }
        print(f"  MJX batch={B:4d}: {phys:,.0f} physics steps/s "
              f"= {phys / 10.0:,.0f} control steps/s", flush=True)
    return out


def mjx_envlike(xml: str, batch: int = 256, n: int = 512) -> dict:
    """Approximate a full RL env step: 10-substep physics + obs concat + simple reward."""
    import jax
    import jax.numpy as jnp
    import mujoco
    import mujoco.mjx as mjx

    m = mujoco.MjModel.from_xml_path(xml)
    mx = mjx.put_model(m)
    d0 = mjx.make_data(m)

    # actor-ish obs: qpos/qvel slices + ctrl; reward: upright-ish term from xmat
    qadr = np.arange(m.nq - m.nu, m.nq)  # joint qpos block (29)
    vadr = np.arange(m.nv - m.nu, m.nv)  # joint qvel block (29)

    def obs(d):
        return jnp.concatenate([d.qpos, d.qvel, d.ctrl])

    def sys_fn(mx, d, act):
        d = d.replace(ctrl=act)

        def sub(d, _):
            return mjx.step(mx, d), None

        d, _ = jax.lax.scan(sub, d, None, length=10)
        up = 1.0 - d.xmat[1].reshape(3, 3)[2, 2]  # torso tilt-ish
        rew = jnp.exp(-5.0 * jnp.abs(up)) * 0.1 - 0.001 * jnp.sum(jnp.square(d.ctrl))
        return d, (obs(d), rew)

    def rollout(mx, d, actions):
        return jax.lax.scan(lambda d, a: sys_fn(mx, d, a), d, actions)

    db = jax.tree.map(lambda x: jnp.broadcast_to(x, (batch,) + x.shape), d0)
    actions = jnp.zeros((n, batch, mx.nu))
    fn = jax.jit(rollout)
    res = fn(mx, db, actions)
    jax.block_until_ready(res)
    t0 = time.perf_counter()
    res = fn(mx, db, actions)
    jax.block_until_ready(res)
    dt = time.perf_counter() - t0
    phys = batch * n / dt
    return {
        "batch": batch, "steps": n, "wall_s": dt,
        "physics_steps_per_s": phys,
        "control_steps_per_s": phys / 10.0,
        "env_steps_per_s": phys / 10.0,  # 1 env step == 1 control step == 10 substeps
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=str(ROOT / "robots/g1/g1.xml"))
    ap.add_argument("--out", default=str(ROOT / "data/library_audit/mjx_throughput.json"))
    ap.add_argument("--skip-numpy", action="store_true")
    args = ap.parse_args()

    xml = args.xml
    print(f"G1 MJCF: {xml}")
    result: dict = {"xml": xml, "note": "1 control step = 10 physics substeps (0.02 s)"}
    if not args.skip_numpy:
        print("numpy/mujoco baseline ...", flush=True)
        result["numpy_mujoco"] = numpy_baseline(xml)
        print(f"  numpy: {result['numpy_mujoco']['control_steps_per_s']:,.0f} control steps/s",
              flush=True)
    print("MJX ...", flush=True)
    result["mjx_raw"] = mjx_raw(xml)
    print("MJX env-like loop ...", flush=True)
    result["mjx_envlike"] = mjx_envlike(xml)
    print(f"  MJX env-like batch=256: {result['mjx_envlike']['env_steps_per_s']:,.0f} env steps/s")

    pathlib.Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
