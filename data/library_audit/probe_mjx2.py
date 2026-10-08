"""MJX-CPU vs numpy/MuJoCo throughput on the project's REAL solo scene.

Protocol (load-bias aware): both paths are measured in the SAME process run,
back to back, on the SAME model file, with load average and CPU affinity
recorded before/after each block; a run can additionally be pinned with
`taskset -c 0`. Runs append to data/library_audit/mjx_throughput.json.

Model: the composed solo scene (floor + prefixed G1 + mocap markers) is dumped
with mj_saveLastXML; the 4 shoulder `cylinder` collision geoms are rewritten to
`capsule` (MJX has no (cylinder, mesh) collision function; capsule-mesh is
supported) -> `solo_scene_mjx.xml`.

Usage: <libtest venv>/bin/python probe_mjx2.py [--numpy-steps 600] [--mjx-steps 400]
                                               [--batches 1,64,256] [--tag NAME]
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np

REPO = pathlib.Path("/home/ubuntu/grappling")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
OUT_DIR = REPO / "data/library_audit"
RESULTS = OUT_DIR / "mjx_throughput.json"


def add_meshdir(xml_text: str, meshdir: str) -> str:
    """Insert meshdir= into the first <compiler> tag if absent."""
    import re

    m = re.search(r"<compiler\b[^>]*/>", xml_text)
    if not m:
        return xml_text
    tag = m.group(0)[:-2]
    if "meshdir" not in tag:
        tag += f' meshdir="{meshdir}"'
    return xml_text[: m.start()] + tag + "/>" + xml_text[m.end():]


def env_stamp() -> dict:
    out = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "loadavg": [round(x, 2) for x in os.getloadavg()],
        "affinity_cores": len(os.sched_getaffinity(0)),
        "cwd": str(pathlib.Path.cwd()),
    }
    try:
        p = subprocess.run(["pgrep", "-af", "solo.train"], capture_output=True, text=True)
        out["training_procs"] = p.stdout.strip().splitlines() or []
    except Exception as e:  # noqa: BLE001
        out["training_procs"] = [f"pgrep failed: {e}"]
    return out


def build_models() -> tuple:
    import mujoco

    from solo.scene import load_solo_model, stand_frame, SUBSTEPS

    m = load_solo_model()
    q, ctrl = stand_frame(m)

    dump = OUT_DIR / "solo_scene_dump.xml"
    xml_text = None
    try:
        from solo.scene import build_solo_spec

        xml_text = build_solo_spec().to_xml()
    except Exception as e:  # noqa: BLE001
        try:
            mujoco.mj_saveLastXML(str(dump), m)
        except Exception:  # noqa: BLE001
            mujoco.mj_saveLastXML(str(dump), m, str(OUT_DIR / "scene_assets"))
        xml_text = dump.read_text()
        print(f"(spec.to_xml failed: {type(e).__name__}; used mj_saveLastXML)", flush=True)
    dump.write_text(xml_text)
    text = dump.read_text()
    n_cyl = text.count('type="cylinder"')
    text = text.replace('type="cylinder"', 'type="capsule"')
    text = add_meshdir(text, str(REPO / "robots/g1/assets"))
    mjx_xml = OUT_DIR / "solo_scene_mjx.xml"
    mjx_xml.write_text(text)

    try:
        m2 = mujoco.MjModel.from_xml_path(str(mjx_xml))
        reload_note = "reload OK"
    except Exception as e:  # noqa: BLE001
        # meshes may need to sit next to the XML
        src_assets = REPO / "robots/g1/assets"
        dst = OUT_DIR / "assets"
        shutil.copytree(src_assets, dst, dirs_exist_ok=True)
        m2 = mujoco.MjModel.from_xml_path(str(mjx_xml))
        reload_note = f"reload after asset copy: {type(e).__name__}"
    fwd = {
        "nq": int(m2.nq), "nv": int(m2.nv), "nu": int(m2.nu),
        "nbody": int(m2.nbody), "ngeom": int(m2.ngeom), "nmocap": int(m2.nmocap),
        "timestep": float(m2.opt.timestep),
        "total_mass_orig": float(sum(m.body_mass)), "total_mass_mjx": float(sum(m2.body_mass)),
        "cylinders_rewritten": n_cyl, "reload": reload_note,
    }
    assert (m2.nq, m2.nv, m2.nu) == (m.nq, m.nv, m.nu)
    return m, m2, q, ctrl, SUBSTEPS, fwd


def numpy_bench(m, q, ctrl, substeps: int, n_control: int) -> dict:
    import mujoco

    d = mujoco.MjData(m)
    d.qpos[:] = q
    d.ctrl[:] = ctrl
    mujoco.mj_forward(m, d)

    def run(n: int):
        for _ in range(n):
            d.ctrl[:] = ctrl
            for _ in range(substeps):
                mujoco.mj_step(m, d)

    run(50)  # warmup
    before = env_stamp()
    t0 = time.perf_counter()
    run(n_control)
    dt = time.perf_counter() - t0
    after = env_stamp()
    return {
        "control_steps": n_control,
        "substeps": substeps,
        "control_steps_per_s": n_control / dt,
        "physics_steps_per_s": n_control * substeps / dt,
        "wall_s": dt,
        "loadavg_before": before["loadavg"], "loadavg_after": after["loadavg"],
    }


def _mjx_setup(mjx_xml: pathlib.Path, q, ctrl, substeps: int):
    import jax
    import jax.numpy as jnp
    import mujoco
    import mujoco.mjx as mjx

    m = mujoco.MjModel.from_xml_path(str(mjx_xml))
    mx = mjx.put_model(m)
    d = mujoco.MjData(m)
    d.qpos[:] = q
    d.ctrl[:] = ctrl
    mujoco.mj_forward(m, d)
    d0 = mjx.put_data(m, d)

    def control_step(mx, d, act):
        d = d.replace(ctrl=act)

        def sub(d, _):
            return mjx.step(mx, d), None

        d, _ = jax.lax.scan(sub, d, None, length=substeps)
        return d

    return m, mx, d0, control_step


def mjx_bench(mjx_xml: pathlib.Path, q, ctrl, substeps: int, batches, n_control: int) -> dict:
    import jax
    import jax.numpy as jnp

    m, mx, d0, control_step = _mjx_setup(mjx_xml, q, ctrl, substeps)
    out = {"batches": {}, "jax": jax.__version__, "devices": [str(x) for x in jax.devices()]}

    def rollout_single(d, acts):
        def body(d, a):
            return control_step(mx, d, a), None

        d, _ = jax.lax.scan(body, d, acts)
        return d

    fn = jax.jit(jax.vmap(rollout_single, in_axes=(0, 1)))
    for B in batches:
        db = jax.tree.map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), d0)
        acts = jnp.broadcast_to(jnp.asarray(ctrl, np.float32), (n_control, B, m.nu))
        r = fn(db, acts)  # compile + warmup
        jax.block_until_ready(r)
        before = env_stamp()
        t0 = time.perf_counter()
        r = fn(db, acts)
        jax.block_until_ready(r)
        dt = time.perf_counter() - t0
        after = env_stamp()
        phys = B * n_control * substeps / dt
        out["batches"][str(B)] = {
            "control_steps": n_control, "substeps": substeps,
            "control_steps_per_s": B * n_control / dt,
            "physics_steps_per_s": phys, "wall_s": dt,
            "loadavg_before": before["loadavg"], "loadavg_after": after["loadavg"],
        }
    return out


def mjx_envlike_bench(mjx_xml: pathlib.Path, q, ctrl, substeps: int, batch: int, n: int) -> dict:
    """Full-jitted env-like step: 10 substeps + obs concat + 2 reward terms + done."""
    import jax
    import jax.numpy as jnp
    import mujoco

    m, mx, d0, control_step = _mjx_setup(mjx_xml, q, ctrl, substeps)
    up_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "a_torso_link")

    def obs(d):
        return jnp.concatenate([d.qpos, d.qvel, d.ctrl])

    def env_step(mx, d, act):
        d = control_step(mx, d, act)
        tilt = 1.0 - d.xmat[up_body].reshape(3, 3)[2, 2]
        h = d.qpos[2]
        rew = jnp.exp(-5.0 * tilt) + 2.0 * jnp.exp(-8.0 * (h - 0.79) ** 2) - 0.001 * jnp.sum(d.ctrl**2)
        done = (h < 0.4) | (tilt > 0.5)
        return d, (obs(d), rew, done)

    def scan_env(d, acts):
        return jax.lax.scan(lambda d, a: env_step(mx, d, a), d, acts)

    fn = jax.jit(jax.vmap(scan_env, in_axes=(0, 1)))
    db = jax.tree.map(lambda x: jnp.broadcast_to(x, (batch,) + x.shape), d0)
    acts = jnp.broadcast_to(jnp.asarray(ctrl, np.float32), (n, batch, m.nu))
    r = fn(db, acts)
    jax.block_until_ready(r)
    before = env_stamp()
    t0 = time.perf_counter()
    r = fn(db, acts)
    jax.block_until_ready(r)
    dt = time.perf_counter() - t0
    after = env_stamp()
    return {
        "batch": batch, "control_steps": n, "substeps": substeps,
        "env_steps_per_s": batch * n / dt,
        "physics_steps_per_s": batch * n * substeps / dt,
        "wall_s": dt,
        "loadavg_before": before["loadavg"], "loadavg_after": after["loadavg"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--numpy-steps", type=int, default=600)
    ap.add_argument("--numpy-repeat", type=int, default=3)
    ap.add_argument("--mjx-steps", type=int, default=400)
    ap.add_argument("--batches", default="1,64,256")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    start = env_stamp()
    _, m2, q, ctrl, substeps, facts = build_models()
    print("model facts:", json.dumps(facts), flush=True)

    record: dict = {"tag": args.tag, "start": start, "model_facts": facts,
                    "substeps": substeps, "numpy_steps": args.numpy_steps,
                    "mjx_steps": args.mjx_steps}
    record["numpy_runs"] = [numpy_bench(m2, q, ctrl, substeps, args.numpy_steps)
                            for _ in range(args.numpy_repeat)]
    best = max(r["control_steps_per_s"] for r in record["numpy_runs"])
    record["numpy_best_control_steps_per_s"] = best
    print("numpy runs:", json.dumps(record["numpy_runs"]), flush=True)

    batches = [int(b) for b in args.batches.split(",") if b]
    record["mjx_raw"] = mjx_bench(OUT_DIR / "solo_scene_mjx.xml", q, ctrl, substeps,
                                  batches, args.mjx_steps)
    print("mjx_raw:", json.dumps(record["mjx_raw"]), flush=True)
    record["mjx_envlike"] = mjx_envlike_bench(OUT_DIR / "solo_scene_mjx.xml", q, ctrl,
                                              substeps, 256, args.mjx_steps)
    print("mjx_envlike:", json.dumps(record["mjx_envlike"]), flush=True)
    record["end"] = env_stamp()

    runs = []
    if RESULTS.exists():
        try:
            runs = json.loads(RESULTS.read_text()).get("runs", [])
        except Exception:  # noqa: BLE001
            runs = []
    runs.append(record)
    RESULTS.write_text(json.dumps({"runs": runs}, indent=1))
    print(f"appended run to {RESULTS}  (runs={len(runs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
