#!/usr/bin/env python3
"""Env bring-up smoke test for the ARM CPU-only sim stack.

Prints mujoco/numpy/torch versions, builds a trivial MJCF (ground plane +
free-joint sphere), steps 100 physics steps, and renders one offscreen PNG
to reports/2026-10-07/smoke_env.png using MUJOCO_GL=egl (software EGL via
Mesa on this GPU-less aarch64 host).

Run from repo root with the venv active:
    MUJOCO_GL=egl python scripts/smoke_env.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402  (needs MUJOCO_GL set before mujoco import)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PNG = REPO_ROOT / "reports" / "2026-10-07" / "smoke_env.png"

XML = """
<mujoco model="smoke_env">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.35 0.35 0.38 1"/>
    <body name="ball" pos="0 0 0.5">
      <freejoint name="ball_root"/>
      <geom name="ball_geom" type="sphere" size="0.08" mass="0.2"
            rgba="0.85 0.25 0.10 1"/>
    </body>
  </worldbody>
</mujoco>
"""

N_STEPS = 100


def main() -> int:
    print(f"mujoco   {mujoco.__version__}")
    print(f"numpy    {np.__version__}")
    print(f"torch    {torch.__version__} "
          f"(cuda.is_available={torch.cuda.is_available()}, "
          f"cpu.is_available={torch.cpu.is_available()})")
    print(f"MUJOCO_GL={os.environ['MUJOCO_GL']}")

    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    z_start = float(data.qpos[2])
    for _ in range(N_STEPS):
        mujoco.mj_step(model, data)
    z_end = float(data.qpos[2])
    print(f"physics  free sphere z {z_start:.3f} -> {z_end:.3f} m after "
          f"{N_STEPS} steps (t={data.time:.3f} s)")

    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.0, 0.0, 0.2]
    camera.distance = 2.0
    camera.elevation = -15
    camera.azimuth = 135

    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render()
    renderer.close()  # explicit close avoids EGL-free noise in __del__

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(OUT_PNG, rgb)
    size_kb = OUT_PNG.stat().st_size / 1024
    nonblack = float((rgb.sum(axis=-1) > 30).mean())
    print(f"render   {rgb.shape[0]}x{rgb.shape[1]} RGB -> {OUT_PNG} "
          f"({size_kb:.1f} KiB, nonblack {nonblack:.1%})")

    # --- self-checks ---
    assert torch.cuda.is_available() is False, "no GPU expected on this host"
    assert torch.cpu.is_available(), "torch CPU must be available"
    assert z_end < z_start, "sphere must fall under gravity"
    assert abs(data.time - N_STEPS * model.opt.timestep) < 1e-9, "sim clock"
    assert rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8
    assert nonblack > 0.5, "render must not be black"
    assert OUT_PNG.stat().st_size > 5 * 1024, "PNG must exceed 5 KiB"
    print("SMOKE ENV OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
