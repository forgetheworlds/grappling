# EnvSetup — ARM CPU-only Python sim stack bring-up

Date: 2026-10-07 · Agent: EnvSetup (wave 1) · Host: 4-core ARM Neoverse-N1
aarch64, 23 GB RAM, no GPU, Ubuntu 24.04, Python 3.12.3.

## Result: SUCCESS (all acceptance criteria met)

## Deliverables

- `.venv/` — project venv with the full stack
- `scripts/smoke_env.py` — version dump + MJCF sphere drop + offscreen render
- `docs/SETUP.md` — bare-Ubuntu-to-smoke-run instructions
- `reports/2026-10-07/smoke_env.png` — 640x480 RGB render, 30,651 bytes (>5 KiB)
- `reports/2026-10-07/env_setup.md` — this report

## Installed versions (pip freeze, key lines)

```
mujoco==3.15.0
numpy==2.5.3
scipy==1.18.1
pytest==9.1.1
imageio==2.38.0
imageio-ffmpeg==0.6.0
matplotlib==3.11.2
torch==2.14.1+cpu
```

pip 26.2.1 (upgraded from 24.0). Venv: `python3 -m venv .venv`, Python 3.12.3.

## Steps as executed

1. `python3 -m venv .venv` → `pip install --upgrade pip` → pip 26.2.1. OK.
2. `pip install mujoco numpy scipy pytest imageio imageio-ffmpeg matplotlib` — OK,
   one shot, ~19 s (also pulls glfw, pyopengl, pillow, absl-py).
3. **torch CPU (aarch64)**: `pip install torch --index-url
   https://download.pytorch.org/whl/cpu` worked on the FIRST try (no fallback to
   PyPI needed). Wheel: torch-2.14.1+cpu aarch64 linux.
   Verify: `python -c "import torch; print(torch.__version__, torch.cpu.is_available())"`
   → `2.14.1+cpu True`; `torch.cuda.is_available()` → `False` (CPU-only confirmed).
4. **Headless MuJoCo rendering**: `MUJOCO_GL=egl` works out of the box
   (software EGL via Mesa; `libegl1` was already installed). Offscreen
   `Renderer.render()` returns a valid RGB frame (mean 60.3, max 255, 83%
   non-black on the probe scene). No apt installs were needed.
   - Cosmetic stderr noise (expected, harmless, exit code 0):
     `libEGL warning: failed to open /dev/dri/renderD128: Permission denied`
     (x2 + card0) — Mesa probing DRM devices on a GPU-less host before
     falling back to software rendering. `EGL_PLATFORM=surfaceless` does NOT
     silence it; MuJoCo already uses the surfaceless platform.
   - Telemetry: relying on `Renderer.__del__` at interpreter exit prints
     `OpenGL.raw.EGL._errors.EGLError` "Exception ignored in __del__" noise.
     Explicit `renderer.close()` before exit avoids it entirely —
     `scripts/smoke_env.py` does this; documented in SETUP.md.
   - osmesa NOT tested as primary (EGL succeeded); fallback documented in
     SETUP.md troubleshooting (`libosmesa6-dev` + `MUJOCO_GL=osmesa`).
5. `scripts/smoke_env.py` written; run from repo root with venv active —
   stdout pasted below, exit 0.
6. `docs/SETUP.md` written: exact commands (apt → venv → pip → torch cpu
   index → `export MUJOCO_GL=egl` → run smoke), expected output, and a
   troubleshooting table.

## smoke_env.py stdout (verification run, 2026-10-07)

Command: `source .venv/bin/activate && MUJOCO_GL=egl python scripts/smoke_env.py`

```
mujoco   3.15.0
numpy    2.5.3
torch    2.14.1+cpu (cuda.is_available=False, cpu.is_available=True)
MUJOCO_GL=egl
physics  free sphere z 0.500 -> 0.302 m after 100 steps (t=0.200 s)
render   480x640 RGB -> /home/ubuntu/grappling/reports/2026-10-07/smoke_env.png (29.9 KiB, nonblack 58.6%)
SMOKE ENV OK
```

Exit code 0. (Three `libEGL warning: failed to open /dev/dri/...` lines go to
stderr only; stdout is clean.)

Physics cross-check: z(0.2 s) = 0.5 − ½·9.81·0.2² ≈ 0.304 m vs simulated
0.302 m (integration + damping) — gravity integration correct.

PNG: `file` reports `PNG image data, 640 x 480, 8-bit/color RGB,
non-interlaced`, 30,651 bytes. Visual check (vision model): light-gray ground
plane + orange sphere mid-air, clean render, no corruption/black frame.

## Self-checks inside smoke_env.py (all assert green)

- `torch.cuda.is_available() is False` and `torch.cpu.is_available()`
- sphere fell (`z_end < z_start`); sim clock = 100 × 0.002 s
- render is 480x640x3 uint8, >50% non-black pixels
- PNG on disk > 5 KiB

## Failures / deviations

None blocking. Recorded oddities:
- EGL `__del__` teardown noise (cosmetic) — avoided via explicit `close()`.
- `/dev/dri` permission warnings (cosmetic) — Mesa software fallback, no GPU.
- PyPI fallback for torch NOT exercised (CPU index worked first try).

## Notes for downstream agents (G1Model, GrappleMapParser)

- Use `MUJOCO_GL=egl` and explicit `renderer.close()`.
- MuJoCo Python 3.15.0, numpy 2.x API (2.5.3) — no legacy np.float etc.
- torch 2.14.1+cpu imports in ~2 s; keep threads modest (4 cores total).
