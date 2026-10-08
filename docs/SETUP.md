# SETUP.md — ARM aarch64 CPU-only Sim Stack (Ubuntu 24.04)

Verified working setup for this project on the target host:
4-core ARM Neoverse-N1 (aarch64), 23 GB RAM, **no GPU**, Ubuntu 24.04,
Python 3.12, passwordless sudo. Everything lives in the project venv
(Ubuntu 24.04 PEP 668 blocks system-wide `pip install`).

## 1. System packages

```bash
sudo apt-get update
sudo apt-get install -y python3-venv git ffmpeg cmake \
    libegl1 libosmesa6 libgl1
```

- `libegl1` is what MuJoCo's default headless backend (`MUJOCO_GL=egl`)
  uses. On this GPU-less host Mesa falls back to **software EGL** — no
  graphics hardware needed.
- `libosmesa6` is only the fallback backend (see Troubleshooting). If you
  skip it, EGL still works; on the reference host only `libegl1` was
  actually required.

## 2. Python venv + packages

```bash
cd /home/ubuntu/grappling        # repo root
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

pip install mujoco numpy scipy pytest imageio imageio-ffmpeg matplotlib
```

## 3. CPU-only PyTorch (aarch64)

Use PyTorch's CPU wheel index — this succeeds on aarch64 and gives a
CPU-only build (`+cpu` suffix, no CUDA):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__, torch.cpu.is_available())"
# expected: 2.14.1+cpu True   (and torch.cuda.is_available() == False)
```

If that index is unreachable, plain `pip install torch` also installs a
CPU wheel on aarch64 (not verified here; the CPU index worked first try).

## 4. Headless rendering backend

```bash
export MUJOCO_GL=egl
```

Put this in your shell rc or export it per-command. On the reference host
`MUJOCO_GL=egl` renders correctly out of the box (software EGL via Mesa).
Expected cosmetic stderr noise on a GPU-less host — safe to ignore:

```
libEGL warning: failed to open /dev/dri/renderD128: Permission denied
```

Scripts must call `renderer.close()` explicitly (as `scripts/smoke_env.py`
does); relying on `__del__` at interpreter exit prints EGL-free exceptions
on this stack (also cosmetic, but noisy).

## 5. Verify

```bash
cd /home/ubuntu/grappling
source .venv/bin/activate
MUJOCO_GL=egl python scripts/smoke_env.py
```

Expected stdout (reference host, 2026-10-07):

```
mujoco   3.15.0
numpy    2.5.3
torch    2.14.1+cpu (cuda.is_available=False, cpu.is_available=True)
MUJOCO_GL=egl
physics  free sphere z 0.500 -> 0.302 m after 100 steps (t=0.200 s)
render   480x640 RGB -> .../reports/2026-10-07/smoke_env.png (29.9 KiB, nonblack 58.6%)
SMOKE ENV OK
```

The script self-checks: torch is CPU-only, the free sphere fell under
gravity, the render is non-black, and the PNG exceeds 5 KiB.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `EGL is not available` / egl import errors | `sudo apt-get install -y libegl1`, retry `MUJOCO_GL=egl` |
| EGL still fails | `sudo apt-get install -y libosmesa6-dev libgl1` then `export MUJOCO_GL=osmesa` (software, slower; not needed on the reference host) |
| `externally-managed-environment` from pip | you forgot to activate the venv (`source .venv/bin/activate`) |
| EGL-free `Exception ignored in __del__` noise at exit | call `renderer.close()` explicitly before exit |
