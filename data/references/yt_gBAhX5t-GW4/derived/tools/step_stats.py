"""Step statistics for a window: foot-repositioning events, step length,
pelvis travel — measured in the metric image plane (absolute, not hip-centred).

Foot lifts are detected as bursts of horizontal displacement of the ankle
landmark (> LIFT_M within WIN_S), because the monocular ankle HEIGHT signal
oscillates by ~5 cm even when planted (landmark noise) and cannot separate a
real 3-5 cm shuffle lift from noise. Repositioning events therefore mix steps
and drags; the report states this limitation.

Usage: python step_stats.py --npz ... --t0 112 --t1 132
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from analyze_pose import LANK, Pose, RANK  # noqa: E402
from measure_takes import body_frame, image_metric, proj  # noqa: E402

WIN_S = 0.27
LIFT_M = 0.10


def reposition_events(uv: np.ndarray, fps: float) -> list[tuple[int, int, np.ndarray]]:
    """(i0, i1, displacement_uv) bursts where the landmark moves > LIFT_M
    within WIN_S."""
    n = len(uv)
    w = max(2, int(round(WIN_S * fps)))
    out = []
    i = 0
    while i < n - 1:
        j = min(n - 1, i + w)
        d = uv[j] - uv[i]
        if np.linalg.norm(d) > LIFT_M:
            # extend while still moving
            k = j
            while k + 1 < n and np.linalg.norm(uv[k + 1] - uv[j]) < 0.02:
                k += 1
            out.append((i, k, uv[k] - uv[i]))
            i = k + 1
        else:
            i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--t0", type=float, required=True)
    ap.add_argument("--t1", type=float, required=True)
    args = ap.parse_args()

    p = Pose(Path(args.npz))
    s = p.signals()
    sel = (p.t >= args.t0) & (p.t <= args.t1)
    idx = np.where(sel)[0]
    assert len(idx) > 3, f"no frames in [{args.t0},{args.t1}]"
    w = s["w"][idx]
    bf = body_frame(w)
    im, scale = image_metric(p, idx)
    dt = float(np.median(np.diff(p.t)))
    dur = len(idx) * dt
    print(f"window {args.t0}-{args.t1}s frames={len(idx)} dur={dur:.2f}s "
          f"scale~{np.median(scale):.3f} m/unit")

    # body axes in the image plane (u,v), from the hip line and facing
    fwd2 = np.array([np.nanmedian(bf["fwd"][:, 0]), np.nanmedian(bf["fwd"][:, 2])])
    lat2 = np.array([np.nanmedian(bf["lat"][:, 0]), np.nanmedian(bf["lat"][:, 2])])
    fwd2 /= max(np.linalg.norm(fwd2), 1e-9)
    lat2 /= max(np.linalg.norm(lat2), 1e-9)

    total_events = 0
    for name, j in (("left", LANK), ("right", RANK)):
        uv = im[:, j]
        ev = reposition_events(uv, 1.0 / dt)
        total_events += len(ev)
        print(f"  {name}: events={len(ev)} rate={len(ev)/dur:.2f}/s")
        for a, b, d in ev:
            f = float(np.dot(d, fwd2))
            l = float(np.dot(d, lat2))
            print(f"      t={p.t[idx[a]]:7.2f}->{p.t[idx[b]]:7.2f} "
                  f"len={np.linalg.norm(d):.2f} m  fwd={f:+.2f} lat={l:+.2f}")
    print(f"  cadence (both feet): {total_events/dur:.2f} events/s")

    la = proj(bf, w[:, LANK])
    ra = proj(bf, w[:, RANK])
    with np.errstate(invalid="ignore"):
        width = np.nanmedian(np.abs(la[:, 1] - ra[:, 1]))
        depth = np.nanmedian(np.abs(la[:, 0] - ra[:, 0]))
    print(f"  stance width median={width:.3f} m  depth median={depth:.3f} m")

    pel = 0.5 * (im[:, 23] + im[:, 24])
    net = np.linalg.norm(pel[-1] - pel[0])
    path = float(np.nansum(np.linalg.norm(np.diff(pel, axis=0), axis=1)))
    print(f"  pelvis travel: net={net:.2f} m ({net/dur:.2f} m/s) path={path:.2f} m")


if __name__ == "__main__":
    main()
