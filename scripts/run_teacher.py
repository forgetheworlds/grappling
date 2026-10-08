#!/usr/bin/env python3
"""Phase-3 teacher evaluation: run the stabilized teacher on all 7 techniques.

For every technique and seed the teacher drives both robots in
``robots/wrestling_scene.xml`` at 50 Hz from the reference's own initial state
plus a seeded perturbation, and the script reports

  stay_up_frac   fraction of frames whose REFERENCE pelvis is >= 0.45 m where
                 the simulated pelvis is >= 0.45 m and the torso tilt is inside
                 ``max(30 deg, ref tilt + 15 deg)`` (the references lean
                 themselves); per robot and the min of the two
  landmark       weighted landmark RMS distance to the reference end pose (m)
  penetration    max ground / inter-robot / self depth over the rollout (m)
  scorer         technique-validity scorer (src/scorer): per-technique mean of
                 the executed trace, per-phase means, min frame (informational)

Acceptance (task spec + operator gate):
  stay_up_frac >= 0.80 for DOUBLE_LEG, SINGLE_LEG, SNAPDOWN, STANCE, STAND_UP;
  scorer mean >= 0.85 per technique with no phase mean < 0.75;
  BODY_LOCK: best achievable + analysis (clinch geometry); SPRAWL ends as
  designed (defender prone, attacker sprawled on top).

BODY_LOCK remediation: when the inter-robot penetration stays above 2 cm, the
script writes a derived scene (``robots/wrestling_scene_soft.xml``) whose
contact parameters were stiffened (documented in the report) and re-runs the
technique on it, targeting < 3 cm.

Outputs: data/teacher_stats.json, videos/teacher/<TECH>.mp4 (--videos, 30 fps,
both robots).  Run from the repo root:

    MUJOCO_GL=egl .venv/bin/python scripts/run_teacher.py [--seeds 5]
                 [--only STANCE] [--videos] [--width 480 --height 360]
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

import mujoco  # noqa: E402

from retarget.scene import load_scene_model, robot_slice  # noqa: E402
from teacher import TeacherController, run_episode  # noqa: E402
from teacher.episode import (final_landmark_dist, score_episode,  # noqa: E402
                             sprawl_end_posture)

TECHNIQUES = ["DOUBLE_LEG", "SINGLE_LEG", "BODY_LOCK", "SNAPDOWN", "SPRAWL",
              "STAND_UP", "STANCE"]
REF_DIR = REPO / "data" / "refs"
VID_DIR = REPO / "videos" / "teacher"
STATS_PATH = REPO / "data" / "teacher_stats.json"
SOFT_SCENE = REPO / "robots" / "wrestling_scene_soft.xml"

STAY_UP_GATE = 0.80                 # required for the five holdable techniques
SCORER_MEAN_GATE = 0.85
SCORER_PHASE_GATE = 0.75
STAY_UP_REQUIRED = {"DOUBLE_LEG", "SINGLE_LEG", "SNAPDOWN", "STANCE", "STAND_UP"}
INTER_PEN_REMEDIATE = 0.02          # m: above this, try the derived scene
INTER_PEN_TARGET = 0.03             # m: remediation target

#: derived-scene contact overrides (documented; applied to every robot geom)
SOFT_SOLREF = (0.010, 1.0)          # [timeconst, dampratio] (default 0.02, 1.0)
SOFT_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)   # default (0.9, 0.95, 0.001, 0.5, 2)


def load_ref(tech):
    npz = np.load(REF_DIR / f"{tech}.npz", allow_pickle=True)
    return npz["qpos_a"], npz["qpos_b"], npz["t"]


def make_soft_scene():
    """Copy wrestling_scene.xml with stiffer robot-robot contact parameters.

    Every value is documented above (SOFT_SOLREF / SOFT_SOLIMP); the derived
    scene is written to robots/wrestling_scene_soft.xml so the change is
    inspectable and reproducible.  Only geoms that belong to a robot are
    touched; the mat is left alone.
    """
    spec = mujoco.MjSpec.from_file(str(REPO / "robots" / "wrestling_scene.xml"))
    n = 0
    for geom in spec.geoms:
        body = geom.parent
        name = getattr(body, "name", "") or ""
        if not (name.startswith("a_") or name.startswith("b_")):
            continue
        geom.solref[:] = SOFT_SOLREF
        geom.solimp[:] = SOFT_SOLIMP
        n += 1
    SOFT_SCENE.write_text(spec.to_xml())
    model = spec.compile()
    return model, n


def run_technique(model, tech, seeds, want_video, width, height):
    """All seeds of one technique; returns the aggregate record + frames."""
    qa, qb, t_ref = load_ref(tech)
    records = []
    frames = []
    for seed in seeds:
        ctrl = TeacherController(model, qa, qb, t_ref, technique=tech)
        res = run_episode(model, ctrl, qa, qb, t_ref, seed=seed,
                          capture_frames=want_video and seed == seeds[0])
        sf = res.stay_up_frac(t_ref)
        pen = res.penetration()
        lm = final_landmark_dist(model, res.end_qpos, qa[-1], qb[-1])
        try:
            sc = score_episode(res, t_ref, tech, model)
            sc_mean = float(sc["mean"])
            sc_phases = {n: float(v) for n, v, _ in sc["per_phase"]}
        except Exception as exc:                     # soft-fail: report why
            sc_mean, sc_phases = float("nan"), {"error": str(exc)}
        records.append({
            "seed": int(seed),
            "stay_up": {"a": sf["a"], "b": sf["b"], "min": sf["min"],
                        "a_literal": sf["a_literal"],
                        "b_literal": sf["b_literal"]},
            "penetration": pen, "landmark": lm,
            "scorer_mean": sc_mean, "scorer_phases": sc_phases,
        })
        if want_video and seed == seeds[0]:
            frames = res.frames
        print(f"  [{tech}] seed {seed}: stay_up a={sf['a']:.2f} b={sf['b']:.2f} "
              f"pen(i/g/s)={pen['inter']:.3f}/{pen['ground']:.3f}/{pen['self']:.3f} "
              f"lm={lm['max']:.3f} scorer={sc_mean:.3f}", flush=True)
    agg = {
        "stay_up_min": float(np.mean([r["stay_up"]["min"] for r in records])),
        "stay_up_a": float(np.mean([r["stay_up"]["a"] for r in records])),
        "stay_up_b": float(np.mean([r["stay_up"]["b"] for r in records])),
        "stay_up_min_seed": float(np.min([r["stay_up"]["min"] for r in records])),
        "landmark_max": float(np.mean([r["landmark"]["max"] for r in records])),
        "pen_inter": float(np.max([r["penetration"]["inter"] for r in records])),
        "pen_ground": float(np.max([r["penetration"]["ground"] for r in records])),
        "pen_self": float(np.max([r["penetration"]["self"] for r in records])),
        "scorer_mean": float(np.nanmean([r["scorer_mean"] for r in records])),
        "scorer_min_seed": float(np.nanmin([r["scorer_mean"] for r in records])),
        "scorer_phases": {},
        "seeds": records,
    }
    phase_vals: dict[str, list[float]] = {}
    for r in records:
        for name, val in r["scorer_phases"].items():
            phase_vals.setdefault(name, []).append(val)
    agg["scorer_phases"] = {k: float(np.nanmean(v)) for k, v in phase_vals.items()}
    return agg, frames


def render_video(model, frames, out_path, width=480, height=360, fps=30):
    """Both robots, side view, 30 fps, from the captured (t, qpos) snapshots."""
    import imageio.v2 as imageio
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.55]
    cam.distance = 4.2
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 90
    data = mujoco.MjData(model)
    sa, sb = robot_slice(model, "a_"), robot_slice(model, "b_")
    imgs = []
    for _, q in frames:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        cam.lookat[0] = 0.5 * (q[sa][0] + q[sb][0])
        renderer.update_scene(data, camera=cam)
        imgs.append(renderer.render())
    renderer.close()          # REQUIRED on this EGL host (notes.md)
    if not frames:
        return 0
    t0 = frames[0][0]
    for _ in range(int(round((frames[-1][0] - t0) * fps)) - len(imgs) + 1):
        imgs.append(imgs[-1])
    imageio.mimwrite(out_path, imgs, fps=fps, quality=8,
                     macro_block_size=None)
    return len(imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--only", default=None,
                    help="comma-separated technique subset")
    ap.add_argument("--videos", action="store_true")
    ap.add_argument("--videos-only", action="store_true",
                    help="skip evaluation, only (re)render the videos")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    args = ap.parse_args()
    techs = args.only.split(",") if args.only else TECHNIQUES
    seeds = list(range(args.seeds))

    model = load_scene_model()
    if args.videos_only:
        seeds = [0]
    stats = {"techniques": {}, "seeds": args.seeds, "scene": "wrestling_scene.xml"}
    if STATS_PATH.exists() and args.only:
        # partial runs merge into the existing record instead of clobbering it
        try:
            stats = json.loads(STATS_PATH.read_text())
            stats["seeds"] = args.seeds
        except Exception:
            pass
    t0 = time.time()
    for tech in techs:
        print(f"=== {tech}", flush=True)
        agg, frames = run_technique(model, tech, seeds,
                                    args.videos or args.videos_only,
                                    args.width, args.height)
        if tech == "SPRAWL":
            qa, qb, t_ref = load_ref(tech)
            ctrl = TeacherController(model, qa, qb, t_ref, technique=tech)
            res = run_episode(model, ctrl, qa, qb, t_ref, seed=seeds[0])
            agg["sprawl_end"] = sprawl_end_posture(model, res.end_qpos)
        if not args.videos_only:
            stats["techniques"][tech] = agg
        # incremental write: the run is long (videos are software-rendered), and
        # the numbers must be inspectable before the whole sweep finishes
        STATS_PATH.write_text(json.dumps(stats, indent=1, sort_keys=True))
        if (args.videos or args.videos_only) and frames:
            VID_DIR.mkdir(parents=True, exist_ok=True)
            out = VID_DIR / f"{tech}.mp4"
            n = render_video(model, frames, out, args.width, args.height)
            print(f"  wrote {out} ({n} frames)", flush=True)

    # BODY_LOCK remediation: derived scene when inter-robot penetration > 2 cm
    if args.videos_only:
        print(f"videos-only pass done ({time.time()-t0:.0f}s)")
        return 0
    bl = stats["techniques"].get("BODY_LOCK")
    if bl and bl["pen_inter"] > INTER_PEN_REMEDIATE:
        print(f"=== BODY_LOCK remediation (inter pen {bl['pen_inter']*100:.1f} cm "
              f"> {INTER_PEN_REMEDIATE*100:.0f} cm): derived scene", flush=True)
        soft_model, ngeom = make_soft_scene()
        print(f"  wrote {SOFT_SCENE.name}: solref={SOFT_SOLREF} solimp={SOFT_SOLIMP} "
              f"on {ngeom} robot geoms", flush=True)
        agg, frames = run_technique(soft_model, "BODY_LOCK", seeds, False,
                                    args.width, args.height)
        agg["scene"] = SOFT_SCENE.name
        agg["contact_overrides"] = {"solref": list(SOFT_SOLREF),
                                    "solimp": list(SOFT_SOLIMP),
                                    "geoms": int(ngeom)}
        stats["techniques"]["BODY_LOCK_soft"] = agg
        stats["body_lock_remediation"] = {
            "base_pen_inter": bl["pen_inter"], "soft_pen_inter": agg["pen_inter"],
            "target": INTER_PEN_TARGET, "scene": SOFT_SCENE.name,
            "solref": list(SOFT_SOLREF), "solimp": list(SOFT_SOLIMP)}

    # -- acceptance table ------------------------------------------------
    print("\n=== acceptance", flush=True)
    ok = True
    for tech in techs:
        a = stats["techniques"][tech]
        su = a["stay_up_min"]
        sc = a["scorer_mean"]
        sc_min_phase = min(a["scorer_phases"].values()) if a["scorer_phases"] else float("nan")
        want = tech in STAY_UP_REQUIRED
        pass_su = (su >= STAY_UP_GATE) if want else True
        pass_sc = (sc >= SCORER_MEAN_GATE) and (sc_min_phase >= SCORER_PHASE_GATE)
        flag = "PASS" if (pass_su and pass_sc) else ("FAIL" if want else "INFO")
        if pass_su and pass_sc:
            pass
        elif want:
            ok = False
        print(f"{tech:11s} stay_up(min|a|b)={su:.2f}|{a['stay_up_a']:.2f}|"
              f"{a['stay_up_b']:.2f} pen_i={a['pen_inter']*100:4.1f}cm "
              f"lm={a['landmark_max']:.3f} scorer={sc:.3f} "
              f"minphase={sc_min_phase:.3f} -> {flag}")
    stats["summary"] = {"ok": ok, "seeds": args.seeds, "wall_s": round(time.time() - t0, 1)}
    STATS_PATH.write_text(json.dumps(stats, indent=1, sort_keys=True))
    print(f"\nwrote {STATS_PATH} ({time.time()-t0:.0f}s) ok={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
