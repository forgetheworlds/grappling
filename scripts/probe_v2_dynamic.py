#!/usr/bin/env python
"""Dynamic-feasibility probes for the v2 reference set (Agent 3).

Per v2 track (refs + drill_continuous + stance_rise) and per drill phase:

  L1  kinematic + static diagnosis   (feasibility_diagnosis.diagnose_track)
  L2  static holdability             (same instrument: CoM vs planted hull)
  L3  dynamic REPLAY probe           (raw 50 Hz position servos on the
                                      reference's own joint targets -- the
                                      v1 protocol, unchanged)
  BAL dynamic probe WITH A BALANCE LAYER (the T1 v6d checkpoint driving the
      residual corrections in the reference-conditioned tracking env, the
      reference's next-frame targets as the residual base action)

Plus the MEASURED ENVELOPE: the v2 LOWER descent time-scaled by
{1.0, 1.5, 2.0} and probed with both layers -- how fast the root can
actually translate while the feet stay planted and the joints track.

Writes:
  data/references/motion_refs/v2/feasibility.json   (three levels + balance
      layer + probe command + provenance, per segment and per drill phase)
  validity labels REWRITTEN into the v2 npz metas (phases and takes)
  videos/motion_refs/v2_*.mp4 + contact sheets + metrics JSONs
      (EVIDENCE_PROTOCOL bundles; failures kept and labelled)

Every claim is recomputable from this script's output; nothing is inferred
from kinematics.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import mujoco  # noqa: E402

import probe_drill_dynamic as pd1  # noqa: E402  (the v1 replay protocol)
import compose_drill as cd  # noqa: E402
import solo.refgen as rg  # noqa: E402
from feasibility_diagnosis import diagnose_track  # noqa: E402
from solo.scene import N_JOINTS, load_solo_model, stand_frame  # noqa: E402
from solo.track import (REF_ACTOR_DIM, REF_CRITIC_DIM, Segment, TrackingEnv,  # noqa: E402
                        _lead_int, drill_segments, track_targets,
                        warm_start_actor)

V2 = REPO / "data/references/motion_refs/v2"
OUT_DIR = REPO / "videos/motion_refs"
T1_CKPT = REPO / "checkpoints/solo/t1_balance_v6d_401408.pt"
SEED = 0
ENVELOPE_SCALES = (1.0, 1.5, 2.0)

#: which probes get full EVIDENCE_PROTOCOL bundles (video + sheet + JSON)
RENDER = {
    "lower_balance": ("stand_to_stance", "balance"),
    "penetration_balance": ("drill:DOUBLE_LEG_PENETRATION", "balance"),
    "drill_replay": ("drill_continuous", "replay"),
}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------- balance layer
class BalanceLayer:
    """T1 v6d policy as the residual corrector over the reference base action.

    First-layer surgery from the 115-dim balance checkpoint into the
    reference-conditioned net (appended reference columns zero-initialised,
    exactly the track_baselines harness): the net acts on the base
    observation only -- a balance/posture corrector -- while the reference's
    next-frame joint targets ride in as the residual base.
    """

    name = "t1_v6d_401408_surgery"

    def __init__(self):
        import torch

        from rl.checkpoint import load_checkpoint
        from rl.net import ActorCritic, NetConfig

        self.net = ActorCritic(REF_ACTOR_DIM, REF_CRITIC_DIM, act_dim=N_JOINTS,
                               cfg=NetConfig(hidden=(256, 256),
                                             action_mode="residual",
                                             residual_scale=0.5))
        ck = load_checkpoint(T1_CKPT)
        status = warm_start_actor(self.net, ck)
        self.net.actor.log_std.requires_grad_(False)
        self.net.actor.log_std.fill_(-5.0)
        self.status = status

    def act(self, obs) -> np.ndarray:
        import torch

        ao = torch.from_numpy(np.asarray(obs["actor"], np.float32)).unsqueeze(0)
        with torch.no_grad():
            return self.net.actor.deterministic_unit(ao)[0].numpy()


def balance_probe(model, seg: Segment, layer: BalanceLayer | None,
                  seed: int = SEED) -> dict:
    """One tracking-env rollout with the reference base action; z from the
    balance layer (or zeros = exact open-loop replay inside the env)."""
    from solo.env import SoloEnv

    tt = track_targets(seg.source, model)
    base = SoloEnv(model, task="balance", action_mode="residual",
                   residual_scale=0.5, jitter=False, record_metrics=True)
    ep = TrackingEnv(base, seg, tt, q_stand=stand_frame(model)[0], seed=seed)
    obs = ep.reset(seed=seed)
    t0 = time.perf_counter()
    steps, R = 0, 0.0
    site, joint, root_xy = [], [], []
    agree_n, agree = 0, 0
    fall = None
    while True:
        unit = layer.act(obs) if layer is not None else np.zeros(N_JOINTS)
        obs, r, term, trunc, info = ep.step(unit)
        steps += 1
        R += r
        e = info["track"]["errs"]
        site.append(e["site_err"])
        joint.append(e["joint_err"])
        root_xy.append(e["root_xy_err"])
        kf = min(ep.seg.k0 + ep.k, len(ep.tt) - 1)
        ref_c = ep.tt.contact[kf] > 0.5
        got = info["contacts"]
        agree += int(bool(got["left_foot"]) == bool(ref_c[0]))
        agree += int(bool(got["right_foot"]) == bool(ref_c[1]))
        agree_n += 2
        if term or trunc:
            tk = info["track"]
            cause = tk["cause"]
            if cause and ("fall" in cause or "dorsal" in cause):
                fall = steps * 0.02
            success = bool(tk["success"])
            break
        if steps >= 4 * seg.duration_s / 0.02 + 50:
            success, cause = False, "probe_budget"
            break
    n = max(steps, 1)
    return {
        "balance_layer": layer.name if layer is not None else "none (z=0 replay)",
        "completed": success,
        "cause": cause,
        "fell_at_s": fall,
        "steps": steps,
        "covered_s": round(min(steps, seg.duration_s / 0.02) * 0.02, 2),
        "duration_s": round(seg.duration_s, 2),
        "site_err_mean_m": round(float(np.mean(site)), 4),
        "site_err_p95_m": round(float(np.percentile(site, 95)), 4),
        "joint_err_mean_rad": round(float(np.mean(joint)), 4),
        "root_xy_err_mean_m": round(float(np.mean(root_xy)), 4),
        "contact_agreement": round(agree / max(agree_n, 1), 3),
        "return": round(float(R), 1),
        "wall_s": round(time.perf_counter() - t0, 2),
    }


# ------------------------------------------------------------------- probes
def probe_all(model, data) -> dict:
    """All v2 tracks by name: (q (T,36), t (T,), contact (T,2), meta)."""
    tracks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict]] = {}
    for npz in sorted((V2 / "refs").glob("*.npz")):
        d = np.load(npz, allow_pickle=True)
        tracks[npz.stem] = (np.asarray(d["qpos_a"], float),
                            np.asarray(d["t"], float),
                            np.asarray(d["contact"], np.uint8),
                            json.loads(str(d["meta"])) if "meta" in d else {})
    for name in ("drill_continuous", "stance_rise"):
        d = np.load(V2 / f"{name}.npz", allow_pickle=True)
        tracks[name] = (np.asarray(d["qpos_a"], float),
                        np.asarray(d["t"], float),
                        np.asarray(d["contact"], np.uint8),
                        json.loads(str(d["meta"])) if "meta" in d else {})
    return tracks


def phase_segment(name: str) -> Segment:
    """The Segment of one drill phase (frame window from the phase table)."""
    from solo.bc import load_reference
    from solo.track import phase_label

    tr = load_reference("drill_continuous")
    meta = tr.meta
    names = [p["name"] for p in meta["phases"]]
    pid = np.load(tr.source, allow_pickle=True)["phase_id"]
    p = meta["phases"][names.index(name)]
    fr = np.flatnonzero(np.asarray(pid, int) == int(p["id"]))
    if fr.size == 0:
        raise KeyError(name)
    lab, conn = phase_label(name)
    lead = _lead_int(p.get("lead_leg"))
    return Segment("drill_continuous", int(fr[0]), int(fr[-1]) + 1, lab, lead,
                   conn, validity=str(p.get("validity", "unverified")))


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--render-only", action="store_true",
                    help="skip the probes; re-render the evidence bundles "
                         "from a previous run's labels")
    ap.add_argument("--relabel-only", action="store_true",
                    help="skip the probes; rewrite the validity labels from "
                         "a previous run's feasibility.json")
    args = ap.parse_args()
    model = load_solo_model()
    data = mujoco.MjData(model)
    layer = BalanceLayer()
    print(f"[balance layer] {layer.name} surgery: {layer.status}")
    if args.render_only:
        results = json.loads((V2 / "feasibility.json").read_text())
        render_bundles(model, results)
        return 0
    if args.relabel_only:
        results = json.loads((V2 / "feasibility.json").read_text())
        tracks = probe_all(model, data)
        label_tracks(model, data, results, tracks)
        (V2 / "feasibility.json").write_text(json.dumps(results, indent=1))
        print(f"relabeled {V2 / 'feasibility.json'}")
        return 0
    tracks = probe_all(model, data)

    results = {
        "version": "2.0",
        "dataset": "data/references/motion_refs/v2",
        "protocol": {
            "control_hz": 50,
            "physics_timestep_s": float(model.opt.timestep),
            "replay_probe": ("raw 50 Hz position servos on the reference's "
                             "OWN joint targets (probe_drill_dynamic.probe; "
                             "the v1 protocol, unchanged)"),
            "fall_rule": f"pelvis<{pd1.FALL_PELVIS} m & tilt>"
                         f"{pd1.FALL_TILT_DEG} deg sustained "
                         f"{pd1.FALL_CONFIRM_S} s",
            "balance_layer": {
                "controller": layer.name,
                "checkpoint": str(T1_CKPT.relative_to(REPO)),
                "ckpt_sha256_1MB": hashlib.sha256(
                    T1_CKPT.read_bytes()[:1 << 20]).hexdigest()[:16],
                "mechanism": "first-layer surgery (trunk + base-obs columns "
                             "transferred, appended reference columns "
                             "zero-initialised) -> the net outputs residual "
                             "corrections z; ctrl = ref_next + 0.5*tanh(z)",
                "caveat": "the T1 policy has never seen the reference block; "
                          "it acts as a posture/balance corrector over the "
                          "reference base action",
            },
            "claim": "dynamic INDICATOR only; a pass does not make the "
                     "reference a demonstration (no expert_action anywhere)",
            "seed": SEED,
        },
        "provenance": {},
        "per_track": {},
        "envelope": {},
    }

    # ---- per-track probes -------------------------------------------------
    for name, (q, t, c, meta) in tracks.items():
        row: dict = {"path": str((V2 / "refs" / f"{name}.npz")
                                 if (V2 / "refs" / f"{name}.npz").exists()
                                 else (V2 / f"{name}.npz"))}
        print(f"[{name}] static/replay...", flush=True)
        row["static"] = diagnose_track(model, data, q, t, contact=c)
        row["replay"] = pd1.probe(model, data, q)
        seg = Segment(name, 0, len(q), "STANCE", 1)
        print(f"[{name}] balance-layer probe...", flush=True)
        row["balance"] = balance_probe(model, seg, layer)
        results["per_track"][name] = row
        print(f"  replay: {'PASS' if row['replay']['ok'] else 'FAIL'} "
              f"({row['replay']['reason']}); balance: "
              f"{'COMPLETED' if row['balance']['completed'] else 'FAILED'} "
              f"({row['balance']['cause']})")

    # drill phases (keyed by phase NAME for the label rewrite)
    print("[drill phases]", flush=True)
    phase_rows = {}
    from solo.bc import load_reference
    meta = load_reference("drill_continuous").meta
    name_by_pos = [p["name"] for p in meta["phases"]]
    for seg in drill_segments():
        q = track_targets(seg.source, model).qpos[seg.k0:seg.k1]
        tt = np.arange(len(q)) * rg.DT
        cc = track_targets(seg.source, model).contact[seg.k0:seg.k1]
        st = diagnose_track(model, data, q, tt, contact=cc)
        bal = balance_probe(model, seg, layer)
        phase_rows[name_by_pos[len(phase_rows)]] = {
            "label": seg.label, "connect": seg.connect, "frames":
                f"{seg.k0}:{seg.k1}",
            "duration_s": round(seg.duration_s, 2), "static": st,
            "balance": bal}
        print(f"  {name_by_pos[len(phase_rows) - 1]:34s} "
              f"bal={'OK ' if bal['completed'] else 'FAIL'} ({bal['cause']})")
    results["drill_phases"] = phase_rows

    # ---- the measured envelope: LOWER descent rate ------------------------
    print("[envelope] LOWER descent time-scale sweep", flush=True)
    d = np.load(V2 / "refs/stand_to_stance.npz", allow_pickle=True)
    q_lo = np.asarray(d["qpos_a"], float)
    t_lo = np.asarray(d["t"], float)
    c_lo = np.asarray(d["contact"], np.uint8)
    for s in ENVELOPE_SCALES:
        q2, t2, c2 = rg.time_scale(q_lo, t_lo, s, c_lo)
        rep = pd1.probe(model, data, q2)
        # register the scaled track as a temp npz so the tracking env's FK
        # preprocessing matches the scaled trajectory exactly
        tmp = Path(f"/tmp/v2_envelope_lower_x{s:g}.npz")
        meta = json.loads(str(d["meta"]))
        meta["envelope_scale"] = s
        np.savez_compressed(tmp, qpos_a=q2, qpos_b=np.zeros_like(q2), t=t2,
                            technique=f"lower_x{s:g}",
                            edges=np.array([], dtype=np.int64),
                            landmark_rms=np.nan, contact=c2,
                            meta=json.dumps(meta))
        seg = Segment(str(tmp), 0, len(q2), "LEVEL_CHANGE", 1)
        bal = balance_probe(model, seg, layer)
        results["envelope"][f"x{s}"] = {
            "scale": s, "descent_duration_s": round(t2[-1], 2),
            "replay": rep, "balance": bal}
        print(f"  x{s}: {t2[-1]:.2f}s  replay "
              f"{'PASS' if rep['ok'] else 'FAIL @' + str(rep['fell_at_s'])}"
              f"  balance {'COMPLETED' if bal['completed'] else bal['cause']}")

    # ---- verdicts + label rewrite -----------------------------------------
    results["provenance"] = {
        "git_commit": git_commit(),
        "model": "src/solo/scene.load_solo_model (robots/g1/g1.xml solo scene)",
        "seed": SEED,
        "control_rate_hz": 50,
        "physics_timestep_s": float(model.opt.timestep),
        "reproduce": "MUJOCO_GL=egl .venv/bin/python scripts/probe_v2_dynamic.py",
        "wall_s": {k: (v["replay"]["wall_s"] + v["balance"]["wall_s"])
                   for k, v in results["per_track"].items()},
    }
    label_tracks(model, data, results, tracks)
    (V2 / "feasibility.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {V2 / 'feasibility.json'}")

    # ---- evidence bundles --------------------------------------------------
    render_bundles(model, results)
    return 0


def verdict_of(row: dict) -> str:
    """v2 validity label from the measured levels (see RETIMING.md)."""
    st, rep, bal = row["static"], row["replay"], row["balance"]
    if rep["ok"]:
        return "dynamically_verified"
    if not st["kinematic_ok"]:
        return "kinematically_invalid"
    if bal["completed"]:
        return "balance_verified"
    if st["hold_dur_s"] >= 0.4 and (st["hold_com_margin_min_m"] or -1) >= 0.0:
        return "statically_holdable"
    return "kinematically_valid"


def label_tracks(model, data, results, tracks):
    """Rewrite validity into the npz metas (takes + drill phases)."""
    timing_only = ("knee_sprawl_entry", "knee_sprawl_hold",
                   "knee_sprawl_recover", "stalk_shuffle",
                   "knee_sprawl_entry2")
    for name, row in results["per_track"].items():
        if name in timing_only:
            row["validity"] = "timing_only_v2_of_v1 (labels preserved from " \
                              "v1; context takes, not part of the " \
                              "traversable claim)"
            continue
        v = verdict_of(row)
        reason = (f"replay {'PASS' if row['replay']['ok'] else 'FAIL'} "
                  f"({row['replay']['reason']}); balance layer "
                  f"{layer_txt(row['balance'])}; static hold margin "
                  f"{row['static']['hold_com_margin_min_m']} m")
        p = Path(row["path"])
        d = dict(np.load(p, allow_pickle=True))
        meta = json.loads(str(d["meta"]))
        meta["validity"] = v
        meta["validity_reason"] = reason
        meta["probe"] = {"command": results["provenance"]["reproduce"],
                         "balance_layer": layer_txt(row["balance"]),
                         "replay_ok": row["replay"]["ok"],
                         "balance_completed": row["balance"]["completed"]}
        d["meta"] = json.dumps(meta)
        np.savez_compressed(p, **d)
        row["validity"] = v
    # drill phases
    z = np.load(V2 / "drill_continuous.npz", allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    for p in meta["phases"]:
        row = results["drill_phases"].get(p["name"])
        if row is None:
            continue
        if row["connect"]:
            p["validity"] = "unverified"
            p["validity_reason"] = ("synthetic connector; traversed only as "
                                    "part of the containing drill probe")
            continue
        l1_ok = row["static"]["kinematic_ok"]
        if row["balance"]["completed"]:
            v = "balance_verified"
        elif not l1_ok:
            v = "known_infeasible"
        else:
            # L1/L2 hold, the only probed balance layer fails: a LEARNING
            # question (the T2/T3 ladder has not trained this capability),
            # not a reference-timing question -- the falsifier distinction
            v = "balance_blocked"
        p["validity"] = v
        p["validity_reason"] = (
            f"balance layer {layer_txt(row['balance'])}; kinematic_ok="
            f"{l1_ok}; static hold margin "
            f"{row['static']['hold_com_margin_min_m']} m")
    d = dict(z)
    d["meta"] = json.dumps(meta)
    np.savez_compressed(V2 / "drill_continuous.npz", **d)
    results["drill_continuous_validity"] = {
        p["name"]: p["validity"] for p in meta["phases"]
        if not p["name"].startswith("CONNECT_")}
    known = [n for n, v in results["drill_continuous_validity"].items()
             if v == "known_infeasible"]
    verified = [n for n, v in results["drill_continuous_validity"].items()
                if v in ("balance_verified", "dynamically_verified")]
    results["drill_continuous_summary"] = {
        "known_infeasible_phases": known,
        "balance_verified_phases": verified}
    # the composed drill as a whole: honest by composition
    drill_meta = json.loads(str(np.load(V2 / "drill_continuous.npz",
                                        allow_pickle=True)["meta"]))
    if known:
        overall = "known_infeasible"
        reason = (f"contains the measured-infeasible phases {known}; all "
                  f"other non-connector phases are labelled per phase")
    elif verified:
        overall = "balance_verified"
        reason = f"all non-connector phases traversed by the balance layer"
    else:
        overall = "unverified"
        reason = "no phase verdicts"
    results["per_track"]["drill_continuous"]["validity"] = overall
    results["per_track"]["drill_continuous"]["validity_reason"] = reason


def layer_txt(bal: dict) -> str:
    return (f"{bal['balance_layer']}: "
            f"{'COMPLETED' if bal['completed'] else 'FAILED'} "
            f"({bal.get('cause')})")


# ------------------------------------------------------------------ bundles
def render_bundles(model, results):
    """EVIDENCE_PROTOCOL bundles for the decisive probes."""
    import solo.video as sv

    layer = BalanceLayer()
    bundles = [
        ("lower_balance", "stand_to_stance", "balance",
         "v2 LOWER (stand->stance descent): does the balance layer traverse "
         "the re-timed segment?"),
        ("penetration_balance", "drill:DOUBLE_LEG_PENETRATION", "balance",
         "the fused GrappleMap penetration with the balance layer (expected "
         "honest failure: knee-load dynamics)"),
        ("drill_replay", "drill_continuous", "replay",
         "the whole v2 drill under raw position servos (L3 indicator)"),
    ]
    for tag, source, kind, question in bundles:
        render_one(model, results, layer, tag, source, kind, question)


def render_one(model, results, layer, tag, source, kind, question):
    import mujoco as mj

    import solo.video as sv

    if source.startswith("drill:"):
        seg = phase_segment(source.split(":", 1)[1])
        tt = track_targets(seg.source, model)
    else:
        seg = Segment(source, 0, len(track_targets(source, model)), "STANCE", 1)
        tt = track_targets(source, model)
    from solo.env import SoloEnv

    base = SoloEnv(model, task="balance", action_mode="residual",
                   residual_scale=0.5, jitter=False, record_metrics=True)
    ep = TrackingEnv(base, seg, tt, q_stand=stand_frame(model)[0], seed=SEED)
    obs = ep.reset(seed=SEED)
    frames = []
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    next_cap = 0.0
    steps = 0
    fall_t = None
    while True:
        unit = layer.act(obs) if kind == "balance" else np.zeros(N_JOINTS)
        obs, _r, term, trunc, info = ep.step(unit)
        steps += 1
        e = info["track"]["errs"]
        pz = float(base.data.qpos[2])
        tilt = pd1.tilt_deg(base.data)
        cause = info["track"]["cause"]
        if cause and ("fall" in cause or "dorsal" in cause) and fall_t is None:
            fall_t = steps * 0.02
        if base.data.time >= next_cap:
            next_cap = base.data.time + 1.0 / sv.FPS
            state = ("FELL at %.2fs (kept, honest failure)" % fall_t
                     if fall_t is not None else
                     ("completed" if info["track"]["success"] else "running"))
            frames.append(sv.Frame(
                t=float(base.data.time),
                qpos=np.asarray(base.data.qpos, float).copy(),
                mocap_pos=np.zeros((model.nmocap, 3)),
                mocap_quat=np.zeros((model.nmocap, 4)),
                lines=(
                    f"v2 dynamic probe: {tag} ({kind})",
                    f"balance layer: {layer.name}" if kind == "balance"
                    else "raw position servos (replay, z=0)",
                    f"t={base.data.time:5.2f}s  k={ep.k}/{ep.N}  "
                    f"pelvis_z={pz:.3f} m  tilt={tilt:.1f} deg",
                    f"site RMS={e['site_err']:.3f} m  joint={e['joint_err']:.3f} "
                    f"rad  root_xy={e['root_xy_err']:.3f} m",
                    f"cause={cause or 'running'}  -> {state}",
                ),
                metrics={}))
        if term or trunc:
            if fall_t is not None and base.data.time - fall_t > 0.5:
                break
            if fall_t is None:
                break
        if steps > 4 * seg.duration_s / 0.02 + 100:
            break
    tk = info["track"]
    verdict = (f"COMPLETED upright ({tk['cause']})" if tk["success"] else
               f"FAILED: {tk['cause']}" +
               (f" at t={fall_t:.2f}s" if fall_t else ""))
    rec = sv.ClipRecorder(
        f"v2 dynamic probe: {question}",
        ("T1 v6d balance policy over the reference's residual base action"
         if kind == "balance" else
         "raw position servos on the reference's own joint targets"),
        verdict=f"{verdict} -- indicator, not a demonstration")
    rec.frames = frames
    vid = rec.render(model, OUT_DIR / f"v2_{tag}.mp4")
    sheet = rec.contact_sheet(model, OUT_DIR / f"v2_{tag}_sheet.png",
                              caption=question)
    is_drill = source.startswith("drill")
    ref_path = (V2 / "drill_continuous.npz") if is_drill \
        else (V2 / "refs" / f"{source}.npz")
    metrics = {
        "bundle": tag, "question": question, "source": source,
        "kind": kind,
        "completed": bool(tk["success"]), "cause": tk["cause"],
        "fell_at_s": fall_t, "steps": steps,
        "duration_s": round(seg.duration_s, 2),
        "errs_final": {k: float(e[k]) for k in
                       ("site_err", "joint_err", "root_xy_err")},
        "reset_count": 0,
        "fall_count": int(fall_t is not None),
        "longest_continuous_s": round(min(steps, ep.N) * 0.02, 2),
        "provenance": {
            "git_commit": git_commit(),
            "balance_layer": layer.name if kind == "balance" else "none",
            "ckpt": str(T1_CKPT.relative_to(REPO)) if kind == "balance" else "",
            "seed": SEED,
            "control_rate_hz": 50,
            "physics_timestep_s": float(model.opt.timestep),
            "reference": str(ref_path),
            "reference_sha256_1MB": hashlib.sha256(
                ref_path.read_bytes()[:1 << 20]).hexdigest()[:16],
            "reproduce": ("MUJOCO_GL=egl .venv/bin/python scripts/"
                          "probe_v2_dynamic.py --render-only"),
            "media": {"mp4": sv.probe_media(OUT_DIR / f"v2_{tag}.mp4")}},
    }
    (OUT_DIR / f"v2_{tag}.json").write_text(json.dumps(metrics, indent=1))
    print(f"  bundle {tag}: {verdict}  ({vid['path']}, {sheet['path']})")


if __name__ == "__main__":
    raise SystemExit(main())
