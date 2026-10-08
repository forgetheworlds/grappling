#!/usr/bin/env python3
"""Calibrate the back-to-mat detector (deliverable 10) against data/refs.

Method
------
1. For every technique in ``data/refs/*.npz`` run a PD-replay rollout in
   ``robots/wrestling_scene.xml`` (``ctrl = q_ref(t)``, the reference's own
   first frame as the start state, 0.5 s hold tail) and record the detector's
   features at 50 Hz for both robots (:class:`wrestling.backdet.FeatureSequence`).

2. Label samples (measured labels; see the report for the full rationale):

   * ``POSITIVE`` -- the SPRAWL defender (robot ``b_``): the terminal
     back-on-mat episode, defined as the dorsal-contact samples after the last
     sample with ``pelvis_z >= 0.20 m``, restricted to parts of the episode
     that have already lasted ``LABEL_GRACE = 0.5 s`` ("clearly established"
     frames; the grace is a labelling parameter, independent of the detector's
     own confirmation period).  An independent cross-check verifies the
     contacting side with the contact normal (``dorsal_normal``).
   * ``NEG_STAND`` -- all samples of all rollouts (both robots) with
     ``pelvis_z >= 0.55 m`` (clearly standing: knees/hands/sprawl postures are
     all far below this).
   * ``NEG_GROUND_START`` -- the ground start of the STAND_UP rollout
     (knees/hands recovery pose, ``t <= 1.0 s``, both robots) -- the mission's
     explicitly non-terminal ground state.
   * everything else ("other") is reported but not used for the operating
     point: PD-replay rollouts of the shooting techniques end in untaught
     collapses whose geometry is only meaningful once the phase-3 teacher
     exists.

3. Sweep ``tilt_threshold_deg x pelvis_z_threshold x confirm_s`` over the grid
   below.  Sensitivity = fraction of established positive samples in which the
   detector has confirmed (same run-length rule as the streaming detector);
   specificity = fraction of negative samples in which it has NOT confirmed.

4. Pick an operating point: gate ``sens >= 0.95`` and ``spec >= 0.95``, then
   prefer the most persistent confirmation within a 0.4 s detection-latency
   budget, then the tilt threshold closest to the midpoint of
   [standing max tilt, positive min tilt], then the pelvis threshold closest
   to 0.35 m.  The chosen config must equal ``wrestling.backdet.DEFAULT_CONFIG``
   (the environment's default) -- asserted at the end.

Outputs: printed table + ``data/backdet_calibration.json``.

Run from repo root:
    .venv/bin/python scripts/calibrate_backdet.py [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco  # noqa: E402

from wrestling.backdet import (  # noqa: E402
    DEFAULT_CONFIG,
    ROBOTS,
    BackDetConfig,
    FeatureSequence,
    back_features,
    confirmed_mask,
    first_confirmed_index,
)
from wrestling.env import (  # noqa: E402
    MODEL_DT,
    STEP_DT,
    load_wrestling_model,
    reference_trace,
)

REPO = Path(__file__).resolve().parents[1]
OUT_JSON = REPO / "data" / "backdet_calibration.json"
TECHNIQUES = ["STANCE", "DOUBLE_LEG", "SINGLE_LEG", "BODY_LOCK",
              "SNAPDOWN", "SPRAWL", "STAND_UP"]
SAMPLE_EVERY = int(round(STEP_DT / MODEL_DT))   # 10 substeps -> 50 Hz samples
HOLD_TAIL = 0.5                                  # s of extra replay
LABEL_GRACE = 0.5                                # s, "established" positive frames
PELVIS_DOWN_MAX = 0.20                           # m, defender "gone low"
STAND_PELVIS_MIN = 0.55                          # m, "clearly standing"
GROUND_START_T = 1.0                             # s, STAND_UP ground-start window
LATENCY_BUDGET = 0.35                            # s, detection-latency gate
POS_TECHNIQUE, POS_ROBOT = "SPRAWL", "b"         # the labelled back-to-mat case
NEG_START_TECHNIQUE = "STAND_UP"

GRID_TILT = [40, 45, 50, 55, 60, 65, 70, 75]
GRID_PELZ = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
GRID_CONFIRM = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]


# --------------------------------------------------------------- data capture
def replay_rollout(model, technique: str) -> dict:
    """PD-replay one reference; returns {robot: FeatureSequence}."""
    tr = reference_trace(technique)
    qpos = {"a": tr["qpos_a"], "b": tr["qpos_b"]}
    t_ref = tr["t"]
    data = mujoco.MjData(model)
    data.qpos[0:36] = qpos["a"][0]
    data.qpos[36:72] = qpos["b"][0]
    mujoco.mj_forward(model, data)

    def ref_at(t, q):
        i = int(np.clip(np.searchsorted(t_ref, t), 1, len(t_ref) - 1))
        w = (t - t_ref[i - 1]) / (t_ref[i] - t_ref[i - 1])
        return q[i - 1, 7:36] * (1 - w) + q[i, 7:36] * w

    n_steps = int((float(t_ref[-1]) + HOLD_TAIL) / MODEL_DT)
    rows: dict[str, list] = {r: [] for r in ROBOTS}
    for step in range(n_steps):
        t = step * MODEL_DT
        tq = min(t, float(t_ref[-1]))
        data.ctrl[0:29] = ref_at(tq, qpos["a"])
        data.ctrl[29:58] = ref_at(tq, qpos["b"])
        mujoco.mj_step(model, data)
        if step % SAMPLE_EVERY == 0:
            for r in ROBOTS:
                rows[r].append(back_features(model, data, r))
    return {r: FeatureSequence.from_rows(technique, r, rows[r]) for r in ROBOTS}


def collect(model) -> dict:
    seqs = {}
    for tech in TECHNIQUES:
        for robot, seq in replay_rollout(model, tech).items():
            seqs[(tech, robot)] = seq
    return seqs


# ------------------------------------------------------------------- labelling
def build_labels(seqs: dict) -> dict:
    """Return {(technique, robot): mask} per label set, with measured evidence."""
    pos_key = (POS_TECHNIQUE, POS_ROBOT)
    seq = seqs[pos_key]
    up = np.flatnonzero(seq.pelz >= PELVIS_DOWN_MAX)
    t_down = float(seq.t[up[-1]]) if len(up) else -np.inf
    pos_raw = (seq.t > t_down) & seq.dorsal
    pos_established = confirmed_mask(pos_raw, seq.t, LABEL_GRACE)

    neg_stand, neg_ground, other = {}, {}, {}
    for key, s in seqs.items():
        stand = s.pelz >= STAND_PELVIS_MIN
        ground_start = np.zeros_like(stand)
        if key[0] == NEG_START_TECHNIQUE:
            ground_start = s.t <= GROUND_START_T
        neg_stand[key] = stand
        neg_ground[key] = ground_start
        other[key] = ~(stand | ground_start)
    other[pos_key] = other[pos_key] & ~pos_established

    n_pos = int(pos_established.sum())
    n_neg = int(sum(m.sum() for m in neg_stand.values()) + sum(m.sum() for m in neg_ground.values()))
    assert n_pos > 0 and n_neg > 0, (n_pos, n_neg)
    neg_dorsal = [s.tilt[m & s.dorsal] for key, s in seqs.items()
                  for m in [neg_stand[key] | neg_ground[key]]]
    neg_dorsal = np.concatenate([a for a in neg_dorsal if a.size]) if any(
        a.size for a in neg_dorsal) else np.array([0.0])
    t_episode = float(seq.t[np.flatnonzero(pos_raw)[0]])
    evidence = {
        "positive_window": [float(seq.t[pos_established][0]), float(seq.t[pos_established][-1])],
        "positive_samples": n_pos,
        "defender_goes_low_at_s": None if not len(up) else round(t_down, 3),
        "episode_onset_s": round(t_episode, 3),
        "dorsal_normal_agreement": float(seq.dnormal[pos_raw].mean()) if pos_raw.any() else None,
        "front_contacts_in_positive_raw": int(seq.front[pos_raw].sum()),
        "negative_samples": n_neg,
        "neg_stand_samples": int(sum(m.sum() for m in neg_stand.values())),
        "neg_ground_start_samples": int(sum(m.sum() for m in neg_ground.values())),
        "pos_tilt_min": float(seq.tilt[pos_established].min()),
        "pos_tilt_median": float(np.median(seq.tilt[pos_established])),
        "pos_pelz_max": float(seq.pelz[pos_established].max()),
        "neg_stand_tilt_max": float(max(s.tilt[m].max() for key, s in seqs.items()
                                        for m in [neg_stand[key]] if m.any())),
        # the binding negative bound for the tilt gate: negatives that *do*
        # touch the mat with the torso (upright-ish, e.g. knees/hands starts)
        "neg_dorsal_tilt_max": float(neg_dorsal.max()),
        "neg_dorsal_samples": int(neg_dorsal.size),
        "pos_episodes_s": _episodes(seq.t, pos_established),
    }
    return {"pos": {pos_key: pos_established}, "neg_stand": neg_stand,
            "neg_ground": neg_ground, "other": other, "evidence": evidence}


def _episodes(t, mask) -> list:
    out = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append([round(float(t[i]), 3), round(float(t[j]), 3)])
            i = j + 1
        else:
            i += 1
    return out


# ---------------------------------------------------------------------- sweep
def eval_config(seqs: dict, labels: dict, cfg: BackDetConfig) -> dict:
    tp = fn = 0
    neg_confirmed = neg_total = 0
    first_triggers = {}
    for key, s in seqs.items():
        conf = confirmed_mask(s.cond(cfg), s.t, cfg.confirm_s)
        idx = np.flatnonzero(conf)
        first_triggers[key] = None if not len(idx) else round(float(s.t[idx[0]]), 3)
        if key in labels["pos"]:
            m = labels["pos"][key]
            tp += int((conf & m).sum())
            fn += int((~conf & m).sum())
        for name in ("neg_stand", "neg_ground"):
            m = labels[name].get(key)
            if m is None or not m.any():
                continue
            neg_confirmed += int((conf & m).sum())
            neg_total += int(m.sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = 1.0 - neg_confirmed / neg_total if neg_total else float("nan")
    t_trigger = first_triggers.get((POS_TECHNIQUE, POS_ROBOT))
    onset = labels["evidence"]["episode_onset_s"]
    latency = None if t_trigger is None else round(t_trigger - onset, 3)
    return {"sens": sens, "spec": spec, "first_triggers": first_triggers,
            "latency": latency}


def sweep(seqs: dict, labels: dict) -> list:
    out = []
    for tilt in GRID_TILT:
        for pelz in GRID_PELZ:
            for confirm in GRID_CONFIRM:
                cfg = BackDetConfig(tilt_threshold_deg=float(tilt),
                                    pelvis_z_threshold=float(pelz),
                                    confirm_s=float(confirm))
                m = eval_config(seqs, labels, cfg)
                out.append({"tilt": tilt, "pelz": pelz, "confirm": confirm, **m})
    return out


def choose(results: list, evidence: dict) -> dict:
    """Gate on >=0.95 sens/spec and a short detection latency; then prefer the
    most persistent confirmation, a balanced tilt margin around the midpoint of
    [largest negative tilt touching the mat, smallest positive tilt], and a
    pelvis threshold near 0.35 m."""
    gated = [r for r in results if r["sens"] >= 0.95 and r["spec"] >= 0.95]
    passing = [r for r in gated if r["latency"] is not None and r["latency"] <= LATENCY_BUDGET]
    if not passing:
        passing = gated
    assert passing, "no operating point reaches sens/spec >= 0.95"

    pos_tilt_min = evidence["pos_tilt_min"]
    neg_tilt_max = evidence["neg_dorsal_tilt_max"]
    tilt_mid = 0.5 * (pos_tilt_min + neg_tilt_max)

    def score(r):
        # max confirmation first, then tilt closest to the balanced midpoint,
        # then pelvis z closest to 0.35 m
        return (-r["confirm"], abs(r["tilt"] - tilt_mid), abs(r["pelz"] - 0.35))

    best = sorted(passing, key=score)[0]
    cfg = BackDetConfig(tilt_threshold_deg=float(best["tilt"]),
                        pelvis_z_threshold=float(best["pelz"]),
                        confirm_s=float(best["confirm"]),
                        dorsal_x_max=DEFAULT_CONFIG.dorsal_x_max)
    margins = {
        "tilt_margin_pos_deg": round(pos_tilt_min - cfg.tilt_threshold_deg, 2),
        "tilt_margin_neg_deg": round(cfg.tilt_threshold_deg - neg_tilt_max, 2),
        "tilt_balanced_midpoint_deg": round(tilt_mid, 2),
        "pelz_margin_pos_m": round(cfg.pelvis_z_threshold - evidence["pos_pelz_max"], 3),
        "latency_budget_s": LATENCY_BUDGET,
        "passing_configs": len(gated),
        "passing_within_latency_budget": len(passing),
    }
    return {"config": cfg, "metrics": {"sens": best["sens"], "spec": best["spec"]},
            "latency": best["latency"], "margins": margins,
            "first_triggers": best["first_triggers"]}


def first_trigger_latency(seqs, labels, cfg) -> dict:
    key = (POS_TECHNIQUE, POS_ROBOT)
    s = seqs[key]
    idx = first_confirmed_index(s.cond(cfg), s.t, cfg.confirm_s)
    t0 = float(s.t[np.flatnonzero(s.dorsal)[0]]) if s.dorsal.any() else None
    return {
        "first_dorsal_contact_s": t0,
        "first_trigger_s": None if idx is None else round(float(s.t[idx]), 3),
        "latency_s": None if idx is None or t0 is None else round(float(s.t[idx]) - t0, 3),
    }


# ----------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=OUT_JSON)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    model = load_wrestling_model()
    seqs = collect(model)
    labels = build_labels(seqs)
    ev = labels["evidence"]

    print("=== dataset ===")
    print(f"rollouts: {len(TECHNIQUES)} techniques x 2 robots, "
          f"features at {1/STEP_DT:.0f} Hz, hold tail {HOLD_TAIL} s")
    print(f"POSITIVE  SPRAWL defender: {ev['positive_samples']} established samples, "
          f"window {ev['positive_window']}, episodes(s) {ev['pos_episodes_s']}")
    print(f"          defender goes low at {ev['defender_goes_low_at_s']} s; "
          f"dorsal-normal agreement {ev['dorsal_normal_agreement']:.2f}; "
          f"front contacts in window {ev['front_contacts_in_positive_raw']}")
    print(f"          tilt min {ev['pos_tilt_min']:.1f} deg, median {ev['pos_tilt_median']:.1f} deg, "
          f"pelvis-z max {ev['pos_pelz_max']:.3f} m; episode onset {ev['episode_onset_s']} s")
    print(f"NEG_STAND {ev['neg_stand_samples']} samples (pelvis >= {STAND_PELVIS_MIN} m), "
          f"max tilt {ev['neg_stand_tilt_max']:.1f} deg")
    print(f"NEG_GROUND_START {ev['neg_ground_start_samples']} samples "
          f"(STAND_UP t <= {GROUND_START_T} s); "
          f"mat-touching negatives max tilt {ev['neg_dorsal_tilt_max']:.1f} deg "
          f"({ev['neg_dorsal_samples']} samples)")

    results = sweep(seqs, labels)
    pick = choose(results, ev)
    cfg: BackDetConfig = pick["config"]

    n_pass = pick["margins"]["passing_configs"]
    n_pass_budget = pick["margins"]["passing_within_latency_budget"]

    def table(rows, axis, fixed, val):
        print(f"--- sensitivity/specificity vs {axis} "
              f"(pelvis z={fixed['pelz']}, confirm={fixed['confirm']}) ---")
        print(f"{axis:>8} {'sens':>7} {'spec':>7}")
        for r in rows:
            mark = "  <-- chosen" if (r[axis] == val and r["pelz"] == fixed["pelz"]
                                      and r["confirm"] == fixed["confirm"]) else ""
            print(f"{r[axis]:>8} {r['sens']:>7.3f} {r['spec']:>7.3f}{mark}")

    chosen_rows = [r for r in results if r["pelz"] == cfg.pelvis_z_threshold
                   and r["confirm"] == cfg.confirm_s]
    table(sorted(chosen_rows, key=lambda r: r["tilt"]), "tilt",
          {"pelz": cfg.pelvis_z_threshold, "confirm": cfg.confirm_s}, cfg.tilt_threshold_deg)
    chosen_rows = [r for r in results if r["tilt"] == cfg.tilt_threshold_deg
                   and r["confirm"] == cfg.confirm_s]
    table(sorted(chosen_rows, key=lambda r: r["pelz"]), "pelz",
          {"pelz": cfg.pelvis_z_threshold, "confirm": cfg.confirm_s}, cfg.pelvis_z_threshold)
    chosen_rows = [r for r in results if r["tilt"] == cfg.tilt_threshold_deg
                   and r["pelz"] == cfg.pelvis_z_threshold]
    table(sorted(chosen_rows, key=lambda r: r["confirm"]), "confirm",
          {"pelz": cfg.pelvis_z_threshold, "confirm": cfg.confirm_s}, cfg.confirm_s)

    lat = first_trigger_latency(seqs, labels, cfg)
    print("=== operating point ===")
    print(f"config: {cfg.as_dict()}")
    print(f"sens={pick['metrics']['sens']:.3f} spec={pick['metrics']['spec']:.3f} "
          f"latency={pick['latency']} s ({n_pass} configs pass >=0.95/>=0.95; "
          f"{n_pass_budget} also within the {LATENCY_BUDGET:.2f} s latency budget)")
    print(f"margins: tilt +{pick['margins']['tilt_margin_pos_deg']:.1f}/-"
          f"{pick['margins']['tilt_margin_neg_deg']:.1f} deg around the chosen threshold "
          f"(balanced midpoint {pick['margins']['tilt_balanced_midpoint_deg']} deg), "
          f"pelvis z +{pick['margins']['pelz_margin_pos_m']:.3f} m above the positive max")
    print(f"latency on the positive episode: first dorsal contact {lat['first_dorsal_contact_s']} s "
          f"-> trigger {lat['first_trigger_s']} s (latency {lat['latency_s']} s)")

    other = {f"{k[0]}/{k[1]}": t for k, t in pick["first_triggers"].items()
             if t is not None and k not in labels["pos"]}
    print("=== detector triggers outside the labelled sets (reported, not scored) ===")
    for name, t in sorted(other.items()):
        s = seqs[tuple(name.split("/"))]
        m = labels["other"][tuple(name.split("/"))]
        print(f"  {name}: first confirmed at {t} s "
              f"({'other/untaught collapse frames' if m.any() else 'labelled negative frames'})")
    if not other:
        print("  none")

    payload = {
        "generated_by": "scripts/calibrate_backdet.py",
        "config": cfg.as_dict(),
        "metrics": {"sens": pick["metrics"]["sens"], "spec": pick["metrics"]["spec"]},
        "margins": pick["margins"],
        "latency": lat,
        "evidence": ev,
        "trigger_times": {f"{k[0]}/{k[1]}": v for k, v in pick["first_triggers"].items()},
        "tables": {
            "all_configs": [
                {"tilt": r["tilt"], "pelz": r["pelz"], "confirm": r["confirm"],
                 "sens": round(r["sens"], 4), "spec": round(r["spec"], 4)}
                for r in results
            ],
        },
        "labels": {
            "positive": f"{POS_TECHNIQUE}/{POS_ROBOT} dorsal episode after "
                        f"pelvis_z >= {PELVIS_DOWN_MAX} m, established >= {LABEL_GRACE} s",
            "negative_stand": f"pelvis_z >= {STAND_PELVIS_MIN} m, all rollouts",
            "negative_ground_start": f"{NEG_START_TECHNIQUE} t <= {GROUND_START_T} s",
        },
        "wall_time_s": round(time.time() - t0, 2),
    }
    if not args.no_write:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.json.relative_to(REPO)}")

    # self-checks
    assert pick["metrics"]["sens"] >= 0.95, pick["metrics"]
    assert pick["metrics"]["spec"] >= 0.95, pick["metrics"]
    assert cfg == DEFAULT_CONFIG, (
        f"calibrated operating point {cfg.as_dict()} != backdet.DEFAULT_CONFIG "
        f"{DEFAULT_CONFIG.as_dict()}")
    print("self-check OK: sens/spec >= 0.95 and config == backdet.DEFAULT_CONFIG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
