"""T1 mid-run sanity probe: is the deterministic policy collapsed pre-push, or
does it only fail push recovery? Also compare deterministic vs stochastic action.

Writes data/solo/metrics/t1_midrun_probe_summary.json (territory-safe).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from solo.baselines import PolicyController, ZeroActionController  # noqa: E402
from solo.env import SoloEnv  # noqa: E402
from solo.eval import battery_pushes, evaluate  # noqa: E402
from solo.lock import SimLock  # noqa: E402
from solo.scene import load_solo_model  # noqa: E402
from solo.train import SoloTrainer, TrainConfig  # noqa: E402

CKPT = str(REPO / "checkpoints" / "solo" / "t1_balance.pt")
OUT = REPO / "data" / "solo" / "metrics" / "t1_midrun_probe_summary.json"


def main() -> int:
    cfg = TrainConfig(task="balance", steps=2_000_000, rollout_steps=2048,
                      gamma=0.995, out=CKPT)
    tr = SoloTrainer(cfg)
    tr.load(CKPT)
    net = tr.net
    net.eval()
    model = tr.model
    out = {"ckpt": {"steps_done": tr.steps_done, "iteration": tr.iteration}}

    def quiet_roll(ctrl_factory, seeds, max_s):
        env = SoloEnv(model=model, task="balance", seed=0)
        env.horizon = float(max_s)
        rows = []
        for s in seeds:
            env.reset(seed=s, push=None, command=None)
            ctrl = ctrl_factory(env, s)
            ups, pz, rew = [], [], 0.0
            for _ in range(int(round(max_s / 0.02))):
                o, r, te, tr_, info = env.step(ctrl(env, env.data))
                rows_step = info.get("metrics")
                rew += float(r)
                if rows_step:
                    ups.append(rows_step["upright"]); pz.append(rows_step["pelvis_z"])
                if te or tr_:
                    break
            rows.append({"seed": s, "steps": len(ups),
                         "mean_upright": round(float(np.mean(ups)), 4) if ups else None,
                         "min_upright": round(float(np.min(ups)), 4) if ups else None,
                         "mean_pelvis_z": round(float(np.mean(pz)), 4) if pz else None,
                         "final_pelvis_z": round(float(pz[-1]), 4) if pz else None,
                         "reward_sum": round(rew, 3)})
        return rows

    # 1) deterministic, NO push, 2 s, seeds 0-2
    out["no_push_det"] = quiet_roll(lambda e, s: PolicyController(net), [0, 1, 2], 2.0)
    # 2) zero-action reference, NO push
    out["no_push_zero"] = quiet_roll(lambda e, s: ZeroActionController(), [0, 1, 2], 2.0)

    # 3) reduced battery (mags 4 & 12, 4 dirs) deterministic vs stochastic
    full = battery_pushes()  # 48: 6 magnitudes x 8 dirs; m4 = idx 0-7, m12 = idx 16-23
    red = [full[i] for i in (0, 2, 4, 6, 16, 18, 20, 22)]
    for tag, stoch in (("det", False), ("stoch", True)):
        rep = evaluate(lambda env, seed, st=stoch: PolicyController(net, stochastic=st),
                       task="balance", push_plan=red, seed0=0, name=f"probe_{tag}",
                       out_dir=None, verbose=False)
        a = rep["aggregate"]
        out[f"reduced_battery_{tag}"] = {k: a.get(k) for k in
            ("fall_rate", "mean_upright", "max_recoverable_impulse",
             "recovery_success_rate", "com_offset_max", "steps_after_push_mean")}
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
