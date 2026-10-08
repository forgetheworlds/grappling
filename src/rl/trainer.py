"""Training loop + deterministic evaluation for the Phase-5 PPO backbone.

:class:`Trainer` wires the pieces together: PISTY curriculum, vec env backend,
rollout collector, PPO update, stage transitions, atomic checkpointing and
resume.  ``run()`` returns a summary dict with measured throughput, episode
statistics and the action-bounds check (the smoke acceptance criteria).

``evaluate()`` runs seeded matches with the deterministic (mean) policy and
returns a per-exchange table plus movement/outcome summary counts.
"""

from __future__ import annotations

import copy
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from wrestling.env import STEP_DT, load_wrestling_model

from .checkpoint import apply_checkpoint, load_checkpoint, restore_rng, save_checkpoint, warm_start_from_bc
from .curriculum import Curriculum, CommandScheduler, StageConfig, stage_from_dict
from .net import ActorCritic, action_mapper_for, compose_action
from .obs import ActorObsBuilder, actor_obs_dim
from .ppo import PPOConfig, lr_at, ppo_update
from .privileged import PRIV_DIM, unpack
from .reward import ScorerAdapter, StageReward
from .rollout import OpponentPolicy, RolloutCollector
from .scripted import reference_base_ctrl
from .vec import RESET_SEED_STRIDE, VecWrestlingEnv


def resolve_backend(n_envs: int, backend: str = "auto") -> str:
    """Measured on this host: subproc wins at 3 envs, sequential at <= 2."""
    if backend == "auto":
        return "subproc" if n_envs >= 3 else "sequential"
    if backend not in ("sequential", "subproc"):
        raise ValueError(f"unknown backend {backend!r}")
    return backend


def make_policy(cfg: PPOConfig) -> ActorCritic:
    actor_dim = actor_obs_dim(cfg.frame_stack)
    return ActorCritic(actor_dim, actor_dim + PRIV_DIM, act_dim=29, cfg=cfg.net_config())


class Trainer:
    """Resumable PPO trainer over the PISTY stage ladder."""

    def __init__(self, cfg: PPOConfig, *, out_path, curriculum: Curriculum | None = None,
                 learner_robot: str = "a", backend: str = "auto", n_envs: int | None = None,
                 env_kwargs: dict | None = None, ckpt_every: int = 5000,
                 scorer_enabled: bool = True, scorer_every: int = 5,
                 eval_every: int | None = None, eval_episodes: int = 1,
                 verbose: int = 1):
        self.cfg = cfg
        self.out_path = str(out_path)
        self.learner_robot = learner_robot
        self.n_envs = int(n_envs if n_envs is not None else cfg.n_envs)
        self.backend = resolve_backend(self.n_envs, backend)
        self.env_kwargs = dict(env_kwargs or {})
        self.ckpt_every = int(ckpt_every)
        self.scorer_enabled = bool(scorer_enabled)
        self.scorer_every = int(scorer_every)
        self.eval_every = None if eval_every is None else int(eval_every)
        self.eval_episodes = int(eval_episodes)
        self.eval_summary: dict | None = None
        self._last_eval_step = 0
        self.verbose = int(verbose)
        self.curriculum = curriculum or Curriculum()
        self.model = load_wrestling_model()
        torch.manual_seed(cfg.seed)
        torch.set_num_threads(cfg.torch_threads)
        self.policy = make_policy(cfg)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.mapper = action_mapper_for(self.model, learner_robot, cfg.net_config())
        self.opponent_mapper = action_mapper_for(self.model, "b" if learner_robot == "a" else "a",
                                                 cfg.net_config())
        self.step_count = 0
        self.update_count = 0
        self.target_steps = 0
        self._last_ckpt_step = 0
        self._stop = False
        self._partial = False
        self.bc_status: dict | None = None
        self.stats = self._fresh_stats()
        self.vec: VecWrestlingEnv | None = None
        self.collector: RolloutCollector | None = None
        self.scheduler: CommandScheduler | None = None
        self.stage_reward: StageReward | None = None
        self.scorer: ScorerAdapter | None = None
        self.opponent: OpponentPolicy | None = None
        self._iter_times: list[tuple[float, int]] = []
        self._sim_frac: list[float] = []
        self._configure_stage(first=True)

    # ------------------------------------------------------------------ stats
    @staticmethod
    def _fresh_stats() -> dict:
        return {"matches": 0, "exchanges": 0, "wins": 0, "losses": 0, "draws": 0,
                "causes": Counter(), "oob_events": 0, "bounds_bad": 0,
                "match_returns": [], "exchange_durations": []}

    def _absorb(self, meta: dict, update_stats: dict) -> None:
        s = self.stats
        s["matches"] += int(meta.get("matches_completed", 0))
        s["bounds_bad"] += int(meta.get("bounds_bad", 0))
        s["match_returns"].extend(float(x) for x in meta.get("match_returns", []))
        learner = self.learner_robot
        for rec in meta.get("exchange_records", []):
            s["exchanges"] += 1
            s["causes"][rec.get("cause", "?")] += 1
            s["exchange_durations"].append(float(rec.get("duration", 0.0)))
            s["oob_events"] += int(sum(rec.get("oob_events", {}).values()))
            if rec.get("winner") == learner:
                s["wins"] += 1
            elif rec.get("loser") == learner:
                s["losses"] += 1
            else:
                s["draws"] += 1
            # the advancement gate sees the per-exchange sample: outcome,
            # technique-scorer similarity (None when unmeasured) and whether the
            # learner held its own ground (no back taken, no OOB forfeit); see
            # rl.curriculum.ExchangeSample
            stood = ((rec.get("back_triggers") or {}).get(learner) is None
                     and not (rec.get("cause") == "oob" and rec.get("loser") == learner))
            self.curriculum.on_exchange(
                1 if rec.get("winner") == learner else -1 if rec.get("loser") == learner else 0,
                similarity=rec.get("similarity"), stood=stood)
        if update_stats:
            self._last_update = dict(update_stats)

    # ------------------------------------------------------------------ stage
    def _configure_stage(self, first: bool = False) -> None:
        stage = self.curriculum.stage
        if self.vec is not None:
            self.vec.close()
        self.scheduler = CommandScheduler(stage.techniques, command_on=stage.command_on,
                                          phase_jitter_s=stage.perturbations.phase_jitter_s,
                                          seed=self.cfg.seed)
        need_scorer = stage.use_scorer and stage.weights.technique_similarity > 0.0
        if self.scorer is None:
            self.scorer = ScorerAdapter(self.model, enabled=self.scorer_enabled and need_scorer,
                                        score_every=self.scorer_every)
        elif need_scorer and not self.scorer.available:
            self.scorer = ScorerAdapter(self.model, enabled=self.scorer_enabled,
                                        score_every=self.scorer_every)
        self.stage_reward = StageReward(stage.weights, scorer=self.scorer,
                                        learner_robot=self.learner_robot,
                                        gamma=self.cfg.gamma, dt=STEP_DT)
        mode = "pair" if stage.opponent.kind == "policy" else "learner_only"
        self.vec = VecWrestlingEnv(self.n_envs, backend=self.backend, seed=self.cfg.seed,
                                   learner_robot=self.learner_robot,
                                   opponent_spec=stage.opponent, env_kwargs=self.env_kwargs,
                                   weights=stage.weights, mode=mode)
        self.opponent = None
        if mode == "pair":
            opp = make_policy(self.cfg)
            if stage.opponent.checkpoint:
                ck = load_checkpoint(stage.opponent.checkpoint)
                opp.load_state_dict(ck["policy"])
                note = f"opponent loaded from {stage.opponent.checkpoint}"
            else:
                opp.load_state_dict(copy.deepcopy(self.policy.state_dict()))
                note = "opponent = frozen snapshot of the current policy"
            for p in opp.parameters():
                p.requires_grad_(False)
            opp.eval()
            self.opponent = OpponentPolicy(opp, self.opponent_mapper, robot="b" if self.learner_robot == "a" else "a")
            if self.verbose:
                print(f"[trainer] stage {stage.key}: {note}")
        if self.collector is None:
            self.collector = RolloutCollector(policy=self.policy, mapper=self.mapper,
                                              scheduler=self.scheduler, stage_reward=self.stage_reward,
                                              cfg=self.cfg, learner_robot=self.learner_robot,
                                              opponent=self.opponent, seed=self.cfg.seed)
        else:
            self.collector.scheduler = self.scheduler
            self.collector.stage_reward = self.stage_reward
            self.collector.opponent = self.opponent
        self.collector.attach(self.vec)
        if first and stage.bc_checkpoint:
            self.bc_status = warm_start_from_bc(self.policy, stage.bc_checkpoint)
            if self.verbose:
                print(f"[trainer] BC warm start: {self.bc_status['note']} ({stage.bc_checkpoint})")

    def _on_stage_advance(self) -> None:
        old = self.curriculum.index
        self._configure_stage()
        stage = self.curriculum.stage
        if stage.bc_checkpoint:
            self.bc_status = warm_start_from_bc(self.policy, stage.bc_checkpoint)
        if self.verbose:
            print(f"[trainer] stage advanced {old} -> {self.curriculum.index} "
                  f"({stage.key} {stage.name}); env/opponent rebuilt")

    # -------------------------------------------------------------------- run
    def request_stop(self) -> None:
        """Ask :meth:`run` to stop after the current rollout (signal handler hook)."""
        self._stop = True

    def run(self, total_steps: int | None = None) -> dict:
        if total_steps is not None:
            self.target_steps = int(total_steps)
        if self.target_steps <= 0:
            raise ValueError("run(total_steps) needs a positive target")
        stage = self.curriculum.stage
        t0 = time.perf_counter()
        while self.step_count < self.target_steps and not self._stop:
            noise = stage.perturbations.noise_std(self.curriculum.steps_in_stage)
            remaining = self.target_steps - self.step_count
            per_env = max(1, int(np.ceil(remaining / self.n_envs)))
            steps_this = min(int(self.cfg.rollout_steps), per_env)
            rollout = self.collector.collect(n_steps=steps_this, noise_std=noise,
                                             stop_check=lambda: self._stop)
            if rollout.meta.get("partial") or rollout.advantages is None:
                self._partial = True
                break
            progress = min(1.0, self.step_count / max(1, self.target_steps))
            lr = lr_at(progress, self.cfg)
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(self.cfg.seed) * 1_000_003 + self.update_count)
            up = ppo_update(self.policy, self.optimizer, rollout, self.cfg, lr, gen)
            self.step_count += int(rollout.meta["steps"])
            self.update_count += 1
            self.curriculum.tick(int(rollout.meta["steps"]))
            self._absorb(rollout.meta, up)
            self._iter_times.append((time.perf_counter() - t0, int(rollout.meta["steps"])))
            self._log_iteration(up, rollout)
            if self.curriculum.maybe_advance():
                self._on_stage_advance()
                stage = self.curriculum.stage
            self._maybe_checkpoint()
            self._maybe_eval()
        wall = time.perf_counter() - t0
        summary = self.summary(wall_s=wall)
        self.save()
        if self.verbose:
            print(self._summary_text(summary))
        return summary

    # ------------------------------------------------------------ checkpointing
    def state_dict(self) -> dict:
        return {
            "step_count": self.step_count,
            "update_count": self.update_count,
            "target_steps": self.target_steps,
            "curriculum": self.curriculum.state_dict(),
            "stats": {**{k: v for k, v in self.stats.items() if k != "causes"},
                      "causes": dict(self.stats["causes"])},
            "stage": self.curriculum.stage.as_dict(),
            "bc_status": self.bc_status,
            "eval": self.eval_summary,
            "collector_rng": (None if self.collector is None
                              else self.collector.rng.bit_generator.state),
        }

    def extra_dict(self) -> dict:
        return {"env": self.env_kwargs, "n_envs": self.n_envs, "backend": self.backend,
                "learner_robot": self.learner_robot, "seed": self.cfg.seed,
                "actor_obs_dim": actor_obs_dim(self.cfg.frame_stack),
                "critic_obs_dim": actor_obs_dim(self.cfg.frame_stack) + PRIV_DIM}

    def save(self, path: str | None = None) -> str:
        p = save_checkpoint(path or self.out_path, policy=self.policy, optimizer=self.optimizer,
                            cfg=self.cfg, state=self.state_dict(), extra=self.extra_dict())
        self._last_ckpt_step = self.step_count
        return p

    def _maybe_checkpoint(self) -> None:
        if self.step_count - self._last_ckpt_step >= self.ckpt_every:
            p = self.save()
            if self.verbose:
                print(f"[trainer] checkpoint @ {self.step_count} steps -> {p}")

    def _maybe_eval(self) -> None:
        """Periodic deterministic evaluation (only when ``eval_every`` is set).

        Uses mean actions and a separate single-env env, so it consumes no
        training RNG; the summary is stored in the checkpoint (``state['eval']``).
        """
        if not self.eval_every or self.step_count - self._last_eval_step < self.eval_every:
            return
        summary = evaluate(self.policy, stage=self.curriculum.stage, env_kwargs=self.env_kwargs,
                           episodes=self.eval_episodes, seed=self.cfg.seed + 10_000,
                           learner_robot=self.learner_robot, frame_stack=self.cfg.frame_stack,
                           check_determinism=False)
        self.eval_summary = {k: v for k, v in summary.items() if k != "table"}
        self._last_eval_step = self.step_count
        if self.verbose:
            print(f"[trainer] eval @ {self.step_count} steps: "
                  f"exchanges {summary['exchanges']} causes {summary['causes']} "
                  f"W/L/D {summary['learner_wins']}/{summary['learner_losses']}/{summary['draws']} "
                  f"standing {summary['movement']['standing_fraction']:.2f}")

    def load(self, path: str) -> dict:
        """Resume: restore weights, optimizer, curriculum, stats, stage, RNG."""
        ckpt = load_checkpoint(path)
        cfg = PPOConfig.from_dict(ckpt.get("config") or {})
        if cfg.frame_stack != self.cfg.frame_stack or tuple(cfg.hidden) != tuple(self.cfg.hidden):
            raise ValueError("checkpoint architecture differs from the trainer config "
                             f"(frame_stack {cfg.frame_stack}, hidden {cfg.hidden})")
        apply_checkpoint(ckpt, policy=self.policy, optimizer=self.optimizer)
        state = ckpt.get("state") or {}
        self.step_count = int(state.get("step_count", 0))
        self.update_count = int(state.get("update_count", 0))
        self.target_steps = int(state.get("target_steps", 0))
        self.curriculum.load_state_dict(state.get("curriculum") or {})
        st = state.get("stats") or {}
        self.stats = self._fresh_stats()
        for k, v in st.items():
            if k == "causes":
                self.stats["causes"] = Counter(v)
            elif k in self.stats:
                self.stats[k] = v
        self.bc_status = state.get("bc_status")
        self.eval_summary = state.get("eval")
        saved_stage = state.get("stage")
        if saved_stage:  # keep opponent/weight overrides that were active at save time
            stages = list(self.curriculum.stages)
            stages[self.curriculum.index] = stage_from_dict(saved_stage)
            self.curriculum.stages = tuple(stages)
        self._configure_stage()   # rebuild env/opponent at the saved stage
        crng = state.get("collector_rng")
        if crng is not None and self.collector is not None:
            self.collector.rng.bit_generator.state = crng
        restore_rng(ckpt.get("rng") or {})
        if self.verbose:
            print(f"[trainer] resumed {path} at step {self.step_count} "
                  f"(stage {self.curriculum.stage.key}, {self.update_count} updates)")
        return ckpt

    # ----------------------------------------------------------------- logging
    def steps_per_s(self) -> float:
        if len(self._iter_times) < 2:
            return float("nan")
        (t_prev, _), (t_now, steps) = self._iter_times[-2], self._iter_times[-1]
        return steps / max(1e-9, t_now - t_prev)

    def _log_iteration(self, up: dict, rollout) -> None:
        if not self.verbose:
            return
        s = self.stats
        ret = np.mean(s["match_returns"][-20:]) if s["match_returns"] else float("nan")
        sim = self.stage_reward.last_similarity
        msg = (f"[iter {self.update_count:4d}] steps {self.step_count:7d} "
               f"| {self.steps_per_s():6.1f} steps/s "
               f"| stage {self.curriculum.stage.key} "
               f"| ret {ret:7.3f} | matches {s['matches']:4d} "
               f"| exch {s['exchanges']:5d} (W/L/D {s['wins']}/{s['losses']}/{s['draws']}) "
               f"| sim {'--' if sim is None else f'{sim:4.2f}'} "
               f"| pi {up['policy_loss']:+.3f} v {up['value_loss']:.3f} "
               f"ent {up['entropy']:.2f} kl {up['approx_kl']:.4f} lr {up['lr']:.2e}")
        print(msg)

    def summary(self, wall_s: float | None = None) -> dict:
        s = self.stats
        steps = self.step_count
        return {
            "steps": steps,
            "iterations": self.update_count,
            "wall_s": wall_s,
            "steps_per_s": (steps / wall_s) if (wall_s and wall_s > 1e-6) else None,
            "stage": self.curriculum.stage.key,
            "matches": s["matches"],
            "exchanges": s["exchanges"],
            "learner_wins": s["wins"], "learner_losses": s["losses"], "draws": s["draws"],
            "causes": dict(s["causes"]),
            "mean_exchange_duration_s": float(np.mean(s["exchange_durations"])) if s["exchange_durations"] else None,
            "oob_events": s["oob_events"],
            "bounds_bad": s["bounds_bad"],
            "mean_match_return": float(np.mean(s["match_returns"])) if s["match_returns"] else None,
            "checkpoint": self.out_path,
            "interrupted": bool(self._stop),
            "partial_rollout_discarded": bool(self._partial),
            "scorer": self.stage_reward.status(),
            "policy_params": self.policy.n_params(),
        }

    @staticmethod
    def _summary_text(summary: dict) -> str:
        return ("[trainer] summary: " + ", ".join(
            f"{k}={v}" for k, v in summary.items() if k not in ("causes", "scorer")) +
            f" | causes={summary['causes']}")


# --------------------------------------------------------------------- eval
def evaluate(policy: ActorCritic, *, stage: StageConfig, env_kwargs: dict, episodes: int = 3,
             seed: int = 100, learner_robot: str = "a", frame_stack: int = 1,
             check_determinism: bool = True, max_steps: int | None = None) -> dict:
    """Deterministic seeded evaluation: per-exchange table + summary counts."""
    model = load_wrestling_model()
    cfg = PPOConfig(frame_stack=frame_stack, n_envs=1)
    mapper = action_mapper_for(model, learner_robot, cfg.net_config())
    opp_robot = "b" if learner_robot == "a" else "a"
    opp_mapper = action_mapper_for(model, opp_robot, cfg.net_config())
    mode = "pair" if stage.opponent.kind == "policy" else "learner_only"
    vec = VecWrestlingEnv(1, backend="sequential", seed=seed, learner_robot=learner_robot,
                          opponent_spec=stage.opponent, env_kwargs=env_kwargs,
                          weights=stage.weights, mode=mode)
    scheduler = CommandScheduler(stage.techniques, command_on=stage.command_on,
                                 phase_jitter_s=stage.perturbations.phase_jitter_s, seed=seed)
    builder = ActorObsBuilder(frame_stack)
    lit = 0 if learner_robot == "a" else 1
    opponent = None
    if mode == "pair":
        opp = make_policy(cfg)
        if stage.opponent.checkpoint:
            opp.load_state_dict(load_checkpoint(stage.opponent.checkpoint)["policy"])
        else:
            opp.load_state_dict(copy.deepcopy(policy.state_dict()))
        opp.eval()
        opponent = OpponentPolicy(opp, opp_mapper, robot=opp_robot)

    def run_episode(ep_seed: int) -> dict:
        batch = vec.reset(seeds=[ep_seed])
        cmds = [scheduler.command(0, 0, 0.0)]
        pair = (batch.obs_a[0], batch.obs_b[0])
        oa, ob = builder.reset(pair, (cmds[0], cmds[0]))
        obs = oa if lit == 0 else ob
        exchanges: list[dict] = []
        tele = {"steps": 0, "pelvis_z": [], "standing": 0, "ground": 0, "limb": 0,
                "dorsal": 0, "dist": []}
        steps = 0
        limit = int(max_steps or (float(env_kwargs.get("match_clock", 180.0)) / STEP_DT + 50))
        while steps < limit:
            with torch.no_grad():
                unit = policy.actor.deterministic_unit(torch.as_tensor(obs[None, :])).numpy()
            base = None
            if mapper.mode == "residual":
                base = reference_base_ctrl(cmds[0].technique, learner_robot, cmds[0].phase)
            ctrl = mapper.to_ctrl(unit[0], base)
            if mode == "learner_only":
                action = ctrl[None, :]
            else:
                ou = opponent.mean_unit(obs_b[None, :]) if lit == 0 else opponent.mean_unit(obs_a[None, :])
                oc = opp_mapper.to_ctrl(ou[0])
                action = compose_action(ctrl, oc, learner_robot)[None, :]
            batch = vec.step(action)
            info = batch.infos[0]
            rec = info.get("exchange_ended")
            if rec is not None:
                exchanges.append(dict(rec))
            p = unpack(batch.priv[0])
            q_self = p["qpos_a"] if learner_robot == "a" else p["qpos_b"]
            q_opp = p["qpos_b"] if learner_robot == "a" else p["qpos_a"]
            back = p["back_a"] if learner_robot == "a" else p["back_b"]
            contact = p["contact_a"] if learner_robot == "a" else p["contact_b"]
            tele["steps"] += 1
            tele["pelvis_z"].append(float(q_self[2]))
            tele["standing"] += int(q_self[2] >= 0.5)
            tele["ground"] += int(q_self[2] < 0.35)
            tele["limb"] += int(contact[1] > 0.5 or contact[2] > 0.5)   # knees/hands
            tele["dorsal"] += int(back[2] > 0.5)
            tele["dist"].append(float(np.linalg.norm(q_opp[0:3] - q_self[0:3])))
            next_cmds = [scheduler.command(0, int(info.get("exchange_index", 0)),
                                           float(info.get("exchange_time", 0.0)))]
            if batch.dones[0]:
                builder.reseed((batch.obs_a[0], batch.obs_b[0]))
            oa, ob = builder.step((batch.obs_a[0], batch.obs_b[0]), (next_cmds[0], next_cmds[0]))
            obs = oa if lit == 0 else ob
            cmds = next_cmds
            steps += 1
            if batch.dones[0]:
                break
        return {"episode_seed": int(ep_seed), "steps": tele["steps"], "exchanges": exchanges,
                "telemetry": {k: (float(np.mean(v)) if isinstance(v, list) and v else v)
                              for k, v in tele.items()}}

    runs = [run_episode(seed + i * RESET_SEED_STRIDE) for i in range(int(episodes))]
    determinism = None
    if check_determinism:
        twin = run_episode(seed)
        determinism = (twin["exchanges"] == runs[0]["exchanges"]
                       and abs(twin["steps"] - runs[0]["steps"]) == 0)
    vec.close()

    table = []
    for run in runs:
        for rec in run["exchanges"]:
            table.append({
                "episode_seed": run["episode_seed"], "index": rec.get("index"),
                "cause": rec.get("cause"), "winner": rec.get("winner"),
                "loser": rec.get("loser"), "duration": rec.get("duration"),
                "back_triggers": rec.get("back_triggers"),
                "oob_events": rec.get("oob_events"),
                "ambiguous": rec.get("ambiguous"),
            })
    causes = Counter(r["cause"] for r in table)
    wins = sum(1 for r in table if r["winner"] == learner_robot)
    losses = sum(1 for r in table if r["loser"] == learner_robot)
    summary = {
        "episodes": int(episodes),
        "episode_steps": [r["steps"] for r in runs],
        "exchanges": len(table),
        "causes": dict(causes),
        "learner_wins": wins, "learner_losses": losses,
        "draws": len(table) - wins - losses,
        "mean_exchange_duration_s": float(np.mean([r["duration"] for r in table])) if table else None,
        "oob_events": int(sum(sum(r["oob_events"].values()) for r in table)),
        "movement": {
            "mean_pelvis_z_m": float(np.mean([r["telemetry"]["pelvis_z"] for r in runs])),
            "standing_fraction": float(np.sum([r["telemetry"]["standing"] for r in runs])
                                       / max(1, np.sum([r["telemetry"]["steps"] for r in runs]))),
            "ground_fraction": float(np.sum([r["telemetry"]["ground"] for r in runs])
                                     / max(1, np.sum([r["telemetry"]["steps"] for r in runs]))),
            "knee_hand_contact_fraction": float(np.sum([r["telemetry"]["limb"] for r in runs])
                                                / max(1, np.sum([r["telemetry"]["steps"] for r in runs]))),
            "dorsal_contact_fraction": float(np.sum([r["telemetry"]["dorsal"] for r in runs])
                                             / max(1, np.sum([r["telemetry"]["steps"] for r in runs]))),
            "mean_pelvis_distance_m": float(np.mean([r["telemetry"]["dist"] for r in runs])),
        },
        "deterministic": determinism,
        "table": table,
    }
    return summary


if __name__ == "__main__":  # self-check: 2 short rollouts + update on the smoke config
    from .scripted import OpponentSpec

    cfg = PPOConfig(rollout_steps=64, n_envs=2, seed=0, frame_stack=1)
    cur = Curriculum()
    t = Trainer(cfg, out_path="/tmp/rl_trainer_selfcheck.pt", curriculum=cur,
                backend="sequential", env_kwargs={"match_clock": 6.0, "exchange_timeout": 3.0},
                ckpt_every=10_000, verbose=0)
    summary = t.run(200)
    assert summary["steps"] >= 200 and summary["bounds_bad"] == 0, summary
    assert Path(summary["checkpoint"]).exists()
    t2 = Trainer(cfg, out_path="/tmp/rl_trainer_selfcheck2.pt", backend="sequential",
                 env_kwargs={"match_clock": 6.0, "exchange_timeout": 3.0}, verbose=0)
    t2.load(summary["checkpoint"])
    assert t2.step_count == summary["steps"]
    print("rl.trainer self-check OK:", {k: summary[k] for k in ("steps", "matches", "exchanges",
                                                               "bounds_bad", "steps_per_s")})
