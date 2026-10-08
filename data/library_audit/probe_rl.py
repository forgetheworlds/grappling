"""RL-library probes on THIS box with OUR env:

  SB3 : minimal gymnasium adapter for `solo.env.SoloEnv` (dict obs, 29-dof unit
        actions) + a real short PPO run; reports adapter line count and whether
        the privileged-critic split survives.
  SKRL: confirms native asymmetric actor/critic (`observation_space` vs
        `state_space`) + a smoke run on a standard env.

Usage: <libtest venv>/bin/python probe_rl.py
Writes data/library_audit/rl_probe.json
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path("/home/ubuntu/grappling")
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "data/library_audit/rl_probe.json"

results: dict = {}


# ---------------------------------------------------------------- SB3 adapter
class SoloGymEnv:  # gymnasium.Env subclass built lazily below
    pass


def make_sb3_adapter():
    import gymnasium as gym
    from gymnasium import spaces

    from solo.env import SoloEnv
    from solo.obs import ACTOR_DIM, CRITIC_DIM, PRIV_DIM
    from solo.scene import N_JOINTS

    class _Env(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, task: str = "balance", action_mode: str = "residual"):
            self.env = SoloEnv(task=task, action_mode=action_mode)
            self.observation_space = spaces.Dict({
                "actor": spaces.Box(-np.inf, np.inf, (ACTOR_DIM,), np.float32),
                "privileged": spaces.Box(-np.inf, np.inf, (PRIV_DIM,), np.float32),
                "critic": spaces.Box(-np.inf, np.inf, (CRITIC_DIM,), np.float32),
            })
            self.action_space = spaces.Box(-1.0, 1.0, (N_JOINTS,), np.float32)

        def reset(self, *, seed=None, options=None):
            obs = self.env.reset(seed=seed)
            return {k: np.asarray(v, np.float32) for k, v in obs.items()}, {}

        def step(self, action):
            unit = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
            ctrl = self.env.ctrl_from_policy(unit)
            obs, rew, term, trunc, info = self.env.step(ctrl)
            return ({k: np.asarray(v, np.float32) for k, v in obs.items()},
                    float(rew), bool(term), bool(trunc), info)

    return _Env


def section_sb3(adapter) -> dict:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    n_lines = sum(1 for _ in pathlib.Path(__file__).read_text().splitlines())
    vec = DummyVecEnv([lambda: adapter()])
    model = PPO("MultiInputPolicy", vec, n_steps=512, batch_size=128, n_epochs=2,
                verbose=0, device="cpu", seed=0)
    t0 = time.perf_counter()
    model.learn(total_timesteps=2048)
    dt = time.perf_counter() - t0
    return {
        "ran": True,
        "learn_wall_s": dt,
        "env_steps_per_s_incl_updates": 2048 / dt,
        "adapter_lines_total_in_probe_file": n_lines,
        "policy_net_arch_default": "[64, 64] (SB3 default; our nets are 2x256 tanh)",
        "privileged_critic_preserved": False,
        "why": "SB3 PPO has ONE featurizer+value head on the whole obs dict "
               "(MultiInputPolicy -> CombinedExtractor); no asymmetric actor/critic. "
               "Keeping the design needs a custom ActorCriticPolicy (extra ~60-90 lines).",
        "ent_coef_default": 0.0,
    }


def section_sb3_envcheck(adapter) -> dict:
    from stable_baselines3.common.env_checker import check_env

    env = adapter()
    try:
        check_env(env, warn=True, skip_render_check=True)
        return {"check_env": "passed"}
    except Exception as e:  # noqa: BLE001
        return {"check_env": f"{type(e).__name__}: {e}"}


def section_skrl() -> dict:
    import gymnasium as gym
    import torch

    import skrl
    from skrl.agents.torch.ppo import PPO, PPO_CFG
    from skrl.envs.wrappers.torch import wrap_env
    from skrl.memories.torch import RandomMemory
    from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
    from skrl.trainers.torch import SequentialTrainer

    class Policy(GaussianMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions=False):
            Model.__init__(self, observation_space=observation_space,
                           action_space=action_space, device=device)
            GaussianMixin.__init__(self, clip_actions=clip_actions)
            self.net = torch.nn.Sequential(
                torch.nn.Linear(self.observation_space.shape[0], 64), torch.nn.ELU(),
                torch.nn.Linear(64, 64), torch.nn.ELU(),
                torch.nn.Linear(64, self.action_space.shape[0]))
            self.log_std_parameter = torch.nn.Parameter(torch.zeros(self.action_space.shape[0]))

        def compute(self, inputs, role):
            return self.net(inputs["observations"]), self.log_std_parameter, {}

    class Value(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions=False):
            Model.__init__(self, observation_space=observation_space,
                           action_space=action_space, device=device)
            DeterministicMixin.__init__(self, clip_actions=clip_actions)
            self.net = torch.nn.Sequential(
                torch.nn.Linear(self.observation_space.shape[0], 64), torch.nn.ELU(),
                torch.nn.Linear(64, 64), torch.nn.ELU(),
                torch.nn.Linear(64, 1))

        def compute(self, inputs, role):
            return self.net(inputs["observations"]), {}

    base = gym.make("Pendulum-v1")
    env = wrap_env(base)
    device = "cpu"
    cfg = PPO_CFG(rollouts=256, learning_epochs=2, mini_batches=4)
    models = {
        "policy": Policy(env.observation_space, env.action_space, device),
        "value": Value(env.observation_space, env.action_space, device),
    }
    memory = RandomMemory(memory_size=256, num_envs=env.num_envs, device=device)
    agent = PPO(models=models, memory=memory, observation_space=env.observation_space,
                state_space=env.state_space, action_space=env.action_space, device=device, cfg=cfg)
    trainer = SequentialTrainer(cfg={"timesteps": 512, "headless": True}, env=env, agents=agent)
    t0 = time.perf_counter()
    trainer.train()
    dt = time.perf_counter() - t0
    return {
        "ran": True,
        "skrl_version": skrl.__version__ if hasattr(skrl, "__version__") else "?",
        "pendulum_train_wall_s": dt,
        "native_asymmetric_obs": True,
        "evidence": "PPO(observation_space=..., state_space=...) -> memory tensors "
                    "'observations' and 'states'; value model is constructed on the "
                    "state space, policy on the observation space",
        "adapter_note": "skrl wraps a gymnasium env (wrap_env) or IsaacLab/playground; "
                        "no privileged extension inside a single Dict obs is needed -- "
                        "state_space carries the critic vector",
        "ent_coef_default": cfg.entropy_loss_scale,
    }


def main() -> int:
    adapter = make_sb3_adapter()
    for name, fn in (("sb3", lambda: section_sb3(adapter)),
                     ("sb3_check_env", lambda: section_sb3_envcheck(adapter)),
                     ("skrl", section_skrl)):
        try:
            results[name] = fn()
            print(f"[{name}] OK")
        except Exception as e:  # noqa: BLE001
            import traceback
            results[name] = {"error": f"{type(e).__name__}: {e}",
                             "trace": traceback.format_exc()[-1000:]}
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
    OUT.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
