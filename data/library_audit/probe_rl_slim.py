"""Slim, bounded RL probe: SB3 end-to-end on SoloEnv + skrl asymmetric wiring.

Stages print immediately (flushed) so a `timeout` kill still leaves evidence.
Usage: timeout 180 <libtest venv>/bin/python probe_rl_slim.py
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

REPO = "/home/ubuntu/grappling"
sys.path.insert(0, REPO + "/src")


def log(obj: dict) -> None:
    import os
    obj["loadavg"] = [round(x, 2) for x in os.getloadavg()]
    print(json.dumps(obj), flush=True)


def make_adapter():
    import gymnasium as gym
    from gymnasium import spaces

    from solo.env import SoloEnv
    from solo.obs import ACTOR_DIM, CRITIC_DIM, PRIV_DIM
    from solo.scene import N_JOINTS

    class SoloGym(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, task="balance", action_mode="residual"):
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
            ctrl = self.env.ctrl_from_policy(np.clip(np.asarray(action, np.float64), -1, 1))
            obs, rew, term, trunc, info = self.env.step(ctrl)
            info = dict(info)
            info.pop("episode", None)  # SB3/Monitor reserve the 'episode' info key
            return ({k: np.asarray(v, np.float32) for k, v in obs.items()},
                    float(rew), bool(term), bool(trunc), info)

    return SoloGym


def sb3(adapter, steps=None) -> None:
    import os as _os

    steps = int(_os.environ.get("RL_STEPS", "512")) if steps is None else steps
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    t0 = time.perf_counter()
    env = DummyVecEnv([adapter])
    log({"stage": "sb3_env_built", "wall_s": time.perf_counter() - t0})
    model = PPO("MultiInputPolicy", env, n_steps=256, batch_size=64, n_epochs=1,
                verbose=0, device="cpu", seed=0)
    t0 = time.perf_counter()
    model.learn(total_timesteps=steps)
    wall = time.perf_counter() - t0
    log({"stage": "sb3_learn_done", "steps": steps, "wall_s": wall,
         "env_steps_per_s": steps / wall})


def skrl(asymmetric=True) -> None:
    import torch
    from gymnasium import spaces
    from skrl.agents.torch.ppo import PPO, PPO_CFG
    from skrl.memories.torch import RandomMemory
    from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
    from solo.obs import ACTOR_DIM, CRITIC_DIM
    from solo.scene import N_JOINTS

    device = "cpu"

    class Policy(GaussianMixin, Model):
        def __init__(self, obs_space):
            Model.__init__(self, observation_space=obs_space,
                           action_space=spaces.Box(-1, 1, (N_JOINTS,), np.float32), device=device)
            GaussianMixin.__init__(self, clip_actions=False)
            self.net = torch.nn.Sequential(torch.nn.Linear(ACTOR_DIM, 64), torch.nn.ELU(),
                                           torch.nn.Linear(64, 64), torch.nn.ELU(),
                                           torch.nn.Linear(64, N_JOINTS))
            self.log_std_parameter = torch.nn.Parameter(torch.zeros(N_JOINTS))

        def compute(self, inputs, role):
            return self.net(inputs["observations"]), self.log_std_parameter, {}

    class Value(DeterministicMixin, Model):
        def __init__(self, obs_space):
            Model.__init__(self, observation_space=obs_space,
                           action_space=spaces.Box(-1, 1, (N_JOINTS,), np.float32), device=device)
            DeterministicMixin.__init__(self, clip_actions=False)
            self.net = torch.nn.Sequential(torch.nn.Linear(CRITIC_DIM, 64), torch.nn.ELU(),
                                           torch.nn.Linear(64, 64), torch.nn.ELU(),
                                           torch.nn.Linear(64, 1))

        def compute(self, inputs, role):
            return self.net(inputs["observations"]), {}

    actor_space = spaces.Box(-np.inf, np.inf, (ACTOR_DIM,), np.float32)
    critic_space = spaces.Box(-np.inf, np.inf, (CRITIC_DIM,), np.float32)
    models = {"policy": Policy(actor_space), "value": Value(critic_space)}
    memory = RandomMemory(memory_size=256, num_envs=1, device=device)
    agent = PPO(models=models, memory=memory, observation_space=actor_space,
                state_space=critic_space,
                action_space=spaces.Box(-1, 1, (N_JOINTS,), np.float32),
                device=device, cfg=PPO_CFG(rollouts=64, learning_epochs=1, mini_batches=2))
    agent.init()
    log({"stage": "skrl_agent_init_ok",
         "policy_input": ACTOR_DIM, "value_input": CRITIC_DIM,
         "policy_params": int(sum(p.numel() for p in agent.policy.parameters())),
         "value_params": int(sum(p.numel() for p in agent.value.parameters())),
         "entropy_loss_scale_default": agent.cfg.entropy_loss_scale})


def main() -> int:
    import os
    import traceback

    only = os.environ.get("RL_STAGE", "")
    adapter = make_adapter()
    if only != "skrl":
        try:
            sb3(adapter)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            log({"stage": "sb3_error", "error": f"{type(e).__name__}: {e}",
                 "trace": traceback.format_exc()[-900:]})
    if only != "sb3":
        try:
            skrl()
        except Exception as e:  # noqa: BLE001
            log({"stage": "skrl_error", "error": f"{type(e).__name__}: {e}"})

    os._exit(0)  # torch/xla background threads can hang interpreter shutdown
    return 0


if __name__ == "__main__":
    sys.exit(main())
