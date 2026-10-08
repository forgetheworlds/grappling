"""Rollout vectorization: N independent wrestling matches for one learner.

Two interchangeable backends behind one interface:

* ``sequential`` -- envs stepped one after another in the calling process;
* ``subproc``    -- one worker process per env (fork), actions/observations
  over pipes; the parent does policy/network work while workers simulate.

Both return the same batched transition: env obs pairs, the privileged vector
(``rl.privileged``), the env's weighted rewards, done flags and info dicts.
Matches auto-reset inside the backend with deterministic per-match seeds
(``seed + match_index * RESET_SEED_STRIDE``); the returned obs on a ``done`` is
the *reset* obs, and ``(1 - done)`` masks the bootstrap value in GAE.

Action convention: ``mode="learner_only"`` takes ``(N, 29)`` learner actions
and fills the opponent half from the stage's scripted controller inside the
env; ``mode="pair"`` takes full ``(N, 58)`` actions (policy-vs-policy).

``python -m src.rl.vec`` measures both backends on this host (see report).
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass, field

import numpy as np

from wrestling.env import EXCHANGE_TIMEOUT, MATCH_CLOCK, N_JOINTS, WrestlingEnv

from .privileged import PRIV_DIM, PrivilegedObsBuilder
from .reward import RewardWeights, StageReward
from .scripted import OpponentSpec

#: auto-reset match seeds: ``seed + match_index * RESET_SEED_STRIDE``
RESET_SEED_STRIDE = 1_000_003
#: worker startup grace (s) for the subproc backend
_START_TIMEOUT = 120.0


@dataclass
class StepBatch:
    """One batched control step from N envs."""

    obs_a: np.ndarray          # (N, 84) float32
    obs_b: np.ndarray          # (N, 84) float32
    priv: np.ndarray           # (N, PRIV_DIM) float32
    rewards: np.ndarray        # (N, 2) float64
    dones: np.ndarray          # (N,) bool
    infos: list = field(default_factory=list)
    bounds_bad: np.ndarray | None = None   # (N,) int: learner action entries outside ctrlrange

    @property
    def n(self) -> int:
        return int(self.obs_a.shape[0])

    def obs_pair(self, i: int):
        return self.obs_a[i], self.obs_b[i]


def _env_kwargs_default() -> dict:
    return {"match_clock": MATCH_CLOCK, "exchange_timeout": EXCHANGE_TIMEOUT}


def build_env(seed: int, *, opponent_spec: OpponentSpec, learner_robot: str,
              env_kwargs: dict | None = None, weights: RewardWeights | None = None,
              mode: str = "learner_only") -> WrestlingEnv:
    """Construct one env with its scripted opponent + stage event weights."""
    kw = dict(_env_kwargs_default())
    kw.update(env_kwargs or {})
    opp_robot = "b" if learner_robot == "a" else "a"
    ctrl = opponent_spec.make(opp_robot) if mode == "learner_only" else None
    if mode == "learner_only" and ctrl is None:
        raise ValueError(f"mode 'learner_only' needs a scripted controller, got kind "
                         f"{opponent_spec.kind!r}")
    controllers = (ctrl, None) if learner_robot == "b" else (None, ctrl)
    env = WrestlingEnv(controllers=controllers, seed=int(seed), **kw)
    stage_reward = StageReward(weights or RewardWeights())
    env.reward_fn = stage_reward.reward_fn
    env.reset(seed=int(seed))
    return env


def _learner_bounds(model, learner_robot: str):
    sl = slice(0, N_JOINTS) if learner_robot == "a" else slice(N_JOINTS, 2 * N_JOINTS)
    return (np.asarray(model.actuator_ctrlrange[:, 0][sl], np.float64),
            np.asarray(model.actuator_ctrlrange[:, 1][sl], np.float64))


def _count_bad(u: np.ndarray, lo: np.ndarray, hi: np.ndarray, tol: float = 1e-9) -> int:
    a = np.asarray(u, np.float64).reshape(-1)
    return int(np.sum((a < lo - tol) | (a > hi + tol)))


class _ChildState:
    """Per-worker env + privileged builder + seed bookkeeping (subproc side)."""

    def __init__(self, seed: int, opponent_spec: OpponentSpec, learner_robot: str,
                 env_kwargs: dict, weights: RewardWeights, mode: str):
        self.learner_robot = learner_robot
        self.mode = mode
        self.seed = int(seed)
        self.match_index = 0
        self.env = build_env(seed, opponent_spec=opponent_spec, learner_robot=learner_robot,
                             env_kwargs=env_kwargs, weights=weights, mode=mode)
        self.priv_builder = PrivilegedObsBuilder(self.env.model)
        self.lo, self.hi = _learner_bounds(self.env.model, learner_robot)
        self.learner_slice = slice(0, N_JOINTS) if learner_robot == "a" else slice(N_JOINTS, 2 * N_JOINTS)
        self.opp_slice = slice(N_JOINTS, 2 * N_JOINTS) if learner_robot == "a" else slice(0, N_JOINTS)
        self.opp_index = 0 if learner_robot == "b" else 1

    def _next_seed(self) -> int:
        self.match_index += 1
        return self.seed + self.match_index * RESET_SEED_STRIDE

    def _result(self, obs, reward, done, info, bounds_bad: int, final_info=None):
        obs_a, obs_b = obs
        priv = self.priv_builder.build(self.env.model, self.env.data)
        if final_info is not None:
            info = dict(info)
            info["final_info"] = final_info
        return (np.asarray(obs_a, np.float32), np.asarray(obs_b, np.float32), priv,
                np.asarray(reward, np.float64), bool(done), info, int(bounds_bad))

    def step(self, action):
        u = np.asarray(action, np.float64)
        if self.mode == "learner_only":
            ctrl = self.env.controllers[self.opp_index]
            opp_u = np.asarray(ctrl(self.env, self.env.data), np.float64)
            full = np.empty(2 * N_JOINTS)
            full[self.learner_slice] = u
            full[self.opp_slice] = opp_u
        else:
            full = u
        bad = _count_bad(full[self.learner_slice], self.lo, self.hi)
        obs, reward, terminated, _trunc, info = self.env.step(full)
        final_info = None
        if terminated:
            final_info = {
                "match_index": int(self.match_index),
                "match_time": float(info["t"]),
                "completed_exchanges": len(self.env.exchange_log),
                "next_match_seed": int(self.seed + (self.match_index + 1) * RESET_SEED_STRIDE),
            }
            obs = self.env.reset(seed=self._next_seed())
        return self._result(obs, reward, terminated, info, bad, final_info)

    def reset(self, seed=None, pose=None):
        if seed is None:
            seed = self.seed + self.match_index * RESET_SEED_STRIDE
        obs = self.env.reset(seed=int(seed), pose=pose)
        return self._result(obs, (0.0, 0.0), False, {}, 0)


def _worker_main(conn, seed, opponent_spec, learner_robot, env_kwargs, weights, mode):
    """Subprocess entry: own env, serve step/reset/close until told to stop.

    SIGINT is ignored here: the parent owns the interrupt (save + graceful stop).
    """
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        st = _ChildState(seed, opponent_spec, learner_robot, env_kwargs, weights, mode)
        conn.send(("ready", None))
    except Exception as exc:  # pragma: no cover - startup failure path
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
        conn.close()
        return
    try:
        while True:
            msg = conn.recv()
            cmd = msg[0]
            if cmd == "close":
                break
            if cmd == "step":
                conn.send(("ok", st.step(msg[1])))
            elif cmd == "reset":
                _seed, pose = msg[1], msg[2]
                conn.send(("ok", st.reset(_seed, pose)))
            else:  # pragma: no cover
                conn.send(("error", f"unknown command {cmd!r}"))
    except (EOFError, KeyboardInterrupt):  # pragma: no cover - parent vanished
        pass
    finally:
        conn.close()


class VecWrestlingEnv:
    """N independent matches; identical transition format from both backends."""

    def __init__(self, n_envs: int, *, backend: str = "sequential", seed: int = 0,
                 learner_robot: str = "a", opponent_spec: OpponentSpec | None = None,
                 env_kwargs: dict | None = None, weights: RewardWeights | None = None,
                 mode: str = "learner_only", start_method: str = "fork"):
        if n_envs < 1:
            raise ValueError("n_envs must be >= 1")
        if learner_robot not in ("a", "b"):
            raise ValueError(learner_robot)
        if mode not in ("learner_only", "pair"):
            raise ValueError(f"unknown mode {mode!r}")
        if backend not in ("sequential", "subproc"):
            raise ValueError(f"unknown backend {backend!r}")
        self.n_envs = int(n_envs)
        self.backend = backend
        self.mode = mode
        self.learner_robot = learner_robot
        self.opponent_spec = opponent_spec or OpponentSpec()
        self.env_kwargs = dict(env_kwargs or {})
        self.weights = weights or RewardWeights()
        self.seed = int(seed)
        self._actions_dim = N_JOINTS if mode == "learner_only" else 2 * N_JOINTS
        self._closed = False
        self._startup_s = 0.0
        if backend == "sequential":
            t0 = time.perf_counter()
            self._states = [
                _ChildState(seed + i, self.opponent_spec, learner_robot, self.env_kwargs,
                            self.weights, mode)
                for i in range(self.n_envs)
            ]
            self._startup_s = time.perf_counter() - t0
        else:
            ctx = mp.get_context(start_method)
            t0 = time.perf_counter()
            self._conns = []
            self._procs = []
            for i in range(self.n_envs):
                parent, child = ctx.Pipe()
                p = ctx.Process(target=_worker_main,
                                args=(child, seed + i, self.opponent_spec, learner_robot,
                                      self.env_kwargs, self.weights, mode),
                                daemon=True)
                p.start()
                child.close()
                self._conns.append(parent)
                self._procs.append(p)
            for conn in self._conns:
                self._wait_ready(conn)
            self._startup_s = time.perf_counter() - t0

    # ------------------------------------------------------------------ comms
    @staticmethod
    def _wait_ready(conn) -> None:
        status, payload = conn.recv()
        if status == "error":
            raise RuntimeError(f"vec worker failed to start: {payload}")

    @staticmethod
    def _recv_ok(conn):
        status, payload = conn.recv()
        if status != "ok":
            raise RuntimeError(f"vec worker error: {payload}")
        return payload

    def _send_all(self, cmd, arrays=None, poses=None) -> None:
        for i, conn in enumerate(self._conns):
            if cmd == "step":
                conn.send(("step", arrays[i]))
            elif cmd == "reset":
                conn.send(("reset", None if arrays is None else arrays[i], poses[i] if poses else None))
            else:  # pragma: no cover
                raise AssertionError(cmd)

    # ------------------------------------------------------------------- API
    def reset(self, seeds=None, poses=None) -> StepBatch:
        """Reset all envs (default: current deterministic per-env seeds)."""
        self._check_open()
        if self.backend == "sequential":
            results = [st.reset(None if seeds is None else seeds[i], None if poses is None else poses[i])
                       for i, st in enumerate(self._states)]
        else:
            self._send_all("reset", arrays=seeds, poses=poses)
            results = [self._recv_ok(conn) for conn in self._conns]
        return self._batch(results)

    def step(self, actions) -> StepBatch:
        """Step all envs with ``(N, 29)`` (learner_only) or ``(N, 58)`` (pair)."""
        self._check_open()
        a = np.asarray(actions, dtype=np.float64)
        if a.shape != (self.n_envs, self._actions_dim):
            raise ValueError(f"actions shape {a.shape} != ({self.n_envs}, {self._actions_dim})")
        if self.backend == "sequential":
            results = [st.step(a[i]) for i, st in enumerate(self._states)]
        else:
            self._send_all("step", arrays=a)
            results = [self._recv_ok(conn) for conn in self._conns]
        return self._batch(results)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.backend == "subproc":
            for conn in self._conns:
                try:
                    conn.send(("close", None))
                except (BrokenPipeError, OSError):
                    pass
            for p in self._procs:
                p.join(timeout=5.0)
                if p.is_alive():  # pragma: no cover - defensive
                    p.terminate()
            for conn in self._conns:
                conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass

    # ---------------------------------------------------------------- helpers
    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("VecWrestlingEnv is closed")

    @staticmethod
    def _batch(results) -> StepBatch:
        obs_a = np.stack([r[0] for r in results]).astype(np.float32)
        obs_b = np.stack([r[1] for r in results]).astype(np.float32)
        priv = np.stack([r[2] for r in results]).astype(np.float32)
        rewards = np.stack([r[3] for r in results]).astype(np.float64)
        dones = np.asarray([r[4] for r in results], dtype=bool)
        infos = [r[5] for r in results]
        bad = np.asarray([r[6] for r in results], dtype=np.int64)
        return StepBatch(obs_a, obs_b, priv, rewards, dones, infos, bad)


# ------------------------------------------------------------------ benchmark
def benchmark(n_envs: int = 3, steps: int = 200, seed: int = 0, warmup: int = 5) -> dict:
    """Measure both backends on this host (StandHold opponent, uniform actions).

    Returns a dict of measurements; ``python -m src.rl.vec`` prints it.
    """
    from wrestling.env import load_wrestling_model

    env_kwargs = {"match_clock": 30.0, "exchange_timeout": 6.0}
    model = load_wrestling_model()
    lo = np.asarray(model.actuator_ctrlrange[0:N_JOINTS, 0], np.float64)
    hi = np.asarray(model.actuator_ctrlrange[0:N_JOINTS, 1], np.float64)
    mid = 0.5 * (lo + hi)
    rng = np.random.default_rng(seed)
    actions = mid[None, :] + 0.2 * (hi - lo)[None, :] * rng.uniform(-1, 1, size=(n_envs, N_JOINTS))
    out = {}
    for backend in ("sequential", "subproc"):
        vec = VecWrestlingEnv(n_envs, backend=backend, seed=seed,
                              opponent_spec=OpponentSpec("stand_hold"),
                              env_kwargs=env_kwargs)
        vec.reset()
        for _ in range(warmup):
            vec.step(actions)
        t0 = time.perf_counter()
        for _ in range(steps):
            batch = vec.step(actions)
        wall = time.perf_counter() - t0
        vec.close()
        out[backend] = {
            "n_envs": n_envs, "steps_per_env": steps, "wall_s": wall,
            "env_steps": n_envs * steps,
            "env_steps_per_s": (n_envs * steps) / wall,
            "ms_per_step_batch": 1000.0 * wall / steps,
            "startup_s": vec._startup_s,
            "last_rewards_mean": float(batch.rewards.mean()),
        }
    # single-env sequential reference (raw env throughput)
    vec1 = VecWrestlingEnv(1, backend="sequential", seed=seed,
                           opponent_spec=OpponentSpec("stand_hold"), env_kwargs=env_kwargs)
    actions1 = actions[:1]
    vec1.reset()
    t0 = time.perf_counter()
    for _ in range(steps):
        vec1.step(actions1)
    wall1 = time.perf_counter() - t0
    vec1.close()
    out["sequential_1_env"] = {"env_steps": steps, "wall_s": wall1,
                               "env_steps_per_s": steps / wall1,
                               "ms_per_step": 1000.0 * wall1 / steps}
    return out


if __name__ == "__main__":  # benchmark + self-check
    res = benchmark()
    import json

    print("rl.vec benchmark (backend comparison on this host):")
    print(json.dumps(res, indent=2, sort_keys=True))
    seq = res["sequential"]["env_steps_per_s"]
    sub = res["subproc"]["env_steps_per_s"]
    print(f"VERDICT: subproc/sequential throughput ratio = {sub / seq:.2f}x "
          f"({sub:.0f} vs {seq:.0f} env-steps/s)")
    # correctness self-check: same seeds + same actions -> identical transitions
    kw = {"match_clock": 10.0, "exchange_timeout": 4.0}
    fixed = np.zeros((2, N_JOINTS))
    v1 = VecWrestlingEnv(2, backend="sequential", seed=5, opponent_spec=OpponentSpec("stand_hold"), env_kwargs=kw)
    r1 = v1.reset()
    b1 = v1.step(fixed)
    v1.close()
    v2 = VecWrestlingEnv(2, backend="subproc", seed=5, opponent_spec=OpponentSpec("stand_hold"), env_kwargs=kw)
    r2 = v2.reset()
    b2 = v2.step(fixed)
    v2.close()
    assert np.allclose(r1.obs_a, r2.obs_a), "reset mismatch between backends"
    assert np.allclose(b1.obs_a, b2.obs_a) and np.allclose(b1.priv, b2.priv), "step mismatch"
    print("rl.vec self-check OK: sequential and subproc transitions agree")
