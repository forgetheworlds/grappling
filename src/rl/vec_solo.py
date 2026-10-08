"""Vectorised solo-drill env: N independent ``SoloEnv`` workers for one learner.

Design
------
``VecSoloEnv`` owns ``n_envs`` environments behind the interface the
single-env trainer uses.  Two interchangeable backends:

* ``subproc``    -- one child process per env (fork); the parent does
  policy/network work while the children simulate.  Per control step the
  parent sends each worker its 29-dim *unit* action and receives the
  observation dict (``{"actor", "privileged", "critic"}`` float32), the reward,
  the terminated/truncated flags and the env ``info``.
* ``sequential`` -- the same protocol stepped in-process, one env at a time
  (no-IPC reference; lets tests separate physics equivalence from IPC).

The action-mode contract (``SoloEnv.ctrl_from_policy`` / ``resolve_action``)
stays inside the environment: the parent only ever ships unit actions, exactly
like ``SoloTrainer``'s single-env path.

Semantics that must not change (asserted by ``tests/solo/test_vec_solo.py``)
---------------------------------------------------------------------------
* **Episode reset** -- ``terminated`` or ``truncated`` ends an episode and the
  *parent* resets that env (never a worker) with the next seed of that env's
  stream.  One owner for seed bookkeeping; mirrors ``SoloTrainer``'s
  ``episode_seed += 1`` at every episode end.
* **Per-env seed streams** -- worker ``i`` starts at ``seed + i * SEED_STRIDE``
  and advances by +1 per episode.  ``n_envs=1`` reduces exactly to the
  trainer's stream (``seed, seed + 1, seed + 2, ...``): that is what makes the
  equivalence test byte-exact.  The stride keeps the N streams disjoint.
* **Push curriculum** -- recomputed by the parent at each reset with
  ``schedule_for(steps, seed)`` where ``steps`` is the total number of env
  transitions so far (the trainer's ``steps_done``; ``n_envs=1`` reduces to the
  trainer's own call sequence).
* **Observation dict** -- keys/shapes/dtypes come from ``SoloEnv``; nothing
  here rewrites them (tested).
* **Reset reproducibility** -- ``state()``/``config()`` record each worker's
  base seed and current episode index, and ``load_state()`` re-resets every
  env at its recorded seed, so a checkpointed run reloads onto the same four
  streams.

Worker death is surfaced: every exchange is guarded, and a dead/killed worker
raises ``RuntimeError`` ("vec_solo worker <i> died ...") instead of silently
returning stale observations.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass

import numpy as np

from solo.env import SoloEnv
from solo.scene import N_JOINTS

#: per-worker base-seed stride: worker ``i`` starts at ``seed + i * SEED_STRIDE``
#: (same convention as :data:`rl.vec.RESET_SEED_STRIDE`; disjoint streams).
SEED_STRIDE = 1_000_003
#: worker startup + ready-handshake grace (s)
START_TIMEOUT = 180.0
#: join grace when closing a worker (s)
_JOIN_TIMEOUT = 5.0
_OBS_KEYS = ("actor", "privileged", "critic")


@dataclass
class SoloVecBatch:
    """One batched control step from N envs.

    ``obs``      -- ``{"actor", "privileged", "critic"}`` float32 ``(N, D)``
    ``rewards``  -- ``(N,)`` float64
    ``terminated``/``truncated`` -- ``(N,)`` bool
    ``infos``    -- per-env ``info`` dicts (``None`` when disabled)
    ``seeds``    -- ``(N,)`` int64, the seed of each env's *current* episode
                    (already advanced for envs that reset during this step)
    """

    obs: dict[str, np.ndarray]
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    infos: list
    seeds: np.ndarray


class _EnvServer:
    """One env + its seed/curriculum bookkeeping (worker side or in-process)."""

    def __init__(self, seed: int, *, task: str, weights, action_mode: str,
                 residual_scale: float, curriculum, model, record_metrics: bool,
                 include_state: bool, include_info: bool, term_set=None,
                 joint_mask=None):
        self.include_state = bool(include_state)
        self.include_info = bool(include_info)
        self.curriculum = curriculum
        self.env = SoloEnv(model, task=task, seed=int(seed), weights=weights,
                           action_mode=action_mode,
                           residual_scale=residual_scale,
                           term_set=term_set, joint_mask=joint_mask,
                           record_metrics=record_metrics)
        if curriculum is not None:
            # mirror SoloTrainer.__init__: install the schedule, then reset
            self.env.set_push_schedule(curriculum.schedule_for(0, int(seed)))
            self.env.reset(seed=int(seed))
        self.seed = int(seed)

    # ------------------------------------------------------------- requests
    def ready(self):
        return (self.env.observation(), self._state(), self.env.config())

    def step(self, unit):
        ctrl = self.env.ctrl_from_policy(unit)      # the mode contract lives here
        obs, reward, terminated, truncated, info = self.env.step(ctrl)
        return (obs, float(reward), bool(terminated), bool(truncated),
                info if self.include_info else None, self._state())

    def reset(self, seed, steps):
        seed = self.seed if seed is None else int(seed)
        if self.curriculum is not None:
            self.env.set_push_schedule(self.curriculum.schedule_for(int(steps), seed))
        self.seed = seed
        return (self.env.reset(seed=seed), self._state())

    def probe(self, unit):
        """Startup mapping check on this env, then re-reset the episode.

        Mirrors ``SoloTrainer._startup_check``: one deterministic step, measure
        ``|ctrl - base_action|``, then reset to the same episode seed (the
        curriculum is deliberately *not* recomputed, exactly like the
        single-env check).
        """
        ctrl = self.env.ctrl_from_policy(unit)
        self.env.step(ctrl)
        diff = float(np.abs(np.asarray(self.env.data.ctrl, np.float64)
                            - self.env._base_action).max())
        return (diff, self.env.reset(seed=self.seed), self._state())

    def _state(self):
        return self.env.data.qpos.copy() if self.include_state else None


def _worker_main(conn, init: dict) -> None:
    """Subprocess entry: own env, serve step/reset/probe until told to close.

    SIGINT is ignored here: the parent owns the interrupt (save + graceful stop).
    """
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        server = _EnvServer(**init)
        conn.send(("ready", server.ready()))
    except Exception as exc:  # pragma: no cover - startup failure path
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
        conn.close()
        return
    try:
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                break
            cmd = msg[0]
            if cmd == "close":
                break
            try:
                if cmd == "step":
                    conn.send(("ok", server.step(msg[1])))
                elif cmd == "reset":        # payload: (seed, total_steps)
                    conn.send(("ok", server.reset(*msg[1])))
                elif cmd == "probe":
                    conn.send(("ok", server.probe(msg[1])))
                else:  # pragma: no cover
                    conn.send(("error", f"unknown command {cmd!r}"))
            except Exception as exc:  # surface, do not die silently
                try:
                    conn.send(("error", f"{type(exc).__name__}: {exc}"))
                except Exception:
                    pass
                break
    finally:
        conn.close()


class _ProcWorker:
    """Parent-side handle for one subprocess worker (guarded IPC)."""

    def __init__(self, idx: int, proc: mp.Process, conn):
        self.idx = idx
        self.proc = proc
        self.conn = conn

    def send(self, msg) -> None:
        try:
            self.conn.send(msg)
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            raise RuntimeError(
                f"vec_solo worker {self.idx} is not reachable "
                f"(exitcode={self.proc.exitcode}); refusing to continue") from exc

    def recv_status(self, timeout: float | None = None) -> tuple:
        """Receive one ``(status, payload)`` message; surface worker death."""
        try:
            if timeout is not None and not self.conn.poll(timeout):
                raise RuntimeError(
                    f"vec_solo worker {self.idx} sent nothing within {timeout}s")
            return self.conn.recv()
        except (EOFError, ConnectionResetError, OSError) as exc:
            raise RuntimeError(
                f"vec_solo worker {self.idx} died (exitcode={self.proc.exitcode}): "
                f"refusing to return stale observations") from exc

    def recv(self):
        status, payload = self.recv_status()
        if status != "ok":
            raise RuntimeError(f"vec_solo worker {self.idx} error: {payload}")
        return payload

    def shutdown(self) -> None:
        try:
            if self.proc.is_alive():
                try:
                    self.conn.send(("close", None))
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
            self.proc.join(timeout=_JOIN_TIMEOUT)
            if self.proc.is_alive():  # pragma: no cover - defensive
                self.proc.terminate()
                self.proc.join(timeout=_JOIN_TIMEOUT)
            if self.proc.is_alive():  # pragma: no cover - defensive
                self.proc.kill()
        finally:
            try:
                self.conn.close()
            except OSError:  # pragma: no cover
                pass


class VecSoloEnv:
    """N independent ``SoloEnv`` instances behind one batched interface.

    ``term_set``/``joint_mask`` are forwarded verbatim into every worker's
    ``SoloEnv(...)`` (literature reward set / frozen-joint action mask); both
    default to ``None`` = today's behaviour.  Callers build the mask from the
    model (``solo.lit.joint_mask(model)``) -- it is a frozen dataclass, and the
    model is identical in every worker.
    """

    def __init__(self, n_envs: int, *, backend: str = "subproc", seed: int = 0,
                 task: str = "balance", action_mode: str = "absolute",
                 residual_scale: float = 0.5, weights=None, curriculum=None,
                 model=None, record_metrics: bool = True,
                 include_state: bool = False, include_info: bool = True,
                 term_set: str | None = None, joint_mask=None,
                 start_method: str = "fork"):
        if int(n_envs) < 1:
            raise ValueError("n_envs must be >= 1")
        if backend not in ("sequential", "subproc"):
            raise ValueError(f"unknown backend {backend!r}")
        if action_mode not in ("absolute", "residual"):
            raise ValueError(action_mode)
        if model is not None and start_method != "fork":
            raise ValueError("model can only be shared through the fork start method")
        self.n_envs = int(n_envs)
        self.backend = backend
        self.base_seed = int(seed)
        self.task = task
        self.action_mode = action_mode
        self.residual_scale = float(residual_scale)
        self.include_state = bool(include_state)
        #: worker i starts at base_seed + i * SEED_STRIDE, +1 per episode
        self.worker_seeds = [self.base_seed + i * SEED_STRIDE
                             for i in range(self.n_envs)]
        self.episode_indices = [0] * self.n_envs
        #: total env transitions stepped through this object (== trainer steps_done)
        self.steps = 0
        self.obs: dict[str, np.ndarray] = {}
        self.qpos: np.ndarray | None = None
        self._env_config0: dict = {}
        self._closed = False
        init = {"task": task, "weights": weights, "action_mode": action_mode,
                "residual_scale": float(residual_scale), "curriculum": curriculum,
                "model": model, "record_metrics": bool(record_metrics),
                "include_state": self.include_state,
                "include_info": bool(include_info),
                "term_set": term_set, "joint_mask": joint_mask}
        t0 = time.perf_counter()
        if backend == "sequential":
            self._servers = [_EnvServer(self.worker_seeds[i], **init)
                             for i in range(self.n_envs)]
            self._workers: list[_ProcWorker] = []
            ready = [s.ready() for s in self._servers]
        else:
            self._servers = []
            self._workers = []
            ctx = mp.get_context(start_method)
            try:
                for i in range(self.n_envs):
                    parent_conn, child_conn = ctx.Pipe()
                    p = ctx.Process(target=_worker_main,
                                    args=(child_conn, dict(init, seed=self.worker_seeds[i])),
                                    daemon=True)
                    p.start()
                    child_conn.close()
                    self._workers.append(_ProcWorker(i, p, parent_conn))
                ready = []
                for w in self._workers:
                    status, payload = w.recv_status(timeout=START_TIMEOUT)
                    if status != "ready":
                        raise RuntimeError(
                            f"vec_solo worker {w.idx} failed to start: {payload}")
                    ready.append(payload)
            except Exception:
                self.close()
                raise
        self.startup_s = time.perf_counter() - t0
        self._set_obs(ready)
        self._env_config0 = dict(ready[0][2])

    # ------------------------------------------------------------------ API
    @property
    def current_seeds(self) -> list[int]:
        """Seed of each env's current (in-flight) episode."""
        return [self.worker_seeds[i] + self.episode_indices[i]
                for i in range(self.n_envs)]

    @property
    def worker_pids(self) -> list[int]:
        """Worker process ids (diagnostics/fault injection; [] for sequential)."""
        return [w.proc.pid for w in self._workers]

    def step(self, actions) -> SoloVecBatch:
        """Step all envs with ``(N, 29)`` unit actions.

        Terminated/truncated envs are reset in this call (next seed of their
        stream, push schedule recomputed for the new ``steps`` count); the
        returned obs row for a reset env is its *reset* obs -- the same
        convention as ``SoloTrainer``'s single-env loop.
        """
        self._check_open()
        a = np.asarray(actions, dtype=np.float64)
        if a.shape != (self.n_envs, N_JOINTS):
            raise ValueError(
                f"actions shape {a.shape} != ({self.n_envs}, {N_JOINTS})")
        if self.backend == "subproc":
            for i, w in enumerate(self._workers):
                w.send(("step", a[i]))
            results = [w.recv() for w in self._workers]
        else:
            results = [s.step(a[i]) for i, s in enumerate(self._servers)]
        self.steps += self.n_envs
        obs = [r[0] for r in results]
        rewards = np.asarray([r[1] for r in results], dtype=np.float64)
        terminated = np.asarray([r[2] for r in results], dtype=bool)
        truncated = np.asarray([r[3] for r in results], dtype=bool)
        infos = [r[4] for r in results]
        states = [r[5] for r in results]
        done = np.flatnonzero(terminated | truncated)
        if done.size:  # parent owns resets and the seed bookkeeping
            resets = []
            for i in done.tolist():
                self.episode_indices[i] += 1
                resets.append((i, self.worker_seeds[i] + self.episode_indices[i],
                               self.steps))
            if self.backend == "subproc":
                for i, seed_i, steps_i in resets:
                    self._workers[i].send(("reset", (seed_i, steps_i)))
                for i, _seed_i, _steps_i in resets:
                    obs[i], states[i] = self._workers[i].recv()
            else:
                for i, seed_i, steps_i in resets:
                    obs[i], states[i] = self._servers[i].reset(seed_i, steps_i)
        self._set_obs(list(zip(obs, states)))
        return SoloVecBatch(obs=self.obs, rewards=rewards, terminated=terminated,
                            truncated=truncated, infos=infos,
                            seeds=np.asarray(self.current_seeds, dtype=np.int64))

    def reset(self, seeds=None, *, steps: int | None = None) -> dict:
        """Reset all envs (default: current episode seeds); returns batched obs."""
        self._check_open()
        seeds = self.current_seeds if seeds is None else [int(s) for s in seeds]
        if len(seeds) != self.n_envs:
            raise ValueError(f"seeds must have {self.n_envs} entries")
        if steps is not None:
            self.steps = int(steps)
        if self.backend == "subproc":
            for i, w in enumerate(self._workers):
                w.send(("reset", (seeds[i], self.steps)))
            results = [w.recv() for w in self._workers]
        else:
            results = [s.reset(seeds[i], self.steps)
                       for i, s in enumerate(self._servers)]
        self._set_obs(results)
        return self.obs

    def probe(self, unit) -> float:
        """Startup mapping check on env 0 (see ``_EnvServer.probe``)."""
        self._check_open()
        u = np.asarray(unit, dtype=np.float64).reshape(N_JOINTS)
        if self.backend == "subproc":
            self._workers[0].send(("probe", u))
            diff, obs0, state0 = self._workers[0].recv()
        else:
            diff, obs0, state0 = self._servers[0].probe(u)
        for k in _OBS_KEYS:
            self.obs[k][0] = obs0[k]
        if self.include_state and state0 is not None and self.qpos is not None:
            self.qpos[0] = state0
        return float(diff)

    # ------------------------------------------------------- state / config
    def state(self) -> dict:
        """JSON-able resume state: seeds + episode indices + step count."""
        return {"steps": int(self.steps),
                "worker_seeds": list(self.worker_seeds),
                "episode_indices": list(self.episode_indices),
                "current_seeds": list(self.current_seeds)}

    def load_state(self, state: dict) -> None:
        """Restore seed bookkeeping and re-reset every env at its saved seed.

        Same replay semantics as ``SoloTrainer.load``: the saved *current*
        episode seeds are replayed, with the push schedule recomputed at the
        saved step count.
        """
        saved = [int(s) for s in state.get("worker_seeds") or []]
        if saved and saved != list(self.worker_seeds):
            raise ValueError(
                f"checkpoint worker_seeds {saved} != this vec env's "
                f"{self.worker_seeds}")
        idx = state.get("episode_indices")
        if idx is not None:
            if len(idx) != self.n_envs:
                raise ValueError("episode_indices length mismatch")
            self.episode_indices = [int(x) for x in idx]
        self.steps = int(state.get("steps", 0))
        self.reset(seeds=self.current_seeds, steps=self.steps)

    def config(self) -> dict:
        """JSON-able description: worker-0 env conditions + the vec/seed block."""
        out = dict(self._env_config0)
        out.update({
            "vec": True,
            "n_envs": self.n_envs,
            "backend": self.backend,
            "seed": self.base_seed,
            "worker_seeds": list(self.worker_seeds),
            "episode_indices": list(self.episode_indices),
            "current_seeds": list(self.current_seeds),
            "steps": int(self.steps),
        })
        return out

    # ---------------------------------------------------------------- close
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for w in self._workers:
            w.shutdown()
        self._workers = []

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

    # -------------------------------------------------------------- helpers
    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("VecSoloEnv is closed")

    def _set_obs(self, results) -> None:
        """``results``: list of ``(obs_dict, state_or_None)`` in worker order."""
        self.obs = {k: np.stack([r[0][k] for r in results]).astype(np.float32,
                                                                   copy=False)
                    for k in _OBS_KEYS}
        if self.include_state:
            self.qpos = np.stack([r[1] for r in results])


# ------------------------------------------------------------------ benchmark
def box_state() -> dict:
    """Current box contention: load average + whether solo-t1-v5 is running."""
    out = {"loadavg": [round(float(x), 2) for x in os.getloadavg()],
           "solo_t1_v5": []}
    try:
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as fh:
                    cmd = fh.read().decode("utf-8", "replace").replace("\0", " ")
            except OSError:
                continue
            if "solo.train" in cmd and "t1_balance_v5" in cmd:
                out["solo_t1_v5"].append({"pid": int(pid), "cmd": cmd.strip()})
    except OSError:  # pragma: no cover - non-Linux
        pass
    return out


def benchmark(n_envs_list=(1, 2, 4), steps: int = 300, seed: int = 0,
              warmup: int = 10) -> dict:
    """Measure subproc vec throughput (and the in-process 1-env reference).

    ``steps`` is *control steps* (env steps = ``n_envs * steps``).  Box load is
    recorded before/after each measurement; this workstation is shared, so the
    numbers are only meaningful next to that record (see the report).
    """
    from dataclasses import replace

    from solo.curriculum import DEFAULT_CURRICULUM

    curriculum = replace(DEFAULT_CURRICULUM, start_steps=0, warmup_steps=1,
                         seed=seed)
    rng = np.random.default_rng(seed)
    units = rng.uniform(-1.0, 1.0, size=(max(n_envs_list), N_JOINTS))
    out = {"box_before": box_state(), "steps_per_env": int(steps),
           "warmup": int(warmup), "results": {}}

    # in-process single env (the trainer's current path), same workload
    env = SoloEnv(task="balance", seed=seed)
    env.set_push_schedule(curriculum.schedule_for(0, seed))
    env.reset(seed=seed)
    for _ in range(warmup):
        env.step(env.ctrl_from_policy(units[0]))
    st0 = box_state()
    t0 = time.perf_counter()
    ep_seed, steps_done = seed, 0
    for _ in range(steps):
        unit = units[0]
        obs, reward, terminated, truncated, info = env.step(env.ctrl_from_policy(unit))
        steps_done += 1
        if terminated or truncated:
            ep_seed += 1
            env.set_push_schedule(curriculum.schedule_for(steps_done, ep_seed))
            env.reset(seed=ep_seed)
    wall = time.perf_counter() - t0
    out["results"]["inproc_1env"] = {
        "n_envs": 1, "env_steps": steps, "wall_s": round(wall, 4),
        "env_steps_per_s": round(steps / wall, 1),
        "control_steps_per_s": round(steps / wall, 1),
        "box": {"before": st0, "after": box_state()}}
    for n in n_envs_list:
        st0 = box_state()
        vec = VecSoloEnv(n, backend="subproc", seed=seed, curriculum=curriculum)
        a = units[:n].copy()
        for _ in range(warmup):
            vec.step(a)
        t0 = time.perf_counter()
        for _ in range(steps):
            vec.step(a)
        wall = time.perf_counter() - t0
        env_steps = n * steps
        out["results"][f"subproc_n{n}"] = {
            "n_envs": n, "env_steps": env_steps, "wall_s": round(wall, 4),
            "env_steps_per_s": round(env_steps / wall, 1),
            "control_steps_per_s": round(steps / wall, 1),
            "ms_per_control_step": round(1000.0 * wall / steps, 3),
            "startup_s": round(vec.startup_s, 3),
            "box": {"before": st0, "after": box_state()}}
        vec.close()
    # info-payload cost at the largest n (parity vs speed, measured)
    n = max(n_envs_list)
    vec = VecSoloEnv(n, backend="subproc", seed=seed, curriculum=curriculum,
                     include_info=False)
    a = units[:n].copy()
    for _ in range(warmup):
        vec.step(a)
    t0 = time.perf_counter()
    for _ in range(steps):
        vec.step(a)
    wall = time.perf_counter() - t0
    out["results"][f"subproc_n{n}_no_info"] = {
        "n_envs": n, "env_steps": n * steps, "wall_s": round(wall, 4),
        "env_steps_per_s": round(n * steps / wall, 1),
        "box": {"after": box_state()}}
    vec.close()
    base = out["results"]["inproc_1env"]["env_steps_per_s"]
    for key, r in out["results"].items():
        r["ratio_vs_inproc_1env"] = round(r["env_steps_per_s"] / base, 2)
    out_ratio = out["results"].get("subproc_n4", {}).get("env_steps_per_s")
    base_ratio = out["results"].get("subproc_n1", {}).get("env_steps_per_s")
    if out_ratio and base_ratio:
        out["subproc_n4_vs_subproc_n1"] = round(out_ratio / base_ratio, 2)
    out["box_after"] = box_state()
    return out


def _single_env_hash_stream(steps: int, seed: int = 7, mode: str = "residual") -> str:
    """Self-check reference: in-process SoloEnv driven like the trainer."""
    import hashlib

    h = hashlib.sha256()
    env = SoloEnv(task="balance", seed=seed, action_mode=mode,
                  residual_scale=0.37)
    rng = np.random.default_rng(1234)
    ep_seed = seed

    def mix(*arrs):
        for arr in arrs:
            h.update(np.ascontiguousarray(arr).tobytes())

    obs = env.reset(seed=ep_seed)
    mix(env.data.qpos, obs["actor"], obs["privileged"], obs["critic"])
    for _ in range(steps):
        unit = rng.uniform(-1.0, 1.0, N_JOINTS)
        obs, reward, terminated, truncated, info = env.step(env.ctrl_from_policy(unit))
        if terminated or truncated:
            ep_seed += 1
            obs = env.reset(seed=ep_seed)
        mix(np.float64(reward), np.bool_(terminated), np.bool_(truncated),
            env.data.qpos, obs["actor"], obs["privileged"], obs["critic"])
        if terminated or truncated:
            mix(np.int64(ep_seed))
    return h.hexdigest()


def _vec_hash_stream(steps: int, seed: int = 7, mode: str = "residual") -> str:
    """Self-check: the subproc vec backend at n_envs=1, same driver."""
    import hashlib

    h = hashlib.sha256()
    vec = VecSoloEnv(1, backend="subproc", seed=seed, action_mode=mode,
                     residual_scale=0.37, include_state=True)
    rng = np.random.default_rng(1234)

    def mix(*arrs):
        for arr in arrs:
            h.update(np.ascontiguousarray(arr).tobytes())

    def mix_obs():
        mix(vec.qpos, vec.obs["actor"], vec.obs["privileged"], vec.obs["critic"])

    mix_obs()
    try:
        for _ in range(steps):
            unit = rng.uniform(-1.0, 1.0, N_JOINTS)
            batch = vec.step(unit.reshape(1, N_JOINTS))
            mix(batch.rewards, batch.terminated, batch.truncated)
            mix_obs()
            if batch.terminated[0] or batch.truncated[0]:
                mix(batch.seeds)
    finally:
        vec.close()
    return h.hexdigest()


if __name__ == "__main__":  # benchmark + equivalence self-check
    import json

    print("rl.vec_solo self-check: single-env vs subproc n=1 hash stream ...")
    a = _single_env_hash_stream(220)
    b = _vec_hash_stream(220)
    assert a == b, f"equivalence self-check FAILED: {a[:16]} != {b[:16]}"
    print(f"  OK (sha256 {a[:16]}... over 220 steps incl. episode resets)")
    res = benchmark()
    print("rl.vec_solo benchmark (box-annotated; shared 4-core host):")
    print(json.dumps(res, indent=2, sort_keys=True))
    r = res["results"]
    print(f"VERDICT: subproc n=4 / in-process n=1 = "
          f"{r['subproc_n4']['ratio_vs_inproc_1env']}x env-steps/s "
          f"({r['subproc_n4']['env_steps_per_s']} vs "
          f"{r['inproc_1env']['env_steps_per_s']}); "
          f"subproc n=4 / subproc n=1 = {res.get('subproc_n4_vs_subproc_n1')}x")
