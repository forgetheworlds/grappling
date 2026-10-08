# Library adoption audit — learned balance/locomotion for the 29-dof G1

**Agent:** LibAudit · **Date:** 2026-10-08 · **Box:** 4-core aarch64 (Neoverse-N1), 23 GB, **no GPU**,
kernel 6.17, Python 3.12.3, MuJoCo 3.15.0, torch 2.14.1+cpu, numpy 2.5.3.
**Throwaway venv:** `/home/ubuntu/venvs/libtest` (project `.venv` untouched; its `site-packages` is
exposed read-only via `_grappling_reuse.pth` so mujoco/torch/numpy/scipy were reused, not reinstalled).
**Evidence files:** `data/library_audit/*.py|json|jsonl`, `reports/2026-10-08/library_audit.md`.

**Machine-load statement (mandatory for every number):** the box was shared the whole session with the
running v5 training service (`src.solo.train … t1_balance_v5.pt`, PID 3196830) and ~9 other agents.
Load average went from **3.9 → 7.5 → 19.8 (4 cores)** during the audit. Anything measured at load > ~8
is not a usable throughput number; where that applies it is labelled **UNTESTED-UNDER-LOAD**.

---

## 0. Decision table

| Layer | Candidate | Installs here? (aarch64) | Probe actually run | Adapter cost to our env | Verdict |
|---|---|---|---|---|---|
| RL | **CleanRL** | pip `cleanrl` → 0.4.8 (PyPI newest 1.2.0 declares `requires_python <3.11`); maintained code = repo single file | repo `ppo.py`/`ppo_continuous_action.py` read; nets 64×64, `ent_coef=0.0` default | gym-vector API; dict obs flattened (`FlattenObservation`); single-obs nets ⇒ privileged split lost; ~30–60 edited lines | **ADOPT-FOR-REFERENCE** (best PPO reference; not a library to depend on) |
| RL | **Stable-Baselines3** 2.9.0 | pip ✅ (pure py; torch reused) | 40-line gymnasium adapter; **`PPO.learn(128)` completed on our env** (8.94 env steps/s at load ~12, bounded); needed 1 extra line (drop `info["episode"]`, which SB3 reserves) | needs gymnasium.Env + spaces: ~40 lines. **No asymmetric actor/critic** (one featurizer on the whole Dict) ⇒ +60–90 lines custom policy to keep our design; defaults: `ent_coef=0.0`, net `[64,64]` | **ADOPT-FOR-REFERENCE** |
| RL | **SKRL** 2.1.0 | pip ✅ (pure py) | source-verified: `PPO(observation_space=…, state_space=…)` — policy and value can take **different** vectors; memory keeps `observations` and `states` tensors separately | gym wrapper + agent config (~40 lines) and the privileged vector goes in `state_space` — the only surveyed lib with first-class asymmetric AC | **ADOPT-FOR-REFERENCE** (the RL library to use *if* we replace the hand-rolled trainer) |
| RL | **rl_games** 1.6.5 | `pip install rl-games` did **not complete in a 180 s bound** (fallback source build of `psutil 5.9.8`, no cp312 aarch64 wheel); `import rl_games` fails | designed for IsaacGym/IsaacLab GPU pipelines; no IsaacGym on this box | n/a | **SKIP** (GPU/Isaac-centric; nothing runnable here) |
| IK | **mink** 1.3.0 | pip ✅ (aarch64 wheel for cp312 + `daqp` wheel) | whole-body IK on **our real solo model**: converged 12 iters → 0.906 mm / 0.0093°; **0.38–0.70 ms per solve** (see §3) | zero model conversion (MJCF-native) | **ADOPT** |
| IK | **pinocchio** (`pin`) 4.1.0 | pip ✅ (aarch64 wheel; brings eigenpy/coal/cmeel) | RNEA 4.5 µs, ABA 12.8–13.7 µs, CRBA 5.3–8.9 µs, frame Jacobian 1.3 µs (33-dof sample humanoid) | needs a URDF (repo is MJCF-only); no MJCF loader | **ADOPT-FOR-REFERENCE** |
| IK | **pin-pink** 4.4.0 | pip ✅ (pure py; `qpsolvers` solver must be named — `solver="daqp"`, `None` is not accepted) | pinocchio sample humanoid: `solve_ik(solver="daqp")` **12.3 ms/call at load 15.4** (idle ≈ sub-ms; the load-4 runs of mink/ikpy show the scaling), converged to **3.4e-6 m** frame error in 100 iters | needs URDF; no MJCF loader | **ADOPT-FOR-REFERENCE** (redundant with mink here) |
| IK | **ikpy** 4.1.0 | pip ✅ (pure py) | synthetic 6-dof chain: **16–38 ms**/call, residual 0.20 m; needs URDF | full URDF authoring for a robot we only have as MJCF | **SKIP** |
| Predictive | **mujoco_mpc** | **no PyPI package (HTTP 404)**; source build required | **build UNTESTED** — see §2 (load ≈18–20/4 cores; bounded-probe mandate). Static evidence: CMake fetches MuJoCo(commit 088079e)+abseil+glfw+gtest+menagerie+dm_control; CI = ubuntu-22.04/macos x86 only; README calls it a research prototype; Python API documented for Python 3.10 (box has 3.12); **no G1 task ships** (`tasks/`: humanoid, op3, walker, quadruped, …) — driving our G1 needs a new C++ task | **SKIP for adoption; ADOPT-FOR-REFERENCE** (predictive-sampling idea; usable only if we invest in a C++ task + a 3.10 env) |
| Sim | **JAX + MJX** (jax 0.11.2, mujoco-mjx 3.15.0) | pip ✅ (aarch64 `jaxlib` wheel present; mjx pure py). **Loads our real scene** after rewriting 4 `cylinder` colliders to `capsule` (MJX has no (cylinder, mesh) collision fn) | **Throughput UNTESTED-UNDER-LOAD** — see §1. First attempt killed at ~40 min (JIT compile of a batched scan); bounded retry at load 19.8: `import jax` did not finish in 50 s | env must be re-written in jax (obs/reward/termination); model needs primitive colliders (or feet-only pattern) | **NO VERDICT on throughput** — must be re-measured in an idle window; see §7 |
| Sim | **mujoco_playground** 0.2.0 | **not on PyPI**; `pip install git+https://github.com/google-deepmind/mujoco_playground.git` ✅ (module `mujoco_playground`; pulled warp-lang) | ships **G1JoystickFlatTerrain/RoughTerrain** (feet-only MJX G1 model), env exposes `privileged_state` (asymmetric), PPO configs (their budget: 200 M steps), and a **pretrained `g1_policy.onnx`**. Bounded run: timed out in first-run setup (was cloning `mujoco_menagerie`, 11 % in 240 s at load 15) ⇒ env throughput **UNTESTED-UNDER-LOAD** | it is the reference G1 locomotion stack: reward/obs/domain-randomization + policy | **ADOPT-FOR-REFERENCE** (strongest single reference for our bottleneck) |
| Sim | **brax** 0.14.2 | pip ✅ (pure py) | installs with MJX; MJX-based PPO reference | — | **ADOPT-FOR-REFERENCE** (comes with playground) |
| API | **gymnasium** 1.4.0 | pip ✅ (pure py) | our `SoloEnv` needed exactly a **40-line** adapter (spaces + reset/step shims) — measured, see §4 | 40 lines, mechanical | **ADOPT** (as the interface standard; unlocks SB3/skrl) |
| GPU | **warp-lang** 1.18.0 | pip ✅ (aarch64 wheel) but `wp.init()`: *“Could not find or load the NVIDIA CUDA driver. GPU execution will not be available”*, devices=`['cpu']` | — | — | **SKIP** on this box (GPU-only in practice; the MJX-warp backend is unusable) |

---

## 1. Q1 — JAX/MJX on aarch64: installs and loads; **throughput UNTESTED**

**Installs (all aarch64-native, no source builds):**
```
python3 -m venv /home/ubuntu/venvs/libtest
pip install jax                 # jax 0.11.2 + jaxlib 0.11.2 (manylinux_2_27_aarch64 wheel)
pip install mujoco-mjx brax gymnasium   # 3.15.0 / 0.14.2 / 1.4.0 (pure py)
pip install "git+https://github.com/google-deepmind/mujoco_playground.git"  # playground 0.2.0
```
`jax.devices()` → `[CpuDevice(id=0)]` — JAX-CPU works on this box. `mujoco_playground` is **not on
PyPI** (404); the GitHub install works.

**Our real scene loads into MJX** (`data/library_audit/solo_scene_mjx.xml`, exported via
`MjSpec.to_xml()` of the composed solo spec): `mjx.put_model` + `put_data` succeed once the four
shoulder `cylinder` collision geoms are rewritten to `capsule` (MJX collision registry 3.15 has
(capsule, mesh) but **no (cylinder, mesh)**). nq=36/nv=35/nu=29/nmocap=1, total mass identical
(33.341 kg). MJX warns for **22 meshes** “coplanar face with more than 20 vertices … may lead to
performance issues and inaccuracies in collision detection” — i.e. our vendored STL colliders are not
MJX-optimised (playground ships a `feetonly` G1 model instead).

**Throughput: UNTESTED-UNDER-LOAD.** Evidence trail:
* Attempt 1 (`probe_mjx.py`, g1.xml without floor, load 3.9): numpy baseline 1 900 control steps/s —
  but that model has no floor/contacts, so it is not comparable; the attempt died on (cylinder, mesh).
* Attempt 2/3 (`probe_mjx2.py`, real solo scene, load 4.3–7.5): model conversion OK; numpy same-model
  baseline measured **1 086.9 control steps/s** (load 4.27) and **125.6** (load 7.54) — an **8.7×
  load sensitivity**; the MJX run itself was killed by the orchestrator after ~40 min (still in JIT
  compile of the batched scan), per its instruction **no throughput conclusion is drawn**.
* Bounded retry (`probe_mjx3_bounded.py`, 150 s cap): at **load 19.8/4 cores** even `import jax`
  (jaxlib/XLA init) did not complete within 50 s (staged diagnostic: numpy 4.2 s, mujoco 9.5 s, then
  timeout). Nothing measurable.
* `probe_playground.py` (G1JoystickFlatTerrain, bounded 240 s, load ~15): **timed out during first-run
  setup** — `registry.load(...)` does **not** have the menagerie bundled; it started cloning
  `mujoco_menagerie` from GitHub and reached only **11 % in 240 s**. So no env throughput number
  (UNTESTED-UNDER-LOAD). First-run adoption cost on this box: one menagerie clone (budget ~10–40 min
  at load ~15; done once).

**Same-session numpy comparison (valid ratio, both under the stated load):**
| path | load avg | control steps/s | source |
|---|---|---|---|
| numpy+MuJoCo, physics only (10 substeps/step), same solo scene | 4.27 | **1 086.9** | probe_mjx2 run, 600 steps, 0.552 s |
| numpy+MuJoCo, same | 7.54 | 125.6 | probe_mjx2 run, 4.78 s |
| bare `SoloEnv` (physics + Python obs/reward/termination) | (repo, load 3–8.7) | **594–615** | `reports/2026-10-08/solo_env.md` §4.5 |
| `src/solo/train.py` end-to-end (PPO included) | (repo) | **175–215** (v5 run last-100 mean **264.6**) | solo_env.md; t1_midrun_read.md §2 |
| wrestling vec envs (3 subproc) | (repo) | 263–437 | rl_infra.md §5 |

**Loud statement, honestly scoped:** MJX *installs* and *loads our exact G1 scene* on this aarch64 CPU
(4-geom edit), but **the decisive throughput number does not exist yet** — every attempt either hit
the orchestrator's stop, or ran on a box at load 8–20/4 cores where JAX/XLA could not even import/JIT
in bounded time. Do **not** choose the training budget from this report. The smallest decisive test is
in §7 (idle-window measurement). If that test shows ≥2× the 594–615 steps/s bare-env rate, the
training budget changes; if it does not, MJX buys nothing here.

## 2. Q2 — mujoco_mpc

* **Not installable from PyPI** (`mujoco-mpc`, `mujoco_mpc` → HTTP 404 on pypi.org; verified).
* Source tree cloned (`github.com/google-deepmind/mujoco_mpc`, commit as of today). Build =
  CMake ≥3.16 + Ninja/clang; configure **fetches** MuJoCo (pinned `088079e`), abseil-cpp, glfw,
  googletest, `mujoco_menagerie`, `dm_control`, lodepng. The Python API is a setuptools CMake
  extension whose README requires **Python 3.10** (this box: 3.12.3); upstream CI builds only
  ubuntu-22.04 and macos-15 (both x86). README: “MJPC is not production-quality software, it is a
  **research prototype**.”
* **Can it drive a 29-dof position-servo G1?** Not out of the box: MJPC tasks are C++
  residual/transition definitions and **no G1 task ships** (`mjpc/tasks/`: acrobot, allegro, bimanual,
  cartpole, fingers, humanoid, manipulation, op3, panda, particle, quadrotor, quadruped, rubik,
  shadow_reorient, swimmer, walker). Our position servos are ordinary MuJoCo actuators, so the *model*
  side is fine, but the *task* (balance reward + termination + push handling) would have to be written
  in C++ and kept in sync with the Python model — exactly the kind of coupling we are trying to remove.
* **Build probe: UNTESTED** (deliberately not started): no wheel → full C++ dependency build on a
  4-core box already at load ≈18–20 with ~9 agent processes, under an explicit bounded-probe mandate.
  The exact command for an idle window is in §7. Verdict: **SKIP for adoption; ADOPT-FOR-REFERENCE**
  (predictive sampling is worth reading; it is a better version of what our hand-built quasi-static
  stabiliser does).

## 3. Q3 — IK / whole-body (measured on **our** G1, not a surrogate)

`data/library_audit/probe_ik.py` / `ik_probe.json` (load ~4–5 for the two usable runs):

| lib | task | result |
|---|---|---|
| **mink 1.3.0** | `mink.Configuration(load_solo_model())` (a_ prefix intact); tasks: left-foot step (+0.12 m x, +0.08 m z, orientation held), right foot planted, torso orientation, CoM over support, posture reg; `ConfigurationLimit` (joint limits) | **0.379 / 0.702 ms per `solve_ik` call** (200-call mean, daqp); **12 iterations → 0.906 mm position / 0.0093° orientation error**; whole convergence loop 5.5–8.7 ms. Signature: `solve_ik(configuration, tasks, dt, solver, damping, safety_break, limits, constraints)` |
| **pinocchio 4.1.0** | 33-dof sample humanoid (`buildSampleModelHumanoidRandom`) | RNEA 4.45–4.60 µs, ABA 12.8–13.7 µs, CRBA 5.3–8.9 µs, `computeFrameJacobian` 1.3 µs — analytics fast enough for a 1 kHz controller; **G1 needs a URDF we don't have** |
| **pin-pink 4.4.0** | pinocchio sample humanoid; `pink.solve_ik(robot, tasks, 0.01, solver="daqp", damping=1e-2)` with `PostureTask` + `FrameTask` (target +0.1 m) | **works**: converged to **3.4e-6 m** frame error in 100 iters; **12.3 ms/solve at load 15.4** (contended; the same box gave mink 29.6 ms vs 0.38 ms at load 4 ⇒ expect sub-ms idle) |
| **ikpy 4.1.0** | synthetic 6-dof chain (no URDF exists for the G1 in-repo) | 16.0–38.4 ms per `inverse_kinematics`, residual 0.20 m on my synthetic chain; pure python; URDF-only |

**Load-pollution evidence (why the numbers above are the load-4–5 runs, not the 15:07 rerun):** the same
`probe_ik.py` rerun at load ≈18 returned mink 29.6 ms (vs 0.379), RNEA 336 µs (vs 4.5), ikpy 662 ms
(vs 16) — a uniform **~75× slowdown** from contention, with identical code and model. Any per-call
timing from a contended window must be discarded; the qualitative results (convergence, API shape)
survive.

Verdicts: **mink ADOPT** (drop-in on our MJCF, sub-millisecond, replaces `src/retarget` least-squares
for single-robot tasks; keep the joint solve only if two-robot coupling is still needed),
pinocchio/pink ADOPT-FOR-REFERENCE, ikpy SKIP.

## 4. Q4 — RL adapter cost and the entropy pathology

Our contract (`src/solo/env.py`, `src/solo/obs.py`): obs dict `{actor:115, privileged:43,
critic:158}`; action = 29 unit targets (absolute: `mid+half·u`; residual: `clip(base+0.5·tanh(u))`);
`step → (obs, r, terminated, truncated, info)`; `reset(seed=…) → obs` (no info). Not gym-API mainly
because: no spaces, dict obs with an asymmetric critic split, reset/step signatures and termination
semantics are custom.

* **gymnasium adapter: 40 lines measured** (the wrapper inside `data/library_audit/probe_rl.py`:
  `spaces.Dict` of the three vectors + `spaces.Box(-1,1,(29,))`, reset/step shims, `ctrl_from_policy`).
* **SB3**: runs through that adapter. Two concrete adapter costs, both found by running it
  (`data/library_audit/probe_rl_slim.py`, `sb3_trace*.txt`):
  (1) **info-key collision** — `SoloEnv` sets `info["episode"] = int(...)`, which SB3's Monitor/
  logger reserves; SB3 crashed at its first `dump_logs` (`TypeError: object of type 'int' has no len()`,
  `on_policy_algorithm.py:290`) until the adapter popped that key (1 line). After the fix a bounded
  `PPO.learn(128)` **completed**: **8.94 env steps/s at load ~12** (bounded run, contended — a
  “it really trains our env” fact, not a throughput claim).
  (2) **no asymmetric actor/critic** — `MultiInputPolicy` feeds ONE `CombinedExtractor` from the whole
  dict into a single net, so our privileged critic is lost; preserving the design needs a custom
  `ActorCriticPolicy` (~+60–90 lines). SB3 defaults: `ent_coef=0.0`, `n_steps=2048`, `batch_size=64`,
  `n_epochs=10`, net `[64, 64]`.
* **CleanRL**: pip package is stale (0.4.8; newest PyPI 1.2.0 rejects py≥3.11) — the reference is the
  single file, so “adoption” = vendoring `ppo_continuous_action.py` and editing: env construction
  (~line 136 `DummyVecEnv`), `FlattenObservation` (dict obs collapse), the storage tensors
  (lines 193–198, single `obs` shape), and the `Agent` nets (64×64, single input for actor+critic).
  Estimate **~30–60 edited lines**; privileged critic lost unless nets are edited. Current master
  defaults: `ent_coef=0.0`, 64×64 tanh.
* **SKRL**: the only surveyed library with **native asymmetric actor/critic** —
  `PPO(models=…, observation_space=<actor>, state_space=<critic>)`, and its memory stores
  `observations` and `states` as separate tensors. **Run (bounded, load ~14):** a PPO agent was built
  and `init()`-ed with **our exact spaces** — policy input **115** (actor), value input **158**
  (actor+privileged), 13 498 / 14 401 parameters, default `entropy_loss_scale=0.0`
  (`data/library_audit/probe_rl_slim.py`). Adapter = gym wrapper (~40 lines) + config. Caveat: `skrl`
  imports the deprecated `gym` 0.26 (warning under numpy 2.5) — that package arrived with the
  `cleanrl` pip install.
* **Entropy pathology (answer):** *all three* libraries' **defaults would trivially avoid the
  entropy-dominance** we hit — their default entropy coefficient is **0.0** (SB3 `ent_coef=0.0`;
  CleanRL master `ent_coef=0.0`; SKRL `entropy_loss_scale=0.0`) whereas our run used **0.01** with a
  near-constant balance reward. But this is not a library advantage: with 0.0 the exploration pressure
  is gone too, and the *actual* T1 failure (policy collapsing into a non-terminating limb-supported
  pose, `mean_upright=0.0415` at 1.1 M steps) is a **reward-design** problem — already addressed
  in-repo by `--alive-weight 10` + `entropy_coef 0.001` (v5, running). No RL library fixes that; their
  smaller 64×64 default nets are also not the cause (our nets are 2×256, ~97 k/131 k params).

## 5. Q5 — GPU-only / not usable on this box

| item | evidence |
|---|---|
| **warp-lang 1.18.0** | aarch64 wheel installs, but `wp.init()` → *“Could not find or load the NVIDIA CUDA driver. GPU execution will not be available”*; `wp.get_devices() == ['cpu']`. Practically GPU-only for our purposes. |
| **MJX warp backend** (`mujoco_warp`) | same dependency chain: `mujoco_mjx` prints “Failed to import warp” without it; with warp installed there is still no CUDA driver. XLA-CPU is the only MJX backend on this box. |
| **rl_games** | pip wheel installs, but it is built around IsaacGym/IsaacLab (GPU, NVIDIA stack not present here) — nothing to run. |

## 6. Process note (requested)

The MJX probe was **not wall-clock bounded from the start**: the first full attempt (batched JIT
compile of a 400-control-step, 256-env scan on a loaded 4-core ARM box) ran ~40 minutes before the
orchestrator stopped it, and the bounded retries then hit a load of 19.8/4 cores where `import jax`
alone exceeded 50 s. Lesson for the next session: **every probe gets an explicit wall-clock budget
(suggested ≤150 s) and reports “steps completed / steps·s⁻¹ within the window”; kill on timeout.**
A probe result on a box at load > ~2× cores is untrustworthy for throughput and must be tagged
UNTESTED-UNDER-LOAD. (This was the orchestrator's call, recorded here as asked.)

## 7. Recommendation and smallest proof experiment

**Adopt first for the current bottleneck (learned balance/locomotion): `mujoco_playground` — the
MJX-G1 reference stack (with `pip install jax mujoco-mjx` + playground from GitHub).** Rationale: it
is the only candidate that attacks the bottleneck *itself* (it ships a working G1 locomotion env with
an asymmetric `privileged_state`, domain randomisation, PPO configs, and a **pretrained `g1_policy.onnx`**),
and it is the same stack whose throughput question (MJX-CPU) is still open. It can be adopted at zero
risk today as a **reference** (reward/obs design), and conditionally as the **compute backend**.
Secondary, unconditional adoption (proven today, for the IK layer only): **mink** — 0.38–0.70 ms/solve
on our real model, 12 iterations to sub-millimetre foot placement.

**Smallest experiment that would prove the adoption pays off (one idle-window session, ≤30 min):**
0. One-time setup: let `registry.load("G1JoystickFlatTerrain")` finish its `mujoco_menagerie` clone
   (~10–40 min at load 15; it must not be inside the 240 s probe bound).
1. In an idle window (no training service; load < 4), run `data/library_audit/probe_mjx2.py`-style
   bounded probes: playground `G1JoystickFlatTerrain` under MJX on CPU, batch 64 (and 256 if <2 min),
   fixed 100-step window, recording load average. Compare to our same-session numpy baseline.
2. In the same window, instantiate playground's G1 env with the shipped **`g1_policy.onnx`** policy
   and roll out ~10 s of sim (≈500 steps) with a constant joystick command; acceptance =
   it stands/tracks without falling.
3. Pass criteria: **(a)** MJX env-steps/s ≥ 2× our 594–615 bare-env rate (i.e. ≥ ~1 200/s) **and**
   **(b)** the pretrained policy balances on CPU here. If both pass → port the T1 balance task into the
   playground MJX env (feet-only collider pattern) and re-run the v5 command for the same wall clock;
   if either fails → keep the numpy env, adopt only playground's reward/obs design (and mink for IK).

**Exact commands for that window** (also for the MPC build, which was not attempted):
```bash
# MJX/playground throughput (bounded)
timeout 150 <libtest>/bin/python data/library_audit/probe_mjx2.py --tag idle-window
timeout 240 <libtest>/bin/python data/library_audit/probe_playground.py --batch 64 --steps 100
# mujoco_mpc build attempt (NOT attempted today; ~30–90 min at load 1)
sudo apt-get install -y libgl1-mesa-dev libxinerama-dev libxcursor-dev libxrandr-dev \
    libxi-dev ninja-build zlib1g-dev
cd /tmp/mujoco_mpc && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DMJPC_BUILD_GRPC_SERVICE=OFF \
    -DMJPC_BUILD_TESTS=OFF -DPYMJPC_BUILD_TESTS=OFF -G Ninja
cmake --build . -j4            # needs Python 3.10 for the python bindings afterwards
```

## 8. Explicitly NOT tested (with reasons — do not read these as pass/fail)

| item | status | reason |
|---|---|---|
| MJX-CPU steps/s for our G1 scene | **UNTESTED-UNDER-LOAD** | first attempt killed at ~40 min (JIT compile); bounded retries stalled before/inside `import jax` on a box at load 15–20/4 cores |
| playground `G1JoystickFlatTerrain` steps/s on CPU | **UNTESTED-UNDER-LOAD** | bounded 240 s attempt timed out in first-run setup: it was still cloning `mujoco_menagerie` (11 % in 240 s, box at load ~15) |
| running the pretrained `g1_policy.onnx` | **UNTESTED** | needs the MJX/playground data pipeline up, which the box could not provide in a bounded window today |
| mujoco_mpc build | **UNTESTED** | no wheel; CMake build fetches MuJoCo+abseil+glfw+dm_control+menagerie; box at load ≈18–20 under a bounded-probe mandate — not attempted |
| any *learning* result from a new library | **none claimed** | the completed SB3/skrl runs are 128-step/setup smoke runs, not learning results; the report only claims runtime/adapter facts |
| rl_games install | incomplete in a 180 s bound | source build of the `psutil<6` dependency (no cp312 aarch64 wheel); IsaacGym absent anyway |
