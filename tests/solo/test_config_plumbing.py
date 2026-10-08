"""Config-plumbing regression test for the CLI -> SoloTrainer -> SoloEnv seam.

Two real bugs shipped through this seam in one session (see commit cdfbf13):

* **bug (1) double-map** -- ``SoloTrainer.collect`` pre-mapped the policy unit
  action with ``env.ctrl_from_unit`` (``mid + half*u``) and ``step`` then applied
  the residual contract ``base + scale*tanh(.)`` on top.  The asserts that catch
  it here are in :func:`test_residual_mapping_reaches_step_single_and_mode_aware`:
  ``ctrl_from_policy(u) == u`` in residual mode, ``step`` writes exactly
  ``base + scale*tanh(u)``, and (strongest) every action the *trainer's own*
  ``collect()`` hands to ``step`` satisfies ``max|a| <= 1`` -- the buggy
  ``ctrl_from_unit`` path emits ``mid + half*u`` with ``mid`` up to ~1.4 rad on
  the knee, so its captured actions exceed 1.0.
* **bug (2) no propagation** -- ``SoloTrainer`` never passed ``action_mode`` into
  ``SoloEnv``, so ``--action-mode residual`` trained an absolute env.  The assert
  that catches it is ``trainer.env.action_mode == cfg.action_mode`` (and
  ``env.residual_scale == cfg.residual_scale``) in
  :func:`test_env_side_config_reaches_the_live_env`.

The rest of the file pins the values on the live objects (not on a re-parsed
argparse namespace) and their survival through the checkpoint, because the same
seam produced an eval that guessed the action mapping.

Values confirmed NON-APPLICABLE / not testable, with the reason (they have no
CLI flag and no trainer plumbing at all -- so they cannot be a parsed-but-unused
flag; they are recorded in ``env.config()`` and therefore reproducible by eval):

* **episode length / horizon** -- ``SoloTrainer`` never passes ``horizon``; the
  env uses the task preset (``balance`` -> 8.0 s).  Training has no
  ``--horizon``/``--max-episode`` flag.  ``env.config()["horizon"]`` records it;
  eval can override with ``solo.eval.evaluate(max_episode_s=...)``.
* **reset noise + start pose** -- ``RESET_JOINT_NOISE`` / ``RESET_XY_NOISE`` /
  ``RESET_YAW_DEG`` / ``RESET_TILT_DEG`` and ``TaskSpec.start`` are module
  constants / preset fields.  The trainer does not pass ``jitter=``/``pose=``, so
  there is nothing to route.  ``env.config()["reset_jitter"]`` records them.
* **control dt** -- ``scene.STEP_DT``/``MODEL_DT`` are compile-time model facts;
  ``SoloEnv.__init__`` refuses a model whose timestep differs.  Recorded in
  ``env.config()["model"]``; not a config value.
* **termination thresholds** -- ``TaskSpec.terminate_on_fall/dorsal`` (booleans)
  and ``FallDetConfig`` come from the preset; the trainer passes neither.  Both
  are recorded in ``env.config()`` (``spec`` + ``fall``).
* **``--gamma``** -- reaches PPO only (``ppo_config()``); the env's reward gamma
  is the per-task ``reward.TASK_GAMMA`` (``SoloEnv(gamma=None)``), because the
  reward module deliberately owns task-level shaping gammas (rule 6).  Both
  values are on the checkpoint (``config.train.gamma`` and
  ``config.env.reward.gamma``) and eval reproduces the *reward* one, so there is
  no train/eval disagreement -- but a ``--gamma`` change does not move the
  shot/recovery potential terms.
* **trajectory reward weights** -- only ``alive`` is CLI-exposed
  (``--alive-weight``); the other ``RewardWeights`` fields are not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# The repo has no installed package / conftest; every test file puts src/ on the
# path itself (same convention as tests/test_solo.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rl.checkpoint import apply_checkpoint, load_checkpoint  # noqa: E402
from rl.net import ActorCritic, NetConfig  # noqa: E402
from solo.env import SoloEnv  # noqa: E402
from solo.obs import ACTOR_DIM, CRITIC_DIM  # noqa: E402
from solo.pushes import TRAIN_MAX_IMPULSE  # noqa: E402
from solo.scene import N_JOINTS, load_solo_model  # noqa: E402
from solo.train import SoloTrainer, TrainConfig, resolve_action_mode  # noqa: E402
import solo.train as solo_train  # noqa: E402

# Distinctive, non-default env-side values: every one differs from its default,
# so a value that never reaches the live object is detected by the asserts below.
MODE = "residual"          # default "absolute"
SCALE = 0.37               # default 0.5
ALIVE = 10.0               # default 1.0
SEED = 7                   # default 0
PUSH_START = 1_234         # default 20_000
PUSH_WARMUP = 5_678        # default 400_000
PUSH_SEED = 99             # default 0
HIDDEN = (64, 32)          # default (256, 256)


@pytest.fixture(scope="module")
def model():
    return load_solo_model()


@pytest.fixture()
def config(tmp_path):
    return TrainConfig(
        task="balance", steps=8, rollout_steps=4, seed=SEED,
        action_mode=MODE, residual_scale=SCALE, alive_weight=ALIVE,
        push_curriculum=True, push_start_steps=PUSH_START,
        push_warmup_steps=PUSH_WARMUP, push_seed=PUSH_SEED,
        entropy_coef=0.001, gamma=0.977, lam=0.9, lr=1e-4, clip=0.11,
        epochs=2, minibatches=2, value_coef=0.4, hidden=HIDDEN,
        out=str(tmp_path / "ck.pt"), log_every=0, save_every=0,
    )


@pytest.fixture()
def trainer(config, model):
    return SoloTrainer(config, model=model)


# --------------------------------------------------------------------- live env
def test_env_side_config_reaches_the_live_env(trainer, config):
    """Every env-side value must be on the live SoloEnv, not just in the cfg."""
    env = trainer.env

    # bug (2): pre-fix this was "absolute"/0.5 because the trainer called
    # SoloEnv(...) without action_mode/residual_scale.
    assert env.action_mode == config.action_mode == MODE
    assert env.residual_scale == pytest.approx(SCALE)

    # seed reaches the env, and the trainer's episode stream starts from it
    assert env.seed == SEED and env.config()["seed"] == SEED
    assert trainer.episode_seed == SEED

    # --alive-weight: it is applied as a RewardWeights override on the live
    # reward object (weights=None is used only for exactly 1.0, which is the
    # default RewardWeights, so behaviour is identical there).
    assert env.reward.weights.alive == pytest.approx(ALIVE)
    assert env.config()["reward"]["weights"]["alive"] == pytest.approx(ALIVE)

    # push curriculum: the trainer's *replaced* ramp (distinctive start/warmup/
    # seed) must be the schedule the env actually installs on reset.
    assert trainer.push_curriculum is not None
    assert trainer.push_curriculum.start_steps == PUSH_START
    assert trainer.push_curriculum.warmup_steps == PUSH_WARMUP
    assert trainer.push_curriculum.seed == PUSH_SEED
    # at steps_done=0 the ramp is not yet unlocked; advance to the distinctive
    # start and prove the env's live schedule changed (the pre-fix/default ramp
    # would still be empty here).
    trainer.steps_done = PUSH_START
    trainer._apply_push_curriculum()
    env.reset(seed=trainer.episode_seed)
    assert env.push_schedule is not None
    assert len(env.push_schedule.pushes) >= 1
    assert all(p.impulse <= TRAIN_MAX_IMPULSE for p in env.push_schedule.pushes)
    # the default ramp is provably still empty at this step count -> the assert
    # above is testing cfg.push_start_steps, not the preset default.
    from solo.curriculum import DEFAULT_CURRICULUM

    assert DEFAULT_CURRICULUM.schedule_for(PUSH_START, 0).pushes == ()

    # The knobs with no CLI flag and no trainer plumbing: pin them as recorded
    # defaults (see the module docstring for why they are not routed).
    assert env.spec.horizon == pytest.approx(8.0)
    assert env.config()["horizon"] == pytest.approx(8.0)
    assert env.jitter_reset is True
    assert env.config()["reset_jitter"] == {
        "joint_noise": 0.03, "xy_noise": 0.02, "yaw_deg": 10.0, "tilt_deg": 2.0}
    assert env.spec.start == "stand"


# ----------------------------------------------------------------- action map
def test_residual_mapping_reaches_step_single_and_mode_aware(trainer):
    """bug (1): residual actions must be mapped exactly once, in the env."""
    env = trainer.env
    u = np.linspace(-1.0, 1.0, N_JOINTS)

    # residual mode hands the policy unit to step unchanged (no pre-map) ...
    assert np.array_equal(env.ctrl_from_policy(u), u)
    # ... while the absolute map is still the ctrlrange map, and the residual
    # path must NOT be that map (bug 1 fed exactly this into step).
    assert np.allclose(env.ctrl_from_unit(u),
                       np.clip(0.5 * (env.lo + env.hi) + 0.5 * (env.hi - env.lo) * u,
                               env.lo, env.hi))
    assert not np.allclose(env.ctrl_from_policy(u), env.ctrl_from_unit(u))

    # the live step path applies exactly base + scale*tanh(u), once
    env.horizon = 1e9
    env.reset(seed=SEED)
    base = env._base_action.copy()
    env.step(env.ctrl_from_policy(u))
    expect = np.clip(base + SCALE * np.tanh(u), env.lo, env.hi)
    assert np.allclose(np.asarray(env.data.ctrl, np.float64), expect, atol=1e-12)

    # the pre-fix double map (mid + half*u fed through step) is a different env,
    # demonstrably not what step wrote
    mid = 0.5 * (env.lo + env.hi)
    half = 0.5 * (env.hi - env.lo)
    pre_fixed = np.clip(base + SCALE * np.tanh(np.clip(mid + half * u, env.lo, env.hi)),
                        env.lo, env.hi)
    assert not np.allclose(pre_fixed, expect)
    # the separation is large (tens of mrad) -- far above numerical noise, so a
    # regression to the double map cannot hide behind a loose tolerance
    assert float(np.abs(pre_fixed - expect).max()) > 0.1

    # STRONGEST: drive the trainer's own collect() and inspect what it hands the
    # env.  In residual mode the action is the clipped unit (|a| <= 1); the
    # pre-fix ctrl_from_unit path emits mid + half*u, |.| up to ~1.4 rad.
    recorded: list[np.ndarray] = []
    real_step = env.step

    def spy(action):
        recorded.append(np.asarray(action, dtype=np.float64).copy())
        return real_step(action)

    env.step = spy
    try:
        trainer.steps_done = 0
        trainer._obs = env.reset(seed=trainer.episode_seed)
        trainer.collect()
    finally:
        env.step = real_step
    assert recorded, "collect() stepped the env but the spy saw nothing"
    worst = max(float(np.abs(a).max()) for a in recorded)
    assert worst <= 1.0 + 1e-9, (
        f"trainer handed step an action of magnitude {worst} > 1 in residual "
        "mode -> the unit action was pre-mapped (bug 1)")


# ------------------------------------------------------------------ checkpoint
def test_checkpoint_round_trip_preserves_env_config(config, model):
    """Every value above must survive save/load and be resolvable for eval."""
    tr = SoloTrainer(config, model=model)
    written = tr.save()
    ckpt = load_checkpoint(written)

    # 1. eval resolves the mode and scale the policy was TRAINED under
    mode, scale = resolve_action_mode(ckpt, None)
    assert mode == MODE
    assert scale == pytest.approx(SCALE)
    # an explicit CLI override still wins; a missing field with no override fails
    assert resolve_action_mode(ckpt, "absolute")[0] == "absolute"
    with pytest.raises(ValueError):
        resolve_action_mode({"config": {}}, None)

    # 2. the raw config dicts carry every env-side value
    train = ckpt["config"]["train"]
    env_cfg = ckpt["config"]["env"]
    assert train["action_mode"] == MODE
    assert train["residual_scale"] == pytest.approx(SCALE)
    assert train["alive_weight"] == pytest.approx(ALIVE)
    assert train["seed"] == SEED
    assert train["push_start_steps"] == PUSH_START
    assert train["push_warmup_steps"] == PUSH_WARMUP
    assert train["push_seed"] == PUSH_SEED
    assert tuple(train["hidden"]) == HIDDEN
    assert env_cfg["action_mode"] == MODE
    assert env_cfg["residual_scale"] == pytest.approx(SCALE)
    assert env_cfg["seed"] == SEED
    assert env_cfg["reward"]["weights"]["alive"] == pytest.approx(ALIVE)

    # 3. rebuild the net from the SAVED hidden and load the weights: a strict
    # state_dict load only succeeds if the saved architecture matches.
    net = ActorCritic(ACTOR_DIM, CRITIC_DIM, act_dim=N_JOINTS,
                      cfg=NetConfig(hidden=tuple(train["hidden"]), action_mode=mode,
                                    residual_scale=scale))
    apply_checkpoint(ckpt, policy=net)

    # 4. rebuild the trainer from the checkpoint's own train config: the live
    # env must again hold the non-default values (an eval that does this cannot
    # silently fall back to the absolute/0.5 defaults).
    known = set(TrainConfig.__dataclass_fields__)
    cfg2 = TrainConfig(**{k: v for k, v in train.items() if k in known})
    cfg2.hidden = tuple(cfg2.hidden)
    cfg2.out = ""
    tr2 = SoloTrainer(cfg2, model=model)
    assert tr2.env.action_mode == MODE
    assert tr2.env.residual_scale == pytest.approx(SCALE)
    assert tr2.env.reward.weights.alive == pytest.approx(ALIVE)
    assert tr2.push_curriculum.start_steps == PUSH_START
    assert tr2.push_curriculum.warmup_steps == PUSH_WARMUP

    # 5. eval-side reproduction through the public env kwargs path
    env2 = SoloEnv(model=model, task="balance", seed=0,
                   action_mode=mode, residual_scale=scale)
    assert env2.action_mode == MODE
    assert env2.residual_scale == pytest.approx(SCALE)

    # 6. Regression pin for the SECOND eval-side read of the same seam: the
    # monitor (scripts/solo_env_smoke.py:376) reads ckpt["cfg"] for `hidden`, but
    # checkpoints store under "config" -- its fallback is therefore always
    # (256, 256).  Pin the schema so the mismatch is visible here.
    assert "config" in ckpt and ckpt.get("cfg") is None


# --------------------------------------------------------------- resume (bug 1)
def _capture_trainers(monkeypatch):
    """Replace SoloTrainer with a capturing, non-running subclass."""
    created: list = []
    real = solo_train.SoloTrainer

    class Spy(real):  # type: ignore[misc, valid-type]
        def __init__(self, cfg, model=None):
            super().__init__(cfg, model=model)
            created.append(self)

        def run(self):
            return {}

    monkeypatch.setattr(solo_train, "SoloTrainer", Spy)
    return created


def test_resume_uses_checkpoint_action_mode_not_cli_defaults(config, model, tmp_path,
                                                            monkeypatch):
    """bug (1): on --resume the checkpoint's saved mode is authoritative.

    Pre-fix, ``_run_train`` built the trainer from argparse defaults, so
    resuming a residual checkpoint WITHOUT --action-mode silently trained
    through the absolute mapping.  The fetch below reproduces that: with the
    checkpoint saved in residual mode, resuming with no --action-mode must
    rebuild a LIVE env in residual mode at the checkpoint's scale.
    """
    ckpt_path = SoloTrainer(config, model=model).save(str(tmp_path / "resume.pt"))
    ck = load_checkpoint(ckpt_path)
    assert resolve_action_mode(ck, None) == (MODE, pytest.approx(SCALE))

    created = _capture_trainers(monkeypatch)
    # NOTE: no --action-mode / --residual-scale on purpose (CLI default path).
    rc = solo_train.main(["--resume", str(ckpt_path), "--lock", "off",
                          "--steps", "1", "--rollout-steps", "1",
                          "--hidden", ",".join(str(h) for h in HIDDEN)])
    assert rc == 0 and created, "resume never constructed a trainer"
    tr = created[0]

    # PRE-FIX FAILURE: tr.env.action_mode == "absolute" (argparse default), so
    # this assert is the one that catches bug (1) ...
    assert tr.env.action_mode == MODE
    # ... and this one catches the scale silently snapping back to 0.5.
    assert tr.env.residual_scale == pytest.approx(SCALE)
    assert tr.cfg.action_mode == MODE and tr.cfg.residual_scale == pytest.approx(SCALE)

    # The trainer's own startup check (the printed max|ctrl-base| line) reflects
    # residual mode: with the untrained policy the first step writes ~base.  In
    # the pre-fix ABSOLUTE env ctrl_from_unit(0) writes the ctrlrange midpoint,
    # a difference of order 0.1 rad, so this threshold separates the two modes.
    assert tr.startup_ctrl_diff < 0.02, (
        f"startup max|ctrl-base|={tr.startup_ctrl_diff} rad: env is not residual")


def test_resume_cli_still_overrides_checkpoint(config, model, tmp_path, monkeypatch):
    """An explicit --action-mode on resume is an intentional override."""
    ckpt_path = SoloTrainer(config, model=model).save(str(tmp_path / "override.pt"))
    created = _capture_trainers(monkeypatch)
    rc = solo_train.main(["--resume", str(ckpt_path), "--action-mode", "absolute",
                          "--lock", "off", "--steps", "1", "--rollout-steps", "1",
                          "--hidden", ",".join(str(h) for h in HIDDEN)])
    assert rc == 0 and created
    assert created[0].env.action_mode == "absolute"


# ---------------------------------------------------- eval-side hidden (bug 2)
def test_monitor_reads_config_key_and_rebuilds_net_at_saved_width(config, model,
                                                                 tmp_path):
    """bug (2): the monitor must read ``config`` (not ``cfg``) for `hidden`.

    A non-default-hidden checkpoint must be evaluable: the saved width has to be
    recovered so apply_checkpoint's strict load matches.  Pre-fix, the monitor's
    ``ckpt.get("cfg")`` never exists, so `hidden` fell back to (256, 256) and a
    (64, 32) checkpoint raised a size-mismatch instead of evaluating.
    """
    ckpt_path = SoloTrainer(config, model=model).save(str(tmp_path / "hidden.pt"))
    ckpt = load_checkpoint(ckpt_path)

    # the FIXED read (scripts/solo_env_smoke.py:376)
    tc = (ckpt.get("config") or {}).get("train", {})
    hidden = tuple(tc.get("hidden", (256, 256)))
    assert hidden == HIDDEN

    # the PRE-FIX key never exists -> the always-default fallback that broke eval
    assert ckpt.get("cfg") is None
    assert tuple((ckpt.get("cfg") or {}).get("train", {}).get("hidden",
                                                             (256, 256))) == (256, 256)

    # rebuild at the SAVED width: strict load succeeds (checkpoint evaluates) ...
    net = ActorCritic(ACTOR_DIM, CRITIC_DIM, act_dim=N_JOINTS,
                      cfg=NetConfig(hidden=hidden, action_mode=MODE, residual_scale=SCALE))
    apply_checkpoint(ckpt, policy=net)

    # ... while the default-width net (the pre-fix fallback) size-mismatches.
    bad = ActorCritic(ACTOR_DIM, CRITIC_DIM, act_dim=N_JOINTS,
                      cfg=NetConfig(hidden=(256, 256), action_mode=MODE,
                                    residual_scale=SCALE))
    with pytest.raises(RuntimeError):
        apply_checkpoint(ckpt, policy=bad)


# ------------------------------------------------------------- new CLI flags
def test_residual_scale_flag_reaches_the_live_env(monkeypatch):
    """Acceptance (c): --residual-scale flows CLI -> cfg -> SoloTrainer -> env."""
    created = _capture_trainers(monkeypatch)
    rc = solo_train.main(["--action-mode", "residual", "--residual-scale", "0.37",
                          "--lock", "off", "--steps", "1", "--rollout-steps", "1"])
    assert rc == 0 and created
    env = created[0].env
    assert env.action_mode == "residual"
    assert env.residual_scale == pytest.approx(0.37)
    assert created[0].cfg.residual_scale == pytest.approx(0.37)


def test_push_seed_and_torch_threads_flags_reach_cfg(monkeypatch):
    """The two other previously un-plumbed TrainConfig fields now have flags."""
    created = _capture_trainers(monkeypatch)
    rc = solo_train.main(["--push-seed", "99", "--torch-threads", "2",
                          "--lock", "off", "--steps", "1", "--rollout-steps", "1"])
    assert rc == 0 and created
    tr = created[0]
    assert tr.cfg.push_seed == 99
    assert tr.cfg.torch_threads == 2
    # push_seed actually reaches the live schedule replacement
    assert tr.push_curriculum is not None and tr.push_curriculum.seed == 99
