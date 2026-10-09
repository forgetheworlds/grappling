"""``stance_terminate``: an invalid stance ends the episode like a fall.

The lit reward prices a crouch at 0.809/step against the certified stance's
1.965/step, yet with a large fall penalty the policy STILL drifted into the
crouch -- because "never fall by never standing" was an available optimum.  This
option closes it: an invalid stance held >0.5 s terminates with the same penalty
as a fall (env.py STANCE_TERMINATE_STEPS).

The positive control is the certified stance (must NOT terminate); the negative
control is the SAME stance with the root lowered 0.15 m -- torso upright (so the
fall detector stays quiet) but outside the stance predicate's pelvis band.  It
must terminate with cause "stance" at ~25 steps, and the episode must NOT
terminate when stance_terminate is off.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from solo.env import SoloEnv, STANCE_TERMINATE_STEPS  # noqa: E402
from solo.scene import load_solo_model  # noqa: E402
from solo.stance import stance_qpos  # noqa: E402


def _run(tmp_pose, terminate, steps=40):
    m = load_solo_model()
    env = SoloEnv(m, task="balance", seed=0, action_mode="residual",
                  stance_return=True, stance_terminate=terminate, jitter=False)
    env.reset(pose=tmp_pose, jitter=False)
    ctrl = np.zeros(29)                      # residual: base (stand) action only
    for k in range(steps):
        obs, r, term, trunc, info = env.step(env.ctrl_from_policy(ctrl))
        if term or trunc:
            cause = ((info.get("episode_end") or {}).get("termination"))
            return k, cause
    return None, None


@pytest.mark.parametrize("terminate", [True, False])
def test_lowered_stance_terminates_as_stance(terminate):
    m = load_solo_model()
    # a kinematically valid deep crouch (feet planted, pelvis 0.62 m): outside
    # the stance predicate's pelvis band (+-0.06 m of 0.79), torso level, so the
    # fall detector stays quiet -- the exact crouch-escape posture.
    bad = stance_qpos(height=0.62, model=m)
    k, cause = _run(bad, terminate)
    if terminate:
        assert cause == "stance", (k, cause)
        assert STANCE_TERMINATE_STEPS - 3 <= k <= STANCE_TERMINATE_STEPS + 8, (k, cause)
    else:
        assert cause != "stance", (k, cause)


def test_valid_stance_is_never_terminated():
    m = load_solo_model()
    k, cause = _run(stance_qpos(model=m), True, steps=40)
    assert cause is None, (k, cause)
