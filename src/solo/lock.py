"""Advisory cross-process simulation lock (``data/locks/sim.lock``).

Main's resource directive (2026-10-08): several agents run MuJoCo on the same
4-core box, so any run longer than ~60 s must hold this lock; if it is held,
wait (or report) instead of racing.  Advisory only -- every participant must
opt in.

Usage::

    from solo.lock import SimLock

    with SimLock(owner="solo baselines", wait_s=600):
        ...run sims...
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO / "data" / "locks" / "sim.lock"


class SimLockTimeout(RuntimeError):
    """Raised when the advisory sim lock is held for longer than ``wait_s``."""


@dataclass
class SimLock:
    """Advisory exclusive lock around long simulation runs."""

    owner: str = "unknown"
    wait_s: float = 600.0
    poll_s: float = 2.0
    path: Path = LOCK_PATH
    verbose: bool = True

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._fh = None

    def _holder_info(self) -> str:
        try:
            return self.path.read_text().strip()
        except OSError:
            return "unknown"

    def acquire(self) -> "SimLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + float(self.wait_s)
        while True:
            fh = open(self.path, "a+")
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                fh.close()
                if time.monotonic() >= deadline:
                    raise SimLockTimeout(
                        f"sim lock {self.path} held by {self._holder_info()!r} "
                        f"longer than {self.wait_s:.0f}s")
                if self.verbose:
                    print(f"[simlock] waiting for {self._holder_info()!r} ...")
                time.sleep(self.poll_s)
                continue
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps({"owner": self.owner, "pid": os.getpid(),
                                 "t": time.time()}) + "\n")
            fh.flush()
            self._fh = fh
            if self.verbose:
                print(f"[simlock] acquired by {self.owner!r}")
            return self

    def release(self) -> None:
        if self._fh is not None:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            finally:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> "SimLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


if __name__ == "__main__":  # self-check
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "sim.lock"
        a = SimLock(owner="self-check-a", wait_s=1.0, poll_s=0.05, path=p, verbose=False)
        b = SimLock(owner="self-check-b", wait_s=0.3, poll_s=0.05, path=p, verbose=False)
        a.acquire()
        try:
            b.acquire()
            raise AssertionError("second acquire must not succeed while held")
        except SimLockTimeout:
            pass
        finally:
            a.release()
        b.acquire()
        b.release()
    print("solo.lock self-check OK:", {"lock": str(LOCK_PATH)})
