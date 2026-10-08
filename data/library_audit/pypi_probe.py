"""Query PyPI for wheel availability of the audit candidates (aarch64/CPU box).

Usage: python pypi_probe.py
Writes nothing; prints one line per package and a JSON blob at the end.
"""
from __future__ import annotations

import json
import sys
import urllib.request

NAMES = [
    "jax", "jaxlib", "mujoco-mjx", "brax", "mujoco-playground", "gymnasium",
    "stable-baselines3", "skrl", "cleanrl", "sb3-contrib",
    "mink", "pink", "pin-pink", "pin", "pinocchio", "ikpy", "qpsolvers", "daqp",
    "warp-lang", "mujoco-mpc",
]


def fetch(name: str):
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json", timeout=20) as r:
            return json.load(r)
    except Exception as e:  # noqa: BLE001
        return {"__error__": str(e)}


def wheel_tags(files) -> dict:
    tags = {"aarch64_wheels": [], "py3_none_any": [], "sdist": [], "x86_64_only": 0}
    seen_linux = 0
    for f in files:
        fn = f["filename"]
        if fn.endswith(".tar.gz"):
            tags["sdist"].append(fn)
            continue
        if "none-any.whl" in fn:
            tags["py3_none_any"].append(fn)
            continue
        if "aarch64" in fn:
            tags["aarch64_wheels"].append(fn)
            continue
        if ("x86_64" in fn or "amd64" in fn or "i686" in fn) and fn.endswith(".whl"):
            seen_linux += 1
    tags["x86_64_only"] = seen_linux
    return tags


def main() -> int:
    out = {}
    for name in NAMES:
        data = fetch(name)
        if "__error__" in data:
            print(f"{name:20s} NOT FOUND / error: {data['__error__'][:80]}")
            out[name] = {"error": data["__error__"]}
            continue
        info = data["info"]
        files = data["urls"]
        tags = wheel_tags(files)
        ver = info["version"]
        summary = (info.get("summary") or "")[:70]
        has_aarch64 = bool(tags["aarch64_wheels"])
        has_pure = bool(tags["py3_none_any"])
        print(f"{name:20s} v{ver:<12s} aarch64_wheel={has_aarch64} pure_py={has_pure} "
              f"x86wheels={tags['x86_64_only']} | {summary}")
        out[name] = {
            "version": ver, "summary": info.get("summary"),
            "requires_python": info.get("requires_python"),
            "aarch64_wheels": tags["aarch64_wheels"][:4],
            "py3_none_any": tags["py3_none_any"][:2],
            "sdist": tags["sdist"][:2],
            "x86_64_only": tags["x86_64_only"],
        }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
