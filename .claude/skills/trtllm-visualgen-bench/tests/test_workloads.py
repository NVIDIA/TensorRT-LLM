#!/usr/bin/env python3
"""Check every shipped workload loads and can be warmed.

  test_workloads.py [WORKLOAD ...]     no args: every yaml under ../workloads

Two invariants, both independent of which pipeline serves the workload:

* Every path it names -- `prompt_file`, a reference, `extra_params.action_file`
  -- resolves, so the loader does not fail after a file moves, and holds the
  file itself rather than the pointer git-lfs leaves when the object was never
  pulled, which the server cannot decode.
* `serve_check.py --warmup-from` yields a `compilation_config`, so step 2 can
  warm the shape step 3 requests; a shape the server was never told to warm
  compiles inside the measured latency.

How a pinned shape relates to its reference is not among them. Each pipeline
answers it differently -- Cosmos3 buckets to a 32-aligned size, Wan and LTX-2
resize to the pin, Qwen-Image-Edit fits the reference's own aspect -- so a
single rule here either misses the case it was written for or fails the others.
"""

import subprocess
import sys
from pathlib import Path

import yaml

SKILL = Path(__file__).resolve().parent.parent
WORKLOADS = SKILL / "workloads"
SERVE_CHECK = SKILL / "scripts" / "serve_check.py"
# How a git-lfs pointer starts: what a clone without `git lfs pull` holds.
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


def unusable(file: Path) -> str:
    """Why `file` cannot be a workload's input; empty when it can."""
    if not file.is_file():
        return "does not resolve"
    with file.open("rb") as handle:
        if handle.read(len(LFS_POINTER)) == LFS_POINTER:
            return "is a git-lfs pointer; run `git lfs pull`"
    return ""


def check(path: Path) -> list[str]:
    """Every way this workload is broken, as messages."""
    problems = []
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))

    common = doc.get("common_params") or {}
    for index, request in enumerate(doc.get("requests") or []):
        request = request or {}
        for key, value in request.items():
            if key == "prompt_file":
                located = [value] if isinstance(value, str) else []
            elif key.endswith("_reference"):
                # Only a path names a local file; url and base64 do not.
                located = [
                    item.get("content")
                    for item in (value if isinstance(value, list) else [value])
                    if isinstance(item, dict) and item.get("format") == "path"
                ]
            else:
                continue
            for name in located:
                if not isinstance(name, str):
                    problems.append(f"requests[{index}].{key}: {name} does not resolve")
                elif reason := unusable((path.parent / name).resolve()):
                    problems.append(f"requests[{index}].{key}: {name} {reason}")

        extra_all = {
            **(common.get("extra_params") or {}),
            **(request.get("extra_params") or {}),
        }
        located = extra_all.get("action_file")
        if isinstance(located, str) and (reason := unusable((path.parent / located).resolve())):
            problems.append(f"extra_params.action_file: {located} {reason}")

    done = subprocess.run(
        [sys.executable, str(SERVE_CHECK), "--warmup-from", str(path)],
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        problems.append(f"--warmup-from: {(done.stderr or done.stdout).strip()}")
    elif "compilation_config" not in done.stdout:
        problems.append("--warmup-from printed no compilation_config")

    return problems


def main() -> int:
    paths = [Path(arg) for arg in sys.argv[1:]] or sorted(WORKLOADS.glob("*.yaml"))
    if not paths:
        sys.exit(f"no workloads under {WORKLOADS}")

    failed = 0
    for path in paths:
        problems = check(path)
        for problem in problems:
            print(f"{path.name}: {problem}", file=sys.stderr)
        failed += bool(problems)

    print(f"{len(paths) - failed}/{len(paths)} workloads ok")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
