# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify and install the exact local OpenEngine Python sibling package."""

import argparse
import re
import runpy
import subprocess
import sys
from pathlib import Path


def _run(*args: str, cwd: Path) -> str:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sibling",
        type=Path,
        help="OpenEngine checkout (default: ../openengine-trtllm)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify the sibling without invoking pip",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sibling = (args.sibling or root.parent / "openengine-trtllm").resolve()
    expected = (root / "OPENENGINE_COMMIT").read_text(encoding="utf-8").strip()
    if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
        raise RuntimeError("OPENENGINE_COMMIT must contain one full lowercase Git SHA")
    packaged = runpy.run_path(str(root / "tensorrt_llm" / "openengine" / "_schema_pin.py"))[
        "OPENENGINE_COMMIT"
    ]
    if packaged != expected:
        raise RuntimeError(
            f"Packaged OpenEngine pin is {packaged}, but OPENENGINE_COMMIT contains {expected}"
        )

    actual = _run("git", "rev-parse", "HEAD", cwd=sibling)
    if actual != expected:
        raise RuntimeError(f"OpenEngine sibling is at {actual}, but TensorRT-LLM pins {expected}")
    dirty = _run(
        "git",
        "status",
        "--porcelain",
        "--",
        "packages/python",
        "proto",
        cwd=sibling,
    )
    if dirty:
        raise RuntimeError("OpenEngine Python/proto sources have uncommitted changes")

    package = sibling / "packages" / "python"
    if not (package / "pyproject.toml").is_file():
        raise RuntimeError(f"OpenEngine Python package is missing: {package}")
    if not args.verify_only:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-e",
                str(package),
                "-r",
                str(root / "requirements-openengine.txt"),
            ],
            cwd=root,
            check=True,
        )

    print(f"Verified OpenEngine {expected}")
    print(f"export OPENENGINE_SCHEMA_RELEASE={expected}")


if __name__ == "__main__":
    main()
