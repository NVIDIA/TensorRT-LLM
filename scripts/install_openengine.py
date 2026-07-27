# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate and install Python bindings from the pinned OpenEngine schema."""

import argparse
import importlib.resources
import re
import runpy
import shutil
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
        help="OpenEngine source checkout (default: ../openengine-trtllm)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify the sibling without invoking pip",
    )
    parser.add_argument(
        "--source-identity",
        help="Pinned identity for a source export without Git metadata",
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

    if (sibling / ".git").exists():
        actual = _run("git", "rev-parse", "HEAD", cwd=sibling)
        if actual != expected:
            raise RuntimeError(
                f"OpenEngine sibling is at {actual}, but TensorRT-LLM pins {expected}"
            )
        dirty = _run("git", "status", "--porcelain", "--", "proto", cwd=sibling)
        if dirty:
            raise RuntimeError("OpenEngine proto sources have uncommitted changes")
    elif args.source_identity != expected:
        raise RuntimeError(
            f"A source export without Git metadata requires --source-identity {expected}"
        )

    proto_root = sibling / "proto"
    proto_files = sorted((proto_root / "openengine" / "v1").glob("*.proto"))
    if not proto_files:
        raise RuntimeError(f"OpenEngine proto sources are missing: {proto_root}")
    if not args.verify_only:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(root / "requirements-openengine.txt"),
            ],
            cwd=root,
            check=True,
        )
        generated = root / "build" / "openengine-python"
        shutil.rmtree(generated, ignore_errors=True)
        generated.mkdir(parents=True)
        grpc_tools_include = importlib.resources.files("grpc_tools").joinpath("_proto")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                f"-I{proto_root}",
                f"-I{grpc_tools_include}",
                f"--python_out={generated}",
                f"--grpc_python_out={generated}",
                *(str(path) for path in proto_files),
            ],
            cwd=root,
            check=True,
        )
        for package in (generated / "openengine", generated / "openengine" / "v1"):
            package.mkdir(parents=True, exist_ok=True)
            (package / "__init__.py").touch()
        (generated / "pyproject.toml").write_text(
            """\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "tensorrt-llm-openengine-bindings"
version = "0.0.0"

[tool.setuptools.packages.find]
where = ["."]
include = ["openengine*"]
""",
            encoding="utf-8",
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                "-e",
                str(generated),
            ],
            cwd=root,
            check=True,
        )

    print(f"Verified OpenEngine {expected}")
    print(f"export OPENENGINE_SCHEMA_RELEASE={expected}")


if __name__ == "__main__":
    main()
