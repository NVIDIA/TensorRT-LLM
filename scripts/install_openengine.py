# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate and install Python bindings from the pinned OpenEngine schema."""

import argparse
import importlib.resources
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_BSR_MODULE_PATTERN = re.compile(r"buf\.build/openengine/openengine:([0-9a-f]{32})")


def _run(*args: str, cwd: Path) -> str:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _bsr_commit(module: str) -> str:
    match = _BSR_MODULE_PATTERN.fullmatch(module)
    if match is None:
        raise RuntimeError(
            "OpenEngine BSR module must include an exact 32-character lowercase "
            "hex commit: buf.build/openengine/openengine:<commit>"
        )
    return match.group(1)


def _proto_root(source: Path) -> Path:
    nested = source / "proto"
    if (nested / "openengine" / "v1" / "openengine.proto").is_file():
        return nested
    return source


def _local_source(source: Path, expected: str, source_identity: str | None) -> tuple[Path, str]:
    if (source / ".git").exists():
        actual = _run("git", "rev-parse", "HEAD", cwd=source)
        if actual != expected:
            raise RuntimeError(
                f"OpenEngine sibling is at {actual}, but TensorRT-LLM pins {expected}"
            )
        dirty = _run("git", "status", "--porcelain", "--", "proto", cwd=source)
        if dirty:
            raise RuntimeError("OpenEngine proto sources have uncommitted changes")
        return _proto_root(source), expected
    if source_identity is None:
        raise RuntimeError(
            "A source export without Git metadata requires --source-identity "
            "with its exact 32-character BSR commit"
        )
    if re.fullmatch(r"[0-9a-f]{32}", source_identity) is None:
        raise RuntimeError("--source-identity must be an exact 32-character BSR commit")
    return _proto_root(source), source_identity


def _export_bsr(module: str, output: Path) -> tuple[Path, str]:
    identity = _bsr_commit(module)
    if shutil.which("buf") is None:
        raise RuntimeError("buf is required to export an OpenEngine BSR module")
    subprocess.run(["buf", "export", module, "--output", str(output)], check=True)
    return _proto_root(output), identity


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
        help="Exact 32-character BSR commit for an exported source tree without Git metadata",
    )
    parser.add_argument(
        "--buf-module",
        help=("Immutable BSR module input; may also be supplied through OPENENGINE_BSR_MODULE"),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
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

    module = args.buf_module or os.getenv("OPENENGINE_BSR_MODULE")
    if module and args.sibling:
        raise RuntimeError("Use exactly one of --sibling and --buf-module")

    with tempfile.TemporaryDirectory(prefix="trtllm-openengine-") as temporary:
        if module:
            proto_root, schema_release = _export_bsr(module, Path(temporary))
        else:
            sibling = (args.sibling or root.parent / "openengine-trtllm").resolve()
            proto_root, schema_release = _local_source(sibling, expected, args.source_identity)
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
            (generated / "openengine" / "_schema_identity.py").write_text(
                '"""Generated OpenEngine schema identity."""\n\n'
                f"SCHEMA_RELEASE = {schema_release!r}\n",
                encoding="utf-8",
            )
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

    print(f"Verified OpenEngine {schema_release}")
    print(f"export OPENENGINE_SCHEMA_RELEASE={schema_release}")


if __name__ == "__main__":
    main()
