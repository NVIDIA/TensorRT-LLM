# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import difflib
import json
import os
import stat
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MANIFEST_PATH = _REPO_ROOT / "tensorrt_llm/usage/llm_args_golden_manifest.json"


def golden_manifest():
    original_path = sys.path.copy()
    repo_root = str(_REPO_ROOT)
    try:
        sys.path[:] = [entry for entry in sys.path if entry != repo_root]
        sys.path.insert(0, repo_root)
        from tensorrt_llm.usage.llmapi_config import golden_manifest as build_golden_manifest

        return build_golden_manifest()
    finally:
        sys.path[:] = original_path


def _render_manifest() -> str:
    return json.dumps(golden_manifest(), indent=2, sort_keys=True) + "\n"


def _read_manifest(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", newline="") as manifest_file:
            return manifest_file.read()
    except FileNotFoundError:
        return ""


def _write_manifest(path: Path = _DEFAULT_MANIFEST_PATH, *, generated: str | None = None) -> bool:
    if generated is None:
        generated = _render_manifest()
    if _read_manifest(path) == generated:
        return False

    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        mode = 0o644

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as temporary_file:
            temporary_file.write(generated)
        temporary_path.chmod(mode)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return True


def _check_manifest(path: Path = _DEFAULT_MANIFEST_PATH, *, generated: str | None = None) -> bool:
    committed = _read_manifest(path)
    if generated is None:
        generated = _render_manifest()
    if committed == generated:
        return True

    diff = difflib.unified_diff(
        committed.splitlines(keepends=True),
        generated.splitlines(keepends=True),
        fromfile=f"{path} (committed)",
        tofile=f"{path} (generated)",
    )
    sys.stderr.writelines(diff)
    return False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or check the committed LLM-args telemetry manifest."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check the committed manifest without modifying it.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    generated = _render_manifest()
    try:
        if args.check:
            return 0 if _check_manifest(generated=generated) else 1
        changed = _write_manifest(generated=generated)
    except OSError as error:
        print(f"Failed to access {_DEFAULT_MANIFEST_PATH}: {error}", file=sys.stderr)
        return 2

    if changed:
        print(f"Updated {_DEFAULT_MANIFEST_PATH.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
