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
"""Regenerate, or check, the MoE implementation catalog in the MoE perf guide.

Needs a TensorRT-LLM build matching the checkout, since the catalog is read
from the implementation registry.
"""

import argparse
import difflib
import sys
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GUIDE_PATH = _REPO_ROOT / "docs/source/developer-guide/moe-perf-guide.md"


def _generated() -> dict[Path, str]:
    """What each generated file should hold, by path."""
    original_path = sys.path.copy()
    repo_root = str(_REPO_ROOT)
    try:
        sys.path[:] = [entry for entry in sys.path if entry != repo_root]
        sys.path.insert(0, repo_root)
        from tensorrt_llm._torch.moe.fused_moe.impl_catalog import splice_moe_impl_catalog

        return {
            _GUIDE_PATH: splice_moe_impl_catalog(_GUIDE_PATH.read_text(encoding="utf-8")),
        }
    finally:
        sys.path[:] = original_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or check the MoE implementation catalog in the MoE perf guide."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check the committed files without modifying them.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    stale = False
    for path, generated in _generated().items():
        committed = path.read_text(encoding="utf-8") if path.is_file() else ""
        if generated == committed:
            continue
        if args.check:
            sys.stderr.writelines(
                difflib.unified_diff(
                    committed.splitlines(keepends=True),
                    generated.splitlines(keepends=True),
                    fromfile=f"{path} (committed)",
                    tofile=f"{path} (generated)",
                )
            )
            stale = True
            continue
        path.write_text(generated, encoding="utf-8")
        print(f"Updated {path.relative_to(_REPO_ROOT)}")
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
