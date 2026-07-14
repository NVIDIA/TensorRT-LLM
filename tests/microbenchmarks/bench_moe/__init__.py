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

"""MoE microbenchmark package entrypoint.

The implementation lives in focused submodules. This package mirrors the
historical ``bench_moe.*`` import surface by re-exporting every non-dunder
attribute, including underscore-prefixed helpers used by tests and scripts.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_MICROBENCH_DIR = Path(__file__).resolve().parent.parent
_TESTS_UNITTEST_DIR = _MICROBENCH_DIR.parent / "unittest"
if str(_TESTS_UNITTEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_UNITTEST_DIR))

_REPO_ROOT = _MICROBENCH_DIR.parent.parent
if (_REPO_ROOT / "tensorrt_llm" / "bindings").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_PACKAGE_PARENT = Path(__file__).resolve().parent.parent
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))


def _mirror_module(module_name: str) -> None:
    module = importlib.import_module(module_name)
    for name, value in module.__dict__.items():
        if not (name.startswith("__") and name.endswith("__")):
            globals()[name] = value


for _module_name in (
    "bench_moe.backend",
    "bench_moe.quantize",
    "bench_moe.specs",
    "bench_moe.search",
    "bench_moe.utils",
    "bench_moe.mapping",
    "bench_moe.build",
    "bench_moe.routing",
    "bench_moe.timing",
    "bench_moe.results",
    "bench_moe.cli",
    "bench_moe.case_runner",
    "bench_moe.worker",
):
    _mirror_module(_module_name)

del _module_name
