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
"""Opt-in startup qualification; never evict checkpoint pages on a shared host."""

import os
import sys
from pathlib import Path

import pytest
import yaml

from defs.trt_test_alternative import check_call

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BENCHMARK_DIR = _REPO_ROOT / "jenkins" / "scripts" / "startup_benchmark"
_MATRIX = _BENCHMARK_DIR / "matrix.yaml"
with _MATRIX.open(encoding="utf-8") as _matrix_file:
    _CASES = tuple(case["name"] for case in yaml.safe_load(_matrix_file)["cases"])


@pytest.mark.parametrize("case_name", _CASES)
def test_startup_benchmark(case_name: str, output_dir: str | None, tmp_path: Path) -> None:
    """Run one row and preserve its startup artifacts in the QA result directory."""
    if os.environ.get("TRTLLM_STARTUP_BENCHMARK_ENABLE") != "1":
        pytest.skip("Opt in with TRTLLM_STARTUP_BENCHMARK_ENABLE=1 on an exclusive node")
    if os.environ.get("TRTLLM_STARTUP_BENCHMARK_EXCLUSIVE_NODE") != "1":
        pytest.fail(
            "An exclusive whole-node allocation is required; set "
            "TRTLLM_STARTUP_BENCHMARK_EXCLUSIVE_NODE=1 only after reserving it",
            pytrace=False,
        )
    runtime_image = os.environ.get("TRTLLM_STARTUP_BENCHMARK_RUNTIME_IMAGE")
    if not runtime_image:
        pytest.fail(
            "Set TRTLLM_STARTUP_BENCHMARK_RUNTIME_IMAGE to the runtime image", pytrace=False
        )

    profile = os.environ.get("TRTLLM_STARTUP_BENCHMARK_PROFILE", "application_cold")
    variants = os.environ.get("TRTLLM_STARTUP_BENCHMARK_VARIANTS", "native,rank_striped")
    repeats = os.environ.get("TRTLLM_STARTUP_BENCHMARK_REPEATS", "1")
    result_root = Path(output_dir) if output_dir else tmp_path
    command = [
        sys.executable,
        str(_BENCHMARK_DIR / "runner.py"),
        "run",
        "--matrix",
        str(_MATRIX),
        "--cases",
        case_name,
        "--variants",
        variants,
        "--repeats",
        repeats,
        "--profile",
        profile,
        "--output",
        str(result_root / "startup_benchmark" / case_name / profile),
        "--exclusive-node",
        "--runtime-image",
        runtime_image,
    ]
    runtime_cache_seed = os.environ.get("TRTLLM_STARTUP_BENCHMARK_RUNTIME_CACHE_SEED")
    if runtime_cache_seed:
        command.extend(["--runtime-cache-seed", runtime_cache_seed])
    check_call(command, cwd=_REPO_ROOT)
