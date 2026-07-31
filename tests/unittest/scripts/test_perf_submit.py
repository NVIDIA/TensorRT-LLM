#!/usr/bin/env python3
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

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SUBMIT_PATHS = (
    REPO_ROOT / "jenkins" / "scripts" / "perf" / "submit.py",
    REPO_ROOT / "jenkins" / "scripts" / "perf" / "local" / "submit.py",
)
EXAMPLE_SUBMIT_PATH = REPO_ROOT / "examples" / "disaggregated" / "slurm" / "benchmark" / "submit.py"


def _load_module(path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location(f"perf_submit_{path.parent.name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=SUBMIT_PATHS, ids=("ci", "local"))
def submit_module(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _load_module(request.param, monkeypatch)


@pytest.fixture
def example_submit_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _load_module(EXAMPLE_SUBMIT_PATH, monkeypatch)


def _get_benchmark_config(module: ModuleType, concurrency):
    config = {"benchmark": {"concurrency_list": concurrency}}
    if Path(module.__file__).parent.name == "local":
        return module.get_benchmark_config(config, "gen_only")
    return module.get_benchmark_config(config)


@pytest.mark.parametrize("concurrency", ("1", 1, "4301"))
def test_get_benchmark_config_accepts_positive_integer(submit_module: ModuleType, concurrency):
    benchmark_config = _get_benchmark_config(submit_module, concurrency)

    assert benchmark_config["concurrency"] == int(concurrency)


@pytest.fixture
def ci_submit_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _load_module(SUBMIT_PATHS[0], monkeypatch)


@pytest.mark.parametrize(
    "concurrency",
    (True, 1.5, [], {}, "0", 0, "-1", -1, "1.5", "not-an-integer", None),
)
def test_get_benchmark_config_rejects_invalid_concurrency(submit_module: ModuleType, concurrency):
    with pytest.raises(ValueError, match="benchmark.concurrency_list must be a positive integer"):
        _get_benchmark_config(submit_module, concurrency)


@pytest.mark.parametrize(
    "concurrency",
    (True, 1.5, [], {}, "0", 0, "-1", -1, "1.5", "not-an-integer", None),
)
def test_example_worker_environment_rejects_invalid_concurrency(
    example_submit_module: ModuleType, concurrency
):
    with pytest.raises(ValueError, match="benchmark.concurrency_list must be a positive integer"):
        example_submit_module.build_worker_environment(
            worker_config={},
            env_config={},
            role="GEN",
            benchmark_mode="gen_only",
            nsys_on=False,
            profile_range="",
            concurrency=concurrency,
        )


def test_example_worker_environment_exports_positive_concurrency(example_submit_module: ModuleType):
    worker_environment = example_submit_module.build_worker_environment(
        worker_config={},
        env_config={},
        role="GEN",
        benchmark_mode="gen_only",
        nsys_on=False,
        profile_range="",
        concurrency="4301",
    )

    assert worker_environment["TLLM_BENCHMARK_REQ_QUEUES_SIZE"] == "4301"


def test_ci_submit_selects_same_least_duration_shard_as_pytest_split(ci_submit_module, tmp_path):
    test_lines = [
        "perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-gb300_deepseek-r1] TIMEOUT (90)",
        "perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-gb300_kimi-k25] TIMEOUT (90)",
        "perf/test_perf_sanity.py::test_e2e[disagg_upload-e2e-gb300_deepseek-r1] TIMEOUT (90)",
        "perf/test_perf_sanity.py::test_e2e[disagg_upload-e2e-gb300_kimi-k25] TIMEOUT (90)",
    ]
    test_list_path = tmp_path / "test_list.txt"
    test_list_path.write_text("\n".join(test_lines), encoding="utf-8")

    durations_dir = tmp_path / "tests" / "integration" / "defs"
    durations_dir.mkdir(parents=True)
    durations = {
        ci_submit_module._test_nodeid(test_lines[0]): 836.268,
        ci_submit_module._test_nodeid(test_lines[1]): 1462.754,
        ci_submit_module._test_nodeid(test_lines[2]): 2211.1548,
        ci_submit_module._test_nodeid(test_lines[3]): 2548.912,
    }
    (durations_dir / ".test_durations").write_text(json.dumps(durations), encoding="utf-8")
    script_prefix_lines = [
        'export pytestCommand="pytest --splitting-algorithm least_duration '
        "--splits 4 --group 3 "
        '--durations-path /remote/tests/integration/defs/.test_durations"'
    ]

    selected = ci_submit_module.select_test_case_line(
        test_list_path,
        tmp_path,
        script_prefix_lines,
        split_group=3,
    )

    assert selected == test_lines[1]


def test_ci_submit_rejects_split_group_disagreement(ci_submit_module, tmp_path):
    test_list_path = tmp_path / "test_list.txt"
    test_list_path.write_text(
        "perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-gb300-kimi]\n",
        encoding="utf-8",
    )
    script_prefix_lines = [
        'export pytestCommand="pytest --splitting-algorithm least_duration --splits 1 --group 1"'
    ]

    with pytest.raises(ValueError, match="disagrees with pytest --group"):
        ci_submit_module.select_test_case_line(
            test_list_path,
            tmp_path,
            script_prefix_lines,
            split_group=2,
        )
