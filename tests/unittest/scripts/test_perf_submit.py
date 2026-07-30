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
from pathlib import Path
from types import ModuleType

import pytest
import yaml
from test_common.perf_sanity_agreement import (
    expected_disagg_lifecycle_roles,
    extract_agreement_arm_log,
    format_agreement_arm_marker,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SUBMIT_PATHS = (
    REPO_ROOT / "jenkins" / "scripts" / "perf" / "submit.py",
    REPO_ROOT / "jenkins" / "scripts" / "perf" / "local" / "submit.py",
)
EXAMPLE_SUBMIT_PATH = REPO_ROOT / "examples" / "disaggregated" / "slurm" / "benchmark" / "submit.py"
DISAGG_CONFIG_DIR = REPO_ROOT / "tests" / "scripts" / "perf-sanity" / "disaggregated"
AGREEMENT_AB_CONFIG = (
    DISAGG_CONFIG_DIR
    / "gb300_deepseek-r1-fp4_128k8k_con256_ctx1_pp4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml"
)


def test_extract_agreement_arm_log_scopes_cumulative_log_to_requested_arm():
    async_lines = [
        f"{rank}: {format_agreement_arm_marker('START', 0, 'CTX_0', str(rank))}"
        for rank in range(4)
    ]
    async_lines.extend(
        (f"PYTHON_ASYNC_CONSENSUS transition=mode_active rank={rank} terminal=1 peer_ready=1")
        for rank in range(4)
    )
    async_lines.extend(
        f"{rank}: {format_agreement_arm_marker('END', 0, 'CTX_0', str(rank))}" for rank in range(4)
    )
    sync_lines = [
        f"{rank}: {format_agreement_arm_marker('START', 1, 'CTX_0', str(rank))}"
        for rank in range(4)
    ]
    sync_lines.extend(
        (f"PYTHON_ASYNC_CONSENSUS transition=mode_active rank={rank} terminal=0 peer_ready=1")
        for rank in range(4)
    )
    sync_lines.extend(
        f"{rank}: {format_agreement_arm_marker('END', 1, 'CTX_0', str(rank))}" for rank in range(4)
    )

    scoped_log = extract_agreement_arm_log(
        "\n".join((*async_lines, *sync_lines)),
        server_idx=1,
        expected_ctx_roles=4,
    )

    assert scoped_log is not None
    assert "terminal=0 peer_ready=1" in scoped_log
    assert "terminal=1 peer_ready=1" not in scoped_log


def test_extract_agreement_arm_log_waits_for_every_ctx_role():
    log_lines = [format_agreement_arm_marker("START", 0, "CTX_0", str(rank)) for rank in range(4)]
    log_lines.extend(format_agreement_arm_marker("END", 0, "CTX_0", str(rank)) for rank in range(3))

    assert (
        extract_agreement_arm_log(
            "\n".join(log_lines),
            server_idx=0,
            expected_ctx_roles=4,
        )
        is None
    )


def test_extract_agreement_arm_log_includes_late_evidence_before_next_arm():
    first_arm = [
        *(format_agreement_arm_marker("START", 0, "CTX_0", str(rank)) for rank in range(4)),
        *(format_agreement_arm_marker("END", 0, "CTX_0", str(rank)) for rank in range(4)),
        "PYTHON_CONTEXT_ACTIVATION_SEQUENCE rank=3 count=256 digest=abc123",
    ]
    second_arm = [
        *(format_agreement_arm_marker("START", 1, "CTX_0", str(rank)) for rank in range(4)),
    ]

    scoped_log = extract_agreement_arm_log(
        "\n".join((*first_arm, *second_arm)),
        server_idx=0,
        expected_ctx_roles=4,
    )

    assert scoped_log is not None
    assert "count=256 digest=abc123" in scoped_log


def test_expected_disagg_lifecycle_roles_matches_three_node_topology():
    assert expected_disagg_lifecycle_roles(
        num_ctx_servers=1,
        ctx_world_size=4,
        num_gen_servers=1,
        gen_world_size=8,
    ) == {
        "CTX_0.0",
        "CTX_0.1",
        "CTX_0.2",
        "CTX_0.3",
        "GEN_0.0",
        "GEN_0.1",
        "GEN_0.2",
        "GEN_0.3",
        "GEN_0.4",
        "GEN_0.5",
        "GEN_0.6",
        "GEN_0.7",
        "DISAGG_SERVER.0",
        "BENCHMARK.0",
    }


def test_python_consensus_ab_config_changes_only_terminal_mode():
    with open(AGREEMENT_AB_CONFIG) as config_file:
        config = yaml.safe_load(config_file)

    benchmark = config["benchmark"]
    base_env = {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in config["environment"]["ctx_worker_env_var"].split()
        if "=" in item
    }
    arms = benchmark["agreement_ab_arms"]
    effective_envs = []
    for arm in arms:
        effective_env = dict(base_env)
        effective_env.update(
            item.split("=", 1) for item in arm.get("ctx_worker_env_var", "").split() if "=" in item
        )
        effective_envs.append(effective_env)

    assert [arm["name"] for arm in arms] == ["async-terminal", "sync-terminal"]
    assert [
        env["TRTLLM_PYTHON_TRANSCEIVER_ASYNC_CTX_TERMINAL_CONSENSUS"] for env in effective_envs
    ] == ["1", "0"]
    assert [
        env["TRTLLM_PYTHON_TRANSCEIVER_ASYNC_CTX_PEER_READY_CONSENSUS"] for env in effective_envs
    ] == ["1", "1"]
    assert benchmark["multi_round"] == 1
    assert benchmark["concurrency_list"] == "256"


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
