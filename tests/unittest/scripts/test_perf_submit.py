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
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SUBMIT_PATHS = (
    REPO_ROOT / "jenkins" / "scripts" / "perf" / "submit.py",
    REPO_ROOT / "jenkins" / "scripts" / "perf" / "local" / "submit.py",
)
DISAGG_CONFIG_DIR = REPO_ROOT / "tests" / "scripts" / "perf-sanity" / "disaggregated"


@pytest.fixture(params=SUBMIT_PATHS, ids=("ci", "local"))
def submit_module(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(request.param.parent))
    spec = importlib.util.spec_from_file_location(
        f"perf_submit_{request.param.parent.name}", request.param
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ci_submit_module():
    spec = importlib.util.spec_from_file_location("perf_submit_ci", SUBMIT_PATHS[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("config_name", "expected_queue_size"),
    (
        ("gb300_deepseek-v4-pro-fp4_8k1k_con8_ctx1_dep4_gen4_tep8_eplb0_mtp3_ccb-NIXL", 1),
        (
            "gb300_deepseek-v4-pro-fp4_8k1k_con180_ctx3_dep4_gen1_dep32_eplb384_mtp3_ccb-NIXL",
            128,
        ),
        (
            "gb300_deepseek-v4-pro-fp4_8k1k_con666_ctx6_dep4_gen1_dep16_eplb384_mtp3_ccb-NIXL",
            512,
        ),
        (
            "gb300_deepseek-v4-pro-fp4_8k1k_con4301_ctx12_dep4_gen1_dep8_eplb384_mtp1_ccb-NIXL",
            4096,
        ),
    ),
)
def test_gen_only_queue_size_does_not_exceed_executor_capacity(
    submit_module, config_name, expected_queue_size
):
    with open(DISAGG_CONFIG_DIR / f"{config_name}.yaml") as config_file:
        config = yaml.safe_load(config_file)

    concurrency = config["benchmark"]["concurrency_list"]

    assert (
        submit_module.get_benchmark_request_queue_size(config, concurrency) == expected_queue_size
    )


def test_gen_only_queue_size_preserves_reachable_concurrency(submit_module):
    config = {
        "worker_config": {
            "gen": {
                "max_batch_size": 8,
                "tensor_parallel_size": 4,
                "enable_attention_dp": True,
            }
        }
    }

    assert submit_module.get_benchmark_request_queue_size(config, 16) == 16


def test_kv_transfer_trace_common_vars_cover_all_external_ranks(ci_submit_module):
    trace_vars = ci_submit_module.get_kv_transfer_trace_common_vars(
        "TLLM_ENABLE_KV_TRANSFER_TRACE=1 CTX_ONLY=1",
        "TLLM_ENABLE_KV_TRANSFER_TRACE=1 GEN_ONLY=1",
        "/tmp/results/disagg-kimi",
        {
            "num_ctx_servers": 1,
            "gpus_per_ctx_server": 4,
            "num_gen_servers": 1,
            "gpus_per_gen_server": 16,
        },
    )

    assert (
        "TLLM_KV_TRANSFER_TRACE_OUTPUT_DIR=/tmp/results/disagg-kimi/kv_transfer_traces"
    ) in trace_vars
    assert "TLLM_KV_TRANSFER_TRACE_REQUIRED=1" in trace_vars
    assert "TLLM_KV_TRANSFER_TRACE_EXPECTED_CTX_FILES=4" in trace_vars
    assert "TLLM_KV_TRANSFER_TRACE_EXPECTED_GEN_FILES=16" in trace_vars


def test_kv_transfer_trace_common_vars_disabled_by_default(ci_submit_module):
    assert (
        ci_submit_module.get_kv_transfer_trace_common_vars(
            "CTX_ONLY=1",
            "GEN_ONLY=1",
            "/tmp/results/disagg-kimi",
            {
                "num_ctx_servers": 1,
                "gpus_per_ctx_server": 4,
                "num_gen_servers": 1,
                "gpus_per_gen_server": 16,
            },
        )
        == ""
    )


def test_ci_submit_selects_same_least_duration_shard_as_pytest_split(ci_submit_module, tmp_path):
    test_lines = [
        "perf/test_perf_sanity.py::test_e2e[disagg_upload-e2e-gb300_deepseek-r1] TIMEOUT (90)",
        "perf/test_perf_sanity.py::test_e2e[disagg_upload-e2e-gb300_kimi-k25] TIMEOUT (90)",
        "perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-gb300_deepseek-r1] TIMEOUT (90)",
        "perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-gb300_kimi-k25] TIMEOUT (90)",
    ]
    test_list_path = tmp_path / "test_list.txt"
    test_list_path.write_text("\n".join(test_lines), encoding="utf-8")

    durations_dir = tmp_path / "tests" / "integration" / "defs"
    durations_dir.mkdir(parents=True)
    durations = {
        ci_submit_module._test_nodeid(test_lines[0]): 2211.1548,
        ci_submit_module._test_nodeid(test_lines[1]): 2548.912,
        ci_submit_module._test_nodeid(test_lines[2]): 836.268,
        ci_submit_module._test_nodeid(test_lines[3]): 1462.754,
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

    assert selected == test_lines[3]


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
