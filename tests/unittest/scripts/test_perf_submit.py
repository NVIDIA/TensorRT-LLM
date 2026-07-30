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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SUBMIT_PATHS = (
    REPO_ROOT / "jenkins" / "scripts" / "perf" / "submit.py",
    REPO_ROOT / "jenkins" / "scripts" / "perf" / "local" / "submit.py",
)
DISAGG_CONFIG_DIR = REPO_ROOT / "tests" / "scripts" / "perf-sanity" / "disaggregated"
PR16645_MAIN_AB_CONFIG = (
    DISAGG_CONFIG_DIR
    / "gb300_deepseek-r1-fp4_128k8k_con256_ctx1_pp4_gen1_dep8_eplb0_mtp1_ccb-NIXL.yaml"
)


def test_pr16645_main_ab_config_holds_common_python_context_first_workload():
    with open(PR16645_MAIN_AB_CONFIG) as config_file:
        config = yaml.safe_load(config_file)

    benchmark = config["benchmark"]
    worker_config = config["worker_config"]
    ctx_env = dict(
        item.split("=", 1)
        for item in config["environment"]["ctx_worker_env_var"].split()
        if "=" in item
    )

    assert config["server_config_extra"] == {"schedule_style": "context_first"}
    assert benchmark["multi_round"] == 1
    assert benchmark["post_benchmark_drain_seconds"] == 60
    assert benchmark["concurrency_list"] == "256"
    assert {
        role: worker_config[role]["cache_transceiver_config"]["transceiver_runtime"]
        for role in ("ctx", "gen")
    } == {"ctx": "PYTHON", "gen": "PYTHON"}
    assert {
        role: worker_config[role]["cache_transceiver_config"]["max_tokens_in_buffer"]
        for role in ("ctx", "gen")
    } == {"ctx": 131104, "gen": 131104}
    assert ctx_env == {
        "TRTLLM_PYTHON_TRANSCEIVER_ASYNC_CTX_TERMINAL_CONSENSUS": "1",
        "TRTLLM_PYTHON_TRANSCEIVER_ASYNC_CTX_PEER_READY_CONSENSUS": "1",
    }


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
