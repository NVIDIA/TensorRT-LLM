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
"""GPU-free unit tests for the disaggregated readiness budget.

setup_disagg_cluster picks its readiness budget from the config when the
caller passes none, so every pipeline-parallel config gets the longer budget
without each test having to ask for it. Run with:

    pytest -sv disaggregated/test_server_start_timeout.py
"""

from pathlib import Path

import pytest
import yaml
from test_disaggregated import (
    DEFAULT_SERVER_START_TIMEOUT,
    PP_DISAGG_SERVER_START_TIMEOUT,
    resolve_server_start_timeout,
)

_TEST_CONFIGS = Path(__file__).parent / "test_configs"


def _config(ctx_pp: int, gen_pp: int) -> dict:
    return {
        "context_servers": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": ctx_pp,
        },
        "generation_servers": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": gen_pp,
        },
    }


@pytest.mark.parametrize("ctx_pp, gen_pp", [(2, 1), (1, 2), (4, 4)])
def test_pipeline_parallel_config_gets_pp_budget(ctx_pp, gen_pp):
    assert (
        resolve_server_start_timeout(_config(ctx_pp, gen_pp), None)
        == PP_DISAGG_SERVER_START_TIMEOUT
    )


def test_non_pp_config_gets_default_budget():
    assert resolve_server_start_timeout(_config(1, 1), None) == DEFAULT_SERVER_START_TIMEOUT


def test_missing_server_sections_get_default_budget():
    assert resolve_server_start_timeout({}, None) == DEFAULT_SERVER_START_TIMEOUT


@pytest.mark.parametrize("config", [_config(1, 1), _config(4, 4)])
def test_explicit_budget_wins(config):
    assert resolve_server_start_timeout(config, 1200) == 1200


@pytest.mark.parametrize(
    "config_name",
    [
        "disagg_config_ctxpp2_genpp2.yaml",
        "disagg_config_ctxtp2_genpp2.yaml",
        "disagg_config_ctxpp2_gentp2.yaml",
        "disagg_config_ctxpp4_gentp4.yaml",
        "disagg_config_ctxpp4_genpp4.yaml",
        "disagg_config_ctxtp2pp2_gentp2pp2.yaml",
        "disagg_config_overlap_gen_first_pp4.yaml",
    ],
)
def test_qwen3_pp_test_configs_get_pp_budget(config_name):
    config = yaml.safe_load((_TEST_CONFIGS / config_name).read_text())
    assert resolve_server_start_timeout(config, None) == PP_DISAGG_SERVER_START_TIMEOUT
