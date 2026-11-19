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

setup_disagg_cluster picks its readiness budget from the config and the GPU
when the caller passes none: 900s for a pipeline-parallel config on an
SM100-family GPU, 300s otherwise. The expected budgets are written out as
numbers so that a change to either constant fails here. Run with:

    pytest -sv disaggregated/test_server_start_timeout.py
"""

from pathlib import Path

import pytest
import yaml
from disagg_test_utils import resolve_server_start_timeout

_TEST_CONFIGS = Path(__file__).parent / "test_configs"


def _config(ctx_pp: int, gen_pp: int) -> dict:
    return {
        "context_servers": {"tensor_parallel_size": 1, "pipeline_parallel_size": ctx_pp},
        "generation_servers": {"tensor_parallel_size": 1, "pipeline_parallel_size": gen_pp},
    }


@pytest.mark.parametrize(
    "config, sm_version",
    [
        pytest.param(_config(2, 1), 100, id="ctx_pp2-sm100"),
        pytest.param(_config(1, 2), 103, id="gen_pp2-sm103"),
        pytest.param(_config(4, 4), 107, id="pp4-sm107"),
    ],
)
def test_pp_on_sm100_family_gets_900s(config, sm_version):
    assert resolve_server_start_timeout(config, None, sm_version) == 900


@pytest.mark.parametrize("sm_version", [90, 120], ids=["sm90", "sm120"])
def test_pp_on_other_gpus_gets_300s(sm_version):
    assert resolve_server_start_timeout(_config(4, 4), None, sm_version) == 300


def test_non_pp_config_gets_300s():
    assert resolve_server_start_timeout(_config(1, 1), None, 100) == 300


def test_missing_server_sections_get_300s():
    assert resolve_server_start_timeout({}, None, 100) == 300


def test_top_level_pp_applies_to_servers_without_their_own():
    config = {"pipeline_parallel_size": 2, "context_servers": {}, "generation_servers": {}}
    assert resolve_server_start_timeout(config, None, 100) == 900


def test_server_pp_overrides_top_level_pp():
    config = {
        "pipeline_parallel_size": 2,
        "context_servers": {"pipeline_parallel_size": 1},
        "generation_servers": {"pipeline_parallel_size": 1},
    }
    assert resolve_server_start_timeout(config, None, 100) == 300


@pytest.mark.parametrize(
    "config, explicit",
    [
        pytest.param(_config(4, 4), 600, id="pp4-below_pp_budget"),
        pytest.param(_config(1, 1), 1200, id="no_pp-above_default"),
    ],
)
def test_explicit_budget_wins(config, explicit):
    assert resolve_server_start_timeout(config, explicit, 100) == explicit


def test_every_test_config_follows_pp_rule():
    """Each checked-in disagg config gets 900s on SM100 exactly when a server section uses PP."""
    mismatches = []
    num_pp = 0
    for path in sorted(_TEST_CONFIGS.glob("*.yaml")):
        config = yaml.safe_load(path.read_text())
        if not isinstance(config, dict):
            continue
        # A server section without its own pipeline_parallel_size inherits the top-level one.
        top_level_pp = config.get("pipeline_parallel_size", 1)
        uses_pp = any(
            (config.get(servers) or {}).get("pipeline_parallel_size", top_level_pp) > 1
            for servers in ("context_servers", "generation_servers")
        )
        num_pp += uses_pp
        expected = 900 if uses_pp else 300
        actual = resolve_server_start_timeout(config, None, 100)
        if actual != expected:
            mismatches.append(f"{path.name}: {actual}s, expected {expected}s")
    assert not mismatches, "\n".join(mismatches)
    # The Qwen3-0.6B and DeepSeek-V3-Lite PP configs; guards against an empty glob.
    assert num_pp >= 10
