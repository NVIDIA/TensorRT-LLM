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
"""L0 tests for validating curated YAML files against TorchLlmArgs."""

from pathlib import Path

import pytest

from tensorrt_llm.llmapi import llm_args as llm_args_module

from . import yaml_validation_harness as yaml_harness

CONFIG_ROOT = Path(__file__).parents[3] / "examples" / "configs"
CURATED_DIR = CONFIG_ROOT / "curated"

CURATED_CONFIGS = yaml_harness.collect_yaml_files(CURATED_DIR, "**/*.yaml")

DEPRECATED_KEY_MAP = {
    "kv_cache_free_gpu_memory_fraction": "kv_cache_config.free_gpu_memory_fraction",
    "moe_backend": "moe_config.backend",
    "use_cuda_graph": "cuda_graph_config",
}


@pytest.fixture(autouse=True)
def mock_gpu_environment():
    """Mock GPU functions for CPU-only schema test execution."""
    with yaml_harness.mock_cuda_for_schema_validation():
        yield


def get_config_id(config_path: Path) -> str:
    return str(config_path.relative_to(CURATED_DIR))


@pytest.mark.part0
@pytest.mark.parametrize("config_path", CURATED_CONFIGS, ids=get_config_id)
def test_curated_config_validates_against_llm_args(config_path: Path):
    config_dict = yaml_harness.load_yaml_dict(config_path)
    yaml_harness.validate_torch_llm_args_config(config_dict)


@pytest.mark.part0
@pytest.mark.parametrize("config_path", CURATED_CONFIGS, ids=get_config_id)
def test_curated_config_does_not_use_deprecated_keys(config_path: Path):
    config_dict = yaml_harness.load_yaml_dict(config_path)
    yaml_harness.assert_no_deprecated_keys(config_dict, DEPRECATED_KEY_MAP)


@pytest.mark.part0
@pytest.mark.parametrize("config_path", CURATED_CONFIGS, ids=get_config_id)
def test_curated_config_does_not_set_default_leaves(config_path: Path):
    config_dict = yaml_harness.load_yaml_dict(config_path)
    default_cfg = llm_args_module.TorchLlmArgs(
        model="dummy/model", skip_tokenizer_init=True
    ).model_dump(mode="json")
    yaml_harness.assert_no_default_valued_leaves(config_dict, default_cfg)


@pytest.mark.part0
def test_curated_config_count():
    assert len(CURATED_CONFIGS) > 0, "No curated config files found"
