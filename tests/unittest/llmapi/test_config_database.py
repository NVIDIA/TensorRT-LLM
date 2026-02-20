# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""L0 wrapper tests for validating database YAML files against TorchLlmArgs."""

from pathlib import Path

import pytest

from . import yaml_validation_harness as yaml_harness

CONFIG_ROOT = Path(__file__).parents[3] / "examples" / "configs"
DATABASE_DIR = CONFIG_ROOT / "database"

DATABASE_CONFIGS = yaml_harness.collect_yaml_files(
    DATABASE_DIR, "**/*.yaml", exclude_names={"lookup.yaml"}
)


@pytest.fixture(autouse=True)
def mock_gpu_environment():
    """Mock GPU functions for CPU-only test execution."""
    with yaml_harness.mock_cuda_for_schema_validation():
        yield


def get_config_id(config_path: Path) -> str:
    return str(config_path.relative_to(DATABASE_DIR))


@pytest.mark.part0
@pytest.mark.parametrize("config_path", DATABASE_CONFIGS, ids=get_config_id)
def test_config_validates_against_llm_args(config_path: Path):
    config_dict = yaml_harness.load_yaml_dict(config_path)
    yaml_harness.validate_torch_llm_args_config(config_dict)


def _assert_kv_cache_block_reuse_policy(config_dict: dict) -> None:
    assert config_dict.get("kv_cache_config", {}).get("enable_block_reuse") is not False


@pytest.mark.part0
@pytest.mark.parametrize("config_path", DATABASE_CONFIGS, ids=get_config_id)
def test_config_does_not_disable_kv_cache_block_reuse(config_path: Path):
    config_dict = yaml_harness.load_yaml_dict(config_path)
    yaml_harness.assert_custom_policy(config_dict, _assert_kv_cache_block_reuse_policy)


@pytest.mark.part0
def test_database_config_count():
    assert len(DATABASE_CONFIGS) > 0, "No database config files found"
