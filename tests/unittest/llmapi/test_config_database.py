# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""L0 tests for validating config database YAML files against TorchLlmArgs."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from tensorrt_llm.llmapi.llm_args import TorchLlmArgs, update_llm_args_with_extra_dict

CONFIG_ROOT = Path(__file__).parents[3] / "examples" / "configs"
DATABASE_DIR = CONFIG_ROOT / "database"

DATABASE_CONFIGS = (
    [c for c in DATABASE_DIR.rglob("*.yaml") if c.name != "lookup.yaml"]
    if DATABASE_DIR.exists()
    else []
)


@pytest.fixture(autouse=True)
def mock_gpu_environment():
    """Mock GPU functions for CPU-only test execution."""
    mock_props = Mock()
    mock_props.major = 8

    with patch("torch.cuda.device_count", return_value=8):
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            with patch("torch.cuda.is_available", return_value=True):
                yield


def get_config_id(config_path: Path) -> str:
    return str(config_path.relative_to(DATABASE_DIR))


@pytest.mark.part0
@pytest.mark.parametrize("config_path", DATABASE_CONFIGS, ids=get_config_id)
def test_config_validates_against_llm_args(config_path: Path):
    with open(config_path) as f:
        config_dict = yaml.safe_load(f) or {}

    base_args = TorchLlmArgs(model="dummy/model", skip_tokenizer_init=True)
    merged = update_llm_args_with_extra_dict(base_args.model_dump(), config_dict)
    TorchLlmArgs(**merged)


@pytest.mark.part0
@pytest.mark.parametrize("config_path", DATABASE_CONFIGS, ids=get_config_id)
def test_config_does_not_disable_kv_cache_block_reuse(config_path: Path):
    with open(config_path) as f:
        config_dict = yaml.safe_load(f) or {}

    assert config_dict.get("kv_cache_config", {}).get("enable_block_reuse") is not False


@pytest.mark.part0
def test_database_config_count():
    assert len(DATABASE_CONFIGS) > 0, "No database config files found"
