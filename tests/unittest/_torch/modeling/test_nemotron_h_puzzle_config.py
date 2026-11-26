# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

import pytest
from transformers import AutoConfig

from tensorrt_llm._torch.models.modeling_nemotron_h import (
    NemotronHPuzzleConfig,
    _get_layer_moe_param,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def puzzle_fields() -> dict[str, object]:
    return {
        "model_type": "nemotron_h_puzzle",
        "architectures": ["NemotronHPuzzleForCausalLM"],
        "auto_map": {"AutoConfig": "configuration_nemotron_h_puzzle.NemotronHPuzzleConfig"},
        "layers_block_type": ["mamba", "moe", "attention", "mamba", "moe"],
        "block_configs": [
            {"block_type": "mamba"},
            {"block_type": "moe", "moe_intermediate_size": 1280, "num_experts_per_tok": 4},
            {"block_type": "attention"},
            {"block_type": "mamba"},
            {"block_type": "moe", "moe_intermediate_size": 2560, "num_experts_per_tok": 8},
        ],
        "moe_intermediate_size": 4096,
        "num_experts_per_tok": 2,
        "num_nextn_predict_layers": 1,
        "mtp_layers_block_type": ["attention", "moe"],
        "mtp_block_configs": [
            {"block_type": "attention"},
            {"block_type": "moe", "moe_intermediate_size": 1024, "num_experts_per_tok": 2},
        ],
    }


def _assert_layer_fields(config: NemotronHPuzzleConfig) -> None:
    assert config.num_hidden_layers == 5
    assert config.hybrid_override_pattern == "ME*ME"
    assert config.mtp_hybrid_override_pattern == "*E"


@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_autoconfig_derives_puzzle_layer_fields(
    tmp_path: Path, puzzle_fields: dict[str, object], trust_remote_code: bool
) -> None:
    (tmp_path / "config.json").write_text(json.dumps(puzzle_fields))

    config = AutoConfig.from_pretrained(
        tmp_path, trust_remote_code=trust_remote_code, local_files_only=True
    )

    assert type(config) is NemotronHPuzzleConfig
    _assert_layer_fields(config)
    assert config.block_configs == puzzle_fields["block_configs"]
    assert config.mtp_block_configs == puzzle_fields["mtp_block_configs"]
    assert _get_layer_moe_param(config, 1, "moe_intermediate_size") == 1280
    assert _get_layer_moe_param(config, 4, "moe_intermediate_size") == 2560
    assert _get_layer_moe_param(config, 1, "num_experts_per_tok") == 4
    assert _get_layer_moe_param(config, 4, "num_experts_per_tok") == 8


def test_puzzle_config_round_trip(tmp_path: Path, puzzle_fields: dict[str, object]) -> None:
    config = NemotronHPuzzleConfig.from_dict(puzzle_fields)
    config.save_pretrained(tmp_path)

    restored = AutoConfig.from_pretrained(tmp_path, trust_remote_code=True, local_files_only=True)

    assert type(restored) is NemotronHPuzzleConfig
    _assert_layer_fields(restored)
    assert restored.layers_block_type == puzzle_fields["layers_block_type"]
    assert restored.block_configs == puzzle_fields["block_configs"]
    assert restored.mtp_block_configs == puzzle_fields["mtp_block_configs"]
    assert restored.architectures == ["NemotronHPuzzleForCausalLM"]


def test_puzzle_config_accepts_legacy_pattern(puzzle_fields: dict[str, object]) -> None:
    fields = dict(puzzle_fields)
    del fields["layers_block_type"]
    fields["hybrid_override_pattern"] = "ME*ME"
    fields["num_hidden_layers"] = 5

    config = NemotronHPuzzleConfig.from_dict(fields)

    _assert_layer_fields(config)
    assert config.layers_block_type == puzzle_fields["layers_block_type"]
