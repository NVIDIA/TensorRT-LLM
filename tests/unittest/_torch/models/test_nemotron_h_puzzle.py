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
"""Tests for NemotronHPuzzle model support."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.models.modeling_nemotron_h import (
    NemotronHForCausalLM,
    _get_layer_moe_param,
)


@dataclass
class _MambaBlock:
    block_type: str = "mamba"


@dataclass
class _MoeBlock:
    block_type: str = "moe"
    moe_intermediate_size: int = 1280
    n_routed_experts: int = 512
    num_experts_per_tok: int = 4
    moe_latent_size: int = 1024
    moe_shared_expert_intermediate_size: int = 5376


def _make_puzzle_config(use_dataclass=False):
    """Minimal config mimicking the real puzzle model."""
    if use_dataclass:
        bcs = [
            _MambaBlock(),
            _MoeBlock(num_experts_per_tok=4),
            _MambaBlock(),
            _MoeBlock(moe_intermediate_size=2048, num_experts_per_tok=12),
        ]
    else:
        bcs = [
            {"block_type": "mamba"},
            {
                "block_type": "moe",
                "moe_intermediate_size": 1280,
                "n_routed_experts": 512,
                "num_experts_per_tok": 4,
                "moe_latent_size": 1024,
                "moe_shared_expert_intermediate_size": 5376,
            },
            {"block_type": "mamba"},
            {
                "block_type": "moe",
                "moe_intermediate_size": 2048,
                "n_routed_experts": 512,
                "num_experts_per_tok": 12,
                "moe_latent_size": 1024,
                "moe_shared_expert_intermediate_size": 5376,
            },
        ]
    return SimpleNamespace(
        block_configs=bcs,
        mtp_block_configs=[
            {"block_type": "attention"},
            {
                "block_type": "moe",
                "moe_intermediate_size": 2688,
                "n_routed_experts": 512,
                "num_experts_per_tok": 22,
                "moe_latent_size": 1024,
                "moe_shared_expert_intermediate_size": 5376,
            },
        ],
    )


class TestPerLayerMoeParams:
    """The key change: block_configs can be dicts or HF dataclass objects,
    and per-layer values must differ while MTP falls back to globals."""

    @pytest.mark.parametrize("use_dc", [False, True], ids=["dict", "dataclass"])
    def test_varying_params_per_layer(self, use_dc):
        config = _make_puzzle_config(use_dataclass=use_dc)
        NemotronHForCausalLM._normalize_puzzle_config(config)

        # MoE layer 1: top_k=4, intermediate=1280
        assert _get_layer_moe_param(config, 1, "num_experts_per_tok") == 4
        assert _get_layer_moe_param(config, 1, "moe_intermediate_size") == 1280
        # MoE layer 3: top_k=12, intermediate=2048
        assert _get_layer_moe_param(config, 3, "num_experts_per_tok") == 12
        assert _get_layer_moe_param(config, 3, "moe_intermediate_size") == 2048

    @pytest.mark.parametrize("use_dc", [False, True], ids=["dict", "dataclass"])
    def test_mtp_layer_gets_global_defaults(self, use_dc):
        """MTP layer_idx beyond block_configs range uses globals from mtp_block_configs."""
        config = _make_puzzle_config(use_dataclass=use_dc)
        NemotronHForCausalLM._normalize_puzzle_config(config)

        mtp_idx = len(config.block_configs)  # beyond range
        assert _get_layer_moe_param(config, mtp_idx, "num_experts_per_tok") == 22
        assert _get_layer_moe_param(config, mtp_idx, "moe_intermediate_size") == 2688

    @pytest.mark.parametrize("use_dc", [False, True], ids=["dict", "dataclass"])
    def test_normalize_sets_all_global_attrs(self, use_dc):
        config = _make_puzzle_config(use_dataclass=use_dc)
        NemotronHForCausalLM._normalize_puzzle_config(config)

        for attr in (
            "n_routed_experts",
            "moe_intermediate_size",
            "num_experts_per_tok",
            "moe_latent_size",
            "moe_shared_expert_intermediate_size",
        ):
            assert getattr(config, attr) is not None, f"{attr} not set"

    def test_normalize_preserves_existing_attrs(self):
        config = _make_puzzle_config()
        config.n_routed_experts = 999
        NemotronHForCausalLM._normalize_puzzle_config(config)
        assert config.n_routed_experts == 999

    def test_normalize_noop_without_block_configs(self):
        config = SimpleNamespace()
        NemotronHForCausalLM._normalize_puzzle_config(config)
        assert not hasattr(config, "n_routed_experts")

    def test_standard_config_passthrough(self):
        """Non-puzzle model: no block_configs, returns global directly."""
        config = SimpleNamespace(n_routed_experts=512)
        assert _get_layer_moe_param(config, 0, "n_routed_experts") == 512
