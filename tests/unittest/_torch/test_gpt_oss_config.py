# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
from transformers import GptOssConfig

from tensorrt_llm._torch.configs.gpt_oss import normalize_gpt_oss_config

pytestmark = pytest.mark.cpu_only


def test_native_gpt_oss_config_preserves_experts_and_rope() -> None:
    config = GptOssConfig(num_local_experts=16, num_experts_per_tok=2)
    before = deepcopy(config.to_dict())

    normalize_gpt_oss_config(config)

    assert config.to_dict() == before


def test_legacy_expert_names_preserve_native_rope() -> None:
    config = GptOssConfig(num_experts=32, experts_per_token=8)
    before_rope = deepcopy(config.rope_scaling)

    normalize_gpt_oss_config(config)

    assert config.num_local_experts == 32
    assert config.num_experts_per_tok == 8
    assert config.rope_scaling == before_rope


def test_legacy_gpt_oss_fields_are_normalized() -> None:
    config = GptOssConfig(
        num_experts=16,
        experts_per_token=2,
        rope_scaling_factor=8.0,
        rope_ntk_beta=16.0,
        rope_ntk_alpha=2.0,
        initial_context_length=2048,
    )

    normalize_gpt_oss_config(config)

    assert config.num_local_experts == 16
    assert config.num_experts_per_tok == 2
    assert config.rope_scaling == {
        "factor": 8.0,
        "beta_fast": 16.0,
        "beta_slow": 2.0,
        "original_max_position_embeddings": 2048,
    }
