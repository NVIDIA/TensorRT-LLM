# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.checkpoints.hf.qwen3_next_weight_mapper import (
    Qwen3NextHfWeightMapper,
)


def _make_mapper() -> Qwen3NextHfWeightMapper:
    mapper = Qwen3NextHfWeightMapper()
    mapper._config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=2,
            linear_value_head_dim=3,
        )
    )
    return mapper


def _pack_by_key_group(tensors: list[torch.Tensor], num_groups: int) -> torch.Tensor:
    trailing_shape = tensors[0].shape[1:]
    return torch.cat(
        [tensor.reshape(num_groups, -1, *trailing_shape) for tensor in tensors],
        dim=1,
    ).reshape(-1, *trailing_shape)


def test_combine_gdn_input_projections_uses_consumer_order():
    mapper = _make_mapper()
    num_groups = 2
    input_dim = 5
    q = torch.arange(4 * input_dim).reshape(4, input_dim)
    k = q + 100
    v = torch.arange(12 * input_dim).reshape(12, input_dim) + 200
    z = v + 1000
    b = torch.arange(4 * input_dim).reshape(4, input_dim) + 2000
    a = b + 1000

    prefix = "model.layers.0.linear_attn"
    weights = {
        f"{prefix}.in_proj_qkvz.weight": _pack_by_key_group([q, k, v, z], num_groups),
        f"{prefix}.in_proj_ba.weight": _pack_by_key_group([b, a], num_groups),
        "model.embed_tokens.weight": torch.ones(2, 2),
    }

    combined = mapper._combine_gdn_input_projections(weights)

    torch.testing.assert_close(
        combined[f"{prefix}.in_proj_qkvzba.weight"],
        torch.cat([q, k, v, z, b, a], dim=0),
    )
    assert "model.embed_tokens.weight" in combined
    assert f"{prefix}.in_proj_qkvz.weight" not in combined
    assert f"{prefix}.in_proj_ba.weight" not in combined


def test_combine_gdn_input_projections_rejects_different_scalar_metadata():
    mapper = _make_mapper()
    prefix = "model.layers.0.linear_attn"
    weights = {
        f"{prefix}.in_proj_qkvz.alpha": torch.tensor(1.0),
        f"{prefix}.in_proj_ba.alpha": torch.tensor(2.0),
    }

    with pytest.raises(ValueError, match="Cannot combine non-row"):
        mapper._combine_gdn_input_projections(weights)
