# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.checkpoints.hf.qwen3_5_weight_mapper import Qwen3_5MoeHfWeightMapper
from tensorrt_llm._torch.models.checkpoints.hf.qwen3_next_weight_mapper import (
    Qwen3NextHfWeightMapper,
)


def _make_mapper(tp_size=1, enable_attention_dp=False) -> Qwen3NextHfWeightMapper:
    mapper = Qwen3NextHfWeightMapper()
    mapper._config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            num_hidden_layers=1,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=2,
            linear_value_head_dim=3,
        ),
        mapping=SimpleNamespace(
            tp_size=tp_size,
            tp_rank=0,
            enable_attention_dp=enable_attention_dp,
        ),
    )
    return mapper


def _pack_by_key_group(tensors: list[torch.Tensor], num_groups: int) -> torch.Tensor:
    trailing_shape = tensors[0].shape[1:]
    return torch.cat(
        [tensor.reshape(num_groups, -1, *trailing_shape) for tensor in tensors],
        dim=1,
    ).reshape(-1, *trailing_shape)


def _pack_rank_major(tensors: list[torch.Tensor], tp_size: int) -> torch.Tensor:
    rank_shards = []
    for rank in range(tp_size):
        rank_shards.append(
            torch.cat(
                [torch.chunk(tensor, tp_size, dim=0)[rank] for tensor in tensors],
                dim=0,
            )
        )
    return torch.cat(rank_shards, dim=0)


@pytest.mark.parametrize(
    "tp_size,enable_attention_dp,effective_tp_size",
    [(1, False, 1), (2, False, 2), (4, True, 1)],
)
def test_combine_gdn_input_projections_uses_rank_major_consumer_order(
    tp_size,
    enable_attention_dp,
    effective_tp_size,
):
    mapper = _make_mapper(tp_size, enable_attention_dp)
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

    combined = mapper.preprocess_weights(weights)

    torch.testing.assert_close(
        combined[f"{prefix}.in_proj_qkvzba.weight"],
        _pack_rank_major([q, k, v, z, b, a], effective_tp_size),
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
        mapper._combine_gdn_input_projections(weights, tp_size=1)


def test_combine_gdn_input_projections_rejects_missing_projection():
    mapper = _make_mapper()
    prefix = "model.layers.0.linear_attn"
    weights = {
        f"{prefix}.in_proj_qkvz.weight": torch.ones(32, 5),
    }

    with pytest.raises(ValueError, match="Expected both QKVZ and BA"):
        mapper._combine_gdn_input_projections(weights, tp_size=1)


def test_combine_gdn_input_projections_rejects_trailing_shape_mismatch():
    mapper = _make_mapper()
    prefix = "model.layers.0.linear_attn"
    weights = {
        f"{prefix}.in_proj_qkvz.weight": torch.ones(32, 5),
        f"{prefix}.in_proj_ba.weight": torch.ones(8, 6),
    }

    with pytest.raises(ValueError, match="trailing shapes do not match"):
        mapper._combine_gdn_input_projections(weights, tp_size=1)


def test_combine_gdn_input_projections_rejects_existing_combined_key():
    mapper = _make_mapper()
    prefix = "model.layers.0.linear_attn"
    weights = {
        f"{prefix}.in_proj_qkvz.weight": torch.ones(32, 5),
        f"{prefix}.in_proj_ba.weight": torch.ones(8, 5),
        f"{prefix}.in_proj_qkvzba.weight": torch.ones(40, 5),
    }

    with pytest.raises(ValueError, match="already exists"):
        mapper._combine_gdn_input_projections(weights, tp_size=1)


def test_combine_gdn_input_projections_rejects_unsupported_tp_size():
    mapper = _make_mapper()

    with pytest.raises(ValueError, match="must be divisible by TP size 4"):
        mapper._combine_gdn_input_projections({}, tp_size=4)


def test_dequantize_linear_attn_fp8_projections_handles_qkvz_and_ba():
    mapper = Qwen3_5MoeHfWeightMapper()
    mapper._config = SimpleNamespace(pretrained_config=SimpleNamespace(torch_dtype=torch.bfloat16))
    prefix = "model.layers.0.linear_attn"
    weights = {
        f"{prefix}.in_proj_qkvz.weight": torch.ones(32, 5),
        f"{prefix}.in_proj_qkvz.weight_scale_inv": torch.full((1, 1), 2.0),
        f"{prefix}.in_proj_ba.weight": torch.ones(8, 5),
        f"{prefix}.in_proj_ba.weight_scale_inv": torch.full((1, 1), 3.0),
    }

    dequantized = mapper._dequantize_linear_attn_fp8_projections(weights)

    torch.testing.assert_close(
        dequantized[f"{prefix}.in_proj_qkvz.weight"],
        torch.full((32, 5), 2.0, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        dequantized[f"{prefix}.in_proj_ba.weight"],
        torch.full((8, 5), 3.0, dtype=torch.bfloat16),
    )
    assert f"{prefix}.in_proj_qkvz.weight_scale_inv" not in dequantized
    assert f"{prefix}.in_proj_ba.weight_scale_inv" not in dequantized
