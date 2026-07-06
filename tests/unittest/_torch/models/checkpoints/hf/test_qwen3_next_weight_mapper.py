# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.checkpoints.hf.qwen3_5_weight_mapper import Qwen3_5MoeHfWeightMapper
from tensorrt_llm._torch.models.checkpoints.hf.qwen3_next_weight_mapper import (
    Qwen3NextHfWeightMapper,
)
from tensorrt_llm._torch.models.modeling_qwen3_5 import (
    Qwen35ConfigCompat,
    _normalize_qwen35_exclude_modules,
)
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


def _make_mapper(
    tp_size=1,
    enable_attention_dp=False,
    quant_algo=QuantAlgo.NVFP4,
) -> Qwen3NextHfWeightMapper:
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
        quant_config=QuantConfig(quant_algo=quant_algo),
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


@pytest.mark.parametrize("quant_algo", [None, QuantAlgo.FP8_BLOCK_SCALES])
def test_preprocess_keeps_split_gdn_input_projections_for_non_nvfp4(quant_algo):
    mapper = _make_mapper(quant_algo=quant_algo)
    prefix = "model.layers.0.linear_attn"
    qkvz = torch.empty((32, 5), dtype=torch.float8_e4m3fn)
    ba = torch.ones((8, 5), dtype=torch.bfloat16)
    scale = torch.ones((1, 1), dtype=torch.float32)
    weights = {
        f"{prefix}.in_proj_qkvz.weight": qkvz,
        f"{prefix}.in_proj_qkvz.weight_scale_inv": scale,
        f"{prefix}.in_proj_ba.weight": ba,
    }

    preprocessed = mapper.preprocess_weights(weights)

    assert preprocessed[f"{prefix}.in_proj_qkvz.weight"] is qkvz
    assert preprocessed[f"{prefix}.in_proj_qkvz.weight_scale_inv"] is scale
    assert preprocessed[f"{prefix}.in_proj_ba.weight"] is ba
    assert f"{prefix}.in_proj_qkvzba.weight" not in preprocessed


@pytest.mark.parametrize(
    "quant_algo,expected",
    [
        (QuantAlgo.NVFP4, "model.layers.1.linear_attn.in_proj_qkvzba*"),
        (QuantAlgo.FP8_BLOCK_SCALES, "model.layers.1.linear_attn.in_proj_ba*"),
    ],
)
def test_qwen35_exclude_modules_follow_projection_layout(quant_algo, expected):
    model_config = SimpleNamespace(
        quant_config=QuantConfig(
            quant_algo=quant_algo,
            exclude_modules=["model.language_model.layers.1.linear_attn.in_proj_a"],
        ),
        pretrained_config=SimpleNamespace(num_hidden_layers=2),
    )

    _normalize_qwen35_exclude_modules(model_config)

    assert expected in model_config.quant_config.exclude_modules


def test_qwen35_raw_fp8_excludes_map_to_split_projections():
    normalized = Qwen35ConfigCompat._normalize_exclude_modules(
        [
            "model.language_model.layers.1.linear_attn.in_proj_qkv",
            "model.language_model.layers.1.linear_attn.in_proj_a",
        ]
    )

    assert normalized == [
        "model.layers.1.linear_attn.in_proj_ba",
        "model.layers.1.linear_attn.in_proj_qkvz",
    ]


def test_qwen35_fp8_preprocess_dequantizes_and_keeps_split_projections():
    mapper = Qwen3_5MoeHfWeightMapper()
    mapper._config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            num_hidden_layers=1,
            num_experts=1,
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            torch_dtype=torch.bfloat16,
        ),
        mapping=SimpleNamespace(tp_size=1, tp_rank=0, enable_attention_dp=False),
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
    )
    prefix = "model.language_model.layers.0.linear_attn"
    weights = {
        f"{prefix}.in_proj_qkv.weight": torch.ones((384, 128), dtype=torch.float8_e4m3fn),
        f"{prefix}.in_proj_qkv.weight_scale_inv": torch.ones((3, 1)),
        f"{prefix}.in_proj_z.weight": torch.ones((128, 128), dtype=torch.float8_e4m3fn),
        f"{prefix}.in_proj_z.weight_scale_inv": torch.ones((1, 1)),
        f"{prefix}.in_proj_b.weight": torch.ones((1, 128), dtype=torch.bfloat16),
        f"{prefix}.in_proj_a.weight": torch.ones((1, 128), dtype=torch.bfloat16),
    }

    preprocessed = mapper.preprocess_weights(weights)
    runtime_prefix = "model.layers.0.linear_attn"

    assert preprocessed[f"{runtime_prefix}.in_proj_qkvz.weight"].shape == (512, 128)
    assert preprocessed[f"{runtime_prefix}.in_proj_qkvz.weight"].dtype == torch.bfloat16
    assert preprocessed[f"{runtime_prefix}.in_proj_ba.weight"].shape == (2, 128)
    assert f"{runtime_prefix}.in_proj_qkvzba.weight" not in preprocessed
    assert not any(name.endswith("weight_scale_inv") for name in preprocessed)


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
