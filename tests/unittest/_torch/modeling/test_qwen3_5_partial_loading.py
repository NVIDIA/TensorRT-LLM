# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from tensorrt_llm._torch.models.checkpoints.hf.qwen3_5_weight_mapper import Qwen3_5MoeHfWeightMapper
from tensorrt_llm._torch.models.modeling_qwen3_5 import _Qwen3_5VLModel
from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VisionModelBase
from tensorrt_llm.quantization import QuantAlgo


def test_qwen35_vl_propagates_partial_loading_to_vision_encoder():
    model = SimpleNamespace(
        mm_encoder=MagicMock(),
        llm=MagicMock(),
    )
    model.llm.model_config = MagicMock()
    mapper = Qwen3_5MoeHfWeightMapper()
    mapper._model = model.llm
    mapper._config = model.llm.model_config

    _Qwen3_5VLModel.load_weights(
        model,
        {},
        mapper,
        allow_partial_loading=True,
    )

    model.mm_encoder.load_weights.assert_called_once_with(
        {}, allow_partial_loading=True
    )
    assert model.llm.load_weights.call_args.kwargs["allow_partial_loading"] is True


class _VisualStub(nn.Module):

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_heads=2)


def test_qwen3_vision_loader_propagates_partial_loading():
    model = Qwen3VisionModelBase.__new__(Qwen3VisionModelBase)
    nn.Module.__init__(model)
    model.visual = _VisualStub()
    weights = {
        "model.visual.blocks.0.attn.qkv.weight": torch.empty(6, 2),
    }

    with patch(
        "tensorrt_llm._torch.models.modeling_qwen3vl._load_weights_impl"
    ) as load_weights_impl:
        model.load_weights(weights, allow_partial_loading=True)

    assert load_weights_impl.call_args.kwargs["allow_partial_loading"] is True
    converted_weights = load_weights_impl.call_args.args[1]
    assert converted_weights["blocks.0.attn.q_proj.weight"].shape == (2, 2)
    assert converted_weights["blocks.0.attn.k_proj.weight"].shape == (2, 2)
    assert converted_weights["blocks.0.attn.v_proj.weight"].shape == (2, 2)


def test_qwen35_partial_block_fp8_qkvz_waits_for_weights_and_scales():
    mapper = Qwen3_5MoeHfWeightMapper()
    prefix = "model.layers.0.linear_attn.in_proj_"
    qkv_weight = torch.empty(4, 2, dtype=torch.float8_e4m3fn)
    z_weight = torch.empty(2, 2, dtype=torch.float8_e4m3fn)
    qkv_scale = torch.empty(3, 1, dtype=torch.float32)
    z_scale = torch.empty(1, 1, dtype=torch.float32)

    mapper.begin_update_weights()
    assert not mapper._stage_partial_split_projections(
        {f"{prefix}qkv.weight": qkv_weight}, QuantAlgo.FP8_BLOCK_SCALES)
    assert not mapper._stage_partial_split_projections(
        {f"{prefix}z.weight": z_weight}, QuantAlgo.FP8_BLOCK_SCALES)
    assert not mapper._stage_partial_split_projections(
        {f"{prefix}qkv.weight_scale_inv": qkv_scale},
        QuantAlgo.FP8_BLOCK_SCALES,
    )
    ready = mapper._stage_partial_split_projections(
        {f"{prefix}z.weight_scale_inv": z_scale},
        QuantAlgo.FP8_BLOCK_SCALES,
    )

    assert ready == {
        f"{prefix}qkv.weight": qkv_weight,
        f"{prefix}z.weight": z_weight,
        f"{prefix}qkv.weight_scale_inv": qkv_scale,
        f"{prefix}z.weight_scale_inv": z_scale,
    }
    mapper.finalize_update_weights()


def test_qwen35_partial_bf16_qkvz_does_not_wait_for_scales():
    mapper = Qwen3_5MoeHfWeightMapper()
    prefix = "model.layers.0.linear_attn.in_proj_"
    qkv_weight = torch.empty(4, 2, dtype=torch.bfloat16)
    z_weight = torch.empty(2, 2, dtype=torch.bfloat16)

    mapper.begin_update_weights()
    assert not mapper._stage_partial_split_projections(
        {f"{prefix}qkv.weight": qkv_weight}, QuantAlgo.FP8_BLOCK_SCALES)
    ready = mapper._stage_partial_split_projections(
        {f"{prefix}z.weight": z_weight}, QuantAlgo.FP8_BLOCK_SCALES)

    assert ready == {
        f"{prefix}qkv.weight": qkv_weight,
        f"{prefix}z.weight": z_weight,
    }
    mapper.finalize_update_weights()
