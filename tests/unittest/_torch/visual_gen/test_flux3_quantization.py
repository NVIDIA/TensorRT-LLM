# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from flux3_test_utils import assert_parity

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.flux3.fp8 import quantize_fp8r
from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import (
    Flux3ModeBlock,
    _remap_checkpoint_keys,
    _split_mlp_in_weights,
    _use_fused_proj_out,
    _use_fused_qkv,
)
from tensorrt_llm._torch.visual_gen.quantization.loader import DynamicLinearWeightLoader
from tensorrt_llm._torch.visual_gen.quantization.ops import quantize_nvfp4
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.quantization.utils import fp4_utils


def test_topology_aware_checkpoint_key_remapping():
    weights = {
        "blocks.0.q_proj.weight": torch.empty(1),
        "blocks.0.k_proj.weight": torch.empty(1),
        "blocks.0.v_proj.weight": torch.empty(1),
        "blocks.0.attn_out.weight": torch.empty(1),
        "blocks.0.mlp_out.weight": torch.empty(1),
    }
    separate = _remap_checkpoint_keys(weights, fuse_qkv=False)
    assert "blocks.0.attn.to_q.weight" in separate
    assert "blocks.0.attn.to_k.weight" in separate
    assert "blocks.0.attn.to_v.weight" in separate
    assert "blocks.0.attn_out.weight" in separate
    assert "blocks.0.ff.down_proj.weight" in separate


def test_tp_mlp_split_preserves_native_nvfp4_tensors():
    weights = {
        "blocks.0.mlp_in.weight": torch.arange(16).reshape(4, 4),
        "blocks.0.mlp_in.weight_scale": torch.arange(8).reshape(4, 2),
        "blocks.0.mlp_in.weight_scale_2": torch.tensor(0.5),
        "blocks.0.mlp_in.input_scale": torch.tensor(0.25),
    }
    split = _split_mlp_in_weights(weights)
    torch.testing.assert_close(
        split["blocks.0.ff.gate_up_proj.gate.weight"], weights["blocks.0.mlp_in.weight"][:2]
    )
    torch.testing.assert_close(
        split["blocks.0.ff.gate_up_proj.up.weight"], weights["blocks.0.mlp_in.weight"][2:]
    )
    torch.testing.assert_close(
        split["blocks.0.ff.gate_up_proj.gate.weight_scale"],
        weights["blocks.0.mlp_in.weight_scale"][:2],
    )
    assert (
        split["blocks.0.ff.gate_up_proj.gate.weight_scale_2"]
        is weights["blocks.0.mlp_in.weight_scale_2"]
    )
    assert (
        split["blocks.0.ff.gate_up_proj.up.input_scale"] is weights["blocks.0.mlp_in.input_scale"]
    )


def test_static_nvfp4_uses_unfused_projections():
    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(),
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
    )
    assert not _use_fused_qkv(config)
    assert not _use_fused_proj_out(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 layout ops require CUDA")
def test_native_nvfp4_loader_swizzles_row_major_scale():
    torch.manual_seed(2)
    weight = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    qweight, linear_scale, scale_2 = quantize_nvfp4(weight)
    swizzled = torch.ops.trtllm.block_scale_interleave(linear_scale.view(fp4_utils.float4_sf_dtype))

    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    model_config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(), quant_config=quant_config
    )
    linear = Flux3ModeBlock(128, 2, model_config, mlp_ratio=1).attn_out
    checkpoint_weights = {
        "attn_out.weight": qweight.cpu(),
        "attn_out.weight_scale": linear_scale.cpu(),
        "attn_out.weight_scale_2": scale_2.cpu(),
    }
    loader = DynamicLinearWeightLoader(model_config)
    loader.load_linear_weights(
        linear,
        "attn_out",
        [
            {
                "weight": checkpoint_weights["attn_out.weight"],
                "weight_scale": checkpoint_weights["attn_out.weight_scale"],
                "weight_scale_2": checkpoint_weights["attn_out.weight_scale_2"],
            }
        ],
    )

    assert torch.equal(linear.weight.cpu(), qweight.cpu())
    assert torch.equal(linear.weight_scale.cpu(), swizzled.view(torch.uint8).cpu().reshape(-1))


@pytest.mark.parametrize("layout", ["small", "warp", "strided"])
@pytest.mark.parametrize("compile_model", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FP8 quantization requires CUDA")
def test_fp8r_activation_recipe(layout, compile_model):
    """L1 T1: FLUX.3 CUDA quantization versus BFL's FP32 recipe, eager and compiled."""
    torch.manual_seed(42)
    tokens = (
        8 if layout == "small" else torch.cuda.get_device_properties(0).multi_processor_count * 16
    )
    x = torch.randn(2, tokens, 256, device="cuda", dtype=torch.bfloat16)
    if layout == "strided":
        x = x[..., ::2]
    x[:, 0].zero_()
    x[:, 1].mul_(1e-8)
    rows = x.float().reshape(-1, x.shape[-1])
    scales = (rows.abs().amax(dim=-1) / 448).clamp_min(1e-12)
    expected = (rows / scales[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    run = torch.compile(quantize_fp8r, fullgraph=True) if compile_model else quantize_fp8r
    actual, actual_scales = run(x)
    assert actual.dtype == torch.float8_e4m3fn
    assert actual_scales.dtype == torch.float32
    assert actual.shape == expected.shape
    assert actual_scales.shape == scales.shape
    torch.testing.assert_close(actual_scales, scales, rtol=1e-6, atol=0)
    decoded = actual.float() * actual_scales[:, None]
    reference = expected.float() * scales[:, None]
    for indices in ([0, tokens], [1, tokens + 1], slice(None)):
        assert_parity(decoded[indices], reference[indices])
