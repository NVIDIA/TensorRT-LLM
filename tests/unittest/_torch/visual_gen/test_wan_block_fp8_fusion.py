# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WanBlock static-FP8 fusions against the same block with the fusions off."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.config import (
    DiffusionModelConfig,
    create_attention_metadata_state,
)
from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanBlock
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() < 89, reason="static FP8 GEMMs need SM89+"
)

E4M3 = torch.float8_e4m3fn
E4M3_MAX = 448.0
HEADS, HEAD_DIM, FFN = 24, 128, 8192
HIDDEN = HEADS * HEAD_DIM
INPUT_SCALE = 8.0 / E4M3_MAX
WEIGHT_SCALE = 0.1 / E4M3_MAX


def _cosine(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _config(backend, quant_attention, fp8=True):
    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(
            num_attention_heads=HEADS,
            attention_head_dim=HEAD_DIM,
            ffn_dim=FFN,
            eps=1e-6,
            cross_attn_norm=True,
        ),
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8) if fp8 else QuantConfig(),
        attention=AttentionConfig(backend=backend, quant_attention_config=quant_attention),
    )
    config.attention_metadata_state = create_attention_metadata_state()
    return config


def _block(config):
    block = WanBlock(config, _layer_idx=0).cuda()
    g = torch.Generator(device="cuda").manual_seed(0)

    def randn(shape, scale):
        return torch.randn(shape, device="cuda", generator=g) * scale

    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, Linear):
                w = randn(module.weight.shape, 0.02)
                if module.weight.dtype == E4M3:
                    module.weight.copy_((w / WEIGHT_SCALE).to(E4M3))
                    module.weight_scale.fill_(WEIGHT_SCALE)
                    module.input_scale.fill_(INPUT_SCALE)
                    module.inv_input_scale.fill_(1.0 / INPUT_SCALE)
                else:
                    module.weight.copy_(w)
                if module.bias is not None:
                    module.bias.copy_(randn(module.bias.shape, 0.02))
        for name, p in block.named_parameters():
            if "norm" in name and p.dtype != E4M3:
                p.copy_(
                    1.0 + randn(p.shape, 0.1) if name.endswith("weight") else randn(p.shape, 0.1)
                )
        block.scale_shift_table.copy_(randn(block.scale_shift_table.shape, 0.1))
    for module in block.modules():
        if isinstance(module, Linear):
            module.post_load_weights()
    return block


def _inputs(batch, seq, text_len=64):
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(batch, seq, HIDDEN, device="cuda", generator=g).bfloat16()
    text = torch.randn(batch, text_len, HIDDEN, device="cuda", generator=g).bfloat16()
    temb = (torch.randn(batch, 6, HIDDEN, device="cuda", generator=g) * 0.1).bfloat16()
    pos = torch.arange(seq, dtype=torch.float64)
    inv = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM))
    angles = torch.outer(pos, inv)
    cos = angles.cos().repeat_interleave(2, dim=-1).float().view(1, seq, 1, HEAD_DIM).cuda()
    sin = angles.sin().repeat_interleave(2, dim=-1).float().view(1, seq, 1, HEAD_DIM).cuda()
    return x, text, temb, cos, sin


def _fusions_off(block):
    block._fp8_block = False
    block.attn1.qkv_proj._fp8_fuse_bias = False
    block.attn2.to_q._fp8_fuse_bias = False
    block.attn1._fp8_out_scale = None
    block.ffn._use_fp8_gelu_fusion = False


@pytest.mark.parametrize(
    "backend,quant_attention",
    [
        ("VANILLA", None),
        pytest.param(
            "TRTLLM",
            QuantAttentionConfig(qk_dtype="fp8", v_dtype="fp8"),
            marks=pytest.mark.skipif(
                get_sm_version() != 107, reason="TRTLLM dense FP8 attention runs on SM107"
            ),
        ),
    ],
)
@pytest.mark.parametrize("batch", [1, 2])
def test_fused_block_matches_unfused(backend, quant_attention, batch):
    block = _block(_config(backend, quant_attention))
    assert block.plan_fp8_fusions()
    assert block._fp8_block and block.ffn._use_fp8_gelu_fusion
    fp8_attention_out = block.attn1._fp8_qkv_producer
    assert (block.attn1._fp8_out_scale is not None) == fp8_attention_out

    x, text, temb, cos, sin = _inputs(batch, 1560)
    with torch.no_grad():
        fused = block(x, text, temb, cos, sin)
        _fusions_off(block)
        reference = block(x, text, temb, cos, sin)
    assert fused.shape == reference.shape and fused.dtype == torch.bfloat16
    assert _cosine(fused, reference) > 0.999


def test_per_token_temb_takes_the_unfused_path():
    block = _block(_config("VANILLA", None))
    assert block.plan_fp8_fusions()
    x, text, temb, cos, sin = _inputs(1, 256)
    temb_per_token = temb[:, None].expand(1, 256, 6, HIDDEN).contiguous()
    with torch.no_grad():
        fused_plan = block(x, text, temb_per_token, cos, sin)
        _fusions_off(block)
        reference = block(x, text, temb_per_token, cos, sin)
    assert _cosine(fused_plan, reference) > 0.999


def test_bf16_block_is_not_planned():
    block = _block(_config("VANILLA", None, fp8=False))
    assert not block.plan_fp8_fusions(log=True)
    assert not block._fp8_block
