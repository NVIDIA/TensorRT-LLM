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
"""Tests for the MiniMax-H3 ViT VAE decoder kernels: the q|k GEMM with RMSNorm + RoPE in its epilogue, the
attention processor that takes q and k from it, and the FFN SwiGLU in its projection's GEMM epilogue."""

import copy

import pytest
import torch
import torch.nn.functional as F

vae_module = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_minimax_h3")
from diffusers.models.activations import SwiGLU  # noqa: E402

import tensorrt_llm._torch.visual_gen.models.minimax_h3.kernels.vae_swiglu_quack as vae_swiglu  # noqa: E402
from tensorrt_llm._utils import get_sm_version  # noqa: E402

requires_sm100 = pytest.mark.skipif(
    not (torch.cuda.is_available() and get_sm_version() in (100, 103)),
    reason="needs an SM100/SM103 GPU (QuACK SM100 epilogue GEMMs)",
)

BATCH, SEQ, HEADS, HEAD_DIM = 2, 301, 4, 64
EPS = 1e-5


def _decoder() -> torch.nn.Module:
    """Two-block decoder at the production head_dim, its Linear layers in fp16 as the pipeline decodes."""
    torch.manual_seed(0)
    decoder = (
        vae_module.MiniMaxH3VideoViTDecoder3d(
            in_channels=4,
            patch_size=2,
            patch_size_t=1,
            num_layers=2,
            num_attention_heads=HEADS,
            attention_head_dim=HEAD_DIM,
            ffn_mult=2,
        )
        .cuda()
        .eval()
    )
    for module in decoder.modules():
        if isinstance(module, torch.nn.Linear):
            module.to(torch.float16)
    return decoder


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Positions in the decoder's normalized [-1, 1) range and a unit-scale fp32 block input."""
    torch.manual_seed(1)
    position_ids = torch.rand((BATCH, SEQ, 3), device="cuda") * 2 - 1
    hidden_states = torch.randn((BATCH, SEQ, HEADS * HEAD_DIM), device="cuda")
    return position_ids, hidden_states


def _norm_rope(y: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + EPS)
    rotary_dim = cos.shape[-1]
    first, second = y[..., :rotary_dim].chunk(2, dim=-1)
    rotated = torch.cat([-second, first], dim=-1)
    return torch.cat([y[..., :rotary_dim] * cos + rotated * sin, y[..., rotary_dim:]], dim=-1)


def _reference(attn, hidden_states, cos, sin):
    """fp64 q, k and output of Diffusers' attention math on the fp16 input every path sees."""
    x = hidden_states.half().double()
    cos, sin = cos.double(), sin.double()

    def project(linear):
        y = F.linear(x, linear.weight.double(), linear.bias.double())
        return y.unflatten(2, (attn.heads, -1))

    q = _norm_rope(project(attn.to_q), cos, sin)
    k = _norm_rope(project(attn.to_k), cos, sin)
    scores = torch.einsum("bshd,bthd->bhst", q, k) * attn.dim_head**-0.5
    out = torch.einsum("bhst,bthd->bshd", scores.softmax(-1), project(attn.to_v)).flatten(2)
    out = F.linear(out, attn.to_out[0].weight.double(), attn.to_out[0].bias.double())
    return q, k, out


@requires_sm100
def test_vae_qk_epilogue_matches_fp64_reference():
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.kernels.vae_qk_epilogue import (
        prepare_vae_qk_epilogue,
        vae_qk,
    )

    decoder = _decoder()
    stock = copy.deepcopy(decoder)
    perm = prepare_vae_qk_epilogue(decoder)
    assert perm is not None
    position_ids, hidden_states = _inputs()
    with torch.inference_mode():
        q, k = vae_qk(decoder.transformer_blocks[0].attn, hidden_states, decoder.rope(position_ids))
        ref_q, ref_k, _ = _reference(
            stock.transformer_blocks[0].attn, hidden_states, *stock.rope(position_ids)
        )
    for actual, expected in ((q, ref_q[..., perm]), (k, ref_k[..., perm])):
        assert actual.dtype == torch.float16 and actual.shape == expected.shape
        # One fp16 rounding of the fp32 result (half an ulp); MUFU sin/cos and the fp32 accumulation add ~1e-6.
        assert (actual.double() - expected).abs().max() <= expected.abs().max() * 2**-10


@requires_sm100
def test_decoder_attention_matches_stock_processor():
    """Each decoder attention with q/k from the epilogue GEMM against Diffusers' processor under the
    pipeline's fp16 autocast."""
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.kernels.vae_qk_epilogue import (
        MiniMaxH3VideoQKEpilogueAttnProcessor,
        install_vae_qk_epilogue,
    )

    decoder = _decoder()
    stock = copy.deepcopy(decoder)
    attns = [block.attn for block in decoder.transformer_blocks]
    assert install_vae_qk_epilogue(decoder) == len(attns)
    assert all(type(attn.processor) is MiniMaxH3VideoQKEpilogueAttnProcessor for attn in attns)
    position_ids, hidden_states = _inputs()
    stock_attns = [block.attn for block in stock.transformer_blocks]
    with torch.inference_mode():
        with torch.autocast("cuda", torch.float16):
            rotary, stock_rotary = decoder.rope(position_ids), stock.rope(position_ids)
            actual = [attn(hidden_states, rotary) for attn in attns]
            expected = [attn(hidden_states, stock_rotary) for attn in stock_attns]
        oracle = [_reference(attn, hidden_states, *stock_rotary)[2] for attn in stock_attns]
    # Both paths round to fp16 at v, inside the attention and at to_out; the epilogue rounds q and k once, so it
    # stays within the stock path's own fp16 error.
    for out, stock_out, ref in zip(actual, expected, oracle):
        err = (out.double() - ref).abs()
        stock_err = (stock_out.double() - ref).abs()
        assert err.norm() <= 1.25 * stock_err.norm()
        assert err.max() <= 2 * stock_err.max()


@requires_sm100
@pytest.mark.parametrize(
    ("dim", "inner", "bias"),
    [(256, 512, True), (264, 200, False)],
    ids=["bias", "no-bias-partial-tiles"],
)
def test_quack_swiglu_matches_diffusers_swiglu(dim, inner, bias):
    torch.manual_seed(0)
    stock = SwiGLU(dim, inner, bias=bias).cuda().half()
    fused = copy.deepcopy(stock)
    assert vae_swiglu.fuse_vae_swiglu(fused) == 1
    assert type(fused) is vae_swiglu.QuackSwiGLU
    hidden_states = torch.randn((BATCH, SEQ, dim), device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", torch.float16):
        actual = fused(hidden_states)
        expected = stock(hidden_states)
    assert actual.dtype == torch.float16 and actual.shape == expected.shape == (BATCH, SEQ, inner)
    proj_bias = None if stock.proj.bias is None else stock.proj.bias.double()
    hidden, gate = F.linear(
        hidden_states.half().double(), stock.proj.weight.double(), proj_bias
    ).chunk(2, dim=-1)
    oracle = hidden * F.silu(gate)
    err = (actual.double() - oracle).abs()
    # One fp16 rounding of the fp32 epilogue result (half an ulp; the approximate exp and reciprocal of its sigmoid
    # add ~1e-7 relative); Diffusers' path rounds after the projection, the SiLU and the product.
    assert err.max() <= oracle.abs().max() * 2**-10
    assert err.norm() <= (expected.double() - oracle).norm()
