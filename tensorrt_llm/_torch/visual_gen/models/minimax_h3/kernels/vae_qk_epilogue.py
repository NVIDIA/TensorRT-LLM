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
"""MiniMax-H3 ViT VAE decoder Q|K projection with bias, per-head RMSNorm and 3-axis RoPE in its GEMM epilogue.

QuACK SM100 epilogue: per-head RMS statistics of the biased accumulator, then the angle
pos_t * f_t + pos_h * f_h + pos_w * f_w per rotation pair in turns with fast-math sin/cos. The decoder has
head_dim 64, fp16, no norm weight, bias on the q and k projections, and RoPE on the first 48 dims of each head;
those rotate-half pairs (i, i + 24) are stored adjacent (pairs 0..7 rotate by t, 8..15 by h, 16..23 by w).

``install_vae_qk_epilogue`` sets this up on a decoder: each attention's q and k weights become row views of one
merged, pair-permuted ``[2 * inner, dim]`` weight, the decoder's rope module also returns the per-token positions,
and each attention's processor takes q and k from ``vae_qk``. q and k reach the attention with their head dims in
the permuted order, which leaves the attention scores unchanged.
"""

import math
from typing import Optional

import cutlass.cute as cute
import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
    MiniMaxH3VideoAttention,
    MiniMaxH3VideoAttnProcessor,
    MiniMaxH3VideoRotaryPosEmbed,
)
from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.head_rmsnorm import HeadRstd
from quack.epilogue.math import pack, unpack
from quack.epilogue.ops import ColVecLoad, RowVecLoad
from quack.epilogue.rotary import _angle_turns, _sincos_turns

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from .qk_epilogue import mrope_freqs, pair_permutation

_CONFIG = {
    "tile_M": 256,
    "tile_N": 256,
    "cluster_M": 2,
    "cluster_N": 1,
    "max_swizzle_size": 16,
    "is_dynamic_persistent": True,
}
_EPS = 1e-5


def _biased_sq_prepass(acc, b):
    y = acc + b
    return {"qk": y * y}


@gemm_epilogue(
    ops={
        "qk": HeadRstd("qk", eps=_EPS),
        "b": RowVecLoad("b"),
        "pos_t": ColVecLoad("pos_t"),
        "pos_h": ColVecLoad("pos_h"),
        "pos_w": ColVecLoad("pos_w"),
        "freq_t": RowVecLoad("freq_t"),
        "freq_h": RowVecLoad("freq_h"),
        "freq_w": RowVecLoad("freq_w"),
    },
    prepass=_biased_sq_prepass,
    prepass_outs=("qk",),
    mode="acc_pair",
)
def vae_qk_rope_epi(acc, qk, b, pos_t, pos_h, pos_w, freq_t, freq_h, freq_w):
    x1, x2 = unpack((acc + b) * qk)
    t1, lo1 = _angle_turns(pos_t, freq_t)
    t2, lo2 = _angle_turns(pos_h, freq_h)
    t3, lo3 = _angle_turns(pos_w, freq_w)
    t = cute.math.fma(t3, 1.0, cute.math.fma(t1, 1.0, t2))
    s, c = _sincos_turns(t, (lo1 + lo2) + lo3)
    return {"D": pack(x1 * c - x2 * s, x1 * s + x2 * c)}


def vae_freqs(inv_freq: torch.Tensor, qk_cols: int, head_dim: int) -> tuple[torch.Tensor, ...]:
    """(freq_t, freq_h, freq_w) tables for the VAE angle 2 * pi * pos * inv_freq (one turn per unit of pos * inv_freq).

    ``inv_freq`` is ``MiniMaxH3VideoRotaryPosEmbed.inv_freq`` (per-axis, rope_dim / 6 values)."""
    return mrope_freqs(inv_freq.double() * (2.0 * math.pi), qk_cols, head_dim=head_dim)


@torch.library.custom_op(
    "trtllm::minimax_h3_vae_qk_gemm_norm_rope",
    mutates_args=(),
    tags=(torch.Tag.needs_fixed_stride_order,),
)
def vae_qk_gemm_norm_rope(
    x: torch.Tensor,
    qk_weight: torch.Tensor,
    qk_bias: torch.Tensor,
    head_dim: int,
    pos_t: torch.Tensor,
    pos_h: torch.Tensor,
    pos_w: torch.Tensor,
    freq_t: torch.Tensor,
    freq_h: torch.Tensor,
    freq_w: torch.Tensor,
) -> torch.Tensor:
    """x: fp16 [..., K] contiguous; qk_weight: fp16 [Nqk, K] and qk_bias fp16 [Nqk] with each head's dims
    pair-permuted; positions fp32 [M] for the M = x.numel() // K rows; freqs from ``vae_freqs``. Returns fp16
    [..., Nqk] in the pair-permuted head layout."""
    out = torch.empty((*x.shape[:-1], qk_weight.shape[0]), dtype=x.dtype, device=x.device)
    vae_qk_rope_epi.gemm(
        x.view(-1, x.shape[-1]),
        qk_weight,
        out.view(-1, qk_weight.shape[0]),
        epi_args=dict(
            qk=head_dim,
            b=qk_bias,
            pos_t=pos_t,
            pos_h=pos_h,
            pos_w=pos_w,
            freq_t=freq_t,
            freq_h=freq_h,
            freq_w=freq_w,
        ),
        **_CONFIG,
    )
    return out


@vae_qk_gemm_norm_rope.register_fake
def _(x, qk_weight, qk_bias, head_dim, pos_t, pos_h, pos_w, freq_t, freq_h, freq_w):
    return x.new_empty((*x.shape[:-1], qk_weight.shape[0]))


class MiniMaxH3VideoRotaryPosEmbedWithPositions(MiniMaxH3VideoRotaryPosEmbed):
    """Returns ``(cos, sin, pos_t, pos_h, pos_w)``: the rotary tables and each token's positions, fp32 ``[B * S]``."""

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        cos, sin = super().forward(position_ids)
        pos = position_ids.to(torch.float32)
        return (
            cos,
            sin,
            *(pos[..., axis].reshape(-1).contiguous() for axis in range(pos.shape[-1])),
        )


def vae_qk(
    attn: MiniMaxH3VideoAttention,
    hidden_states: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """q and k ``[B, S, H, D]`` of an attention prepared by ``prepare_vae_qk_epilogue``, head dims pair-permuted;
    ``rotary_emb`` is the output of ``MiniMaxH3VideoRotaryPosEmbedWithPositions``."""
    inner = attn.heads * attn.dim_head
    weight = attn.h3_qk_weight
    qk = torch.ops.trtllm.minimax_h3_vae_qk_gemm_norm_rope(
        hidden_states.to(weight.dtype),
        weight,
        attn.h3_qk_bias,
        attn.dim_head,
        *rotary_emb[2:5],
        *attn.h3_qk_freqs,
    )
    return qk[..., :inner].unflatten(2, (attn.heads, -1)), qk[..., inner:].unflatten(
        2, (attn.heads, -1)
    )


def _eligible(attn: torch.nn.Module, rope_dim: int) -> bool:
    if (
        not isinstance(attn, MiniMaxH3VideoAttention)
        or type(attn.processor) is not MiniMaxH3VideoAttnProcessor
    ):
        return False
    projs = (attn.to_q, attn.to_k)
    if any(
        proj.weight.dtype != torch.float16 or proj.bias is None or not proj.weight.is_cuda
        for proj in projs
    ):
        return False
    norms = (attn.norm_q, attn.norm_k)
    if any(
        not isinstance(n, torch.nn.RMSNorm) or n.weight is not None or n.eps != _EPS for n in norms
    ):
        return False
    inner = attn.heads * attn.dim_head
    return (
        attn.dim_head == 64
        and rope_dim <= attn.dim_head
        and (2 * inner) % _CONFIG["tile_N"] == 0
        and attn.to_q.weight.shape == attn.to_k.weight.shape == (inner, attn.to_q.in_features)
    )


@torch.no_grad()
def _merge_qk(attn: MiniMaxH3VideoAttention, perm: torch.Tensor, inv_freq: torch.Tensor) -> None:
    """Row-view ``to_q``/``to_k`` into one merged weight and bias with each head's dims pair-permuted."""
    inner = attn.heads * attn.dim_head

    def permuted(t: torch.Tensor) -> torch.Tensor:
        return t.view(attn.heads, attn.dim_head, *t.shape[1:])[:, perm].reshape(t.shape)

    weight = torch.cat([permuted(attn.to_q.weight), permuted(attn.to_k.weight)]).contiguous()
    bias = torch.cat([permuted(attn.to_q.bias), permuted(attn.to_k.bias)]).contiguous()
    # Plain tensors: registered, they would alias to_q/to_k at a different size, which CPU offloading rejects.
    attn.h3_qk_weight = weight
    attn.h3_qk_bias = bias
    for lin, w, b in zip((attn.to_q, attn.to_k), weight.split(inner), bias.split(inner)):
        lin.weight = torch.nn.Parameter(w, requires_grad=False)
        lin.bias = torch.nn.Parameter(b, requires_grad=False)
    attn.h3_qk_freqs = vae_freqs(inv_freq.to(weight.device), 2 * inner, attn.dim_head)


def prepare_vae_qk_epilogue(decoder: torch.nn.Module) -> Optional[list[int]]:
    """Prepare every attention of ``decoder`` for ``vae_qk`` when all of them and its rope module qualify.

    Returns the head-dim permutation of q and k (stored dim ``i`` holds dim ``perm[i]``), or None when nothing was
    changed. Once prepared, the attentions' processor must take q and k from ``vae_qk``.
    """
    rope = getattr(decoder, "rope", None)
    if type(rope) is not MiniMaxH3VideoRotaryPosEmbed or get_sm_version() not in (100, 103):
        return None
    rope_dim = 6 * rope.inv_freq.numel()
    attns = [m for m in decoder.modules() if isinstance(m, MiniMaxH3VideoAttention)]
    if not attns or not all(_eligible(a, rope_dim) for a in attns):
        return None
    perm = pair_permutation(attns[0].dim_head, rope_dim, device=attns[0].to_q.weight.device)
    for attn in attns:
        _merge_qk(attn, perm, rope.inv_freq)
    rope.__class__ = MiniMaxH3VideoRotaryPosEmbedWithPositions
    return perm.tolist()


class MiniMaxH3VideoQKEpilogueAttnProcessor(MiniMaxH3VideoAttnProcessor):
    """``MiniMaxH3VideoAttnProcessor`` with q and k from ``vae_qk``."""

    def __call__(
        self,
        attn: MiniMaxH3VideoAttention,
        hidden_states: torch.Tensor,
        rotary_emb: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:
        # One fp16 cast feeds both the epilogue GEMM and to_v.
        hidden_states = hidden_states.to(attn.h3_qk_weight.dtype)
        query, key = vae_qk(attn, hidden_states, rotary_emb)
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        return attn.to_out[0](hidden_states.flatten(2, 3))


def install_vae_qk_epilogue(decoder: torch.nn.Module) -> int:
    """Run every decoder attention's q/k projection, RMSNorm and RoPE in one epilogue GEMM; returns how many.

    Needs SM100/SM103 and every attention still on Diffusers' processor. The decoder runs under the fp16
    autocast of ``MiniMaxH3Pipeline._decode_video``.
    """
    if not torch.cuda.is_available() or prepare_vae_qk_epilogue(decoder) is None:
        return 0
    attns = [m for m in decoder.modules() if isinstance(m, MiniMaxH3VideoAttention)]
    for attn in attns:
        attn.set_processor(MiniMaxH3VideoQKEpilogueAttnProcessor())
    logger.info(
        f"MiniMax-H3 VAE decoder: q/k RMSNorm + RoPE in the projection epilogue on {len(attns)} blocks"
    )
    return len(attns)
