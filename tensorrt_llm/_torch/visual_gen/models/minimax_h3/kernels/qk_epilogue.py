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
"""MiniMax-H3 Q|K projection with per-head RMSNorm and 3-axis RoPE in its GEMM epilogue (QuACK, SM100).

The Q/K head dimensions are stored pair-permuted so that each RoPE pair (i, i + rotary_dim / 2) is adjacent
(QuACK's ``acc_pair`` epilogue mode). Pairs 0..15 rotate by the t position, 16..31 by h and 32..47 by w, each
with H3's 16 inverse frequencies; the remaining pairs pass through. Angles are float-float sums in turns with
fast-math sin/cos.
"""

import cutlass.cute as cute
import torch
import torch.nn.functional as F
from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.head_rmsnorm import HeadRstd
from quack.epilogue.math import pack, unpack
from quack.epilogue.ops import ColVecLoad, RowVecLoad
from quack.epilogue.rotary import _angle_turns, _sincos_turns, make_mrope_inv_freq

# The per-head RMSNorm eps is compiled into the epilogue.
QK_EPILOGUE_EPS = 1e-5
_CONFIG = {"tile_M": 256, "tile_N": 256, "cluster_M": 2, "cluster_N": 1}


def _sq_prepass(acc):
    return {"qk": acc * acc}


@gemm_epilogue(
    ops={
        "qk": HeadRstd("qk", eps=QK_EPILOGUE_EPS),
        "w": RowVecLoad("w"),
        "pos_t": ColVecLoad("pos_t"),
        "pos_h": ColVecLoad("pos_h"),
        "pos_w": ColVecLoad("pos_w"),
        "freq_t": RowVecLoad("freq_t"),
        "freq_h": RowVecLoad("freq_h"),
        "freq_w": RowVecLoad("freq_w"),
    },
    prepass=_sq_prepass,
    prepass_outs=("qk",),
    mode="acc_pair",
)
def h3_qk_mrope_epi(acc, qk, w, pos_t, pos_h, pos_w, freq_t, freq_h, freq_w):
    x1, x2 = unpack(acc * qk * w)
    t1, lo1 = _angle_turns(pos_t, freq_t)
    t2, lo2 = _angle_turns(pos_h, freq_h)
    t3, lo3 = _angle_turns(pos_w, freq_w)
    t = cute.math.fma(t3, 1.0, cute.math.fma(t1, 1.0, t2))
    s, c = _sincos_turns(t, (lo1 + lo2) + lo3)
    return {"D": pack(x1 * c - x2 * s, x1 * s + x2 * c)}


def pair_permutation(head_dim: int, rotary_dim: int, device=None) -> torch.Tensor:
    """Source column of each permuted column: 2i <- i, 2i + 1 <- i + rotary_dim / 2; the tail keeps its order."""
    half = rotary_dim // 2
    first = torch.arange(half, device=device)
    rotary = torch.stack((first, first + half), dim=1).reshape(-1)
    return torch.cat((rotary, torch.arange(rotary_dim, head_dim, device=device)))


def mrope_freqs(
    inv_freq: torch.Tensor, qk_cols: int, head_dim: int = 128
) -> tuple[torch.Tensor, ...]:
    """``(freq_t, freq_h, freq_w)``: fp32 ``[qk_cols]`` float-float tables in turns for the permuted Q|K columns.

    ``inv_freq`` is the per-axis ``[rope_freq_dim]`` inverse frequency.
    """
    n = inv_freq.shape[0]
    per_pair = torch.cat((inv_freq, inv_freq, inv_freq, inv_freq.new_zeros(head_dim // 2 - 3 * n)))
    tables = make_mrope_inv_freq(per_pair.double(), (n, n, head_dim // 2 - 2 * n), qk_cols)
    return tuple(t.contiguous() for t in tables)


def norm_rowvec(
    q_weight: torch.Tensor, k_weight: torch.Tensor, q_heads: int, k_heads: int
) -> torch.Tensor:
    """fp32 ``[(q_heads + k_heads) * head_dim]``: each head's norm weight repeated over its heads."""
    return torch.cat(
        (q_weight.float().repeat(q_heads), k_weight.float().repeat(k_heads))
    ).contiguous()


@torch.library.custom_op("trtllm::minimax_h3_qk_gemm_norm_mrope", mutates_args=())
def qk_gemm_norm_mrope(
    x: torch.Tensor,
    qk_weight: torch.Tensor,
    rowvec: torch.Tensor,
    head_weight: torch.Tensor,
    pos_t: torch.Tensor,
    pos_h: torch.Tensor,
    pos_w: torch.Tensor,
    freq_t: torch.Tensor,
    freq_h: torch.Tensor,
    freq_w: torch.Tensor,
) -> torch.Tensor:
    """Normalized and rotated ``x @ qk_weight^T``.

    Args:
        x: bf16 ``[S, K]``.
        qk_weight: bf16 ``[Nqk, K]``, the pair-permuted Q|K rows.
        rowvec: ``norm_rowvec`` of the pair-permuted norm weights.
        head_weight: any ``[head_dim]`` tensor; its length is the head width.
        pos_t, pos_h, pos_w: fp32 ``[S]`` positions.
        freq_t, freq_h, freq_w: ``mrope_freqs``.

    Returns:
        bf16 ``[S, Nqk]``.
    """
    out = torch.empty((x.shape[0], qk_weight.shape[0]), dtype=x.dtype, device=x.device)
    h3_qk_mrope_epi.gemm(
        x,
        qk_weight,
        out,
        epi_args=dict(
            qk=head_weight,
            w=rowvec,
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


@qk_gemm_norm_mrope.register_fake
def _(x, qk_weight, rowvec, head_weight, pos_t, pos_h, pos_w, freq_t, freq_h, freq_w):
    return x.new_empty((x.shape[0], qk_weight.shape[0]))


class MiniMaxH3QKEpilogue:
    """One attention's Q|K epilogue GEMM: ``(x [S, K], qkv_weight, (pos_t, pos_h, pos_w)) -> (q, k, v)``.

    Construction pair-permutes the attention's Q/K weight rows and norm weights in place; ``unpermute``
    restores the split-half head layout on the unfused path. V comes from its own GEMM. Its tables are built on
    ``device``, the GPU the attention runs on; with CPU offloading the weights are still on the host at construction.
    """

    def __init__(self, attn, inv_freq: torch.Tensor, rotary_dim: int, device: torch.device) -> None:
        weight = attn.qkv_proj.weight
        perm = pair_permutation(attn.head_dim, rotary_dim, device=weight.device)
        qk_rows = 2 * attn.local_q_dim
        weight[:qk_rows] = (
            weight[:qk_rows].view(-1, attn.head_dim, weight.shape[-1])[:, perm].reshape(qk_rows, -1)
        )
        attn.norm_q.weight.copy_(attn.norm_q.weight[perm])
        attn.norm_k.weight.copy_(attn.norm_k.weight[perm])
        self.unpermute = torch.argsort(perm).to(device)
        self.q_cols = attn.local_q_dim
        self.qk_cols = qk_rows
        self.head_weight = attn.norm_q.weight.detach().to(device)
        self.rowvec = norm_rowvec(
            attn.norm_q.weight,
            attn.norm_k.weight,
            attn.local_num_attention_heads,
            attn.local_num_key_value_heads,
        ).to(device)
        self.freqs = tuple(t.to(device) for t in mrope_freqs(inv_freq, qk_rows))

    def __call__(
        self, x: torch.Tensor, qkv_weight: torch.Tensor, positions: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qk = torch.ops.trtllm.minimax_h3_qk_gemm_norm_mrope(
            x, qkv_weight[: self.qk_cols], self.rowvec, self.head_weight, *positions, *self.freqs
        )
        return qk[:, : self.q_cols], qk[:, self.q_cols :], F.linear(x, qkv_weight[self.qk_cols :])
