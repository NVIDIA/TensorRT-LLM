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
"""Numerics for the Q RoPE fold in `deepseek_v4_q_norm_fused_fp8`.

The norm + FP8 quantize half of this op already existed; what is new is that
passing `rotary_cos_sin` makes it also rotate the rope tail and write it FP8
into the same row, so `applyMLARopeAndAssignQKVKernelGeneration` is not launched
on the DSv4 path at all. Nothing else covers that: the sparse-MLA backend tests
build a bf16 KV cache, which turns the fusion off, and GSM8K only sees the end
result.

The two position rules are separate code -- generation divides by a uniform
query length, context binary-searches `cu_q_seqlens` and adds the chunked-prefill
cached offset -- and they select different vectorizations of the kernel
(generation wide, context narrow), so both are exercised here.
"""

import pytest
import torch
from utils.util import skip_pre_blackwell

# DSv4-Pro Q geometry: 448 nope + 64 rope per head.
HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = HEAD_DIM - NOPE_DIM
EPS = 1e-6
QUANT_SCALE = 0.5
MAX_POSITIONS = 512


def _make_cos_sin(device: torch.device) -> torch.Tensor:
    """Rope table in the layout every MLA kernel here indexes.

    The pointer is `float2 const*` strided by ROPE_DIM per position, so a row is
    ROPE_DIM float2 entries even though only the first ROPE_DIM/2 -- one per
    rotated pair -- are ever read. Filling the unused tail with NaN keeps a
    stride mistake from silently reading plausible numbers.
    """
    table = torch.full((MAX_POSITIONS, ROPE_DIM, 2), float("nan"), dtype=torch.float32)
    # Angles are decorrelated across positions on purpose. A smooth table (e.g.
    # linspace) makes neighbouring positions nearly identical, and an off-by-one
    # in the position arithmetic then lands inside any sane tolerance.
    generator = torch.Generator().manual_seed(1234)
    angles = torch.rand((MAX_POSITIONS, ROPE_DIM // 2), generator=generator) * (2 * torch.pi)
    table[:, : ROPE_DIM // 2, 0] = torch.cos(angles)
    table[:, : ROPE_DIM // 2, 1] = torch.sin(angles)
    return table.to(device)


def _reference(
    q: torch.Tensor, cos_sin: torch.Tensor, positions: torch.Tensor, num_heads: int
) -> torch.Tensor:
    """RMS-norm over the whole 512-wide head, rotate the tail, scale for FP8."""
    num_tokens = q.shape[0]
    x = q.view(num_tokens, num_heads, HEAD_DIM).float()
    inv_rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    normed = x * inv_rms

    nope = normed[..., :NOPE_DIM] * QUANT_SCALE

    # GPT-J interleave: pair j is elements (2j, 2j+1) of the rope tail.
    rope = normed[..., NOPE_DIM:]
    even, odd = rope[..., 0::2], rope[..., 1::2]
    coef = cos_sin[positions][:, : ROPE_DIM // 2, :]  # [tokens, 32, 2]
    cos = coef[..., 0].unsqueeze(1)  # broadcast over heads
    sin = coef[..., 1].unsqueeze(1)
    rotated = torch.empty_like(rope)
    rotated[..., 0::2] = cos * even - sin * odd
    rotated[..., 1::2] = cos * odd + sin * even

    return torch.cat([nope, rotated * QUANT_SCALE], dim=-1)


def _run_op(q, num_heads, cos_sin, cache_seq_lens, seq_len, cu_q_seqlens):
    num_tokens = q.shape[0]
    quant_q = torch.zeros(
        (num_tokens, num_heads * HEAD_DIM), dtype=torch.float8_e4m3fn, device=q.device
    )
    # Sentinel: the fold must leave q_pe untouched, because the rope tail is
    # supposed to land in quant_q instead.
    q_pe = torch.full((num_tokens, num_heads * ROPE_DIM), 7.0, dtype=q.dtype, device=q.device)
    quant_scale = torch.tensor([QUANT_SCALE], dtype=torch.float32, device=q.device)

    torch.ops.trtllm.deepseek_v4_q_norm_fused_fp8(
        q,
        quant_q,
        q_pe,
        num_heads,
        HEAD_DIM,
        NOPE_DIM,
        EPS,
        quant_scale,
        cos_sin,
        cache_seq_lens,
        seq_len,
        cu_q_seqlens,
    )
    return quant_q, q_pe


def _assert_matches(quant_q, q_pe, reference, num_heads):
    """Compare in FP8, not in float.

    A relative tolerance on the dequantized values has to be at least one e4m3
    step (~13%) to absorb rounding, and that is wide enough to swallow real bugs
    -- normalizing over 448 dims instead of 512 is only a 6.9% shift. So quantize
    the reference the same way and require the codes to agree. The kernel folds
    inv_rms and the quant scale into a single multiply where the reference uses
    two, so a few values sit on the other side of a rounding boundary; those get
    a small budget, capped at one FP8 step each.
    """
    num_tokens = quant_q.shape[0]
    got = quant_q.view(num_tokens, num_heads, HEAD_DIM).float()
    expected = reference.to(torch.float8_e4m3fn).float()

    differing = got != expected
    frac = differing.float().mean().item()
    assert frac < 0.01, f"{frac:.4%} of FP8 codes differ from the reference"

    if differing.any():
        scale = torch.maximum(got.abs(), expected.abs()).clamp_min(1e-6)
        worst = ((got - expected).abs() / scale)[differing].max().item()
        assert worst < 0.13, f"a differing code is off by more than one FP8 step ({worst:.3f})"

    assert torch.all(q_pe == 7.0), (
        "q_pe was written; the rope tail must go to quant_q on the fused path"
    )


@skip_pre_blackwell
@pytest.mark.parametrize(
    "num_heads,seq_len",
    [(4, 2), (6, 3)],
    ids=["heads4_seqlen2_pow2", "heads6_seqlen3_divide"],
)
def test_fused_rope_generation_positions(num_heads: int, seq_len: int) -> None:
    """Uniform query length: position = cache_len[batch] - seq_len + local_token.

    The parameters flip both power-of-two shortcuts the kernel takes (`row /
    num_heads` and `token / seq_len` become shift/mask only when the host says
    the divisor is a power of two), so the ids name which arithmetic runs.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    num_seqs = 3
    num_tokens = num_seqs * seq_len

    q = torch.randn((num_tokens, num_heads * HEAD_DIM), dtype=torch.bfloat16, device=device)
    cos_sin = _make_cos_sin(device)
    cache_seq_lens = torch.tensor([16, 40, 71], dtype=torch.int32, device=device)

    quant_q, q_pe = _run_op(q, num_heads, cos_sin, cache_seq_lens, seq_len, None)

    token = torch.arange(num_tokens, device=device)
    positions = cache_seq_lens[token // seq_len] - seq_len + (token % seq_len)
    reference = _reference(q, cos_sin, positions.long(), num_heads)
    _assert_matches(quant_q, q_pe, reference, num_heads)


@skip_pre_blackwell
@pytest.mark.parametrize(
    "num_heads,cached_offset",
    [(4, 0), (6, 5)],
    ids=["heads4_fresh_prefill", "heads6_chunked_prefill"],
)
def test_fused_rope_context_positions(num_heads: int, cached_offset: int) -> None:
    """Ragged: position = local_token + (cache_len[seq] - current_seq_len).

    `cached_offset > 0` is the chunked-prefill / block-reuse case, where part of
    the sequence is already in the KV cache and this chunk's first token is not
    at position 0. Every other test in the suite pins it to zero by disabling
    block reuse, so the second parameter is the only thing that walks that term.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    seq_lens = [5, 3, 7]
    num_tokens = sum(seq_lens)

    cu_q_seqlens = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    cache_seq_lens = torch.tensor(
        [s + cached_offset for s in seq_lens], dtype=torch.int32, device=device
    )
    q = torch.randn((num_tokens, num_heads * HEAD_DIM), dtype=torch.bfloat16, device=device)
    cos_sin = _make_cos_sin(device)

    quant_q, q_pe = _run_op(q, num_heads, cos_sin, cache_seq_lens, 0, cu_q_seqlens)

    positions = torch.cat(
        [torch.arange(length, device=device) + cached_offset for length in seq_lens]
    )
    reference = _reference(q, cos_sin, positions.long(), num_heads)
    _assert_matches(quant_q, q_pe, reference, num_heads)
