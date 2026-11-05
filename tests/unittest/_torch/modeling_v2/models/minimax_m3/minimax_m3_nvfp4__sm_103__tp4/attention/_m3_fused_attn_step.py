# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A short-sequence decode step of one MiniMax-M3 TP4 sparse layer, for the fused attention entries' tests
(``m3_fused_decode_attn``, ``m3_attn_mega``, ``m3_attn_c1``).

The history is written into the paged NVFP4 pool by the in-tree scatter (``fused_write_layer_caches_nvfp4``, which
quantizes with ``torch.ops.trtllm.fp4_quantize``) at the same ``kv_quant_scale`` the step inserts with. The new
tokens' slots and every row past a request's length hold random bytes, so a kernel that skipped the insert or
read past the length is caught. Pages are shuffled so a kernel that ignored the block table reads the wrong ones.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.attention.m3_fused_decode_attn import (  # noqa: E501
    attend_cached,
    nvfp4_dequantize,
    query_rows,
    v_scale_rows,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_scatter import (
    fused_write_layer_caches_nvfp4,
)

NUM_HEADS = 16
HEAD_DIM = 128
PAGE = 128
PACKED_WIDTH = (NUM_HEADS + 4) * HEAD_DIM
HIDDEN = 6144
#: The MXFP8 hand-off buffers' rows (the target's MERGE_MXFP8 buffer).
MXFP8_ROWS = 128
#: NVFP4 second-level scales: block scales stay inside E4M3 for |x| up to 16 (K) and 8 (V, N(0, 1) rows).
K_QUANT = 448.0 * 6.0 / 16.0
V_QUANT = 448.0 * 6.0 / 8.0
ROPE_THETA = 5.0e6
MAX_POSITION = 2304


def rotary_cos_sin(device) -> torch.Tensor:
    """fp32 [max_pos, 2, 32]: NeoX cos / sin of the 64 rotary dims at every position."""
    inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float64, device=device) / 64)
    )
    ang = (
        torch.arange(MAX_POSITION, dtype=torch.float64, device=device)[:, None] * inv_freq[None, :]
    )
    return torch.stack([ang.cos(), ang.sin()], dim=1).float().contiguous()


def build_step(
    seq_lens: Sequence[int], qlen: int, seed: int, packed: torch.Tensor = None
) -> SimpleNamespace:
    """Caches, metadata and weights of one step; ``packed`` (bf16 [T, 2560]) defaults to N(0, 1) rows."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    batch = len(seq_lens)
    t_total = batch * qlen
    max_blocks = max(math.ceil(s / PAGE) for s in seq_lens)
    num_pages = batch * (max_blocks + 1) + 3
    # One spare column: the kernel must read the columns the length needs, not the table's width.
    block_table = torch.randperm(num_pages, device=dev, generator=g)[: batch * (max_blocks + 1)]
    block_table = block_table.to(torch.int32).reshape(batch, max_blocks + 1).contiguous()
    kv_data = torch.randint(
        0, 256, (num_pages, 2, 1, PAGE, HEAD_DIM // 2), dtype=torch.uint8, device=dev, generator=g
    )
    kv_scale = torch.randint(
        0, 256, (num_pages, 2, 1, PAGE, HEAD_DIM // 16), dtype=torch.uint8, device=dev, generator=g
    )
    idx_k = torch.randint(
        0, 256, (num_pages, 1, PAGE, HEAD_DIM), dtype=torch.uint8, device=dev, generator=g
    ).view(torch.float8_e4m3fn)
    kv_quant_scale = torch.tensor([1.0, K_QUANT, V_QUANT], dtype=torch.float32, device=dev)

    hist_slots = []
    for r, n in enumerate(seq_lens):
        for pos in range(n - qlen):
            hist_slots.append(int(block_table[r, pos // PAGE]) * PAGE + pos % PAGE)
    if hist_slots:
        h = len(hist_slots)
        k_hist = (torch.randn(h, HEAD_DIM, device=dev, generator=g) * 1.5).to(torch.bfloat16)
        v_hist = torch.randn(h, HEAD_DIM, device=dev, generator=g).to(torch.bfloat16)
        i_hist = torch.randn(h, HEAD_DIM, device=dev, generator=g).to(torch.bfloat16)
        assert fused_write_layer_caches_nvfp4(
            kv_data[:, 0],
            kv_data[:, 1],
            kv_scale[:, 0],
            kv_scale[:, 1],
            idx_k,
            torch.tensor(hist_slots, dtype=torch.int32, device=dev),
            k_hist,
            v_hist,
            i_hist,
            kv_quant_scale,
        )

    positions, slots = [], []
    for r, n in enumerate(seq_lens):
        for t in range(qlen):
            pos = n - qlen + t
            positions.append(pos)
            slots.append(int(block_table[r, pos // PAGE]) * PAGE + pos % PAGE)
    if packed is None:
        packed = torch.randn(t_total, PACKED_WIDTH, device=dev, generator=g).to(torch.bfloat16)

    def norm_weight():
        return (torch.randn(HEAD_DIM, device=dev, generator=g) * 0.1).to(torch.bfloat16)

    return SimpleNamespace(
        packed=packed,
        kv_data=kv_data,
        kv_scale=kv_scale,
        idx_k=idx_k,
        out_cache_loc=torch.tensor(slots, dtype=torch.int32, device=dev),
        position_ids=torch.tensor(positions, dtype=torch.int32, device=dev),
        block_table=block_table,
        seq_lens=torch.tensor(list(seq_lens), dtype=torch.int32, device=dev),
        kv_quant_scale=kv_quant_scale,
        k_global_scale=torch.tensor([1.0 / K_QUANT], dtype=torch.float32, device=dev),
        v_global_scale=torch.tensor([1.0 / V_QUANT], dtype=torch.float32, device=dev),
        q_norm_weight=norm_weight(),
        k_norm_weight=norm_weight(),
        index_k_norm_weight=norm_weight(),
        rotary_cos_sin=rotary_cos_sin(dev),
        eps=1e-6,
        sm_scale=HEAD_DIM**-0.5,
        decode_query_len=qlen,
        num_tokens=t_total,
    )


def attention_args(s: SimpleNamespace) -> tuple:
    """The attention arguments every entry takes, in schema order (``kv_data`` .. ``rotary_cos_sin``, the two
    scalars and ``decode_query_len``)."""
    return (
        s.kv_data,
        s.kv_scale,
        s.idx_k,
        s.out_cache_loc,
        s.position_ids,
        s.block_table,
        s.seq_lens,
        s.kv_quant_scale,
        s.k_global_scale,
        s.v_global_scale,
        s.q_norm_weight,
        s.k_norm_weight,
        s.index_k_norm_weight,
        s.rotary_cos_sin,
        s.eps,
        s.sm_scale,
        s.decode_query_len,
    )


def swizzle(rows_exp: torch.Tensor, padded_rows: int) -> torch.Tensor:
    """[rows, k_blocks] UE8M0 bytes -> the 128x4-swizzled buffer of ``padded_rows`` rows (filler 127)."""
    rows, kb = rows_exp.shape
    k_tiles = (kb + 3) // 4
    out = torch.full((padded_rows * k_tiles * 4,), 127, dtype=torch.uint8, device=rows_exp.device)
    r = torch.arange(rows, device=rows_exp.device)[:, None]
    k = torch.arange(kb, device=rows_exp.device)[None, :]
    off = (
        (r // 128) * k_tiles * 512 + (k // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + k % 4
    )
    out[off.reshape(-1)] = rows_exp.reshape(-1)
    return out


def mxfp8_operand(
    rows: int, cols: int, exp_lo: int, exp_hi: int, g: torch.Generator, padded_rows: int
):
    """A random MXFP8 operand: E4M3 values from N(0, 1) and UE8M0 exponents in [exp_lo, exp_hi], swizzled."""
    dev = torch.device("cuda")
    q = torch.randn(rows, cols, device=dev, generator=g).to(torch.float8_e4m3fn)
    ex = torch.randint(
        exp_lo, exp_hi + 1, (rows, cols // 32), dtype=torch.uint8, device=dev, generator=g
    )
    return q, swizzle(ex, padded_rows)


def new_slot_mask(s: SimpleNamespace, num_pages: int) -> torch.Tensor:
    """bool [pages, 128]: the rows the step inserts."""
    mask = torch.zeros(num_pages, PAGE, dtype=torch.bool, device=s.kv_data.device)
    for slot in s.out_cache_loc.tolist():
        mask[slot // PAGE, slot % PAGE] = True
    return mask


# Band of the attention output, in bf16 ulps (2^-8) of each [128] head row's largest magnitude, and of the relative
# RMS, against the float64 decode over the caches the op wrote. Both sides read the same operands (E4M3 q, E2M1 x
# E4M3 K/V, all exact in fp16), so the gap is the bf16 rounding on both sides (half an ulp each), the fp16
# probabilities before PV (2^-11 relative), fp32 accumulation and approximate exp2/log2 in the split-K partials and
# their LSE merge. Measured over 46 steps: at most 2.0 ulps per element, 0.2 ulp RMS.
ATTN_ELEMENT_ULP = 3.0
ATTN_RMS_ULP = 0.5
# Inserted rows against the reference's NVFP4 / E4M3 recipes: the kernel's RMSNorm sums the squares in another order
# and uses approximate rsqrt / reciprocals, so an element on a rounding tie can land one code apart. Measured over
# 46 steps: no differing scale byte, at most 3 differing value codes in a step (1 in 300 of the inserted elements).
# Such a code moves one key's (or value's) element by an E2M1 step, up to 44 ulps of an output row, which is why
# the output is gated against the decode over the op's own caches rather than the reference's.
CODE_MISMATCH_MIN = 4
CODE_MISMATCH_FRACTION = 0.005


def check_insert(s: SimpleNamespace, before: tuple, ref) -> None:
    """Every row the step does not insert is untouched; inserted rows match the reference's recipe."""
    kv_data0, kv_scale0, idx0 = before
    pages = s.kv_data.shape[0]
    mask = new_slot_mask(s, pages)
    keep = ~mask
    for plane in (0, 1):
        assert torch.equal(s.kv_data[:, plane, 0][keep], kv_data0[:, plane, 0][keep]), (
            "a row past the insert moved"
        )
    assert torch.equal(s.kv_scale[:, 0, 0][keep], kv_scale0[:, 0, 0][keep])
    assert torch.equal(s.idx_k.view(torch.uint8)[:, 0][keep], idx0.view(torch.uint8)[:, 0][keep])
    v_now = torch.stack([v_scale_rows(s.kv_scale[p, 1, 0]) for p in range(pages)])
    v_was = torch.stack([v_scale_rows(kv_scale0[p, 1, 0]) for p in range(pages)])
    v_ref = torch.stack([v_scale_rows(ref.kv_scale[p, 1, 0]) for p in range(pages)])
    assert torch.equal(v_now[keep], v_was[keep]), "a V scale outside the inserted rows moved"
    assert torch.equal(s.kv_scale[:, 0, 0][mask], ref.kv_scale[:, 0, 0][mask]), "K block scales"
    assert torch.equal(v_now[mask], v_ref[mask]), "V block scales (4x4 token-quad order)"
    n = int(mask.sum()) * HEAD_DIM
    for plane, scales in ((0, s.kv_scale[:, 0, 0][mask]), (1, v_now[mask])):
        got = nvfp4_dequantize(s.kv_data[:, plane, 0][mask], scales)
        want = nvfp4_dequantize(ref.kv_data[:, plane, 0][mask], scales)
        assert int((got != want).sum()) <= max(CODE_MISMATCH_MIN, CODE_MISMATCH_FRACTION * n), (
            f"plane {plane} E2M1 codes"
        )
    got = s.idx_k.view(torch.uint8)[:, 0][mask]
    want = ref.idx_k.view(torch.uint8)[:, 0][mask]
    assert int((got != want).sum()) <= max(CODE_MISMATCH_MIN, CODE_MISMATCH_FRACTION * n), (
        "index-K E4M3 codes"
    )


def decode_over_written_caches(s: SimpleNamespace, packed: torch.Tensor) -> torch.Tensor:
    """The reference's decode (bf16 [T, 2048]) of ``packed``'s queries over the caches as the op left them."""
    q = query_rows(packed, s.position_ids, s.q_norm_weight, s.eps, s.rotary_cos_sin)
    return attend_cached(
        q,
        s.kv_data,
        s.kv_scale,
        s.block_table,
        s.seq_lens,
        s.k_global_scale,
        s.v_global_scale,
        s.sm_scale,
        s.decode_query_len,
    )
