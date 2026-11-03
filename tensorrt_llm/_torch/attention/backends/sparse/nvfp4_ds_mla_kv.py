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
"""``nvfp4_ds_mla``: inline NVFP4 latent KV records for DSA sparse MLA (Kimi K3.1).

Each token of an MLA layer occupies one 352-byte record, the ``nvfp4.fp8rope`` cache format of the PrimTS sparse MLA
kernels and vLLM's ``nvfp4_ds_mla`` (same bytes):

- ``[0, 256)``: the 512 NoPE latent values as E2M1, two per byte (even element in the low nibble);
- ``[256, 320)``: the 64 RoPE values as unscaled E4M3;
- ``[320, 352)``: one E4M3 scale per 16 NoPE values; the scale of block ``s`` is byte ``320 + 8 * (s % 4) + s // 4``.

Quantization of the values ``v = x / g`` held by the records: ``sf = e4m3(max(amax_16(v) / 6, 2^-9))``,
``q = e2m1(v * (1 / float(sf)))`` with round to nearest even and saturation, ``rope = e4m3(x / g_rope)`` (round to
nearest even, saturating). Dequantization: ``x = float(q) * float(sf) * g``, ``x_rope = float(rope) * g_rope``.

The tensor scales ``g`` (NoPE) and ``g_rope`` (:class:`RecordScales`) move the values into the normal range of the
E4M3 block scales and of the E4M3 RoPE bytes; the attention folds them into its softmax and output scales and into
the scale of the query's RoPE part, so the byte layout is unchanged (``g = g_rope = 1`` stores vLLM's values).
"""

import math
import os
from dataclasses import dataclass
from functools import lru_cache

import torch
import triton
import triton.language as tl

NOPE_DIM = 512
ROPE_DIM = 64
BLOCK = 16
NUM_BLOCKS = NOPE_DIM // BLOCK
ROPE_OFFSET = NOPE_DIM // 2
SCALE_OFFSET = ROPE_OFFSET + ROPE_DIM
TOKEN_BYTES = SCALE_OFFSET + NUM_BLOCKS
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MAX = 6.0
_E4M3_MAX = 448.0
NOPE_AMAX_ENV = "TRTLLM_NVFP4_DS_MLA_NOPE_AMAX"
ROPE_AMAX_ENV = "TRTLLM_NVFP4_DS_MLA_ROPE_AMAX"
# Largest NoPE / RoPE magnitude the records represent (saturation bound): g = amax / (6 * 448), g_rope = amax / 448.
# Kimi K3.1 latents: NoPE 16-value block maxima reach ~115 on outlier tokens while whole layers sit at 0.016-0.1, below
# the 0.09375 (6 * 2^-6) where unscaled E4M3 block scales turn subnormal; g = 1/8 keeps scales normal down to 0.0117 and
# saturates at 336. RoPE magnitudes reach ~70 and the query's RoPE part ~17: g_rope = 1/2 stores up to 224 and scales
# the query's RoPE part by g_rope / g = 4.
DEFAULT_NOPE_AMAX = 6.0 * 448.0 / 8.0
DEFAULT_ROPE_AMAX = 448.0 / 2.0


@dataclass(frozen=True)
class RecordScales:
    """Tensor scales of the values a record holds: NoPE ``x / nope``, RoPE ``x / rope``."""

    nope: float = 1.0
    rope: float = 1.0


def _amax_from_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        amax = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive finite float, got {value!r}") from error
    if not math.isfinite(amax) or amax <= 0:
        raise ValueError(f"{name} must be a positive finite float, got {value!r}")
    return amax


@lru_cache(maxsize=1)
def record_scales() -> RecordScales:
    """The process's record scales (``TRTLLM_NVFP4_DS_MLA_NOPE_AMAX`` / ``_ROPE_AMAX``). Writers and readers of one
    cache, including both sides of a disaggregated KV transfer, must use the same values."""
    nope = _amax_from_env(NOPE_AMAX_ENV, DEFAULT_NOPE_AMAX) / (_E2M1_MAX * _E4M3_MAX)
    rope = _amax_from_env(ROPE_AMAX_ENV, DEFAULT_ROPE_AMAX) / _E4M3_MAX
    return RecordScales(nope=nope, rope=rope)


def scale_byte(block: int) -> int:
    """Record byte of the E4M3 scale of NoPE block ``block``."""
    return SCALE_OFFSET + 8 * (block % 4) + block // 4


@triton.jit
def _e2m1_code(v):
    # Round-to-nearest-even onto {0, 0.5, 1, 1.5, 2, 3, 4, 6} with saturation; ties go to the even codes 0, 2, 4, 6.
    m = tl.abs(v)
    code = (
        (m > 0.25).to(tl.int32)
        + (m >= 0.75).to(tl.int32)
        + (m > 1.25).to(tl.int32)
        + (m >= 1.75).to(tl.int32)
        + (m > 2.5).to(tl.int32)
        + (m >= 3.5).to(tl.int32)
        + (m > 5.0).to(tl.int32)
    )
    return code + (v < 0).to(tl.int32) * 8


@triton.jit
def _quant_scatter_kernel(
    rows_ptr,  # bf16 [T, 576]
    loc_ptr,  # int32 / int64 [T]: record row of each token, < 0 skips
    pool_ptr,  # uint8 [N, 352]
    rows_stride,
    inv_nope,  # 1 / g
    inv_rope,  # 1 / g_rope
    BLOCK_C: tl.constexpr,
):
    t = tl.program_id(0)
    loc = tl.load(loc_ptr + t).to(tl.int64)
    if loc >= 0:
        dst = pool_ptr + loc * 352
        src = rows_ptr + t.to(tl.int64) * rows_stride
        # NoPE: 32 blocks x 16 values.
        b = tl.arange(0, 32)
        e = tl.arange(0, BLOCK_C)
        x = tl.load(src + b[:, None] * BLOCK_C + e[None, :]).to(tl.float32) * inv_nope
        amax = tl.max(tl.abs(x), axis=1)
        sf = tl.maximum(tl.math.div_rn(amax, 6.0), 0.001953125).to(tl.float8e4nv)
        sff = sf.to(tl.float32)
        inv = tl.math.div_rn(tl.full(sff.shape, 1.0, tl.float32), sff)
        code = _e2m1_code(x * inv[:, None])
        pairs = tl.reshape(code, (32, BLOCK_C // 2, 2))
        lo, hi = tl.split(pairs)
        packed = (lo | (hi << 4)).to(tl.uint8)
        p = tl.arange(0, BLOCK_C // 2)
        tl.store(dst + b[:, None] * (BLOCK_C // 2) + p[None, :], packed)
        tl.store(dst + 320 + 8 * (b % 4) + b // 4, sf.to(tl.uint8, bitcast=True))
        # RoPE: unscaled E4M3.
        r = tl.arange(0, 64)
        y = tl.load(src + 512 + r).to(tl.float32) * inv_rope
        tl.store(dst + 256 + r, y.to(tl.float8e4nv).to(tl.uint8, bitcast=True))


def quant_scatter(
    pool: torch.Tensor, loc: torch.Tensor, rows: torch.Tensor, scales: RecordScales = RecordScales()
) -> None:
    """Quantize latent rows ``[T, 576]`` (BF16, RoPE applied) into records ``pool[loc[t]]`` of a ``[N, 352]`` pool."""
    if rows.shape[0] == 0:
        return
    if rows.shape[-1] != NOPE_DIM + ROPE_DIM or rows.stride(-1) != 1:
        raise ValueError(f"latent rows must be [T, {NOPE_DIM + ROPE_DIM}] with unit last stride")
    if pool.dtype != torch.uint8 or pool.shape[-1] != TOKEN_BYTES or not pool.is_contiguous():
        raise ValueError(
            f"the nvfp4_ds_mla pool must be a contiguous uint8 [N, {TOKEN_BYTES}] view"
        )
    _quant_scatter_kernel[(rows.shape[0],)](
        rows,
        loc.contiguous(),
        pool,
        rows.stride(0),
        1.0 / scales.nope,
        1.0 / scales.rope,
        BLOCK_C=BLOCK,
        num_warps=2,
    )


# Eight E2M1 values (one code word) times their E4M3 block scale -> eight E4M3 bytes, as the PrimTS nvfp4.fp8rope
# loader's f16x2 path: exact half-precision products rounded once to E4M3 with saturation (the same bytes as its
# packed mul.rn.e4m3x4.e2m1x4.e4m3x4.satfinite path). $3 holds the scale byte twice in its low half.
_E2M1X8_TO_E4M3_ASM = tl.constexpr(
    """
{
    .reg .b16 sf;
    .reg .b32 s2;
    .reg .b8 b0, b1, b2, b3;
    .reg .b32 h0, h1, h2, h3;
    .reg .b16 e0, e1, e2, e3;
    mov.b32 {sf, _}, $3;
    cvt.rn.f16x2.e4m3x2 s2, sf;
    mov.b32 {b0, b1, b2, b3}, $2;
    cvt.rn.f16x2.e2m1x2 h0, b0;
    cvt.rn.f16x2.e2m1x2 h1, b1;
    cvt.rn.f16x2.e2m1x2 h2, b2;
    cvt.rn.f16x2.e2m1x2 h3, b3;
    mul.rn.f16x2 h0, h0, s2;
    mul.rn.f16x2 h1, h1, s2;
    mul.rn.f16x2 h2, h2, s2;
    mul.rn.f16x2 h3, h3, s2;
    cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;
    cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;
    cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;
    cvt.rn.satfinite.e4m3x2.f16x2 e3, h3;
    mov.b32 $0, {e0, e1};
    mov.b32 $1, {e2, e3};
}
"""
)


@triton.jit
def _paged_records_to_e4m3_kernel(
    pool_ptr,  # uint8 [N, 352] records
    pool_words_ptr,  # int32 view of the pool (88 words per record)
    block_table_ptr,  # int32 [R, max_blocks]
    bt_stride0,
    bt_stride1,
    kv_indptr_ptr,  # int64 [R + 1]: request r owns output rows [indptr[r], indptr[r + 1])
    out_words_ptr,  # int32 view of the uint8 [M, 576] output (144 words per row)
    max_blocks,
    tokens_per_block,
    stride_factor,  # pool rows per block-table unit
    layer_id,
    BT: tl.constexpr,
):
    # Grid (position blocks, requests): the position axis is the one that can exceed 65535 programs.
    r = tl.program_id(1)
    start = tl.load(kv_indptr_ptr + r)
    kv_len = tl.load(kv_indptr_ptr + r + 1) - start
    pos = tl.program_id(0) * BT + tl.arange(0, BT)
    in_req = pos < kv_len
    blk = pos // tokens_per_block
    base = tl.load(
        block_table_ptr + r * bt_stride0 + blk * bt_stride1,
        mask=in_req & (blk < max_blocks),
        other=-1,
    )
    valid = in_req & (base >= 0)
    src = tl.where(
        valid,
        base.to(tl.int64) * stride_factor + pos % tokens_per_block + layer_id * tokens_per_block,
        0,
    )
    # NoPE code words 0..63: word w holds values 8w .. 8w + 7 of block w // 2.
    w = tl.arange(0, 64)
    words = tl.load(pool_words_ptr + src[:, None] * 88 + w[None, :], mask=valid[:, None], other=0)
    # Scale bytes: byte i = 8 * (b % 4) + b // 4 holds block b -> [4, 8] transposed is block order.
    i = tl.arange(0, 32)
    sf = tl.load(pool_ptr + src[:, None] * 352 + 320 + i[None, :], mask=valid[:, None], other=0)
    sf = tl.reshape(tl.trans(tl.reshape(sf, (BT, 4, 8)), 0, 2, 1), (BT, 32)).to(tl.int32)
    sf = tl.reshape(tl.broadcast_to(sf[:, :, None], (BT, 32, 2)), (BT, 64)) * 0x101
    lo, hi = tl.inline_asm_elementwise(
        _E2M1X8_TO_E4M3_ASM,
        "=r,=r,r,r",
        [words, sf],
        dtype=(tl.int32, tl.int32),
        is_pure=True,
        pack=1,
    )
    nope = tl.reshape(tl.join(lo, hi), (BT, 128))
    dst = (start + pos).to(tl.int64) * 144
    c = tl.arange(0, 128)
    tl.store(out_words_ptr + dst[:, None] + c[None, :], nope, mask=in_req[:, None])
    # RoPE words 64..79 -> output words 128..143.
    e = tl.arange(0, 16)
    rope = tl.load(
        pool_words_ptr + src[:, None] * 88 + 64 + e[None, :], mask=valid[:, None], other=0
    )
    tl.store(out_words_ptr + dst[:, None] + 128 + e[None, :], rope, mask=in_req[:, None])


def paged_records_to_e4m3(
    pool: torch.Tensor,
    block_table: torch.Tensor,
    kv_indptr: torch.Tensor,
    max_kv_len: int,
    tokens_per_block: int,
    stride_factor: int,
    layer_id: int,
    out: torch.Tensor,
) -> None:
    """E4M3 D576 rows of the records of ``R`` requests: ``out[kv_indptr[r] + p]`` = record of token ``p`` of request
    ``r`` (pool row ``block_table[r, p // tpb] * stride_factor + p % tpb + layer_id * tpb``, the row
    ``convert_req_index_to_global`` returns) converted as the PrimTS ``nvfp4.fp8rope`` loader does: each NoPE value
    E2M1 x its E4M3 block scale rounded once to E4M3 (nearest even, saturating), the RoPE bytes copied. Rows of
    unmapped blocks are zero. ``out`` is uint8 ``[>= kv_indptr[R], 576]``."""
    num_requests = block_table.shape[0]
    if num_requests == 0 or max_kv_len <= 0:
        return
    if pool.dtype != torch.uint8 or pool.shape[-1] != TOKEN_BYTES:
        raise ValueError(f"the nvfp4_ds_mla pool must be a uint8 [N, {TOKEN_BYTES}] view")
    if out.dtype != torch.uint8 or out.shape[-1] != NOPE_DIM + ROPE_DIM or not out.is_contiguous():
        raise ValueError(f"out must be a contiguous uint8 [M, {NOPE_DIM + ROPE_DIM}] tensor")
    if not pool.is_contiguous():
        raise ValueError("the nvfp4_ds_mla pool view must be contiguous")
    bt = 16
    _paged_records_to_e4m3_kernel[(triton.cdiv(max_kv_len, bt), num_requests)](
        pool,
        pool.view(torch.int32),
        block_table,
        block_table.stride(0),
        block_table.stride(1),
        kv_indptr,
        out.view(torch.int32),
        block_table.shape[1],
        tokens_per_block,
        stride_factor,
        layer_id,
        BT=bt,
        num_warps=4,
    )


def dequant_rows(records: torch.Tensor, scales: RecordScales = RecordScales()) -> torch.Tensor:
    """FP32 ``[M, 576]`` values of ``[M, 352]`` records (reference / tests)."""
    m = records.shape[0]
    table = torch.tensor(_E2M1 + tuple(-v for v in _E2M1), device=records.device)
    packed = records[:, :ROPE_OFFSET].long()
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(m, NOPE_DIM)
    pos = torch.tensor([scale_byte(s) for s in range(NUM_BLOCKS)], device=records.device)
    sf = records[:, pos].contiguous().view(torch.float8_e4m3fn).float()
    nope = (table[codes].reshape(m, NUM_BLOCKS, BLOCK) * sf[..., None]).reshape(m, NOPE_DIM)
    rope = records[:, ROPE_OFFSET:SCALE_OFFSET].contiguous().view(torch.float8_e4m3fn).float()
    return torch.cat([nope * scales.nope, rope * scales.rope], dim=-1)


def quant_rows_reference(rows: torch.Tensor, scales: RecordScales = RecordScales()) -> torch.Tensor:
    """Torch reference of :func:`quant_scatter` for ``[T, 576]`` rows: ``[T, 352]`` records."""
    x = rows.float()
    t = x.shape[0]
    nope = (x[:, :NOPE_DIM] * (1.0 / scales.nope)).reshape(t, NUM_BLOCKS, BLOCK)
    amax = nope.abs().amax(-1)
    sf = torch.clamp(amax / 6.0, min=2.0**-9, max=448.0).to(torch.float8_e4m3fn)
    v = nope * (1.0 / sf.float())[..., None]
    m = v.abs()
    code = (
        (m > 0.25).int()
        + (m >= 0.75).int()
        + (m > 1.25).int()
        + (m >= 1.75).int()
        + (m > 2.5).int()
        + (m >= 3.5).int()
        + (m > 5.0).int()
    ) + (v < 0).int() * 8
    code = code.reshape(t, NOPE_DIM).to(torch.uint8)
    out = torch.empty(t, TOKEN_BYTES, dtype=torch.uint8, device=rows.device)
    out[:, :ROPE_OFFSET] = code[:, 0::2] | (code[:, 1::2] << 4)
    out[:, ROPE_OFFSET:SCALE_OFFSET] = (
        (x[:, NOPE_DIM:] * (1.0 / scales.rope))
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    pos = torch.tensor([scale_byte(s) for s in range(NUM_BLOCKS)], device=rows.device)
    out[:, pos] = sf.view(torch.uint8)
    return out


__all__ = [
    "NOPE_DIM",
    "ROPE_DIM",
    "TOKEN_BYTES",
    "RecordScales",
    "dequant_rows",
    "paged_records_to_e4m3",
    "quant_rows_reference",
    "quant_scatter",
    "record_scales",
    "scale_byte",
]
