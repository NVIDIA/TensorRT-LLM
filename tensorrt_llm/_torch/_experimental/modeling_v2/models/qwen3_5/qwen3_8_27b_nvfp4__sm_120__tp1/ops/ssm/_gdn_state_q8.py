# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The quantized Gated-DeltaNet recurrent-state pool: its layout, the Triton row
quantizer / dequantizer the decode kernels inline, and the gather / scatter that
bridge the pool and the chunked prefill kernel's fp32 state. (Names keep the ``q8``
prefix of the original 8-bit format; the codes are now 16-bit.)

Pool layout per slot: int8-typed bytes ``[HV, 2 * V + Q8_SCALE_ROWS, K]`` (K innermost).
For each value head, byte rows ``0..2V-1`` hold the state as ``V`` rows of ``K`` int16
codes (value row ``h[v, :]`` = byte rows ``2v, 2v + 1``), and the ``Q8_SCALE_ROWS * K``
bytes after them hold the ``V`` fp32 row scales (``V * 4 == Q8_SCALE_ROWS * K`` at
``K == V == 128``). ``h[v, k] ~= code[v, k] * scale[v]`` with
``scale = amax(|h[v, :]|) / 32767`` and ``code = round_half_even(h * (32767 / amax))`` in
``[-32767, 32767]``. A zero row stores ``scale = 0`` and code 0, so zeroed bytes (a reset
slot) dequantize to an exact zero state. The pool stays int8-typed so the cache manager
and the ``mamba_ssm_cache_dtype: int8`` route are unchanged; only its row count
(``config_utils.ssm_state_rows``) reflects the 16-bit codes.

Why 16-bit codes: the state is requantized after every decode token, so each row's
rounding error is re-injected thousands of times over a long generation, and in the
long-memory heads of the first GDN layers it accumulates. e4m3 rows (3 mantissa bits)
break long reasoning past ~4K tokens (GPQA Diamond 60.7 vs 87.9 for an fp32 state);
int8 rows (step amax / 127) hold to ~32K tokens but still lose the longest traces (GPQA
84.85; on the 32 questions with the longest reasoning 34.6% vs 50.8%); fp16 / bf16 states
fail the same check. int16 rows (per-step relative error ~1.5e-5, vs 3.9e-3 for int8 and
3.6e-4 for fp16) match an fp32 state: GPQA 88.10, hardest 32 at 50.4%.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

#: Extra K-wide rows per value head holding the V fp32 row scales.
Q8_SCALE_ROWS = 4
#: The int16 magnitude a row's absolute maximum is scaled to.
Q8_QMAX = 32767.0


def q8_state_rows(v: int, k: int) -> int:
    """Rows of K bytes per value head: 2V rows of int16 codes + the fp32 row scales."""
    assert (v * 4) % k == 0, f"V={v} fp32 scales must fill whole K={k} rows"
    return 2 * v + (v * 4) // k


@triton.jit
def q8_load_rows(p_head, o_v, o_k, mask, K: tl.constexpr, V: tl.constexpr):
    """Dequantize rows ``o_v`` of one head's state to fp32 [len(o_v), K].

    ``p_head`` points at the head's ``[2V + scale rows, K]`` int8 block."""
    p_q = p_head.to(tl.pointer_type(tl.int16))
    q = tl.load(p_q + o_v[:, None] * K + o_k[None, :], mask=mask, other=0)
    p_s = (p_head + 2 * V * K).to(tl.pointer_type(tl.float32))
    s = tl.load(p_s + o_v, mask=mask, other=0.0)
    return q.to(tl.float32) * s[:, None]


@triton.jit
def q8_store_rows(p_head, o_v, o_k, h, mask, K: tl.constexpr, V: tl.constexpr):
    """Quantize fp32 rows ``h`` [len(o_v), K] (one scale per row) and store the
    int16 rows and their fp32 scales into one head's block."""
    amax = tl.max(tl.abs(h), axis=1)
    qmax = 32767.0
    nz = amax > 0.0
    inv = tl.where(nz, qmax / tl.where(nz, amax, 1.0), 0.0)
    s = amax / qmax
    y = tl.minimum(tl.maximum(libdevice.rint(h * inv[:, None]), -qmax), qmax)
    p_q = p_head.to(tl.pointer_type(tl.int16))
    tl.store(p_q + o_v[:, None] * K + o_k[None, :], y.to(tl.int16), mask=mask)
    p_s = (p_head + 2 * V * K).to(tl.pointer_type(tl.float32))
    tl.store(p_s + o_v, s, mask=mask)


@triton.jit
def _q8_gather_dequant_kernel(
    pool,
    out,
    indices,
    s_pool_n,
    s_out_n,
    HV: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
):
    i_n = tl.program_id(0)
    i_h = tl.program_id(1)
    i_vb = tl.program_id(2)
    ROWS: tl.constexpr = 2 * V + (V * 4) // K
    slot = tl.load(indices + i_n).to(tl.int64)
    valid = slot >= 0
    o_v = i_vb * BV + tl.arange(0, BV)
    o_k = tl.arange(0, K)
    p_head = pool + slot * s_pool_n + i_h * ROWS * K
    x = q8_load_rows(p_head, o_v, o_k, valid, K, V)
    p_out = out + i_n.to(tl.int64) * s_out_n + i_h * V * K + o_v[:, None] * K + o_k[None, :]
    tl.store(p_out, x)


@triton.jit
def _q8_quant_scatter_kernel(
    src,
    pool,
    indices,
    s_src_n,
    s_pool_n,
    HV: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
):
    i_n = tl.program_id(0)
    i_h = tl.program_id(1)
    i_vb = tl.program_id(2)
    ROWS: tl.constexpr = 2 * V + (V * 4) // K
    slot = tl.load(indices + i_n).to(tl.int64)
    valid = slot >= 0
    o_v = i_vb * BV + tl.arange(0, BV)
    o_k = tl.arange(0, K)
    x = tl.load(src + i_n.to(tl.int64) * s_src_n + i_h * V * K + o_v[:, None] * K + o_k[None, :])
    p_head = pool + slot * s_pool_n + i_h * ROWS * K
    q8_store_rows(p_head, o_v, o_k, x, valid, K, V)


#: Value rows per program of the gather / scatter (4 planes of a 128-row head).
_IO_BV = 32


def q8_gather_dequant(pool: torch.Tensor, indices: torch.Tensor, v: int) -> torch.Tensor:
    """fp32 ``[len(indices), HV, V, K]`` = dequantized ``pool[indices]`` (a negative
    index yields a zero state)."""
    hv, k = pool.shape[1], pool.shape[3]
    n = indices.shape[0]
    out = torch.empty(n, hv, v, k, dtype=torch.float32, device=pool.device)
    if n == 0:
        return out
    _q8_gather_dequant_kernel[(n, hv, triton.cdiv(v, _IO_BV))](
        pool, out, indices, pool.stride(0), out.stride(0),
        HV=hv, V=v, K=k, BV=_IO_BV, num_warps=4,
    )  # fmt: skip
    return out


def q8_quant_scatter(src: torch.Tensor, pool: torch.Tensor, indices: torch.Tensor) -> None:
    """``pool[indices] = quantize(src)`` for fp32 ``src`` ``[n, HV, V, K]`` (negative
    indices are skipped)."""
    n, hv, v, k = src.shape
    if n == 0:
        return
    _q8_quant_scatter_kernel[(n, hv, triton.cdiv(v, _IO_BV))](
        src, pool, indices, src.stride(0), pool.stride(0),
        HV=hv, V=v, K=k, BV=_IO_BV, num_warps=4,
    )  # fmt: skip


# ---------------------------------------------------------------------------------------
# Plain-torch twins, for the entries' `reference`s and their tests.
# ---------------------------------------------------------------------------------------


def dequantize_slots(pool: torch.Tensor, slots: list[int]) -> torch.Tensor:
    """fp32 ``[len(slots), HV, V, K]``: ``code * scale`` per row, zero for a negative
    slot. Exact: the product of an int16 value and an fp32 scale rounds once, as the
    kernels' does."""
    hv, rows, k = pool.shape[1:]
    v = rows * k // (2 * k + 4)
    assert q8_state_rows(v, k) == rows, tuple(pool.shape)
    out = torch.zeros(len(slots), hv, v, k, dtype=torch.float32, device=pool.device)
    for i, slot in enumerate(slots):
        if slot < 0:
            continue
        block = pool[slot]
        codes = block[:, : 2 * v].contiguous().reshape(hv, v, 2 * k).view(torch.int16).float()
        scales = block[:, 2 * v :].contiguous().view(torch.float32).reshape(hv, v)
        out[i] = codes * scales[:, :, None]
    return out


def quantize_rows(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(codes int16 [..., V, K], scales fp32 [..., V])`` of fp32 rows ``h``: the
    scale ``amax / 32767``, the codes ``round_half_even(h * (32767 / amax))`` clamped to
    [-32767, 32767], a zero row zero."""
    h = h.float()
    amax = h.abs().amax(-1)
    nz = amax > 0
    inv = torch.where(nz, Q8_QMAX / torch.where(nz, amax, torch.ones_like(amax)), 0.0)
    codes = torch.round(h * inv[..., None]).clamp(-Q8_QMAX, Q8_QMAX).to(torch.int16)
    return codes, amax / Q8_QMAX


def store_slots(pool: torch.Tensor, slots: list[int], h: torch.Tensor) -> None:
    """``pool[slot] = quantize(h[i])`` for every non-negative slot, in place."""
    v = h.shape[2]
    hv, k = pool.shape[1], pool.shape[3]
    for i, slot in enumerate(slots):
        if slot < 0:
            continue
        codes, scales = quantize_rows(h[i])
        pool[slot, :, : 2 * v] = codes.contiguous().view(torch.int8).reshape(hv, 2 * v, k)
        pool[slot, :, 2 * v :] = scales.contiguous().view(torch.int8).reshape(hv, -1, k)
