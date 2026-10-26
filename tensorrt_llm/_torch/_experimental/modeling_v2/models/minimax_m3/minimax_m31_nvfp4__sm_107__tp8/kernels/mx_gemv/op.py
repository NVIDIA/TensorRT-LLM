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
"""MXFP8 x MXFP8 decode GEMV (tcgen05, block-scaled) at the TP2 per-rank projection shapes (SM107).

Compiled on the first call per (N, K, split, ring, early trigger, PDL, push, MMA width), which must happen outside
CUDA-graph capture, or ahead of any call by ``precompile``; once ``freeze`` closes compiles, a call whose kernel is not
compiled declines instead.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from tensorrt_llm._torch import cute_aot
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from ...m31_switches import m31_disabled

MAX_TOKENS = 48  # capacity: fixed: the decode kernels' row bound (decode steps)
MMA_WIDTHS = (
    8,
    16,
    32,
    48,
)  # token columns of the compiled MMA; a call uses the smallest that holds M
_SMEM_LIMIT = 220 * 1024  # capacity: fixed: the device's shared-memory limit


def _mma_n(m: int) -> int:
    return next(w for w in MMA_WIDTHS if m <= w)


def _smem_bytes(ring: int, tiles: int, split: int, mma_n: int) -> int:
    return (
        ring * (128 * 128 + 512)
        + tiles * (mma_n * 128 + 512)
        + split * (128 // split) * (mma_n + 4) * 4
        + 1024
    )


_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}

# (N, K) -> (split, ring[, contig, nacc, a_tmem, st_async]) of the TP2 per-rank projections. The optional fields default
# to _CONTIG, _NACC, _A_TMEM, _ST_ASYNC. SM107 graph us at M 1 / 4 / 8 over weights rotated beyond L2, against the
# stock MXFP8 quantize + FlashInfer CuTe DSL GEMM: QKV 4.11 / 4.12 / 4.10 vs 8.6-8.8, o_proj 3.90 / 3.96 / 3.70 vs
# 6.9-7.2, shared gate_up 3.11 / 3.09 / 3.10 vs 8.1-8.4, shared down 2.22 vs 4.7-4.9, dense gate_up 8.15 / 8.56 / 8.55
# vs 10.2-10.5, dense down 5.89 / 5.72 / 5.46 vs 8.7-9.0.
_CONFIGS: Dict[Tuple[int, int], tuple] = {
    # The sm_107 tp8 target's per-rank projections (8 ranks over two trays), (split, ring) from an SM107 sweep
    # (locked, MXFP8 in, back-to-back PDL, rows 1 / 4 / 8; the GB300 tp8 table has none of these shapes on SM107).
    # sparse-layer QKV + index projection: 2.37 / 2.43 / 2.42 us
    (1536, 6144): (8, 8),
    # dense-layer QKV: 2.34 / 2.35 / 2.36 us
    (1280, 6144): (8, 8),
    # o_proj: 2.05 / 2.07 / 2.07 us
    (6144, 1024): (2, 12),
    # shared-expert gate_up: 2.16 / 2.17 / 2.17 us
    (768, 6144): (8, 8),
    # shared-expert down: 1.49 / 1.50 / 1.52 us
    (6144, 384): (1, 8),
    # dense-layer gate_up: 3.09 / 3.12 / 3.08 us
    (3072, 6144): (4, 12),
    # dense-layer down: 2.29 / 2.30 / 2.29 us
    (6144, 1536): (2, 8),
}
# Per-token-count configs (N, K, M); none at TP2.
_CONFIGS_BY_M: Dict[Tuple[int, int, int], tuple] = {}
# Above 8 tokens (wider than the TP2 decode step); none at TP2.
_CONFIGS_WIDE: Dict[Tuple[int, int], tuple] = {}
# VRCFG-SWI: the swiglu-prologue calls (bf16 gate | up in) at up to SWIGLU_CFG_MAX_ROWS rows take their own ring: the
# prologue's per-k-tile quantize shifts the best ring depth. Same split and k-tile order as the table, so bit for bit
# its output. SM107 sweep (8 rows, PDL back-to-back, weights rotated past L2): dense down 4.66 vs 5.86 us. Kill
# switch TRTLLM_M31_DISABLE_MXGEMV_VRCFG, read at import.
MXGEMV_VRCFG = not m31_disabled("MXGEMV_VRCFG")
SWIGLU_CFG_MAX_ROWS = 8  # capacity: fixed: the swept rows
_CONFIGS_SWIGLU: Dict[Tuple[int, int], tuple] = {(6144, 1536): (2, 3, 0)} if MXGEMV_VRCFG else {}

# Defaults of the optional config fields: contiguous k-tile ranges per split rank, one TMEM accumulator, the A
# operand staged in TMEM (when its columns fit), st.async split-K partial exchange.
_CONTIG = True
_NACC = 1
_A_TMEM = True
_ST_ASYNC = True
_ensured = set()


_DUMMY: Dict[torch.device, torch.Tensor] = {}


def _dummy_timers(device) -> torch.Tensor:
    """Placeholder for the kernel's timer buffer, which it only writes when compiled with timers on."""
    t = _DUMMY.get(device)
    if t is None:
        t = _DUMMY[device] = torch.zeros(8, dtype=torch.int64, device=device)
    return t


def _arg(t: torch.Tensor, align: int = 16):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t.detach(), assumed_align=align).mark_layout_dynamic(leading_dim=t.dim() - 1)


# (rule 1b, deterministic) The decode projections at up to STRIDED_MAX_ROWS rows split K over strided k-tiles per
# split rank (contig 0: rank r owns k-tiles r, r + split, ...) instead of contiguous ranges: another fixed fp32 order.
# SM107 TP8 4 rows, us/call contiguous -> strided: sparse QKV 2.405 -> 2.371, dense QKV 2.323 -> 2.291,
# o_proj 2.061 -> 2.019, dense gate_up 3.047 -> 3.006, dense down 2.313 -> 2.287; deterministic (3 runs bitwise) and
# max error vs the fp64 product equal to the contiguous order's. Kill: False.
STRIDED_K_TILES = True
STRIDED_MAX_ROWS = 8  # capacity: fixed: the strided-input kernels' built rows
_STRIDED = frozenset({(1536, 6144), (1280, 6144), (6144, 1024), (3072, 6144), (6144, 1536)})

# #117: the producer epilogue serves the sparse-layer QKV + index projection, 12 packed heads of 128 (8 q, k, v,
# index-q, index-k), one head per split-8 cluster.
PROD_N = 12 * 128  # capacity: fixed: the producer's packed heads
# The kernel's producer arguments on a call without the producer epilogue (20 addresses / strides, pages, eps).
PROD_OFF = (0,) * 21 + (0.0,)


def get_config(n: int, k: int, m: Optional[int] = None) -> Optional[tuple]:
    cfg = _CONFIGS_BY_M.get((n, k, m)) if m is not None else None
    if cfg is None and m is not None and m > 8:
        cfg = _CONFIGS_WIDE.get((n, k))
    cfg = cfg or _CONFIGS.get((n, k))
    # The split-K epilogue gives each of the split CTAs 128 // split rows of a 128-row output tile.
    if cfg is None or n % 128 != 0 or k % 128 != 0 or (k // 128) % cfg[0] != 0 or 128 % cfg[0] != 0:
        return None
    if (
        STRIDED_K_TILES
        and m is not None
        and m <= STRIDED_MAX_ROWS
        and (n, k) in _STRIDED
        and len(cfg) == 2
    ):
        cfg = cfg + (0,)
    return cfg


def _declines(n: int, k: int, m: int, x_bf16: bool, cfg: tuple) -> bool:
    """Whether a (split, ring, ...) config declines an [M, K] x [N, K]^T call on shape grounds."""
    split = cfg[0]
    if not 0 < m <= MAX_TOKENS or 128 % split != 0 or (k // 128) % split != 0:
        return True
    tiles = k // 128 // split
    mma_n = _mma_n(m)
    # 17..32 rows at K < 1024 decline (an unsplit K below 1024 is slower than the FlashInfer GEMM there); no TP2
    # projection has K < 1024.
    # The in-kernel bf16 -> MXFP8 quantization (x_bf16) costs one 32-element block per thread and round on 128 threads;
    # beyond 8 rounds per CTA, mxfp8_quantize + this GEMV (or the fallback GEMM) is faster.
    if (16 < m <= 32 and k < 1024) or (x_bf16 and tiles * 4 * mma_n > 8 * 128):
        return True
    return _smem_bytes(min(cfg[1], tiles, 2), tiles, split, mma_n) > _SMEM_LIMIT


def serves(n: int, k: int, m: int, x_bf16: bool) -> bool:
    """Whether the GEMV with its default config takes an [M, K] activation (bf16 or MXFP8) against an [N, K] weight."""
    cfg = get_config(n, k, m)
    return cfg is not None and not _declines(n, k, m, x_bf16, cfg)


def _resolve(
    m: int,
    n: int,
    k: int,
    x_bf16: bool,
    trigger_early: bool,
    cfg: Optional[tuple],
    use_pdl: Optional[bool],
    ar: Optional[tuple],
    x_swiglu: bool = False,
    prod: bool = False,
    gu_swiglu: bool = False,
) -> Optional[tuple]:
    """The compile key of an [M, K] x [N, K]^T call and its runtime push words, or None when the call is declined.

    ``ar`` is the push tuple of ``_run``; only its (world, bucket tokens, mode) enter the key, the addresses and the
    rank are runtime arguments."""
    if cfg is None and x_swiglu and m <= SWIGLU_CFG_MAX_ROWS:
        cfg = _CONFIGS_SWIGLU.get((n, k))
    cfg = cfg or get_config(n, k, m)
    if cfg is None or _declines(n, k, m, x_bf16, cfg):
        return None
    split, ring = cfg[0], cfg[1]
    contig = bool(cfg[2]) if len(cfg) > 2 else _CONTIG
    nacc = int(cfg[3]) if len(cfg) > 3 else _NACC
    a_tmem = bool(cfg[4]) if len(cfg) > 4 else _A_TMEM
    st_async = bool(cfg[5]) if len(cfg) > 5 else _ST_ASYNC
    if prod and not (
        n == PROD_N
        and split == 8
        and m <= split
        and st_async
        and not x_bf16
        and not x_swiglu
        and ar is None
    ):
        # The producer epilogue: the sparse QKV + index projection (one 128-row packed head per cluster of 8), at most
        # one token per cluster rank, the st.async exchange, MXFP8 input, no all-reduce push.
        return None
    if gu_swiglu and not (
        (128 // split // 2) * _mma_n(m) <= 128
        and n % 128 == 0
        and not x_swiglu
        and not prod
        and ar is None
    ):
        # The SwiGLU epilogue: one owner thread per (gate / up pair, token) among the 128 epilogue threads, bf16 out.
        return None
    if nacc != 1:
        # More than one TMEM accumulator is broken on SM107 (nondeterministic or garbage
        # outputs). No shipped table uses it; decline (the caller's fallback runs) until the kernel path is fixed.
        return None
    ring = min(ring, k // 128 // split)
    tiles = k // 128 // split
    mma_n = _mma_n(m)
    while ring > 2 and _smem_bytes(ring, tiles, split, mma_n) > _SMEM_LIMIT:
        ring -= 1
    if a_tmem:
        from .m3_mx_gemv_kernel import tmem_cols

        if tmem_cols(ring, nacc, True, tiles, mma_n) > 512:
            a_tmem = False
    # ar = (multicast workspace address, flags address, rank, world, bucket tokens, mode): push the output rows into
    # the next m3_mc_allreduce call's exchange slots (mode 2 also defers the dependents' launch until after the push).
    ar_ws, ar_flags, ar_rank, ar_mode, ar_world, ar_maxt = 0, 0, 0, 0, 1, 8
    if ar is not None and split <= 4 and n % 2 == 0:
        ar_ws, ar_flags, ar_rank, ar_world, ar_maxt, ar_mode = ar
        if ar_mode == 2:
            trigger_early = False
    if use_pdl is None:
        use_pdl = get_env_enable_pdl()
    # The kernel's ``timed`` flag (per-CTA milestone timers) is always off.
    key = (
        n,
        k,
        split,
        ring,
        trigger_early,
        contig,
        False,
        nacc,
        a_tmem,
        st_async,
        x_bf16,
        ar_mode,
        ar_world,
        ar_maxt,
        mma_n,
        use_pdl,
        x_swiglu,
        prod,
        gu_swiglu,
    )
    return key, (ar_ws, ar_flags, ar_rank)


def _compile(
    key: tuple, args: tuple, m: int, runtime_ar: tuple, stream, extra: tuple, prod_args: tuple
):
    """cute.compile the kernel of ``key`` (see ``_resolve``) on these argument views; cached."""
    import cutlass.cute as cute

    from . import m3_mx_gemv_kernel as kernel

    (
        n,
        k,
        split,
        ring,
        trigger_early,
        contig,
        timed,
        nacc,
        a_tmem,
        st_async,
        x_bf16,
        ar_mode,
        ar_world,
        ar_maxt,
        mma_n,
        use_pdl,
        x_swiglu,
        prod,
        gu_swiglu,
    ) = key
    ar_ws, ar_flags, ar_rank = runtime_ar
    with _lock:
        fn = _compiled.get(key)
        if fn is None:
            fn = _compiled[key] = cute_aot.compile_or_load(
                f"m3_mx_gemv {key}",
                lambda: cute.compile(
                    kernel.m3_mx_gemv,
                    *args,
                    m,
                    n,
                    k,
                    split,
                    ring,
                    trigger_early,
                    contig,
                    timed,
                    nacc,
                    a_tmem,
                    st_async,
                    x_bf16,
                    ar_ws,
                    ar_flags,
                    ar_rank,
                    ar_mode,
                    ar_world,
                    ar_maxt,
                    mma_n,
                    use_pdl,
                    stream,
                    *extra,
                    False,
                    False,
                    x_swiglu,
                    gu_swiglu,
                    *prod_args,
                    prod,
                ),
                sources=_aot_sources(),
            )
    return fn


def _views(
    x: torch.Tensor,
    xsf: Optional[torch.Tensor],
    weight,
    weight_scale,
    y,
    x_bf16: bool,
    x_swiglu: bool = False,
) -> tuple:
    # x_swiglu: x is [M, 2K] (gate | up); the GEMV's K is half its width.
    k = x.shape[1] // 2 if x_swiglu else x.shape[1]
    if x_bf16:
        x_bytes = x.view(torch.uint8)
        xsf = _placeholder_scales(k, x.device)
        x_words = x.view(torch.int32)
    else:
        x_bytes = x.view(torch.uint8)
        x_words = _placeholder_words(x.device)
    return (
        _arg(weight.view(torch.uint8)),
        _arg(weight_scale.view(torch.uint8).reshape(-1)),
        _arg(x_bytes),
        _arg(xsf.view(torch.uint8).reshape(-1)),
        _arg(y.view(-1) if y.numel() else _placeholder_words(x.device).view(torch.bfloat16)),
        _arg(x_words.reshape(-1)),
        _arg(_dummy_timers(x.device)),
    )


def _run(
    x: torch.Tensor,
    xsf: Optional[torch.Tensor],
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_bf16: bool,
    trigger_early: bool,
    cfg: Optional[tuple],
    use_pdl: Optional[bool],
    ar: Optional[tuple] = None,
    x_swiglu: bool = False,
    prod_args: Optional[tuple] = None,
    gu_swiglu: bool = False,
) -> Optional[torch.Tensor]:
    global last_pushed, last_uncompiled
    last_pushed = False
    last_uncompiled = None
    import cuda.bindings.driver as cuda_driver

    m, k = x.shape
    if x_swiglu:
        k //= 2
    n = weight.shape[0]
    if weight.shape[1] != k:
        return None
    prod = prod_args is not None
    resolved = _resolve(m, n, k, x_bf16, trigger_early, cfg, use_pdl, ar, x_swiglu, prod, gu_swiglu)
    if resolved is None:
        return None
    key, runtime_ar = resolved
    if not prod:
        prod_args = PROD_OFF
    fn = _compiled.get(key)
    if fn is None:
        if _frozen[0]:
            # Compiles are closed (the model built every kernel it dispatches to before warmup): decline, the caller
            # reports the miss.
            last_uncompiled = key
            return None
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "m3_mx_gemv must run once per shape outside CUDA-graph capture first"
            )
    # prod: no packed row is stored (the producer epilogue consumes it); y is the placeholder.
    y = torch.empty(
        0 if prod else m, n // 2 if gu_swiglu else n, dtype=torch.bfloat16, device=x.device
    )
    args = _views(x, xsf, weight, weight_scale, y, x_bf16, x_swiglu)
    extra = _extra_args(x.device)
    stream = cuda_driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    if fn is None:
        fn = _compile(key, args, m, runtime_ar, stream, extra, prod_args)
    ar_ws, ar_flags, ar_rank = runtime_ar
    fn(*args, m, ar_ws, ar_flags, ar_rank, stream, *extra, *prod_args)
    last_pushed = key[11] > 0
    return y


def precompile(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    m: int,
    x_bf16: bool,
    use_pdl: Optional[bool] = None,
    ar: Optional[Tuple[int, int, int]] = None,
    x_swiglu: bool = False,
    prod: bool = False,
    gu_swiglu: bool = False,
) -> Optional[tuple]:
    """Compile, without launching, the kernel an M-row call against ``weight`` takes; returns its key, or None when
    such a call is declined. ``ar`` = (world, bucket tokens, mode) of a pushing call (its addresses and rank are
    runtime arguments, so none is needed here). Runs outside capture, even while compiles are frozen."""
    import cuda.bindings.driver as cuda_driver

    n, k = weight.shape
    push = None if ar is None else (0, 0, 0, *ar)
    resolved = _resolve(m, n, k, x_bf16, True, None, use_pdl, push, x_swiglu, prod, gu_swiglu)
    if resolved is None:
        return None
    key, runtime_ar = resolved
    if key in _compiled:
        return key
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("m3_mx_gemv.precompile allocates and compiles: call it outside capture")
    dev = weight.device
    x = torch.zeros(
        m,
        2 * k if x_swiglu else k,
        dtype=torch.bfloat16 if x_bf16 else torch.float8_e4m3fn,
        device=dev,
    )
    xsf = None if x_bf16 else torch.full((128 * (k // 32),), 127, dtype=torch.uint8, device=dev)
    y = torch.empty(m, n // 2 if gu_swiglu else n, dtype=torch.bfloat16, device=dev)
    args = _views(x, xsf, weight, weight_scale, y, x_bf16, x_swiglu)
    stream = cuda_driver.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    _compile(key, args, m, runtime_ar, stream, _extra_args(dev), PROD_OFF)
    return key


def freeze(frozen: bool = True) -> None:
    """Close (or reopen) compiles: while frozen, a call whose kernel is not compiled yet declines (returns None and
    sets ``last_uncompiled``) instead of compiling. ``precompile`` and ``ensure_compiled`` still compile."""
    _frozen[0] = bool(frozen)


def frozen() -> bool:
    return _frozen[0]


def is_compiled(key: tuple) -> bool:
    return key in _compiled


def _extra_args(device) -> tuple:
    """Placeholder arguments of the kernel's draft-input prologue and RoPE / KV epilogue (compiled off)."""
    i32 = _placeholder_words(device)
    f32 = i32.view(torch.float32)
    u8 = i32.view(torch.uint8)
    d = (_arg(i32), _arg(i32), _arg(i32), _arg(i32), _arg(f32), 0, _arg(i32), _arg(i32), 0)
    r = (_arg(f32), _arg(i32), _arg(i32), _arg(u8), _arg(u8), _arg(f32), _arg(i32))
    return d + r


def m3_mx_gemv(
    xq: torch.Tensor,
    xsf: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    trigger_early: bool = True,
    cfg: Optional[tuple] = None,
    use_pdl: Optional[bool] = None,
    ar: Optional[tuple] = None,
) -> Optional[torch.Tensor]:
    """``xq`` E4M3 [M <= 8, K] with ``xsf`` its 128x4-swizzled UE8M0 scales; ``weight`` E4M3 [N, K] with
    ``weight_scale`` swizzled. Returns bf16 [M, N], or None when the shape is not served."""
    return _run(xq, xsf, weight, weight_scale, False, trigger_early, cfg, use_pdl, ar)


def m3_mx_gemv_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    trigger_early: bool = True,
    cfg: Optional[tuple] = None,
    use_pdl: Optional[bool] = None,
    ar: Optional[tuple] = None,
) -> Optional[torch.Tensor]:
    """The same GEMV for a bf16 ``x`` [M <= 8, K] (contiguous), quantized to MXFP8 inside the kernel exactly as
    torch.ops.trtllm.mxfp8_quantize(x, True) does."""
    if x.dtype != torch.bfloat16 or x.dim() != 2 or not x.is_contiguous() or x.data_ptr() % 16 != 0:
        return None
    return _run(x, None, weight, weight_scale, True, trigger_early, cfg, use_pdl, ar)


def m3_mx_gemv_swiglu(
    gate_up: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    trigger_early: bool = True,
    cfg: Optional[tuple] = None,
    use_pdl: Optional[bool] = None,
    ar: Optional[tuple] = None,
) -> Optional[torch.Tensor]:
    """``m3_mx_gemv_bf16(swiglu(gate_up), ...)`` in one launch: ``gate_up`` bf16 [M <= 8, 2K] (contiguous, gate then
    up) is activated in the kernel's quantize prologue with the swigluoai arithmetic of the Triton silu_and_mul kernel
    (alpha 1.702, limit 7, the (up + 1) offset; bit for bit), then quantized as the bf16 path does."""
    if (
        gate_up.dtype != torch.bfloat16
        or gate_up.dim() != 2
        or not gate_up.is_contiguous()
        or gate_up.data_ptr() % 16 != 0
        or gate_up.shape[1] % 2 != 0
    ):
        return None
    return _run(gate_up, None, weight, weight_scale, True, trigger_early, cfg, use_pdl, ar, True)


def gu_interleave(weight: torch.Tensor) -> torch.Tensor:
    """A gate_up weight [2F, K] (gate rows then up rows, F a multiple of 64) reordered per 128-row tile for the SwiGLU
    epilogue: tile i holds gate rows 64 i .. 64 i + 63 then up rows 64 i .. 64 i + 63. Rows only move, so each output
    feature's dot product (bytes and K order) is unchanged."""
    f = weight.shape[0] // 2
    gate, up = weight[:f].reshape(f // 64, 64, -1), weight[f:].reshape(f // 64, 64, -1)
    return torch.cat([gate, up], dim=1).reshape(2 * f, -1).contiguous()


def m3_mx_gemv_gu_swiglu(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    trigger_early: bool = True,
    use_pdl: Optional[bool] = None,
) -> Optional[torch.Tensor]:
    """``swiglu(m3_mx_gemv_bf16(x, gate_up))`` in one launch: ``weight`` / ``weight_scale`` are the MXFP8 form of a
    ``gu_interleave``d gate_up [2F, K]; returns bf16 h [M, F], bit for bit the bf16 GEMV's gate_up rows followed by the
    swigluoai of the Triton silu_and_mul kernel (alpha 1.702, limit 7, (up + 1)). None when not served."""
    if x.dtype != torch.bfloat16 or x.dim() != 2 or not x.is_contiguous() or x.data_ptr() % 16 != 0:
        return None
    return _run(
        x, None, weight, weight_scale, True, trigger_early, None, use_pdl, None, False, None, True
    )


def m3_mx_gemv_producer(
    xq: torch.Tensor,
    xsf: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    q: torch.Tensor,
    index_q: torch.Tensor,
    kv_data: torch.Tensor,
    kv_scale: torch.Tensor,
    index_k: torch.Tensor,
    out_cache_loc: torch.Tensor,
    kv_quant_scale: torch.Tensor,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    index_q_weight: torch.Tensor,
    index_k_weight: torch.Tensor,
    rotary_cos_sin: torch.Tensor,
    position_ids: torch.Tensor,
    trigger_early: bool = True,
    use_pdl: Optional[bool] = None,
) -> bool:
    """#117: ``m3_producer(m3_mx_gemv(xq, xsf, weight, weight_scale), ...)`` in one launch, bit for bit: the QKV +
    index GEMV of the sparse layer (``weight`` [12 x 128, K]) whose owner epilogue hands each token's packed head to
    the producer phase (``m3_layer.phase_producer.producer_head``). Writes ``q`` FP8 [T, 8, 128] and ``index_q`` FP8
    [T, 1, 128] and the token's cache slot (NVFP4 K / V + scales, FP8 index-K) exactly as the producer does; the
    caches and tensors are those ``minimax_m3_producer_pw`` takes (kv_data [pages, 2, 1, 128, 64] u8, kv_scale [pages,
    2, 1, 128, 8] u8, index_k [pages, 1, 128, 128] e4m3). Returns False (nothing launched) when the call is declined.
    """
    m = xq.shape[0]
    if (
        m > 8
        or weight.shape[0] != PROD_N
        or q.shape != (m, 8, 128)
        or index_q.shape != (m, 1, 128)
        or not (q.is_contiguous() and index_q.is_contiguous())
        or kv_data.dim() != 5
        or kv_scale.dim() != 5
        or index_k.dim() != 4
    ):
        return False
    tensors = (
        q,
        index_q,
        index_k,
        kv_data,
        kv_scale,
        out_cache_loc,
        kv_quant_scale,
        position_ids,
        rotary_cos_sin,
        q_weight,
        k_weight,
        index_q_weight,
        index_k_weight,
    )
    prod_args = (
        *(t.data_ptr() for t in tensors),
        index_k.stride(0),
        index_k.stride(2),
        kv_data.stride(0),
        kv_data.stride(1),
        kv_data.stride(3),
        kv_scale.stride(0),
        kv_scale.stride(1),
        kv_data.shape[0],
        float(eps),
    )
    out = _run(
        xq, xsf, weight, weight_scale, False, trigger_early, None, use_pdl, None, False, prod_args
    )
    return out is not None


_PLACEHOLDERS: Dict[tuple, torch.Tensor] = {}
# Whether the last call pushed its output into an all-reduce (see _run's ``ar``).
last_pushed = False
# The key of the last call declined because compiles are frozen and its kernel was not compiled (see ``freeze``).
last_uncompiled: Optional[tuple] = None
_frozen = [False]


def _placeholder_scales(k: int, device) -> torch.Tensor:
    key = ("sf", k, device)
    t = _PLACEHOLDERS.get(key)
    if t is None:
        t = _PLACEHOLDERS[key] = torch.zeros(k // 128 * 512, dtype=torch.uint8, device=device)
    return t


def _placeholder_words(device) -> torch.Tensor:
    key = ("w", device)
    t = _PLACEHOLDERS.get(key)
    if t is None:
        t = _PLACEHOLDERS[key] = torch.zeros(16, dtype=torch.int32, device=device)
    return t


def ensure_compiled(weight: torch.Tensor, weight_scale: torch.Tensor, use_pdl: bool) -> None:
    """Compiles the kernel of this weight's shape outside capture (a one-token dummy call)."""
    if (
        weight.dim() != 2
        or weight.dtype != torch.float8_e4m3fn
        or get_config(*weight.shape) is None
    ):
        return
    n, k = weight.shape
    if (n, k, use_pdl) in _ensured:
        return
    _ensured.add((n, k, use_pdl))
    xq = torch.zeros(1, k, dtype=torch.float8_e4m3fn, device=weight.device)
    xsf = torch.full((128 * (k // 32),), 127, dtype=torch.uint8, device=weight.device)
    m3_mx_gemv(xq, xsf, weight, weight_scale, use_pdl=use_pdl)
    for width in MMA_WIDTHS[1:]:
        m3_mx_gemv(
            torch.zeros(width, k, dtype=torch.float8_e4m3fn, device=weight.device),
            xsf,
            weight,
            weight_scale,
            use_pdl=use_pdl,
        )
    m3_mx_gemv_bf16(
        torch.zeros(1, k, dtype=torch.bfloat16, device=weight.device),
        weight,
        weight_scale,
        use_pdl=use_pdl,
    )


def _aot_sources():
    """The Python files this module's CuTe DSL kernels are traced from (the target package and the shared Blackwell
    kernel helpers): the ``sources`` of their shipped AOT objects."""
    target = Path(__file__).resolve().parents[2]
    shared = Path(cute_aot.__file__).resolve().parent / "cute_dsl_kernels" / "blackwell"
    return cute_aot.py_files(str(target), str(shared))
