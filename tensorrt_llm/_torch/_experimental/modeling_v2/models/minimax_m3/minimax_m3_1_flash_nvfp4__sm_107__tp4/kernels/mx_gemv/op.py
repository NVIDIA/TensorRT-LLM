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
"""MXFP8 x MXFP8 decode GEMV (tcgen05, block-scaled) for already-quantized activations of at most 8 tokens.

Compiled on the first call per (N, K, split, ring, early trigger, PDL, push, MMA width), which must happen outside
CUDA-graph capture, or ahead of any call by ``precompile``; once ``freeze`` closes compiles, a call whose kernel is not
compiled declines instead.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import cute_aot
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

# (N, K) -> (split, ring): SM103 exposed time behind a 4.25 us PDL predecessor at M 1 / 8, vs the fused low-M GEMM:
# QKV 2.78 / 3.16 vs 4.58 / 6.21 us, o_proj 2.23 / 2.41 vs 4.40 / 5.09, shared gate_up 2.43 / 2.74 vs 3.56 / 4.30,
# shared down 1.74 / 1.66 vs 2.92 / 3.38. A config is (split, ring[, contig, nacc, a_tmem, st_async]); the optional
# fields default to _CONTIG, _NACC, _A_TMEM, _ST_ASYNC.
_CONFIGS: Dict[Tuple[int, int], tuple] = {
    (2560, 6144): (4, 12),
    (6144, 2048): (2, 8),
    # Shared-expert gate_up: split 2, ring 12, single accumulator, A operand from shared memory.
    (1536, 6144): (2, 12, 1, 1, 0, 1),
    # Shared-expert down: unsplit, ring 3, single accumulator, A operand from shared memory.
    (6144, 768): (1, 3, 1, 1, 0, 1),
    (2304, 6144): (4, 12),
    (6144, 3072): (2, 12),
    # Dense-layer gate_up: 6.4 / 6.3 us at M 1 / 8 vs FlashInfer 8.8.
    (6144, 6144): (2, 12),
}
# Per-token-count configs (N, K, M): the grid that suits the concurrent routed-expert engine depends on how long
# that engine runs. At 2 tokens the shared-expert gate_up splits K over 4 CTAs.
_CONFIGS_BY_M: Dict[Tuple[int, int, int], tuple] = {
    (1536, 6144, 2): (4, 12, 1, 1, 0, 1),
}
# Above 8 tokens (spec-decode verify rows): the shared-expert gate_up splits K over 4 CTAs.
_CONFIGS_WIDE: Dict[Tuple[int, int], tuple] = {
    (1536, 6144): (4, 12, 1, 1, 0, 1),
}

# Defaults of the optional config fields: contiguous k-tile ranges per split rank, one TMEM accumulator, the A
# operand staged in TMEM (when its columns fit), st.async split-K partial exchange.
_CONTIG = True
_NACC = 1
_A_TMEM = True
_ST_ASYNC = True
_ensured = set()

# GEMV_PUSHOPT: the push epilogue of split-K calls that push into the next boundary all-reduce (mode 2: the o_proj, the
# dense down projection) takes the kernel's PO_EARLY_PARITY | PO_VEC16 (m3_mx_gemv_kernel): the call parities loaded
# under the activation loads and MMAs instead of after a second grid wait, and one 16-byte multicast store per lane
# instead of 8 4-byte stores per lane pair. Bit for bit the same y and the same pushed words. SM107 TP4 attention
# boundary chains (tests/microbenchmarks/m31_dense: 8 rows, 24 layers, 4 ranks, the real m3_lc_attn before the
# o_proj, the real multicast all-reduce after it): -0.6..-1.0 us per layer. (Also tried and dropped: releasing the
# all-reduce once the accumulators are ready, +0.14..+0.34 us per layer; y stored after the pushes, flat.) Kill switch
# TRTLLM_M31_DISABLE_GEMV_PUSHOPT=1 (the deployed epilogue), read at import.
GEMV_PUSHOPT = not m31_disabled("GEMV_PUSHOPT")
PUSH_OPT = 1 | 2  # PO_EARLY_PARITY | PO_VEC16

# The producer epilogue (m3_mx_gemv_kernel PR_*): the sparse layers' fused QKV + index projection (20 heads of 128)
# runs m3_producer's norm / RoPE / quantize / cache insert in its split-K epilogue (``producer_args``). Bits: PR_ON 1,
# PR_Y 2 (y stored too), PR_PREWAIT 4 (step inputs read before the dependency).
PR_N = 20 * 128
PR_ON = 1
PR_Y = 2
PR_PREWAIT = 4


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


def get_config(n: int, k: int, m: Optional[int] = None) -> Optional[tuple]:
    cfg = _CONFIGS_BY_M.get((n, k, m)) if m is not None else None
    if cfg is None and m is not None and m > 8:
        cfg = _CONFIGS_WIDE.get((n, k))
    cfg = cfg or _CONFIGS.get((n, k))
    # The split-K epilogue gives each of the split CTAs 128 // split rows of a 128-row output tile.
    if cfg is None or n % 128 != 0 or k % 128 != 0 or (k // 128) % cfg[0] != 0 or 128 % cfg[0] != 0:
        return None
    return cfg


def _declines(n: int, k: int, m: int, x_bf16: bool, cfg: tuple) -> bool:
    """Whether a (split, ring, ...) config declines an [M, K] x [N, K]^T call on shape grounds."""
    split = cfg[0]
    if not 0 < m <= MAX_TOKENS or 128 % split != 0 or (k // 128) % split != 0:
        return True
    tiles = k // 128 // split
    mma_n = _mma_n(m)
    # SM103 exposed time vs the FlashInfer prequantized GEMM (behind a PDL predecessor), at 24 / 48 tokens: QKV 3.76 /
    # 4.63 vs 7.46 / 11.44 us, o_proj 2.74 / 3.33 vs 4.46 / 5.84, shared gate_up 4.57 / 5.71 vs 7.16 / 11.25. Behind
    # only for K = 768 (unsplit shared down) at 17-32 tokens: 3.1 vs 2.9.
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
    prod: int = 0,
) -> Optional[tuple]:
    """The compile key of an [M, K] x [N, K]^T call and its runtime push words, or None when the call is declined.

    ``ar`` is the push tuple of ``_run``; only its (world, bucket tokens, mode) enter the key, the addresses and the
    rank are runtime arguments. ``prod`` (the kernel's PR_* bits) asks for the producer epilogue (see
    ``producer_args``); a call it cannot take (not the 20-head QKV tile, over 8 rows, a split not dividing 8, a push,
    no st.async exchange) is declined."""
    cfg = cfg or get_config(n, k, m)
    if cfg is None or _declines(n, k, m, x_bf16, cfg):
        return None
    split, ring = cfg[0], cfg[1]
    contig = bool(cfg[2]) if len(cfg) > 2 else _CONTIG
    nacc = int(cfg[3]) if len(cfg) > 3 else _NACC
    a_tmem = bool(cfg[4]) if len(cfg) > 4 else _A_TMEM
    st_async = bool(cfg[5]) if len(cfg) > 5 else _ST_ASYNC
    # a_tiles (a_tmem): only the first a_tiles k-tiles' A goes to TMEM, the rest is read from the smem ring, which must
    # then hold every k-tile (never refilled); 0 = all in TMEM.
    a_tiles = int(cfg[6]) if len(cfg) > 6 else 0
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

        if tmem_cols(ring, nacc, True, tiles, mma_n, a_tiles) > 512:
            a_tmem = False
    if not a_tmem or not 0 < a_tiles < tiles:
        a_tiles = 0
    elif ring < tiles:
        return None
    # ar = (multicast workspace address, flags address, rank, world, bucket tokens, mode): push the output rows into
    # the next m3_mc_allreduce call's exchange slots (mode 2 also defers the dependents' launch until after the push).
    ar_ws, ar_flags, ar_rank, ar_mode, ar_world, ar_maxt = 0, 0, 0, 0, 1, 8
    push_opt = 0
    if ar is not None and split <= 4 and n % 2 == 0:
        ar_ws, ar_flags, ar_rank, ar_world, ar_maxt, ar_mode = ar
        if ar_mode == 2:
            trigger_early = False
            if GEMV_PUSHOPT and split > 1:
                push_opt = PUSH_OPT
    if prod:
        if (
            n != PR_N
            or mma_n != 8
            or not 1 < split <= 8
            or 8 % split != 0
            or not st_async
            or ar is not None
            or x_swiglu
        ):
            return None
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
        push_opt,
        a_tiles,
        int(prod),
    )
    return key, (ar_ws, ar_flags, ar_rank)


def _compile(key: tuple, args: tuple, m: int, runtime_ar: tuple, stream, extra: tuple):
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
        push_opt,
        a_tiles,
        prod,
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
                    push_opt,
                    a_tiles,
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
    prod: Optional[tuple] = None,
) -> Optional[torch.Tensor]:
    global last_pushed, last_uncompiled, last_prod
    last_pushed = False
    last_prod = False
    last_uncompiled = None
    import cuda.bindings.driver as cuda_driver

    m, k = x.shape
    if x_swiglu:
        k //= 2
    n = weight.shape[0]
    if weight.shape[1] != k:
        return None
    resolved = _resolve(
        m, n, k, x_bf16, trigger_early, cfg, use_pdl, ar, x_swiglu, 0 if prod is None else prod[0]
    )
    if resolved is None:
        return None
    key, runtime_ar = resolved
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
    y = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    args = _views(x, xsf, weight, weight_scale, y, x_bf16, x_swiglu)
    extra = _extra_args(x.device, None if prod is None else prod[1])
    stream = cuda_driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    if fn is None:
        fn = _compile(key, args, m, runtime_ar, stream, extra)
    ar_ws, ar_flags, ar_rank = runtime_ar
    fn(*args, m, ar_ws, ar_flags, ar_rank, stream, *extra)
    last_pushed = key[11] > 0
    last_prod = key[19] != 0
    return y


def precompile(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    m: int,
    x_bf16: bool,
    use_pdl: Optional[bool] = None,
    ar: Optional[Tuple[int, int, int]] = None,
    x_swiglu: bool = False,
    cfg: Optional[tuple] = None,
    prod: int = 0,
) -> Optional[tuple]:
    """Compile, without launching, the kernel an M-row call against ``weight`` takes; returns its key, or None when
    such a call is declined. ``ar`` = (world, bucket tokens, mode) of a pushing call (its addresses and rank are
    runtime arguments, so none is needed here); ``cfg`` a (split, ring, ...) override; ``prod`` the producer
    epilogue's PR_* bits. Runs outside capture, even while compiles are frozen."""
    import cuda.bindings.driver as cuda_driver

    n, k = weight.shape
    push = None if ar is None else (0, 0, 0, *ar)
    resolved = _resolve(m, n, k, x_bf16, True, cfg, use_pdl, push, x_swiglu, prod)
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
    y = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
    args = _views(x, xsf, weight, weight_scale, y, x_bf16, x_swiglu)
    stream = cuda_driver.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    _compile(key, args, m, runtime_ar, stream, _extra_args(dev))
    return key


def freeze(frozen: bool = True) -> None:
    """Close (or reopen) compiles: while frozen, a call whose kernel is not compiled yet declines (returns None and
    sets ``last_uncompiled``) instead of compiling. ``precompile`` and ``ensure_compiled`` still compile."""
    _frozen[0] = bool(frozen)


def frozen() -> bool:
    return _frozen[0]


def is_compiled(key: tuple) -> bool:
    return key in _compiled


def _extra_args(device, pr: Optional[tuple] = None) -> tuple:
    """Placeholder arguments of the kernel's draft-input prologue and RoPE / KV epilogue (compiled off), then the
    producer epilogue's (``producer_args``; zeros when off)."""
    i32 = _placeholder_words(device)
    f32 = i32.view(torch.float32)
    u8 = i32.view(torch.uint8)
    d = (_arg(i32), _arg(i32), _arg(i32), _arg(i32), _arg(f32), 0, _arg(i32), _arg(i32), 0)
    r = (_arg(f32), _arg(i32), _arg(i32), _arg(u8), _arg(u8), _arg(f32), _arg(i32))
    return d + r + (_PR_OFF if pr is None else pr)


# The producer epilogue's 24 runtime arguments when it is compiled off.
_PR_OFF = (0,) * 22 + (0, 0.0)


def producer_args(
    q: torch.Tensor,
    index_q: torch.Tensor,
    kv_data: torch.Tensor,
    kv_scale: torch.Tensor,
    index_k: torch.Tensor,
    out_cache_loc: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_cos_sin: torch.Tensor,
    kv_quant_scale: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    index_q_weight: torch.Tensor,
    index_k_weight: torch.Tensor,
    eps: float,
) -> Optional[tuple]:
    """The producer epilogue's runtime arguments for m3_producer's operands (ops/attention/m3_producer, the TP4
    heads: q [T, 16, 128] and index-q [T, 1, 128] E4M3 outputs, the NVFP4 pool [pages, 2, 1, 128, 64] uint8 and its
    scales [pages, 2, 1, 128, 8], the E4M3 index-K [pages, 1, 128, 128], int32 slots and position ids, the fp32 RoPE
    table [positions, 2, 32] and KV quant scales, the bf16 [128] norm weights), or None when one is not the layout the
    producer kernel (and so this epilogue) is built for. ``eps`` is rounded to fp32 as the producer's op does."""
    import struct

    t = q.shape[0]
    ok = (
        q.dtype == torch.float8_e4m3fn
        and q.is_contiguous()
        and tuple(q.shape[1:]) == (16, 128)
        and index_q.dtype == torch.float8_e4m3fn
        and index_q.is_contiguous()
        and tuple(index_q.shape) == (t, 1, 128)
        and kv_data.dtype == torch.uint8
        and kv_data.dim() == 5
        and tuple(kv_data.shape[1:]) == (2, 1, 128, 64)
        and kv_data.stride()[2:] == (128 * 64, 64, 1)
        and kv_data.stride(0) % 2 == 0
        and kv_data.stride(1) % 2 == 0
        and kv_data.data_ptr() % 2 == 0
        and kv_scale.dtype == torch.uint8
        and kv_scale.dim() == 5
        and tuple(kv_scale.shape[1:]) == (2, 1, 128, 8)
        and kv_scale.stride()[2:] == (128 * 8, 8, 1)
        and kv_scale.shape[0] == kv_data.shape[0]
        and index_k.dtype == torch.float8_e4m3fn
        and index_k.dim() == 4
        and tuple(index_k.shape[1:]) == (1, 128, 128)
        and index_k.shape[0] == kv_data.shape[0]
        and index_k.stride(3) == 1
        and index_k.stride(2) == 128
        and index_k.stride(0) % 4 == 0
        and index_k.data_ptr() % 4 == 0
        and out_cache_loc.dtype == torch.int32
        and out_cache_loc.is_contiguous()
        and out_cache_loc.numel() >= t
        and position_ids.dtype == torch.int32
        and position_ids.is_contiguous()
        and position_ids.numel() >= t
        and rotary_cos_sin.dtype == torch.float32
        and rotary_cos_sin.is_contiguous()
        and rotary_cos_sin.dim() == 3
        and tuple(rotary_cos_sin.shape[1:]) == (2, 32)
        and rotary_cos_sin.data_ptr() % 16 == 0
        and kv_quant_scale.dtype == torch.float32
        and kv_quant_scale.is_contiguous()
        and kv_quant_scale.numel() >= 3
        and all(
            w.dtype == torch.bfloat16
            and w.is_contiguous()
            and w.numel() == 128
            and w.data_ptr() % 8 == 0
            for w in (q_weight, k_weight, index_q_weight, index_k_weight)
        )
    )
    if not ok:
        return None
    eps32 = struct.unpack("f", struct.pack("f", float(eps)))[0]
    return (
        q.data_ptr(),
        index_q.data_ptr(),
        kv_data.data_ptr(),
        kv_scale.data_ptr(),
        index_k.data_ptr(),
        out_cache_loc.data_ptr(),
        position_ids.data_ptr(),
        rotary_cos_sin.data_ptr(),
        kv_quant_scale.data_ptr(),
        q_weight.data_ptr(),
        k_weight.data_ptr(),
        index_q_weight.data_ptr(),
        index_k_weight.data_ptr(),
        kv_data.stride(0),
        kv_data.stride(1),
        kv_data.stride(2),
        kv_data.stride(3),
        kv_scale.stride(0),
        kv_scale.stride(1),
        kv_scale.stride(2),
        index_k.stride(0),
        index_k.stride(2),
        kv_data.shape[0],
        eps32,
    )


def m3_mx_gemv(
    xq: torch.Tensor,
    xsf: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    trigger_early: bool = True,
    cfg: Optional[tuple] = None,
    use_pdl: Optional[bool] = None,
    ar: Optional[tuple] = None,
    prod: Optional[tuple] = None,
) -> Optional[torch.Tensor]:
    """``xq`` E4M3 [M <= 8, K] with ``xsf`` its 128x4-swizzled UE8M0 scales; ``weight`` E4M3 [N, K] with
    ``weight_scale`` swizzled. Returns bf16 [M, N], or None when the shape is not served. ``prod`` = (PR_* bits,
    ``producer_args``) runs the producer epilogue (``last_prod``); y then holds the output only with PR_Y."""
    return _run(xq, xsf, weight, weight_scale, False, trigger_early, cfg, use_pdl, ar, False, prod)


def m3_mx_gemv_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    trigger_early: bool = True,
    cfg: Optional[tuple] = None,
    use_pdl: Optional[bool] = None,
    ar: Optional[tuple] = None,
    prod: Optional[tuple] = None,
) -> Optional[torch.Tensor]:
    """The same GEMV for a bf16 ``x`` [M <= 8, K] (contiguous), quantized to MXFP8 inside the kernel exactly as
    torch.ops.trtllm.mxfp8_quantize(x, True) does. ``prod`` as for ``m3_mx_gemv``."""
    if x.dtype != torch.bfloat16 or x.dim() != 2 or not x.is_contiguous() or x.data_ptr() % 16 != 0:
        return None
    return _run(x, None, weight, weight_scale, True, trigger_early, cfg, use_pdl, ar, False, prod)


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


_PLACEHOLDERS: Dict[tuple, torch.Tensor] = {}
# Whether the last call pushed its output into an all-reduce (see _run's ``ar``).
last_pushed = False
# Whether the last call ran the producer epilogue (see _run's ``prod``).
last_prod = False
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
    shared = cute_aot.BLACKWELL_DIR
    return cute_aot.py_files(str(target), str(shared))
