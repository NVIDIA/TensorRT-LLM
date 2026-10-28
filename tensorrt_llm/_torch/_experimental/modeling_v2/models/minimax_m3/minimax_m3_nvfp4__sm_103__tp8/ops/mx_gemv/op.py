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
"""MXFP8 x MXFP8 decode GEMV (tcgen05, block-scaled) at the TP8 per-rank projection shapes.

Compiled on the first call per (N, K, split, ring, early trigger, PDL, push, MMA width), which must happen outside
CUDA-graph capture, or ahead of any call by ``precompile``; once ``freeze`` closes compiles, a call whose kernel is not
compiled declines instead.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

import torch

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

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

# (N, K) -> (split, ring[, contig, nacc, a_tmem, st_async]) of the TP8 per-rank projections: GB300 exposed time
# behind a PDL predecessor. The optional fields default to _CONTIG, _NACC, _A_TMEM, _ST_ASYNC. The sparse-layer QKV
# splits K over 8 CTAs (96 CTAs at 1 token; bit-identical to splits 4 and 2).
_CONFIGS: Dict[Tuple[int, int], tuple] = {
    # sparse-layer QKV + index projection
    (1536, 6144): (8, 8),
    # dense-layer QKV
    (1280, 6144): (4, 12),
    # o_proj
    (6144, 1024): (4, 8),
    # dense-layer gate_up
    (3072, 6144): (4, 12),
    # dense-layer down
    (6144, 1536): (2, 12),
}
# Per-token-count configs (N, K, M): at 2 tokens the sparse-layer QKV splits K over 4 CTAs, A operand from shared
# memory.
_CONFIGS_BY_M: Dict[Tuple[int, int, int], tuple] = {
    (1536, 6144, 2): (4, 12, 1, 1, 0, 1),
}
# Above 8 tokens (spec-decode verify rows): the same 4-CTA split for the sparse-layer QKV.
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
    # 17..32 rows at K < 1024 decline (an unsplit K below 1024 is slower than the FlashInfer GEMM there); no TP8
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
) -> Optional[tuple]:
    """The compile key of an [M, K] x [N, K]^T call and its runtime push words, or None when the call is declined.

    ``ar`` is the push tuple of ``_run``; only its (world, bucket tokens, mode) enter the key, the addresses and the
    rank are runtime arguments."""
    cfg = cfg or get_config(n, k, m)
    if cfg is None or _declines(n, k, m, x_bf16, cfg):
        return None
    split, ring = cfg[0], cfg[1]
    contig = bool(cfg[2]) if len(cfg) > 2 else _CONTIG
    nacc = int(cfg[3]) if len(cfg) > 3 else _NACC
    a_tmem = bool(cfg[4]) if len(cfg) > 4 else _A_TMEM
    st_async = bool(cfg[5]) if len(cfg) > 5 else _ST_ASYNC
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
    ) = key
    ar_ws, ar_flags, ar_rank = runtime_ar
    with _lock:
        fn = _compiled.get(key)
        if fn is None:
            fn = _compiled[key] = cute.compile(
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
            )
    return fn


def _views(
    x: torch.Tensor, xsf: Optional[torch.Tensor], weight, weight_scale, y, x_bf16: bool
) -> tuple:
    k = x.shape[1]
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
) -> Optional[torch.Tensor]:
    global last_pushed, last_uncompiled
    last_pushed = False
    last_uncompiled = None
    import cuda.bindings.driver as cuda_driver

    m, k = x.shape
    n = weight.shape[0]
    if weight.shape[1] != k:
        return None
    resolved = _resolve(m, n, k, x_bf16, trigger_early, cfg, use_pdl, ar)
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
    args = _views(x, xsf, weight, weight_scale, y, x_bf16)
    extra = _extra_args(x.device)
    stream = cuda_driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    if fn is None:
        fn = _compile(key, args, m, runtime_ar, stream, extra)
    ar_ws, ar_flags, ar_rank = runtime_ar
    fn(*args, m, ar_ws, ar_flags, ar_rank, stream, *extra)
    last_pushed = key[11] > 0
    return y


def precompile(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    m: int,
    x_bf16: bool,
    use_pdl: Optional[bool] = None,
    ar: Optional[Tuple[int, int, int]] = None,
) -> Optional[tuple]:
    """Compile, without launching, the kernel an M-row call against ``weight`` takes; returns its key, or None when
    such a call is declined. ``ar`` = (world, bucket tokens, mode) of a pushing call (its addresses and rank are
    runtime arguments, so none is needed here). Runs outside capture, even while compiles are frozen."""
    import cuda.bindings.driver as cuda_driver

    n, k = weight.shape
    push = None if ar is None else (0, 0, 0, *ar)
    resolved = _resolve(m, n, k, x_bf16, True, None, use_pdl, push)
    if resolved is None:
        return None
    key, runtime_ar = resolved
    if key in _compiled:
        return key
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("m3_mx_gemv.precompile allocates and compiles: call it outside capture")
    dev = weight.device
    x = torch.zeros(m, k, dtype=torch.bfloat16 if x_bf16 else torch.float8_e4m3fn, device=dev)
    xsf = None if x_bf16 else torch.full((128 * (k // 32),), 127, dtype=torch.uint8, device=dev)
    y = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
    args = _views(x, xsf, weight, weight_scale, y, x_bf16)
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
