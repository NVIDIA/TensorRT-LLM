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

import sys
import threading
from typing import Dict, Optional, Tuple, cast

import torch

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl as get_env_enable_pdl
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.gemm import mxfp8_core as _core
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.gemm import mxfp8_host as _host

MAX_TOKENS = 48  # capacity: fixed: the decode kernels' row bound (decode steps)
MMA_WIDTHS = (
    8,
    16,
    32,
    48,
)  # token columns of the compiled MMA; a call uses the smallest that holds M
_SMEM_LIMIT = 220 * 1024  # capacity: fixed: the device's shared-memory limit


def _mma_n(m: int) -> int:
    return _host.gemv_mma_n(m, mma_widths=MMA_WIDTHS)


def _smem_bytes(ring: int, tiles: int, split: int, mma_n: int) -> int:
    return _host.gemv_smem_bytes(ring, tiles, split, mma_n)


_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}

# (N, K) -> (split, ring[, contig, nacc, a_tmem, st_async]) of the TP2 per-rank projections. The optional fields default
# to _CONTIG, _NACC, _A_TMEM, _ST_ASYNC. SM107 graph us at M 1 / 4 / 8 over weights rotated beyond L2, against the
# stock MXFP8 quantize + FlashInfer CuTe DSL GEMM: QKV 4.11 / 4.12 / 4.10 vs 8.6-8.8, o_proj 3.90 / 3.96 / 3.70 vs
# 6.9-7.2, shared gate_up 3.11 / 3.09 / 3.10 vs 8.1-8.4, shared down 2.22 vs 4.7-4.9, dense gate_up 8.15 / 8.56 / 8.55
# vs 10.2-10.5, dense down 5.89 / 5.72 / 5.46 vs 8.7-9.0.
_CONFIGS: Dict[Tuple[int, int], tuple] = {
    # The sm_107 tp8 target's per-rank projections (8 ranks over two trays), (split, ring) from the SM107 sweep 778596
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

# Defaults of the optional config fields: contiguous k-tile ranges per split rank, one TMEM accumulator, the A
# operand staged in TMEM (when its columns fit), st.async split-K partial exchange.
_CONTIG = True
_NACC = 1
_A_TMEM = True
_ST_ASYNC = True
_ensured = set()


_DUMMY: Dict[torch.device, torch.Tensor] = {}


def _dummy_timers(device) -> torch.Tensor:
    "Placeholder for the kernel's timer buffer, which it only writes when compiled with timers on."
    return _host.gemv_dummy_timers(device, dummy=_DUMMY)


def _arg(t: torch.Tensor, align: int = 16):
    return _host.gemv_arg(t, align)


# (rule 1b, deterministic) The decode projections at up to STRIDED_MAX_ROWS rows split K over strided k-tiles per
# split rank (contig 0: rank r owns k-tiles r, r + split, ...) instead of contiguous ranges: another fixed fp32 order.
# SM107 TP8 4 rows (783579), us/call contiguous -> strided: sparse QKV 2.405 -> 2.371, dense QKV 2.323 -> 2.291,
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
    return _host.gemv_get_config(
        n,
        k,
        m,
        configs_by_m=_CONFIGS_BY_M,
        configs_wide=_CONFIGS_WIDE,
        configs=_CONFIGS,
        strided=STRIDED_K_TILES,
        strided_max_rows=STRIDED_MAX_ROWS,
        strided_shapes=_STRIDED,
    )


def _declines(n: int, k: int, m: int, x_bf16: bool, cfg: tuple) -> bool:
    "Whether a (split, ring, ...) config declines an [M, K] x [N, K]^T call on shape grounds."
    return _host.gemv_declines(
        n,
        k,
        m,
        x_bf16,
        cfg,
        max_tokens=MAX_TOKENS,
        smem_limit=_SMEM_LIMIT,
        select_mma_n=_mma_n,
        smem_bytes=_smem_bytes,
    )


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
) -> Optional[tuple]:
    """The compile key of an [M, K] x [N, K]^T call and its runtime push words, or None when the call is declined.

    ``ar`` is the push tuple of ``_run``; only its (world, bucket tokens, mode) enter the key, the addresses and the
    rank are runtime arguments."""
    return _core._resolve(
        _CORE_TARGET,
        m,
        n,
        k,
        x_bf16,
        trigger_early,
        cfg,
        use_pdl,
        ar,
        x_swiglu,
        prod,
        has_swiglu=True,
        has_producer=True,
    )


def _compile(
    key: tuple, args: tuple, m: int, runtime_ar: tuple, stream, extra: tuple, prod_args: tuple
):
    """cute.compile the kernel of ``key`` (see ``_resolve``) on these argument views; cached."""
    return _core._compile(
        _CORE_TARGET,
        key,
        args,
        m,
        runtime_ar,
        stream,
        extra,
        prod_args,
        has_swiglu=True,
        has_producer=True,
    )


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
    return _host.gemv_views(
        x,
        xsf,
        weight,
        weight_scale,
        y,
        x_bf16,
        x_swiglu,
        placeholder_scales=_placeholder_scales,
        placeholder_words=_placeholder_words,
        arg=_arg,
        dummy_timers=_dummy_timers,
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
) -> Optional[torch.Tensor]:
    return _core._run(
        _CORE_TARGET,
        x,
        xsf,
        weight,
        weight_scale,
        x_bf16,
        trigger_early,
        cfg,
        use_pdl,
        ar,
        x_swiglu,
        prod_args,
        has_swiglu=True,
        has_producer=True,
    )


def precompile(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    m: int,
    x_bf16: bool,
    use_pdl: Optional[bool] = None,
    ar: Optional[Tuple[int, int, int]] = None,
    x_swiglu: bool = False,
    prod: bool = False,
) -> Optional[tuple]:
    """Compile, without launching, the kernel an M-row call against ``weight`` takes; returns its key, or None when
    such a call is declined. ``ar`` = (world, bucket tokens, mode) of a pushing call (its addresses and rank are
    runtime arguments, so none is needed here). Runs outside capture, even while compiles are frozen."""
    return _core.precompile(
        _CORE_TARGET,
        weight,
        weight_scale,
        m,
        x_bf16,
        use_pdl,
        ar,
        x_swiglu,
        prod,
        has_swiglu=True,
        has_producer=True,
    )


def freeze(frozen: bool = True) -> None:
    """Close (or reopen) compiles: while frozen, a call whose kernel is not compiled yet declines (returns None and
    sets ``last_uncompiled``) instead of compiling. ``precompile`` and ``ensure_compiled`` still compile."""
    _frozen[0] = bool(frozen)


def frozen() -> bool:
    return _frozen[0]


def is_compiled(key: tuple) -> bool:
    return key in _compiled


def _extra_args(device) -> tuple:
    "Placeholder arguments of the kernel's draft-input prologue and RoPE / KV epilogue (compiled off)."
    return _host.gemv_extra_args(device, placeholder_words=_placeholder_words, arg=_arg)


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
    return _host.gemv_placeholder_scales(k, device, placeholders=_PLACEHOLDERS)


def _placeholder_words(device) -> torch.Tensor:
    return _host.gemv_placeholder_words(device, placeholders=_PLACEHOLDERS)


def ensure_compiled(weight: torch.Tensor, weight_scale: torch.Tensor, use_pdl: bool) -> None:
    "Compiles the kernel of this weight's shape outside capture (a one-token dummy call)."
    return _host.gemv_ensure_compiled(
        weight,
        weight_scale,
        use_pdl,
        get_config=get_config,
        ensured=_ensured,
        mma_widths=MMA_WIDTHS,
        mx_gemv=m3_mx_gemv,
        mx_gemv_bf16=m3_mx_gemv_bf16,
    )


_CORE_TARGET = cast(_core.GemvTarget, sys.modules[__name__])
