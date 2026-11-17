# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GEMV resolution, compilation and launch shared by the SM107 thin targets.

Feature bindings preserve each target's original compile-key arity and exact
native argument tail. Tables, compilation locks/caches and diagnostic state stay
in the target. Producer fields are read only for a producer-capable target.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Callable, Optional, Protocol, Tuple

import torch


class GemvTarget(Protocol):
    get_config: Callable[..., Optional[tuple]]
    _declines: Callable[..., bool]
    _CONTIG: bool
    _NACC: int
    _A_TMEM: bool
    _ST_ASYNC: bool
    _mma_n: Callable[[int], int]
    _smem_bytes: Callable[..., int]
    _SMEM_LIMIT: int
    get_env_enable_pdl: Callable[[], bool]
    PROD_N: int
    PROD_OFF: tuple
    _lock: AbstractContextManager[bool]
    _compiled: dict[tuple, Any]
    _frozen: list[bool]
    last_pushed: bool
    last_uncompiled: Optional[tuple]
    _resolve: Callable[..., Optional[tuple]]
    _views: Callable[..., tuple]
    _extra_args: Callable[..., tuple]
    _compile: Callable[..., Any]


def _resolve(
    state: GemvTarget,
    /,
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
    *,
    has_swiglu: bool,
    has_producer: bool,
) -> Optional[tuple]:
    """The compile key of an [M, K] x [N, K]^T call and its runtime push words, or None when the call is declined.

    ``ar`` is the push tuple of ``_run``; only its (world, bucket tokens, mode) enter the key, the addresses and the
    rank are runtime arguments."""
    cfg = cfg or state.get_config(n, k, m)
    if cfg is None or state._declines(n, k, m, x_bf16, cfg):
        return None
    split, ring = (cfg[0], cfg[1])
    contig = bool(cfg[2]) if len(cfg) > 2 else state._CONTIG
    nacc = int(cfg[3]) if len(cfg) > 3 else state._NACC
    a_tmem = bool(cfg[4]) if len(cfg) > 4 else state._A_TMEM
    st_async = bool(cfg[5]) if len(cfg) > 5 else state._ST_ASYNC
    if has_producer and (
        prod
        and (
            not (
                n == state.PROD_N
                and split == 8
                and (m <= split)
                and st_async
                and (not x_bf16)
                and (not x_swiglu)
                and (ar is None)
            )
        )
    ):
        return None
    if nacc != 1:
        return None
    ring = min(ring, k // 128 // split)
    tiles = k // 128 // split
    mma_n = state._mma_n(m)
    while ring > 2 and state._smem_bytes(ring, tiles, split, mma_n) > state._SMEM_LIMIT:
        ring -= 1
    if a_tmem:
        from tensorrt_llm._torch.modeling_v2.models.minimax_m3.kernels.gemm.m3_mx_gemv_kernel import (
            tmem_cols,
        )

        if tmem_cols(ring, nacc, True, tiles, mma_n) > 512:
            a_tmem = False
    ar_ws, ar_flags, ar_rank, ar_mode, ar_world, ar_maxt = (0, 0, 0, 0, 1, 8)
    if ar is not None and split <= 4 and (n % 2 == 0):
        ar_ws, ar_flags, ar_rank, ar_world, ar_maxt, ar_mode = ar
        if ar_mode == 2:
            trigger_early = False
    if use_pdl is None:
        use_pdl = state.get_env_enable_pdl()
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
        *((x_swiglu,) if has_swiglu else ()),
        *((prod,) if has_producer else ()),
    )
    return (key, (ar_ws, ar_flags, ar_rank))


def _compile(
    state: GemvTarget,
    /,
    key: tuple,
    args: tuple,
    m: int,
    runtime_ar: tuple,
    stream,
    extra: tuple,
    prod_args: tuple = (),
    *,
    has_swiglu: bool,
    has_producer: bool,
):
    """cute.compile the kernel of ``key`` (see ``_resolve``) on these argument views; cached."""
    import cutlass.cute as cute

    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.kernels.gemm import (
        m3_mx_gemv_kernel as kernel,
    )

    if has_producer:
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
        ) = key
    elif has_swiglu:
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
        ) = key
    else:
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
    with state._lock:
        fn = state._compiled.get(key)
        if fn is None:
            fn = state._compiled[key] = cute.compile(
                kernel.m3_mx_gemv_producer if has_producer else kernel.m3_mx_gemv,
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
                *((x_swiglu,) if has_swiglu else ()),
                *(prod_args if has_producer else ()),
                *((prod,) if has_producer else ()),
            )
    return fn


def _run(
    state: GemvTarget,
    /,
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
    *,
    has_swiglu: bool,
    has_producer: bool,
) -> Optional[torch.Tensor]:
    state.last_pushed = False
    state.last_uncompiled = None
    import cuda.bindings.driver as cuda_driver

    m, k = x.shape
    if has_swiglu and x_swiglu:
        k //= 2
    n = weight.shape[0]
    if weight.shape[1] != k:
        return None
    if has_producer:
        prod = prod_args is not None
    resolved = state._resolve(
        m,
        n,
        k,
        x_bf16,
        trigger_early,
        cfg,
        use_pdl,
        ar,
        *((x_swiglu,) if has_swiglu else ()),
        *((prod,) if has_producer else ()),
    )
    if resolved is None:
        return None
    key, runtime_ar = resolved
    if has_producer and (not prod):
        prod_args = state.PROD_OFF
    fn = state._compiled.get(key)
    if fn is None:
        if state._frozen[0]:
            state.last_uncompiled = key
            return None
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "m3_mx_gemv must run once per shape outside CUDA-graph capture first"
            )
    y = torch.empty(0 if has_producer and prod else m, n, dtype=torch.bfloat16, device=x.device)
    args = state._views(
        x, xsf, weight, weight_scale, y, x_bf16, *((x_swiglu,) if has_swiglu else ())
    )
    extra = state._extra_args(x.device)
    stream = cuda_driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    if fn is None:
        fn = state._compile(
            key, args, m, runtime_ar, stream, extra, *((prod_args,) if has_producer else ())
        )
    ar_ws, ar_flags, ar_rank = runtime_ar
    fn(*args, m, ar_ws, ar_flags, ar_rank, stream, *extra, *(prod_args if has_producer else ()))
    state.last_pushed = key[11] > 0
    return y


def precompile(
    state: GemvTarget,
    /,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    m: int,
    x_bf16: bool,
    use_pdl: Optional[bool] = None,
    ar: Optional[Tuple[int, int, int]] = None,
    x_swiglu: bool = False,
    prod: bool = False,
    *,
    has_swiglu: bool,
    has_producer: bool,
) -> Optional[tuple]:
    """Compile, without launching, the kernel an M-row call against ``weight`` takes; returns its key, or None when
    such a call is declined. ``ar`` = (world, bucket tokens, mode) of a pushing call (its addresses and rank are
    runtime arguments, so none is needed here). Runs outside capture, even while compiles are frozen."""
    import cuda.bindings.driver as cuda_driver

    n, k = weight.shape
    push = None if ar is None else (0, 0, 0, *ar)
    resolved = state._resolve(
        m,
        n,
        k,
        x_bf16,
        True,
        None,
        use_pdl,
        push,
        *((x_swiglu,) if has_swiglu else ()),
        *((prod,) if has_producer else ()),
    )
    if resolved is None:
        return None
    key, runtime_ar = resolved
    if key in state._compiled:
        return key
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("m3_mx_gemv.precompile allocates and compiles: call it outside capture")
    dev = weight.device
    x = torch.zeros(
        m,
        2 * k if has_swiglu and x_swiglu else k,
        dtype=torch.bfloat16 if x_bf16 else torch.float8_e4m3fn,
        device=dev,
    )
    xsf = None if x_bf16 else torch.full((128 * (k // 32),), 127, dtype=torch.uint8, device=dev)
    y = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
    args = state._views(
        x, xsf, weight, weight_scale, y, x_bf16, *((x_swiglu,) if has_swiglu else ())
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    state._compile(
        key,
        args,
        m,
        runtime_ar,
        stream,
        state._extra_args(dev),
        *((state.PROD_OFF,) if has_producer else ()),
    )
    return key
