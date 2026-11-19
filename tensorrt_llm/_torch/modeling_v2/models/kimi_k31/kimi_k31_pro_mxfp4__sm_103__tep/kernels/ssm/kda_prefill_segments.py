# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Segment-parallel KDA prefill of one long sequence from the engine's BT16 decomposition kernels.

The engine's decomposition runs a chunk-parallel factor prep (kernel 1) and a serial DV2 chain (kernel 2: two CTAs
per head walk all 16-token chunks). With few heads per rank (12 at TEP8) the chain occupies 24 SMs. Per chunk the
chain is affine in the V-first state rows: ``X_next = X @ A_c + B_c``. This module cuts the sequence into ``P``
chunk-aligned segments and runs the chain as pseudo-sequences over one shared factor prep:

  1. segment 0 from the true initial state, and segments 1 .. P-2 from a zero state (giving ``N_p``) in one chain
     launch; on a side stream, segments 1 .. P-2 from an identity state with zero values (giving ``M_p``);
  2. the true segment starts ``X_{p+1} = X_p @ M_p + N_p`` (batched over heads, fp32);
  3. segments 1 .. P-1 again from their true starts, writing their outputs and the final state.

The chain kernels, their tensor maps and the workspace layout are the engine's, compiled here as separate hosts.
Not bitwise to the sequential chain: the identity-state pass carries its own bf16 rounding of the running product.
"""

from __future__ import annotations

import functools

import torch

#: Segments per sequence: phase 1 runs (1 + 2 (P - 2)) * 2 CTAs per head, 120 at 12 heads.
DEFAULT_SEGMENTS = 4
#: Tokens per chunk of the BT16 kernels; segment cuts are chunk-aligned.
CHUNK = 16


@functools.lru_cache(maxsize=None)
def _compiled(lower_bound: float):
    """(prep, chain) hosts of the engine BT16 decomposition: BF16 activations, fp32 V-first state rows, slot ids."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream

    import tensorrt_llm._torch.attention.kernels.blackwell.kimi_k3_kda.kda_chunked_bt16 as bt16

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm the segment-parallel KDA prefill before CUDA graph capture")
    fakes, _, _, _ = bt16._unified_fakes(
        cutlass.BFloat16, cutlass.Float32, True, True, False, True, False, cutlass.BFloat16
    )
    (
        fq,
        fk,
        fv,
        fgate,
        fa_log,
        fdt,
        fbeta,
        fcu,
        _order,
        _,
        fidx,
        fcuc,
        _,
        fkd,
        fqd,
        fw,
        fqk,
        fdiag,
    ) = fakes[:18]
    fstate, fout, ffinal = fakes[18:21]
    prep = cute.compile(
        bt16.host_prep,
        fq,
        fk,
        fgate,
        fa_log,
        fdt,
        fbeta,
        fcu,
        fcuc,
        fkd,
        fqd,
        fw,
        fqk,
        fdiag,
        make_fake_stream(),
        0,
        0,
        0,
        0,
        True,
        lower_bound * bt16.LOG2_E,
        bt16.THREADS_PER_CTA,
        cutlass.BFloat16,
        options="--enable-tvm-ffi",
    )
    chain = cute.compile(
        bt16.host_chain_dv2,
        fv,
        fcu,
        fcuc,
        fkd,
        fqd,
        fw,
        fqk,
        fdiag,
        fidx,
        fstate,
        fout,
        ffinal,
        make_fake_stream(),
        0,
        0,
        bt16.DEFAULT_SCALE,
        None,
        None,
        None,
        0,
        bt16.THREADS_PER_CTA,
        options="--enable-tvm-ffi",
    )
    return prep, chain, bt16


@functools.lru_cache(maxsize=64)
def _plan(total: int, segments: int, device: str):
    """Chunk-aligned cuts and the int64 token / int32 chunk offsets of the three chain launches."""
    chunks = -(-total // CHUNK)
    cuts = [min(total, (chunks * p // segments) * CHUNK) for p in range(segments)] + [total]
    i64 = lambda xs: torch.tensor(xs, dtype=torch.int64, device=device)  # noqa: E731
    i32 = lambda xs: torch.tensor(xs, dtype=torch.int32, device=device)  # noqa: E731
    as_chunks = lambda xs: i32([-(-x // CHUNK) for x in xs])  # noqa: E731
    first, mid, rest = cuts[:segments], cuts[1:segments], cuts[1:]
    return dict(
        cu_all=i64([0, total]),
        cc_all=i32([0, chunks]),
        cu_first=i64(first),
        cc_first=as_chunks(first),
        cu_mid=i64(mid),
        cc_mid=as_chunks(mid),
        cu_rest=i64(rest),
        cc_rest=as_chunks(rest),
        slots=i32(list(range(segments))),
        chunks=chunks,
    )


#: Zero value tensors of the identity-map pass, one per (row shape, dtype, device), grown to the longest
#: sequence seen and sliced to each call's length, so their memory is bounded by the longest prefill chunk;
#: read only.
_ZEROS: dict = {}


def _zeros_like(v: torch.Tensor) -> torch.Tensor:
    """A read-only zero tensor of ``v``'s shape ``[1, T, ...]``, a contiguous slice of a cached buffer."""
    if v.shape[0] != 1:
        return torch.zeros_like(v)
    tokens = v.shape[1]
    key = (tuple(v.shape[2:]), v.dtype, str(v.device))
    zeros = _ZEROS.get(key)
    if zeros is None or zeros.shape[1] < tokens:
        zeros = _ZEROS[key] = torch.zeros_like(v)
    return zeros[:, :tokens]


def _state_rows(n: int, heads: int, device) -> torch.Tensor:
    """fp32 V-first rows [n, heads, 128, 128]; the chain takes slot strides that are multiples of 8 elements."""
    storage = torch.empty(n, heads * 128 * 128 + 16, dtype=torch.float32, device=device)
    return storage[:, : heads * 128 * 128].view(n, heads, 128, 128)


def kda_prefill_segments(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    slot: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    lower_bound: float,
    segments: int = DEFAULT_SEGMENTS,
) -> torch.Tensor:
    """One sequence of ``T`` packed tokens ([1, T, H, 128] bf16 q / k / v / g, [1, T, H] bf16 beta logits) from
    the fp32 V-first pool row ``slot`` (int32 [1] on the device); writes the final state back to that row and
    returns the bf16 output."""
    import cuda.bindings.driver as cuda_driver

    prep, chain, bt16 = _compiled(float(lower_bound))
    total, heads = q.shape[1], q.shape[2]
    device = q.device
    plan = _plan(total, segments, str(device))
    workspace = torch.empty(
        bt16._decomp_ws_bytes(heads, plan["chunks"]), dtype=torch.uint8, device=device
    )
    ws = bt16._partition_workspace(workspace, heads, plan["chunks"])
    num_ctas, cpc = bt16._k1_grid(plan["chunks"], heads)
    beta = beta.contiguous()
    dt = dt_bias.reshape(heads, 128)
    out = torch.empty_like(v)
    main = torch.cuda.current_stream(device)
    ws_args = (ws["kd"], ws["qd"], ws["w"], ws["qk"], ws["diag"])
    prep(
        q,
        k,
        g,
        A_log,
        dt,
        beta,
        plan["cu_all"],
        plan["cc_all"],
        *ws_args,
        cuda_driver.CUstream(main.cuda_stream),
        num_ctas,
        cpc,
        0,
        heads,
    )

    n_mid = segments - 2
    first = _state_rows(segments - 1, heads, device)
    slot = slot.to(torch.int64)
    first[0].copy_(state_pool.index_select(0, slot)[0])
    first[1:].zero_()
    if n_mid:
        mids = _state_rows(n_mid, heads, device)
        mids.copy_(torch.eye(128, device=device).expand(n_mid, heads, 128, 128))
        scratch = torch.empty_like(v)
        zeros = _zeros_like(v)
        side = torch.cuda.Stream(device)
        side.wait_stream(main)
        with torch.cuda.stream(side):
            chain(
                zeros,
                plan["cu_mid"],
                plan["cc_mid"],
                *ws_args,
                plan["slots"][:n_mid],
                mids,
                scratch,
                mids,
                cuda_driver.CUstream(side.cuda_stream),
                0,
                heads,
                scale,
                None,
                None,
                None,
                0,
            )
        for t in (mids, scratch, workspace, zeros):
            t.record_stream(side)
    chain(
        v,
        plan["cu_first"],
        plan["cc_first"],
        *ws_args,
        plan["slots"][: segments - 1],
        first,
        out,
        first,
        cuda_driver.CUstream(main.cuda_stream),
        0,
        heads,
        scale,
        None,
        None,
        None,
        0,
    )
    rest = _state_rows(segments - 1, heads, device)
    rest[0].copy_(first[0])
    if n_mid:
        main.wait_stream(side)
        for p in range(n_mid):
            torch.baddbmm(first[p + 1], rest[p], mids[p], out=rest[p + 1])
    chain(
        v,
        plan["cu_rest"],
        plan["cc_rest"],
        *ws_args,
        plan["slots"][: segments - 1],
        rest,
        out,
        rest,
        cuda_driver.CUstream(main.cuda_stream),
        0,
        heads,
        scale,
        None,
        None,
        None,
        0,
    )
    state_pool.index_copy_(0, slot, rest[segments - 2 :])
    return out
