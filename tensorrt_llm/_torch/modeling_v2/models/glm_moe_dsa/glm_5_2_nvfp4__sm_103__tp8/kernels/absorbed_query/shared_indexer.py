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
"""shared_indexer_absorbed_query: one-kernel query region of the shared-indexer layers (absorbed_query package).

On a GLM-5.x DSA layer without its own indexer (the indexer reuses the
previous full layer's top-k), the absorbed_query path runs the native ``fused_a`` kv_a
GEMM, the native ``q_a_layernorm`` (FlashInfer RMSNorm), the absorbed_query kernel
in its native-q form and then, inside ``MLA.forward_absorption_generation``,
the native absorption BMM. With package ``shared_indexer_absorbed_query`` selected (on top of
``absorbed_query``) ``shared_dsa_proj`` replaces everything after the kv_a GEMM by one kernel
that returns the **preabsorbed** query_front packet (``q``, ``latent`` =
``[rmsnorm(compressed_kv) | k_pe]`` and ``fused_q[..., :kv_lora]``):

* R <= 8 rows (``SharedQueryKernelConfig.cute``, default): the CuTe-DSL cluster
  kernel of ``shared_indexer_cute`` (one 8-CTA cluster per head, the
  normalized rows recomputed per CTA, the no-RoPE ``q`` tiles exchanged
  through distributed shared memory; see that module).
* otherwise (R <= 16, or ``cute=False``): ``shared_query_kernel`` below.

``shared_query_kernel`` (Triton):

* norm programs reproduce the native ``q_a_layernorm`` (FlashInfer CuTe-DSL
  ``RMSNormKernel``) bit for bit -- same lane-to-column map, same per-lane
  fold order, same butterfly, ``rsqrt``, ``bf16((x * rstd) * w)`` -- write
  the normalized rows to a scratch buffer and release them to the producers;
  they also produce ``latent`` with the same reproduction of the native
  ``kv_a_layernorm``;
* ``q_b`` producer programs (the absorbed_query tiling: ``BN`` output columns over the
  full ``K``) wait for the normalized rows, compute ``q`` with the same MMA
  sequence as absorbed_query and publish each head's no-RoPE tiles;
* absorption jobs (absorbed_query's shifted layout: producers of head ``h + 1`` absorb
  head ``h``, ``HEAD_DIM // BN`` extra programs absorb the last head) write
  ``fused_q[:, h, :kv_lora] = bf16(q_nope[:, h] @ W_UK[h]^T)``.

Both kernels are launched with programmatic dependent launch right after the
kv_a GEMM. Before ``griddepcontrol.wait`` a program only reads weights (the
Triton kernel asks L2 to prefetch its ``q_b`` tile and ``W_UK`` slice, the
CuTe kernel copies them into shared memory), so the weight stream overlaps
the kv_a GEMM; every activation, output and counter access comes after it.
With ``early_trigger`` (default; off: the implicit trigger at grid
completion) every program issues ``griddepcontrol.launch_dependents`` right
after its own wait, so the next kernel (its programmatic dependent) can be
scheduled while this one runs.

PDL contract with the dependent (whichever kernel follows: the native
``mla_rope_generation`` kernel, or another package's replacement such as
rope_kv_append's fused rope):

* What the trigger guarantees. The dependent is released only once every
  program of this grid has issued the trigger (or exited). At that point
  every program is resident and past its own wait. So (1) everything this
  kernel's wait covers has completed, as it has at an implicit trigger, and
  (2) a waiting dependent program cannot keep a program of this grid, or a
  counter it spins on, from being scheduled.
* What the dependent must do:

  - execute ``griddepcontrol.wait`` before reading any output of this kernel
    (``q``, ``latent`` / ``compressed_kv`` / ``k_pe``, ``fused_q``) or
    anything derived from them. This holds for every proxy, TMA included:
    the outputs are generic-proxy stores of a grid that has completed when
    that wait returns;
  - write nothing before its wait that this kernel reads or writes: ``kva``,
    the weights, the per-layer counters, the outputs, and any graph-pool
    memory that may back them;
  - read nothing before its wait that this kernel writes;
  - never spin on this kernel's progress before its wait.

These rules do not depend on ``early_trigger``: with the implicit trigger
the dependent is released when this grid's programs exit, and the same
rules apply. The served successors follow them. The native rope kernel
reads its inputs after its wait. rope_kv_append's fused rope reads ``fused_q`` and
``latent`` after its wait. Its per-step metadata reads happen after the
wait by default; in its opt-in prefetch mode they happen before the wait,
under rope_kv_append's own condition on the metadata writers. That condition is
unaffected by this trigger, because the trigger comes after this kernel's
own wait.

Synchronization of the Triton kernel, per layer: with ``epoch_sync``
(default) monotonic 64-bit counters ``epochs`` = [finished programs,
released norm programs, published no-RoPE tiles per head] that are never
reset; a program derives the launch index from the finished-program count
(the grid is the same in every launch) and waits for ``(launch + 1) *
expected`` releases. Without it, absorbed_query's int32 counters ``sync`` that the last
consumer resets to zero. Every spin-wait targets counters released by lower
program ids (norm programs first, then producers, then the extra
absorption programs), so the progress argument of ``kernels.py`` holds. The
CuTe kernel has no inter-cluster dependency.

Numerics: ``q``, the absorbed query and ``latent`` are bit-identical to the
native ``forward_dsa_proj`` + absorption BMM when the per-lane fold order
(``SharedQueryKernelConfig.norm_order``) matches the native RMSNorm kernel; any
other order is a different rounding and is rejected by the contract tests.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from ...dispatch import log_capability
from . import kernels as aq_kernels
from .query_proj import (
    _HEAD_DIM,
    _KV_LORA,
    _LOCAL_HEADS,
    _NOPE,
    _Q_LORA,
    _ROPE,
    AbsorbedQueryPacket,
    absorb_layout,
    call_eligibility,
    check_package_selectors,
    module_eligibility,
)

# Rows of one shared_indexer_absorbed_query launch (the absorbed_query support domain).
_SHARED_QUERY_MAX_ROWS = 16
# Per-lane fold orders of the RMSNorm reproduction (see _fi_lane_sumsq).
NORM_ORDERS = (0, 1, 2, 3, 4, 5, 6)


@dataclass(frozen=True)
class SharedQueryKernelConfig:
    # The CuTe-DSL cluster kernel (``shared_indexer_cute``) for R <= 8 rows;
    # the Triton kernel below otherwise.
    cute: bool = True
    # CuTe kernel: copy the weights into shared memory before the
    # grid-dependency wait (overlapping the kv_a GEMM).
    cute_weights_before_wait: bool = True
    # CuTe kernel: st.async pushes + per-CTA mbarrier (True) or plain
    # cluster stores + one cluster barrier (False).
    cute_push_async: bool = True
    block_m: int = 16
    # Rows per norm program (the native FlashInfer kernel uses 4 per CTA).
    norm_rows: int = 4
    bn: int = 32
    bk: int = 256
    num_warps: int = 4
    num_stages: int = 4
    # Per-lane fold order of the native RMSNorm reproduction.
    norm_order: int = 0
    # Order 0 with 16-byte loads (one vector per lane and 256-column block).
    norm_vectorized: bool = True
    # q_b tile and absorption computed transposed (weights as the MMA M
    # operand, the padded token rows as N); needs bn, bc in {64, 128}.
    gemm_transposed: bool = False
    absorb_transposed: bool = False
    # L2 prefetch of the q_b tile and W_UK slice before griddepcontrol.wait.
    prefetch_l2: bool = True
    # griddepcontrol.launch_dependents right after each program's wait
    # (see the PDL contract in the module docstring), so the successor (the
    # native rope kernel or rope_kv_append's fused rope kernel) can be scheduled while
    # this kernel runs; the successor still waits for this grid before it
    # reads any output.
    early_trigger: bool = True
    # Load each producer's W_UK slice into registers before the wait.
    preload_wuk: bool = True
    # Counter poll: 0 atom.add 0 (absorbed_query), 1 ld.acquire.gpu, 2 nanosleep + ld.
    poll_mode: int = 1
    # Monotonic 64-bit counters with per-launch targets instead of absorbed_query's
    # last-consumer resets (no program waits for an atomic's return value).
    epoch_sync: bool = True
    # Tests only: skew programs by up to this many ns (kernels._stress_delay).
    debug_delay_ns: int = 0
    # Tests only: keep the normalized q_a rows (self.debug["qn"]).
    debug_keep_qn: bool = False
    # Tests only: per-program %globaltimer stamps (self.debug["ts"]).
    debug_timestamps: bool = False


# --------------------------------------------------------------------------
# Reproduction of the native FlashInfer CuTe-DSL RMSNorm (H in {512, 2048}):
# one warp per row, lane t owns columns 256 * b + 8 * t + v (b < H / 256,
# v < 8), per-lane fold of the squares, butterfly over the 32 lanes,
# rstd = rsqrt(sum / H + eps), y = bf16((x * rstd) * (w + 0)).
# --------------------------------------------------------------------------


@triton.jit
def _tree8(a0, a1, a2, a3, a4, a5, a6, a7):
    return ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))


@triton.jit
def _ld_sq(base, rmask, off: tl.constexpr):
    x = tl.load(base + off, mask=rmask[:, None], other=0.0).to(tl.float32)
    return x * x


@triton.jit
def _tree8_at(base, rmask, b: tl.constexpr):
    o: tl.constexpr = 256 * b
    return _tree8(
        _ld_sq(base, rmask, o + 0),
        _ld_sq(base, rmask, o + 1),
        _ld_sq(base, rmask, o + 2),
        _ld_sq(base, rmask, o + 3),
        _ld_sq(base, rmask, o + 4),
        _ld_sq(base, rmask, o + 5),
        _ld_sq(base, rmask, o + 6),
        _ld_sq(base, rmask, o + 7),
    )


@triton.jit
def _seq8_at(base, rmask, b: tl.constexpr, ROWS: tl.constexpr):
    o: tl.constexpr = 256 * b
    s = tl.zeros([ROWS, 32], dtype=tl.float32)
    for v in tl.static_range(8):
        s = s + _ld_sq(base, rmask, o + v)
    return s


@triton.jit
def _fi_lane_sumsq(base, rmask, NB: tl.constexpr, ORDER: tl.constexpr, ROWS: tl.constexpr):
    """``[ROWS, 32]`` per-lane sums of squares, NB in {2, 8}.

    ``base`` points at column ``8 * lane`` of each row. Orders: 0 sequential
    (v fastest), 1 sequential (b fastest), 2 pairwise tree, 3 per-v
    accumulators over b then sequential over v, 4 the same then a tree over
    v, 5 per-b tree over v then sequential over b, 6 per-b sequential over v
    then a tree over b.
    """
    s = tl.zeros([ROWS, 32], dtype=tl.float32)
    if ORDER == 0:
        for b in tl.static_range(NB):
            for v in tl.static_range(8):
                s = s + _ld_sq(base, rmask, 256 * b + v)
    elif ORDER == 1:
        for v in tl.static_range(8):
            for b in tl.static_range(NB):
                s = s + _ld_sq(base, rmask, 256 * b + v)
    elif ORDER == 2:
        if NB == 2:
            s = _tree8_at(base, rmask, 0) + _tree8_at(base, rmask, 1)
        else:
            s = _tree8(
                _tree8_at(base, rmask, 0),
                _tree8_at(base, rmask, 1),
                _tree8_at(base, rmask, 2),
                _tree8_at(base, rmask, 3),
                _tree8_at(base, rmask, 4),
                _tree8_at(base, rmask, 5),
                _tree8_at(base, rmask, 6),
                _tree8_at(base, rmask, 7),
            )
    elif ORDER == 3 or ORDER == 4:
        a0 = tl.zeros([ROWS, 32], dtype=tl.float32)
        a1 = tl.zeros([ROWS, 32], dtype=tl.float32)
        a2 = tl.zeros([ROWS, 32], dtype=tl.float32)
        a3 = tl.zeros([ROWS, 32], dtype=tl.float32)
        a4 = tl.zeros([ROWS, 32], dtype=tl.float32)
        a5 = tl.zeros([ROWS, 32], dtype=tl.float32)
        a6 = tl.zeros([ROWS, 32], dtype=tl.float32)
        a7 = tl.zeros([ROWS, 32], dtype=tl.float32)
        for b in tl.static_range(NB):
            a0 = a0 + _ld_sq(base, rmask, 256 * b + 0)
            a1 = a1 + _ld_sq(base, rmask, 256 * b + 1)
            a2 = a2 + _ld_sq(base, rmask, 256 * b + 2)
            a3 = a3 + _ld_sq(base, rmask, 256 * b + 3)
            a4 = a4 + _ld_sq(base, rmask, 256 * b + 4)
            a5 = a5 + _ld_sq(base, rmask, 256 * b + 5)
            a6 = a6 + _ld_sq(base, rmask, 256 * b + 6)
            a7 = a7 + _ld_sq(base, rmask, 256 * b + 7)
        if ORDER == 3:
            s = ((((((a0 + a1) + a2) + a3) + a4) + a5) + a6) + a7
        else:
            s = _tree8(a0, a1, a2, a3, a4, a5, a6, a7)
    elif ORDER == 5:
        for b in tl.static_range(NB):
            s = s + _tree8_at(base, rmask, b)
    else:
        if NB == 2:
            s = _seq8_at(base, rmask, 0, ROWS) + _seq8_at(base, rmask, 1, ROWS)
        else:
            s = _tree8(
                _seq8_at(base, rmask, 0, ROWS),
                _seq8_at(base, rmask, 1, ROWS),
                _seq8_at(base, rmask, 2, ROWS),
                _seq8_at(base, rmask, 3, ROWS),
                _seq8_at(base, rmask, 4, ROWS),
                _seq8_at(base, rmask, 5, ROWS),
                _seq8_at(base, rmask, 6, ROWS),
                _seq8_at(base, rmask, 7, ROWS),
            )
    return s


@triton.jit
def _fi_butterfly32(s, ROWS: tl.constexpr):
    """Sum over the 32 lanes in shuffle-xor butterfly order (offsets 1, 2,
    4, 8, 16), i.e. a balanced pairwise tree in lane order."""
    s = tl.sum(tl.reshape(s, [ROWS, 16, 2]), axis=2)
    s = tl.sum(tl.reshape(s, [ROWS, 8, 2]), axis=2)
    s = tl.sum(tl.reshape(s, [ROWS, 4, 2]), axis=2)
    s = tl.sum(tl.reshape(s, [ROWS, 2, 2]), axis=2)
    return tl.sum(s, axis=1)


@triton.jit
def _fi_rmsnorm_rows(
    x_ptr,
    stride_x,
    w_ptr,
    y_ptr,
    stride_y,
    rows,
    rmask,
    eps,
    H: tl.constexpr,
    ROWS: tl.constexpr,
    ORDER: tl.constexpr,
):
    """``y[rows, :H] = rmsnorm(x[rows, :H]) * w`` as the native kernel."""
    lanes = tl.arange(0, 32)
    base = x_ptr + rows[:, None] * stride_x + 8 * lanes[None, :]
    s = _fi_lane_sumsq(base, rmask, H // 256, ORDER, ROWS)
    rstd = tl.rsqrt(_fi_butterfly32(s, ROWS) / H + eps)
    for c in tl.static_range(H // 256):
        cols = c * 256 + tl.arange(0, 256)
        x = tl.load(
            x_ptr + rows[:, None] * stride_x + cols[None, :], mask=rmask[:, None], other=0.0
        ).to(tl.float32)
        w = tl.load(w_ptr + cols).to(tl.float32) + 0.0
        y = (x * rstd[:, None]) * w[None, :]
        tl.store(
            y_ptr + rows[:, None] * stride_y + cols[None, :], y.to(tl.bfloat16), mask=rmask[:, None]
        )


@triton.jit
def _v8_leaves(x, ROWS: tl.constexpr):
    """The 8 columns ``v`` of ``x [ROWS, 32, 8]`` as ``[ROWS, 32]`` tensors,
    in ``v`` order (register-level reshape/split, no data movement)."""
    e, o = tl.split(tl.reshape(x, [ROWS, 32, 4, 2]))
    ee, eo = tl.split(tl.reshape(e, [ROWS, 32, 2, 2]))
    oe, oo = tl.split(tl.reshape(o, [ROWS, 32, 2, 2]))
    v0, v4 = tl.split(ee)
    v2, v6 = tl.split(eo)
    v1, v5 = tl.split(oe)
    v3, v7 = tl.split(oo)
    return v0, v1, v2, v3, v4, v5, v6, v7


@triton.jit
def _fi_rmsnorm_rows_vec(
    x_ptr,
    stride_x,
    w_ptr,
    y_ptr,
    stride_y,
    rows,
    rmask,
    eps,
    H: tl.constexpr,
    ROWS: tl.constexpr,
    ts_ptr=None,
    pid=0,
    TS: tl.constexpr = False,
):
    """Fold order 0 of ``_fi_rmsnorm_rows`` with 16-byte loads: each lane
    loads its 8 consecutive columns of a 256-column block at once and folds
    them in column order (blocks outer, columns inner)."""
    lanes = tl.arange(0, 32)
    vv = tl.arange(0, 8)
    offs = rows[:, None, None] * stride_x + 8 * lanes[None, :, None] + vv[None, None, :]
    s = tl.zeros([ROWS, 32], dtype=tl.float32)
    for b in tl.static_range(H // 256):
        x = tl.load(x_ptr + offs + 256 * b, mask=rmask[:, None, None], other=0.0).to(tl.float32)
        v0, v1, v2, v3, v4, v5, v6, v7 = _v8_leaves(x, ROWS)
        s = s + v0 * v0
        s = s + v1 * v1
        s = s + v2 * v2
        s = s + v3 * v3
        s = s + v4 * v4
        s = s + v5 * v5
        s = s + v6 * v6
        s = s + v7 * v7
    rstd = tl.rsqrt(_fi_butterfly32(s, ROWS) / H + eps)
    if TS:
        tl.debug_barrier()
        _ts(ts_ptr, pid, 3, TS)
    # One load of the whole rows and gain, then one store: separate loads
    # per block would each wait behind the previous block's stores.
    cols = tl.arange(0, H)
    x = tl.load(
        x_ptr + rows[:, None] * stride_x + cols[None, :], mask=rmask[:, None], other=0.0
    ).to(tl.float32)
    w = tl.load(w_ptr + cols).to(tl.float32) + 0.0
    y = (x * rstd[:, None]) * w[None, :]
    tl.store(
        y_ptr + rows[:, None] * stride_y + cols[None, :], y.to(tl.bfloat16), mask=rmask[:, None]
    )
    if TS:
        tl.debug_barrier()
        _ts(ts_ptr, pid, 4, TS)


@triton.jit
def _fi_rmsnorm_vec_kernel(
    x_ptr, stride_x, w_ptr, y_ptr, stride_y, R, eps, H: tl.constexpr, ROWS: tl.constexpr
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    _fi_rmsnorm_rows_vec(x_ptr, stride_x, w_ptr, y_ptr, stride_y, rows, rows < R, eps, H, ROWS)


@triton.jit
def _fi_rmsnorm_kernel(
    x_ptr,
    stride_x,
    w_ptr,
    y_ptr,
    stride_y,
    R,
    eps,
    H: tl.constexpr,
    ROWS: tl.constexpr,
    ORDER: tl.constexpr,
):
    """Standalone launch of the reproduction (tests and probes)."""
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    _fi_rmsnorm_rows(x_ptr, stride_x, w_ptr, y_ptr, stride_y, rows, rows < R, eps, H, ROWS, ORDER)


def fi_rmsnorm_reproduction(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    order: int,
    rows_per_program: int = 4,
    vectorized: bool = False,
) -> torch.Tensor:
    """Run the RMSNorm reproduction on ``x [R, H]`` (row stride free);
    ``vectorized`` selects the 16-byte-load implementation of order 0."""
    rows, h = x.shape
    assert h in (512, 2048) and x.stride(1) == 1
    y = torch.empty(rows, h, dtype=torch.bfloat16, device=x.device)
    grid = (triton.cdiv(rows, rows_per_program),)
    if vectorized:
        assert order == 0
        _fi_rmsnorm_vec_kernel[grid](
            x,
            x.stride(0),
            weight,
            y,
            y.stride(0),
            rows,
            eps,
            H=h,
            ROWS=rows_per_program,
            num_warps=4,
        )
    else:
        _fi_rmsnorm_kernel[grid](
            x,
            x.stride(0),
            weight,
            y,
            y.stride(0),
            rows,
            eps,
            H=h,
            ROWS=rows_per_program,
            ORDER=order,
            num_warps=4,
        )
    return y


# --------------------------------------------------------------------------
# The shared_indexer_absorbed_query kernel.
# --------------------------------------------------------------------------


@triton.jit
def _l2_prefetch_1k(base_ptr, N: tl.constexpr, n_valid):
    """Ask L2 to prefetch ``n_valid`` (<= N) consecutive 1 KiB chunks."""
    idx = tl.minimum(tl.arange(0, N), n_valid - 1)
    ptrs = base_ptr.to(tl.pointer_type(tl.int8)) + idx * 1024
    tl.inline_asm_elementwise(
        "cp.async.bulk.prefetch.L2.global [$1], 1024; mov.u32 $0, 0;",
        "=r,l",
        [ptrs],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _shared_query_gemm(
    t,
    x_ptr,
    stride_x,
    w_ptr,
    out_ptr,
    OUT_COLS: tl.constexpr,
    rows,
    rmask,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """``out[:, t*BN:(t+1)*BN] = x @ W[t*BN:(t+1)*BN]^T`` (absorbed_query's MMA sequence);
    ``x`` was written by another program of this grid, so it is read with
    ``.cg`` (L2)."""
    cols = tl.arange(0, BK)
    n_offs = t * BN + tl.arange(0, BN)
    acc = tl.zeros([BLOCK_M, BN], dtype=tl.float32)
    for k0 in tl.range(0, K, BK):
        x = tl.load(
            x_ptr + rows[:, None] * stride_x + (k0 + cols)[None, :],
            mask=rmask[:, None],
            other=0.0,
            cache_modifier=".cg",
        )
        w = tl.load(w_ptr + n_offs[None, :] * K + (k0 + cols)[:, None])
        acc = tl.dot(x, w, acc)
    tl.store(
        out_ptr + rows[:, None] * OUT_COLS + n_offs[None, :],
        acc.to(tl.bfloat16),
        mask=rmask[:, None],
    )


@triton.jit
def _shared_query_gemm_t(
    t,
    x_ptr,
    stride_x,
    w_ptr,
    out_ptr,
    OUT_COLS: tl.constexpr,
    rows,
    rmask,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """The same tile as ``_shared_query_gemm``, computed transposed:
    ``out^T [BN, BLOCK_M] = W[t*BN:(t+1)*BN] @ x^T``, so the MMA ``M`` is the
    weight tile (64 or 128 rows, eligible for tcgen05) and ``N`` the padded
    token rows. Same products, same ascending K order."""
    kk = tl.arange(0, BK)
    n_offs = t * BN + tl.arange(0, BN)
    acc = tl.zeros([BN, BLOCK_M], dtype=tl.float32)
    for k0 in tl.range(0, K, BK):
        w = tl.load(w_ptr + n_offs[:, None] * K + (k0 + kk)[None, :])
        x = tl.load(
            x_ptr + rows[None, :] * stride_x + (k0 + kk)[:, None],
            mask=rmask[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        acc = tl.dot(w, x, acc)
    tl.store(
        out_ptr + rows[None, :] * OUT_COLS + n_offs[:, None],
        acc.to(tl.bfloat16),
        mask=rmask[None, :],
    )


@triton.jit
def _ld_wuk_t(a, wuk_ptr, KV_LORA: tl.constexpr, NOPE: tl.constexpr, BC: tl.constexpr):
    """``W_UK[h, c0:c0+BC, :NOPE]`` of absorption job ``a`` as [BC, NOPE]
    (split 128 + NOPE-128 along NOPE)."""
    chunks_per_head: tl.constexpr = KV_LORA // BC
    h = a // chunks_per_head
    c0 = (a % chunks_per_head) * BC
    cc = tl.arange(0, BC)
    base = wuk_ptr + h * (KV_LORA * NOPE) + (c0 + cc)[:, None] * NOPE
    w_lo = tl.load(base + tl.arange(0, 128)[None, :])
    w_hi = tl.load(base + 128 + tl.arange(0, NOPE - 128)[None, :])
    return w_lo, w_hi


@triton.jit
def _poll_once(ptr, POLL: tl.constexpr, W64: tl.constexpr = False):
    """One acquire read of an int32 (``W64``: int64) counter. ``POLL`` 0:
    ``atom.add 0`` (absorbed_query's protocol); 1: ``ld.acquire.gpu``; 2: ``nanosleep``
    then ``ld.acquire.gpu``."""
    if POLL == 0:
        v = tl.atomic_add(ptr, 0, sem="acquire", scope="gpu")
    else:
        p = ptr + tl.zeros([1], dtype=tl.int32)
        if W64:
            if POLL == 1:
                r = tl.inline_asm_elementwise(
                    "ld.acquire.gpu.global.b64 $0, [$1];",
                    "=l,l",
                    [p],
                    dtype=tl.int64,
                    is_pure=False,
                    pack=1,
                )
            else:
                r = tl.inline_asm_elementwise(
                    "nanosleep.u32 32; ld.acquire.gpu.global.b64 $0, [$1];",
                    "=l,l",
                    [p],
                    dtype=tl.int64,
                    is_pure=False,
                    pack=1,
                )
        else:
            if POLL == 1:
                r = tl.inline_asm_elementwise(
                    "ld.acquire.gpu.global.b32 $0, [$1];",
                    "=r,l",
                    [p],
                    dtype=tl.int32,
                    is_pure=False,
                    pack=1,
                )
            else:
                r = tl.inline_asm_elementwise(
                    "nanosleep.u32 32; ld.acquire.gpu.global.b32 $0, [$1];",
                    "=r,l",
                    [p],
                    dtype=tl.int32,
                    is_pure=False,
                    pack=1,
                )
        v = tl.max(r, axis=0)
    return v


@triton.jit
def _wait_ge(ptr, target, POLL: tl.constexpr, W64: tl.constexpr = False):
    """Spin until ``*ptr >= target`` (acquire), then a CTA barrier so every
    thread's later reads are ordered after the acquire."""
    v = _poll_once(ptr, POLL, W64)
    while v < target:
        v = _poll_once(ptr, POLL, W64)
    tl.debug_barrier()


@triton.jit
def _ts(ts_ptr, pid, k: tl.constexpr, TS: tl.constexpr):
    """Tests only: record ``%globaltimer`` (ns) at point ``k`` of ``pid``."""
    if TS:
        tl.store(ts_ptr + pid * 8 + k, tl.extra.cuda.globaltimer())


@triton.jit
def _shared_query_absorb_job(
    a,
    w_lo,
    w_hi,
    q_ptr,
    fq_ptr,
    stride_fq_row,
    sync_ptr,
    ready_ptr,
    ready_target,
    rows,
    rmask,
    ts_ptr,
    pid,
    KV_LORA: tl.constexpr,
    ROPE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE: tl.constexpr,
    H: tl.constexpr,
    BN: tl.constexpr,
    BC: tl.constexpr,
    POLL: tl.constexpr,
    TS: tl.constexpr,
    ABS_T: tl.constexpr,
    EPOCH: tl.constexpr,
    DELAY_NS: tl.constexpr,
):
    """Absorption job ``a`` (head ``a // (KV_LORA // BC)``, ``BC`` latent
    columns) with its ``W_UK`` slice already loaded: absorbed_query's
    ``kernels._absorb_job`` (same products, same d order) with a selectable
    poll. Waits for ``ready_ptr[h] >= ready_target``; without ``EPOCH`` it
    then runs absorbed_query's done/reset protocol on ``sync_ptr``. ``ABS_T`` computes
    the transposed tile (``W_UK`` slice as the MMA ``M`` operand, from
    ``_ld_wuk_t``)."""
    chunks_per_head: tl.constexpr = KV_LORA // BC
    h = a // chunks_per_head
    c0 = (a % chunks_per_head) * BC
    cc = tl.arange(0, BC)
    d_lo = tl.arange(0, 128)
    d_hi = tl.arange(0, NOPE - 128)
    _wait_ge(ready_ptr + h, ready_target, POLL, EPOCH)
    _ts(ts_ptr, pid, 5, TS)
    if ABS_T:
        q_base = q_ptr + rows[None, :] * (H * HEAD_DIM) + h * HEAD_DIM
        q_lo = tl.load(q_base + d_lo[:, None], mask=rmask[None, :], other=0.0, cache_modifier=".cg")
        q_hi = tl.load(
            q_base + 128 + d_hi[:, None], mask=rmask[None, :], other=0.0, cache_modifier=".cg"
        )
        acc = tl.dot(w_lo, q_lo)
        acc = tl.dot(w_hi, q_hi, acc)
        tl.store(
            fq_ptr + rows[None, :] * stride_fq_row + h * (KV_LORA + ROPE) + (c0 + cc)[:, None],
            acc.to(tl.bfloat16),
            mask=rmask[None, :],
        )
    else:
        q_base = q_ptr + rows[:, None] * (H * HEAD_DIM) + h * HEAD_DIM
        q_lo = tl.load(q_base + d_lo[None, :], mask=rmask[:, None], other=0.0, cache_modifier=".cg")
        q_hi = tl.load(
            q_base + 128 + d_hi[None, :], mask=rmask[:, None], other=0.0, cache_modifier=".cg"
        )
        acc = tl.dot(q_lo, w_lo)
        acc = tl.dot(q_hi, w_hi, acc)
        tl.store(
            fq_ptr + rows[:, None] * stride_fq_row + h * (KV_LORA + ROPE) + (c0 + cc)[None, :],
            acc.to(tl.bfloat16),
            mask=rmask[:, None],
        )
    aq_kernels._stress_delay(a, 3, DELAY_NS)
    if not EPOCH:
        tl.debug_barrier()
        done = tl.atomic_add(sync_ptr + H + h, 1, sem="acq_rel", scope="gpu")
        if done == chunks_per_head - 1:
            tl.atomic_xchg(sync_ptr + h, 0, sem="relaxed", scope="gpu")
            tl.atomic_xchg(sync_ptr + H + h, 0, sem="relaxed", scope="gpu")
    _ts(ts_ptr, pid, 6, TS)


@triton.jit
def _shared_query_ld_wuk(
    job,
    valid,
    wuk_ptr,
    KV_LORA: tl.constexpr,
    NOPE: tl.constexpr,
    BC: tl.constexpr,
    ABS_T: tl.constexpr,
):
    if ABS_T:
        w_lo, w_hi = _ld_wuk_t(job, wuk_ptr, KV_LORA, NOPE, BC)
    else:
        w_lo, w_hi = aq_kernels._ld_wuk(job, valid, wuk_ptr, KV_LORA, NOPE, BC)
    return w_lo, w_hi


@triton.jit
def shared_query_kernel(
    kva_ptr,
    stride_kva,
    gq_ptr,
    gkv_ptr,
    wqb_ptr,
    wuk_ptr,
    qn_ptr,
    q_ptr,
    latent_ptr,
    fq_ptr,
    stride_fq_row,
    sync_ptr,
    ep_ptr,
    ts_ptr,
    R,
    n_norm,
    n_programs,
    eps_q,
    eps_kv,
    Q_LORA: tl.constexpr,
    KV_LORA: tl.constexpr,
    ROPE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NORM_ROWS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,
    N_QB: tl.constexpr,
    N_ABS: tl.constexpr,
    ABS_JOB0: tl.constexpr,
    PRODUCER_JOB_SHIFT: tl.constexpr,
    NORM_ORDER: tl.constexpr,
    NORM_VEC: tl.constexpr,
    GEMM_T: tl.constexpr,
    ABS_T: tl.constexpr,
    USE_PDL: tl.constexpr,
    PREFETCH_L2: tl.constexpr,
    PF_W: tl.constexpr,
    PF_WUK: tl.constexpr,
    EARLY_TRIGGER: tl.constexpr,
    PRELOAD_WUK: tl.constexpr,
    POLL: tl.constexpr,
    EPOCH: tl.constexpr = False,
    TS: tl.constexpr = False,
    DELAY_NS: tl.constexpr = 0,
):
    """Programs: ``[0, n_norm)`` norm programs (``NORM_ROWS`` rows each),
    then ``N_QB`` ``q_b`` producers (producer ``t >= PRODUCER_JOB_SHIFT``
    then runs absorption job ``t - PRODUCER_JOB_SHIFT``), then ``N_ABS``
    absorption programs (jobs ``ABS_JOB0 + i``).

    Counter protocols: without ``EPOCH``, absorbed_query's (int32 ``sync_ptr``, reset by
    the last consumer of every counter). With ``EPOCH`` the counters in the
    int64 ``ep_ptr`` are never reset: ``ep_ptr[0]`` counts finished
    programs over all launches, ``ep_ptr[1]`` released norm programs and
    ``ep_ptr[2 + h]`` published no-RoPE tiles of head ``h``. The grid size
    ``n_programs`` and ``n_norm`` are the same in every launch, and a program
    reads ``ep_ptr[0]`` after ``griddepcontrol.wait`` (every earlier launch
    has finished) and before it finishes, so ``launch = ep_ptr[0] //
    n_programs`` and its wait targets are ``(launch + 1) * count``. No
    program waits for an atomic's return value.

    ``TS`` (tests only) records ``%globaltimer`` per program into
    ``ts_ptr [grid, 8]``: 0 entry, 1 after the grid-dependency wait, 2 norm
    rows released / norm flag acquired, 3 ``q`` tile stored, 4 tile
    published, 5 absorption inputs acquired, 6 absorption done, 7 exit (norm
    programs: 3 rstd ready, 4 rows stored, 5 kv norm done).
    """
    pid = tl.program_id(0)
    _ts(ts_ptr, pid, 0, TS)
    FLAG: tl.constexpr = 2 * H
    chunks_per_head: tl.constexpr = KV_LORA // BC
    if pid < n_norm:
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
            if EARLY_TRIGGER:
                tl.extra.cuda.gdc_launch_dependents()
        _ts(ts_ptr, pid, 1, TS)
        # (Names differ from the other roles' [BLOCK_M] row vectors: Triton
        # merges same-named values across the branches of a runtime if.)
        nrows = pid * NORM_ROWS + tl.arange(0, NORM_ROWS)
        nmask = nrows < R
        if NORM_VEC:
            _fi_rmsnorm_rows_vec(
                kva_ptr,
                stride_kva,
                gq_ptr,
                qn_ptr,
                Q_LORA,
                nrows,
                nmask,
                eps_q,
                Q_LORA,
                NORM_ROWS,
                ts_ptr,
                pid,
                TS,
            )
        else:
            _fi_rmsnorm_rows(
                kva_ptr,
                stride_kva,
                gq_ptr,
                qn_ptr,
                Q_LORA,
                nrows,
                nmask,
                eps_q,
                Q_LORA,
                NORM_ROWS,
                NORM_ORDER,
            )
        aq_kernels._stress_delay(pid, 11, DELAY_NS)
        # Every thread's qn stores precede the release (CTA barrier).
        tl.debug_barrier()
        if EPOCH:
            tl.atomic_add(ep_ptr + 1, 1, sem="release", scope="gpu")
        else:
            tl.atomic_add(sync_ptr + FLAG, 1, sem="release", scope="gpu")
        _ts(ts_ptr, pid, 2, TS)
        # Off the critical path: latent = [rmsnorm(compressed_kv) | k_pe].
        if NORM_VEC:
            _fi_rmsnorm_rows_vec(
                kva_ptr + Q_LORA,
                stride_kva,
                gkv_ptr,
                latent_ptr,
                KV_LORA + ROPE,
                nrows,
                nmask,
                eps_kv,
                KV_LORA,
                NORM_ROWS,
            )
        else:
            _fi_rmsnorm_rows(
                kva_ptr + Q_LORA,
                stride_kva,
                gkv_ptr,
                latent_ptr,
                KV_LORA + ROPE,
                nrows,
                nmask,
                eps_kv,
                KV_LORA,
                NORM_ROWS,
                NORM_ORDER,
            )
        rc = tl.arange(0, ROPE)
        k_pe = tl.load(
            kva_ptr + nrows[:, None] * stride_kva + Q_LORA + KV_LORA + rc[None, :],
            mask=nmask[:, None],
            other=0.0,
        )
        tl.store(
            latent_ptr + nrows[:, None] * (KV_LORA + ROPE) + KV_LORA + rc[None, :],
            k_pe,
            mask=nmask[:, None],
        )
        _ts(ts_ptr, pid, 5, TS)
        if EPOCH:
            tl.atomic_add(ep_ptr, 1, sem="relaxed", scope="gpu")
    elif pid < n_norm + N_QB:
        t = pid - n_norm
        rows = tl.arange(0, BLOCK_M)
        rmask = rows < R
        job = tl.maximum(t - PRODUCER_JOB_SHIFT, 0)
        wuk_job = (
            wuk_ptr
            + (job // chunks_per_head) * (KV_LORA * NOPE)
            + (job % chunks_per_head) * BC * NOPE
        )
        if PREFETCH_L2:
            # Weights only (never written by an earlier grid).
            _l2_prefetch_1k(wqb_ptr + t * BN * Q_LORA, PF_W, (BN * Q_LORA * 2) // 1024)
            if t >= PRODUCER_JOB_SHIFT:
                _l2_prefetch_1k(wuk_job, PF_WUK, (BC * NOPE * 2) // 1024)
        if PRELOAD_WUK:
            w_lo, w_hi = _shared_query_ld_wuk(
                job, t >= PRODUCER_JOB_SHIFT, wuk_ptr, KV_LORA, NOPE, BC, ABS_T
            )
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
            if EARLY_TRIGGER:
                tl.extra.cuda.gdc_launch_dependents()
        _ts(ts_ptr, pid, 1, TS)
        if EPOCH:
            launch1 = tl.load(ep_ptr, volatile=True) // n_programs + 1
            _wait_ge(ep_ptr + 1, launch1 * n_norm, POLL, True)
            _ts(ts_ptr, pid, 2, TS)
        else:
            # Wait for every norm program, then count this producer out; the
            # last producer resets the flag for the next launch.
            _wait_ge(sync_ptr + FLAG, n_norm, POLL)
            _ts(ts_ptr, pid, 2, TS)
            done = tl.atomic_add(sync_ptr + FLAG + 1, 1, sem="acq_rel", scope="gpu")
            if done == N_QB - 1:
                tl.atomic_xchg(sync_ptr + FLAG, 0, sem="relaxed", scope="gpu")
                tl.atomic_xchg(sync_ptr + FLAG + 1, 0, sem="relaxed", scope="gpu")
        if GEMM_T:
            _shared_query_gemm_t(
                t,
                qn_ptr,
                Q_LORA,
                wqb_ptr,
                q_ptr,
                H * HEAD_DIM,
                rows,
                rmask,
                Q_LORA,
                BLOCK_M,
                BN,
                BK,
            )
        else:
            _shared_query_gemm(
                t,
                qn_ptr,
                Q_LORA,
                wqb_ptr,
                q_ptr,
                H * HEAD_DIM,
                rows,
                rmask,
                Q_LORA,
                BLOCK_M,
                BN,
                BK,
            )
        _ts(ts_ptr, pid, 3, TS)
        if (t % (HEAD_DIM // BN)) < (NOPE // BN):
            aq_kernels._stress_delay(t, 1, DELAY_NS)
            tl.debug_barrier()
            if EPOCH:
                tl.atomic_add(ep_ptr + 2 + t // (HEAD_DIM // BN), 1, sem="release", scope="gpu")
            else:
                tl.atomic_add(sync_ptr + t // (HEAD_DIM // BN), 1, sem="release", scope="gpu")
        _ts(ts_ptr, pid, 4, TS)
        if t >= PRODUCER_JOB_SHIFT:
            if not PRELOAD_WUK:
                w_lo, w_hi = _shared_query_ld_wuk(job, True, wuk_ptr, KV_LORA, NOPE, BC, ABS_T)
            if EPOCH:
                _shared_query_absorb_job(
                    job,
                    w_lo,
                    w_hi,
                    q_ptr,
                    fq_ptr,
                    stride_fq_row,
                    sync_ptr,
                    ep_ptr + 2,
                    launch1 * (NOPE // BN),
                    rows,
                    rmask,
                    ts_ptr,
                    pid,
                    KV_LORA,
                    ROPE,
                    HEAD_DIM,
                    NOPE,
                    H,
                    BN,
                    BC,
                    POLL,
                    TS,
                    ABS_T,
                    True,
                    DELAY_NS,
                )
            else:
                _shared_query_absorb_job(
                    job,
                    w_lo,
                    w_hi,
                    q_ptr,
                    fq_ptr,
                    stride_fq_row,
                    sync_ptr,
                    sync_ptr,
                    NOPE // BN,
                    rows,
                    rmask,
                    ts_ptr,
                    pid,
                    KV_LORA,
                    ROPE,
                    HEAD_DIM,
                    NOPE,
                    H,
                    BN,
                    BC,
                    POLL,
                    TS,
                    ABS_T,
                    False,
                    DELAY_NS,
                )
        if EPOCH:
            tl.atomic_add(ep_ptr, 1, sem="relaxed", scope="gpu")
    else:
        rows = tl.arange(0, BLOCK_M)
        rmask = rows < R
        a = ABS_JOB0 + pid - n_norm - N_QB
        w_lo, w_hi = _shared_query_ld_wuk(a, True, wuk_ptr, KV_LORA, NOPE, BC, ABS_T)
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
            if EARLY_TRIGGER:
                tl.extra.cuda.gdc_launch_dependents()
        _ts(ts_ptr, pid, 1, TS)
        if EPOCH:
            launch1 = tl.load(ep_ptr, volatile=True) // n_programs + 1
            _shared_query_absorb_job(
                a,
                w_lo,
                w_hi,
                q_ptr,
                fq_ptr,
                stride_fq_row,
                sync_ptr,
                ep_ptr + 2,
                launch1 * (NOPE // BN),
                rows,
                rmask,
                ts_ptr,
                pid,
                KV_LORA,
                ROPE,
                HEAD_DIM,
                NOPE,
                H,
                BN,
                BC,
                POLL,
                TS,
                ABS_T,
                True,
                DELAY_NS,
            )
            tl.atomic_add(ep_ptr, 1, sem="relaxed", scope="gpu")
        else:
            _shared_query_absorb_job(
                a,
                w_lo,
                w_hi,
                q_ptr,
                fq_ptr,
                stride_fq_row,
                sync_ptr,
                sync_ptr,
                NOPE // BN,
                rows,
                rmask,
                ts_ptr,
                pid,
                KV_LORA,
                ROPE,
                HEAD_DIM,
                NOPE,
                H,
                BN,
                BC,
                POLL,
                TS,
                ABS_T,
                False,
                DELAY_NS,
            )
    _ts(ts_ptr, pid, 7, TS)


class SharedIndexerProjection:
    """query_front producer (preabsorbed form) for one shared-indexer layer."""

    def __init__(self, mla, config: SharedQueryKernelConfig = SharedQueryKernelConfig()):
        assert config.norm_order in NORM_ORDERS
        self.mla = mla
        self.config = config
        device = mla.q_b_proj.weight.device
        self.sync = torch.zeros(2 * _LOCAL_HEADS + 2, dtype=torch.int32, device=device)
        # Epoch protocol: [finished programs, released norm programs,
        # published no-RoPE tiles per head]; never reset.
        self.epochs = torch.zeros(2 + _LOCAL_HEADS, dtype=torch.int64, device=device)
        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

        self.use_pdl = get_env_enable_pdl()
        self.debug = {}

    def launch(self, kva: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the shared_indexer_absorbed_query kernel on a native kv_a output ``[R, 2624]``.

        Returns ``(q [R, H*256], latent [R, 576], fused_q [R, H, 576])``;
        ``fused_q[..., :512]`` holds the absorbed query, ``fused_q[...,
        512:]`` is left for the native RoPE kernel.
        """
        mla = self.mla
        cfg = self.config
        rows = kva.shape[0]
        assert kva.stride(1) == 1 and 1 <= rows <= cfg.block_m
        if cfg.cute and rows <= 8 and kva.is_contiguous():
            from . import shared_indexer_cute

            qn = kva.new_empty(rows, _Q_LORA) if cfg.debug_keep_qn else None
            ts = self._ts_buffer(72, kva.device) if cfg.debug_timestamps else None
            q, latent, fused_q = shared_indexer_cute.launch(
                mla,
                kva,
                self.use_pdl,
                cfg.early_trigger,
                qn,
                ts,
                cfg.cute_weights_before_wait,
                cfg.cute_push_async,
            )
            if cfg.debug_keep_qn:
                self.debug["qn"] = qn
            if cfg.debug_timestamps:
                self.debug["ts"] = ts
                self.debug["grid"] = dict(cute=True)
            return q, latent, fused_q
        q = kva.new_empty(rows, _LOCAL_HEADS * _HEAD_DIM)
        qn = kva.new_empty(rows, _Q_LORA)
        latent = kva.new_empty(rows, _KV_LORA + _ROPE)
        fused_q = kva.new_empty(rows, _LOCAL_HEADS, _KV_LORA + _ROPE)
        n_qb = (_LOCAL_HEADS * _HEAD_DIM) // cfg.bn
        layout = absorb_layout(n_qb, True, True, True, 64)
        pf_w = triton.next_power_of_2((cfg.bn * _Q_LORA * 2) // 1024)
        pf_wuk = triton.next_power_of_2(max(1, (layout["BC"] * _NOPE * 2) // 1024))
        # The epoch protocol needs the same grid in every launch.
        n_norm = triton.cdiv(cfg.block_m if cfg.epoch_sync else rows, cfg.norm_rows)
        grid = (n_norm + n_qb + layout["N_ABS"],)
        ts = self._ts_buffer(grid[0], kva.device) if cfg.debug_timestamps else self.sync
        compiled = shared_query_kernel[grid](
            kva,
            kva.stride(0),
            mla.q_a_layernorm.weight,
            mla.kv_a_layernorm.weight,
            mla.q_b_proj.weight,
            mla.k_b_proj_trans,
            qn,
            q,
            latent,
            fused_q,
            fused_q.stride(0),
            self.sync,
            self.epochs,
            ts,
            rows,
            n_norm,
            grid[0],
            mla.q_a_layernorm.variance_epsilon,
            mla.kv_a_layernorm.variance_epsilon,
            Q_LORA=_Q_LORA,
            KV_LORA=_KV_LORA,
            ROPE=_ROPE,
            HEAD_DIM=_HEAD_DIM,
            NOPE=_NOPE,
            H=_LOCAL_HEADS,
            BLOCK_M=cfg.block_m,
            NORM_ROWS=cfg.norm_rows,
            BN=cfg.bn,
            BK=cfg.bk,
            N_QB=n_qb,
            NORM_ORDER=cfg.norm_order,
            NORM_VEC=cfg.norm_vectorized and cfg.norm_order == 0,
            GEMM_T=cfg.gemm_transposed,
            ABS_T=cfg.absorb_transposed,
            USE_PDL=self.use_pdl,
            PREFETCH_L2=cfg.prefetch_l2,
            PF_W=pf_w,
            PF_WUK=pf_wuk,
            EARLY_TRIGGER=cfg.early_trigger,
            PRELOAD_WUK=cfg.preload_wuk,
            POLL=cfg.poll_mode,
            EPOCH=cfg.epoch_sync,
            TS=cfg.debug_timestamps,
            DELAY_NS=cfg.debug_delay_ns,
            **layout,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
            launch_pdl=self.use_pdl,
        )
        self.launch_shape = (n_norm, grid[0])
        if cfg.debug_keep_qn:
            self.debug["qn"] = qn
        if cfg.debug_timestamps:
            self.debug["compiled"] = compiled
            self.debug["ts"] = ts
            self.debug["grid"] = dict(n_norm=n_norm, n_qb=n_qb, **layout)
        return q, latent, fused_q

    def _ts_buffer(self, programs: int, device) -> torch.Tensor:
        """Tests: a persistent stamp buffer (no allocation kernel between
        the kv_a GEMM and the shared_indexer_absorbed_query kernel inside a captured graph)."""
        buf = self.debug.get("ts_buf")
        if buf is None or buf.shape[0] != programs:
            buf = torch.zeros(programs, 8, dtype=torch.int64, device=device)
            self.debug["ts_buf"] = buf
        return buf

    def counters_quiescent(self) -> bool:
        """Tests: the counters are consistent with completed launches only
        (absorbed_query protocol: all zero; epoch protocol: every counter at the same
        whole number of launches)."""
        if int(self.sync.abs().sum()) != 0:
            return False
        if not self.config.epoch_sync:
            return int(self.epochs.abs().sum()) == 0
        e = self.epochs.tolist()
        if not hasattr(self, "launch_shape"):
            return not any(e)
        n_norm, grid = self.launch_shape
        launches, rem = divmod(e[0], grid)
        tiles = _NOPE // self.config.bn
        return rem == 0 and e[1] == launches * n_norm and all(v == launches * tiles for v in e[2:])

    def project(self, hidden_states: torch.Tensor) -> AbsorbedQueryPacket:
        """Native kv_a GEMM (bridge), then the shared_indexer_absorbed_query kernel."""
        kva = self.mla.kv_a_proj_with_mqa(hidden_states)
        q, latent, fused_q = self.launch(kva)
        return AbsorbedQueryPacket([q, latent[:, :_KV_LORA], latent[:, _KV_LORA:], latent], fused_q)


def _native_rmsnorm_is_flashinfer_cute(norm, hidden: int) -> Tuple[bool, str]:
    """Whether ``norm`` (an ``RMSNorm``) runs the FlashInfer CuTe-DSL kernel
    this package reproduces."""
    from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE

    if not IS_FLASHINFER_AVAILABLE:
        return False, "FlashInfer not available"
    if (
        getattr(norm, "use_cuda_tile", False)
        or norm.use_gemma
        or getattr(norm, "is_nvfp4", False)
        or getattr(norm, "return_hp_output", False)
    ):
        return False, "RMSNorm variant not reproduced"
    w = norm.weight
    if w.dtype != torch.bfloat16 or tuple(w.shape) != (hidden,):
        return False, "RMSNorm weight must be BF16"
    try:
        from flashinfer.norm import _use_cuda_norm

        if _use_cuda_norm(w.device):
            return False, "FlashInfer CUDA (non-CuTe) RMSNorm selected"
    except ImportError:
        return False, "FlashInfer norm dispatch not inspectable"
    return True, ""


def shared_module_eligibility(mla) -> Tuple[bool, str]:
    """Static support check of shared_indexer_absorbed_query; no GPU work."""
    ok, reason = module_eligibility(mla)
    if not ok:
        return ok, reason
    if mla.mqa.indexer is not None:
        return False, "full indexer layer (shared_indexer_absorbed_query serves shared layers)"
    for name, hidden in (("q_a_layernorm", _Q_LORA), ("kv_a_layernorm", _KV_LORA)):
        ok, reason = _native_rmsnorm_is_flashinfer_cute(getattr(mla, name), hidden)
        if not ok:
            return False, f"{name}: {reason}"
    return True, ""


def shared_dsa_proj(
    mla, position_ids: Optional[torch.Tensor], hidden_states: torch.Tensor, attn_metadata
) -> Optional[AbsorbedQueryPacket]:
    """shared_indexer_absorbed_query: preabsorbed query_front packet for a shared-indexer target layer.

    Returns ``None`` (before any GPU work) for full-indexer and draft layers
    and outside the support domain; the caller then runs absorbed_query's ``dsa_proj``
    (or the native path). The per-module decision is cached in
    ``mla._shared_indexer_query`` and logged once under ``shared_indexer_absorbed_query``.
    """
    del position_ids  # shared layers have no indexer RoPE
    runner = getattr(mla, "_shared_indexer_query", None)
    if runner is None:
        check_package_selectors()
        ok, reason = shared_module_eligibility(mla)
        log_capability("shared_indexer_absorbed_query", f"layer{mla.layer_idx}", ok, reason)
        runner = SharedIndexerProjection(mla) if ok else False
        mla._shared_indexer_query = runner
    if runner is False:
        return None
    ok, _ = call_eligibility(mla, hidden_states, attn_metadata)
    if not ok or hidden_states.shape[0] > _SHARED_QUERY_MAX_ROWS:
        return None
    return runner.project(hidden_states)
