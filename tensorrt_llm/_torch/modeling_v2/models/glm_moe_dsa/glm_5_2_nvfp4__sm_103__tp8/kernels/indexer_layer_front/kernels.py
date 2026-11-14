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
"""indexer_layer_front fork of absorbed_query's fused full-layer query kernel (``absorbed_query/kernels.py``).

``front_qproj_indexer_kernel`` writes the same bytes as absorbed_query's
``qproj_indexer_kernel`` on its default path (``PRENORM``: native
``q_a_layernorm`` output feeding the ``q_b`` and indexer ``wq_b`` MMA loops).
``NORM_MODE`` selects where q_a is normalized:

* 0 (``native``): ``qn`` is the native ``q_a_layernorm`` output (the same
  input as absorbed_query's PRENORM path); the producers run shared_indexer_absorbed_query's
  ``_shared_query_gemm`` on it. The indexer_layer_front changes of this mode are the trigger
  placement and the position reads after the wait;
* 1 (``inline``): q_a is normalized inside the kernel and the native RMSNorm
  launch disappears. Every ``q_b`` and indexer ``wq_b`` producer computes
  ``rstd`` of its rows from ``kva`` with the bit-exact reproduction of the
  native FlashInfer RMSNorm (``_fi_rmsnorm_rows_exact``'s arithmetic:
  lane-to-column map, per-lane fold order, xor butterfly, ``rsqrt``) and forms
  each K tile of ``qn = bf16((x * rstd) * (w + 0))`` in registers right before
  absorbed_query's MMA on it (``_front_norm_gemm``: the same tiles, products and ascending
  K order as absorbed_query's ``_prenorm_gemm``). No program waits for another to obtain
  ``qn``, and ``qn`` is not stored (tests: ``WRITE_QN``, program 0);
* 2 (``nr<k>``): q_a is normalized inside the kernel by norm programs (the
  lowest program ids ``[0, n_norm)``, ``NORM_ROWS`` rows each) that store
  ``qn`` to a per-call scratch with the same arithmetic and release counter
  ``NS`` (``atom.add.release.gpu`` after a CTA barrier); the producers spin on
  it (acquire) and run ``_shared_query_gemm`` on ``qn``.

``PREFETCH_W``: each producer asks L2 for its weight tile (``BN`` rows of
``wq_b`` / indexer ``wq_b``) before the grid-dependency wait. Every other
role is absorbed_query's.

Program roles, in program-id order: norm (``NORM_MODE`` 2), ``q_b``
producers (absorbing producers run absorption job ``p - PRODUCER_JOB_SHIFT``),
``N_IQ`` indexer ``wq_b`` producers, extra absorption programs, ``N_QPOST``
indexer query post-processing programs (one per indexer head),
``compressed_kv`` norm, indexer key post-processing. With the
``skip_reused_iq`` option (MTP draft steps that reuse the shared top-k) the
host passes ``N_IQ = N_QPOST = 0``: ``q_fp8``, ``q_scale`` and ``weights``
are left unwritten and every other output is unchanged.
``sync_ptr`` = [2 * H absorption counters, IH indexer-head counters, NS
(released norm programs), NS + 1 (consumers that passed the norm wait)].

Memory order: before ``griddepcontrol.wait`` a
program reads only immutable weights (the absorbing producers' ``W_UK``
slice, the producers' L2 prefetch of their weight tile); positions, cos/sin
rows, activations, counters and every output come after it (absorbed_query reads
positions before its wait; this fork does not). The last
consumer of ``NS`` (``NS + 1`` reaching ``n_cons - 1``) resets both words
after every consumer has passed its wait; counters are touched only after
the grid-dependency wait, so a launch never sees the counters of an
incomplete earlier launch. Every spin-wait targets counters released by
lower program ids, so the launch completes with one resident program slot
under in-order dispatch (absorbed_query's progress argument, norm programs first).

Trigger (``EARLY_TRIGGER``): every program executes
``griddepcontrol.launch_dependents`` right after its last global load, which
in every role comes after all of that program's spin-waits. The dependent
(the generation rope) is released only when every program has done so, i.e.
when all programs are resident and past their waits, so its residency
cannot delay this grid. The dependent reads this kernel's outputs only after
its own ``griddepcontrol.wait``.
"""

import triton
import triton.language as tl

from ..absorbed_query.kernels import (
    _fi_butterfly32,
    _fp8_ue8m0_scale,
    _ld_wuk,
    _rope_pairs,
    _stress_delay,
    _v8_leaves,
)
from ..absorbed_query.shared_indexer import _l2_prefetch_1k, _shared_query_gemm


@triton.jit
def _trigger(EARLY_TRIGGER: tl.constexpr):
    if EARLY_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _front_rmsnorm_rows(
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
    EARLY_TRIGGER: tl.constexpr,
):
    """``_fi_rmsnorm_rows_exact`` (absorbed_query) with the trigger between its last load and its store."""
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
    cols = tl.arange(0, H)
    x = tl.load(
        x_ptr + rows[:, None] * stride_x + cols[None, :], mask=rmask[:, None], other=0.0
    ).to(tl.float32)
    w = tl.load(w_ptr + cols).to(tl.float32) + 0.0
    _trigger(EARLY_TRIGGER)
    y = (x * rstd[:, None]) * w[None, :]
    tl.store(
        y_ptr + rows[:, None] * stride_y + cols[None, :], y.to(tl.bfloat16), mask=rmask[:, None]
    )


@triton.jit
def _front_rstd(x_ptr, stride_x, rows, rmask, eps, H: tl.constexpr, ROWS: tl.constexpr):
    """``rstd`` of ``_fi_rmsnorm_rows_exact`` (absorbed_query): same loads, fold order, butterfly and ``rsqrt``."""
    lanes = tl.arange(0, 32)
    vv = tl.arange(0, 8)
    offs = rows[:, None, None] * stride_x + 8 * lanes[None, :, None] + vv[None, None, :]
    s = tl.zeros([ROWS, 32], dtype=tl.float32)
    # A rolled loop over the 256-column blocks (same fold order) bounds the live registers.
    for b in tl.range(0, H // 256, num_stages=2):
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
    return tl.rsqrt(_fi_butterfly32(s, ROWS) / H + eps)


@triton.jit
def _front_norm_gemm(
    t,
    x_ptr,
    stride_x,
    g_ptr,
    w_ptr,
    out_ptr,
    OUT_COLS: tl.constexpr,
    rows,
    rmask,
    eps,
    qn_ptr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    WRITE_QN: tl.constexpr,
    EARLY_TRIGGER: tl.constexpr,
):
    """``out[:, t*BN:(t+1)*BN] = qn @ W[t*BN:(t+1)*BN]^T`` with ``qn`` formed tile by tile.

    ``qn = bf16((x * rstd) * (w + 0))`` element for element as the native
    FlashInfer RMSNorm (``_fi_rmsnorm_rows_exact``), then absorbed_query's MMA sequence
    (``_prenorm_gemm``: same tiles, products and ascending K order). With
    ``WRITE_QN`` (tests) the caller's program 0 also stores ``qn``.
    """
    rstd = _front_rstd(x_ptr, stride_x, rows, rmask, eps, K, BLOCK_M)
    cols = tl.arange(0, BK)
    n_offs = t * BN + tl.arange(0, BN)
    acc = tl.zeros([BLOCK_M, BN], dtype=tl.float32)
    for k0 in tl.range(0, K, BK):
        x = tl.load(
            x_ptr + rows[:, None] * stride_x + (k0 + cols)[None, :], mask=rmask[:, None], other=0.0
        ).to(tl.float32)
        g = tl.load(g_ptr + k0 + cols).to(tl.float32) + 0.0
        xn = ((x * rstd[:, None]) * g[None, :]).to(tl.bfloat16)
        if WRITE_QN:
            if t == 0:
                tl.store(qn_ptr + rows[:, None] * K + (k0 + cols)[None, :], xn, mask=rmask[:, None])
        w = tl.load(w_ptr + n_offs[None, :] * K + (k0 + cols)[:, None])
        acc = tl.dot(xn, w, acc)
    _trigger(EARLY_TRIGGER)
    tl.store(
        out_ptr + rows[:, None] * OUT_COLS + n_offs[None, :],
        acc.to(tl.bfloat16),
        mask=rmask[:, None],
    )


@triton.jit
def _prefetch_w_tile(t, w_ptr, K: tl.constexpr, BN: tl.constexpr, PREFETCH_W: tl.constexpr):
    """L2 prefetch of weight rows ``[t*BN, (t+1)*BN)`` (``BN * K`` BF16, contiguous)."""
    if PREFETCH_W:
        n_chunks: tl.constexpr = BN * K * 2 // 1024
        _l2_prefetch_1k(w_ptr + t * BN * K, n_chunks, n_chunks)


@triton.jit
def _wait_norm(sync_ptr, NS: tl.constexpr, n_norm, n_cons):
    """Acquire-spin until every norm program has released ``qn``, then count
    this consumer; the last one resets both counters for the next launch."""
    ready = tl.atomic_add(sync_ptr + NS, 0, sem="acquire", scope="gpu")
    while ready < n_norm:
        ready = tl.atomic_add(sync_ptr + NS, 0, sem="acquire", scope="gpu")
    tl.debug_barrier()
    done = tl.atomic_add(sync_ptr + NS + 1, 1, sem="acq_rel", scope="gpu")
    if done == n_cons - 1:
        tl.atomic_xchg(sync_ptr + NS, 0, sem="relaxed", scope="gpu")
        tl.atomic_xchg(sync_ptr + NS + 1, 0, sem="relaxed", scope="gpu")


@triton.jit
def _front_absorb_job(
    a,
    w_lo,
    w_hi,
    q_ptr,
    fq_ptr,
    stride_fq_row,
    sync_ptr,
    rows,
    rmask,
    KV_LORA: tl.constexpr,
    ROPE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE: tl.constexpr,
    H: tl.constexpr,
    BN: tl.constexpr,
    BC: tl.constexpr,
    EARLY_TRIGGER: tl.constexpr,
    DELAY_NS: tl.constexpr = 0,
):
    """absorbed_query's ``_absorb_job`` with the trigger after the ``q`` tile loads."""
    chunks_per_head: tl.constexpr = KV_LORA // BC
    h = a // chunks_per_head
    c0 = (a % chunks_per_head) * BC
    cc = tl.arange(0, BC)
    d_lo = tl.arange(0, 128)
    d_hi = tl.arange(0, NOPE - 128)
    ready = tl.atomic_add(sync_ptr + h, 0, sem="acquire", scope="gpu")
    while ready < NOPE // BN:
        ready = tl.atomic_add(sync_ptr + h, 0, sem="acquire", scope="gpu")
    tl.debug_barrier()
    q_base = q_ptr + rows[:, None] * (H * HEAD_DIM) + h * HEAD_DIM
    q_lo = tl.load(q_base + d_lo[None, :], mask=rmask[:, None], other=0.0, cache_modifier=".cg")
    q_hi = tl.load(
        q_base + 128 + d_hi[None, :], mask=rmask[:, None], other=0.0, cache_modifier=".cg"
    )
    _trigger(EARLY_TRIGGER)
    acc = tl.dot(q_lo, w_lo)
    acc = tl.dot(q_hi, w_hi, acc)
    tl.store(
        fq_ptr + rows[:, None] * stride_fq_row + h * (KV_LORA + ROPE) + (c0 + cc)[None, :],
        acc.to(tl.bfloat16),
        mask=rmask[:, None],
    )
    _stress_delay(a, 3, DELAY_NS)
    tl.debug_barrier()
    done = tl.atomic_add(sync_ptr + H + h, 1, sem="acq_rel", scope="gpu")
    if done == chunks_per_head - 1:
        tl.atomic_xchg(sync_ptr + h, 0, sem="relaxed", scope="gpu")
        tl.atomic_xchg(sync_ptr + H + h, 0, sem="relaxed", scope="gpu")


@triton.jit
def _front_qpost_program(
    i,
    iq_ptr,
    isync_ptr,
    kw_ptr,
    stride_kw,
    pos_ptr,
    cos_sin_ptr,
    qfp8_ptr,
    qscale_ptr,
    wout_ptr,
    w_scale,
    rows,
    rmask,
    IHD: tl.constexpr,
    IH: tl.constexpr,
    ROPE_I: tl.constexpr,
    BN: tl.constexpr,
    USE_PDL: tl.constexpr,
    EARLY_TRIGGER: tl.constexpr,
    DELAY_NS: tl.constexpr = 0,
):
    """absorbed_query's ``_qpost_program``; positions and cos/sin are read after the wait."""
    half: tl.constexpr = ROPE_I // 2
    nope: tl.constexpr = IHD - ROPE_I
    j = tl.arange(0, half)
    c = tl.arange(0, nope)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    pos = tl.load(pos_ptr + rows, mask=rmask, other=0).to(tl.int64)
    cs_row = cos_sin_ptr + pos[:, None] * ROPE_I
    cos = tl.load(cs_row + j[None, :], mask=rmask[:, None], other=0.0)
    sin = tl.load(cs_row + half + j[None, :], mask=rmask[:, None], other=0.0)
    w = tl.load(kw_ptr + rows * stride_kw + IHD + i, mask=rmask, other=0.0).to(tl.float32)

    ready = tl.atomic_add(isync_ptr + i, 0, sem="acquire", scope="gpu")
    while ready < IHD // BN:
        ready = tl.atomic_add(isync_ptr + i, 0, sem="acquire", scope="gpu")
    tl.debug_barrier()

    base = iq_ptr + rows[:, None] * (IH * IHD) + i * IHD
    xe = tl.load(base + 2 * j[None, :], mask=rmask[:, None], other=0.0, cache_modifier=".cg").to(
        tl.float32
    )
    xo = tl.load(
        base + 2 * j[None, :] + 1, mask=rmask[:, None], other=0.0, cache_modifier=".cg"
    ).to(tl.float32)
    xn = tl.load(
        base + ROPE_I + c[None, :], mask=rmask[:, None], other=0.0, cache_modifier=".cg"
    ).to(tl.float32)
    _trigger(EARLY_TRIGGER)
    oe, oo = _rope_pairs(xe, xo, cos, sin)
    amax = tl.maximum(
        tl.maximum(tl.max(tl.abs(oe), axis=1), tl.max(tl.abs(oo), axis=1)),
        tl.max(tl.abs(xn), axis=1),
    )
    scale, inv = _fp8_ue8m0_scale(amax)
    out = qfp8_ptr + rows[:, None] * (IH * IHD) + i * IHD
    tl.store(out + 2 * j[None, :], (oe * inv[:, None]).to(tl.float8e4nv), mask=rmask[:, None])
    tl.store(out + 2 * j[None, :] + 1, (oo * inv[:, None]).to(tl.float8e4nv), mask=rmask[:, None])
    tl.store(out + ROPE_I + c[None, :], (xn * inv[:, None]).to(tl.float8e4nv), mask=rmask[:, None])
    tl.store(qscale_ptr + rows * IH + i, scale, mask=rmask)
    tl.store(wout_ptr + rows * IH + i, w * scale * w_scale, mask=rmask)
    # Single consumer per head: reset for the next launch.
    _stress_delay(i, 7, DELAY_NS)
    tl.atomic_xchg(isync_ptr + i, 0, sem="relaxed", scope="gpu")


@triton.jit
def _front_kpost_program(
    kln_ptr,
    stride_kln,
    pos_ptr,
    cos_sin_ptr,
    kfp8_ptr,
    kscale_ptr,
    rows,
    rmask,
    IHD: tl.constexpr,
    ROPE_I: tl.constexpr,
    USE_PDL: tl.constexpr,
    EARLY_TRIGGER: tl.constexpr,
):
    """absorbed_query's ``_kpost_program``; positions and cos/sin are read after the wait."""
    half: tl.constexpr = ROPE_I // 2
    nope: tl.constexpr = IHD - ROPE_I
    j = tl.arange(0, half)
    c = tl.arange(0, nope)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    pos = tl.load(pos_ptr + rows, mask=rmask, other=0).to(tl.int64)
    cs_row = cos_sin_ptr + pos[:, None] * ROPE_I
    cos = tl.load(cs_row + j[None, :], mask=rmask[:, None], other=0.0)
    sin = tl.load(cs_row + half + j[None, :], mask=rmask[:, None], other=0.0)
    base = kln_ptr + rows[:, None] * stride_kln
    ye = tl.load(base + 2 * j[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    yo = tl.load(base + 2 * j[None, :] + 1, mask=rmask[:, None], other=0.0).to(tl.float32)
    yn = tl.load(base + ROPE_I + c[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    _trigger(EARLY_TRIGGER)
    oe, oo = _rope_pairs(ye, yo, cos, sin)
    amax = tl.maximum(
        tl.maximum(tl.max(tl.abs(oe), axis=1), tl.max(tl.abs(oo), axis=1)),
        tl.max(tl.abs(yn), axis=1),
    )
    scale, inv = _fp8_ue8m0_scale(amax)
    out = kfp8_ptr + rows[:, None] * IHD
    tl.store(out + 2 * j[None, :], (oe * inv[:, None]).to(tl.float8e4nv), mask=rmask[:, None])
    tl.store(out + 2 * j[None, :] + 1, (oo * inv[:, None]).to(tl.float8e4nv), mask=rmask[:, None])
    tl.store(out + ROPE_I + c[None, :], (yn * inv[:, None]).to(tl.float8e4nv), mask=rmask[:, None])
    tl.store(kscale_ptr + rows, scale, mask=rmask)


@triton.jit
def _front_kv_program(
    kva_ptr,
    stride_kva,
    gkv_ptr,
    latent_ptr,
    rows,
    rmask,
    eps,
    Q_LORA: tl.constexpr,
    KV_LORA: tl.constexpr,
    ROPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    EARLY_TRIGGER: tl.constexpr,
):
    """absorbed_query's ``_kv_program`` (native ``kv_a_layernorm``, bit for bit, plus ``k_pe``)."""
    rc = tl.arange(0, ROPE)
    k_pe = tl.load(
        kva_ptr + rows[:, None] * stride_kva + Q_LORA + KV_LORA + rc[None, :],
        mask=rmask[:, None],
        other=0.0,
    )
    _front_rmsnorm_rows(
        kva_ptr + Q_LORA,
        stride_kva,
        gkv_ptr,
        latent_ptr,
        KV_LORA + ROPE,
        rows,
        rmask,
        eps,
        KV_LORA,
        BLOCK_M,
        EARLY_TRIGGER,
    )
    tl.store(
        latent_ptr + rows[:, None] * (KV_LORA + ROPE) + KV_LORA + rc[None, :],
        k_pe,
        mask=rmask[:, None],
    )


@triton.jit
def front_qproj_indexer_kernel(
    kva_ptr,
    stride_kva,
    gq_ptr,
    gkv_ptr,
    wqb_ptr,
    wuk_ptr,
    q_ptr,
    latent_ptr,
    fq_ptr,
    stride_fq_row,
    sync_ptr,
    qn_ptr,
    n_norm,
    R,
    eps_q,
    eps_kv,
    wiq_ptr,
    iq_ptr,
    kw_ptr,
    stride_kw,
    kln_ptr,
    stride_kln,
    pos_ptr,
    cos_sin_ptr,
    qfp8_ptr,
    qscale_ptr,
    kfp8_ptr,
    kscale_ptr,
    wout_ptr,
    w_scale,
    Q_LORA: tl.constexpr,
    KV_LORA: tl.constexpr,
    ROPE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE: tl.constexpr,
    H: tl.constexpr,
    IHD: tl.constexpr,
    IH: tl.constexpr,
    ROPE_I: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,
    N_QB: tl.constexpr,
    N_IQ: tl.constexpr,
    N_ABS: tl.constexpr,
    DO_ABSORB: tl.constexpr,
    NORM_ROWS: tl.constexpr,
    USE_PDL: tl.constexpr,
    EARLY_TRIGGER: tl.constexpr,
    ABSORB_IN_PRODUCER: tl.constexpr = False,
    PRODUCER_JOB_SHIFT: tl.constexpr = 0,
    ABS_JOB0: tl.constexpr = 0,
    NORM_MODE: tl.constexpr = 0,
    N_QPOST: tl.constexpr = 0,
    PREFETCH_W: tl.constexpr = False,
    WRITE_QN: tl.constexpr = False,
    DELAY_NS: tl.constexpr = 0,
):
    """See the module docstring. ``n_norm`` (runtime) = ceil(R / NORM_ROWS) norm programs
    with ``NORM_MODE`` 2, else 0."""
    NS: tl.constexpr = 2 * H + IH
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    rmask = rows < R
    t = pid - n_norm
    if pid < n_norm:
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
        nrows = pid * NORM_ROWS + tl.arange(0, NORM_ROWS)
        _front_rmsnorm_rows(
            kva_ptr,
            stride_kva,
            gq_ptr,
            qn_ptr,
            Q_LORA,
            nrows,
            nrows < R,
            eps_q,
            Q_LORA,
            NORM_ROWS,
            EARLY_TRIGGER,
        )
        _stress_delay(pid, 11, DELAY_NS)
        tl.debug_barrier()
        tl.atomic_add(sync_ptr + NS, 1, sem="release", scope="gpu")
    elif t < N_QB:
        _prefetch_w_tile(t, wqb_ptr, Q_LORA, BN, PREFETCH_W)
        if ABSORB_IN_PRODUCER:
            job = tl.maximum(t - PRODUCER_JOB_SHIFT, 0)
            w_lo, w_hi = _ld_wuk(job, t >= PRODUCER_JOB_SHIFT, wuk_ptr, KV_LORA, NOPE, BC)
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
        if NORM_MODE == 1:
            _front_norm_gemm(
                t,
                kva_ptr,
                stride_kva,
                gq_ptr,
                wqb_ptr,
                q_ptr,
                H * HEAD_DIM,
                rows,
                rmask,
                eps_q,
                qn_ptr,
                Q_LORA,
                BLOCK_M,
                BN,
                BK,
                WRITE_QN,
                False,
            )
        else:
            if NORM_MODE == 2:
                _wait_norm(sync_ptr, NS, n_norm, N_QB + N_IQ)
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
        if DO_ABSORB:
            if (t % (HEAD_DIM // BN)) < (NOPE // BN):
                _stress_delay(t, 1, DELAY_NS)
                # Publish this no-RoPE tile of head t // (HEAD_DIM // BN).
                tl.debug_barrier()
                tl.atomic_add(sync_ptr + t // (HEAD_DIM // BN), 1, sem="release", scope="gpu")
        if ABSORB_IN_PRODUCER:
            if t >= PRODUCER_JOB_SHIFT:
                _front_absorb_job(
                    job,
                    w_lo,
                    w_hi,
                    q_ptr,
                    fq_ptr,
                    stride_fq_row,
                    sync_ptr,
                    rows,
                    rmask,
                    KV_LORA,
                    ROPE,
                    HEAD_DIM,
                    NOPE,
                    H,
                    BN,
                    BC,
                    EARLY_TRIGGER,
                    DELAY_NS,
                )
            else:
                _trigger(EARLY_TRIGGER)
        else:
            _trigger(EARLY_TRIGGER)
    elif t < N_QB + N_IQ:
        it = t - N_QB
        _prefetch_w_tile(it, wiq_ptr, Q_LORA, BN, PREFETCH_W)
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
        if NORM_MODE == 1:
            # Triggers between its last load and its store.
            _front_norm_gemm(
                it,
                kva_ptr,
                stride_kva,
                gq_ptr,
                wiq_ptr,
                iq_ptr,
                IH * IHD,
                rows,
                rmask,
                eps_q,
                qn_ptr,
                Q_LORA,
                BLOCK_M,
                BN,
                BK,
                False,
                EARLY_TRIGGER,
            )
        else:
            if NORM_MODE == 2:
                _wait_norm(sync_ptr, NS, n_norm, N_QB + N_IQ)
            _shared_query_gemm(
                it, qn_ptr, Q_LORA, wiq_ptr, iq_ptr, IH * IHD, rows, rmask, Q_LORA, BLOCK_M, BN, BK
            )
            _trigger(EARLY_TRIGGER)
        _stress_delay(it, 5, DELAY_NS)
        tl.debug_barrier()
        tl.atomic_add(sync_ptr + 2 * H + it // (IHD // BN), 1, sem="release", scope="gpu")
    elif t < N_QB + N_IQ + N_ABS:
        a = ABS_JOB0 + t - N_QB - N_IQ
        w_lo, w_hi = _ld_wuk(a, True, wuk_ptr, KV_LORA, NOPE, BC)
        # Weights are never written by an earlier grid; only the counter and
        # buffer accesses wait for the programmatic dependency.
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
        _front_absorb_job(
            a,
            w_lo,
            w_hi,
            q_ptr,
            fq_ptr,
            stride_fq_row,
            sync_ptr,
            rows,
            rmask,
            KV_LORA,
            ROPE,
            HEAD_DIM,
            NOPE,
            H,
            BN,
            BC,
            EARLY_TRIGGER,
            DELAY_NS,
        )
    elif t < N_QB + N_IQ + N_ABS + N_QPOST:
        _front_qpost_program(
            t - N_QB - N_IQ - N_ABS,
            iq_ptr,
            sync_ptr + 2 * H,
            kw_ptr,
            stride_kw,
            pos_ptr,
            cos_sin_ptr,
            qfp8_ptr,
            qscale_ptr,
            wout_ptr,
            w_scale,
            rows,
            rmask,
            IHD,
            IH,
            ROPE_I,
            BN,
            USE_PDL,
            EARLY_TRIGGER,
            DELAY_NS,
        )
    elif t == N_QB + N_IQ + N_ABS + N_QPOST:
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
        _front_kv_program(
            kva_ptr,
            stride_kva,
            gkv_ptr,
            latent_ptr,
            rows,
            rmask,
            eps_kv,
            Q_LORA,
            KV_LORA,
            ROPE,
            BLOCK_M,
            EARLY_TRIGGER,
        )
    else:
        _front_kpost_program(
            kln_ptr,
            stride_kln,
            pos_ptr,
            cos_sin_ptr,
            kfp8_ptr,
            kscale_ptr,
            rows,
            rmask,
            IHD,
            ROPE_I,
            USE_PDL,
            EARLY_TRIGGER,
        )


def grid_size(n_norm: int, n_qb: int, n_iq: int, n_abs: int, ih: int) -> int:
    """Programs of one launch (see the role list in the module docstring)."""
    return n_norm + n_qb + n_iq + n_abs + ih + 2


def num_norm_programs(rows: int, norm_rows: int) -> int:
    return triton.cdiv(rows, norm_rows)
