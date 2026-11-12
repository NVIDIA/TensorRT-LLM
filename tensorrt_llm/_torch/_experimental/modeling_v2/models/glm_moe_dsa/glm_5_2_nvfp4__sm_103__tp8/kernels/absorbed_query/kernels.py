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
"""Triton kernels of the absorbed_query/projection package.

``qproj_kernel`` consumes the native fused ``kv_a_proj_with_mqa`` output
``kva = [q_a | compressed_kv | k_pe]`` of ``R <= BLOCK_M`` token rows (and,
with ``PRENORM``, the native ``q_a_layernorm`` output) and produces, in one
launch:

* ``q = q_b_proj(rmsnorm(q_a))`` ``[R, H * 256]`` (BF16, the native ``q``),
* optionally ``qr = rmsnorm(q_a)`` ``[R, 2048]`` for the native indexer,
* ``latent = [rmsnorm(compressed_kv) | k_pe]`` ``[R, 576]``,
* optionally the absorbed query ``fused_q[:, h, :512] = q_nope[:, h] @ W_UK[h]``
  written into the caller's ``[R, H, 576]`` buffer.

``qproj_indexer_kernel`` additionally produces the token-wise indexer
outputs of full indexer layers.

Numerics follow the native chain bit for bit on the default path
(``PRENORM``, the native kv_a GEMM, ``q_a_layernorm``, fused ``wk`` /
``weights_proj`` GEMM and ``k_norm`` as bridges): ``q`` and the indexer
``wq_b`` output are FP32-accumulated MMA chains in ascending K order (equal to
the native GEMMs), the ``compressed_kv`` RMSNorm reproduces the native
FlashInfer kernel's reduction order (``_fi_rmsnorm_rows_exact``), the
indexer RoPE its FMA pattern (``_rope_pairs``) and the FP8 tail the native
``fusedCatFp8Kernel``; ``q`` is rounded to BF16 before the absorption (native
intermediate). The experimental paths (own hidden GEMV, prefetching kernel's
in-kernel q_a norm, ``PRENORM=False``) are not bit-exact.

Program roles, in program-id order (each kernel documents its own list):
``q_b`` producer tiles ``[0, N_QB)`` (each ``BN`` output columns over the full
``K``; ``TPH = HEAD_DIM // BN`` tiles per head, the first ``NOPE // BN`` of a
head are its no-RoPE tiles), then indexer ``wq_b`` producer tiles, then the
programs that only consume.

Inter-program synchronization contract
--------------------------------------
* A producer stores its tile, executes a CTA barrier and then one thread
  releases the tile with ``atom.add.release.gpu`` on a per-head counter. A
  consumer spins with ``atom.add.acquire.gpu`` (adding 0) until the counter
  reaches the number of producer tiles it needs, executes a CTA barrier and
  only then reads the tiles, with ``ld.global.cg`` (L2, never a stale L1 line).
* Every counter is per module (per layer) and is zero between launches. The
  last consumer of a counter resets it (relaxed exchange) after every
  consumer of that counter has read its tiles: for the absorption counters a
  second ``done`` counter (``acq_rel``) counts the consumers of a head; an
  indexer-head counter has a single consumer. Counters are touched only after
  ``griddepcontrol.wait``, so under PDL a launch never observes the counters
  of an earlier launch that has not completed.
* Forward progress: every spin-wait waits only on counters released by
  programs with a *lower* program id. Absorption job ``j`` (head
  ``j // TPH``, ``BC`` latent columns) is run, with ``ABSORB_IN_PRODUCER``
  and ``PRODUCER_JOB_SHIFT == TPH``, by producer ``j + TPH`` (a producer of
  the next head) after it has published its own tile, and the last head's
  jobs by ``TPH`` extra consumer programs placed after all producers. So the
  lowest-numbered unfinished program never waits, and a launch completes
  with a single resident program slot when programs are dispatched in id
  order, and on any dispatch order when the whole grid is resident at once
  (the default pipelined grids, <= 234 programs of <= 96 KB shared memory,
  fit on an idle sm_103 GPU; the experimental prefetching kernel's do not).
  ``PRODUCER_JOB_SHIFT == 0`` is the legacy layout (producer ``p`` runs job
  ``p`` of its own head group), which needs ``NOPE // BN`` co-resident
  programs of one head group; it is kept only as a stress-test reference.
* ``DELAY_NS > 0`` (tests only) delays a deterministic subset of producers
  before they publish and of consumers before they finish, by up to
  ``DELAY_NS`` nanoseconds of ``%globaltimer``, to stress the protocol.
"""

import triton
import triton.language as tl


@triton.jit
def _rms_rcp(
    src_ptr, stride_src, rows, rmask, N: tl.constexpr, BLOCK_M: tl.constexpr, BK: tl.constexpr, eps
):
    cols = tl.arange(0, BK)
    ss = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k0 in tl.range(0, N, BK, num_stages=N // BK):
        x = tl.load(
            src_ptr + rows[:, None] * stride_src + (k0 + cols)[None, :],
            mask=rmask[:, None],
            other=0.0,
        ).to(tl.float32)
        ss += tl.sum(x * x, axis=1)
    return tl.rsqrt(ss / N + eps)


@triton.jit
def _rms_rcp_partials(
    ss_ptr,
    group: tl.constexpr,
    n_parts,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MAX_PARTS: tl.constexpr,
    eps,
):
    parts = tl.arange(0, MAX_PARTS)
    rows = tl.arange(0, BLOCK_M)
    ss = tl.load(
        ss_ptr + parts[:, None] * 2 * BLOCK_M + group * BLOCK_M + rows[None, :],
        mask=(parts < n_parts)[:, None],
        other=0.0,
    )
    return tl.rsqrt(tl.sum(ss, axis=0) / N + eps)


# --------------------------------------------------------------------------
# Bit-exact reproduction of the native FlashInfer CuTe-DSL RMSNorm (the
# ``kv_a_layernorm`` / ``q_a_layernorm`` kernels, H in {512, 2048}): one warp
# per row, lane t owns columns 256 * b + 8 * t + v (b < H / 256, v < 8), a
# sequential per-lane fold of the squares in column order (blocks outer,
# columns inner), a butterfly over the 32 lanes (xor offsets 1, 2, 4, 8, 16),
# rstd = rsqrt(sum / H + eps), y = bf16((x * rstd) * (w + 0)). Every add is
# explicit (reshape/split, sums over axes of size 2), so the result does not
# depend on the compiled layout or on the rows per program. The same
# arithmetic as ``shared_indexer._fi_rmsnorm_rows_vec`` (shared_indexer_absorbed_query), checked bit
# for bit against the native kernel on real and wide-range rows.
# --------------------------------------------------------------------------


@triton.jit
def _fi_butterfly32(s, ROWS: tl.constexpr):
    """Sum of ``s [ROWS, 32]`` over the lanes in shuffle-xor butterfly order
    (offsets 1, 2, 4, 8, 16): a balanced pairwise tree in lane order."""
    s = tl.sum(tl.reshape(s, [ROWS, 16, 2]), axis=2)
    s = tl.sum(tl.reshape(s, [ROWS, 8, 2]), axis=2)
    s = tl.sum(tl.reshape(s, [ROWS, 4, 2]), axis=2)
    s = tl.sum(tl.reshape(s, [ROWS, 2, 2]), axis=2)
    return tl.sum(s, axis=1)


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
def _fi_rmsnorm_rows_exact(
    x_ptr, stride_x, w_ptr, y_ptr, stride_y, rows, rmask, eps, H: tl.constexpr, ROWS: tl.constexpr
):
    """``y[rows, :H] = bf16(rmsnorm(x[rows, :H]) * w)`` bit for bit as the
    native FlashInfer RMSNorm (see above)."""
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
    y = (x * rstd[:, None]) * w[None, :]
    tl.store(
        y_ptr + rows[:, None] * stride_y + cols[None, :], y.to(tl.bfloat16), mask=rmask[:, None]
    )


@triton.jit
def _stress_delay(pid, salt: tl.constexpr, DELAY_NS: tl.constexpr):
    """Test-only skew: spin ``(h / 10) * DELAY_NS`` ns, ``h = (37 pid + salt)
    mod 11`` (so some programs do not wait and others wait up to DELAY_NS)."""
    if DELAY_NS > 0:
        wait_ns = ((pid * 37 + salt) % 11) * (DELAY_NS // 10)
        t0 = tl.extra.cuda.globaltimer()
        while tl.extra.cuda.globaltimer() - t0 < wait_ns:
            pass


@triton.jit
def _ld_wuk(a, valid, wuk_ptr, KV_LORA: tl.constexpr, NOPE: tl.constexpr, BC: tl.constexpr):
    """``W_UK[h, c0:c0+BC, :NOPE]`` of absorption job ``a`` as [NOPE, BC]
    (split 128 + NOPE-128 along NOPE); nothing is read unless ``valid``."""
    chunks_per_head: tl.constexpr = KV_LORA // BC
    h = a // chunks_per_head
    c0 = (a % chunks_per_head) * BC
    cc = tl.arange(0, BC)
    w_base = wuk_ptr + h * (KV_LORA * NOPE) + (c0 + cc)[None, :] * NOPE
    w_lo = tl.load(w_base + tl.arange(0, 128)[:, None], mask=valid, other=0.0)
    w_hi = tl.load(w_base + 128 + tl.arange(0, NOPE - 128)[:, None], mask=valid, other=0.0)
    return w_lo, w_hi


@triton.jit
def _qb_program(
    pid,
    kva_ptr,
    stride_kva,
    gq_ptr,
    wqb_ptr,
    q_ptr,
    qr_ptr,
    sync_ptr,
    ss_ptr,
    n_parts,
    rows,
    rmask,
    eps,
    Q_LORA: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    WRITE_QR: tl.constexpr,
    DO_ABSORB: tl.constexpr,
    SS_PARTIALS: tl.constexpr,
    MAX_PARTS: tl.constexpr,
    PRENORM: tl.constexpr,
    DELAY_NS: tl.constexpr = 0,
):
    if PRENORM:
        # kva_ptr/stride_kva address the already normalized BF16 q_a rows.
        _prenorm_gemm(
            pid,
            kva_ptr,
            stride_kva,
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
        _norm_gemm(
            pid,
            kva_ptr,
            stride_kva,
            gq_ptr,
            wqb_ptr,
            q_ptr,
            qr_ptr,
            ss_ptr,
            n_parts,
            rows,
            rmask,
            eps,
            Q_LORA,
            HEAD_DIM,
            H,
            BLOCK_M,
            BN,
            BK,
            WRITE_QR,
            SS_PARTIALS,
            MAX_PARTS,
        )
    if DO_ABSORB:
        if (pid % (HEAD_DIM // BN)) < (NOPE // BN):
            _stress_delay(pid, 1, DELAY_NS)
            # Publish this no-RoPE tile of head pid // (HEAD_DIM // BN): every
            # thread's store is ordered before the release by the barrier.
            tl.debug_barrier()
            tl.atomic_add(sync_ptr + pid // (HEAD_DIM // BN), 1, sem="release", scope="gpu")


@triton.jit
def _prenorm_gemm(
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
    """``out[:, t*BN:(t+1)*BN] = x @ W[t*BN:(t+1)*BN]^T`` for BF16 ``x``."""
    cols = tl.arange(0, BK)
    n_offs = t * BN + tl.arange(0, BN)
    acc = tl.zeros([BLOCK_M, BN], dtype=tl.float32)
    for k0 in tl.range(0, K, BK):
        x = tl.load(
            x_ptr + rows[:, None] * stride_x + (k0 + cols)[None, :], mask=rmask[:, None], other=0.0
        )
        w = tl.load(w_ptr + n_offs[None, :] * K + (k0 + cols)[:, None])
        acc = tl.dot(x, w, acc)
    tl.store(
        out_ptr + rows[:, None] * OUT_COLS + n_offs[None, :],
        acc.to(tl.bfloat16),
        mask=rmask[:, None],
    )


@triton.jit
def _norm_gemm(
    pid,
    kva_ptr,
    stride_kva,
    gq_ptr,
    wqb_ptr,
    q_ptr,
    qr_ptr,
    ss_ptr,
    n_parts,
    rows,
    rmask,
    eps,
    Q_LORA: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    WRITE_QR: tl.constexpr,
    SS_PARTIALS: tl.constexpr,
    MAX_PARTS: tl.constexpr,
):
    if SS_PARTIALS:
        rcp = _rms_rcp_partials(ss_ptr, 0, n_parts, Q_LORA, BLOCK_M, MAX_PARTS, eps)
    else:
        rcp = _rms_rcp(kva_ptr, stride_kva, rows, rmask, Q_LORA, BLOCK_M, BK, eps)
    cols = tl.arange(0, BK)
    n_offs = pid * BN + tl.arange(0, BN)
    acc = tl.zeros([BLOCK_M, BN], dtype=tl.float32)
    for k0 in tl.range(0, Q_LORA, BK):
        x = tl.load(
            kva_ptr + rows[:, None] * stride_kva + (k0 + cols)[None, :],
            mask=rmask[:, None],
            other=0.0,
        ).to(tl.float32)
        g = tl.load(gq_ptr + k0 + cols).to(tl.float32)
        xn = (x * rcp[:, None] * g[None, :]).to(tl.bfloat16)
        if WRITE_QR:
            if pid == 0:
                tl.store(
                    qr_ptr + rows[:, None] * Q_LORA + (k0 + cols)[None, :], xn, mask=rmask[:, None]
                )
        w = tl.load(wqb_ptr + n_offs[None, :] * Q_LORA + (k0 + cols)[:, None])
        acc = tl.dot(xn, w, acc)
    tl.store(
        q_ptr + rows[:, None] * (H * HEAD_DIM) + n_offs[None, :],
        acc.to(tl.bfloat16),
        mask=rmask[:, None],
    )


@triton.jit
def _absorb_program(
    a,
    wuk_ptr,
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
    USE_PDL: tl.constexpr,
    DELAY_NS: tl.constexpr = 0,
):
    chunks_per_head: tl.constexpr = KV_LORA // BC
    h = a // chunks_per_head
    c0 = (a % chunks_per_head) * BC
    cc = tl.arange(0, BC)
    d_lo = tl.arange(0, 128)
    d_hi = tl.arange(0, NOPE - 128)
    # W_UK[h] is k_b_proj_trans[h] = [KV_LORA, NOPE]; load the [NOPE, BC]
    # slice for this program before waiting so it overlaps the q_b GEMM.
    w_base = wuk_ptr + h * (KV_LORA * NOPE) + (c0 + cc)[None, :] * NOPE
    w_lo = tl.load(w_base + d_lo[:, None])
    w_hi = tl.load(w_base + 128 + d_hi[:, None])
    # Weights are never written by an earlier grid, so only the counter and
    # buffer accesses below have to wait for the programmatic dependency.
    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    ready = tl.atomic_add(sync_ptr + h, 0, sem="acquire", scope="gpu")
    while ready < NOPE // BN:
        ready = tl.atomic_add(sync_ptr + h, 0, sem="acquire", scope="gpu")
    tl.debug_barrier()

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

    # Every program of head h has read the q tiles once this count reaches
    # chunks_per_head; the last one resets both counters for the next launch.
    _stress_delay(a, 3, DELAY_NS)
    tl.debug_barrier()
    done = tl.atomic_add(sync_ptr + H + h, 1, sem="acq_rel", scope="gpu")
    if done == chunks_per_head - 1:
        tl.atomic_xchg(sync_ptr + h, 0, sem="relaxed", scope="gpu")
        tl.atomic_xchg(sync_ptr + H + h, 0, sem="relaxed", scope="gpu")


@triton.jit
def _kv_program(
    kva_ptr,
    stride_kva,
    gkv_ptr,
    latent_ptr,
    ss_ptr,
    n_parts,
    rows,
    rmask,
    eps,
    Q_LORA: tl.constexpr,
    KV_LORA: tl.constexpr,
    ROPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SS_PARTIALS: tl.constexpr,
    MAX_PARTS: tl.constexpr,
):
    if SS_PARTIALS:
        # Experimental own hidden GEMV only (not bit-exact: its partial sums
        # follow the GEMV's column split).
        kc = tl.arange(0, KV_LORA)
        x = tl.load(
            kva_ptr + rows[:, None] * stride_kva + Q_LORA + kc[None, :],
            mask=rmask[:, None],
            other=0.0,
        ).to(tl.float32)
        rcp = _rms_rcp_partials(ss_ptr, 1, n_parts, KV_LORA, BLOCK_M, MAX_PARTS, eps)
        g = tl.load(gkv_ptr + kc).to(tl.float32)
        y = (x * rcp[:, None] * g[None, :]).to(tl.bfloat16)
        tl.store(
            latent_ptr + rows[:, None] * (KV_LORA + ROPE) + kc[None, :], y, mask=rmask[:, None]
        )
    else:
        # The native kv_a_layernorm, bit for bit.
        _fi_rmsnorm_rows_exact(
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
        )
    rc = tl.arange(0, ROPE)
    k_pe = tl.load(
        kva_ptr + rows[:, None] * stride_kva + Q_LORA + KV_LORA + rc[None, :],
        mask=rmask[:, None],
        other=0.0,
    )
    tl.store(
        latent_ptr + rows[:, None] * (KV_LORA + ROPE) + KV_LORA + rc[None, :],
        k_pe,
        mask=rmask[:, None],
    )


@triton.jit
def qproj_kernel(
    kva_ptr,
    stride_kva,
    gq_ptr,
    gkv_ptr,
    wqb_ptr,
    wuk_ptr,
    q_ptr,
    qr_ptr,
    latent_ptr,
    fq_ptr,
    stride_fq_row,
    sync_ptr,
    ss_ptr,
    n_parts,
    R,
    eps_q,
    eps_kv,
    Q_LORA: tl.constexpr,
    KV_LORA: tl.constexpr,
    ROPE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,
    N_QB: tl.constexpr,
    N_ABS: tl.constexpr,
    WRITE_QR: tl.constexpr,
    DO_ABSORB: tl.constexpr,
    SS_PARTIALS: tl.constexpr,
    MAX_PARTS: tl.constexpr,
    USE_PDL: tl.constexpr,
    PRENORM: tl.constexpr = False,
    qn_ptr=None,
    stride_qn=0,
    ABSORB_IN_PRODUCER: tl.constexpr = False,
    PRODUCER_JOB_SHIFT: tl.constexpr = 0,
    ABS_JOB0: tl.constexpr = 0,
    DELAY_NS: tl.constexpr = 0,
):
    """Programs: ``[0, N_QB)`` q_b producers (with ``ABSORB_IN_PRODUCER``,
    producer ``p >= PRODUCER_JOB_SHIFT`` then runs absorption job
    ``p - PRODUCER_JOB_SHIFT``); ``[N_QB, N_QB + N_ABS)`` absorption jobs
    ``ABS_JOB0 + i``; the last program normalizes ``compressed_kv``."""
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    rmask = rows < R
    if pid < N_QB:
        if ABSORB_IN_PRODUCER:
            job = tl.maximum(pid - PRODUCER_JOB_SHIFT, 0)
            w_lo, w_hi = _ld_wuk(job, pid >= PRODUCER_JOB_SHIFT, wuk_ptr, KV_LORA, NOPE, BC)
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
        if PRENORM:
            _qb_program(
                pid,
                qn_ptr,
                stride_qn,
                gq_ptr,
                wqb_ptr,
                q_ptr,
                qr_ptr,
                sync_ptr,
                ss_ptr,
                n_parts,
                rows,
                rmask,
                eps_q,
                Q_LORA,
                HEAD_DIM,
                NOPE,
                H,
                BLOCK_M,
                BN,
                BK,
                False,
                DO_ABSORB,
                False,
                MAX_PARTS,
                True,
                DELAY_NS,
            )
        else:
            _qb_program(
                pid,
                kva_ptr,
                stride_kva,
                gq_ptr,
                wqb_ptr,
                q_ptr,
                qr_ptr,
                sync_ptr,
                ss_ptr,
                n_parts,
                rows,
                rmask,
                eps_q,
                Q_LORA,
                HEAD_DIM,
                NOPE,
                H,
                BLOCK_M,
                BN,
                BK,
                WRITE_QR,
                DO_ABSORB,
                SS_PARTIALS,
                MAX_PARTS,
                False,
                DELAY_NS,
            )
        if ABSORB_IN_PRODUCER:
            if pid >= PRODUCER_JOB_SHIFT:
                _absorb_job(
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
                    DELAY_NS,
                )
    elif pid < N_QB + N_ABS:
        _absorb_program(
            ABS_JOB0 + pid - N_QB,
            wuk_ptr,
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
            USE_PDL,
            DELAY_NS,
        )
    else:
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
        _kv_program(
            kva_ptr,
            stride_kva,
            gkv_ptr,
            latent_ptr,
            ss_ptr,
            n_parts,
            rows,
            rmask,
            eps_kv,
            Q_LORA,
            KV_LORA,
            ROPE,
            BLOCK_M,
            SS_PARTIALS,
            MAX_PARTS,
        )


@triton.jit
def _gemv_cols(
    x_ptr,
    stride_x,
    w_ptr,
    out_ptr,
    stride_o,
    ss_ptr,
    R,
    col0,
    n_cols,
    SS_SPLIT: tl.constexpr,
    SS_END: tl.constexpr,
    WRITE_SS: tl.constexpr,
    CPP: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)
    rmask = rows < R
    lane = tl.arange(0, BN)
    cols = col0 + lane
    cmask = (lane < CPP) & (cols < n_cols)
    kk = tl.arange(0, BK)
    acc = tl.zeros([BLOCK_M, BN], dtype=tl.float32)
    for k0 in tl.range(0, K, BK):
        x = tl.load(
            x_ptr + rows[:, None] * stride_x + (k0 + kk)[None, :], mask=rmask[:, None], other=0.0
        )
        w = tl.load(w_ptr + cols[None, :] * K + (k0 + kk)[:, None], mask=cmask[None, :], other=0.0)
        acc = tl.dot(x, w, acc)
    out = acc.to(tl.bfloat16)
    tl.store(
        out_ptr + rows[:, None] * stride_o + cols[None, :],
        out,
        mask=rmask[:, None] & cmask[None, :],
    )
    if WRITE_SS:
        # Per-program partial sums of squares of the BF16 outputs over the
        # RMSNorm groups [0, SS_SPLIT) and [SS_SPLIT, SS_END); the query
        # kernel reduces them instead of re-reading the rows.
        sq = out.to(tl.float32)
        sq = sq * sq
        in_a = cmask & (cols < SS_SPLIT)
        in_b = cmask & (cols >= SS_SPLIT) & (cols < SS_END)
        ss_a = tl.sum(tl.where(in_a[None, :], sq, 0.0), axis=1)
        ss_b = tl.sum(tl.where(in_b[None, :], sq, 0.0), axis=1)
        pid = tl.program_id(0)
        tl.store(ss_ptr + pid * 2 * BLOCK_M + rows, ss_a)
        tl.store(ss_ptr + pid * 2 * BLOCK_M + BLOCK_M + rows, ss_b)


@triton.jit
def hidden_gemv_kernel(
    x_ptr,
    stride_x,
    w0_ptr,
    w1_ptr,
    out0_ptr,
    stride_o0,
    out1_ptr,
    stride_o1,
    ss_ptr,
    R,
    N0,
    N1,
    P0: tl.constexpr,
    CPP0: tl.constexpr,
    CPP1: tl.constexpr,
    SS_SPLIT: tl.constexpr,
    SS_END: tl.constexpr,
    WRITE_SS: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """``out0 = x @ W0^T`` and ``out1 = x @ W1^T`` for ``R <= BLOCK_M`` rows.

    ``W0 [N0, K]`` (the native fused ``kv_a_proj_with_mqa`` weight) and the
    optional ``W1 [N1, K]`` (the indexer's fused ``wk``/``weights_proj``
    weight) share the input. Programs ``[0, P0)`` own ``CPP0`` consecutive
    columns of ``W0``, the rest ``CPP1`` columns of ``W1`` (``BN`` is the
    next power of two), each over the full ``K``, so the work is split
    evenly over about one wave. FP32 accumulation, BF16 outputs. With
    ``WRITE_SS``, ``W0`` programs also emit RMSNorm partial sums (see
    ``_gemv_cols``) into ``ss_ptr [P0, 2, BLOCK_M]``.
    """
    pid = tl.program_id(0)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    if pid < P0:
        _gemv_cols(
            x_ptr,
            stride_x,
            w0_ptr,
            out0_ptr,
            stride_o0,
            ss_ptr,
            R,
            pid * CPP0,
            N0,
            SS_SPLIT,
            SS_END,
            WRITE_SS,
            CPP0,
            K,
            BLOCK_M,
            BN,
            BK,
        )
    else:
        _gemv_cols(
            x_ptr,
            stride_x,
            w1_ptr,
            out1_ptr,
            stride_o1,
            ss_ptr,
            R,
            (pid - P0) * CPP1,
            N1,
            0,
            0,
            False,
            CPP1,
            K,
            BLOCK_M,
            BN,
            BK,
        )


# --------------------------------------------------------------------------
# Full-indexer layers: the same query kernel plus the token-wise indexer
# projection (the index_selection contract, token-wise part), reproducing the native
# ``Indexer.pre_indexer_proj`` chain for BF16 projections and FP8 keys bit
# for bit:
#   q = wq_b(qr); rope(q[:, :, :64]); fused_cat_fp8(q)          (per head)
#   k = k_norm(wk(hidden)); rope(k[:, :64]); fused_cat_fp8(k)
#   weights = weights_proj(hidden) * q_scale * softmax_scale * n_heads^-0.5
# The fused wk/weights_proj GEMM and ``k_norm`` (LayerNorm under
# torch.compile) run natively as bridges; the kernel consumes their outputs.
# --------------------------------------------------------------------------


@triton.jit
def _iq_program(
    t,
    kva_ptr,
    stride_kva,
    gq_ptr,
    wiq_ptr,
    iq_ptr,
    isync_ptr,
    ss_ptr,
    n_parts,
    rows,
    rmask,
    eps,
    Q_LORA: tl.constexpr,
    IHD: tl.constexpr,
    IH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SS_PARTIALS: tl.constexpr,
    MAX_PARTS: tl.constexpr,
    PRENORM: tl.constexpr,
    DELAY_NS: tl.constexpr = 0,
):
    if PRENORM:
        _prenorm_gemm(
            t, kva_ptr, stride_kva, wiq_ptr, iq_ptr, IH * IHD, rows, rmask, Q_LORA, BLOCK_M, BN, BK
        )
    else:
        _iq_norm_gemm(
            t,
            kva_ptr,
            stride_kva,
            gq_ptr,
            wiq_ptr,
            iq_ptr,
            ss_ptr,
            n_parts,
            rows,
            rmask,
            eps,
            Q_LORA,
            IHD,
            IH,
            BLOCK_M,
            BN,
            BK,
            SS_PARTIALS,
            MAX_PARTS,
        )
    _stress_delay(t, 5, DELAY_NS)
    tl.debug_barrier()
    tl.atomic_add(isync_ptr + t // (IHD // BN), 1, sem="release", scope="gpu")


@triton.jit
def _iq_norm_gemm(
    t,
    kva_ptr,
    stride_kva,
    gq_ptr,
    wiq_ptr,
    iq_ptr,
    ss_ptr,
    n_parts,
    rows,
    rmask,
    eps,
    Q_LORA: tl.constexpr,
    IHD: tl.constexpr,
    IH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SS_PARTIALS: tl.constexpr,
    MAX_PARTS: tl.constexpr,
):
    if SS_PARTIALS:
        rcp = _rms_rcp_partials(ss_ptr, 0, n_parts, Q_LORA, BLOCK_M, MAX_PARTS, eps)
    else:
        rcp = _rms_rcp(kva_ptr, stride_kva, rows, rmask, Q_LORA, BLOCK_M, BK, eps)
    cols = tl.arange(0, BK)
    n_offs = t * BN + tl.arange(0, BN)
    acc = tl.zeros([BLOCK_M, BN], dtype=tl.float32)
    for k0 in tl.range(0, Q_LORA, BK):
        x = tl.load(
            kva_ptr + rows[:, None] * stride_kva + (k0 + cols)[None, :],
            mask=rmask[:, None],
            other=0.0,
        ).to(tl.float32)
        g = tl.load(gq_ptr + k0 + cols).to(tl.float32)
        xn = (x * rcp[:, None] * g[None, :]).to(tl.bfloat16)
        w = tl.load(wiq_ptr + n_offs[None, :] * Q_LORA + (k0 + cols)[:, None])
        acc = tl.dot(xn, w, acc)
    tl.store(
        iq_ptr + rows[:, None] * (IH * IHD) + n_offs[None, :],
        acc.to(tl.bfloat16),
        mask=rmask[:, None],
    )


@triton.jit
def _fp8_ue8m0_scale(amax):
    """``2^ceil(log2(max(amax, 1e-12) / 448))`` and its exact inverse, as in
    the native ``fusedCatFp8Kernel<UseUe8m0=true>``."""
    ratio = tl.maximum(amax, 1.0e-12) * (1.0 / 448.0)
    bits = ratio.to(tl.uint32, bitcast=True)
    exp_bits = bits & 0x7F800000
    exp_bits = tl.where((bits & 0x007FFFFF) != 0, exp_bits + 0x00800000, exp_bits)
    scale = exp_bits.to(tl.float32, bitcast=True)
    inv_scale = ((254 << 23) - exp_bits).to(tl.float32, bitcast=True)
    return scale, inv_scale


@triton.jit
def _mul_rn(a, b):
    """FP32 multiply rounded once (never contracted into an FMA)."""
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _fma_rn(a, b, c):
    """``a * b + c`` with a single rounding."""
    return tl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;", "=r,r,r,r", [a, b, c], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _rope_pairs(x_even, x_odd, cos, sin):
    """Interleaved (GPT-J) RoPE on BF16 pairs, FP32 math, BF16 result, with
    the exact arithmetic of the native indexer RoPE (FlashInfer
    ``vec_apply_llama_rope_cos_sin_interleave_reuse_half``: ``x[i] * cos +
    (+-x[i ^ 1]) * sin`` compiled as ``fma(x[i], cos, rn((+-x[i ^ 1]) *
    sin))``), pinned with explicit ``fma.rn`` / ``mul.rn``. Checked bit for
    bit against the native kernel on real and synthetic rows."""
    o_even = _fma_rn(x_even, cos, -_mul_rn(x_odd, sin))
    o_odd = _fma_rn(x_odd, cos, _mul_rn(x_even, sin))
    return (o_even.to(tl.bfloat16).to(tl.float32), o_odd.to(tl.bfloat16).to(tl.float32))


@triton.jit
def _qpost_program(
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
    DELAY_NS: tl.constexpr = 0,
):
    half: tl.constexpr = ROPE_I // 2
    nope: tl.constexpr = IHD - ROPE_I
    j = tl.arange(0, half)
    c = tl.arange(0, nope)
    pos = tl.load(pos_ptr + rows, mask=rmask, other=0).to(tl.int64)
    cs_row = cos_sin_ptr + pos[:, None] * ROPE_I
    cos = tl.load(cs_row + j[None, :], mask=rmask[:, None], other=0.0)
    sin = tl.load(cs_row + half + j[None, :], mask=rmask[:, None], other=0.0)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
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
def _kpost_program(
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
):
    """Indexer key tail on the native ``k_norm`` output ``kln [R, IHD]``
    (the LayerNorm itself stays native: its inductor reduction order
    depends on the compiled configuration): native RoPE arithmetic
    (``_rope_pairs``) and the native ``fusedCatFp8Kernel<UseUe8m0>``
    quantization."""
    half: tl.constexpr = ROPE_I // 2
    nope: tl.constexpr = IHD - ROPE_I
    j = tl.arange(0, half)
    c = tl.arange(0, nope)
    pos = tl.load(pos_ptr + rows, mask=rmask, other=0).to(tl.int64)
    cs_row = cos_sin_ptr + pos[:, None] * ROPE_I
    cos = tl.load(cs_row + j[None, :], mask=rmask[:, None], other=0.0)
    sin = tl.load(cs_row + half + j[None, :], mask=rmask[:, None], other=0.0)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    base = kln_ptr + rows[:, None] * stride_kln
    ye = tl.load(base + 2 * j[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    yo = tl.load(base + 2 * j[None, :] + 1, mask=rmask[:, None], other=0.0).to(tl.float32)
    yn = tl.load(base + ROPE_I + c[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
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
def qproj_indexer_kernel(
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
    ss_ptr,
    n_parts,
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
    SS_PARTIALS: tl.constexpr,
    MAX_PARTS: tl.constexpr,
    USE_PDL: tl.constexpr,
    PRENORM: tl.constexpr = False,
    qn_ptr=None,
    stride_qn=0,
    ABSORB_IN_PRODUCER: tl.constexpr = False,
    PRODUCER_JOB_SHIFT: tl.constexpr = 0,
    ABS_JOB0: tl.constexpr = 0,
    DELAY_NS: tl.constexpr = 0,
):
    """``qproj_kernel`` plus the indexer's token-wise projection.

    Program roles, in program-id order (every consumer after the producers
    it waits on): ``q_b`` tiles (with ``ABSORB_IN_PRODUCER``, producer
    ``p >= PRODUCER_JOB_SHIFT`` then runs absorption job
    ``p - PRODUCER_JOB_SHIFT``), indexer ``wq_b`` tiles (publish per indexer
    head), absorption jobs ``ABS_JOB0 + i``, indexer query post-processing
    (RoPE, FP8, weights; one program per indexer head), ``compressed_kv``
    norm, indexer key post-processing. ``sync_ptr`` holds ``2 * H``
    absorption counters followed by ``IH`` indexer-head counters.
    """
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    rmask = rows < R
    if pid < N_QB:
        if ABSORB_IN_PRODUCER:
            job = tl.maximum(pid - PRODUCER_JOB_SHIFT, 0)
            w_lo, w_hi = _ld_wuk(job, pid >= PRODUCER_JOB_SHIFT, wuk_ptr, KV_LORA, NOPE, BC)
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
        if PRENORM:
            _qb_program(
                pid,
                qn_ptr,
                stride_qn,
                gq_ptr,
                wqb_ptr,
                q_ptr,
                q_ptr,
                sync_ptr,
                ss_ptr,
                n_parts,
                rows,
                rmask,
                eps_q,
                Q_LORA,
                HEAD_DIM,
                NOPE,
                H,
                BLOCK_M,
                BN,
                BK,
                False,
                DO_ABSORB,
                False,
                MAX_PARTS,
                True,
                DELAY_NS,
            )
        else:
            _qb_program(
                pid,
                kva_ptr,
                stride_kva,
                gq_ptr,
                wqb_ptr,
                q_ptr,
                q_ptr,
                sync_ptr,
                ss_ptr,
                n_parts,
                rows,
                rmask,
                eps_q,
                Q_LORA,
                HEAD_DIM,
                NOPE,
                H,
                BLOCK_M,
                BN,
                BK,
                False,
                DO_ABSORB,
                SS_PARTIALS,
                MAX_PARTS,
                False,
                DELAY_NS,
            )
        if ABSORB_IN_PRODUCER:
            if pid >= PRODUCER_JOB_SHIFT:
                _absorb_job(
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
                    DELAY_NS,
                )
    elif pid < N_QB + N_IQ:
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
        if PRENORM:
            _iq_program(
                pid - N_QB,
                qn_ptr,
                stride_qn,
                gq_ptr,
                wiq_ptr,
                iq_ptr,
                sync_ptr + 2 * H,
                ss_ptr,
                n_parts,
                rows,
                rmask,
                eps_q,
                Q_LORA,
                IHD,
                IH,
                BLOCK_M,
                BN,
                BK,
                False,
                MAX_PARTS,
                True,
                DELAY_NS,
            )
        else:
            _iq_program(
                pid - N_QB,
                kva_ptr,
                stride_kva,
                gq_ptr,
                wiq_ptr,
                iq_ptr,
                sync_ptr + 2 * H,
                ss_ptr,
                n_parts,
                rows,
                rmask,
                eps_q,
                Q_LORA,
                IHD,
                IH,
                BLOCK_M,
                BN,
                BK,
                SS_PARTIALS,
                MAX_PARTS,
                False,
                DELAY_NS,
            )
    elif pid < N_QB + N_IQ + N_ABS:
        _absorb_program(
            ABS_JOB0 + pid - N_QB - N_IQ,
            wuk_ptr,
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
            USE_PDL,
            DELAY_NS,
        )
    elif pid < N_QB + N_IQ + N_ABS + IH:
        _qpost_program(
            pid - N_QB - N_IQ - N_ABS,
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
            DELAY_NS,
        )
    elif pid == N_QB + N_IQ + N_ABS + IH:
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
        _kv_program(
            kva_ptr,
            stride_kva,
            gkv_ptr,
            latent_ptr,
            ss_ptr,
            n_parts,
            rows,
            rmask,
            eps_kv,
            Q_LORA,
            KV_LORA,
            ROPE,
            BLOCK_M,
            SS_PARTIALS,
            MAX_PARTS,
        )
    else:
        _kpost_program(
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
        )


# --------------------------------------------------------------------------
# Prefetching variant (shared and full indexer layers).
#
# The cp.async-pipelined producers above are latency bound: each CTA keeps
# only (num_stages - 1) K-tiles in flight and starts its weight stream only
# after the RMSNorm pre-pass. Here every producer issues its whole [K, BN]
# weight tile (and its absorption W_UK slice) as independent loads at
# program start -- before ``griddepcontrol.wait``, so under programmatic
# dependent launch the weight stream overlaps the preceding native kv_a
# GEMM -- then waits, loads the eight q_a row tiles at once, normalizes and
# multiplies.
# Each producer program of head h + 1 also performs one absorption job of
# head h after publishing its own tile (produce-then-consume), and TPH extra
# programs after all producers absorb the last head, so every wait targets
# lower program ids (see the module docstring).
# --------------------------------------------------------------------------


@triton.jit
def _ld_w(w_ptr, n_offs, j: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    k = j * BK + tl.arange(0, BK)
    return tl.load(w_ptr + n_offs[None, :] * K + k[:, None])


@triton.jit
def _ld_x(kva_ptr, stride_kva, rows, rmask, j: tl.constexpr, BK: tl.constexpr):
    k = j * BK + tl.arange(0, BK)
    return tl.load(
        kva_ptr + rows[:, None] * stride_kva + k[None, :], mask=rmask[:, None], other=0.0
    )


@triton.jit
def _sq(x):
    xf = x.to(tl.float32)
    return tl.sum(xf * xf, axis=1)


@triton.jit
def _nx(x, rcp, gq_ptr, j: tl.constexpr, BK: tl.constexpr):
    g = tl.load(gq_ptr + j * BK + tl.arange(0, BK)).to(tl.float32)
    return (x.to(tl.float32) * rcp[:, None] * g[None, :]).to(tl.bfloat16)


@triton.jit
def _prefetch_tile_gemm(
    t,
    w_ptr,
    out_ptr,
    OUT_COLS: tl.constexpr,
    kva_ptr,
    stride_kva,
    gq_ptr,
    qr_ptr,
    rows,
    rmask,
    eps,
    Q_LORA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BN: tl.constexpr,
    WRITE_QR: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """One [R, BN] tile of ``rmsnorm(q_a) @ W[t*BN:(t+1)*BN]^T`` with the
    whole weight tile requested before the grid-dependency wait."""
    BK: tl.constexpr = Q_LORA // 8
    n_offs = t * BN + tl.arange(0, BN)
    w0 = _ld_w(w_ptr, n_offs, 0, Q_LORA, BK)
    w1 = _ld_w(w_ptr, n_offs, 1, Q_LORA, BK)
    w2 = _ld_w(w_ptr, n_offs, 2, Q_LORA, BK)
    w3 = _ld_w(w_ptr, n_offs, 3, Q_LORA, BK)
    w4 = _ld_w(w_ptr, n_offs, 4, Q_LORA, BK)
    w5 = _ld_w(w_ptr, n_offs, 5, Q_LORA, BK)
    w6 = _ld_w(w_ptr, n_offs, 6, Q_LORA, BK)
    w7 = _ld_w(w_ptr, n_offs, 7, Q_LORA, BK)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    x0 = _ld_x(kva_ptr, stride_kva, rows, rmask, 0, BK)
    x1 = _ld_x(kva_ptr, stride_kva, rows, rmask, 1, BK)
    x2 = _ld_x(kva_ptr, stride_kva, rows, rmask, 2, BK)
    x3 = _ld_x(kva_ptr, stride_kva, rows, rmask, 3, BK)
    x4 = _ld_x(kva_ptr, stride_kva, rows, rmask, 4, BK)
    x5 = _ld_x(kva_ptr, stride_kva, rows, rmask, 5, BK)
    x6 = _ld_x(kva_ptr, stride_kva, rows, rmask, 6, BK)
    x7 = _ld_x(kva_ptr, stride_kva, rows, rmask, 7, BK)
    ss = _sq(x0) + _sq(x1) + _sq(x2) + _sq(x3) + _sq(x4) + _sq(x5) + _sq(x6) + _sq(x7)
    rcp = tl.rsqrt(ss / Q_LORA + eps)
    xn0 = _nx(x0, rcp, gq_ptr, 0, BK)
    xn1 = _nx(x1, rcp, gq_ptr, 1, BK)
    xn2 = _nx(x2, rcp, gq_ptr, 2, BK)
    xn3 = _nx(x3, rcp, gq_ptr, 3, BK)
    xn4 = _nx(x4, rcp, gq_ptr, 4, BK)
    xn5 = _nx(x5, rcp, gq_ptr, 5, BK)
    xn6 = _nx(x6, rcp, gq_ptr, 6, BK)
    xn7 = _nx(x7, rcp, gq_ptr, 7, BK)
    if WRITE_QR:
        if t == 0:
            kk = tl.arange(0, BK)
            qr_row = qr_ptr + rows[:, None] * Q_LORA + kk[None, :]
            tl.store(qr_row + 0 * BK, xn0, mask=rmask[:, None])
            tl.store(qr_row + 1 * BK, xn1, mask=rmask[:, None])
            tl.store(qr_row + 2 * BK, xn2, mask=rmask[:, None])
            tl.store(qr_row + 3 * BK, xn3, mask=rmask[:, None])
            tl.store(qr_row + 4 * BK, xn4, mask=rmask[:, None])
            tl.store(qr_row + 5 * BK, xn5, mask=rmask[:, None])
            tl.store(qr_row + 6 * BK, xn6, mask=rmask[:, None])
            tl.store(qr_row + 7 * BK, xn7, mask=rmask[:, None])
    acc = tl.dot(xn0, w0)
    acc = tl.dot(xn1, w1, acc)
    acc = tl.dot(xn2, w2, acc)
    acc = tl.dot(xn3, w3, acc)
    acc = tl.dot(xn4, w4, acc)
    acc = tl.dot(xn5, w5, acc)
    acc = tl.dot(xn6, w6, acc)
    acc = tl.dot(xn7, w7, acc)
    tl.store(
        out_ptr + rows[:, None] * OUT_COLS + n_offs[None, :],
        acc.to(tl.bfloat16),
        mask=rmask[:, None],
    )


@triton.jit
def _absorb_job(
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
    DELAY_NS: tl.constexpr = 0,
):
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
def qproj_prefetch_kernel(
    kva_ptr,
    stride_kva,
    gq_ptr,
    gkv_ptr,
    wqb_ptr,
    wuk_ptr,
    q_ptr,
    qr_ptr,
    latent_ptr,
    fq_ptr,
    stride_fq_row,
    sync_ptr,
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
    BC: tl.constexpr,
    N_QB: tl.constexpr,
    N_IQ: tl.constexpr,
    WRITE_QR: tl.constexpr,
    DO_ABSORB: tl.constexpr,
    FUSE_INDEXER: tl.constexpr,
    USE_PDL: tl.constexpr,
    N_ABS: tl.constexpr = 0,
    PRODUCER_JOB_SHIFT: tl.constexpr = 0,
    ABS_JOB0: tl.constexpr = 0,
    DELAY_NS: tl.constexpr = 0,
):
    """Prefetching query kernel. Program roles in program-id order:

    * ``[0, N_QB)``: ``q_b`` tile ``pid`` (prefetched), then (``DO_ABSORB``,
      ``pid >= PRODUCER_JOB_SHIFT``) absorption job ``pid -
      PRODUCER_JOB_SHIFT`` (``N_QB == H * KV_LORA // BC``; its ``W_UK`` slice
      is also prefetched before the wait);
    * ``[N_QB, N_QB + N_IQ)`` (``FUSE_INDEXER``): indexer ``wq_b`` tiles;
    * ``N_ABS`` absorption jobs ``ABS_JOB0 + i``;
    * ``IH`` indexer-query post-processing programs (``FUSE_INDEXER``);
    * the ``compressed_kv`` norm / latent program;
    * the indexer-key program (``FUSE_INDEXER``).
    """
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    rmask = rows < R
    if pid < N_QB:
        if DO_ABSORB:
            job = tl.maximum(pid - PRODUCER_JOB_SHIFT, 0)
            w_lo, w_hi = _ld_wuk(job, pid >= PRODUCER_JOB_SHIFT, wuk_ptr, KV_LORA, NOPE, BC)
        _prefetch_tile_gemm(
            pid,
            wqb_ptr,
            q_ptr,
            H * HEAD_DIM,
            kva_ptr,
            stride_kva,
            gq_ptr,
            qr_ptr,
            rows,
            rmask,
            eps_q,
            Q_LORA,
            BLOCK_M,
            BN,
            WRITE_QR,
            USE_PDL,
        )
        if DO_ABSORB:
            if (pid % (HEAD_DIM // BN)) < (NOPE // BN):
                _stress_delay(pid, 1, DELAY_NS)
                tl.debug_barrier()
                tl.atomic_add(sync_ptr + pid // (HEAD_DIM // BN), 1, sem="release", scope="gpu")
            if pid >= PRODUCER_JOB_SHIFT:
                _absorb_job(
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
                    DELAY_NS,
                )
    elif pid < N_QB + N_IQ:
        t = pid - N_QB
        _prefetch_tile_gemm(
            t,
            wiq_ptr,
            iq_ptr,
            IH * IHD,
            kva_ptr,
            stride_kva,
            gq_ptr,
            qr_ptr,
            rows,
            rmask,
            eps_q,
            Q_LORA,
            BLOCK_M,
            BN,
            False,
            USE_PDL,
        )
        _stress_delay(t, 5, DELAY_NS)
        tl.debug_barrier()
        tl.atomic_add(sync_ptr + 2 * H + t // (IHD // BN), 1, sem="release", scope="gpu")
    elif pid < N_QB + N_IQ + N_ABS:
        _absorb_program(
            ABS_JOB0 + pid - N_QB - N_IQ,
            wuk_ptr,
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
            USE_PDL,
            DELAY_NS,
        )
    elif pid < N_QB + N_IQ + N_ABS + IH:
        _qpost_program(
            pid - N_QB - N_IQ - N_ABS,
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
            DELAY_NS,
        )
    elif pid == N_QB + N_IQ + N_ABS + IH:
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
        _kv_program(
            kva_ptr,
            stride_kva,
            gkv_ptr,
            latent_ptr,
            sync_ptr,
            0,
            rows,
            rmask,
            eps_kv,
            Q_LORA,
            KV_LORA,
            ROPE,
            BLOCK_M,
            False,
            16,
        )
    else:
        _kpost_program(
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
        )
