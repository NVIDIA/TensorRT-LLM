# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
"""Transports the column-sharded pool and unpool GEMMs publish into on 33..512-row decode steps (TP4).

From 33 rows the built-in MoE multiplies the whole replicated pool (8192 -> 4096) and
unpool (4096 -> 8192) weights on every rank: 67 MB per projection per rank. Here each
rank multiplies only its column slice (16.8 MB: pool ``[R, 8192] x [1024, 8192]^T``,
unpool ``[R, 4096] x [2048, 4096]^T``) and the GEMM's output tensor *is* this rank's slot
of a multicast transport, so the GEMM's own epilogue stores publish the slice to every
rank. The consumers -- the pre-expert RMSNorm + NVFP4 quantization
(``pool_publish_fp4quant``) and the post-FFN join (``unpool_publish_join`` below) --
poll their rows of their own copy, reset them and compute the built-in kernels'
arithmetic on the gathered row.

Transport (Lamport, payload as flag). Every rank holds ``BUFFERS x TP source ranks x
max_rows x columns`` BF16 -- rank-major, so rank ``r``'s slot of buffer ``b`` is one
contiguous ``[rows, columns]`` block, a GEMM output as it is -- filled with the sentinel
word ``0xFFFEFFFE``: two BF16 NaNs with payload ``0x7E``. Float-to-BF16 rounding
(``cvt.rn.bf16{x2}.f32``, the GEMM epilogues') turns every NaN into the canonical
``0x7FFF``, so no GEMM output holds the half ``0xFFFE``; a 16-bit half that still holds
it has not arrived. The test is per 16-bit half, so a store that reached the poller in
pieces reads as not arrived until every piece did. Own slot included: the producer's
multicast stores reach the local copy asynchronously, like the peers'.

Ownership and reuse (cuFlow):

* One writer per element per call: rank ``r``'s slice GEMM (``publish_projection``,
  cuBLAS through ``torch.mm`` with the slot as its output) stores each element of slot
  ``r`` once, from its epilogue, with its final value. cuBLAS picks a beta-zero kernel
  (no read-back of the output); where it splits K it reduces the partial products from
  its workspace in a separate reduction kernel, whose single store is the publication.
  The census test drives this with a concurrent poller at every served row count;
  the census holds for the cuBLAS it ran on, so the path publishes only with that
  cuBLAS, torch and device (``producer_refusal``).
* One reader and resetter per element per call: the consumer thread of each rank that
  owns that element of its row, on its own copy.
* Reuse: no two consecutive calls on a transport use the same buffer, the last call of a
  step and the first of the next included (``buffer_sequence``: the buffers alternate
  call by call and the last call of an odd cycle takes the third). Rank ``A``'s producer
  of call ``m`` stores only after ``A``'s call ``m - 1`` completed (stream order), which
  needed rank ``B``'s call-``(m - 1)`` slot, which ``B`` stored only after every earlier
  call of ``B`` -- among them ``B``'s last call on that buffer, resets included --
  completed.
* Progress: producers wait on nothing remote; a consumer thread waits only for its own
  elements of the call. A consumer is launched in stream order after its producer, so no
  poller of a call is ever in flight on a rank while that rank still has to launch a
  producer the poll depends on (no co-residency assumption).
* Poll guard: a poll that has seen no arrival for its guard sets a guard word in pinned
  host memory and keeps polling; it never consumes a half that has not arrived. A late
  peer still completes the call exactly; a slot that never arrives leaves the rank
  waiting (the host watch logs the rank, the executor's hang detector is the backstop).
  ``guard_break``, which stops the poll, exists for tests of a broken call sequence only.

Calls on one transport must be issued in the same order on every rank, with the same
row counts and buffers. The transports are this path's alone: the 8-row and 32-row
inboxes and every other collective use their own memory.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from typing import Optional, Sequence, Tuple

import torch
from cuda.bindings import runtime as _cudart

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ...sm107_defaults import env as _sm107_env
from .unpool_direct_boundary import buffer_sequence, check_buffer_sequence

__all__ = [
    "BUFFERS",
    "MAX_ROWS",
    "MIN_ROWS",
    "POOL_HIDDEN",
    "PublishTransport",
    "UNPOOL_HIDDEN",
    "WORLD",
    "WORLDS",
    "buffer_sequence",
    "check_buffer_sequence",
    "local_columns",
    "producer_refusal",
    "publish_projection",
    "unpool_publish_join",
]

#: The TP size the transports connect: one GB300 tray.
WORLD = 4
#: The TP sizes a transport serves: one tray, or (two-tray TP8) two trays over MNNVL, whose 33..64-row
#: decode steps (5..8 requests) multiply the whole replicated pool and unpool weights on every rank too.
WORLDS = (4, 8)
#: The decode rows the transports serve: five to 64 requests at K = 7 (and the counts between).
MIN_ROWS, MAX_ROWS = 33, 512
#: Width of the pooled (expert) activation and of the residual stream.
POOL_HIDDEN, UNPOOL_HIDDEN = 4096, 8192
#: Buffers of a transport (``buffer_sequence``).
BUFFERS = 3
#: The empty-slot word: two BF16 NaNs with the payload no float-to-BF16 rounding produces.
SENTINEL = 0xFFFEFFFE
SENTINEL_HALF = 0xFFFE
_SENTINEL_I16 = SENTINEL_HALF - (1 << 16)
#: Per-rank control words in pinned host memory: ``GUARD_WORD`` is set by a poll that
#: outlived its guard.
CONTROL_WORDS = 4
GUARD_WORD = 0
#: Default poll guard of the CuTe consumer in ns, far above any wait the protocol produces.
GUARD_NS = 1 << 33
#: Default poll guard of the Triton consumer in polling rounds (each round reloads the row;
#: far above any wait the protocol produces).
GUARD_ROUNDS = 1 << 24
#: Seconds between two reads of the guard word by the watch thread.
GUARD_WATCH_S = 1.0
#: Seconds between two warnings of the watch thread while polls keep outliving their guard.
GUARD_RELOG_S = 60.0
#: The producers the census certified: at every served row count of both sites, cuBLAS
#: (``torch.mm`` into the slot) picks beta-zero kernels that store each element once, and
#: splits K only through a separate reduction kernel. Keyed by (torch version, cuBLASLt
#: version, SM count): cuBLAS's kernel choice depends on all three. 152 SMs is GB300 (sm103),
#: 212 SMs Rubin (sm107).
CERTIFIED_PRODUCERS = frozenset(
    {
        ("2.14.0a0+b2c75dd062.nv26.09", 130800, 152),
        ("2.14.0a0+b2c75dd062.nv26.09", 130800, 212),
    }
)
#: The TP8 slice shapes (pool ``[R, 8192] x [512, 8192]^T``, unpool ``[R, 4096] x [1024, 4096]^T``) are other
#: GEMMs than the TP4 census drove, so TP8 publishes only with a producer the TP8 census
#: (a device census, 33..448 rows, both sites) certified
#: (two-tray TP8). Same key as above.
CERTIFIED_PRODUCERS_TP8 = frozenset({("2.14.0a0+b2c75dd062.nv26.09", 130800, 212)})
#: The GB300 (152-SM) TP8 cuBLAS producer census (pool / unpool slices, 33..448 rows, eager module loading).
#: Consulted only by ``MUSESPARK_OPEN_SM103_TEP8_PUBLISH_WIDE
#: =cublas`` (kernels/projection_publish), so the default sm103 TP8 build keeps refusing exactly as before.
CERTIFIED_PRODUCERS_TP8_SM103_WIDE = frozenset({("2.14.0a0+b2c75dd062.nv26.09", 130800, 152)})
#: ``MUSESPARK_OPEN_SM103_TEP8_PUBLISH_WIDE=splitk1``: split_k 1 (no K split: every element one full-K FP32
#: accumulation in K order, rounded once) at the certified tiles with the most stages that fit. On GB300 its slices are
#: bitwise equal to the replicated full-weight cuBLAS GEMM the built-in runs at 64..128 rows (no split-K): device
#: probe, every multiple of 8 and odd rows in 64..128, both sites, 0 differing elements.
TP8_PUBLISH_SPLITK1_TACTICS = {
    (UNPOOL_HIDDEN, POOL_HIDDEN // 8): (64, 16, 1, 11),  # pool
    (POOL_HIDDEN, UNPOOL_HIDDEN // 8): (64, 32, 1, 9),  # unpool
}
#: Environment variables that resize torch's cuBLAS workspaces, which changes cuBLAS's choice
#: of split-K kernels; the census ran with none set.
CUBLAS_WORKSPACE_ENV = ("CUBLAS_WORKSPACE_CONFIG", "CUBLASLT_WORKSPACE_SIZE")

#: Opt-in: ``MUSESPARK_OPEN_TEP8_PUBLISH_SPLITK=1`` publishes the TP8 pool and unpool slices of 33..64-row
#: decode steps (5..8 requests at K = 7) from the split-K kernel (``ops/gemm/splitk_gemm.py``, flashinfer's
#: ``dense_bf16_gemm_sm100_splitk`` moved into this target) instead of cuBLAS. The kernel reduces its K slices
#: in-kernel (cluster DSMEM) and only its owner CTA stores, each element of the slot once with its final value, from
#: registers (no read-back, no partials in the slot: the census property), so a publication is one launch instead of
#: cuBLAS's nvjet + splitKreduce, and the weight is staged before the dependency wait. split_k 4 is the 8-row cells'
#: (``dense_gemms`` SM107_KERNELS), so every row keeps its one-request bits; against cuBLAS one BF16 ulp at most
#: (99.87-99.96 % of elements equal). Needs the kernel's 64-row tiles (``MUSESPARK_OPEN_TEP8_WIDE64_SPLITK=1``).
#: Publishes only with the (SM count, pool producer key, unpool producer key) that the split-K producer census certified
#: (``CERTIFIED_SPLITK_PRODUCERS_TP8``: a device census, every row count 33..64 of both
#: sites, two trials each); a producer key is ``splitk_gemm.producer_key``: the kernel's device code and what the
#: publication compiles with under this process's env, so host-side edits of the kernel file keep the certification and
#: a kernel change drops it. Otherwise, and with the env unset, cuBLAS as before. Rank-uniform: every rank reads the
#: same env, files and device type.
TEP8_PUBLISH_SPLITK_ENV = "MUSESPARK_OPEN_TEP8_PUBLISH_SPLITK"
#: Read once, at import (as the other gates).
TEP8_PUBLISH_SPLITK = _sm107_env(TEP8_PUBLISH_SPLITK_ENV, "0") == "1"
#: Rows the split-K producer serves (the kernel's M <= 64 under the wide64 gate); other row counts keep cuBLAS.
SPLITK_MIN_ROWS, SPLITK_MAX_ROWS = 33, 64
#: (K, columns) of the TP8 slice -> (mma_m, mma_n, split_k, ab_stages): the fastest split-K-4 tile at 40..64 rows
#: behind an early-releasing predecessor.
TP8_PUBLISH_SPLITK_TACTICS = {
    (UNPOOL_HIDDEN, POOL_HIDDEN // 8): (64, 16, 4, 10),  # pool [R, 8192] x [512, 8192]^T
    (POOL_HIDDEN, UNPOOL_HIDDEN // 8): (64, 32, 4, 8),  # unpool [R, 4096] x [1024, 4096]^T
}
#: (SM count, pool producer key, unpool producer key). The census first ran keyed on the split-K kernel
#: file (with and without MUSESPARK_OPEN_SPLITK_OVERSIZED_SMEM; 128 checks, 0 failures), both with these tactics at
#: 212 SMs. The entry below is that census
#: carried over to ``ops/gemm/splitk_gemm.py``: under the F11 env (MUSESPARK_OPEN_SPLITK_RS unset) that kernel traces
#: the same code for these two bias-free tactics (its reduce-scatter branches are compile-time out) at the same
#: launch (opt-in shared memory, mailbox v4, owner store, PDL). A census rerun on the unified file confirms it.
#: The same producer keys on GB300 (152 SMs, sm_103 under MUSESPARK_OPEN_SM103_TEP8_F11=1, whose module
#: defaults give the same mailbox / opt-in / PDL launch): census on device, every row count 33..64 of both sites,
#: concurrent poller + sentinel read-back check, and each 8-row block bitwise equal to sm_103's TP8 8-row pool /
#: unpool cell (split-K 4). Served there through
#: kernels/projection_publish's MUSESPARK_OPEN_SM103_TEP8_PUBLISH_SPLITK_ONLY (the TP8 cuBLAS census is Rubin's).
CERTIFIED_SPLITK_PRODUCERS_TP8 = frozenset(
    {
        (
            212,
            ("95e7ff00cb4507f61ead2ef332c6e22c", (64, 16, 4, 10), "owner", "v4", "opt-in", True),
            ("95e7ff00cb4507f61ead2ef332c6e22c", (64, 32, 4, 8), "owner", "v4", "opt-in", True),
        ),
        (
            152,
            ("95e7ff00cb4507f61ead2ef332c6e22c", (64, 16, 4, 10), "owner", "v4", "opt-in", True),
            ("95e7ff00cb4507f61ead2ef332c6e22c", (64, 32, 4, 8), "owner", "v4", "opt-in", True),
        ),
        # The same census, carried over to the reformatted kernel file: the pre-commit pass (ruff format) re-wrapped
        # lines inside splitk_gemm.py's device-code region, which changed its line md5 (95e7ff00... -> 534e6f63...)
        # and dropped the certification, so TP8 decode steps of 33..448 rows stopped publishing. The region's AST is
        # unchanged (ast.dump md5 30bd80a0a5c20168c7c143158b0d631c before and after), so the kernel is the certified
        # one; the sidestream gate also checks its outputs bitwise against the pre-reformat file's on device.
        (
            212,
            ("534e6f639ccb0b27b855884d278a2883", (64, 16, 4, 10), "owner", "v4", "opt-in", True),
            ("534e6f639ccb0b27b855884d278a2883", (64, 32, 4, 8), "owner", "v4", "opt-in", True),
        ),
        (
            152,
            ("534e6f639ccb0b27b855884d278a2883", (64, 16, 4, 10), "owner", "v4", "opt-in", True),
            ("534e6f639ccb0b27b855884d278a2883", (64, 32, 4, 8), "owner", "v4", "opt-in", True),
        ),
    }
)
#: Whether the kernel launches with programmatic dependent launch (the model's ``dispatch.LAUNCH_PDL``): it reads only
#: the weight (a parameter) before its dependency wait, and stores into the slot only after it.
_SPLITK_PDL = True
_splitk_state: dict = {}


def _splitk_site_key(splitk, tactic: Tuple[int, int, int, int]):
    """The producer key of one site's tactic, the same at every served row count, else None."""
    keys = {
        splitk.producer_key(rows, splitk.SplitKTactic(*tactic), _SPLITK_PDL)
        for rows in (SPLITK_MIN_ROWS, SPLITK_MAX_ROWS)
    }
    return keys.pop() if len(keys) == 1 else None


def _splitk_publisher():
    """``(run_splitk_dense, SplitKTactic)`` when the split-K producer is selected and certified here, else None.

    Decided once per process and logged once; every rank decides alike (same env, files and device type).
    """
    if "publisher" in _splitk_state:
        return _splitk_state["publisher"]
    publisher, reason = None, None
    if not TEP8_PUBLISH_SPLITK:
        _splitk_state["publisher"] = None
        return None
    try:
        from ..gemm import splitk_gemm as splitk

        enabled = getattr(splitk, "_wide64_enabled", None)
        if enabled is None or not enabled():
            reason = "the split-K kernel does not serve 33..64 rows (needs MUSESPARK_OPEN_TEP8_WIDE64_SPLITK=1)"
        else:
            key = (
                torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count,
                _splitk_site_key(
                    splitk, TP8_PUBLISH_SPLITK_TACTICS[(UNPOOL_HIDDEN, POOL_HIDDEN // 8)]
                ),
                _splitk_site_key(
                    splitk, TP8_PUBLISH_SPLITK_TACTICS[(POOL_HIDDEN, UNPOOL_HIDDEN // 8)]
                ),
            )
            if key not in CERTIFIED_SPLITK_PRODUCERS_TP8:
                reason = f"the split-K producer census does not cover {key}"
            else:
                publisher = (splitk.run_splitk_dense, splitk.SplitKTactic)
    except Exception as error:  # noqa: BLE001 - an unusable producer keeps cuBLAS
        reason = f"split-K kernel unavailable ({error!r})"
    if publisher is not None:
        logger.info(
            "[musespark_open] projection_publish: TP8 pool / unpool slices of "
            f"{SPLITK_MIN_ROWS}..{SPLITK_MAX_ROWS}-row steps publish from flashinfer split-K "
            f"({TEP8_PUBLISH_SPLITK_ENV}=1)"
        )
    else:
        logger.warning(
            f"[musespark_open] projection_publish: {TEP8_PUBLISH_SPLITK_ENV}=1 ignored, cuBLAS publishes: {reason}"
        )
    _splitk_state["publisher"] = publisher
    return publisher


#: Opt-in: ``MUSESPARK_OPEN_TEP8_POOL_PROGRAMMATIC_FORK=1`` gives the TP8 pool slice GEMM of a 33..64-row
#: step (on the MoE preparation stream) a programmatic edge from the post-attention hop instead of the full edge of the
#: stream fork. At TEP8 bs4-8 the pool GEMM starts 3.3-4.5 us after the wide hop ends (bs8: 4.35-4.77 us, p10-p90),
#: while the router GEMM -- the hop's programmatic dependent on the main stream -- starts before it ends; the pool chain
#: (GEMM, consumer) then trails the router chain by ~4.3 us per layer and is the join's critical input. With the edge
#: programmatic the hop's trigger launches both dependents (a grid may have two; CUDA 13.4 graphs take a kernel node with
#: two outgoing programmatic edges): the split-K GEMM stages its weight slice before ``griddepcontrol.wait`` and reads
#: the hop's output, and stores into the transport, only after it -- so only launch order and edges change (bitwise).
#: Taken only while capturing, only when the split-K producer (``_splitk_tactic``) publishes the call (a kernel that
#: waits on its prerequisite grid before reading), and only when the stream fork's source was exactly one kernel node.
TEP8_POOL_PROGRAMMATIC_ENV = "MUSESPARK_OPEN_TEP8_POOL_PROGRAMMATIC_FORK"
TEP8_POOL_PROGRAMMATIC = _sm107_env(TEP8_POOL_PROGRAMMATIC_ENV, "0") == "1"

#: Opt-in: ``MUSESPARK_OPEN_TEP8_JOIN_PDL=1`` launches the post-FFN join (``UnpoolPublishJoin``,
#: ``_unpool_publish_join_kernel``) with programmatic dependent launch and an entry trigger, when the model launches with
#: PDL (``dispatch.LAUNCH_PDL``). Every program first waits for its prerequisite grid (``griddepcontrol.wait``, before
#: any read: the residual, the gains and the poll of the transport all follow it), then releases its dependents
#: (``griddepcontrol.launch_dependents``). The next layer's QKV GEMM -- flashinfer's split-K, a PDL launch that stages
#: its weight slice before its own dependency wait and reads the join's output only after it -- then becomes resident
#: on the SMs the join's programs leave and streams its weight while the join waits for the exchange; it can read
#: ``next_input`` only after this grid completes, and the join's completion still implies its predecessor's (the join
#: waited). Only launch order changes: bitwise. Unset: the plain launch.
TEP8_JOIN_PDL_ENV = "MUSESPARK_OPEN_TEP8_JOIN_PDL"
TEP8_JOIN_PDL = _sm107_env(TEP8_JOIN_PDL_ENV, "0") == "1"


_join_pdl_logged: list = []


def _join_pdl() -> bool:
    """Whether the join launches with PDL: the gate, and the model launching with PDL (logged once)."""
    if not TEP8_JOIN_PDL:
        return False
    try:
        from ...dispatch import LAUNCH_PDL
    except Exception:  # noqa: BLE001 - no model dispatch module: keep the plain launch
        return False
    if not _join_pdl_logged:
        _join_pdl_logged.append(True)
        logger.info(
            f"[musespark_open] projection_publish: {TEP8_JOIN_PDL_ENV}=1: the post-FFN join launches "
            f"{'with PDL and an entry trigger' if LAUNCH_PDL else 'plain (the model launches without PDL)'}"
        )
    return bool(LAUNCH_PDL)


#: Opt-in: ``MUSESPARK_OPEN_PUBLISH_JOIN_CLUSTER=1`` runs the post-FFN join (``UnpoolPublishJoin``) as a
#: cluster kernel (NVRTC, CUDA C++) instead of the Triton program a row: the same 512 threads a row as the Triton join
#: (layout #blocked: Triton thread T owns elements [8T, 8T+8) and [4096+8T, 4096+8T+8)) split over a cluster of
#: CL CTAs (8 up to 128 rows, 4 above; ``=4`` / ``=8`` force one), with the arithmetic of the PTX Triton emits for this
#: kernel at BLOCK_H 8192 and 16 warps, operation for operation: the BF16 gain product, the fma / add in-thread tree,
#: the xor-16..1 warp butterfly, the 16 warp sums reduced in the order every lane of Triton's xor-8..1 cross-warp
#: butterfly computes (shared across the cluster through DSMEM st.async + mbarrier), div.full.f32 by 8192,
#: rsqrt.approx.ftz.f32, fma(residual, alpha, normed * beta), and the FP32-residual variant's mul / add next-variance
#: tree. Bitwise the Triton join on every output (worlds 16 / 8 / 4, rows 33..448, NORM_FP32 0 / 1,
#: gain on, 3 seeds each; the transport polled per thread and reset to the sentinel as the Triton join does: each
#: element still has one reader and one resetter). Launched as the Triton join is launched: with PDL and an entry
#: trigger under ``MUSESPARK_OPEN_TEP8_JOIN_PDL=1``, plain otherwise. Compiled with NVRTC on the first eager
#: call per (device, gain, FP32 residual, cluster); a call inside a capture before that keeps the Triton join (logged
#: once). Unset = unchanged.
JOIN_CLUSTER_ENV = "MUSESPARK_OPEN_PUBLISH_JOIN_CLUSTER"
JOIN_CLUSTER = _sm107_env(JOIN_CLUSTER_ENV, "0").strip()
#: Rows from which the auto cluster size is 4 (lab: CL 8 -2.07 / -1.89 / -1.50 us at 40 / 64 / 128 rows, CL 4 -0.80
#: at 192 where CL 8 ties).
JOIN_CLUSTER_4_FROM_ROWS = 129
_JOIN_CL_FN: dict = {}
_JOIN_CL_LOGGED: list = []

_JOIN_CL_SRC = r"""
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
#ifndef NORM_FP32
#define NORM_FP32 0
#endif
#ifndef HAS_GAIN
#define HAS_GAIN 1
#endif
#ifndef PUB
#define PUB 1
#endif
#ifndef CL
#define CL 4  // CTAs a row; each runs 512 / CL of the Triton join's 512 threads
#endif
#define NT (512 / CL)
#define NW (NT / 32)
__device__ __forceinline__ float bf2f(u32 h) { return __uint_as_float(h << 16); }
__device__ __forceinline__ u32 f2bf(float f) { unsigned short r; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f)); return r; }
__device__ __forceinline__ float rbf(float f) { return bf2f(f2bf(f)); }
__device__ __forceinline__ float div_full(float a, float b) { float r; asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float rsqrt_ftz(float a) { float r; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ void unpack8(uint4 q, float* v) {
  v[0] = bf2f(q.x & 0xffffu); v[1] = bf2f(q.x >> 16); v[2] = bf2f(q.y & 0xffffu); v[3] = bf2f(q.y >> 16);
  v[4] = bf2f(q.z & 0xffffu); v[5] = bf2f(q.z >> 16); v[6] = bf2f(q.w & 0xffffu); v[7] = bf2f(q.w >> 16); }
__device__ __forceinline__ uint4 ld16(const void* p) { return *reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ uint4 ld_vol(const void* p) {
  uint4 q; asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(q.x), "=r"(q.y), "=r"(q.z), "=r"(q.w) : "l"(p) : "memory");
  return q; }
__device__ __forceinline__ bool has_sentinel(uint4 q) {
  u32 w[4] = {q.x, q.y, q.z, q.w};
  bool s = false;
#pragma unroll
  for (int i = 0; i < 4; ++i) s |= ((w[i] & 0xffffu) == 0xfffeu) | ((w[i] >> 16) == 0xfffeu);
  return s; }
// Triton's in-thread order (fma tree: the first variance, and the BF16-first next variance)
__device__ __forceinline__ float thread_sum_fma(const float* x, const float* y) {
  const float ax = __fmaf_rn(x[0], x[0], __fmul_rn(x[2], x[2])), ay = __fmaf_rn(x[1], x[1], __fmul_rn(x[3], x[3]));
  const float bx = __fmaf_rn(x[4], x[4], __fmul_rn(x[6], x[6])), by = __fmaf_rn(x[5], x[5], __fmul_rn(x[7], x[7]));
  const float cx = __fmaf_rn(y[0], y[0], __fmul_rn(y[2], y[2])), cy = __fmaf_rn(y[1], y[1], __fmul_rn(y[3], y[3]));
  const float dx = __fmaf_rn(y[4], y[4], __fmul_rn(y[6], y[6])), dy = __fmaf_rn(y[5], y[5], __fmul_rn(y[7], y[7]));
  return __fadd_rn(__fadd_rn(__fadd_rn(ax, bx), __fadd_rn(cx, dx)), __fadd_rn(__fadd_rn(ay, by), __fadd_rn(cy, dy))); }
// the FP32-residual next variance: squares then an add tree, no fma (the PTX Triton emits there)
__device__ __forceinline__ float thread_sum_mul(const float* x, const float* y) {
  float q[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) { q[i] = __fmul_rn(x[i], x[i]); q[8 + i] = __fmul_rn(y[i], y[i]); }
  const float ax = __fadd_rn(q[0], q[2]), ay = __fadd_rn(q[1], q[3]), bx = __fadd_rn(q[4], q[6]), by = __fadd_rn(q[5], q[7]);
  const float cx = __fadd_rn(q[8], q[10]), cy = __fadd_rn(q[9], q[11]), dx = __fadd_rn(q[12], q[14]), dy = __fadd_rn(q[13], q[15]);
  return __fadd_rn(__fadd_rn(__fadd_rn(ax, bx), __fadd_rn(cx, dx)), __fadd_rn(__fadd_rn(ay, by), __fadd_rn(cy, dy))); }
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
// the row total: warp butterfly, the 16 warp sums to every CTA of the row's cluster (st.async + mbarrier),
// then the tree every lane of Triton's cross-warp xor-8/4/2/1 butterfly computes
__device__ __forceinline__ float row_sum(float s, float* wsum, u64* bar, int rank) {
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, off));
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane < CL) {  // lane q sends this warp's sum to CTA q
    u32 raddr, rbar;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(raddr) : "r"(smem_u32(wsum + NW * rank + warp)), "r"(lane));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rbar) : "r"(smem_u32(bar)), "r"(lane));
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                 :: "r"(raddr), "r"(__float_as_uint(s)), "r"(rbar) : "memory");
  }
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n @!p bra W;\n}"
               :: "r"(smem_u32(bar)) : "memory");
  float W[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) W[i] = wsum[i];
  const float a0 = __fadd_rn(__fadd_rn(W[0], W[8]), __fadd_rn(W[4], W[12]));
  const float a2 = __fadd_rn(__fadd_rn(W[2], W[10]), __fadd_rn(W[6], W[14]));
  const float a1 = __fadd_rn(__fadd_rn(W[1], W[9]), __fadd_rn(W[5], W[13]));
  const float a3 = __fadd_rn(__fadd_rn(W[3], W[11]), __fadd_rn(W[7], W[15]));
  return __fadd_rn(__fadd_rn(a0, a2), __fadd_rn(a1, a3)); }

extern "C" __global__ void __launch_bounds__(NT)
pubjoin4(float* out, const float* residual, const u16* branch, const u16* gain, const float* alpha, const float* beta,
         float eps, u16* next_out, const u16* next_w, float next_eps, int* guard, int buffer, int world, int capacity,
         int columns, long long guard_rounds, int trigger) {
  __shared__ __align__(16) float wsum0[16];
  __shared__ __align__(16) float wsum1[16];
  __shared__ __align__(8) u64 bars[2];
  const int t = threadIdx.x, rank = blockIdx.x % CL, row = blockIdx.x / CL;
  const int T = rank * NT + t;
  const int c0 = 8 * T, c1 = 4096 + 8 * T;
  if (t == 0) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(bars + i)) : "memory");
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 64;" :: "r"(smem_u32(bars + i)) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n\tbarrier.cluster.wait.aligned;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (trigger) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  // every input first (their latencies overlap), the branch last
  const long long rb = (long long)row * 8192;
  uint4 g0q = ld16(gain + c0), g1q = ld16(gain + c1), w0q = ld16(next_w + c0), w1q = ld16(next_w + c1);
  float r0[8], r1[8], a0[8], a1[8], b0[8], b1[8];
  *reinterpret_cast<float4*>(r0) = *reinterpret_cast<const float4*>(residual + rb + c0);
  *reinterpret_cast<float4*>(r0 + 4) = *reinterpret_cast<const float4*>(residual + rb + c0 + 4);
  *reinterpret_cast<float4*>(r1) = *reinterpret_cast<const float4*>(residual + rb + c1);
  *reinterpret_cast<float4*>(r1 + 4) = *reinterpret_cast<const float4*>(residual + rb + c1 + 4);
  *reinterpret_cast<float4*>(a0) = *reinterpret_cast<const float4*>(alpha + c0);
  *reinterpret_cast<float4*>(a0 + 4) = *reinterpret_cast<const float4*>(alpha + c0 + 4);
  *reinterpret_cast<float4*>(a1) = *reinterpret_cast<const float4*>(alpha + c1);
  *reinterpret_cast<float4*>(a1 + 4) = *reinterpret_cast<const float4*>(alpha + c1 + 4);
  *reinterpret_cast<float4*>(b0) = *reinterpret_cast<const float4*>(beta + c0);
  *reinterpret_cast<float4*>(b0 + 4) = *reinterpret_cast<const float4*>(beta + c0 + 4);
  *reinterpret_cast<float4*>(b1) = *reinterpret_cast<const float4*>(beta + c1);
  *reinterpret_cast<float4*>(b1 + 4) = *reinterpret_cast<const float4*>(beta + c1 + 4);
  float x[8], y[8];
#if PUB
  const int s0 = c0 / columns, s1 = c1 / columns;
  u16* box0 = const_cast<u16*>(branch) + ((long long)(buffer * world + s0) * capacity + row) * columns + (c0 - s0 * columns);
  u16* box1 = const_cast<u16*>(branch) + ((long long)(buffer * world + s1) * capacity + row) * columns + (c1 - s1 * columns);
  uint4 q0 = ld_vol(box0), q1 = ld_vol(box1);
  long long rounds = 0;
  while (has_sentinel(q0) | has_sentinel(q1)) {
    q0 = ld_vol(box0); q1 = ld_vol(box1);
    if (guard_rounds > 0 && ++rounds == guard_rounds) atomicExch_system(guard, 1);
  }
  const uint4 sent = make_uint4(0xfffefffeu, 0xfffefffeu, 0xfffefffeu, 0xfffefffeu);
  *reinterpret_cast<uint4*>(box0) = sent;  // retire the consumed halves
  *reinterpret_cast<uint4*>(box1) = sent;
  unpack8(q0, x); unpack8(q1, y);
#else
  unpack8(ld16(branch + rb + c0), x); unpack8(ld16(branch + rb + c1), y);
#endif
#if HAS_GAIN
  {
    float g0[8], g1[8];
    unpack8(g0q, g0); unpack8(g1q, g1);
#pragma unroll
    for (int i = 0; i < 8; ++i) { x[i] = rbf(__fmul_rn(x[i], g0[i])); y[i] = rbf(__fmul_rn(y[i], g1[i])); }
  }
#endif
  const float inv = rsqrt_ftz(__fadd_rn(div_full(row_sum(thread_sum_fma(x, y), wsum0, bars, rank), 8192.0f), eps));
  float u0[8], u1[8], n0[8], n1[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    u0[i] = __fmaf_rn(r0[i], a0[i], __fmul_rn(rbf(__fmul_rn(inv, x[i])), b0[i]));
    u1[i] = __fmaf_rn(r1[i], a1[i], __fmul_rn(rbf(__fmul_rn(inv, y[i])), b1[i]));
#if NORM_FP32
    n0[i] = u0[i]; n1[i] = u1[i];
#else
    n0[i] = rbf(u0[i]); n1[i] = rbf(u1[i]);
#endif
  }
  float* o = out + rb;
  *reinterpret_cast<float4*>(o + c0) = make_float4(u0[0], u0[1], u0[2], u0[3]);
  *reinterpret_cast<float4*>(o + c0 + 4) = make_float4(u0[4], u0[5], u0[6], u0[7]);
  *reinterpret_cast<float4*>(o + c1) = make_float4(u1[0], u1[1], u1[2], u1[3]);
  *reinterpret_cast<float4*>(o + c1 + 4) = make_float4(u1[4], u1[5], u1[6], u1[7]);
#if NORM_FP32
  const float sum2 = thread_sum_mul(n0, n1);
#else
  const float sum2 = thread_sum_fma(n0, n1);
#endif
  const float inv2 = rsqrt_ftz(__fadd_rn(div_full(row_sum(sum2, wsum1, bars + 1, rank), 8192.0f), next_eps));
  float w0[8], w1[8];
  unpack8(w0q, w0); unpack8(w1q, w1);
  u32 p0[4], p1[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    p0[i] = f2bf(__fmul_rn(__fmul_rn(inv2, n0[2 * i]), __fadd_rn(w0[2 * i], 1.0f)))
          | (f2bf(__fmul_rn(__fmul_rn(inv2, n0[2 * i + 1]), __fadd_rn(w0[2 * i + 1], 1.0f))) << 16);
    p1[i] = f2bf(__fmul_rn(__fmul_rn(inv2, n1[2 * i]), __fadd_rn(w1[2 * i], 1.0f)))
          | (f2bf(__fmul_rn(__fmul_rn(inv2, n1[2 * i + 1]), __fadd_rn(w1[2 * i + 1], 1.0f))) << 16);
  }
  u16* no = next_out + rb;
  *reinterpret_cast<uint4*>(no + c0) = make_uint4(p0[0], p0[1], p0[2], p0[3]);
  *reinterpret_cast<uint4*>(no + c1) = make_uint4(p1[0], p1[1], p1[2], p1[3]);
  if (!trigger) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
"""


def _join_cl_size(rows: int) -> int:
    if JOIN_CLUSTER in ("4", "8"):
        return int(JOIN_CLUSTER)
    return 4 if rows >= JOIN_CLUSTER_4_FROM_ROWS else 8


def _join_cl_check(r):
    if int(r[0]) != 0:
        raise RuntimeError(f"projection_publish join cluster kernel: CUDA/NVRTC error {r!r}")
    return r[1] if len(r) == 2 else r[1:]


def _join_cl_function(device: torch.device, has_gain: bool, norm_fp32: bool, cl: int):
    """The compiled cluster join, or ``None`` if it is not compiled yet and a capture is running."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, bool(has_gain), bool(norm_fp32), int(cl))
    fn = _JOIN_CL_FN.get(key)
    if fn is not None:
        return fn
    if torch.cuda.is_current_stream_capturing():
        if not _JOIN_CL_LOGGED:
            _JOIN_CL_LOGGED.append(True)
            logger.warning(
                f"[musespark_open] projection_publish: {JOIN_CLUSTER_ENV}: first join inside a CUDA-graph capture; this "
                "capture keeps the Triton join (the cluster kernel compiles on an eager call)"
            )
        return None
    from cuda.bindings import driver as cu
    from cuda.bindings import nvrtc

    major, minor = torch.cuda.get_device_capability(index)
    prog = _join_cl_check(
        nvrtc.nvrtcCreateProgram(_JOIN_CL_SRC.encode(), b"publish_join_cluster.cu", 0, [], [])
    )
    opts = [
        f"--gpu-architecture=sm_{major}{minor}".encode(),
        b"-std=c++17",
        b"-default-device",
        b"-DPUB=1",
        f"-DHAS_GAIN={int(bool(has_gain))}".encode(),
        f"-DNORM_FP32={int(bool(norm_fp32))}".encode(),
        f"-DCL={int(cl)}".encode(),
    ]
    r = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if int(r[0]) != 0:
        n = _join_cl_check(nvrtc.nvrtcGetProgramLogSize(prog))
        log = b" " * n
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(
            "projection_publish join cluster kernel: NVRTC compile failed:\n"
            + log.decode(errors="replace")
        )
    n = _join_cl_check(nvrtc.nvrtcGetCUBINSize(prog))
    cubin = b" " * n
    _join_cl_check(nvrtc.nvrtcGetCUBIN(prog, cubin))
    with torch.cuda.device(index):
        mod = _join_cl_check(cu.cuModuleLoadData(cubin))
        f = _join_cl_check(cu.cuModuleGetFunction(mod, b"pubjoin4"))
    regs = int(
        _join_cl_check(cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, f))
    )
    lmem = int(
        _join_cl_check(
            cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, f)
        )
    )
    _JOIN_CL_FN[key] = (f, mod)
    logger.info(
        f"[musespark_open] projection_publish: {JOIN_CLUSTER_ENV}={JOIN_CLUSTER}: post-FFN join cluster kernel "
        f"compiled for sm_{major}{minor} on device {index} (cluster {cl} x {512 // cl} threads, gain {bool(has_gain)}, "
        f"FP32 next-norm residual {bool(norm_fp32)}, regs {regs}, local {lmem} B)"
    )
    return _JOIN_CL_FN[key]


def _join_cl_launch(
    fn,
    cl,
    rows,
    updated,
    residual,
    receive,
    gain,
    alpha,
    beta,
    post_eps,
    next_input,
    weight,
    pre_eps,
    guard_ptr,
    buffer,
    world,
    capacity,
    columns,
    guard_rounds,
    pdl,
):
    from cuda.bindings import driver as cu

    cfg = cu.CUlaunchConfig()
    cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = cl * rows, 1, 1
    cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = 512 // cl, 1, 1
    cfg.sharedMemBytes = 0
    cfg.hStream = cu.CUstream(torch.cuda.current_stream(residual.device).cuda_stream)
    a1 = cu.CUlaunchAttribute()
    a1.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    a1.value.clusterDim.x, a1.value.clusterDim.y, a1.value.clusterDim.z = cl, 1, 1
    attrs = [a1]
    if pdl:
        a2 = cu.CUlaunchAttribute()
        a2.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        a2.value.programmaticStreamSerializationAllowed = 1
        attrs.append(a2)
    cfg.attrs = attrs
    cfg.numAttrs = len(attrs)
    vp, ll, fl, ii = ctypes.c_void_p, ctypes.c_longlong, ctypes.c_float, ctypes.c_int
    args = (
        (
            updated.data_ptr(),
            residual.data_ptr(),
            receive.data_ptr(),
            gain.data_ptr(),
            alpha.data_ptr(),
            beta.data_ptr(),
            float(post_eps),
            next_input.data_ptr(),
            weight.data_ptr(),
            float(pre_eps),
            int(guard_ptr),
            int(buffer),
            int(world),
            int(capacity),
            int(columns),
            int(guard_rounds),
            int(bool(pdl)),
        ),
        (vp, vp, vp, vp, vp, vp, fl, vp, vp, fl, vp, ii, ii, ii, ii, ll, ii),
    )
    _join_cl_check((cu.cuLaunchKernelEx(cfg, fn[0], args, 0)[0], None))


def capture_dependencies(stream: torch.cuda.Stream):
    """``(nodes, edge_data)`` the next node captured on ``stream`` would depend on, or None when not capturing."""
    from cuda.bindings import runtime as cudart

    err, status, _, _, nodes, edges, count = cudart.cudaStreamGetCaptureInfo(stream.cuda_stream)
    if (
        err != cudart.cudaError_t.cudaSuccess
        or status != cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive
    ):
        return None
    return list(nodes)[:count], list(edges)[:count]


def single_kernel_node(nodes) -> bool:
    """Whether ``nodes`` is exactly one kernel node (the only kind a programmatic edge may start from)."""
    from cuda.bindings import runtime as cudart

    if len(nodes) != 1:
        return False
    err, kind = cudart.cudaGraphNodeGetType(nodes[0])
    return (
        err == cudart.cudaError_t.cudaSuccess
        and kind == cudart.cudaGraphNodeType.cudaGraphNodeTypeKernel
    )


def programmatic_from(stream: torch.cuda.Stream, sources) -> bool:
    """Make the capture edges from ``sources`` to ``stream``'s next node programmatic (from the kernel's programmatic
    port), keeping every other dependency as it is. True when an edge was changed."""
    from cuda.bindings import runtime as cudart

    current = capture_dependencies(stream)
    if current is None:
        return False
    nodes, edges = current
    keys = {int(node) for node in sources}
    changed = False
    new_edges = []
    for node, edge in zip(nodes, edges):
        data = cudart.cudaGraphEdgeData()
        data.from_port, data.to_port, data.type = edge.from_port, edge.to_port, edge.type
        if int(node) in keys:
            data.from_port = cudart.cudaGraphKernelNodePortProgrammatic
            data.type = cudart.cudaGraphDependencyType.cudaGraphDependencyTypeProgrammatic
            changed = True
        new_edges.append(data)
    if not changed:
        return False
    (err,) = cudart.cudaStreamUpdateCaptureDependencies(
        stream.cuda_stream,
        nodes,
        new_edges,
        len(nodes),
        cudart.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies,
    )
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"[musespark_open] projection_publish: programmatic pool fork failed: {err}"
        )
    return True


def _splitk_tactic(hidden: torch.Tensor, weight: torch.Tensor, slot: torch.Tensor):
    """The split-K tactic for this publication, or None (cuBLAS)."""
    if not SPLITK_MIN_ROWS <= hidden.shape[0] <= SPLITK_MAX_ROWS or hidden.dim() != 2:
        return None
    tactic = TP8_PUBLISH_SPLITK_TACTICS.get((hidden.shape[1], weight.shape[0]))
    if tactic is None or _splitk_publisher() is None:
        return None
    if not (
        hidden.dtype == weight.dtype == slot.dtype == torch.bfloat16
        and hidden.is_contiguous()
        and weight.is_contiguous()
        and slot.is_contiguous()
        and tuple(slot.shape) == (hidden.shape[0], weight.shape[0])
        and hidden.data_ptr() % 32 == 0
        and weight.data_ptr() % 32 == 0
        and slot.data_ptr() % 16 == 0
        and not torch.is_grad_enabled()
    ):
        return None
    return tactic


def local_columns(width: int, world: int = WORLD) -> int:
    """Columns of a ``width``-wide projection output one of ``world`` ranks computes."""
    return width // world


class PublishTransport:
    """One site's multicast transport on every rank of a TP4 group, for up to ``max_rows`` rows.

    ``slot(buffer, rows)`` is this rank's ``[rows, columns]`` BF16 slot of ``buffer`` on the
    multicast address: a GEMM given it as its output publishes into every rank's copy;
    nothing may read it. ``receive()`` is this rank's copy (``int16``,
    ``[BUFFERS, WORLD, max_rows, columns]``), which the consumers poll and reset.
    Construction is collective over the group (fabric allocation, sentinel fill) and
    succeeds on every rank or raises on every rank.

    The guard word lives in pinned host memory, so the host reads it while the device still
    polls (``guard_fired(sync=False)``, ``start_guard_watch``).
    """

    def __init__(self, mapping: Mapping, columns: int, max_rows: int = MAX_ROWS, comm=None):
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_device_index,
            _mnnvl_workspace_all_succeeded,
        )

        if mapping.tp_size not in WORLDS or mapping.enable_attention_dp or mapping.cp_size != 1:
            raise ValueError(
                "the projection publish transport needs TP4 or TP8 without attention DP or CP"
            )
        world = mapping.tp_size
        if columns not in (local_columns(POOL_HIDDEN, world), local_columns(UNPOOL_HIDDEN, world)):
            raise ValueError(f"no projection site has {columns} columns per rank at TP{world}")
        if not MIN_ROWS <= max_rows <= MAX_ROWS:
            raise ValueError(f"max_rows must be in {MIN_ROWS}..{MAX_ROWS}, got {max_rows}")
        self.rank = mapping.tp_rank
        self.world = mapping.tp_size
        self.columns = int(columns)
        self.max_rows = int(max_rows)
        self.nbytes = BUFFERS * self.world * self.max_rows * self.columns * 2
        self.device = torch.device("cuda", _mnnvl_device_index(mapping))
        self._watch: Optional[threading.Thread] = None
        #: Warnings the watch thread logged (tests).
        self.guard_warnings = 0
        #: Device memory the transport took on this rank (the multicast allocation rounds up).
        self.allocated = 0
        self._comm = comm if comm is not None else _get_mnnvl_workspace_comm(mapping)
        refusal: Optional[str] = None
        if torch.cuda.get_device_capability(self.device) not in ((10, 3), (10, 7)):
            refusal = "the projection publish transport runs on SM103 and SM107 only"
        elif not prefer_pinned():
            refusal = "the guard words need pinned host memory, which this platform does not pin"
        if not _mnnvl_workspace_all_succeeded(self._comm, refusal is None):
            raise RuntimeError(
                "projection publish transport refused on at least one rank"
                f"{f' (this rank: {refusal})' if refusal is not None else ''}"
            )
        error: Optional[BaseException] = None
        try:
            torch.cuda.synchronize(self.device)
            free_before, _ = torch.cuda.mem_get_info(self.device)
            self._handle = _make_mnnvl_mcast_buffer(
                self._comm, self.nbytes, mapping, mapping.is_multi_node()
            )
            shape = (BUFFERS, self.world, self.max_rows, self.columns)
            self._receive = self._handle.get_uc_buffer(self.rank, shape, torch.int16, 0)
            self._slots = self._handle.get_mc_buffer(shape, torch.bfloat16, 0)
            with torch.cuda.device(self.device):
                self._receive.fill_(_SENTINEL_I16)
                # Pinned host memory is mapped into the device's address space at the same
                # address (unified addressing): a guard store reaches it at system scope and
                # the host reads it without a synchronization.
                self.control = torch.zeros(
                    CONTROL_WORDS, dtype=torch.int32, pin_memory=prefer_pinned()
                )
                status, pointer = _cudart.cudaHostGetDevicePointer(self.control.data_ptr(), 0)
                if status != _cudart.cudaError_t.cudaSuccess:
                    raise RuntimeError(f"cudaHostGetDevicePointer of the guard words: {status}")
                self._control_ptr = int(pointer)
                torch.cuda.synchronize(self.device)
            self.allocated = max(0, free_before - torch.cuda.mem_get_info(self.device)[0])
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "projection publish transport failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error

    def slot(self, buffer: int, rows: int) -> torch.Tensor:
        """This rank's ``[rows, columns]`` BF16 slot of ``buffer``, on the multicast address.

        Write-only: a store through it reaches every rank's copy; nothing may read it.
        """
        if buffer not in range(BUFFERS) or not 1 <= rows <= self.max_rows:
            raise ValueError(f"no slot for buffer {buffer}, {rows} rows")
        return self._slots[buffer, self.rank, :rows]

    def is_slot(self, tensor: torch.Tensor, buffer: int, rows: int) -> bool:
        """Whether ``tensor`` is this rank's slot of ``buffer`` at ``rows`` rows (address and shape)."""
        return (
            buffer in range(BUFFERS)
            and 1 <= rows <= self.max_rows
            and tensor.dim() == 2
            and tuple(tensor.shape) == (rows, self.columns)
            and tensor.data_ptr() == self._slots[buffer, self.rank].data_ptr()
        )

    def receive(self) -> torch.Tensor:
        """This rank's copy of the transport, ``int16 [BUFFERS, WORLD, max_rows, columns]``."""
        return self._receive

    def control_ptr(self) -> int:
        """The device address of the guard words (pinned host memory)."""
        return self._control_ptr

    def is_clean(self) -> bool:
        """Whether every half of this rank's copy holds the sentinel (synchronizes; tests)."""
        torch.cuda.synchronize(self.device)
        return bool((self._receive == _SENTINEL_I16).all().item())

    def reset(self) -> None:
        """Every half of this rank's copy back to the sentinel (tests; nothing may be in flight)."""
        torch.cuda.synchronize(self.device)
        self._receive.fill_(_SENTINEL_I16)
        torch.cuda.synchronize(self.device)

    def guard_fired(self, sync: bool = True) -> bool:
        """Whether a poll of this transport ever outlasted its guard.

        ``sync``: synchronize the device first (tests). Without it the word is read as it
        stands in host memory, while kernels may still be polling.
        """
        if sync:
            torch.cuda.synchronize(self.device)
        return int(self.control[GUARD_WORD]) != 0

    def clear_guard(self) -> None:
        """Zero the guard words (tests; nothing may be in flight)."""
        torch.cuda.synchronize(self.device)
        self.control.zero_()

    def start_guard_watch(
        self,
        name: str,
        interval_s: float = GUARD_WATCH_S,
        abort: bool = False,
        relog_s: float = GUARD_RELOG_S,
    ) -> bool:
        """Read the guard word from a host thread every ``interval_s`` seconds (0: no thread).

        An expired guard means a poll has waited longer than its guard for its slot. The
        consumer keeps polling and consumes nothing before the slot arrives: a peer that is
        only late (a host stall) still completes the call exactly, while a slot that never
        arrives (the ranks' call sequences differ, or a peer stopped) leaves this rank
        waiting, which the executor's hang detector reports. The thread logs a warning
        naming the rank, clears the word and keeps watching (a poll still waiting may set it
        again; at most one warning per ``relog_s``). With ``abort`` it aborts the process at
        the first expiry instead. Idempotent; returns whether a thread runs.
        """
        if self._watch is not None:
            return True
        if interval_s <= 0:
            return False
        stop = threading.Event()

        def watch():
            logged = None
            while not stop.wait(interval_s):
                if not self.guard_fired(sync=False):
                    continue
                if abort:
                    logger.error(
                        f"[musespark_open] projection_publish: rank {self.rank} polled the {name} "
                        "transport past its guard without its slot arriving; aborting"
                    )
                    os.abort()
                self.control[GUARD_WORD] = 0
                now = time.monotonic()
                if logged is not None and now - logged < relog_s:
                    continue
                logged = now
                self.guard_warnings += 1
                logger.warning(
                    f"[musespark_open] projection_publish: rank {self.rank} has polled the {name} "
                    "transport for longer than its guard without its slot arriving. The consumer "
                    "keeps polling and consumes nothing before the slot arrives: a late peer "
                    "completes the call normally; if the slot never arrives (the ranks' call "
                    "sequences differ, or a peer stopped), this rank waits and the executor's hang "
                    "detector reports it"
                )

        self._watch_stop = stop
        self._watch = threading.Thread(
            target=watch, name=f"musespark_open.projection_publish_{name}_watch", daemon=True
        )
        self._watch.start()
        return True

    def stop_guard_watch(self) -> None:
        """Stop the watch thread, if one runs (tests)."""
        if self._watch is not None:
            self._watch_stop.set()
            self._watch.join()
            self._watch = None


def cublaslt_version() -> Optional[int]:
    """``cublasLtGetVersion()`` of the cuBLASLt this process loaded (torch's), or None."""
    try:
        with open("/proc/self/maps") as maps:
            paths = sorted({line.split()[-1] for line in maps if "libcublasLt.so" in line})
    except OSError:
        return None
    for path in paths:
        try:
            library = ctypes.CDLL(path)
        except OSError:
            continue
        library.cublasLtGetVersion.restype = ctypes.c_size_t
        return int(library.cublasLtGetVersion())
    return None


def producer_refusal(device: torch.device, world: int = WORLD) -> Optional[str]:
    """Why ``publish_projection`` may not publish on ``device`` in this process, or None.

    The single-store property of the cuBLAS producer is a property of the kernels cuBLAS
    picks, which the census certified for one torch, cuBLASLt and SM count
    (``CERTIFIED_PRODUCERS``) with torch's default cuBLAS workspaces. Anything else may
    pick a kernel that writes partial sums into the slot or reads it back.
    """
    torch.cuda.init()
    key = (
        str(torch.__version__),
        cublaslt_version(),
        torch.cuda.get_device_properties(device).multi_processor_count,
    )
    certified = CERTIFIED_PRODUCERS_TP8 if world == 8 else CERTIFIED_PRODUCERS
    if key not in certified:
        return (
            f"the TP{world} cuBLAS producer census does not cover (torch, cuBLASLt, SMs) = {key} "
            f"(certified: {sorted(certified)})"
        )
    overrides = [name for name in CUBLAS_WORKSPACE_ENV if _sm107_env(name)]
    if overrides:
        return f"{', '.join(overrides)} resize the cuBLAS workspaces the producer census ran with"
    return None


def wide_producer_refusal(device: torch.device) -> Optional[str]:
    """``producer_refusal`` for the GB300 TP8 cuBLAS publication (``CERTIFIED_PRODUCERS_TP8_SM103_WIDE``)."""
    torch.cuda.init()
    key = (
        str(torch.__version__),
        cublaslt_version(),
        torch.cuda.get_device_properties(device).multi_processor_count,
    )
    if key not in CERTIFIED_PRODUCERS_TP8_SM103_WIDE:
        return f"the GB300 TP8 cuBLAS producer census does not cover {key}"
    overrides = [name for name in CUBLAS_WORKSPACE_ENV if _sm107_env(name)]
    if overrides:
        return f"{', '.join(overrides)} resize the cuBLAS workspaces the producer census ran with"
    return None


def publish_projection_chunked(
    hidden: torch.Tensor, weight: torch.Tensor, slot: torch.Tensor
) -> bool:
    """Publish a 65..448-row TP8 slice from the certified split-K producer, one launch per 64 rows.

    Rows are independent in a GEMM, so every row carries the bits the producer gives it in a 33..64-row call
    (split_k 4 = sm_103's TP8 8-row cell: each row's one-request bits), and each launch stores its rows of the slot
    once (the producer census). The launches run in stream order with PDL; a launch's dependency wait implies its
    predecessor's. Returns False (nothing launched) when the producer is not selected or a chunk does not fit it.
    """
    return _publish_chunked(hidden, weight, slot, None)


_chunk_streams: dict = {}


def publish_projection_chunked_k1(
    hidden: torch.Tensor, weight: torch.Tensor, slot: torch.Tensor
) -> bool:
    """``splitk1``: ``publish_projection_chunked`` at split_k 1 (``TP8_PUBLISH_SPLITK1_TACTICS``): bitwise equal
    to the built-in replicated GEMM's columns of this rank. A split_k-1 launch has one CTA per (64-column, row) tile
    streaming the whole K, so the 64-row chunks run concurrently: the first on the current stream (PDL), the others
    forked onto a side stream (plain launches) and joined back before the slot's consumer is enqueued."""
    return _publish_chunked(hidden, weight, slot, TP8_PUBLISH_SPLITK1_TACTICS)


def _publish_chunked(hidden, weight, slot, table) -> bool:
    rows = hidden.shape[0] if hidden.dim() == 2 else 0
    if rows <= SPLITK_MAX_ROWS or slot.dim() != 2 or slot.shape[0] != rows:
        return False
    # 64-row launches, the last one the remainder (1..64 rows): the chunked producer's census and bits ran on exactly
    # these launches (every call of 33..128 rows, device-checked).
    starts = list(range(0, rows, SPLITK_MAX_ROWS))
    bounds = list(zip(starts, starts[1:] + [rows]))
    head = _splitk_tactic(hidden[:SPLITK_MAX_ROWS], weight, slot[:SPLITK_MAX_ROWS])
    if head is None or not (hidden.is_contiguous() and slot.is_contiguous()):
        return False
    tactics = [head] * len(bounds)
    run_splitk_dense, splitk_tactic = _splitk_publisher()
    if table is None and WIDE_ONE:
        # The split_k-4 publication of any 65..448-row slice in ONE launch (row tiles on the grid's second
        # dimension), at the 65+-row tile of this site (``wide_splitk_tactic``); every row keeps its one-request bits.
        wt = wide_splitk_tactic(hidden.shape[1], weight.shape[0], rows)
        if wt is None:
            return False
        _run_splitk_one(hidden, weight, slot, splitk_tactic(*wt))
        logger.info_once(
            f"[musespark_open] projection_publish: {WIDE_ONE_ENV}=1: 65..{WIDE_ONE_K4_MAX_ROWS}-row split_k-4 "
            f"slices publish in one launch (tactics {_wide_splitk_tactics()})",
            key="musespark_open.t8r5.publish_wide_one_k4",
        )
        return True
    if table is None:
        for (a, b), tactic in zip(bounds, tactics):
            run_splitk_dense(
                hidden[a:b], weight.t(), None, slot[a:b], _SPLITK_PDL, splitk_tactic(*tactic)
            )
        return True
    tactic = splitk_tactic(*table[(hidden.shape[1], weight.shape[0])])
    if WIDE_ONE and table is TP8_PUBLISH_SPLITK1_TACTICS and rows <= 2 * SPLITK_MAX_ROWS:
        _run_splitk_one(hidden, weight, slot, tactic)
        logger.info_once(
            f"[musespark_open] projection_publish: {WIDE_ONE_ENV}=1: 65..128-row split_k-1 slices "
            "publish in one launch",
            key="musespark_open.t8r4.publish_wide_one",
        )
        return True
    main = torch.cuda.current_stream(hidden.device)
    side = _chunk_streams.get(hidden.device)
    if side is None:
        # A stream of its own, never one of PyTorch's pool (a pooled stream can be the capturing stream itself, or
        # another fast path's side stream, whose in-order work -- a spinning consumer among it -- the chunk would
        # queue behind).
        from ...kernels.dense_gemms.dispatcher import dedicated_stream

        side = _chunk_streams.setdefault(hidden.device, dedicated_stream(hidden.device))
    side.wait_stream(main)
    (a0, b0), rest = bounds[0], bounds[1:]
    with torch.cuda.stream(side):
        for a, b in rest:
            run_splitk_dense(hidden[a:b], weight.t(), None, slot[a:b], False, tactic)
    run_splitk_dense(hidden[a0:b0], weight.t(), None, slot[a0:b0], _SPLITK_PDL, tactic)
    # The join orders every later use (and the allocator's reuse of ``hidden``, allocated on ``main``) after the side
    # launches; ``weight`` is a parameter and ``slot`` transport memory.
    main.wait_stream(side)
    return True


#: ``MUSESPARK_OPEN_SM103_TEP8_PUBLISH_WIDE_ONE=1`` (with ``PUBLISH_WIDE=splitk1``; default off, the environment
#: only) publishes a 65..128-row slice as ONE launch of the split_k-1 kernel over every row (its grid's second dimension
#: takes the row tiles) on the current stream (PDL), instead of 64-row launches forked onto a side stream and joined
#: back. Same tactic, same compile options as the 64-row launch (one producer key), split_k 1: every element one
#: full-K FP32 accumulation in K order, rounded once, so the bits do not depend on the launch (device-checked).
#: Without the fork / join the post-FFN join and the next consumer stay
#: programmatic dependents of the publication, and every CTA stages its weight before the dependency wait.
WIDE_ONE_ENV = "MUSESPARK_OPEN_SM103_TEP8_PUBLISH_WIDE_ONE"
_WIDE_ONE = os.environ.get(WIDE_ONE_ENV, "").strip()
if _WIDE_ONE not in ("", "0", "1"):
    raise ValueError(f"{WIDE_ONE_ENV}={_WIDE_ONE!r}: expected 0 or 1")
WIDE_ONE = _WIDE_ONE == "1"


#: The most rows a one-launch split_k-4 publication takes (the transports' bound).
WIDE_ONE_K4_MAX_ROWS = 448

#: ``MUSESPARK_OPEN_SM103_TEP8_PUBLISH_WIDE_TACTICS`` (``;``-separated ``<site>[@<min rows>]=M,N,4,S`` entries,
#: site ``pool`` or ``unpool``; default the 33..64-row split-K tactics at every row count) names the tile of the
#: one-launch split_k-4 publication of 65+-row slices (``PUBLISH_WIDE=splitk`` + ``PUBLISH_WIDE_ONE=1``); an entry with
#: ``@R`` serves calls of R rows or more (e.g. ``pool=64,32,4,8;pool@129=64,64,4,5``). Only split_k 4 is accepted: the K
#: partition (4 cluster ranks, each its K/4 in CTA_K order, reduced in cluster-rank order through DSMEM) is what fixes
#: an element's bits; the output-column tile (mma_m) and the token tile (mma_n) only place the element, so every row
#: keeps the bits of sm_103's TP8 8-row cell (device-checked: every tile below, rows
#: 65..256). Read at first use; rank-uniform.
WIDE_TACTICS_ENV = "MUSESPARK_OPEN_SM103_TEP8_PUBLISH_WIDE_TACTICS"


def _wide_splitk_tactics() -> dict:
    """{(K, columns): [(min rows, tactic), ...] sorted by min rows}."""
    cached = _splitk_state.get("wide_tactics")
    if cached is not None:
        return cached
    sites = {"pool": (UNPOOL_HIDDEN, POOL_HIDDEN // 8), "unpool": (POOL_HIDDEN, UNPOOL_HIDDEN // 8)}
    table = {key: {SPLITK_MAX_ROWS + 1: TP8_PUBLISH_SPLITK_TACTICS[key]} for key in sites.values()}
    value = _sm107_env(WIDE_TACTICS_ENV, "").strip()
    for part in filter(None, (p.strip() for p in value.split(";"))):
        head, _, fields = part.partition("=")
        site, _, low = head.strip().partition("@")
        try:
            tactic = tuple(int(f) for f in fields.split(","))
            low = int(low) if low else SPLITK_MAX_ROWS + 1
        except ValueError:
            tactic, low = (), 0
        if (
            site not in sites
            or len(tactic) != 4
            or tactic[2] != 4
            or not SPLITK_MAX_ROWS < low <= WIDE_ONE_K4_MAX_ROWS
        ):
            raise ValueError(
                f"{WIDE_TACTICS_ENV}={value!r}: expected <pool|unpool>[@<65..{WIDE_ONE_K4_MAX_ROWS}>]="
                "M,N,4,S entries (split_k 4)"
            )
        table[sites[site]][low] = tactic
    table = {key: sorted(v.items()) for key, v in table.items()}
    _splitk_state["wide_tactics"] = table
    return table


def wide_splitk_tactic(k: int, n: int, rows: int) -> Optional[Tuple[int, int, int, int]]:
    """The one-launch split_k-4 tile of a ``rows``-row ``[R, k] x [n, k]^T`` slice, or None (not a TP8 publish site)."""
    ranges = _wide_splitk_tactics().get((k, n))
    if not ranges:
        return None
    return [tactic for low, tactic in ranges if low <= rows][-1]


def _run_splitk_one(hidden: torch.Tensor, weight: torch.Tensor, slot: torch.Tensor, tactic) -> None:
    """``WIDE_ONE``: ``run_splitk_dense`` of all ``rows`` in one launch, compiled exactly as the 64-row
    launch (``_launch_options(SPLITK_MAX_ROWS, split_k)``, the 33..64-row publication's producer key at split_k 4);
    the tactic is validated at the 64-row launch's M (tile, K partition, shared memory), the kernel itself tiles any
    M. split_k 1: 65..128 rows; split_k 4: 65..``WIDE_ONE_K4_MAX_ROWS`` rows."""
    from ..gemm import splitk_gemm as sk

    a, b = hidden, weight.t()
    m, n, k = sk._validate_runtime_tensors(a, b, None, slot)
    top = {1: 2 * SPLITK_MAX_ROWS, 4: WIDE_ONE_K4_MAX_ROWS}.get(tactic.split_k, 0)
    if not SPLITK_MAX_ROWS < m <= top:
        raise ValueError(
            f"publish WIDE_ONE: split_k 1 at {SPLITK_MAX_ROWS + 1}..{2 * SPLITK_MAX_ROWS} rows or "
            f"split_k 4 at {SPLITK_MAX_ROWS + 1}..{WIDE_ONE_K4_MAX_ROWS}, got {tactic} at {m}"
        )
    sk.validate_tactic(tactic, SPLITK_MAX_ROWS, n, k)
    cute_tensors = sk._to_cute_swap(a, b, slot, None)
    compiled = sk._get_compiled_splitk_kernel(
        dtype=a.dtype,
        tactic=tactic,
        use_pdl=_SPLITK_PDL,
        has_bias=False,
        leading_dims=cute_tensors[4],
        **sk._launch_options(SPLITK_MAX_ROWS, tactic.split_k),
    )
    compiled(*cute_tensors[:3], sk._cuda.CUstream(torch.cuda.current_stream(a.device).cuda_stream))


#: SM counts where the split_k-1 producer passed the census (single store, every row 33..128 in 64-row launches, both
#: sites) with the split-K census's device code.
SPLITK1_CERTIFIED_SMS = frozenset({152})


def splitk1_refusal() -> Optional[str]:
    """Why the split_k-1 producer may not publish here, or None: same device code as a certified split-K producer
    (``CERTIFIED_SPLITK_PRODUCERS_TP8``), owner-store epilogue, an SM count its census ran on."""
    try:
        from ..gemm import splitk_gemm as splitk

        keys = splitk1_producer_keys(splitk)
    except Exception as error:  # noqa: BLE001
        return f"split_k-1 producer unavailable ({error!r})"
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    md5s = {entry[1][0] for entry in CERTIFIED_SPLITK_PRODUCERS_TP8 if entry[0] == sms}
    if sms not in SPLITK1_CERTIFIED_SMS:
        return f"no split_k-1 census on {sms} SMs"
    if any(k is None or k[0] not in md5s or k[2] != "owner" for k in keys):
        return f"split_k-1 producer keys {keys} not covered by the split-K census device code {sorted(md5s)}"
    return None


def splitk1_producer_keys(splitk) -> Tuple:
    """(pool key, unpool key) of the split_k-1 producer (``splitk_gemm.producer_key`` at 33 and 64 rows), else None."""
    keys = []
    for kn in ((UNPOOL_HIDDEN, POOL_HIDDEN // 8), (POOL_HIDDEN, UNPOOL_HIDDEN // 8)):
        keys.append(_splitk_site_key(splitk, TP8_PUBLISH_SPLITK1_TACTICS[kn]))
    return tuple(keys)


def publish_projection(hidden: torch.Tensor, weight: torch.Tensor, slot: torch.Tensor) -> None:
    """``hidden @ weight^T`` (cuBLAS through ``torch.mm``) stored straight into ``slot``.

    ``slot`` is a transport slot (``PublishTransport.slot``): the GEMM's epilogue stores
    are the publication. ``weight`` is the rank's ``[columns, K]`` row slice of the
    projection's weight (contiguous rows of the loaded weight; no copy). Only with a
    certified producer (``producer_refusal``). Opt-in: a TP8 slice of
    33..64 rows publishes from the split-K kernel with ``MUSESPARK_OPEN_TEP8_PUBLISH_SPLITK=1``
    and a certified split-K producer (``_splitk_publisher``).
    """
    tactic = _splitk_tactic(hidden, weight, slot)
    if tactic is not None:
        run_splitk_dense, splitk_tactic = _splitk_publisher()
        run_splitk_dense(hidden, weight.t(), None, slot, _SPLITK_PDL, splitk_tactic(*tactic))
        return
    torch.mm(hidden, weight.t(), out=slot)


class _GuardWords:
    """The guard words handed to the Triton consumer as a pointer argument.

    Triton reads ``data_ptr()`` and ``dtype`` of a pointer argument; the pinned guard
    words' device address is their host address (unified addressing).
    """

    dtype = torch.int32

    def __init__(self, pointer: int) -> None:
        self._pointer = int(pointer)

    def data_ptr(self) -> int:
        return self._pointer


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - the image ships Triton
    triton = None

if triton is not None:

    @triton.jit(do_not_specialize=["buffer"])
    def _unpool_publish_join_kernel(
        out_ptr,  # [T, HIDDEN] fp32, the new residual stream
        residual_ptr,  # [T, HIDDEN] fp32, the old one
        receive_ptr,  # int16 [BUFFERS, WORLD, CAPACITY, COLUMNS], this rank's copy
        gain_ptr,  # [HIDDEN] (1 + w) when HAS_GAIN, else unread
        alpha_ptr,  # [HIDDEN] fp32
        beta_ptr,  # [HIDDEN] fp32
        eps,
        next_out_ptr,
        next_weight_ptr,
        next_eps,
        guard_ptr,  # int32, pinned host memory
        buffer,
        WORLD_SIZE: tl.constexpr,
        CAPACITY: tl.constexpr,
        COLUMNS: tl.constexpr,
        HIDDEN: tl.constexpr,
        BLOCK_H: tl.constexpr,
        HAS_GAIN: tl.constexpr,
        NORM_FP32_RESIDUAL: tl.constexpr,
        SENTINEL_I16: tl.constexpr,
        GUARD: tl.constexpr,
        GUARD_BREAK: tl.constexpr,
        PDL: tl.constexpr = False,
    ):
        """The built-in ``_gated_residual_after_rmsnorm_kernel`` (FUSE_NEXT_NORM) on a published branch.

        The branch row is read from the transport: element ``c`` of row ``token`` is element
        ``c % COLUMNS`` of source rank ``c // COLUMNS``'s slot of ``buffer``. The row is
        polled until no half holds the sentinel, reset to the sentinel, and then the
        built-in kernel's statements run unchanged on it (the same ``[BLOCK_H]`` tensors and
        ``num_warps``, so the same reductions).
        """
        if PDL:
            # TEP8_JOIN_PDL: wait for the prerequisite grid before any read, then let the
            # dependent (the next QKV GEMM) become resident and stage its weight.
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        token = tl.program_id(0)
        offs = tl.arange(0, BLOCK_H)
        mask = offs < HIDDEN
        base = token * HIDDEN + offs

        source = offs // COLUMNS
        box = (
            receive_ptr
            + ((buffer * WORLD_SIZE + source) * CAPACITY + token) * COLUMNS
            + (offs - source * COLUMNS)
        )
        raw = tl.load(box, mask=mask, other=0, volatile=True)
        pending = tl.max((raw == SENTINEL_I16).to(tl.int32), axis=0)
        rounds = 0
        while pending != 0:
            raw = tl.load(box, mask=mask, other=0, volatile=True)
            pending = tl.max((raw == SENTINEL_I16).to(tl.int32), axis=0)
            if GUARD > 0:
                rounds += 1
                if rounds == GUARD:
                    tl.atomic_xchg(guard_ptr, 1, sem="relaxed", scope="sys")
                    if GUARD_BREAK:
                        pending = pending * 0
        # Retire the consumed halves before this grid can complete.
        tl.store(box, tl.full([BLOCK_H], SENTINEL_I16, tl.int16), mask=mask)
        branch = raw.to(tl.bfloat16, bitcast=True)

        scaled = branch.to(tl.float32)
        if HAS_GAIN:
            gain = tl.load(gain_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            scaled = (scaled * gain).to(branch.dtype).to(tl.float32)
        variance = tl.sum(scaled * scaled, axis=0) / HIDDEN
        normed = (scaled * tl.rsqrt(variance + eps)).to(branch.dtype).to(tl.float32)

        residual = tl.load(residual_ptr + base, mask=mask, other=0.0).to(tl.float32)
        alpha = tl.load(alpha_ptr + offs, mask=mask, other=0.0)
        beta = tl.load(beta_ptr + offs, mask=mask, other=0.0)
        updated = alpha * residual + beta * normed
        tl.store(out_ptr + base, updated, mask=mask)
        if NORM_FP32_RESIDUAL:
            next_input = updated
        else:
            next_input = updated.to(branch.dtype).to(tl.float32)
        next_variance = tl.sum(next_input * next_input, axis=0) / HIDDEN
        next_weight = tl.load(next_weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        next_output = next_input * tl.rsqrt(next_variance + next_eps)
        next_output = next_output * (next_weight + 1.0)
        tl.store(next_out_ptr + base, next_output, mask=mask)


def _residual_num_warps() -> int:
    """The built-in join's warps per program (its reductions follow from them)."""
    from tensorrt_llm._torch.models.modeling_musespark_open import _RESIDUAL_NUM_WARPS

    return _RESIDUAL_NUM_WARPS


class UnpoolPublishJoin(OpWrapper):
    """The built-in post-FFN join (residual join + next pre-attention norm) on a published unpool slot.

    ``transport`` is the unpool site's ``PublishTransport``; ``buffer`` the call's buffer,
    whose slots every rank's unpool GEMM wrote for this call (``publish_projection``).
    ``rows`` rows; ``residual`` FP32 ``[rows, 8192]``; ``gain``, ``alpha``, ``beta``,
    ``weight``, the epsilons and ``norm_fp32_residual`` are the built-in
    ``MuseSparkOpenGatedResidual._add_after_rmsnorm_and_norm``'s. Returns its
    ``(updated, next_input)`` for the gathered branch.

    A collective over the TP4 group: every rank calls it with the same row count and
    buffer, in the same order relative to its other calls on ``transport``, and no two
    consecutive calls on one buffer (``buffer_sequence``). ``guard`` (polling rounds, 0:
    none) and ``guard_break`` select the poll guard (``guard_break`` for tests only).
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    note = """
    Certified bit for bit against the built-in Triton join run on the gathered branch
    (the four ranks' slots concatenated), through CUDA-graph replays with changing inputs,
    back-to-back 61-layer chains on the model's buffer sequence with a delayed rank, and
    against `reference` within its band. The slots are published by the production
    producer (cuBLAS into the multicast slot) and by an elementwise copy.
    """

    def __call__(
        self,
        transport: PublishTransport,
        buffer: int,
        rows: int,
        residual: torch.Tensor,
        gain: Optional[torch.Tensor],
        alpha: torch.Tensor,
        beta: torch.Tensor,
        weight: torch.Tensor,
        post_eps: float,
        pre_eps: float,
        norm_fp32_residual: bool = False,
        guard: int = GUARD_ROUNDS,
        guard_break: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        updated = torch.empty_like(residual)
        next_input = torch.empty(
            (rows, UNPOOL_HIDDEN), dtype=torch.bfloat16, device=residual.device
        )
        pdl = _join_pdl()
        if (
            JOIN_CLUSTER not in ("", "0")
            and not guard_break
            and residual.is_contiguous()
            and next_input.is_contiguous()
        ):
            # The bitwise cluster join (JOIN_CLUSTER above).
            cl = _join_cl_size(rows)
            fn = _join_cl_function(residual.device, gain is not None, bool(norm_fp32_residual), cl)
            if fn is not None:
                _join_cl_launch(
                    fn,
                    cl,
                    rows,
                    updated,
                    residual,
                    transport.receive(),
                    gain if gain is not None else weight,
                    alpha,
                    beta,
                    post_eps,
                    next_input,
                    weight,
                    pre_eps,
                    transport.control_ptr(),
                    int(buffer),
                    transport.world,
                    transport.max_rows,
                    transport.columns,
                    int(guard),
                    pdl,
                )
                return updated, next_input
        _unpool_publish_join_kernel[(rows,)](
            updated,
            residual,
            transport.receive(),
            gain if gain is not None else alpha,
            alpha,
            beta,
            post_eps,
            next_input,
            weight,
            pre_eps,
            _GuardWords(transport.control_ptr()),
            int(buffer),
            WORLD_SIZE=transport.world,
            CAPACITY=transport.max_rows,
            COLUMNS=transport.columns,
            HIDDEN=UNPOOL_HIDDEN,
            BLOCK_H=UNPOOL_HIDDEN,
            HAS_GAIN=gain is not None,
            NORM_FP32_RESIDUAL=bool(norm_fp32_residual),
            SENTINEL_I16=_SENTINEL_I16,
            GUARD=int(guard),
            GUARD_BREAK=bool(guard_break),
            num_warps=_residual_num_warps(),
            # Passed only with the gate, so the plain build is today's.
            **({"PDL": True, "launch_pdl": True} if pdl else {}),
        )
        return updated, next_input

    def reference(
        self,
        per_rank_slices: Sequence[torch.Tensor],
        residual: torch.Tensor,
        gain: Optional[torch.Tensor],
        alpha: torch.Tensor,
        beta: torch.Tensor,
        weight: torch.Tensor,
        post_eps: float,
        pre_eps: float,
        norm_fp32_residual: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The join on the concatenated slices, float64 between the built-in's BF16 roundings.

        The collective exception to mirroring: ``per_rank_slices`` is every rank's published
        slice in ascending rank order.
        """
        branch = torch.cat(list(per_rank_slices), dim=1)
        scaled = branch.double()
        if gain is not None:
            scaled = (scaled * gain.double()).to(torch.bfloat16).double()
        normed = scaled * torch.rsqrt(scaled.square().mean(-1, keepdim=True) + post_eps)
        normed = normed.to(torch.bfloat16).double()
        updated = alpha.double() * residual.double() + beta.double() * normed
        nxt = updated if norm_fp32_residual else updated.to(torch.bfloat16).double()
        out = nxt * torch.rsqrt(nxt.square().mean(-1, keepdim=True) + pre_eps)
        out = out * (weight.double() + 1.0)
        return updated.float(), out.to(torch.bfloat16)

    def is_valid(
        self,
        transport: PublishTransport,
        buffer: int,
        rows: int,
        residual: torch.Tensor,
        gain: Optional[torch.Tensor],
        alpha: torch.Tensor,
        beta: torch.Tensor,
        weight: torch.Tensor,
        post_eps: float,
        pre_eps: float,
        norm_fp32_residual: bool = False,
        guard: int = GUARD_ROUNDS,
        guard_break: bool = False,
        **unused_kwargs,
    ) -> None:
        """Rows beyond the transport's capacity address outside it; a strided or misaligned
        operand is read as if packed; a non-unpool transport has another row layout."""
        if transport.columns != local_columns(UNPOOL_HIDDEN, transport.world):
            raise ValueError(
                f"transport carries {transport.columns}-column slots, not the unpool's"
            )
        if buffer not in range(BUFFERS) or not 1 <= rows <= transport.max_rows:
            raise ValueError(f"buffer {buffer} / rows {rows} outside the transport")
        expected = (
            (residual, (rows, UNPOOL_HIDDEN), torch.float32),
            (alpha, (UNPOOL_HIDDEN,), torch.float32),
            (beta, (UNPOOL_HIDDEN,), torch.float32),
            (weight, (UNPOOL_HIDDEN,), torch.bfloat16),
        ) + (((gain, (UNPOOL_HIDDEN,), torch.bfloat16),) if gain is not None else ())
        for tensor, shape, dtype in expected:
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16
                or tensor.device != transport.device
            ):
                raise ValueError(
                    f"expected a contiguous 16-byte aligned {dtype} {shape} tensor on "
                    f"{transport.device}, got {tensor.dtype} {tuple(tensor.shape)}"
                )


unpool_publish_join = UnpoolPublishJoin()


def available() -> bool:
    """Whether this image can build the unpool consumer (Triton)."""
    return triton is not None


if JOIN_CLUSTER not in ("", "0"):
    logger.info(
        f"[musespark_open] projection_publish: {JOIN_CLUSTER_ENV}={JOIN_CLUSTER}: the post-FFN join of 33..512-row "
        "decode steps runs as the bitwise cluster kernel (8 CTAs a row up to 128 rows, 4 above)"
    )
