# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
"""A sliding-window layer's decode attention block, from the QKV projection to o_proj's input.

Between the QKV projection and the output projection a sliding-window decode
step of this target launches four dependent kernels: ``preattn_prep`` (the
weightless Q/K norms, the query pre-scale, RoPE and the gate copy),
``paged_decode_attention``'s split and combine, and the output norm-gate. The
window's FP8 K/V is half a megabyte, so the block is launch edges and the
dependent loads behind each one, not bytes.

This entry is that block with fewer edges, in the same arithmetic:

* The split kernel reads what does not depend on its predecessor *before*
  ``griddepcontrol.wait``: the sequence length, the block-table entries of its
  key block, the cached K/V of the keys before this step (``j < past``), and,
  with ``fused_prep``, the step's positions and their RoPE table rows. Only the
  projection operand is read after the wait.
* ``fused_prep``: the split programs normalize, scale and rotate their own
  query rows and the step's keys from the raw projection, with
  ``preattn_prep``'s statements (its thread-order sum, ``rsqrt.approx.ftz``,
  BF16 roundings, ``fma.rn.ftz`` / ``mul.rn.ftz`` rotation), so the prep
  launch is gone. The gate stays where the projection put it.
* ``gate``: the combine kernel applies the per-head RMSNorm and the BF16
  sigmoid gate to the head it has just merged, with the built-in norm-gate
  kernel's summation tree written out, and lets the output projection launch
  right after its own wait, as the PDL norm-gate copy does.

Every value is the one the four-kernel chain computes, bit for bit: the KV
cache bytes, the attention output and the gated o_proj input.

Each program of the split grid owns exactly one 32-key block (``num_splits``
must cover the window's largest block count, which ``window_splits`` does);
that is what lets its cached K/V be read before the wait.

``blocks_per_program`` > 1 runs the split as fewer programs: each owns that
many consecutive key blocks, reads the cached K/V of all of them before the
wait, prepares its query rows once, then runs each block with the
one-block program's statements from a fresh online-softmax state and stores
that block's partial in the block's own slot. The partials, the combine and
every output bit are the ones of ``blocks_per_program = 1``; what changes is
how many programs the grid has (``ceil(num_splits / blocks_per_program)`` per
head and row group), which decides how many rounds of resident programs a
step of several sequences needs.

``split_max_registers`` (with ``blocks_per_program`` > 1) compiles that split
kernel under a register cap per thread (Triton's ``maxnreg``): register
allocation only, so the same instructions on the same values and the same
bits, with more programs resident per SM (``SPLIT_REGISTER_CAP``: three).
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import triton
import triton.language as tl

try:
    from triton.language.extra import libdevice
except ImportError:  # the pre-3.2 layout
    from triton.language.extra.cuda import libdevice

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

from .paged_decode_attention import (
    BLOCK_N,
    MAX_ROWS,
    NUM_WARPS,
    Q_PER_KV,
    ROWS_PER_PROGRAM,
    _to_cache_fp8,
)
from .paged_decode_attention import HEAD_DIM as _HEAD_DIM
from .paged_decode_attention import paged_decode_attention as _paged_decode_attention
from .preattn_prep import _normalize, _rotate, _rsqrt_approx_ftz
from .preattn_prep import preattn_prep as _preattn_prep

HEAD_DIM = _HEAD_DIM
#: Stages of the split kernel's key loop: it runs at most once per program.
NUM_STAGES = 1
#: The key blocks one split program may own (``blocks_per_program``).
MAX_BLOCKS_PER_PROGRAM = 4
#: Registers per thread at which three 128-thread split programs fit an SM's 64K registers
#: (65536 / 384 = 170, rounded down to the allocation granule of 8).
SPLIT_REGISTER_CAP = 168
#: The register caps ``split_max_registers`` takes.
MIN_SPLIT_REGISTERS = 128
MAX_SPLIT_REGISTERS = 255

#: The one-cluster SWA block (CL8), ported from the sm_107 tree (lever ``swa_one_cluster_block``): a TEP4
#: decode step of ONE request (4 KV heads and 32 query heads per rank, 32-key FP8 pages, window 2048, fused prep with
#: the norm-gate, q_fp8, at most 8 rows) runs the whole sliding-window block as ONE cluster launch instead of the
#: split + combine pair: grid (8, 4, 1), a cluster of 8 CTAs per KV head, each CTA one SM (320 threads, 180 KB dynamic
#: SMEM); before the dependency wait the page ids and the window's cached K/V; after it the q/k prep, the step's K/V
#: append (the stock bytes), mma.sync attention over 288 keys per CTA and the partials merged through distributed
#: shared memory, then the norm-gate. Not bit-identical to the pair (another softmax partition, P in f16).
#: ``MUSESPARK_OPEN_SWA_CL8=all`` turns it on; unset or ``0`` leaves the pair unchanged. The kernel is compiled with
#: NVRTC on its first eager call per device (a call inside a CUDA-graph capture before that falls back to the pair
#: and warns once).
_SWA_CL8 = os.environ.get("MUSESPARK_OPEN_SWA_CL8", "0")
#: Requests one CL8 launch takes in one wave of clusters: a request needs one cluster of 8 CTAs per KV head (4 at TEP4) and a
#: GB300 holds 15 such clusters (``cuOccupancyMaxActiveClusters``, 180 KB SMEM per CTA), so 3 requests (12 clusters). At 4
#: requests (16 clusters) the launch takes two waves and is slower than the built-in prep + FMHA + norm-gate chain.
CL8_MAX_REQUESTS = 3

_CL8_SMEM = 180 * 1024
_CL8_SRC = r"""
typedef unsigned short u16; typedef unsigned int u32; typedef unsigned char u8;
typedef unsigned long long u64;

#define D 64
#define PG 32
#ifndef HKV
#define HKV 4  // KV heads per rank (tep4); -DHKV=2 builds tep8
#endif
#define GQ 8
#define R 64
#define S 8
#define PAGES 9
#define KEYS (PAGES * PG)
#define RM (R / S)
#define NPART S
#define WINDOW 2048
#define ROWF 68
#define NT 320
#define NTM 256  // the attention warps 0-7; warps 8-9 prep the step's new keys and values
#define SK_OFF 0                         // K: FP8 [288][64 B]
#define SV_OFF (KEYS * D)                // V: f16 [288][128 B]
#define SQ_OFF (SV_OFF + KEYS * D * 2)   // Q: FP8 [64][64 B]
#define RECV_OFF (SQ_OFF + R * D)
#define SIDS_OFF (RECV_OFF + NPART * RM * ROWF * 4)
#define WID_OFF (SIDS_OFF + 128)
#define CS_OFF (WID_OFF + 128)
#define SCR_OFF (CS_OFF + 8 * 64 * 4)          // key-half combine scratch [4][36][32] floats
#define BAR_OFF (SCR_OFF + 4 * 36 * 32 * 4)
#define RECV_BYTES (S * RM * 66 * 4)            // 8 senders x 8 rows x (64 O + m + l)
#define NEG_INF __int_as_float(0xff800000)
#ifdef CL8_POS64
typedef long long pos_t;  // int64 position ids (a compile-time variant: a runtime select costs ~2 us here)
#else
typedef int pos_t;
#endif
#ifndef CL8_ABL
#define CL8_ABL 0
#endif
// The QKV projection's output (q, k, v and the gate view) is written by the PDL predecessor while
// this grid already runs. Declared `const __restrict__`, it is data the compiler may treat as
// invariant for the whole kernel: a build with that qualifier loaded the gate through the
// read-only path (LDG.E.CONSTANT) ABOVE griddepcontrol.wait (ACQBULK) and so read the previous
// layer's gate in the engine (GSM8K 2/400). These two pointers therefore carry no __restrict__
// (plain LDG after ACQBULK, checked in the SASS).
#define LDP(p) (*(p))  // a load before the dependency wait
#ifdef CL8_INSTR
#define TSTAMP(k) do { __syncthreads(); if (tid == 0) { u64 t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); \
    tbuf[(blockIdx.x + 8 * (blockIdx.y + HKV * blockIdx.z)) * 16 + (k)] = t_; } } while (0)
#define TSTAMPM(k) do { asm volatile("bar.sync 1, 256;" ::: "memory"); if (tid == 0) { u64 t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); \
    tbuf[(blockIdx.x + 8 * (blockIdx.y + HKV * blockIdx.z)) * 16 + (k)] = t_; } } while (0)
#else
#define TSTAMP(k) do {} while (0)
#define TSTAMPM(k) do {} while (0)
#endif

__device__ __forceinline__ u32 sm32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ u32 swz(int row, int chunk) { return (u32)(row * 128 + ((chunk ^ (row & 7)) << 4)); }
// 64-byte rows (FP8 tiles): 16-byte chunk XOR (row / 2) % 4, conflict-free for ldmatrix's 8 rows
__device__ __forceinline__ u32 swz8(int row, int chunk) { return (u32)(row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4)); }
__device__ __forceinline__ u32 ex2h2(u32 x) { u32 r; asm("ex2.approx.f16x2 %0, %1;" : "=r"(r) : "r"(x)); return r; }
__device__ __forceinline__ void mma16832(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float f_add(float a, float b) { float r; asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_mul(float a, float b) { float r; asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_mul_ftz(float a, float b) { float r; asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float f_fma_ftz(float a, float b, float c) { float r; asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r; }
__device__ __forceinline__ float f_rsqrt(float x) { float r; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
__device__ __forceinline__ float f_ex2(float x) { float r; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
__device__ __forceinline__ u16 f2bf(float x) { u16 h; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(x)); return h; }
__device__ __forceinline__ float bf2f(u32 h) { return __uint_as_float((h & 0xffffu) << 16); }
__device__ __forceinline__ float bf16r(float x) { return bf2f(f2bf(x)); }
// two fp32 -> two E4M3 bytes, a in the low byte (RN, satfinite)
__device__ __forceinline__ u32 e4m3x2(float a, float b) { u16 r; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(b), "f"(a)); return (u32)r; }
// two E4M3 bytes (low byte first) -> f16x2 (low half first); exact
__device__ __forceinline__ u32 e4m3x2_f16x2(u32 v) { u32 r; u16 h = (u16)v; asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(h)); return r; }
// two fp32 -> f16x2 (lo in the low half)
__device__ __forceinline__ u32 f16x2(float lo, float hi) { u32 r; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
__device__ __forceinline__ void ldsm4(u32 a, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldsm4t(u32 a, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void mma16816(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void st_async2(u32 addr, float a, float b, u32 rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];" :: "r"(addr), "f"(a), "f"(b), "r"(rbar) : "memory");
}
__device__ __forceinline__ u32 mapa(u32 a, u32 rank) { u32 r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank)); return r; }
__device__ __forceinline__ void st_cluster2(u32 addr, float a, float b) {
  asm volatile("st.shared::cluster.v2.f32 [%0], {%1, %2};" :: "r"(addr), "f"(a), "f"(b) : "memory");
}

// preattn_prep's norm (+ scale) and RoPE of one thread's 16 dims (quarter `qt` = lanes 2qt, 2qt+1 of a 64-wide
// head). The head's four quarters are lanes 4k .. 4k+3 of the warp (lane ^ 1, lane ^ 2). y: BF16 values as fp32.
__device__ __forceinline__ void prep16(const float* x, float* y, float eps, float scale, const float* cs_row, int qt,
                                       int rope) {
  float sl[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    float s = f_add(f_mul(x[8 * j], x[8 * j]), f_mul(x[8 * j + 1], x[8 * j + 1]));
#pragma unroll
    for (int e = 2; e < 8; ++e) s = f_add(s, f_mul(x[8 * j + e], x[8 * j + e]));
    sl[j] = s;
  }
  float u = f_add(sl[0], sl[1]);                                 // lane bit 0
  float v = f_add(u, __shfl_xor_sync(0xffffffffu, u, 1));        // lane bit 1
  float total = f_add(v, __shfl_xor_sync(0xffffffffu, v, 2));    // lane bit 2
  float inv = f_rsqrt(f_add(f_mul(total, 0.015625f), eps));
#pragma unroll
  for (int i = 0; i < 16; ++i) y[i] = bf16r(f_mul(bf16r(f_mul(x[i], inv)), scale));
  if (rope) {
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      const float c = cs_row[qt * 8 + p], s = cs_row[32 + qt * 8 + p];
      const float e = y[2 * p], od = y[2 * p + 1];
      y[2 * p] = bf16r(f_fma_ftz(e, c, f_mul_ftz(-od, s)));
      y[2 * p + 1] = bf16r(f_fma_ftz(od, c, f_mul_ftz(e, s)));
    }
  }
}

__device__ __forceinline__ void load16_bf16(const u16* p, float* x) {
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const uint4 v = *reinterpret_cast<const uint4*>(p + 8 * i);
    x[8 * i + 0] = bf2f(v.x); x[8 * i + 1] = bf2f(v.x >> 16);
    x[8 * i + 2] = bf2f(v.y); x[8 * i + 3] = bf2f(v.y >> 16);
    x[8 * i + 4] = bf2f(v.z); x[8 * i + 5] = bf2f(v.z >> 16);
    x[8 * i + 6] = bf2f(v.w); x[8 * i + 7] = bf2f(v.w >> 16);
  }
}

extern "C" __global__ void __launch_bounds__(NT, 1)
swa_cl8(const u16* qkv, long long qkv_stride, u8* __restrict__ pool, const int* __restrict__ bt,
        long long bt_req_stride, long long bt_kv_stride, const int* __restrict__ seq_lens,
        const float* __restrict__ kv_scale, const float* __restrict__ cos_sin, const pos_t* __restrict__ positions,
        float prep_eps, float q_scale, float sm_scale_log2, u16* __restrict__ out, long long out_stride,
        const u16* gate, long long gate_tok_stride, long long gate_head_stride, float gate_eps,
        int M, int q_stride, int k_col0, int v_col0, int rope, int pdl, u64* tbuf) {
  extern __shared__ __align__(128) u8 smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int rank = blockIdx.x, h = blockIdx.y, req = blockIdx.z;
  TSTAMP(0);
#ifdef CL8_INSTR
  if (tid == 0) { u32 sm_; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_)); tbuf[(blockIdx.x + 8 * (blockIdx.y + HKV * blockIdx.z)) * 16 + 9] = sm_; }
#endif

  // ---------------- before the dependency wait: graph inputs and older cache bytes only
  const int L = LDP(seq_lens + req);
  const int past = L - M;
  int lo = past - (WINDOW - 1);
  if (lo < 0) lo = 0;
  const int first_page = (lo >> 5) + rank * PAGES;
  const int key0 = first_page * PG;
  const float scale = LDP(kv_scale);
  const float sb = bf16r(scale);
  const float dequant = __fdiv_rn(1.0f, scale);
  const float qk_scale = f_mul(f_mul(sm_scale_log2, dequant), dequant);
  int* wpid = reinterpret_cast<int*>(smem + WID_OFF);
  float* cs = reinterpret_cast<float*>(smem + CS_OFF);
  const int* btr = bt + (long long)req * bt_req_stride;
  if (tid == 0) {
    const u32 bar = sm32(smem + BAR_OFF);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar) : "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(RECV_BYTES) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // arrive now, wait right before the first push (by then every peer's mbarrier is armed): off the critical path
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  // warps 0-7: the window's cached K/V into registers, each thread loading its own pages' ids (no SMEM hand-off):
  // keys < past only; keys >= L and missing pages are zero; keys in [past, L) are the step's own (warps 8-9)
  uint4 rb[9];
  if (tid < NTM) {
    int pids[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      const int pl = 2 * i + (tid >> 7), kv = pl / PAGES, p = pl % PAGES, page = first_page + p;
      pids[i] = (page * PG < past) ? LDP(btr + kv * bt_kv_stride + page) : -1;
    }
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      const int c = i * NTM + tid, pl = c >> 7, w = c & 127, p = pl % PAGES;
      const int key = (first_page + p) * PG + (w >> 2);
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (pids[i] >= 0 && key < past)
        v = LDP(reinterpret_cast<const uint4*>(pool + ((long long)pids[i] * HKV + h) * (PG * D) + w * 16));
      rb[i] = v;
    }
  } else {
    // warps 8-9: the new rows' page ids and the step's RoPE rows
    const int u = tid - NTM;
    if (u < 16) {
      const int t = u >> 1, kv = u & 1, pos = past + t, page = pos >> 5;
      int pid = -1;
      if (t < M && page >= first_page && page < first_page + PAGES) pid = LDP(btr + kv * bt_kv_stride + page);
      wpid[2 * t + kv] = pid;
    }
    if (rope) {
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int i = k * 64 + u, t = i >> 6, c = i & 63;
        float v = 0.f;
        if (t < M) v = LDP(cos_sin + (long long)LDP(positions + req * M + t) * 64 + c);
        cs[i] = v;
      }
    }
  }
  TSTAMP(1);
  // ---------------- the dependency wait: the QKV projection (q, k, v and the gate) from here on
  if (pdl) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
  __syncthreads();  // wpid / RoPE rows
  TSTAMP(2);
  // gate of the token this CTA merges (token `rank`): warp = query head, two columns per lane
  float gv0 = 0.f, gv1 = 0.f;
  if (rank < M && warp < 8) {
    const u32 g2 = *reinterpret_cast<const u32*>(gate + (long long)(req * M + rank) * gate_tok_stride +
                                                 (long long)(h * GQ + warp) * gate_head_stride + 2 * lane);
    gv0 = bf2f(g2); gv1 = bf2f(g2 >> 16);
  }
  // query rows: row = token * 8 + query head, four threads per row (16 dims each). The q loads are issued first,
  // the window's K/V tiles are written while they are in flight, then the prep runs.
  if (warp < 8) {
    const int row = tid >> 2, qt = tid & 3, t = row >> 3, qh = h * GQ + (row & 7);
    float x[16], y[16];
    if (t < M) {
      load16_bf16(qkv + (long long)(req * M + t) * qkv_stride + (long long)qh * q_stride + qt * 16, x);
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) x[i] = 0.f;
    }
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      const int c = i * NTM + tid, pl = c >> 7, w = c & 127, kv = pl / PAGES, p = pl % PAGES;
      const int krow = p * PG + (w >> 2), part = w & 3;
      const int key = key0 + krow;
      if (key >= past && key < L) continue;  // the step's own keys: warps 8-9 write them
      const uint4 v = rb[i];
      if (kv == 0) {
        *reinterpret_cast<uint4*>(smem + SK_OFF + swz8(krow, part)) = v;
      } else {
        uint4 a, b;
        a.x = e4m3x2_f16x2(v.x); a.y = e4m3x2_f16x2(v.x >> 16); a.z = e4m3x2_f16x2(v.y); a.w = e4m3x2_f16x2(v.y >> 16);
        b.x = e4m3x2_f16x2(v.z); b.y = e4m3x2_f16x2(v.z >> 16); b.z = e4m3x2_f16x2(v.w); b.w = e4m3x2_f16x2(v.w >> 16);
        *reinterpret_cast<uint4*>(smem + SV_OFF + swz(krow, 2 * part)) = a;
        *reinterpret_cast<uint4*>(smem + SV_OFF + swz(krow, 2 * part + 1)) = b;
      }
    }
    if (CL8_ABL == 5) {
#pragma unroll
      for (int i = 0; i < 16; ++i) y[i] = x[i];
    } else {
      prep16(x, y, prep_eps, q_scale, cs + (t < M ? t : 0) * 64, qt, rope);
    }
    u32 q8[4];
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      float a = 0.f, b = 0.f;
      if (t < M) { a = bf16r(f_mul(y[2 * p], sb)); b = bf16r(f_mul(y[2 * p + 1], sb)); }
      const u32 e = e4m3x2(a, b);
      if (p & 1) q8[p >> 1] |= e << 16; else q8[p >> 1] = e;
    }
    *reinterpret_cast<uint4*>(smem + SQ_OFF + swz8(row, qt)) = make_uint4(q8[0], q8[1], q8[2], q8[3]);
  }
  // the step's keys (warp 8) and values (warp 9) on this CTA's pages, four lanes per token
  // the step's rows are consecutive positions on at most two pages: this CTA holds one iff it holds the first or last
  if (warp >= 8 && (wpid[0] >= 0 || wpid[2 * (M - 1)] >= 0)) {
    const int kvsel = warp - 8, t = lane >> 2, qt = lane & 3;
    const int pid = (t < M) ? wpid[2 * t + kvsel] : -1;
    float x[16], y[16];
    if (pid >= 0) {
      load16_bf16(qkv + (long long)(req * M + t) * qkv_stride + (kvsel ? v_col0 : k_col0) + h * D + qt * 16, x);
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) x[i] = 0.f;
    }
    prep16(x, y, prep_eps, 1.0f, cs + t * 64, qt, rope);  // keys; the value warp discards it
    if (pid >= 0) {
      u32 e8[4], w[8];
#pragma unroll
      for (int p = 0; p < 8; ++p) {
        const float a = bf16r(f_mul(kvsel ? x[2 * p] : y[2 * p], sb));
        const float b = bf16r(f_mul(kvsel ? x[2 * p + 1] : y[2 * p + 1], sb));
        const u32 e = e4m3x2(a, b);
        if (p & 1) e8[p >> 1] |= e << 16; else e8[p >> 1] = e;
        w[p] = e4m3x2_f16x2(e);
      }
      const int pos = past + t;
      u8* dst = pool + ((long long)pid * HKV + h) * (PG * D) + (pos & 31) * D + qt * 16;
      *reinterpret_cast<uint4*>(dst) = make_uint4(e8[0], e8[1], e8[2], e8[3]);
      const int row = pos - key0;
      if (kvsel == 0) {
        *reinterpret_cast<uint4*>(smem + SK_OFF + swz8(row, qt)) = make_uint4(e8[0], e8[1], e8[2], e8[3]);
      } else {
        *reinterpret_cast<uint4*>(smem + SV_OFF + swz(row, 2 * qt)) = make_uint4(w[0], w[1], w[2], w[3]);
        *reinterpret_cast<uint4*>(smem + SV_OFF + swz(row, 2 * qt + 1)) = make_uint4(w[4], w[5], w[6], w[7]);
      }
    }
  }
  __syncthreads();
  TSTAMP(3);

  if (warp < 8) {  // ======== attention warps
  // ---------------- S = Q K^T: warp w = row block rb (rows 16rb.., tokens 2rb, 2rb+1) x key half kh (144 keys)
  const int rb = warp & 3, kh = warp >> 2;
  const int g = lane >> 2, c4 = lane & 3, mi = lane >> 3, rin = lane & 7;  // codespell:ignore
  u32 qa[2][4];
  {
    const u32 sq = sm32(smem + SQ_OFF);
    const int row = rb * 16 + (mi & 1) * 8 + rin;  // codespell:ignore
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) ldsm4(sq + swz8(row, 2 * kk + (mi >> 1)), qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3]);
  }
  const int t_lo = 2 * rb, t_hi = t_lo + 1;
  const int qp_lo = past + t_lo, qp_hi = past + t_hi;
  // the keys row t attends: [max(lo, qp - WINDOW + 1), qp] (empty for t >= M)
  const int jlo_lo = max(lo, qp_lo - WINDOW + 1), jhi_lo = t_lo < M ? qp_lo : -1;
  const int jlo_hi = max(lo, qp_hi - WINDOW + 1), jhi_hi = t_hi < M ? qp_hi : -1;
  const int jhi_u = max(jhi_lo, jhi_hi);  // the union of both rows' ranges is [jlo_lo, jhi_u]
  const int kbase = key0 + kh * 144;
  float sa[18][4];
#pragma unroll
  for (int n = 0; n < 18; ++n) sa[n][0] = sa[n][1] = sa[n][2] = sa[n][3] = 0.f;
  if (CL8_ABL != 1) {
    const u32 sk = sm32(smem + SK_OFF);
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
      for (int n = 0; n < 18; n += 2) {
        // matrices: (tile n, dims 32kk..+15), (tile n, +16..31), (tile n+1, ..), (tile n+1, ..): FP8 bytes
        const int krow = kh * 144 + 8 * (n + (mi >> 1)) + rin;  // codespell:ignore
        u32 b0, b1, b2, b3;
        ldsm4(sk + swz8(krow, 2 * kk + (mi & 1)), b0, b1, b2, b3);
        mma16832(sa[n], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b0, b1);
        mma16832(sa[n + 1], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b2, b3);
      }
    }
  }
  TSTAMPM(4);
  // mask (only tiles that straddle a row's range), softmax over the warp's 144 keys; the scale (> 0) is folded
  // into the exponent: p = 2^(s * qk_scale - m * qk_scale)
  float m_lo = NEG_INF, m_hi = NEG_INF;
#pragma unroll
  for (int n = 0; n < 18; ++n) {
    const int jt = kbase + 8 * n;
    const bool in_lo = jt >= jlo_lo && jt + 7 <= jhi_lo, in_hi = jt >= jlo_hi && jt + 7 <= jhi_hi;
    if (!(in_lo && in_hi)) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int j = jt + 2 * c4 + e;
        if ((unsigned)(j - jlo_lo) > (unsigned)(jhi_lo - jlo_lo)) sa[n][e] = NEG_INF;
        if ((unsigned)(j - jlo_hi) > (unsigned)(jhi_hi - jlo_hi)) sa[n][2 + e] = NEG_INF;
      }
    }
    m_lo = fmaxf(m_lo, fmaxf(sa[n][0], sa[n][1]));
    m_hi = fmaxf(m_hi, fmaxf(sa[n][2], sa[n][3]));
  }
  m_lo = fmaxf(m_lo, __shfl_xor_sync(0xffffffffu, m_lo, 1)); m_lo = fmaxf(m_lo, __shfl_xor_sync(0xffffffffu, m_lo, 2));
  m_hi = fmaxf(m_hi, __shfl_xor_sync(0xffffffffu, m_hi, 1)); m_hi = fmaxf(m_hi, __shfl_xor_sync(0xffffffffu, m_hi, 2));
  // ONE2's m is max(-1e30, max(s * scale)); here m is kept in raw units and scaled once
  const float ms_lo = fmaxf(m_lo * qk_scale, -1.0e30f), ms_hi = fmaxf(m_hi * qk_scale, -1.0e30f);
  // P = 2^(s * scale - m) in f16 straight from the MUFU (ex2.approx.f16x2: two per op); the row sums come from the
  // P.V MMA itself (a ones column), so they are the sums of exactly the P values that weigh V
  u32 pl[18], ph[18];
#pragma unroll
  for (int n = 0; n < 18; ++n) {
    const u32 xl = f16x2(fmaf(sa[n][0], qk_scale, -ms_lo), fmaf(sa[n][1], qk_scale, -ms_lo));
    const u32 xh = f16x2(fmaf(sa[n][2], qk_scale, -ms_hi), fmaf(sa[n][3], qk_scale, -ms_hi));
    pl[n] = (CL8_ABL == 3) ? xl : ex2h2(xl);
    ph[n] = (CL8_ABL == 3) ? xh : ex2h2(xh);
  }
  const float m_lo_out = ms_lo, m_hi_out = ms_hi;
  TSTAMPM(5);
  // ---------------- O = P V over the warp's keys, plus l = P . 1 in an extra n8 tile (column 64)
  float oa[8][4], la[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int n = 0; n < 8; ++n) oa[n][0] = oa[n][1] = oa[n][2] = oa[n][3] = 0.f;
  {
    const u32 sv = sm32(smem + SV_OFF);
    const u32 ones = (g == 0) ? 0x3C003C00u : 0u;  // B column 0 of the extra tile: two f16 1.0
#pragma unroll
    for (int kk = 0; kk < 9; ++kk) {
      if (CL8_ABL == 1 || CL8_ABL == 2) continue;
      const u32 a0 = pl[2 * kk], a1 = ph[2 * kk], a2 = pl[2 * kk + 1], a3 = ph[2 * kk + 1];
      const int vrow = kh * 144 + 16 * kk + (mi & 1) * 8 + rin;  // codespell:ignore
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        u32 b0, b1, b2, b3;
        ldsm4t(sv + swz(vrow, 2 * q + (mi >> 1)), b0, b1, b2, b3);
        mma16816(oa[2 * q], a0, a1, a2, a3, b0, b1);
        mma16816(oa[2 * q + 1], a0, a1, a2, a3, b2, b3);
      }
      mma16816(la, a0, a1, a2, a3, ones, ones);
    }
  }
  // row sums live in column 64 (lanes with c4 == 0): give them to the whole quad
  const float l_lo = __shfl_sync(0xffffffffu, la[0], lane & ~3), l_hi = __shfl_sync(0xffffffffu, la[2], lane & ~3);
  TSTAMPM(6);
  // ---------------- combine the two key halves of each row block in the CTA (warp rb + 4 into warp rb)
  float* scr = reinterpret_cast<float*>(smem + SCR_OFF) + rb * 36 * 32 + lane;
  if (kh == 1) {
#pragma unroll
    for (int n = 0; n < 8; ++n) {
#pragma unroll
      for (int e = 0; e < 4; ++e) scr[(4 * n + e) * 32] = oa[n][e];
    }
    scr[32 * 32] = m_lo_out; scr[33 * 32] = m_hi_out; scr[34 * 32] = l_lo; scr[35 * 32] = l_hi;
  }
  asm volatile("bar.sync 1, 256;" ::: "memory");
  if (kh == 0) {
    const float m1_lo = scr[32 * 32], m1_hi = scr[33 * 32], l1_lo = scr[34 * 32], l1_hi = scr[35 * 32];
    const float mlo = fmaxf(m_lo_out, m1_lo), mhi = fmaxf(m_hi_out, m1_hi);
    const float a_lo = f_ex2(m_lo_out - mlo), b_lo = f_ex2(m1_lo - mlo);
    const float a_hi = f_ex2(m_hi_out - mhi), b_hi = f_ex2(m1_hi - mhi);
    const float L_lo = l_lo * a_lo + l1_lo * b_lo, L_hi = l_hi * a_hi + l1_hi * b_hi;
    // push: row lo -> CTA t_lo, row hi -> CTA t_hi (only tokens < M: no CTA waits for the others)
    float* recv = reinterpret_cast<float*>(smem + RECV_OFF);
    const u32 slot = sm32(recv + (rank * RM + g) * ROWF);
    const u32 bar = sm32(smem + BAR_OFF);
    const u32 ra_lo = mapa(slot, t_lo), ra_hi = mapa(slot, t_hi);
    const u32 rb_lo = mapa(bar, t_lo), rb_hi = mapa(bar, t_hi);
    const float dq_lo = dequant, dq_hi = dequant;
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      const float o0 = (oa[n][0] * a_lo + scr[(4 * n + 0) * 32] * b_lo) * dq_lo;
      const float o1 = (oa[n][1] * a_lo + scr[(4 * n + 1) * 32] * b_lo) * dq_lo;
      const float o2 = (oa[n][2] * a_hi + scr[(4 * n + 2) * 32] * b_hi) * dq_hi;
      const float o3 = (oa[n][3] * a_hi + scr[(4 * n + 3) * 32] * b_hi) * dq_hi;
      if (t_lo < M) st_async2(ra_lo + (8 * n + 2 * c4) * 4, o0, o1, rb_lo);
      if (t_hi < M) st_async2(ra_hi + (8 * n + 2 * c4) * 4, o2, o3, rb_hi);
    }
    if (c4 == 0) {
      if (t_lo < M) st_async2(ra_lo + 64 * 4, mlo, L_lo, rb_lo);
      if (t_hi < M) st_async2(ra_hi + 64 * 4, mhi, L_hi, rb_hi);
    }
  }
  }  // ======== attention warps
  TSTAMP(7);
  // ---------------- merge token `rank` once its 8 partials (one per CTA) have landed: query head = warp
  if (rank < M && warp < 8 && CL8_ABL != 6) {
    const u32 bar = sm32(smem + BAR_OFF);
    u32 done = 0;
    while (!done) {
      asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                   : "=r"(done) : "r"(bar) : "memory");
    }
    const float* recv = reinterpret_cast<const float*>(smem + RECV_OFF);
    float ms[NPART], ls[NPART], mmax = NEG_INF;
#pragma unroll
    for (int s = 0; s < NPART; ++s) {
      const float2 ml = *reinterpret_cast<const float2*>(recv + (s * RM + warp) * ROWF + 64);
      ms[s] = ml.x; ls[s] = ml.y;
      mmax = fmaxf(mmax, ms[s]);
    }
    float den = 0.f, n0 = 0.f, n1 = 0.f;
#pragma unroll
    for (int s = 0; s < NPART; ++s) {
      const float w = ls[s] > 0.f ? f_ex2(ms[s] - mmax) : 0.f;
      den += w * ls[s];
      const float2 o = *reinterpret_cast<const float2*>(recv + (s * RM + warp) * ROWF + 2 * lane);
      n0 += o.x * w; n1 += o.y * w;
    }
    const float a0 = bf16r(n0 / den), a1 = bf16r(n1 / den);
    float sq = a0 * a0 + a1 * a1;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    const float inv = f_rsqrt(sq * 0.015625f + gate_eps);
    const float r0 = bf16r(a0 * inv) * bf16r(1.f / (1.f + __expf(-gv0)));
    const float r1 = bf16r(a1 * inv) * bf16r(1.f / (1.f + __expf(-gv1)));
    *reinterpret_cast<u32*>(out + (long long)(req * M + rank) * out_stride + (h * GQ + warp) * D + 2 * lane) =
        (u32)f2bf(r0) | ((u32)f2bf(r1) << 16);
  }
  TSTAMP(8);
}
"""


_CL8_FN = {}
_CL8_WARNED = set()


def _cl8_log(msg: str, warning: bool = False) -> None:
    try:
        from tensorrt_llm.logger import logger

        (logger.warning if warning else logger.info)(msg)
    except Exception:  # noqa: BLE001 - standalone copies load this file without tensorrt_llm
        print(msg, flush=True)


def _cl8_check(r):
    if int(r[0]) != 0:
        raise RuntimeError(f"swa_block CL8: CUDA/NVRTC error {r!r}")
    return r[1] if len(r) == 2 else r[1:]


def _cl8_on(hkv: int, reqs: int) -> bool:
    """Whether ``MUSESPARK_OPEN_SWA_CL8`` takes a step of ``reqs`` requests at ``hkv`` KV heads per rank: TEP4 only; ``all``
    1..``CL8_MAX_REQUESTS`` requests, ``1`` 2..``CL8_MAX_REQUESTS``."""
    return (
        hkv == 4
        and reqs <= CL8_MAX_REQUESTS
        and (_SWA_CL8 == "all" or (_SWA_CL8 == "1" and reqs > 1))
    )


def _cl8_function(device: torch.device, pos64: bool, hkv: int):
    """The compiled CL8 kernel of ``device`` (int64 or int32 position ids: two compile-time builds; ``hkv`` KV heads
    per rank: 4 at tep4, 2 at tep8); ``None`` if it is not compiled yet and a capture is running."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, bool(pos64), int(hkv))
    fn = _CL8_FN.get(key)
    if fn is not None:
        return fn
    if torch.cuda.is_current_stream_capturing():
        if key not in _CL8_WARNED:
            _CL8_WARNED.add(key)
            _cl8_log(
                "[musespark_open] swa_block CL8: first call inside a CUDA-graph capture; this "
                "capture keeps the path CL8 replaces (compile happens on an eager call)",
                warning=True,
            )
        return None
    from cuda.bindings import driver as cu
    from cuda.bindings import nvrtc

    capability = torch.cuda.get_device_capability(index)
    if tuple(capability) != (10, 3):
        raise RuntimeError(
            f"swa_block CL8 is built for sm_103 only, not compute capability {capability}"
        )
    prog = _cl8_check(nvrtc.nvrtcCreateProgram(_CL8_SRC.encode(), b"swa_cl8.cu", 0, [], []))
    opts = [b"--gpu-architecture=sm_103a", b"-std=c++17", b"-default-device"]
    if hkv != 4:
        # The source's HKV defaults to 4 (the TEP4 build); the TEP8 build passes -DHKV=2.
        opts.append(f"-DHKV={int(hkv)}".encode())
    if pos64:
        opts.append(b"-DCL8_POS64")
    r = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    n = _cl8_check(nvrtc.nvrtcGetProgramLogSize(prog))
    log = b" " * n
    nvrtc.nvrtcGetProgramLog(prog, log)
    if int(r[0]) != 0:
        raise RuntimeError("swa_block CL8 NVRTC compile failed:\n" + log.decode(errors="replace"))
    n = _cl8_check(nvrtc.nvrtcGetCUBINSize(prog))
    cubin = b" " * n
    _cl8_check(nvrtc.nvrtcGetCUBIN(prog, cubin))
    with torch.cuda.device(index):
        mod = _cl8_check(cu.cuModuleLoadData(cubin))
        f = _cl8_check(cu.cuModuleGetFunction(mod, b"swa_cl8"))
        _cl8_check(
            cu.cuFuncSetAttribute(
                f,
                cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                _CL8_SMEM,
            )
        )
    regs = int(
        _cl8_check(cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, f))
    )
    lmem = int(
        _cl8_check(
            cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, f)
        )
    )
    _CL8_FN[key] = (f, mod)
    _cl8_log(
        f"[musespark_open] swa_block CL8: MUSESPARK_OPEN_SWA_CL8={_SWA_CL8} compiled for sm_103a on "
        f"device {index} (regs {regs}, local {lmem} B, smem {_CL8_SMEM // 1024} KB, cluster 8, "
        f"320 threads, {int(hkv)} KV heads per rank, {'int64' if pos64 else 'int32'} positions)"
    )
    return _CL8_FN[key]


def _cl8_eligible(
    qkv,
    kv_pool,
    num_q_heads,
    num_kv_heads,
    window,
    rows,
    q_fp8,
    fused_prep,
    gate,
    cos_sin,
    position_ids,
    hkv,
) -> bool:
    """A call the CL8 build of ``hkv`` KV heads per rank serves: ``hkv`` KV / ``8 hkv`` query heads
    per rank (4 / 32 at TEP4, 2 / 16 at TEP8), the kernel's other conditions."""
    return (
        fused_prep
        and gate is not None
        and q_fp8
        and num_kv_heads == hkv
        and num_q_heads == 8 * hkv
        and window == 2048
        and 1 <= rows <= MAX_ROWS
        and kv_pool.dim() == 4
        and tuple(kv_pool.shape[1:]) == (hkv, 32, HEAD_DIM)
        and qkv.stride(0) % 8 == 0
        and qkv.data_ptr() % 16 == 0
        and kv_pool.data_ptr() % 16 == 0
        and (
            cos_sin is None
            or position_ids is None
            or position_ids.dtype in (torch.int32, torch.int64)
        )
    )


def _cl8_launch(
    fn,
    qkv,
    kv_pool,
    bt,
    seq_lens,
    kv_scale,
    cos_sin,
    positions,
    prep_eps,
    q_scale,
    gate,
    gate_eps,
    out,
    qkv_permuted,
    enable_pdl,
    rope,
):
    import ctypes

    from cuda.bindings import driver as cu

    reqs = seq_lens.numel()
    rows = qkv.shape[0] // reqs
    hkv = int(kv_pool.shape[1])
    hq = 8 * hkv
    # the raw projection: [q | gate] per query head (or [q | k | v | gate] permuted), then K and V of the KV heads
    q_stride, k_col0 = (64, hq * 64) if qkv_permuted else (128, 2 * hq * 64)
    v_col0 = k_col0 + hkv * 64
    gts = gate.stride(0)
    ghs = gate.stride(1) if gate.dim() == 3 else HEAD_DIM
    cfg = cu.CUlaunchConfig()
    cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = 8, hkv, reqs
    cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = 320, 1, 1
    cfg.sharedMemBytes = _CL8_SMEM
    cfg.hStream = cu.CUstream(torch.cuda.current_stream(qkv.device).cuda_stream)
    a1 = cu.CUlaunchAttribute()
    a1.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    a1.value.clusterDim.x, a1.value.clusterDim.y, a1.value.clusterDim.z = 8, 1, 1
    attrs = [a1]
    if enable_pdl:
        a2 = cu.CUlaunchAttribute()
        a2.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        a2.value.programmaticStreamSerializationAllowed = 1
        attrs.append(a2)
    cfg.attrs = attrs
    cfg.numAttrs = len(attrs)
    vp, ll, fl, ii = ctypes.c_void_p, ctypes.c_longlong, ctypes.c_float, ctypes.c_int
    args = (
        (
            qkv.data_ptr(),
            qkv.stride(0),
            kv_pool.data_ptr(),
            bt.data_ptr(),
            bt.stride(0),
            bt.stride(1),
            seq_lens.data_ptr(),
            kv_scale.data_ptr(),
            cos_sin.data_ptr() if rope else 0,
            positions.data_ptr() if rope else 0,
            float(prep_eps),
            float(q_scale),
            math.log2(math.e) / 8.0,
            out.data_ptr(),
            out.stride(0),
            gate.data_ptr(),
            gts,
            ghs,
            float(gate_eps),
            rows,
            q_stride,
            k_col0,
            v_col0,
            int(rope),
            int(bool(enable_pdl)),
            0,
        ),
        (
            vp,
            ll,
            vp,
            vp,
            ll,
            ll,
            vp,
            vp,
            vp,
            vp,
            fl,
            fl,
            fl,
            vp,
            ll,
            vp,
            ll,
            ll,
            fl,
            ii,
            ii,
            ii,
            ii,
            ii,
            ii,
            vp,
        ),
    )
    _cl8_check(cu.cuLaunchKernelEx(cfg, fn[0], args, 0))
    return out


@triton.jit
def _prep_rows(
    ptr,
    row_off,
    col_off,
    mask,
    cos,
    sin,
    scale,
    eps,
    N: tl.constexpr,
    APPLY_ROPE: tl.constexpr,
):
    """``preattn_prep`` of ``N`` 64-wide heads of the raw projection, as BF16 ``[N, 64]``.

    Head ``n`` starts at ``ptr + row_off[n] + col_off[n]``. Thread ``l`` of a head
    owns elements ``8 l .. 8 l + 7``; its squares are summed in order with
    ``add.rn``, then threads pair over bit 0, 1, 2 -- ``preattn_prep``'s tree.
    The norm, the scale and the rotation are its elementwise statements on the
    same values, so each element rounds exactly as there. Masked heads read 0.
    """
    d = tl.arange(0, 64)
    lane = tl.arange(0, 8)
    head = ptr + row_off[:, None] + col_off[:, None]
    m = mask[:, None]
    x0 = tl.load(head + lane[None, :] * 8 + 0, mask=m, other=0.0).to(tl.float32)
    x1 = tl.load(head + lane[None, :] * 8 + 1, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(head + lane[None, :] * 8 + 2, mask=m, other=0.0).to(tl.float32)
    x3 = tl.load(head + lane[None, :] * 8 + 3, mask=m, other=0.0).to(tl.float32)
    x4 = tl.load(head + lane[None, :] * 8 + 4, mask=m, other=0.0).to(tl.float32)
    x5 = tl.load(head + lane[None, :] * 8 + 5, mask=m, other=0.0).to(tl.float32)
    x6 = tl.load(head + lane[None, :] * 8 + 6, mask=m, other=0.0).to(tl.float32)
    x7 = tl.load(head + lane[None, :] * 8 + 7, mask=m, other=0.0).to(tl.float32)
    s = libdevice.add_rn(x0 * x0, x1 * x1)
    s = libdevice.add_rn(s, x2 * x2)
    s = libdevice.add_rn(s, x3 * x3)
    s = libdevice.add_rn(s, x4 * x4)
    s = libdevice.add_rn(s, x5 * x5)
    s = libdevice.add_rn(s, x6 * x6)
    s = libdevice.add_rn(s, x7 * x7)
    # lane = 4a + 2b + c: pair over c (bit 0), then b, then a.
    s = tl.reshape(s, [N, 2, 2, 2])
    total = tl.sum(tl.sum(tl.sum(s, axis=3), axis=2), axis=1)
    inverse_rms = _rsqrt_approx_ftz(total / 64.0 + eps)
    x = tl.load(head + d[None, :], mask=m, other=0.0).to(tl.float32)
    y = _normalize(x, inverse_rms[:, None], scale)
    if APPLY_ROPE:
        even, odd = tl.split(tl.reshape(y, [N, 32, 2]))
        even, odd = _rotate(even, odd, cos, sin)
        y = tl.reshape(tl.join(even, odd), [N, 64])
    return y.to(tl.bfloat16)


@triton.jit
def _rope_rows(cos_sin_ptr, positions_ptr, index, mask, APPLY_ROPE: tl.constexpr):
    """``(cos, sin)`` ``[N, 32]`` of the positions ``positions[index]`` (zeros where masked)."""
    half = tl.arange(0, 32)
    if APPLY_ROPE:
        position = tl.load(positions_ptr + index, mask=mask, other=0).to(tl.int64)
        row = cos_sin_ptr + position[:, None] * 64 + half[None, :]
        cos = tl.load(row, mask=mask[:, None], other=0.0)
        sin = tl.load(row + 32, mask=mask[:, None], other=0.0)
    else:
        cos = tl.zeros([index.shape[0], 32], tl.float32)
        sin = tl.zeros([index.shape[0], 32], tl.float32)
    return cos, sin


@triton.jit
def _swa_split_kernel(
    qkv_ptr,
    qkv_stride,
    pool_ptr,
    bt_ptr,
    bt_stride_kv,
    seq_len_ptr,
    scale_ptr,
    cos_sin_ptr,
    positions_ptr,
    prep_eps,
    q_scale,
    part_o_ptr,
    part_ml_ptr,
    sm_scale_log2,
    num_splits,
    M: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
    R: tl.constexpr,
    Q_FP8: tl.constexpr,
    FUSED_PREP: tl.constexpr,
    QKV_PERMUTED: tl.constexpr,
    APPLY_ROPE: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    REQS: tl.constexpr = 1,
):
    G: tl.constexpr = HQ // HKV
    PAGE: tl.constexpr = HKV * P * D
    split = tl.program_id(0)
    h = tl.program_id(1)
    row_group = tl.program_id(2)
    if REQS > 1:
        # Several sequences of M rows each, request-major in the operand: program
        # (split, h, req * RG + row_group) is the one-sequence program of request
        # ``req`` on its own rows, length, block-table row, positions and partials.
        RG: tl.constexpr = (M * G + R - 1) // R
        req = row_group // RG
        row_group = row_group % RG
        qkv_ptr += req * M * qkv_stride
        bt_ptr += req * 2 * bt_stride_kv  # [requests, 2, pages], K then V rows
        seq_len_ptr += req
        if APPLY_ROPE:
            positions_ptr += req * M
        part_o_ptr += req * num_splits * HKV * M * G * D
        part_ml_ptr += req * num_splits * HKV * M * G * 2
    d = tl.arange(0, D)

    # Column offsets of this program's heads in the operand: the prep output
    # [q | k | v], or the raw projection (checkpoint order [q_0 | gate_0 | ... |
    # k | v], or the low-latency layout [q | k | v | gate]).
    q_stride = D
    k_col = HQ * D + h * D
    v_col = (HQ + HKV) * D + h * D
    if FUSED_PREP:
        if not QKV_PERMUTED:
            q_stride = 2 * D
            k_col = 2 * HQ * D + h * D
            v_col = 2 * HQ * D + HKV * D + h * D

    # ---- Before the wait: graph inputs and cache bytes no grid in flight writes.
    # seq_lens, the block table and the positions are written before the graph;
    # the cached keys (j < past) were written by earlier steps; this step's own
    # keys (j >= past) are written by this grid's (0, h, 0) program only.
    L = tl.load(seq_len_ptr)
    past = L - M
    if WINDOW > 0:
        lo = tl.maximum(past - WINDOW + 1, 0)
    else:
        lo = 0
    base = (lo // BLOCK) * BLOCK
    n_blocks = tl.cdiv(L - base, BLOCK)
    per_split = tl.cdiv(n_blocks, num_splits)
    b0 = split * per_split
    b1 = tl.minimum(b0 + per_split, n_blocks)
    scale = tl.load(scale_ptr)
    scale_bf16 = scale.to(tl.bfloat16)
    dequant = 1.0 / scale

    j = base + b0 * BLOCK + tl.arange(0, BLOCK)
    live = b0 < b1
    # Never touch a page outside the attended range: a sliding-window layer's
    # block table need not hold a valid page there.
    cached = (j >= lo) & (j < past) & live
    new = (j >= past) & (j < L) & live
    page_k = tl.load(bt_ptr + j // P, mask=cached, other=0).to(tl.int64)
    page_v = tl.load(bt_ptr + bt_stride_kv + j // P, mask=cached, other=0).to(tl.int64)
    in_page = (h * P + j % P) * D
    k_old = tl.load(
        pool_ptr + (page_k * PAGE + in_page)[:, None] + d[None, :],
        mask=cached[:, None],
        other=0.0,
    )
    v_old = tl.load(
        pool_ptr + (page_v * PAGE + in_page)[:, None] + d[None, :],
        mask=cached[:, None],
        other=0.0,
    )

    writer = (split == 0) & (row_group == 0)
    tok = tl.arange(0, 8)
    tok_ok = tok < M
    write_ok = tok_ok & writer
    pos = past + tok
    w_page_k = tl.load(bt_ptr + pos // P, mask=write_ok, other=0).to(tl.int64)
    w_page_v = tl.load(bt_ptr + bt_stride_kv + pos // P, mask=write_ok, other=0).to(tl.int64)
    slot = (h * P + pos % P) * D

    r = row_group * R + tl.arange(0, R)  # row = token * G + query head within the KV group
    t = r // G
    row_ok = t < M
    new_row = j - past
    if FUSED_PREP:
        cos_q, sin_q = _rope_rows(cos_sin_ptr, positions_ptr, t, row_ok, APPLY_ROPE)
        cos_k, sin_k = _rope_rows(cos_sin_ptr, positions_ptr, new_row, new, APPLY_ROPE)
        cos_w, sin_w = _rope_rows(cos_sin_ptr, positions_ptr, tok, write_ok, APPLY_ROPE)

    if LAUNCH_WITH_PDL:
        # From here on the operand: the previous kernel's output. The combine
        # kernel may launch now and waits for this grid.
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    if writer:
        # The step's K/V into the paged cache: what applyBiasRopeUpdateKVCacheV2 stores.
        if FUSED_PREP:
            k_new = _prep_rows(
                qkv_ptr,
                tok * qkv_stride,
                tl.zeros([8], tl.int32) + k_col,
                tok_ok,
                cos_w,
                sin_w,
                1.0,
                prep_eps,
                8,
                APPLY_ROPE,
            )
        else:
            k_new = tl.load(
                qkv_ptr + tok[:, None] * qkv_stride + k_col + d[None, :],
                mask=tok_ok[:, None],
                other=0.0,
            )
        v_new = tl.load(
            qkv_ptr + tok[:, None] * qkv_stride + v_col + d[None, :],
            mask=tok_ok[:, None],
            other=0.0,
        )
        tl.store(
            pool_ptr + (w_page_k * PAGE + slot)[:, None] + d[None, :],
            _to_cache_fp8(k_new, scale_bf16),
            mask=tok_ok[:, None],
        )
        tl.store(
            pool_ptr + (w_page_v * PAGE + slot)[:, None] + d[None, :],
            _to_cache_fp8(v_new, scale_bf16),
            mask=tok_ok[:, None],
        )

    q_head = h * G + r % G
    if FUSED_PREP:
        q = _prep_rows(
            qkv_ptr,
            t * qkv_stride,
            q_head * q_stride,
            row_ok,
            cos_q,
            sin_q,
            q_scale,
            prep_eps,
            R,
            APPLY_ROPE,
        )
        q = tl.where(row_ok[:, None], q, 0.0)
    else:
        q = tl.load(
            qkv_ptr + t[:, None] * qkv_stride + q_head[:, None] * D + d[None, :],
            mask=row_ok[:, None],
            other=0.0,
        )
    qk_scale = sm_scale_log2 * dequant
    if Q_FP8:
        q = _to_cache_fp8(q, scale_bf16).to(tl.bfloat16)
        qk_scale = qk_scale * dequant
    q_pos = past + t
    m_i = tl.full([R], -1.0e30, tl.float32)
    l_i = tl.zeros([R], tl.float32)
    acc = tl.zeros([R, D], tl.float32)
    # Whether this program's key block holds any of the step's own keys.
    has_new = base + b0 * BLOCK + BLOCK > past
    for b in range(b0, b1):  # at most once: one key block per program
        if FUSED_PREP:
            k_new = tl.zeros([BLOCK, D], tl.bfloat16)
            if has_new:
                k_new = _prep_rows(
                    qkv_ptr,
                    new_row * qkv_stride,
                    tl.zeros([BLOCK], tl.int32) + k_col,
                    new,
                    cos_k,
                    sin_k,
                    1.0,
                    prep_eps,
                    BLOCK,
                    APPLY_ROPE,
                )
        else:
            k_new = tl.load(
                qkv_ptr + new_row[:, None] * qkv_stride + k_col + d[None, :],
                mask=new[:, None],
                other=0.0,
            )
        v_new = tl.load(
            qkv_ptr + new_row[:, None] * qkv_stride + v_col + d[None, :],
            mask=new[:, None],
            other=0.0,
        )
        k = tl.where(new[:, None], _to_cache_fp8(k_new, scale_bf16), k_old)
        v = tl.where(new[:, None], _to_cache_fp8(v_new, scale_bf16), v_old)
        s = tl.dot(q, tl.trans(k.to(tl.bfloat16))) * qk_scale
        keep = (j[None, :] <= q_pos[:, None]) & (j[None, :] >= lo)
        if WINDOW > 0:
            keep = keep & (j[None, :] > q_pos[:, None] - WINDOW)
        s = tl.where(keep, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
        m_i = m_new
    # An empty split (b0 >= b1) stores l = 0, which the combine weighs as zero.
    part = (split * HKV + h) * (M * G) + r
    tl.store(part_o_ptr + part[:, None] * D + d[None, :], acc * dequant, mask=row_ok[:, None])
    tl.store(part_ml_ptr + part * 2, m_i, mask=row_ok)
    tl.store(part_ml_ptr + part * 2 + 1, l_i, mask=row_ok)


@triton.jit
def _block_kv(
    pool_ptr,
    bt_ptr,
    bt_stride_kv,
    h,
    base,
    blk,
    n_blocks,
    lo,
    past,
    L,
    D: tl.constexpr,
    P: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Key block ``blk``'s keys ``j``, its step-key mask and its cached K/V (zeros elsewhere).

    ``_swa_split_kernel``'s pre-wait statements for a program whose block is ``blk``: the
    block-table entries and the FP8 K/V of the cached keys ``lo <= j < past``; a block at or
    past ``n_blocks`` is empty and loads nothing.
    """
    d = tl.arange(0, D)
    j = base + blk * BLOCK + tl.arange(0, BLOCK)
    live = blk < n_blocks
    cached = (j >= lo) & (j < past) & live
    new = (j >= past) & (j < L) & live
    page_k = tl.load(bt_ptr + j // P, mask=cached, other=0).to(tl.int64)
    page_v = tl.load(bt_ptr + bt_stride_kv + j // P, mask=cached, other=0).to(tl.int64)
    in_page = (h * P + j % P) * D
    k_old = tl.load(
        pool_ptr + (page_k * PAGE + in_page)[:, None] + d[None, :],
        mask=cached[:, None],
        other=0.0,
    )
    v_old = tl.load(
        pool_ptr + (page_v * PAGE + in_page)[:, None] + d[None, :],
        mask=cached[:, None],
        other=0.0,
    )
    return j, new, k_old, v_old


@triton.jit
def _block_attend(
    m_i,
    l_i,
    acc,
    q,
    qk_scale,
    q_pos,
    j,
    new,
    k_old,
    v_old,
    blk,
    n_blocks,
    base,
    lo,
    past,
    qkv_ptr,
    qkv_stride,
    k_col,
    v_col,
    cos_sin_ptr,
    positions_ptr,
    prep_eps,
    scale_bf16,
    D: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
    FUSED_PREP: tl.constexpr,
    APPLY_ROPE: tl.constexpr,
):
    """One key block's online-softmax update of ``(m_i, l_i, acc)``: ``_swa_split_kernel``'s
    key loop for a program whose block is ``blk``, statement for statement (the step's keys
    of the block are prepared from the projection here; their RoPE rows are read here too,
    which is after the wait and the same values)."""
    d = tl.arange(0, D)
    new_row = j - past
    # Whether this key block holds any of the step's own keys.
    has_new = base + blk * BLOCK + BLOCK > past
    for b in range(blk, tl.minimum(blk + 1, n_blocks)):  # at most once
        if FUSED_PREP:
            k_new = tl.zeros([BLOCK, D], tl.bfloat16)
            if has_new:
                cos_k, sin_k = _rope_rows(cos_sin_ptr, positions_ptr, new_row, new, APPLY_ROPE)
                k_new = _prep_rows(
                    qkv_ptr,
                    new_row * qkv_stride,
                    tl.zeros([BLOCK], tl.int32) + k_col,
                    new,
                    cos_k,
                    sin_k,
                    1.0,
                    prep_eps,
                    BLOCK,
                    APPLY_ROPE,
                )
        else:
            k_new = tl.load(
                qkv_ptr + new_row[:, None] * qkv_stride + k_col + d[None, :],
                mask=new[:, None],
                other=0.0,
            )
        v_new = tl.load(
            qkv_ptr + new_row[:, None] * qkv_stride + v_col + d[None, :],
            mask=new[:, None],
            other=0.0,
        )
        k = tl.where(new[:, None], _to_cache_fp8(k_new, scale_bf16), k_old)
        v = tl.where(new[:, None], _to_cache_fp8(v_new, scale_bf16), v_old)
        s = tl.dot(q, tl.trans(k.to(tl.bfloat16))) * qk_scale
        keep = (j[None, :] <= q_pos[:, None]) & (j[None, :] >= lo)
        if WINDOW > 0:
            keep = keep & (j[None, :] > q_pos[:, None] - WINDOW)
        s = tl.where(keep, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
        m_i = m_new
    return m_i, l_i, acc


@triton.jit
def _store_partial(
    part_o_ptr,
    part_ml_ptr,
    slot,
    slots,
    h,
    r,
    row_ok,
    acc,
    m_i,
    l_i,
    dequant,
    HKV: tl.constexpr,
    MG: tl.constexpr,
    D: tl.constexpr,
):
    """Store one partial at ``slot`` (nothing at or past ``slots``), as the split kernel does."""
    d = tl.arange(0, D)
    ok = row_ok & (slot < slots)
    part = (slot * HKV + h) * MG + r
    tl.store(part_o_ptr + part[:, None] * D + d[None, :], acc * dequant, mask=ok[:, None])
    tl.store(part_ml_ptr + part * 2, m_i, mask=ok)
    tl.store(part_ml_ptr + part * 2 + 1, l_i, mask=ok)


@triton.jit
def _swa_split_bpp_kernel(
    qkv_ptr,
    qkv_stride,
    pool_ptr,
    bt_ptr,
    bt_stride_kv,
    seq_len_ptr,
    scale_ptr,
    cos_sin_ptr,
    positions_ptr,
    prep_eps,
    q_scale,
    part_o_ptr,
    part_ml_ptr,
    sm_scale_log2,
    num_splits,
    M: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
    R: tl.constexpr,
    Q_FP8: tl.constexpr,
    FUSED_PREP: tl.constexpr,
    QKV_PERMUTED: tl.constexpr,
    APPLY_ROPE: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    BPP: tl.constexpr,
    REQS: tl.constexpr = 1,
    MERGE: tl.constexpr = False,
):
    """``_swa_split_kernel`` with ``BPP`` (2..4) consecutive key blocks per program.

    Program ``(s, h, row group)`` owns blocks ``s * BPP .. s * BPP + BPP - 1``. Before the
    wait it reads what the one-block programs of those blocks read there (their cached K/V,
    the graph inputs) except the RoPE rows of the step's keys, which it reads with the
    projection; after the wait it writes the step's K/V when it owns block 0 and prepares its
    query rows once. Each block then runs that block's one-block statements from the initial
    online-softmax state and stores its partial in slot ``block`` (``num_splits`` slots, the
    combine unchanged), so every partial and every cache byte is the one-block kernel's.

    ``MERGE`` (not used by the ops; the profile bench measures it) instead carries the
    softmax state across the program's blocks and stores one partial per program in slot
    ``s`` of ``ceil(num_splits / BPP)``: a different association of the same FP32 merge,
    which does not reproduce the one-block bits.
    """
    G: tl.constexpr = HQ // HKV
    PAGE: tl.constexpr = HKV * P * D
    MG: tl.constexpr = M * G
    pid = tl.program_id(0)
    h = tl.program_id(1)
    row_group = tl.program_id(2)
    if MERGE:
        slots = tl.cdiv(num_splits, BPP)
    else:
        slots = num_splits
    if REQS > 1:
        # As _swa_split_kernel: request ``req``'s rows, length, table row, positions, partials.
        RG: tl.constexpr = (M * G + R - 1) // R
        req = row_group // RG
        row_group = row_group % RG
        qkv_ptr += req * M * qkv_stride
        bt_ptr += req * 2 * bt_stride_kv  # [requests, 2, pages], K then V rows
        seq_len_ptr += req
        if APPLY_ROPE:
            positions_ptr += req * M
        part_o_ptr += req * slots * HKV * MG * D
        part_ml_ptr += req * slots * HKV * MG * 2
    d = tl.arange(0, D)

    q_stride = D
    k_col = HQ * D + h * D
    v_col = (HQ + HKV) * D + h * D
    if FUSED_PREP:
        if not QKV_PERMUTED:
            q_stride = 2 * D
            k_col = 2 * HQ * D + h * D
            v_col = 2 * HQ * D + HKV * D + h * D

    # ---- Before the wait: as _swa_split_kernel, for every block this program owns.
    L = tl.load(seq_len_ptr)
    past = L - M
    if WINDOW > 0:
        lo = tl.maximum(past - WINDOW + 1, 0)
    else:
        lo = 0
    base = (lo // BLOCK) * BLOCK
    n_blocks = tl.cdiv(L - base, BLOCK)
    scale = tl.load(scale_ptr)
    scale_bf16 = scale.to(tl.bfloat16)
    dequant = 1.0 / scale

    blk = pid * BPP
    j0, new0, k_old0, v_old0 = _block_kv(
        pool_ptr, bt_ptr, bt_stride_kv, h, base, blk, n_blocks, lo, past, L, D, P, PAGE, BLOCK
    )
    if BPP > 1:
        j1, new1, k_old1, v_old1 = _block_kv(
            pool_ptr, bt_ptr, bt_stride_kv, h, base, blk + 1, n_blocks, lo, past, L, D, P, PAGE,
            BLOCK,
        )  # fmt: skip
    if BPP > 2:
        j2, new2, k_old2, v_old2 = _block_kv(
            pool_ptr, bt_ptr, bt_stride_kv, h, base, blk + 2, n_blocks, lo, past, L, D, P, PAGE,
            BLOCK,
        )  # fmt: skip
    if BPP > 3:
        j3, new3, k_old3, v_old3 = _block_kv(
            pool_ptr, bt_ptr, bt_stride_kv, h, base, blk + 3, n_blocks, lo, past, L, D, P, PAGE,
            BLOCK,
        )  # fmt: skip

    writer = (pid == 0) & (row_group == 0)
    tok = tl.arange(0, 8)
    tok_ok = tok < M
    write_ok = tok_ok & writer
    pos = past + tok
    w_page_k = tl.load(bt_ptr + pos // P, mask=write_ok, other=0).to(tl.int64)
    w_page_v = tl.load(bt_ptr + bt_stride_kv + pos // P, mask=write_ok, other=0).to(tl.int64)
    slot = (h * P + pos % P) * D

    r = row_group * R + tl.arange(0, R)  # row = token * G + query head within the KV group
    t = r // G
    row_ok = t < M
    if FUSED_PREP:
        cos_q, sin_q = _rope_rows(cos_sin_ptr, positions_ptr, t, row_ok, APPLY_ROPE)
        cos_w, sin_w = _rope_rows(cos_sin_ptr, positions_ptr, tok, write_ok, APPLY_ROPE)

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    if writer:
        # The step's K/V into the paged cache, as _swa_split_kernel's (0, h, 0) program.
        if FUSED_PREP:
            k_new = _prep_rows(
                qkv_ptr,
                tok * qkv_stride,
                tl.zeros([8], tl.int32) + k_col,
                tok_ok,
                cos_w,
                sin_w,
                1.0,
                prep_eps,
                8,
                APPLY_ROPE,
            )
        else:
            k_new = tl.load(
                qkv_ptr + tok[:, None] * qkv_stride + k_col + d[None, :],
                mask=tok_ok[:, None],
                other=0.0,
            )
        v_new = tl.load(
            qkv_ptr + tok[:, None] * qkv_stride + v_col + d[None, :],
            mask=tok_ok[:, None],
            other=0.0,
        )
        tl.store(
            pool_ptr + (w_page_k * PAGE + slot)[:, None] + d[None, :],
            _to_cache_fp8(k_new, scale_bf16),
            mask=tok_ok[:, None],
        )
        tl.store(
            pool_ptr + (w_page_v * PAGE + slot)[:, None] + d[None, :],
            _to_cache_fp8(v_new, scale_bf16),
            mask=tok_ok[:, None],
        )

    # The query rows, once for all the program's blocks.
    q_head = h * G + r % G
    if FUSED_PREP:
        q = _prep_rows(
            qkv_ptr,
            t * qkv_stride,
            q_head * q_stride,
            row_ok,
            cos_q,
            sin_q,
            q_scale,
            prep_eps,
            R,
            APPLY_ROPE,
        )
        q = tl.where(row_ok[:, None], q, 0.0)
    else:
        q = tl.load(
            qkv_ptr + t[:, None] * qkv_stride + q_head[:, None] * D + d[None, :],
            mask=row_ok[:, None],
            other=0.0,
        )
    qk_scale = sm_scale_log2 * dequant
    if Q_FP8:
        q = _to_cache_fp8(q, scale_bf16).to(tl.bfloat16)
        qk_scale = qk_scale * dequant
    q_pos = past + t

    m_i = tl.full([R], -1.0e30, tl.float32)
    l_i = tl.zeros([R], tl.float32)
    acc = tl.zeros([R, D], tl.float32)
    m_i, l_i, acc = _block_attend(
        m_i, l_i, acc, q, qk_scale, q_pos, j0, new0, k_old0, v_old0, blk, n_blocks, base, lo,
        past, qkv_ptr, qkv_stride, k_col, v_col, cos_sin_ptr, positions_ptr, prep_eps,
        scale_bf16, D, WINDOW, BLOCK, FUSED_PREP, APPLY_ROPE,
    )  # fmt: skip
    if not MERGE:
        _store_partial(part_o_ptr, part_ml_ptr, blk, slots, h, r, row_ok, acc, m_i, l_i,
                       dequant, HKV, MG, D)  # fmt: skip
    if BPP > 1:
        if not MERGE:
            m_i = tl.full([R], -1.0e30, tl.float32)
            l_i = tl.zeros([R], tl.float32)
            acc = tl.zeros([R, D], tl.float32)
        m_i, l_i, acc = _block_attend(
            m_i, l_i, acc, q, qk_scale, q_pos, j1, new1, k_old1, v_old1, blk + 1, n_blocks,
            base, lo, past, qkv_ptr, qkv_stride, k_col, v_col, cos_sin_ptr, positions_ptr,
            prep_eps, scale_bf16, D, WINDOW, BLOCK, FUSED_PREP, APPLY_ROPE,
        )  # fmt: skip
        if not MERGE:
            _store_partial(part_o_ptr, part_ml_ptr, blk + 1, slots, h, r, row_ok, acc, m_i,
                           l_i, dequant, HKV, MG, D)  # fmt: skip
    if BPP > 2:
        if not MERGE:
            m_i = tl.full([R], -1.0e30, tl.float32)
            l_i = tl.zeros([R], tl.float32)
            acc = tl.zeros([R, D], tl.float32)
        m_i, l_i, acc = _block_attend(
            m_i, l_i, acc, q, qk_scale, q_pos, j2, new2, k_old2, v_old2, blk + 2, n_blocks,
            base, lo, past, qkv_ptr, qkv_stride, k_col, v_col, cos_sin_ptr, positions_ptr,
            prep_eps, scale_bf16, D, WINDOW, BLOCK, FUSED_PREP, APPLY_ROPE,
        )  # fmt: skip
        if not MERGE:
            _store_partial(part_o_ptr, part_ml_ptr, blk + 2, slots, h, r, row_ok, acc, m_i,
                           l_i, dequant, HKV, MG, D)  # fmt: skip
    if BPP > 3:
        if not MERGE:
            m_i = tl.full([R], -1.0e30, tl.float32)
            l_i = tl.zeros([R], tl.float32)
            acc = tl.zeros([R, D], tl.float32)
        m_i, l_i, acc = _block_attend(
            m_i, l_i, acc, q, qk_scale, q_pos, j3, new3, k_old3, v_old3, blk + 3, n_blocks,
            base, lo, past, qkv_ptr, qkv_stride, k_col, v_col, cos_sin_ptr, positions_ptr,
            prep_eps, scale_bf16, D, WINDOW, BLOCK, FUSED_PREP, APPLY_ROPE,
        )  # fmt: skip
        if not MERGE:
            _store_partial(part_o_ptr, part_ml_ptr, blk + 3, slots, h, r, row_ok, acc, m_i,
                           l_i, dequant, HKV, MG, D)  # fmt: skip
    if MERGE:
        _store_partial(part_o_ptr, part_ml_ptr, pid, slots, h, r, row_ok, acc, m_i, l_i,
                       dequant, HKV, MG, D)  # fmt: skip


def launch_split_bpp(
    qkv: torch.Tensor,
    kv_pool: torch.Tensor,
    block_tables: torch.Tensor,
    table_stride_kv: int,
    seq_lens: torch.Tensor,
    kv_scale_orig_quant: torch.Tensor,
    cos_sin: Optional[torch.Tensor],
    positions: Optional[torch.Tensor],
    prep_eps: float,
    q_scale: float,
    part_o: torch.Tensor,
    part_ml: torch.Tensor,
    num_splits: int,
    *,
    rows: int,
    reqs: int,
    num_q_heads: int,
    num_kv_heads: int,
    window: int,
    q_fp8: bool,
    fused_prep: bool,
    qkv_permuted: bool,
    enable_pdl: bool,
    blocks_per_program: int,
    merge: bool = False,
    max_registers: Optional[int] = None,
):
    """Launch ``_swa_split_bpp_kernel`` of a ``reqs``-sequence step into ``part_o`` / ``part_ml``.

    ``blocks_per_program`` 2..4 key blocks per program, ``ceil(num_splits /
    blocks_per_program)`` programs per head and row group, with the arguments the ops pass
    ``_swa_split_kernel`` (``blocks_per_program = 1`` is that kernel, which the ops launch
    themselves). ``cos_sin`` / ``positions`` are ``None`` without RoPE. ``merge`` is for
    measurements only (see the kernel); the ops never pass it. ``max_registers``: the compile's
    register cap per thread (``maxnreg``; register allocation only, the same instructions'
    results). Returns the compiled kernel.
    """
    rope = cos_sin is not None
    row_groups = reqs * triton.cdiv(rows * Q_PER_KV, ROWS_PER_PROGRAM)
    args = (
        qkv,
        qkv.stride(0),
        kv_pool,
        block_tables,
        table_stride_kv,
        seq_lens,
        kv_scale_orig_quant,
        cos_sin if rope else kv_scale_orig_quant,
        positions if rope else seq_lens,
        prep_eps,
        q_scale,
        part_o,
        part_ml,
        math.log2(math.e) / math.sqrt(HEAD_DIM),
        num_splits,
    )
    meta = dict(
        M=rows,
        HQ=num_q_heads,
        HKV=num_kv_heads,
        D=HEAD_DIM,
        P=kv_pool.shape[2],
        WINDOW=window,
        BLOCK=BLOCK_N,
        R=ROWS_PER_PROGRAM,
        Q_FP8=q_fp8,
        FUSED_PREP=fused_prep,
        QKV_PERMUTED=qkv_permuted,
        APPLY_ROPE=rope,
        LAUNCH_WITH_PDL=enable_pdl,
    )
    if reqs > 1:
        meta["REQS"] = reqs
    if max_registers is not None:
        meta["maxnreg"] = max_registers
    programs = triton.cdiv(num_splits, blocks_per_program)
    return _swa_split_bpp_kernel[(programs, num_kv_heads, row_groups)](
        *args,
        **meta,
        BPP=blocks_per_program,
        MERGE=merge,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        launch_pdl=enable_pdl,
    )


def check_blocks_per_program(
    blocks_per_program: int, split_max_registers: Optional[int] = None
) -> None:
    """Raise ValueError unless ``blocks_per_program`` is 1..``MAX_BLOCKS_PER_PROGRAM`` and
    ``split_max_registers`` is ``None`` or, with ``blocks_per_program`` > 1, a cap in
    ``[MIN_SPLIT_REGISTERS, MAX_SPLIT_REGISTERS]`` (the one-block kernel has no cap)."""
    if (
        isinstance(blocks_per_program, bool)
        or not isinstance(blocks_per_program, int)
        or not 1 <= blocks_per_program <= MAX_BLOCKS_PER_PROGRAM
    ):
        raise ValueError(
            f"swa_block takes 1..{MAX_BLOCKS_PER_PROGRAM} key blocks per split program, got "
            f"{blocks_per_program!r}"
        )
    if split_max_registers is None:
        return
    if (
        blocks_per_program == 1
        or isinstance(split_max_registers, bool)
        or not isinstance(split_max_registers, int)
        or not MIN_SPLIT_REGISTERS <= split_max_registers <= MAX_SPLIT_REGISTERS
    ):
        raise ValueError(
            f"swa_block's split register cap must be None or {MIN_SPLIT_REGISTERS}.."
            f"{MAX_SPLIT_REGISTERS} with 2..{MAX_BLOCKS_PER_PROGRAM} key blocks per split "
            f"program, got {split_max_registers!r} at {blocks_per_program!r}"
        )


@triton.jit
def _norm_gate_sum_of_squares(x):
    """``sum(x * x)`` of one 64-wide head in the built-in norm-gate kernel's add order.

    That kernel (4 warps, one element per thread) reduces with a butterfly
    across the 32 threads of a warp, thread-index bit 4 first, then joins the two warps
    holding the head: element ``32 w + 16 b4 + 8 b3 + 4 b2 + 2 b1 + b0`` is
    paired over ``b4``, ``b3``, ``b2``, ``b1``, ``b0``, then ``w``. Each size-2
    sum is one IEEE add, which commutes, so the result is that kernel's bits.
    """
    sq = tl.reshape(x * x, [2, 2, 2, 2, 2, 2])
    sq = tl.sum(sq, axis=1)  # b4
    sq = tl.sum(sq, axis=1)  # b3
    sq = tl.sum(sq, axis=1)  # b2
    sq = tl.sum(sq, axis=1)  # b1
    sq = tl.sum(sq, axis=1)  # b0
    return tl.sum(sq, axis=0)  # w


@triton.jit
def _swa_combine_kernel(
    part_o_ptr,
    part_ml_ptr,
    out_ptr,
    out_stride,
    gate_ptr,
    gate_token_stride,
    gate_head_stride,
    gate_eps,
    num_splits,
    M: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    FUSED_GATE: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    REQS: tl.constexpr = 1,
):
    G: tl.constexpr = HQ // HKV
    row = tl.program_id(0)  # token * G + query head within the KV group
    h = tl.program_id(1)
    if REQS > 1:
        # Request ``program_id(2)``'s partials, output rows and gate rows.
        req = tl.program_id(2)
        part_o_ptr += req * num_splits * HKV * M * G * D
        part_ml_ptr += req * num_splits * HKV * M * G * 2
        out_ptr += req * M * out_stride
        if FUSED_GATE:
            gate_ptr += req * M * gate_token_stride
    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()
        if FUSED_GATE:
            # As the PDL norm-gate copy: the output projection may launch once
            # this grid has passed its wait.
            tl.extra.cuda.gdc_launch_dependents()
    sp = tl.arange(0, BLOCK_S)
    s_ok = sp < num_splits
    d = tl.arange(0, D)
    part = (sp * HKV + h) * (M * G) + row
    m_split = tl.load(part_ml_ptr + part * 2, mask=s_ok, other=-1.0e30)
    l_split = tl.load(part_ml_ptr + part * 2 + 1, mask=s_ok, other=0.0)
    m_max = tl.max(m_split, 0)
    w = tl.where(s_ok & (l_split > 0), tl.math.exp2(m_split - m_max), 0.0)
    den = tl.sum(w * l_split, 0)
    o = tl.load(part_o_ptr + part[:, None] * D + d[None, :], mask=s_ok[:, None], other=0.0)
    num = tl.sum(o * w[:, None], 0)
    t = row // G
    head = h * G + row % G
    attention = (num / den).to(tl.bfloat16)
    if FUSED_GATE:
        # The built-in _attention_norm_gate_kernel's statements on this head.
        x = attention.to(tl.float32)
        variance = _norm_gate_sum_of_squares(x) / D
        normed = (x * tl.rsqrt(variance + gate_eps)).to(tl.bfloat16)
        g = tl.load(gate_ptr + t * gate_token_stride + head * gate_head_stride + d).to(tl.float32)
        sigmoid = tl.sigmoid(g).to(tl.bfloat16)
        result = normed.to(tl.float32) * sigmoid.to(tl.float32)
        tl.store(out_ptr + t * out_stride + head * D + d, result)
    else:
        tl.store(out_ptr + t * out_stride + head * D + d, attention)


def max_blocks(window: int, rows: int) -> int:
    """The most 32-key blocks a ``window``-key step of ``rows`` rows can attend."""
    return triton.cdiv(window + rows - 2, BLOCK_N) + 1


def gate_head_stride(gate: torch.Tensor) -> int:
    """The stride between heads of ``gate`` as the built-in norm-gate launch computes it."""
    return gate.stride(1) if gate.ndim == 3 else HEAD_DIM


class SwaBlockAttention(OpWrapper):
    """A sliding-window layer's decode attention block, with fewer launches, the same bits.

    With ``M`` rows (positions ``L - M .. L - 1``, ``L = seq_lens[0]``), ``H``
    query and ``G = H / 8`` KV heads of width 64:

    * ``fused_prep=False``: ``qkv`` is ``preattn_prep``'s fused operand
      ``[M, (H + 2G) * 64]`` and the result is ``paged_decode_attention``'s.
    * ``fused_prep=True``: ``qkv`` is the raw QKV projection
      ``[M, 2 (H + G) * 64]`` (checkpoint order, or ``qkv_permuted``), and the
      result is ``paged_decode_attention(preattn_prep(qkv, ...)[0], ...)``
      with ``cos_sin``, ``position_ids``, ``prep_eps``, ``q_scale`` as
      ``preattn_prep`` takes them (``cos_sin=None``: NoPE).
    * ``gate`` (``[M, H * 64]`` or a ``[M, H, 64]`` view, unit last stride):
      the result is additionally the built-in output norm-gate,
      ``bf16(rmsnorm(head, gate_eps)) * bf16(sigmoid(gate head))`` per head.

    The step's K/V are written into the paged FP8 cache in place, as
    ``paged_decode_attention`` writes them. Returns a fresh ``[M, H * 64]``
    BF16 tensor. Batch 1, one sequence.

    Fusion boundary: the QKV projection before, the output projection after.

    ``enable_pdl``: both kernels launch with programmatic dependent launch. The
    split kernel reads its graph inputs and the cached K/V before
    ``griddepcontrol.wait`` and the operand after it, then lets the combine
    launch; the combine waits and, with ``gate``, lets its dependent launch.

    ``blocks_per_program`` (1..4): key blocks per split program (see the module
    docstring); 1 is the one-block kernel, every value computes the same bits.
    ``split_max_registers`` (``None`` or, with 2..4 blocks, a register cap per thread):
    the split kernel's compile register cap, the same bits.
    """

    ARCHS = frozenset({Arch.SM_103})

    note = """
    Metadata consumed: seq_lens[0], the sequence's K and V page ids, the
    positions (with fused_prep and RoPE), all on the device and read before
    griddepcontrol.wait; the layer's pool view; the FP8 KV scale. Call site:
    the target's sliding-window attention layers on decode steps of one
    sequence and at most 8 rows, under the swa_block fast path. Grids: split
    (ceil(num_splits / blocks_per_program), G, M*8/32), combine (M*8, G).

    blocks_per_program (1..4, default 1): key blocks per split program. 1 is
    the one-block kernel; 2..4, uncapped and under split_max_registers =
    SPLIT_REGISTER_CAP, are gated bit for bit against it (cache bytes,
    output, gated output, 1..4 sequences, window and page boundaries, a late
    PDL operand, CUDA-graph replays).

    Gated by its own tests, bit for bit against the chain it replaces
    (preattn_prep -> paged_decode_attention -> the built-in norm-gate kernel):
    KV cache bytes, attention output and gated output, at 1..8 rows across the
    window and page boundaries, eagerly and through CUDA-graph replays at
    changing contexts, behind a PDL predecessor that writes the operand late
    (the negative control); the PTX audit lists every global load before the
    wait and traces each to the seq_lens, block-table, pool, positions,
    cos/sin or scale parameter.

    Loud preconditions (ValueError): those of paged_decode_attention and, with
    fused_prep, of preattn_prep; a sliding window (window > 0); num_splits >=
    the window's largest block count; with gate, a BF16 [M, H*64] or
    [M, H, 64] gate of unit last stride.

    Not certified: full (window 0) attention, more than one sequence, head
    width other than 64, a query/KV head ratio other than 8.
    """

    def __call__(
        self,
        qkv: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        window: int,
        num_splits: int,
        q_fp8: bool = False,
        fused_prep: bool = False,
        cos_sin: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        prep_eps: float = 0.0,
        q_scale: float = 1.0,
        qkv_permuted: bool = False,
        gate: Optional[torch.Tensor] = None,
        gate_eps: float = 0.0,
        enable_pdl: bool = False,
        blocks_per_program: int = 1,
        split_max_registers: Optional[int] = None,
    ) -> torch.Tensor:
        check_blocks_per_program(blocks_per_program, split_max_registers)
        rows = qkv.shape[0]
        page = kv_pool.shape[2] if kv_pool.dim() == 4 else 0
        width = (2 if fused_prep else 1) * (num_q_heads + num_kv_heads) * HEAD_DIM
        if not fused_prep:
            width += num_kv_heads * HEAD_DIM
        if (
            qkv.dtype != torch.bfloat16
            or qkv.dim() != 2
            or qkv.stride(-1) != 1
            or qkv.shape[1] != width
            or num_q_heads != Q_PER_KV * num_kv_heads
            or not 1 <= rows <= MAX_ROWS
            or kv_pool.dtype != torch.float8_e4m3fn
            or kv_pool.dim() != 4
            or tuple(kv_pool.shape[1:]) != (num_kv_heads, page, HEAD_DIM)
            or not kv_pool.is_contiguous()
            or page & (page - 1) != 0
            or block_tables.dtype != torch.int32
            or block_tables.dim() != 2
            or block_tables.shape[0] != 2
            or block_tables.stride(-1) != 1
            or seq_lens.dtype != torch.int32
            or kv_scale_orig_quant.dtype != torch.float32
            or window <= 0
            or num_splits < max_blocks(window, rows)
        ):
            raise ValueError(
                "swa_block takes BF16 qkv [1..8, "
                + ("2 (H + G) * 64" if fused_prep else "(H + 2G) * 64")
                + "] with H = 8 G, an FP8 E4M3 pool [pages, G, P, 64] (P a power of two), an int32 "
                "[2, pages] block table, int32 seq_lens, an fp32 scale, a sliding window and at least "
                f"one split per key block; got qkv {qkv.dtype} {tuple(qkv.shape)}, H={num_q_heads} "
                f"G={num_kv_heads}, pool {kv_pool.dtype} {tuple(kv_pool.shape)}, table "
                f"{block_tables.dtype} {tuple(block_tables.shape)}, seq_lens {seq_lens.dtype}, scale "
                f"{kv_scale_orig_quant.dtype}, window {window}, splits {num_splits} (need "
                f"{max_blocks(window, rows) if window > 0 else 'a window'})"
            )
        rope = fused_prep and cos_sin is not None
        positions = None
        if rope:
            positions = position_ids.reshape(-1)
            if (
                positions.numel() != rows
                or cos_sin.dtype != torch.float32
                or cos_sin.dim() != 3
                or tuple(cos_sin.shape[1:]) != (2, HEAD_DIM // 2)
                or not cos_sin.is_contiguous()
            ):
                raise ValueError(
                    f"swa_block RoPE needs one position per row and a contiguous fp32 "
                    f"[max_positions, 2, {HEAD_DIM // 2}] table, got {positions.numel()} positions "
                    f"for {rows} rows and a {cos_sin.dtype} {tuple(cos_sin.shape)} table"
                )
        if gate is not None and (
            gate.dtype != torch.bfloat16
            or gate.ndim not in (2, 3)
            or gate.stride(-1) != 1
            or gate.shape[0] != rows
            or gate.numel() != rows * num_q_heads * HEAD_DIM
            or (gate.ndim == 3 and gate.shape[-1] != HEAD_DIM)
        ):
            raise ValueError(
                f"swa_block's gate must be BF16 [rows, H * {HEAD_DIM}] or [rows, H, {HEAD_DIM}] "
                f"with unit last stride, got {gate.dtype} {tuple(gate.shape)}"
            )
        if _SWA_CL8 == "all" and _cl8_eligible(
            qkv,
            kv_pool,
            num_q_heads,
            num_kv_heads,
            window,
            rows,
            q_fp8,
            fused_prep,
            gate,
            cos_sin,
            position_ids,
            4,
        ):
            fn = _cl8_function(qkv.device, rope and positions.dtype == torch.int64, 4)
            if fn is not None:
                out = torch.empty(
                    (rows, num_q_heads * HEAD_DIM), dtype=torch.bfloat16, device=qkv.device
                )
                return _cl8_launch(
                    fn,
                    qkv,
                    kv_pool,
                    block_tables.unsqueeze(0) if block_tables.dim() == 2 else block_tables,
                    seq_lens,
                    kv_scale_orig_quant,
                    cos_sin,
                    positions,
                    prep_eps,
                    q_scale,
                    gate,
                    gate_eps,
                    out,
                    qkv_permuted,
                    enable_pdl,
                    rope,
                )
        out = torch.empty((rows, num_q_heads * HEAD_DIM), dtype=torch.bfloat16, device=qkv.device)
        group_rows = rows * Q_PER_KV
        part_o = torch.empty(
            (num_splits * num_kv_heads * group_rows, HEAD_DIM),
            dtype=torch.float32,
            device=qkv.device,
        )
        part_ml = torch.empty(
            (num_splits * num_kv_heads * group_rows, 2), dtype=torch.float32, device=qkv.device
        )
        if blocks_per_program > 1:
            launch_split_bpp(
                qkv, kv_pool, block_tables, block_tables.stride(0), seq_lens, kv_scale_orig_quant,
                cos_sin if rope else None, positions, prep_eps, q_scale, part_o, part_ml,
                num_splits, rows=rows, reqs=1, num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads, window=window, q_fp8=q_fp8, fused_prep=fused_prep,
                qkv_permuted=qkv_permuted, enable_pdl=enable_pdl,
                blocks_per_program=blocks_per_program, max_registers=split_max_registers,
            )  # fmt: skip
        else:
            _swa_split_kernel[
                (num_splits, num_kv_heads, triton.cdiv(group_rows, ROWS_PER_PROGRAM))
            ](
                qkv,
                qkv.stride(0),
                kv_pool,
                block_tables,
                block_tables.stride(0),
                seq_lens,
                kv_scale_orig_quant,
                cos_sin if rope else kv_scale_orig_quant,
                positions if rope else seq_lens,
                prep_eps,
                q_scale,
                part_o,
                part_ml,
                math.log2(math.e) / math.sqrt(HEAD_DIM),
                num_splits,
                M=rows,
                HQ=num_q_heads,
                HKV=num_kv_heads,
                D=HEAD_DIM,
                P=page,
                WINDOW=window,
                BLOCK=BLOCK_N,
                R=ROWS_PER_PROGRAM,
                Q_FP8=q_fp8,
                FUSED_PREP=fused_prep,
                QKV_PERMUTED=qkv_permuted,
                APPLY_ROPE=rope,
                LAUNCH_WITH_PDL=enable_pdl,
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES,
                launch_pdl=enable_pdl,
            )
        _swa_combine_kernel[(group_rows, num_kv_heads)](
            part_o,
            part_ml,
            out,
            out.stride(0),
            out if gate is None else gate,
            0 if gate is None else gate.stride(0),
            0 if gate is None else gate_head_stride(gate),
            gate_eps,
            num_splits,
            M=rows,
            HQ=num_q_heads,
            HKV=num_kv_heads,
            D=HEAD_DIM,
            BLOCK_S=triton.next_power_of_2(num_splits),
            FUSED_GATE=gate is not None,
            LAUNCH_WITH_PDL=enable_pdl,
            num_warps=4,
            launch_pdl=enable_pdl,
        )
        return out

    def reference(
        self,
        qkv: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        window: int,
        num_splits: int,
        q_fp8: bool = False,
        fused_prep: bool = False,
        cos_sin: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        prep_eps: float = 0.0,
        q_scale: float = 1.0,
        qkv_permuted: bool = False,
        gate: Optional[torch.Tensor] = None,
        gate_eps: float = 0.0,
        enable_pdl: bool = False,
        blocks_per_program: int = 1,
        split_max_registers: Optional[int] = None,
    ) -> torch.Tensor:
        """The three references in sequence: ``preattn_prep``'s (with ``fused_prep``), then
        ``paged_decode_attention``'s (which writes the step's rows into ``kv_pool``), then the
        norm-gate in float64 between its BF16 rounding points (with ``gate``).

        ``num_splits``, ``enable_pdl``, ``blocks_per_program`` and ``split_max_registers`` change
        how the op is cut and launched, not what it computes, and are ignored here.
        """
        if fused_prep:
            rope = cos_sin is not None
            qkv, _ = _preattn_prep.reference(
                qkv,
                cos_sin,
                position_ids if rope else None,
                num_q_heads,
                num_kv_heads,
                prep_eps,
                q_scale,
                qkv_permuted,
            )
        out = _paged_decode_attention.reference(
            qkv,
            kv_pool,
            block_tables,
            seq_lens,
            kv_scale_orig_quant,
            num_q_heads,
            num_kv_heads,
            window,
            num_splits,
            q_fp8,
        )
        if gate is None:
            return out
        rows = out.shape[0]
        x = out.reshape(rows, num_q_heads, HEAD_DIM).double()
        normed = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + gate_eps)).to(torch.bfloat16)
        g = gate.reshape(rows, num_q_heads, HEAD_DIM).double()
        sigmoid = torch.sigmoid(g).to(torch.bfloat16)
        gated = (normed.double() * sigmoid.double()).to(torch.bfloat16)
        return gated.reshape(rows, num_q_heads * HEAD_DIM)

    def is_valid(
        self,
        qkv: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        window: int,
        num_splits: int,
        q_fp8: bool = False,
        fused_prep: bool = False,
        cos_sin: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        prep_eps: float = 0.0,
        q_scale: float = 1.0,
        qkv_permuted: bool = False,
        gate: Optional[torch.Tensor] = None,
        gate_eps: float = 0.0,
        enable_pdl: bool = False,
        blocks_per_program: int = 1,
        split_max_registers: Optional[int] = None,
    ) -> None:
        """What the kernels take and answer wrongly with no error: those of
        ``paged_decode_attention`` (a context shorter than the step, a block table too short, page
        ids outside the pool) and, with RoPE, a position outside the table."""
        rows = qkv.shape[0]
        length = int(seq_lens[0])
        page = kv_pool.shape[2]
        if length < rows:
            raise ValueError(f"seq_lens[0] = {length} is shorter than the step's {rows} rows")
        pages = (length + page - 1) // page
        if block_tables.shape[1] < pages:
            raise ValueError(
                f"a [2, {block_tables.shape[1]}] block table cannot address {length} keys "
                f"of {page}-token pages"
            )
        lo = max(length - rows - window + 1, 0) if window > 0 else 0
        used = block_tables[:, lo // page : pages]
        if used.numel() and (int(used.min()) < 0 or int(used.max()) >= kv_pool.shape[0]):
            raise ValueError(
                f"page ids [{int(used.min())}, {int(used.max())}] outside the {kv_pool.shape[0]}-page pool"
            )
        if fused_prep and cos_sin is not None and position_ids is not None and position_ids.numel():
            low, high = int(position_ids.min()), int(position_ids.max())
            if low < 0 or high >= cos_sin.shape[0]:
                raise ValueError(
                    f"positions [{low}, {high}] outside the RoPE table of {cos_sin.shape[0]}"
                )


swa_block_attention = SwaBlockAttention()


#: Sequences one multi-request call takes: a TP4 decode step of up to four requests.
MULTI_MAX_REQUESTS = 4


class SwaBlockMultiAttention(OpWrapper):
    """``swa_block_attention`` of a decode step of several sequences, in the same two launches.

    ``qkv`` holds ``B = seq_lens.numel()`` sequences of ``M = rows / B`` rows each,
    request-major (rows ``b M .. b M + M - 1`` are sequence ``b``'s, at positions
    ``seq_lens[b] - M .. seq_lens[b] - 1``); ``block_tables`` is ``[B, 2, pages]`` (each
    sequence's K and V page ids, the TRT-LLM table's layout); ``position_ids`` and ``gate``
    are per row in the same order. Row block ``b`` of the result, and every cache byte the
    call writes for sequence ``b``, is what ``swa_block_attention`` computes for that
    sequence alone, bit for bit: each program is a one-sequence program of the same kernel
    on its sequence's rows, length, page ids, positions and partials (the kernels' ``REQS``
    axis; ``REQS = 1`` is the one-sequence build, unchanged).

    Fusion boundary, PDL and pre-wait loads: as ``swa_block_attention``; ``blocks_per_program``
    and ``split_max_registers`` as there.
    """

    ARCHS = frozenset({Arch.SM_103})

    note = """
    Metadata consumed: seq_lens[0..B), each sequence's K and V page ids, the
    positions (with fused_prep and RoPE), all on the device and read before
    griddepcontrol.wait; the layer's pool view; the FP8 KV scale. Call site:
    the target's sliding-window attention layers on TP4 decode steps of 2..4
    sequences of equal row count (each request's spec-decoding verify), under
    the wide_swa_block fast path. Grids: split (ceil(num_splits /
    blocks_per_program), G, B*ceil(M*8/32)), combine (M*8, G, B).

    Gated by its own tests, bit for bit against swa_block_attention of each
    sequence alone (cache bytes and output), eagerly and through CUDA-graph
    replays at changing contexts.

    Loud preconditions (ValueError): those of swa_block_attention per sequence;
    2..4 sequences, rows a multiple of their number, at most 8 rows each; an
    int32 [B, 2, pages] block table whose sequence stride is two rows;
    contiguous seq_lens and positions.

    Not certified: sequences of different row counts, more than 4 sequences.
    """

    def __call__(
        self,
        qkv: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        window: int,
        num_splits: int,
        q_fp8: bool = False,
        fused_prep: bool = False,
        cos_sin: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        prep_eps: float = 0.0,
        q_scale: float = 1.0,
        qkv_permuted: bool = False,
        gate: Optional[torch.Tensor] = None,
        gate_eps: float = 0.0,
        enable_pdl: bool = False,
        blocks_per_program: int = 1,
        split_max_registers: Optional[int] = None,
    ) -> torch.Tensor:
        check_blocks_per_program(blocks_per_program, split_max_registers)
        total = qkv.shape[0]
        reqs = seq_lens.numel()
        rows = total // reqs if reqs else 0
        page = kv_pool.shape[2] if kv_pool.dim() == 4 else 0
        width = (2 if fused_prep else 1) * (num_q_heads + num_kv_heads) * HEAD_DIM
        if not fused_prep:
            width += num_kv_heads * HEAD_DIM
        if (
            not 2 <= reqs <= MULTI_MAX_REQUESTS
            or rows * reqs != total
            or not 1 <= rows <= MAX_ROWS
            or qkv.dtype != torch.bfloat16
            or qkv.dim() != 2
            or qkv.stride(-1) != 1
            or qkv.shape[1] != width
            or num_q_heads != Q_PER_KV * num_kv_heads
            or kv_pool.dtype != torch.float8_e4m3fn
            or kv_pool.dim() != 4
            or tuple(kv_pool.shape[1:]) != (num_kv_heads, page, HEAD_DIM)
            or not kv_pool.is_contiguous()
            or page & (page - 1) != 0
            or block_tables.dtype != torch.int32
            or block_tables.dim() != 3
            or tuple(block_tables.shape[:2]) != (reqs, 2)
            or block_tables.stride(-1) != 1
            or block_tables.stride(0) != 2 * block_tables.stride(1)
            or seq_lens.dtype != torch.int32
            or not seq_lens.is_contiguous()
            or kv_scale_orig_quant.dtype != torch.float32
            or window <= 0
            or num_splits < max_blocks(window, rows)
        ):
            raise ValueError(
                f"swa_block_multi takes 2..{MULTI_MAX_REQUESTS} sequences of 1..{MAX_ROWS} rows "
                "each, request-major BF16 qkv [B * M, "
                + ("2 (H + G) * 64" if fused_prep else "(H + 2G) * 64")
                + "] with H = 8 G, an FP8 E4M3 pool [pages, G, P, 64] (P a power of two), an int32 "
                "[B, 2, pages] block table of sequence stride two rows, contiguous int32 seq_lens "
                "[B], an fp32 scale, a sliding window and at least one split per key block; got "
                f"qkv {qkv.dtype} {tuple(qkv.shape)}, {reqs} sequences, H={num_q_heads} "
                f"G={num_kv_heads}, pool {kv_pool.dtype} {tuple(kv_pool.shape)}, table "
                f"{block_tables.dtype} {tuple(block_tables.shape)} strides "
                f"{tuple(block_tables.stride())}, seq_lens {seq_lens.dtype}, scale "
                f"{kv_scale_orig_quant.dtype}, window {window}, splits {num_splits}"
            )
        rope = fused_prep and cos_sin is not None
        positions = None
        if rope:
            positions = position_ids.reshape(-1)
            if (
                positions.numel() != total
                or not positions.is_contiguous()
                or cos_sin.dtype != torch.float32
                or cos_sin.dim() != 3
                or tuple(cos_sin.shape[1:]) != (2, HEAD_DIM // 2)
                or not cos_sin.is_contiguous()
            ):
                raise ValueError(
                    f"swa_block_multi RoPE needs one contiguous position per row and a contiguous "
                    f"fp32 [max_positions, 2, {HEAD_DIM // 2}] table, got {positions.numel()} "
                    f"positions for {total} rows and a {cos_sin.dtype} {tuple(cos_sin.shape)} table"
                )
        if gate is not None and (
            gate.dtype != torch.bfloat16
            or gate.ndim not in (2, 3)
            or gate.stride(-1) != 1
            or gate.shape[0] != total
            or gate.numel() != total * num_q_heads * HEAD_DIM
            or (gate.ndim == 3 and gate.shape[-1] != HEAD_DIM)
        ):
            raise ValueError(
                f"swa_block_multi's gate must be BF16 [rows, H * {HEAD_DIM}] or "
                f"[rows, H, {HEAD_DIM}] with unit last stride, got {gate.dtype} {tuple(gate.shape)}"
            )
        if _cl8_on(num_kv_heads, reqs) and _cl8_eligible(
            qkv,
            kv_pool,
            num_q_heads,
            num_kv_heads,
            window,
            rows,
            q_fp8,
            fused_prep,
            gate,
            cos_sin,
            position_ids,
            4,
        ):
            fn = _cl8_function(qkv.device, rope and positions.dtype == torch.int64, 4)
            if fn is not None:
                out = torch.empty(
                    (total, num_q_heads * HEAD_DIM), dtype=torch.bfloat16, device=qkv.device
                )
                return _cl8_launch(
                    fn,
                    qkv,
                    kv_pool,
                    block_tables,
                    seq_lens,
                    kv_scale_orig_quant,
                    cos_sin,
                    positions,
                    prep_eps,
                    q_scale,
                    gate,
                    gate_eps,
                    out,
                    qkv_permuted,
                    enable_pdl,
                    rope,
                )
        out = torch.empty((total, num_q_heads * HEAD_DIM), dtype=torch.bfloat16, device=qkv.device)
        group_rows = rows * Q_PER_KV
        part_o = torch.empty(
            (reqs * num_splits * num_kv_heads * group_rows, HEAD_DIM),
            dtype=torch.float32,
            device=qkv.device,
        )
        part_ml = torch.empty(
            (reqs * num_splits * num_kv_heads * group_rows, 2),
            dtype=torch.float32,
            device=qkv.device,
        )
        if blocks_per_program > 1:
            launch_split_bpp(
                qkv, kv_pool, block_tables, block_tables.stride(1), seq_lens, kv_scale_orig_quant,
                cos_sin if rope else None, positions, prep_eps, q_scale, part_o, part_ml,
                num_splits, rows=rows, reqs=reqs, num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads, window=window, q_fp8=q_fp8, fused_prep=fused_prep,
                qkv_permuted=qkv_permuted, enable_pdl=enable_pdl,
                blocks_per_program=blocks_per_program, max_registers=split_max_registers,
            )  # fmt: skip
        else:
            _swa_split_kernel[
                (num_splits, num_kv_heads, reqs * triton.cdiv(group_rows, ROWS_PER_PROGRAM))
            ](
                qkv,
                qkv.stride(0),
                kv_pool,
                block_tables,
                block_tables.stride(1),
                seq_lens,
                kv_scale_orig_quant,
                cos_sin if rope else kv_scale_orig_quant,
                positions if rope else seq_lens,
                prep_eps,
                q_scale,
                part_o,
                part_ml,
                math.log2(math.e) / math.sqrt(HEAD_DIM),
                num_splits,
                M=rows,
                HQ=num_q_heads,
                HKV=num_kv_heads,
                D=HEAD_DIM,
                P=page,
                WINDOW=window,
                BLOCK=BLOCK_N,
                R=ROWS_PER_PROGRAM,
                Q_FP8=q_fp8,
                FUSED_PREP=fused_prep,
                QKV_PERMUTED=qkv_permuted,
                APPLY_ROPE=rope,
                LAUNCH_WITH_PDL=enable_pdl,
                REQS=reqs,
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES,
                launch_pdl=enable_pdl,
            )
        _swa_combine_kernel[(group_rows, num_kv_heads, reqs)](
            part_o,
            part_ml,
            out,
            out.stride(0),
            out if gate is None else gate,
            0 if gate is None else gate.stride(0),
            0 if gate is None else gate_head_stride(gate),
            gate_eps,
            num_splits,
            M=rows,
            HQ=num_q_heads,
            HKV=num_kv_heads,
            D=HEAD_DIM,
            BLOCK_S=triton.next_power_of_2(num_splits),
            FUSED_GATE=gate is not None,
            LAUNCH_WITH_PDL=enable_pdl,
            REQS=reqs,
            num_warps=4,
            launch_pdl=enable_pdl,
        )
        return out

    def reference(
        self,
        qkv: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        window: int,
        num_splits: int,
        q_fp8: bool = False,
        fused_prep: bool = False,
        cos_sin: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        prep_eps: float = 0.0,
        q_scale: float = 1.0,
        qkv_permuted: bool = False,
        gate: Optional[torch.Tensor] = None,
        gate_eps: float = 0.0,
        enable_pdl: bool = False,
        blocks_per_program: int = 1,
        split_max_registers: Optional[int] = None,
    ) -> torch.Tensor:
        """``swa_block_attention.reference`` of each sequence alone, rows concatenated."""
        reqs = seq_lens.numel()
        rows = qkv.shape[0] // reqs
        positions = position_ids.reshape(-1) if position_ids is not None else None
        outs = []
        for b in range(reqs):
            sl = slice(b * rows, (b + 1) * rows)
            outs.append(
                swa_block_attention.reference(
                    qkv[sl],
                    kv_pool,
                    block_tables[b],
                    seq_lens[b : b + 1],
                    kv_scale_orig_quant,
                    num_q_heads,
                    num_kv_heads,
                    window,
                    num_splits,
                    q_fp8,
                    fused_prep,
                    cos_sin,
                    positions[sl] if positions is not None else None,
                    prep_eps,
                    q_scale,
                    qkv_permuted,
                    gate[sl] if gate is not None else None,
                    gate_eps,
                )
            )
        return torch.cat(outs)

    def is_valid(
        self,
        qkv: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        window: int,
        num_splits: int,
        q_fp8: bool = False,
        fused_prep: bool = False,
        cos_sin: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        prep_eps: float = 0.0,
        q_scale: float = 1.0,
        qkv_permuted: bool = False,
        gate: Optional[torch.Tensor] = None,
        gate_eps: float = 0.0,
        enable_pdl: bool = False,
        blocks_per_program: int = 1,
        split_max_registers: Optional[int] = None,
    ) -> None:
        """``swa_block_attention.is_valid`` of each sequence, and no page written by two of them."""
        reqs = seq_lens.numel()
        rows = qkv.shape[0] // reqs
        positions = position_ids.reshape(-1) if position_ids is not None else None
        page = kv_pool.shape[2]
        written = set()
        for b in range(reqs):
            sl = slice(b * rows, (b + 1) * rows)
            swa_block_attention.is_valid(
                qkv[sl],
                kv_pool,
                block_tables[b],
                seq_lens[b : b + 1],
                kv_scale_orig_quant,
                num_q_heads,
                num_kv_heads,
                window,
                num_splits,
                q_fp8,
                fused_prep,
                cos_sin,
                positions[sl] if positions is not None else None,
            )
            length = int(seq_lens[b])
            first, last = (length - rows) // page, (length - 1) // page
            for kv in range(2):
                pages = set(block_tables[b, kv, first : last + 1].tolist())
                if pages & written:
                    raise ValueError(f"sequence {b} writes a page another sequence writes")
                written |= pages


swa_block_multi_attention = SwaBlockMultiAttention()
