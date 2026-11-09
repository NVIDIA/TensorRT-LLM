# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""

from __future__ import annotations

from ...sm107_defaults import env as _sm107_env

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

#: MUSESPARK_OPEN_SWA_MAXNREG=1: cap the split kernel's registers per launch so its grid fits one wave.
#: Read once at import; unset or 0 launches exactly as the image does. Register
#: allocation only: the same statements, so the same bits (outputs, KV cache bytes and partials checked
#: identical over 8 seeds x 1..8 requests at 2 and 4 KV heads).
_SWA_MAXNREG = _sm107_env("MUSESPARK_OPEN_SWA_MAXNREG", "0") == "1"
#: The split kernel takes 144 registers (3 CTAs of 4 warps per SM); 128 fits 4 and 96 fits 5.
_SPLIT_CTAS_PER_SM = {None: 3, 128: 4, 96: 5}
_SM_COUNT = {}
#: MUSESPARK_OPEN_SWA_ROWS64=1: on decode steps of 2+ sequences (``swa_block_multi_attention``) each split program
#: covers all 64 query rows of its KV head (8 tokens x 8 query heads) instead of 32, which halves the split grid.
#: NOT bit-identical: the softmax row sum l of each 32-key block is reduced in another
#: order (a different warp layout of the [64, 32] scores), so l differs by up to 4 fp32 ulp on about a third of
#: the rows and roughly 1 in 10^5 BF16 outputs flips by one ulp; S, P, P.V, m, the combine, the norm-gate and every
#: KV cache byte are unchanged. One-sequence steps keep the image's 32-row programs. Read once at import.
_SWA_ROWS64 = _sm107_env("MUSESPARK_OPEN_SWA_ROWS64", "0") == "1"
#: Warps of the 64-row split per (KV heads per rank, sequences), the faster of 4 and 8 as measured in the 46-layer
#: PDL graph; 4 elsewhere. ROWS64 applies only where the image's 32-row grid takes more than one wave.
_ROWS64_WARPS = {(4, 2): 8, (2, 4): 8}
#: MUSESPARK_OPEN_SWA_ONE_LAUNCH=1: decode steps of 2+ sequences run the whole block (prep, attention, merge,
#: norm-gate, cache write) in ONE launch (``_swa_one_launch_kernel``) when its grid fits one CTA per SM and the
#: image's 32-row split grid would take more than one wave. NOT bit-identical (see that kernel). Precedence:
#: ONE_LAUNCH, then ROWS64, then MAXNREG. Read once at import.
_SWA_ONE_LAUNCH = _sm107_env("MUSESPARK_OPEN_SWA_ONE_LAUNCH", "0") == "1"
#: Keys per one-launch program: 17 programs per (sequence, KV head) cover the 66 key blocks of a 2048-key window.
_ONE_LAUNCH_KEYS = 128
#: MUSESPARK_OPEN_SWA_PREP_SPLIT=1: on decode steps of 2+ sequences whose 32-row split grid takes more than one
#: wave, run the prep as its own launch (``preattn_prep``, dependents released at its entry) and the block on its
#: fused operand -- the image's own ``SwaBlockPlan.fused_prep=False`` composition, certified bit-identical to the
#: fused-prep build. Bit-identical; the split programs then load prepared rows instead of each re-running the
#: query prep, so the split grid is lighter. Applied after ONE_LAUNCH (which needs the raw projection)
#: and before ROWS64 / MAXNREG (which then apply to the unfused launch). Read once at import.
_SWA_PREP_SPLIT = _sm107_env("MUSESPARK_OPEN_SWA_PREP_SPLIT", "0") == "1"
#: MUSESPARK_OPEN_SWA_ONE2=1: TEP4 decode steps of 2..4 sequences (4 KV heads per rank)
#: and TEP8 steps of 5..8 sequences (2 KV heads per rank;
#: fused prep with the norm-gate, 32-key pages) run the whole sliding-window block in ONE launch -- prep, K/V append,
#: window attention over 17 programs of 128 keys per (sequence, KV head), a merge by the last 4 programs of each group
#: to arrive and the norm-gate. The appended K/V bytes are the image's; the output is not bit-identical (its own
#: softmax split and merge; mean error vs the float reference within ~7 % of the image's). Grids above one wave
#: (17 x 4 x 4 = 272 programs) run with 128 registers (2 CTAs/SM). Other shapes take the paths below unchanged.
_SWA_ONE2 = _sm107_env("MUSESPARK_OPEN_SWA_ONE2", "0") == "1"
_ONE2_KEYS = 128
#: ONE2 at TEP8 also takes decode steps of 5..8 sequences (2 KV heads per rank), which the image runs on the
#: built-in FMHA; kernels/swa_block/attention.py lifts the wide_swa_block cap to it for those layers.
_ONE2_MAX_REQUESTS = 8

#: The one-cluster SWA block (CL8) has one kernel source, built per KV-head count (``hkv``), and two gates, each
#: serving only its own head count and request range:
#: - MUSESPARK_OPEN_SWA_CL8 (``_SWA_CL8``): TEP4, 4 KV heads per rank, 2..4 requests ("all": also 1).
#: - MUSESPARK_OPEN_TEP8_SWA_CL8 (``_TEP8_SWA_CL8``): TEP8, 2 KV heads per rank, 5..8 requests.
#:
#: MUSESPARK_OPEN_SWA_CL8: "1" -- TEP4 decode steps of 2..4 requests (4 KV
#: heads and 32 query heads per rank, 32-key FP8 pages, window 2048, fused prep with the norm-gate, q_fp8, at most 8
#: rows per request) run the whole sliding-window block as ONE cluster launch instead of ONE2: grid (8, 4, requests),
#: cluster of 8 CTAs per (request, KV head), each CTA one SM (320 threads, 180 KB dynamic SMEM requested); before the
#: dependency wait the page ids and the window's cached K/V; after it the q/k prep (preattn_prep's statements), the
#: step's K/V append (the image's bytes), mma.sync attention over 288 keys per CTA (FP8 Q.K^T, f16 P.V), and the
#: partials merged through distributed shared memory (st.async + mbarrier), then the norm-gate. "all" also takes
#: one-request steps. Outputs are not bit-identical to ONE2 (another softmax partition, P in f16): GSM8K. Unset or "0"
#: = unchanged. Read once at import; the kernel is compiled with NVRTC on its first eager call per device (a call
#: inside a CUDA-graph capture before that falls back to the ONE2 path and warns once).
_SWA_CL8 = _sm107_env("MUSESPARK_OPEN_SWA_CL8", "0")
#: Opt-in: MUSESPARK_OPEN_TEP8_SWA_CL8=1 -- TEP8 decode steps of 5..8 requests (2 KV heads and 16
#: query heads per rank, 32-key FP8 pages, window 2048, fused prep with the norm-gate, q_fp8, at most 8 rows per
#: request) run the whole sliding-window block as ONE cluster launch instead of ONE2: the TEP4 CL8 kernel
#: (the fixed build: no ``__restrict__`` on the QKV projection or the gate)
#: compiled with 2 KV heads per rank (``-DHKV=2``), grid (8, 2, requests), cluster of 8 CTAs per (request, KV head),
#: 320 threads, 180 KB dynamic SMEM, one CTA per SM. Each CTA computes what the TEP4 build computes for its (request, KV head):
#: 8 query heads of 64 per KV head at both TP sizes. Not bit-identical to ONE2 (another softmax partition, P in f16);
#: K/V cache bytes are the image's. Unset or "0" = unchanged. Read once at import; compiled with NVRTC on the
#: first eager call per device (a call inside a capture before that keeps the ONE2 path and warns once).
_TEP8_SWA_CL8 = _sm107_env("MUSESPARK_OPEN_TEP8_SWA_CL8", "0")
#: Requests per step the TEP8 CL8 path takes (the ONE2 range at 2 KV heads).
_CL8_TEP8_MIN_REQUESTS, _CL8_TEP8_MAX_REQUESTS = 5, 8
#: CL8 debug knobs (engine failure): MUSESPARK_OPEN_SWA_CL8_PDL=0 launches CL8 without PDL;
#: MUSESPARK_OPEN_SWA_CL8_TRIGGER=end lets dependents launch only after the output store (default: right after the wait).
_CL8_PDL = _sm107_env("MUSESPARK_OPEN_SWA_CL8_PDL", "1") != "0"
_CL8_TRIGGER_END = _sm107_env("MUSESPARK_OPEN_SWA_CL8_TRIGGER", "") == "end"
#: MUSESPARK_OPEN_SWA_CL8_PREWAIT=0: CL8 waits at entry (no global load before the dependency wait).
_CL8_NO_PREWAIT = _sm107_env("MUSESPARK_OPEN_SWA_CL8_PREWAIT", "1") == "0"
#: MUSESPARK_OPEN_SWA_CL8_DEBUG=<path prefix>: CL8 re-reads every pre-wait input after the dependency wait and counts
#: the ones that changed (per device, int64 [seq_len, window page id, window K/V bytes, new-row page id, RoPE row,
#: CTAs checked]); eager multi-op calls append the running counts to <prefix>.dev<d>.txt every 500 calls.
_CL8_DEBUG = _sm107_env("MUSESPARK_OPEN_SWA_CL8_DEBUG", "")
#: MUSESPARK_OPEN_SWA_CL8_RESTRICT=1 (A/B only): qkv and gate get the original `const __restrict__` back, under which the
#: gate load is scheduled above griddepcontrol.wait (the engine failure). Default: the fix, no qualifier.
_CL8_RESTRICT_PRODUCED = _sm107_env("MUSESPARK_OPEN_SWA_CL8_RESTRICT", "0") == "1"
#: MUSESPARK_OPEN_SWA_CL8_PREWAIT=cg: the pre-wait loads bypass L1 (ld.global.cg).
_CL8_PREWAIT_CG = _sm107_env("MUSESPARK_OPEN_SWA_CL8_PREWAIT", "1") == "cg"
_CL8_DBG = {}
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
// The QKV projection's output (q, k, v and the gate view) is written by the PDL predecessor while this grid already
// runs. Declared `const __restrict__`, it is data the compiler may treat as invariant for the whole kernel: the original
// build loaded the gate through the read-only path (LDG.E.CONSTANT) ABOVE griddepcontrol.wait (ACQBULK) and so read
// the previous layer's gate in the engine (GSM8K 2/400). These two pointers therefore carry no __restrict__ (plain
// LDG after ACQBULK, checked in the SASS). CL8_RESTRICT_PRODUCED restores
// the original qualifier, for the A/B only.
#ifdef CL8_RESTRICT_PRODUCED
#define PRODUCED __restrict__
#else
#define PRODUCED
#endif
#ifdef CL8_PREWAIT_CG
#define LDP(p) __ldcg(p)  // pre-wait loads at L2 (ld.global.cg), never from a line the SM's L1 kept from an older grid
#else
#define LDP(p) (*(p))
#endif
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
swa_cl8(const u16* PRODUCED qkv, long long qkv_stride, u8* __restrict__ pool, const int* __restrict__ bt,
        long long bt_req_stride, long long bt_kv_stride, const int* __restrict__ seq_lens,
        const float* __restrict__ kv_scale, const float* __restrict__ cos_sin, const pos_t* __restrict__ positions,
        float prep_eps, float q_scale, float sm_scale_log2, u16* __restrict__ out, long long out_stride,
        const u16* PRODUCED gate, long long gate_tok_stride, long long gate_head_stride, float gate_eps,
        int M, int q_stride, int k_col0, int v_col0, int rope, int pdl, u64* tbuf) {
  extern __shared__ __align__(128) u8 smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int rank = blockIdx.x, h = blockIdx.y, req = blockIdx.z;
  TSTAMP(0);
#ifdef CL8_INSTR
  if (tid == 0) { u32 sm_; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_)); tbuf[(blockIdx.x + 8 * (blockIdx.y + HKV * blockIdx.z)) * 16 + 9] = sm_; }
#endif

  // ---------------- before the dependency wait: graph inputs and older cache bytes only
#ifdef CL8_NO_PREWAIT
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
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
#ifdef CL8_DEBUG
  int pids_dbg[9];
#endif
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
#ifdef CL8_DEBUG
#pragma unroll
    for (int i = 0; i < 9; ++i) pids_dbg[i] = pids[i];
#endif
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
#ifndef CL8_TRIGGER_END
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
  }
  __syncthreads();  // wpid / RoPE rows
#ifdef CL8_DEBUG
  // which pre-wait input changed by the time the dependency wait returned (tbuf: 0 seq_len, 1 window page id,
  // 2 window K/V bytes, 3 new-row page id, 4 RoPE row, 5 CTAs checked)
  if (pdl) {
    if (tid == 0) {
      if (__ldcg(seq_lens + req) != L) atomicAdd(&tbuf[0], 1ull);
      atomicAdd(&tbuf[5], 1ull);
    }
    if (tid < NTM) {
      unsigned long long nb = 0, nk = 0;
#pragma unroll
      for (int i = 0; i < 9; ++i) {
        const int pl = 2 * i + (tid >> 7), kv = pl / PAGES, p = pl % PAGES, page = first_page + p;
        const int pid2 = (page * PG < past) ? __ldcg(btr + kv * bt_kv_stride + page) : -1;
        nb += (pid2 != pids_dbg[i]);
        const int c = i * NTM + tid, w = c & 127, key = (first_page + p) * PG + (w >> 2);
        if (pids_dbg[i] >= 0 && key < past) {
          const uint4 v2 = __ldcg(reinterpret_cast<const uint4*>(pool + ((long long)pids_dbg[i] * HKV + h) * (PG * D) + w * 16));
          nk += (v2.x != rb[i].x || v2.y != rb[i].y || v2.z != rb[i].z || v2.w != rb[i].w);
        }
      }
      if (nb) atomicAdd(&tbuf[1], nb);
      if (nk) atomicAdd(&tbuf[2], nk);
    } else {
      const int u = tid - NTM;
      if (u < 16) {
        const int t = u >> 1, kv = u & 1, pos = past + t, page = pos >> 5;
        int pid = -1;
        if (t < M && page >= first_page && page < first_page + PAGES) pid = __ldcg(btr + kv * bt_kv_stride + page);
        if (pid != wpid[2 * t + kv]) atomicAdd(&tbuf[3], 1ull);
      }
      if (rope) {
        unsigned long long nr = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int i = k * 64 + u, t = i >> 6, c = i & 63;
          const float v = (t < M) ? __ldcg(cos_sin + (long long)__ldcg(positions + req * M + t) * 64 + c) : 0.f;
          nr += (__float_as_uint(v) != __float_as_uint(cs[i]));
        }
        if (nr) atomicAdd(&tbuf[4], nr);
      }
    }
  }
#endif
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
  const int g = lane >> 2, c4 = lane & 3, mi = lane >> 3, rin = lane & 7;
  u32 qa[2][4];
  {
    const u32 sq = sm32(smem + SQ_OFF);
    const int row = rb * 16 + (mi & 1) * 8 + rin;
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
        const int krow = kh * 144 + 8 * (n + (mi >> 1)) + rin;
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
      const int vrow = kh * 144 + 16 * kk + (mi & 1) * 8 + rin;
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
#ifdef CL8_TRIGGER_END
  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
"""


_CL8_FN = {}
_CL8_WARNED = set()


def _cl8_log(msg: str, warning: bool = False) -> None:
    try:
        from tensorrt_llm.logger import logger

        (logger.warning if warning else logger.info)(msg)
    except Exception:  # noqa: BLE001 - lab copies load this file without tensorrt_llm
        print(msg, flush=True)


def _cl8_check(r):
    if int(r[0]) != 0:
        raise RuntimeError(f"swa_block CL8: CUDA/NVRTC error {r!r}")
    return r[1] if len(r) == 2 else r[1:]


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
            _cl8_log("[musespark_open] swa_block CL8: first call inside a CUDA-graph capture; this capture keeps the "
                     f"{'ONE2' if hkv == 4 else 'ONE2'} path (compile happens on an eager call)", warning=True)
        return None
    from cuda.bindings import driver as cu
    from cuda.bindings import nvrtc

    major, minor = torch.cuda.get_device_capability(index)
    prog = _cl8_check(nvrtc.nvrtcCreateProgram(_CL8_SRC.encode(), b"swa_cl8.cu", 0, [], []))
    opts = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"-std=c++17", b"-default-device"]
    if hkv != 4:
        # The source's HKV defaults to 4 (the tep4 build, same options); the TEP8 build uses -DHKV=2 here.
        opts.append(f"-DHKV={int(hkv)}".encode())
    if pos64:
        opts.append(b"-DCL8_POS64")
    if _CL8_TRIGGER_END:
        opts.append(b"-DCL8_TRIGGER_END")
    if _CL8_NO_PREWAIT:
        opts.append(b"-DCL8_NO_PREWAIT")
    if _CL8_DEBUG:
        opts.append(b"-DCL8_DEBUG")
    if _CL8_PREWAIT_CG:
        opts.append(b"-DCL8_PREWAIT_CG")
    if _CL8_RESTRICT_PRODUCED:
        opts.append(b"-DCL8_RESTRICT_PRODUCED")
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
        _cl8_check(cu.cuFuncSetAttribute(f, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                         _CL8_SMEM))
    regs = int(_cl8_check(cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, f)))
    lmem = int(_cl8_check(cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, f)))
    _CL8_FN[key] = (f, mod)
    # Each build keeps its own install line: TEP4's, and TEP8's (gate name, KV heads).
    if hkv == 4:
        head, heads = f"[musespark_open] swa_block CL8: MUSESPARK_OPEN_SWA_CL8={_SWA_CL8}", ""
    else:
        head = f"[musespark_open] swa_block CL8 (TEP8): MUSESPARK_OPEN_TEP8_SWA_CL8={_TEP8_SWA_CL8}"
        heads = f"{int(hkv)} KV heads per rank, "
    _cl8_log(f"{head} compiled for sm_{major}{minor} on "
             f"device {index} (regs {regs}, local {lmem} B, smem {_CL8_SMEM // 1024} KB, cluster 8, 320 threads, "
             f"{heads}{'int64' if pos64 else 'int32'} positions, PDL {'on' if _CL8_PDL else 'off'}, trigger "
             f"{'at the end' if _CL8_TRIGGER_END else 'after the wait'}, pre-wait loads "
             f"{'off' if _CL8_NO_PREWAIT else ('at L2 (cg)' if _CL8_PREWAIT_CG else 'on')}"
             f"{', DEBUG re-reads' if _CL8_DEBUG else ''}"
             f"{', qkv/gate __restrict__ (A/B only)' if _CL8_RESTRICT_PRODUCED else ''})")
    return _CL8_FN[key]


def _cl8_eligible(qkv, kv_pool, num_q_heads, num_kv_heads, window, rows, q_fp8, fused_prep, gate, cos_sin,
                  position_ids, hkv) -> bool:
    """A call the CL8 build of ``hkv`` KV heads per rank serves: ``hkv`` KV / ``8 hkv`` query heads per rank (4 / 32
    at tep4, 2 / 16 at tep8), the kernel's other conditions."""
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
        and (cos_sin is None or position_ids is None or position_ids.dtype in (torch.int32, torch.int64))
    )


def _cl8_dbg_ptr(device) -> int:
    """The debug counters' address (0 without MUSESPARK_OPEN_SWA_CL8_DEBUG); every 500 eager calls the counts are
    appended to <prefix>.dev<d>.txt."""
    if not _CL8_DEBUG:
        return 0
    index = device.index if device.index is not None else torch.cuda.current_device()
    st = _CL8_DBG.get(index)
    if st is None:
        if torch.cuda.is_current_stream_capturing():
            return 0
        st = _CL8_DBG[index] = {"buf": torch.zeros(16, dtype=torch.int64, device=device), "calls": 0}
    if not torch.cuda.is_current_stream_capturing():
        st["calls"] += 1
        if st["calls"] % 500 == 0:
            c = st["buf"].tolist()
            with open(f"{_CL8_DEBUG}.dev{index}.txt", "a") as f:
                f.write(f"calls {st['calls']} seq_len {c[0]} window_page {c[1]} window_kv {c[2]} new_page {c[3]} "
                        f"rope {c[4]} ctas {c[5]}\n")
    return st["buf"].data_ptr()


def _cl8_launch(fn, qkv, kv_pool, bt, seq_lens, kv_scale, cos_sin, positions, prep_eps, q_scale, gate, gate_eps,
                out, qkv_permuted, enable_pdl, rope):
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
    args = ((qkv.data_ptr(), qkv.stride(0), kv_pool.data_ptr(), bt.data_ptr(), bt.stride(0), bt.stride(1),
             seq_lens.data_ptr(), kv_scale.data_ptr(), cos_sin.data_ptr() if rope else 0,
             positions.data_ptr() if rope else 0, float(prep_eps), float(q_scale), math.log2(math.e) / 8.0,
             out.data_ptr(), out.stride(0), gate.data_ptr(), gts, ghs, float(gate_eps), rows, q_stride, k_col0,
             v_col0, int(rope), int(bool(enable_pdl)), _cl8_dbg_ptr(qkv.device)),
            (vp, ll, vp, vp, ll, ll, vp, vp, vp, vp, fl, fl, fl, vp, ll, vp, ll, ll, fl, ii, ii, ii, ii, ii, ii, vp))
    _cl8_check(cu.cuLaunchKernelEx(cfg, fn[0], args, 0))
    return out



#: MUSESPARK_OPEN_SWA_SHADOW=<path prefix> (numerical verification; TEP4 CL8 only). On EAGER decode
#: calls of the multi op (run the engine with cuda_graph_config null; captured calls are untouched) that CL8 can take,
#: the first MUSESPARK_OPEN_SWA_SHADOW_FIRST (default 400) and then every MUSESPARK_OPEN_SWA_SHADOW_EVERY-th (default
#: 23) run ONE2, CL8 and the op's reference on the same live inputs: the step's K/V bytes of all three are compared,
#: the outputs are scored against fp64 references (q' exact; q' as the E4M3 the op computes), one JSON line per call
#: goes to <prefix>.rank<r>.pid<p>.jsonl, and the first MUSESPARK_OPEN_SWA_SHADOW_DUMP (default 2) compared calls per
#: request count are saved with their inputs (<prefix>.rank<r>.pid<p>.bs<B>.<k>.pt) for an offline replay. The call
#: returns the deployed arm's output (CL8 under MUSESPARK_OPEN_SWA_CL8, else ONE2) and leaves that arm's K/V bytes
#: in the pool. Unset = unchanged. TEP8: at 2 KV heads per rank the covered calls are TEP8 steps of
#: 5..8 requests (ONE2 launched as at TEP8, NM=16; CL8 built with -DHKV=2) and the deployed arm is
#: CL8 under MUSESPARK_OPEN_TEP8_SWA_CL8=1, else ONE2.
_SWA_SHADOW = _sm107_env("MUSESPARK_OPEN_SWA_SHADOW", "")
_SHADOW_EVERY = max(1, int(_sm107_env("MUSESPARK_OPEN_SWA_SHADOW_EVERY", "23")))
_SHADOW_FIRST = int(_sm107_env("MUSESPARK_OPEN_SWA_SHADOW_FIRST", "400"))
_SHADOW_DUMP = int(_sm107_env("MUSESPARK_OPEN_SWA_SHADOW_DUMP", "2"))
_SHADOW = {"calls": 0, "bypass": False, "samples": 0, "dumps": {}}


def _shadow_rank() -> int:
    for k in ("SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMIX_RANK", "RANK", "LOCAL_RANK"):
        if _sm107_env(k, "").isdigit():
            return int(os.environ[k])
    return 0


def _shadow_one2(qkv, kv_pool, block_tables, seq_lens, kv_scale, num_q_heads, num_kv_heads, window, rows, reqs,
                 cos_sin, positions, rope, prep_eps, q_scale, q_fp8, qkv_permuted, gate, gate_eps):
    """The ONE2 launch (NM=16 at 2 KV heads), statement for statement, without PDL (eager)."""
    counters = _one2_counters(qkv.device)
    programs = triton.cdiv(max_blocks(window, rows) * BLOCK_N, _ONE2_KEYS)
    groups = reqs * num_kv_heads
    sms = torch.cuda.get_device_properties(qkv.device).multi_processor_count
    part_o = torch.empty((groups * programs * 64, HEAD_DIM), dtype=torch.float32, device=qkv.device)
    part_ml = torch.empty((groups * programs * 64, 2), dtype=torch.float32, device=qkv.device)
    out = torch.empty((qkv.shape[0], num_q_heads * HEAD_DIM), dtype=torch.bfloat16, device=qkv.device)
    options = {"maxnreg": 128} if programs * groups > sms else {}
    _swa_one2_kernel[(programs, num_kv_heads, reqs)](
        qkv, qkv.stride(0), kv_pool, block_tables, block_tables.stride(1), seq_lens, kv_scale,
        cos_sin if rope else kv_scale, positions if rope else seq_lens, prep_eps, q_scale, part_o, part_ml,
        math.log2(math.e) / math.sqrt(HEAD_DIM), out, out.stride(0), gate, gate.stride(0), gate_head_stride(gate),
        gate_eps, counters, M=rows, HQ=num_q_heads, HKV=num_kv_heads, D=HEAD_DIM, P=BLOCK_N, WINDOW=window,
        KEYS=_ONE2_KEYS, S=programs, SP=triton.next_power_of_2(programs), RM=4, Q_FP8=q_fp8,
        QKV_PERMUTED=qkv_permuted, APPLY_ROPE=rope, LAUNCH_WITH_PDL=False,
        NM=16 if num_kv_heads == 2 else 4, num_warps=8, num_stages=1,
        launch_pdl=False, **options)
    return out


def _shadow_rows_index(block_tables, seq_lens, rows, reqs):
    """(page ids, slots) of the step's K and V rows: [reqs][2] pairs of index tensors."""
    idx = []
    for r in range(reqs):
        L = int(seq_lens[r])
        pos = torch.arange(L - rows, L, device=block_tables.device)
        idx.append([(block_tables[r, kv, pos // BLOCK_N].long(), pos % BLOCK_N) for kv in range(2)])
    return idx


def _shadow_get_kv(kv_pool, idx):
    raw = kv_pool.view(torch.uint8)  # fp8 has no CUDA advanced indexing
    return [raw[pid, :, slot, :].clone() for pair in idx for pid, slot in pair]


def _shadow_put_kv(kv_pool, idx, rows_bytes):
    raw = kv_pool.view(torch.uint8)
    for (pid, slot), b in zip([x for pair in idx for x in pair], rows_bytes):
        raw[pid, :, slot, :] = b


def _shadow_ref64(prepped, kv_pool, block_tables, seq_lens, kv_scale, H, G, window, rows, reqs, gate, gate_eps):
    """fp64 attention + norm-gate over the pool as it is after the step (its K/V rows included), from the prepped
    q' (fp64 prep, BF16 at the op's three rounding points): {"exact": q' as is, "q8": q' through the op's E4M3}."""
    D = HEAD_DIM
    dev = prepped.device
    raw = kv_pool.view(torch.uint8)
    scale = float(kv_scale.reshape(-1)[0])
    sb = float(torch.tensor(scale, dtype=torch.float32).to(torch.bfloat16).float())
    res = {"exact": [], "q8": []}
    for r in range(reqs):
        L = int(seq_lens[r])
        past = L - rows
        lo = max(past - window + 1, 0)
        j = torch.arange(lo, L, device=dev)

        def gather(kv):
            b = raw[block_tables[r, kv, j // BLOCK_N].long(), :, j % BLOCK_N, :]  # [n, G, D]
            return b.view(torch.float8_e4m3fn).float().double() / scale

        Kh = gather(0).repeat_interleave(H // G, dim=1)
        Vh = gather(1).repeat_interleave(H // G, dim=1)
        q = prepped[r * rows:(r + 1) * rows, : H * D].reshape(rows, H, D)
        q8 = (q.float() * sb).to(torch.bfloat16).float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        qs = {"exact": q.double(), "q8": q8.float().double() / scale}
        qp = past + torch.arange(rows, device=dev)
        mask = (j[None, :] <= qp[:, None]) & (j[None, :] > qp[:, None] - window)
        for k, qq in qs.items():
            s = torch.einsum("thd,nhd->thn", qq, Kh) / math.sqrt(D)
            s = s.masked_fill(~mask[:, None, :], float("-inf"))
            res[k].append(torch.einsum("thn,nhd->thd", torch.softmax(s, dim=-1), Vh))
    sig = torch.sigmoid(gate.reshape(rows * reqs, H, D).double())
    out = {}
    for k, parts in res.items():
        x = torch.cat(parts)
        out[k] = ((x * torch.rsqrt(x.square().mean(-1, keepdim=True) + gate_eps)) * sig).reshape(rows * reqs, H * D)
    return out


_ULP_EDGES = (0.5, 1.0, 2.0, 4.0, 8.0)


def _shadow_stats(y, ref):
    yd = y.double()
    d = (yd - ref).abs()
    ulp = torch.exp2(torch.floor(torch.log2(ref.abs().clamp_min(2.0 ** -126))) - 7)
    u = d / ulp
    hist, prev = [], None
    for e in _ULP_EDGES:
        hist.append(int(((u <= e) & (u > prev)).sum()) if prev is not None else int((u <= e).sum()))
        prev = e
    hist.append(int((u > _ULP_EDGES[-1]).sum()))
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "max_ulp": float(u.max()),
            "mean_ulp": float(u.mean()), "ulp_hist": hist, "n": int(d.numel()),
            "rn_frac": float((y.view(torch.int16) == ref.to(torch.bfloat16).view(torch.int16)).double().mean()),
            "finite": bool(torch.isfinite(yd).all())}


def _shadow_compare(op, args, kw, rope, positions, rows, reqs):
    import json

    st = _SHADOW
    qkv, kv_pool, block_tables, seq_lens, kv_scale, H, G, window, num_splits = args
    if G == 2:
        # TEP8 steps of 5..8 requests (the ONE2 range at 2 KV heads)
        eligible = (_cl8_eligible(qkv, kv_pool, H, G, window, rows, kw["q_fp8"], kw["fused_prep"], kw["gate"],
                                  kw["cos_sin"], kw["position_ids"], 2)
                    and _CL8_TEP8_MIN_REQUESTS <= reqs <= _CL8_TEP8_MAX_REQUESTS and kv_pool.shape[2] == BLOCK_N
                    and _SWA_ONE2 and rows <= MAX_ROWS)
    else:
        # TEP4 steps of 2..4 requests
        eligible = (_cl8_eligible(qkv, kv_pool, H, G, window, rows, kw["q_fp8"], kw["fused_prep"], kw["gate"],
                                  kw["cos_sin"], kw["position_ids"], 4) and 2 <= reqs <= 4
                    and kv_pool.shape[2] == BLOCK_N)
    if eligible:
        st["calls"] += 1
    if not eligible or (st["calls"] > _SHADOW_FIRST and st["calls"] % _SHADOW_EVERY):
        st["bypass"] = True
        try:
            return op(*args, **kw)
        finally:
            st["bypass"] = False
    cs = kw["cos_sin"]
    use_cl8 = (_TEP8_SWA_CL8 == "1") if G == 2 else (_SWA_CL8 in ("1", "all"))
    idx = _shadow_rows_index(block_tables, seq_lens, rows, reqs)
    o1 = _shadow_one2(qkv, kv_pool, block_tables, seq_lens, kv_scale, H, G, window, rows, reqs, cs, positions, rope,
                      kw["prep_eps"], kw["q_scale"], kw["q_fp8"], kw["qkv_permuted"], kw["gate"], kw["gate_eps"])
    kv1 = _shadow_get_kv(kv_pool, idx)
    fn = _cl8_function(qkv.device, rope and positions.dtype == torch.int64, G)
    o2 = torch.empty_like(o1)
    _cl8_launch(fn, qkv, kv_pool, block_tables, seq_lens, kv_scale, cs, positions, kw["prep_eps"], kw["q_scale"],
                kw["gate"], kw["gate_eps"], o2, kw["qkv_permuted"], False, rope)
    kv2 = _shadow_get_kv(kv_pool, idx)
    prepped, _ = _preattn_prep.reference(qkv, cs, positions if rope else None, H, G, kw["prep_eps"], kw["q_scale"],
                                         kw["qkv_permuted"])
    refs = _shadow_ref64(prepped, kv_pool, block_tables, seq_lens, kv_scale, H, G, window, rows, reqs, kw["gate"],
                         kw["gate_eps"])
    kw_ref = {k: v for k, v in kw.items() if k != "enable_pdl"}
    o3 = op.reference(*args, **kw_ref)
    kv3 = _shadow_get_kv(kv_pool, idx)
    _shadow_put_kv(kv_pool, idx, kv2 if use_cl8 else kv1)  # the deployed arm's bytes stay in the pool
    torch.cuda.synchronize()
    rank = _shadow_rank()
    rec = {"rank": rank, "call": st["calls"], "sample": st["samples"], "reqs": reqs, "rows": rows,
           "L": [int(x) for x in seq_lens.tolist()], "rope": bool(rope), "qkv_permuted": bool(kw["qkv_permuted"]),
           "q_scale": float(kw["q_scale"]), "prep_eps": float(kw["prep_eps"]), "gate_eps": float(kw["gate_eps"]),
           "kv_scale": float(kv_scale.reshape(-1)[0]), "gate_shape": list(kw["gate"].shape),
           "gate_strides": list(kw["gate"].stride()), "qkv_stride": qkv.stride(0), "bt_strides": list(block_tables.stride()),
           "pos_dtype": str(positions.dtype) if positions is not None else None,
           "kv_one2_vs_cl8": all(torch.equal(a, b) for a, b in zip(kv1, kv2)),
           "kv_ref_vs_cl8": all(torch.equal(a, b) for a, b in zip(kv3, kv2)),
           "cl8_vs_one2": {"max_abs": float((o2.float() - o1.float()).abs().max()),
                           "differ_frac": float((o2.view(torch.int16) != o1.view(torch.int16)).double().mean())}}
    for name, y in (("one2", o1), ("cl8", o2), ("ref", o3)):
        rec[name] = {k: _shadow_stats(y, ref) for k, ref in refs.items()}
    st["samples"] += 1
    tag = f"{_SWA_SHADOW}.rank{rank}.pid{os.getpid()}"
    with open(f"{tag}.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    n_dump = st["dumps"].get(reqs, 0)
    if n_dump < _SHADOW_DUMP:
        st["dumps"][reqs] = n_dump + 1
        need = []
        for r in range(reqs):
            L = int(seq_lens[r])
            lo = max(L - rows - window + 1, 0)
            for kv in range(2):
                need.append(block_tables[r, kv, lo // BLOCK_N:(L - 1) // BLOCK_N + 1].long())
        need = torch.unique(torch.cat(need))
        torch.save({"rec": rec, "qkv": qkv.detach().clone(), "gate": kw["gate"].detach().clone(),
                    "gate_meta": {"shape": list(kw["gate"].shape), "strides": list(kw["gate"].stride()),
                                  "offset_in_qkv": (kw["gate"].data_ptr() - qkv.data_ptr()) // 2},
                    "block_tables": block_tables.detach().clone(), "seq_lens": seq_lens.detach().clone(),
                    "positions": positions.detach().clone() if positions is not None else None,
                    "cos_sin_rows": cs[positions.long()].detach().clone() if rope else None,
                    "kv_scale": kv_scale.detach().clone(), "pages": need.cpu(),
                    "pool_pages": kv_pool.view(torch.uint8)[need].detach().cpu(),
                    "pool_shape": list(kv_pool.shape), "window": window, "H": H, "G": G, "num_splits": num_splits,
                    "kw": {k: v for k, v in kw.items() if not torch.is_tensor(v)},
                    "out_one2": o1.cpu(), "out_cl8": o2.cpu(), "out_ref": o3.cpu(),
                    "kv_one2": [x.cpu() for x in kv1], "kv_cl8": [x.cpu() for x in kv2]},
                   f"{tag}.bs{reqs}.{n_dump}.pt")
    return o2 if use_cl8 else o1


if _SWA_SHADOW:
    _cl8_log(f"[musespark_open] swa_block SHADOW: MUSESPARK_OPEN_SWA_SHADOW={_SWA_SHADOW}: the first {_SHADOW_FIRST} "
             f"then every {_SHADOW_EVERY}th eager multi-op call run ONE2 + CL8 + the reference vs fp64 "
             f"(dumps {_SHADOW_DUMP} per request count; deployed arm {'CL8' if _SWA_CL8 in ('1', 'all') else 'ONE2'})")
    _cl8_log(f"[musespark_open] swa_block SHADOW (TEP8): MUSESPARK_OPEN_SWA_SHADOW={_SWA_SHADOW}: the first "
             f"{_SHADOW_FIRST} then every {_SHADOW_EVERY}th eager TEP8 multi-op call of {_CL8_TEP8_MIN_REQUESTS}-"
             f"{_CL8_TEP8_MAX_REQUESTS} requests run ONE2 + CL8 + the reference vs fp64 (dumps {_SHADOW_DUMP} per "
             f"request count; deployed arm {'CL8' if _TEP8_SWA_CL8 == '1' else 'ONE2'})")
if _SWA_CL8 not in ("0", ""):
    _cl8_log(f"[musespark_open] swa_block CL8 gate: MUSESPARK_OPEN_SWA_CL8={_SWA_CL8} "
             f"({'1-4' if _SWA_CL8 == 'all' else '2-4'} requests per TEP4 decode step run the one-cluster SWA block)")
if _TEP8_SWA_CL8 not in ("0", ""):
    _cl8_log(f"[musespark_open] swa_block CL8 gate (TEP8): MUSESPARK_OPEN_TEP8_SWA_CL8={_TEP8_SWA_CL8} "
             f"({_CL8_TEP8_MIN_REQUESTS}-{_CL8_TEP8_MAX_REQUESTS} requests per TEP8 decode step run the one-cluster "
             "SWA block)")


def _split_maxnreg(ctas: int, device: torch.device) -> dict:
    """``{"maxnreg": n}`` for a split grid of ``ctas`` programs under ``MUSESPARK_OPEN_SWA_MAXNREG``, else ``{}``.

    No cap when the grid already fits one wave (one request at TP4); the smallest cap that makes it one wave
    otherwise (128 up to 4 per SM, 96 up to 5); 128 past that (fewer spills when no cap fits one wave).
    Measured per layer (split + combine, 46-layer PDL graph, ~155k context): TP4 2 requests 10.87 -> 10.30 us
    (96), 3 requests 14.57 -> 12.78 (128), 4 requests 17.44 -> 16.17 (128); TP8 4 requests 11.09 -> 10.27 (96).
    Host-side arithmetic only, so it is CUDA-graph safe; each cap is one more compiled variant.
    """
    if not _SWA_MAXNREG:
        return {}
    index = device.index if device.index is not None else torch.cuda.current_device()
    sms = _SM_COUNT.get(index)
    if sms is None:
        sms = _SM_COUNT[index] = torch.cuda.get_device_properties(index).multi_processor_count
    if ctas <= _SPLIT_CTAS_PER_SM[None] * sms:
        return {}
    for cap in (128, 96):
        if ctas <= _SPLIT_CTAS_PER_SM[cap] * sms:
            return {"maxnreg": cap}
    return {"maxnreg": 128}


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


@triton.jit
def _fence_acq_rel_gpu(x):
    return tl.inline_asm_elementwise(
        "fence.acq_rel.gpu; mov.u32 $0, $1;", "=r,r", [x], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _rows_norm_gate_sum_of_squares(x, RM: tl.constexpr):
    """``_norm_gate_sum_of_squares`` of each row of ``[RM, 64]``, the same add tree per row."""
    sq = tl.reshape(x * x, [RM, 2, 2, 2, 2, 2, 2])
    sq = tl.sum(sq, axis=2)  # b4
    sq = tl.sum(sq, axis=2)  # b3
    sq = tl.sum(sq, axis=2)  # b2
    sq = tl.sum(sq, axis=2)  # b1
    sq = tl.sum(sq, axis=2)  # b0
    return tl.sum(sq, axis=1)  # w


@triton.jit
def _swa_one_launch_kernel(
    qkv_ptr, qkv_stride, pool_ptr, bt_ptr, bt_stride_kv, seq_len_ptr, scale_ptr, cos_sin_ptr, positions_ptr,
    prep_eps, q_scale, part_o_ptr, part_ml_ptr, sm_scale_log2,
    out_ptr, out_stride, gate_ptr, gate_token_stride, gate_head_stride, gate_eps, cnt_ptr,
    M: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, P: tl.constexpr,
    WINDOW: tl.constexpr, KEYS: tl.constexpr, S: tl.constexpr, SP: tl.constexpr, RM: tl.constexpr,
    Q_FP8: tl.constexpr, QKV_PERMUTED: tl.constexpr, APPLY_ROPE: tl.constexpr, LAUNCH_WITH_PDL: tl.constexpr,
):
    """The sliding-window block of several sequences in one launch (MUSESPARK_OPEN_SWA_ONE_LAUNCH).

    Grid (S, G, sequences). Program ``s`` of the (sequence, KV head) group owns keys ``base + s KEYS ..`` of the
    window for all 64 query rows of the head (8 tokens x 8 query heads): before the wait it reads only what the
    image's split kernel reads there (seq_lens, its page ids, its cached K/V ``j < past``, positions and RoPE
    rows); after it, the raw projection (prep with the image's statements; program 0 writes the step's K/V to
    the cache as the image does), S = Q K^T, one softmax over its KEYS keys, P V, and its partial. A group
    barrier through L2 (release add, acquire spin on a per-group counter) follows; then each program merges
    ``RM`` rows from the group's ``S`` partials in program order and applies the image's norm-gate statements.
    The last program to leave resets the group's counters to 0. The barrier needs the whole grid resident:
    the launcher takes it only for grids of at most one CTA per SM; the dependent launches only after every
    program passed its wait (all resident by then).

    Numerics (not the image's bits): P is rounded to BF16 against the KEYS-key maximum, and S partials are
    merged sequentially instead of 66 one-block partials in the combine's 128-lane tree.
    """
    G: tl.constexpr = HQ // HKV
    R: tl.constexpr = 8 * G
    PAGE: tl.constexpr = HKV * P * D
    s_id = tl.program_id(0)
    h = tl.program_id(1)
    req = tl.program_id(2)
    group = req * HKV + h
    qkv_ptr += req * M * qkv_stride
    bt_ptr += req * 2 * bt_stride_kv
    seq_len_ptr += req
    if APPLY_ROPE:
        positions_ptr += req * M
    out_ptr += req * M * out_stride
    gate_ptr += req * M * gate_token_stride
    part_o_ptr += group * S * R * D
    part_ml_ptr += group * S * R * 2
    d = tl.arange(0, D)
    q_stride = 2 * D
    k_col = 2 * HQ * D + h * D
    v_col = 2 * HQ * D + HKV * D + h * D
    if QKV_PERMUTED:
        q_stride = D
        k_col = HQ * D + h * D
        v_col = (HQ + HKV) * D + h * D

    L = tl.load(seq_len_ptr)
    past = L - M
    lo = tl.maximum(past - WINDOW + 1, 0)
    base = (lo // 32) * 32
    j = base + s_id * KEYS + tl.arange(0, KEYS)
    live = j < L
    cached = (j >= lo) & (j < past) & live
    new = (j >= past) & live
    scale = tl.load(scale_ptr)
    scale_bf16 = scale.to(tl.bfloat16)
    dequant = 1.0 / scale
    page_k = tl.load(bt_ptr + j // P, mask=cached, other=0).to(tl.int64)
    page_v = tl.load(bt_ptr + bt_stride_kv + j // P, mask=cached, other=0).to(tl.int64)
    in_page = (h * P + j % P) * D
    k_old = tl.load(pool_ptr + (page_k * PAGE + in_page)[:, None] + d[None, :], mask=cached[:, None], other=0.0)
    v_old = tl.load(pool_ptr + (page_v * PAGE + in_page)[:, None] + d[None, :], mask=cached[:, None], other=0.0)
    writer = s_id == 0
    tok = tl.arange(0, 8)
    tok_ok = tok < M
    write_ok = tok_ok & writer
    pos = past + tok
    w_page_k = tl.load(bt_ptr + pos // P, mask=write_ok, other=0).to(tl.int64)
    w_page_v = tl.load(bt_ptr + bt_stride_kv + pos // P, mask=write_ok, other=0).to(tl.int64)
    slot = (h * P + pos % P) * D
    r = tl.arange(0, R)
    t = r // G
    row_ok = t < M
    q_head = h * G + r % G
    new_row = j - past
    cos_q, sin_q = _rope_rows(cos_sin_ptr, positions_ptr, t, row_ok, APPLY_ROPE)
    cos_k, sin_k = _rope_rows(cos_sin_ptr, positions_ptr, new_row, new, APPLY_ROPE)
    cos_w, sin_w = _rope_rows(cos_sin_ptr, positions_ptr, tok, write_ok, APPLY_ROPE)
    rm = s_id * RM + tl.arange(0, RM)
    rm_ok = (rm < R) & ((rm // G) < M)
    t_m = rm // G
    head_m = h * G + rm % G

    if LAUNCH_WITH_PDL:
        # From here on the raw QKV projection (the gate is a view of it): the producer's output.
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    g = tl.load(
        gate_ptr + t_m[:, None] * gate_token_stride + head_m[:, None] * gate_head_stride + d[None, :],
        mask=rm_ok[:, None],
        other=0.0,
    )
    if writer:
        k_w = _prep_rows(
            qkv_ptr, tok * qkv_stride, tl.zeros([8], tl.int32) + k_col, tok_ok, cos_w, sin_w, 1.0, prep_eps, 8,
            APPLY_ROPE,
        )
        v_w = tl.load(qkv_ptr + tok[:, None] * qkv_stride + v_col + d[None, :], mask=tok_ok[:, None], other=0.0)
        tl.store(
            pool_ptr + (w_page_k * PAGE + slot)[:, None] + d[None, :], _to_cache_fp8(k_w, scale_bf16),
            mask=tok_ok[:, None],
        )
        tl.store(
            pool_ptr + (w_page_v * PAGE + slot)[:, None] + d[None, :], _to_cache_fp8(v_w, scale_bf16),
            mask=tok_ok[:, None],
        )
    q = _prep_rows(
        qkv_ptr, t * qkv_stride, q_head * q_stride, row_ok, cos_q, sin_q, q_scale, prep_eps, R, APPLY_ROPE
    )
    q = tl.where(row_ok[:, None], q, 0.0)
    qk_scale = sm_scale_log2 * dequant
    if Q_FP8:
        q = _to_cache_fp8(q, scale_bf16).to(tl.bfloat16)
        qk_scale = qk_scale * dequant
    has_new = base + s_id * KEYS + KEYS > past
    k_new = tl.zeros([KEYS, D], tl.bfloat16)
    if has_new:
        k_new = _prep_rows(
            qkv_ptr, new_row * qkv_stride, tl.zeros([KEYS], tl.int32) + k_col, new, cos_k, sin_k, 1.0, prep_eps,
            KEYS, APPLY_ROPE,
        )
    v_new = tl.load(qkv_ptr + new_row[:, None] * qkv_stride + v_col + d[None, :], mask=new[:, None], other=0.0)
    k = tl.where(new[:, None], _to_cache_fp8(k_new, scale_bf16), k_old)
    v = tl.where(new[:, None], _to_cache_fp8(v_new, scale_bf16), v_old)
    sc = tl.dot(q, tl.trans(k.to(tl.bfloat16))) * qk_scale
    q_pos = past + t
    keep = (j[None, :] <= q_pos[:, None]) & (j[None, :] >= lo) & (j[None, :] > q_pos[:, None] - WINDOW)
    sc = tl.where(keep, sc, -float("inf"))
    m_i = tl.maximum(tl.full([R], -1.0e30, tl.float32), tl.max(sc, 1))
    pr = tl.math.exp2(sc - m_i[:, None])
    l_i = tl.sum(pr, 1)
    acc = tl.dot(pr.to(tl.bfloat16), v.to(tl.bfloat16))
    tl.store(part_o_ptr + (s_id * R + r)[:, None] * D + d[None, :], acc * dequant)
    tl.store(part_ml_ptr + (s_id * R + r) * 2, m_i)
    tl.store(part_ml_ptr + (s_id * R + r) * 2 + 1, l_i)

    # Group barrier: every thread's partial stores, then one release add per program; acquire spin.
    _fence_acq_rel_gpu(tl.zeros([R], tl.int32))
    tl.debug_barrier()
    tl.atomic_add(cnt_ptr + group * 2, 1, sem="release", scope="gpu")
    arrived = tl.atomic_add(cnt_ptr + group * 2, 0, sem="acquire", scope="gpu")
    while arrived < S:
        arrived = tl.atomic_add(cnt_ptr + group * 2, 0, sem="acquire", scope="gpu")
    tl.debug_barrier()

    sp = tl.arange(0, SP)
    s_ok = sp < S
    pidx = sp[:, None] * R + rm[None, :]
    mm = tl.load(part_ml_ptr + pidx * 2, mask=s_ok[:, None] & rm_ok[None, :], other=-1.0e30, cache_modifier=".cg")
    ll = tl.load(part_ml_ptr + pidx * 2 + 1, mask=s_ok[:, None] & rm_ok[None, :], other=0.0, cache_modifier=".cg")
    m_max = tl.max(mm, 0)
    w = tl.where(s_ok[:, None] & (ll > 0), tl.math.exp2(mm - m_max[None, :]), 0.0)
    den = tl.sum(w * ll, 0)
    o = tl.load(
        part_o_ptr + pidx[:, :, None] * D + d[None, None, :],
        mask=(s_ok[:, None] & rm_ok[None, :])[:, :, None],
        other=0.0,
        cache_modifier=".cg",
    )
    num = tl.sum(o * w[:, :, None], 0)
    attention = (num / den[:, None]).to(tl.bfloat16)
    x = attention.to(tl.float32)
    variance = _rows_norm_gate_sum_of_squares(x, RM) / D
    normed = (x * tl.rsqrt(variance + gate_eps)[:, None]).to(tl.bfloat16)
    sigmoid = tl.sigmoid(g.to(tl.float32)).to(tl.bfloat16)
    result = normed.to(tl.float32) * sigmoid.to(tl.float32)
    tl.store(
        out_ptr + t_m[:, None] * out_stride + head_m[:, None] * D + d[None, :], result, mask=rm_ok[:, None]
    )
    left = tl.atomic_add(cnt_ptr + group * 2 + 1, 1, sem="acq_rel", scope="gpu")
    if left == S - 1:
        tl.atomic_xchg(cnt_ptr + group * 2, 0, sem="relaxed", scope="gpu")
        tl.atomic_xchg(cnt_ptr + group * 2 + 1, 0, sem="relaxed", scope="gpu")


@triton.jit
def _swa_one2_kernel(
    qkv_ptr, qkv_stride, pool_ptr, bt_ptr, bt_stride_kv, seq_len_ptr, scale_ptr, cos_sin_ptr, positions_ptr,
    prep_eps, q_scale, part_o_ptr, part_ml_ptr, sm_scale_log2,
    out_ptr, out_stride, gate_ptr, gate_token_stride, gate_head_stride, gate_eps, cnt_ptr,
    M: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, P: tl.constexpr,
    WINDOW: tl.constexpr, KEYS: tl.constexpr, S: tl.constexpr, SP: tl.constexpr, RM: tl.constexpr,
    Q_FP8: tl.constexpr, QKV_PERMUTED: tl.constexpr, APPLY_ROPE: tl.constexpr, LAUNCH_WITH_PDL: tl.constexpr,
    ABL: tl.constexpr = 0, NM: tl.constexpr = 4,
):
    G: tl.constexpr = HQ // HKV
    R: tl.constexpr = 8 * G  # 64 query rows of one KV head: 8 token slots x 8 query heads
    PAGE: tl.constexpr = HKV * P * D
    s_id = tl.program_id(0)
    h = tl.program_id(1)
    req = tl.program_id(2)
    group = req * HKV + h
    qkv_ptr += req * M * qkv_stride
    bt_ptr += req * 2 * bt_stride_kv
    seq_len_ptr += req
    if APPLY_ROPE:
        positions_ptr += req * M
    out_ptr += req * M * out_stride
    gate_ptr += req * M * gate_token_stride
    part_o_ptr += group * S * R * D
    part_ml_ptr += group * S * R * 2
    d = tl.arange(0, D)
    q_stride = 2 * D
    k_col = 2 * HQ * D + h * D
    v_col = 2 * HQ * D + HKV * D + h * D
    if QKV_PERMUTED:
        q_stride = D
        k_col = HQ * D + h * D
        v_col = (HQ + HKV) * D + h * D

    # ---- before the wait: graph inputs, cached K/V (j < past), RoPE rows, the gate rows this program merges
    L = tl.load(seq_len_ptr)
    past = L - M
    lo = tl.maximum(past - WINDOW + 1, 0)
    base = (lo // 32) * 32
    j = base + s_id * KEYS + tl.arange(0, KEYS)
    live = j < L
    cached = (j >= lo) & (j < past) & live
    new = (j >= past) & live
    scale = tl.load(scale_ptr)
    scale_bf16 = scale.to(tl.bfloat16)
    dequant = 1.0 / scale
    page_k = tl.load(bt_ptr + j // P, mask=cached, other=0).to(tl.int64)
    page_v = tl.load(bt_ptr + bt_stride_kv + j // P, mask=cached, other=0).to(tl.int64)
    in_page = (h * P + j % P) * D
    k_old = tl.load(pool_ptr + (page_k * PAGE + in_page)[:, None] + d[None, :], mask=cached[:, None], other=0.0)
    v_old = tl.load(pool_ptr + (page_v * PAGE + in_page)[:, None] + d[None, :], mask=cached[:, None], other=0.0)
    holds_new = base + s_id * KEYS + KEYS > past
    tok = tl.arange(0, 16)
    tok_ok = tok < M
    write_ok = tok_ok & holds_new
    pos = past + tok
    w_page_k = tl.load(bt_ptr + pos // P, mask=write_ok, other=0).to(tl.int64)
    w_page_v = tl.load(bt_ptr + bt_stride_kv + pos // P, mask=write_ok, other=0).to(tl.int64)
    slot = (h * P + pos % P) * D
    r = tl.arange(0, R)
    t = r // G
    row_ok = t < M
    q_head = h * G + r % G
    new_row = j - past
    cos_q, sin_q = _rope_rows(cos_sin_ptr, positions_ptr, t, row_ok, APPLY_ROPE)
    cos_w, sin_w = _rope_rows(cos_sin_ptr, positions_ptr, tok, write_ok, APPLY_ROPE)

    if LAUNCH_WITH_PDL:
        # From here on the QKV projection (raw q/k/v and the gate, a strided view of it): the producer's output.
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    k16 = tl.zeros([16, D], tl.bfloat16)
    v16 = tl.zeros([16, D], tl.bfloat16)
    if holds_new:
        k_w = _prep_rows(qkv_ptr, tok * qkv_stride, tl.zeros([16], tl.int32) + k_col, tok_ok, cos_w, sin_w, 1.0,
                         prep_eps, 16, APPLY_ROPE)
        v_w = tl.load(qkv_ptr + tok[:, None] * qkv_stride + v_col + d[None, :], mask=tok_ok[:, None], other=0.0)
        k8 = _to_cache_fp8(k_w, scale_bf16)
        v8 = _to_cache_fp8(v_w, scale_bf16)
        tl.store(pool_ptr + (w_page_k * PAGE + slot)[:, None] + d[None, :], k8, mask=tok_ok[:, None])
        tl.store(pool_ptr + (w_page_v * PAGE + slot)[:, None] + d[None, :], v8, mask=tok_ok[:, None])
        k16 = tl.where(tok_ok[:, None], k8.to(tl.bfloat16), 0.0)
        v16 = tl.where(tok_ok[:, None], v8.to(tl.bfloat16), 0.0)

    if ABL >= 2:  # ablation: raw q, no prep
        q = tl.load(qkv_ptr + t[:, None] * qkv_stride + (q_head * q_stride)[:, None] + d[None, :],
                    mask=row_ok[:, None], other=0.0)
    else:
        q = _prep_rows(qkv_ptr, t * qkv_stride, q_head * q_stride, row_ok, cos_q, sin_q, q_scale, prep_eps, R,
                       APPLY_ROPE)
    q = tl.where(row_ok[:, None], q, 0.0)
    qk_scale = sm_scale_log2 * dequant
    if Q_FP8:
        q = _to_cache_fp8(q, scale_bf16).to(tl.bfloat16)
        qk_scale = qk_scale * dequant
    k = k_old.to(tl.bfloat16)
    v = v_old.to(tl.bfloat16)
    if holds_new:
        onehot = ((new_row[:, None] == tok[None, :]) & new[:, None]).to(tl.bfloat16)  # [KEYS, 16]
        k_ins = tl.dot(onehot, k16)
        v_ins = tl.dot(onehot, v16)
        k = tl.where(new[:, None], k_ins.to(tl.bfloat16), k)
        v = tl.where(new[:, None], v_ins.to(tl.bfloat16), v)
    if ABL >= 3:  # ablation: no MMA / softmax work
        sc = tl.zeros([R, KEYS], tl.float32) + tl.sum(k.to(tl.float32), 1)[None, :] * 0.0 + tl.sum(
            q.to(tl.float32), 1)[:, None] * 0.0
    else:
        sc = tl.dot(q, tl.trans(k)) * qk_scale
    q_pos = past + t
    keep = (j[None, :] <= q_pos[:, None]) & (j[None, :] >= lo) & (j[None, :] > q_pos[:, None] - WINDOW)
    sc = tl.where(keep, sc, -float("inf"))
    m_i = tl.maximum(tl.full([R], -1.0e30, tl.float32), tl.max(sc, 1))
    p = tl.math.exp2(sc - m_i[:, None])
    l_i = tl.sum(p, 1)
    if ABL >= 3:
        acc = tl.zeros([R, D], tl.float32) + tl.sum(v.to(tl.float32), 0)[None, :] * 0.0 + l_i[:, None] * 0.0
    else:
        acc = tl.dot(p.to(tl.bfloat16), v)
    tl.store(part_o_ptr + (s_id * R + r)[:, None] * D + d[None, :], acc * dequant)
    tl.store(part_ml_ptr + (s_id * R + r) * 2, m_i)
    tl.store(part_ml_ptr + (s_id * R + r) * 2 + 1, l_i)

    if ABL >= 1:  # ablation: partial only, no barrier / merge / norm-gate
        return
    # ---- arrive; the last NM arrivals of the group each merge 64 / NM rows once all S partials are stored
    _fence_acq_rel_gpu(tl.zeros([R], tl.int32))
    tl.debug_barrier()
    arrived = tl.atomic_add(cnt_ptr + group * 2, 1, sem="acq_rel", scope="gpu")
    if arrived >= S - NM:
        RM2: tl.constexpr = R // NM
        rm = (arrived - (S - NM)) * RM2 + tl.arange(0, RM2)
        rm_ok = (rm < R) & ((rm // G) < M)
        t_m = rm // G
        head_m = h * G + rm % G
        cur = tl.atomic_add(cnt_ptr + group * 2, 0, sem="acquire", scope="gpu")
        while cur < S:
            cur = tl.atomic_add(cnt_ptr + group * 2, 0, sem="acquire", scope="gpu")
        tl.debug_barrier()
        g = tl.load(gate_ptr + t_m[:, None] * gate_token_stride + head_m[:, None] * gate_head_stride + d[None, :],
                    mask=rm_ok[:, None], other=0.0)
        sp = tl.arange(0, SP)
        s_ok = sp < S
        pidx = sp[:, None] * R + rm[None, :]  # [SP, RM2]
        mm = tl.load(part_ml_ptr + pidx * 2, mask=s_ok[:, None] & rm_ok[None, :], other=-1.0e30, cache_modifier=".cg")
        ll = tl.load(part_ml_ptr + pidx * 2 + 1, mask=s_ok[:, None] & rm_ok[None, :], other=0.0, cache_modifier=".cg")
        m_max = tl.max(mm, 0)
        w = tl.where(s_ok[:, None] & (ll > 0), tl.math.exp2(mm - m_max[None, :]), 0.0)
        den = tl.sum(w * ll, 0)
        o = tl.load(part_o_ptr + pidx[:, :, None] * D + d[None, None, :],
                    mask=(s_ok[:, None] & rm_ok[None, :])[:, :, None], other=0.0, cache_modifier=".cg")
        num = tl.sum(o * w[:, :, None], 0)
        attention = (num / den[:, None]).to(tl.bfloat16)
        x = attention.to(tl.float32)
        variance = _rows_norm_gate_sum_of_squares(x, RM2) / D
        normed = (x * tl.rsqrt(variance + gate_eps)[:, None]).to(tl.bfloat16)
        sig = tl.sigmoid(g.to(tl.float32)).to(tl.bfloat16)
        res = normed.to(tl.float32) * sig.to(tl.float32)
        tl.store(out_ptr + t_m[:, None] * out_stride + head_m[:, None] * D + d[None, :], res.to(tl.bfloat16),
                 mask=rm_ok[:, None])
        left = tl.atomic_add(cnt_ptr + group * 2 + 1, 1, sem="acq_rel", scope="gpu")
        if left == NM - 1:
            tl.atomic_xchg(cnt_ptr + group * 2, 0, sem="relaxed", scope="gpu")
            tl.atomic_xchg(cnt_ptr + group * 2 + 1, 0, sem="relaxed", scope="gpu")


_ONE2_COUNTERS = {}


def _one2_counters(device: torch.device) -> Optional[torch.Tensor]:
    """The device's zeroed ``_swa_one2_kernel`` counters (each launch's mergers reset theirs), created on first use;
    ``None`` if it does not exist yet and a capture is running (never allocate inside a captured graph)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    buf = _ONE2_COUNTERS.get(index)
    if buf is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        buf = _ONE2_COUNTERS[index] = torch.zeros(4096, dtype=torch.int32, device=device)
    return buf


_ONE_LAUNCH_COUNTERS = {}


def _one_launch_counters(device: torch.device) -> Optional[torch.Tensor]:
    """The device's zeroed group-counter buffer (each launch's last programs reset theirs), created on first use;
    ``None`` if it does not exist yet and a capture is running (never allocate inside a captured graph)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    buf = _ONE_LAUNCH_COUNTERS.get(index)
    if buf is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        buf = _ONE_LAUNCH_COUNTERS[index] = torch.zeros(4096, dtype=torch.int32, device=device)
    return buf


def _one_launch_programs(window: int, rows: int) -> int:
    return triton.cdiv(max_blocks(window, rows) * BLOCK_N, _ONE_LAUNCH_KEYS)


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
    """

    ARCHS = frozenset({Arch.SM_103})

    note = """
    Metadata consumed: seq_lens[0], the sequence's K and V page ids, the
    positions (with fused_prep and RoPE), all on the device and read before
    griddepcontrol.wait; the layer's pool view; the FP8 KV scale. Call site:
    the target's sliding-window attention layers on decode steps of one
    sequence and at most 8 rows, under the swa_block fast path. Grids: split
    (num_splits, G, M*8/32), combine (M*8, G).

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
    ) -> torch.Tensor:
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
        if _SWA_CL8 == "all" and _cl8_eligible(qkv, kv_pool, num_q_heads, num_kv_heads, window, rows, q_fp8,
                                               fused_prep, gate, cos_sin, position_ids, 4):
            fn = _cl8_function(qkv.device, rope and positions.dtype == torch.int64, 4)
            if fn is not None:
                out = torch.empty((rows, num_q_heads * HEAD_DIM), dtype=torch.bfloat16, device=qkv.device)
                return _cl8_launch(fn, qkv, kv_pool, block_tables.unsqueeze(0) if block_tables.dim() == 2 else
                                   block_tables, seq_lens, kv_scale_orig_quant, cos_sin, positions, prep_eps, q_scale,
                                   gate, gate_eps, out, qkv_permuted, enable_pdl and _CL8_PDL, rope)
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
        split_grid = (num_splits, num_kv_heads, triton.cdiv(group_rows, ROWS_PER_PROGRAM))
        _swa_split_kernel[split_grid](
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
            **_split_maxnreg(math.prod(split_grid), qkv.device),
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
    ) -> torch.Tensor:
        """The three references in sequence: ``preattn_prep``'s (with ``fused_prep``), then
        ``paged_decode_attention``'s (which writes the step's rows into ``kv_pool``), then the
        norm-gate in float64 between its BF16 rounding points (with ``gate``).

        ``num_splits`` and ``enable_pdl`` change how the op is cut and launched, not what it
        computes, and are ignored here.
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

    Fusion boundary, PDL and pre-wait loads: as ``swa_block_attention``.
    """

    ARCHS = frozenset({Arch.SM_103})

    note = """
    Metadata consumed: seq_lens[0..B), each sequence's K and V page ids, the
    positions (with fused_prep and RoPE), all on the device and read before
    griddepcontrol.wait; the layer's pool view; the FP8 KV scale. Call site:
    the target's sliding-window attention layers on TP4 decode steps of 2..4
    sequences of equal row count (each request's spec-decoding verify), under
    the wide_swa_block fast path. Grids: split (num_splits, G, B*ceil(M*8/32)),
    combine (M*8, G, B).

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
    ) -> torch.Tensor:
        total = qkv.shape[0]
        reqs = seq_lens.numel()
        rows = total // reqs if reqs else 0
        page = kv_pool.shape[2] if kv_pool.dim() == 4 else 0
        width = (2 if fused_prep else 1) * (num_q_heads + num_kv_heads) * HEAD_DIM
        if not fused_prep:
            width += num_kv_heads * HEAD_DIM
        if (
            # ONE2 lifts the cap to 8 at TEP8 (2 KV heads), as kernels/swa_block/attention.py's caller does;
            # 4 KV heads keep 2..4.
            not 2 <= reqs <= (_ONE2_MAX_REQUESTS if _SWA_ONE2 and num_kv_heads == 2 else MULTI_MAX_REQUESTS)
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
        if _SWA_SHADOW and not _SHADOW["bypass"] and not torch.cuda.is_current_stream_capturing():
            return _shadow_compare(
                self,
                (qkv, kv_pool, block_tables, seq_lens, kv_scale_orig_quant, num_q_heads, num_kv_heads, window,
                 num_splits),
                dict(q_fp8=q_fp8, fused_prep=fused_prep, cos_sin=cos_sin, position_ids=position_ids, prep_eps=prep_eps,
                     q_scale=q_scale, qkv_permuted=qkv_permuted, gate=gate, gate_eps=gate_eps, enable_pdl=enable_pdl),
                rope, positions, rows, reqs)
        if _SWA_CL8 in ("1", "all") and _cl8_eligible(qkv, kv_pool, num_q_heads, num_kv_heads, window, rows, q_fp8,
                                                     fused_prep, gate, cos_sin, position_ids, 4):
            fn = _cl8_function(qkv.device, rope and positions.dtype == torch.int64, 4)
            if fn is not None:
                out = torch.empty((total, num_q_heads * HEAD_DIM), dtype=torch.bfloat16, device=qkv.device)
                return _cl8_launch(fn, qkv, kv_pool, block_tables, seq_lens, kv_scale_orig_quant, cos_sin, positions,
                                   prep_eps, q_scale, gate, gate_eps, out, qkv_permuted, enable_pdl and _CL8_PDL, rope)
        if (
            _TEP8_SWA_CL8 == "1"
            and _CL8_TEP8_MIN_REQUESTS <= reqs <= _CL8_TEP8_MAX_REQUESTS
            and page == BLOCK_N
            and _cl8_eligible(qkv, kv_pool, num_q_heads, num_kv_heads, window, rows, q_fp8, fused_prep, gate,
                              cos_sin, position_ids, 2)
        ):
            # The one-cluster SWA block at TEP8 (MUSESPARK_OPEN_TEP8_SWA_CL8=1).
            fn = _cl8_function(qkv.device, rope and positions.dtype == torch.int64, num_kv_heads)
            if fn is not None:
                out = torch.empty((total, num_q_heads * HEAD_DIM), dtype=torch.bfloat16, device=qkv.device)
                return _cl8_launch(fn, qkv, kv_pool, block_tables, seq_lens, kv_scale_orig_quant, cos_sin, positions,
                                   prep_eps, q_scale, gate, gate_eps, out, qkv_permuted, enable_pdl and _CL8_PDL, rope)
        out = torch.empty((total, num_q_heads * HEAD_DIM), dtype=torch.bfloat16, device=qkv.device)
        group_rows = rows * Q_PER_KV
        sms = None
        if _SWA_ONE_LAUNCH or _SWA_ROWS64 or _SWA_PREP_SPLIT or _SWA_ONE2:
            index = qkv.device.index if qkv.device.index is not None else torch.cuda.current_device()
            sms = _SM_COUNT.get(index)
            if sms is None:
                sms = _SM_COUNT[index] = torch.cuda.get_device_properties(index).multi_processor_count
        # The image's 32-row split grid; the alternatives apply only where it takes more than one wave.
        multi_wave = sms is not None and (
            num_splits * num_kv_heads * reqs * triton.cdiv(group_rows, ROWS_PER_PROGRAM)
            > _SPLIT_CTAS_PER_SM[None] * sms
        )
        if (
            _SWA_ONE2
            and fused_prep
            and gate is not None
            and (
                (num_kv_heads == 4 and 2 <= reqs <= MULTI_MAX_REQUESTS)
                or (num_kv_heads == 2 and MULTI_MAX_REQUESTS < reqs <= _ONE2_MAX_REQUESTS)
            )
            and page == BLOCK_N
            and rows <= MAX_ROWS
        ):
            counters = _one2_counters(qkv.device)
            programs = triton.cdiv(max_blocks(window, rows) * BLOCK_N, _ONE2_KEYS)
            groups = reqs * num_kv_heads
            if counters is not None and 2 * groups <= counters.numel() and programs >= 4:
                part_o = torch.empty((groups * programs * 64, HEAD_DIM), dtype=torch.float32, device=qkv.device)
                part_ml = torch.empty((groups * programs * 64, 2), dtype=torch.float32, device=qkv.device)
                options = {"maxnreg": 128} if programs * groups > sms else {}
                _swa_one2_kernel[(programs, num_kv_heads, reqs)](
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
                    out,
                    out.stride(0),
                    gate,
                    gate.stride(0),
                    gate_head_stride(gate),
                    gate_eps,
                    counters,
                    M=rows,
                    HQ=num_q_heads,
                    HKV=num_kv_heads,
                    D=HEAD_DIM,
                    P=page,
                    WINDOW=window,
                    KEYS=_ONE2_KEYS,
                    S=programs,
                    SP=triton.next_power_of_2(programs),
                    RM=4,
                    Q_FP8=q_fp8,
                    QKV_PERMUTED=qkv_permuted,
                    APPLY_ROPE=rope,
                    LAUNCH_WITH_PDL=enable_pdl,
                    # 16 mergers of 4 rows at TEP8 5..8 sequences (-0.8 us/layer at bs8 in the lab; same bits).
                    NM=16 if num_kv_heads == 2 else 4,
                    num_warps=8,
                    num_stages=1,
                    launch_pdl=enable_pdl,
                    **options,
                )
                return out
        if _SWA_ONE_LAUNCH and multi_wave and fused_prep and gate is not None:
            programs = _one_launch_programs(window, rows)
            counters = _one_launch_counters(qkv.device)
            if programs * num_kv_heads * reqs <= sms and counters is not None and 2 * reqs * num_kv_heads <= 4096:
                rows_merged = triton.next_power_of_2(triton.cdiv(64, programs))  # 64 rows: 8 slots x 8 heads
                part_o = torch.empty(
                    (reqs * num_kv_heads * programs * 64, HEAD_DIM), dtype=torch.float32, device=qkv.device
                )
                part_ml = torch.empty(
                    (reqs * num_kv_heads * programs * 64, 2), dtype=torch.float32, device=qkv.device
                )
                _swa_one_launch_kernel[(programs, num_kv_heads, reqs)](
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
                    out,
                    out.stride(0),
                    gate,
                    gate.stride(0),
                    gate_head_stride(gate),
                    gate_eps,
                    counters,
                    M=rows,
                    HQ=num_q_heads,
                    HKV=num_kv_heads,
                    D=HEAD_DIM,
                    P=page,
                    WINDOW=window,
                    KEYS=_ONE_LAUNCH_KEYS,
                    S=programs,
                    SP=triton.next_power_of_2(programs),
                    RM=rows_merged,
                    Q_FP8=q_fp8,
                    QKV_PERMUTED=qkv_permuted,
                    APPLY_ROPE=rope,
                    LAUNCH_WITH_PDL=enable_pdl,
                    num_warps=8,
                    num_stages=1,
                    launch_pdl=enable_pdl,
                )
                return out
        if _SWA_PREP_SPLIT and multi_wave and fused_prep:
            prepped, prep_gate = _preattn_prep(
                qkv,
                cos_sin if rope else None,
                positions if rope else None,
                num_q_heads,
                num_kv_heads,
                prep_eps,
                q_scale,
                qkv_permuted,
                enable_pdl=enable_pdl,
                trigger_at_entry=True,
            )
            return self(
                prepped,
                kv_pool,
                block_tables,
                seq_lens,
                kv_scale_orig_quant,
                num_q_heads,
                num_kv_heads,
                window,
                num_splits,
                q_fp8,
                fused_prep=False,
                gate=None if gate is None else prep_gate,
                gate_eps=gate_eps,
                enable_pdl=enable_pdl,
            )
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
        rows_per_program, split_warps, split_options = ROWS_PER_PROGRAM, NUM_WARPS, None
        if _SWA_ROWS64 and multi_wave:
            rows_per_program = 2 * ROWS_PER_PROGRAM
            split_warps = _ROWS64_WARPS.get((num_kv_heads, reqs), NUM_WARPS)
            split_options = {}
        split_grid = (num_splits, num_kv_heads, reqs * triton.cdiv(group_rows, rows_per_program))
        if split_options is None:
            # The caps were measured on the fused-prep build (144 registers); the unfused build is lighter.
            split_options = _split_maxnreg(math.prod(split_grid), qkv.device) if fused_prep else {}
        _swa_split_kernel[split_grid](
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
            R=rows_per_program,
            Q_FP8=q_fp8,
            FUSED_PREP=fused_prep,
            QKV_PERMUTED=qkv_permuted,
            APPLY_ROPE=rope,
            LAUNCH_WITH_PDL=enable_pdl,
            REQS=reqs,
            num_warps=split_warps,
            num_stages=NUM_STAGES,
            launch_pdl=enable_pdl,
            **split_options,
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
