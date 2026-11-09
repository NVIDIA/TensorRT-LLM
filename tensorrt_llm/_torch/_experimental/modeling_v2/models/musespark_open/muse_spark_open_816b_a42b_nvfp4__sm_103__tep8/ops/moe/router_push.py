# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE router logits as each rank's expert columns plus a multicast all-gather, in one kernel (TP16, TP8).

Moved into this target from ``tensorrt_llm/_torch/models/musespark_open_tep16_router_push.py``. Its one caller,
``kernels/router_gemm/router.py``, imports it on its first served call, never at import: importing it at module
level changed a TP4 context engine's first tokens (6 of 256 near-ties, deterministic) although no TP4 call
reaches it. The kernel is NVRTC CUDA (``extern "C" tp_router_push``), so its name does not depend on this path.

TP16: the MoE router logits as each
rank's 16 expert columns plus a multicast all-gather, in one kernel.

At TP16 the ``router_gemm`` fast path runs the image's BF16x3 split-K router GEMM over the whole replicated FP32
``[256, 8192]`` weight on every rank, then its split-K sum. A 16-column shard per rank
exchanged through a zero-padded MNNVL FP32 all-reduce is slower still. Here one
kernel ``tp_router_push`` does all of it:

* **Shard GEMM (CUDA cores, exact products).** Rank r computes experts [L r, L r + L) (L = 256 / ranks). The grid is
  one cluster of ``CL`` CTAs per 16-row block. CTA s of a cluster owns K range [s 8192/CL, (s+1) 8192/CL). Thread
  (row, expert group) accumulates ``fma(x, w, acc)`` (each product exact, one rounding per term) into 8 accumulators,
  8 consecutive K elements at a time (chains of 8192 / (CL * 8) terms), then adds them as a fixed tree.
* **Split reduction (cluster, deterministic).** Every CTA stores its partial into the cluster leader's shared memory
  (DSMEM); the leader adds the CL splits as a fixed pairwise tree and multiplies by the metap multiplier in FP32.
* **Exchange (Lamport, multicast).** The leader stores its ``[16, L]`` logits once to the multicast address of every
  rank's gather slot (buffer, row, L r + e), polls the block's full ``[16, 256]`` rows (stale-only, all loads in
  flight), writes the sentinel back and stores the logits. Sentinel: 0xFFFFFFFF (a NaN); published NaNs are
  canonicalized to 0x7FC00000. Buffer parity: a per-row-block call counter (each call adds 1 per row block, on every
  rank alike). Every wait is on the same-index cluster of another rank, and the grid is capped at the resident
  cluster capacity, so all clusters of a call are resident at once.

Values: not the image op's bits (another K split and summation order); both accumulate exact products in FP32, so
routing can differ only where two ranked scores are within FP32 rounding: numerics- and drift-gated. A row's logits do
not depend on the call's other rows, every rank holds the same bits, and a call is deterministic.

Armed by ``MUSESPARK_OPEN_TEP16_ROUTER_PUSH=1`` at TP16 (``kernels/router_gemm/router.py`` asks
``logits``); the transport is created at the first eager call (collective), and a call under CUDA graph capture before
that returns None (the caller runs the image's op).

Step A (the routing fold): with ``MUSESPARK_OPEN_TEP16_ROUTER_PUSH_ROUTE=1`` as well, the leader of
each 16-row block also runs the routing method's ``_full_routing_kernel`` on the block's logits, bit for bit (build
``ROUTE``): per row a warp takes 8 experts per lane, ``scores = sigmoid(logit)`` as Triton lowers ``tl.sigmoid`` at FP32
(``ex2.approx.f32`` of ``-x * log2e``, ``1 + e``, ``div.full.f32``), ``ranked = scores + bias`` with signed zeros made
equal, the top 8 of the key ``(ordered(ranked), 255 - expert)`` (largest first, ties to the lowest expert id) in eight
warp arg-max rounds, the picked unbiased scores summed in the kernel's order ``((p0+p1)+(p4+p5))+((p2+p3)+(p6+p7))``
(derived from that kernel), and ``div.rn`` by ``sum + eps``. It
writes the ``int32 [rows, 8]`` selected experts and ``FP32 [rows, 8]`` weights the routing memo then holds
(``router.py``), so the step's ``routing_method.apply`` calls return them without launching ``_full_routing_kernel``.
Knob: ``MUSESPARK_OPEN_TEP16_ROUTER_PUSH_ROUTE_MIN_ROWS`` (default 1): step A only from that many rows. Without the
second switch the build is the v1 source exactly (every addition sits under ``#if ROUTE``).

Step A v2: calls of at most ``MUSESPARK_OPEN_TEP16_ROUTER_PUSH_ROUTE_ROW_MAX`` rows (default 64, bs8)
route per row: the 15 non-leader CTAs of each cluster stay past the cluster barrier, take the buffer parity from the
cluster's call counter, and every CTA polls its own row (64 threads, one vector each), computes the row's 256 scores and
ordered keys one per thread, and warp 0 runs the same selection rounds and writes the row (v1 routed 16 rows per leader
CTA, two per warp). Above the threshold the call is v1's. The non-leaders' longer life holds their shared memory a few
microseconds more, which is why the larger calls (where the pool branch is co-critical) keep v1.

TEP8: the same kernel at TP8
(two trays of one NVLink domain), armed by ``MUSESPARK_OPEN_TEP8_ROUTER_PUSH=1`` (+ ``..._ROUTE=1`` for step A) on a TP8
mapping; the TEP16 switches keep arming TP16 only.
* ``WORLD 8``: rank r computes experts [32 r, 32 r + 32) as NEG = 2 expert-group clusters per 16-row block.
* Cluster size per world: ``MUSESPARK_OPEN_TEP8_ROUTER_PUSH_CLUSTER`` (default 8 at TP8: 1024 of K per CTA, 64 CTAs at
  64 rows, so the pool GEMM's 128 whole-SM CTAs start beside it). Every CTA still adds a contiguous power-of-two run of
  the same K tree, so the logits are the TEP16 push's bits exactly (any CL, any WORLD).
* Step A v2 (per-row routing) whenever NEG x CL == 16: CTA (expert group g, cluster rank c) routes row 16 rb + g CL + c,
  so each row is routed once, by one CTA, with the same epilogue (TP16: g = 0, CL = 16, unchanged).
* The v1 routing tile needs [16][256] floats: ``red`` is sized max(CL x 256, 16 x 256) (unchanged at CL 16).
"""

from __future__ import annotations

from ...sm107_defaults import env as _sm107_env

import ctypes
import os
from typing import Dict, Optional, Tuple

import torch

EXPERTS = 256
HIDDEN = 8192
MAX_ROWS = int(_sm107_env("MUSESPARK_OPEN_TEP16_ROUTER_PUSH_MAX_ROWS", "384"))
MIN_ROWS = int(_sm107_env("MUSESPARK_OPEN_TEP16_ROUTER_PUSH_MIN_ROWS", "9"))
CLUSTER = int(_sm107_env("MUSESPARK_OPEN_TEP16_ROUTER_PUSH_CLUSTER", "16"))
WORLDS = (16,)
#: TEP8: its own switches, row window and cluster size
TEP8_MIN_ROWS = int(_sm107_env("MUSESPARK_OPEN_TEP8_ROUTER_PUSH_MIN_ROWS", "9"))
TEP8_MAX_ROWS = int(_sm107_env("MUSESPARK_OPEN_TEP8_ROUTER_PUSH_MAX_ROWS", "128"))
TEP8_CLUSTER = int(_sm107_env("MUSESPARK_OPEN_TEP8_ROUTER_PUSH_CLUSTER", "8"))


def _tep8_armed() -> bool:
    return _sm107_env("MUSESPARK_OPEN_TEP8_ROUTER_PUSH", "0") == "1"


def _tep16_armed() -> bool:
    return _sm107_env("MUSESPARK_OPEN_TEP16_ROUTER_PUSH", "0") == "1"


def _world_rows(world: int):
    """(min rows, max rows, cluster) of the push at this TP size."""
    if world == 8:
        return TEP8_MIN_ROWS, min(TEP8_MAX_ROWS, MAX_ROWS), TEP8_CLUSTER
    return MIN_ROWS, MAX_ROWS, CLUSTER
BUFFERS = 2
THREADS = 256
SENTINEL_I32 = -1


TOP_K = 8
#: Step A only from this many rows (rank-uniform); below it the routing method computes its own routing.
ROUTE_MIN_ROWS = int(_sm107_env("MUSESPARK_OPEN_TEP16_ROUTER_PUSH_ROUTE_MIN_ROWS", "1"))
#: Step A v2: calls of at most this many rows route per row in every CTA of the cluster; v1 above.
ROUTE_ROW_MAX = int(_sm107_env("MUSESPARK_OPEN_TEP16_ROUTER_PUSH_ROUTE_ROW_MAX", "64"))


def armed() -> bool:
    return _tep16_armed() or _tep8_armed()


def _world_armed(world) -> bool:
    """Each switch arms its own TP size."""
    return (world == 16 and _tep16_armed()) or (world == 8 and _tep8_armed())


def route_armed() -> bool:
    """Step A: the push also writes the routing method's selection and weights."""
    return (_tep16_armed() and _sm107_env("MUSESPARK_OPEN_TEP16_ROUTER_PUSH_ROUTE", "0") == "1") or (
        _tep8_armed() and _sm107_env("MUSESPARK_OPEN_TEP8_ROUTER_PUSH_ROUTE", "0") == "1")


def gather_bytes(max_rows: int = MAX_ROWS) -> int:
    return BUFFERS * max_rows * EXPERTS * 4


SRC = r"""
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
#ifndef WORLD
#define WORLD 16
#endif
#ifndef CL
#define CL 16
#endif
#ifndef MAX_ROWS
#define MAX_ROWS 384
#endif
#ifndef SUB
#define SUB 4          // exact-product FMAs per leaf of the K tree
#endif
#define HID 8192
#define E 256
#define L (E / WORLD)  // experts of this rank
#define EG 16          // experts per cluster tile
#define NEG (L / EG)   // expert groups (clusters) per row block: 1 at TP16, 4 at TP4
#define BM 16          // rows per cluster tile
#define NL (BM * EG)   // 256 logits per cluster tile
#define NT 256         // threads per CTA
#define P 8            // contiguous K parts of a CTA (adjacent lanes), combined by shuffles
#define RB 4           // rows of a thread's logit block
#define EB 2           // experts of a thread's logit block
#define KC (HID / CL)  // K per CTA
#define KP (KC / P)    // K per part
#define NCH (KP / 8)   // 8-element chunks per part
#define QW (KP + 4)    // padded weight part (floats): the 8 parts of an expert hit disjoint banks
#define QX (KP + 8)    // padded activation part (BF16): the 8 parts of a row hit disjoint banks
#define SENT 0xFFFFFFFFu
#ifndef TIMING
#define TIMING 0
#endif
#if TIMING
#define STAMP(i) do { if (t == 0) { u64 g_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g_)); tmv[i] = g_; } } while (0)
#define FLUSH() do { if (t == 0 && timing) { for (int i_ = 0; i_ < 8; ++i_) timing[blockIdx.x * 8 + i_] = tmv[i_]; } } while (0)
#else
#define STAMP(i) do { } while (0)
#define FLUSH() do { } while (0)
#endif
static_assert(L % EG == 0 && (NCH & (NCH - 1)) == 0 && NT == (NL / (RB * EB)) * P && 8 % SUB == 0, "layout");
constexpr int ilog2(int n) { return n <= 1 ? 0 : 1 + ilog2(n / 2); }
static_assert(ilog2(NCH) < 6, "K tree depth");
#ifndef ROUTE
#define ROUTE 0
#endif
#ifndef ROUTE_LAB
#define ROUTE_LAB 0
#endif
#if ROUTE
// The weight sum's shuffle partners (element xor distances), the routing kernel's tree for a [256] FP32 sum with 2
// elements per thread: ((p0+p1)+(p4+p5))+((p2+p3)+(p6+p7)). The lab build can try other orders.
#ifndef TREE_S0
#define TREE_S0 1
#endif
#ifndef TREE_S1
#define TREE_S1 4
#endif
#ifndef TREE_S2
#define TREE_S2 2
#endif
#define TOP_K 8
#define ROUTE_PARAMS , const float* bias, float eps, int* sel, float* wts
// tl.sigmoid at FP32, as Triton lowers it: 1 / (1 + exp(-x)), exp(y) = ex2.approx.f32(y * log2e), "/" = div.full.f32
__device__ __forceinline__ float sig_tri(float x) {
  const float y = __fmul_rn(-x, 1.44269504088896340736f);
  float e, s;
  asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(y));
  const float d = __fadd_rn(1.0f, e);
  asm("div.full.f32 %0, %1, %2;" : "=f"(s) : "f"(1.0f), "f"(d));
  return s;
}
// _full_routing_kernel (EXPERTS 256, TORCH_ORDER off) on the [BM][E] logits tile lgb (shared), rows row0.. of the call.
// Warp w takes rows w and w + NW together (interleaved, independent chains); lane l holds experts l + 32 j. The top 8
// in 8 rounds of two warp reductions in hardware: the largest ordered ranked score (REDUX.MAX), then the lowest expert
// id among the lanes holding it (REDUX.MIN), which is the kernel's (score, 255 - id) order. Writes sel / wts rows
// [row0, min(row0 + BM, rows)).
#define NW (NT / 32)
#define RPW (BM / NW)  // rows per warp
static_assert(BM % NW == 0, "rows per warp");
__device__ __forceinline__ void route_block(const float* lgb, const float* bias_s, float eps, int row0, int rows,
                                            int* sel, float* wts, int t) {
  const int warp = t >> 5, lane = t & 31;
  if (row0 + warp >= rows) return;  // warp-uniform: the warp's first row is past the call (so are the others)
  u32 ord[RPW][8];
#pragma unroll
  for (int q = 0; q < RPW; ++q) {
    const float* rowp = lgb + (warp + q * NW) * E;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int e = lane + 32 * j;
      float r = __fadd_rn(sig_tri(rowp[e]), bias_s[e]);
      r = r == 0.0f ? 0.0f : r;  // signed zeros equal before the id tie-break
      const u32 b = __float_as_uint(r);
      ord[q][j] = (b >> 31) ? ~b : (b ^ 0x80000000u);
    }
  }
  u32 taken[RPW];
  int myid[RPW];
#pragma unroll
  for (int q = 0; q < RPW; ++q) { taken[q] = 0; myid[q] = 0; }
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    u32 lb[RPW], lj[RPW];
#pragma unroll
    for (int q = 0; q < RPW; ++q) {  // this lane's best remaining: largest ord, the lowest j on a tie
      lb[q] = 0; lj[q] = 8;
#pragma unroll
      for (int j = 7; j >= 0; --j)
        if (!((taken[q] >> j) & 1u) && (lj[q] == 8 || ord[q][j] >= lb[q])) { lb[q] = ord[q][j]; lj[q] = j; }
    }
    u32 m[RPW];
#pragma unroll
    for (int q = 0; q < RPW; ++q) m[q] = __reduce_max_sync(0xffffffffu, lb[q]);
    u32 id[RPW];
#pragma unroll
    for (int q = 0; q < RPW; ++q)
      id[q] = __reduce_min_sync(0xffffffffu, lb[q] == m[q] ? (u32)(lane + 32 * (int)lj[q]) : 0xffffffffu);
#pragma unroll
    for (int q = 0; q < RPW; ++q) {
      if ((int)(id[q] & 31u) == lane) taken[q] |= 1u << (id[q] >> 5);
      if (lane == k) myid[q] = (int)id[q];
    }
  }
#pragma unroll
  for (int q = 0; q < RPW; ++q) {
    const int rr = warp + q * NW, grow = row0 + rr;
    const float* rowp = lgb + rr * E;
    const float p = lane < TOP_K ? sig_tri(rowp[myid[q]]) : 0.0f;
    float a = __fadd_rn(p, __shfl_xor_sync(0xffffffffu, p, TREE_S0));
    a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, TREE_S1));
    a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, TREE_S2));
    const float denom = __fadd_rn(a, eps);
    if (lane < TOP_K && grow < rows) {
      sel[(long long)grow * TOP_K + lane] = myid[q];
      wts[(long long)grow * TOP_K + lane] = __fdiv_rn(p, denom);
    }
  }
}
#else
#define ROUTE_PARAMS
#endif

__device__ __forceinline__ float bf_lo(u32 w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(u32 w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ uint4 ld_poll(const void* p) {
  uint4 q;
  asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(q.x), "=r"(q.y), "=r"(q.z), "=r"(q.w)
               : "l"(p) : "memory");
  return q; }
__device__ __forceinline__ void st_plain(void* p, uint4 q) {
  asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(p), "r"(q.x), "r"(q.y), "r"(q.z), "r"(q.w)
               : "memory"); }
// 16-byte global -> shared copy (L2 only); src_bytes 0 zero-fills
__device__ __forceinline__ void cp16(void* sdst, const void* gsrc, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
               :: "r"((u32)__cvta_generic_to_shared(sdst)), "l"(gsrc), "r"(src_bytes) : "memory"); }
__device__ __forceinline__ bool stale4(uint4 q) { return (q.x == SENT) | (q.y == SENT) | (q.z == SENT) | (q.w == SENT); }
__device__ __forceinline__ u32 canon(float f) {
  const u32 b = __float_as_uint(f);
  return ((b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu)) ? 0x7fc00000u : b; }
__device__ __forceinline__ long long goff(u32 buf, int row, int col) {
  return (((long long)buf * MAX_ROWS + row) * E + col) * 4; }

#if ROUTE
// v2: calls of at most ROUTE_ROW_MAX rows route per row in every CTA of the cluster (CTA c: row 16 rb + c),
// v1's leader epilogue above. Poll by 64 threads (one 16-byte vector each), then one expert per thread (sigmoid, ranked,
// ordered key into shared memory), then warp 0 selects with the same REDUX rounds and writes the row.
#ifndef ROUTE_ROW_MAX
#define ROUTE_ROW_MAX 64
#endif
static_assert(NT == E, "one expert per thread in the per-row epilogue");
__device__ __forceinline__ u32 ld_relaxed_gpu(const u32* p) {
  u32 v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v; }
__device__ __forceinline__ void route_row_cta(char* gather, u32 buf, int grow, float* out, float* scratch,
                                              const float* bias_s, float eps, int* sel, float* wts, int t) {
  float* lgr = scratch;                                  // [E] logits
  u32* ords = reinterpret_cast<u32*>(scratch + E);       // [E] ordered ranked keys
  float* scs = scratch + 2 * E;                          // [E] unbiased scores
  if (t < E / 4) {
    char* gp = gather + goff(buf, grow, 4 * t);
    uint4 q;
    do { q = ld_poll(gp); } while (stale4(q));
    st_plain(gp, make_uint4(SENT, SENT, SENT, SENT));
    st_plain(out + (long long)grow * E + 4 * t, q);
    *reinterpret_cast<uint4*>(lgr + 4 * t) = q;
  }
  __syncthreads();
  {
    const float s = sig_tri(lgr[t]);
    float r = __fadd_rn(s, bias_s[t]);
    r = r == 0.0f ? 0.0f : r;  // signed zeros equal before the id tie-break
    const u32 b = __float_as_uint(r);
    ords[t] = (b >> 31) ? ~b : (b ^ 0x80000000u);
    scs[t] = s;
  }
  __syncthreads();
  if (t >= 32) return;
  const int lane = t;
  u32 ord[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) ord[j] = ords[lane + 32 * j];
  u32 taken = 0;
  int myid = 0;
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    u32 lb = 0, lj = 8;
#pragma unroll
    for (int j = 7; j >= 0; --j)
      if (!((taken >> j) & 1u) && (lj == 8 || ord[j] >= lb)) { lb = ord[j]; lj = j; }
    const u32 m = __reduce_max_sync(0xffffffffu, lb);
    const u32 id = __reduce_min_sync(0xffffffffu, lb == m ? (u32)(lane + 32 * (int)lj) : 0xffffffffu);
    if ((int)(id & 31u) == lane) taken |= 1u << (id >> 5);
    if (lane == k) myid = (int)id;
  }
  const float p = lane < TOP_K ? scs[myid] : 0.0f;
  float a = __fadd_rn(p, __shfl_xor_sync(0xffffffffu, p, TREE_S0));
  a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, TREE_S1));
  a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, TREE_S2));
  const float denom = __fadd_rn(a, eps);
  if (lane < TOP_K) {
    sel[(long long)grow * TOP_K + lane] = myid;
    wts[(long long)grow * TOP_K + lane] = __fdiv_rn(p, denom);
  }
}
#endif
// A logit's K tree is one balanced binary tree over the 1024 8-element chunks of K (leaves: SUB exact-product FMAs
// from 0, the leaves of a chunk added in order; then chunks added pairwise): a thread adds its part's NCH contiguous
// chunks (binary-counter stack) for each logit of its RB x EB block, the P parts of a logit are added by shuffles
// (xor 1, 2, 4: adjacent lanes), the cluster leader adds the CL CTAs pairwise; then the multiplier. Any such split
// keeps the same tree, hence the same bits.
extern "C" __global__ void __launch_bounds__(NT)
tp_router_push(const u16* x, const float* w, float scale, float* out, u32* rbcnt, char* gather, char* gather_mc,
               int rows, int rank, u64* timing ROUTE_PARAMS) {
#if TIMING
  u64 tmv[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
  extern __shared__ __align__(16) unsigned char dyn[];
  u16* xs = reinterpret_cast<u16*>(dyn);                          // [BM][P][QX] BF16
  float* ws = reinterpret_cast<float*>(dyn + BM * P * QX * 2);    // [EG][P][QW] FP32
#if ROUTE
  // TP8: the v1 routing tile ([BM][E] floats) reuses red, which CL < 16 makes smaller than that
  __shared__ __align__(16) float red[(CL * NL > BM * E) ? CL * NL : BM * E];
#else
  __shared__ __align__(16) float red[CL * NL];
#endif
  __shared__ __align__(16) float lg[NL];
  __shared__ u32 s_buf;
#if ROUTE
  static_assert(((CL * NL > BM * E) ? CL * NL : BM * E) >= BM * E, "the routing tile reuses red");
  __shared__ __align__(16) float bias_s[E];
#endif
  const int t = threadIdx.x, lane = t & 31;
  u32 crank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
  const int cid = blockIdx.x / CL;
  const int rb = cid / NEG, eg = cid % NEG;
  const int part = lane % P;
  const int blk = (t / 32) * (32 / P) + lane / P;     // logit block 0..31
  const int rg = blk / (EG / EB), eb = blk % (EG / EB);
  const int kb = (int)crank * KC;
  const int e0 = rank * L + eg * EG;  // first expert (row of w) of this cluster tile
  STAMP(0);
  // every CTA of the cluster is running before any DSMEM store (the matching wait comes right before them)
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  // the weight tile is immutable: its copy may precede the dependency wait
  for (int i = t; i < EG * (KC / 4); i += NT) {
    const int er = i / (KC / 4), k = (i % (KC / 4)) * 4;
    cp16(ws + (er * P + k / KP) * QW + k % KP, w + (long long)(e0 + er) * HID + kb + k, 16);
  }
#if ROUTE
  // the expert bias is a parameter (immutable): the leader stages it before the dependency wait as well
  if (t < E / 4) cp16(bias_s + 4 * t, bias + 4 * t, 16);  // every CTA: the per-row epilogue (v2) reads it
#endif
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  STAMP(1);
  if (crank == 0 && t == 0) {
    const u32 old = atomicAdd(rbcnt + cid, 1u);
    s_buf = old & 1u;
  }
  // the activation tile, after the wait (the predecessor wrote it); rows past the call zero-filled
  for (int i = t; i < BM * (KC / 8); i += NT) {
    const int rr = i / (KC / 8), k = (i % (KC / 8)) * 8;
    const int grow = rb * BM + rr;
    cp16(xs + (rr * P + k / KP) * QX + k % KP, x + (long long)(grow < rows ? grow : 0) * HID + kb + k,
         grow < rows ? 16 : 0);
  }
  asm volatile("cp.async.commit_group;\n\tcp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  STAMP(2);
  float lvl[RB][EB][ilog2(NCH) + 1];
#pragma unroll
  for (int c = 0; c < NCH; ++c) {
    float xv[RB][8], wv[EB][8];
#pragma unroll
    for (int i = 0; i < RB; ++i) {
      const uint4 q = *reinterpret_cast<const uint4*>(xs + ((rg * RB + i) * P + part) * QX + 8 * c);
      xv[i][0] = bf_lo(q.x); xv[i][1] = bf_hi(q.x); xv[i][2] = bf_lo(q.y); xv[i][3] = bf_hi(q.y);
      xv[i][4] = bf_lo(q.z); xv[i][5] = bf_hi(q.z); xv[i][6] = bf_lo(q.w); xv[i][7] = bf_hi(q.w);
    }
#pragma unroll
    for (int j = 0; j < EB; ++j) {
      const float* src = ws + ((eb * EB + j) * P + part) * QW + 8 * c;
      const float4 wa = *reinterpret_cast<const float4*>(src), wb = *reinterpret_cast<const float4*>(src + 4);
      wv[j][0] = wa.x; wv[j][1] = wa.y; wv[j][2] = wa.z; wv[j][3] = wa.w;
      wv[j][4] = wb.x; wv[j][5] = wb.y; wv[j][6] = wb.z; wv[j][7] = wb.w;
    }
#pragma unroll
    for (int i = 0; i < RB; ++i)
#pragma unroll
      for (int j = 0; j < EB; ++j) {
        float v = 0.0f;
#pragma unroll
        for (int h = 0; h < 8 / SUB; ++h) {
          float leaf = 0.0f;
#pragma unroll
          for (int u = 0; u < SUB; ++u) leaf = __fmaf_rn(xv[i][h * SUB + u], wv[j][h * SUB + u], leaf);
          v = h == 0 ? leaf : __fadd_rn(v, leaf);
        }
#pragma unroll
        for (int b = 0; b <= ilog2(NCH); ++b) {
          if ((c >> b) & 1) {
            v = __fadd_rn(lvl[i][j][b], v);
          } else {
            lvl[i][j][b] = v;
            break;
          }
        }
      }
  }
  float pv[RB][EB];
#pragma unroll
  for (int i = 0; i < RB; ++i)
#pragma unroll
    for (int j = 0; j < EB; ++j) {
      float v = lvl[i][j][ilog2(NCH)];
#pragma unroll
      for (int m = 1; m < P; m *= 2) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, m));
      pv[i][j] = v;
    }
  STAMP(3);
  // this CTA's partials into slots [crank][logit] of the leader's shared memory
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
  if (part == 0) {
#pragma unroll
    for (int i = 0; i < RB; ++i)
#pragma unroll
      for (int j = 0; j < EB; ++j) {
        const u32 local = (u32)__cvta_generic_to_shared(&red[(int)crank * NL + (rg * RB + i) * EG + eb * EB + j]);
        u32 remote;
        asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(remote) : "r"(local));
        asm volatile("st.shared::cluster.f32 [%0], %1;" :: "r"(remote), "f"(pv[i][j]) : "memory");
      }
  }
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
  STAMP(4);
#if ROUTE
  // v2: every CTA of the row block's NEG clusters routes one row (TP8: CTA (eg, crank) takes row eg CL + crank)
  const bool per_row = NEG * CL == BM && rows <= ROUTE_ROW_MAX;
  if (crank != 0 && !per_row) { FLUSH(); return; }
  u32 buf = 0;
  if (crank == 0) {
#else
  if (crank != 0) { FLUSH(); return; }
#endif
  // ---- leader: the CL splits pairwise, the multiplier, publication; expert group 0 then takes the block's rows ----
  {
    float v[CL];
#pragma unroll
    for (int s = 0; s < CL; ++s) v[s] = red[s * NL + t];
#pragma unroll
    for (int span = 1; span < CL; span *= 2)
#pragma unroll
      for (int s = 0; s < CL; s += 2 * span) v[s] = __fadd_rn(v[s], v[s + span]);
    lg[t] = __fmul_rn(v[0], scale);
  }
  __syncthreads();
#if ROUTE
  buf = s_buf;
#else
  const u32 buf = s_buf;
#endif
  if (t < NL / 4) {
    const int rr = t / (EG / 4), cc = (t % (EG / 4)) * 4;
    const int grow = rb * BM + rr;
    if (grow < rows) {
      const float* src = &lg[rr * EG + cc];
      st_plain(gather_mc + goff(buf, grow, e0 + cc),
               make_uint4(canon(src[0]), canon(src[1]), canon(src[2]), canon(src[3])));
    }
  }
#if ROUTE
  }
#endif
  STAMP(5);
#if ROUTE
  if (per_row) {
    // the buffer parity from the cluster's call counter: the leader's atomic precedes the cluster barrier (release /
    // acquire), and the next call's atomic waits for this grid (griddepcontrol.wait)
    if (crank != 0) buf = (ld_relaxed_gpu(rbcnt + cid) - 1u) & 1u;
    const int grow = rb * BM + eg * CL + (int)crank;
    if (grow < rows) route_row_cta(gather, buf, grow, out, red, bias_s, eps, sel, wts, t);
    FLUSH();
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    return;
  }
#endif
  if (eg != 0) { FLUSH(); return; }
  // the block's rows: BM x 256 FP32 = 1024 vectors, 4 per thread, all polls in flight
  constexpr int PV = BM * E / 4 / NT;
  uint4 q[PV];
  char* gp[PV];
  u32 pend = 0;
#pragma unroll
  for (int i = 0; i < PV; ++i) {
    const int v = t + i * NT;
    const int grow = rb * BM + v / (E / 4);
    gp[i] = gather + goff(buf, grow < rows ? grow : 0, (v % (E / 4)) * 4);
    if (grow < rows) pend |= 1u << i;
  }
  const u32 on = pend;
  while (pend) {
#pragma unroll
    for (int i = 0; i < PV; ++i)
      if ((pend >> i) & 1u) q[i] = ld_poll(gp[i]);
    u32 next = 0;
#pragma unroll
    for (int i = 0; i < PV; ++i)
      if (((pend >> i) & 1u) && stale4(q[i])) next |= 1u << i;
    pend = next;
  }
  STAMP(6);
  const uint4 S4 = make_uint4(SENT, SENT, SENT, SENT);
#pragma unroll
  for (int i = 0; i < PV; ++i) {
    if (!((on >> i) & 1u)) continue;
    const int v = t + i * NT;
    st_plain(gp[i], S4);
    st_plain(out + (long long)(rb * BM + v / (E / 4)) * E + (v % (E / 4)) * 4, q[i]);
  }
  STAMP(7);
#if ROUTE
  // step A: the block's rows into the (consumed) split-partial buffer, then the routing method's selection
  {
    float* lgb = red;
#pragma unroll
    for (int i = 0; i < PV; ++i) {
      if (!((on >> i) & 1u)) continue;
      const int v = t + i * NT;
      *reinterpret_cast<uint4*>(lgb + (v / (E / 4)) * E + (v % (E / 4)) * 4) = q[i];
    }
    __syncthreads();
    route_block(lgb, bias_s, eps, rb * BM, rows, sel, wts, t);
  }
#endif
  FLUSH();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

#if ROUTE_LAB
// Lab entry (one GPU, no exchange): the same routing on [rows, 256] FP32 logits from global memory, one CTA per 16-row
// block, the tile staged as the push's leader stages it.
extern "C" __global__ void __launch_bounds__(NT)
route_lab(const float* logits, const float* bias, float eps, int* sel, float* wts, int rows) {
  __shared__ __align__(16) float lgb[BM * E];
  __shared__ __align__(16) float bias_s[E];
  const int t = threadIdx.x, row0 = blockIdx.x * BM;
  if (t < E / 4) *reinterpret_cast<float4*>(bias_s + 4 * t) = *reinterpret_cast<const float4*>(bias + 4 * t);
  for (int v = t; v < BM * E / 4; v += NT) {
    const int grow = row0 + v / (E / 4);
    if (grow < rows)
      *reinterpret_cast<float4*>(lgb + 4 * v) = *reinterpret_cast<const float4*>(logits + (long long)row0 * E + 4 * v);
  }
  __syncthreads();
  route_block(lgb, bias_s, eps, row0, rows, sel, wts, t);
}
#endif
"""


def _chk(r):
    if int(r[0]) != 0:
        raise RuntimeError(f"tp_router_push: CUDA/NVRTC error {r!r}")
    return r[1] if len(r) == 2 else r[1:]


class Kernel:
    """One compiled build of ``tp_router_push`` on one device: ``tpl`` threads per logit (4, 2 or 1; every build
    computes the same bits), its resources and its resident-cluster capacity."""

    def __init__(self, device_index: int, world: int, cluster: int = CLUSTER, tpl: int = 4,
                 max_rows: int = MAX_ROWS, extra: Tuple[str, ...] = (), timing: bool = False, route: bool = False):
        from cuda.bindings import driver as cu
        from cuda.bindings import nvrtc

        self.cu = cu
        self.route = bool(route)
        major, minor = torch.cuda.get_device_capability(device_index)
        prog = _chk(nvrtc.nvrtcCreateProgram(SRC.encode(), b"tp_router_push.cu", 0, [], []))
        opts = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"-std=c++17", b"-default-device",
                f"-DWORLD={int(world)}".encode(), f"-DCL={int(cluster)}".encode(),
                f"-DMAX_ROWS={int(max_rows)}".encode(), f"-DTIMING={int(bool(timing))}".encode()] + \
            ([b"-DROUTE=1", f"-DROUTE_ROW_MAX={ROUTE_ROW_MAX}".encode()] if self.route else []) + \
            [e.encode() for e in extra]
        r = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if int(r[0]) != 0:
            n = _chk(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("tp_router_push NVRTC compile failed:\n" + log.decode(errors="replace"))
        n = _chk(nvrtc.nvrtcGetCUBINSize(prog))
        self.cubin = b" " * n
        _chk(nvrtc.nvrtcGetCUBIN(prog, self.cubin))
        self.world, self.cluster, self.tpl, self.max_rows = world, cluster, tpl, max_rows
        self.threads = 256
        kp = HIDDEN // cluster // 8
        self.dyn_smem = 16 * 8 * (kp + 8) * 2 + 16 * 8 * (kp + 4) * 4
        self.bm = 16
        self.neg = (EXPERTS // world) // 16
        with torch.cuda.device(device_index):
            self.module = _chk(cu.cuModuleLoadData(self.cubin))
            self.fn = _chk(cu.cuModuleGetFunction(self.module, b"tp_router_push"))
            attr = cu.CUfunction_attribute
            if cluster > 8:
                _chk((cu.cuFuncSetAttribute(self.fn, attr.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1)[0],
                      None))
            _chk((cu.cuFuncSetAttribute(self.fn, attr.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                        self.dyn_smem)[0], None))
            self.regs = int(_chk(cu.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_NUM_REGS, self.fn)))
            self.local_bytes = int(_chk(cu.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, self.fn)))
            self.smem = int(_chk(cu.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, self.fn))) + \
                self.dyn_smem
            cfg = self._make_config(cluster, cu.CUstream(torch.cuda.current_stream().cuda_stream), False)
            self.max_clusters = int(_chk(cu.cuOccupancyMaxActiveClusters(self.fn, cfg)))
        self._cfg: Dict[Tuple[int, int, bool], object] = {}

    def clusters(self, rows: int) -> int:
        return (rows + self.bm - 1) // self.bm * self.neg

    def fits(self, rows: int) -> bool:
        """Every cluster of a call resident at once (a leader waits on the same cluster of every rank)."""
        return 1 <= rows <= self.max_rows and self.clusters(rows) <= self.max_clusters

    def _make_config(self, grid: int, stream, pdl: bool):
        cu = self.cu
        cfg = cu.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = self.threads, 1, 1
        cfg.sharedMemBytes = self.dyn_smem
        cfg.hStream = stream
        a1 = cu.CUlaunchAttribute()
        a1.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        a1.value.clusterDim.x, a1.value.clusterDim.y, a1.value.clusterDim.z = self.cluster, 1, 1
        attrs = [a1]
        if pdl:
            a2 = cu.CUlaunchAttribute()
            a2.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
            a2.value.programmaticStreamSerializationAllowed = 1
            attrs.append(a2)
        cfg.attrs = attrs
        cfg.numAttrs = len(attrs)
        return cfg

    def launch(self, rows: int, x: int, w: int, scale: float, out: int, rbcnt: int, gather: int, gather_mc: int,
               rank: int, stream: int, pdl: bool = True, timing: int = 0,
               route: Optional[Tuple[int, float, int, int]] = None):
        assert self.fits(rows), (rows, self.clusters(rows), self.max_clusters)
        assert (route is not None) == self.route, "a ROUTE build takes (bias, eps, selected, weights), others none"
        grid = self.clusters(rows) * self.cluster
        key = (grid, stream, pdl)
        cfg = self._cfg.get(key)
        if cfg is None:
            cfg = self._make_config(grid, self.cu.CUstream(stream), pdl)
            self._cfg[key] = cfg
        vp, fl, ii = ctypes.c_void_p, ctypes.c_float, ctypes.c_int
        vals = (x, w, float(scale), out, rbcnt, gather, gather_mc, int(rows), int(rank), timing)
        types = (vp, vp, fl, vp, vp, vp, vp, ii, ii, vp)
        if route is not None:
            bias, eps, sel, wts = route
            vals += (bias, float(eps), sel, wts)
            types += (vp, fl, vp, vp)
        _chk((self.cu.cuLaunchKernelEx(cfg, self.fn, (vals, types), 0)[0], None))


class RouteLab:
    """Lab build (one GPU, no exchange): ``route_lab`` runs the push's routing epilogue on ``[rows, 256]`` FP32
    logits from global memory. ``tree`` picks the weight sum's shuffle partners (the shipped order is (1, 4, 2))."""

    def __init__(self, device_index: int, tree: Tuple[int, int, int] = (1, 4, 2)):
        from cuda.bindings import driver as cu
        from cuda.bindings import nvrtc

        self.cu = cu
        major, minor = torch.cuda.get_device_capability(device_index)
        prog = _chk(nvrtc.nvrtcCreateProgram(SRC.encode(), b"route_lab.cu", 0, [], []))
        opts = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"-std=c++17", b"-default-device",
                b"-DWORLD=16", b"-DROUTE=1", b"-DROUTE_LAB=1"] + \
            [f"-DTREE_S{i}={int(s)}".encode() for i, s in enumerate(tree)]
        r = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if int(r[0]) != 0:
            n = _chk(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("route_lab NVRTC compile failed:\n" + log.decode(errors="replace"))
        n = _chk(nvrtc.nvrtcGetCUBINSize(prog))
        self.cubin = b" " * n
        _chk(nvrtc.nvrtcGetCUBIN(prog, self.cubin))
        with torch.cuda.device(device_index):
            self.module = _chk(cu.cuModuleLoadData(self.cubin))
            self.fn = _chk(cu.cuModuleGetFunction(self.module, b"route_lab"))

    def __call__(self, logits: torch.Tensor, bias: torch.Tensor, eps: float,
                 out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        rows = logits.shape[0]
        assert logits.dtype == torch.float32 and logits.is_contiguous() and logits.shape[1] == EXPERTS
        if out is None:
            out = (torch.full((rows, TOP_K), -1, dtype=torch.int32, device=logits.device),
                   torch.full((rows, TOP_K), float("nan"), dtype=torch.float32, device=logits.device))
        sel, wts = out
        cu = self.cu
        cfg = cu.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = (rows + 15) // 16, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = THREADS, 1, 1
        cfg.sharedMemBytes = 0
        cfg.hStream = cu.CUstream(torch.cuda.current_stream(logits.device).cuda_stream)
        cfg.attrs = []
        cfg.numAttrs = 0
        vp, fl, ii = ctypes.c_void_p, ctypes.c_float, ctypes.c_int
        args = ((logits.data_ptr(), bias.data_ptr(), float(eps), sel.data_ptr(), wts.data_ptr(), int(rows)),
                (vp, vp, fl, vp, vp, ii))
        _chk((cu.cuLaunchKernelEx(cfg, self.fn, args, 0)[0], None))
        return sel, wts


class RouterPush:
    """This rank's router shard + multicast exchange: the gather workspace, the cluster counters, the builds."""

    TPLS = (8,)  # one build: 8 K parts per CTA, 4 x 2 logits per thread

    def __init__(self, world: int, rank: int, device: torch.device, uc_ptr: int, mc_ptr: int, keepalive=None,
                 cluster: int = CLUSTER, tpls: Tuple[int, ...] = TPLS):
        self.world, self.rank, self.device = world, rank, device
        self.gather, self.gather_mc = int(uc_ptr), int(mc_ptr)
        self.kernels = [Kernel(device.index, world, cluster, tpl) for tpl in tpls]
        self.kernel = self.kernels[0]
        self.cluster = cluster
        #: step A's build (``ROUTE``), compiled at the first routed call (eager: the first call is never captured)
        self.route_kernel: Optional[Kernel] = None
        self.rbcnt = torch.zeros(self.kernel.clusters(MAX_ROWS), dtype=torch.int32, device=device)
        self._timed: Dict[int, Kernel] = {}
        self._keepalive = keepalive

    def route_build(self) -> Kernel:
        if self.route_kernel is None:
            self.route_kernel = Kernel(self.device.index, self.world, self.cluster, self.kernel.tpl, route=True)
        return self.route_kernel

    @classmethod
    def from_mapping(cls, mapping, device: torch.device, cluster: int = CLUSTER, tpls: Tuple[int, ...] = TPLS) -> "RouterPush":
        """Collective over the TP group: the gather buffer (fabric handles), filled with the sentinel."""
        from tensorrt_llm._torch.distributed.ops import _get_mnnvl_workspace_comm, _make_mnnvl_mcast_buffer

        world, rank = mapping.tp_size, mapping.tp_rank
        size = gather_bytes()
        comm = _get_mnnvl_workspace_comm(mapping)
        handle = _make_mnnvl_mcast_buffer(comm, size, mapping, mapping.is_multi_node())
        words = size // 4
        uc = handle.get_uc_buffer(rank, (words,), torch.int32, 0)
        mc = handle.get_mc_buffer((words,), torch.int32, 0)
        uc.fill_(SENTINEL_I32)
        push = cls(world, rank, device, uc.data_ptr(), mc.data_ptr(), keepalive=(handle, uc, mc), cluster=cluster,
                   tpls=tpls)
        push.local_words = uc
        torch.cuda.synchronize(device)
        comm.Barrier()
        return push

    def build_for(self, rows: int, tpl: Optional[int] = None) -> Optional[Kernel]:
        """The widest build whose clusters for ``rows`` are all resident (the same on every rank), or None."""
        for k in self.kernels:
            if (tpl is None or k.tpl == tpl) and k.fits(rows):
                return k
        return None

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, scale: float, out: Optional[torch.Tensor] = None,
                 pdl: bool = True, timing: Optional[torch.Tensor] = None, tpl: Optional[int] = None,
                 route: Optional[Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """``route`` (step A): ``(bias [256] FP32, eps, selected [rows, 8] int32, weights [rows, 8] FP32)``; the
        kernel then also writes ``selected`` and ``weights`` as ``_full_routing_kernel`` would."""
        rows = x.shape[0]
        if out is None:
            out = torch.empty((rows, EXPERTS), dtype=torch.float32, device=x.device)
        rargs = None
        if route is not None:
            bias, eps, sel, wts = route
            k = self.route_build()
            if not k.fits(rows):
                raise RuntimeError(f"tp_router_push: the routed build does not keep {rows} rows resident")
            rargs = (bias.data_ptr(), float(eps), sel.data_ptr(), wts.data_ptr())
        else:
            k = self.build_for(rows, tpl)
            if k is None:
                raise RuntimeError(f"tp_router_push: no build keeps {rows} rows resident")
            if timing is not None:
                if k.tpl not in self._timed:
                    self._timed[k.tpl] = Kernel(self.device.index, self.world, k.cluster, k.tpl, timing=True)
                k = self._timed[k.tpl]
        k.launch(rows, x.data_ptr(), weight.data_ptr(), scale, out.data_ptr(), self.rbcnt.data_ptr(),
                 self.gather, self.gather_mc, self.rank, torch.cuda.current_stream(x.device).cuda_stream,
                 pdl=pdl, timing=0 if timing is None else timing.data_ptr(), route=rargs)
        return out


_PUSH: Dict[int, RouterPush] = {}


def route_eligible(routing, rows: int) -> bool:
    """Whether ``routing.apply`` on this call's logits (FP32, contiguous, ``[rows, 256]``) would run
    ``_full_routing_kernel`` with ``TORCH_ORDER`` off and memoize its result: the same conditions, read from the
    routing method (rank-uniform). Anything else keeps the routing method's own computation."""
    if routing is None or not getattr(routing, "_memo_enabled", False) or not hasattr(routing, "_routing_memo"):
        return False
    bias = getattr(routing, "e_score_correction_bias", None)
    return (
        rows > 0
        and rows >= ROUTE_MIN_ROWS
        and (rows <= 128 or bool(getattr(routing, "_full_routing_all_batches", False)))
        and not (bool(getattr(routing, "_full_routing_torch_order", False)) and rows > 128)
        and getattr(routing, "top_k", None) == TOP_K
        and getattr(routing, "output_dtype", None) == torch.float32
        and bias is not None
        and bias.is_cuda
        and bias.dtype == torch.float32
        and tuple(bias.shape) == (EXPERTS,)
        and bias.is_contiguous()
        and bias.data_ptr() % 16 == 0
        and isinstance(getattr(routing, "renorm_eps", None), (int, float))
    )


def _push_for(mapping, x: torch.Tensor, weight: torch.Tensor) -> Optional[RouterPush]:
    """This device's push for the call, or None (the caller's op runs). Decided on what every rank shares (row count,
    dtypes, shapes), never on an address; the transport is created at the first eager call (collective)."""
    world = getattr(mapping, "tp_size", None)
    if world not in (8, 16) or not _world_armed(world):
        return None
    min_rows, max_rows, cluster = _world_rows(world)
    if (
        x.dim() != 2
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or x.shape[1] != HIDDEN
        or not min_rows <= x.shape[0] <= max_rows
        or weight.dtype != torch.float32
        or tuple(weight.shape) != (EXPERTS, HIDDEN)
        or not weight.is_contiguous()
        or torch.compiler.is_compiling()
    ):
        return None
    push = _PUSH.get(x.device.index)
    if push is not None and push.build_for(x.shape[0]) is None:
        return None
    if push is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        from tensorrt_llm.logger import logger

        push = RouterPush.from_mapping(mapping, x.device, cluster=cluster)
        _PUSH[x.device.index] = push
        k = push.kernel
        logger.info(
            f"[musespark_open] {'tep8' if world == 8 else 'tep16'} router push: rank {mapping.tp_rank} computes experts "
            f"{mapping.tp_rank * (EXPERTS // mapping.tp_size)}..{(mapping.tp_rank + 1) * (EXPERTS // mapping.tp_size) - 1} "
            f"of {min_rows}..{max_rows}-row calls (FP32 FMA, {k.cluster} K splits per 16-row cluster, builds "
            f"{[(b.tpl, b.max_clusters) for b in push.kernels]} (threads per logit, resident clusters)) "
            f"and multicasts them ({gather_bytes() / 2**20:.2f} MiB gather per rank)"
        )
        if route_armed():
            r = push.route_build()
            logger.info(
                f"[musespark_open] router push + routing: rank {mapping.tp_rank} also writes the routing "
                f"method's top-8 selection and weights (bit for bit _full_routing_kernel, TORCH_ORDER off, up to 128 "
                f"rows; per row in every cluster CTA up to {ROUTE_ROW_MAX} rows) into the routing memo; routed build {r.regs} regs, {r.local_bytes} B local, "
                f"{r.max_clusters} resident clusters"
            )
    return push


def logits(mapping, x: torch.Tensor, weight: torch.Tensor, scale: float) -> Optional[torch.Tensor]:
    """The full ``[rows, 256]`` FP32 router logits on the shard + multicast exchange, or None (the caller's op runs)."""
    push = _push_for(mapping, x, weight)
    return None if push is None else push(x, weight, scale)


def logits_routed(mapping, x: torch.Tensor, weight: torch.Tensor, scale: float, routing):
    """Step A: ``(logits, selected, weights)``, where ``selected`` / ``weights`` are what ``routing.apply``
    would compute from these logits (``_full_routing_kernel``, bit for bit), or ``(logits, None, None)`` when the
    routing method would take another path here or the routed build is not compiled yet under capture; None when the
    push declines the call (the caller's op runs). Every decision reads rank-uniform facts only."""
    push = _push_for(mapping, x, weight)
    if push is None:
        return None
    rows = x.shape[0]
    capturing = torch.cuda.is_current_stream_capturing()
    if route_eligible(routing, rows) and not (capturing and push.route_kernel is None) and push.route_build().fits(rows):
        sel = torch.empty((rows, TOP_K), dtype=torch.int32, device=x.device)
        wts = torch.empty((rows, TOP_K), dtype=torch.float32, device=x.device)
        out = push(x, weight, scale, route=(routing.e_score_correction_bias, float(routing.renorm_eps), sel, wts))
        return out, sel, wts
    return push(x, weight, scale), None, None
