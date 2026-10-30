/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Fused MoE experts for decode-sized steps on the SM100 tensor cores: the routed experts (NVFP4
// W4A4, or BF16) and the BF16 shared expert of one MoE layer in one persistent launch.
//
// The weights are the MMA's M side (one 128-row tile per unit); the N side is every token of the
// step (rounded up to 8 for NVFP4, 16 for BF16, <= 128), so all units of a kind share one token
// tile (and its scale atoms: the CUTLASS scale layout of <= 128 rows is the MMA's atom layout) and
// a stage is a few asynchronous copies issued by one lane. A k-tile is 128 bytes per row: 256
// NVFP4 values (four kind::mxf4nvf4 block16 MMAs of K = 64) or 64 BF16 values (four kind::f16
// MMAs of K = 16), the operand descriptors advancing 32 bytes per MMA either way.
// Units, in queue order:
//   S1 (64-feature block, split-K part): the shared expert's 64 gate + 64 up rows (interleaved by
//      64) over a quarter of the hidden size; the last part to arrive sums the four fp32 partials
//      (in part order), applies SwiGLU to the bf16-rounded sums and stores the block of sh_h;
//   R1 (expert, 64-feature block): the routed expert's gate / up tile (w13 interleaved by 64,
//      kGu64); the epilogue takes the expert's routed tokens' columns: SwiGLU, then the NVFP4
//      quantization (global scale gs2) of the expert's hidden rows (grouped rows offsets[g] + i,
//      scales in the expert's own 128-row scale layout), or bf16 hidden rows;
//   S2 (128 output features): the shared expert's down rows over its intermediate size;
//   R2 (expert, 128 output features): the routed expert's down rows with its hidden rows as N.
//      RB > 1 (MoE tensor parallelism: a small per-rank intermediate, e.g. 512 of 2048, so one
//      128-row block is only two NVFP4 k-tiles): a unit takes RB consecutive 128-row blocks, one
//      accumulator each (RB x NMAX tensor-memory columns per buffer), so it streams as many weight
//      bytes as an expert-parallel R2 unit instead of RB small units with their fixed costs.
// A phase-2 k-tile's hidden rows are loaded once the phase-1 units of its features have stored
// them (a bitmask per expert / for the shared expert), the weight tiles before that.
// Warps: 0 producer (TMA / bulk copies), 1 MMA issuer (scale atoms into tensor memory by
// tcgen05.cp, tcgen05.mma), 2-5 epilogue (tensor memory -> outputs).
#include <cuda.h>
#include <cudaTypedefs.h>

#include <map>
#include <mutex>
#include <tuple>

#include "cute/arch/copy_sm90_tma.hpp"
#include "fp4_umma.cuh"
#include "moe_fp4.h"
#include "moe_route.cuh"
#include "nvfp4.h"
#include "pdl.cuh"

namespace glm {
namespace k {

namespace {
using namespace fp4mma;
using Barrier = cutlass::arch::ClusterBarrier;
using TxBarrier = cutlass::arch::ClusterTransactionBarrier;

constexpr int kThreads = 192;      // producer, MMA, 4 epilogue warps
constexpr int kTileB = 128 * 128;  // one 128-row weight tile of a k-tile (128 bytes per row)
constexpr int kRing = 16;          // unit records in flight (producer ahead of the epilogue)
constexpr int kSplit = kMoeFp4ShSplit;
constexpr int kTab = 320;          // work items / groups staged in shared memory
static_assert(kTab <= kMoeFp4HistMax, "the routing history holds every local expert of a launch");
enum : int { kS1 = 0, kR1 = 1, kS2 = 2, kR2 = 3, kEnd = -1 };
// (moe_fp4_kernel's MD) kMdDense: the dense MLP; else the MoE, with the option bits kMdRs: the in-CTA routing
// from the router's {key, score} words (MoeFp4Args::rt_ks); kMdEw: the epilogue requests a routed unit's output scale
// (alpha13 / alpha2) before its accumulator wait
enum : int { kMdMoe = 0, kMdDense = 1, kMdRs = 4, kMdEw = 8 };

struct Fp4Unit {
  int kind, gi, blk, ks, g, rb, mrows;
  int fp4, n, nkt;  // NVFP4 operands, MMA N, k-tiles
};
// 128-row blocks (accumulators) of a unit: RB for R2 units of the RB > 1 instantiation, else 1
template <int RB>
__device__ __forceinline__ int unit_blocks(const Fp4Unit& u) {
  return RB > 1 && u.kind == kR2 ? RB : 1;
}

// NMAX: the largest MMA N of the instantiation; stages fill the shared memory
template <int NMAX, int RB = 1>
struct Fp4Cfg {
  // (NMAX 8: the dense MLP's instantiation, ten k-tiles of 8-row token tiles)
  static constexpr int kStages = NMAX <= 8 ? 10 : (NMAX <= 32 ? 8 : (NMAX <= 64 ? 6 : 5));
  // tensor-memory columns: accumulators [buffer][RB blocks] (NMAX each), then per k-tile parity the
  // weight tile's 4 scale atoms (16 columns) and the token / hidden rows' 4 atoms (16 columns)
  static constexpr int kAccCols = RB * NMAX;  // one accumulator buffer
  static constexpr int kUsed = 2 * kAccCols + 64;
  static constexpr int kTmemCols =
      RB == 1 ? (NMAX <= 32 ? 128 : 512) : (kUsed <= 128 ? 128 : (kUsed <= 256 ? 256 : 512));
  static constexpr int kSfaCol = 2 * kAccCols, kSfbCol = 2 * kAccCols + 32;
  static_assert(kUsed <= 512, "tensor memory: accumulators + scale atoms exceed 512 columns");
};

// KT: k-tiles per stage (2: one 3-D TMA box of two k-tiles per operand, one 4 KB scale copy per
// operand, one barrier round per two k-tiles)
template <int NMAX, int KT = 1>
struct Fp4Smem {
  static constexpr int S = Fp4Cfg<NMAX>::kStages / KT;
  uint8_t a[S][KT * kTileB];      // weight tiles, 128-byte swizzle (1024-aligned), k-tile j at j kTileB
  uint8_t b[S][KT * NMAX * 128];  // token / hidden rows, 128-byte swizzle, k-tile j at j x box bytes
  uint8_t sfa[S][KT * 4 * 512];   // NVFP4: the weight tiles' scale atoms of the k-tiles
  uint8_t sfb[S][KT * 4 * 512];   // NVFP4: the rows' scale atoms of the k-tiles
  __nv_bfloat16 gu[128][NMAX];    // phase-1 epilogue: gate (rows 0..63) / up (64..127) outputs
  // accf / acce: accumulator buffer full (MMAs done) / empty (epilogue has read it); unitf: the
  // MMA warp hands the buffer's unit to the epilogue (release after it acquired it from the
  // producer: the tensor cores' commit is no release of the producer's unit record)
  uint64_t full[S], empty[S], accf[2], acce[2], unitf[2];
  Fp4Unit u[kRing];               // units by sequence number (written by the producer)
  // the dispatch's tables, staged once for the producer (kTab entries; larger: global memory)
  int act[kTab], off[kTab + 1];
  uint32_t tmem;
  int last;                       // S1: this part arrived last
};

// in-kernel routing (the RT instantiations, after Fp4Smem): per local expert the bitmask of its
// tokens and its work item, per pair its expert (-1 another rank's, -2 the shared expert), the
// grouped rows' tokens, the work item count
struct RtSmem {
  unsigned long long mask[kTab];
  int item[kTab];  // a local expert's work item
  int g[kMoeFp4RtMaxT * kMoeFp4RtMaxK1];
  // per grouped row: its token (bits 0-15) and its output row (bits 16-31: the grouped row itself, or with y_tm the
  // pair's token-major row t K1 + slot)
  int rsrc[kMoeFp4RtMaxT * kMoeFp4RtMaxK1];
  int items;
  int early;  // (rt_local) token 0's experts are the work items 0 .. top_k - 1 (RT_T0)
};

// this launch's counter buffer (after the grid dependency wait: the previous launch handed it
// over); the last CTA (its MMA warp, idle until its first unit) zeroes the previous launch's
// buffer for the next launch
__device__ __forceinline__ int* moe_fp4_ctr(int* hdr, int& par, int warp, int lane) {
  par = __ldcg(hdr) & 1;
  if (warp == 1 && blockIdx.x == gridDim.x - 1) {
    int* const o = hdr + kMoeFp4CtrHdr + (par ^ 1) * kMoeFp4CtrSize;
    const int nu = __ldcg(o + kMoeFp4CtrNused);
    for (int i = lane; i < kMoeFp4CtrItems + nu; i += 32) o[i] = 0;
    if (lane == 0) {
      o[kMoeFp4CtrRt] = 0;
      o[kMoeFp4CtrNused] = 0;
    }
    __syncwarp();
  }
  return hdr + kMoeFp4CtrHdr + par * kMoeFp4CtrSize;
}

// the kernel's end: the last CTA out restores the queue and the count and hands the other buffer
// (zeroed by this launch) to the next launch, recording the work items this one used. Every
// claim and count before it has returned: plain stores, ordered before the next launch's reads by
// the kernel's completion.
__device__ __forceinline__ void moe_fp4_ctr_end(int* hdr, int* ctr, int par, int n_items) {
  if (atomicAdd(hdr + 2, 1) == (int)gridDim.x - 1) {
    ctr[kMoeFp4CtrNused] = n_items;
    hdr[kMoeFp4CtrTag] = hdr[kMoeFp4CtrTag] + 1;  // (the launch epoch of the tagged routed ids)
    hdr[1] = 0;
    hdr[2] = 0;
    hdr[0] = par ^ 1;
  }
}

__device__ __forceinline__ unsigned ld_acquire_u(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ float rcp_approx_f(float a) {
  float b;
  asm("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

// as swiglu_quant_nvfp4 / swiglu_bf16: clamp, SiLU, product rounded to bf16 (inputs bf16-rounded)
__device__ __forceinline__ float swiglu_f(float g, float u, float limit) {
  g = fminf(g, limit);
  u = fminf(fmaxf(u, -limit), limit);
  return __bfloat162float(__float2bfloat16(g / (1.f + __expf(-g)) * u));
}

// swiglu_f with the approximate division (nvfp4.cu's swiglu1_fast, as the dense MLP's separate path)
__device__ __forceinline__ float swiglu_fast_f(float g, float u, float limit) {
  g = fminf(g, limit);
  u = fminf(fmaxf(u, -limit), limit);
  return __bfloat162float(__float2bfloat16(__fdividef(g, 1.f + __expf(-g)) * u));
}

__device__ __forceinline__ uint32_t e2m1_nibble(float v) {
  uint16_t r;
  asm("{\n .reg .b8 b;\n cvt.rn.satfinite.e2m1x2.f32 b, %1, %1;\n cvt.u16.u8 %0, b;\n}\n"
      : "=h"(r)
      : "f"(v));
  return r & 15u;
}

// SHPF (MoeFp4Args::sh_pf): the 32 KB bulk prefetches c0, c0 + step, ... of the shared expert's gate / up weights,
// then (sh_pf 2) its down weights, into L2 (one SM's TMA unit drains them at ~0.25 TB/s)
__device__ __forceinline__ void shpf_issue(const MoeFp4Args& p, int c0, int step) {
  constexpr size_t kChunk = 32 * 1024;
  const size_t gu = (size_t)2 * p.sh_inter * p.H * sizeof(bf16);
  const size_t dn = p.sh_pf > 1 ? (size_t)p.H * p.sh_inter * sizeof(bf16) : 0;
  const size_t ngu = (gu + kChunk - 1) / kChunk, n = ngu + (dn + kChunk - 1) / kChunk;
  for (size_t c = c0; c < n; c += step) {
    const bool g = c < ngu;
    const size_t off = (g ? c : c - ngu) * kChunk, len = (g ? gu : dn) - off;
    const uint8_t* src = reinterpret_cast<const uint8_t*>(g ? p.sh_gu : p.sh_dn) + off;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(src), "r"((uint32_t)(len < kChunk ? len : kChunk))
                 : "memory");
  }
}

__device__ __forceinline__ void epi_sync() {  // the 4 epilogue warps
  asm volatile("bar.sync 1, 128;\n" ::: "memory");
}

// runtime-N instruction descriptors (n_dim = N / 8)
__device__ __forceinline__ uint64_t idesc_fp4n(uint32_t tsfa, uint32_t tsfb, int N) {
  cute::UMMA::InstrDescriptorBlockScaled d = cute::UMMA::make_instr_desc_block_scaled<
      cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, cutlass::float_ue4m3_t, 128, 8,
      cute::UMMA::Major::K, cute::UMMA::Major::K>();
  d.n_dim_ = N >> 3;
  return cute::UMMA::make_runtime_instr_desc_block_scaled(d, tsfa, tsfb);
}
__device__ __forceinline__ uint32_t idesc_bf16n(int N) {
  cute::UMMA::InstrDescriptor d =
      cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float, 128, 16,
                                  cute::UMMA::Major::K, cute::UMMA::Major::K>();
  d.n_dim_ = N >> 3;
  return static_cast<uint32_t>(d);
}

__device__ __forceinline__ void mma_fp4(uint64_t da, uint64_t db, uint32_t td, bool accumulate,
                                        uint32_t tsfa, uint32_t tsfb, uint64_t idesc) {
  if (cute::elect_one_sync())
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], "
        "p;\n}\n" ::"r"(td),
        "l"(da), "l"(db), "r"(uint32_t(idesc >> 32)), "r"(accumulate ? 1u : 0u), "r"(tsfa),
        "r"(tsfb));
}
__device__ __forceinline__ void mma_bf16(uint64_t da, uint64_t db, uint32_t td, bool accumulate,
                                         uint32_t idesc) {
  if (cute::elect_one_sync())
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n" ::"r"(td),
        "l"(da), "l"(db), "r"(idesc), "r"(accumulate ? 1u : 0u));
}

// Lean issue (the MMA warp's elected thread only): the per-MMA elect / runtime descriptor building around a
// k-tile's 4 MMAs + 8 scale copies would take the issuing warp longer than the tensor pipe takes to run them, so
// the MMA warp's loop issues them from precomputed descriptors in one elected branch
__device__ __forceinline__ void mma_fp4_raw(uint64_t da, uint64_t db, uint32_t td, uint32_t acc,
                                            uint32_t idh, uint32_t tsfa, uint32_t tsfb) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], "
      "p;\n}\n" ::"r"(td),
      "l"(da), "l"(db), "r"(idh), "r"(acc), "r"(tsfa), "r"(tsfb));
}
__device__ __forceinline__ void mma_bf16_raw(uint64_t da, uint64_t db, uint32_t td, uint32_t acc,
                                             uint32_t idesc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n" ::"r"(td),
      "l"(da), "l"(db), "r"(idesc), "r"(acc));
}
__device__ __forceinline__ void sf_copy_raw(uint32_t tmem_col, uint64_t desc) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n" ::"r"(tmem_col), "l"(desc));
}

// The producer's lane control (CONV: the whole warp runs the producer loop, one elected lane issues;
// else lane 0 alone runs it)
template <bool CONV>
__device__ __forceinline__ bool prod_elect() {
  if constexpr (CONV) return cute::elect_one_sync();
  else return true;
}
template <bool CONV>
__device__ __forceinline__ void prod_sync() {
  if constexpr (CONV) __syncwarp();
}
template <bool CONV>
__device__ __forceinline__ int prod_bcast(int v) {  // (CONV: lane 0's value, warp-uniform)
  if constexpr (CONV) return __shfl_sync(0xffffffffu, v, 0);
  else return v;
}

// the operand maps: routed w13 / w2 / tokens / hidden rows, shared gate-up / down / tokens / hidden
struct Fp4Maps {
  CUtensorMap w13, w2, x, h, s13, sdn, sx, sh;
};

// Dense MLP epilogues (kMdDense), not inlined: the epilogue warps run them once before the grid
// dependency wait on scratch (dry: no global stores), which brings their code into the SM's
// instruction cache, so the real pass after the MMAs runs warm. The arithmetic is the generic
// epilogue's (bitwise).
// gate / up: this lane's accumulator row (tile row 32 q + lane) of the first ncol columns, bf16 with
// the output scale, into its row of the gate / up tile
__device__ __noinline__ void dn_epi_gu(uint32_t tbase, int ncol, float al, __nv_bfloat16* gu_row) {
  for (int c0 = 0; c0 < ncol; c0 += 8) {
    float v[8];
    tmem_ld8(tbase + c0, v);
    tmem_wait_ld();
#pragma unroll
    for (int i = 0; i < 8; ++i) gu_row[c0 + i] = __float2bfloat16(v[i] * al);
  }
}
// SwiGLU (approximate division) and NVFP4 of hidden rows 0 .. nrow - 1 of 64-feature block blk from
// the gate / up tile gu [128][ld]: per warp pass (row, 32-feature half), lanes = features
__device__ __noinline__ void dn_epi_sw(const __nv_bfloat16* gu, int ld, int nrow, int blk, int q, int lane,
                                       float limit, float sfs, uint8_t* hq, uint8_t* hsg, int Il, bool dry) {
  for (int w = q; w < 2 * nrow; w += 4) {
    const int i = w >> 1, fh = w & 1;
    const int fl = 32 * fh + lane, f = blk * 64 + fl;
    const float h = swiglu_fast_f(__bfloat162float(gu[fl * ld + i]), __bfloat162float(gu[(64 + fl) * ld + i]), limit);
    float amax = fabsf(h);
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 4));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 8));
    float sfv = sfs * (amax * rcp_approx_f(6.f));
    const __nv_fp8_e4m3 s8(sfv);
    sfv = static_cast<float>(s8);
    const float osc = amax != 0.f ? rcp_approx_f(sfv * rcp_approx_f(sfs)) : 0.f;
    const uint32_t nib = e2m1_nibble(h * osc);
    const uint32_t byte = nib | (__shfl_down_sync(0xffffffff, nib, 1) << 4);
    uint32_t wd = byte | (__shfl_down_sync(0xffffffff, byte, 2) << 8);
    wd |= __shfl_down_sync(0xffffffff, wd, 4) << 16;
    if (!dry) {
      if ((lane & 7) == 0) *reinterpret_cast<uint32_t*>(hq + (size_t)i * (Il / 2) + (f >> 1)) = wd;
      if ((lane & 15) == 0) hsg[nvfp4_sf_off(i, f >> 4, Il)] = s8.__x;
    }
  }
}
// down: rows 0 .. rows - 1 (columns) of output feature n (this lane's tile row), bf16 into y [rows, H]
__device__ __noinline__ void dn_epi_dn(uint32_t tbase, int rows, int n, float al, bf16* y, int H, bool dry) {
  for (int c0 = 0; c0 < rows; c0 += 8) {
    float v[8];
    tmem_ld8(tbase + c0, v);
    tmem_wait_ld();
#pragma unroll
    for (int i = 0; i < 8; ++i)
      if (c0 + i < rows && !dry) y[(size_t)(c0 + i) * H + n] = __float2bfloat16(v[i] * al);
  }
}

// The shared expert's block sums by its down units' producers (the warp producer): while a down unit
// waits for hidden-row blocks, a block whose kSplit gate / up partials are all published and not yet
// taken is taken (its part count incremented past kSplit) and summed by this warp: the partials in
// part order, bf16, SwiGLU, its sh_T rows of sh_h, released with its flag -- the sums, roundings and
// stores of the gate / up epilogue's last part otherwise (bitwise), off the gate / up CTAs' epilogues,
// whose next gate / up units' flags they would delay (and on a warp that only waits). Returns the flags
// once every block in `need` is set.
__device__ __forceinline__ unsigned sh_sum_wait(unsigned* done_sh, int* parts, unsigned need, int nblk,
                                             const float* sh_part, int ldp, int sh_T, bf16* sh_h, int Is,
                                             float limit, int lane) {
  for (;;) {
    unsigned h = ld_acquire_u(done_sh);
    h = __shfl_sync(0xffffffffu, h, 0);
    if ((h & need) == need) return h;
    // the blocks with every part published and no summer yet (one per lane)
    const int c = lane < nblk && !((h >> lane) & 1u)
                      ? (int)ld_acquire_u(reinterpret_cast<const unsigned*>(parts + lane)) : 0;
    unsigned ready = __ballot_sync(0xffffffffu, c == kSplit);
    int b = -1;
    while (ready) {
      const int cand = __ffs(ready) - 1;
      int won = 0;
      if (lane == 0) won = atomicAdd(parts + cand, 1) == kSplit;
      if (__shfl_sync(0xffffffffu, won, 0)) {
        b = cand;
        break;
      }
      ready &= ready - 1;
    }
    if (b < 0) {
      __nanosleep(32);
      continue;
    }
    // block b: lane = feature f (and f + 32) of the block: gate row f, up row 64 + f of the parts
    const float* const s0 = sh_part + (size_t)b * kSplit * 128 * ldp;
    for (int c0 = 0; c0 < sh_T; c0 += 4) {
      float4 a[4];  // [gate f, up f, gate f + 32, up f + 32]
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int row = (r & 1) * 64 + (r >> 1) * 32 + lane;
        const float* const q = s0 + (size_t)row * ldp + c0;
        a[r] = __ldcg(reinterpret_cast<const float4*>(q));
#pragma unroll
        for (int k2 = 1; k2 < kSplit; ++k2) {
          const float4 v = __ldcg(reinterpret_cast<const float4*>(q + (size_t)k2 * 128 * ldp));
          a[r].x += v.x;
          a[r].y += v.y;
          a[r].z += v.z;
          a[r].w += v.w;
        }
      }
      // (the sums bf16-rounded as the epilogue's gate / up tile holds them)
      __nv_bfloat16 gb[8], ub[8];  // [fh * 4 + column]
#pragma unroll
      for (int fh = 0; fh < 2; ++fh) {
        gb[4 * fh] = __float2bfloat16(a[2 * fh].x), ub[4 * fh] = __float2bfloat16(a[2 * fh + 1].x);
        gb[4 * fh + 1] = __float2bfloat16(a[2 * fh].y), ub[4 * fh + 1] = __float2bfloat16(a[2 * fh + 1].y);
        gb[4 * fh + 2] = __float2bfloat16(a[2 * fh].z), ub[4 * fh + 2] = __float2bfloat16(a[2 * fh + 1].z);
        gb[4 * fh + 3] = __float2bfloat16(a[2 * fh].w), ub[4 * fh + 3] = __float2bfloat16(a[2 * fh + 1].w);
      }
#pragma unroll 1
      for (int k = 0; k < 8; ++k) {
        const int fh = k >> 2, i = k & 3;
        const float hv = swiglu_f(__bfloat162float(gb[k]), __bfloat162float(ub[k]), limit);
        if (c0 + i < sh_T) sh_h[(size_t)(c0 + i) * Is + b * 64 + 32 * fh + lane] = __float2bfloat16(hv);
      }
    }
    // (the next units' async-proxy loads of these rows, here and in the other down units' CTAs)
    asm volatile("fence.proxy.async.global;\n" ::: "memory");
    __syncwarp();
    if (lane == 0) {
      asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
      atomicOr(done_sh, 1u << b);
    }
    __syncwarp();
  }
}

}  // namespace

// RT: the routing and the tables in the kernel (p.rt_*), else moe_dispatch_small's tables.
// MD: kMdDense (a dense NVFP4 MLP as one expert: one static unit per CTA, its weights issued before
// the grid dependency wait; moe_fp4_dense), else the MoE.
template <int NMAX, bool CONV, int KT, int RB = 1, bool RT = false, bool RTX = false, int MD = 0>
__global__ void __launch_bounds__(kThreads, 1)
    moe_fp4_kernel(const MoeFp4Args p, int NP, const __grid_constant__ Fp4Maps tm) {
  if (!p.late_trigger) pdl_trigger();
  // the shared expert's weights into L2 while the dispatch still runs (static: no dependency on it), dealt over
  // every CTA by thread 64 at the start (sh_pf_ctas 0), else see below
  if (p.sh_pf && p.sh_T > 0 && p.sh_inter > 0 && p.sh_pf_ctas <= 0 && threadIdx.x == 64)
    shpf_issue(p, blockIdx.x, gridDim.x);
  using Cfg = Fp4Cfg<NMAX, RB>;
  constexpr int S = Fp4Smem<NMAX, KT>::S;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t pad = (1024u - (smem_u32(smem_raw) & 1023u)) & 1023u;
  Fp4Smem<NMAX, KT>& sm = *reinterpret_cast<Fp4Smem<NMAX, KT>*>(smem_raw + pad);
  RtSmem& srt = *reinterpret_cast<RtSmem*>(smem_raw + pad + sizeof(Fp4Smem<NMAX, KT>));  // (RT only)
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int s = 0; s < S; ++s) {
      Barrier::init(&sm.full[s], 1);
      Barrier::init(&sm.empty[s], 1);
    }
    for (int i = 0; i < 2; ++i) {
      Barrier::init(&sm.accf[i], 1);
      Barrier::init(&sm.acce[i], 4);
      Barrier::init(&sm.unitf[i], 1);
    }
    cutlass::arch::fence_barrier_init();
  }
  cute::TMEM::Allocator1Sm tmem_alloc{};
  if (warp == 1) tmem_alloc.allocate(Cfg::kTmemCols, &sm.tmem);
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const uint32_t tmem = sm.tmem;
  // (sh_pf_ctas > 0) the last sh_pf_ctas CTAs deal them, by the producer's lane 0 after the setup (it only waits then;
  // thread 64 at the start would delay its CTA's setup barrier and routing): no S1 CTA (the first ones) among them
  if (p.sh_pf && p.sh_T > 0 && p.sh_inter > 0 && p.sh_pf_ctas > 0 && tid == 0) {
    const int nc = min(p.sh_pf_ctas, (int)gridDim.x), j = (int)blockIdx.x - ((int)gridDim.x - nc);
    if (j >= 0) shpf_issue(p, j, nc);
  }
  // S1_PRE (kMoeFp4OptS1Pre): the shared expert's gate / up units are the first units of CTAs
  // 0 .. e1s - 1 (static, not claimed; the queue's claims start after them). Their weights never
  // depend on earlier kernels: each of those CTAs records its unit and issues its first S weight
  // tiles while the grid dependency is pending; the token tiles follow after the wait.
  const bool s1pre = (p.opts & kMoeFp4OptS1Pre) && p.sh_T > 0 && p.sh_inter > 0;
  const int e1s = s1pre ? min((p.sh_inter / 64) * kSplit, (int)gridDim.x) : 0;
  int pre = 0;  // stages of the CTA's first unit whose weights were issued before the wait
  if (warp == 0 && (int)blockIdx.x < e1s) {
    Fp4Unit cu{};
    cu.kind = kS1;
    cu.blk = blockIdx.x / kSplit;
    cu.ks = blockIdx.x % kSplit;
    cu.nkt = p.H / 64 / kSplit;
    cu.n = (p.sh_T + 15) & ~15;
    pre = min(S, cu.nkt / KT);
    if (cute::elect_one_sync()) {
      sm.u[0] = cu;
      constexpr uint64_t kHint = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_FIRST);
      for (int st = 0; st < pre; ++st) {
        TxBarrier::arrive_and_expect_tx(&sm.full[st], KT * (kTileB + cu.n * 128));
        if constexpr (KT == 2)
          cute::SM90_TMA_LOAD_3D::copy(&tm.s13, &sm.full[st], kHint, sm.a[st], 0, cu.blk * 128,
                                       cu.ks * cu.nkt + KT * st);
        else
          cute::SM90_TMA_LOAD_2D::copy(&tm.s13, &sm.full[st], kHint, sm.a[st],
                                       (cu.ks * cu.nkt + st) * 128, cu.blk * 128);
      }
    }
    __syncwarp();
  }
  if constexpr (MD == kMdDense) if (warp == 0) {
    // dense MLP (one expert, p.rt_T rows): CTA u's static unit is gate / up block u (u < Il / 64) or
    // down block u - Il / 64; its record and its first S stages' weight tiles and scale atoms before
    // the grid dependency wait (static weights), the rows' tiles after it
    const int nbd = p.Il / 64, u = blockIdx.x;
    const bool g1 = u < nbd;
    Fp4Unit cu{};
    cu.kind = g1 ? kR1 : kR2;
    cu.blk = g1 ? u : u - nbd;
    cu.mrows = p.rt_T;
    cu.fp4 = 1;
    cu.nkt = g1 ? p.H >> 8 : p.Il >> 8;
    cu.n = NP;
    pre = min(S, cu.nkt / KT);
    if (cute::elect_one_sync()) {
      sm.u[0] = cu;
      constexpr uint64_t kHint = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_FIRST);
      const CUtensorMap* wm = g1 ? &tm.w13 : &tm.w2;
      const uint8_t* sfw = g1 ? p.sf13 + (size_t)cu.blk * (p.H / 64) * 512
                              : p.sf2 + (size_t)cu.blk * (p.Il / 64) * 512;
      for (int st = 0; st < pre; ++st) {
        TxBarrier::arrive_and_expect_tx(&sm.full[st], KT * (kTileB + NP * 128 + 2 * 2048));
        if constexpr (KT == 2)
          cute::SM90_TMA_LOAD_3D::copy(wm, &sm.full[st], kHint, sm.a[st], 0, cu.blk * 128, KT * st);
        else
          cute::SM90_TMA_LOAD_2D::copy(wm, &sm.full[st], kHint, sm.a[st], st * 128, cu.blk * 128);
        cute::SM90_BULK_COPY_G2S::copy(sfw + (size_t)st * KT * 2048, &sm.full[st], sm.sfa[st], KT * 2048);
      }
    }
    __syncwarp();
  }
  // in-kernel routing (p.rt_logits): the epilogue warps route the tokens and build the tables after
  // the wait (their expert biases and zeroed token masks before it) while the producer and MMA warps
  // start (a CTA's static S1 unit); the producer takes the tables before its first other unit
  float rt_bl[RT ? kMoeFp4RtMaxE / 32 : 1];
  if constexpr (RT) if (warp >= 2) {
    if constexpr (!(MD & kMdRs)) {  // (kMdRs: the router added them)
#pragma unroll
      for (int i = 0; i < kMoeFp4RtMaxE / 32; ++i) {
        const int e = lane + 32 * i;
        rt_bl[i] = e < p.rt_E ? __ldg(p.rt_bias + e) : 0.f;
      }
    }
    for (int g = tid - 64; g < p.G; g += 128) srt.mask[g] = 0ull;
  }
  constexpr bool rt_local = RT && !RTX;  // (in-kernel routing within each CTA)
  // the shared expert's block sums by its down units' producers (SH_SUM): the warp producer, the tokens routed within
  // each CTA (the other instantiations keep the last part's sum)
  constexpr bool kS2S = CONV && rt_local;
  if constexpr (MD == kMdDense) if (warp >= 2) {
    // dense: the epilogue's code warmed on scratch before the wait (the accumulator columns hold no
    // data yet; nothing is stored to global memory)
    const int q = warp & 3, row = 32 * q + lane;
    const uint32_t tb = tmem + ((uint32_t)(32 * q) << 16);
    if ((int)blockIdx.x < p.Il / 64) {
      dn_epi_gu(tb, NP, 1.f, &sm.gu[row][0]);
      epi_sync();
      dn_epi_sw(&sm.gu[0][0], NMAX, p.rt_T, 0, q, lane, p.limit, 1.f, nullptr, nullptr, p.Il, true);
      epi_sync();
    } else {
      dn_epi_dn(tb, p.rt_T, row, 1.f, nullptr, p.H, true);
    }
  }
  if constexpr (rt_local) {
    // in-kernel routing within each CTA: epilogue warp w routes tokens w, w + 4, ... (moe_route_warp's ids /
    // weights, bitwise; CTA 0 writes them) after its own grid dependency wait. The routing runs once per
    // launch and its instruction fetch is most of its latency: with RT_WARM the MMA warp (idle until its
    // first unit's data) runs the same code first, dry (one token of zero logits, nothing written), while
    // the dependency is pending, so the epilogue warps' pass finds it in the instruction cache
    const bool warm = (p.opts & kMoeFp4OptRtWarm) != 0;
    if constexpr ((MD & kMdRs) != 0) {
      if ((warp >= 2 && warp < 6) || (warp == 1 && warm)) {
        const bool dry = warp == 1;
        if (!dry) pdl_wait();
        // kMdRs: the router's {key, score} words (moe_router_mma_ks: the scores' exp / reciprocal and the biases'
        // keys made there by 16 threads per CTA, off this critical path); the selection and weights as below
        // (route_token_warp_ks: bitwise route_token_warp_lv's). The dry pass: distinct keys, nothing written
        const int K1 = p.rt_topk + 1, TK = p.rt_topk, T = dry ? 1 : p.rt_T, E = p.rt_E, G = p.G, lo = p.rt_lo;
        const int ew = dry ? 0 : warp - 2;
        const bool out = blockIdx.x == 0 && !dry;
        const bool ei = (p.opts & kMoeFp4OptRtEi) != 0;
        constexpr int MP = kMoeFp4RtMaxE / 32;
        const unsigned long long* const ksw = reinterpret_cast<const unsigned long long*>(p.rt_logits);
        uint32_t kv[MP], kvn[MP];  // the token's keys / scores; the warp's next token's, loaded meanwhile
        float sv[MP], svn[MP];
#pragma unroll
        for (int i = 0; i < MP; ++i) {
          const int e = lane + 32 * i;
          const unsigned long long w = ew < T && e < E && !dry ? __ldcg(ksw + (size_t)ew * E + e)
                                                               : (dry ? (unsigned long long)(e + 1) : 0ull);
          kv[i] = (uint32_t)w;
          sv[i] = __uint_as_float((uint32_t)(w >> 32));
        }
        for (int t = ew; t < T; t += 4) {
          const bool nx = t + 4 < T;
#pragma unroll
          for (int i = 0; i < MP; ++i) {
            const int e = lane + 32 * i;
            const unsigned long long w = nx && e < E ? __ldcg(ksw + (size_t)(t + 4) * E + e) : 0ull;
            kvn[i] = (uint32_t)w;
            svn[i] = __uint_as_float((uint32_t)(w >> 32));
          }
          const int id = route_token_warp_ks<MP, true>(kv, sv, E, TK, p.rt_scale, E,
                                                       out ? p.rt_ids + (size_t)t * K1 : nullptr,
                                                       out ? p.rt_w + (size_t)t * K1 : nullptr, lane, ei);
          const int g = id >= lo && id < lo + G ? id - lo : (id >= E ? -2 : -1);
          if (lane < K1 && !dry) {
            srt.g[t * K1 + lane] = g;
            if (g >= 0) atomicOr(&srt.mask[g], 1ull << t);
          }
          if (t == 0 && !dry) {  // (token 0's experts for the producer, bar 4: RT_T0)
            const bool loc = __all_sync(0xffffffffu, lane >= TK || g >= 0);
            if (lane == 0) srt.early = loc && (p.opts & kMoeFp4OptRtT0) != 0;
            asm volatile("bar.arrive 4, 64;\n" ::: "memory");
          }
#pragma unroll
          for (int i = 0; i < MP; ++i) kv[i] = kvn[i], sv[i] = svn[i];
        }
        // early items (kMoeFp4OptRtEi): the work items (the experts in the order of their first pair) built here,
        // right after the routing, by epilogue warp 0 and handed to the producer (bar 2), instead of in the epilogue
        // branch after the common setup and the counter header's load; the dry pass runs this code too (no pairs,
        // nothing written), so that it is warm. T x TK <= 128 here (<= 8 tokens of top-8).
        if (ei) {
          if (!dry) epi_sync();
          if (ew == 0) {
            int nb = 0;  // (items of the earlier rounds)
#pragma unroll 1
            for (int p0 = 0; p0 < T * TK; p0 += 32) {
              const int pi = p0 + lane;
              bool first = false;
              int g = -1;
              if (pi < T * TK && !dry) {
                const int t = pi / TK;
                g = srt.g[t * K1 + pi % TK];
                first = g >= 0 && (srt.mask[g] & ((1ull << t) - 1ull)) == 0ull;
              }
              const unsigned um = __ballot_sync(0xffffffffu, first);
              if (first) {
                const int r = nb + __popc(um & ((1u << lane) - 1u));
                sm.act[r] = g;
                srt.item[g] = r;
              }
              nb += __popc(um);
            }
            if (lane == 0 && !dry) srt.items = nb;
          }
          if (!dry) asm volatile("bar.arrive 2, 160;\n" ::: "memory");  // the producer's items
        }
      }
    } else if ((warp >= 2 && warp < 6) || (warp == 1 && warm)) {
      const bool dry = warp == 1;
      if (dry) {
#pragma unroll
        for (int i = 0; i < kMoeFp4RtMaxE / 32; ++i) rt_bl[i] = 0.f;
      } else {
        pdl_wait();
      }
      const int K1 = p.rt_topk + 1, TK = p.rt_topk, T = dry ? 1 : p.rt_T, E = p.rt_E, G = p.G, lo = p.rt_lo;
      const int ew = dry ? 0 : warp - 2;
      const bool out = blockIdx.x == 0 && !dry;
      constexpr int MP = kMoeFp4RtMaxE / 32;
      float lv[MP], lvn[MP];  // the token's logits; the warp's next token's, loaded meanwhile
#pragma unroll
      for (int i = 0; i < MP; ++i) {
        const int e = lane + 32 * i;
        lv[i] = ew < T && e < E && !dry ? __ldcg(p.rt_logits + (size_t)ew * E + e) : 0.f;
      }
      for (int t = ew; t < T; t += 4) {
        const bool nx = t + 4 < T;
#pragma unroll
        for (int i = 0; i < MP; ++i) {
          const int e = lane + 32 * i;
          lvn[i] = nx && e < E ? __ldcg(p.rt_logits + (size_t)(t + 4) * E + e) : 0.f;
        }
        const int id = route_token_warp_lv<MP, true, true>(lv, rt_bl, E, TK, p.rt_scale, E,
                                                     out ? p.rt_ids + (size_t)t * K1 : nullptr,
                                                     out ? p.rt_w + (size_t)t * K1 : nullptr, lane);
        const int g = id >= lo && id < lo + G ? id - lo : (id >= E ? -2 : -1);
        if (lane < K1 && !dry) {
          srt.g[t * K1 + lane] = g;
          if (g >= 0) atomicOr(&srt.mask[g], 1ull << t);
        }
        if (t == 0 && !dry) {  // (token 0's experts for the producer, bar 4: RT_T0)
          const bool loc = __all_sync(0xffffffffu, lane >= TK || g >= 0);
          if (lane == 0) srt.early = loc && (p.opts & kMoeFp4OptRtT0) != 0;
          asm volatile("bar.arrive 4, 64;\n" ::: "memory");
        }
#pragma unroll
        for (int i = 0; i < MP; ++i) lv[i] = lvn[i];
      }
    }
  }
  pdl_wait();  // (the warps that waited above: a no-op)
  if (p.late_trigger) pdl_trigger();
  int cpar;
  // (dense: the producer reads the counter header after its first loads, see below)
  int* const ctr = MD == kMdDense && warp == 0 ? nullptr : moe_fp4_ctr(p.ctr, cpar, warp, lane);
  // tab_early: the producer's first claim, the dispatch's tables and the item count requested at once
  // (one work item per expert, n_items <= G: the tables' G + 1 entries are read without waiting for
  // the count), three independent round trips instead of three dependent ones before the first TMA
  // (in-kernel routing: the first claim only)
  const bool tab0 = !RT && MD != kMdDense && p.tab_early && p.G <= kTab;
  int u0 = 0;
  int va[kTab / 32], vo[kTab / 32 + 1];  // (all loads issued before the first use)
  if constexpr (RT)
    if (warp == 0 && lane == 0 && (int)blockIdx.x >= e1s) u0 = e1s + atomicAdd(p.ctr + 1, 1);
  if (tab0 && warp == 0) {
    if (lane == 0 && (int)blockIdx.x >= e1s) u0 = e1s + atomicAdd(p.ctr + 1, 1);
#pragma unroll
    for (int j = 0; j < kTab / 32; ++j) va[j] = lane + 32 * j < p.G ? p.active[lane + 32 * j] : 0;
#pragma unroll
    for (int j = 0; j <= kTab / 32; ++j) vo[j] = lane + 32 * j <= p.G ? p.offsets[lane + 32 * j] : 0;
  }
  // (in-kernel routing: the producer's, from the tables; dense: one item, its rows p.rt_T)
  int n_items = RT ? 0 : (MD == kMdDense ? 1 : *p.num_active);
  if (tab0 && warp == 0) {
#pragma unroll
    for (int j = 0; j < kTab / 32; ++j)
      if (lane + 32 * j < p.G) sm.act[lane + 32 * j] = va[j];
#pragma unroll
    for (int j = 0; j <= kTab / 32; ++j)
      if (lane + 32 * j <= p.G) sm.off[lane + 32 * j] = vo[j];
  }
  const int H = p.H, Il = p.Il;
  const int Is = p.sh_T > 0 ? p.sh_inter : 0;
  const bool rbf = p.routed_bf16;
  const int nb1 = Il / 64, nb2 = H / 128 / RB;  // routed units per expert
  // the routed units' (expert, block) by shifts when both counts are powers of two (the decode sits
  // between two units' loads on the producer's critical path)
  const bool pow2 = (nb1 & (nb1 - 1)) == 0 && (nb2 & (nb2 - 1)) == 0;
  const int sh1 = 31 - __clz(nb1), sh2 = 31 - __clz(nb2);
  // k-tiles of the routed units (128 bytes per row: 256 NVFP4 or 64 BF16 values) and the experts'
  // weight-scale strides, outside the per-unit decode
  // (RB > 1: a down unit's RB blocks, Il / 256 k-tiles each)
  const int nkt1 = rbf ? H >> 6 : H >> 8, nkt2 = (rbf ? Il >> 6 : Il >> 8) * RB;
  const size_t sf13_g = nvfp4_sf_bytes(2 * Il, H), sf2_g = nvfp4_sf_bytes(H, Il);
  const int ns1 = Is / 64, ns2 = Is ? H / 128 : 0;
  const int e1 = ns1 * kSplit;
  int e2 = e1 + n_items * nb1, e3 = e2 + ns2, n_all = e3 + n_items * nb2;
  // MMA N of the routed / shared units (BF16 MMAs: multiples of 16)
  const int Nr = rbf ? (NP + 15) & ~15 : NP;
  const int Ns = (p.sh_T + 15) & ~15;
  int* const queue = p.ctr + 1;  // (the header's)
  unsigned* const done_sh = reinterpret_cast<unsigned*>(ctr);
  int* const parts = ctr + 1;
  unsigned* const done1 = reinterpret_cast<unsigned*>(ctr + kMoeFp4CtrItems);
  const size_t hs_group = nvfp4_sf_bytes(128, Il);  // an expert's hidden-row scale region
  // the producer reads every unit's group and rows: staged in shared memory once (one round trip
  // per CTA instead of two dependent ones per unit on the producer's critical path)
  const bool tab = RT || (tab0 ? n_items <= p.G : n_items <= kTab && p.G <= kTab);
  if (tab && warp == 0) {
    if constexpr (MD == kMdDense) {
      if (lane == 0) {
        sm.act[0] = 0;
        sm.off[0] = 0;
        sm.off[1] = p.rt_T;
      }
    } else if (!RT && !tab0) {
      for (int i = lane; i < n_items; i += 32) sm.act[i] = p.active[i];
      for (int i = lane; i <= p.G; i += 32) sm.off[i] = p.offsets[i];
    }
    __syncwarp();
    if (!RT && p.rt_hist && blockIdx.x == 0) {  // (the dispatch's experts and their rows, as the
      const int nu = min(n_items, kMoeFp4HistMax);  //  in-kernel routing records them: rt_hist)
      for (int i = lane; i < nu; i += 32) {
        const int g = sm.act[i];
        p.rt_hist[1 + i] = g;
        p.rt_hist[1 + kMoeFp4HistMax + i] = sm.off[g + 1] - sm.off[g];
      }
      if (lane == 0) p.rt_hist[0] = nu;
    }
  }

  if (warp == 0) {
    // ---------------------------------------------------------------- producer
    // CONV: the whole warp runs the loop, as CUTLASS's producers do, every unit's fields broadcast from
    // lane 0 (warp-uniform): the copies' operands then need no per-copy register-to-uniform loop, which
    // lane 0 alone runs around each of the four copies of every k-tile. Else lane 0 alone runs the loop.
    if constexpr (MD == kMdDense) {
      // dense: the CTA's static unit (its record and first stages' weights issued before the wait),
      // then a lean stage loop (the generic decode's cold code would delay the first token tiles)
      constexpr uint64_t kHintW = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_FIRST);
      constexpr uint64_t kHintX = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_LAST);
      const int nbd = Il / 64, u = blockIdx.x;
      const bool g1 = u < nbd;
      const int blk = g1 ? u : u - nbd, nkt = g1 ? H >> 8 : Il >> 8;
      const CUtensorMap* wm = g1 ? &tm.w13 : &tm.w2;
      const CUtensorMap* bm = g1 ? &tm.x : &tm.h;
      const uint8_t* sfw = g1 ? p.sf13 + (size_t)blk * (H / 64) * 512 : p.sf2 + (size_t)blk * (Il / 64) * 512;
      const uint8_t* sfr = g1 ? p.xs : p.hsa;
      int s = 0, ph = 0, have_w = -1;
      unsigned have = 0;  // (down CTA: the flag word have_w's blocks seen stored)
      int* dctr = nullptr;  // (this launch's counter buffer: an L2 round trip, after the first loads)
      for (int kt = 0; kt < nkt; kt += KT) {
        Barrier::wait(&sm.empty[s], ph ^ 1);
        const bool issued = kt < pre * KT;  // (weights issued before the wait)
        if (!issued && cute::elect_one_sync()) {
          TxBarrier::arrive_and_expect_tx(&sm.full[s], KT * (kTileB + NP * 128 + 2 * 2048));
          cute::SM90_TMA_LOAD_3D::copy(wm, &sm.full[s], kHintW, sm.a[s], 0, blk * 128, kt);
          cute::SM90_BULK_COPY_G2S::copy(sfw + (size_t)kt * 2048, &sm.full[s], sm.sfa[s], KT * 2048);
        }
        __syncwarp();
        if (!g1) {
          // these k-tiles' hidden blocks (four 64-feature blocks per k-tile, a flag word per 32)
          const int b0 = 4 * kt, w = b0 >> 5;
          const unsigned need = ((1u << (4 * KT)) - 1u) << (b0 & 31);
          if (w != have_w) {
            have_w = w;
            have = 0;
          }
          if ((have & need) != need) {
            if (!dctr) dctr = moe_fp4_ctr(p.ctr, cpar, 0, lane);
            const unsigned* const dn1 = reinterpret_cast<const unsigned*>(dctr + kMoeFp4CtrItems);
            while (((have = ld_acquire_u(dn1 + w)) & need) != need) __nanosleep(32);
            have = __shfl_sync(0xffffffffu, have, 0);
            asm volatile("fence.proxy.async.global;\n" ::: "memory");
          }
        }
        if (cute::elect_one_sync()) {
          cute::SM90_TMA_LOAD_3D::copy(bm, &sm.full[s], kHintX, sm.b[s], 0, 0, kt);
          cute::SM90_BULK_COPY_G2S::copy(sfr + (size_t)kt * 2048, &sm.full[s], sm.sfb[s], KT * 2048);
        }
        __syncwarp();
        if (++s == S) {
          s = 0;
          ph ^= 1;
        }
      }
      Barrier::wait(&sm.empty[s], ph ^ 1);
      if (cute::elect_one_sync()) {
        Fp4Unit e{};
        e.kind = kEnd;
        sm.u[1] = e;
        TxBarrier::arrive_and_expect_tx(&sm.full[s], 0);
      }
      __syncwarp();
      if (!dctr) dctr = moe_fp4_ctr(p.ctr, cpar, 0, lane);
      if (lane == 0) moe_fp4_ctr_end(p.ctr, dctr, cpar, (Il / 64 + 31) >> 5);
    } else if (CONV || lane == 0) {
      constexpr uint64_t kHintW = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_FIRST);
      constexpr uint64_t kHintX = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_LAST);
      int s = 0, ph = 0;
      // pair claims (kMoeFp4OptPair): with more than 8 units per CTA, two units per atomic while
      // more than 4 per CTA remain (one claim round trip per two unit switches; single claims near
      // the end keep the tail balanced)
      bool pair = (p.opts & kMoeFp4OptPair) && n_all > 8 * (int)gridDim.x;
      bool rt_wait = RT, rt_wait3 = RT;  // (in-kernel routing: the items / offsets not taken yet)
      bool rt_t0 = RT;                   // (rt_local: token 0's routing not taken yet, bar 4)
      int u_pair = -1, last = 0;  // (lane 0)
      auto claim = [&](int seq) {
        int v;
        if (seq == 0 && pre) {
          v = (int)blockIdx.x;
        } else if (seq == 0 && (RT || tab0)) {
          v = u0;
        } else if (u_pair >= 0) {
          v = u_pair;
          u_pair = -1;
        } else {
          const bool two = pair && last + 4 * (int)gridDim.x < n_all;
          v = e1s + atomicAdd(queue, two ? 2 : 1);
          if (two) u_pair = v + 1;
        }
        last = v;
        return v;
      };
      // S2_EARLY (kS2S): the queue holds the shared expert's down units right after the gate / up units the CTAs
      // without a shared gate / up unit claim at entry (W of them): the shared gate / up CTAs, claiming next, take
      // them (their last parts' epilogues sum the shared blocks: no SH_SUM with S2_EARLY), instead of 32 other CTAs
      // holding them from the end of the gate / up claims until the sums while the second-wave gate / up units go to
      // the shared gate / up CTAs; s2e_map: a queue position -> its unit in the order S1, R1, S2, R2
      const bool s2e = kS2S && (p.opts & kMoeFp4OptS2Early) && ns2 > 0;
      const int s2w = (int)gridDim.x - e1s;
      auto s2e_map = [&](int v) {
        if (!s2e) return v;
        const int w1 = min(s2w, e2 - e1);
        if (v >= e1 + w1 && v < e3) return v < e1 + w1 + ns2 ? e2 + (v - e1 - w1) : v - ns2;
        return v;
      };
      for (int seq = 0;; ++seq) {
        int u;
        if constexpr (CONV) {
          int uv = 0;
          if (lane == 0) uv = claim(seq);
          u = prod_bcast<CONV>(uv);
        } else {
          u = claim(seq);
        }
        // (in-kernel routing: 1 when this unit's first stage was issued right after the items)
        [[maybe_unused]] int fast = 0;
        if constexpr (RT) {
          // in-kernel routing: the epilogue warps' work items (bar 2) before the first unit past
          // the shared expert's gate / up units (the unit decode needs the item count and the
          // items' experts), their rows' offsets (bar 3) before the first down unit
          if (rt_wait && u >= e1) {
            // a gate / up unit's expert-independent fields while the routing runs, its first stage
            // issued as soon as the items are published (the decode and the stage loop's first
            // pass are cold code; they follow while this stage loads)
            const int r = u - e1;
            int f_gi = r >> sh1, f_blk = r & ((1 << sh1) - 1);
            size_t f_sfw = (size_t)f_blk * (H / 64) * 512;
            auto first_stage = [&](int g) {  // (the unit's record and its first stage)
              Barrier::wait(&sm.empty[s], ph ^ 1);
              if (prod_elect<CONV>()) {
                Fp4Unit fu{};
                fu.kind = kR1;
                fu.gi = f_gi;
                fu.blk = f_blk;
                fu.g = g;
                fu.fp4 = !rbf;
                fu.nkt = nkt1;
                fu.n = Nr;
                sm.u[seq % kRing] = fu;
                constexpr uint64_t kHintW1 = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_FIRST);
                constexpr uint64_t kHintX1 = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_LAST);
                TxBarrier::arrive_and_expect_tx(&sm.full[s], KT * (kTileB + Nr * 128 + (rbf ? 0 : 2 * 2048)));
                const int wr = g * 2 * Il + f_blk * 128;
                if constexpr (KT == 2) {
                  cute::SM90_TMA_LOAD_3D::copy(&tm.w13, &sm.full[s], kHintW1, sm.a[s], 0, wr, 0);
                  cute::SM90_TMA_LOAD_3D::copy(&tm.x, &sm.full[s], kHintX1, sm.b[s], 0, 0, 0);
                } else {
                  cute::SM90_TMA_LOAD_2D::copy(&tm.w13, &sm.full[s], kHintW1, sm.a[s], 0, wr);
                  cute::SM90_TMA_LOAD_2D::copy(&tm.x, &sm.full[s], kHintX1, sm.b[s], 0, 0);
                }
                if (!rbf) {
                  cute::SM90_BULK_COPY_G2S::copy(p.sf13 + (size_t)g * sf13_g + f_sfw, &sm.full[s], sm.sfa[s],
                                                 KT * 2048);
                  cute::SM90_BULK_COPY_G2S::copy(p.xs, &sm.full[s], sm.sfb[s], KT * 2048);
                }
              }
              prod_sync<CONV>();
              if (++s == S) {
                s = 0;
                ph ^= 1;
              }
              fast = 1;
            };
            if constexpr (rt_local) {
              // token 0's routing (epilogue warp 2, bar 4): when all of its experts are on this rank
              // they are the work items 0 .. top_k - 1 in slot order (the items are the experts in
              // the order of their first pair), so a unit of theirs runs before the other tokens'
              // routing and the items (RT_T0): its item entry (the value the items will hold) and at
              // least top_k items for its decode; the items are taken at the next unit.
              // S2_EARLY: only a position below the S2 slots whatever the item count (r < s2w): the
              // position is decoded before the items, unmapped; at Il 2048 (EP) token 0's 256 units
              // reach past the entry claims, and the shared gate / up CTAs' first claims (the S2
              // slots) would become gate / up first stages under S2 records (the rings would
              // desynchronize)
              if (rt_t0) {
                asm volatile("bar.sync 4, 64;\n" ::: "memory");
                rt_t0 = false;
                if (pow2 && srt.early && f_gi < p.rt_topk && (!s2e || r < s2w)) {
                  const int g = prod_bcast<CONV>(srt.g[f_gi]);
                  if (lane == 0) sm.act[f_gi] = g;
                  __syncwarp();
                  e2 = e1 + p.rt_topk * nb1;
                  e3 = e2 + ns2;
                  n_all = e3 + p.rt_topk * nb2;
                  first_stage(g);
                }
              }
            }
            if (!fast) {
              asm volatile("bar.sync 2, 160;\n" ::: "memory");
              n_items = srt.items;
              e2 = e1 + n_items * nb1;
              e3 = e2 + ns2;
              n_all = e3 + n_items * nb2;
              pair = (p.opts & kMoeFp4OptPair) && n_all > 8 * (int)gridDim.x;
              rt_wait = false;
              int uq = u;
              if constexpr (kS2S) {
                uq = s2e_map(u);
                f_gi = (uq - e1) >> sh1, f_blk = (uq - e1) & ((1 << sh1) - 1);
                f_sfw = (size_t)f_blk * (H / 64) * 512;
              }
              if (pow2 && uq < e2) first_stage(prod_bcast<CONV>(sm.act[f_gi] & 0xffff));
            }
          }
          if (rt_wait3 && (kS2S ? s2e_map(u) : u) >= e2) {
            asm volatile("bar.sync 3, 160;\n" ::: "memory");
            rt_wait3 = false;
          }
        }
        if constexpr (kS2S) u = s2e_map(u);  // (the unit in the order S1, R1, S2, R2 of the decode below)
        Fp4Unit cu{};
        cu.kind = kEnd;
        const CUtensorMap* wmap = nullptr;
        const CUtensorMap* bmap = nullptr;
        int wrow = 0, brow = 0, kt0 = 0, bbytes = 0;
        const uint8_t* sfw = nullptr;
        const uint8_t* sfr = nullptr;
        const unsigned* need_mask = nullptr;
        if (u < e1) {
          cu.kind = kS1;
          cu.blk = u / kSplit;
          cu.ks = u % kSplit;
          cu.nkt = H / 64 / kSplit;
          cu.n = Ns;
          kt0 = cu.ks * cu.nkt;
          wmap = &tm.s13, wrow = cu.blk * 128, bmap = &tm.sx, bbytes = Ns * 128;
        } else if (u < e2 || (u >= e3 && u < n_all)) {
          const bool ph1 = u < e2;
          cu.kind = ph1 ? kR1 : kR2;
          if (pow2) {
            const int r = ph1 ? u - e1 : u - e3, sh = ph1 ? sh1 : sh2;
            cu.gi = r >> sh;
            cu.blk = r & ((1 << sh) - 1);
          } else {
            cu.gi = ph1 ? (u - e1) / nb1 : (u - e3) / nb2;
            cu.blk = ph1 ? (u - e1) % nb1 : (u - e3) % nb2;
          }
          cu.g = prod_bcast<CONV>((tab ? sm.act[cu.gi] : p.active[cu.gi]) & 0xffff);
          // (in-kernel routing: the offsets by work item, taken for down units only; the epilogue
          // reads a gate / up unit's)
          const int oi = RT ? cu.gi : cu.g;
          cu.rb = RT && ph1 ? 0 : prod_bcast<CONV>(tab ? sm.off[oi] : p.offsets[oi]);
          cu.mrows = RT && ph1 ? 0 : prod_bcast<CONV>(tab ? sm.off[oi + 1] : p.offsets[oi + 1]) - cu.rb;
          cu.fp4 = !rbf;
          cu.nkt = ph1 ? nkt1 : nkt2;
          // phase 2: the expert's hidden rows (rounded up), else every token
          cu.n = ph1 ? Nr : min(Nr, rbf ? (cu.mrows + 15) & ~15 : (cu.mrows + 7) & ~7);
          bbytes = Nr * 128;
          if (ph1) {
            wmap = &tm.w13, wrow = cu.g * 2 * Il + cu.blk * 128, bmap = &tm.x;
            if (!rbf) {
              // (scale atoms of the tile's 128 rows: row block blk of the expert's w13)
              sfw = p.sf13 + (size_t)cu.g * sf13_g + (size_t)cu.blk * (H / 64) * 512;
              sfr = p.xs;
            }
          } else {
            wmap = &tm.w2, wrow = cu.g * H + cu.blk * (RB * 128), bmap = &tm.h, brow = cu.rb;
            need_mask = done1 + cu.gi;
            if (!rbf) {
              sfw = p.sf2 + (size_t)cu.g * sf2_g + (size_t)cu.blk * RB * (Il / 64) * 512;
              sfr = p.hsa + (size_t)cu.gi * hs_group;
            }
          }
        } else if (u < e3) {
          cu.kind = kS2;
          cu.blk = u - e2;
          cu.nkt = Is / 64;
          cu.n = Ns;
          wmap = &tm.sdn, wrow = cu.blk * 128, bmap = &tm.sh, bbytes = Ns * 128;
          need_mask = done_sh;
        }
        const bool fp4 = cu.fp4;
        unsigned have = 0;  // the hidden-row blocks seen stored (re-read only while some are missing)
        // RB > 1: k-tiles per 128-row block of the unit and the block-local k-tile of the stage
        [[maybe_unused]] const int kpb = RB > 1 ? (cu.kind == kR2 ? (int)((unsigned)cu.nkt / RB) : cu.nkt) : 0;
        [[maybe_unused]] int kb = fast ? KT : 0;
        for (int kt = fast ? KT : 0;; kt += KT) {
          Barrier::wait(&sm.empty[s], ph ^ 1);
          if (cu.kind == kEnd) {
            if (prod_elect<CONV>()) {
              sm.u[seq % kRing] = cu;
              TxBarrier::arrive_and_expect_tx(&sm.full[s], 0);
            }
            prod_sync<CONV>();
            break;
          }
          // RB > 1: the stage's first k-tile is k-tile kk of the unit's current 128-row block
          int kk = kt;
          if constexpr (RB > 1) kk = kb;
          if (prod_elect<CONV>() && !(seq == 0 && kt < pre * KT)) {  // (pre: issued before the wait)
            if (kt == 0) sm.u[seq % kRing] = cu;
            TxBarrier::arrive_and_expect_tx(&sm.full[s], KT * (kTileB + bbytes + (fp4 ? 2 * 2048 : 0)));
            if constexpr (KT == 2)
              cute::SM90_TMA_LOAD_3D::copy(wmap, &sm.full[s], kHintW, sm.a[s], 0, wrow, kt0 + kk);
            else
              cute::SM90_TMA_LOAD_2D::copy(wmap, &sm.full[s], kHintW, sm.a[s], (kt0 + kk) * 128, wrow);
            if (fp4)
              cute::SM90_BULK_COPY_G2S::copy(sfw + (size_t)kk * 2048, &sm.full[s], sm.sfa[s],
                                             KT * 2048);
          }
          prod_sync<CONV>();
          if (need_mask) {
            // this k-tile's hidden features are stored (64-feature blocks: four per NVFP4 k-tile)
            // (an acquire load + proxy fence is an L2 round trip on the producer's critical path:
            // only while blocks this k-tile needs have not been seen)
            const unsigned need = fp4 ? ((1u << (4 * KT)) - 1u) << (4 * kk) : ((1u << KT) - 1u) << kk;
            if ((have & need) != need) {
              if (kS2S && (p.opts & kMoeFp4OptShSum) && cu.kind == kS2) {  // (summed by the waiting warps)
                have = sh_sum_wait(done_sh, parts, need, ns1, p.sh_part, Ns, p.sh_T, p.sh_h, Is, p.limit, lane);
              } else {
                while (((have = ld_acquire_u(need_mask)) & need) != need) __nanosleep(32);
                have = (unsigned)prod_bcast<CONV>((int)have);
              }
              // (the async-proxy loads of these rows after the acquire; later k-tiles whose blocks
              // were already seen follow this fence in program order)
              asm volatile("fence.proxy.async.global;\n" ::: "memory");
            }
          }
          if (prod_elect<CONV>()) {
            if constexpr (KT == 2)
              cute::SM90_TMA_LOAD_3D::copy(bmap, &sm.full[s], kHintX, sm.b[s], 0, brow, kt0 + kk);
            else
              cute::SM90_TMA_LOAD_2D::copy(bmap, &sm.full[s], kHintX, sm.b[s], (kt0 + kk) * 128,
                                           brow);
            if (fp4)
              cute::SM90_BULK_COPY_G2S::copy(sfr + (size_t)kk * 2048, &sm.full[s], sm.sfb[s],
                                             KT * 2048);
          }
          prod_sync<CONV>();
          if constexpr (RB > 1) {
            kb += KT;
            if (kb >= kpb) {  // (the unit's next block: its rows and scale atoms)
              kb = 0;
              wrow += 128;
              sfw += (size_t)(Il / 64) * 512;
            }
          }
          if (++s == S) {
            s = 0;
            ph ^= 1;
          }
          if (kt + KT >= cu.nkt) break;
        }
        if (cu.kind == kEnd) break;
      }
      // this CTA claims no more units: it counts itself finished now (the last CTA to do so restores
      // the header for the next launch) instead of after its last epilogue, so the CTA's exit does not
      // wait for the count's round trip (every claim of this launch precedes every count)
      if (lane == 0) moe_fp4_ctr_end(p.ctr, ctr, cpar, n_items);
    }
  } else if (warp == 1) {
    // ---------------------------------------------------------------- MMA issuer
    int s = 0, ph = 0, buf = 0, aph = 0, par = 0;
    // stage 0's descriptors: operands (K-major, 128-byte swizzle) and scale atoms; a stage or K step
    // adds its byte offset >> 4 to the start address field (shared memory < 256 KB: no carry out of
    // the 14-bit field)
    const uint64_t da0 = desc_k_sw128(smem_u32(sm.a[0])), db0 = desc_k_sw128(smem_u32(sm.b[0]));
    const uint64_t dsa0 = desc_sf_atom(smem_u32(sm.sfa[0])), dsb0 = desc_sf_atom(smem_u32(sm.sfb[0]));
    for (int seq = 0;; ++seq) {
      Barrier::wait(&sm.full[s], ph);
      const Fp4Unit cu = sm.u[seq % kRing];
      Barrier::wait(&sm.acce[buf], aph ^ 1);  // the epilogue has read this accumulator buffer
      __syncwarp();
      if (lane == 0) Barrier::arrive(&sm.unitf[buf]);
      if (cu.kind == kEnd) break;  // (the epilogue sees the end marker)
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      const uint32_t accbuf = tmem + buf * Cfg::kAccCols;
      // the unit's instruction descriptors (N; the scale ids are the scale columns' lane bits, 0)
      const uint32_t idb = idesc_bf16n(cu.n);
      const uint32_t idf = uint32_t(idesc_fp4n(tmem + Cfg::kSfaCol, tmem + Cfg::kSfbCol, cu.n) >> 32);
      // the B box of this unit's kind: the k-tile stride within a stage (KT 2)
      const uint32_t bstep = (uint32_t)((cu.kind == kR1 || cu.kind == kR2 ? Nr : Ns) * 128) >> 4;
      // RB > 1: k-tiles per 128-row block (accumulator) of the unit, the block-local k-tile of the
      // stage and the block's accumulator
      [[maybe_unused]] const int kpb = RB > 1 ? (cu.kind == kR2 ? (int)((unsigned)cu.nkt / RB) : cu.nkt) : 0;
      [[maybe_unused]] int kb = 0;
      [[maybe_unused]] uint32_t accb = accbuf;
      for (int kt = 0; kt < cu.nkt; kt += KT) {
        if (kt > 0) {
          Barrier::wait(&sm.full[s], ph);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        }
        uint32_t acc = accbuf;
        int kl = kt;  // (the stage's first k-tile within the accumulator's block)
        if constexpr (RB > 1) {
          acc = accb;
          kl = kb;
          kb += KT;
          if (kb >= kpb) {  // (the next stage starts the unit's next block: its accumulator)
            kb = 0;
            accb += NMAX;
          }
        }
        // stage s's operand descriptors: stage 0's plus the byte offset >> 4 (start address field)
        const uint64_t da = da0 + (uint64_t)(s * ((KT * kTileB) >> 4));
        const uint64_t db = db0 + (uint64_t)(s * ((KT * NMAX * 128) >> 4));
        if (cute::elect_one_sync()) {
#pragma unroll
          for (int kk = 0; kk < KT; ++kk) {
            const uint64_t dak = da + (uint64_t)(kk * (kTileB >> 4)), dbk = db + (uint64_t)(kk * bstep);
            const uint32_t acc0 = (kl > 0 || kk > 0) ? 1u : 0u;
            if (cu.fp4) {
              const int pk = par ^ (kk & 1);
              const uint32_t sfa = tmem + Cfg::kSfaCol + pk * 16, sfb = tmem + Cfg::kSfbCol + pk * 16;
              const uint64_t dsa = dsa0 + (uint64_t)(s * ((KT * 2048) >> 4) + kk * (2048 >> 4)),
                             dsb = dsb0 + (uint64_t)(s * ((KT * 2048) >> 4) + kk * (2048 >> 4));
#pragma unroll
              for (int j = 0; j < 4; ++j) {
                sf_copy_raw(sfa + 4 * j, dsa + (uint64_t)(j * (512 >> 4)));
                sf_copy_raw(sfb + 4 * j, dsb + (uint64_t)(j * (512 >> 4)));
              }
#pragma unroll
              for (int j = 0; j < 4; ++j)
                mma_fp4_raw(dak + 2 * j, dbk + 2 * j, acc, (acc0 || j > 0) ? 1u : 0u, idf, sfa + 4 * j,
                            sfb + 4 * j);
            } else {
#pragma unroll
              for (int j = 0; j < 4; ++j)
                mma_bf16_raw(dak + 2 * j, dbk + 2 * j, acc, (acc0 || j > 0) ? 1u : 0u, idb);
            }
          }
        }
        __syncwarp();
        if (cu.fp4 && (KT & 1)) par ^= 1;
        cutlass::arch::umma_arrive(&sm.empty[s]);  // the stage is free once these complete
        if (++s == S) {
          s = 0;
          ph ^= 1;
        }
      }
      cutlass::arch::umma_arrive(&sm.accf[buf]);
      if (++buf == 2) {
        buf = 0;
        aph ^= 1;
      }
    }
  } else {
    // ---------------------------------------------------------------- epilogue (warps 2-5)
    const int q = warp & 3;  // tensor-memory lane quadrant: tile rows 32 q .. 32 q + 31
    const int row = 32 * q + lane;
    if constexpr (RT) {
      // in-kernel routing: every pair's local expert (srt.g: -1 another rank's, -2 the shared
      // expert) and per local expert the bitmask of its tokens (srt.mask)
      const int K1 = p.rt_topk + 1, TK = p.rt_topk, T = p.rt_T, E = p.rt_E, G = p.G, lo = p.rt_lo;
      const int ew = warp - 2, et = tid - 64;
      const bool out = blockIdx.x == 0;
      if constexpr (RTX) {
        // over the CTAs (more than kMoeFp4RtLocalT tokens): epilogue warp w of CTA c0 + j routes
        // token 4 j + w (c0: the first CTA without a static shared-expert unit) and writes its ids
        // / weights (moe_route_warp's, bitwise); every CTA then reads all of them once the count of
        // routed tokens is complete
        // (up to 12 tokens the ids are published as {epoch + 1, id} words, single-copy atomic: each
        // reader polls its words until they carry this launch's epoch, no fence, count or barrier
        // behind them; past 12 tokens the readers' polls of every word cost more than the fence and
        // count they save: the routing warps fence and count, one thread per CTA polls the count)
        const bool tagged = T <= 12;
        int* const rt_ready = ctr + kMoeFp4CtrRt;
        unsigned long long* const tag =
            reinterpret_cast<unsigned long long*>(p.ctr + kMoeFp4CtrTag + 2);
        const unsigned long long ep = (unsigned long long)(unsigned)(__ldcg(p.ctr + kMoeFp4CtrTag) + 1) << 32;
        const int c0 = e1s + (T + 3) / 4 <= (int)gridDim.x ? e1s : 0;
        const int t = ((int)blockIdx.x - c0) * 4 + ew;
        if ((int)blockIdx.x >= c0 && t < T) {
          const int id = route_token_warp<kMoeFp4RtMaxE / 32>(p.rt_logits + (size_t)t * E, rt_bl, E, TK,
                                                              p.rt_scale, E, p.rt_ids + (size_t)t * K1,
                                                              p.rt_w + (size_t)t * K1, lane);
          if (tagged) {
            if (lane < K1)
              asm volatile("st.relaxed.gpu.global.b64 [%0], %1;\n" ::"l"(tag + (size_t)t * K1 + lane),
                           "l"(ep | (unsigned)id)
                           : "memory");
          } else {
            __syncwarp();
            if (lane == 0) {
              __threadfence();
              atomicAdd(rt_ready, 1);
            }
          }
        }
        if (!tagged) {
          if (et == 0)
            while ((int)ld_acquire_u(reinterpret_cast<const unsigned*>(rt_ready)) < T) __nanosleep(20);
          epi_sync();
        }
        for (int i = et; i < T * K1; i += 128) {
          unsigned long long v = 0ull;
          if (tagged) {
            for (;;) {
              asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];\n" : "=l"(v) : "l"(tag + i) : "memory");
              if ((v & 0xffffffff00000000ull) == ep) break;
              __nanosleep(20);
            }
          }
          const int id = tagged ? (int)(unsigned)v : __ldcg(p.rt_ids + i);
          const int g = id >= lo && id < lo + G ? id - lo : (id >= E ? -2 : -1);
          srt.g[i] = g;
          if (g >= 0) atomicOr(&srt.mask[g], 1ull << (i / K1));
        }
      }
      // (within each CTA: the tokens were routed before the role split, see above)
      epi_sync();
      // (early items, kMoeFp4OptRtEi: the items were built and handed over in the routing block)
      const bool ei_done = rt_local && (MD & kMdRs) != 0 && (p.opts & kMoeFp4OptRtEi) != 0;
      // the work items (one per local expert with rows) for the producer, then the rows' tables.
      // Up to 128 pairs: the experts in the order of their first pair (a ballot per 32 pairs), their
      // offsets after the producer has the items (its gate / up units need the experts only); else
      // in expert order with their offsets (a scan of the experts' token counts, lane l taking the
      // experts [l per, l per + per): longer than the ballots for few pairs of many experts).
      const bool few = T * TK <= 128;
      if (ew == 0 && !ei_done) {
        if (few) {
          int nb = 0;  // (items of the earlier rounds)
#pragma unroll 1
          for (int p0 = 0; p0 < T * TK; p0 += 32) {
            const int pi = p0 + lane;
            bool first = false;
            int g = -1;
            if (pi < T * TK) {
              const int t = pi / TK;
              g = srt.g[t * K1 + pi % TK];
              first = g >= 0 && (srt.mask[g] & ((1ull << t) - 1ull)) == 0ull;
            }
            const unsigned um = __ballot_sync(0xffffffffu, first);
            if (first) {
              const int r = nb + __popc(um & ((1u << lane) - 1u));
              sm.act[r] = g;
              srt.item[g] = r;
            }
            nb += __popc(um);
          }
          if (lane == 0) srt.items = nb;
        } else {
          const int per = (G + 31) >> 5;
          int rows_l = 0, items_l = 0;
#pragma unroll 1
          for (int i = 0; i < per; ++i) {
            const int g = lane * per + i;
            const int c = g < G ? __popcll(srt.mask[g]) : 0;
            rows_l += c;
            items_l += c > 0;
          }
          int v = rows_l << 16 | items_l;  // (rows <= 64 x 16, items <= kTab)
#pragma unroll
          for (int o = 1; o < 32; o <<= 1) {
            const int u = __shfl_up_sync(0xffffffffu, v, o);
            if (lane >= o) v += u;
          }
          int rows = (v >> 16) - rows_l, items = (v & 0xffff) - items_l;
#pragma unroll 1
          for (int i = 0; i < per; ++i) {
            const int g = lane * per + i;
            const int c = g < G ? __popcll(srt.mask[g]) : 0;
            if (c > 0) {
              sm.act[items] = g;
              sm.off[items] = rows;
              srt.item[g] = items++;
              rows += c;
            }
          }
          if (lane == 31) {
            sm.off[v & 0xffff] = v >> 16;
            srt.items = v & 0xffff;
          }
        }
      }
      if (!ei_done) asm volatile("bar.arrive 2, 160;\n" ::: "memory");  // the producer's items
      if (few && ew == 0) {  // the items' offsets (exclusive prefix of their token counts)
        __syncwarp();
        const int nu = srt.items;
        int carry = 0;  // (rows of the earlier rounds' items)
#pragma unroll 1
        for (int b0 = 0; b0 < nu; b0 += 32) {
          const int i = b0 + lane;
          const int c = i < nu ? __popcll(srt.mask[sm.act[i]]) : 0;
          int v = c;
#pragma unroll
          for (int o = 1; o < 32; o <<= 1) {
            const int u = __shfl_up_sync(0xffffffffu, v, o);
            if (lane >= o) v += u;
          }
          if (i < nu) sm.off[i] = carry + v - c;
          carry += __shfl_sync(0xffffffffu, v, 31);
        }
        if (lane == 0) sm.off[nu] = carry;
      }
      epi_sync();
      for (int i = et; i < T * K1; i += 128) {  // every pair's grouped row, the rows' tokens
        const int t = i / K1, g = srt.g[i];
        int pos = -1;
        if (g >= 0) {
          pos = sm.off[srt.item[g]] + __popcll(srt.mask[g] & ((1ull << t) - 1ull));
          srt.rsrc[pos] = t | (p.y_tm ? i : pos) << 16;
        } else if (g == -2) {
          pos = T * TK + t;
        }
        if (out) p.rt_pos[i] = pos;
      }
      if (out && p.rt_hist) {  // (the local experts and their token counts, for the next step's
        const int nu = srt.items;  //  L2 prefetch of this layer: plain stores, read a step later)
        for (int i = et; i < nu; i += 128) {
          const int g = sm.act[i];
          p.rt_hist[1 + i] = g;
          p.rt_hist[1 + kMoeFp4HistMax + i] = __popcll(srt.mask[g]);
        }
        if (et == 0) p.rt_hist[0] = nu;
      }
      epi_sync();
      asm volatile("bar.arrive 3, 160;\n" ::: "memory");  // the producer's offsets
    }
    int buf = 0, aph = 0;
    const float sfs = 1.f / p.gs2;
    // the stored-block flags released by one fence of the flagging thread (REL1)
    const bool rel1 = (p.opts & kMoeFp4OptRel1) != 0;
    for (int seq = 0;; ++seq) {
      Barrier::wait(&sm.unitf[buf], aph);
      const Fp4Unit cu = sm.u[seq % kRing];
      if (cu.kind == kEnd) break;
      // kMdEw: a routed NVFP4 unit's output scale requested now, before the accumulator wait (an L2 round trip under the
      // weight stream instead of on the unit's epilogue chain after the wait)
      [[maybe_unused]] float al_pf = 1.f;
      if constexpr ((MD & kMdEw) != 0)
        if (cu.fp4 && (cu.kind == kR1 || cu.kind == kR2)) al_pf = __ldg((cu.kind == kR1 ? p.alpha13 : p.alpha2) + cu.g);
      Barrier::wait(&sm.accf[buf], aph);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      const bool routed = cu.kind == kR1 || cu.kind == kR2;
      // (in-kernel routing: a gate / up unit's rows by its work item, from the tables built above)
      const bool rt1 = RT && cu.kind == kR1;
      const int rb = rt1 ? sm.off[cu.gi] : cu.rb, mrows = rt1 ? sm.off[cu.gi + 1] - rb : cu.mrows;
      const uint32_t tbase = tmem + ((uint32_t)(32 * q) << 16) + buf * Cfg::kAccCols;
      auto release = [&] {  // the accumulator buffer has been read
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        __syncwarp();
        if (lane == 0) Barrier::arrive(&sm.acce[buf]);
        if (++buf == 2) {
          buf = 0;
          aph ^= 1;
        }
      };
      if constexpr (MD == kMdDense) {
        // dense: the warmed epilogues (one expert: hidden row i = token i, rows at hq / hsa from 0)
        if (cu.kind == kR2) {
          dn_epi_dn(tbase, mrows, cu.blk * 128 + row, __ldg(p.alpha2), p.y, H, false);
          release();
          continue;
        }
        dn_epi_gu(tbase, NP, __ldg(p.alpha13), &sm.gu[row][0]);
        release();
        epi_sync();
        dn_epi_sw(&sm.gu[0][0], NMAX, mrows, cu.blk, q, lane, p.limit, sfs, static_cast<uint8_t*>(p.hq), p.hsa, Il,
                  false);
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
        epi_sync();
        if (q == 0 && lane == 0) {
          asm volatile("fence.acq_rel.gpu;\n" ::: "memory");  // (one fence: cumulative over the barrier)
          atomicOr(done1 + (cu.blk >> 5), 1u << (cu.blk & 31));
        }
        continue;
      }
      if (cu.kind == kR2 || cu.kind == kS2) {
        // down projection: column i (routed: the expert's hidden row = grouped row rb + i, output row rb + i or,
        // in-kernel routing, the row in rsrc's high half: y_tm's token-major rows; shared: shared token i) of output
        // features 128 blk + 32 q + lane, bf16
        const float al = (MD & kMdEw) && cu.kind == kR2 ? al_pf : (cu.fp4 ? __ldg(p.alpha2 + cu.g) : 1.f);
        const int rows = routed ? mrows : p.sh_T;
        const int nbe = unit_blocks<RB>(cu);
        for (int jb = 0; jb < nbe; ++jb) {  // (RB > 1: the unit's blocks, accumulator jb)
          const int n = (cu.blk * nbe + jb) * 128 + row;
          for (int c0 = 0; c0 < rows; c0 += 8) {
            int dr[8];  // (in-kernel routing: the output rows, requested before the accumulator's load)
            if constexpr (RT) {
              if (routed)
#pragma unroll
                for (int i = 0; i < 8; ++i) dr[i] = srt.rsrc[rb + c0 + i] >> 16;
            }
            float v[8];
            tmem_ld8(tbase + jb * NMAX + c0, v);
            tmem_wait_ld();
#pragma unroll
            for (int i = 0; i < 8; ++i)
              if (c0 + i < rows) {
                bf16* dst = routed ? p.y + (size_t)(RT ? dr[i] : rb + c0 + i) * H
                                   : p.sh_y + (size_t)(c0 + i) * p.sh_ldy;
                dst[n] = __float2bfloat16(v[i] * al);
              }
          }
        }
        release();
        continue;
      }
      const int ncol = routed ? NP : p.sh_T;  // columns of the phase-1 tile used
      if (cu.kind == kS1) {
        // split-K part: the fp32 partial of the tile's rows to its slot; the last part of the
        // block sums the slots in part order
        const int ldp = Ns;
        float* slot = p.sh_part + ((size_t)(cu.blk * kSplit + cu.ks) * 128 + row) * ldp;
        for (int c0 = 0; c0 < ncol; c0 += 8) {
          float v[8];
          tmem_ld8(tbase + c0, v);
          tmem_wait_ld();
          reinterpret_cast<float4*>(slot + c0)[0] = make_float4(v[0], v[1], v[2], v[3]);
          reinterpret_cast<float4*>(slot + c0)[1] = make_float4(v[4], v[5], v[6], v[7]);
        }
        release();
        if (kS2S && (p.opts & kMoeFp4OptShSum)) {
          // (the down units' producers take and sum the block once its kSplit parts are counted)
          epi_sync();
          if (q == 0 && lane == 0) {
            asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
            atomicAdd(parts + cu.blk, 1);
          }
          continue;
        }
        if (rel1) {
          // one acq-rel fence by the counting thread (cumulative over the CTA barrier), not one
          // full fence per thread: the partials' release, and the last part's acquire of the others
          epi_sync();
          if (q == 0 && lane == 0) {
            asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
            sm.last = atomicAdd(parts + cu.blk, 1) == kSplit - 1;
            if (sm.last) asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
          }
          epi_sync();
          if (!sm.last) continue;
        } else {
          __threadfence();
          epi_sync();
          if (q == 0 && lane == 0) sm.last = atomicAdd(parts + cu.blk, 1) == kSplit - 1;
          epi_sync();
          if (!sm.last) continue;
          __threadfence();
        }
        const float* s0 = p.sh_part + ((size_t)cu.blk * kSplit * 128 + row) * ldp;
        for (int c0 = 0; c0 < ncol; c0 += 4) {
          float4 a = __ldcg(reinterpret_cast<const float4*>(s0 + c0));
#pragma unroll
          for (int k2 = 1; k2 < kSplit; ++k2) {
            const float4 b = __ldcg(reinterpret_cast<const float4*>(s0 + (size_t)k2 * 128 * ldp + c0));
            a.x += b.x;
            a.y += b.y;
            a.z += b.z;
            a.w += b.w;
          }
          sm.gu[row][c0] = __float2bfloat16(a.x);
          sm.gu[row][c0 + 1] = __float2bfloat16(a.y);
          sm.gu[row][c0 + 2] = __float2bfloat16(a.z);
          sm.gu[row][c0 + 3] = __float2bfloat16(a.w);
        }
      } else {
        // R1: rows 0..63 gate, 64..127 up of 64 features, as the separate GEMM writes them (bf16,
        // output scale applied), for every token column
        const float al = (MD & kMdEw) ? al_pf : (cu.fp4 ? __ldg(p.alpha13 + cu.g) : 1.f);
        for (int c0 = 0; c0 < ncol; c0 += 8) {
          float v[8];
          tmem_ld8(tbase + c0, v);
          tmem_wait_ld();
#pragma unroll
          for (int i = 0; i < 8; ++i) sm.gu[row][c0 + i] = __float2bfloat16(v[i] * al);
        }
        release();
      }
      epi_sync();
      // per warp pass: (hidden row i, 32-feature half fh): SwiGLU, then NVFP4 over 16-feature
      // blocks (lanes 0..15 / 16..31, 4 bytes per 8 lanes) or bf16
      const int nrow = routed ? mrows : p.sh_T;
      uint8_t* hsg = p.hsa + (size_t)cu.gi * hs_group;
      for (int w = q; w < 2 * nrow; w += 4) {
        const int i = w >> 1, fh = w & 1;
        const int t = routed ? (RT ? srt.rsrc[rb + i] & 0xffff : p.a_rows[rb + i]) : i;  // the token (column)
        const int fl = 32 * fh + lane;                // feature within the 64-feature block
        const int f = cu.blk * 64 + fl;               // feature of the expert
        const float h = swiglu_f(__bfloat162float(sm.gu[fl][t]),
                                 __bfloat162float(sm.gu[64 + fl][t]), p.limit);
        if (!cu.fp4) {
          bf16* dst = routed ? static_cast<bf16*>(p.hq) + (size_t)(rb + i) * Il
                             : p.sh_h + (size_t)i * Is;
          dst[f] = __float2bfloat16(h);
          continue;
        }
        float amax = fabsf(h);
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 4));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 8));
        float sfv = sfs * (amax * rcp_approx_f(6.f));
        const __nv_fp8_e4m3 s8(sfv);
        sfv = static_cast<float>(s8);
        const float osc = amax != 0.f ? rcp_approx_f(sfv * rcp_approx_f(sfs)) : 0.f;
        const uint32_t nib = e2m1_nibble(h * osc);
        const uint32_t byte = nib | (__shfl_down_sync(0xffffffff, nib, 1) << 4);
        uint32_t wd = byte | (__shfl_down_sync(0xffffffff, byte, 2) << 8);
        wd |= __shfl_down_sync(0xffffffff, wd, 4) << 16;
        if ((lane & 7) == 0)
          *reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(p.hq) + (size_t)(rb + i) * (Il / 2) +
                                       (f >> 1)) = wd;
        if ((lane & 15) == 0) hsg[nvfp4_sf_off(i, f >> 4, Il)] = s8.__x;
      }
      // this block of the hidden rows is stored
      if (!rel1) __threadfence();
      asm volatile("fence.proxy.async.global;\n" ::: "memory");
      epi_sync();  // (also: sm.gu is free for the next unit)
      if (q == 0 && lane == 0) {
        // (rel1: one acq-rel fence, cumulative over the CTA barrier, releases every thread's rows)
        if (rel1)
          asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
        else
          __threadfence();
        atomicOr(routed ? done1 + cu.gi : done_sh, 1u << cu.blk);
      }
    }
  }
  __syncthreads();
  if (warp == 1) {
    tmem_alloc.release_allocation_lock();
    tmem_alloc.free(tmem, Cfg::kTmemCols);
  }
}

// ------------------------------------------------------------------------------------------------
// Wide variant (NVFP4 routed experts): the activations are the MMA's M (128 rows: the step's tokens,
// an expert's hidden rows or the shared tokens, rows beyond them unused) and the weights its N = 256
// rows, so one MMA instruction (a fixed ~85 ns of the tensor pipe whatever N) moves 8 KB of weights,
// twice the tall layout's 4 KB, and the scale copies per weight byte drop by a quarter. Units:
//   S1 (128-feature block, split-K part of 8): shared gate / up rows [g64 u64 g64 u64], BF16;
//   R1 (expert, 128-feature block): w13 rows of the block, the epilogue's lanes = tokens: a routed
//      token's lane applies SwiGLU to its 128 features and quantizes its hidden row's block;
//   S2 (256 output features): shared down rows, BF16;  R2 (expert, 256 output features): w2 rows,
//      lanes = the expert's hidden rows.
// One 256-column accumulator: the epilogue drains it to registers (bf16, or the S1 partials to
// global) before releasing it, then computes while the next unit's MMAs run.
namespace {
constexpr int kWB = 256 * 128;  // weight tile of a k-tile: 256 rows x 128 bytes
constexpr int kWSplit = kMoeFp4WideSplit;  // shared gate / up: split-K parts
constexpr int kWSfCol = 256;    // tensor memory: accumulator [0, 256), scale atoms from 256
constexpr int kWSfPar = 48;     // per k-tile parity: activations 4 atoms (16), weights 8 (32)

struct WUnit {
  int kind, gi, blk, ks, g, rb, mrows, nkt;
};

// shared memory: [S][abytes] activation tiles (abytes = the rows loaded x 128 B: the MMA reads
// 128 rows, those past the tile land in the next tile / the weight tiles and give unused lanes),
// [S][kWB] weight tiles, [S][4096] weight scale atoms, [S][2048] activation scale atoms, control
struct WCtl {
  uint64_t full[5], empty[5], accf, acce, unitf;
  WUnit u[kRing];
  uint32_t tmem;
  int last;
};
__host__ __device__ constexpr size_t wide_smem(int S, int abytes) {
  return (size_t)S * (abytes + kWB + 4096 + 2048) + sizeof(WCtl) + 1024;
}

__device__ __forceinline__ void tmem_ld32(uint32_t taddr, uint32_t* r) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,"
      "%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
        "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]),
        "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]),
        "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]),
        "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr));
}

__device__ __forceinline__ void tmem_ld64(uint32_t taddr, uint32_t* r) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]), "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]), "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63])
               : "r"(taddr));
}

__device__ __forceinline__ void tmem_ld128(uint32_t taddr, uint32_t* r) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x128.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, [%128];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]), "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]), "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63]), "=r"(r[64]), "=r"(r[65]), "=r"(r[66]), "=r"(r[67]), "=r"(r[68]), "=r"(r[69]), "=r"(r[70]), "=r"(r[71]), "=r"(r[72]), "=r"(r[73]), "=r"(r[74]), "=r"(r[75]), "=r"(r[76]), "=r"(r[77]), "=r"(r[78]), "=r"(r[79]), "=r"(r[80]), "=r"(r[81]), "=r"(r[82]), "=r"(r[83]), "=r"(r[84]), "=r"(r[85]), "=r"(r[86]), "=r"(r[87]), "=r"(r[88]), "=r"(r[89]), "=r"(r[90]), "=r"(r[91]), "=r"(r[92]), "=r"(r[93]), "=r"(r[94]), "=r"(r[95]), "=r"(r[96]), "=r"(r[97]), "=r"(r[98]), "=r"(r[99]), "=r"(r[100]), "=r"(r[101]), "=r"(r[102]), "=r"(r[103]), "=r"(r[104]), "=r"(r[105]), "=r"(r[106]), "=r"(r[107]), "=r"(r[108]), "=r"(r[109]), "=r"(r[110]), "=r"(r[111]), "=r"(r[112]), "=r"(r[113]), "=r"(r[114]), "=r"(r[115]), "=r"(r[116]), "=r"(r[117]), "=r"(r[118]), "=r"(r[119]), "=r"(r[120]), "=r"(r[121]), "=r"(r[122]), "=r"(r[123]), "=r"(r[124]), "=r"(r[125]), "=r"(r[126]), "=r"(r[127])
               : "r"(taddr));
}

__device__ __forceinline__ uint32_t pack_bf16(float a, float b) {
  const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ float bf16_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

// v[0..7] -> eight e2m1 nibbles, v[0] in the lowest
__device__ __forceinline__ uint32_t e2m1x8(const float* v) {
  uint32_t r;
  asm volatile(
      "{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "}\n"
      : "=r"(r)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
  return r;
}

// one 16-value block as nvfp4_quant / swiglu_quant_nvfp4: scale byte and 8 bytes of values
__device__ __forceinline__ uint8_t quant16(const float* v, float sfs, uint2& out) {
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
  float sfv = sfs * (amax * rcp_approx_f(6.f));
  const __nv_fp8_e4m3 s8(sfv);
  sfv = static_cast<float>(s8);
  const float osc = amax != 0.f ? rcp_approx_f(sfv * rcp_approx_f(sfs)) : 0.f;
  float w[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = v[i] * osc;
  out.x = e2m1x8(w);
  out.y = e2m1x8(w + 8);
  return s8.__x;
}
}  // namespace

// NA / NAs: the token / shared-token rows loaded per activation tile (multiples of 8)
template <int S>
__global__ void __launch_bounds__(kThreads, 1)
    moe_fp4w_kernel(const MoeFp4Args p, int T, int NA, int NAs, const __grid_constant__ Fp4Maps tm) {
  if (!p.late_trigger) pdl_trigger();
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t pad = (1024u - (smem_u32(smem_raw) & 1023u)) & 1023u;
  const int abytes = max(NA, NAs) * 128;  // (multiple of 1024)
  uint8_t* const a_t = smem_raw + pad;
  uint8_t* const w_t = a_t + S * abytes;
  uint8_t* const sfw_t = w_t + S * kWB;
  uint8_t* const sfa_t = sfw_t + S * 4096;
  WCtl& sm = *reinterpret_cast<WCtl*>(sfa_t + S * 2048);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int s = 0; s < S; ++s) {
      Barrier::init(&sm.full[s], 1);
      Barrier::init(&sm.empty[s], 1);
    }
    Barrier::init(&sm.accf, 1);
    Barrier::init(&sm.acce, 4);
    Barrier::init(&sm.unitf, 1);
    cutlass::arch::fence_barrier_init();
  }
  cute::TMEM::Allocator1Sm tmem_alloc{};
  if (warp == 1) tmem_alloc.allocate(512, &sm.tmem);
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const uint32_t tmem = sm.tmem;
  pdl_wait();
  if (p.late_trigger) pdl_trigger();
  int cpar;
  int* const ctr = moe_fp4_ctr(p.ctr, cpar, warp, lane);
  const int n_items = *p.num_active;
  const int H = p.H, Il = p.Il;
  const int Is = p.sh_T > 0 ? p.sh_inter : 0;
  const int nb1 = Il / 128, nb2 = H / 256;  // routed units per expert
  const int ns1 = Is / 128, ns2 = Is ? H / 256 : 0;
  const int e1 = ns1 * kWSplit, e2 = e1 + n_items * nb1, e3 = e2 + ns2, n_all = e3 + n_items * nb2;
  int* const queue = p.ctr + 1;  // (the header's)
  unsigned* const done_sh = reinterpret_cast<unsigned*>(ctr);
  int* const parts = ctr + 1;
  unsigned* const done1 = reinterpret_cast<unsigned*>(ctr + kMoeFp4CtrItems);
  const size_t hs_group = nvfp4_sf_bytes(128, Il);  // an expert's hidden-row scale region

  if (warp == 0) {
    // ---------------------------------------------------------------- producer
    if (lane == 0) {
      constexpr uint64_t kHintW = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_FIRST);
      constexpr uint64_t kHintX = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_LAST);
      int s = 0, ph = 0;
      for (int seq = 0;; ++seq) {
        const int u = atomicAdd(queue, 1);
        WUnit cu{};
        cu.kind = kEnd;
        const CUtensorMap* wmap = nullptr;
        const CUtensorMap* amap = nullptr;
        int wrow = 0, arow = 0, kt0 = 0, a_ld = 0;  // a_ld: activation bytes loaded
        const uint8_t* sfw0 = nullptr;
        const uint8_t* sfw1 = nullptr;
        const uint8_t* sfa = nullptr;
        const unsigned* need_mask = nullptr;
        bool fp4 = false;
        if (u < e1) {
          cu.kind = kS1;
          cu.blk = u / kWSplit;
          cu.ks = u % kWSplit;
          cu.nkt = H / 64 / kWSplit;
          kt0 = cu.ks * cu.nkt;
          wmap = &tm.s13, wrow = cu.blk * 256, amap = &tm.sx, a_ld = NAs * 128;
        } else if (u < e2 || (u >= e3 && u < n_all)) {
          const bool ph1 = u < e2;
          cu.kind = ph1 ? kR1 : kR2;
          cu.gi = ph1 ? (u - e1) / nb1 : (u - e3) / nb2;
          cu.blk = ph1 ? (u - e1) % nb1 : (u - e3) % nb2;
          cu.g = p.active[cu.gi] & 0xffff;
          cu.rb = p.offsets[cu.g];
          cu.mrows = p.offsets[cu.g + 1] - cu.rb;
          cu.nkt = (ph1 ? H : Il) / 256;
          fp4 = true;
          a_ld = NA * 128;
          if (ph1) {
            wmap = &tm.w13, wrow = cu.g * 2 * Il + cu.blk * 256, amap = &tm.x;
            // (the tile's two 128-row blocks 2 blk, 2 blk + 1 of the expert's w13 scales)
            sfw0 = p.sf13 + (size_t)cu.g * nvfp4_sf_bytes(2 * Il, H) +
                   (size_t)(2 * cu.blk) * (H / 64) * 512;
            sfw1 = sfw0 + (size_t)(H / 64) * 512;
            sfa = p.xs;
          } else {
            wmap = &tm.w2, wrow = cu.g * H + cu.blk * 256, amap = &tm.h, arow = cu.rb;
            need_mask = done1 + cu.gi;
            sfw0 = p.sf2 + (size_t)cu.g * nvfp4_sf_bytes(H, Il) +
                   (size_t)(2 * cu.blk) * (Il / 64) * 512;
            sfw1 = sfw0 + (size_t)(Il / 64) * 512;
            sfa = p.hsa + (size_t)cu.gi * hs_group;
          }
        } else if (u < e3) {
          cu.kind = kS2;
          cu.blk = u - e2;
          cu.nkt = Is / 64;
          wmap = &tm.sdn, wrow = cu.blk * 256, amap = &tm.sh, a_ld = NAs * 128;
          need_mask = done_sh;
        }
        for (int kt = 0;; ++kt) {
          Barrier::wait(&sm.empty[s], ph ^ 1);
          if (kt == 0) sm.u[seq % kRing] = cu;
          if (cu.kind == kEnd) {
            TxBarrier::arrive_and_expect_tx(&sm.full[s], 0);
            break;
          }
          TxBarrier::arrive_and_expect_tx(&sm.full[s], kWB + a_ld + (fp4 ? 4096 + 2048 : 0));
          cute::SM90_TMA_LOAD_2D::copy(wmap, &sm.full[s], kHintW, w_t + s * kWB, (kt0 + kt) * 128,
                                       wrow);
          if (fp4) {
            cute::SM90_BULK_COPY_G2S::copy(sfw0 + (size_t)kt * 2048, &sm.full[s], sfw_t + s * 4096,
                                           2048);
            cute::SM90_BULK_COPY_G2S::copy(sfw1 + (size_t)kt * 2048, &sm.full[s],
                                           sfw_t + s * 4096 + 2048, 2048);
          }
          if (need_mask) {
            // the k-tile's hidden features are stored: R2 256 = two 128-feature R1 blocks, S2 64
            // (BF16) = half an S1 block
            const unsigned need = cu.kind == kR2 ? 3u << (2 * kt) : 1u << (kt >> 1);
            while ((ld_acquire_u(need_mask) & need) != need) __nanosleep(32);
            asm volatile("fence.proxy.async.global;\n" ::: "memory");
          }
          cute::SM90_TMA_LOAD_2D::copy(amap, &sm.full[s], kHintX, a_t + s * abytes, (kt0 + kt) * 128,
                                       arow);
          if (fp4)
            cute::SM90_BULK_COPY_G2S::copy(sfa + (size_t)kt * 2048, &sm.full[s], sfa_t + s * 2048, 2048);
          if (++s == S) {
            s = 0;
            ph ^= 1;
          }
          if (kt + 1 == cu.nkt) break;
        }
        if (cu.kind == kEnd) break;
      }
    }
  } else if (warp == 1) {
    // ---------------------------------------------------------------- MMA issuer
    int s = 0, ph = 0, par = 0;
    const uint32_t idw = idesc_bf16n(256);
    for (int seq = 0;; ++seq) {
      Barrier::wait(&sm.full[s], ph);
      const WUnit cu = sm.u[seq % kRing];
      if (seq > 0) Barrier::wait(&sm.acce, (seq - 1) & 1);  // the accumulator is drained
      __syncwarp();
      if (lane == 0) Barrier::arrive(&sm.unitf);
      if (cu.kind == kEnd) break;  // (the epilogue sees the end marker)
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      const bool fp4 = cu.kind == kR1 || cu.kind == kR2;
      for (int kt = 0; kt < cu.nkt; ++kt) {
        if (kt > 0) {
          Barrier::wait(&sm.full[s], ph);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        }
        const uint32_t a0 = smem_u32(a_t + s * abytes), w0 = smem_u32(w_t + s * kWB);
        if (fp4) {
          const uint32_t sfa = tmem + kWSfCol + par * kWSfPar, sfw = sfa + 16;
          if (cute::elect_one_sync()) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              sf_copy(sfa + 4 * j, smem_u32(sfa_t + s * 2048 + 512 * j));
              sf_copy(sfw + 8 * j, smem_u32(sfw_t + s * 4096 + 512 * j));
              sf_copy(sfw + 8 * j + 4, smem_u32(sfw_t + s * 4096 + 2048 + 512 * j));
            }
          }
          __syncwarp();
#pragma unroll
          for (int j = 0; j < 4; ++j)
            mma_fp4(desc_k_sw128(a0 + 32 * j), desc_k_sw128(w0 + 32 * j), tmem, kt > 0 || j > 0,
                    sfa + 4 * j, sfw + 8 * j, idesc_fp4n(sfa + 4 * j, sfw + 8 * j, 256));
          par ^= 1;
        } else {
#pragma unroll
          for (int j = 0; j < 4; ++j)
            mma_bf16(desc_k_sw128(a0 + 32 * j), desc_k_sw128(w0 + 32 * j), tmem, kt > 0 || j > 0,
                     idw);
        }
        cutlass::arch::umma_arrive(&sm.empty[s]);  // the stage is free once these complete
        if (++s == S) {
          s = 0;
          ph ^= 1;
        }
      }
      cutlass::arch::umma_arrive(&sm.accf);
    }
  } else {
    // ---------------------------------------------------------------- epilogue (warps 2-5)
    const int q = warp & 3;  // tensor-memory lane quadrant: activation rows 32 q .. 32 q + 31
    const int row = 32 * q + lane;
    const uint32_t tb = tmem + ((uint32_t)(32 * q) << 16);
    const float sfs = 1.f / p.gs2;
    // the stored-block flags and split-K counts released by one fence of the flagging / counting
    // thread (REL1)
    const bool wrel1 = (p.opts & kMoeFp4OptRel1) != 0;
    auto release = [&] {  // the accumulator has been read
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncwarp();
      if (lane == 0) Barrier::arrive(&sm.acce);
    };
    for (int seq = 0;; ++seq) {
      Barrier::wait(&sm.unitf, seq & 1);
      const WUnit cu = sm.u[seq % kRing];
      if (cu.kind == kEnd) break;
      // R1: this lane's token's hidden row of the expert (if routed to it)
      int hi = -1;
      if (cu.kind == kR1 && row < T)
        for (int i = 0; i < cu.mrows; ++i)
          if (__ldg(p.a_rows + cu.rb + i) == row) {
            hi = i;
            break;
          }
      const int lim = cu.kind == kR1 ? T : cu.kind == kR2 ? cu.mrows : p.sh_T;
      const bool qact = 32 * q < lim;  // (warp-uniform) rows of this quadrant are used
      const bool act = row < lim;
      Barrier::wait(&sm.accf, seq & 1);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      if (cu.kind == kR2 || cu.kind == kS2) {
        // down projection: activation row (routed: hidden row rb + row; shared: token row) x 256
        // output features of block blk, bf16
        const float al = cu.kind == kR2 ? __ldg(p.alpha2 + cu.g) : 1.f;
        bf16* dst = cu.kind == kR2 ? p.y + (size_t)(cu.rb + row) * H + cu.blk * 256
                                   : p.sh_y + (size_t)row * p.sh_ldy + cu.blk * 256;
        if (qact) {
          // two 128-column halves; the accumulator is released before the second half's stores
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            uint32_t r[128];
            tmem_ld128(tb + 128 * c, r);
            tmem_wait_ld();
            if (c == 1) release();
            if (act) {
#pragma unroll
              for (int k2 = 0; k2 < 16; ++k2) {
                uint4 o;
                o.x = pack_bf16(__uint_as_float(r[8 * k2]) * al, __uint_as_float(r[8 * k2 + 1]) * al);
                o.y = pack_bf16(__uint_as_float(r[8 * k2 + 2]) * al, __uint_as_float(r[8 * k2 + 3]) * al);
                o.z = pack_bf16(__uint_as_float(r[8 * k2 + 4]) * al, __uint_as_float(r[8 * k2 + 5]) * al);
                o.w = pack_bf16(__uint_as_float(r[8 * k2 + 6]) * al, __uint_as_float(r[8 * k2 + 7]) * al);
                reinterpret_cast<uint4*>(dst + 128 * c)[k2] = o;
              }
            }
          }
        } else {
          release();
        }
        continue;
      }
      if (cu.kind == kS1) {
        // split-K part: this token's fp32 partial (256 columns) to its slot; the last part of the
        // block sums the slots in part order
        float* slot = p.sh_part + ((size_t)(cu.blk * kWSplit + cu.ks) * NAs + row) * 256;
        if (qact) {
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            uint32_t r[128];
            tmem_ld128(tb + 128 * c, r);
            tmem_wait_ld();
            if (c == 1) release();
            if (act) {
#pragma unroll
              for (int k2 = 0; k2 < 32; ++k2)
                reinterpret_cast<float4*>(slot + 128 * c)[k2] =
                    make_float4(__uint_as_float(r[4 * k2]), __uint_as_float(r[4 * k2 + 1]),
                                __uint_as_float(r[4 * k2 + 2]), __uint_as_float(r[4 * k2 + 3]));
            }
          }
        } else {
          release();
        }
        if (wrel1) {
          // one acq-rel fence by the counting thread (cumulative over the CTA barrier): the
          // partials' release, and the last part's acquire of the others (as the tall kernel)
          epi_sync();
          if (q == 0 && lane == 0) {
            asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
            sm.last = atomicAdd(parts + cu.blk, 1) == kWSplit - 1;
            if (sm.last) asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
          }
          epi_sync();
        } else {
          __threadfence();
          epi_sync();
          if (q == 0 && lane == 0) sm.last = atomicAdd(parts + cu.blk, 1) == kWSplit - 1;
          epi_sync();
          if (sm.last) __threadfence();
        }
        if (!sm.last) continue;
        {
          // the 8 partials of (token t, features 4 fq .. 4 fq + 3) summed in part order, over the
          // 128 epilogue threads: gate columns 128 (f / 64) + f % 64, up columns + 64
          const size_t ps = (size_t)NAs * 256;  // part stride
          for (int it = tid - 64; it < p.sh_T * 32; it += 128) {
            const int t = it >> 5, f0 = 4 * (it & 31), gc = 128 * (f0 / 64) + f0 % 64;
            const float* s0 = p.sh_part + ((size_t)cu.blk * kWSplit * NAs + t) * 256 + gc;
            float4 g[kWSplit], u[kWSplit];
#pragma unroll
            for (int k2 = 0; k2 < kWSplit; ++k2) {
              g[k2] = __ldcg(reinterpret_cast<const float4*>(s0 + k2 * ps));
              u[k2] = __ldcg(reinterpret_cast<const float4*>(s0 + k2 * ps + 64));
            }
#pragma unroll
            for (int k2 = 1; k2 < kWSplit; ++k2) {
              g[0].x += g[k2].x, g[0].y += g[k2].y, g[0].z += g[k2].z, g[0].w += g[k2].w;
              u[0].x += u[k2].x, u[0].y += u[k2].y, u[0].z += u[k2].z, u[0].w += u[k2].w;
            }
            auto r16 = [](float x) { return __bfloat162float(__float2bfloat16(x)); };
            uint2 o;
            o.x = pack_bf16(swiglu_f(r16(g[0].x), r16(u[0].x), p.limit),
                            swiglu_f(r16(g[0].y), r16(u[0].y), p.limit));
            o.y = pack_bf16(swiglu_f(r16(g[0].z), r16(u[0].z), p.limit),
                            swiglu_f(r16(g[0].w), r16(u[0].w), p.limit));
            *reinterpret_cast<uint2*>(p.sh_h + (size_t)t * Is + cu.blk * 128 + f0) = o;
          }
        }
        if (!wrel1) __threadfence();
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
        epi_sync();
        if (q == 0 && lane == 0) {
          if (wrel1)
            asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
          else
            __threadfence();
          atomicOr(done_sh, 1u << cu.blk);
        }
        continue;
      }
      // R1: the gate / up outputs of the 256 columns [g64 u64 g64 u64] as the separate GEMM writes
      // them (bf16, output scale applied), drained into registers
      const float al = __ldg(p.alpha13 + cu.g);
      uint32_t v[128];
      if (qact) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          uint32_t r[64];
          tmem_ld64(tb + 64 * c, r);
          tmem_wait_ld();
#pragma unroll
          for (int k2 = 0; k2 < 32; ++k2)
            v[32 * c + k2] = pack_bf16(__uint_as_float(r[2 * k2]) * al,
                                       __uint_as_float(r[2 * k2 + 1]) * al);
        }
      }
      release();
      if (hi >= 0) {
        // the token's 128 features of the block: SwiGLU, NVFP4 (16-feature blocks)
        uint8_t* hrow = static_cast<uint8_t*>(p.hq) + (size_t)(cu.rb + hi) * (Il / 2) + cu.blk * 64;
        uint8_t* hsg = p.hsa + (size_t)cu.gi * hs_group;
#pragma unroll
        for (int grp = 0; grp < 8; ++grp) {
          const int f0 = 16 * grp, gc = 128 * (f0 / 64) + f0 % 64;
          float x[16];
#pragma unroll
          for (int j = 0; j < 16; j += 2) {
            const uint32_t gv = v[(gc + j) / 2], uv = v[(gc + 64 + j) / 2];
            x[j] = swiglu_f(bf16_lo(gv), bf16_lo(uv), p.limit);
            x[j + 1] = swiglu_f(bf16_hi(gv), bf16_hi(uv), p.limit);
          }
          uint2 o;
          const uint8_t s8 = quant16(x, sfs, o);
          *reinterpret_cast<uint2*>(hrow + grp * 8) = o;
          hsg[nvfp4_sf_off(hi, (cu.blk * 128 + f0) / 16, Il)] = s8;
        }
      }
      // this block of the expert's hidden rows is stored (wrel1: one acq-rel fence by the flagging
      // thread, cumulative over the CTA barrier, instead of a full fence per thread plus one)
      if (!wrel1) __threadfence();
      asm volatile("fence.proxy.async.global;\n" ::: "memory");
      epi_sync();
      if (q == 0 && lane == 0) {
        if (wrel1)
          asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
        else
          __threadfence();
        atomicOr(done1 + cu.gi, 1u << cu.blk);
      }
    }
  }
  __syncthreads();
  if (warp == 1) {
    tmem_alloc.release_allocation_lock();
    tmem_alloc.free(tmem, 512);
  }
  if (tid == 0) moe_fp4_ctr_end(p.ctr, ctr, cpar, n_items);
}

namespace {
// 2-D uint8 tensor map: rows x cols bytes, box (128 bytes, box_rows), 128-byte swizzle
CUtensorMap make_map(const void* base, int64_t rows, int64_t cols, int box_rows) {
  static PFN_cuTensorMapEncodeTiled_v12000 encode = [] {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult qr;
    CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000,
                                                cudaEnableDefault, &qr));
    CHECK(fn != nullptr && qr == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled not found");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
  }();
  CUtensorMap m;
  cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)cols};
  cuuint32_t box[2] = {128, (cuuint32_t)box_rows};
  cuuint32_t es[2] = {1, 1};
  const CUresult r =
      encode(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dims, strides, box, es,
             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
             CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: " + std::to_string((int)r));
  return m;
}

// 3-D view [cols / 128][rows][128] bytes of a rows x cols uint8 matrix: box (128 bytes, box_rows, 2)
// = two consecutive 128-byte k-tiles of box_rows rows, each a 128-byte-swizzled tile in shared memory
CUtensorMap make_map3(const void* base, int64_t rows, int64_t cols, int box_rows) {
  static PFN_cuTensorMapEncodeTiled_v12000 encode = [] {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult qr;
    CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000,
                                                cudaEnableDefault, &qr));
    CHECK(fn != nullptr && qr == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled not found");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
  }();
  CUtensorMap m;
  cuuint64_t dims[3] = {128, (cuuint64_t)rows, (cuuint64_t)(cols / 128)};
  cuuint64_t strides[2] = {(cuuint64_t)cols, 128};
  cuuint32_t box[3] = {128, (cuuint32_t)box_rows, 2};
  cuuint32_t es[3] = {1, 1, 1};
  const CUresult r =
      encode(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<void*>(base), dims, strides, box, es,
             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
             CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (3-D) failed: " + std::to_string((int)r));
  return m;
}

const CUtensorMap& cached_map3(const void* base, int64_t rows, int64_t cols, int box_rows) {
  static std::mutex mu;
  static std::map<std::tuple<const void*, int64_t, int64_t, int>, CUtensorMap> maps;
  std::lock_guard<std::mutex> lk(mu);
  const auto key = std::make_tuple(base, rows, cols, box_rows);
  auto it = maps.find(key);
  if (it == maps.end()) it = maps.emplace(key, make_map3(base, rows, cols, box_rows)).first;
  return it->second;
}

const CUtensorMap& cached_map(const void* base, int64_t rows, int64_t cols, int box_rows) {
  static std::mutex mu;
  static std::map<std::tuple<const void*, int64_t, int64_t, int>, CUtensorMap> maps;
  std::lock_guard<std::mutex> lk(mu);
  const auto key = std::make_tuple(base, rows, cols, box_rows);
  auto it = maps.find(key);
  if (it == maps.end()) it = maps.emplace(key, make_map(base, rows, cols, box_rows)).first;
  return it->second;
}

template <int NMAX, bool CONV, int KT = 1, int RB = 1, bool RT = false, bool RTX = false, int MD = 0>
void launch(const MoeFp4Args& a, int NP, int ctas, cudaStream_t st) {
  const int smem = sizeof(Fp4Smem<NMAX, KT>) + (RT ? sizeof(RtSmem) : 0) + 1024;
  // (KT 2: the operand maps are 3-D views, two k-tiles per box)
  auto cmap = [](const void* base, int64_t rows, int64_t cols, int box_rows) -> const CUtensorMap& {
    return KT == 2 ? cached_map3(base, rows, cols, box_rows) : cached_map(base, rows, cols, box_rows);
  };
  static bool configured[64] = {};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  if (!configured[dev]) {
    CUDA_CHECK(cudaFuncSetAttribute(moe_fp4_kernel<NMAX, CONV, KT, RB, RT, RTX, MD>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    configured[dev] = true;
  }
  // row bytes of the routed operands: NVFP4 half a byte per value, BF16 two
  const int64_t Hb = a.routed_bf16 ? 2 * (int64_t)a.H : a.H / 2;
  const int64_t Ilb = a.routed_bf16 ? 2 * (int64_t)a.Il : a.Il / 2;
  const int Nr = a.routed_bf16 ? (NP + 15) & ~15 : NP;
  Fp4Maps m;
  m.w13 = cmap(a.w13, (int64_t)a.G * 2 * a.Il, Hb, 128);
  m.w2 = cmap(a.w2, (int64_t)a.G * a.H, Ilb, 128);
  m.x = cmap(a.xq, a.x_rows, Hb, Nr);
  m.h = cmap(a.hq, a.h_rows, Ilb, Nr);
  if (a.sh_inter > 0 && a.sh_T > 0) {
    const int Ns = (a.sh_T + 15) & ~15;
    m.s13 = cmap(a.sh_gu, 2 * (int64_t)a.sh_inter, 2 * (int64_t)a.H, 128);
    m.sdn = cmap(a.sh_dn, a.H, 2 * (int64_t)a.sh_inter, 128);
    m.sx = cmap(a.sh_x, a.sh_x_rows, 2 * (int64_t)a.H, Ns);
    m.sh = cmap(a.sh_h, a.sh_h_rows, 2 * (int64_t)a.sh_inter, Ns);
  } else {
    m.s13 = m.sdn = m.sx = m.sh = m.w13;  // (unused)
  }
  launch_pdl(moe_fp4_kernel<NMAX, CONV, KT, RB, RT, RTX, MD>, dim3(ctas), dim3(kThreads), smem, st, a, NP, m);
}
}  // namespace

void moe_fp4_dense(const MoeFp4Args& a_in, int T, cudaStream_t st) {
  MoeFp4Args a = a_in;
  CHECK(T >= 1 && T <= 16, "moe_fp4_dense: 1 .. 16 tokens");
  CHECK(!a.routed_bf16 && !a.rt_logits && !a.y_tm && a.sh_T == 0 && a.G == 1 && a.H % 512 == 0 && a.Il % 512 == 0 &&
            a.Il / 64 <= 32 * kMoeFp4MaxItems && a.H <= 128 * kMoeFp4MaxItems,
        "moe_fp4_dense: NVFP4, one expert, H / Il multiples of 512 (two k-tiles per stage)");
  a.rt_T = T;  // (the rows)
  a.opts = kMoeFp4OptRel1;
  // one CTA per unit: the gate / up blocks, then the down blocks (9-16 tokens: 16-row token tiles,
  // four stages of shared memory instead of five)
  if (T <= 8)
    launch<8, true, 2, 1, false, false, kMdDense>(a, 8, a.Il / 64 + a.H / 128, st);
  else
    launch<16, true, 2, 1, false, false, kMdDense>(a, 16, a.Il / 64 + a.H / 128, st);
}

void moe_fp4_wide(const MoeFp4Args& a_in, int T, int ctas, cudaStream_t st) {
  MoeFp4Args a = a_in;
  // one fence per stored block / split-K count by the flagging thread (REL1)
  a.opts = kMoeFp4OptRel1;
  CHECK(!a.routed_bf16, "moe_fp4_wide: NVFP4 routed experts");
  CHECK(!a.rt_logits && !a.y_tm, "moe_fp4_wide: no in-kernel routing");
  CHECK(a.H % 256 == 0 && a.Il % 256 == 0 && a.Il / 128 <= 32,
        "moe_fp4_wide: H, Il multiples of 256, Il <= 4096");
  CHECK(T >= 1 && T <= kMoeFp4MaxTokens && a.sh_T >= 0 && a.sh_T <= kMoeFp4MaxTokens,
        "moe_fp4_wide: 1 .. 128 tokens");
  const bool sh = a.sh_inter > 0 && a.sh_T > 0;
  CHECK(!sh || (a.sh_inter % 128 == 0 && a.sh_inter / 128 <= 32 && (a.H / 64) % kWSplit == 0 &&
                a.sh_part && a.sh_y && a.sh_h_rows >= 1),
        "moe_fp4_wide: shared expert intermediate % 128, <= 4096; split-K partials, output");
  const int NA = (T + 7) & ~7, NAs = sh ? (a.sh_T + 7) & ~7 : 8;
  // five stages while they fit (activation tiles of <= 48 rows), else four
  const int abytes = std::max(NA, NAs) * 128;
  const bool s5 = wide_smem(5, abytes) <= 232448;
  const int smem = (int)wide_smem(s5 ? 5 : 4, abytes);
  static int configured[64] = {};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  if (!configured[dev]) {
    // (the largest size each instantiation may use)
    CUDA_CHECK(cudaFuncSetAttribute(moe_fp4w_kernel<5>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    232448));
    CUDA_CHECK(cudaFuncSetAttribute(moe_fp4w_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)wide_smem(4, 128 * 128)));
    configured[dev] = 1;
  }
  Fp4Maps m;
  m.w13 = cached_map(a.w13, (int64_t)a.G * 2 * a.Il, a.H / 2, 256);
  m.w2 = cached_map(a.w2, (int64_t)a.G * a.H, a.Il / 2, 256);
  m.x = cached_map(a.xq, a.x_rows, a.H / 2, NA);
  m.h = cached_map(a.hq, a.h_rows, a.Il / 2, NA);
  if (sh) {
    m.s13 = cached_map(a.sh_gu, 2 * (int64_t)a.sh_inter, 2 * (int64_t)a.H, 256);
    m.sdn = cached_map(a.sh_dn, a.H, 2 * (int64_t)a.sh_inter, 256);
    m.sx = cached_map(a.sh_x, a.sh_x_rows, 2 * (int64_t)a.H, NAs);
    m.sh = cached_map(a.sh_h, a.sh_h_rows, 2 * (int64_t)a.sh_inter, NAs);
  } else {
    m.s13 = m.sdn = m.sx = m.sh = m.w13;  // (unused)
  }
  if (s5)
    launch_pdl(moe_fp4w_kernel<5>, dim3(ctas), dim3(kThreads), smem, st, a, T, NA, NAs, m);
  else
    launch_pdl(moe_fp4w_kernel<4>, dim3(ctas), dim3(kThreads), smem, st, a, T, NA, NAs, m);
}

bool moe_fp4_route_in_ok(int T, int E, int top_k, int G) {
  // (past 32 tokens: the 64-column form with two k-tiles per stage)
  return T >= 1 && T <= kMoeFp4RtMaxT && top_k + 1 <= kMoeFp4RtMaxK1 && E <= kMoeFp4RtMaxE && G <= kTab;
}

// in-kernel routing within each CTA up to this many tokens under MoE tensor parallelism (kMoeFp4RtLocalT under expert
// parallelism), over the CTAs beyond (see moe_fp4_fused)
constexpr int kRtLocalTMoeTp = 8;

bool moe_fp4_route_ks_ok(int T, int H, int Il, int sh_inter, bool routed_bf16) {
  // (moe_fp4_fused's instance choice for rt_ks: its CHECK holds the two together)
  return !routed_bf16 && Il == 512 && T >= 1 && T <= kRtLocalTMoeTp && (H / 128) % 2 == 0 && (H / 256) % 2 == 0 &&
         (Il / 256) % 2 == 0 && (sh_inter <= 0 || ((H / 64 / kSplit) % 2 == 0 && (sh_inter / 64) % 2 == 0));
}

void moe_fp4_fused(const MoeFp4Args& a_in, int T, int ctas, cudaStream_t st) {
  MoeFp4Args a = a_in;
  CHECK(a.H % 256 == 0 && a.Il % 256 == 0, "moe_fp4_fused: H, Il multiples of 256");
  CHECK(a.Il / 64 <= 32, "moe_fp4_fused: at most 32 phase-1 blocks per expert (bitmask)");
  CHECK(T >= 1 && T <= kMoeFp4MaxTokens && a.sh_T >= 0 && a.sh_T <= kMoeFp4MaxTokens,
        "moe_fp4_fused: 1 .. 128 tokens");
  const bool sh = a.sh_inter > 0 && a.sh_T > 0;
  CHECK(!sh || (a.sh_inter % 64 == 0 && a.sh_inter / 64 <= 32 && (a.H / 64) % kSplit == 0 &&
                a.sh_part && a.sh_y && a.sh_h_rows >= 1),
        "moe_fp4_fused: shared expert intermediate % 64, <= 2048; split-K partials, output");
  // MMA N: the routed tokens rounded up to 8 (NVFP4) / 16 (BF16), the shared tokens to 16
  const int NP = (T + 7) & ~7;
  const int need = std::max(a.routed_bf16 ? (NP + 15) & ~15 : NP, sh ? (a.sh_T + 15) & ~15 : 0);
  // every launch's options (S2_EARLY and SH_SUM per launch, below); the producer is always the whole warp (CONV)
  constexpr unsigned kOpts =
      kMoeFp4OptRel1 | kMoeFp4OptS1Pre | kMoeFp4OptPair | kMoeFp4OptRtWarm | kMoeFp4OptRtT0 | kMoeFp4OptRtEi;
  // S2_EARLY for NVFP4 routed experts only: the BF16 ones' (the MTP layer's passes) gate / up units are 1 MB, and
  // past the 116 entry claims (2+ tokens) the remaining ones would follow the S2 slots and start late. SH_SUM: off
  // where the shared gate / up CTAs take no gate / up unit -- with S2_EARLY (their next units are the S2 units) or
  // when every gate / up unit is an entry claim (the MTP layer's 1-token passes: 64 units) -- so that the last
  // part's epilogue sum (128 threads, sooner than a down unit's warp) delays no gate / up flag; on otherwise (their
  // gate / up units' flags would wait for the sums).
  const bool s2 = !a.routed_bf16;
  const int s1c = std::min((a.sh_inter / 64) * kSplit, ctas);
  const bool r1_entry = a.rt_logits && std::min(T * a.rt_topk, a.G) * (a.Il / 64) <= ctas - s1c;
  const bool shsum = !(s2 || r1_entry);
  a.opts = kOpts | (s2 ? kMoeFp4OptS2Early : 0u) | (shsum ? kMoeFp4OptShSum : 0u);
  // in-kernel routing (the warp producer joins the epilogue warps' named barrier): tokens / slots / experts within
  // the tables' limits, the routing outputs
  CHECK(!a.rt_logits || (need <= 64 && T == a.rt_T && T <= kMoeFp4RtMaxT &&
                         a.rt_topk + 1 <= kMoeFp4RtMaxK1 &&
                         a.rt_E <= kMoeFp4RtMaxE && a.G <= kTab && a.rt_bias && a.rt_ids && a.rt_w &&
                         a.rt_pos),
        "moe_fp4_fused: in-kernel routing limits");
  CHECK(!a.y_tm || a.rt_logits, "moe_fp4_fused: token-major outputs (y_tm) with the in-kernel routing only");
  // two k-tiles per stage up to 64 columns, when every unit's k-tile count is even
  const int kb = a.routed_bf16 ? 64 : 256;  // routed values per 128-byte k-tile
  const bool kt2 = need <= 64 && (a.H / kb) % 2 == 0 && (a.Il / kb) % 2 == 0 &&
                   (!sh || ((a.H / 64 / kSplit) % 2 == 0 && (a.sh_inter / 64) % 2 == 0));
  // MoE tensor parallelism (a per-rank intermediate of 512): down units of 2 x 128 rows (two k-tiles per stage): a
  // shorter tail after the last gate / up units than units of 4 x 128 rows, fewer per-unit costs than 1 x 128 rows
  const bool tp_u = a.Il == 512 && !a.routed_bf16;
  const bool rb2 = tp_u && kt2 && (a.H / 128) % 2 == 0;
  // the routing within each CTA up to kRtLocalTMoeTp tokens under MoE tensor parallelism: at 5-8 tokens each epilogue
  // warp routes two tokens (its code warm, RT_WARM) instead of one warp per token over two CTAs whose ids every CTA
  // then polls, and the 5-8-token steps get token 0's early units (RT_T0) and the shared expert's down units after
  // the entry claims (S2_EARLY). Under expert parallelism (a per-rank intermediate of 2048: token 0's experts are
  // rarely all on the rank, so no early units, and the two routing rounds delay every entry unit) up to
  // kMoeFp4RtLocalT tokens, over the CTAs beyond.
  CHECK(!a.rt_ks || (a.rt_logits && need <= 32 && T <= kRtLocalTMoeTp && rb2),
        "moe_fp4_fused: rt_ks only for the in-CTA routing of the MoE TP instance (moe_fp4_route_ks_ok)");
  if (a.rt_logits && need > 32) {  // (in-kernel routing, 33-64 columns: two k-tiles per stage)
    CHECK(kt2, "moe_fp4_fused: in-kernel routing past 32 columns needs two k-tiles per stage");
    launch<64, true, 2, 1, true, true>(a, NP, ctas, st);
  } else if (a.rt_logits && T > (tp_u ? kRtLocalTMoeTp : kMoeFp4RtLocalT)) {  // (in-kernel routing over the CTAs)
    if (rb2)
      launch<32, true, 2, 2, true, true>(a, NP, ctas, st);
    else if (kt2)
      launch<32, true, 2, 1, true, true>(a, NP, ctas, st);
    else
      launch<32, true, 1, 1, true, true>(a, NP, ctas, st);
  } else if (a.rt_logits) {  // (in-kernel routing within each CTA)
    // the {key, score} routing's MoE TP instance, its epilogue's output scales requested before the accumulator wait
    if (a.rt_ks)
      launch<32, true, 2, 2, true, false, kMdRs | kMdEw>(a, NP, ctas, st);
    else if (rb2)
      launch<32, true, 2, 2, true>(a, NP, ctas, st);
    else if (kt2)
      launch<32, true, 2, 1, true>(a, NP, ctas, st);
    else
      launch<32, true, 1, 1, true>(a, NP, ctas, st);
  } else if (need <= 32 && rb2) {
    launch<32, true, 2, 2>(a, NP, ctas, st);
  } else if (need <= 32) {
    if (kt2)
      launch<32, true, 2>(a, NP, ctas, st);
    else
      launch<32, true>(a, NP, ctas, st);
  } else if (kt2) {
    launch<64, true, 2>(a, NP, ctas, st);  // (3 stages of 2 k-tiles; the 128-column form has 5 of 1)
  } else {
    launch<128, true>(a, NP, ctas, st);
  }
}

}  // namespace k
}  // namespace glm
