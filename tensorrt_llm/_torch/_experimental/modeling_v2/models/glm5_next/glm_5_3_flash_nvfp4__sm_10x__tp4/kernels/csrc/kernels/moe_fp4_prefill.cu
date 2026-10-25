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
// Fused NVFP4 MoE experts for prefill-sized steps on the SM100 tensor cores: the grouped-GEMM path's
// gate / up GEMM, SwiGLU + NVFP4 quantization and down GEMM in one persistent launch, in the tall form
// of moe_fp4.cu (the weights are the MMA's M side, one 128-row tile per unit; the N side is a work
// item's rows). Work items: an expert of up to kItemMax (208) grouped rows is one item (beyond 128 rows
// its rows sit in two row-tile slots per k-tile and every K step issues two MMAs into one accumulator),
// a larger expert is split into 128-row items (each re-streams the expert's weights). Units:
//   R1 (item, 64-feature block): w13 rows [gate f..f+63 | up f..f+63] x the item's permuted token rows
//      (moe_permute_rows); the epilogue applies SwiGLU and quantizes the 64 hidden features of every
//      row to NVFP4 (global scale gs2, swiglu_quant_nvfp4's arithmetic) into the hidden rows (the
//      item's own scale block);
//   R2 (item, 128 output features): w2 rows x the item's hidden rows -> y rows (bf16, alpha2).
// Queue order: every R1 unit (item-major), then every R2 unit (item-major); an R2 unit's k-tile waits
// for the R1 blocks of its features (a bitmask per item). The separate path writes h13 (bf16) between
// two grouped GEMMs and re-reads it for SwiGLU: here it never leaves the SM. Outputs are bitwise the
// separate path's.
// Warps: 0 weight producer (claims the next unit while the current one streams; writes the unit
// records), 1 MMA issuer, 2-9 epilogue (two per tensor-memory lane quadrant), 10 token / hidden-row
// producer (and the R2 units' hidden-row waits). The weight tiles (HBM) and the row tiles (mostly L2
// hits) have producers and rings of their own depth (one producer issuing all four copies of a k-tile
// would be busy most of the launch, and a shared ring leaves the weights' share of the bytes in flight
// too small).
#include <cuda.h>
#include <cudaTypedefs.h>

#include <map>
#include <mutex>
#include <tuple>

#include "cute/arch/copy_sm90_tma.hpp"
#include "fp4_umma.cuh"
#include "moe_fp4_prefill.h"
#include "nvfp4.h"
#include "pdl.cuh"

namespace glm {
namespace k {

namespace {
using namespace fp4mma;
using Barrier = cutlass::arch::ClusterBarrier;
using TxBarrier = cutlass::arch::ClusterTransactionBarrier;

// warps: 0 weight producer, 1 MMA, 2-9 epilogue (two per lane quadrant), 10 token / hidden-row producer
constexpr int kThreads = 352;
constexpr int kEpiWarps = 8;
constexpr int kWarpB = 10;
constexpr int kTileB = 128 * 128;  // one 128-row weight tile of a k-tile (128 bytes per row)
constexpr int kN = 128;            // rows per row-tile slot (one MMA's N)
// rows per work item: an expert of up to kItemMax rows is one item (its rows in two slots per k-tile
// beyond 128: two MMAs per K step into one accumulator), larger experts are 128-row items
constexpr int kItemMax = 208;
constexpr int kAccCols = kItemMax;  // accumulator columns per buffer
constexpr int kRing = 16;          // unit records in flight (producer ahead of the epilogue)
constexpr int kGuCols = 64;        // chunk rows staged per R1 epilogue pass
constexpr int kGuPitch = kGuCols + 8;  // bf16 per staged gate / up row (conflict-free column reads)
enum : int { kR1 = 1, kR2 = 3, kEnd = -1 };

struct PfUnit {
  int kind, item, blk, g, ch, rb, rows, n, nkt;
};

struct PfMaps {
  CUtensorMap w13, w2, x32, x64, x128, h32, h64, h128;
};

// W weight stages (HBM: the deeper ring), X token / hidden-row stages (mostly L2 hits)
template <int W, int X>
struct PfSmem {
  uint8_t a[W][kTileB];          // weight tile, 128-byte swizzle (1024-aligned)
  uint8_t b[X][kN * 128];        // token / hidden rows, 128-byte swizzle
  uint8_t sfa[W][4 * 512];       // the weight tile's scale atoms of the k-tile
  uint8_t sfb[X][4 * 512];       // the rows' scale atoms of the k-tile
  __nv_bfloat16 gu[128][kGuPitch];  // R1 epilogue: gate (rows 0..63) / up (64..127), 64 rows per pass
  uint64_t wfull[W], wempty[W], xfull[X], xempty[X], accf[2], acce[2], unitf[2], uready[kRing];
  PfUnit u[kRing];
  int off[kMoePfMaxGroups + 1], pad[kMoePfMaxGroups + 1];
  int cum[kMoePfMaxGroups + 1];  // work items before group g
  uint32_t tmem;
};

constexpr int kTmemCols = 512;
// tensor-memory columns: accumulators [2][kAccCols], then per k-tile parity the weight tile's 4 scale
// atoms (16 columns) and the rows' 4 atoms of each of the two 128-row slots (32 columns)
constexpr int kSfaCol = 2 * kAccCols, kSfbCol = 2 * kAccCols + 32;
static_assert(kSfbCol + 64 <= 512, "tensor-memory columns");

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

__device__ __forceinline__ void epi_sync() {  // the epilogue warps
  asm volatile("bar.sync 1, 256;\n" ::: "memory");
}

// as swiglu_quant_nvfp4 (swiglu1_fast): clamp, SiLU with div.approx, product rounded to bf16
__device__ __forceinline__ float swiglu_fast(float g, float u, float limit) {
  g = fminf(g, limit);
  u = fminf(fmaxf(u, -limit), limit);
  return __bfloat162float(__float2bfloat16(__fdividef(g, 1.f + __expf(-g)) * u));
}

// v[0..7] -> eight e2m1 nibbles, v[0] in the lowest (nvfp4.cu's e2m1x8)
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

__device__ __forceinline__ uint64_t idesc_fp4n(uint32_t tsfa, uint32_t tsfb, int N) {
  cute::UMMA::InstrDescriptorBlockScaled d = cute::UMMA::make_instr_desc_block_scaled<
      cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, cutlass::float_ue4m3_t, 128, 8,
      cute::UMMA::Major::K, cute::UMMA::Major::K>();
  d.n_dim_ = N >> 3;
  return cute::UMMA::make_runtime_instr_desc_block_scaled(d, tsfa, tsfb);
}

__device__ __forceinline__ void mma_fp4_raw(uint64_t da, uint64_t db, uint32_t td, uint32_t acc,
                                            uint32_t idh, uint32_t tsfa, uint32_t tsfb) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], "
      "p;\n}\n" ::"r"(td),
      "l"(da), "l"(db), "r"(idh), "r"(acc), "r"(tsfa), "r"(tsfb));
}
__device__ __forceinline__ void sf_copy_raw(uint32_t tmem_col, uint64_t desc) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n" ::"r"(tmem_col), "l"(desc));
}
}  // namespace

template <int W, int X>
__global__ void __launch_bounds__(kThreads, 1)
    moe_fp4_prefill_kernel(const MoePrefillArgs p, const __grid_constant__ PfMaps tm) {
  pdl_trigger();
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t pad0 = (1024u - (smem_u32(smem_raw) & 1023u)) & 1023u;
  PfSmem<W, X>& sm = *reinterpret_cast<PfSmem<W, X>*>(smem_raw + pad0);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int s = 0; s < W; ++s) {
      Barrier::init(&sm.wfull[s], 1);
      Barrier::init(&sm.wempty[s], 1);
    }
    for (int s = 0; s < X; ++s) {
      Barrier::init(&sm.xfull[s], 1);
      Barrier::init(&sm.xempty[s], 1);
    }
    for (int r = 0; r < kRing; ++r) Barrier::init(&sm.uready[r], 1);
    for (int i = 0; i < 2; ++i) {
      Barrier::init(&sm.accf[i], 1);
      Barrier::init(&sm.acce[i], kEpiWarps);
      Barrier::init(&sm.unitf[i], 1);
    }
    cutlass::arch::fence_barrier_init();
  }
  cute::TMEM::Allocator1Sm tmem_alloc{};
  if (warp == 1) tmem_alloc.allocate(kTmemCols, &sm.tmem);
  const int G = p.G, H = p.H, Il = p.Il;
  pdl_wait();
  int* const queue = p.ctr;
  int* const finished = p.ctr + 1;
  unsigned* const done1 = reinterpret_cast<unsigned*>(p.ctr + 2);
  // the producer's first claim while the tables load
  int u_next = 0;
  if (tid == 0) u_next = atomicAdd(queue, 1);
  // tables: group offsets / padded offsets, and the work items' prefix over the groups (warp 0)
  if (warp == 0) {
    int run = 0;
    for (int g0 = 0; g0 <= G; g0 += 32) {
      const int g = g0 + lane;
      int o = 0, pd = 0, o1 = 0;
      if (g <= G) {
        o = p.offsets[g];
        pd = p.pad[g];
      }
      if (g < G) o1 = p.offsets[g + 1];
      const int m = o1 - o;
      const int c = g < G && m > 0 ? (m <= kItemMax ? 1 : (m + kN - 1) / kN) : 0;
      int incl = c;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const int v = __shfl_up_sync(0xffffffff, incl, d);
        if (lane >= d) incl += v;
      }
      if (g <= G) {
        sm.off[g] = o;
        sm.pad[g] = pd;
        sm.cum[g] = run + incl - c;  // items before group g
      }
      run += __shfl_sync(0xffffffff, incl, 31);
    }
  }
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const uint32_t tmem = sm.tmem;
  const int n_items = sm.cum[G];
  const int nb1 = Il / 64, nb2 = H / 128;  // units per item
  const int n1 = n_items * nb1, n_all = n1 + n_items * nb2;
  // a work item's hidden-row scales: the CUTLASS layout of [kMoePfHsRows, Il]
  const size_t hs_group = nvfp4_sf_bytes(kMoePfHsRows, Il);
  const size_t hs_blk = nvfp4_sf_bytes(128, Il);  // (its second 128-row block)
  // item -> (group, chunk): the last group whose prefix is <= item
  auto item_group = [&](int item) {
    int lo = 0, hi = G;  // cum[lo] <= item < cum[hi]
    while (hi - lo > 1) {
      const int mid = (lo + hi) >> 1;
      if (sm.cum[mid] <= item)
        lo = mid;
      else
        hi = mid;
    }
    return lo;
  };

  if (warp == 0) {
    // ---------------------------------------------------------------- weight producer
    // (claims and decodes the units: the unit record goes to the ring for the other warps)
    if (lane == 0) {
      constexpr uint64_t kHintW = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_FIRST);
      int s = 0, ph = 0;
      for (int seq = 0;; ++seq) {
        const int u = u_next;
        // claim the next unit now: its L2 round trip overlaps this unit's loads
        if (u < n_all) u_next = atomicAdd(queue, 1);
        PfUnit cu{};
        cu.kind = kEnd;
        const CUtensorMap* wmap = nullptr;
        int wrow = 0;
        const uint8_t* sfw = nullptr;
        if (u < n_all) {
          const bool ph1 = u < n1;
          cu.kind = ph1 ? kR1 : kR2;
          // group-major, then block, then the group's items: the items of an expert larger than one
          // item read the same weight tile in consecutive units (claimed together: the repeats hit L2)
          const int nb = ph1 ? nb1 : nb2, lu = ph1 ? u : u - n1;
          cu.g = item_group(lu / nb);
          const int ci = sm.cum[cu.g + 1] - sm.cum[cu.g], loc = lu - sm.cum[cu.g] * nb;
          cu.blk = loc / ci;
          cu.ch = loc % ci;
          cu.item = sm.cum[cu.g] + cu.ch;
          cu.rb = sm.off[cu.g] + cu.ch * kN;
          {
            const int m = sm.off[cu.g + 1] - sm.off[cu.g];
            cu.rows = m <= kItemMax ? m : min(kN, m - cu.ch * kN);
          }
          cu.n = (cu.rows + 7) & ~7;
          if (ph1) {
            cu.nkt = H / 256;
            wmap = &tm.w13, wrow = cu.g * 2 * Il + cu.blk * 128;
            sfw = p.sf13 + (size_t)cu.g * nvfp4_sf_bytes(2 * Il, H) + (size_t)cu.blk * (H / 64) * 512;
          } else {
            cu.nkt = Il / 256;
            wmap = &tm.w2, wrow = cu.g * H + cu.blk * 128;
            sfw = p.sf2 + (size_t)cu.g * nvfp4_sf_bytes(H, Il) + (size_t)cu.blk * (Il / 64) * 512;
          }
        }
        sm.u[seq % kRing] = cu;
        Barrier::arrive(&sm.uready[seq % kRing]);
        for (int kt = 0;; ++kt) {
          Barrier::wait(&sm.wempty[s], ph ^ 1);
          if (cu.kind == kEnd) {
            TxBarrier::arrive_and_expect_tx(&sm.wfull[s], 0);
            break;
          }
          TxBarrier::arrive_and_expect_tx(&sm.wfull[s], kTileB + 2048);
          cute::SM90_TMA_LOAD_2D::copy(wmap, &sm.wfull[s], kHintW, sm.a[s], kt * 128, wrow);
          cute::SM90_BULK_COPY_G2S::copy(sfw + (size_t)kt * 2048, &sm.wfull[s], sm.sfa[s], 2048);
          if (++s == W) {
            s = 0;
            ph ^= 1;
          }
          if (kt + 1 == cu.nkt) break;
        }
        if (cu.kind == kEnd) break;
      }
    }
  } else if (warp == kWarpB) {
    // ---------------------------------------------------------------- token / hidden-row producer
    if (lane == 0) {
      constexpr uint64_t kHintX = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_LAST);
      int s = 0, ph = 0;
      for (int seq = 0;; ++seq) {
        Barrier::wait(&sm.uready[seq % kRing], (seq / kRing) & 1);
        const PfUnit cu = sm.u[seq % kRing];
        // slot j of a k-tile: rows [128 j, 128 j + rj) of the item (two slots beyond 128 rows)
        const int nsl = cu.rows > kN ? 2 : 1;
        const CUtensorMap* bmap[2] = {nullptr, nullptr};
        const uint8_t* sfr[2] = {nullptr, nullptr};
        int bx[2] = {0, 0};
        const unsigned* need_mask = nullptr;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          if (j >= nsl) break;
          const int rj = min(kN, cu.rows - kN * j);
          bx[j] = rj <= 32 ? 32 : (rj <= 64 ? 64 : 128);
          if (cu.kind == kR1) {
            bmap[j] = bx[j] == 32 ? &tm.x32 : (bx[j] == 64 ? &tm.x64 : &tm.x128);
            sfr[j] = p.xs + (size_t)sm.pad[cu.g] * (H / 16) + (size_t)(cu.ch + j) * (H / 64) * 512;
          } else if (cu.kind == kR2) {
            bmap[j] = bx[j] == 32 ? &tm.h32 : (bx[j] == 64 ? &tm.h64 : &tm.h128);
            sfr[j] = p.hsa + (size_t)cu.item * hs_group + j * hs_blk;
          }
        }
        if (cu.kind == kR2) need_mask = done1 + cu.item;
        unsigned have = 0;  // the hidden-row blocks seen stored (re-read only while some are missing)
        for (int kt = 0;; ++kt) {
          if (cu.kind == kEnd) {
            Barrier::wait(&sm.xempty[s], ph ^ 1);
            TxBarrier::arrive_and_expect_tx(&sm.xfull[s], 0);
            break;
          }
          if (need_mask) {
            // this k-tile's hidden features are stored (64-feature blocks: four per k-tile)
            const unsigned need = 0xFu << (4 * kt);
            if ((have & need) != need) {
              while (((have = ld_acquire_u(need_mask)) & need) != need) __nanosleep(32);
              asm volatile("fence.proxy.async.global;\n" ::: "memory");
            }
          }
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            if (j >= nsl) break;
            Barrier::wait(&sm.xempty[s], ph ^ 1);
            TxBarrier::arrive_and_expect_tx(&sm.xfull[s], bx[j] * 128 + 2048);
            cute::SM90_TMA_LOAD_2D::copy(bmap[j], &sm.xfull[s], kHintX, sm.b[s], kt * 128, cu.rb + kN * j);
            cute::SM90_BULK_COPY_G2S::copy(sfr[j] + (size_t)kt * 2048, &sm.xfull[s], sm.sfb[s], 2048);
            if (++s == X) {
              s = 0;
              ph ^= 1;
            }
          }
          if (kt + 1 == cu.nkt) break;
        }
        if (cu.kind == kEnd) break;
      }
    }
  } else if (warp == 1) {
    // ---------------------------------------------------------------- MMA issuer
    int ws = 0, wph = 0, xs = 0, xph = 0, buf = 0, aph = 0, par = 0;
    const uint64_t da0 = desc_k_sw128(smem_u32(sm.a[0])), db0 = desc_k_sw128(smem_u32(sm.b[0]));
    const uint64_t dsa0 = desc_sf_atom(smem_u32(sm.sfa[0])), dsb0 = desc_sf_atom(smem_u32(sm.sfb[0]));
    for (int seq = 0;; ++seq) {
      Barrier::wait(&sm.wfull[ws], wph);  // (the weight producer wrote the unit record before)
      const PfUnit cu = sm.u[seq % kRing];
      Barrier::wait(&sm.acce[buf], aph ^ 1);  // the epilogue has read this accumulator buffer
      __syncwarp();
      if (lane == 0) Barrier::arrive(&sm.unitf[buf]);
      if (cu.kind == kEnd) break;
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      const uint32_t acc = tmem + buf * kAccCols;
      const bool two = cu.rows > kN;  // (rows beyond 128: a second slot, a second MMA per K step)
      const uint32_t idf = uint32_t(idesc_fp4n(tmem + kSfaCol, tmem + kSfbCol, two ? kN : cu.n) >> 32);
      const uint32_t idf2 = uint32_t(idesc_fp4n(tmem + kSfaCol, tmem + kSfbCol, two ? cu.n - kN : 8) >> 32);
      // (the per-k-tile issue path is the MMA warp's critical path: the one- and two-slot loops are
      // separate straight-line loops)
      if (!two) {
        for (int kt = 0; kt < cu.nkt; ++kt) {
          if (kt > 0) Barrier::wait(&sm.wfull[ws], wph);
          Barrier::wait(&sm.xfull[xs], xph);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          const uint64_t da = da0 + (uint64_t)(ws * (kTileB >> 4));
          const uint64_t db = db0 + (uint64_t)(xs * ((kN * 128) >> 4));
          if (cute::elect_one_sync()) {
            const uint32_t sfa = tmem + kSfaCol + par * 16, sfb = tmem + kSfbCol + par * 32;
            const uint64_t dsa = dsa0 + (uint64_t)(ws * (2048 >> 4)), dsb = dsb0 + (uint64_t)(xs * (2048 >> 4));
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              sf_copy_raw(sfa + 4 * j, dsa + (uint64_t)(j * (512 >> 4)));
              sf_copy_raw(sfb + 4 * j, dsb + (uint64_t)(j * (512 >> 4)));
            }
#pragma unroll
            for (int j = 0; j < 4; ++j)
              mma_fp4_raw(da + 2 * j, db + 2 * j, acc, (kt > 0 || j > 0) ? 1u : 0u, idf, sfa + 4 * j,
                          sfb + 4 * j);
          }
          __syncwarp();
          par ^= 1;
          // the stages are free once these MMAs complete
          cutlass::arch::umma_arrive(&sm.wempty[ws]);
          cutlass::arch::umma_arrive(&sm.xempty[xs]);
          if (++ws == W) {
            ws = 0;
            wph ^= 1;
          }
          if (++xs == X) {
            xs = 0;
            xph ^= 1;
          }
        }
      } else {
        for (int kt = 0; kt < cu.nkt; ++kt) {
          if (kt > 0) Barrier::wait(&sm.wfull[ws], wph);
          const int xs2 = xs + 1 == X ? 0 : xs + 1, xph2 = xs + 1 == X ? xph ^ 1 : xph;
          Barrier::wait(&sm.xfull[xs], xph);
          Barrier::wait(&sm.xfull[xs2], xph2);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          const uint64_t da = da0 + (uint64_t)(ws * (kTileB >> 4));
          const uint64_t db = db0 + (uint64_t)(xs * ((kN * 128) >> 4));
          const uint64_t db2 = db0 + (uint64_t)(xs2 * ((kN * 128) >> 4));
          if (cute::elect_one_sync()) {
            const uint32_t sfa = tmem + kSfaCol + par * 16, sfb = tmem + kSfbCol + par * 32;
            const uint64_t dsa = dsa0 + (uint64_t)(ws * (2048 >> 4)), dsb = dsb0 + (uint64_t)(xs * (2048 >> 4));
            const uint64_t dsb2 = dsb0 + (uint64_t)(xs2 * (2048 >> 4));
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              sf_copy_raw(sfa + 4 * j, dsa + (uint64_t)(j * (512 >> 4)));
              sf_copy_raw(sfb + 4 * j, dsb + (uint64_t)(j * (512 >> 4)));
              sf_copy_raw(sfb + 16 + 4 * j, dsb2 + (uint64_t)(j * (512 >> 4)));
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              mma_fp4_raw(da + 2 * j, db + 2 * j, acc, (kt > 0 || j > 0) ? 1u : 0u, idf, sfa + 4 * j,
                          sfb + 4 * j);
              mma_fp4_raw(da + 2 * j, db2 + 2 * j, acc + kN, (kt > 0 || j > 0) ? 1u : 0u, idf2,
                          sfa + 4 * j, sfb + 16 + 4 * j);
            }
          }
          __syncwarp();
          par ^= 1;
          cutlass::arch::umma_arrive(&sm.wempty[ws]);
          cutlass::arch::umma_arrive(&sm.xempty[xs]);
          cutlass::arch::umma_arrive(&sm.xempty[xs2]);
          if (++ws == W) {
            ws = 0;
            wph ^= 1;
          }
          xs = xs2 + 1 == X ? 0 : xs2 + 1;
          xph = xs2 + 1 == X ? xph2 ^ 1 : xph2;
        }
      }
      cutlass::arch::umma_arrive(&sm.accf[buf]);
      if (++buf == 2) {
        buf = 0;
        aph ^= 1;
      }
    }
  } else {
    // ---------------------------------------------------------------- epilogue (warps 2-9)
    // two warps per tensor-memory lane quadrant (q = warp % 4: tile rows 32 q .. 32 q + 31), splitting
    // the columns in alternating 16-column groups (half = which of the pair)
    const int ew = warp - 2, q = warp & 3, half = ew >> 2;
    const int row = 32 * q + lane, tid_e = 32 * ew + lane;
    int buf = 0, aph = 0;
    const float sfs = 1.f / p.gs2;
    for (int seq = 0;; ++seq) {
      Barrier::wait(&sm.unitf[buf], aph);
      const PfUnit cu = sm.u[seq % kRing];
      if (cu.kind == kEnd) break;
      Barrier::wait(&sm.accf[buf], aph);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      const int rows = cu.rows, rb = cu.rb;
      const uint32_t tbase = tmem + ((uint32_t)(32 * q) << 16) + buf * kAccCols;
      auto release = [&] {  // the accumulator buffer has been read
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        __syncwarp();
        if (lane == 0) Barrier::arrive(&sm.acce[buf]);
        if (++buf == 2) {
          buf = 0;
          aph ^= 1;
        }
      };
      if (cu.kind == kR2) {
        // down projection: column i = hidden row rb + i, output feature 128 blk + row, bf16. Per 16
        // columns the warp's 32 features x 16 rows pass through its 1 KB staging tile in sm.gu (lane
        // pairs pack two features per word: even lanes rows 0..7, odd lanes rows 8..15), then every
        // thread stores 16 bytes (8 features of a row): 2 vector stores per thread instead of 16
        // two-byte stores
        const float al = __ldg(p.alpha2 + cu.g);
        uint32_t* const stg = reinterpret_cast<uint32_t*>(&sm.gu[0][0]) + ew * 256;
        const bool odd = lane & 1;
        __nv_bfloat16* const yb = p.y + (size_t)rb * H + cu.blk * 128 + 32 * q;
        for (int c0 = 16 * half; c0 < rows; c0 += 32) {
          float v[16];
          tmem_ld16(tbase + c0, v);
          tmem_wait_ld();
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const float mine = odd ? v[8 + j] : v[j];
            const float other = __shfl_xor_sync(0xffffffffu, odd ? v[j] : v[8 + j], 1);
            const __nv_bfloat162 w2 = __floats2bfloat162_rn((odd ? other : mine) * al,
                                                            (odd ? mine : other) * al);
            const int r = odd ? 8 + j : j;
            // (rows 8..15 swapped pairwise: the even and odd lanes' words in opposite bank halves)
            stg[(r ^ (r >> 3)) * 16 + (lane >> 1)] = *reinterpret_cast<const uint32_t*>(&w2);
          }
          __syncwarp();
#pragma unroll
          for (int k = 0; k < 2; ++k) {
            const int slot = (lane >> 2) + 8 * k, part = lane & 3, r = slot ^ (slot >> 3);
            const uint4 w = *reinterpret_cast<const uint4*>(stg + slot * 16 + part * 4);
            if (c0 + r < rows) *reinterpret_cast<uint4*>(yb + (size_t)(c0 + r) * H + part * 8) = w;
          }
          __syncwarp();
        }
        epi_sync();  // (every warp's staging reads are done: sm.gu is free for an R1 unit's staging)
        release();
        continue;
      }
      // R1: per pass of kGuCols chunk rows, the gate (tile rows 0..63) / up (64..127) outputs of 64
      // features staged (bf16, output scale applied); then per (row i, 16-feature block b) item:
      // warp-uniform b, 32 consecutive rows per warp (conflict-free staging reads), swiglu_quant_nvfp4's
      // arithmetic, 8 bytes of the hidden row and its scale byte
      {
        const float al = __ldg(p.alpha13 + cu.g);
        uint8_t* hsg = p.hsa + (size_t)cu.item * hs_group;
        for (int p0 = 0; p0 < rows; p0 += kGuCols) {
          const int pr = min(kGuCols, rows - p0);
          if (p0 > 0) epi_sync();  // (the previous pass's staging reads are done)
          for (int c0 = 16 * half; c0 < pr; c0 += 32) {
            float v[16];
            tmem_ld16(tbase + p0 + c0, v);
            tmem_wait_ld();
            __align__(16) __nv_bfloat16 hb[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) hb[i] = __float2bfloat16(v[i] * al);
            *reinterpret_cast<uint4*>(&sm.gu[row][c0]) = *reinterpret_cast<const uint4*>(hb);
            *reinterpret_cast<uint4*>(&sm.gu[row][c0 + 8]) = *reinterpret_cast<const uint4*>(hb + 8);
          }
          if (p0 + kGuCols >= rows) release();  // (the last pass has read the accumulator)
          epi_sync();
          const int nit = 4 * ((pr + 31) & ~31);
          for (int it = tid_e; it < nit; it += 32 * kEpiWarps) {
            const int b = (it >> 5) & 3, il = (it >> 7) * 32 + (it & 31);
            if (il >= pr) continue;
            const int i = p0 + il;
            float h[16];
            float amax = 0.f;
#pragma unroll
            for (int e = 0; e < 16; ++e) {
              const int fl = 16 * b + e;
              h[e] = swiglu_fast(__bfloat162float(sm.gu[fl][il]), __bfloat162float(sm.gu[64 + fl][il]),
                                 p.limit);
              amax = fmaxf(amax, fabsf(h[e]));
            }
            float sfv = sfs * (amax * rcp_approx_f(6.f));
            const __nv_fp8_e4m3 s8(sfv);
            sfv = static_cast<float>(s8);
            const float osc = amax != 0.f ? rcp_approx_f(sfv * rcp_approx_f(sfs)) : 0.f;
#pragma unroll
            for (int e = 0; e < 16; ++e) h[e] *= osc;
            *reinterpret_cast<uint2*>(p.hq + (size_t)(rb + i) * (Il / 2) + cu.blk * 32 + b * 8) =
                make_uint2(e2m1x8(h), e2m1x8(h + 8));
            hsg[nvfp4_sf_off(i, 4 * cu.blk + b, Il)] = s8.__x;
          }
        }
      }
      // this block of the hidden rows is stored: every thread orders its stores for the async proxy
      // (the R2 units' TMA loads), the barrier gathers them, one release publishes the block (a
      // __threadfence in every thread would wait for its own stores)
      asm volatile("fence.proxy.async.global;\n" ::: "memory");
      epi_sync();  // (also: sm.gu is free for the next unit)
      if (warp == 2 && lane == 0) {
        __threadfence();
        atomicOr(done1 + cu.item, 1u << cu.blk);
      }
    }
  }
  __syncthreads();
  if (warp == 1) {
    tmem_alloc.release_allocation_lock();
    tmem_alloc.free(tmem, kTmemCols);
  }
  // the last CTA out restores the counters for the next launch
  if (tid == 0) {
    __threadfence();
    if (atomicAdd(finished, 1) == (int)gridDim.x - 1) {
      for (int i = 0; i < n_items; ++i) done1[i] = 0;
      *queue = 0;
      *finished = 0;
      __threadfence();
    }
  }
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

const CUtensorMap& cached_map(const void* base, int64_t rows, int64_t cols, int box_rows) {
  static std::mutex mu;
  static std::map<std::tuple<const void*, int64_t, int64_t, int>, CUtensorMap> maps;
  std::lock_guard<std::mutex> lk(mu);
  const auto key = std::make_tuple(base, rows, cols, box_rows);
  auto it = maps.find(key);
  if (it == maps.end()) it = maps.emplace(key, make_map(base, rows, cols, box_rows)).first;
  return it->second;
}
}  // namespace

void moe_fp4_prefill(const MoePrefillArgs& a, int ctas, cudaStream_t st) {
  CHECK(a.H % 256 == 0 && a.Il % 256 == 0, "moe_fp4_prefill: H, Il multiples of 256");
  CHECK(a.Il / 64 <= 32, "moe_fp4_prefill: at most 32 phase-1 blocks per item (bitmask)");
  CHECK(a.G >= 1 && a.G <= kMoePfMaxGroups, "moe_fp4_prefill: groups");
  const int max_items = moe_fp4_prefill_items(a.rows > 0 ? a.rows : a.h_rows, a.G);
  CHECK(max_items <= kMoePfMaxItems && a.hs_items >= max_items,
        "moe_fp4_prefill: work items exceed the counters / hidden-row scale blocks");
  // ring depths: 6 weight + 5 row-tile stages
  constexpr int W = 6, X = 5;
  const int smem = sizeof(PfSmem<W, X>) + 1024;
  static bool configured[64] = {};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  if (!configured[dev]) {
    CUDA_CHECK(cudaFuncSetAttribute(moe_fp4_prefill_kernel<W, X>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem));
    configured[dev] = true;
  }
  PfMaps m;
  m.w13 = cached_map(a.w13, (int64_t)a.G * 2 * a.Il, a.H / 2, 128);
  m.w2 = cached_map(a.w2, (int64_t)a.G * a.H, a.Il / 2, 128);
  m.x32 = cached_map(a.xq, a.x_rows, a.H / 2, 32);
  m.x64 = cached_map(a.xq, a.x_rows, a.H / 2, 64);
  m.x128 = cached_map(a.xq, a.x_rows, a.H / 2, 128);
  m.h32 = cached_map(a.hq, a.h_rows, a.Il / 2, 32);
  m.h64 = cached_map(a.hq, a.h_rows, a.Il / 2, 64);
  m.h128 = cached_map(a.hq, a.h_rows, a.Il / 2, 128);
  launch_pdl(moe_fp4_prefill_kernel<W, X>, dim3(ctas), dim3(kThreads), smem, st, a, m);
}

}  // namespace k
}  // namespace glm
