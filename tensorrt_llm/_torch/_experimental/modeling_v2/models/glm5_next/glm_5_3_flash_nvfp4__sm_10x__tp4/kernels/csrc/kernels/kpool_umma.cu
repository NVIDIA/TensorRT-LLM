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
// Indexer pool scoring on the SM100 tensor cores (tcgen05.mma into tensor memory, TMA loads).
//
// For a tile of up to 8 consecutive tokens of one sequence and that sequence's pools:
//   score[t][p] = sum_h w[t][h] / sqrt(H) * relu(q[t][h] . k[p] / sqrt(D))
// computed as a GEMM with the pools as rows (M = 128 per MMA, A = pool keys) and the (token, head)
// pairs as columns (N = 8 x 32 = 256, B = queries), K = D = 128. Each tensor-memory lane (one
// pool) then holds all 256 dot products of its pool, so the ReLU and the weighted head sum stay in
// one thread's registers. Warp roles: 0-7 = epilogue (warp w reads TMEM lanes 32 (w % 4) .. + 31
// for 4 of the 8 tokens, whose head weights it keeps in registers: 232 per thread through warpgroup
// register reallocation; two warps per scheduler hide the TMEM load latency), 8 = TMA producer
// (the queries once, then a 5-stage ring of 128-pool key tiles: 8 cache pages x 2 swizzle atoms of
// 64 dims, lane p loading page p), 9 = MMA issuer (two TMEM accumulators of 256 columns, so the
// MMAs of a tile overlap the epilogue of the previous one; N = 128 for tiles of <= 4 tokens).
// Shared memory operands are K-major with the 128-byte swizzle: [atom][row][128 B].
#include <cuda.h>
#include <cudaTypedefs.h>

#include <map>
#include <mutex>
#include <tuple>

#include "cute/arch/copy_sm100.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/arch/mma_sm100_desc.hpp"
#include "cute/arch/mma_sm100_umma.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "mla.h"
#include "pdl.cuh"

namespace glm {
namespace k {

namespace {
constexpr int kUT = 8;                  // tokens per tile
constexpr int kUN = kUT * kIdxHeads;    // MMA N: (token, head) pairs
constexpr int kUP = 128;                // MMA M: pools per tile
constexpr int kUStages = 5;             // key-tile ring
constexpr int kAtom = 128;              // bytes per swizzle-atom row (64 bf16)
constexpr int kKTile = 2 * kUP * kAtom;  // 32 KB
constexpr int kQTile = 2 * kUN * kAtom;  // 64 KB
constexpr int kUThreads = 384;  // 8 epilogue warps, then producer, MMA and two more producer warps
constexpr int kProducerWarp = 8, kMmaWarp = 9;
constexpr int kRA = 4;  // block-table read-ahead of the producer (key tiles)

struct UmmaSmem {
  uint8_t q[kQTile];
  uint8_t k[kUStages][kKTile];
  uint64_t full[kUStages], empty[kUStages], accf[2], acce[2], qfull;
  float w[kUT][kIdxHeads];
  int ncand[kUT];
  uint32_t tmem;
};

using Barrier = cutlass::arch::ClusterBarrier;
using TxBarrier = cutlass::arch::ClusterTransactionBarrier;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// registers whose uses must stay after a preceding tmem_wait (an empty volatile asm after it)
__device__ __forceinline__ void tmem_dep(uint32_t (&d)[32]) {
  asm volatile(""
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]),
                 "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]),
                 "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]),
                 "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]),
                 "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31])
               :
               : "memory");
}

// tcgen05.wait::ld that the loaded registers depend on (their uses cannot move above it)
__device__ __forceinline__ void tmem_wait(uint32_t (&d)[32]) {
  asm volatile(
      "tcgen05.wait::ld.sync.aligned;\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]),
        "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]),
        "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]),
        "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]),
        "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31])
      :
      : "memory");
}

// UMMA shared-memory descriptor of a K-major operand with the 128-byte swizzle (rows 128 B
// apart, 8-row groups 1024 B apart), starting at smem_addr
__device__ __forceinline__ uint64_t desc_k_sw128(uint32_t smem_addr) {
  cute::UMMA::SmemDescriptor d;
  d.start_address_ = static_cast<uint16_t>((smem_addr >> 4) & 0x3FFF);
  d.leading_byte_offset_ = 1;
  d.stride_byte_offset_ = 1024 >> 4;
  d.version_ = 1;
  d.base_offset_ = 0;
  d.lbo_mode_ = 0;
  d.layout_type_ = static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_128B);
  return d.desc_;
}
}  // namespace

// PQ (launches with meta.pre): the barriers and the TMEM allocation before the PDL wait, and the key tiles
// whose pools all precede the pool of the tile's first token (complete before this step: nothing this step writes) issued
// by the producer then too, in ring order; after the wait the epilogue warps load the query tile by 16-byte cp.async
// into the 128-byte-swizzled layout the tensor copy writes (the TMA engine still streams the keys: a tensor copy of the
// query queued behind them) together with the head weights, and arrive on its barrier after a proxy fence (tiles of more
// than 4 tokens: no keys before the wait, the query by the producer's tensor copy and the producer split as in the
// other instance). The same data in the same shared-memory bytes and the same MMAs: bitwise
template <bool PQ>
__global__ void __launch_bounds__(kUThreads, 1) kpool_score_umma_kernel(
    float* __restrict__ scores, const float* __restrict__ w, int ld_w, const MlaTokMeta meta,
    const int2* __restrict__ tiles, int row0, int stride, const __grid_constant__ CUtensorMap tm_q,
    const __grid_constant__ CUtensorMap tm_k, const __grid_constant__ CUtensorMap tm_q2,
    const bf16* __restrict__ q_ptr, int64_t q_rows) {
  pdl_trigger();
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  // 1024-byte aligned for the swizzle atoms (offset from the array, so accesses stay LDS / STS)
  const uint32_t pad = (1024u - (smem_u32(smem_raw) & 1023u)) & 1023u;
  UmmaSmem& sm = *reinterpret_cast<UmmaSmem*>(smem_raw + pad);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int2 tile = tiles[blockIdx.x];  // (tiles: uploaded or static)
  const int t0 = tile.x, nt = min(tile.y, kUT);
  // meta.pre: the token positions before the PDL wait, and the block-table entries of this
  // CTA's pool tiles into L2
  int ra[kRA];  // (producer warp: the block-table entries of its next kRA key tiles)
#pragma unroll
  for (int r = 0; r < kRA; ++r) ra[r] = 0;
  if (meta.pre) {
    if (tid < kUT) sm.ncand[tid] = tid < nt ? (meta.positions[t0 + tid] + 1) / kKpool : 0;
    __syncthreads();
    int nc = 0;
#pragma unroll
    for (int j = 0; j < kUT; ++j) nc = max(nc, sm.ncand[j]);
    const int64_t* bt = meta.block_tables + (size_t)meta.token_seq[t0] * meta.max_blocks;
    for (int pt = blockIdx.y + gridDim.y * tid; pt * kUP < nc; pt += gridDim.y * kUThreads)
      prefetch_l2(bt + pt * (kUP / kPoolsPerPage));  // (a tile's 4 entries: one line)
    if (warp >= kProducerWarp && warp != kMmaWarp) {  // producers: their first kRA key tiles' entries
#pragma unroll
      for (int r = 0; r < kRA; ++r) {
        const int j = ((int)blockIdx.y + r * (int)gridDim.y) * kUP + lane * kPoolsPerPage;
        ra[r] = lane < kUP / kPoolsPerPage && j < nc ? (int)(bt[j / kPoolsPerPage] * meta.idx_page_rows) : 0;
      }
    }
  }
  constexpr uint64_t kHint = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
  cute::TMEM::Allocator1Sm tmem_alloc{};
  auto setup = [&]() {  // barriers + TMEM (no predecessor data)
    if (tid == 0) {
      for (int s = 0; s < kUStages; ++s) {
        Barrier::init(&sm.full[s], 1);
        Barrier::init(&sm.empty[s], 1);
      }
      for (int a = 0; a < 2; ++a) {
        Barrier::init(&sm.accf[a], 1);
        Barrier::init(&sm.acce[a], 8 * 32);  // every epilogue thread
      }
      Barrier::init(&sm.qfull, PQ && nt <= kUT / 2 ? 8 * 32 : 1);  // (PQ decode tiles: every epilogue thread's copies)
      cutlass::arch::fence_barrier_init();
    }
    if (warp == kMmaWarp) tmem_alloc.allocate(2 * kUN, &sm.tmem);
  };
  int max_nc = 0, kp_pre = 0;  // (PQ, producer warp: key tiles issued before the wait)
  bool live = true;
  if constexpr (PQ) {  // (meta.pre: sm.ncand known)
#pragma unroll
    for (int j = 0; j < kUT; ++j) max_nc = max(max_nc, sm.ncand[j]);
    const int npt0 = (max_nc + kUP - 1) / kUP;
    live = max_nc > kSelPools && (int)blockIdx.y < npt0;  // (CTA-uniform)
    if (live) {
      setup();
      __syncthreads();  // (barriers initialized before the first copy; the TMEM address in sm.tmem)
      if (warp == kProducerWarp && nt <= kUT / 2) {  // (decode tiles: one producer warp)
        const int old_lim = meta.positions[t0] / kKpool;  // pools below: complete before this step
#pragma unroll
        for (int r = 0; r < (kRA < kUStages ? kRA : kUStages); ++r) {
          const int pt = (int)blockIdx.y + r * (int)gridDim.y;
          if (pt >= npt0 || (pt + 1) * kUP > old_lim) break;
          if (lane == 0) TxBarrier::arrive_and_expect_tx(&sm.full[r], kKTile);
          __syncwarp();
          if (lane < kUP / kPoolsPerPage) {
            uint8_t* dst = sm.k[r] + lane * (kPoolsPerPage * kAtom);
            for (int a = 0; a < 2; ++a)
              cute::SM90_TMA_LOAD_2D::copy(&tm_k, &sm.full[r], kHint, dst + a * (kUP * kAtom), a * 64, ra[r]);
          }
          ++kp_pre;
        }
      }
    }
  }
  pdl_wait();
  if constexpr (PQ) {
    if (!live) return;
  } else {
    if (!meta.pre && tid < kUT) sm.ncand[tid] = tid < nt ? (meta.positions[t0 + tid] + 1) / kKpool : 0;
    __syncthreads();
#pragma unroll
    for (int j = 0; j < kUT; ++j) max_nc = max(max_nc, sm.ncand[j]);
    if (max_nc <= kSelPools) return;  // no sparse token in the tile
  }
  const int npt = (max_nc + kUP - 1) / kUP;
  if constexpr (!PQ)
    if ((int)blockIdx.y >= npt) return;
  const int b = meta.token_seq[t0];
  if constexpr (PQ) {
    if (warp < kProducerWarp && nt <= kUT / 2) {
      // the query rows (tiles of <= 4 tokens: 128 (token, head) rows) of both 64-column atoms; row r's 16-byte chunk c at
      // r * 128 + 16 (c ^ r % 8), as the 128-byte swizzle stores it (rows past q_rows: zeros, as the tensor copy's
      // out-of-bounds fill)
      constexpr int qr = kUN / 2;
      const uint8_t* qsrc = reinterpret_cast<const uint8_t*>(q_ptr);
      for (int i = tid; i < 2 * qr * 8; i += 8 * 32) {
        const int a = i / (qr * 8), r = (i / 8) % qr, c = i % 8;
        const int64_t row = (int64_t)t0 * kIdxHeads + r;
        const bool ok = row < q_rows;
        const uint32_t dst = smem_u32(sm.q + a * (kUN * kAtom) + r * kAtom + ((c ^ (r & 7)) << 4));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst),
                     "l"(qsrc + (ok ? row * (kIdxDim * 2) + a * 128 + c * 16 : 0)), "r"(ok ? 16 : 0)
                     : "memory");
      }
    }
  }
  for (int i = tid; i < kUT * kIdxHeads; i += kUThreads) {
    const int j = i / kIdxHeads, h = i % kIdxHeads;
    // head weight / sqrt(H), with the 1 / sqrt(D) of the dot products (relu commutes with it)
    sm.w[j][h] = j < nt ? w[(size_t)(t0 + j) * ld_w + h] * (rsqrtf((float)kIdxHeads) *
                                                           rsqrtf((float)kIdxDim))
                        : 0.f;
  }
  if constexpr (PQ) {
    static_assert(kUT * kIdxHeads == 8 * 32, "PQ: the head weights loaded by the epilogue warps");
    if (warp < kProducerWarp) {
      if (nt <= kUT / 2) {
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // (the query bytes for the MMA's async proxy)
        Barrier::arrive(&sm.qfull);
      }
      asm volatile("bar.sync 1, 256;" ::: "memory");  // (every epilogue warp's head weights in sm.w)
    }
  } else {
    setup();
    __syncthreads();
  }
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const uint32_t tmem = sm.tmem;

  // warpgroup register reallocation (once per warpgroup, before its warps split into roles):
  // 40 for warps 8-11, 232 for the epilogue warpgroups (weights in registers)
  if (warp >= kProducerWarp) {
    cutlass::arch::warpgroup_reg_dealloc<40>();
    // tiles of more than 4 tokens (prefill): the pages split over three producer warps; decode
    // tiles (<= 4 tokens, few key tiles per CTA) keep the one producer warp
    const bool split = nt > kUT / 2;
    if (warp == kProducerWarp || (split && warp != kMmaWarp)) {
      // producers (warp 8; with split also 10, 11): lane p < 8 loads cache page p of each 128-pool
      // tile (one warp issuing a tile's 16 TMA copies, whose per-lane coordinates serialize, would
      // leave the MMA waiting for data at prefill sizes)
      const int pw = warp == kProducerWarp ? 0 : warp - kMmaWarp;  // 0, 1, 2
      const bool mine = split ? lane >= 3 * pw && lane < min(kUP / kPoolsPerPage, 3 * pw + 3)
                              : lane < kUP / kPoolsPerPage;
      if ((!PQ || split) && warp == kProducerWarp && lane == 0) {  // (tiles of <= 4 tokens: their 128 (token, head) rows)
        const bool half = nt <= kUT / 2;
        TxBarrier::arrive_and_expect_tx(&sm.qfull, half ? kQTile / 2 : kQTile);
        for (int a = 0; a < 2; ++a)
          cute::SM90_TMA_LOAD_2D::copy(half ? &tm_q2 : &tm_q, &sm.qfull, kHint,
                                       sm.q + a * (kUN * kAtom), a * 64, t0 * kIdxHeads);
      }
      constexpr int kPages = kUP / kPoolsPerPage;
      const int64_t* bt = meta.block_tables + (size_t)b * meta.max_blocks;
      // first pool-key row of the lane's page in the INDEX_KEY pool (pools past the tile's last
      // read the first page's rows, whose scores are never written)
      auto page_row = [&](int pt) -> int {
        const int j = pt * kUP + lane * kPoolsPerPage;
        return mine && j < max_nc ? (int)(bt[j / kPoolsPerPage] * meta.idx_page_rows) : 0;
      };
      // the block-table entries kRA tiles ahead, in registers (slot r holds tile pt0 + r G): with one
      // tile of read-ahead every key tile's TMA would wait for its entry's load, one memory latency per
      // tile; the TMA of a tile needs its entry loaded kRA tiles earlier
      const int G = gridDim.y;
      if (!meta.pre) {  // (meta.pre: loaded before the PDL wait)
#pragma unroll
        for (int r = 0; r < kRA; ++r) ra[r] = page_row(blockIdx.y + r * G);
      }
      int s = 0, ph = 0;
      for (int pt0 = blockIdx.y; pt0 < npt; pt0 += kRA * G) {
#pragma unroll
        for (int r = 0; r < kRA; ++r) {
          const int pt = pt0 + r * G;
          if (pt < npt) {
            const int row = ra[r];
            ra[r] = page_row(pt + kRA * G);
            if (PQ && pt0 == (int)blockIdx.y && r < kp_pre) {  // (PQ: issued before the PDL wait)
              if (++s == kUStages) {
                s = 0;
                ph ^= 1;
              }
              continue;
            }
            Barrier::wait(&sm.empty[s], ph ^ 1);
            if (warp == kProducerWarp && lane == 0)
              TxBarrier::arrive_and_expect_tx(&sm.full[s], kKTile);  // (the other warps' copies
            __syncwarp();                                             //  may land first: tx < 0)
            if (mine) {
              uint8_t* dst = sm.k[s] + lane * (kPoolsPerPage * kAtom);
              for (int a = 0; a < 2; ++a)
                cute::SM90_TMA_LOAD_2D::copy(&tm_k, &sm.full[s], kHint, dst + a * (kUP * kAtom),
                                             a * 64, row);
            }
            if (++s == kUStages) {
              s = 0;
              ph ^= 1;
            }
          }
        }
      }
    } else if (warp == kMmaWarp) {  // MMA issuer (the whole warp; one elected thread issues)
      // tiles of <= 4 tokens (decode with MTP 3) need only half the columns
      const uint64_t idesc = nt <= kUT / 2
                                 ? cute::UMMA::make_runtime_instr_desc<
                                       cute::bfloat16_t, cute::bfloat16_t, float, kUP, kUN / 2,
                                       cute::UMMA::Major::K, cute::UMMA::Major::K>()
                                 : cute::UMMA::make_runtime_instr_desc<
                                       cute::bfloat16_t, cute::bfloat16_t, float, kUP, kUN,
                                       cute::UMMA::Major::K, cute::UMMA::Major::K>();
      Barrier::wait(&sm.qfull, 0);
      const uint32_t q_addr = smem_u32(sm.q);
      int s = 0, ph = 0, buf = 0, aph = 0;
      for (int pt = blockIdx.y; pt < npt; pt += gridDim.y) {
        Barrier::wait(&sm.acce[buf], aph ^ 1);  // the epilogue has drained this accumulator
        Barrier::wait(&sm.full[s], ph);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        const uint32_t k_addr = smem_u32(sm.k[s]);
  #pragma unroll
        for (int kb = 0; kb < kIdxDim / 16; ++kb) {
          const uint64_t da = desc_k_sw128(k_addr + (kb / 4) * (kUP * kAtom) + (kb % 4) * 32);
          const uint64_t db = desc_k_sw128(q_addr + (kb / 4) * (kUN * kAtom) + (kb % 4) * 32);
          cute::SM100_MMA_F16BF16_SS<cute::bfloat16_t, cute::bfloat16_t, float, kUP, kUN,
                                     cute::UMMA::Major::K, cute::UMMA::Major::K>::fma(
              da, db, tmem + buf * kUN, kb > 0 ? 1u : 0u, idesc);
        }
        cutlass::arch::umma_arrive(&sm.empty[s]);  // the key stage is free once these complete
        cutlass::arch::umma_arrive(&sm.accf[buf]);
        if (++s == kUStages) {
          s = 0;
          ph ^= 1;
        }
        if (++buf == 2) {
          buf = 0;
          aph ^= 1;
        }
      }
    }
  } else if (nt <= kUT / 2) {
    // epilogue of tiles of <= 4 tokens (decode with MTP 3): warp w reads TMEM lanes 32 (w % 4) ..
    // + 31 (one pool per thread) for tokens {2 h, 2 h + 1}, h = w / 4, both tokens' rows loaded
    // before one wait (four tokens on one warp per scheduler exposed each load's latency in turn)
    cutlass::arch::warpgroup_reg_alloc<232>();
    const int quarter = warp & 3, tb = (warp >> 2) * 2;
    const int nw = max(0, min(nt, tb + 2) - tb);  // the warp's tokens
    const int prow = quarter * 32 + lane;
    int ncr[2];
    float2 wr[2][kIdxHeads / 2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      ncr[i] = sm.ncand[tb + i];
#pragma unroll
      for (int h = 0; h < kIdxHeads / 2; ++h)
        wr[i][h] = reinterpret_cast<const float2*>(sm.w[tb + i])[h];
    }
    int buf = 0, aph = 0;
    for (int pt = blockIdx.y; pt < npt; pt += gridDim.y) {
      Barrier::wait(&sm.accf[buf], aph);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      const int p = pt * kUP + prow;
      const uint32_t taddr = tmem + ((uint32_t)(quarter * 32) << 16) + buf * kUN;
      uint32_t d[2][32];
      if (nw > 0)
        cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b32x::copy(
            taddr + tb * kIdxHeads, d[0][0], d[0][1], d[0][2], d[0][3], d[0][4], d[0][5], d[0][6],
            d[0][7], d[0][8], d[0][9], d[0][10], d[0][11], d[0][12], d[0][13], d[0][14], d[0][15],
            d[0][16], d[0][17], d[0][18], d[0][19], d[0][20], d[0][21], d[0][22], d[0][23], d[0][24],
            d[0][25], d[0][26], d[0][27], d[0][28], d[0][29], d[0][30], d[0][31]);
      if (nw > 1)
        cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b32x::copy(
            taddr + (tb + 1) * kIdxHeads, d[1][0], d[1][1], d[1][2], d[1][3], d[1][4], d[1][5],
            d[1][6], d[1][7], d[1][8], d[1][9], d[1][10], d[1][11], d[1][12], d[1][13], d[1][14],
            d[1][15], d[1][16], d[1][17], d[1][18], d[1][19], d[1][20], d[1][21], d[1][22], d[1][23],
            d[1][24], d[1][25], d[1][26], d[1][27], d[1][28], d[1][29], d[1][30], d[1][31]);
      if (nw > 0) {
        tmem_wait(d[0]);
        tmem_dep(d[1]);
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        if (i < nw && ncr[i] > kSelPools && p < ncr[i]) {
          float2 a0 = make_float2(0.f, 0.f), a1 = a0;
#pragma unroll
          for (int h = 0; h < kIdxHeads; h += 4) {
            const float2 x0 = make_float2(fmaxf(__uint_as_float(d[i][h]), 0.f),
                                          fmaxf(__uint_as_float(d[i][h + 1]), 0.f));
            const float2 x1 = make_float2(fmaxf(__uint_as_float(d[i][h + 2]), 0.f),
                                          fmaxf(__uint_as_float(d[i][h + 3]), 0.f));
            a0 = __ffma2_rn(x0, wr[i][h / 2], a0);
            a1 = __ffma2_rn(x1, wr[i][h / 2 + 1], a1);
          }
          scores[(size_t)(t0 + tb + i - row0) * stride + p] = (a0.x + a1.x) + (a0.y + a1.y);
        }
      }
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      Barrier::arrive(&sm.acce[buf]);
      if (++buf == 2) {
        buf = 0;
        aph ^= 1;
      }
    }
  } else {  // epilogue: warp w reads TMEM lanes 32 (w % 4) .. + 31 (one pool per thread), tokens
            // [4 h, 4 h + 4) with h = w / 4
    cutlass::arch::warpgroup_reg_alloc<232>();
    const int quarter = warp & 3, tb = (warp >> 2) * (kUT / 2);
    const int nw = max(0, min(nt, tb + kUT / 2) - tb);  // the warp's tokens
    const int prow = quarter * 32 + lane;
    int ncr[kUT / 2];
    float2 wr[kUT / 2][kIdxHeads / 2];  // the warp's token weights, kept for the CTA's lifetime
#pragma unroll
    for (int i = 0; i < kUT / 2; ++i) {
      ncr[i] = sm.ncand[tb + i];
#pragma unroll
      for (int h = 0; h < kIdxHeads / 2; ++h)
        wr[i][h] = reinterpret_cast<const float2*>(sm.w[tb + i])[h];
    }
    int buf = 0, aph = 0;
    for (int pt = blockIdx.y; pt < npt; pt += gridDim.y) {
      Barrier::wait(&sm.accf[buf], aph);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      const int p = pt * kUP + prow;
      const uint32_t taddr = tmem + ((uint32_t)(quarter * 32) << 16) + buf * kUN;
      // per token: its 32 heads from TMEM, then sum_h w'_h relu(x_h) with packed FMAs into four
      // partial sums (short dependency chains). nw is warp-uniform, so the waits stay converged;
      // the other epilogue warp of the scheduler computes while this one waits for its load.
#pragma unroll
      for (int i = 0; i < kUT / 2; ++i) {
        if (i >= nw) break;
        uint32_t d[32];
        cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b32x::copy(
            taddr + (tb + i) * kIdxHeads, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8],
            d[9], d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19], d[20],
            d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31]);
        tmem_wait(d);
        if (ncr[i] > kSelPools && p < ncr[i]) {
          float2 a0 = make_float2(0.f, 0.f), a1 = a0;
#pragma unroll
          for (int h = 0; h < kIdxHeads; h += 4) {
            const float2 x0 = make_float2(fmaxf(__uint_as_float(d[h]), 0.f),
                                          fmaxf(__uint_as_float(d[h + 1]), 0.f));
            const float2 x1 = make_float2(fmaxf(__uint_as_float(d[h + 2]), 0.f),
                                          fmaxf(__uint_as_float(d[h + 3]), 0.f));
            a0 = __ffma2_rn(x0, wr[i][h / 2], a0);
            a1 = __ffma2_rn(x1, wr[i][h / 2 + 1], a1);
          }
          scores[(size_t)(t0 + tb + i - row0) * stride + p] = (a0.x + a1.x) + (a0.y + a1.y);
        }
      }
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      Barrier::arrive(&sm.acce[buf]);
      if (++buf == 2) {
        buf = 0;
        aph ^= 1;
      }
    }
  }
  __syncthreads();
  if (warp == kMmaWarp) {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    tmem_alloc.release_allocation_lock();
    tmem_alloc.free(tmem, 2 * kUN);
  }
}

namespace {
// 2-D bf16 tensor map: rows of 128 values, boxes of 64 values x box_rows rows, 128-byte swizzle
CUtensorMap make_tensor_map(const void* base, int64_t rows, int box_rows) {
  static PFN_cuTensorMapEncodeTiled_v12000 encode = [] {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000,
                                                cudaEnableDefault, &q));
    CHECK(fn != nullptr && q == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled not found");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
  }();
  CUtensorMap m;
  cuuint64_t dims[2] = {(cuuint64_t)kIdxDim, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)kIdxDim * sizeof(bf16)};
  cuuint32_t box[2] = {64, (cuuint32_t)box_rows};
  cuuint32_t es[2] = {1, 1};
  const CUresult r = encode(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), dims,
                            strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: " + std::to_string((int)r));
  return m;
}

const CUtensorMap& cached_map(const void* base, int64_t rows, int box_rows) {
  static std::mutex mu;
  static std::map<std::tuple<const void*, int64_t, int>, CUtensorMap> maps;
  std::lock_guard<std::mutex> lk(mu);
  auto key = std::make_tuple(base, rows, box_rows);
  auto it = maps.find(key);
  if (it == maps.end()) it = maps.emplace(key, make_tensor_map(base, rows, box_rows)).first;
  return it->second;
}
}  // namespace

bool kpool_score_umma(float* scores, const bf16* q_idx, int64_t q_rows, const float* w, int ld_w,
                      const bf16* idx_cache, int64_t idx_rows, const MlaTokMeta& meta,
                      const int2* tiles, int ntiles, int row0, int stride, cudaStream_t st) {
  if (q_rows <= 0 || idx_rows <= 0) return false;
  // the PQ instance for launches with the metadata uploaded before the step (decode)
  const bool pq = meta.pre;
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  static int sms[64] = {};
  static bool configured[64] = {};
  const int smem = (int)sizeof(UmmaSmem) + 1024;
  if (!configured[dev]) {
    CUDA_CHECK(cudaDeviceGetAttribute(&sms[dev], cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaFuncSetAttribute(kpool_score_umma_kernel<false>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    CUDA_CHECK(cudaFuncSetAttribute(kpool_score_umma_kernel<true>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    configured[dev] = true;
  }
  const CUtensorMap tq = cached_map(q_idx, q_rows, kUN);
  // the INDEX_KEY pool as rows of 128 values; a box is one page's pool keys
  const CUtensorMap tk = cached_map(idx_cache, idx_rows, kPoolsPerPage);
  const CUtensorMap tq2 = cached_map(q_idx, q_rows, kUN / 2);
  // pool-tile slices per token tile (one CTA per SM: TMEM and shared memory hold one): the count
  // with the fewest tile loads on the busiest SM, a CTA's query load counting as two tiles
  const int npt = std::max(1, (int)cdiv(stride, kUP));
  int splits = 1;
  int64_t best = INT64_MAX;
  for (int s = 1; s <= std::min(npt, 4 * sms[dev]); ++s) {
    const int64_t waves = cdiv((int64_t)ntiles * s, (int64_t)sms[dev]);
    const int64_t cost = waves * (cdiv(npt, s) + 2);
    if (cost < best) {
      best = cost;
      splits = s;
    }
  }
  if (pq)
    launch_pdl(kpool_score_umma_kernel<true>, dim3(ntiles, splits), dim3(kUThreads), smem, st, scores, w, ld_w, meta,
               tiles, row0, stride, tq, tk, tq2, q_idx, q_rows);
  else
    launch_pdl(kpool_score_umma_kernel<false>, dim3(ntiles, splits), dim3(kUThreads), smem, st, scores, w, ld_w, meta,
               tiles, row0, stride, tq, tk, tq2, q_idx, q_rows);
  return true;
}

}  // namespace k
}  // namespace glm
