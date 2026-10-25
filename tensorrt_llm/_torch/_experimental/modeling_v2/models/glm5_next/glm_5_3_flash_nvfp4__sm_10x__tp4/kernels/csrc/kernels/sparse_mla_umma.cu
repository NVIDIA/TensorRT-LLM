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
// Sparse MLA attention on the SM100 tensor cores (tcgen05.mma into tensor memory).
//
// One CTA per query tile: up to 64 query rows, the H heads of 64 / H consecutive tokens of one
// sequence. A pre-pass (sparse_mla_union_kernel) writes each tile's union of visible pools
// (sorted) with a bit per token that sees the pool. The CTA streams the union in tiles of 32
// cache rows (8 pools, a 4-stage ring): S = Q K^T (M = 64, N = 32, K = 512), masked online
// softmax, O += P V (N = 2 x 256, K = 32). V is the same latent rows: the shared-memory tile is the
// K-major B operand of the first product and the MN-major B operand of the second. The QK product
// of tile i + 2 is queued before the PV product of tile i (three S / P buffers), so the tensor
// cores work while the softmax of tile i runs.
// Tensor memory (512 columns): with M = 64 an accumulator occupies the lower 16 lanes of each
// 32-lane quadrant, so O (64 x 512 fp32) takes all columns of those lanes and the S buffers
// (64 x 32) sit in the upper 16 lanes. Softmax warp q owns rows 16q .. 16q + 15: their S in lanes
// 32q + 16 .. + 31, their O in lanes 32q .. + 15; two threads (t, t + 16) share a row.
// Lazy rescaling (as in FlashAttention 4): the running max that P is relative to is raised (and
// O rescaled in tensor memory) only when a row's max exceeds it by more than 2^8.
// Warp roles: 0-3 softmax / epilogue; 4-7 producers: the query tile by TMA, then per KV tile the
// rows' cache addresses, one visibility bit mask per token (position, causality, pool
// membership), and cp.async row gathers (16 B per copy, written in the 128-byte-swizzled layout;
// TMA boxes of one pool's 4 rows x 64 dims would be issue-bound); 8 MMA issuer.
#include <cuda.h>
#include <cudaTypedefs.h>

#include <map>
#include <mutex>
#include <tuple>

#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/arch/mma_sm100_desc.hpp"
#include "cute/arch/mma_sm100_umma.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cutlass/arch/barrier.h"
#include "mla.h"
#include "pdl.cuh"

namespace glm {
namespace k {

namespace {
constexpr int kSQ = 64;                  // query rows per tile (MMA M)
constexpr int kSK = 32;                  // cache rows per KV tile (16 pools)
constexpr int kSPools = kSK / kKpool;    // pools per KV tile
constexpr int kSStages = 4;              // KV tile ring
constexpr int kSBuf = 3;                 // S and P buffers (the QK products run two tiles ahead)
constexpr int kSAtoms = kLatent / 64;    // 64-dim swizzle atoms per row (8)
constexpr int kQAtom = kSQ * 128;        // bytes of one atom of the query tile
constexpr int kKAtom = kSK * 128;        // bytes of one atom of a KV tile
constexpr int kQTileB = kSAtoms * kQAtom;  // 64 KB
constexpr int kKTileB = kSAtoms * kKAtom;  // 32 KB
constexpr int kSThreads = 288;
constexpr int kSProducer = 4, kSProducers = 4, kSMma = 8;  // producer warps 4-7
constexpr int kSPT = kSProducers * 32;
constexpr int kSTokMax = 8;              // tokens per tile (H >= 8)
constexpr float kRescaleLog2 = 8.f;

struct SmlaSmem {
  uint8_t q[kQTileB];
  uint8_t kv[kSStages][kKTileB];
  uint8_t p[kSBuf][kSQ * 128];  // P: 64 rows x kSK cache rows (bf16, K-major, 128-B swizzle)
  uint32_t vis[kSStages][kSTokMax][kSK / 32];  // bit c: token j sees cache row c of the tile
  int64_t kaddr[kSStages][kSK];  // element offset of each row in the cache
  int tok_pos[kSTokMax];
  // pvdone[i % 2]: the PV product of tile i is done (two barriers: with the QK products two tiles
  // ahead, the softmax of tile j waits for PV j - 1 while PV j - 2 may still be pending)
  uint64_t full[kSStages], empty[kSStages], sfull[kSBuf], sfree[kSBuf], pfull[kSBuf], pvdone[2],
      odone, qfull;
  uint32_t tmem;
};

using Barrier = cutlass::arch::ClusterBarrier;
using TxBarrier = cutlass::arch::ClusterTransactionBarrier;

__device__ __forceinline__ uint32_t smem32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ int64_t phys_row_u(const MlaTokMeta& m, int b, int pos) {
  const int64_t page = m.block_tables[(size_t)b * m.max_blocks + pos / kMlaBlock];
  return page * m.lat_page_rows + pos % kMlaBlock;
}

// 128-byte-swizzled shared-memory descriptors: K-major (rows 128 B apart, 8-row groups 1024 B
// apart) and MN-major (atoms of 64 MN elements lbo bytes apart, 8-row K groups 1024 B apart)
__device__ __forceinline__ uint64_t desc_kmaj(uint32_t addr) {
  cute::UMMA::SmemDescriptor d;
  d.start_address_ = static_cast<uint16_t>((addr >> 4) & 0x3FFF);
  d.leading_byte_offset_ = 1;
  d.stride_byte_offset_ = 1024 >> 4;
  d.version_ = 1;
  d.base_offset_ = 0;
  d.lbo_mode_ = 0;
  d.layout_type_ = static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_128B);
  return d.desc_;
}
__device__ __forceinline__ uint64_t desc_mnmaj(uint32_t addr, uint32_t lbo) {
  cute::UMMA::SmemDescriptor d;
  d.start_address_ = static_cast<uint16_t>((addr >> 4) & 0x3FFF);
  d.leading_byte_offset_ = static_cast<uint16_t>(lbo >> 4);
  d.stride_byte_offset_ = 1024 >> 4;
  d.version_ = 1;
  d.base_offset_ = 0;
  d.lbo_mode_ = 0;
  d.layout_type_ = static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_128B);
  return d.desc_;
}

// one kind::f16 MMA, issued by the calling thread (the MMA warp's elected thread issues a whole
// product: per-MMA elect + runtime descriptors would make the warp, not the tensor pipe, the limit)
__device__ __forceinline__ void mma_f16_raw(uint32_t td, uint64_t da, uint64_t db, uint32_t idesc,
                                            uint32_t acc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n" ::"r"(td),
      "l"(da), "l"(db), "r"(idesc), "r"(acc));
}

// 16 lanes x 32 columns per thread half: threads t < 16 read lane base + t, columns [c, c + 32);
// threads t >= 16 lane base + t - 16, columns [c + 32, c + 64)
__device__ __forceinline__ void tmem_ld16(uint32_t addr, uint32_t (&d)[32]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.16x32bx2.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,"
      "%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32], 32;\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]),
        "=r"(d[7]), "=r"(d[8]), "=r"(d[9]), "=r"(d[10]), "=r"(d[11]), "=r"(d[12]), "=r"(d[13]),
        "=r"(d[14]), "=r"(d[15]), "=r"(d[16]), "=r"(d[17]), "=r"(d[18]), "=r"(d[19]),
        "=r"(d[20]), "=r"(d[21]), "=r"(d[22]), "=r"(d[23]), "=r"(d[24]), "=r"(d[25]),
        "=r"(d[26]), "=r"(d[27]), "=r"(d[28]), "=r"(d[29]), "=r"(d[30]), "=r"(d[31])
      : "r"(addr));
}
__device__ __forceinline__ void tmem_st16(uint32_t addr, const uint32_t (&d)[32]) {
  asm volatile(
      "tcgen05.st.sync.aligned.16x32bx2.x32.b32 [%0], 32, {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,"
      "%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};\n"
      :
      : "r"(addr), "r"(d[0]), "r"(d[1]), "r"(d[2]), "r"(d[3]), "r"(d[4]), "r"(d[5]), "r"(d[6]),
        "r"(d[7]), "r"(d[8]), "r"(d[9]), "r"(d[10]), "r"(d[11]), "r"(d[12]), "r"(d[13]),
        "r"(d[14]), "r"(d[15]), "r"(d[16]), "r"(d[17]), "r"(d[18]), "r"(d[19]), "r"(d[20]),
        "r"(d[21]), "r"(d[22]), "r"(d[23]), "r"(d[24]), "r"(d[25]), "r"(d[26]), "r"(d[27]),
        "r"(d[28]), "r"(d[29]), "r"(d[30]), "r"(d[31])
      : "memory");
}
// N = 16 or 32 columns per thread half (threads t >= 16: columns [c + N, c + 2N))
template <int N>
__device__ __forceinline__ void tmem_ld_half(uint32_t addr, uint32_t (&d)[N]) {
  if constexpr (N == 32) {
    tmem_ld16(addr, d);
  } else {
    static_assert(N == 16, "16 or 32 columns");
    asm volatile(
        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,"
        "%13,%14,%15}, [%16], 16;\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]),
          "=r"(d[7]), "=r"(d[8]), "=r"(d[9]), "=r"(d[10]), "=r"(d[11]), "=r"(d[12]),
          "=r"(d[13]), "=r"(d[14]), "=r"(d[15])
        : "r"(addr));
  }
}
template <int N>
__device__ __forceinline__ void tmem_wait_half(uint32_t (&d)[N]) {
  if constexpr (N == 32) {
    asm volatile(
        "tcgen05.wait::ld.sync.aligned;\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]),
          "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]),
          "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]),
          "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]),
          "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]),
          "+r"(d[31])
        :
        : "memory");
  } else {
    asm volatile(
        "tcgen05.wait::ld.sync.aligned;\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]),
          "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]),
          "+r"(d[13]), "+r"(d[14]), "+r"(d[15])
        :
        : "memory");
  }
}
// tcgen05.wait::ld that the loaded registers depend on
__device__ __forceinline__ void tmem_wait_ld(uint32_t (&d)[32]) {
  asm volatile(
      "tcgen05.wait::ld.sync.aligned;\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]),
        "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]),
        "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]),
        "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]),
        "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31])
      :
      : "memory");
}

__device__ __forceinline__ float ex2(float x) {  // 2^x (-inf -> 0)
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

// per-tile union in the workspace: npools, pools[cap], rows[cap] (cache row of the pool's first
// position: a pool's kKpool positions share a page), member[cap] (bit j: token j)
__host__ __device__ __forceinline__ int union_cap(int H) { return (kSQ / H) * (kSelPools + 1); }
__host__ __device__ __forceinline__ size_t union_bytes(int H) {
  return ((size_t)16 + (size_t)union_cap(H) * 9 + 15) / 16 * 16;
}
}  // namespace

// ------------------------------------------------------------------ union pre-pass
// grid (ntiles), 256 threads: the tile's tokens' visible pools (the dense prefix [0, ncand] or
// the selected pools + the tail pool ncand) as a sorted list with token bits
__global__ void __launch_bounds__(256) sparse_mla_union_kernel(
    uint8_t* __restrict__ ws, const int* __restrict__ sel, const MlaTokMeta meta,
    const int2* __restrict__ tiles, int H, int bitmap_words) {
  pdl_entry();
  extern __shared__ __align__(16) uint32_t bitmap[];
  __shared__ int scan[256];
  __shared__ int tok_nc[kSTokMax];
  const int tid = threadIdx.x;
  const int2 tile = tiles[blockIdx.x];
  const int t0 = tile.x, nt = tile.y;
  uint8_t* u = ws + (size_t)blockIdx.x * union_bytes(H);
  int* np_out = reinterpret_cast<int*>(u);
  int* pools = reinterpret_cast<int*>(u + 16);
  int* prow = pools + union_cap(H);
  uint8_t* member = u + 16 + (size_t)union_cap(H) * 8;
  const int bseq = meta.token_seq[t0];
  if (tid < nt) tok_nc[tid] = (meta.positions[t0 + tid] + 1) / kKpool;
  for (int i = tid; i < bitmap_words; i += 256) bitmap[i] = 0;
  __syncthreads();
  for (int j = 0; j < nt; ++j) {
    const int nc = tok_nc[j];
    if (nc <= kSelPools) {
      for (int w = tid; w <= nc / 32; w += 256) {
        const int lo = w * 32, hi = min(nc, lo + 31);
        const uint32_t bits = hi - lo == 31 ? 0xffffffffu : ((2u << (hi - lo)) - 1u);
        atomicOr(&bitmap[w], bits);
      }
    } else {
      const int* ts = sel + (size_t)(t0 + j) * kSelPools;
      for (int k = tid; k < kSelPools; k += 256) {
        const int pl = ts[k];
        atomicOr(&bitmap[pl >> 5], 1u << (pl & 31));
      }
      if (tid == 0) atomicOr(&bitmap[nc >> 5], 1u << (nc & 31));
    }
  }
  __syncthreads();
  // compact (block exclusive scan of popcounts)
  const int per = (bitmap_words + 255) / 256;
  const int w0 = tid * per, w1 = min(bitmap_words, w0 + per);
  int cnt = 0;
  for (int w = w0; w < w1; ++w) cnt += __popc(bitmap[w]);
  scan[tid] = cnt;
  __syncthreads();
  for (int o = 1; o < 256; o <<= 1) {
    const int v = tid >= o ? scan[tid - o] : 0;
    __syncthreads();
    scan[tid] += v;
    __syncthreads();
  }
  int idx = scan[tid] - cnt;
  for (int w = w0; w < w1; ++w) {
    uint32_t bits = bitmap[w];
    while (bits) {
      const int bit = __ffs(bits) - 1;
      bits &= bits - 1;
      const int pl = w * 32 + bit;
      uint8_t m = 0;  // dense tokens and tail pools arithmetically; selections below
      for (int j = 0; j < nt; ++j) {
        const int nc = tok_nc[j];
        if (nc <= kSelPools ? pl <= nc : pl == nc) m |= 1u << j;
      }
      pools[idx] = pl;
      prow[idx] = (int)phys_row_u(meta, bseq, pl * kKpool);
      member[idx] = m;
      ++idx;
    }
  }
  const int npools = scan[255];
  if (tid == 255) *np_out = npools;
  __syncthreads();
  for (int j = 0; j < nt; ++j) {
    if (tok_nc[j] <= kSelPools) continue;
    const int* ts = sel + (size_t)(t0 + j) * kSelPools;
    for (int k = tid; k < kSelPools; k += 256) {
      const int pl = ts[k];
      int lo = 0, hi = npools - 1;
      while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (pools[mid] < pl) lo = mid + 1; else hi = mid;
      }
      atomicOr(reinterpret_cast<unsigned int*>(member + (lo & ~3)), (1u << j) << (8 * (lo & 3)));
    }
    __syncthreads();
  }
}

// decode tiles (<= 4 tokens): the same workspace with short dependency chains -- a bit map per
// token in shared memory (no searches in the written list), the selections and the block-table
// lookups issued together. grid (ntiles), 1024 threads; shared memory 4 x words + union_cap ints.
// The bit maps are cleared and scanned only up to the tile's longest context (the tail pool of its
// last token, not max_pools); the list in ascending pool order. Wait, then trigger: the attention
// behind it launches only after this grid's wait, so its prologue (the query tile from the absorb)
// may run before its own wait (FmlaUnionArgs::pre).
__global__ void __launch_bounds__(1024) sparse_mla_union_dec2_kernel(
    uint8_t* __restrict__ ws, const int* sel, const MlaTokMeta meta,
    const int2* __restrict__ tiles, int H, int words) {
  pdl_wait();
  pdl_trigger();
  extern __shared__ __align__(16) uint32_t ubits[];  // [4][words], then the pool list
  int* list = reinterpret_cast<int*>(ubits + 4 * words);
  __shared__ int warp_tot[32];
  __shared__ int tok_nc[4];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const int2 tile = tiles[blockIdx.x];
  const int t0 = tile.x, nt = min(tile.y, 4);
  uint8_t* u = ws + (size_t)blockIdx.x * union_bytes(H);
  int* pools = reinterpret_cast<int*>(u + 16);
  int* prow = pools + union_cap(H);
  uint8_t* member = u + 16 + (size_t)union_cap(H) * 8;
  const int bseq = meta.token_seq[t0];
  if (tid < 4) tok_nc[tid] = tid < nt ? (meta.positions[t0 + tid] + 1) / kKpool : -1;
  // the selections (the PDL predecessor's output: L2 loads), 2 per thread, in flight together
  constexpr int kQ = 4 * kSelPools / 1024;
  int v[kQ];
#pragma unroll
  for (int q = 0; q < kQ; ++q) {
    const int i = tid + q * 1024, j = i / kSelPools;
    v[q] = j < nt ? __ldcg(sel + (size_t)(t0 + j) * kSelPools + i % kSelPools) : -1;
  }
  __syncthreads();
  int ncmax = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) ncmax = max(ncmax, tok_nc[j]);
  const int wn = min(words, ncmax / 32 + 1);  // words in use: up to the longest token's tail pool
  for (int i = tid; i < 4 * wn; i += 1024) ubits[(i / wn) * words + i % wn] = 0u;
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int nc = tok_nc[j];
    if (nc < 0) continue;
    uint32_t* tb = ubits + j * words;
    if (nc <= kSelPools) {  // the dense prefix and the tail pool: [0, nc]
      for (int w = tid; w <= nc / 32; w += 1024) {
        const int lo = w * 32, hi = min(nc, lo + 31);
        tb[w] = hi - lo == 31 ? 0xffffffffu : ((2u << (hi - lo)) - 1u);
      }
    } else if (tid == 0) {  // the tail pool
      atomicOr(&tb[nc >> 5], 1u << (nc & 31));
    }
  }
#pragma unroll
  for (int q = 0; q < kQ; ++q) {  // the selected pools of the sparse tokens
    const int i = tid + q * 1024, j = i / kSelPools;
    if (j < nt && tok_nc[j] > kSelPools && v[q] >= 0)
      atomicOr(&ubits[j * words + (v[q] >> 5)], 1u << (v[q] & 31));
  }
  __syncthreads();
  // compact the union (block exclusive scan of popcounts over contiguous word ranges)
  const int per = (wn + 1023) / 1024;
  const int w0 = min(wn, tid * per), w1 = min(wn, w0 + per);
  int cnt = 0;
  for (int w = w0; w < w1; ++w)
    cnt += __popc(ubits[w] | ubits[words + w] | ubits[2 * words + w] | ubits[3 * words + w]);
  int inc = cnt;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int x = __shfl_up_sync(0xffffffffu, inc, o);
    if (lane >= o) inc += x;
  }
  if (lane == 31) warp_tot[wid] = inc;
  __syncthreads();
  int wt = lane < 32 ? warp_tot[lane] : 0, wi = wt;  // warp totals: exclusive base and the sum
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int x = __shfl_up_sync(0xffffffffu, wi, o);
    if (lane >= o) wi += x;
  }
  const int base = __shfl_sync(0xffffffffu, wi - wt, wid);
  const int total = __shfl_sync(0xffffffffu, wi, 31);
  int idx = base + inc - cnt;
  for (int w = w0; w < w1; ++w) {
    const uint32_t b0 = ubits[w], b1 = ubits[words + w], b2 = ubits[2 * words + w],
                   b3 = ubits[3 * words + w];
    uint32_t bits = b0 | b1 | b2 | b3;
    while (bits) {
      const int bit = __ffs(bits) - 1;
      bits &= bits - 1;
      const int pl = w * 32 + bit;
      list[idx] = pl;
      pools[idx] = pl;
      member[idx] = (uint8_t)(((b0 >> bit) & 1u) | (((b1 >> bit) & 1u) << 1) |
                              (((b2 >> bit) & 1u) << 2) | (((b3 >> bit) & 1u) << 3));
      ++idx;
    }
  }
  if (tid == 0) *reinterpret_cast<int*>(u) = total;
  __syncthreads();
  // the cache row of each pool's first position: the block-table loads issued together
  for (int i0 = tid; i0 < total; i0 += 2 * 1024) {
    int r[2];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int i = i0 + q * 1024;
      r[q] = i < total ? (int)phys_row_u(meta, bseq, list[i] * kKpool) : 0;
    }
#pragma unroll
    for (int q = 0; q < 2; ++q)
      if (i0 + q * 1024 < total) prow[i0 + q * 1024] = r[q];
  }
}

// ------------------------------------------------------------------ attention
__global__ void __launch_bounds__(kSThreads, 1) sparse_mla_umma_kernel(
    bf16* __restrict__ out, const uint8_t* __restrict__ ws, const bf16* __restrict__ cache,
    const MlaTokMeta meta, const int2* __restrict__ tiles, int H, float scale,
    const __grid_constant__ CUtensorMap tm_q) {
  pdl_entry();
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t pad = (1024u - (smem32(smem_raw) & 1023u)) & 1023u;
  SmlaSmem& sm = *reinterpret_cast<SmlaSmem*>(smem_raw + pad);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int2 tile = tiles[blockIdx.x];
  const int t0 = tile.x, nt = tile.y, rowsQ = nt * H;
  const uint8_t* u = ws + (size_t)blockIdx.x * union_bytes(H);
  const int npools = *reinterpret_cast<const int*>(u);
  const int* upools = reinterpret_cast<const int*>(u + 16);
  const int* urows = upools + union_cap(H);
  const uint8_t* umember = u + 16 + (size_t)union_cap(H) * 8;
  const int ntile = (npools + kSPools - 1) / kSPools;
  if (tid < nt) sm.tok_pos[tid] = meta.positions[t0 + tid];
  if (tid == 0) {
    for (int s = 0; s < kSStages; ++s) {
      Barrier::init(&sm.full[s], kSPT + kSK);  // every copy thread's cp.async + the metadata
      Barrier::init(&sm.empty[s], 1);
    }
    for (int s = 0; s < kSBuf; ++s) {
      Barrier::init(&sm.sfull[s], 1);
      Barrier::init(&sm.sfree[s], 4 * 32);
      Barrier::init(&sm.pfull[s], 4 * 32);
    }
    Barrier::init(&sm.pvdone[0], 1);
    Barrier::init(&sm.pvdone[1], 1);
    Barrier::init(&sm.odone, 1);
    Barrier::init(&sm.qfull, 1);
    cutlass::arch::fence_barrier_init();
  }
  cute::TMEM::Allocator1Sm tmem_alloc{};
  if (warp == kSMma) tmem_alloc.allocate(512, &sm.tmem);
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const uint32_t tmem = sm.tmem;
  constexpr uint64_t kHint = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);

  if (warp >= kSProducer && warp < kSProducer + kSProducers) {
    const int pt = tid - kSProducer * 32;
    if (pt == 0) {
      TxBarrier::arrive_and_expect_tx(&sm.qfull, kQTileB);
      for (int a = 0; a < kSAtoms; ++a)
        cute::SM90_TMA_LOAD_2D::copy(&tm_q, &sm.qfull, kHint, sm.q + a * kQAtom, a * 64, t0 * H);
    }
    // row pt of tile j (pool pt / 4 of the tile's): pool, cache row of the pool, token bits,
    // loaded two tiles ahead (each load is a round trip to L2; issued in the loop, with the
    // block-table lookup they depend on, they would make every tile wait on them)
    auto meta_load = [&](int j, int& pl, int& row, uint32_t& mem) {
      const int idx = j * kSPools + pt / kKpool;
      const bool in = j < ntile && idx < npools;
      pl = in ? upools[idx] : -1;
      row = in ? urows[idx] : 0;
      mem = in ? (uint32_t)umember[idx] : 0u;
    };
    int pl0 = -1, row0 = 0, pl1 = -1, row1 = 0;
    uint32_t mem0 = 0u, mem1 = 0u;
    if (pt < kSK) {
      meta_load(0, pl0, row0, mem0);
      meta_load(1, pl1, row1, mem1);
    }
    auto kv_tile = [&](int j, int& pl_r, int& row_r, uint32_t& mem_r) {
      const int s = j % kSStages;
      const int pl = pl_r, prow = row_r;
      const uint32_t mem = mem_r;
      if (pt < kSK) meta_load(j + 2, pl_r, row_r, mem_r);
      Barrier::wait(&sm.empty[s], ((j / kSStages) & 1) ^ 1);
      if (pt < kSK) {  // row pt of the tile: address, visibility
        const int pos = pl >= 0 ? pl * kKpool + pt % kKpool : -1;
        sm.kaddr[s][pt] = pos >= 0 ? (int64_t)(prow + pt % kKpool) * kLatent : 0;
#pragma unroll
        for (int t = 0; t < kSTokMax; ++t) {
          const bool v = t < nt && pos >= 0 && pos <= sm.tok_pos[t] && ((mem >> t) & 1u);
          const uint32_t m = __ballot_sync(0xffffffff, v);
          if (lane == 0) sm.vis[s][t][pt / 32] = m;
        }
      }
      asm volatile("bar.sync 1, %0;\n" ::"n"(kSPT));  // (row addresses of this tile)
      const uint32_t dst0 = smem32(sm.kv[s]);
#pragma unroll 4
      for (int i = pt; i < kSK * (kLatent / 8); i += kSPT) {
        const int r = i >> 6, c = i & 63;  // row, 16-byte chunk (8 dims)
        const bf16* src = cache + sm.kaddr[s][r] + c * 8;
        const uint32_t dst = dst0 + (c >> 3) * kKAtom + r * 128 + (((c & 7) ^ (r & 7)) << 4);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(
                       smem32(&sm.full[s])));
      if (pt < kSK) Barrier::arrive(&sm.full[s]);  // (with the metadata stores)
    };
    for (int j = 0; j < ntile; j += 2) {
      kv_tile(j, pl0, row0, mem0);
      if (j + 1 < ntile) kv_tile(j + 1, pl1, row1, mem1);
    }
  } else if (warp == kSMma) {
    const uint64_t id_qk = cute::UMMA::make_runtime_instr_desc<
        cute::bfloat16_t, cute::bfloat16_t, float, kSQ, kSK, cute::UMMA::Major::K,
        cute::UMMA::Major::K>();
    const uint64_t id_pv = cute::UMMA::make_runtime_instr_desc<
        cute::bfloat16_t, cute::bfloat16_t, float, kSQ, 256, cute::UMMA::Major::K,
        cute::UMMA::Major::MN>();
    // the products' descriptors: a stage / K step adds its byte offset >> 4 to the start address
    // field (shared memory < 256 KB: no carry out of the 14-bit field)
    const uint32_t idq = uint32_t(id_qk >> 32), idp = uint32_t(id_pv >> 32);
    const uint64_t dq0 = desc_kmaj(smem32(sm.q));
    Barrier::wait(&sm.qfull, 0);
    auto do_pv = [&](int i) {
      const int s = i % kSStages, pb = i % kSBuf;
      Barrier::wait(&sm.pfull[pb], (i / kSBuf) & 1);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      if (cute::elect_one_sync()) {
        const uint64_t dp0 = desc_kmaj(smem32(sm.p[pb])), dv0 = desc_mnmaj(smem32(sm.kv[s]), kKAtom);
#pragma unroll
        for (int ks = 0; ks < kSK / 16; ++ks)
#pragma unroll
          for (int nh = 0; nh < 2; ++nh)
            mma_f16_raw(tmem + nh * 256, dp0 + (uint64_t)((ks * 32) >> 4),
                        dv0 + (uint64_t)((nh * 4 * kKAtom + ks * 2048) >> 4), idp,
                        (i > 0 || ks > 0) ? 1u : 0u);
      }
      __syncwarp();
      cutlass::arch::umma_arrive(&sm.empty[s]);
      cutlass::arch::umma_arrive(&sm.pvdone[i & 1]);
    };
    auto do_qk = [&](int j) {
      const int s = j % kSStages, sb = j % kSBuf;
      Barrier::wait(&sm.full[s], (j / kSStages) & 1);
      Barrier::wait(&sm.sfree[sb], ((j / kSBuf) & 1) ^ 1);
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // (cp.async data -> MMA)
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      const uint32_t s_tmem = tmem + (16u << 16) + sb * kSK;
      if (cute::elect_one_sync()) {
        const uint64_t dk0 = desc_kmaj(smem32(sm.kv[s]));
#pragma unroll
        for (int kb = 0; kb < kLatent / 16; ++kb)
          mma_f16_raw(s_tmem, dq0 + (uint64_t)(((kb / 4) * kQAtom + (kb % 4) * 32) >> 4),
                      dk0 + (uint64_t)(((kb / 4) * kKAtom + (kb % 4) * 32) >> 4), idq,
                      kb > 0 ? 1u : 0u);
      }
      __syncwarp();
      cutlass::arch::umma_arrive(&sm.sfull[sb]);
    };
    // the QK product of tile i + 2 is queued before the PV product of tile i, so the tensor
    // cores have work while the softmax of tile i runs
    for (int j = 0; j < min(2, ntile); ++j) do_qk(j);
    for (int i = 0; i < ntile; ++i) {
      if (i + 2 < ntile) do_qk(i + 2);
      do_pv(i);
    }
    cutlass::arch::umma_arrive(&sm.odone);  // (every product done: the epilogue reads O)
  } else if (warp < 4) {  // softmax / epilogue
    const int half = lane >> 4, r = warp * 16 + (lane & 15);
    const int tok = min(r / H, kSTokMax - 1);
    const float sl2 = scale * 1.4426950408889634f;
    float m_used = -INFINITY, l = 0.f;
    const uint32_t lane_s = (uint32_t)(warp * 32 + 16) << 16, lane_o = (uint32_t)(warp * 32) << 16;
    for (int j = 0; j < ntile; ++j) {
      const int s = j % kSStages, sb = j % kSBuf;
      Barrier::wait(&sm.full[s], (j / kSStages) & 1);  // (row metadata of this tile)
      Barrier::wait(&sm.sfull[sb], (j / kSBuf) & 1);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      constexpr int kSC = kSK / 2;  // columns per thread (the row's other half: thread t ^ 16)
      uint32_t v[kSC];
      tmem_ld_half<kSC>(tmem + lane_s + sb * kSK, v);
      tmem_wait_half<kSC>(v);
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      Barrier::arrive(&sm.sfree[sb]);
      const uint32_t vm = sm.vis[s][tok][(half * kSC) / 32] >> ((half * kSC) % 32);
      // row max over the raw scores (the scale is positive), masked entries -inf
      float x[kSC], mx = -INFINITY;
#pragma unroll
      for (int i = 0; i < kSC; ++i) {
        x[i] = (vm >> i) & 1u ? __uint_as_float(v[i]) : -INFINITY;
        mx = fmaxf(mx, x[i]);
      }
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 16)) * sl2;  // (log2 domain)
      const float m_old = m_used;
      const bool raise = mx > m_used + kRescaleLog2;
      if (raise) m_used = mx;
      const float alpha = raise ? (m_old == -INFINITY ? 0.f : ex2(m_old - m_used)) : 1.f;
      // p = 2^(s * scale * log2(e) - m): masked entries and rows without visible entries give 0
      const float mref = m_used == -INFINITY ? 0.f : m_used;
      float sum = 0.f;
      uint32_t pk[kSC / 2];
#pragma unroll
      for (int i = 0; i < kSC / 2; ++i) {
        const __nv_bfloat162 pp = __floats2bfloat162_rn(ex2(fmaf(x[2 * i], sl2, -mref)),
                                                        ex2(fmaf(x[2 * i + 1], sl2, -mref)));
        const float2 pf = __bfloat1622float2(pp);
        sum += pf.x + pf.y;
        pk[i] = *reinterpret_cast<const uint32_t*>(&pp);
      }
      sum += __shfl_xor_sync(0xffffffff, sum, 16);
      l = l * alpha + sum;
      // P row r, columns [kSC half, + kSC): 16-byte chunks, 128-byte swizzle
      uint8_t* prow = sm.p[sb] + r * 128;
#pragma unroll
      for (int c = 0; c < kSC / 8; ++c) {
        const int chunk = (half * (kSC / 8) + c) ^ (r & 7);
        *reinterpret_cast<uint4*>(prow + chunk * 16) =
            make_uint4(pk[4 * c], pk[4 * c + 1], pk[4 * c + 2], pk[4 * c + 3]);
      }
      // O rows of this warp hold the previous tiles relative to m_old: rescale them before this
      // tile's PV adds to them (rare: only when some row's max rose by more than 2^8)
      if (__any_sync(0xffffffff, raise && m_old != -INFINITY) && j > 0) {
        Barrier::wait(&sm.pvdone[(j - 1) & 1], ((j - 1) >> 1) & 1);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll 1
        for (int cb = 0; cb < kLatent / 64; ++cb) {
          uint32_t o[32];
          tmem_ld16(tmem + lane_o + cb * 64, o);
          tmem_wait_ld(o);
#pragma unroll
          for (int i = 0; i < 32; ++i) o[i] = __float_as_uint(__uint_as_float(o[i]) * alpha);
          tmem_st16(tmem + lane_o + cb * 64, o);
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // (P for the tensor cores)
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      Barrier::arrive(&sm.pfull[sb]);
    }
    // ---- O / l -> out
    Barrier::wait(&sm.odone, 0);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    const float inv = l > 0.f ? 1.f / l : 0.f;
#pragma unroll 1
    for (int cb = 0; cb < kLatent / 64; ++cb) {
      uint32_t o[32];
      tmem_ld16(tmem + lane_o + cb * 64, o);
      tmem_wait_ld(o);
      if (r < rowsQ) {
        bf16* dst = out + ((size_t)t0 * H + r) * kLatent + cb * 64 + half * 32;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          uint32_t w[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const __nv_bfloat162 pp = __floats2bfloat162_rn(
                ntile > 0 ? __uint_as_float(o[8 * c + 2 * e]) * inv : 0.f,
                ntile > 0 ? __uint_as_float(o[8 * c + 2 * e + 1]) * inv : 0.f);
            w[e] = *reinterpret_cast<const uint32_t*>(&pp);
          }
          *reinterpret_cast<uint4*>(dst + c * 8) = make_uint4(w[0], w[1], w[2], w[3]);
        }
      }
    }
  }
  __syncthreads();
  if (warp == kSMma) {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    tmem_alloc.release_allocation_lock();
    tmem_alloc.free(tmem, 512);
  }
}

namespace {
// 2-D bf16 tensor map: rows of 512 values, boxes of 64 values x box_rows rows, 128-byte swizzle
CUtensorMap make_map_512(const void* base, int64_t rows, int box_rows) {
  static PFN_cuTensorMapEncodeTiled_v12000 encode = [] {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000,
                                                cudaEnableDefault, &q));
    CHECK(fn != nullptr && q == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled not found");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
  }();
  CUtensorMap m;
  cuuint64_t dims[2] = {(cuuint64_t)kLatent, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)kLatent * sizeof(bf16)};
  cuuint32_t box[2] = {64, (cuuint32_t)box_rows};
  cuuint32_t es[2] = {1, 1};
  const CUresult r = encode(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), dims,
                            strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: " + std::to_string((int)r));
  return m;
}

const CUtensorMap& map_512(const void* base, int64_t rows, int box_rows) {
  static std::mutex mu;
  static std::map<std::tuple<const void*, int64_t, int>, CUtensorMap> maps;
  std::lock_guard<std::mutex> lk(mu);
  auto key = std::make_tuple(base, rows, box_rows);
  auto it = maps.find(key);
  if (it == maps.end()) it = maps.emplace(key, make_map_512(base, rows, box_rows)).first;
  return it->second;
}
}  // namespace

// the union pre-pass alone (the FlashMLA-based tiled path reads the same workspace)
size_t sparse_mla_union_ws_bytes(int H) { return union_bytes(H); }
int sparse_mla_union_capacity(int H) { return union_cap(H); }
void sparse_mla_union(void* ws, const int* sel, const MlaTokMeta& meta, const int2* tiles,
                      int ntiles, int H, int max_pools, cudaStream_t st) {
  const int words = (max_pools + 1 + 31) / 32;
  CHECK((size_t)words * 4 <= 200 * 1024, "sparse attention union: context too long");
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  static size_t union_smem[64] = {};
  if (union_smem[dev] < (size_t)words * 4) {
    CUDA_CHECK(cudaFuncSetAttribute(sparse_mla_union_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, words * 4));
    union_smem[dev] = (size_t)words * 4;
  }
  launch_pdl(sparse_mla_union_kernel, dim3(ntiles), dim3(256), (size_t)words * 4, st,
             static_cast<uint8_t*>(ws), sel, meta, tiles, H, words);
}

void sparse_mla_union_dec(void* ws, const int* sel, const MlaTokMeta& meta, const int2* tiles,
                          int ntiles, int H, int max_pools, cudaStream_t st) {
  const int words = (max_pools + 1 + 31) / 32;
  const size_t smem = ((size_t)4 * words + union_cap(H)) * 4;
  CHECK(smem <= 200 * 1024 && 64 / H <= 4, "decode union: context too long or tiles too wide");
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  static size_t dec2_smem[64] = {};
  if (smem > 48 * 1024 && dec2_smem[dev] < smem) {
    CUDA_CHECK(cudaFuncSetAttribute(sparse_mla_union_dec2_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    dec2_smem[dev] = smem;
  }
  launch_pdl(sparse_mla_union_dec2_kernel, dim3(ntiles), dim3(1024), smem, st,
             static_cast<uint8_t*>(ws), sel, meta, tiles, H, words);  // (its dependents launch after its wait)
}

bool sparse_mla_attn_umma(bf16* out, const bf16* q, int64_t q_rows, const bf16* cache,
                          int64_t cache_rows, const int* sel, const MlaTokMeta& meta,
                          const int2* tiles, int ntiles, int H, int max_pools, float scale,
                          void* ws, size_t ws_bytes, cudaStream_t st) {
  if (ws == nullptr || q_rows <= 0 || cache_rows <= 0 || H < kSQ / kSTokMax ||
      kSQ % H != 0 || (size_t)ntiles * union_bytes(H) > ws_bytes)
    return false;
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  static bool configured[64] = {};
  const int smem = (int)sizeof(SmlaSmem) + 1024;
  if (!configured[dev]) {
    CUDA_CHECK(cudaFuncSetAttribute(sparse_mla_umma_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    configured[dev] = true;
  }
  uint8_t* u = static_cast<uint8_t*>(ws);
  sparse_mla_union(u, sel, meta, tiles, ntiles, H, max_pools, st);
  const CUtensorMap tq = map_512(q, q_rows, kSQ);
  launch_pdl(sparse_mla_umma_kernel, dim3(ntiles), dim3(kSThreads), smem, st, out,
             (const uint8_t*)u, cache, meta, tiles, H, scale, tq);
  return true;
}

}  // namespace k
}  // namespace glm
