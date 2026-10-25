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
// Sparse MLA kernels: paged cache append, kpool key refresh, pool scoring + top-k selection
// and the sparse latent attention (tensor-core flash attention over gathered cache rows).
#include <algorithm>
#include <type_traits>

#include "pdl.cuh"
#include <cooperative_groups.h>
#include <mma.h>

#include "mla.h"
#include "reduce.cuh"

namespace glm {
namespace k {

// latent row (of kLatent) of position pos of batch row b
__device__ __forceinline__ int64_t phys_row(const MlaTokMeta& m, int b, int pos) {
  const int64_t page = m.block_tables[(size_t)b * m.max_blocks + pos / kMlaBlock];
  return page * m.lat_page_rows + pos % kMlaBlock;
}

// INDEX_KEY row (of kIdxDim) of the key of pool j (tokens [4j, 4j + 4)) of batch row b
__device__ __forceinline__ int64_t pool_slot(const MlaTokMeta& m, int b, int j) {
  const int64_t page = m.block_tables[(size_t)b * m.max_blocks + j / kPoolsPerPage];
  return page * m.idx_page_rows + j % kPoolsPerPage;
}
// element offset of the [k | gate] row of position pos of batch row b in the INDEX_KEY pool
__device__ __forceinline__ int64_t kg_off(const MlaTokMeta& m, int b, int pos) {
  const int64_t page = m.block_tables[(size_t)b * m.max_blocks + pos / kMlaBlock];
  return (page * m.idx_page_rows + kIdxKgRow0) * kIdxDim + (int64_t)(pos % kMlaBlock) * kKgRow;
}

// ------------------------------------------------------------------ cache append
__global__ void mla_append_kernel(bf16* __restrict__ latent_cache, bf16* __restrict__ idx_cache,
                                  const bf16* __restrict__ latent, const float* __restrict__ kgw,
                                  int ld_kgw, const bf16* __restrict__ kn_w,
                                  const bf16* __restrict__ kn_b, const MlaTokMeta meta) {
  pdl_entry();
  const int t = blockIdx.x, tid = threadIdx.x;  // 128 threads
  const int b = meta.token_seq[t], pos = meta.positions[t];
  const int64_t row = phys_row(meta, b, pos);
  // latent: 512 bf16 = 64 x 16B
  if (tid < kLatent / 8)
    reinterpret_cast<uint4*>(latent_cache + row * kLatent)[tid] =
        reinterpret_cast<const uint4*>(latent + (size_t)t * kLatent)[tid];
  // indexer key: LayerNorm(bf16(k)), eps 1e-6; gate: bf16
  __shared__ float sh[32];
  const float* src = kgw + (size_t)t * ld_kgw;
  const float kv = bf16r(src[tid]);
  const float mean = block_sum(kv, sh) / kIdxDim;
  const float dv = kv - mean;
  const float var = block_sum(dv * dv, sh) / kIdxDim;
  const bf16 kn = __float2bfloat16(dv * rsqrtf(var + 1e-6f) * __bfloat162float(kn_w[tid]) +
                                   __bfloat162float(kn_b[tid]));
  const bf16 g = __float2bfloat16(src[kIdxDim + tid]);
  bf16* st = idx_cache + kg_off(meta, b, pos);
  st[tid] = kn;
  st[kIdxDim + tid] = g;
}

void mla_append(bf16* latent_cache, bf16* idx_cache, const bf16* latent, const float* kgw,
                int ld_kgw, const bf16* kn_w, const bf16* kn_b, const MlaTokMeta& meta, int T,
                cudaStream_t st) {
  if (T == 0) return;
  launch_pdl(mla_append_kernel, dim3(T), dim3(kIdxDim), 0, st, latent_cache, idx_cache, latent,
             kgw, ld_kgw, kn_w, kn_b, meta);
}

// ------------------------------------------------------------------ kpool key refresh
__global__ void kpool_update_kernel(bf16* __restrict__ idx_cache, const bf16* __restrict__ ape,
                                    const MlaTokMeta meta) {
  pdl_entry();
  const int t = blockIdx.x, d = threadIdx.x;
  const int pos = meta.positions[t];
  if ((pos + 1) % kKpool != 0) return;
  const int b = meta.token_seq[t];
  const int p0 = pos + 1 - kKpool;
  float kk[kKpool], lg[kKpool], mx = -INFINITY;
#pragma unroll
  for (int m = 0; m < kKpool; ++m) {
    // position p0 + m: its [k | gate] row in the page (mla_append ran before this kernel)
    const bf16* r = idx_cache + kg_off(meta, b, p0 + m);
    kk[m] = __bfloat162float(r[d]);
    lg[m] = __bfloat162float(r[kIdxDim + d]) + __bfloat162float(ape[m * kIdxDim + d]);
    mx = fmaxf(mx, lg[m]);
  }
  float sum = 0.f;
#pragma unroll
  for (int m = 0; m < kKpool; ++m) {
    lg[m] = expf(lg[m] - mx);
    sum += lg[m];
  }
  float key = 0.f;
#pragma unroll
  for (int m = 0; m < kKpool; ++m) key += bf16r(bf16r(lg[m] / sum) * kk[m]);
  idx_cache[pool_slot(meta, b, pos / kKpool) * kIdxDim + d] = __float2bfloat16(key);
}

void kpool_update(bf16* idx_cache, const bf16* ape, const MlaTokMeta& meta, int T,
                  cudaStream_t st) {
  if (T == 0) return;
  launch_pdl(kpool_update_kernel, dim3(T), dim3(kIdxDim), 0, st, idx_cache, ape, meta);
}

// ------------------------------------------------------------------ fused cache update (decode)
// mla_append + kpool_update of a step of at most kCacheFuseMaxT tokens in one CTA (a warp per
// token, at least 128 threads): (1) per token the latent row and the indexer
// [LayerNorm(key) | gate] row into the pages (and the latter into shared memory); (2) the pool
// keys completed in the step (a 128-thread group per token, thread d: dimension d), with the
// members of earlier steps read from their pages. Bitwise the two kernels: each LayerNorm sum
// follows block_sum's order over the 128 values (per-warp xor trees, then the second tree, where
// the zero lanes add +0).
__device__ __forceinline__ float sum128_as_block(const float (&v)[4]) {
  float s[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) s[j] = warp_sum(v[j]);  // block_sum's sum of warp j
  return ((s[0] + 0.f) + (s[2] + 0.f)) + ((s[1] + 0.f) + (s[3] + 0.f));
}

// (the [k | gate] rows and the earlier steps' member rows in dynamic shared memory: 64 KB at 32 tokens)
struct CacheFuseSmem {
  bf16 kg[kCacheFuseMaxT][kKgRow];
  bf16 ring[kCacheFuseMaxT][kKpool - 1][kKgRow];
};
__global__ void __launch_bounds__(32 * kCacheFuseMaxT) mla_cache_update_kernel(
    bf16* __restrict__ latent_cache, bf16* __restrict__ idx_cache,
    const bf16* __restrict__ latent, const float* __restrict__ kgw, int ld_kgw,
    const bf16* __restrict__ kn_w, const bf16* __restrict__ kn_b, const bf16* __restrict__ ape,
    const MlaTokMeta meta, int T) {
  pdl_trigger();
  extern __shared__ __align__(16) uint8_t cf_smem[];
  CacheFuseSmem& cf = *reinterpret_cast<CacheFuseSmem*>(cf_smem);
  auto& kg = cf.kg;
  // the tokens' batch row, position, latent cache row, [k | gate] row, completed pool's slot, and
  // the [k | gate] rows of that pool's earlier positions that are not in the step (written by
  // earlier steps): with meta.pre loaded before the PDL wait (off the critical path)
  __shared__ int s_b[kCacheFuseMaxT], s_pos[kCacheFuseMaxT];
  __shared__ int64_t s_row[kCacheFuseMaxT], s_pslot[kCacheFuseMaxT], s_kg[kCacheFuseMaxT];
  auto& s_ring = cf.ring;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nwarps = blockDim.x >> 5, ngroups = blockDim.x >> 7;
  auto load_meta = [&]() {
    if (tid < T) {
      const int b = meta.token_seq[tid], pos = meta.positions[tid];
      s_b[tid] = b;
      s_pos[tid] = pos;
      s_row[tid] = latent ? phys_row(meta, b, pos) : 0;
      s_kg[tid] = kg_off(meta, b, pos);
      s_pslot[tid] = (pos + 1) % kKpool == 0 ? pool_slot(meta, b, pos / kKpool) : -1;
    }
    __syncthreads();
    constexpr int kC = kKgRow / 8;  // 16-byte chunks per row
    for (int i = tid; i < T * (kKpool - 1) * kC; i += blockDim.x) {
      const int t = i / ((kKpool - 1) * kC), m = (i / kC) % (kKpool - 1), c = i % kC;
      const int pos = s_pos[t];
      if ((pos + 1) % kKpool != 0) continue;
      const int b = s_b[t], q = pos + 1 - kKpool + m, tq = t - (kKpool - 1) + m;
      const bool in_step = tq >= 0 && s_b[tq] == b && s_pos[tq] == q;
      if (!in_step)
        reinterpret_cast<uint4*>(&s_ring[t][m][0])[c] =
            reinterpret_cast<const uint4*>(idx_cache + kg_off(meta, b, q))[c];
    }
  };
  // the LayerNorm weights of this lane's dimensions and the pool positional terms of this
  // thread's dimension (constants) before the PDL wait too
  float knw[4], knb[4], apv[kKpool];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    knw[j] = __bfloat162float(kn_w[lane + 32 * j]);
    knb[j] = __bfloat162float(kn_b[lane + 32 * j]);
  }
#pragma unroll
  for (int m = 0; m < kKpool; ++m) apv[m] = __bfloat162float(ape[m * kIdxDim + (tid & (kIdxDim - 1))]);
  if (meta.pre) load_meta();
  pdl_wait();
  if (!meta.pre) load_meta();
  __syncthreads();
  for (int t = warp; t < T; t += nwarps) {
    if (latent) {  // (else the norms wrote the latent rows into the cache)
      const int64_t row = s_row[t];
      for (int c = lane; c < kLatent / 8; c += 32)
        reinterpret_cast<uint4*>(latent_cache + row * kLatent)[c] =
            reinterpret_cast<const uint4*>(latent + (size_t)t * kLatent)[c];
    }
    const float* src = kgw + (size_t)t * ld_kgw;
    float kv[4], dv[4], gv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) kv[j] = bf16r(src[lane + 32 * j]);
    // (the gate values requested with the keys: loaded where they are stored, after the two reductions, they cost a
    // second L2 round trip on the cache update's path)
#pragma unroll
    for (int j = 0; j < 4; ++j) gv[j] = src[kIdxDim + lane + 32 * j];
    const float mean = sum128_as_block(kv) / kIdxDim;
#pragma unroll
    for (int j = 0; j < 4; ++j) dv[j] = kv[j] - mean;
    float sq[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) sq[j] = dv[j] * dv[j];
    const float var = sum128_as_block(sq) / kIdxDim;
    bf16* kg_dst = idx_cache + s_kg[t];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int d = lane + 32 * j;
      const bf16 kn = __float2bfloat16(dv[j] * rsqrtf(var + 1e-6f) * knw[j] + knb[j]);
      const bf16 g = __float2bfloat16(gv[j]);
      kg[t][d] = kn;
      kg[t][kIdxDim + d] = g;
      kg_dst[d] = kn;
      kg_dst[kIdxDim + d] = g;
    }
  }
  __syncthreads();
  const int d = tid & (kIdxDim - 1);
  for (int t = tid >> 7; t < T; t += ngroups) {
    const int pos = s_pos[t];
    if ((pos + 1) % kKpool != 0) continue;
    const int b = s_b[t];
    const int p0 = pos + 1 - kKpool;
    float kk[kKpool], lg[kKpool], mx = -INFINITY;
#pragma unroll
    for (int m = 0; m < kKpool; ++m) {
      const int q = p0 + m, tq = t - (kKpool - 1) + m;
      const bool in_step = tq >= 0 && s_b[tq] == b && s_pos[tq] == q;
      const bf16* r = in_step ? &kg[tq][0] : &s_ring[t][m][0];
      kk[m] = __bfloat162float(r[d]);
      lg[m] = __bfloat162float(r[kIdxDim + d]) + apv[m];
      mx = fmaxf(mx, lg[m]);
    }
    float sum = 0.f;
#pragma unroll
    for (int m = 0; m < kKpool; ++m) {
      lg[m] = expf(lg[m] - mx);
      sum += lg[m];
    }
    float key = 0.f;
#pragma unroll
    for (int m = 0; m < kKpool; ++m) key += bf16r(bf16r(lg[m] / sum) * kk[m]);
    idx_cache[s_pslot[t] * kIdxDim + d] = __float2bfloat16(key);
  }
}

bool mla_cache_update(bf16* latent_cache, bf16* idx_cache, const bf16* latent, const float* kgw,
                      int ld_kgw, const bf16* kn_w, const bf16* kn_b, const bf16* ape,
                      const MlaTokMeta& meta, int T, cudaStream_t st) {
  if (T <= 0 || T > kCacheFuseMaxT) return false;
  // a warp per token, in whole 128-thread groups (pool keys: a group per token)
  const int threads = std::max(128, 128 * (int)cdiv(T, 4));
  static bool configured[64] = {};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  if (!configured[dev]) {
    CUDA_CHECK(cudaFuncSetAttribute(mla_cache_update_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)sizeof(CacheFuseSmem)));
    configured[dev] = true;
  }
  launch_pdl(mla_cache_update_kernel, dim3(1), dim3(threads), sizeof(CacheFuseSmem), st, latent_cache,
             idx_cache, latent, kgw, ld_kgw, kn_w, kn_b, ape, meta, T);
  return true;
}

// ------------------------------------------------------------------ pool scoring
constexpr int kScoreTile = 64;    // pools per tile
constexpr int kScoreSplit = 8;    // min CTAs per token (more for few tokens)

__global__ void __launch_bounds__(128) kpool_score_kernel(float* __restrict__ scores,
                                                          const bf16* __restrict__ q_idx,
                                                          const float* __restrict__ w, int ld_w,
                                                          const bf16* __restrict__ idx_cache,
                                                          const MlaTokMeta meta, int max_pools) {
  pdl_entry();
  using namespace nvcuda;
  const int t = blockIdx.x, tid = threadIdx.x, warp = tid >> 5;
  const int pos = meta.positions[t];
  const int ncand = (pos + 1) / kKpool;
  if (ncand <= kSelPools) return;
  const int b = meta.token_seq[t];
  __shared__ __align__(32) bf16 sQ[kIdxHeads * kIdxDim];
  __shared__ __align__(32) bf16 sK[kScoreTile * kIdxDim];
  __shared__ __align__(32) float sC[kIdxHeads * kScoreTile];
  __shared__ float sW[kIdxHeads];
  for (int i = tid; i < kIdxHeads * kIdxDim / 8; i += 128)
    reinterpret_cast<uint4*>(sQ)[i] =
        reinterpret_cast<const uint4*>(q_idx + (size_t)t * kIdxHeads * kIdxDim)[i];
  if (tid < kIdxHeads) sW[tid] = w[(size_t)t * ld_w + tid] * rsqrtf((float)kIdxHeads);
  const float qs = rsqrtf((float)kIdxDim);
  const int ntiles = cdiv(ncand, kScoreTile);
  for (int tile = blockIdx.y; tile < ntiles; tile += gridDim.y) {
    const int j0 = tile * kScoreTile;
    __syncthreads();
    for (int i = tid; i < kScoreTile * kIdxDim / 8; i += 128) {
      const int pj = i / (kIdxDim / 8), c = i % (kIdxDim / 8);
      const int j = j0 + pj;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (j < ncand)
        v = reinterpret_cast<const uint4*>(idx_cache + pool_slot(meta, b, j) * kIdxDim)[c];
      reinterpret_cast<uint4*>(sK)[i] = v;
    }
    __syncthreads();
    const int mt = warp >> 1;
    for (int nt = (warp & 1) * 2; nt < (warp & 1) * 2 + 2; ++nt) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.f);
#pragma unroll
      for (int kk = 0; kk < kIdxDim / 16; ++kk) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
        wmma::load_matrix_sync(fa, sQ + mt * 16 * kIdxDim + kk * 16, kIdxDim);
        wmma::load_matrix_sync(fb, sK + nt * 16 * kIdxDim + kk * 16, kIdxDim);
        wmma::mma_sync(acc, fa, fb, acc);
      }
      wmma::store_matrix_sync(sC + mt * 16 * kScoreTile + nt * 16, acc, kScoreTile,
                              wmma::mem_row_major);
    }
    __syncthreads();
    if (tid < kScoreTile) {
      const int j = j0 + tid;
      if (j < ncand) {
        float s = 0.f;
#pragma unroll 8
        for (int h = 0; h < kIdxHeads; ++h) s += fmaxf(sC[h * kScoreTile + tid] * qs, 0.f) * sW[h];
        scores[(size_t)t * max_pools + j] = s;
      }
    }
  }
}

// Query-tiled pool scoring: one CTA per (tile of <= 8 consecutive tokens of a sequence, slice of
// pool tiles); warp w scores token w's 32 heads against each 64-pool tile with mma.sync. Pool keys
// are streamed once per token tile (decode with MTP: the 1 + N tokens of a sequence share them)
// through a two-stage cp.async pipeline. Score rows are relative to row0 (stride: row length).
constexpr int kScT = 8;           // tokens per tile
constexpr int kScP = kIdxDim + 8; // padded smem row (bf16)

struct ScoreTiledSmem {
  bf16 q[kScT * kIdxHeads][kScP];
  bf16 k[2][kScoreTile][kScP];
  float w[kScT][kIdxHeads];
  int ncand[kScT];
};

__device__ __forceinline__ void sc_mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void sc_ldsm4(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(p))));
}
__device__ __forceinline__ void sc_cp16(void* dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(
                   static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
               "l"(src), "r"(valid ? 16 : 0));
}

__global__ void __launch_bounds__(256) kpool_score_tiled_kernel(
    float* __restrict__ scores, const bf16* __restrict__ q_idx, const float* __restrict__ w,
    int ld_w, const bf16* __restrict__ idx_cache, const MlaTokMeta meta, const int2* tiles,
    int row0, int stride) {
  pdl_entry();
  extern __shared__ __align__(128) uint8_t smem_raw[];
  ScoreTiledSmem& sm = *reinterpret_cast<ScoreTiledSmem*>(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int2 tile = tiles[blockIdx.x];
  const int t0 = tile.x, nt = min(tile.y, kScT);
  if (tid < kScT) sm.ncand[tid] = tid < nt ? (meta.positions[t0 + tid] + 1) / kKpool : 0;
  __syncthreads();
  int max_nc = 0;
#pragma unroll
  for (int j = 0; j < kScT; ++j) max_nc = max(max_nc, sm.ncand[j]);
  if (max_nc <= kSelPools) return;  // no sparse token in the tile
  const int ntiles = cdiv(max_nc, kScoreTile);
  if ((int)blockIdx.y >= ntiles) return;
  const int b = meta.token_seq[t0];
  // pool-key tile pt -> stage buffer (16-byte cp.async chunks; pools past max_nc read as zeros)
  auto load_keys = [&](int pt, int stage) {
    const int j0 = pt * kScoreTile;
    for (int i = tid; i < kScoreTile * (kIdxDim / 8); i += 256) {
      const int pj = i / (kIdxDim / 8), c = i % (kIdxDim / 8);
      const int j = j0 + pj;
      const bool ok = j < max_nc;
      const bf16* src = ok ? idx_cache + pool_slot(meta, b, j) * kIdxDim + c * 8 : idx_cache;
      sc_cp16(&sm.k[stage][pj][c * 8], src, ok);
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };
  load_keys(blockIdx.y, 0);
  for (int i = tid; i < kScT * kIdxHeads * (kIdxDim / 8); i += 256) {
    const int r = i / (kIdxDim / 8), c = i % (kIdxDim / 8);
    const int j = r / kIdxHeads;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (j < nt) v = reinterpret_cast<const uint4*>(q_idx + ((size_t)(t0 + j) * kIdxHeads + r % kIdxHeads) * kIdxDim)[c];
    *reinterpret_cast<uint4*>(&sm.q[r][c * 8]) = v;
  }
  for (int i = tid; i < kScT * kIdxHeads; i += 256) {
    const int j = i / kIdxHeads, h = i % kIdxHeads;
    sm.w[j][h] = j < nt ? w[(size_t)(t0 + j) * ld_w + h] * rsqrtf((float)kIdxHeads) : 0.f;
  }
  const float qs = rsqrtf((float)kIdxDim);
  const int my_nc = sm.ncand[warp];
  const bool active = warp < nt && my_nc > kSelPools;
  const int g = lane >> 2, tq = lane & 3;
  float* out_row = scores + (size_t)(t0 + warp - row0) * stride;
  int stage = 0;
  for (int pt = blockIdx.y; pt < ntiles; pt += gridDim.y, stage ^= 1) {
    const int j0 = pt * kScoreTile;
    const int nxt = pt + gridDim.y;
    if (nxt < ntiles) {
      load_keys(nxt, stage ^ 1);  // (the other stage was consumed before the last barrier)
      asm volatile("cp.async.wait_group 1;\n" ::);
    } else {
      asm volatile("cp.async.wait_group 0;\n" ::);
    }
    __syncthreads();  // this stage's keys (and on the first tile the queries) are visible
    if (active && j0 < my_nc) {
      float acc[2][8][4];
#pragma unroll
      for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 8; ++ni) acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < kIdxDim / 16; ++ks) {
        uint32_t a[2][4];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
          sc_ldsm4(a[mi], &sm.q[warp * kIdxHeads + mi * 16 + (lane & 15)][ks * 16 + (lane >> 4) * 8]);
#pragma unroll
        for (int np = 0; np < 4; ++np) {
          uint32_t bb[4];
          sc_ldsm4(bb, &sm.k[stage][np * 16 + (lane & 7) + ((lane >> 4) << 3)][ks * 16 + ((lane >> 3) & 1) * 8]);
#pragma unroll
          for (int mi = 0; mi < 2; ++mi) {
            sc_mma(acc[mi][2 * np], a[mi], bb[0], bb[1]);
            sc_mma(acc[mi][2 * np + 1], a[mi], bb[2], bb[3]);
          }
        }
      }
      // score[pool] = sum_h relu(acc[h][pool] * qs) * w[h]: lane rows h = 16 mi + g (+8)
      const float w0 = sm.w[warp][g], w1 = sm.w[warp][g + 8];
      const float w2 = sm.w[warp][16 + g], w3 = sm.w[warp][24 + g];
#pragma unroll
      for (int ni = 0; ni < 8; ++ni) {
        float s0 = fmaxf(acc[0][ni][0] * qs, 0.f) * w0 + fmaxf(acc[0][ni][2] * qs, 0.f) * w1 +
                   fmaxf(acc[1][ni][0] * qs, 0.f) * w2 + fmaxf(acc[1][ni][2] * qs, 0.f) * w3;
        float s1 = fmaxf(acc[0][ni][1] * qs, 0.f) * w0 + fmaxf(acc[0][ni][3] * qs, 0.f) * w1 +
                   fmaxf(acc[1][ni][1] * qs, 0.f) * w2 + fmaxf(acc[1][ni][3] * qs, 0.f) * w3;
#pragma unroll
        for (int o = 4; o < 32; o <<= 1) {
          s0 += __shfl_xor_sync(0xffffffff, s0, o);
          s1 += __shfl_xor_sync(0xffffffff, s1, o);
        }
        if (g == 0) {
          const int p = j0 + ni * 8 + 2 * tq;
          if (p < my_nc) out_row[p] = s0;
          if (p + 1 < my_nc) out_row[p + 1] = s1;
        }
      }
    }
    __syncthreads();  // this stage is consumed before the next iteration refills it
  }
}

// ------------------------------------------------------------------ top-k (radix select)
__device__ __forceinline__ uint32_t f2key(float f) {
  uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// One CTA selects the top 512 of a token's n pool scores: four 8-bit radix passes find the
// 512th largest key, a final pass writes the pools above it and the needed ties.
__device__ void topk_radix_cta(int* __restrict__ out, const float* __restrict__ sc, int n) {
  const int tid = threadIdx.x;
  __shared__ int hist[256];
  __shared__ uint32_t s_prefix, s_mask;
  __shared__ int s_need, s_cnt_gt, s_cnt_eq;
  if (tid == 0) {
    s_prefix = 0;
    s_mask = 0;
    s_need = kSelPools;
  }
  __shared__ int wsum[8];
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int i = tid; i < 256; i += blockDim.x) hist[i] = 0;
    __syncthreads();
    const uint32_t prefix = s_prefix, mask = s_mask;
    for (int j = tid; j < n; j += blockDim.x) {
      const uint32_t key = f2key(sc[j]);
      if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255], 1);
    }
    __syncthreads();
    // the bin holding the need-th largest key: inclusive suffix sums over bins 255..0 (thread
    // i < 256 owns bin 255 - i), two-level warp scan
    const int need = s_need;
    int c = 0, incl = 0;
    if (tid < 256) {
      c = hist[255 - tid];
      incl = c;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(0xffffffff, incl, o);
        if ((tid & 31) >= o) incl += u;
      }
      if ((tid & 31) == 31) wsum[tid >> 5] = incl;
    }
    __syncthreads();
    if (tid < 256) {
      for (int w = 0; w < (tid >> 5); ++w) incl += wsum[w];
      const int excl = incl - c;
      if (excl < need && incl >= need) {
        const int bin = 255 - tid;
        s_need = need - excl;
        s_prefix = prefix | ((uint32_t)bin << shift);
        s_mask = mask | (255u << shift);
      }
    }
    __syncthreads();
  }
  const uint32_t thr = s_prefix;
  const int n_eq = s_need, n_gt = kSelPools - s_need;
  if (tid == 0) {
    s_cnt_gt = 0;
    s_cnt_eq = 0;
  }
  __syncthreads();
  for (int j = tid; j < n; j += blockDim.x) {
    const uint32_t key = f2key(sc[j]);
    if (key > thr) {
      out[atomicAdd(&s_cnt_gt, 1)] = j;
    } else if (key == thr) {
      const int e = atomicAdd(&s_cnt_eq, 1);
      if (e < n_eq) out[n_gt + e] = j;
    }
  }
}

// Prefill-sized selections (many tokens, long rows): two passes over the scores. Pass 1 builds a
// 12-bit histogram of the keys and finds the bin holding the 512th largest; pass 2 writes the
// pools of higher bins and gathers the boundary bin's candidates into shared memory, where three
// radix passes over their low 20 bits (7 + 7 + 6) finish the selection. A boundary bin with more
// than kTk2Cap candidates falls back to topk_radix_cta. KPT > 0 (decode, rows of at most
// KPT x 1024 pools): the keys come from registers loaded once (keys[r]: pool tid + 1024 r) instead
// of two reads of the scores.
constexpr int kTk2Bits = 12, kTk2Bins = 1 << kTk2Bits, kTk2Cap = 5632;
struct Tk2Smem {
  uint32_t buf[2 * kTk2Cap];  // pass 1: histogram [kTk2Bins]; then the candidates' keys | pools
  int h2[128];
  int wsum[32];
  int s_bin, s_need, s_ccnt, s_cnt, s_cand;
  uint32_t s_prefix, s_mask;
};
// f(j, key) for every pool j < n of the calling thread
template <int KPT, typename F>
__device__ __forceinline__ void tk2_keys(const uint32_t* keys, const float* sc, int n, F&& f) {
  if constexpr (KPT > 0) {
#pragma unroll
    for (int r = 0; r < KPT; ++r) {
      const int j = (int)threadIdx.x + r * 1024;
      if (j < n) f(j, keys[r]);
    }
  } else {  // batches of kU loads in flight per thread (one load, then its atomic, per pool would wait
             // for every load in turn)
    constexpr int kU = 8;
    const int st = blockDim.x;
    int j = threadIdx.x;
    for (; j + (kU - 1) * st < n; j += kU * st) {
      float v[kU];
#pragma unroll
      for (int u = 0; u < kU; ++u) v[u] = sc[j + u * st];
#pragma unroll
      for (int u = 0; u < kU; ++u) f(j + u * st, f2key(v[u]));
    }
    for (; j < n; j += st) f(j, f2key(sc[j]));
  }
}
template <int KPT>
__device__ __forceinline__ void topk2_select(Tk2Smem& sm, int* __restrict__ out,
                                             const float* __restrict__ sc, int n,
                                             const uint32_t* keys) {
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  int* hist = reinterpret_cast<int*>(sm.buf);
  uint32_t* ckey = sm.buf;
  int* cidx = reinterpret_cast<int*>(sm.buf + kTk2Cap);
  for (int i = tid; i < kTk2Bins; i += blockDim.x) hist[i] = 0;
  __syncthreads();
  tk2_keys<KPT>(keys, sc, n, [&](int, uint32_t key) { atomicAdd(&hist[key >> (32 - kTk2Bits)], 1); });
  __syncthreads();
  // boundary bin: thread i owns bins 4095 - 4i - r (r = 0..3, descending); block-wide scan
  int c[4], local = 0;
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    c[r] = hist[kTk2Bins - 1 - 4 * tid - r];
    local += c[r];
  }
  int incl = local;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int u = __shfl_up_sync(0xffffffff, incl, o);
    if (lane >= o) incl += u;
  }
  if (lane == 31) sm.wsum[wid] = incl;
  __syncthreads();
  if (wid == 0) {
    const int v = sm.wsum[lane];
    int inc = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int u = __shfl_up_sync(0xffffffff, inc, o);
      if (lane >= o) inc += u;
    }
    sm.wsum[lane] = inc - v;  // warps before this one
  }
  __syncthreads();
  incl += sm.wsum[wid];
  const int excl = incl - local;
  if (excl < kSelPools && incl >= kSelPools) {
    int e = excl;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      if (e + c[r] >= kSelPools) {
        sm.s_bin = kTk2Bins - 1 - 4 * tid - r;
        sm.s_need = kSelPools - e;
        sm.s_ccnt = c[r];
        break;
      }
      e += c[r];
    }
  }
  if (tid == 0) {
    sm.s_cnt = 0;
    sm.s_cand = 0;
  }
  __syncthreads();  // (every thread has read hist: the buffer can hold candidates now)
  const int bin = sm.s_bin, need1 = sm.s_need, ccnt = sm.s_ccnt, n_gt1 = kSelPools - need1;
  if (ccnt > kTk2Cap) {
    topk_radix_cta(out, sc, n);
    return;
  }
  // pass 2: higher bins are selected; the boundary bin's keys go to shared memory
  tk2_keys<KPT>(keys, sc, n, [&](int j, uint32_t key) {
    const int d = (int)(key >> (32 - kTk2Bits));
    if (d > bin) {
      out[atomicAdd(&sm.s_cnt, 1)] = j;
    } else if (d == bin) {
      const int q = atomicAdd(&sm.s_cand, 1);
      ckey[q] = key;
      cidx[q] = j;
    }
  });
  if (tid == 0) {
    sm.s_prefix = (uint32_t)bin << (32 - kTk2Bits);
    sm.s_mask = ~0u << (32 - kTk2Bits);
  }
  __syncthreads();
  // the need1 largest candidates: radix passes over bits [19, 13], [12, 6], [5, 0]
#pragma unroll 1
  for (int pi = 0; pi < 3; ++pi) {
    const int shift = pi == 0 ? 13 : (pi == 1 ? 6 : 0);
    const int nb = pi == 2 ? 64 : 128;
    if (tid < 128) sm.h2[tid] = 0;
    __syncthreads();
    const uint32_t prefix = sm.s_prefix, mask = sm.s_mask;
    for (int q = tid; q < ccnt; q += blockDim.x) {
      const uint32_t key = ckey[q];
      if ((key & mask) == prefix) atomicAdd(&sm.h2[(key >> shift) & (nb - 1)], 1);
    }
    __syncthreads();
    if (wid == 0) {  // lane owns bins 127 - 4 lane - r (descending)
      const int need = pi == 0 ? need1 : sm.s_need;
      int cc[4], loc = 0;
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int b = 127 - 4 * lane - r;
        cc[r] = b < nb ? sm.h2[b] : 0;
        loc += cc[r];
      }
      int inc = loc;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(0xffffffff, inc, o);
        if (lane >= o) inc += u;
      }
      const int ex = inc - loc;
      if (ex < need && inc >= need) {
        int e = ex;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          if (e + cc[r] >= need) {
            const int b = 127 - 4 * lane - r;
            sm.s_need = need - e;
            sm.s_prefix = prefix | ((uint32_t)b << shift);
            sm.s_mask = mask | ((uint32_t)(nb - 1) << shift);
            break;
          }
          e += cc[r];
        }
      }
    }
    __syncthreads();
  }
  const uint32_t thr = sm.s_prefix;
  const int n_eq = sm.s_need, n_gt2 = need1 - n_eq;
  if (tid == 0) {
    sm.s_cnt = 0;
    sm.s_cand = 0;
  }
  __syncthreads();
  for (int q = tid; q < ccnt; q += blockDim.x) {
    const uint32_t key = ckey[q];
    if (key > thr) {
      out[n_gt1 + atomicAdd(&sm.s_cnt, 1)] = cidx[q];
    } else if (key == thr) {
      const int e = atomicAdd(&sm.s_cand, 1);
      if (e < n_eq) out[n_gt1 + n_gt2 + e] = cidx[q];
    }
  }
}

__global__ void __launch_bounds__(1024) topk_select_2pass_kernel(int* __restrict__ sel,
                                                                 const float* __restrict__ scores,
                                                                 const MlaTokMeta meta,
                                                                 int stride) {
  pdl_entry();
  const int t = blockIdx.x;
  const int n = (meta.positions[t] + 1) / kKpool;
  if (n <= kSelPools) return;
  __shared__ Tk2Smem sm;
  topk2_select<0>(sm, sel + (size_t)t * kSelPools, scores + (size_t)t * stride, n, nullptr);
}

// Decode-sized selections: one 1024-thread CTA per token, the token's keys read once into
// registers when its row has at most KPT x 1024 pools (else the two global passes above); the
// position is read before the PDL wait (uploaded before the step).
template <int KPT>
__global__ void __launch_bounds__(1024, 1) topk_select_reg_kernel(int* __restrict__ sel,
                                                                  const float* __restrict__ scores,
                                                                  const MlaTokMeta meta,
                                                                  int stride) {
  pdl_trigger();
  const int t = blockIdx.x;
  int pos = meta.pre ? meta.positions[t] : 0;
  pdl_wait();
  if (!meta.pre) pos = meta.positions[t];
  const int n = (pos + 1) / kKpool;
  if (n <= kSelPools) return;
  __shared__ Tk2Smem sm;
  const float* sc = scores + (size_t)t * stride;
  int* out = sel + (size_t)t * kSelPools;
  if (n > KPT * 1024) {
    topk2_select<0>(sm, out, sc, n, nullptr);
    return;
  }
  uint32_t keys[KPT];
#pragma unroll
  for (int r = 0; r < KPT; ++r) {
    const int j = (int)threadIdx.x + r * 1024;
    keys[r] = j < n ? f2key(sc[j]) : 0u;
  }
  topk2_select<KPT>(sm, out, sc, n, keys);
}

// The same radix select with one 8-CTA thread-block cluster per token (decode: few tokens with
// long pool lists): CTA r scans its eighth of the scores, each pass merges the eight 256-bin
// histograms through distributed shared memory so that every CTA takes the same decision, and the
// final pass places the selected pools through counters in CTA 0's shared memory. REG: a CTA slice
// of up to kTkRegKeys x 1024 pools (contexts up to 512K tokens) is read once into registers (keys)
// instead of once per pass (for few tokens: the extra registers keep the 1024-thread CTAs at one
// per SM, so many-token launches would need a second wave of clusters).
constexpr int kTkCluster = 8;
constexpr int kTkRegKeys = 16;
// the cluster's selection of one token's n > kSelPools pools (every CTA of the cluster calls it)
template <bool REG>
__device__ __forceinline__ void topk_cluster_select(int* __restrict__ out, const float* __restrict__ sc,
                                                    int n) {
  cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
  const int rank = (int)cluster.block_rank();
  const int tid = threadIdx.x;
  const int per = cdiv(n, kTkCluster);
  const int j0 = rank * per, j1 = min(n, j0 + per);
  const bool in_regs = REG && per <= kTkRegKeys * 1024;  // (uniform over the cluster)
  uint32_t keys[REG ? kTkRegKeys : 1];
  if (in_regs) {
#pragma unroll
    for (int r = 0; r < (REG ? kTkRegKeys : 1); ++r) {
      const int j = j0 + tid + r * 1024;
      keys[r] = j < j1 ? f2key(sc[j]) : 0u;  // (key 0: below every float's key)
    }
  }
  // per pass a histogram buffer of its own parity: a CTA clears buffer p % 2 at pass p only after
  // the pass p - 1 barrier, which every CTA reaches after its pass p - 2 reads of that buffer, so
  // one cluster barrier per pass suffices (the decision is taken by every CTA itself)
  __shared__ int hist[2][256];
  __shared__ uint32_t s_prefix, s_mask;
  __shared__ int s_need, s_cnt_gt, s_cnt_eq;
  __shared__ int wsum[8];
  if (tid == 0) {
    s_prefix = 0;
    s_mask = 0;
    s_need = kSelPools;
    s_cnt_gt = 0;
    s_cnt_eq = 0;
  }
  const int rmax = (j1 - j0 + 1023) >> 10;  // register keys in use (uniform)
#pragma unroll 1
  for (int shift = 24, pass = 0; shift >= 0; shift -= 8, ++pass) {
    int* h = hist[pass & 1];
    if (tid < 256) h[tid] = 0;
    __syncthreads();
    const uint32_t prefix = s_prefix, mask = s_mask;
    if (in_regs) {
#pragma unroll
      for (int r = 0; r < (REG ? kTkRegKeys : 1); ++r) {
        if (r < rmax) {
          const int j = j0 + tid + r * 1024;
          const bool ok = j < j1 && (keys[r] & mask) == prefix;
          const uint32_t a = (uint32_t)__cvta_generic_to_shared(&h[(keys[r] >> shift) & 255]);
          asm volatile("{ .reg .pred q; setp.ne.u32 q, %1, 0; @q red.shared.add.u32 [%0], 1; }" ::
                           "r"(a), "r"((uint32_t)ok) : "memory");
        }
      }
    } else {
      for (int j = j0 + tid; j < j1; j += blockDim.x) {
        const uint32_t key = f2key(sc[j]);
        if ((key & mask) == prefix) atomicAdd(&h[(key >> shift) & 255], 1);
      }
    }
    cluster.sync();  // every CTA's histogram is complete
    const int need = s_need;
    int c = 0, incl = 0;
    if (tid < 256) {
      int v[kTkCluster];
      const uint32_t la = (uint32_t)__cvta_generic_to_shared(&h[255 - tid]);
#pragma unroll
      for (int q = 0; q < kTkCluster; ++q) {
        uint32_t ra;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(la), "r"(q));
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(v[q]) : "r"(ra) : "memory");
      }
#pragma unroll
      for (int q = 0; q < kTkCluster; ++q) c += v[q];
      incl = c;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(0xffffffff, incl, o);
        if ((tid & 31) >= o) incl += u;
      }
      if ((tid & 31) == 31) wsum[tid >> 5] = incl;
    }
    __syncthreads();
    if (tid < 256) {
      for (int w = 0; w < (tid >> 5); ++w) incl += wsum[w];
      const int excl = incl - c;
      if (excl < need && incl >= need) {
        const int bin = 255 - tid;
        s_need = need - excl;
        s_prefix = prefix | ((uint32_t)bin << shift);
        s_mask = mask | (255u << shift);
      }
    }
  }
  __syncthreads();  // (the last decision)
  const uint32_t thr = s_prefix;
  const int n_eq = s_need, n_gt = kSelPools - s_need;
  int* cnt_gt = cluster.map_shared_rank(&s_cnt_gt, 0);
  int* cnt_eq = cluster.map_shared_rank(&s_cnt_eq, 0);
  auto place = [&](int j, uint32_t key) {
    if (key > thr) {
      out[atomicAdd(cnt_gt, 1)] = j;
    } else if (key == thr) {
      const int e = atomicAdd(cnt_eq, 1);
      if (e < n_eq) out[n_gt + e] = j;
    }
  };
  if (in_regs) {
#pragma unroll
    for (int r = 0; r < (REG ? kTkRegKeys : 1); ++r) {
      const int j = j0 + tid + r * 1024;
      if (j < j1) place(j, keys[r]);
    }
  } else {
    for (int j = j0 + tid; j < j1; j += blockDim.x) place(j, f2key(sc[j]));
  }
  cluster.sync();  // CTA 0's counters stay alive until the whole cluster is done
}

// Decode selection with a bounded score range (topk_range_kernel): one 8-CTA cluster of 1024 threads per
// token (the cluster select's footprint), 4 keys per thread in registers for rows of up to kTrMax pools
// (longer rows, or a boundary bin beyond kTrCap keys: topk_cluster_select). Pools are dealt in chunks of 32
// (a line per warp load): chunk g -> CTA g % 8, warp (g / 8) % 32, key g / 256 of the thread.
// 1. A lower bound of the 512th largest key: every warp's 2nd largest key is reached by 2 keys of its own,
//    so the smallest over the cluster's 256 warps by >= 512; with the cluster's maximum: the range [lb, mx].
// 2. A 256-bin histogram of the range per CTA (equal score widths; key bits when the scores are not
//    finite); each key's histogram atomic also gives its slot in (CTA, bin), and the first kTrBk keys of
//    every (CTA, bin) are stored by slot with it, before the histogram's cluster barrier. After that barrier
//    every CTA merges the histograms through distributed shared memory: the bin of the 512th largest and per
//    bin the output offset of this CTA's keys
//    (higher bins of every CTA, then this bin's keys of the CTAs before it). Higher bins are written out at
//    offset + slot; CTA 0 reads the boundary bin's keys of every CTA from their stores and ranks them. No
//    third cluster barrier: each CTA arrives once it reads no other CTA's shared memory any more and waits
//    for the others only before it exits (a CTA with more than kTrBk boundary keys: the radix select).
// Small executed code on purpose (it runs with a cold instruction cache). Same selected set as
// the other selects (order and the choice among keys equal to the 512th arbitrary in all of them).
constexpr int kTrKPT = 4;
constexpr int kTrRound = kTkCluster * 1024;  // pools per key slot (8192)
constexpr int kTrMax = kTrKPT * kTrRound;    // 32768 (pool indices fit 16 bits)
constexpr int kTrBk = 24;                    // keys stored per (CTA, bin)
struct TrSmem {
  uint32_t bkk[256 * kTrBk];         // the CTA's keys by (bin, slot)
  uint16_t bki[256 * kTrBk];         // their pools
  uint32_t gk[kTkCluster * kTrBk];   // CTA 0: the boundary bin's keys of every CTA
  int gi[kTkCluster * kTrBk];
  int hist[256];
  int off[256];                      // output offset of the CTA's keys of each bin
  int bpre[kTkCluster + 1];          // boundary keys of the CTAs before q
  int wsc[8];
  uint32_t wlb[32], wmx[32];
  uint32_t c_lb, c_mx, g_lb, g_mx;
  int s_bin, s_need, s_c, s_bmax;
};

// (NI instance) the range select's radix fallbacks (rows beyond kTrMax pools, a crowded boundary bin) as a call:
// inlined twice they make most of the kernel's instructions, and the hot path's code (fetched from L2 at every call:
// it runs with a cold instruction cache) is spread over them
__device__ __noinline__ void topk_range_fallback(int* __restrict__ out, const float* __restrict__ sc, int n) {
  topk_cluster_select<true>(out, sc, n);
}

__device__ __forceinline__ float tr_key2f(uint32_t k) {
  return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

// LR (topk_range_kernel's LONG instance): rows of kTrMax < n <= kTrLMax pools by the same range select with up to
// kTrLKPT keys per thread instead of the radix select's 4 cluster passes. The same deal of the
// 32-pool chunks (chunk g -> CTA g % 8, warp (g / 8) % 32, key g / 256 of the thread), bounds, bins, slots and barriers as
// the fast path; per key one register (bin << 16 | slot, -1 below the range) instead of the bk / sl pairs (the keys are
// dead after the binning); the stored boundary keys' pools as CTA-local 16-bit ids (key i * 1024 + thread), which CTA 0
// maps back with the owner's rank; the scores by __ldcg (the PDL predecessor's output). A crowded boundary bin: the
// radix select, as the fast path. Same selected set; the order (slots of shared-memory atomics) as arbitrary as the
// fast path's.
constexpr int kTrLKPT = 16;
constexpr int kTrLMax = kTrLKPT * kTrRound;  // 131072 pools (524K tokens)
static_assert(kTrLKPT * 1024 <= 65536, "LR: CTA-local pool ids in 16 bits");

template <bool NI>
__device__ __forceinline__ void tr_long_select(TrSmem& sm, int* __restrict__ out, const float* __restrict__ sc, int n) {
  cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
  const int c = (int)cluster.block_rank();
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const int base = (wid * kTkCluster + c) * 32 + lane;
  const int rmax = (n + kTrRound - 1) / kTrRound;  // key slots in use (cluster-uniform, > kTrKPT): the loops below end
                                                   // there (the unused slots' code skipped, not predicated off)
  uint32_t k[kTrLKPT];
#pragma unroll
  for (int i = 0; i < kTrLKPT; ++i) {
    const int p = i * kTrRound + base;
    k[i] = p < n ? f2key(__ldcg(sc + p)) : 0u;  // (key 0: below every float's key)
  }
  // 1. the warp's two largest keys (each thread's two, then a butterfly merge): every warp holds >= 128 valid keys.
  // lo = the smallest warp's second largest (>= 512 keys >= lo), hi = the largest warp's second largest (<= 256 keys > hi:
  // an open top bin, as SEL's, so that a few outlying scores do not stretch the bins until the boundary bin crowds)
  uint32_t a = 0, b = 0;
#pragma unroll
  for (int i = 0; i < kTrLKPT; ++i) {
    if (i >= rmax) break;
    b = max(b, min(a, k[i]));
    a = max(a, k[i]);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const uint32_t a2 = __shfl_xor_sync(0xffffffffu, a, o), b2 = __shfl_xor_sync(0xffffffffu, b, o);
    b = max(max(b, b2), min(a, a2));
    a = max(a, a2);
  }
  if (lane == 0) {
    sm.wlb[wid] = b;
    sm.wmx[wid] = b;
  }
  if (tid < 256) sm.hist[tid] = 0;
  __syncthreads();
  if (wid == 0) {
    uint32_t x = sm.wlb[lane], y = sm.wmx[lane];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      x = min(x, __shfl_xor_sync(0xffffffffu, x, o));
      y = max(y, __shfl_xor_sync(0xffffffffu, y, o));
    }
    if (lane == 0) {
      sm.c_lb = x;
      sm.c_mx = y;
    }
  }
  cluster.sync();  // every CTA's bound and maximum
  if (wid == 0) {
    uint32_t x = 0xffffffffu, y = 0u;
    if (lane < kTkCluster) {
      x = *cluster.map_shared_rank(&sm.c_lb, lane);
      y = *cluster.map_shared_rank(&sm.c_mx, lane);
    }
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) {
      x = min(x, __shfl_xor_sync(0xffffffffu, x, o));
      y = max(y, __shfl_xor_sync(0xffffffffu, y, o));
    }
    if (lane == 0) {
      sm.g_lb = x;
      sm.g_mx = y;
    }
  }
  __syncthreads();
  const uint32_t lo = sm.g_lb, hi = sm.g_mx;
  const float lov = tr_key2f(lo), hiv = tr_key2f(hi);
  const float inv = 255.f / (hiv - lov);
  const bool vb = isfinite(lov) && isfinite(hiv) && hiv > lov && isfinite(inv);
  const uint32_t d = hi - lo;
  const int s = d < 255u ? 0 : 32 - __clz(d / 255u);
  int bs[kTrLKPT];  // bin << 16 | slot (-1: not in the range); bins 0-254 over [lo, hi], 255: above hi
#pragma unroll
  for (int i = 0; i < kTrLKPT; ++i) {
    if (i >= rmax) break;
    const int bk = i * kTrRound + base >= n || k[i] < lo ? -1
                   : k[i] > hi ? 255
                   : vb ? min(254, (int)((tr_key2f(k[i]) - lov) * inv)) : min(254, (int)((k[i] - lo) >> s));
    bs[i] = -1;
    if (bk >= 0) {
      const int sl = atomicAdd(&sm.hist[bk], 1);
      if (sl < kTrBk) {
        sm.bkk[bk * kTrBk + sl] = k[i];
        sm.bki[bk * kTrBk + sl] = (uint16_t)(i * 1024 + tid);
      }
      bs[i] = bk << 16 | sl;
    }
  }
  cluster.sync();  // every CTA's histogram and keys by slot
  // 2. thread t < 256: bin 255 - t in every CTA; block scan over the descending bins of the totals
  if (tid < 256) {
    int x[kTkCluster];
#pragma unroll
    for (int q = 0; q < kTkCluster; ++q) x[q] = cluster.map_shared_rank(sm.hist, q)[255 - tid];
    int tot = 0, pre = 0, xm = 0;
#pragma unroll
    for (int q = 0; q < kTkCluster; ++q) {
      tot += x[q];
      pre += q < c ? x[q] : 0;
      xm = max(xm, x[q]);
    }
    int it = tot;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int u = __shfl_up_sync(0xffffffffu, it, o);
      if (lane >= o) it += u;
    }
    if (lane == 31) sm.wsc[wid] = it;
    asm volatile("bar.sync 1, 256;\n" ::: "memory");
    for (int w2 = 0; w2 < wid; ++w2) it += sm.wsc[w2];
    sm.off[255 - tid] = it - tot + pre;  // (higher bins of every CTA, this bin's of CTAs < c)
    if (it - tot < kSelPools && it >= kSelPools) {  // the boundary bin
      sm.s_bin = 255 - tid;
      sm.s_need = kSelPools - (it - tot);
      sm.s_c = tot;
      sm.s_bmax = xm;
      int a2 = 0;
#pragma unroll
      for (int q = 0; q < kTkCluster; ++q) {
        sm.bpre[q] = a2;
        a2 += x[q];
      }
    }
  }
  __syncthreads();
  const int bin = sm.s_bin, need_b = sm.s_need, cb = sm.s_c;
  if (sm.s_bmax > kTrBk) {  // (cluster-uniform) a CTA's boundary keys not all stored: the radix select
    cluster.sync();         // (every CTA has read every histogram before the select reuses the cluster)
    if constexpr (NI)
      topk_range_fallback(out, sc, n);
    else
      topk_cluster_select<true>(out, sc, n);
    return;
  }
  if (c == 0) {  // the boundary bin's keys of every CTA (stored before the histogram barrier)
    if (tid < cb) {
      int q = 0;
#pragma unroll
      for (int r = 1; r < kTkCluster; ++r) q += tid >= sm.bpre[r];
      const int e = bin * kTrBk + tid - sm.bpre[q];
      sm.gk[tid] = *cluster.map_shared_rank(&sm.bkk[e], q);
      const int li = *cluster.map_shared_rank(&sm.bki[e], q);  // (key i * 1024 + thread of CTA q)
      sm.gi[tid] = (li >> 10) * kTrRound + (((li & 1023) >> 5) * kTkCluster + q) * 32 + (li & 31);
    }
    __syncthreads();
  }
  // this CTA reads no other CTA's shared memory any more
  asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
#pragma unroll
  for (int i = 0; i < kTrLKPT; ++i) {
    if (i >= rmax) break;
    if ((bs[i] >> 16) > bin) out[sm.off[bs[i] >> 16] + (bs[i] & 0xffff)] = i * kTrRound + base;
  }
  if (c == 0 && tid < cb) {  // the need_b largest boundary keys by rank (larger + equal at lower slots)
    const uint32_t my = sm.gk[tid];
    int rank = 0;
#pragma unroll 4
    for (int q = 0; q < cb; ++q) {
      const uint32_t x = sm.gk[q];
      rank += (x > my) | ((x == my) & (q < tid));
    }
    if (rank < need_b) out[kSelPools - need_b + rank] = sm.gi[tid];
  }
  // (no CTA exits while another may still read its shared memory)
  asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}

template <bool NI, bool LONG = false>
__global__ void __cluster_dims__(kTkCluster, 1, 1) __launch_bounds__(1024)
    topk_range_kernel(int* __restrict__ sel, const float* __restrict__ scores, const MlaTokMeta meta, int stride) {
  pdl_trigger();
  const int t = blockIdx.x / kTkCluster;
  int pos = meta.pre ? meta.positions[t] : 0;  // (uploaded before the step)
  pdl_wait();
  if (!meta.pre) pos = meta.positions[t];
  const int n = (pos + 1) / kKpool;
  if (n <= kSelPools) return;  // (the whole cluster: same token)
  const float* sc = scores + (size_t)t * stride;
  int* out = sel + (size_t)t * kSelPools;
  __shared__ TrSmem sm;
  if (n > kTrMax) {
    if constexpr (LONG) {
      if (n <= kTrLMax) {
        tr_long_select<NI>(sm, out, sc, n);
        return;
      }
    }
    if constexpr (NI)
      topk_range_fallback(out, sc, n);
    else
      topk_cluster_select<true>(out, sc, n);
    return;
  }
  cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
  const int c = (int)cluster.block_rank();
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const int base = (wid * kTkCluster + c) * 32 + lane;
  uint32_t k[kTrKPT];
#pragma unroll
  for (int i = 0; i < kTrKPT; ++i) {
    const int p = i * kTrRound + base;
    k[i] = p < n ? f2key(sc[p]) : 0u;  // (key 0: below every float's key)
  }
  // 1. the warp's two largest keys (each thread's two, then a butterfly merge)
  uint32_t a = 0, b = 0;
#pragma unroll
  for (int i = 0; i < kTrKPT; ++i) {
    b = max(b, min(a, k[i]));
    a = max(a, k[i]);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const uint32_t a2 = __shfl_xor_sync(0xffffffffu, a, o), b2 = __shfl_xor_sync(0xffffffffu, b, o);
    b = max(max(b, b2), min(a, a2));
    a = max(a, a2);
  }
  if (lane == 0) {
    sm.wlb[wid] = b;
    sm.wmx[wid] = a;
  }
  if (tid < 256) sm.hist[tid] = 0;
  __syncthreads();
  if (wid == 0) {
    uint32_t x = sm.wlb[lane], y = sm.wmx[lane];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      x = min(x, __shfl_xor_sync(0xffffffffu, x, o));
      y = max(y, __shfl_xor_sync(0xffffffffu, y, o));
    }
    if (lane == 0) {
      sm.c_lb = x;
      sm.c_mx = y;
    }
  }
  cluster.sync();  // every CTA's bound and maximum
  if (wid == 0) {
    uint32_t x = 0xffffffffu, y = 0u;
    if (lane < kTkCluster) {
      x = *cluster.map_shared_rank(&sm.c_lb, lane);
      y = *cluster.map_shared_rank(&sm.c_mx, lane);
    }
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) {
      x = min(x, __shfl_xor_sync(0xffffffffu, x, o));
      y = max(y, __shfl_xor_sync(0xffffffffu, y, o));
    }
    if (lane == 0) {
      sm.g_lb = x;
      sm.g_mx = y;
    }
  }
  __syncthreads();
  const uint32_t lo = sm.g_lb, hi = sm.g_mx;
  const float lov = tr_key2f(lo), hiv = tr_key2f(hi);
  const float inv = 256.f / (hiv - lov);
  const bool vb = isfinite(lov) && isfinite(hiv) && hiv > lov && isfinite(inv);
  const uint32_t d = hi - lo;
  const int s = d < 256u ? 0 : 32 - __clz(d >> 8);
  int bk[kTrKPT], sl[kTrKPT];
#pragma unroll
  for (int i = 0; i < kTrKPT; ++i) {
    bk[i] = i * kTrRound + base >= n || k[i] < lo ? -1
            : vb ? min(255, (int)((tr_key2f(k[i]) - lov) * inv)) : (int)((k[i] - lo) >> s);
    sl[i] = 0;
    if (bk[i] >= 0) {
      sl[i] = atomicAdd(&sm.hist[bk[i]], 1);
      if (sl[i] < kTrBk) {
        sm.bkk[bk[i] * kTrBk + sl[i]] = k[i];
        sm.bki[bk[i] * kTrBk + sl[i]] = (uint16_t)(i * kTrRound + base);
      }
    }
  }
  cluster.sync();  // every CTA's histogram and keys by slot
  // 2. thread t < 256: bin 255 - t in every CTA; block scan over the descending bins of the totals
  if (tid < 256) {
    int x[kTkCluster];
#pragma unroll
    for (int q = 0; q < kTkCluster; ++q) x[q] = cluster.map_shared_rank(sm.hist, q)[255 - tid];
    int tot = 0, pre = 0, xm = 0;
#pragma unroll
    for (int q = 0; q < kTkCluster; ++q) {
      tot += x[q];
      pre += q < c ? x[q] : 0;
      xm = max(xm, x[q]);
    }
    int it = tot;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int u = __shfl_up_sync(0xffffffffu, it, o);
      if (lane >= o) it += u;
    }
    if (lane == 31) sm.wsc[wid] = it;
    asm volatile("bar.sync 1, 256;\n" ::: "memory");
    for (int w2 = 0; w2 < wid; ++w2) it += sm.wsc[w2];
    sm.off[255 - tid] = it - tot + pre;  // (higher bins of every CTA, this bin's of CTAs < c)
    if (it - tot < kSelPools && it >= kSelPools) {  // the boundary bin
      sm.s_bin = 255 - tid;
      sm.s_need = kSelPools - (it - tot);
      sm.s_c = tot;
      sm.s_bmax = xm;
      int a2 = 0;
#pragma unroll
      for (int q = 0; q < kTkCluster; ++q) {
        sm.bpre[q] = a2;
        a2 += x[q];
      }
    }
  }
  __syncthreads();
  const int bin = sm.s_bin, need_b = sm.s_need, cb = sm.s_c;
  if (sm.s_bmax > kTrBk) {  // (cluster-uniform) a CTA's boundary keys not all stored: the radix select
    cluster.sync();         // (every CTA has read every histogram before the select reuses the cluster)
    if constexpr (NI)
      topk_range_fallback(out, sc, n);
    else
      topk_cluster_select<true>(out, sc, n);
    return;
  }
  if (c == 0) {  // the boundary bin's keys of every CTA (stored before the histogram barrier)
    if (tid < cb) {
      int q = 0;
#pragma unroll
      for (int r = 1; r < kTkCluster; ++r) q += tid >= sm.bpre[r];
      const int e = bin * kTrBk + tid - sm.bpre[q];
      sm.gk[tid] = *cluster.map_shared_rank(&sm.bkk[e], q);
      sm.gi[tid] = *cluster.map_shared_rank(&sm.bki[e], q);
    }
    __syncthreads();
  }
  // this CTA reads no other CTA's shared memory any more
  asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
#pragma unroll
  for (int i = 0; i < kTrKPT; ++i)
    if (bk[i] > bin) out[sm.off[bk[i]] + sl[i]] = i * kTrRound + base;
  if (c == 0 && tid < cb) {  // the need_b largest boundary keys by rank (larger + equal at lower slots)
    const uint32_t my = sm.gk[tid];
    int rank = 0;
#pragma unroll 4
    for (int q = 0; q < cb; ++q) {
      const uint32_t x = sm.gk[q];
      rank += (x > my) | ((x == my) & (q < tid));
    }
    if (rank < need_b) out[kSelPools - need_b + rank] = sm.gi[tid];
  }
  // (no CTA exits while another may still read its shared memory)
  asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}

// Top-k launch by step size: below kTopkCtaMinT tokens (decode) the range select in 8-CTA clusters
// (topk_range_kernel's NI / LONG instance); from kTopkCtaMinT tokens up to the SM count (one wave) one CTA per token
// with the keys in registers (the cluster select's 8 CTAs per token stop fitting one wave from 16 tokens); more
// tokens: the two-pass select.
constexpr int kTopkCtaMinT = 13;
static void launch_topk(int* sel, const float* scores, const MlaTokMeta& meta, int T, int stride,
                        cudaStream_t st) {
  static const int treg_max = [] {
    int dev = 0, n = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
    return n;
  }();
  if (T >= kTopkCtaMinT && T <= treg_max) {
    launch_pdl(topk_select_reg_kernel<32>, dim3(T), dim3(1024), 0, st, sel, scores, meta, stride);
    return;
  }
  if (T < kTopkCtaMinT) {
    static bool loaded[64] = {};  // (the instance loaded at the first launch on each device)
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (!loaded[dev]) {
      cudaFuncAttributes fa;
      CUDA_CHECK(cudaFuncGetAttributes(&fa, topk_range_kernel<true, true>));
      loaded[dev] = true;
    }
    launch_pdl(topk_range_kernel<true, true>, dim3(T * kTkCluster), dim3(1024), 0, st, sel, scores, meta, stride);
  } else {
    launch_pdl(topk_select_2pass_kernel, dim3(T), dim3(1024), 0, st, sel, scores, meta, stride);
  }
}

void kpool_topk(int* sel, const float* scores, const MlaTokMeta& meta, int T, int stride,
                cudaStream_t st) {
  if (T == 0) return;
  launch_topk(sel, scores, meta, T, stride, st);
}

void kpool_select(int* sel, float* scores, const bf16* q_idx, const float* w, int ld_w,
                  const bf16* idx_cache, const MlaTokMeta& meta, int T, int max_pools,
                  int idx_heads, cudaStream_t st) {
  if (T == 0) return;
  CHECK(idx_heads == kIdxHeads, "indexer kernels specialized for 32 heads");
  // CTAs per token: enough to fill the GPU for few tokens, at most one per 64-pool tile
  static int sms = [] {
    int dev = 0, n = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
    return n;
  }();
  const int splits =
      std::max(kScoreSplit, std::min({64, (2 * sms) / T, (int)cdiv(max_pools, kScoreTile)}));
  dim3 grid(T, splits);
  launch_pdl(kpool_score_kernel, dim3(grid), dim3(128), 0, st, scores, q_idx, w, ld_w, idx_cache,
             meta, max_pools);
  launch_topk(sel, scores, meta, T, max_pools, st);
}

void kpool_select_tiled(int* sel, float* scores, const bf16* q_idx, const float* w, int ld_w,
                        const bf16* idx_cache, const MlaTokMeta& meta, const int2* tiles,
                        int ntiles, int row0, int ntok, int stride, int idx_heads,
                        cudaStream_t st, int64_t q_rows, int64_t idx_rows, cudaEvent_t score_done,
                        bool score_only) {
  if (ntiles == 0 || ntok == 0) return;
  CHECK(idx_heads == kIdxHeads, "indexer kernels specialized for 32 heads");
  if (kpool_score_umma(scores, q_idx, q_rows, w, ld_w, idx_cache, idx_rows, meta, tiles, ntiles,
                       row0, stride, st)) {
    if (score_done) CUDA_CHECK(cudaEventRecord(score_done, st));
    if (score_only) return;
    MlaTokMeta m = meta;
    m.token_seq += row0;
    m.positions += row0;
    launch_topk(sel + (size_t)row0 * kSelPools, scores, m, ntok, stride, st);
    return;
  }
  static int sms = [] {
    int dev = 0, n = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
    return n;
  }();
  const int smem = sizeof(ScoreTiledSmem);
  static bool configured[64] = {};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  if (!configured[dev]) {
    CUDA_CHECK(cudaFuncSetAttribute(kpool_score_tiled_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    configured[dev] = true;
  }
  // pool-tile slices per token tile: two CTAs per SM in all (2 fit by shared memory), at most
  // one per 64-pool tile of the longest row
  const int splits = std::max(1, std::min((int)cdiv(2 * sms, ntiles), (int)cdiv(stride, kScoreTile)));
  launch_pdl(kpool_score_tiled_kernel, dim3(ntiles, splits), dim3(256), smem, st, scores, q_idx, w,
             ld_w, idx_cache, meta, tiles, row0, stride);
  if (score_done) CUDA_CHECK(cudaEventRecord(score_done, st));
  if (score_only) return;
  MlaTokMeta m = meta;  // top-k over the tokens [row0, row0 + ntok), score rows relative
  m.token_seq += row0;
  m.positions += row0;
  launch_topk(sel + (size_t)row0 * kSelPools, scores, m, ntok, stride, st);
}

// ------------------------------------------------------------------ sparse attention
constexpr int kHM = 16;           // query heads per CTA (padded M)
constexpr int kTile = 64;         // cache rows per tile
constexpr int kChunks = kLatent / 8;  // 16B chunks per latent row (64)
constexpr int kSpStride = kTile + 4;

template <int NW>  // warps: each owns a 512 / NW slice of the latent dim (scores' K, the output's N)
struct AttnSmem {
  bf16 q[kHM * kLatent];               // 16 KB, swizzled rows
  bf16 kv[2][kTile * kLatent];         // 128 KB, swizzled rows
  float sp[NW][kHM * kSpStride];       // partial scores per warp
  bf16 p[kHM * kTile];                 // probabilities, swizzled rows
  float m[kHM], l[kHM], alpha[kHM];
  int64_t rows[(kMaxRows + kTile - 1) / kTile][kTile];  // every row tile of the split
};

// CL (clusters of 2..16 splits, 16 % nsplit == 0): row tables for at most kClTiles tiles per split, and the
// receive area of the heads this CTA finalizes (16 / nsplit heads x nsplit splits, pushed by the cluster)
constexpr int kClTiles = 17;  // ceil(33 row tiles / 2 splits)
template <int NW>
struct AttnSmemCl {
  bf16 q[kHM * kLatent];
  bf16 kv[2][kTile * kLatent];
  float sp[NW][kHM * kSpStride];
  bf16 p[kHM * kTile];
  float m[kHM], l[kHM], alpha[kHM];
  int64_t rows[kClTiles][kTile];
  float recv[kHM][kLatent];   // [head slot x nsplit + split][column]
  float rm[kHM], rl[kHM];     // [head slot x nsplit + split]
};

// TMA forms: every cache row of a tile is one bulk copy (cp.async.bulk, 1 KB) on the buffer's mbarrier instead of
// 64 cp.async per row, 8 rows per warp. The rows sit unswizzled at a 1040-byte pitch (16-byte rotation per row: the 8
// rows of an ldmatrix matrix hit distinct banks), the same values in the same MMA operands as the swizzled tiles:
// bitwise the cp.async forms.
constexpr int kKvP = kLatent + 8;  // row pitch (elements)
template <int NW>
struct AttnSmemT {
  bf16 q[kHM * kLatent];
  bf16 kv[2][kTile * kKvP];
  float sp[NW][kHM * kSpStride];
  bf16 p[kHM * kTile];
  float m[kHM], l[kHM], alpha[kHM];
  int64_t rows[(kMaxRows + kTile - 1) / kTile][kTile];
  uint64_t bar[2];
};
template <int NW>
struct AttnSmemClT {
  bf16 q[kHM * kLatent];
  bf16 kv[2][kTile * kKvP];
  float sp[NW][kHM * kSpStride];
  bf16 p[kHM * kTile];
  float m[kHM], l[kHM], alpha[kHM];
  int64_t rows[kClTiles][kTile];
  float recv[kHM][kLatent];
  float rm[kHM], rl[kHM];
  uint64_t bar[2];
};
static_assert(sizeof(AttnSmemClT<8>) <= 232448, "cluster attention: shared memory");
// CLW (the FA cluster form at 2..16 splits that need not divide the 16 heads: 8 tokens' clusters of 10 CTAs fit at
// one CTA per SM, 16-CTA clusters only 7): the receive area of up to two heads per CTA (head h: CTA h % nsplit, slot
// (h / nsplit) x nsplit + split) in place of the latent-slice score exchange the FA form does not use
template <int NW>
struct AttnSmemClW {
  bf16 q[kHM * kLatent];
  bf16 kv[2][kTile * kKvP];
  float sp[2 * NW * kHM];  // (FA: the row maxima and probability sums per warp)
  bf16 p[kHM * kTile];
  float m[kHM], l[kHM], alpha[kHM];
  int64_t rows[kClTiles][kTile];
  float recv[2 * kHM][kLatent];
  float rm[2 * kHM], rl[2 * kHM];
  uint64_t bar[3];  // (bar[2]: the SEL instance's partial first tile)
  // (SEL) the selection's mbarriers (bounds, histograms, boundary keys, the list, boundary bin ranges), not aliased: in
  // the first KV buffer with the rest of the selection's scratch they would still be valid mbarrier objects (no
  // mbarrier.inval) when the first row tile's bulk copies overwrite them, and mbarrier state could land on those 40
  // bytes after the tile has (e.g. under GPU time-slicing): corrupt latent values of the first row of every split
  uint64_t skmb[5];
};
static_assert(sizeof(AttnSmemClW<8>) <= 232448, "cluster attention (CLW): shared memory");

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}

__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)),
               "l"(src), "r"(valid ? 16 : 0));
}

// (TMA forms) mbarrier + bulk copy helpers
__device__ __forceinline__ void at_bar_init(uint64_t* b) {
  asm volatile("mbarrier.init.shared.b64 [%0], 1;\n" ::"r"(smem_u32(b)));
}
__device__ __forceinline__ void at_bar_expect(uint64_t* b, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(smem_u32(b)), "r"(tx) : "memory");
}
__device__ __forceinline__ void at_bulk(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
          smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(b))
      : "memory");
}
__device__ __forceinline__ void at_bar_wait(uint64_t* b, uint32_t ph) {
  uint32_t ok = 0;
  while (!ok)
    asm volatile(
        "{ .reg .pred P; mbarrier.try_wait.parity.shared.b64 P, [%1], %2; selp.u32 %0, 1, 0, P; }\n"
        : "=r"(ok)
        : "r"(smem_u32(b)), "r"(ph)
        : "memory");
}

// ---- SEL (sparse_mla_attn_sel): the decode top-k inside the CLW attention's 16-CTA clusters
// CTA c of a token's cluster owns the pools [c S, c S + S), S = ceil(n / 16) <= 2048, thread t its pools c S + t + 256 i
// (i < 8: every warp holds valid keys at long contexts). Bound: lo = the minimum of the 512 quarter-warp maxima (>= 512
// keys >= lo), hi = the maximum (cluster barrier 1); the range top-k's 256 float-linear bins over [lo, hi], each key's
// slot in its (CTA, bin) by a shared-memory atomic, the first kSkBk keys of every (CTA, bin) stored (barrier 2); every
// CTA then reads the 16 histograms (thread t: bin 255 - t of each), finds the boundary bin and ranks the boundary bin's
// gathered keys by (key descending, pool ascending): the same selected set as the other selects, ties at the 512th
// value broken by the pool index (deterministic). Order: ascending pool index (the CTAs in rank order, then (i, warp,
// lane)), each pool stored into the global selection row and, by DSMEM, into the list of every split whose rows it
// covers (barrier 3: the caller's wait). A CTA with more than kSkBk keys in the boundary bin: sk_fallback (radix).
constexpr int kSkCtas = 16;
constexpr int kSkKpt = 8;      // keys per thread: 16 CTAs x 256 threads x 8 = 32768 pools
constexpr int kSkMax = kSkCtas * 256 * kSkKpt;
constexpr int kSkBins = 256;
constexpr int kSkBk = 16;      // keys stored per (CTA, bin)
constexpr int kSkList = 40;    // a split's entries of the ordered selection (129 rows: <= 34)
struct SkSmem {                // (aliased onto the first KV buffer: used before the first gather; its pushes complete
                               // on the mbarriers AttnSmemClW::skmb, which are not aliased)
  uint32_t bnd[kSkCtas][2];    // every CTA's bound / maximum
  uint32_t brg[kSkCtas][2];    // (crowded boundary bin) every CTA's minimum / maximum key in it
  alignas(16) int hall[kSkCtas][kSkBins];  // every CTA's histogram (16-byte pushes)
  alignas(16) int hist[2][kSkBins];  // this CTA's ([1]: the fallback's rounds alternate)
  uint32_t bkk[kSkBins * kSkBk];
  uint16_t bki[kSkBins * kSkBk];
  uint32_t gb[kSkCtas * kSkBk][2];  // the boundary bin's keys of every CTA (key, pool), in CTA order
  int list[kSkList];
  uint32_t wlb[8], wmx[8];
  int wsum[8];
  int red[8][4];
  unsigned bal[kSkKpt][8];
  int boff[kSkKpt * 8];
  int s_bin, s_need, s_cb, s_bmax, f_off, f_eq;
  int bpre[kSkCtas + 1];
  uint8_t gsel[kSkCtas * kSkBk];
};
static_assert(sizeof(SkSmem) <= sizeof(bf16) * kTile * (kLatent + 8), "SEL scratch: the first KV buffer");
// launch parameters of the SEL instance (unused by the other instances)
struct SelArgs {
  const float* scores;  // [T, stride] the step's score rows (kpool_score_umma, the PDL predecessor)
  int* out_sel;         // [T, kSelPools] the selections (ascending pool order)
  int stride;
  unsigned* cnt;        // [2]: mla_qba's CTAs done (its SIG instance), the CTAs of this launch past the wait
  int cnt_target;       // mla_qba's CTAs
};

__device__ __forceinline__ uint32_t sk_ld_acquire(const unsigned* p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
// block-wide inclusive scan of v over the 256 threads (wsum: 8 ints of scratch; two barriers)
__device__ __forceinline__ int sk_scan256(int v, int* wsum) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int u = __shfl_up_sync(0xffffffffu, v, o);
    if (lane >= o) v += u;
  }
  if (lane == 31) wsum[warp] = v;
  __syncthreads();
  for (int w = 0; w < warp; ++w) v += wsum[w];
  return v;
}
// the pool of key i of thread tid in CTA c: 32-pool chunks round-robin over the 16 CTAs, chunk 8 i + warp of the CTA,
// one pool per lane (coalesced loads; a run of similar scores, e.g. repeated text, spreads over the CTAs instead of
// crowding one CTA's boundary bin)
__device__ __forceinline__ int sk_pool(int c, int tid, int i) {
  return 32 * (kSkCtas * (8 * i + (tid >> 5)) + c) + (tid & 31);
}
// a 32-bit value into cluster CTA q's shared memory at the address of p in this CTA
__device__ __forceinline__ void sk_st_remote(const void* p, int q, uint32_t v) {
  uint32_t ra;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(smem_u32(p)), "r"(q));
  asm volatile("st.shared::cluster.u32 [%0], %1;\n" ::"r"(ra), "r"(v) : "memory");
}

// the crowded case (a CTA with more than kSkBk keys in the boundary bin): four 8-bit radix rounds over the cluster
// (histograms alternate between sk.hist[1] / [0], one cluster barrier each) find the 512th key K and the number of keys
// equal to it to take (by pool index); returns this thread's selected keys as a mask (bit i: key i), the CTA's first
// output index in sk.f_off. The keys are reloaded (cold path).
__device__ __noinline__ unsigned sk_fallback(SkSmem& sk, const float* __restrict__ sc, int n) {
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int c = (int)cluster.block_rank();
  uint32_t k[kSkKpt];
#pragma unroll
  for (int i = 0; i < kSkKpt; ++i) k[i] = sk_pool(c, tid, i) < n ? f2key(__ldcg(sc + sk_pool(c, tid, i))) : 0u;
  uint32_t prefix = 0, pmask = 0;
  int need = kSelPools, a1 = 0;  // a1: this thread's digits' keys above K in the CTAs before c
  for (int r = 0; r < 4; ++r) {
    const int sh = 24 - 8 * r;
    int* h = sk.hist[(r + 1) & 1];
    h[tid] = 0;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kSkKpt; ++i)
      if (sk_pool(c, tid, i) < n && (k[i] & pmask) == prefix) atomicAdd(&h[(k[i] >> sh) & 255], 1);
    cluster.sync();
    int x[kSkCtas], tot = 0, pre = 0;
#pragma unroll
    for (int q = 0; q < kSkCtas; ++q) {
      x[q] = cluster.map_shared_rank(h, q)[255 - tid];
      tot += x[q];
      pre += q < c ? x[q] : 0;
    }
    const int it = sk_scan256(tot, sk.wsum);
    if (it - tot < need && it >= need) {
      sk.s_bin = 255 - tid;
      sk.s_need = need - (it - tot);
      sk.f_eq = pre;  // (the last round: keys equal to K in the CTAs before c)
    }
    __syncthreads();
    const int D = sk.s_bin;
    a1 += tid < 255 - D ? pre : 0;
    need = sk.s_need;
    prefix |= (uint32_t)D << sh;
    pmask |= 255u << sh;
    __syncthreads();  // (s_bin / s_need / wsum reused by the next round)
  }
  // this CTA reads no other CTA's shared memory any more: the last round's histograms (sk.hist[0]) are read by every
  // CTA after that round's barrier only, and SkSmem is the first KV buffer, which a CTA overwrites with its first row
  // tile as soon as its own list is complete (the CTAs covering its entries, not the whole cluster): sk_select waits
  // on this arrive before it returns (without it a slow CTA's last round could count KV data: a wrong selection, an
  // out-of-range push, a list that never completes)
  asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
  // keys == K (= prefix): CTA c takes the first e_c of its own in (i, warp, lane) order
  const int eqpre = sk.f_eq;
  const int e_c = max(0, need - eqpre);
  unsigned ab = 0, eqm = 0;
#pragma unroll
  for (int i = 0; i < kSkKpt; ++i) {
    const bool v = sk_pool(c, tid, i) < n;
    ab |= (v && k[i] > prefix) ? 1u << i : 0u;
    eqm |= (v && k[i] == prefix) ? 1u << i : 0u;
  }
  // ranks of the equal keys within the CTA: ballots per (i, warp), one warp scan
#pragma unroll
  for (int i = 0; i < kSkKpt; ++i) {
    const unsigned bw = __ballot_sync(0xffffffffu, (eqm >> i) & 1u);
    if (lane == 0) sk.bal[i][warp] = bw;
  }
  // a1 summed over the CTA
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) a1 += __shfl_xor_sync(0xffffffffu, a1, o);
  if (lane == 0) sk.red[warp][0] = a1;
  __syncthreads();
  if (warp == 0) {
    const int w0 = 2 * lane, w1 = w0 + 1;
    const int p0 = __popc(sk.bal[w0 >> 3][w0 & 7]), p1 = __popc(sk.bal[w1 >> 3][w1 & 7]);
    int v = p0 + p1;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int u = __shfl_up_sync(0xffffffffu, v, o);
      if (lane >= o) v += u;
    }
    sk.boff[w0] = v - p0 - p1;
    sk.boff[w1] = v - p1;
  }
  __syncthreads();
  unsigned m = ab;
#pragma unroll
  for (int i = 0; i < kSkKpt; ++i)
    if ((eqm >> i) & 1u) {
      const int rk = sk.boff[i * 8 + warp] + __popc(sk.bal[i][warp] & ((1u << lane) - 1u));
      if (rk < e_c) m |= 1u << i;
    }
  int A1 = 0;
#pragma unroll
  for (int w = 0; w < 8; ++w) A1 += sk.red[w][0];
  if (tid == 0) sk.f_off = A1 + min(eqpre, need);
  __syncthreads();  // (bal / boff / red reused by the caller)
  return m;
}

// push helpers (st.async: completes its bytes on the receiving CTA's mbarrier; no release fence, no cluster barrier)
__device__ __forceinline__ uint32_t sk_mapa(const void* p, int q) {
  uint32_t ra;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(smem_u32(p)), "r"(q));
  return ra;
}
__device__ __forceinline__ void sk_push1(uint32_t ra, uint32_t rm, uint32_t v) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];\n" ::"r"(ra), "r"(v),
               "r"(rm)
               : "memory");
}
__device__ __forceinline__ void sk_push2(uint32_t ra, uint32_t rm, uint32_t v0, uint32_t v1) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];\n" ::"r"(ra),
               "r"(v0), "r"(v1), "r"(rm)
               : "memory");
}
__device__ __forceinline__ void sk_push4(uint32_t ra, uint32_t rm, int4 v) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];\n" ::"r"(ra),
               "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(rm)
               : "memory");
}

// the selection of the token's n > kSelPools pools by the 16 CTAs of its cluster (all threads of every CTA), every
// exchange pushed (st.async) into the receivers' shared memory on their mbarriers (mb: AttnSmemClW::skmb, initialized
// and published by the caller before its PDL wait): the bounds (mb 0), the 16 histograms (mb 1), the boundary bin's
// keys (mb 2), the ordered selection's entries into the lists of the splits that cover them (mb 3: split s holds the
// entries [32 s, 32 s + 32], the 33rd only with rows past 2048). The caller waits on mb 3 before reading sk.list.
__device__ __forceinline__ void sk_select(SkSmem& sk, uint64_t* mb, const float* __restrict__ sc, int n, int nrows) {
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int c = (int)cluster.block_rank();
  uint32_t k[kSkKpt];
#pragma unroll
  for (int i = 0; i < kSkKpt; ++i) k[i] = sk_pool(c, tid, i) < n ? f2key(__ldcg(sc + sk_pool(c, tid, i))) : 0u;
  // 1. the bounds: lo = the minimum of the 256 half-warps' second largest keys (>= 512 keys >= lo), hi = the maximum of
  // the 128 warps' second largest (<= 128 keys > hi: those go to an open top bin, so the bins resolve the boundary's
  // range rather than [lo, the maximum], which a few outlying scores stretch until the boundary bin crowds)
  uint32_t a = 0, b = 0;
#pragma unroll
  for (int i = 0; i < kSkKpt; ++i) {
    b = max(b, min(a, k[i]));
    a = max(a, k[i]);
  }
#pragma unroll
  for (int o = 1; o < 16; o <<= 1) {
    const uint32_t a2 = __shfl_xor_sync(0xffffffffu, a, o), b2 = __shfl_xor_sync(0xffffffffu, b, o);
    b = max(max(b, b2), min(a, a2));
    a = max(a, a2);
  }
  {
    const uint32_t a2 = __shfl_xor_sync(0xffffffffu, a, 16), b2 = __shfl_xor_sync(0xffffffffu, b, 16);
    if (lane == 0) {
      sk.wlb[warp] = min(b, b2);
      sk.wmx[warp] = max(max(b, b2), min(a, a2));
    }
  }
  // (every CTA's mbarriers initialized: the kernel prologue's cluster arrive)
  asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
  __syncthreads();
  if (warp == 0) {
    uint32_t x = lane < 8 ? sk.wlb[lane] : 0xffffffffu, y = lane < 8 ? sk.wmx[lane] : 0u;
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) {
      x = min(x, __shfl_xor_sync(0xffffffffu, x, o));
      y = max(y, __shfl_xor_sync(0xffffffffu, y, o));
    }
    x = __shfl_sync(0xffffffffu, x, 0);  // (lanes 0-7 hold the reduction)
    y = __shfl_sync(0xffffffffu, y, 0);
    if (lane < kSkCtas) sk_push2(sk_mapa(&sk.bnd[c][0], lane), sk_mapa(&mb[0], lane), x, y);
  }
  at_bar_wait(&mb[0], 0);  // every CTA's bound and maximum
  uint32_t lo = 0xffffffffu, hi = 0u;
#pragma unroll
  for (int q = 0; q < kSkCtas; ++q) {
    lo = min(lo, sk.bnd[q][0]);
    hi = max(hi, sk.bnd[q][1]);
  }
  // 2. 255 float-linear bins over [lo, hi] (key bins if not finite: key-space bins alone crowd the boundary bin, the
  // radix fallback would run at every call) and bin 255 for the keys above hi; each key's slot in its bin by an atomic,
  // the first kSkBk of every (CTA, bin) stored
  const float lov = tr_key2f(lo), hiv = tr_key2f(hi);
  const float inv = 255.f * __frcp_rn(hiv - lov);
  const bool vb = isfinite(lov) && isfinite(hiv) && hiv > lov && isfinite(inv);
  const uint32_t d = hi - lo;
  const int s = d < 255u ? 0 : 32 - __clz(d / 255u);
  int bk[kSkKpt], sl[kSkKpt];
#pragma unroll
  for (int i = 0; i < kSkKpt; ++i) {
    bk[i] = sk_pool(c, tid, i) >= n || k[i] < lo ? -1
            : k[i] > hi ? 255
            : vb ? min(254, (int)((tr_key2f(k[i]) - lov) * inv)) : min(254, (int)((k[i] - lo) >> s));
    sl[i] = 0;
    if (bk[i] >= 0) {
      sl[i] = atomicAdd(&sk.hist[0][bk[i]], 1);
      if (sl[i] < kSkBk) {
        sk.bkk[bk[i] * kSkBk + sl[i]] = k[i];
        sk.bki[bk[i] * kSkBk + sl[i]] = (uint16_t)sk_pool(c, tid, i);
      }
    }
  }
  __syncthreads();
  // the histogram to every CTA (thread t: bins [4 (t % 64), + 4) to CTAs t / 64 + 4 j)
  {
    const int ch = tid & 63;
    const int4 v = *reinterpret_cast<const int4*>(&sk.hist[0][4 * ch]);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int q = (tid >> 6) + 4 * j;
      sk_push4(sk_mapa(&sk.hall[c][4 * ch], q), sk_mapa(&mb[1], q), v);
    }
  }
  at_bar_wait(&mb[1], 0);
  // 3. thread t: bin 255 - t of every CTA; the boundary bin from the scan over the descending bins
  int x[kSkCtas], tot = 0, pre = 0, xm = 0;
#pragma unroll
  for (int q = 0; q < kSkCtas; ++q) {
    x[q] = sk.hall[q][255 - tid];
    tot += x[q];
    pre += q < c ? x[q] : 0;
    xm = max(xm, x[q]);
  }
  const int it = sk_scan256(tot, sk.wsum);
  if (it - tot < kSelPools && it >= kSelPools) {
    sk.s_bin = 255 - tid;
    sk.s_need = kSelPools - (it - tot);
    sk.s_cb = tot;
    sk.s_bmax = xm;
    int a = 0;
#pragma unroll
    for (int q = 0; q < kSkCtas; ++q) {
      sk.bpre[q] = a;
      a += x[q];
    }
    sk.bpre[kSkCtas] = a;
    if (xm <= kSkBk) at_bar_expect(&mb[2], (uint32_t)tot * 8u);  // (the boundary keys of every CTA)
  }
  __syncthreads();
  const int B = sk.s_bin, need_b = sk.s_need, cb = sk.s_cb;
  unsigned sm8 = 0;       // this thread's selected keys (bit i)
  int off = 0;            // the CTA's first output index
  bool bnd_late = false;  // (cluster-uniform) the boundary keys after the others
  bool fb = false;        // (cluster-uniform) the radix select ran: its cluster barrier arrived, waited below
  if (sk.s_bmax > kSkBk) {  // (cluster-uniform) a CTA's boundary keys not all stored
    // exact ties at the 512th key (identical pools: e.g. repeated text) crowd it: every CTA's minimum / maximum key in
    // the boundary bin to every CTA (mb 4); all equal -> the keys above the bin and the first of the tied keys by pool
    // index (the order of the ranking below), else the radix select
    uint32_t mn = 0xffffffffu, mxb = 0u;
#pragma unroll
    for (int i = 0; i < kSkKpt; ++i)
      if (bk[i] == B) {
        mn = min(mn, k[i]);
        mxb = max(mxb, k[i]);
      }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      mn = min(mn, __shfl_xor_sync(0xffffffffu, mn, o));
      mxb = max(mxb, __shfl_xor_sync(0xffffffffu, mxb, o));
    }
    int a1 = tid < 255 - B ? pre : 0;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a1 += __shfl_xor_sync(0xffffffffu, a1, o);
    if (lane == 0) {
      sk.wlb[warp] = mn;
      sk.wmx[warp] = mxb;
      sk.red[warp][0] = a1;
    }
    __syncthreads();
    if (warp == 0) {
      uint32_t x = lane < 8 ? sk.wlb[lane] : 0xffffffffu, y = lane < 8 ? sk.wmx[lane] : 0u;
#pragma unroll
      for (int o = 4; o > 0; o >>= 1) {
        x = min(x, __shfl_xor_sync(0xffffffffu, x, o));
        y = max(y, __shfl_xor_sync(0xffffffffu, y, o));
      }
      x = __shfl_sync(0xffffffffu, x, 0);
      y = __shfl_sync(0xffffffffu, y, 0);
      if (lane < kSkCtas) sk_push2(sk_mapa(&sk.brg[c][0], lane), sk_mapa(&mb[4], lane), x, y);
    }
    at_bar_wait(&mb[4], 0);
    uint32_t gmn = 0xffffffffu, gmx = 0u;
#pragma unroll
    for (int q = 0; q < kSkCtas; ++q) {
      gmn = min(gmn, sk.brg[q][0]);
      gmx = max(gmx, sk.brg[q][1]);
    }
    if (gmn == gmx) {  // (cluster-uniform) all tied: CTA c takes the first e_c of its own, in (i, warp, lane) order
      const int eqpre = sk.bpre[c], e_c = max(0, min(need_b - eqpre, sk.bpre[c + 1] - eqpre));
#pragma unroll
      for (int i = 0; i < kSkKpt; ++i) {
        const unsigned bw = __ballot_sync(0xffffffffu, bk[i] == B);
        if (lane == 0) sk.bal[i][warp] = bw;
      }
      __syncthreads();
      if (warp == 0) {
        const int w0 = 2 * lane, w1 = w0 + 1;
        const int p0 = __popc(sk.bal[w0 >> 3][w0 & 7]), p1 = __popc(sk.bal[w1 >> 3][w1 & 7]);
        int v = p0 + p1;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
          const int u = __shfl_up_sync(0xffffffffu, v, o);
          if (lane >= o) v += u;
        }
        sk.boff[w0] = v - p0 - p1;
        sk.boff[w1] = v - p1;
      }
      __syncthreads();
#pragma unroll
      for (int i = 0; i < kSkKpt; ++i)
        if (bk[i] > B ||
            (bk[i] == B && sk.boff[i * 8 + warp] + __popc(sk.bal[i][warp] & ((1u << lane) - 1u)) < e_c))
          sm8 |= 1u << i;
#pragma unroll
      for (int w = 0; w < 8; ++w) off += sk.red[w][0];
      off += min(eqpre, need_b);
      __syncthreads();  // (bal / boff reused below)
    } else {
      sm8 = sk_fallback(sk, sc, n);
      off = sk.f_off;
      fb = true;
    }
  } else {
    // this CTA's boundary keys to every CTA (thread t: key t % 16 to CTA t / 16), at their place in the cluster's order
    {
      const int j = tid & 15, q = tid >> 4, nb = sk.bpre[c + 1] - sk.bpre[c];
      if (j < nb)
        sk_push2(sk_mapa(&sk.gb[sk.bpre[c] + j][0], q), sk_mapa(&mb[2], q), sk.bkk[B * kSkBk + j],
                 (uint32_t)sk.bki[B * kSkBk + j]);
    }
    // the keys above the boundary bin first (CTA-major, ascending: at the above keys of the CTAs before c), pushed while
    // the boundary keys are exchanged; the need_b selected boundary keys after them, by rank (below)
    int a1 = tid < 255 - B ? pre : 0;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a1 += __shfl_xor_sync(0xffffffffu, a1, o);
    if (lane == 0) sk.red[warp][0] = a1;
    __syncthreads();
#pragma unroll
    for (int w = 0; w < 8; ++w) off += sk.red[w][0];
#pragma unroll
    for (int i = 0; i < kSkKpt; ++i)
      if (bk[i] > B) sm8 |= 1u << i;
    bnd_late = true;
  }
  // 4. the CTA's selected pools in ascending order: ballots per (i, warp), one warp scan over the 64 words
  unsigned bw[kSkKpt];
#pragma unroll
  for (int i = 0; i < kSkKpt; ++i) {
    bw[i] = __ballot_sync(0xffffffffu, (sm8 >> i) & 1u);
    if (lane == 0) sk.bal[i][warp] = bw[i];
  }
  __syncthreads();
  if (warp == 0) {
    const int w0 = 2 * lane, w1 = w0 + 1;
    const int p0 = __popc(sk.bal[w0 >> 3][w0 & 7]), p1 = __popc(sk.bal[w1 >> 3][w1 & 7]);
    int v = p0 + p1;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int u = __shfl_up_sync(0xffffffffu, v, o);
      if (lane >= o) v += u;
    }
    sk.boff[w0] = v - p0 - p1;
    sk.boff[w1] = v - p1;
  }
  __syncthreads();
  // (a sparse token's 2048 + rr rows, rr < 4, over 16 splits: split s has the rows [128 s + min(s, rr), ...), 128 or 129,
  // i.e. the entries [32 s, 32 s + 31], and 32 s + 32 too when rr > 0: entry g at g % 32 of split g / 32, and the
  // 33rd of split g / 32 - 1 for g % 32 == 0)
  const bool rr = nrows > kSelPools * kKpool;
#pragma unroll
  for (int i = 0; i < kSkKpt; ++i)
    if ((sm8 >> i) & 1u) {
      const int g = off + sk.boff[i * 8 + warp] + __popc(bw[i] & ((1u << lane) - 1u));
      const uint32_t pool = (uint32_t)sk_pool(c, tid, i);
      sk_push1(sk_mapa(&sk.list[g & 31], g >> 5), sk_mapa(&mb[3], g >> 5), pool);
      if ((g & 31) == 0 && g > 0 && rr) sk_push1(sk_mapa(&sk.list[32], (g >> 5) - 1), sk_mapa(&mb[3], (g >> 5) - 1), pool);
    }
  if (bnd_late) {  // the boundary bin's keys of every CTA ranked by (key descending, pool ascending); each CTA pushes its
                   // own selected ones at kSelPools - need_b + rank
    at_bar_wait(&mb[2], 0);
    if (tid >= sk.bpre[c] && tid < sk.bpre[c + 1]) {
      const uint32_t my = sk.gb[tid][0], mi = sk.gb[tid][1];
      int rank = 0;
#pragma unroll 4
      for (int q = 0; q < cb; ++q) {
        const uint32_t y = sk.gb[q][0];
        rank += (y > my) | ((y == my) & (sk.gb[q][1] < mi));
      }
      if (rank < need_b) {
        const int g = kSelPools - need_b + rank;
        sk_push1(sk_mapa(&sk.list[g & 31], g >> 5), sk_mapa(&mb[3], g >> 5), mi);
        if ((g & 31) == 0 && g > 0 && rr) sk_push1(sk_mapa(&sk.list[32], (g >> 5) - 1), sk_mapa(&mb[3], (g >> 5) - 1), mi);
      }
    }
  }
  if (fb) {  // (every CTA past its last read of another CTA's histograms before this CTA's KV buffer is overwritten)
    __syncwarp();
    asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
  }
}

// element offset of (row, chunk) in a swizzled [rows, 64 chunks] bf16 tile
__device__ __forceinline__ int swz(int row, int chunk) {
  return row * kLatent + ((chunk ^ (row & 7)) << 3);
}
// P tile: [16, 64] -> 8 chunks per row
__device__ __forceinline__ int swzp(int row, int chunk) {
  return row * kTile + ((chunk ^ (row & 7)) << 3);
}

// PRE (the CLW instance of the chained MTP passes, vis set): every input of a split's row table is
// final when the grid starts -- the passes' positions, slots, shared selection and visible-pool counts are the first
// pass's consumer's outputs, complete before the pass's input projection passed its PDL wait (gemv_tma's late trigger:
// no later kernel of the pass launches earlier), the block tables are uploaded before the step, and every attended
// cache row except the token's own (its last row, at its position: written by the pass's norms, before the PDL
// predecessor) was written by a completed kernel -- so the row table is built and the row tiles of the prologue are
// gathered before the PDL wait, the own row's copy issued after it (with the query). The same rows into the same buffers
// as the CLW prologue after its wait (P3's third buffer for a split's partial first tile): bitwise the CLW instance.
// Returns the visible-pool count; own_i / own_r: the own row's tile / row when its copy is deferred (else -1).
template <int NW>
__device__ __forceinline__ int attn_pre_rows(AttnSmemClW<NW>& sm, const bf16* __restrict__ cache,
                                             const int* __restrict__ sel, const int* __restrict__ vis,
                                             const MlaTokMeta& meta, int t, int b, int pos, bool staged,
                                             const int64_t* sbt, int split, int nsplit, int Hc, int& own_i, int& own_r) {
  constexpr int NT = 32 * NW;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ncand = vis[t];
  const bool sparse = ncand > kSelPools;
  const int nrows = sparse ? kSelPools * kKpool + (pos + 1 - ncand * kKpool) : pos + 1;
  const int qr = nrows / nsplit, rr = nrows % nsplit;
  const int wlen = qr + (split < rr ? 1 : 0), wr0 = split * qr + min(split, rr);
  const int nt = cdiv(wlen, kTile), wf = wlen - (nt - 1) * kTile;
  const int* tsel = sel + (size_t)t * kSelPools;
  if (nt > kClTiles) __trap();  // (as the kernel's check)
  if (staged) {  // (the staged block-table entries)
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
  }
  for (int e = tid; e < nt * kTile; e += NT) {  // (the CLW prologue's row table)
    const int i = e / kTile, r = e % kTile;
    int64_t row = -1;
    if (i > 0 || r < wf) {
      const int g = wr0 + (i == 0 ? r : wf + (i - 1) * kTile + r);
      if (g < nrows) {
        const int p = !sparse ? g
                      : g < kSelPools * kKpool ? tsel[g / kKpool] * kKpool + g % kKpool
                                               : ncand * kKpool + (g - kSelPools * kKpool);
        row = staged ? sbt[p / kMlaBlock] * meta.lat_page_rows + p % kMlaBlock : phys_row(meta, b, p);
      }
    }
    sm.rows[i][r] = row;
  }
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // (the staged table read before the copies)
  __syncthreads();
  const int nkmax = (Hc + nsplit - 1) / nsplit, zr0 = min(kTile, (wf + 15) / 16 * 16);
  const bool p3 = nt >= 3 && wf < kTile &&
                  (2 * kHM - nkmax * nsplit) * kLatent * (int)sizeof(float) >= zr0 * kKvP * (int)sizeof(bf16);
  const int npro = min(nt, p3 ? 3 : 2);  // (the tiles the prologue loads)
  own_i = own_r = -1;
  if (wlen > 0 && wr0 + wlen == nrows) {  // (the split of the last row)
    const int i = wlen - 1 < wf ? 0 : 1 + (wlen - 1 - wf) / kTile;
    if (i < npro) {
      own_i = i;
      own_r = i == 0 ? wlen - 1 : (wlen - 1 - wf) % kTile;
    }
  }
  bf16* const pkv = reinterpret_cast<bf16*>(&sm.recv[nkmax * nsplit][0]);
  for (int i = 0; i < npro; ++i) {  // (load_tile: P3 tiles 0 / 1 / 2 into buffers 2 / 0 / 1, else 0 / 1)
    const int buf = p3 ? (i == 0 ? 2 : (i - 1) & 1) : i;
    bf16* const dst = buf == 2 ? pkv : sm.kv[buf];
    const int nv = i == 0 ? wf : kTile;
    if (lane < kTile / NW) {
      const int r = warp * (kTile / NW) + lane;
      if (tid == 0) at_bar_expect(&sm.bar[buf], (uint32_t)nv * kLatent * 2);
      if (r < nv && !(i == own_i && r == own_r))
        at_bulk(dst + r * kKvP, cache + sm.rows[i][r] * kLatent, kLatent * 2, &sm.bar[buf]);
    }
    const int zr = min(kTile, (nv + 15) / 16 * 16);
    if (nv < zr)
      for (int e = nv * kChunks + tid; e < zr * kChunks; e += NT)
        *reinterpret_cast<uint4*>(dst + (e / kChunks) * kKvP + (e % kChunks) * 8) = make_uint4(0, 0, 0, 0);
  }
  return ncand;
}

// grid (T, nsplit, head groups of kHM): split s handles a contiguous range of the token's row
// tiles (split-KV for few tokens with long row lists); with nsplit > 1 the unnormalized partial
// output and its softmax max / sum go to ws ([T][groups][nsplit][kHM] x (kLatent + 2) floats) for
// mla_attn_combine. NW warps (4 or 8): warp w takes the latent slice [512 w / NW, 512 (w + 1) / NW)
// of the scores' K and of the output columns (8 warps: half the per-warp MMA chain per row tile,
// two warps per scheduler to hide its latencies).
// CL (decode, few tokens): grid (nsplit, T, groups) in clusters of the nsplit CTAs of one (token, head
// group): the splits' partials stay in shared memory and the cluster combines them itself (CTA s
// finalizes heads s, s + nsplit, ... with mla_attn_combine_kernel's operations in split order: the
// same bits), so no partials workspace and no combine launch.
// SEL (the CLW instance at 16 splits, sparse_mla_attn_sel): the decode top-k by the cluster itself (sk_select over the
// score rows of sa, its PDL predecessor kpool_score_umma's), the row table from the cluster's ordered selection, every
// row tile of the split gathered before the query exists (P3: a 1-3 row first tile in the receive rows a 16-split
// cluster leaves unused), then mla_qba's CTAs awaited on sa.cnt (its SIG instance) and the dependents triggered, the
// query loaded, and the CLW tile loop / combine unchanged: bitwise the CLW instance on that selection
template <int NW, bool CL, bool TMA = false, bool FA = false, bool WR = false, bool SEL = false, bool PRE = false>
__global__ void __launch_bounds__(32 * NW) sparse_mla_attn_kernel(
    bf16* __restrict__ out, const bf16* __restrict__ q, const bf16* __restrict__ cache,
    const int* __restrict__ sel, const MlaTokMeta meta, int H, float scale, float* ws,
    const int* __restrict__ vis, const SelArgs sa) {
  if (!SEL && !PRE) pdl_trigger();  // (PRE: after the wait, below)
  static_assert(!PRE || (WR && !SEL), "PRE: the CLW instance");
  constexpr bool P3 = SEL || PRE;
  constexpr int NT = 32 * NW;           // threads
  constexpr int KS = kLatent / 16 / NW;  // score k-steps per warp (16 latent values each)
  constexpr int NP = kChunks / NW / 2;   // output column pairs of 8-column tiles per warp
  constexpr int TPR = NT / kHM;          // softmax threads per score row
  constexpr int CPT = kTile / TPR;       // softmax columns per thread (8 or 4)
  extern __shared__ __align__(128) uint8_t smem_raw[];
  static_assert(!WR || (CL && TMA && FA), "CLW: the FA cluster form");
  static_assert(!SEL || (WR && NW == 8), "SEL: the CLW instance");
  using Smem = std::conditional_t<
      WR, AttnSmemClW<NW>,
      std::conditional_t<TMA, std::conditional_t<CL, AttnSmemClT<NW>, AttnSmemT<NW>>,
                         std::conditional_t<CL, AttnSmemCl<NW>, AttnSmem<NW>>>>;
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
  const int t = CL ? blockIdx.y : blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // (SEL) the selection's scratch and this split's list in the first KV buffer
  SkSmem& sk = *reinterpret_cast<SkSmem*>(&sm.kv[0][0]);
  if constexpr (TMA) {
    if (tid == 0) {  // (initialized before the first block barrier, which every bulk copy follows)
      at_bar_init(&sm.bar[0]);
      at_bar_init(&sm.bar[1]);
      if constexpr (P3) at_bar_init(&sm.bar[2]);
      if constexpr (SEL) {  // (the selection's pushes: bounds, histograms, boundary keys, the list)
#pragma unroll
        for (int i = 0; i < 5; ++i) at_bar_init(&sm.skmb[i]);
        at_bar_expect(&sm.skmb[0], kSkCtas * 8u);
        at_bar_expect(&sm.skmb[1], kSkCtas * kSkBins * 4u);
        at_bar_expect(&sm.skmb[4], kSkCtas * 8u);
      }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
  }
  if constexpr (SEL) {
    sk.hist[0][tid] = 0;
    __syncthreads();
    // (every CTA's mbarriers initialized before any push: this arrive's wait follows the PDL wait)
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
  }
  // element offset of (row, 16-byte chunk) in a KV tile: swizzled rows, or the TMA forms' padded rows
  auto kvi = [](int row, int chunk) -> int {
    if constexpr (TMA)
      return row * kKvP + chunk * 8;
    else
      return swz(row, chunk);
  };
  const int split = CL ? blockIdx.x : blockIdx.y, nsplit = CL ? gridDim.x : gridDim.y;
  // head group: heads [h0, h0 + Hc) of the token (rows of q / out with H heads per token)
  const int h0 = blockIdx.z * kHM, Hc = min(kHM, H - h0);
  ws += (size_t)blockIdx.z * nsplit * kHM * (kLatent + 2) +
        (size_t)t * (gridDim.z - 1) * nsplit * kHM * (kLatent + 2);
  int b = 0, pos = 0;
  // the sequence's block-table entries [0, pos / 128] staged in the second KV buffer before the PDL wait
  // (uploaded before the step; the buffer's first load is issued after the row table): the row table
  // after the wait then costs one memory round trip (the selections) instead of two
  int64_t* sbt = reinterpret_cast<int64_t*>(&sm.kv[1][0]);
  constexpr int kSbtMax = (int)(sizeof(sm.kv[1]) / sizeof(int64_t)) - 2;
  bool staged = false;
  if (PRE || meta.pre) {  // (uploaded before the step; PRE: see attn_pre_rows) cluster form: asynchronous copies
    // (cp.async.cg, L2-coherent) of the 16-byte chunks from the row's 16-byte aligned start; sbt then indexed from it
    b = meta.token_seq[t];
    pos = meta.positions[t];
    const int nb = pos / kMlaBlock + 1;
    if (CL && nb <= kSbtMax) {
      const int64_t* bt = meta.block_tables + (size_t)b * meta.max_blocks;
      const int64_t* bt0 = reinterpret_cast<const int64_t*>(reinterpret_cast<uintptr_t>(bt) & ~(uintptr_t)15);
      const int o0 = (int)(bt - bt0), nc = (o0 + nb + 1) / 2;
      for (int i = tid; i < nc; i += NT) cp_async16(sbt + 2 * i, bt0 + 2 * i, true);
      asm volatile("cp.async.commit_group;\n" ::);
      sbt += o0;
      staged = true;
    } else {  // (split form: the row's lines into L2, spread over the splits)
      const int64_t* bt = meta.block_tables + (size_t)b * meta.max_blocks;
      const int lines = (pos / kMlaBlock + 16) / 16;  // 128-byte lines of entries [0, pos / 128]
      for (int l = split * NT + tid; l < lines; l += nsplit * NT) prefetch_l2(bt + l * 16);
    }
  }
  int pre_ncand = 0, own_i = -1, own_r = -1;  // (PRE: the row table and the prologue's gathers before the wait)
  if constexpr (PRE)
    pre_ncand = attn_pre_rows<NW>(sm, cache, sel, vis, meta, t, b, pos, staged, sbt, split, nsplit, Hc, own_i, own_r);
  pdl_wait();
  // (PRE: the dependents -- the w_vc and, at its start, the o_proj -- launch once mla_qba has completed: launched at this
  // grid's start they would stage their weights (4 + 12 MB) under mla_qba and delay its absorb weights; they still
  // stage them under this grid's tiles)
  if constexpr (PRE) pdl_trigger();
  if (!PRE && !meta.pre) {
    b = meta.token_seq[t];
    pos = meta.positions[t];
  }
  const int ncand = PRE ? pre_ncand : vis ? vis[t] : (pos + 1) / kKpool;
  const bool sparse = ncand > kSelPools;
  const int nrows = sparse ? kSelPools * kKpool + (pos + 1 - ncand * kKpool) : pos + 1;
  const int* tsel = sel + (size_t)t * kSelPools;
  // (SEL) the selection by the cluster
  if constexpr (SEL) {
    if (sparse && tid == 0)  // (this split's list: the entries [32 s, 32 s + 32), and 32 s + 32 with rows past 2048)
      at_bar_expect(&sm.skmb[3], 4u * (uint32_t)(min(kSelPools, 32 * split + 32 + (nrows > kSelPools * kKpool ? 1 : 0)) -
                                                 32 * split));
    if (sparse)
      sk_select(sk, sm.skmb, sa.scores + (size_t)t * sa.stride, ncand, nrows);
    else  // (the prologue's cluster arrive: its wait in sk_select, before the first push)
      asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
  }
  // row tiles per split: ntiles_all / nsplit, the remainder to the last splits (a long context's 33rd
  // tile holds the 1-3 rows after the selected pools: 2 full tiles per split instead of 3 at 16 splits).
  // Cluster form: ceil(ntiles_all / nsplit) per split, the last splits empty: every working split's partial
  // rows cross the cluster (DSMEM), so 16 working splits cost more than the third tile
  const int ntiles_all = cdiv(nrows, kTile);
  int tile0, ntiles;
  // (CLW) the split's rows [wr0, wr0 + wlen): nrows / nsplit each (the remainder one more to the first splits), as a
  // first tile of the wf rows past the full tiles (landing and computed while the first full tile lands: at 10
  // splits a long context's 2051 rows are 13-14 + 3 x 64 per split instead of 4 full tiles on 8 of them) and full tiles
  int wr0 = 0, wf = 0;
  if constexpr (WR) {
    const int qr = nrows / nsplit, rr = nrows % nsplit;
    const int wlen = qr + (split < rr ? 1 : 0);
    wr0 = split * qr + min(split, rr);
    tile0 = 0;
    ntiles = cdiv(wlen, kTile);
    wf = wlen - (ntiles - 1) * kTile;
  } else if constexpr (CL) {
    const int per = cdiv(ntiles_all, nsplit);
    tile0 = min(ntiles_all, split * per);
    ntiles = min(ntiles_all, tile0 + per);
  } else {
    const int qt = ntiles_all / nsplit, lo = nsplit - ntiles_all % nsplit;
    tile0 = split * qt + max(0, split - lo);
    ntiles = tile0 + qt + (split >= lo ? 1 : 0);
  }
  // (empty split: no row tiles; with CL it still takes part in the cluster's combine)
  const bool empty = nsplit > 1 && ntiles == tile0;
  if (!CL && empty) {
    // empty split: only mark it (the combine skips splits with max = -inf)
    if (staged) asm volatile("cp.async.wait_group 0;\n" ::);
    if (tid < kHM) {
      float* w = ws + ((size_t)t * nsplit + split) * kHM * (kLatent + 2);
      w[(size_t)tid * (kLatent + 2) + kLatent] = -INFINITY;
      w[(size_t)tid * (kLatent + 2) + kLatent + 1] = 0.f;
    }
    return;
  }

  // Q -> smem with the first row tile's cp.async group (rows >= H zeroed; SEL: after the q path's CTAs, below)
  for (int i = tid; i < (SEL || (CL && empty) ? 0 : kHM * kChunks); i += NT) {
    const int r = i / kChunks, c = i % kChunks;
    cp_async16(&sm.q[swz(r, c)], q + ((size_t)t * H + h0 + min(r, Hc - 1)) * kLatent + c * 8,
               r < Hc);
  }
  if (tid < kHM) {
    sm.m[tid] = -INFINITY;
    sm.l[tid] = 0.f;
  }
  auto row_of = [&](int r) -> int64_t {
    if (r >= nrows) return -1;
    int p;
    if (!sparse)
      p = r;
    else if (r < kSelPools * kKpool)
      p = (SEL ? sk.list[r / kKpool - wr0 / kKpool] : tsel[r / kKpool]) * kKpool + r % kKpool;
    else
      p = ncand * kKpool + (r - kSelPools * kKpool);
    return staged ? sbt[p / kMlaBlock] * meta.lat_page_rows + p % kMlaBlock : phys_row(meta, b, p);
  };
  // (P3) the first tile's buffer in the receive rows the cluster leaves unused (bitwise: the same rows elsewhere)
  bf16* pkv = nullptr;
  bool p3 = false;
  if constexpr (P3) {
    const int nkmax = (Hc + nsplit - 1) / nsplit;
    const int zr0 = min(kTile, (wf + 15) / 16 * 16);
    p3 = ntiles - tile0 >= 3 && wf < kTile &&
         (2 * kHM - nkmax * nsplit) * kLatent * (int)sizeof(float) >= zr0 * kKvP * (int)sizeof(bf16);
    pkv = reinterpret_cast<bf16*>(&sm.recv[nkmax * nsplit][0]);
  }
  auto kvbuf = [&](int buf) -> bf16* {
    if constexpr (P3)
      return buf == 2 ? pkv : sm.kv[buf];
    else
      return sm.kv[buf];
  };
  // row tile i of the split (rows from sm.rows[i]) into buffer buf
  auto load_tile = [&](int i, int buf) {
    if constexpr (TMA) {
      // the tile's valid rows (a prefix: row_of is -1 from nrows on), kTile / NW rows per warp (a bulk copy
      // instruction of 32 active lanes is slow to issue: spread over every warp, a few lanes each); rows
      // past the end zeroed by every thread (their probabilities are 0: the rows must be finite)
      const int nv = WR ? (i == 0 ? wf : kTile) : max(0, min(kTile, nrows - (tile0 + i) * kTile));
      if (lane < kTile / NW) {
        const int r = warp * (kTile / NW) + lane;
        if (tid == 0) at_bar_expect(&sm.bar[buf], (uint32_t)nv * kLatent * 2);
        if (r < nv) at_bulk(kvbuf(buf) + r * kKvP, cache + sm.rows[i][r] * kLatent, kLatent * 2, &sm.bar[buf]);
      }
      // (the rows the MMAs read: every group in the cluster form, else up to the last valid group)
      const int zr = (CL && !WR) ? kTile : min(kTile, (nv + 15) / 16 * 16);
      if (nv < zr)
        for (int e = nv * kChunks + tid; e < zr * kChunks; e += NT)
          *reinterpret_cast<uint4*>(kvbuf(buf) + kvi(e / kChunks, e % kChunks)) = make_uint4(0, 0, 0, 0);
    } else {
      for (int e = tid; e < kTile * kChunks; e += NT) {
        const int r = e / kChunks, c = e % kChunks;
        const int64_t pr = sm.rows[i][r];
        const bf16* src = cache + (pr < 0 ? 0 : pr * kLatent) + c * 8;
        cp_async16(&sm.kv[buf][swz(r, c)], src, pr >= 0);
      }
      asm volatile("cp.async.commit_group;\n" ::);
    }
  };
  if (CL && ntiles - tile0 > kClTiles) __trap();  // (rows <= kMaxRows: at most 33 tiles over >= 2 splits)
  // the cache rows of every tile of the split at once (independent lookups: one latency), then
  // the first two tiles' loads
  if (!PRE && staged) {  // (the staged block-table entries)
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
  }
  if constexpr (SEL) {
    if (sparse) {  // (this split's list complete; its entries [32 s, 32 s + 32) into the global row)
      at_bar_wait(&sm.skmb[3], 0);
      if (tid < 32) sa.out_sel[(size_t)t * kSelPools + 32 * split + tid] = sk.list[tid];
    }
  }
  for (int e = tid; e < (PRE ? 0 : (ntiles - tile0) * kTile); e += NT) {
    if constexpr (WR) {
      const int i = e / kTile, r = e % kTile;
      sm.rows[i][r] = (i > 0 || r < wf) ? row_of(wr0 + (i == 0 ? r : wf + (i - 1) * kTile + r)) : -1;
    } else {
      sm.rows[e / kTile][e % kTile] = row_of(tile0 * kTile + e);
    }
  }
  // (TMA: the staged block table in the second buffer is read through the generic proxy before the bulk
  // copies overwrite it)
  if constexpr (TMA) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  __syncthreads();
  // (SEL) the q path's counter requested before the gathers are issued, by a thread that issues no bulk copy
  uint32_t qv = 0;
  if constexpr (SEL)
    if (tid == NT - 1) qv = sk_ld_acquire(sa.cnt);
  if constexpr (PRE) {  // (the own row's copy, deferred past the wait by attn_pre_rows; the other rows are in flight)
    if (own_i >= 0 && tid == own_r / (kTile / NW) * 32 + own_r % (kTile / NW)) {
      const int buf = p3 ? (own_i == 0 ? 2 : (own_i - 1) & 1) : own_i;
      at_bulk(kvbuf(buf) + own_r * kKvP, cache + sm.rows[own_i][own_r] * kLatent, kLatent * 2, &sm.bar[buf]);
    }
  } else if (p3) {  // (P3: the partial first tile in its own buffer, the next two in the KV buffers)
    load_tile(0, 2);
    load_tile(1, 0);
    load_tile(2, 1);
  } else {
    if (tile0 < ntiles) load_tile(0, 0);
    if (tile0 + 1 < ntiles) load_tile(1, 1);
  }
  if constexpr (SEL) {
    // mla_qba's CTAs (q of this token) done: then the dependents may launch and the query is loaded
    if (tid == NT - 1) {
      // (one acquire load in flight: more polling loads in flight on the counter's line, from every CTA of the
      // launch, would make it an L2 hot spot)
      while ((int)qv < sa.cnt_target) qv = sk_ld_acquire(sa.cnt);
    }
    __syncthreads();
    pdl_trigger();
    for (int i = tid; i < kHM * kChunks; i += NT) {
      const int r = i / kChunks, c = i % kChunks;
      cp_async16(&sm.q[swz(r, c)], q + ((size_t)t * H + h0 + min(r, Hc - 1)) * kLatent + c * 8, r < Hc);
    }
  }
  if constexpr (TMA) asm volatile("cp.async.commit_group;\n" ::);  // (Q)

  float o[2 * NP][4];
#pragma unroll
  for (int i = 0; i < 2 * NP; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  // (FA: the query's MMA fragments for every latent slice and k-step, loaded once while the first tile lands)
  static_assert(!FA || (NW == 8 && TMA), "FA form: 8 warps of 8 rows, with the TMA row copies");
  uint32_t qf[FA ? 8 : 1][FA ? 4 : 1][4];

  // (SEL, MG) a split of two full tiles and a first tile of 1-8 rows (P3's buffer): two iterations, the first tile's
  // rows merged into the last one as a ninth 8-row block (scored by warp 0, one more P V k-step for every warp): another
  // fp32 order for those tokens, one tile iteration fewer on the cluster's slowest splits (all tiles are gathered before
  // the query exists, so a third iteration would be pure critical path)
  const bool mg = SEL && p3 && wf <= 8;
  const int tfirst = mg ? tile0 + 1 : tile0;
  bf16* const p2 = [&]() -> bf16* {  // (MG) the block's probabilities [16 heads][16 rows], rows 8-15 zero
    if constexpr (SEL)
      return reinterpret_cast<bf16*>(&sm.recv[2 * kHM - 1][256]);
    else
      return nullptr;
  }();
  if (mg && tid < kHM * 8) p2[(tid >> 3) * 16 + 8 + (tid & 7)] = __float2bfloat16_rn(0.f);
  for (int tile = tfirst; tile < ntiles; ++tile) {
    // (P3: tile 0 in buffer 2, tile i >= 1 in buffer (i - 1) & 1)
    const int buf = p3 ? (tile == tile0 ? 2 : (tile - tile0 - 1) & 1) : (tile - tile0) & 1;
    const uint32_t bph = p3 ? (tile == tile0 ? 0u : (uint32_t)(((tile - tile0 - 1) >> 1) & 1))
                            : (uint32_t)(((tile - tile0) >> 1) & 1);
    const bool mx2 = mg && tile == ntiles - 1;  // (MG: the merged rows in this iteration)
    if constexpr (TMA) {
      if (tile == tfirst) {
        asm volatile("cp.async.wait_group 0;\n" ::);  // (Q)
        if constexpr (FA) {  // (FA) the query fragments while the first tile is still in flight
          __syncthreads();
#pragma unroll
          for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
              ldsm_x4(qf[j][ks], &sm.q[swz(lane & 15, j * 8 + ks * 2 + (lane >> 4))]);
        }
      }
      at_bar_wait(&sm.bar[buf], bph);
      if (mx2) at_bar_wait(&sm.bar[2], 0);
    } else if (tile + 1 < ntiles) {
      asm volatile("cp.async.wait_group 1;\n" ::);
    } else {
      asm volatile("cp.async.wait_group 0;\n" ::);
    }
    __syncthreads();
    const bf16* kv = kvbuf(buf);
    // 16-row groups of the tile holding a row (a long context's tail tile: 1-3 rows): the others'
    // scores are masked and their probabilities are 0 against zero-filled rows, so their MMAs are
    // skipped (the same bits)
    // (cluster form: every group; its tail tile is on a split with fewer tiles, and the checks in the
    // unrolled MMA loops cost the full tiles more than they save)
    const int vg = WR   ? (tile == 0 ? (wf + 15) / 16 : kTile / 16)
                   : CL ? kTile / 16
                        : min(kTile / 16, (nrows - tile * kTile + 15) / 16);
    // FA (the cluster form): warp w scores the tile's rows [8 w, 8 w + 8) for
    // every head over the whole latent dim: for each latent slice the base form's per-warp MMAs (same operands,
    // same order) into their own accumulator, the 8 slices' partials summed in slice order in registers (the
    // base form's sum over sp); the row maxima and the probability sums cross the warps as 16 values per warp,
    // the sums combined in the base form's butterfly order: bitwise the base form, without the 32 KB
    // partial-score exchange and its 8-way reads per tile
    float fa_al[2] = {0.f, 0.f};  // (FA) alpha of heads lane / 4 and lane / 4 + 8
    if constexpr (FA) {
      float x[4] = {0.f, 0.f, 0.f, 0.f};
      if ((warp >> 1) < vg) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int kk = 0; kk < 2; ++kk) {
            uint32_t bb[4];
            ldsm_x4(bb, &kv[kvi(warp * 8 + (lane & 7), j * 8 + kk * 4 + (lane >> 3))]);
            mma16816(acc, qf[j][2 * kk], bb[0], bb[1]);
            mma16816(acc, qf[j][2 * kk + 1], bb[2], bb[3]);
          }
#pragma unroll
          for (int e = 0; e < 4; ++e) x[e] += acc[e];
        }
      }
      // x[0..1]: head lane / 4, columns c, c + 1 (c = 8 warp + 2 (lane % 4)); x[2..3]: head lane / 4 + 8
      const int hA = lane >> 2, hB = hA + 8, cA = warp * 8 + 2 * (lane & 3);
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e)
        v[e] = (WR ? cA + (e & 1) < (tile == 0 ? wf : kTile) : tile * kTile + cA + (e & 1) < nrows) ? x[e] * scale
                                                                                                : -INFINITY;
      float mA = fmaxf(v[0], v[1]), mB = fmaxf(v[2], v[3]);
      float v2[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};  // (MG) warp 0: the merged rows' scores
      if (SEL && mx2 && warp == 0) {
        const bf16* kv2 = kvbuf(2);
        float x2[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int kk = 0; kk < 2; ++kk) {
            uint32_t bb[4];
            ldsm_x4(bb, &kv2[kvi(lane & 7, j * 8 + kk * 4 + (lane >> 3))]);
            mma16816(acc, qf[j][2 * kk], bb[0], bb[1]);
            mma16816(acc, qf[j][2 * kk + 1], bb[2], bb[3]);
          }
#pragma unroll
          for (int e = 0; e < 4; ++e) x2[e] += acc[e];
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) v2[e] = 2 * (lane & 3) + (e & 1) < wf ? x2[e] * scale : -INFINITY;
        mA = fmaxf(mA, fmaxf(v2[0], v2[1]));
        mB = fmaxf(mB, fmaxf(v2[2], v2[3]));
      }
#pragma unroll
      for (int o2 = 1; o2 < 4; o2 <<= 1) {
        mA = fmaxf(mA, __shfl_xor_sync(0xffffffff, mA, o2));
        mB = fmaxf(mB, __shfl_xor_sync(0xffffffff, mB, o2));
      }
      float* fmx = reinterpret_cast<float*>(&sm.sp);  // [warp][head] row maxima
      float* fps = fmx + NW * kHM;  // [warp][head] probability sums of the warp's 8 columns
      if ((lane & 3) == 0) {
        fmx[warp * kHM + hA] = mA;
        fmx[warp * kHM + hB] = mB;
      }
      __syncthreads();
      float tA = -INFINITY, tB = -INFINITY;
#pragma unroll
      for (int w = 0; w < NW; ++w) {
        tA = fmaxf(tA, fmx[w * kHM + hA]);
        tB = fmaxf(tB, fmx[w * kHM + hB]);
      }
      const float moA = sm.m[hA], moB = sm.m[hB];
      const float mnA = fmaxf(moA, tA), mnB = fmaxf(moB, tB);
      const __nv_bfloat162 ppA = __floats2bfloat162_rn(mnA == -INFINITY ? 0.f : __expf(v[0] - mnA),
                                                       mnA == -INFINITY ? 0.f : __expf(v[1] - mnA));
      const __nv_bfloat162 ppB = __floats2bfloat162_rn(mnB == -INFINITY ? 0.f : __expf(v[2] - mnB),
                                                       mnB == -INFINITY ? 0.f : __expf(v[3] - mnB));
      *reinterpret_cast<__nv_bfloat162*>(&sm.p[swzp(hA, warp)] + 2 * (lane & 3)) = ppA;
      *reinterpret_cast<__nv_bfloat162*>(&sm.p[swzp(hB, warp)] + 2 * (lane & 3)) = ppB;
      // the base form's sums: per 4 columns (0 + pair 0) + pair 1, then the butterfly over the 16 groups (the
      // first level inside the warp: groups 2 warp, 2 warp + 1)
      float sA = __low2float(ppA) + __high2float(ppA), sB = __low2float(ppB) + __high2float(ppB);
      if (SEL && mx2 && warp == 0) {  // (MG) the merged rows' probabilities, in warp 0's sums
        const __nv_bfloat162 p2A = __floats2bfloat162_rn(mnA == -INFINITY ? 0.f : __expf(v2[0] - mnA),
                                                         mnA == -INFINITY ? 0.f : __expf(v2[1] - mnA));
        const __nv_bfloat162 p2B = __floats2bfloat162_rn(mnB == -INFINITY ? 0.f : __expf(v2[2] - mnB),
                                                         mnB == -INFINITY ? 0.f : __expf(v2[3] - mnB));
        *reinterpret_cast<__nv_bfloat162*>(p2 + hA * 16 + 2 * (lane & 3)) = p2A;
        *reinterpret_cast<__nv_bfloat162*>(p2 + hB * 16 + 2 * (lane & 3)) = p2B;
        sA += __low2float(p2A) + __high2float(p2A);
        sB += __low2float(p2B) + __high2float(p2B);
      }
      sA += __shfl_xor_sync(0xffffffff, sA, 1);
      sB += __shfl_xor_sync(0xffffffff, sB, 1);
      sA += __shfl_xor_sync(0xffffffff, sA, 2);
      sB += __shfl_xor_sync(0xffffffff, sB, 2);
      if ((lane & 3) == 0) {
        fps[warp * kHM + hA] = sA;
        fps[warp * kHM + hB] = sB;
      }
      fa_al[0] = moA == -INFINITY ? 0.f : __expf(moA - mnA);
      fa_al[1] = moB == -INFINITY ? 0.f : __expf(moB - mnB);
      __syncthreads();
      if (tid < kHM) {  // (read by the next tile after its first barrier, and by the tail)
        const float* a = fps + tid;
        const float sum = ((a[0] + a[kHM]) + (a[2 * kHM] + a[3 * kHM])) +
                          ((a[4 * kHM] + a[5 * kHM]) + (a[6 * kHM] + a[7 * kHM]));
        float t = -INFINITY;
#pragma unroll
        for (int w = 0; w < NW; ++w) t = fmaxf(t, fmx[w * kHM + tid]);
        const float m_old = sm.m[tid], m_new = fmaxf(m_old, t);
        const float alpha = m_old == -INFINITY ? 0.f : __expf(m_old - m_new);
        sm.l[tid] = sm.l[tid] * alpha + sum;
        sm.m[tid] = m_new;
      }
    } else {
      // ---- partial scores over this warp's slice of the latent dim
      float s[8][4];
#pragma unroll
      for (int i = 0; i < 8; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < KS; ++ks) {
        const int kc = warp * 2 * KS + ks * 2;  // chunk index of k0
        uint32_t a[4];
        ldsm_x4(a, &sm.q[swz(lane & 15, kc + (lane >> 4))]);
#pragma unroll
        for (int np = 0; np < 4; ++np) {
          if (np >= vg) continue;
          uint32_t bb[4];
          const int r = np * 16 + (lane & 7) + ((lane >> 4) << 3);
          ldsm_x4(bb, &kv[kvi(r, kc + ((lane >> 3) & 1))]);
          mma16816(s[2 * np], a, bb[0], bb[1]);
          mma16816(s[2 * np + 1], a, bb[2], bb[3]);
        }
      }
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int col = nt * 8 + 2 * (lane & 3), row = lane >> 2;
        sm.sp[warp][row * kSpStride + col] = s[nt][0];
        sm.sp[warp][row * kSpStride + col + 1] = s[nt][1];
        sm.sp[warp][(row + 8) * kSpStride + col] = s[nt][2];
        sm.sp[warp][(row + 8) * kSpStride + col + 1] = s[nt][3];
      }
      __syncthreads();
      // ---- online softmax: thread -> (row = tid / TPR, CPT columns)
      {
        const int row = tid / TPR, c0 = (tid % TPR) * CPT;
        float v[CPT], mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < CPT; ++i) {
          const int col = c0 + i;
          float x = 0.f;
#pragma unroll
          for (int w = 0; w < NW; ++w) x += sm.sp[w][row * kSpStride + col];
          x = (tile * kTile + col < nrows) ? x * scale : -INFINITY;
          v[i] = x;
          mx = fmaxf(mx, x);
        }
#pragma unroll
        for (int o2 = 1; o2 < TPR; o2 <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o2));
        const float m_old = sm.m[row];
        const float m_new = fmaxf(m_old, mx);
        float sum = 0.f;
        uint32_t pk[CPT / 2];
#pragma unroll
        for (int i = 0; i < CPT / 2; ++i) {
          const float p0 = m_new == -INFINITY ? 0.f : __expf(v[2 * i] - m_new);
          const float p1 = m_new == -INFINITY ? 0.f : __expf(v[2 * i + 1] - m_new);
          const __nv_bfloat162 pp = __floats2bfloat162_rn(p0, p1);
          pk[i] = *reinterpret_cast<const uint32_t*>(&pp);
          sum += __low2float(pp) + __high2float(pp);
        }
        // columns [c0, c0 + CPT) of the row: chunk c0 / 8 (all of it, or half for CPT 4)
        bf16* pdst = &sm.p[swzp(row, c0 >> 3)] + (c0 & 7);
        if constexpr (CPT == 8)
          *reinterpret_cast<uint4*>(pdst) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
        else
          *reinterpret_cast<uint2*>(pdst) = make_uint2(pk[0], pk[1]);
#pragma unroll
        for (int o2 = 1; o2 < TPR; o2 <<= 1) sum += __shfl_xor_sync(0xffffffff, sum, o2);
        __syncwarp();
        if ((tid % TPR) == 0) {
          const float alpha = m_old == -INFINITY ? 0.f : __expf(m_old - m_new);
          sm.alpha[row] = alpha;
          sm.l[row] = sm.l[row] * alpha + sum;
          sm.m[row] = m_new;
        }
      }
      __syncthreads();
    }
    // ---- O = alpha * O + P V over this warp's slice of the latent dim
    {
      const float a0 = FA ? fa_al[0] : sm.alpha[lane >> 2], a1 = FA ? fa_al[1] : sm.alpha[(lane >> 2) + 8];
#pragma unroll
      for (int i = 0; i < 2 * NP; ++i) {
        o[i][0] *= a0;
        o[i][1] *= a0;
        o[i][2] *= a1;
        o[i][3] *= a1;
      }
#pragma unroll
      for (int ks = 0; ks < kTile / 16; ++ks) {
        if (ks >= vg) continue;
        uint32_t a[4];
        ldsm_x4(a, &sm.p[swzp(lane & 15, ks * 2 + (lane >> 4))]);
#pragma unroll
        for (int np = 0; np < NP; ++np) {
          uint32_t bb[4];
          const int r = ks * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
          const int c = warp * 2 * NP + np * 2 + (lane >> 4);
          ldsm_x4_t(bb, &kv[kvi(r, c)]);
          mma16816(o[2 * np], a, bb[0], bb[1]);
          mma16816(o[2 * np + 1], a, bb[2], bb[3]);
        }
      }
      if (SEL && mx2) {  // (MG) the merged rows: one more k-step (rows 8-15 of the block zero, their probabilities 0)
        const bf16* kv2 = kvbuf(2);
        uint32_t a[4];
        ldsm_x4(a, p2 + (lane & 15) * 16 + (lane >> 4) * 8);
#pragma unroll
        for (int np = 0; np < NP; ++np) {
          uint32_t bb[4];
          const int r = (lane & 7) + (((lane >> 3) & 1) << 3);
          const int c = warp * 2 * NP + np * 2 + (lane >> 4);
          ldsm_x4_t(bb, &kv2[kvi(r, c)]);
          mma16816(o[2 * np], a, bb[0], bb[1]);
          mma16816(o[2 * np + 1], a, bb[2], bb[3]);
        }
      }
    }
    __syncthreads();
    // (buffer consumed; P3: the partial tile's buffer is not reused, tile i >= 1's takes tile i + 2)
    if (tile + 2 < ntiles && !(p3 && tile == tile0)) load_tile(tile + 2 - tile0, buf);
  }
  if constexpr (CL) {
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    // every split pushes its unnormalized partial rows, max and sum to the CTA finalizing each head (head
    // h: CTA h % nsplit, slot h / nsplit): the rows staged in the consumed first KV buffer, then pushed by a
    // warp per row with 16-byte stores (512 contiguous bytes per store and destination); the receive areas
    // are not used before, so one barrier orders the pushes before the reads
    constexpr int kStg = kLatent + 8;  // (staging pitch: conflict-free float2 stores)
    float* stg = reinterpret_cast<float*>(&sm.kv[0][0]);
    {
      const int r0 = lane >> 2, r1 = r0 + 8;
#pragma unroll
      for (int nt = 0; nt < 2 * NP; ++nt) {
        const int col = warp * (kLatent / NW) + nt * 8 + 2 * (lane & 3);
        *reinterpret_cast<float2*>(stg + r0 * kStg + col) = make_float2(o[nt][0], o[nt][1]);
        *reinterpret_cast<float2*>(stg + r1 * kStg + col) = make_float2(o[nt][2], o[nt][3]);
      }
      if (tid < Hc) {
        const int slot = (tid / nsplit) * nsplit + split, dst = tid % nsplit;
        *cluster.map_shared_rank(&sm.rm[slot], dst) = sm.m[tid];
        *cluster.map_shared_rank(&sm.rl[slot], dst) = sm.l[tid];
      }
    }
    __syncthreads();
    for (int h = empty ? Hc : warp; h < Hc; h += NW) {  // (an empty split's rows are never read: max -inf)
      float* d = cluster.map_shared_rank(&sm.recv[(h / nsplit) * nsplit + split][0], h % nsplit);
      const float* sr = stg + h * kStg;
#pragma unroll
      for (int c = lane * 4; c < kLatent; c += 128)
        *reinterpret_cast<float4*>(d + c) = *reinterpret_cast<const float4*>(sr + c);
    }
    cluster.sync();  // every partial of this CTA's heads has arrived (no global stores pending)
    if constexpr (WR) {
      // (CLW) heads split and split + nsplit at once: the factors of each by warp k, then every column pair of
      // both; per head mla_attn_combine_kernel's operations in split order (the same bits as below)
      const int nk = split < Hc ? (Hc - 1 - split) / nsplit + 1 : 0;
      float* fac = reinterpret_cast<float*>(&sm.sp);  // [k][split]
      if (warp < nk) {
        const int s0 = warp * nsplit;
        const float ms = lane < nsplit ? sm.rm[s0 + lane] : -INFINITY;
        const float ls = lane < nsplit ? sm.rl[s0 + lane] : 0.f;
        float m = ms;
#pragma unroll
        for (int o2 = 16; o2 > 0; o2 >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o2));
        const bool ok = ms != -INFINITY;
        const float f = ok ? __expf(ms - m) : 0.f;
        if (lane < nsplit) fac[warp * kMaxAttnSplits + lane] = f;
        const unsigned vm = __ballot_sync(0xffffffffu, ok && lane < nsplit);
        float l = 0.f;
#pragma unroll
        for (int sp = 0; sp < kMaxAttnSplits; ++sp) {
          const float fs = __shfl_sync(0xffffffffu, f, sp), lsp = __shfl_sync(0xffffffffu, ls, sp);
          if ((vm >> sp) & 1u) l += lsp * fs;
        }
        if (lane == 0) {
          sm.l[warp] = l > 0.f ? 1.f / l : 0.f;
          reinterpret_cast<unsigned*>(sm.m)[warp] = vm;
        }
      }
      __syncthreads();
      for (int e = tid; e < nk * (kLatent / 2); e += NT) {
        const int k = e / (kLatent / 2), c = (e % (kLatent / 2)) * 2, s0 = k * nsplit;
        const float inv = sm.l[k];
        const unsigned vm = reinterpret_cast<const unsigned*>(sm.m)[k];
        float a0 = 0.f, a1 = 0.f;
#pragma unroll 4
        for (int sp = 0; sp < nsplit; ++sp) {
          if ((vm >> sp) & 1u) {
            const float2 v = *reinterpret_cast<const float2*>(&sm.recv[s0 + sp][c]);
            const float f = fac[k * kMaxAttnSplits + sp];
            a0 += v.x * f;
            a1 += v.y * f;
          }
        }
        *reinterpret_cast<__nv_bfloat162*>(out + ((size_t)t * H + h0 + split + k * nsplit) * kLatent + c) =
            __floats2bfloat162_rn(a0 * inv, a1 * inv);
      }
      if constexpr (SEL) {  // (the last CTA past the wait resets the counters: the next mla_qba is a later step's)
        if (tid == 0 && atomicAdd(sa.cnt + 1, 1u) == gridDim.x * gridDim.y * gridDim.z - 1) {
          sa.cnt[0] = 0u;
          sa.cnt[1] = 0u;
        }
      }
      return;
    }
    // heads split, split + nsplit, ...: mla_attn_combine_kernel's operations in split order (the factors
    // e^{m_s - m} once per split, the sum by one thread: the same bits)
    for (int k = 0; split + k * nsplit < Hc; ++k) {
      const int hh = split + k * nsplit, s0 = k * nsplit;
      if (warp == 0) {
        const float ms = lane < nsplit ? sm.rm[s0 + lane] : -INFINITY;
        const float ls = lane < nsplit ? sm.rl[s0 + lane] : 0.f;
        float m = ms;
#pragma unroll
        for (int o2 = 16; o2 > 0; o2 >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o2));
        const bool ok = ms != -INFINITY;
        const float f = ok ? __expf(ms - m) : 0.f;
        if (lane < nsplit) sm.alpha[lane] = f;
        const unsigned vm = __ballot_sync(0xffffffffu, ok && lane < nsplit);
        float l = 0.f;
#pragma unroll
        for (int sp = 0; sp < kMaxAttnSplits; ++sp) {
          const float fs = __shfl_sync(0xffffffffu, f, sp), lsp = __shfl_sync(0xffffffffu, ls, sp);
          if ((vm >> sp) & 1u) l += lsp * fs;
        }
        if (lane == 0) {
          sm.l[0] = l > 0.f ? 1.f / l : 0.f;
          reinterpret_cast<unsigned*>(sm.m)[0] = vm;
        }
      }
      __syncthreads();
      const float inv = sm.l[0];
      const unsigned vm = reinterpret_cast<const unsigned*>(sm.m)[0];
      for (int c = tid * 2; c < kLatent; c += NT * 2) {
        float a0 = 0.f, a1 = 0.f;
#pragma unroll 4
        for (int sp = 0; sp < nsplit; ++sp) {
          if ((vm >> sp) & 1u) {
            const float2 v = *reinterpret_cast<const float2*>(&sm.recv[s0 + sp][c]);
            const float f = sm.alpha[sp];
            a0 += v.x * f;
            a1 += v.y * f;
          }
        }
        *reinterpret_cast<__nv_bfloat162*>(out + ((size_t)t * H + h0 + hh) * kLatent + c) =
            __floats2bfloat162_rn(a0 * inv, a1 * inv);
      }
      __syncthreads();  // (sm.alpha / sm.l[0] / the mask of the next head)
    }
    return;
  }
  if (nsplit > 1) {
    // partial result: unnormalized O, running max and sum per head row
    __syncthreads();
    float* w = ws + ((size_t)t * nsplit + split) * kHM * (kLatent + 2);
    const int r0 = lane >> 2, r1 = r0 + 8;
#pragma unroll
    for (int nt = 0; nt < 2 * NP; ++nt) {
      const int col = warp * (kLatent / NW) + nt * 8 + 2 * (lane & 3);
      *reinterpret_cast<float2*>(w + (size_t)r0 * (kLatent + 2) + col) = make_float2(o[nt][0], o[nt][1]);
      *reinterpret_cast<float2*>(w + (size_t)r1 * (kLatent + 2) + col) = make_float2(o[nt][2], o[nt][3]);
    }
    if (tid < kHM) {
      w[(size_t)tid * (kLatent + 2) + kLatent] = sm.m[tid];
      w[(size_t)tid * (kLatent + 2) + kLatent + 1] = sm.l[tid];
    }
    return;
  }
  // ---- normalize and store
  const int r0 = lane >> 2, r1 = r0 + 8;
  const float inv0 = sm.l[r0] > 0.f ? 1.f / sm.l[r0] : 0.f;
  const float inv1 = sm.l[r1] > 0.f ? 1.f / sm.l[r1] : 0.f;
#pragma unroll
  for (int nt = 0; nt < 2 * NP; ++nt) {
    const int col = warp * (kLatent / NW) + nt * 8 + 2 * (lane & 3);
    if (r0 < Hc)
      *reinterpret_cast<__nv_bfloat162*>(out + ((size_t)t * H + h0 + r0) * kLatent + col) =
          __floats2bfloat162_rn(o[nt][0] * inv0, o[nt][1] * inv0);
    if (r1 < Hc)
      *reinterpret_cast<__nv_bfloat162*>(out + ((size_t)t * H + h0 + r1) * kLatent + col) =
          __floats2bfloat162_rn(o[nt][2] * inv1, o[nt][3] * inv1);
  }
}

// ------------------------------------------------------------------ query-tiled sparse attention
// One CTA per query tile: up to kTileTok consecutive tokens of one sequence (consecutive
// positions), their H heads stacked as Q rows (tokens * H <= 64). The CTA streams the union of
// the tokens' visible cache rows once, masking each (token, row) pair by the token's own
// visibility: a selected pool (sparse) or the dense prefix, the incomplete tail pool, causality.
constexpr int kTQ = 64;               // Q rows per CTA
constexpr int kTK = 32;               // cache rows per KV tile
constexpr int kTKP = kTK / kKpool;    // pools per KV tile
constexpr int kTThreads = 256;        // 8 warps: m-tile (0..3) x latent half (0..1)
constexpr int kTileTok = 8;
constexpr int kMaxUnion = kTileTok * (kSelPools + 1);
constexpr int kSpS = kTK + 4;

struct TAttnSmem {
  bf16 q[kTQ * kLatent];        // swizzled rows
  bf16 kv[2][kTK * kLatent];    // swizzled rows
  float sp[2][kTQ * kSpS];      // partial scores per latent half
  bf16 p[kTQ * kTK];            // probabilities
  float m[kTQ], l[kTQ], alpha[kTQ];
  int64_t rows[2][kTK];
  int kpos[2][kTK];             // position of each tile row (-1: padding)
  uint32_t kmem[2][kTK];        // tokens that may see each tile row (bit j)
  int pools[kMaxUnion];         // sorted union of visible pools
  uint32_t member[kMaxUnion];   // bit j: pool visible to token j
  int tok_pos[kTileTok], tok_ncand[kTileTok];
  int scan[kTThreads];
  int npools;
};

// P tile [64, 32]: 4 chunks per 64-byte row
__device__ __forceinline__ int swzp4(int row, int chunk) {
  return row * kTK + ((chunk ^ ((row >> 1) & 3)) << 3);
}

__global__ void __launch_bounds__(kTThreads) sparse_mla_attn_tiled_kernel(
    bf16* __restrict__ out, const bf16* __restrict__ q, const bf16* __restrict__ cache,
    const int* __restrict__ sel, const MlaTokMeta meta, const int2* __restrict__ tiles, int H,
    float scale, int bitmap_words) {
  pdl_entry();
  extern __shared__ __align__(128) uint8_t smem_raw[];
  TAttnSmem& sm = *reinterpret_cast<TAttnSmem*>(smem_raw);
  uint32_t* bitmap = reinterpret_cast<uint32_t*>(smem_raw + sizeof(TAttnSmem));
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int2 tile = tiles[blockIdx.x];
  const int t0 = tile.x, nt = tile.y, rowsQ = nt * H;
  const int b = meta.token_seq[t0];
  if (tid < nt) {
    const int pos = meta.positions[t0 + tid];
    sm.tok_pos[tid] = pos;
    sm.tok_ncand[tid] = (pos + 1) / kKpool;
  }
  for (int i = tid; i < kTQ * kChunks; i += kTThreads) {
    const int r = i / kChunks, c = i % kChunks;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (r < rowsQ) v = reinterpret_cast<const uint4*>(q + ((size_t)t0 * H + r) * kLatent)[c];
    *reinterpret_cast<uint4*>(&sm.q[swz(r, c)]) = v;
  }
  if (tid < kTQ) {
    sm.m[tid] = -INFINITY;
    sm.l[tid] = 0.f;
  }
  for (int i = tid; i < bitmap_words; i += kTThreads) bitmap[i] = 0;
  __syncthreads();
  // ---- union of visible pools: dense prefix [0, ncand] or selected pools + tail pool ncand
  for (int j = 0; j < nt; ++j) {
    const int nc = sm.tok_ncand[j];
    if (nc <= kSelPools) {
      for (int w = tid; w <= nc / 32; w += kTThreads) {
        const int lo = w * 32, hi = min(nc, lo + 31);  // bits [lo, hi]
        const uint32_t bits = hi - lo == 31 ? 0xffffffffu : ((2u << (hi - lo)) - 1u);
        atomicOr(&bitmap[w], bits);
      }
    } else {
      const int* ts = sel + (size_t)(t0 + j) * kSelPools;
      for (int k = tid; k < kSelPools; k += kTThreads) {
        const int pl = ts[k];
        atomicOr(&bitmap[pl >> 5], 1u << (pl & 31));
      }
      if (tid == 0) atomicOr(&bitmap[nc >> 5], 1u << (nc & 31));
    }
  }
  __syncthreads();
  // ---- compact the bitmap into the sorted pool list (block exclusive scan of popcounts)
  {
    const int per = (bitmap_words + kTThreads - 1) / kTThreads;
    const int w0 = tid * per, w1 = min(bitmap_words, w0 + per);
    int cnt = 0;
    for (int w = w0; w < w1; ++w) cnt += __popc(bitmap[w]);
    sm.scan[tid] = cnt;
    __syncthreads();
    for (int o = 1; o < kTThreads; o <<= 1) {
      const int v = tid >= o ? sm.scan[tid - o] : 0;
      __syncthreads();
      sm.scan[tid] += v;
      __syncthreads();
    }
    int idx = sm.scan[tid] - cnt;
    for (int w = w0; w < w1; ++w) {
      uint32_t bits = bitmap[w];
      while (bits) {
        const int bit = __ffs(bits) - 1;
        bits &= bits - 1;
        sm.pools[idx++] = w * 32 + bit;
      }
    }
    if (tid == kTThreads - 1) sm.npools = sm.scan[tid];
  }
  __syncthreads();
  const int npools = sm.npools;
  // ---- membership: dense tokens / tail pools arithmetically, sparse selections by search
  for (int i = tid; i < npools; i += kTThreads) {
    const int pl = sm.pools[i];
    uint32_t bits = 0;
    for (int j = 0; j < nt; ++j) {
      const int nc = sm.tok_ncand[j];
      if (nc <= kSelPools ? pl <= nc : pl == nc) bits |= 1u << j;
    }
    sm.member[i] = bits;
  }
  __syncthreads();
  for (int j = 0; j < nt; ++j) {
    if (sm.tok_ncand[j] <= kSelPools) continue;
    const int* ts = sel + (size_t)(t0 + j) * kSelPools;
    for (int k = tid; k < kSelPools; k += kTThreads) {
      const int pl = ts[k];
      int lo = 0, hi = npools - 1;
      while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (sm.pools[mid] < pl) lo = mid + 1; else hi = mid;
      }
      atomicOr(&sm.member[lo], 1u << j);
    }
  }
  // ---- attention over the union, kTK rows (kTKP pools) per tile
  const int ntile = (npools + kTKP - 1) / kTKP;
  auto set_rows = [&](int tl, int buf) {
    if (tid < kTK) {
      const int i = tl * kTKP + tid / kKpool;
      if (i < npools) {
        const int pos = sm.pools[i] * kKpool + tid % kKpool;
        sm.kpos[buf][tid] = pos;
        sm.kmem[buf][tid] = sm.member[i];
        sm.rows[buf][tid] = phys_row(meta, b, pos);
      } else {
        sm.kpos[buf][tid] = -1;
        sm.kmem[buf][tid] = 0;
        sm.rows[buf][tid] = -1;
      }
    }
  };
  auto load_tile = [&](int buf) {
    for (int i = tid; i < kTK * kChunks; i += kTThreads) {
      const int r = i / kChunks, c = i % kChunks;
      const int64_t pr = sm.rows[buf][r];
      cp_async16(&sm.kv[buf][swz(r, c)], cache + (pr < 0 ? 0 : pr * kLatent) + c * 8, pr >= 0);
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };
  __syncthreads();  // membership complete
  set_rows(0, 0);
  __syncthreads();
  load_tile(0);
  const int mi = warp & 3, hd = warp >> 2;  // m-tile, latent half
  const bool active = mi * 16 < rowsQ;
  float o[32][4];
#pragma unroll
  for (int i = 0; i < 32; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  for (int tl = 0; tl < ntile; ++tl) {
    const int buf = tl & 1;
    if (tl + 1 < ntile) {
      set_rows(tl + 1, buf ^ 1);
      __syncthreads();
      load_tile(buf ^ 1);
      asm volatile("cp.async.wait_group 1;\n" ::);
    } else {
      asm volatile("cp.async.wait_group 0;\n" ::);
    }
    __syncthreads();
    const bf16* kv = sm.kv[buf];
    // ---- partial scores: rows [16 mi, +16) x tile rows, over latent half hd
    if (active) {
      float sc[4][4];
#pragma unroll
      for (int i = 0; i < 4; ++i) sc[i][0] = sc[i][1] = sc[i][2] = sc[i][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 16; ++ks) {
        const int kc = hd * 32 + ks * 2;
        uint32_t a[4];
        ldsm_x4(a, &sm.q[swz(mi * 16 + (lane & 15), kc + (lane >> 4))]);
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          uint32_t bb[4];
          const int r = np * 16 + (lane & 7) + ((lane >> 4) << 3);
          ldsm_x4(bb, &kv[swz(r, kc + ((lane >> 3) & 1))]);
          mma16816(sc[2 * np], a, bb[0], bb[1]);
          mma16816(sc[2 * np + 1], a, bb[2], bb[3]);
        }
      }
      float* spw = sm.sp[hd];
#pragma unroll
      for (int nt8 = 0; nt8 < 4; ++nt8) {
        const int col = nt8 * 8 + 2 * (lane & 3), row = mi * 16 + (lane >> 2);
        spw[row * kSpS + col] = sc[nt8][0];
        spw[row * kSpS + col + 1] = sc[nt8][1];
        spw[(row + 8) * kSpS + col] = sc[nt8][2];
        spw[(row + 8) * kSpS + col + 1] = sc[nt8][3];
      }
    }
    __syncthreads();
    // ---- masked online softmax: thread -> (row = tid / 4, 8 columns)
    {
      const int row = tid >> 2, c0 = (tid & 3) * 8;
      const int j = row / H;
      const bool rvalid = row < rowsQ;
      const int tpos = rvalid ? sm.tok_pos[j] : -1;
      float v[8], mx = -INFINITY;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int col = c0 + i;
        float x = sm.sp[0][row * kSpS + col] + sm.sp[1][row * kSpS + col];
        const int kp = sm.kpos[buf][col];
        const bool ok = rvalid && kp >= 0 && kp <= tpos && ((sm.kmem[buf][col] >> j) & 1u);
        x = ok ? x * scale : -INFINITY;
        v[i] = x;
        mx = fmaxf(mx, x);
      }
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
      const float m_old = sm.m[row];
      const float m_new = fmaxf(m_old, mx);
      float sum = 0.f;
      uint4 pk;
      __nv_bfloat162* ph = reinterpret_cast<__nv_bfloat162*>(&pk);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float p0 = m_new == -INFINITY ? 0.f : __expf(v[2 * i] - m_new);
        const float p1 = m_new == -INFINITY ? 0.f : __expf(v[2 * i + 1] - m_new);
        const __nv_bfloat162 pp = __floats2bfloat162_rn(p0, p1);
        ph[i] = pp;
        sum += __low2float(pp) + __high2float(pp);
      }
      *reinterpret_cast<uint4*>(&sm.p[swzp4(row, tid & 3)]) = pk;
      sum += __shfl_xor_sync(0xffffffff, sum, 1);
      sum += __shfl_xor_sync(0xffffffff, sum, 2);
      __syncwarp();
      if ((tid & 3) == 0) {
        const float alpha = m_old == -INFINITY ? 0.f : __expf(m_old - m_new);
        sm.alpha[row] = alpha;
        sm.l[row] = sm.l[row] * alpha + sum;
        sm.m[row] = m_new;
      }
    }
    __syncthreads();
    // ---- O = alpha * O + P V: rows [16 mi, +16) x latent half hd
    if (active) {
      const float a0 = sm.alpha[mi * 16 + (lane >> 2)], a1 = sm.alpha[mi * 16 + (lane >> 2) + 8];
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        o[i][0] *= a0;
        o[i][1] *= a0;
        o[i][2] *= a1;
        o[i][3] *= a1;
      }
#pragma unroll
      for (int ks = 0; ks < kTK / 16; ++ks) {
        uint32_t a[4];
        ldsm_x4(a, &sm.p[swzp4(mi * 16 + (lane & 15), ks * 2 + (lane >> 4))]);
#pragma unroll
        for (int np = 0; np < 16; ++np) {
          uint32_t bb[4];
          const int r = ks * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
          const int c = hd * 32 + np * 2 + (lane >> 4);
          ldsm_x4_t(bb, &kv[swz(r, c)]);
          mma16816(o[2 * np], a, bb[0], bb[1]);
          mma16816(o[2 * np + 1], a, bb[2], bb[3]);
        }
      }
    }
    __syncthreads();
  }
  // ---- normalize and store
  if (active) {
    const int r0 = mi * 16 + (lane >> 2), r1 = r0 + 8;
    const float inv0 = sm.l[r0] > 0.f ? 1.f / sm.l[r0] : 0.f;
    const float inv1 = sm.l[r1] > 0.f ? 1.f / sm.l[r1] : 0.f;
#pragma unroll
    for (int nt8 = 0; nt8 < 32; ++nt8) {
      const int col = hd * 256 + nt8 * 8 + 2 * (lane & 3);
      if (r0 < rowsQ)
        *reinterpret_cast<__nv_bfloat162*>(out + ((size_t)t0 * H + r0) * kLatent + col) =
            __floats2bfloat162_rn(o[nt8][0] * inv0, o[nt8][1] * inv0);
      if (r1 < rowsQ)
        *reinterpret_cast<__nv_bfloat162*>(out + ((size_t)t0 * H + r1) * kLatent + col) =
            __floats2bfloat162_rn(o[nt8][2] * inv1, o[nt8][3] * inv1);
    }
  }
}

int sparse_mla_tile_tokens(int H) { return std::min(kTileTok, std::max(1, kTQ / H)); }

void sparse_mla_attn_tiled(bf16* out, const bf16* q, const bf16* latent_cache, const int* sel,
                           const MlaTokMeta& meta, const int2* tiles, int ntiles, int H,
                           int max_pools, float scale, cudaStream_t st, int64_t q_rows,
                           int64_t cache_rows, void* ws, size_t ws_bytes) {
  if (ntiles == 0) return;
  if (sparse_mla_attn_umma(out, q, q_rows, latent_cache, cache_rows, sel, meta, tiles, ntiles, H,
                           max_pools, scale, ws, ws_bytes, st))
    return;
  CHECK(H <= kTQ && kTQ % H == 0, "tiled sparse attention: heads must divide 64");
  const int words = (max_pools + 1 + 31) / 32;
  const size_t smem = sizeof(TAttnSmem) + (size_t)words * 4;
  CHECK(smem <= 227 * 1024, "tiled sparse attention: context too long for the pool bitmap");
  static size_t configured[64] = {};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  if (configured[dev] < smem) {
    CUDA_CHECK(cudaFuncSetAttribute(sparse_mla_attn_tiled_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    configured[dev] = smem;
  }
  launch_pdl(sparse_mla_attn_tiled_kernel, dim3(ntiles), dim3(kTThreads), smem, st, out, q,
             latent_cache, sel, meta, tiles, H, scale, words);
}

// out[t, h] = sum_s O_s e^{m_s - m} / sum_s l_s e^{m_s - m} over the splits; grid (T, H). The
// splits' max / sum first (one load each, in parallel), the factors once per head, then every
// split's column pair loaded at once (the sums in split order: the same operations as one serial
// loop per column pair).
__global__ void __launch_bounds__(256) mla_attn_combine_kernel(bf16* __restrict__ out,
                                                               const float* __restrict__ ws,
                                                               int H, int nsplit) {
  pdl_entry();
  constexpr int kS = kMaxAttnSplits;
  const int t = blockIdx.x, h = blockIdx.y;
  const int groups = (H + kHM - 1) / kHM;
  const size_t stride = (size_t)kHM * (kLatent + 2);
  const float* w = ws + ((size_t)t * groups + h / kHM) * nsplit * stride +
                   (size_t)(h % kHM) * (kLatent + 2);
  __shared__ float s_m[kS], s_l[kS], s_inv;
  if (threadIdx.x < kS) {
    const int sp = threadIdx.x;
    s_m[sp] = sp < nsplit ? w[sp * stride + kLatent] : -INFINITY;
    s_l[sp] = sp < nsplit ? w[sp * stride + kLatent + 1] : 0.f;
  }
  __syncthreads();
  float m = -INFINITY;
  for (int sp = 0; sp < nsplit; ++sp) m = fmaxf(m, s_m[sp]);
  if (threadIdx.x == 0) {
    float l = 0.f;
    for (int sp = 0; sp < nsplit; ++sp) {
      const float ms = s_m[sp];
      if (ms != -INFINITY) l += s_l[sp] * __expf(ms - m);
    }
    s_inv = l > 0.f ? 1.f / l : 0.f;
  }
  __syncthreads();
  const float inv = s_inv;
  for (int c = threadIdx.x * 2; c < kLatent; c += blockDim.x * 2) {
    float2 v[kS];
#pragma unroll
    for (int sp = 0; sp < kS; ++sp)
      if (sp < nsplit && s_m[sp] != -INFINITY)
        v[sp] = *reinterpret_cast<const float2*>(w + sp * stride + c);
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int sp = 0; sp < kS; ++sp) {
      const float ms = s_m[sp];
      if (sp < nsplit && ms != -INFINITY) {
        const float f = __expf(ms - m);
        a0 += v[sp].x * f;
        a1 += v[sp].y * f;
      }
    }
    *reinterpret_cast<__nv_bfloat162*>(out + ((size_t)t * H + h) * kLatent + c) =
        __floats2bfloat162_rn(a0 * inv, a1 * inv);
  }
}

// out[t, h] = sum_s O_s e^{m_s - m} / sum_s l_s e^{m_s - m} over the splits; grid (T, H), 256 threads
// (one column pair each), for the FlashMLA decode path: every load at once (each thread its column
// pair of every split and every split's max / sum: one memory round trip), then each thread the
// combine in split order (the math of mla_attn_combine_kernel, not bitwise: contraction may differ).
__global__ void __launch_bounds__(256) mla_attn_combine1_kernel(bf16* __restrict__ out,
                                                               const float* __restrict__ ws,
                                                               int H, int nsplit) {
  pdl_entry();
  constexpr int kS = kMaxAttnSplits;
  const int t = blockIdx.x, h = blockIdx.y;
  const int groups = (H + kHM - 1) / kHM;
  const size_t stride = (size_t)kHM * (kLatent + 2);
  const float* w = ws + ((size_t)t * groups + h / kHM) * nsplit * stride +
                   (size_t)(h % kHM) * (kLatent + 2);
  const int c = threadIdx.x * 2;
  float ms[kS], ls[kS];
  float2 v[kS];
#pragma unroll
  for (int sp = 0; sp < kS; ++sp)
    if (sp < nsplit) {
      ms[sp] = w[sp * stride + kLatent];
      ls[sp] = w[sp * stride + kLatent + 1];
      if (c < kLatent) v[sp] = *reinterpret_cast<const float2*>(w + sp * stride + c);
    }
  float m = -INFINITY;
#pragma unroll
  for (int sp = 0; sp < kS; ++sp)
    if (sp < nsplit) m = fmaxf(m, ms[sp]);
  float l = 0.f;
#pragma unroll
  for (int sp = 0; sp < kS; ++sp)
    if (sp < nsplit && ms[sp] != -INFINITY) l += ls[sp] * __expf(ms[sp] - m);
  const float inv = l > 0.f ? 1.f / l : 0.f;
  if (c >= kLatent) return;
  float a0 = 0.f, a1 = 0.f;
#pragma unroll
  for (int sp = 0; sp < kS; ++sp) {
    if (sp < nsplit && ms[sp] != -INFINITY) {
      const float f = __expf(ms[sp] - m);
      a0 += v[sp].x * f;
      a1 += v[sp].y * f;
    }
  }
  *reinterpret_cast<__nv_bfloat162*>(out + ((size_t)t * H + h) * kLatent + c) =
      __floats2bfloat162_rn(a0 * inv, a1 * inv);
}

void mla_attn_combine(bf16* out, const float* ws, int T, int H, int nsplit, cudaStream_t st) {
  if (T <= 0) return;
  CHECK(nsplit >= 1 && nsplit <= kMaxAttnSplits, "mla_attn_combine: splits");
  launch_pdl(mla_attn_combine1_kernel, dim3(T, H), dim3(256), 0, st, out, ws, H, nsplit);
}

int sparse_mla_splits(int T, int sm_count, int max_rows, int ws_ctas, int budget_dflt) {
  // at least ~2 row tiles per split, at most one wave: the kernel's 164 KB of shared memory leaves
  // one CTA per SM. With attention DP (64 local heads: 4 head groups per token) the caller passes a
  // larger default budget and workspace: T x 4 CTAs alone fill one wave, so nsplit would be 1 (32 row
  // tiles in sequence per CTA).
  // (the partials workspace holds ws_ctas CTAs: 2 x SMs unless the caller allocated more)
  if (ws_ctas <= 0) ws_ctas = 2 * sm_count;
  if (budget_dflt <= 0) budget_dflt = sm_count;
  const int budget = std::min(budget_dflt, ws_ctas);
  int s = std::max(1, std::min(kMaxAttnSplits, budget / std::max(T, 1)));
  return std::max(1, std::min(s, max_rows / (2 * kTile)));
}

size_t sparse_mla_ws_floats(int T, int nsplit, int H) {
  return nsplit > 1 ? (size_t)T * cdiv(H, kHM) * nsplit * kHM * (kLatent + 2) : 0;
}

void sparse_mla_attn(bf16* out, const bf16* q, const bf16* latent_cache, const int* sel,
                     const MlaTokMeta& meta, int T, int H, float scale, cudaStream_t st,
                     int nsplit, float* ws, const int* vis) {
  if (T == 0) return;
  CHECK(H >= 1 && H <= 4 * kHM, "sparse attention supports at most 64 local heads");
  CHECK(nsplit >= 1 && nsplit <= kMaxAttnSplits && (nsplit == 1 || ws), "bad attention splits");
  // 8 warps with the TMA row copies; FA is the cluster form's only (its 128 query registers put the kernel at 255
  // registers: in the split form the combine's CTAs could then no longer sit beside it)
  static bool configured[64] = {};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  if (!configured[dev]) {
    CUDA_CHECK(cudaFuncSetAttribute(sparse_mla_attn_kernel<8, false, true>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(AttnSmemT<8>)));
    configured[dev] = true;
  }
  const dim3 grid(T, nsplit, (int)cdiv(H, kHM));
  launch_pdl(sparse_mla_attn_kernel<8, false, true>, grid, dim3(256), sizeof(AttnSmemT<8>), st, out, q,
             latent_cache, sel, meta, H, scale, ws, vis, SelArgs{});
  if (nsplit > 1)
    launch_pdl(mla_attn_combine_kernel, dim3(T, H), dim3(256), 0, st, out, ws, H, nsplit);
}

// The CLW instance at the largest split count n <= nsplit (>= kClwMinSplits: a long context's 33 row tiles at most 4
// per split) whose clusters of n CTAs all fit at once at one CTA per SM (16-CTA clusters: 7 on a B200; 10-CTA: 11,
// 9-CTA: 15). Up to 7 tokens (one sequence's verification, the MTP passes) that is 16 splits: a long context's
// 2049-2051 rows are 128-129 rows per split (2 full tiles + a 0-1 row partial tile on every split) instead of 3 full
// tiles on 11 of the 16 splits (the same math as sparse_mla_attn's split form, in another fp32 order).
// (pre: the PRE instance for launches with vis -- the chained MTP passes; see attn_pre_rows)
constexpr int kClwMinSplits = 9;
bool sparse_mla_attn_cl(bf16* out, const bf16* q, const bf16* latent_cache, const int* sel,
                        const MlaTokMeta& meta, int T, int H, float scale, cudaStream_t st, int nsplit,
                        const int* vis, bool pre) {
  if (T <= 0 || nsplit < 2 || nsplit > kMaxAttnSplits || H < 1 || H > 4 * kHM) return false;
  pre = pre && vis;
  auto kern = pre ? sparse_mla_attn_kernel<8, true, true, true, true, false, true>
                  : sparse_mla_attn_kernel<8, true, true, true, true>;
  const size_t smem = sizeof(AttnSmemClW<8>);
  static bool configured[64][2] = {};
  static int max_clusters[64][2][kMaxAttnSplits + 1] = {};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  if (!configured[dev][pre]) {
    CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    configured[dev][pre] = true;
  }
  const int groups = (int)cdiv(H, kHM);
  for (int n = nsplit; n >= kClwMinSplits; --n) {
    int& mc = max_clusters[dev][pre][n];
    if (mc == 0) {
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = dim3(n, 1, 1);
      cfg.blockDim = dim3(256);
      cfg.dynamicSmemBytes = smem;
      cudaLaunchAttribute at[1];
      at[0].id = cudaLaunchAttributeClusterDimension;
      at[0].val.clusterDim.x = n;
      at[0].val.clusterDim.y = 1;
      at[0].val.clusterDim.z = 1;
      cfg.attrs = at;
      cfg.numAttrs = 1;
      int c = 0;
      if (cudaOccupancyMaxActiveClusters(&c, kern, &cfg) != cudaSuccess) {
        (void)cudaGetLastError();
        c = 0;
      }
      mc = c > 0 ? c : -1;
    }
    if (mc < T * groups) continue;
    if (pre)
      launch_pdl_cluster(sparse_mla_attn_kernel<8, true, true, true, true, false, true>, dim3(n, T, groups), dim3(256),
                         smem, st, n, out, q, latent_cache, sel, meta, H, scale, (float*)nullptr, vis, SelArgs{});
    else
      launch_pdl_cluster(sparse_mla_attn_kernel<8, true, true, true, true>, dim3(n, T, groups), dim3(256), smem, st, n,
                         out, q, latent_cache, sel, meta, H, scale, (float*)nullptr, vis, SelArgs{});
    return true;
  }
  return false;
}

// SEL: the decode top-k inside the CLW attention's 16-CTA clusters (sparse_mla_attn_kernel's SEL instance; one launch
// and one dependent hop fewer than kpool_topk + sparse_mla_attn_cl): launched right after the scoring (its PDL
// predecessor, whose CTAs' leaving SMs its clusters take before mla_qba launches); waits for mla_qba's SIG instance on
// cnt (cnt_target CTAs) before it loads the query. *pev (optional): recorded as the launch's programmatic event (the
// main stream's next kernel then launches at this kernel's trigger, after the wait) and set to nullptr. False
// (nothing launched) for other shapes.
bool sparse_mla_sel_fits(int T, int H, int max_pools) {
  return T >= 1 && T <= 7 && H == kHM && max_pools - 1 <= kSkMax;
}
// the SEL instance's 16-CTA clusters that fit at once on this device (-1: none), queried at the first call per device
static int sel_max_clusters() {
  auto kern = sparse_mla_attn_kernel<8, true, true, true, true, true>;
  const size_t smem = sizeof(AttnSmemClW<8>);
  static int max_clusters[64] = {};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  int& mc = max_clusters[dev];
  if (mc == 0) {
    // (mla_qba's instances loaded now: a lazy module load while this kernel waits for one would wait for it)
    mla_qba_preload();
    CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(kSkCtas, 1, 1);
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = smem;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = kSkCtas;
    at[0].val.clusterDim.y = 1;
    at[0].val.clusterDim.z = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    int c = 0;
    if (cudaOccupancyMaxActiveClusters(&c, kern, &cfg) != cudaSuccess) {
      (void)cudaGetLastError();
      c = 0;
    }
    mc = c > 0 ? c : -1;
  }
  return mc;
}
bool sparse_mla_attn_sel_ok(int T, int H, int stride, int max_pools) {
  return sparse_mla_sel_fits(T, H, max_pools) && stride >= max_pools - 1 && sel_max_clusters() >= T;
}
bool sparse_mla_attn_sel(bf16* out, const bf16* q, const bf16* latent_cache, int* sel, const MlaTokMeta& meta,
                         int T, int H, float scale, cudaStream_t st, const float* scores, int stride, int max_pools,
                         unsigned* cnt, int cnt_target, cudaEvent_t* pev) {
  if (!sparse_mla_attn_sel_ok(T, H, stride, max_pools)) return false;
  auto kern = sparse_mla_attn_kernel<8, true, true, true, true, true>;
  const size_t smem = sizeof(AttnSmemClW<8>);
  const SelArgs sa{scores, sel, stride, cnt, cnt_target};
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(kSkCtas, T, 1);
  cfg.blockDim = dim3(256);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute at[3];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  at[1].id = cudaLaunchAttributeClusterDimension;
  at[1].val.clusterDim.x = kSkCtas;
  at[1].val.clusterDim.y = 1;
  at[1].val.clusterDim.z = 1;
  int na = 2;
  const bool ev = pev && *pev;
  if (ev) {
    at[2].id = cudaLaunchAttributeProgrammaticEvent;
    at[2].val.programmaticEvent.event = *pev;
    at[2].val.programmaticEvent.flags = 0;
    at[2].val.programmaticEvent.triggerAtBlockStart = 0;
    na = 3;
  }
  cfg.attrs = at;
  cfg.numAttrs = na;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, out, q, latent_cache, (const int*)sel, meta, H, scale, (float*)nullptr,
                                (const int*)nullptr, sa));
  if (ev) *pev = nullptr;
  return true;
}

}  // namespace k
}  // namespace glm
