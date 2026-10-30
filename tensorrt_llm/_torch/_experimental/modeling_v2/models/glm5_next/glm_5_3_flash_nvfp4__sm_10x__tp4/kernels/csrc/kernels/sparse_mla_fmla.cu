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
// Sparse MLA attention of the step's own tokens through FlashMLA's SM100 sparse prefill kernel
// (fmla_fwd.cu): every token's visible cache rows are expanded to a row-id list (the dense prefix
// [0, position] while the token sees at most 512 complete pools, else the 512 selected pools' rows
// and the rows of its incomplete tail pool), then one CTA per token attends with all 64 heads.
#include "fmla.h"
#include "mla.h"
#include "pdl.cuh"

namespace glm {
namespace k {

namespace {
constexpr int kExpThreads = 256;

// (the runtime's pages: raw V2 slot ids, lat_page_rows latent rows apart; row ids fit 31 bits, checked by
// the launchers against the pool's rows)
__device__ __forceinline__ int cache_row(const int64_t* bt, int pos, int64_t page_rows) {
  return (int)(bt[pos / kMlaBlock] * page_rows + pos % kMlaBlock);
}
}  // namespace

// grid (T): token t's row ids -> idx[t][0, width) (valid prefix len[t], -1 after it)
__global__ void __launch_bounds__(kExpThreads) mla_expand_rows_kernel(
    int* __restrict__ idx, int* __restrict__ len, const int* __restrict__ sel, const MlaTokMeta meta,
    int width) {
  pdl_entry();
  const int t = blockIdx.x, tid = threadIdx.x;
  const int pos = meta.positions[t];
  const int64_t* bt = meta.block_tables + (size_t)meta.token_seq[t] * meta.max_blocks;
  const int64_t pr = meta.lat_page_rows;
  const int nc = (pos + 1) / kKpool;
  int* out = idx + (size_t)t * width;
  if (nc <= kSelPools) {  // dense: positions [0, pos]
    const int n = pos + 1;
    for (int r = tid; r < width; r += kExpThreads) out[r] = r < n ? cache_row(bt, r, pr) : -1;
    if (tid == 0) len[t] = n;
    return;
  }
  // the selected (complete) pools: 4 consecutive rows each (a pool never straddles a page)
  const int* ts = sel + (size_t)t * kSelPools;
  for (int i = tid; i < kSelPools; i += kExpThreads) {
    const int r0 = cache_row(bt, ts[i] * kKpool, pr);
    *reinterpret_cast<int4*>(out + kKpool * i) = make_int4(r0, r0 + 1, r0 + 2, r0 + 3);
  }
  // the incomplete tail pool nc: positions [4 nc, pos]
  const int tail = pos + 1 - nc * kKpool;
  for (int r = kSelPools * kKpool + tid; r < width; r += kExpThreads) {
    const int q = r - kSelPools * kKpool;
    out[r] = q < tail ? cache_row(bt, nc * kKpool + q, pr) : -1;
  }
  if (tid == 0) len[t] = kSelPools * kKpool + tail;
}

void mla_expand_rows(int* idx, int* len, const int* sel, const MlaTokMeta& meta, int T, int width,
                     cudaStream_t st) {
  if (T <= 0) return;
  launch_pdl(mla_expand_rows_kernel, dim3(T), dim3(kExpThreads), 0, st, idx, len, sel, meta, width);
}

size_t sparse_mla_fmla_ws_bytes(int T, int H) {
  const size_t idx = (size_t)T * kFmlaRows * sizeof(int);
  return idx + (size_t)T * sizeof(int) + 2 * (size_t)T * H * sizeof(float) + 1024;
}

// rows expanded per token for 64 local heads (attention DP); fewer local heads (tensor parallel): the query-tiled
// union kernel (sparse_mla_attn_fmla_tiled)
bool sparse_mla_fmla_on(int H) { return H == 64; }

bool sparse_mla_attn_fmla_tiled(bf16* out, const bf16* q, int64_t q_rows, const bf16* cache,
                                int64_t cache_rows, const int* sel, const MlaTokMeta& meta,
                                const int2* tiles, int ntiles, int H, int max_pools, float scale,
                                void* ws, size_t ws_bytes, cudaStream_t st) {
  if (H < 8 || 64 % H != 0 || ntiles <= 0 || q_rows <= 0 || cache_rows <= 0 ||
      cache_rows > (int64_t)0x7fffffff || ws == nullptr ||
      (size_t)ntiles * sparse_mla_union_ws_bytes(H) > ws_bytes)
    return false;
  sparse_mla_union(ws, sel, meta, tiles, ntiles, H, max_pools, st);
  FmlaUnionArgs a;
  a.tiles = tiles;
  a.ntiles = ntiles;
  a.H = H;
  a.sm_scale = scale;
  a.q = q;
  a.q_rows = q_rows;
  a.kv = cache;
  a.kv_rows = cache_rows;
  a.ws = ws;
  a.union_bytes = (int)sparse_mla_union_ws_bytes(H);
  a.union_cap = sparse_mla_union_capacity(H);
  a.positions = meta.positions;
  a.out = out;
  a.stream = st;
  fmla_sparse_union_fwd(a);
  return true;
}

int sparse_mla_fmla_split_splits(int ntiles, int T, int sm_count, int partial_blocks) {
  int s = std::min(kMaxAttnSplits, sm_count / std::max(ntiles, 1));
  if (partial_blocks > 0) s = std::min(s, partial_blocks / std::max(T, 1));
  return std::max(1, s);
}

bool sparse_mla_attn_fmla_split(bf16* out, const bf16* q, int64_t q_rows, const bf16* cache,
                                int64_t cache_rows, const int* sel, const MlaTokMeta& meta,
                                const int2* tiles, int ntiles, int T, int H, int max_pools,
                                float scale, void* ws, size_t ws_bytes, float* partial_ws,
                                size_t partial_blocks, int nsplit, cudaStream_t st) {
  if (H != 16 || ntiles <= 0 || nsplit < 2 ||
      nsplit > kMaxAttnSplits || q_rows <= 0 || cache_rows <= 0 ||
      cache_rows > (int64_t)0x7fffffff || ws == nullptr || partial_ws == nullptr ||
      (size_t)ntiles * sparse_mla_union_ws_bytes(H) > ws_bytes ||
      (size_t)T * nsplit > partial_blocks)
    return false;
  sparse_mla_union_dec(ws, sel, meta, tiles, ntiles, H, max_pools, st);
  FmlaUnionArgs a;
  // the attention's prologue before its PDL wait: the pre-pass triggers it after its own wait
  a.pre = true;
  a.tiles = tiles;
  a.ntiles = ntiles;
  a.H = H;
  a.sm_scale = scale;
  a.q = q;
  a.q_rows = q_rows;
  a.kv = cache;
  a.kv_rows = cache_rows;
  a.ws = ws;
  a.union_bytes = (int)sparse_mla_union_ws_bytes(H);
  a.union_cap = sparse_mla_union_capacity(H);
  a.positions = meta.positions;
  a.out = out;
  a.stream = st;
  a.nsplit = nsplit;
  a.partial_ws = partial_ws;
  fmla_sparse_union_fwd(a);
  mla_attn_combine(out, partial_ws, T, H, nsplit, st);
  return true;
}

bool sparse_mla_attn_fmla(bf16* out, const bf16* q, const bf16* cache, int64_t cache_rows,
                          const int* sel, const MlaTokMeta& meta, int T, int H, float scale,
                          void* ws, size_t ws_bytes, int num_sm, cudaStream_t st) {
  if (!sparse_mla_fmla_on(H) || T <= 0 || ws == nullptr || cache_rows <= 0 ||
      cache_rows > (int64_t)0x7fffffff || sparse_mla_fmla_ws_bytes(T, H) > ws_bytes)
    return false;
  uint8_t* w = static_cast<uint8_t*>(ws);
  int* idx = reinterpret_cast<int*>(w);
  w += (size_t)T * kFmlaRows * sizeof(int);
  int* len = reinterpret_cast<int*>(w);
  w += ((size_t)T * sizeof(int) + 255) / 256 * 256;
  float* mlog = reinterpret_cast<float*>(w);
  float* lse = mlog + (size_t)T * H;
  mla_expand_rows(idx, len, sel, meta, T, kFmlaRows, st);
  FmlaFwdArgs a;
  a.s_q = T;
  a.s_kv = (int)cache_rows;
  a.h_q = H;
  a.topk = kFmlaRows;
  a.sm_scale = scale;
  a.q = q;
  a.stride_q_s_q = H * kLatent;
  a.stride_q_h_q = kLatent;
  a.kv = cache;
  a.stride_kv_s_kv = kLatent;
  a.indices = idx;
  a.stride_indices_s_q = kFmlaRows;
  a.topk_length = len;
  a.out = out;
  a.max_logits = mlog;
  a.lse = lse;
  a.num_sm = num_sm;
  a.stream = st;
  fmla_sparse_fwd(a);
  return true;
}

}  // namespace k
}  // namespace glm
