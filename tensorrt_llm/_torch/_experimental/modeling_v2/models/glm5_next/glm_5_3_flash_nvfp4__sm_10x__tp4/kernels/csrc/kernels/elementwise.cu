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
// Embedding, norms, FP8 activation quantization, SwiGLU and mHC kernels.
#include <cooperative_groups.h>
#include <cuda.h>
#include <cudaTypedefs.h>

#include <map>
#include <mutex>
#include <string>
#include <tuple>

#include "pdl.cuh"
#include "kernels.h"
#include "mla.h"
#include "reduce.cuh"
#include "lamport.cuh"
#include "nvfp4.h"
#include "sp_comm.h"
#include "sp_sync.cuh"

namespace glm {
namespace k {

// ------------------------------------------------------------------ embedding
__global__ void embed_streams_kernel(bf16* __restrict__ streams, const bf16* __restrict__ embed,
                                     const int* __restrict__ ids, int hidden, int hc) {
  pdl_entry();
  const int t = blockIdx.x;
  const int id = ids[t];
  const uint4* src = reinterpret_cast<const uint4*>(embed + (size_t)id * hidden);
  uint4* dst = reinterpret_cast<uint4*>(streams + (size_t)t * hc * hidden);
  const int n = hidden / 8;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    uint4 v = src[i];
    for (int j = 0; j < hc; ++j) dst[j * n + i] = v;
  }
}

void embed_streams(bf16* streams, const bf16* embed, const int* ids, int T, int hidden, int hc,
                   cudaStream_t st) {
  if (T == 0) return;
  launch_pdl(embed_streams_kernel, dim3(T), dim3(256), 0, st, streams, embed, ids, hidden, hc);
}

__global__ void gather_rows_kernel(bf16* __restrict__ dst, const bf16* __restrict__ src,
                                   const int* __restrict__ idx, int row_elems) {
  pdl_entry();
  const int r = blockIdx.x;
  const uint4* s = reinterpret_cast<const uint4*>(src + (size_t)idx[r] * row_elems);
  uint4* d = reinterpret_cast<uint4*>(dst + (size_t)r * row_elems);
  for (int i = threadIdx.x; i < row_elems / 8; i += blockDim.x) d[i] = s[i];
}

void gather_rows(bf16* dst, const bf16* src, const int* idx, int n, int row_elems,
                 cudaStream_t st) {
  if (n == 0) return;
  launch_pdl(gather_rows_kernel, dim3(n), dim3(256), 0, st, dst, src, idx, row_elems);
}

// ------------------------------------------------------------------ L2 prefetch
// Bulk L2 prefetches of a byte range (chunks of `chunk` bytes, one per thread over the grid): issued
// on a side stream while latency-bound kernels leave HBM idle, so that a following weight-streaming
// GEMM reads its weights from L2. A plain launch (no PDL): it runs once its stream dependency is
// complete and exits after issuing.
__global__ void l2_prefetch_kernel(const char* __restrict__ p, size_t bytes, uint32_t chunk) {
  const size_t n = (bytes + chunk - 1) / chunk;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    const size_t off = i * chunk;
    const uint32_t len = (uint32_t)(bytes - off < chunk ? bytes - off : chunk);
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p + off), "r"(len) : "memory");
  }
}

void l2_prefetch(const void* p, size_t bytes, int ctas, size_t chunk, cudaStream_t st) {
  if (bytes < 16) return;
  CHECK(((uintptr_t)p & 15) == 0, "l2_prefetch: 16-byte aligned start needed");
  bytes &= ~(size_t)15;
  chunk = std::max<size_t>(16, std::min<size_t>(chunk, 512 * 1024)) & ~(size_t)15;
  const size_t chunks = (bytes + chunk - 1) / chunk;
  const int grid = (int)std::min<size_t>((size_t)std::max(ctas, 1), (chunks + 31) / 32);
  l2_prefetch_kernel<<<grid, 32, 0, st>>>(static_cast<const char*>(p), bytes, (uint32_t)chunk);
  CUDA_CHECK(cudaGetLastError());
}

// MoE layer weights of the experts recorded at the previous step (MoeL2pf): every CTA ranks the
// recorded experts by token count (stable), then the threads issue the chunks of the capped list
// grid-stride. A plain launch: it reads a history the previous step's kernel wrote (a same-step
// kernel of this layer may be rewriting it meanwhile: any mix of old and new entries is a valid
// expert, so a prefetch can only miss, never touch memory outside the weights)
__global__ void __launch_bounds__(128) moe_l2_prefetch_kernel(MoeL2pf a, uint32_t chunk) {
  __shared__ int ord[512];
  __shared__ int cnt[512];
  const int hmax = min(a.hist_max, 512);
  const int n = min(max(__ldcg(a.hist), 0), hmax);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    ord[i] = __ldcg(a.hist + 1 + i);
    cnt[i] = __ldcg(a.hist + 1 + a.hist_max + i);
  }
  __syncthreads();
  int g_r[4], r_r[4];  // (each thread ranks up to 4 entries: n <= 512 with 128 threads)
  int m = 0;
  for (int i = threadIdx.x; a.by_count && i < n; i += blockDim.x, ++m) {
    const int c = cnt[i];
    int r = 0;
    for (int j = 0; j < n; ++j) r += cnt[j] > c || (cnt[j] == c && j < i);
    g_r[m] = ord[i];
    r_r[m] = r;
  }
  __syncthreads();
  for (int k = 0; k < m; ++k) ord[r_r[k]] = g_r[k];  // (ranks are a permutation: no conflicts)
  __syncthreads();
  // experts per phase within the cap (whole segments of whole experts)
  size_t b1 = a.stride[0] + a.stride[1], b2 = a.stride[2] + a.stride[3];
  int k1, k2;
  if (a.two_phase) {
    k1 = b1 ? (int)min((size_t)n, a.cap / b1) : n;
    k2 = k1 < n || !b2 ? 0 : (int)min((size_t)n, (a.cap - (size_t)n * b1) / b2);
  } else {
    k1 = k2 = b1 + b2 ? (int)min((size_t)n, a.cap / (b1 + b2)) : 0;
  }
  uint32_t c[4];
#pragma unroll
  for (int s = 0; s < 4; ++s) c[s] = (uint32_t)((a.stride[s] + chunk - 1) / chunk);
  const uint32_t p1 = c[0] + c[1], p2 = c[2] + c[3];
  const size_t n1 = (size_t)k1 * p1, total = n1 + (size_t)k2 * p2;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (size_t)gridDim.x * blockDim.x) {
    int r, s;
    uint32_t j;
    if (a.two_phase) {
      const bool ph2 = i >= n1;
      const size_t q = ph2 ? i - n1 : i;
      const uint32_t per = ph2 ? p2 : p1;
      r = (int)(q / per);
      j = (uint32_t)(q % per);
      s = ph2 ? 2 : 0;
      if (j >= c[s]) { j -= c[s]; ++s; }
    } else {
      const uint32_t per = p1 + p2;
      r = (int)(i / per);
      j = (uint32_t)(i % per);
      s = 0;
      while (s < 3 && j >= c[s]) { j -= c[s]; ++s; }
    }
    const size_t off = (size_t)j * chunk;
    if (off >= a.stride[s]) continue;
    const uint32_t len = (uint32_t)min((size_t)chunk, a.stride[s] - off) & ~15u;
    if (!len) continue;
    const int g = ord[r];
    if (g < 0 || g >= a.n_groups) continue;
    const uint8_t* q = a.seg[s] + (size_t)g * a.stride[s] + off;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(q), "r"(len) : "memory");
  }
}

void moe_l2_prefetch(const MoeL2pf& a, int ctas, size_t chunk, cudaStream_t st) {
  CHECK(a.hist && a.hist_max > 0 && a.hist_max <= 512 && a.n_groups > 0, "moe_l2_prefetch: history");
  for (int s = 0; s < 4; ++s)
    CHECK(((uintptr_t)a.seg[s] & 15) == 0 && (a.stride[s] & 15) == 0,
          "moe_l2_prefetch: 16-byte aligned segments needed");
  chunk = std::max<size_t>(16, std::min<size_t>(chunk, 512 * 1024)) & ~(size_t)15;
  moe_l2_prefetch_kernel<<<std::max(ctas, 1), 128, 0, st>>>(a, (uint32_t)chunk);
  CUDA_CHECK(cudaGetLastError());
}

// L2 eviction for timing (GEMM autotune): reads [p, p + bytes) (a multiple of 16 bytes)
__global__ void l2_flush_kernel(const float4* __restrict__ p, size_t n, float* out) {
  float acc = 0.f;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    const float4 v = p[i];
    acc += v.x + v.y + v.z + v.w;
  }
  if (acc == 1234.5f) *out = acc;
}

void l2_flush(const void* p, size_t bytes, float* sink, cudaStream_t st) {
  l2_flush_kernel<<<592, 512, 0, st>>>(static_cast<const float4*>(p), bytes / 16, sink);
  CUDA_CHECK(cudaGetLastError());
}

// ------------------------------------------------------------------ rmsnorm
// h = rmsnorm(bf16(mean_i streams[t, i, :]))
__global__ void hc_head_norm_kernel(bf16* __restrict__ out, const bf16* __restrict__ streams,
                                    const bf16* __restrict__ w, int hidden, int hc, float eps) {
  pdl_entry();
  __shared__ float sh[32];
  extern __shared__ float hbuf[];
  const int t = blockIdx.x;
  const bf16* s = streams + (size_t)t * hc * hidden;
  float ss = 0.f;
  const float inv = 1.f / hc;
  if (hc == 4 && hidden <= 8 * (int)blockDim.x) {
    // (the common shape: every load of the thread's columns issued before the sums -- one memory
    // latency instead of one per column; same columns per thread, same order: bitwise)
    bf16 v[8][4];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int d = threadIdx.x + k * blockDim.x;
#pragma unroll
      for (int i = 0; i < 4; ++i) v[k][i] = d < hidden ? s[i * hidden + d] : __float2bfloat16(0.f);
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int d = threadIdx.x + k * blockDim.x;
      if (d < hidden) {
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) acc += __bfloat162float(v[k][i]);
        const float m = bf16r(acc * inv);
        hbuf[d] = m;
        ss += m * m;
      }
    }
  } else {
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
      float acc = 0.f;
      for (int i = 0; i < hc; ++i) acc += __bfloat162float(s[i * hidden + d]);
      float m = bf16r(acc * inv);
      hbuf[d] = m;
      ss += m * m;
    }
  }
  ss = block_sum(ss, sh);
  const float r = rsqrtf(ss / hidden + eps);
  for (int d = threadIdx.x; d < hidden; d += blockDim.x)
    out[(size_t)t * hidden + d] = __float2bfloat16(hbuf[d] * r * __bfloat162float(w[d]));
}

void hc_head_norm(bf16* out, const bf16* streams, const bf16* w, int T, int hidden, int hc,
                  float eps, cudaStream_t st) {
  if (T == 0) return;
  launch_pdl(hc_head_norm_kernel, dim3(T), dim3(512), hidden * sizeof(float), st, out, streams, w, hidden, hc, eps);
}

// ------------------------------------------------------------------ MTP layer helpers
// One CTA of hidden/16 threads per row; thread i owns elements [16 i, 16 i + 16).
constexpr int kRowVec = 16;

__device__ __forceinline__ void load16(const bf16* p, float* f) {
  load8(p, f);
  load8(p + 8, f + 8);
}
__device__ __forceinline__ void store16(bf16* p, const float* f) {
  store8(p, f);
  store8(p + 8, f + 8);
}

// out[t] = [rmsnorm(e_t, enorm) | rmsnorm(hidden[t], hnorm)], e_t = embed[id_t] (zero at
// position 0), id_t = src[t] >= 0 ? sampled[src[t]] : ids[t].
__global__ void mtp_eh_norm_kernel(bf16* __restrict__ out, const bf16* __restrict__ embed,
                                   const int* __restrict__ ids, const int* __restrict__ src,
                                   const int* __restrict__ sampled,
                                   const int* __restrict__ positions,
                                   const bf16* __restrict__ hidden, const bf16* __restrict__ enorm,
                                   const bf16* __restrict__ hnorm, int H, float eps) {
  pdl_entry();
  __shared__ float sh[32];
  const int t = blockIdx.x, c0 = threadIdx.x * kRowVec;
  const int sidx = src ? src[t] : -1;
  const int id = sidx >= 0 ? sampled[sidx] : ids[t];
  float e[kRowVec], h[kRowVec], w[kRowVec];
  if (positions[t] == 0) {
#pragma unroll
    for (int i = 0; i < kRowVec; ++i) e[i] = 0.f;
  } else {
    load16(embed + (size_t)id * H + c0, e);
  }
  load16(hidden + (size_t)t * H + c0, h);
  float se = 0.f, shh = 0.f;
#pragma unroll
  for (int i = 0; i < kRowVec; ++i) {
    se += e[i] * e[i];
    shh += h[i] * h[i];
  }
  se = block_sum(se, sh);
  shh = block_sum(shh, sh);
  const float re = rsqrtf(se / H + eps), rh = rsqrtf(shh / H + eps);
  load16(enorm + c0, w);
#pragma unroll
  for (int i = 0; i < kRowVec; ++i) e[i] = e[i] * re * w[i];
  store16(out + (size_t)t * 2 * H + c0, e);
  load16(hnorm + c0, w);
#pragma unroll
  for (int i = 0; i < kRowVec; ++i) h[i] = h[i] * rh * w[i];
  store16(out + (size_t)t * 2 * H + H + c0, h);
}

void mtp_eh_norm(bf16* out, const bf16* embed, const int* ids, const int* src, const int* sampled,
                 const int* positions, const bf16* hidden, const bf16* enorm, const bf16* hnorm,
                 int T, int hidden_size, float eps, cudaStream_t st) {
  if (T == 0) return;
  CHECK(hidden_size % (kRowVec * 32) == 0 && hidden_size / kRowVec <= 1024,
        "mtp_eh_norm: unsupported hidden size");
  launch_pdl(mtp_eh_norm_kernel, dim3(T), dim3(hidden_size / kRowVec), 0, st, out, embed, ids,
             src, sampled, positions, hidden, enorm, hnorm, hidden_size, eps);
}

// Row r reads residual row s = rows ? rows[r] : r: v = resid[s] (+ delta[s], bf16-rounded and
// stored back to resid[s] if write_resid); out[r] = bf16(rmsnorm(v) * w); optionally the FP8
// 1x128 quantization of out[r] into xq / xs (token-major scales).
__global__ void add_rmsnorm_kernel(bf16* __restrict__ resid, const bf16* __restrict__ delta,
                                   const int* __restrict__ rows, const bf16* __restrict__ w,
                                   bf16* __restrict__ out, fp8* __restrict__ xq,
                                   float* __restrict__ xs, int H, float eps, int write_resid) {
  pdl_entry();
  __shared__ float sh[32];
  const int r = blockIdx.x, c0 = threadIdx.x * kRowVec;
  const int s = rows ? rows[r] : r;
  float v[kRowVec], wv[kRowVec];
  load16(resid + (size_t)s * H + c0, v);
  if (delta) {
    float d[kRowVec];
    load16(delta + (size_t)s * H + c0, d);
#pragma unroll
    for (int i = 0; i < kRowVec; ++i) v[i] = bf16r(v[i] + d[i]);
    if (write_resid) store16(resid + (size_t)s * H + c0, v);
  }
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < kRowVec; ++i) ss += v[i] * v[i];
  ss = block_sum(ss, sh);
  const float rr = rsqrtf(ss / H + eps);
  load16(w + c0, wv);
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < kRowVec; ++i) {
    v[i] = bf16r(v[i] * rr * wv[i]);
    amax = fmaxf(amax, fabsf(v[i]));
  }
  store16(out + (size_t)r * H + c0, v);
  if (xq) {
    // 8 threads (aligned lanes) per 128-element block
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    const float scale = fmaxf(amax, 1e-12f) * (1.f / 448.f);
    const float ys = __frcp_rn(scale);
    __nv_fp8x4_e4m3* q = reinterpret_cast<__nv_fp8x4_e4m3*>(xq + (size_t)r * H + c0);
#pragma unroll
    for (int i = 0; i < kRowVec / 4; ++i)
      q[i] = __nv_fp8x4_e4m3(make_float4(div_rn(v[4 * i], scale, ys), div_rn(v[4 * i + 1], scale, ys),
                                         div_rn(v[4 * i + 2], scale, ys), div_rn(v[4 * i + 3], scale, ys)));
    if ((threadIdx.x & 7) == 0) xs[(size_t)r * (H / 128) + c0 / 128] = scale;
  }
}

void add_rmsnorm(bf16* resid, const bf16* delta, const int* rows, const bf16* w, bf16* out,
                 fp8* xq, float* xs, int n, int hidden, float eps, bool write_resid,
                 cudaStream_t st) {
  if (n == 0) return;
  CHECK(hidden % (kRowVec * 32) == 0 && hidden / kRowVec <= 1024,
        "add_rmsnorm: unsupported hidden size");
  launch_pdl(add_rmsnorm_kernel, dim3(n), dim3(hidden / kRowVec), 0, st, resid, delta, rows, w,
             out, xq, xs, hidden, eps, write_resid ? 1 : 0);
}

// ------------------------------------------------------------------ FP8 quantization
// One warp per 128-element block: lane l owns elements [4l, 4l+4).
__device__ __forceinline__ void quant_block_store(const float v[4], fp8* q, float* s_out) {
  float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
  amax = warp_max(amax);
  const float scale = fmaxf(amax, 1e-12f) * (1.f / 448.f);  // matches torch (x * (1/448))
  const float ys = __frcp_rn(scale);
  __nv_fp8x4_e4m3 packed(make_float4(div_rn(v[0], scale, ys), div_rn(v[1], scale, ys),
                                     div_rn(v[2], scale, ys), div_rn(v[3], scale, ys)));
  const int lane = threadIdx.x & 31;
  reinterpret_cast<__nv_fp8x4_e4m3*>(q)[lane] = packed;
  if (lane == 0) *s_out = scale;
}

// MLA down-projection norms in one launch: block (t, 0) = rmsnorm of q (row t, columns
// [0, Dq)) into q_out plus its FP8 1x128 quantization (qq / qs, row-major scales); block (t, 1) =
// rmsnorm of the latent (columns [Dq, Dq + Dkv)) into kv_out. 256 threads per row: thread i sums the
// squares of columns i + 256 k in increasing k, then block_sum; the quantization recomputes the
// normalized q values with the same operations instead of reading them back.
// kv_cache (with cm): the latent rows go to the token's row of the paged latent cache instead of
// kv_out (decode: the fused cache update then only does the indexer's part)
__global__ void __launch_bounds__(256) mla_qkv_norm_kernel(
    bf16* __restrict__ q_out, int ldq, fp8* __restrict__ qq, float* __restrict__ qs,
    bf16* __restrict__ kv_out, int ldkv, const bf16* __restrict__ x, int ldx,
    const bf16* __restrict__ wq, const bf16* __restrict__ wkv, int Dq, int Dkv, float eps,
    bf16* __restrict__ kv_cache, const MlaTokMeta cm) {
  pdl_trigger();
  __shared__ float sh[32];
  const int t = blockIdx.x, part = blockIdx.y;
  const int D = part ? Dkv : Dq;
  const bf16* w = part ? wkv : wq;
  // this thread's norm weights (constant) before the PDL wait (D < 2048: at most 8 per thread)
  constexpr int kPer = 8;
  float wv[kPer];
#pragma unroll
  for (int k = 0; k < kPer; ++k) {
    const int i = threadIdx.x + k * 256;
    wv[k] = i < D ? __bfloat162float(w[i]) : 0.f;
  }
  const bool to_cache = part && kv_cache;
  auto cache_row = [&]() {
    const int pos = cm.positions[t];
    const int64_t page =
        cm.block_tables[(size_t)cm.token_seq[t] * cm.max_blocks + pos / kMlaBlock];
    return page * cm.lat_page_rows + pos % kMlaBlock;
  };
  int64_t crow = 0;
  if (to_cache && cm.pre) crow = cache_row();  // (uploaded before the step)
  pdl_wait();
  if (to_cache && !cm.pre) crow = cache_row();
  const bf16* xr = x + (size_t)t * ldx + (part ? Dq : 0);
  bf16* orow = to_cache ? kv_cache + crow * Dkv
                        : (part ? kv_out + (size_t)t * ldkv : q_out + (size_t)t * ldq);
  // the thread's elements of the row (columns tid + 256 k) requested at once and kept in registers (one L2 latency
  // instead of one per element; the outputs reuse them), their squares summed in increasing k
  float xv[kPer];
#pragma unroll
  for (int k = 0; k < kPer; ++k) {
    const int i = threadIdx.x + k * 256;
    xv[k] = i < D ? __bfloat162float(xr[i]) : 0.f;
  }
  float ss = 0.f;
#pragma unroll
  for (int k = 0; k < kPer; ++k)
    if ((int)threadIdx.x + k * 256 < D) ss += xv[k] * xv[k];
  ss = block_sum(ss, sh);
  const float r = rsqrtf(ss / D + eps);
#pragma unroll
  for (int k = 0; k < kPer; ++k) {
    const int i = threadIdx.x + k * 256;
    if (i < D) orow[i] = __float2bfloat16(xv[k] * r * wv[k]);
  }
  if (part == 0 && qq) {
    // one warp per 128-column block, lane: 4 consecutive columns (quant_block_store's mapping)
    const int lane = threadIdx.x & 31, kb = D / 128;
    for (int b = threadIdx.x >> 5; b < kb; b += blockDim.x >> 5) {
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = b * 128 + lane * 4 + e;
        v[e] = bf16r(__bfloat162float(xr[i]) * r * __bfloat162float(w[i]));
      }
      quant_block_store(v, qq + (size_t)t * D + b * 128, qs + (size_t)t * kb + b);
    }
  }
}

void mla_qkv_norm(bf16* q_out, int ldq, fp8* qq, float* qs, bf16* kv_out, int ldkv, const bf16* x,
                  int ldx, const bf16* wq, const bf16* wkv, int T, int Dq, int Dkv, float eps,
                  cudaStream_t st, bf16* kv_cache, const MlaTokMeta* cache_meta) {
  if (T == 0) return;
  CHECK(Dq % 128 == 0 && Dq >= 512 && Dq < 2048 && Dkv >= 512 && Dkv < 2048,
        "mla_qkv_norm: shapes");
  CHECK(!kv_cache || (cache_meta && Dkv == kLatent), "mla_qkv_norm: cache rows need the metadata");
  const MlaTokMeta cm = kv_cache ? *cache_meta : MlaTokMeta{nullptr, nullptr, nullptr, 0};
  CHECK(!kv_cache || cm.lat_page_rows >= kMlaBlock, "mla_qkv_norm: latent page stride");
  launch_pdl(mla_qkv_norm_kernel, dim3(T, 2), dim3(256), 0, st, q_out, ldq, qq, qs, kv_out, ldkv,
             x, ldx, wq, wkv, Dq, Dkv, eps, kv_cache, cm);
}


__device__ __forceinline__ float swiglu1(float g, float u, float limit) {
  g = fminf(g, limit);
  u = fminf(fmaxf(u, -limit), limit);
  return bf16r(g / (1.f + __expf(-g)) * u);
}

__global__ void swiglu_bf16_kernel(bf16* __restrict__ out, const bf16* __restrict__ gu, int M,
                                   int I, float limit, const int* __restrict__ m_dev, bool gu64) {
  pdl_entry();
  if (m_dev) M = min(M, *m_dev);
  const size_t n = (size_t)M * I;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    const size_t m = i / I, j = i % I;
    const bf16* r = gu + m * 2 * I;
    const size_t gc = gu64 ? (j / 64) * 128 + j % 64 : j, uc = gu64 ? gc + 64 : I + j;
    out[i] = __float2bfloat16(swiglu1(__bfloat162float(r[gc]), __bfloat162float(r[uc]), limit));
  }
}

void swiglu_bf16(bf16* out, const bf16* gu, int M, int I, float limit, cudaStream_t st,
                 const int* m_dev, bool gu64) {
  if (M == 0) return;
  const size_t n = (size_t)M * I;
  launch_pdl(swiglu_bf16_kernel, dim3(std::min<size_t>(cdiv(n, 256), 4096)), dim3(256), 0, st,
             out, gu, M, I, limit, m_dev, gu64);
}

// ------------------------------------------------------------------ mHC
constexpr int kMaxHc = 4;


// pre / post / comb from the 24 mixes of one token, computed by the 16 lanes of a half warp
// (lane l -> comb element (l/4, l%4)). Every lane of the warp must call it (lanes 16..31 mirror).
// MUFU reciprocal: what __fdividef(1, x) computes for normal x. The .ftz form is the bare MUFU
// (rcp.approx.f32 wraps it in a denormal / huge-input rescaling of ~5 dependent instructions);
// identical for the Sinkhorn sums, which are >= hc_eps with normal reciprocals.
__device__ __forceinline__ float rcp_approx(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// The comb logits (Sinkhorn's input) of one token from its summed mixes
__device__ __forceinline__ void mhc_comb_logits(const float* mx, float rs, const float* base,
                                                const float* scale, float (&c)[kMaxHc][kMaxHc]) {
  constexpr int hc = kMaxHc;
#pragma unroll
  for (int i = 0; i < hc; ++i)
#pragma unroll
    for (int j = 0; j < hc; ++j) c[i][j] = mx[2 * hc + i * hc + j] * rs * scale[2] + base[2 * hc + i * hc + j];
}

// Sinkhorn of the comb logits in place: row softmax, then iters alternating column / row
// normalizations. Every lane holds the whole 4 x 4 matrix (no shuffles in the 20-iteration chain);
// the sums are grouped as the 16-lane butterfly grouped them ((a + b) + (c + d)) and every division
// is the fast division's c * rcp(x) with one reciprocal per row / column, so the results are
// bitwise those of the lane-per-element form with 4x fewer MUFU operations than per-element
// divisions (this warp runs beside the boundary's other warps).
// (kCompact: the iteration loop kept rolled -- the mHC boundaries' spare warps: a ~75-instruction body run from the
// instruction cache instead of the unrolled ~225 fetched cold; the same operations in the same order)
template <bool kCompact = false>
__device__ __forceinline__ void mhc_sinkhorn(float (&c)[kMaxHc][kMaxHc], float hc_eps, int iters) {
  constexpr int hc = kMaxHc;
#pragma unroll
  for (int i = 0; i < hc; ++i) {
    const float m = fmaxf(fmaxf(c[i][0], c[i][1]), fmaxf(c[i][2], c[i][3]));
#pragma unroll
    for (int j = 0; j < hc; ++j) c[i][j] = expf(c[i][j] - m);
    // fast division: ~1e-7 relative error, far below the bf16 resolution of the streams
    const float inv = rcp_approx((c[i][0] + c[i][1]) + (c[i][2] + c[i][3]));
#pragma unroll
    for (int j = 0; j < hc; ++j) c[i][j] = c[i][j] * inv + hc_eps;
  }
#pragma unroll
  for (int j = 0; j < hc; ++j) {
    const float inv = rcp_approx(((c[0][j] + c[1][j]) + (c[2][j] + c[3][j])) + hc_eps);
#pragma unroll
    for (int i = 0; i < hc; ++i) c[i][j] *= inv;
  }
  if constexpr (kCompact) {
#pragma unroll 1
    for (int it = 0; it < iters - 1; ++it) {
#pragma unroll
      for (int i = 0; i < hc; ++i) {
        const float inv = rcp_approx(((c[i][0] + c[i][1]) + (c[i][2] + c[i][3])) + hc_eps);
#pragma unroll
        for (int j = 0; j < hc; ++j) c[i][j] *= inv;
      }
#pragma unroll
      for (int j = 0; j < hc; ++j) {
        const float inv = rcp_approx(((c[0][j] + c[1][j]) + (c[2][j] + c[3][j])) + hc_eps);
#pragma unroll
        for (int i = 0; i < hc; ++i) c[i][j] *= inv;
      }
    }
    return;
  }
  for (int it = 0; it < iters - 1; ++it) {
#pragma unroll
    for (int i = 0; i < hc; ++i) {
      const float inv = rcp_approx(((c[i][0] + c[i][1]) + (c[i][2] + c[i][3])) + hc_eps);
#pragma unroll
      for (int j = 0; j < hc; ++j) c[i][j] *= inv;
    }
#pragma unroll
    for (int j = 0; j < hc; ++j) {
      const float inv = rcp_approx(((c[0][j] + c[1][j]) + (c[2][j] + c[3][j])) + hc_eps);
#pragma unroll
      for (int i = 0; i < hc; ++i) c[i][j] *= inv;
    }
  }
}

// lane l < 16 of the warp: element l of the 4 x 4 matrix every lane holds
__device__ __forceinline__ float mhc_mat_elem(const float (&c)[kMaxHc][kMaxHc], int l) {
  float v = 0.f;
#pragma unroll
  for (int q = 0; q < kMaxHc * kMaxHc; ++q)
    if (q == l) v = c[q / kMaxHc][q % kMaxHc];
  return v;
}

__device__ __forceinline__ void mhc_mix_warp(const float* mx, float rs, const float* base,
                                             const float* scale, float hc_eps, int iters,
                                             float post_mult, float* pre_out, float* post_out,
                                             float* comb_out) {
  constexpr int hc = kMaxHc;
  float c[hc][hc];
  mhc_comb_logits(mx, rs, base, scale, c);
  mhc_sinkhorn(c, hc_eps, iters);
  const int l = threadIdx.x & 31;
  if (l < hc * hc) {
    comb_out[l] = mhc_mat_elem(c, l);
    if (l < hc) {
      const float pl = mx[l] * rs * scale[0] + base[l];
      pre_out[l] = 1.f / (1.f + expf(-pl)) + hc_eps;
      const float ql = mx[hc + l] * rs * scale[1] + base[hc + l];
      post_out[l] = 1.f / (1.f + expf(-ql)) * post_mult;
    }
  }
}

// The deferred form of mhc_mix_warp's outputs (ArFuse.sk & 1): post as mhc_mix_warp's, and the comb
// LOGITS in comb_out (the next boundary's spare warp runs their Sinkhorn: mhc_sinkhorn)
// (lane l < 16 computes its own logit, element (l / 4, l % 4) of mhc_comb_logits by the same expression: no 16-way
// register select of the whole matrix)
__device__ __forceinline__ void mhc_logits_warp(const float* mx, float rs, const float* base,
                                                const float* scale, float post_mult, float* post_out,
                                                float* comb_out) {
  constexpr int hc = kMaxHc;
  const int l = threadIdx.x & 31;
  if (l < hc * hc) {
    comb_out[l] = mx[2 * hc + l] * rs * scale[2] + base[2 * hc + l];
    if (l < hc) {
      const float ql = mx[hc + l] * rs * scale[1] + base[hc + l];
      post_out[l] = 1.f / (1.f + expf(-ql)) * post_mult;
    }
  }
}

// The consumer side of the deferral (ArFuse.sk & 2): Sinkhorn of the comb logits a boundary left
// in comb for a token (every lane its own copy of the matrix: the chain costs one token's)
template <bool kCompact = false>
__device__ __forceinline__ void mhc_deferred_comb(const float* logits, float hc_eps, int iters,
                                                  float (&c)[kMaxHc][kMaxHc]) {
#pragma unroll
  for (int i = 0; i < kMaxHc; ++i)
#pragma unroll
    for (int j = 0; j < kMaxHc; ++j) c[i][j] = __ldcg(logits + i * kMaxHc + j);
  mhc_sinkhorn<kCompact>(c, hc_eps, iters);
}


// Store 8 outputs of a normalized row (bf16) and optionally their FP8 1x128 quantization. The
// 16 threads covering one 128-element block must be 16 consecutive lanes of a warp, and every
// lane of the warp must call this.
__device__ __forceinline__ void store_out_q(bf16* x_out, fp8* xq, float* xs, size_t row, int d8,
                                            const float* o, int hidden) {
  float r[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) r[j] = bf16r(o[j]);
  store8(x_out + row * hidden + d8 * 8, r);
  if (xq == nullptr) return;
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(r[j]));
#pragma unroll
  for (int o2 = 1; o2 < 16; o2 <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o2));
  const float scale = fmaxf(amax, 1e-12f) * (1.f / 448.f);
  const float ys = __frcp_rn(scale);
  const __nv_fp8x4_e4m3 q0(make_float4(div_rn(r[0], scale, ys), div_rn(r[1], scale, ys),
                                       div_rn(r[2], scale, ys), div_rn(r[3], scale, ys)));
  const __nv_fp8x4_e4m3 q1(make_float4(div_rn(r[4], scale, ys), div_rn(r[5], scale, ys),
                                       div_rn(r[6], scale, ys), div_rn(r[7], scale, ys)));
  *reinterpret_cast<uint2*>(xq + row * hidden + d8 * 8) = make_uint2(q0.__x, q1.__x);
  if ((d8 & 15) == 0) xs[row * (hidden / 128) + d8 / 16] = scale;
}

__global__ void mhc_pre_apply_kernel(float* __restrict__ post_out, float* __restrict__ comb_out,
                                     bf16* __restrict__ x_out, fp8* __restrict__ xq,
                                     float* __restrict__ xs, const bf16* __restrict__ streams,
                                     const float* __restrict__ mixes,
                                     const float* __restrict__ base,
                                     const float* __restrict__ scale,
                                     const bf16* __restrict__ norm_w, int hidden, float rms_eps,
                                     float hc_eps, int iters, float post_mult, float norm_eps) {
  pdl_entry();
  constexpr int hc = kMaxHc;
  __shared__ float sh[32];
  __shared__ float pre_s[hc];
  extern __shared__ float xbuf[];  // [hidden]
  const int t = blockIdx.x;
  const bf16* s = streams + (size_t)t * hc * hidden;
  const int n8 = hc * hidden / 8;
  // 1) mean square over the flattened residual
  float ss = 0.f;
  for (int i = threadIdx.x; i < n8; i += blockDim.x) {
    float f[8];
    load8(s + i * 8, f);
#pragma unroll
    for (int j = 0; j < 8; ++j) ss += f[j] * f[j];
  }
  ss = block_sum(ss, sh);
  // 2) mixes -> pre / post / comb (one warp)
  if (threadIdx.x < 32) {
    const float rs = rsqrtf(ss / (hc * hidden) + rms_eps);
    mhc_mix_warp(mixes + (size_t)t * (2 + hc) * hc, rs, base, scale, hc_eps, iters, post_mult,
                 pre_s, post_out + (size_t)t * hc, comb_out + (size_t)t * hc * hc);
  }
  __syncthreads();
  // 3) layer input = bf16(sum_i pre_i * stream_i), then RMSNorm
  float p[hc];
#pragma unroll
  for (int i = 0; i < hc; ++i) p[i] = pre_s[i];
  float ss2 = 0.f;
  for (int d8 = threadIdx.x; d8 < hidden / 8; d8 += blockDim.x) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < hc; ++i) {
      float f[8];
      load8(s + (size_t)i * hidden + d8 * 8, f);
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] += p[i] * f[j];
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float v = bf16r(acc[j]);
      xbuf[d8 * 8 + j] = v;
      ss2 += v * v;
    }
  }
  ss2 = block_sum(ss2, sh);
  const float r = rsqrtf(ss2 / hidden + norm_eps);
  for (int d8 = threadIdx.x; d8 < hidden / 8; d8 += blockDim.x) {
    float wv[8], o[8];
    load8(norm_w + d8 * 8, wv);
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = xbuf[d8 * 8 + j] * r * wv[j];
    store_out_q(x_out, xq, xs, t, d8, o, hidden);
  }
}

// mhc_pre_apply_kernel's arithmetic (bitwise: every thread sums the same vectors in the same
// order and the block sums are block_sum's) in one read of the streams, without Sinkhorn on the
// streaming path: each thread keeps its 4 x U vectors of 8 in registers between the RMS of the
// streams and the pre-mapping and computes pre itself (mhc_mix_warp's sigmoids); rs is saved for
// mhc_sinkhorn_kernel (comb / post, first read by the next boundary). U = hidden / 2048.
template <int U>
__global__ void __launch_bounds__(256, 4) mhc_pre_apply_reg_kernel(
    float* __restrict__ rs_out, bf16* __restrict__ x_out, fp8* __restrict__ xq,
    float* __restrict__ xs, const bf16* __restrict__ streams, const float* __restrict__ mixes,
    const float* __restrict__ base, const float* __restrict__ scale,
    const bf16* __restrict__ norm_w, int hidden, float rms_eps, float hc_eps, float norm_eps) {
  pdl_entry();
  constexpr int hc = kMaxHc;
  __shared__ float sh[32];
  const int t = blockIdx.x, tid = threadIdx.x;
  const bf16* s = streams + (size_t)t * hc * hidden;
  // 1) mean square over the flattened residual: thread tid's vectors tid + 256 k, k = i U + u
  //    (stream i, element vector tid + 256 u), in k order as mhc_pre_apply_kernel's loop
  uint4 v[hc][U];
#pragma unroll
  for (int i = 0; i < hc; ++i)
#pragma unroll
    for (int u = 0; u < U; ++u)
      v[i][u] = *reinterpret_cast<const uint4*>(s + (size_t)i * hidden + (tid + 256 * u) * 8);
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < hc; ++i)
#pragma unroll
    for (int u = 0; u < U; ++u) {
      float f[8];
      load8(reinterpret_cast<const bf16*>(&v[i][u]), f);
#pragma unroll
      for (int j = 0; j < 8; ++j) ss += f[j] * f[j];
    }
  ss = block_sum(ss, sh);
  const float rs = rsqrtf(ss / (hc * hidden) + rms_eps);
  if (tid == 0) rs_out[t] = rs;
  // 2) pre (mhc_mix_warp's sigmoids of the first hc mixes)
  const float* mx = mixes + (size_t)t * (2 + hc) * hc;
  float p[hc];
#pragma unroll
  for (int i = 0; i < hc; ++i) {
    const float pl = mx[i] * rs * scale[0] + base[i];
    p[i] = 1.f / (1.f + expf(-pl)) + hc_eps;
  }
  // 3) layer input = bf16(sum_i pre_i * stream_i), then RMSNorm
  uint4 xb[U];  // x as bf16 (its values are bf16-rounded)
  float ss2 = 0.f;
#pragma unroll
  for (int u = 0; u < U; ++u) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < hc; ++i) {
      float f[8];
      load8(reinterpret_cast<const bf16*>(&v[i][u]), f);
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] += p[i] * f[j];
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float xr = bf16r(acc[j]);
      ss2 += xr * xr;
    }
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&xb[u]);
#pragma unroll
    for (int j = 0; j < 4; ++j) h[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
  }
  ss2 = block_sum(ss2, sh);
  const float r = rsqrtf(ss2 / hidden + norm_eps);
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const int d8 = tid + 256 * u;
    float wv[8], xv[8], o[8];
    load8(norm_w + d8 * 8, wv);
    load8(reinterpret_cast<const bf16*>(&xb[u]), xv);
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = xv[j] * r * wv[j];
    store_out_q(x_out, xq, xs, t, d8, o, hidden);
  }
}

// comb / post of mhc_pre_apply from the mixes and the saved rs: one thread per token, with
// mhc_mix_warp's arithmetic (whose lanes each compute the whole matrix redundantly)
// comb / post of one token (mhc_sinkhorn_kernel, sp_pre_push_kernel)
__device__ __forceinline__ void mhc_sinkhorn_row(float* __restrict__ post_out,
                                                 float* __restrict__ comb_out,
                                                 const float* __restrict__ mixes,
                                                 const float* __restrict__ rs_in,
                                                 const float* __restrict__ base,
                                                 const float* __restrict__ scale, int t,
                                                 float hc_eps, int iters, float post_mult) {
  constexpr int hc = kMaxHc;
  const float* mx = mixes + (size_t)t * (2 + hc) * hc;
  const float rs = rs_in[t];
  float c[hc][hc];
#pragma unroll
  for (int i = 0; i < hc; ++i)
#pragma unroll
    for (int j = 0; j < hc; ++j) c[i][j] = mx[2 * hc + i * hc + j] * rs * scale[2] + base[2 * hc + i * hc + j];
#pragma unroll
  for (int i = 0; i < hc; ++i) {
    const float m = fmaxf(fmaxf(c[i][0], c[i][1]), fmaxf(c[i][2], c[i][3]));
#pragma unroll
    for (int j = 0; j < hc; ++j) c[i][j] = expf(c[i][j] - m);
    const float inv = rcp_approx((c[i][0] + c[i][1]) + (c[i][2] + c[i][3]));
#pragma unroll
    for (int j = 0; j < hc; ++j) c[i][j] = c[i][j] * inv + hc_eps;
  }
#pragma unroll
  for (int j = 0; j < hc; ++j) {
    const float inv = rcp_approx(((c[0][j] + c[1][j]) + (c[2][j] + c[3][j])) + hc_eps);
#pragma unroll
    for (int i = 0; i < hc; ++i) c[i][j] *= inv;
  }
  for (int it = 0; it < iters - 1; ++it) {
#pragma unroll
    for (int i = 0; i < hc; ++i) {
      const float inv = rcp_approx(((c[i][0] + c[i][1]) + (c[i][2] + c[i][3])) + hc_eps);
#pragma unroll
      for (int j = 0; j < hc; ++j) c[i][j] *= inv;
    }
#pragma unroll
    for (int j = 0; j < hc; ++j) {
      const float inv = rcp_approx(((c[0][j] + c[1][j]) + (c[2][j] + c[3][j])) + hc_eps);
#pragma unroll
      for (int i = 0; i < hc; ++i) c[i][j] *= inv;
    }
  }
  float4* co = reinterpret_cast<float4*>(comb_out + (size_t)t * hc * hc);
#pragma unroll
  for (int i = 0; i < hc; ++i) co[i] = make_float4(c[i][0], c[i][1], c[i][2], c[i][3]);
  float po[hc];
#pragma unroll
  for (int l = 0; l < hc; ++l) {
    const float ql = mx[hc + l] * rs * scale[1] + base[hc + l];
    po[l] = 1.f / (1.f + expf(-ql)) * post_mult;
  }
  *reinterpret_cast<float4*>(post_out + (size_t)t * hc) = make_float4(po[0], po[1], po[2], po[3]);
}

__global__ void __launch_bounds__(128) mhc_sinkhorn_kernel(
    float* __restrict__ post_out, float* __restrict__ comb_out, const float* __restrict__ mixes,
    const float* __restrict__ rs_in, const float* __restrict__ base,
    const float* __restrict__ scale, int T, float hc_eps, int iters, float post_mult) {
  pdl_entry();
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  mhc_sinkhorn_row(post_out, comb_out, mixes, rs_in, base, scale, t, hc_eps, iters, post_mult);
}

void mhc_pre_apply(float* post, float* comb, bf16* x, fp8* xq, float* xs, const bf16* streams,
                   const float* mixes, const float* base, const float* scale, const bf16* norm_w,
                   int T, int hidden, int hc, float rms_eps, float hc_eps, int iters,
                   float post_mult, float norm_eps, cudaStream_t st, float* rs_scratch) {
  CHECK(hc == kMaxHc, "mhc kernels are specialized for hc_mult=4");
  CHECK(hidden % 2048 == 0, "mhc_pre_apply: hidden must be a multiple of 2048");
  if (T == 0) return;
  // the register form (one read of the streams, but two launches) from 3072 tokens
  constexpr int kRegMinTokens = 3072;
  if (rs_scratch && hidden == 4096 && T >= kRegMinTokens) {
    launch_pdl(mhc_pre_apply_reg_kernel<2>, dim3(T), dim3(256), 0, st, rs_scratch, x, xq, xs,
               streams, mixes, base, scale, norm_w, hidden, rms_eps, hc_eps, norm_eps);
    launch_pdl(mhc_sinkhorn_kernel, dim3(cdiv(T, 128)), dim3(128), 0, st, post, comb, mixes,
               rs_scratch, base, scale, T, hc_eps, iters, post_mult);
    return;
  }
  launch_pdl(mhc_pre_apply_kernel, dim3(T), dim3(256), hidden * sizeof(float), st,
      post, comb, x, xq, xs, streams, mixes, base, scale, norm_w, hidden, rms_eps, hc_eps, iters,
      post_mult, norm_eps);
}


// ------------------------------------------------------------------ sequence-parallel boundary
// store_out_q's outputs into nd destinations (this rank's buffers and every peer's)
__device__ __forceinline__ void store_out_q_n(bf16* const* x_out, fp8* const* xq,
                                              float* const* xs, int nd, size_t row, int d8,
                                              const float* o, int hidden) {
  float r[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) r[j] = bf16r(o[j]);
  uint4 u;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(r[2 * i], r[2 * i + 1]);
#pragma unroll
  for (int k = 0; k < kSpMaxTp; ++k)
    if (k < nd) *reinterpret_cast<uint4*>(x_out[k] + row * hidden + d8 * 8) = u;
  if (xq[0] == nullptr) return;
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(r[j]));
#pragma unroll
  for (int o2 = 1; o2 < 16; o2 <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o2));
  const float scale = fmaxf(amax, 1e-12f) * (1.f / 448.f);
  const float ys = __frcp_rn(scale);
  const __nv_fp8x4_e4m3 q0(make_float4(div_rn(r[0], scale, ys), div_rn(r[1], scale, ys),
                                       div_rn(r[2], scale, ys), div_rn(r[3], scale, ys)));
  const __nv_fp8x4_e4m3 q1(make_float4(div_rn(r[4], scale, ys), div_rn(r[5], scale, ys),
                                       div_rn(r[6], scale, ys), div_rn(r[7], scale, ys)));
#pragma unroll
  for (int k = 0; k < kSpMaxTp; ++k)
    if (k < nd)
      *reinterpret_cast<uint2*>(xq[k] + row * hidden + d8 * 8) = make_uint2(q0.__x, q1.__x);
  if ((d8 & 15) == 0) {
#pragma unroll
    for (int k = 0; k < kSpMaxTp; ++k)
      if (k < nd) xs[k][row * (hidden / 128) + d8 / 16] = scale;
  }
}

// block_sum's reduction for each of NR groups of 256 threads (lanes and warps as a 256-thread block)
template <int NR>
__device__ __forceinline__ float group256_sum(float v, float* sh) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, g = threadIdx.x >> 8;
  v = warp_sum(v);
  __syncthreads();
  if (lane == 0) sh[wid] = v;
  __syncthreads();
  v = lane < 8 ? sh[g * 8 + lane] : 0.f;
  return warp_sum(v);
}

// mhc_pre_apply_reg_kernel's arithmetic for this rank's rows (a fixed grid striding over groups of
// NR rows, one per 256 threads), every finished row stored into this rank's x and every peer's (the
// sequence-parallel all-gather as the rows are done); then mhc_sinkhorn_row of the block's rows;
// each block counts in at every peer and the grid's last block waits for every peer's blocks (no
// co-residency needed).
constexpr int kSpPushBlocks = 148;
template <int U, int NR>
__global__ void __launch_bounds__(256 * NR) sp_pre_push_kernel(
    const SpPeers p, int tp, int rank, int q0, int nrows, float* __restrict__ rs_out,
    float* __restrict__ post_out, float* __restrict__ comb_out, bf16* x, fp8* xq, float* xs,
    const bf16* __restrict__ streams, const float* __restrict__ mixes,
    const float* __restrict__ base, const float* __restrict__ scale,
    const bf16* __restrict__ norm_w, int hidden, float rms_eps, float hc_eps, int iters,
    float post_mult, float norm_eps, uint32_t* state) {
  pdl_entry();
  constexpr int hc = kMaxHc;
  __shared__ float sh[32];
  __shared__ bool last;
  const int tid = threadIdx.x, lt = tid & 255, h = tid >> 8;
  const uint32_t epoch = state[4] + 1;
  bf16* xd[kSpMaxTp];
  fp8* qd[kSpMaxTp];
  float* sd[kSpMaxTp];
#pragma unroll
  for (int k = 0; k < kSpMaxTp; ++k) {
    const int r = (rank + k) % tp;
    xd[k] = k == 0 ? x : p.x[r];
    qd[k] = xq == nullptr ? nullptr : (k == 0 ? xq : p.xq[r]);
    sd[k] = xq == nullptr ? nullptr : (k == 0 ? xs : p.xs[r]);
  }
  for (int tg = blockIdx.x; tg * NR < nrows; tg += gridDim.x) {
    const int t = tg * NR + h;
    const bool ok = t < nrows;
    const bf16* s = streams + (size_t)(ok ? t : 0) * hc * hidden;
    uint4 v[hc][U];
#pragma unroll
    for (int i = 0; i < hc; ++i)
#pragma unroll
      for (int u = 0; u < U; ++u)
        v[i][u] = *reinterpret_cast<const uint4*>(s + (size_t)i * hidden + (lt + 256 * u) * 8);
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < hc; ++i)
#pragma unroll
      for (int u = 0; u < U; ++u) {
        float f[8];
        load8(reinterpret_cast<const bf16*>(&v[i][u]), f);
#pragma unroll
        for (int j = 0; j < 8; ++j) ss += f[j] * f[j];
      }
    ss = group256_sum<NR>(ss, sh);
    const float rs = rsqrtf(ss / (hc * hidden) + rms_eps);
    if (lt == 0 && ok) rs_out[t] = rs;
    const float* mx = mixes + (size_t)(ok ? t : 0) * (2 + hc) * hc;
    float pr[hc];
#pragma unroll
    for (int i = 0; i < hc; ++i) {
      const float pl = mx[i] * rs * scale[0] + base[i];
      pr[i] = 1.f / (1.f + expf(-pl)) + hc_eps;
    }
    uint4 xb[U];
    float ss2 = 0.f;
#pragma unroll
    for (int u = 0; u < U; ++u) {
      float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int i = 0; i < hc; ++i) {
        float f[8];
        load8(reinterpret_cast<const bf16*>(&v[i][u]), f);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] += pr[i] * f[j];
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float xr = bf16r(acc[j]);
        ss2 += xr * xr;
      }
      __nv_bfloat162* hb = reinterpret_cast<__nv_bfloat162*>(&xb[u]);
#pragma unroll
      for (int j = 0; j < 4; ++j) hb[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
    }
    ss2 = group256_sum<NR>(ss2, sh);
    const float r = rsqrtf(ss2 / hidden + norm_eps);
    if (ok) {
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int d8 = lt + 256 * u;
        float wv[8], xv[8], o[8];
        load8(norm_w + d8 * 8, wv);
        load8(reinterpret_cast<const bf16*>(&xb[u]), xv);
#pragma unroll
        for (int j = 0; j < 8; ++j) o[j] = xv[j] * r * wv[j];
        store_out_q_n(xd, qd, sd, tp, (size_t)q0 + t, d8, o, hidden);
      }
    }
  }
  __syncthreads();
  {  // thread j: the block's (j / NR)-th row group, its (j % NR)-th row
    const int t = (blockIdx.x + (tid / NR) * gridDim.x) * NR + tid % NR;
    if (t < nrows)
      mhc_sinkhorn_row(post_out, comb_out, mixes, rs_out, base, scale, t, hc_eps, iters,
                       post_mult);
  }
  __syncthreads();
  if (tid < tp && tid != rank) {
    __threadfence_system();
    sp_red_release_sys_add(p.flags[tid] + 3 * kSpMaxTp + rank, 1u);
  }
  if (tid == 0) {
    __threadfence();
    last = atomicAdd(&state[5], 1u) == gridDim.x - 1;
  }
  __syncthreads();
  if (!last) return;
  if (tid < tp && tid != rank) {
    const uint32_t want = gridDim.x * epoch;
    while ((int)(sp_ld_acquire_sys(p.flags[rank] + 3 * kSpMaxTp + tid) - want) < 0) {
    }
  }
  __syncthreads();
  if (tid == 0) {
    state[4] = epoch;
    state[5] = 0;
    __threadfence();
  }
}

void sp_pre_push(const SpPeers& peers, int tp, int rank, int q0, int nrows, int H, float* post,
                 float* comb, bf16* x, fp8* xq, float* xs, const bf16* sq, const float* mixes,
                 const float* base, const float* scale, const bf16* norm_w, float rms_eps,
                 float hc_eps, int iters, float post_mult, float norm_eps, float* rs,
                 uint32_t* state, cudaStream_t st) {
  CHECK(H == 4096 && tp <= kSpMaxTp, "sp_pre_push: hidden 4096");
  CHECK(xq == nullptr || (xs != nullptr && peers.xq[(rank + 1) % tp] != nullptr &&
                          peers.xs[(rank + 1) % tp] != nullptr),
        "sp_pre_push: the peers' FP8 buffers");
  // two rows per 512-thread block, on a fixed grid (the arrival counts are blocks per call)
  launch_pdl(sp_pre_push_kernel<2, 2>, dim3(kSpPushBlocks), dim3(512), 0, st, peers, tp, rank, q0,
             std::max(nrows, 0), rs, post, comb, x, xq, xs, sq, mixes, base, scale, norm_w, H,
             rms_eps, hc_eps, iters, post_mult, norm_eps, state);
}

// ------------------------------------------------------------------ fused mHC boundary
// Two variants. Split (few tokens): G independent CTAs per token each reduce partial mixes over
// a slice; the last CTA of a token (global counter) finishes Sinkhorn, the weighted sum and the
// norm. Cluster (many tokens): see below.
constexpr int kFusedThreads = 512;  // 4 groups x (4 streams x 32 lanes)
constexpr int kMixes = (2 + kMaxHc) * kMaxHc;  // 24
constexpr int kMixPerGroup = kMixes / 4;       // 6
constexpr int kMaxVec = 4;

// CTAs per token; each CTA covers hidden/G elements of every stream (G >= hidden / 1024).
int mhc_fused_groups(int T) { return T <= 16 ? 16 : (T <= 64 ? 8 : 4); }

__global__ void __launch_bounds__(kFusedThreads) mhc_fused_split_kernel(
    bf16* __restrict__ streams, const bf16* __restrict__ x_prev, float* __restrict__ post,
    float* __restrict__ comb, bf16* __restrict__ x_out, fp8* __restrict__ xq,
    float* __restrict__ xs, const bf16* __restrict__ fn,
    const float* __restrict__ base, const float* __restrict__ scale,
    const bf16* __restrict__ norm_w, float* __restrict__ partial, int* __restrict__ counter,
    int hidden, float rms_eps, float hc_eps, int iters, float post_mult, float norm_eps) {
  pdl_trigger();
  constexpr int hc = kMaxHc;
  const int g = blockIdx.x, G = gridDim.x, t = blockIdx.y, tid = threadIdx.x;
  {
    // prefetch this CTA's fn slice (24 x 4 rows of hidden / G elements) into L2
    const int lines_per_row = hidden / G / 64;  // 128-byte lines
    for (int i = tid; i < kMixes * hc * lines_per_row; i += kFusedThreads) {
      const int row = i / lines_per_row, l = i % lines_per_row;
      prefetch_l2(fn + (size_t)row * hidden + g * (hidden / G) + l * 64);
    }
  }
  pdl_wait();
  const int lane = tid & 31, wid = tid >> 5;
  const int grp = tid >> 7;                // mixes [6*grp, 6*grp + 6) (+ sum of squares if 0)
  const int j = (tid >> 5) & 3;            // stream
  const int slice = hidden / G;            // hidden elements per stream handled by this CTA
  bf16* s = streams + (size_t)t * hc * hidden;
  __shared__ float red[kFusedThreads / 32][kMixPerGroup + 1];
  __shared__ bool last;
  float acc[kMixPerGroup + 1];
#pragma unroll
  for (int m = 0; m <= kMixPerGroup; ++m) acc[m] = 0.f;
  const int nv = slice / 8 / 32;  // vectors per thread (1, 2 or 4)
  float r[kMaxVec][8];
  if (x_prev) {
    float c[hc], p;
#pragma unroll
    for (int i = 0; i < hc; ++i) c[i] = comb[(size_t)t * hc * hc + i * hc + j];
    p = post[(size_t)t * hc + j];
#pragma unroll
    for (int q = 0; q < kMaxVec; ++q) {
      if (q >= nv) break;
      const int d = g * slice + (lane + 32 * q) * 8;
      float xv[8];
      load8(x_prev + (size_t)t * hidden + d, xv);
#pragma unroll
      for (int e = 0; e < 8; ++e) r[q][e] = p * xv[e];
#pragma unroll
      for (int i = 0; i < hc; ++i) {
        float rin[8];
        load8(s + (size_t)i * hidden + d, rin);
#pragma unroll
        for (int e = 0; e < 8; ++e) r[q][e] += c[i] * rin[e];
      }
    }
    __syncthreads();  // all old streams of this slice are read before any is overwritten
#pragma unroll
    for (int q = 0; q < kMaxVec; ++q) {
      if (q >= nv) break;
      const int d = g * slice + (lane + 32 * q) * 8;
      if (grp == 0) store8(s + (size_t)j * hidden + d, r[q]);
#pragma unroll
      for (int e = 0; e < 8; ++e) r[q][e] = bf16r(r[q][e]);
    }
  } else {
#pragma unroll
    for (int q = 0; q < kMaxVec; ++q) {
      if (q >= nv) break;
      load8(s + (size_t)j * hidden + g * slice + (lane + 32 * q) * 8, r[q]);
    }
  }
#pragma unroll
  for (int q = 0; q < kMaxVec; ++q) {
    if (q >= nv) break;
    const int d = g * slice + (lane + 32 * q) * 8;
    const bf16* f = fn + (size_t)(grp * kMixPerGroup) * hc * hidden + (size_t)j * hidden + d;
    if (grp == 0) {
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[kMixPerGroup] += r[q][e] * r[q][e];
    }
#pragma unroll
    for (int m = 0; m < kMixPerGroup; ++m) {
      float fv[8];
      load8(f + (size_t)m * hc * hidden, fv);
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[m] += r[q][e] * fv[e];
    }
  }
  // CTA reduction: per group over its 4 warps
#pragma unroll
  for (int m = 0; m <= kMixPerGroup; ++m) {
    const float v = warp_sum(acc[m]);
    if (lane == 0) red[wid][m] = v;
  }
  __syncthreads();
  if (tid <= kMixes) {
    // mix m lives in group m / 6 (sum of squares: group 0, slot 6)
    const int gq = tid == kMixes ? 0 : tid / kMixPerGroup;
    const int slot = tid == kMixes ? kMixPerGroup : tid % kMixPerGroup;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < 4; ++w) v += red[gq * 4 + w][slot];
    partial[((size_t)t * G + g) * 32 + tid] = v;
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) last = atomicAdd(&counter[t], 1) == G - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  // ---- phase B (one CTA per token): finish mixes, Sinkhorn, weighted sum + RMSNorm
  __shared__ float mx[kMixes + 1];
  __shared__ float pre_s[hc];
  __shared__ float sh[32];
  if (tid <= kMixes) {
    float v = 0.f;
    for (int gg = 0; gg < G; ++gg) v += __ldcg(&partial[((size_t)t * G + gg) * 32 + tid]);
    mx[tid] = v;
  }
  if (tid == 0) counter[t] = 0;
  __syncthreads();
  if (tid < 32) {
    const float rs = rsqrtf(mx[kMixes] / (hc * hidden) + rms_eps);
    mhc_mix_warp(mx, rs, base, scale, hc_eps, iters, post_mult, pre_s, post + (size_t)t * hc,
                 comb + (size_t)t * hc * hc);
  }
  __syncthreads();
  float pr[hc];
#pragma unroll
  for (int i = 0; i < hc; ++i) pr[i] = pre_s[i];
  extern __shared__ float xbuf[];
  float ss2 = 0.f;
  for (int d8 = tid; d8 < hidden / 8; d8 += kFusedThreads) {
    float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < hc; ++i) {
      const uint4 u = __ldcg(reinterpret_cast<const uint4*>(s + (size_t)i * hidden + d8 * 8));
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 ff = __bfloat1622float2(h[q]);
        a[2 * q] += pr[i] * ff.x;
        a[2 * q + 1] += pr[i] * ff.y;
      }
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float v = bf16r(a[e]);
      xbuf[d8 * 8 + e] = v;
      ss2 += v * v;
    }
  }
  ss2 = block_sum(ss2, sh);
  const float rn = rsqrtf(ss2 / hidden + norm_eps);
  for (int d8 = tid; d8 < hidden / 8; d8 += kFusedThreads) {
    float wv[8], o[8];
    load8(norm_w + d8 * 8, wv);
#pragma unroll
    for (int e = 0; e < 8; ++e) o[e] = xbuf[d8 * 8 + e] * rn * wv[e];
    store_out_q(x_out, xq, xs, t, d8, o, hidden);
  }
}

// One 8-CTA thread-block cluster per token; CTA r of the cluster owns hidden elements
// [r * hidden / 8, (r + 1) * hidden / 8) of every stream. Thread (warp w, lane l) owns vector
// v = w * 8 + l / 4 (8 elements) of stream j = l % 4, so the 4 streams at a position live in 4
// consecutive lanes of one warp: the in-place post-mapping needs only a warp barrier between
// reading the old streams and writing the new one. The 24 mixes and the RMS sum of squares are
// reduced across the cluster through distributed shared memory; every CTA then runs Sinkhorn
// redundantly and normalizes its own slice.
constexpr int kHcCluster = 8;
constexpr int kHcThreads = 256;                    // x-path threads (8 warps)
constexpr int kHcThreadsAll = kHcThreads + 32;     // + the Sinkhorn warp
constexpr int kHcSlice = 512;                      // hidden elements per CTA
constexpr int kHcFnPitch = kHcSlice * 2 + 32;      // bytes per (mix, stream) fn row in shared
                                                   // memory (conflict-free 16-byte loads)
constexpr int kHcSmem = kMixes * kMaxHc * kHcFnPitch;

__device__ __forceinline__ void hc_cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(
                   static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
               "l"(src));
}
__device__ __forceinline__ void hc_cluster_arrive() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void hc_cluster_wait() {
  asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
// Fast form: the cluster sums' DSMEM pushes are st.async completing on the receiving CTA's mbarrier,
// instead of cluster barriers, whose release (MEMBAR.ALL.GPU) waits on every global store still in
// flight (the in-place streams, the all-reduce's pushes and clearing).
__device__ __forceinline__ uint32_t hc_smem(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ uint32_t hc_mapa(uint32_t a, int r) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(r));
  return d;
}
__device__ __forceinline__ void hc_st_async(uint32_t a, float v, uint32_t mb) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];\n" ::"r"(a),
               "r"(__float_as_uint(v)), "r"(mb)
               : "memory");
}
__device__ __forceinline__ void hc_mb_expect(uint32_t mb, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(mb), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void hc_mb_wait(uint32_t mb) {  // (its first phase)
  uint32_t done;
  do {
    asm volatile(
        "{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0;\nselp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(done)
        : "r"(mb)
        : "memory");
  } while (!done);
}
// fn slice staged by one 5-D TMA tensor copy (kTma instances; host map mhc_fn_map): box {64 elements, 2 (seg & 1),
// 4 streams, slice / 128 (seg >> 1), 24 mixes} with the 128-byte swizzle, so the 16-byte chunk c of 128-byte line L
// lands at chunk c ^ (L & 7) and L & 7 = 2 stream + (seg & 1): the x path's (4 streams x 2 vectors) per 8-lane phase
// hit 8 distinct bank groups (conflict-free, as the padded rows of the cp.async form). One instruction instead of
// 6144 16-byte cp.async: the copies do not occupy the SM's load path ahead of the push / poll loads.
constexpr int kHcFnTmaBytes = kMixes * kMaxHc * kHcSlice * 2;  // 96 KB (the 8-CTA form)
__device__ __forceinline__ void hc_tma_fn(uint32_t dst, const CUtensorMap* tm, int seg_hi0, uint32_t mb) {
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, "
      "%6}], [%7];\n" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(tm)), "r"(0), "r"(0), "r"(0), "r"(seg_hi0), "r"(0), "r"(mb)
      : "memory");
}
// (the swizzled fn row of mix 0 for x-path warp xw, stream j, vector lane >> 2 of the warp's 8; mix m at + m * 4096)
__device__ __forceinline__ int hc_fn_tma_off(int xw, int j, int lane) {
  return ((((xw >> 1) * 4 + j) * 2 + (xw & 1)) << 7) + (((lane >> 2) ^ ((2 * j + (xw & 1)) & 7)) << 4);
}
// (kMoe) ld.global.cg kept after the code before it (the push stores): the post-mapping's inputs
__device__ __forceinline__ uint4 hc_ldcg16_late(const void* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ float hc_ldcg_late(const float* p) {
  float v;
  asm volatile("ld.global.cg.f32 %0, [%1];\n" : "=f"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void hc_bar_x() {  // the 8 x-path warps only
  asm volatile("bar.sync 1, %0;\n" ::"n"(kHcThreads) : "memory");
}
// deferred Sinkhorn (ArFuse.sk & 2): the spare warp hands the comb it computed to the x-path warps
// (named barrier 2: the spare warp arrives, the x-path warps wait)
__device__ __forceinline__ void hc_bar_sk_arrive() {
  asm volatile("bar.arrive 2, %0;\n" ::"n"(kHcThreadsAll) : "memory");
}
__device__ __forceinline__ void hc_bar_sk_sync() {
  asm volatile("bar.sync 2, %0;\n" ::"n"(kHcThreadsAll) : "memory");
}

// The layer input x depends only on pre (4 sigmoids of the mixes); Sinkhorn's comb / post are
// first used by the next boundary. So warp 8 runs Sinkhorn (and publishes comb / post) while
// warps 0-7 finish x. The CTA's fn slice is staged in shared memory and its norm weights in
// registers before the PDL wait (weights never depend on earlier kernels). Both cross-CTA sums
// push their partials into every CTA's shared memory (summed in rank order, as the pull form).
// kFast (few tokens, kMhcFastMaxTokens): the fused all-reduce's exchange overlapped --
// the MoE combine's loads issued at once, the streams loaded before the exchange, every source
// polled at once, the previous call's clearing between the push and the poll, a fence-free
// last-CTA finish -- at 168 instead of 96 registers (one CTA per SM, as few tokens need).
// kMinB 2: the fast form held to two CTAs per SM (one wave of 33 clusters instead of 15)
// The code-warming warp (threads kTcWarm on; the cluster kernel's one-CTA fast form and the tensor-core
// kernel): the boundary's phases would run cold from the instruction cache. This warp starts at launch, skips the
// PDL wait, the all-reduce exchange, every barrier, mbarrier wait, load of earlier kernels' data and store, and runs
// the x path's code once ahead of the other warps (as thread xt of x-path warp 1), so they find it in the SM's
// instruction cache; its share of the cluster barrier arrives at its start; the CTA's barriers are a named barrier
// of the other threads. Separate instances (kWarm): the others compile to the kernels without it.
constexpr int kTcWarm = kHcThreadsAll;
constexpr int kTcThreads = kHcThreadsAll + 32;
__device__ __forceinline__ void tc_bar_cta() {  // the x-path warps + the Sinkhorn warp
  asm volatile("bar.sync 3, %0;\n" ::"n"(kHcThreadsAll) : "memory");
}
template <bool kWarm>
__device__ __forceinline__ void hc_sync_cta(bool dry) {  // (kWarm false: the instances without it)
  if constexpr (kWarm) {
    if (!dry) tc_bar_cta();
  } else {
    __syncthreads();
  }
}

// (kHm) sm_100's mixed-precision FMA / add (FHFMA.BF16 / FHADD.BF16 with a half selector): a bf16 x bf16 product (exact
// in f32) plus an f32, rounded once = fmaf(float(a), float(b), c) bitwise; float(a) + c likewise -- without the bf16 ->
// f32 conversion instructions (lo: the low halves of the bf16 pairs, element 2k; hi: element 2k + 1)
__device__ __forceinline__ float hm_fma_lo(uint32_t a2, uint32_t b2, float c) {
  float d;
  asm("{\n.reg .b16 a, b, x, y;\nmov.b32 {a, x}, %1;\nmov.b32 {b, y}, %2;\nfma.rn.f32.bf16 %0, a, b, %3;\n}\n"
      : "=f"(d) : "r"(a2), "r"(b2), "f"(c));
  return d;
}
__device__ __forceinline__ float hm_fma_hi(uint32_t a2, uint32_t b2, float c) {
  float d;
  asm("{\n.reg .b16 a, b, x, y;\nmov.b32 {x, a}, %1;\nmov.b32 {y, b}, %2;\nfma.rn.f32.bf16 %0, a, b, %3;\n}\n"
      : "=f"(d) : "r"(a2), "r"(b2), "f"(c));
  return d;
}
__device__ __forceinline__ float hm_add_lo(uint32_t a2, float c) {
  float d;
  asm("{\n.reg .b16 a, x;\nmov.b32 {a, x}, %1;\nadd.rn.f32.bf16 %0, a, %2;\n}\n" : "=f"(d) : "r"(a2), "f"(c));
  return d;
}
__device__ __forceinline__ float hm_add_hi(uint32_t a2, float c) {
  float d;
  asm("{\n.reg .b16 a, x;\nmov.b32 {x, a}, %1;\nadd.rn.f32.bf16 %0, a, %2;\n}\n" : "=f"(d) : "r"(a2), "f"(c));
  return d;
}
// (kHm) has_sentinel of any word of a polled chunk: a bf16 lane of u is the sentinel 0x8000 iff that lane of
// u ^ 0x80008000 is zero, found by the zero-lane test (x - 0x00010001) & ~x & 0x80008000 (exact as a boolean: a borrow
// only follows a zero low lane)
__device__ __forceinline__ bool hm_sent4(const uint4& u) {
  const uint32_t a = u.x ^ kSentinel2, b = u.y ^ kSentinel2, c = u.z ^ kSentinel2, d = u.w ^ kSentinel2;
  const uint32_t m = ((a - 0x00010001u) & ~a) | ((b - 0x00010001u) & ~b) | ((c - 0x00010001u) & ~c) |
                     ((d - 0x00010001u) & ~d);
  return (m & kSentinel2) != 0u;
}
// (kHm) packed f32x2 (FFMA2 / FMUL2; a scalar operand as a pair of itself is a broadcast operand): per lane the scalar
// fma.rn / mul.rn, bitwise
__device__ __forceinline__ uint64_t hm_pk(float a, float b) {
  uint64_t r;
  asm("mov.b64 %0, {%1, %2};" : "=l"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ void hm_up(uint64_t v, float& a, float& b) {
  asm("mov.b64 {%0, %1}, %2;" : "=f"(a), "=f"(b) : "l"(v));
}
__device__ __forceinline__ uint64_t hm_fma2(uint64_t a, uint64_t b, uint64_t c) {
  uint64_t d;
  asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c));
  return d;
}
__device__ __forceinline__ uint64_t hm_mul2(uint64_t a, uint64_t b) {
  uint64_t d;
  asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b));
  return d;
}
// (kHm) the post-mapping of one position's 8 elements of stream j: r = fma(p, xv, c0 s0), then fma(c_i, s_i, r) (the
// contraction the base instances use), the 4 streams' raw words in rin, by element pairs
__device__ __forceinline__ void hm_postmap(float p, const float* xv, const float* c, const uint4* rin, float* r) {
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    uint64_t sp[kMaxHc];
#pragma unroll
    for (int i = 0; i < kMaxHc; ++i) {
      const uint32_t w = k == 0 ? rin[i].x : k == 1 ? rin[i].y : k == 2 ? rin[i].z : rin[i].w;
      sp[i] = hm_pk(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
    }
    uint64_t t = hm_mul2(hm_pk(c[0], c[0]), sp[0]);
    t = hm_fma2(hm_pk(p, p), hm_pk(xv[2 * k], xv[2 * k + 1]), t);
#pragma unroll
    for (int i = 1; i < kMaxHc; ++i) t = hm_fma2(hm_pk(c[i], c[i]), sp[i], t);
    hm_up(t, r[2 * k], r[2 * k + 1]);
  }
}
// (kHm) a[e] *= m (8 elements, by pairs)
__device__ __forceinline__ void hm_scale8(float* a, float m) {
#pragma unroll
  for (int k = 0; k < 4; ++k) hm_up(hm_mul2(hm_pk(a[2 * k], a[2 * k + 1]), hm_pk(m, m)), a[2 * k], a[2 * k + 1]);
}

// (kHm) eight bf16-exact floats as 4 bf16 pairs (exact), element 2k in the low half
__device__ __forceinline__ void hm_pack8(const float* r, uint32_t* rp) {
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(r[2 * k], r[2 * k + 1]);
    rp[k] = *reinterpret_cast<const uint32_t*>(&h);
  }
}
// (kHm) two floats rounded to bf16 as one pair (element a in the low half): bitwise bf16r of each (the same RN
// conversion)
__device__ __forceinline__ uint32_t hm_pack2(float a, float b) {
  const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&h);
}
// (kHm) the sum in rank order of the 4 sources' 16-byte chunks (the fast forms' poll), as the f32 form's adds
__device__ __forceinline__ void hm_sum4(const uint4* w, float* acc) {
#pragma unroll
  for (int q = 0; q < 8; ++q) acc[q] = 0.f;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint32_t ws[4] = {w[q].x, w[q].y, w[q].z, w[q].w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      acc[2 * k] = hm_add_lo(ws[k], acc[2 * k]);
      acc[2 * k + 1] = hm_add_hi(ws[k], acc[2 * k + 1]);
    }
  }
}
// (kHm) the 24 mixes and the sum of squares of one thread's 8 elements: a = sum_e r[e] fn[m][e] from 0 in element
// order (the f32 form's FFMA chain), fn rows as 16-byte shared memory loads at row0 + m * pitch
__device__ __forceinline__ void hm_mixes(const uint32_t* rp, const uint8_t* row0, int pitch, float* acc) {
  float a = 0.f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    a = hm_fma_lo(rp[k], rp[k], a);
    a = hm_fma_hi(rp[k], rp[k], a);
  }
  acc[kMixes] = a;
#pragma unroll
  for (int m = 0; m < kMixes; ++m) {
    const uint4 f = *reinterpret_cast<const uint4*>(row0 + m * pitch);
    const uint32_t fw[4] = {f.x, f.y, f.z, f.w};
    float b = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      b = hm_fma_lo(rp[k], fw[k], b);
      b = hm_fma_hi(rp[k], fw[k], b);
    }
    acc[m] = b;
  }
}

// (kFr) the same chains with the 24 fn rows in registers (fw[m]: row m's 8 bf16 of this thread), advanced together one
// element pair at a time (each chain keeps its element order: bitwise hm_mixes)
__device__ __forceinline__ void hm_mixes_r(const uint32_t* rp, const uint4* fw, float* acc) {
#pragma unroll
  for (int m = 0; m <= kMixes; ++m) acc[m] = 0.f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
#pragma unroll
    for (int m = 0; m < kMixes; ++m) {
      const uint32_t f = k == 0 ? fw[m].x : k == 1 ? fw[m].y : k == 2 ? fw[m].z : fw[m].w;
      acc[m] = hm_fma_lo(rp[k], f, acc[m]);
      acc[m] = hm_fma_hi(rp[k], f, acc[m]);
    }
    acc[kMixes] = hm_fma_lo(rp[k], rp[k], acc[kMixes]);
    acc[kMixes] = hm_fma_hi(rp[k], rp[k], acc[kMixes]);
  }
}

// v[0..7] -> eight e2m1 nibbles, v[0] in the lowest (as nvfp4.cu's)
__device__ __forceinline__ uint32_t hc_e2m1x8(const float* v) {
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

// kSk (deferred Sinkhorn, ArFuse.sk; separate instances so the other forms carry none of its code):
// bit 1 writes post and the comb logits instead of running Sinkhorn; bit 2 takes comb as the
// previous boundary's logits, whose Sinkhorn the spare warp runs (every CTA of the token's cluster:
// each needs the whole matrix) while the all-reduce is in flight
// kTma (fast form): the fn slice by one TMA tensor copy (tm_fn, hc_tma_fn) instead of cp.async; the mixes wait on
// its mbarrier, so the post-poll barrier does not wait for the copies either.
// kMoe > 0 (fast form; the host checks x_prev, a fused all-reduce of tp 4, the self-pushed MoE combine (ar.cm.y,
// !ar.pushed) of kMoe pairs per token (ar.cm.K1)): the boundary after a MoE with those conditions compiled in, the
// post-mapping's inputs (post, the streams) loaded after the push, and its combine split over the 4 lanes of a
// position: lane j loads word j (elements 2j, 2j + 1) of every local expert row, sums them in pair order (the vector
// form's products and sums, bitwise) and pushes that word (the Lamport sentinel is per bf16 lane, so 4-byte pushes
// complete a chunk as 16-byte ones do): a quarter of the combine's instructions between the PDL wait and the push.
// kHm (fast form; no Gram form): the post-poll chain with the mixed-precision instructions (hm_mixes, hm_sum4; hidden
// 4096) and fewer instructions on the rest of it (hm_sent4, shorter warp sum / max, spread pushes): bitwise.
// kFr (kMoe + kHm, fn by cp.async): every x-path thread copies its own 24 fn chunks at entry and, after the push (the
// peers' sums in flight), waits for them (its own: no barrier) and loads them into registers: the mixes after the
// post-poll barrier read no shared memory (96 KB per CTA: 768 cycles of the SM's shared-memory bandwidth) and no cp.async
// wait precedes the barrier (loaded after the poll instead, the 8 warps' loads would follow the arrival).
// kMp (kMoe + kWarm: the MoE side's code-warming instances): the buffer rotation's clearing by the code-warming warp
// after its dry pass (its own PDL wait and call-state reads; named barrier 4 hands the state reads to warp 0 before
// this CTA counts itself in) instead of the spare warp after its Sinkhorn, where it would hold the post-poll barrier
// past the x path
// kTm (kMoe + kFr, mhc_fused's moe_tm: MoE tensor parallelism, every pair on this rank; moe_fp4's y_tm): the MoE's
// outputs token-major, pair s of token t at row t kMoe + s, so the rows' words are requested with the weights right
// after the PDL wait instead of after the pair positions' round trip (no pair positions: every pair is summed, in pair
// order)
// kEt (the kMp instance before a KDA layer, set per call by mhc_set_et): the dependents launched at the start (1: the
// KDA in_proj's CTAs take the SMs the MoE kernel leaves, also during its last units) or right after the x path's PDL
// wait (2: once the MoE kernel has completed) or after the x path's push (3: the boundary's own loads and pushes issued
// first), so that they fill their ring during the push / poll (gemv_tma's PD form paces it)
// kXc (the kMp instances): the buffer rotation's clearing by the x path right after its push (its state reads are its
// own, after its PDL wait), while the peers' sums are in flight, instead of the code-warming warp after its dry pass:
// the warming warp then neither waits for the grid nor clears, and warp 0 does not wait for it at named barrier 4
// before this CTA counts itself in (the clearing stores precede the count-in in program order and the post-poll CTA
// barrier). The same stores into the same region: bitwise.
template <bool kFast, int kMinB = 1, int kSk = 0, bool kWarm = false, bool kTma = false, int kMoe = 0, bool kHm = false,
          bool kFr = false, bool kMp = false, bool kTm = false, int kEt = 0, bool kXc = false>
__global__ void __cluster_dims__(kHcCluster, 1, 1) __launch_bounds__(kWarm ? kTcThreads : kHcThreadsAll, kMinB)
    mhc_fused_cluster_kernel(bf16* __restrict__ streams, const bf16* __restrict__ x_prev,
                     float* __restrict__ post, float* __restrict__ comb, bf16* __restrict__ x_out,
                     fp8* __restrict__ xq, float* __restrict__ xs, const bf16* __restrict__ fn,
                     const float* __restrict__ base, const float* __restrict__ scale,
                     const bf16* __restrict__ norm_w, int hidden_arg, float rms_eps, float hc_eps,
                     int iters, float post_mult, float norm_eps, const ArFuse ar, int gram,
                     uint8_t* __restrict__ x4, uint8_t* __restrict__ x4s, float q4_sfs,
                     const __grid_constant__ CUtensorMap tm_fn) {
  if (kEt == 1 || (!kMoe && !(kFast && x_prev && ar.state && ar.cm.y && !ar.pushed))) pdl_trigger();
  // (the next kernel may launch at once -- except after a MoE (the fast form's self-pushed MoE combine):
  // then only after the post-poll barrier below. Launched at the start, the next layer's KDA in_proj /
  // MLA qkv_a CTAs take the SMs the MoE kernel leaves and stream their weights in while this grid's
  // push and poll run)
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  constexpr int hc = kMaxHc;
  // (the host runs the cluster forms at hidden 4096 only (mhc_fused_cluster), the MoE combine's rows (ar.cm.hidden)
  // too: the address arithmetic with the constant)
  constexpr int hidden = kHcCluster * kHcSlice;
  (void)hidden_arg;
  constexpr bool kSkIn = (kSk & 2) != 0, kSkOut = (kSk & 1) != 0;
  static_assert(!kFr || (kMoe > 0 && kHm && !kTma), "kFr: the kMoe + kHm instances");
  static_assert(!kMp || (kMoe > 0 && kWarm), "kMp: the kMoe kWarm instances");
  static_assert(!kTm || (kMoe > 0 && kFr), "kTm: the kMoe kFr instances");
  static_assert(!kXc || kMp, "kXc: the kMp instances");
  extern __shared__ __align__(16) uint8_t hc_fn[];  // [mix * hc + stream][kHcFnPitch bytes]
  const int t = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  // the warming warp (dry): as thread xt of x-path warp 1
  const bool dry = kWarm && tid >= kTcWarm;
  const int xt = dry ? tid - kTcWarm + 32 : tid, xw = xt >> 5;
  const int rank = (int)cluster.block_rank();
  const int j = lane & 3;                                       // stream
  const int v = xw * 8 + (lane >> 2);                           // 8-element vector of the slice
  const int d = rank * kHcSlice + v * 8;                        // first element (x-path threads)
  // partial mixes (+ sum of squares, + with gram the streams' 4 x 4 Gram matrix) per rank
  constexpr int kRed = kMixes + 1 + kMaxHc * kMaxHc;
  __shared__ float part[kHcCluster][kRed];
  __shared__ float part2[kHcCluster];             // partial sums of squares of x per rank
  __shared__ __align__(8) uint64_t hc_mb[2];      // (fast form) the two sums' mbarriers
  __shared__ float red[kHcThreads / 32][kRed];
  __shared__ float mx[kRed];
  __shared__ float wamax[kHcThreads / 32];
  __shared__ float pre_scr[hc], post_scr[hc], comb_scr[hc * hc];
  __shared__ float s_base[kMixes], s_scale[3];
  __shared__ __align__(16) float sk_comb[kSkIn ? hc * hc : 4];  // (kSk & 2) comb by the spare warp
  __shared__ __align__(8) uint64_t hc_mb_fn;      // (kTma) the fn copy's mbarrier
  // (kTma: the copy's destination at the next 1024-byte boundary, the 128-byte swizzle's period)
  uint8_t* const fnb = kTma ? hc_fn + ((1024u - (hc_smem(hc_fn) & 1023u)) & 1023u) : hc_fn;
  // ---- weights: fn slice -> shared memory, norm weights -> registers, the mixes' biases and
  //      scales -> shared memory (read after the cluster sum by Sinkhorn and the pre-mapping)
  if constexpr (kTma) {
    if (tid == 0) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(hc_smem(&hc_mb_fn)) : "memory");
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      hc_mb_expect(hc_smem(&hc_mb_fn), kHcFnTmaBytes);
      hc_tma_fn(hc_smem(fnb), &tm_fn, rank * (kHcSlice / 128), hc_smem(&hc_mb_fn));
    }
  } else if (!dry) {
    if constexpr (kFr) {  // (this thread's own chunks: rows m * hc + j, chunk v)
      if (warp < kHcThreads / 32) {
#pragma unroll 4
        for (int m = 0; m < kMixes; ++m)
          hc_cp_async16(hc_fn + (m * hc + j) * kHcFnPitch + v * 16, fn + (size_t)(m * hc + j) * hidden + d);
      }
    } else {
      for (int i = tid; i < kMixes * hc * (kHcSlice / 8); i += kHcThreadsAll) {
        const int row = i / (kHcSlice / 8), c = i % (kHcSlice / 8);
        hc_cp_async16(hc_fn + row * kHcFnPitch + c * 16,
                      fn + (size_t)row * hidden + rank * kHcSlice + c * 8);
      }
    }
    asm volatile("cp.async.commit_group;\n" ::);
  }
  // (kTm) the biases / scales and the norm weights into registers, stored / converted after the post-mapping: requested
  // behind the CTA's 96 KB of fn copies and consumed here, they would hold the x-path warps before the PDL wait
  float bsv = 0.f, scv = 0.f;
  uint4 wraw = make_uint4(0u, 0u, 0u, 0u);
  float wv[8];
  if constexpr (kTm) {
    if (tid < kMixes) bsv = base[tid];
    if (tid < 3) scv = scale[tid];
    if (warp < kHcThreads / 32 || dry) wraw = *reinterpret_cast<const uint4*>(norm_w + d);
  } else {
    if (tid < kMixes) s_base[tid] = base[tid];
    if (tid < 3) s_scale[tid] = scale[tid];
    if (warp < kHcThreads / 32 || dry) load8(norm_w + d, wv);
  }
  if constexpr (kFast) {
    // the mbarriers, visible to the cluster before any push (this arrive's wait is just before the
    // first push; the barrier completes while this grid waits for the previous kernel)
    if (tid == 0) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(hc_smem(&hc_mb[0])) : "memory");
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(hc_smem(&hc_mb[1])) : "memory");
      // (the only arrival of each: the bytes every CTA of the cluster pushes, expected before any push lands)
      hc_mb_expect(hc_smem(&hc_mb[0]), kHcCluster * (gram ? kRed : kMixes + 1) * 4);
      hc_mb_expect(hc_smem(&hc_mb[1]), kHcCluster * 4);  // (unused by the Gram form)
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
  }
  // all-reduce pushed by the producer GEMV (ar.pushed >= 2): the GEMV triggers this grid only after
  // its own PDL wait, so every input but the sums (streams, comb / post, the call state: kernels
  // before the GEMV) is final when this grid starts (L2 loads), and the sums are polled: the wait
  // moves to the first store (2) or is dropped (3: no store of this grid conflicts with the GEMV)
  const int hoist = !kMoe && kFast && x_prev && ar.state && ar.pushed >= 2 ? ar.pushed : 0;
  // (MoE combine: the token's pair positions and weights, written by the MoE kernel's routing at its start,
  // requested into L2 while it ends: the push's first, dependent loads)
  if ((kMoe || (kFast && x_prev && ar.state && ar.cm.y && !ar.pushed)) && tid < (kTm ? 1 : 2)) {
    const size_t i0 = (size_t)t * ar.cm.K1;
    prefetch_l2(tid || kTm ? (const void*)(ar.cm.w + i0) : (const void*)(ar.cm.pair_pos + i0));
    prefetch_l2(tid || kTm ? (const void*)(ar.cm.w + i0 + ar.cm.K1 - 1)
                           : (const void*)(ar.cm.pair_pos + i0 + ar.cm.K1 - 1));
  }
  if (!hoist && !dry) pdl_wait();
  if (kEt == 2 && !dry) pdl_trigger();
  float r[8];
  uint4 fr[kFr ? kMixes : 1];  // (kFr) this thread's fn rows
  // fused all-reduce: this call's epoch; the previous call's size / protocol (buffer rotation;
  // the fast form loads them with the epoch, so its clearing does not wait on them)
  uint32_t ar_epoch = 0, ar_prev_n = 0, ar_prev_kind = 0;
  bool cl_waited = false;  // (fast form) this thread has waited on the cluster barrier (before its poll)
  if ((kMoe || (ar.state && x_prev)) && !dry) {
    ar_epoch = next_epoch(__ldcg(&ar.state[0]));
    if constexpr (kFast) {
      ar_prev_n = __ldcg(&ar.state[1]);
      ar_prev_kind = __ldcg(&ar.state[3]);
    }
  }
  if (warp < kHcThreads / 32 || dry) {
    bf16* s = streams + (size_t)t * hc * hidden;
    // ---- post-mapping of the previous sublayer output (in place), or the current streams
    if (kMoe || x_prev) {
      // (fast form) the push's inputs are loaded first -- the stream-0 lane's partial sums, or the
      // MoE combine's pair positions and weights -- so the push waits on nothing else
      uint4 u = make_uint4(0, 0, 0, 0);
      int pp[kArMaxK1];
      float ws[kArMaxK1];
      uint32_t ywt[kTm ? kMoe : 1];  // (kTm) the token's rows' words j, requested with the weights
      if constexpr (kMoe > 0) {
        if (!dry) {  // (every lane: lane j of a position combines its word j)
          const int* const ppt = ar.cm.pair_pos + t * kMoe;
          const float* const wt = ar.cm.w + t * kMoe;
          const char* const ybt = reinterpret_cast<const char*>(ar.cm.y + ((size_t)t * kMoe * hidden + d + 2 * j));
#pragma unroll
          for (int s2 = 0; s2 < kMoe; ++s2) {
            if constexpr (kTm)
              ywt[s2] = *reinterpret_cast<const uint32_t*>(ybt + s2 * (2 * hidden));
            else
              pp[s2] = ppt[s2];
            ws[s2] = wt[s2];
          }
        }
      } else if (kFast && ar.state && !ar.pushed && j == 0 && !dry) {
        if (ar.cm.y) {
#pragma unroll
          for (int s2 = 0; s2 < kArMaxK1; ++s2)
            if (s2 < ar.cm.K1) {
              pp[s2] = ar.cm.pair_pos[(size_t)t * ar.cm.K1 + s2];
              ws[s2] = ar.cm.w[(size_t)t * ar.cm.K1 + s2];
            }
        } else {
          u = *reinterpret_cast<const uint4*>(x_prev + (size_t)t * hidden + d);
        }
      }
      float c[hc];
      if constexpr (!kSkIn) {  // (kSk & 2: comb from the spare warp, below)
#pragma unroll
        for (int i = 0; i < hc; ++i) c[i] = dry ? 0.f : __ldcg(comb + (size_t)t * hc * hc + i * hc + j);
      }
      float p = 0.f;  // (kMoe: loaded after the push)
      if constexpr (kMoe == 0) p = dry ? 0.f : __ldcg(post + (size_t)t * hc + j);
      // (fast form) the streams do not depend on the all-reduce: their loads overlap its exchange
      // (kept raw until the sums have arrived: converting them would stall the push)
      uint4 rin[kFast ? hc : 1];
      if constexpr (kFast) {
#pragma unroll
        for (int i = 0; i < hc; ++i)
          if constexpr (kMoe == 0)
            rin[i] = dry ? make_uint4(0, 0, 0, 0)
                         : __ldcg(reinterpret_cast<const uint4*>(s + (size_t)i * hidden + d));
          else if (dry)  // (kMoe: loaded after the push)
            rin[i] = make_uint4(0, 0, 0, 0);
      }
      float xv[8];
      if constexpr (kWarm) {
#pragma unroll
        for (int q = 0; q < 8; ++q) xv[q] = 0.f;
      }
      if (dry) {
        // (the warming warp: straight to the post-mapping)
        if constexpr (kFr) {
#pragma unroll
          for (int m = 0; m < kMixes; ++m)
            fr[m] = *reinterpret_cast<const uint4*>(hc_fn + (m * hc + j) * kHcFnPitch + v * 16);
        }
      } else if (kMoe || ar.state) {
        // fused one-shot all-reduce of this position's 8 elements (stream-0 lane), summed in rank
        // order and rounded to bf16 as allreduce_lamport_kernel's store; then to the other lanes
        uint32_t xpk[4] = {0u, 0u, 0u, 0u};  // (kHm) the stream-0 lane's rounded sums as bf16 pairs
        if constexpr (!kFast) {
          if (j == 0) {
            const size_t e = (size_t)t * hidden + d;
            const size_t slot = (size_t)ar.tp * ar.max_elems;
            uint4 u;
            if (ar.cm.y) {
              float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
              for (int s2 = 0; s2 < ar.cm.K1; ++s2) {
                const int pp = ar.cm.pair_pos[(size_t)t * ar.cm.K1 + s2];
                if (pp < 0) continue;  // expert not on this rank
                const float ws = ar.cm.w[(size_t)t * ar.cm.K1 + s2];
                float f[8];
                load8(ar.cm.y + (size_t)pp * hidden + (d), f);
#pragma unroll
                for (int q = 0; q < 8; ++q) acc[q] += ws * f[q];
              }
              __nv_bfloat162* hp = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
              for (int q = 0; q < 4; ++q) hp[q] = __floats2bfloat162_rn(acc[2 * q], acc[2 * q + 1]);
            } else {
              u = *reinterpret_cast<const uint4*>(x_prev + e);
            }
            u.x = clear_neg_zero(u.x);
            u.y = clear_neg_zero(u.y);
            u.z = clear_neg_zero(u.z);
            u.w = clear_neg_zero(u.w);
            if (!ar.pushed)
              for (int q = 0; q < ar.tp; ++q)
                *reinterpret_cast<uint4*>(ar.peers.buf[q] + (ar_epoch % 3) * slot +
                                          (size_t)ar.rank * ar.max_elems + e) = u;
            const bf16* mine = ar.peers.buf[ar.rank] + (ar_epoch % 3) * slot;
            float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int q = 0; q < ar.tp; ++q) {
              uint4 w;
              do {
                w = ld_volatile16(mine + (size_t)q * ar.max_elems + e);
              } while (has_sentinel(w.x) || has_sentinel(w.y) || has_sentinel(w.z) ||
                       has_sentinel(w.w));
              const __nv_bfloat162* hq = reinterpret_cast<const __nv_bfloat162*>(&w);
#pragma unroll
              for (int q2 = 0; q2 < 4; ++q2) {
                const float2 f = __bfloat1622float2(hq[q2]);
                acc[2 * q2] += f.x;
                acc[2 * q2 + 1] += f.y;
              }
            }
#pragma unroll
            for (int q = 0; q < 8; ++q) xv[q] = bf16r(acc[q]);
          }
        } else {
          // fast form: the exchange overlapped (fused all-reduces are tp == 4)
          const size_t e = (size_t)t * hidden + d;
          const size_t slot = (size_t)ar.tp * ar.max_elems;
          if constexpr (kMoe > 0) {
            // MoE combine, word j of the position (elements 2j, 2j + 1): every local expert row's word as independent
            // loads, then summed in pair order (moe_combine's order; the vector form's products and sums)
            const char* const yb = reinterpret_cast<const char*>(ar.cm.y + d + 2 * j);
            constexpr int hb = 2 * hidden;  // (row bytes)
            uint32_t yw[kArMaxK1];
#pragma unroll
            for (int s2 = 0; s2 < kMoe; ++s2) {  // (pp < 0: an expert not on this rank -- row 0's word, not summed)
              if constexpr (kTm)
                yw[s2] = ywt[s2];
              else
                yw[s2] = *reinterpret_cast<const uint32_t*>(yb + max(pp[s2], 0) * hb);
            }
            float a0 = 0.f, a1 = 0.f;
#pragma unroll
            for (int s2 = 0; s2 < kMoe; ++s2)
              if (kTm || pp[s2] >= 0) {
                const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yw[s2]));
                a0 += ws[s2] * f.x;
                a1 += ws[s2] * f.y;
              }
            const __nv_bfloat162 h2 = __floats2bfloat162_rn(a0, a1);
            const uint32_t uw = clear_neg_zero(*reinterpret_cast<const uint32_t*>(&h2));
            const size_t o = (ar_epoch % 3) * slot + (size_t)ar.rank * ar.max_elems + e + 2 * j;
#pragma unroll
            for (int q = 0; q < 4; ++q) *reinterpret_cast<uint32_t*>(ar.peers.buf[q] + o) = uw;
            if constexpr (kEt == 3) pdl_trigger();  // (kEt 3: the dependents once this lane's word is pushed)
            if constexpr (kXc) {  // the previous call's region of another buffer, while the peers' sums fly
              const int nb = (int)gridDim.y * kHcCluster, lb = (int)blockIdx.y * kHcCluster + rank;
              lamport_clear_small<4>(ar.peers, ar.rank, (int)((ar_epoch + 2) % 3), (int)ar_prev_n, ar_prev_kind,
                                     ar.max_elems, lb, nb, tid, kHcThreads);
            }
            // the post-mapping's inputs (needed once the peers' sums have arrived)
            p = hc_ldcg_late(post + (size_t)t * hc + j);
#pragma unroll
            for (int i = 0; i < hc; ++i) rin[i] = hc_ldcg16_late(s + (size_t)i * hidden + d);
            if constexpr (kFr) {  // (own copies: no barrier) the fn rows into registers while the peers' sums fly
              asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#pragma unroll
              for (int m = 0; m < kMixes; ++m)
                fr[m] = *reinterpret_cast<const uint4*>(hc_fn + (m * hc + j) * kHcFnPitch + v * 16);
            }
          } else if (j == 0 && !ar.pushed) {
            if (ar.cm.y) {
              // MoE combine: every pair's position and weight (loaded above), then every local
              // expert row, as independent loads; summed in pair order (moe_combine's order)
              uint4 yr[kArMaxK1];
#pragma unroll
              for (int s2 = 0; s2 < kArMaxK1; ++s2)
                if (s2 < ar.cm.K1 && pp[s2] >= 0)  // (negative: expert not on this rank)
                  yr[s2] = *reinterpret_cast<const uint4*>(ar.cm.y +
                                                           (size_t)pp[s2] * hidden + d);
              float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
              for (int s2 = 0; s2 < kArMaxK1; ++s2)
                if (s2 < ar.cm.K1 && pp[s2] >= 0) {
                  const __nv_bfloat162* hy = reinterpret_cast<const __nv_bfloat162*>(&yr[s2]);
#pragma unroll
                  for (int q = 0; q < 4; ++q) {
                    const float2 f = __bfloat1622float2(hy[q]);
                    acc[2 * q] += ws[s2] * f.x;
                    acc[2 * q + 1] += ws[s2] * f.y;
                  }
                }
              __nv_bfloat162* hp = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
              for (int q = 0; q < 4; ++q) hp[q] = __floats2bfloat162_rn(acc[2 * q], acc[2 * q + 1]);
            }  // (else: u holds the partial sums, loaded above)
            u.x = clear_neg_zero(u.x);
            u.y = clear_neg_zero(u.y);
            u.z = clear_neg_zero(u.z);
            u.w = clear_neg_zero(u.w);
            for (int q = 0; q < ar.tp; ++q)
              *reinterpret_cast<uint4*>(ar.peers.buf[q] + (ar_epoch % 3) * slot +
                                        (size_t)ar.rank * ar.max_elems + e) = u;
          }
          // the buffer rotation's clearing of the previous call's region (another buffer; every
          // rank's data of that call has arrived) while the peers' data of this call is in flight
          if (kMoe == 0 && ar.tp == 4) {  // (kMoe: the spare warp clears, below)
            const int nb = (int)gridDim.y * kHcCluster, lb = (int)blockIdx.y * kHcCluster + rank;
            lamport_clear_small<4>(ar.peers, ar.rank, (int)((ar_epoch + 2) % 3), (int)ar_prev_n,
                                   ar_prev_kind, ar.max_elems, lb, nb, tid, kHcThreads);
          }
          // the cluster barrier (the mbarriers' initialization, every CTA arrived at its start) before the poll, which
          // waits for the peers anyway: after the post-poll barrier it would hold the first cluster sum's push
          hc_cluster_wait();
          cl_waited = true;
          if (j == 0) {
            // poll every source's chunk at once (one round trip once all have arrived), sum in
            // rank order
            const bf16* mine = ar.peers.buf[ar.rank] + (ar_epoch % 3) * slot;
            constexpr int kTp = 4;
            uint4 w[kTp];
            unsigned need = (1u << kTp) - 1u;
            while (need) {
#pragma unroll
              for (int q = 0; q < kTp; ++q)
                if (need >> q & 1u) w[q] = ld_volatile16(mine + (size_t)q * ar.max_elems + e);
#pragma unroll
              for (int q = 0; q < kTp; ++q) {
                if constexpr (kHm) {  // (branch-free: a complete source holds its chunk, re-testing it is a no-op)
                  need &= ~((hm_sent4(w[q]) ? 0u : 1u) << q);
                } else if ((need >> q & 1u) && !(has_sentinel(w[q].x) || has_sentinel(w[q].y) || has_sentinel(w[q].z) ||
                                                 has_sentinel(w[q].w))) {
                  need &= ~(1u << q);
                }
              }
            }
            float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if constexpr (kHm) {
              hm_sum4(w, acc);
            } else {
#pragma unroll
            for (int q = 0; q < kTp; ++q) {
              const __nv_bfloat162* hq = reinterpret_cast<const __nv_bfloat162*>(&w[q]);
#pragma unroll
              for (int q2 = 0; q2 < 4; ++q2) {
                const float2 f = __bfloat1622float2(hq[q2]);
                acc[2 * q2] += f.x;
                acc[2 * q2 + 1] += f.y;
              }
            }
            }
            if constexpr (kHm) {
#pragma unroll
              for (int k = 0; k < 4; ++k) xpk[k] = hm_pack2(acc[2 * k], acc[2 * k + 1]);
            } else {
#pragma unroll
              for (int q = 0; q < 8; ++q) xv[q] = bf16r(acc[q]);
            }
          }
        }
        if constexpr (kHm) {  // (the rounded sums to the position's 4 lanes as 4 bf16 pairs: half the shuffles)
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const uint32_t u = __shfl_sync(0xffffffff, xpk[k], lane & ~3);
            xv[2 * k] = __uint_as_float(u << 16);
            xv[2 * k + 1] = __uint_as_float(u & 0xffff0000u);
          }
        } else {
#pragma unroll
          for (int q = 0; q < 8; ++q) xv[q] = __shfl_sync(0xffffffff, xv[q], lane & ~3);
        }
      } else {
        load8(x_prev + (size_t)t * hidden + d, xv);
      }
      if constexpr (kSkIn) {  // the spare warp's Sinkhorn of the previous boundary's logits
        if (!dry) hc_bar_sk_sync();
#pragma unroll
        for (int i = 0; i < hc; ++i) c[i] = sk_comb[i * hc + j];
      }
      if constexpr (kHm) {
        hm_postmap(p, xv, c, rin, r);
      } else if constexpr (kMoe > 0) {
        // (the contraction written out: r = fma(p, xv, c0 s0), then fma(c_i, s_i, r) -- the generic instances' form,
        // which nvcc chose there; in the kMoe kWarm instances it chose r = p xv first)
        float rs[hc][8];
#pragma unroll
        for (int i = 0; i < hc; ++i) {
          const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&rin[i]);
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float2 f = __bfloat1622float2(h[q]);
            rs[i][2 * q] = f.x;
            rs[i][2 * q + 1] = f.y;
          }
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          r[e] = __fmaf_rn(p, xv[e], __fmul_rn(c[0], rs[0][e]));
#pragma unroll
          for (int i = 1; i < hc; ++i) r[e] = __fmaf_rn(c[i], rs[i][e], r[e]);
        }
      } else {
#pragma unroll
      for (int e = 0; e < 8; ++e) r[e] = p * xv[e];
#pragma unroll
      for (int i = 0; i < hc; ++i) {
        if constexpr (kFast) {
          float rs8[8];
          const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&rin[i]);
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float2 f = __bfloat1622float2(h[q]);
            rs8[2 * q] = f.x;
            rs8[2 * q + 1] = f.y;
          }
#pragma unroll
          for (int e = 0; e < 8; ++e) r[e] += c[i] * rs8[e];
        } else {
          float rs8[8];
          load8(s + (size_t)i * hidden + d, rs8);
#pragma unroll
          for (int e = 0; e < 8; ++e) r[e] += c[i] * rs8[e];
        }
      }
      }
      __syncwarp();  // the 4 lanes of this position have read the old streams
      if (!dry) {
        if (hoist == 2) pdl_wait();
        store8(s + (size_t)j * hidden + d, r);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) r[e] = bf16r(r[e]);
    } else if (dry) {
#pragma unroll
      for (int e = 0; e < 8; ++e) r[e] = 0.f;
    } else {
      load8(s + (size_t)j * hidden + d, r);
    }
  }
  if constexpr (kSkIn) {
    if (warp == kHcThreads / 32) {
      // (kSk & 2) the previous boundary's Sinkhorn of this token (its logits in comb: final at this
      // grid's start in the hoisted form, else after the PDL wait above), comb to the x path
      float cm[hc][hc];
      mhc_deferred_comb<kHm>(comb + (size_t)t * hc * hc, hc_eps, iters, cm);
      if constexpr (kHm) {  // (lane 0: the 4 rows as 16-byte stores, no 16-way register select)
        if (lane == 0)
#pragma unroll
          for (int i = 0; i < hc; ++i)
            reinterpret_cast<float4*>(sk_comb)[i] = make_float4(cm[i][0], cm[i][1], cm[i][2], cm[i][3]);
      } else {
        if (lane < hc * hc) sk_comb[lane] = mhc_mat_elem(cm, lane);
      }
      __syncwarp();
      hc_bar_sk_arrive();
    }
  }
  if constexpr (kMoe > 0 && !kMp) {
    // the buffer rotation's clearing of the previous call's region (another buffer; every rank's data of that call
    // has arrived) by the spare warp, off the x-path warps' push -> poll path
    if (warp == kHcThreads / 32) {
      const int nb = (int)gridDim.y * kHcCluster, lb = (int)blockIdx.y * kHcCluster + rank;
      lamport_clear_small<4>(ar.peers, ar.rank, (int)((ar_epoch + 2) % 3), (int)ar_prev_n, ar_prev_kind, ar.max_elems,
                             lb, nb, lane, 32);
    }
  }
  if constexpr (!kTma && !kFr) asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  if constexpr (kTm) {  // (the biases / scales for the reads after the first cluster sum; the norm weights)
    if (tid < kMixes) s_base[tid] = bsv;
    if (tid < 3) s_scale[tid] = scv;
    asm volatile("" : "+r"(wraw.x), "+r"(wraw.y), "+r"(wraw.z), "+r"(wraw.w));
    const __nv_bfloat162* hw = reinterpret_cast<const __nv_bfloat162*>(&wraw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __bfloat1622float2(hw[i]);
      wv[2 * i] = f.x;
      wv[2 * i + 1] = f.y;
    }
  }
  hc_sync_cta<kWarm>(dry);
  // (after a MoE: the kWarm instances' dependents launch here, the others' after the first cluster sum below)
  if (kEt == 0 && kWarm && (kMoe || (kFast && x_prev && ar.state && ar.cm.y && !ar.pushed)) && !dry) pdl_trigger();
  // fused all-reduce (fast form): count this CTA in (every thread has read the call state above);
  // the last of the T x 8 CTAs records the call at its end. No fences: the next call starts after
  // this grid has completed (PDL wait), and the clearing stores are ordered before any peer's
  // write into that buffer by this grid's completion.
  uint32_t ar_arrived = 0;
  if constexpr (kMp && !kXc) {
    if (warp == 0) asm volatile("bar.sync 4, 64;\n" ::: "memory");  // (the code-warming warp has read the call state)
  }
  if ((kMoe || (kFast && ar.state && x_prev)) && tid == 0) ar_arrived = atomicAdd(&ar.state[2], 1u);
  if (warp < kHcThreads / 32 || dry) {
    // ---- partial mixes = r . fn[m, stream j, d..d+8] and the sum of squares
    float acc[kMixes + 1];
    if constexpr (kHm) {
      uint32_t rp[4];
      hm_pack8(r, rp);
      if constexpr (kTma) {
        if (!dry) hc_mb_wait(hc_smem(&hc_mb_fn));
      }
      if constexpr (kFr)
        hm_mixes_r(rp, fr, acc);
      else if constexpr (kTma)
        hm_mixes(rp, fnb + hc_fn_tma_off(xw, j, lane), 4096, acc);
      else
        hm_mixes(rp, hc_fn + j * kHcFnPitch + v * 16, hc * kHcFnPitch, acc);
    } else {
    acc[kMixes] = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[kMixes] += r[e] * r[e];
    if constexpr (kTma) {
      if (!dry) hc_mb_wait(hc_smem(&hc_mb_fn));
    }
    const uint8_t* const fnr = fnb + hc_fn_tma_off(xw, j, lane);  // (kTma)
#pragma unroll
    for (int m = 0; m < kMixes; ++m) {
      float fv[8];
      load8(reinterpret_cast<const bf16*>(kTma ? fnr + m * 4096 : hc_fn + (m * hc + j) * kHcFnPitch + v * 16), fv);
      float a = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) a += r[e] * fv[e];
      acc[m] = a;
    }
    }
    // the warp sums of the 25 values as one butterfly reduce-scatter (31 shuffles instead of 125):
    // at offset o every lane keeps the half of its values whose index has its lane bit o, adding
    // the partner's partials; lane i ends with value i's sum, bitwise warp_sum's (same pairs)
    {
      float v[32];
#pragma unroll
      for (int m = 0; m < 32; ++m) v[m] = m <= kMixes ? acc[m] : 0.f;
#pragma unroll
      for (int o = 16, h = 16; o >= 1; o >>= 1, h >>= 1) {
        const bool up = lane & o;
#pragma unroll
        for (int i = 0; i < h; ++i) {
          const float send = up ? v[i] : v[i + h];
          const float keep = up ? v[i + h] : v[i];
          v[i] = keep + __shfl_xor_sync(0xffffffff, send, o);
        }
      }
      if (lane <= kMixes && !dry) red[warp][lane] = v[0];
    }
    if (gram) {
      // the post-mapped streams' Gram matrix G[j][k] = sum_d r_j[d] r_k[d] (the 4 lanes of a
      // position hold its 4 streams), summed over the lanes of each stream j
      float g[kMaxHc];
#pragma unroll
      for (int k2 = 0; k2 < kMaxHc; ++k2) {
        float a = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) a += r[e] * __shfl_sync(0xffffffff, r[e], (lane & ~3) | k2);
        g[k2] = a;
      }
#pragma unroll
      for (int k2 = 0; k2 < kMaxHc; ++k2) {
        g[k2] += __shfl_xor_sync(0xffffffff, g[k2], 4);
        g[k2] += __shfl_xor_sync(0xffffffff, g[k2], 8);
        g[k2] += __shfl_xor_sync(0xffffffff, g[k2], 16);
      }
      if (lane < kMaxHc && !dry)
#pragma unroll
        for (int k2 = 0; k2 < kMaxHc; ++k2) red[warp][kMixes + 1 + lane * kMaxHc + k2] = g[k2];
    }
  }
  hc_sync_cta<kWarm>(dry);
  const int nred = gram ? kRed : kMixes + 1;
  const int ti = dry ? lane : tid;  // (the warming warp: its lanes as the first threads)
  if constexpr (kFast) {  // (the mbarriers' initialization; the x-path threads with a poll waited before it)
    if (!dry && !cl_waited && warp < kHcThreads / 32) hc_cluster_wait();
  }
  if constexpr (kHm) {  // (lane l < 25 of x-path warp w: value l to CTA w; kHm: no Gram form)
    constexpr int kV = kMixes + 1;
    static_assert(kHcThreads / 32 == kHcCluster, "one x-path warp per CTA of the cluster");
    if (lane < kV && (warp < kHcCluster || dry)) {
      float x = 0.f;
#pragma unroll
      for (int w = 0; w < kHcThreads / 32; ++w) x += red[w][lane];
      if (!dry) hc_st_async(hc_mapa(hc_smem(&part[rank][lane]), warp), x, hc_mapa(hc_smem(&hc_mb[0]), warp));
    }
  } else
  if (ti < nred) {  // this CTA's partial, pushed to every CTA of the cluster
    float x = 0.f;
#pragma unroll
    for (int w = 0; w < kHcThreads / 32; ++w) x += red[w][ti];
    if (!dry) {
      if constexpr (kFast) {
        const uint32_t a = hc_smem(&part[rank][tid]), mb = hc_smem(&hc_mb[0]);
#pragma unroll
        for (int q = 0; q < kHcCluster; ++q) hc_st_async(hc_mapa(a, q), x, hc_mapa(mb, q));
      } else {
#pragma unroll
        for (int q = 0; q < kHcCluster; ++q) cluster.map_shared_rank(&part[rank][0], q)[tid] = x;
      }
    }
  }
  // (the outputs' addresses and the NVFP4 scale's constant reciprocal while the first sum is in flight, off the chain
  // after the second: opaque moves keep them computed here)
  bf16* xo = x_out + (size_t)t * hidden + d;
  uint8_t* x4p = reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(x4) + (size_t)t * (hidden / 2) + d / 2);
  uint8_t* x4sp = reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(x4s) + nvfp4_sf_off(t, d / 16, hidden));
  float q4r = rcp_approx(q4_sfs);
  asm volatile("" : "+l"(xo), "+l"(x4p), "+l"(x4sp), "+f"(q4r));
  if (!dry) {
    if constexpr (kFast) {
      if (tid < nred) hc_mb_wait(hc_smem(&hc_mb[0]));
    } else {
      hc_cluster_arrive();
      hc_cluster_wait();
    }
  }
  // (after a MoE: the instances without the code-warming warp)
  if (kEt == 0 && !kWarm && (kMoe || (kFast && x_prev && ar.state && ar.cm.y && !ar.pushed))) pdl_trigger();
  if (ti < nred) {
    float x = 0.f;
#pragma unroll
    for (int q = 0; q < kHcCluster; ++q) x += part[q][ti];
    if (!dry) mx[tid] = x;
  }
  hc_sync_cta<kWarm>(dry);
  // (kHm: hidden is 4096 here (the host's check), so the division by the power of two hc * hidden is the
  // multiplication by its exact reciprocal: the same correctly rounded value)
  const float rs = kHm ? rsqrtf(mx[kMixes] * (1.f / (hc * kHcCluster * kHcSlice)) + rms_eps)
                       : rsqrtf(mx[kMixes] / (hc * hidden) + rms_eps);
  if (warp == kHcThreads / 32) {
    // ---- Sinkhorn: comb / post for the next boundary (rank 0 publishes them)
    if (hoist == 2) pdl_wait();
    if (!kFast && !gram) hc_cluster_arrive();  // (this warp pushes nothing for the second sum)
    if constexpr (kSkOut) {  // (kSk & 1) post and the comb logits: the next boundary's Sinkhorn
      if (rank == 0)
        mhc_logits_warp(mx, rs, s_base, s_scale, post_mult, post + (size_t)t * hc,
                        comb + (size_t)t * hc * hc);
    } else {
      mhc_mix_warp(mx, rs, s_base, s_scale, hc_eps, iters, post_mult, pre_scr,
                   rank == 0 ? post + (size_t)t * hc : post_scr,
                   rank == 0 ? comb + (size_t)t * hc * hc : comb_scr);
    }
    if (!kFast && !gram) hc_cluster_wait();
    return;
  }
  // ---- layer input x = bf16(sum_i pre_i * stream_i): sum over the 4 lanes of a position
  const float pl = mx[j] * rs * s_scale[0] + s_base[j];
  const float pj = 1.f / (1.f + expf(-pl)) + hc_eps;
  float xv[8];
  float pr[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) pr[e] = r[e];
  if constexpr (kHm) hm_scale8(pr, pj);
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    float x = kHm ? pr[e] : pj * r[e];
    x += __shfl_xor_sync(0xffffffff, x, 1);
    x += __shfl_xor_sync(0xffffffff, x, 2);
    xv[e] = bf16r(x);
  }
  float tot = 0.f;
  if (gram) {
    // x's sum of squares from the Gram matrix: sum_jk pre_j pre_k G_jk (no second cluster sum;
    // over x before its bf16 rounding, so rn differs from the rounded-x form in the last bits)
    float pa[kMaxHc];
#pragma unroll
    for (int jj = 0; jj < kMaxHc; ++jj)
      pa[jj] = 1.f / (1.f + expf(-(mx[jj] * rs * s_scale[0] + s_base[jj]))) + hc_eps;
#pragma unroll
    for (int jj = 0; jj < kMaxHc; ++jj)
#pragma unroll
      for (int kk = 0; kk < kMaxHc; ++kk) tot += pa[jj] * pa[kk] * mx[kMixes + 1 + jj * kMaxHc + kk];
  } else {
    float ss = 0.f;
    if (j == 0)
#pragma unroll
      for (int e = 0; e < 8; ++e) ss += xv[e] * xv[e];
    if constexpr (kHm) {  // (the xor 2 / 1 levels add +0 in lane 0)
#pragma unroll
      for (int o = 16; o >= 4; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
    } else {
      ss = warp_sum(ss);
    }
    if (lane == 0 && !dry) red[warp][0] = ss;
    if (!dry) hc_bar_x();
    if constexpr (kHm && kFast) {  // (lane q of warp 0 pushes to CTA q)
      if (warp == 0 && lane < kHcCluster) {
        float x = 0.f;
#pragma unroll
        for (int w = 0; w < kHcThreads / 32; ++w) x += red[w][0];
        hc_st_async(hc_mapa(hc_smem(&part2[rank]), lane), x, hc_mapa(hc_smem(&hc_mb[1]), lane));
      }
    } else
    if (tid == 0) {
      float x = 0.f;
#pragma unroll
      for (int w = 0; w < kHcThreads / 32; ++w) x += red[w][0];
      if constexpr (kFast) {
        const uint32_t a = hc_smem(&part2[rank]), mb = hc_smem(&hc_mb[1]);
#pragma unroll
        for (int q = 0; q < kHcCluster; ++q) hc_st_async(hc_mapa(a, q), x, hc_mapa(mb, q));
      } else {
#pragma unroll
        for (int q = 0; q < kHcCluster; ++q) *cluster.map_shared_rank(&part2[rank], q) = x;
      }
    }
    if (!dry) {
      if constexpr (kFast) {
        hc_mb_wait(hc_smem(&hc_mb[1]));
      } else {
        hc_cluster_arrive();
        hc_cluster_wait();
      }
    }
#pragma unroll
    for (int q = 0; q < kHcCluster; ++q) tot += part2[q];
  }
  const float rn = kHm ? rsqrtf(tot * (1.f / (kHcCluster * kHcSlice)) + norm_eps) : rsqrtf(tot / hidden + norm_eps);
  // ---- RMSNorm (+ FP8 1x128 quantization; a 128-element block spans warps 2b, 2b + 1)
  float o[8];
  float amax = 0.f;
  if constexpr (kHm) {  // ((xv rn) wv by pairs)
#pragma unroll
    for (int k = 0; k < 4; ++k)
      hm_up(hm_mul2(hm_mul2(hm_pk(xv[2 * k], xv[2 * k + 1]), hm_pk(rn, rn)), hm_pk(wv[2 * k], wv[2 * k + 1])), o[2 * k],
            o[2 * k + 1]);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      o[e] = bf16r(o[e]);
      amax = fmaxf(amax, fabsf(o[e]));
    }
  } else {
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    o[e] = bf16r(xv[e] * rn * wv[e]);
    amax = fmaxf(amax, fabsf(o[e]));
  }
  }
  if (j == 0 && !dry) store8(xo, o);
  if (xq) {
    if constexpr (kHm) {  // (the 4 lanes of a position hold the same amax)
#pragma unroll
      for (int o = 16; o >= 4; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    } else {
      amax = warp_max(amax);
    }
    if (lane == 0 && !dry) wamax[warp] = amax;
    if (!dry) hc_bar_x();
    const float sc = fmaxf(fmaxf(wamax[xw & ~1], wamax[xw | 1]), 1e-12f) * (1.f / 448.f);
    if (j == 0) {
      const float ys = __frcp_rn(sc);
      const __nv_fp8x4_e4m3 q0(make_float4(div_rn(o[0], sc, ys), div_rn(o[1], sc, ys),
                                           div_rn(o[2], sc, ys), div_rn(o[3], sc, ys)));
      const __nv_fp8x4_e4m3 q1(make_float4(div_rn(o[4], sc, ys), div_rn(o[5], sc, ys),
                                           div_rn(o[6], sc, ys), div_rn(o[7], sc, ys)));
      if (!dry) {
        *reinterpret_cast<uint2*>(xq + (size_t)t * hidden + d) = make_uint2(q0.__x, q1.__x);
        if ((d & 127) == 0) xs[(size_t)t * (hidden / 128) + d / 128] = sc;
      }
    }
  }
  if (x4) {
    // NVFP4 of x (the MoE's input; nvfp4_quant's quant16 arithmetic): a 16-element block is the
    // j == 0 lanes of positions v, v + 1 (lanes l, l + 4): their amax combined, each lane
    // quantizes and stores its own 8 values (4 bytes), the even one the block's scale
    const float a16 = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 4));
    if (j == 0) {
      float sfv = q4_sfs * (a16 * rcp_approx(6.f));
      const __nv_fp8_e4m3 s8(sfv);
      sfv = static_cast<float>(s8);
      const float osc = a16 != 0.f ? rcp_approx(sfv * q4r) : 0.f;
      float w[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) w[e] = o[e];
      if constexpr (kHm) {
        hm_scale8(w, osc);
      } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) w[e] = o[e] * osc;
      }
      const uint32_t q = hc_e2m1x8(w);
      if (!dry) {
        *reinterpret_cast<uint32_t*>(x4p) = q;
        if (!(v & 1)) *x4sp = s8.__x;
      }
    }
  }
  if constexpr (kMp && !kXc) {
    if (dry) {  // (after the dry pass) the previous call's region of another buffer: the state is final after the wait
      pdl_wait();
      const uint32_t ep = next_epoch(__ldcg(&ar.state[0])), pn = __ldcg(&ar.state[1]), pk = __ldcg(&ar.state[3]);
      const int nb = (int)gridDim.y * kHcCluster, lb = (int)blockIdx.y * kHcCluster + rank;
      lamport_clear_small<4>(ar.peers, ar.rank, (int)((ep + 2) % 3), (int)pn, pk, ar.max_elems, lb, nb, lane, 32);
      asm volatile("bar.arrive 4, 64;\n" ::: "memory");  // (after the clearing, as kPre's: the state reads are done)
    }
  }
  if constexpr (kFast) {
    // the last of the T x 8 CTAs records this all-reduce call (the buffer rotation's state)
    if ((kMoe || (ar.state && x_prev)) && tid == 0 && ar_arrived == gridDim.y * kHcCluster - 1) {
      ar.state[0] = ar_epoch;
      ar.state[1] = (uint32_t)(gridDim.y * hidden);
      ar.state[3] = 1u;
      ar.state[2] = 0u;
    }
  } else if (ar.state && x_prev && !dry) {
    // the fused all-reduce's share of the buffer rotation: clear the previous call's region of
    // the buffer it used, then the last of the T x 8 CTAs records this call
    const int nb = (int)gridDim.y * kHcCluster, lb = (int)blockIdx.y * kHcCluster + rank;
    if (ar.tp == 4)
      lamport_clear<4>(ar.peers, ar.rank, (int)((ar_epoch + 2) % 3), (int)ar.state[1], ar.state[3],
                       ar.max_elems, lb, nb, tid, kHcThreads);
    hc_bar_x();
    if (tid == 0) lamport_finish_thread(ar.state, ar_epoch, (int)gridDim.y * hidden, 1, nb);
  }
}

// ------------------------------------------------------------------ 16-CTA cluster form (few tokens)
// The fast form of mhc_fused_cluster_kernel on 16-CTA clusters (non-portable size; launch_pdl_cluster) for the
// boundary after the attention, whose sums the o_proj GEMV pushes: CTA r of the cluster owns hidden elements
// [256 r, 256 r + 256) with 4 x-path warps, and the CTA (51 KB of fn + 6 KB of shared memory, 6 warps, 80-90
// registers) fits beside an o_proj CTA (gemv_pre_lean with the largest carveout; gemv_tma's pushed form), so it is
// resident and through its prologue while the o_proj still runs, instead of starting when the o_proj's CTAs leave.
// Warp w of CTA r holds exactly what warp 4 (r & 1) + w of CTA r / 2 holds in the 8-CTA form (same positions,
// streams, products and butterfly). The cluster sums go per CTA pair: the even CTA (the 8-CTA rank's warps 0-3)
// pushes the sum from 0 of its 4 warps' partials in warp order, the odd CTA (warps 4-7) its 4 partials as one
// 16-byte st.async; every receiver adds the odd partials on and then the 8 pairs in rank order -- bitwise the 8-CTA
// form's results. The next kernel launches after the second cluster sum (trig 0: at the start): launched at the
// start of a boundary that is resident this early, the router and the MoE (which triggers at its start) would take
// the SMs and cost more than the boundary gains. No Gram form, no MoE combine, no non-fast form.
constexpr int kC16 = 16;
constexpr int kC16Slice = 256;                     // hidden elements per CTA (hidden 4096)
constexpr int kC16XT = kC16Slice / 2;              // x-path threads (4 warps)
constexpr int kC16XW = kC16XT / 32;
constexpr int kC16All = kC16XT + 32;               // + the Sinkhorn warp
constexpr int kC16Warm = kC16All;                  // the code-warming warp's first thread
constexpr int kC16Threads = kC16All + 32;
constexpr int kC16FnPitch = kC16Slice * 2 + 32;    // bytes per (mix, stream) fn row (conflict-free 16-byte loads)
constexpr int kC16Smem = kMixes * kMaxHc * kC16FnPitch;
__device__ __forceinline__ void c16_bar_x() {  // the 4 x-path warps only
  asm volatile("bar.sync 1, %0;\n" ::"n"(kC16XT) : "memory");
}
__device__ __forceinline__ void c16_bar_sk_arrive() {
  asm volatile("bar.arrive 2, %0;\n" ::"n"(kC16All) : "memory");
}
__device__ __forceinline__ void c16_bar_sk_sync() {
  asm volatile("bar.sync 2, %0;\n" ::"n"(kC16All) : "memory");
}
template <bool kWarm>
__device__ __forceinline__ void c16_sync_cta(bool dry) {  // the x-path warps + the Sinkhorn warp
  if constexpr (kWarm) {
    if (!dry) asm volatile("bar.sync 3, %0;\n" ::"n"(kC16All) : "memory");
  } else {
    __syncthreads();
  }
}
__device__ __forceinline__ void hc_st_async4(uint32_t a, float4 v, uint32_t mb) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];\n" ::"r"(a),
               "r"(__float_as_uint(v.x)), "r"(__float_as_uint(v.y)), "r"(__float_as_uint(v.z)), "r"(__float_as_uint(v.w)),
               "r"(mb)
               : "memory");
}

// (trig: 0 the next kernel may launch at the start, else after the second cluster sum)
// (kTma: the fn slice by one TMA tensor copy, as mhc_fused_cluster_kernel's kTma: box {64, 2, 4, 2, 24} = 48 KB)
// (kHm: as mhc_fused_cluster_kernel's)
// (kFr: kHm + kTma, the fn rows in registers before the post-poll barrier, as mhc_fused_cluster_kernel's kFr: loaded
// before the poll)
// (kPre: kWarm, producer-pushed calls only (hoisted: the call state is final at the start), the pre-poll path shortened:
// the call state first, the biases / scales stored after the post-mapping, the previous call's clearing by the
// code-warming warp (named barrier 4 to warp 0 before the CTA counts itself in) instead of the x-path threads between
// the call state and the first poll)
// (kFr: at most 168 registers (two CTAs' worth of the register file), so that a CTA still fits beside a pushed gemv_tma
// o_proj CTA: 2 x 32 x 168 + 2 x 32 x 72 registers in the sub-partitions holding two warps of each)
// (kCl: kPre + kSk & 2, the spare warp's comb logits copied into shared memory at the start)
// (kRf: kCl + a MoE layer's tensor-core router in its RF form after this boundary: the x rows also go
// into the kRouterRfGroups copies at ar_rf_x(ar), -0.0 as +0.0 (the Lamport sentinel), before x_out's own store; the
// router polls them instead of waiting for this grid)
template <int kSk = 0, bool kWarm = false, bool kTma = false, bool kHm = false, bool kFr = false, bool kPre = false,
          bool kCl = false, bool kRf = false>
__global__ void __launch_bounds__(kWarm ? kC16Threads : kC16All, kFr ? 2 : 1)
    mhc_fused_c16_kernel(bf16* __restrict__ streams, const bf16* __restrict__ x_prev,
                         float* __restrict__ post, float* __restrict__ comb, bf16* __restrict__ x_out,
                         fp8* __restrict__ xq, float* __restrict__ xs, const bf16* __restrict__ fn,
                         const float* __restrict__ base, const float* __restrict__ scale,
                         const bf16* __restrict__ norm_w, int hidden_arg, float rms_eps, float hc_eps,
                         int iters, float post_mult, float norm_eps, const ArFuse ar,
                         uint8_t* __restrict__ x4, uint8_t* __restrict__ x4s, float q4_sfs, int trig,
                         const __grid_constant__ CUtensorMap tm_fn) {
  if (trig == 0) pdl_trigger();
  constexpr int hidden = kC16 * kC16Slice;  // (the host runs this form at hidden 4096 only)
  (void)hidden_arg;
  // (kPre) the call state before anything else (final at this grid's start: see hoist below)
  uint32_t ar_epoch = 0, ar_prev_n = 0, ar_prev_kind = 0;
  if constexpr (kPre) {
    ar_epoch = next_epoch(__ldcg(&ar.state[0]));
    ar_prev_n = __ldcg(&ar.state[1]);
    ar_prev_kind = __ldcg(&ar.state[3]);
  }
  // (kPre + kSk & 2) the spare warp's comb logits, final at this grid's start as the call state (hoisted: the producer
  // GEMV triggers after its PDL wait): loaded with it, not after the prologue, so that its Sinkhorn (which can end
  // after the o_proj's sums) starts earlier
  // (the 64 bytes by 4 lanes' 16-byte cp.async at L2 into shared memory: no registers held across the x path's code)
  constexpr bool kClEarly = kPre && kCl && (kSk & 2) != 0;
  static_assert(!kCl || (kPre && (kSk & 2) != 0), "kCl: the kPre instances with the deferred Sinkhorn's input");
  __shared__ __align__(16) float cl_s[kClEarly ? kMaxHc * kMaxHc : 4];
  if constexpr (kClEarly) {
    if ((threadIdx.x >> 5) == kC16XW) {
      if ((threadIdx.x & 31) < 4)
        hc_cp_async16(&cl_s[(threadIdx.x & 31) * 4], comb + (size_t)blockIdx.y * kMaxHc * kMaxHc + (threadIdx.x & 31) * 4);
      asm volatile("cp.async.commit_group;\n" ::);
    }
  }
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  constexpr int hc = kMaxHc;
  constexpr int kV = kMixes + 1;  // the 24 mixes and the sum of squares of the streams
  constexpr int kP = kC16 / 2;    // CTA pairs (= the 8-CTA form's ranks)
  constexpr bool kSkIn = (kSk & 2) != 0, kSkOut = (kSk & 1) != 0;
  static_assert(!kFr || (kHm && kTma), "kFr: the kHm + kTma instances");
  static_assert(!kPre || kWarm, "kPre: the kWarm instances");
  extern __shared__ __align__(16) uint8_t hc_fn[];  // [mix * hc + stream][kC16FnPitch bytes]
  const int t = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const bool dry = kWarm && tid >= kC16Warm;  // the warming warp: as thread xt of x-path warp 1
  const int xt = dry ? tid - kC16Warm + 32 : tid, xw = xt >> 5;
  const int rank = (int)cluster.block_rank();
  const int j = lane & 3;                                       // stream
  const int v = xw * 8 + (lane >> 2);                           // 8-element vector of the slice
  const int d = rank * kC16Slice + v * 8;                       // first element (x-path threads)
  // the cluster sums: [pair][value] the even CTAs' prefix sums and the odd CTAs' 4 partials; this CTA's warp partials
  __shared__ float hs_even[kP][kV], hs_even2[kP];
  __shared__ __align__(16) float4 hs_odd[kP][kV], hs_odd2[kP];
  __shared__ float hs_red[kC16XW][kV], hs_red2[kC16XW];
  __shared__ __align__(8) uint64_t hc_mb[2];      // the two sums' mbarriers
  __shared__ float mx[kV];
  __shared__ float wamax[kC16XW];
  __shared__ float pre_scr[hc], post_scr[hc], comb_scr[hc * hc];
  __shared__ float s_base[kMixes], s_scale[3];
  __shared__ __align__(16) float sk_comb[kSkIn ? hc * hc : 4];  // (kSk & 2) comb by the spare warp
  __shared__ __align__(8) uint64_t hc_mb_fn;      // (kTma) the fn copy's mbarrier
  uint8_t* const fnb = kTma ? hc_fn + ((1024u - (hc_smem(hc_fn) & 1023u)) & 1023u) : hc_fn;
  if constexpr (kTma) {
    if (tid == 0) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(hc_smem(&hc_mb_fn)) : "memory");
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      hc_mb_expect(hc_smem(&hc_mb_fn), kMixes * hc * kC16Slice * 2);
      hc_tma_fn(hc_smem(fnb), &tm_fn, rank * (kC16Slice / 128), hc_smem(&hc_mb_fn));
    }
  } else if (!dry) {
    for (int i = tid; i < kMixes * hc * (kC16Slice / 8); i += kC16All) {
      const int row = i / (kC16Slice / 8), c = i % (kC16Slice / 8);
      hc_cp_async16(hc_fn + row * kC16FnPitch + c * 16,
                    fn + (size_t)row * hidden + rank * kC16Slice + c * 8);
    }
    asm volatile("cp.async.commit_group;\n" ::);
  }
  float bsv = 0.f, scv = 0.f;  // (kPre: stored after the post-mapping)
  if constexpr (kPre) {
    if (tid < kMixes) bsv = base[tid];
    if (tid < 3) scv = scale[tid];
  } else {
    if (tid < kMixes) s_base[tid] = base[tid];
    if (tid < 3) s_scale[tid] = scale[tid];
  }
  float wv[8];
  if (warp < kC16XW || dry) load8(norm_w + d, wv);
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(hc_smem(&hc_mb[0])) : "memory");
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(hc_smem(&hc_mb[1])) : "memory");
    // (the only arrival of each: the bytes the 8 CTA pairs push, expected before any push lands)
    hc_mb_expect(hc_smem(&hc_mb[0]), kP * kV * 20);
    hc_mb_expect(hc_smem(&hc_mb[1]), kP * 20);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
  const int hoist = ar.pushed >= 2 ? ar.pushed : 0;
  if (!hoist && !dry) pdl_wait();
  float r[8];
  uint4 fr[kFr ? kMixes : 1];  // (kFr) this thread's fn rows
  if constexpr (!kPre) {
    if (!dry) {
      ar_epoch = next_epoch(__ldcg(&ar.state[0]));
      ar_prev_n = __ldcg(&ar.state[1]);
      ar_prev_kind = __ldcg(&ar.state[3]);
    }
  } else if (dry) {  // (kPre) the previous call's clearing by the code-warming warp, before its dry pass; named barrier
    // 4 hands it to warp 0 before this CTA counts itself in (the last CTA's call record follows every CTA's clearing)
    const int nb = (int)gridDim.y * kC16, lb = (int)blockIdx.y * kC16 + rank;
    lamport_clear_small<4>(ar.peers, ar.rank, (int)((ar_epoch + 2) % 3), (int)ar_prev_n, ar_prev_kind, ar.max_elems, lb,
                           nb, lane, 32);
    asm volatile("bar.arrive 4, 64;\n" ::: "memory");
  }
  if (warp < kC16XW || dry) {
    bf16* s = streams + (size_t)t * hc * hidden;
    uint4 u = make_uint4(0, 0, 0, 0);
    if (!ar.pushed && j == 0 && !dry) u = *reinterpret_cast<const uint4*>(x_prev + (size_t)t * hidden + d);
    float c[hc];
    if constexpr (!kSkIn) {
#pragma unroll
      for (int i = 0; i < hc; ++i) c[i] = dry ? 0.f : __ldcg(comb + (size_t)t * hc * hc + i * hc + j);
    }
    const float p = dry ? 0.f : __ldcg(post + (size_t)t * hc + j);
    uint4 rin[hc];
#pragma unroll
    for (int i = 0; i < hc; ++i)
      rin[i] = dry ? make_uint4(0, 0, 0, 0) : __ldcg(reinterpret_cast<const uint4*>(s + (size_t)i * hidden + d));
    float xv[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) xv[q] = 0.f;
    if (!dry) {
      const size_t e = (size_t)t * hidden + d;
      const size_t slot = (size_t)ar.tp * ar.max_elems;
      uint32_t xpk[4] = {0u, 0u, 0u, 0u};  // (kHm) the stream-0 lane's rounded sums as bf16 pairs
      if (j == 0 && !ar.pushed) {
        u.x = clear_neg_zero(u.x);
        u.y = clear_neg_zero(u.y);
        u.z = clear_neg_zero(u.z);
        u.w = clear_neg_zero(u.w);
        for (int q = 0; q < ar.tp; ++q)
          *reinterpret_cast<uint4*>(ar.peers.buf[q] + (ar_epoch % 3) * slot + (size_t)ar.rank * ar.max_elems + e) = u;
      }
      if constexpr (!kPre) {  // the previous call's region of another buffer (the rotation's clearing)
        const int nb = (int)gridDim.y * kC16, lb = (int)blockIdx.y * kC16 + rank;
        lamport_clear_small<4>(ar.peers, ar.rank, (int)((ar_epoch + 2) % 3), (int)ar_prev_n, ar_prev_kind, ar.max_elems,
                         lb, nb, tid, kC16XT);
      }
      if constexpr (kFr) {  // the fn rows into registers while the o_proj's sums are in flight
        // (the x-path warps' barrier orders tid 0's mbarrier.init of hc_mb_fn before warps 1-3 wait on it: without it a
        // warp running ahead of warp 0 could poll the previous CTA's bytes there -- a completed mbarrier of the same
        // layout passes at once and the fn rows would be read before this CTA's copy lands. Warp 0 reaches it right
        // after issuing the copy, which the wait below waits for anyway)
        c16_bar_x();
        hc_mb_wait(hc_smem(&hc_mb_fn));
        const uint8_t* const f0 = fnb + hc_fn_tma_off(xw, j, lane);
#pragma unroll
        for (int m = 0; m < kMixes; ++m) fr[m] = *reinterpret_cast<const uint4*>(f0 + m * 2048);
      }
      // the cluster barrier (the mbarriers' initialization, every CTA arrived at its start) before the poll, which
      // waits for the o_proj's sums anyway: after the post-poll barrier it would hold the mixes
      hc_cluster_wait();
      if (j == 0) {  // every source's chunk polled at once, summed in rank order
        const bf16* mine = ar.peers.buf[ar.rank] + (ar_epoch % 3) * slot;
        constexpr int kTp = 4;
        uint4 w[kTp];
        unsigned need = (1u << kTp) - 1u;
        while (need) {
#pragma unroll
          for (int q = 0; q < kTp; ++q)
            if (need >> q & 1u) w[q] = ld_volatile16(mine + (size_t)q * ar.max_elems + e);
#pragma unroll
          for (int q = 0; q < kTp; ++q) {
            if constexpr (kHm) {  // (branch-free: a complete source holds its chunk, re-testing it is a no-op)
              need &= ~((hm_sent4(w[q]) ? 0u : 1u) << q);
            } else if ((need >> q & 1u) && !(has_sentinel(w[q].x) || has_sentinel(w[q].y) || has_sentinel(w[q].z) ||
                                             has_sentinel(w[q].w))) {
              need &= ~(1u << q);
            }
          }
        }
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if constexpr (kHm) {
          hm_sum4(w, acc);
        } else {
#pragma unroll
        for (int q = 0; q < kTp; ++q) {
          const __nv_bfloat162* hq = reinterpret_cast<const __nv_bfloat162*>(&w[q]);
#pragma unroll
          for (int q2 = 0; q2 < 4; ++q2) {
            const float2 f = __bfloat1622float2(hq[q2]);
            acc[2 * q2] += f.x;
            acc[2 * q2 + 1] += f.y;
          }
        }
        }
        if constexpr (kHm) {
#pragma unroll
          for (int k = 0; k < 4; ++k) xpk[k] = hm_pack2(acc[2 * k], acc[2 * k + 1]);
        } else {
#pragma unroll
          for (int q = 0; q < 8; ++q) xv[q] = bf16r(acc[q]);
        }
      }
      if constexpr (kHm) {  // (the rounded sums to the position's 4 lanes as 4 bf16 pairs: half the shuffles)
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const uint32_t u = __shfl_sync(0xffffffff, xpk[k], lane & ~3);
          xv[2 * k] = __uint_as_float(u << 16);
          xv[2 * k + 1] = __uint_as_float(u & 0xffff0000u);
        }
      } else {
#pragma unroll
        for (int q = 0; q < 8; ++q) xv[q] = __shfl_sync(0xffffffff, xv[q], lane & ~3);
      }
    } else if constexpr (kFr) {  // (the warming warp: the same loads, unsynchronized)
      const uint8_t* const f0 = fnb + hc_fn_tma_off(xw, j, lane);
#pragma unroll
      for (int m = 0; m < kMixes; ++m) fr[m] = *reinterpret_cast<const uint4*>(f0 + m * 2048);
    }
    if constexpr (kSkIn) {  // the spare warp's Sinkhorn of the previous boundary's logits
      if (!dry) c16_bar_sk_sync();
#pragma unroll
      for (int i = 0; i < hc; ++i) c[i] = sk_comb[i * hc + j];
    }
    if constexpr (kHm) {
      hm_postmap(p, xv, c, rin, r);
    } else {
#pragma unroll
    for (int e = 0; e < 8; ++e) r[e] = p * xv[e];
#pragma unroll
    for (int i = 0; i < hc; ++i) {
      float rs8[8];
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&rin[i]);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 f = __bfloat1622float2(h[q]);
        rs8[2 * q] = f.x;
        rs8[2 * q + 1] = f.y;
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) r[e] += c[i] * rs8[e];
    }
    }
    __syncwarp();  // the 4 lanes of this position have read the old streams
    if (!dry) {
      if (hoist == 2) pdl_wait();
      store8(s + (size_t)j * hidden + d, r);
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) r[e] = bf16r(r[e]);
  }
  if constexpr (kSkIn) {
    if (warp == kC16XW) {
      float cm[hc][hc];
      if constexpr (kClEarly) {
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncwarp();
#pragma unroll
        for (int i = 0; i < hc * hc; ++i) cm[i / hc][i % hc] = cl_s[i];
        mhc_sinkhorn<kHm>(cm, hc_eps, iters);
      } else {
        mhc_deferred_comb<kHm>(comb + (size_t)t * hc * hc, hc_eps, iters, cm);
      }
      if constexpr (kHm) {  // (lane 0: the 4 rows as 16-byte stores, no 16-way register select)
        if (lane == 0)
#pragma unroll
          for (int i = 0; i < hc; ++i)
            reinterpret_cast<float4*>(sk_comb)[i] = make_float4(cm[i][0], cm[i][1], cm[i][2], cm[i][3]);
      } else {
        if (lane < hc * hc) sk_comb[lane] = mhc_mat_elem(cm, lane);
      }
      __syncwarp();
      c16_bar_sk_arrive();
    }
  }
  if constexpr (!kTma) asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  if constexpr (kPre) {  // (the biases / scales for the reads after the first cluster sum)
    if (tid < kMixes && !dry) s_base[tid] = bsv;
    if (tid < 3 && !dry) s_scale[tid] = scv;
  }
  c16_sync_cta<kWarm>(dry);
  uint32_t ar_arrived = 0;
  if constexpr (kPre) {
    if (warp == 0) asm volatile("bar.sync 4, 64;\n" ::: "memory");  // (the code-warming warp's clearing is done)
  }
  if (tid == 0) ar_arrived = atomicAdd(&ar.state[2], 1u);
  if (warp < kC16XW || dry) {
    // ---- partial mixes = r . fn[m, stream j, d..d+8] and the sum of squares (the 8-CTA form's code)
    float acc[kMixes + 1];
    if constexpr (kHm) {
      uint32_t rp[4];
      hm_pack8(r, rp);
      if constexpr (kTma && !kFr) {
        if (!dry) hc_mb_wait(hc_smem(&hc_mb_fn));
      }
      if constexpr (kFr)
        hm_mixes_r(rp, fr, acc);
      else if constexpr (kTma)
        hm_mixes(rp, fnb + hc_fn_tma_off(xw, j, lane), 2048, acc);
      else
        hm_mixes(rp, hc_fn + j * kC16FnPitch + v * 16, hc * kC16FnPitch, acc);
    } else {
    acc[kMixes] = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[kMixes] += r[e] * r[e];
    if constexpr (kTma) {
      if (!dry) hc_mb_wait(hc_smem(&hc_mb_fn));
    }
    const uint8_t* const fnr = fnb + hc_fn_tma_off(xw, j, lane);  // (kTma)
#pragma unroll
    for (int m = 0; m < kMixes; ++m) {
      float fv[8];
      load8(reinterpret_cast<const bf16*>(kTma ? fnr + m * 2048 : hc_fn + (m * hc + j) * kC16FnPitch + v * 16), fv);
      float a = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) a += r[e] * fv[e];
      acc[m] = a;
    }
    }
    float vv[32];
#pragma unroll
    for (int m = 0; m < 32; ++m) vv[m] = m <= kMixes ? acc[m] : 0.f;
#pragma unroll
    for (int o = 16, h = 16; o >= 1; o >>= 1, h >>= 1) {
      const bool up = lane & o;
#pragma unroll
      for (int i = 0; i < h; ++i) {
        const float send = up ? vv[i] : vv[i + h];
        const float keep = up ? vv[i + h] : vv[i];
        vv[i] = keep + __shfl_xor_sync(0xffffffff, send, o);
      }
    }
    if (lane < kV && !dry) hs_red[warp][lane] = vv[0];
    if (!dry) {
      c16_bar_x();
      if constexpr (kHm) {  // (lane l < 25 of x-path warp w: value l to CTAs 4 w .. 4 w + 3)
        if (lane < kV) {
          const int vi = lane, q0 = 4 * warp;
          const uint32_t mb = hc_smem(&hc_mb[0]);
          if (rank & 1) {
            const float4 o = make_float4(hs_red[0][vi], hs_red[1][vi], hs_red[2][vi], hs_red[3][vi]);
            const uint32_t a = hc_smem(&hs_odd[rank >> 1][vi]);
#pragma unroll
            for (int q = q0; q < q0 + 4; ++q) hc_st_async4(hc_mapa(a, q), o, hc_mapa(mb, q));
          } else {
            float x = 0.f;
#pragma unroll
            for (int w = 0; w < kC16XW; ++w) x += hs_red[w][vi];
            const uint32_t a = hc_smem(&hs_even[rank >> 1][vi]);
#pragma unroll
            for (int q = q0; q < q0 + 4; ++q) hc_st_async(hc_mapa(a, q), x, hc_mapa(mb, q));
          }
        }
      } else
      if (tid < kV) {  // value `tid` of this CTA into every CTA of the cluster
        const uint32_t mb = hc_smem(&hc_mb[0]);
        if (rank & 1) {  // warps 4-7 of the 8-CTA form's rank: their 4 partials
          const float4 o = make_float4(hs_red[0][tid], hs_red[1][tid], hs_red[2][tid], hs_red[3][tid]);
          const uint32_t a = hc_smem(&hs_odd[rank >> 1][tid]);
#pragma unroll
          for (int q = 0; q < kC16; ++q) hc_st_async4(hc_mapa(a, q), o, hc_mapa(mb, q));
        } else {  // warps 0-3: their sum from 0 in warp order (the 8-CTA form's first four adds)
          float x = 0.f;
#pragma unroll
          for (int w = 0; w < kC16XW; ++w) x += hs_red[w][tid];
          const uint32_t a = hc_smem(&hs_even[rank >> 1][tid]);
#pragma unroll
          for (int q = 0; q < kC16; ++q) hc_st_async(hc_mapa(a, q), x, hc_mapa(mb, q));
        }
      }
    }
  }
  // (the outputs' addresses and the NVFP4 scale's constant reciprocal while the first sum is in flight, off the chain
  // after the second: opaque moves keep them computed here)
  bf16* xo = x_out + (size_t)t * hidden + d;
  uint8_t* x4p = reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(x4) + (size_t)t * (hidden / 2) + d / 2);
  uint8_t* x4sp = reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(x4s) + nvfp4_sf_off(t, d / 16, hidden));
  float q4r = rcp_approx(q4_sfs);
  asm volatile("" : "+l"(xo), "+l"(x4p), "+l"(x4sp), "+f"(q4r));
  if (!dry) {
    if (tid < kV) {
      hc_mb_wait(hc_smem(&hc_mb[0]));
      float x2 = 0.f;  // (the 8-CTA form's order: a rank's 8 warps from 0, then the ranks from 0)
#pragma unroll
      for (int q = 0; q < kP; ++q) {
        float x = hs_even[q][tid];
        const float4 o = hs_odd[q][tid];
        x += o.x;
        x += o.y;
        x += o.z;
        x += o.w;
        x2 += x;
      }
      mx[tid] = x2;
    }
  }
  c16_sync_cta<kWarm>(dry);
  const float rs = kHm ? rsqrtf(mx[kMixes] * (1.f / (hc * kC16 * kC16Slice)) + rms_eps)
                       : rsqrtf(mx[kMixes] / (hc * hidden) + rms_eps);
  if (warp == kC16XW) {
    // ---- Sinkhorn: comb / post for the next boundary (rank 0 publishes them)
    if (hoist == 2) pdl_wait();
    if constexpr (kSkOut) {
      if (rank == 0)
        mhc_logits_warp(mx, rs, s_base, s_scale, post_mult, post + (size_t)t * hc, comb + (size_t)t * hc * hc);
    } else {
      mhc_mix_warp(mx, rs, s_base, s_scale, hc_eps, iters, post_mult, pre_scr,
                   rank == 0 ? post + (size_t)t * hc : post_scr, rank == 0 ? comb + (size_t)t * hc * hc : comb_scr);
    }
    return;
  }
  // ---- layer input x = bf16(sum_i pre_i * stream_i): sum over the 4 lanes of a position
  const float pl = mx[j] * rs * s_scale[0] + s_base[j];
  const float pj = 1.f / (1.f + expf(-pl)) + hc_eps;
  float xv[8];
  float pr[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) pr[e] = r[e];
  if constexpr (kHm) hm_scale8(pr, pj);
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    float x = kHm ? pr[e] : pj * r[e];
    x += __shfl_xor_sync(0xffffffff, x, 1);
    x += __shfl_xor_sync(0xffffffff, x, 2);
    xv[e] = bf16r(x);
  }
  float tot = 0.f;
  {
    float ss = 0.f;
    if (j == 0)
#pragma unroll
      for (int e = 0; e < 8; ++e) ss += xv[e] * xv[e];
    if constexpr (kHm) {  // (the xor 2 / 1 levels add +0 in lane 0)
#pragma unroll
      for (int o = 16; o >= 4; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
    } else {
      ss = warp_sum(ss);
    }
    if (lane == 0 && !dry) hs_red2[warp] = ss;
    if (!dry) c16_bar_x();
    if constexpr (kHm) {  // (lane q of warp 0 pushes to CTA q)
      if (warp == 0 && lane < kC16) {
        const uint32_t mb = hc_smem(&hc_mb[1]);
        if (rank & 1) {
          const float4 o = make_float4(hs_red2[0], hs_red2[1], hs_red2[2], hs_red2[3]);
          hc_st_async4(hc_mapa(hc_smem(&hs_odd2[rank >> 1]), lane), o, hc_mapa(mb, lane));
        } else {
          float x = 0.f;
#pragma unroll
          for (int w = 0; w < kC16XW; ++w) x += hs_red2[w];
          hc_st_async(hc_mapa(hc_smem(&hs_even2[rank >> 1]), lane), x, hc_mapa(mb, lane));
        }
      }
    } else
    if (tid == 0) {  // this CTA's share into every CTA of the cluster (as the first sum)
      const uint32_t mb = hc_smem(&hc_mb[1]);
      if (rank & 1) {
        const float4 o = make_float4(hs_red2[0], hs_red2[1], hs_red2[2], hs_red2[3]);
        const uint32_t a = hc_smem(&hs_odd2[rank >> 1]);
#pragma unroll
        for (int q = 0; q < kC16; ++q) hc_st_async4(hc_mapa(a, q), o, hc_mapa(mb, q));
      } else {
        float x = 0.f;
#pragma unroll
        for (int w = 0; w < kC16XW; ++w) x += hs_red2[w];
        const uint32_t a = hc_smem(&hs_even2[rank >> 1]);
#pragma unroll
        for (int q = 0; q < kC16; ++q) hc_st_async(hc_mapa(a, q), x, hc_mapa(mb, q));
      }
    }
    if (!dry) hc_mb_wait(hc_smem(&hc_mb[1]));
#pragma unroll
    for (int q = 0; q < kP; ++q) {  // (the 8-CTA form's order, as above)
      float x = hs_even2[q];
      const float4 o = hs_odd2[q];
      x += o.x;
      x += o.y;
      x += o.z;
      x += o.w;
      tot += x;
    }
  }
  if (trig != 0) pdl_trigger();
  const float rn = kHm ? rsqrtf(tot * (1.f / (kC16 * kC16Slice)) + norm_eps) : rsqrtf(tot / hidden + norm_eps);
  // ---- RMSNorm (+ FP8 1x128 quantization; a 128-element block spans warps 2b, 2b + 1)
  float o[8];
  float amax = 0.f;
  if constexpr (kHm) {  // ((xv rn) wv by pairs)
#pragma unroll
    for (int k = 0; k < 4; ++k)
      hm_up(hm_mul2(hm_mul2(hm_pk(xv[2 * k], xv[2 * k + 1]), hm_pk(rn, rn)), hm_pk(wv[2 * k], wv[2 * k + 1])), o[2 * k],
            o[2 * k + 1]);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      o[e] = bf16r(o[e]);
      amax = fmaxf(amax, fabsf(o[e]));
    }
  } else {
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    o[e] = bf16r(xv[e] * rn * wv[e]);
    amax = fmaxf(amax, fabsf(o[e]));
  }
  }
  if constexpr (kRf) {  // (the router's copies first: its chain starts at these stores)
    if (j == 0 && !dry) {
      uint4 u;
      __nv_bfloat162* const h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
      for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(o[2 * i], o[2 * i + 1]);
      u = make_uint4(clear_neg_zero(u.x), clear_neg_zero(u.y), clear_neg_zero(u.z), clear_neg_zero(u.w));
      uint4* const rp = reinterpret_cast<uint4*>(ar_rf_x(ar) + (size_t)t * hidden + d);
#pragma unroll
      for (int g = 0; g < kRouterRfGroups; ++g) rp[(size_t)g * (kRouterRfT * hidden / 8)] = u;
    }
  }
  if (j == 0 && !dry) store8(xo, o);
  if (xq) {
    if constexpr (kHm) {  // (the 4 lanes of a position hold the same amax)
#pragma unroll
      for (int o = 16; o >= 4; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    } else {
      amax = warp_max(amax);
    }
    if (lane == 0 && !dry) wamax[warp] = amax;
    if (!dry) c16_bar_x();
    const float sc = fmaxf(fmaxf(wamax[xw & ~1], wamax[xw | 1]), 1e-12f) * (1.f / 448.f);
    if (j == 0) {
      const float ys = __frcp_rn(sc);
      const __nv_fp8x4_e4m3 q0(make_float4(div_rn(o[0], sc, ys), div_rn(o[1], sc, ys),
                                           div_rn(o[2], sc, ys), div_rn(o[3], sc, ys)));
      const __nv_fp8x4_e4m3 q1(make_float4(div_rn(o[4], sc, ys), div_rn(o[5], sc, ys),
                                           div_rn(o[6], sc, ys), div_rn(o[7], sc, ys)));
      if (!dry) {
        *reinterpret_cast<uint2*>(xq + (size_t)t * hidden + d) = make_uint2(q0.__x, q1.__x);
        if ((d & 127) == 0) xs[(size_t)t * (hidden / 128) + d / 128] = sc;
      }
    }
  }
  if (x4) {  // NVFP4 of x (the 8-CTA form's code)
    const float a16 = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 4));
    if (j == 0) {
      float sfv = q4_sfs * (a16 * rcp_approx(6.f));
      const __nv_fp8_e4m3 s8(sfv);
      sfv = static_cast<float>(s8);
      const float osc = a16 != 0.f ? rcp_approx(sfv * q4r) : 0.f;
      float w[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) w[e] = o[e];
      if constexpr (kHm) {
        hm_scale8(w, osc);
      } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) w[e] = o[e] * osc;
      }
      const uint32_t q = hc_e2m1x8(w);
      if (!dry) {
        *reinterpret_cast<uint32_t*>(x4p) = q;
        if (!(v & 1)) *x4sp = s8.__x;
      }
    }
  }
  // the last of the T x 16 CTAs records this all-reduce call (the buffer rotation's state)
  if (tid == 0 && ar_arrived == gridDim.y * kC16 - 1) {
    ar.state[0] = ar_epoch;
    ar.state[1] = (uint32_t)(gridDim.y * hidden);
    ar.state[3] = 1u;
    ar.state[2] = 0u;
  }
}

// ------------------------------------------------------------------ tiled mHC boundary
// For many tokens: kernel A (grid: hidden slices x token tiles) applies the post-mapping to a
// tile of kMhTT tokens x a kMhDS-wide hidden slice of all 4 streams and computes the slice's
// partial mixes with tensor cores (fn read once per token tile); kernel B (one CTA per token)
// reduces the partials, runs Sinkhorn and writes the normalized (and FP8-quantized) layer input.
constexpr int kMhTT = 16;                 // tokens per tile
constexpr int kMhDS = 256;                // hidden elements per slice
constexpr int kMhK = kMaxHc * kMhDS;      // flattened K per CTA (1024)
constexpr int kMhKP = kMhK + 8;           // padded smem row (conflict-free ldmatrix)
constexpr int kMhN = 32;                  // 24 mixes padded to 4 n-tiles
constexpr int kMhThreads = 256;

struct MhTileSmem {
  bf16 a[kMhTT][kMhKP];     // post-mapped streams (bf16) of the tile: [token][stream * 256 + d]
  bf16 f[kMhN][kMhKP];      // fn slice: [mix][stream * 256 + d] (rows >= 24 zero)
  float red[8][kMhTT][kMhN];
  float pc[kMhTT][kMaxHc * kMaxHc + kMaxHc];  // comb (16) + post (4) per token
  float ss[kMhTT];
};

__device__ __forceinline__ void mh_ldsm4(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(p))));
}
__device__ __forceinline__ void mh_mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(kMhThreads) mhc_tile_mix_kernel(
    bf16* __restrict__ streams, const bf16* __restrict__ x_prev, const float* __restrict__ post,
    const float* __restrict__ comb, const bf16* __restrict__ fn, float* __restrict__ partial,
    int T, int hidden) {
  pdl_trigger();
  extern __shared__ __align__(128) uint8_t smem_raw[];
  MhTileSmem& sm = *reinterpret_cast<MhTileSmem*>(smem_raw);
  constexpr int hc = kMaxHc;
  const int slice = blockIdx.x, nslice = gridDim.x, t0 = blockIdx.y * kMhTT;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int d0 = slice * kMhDS;
  const int nt = min(kMhTT, T - t0);
  // fn slice -> smem (weights: before the dependency wait)
  for (int i = tid; i < kMhN * (kMhK / 8); i += kMhThreads) {
    const int m = i / (kMhK / 8), c8 = i % (kMhK / 8);
    const int j = c8 / (kMhDS / 8), dd = (c8 % (kMhDS / 8)) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (m < (2 + hc) * hc)
      v = *reinterpret_cast<const uint4*>(fn + (size_t)m * hc * hidden + (size_t)j * hidden + d0 + dd);
    *reinterpret_cast<uint4*>(&sm.f[m][c8 * 8]) = v;
  }
  pdl_wait();
  for (int i = tid; i < kMhTT * (hc * hc + hc); i += kMhThreads) {
    const int t = i / (hc * hc + hc), q = i % (hc * hc + hc);
    float v = 0.f;
    if (t < nt && x_prev)
      v = q < hc * hc ? comb[(size_t)(t0 + t) * hc * hc + q]
                      : post[(size_t)(t0 + t) * hc + q - hc * hc];
    sm.pc[t][q] = v;
  }
  __syncthreads();
  // post-mapping: item = (token, 8-element vector of the slice); warp w covers token w (+8)
  for (int it = 0; it < 2; ++it) {
    const int t = warp + it * 8, v8 = lane;  // 32 vectors of 8 = 256 elements
    float ssum = 0.f;
    if (t < nt) {
      bf16* s = streams + (size_t)(t0 + t) * hc * hidden;
      const int d = d0 + v8 * 8;
      float r[hc][8];
#pragma unroll
      for (int i = 0; i < hc; ++i) load8(s + (size_t)i * hidden + d, r[i]);
      if (x_prev) {
        float xv[8], o[hc][8];
        load8(x_prev + (size_t)(t0 + t) * hidden + d, xv);
#pragma unroll
        for (int j = 0; j < hc; ++j) {
          const float pj = sm.pc[t][hc * hc + j];
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            float acc = pj * xv[e];
#pragma unroll
            for (int i = 0; i < hc; ++i) acc += sm.pc[t][i * hc + j] * r[i][e];
            o[j][e] = acc;
          }
        }
#pragma unroll
        for (int j = 0; j < hc; ++j) {
          store8(s + (size_t)j * hidden + d, o[j]);
#pragma unroll
          for (int e = 0; e < 8; ++e) r[j][e] = bf16r(o[j][e]);
        }
      }
#pragma unroll
      for (int j = 0; j < hc; ++j) {
#pragma unroll
        for (int e = 0; e < 8; ++e) ssum += r[j][e] * r[j][e];
        store8(&sm.a[t][j * kMhDS + v8 * 8], r[j]);
      }
    } else {
      const float z[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int j = 0; j < hc; ++j) store8(&sm.a[t][j * kMhDS + v8 * 8], z);
    }
    ssum = warp_sum(ssum);
    if (lane == 0) sm.ss[t] = ssum;
  }
  __syncthreads();
  // partial mixes [16 tokens x 32] over this slice: warp w takes K range [128 w, 128 w + 128)
  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
#pragma unroll
  for (int ks = 0; ks < kMhK / 8 / 16; ++ks) {
    const int k0 = warp * (kMhK / 8) + ks * 16;
    uint32_t a[4], b0[4], b1[4];
    mh_ldsm4(a, &sm.a[lane & 15][k0 + (lane >> 4) * 8]);
    mh_ldsm4(b0, &sm.f[(lane & 7) + ((lane >> 4) << 3)][k0 + ((lane >> 3) & 1) * 8]);
    mh_ldsm4(b1, &sm.f[16 + (lane & 7) + ((lane >> 4) << 3)][k0 + ((lane >> 3) & 1) * 8]);
    mh_mma(acc[0], a, b0[0], b0[1]);
    mh_mma(acc[1], a, b0[2], b0[3]);
    mh_mma(acc[2], a, b1[0], b1[1]);
    mh_mma(acc[3], a, b1[2], b1[3]);
  }
#pragma unroll
  for (int ni = 0; ni < 4; ++ni)
#pragma unroll
    for (int e = 0; e < 4; ++e)
      sm.red[warp][(lane >> 2) + ((e >> 1) << 3)][ni * 8 + 2 * (lane & 3) + (e & 1)] = acc[ni][e];
  __syncthreads();
  // partial[t][slice][0..23] = mixes, [24] = sum of squares
  for (int i = tid; i < kMhTT * 32; i += kMhThreads) {
    const int t = i / 32, m = i % 32;
    if (t >= nt || m > (2 + hc) * hc) continue;
    float v;
    if (m < (2 + hc) * hc) {
      v = 0.f;
#pragma unroll
      for (int w = 0; w < 8; ++w) v += sm.red[w][t][m];
    } else {
      v = sm.ss[t];
    }
    partial[((size_t)(t0 + t) * nslice + slice) * 32 + m] = v;
  }
}

// As the cluster kernel: an extra warp runs Sinkhorn (comb / post, first used by the next
// boundary) while the 8 warps of the token compute pre directly and finish x; the norm weights
// load before the PDL wait. Sums keep block_sum's order over the 8 x warps (bitwise block_sum's).
__global__ void __launch_bounds__(kMhThreads + 32) mhc_tile_apply_kernel(
    const bf16* __restrict__ streams, float* __restrict__ post, float* __restrict__ comb,
    bf16* __restrict__ x_out, fp8* __restrict__ xq, float* __restrict__ xs,
    const float* __restrict__ base, const float* __restrict__ scale,
    const bf16* __restrict__ norm_w, const float* __restrict__ partial, int nslice, int hidden,
    float rms_eps, float hc_eps, int iters, float post_mult, float norm_eps) {
  pdl_trigger();
  constexpr int hc = kMaxHc;
  constexpr int kMixes = (2 + hc) * hc;
  constexpr int kXW = kMhThreads / 32;  // x-path warps
  const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ float mx[kMixes + 1];
  __shared__ float pre_scr[hc];
  __shared__ float sh[32];
  const int d = tid * 16;
  float w[16];
  if (warp < kXW) {
    load8(norm_w + d, w);
    load8(norm_w + d + 8, w + 8);
  }
  pdl_wait();
  if (tid <= kMixes) {
    float v = 0.f;
    for (int s2 = 0; s2 < nslice; ++s2) v += partial[((size_t)t * nslice + s2) * 32 + tid];
    mx[tid] = v;
  }
  __syncthreads();
  const float rs = rsqrtf(mx[kMixes] / (hc * hidden) + rms_eps);
  if (warp == kXW) {
    mhc_mix_warp(mx, rs, base, scale, hc_eps, iters, post_mult, pre_scr, post + (size_t)t * hc,
                 comb + (size_t)t * hc * hc);
    return;
  }
  float pre[hc];
#pragma unroll
  for (int i = 0; i < hc; ++i) {
    const float pl = mx[i] * rs * scale[0] + base[i];
    pre[i] = 1.f / (1.f + expf(-pl)) + hc_eps;
  }
  // x = bf16(sum_i pre_i * stream_i): thread owns 16 consecutive elements
  const bf16* s = streams + (size_t)t * hc * hidden;
  float x[16];
#pragma unroll
  for (int e = 0; e < 16; ++e) x[e] = 0.f;
#pragma unroll
  for (int i = 0; i < hc; ++i) {
    float f[16];
    load8(s + (size_t)i * hidden + d, f);
    load8(s + (size_t)i * hidden + d + 8, f + 8);
    const float p = pre[i];
#pragma unroll
    for (int e = 0; e < 16; ++e) x[e] += p * f[e];
  }
  float ss = 0.f;
#pragma unroll
  for (int e = 0; e < 16; ++e) {
    x[e] = bf16r(x[e]);
    ss += x[e] * x[e];
  }
  // block_sum over the kXW x warps (named barrier: the Sinkhorn warp does not take part)
  ss = warp_sum(ss);
  if (lane == 0) sh[warp] = ss;
  asm volatile("bar.sync 1, %0;\n" ::"n"(kMhThreads) : "memory");
  ss = warp_sum(lane < kXW ? sh[lane] : 0.f);
  const float rn = rsqrtf(ss / hidden + norm_eps);
  float o[16];
  float amax = 0.f;
#pragma unroll
  for (int e = 0; e < 16; ++e) {
    o[e] = bf16r(x[e] * rn * w[e]);
    amax = fmaxf(amax, fabsf(o[e]));
  }
  store8(x_out + (size_t)t * hidden + d, o);
  store8(x_out + (size_t)t * hidden + d + 8, o + 8);
  if (xq) {
    // 8 threads (aligned lanes) per 128-element block
#pragma unroll
    for (int q = 1; q < 8; q <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, q));
    const float sc = fmaxf(amax, 1e-12f) * (1.f / 448.f);
    __nv_fp8x4_e4m3* qp = reinterpret_cast<__nv_fp8x4_e4m3*>(xq + (size_t)t * hidden + d);
    const float ys = __frcp_rn(sc);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      qp[i] = __nv_fp8x4_e4m3(make_float4(div_rn(o[4 * i], sc, ys), div_rn(o[4 * i + 1], sc, ys),
                                          div_rn(o[4 * i + 2], sc, ys), div_rn(o[4 * i + 3], sc, ys)));
    if ((lane & 7) == 0) xs[(size_t)t * (hidden / 128) + d / 128] = sc;
  }
}

// ------------------------------------------------------------------ tensor-core cluster boundary
// Boundaries of 13..64 tokens (decode of 4..16 sequences with MTP 3), with the fused all-reduce: one
// 8-CTA cluster per tile of kTT tokens; CTA q owns hidden elements [512 q, 512 q + 512) of the
// tile's tokens. x-path thread (warp w, lane l) owns vector v = 32 (w % 2) + l (8 elements) of all
// 4 streams of tokens w / 2 + 4 i (i < kTT / 4; tiles of 1 or 2 tokens: warps < 2 kTT). Per CTA: the all-reduce exchange of its elements
// (as the cluster kernel's fast form), the post-mapping, the tile's 24 partial mixes on tensor
// cores (the fn slice staged in shared memory before the PDL wait, read once per tile instead of
// once per token) and the post-mapped streams' 4 x 4 Gram matrix, ONE cluster sum of both through
// distributed shared memory, the layer input with its RMSNorm (x's sum of squares from the Gram
// matrix: sum_jk pre_j pre_k G_jk, over x before its bf16 rounding) and FP8 / NVFP4 quantization;
// the extra warp of CTA q runs Sinkhorn for token q of the tile. (The per-token cluster kernel
// reads its fn slice once per token and runs out of SMs above 24 tokens; the tiled path takes a
// separate all-reduce and a second kernel.)
constexpr int kTcK = kMaxHc * kHcSlice;        // flattened K per CTA (stream-major), 2048
constexpr int kTcKP = kTcK + 8;                // padded shared row (conflict-free ldmatrix)
constexpr int kTcG = kMaxHc * (kMaxHc + 1) / 2;  // Gram entries (j <= k), 10
constexpr int kTcVals = kMixes + kTcG;         // values per token in the cluster sum, 34

template <int kTT>
struct TcSmem {
  bf16 f[kMixes][kTcKP];                       // fn slice: [mix][stream * 512 + d]
  bf16 a[kTT + 1][kTcKP];                      // post-mapped streams (bf16): [token][stream * 512 + d]
                                               // (row kTT: zeros, the mma rows of no token)
  float red[kHcThreads / 32][kTT][kMixes];     // per-warp partial mixes
  float gw[(kTT + 3) / 4][kHcThreads / 32][kTcG];  // per (token round, warp) partial Gram entries
  float part[kHcCluster][kTT][kTcVals];        // every CTA's partial mixes + Gram entries
  float mx[kTT][kTcVals];
  float pc[kTT][kMaxHc * kMaxHc + kMaxHc];     // comb (16) + post (4) per token
  int2 pw[kTT][kArMaxK1];                      // MoE combine: pair position, weight bits per token
  float base[kMixes], scale[3];
  float pre_scr[kMaxHc];
};

__device__ __forceinline__ void mh_ldsm2(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(p))));
}

// kSk: the deferred Sinkhorn (ArFuse.sk), as the cluster kernel's: bit 1 writes post and the comb
// logits; bit 2 takes comb as the previous boundary's logits, the spare warp running the tile's
// Sinkhorns (one lane per token) while the all-reduce is in flight
template <int kTT, int kSk = 0, bool kWarm = false>
__global__ void __cluster_dims__(kHcCluster, 1, 1) __launch_bounds__(kWarm ? kTcThreads : kHcThreadsAll, 1)
    mhc_tc_cluster_kernel(bf16* __restrict__ streams, const bf16* __restrict__ x_prev,
                          float* __restrict__ post, float* __restrict__ comb,
                          bf16* __restrict__ x_out, fp8* __restrict__ xq, float* __restrict__ xs,
                          const bf16* __restrict__ fn, const float* __restrict__ base,
                          const float* __restrict__ scale, const bf16* __restrict__ norm_w,
                          int T, int hidden, float rms_eps, float hc_eps, int iters,
                          float post_mult, float norm_eps, const ArFuse ar,
                          uint8_t* __restrict__ x4, uint8_t* __restrict__ x4s, float q4_sfs) {
  pdl_trigger();
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  constexpr int hc = kMaxHc;
  constexpr int kIt = (kTT + 3) / 4;  // tokens per x-path thread
  constexpr int kPc = kMaxHc * kMaxHc + kMaxHc;
  constexpr bool kSkIn = (kSk & 2) != 0, kSkOut = (kSk & 1) != 0;
  extern __shared__ __align__(128) uint8_t tc_raw[];
  TcSmem<kTT>& sm = *reinterpret_cast<TcSmem<kTT>*>(tc_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  // the warming warp (dry): as thread xt of x-path warp 1
  const bool dry = kWarm && tid >= kTcWarm;
  const int xt = dry ? tid - kTcWarm + 32 : tid, xw = xt >> 5;
  if (dry) asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
  const int rank = (int)cluster.block_rank();
  const int t0 = blockIdx.y * kTT;
  const int v = xt & 63;                 // the x-path thread's vector of the slice
  const int dl = v * 8;                  // its first element in the slice
  const int d = rank * kHcSlice + dl;    // ... in the hidden dimension
  const bool act = kTT >= 4 || xt < kTT * 64;  // (whole warps) x-path threads with a token
  // ---- weights: fn slice -> shared memory, the zero row, the mixes' biases and scales; norm
  //      weights -> registers
  if (!dry) {
    for (int i = tid; i < kMixes * (kTcK / 8); i += kHcThreadsAll) {
      const int m = i / (kTcK / 8), c = i % (kTcK / 8);
      const int j = c / (kHcSlice / 8), dd = (c % (kHcSlice / 8)) * 8;
      hc_cp_async16(&sm.f[m][c * 8],
                    fn + (size_t)m * hc * hidden + (size_t)j * hidden + rank * kHcSlice + dd);
    }
    asm volatile("cp.async.commit_group;\n" ::);
    for (int i = tid; i < kTcKP / 8; i += kHcThreadsAll)
      *reinterpret_cast<uint4*>(&sm.a[kTT][i * 8]) = make_uint4(0, 0, 0, 0);
  }
  if (tid < kMixes) sm.base[tid] = base[tid];
  if (tid < 3) sm.scale[tid] = scale[tid];
  float wv[8];
  if (warp < kHcThreads / 32 || dry) load8(norm_w + d, wv);
  // (all-reduce pushed by the producer GEMV, ar.pushed >= 2: inputs loaded and sums polled before
  // the PDL wait, as the cluster kernel's fast form)
  const int hoist = ar.state && x_prev && ar.pushed >= 2 ? ar.pushed : 0;
  // (MoE combine: the tile's pair positions and weights, written by the MoE kernel's routing at its start,
  // requested into L2 while it ends: the push's first, dependent loads)
  if (x_prev && ar.state && ar.cm.y && !ar.pushed && tid < 8) {
    // (entries 0, 31, 62, 93 clipped to the last: every 128-byte line of the range)
    const int n = min(kTT, T - t0) * ar.cm.K1, k = min((tid & 3) * 31, n - 1);
    const size_t i0 = (size_t)t0 * ar.cm.K1 + k;
    prefetch_l2(tid < 4 ? (const void*)(ar.cm.pair_pos + i0) : (const void*)(ar.cm.w + i0));
  }
  if (!hoist && !dry) pdl_wait();
  // fused all-reduce: this call's epoch, the previous call's size / protocol (buffer rotation)
  const bool arx = ar.state && x_prev;
  uint32_t ar_epoch = 0, ar_prev_n = 0, ar_prev_kind = 0;
  if (arx && !dry) {
    ar_epoch = next_epoch(__ldcg(&ar.state[0]));
    ar_prev_n = __ldcg(&ar.state[1]);
    ar_prev_kind = __ldcg(&ar.state[3]);
  }
  if (warp < kHcThreads / 32 || dry) {
    // ---- the push's inputs first: the partial sums of the thread's elements, or (staged in shared
    //      memory by a few threads: few load instructions ahead of the expert rows) the MoE
    //      combine's pair positions and weights; the tile's comb / post (shared memory) and the old
    //      streams (raw until the sums have arrived)
    const bool moe = arx && ar.cm.y;
    if (moe && xt < kTT * ar.cm.K1) {
      const int tt = xt / ar.cm.K1, s2 = xt % ar.cm.K1;
      int2 pw = make_int2(-1, 0);
      if (t0 + tt < T && !dry)
        pw = make_int2(ar.cm.pair_pos[(size_t)(t0 + tt) * ar.cm.K1 + s2],
                       __float_as_int(ar.cm.w[(size_t)(t0 + tt) * ar.cm.K1 + s2]));
      if (!dry) sm.pw[tt][s2] = pw;
    }
    uint4 u[kIt];
#pragma unroll
    for (int it = 0; it < kIt; ++it) {
      const int t = t0 + (xt >> 6) + 4 * it;
      u[it] = make_uint4(0, 0, 0, 0);
      if (act && arx && !ar.cm.y && !ar.pushed && t < T && !dry)
        u[it] = *reinterpret_cast<const uint4*>(x_prev + (size_t)t * hidden + d);
    }
    if (x_prev && xt < kTT * kPc) {
      const int tt = xt / kPc, q = xt % kPc;
      if (!kSkIn || q >= hc * hc) {  // (kSk & 2: comb by the spare warp)
        float val = 0.f;
        if (t0 + tt < T && !dry)
          val = q < hc * hc ? __ldcg(comb + (size_t)(t0 + tt) * hc * hc + q)
                            : __ldcg(post + (size_t)(t0 + tt) * hc + q - hc * hc);
        if (!dry) sm.pc[tt][q] = val;
      }
    }
    uint4 rin[kIt][hc];
#pragma unroll
    for (int it = 0; it < kIt; ++it) {
      const int t = t0 + (xt >> 6) + 4 * it;
      const bool ok = act && t < T && !dry;
#pragma unroll
      for (int i = 0; i < hc; ++i)
        rin[it][i] = ok ? __ldcg(reinterpret_cast<const uint4*>(streams + ((size_t)t * hc + i) * hidden + d))
                        : make_uint4(0, 0, 0, 0);
    }
    float xv[kIt][8];
#pragma unroll
    for (int it = 0; it < kIt; ++it)
#pragma unroll
      for (int e = 0; e < 8; ++e) xv[it][e] = 0.f;
    if (arx) {
      // fused one-shot all-reduce of the thread's elements (fused all-reduces are tp == 4)
      const size_t slot = (size_t)ar.tp * ar.max_elems;
      if (moe && !dry) hc_bar_x();  // pair positions / weights (and comb / post) staged
#pragma unroll
      for (int it = 0; it < kIt; ++it) {
        const int t = t0 + (xt >> 6) + 4 * it, tt = t - t0;
        if (!act || t >= T || ar.pushed) continue;
        if (ar.cm.y) {
          // MoE combine: every local expert row as independent loads, summed in pair order
          int pp[kArMaxK1];
          float ws[kArMaxK1];
#pragma unroll
          for (int s2 = 0; s2 < kArMaxK1; ++s2)
            if (s2 < ar.cm.K1) {
              const int2 pw = sm.pw[tt][s2];
              pp[s2] = pw.x;
              ws[s2] = __int_as_float(pw.y);
            }
          uint4 yr[kArMaxK1];
#pragma unroll
          for (int s2 = 0; s2 < kArMaxK1; ++s2) {
            if constexpr (kWarm) yr[s2] = make_uint4(0, 0, 0, 0);
            if (s2 < ar.cm.K1 && pp[s2] >= 0 && !dry)  // (negative: expert not on this rank)
              yr[s2] = *reinterpret_cast<const uint4*>(ar.cm.y + (size_t)pp[s2] * ar.cm.hidden + d);
          }
          float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
          for (int s2 = 0; s2 < kArMaxK1; ++s2)
            if (s2 < ar.cm.K1 && pp[s2] >= 0) {
              const __nv_bfloat162* hy = reinterpret_cast<const __nv_bfloat162*>(&yr[s2]);
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                const float2 f = __bfloat1622float2(hy[q]);
                acc[2 * q] += ws[s2] * f.x;
                acc[2 * q + 1] += ws[s2] * f.y;
              }
            }
          __nv_bfloat162* hp = reinterpret_cast<__nv_bfloat162*>(&u[it]);
#pragma unroll
          for (int q = 0; q < 4; ++q) hp[q] = __floats2bfloat162_rn(acc[2 * q], acc[2 * q + 1]);
        }
        u[it].x = clear_neg_zero(u[it].x);
        u[it].y = clear_neg_zero(u[it].y);
        u[it].z = clear_neg_zero(u[it].z);
        u[it].w = clear_neg_zero(u[it].w);
        const size_t e = (size_t)t * hidden + d;
        if (!dry)
          for (int q = 0; q < ar.tp; ++q)
            *reinterpret_cast<uint4*>(ar.peers.buf[q] + (ar_epoch % 3) * slot +
                                      (size_t)ar.rank * ar.max_elems + e) = u[it];
      }
      // the buffer rotation's clearing of the previous call's region (another buffer) while the
      // peers' data of this call is in flight
      if (ar.tp == 4 && !dry) {
        const int nb = (int)gridDim.y * kHcCluster, lb = (int)blockIdx.y * kHcCluster + rank;
        lamport_clear<4>(ar.peers, ar.rank, (int)((ar_epoch + 2) % 3), (int)ar_prev_n,
                         ar_prev_kind, ar.max_elems, lb, nb, tid, kHcThreads);
      }
      // poll every source's chunk at once, sum in rank order (bf16, allreduce_lamport_kernel's)
      const bf16* mine = ar.peers.buf[ar.rank] + (ar_epoch % 3) * slot;
#pragma unroll
      for (int it = 0; it < kIt; ++it) {
        const int t = t0 + (xt >> 6) + 4 * it;
        if (!act || t >= T || dry) continue;
        const size_t e = (size_t)t * hidden + d;
        constexpr int kTp = 4;
        uint4 w[kTp];
        unsigned need = (1u << kTp) - 1u;
        while (need) {
#pragma unroll
          for (int q = 0; q < kTp; ++q)
            if (need >> q & 1u) w[q] = ld_volatile16(mine + (size_t)q * ar.max_elems + e);
#pragma unroll
          for (int q = 0; q < kTp; ++q)
            if ((need >> q & 1u) && !(has_sentinel(w[q].x) || has_sentinel(w[q].y) ||
                                       has_sentinel(w[q].z) || has_sentinel(w[q].w)))
              need &= ~(1u << q);
        }
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int q = 0; q < kTp; ++q) {
          const __nv_bfloat162* hq = reinterpret_cast<const __nv_bfloat162*>(&w[q]);
#pragma unroll
          for (int q2 = 0; q2 < 4; ++q2) {
            const float2 f = __bfloat1622float2(hq[q2]);
            acc[2 * q2] += f.x;
            acc[2 * q2 + 1] += f.y;
          }
        }
#pragma unroll
        for (int q = 0; q < 8; ++q) xv[it][q] = bf16r(acc[q]);
      }
    } else if (x_prev) {
#pragma unroll
      for (int it = 0; it < kIt; ++it) {
        const int t = t0 + (xt >> 6) + 4 * it;
        if (act && t < T && !dry) load8(x_prev + (size_t)t * hidden + d, xv[it]);
      }
    }
    if (!dry) {
      if (!moe) hc_bar_x();  // comb / post staged
      if constexpr (kSkIn) hc_bar_sk_sync();  // (kSk & 2) comb by the spare warp
      if (hoist == 2) pdl_wait();
    }
    // ---- post-mapping (in place; the thread owns all 4 streams of its elements), the bf16 streams
    //      -> the tensor-core operand, their partial Gram matrix per token
#pragma unroll
    for (int it = 0; it < kIt; ++it) {
      if (!act) break;
      const int tt = (xt >> 6) + 4 * it, t = t0 + tt;
      float r[hc][8];
#pragma unroll
      for (int i = 0; i < hc; ++i) {
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&rin[it][i]);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const float2 f = __bfloat1622float2(h[q]);
          r[i][2 * q] = f.x;
          r[i][2 * q + 1] = f.y;
        }
      }
      if (x_prev && t < T) {
        bf16* s = streams + (size_t)t * hc * hidden;
        float o[hc][8];
#pragma unroll
        for (int j = 0; j < hc; ++j) {
          const float p = sm.pc[tt][hc * hc + j];
#pragma unroll
          for (int e = 0; e < 8; ++e) o[j][e] = p * xv[it][e];
#pragma unroll
          for (int i = 0; i < hc; ++i) {
            const float c = sm.pc[tt][i * hc + j];
#pragma unroll
            for (int e = 0; e < 8; ++e) o[j][e] += c * r[i][e];
          }
          if (!dry) store8(s + (size_t)j * hidden + d, o[j]);
        }
#pragma unroll
        for (int j = 0; j < hc; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) r[j][e] = bf16r(o[j][e]);
      }
      // (tokens past T: the zero streams loaded above)
      float g[16];
      {
        int gi = 0;
#pragma unroll
        for (int j = 0; j < hc; ++j)
#pragma unroll
          for (int k2 = j; k2 < hc; ++k2) {
            float acc = 0.f;
#pragma unroll
            for (int e = 0; e < 8; ++e) acc += r[j][e] * r[k2][e];
            g[gi++] = acc;
          }
      }
#pragma unroll
      for (int i = kTcG; i < 16; ++i) g[i] = 0.f;
#pragma unroll
      for (int j = 0; j < hc; ++j) {
        uint4 ra;
        __nv_bfloat162* hp = reinterpret_cast<__nv_bfloat162*>(&ra);
#pragma unroll
        for (int q = 0; q < 4; ++q) hp[q] = __floats2bfloat162_rn(r[j][2 * q], r[j][2 * q + 1]);
        if (!dry) *reinterpret_cast<uint4*>(&sm.a[tt][j * kHcSlice + dl]) = ra;
      }
      // warp sums of the 10 Gram entries: fold the warp halves, then a 16-lane butterfly
      // reduce-scatter (lane l ends with entry l % 16)
#pragma unroll
      for (int i = 0; i < 16; ++i) g[i] += __shfl_xor_sync(0xffffffff, g[i], 16);
#pragma unroll
      for (int o2 = 8, h = 8; o2 >= 1; o2 >>= 1, h >>= 1) {
        const bool up = lane & o2;
#pragma unroll
        for (int i = 0; i < h; ++i) {
          const float send = up ? g[i] : g[i + h];
          const float keep = up ? g[i + h] : g[i];
          g[i] = keep + __shfl_xor_sync(0xffffffff, send, o2);
        }
      }
      if (lane < kTcG && !dry) sm.gw[it][xw][lane] = g[0];
    }
  }
  if constexpr (kSkIn) {
    if (warp == kHcThreads / 32) {
      // (kSk & 2) the previous boundary's Sinkhorns of the tile's tokens, one lane per token (its
      // logits in comb: final at this grid's start in the hoisted form, else after the PDL wait
      // above), comb into the tile's shared rows for the post-mapping
      if (lane < kTT && t0 + lane < T) {
        float cm[hc][hc];
        mhc_deferred_comb(comb + (size_t)(t0 + lane) * hc * hc, hc_eps, iters, cm);
#pragma unroll
        for (int i = 0; i < hc; ++i)
#pragma unroll
          for (int q = 0; q < hc; ++q) sm.pc[lane][i * hc + q] = cm[i][q];
      }
      __syncwarp();
      hc_bar_sk_arrive();
    }
  }
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  hc_sync_cta<kWarm>(dry);
  // fused all-reduce: count this CTA in (every thread has read the call state); the last of the
  // CTAs records the call at its end (no fences, as the cluster kernel's fast form)
  uint32_t ar_arrived = 0;
  if (arx && tid == 0) ar_arrived = atomicAdd(&ar.state[2], 1u);
  if (warp < kHcThreads / 32 || dry) {
    // ---- partial mixes [kTT tokens x 24] over this slice: warp w takes K range [256 w, 256 w + 256)
    //      (stream w / 2); token rows >= kTT read the zero row
    float acc[3][4];
#pragma unroll
    for (int n = 0; n < 3; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
    const int arow = (lane & 7) < kTT ? (lane & 7) : kTT;
#pragma unroll 4
    for (int ks = 0; ks < kTcK / 8 / 16; ++ks) {
      const int k0 = xw * (kTcK / 8) + ks * 16;
      uint32_t a2[2], b0[4], b2[2];
      mh_ldsm2(a2, &sm.a[arow][k0 + ((lane >> 3) & 1) * 8]);
      mh_ldsm4(b0, &sm.f[(lane & 7) + ((lane >> 4) << 3)][k0 + ((lane >> 3) & 1) * 8]);
      mh_ldsm2(b2, &sm.f[16 + (lane & 7)][k0 + ((lane >> 3) & 1) * 8]);
      const uint32_t af[4] = {a2[0], 0u, a2[1], 0u};
      mh_mma(acc[0], af, b0[0], b0[1]);
      mh_mma(acc[1], af, b0[2], b0[3]);
      mh_mma(acc[2], af, b2[0], b2[1]);
    }
    const int g = lane >> 2;  // the C fragment's token row (rows 8..15 are no tokens)
    if (g < kTT && !dry) {
#pragma unroll
      for (int n = 0; n < 3; ++n) {
        sm.red[xw][g][n * 8 + 2 * (lane & 3)] = acc[n][0];
        sm.red[xw][g][n * 8 + 2 * (lane & 3) + 1] = acc[n][1];
      }
    }
  }
  hc_sync_cta<kWarm>(dry);
  if (xt < kTT * kTcVals) {  // this CTA's partials, pushed to every CTA of the cluster
    const int t = xt / kTcVals, m = xt % kTcVals;
    float x = 0.f;
    if (m < kMixes) {
#pragma unroll
      for (int w = 0; w < kHcThreads / 32; ++w) x += sm.red[w][t][m];
    } else {
      x = sm.gw[t >> 2][2 * (t & 3)][m - kMixes] + sm.gw[t >> 2][2 * (t & 3) + 1][m - kMixes];
    }
    if (!dry)
#pragma unroll
      for (int q = 0; q < kHcCluster; ++q) *cluster.map_shared_rank(&sm.part[rank][t][m], q) = x;
  }
  if (!dry) {  // (the warming warp's share arrived at its start)
    hc_cluster_arrive();
    hc_cluster_wait();
  }
  if (xt < kTT * kTcVals) {
    const int t = xt / kTcVals, m = xt % kTcVals;
    float x = 0.f;
#pragma unroll
    for (int q = 0; q < kHcCluster; ++q) x += sm.part[q][t][m];
    if (!dry) sm.mx[t][m] = x;
  }
  hc_sync_cta<kWarm>(dry);
  // RMS of the post-mapped streams: the Gram trace (entries (0,0), (1,1), (2,2), (3,3))
  auto rms_rs = [&](int tt) {
    const float* G = &sm.mx[tt][kMixes];
    return rsqrtf(((G[0] + G[4]) + (G[7] + G[9])) / (hc * hidden) + rms_eps);
  };
  if (warp == kHcThreads / 32) {
    if (hoist == 2) pdl_wait();
    // ---- Sinkhorn of token `rank` of the tile: comb / post for the next boundary
    if (rank < kTT && t0 + rank < T) {
      if constexpr (kSkOut)  // (kSk & 1) post and the comb logits: the next boundary's Sinkhorn
        mhc_logits_warp(sm.mx[rank], rms_rs(rank), sm.base, sm.scale, post_mult,
                        post + (size_t)(t0 + rank) * hc, comb + (size_t)(t0 + rank) * hc * hc);
      else
        mhc_mix_warp(sm.mx[rank], rms_rs(rank), sm.base, sm.scale, hc_eps, iters, post_mult,
                     sm.pre_scr, post + (size_t)(t0 + rank) * hc, comb + (size_t)(t0 + rank) * hc * hc);
    }
    return;
  }
  // ---- layer input x = bf16(sum_j pre_j * stream_j) (the cluster kernel's pairing); RMSNorm with
  //      x's sum of squares from the Gram matrix (+ FP8 1x128: a 128-element block is 16 lanes; +
  //      NVFP4: a 16-element block is lanes l, l + 1)
  //      (not unrolled: the body keeps no per-token register arrays)
#pragma unroll 1
  for (int it = 0; it < kIt; ++it) {
    if (!act) break;
    const int tt = (xt >> 6) + 4 * it, t = t0 + tt;
    const bool valid = t < T && !dry;
    const float rs = rms_rs(tt);
    float pre[hc];
#pragma unroll
    for (int j = 0; j < hc; ++j)
      pre[j] = 1.f / (1.f + expf(-(sm.mx[tt][j] * rs * sm.scale[0] + sm.base[j]))) + hc_eps;
    const float* G = &sm.mx[tt][kMixes];
    const float ssx = pre[0] * (pre[0] * G[0] + 2.f * (pre[1] * G[1] + pre[2] * G[2] + pre[3] * G[3])) +
                      pre[1] * (pre[1] * G[4] + 2.f * (pre[2] * G[5] + pre[3] * G[6])) +
                      pre[2] * (pre[2] * G[7] + 2.f * pre[3] * G[8]) + pre[3] * pre[3] * G[9];
    const float rn = rsqrtf(fmaxf(ssx, 0.f) / hidden + norm_eps);
    float r[hc][8];
#pragma unroll
    for (int j = 0; j < hc; ++j) {
      const uint4 u = *reinterpret_cast<const uint4*>(&sm.a[tt][j * kHcSlice + dl]);
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 f = __bfloat1622float2(h[q]);
        r[j][2 * q] = f.x;
        r[j][2 * q + 1] = f.y;
      }
    }
    float o[8];
    float amax = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float x = bf16r(__fadd_rn(__fadd_rn(__fmul_rn(pre[0], r[0][e]), __fmul_rn(pre[1], r[1][e])),
                                      __fadd_rn(__fmul_rn(pre[2], r[2][e]), __fmul_rn(pre[3], r[3][e]))));
      o[e] = bf16r(x * rn * wv[e]);
      amax = fmaxf(amax, fabsf(o[e]));
    }
    if (valid) store8(x_out + (size_t)t * hidden + d, o);
    if (xq) {
      float a128 = amax;
#pragma unroll
      for (int o2 = 1; o2 < 16; o2 <<= 1) a128 = fmaxf(a128, __shfl_xor_sync(0xffffffff, a128, o2));
      const float sc = fmaxf(a128, 1e-12f) * (1.f / 448.f);
      if (kWarm || valid) {  // (warm instances: computed by every thread, the warming warp too)
        const float ys = __frcp_rn(sc);
        const __nv_fp8x4_e4m3 q0(make_float4(div_rn(o[0], sc, ys), div_rn(o[1], sc, ys),
                                             div_rn(o[2], sc, ys), div_rn(o[3], sc, ys)));
        const __nv_fp8x4_e4m3 q1(make_float4(div_rn(o[4], sc, ys), div_rn(o[5], sc, ys),
                                             div_rn(o[6], sc, ys), div_rn(o[7], sc, ys)));
        if (valid) {
          *reinterpret_cast<uint2*>(xq + (size_t)t * hidden + d) = make_uint2(q0.__x, q1.__x);
          if ((d & 127) == 0) xs[(size_t)t * (hidden / 128) + d / 128] = sc;
        }
      }
    }
    if (x4) {
      // NVFP4 of x (nvfp4_quant's quant16 arithmetic, as the cluster kernel)
      const float a16 = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
      if (kWarm || valid) {
        float sfv = q4_sfs * (a16 * rcp_approx(6.f));
        const __nv_fp8_e4m3 s8(sfv);
        sfv = static_cast<float>(s8);
        const float osc = a16 != 0.f ? rcp_approx(sfv * rcp_approx(q4_sfs)) : 0.f;
        float w[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) w[e] = o[e] * osc;
        const uint32_t q = hc_e2m1x8(w);
        if (valid) {
          *reinterpret_cast<uint32_t*>(x4 + (size_t)t * (hidden / 2) + d / 2) = q;
          if (!(v & 1)) x4s[nvfp4_sf_off(t, d / 16, hidden)] = s8.__x;
        }
      }
    }
  }
  // the last of the CTAs records this all-reduce call (the buffer rotation's state)
  if (arx && tid == 0 && ar_arrived == gridDim.y * kHcCluster - 1) {
    ar.state[0] = ar_epoch;
    ar.state[1] = (uint32_t)(T * hidden);
    ar.state[3] = 1u;
    ar.state[2] = 0u;
  }
}

// Steps below 32 tokens take the cluster kernel, which also does the pending tensor-parallel
// all-reduce (the tiled path takes a separate all-reduce and a second kernel)
constexpr int kMhcTileMinTokens = 32;

// tokens up to which the cluster boundary takes its fast form (one CTA per SM at most): 12 tokens
// (96 CTAs) still fit one wave; at 16 the 128 CTAs of 8-CTA clusters do not
constexpr int kMhcFastMaxTokens = 12;

// tokens up to which the boundary takes the tensor-core cluster kernel above the fast form (tiles of 4 tokens up to
// kMhcTc4MaxTokens: 15 clusters at one CTA per SM; else of 8)
constexpr int kMhcTcMaxTokens = 64, kMhcTc4MaxTokens = 60;
static bool mhc_tc(int T, int hidden) {
  return T > kMhcFastMaxTokens && T <= kMhcTcMaxTokens && hidden == kHcCluster * kHcSlice;
}

// The 16-CTA cluster form (mhc_fused_c16_kernel) of the fast boundaries of up to kMhcC16MaxTokens tokens with a fused
// all-reduce of partial sums (no MoE combine), for the boundaries in this mask: bit 1 the producer-pushed ones (after
// the attention's o_proj), bit 2 the self-pushing ones (after a dense MLP)
constexpr int kMhcC16MaxTokens = 4;
int mhc_c16_mask() { return 1; }
// (the 16-CTA form's trig, nonzero: the next kernel launches after the second cluster sum)
constexpr int kMhcC16Trig = 3;

// (the cluster kernel's gram, 0: the RMS of the layer input from the rounded x by a second cluster sum)
constexpr int kHcGram = 0;

// the boundaries of T tokens that take the code-warming warp's instances: the fast form's up to 8 tokens, the
// tensor-core kernel's 4-token tiles up to 16 tokens and its 8-token tiles
static bool mhc_warm(int T) {
  constexpr int kWarmFastMaxTokens = 8, kWarmTc4MaxTokens = 16;
  if (T <= kMhcFastMaxTokens) return T <= kWarmFastMaxTokens;
  if (T <= kMhcTc4MaxTokens) return T <= kWarmTc4MaxTokens;
  return true;
}

constexpr int kMoeK1 = 9;  // (the kMoe instances' pairs per token: top 8 + the shared expert)
// The kMoe instances' dynamic shared memory: at 80 registers (the generic form: 140) their CTAs would fit beside other
// kernels' CTAs (the MoE kernel's last ones, the next layer's GEMV); reserving 180 KB keeps them alone on their SMs as
// the generic form's registers do
constexpr int kMhcMoeSmem = std::max(kHcSmem, 180 * 1024);
// the fn weights [24 mixes x 4 streams][hidden] as the 5-D view of hc_tma_fn (one map per weight, cached)
static CUtensorMap mhc_fn_map(const bf16* fn, int hidden, int slice) {
  static PFN_cuTensorMapEncodeTiled_v12000 encode = [] {
    void* f = nullptr;
    cudaDriverEntryPointQueryResult q;
    CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &f, 12000, cudaEnableDefault, &q));
    CHECK(f != nullptr && q == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled not found");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(f);
  }();
  static std::mutex mu;
  static std::map<std::tuple<const void*, int, int>, CUtensorMap> maps;
  std::lock_guard<std::mutex> lk(mu);
  const auto key = std::make_tuple((const void*)fn, hidden, slice);
  const auto it = maps.find(key);
  if (it != maps.end()) return it->second;
  CUtensorMap m;
  cuuint64_t dims[5] = {64, 2, kMaxHc, (cuuint64_t)hidden / 128, kMixes};
  cuuint64_t strides[4] = {128, (cuuint64_t)hidden * sizeof(bf16), 256, (cuuint64_t)kMaxHc * hidden * sizeof(bf16)};
  cuuint32_t box[5] = {64, 2, kMaxHc, (cuuint32_t)slice / 128, kMixes};
  cuuint32_t es[5] = {1, 1, 1, 1, 1};
  const CUresult r = encode(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, const_cast<bf16*>(fn), dims, strides, box, es,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  CHECK(r == CUDA_SUCCESS, "mhc fn tensor map: cuTensorMapEncodeTiled failed: " + std::to_string((int)r));
  maps.emplace(key, m);
  return m;
}

// the early dependent launch (kEt 1 / 2 / 3, the kernel's comment above) of the next MoE-side kMp / kTm boundary call,
// set by the host per call (per host thread)
static thread_local int g_mhc_et = 0;
void mhc_set_et(int mode) { g_mhc_et = mode; }
// (the kEt instance the last mhc_fused call of this thread launched, 0: none -- the in_proj's staged forms follow it)
static thread_local int g_mhc_et_used = 0;
int mhc_et_used() { return g_mhc_et_used; }

bool mhc_fused_cluster(int T, int hidden) {
  return T > 0 && (T < kMhcTileMinTokens || mhc_tc(T, hidden)) && hidden == kHcCluster * kHcSlice;
}

// whether the boundary after a MoE of T tokens reads the MoE's outputs token-major (kTm: the kMoe kHm kFr instances,
// kMp where the code-warming warp runs) when the producer writes them so (moe_fp4's y_tm, MoE tensor parallelism: the
// host's other conditions); the conditions of those instances' choice in mhc_fused, the deferred Sinkhorn's input
// (ArFuse.sk & 2) aside (the host's)
bool mhc_moe_tm_ok(int T, int hidden, int tp, int K1) {
  return mhc_fused_cluster(T, hidden) && !mhc_tc(T, hidden) && T <= kMhcFastMaxTokens &&
         hidden == kHcCluster * kHcSlice && tp == 4 && K1 == kMoeK1;
}

// whether the boundary of T tokens takes a form with the deferred Sinkhorn (ArFuse.sk): the
// cluster kernel's one-CTA fast form, or the tensor-core kernel's tiles of 4 / 8 tokens
bool mhc_sk_ok(int T, int hidden) {
  if (!mhc_fused_cluster(T, hidden)) return false;
  if (mhc_tc(T, hidden)) return true;
  return T <= kMhcFastMaxTokens;
}

bool mhc_fused(bf16* streams, const bf16* x_prev, float* post, float* comb, bf16* x_out,
               fp8* xq, float* xs, const bf16* fn, const float* base, const float* scale,
               const bf16* norm_w, float* partial, int* counter, int T, int hidden, int hc,
               float rms_eps, float hc_eps, int iters, float post_mult, float norm_eps,
               cudaStream_t st, const ArFuse* ar, uint8_t* x4, uint8_t* x4s, float q4_gs, bool moe_tm,
               bool* rf_done) {
  g_mhc_et_used = 0;
  CHECK(hc == kMaxHc, "mhc kernels are specialized for hc_mult=4");
  CHECK(!moe_tm || (ar && ar->cm.y), "mhc_fused: token-major rows of a MoE combine");
  CHECK(!ar || (x_prev && mhc_fused_cluster(T, hidden)), "fused all-reduce: cluster path only");
  CHECK(!ar || !ar->cm.y || ar->cm.K1 <= kArMaxK1, "fused MoE combine: too many pairs per token");
  CHECK(!ar || ar->sk == 0 || mhc_sk_ok(T, hidden),
        "deferred Sinkhorn: a fused-all-reduce boundary of a form that has it (mhc_sk_ok)");
  if (T == 0) return false;
  if (T >= kMhcTileMinTokens && !mhc_tc(T, hidden) && hidden == 4096 && partial) {
    // tiled path: partial holds [T][hidden / 256][32] floats
    const int nslice = hidden / kMhDS;
    const int smem = sizeof(MhTileSmem);
    static bool configured[64] = {};
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (!configured[dev]) {
      CUDA_CHECK(cudaFuncSetAttribute(mhc_tile_mix_kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
      configured[dev] = true;
    }
    launch_pdl(mhc_tile_mix_kernel, dim3(nslice, cdiv(T, kMhTT)), dim3(kMhThreads), smem, st,
               streams, x_prev, post, comb, fn, partial, T, hidden);
    launch_pdl(mhc_tile_apply_kernel, dim3(T), dim3(kMhThreads + 32), 0, st, streams, post, comb,
               x_out, xq, xs, base, scale, norm_w, partial, nslice, hidden, rms_eps, hc_eps,
               iters, post_mult, norm_eps);
    return false;
  }
  if (T > 0 && hidden == kHcCluster * kHcSlice) {
    static bool cl_configured[64] = {};
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (!cl_configured[dev]) {
      CUDA_CHECK(cudaFuncSetAttribute(mhc_fused_cluster_kernel<true>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, kHcSmem));
      CUDA_CHECK(cudaFuncSetAttribute(mhc_fused_cluster_kernel<false>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, kHcSmem));
      CUDA_CHECK(cudaFuncSetAttribute(mhc_tc_cluster_kernel<4>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      (int)sizeof(TcSmem<4>)));
      CUDA_CHECK(cudaFuncSetAttribute(mhc_tc_cluster_kernel<8>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      (int)sizeof(TcSmem<8>)));
      // the deferred-Sinkhorn instances (ArFuse.sk 1 / 2 / 3)
      auto tc_smem = [](auto kern, int bytes) {
        CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
      };
      tc_smem(mhc_fused_cluster_kernel<true, 1, 1>, kHcSmem);
      tc_smem(mhc_tc_cluster_kernel<4, 1>, (int)sizeof(TcSmem<4>));
      tc_smem(mhc_tc_cluster_kernel<4, 2>, (int)sizeof(TcSmem<4>));
      tc_smem(mhc_tc_cluster_kernel<4, 3>, (int)sizeof(TcSmem<4>));
      tc_smem(mhc_tc_cluster_kernel<8, 1>, (int)sizeof(TcSmem<8>));
      tc_smem(mhc_tc_cluster_kernel<8, 2>, (int)sizeof(TcSmem<8>));
      tc_smem(mhc_tc_cluster_kernel<8, 3>, (int)sizeof(TcSmem<8>));
      // the code-warming warp's instances
      tc_smem(mhc_fused_cluster_kernel<true, 1, 0, true>, kHcSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 1, true>, kHcSmem);
      // the fn-by-TMA instances (kHcSmem >= 96 KB + the 1 KB alignment slack)
      static_assert(kHcSmem >= kHcFnTmaBytes + 1023, "kTma: dynamic shared memory");
      tc_smem(mhc_fused_cluster_kernel<true, 1, 0, false, true>, kHcSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 1, false, true>, kHcSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 0, true, true>, kHcSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 1, true, true>, kHcSmem);
      // the MoE-side instances (kMoe kHm kFr, SK 3 / 2; their launches reserve kMhcMoeSmem)
      tc_smem(mhc_fused_cluster_kernel<true, 1, 3, false, false, kMoeK1, true, true>, kMhcMoeSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 2, false, false, kMoeK1, true, true>, kMhcMoeSmem);
      // (kTm: the token-major MoE outputs)
      tc_smem(mhc_fused_cluster_kernel<true, 1, 3, false, false, kMoeK1, true, true, false, true>, kMhcMoeSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 2, false, false, kMoeK1, true, true, false, true>, kMhcMoeSmem);
      // (kMp + kXc: the code-warming warp's, the x path clearing the previous region, SK 2 / 3 x kTm x kEt)
      tc_smem(mhc_fused_cluster_kernel<true, 1, 2, true, false, kMoeK1, true, true, true, false, 0, true>, kMhcMoeSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 2, true, false, kMoeK1, true, true, true, false, 1, true>, kMhcMoeSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 2, true, false, kMoeK1, true, true, true, false, 2, true>, kMhcMoeSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 2, true, false, kMoeK1, true, true, true, false, 3, true>, kMhcMoeSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 2, true, false, kMoeK1, true, true, true, true, 0, true>, kMhcMoeSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 2, true, false, kMoeK1, true, true, true, true, 1, true>, kMhcMoeSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 2, true, false, kMoeK1, true, true, true, true, 2, true>, kMhcMoeSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 2, true, false, kMoeK1, true, true, true, true, 3, true>, kMhcMoeSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 3, true, false, kMoeK1, true, true, true, false, 0, true>, kMhcMoeSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 3, true, false, kMoeK1, true, true, true, false, 1, true>, kMhcMoeSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 3, true, false, kMoeK1, true, true, true, false, 2, true>, kMhcMoeSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 3, true, false, kMoeK1, true, true, true, false, 3, true>, kMhcMoeSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 3, true, false, kMoeK1, true, true, true, true, 0, true>, kMhcMoeSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 3, true, false, kMoeK1, true, true, true, true, 1, true>, kMhcMoeSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 3, true, false, kMoeK1, true, true, true, true, 2, true>, kMhcMoeSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 3, true, false, kMoeK1, true, true, true, true, 3, true>, kMhcMoeSmem);
      // the kHm instances of the other fast boundaries (SK 2 / 3)
      tc_smem(mhc_fused_cluster_kernel<true, 1, 2, false, false, 0, true>, kHcSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 2, true, false, 0, true>, kHcSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 2, false, true, 0, true>, kHcSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 2, true, true, 0, true>, kHcSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 3, false, false, 0, true>, kHcSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 3, true, false, 0, true>, kHcSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 3, false, true, 0, true>, kHcSmem);
      tc_smem(mhc_fused_cluster_kernel<true, 1, 3, true, true, 0, true>, kHcSmem);
      tc_smem(mhc_tc_cluster_kernel<4, 0, true>, (int)sizeof(TcSmem<4>));
      tc_smem(mhc_tc_cluster_kernel<4, 1, true>, (int)sizeof(TcSmem<4>));
      tc_smem(mhc_tc_cluster_kernel<4, 2, true>, (int)sizeof(TcSmem<4>));
      tc_smem(mhc_tc_cluster_kernel<4, 3, true>, (int)sizeof(TcSmem<4>));
      tc_smem(mhc_tc_cluster_kernel<8, 0, true>, (int)sizeof(TcSmem<8>));
      tc_smem(mhc_tc_cluster_kernel<8, 1, true>, (int)sizeof(TcSmem<8>));
      tc_smem(mhc_tc_cluster_kernel<8, 2, true>, (int)sizeof(TcSmem<8>));
      tc_smem(mhc_tc_cluster_kernel<8, 3, true>, (int)sizeof(TcSmem<8>));
      cl_configured[dev] = true;
    }
    ArFuse arv = ar ? *ar : ArFuse{};
    // (RF: an attention-side boundary's cm.w may hold the router's copies, ar_rf_x; only the c16 kRf launch keeps it)
    bf16* const rf_x = ar_rf_x(arv);
    if (!arv.cm.y) arv.cm.w = nullptr;
    const float sfs = x4 ? 1.f / q4_gs : 0.f;
    const bool warm = mhc_warm(T);  // (the code-warming warp's instances, kTcThreads threads)
#define MHC_TC_GO(TT, SK)                                                                         \
  do {                                                                                            \
    if (warm)                                                                                     \
      launch_pdl(mhc_tc_cluster_kernel<TT, SK, true>, dim3(kHcCluster, cdiv(T, TT)),              \
                 dim3(kTcThreads), sizeof(TcSmem<TT>), st, streams, x_prev, post, comb, x_out, xq, \
                 xs, fn, base, scale, norm_w, T, hidden, rms_eps, hc_eps, iters, post_mult,        \
                 norm_eps, arv, x4, x4s, sfs);                                                    \
    else                                                                                          \
      launch_pdl(mhc_tc_cluster_kernel<TT, SK>, dim3(kHcCluster, cdiv(T, TT)),                    \
                 dim3(kHcThreadsAll), sizeof(TcSmem<TT>), st, streams, x_prev, post, comb, x_out, \
                 xq, xs, fn, base, scale, norm_w, T, hidden, rms_eps, hc_eps, iters, post_mult,    \
                 norm_eps, arv, x4, x4s, sfs);                                                    \
  } while (0)
    // the 8-CTA fast boundaries with a fused all-reduce take the fn slice by one TMA tensor copy (kTma, hc_tma_fn)
    // instead of 16-byte cp.async (6144 per CTA), which sit in the SM's load path beside the o_proj CTAs the
    // attention-side boundary is resident with; not after a MoE (the self-pushed MoE combine: its push is not held by
    // the copies)
    const bool fn_tma = T <= kMhcFastMaxTokens && arv.state && !(arv.cm.y && !arv.pushed);
    const CUtensorMap tmf = fn_tma ? mhc_fn_map(fn, hidden, kHcSlice) : CUtensorMap{};
    // the fast boundary after a MoE (the self-pushed MoE combine of kMoeK1 pairs per token, tp 4, fn by cp.async, the
    // deferred Sinkhorn's consumer) takes the kMoe instances
    const bool moe_form = !fn_tma && arv.state && x_prev && arv.cm.y && !arv.pushed && arv.tp == 4 &&
                          arv.cm.K1 == kMoeK1 && (arv.sk & 2);
    // (token-major MoE outputs: only the kTm instances read them -- the producer chose them by mhc_moe_tm_ok)
    CHECK(!moe_tm || (moe_form && mhc_moe_tm_ok(T, hidden, arv.tp, arv.cm.K1) && (arv.sk & 2)),
          "mhc_fused: token-major MoE outputs (moe_tm) without the kTm boundary");
    // the fast form's generic instances (HM: kHm, the boundaries with the deferred Sinkhorn on both sides or the last
    // one of a step)
#define MHC_FAST_GO(SK, HM)                                                                                      \
  do {                                                                                                           \
    if (warm && fn_tma)                                                                                          \
      launch_pdl(mhc_fused_cluster_kernel<true, 1, SK, true, true, 0, HM>, dim3(kHcCluster, T), dim3(kTcThreads), \
                 kHcSmem, st, streams, x_prev, post, comb, x_out, xq, xs, fn, base, scale, norm_w, hidden,        \
                 rms_eps, hc_eps, iters, post_mult, norm_eps, arv, kHcGram, x4, x4s, sfs, tmf);                  \
    else if (warm)                                                                                               \
      launch_pdl(mhc_fused_cluster_kernel<true, 1, SK, true, false, 0, HM>, dim3(kHcCluster, T),                 \
                 dim3(kTcThreads), kHcSmem, st, streams, x_prev, post, comb, x_out, xq, xs, fn, base, scale,      \
                 norm_w, hidden, rms_eps, hc_eps, iters, post_mult, norm_eps, arv, kHcGram, x4, x4s, sfs, tmf);  \
    else if (fn_tma)                                                                                             \
      launch_pdl(mhc_fused_cluster_kernel<true, 1, SK, false, true, 0, HM>, dim3(kHcCluster, T),                 \
                 dim3(kHcThreadsAll), kHcSmem, st, streams, x_prev, post, comb, x_out, xq, xs, fn, base, scale,   \
                 norm_w, hidden, rms_eps, hc_eps, iters, post_mult, norm_eps, arv, kHcGram, x4, x4s, sfs, tmf);  \
    else                                                                                                         \
      launch_pdl(mhc_fused_cluster_kernel<true, 1, SK, false, false, 0, HM>, dim3(kHcCluster, T),                \
                 dim3(kHcThreadsAll), kHcSmem, st, streams, x_prev, post, comb, x_out, xq, xs, fn, base, scale,   \
                 norm_w, hidden, rms_eps, hc_eps, iters, post_mult, norm_eps, arv, kHcGram, x4, x4s, sfs, tmf);  \
  } while (0)
    // the MoE side's kMoe kHm kFr instances (SK 2 / 3): with the code-warming warp, kMp + kXc and the early trigger
    // the host set (kEt, mhc_set_et); TM: kTm
#define MHC_MOE_GO(SK, WARM, TM, ET)                                                                              \
  launch_pdl(mhc_fused_cluster_kernel<true, 1, SK, WARM, false, kMoeK1, true, true, WARM, TM, ET, WARM>,          \
             dim3(kHcCluster, T), dim3(WARM ? kTcThreads : kHcThreadsAll), kMhcMoeSmem, st, streams, x_prev, post, \
             comb, x_out, xq, xs, fn, base, scale, norm_w, hidden, rms_eps, hc_eps, iters, post_mult, norm_eps,   \
             arv, kHcGram, x4, x4s, sfs, tmf)
#define MHC_MOE_WARM_GO(SK, TM)                                                                                   \
  do {                                                                                                            \
    if (g_mhc_et == 3) {                                                                                          \
      g_mhc_et_used = 3;                                                                                          \
      MHC_MOE_GO(SK, true, TM, 3);                                                                                \
    } else if (g_mhc_et == 2) {                                                                                   \
      g_mhc_et_used = 2;                                                                                          \
      MHC_MOE_GO(SK, true, TM, 2);                                                                                \
    } else if (g_mhc_et) {                                                                                        \
      g_mhc_et_used = 1;                                                                                          \
      MHC_MOE_GO(SK, true, TM, 1);                                                                                \
    } else {                                                                                                      \
      MHC_MOE_GO(SK, true, TM, 0);                                                                                \
    }                                                                                                             \
  } while (0)
    // the kHm boundaries (SK 2 / 3): the kMoe instances after a MoE, else the generic ones
#define MHC_FAST_HM_GO(SK)                                                                                        \
  do {                                                                                                            \
    if (moe_tm && warm)                                                                                           \
      MHC_MOE_WARM_GO(SK, true);                                                                                  \
    else if (moe_tm)                                                                                              \
      MHC_MOE_GO(SK, false, true, 0);                                                                             \
    else if (moe_form && warm)                                                                                    \
      MHC_MOE_WARM_GO(SK, false);                                                                                 \
    else if (moe_form)                                                                                            \
      MHC_MOE_GO(SK, false, false, 0);                                                                            \
    else                                                                                                          \
      MHC_FAST_GO(SK, true);                                                                                      \
  } while (0)
    if (ar && arv.state && x_prev && !arv.cm.y && arv.tp == 4 && hidden == kC16 * kC16Slice &&
        T <= kMhcC16MaxTokens && T <= kMhcFastMaxTokens && (mhc_c16_mask() & (arv.pushed ? 1 : 2))) {
      static bool c16_configured[64] = {};
      if (!c16_configured[dev]) {
        auto c16_attr = [](auto kern) {
          CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, kC16Smem));
          CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        };
        static_assert(kC16Smem >= kMixes * kMaxHc * kC16Slice * 2 + 1023, "c16 kTma: dynamic shared memory");
        c16_attr(mhc_fused_c16_kernel<0, false, true>);  // (the fn slice by TMA: kTma)
        c16_attr(mhc_fused_c16_kernel<2, false, true>);
        c16_attr(mhc_fused_c16_kernel<0, true, true>);
        c16_attr(mhc_fused_c16_kernel<2, true, true>);
        c16_attr(mhc_fused_c16_kernel<3, false, true, true>);  // (the kHm instances, SK 3)
        c16_attr(mhc_fused_c16_kernel<3, true, true, true>);
        c16_attr(mhc_fused_c16_kernel<3, false, true, true, true>);  // (kFr)
        c16_attr(mhc_fused_c16_kernel<3, true, true, true, true>);
        c16_attr(mhc_fused_c16_kernel<3, true, true, true, true, true>);  // (kPre)
        c16_attr(mhc_fused_c16_kernel<3, true, true, true, true, true, true>);  // (kPre kCl)
        c16_attr(mhc_fused_c16_kernel<3, true, true, true, true, true, true, true>);  // (kPre kCl kRf)
        // (SK 1 kHm: the first boundary of a step)
        c16_attr(mhc_fused_c16_kernel<1, true, true, true, true, true>);
        c16_attr(mhc_fused_c16_kernel<1, true, true, true, true>);
        c16_attr(mhc_fused_c16_kernel<1, false, true, true, true>);
        c16_attr(mhc_fused_c16_kernel<1, true, true, true>);
        c16_attr(mhc_fused_c16_kernel<1, false, true, true>);
        c16_configured[dev] = true;
      }
      const CUtensorMap tmc = mhc_fn_map(fn, hidden, kC16Slice);
      // (the fn slice by TMA; HM: the kHm instances with the fn rows in registers (kFr), producer-pushed hoisted calls
      // with the call state first (kPre) and, for the deferred Sinkhorn's input, its comb logits at the start (kCl))
#define MHC_C16_GO(SK, HM)                                                                                          \
  do {                                                                                                            \
    if (HM && warm && arv.pushed >= 2 && (SK & 2))                                                                \
      launch_pdl_cluster(mhc_fused_c16_kernel<SK, true, true, HM, HM, HM, HM && (SK & 2) != 0>, dim3(kC16, T),     \
                         dim3(kC16Threads), kC16Smem, st, kC16, streams, x_prev, post, comb, x_out, xq, xs, fn, base, \
                         scale, norm_w, hidden, rms_eps, hc_eps, iters, post_mult, norm_eps, arv, x4, x4s, sfs,        \
                         kMhcC16Trig, tmc);                                                                        \
    else if (HM && warm && arv.pushed >= 2)                                                                       \
      launch_pdl_cluster(mhc_fused_c16_kernel<SK, true, true, HM, HM, HM>, dim3(kC16, T), dim3(kC16Threads),       \
                         kC16Smem, st, kC16, streams, x_prev, post, comb, x_out, xq, xs, fn, base, scale, norm_w,  \
                         hidden, rms_eps, hc_eps, iters, post_mult, norm_eps, arv, x4, x4s, sfs, kMhcC16Trig, tmc); \
    else if (HM && warm)                                                                                          \
      launch_pdl_cluster(mhc_fused_c16_kernel<SK, true, true, HM, HM>, dim3(kC16, T), dim3(kC16Threads), kC16Smem, \
                         st, kC16, streams, x_prev, post, comb, x_out, xq, xs, fn, base, scale, norm_w, hidden,    \
                         rms_eps, hc_eps, iters, post_mult, norm_eps, arv, x4, x4s, sfs, kMhcC16Trig, tmc);       \
    else if (HM)                                                                                                  \
      launch_pdl_cluster(mhc_fused_c16_kernel<SK, false, true, HM, HM>, dim3(kC16, T), dim3(kC16All), kC16Smem,    \
                         st, kC16, streams, x_prev, post, comb, x_out, xq, xs, fn, base, scale, norm_w, hidden,    \
                         rms_eps, hc_eps, iters, post_mult, norm_eps, arv, x4, x4s, sfs, kMhcC16Trig, tmc);       \
    else if (warm)                                                                                                \
      launch_pdl_cluster(mhc_fused_c16_kernel<SK, true, true, HM>, dim3(kC16, T), dim3(kC16Threads), kC16Smem, st, \
                         kC16, streams, x_prev, post, comb, x_out, xq, xs, fn, base, scale, norm_w, hidden,        \
                         rms_eps, hc_eps, iters, post_mult, norm_eps, arv, x4, x4s, sfs, kMhcC16Trig, tmc);       \
    else                                                                                                          \
      launch_pdl_cluster(mhc_fused_c16_kernel<SK, false, true, HM>, dim3(kC16, T), dim3(kC16All), kC16Smem, st,    \
                         kC16, streams, x_prev, post, comb, x_out, xq, xs, fn, base, scale, norm_w, hidden,        \
                         rms_eps, hc_eps, iters, post_mult, norm_eps, arv, x4, x4s, sfs, kMhcC16Trig, tmc);       \
  } while (0)
      // (RF: the kPre kCl instance of SK 3 with the router's copies, the 4-token MoE layers' attention-side boundary;
      // the caller launches the RF router only when rf_done says so)
      if (rf_x && T == kRouterRfT && arv.sk == 3 && warm && arv.pushed >= 2) {
        ArFuse arf = arv;
        arf.cm.w = reinterpret_cast<const float*>(rf_x);
        launch_pdl_cluster(mhc_fused_c16_kernel<3, true, true, true, true, true, true, true>, dim3(kC16, T),
                           dim3(kC16Threads), kC16Smem, st, kC16, streams, x_prev, post, comb, x_out, xq, xs, fn, base,
                           scale, norm_w, hidden, rms_eps, hc_eps, iters, post_mult, norm_eps, arf, x4, x4s, sfs,
                           kMhcC16Trig, tmc);
        if (rf_done) *rf_done = true;
        return x4 != nullptr;
      }
      switch (arv.sk) {
        case 1: MHC_C16_GO(1, true); break;  // (the first boundary of a step)
        case 2: MHC_C16_GO(2, false); break;
        case 3: MHC_C16_GO(3, true); break;
        default: MHC_C16_GO(0, false);
      }
#undef MHC_C16_GO
      return x4 != nullptr;
    }
    if (mhc_tc(T, hidden)) {
      if (T <= kMhcTc4MaxTokens)
        switch (arv.sk) {
          case 1: MHC_TC_GO(4, 1); break;
          case 2: MHC_TC_GO(4, 2); break;
          case 3: MHC_TC_GO(4, 3); break;
          default: MHC_TC_GO(4, 0);
        }
      else
        switch (arv.sk) {
          case 1: MHC_TC_GO(8, 1); break;
          case 2: MHC_TC_GO(8, 2); break;
          case 3: MHC_TC_GO(8, 3); break;
          default: MHC_TC_GO(8, 0);
        }
      return x4 != nullptr;
    }
    if (T <= kMhcFastMaxTokens)
      switch (arv.sk) {
        case 1: MHC_FAST_GO(1, false); break;
        case 2: MHC_FAST_HM_GO(2); break;  // (the last boundary of a step)
        case 3: MHC_FAST_HM_GO(3); break;
        default: MHC_FAST_GO(0, false);
      }
    else
      launch_pdl(mhc_fused_cluster_kernel<false>, dim3(kHcCluster, T), dim3(kHcThreadsAll),
                 kHcSmem, st, streams, x_prev, post, comb, x_out, xq, xs, fn, base, scale, norm_w,
                 hidden, rms_eps, hc_eps, iters, post_mult, norm_eps, arv, kHcGram, x4, x4s,
                 sfs, CUtensorMap{});
#undef MHC_TC_GO
#undef MHC_FAST_GO
#undef MHC_MOE_GO
#undef MHC_MOE_WARM_GO
#undef MHC_FAST_HM_GO
    return x4 != nullptr;
  }
  const int G = mhc_fused_groups(T);
  CHECK(hidden % (G * 256) == 0 && hidden / G <= kMaxVec * 256 &&
            hidden % (8 * kFusedThreads) == 0,
        "hidden not compatible with mhc groups");
  launch_pdl(mhc_fused_split_kernel, dim3(G, T), dim3(kFusedThreads), hidden * sizeof(float), st,
             streams, x_prev, post, comb, x_out, xq, xs, fn, base, scale, norm_w, partial,
             counter, hidden, rms_eps, hc_eps, iters, post_mult, norm_eps);
  return false;
}

__global__ void mhc_post_kernel(bf16* __restrict__ streams, const bf16* __restrict__ x,
                                const float* __restrict__ post, const float* __restrict__ comb,
                                int hidden) {
  pdl_entry();
  constexpr int hc = kMaxHc;
  const int t = blockIdx.y;
  const int d8 = blockIdx.x * blockDim.x + threadIdx.x;
  if (d8 * 8 >= hidden) return;
  float c[hc][hc], p[hc];
#pragma unroll
  for (int i = 0; i < hc; ++i) {
    p[i] = post[(size_t)t * hc + i];
#pragma unroll
    for (int j = 0; j < hc; ++j) c[i][j] = comb[(size_t)t * hc * hc + i * hc + j];
  }
  bf16* s = streams + (size_t)t * hc * hidden + d8 * 8;
  float r[hc][8], xv[8];
#pragma unroll
  for (int i = 0; i < hc; ++i) load8(s + (size_t)i * hidden, r[i]);
  load8(x + (size_t)t * hidden + d8 * 8, xv);
#pragma unroll
  for (int j = 0; j < hc; ++j) {
    float o[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      float acc = 0.f;
#pragma unroll
      for (int i = 0; i < hc; ++i) acc += c[i][j] * r[i][e];
      o[e] = acc + p[j] * xv[e];
    }
    store8(s + (size_t)j * hidden, o);
  }
}

void mhc_post(bf16* streams, const bf16* x, const float* post, const float* comb, int T,
              int hidden, int hc, cudaStream_t st) {
  CHECK(hc == kMaxHc, "mhc kernels are specialized for hc_mult=4");
  if (T == 0) return;
  dim3 grid(cdiv(hidden / 8, 128), T);
  launch_pdl(mhc_post_kernel, dim3(grid), dim3(128), 0, st, streams, x, post, comb, hidden);
}

}  // namespace k
}  // namespace glm
