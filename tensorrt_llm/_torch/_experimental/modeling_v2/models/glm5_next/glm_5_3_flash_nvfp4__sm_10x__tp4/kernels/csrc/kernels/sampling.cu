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
// Vocab-parallel argmax keys (greedy, or temperature sampling by Gumbel-max) and the MTP draft acceptance.
#include "pdl.cuh"
#include <cooperative_groups.h>

#include "reduce.cuh"
#include "sampling.h"
#include "sampling_dev.cuh"

namespace glm {
namespace k {

namespace {
struct ArgMax {
  float v;
  int i;
};

__device__ __forceinline__ ArgMax better(ArgMax a, ArgMax b) {
  return (b.v > a.v || (b.v == a.v && b.i < a.i)) ? b : a;
}

__device__ ArgMax block_argmax(ArgMax x, ArgMax* sh) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    ArgMax y{__shfl_xor_sync(0xffffffff, x.v, o), __shfl_xor_sync(0xffffffff, x.i, o)};
    x = better(x, y);
  }
  __syncthreads();
  if (lane == 0) sh[wid] = x;
  __syncthreads();
  x = lane < (blockDim.x >> 5) ? sh[lane] : ArgMax{-INFINITY, 0x7fffffff};
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    ArgMax y{__shfl_xor_sync(0xffffffff, x.v, o), __shfl_xor_sync(0xffffffff, x.i, o)};
    x = better(x, y);
  }
  return x;
}
}  // namespace

// ------------------------------------------------------------------ vocab-parallel argmax
// key = orderable(max value) << 32 | ~global index: a max-reduction of keys over the ranks gives
// the global argmax with the lowest index among ties. A cluster of CL CTAs per row, each
// over a contiguous range of 4-element groups; partials combined through distributed shared
// memory (the order is irrelevant: (value desc, index asc) is a total order).
// Sampling rows (params, temperature > 0 and top_k != 1): value = logit / T + Gumbel noise, the
// noise of global vocab index i drawn from (seed, step, i) by a counter-based Philox-4x32-10 --
// every rank perturbs its shard identically, so the max over the ranks samples
// softmax(logits / T) exactly (Gumbel-max) with one pass and no logits all-gather.

template <bool SAMPLE, int CL>
__global__ void __launch_bounds__(1024) argmax_key_kernel(unsigned long long* __restrict__ keys,
                                                          const float* __restrict__ logits,
                                                          int Vl, int vocab_offset,
                                                          const SampleParams* __restrict__ params) {
  pdl_entry();
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  __shared__ ArgMax sh[32];
  __shared__ ArgMax part;
  const int r = blockIdx.x / CL, c = (int)cluster.block_rank(), tid = threadIdx.x;
  const float4* row = reinterpret_cast<const float4*>(logits + (size_t)r * Vl);
  bool noise = false;
  float invT = 1.f;
  uint2 key = make_uint2(0, 0);
  uint32_t st_lo = 0, st_hi = 0;
  if (SAMPLE) {
    const SampleParams p = params[r];
    noise = p.temperature > 0.f && p.top_k != 1;
    invT = noise ? 1.f / p.temperature : 1.f;
    key = make_uint2((uint32_t)p.seed, (uint32_t)(p.seed >> 32));
    st_lo = (uint32_t)p.step;
    st_hi = (uint32_t)(p.step >> 32);
  }
  const int ng = Vl / 4, per = (ng + CL - 1) / CL;
  const int g0 = c * per, g1 = min(ng, g0 + per);
  ArgMax am{-INFINITY, 0x7fffffff};
  for (int g = g0 + tid; g < g1; g += blockDim.x) {
    float4 v = row[g];
    if (SAMPLE && noise) {
      const uint4 rn = philox4x32_10(gumbel_counter((uint32_t)(vocab_offset / 4 + g), st_lo, st_hi), key);
      v.x = fmaf(v.x, invT, gumbel_of(rn.x));
      v.y = fmaf(v.y, invT, gumbel_of(rn.y));
      v.z = fmaf(v.z, invT, gumbel_of(rn.z));
      v.w = fmaf(v.w, invT, gumbel_of(rn.w));
    }
    am = better(am, ArgMax{v.x, 4 * g});
    am = better(am, ArgMax{v.y, 4 * g + 1});
    am = better(am, ArgMax{v.z, 4 * g + 2});
    am = better(am, ArgMax{v.w, 4 * g + 3});
  }
  am = block_argmax(am, sh);
  if (tid == 0) part = am;
  cluster.sync();
  if (c == 0 && tid == 0) {
    ArgMax best = part;
    for (int q = 1; q < CL; ++q) best = better(best, *cluster.map_shared_rank(&part, q));
    // (a shard whose every value is NaN keeps the initial index 0x7fffffff, and vocab_offset + it would overflow into a
    // key whose token lies past the vocabulary, which MTP pass 1 / the next draft pass embed. Such a shard's key is its
    // first index at -inf: another shard's finite argmax wins, an all-NaN row gives token 0, as k::head_keys gives it)
    keys[r] = argmax_key_of(best.v, vocab_offset + (best.i < Vl ? best.i : 0));
  }
  cluster.sync();  // the other CTAs' partials are read
}

// the MTP input row (mtp_eh_norm's arithmetic and thread mapping: 16 columns per thread, blockDim
// = H / 16): [rmsnorm(e) * enorm, rmsnorm(h) * hnorm]
__device__ __forceinline__ void mtp_eh_row(bf16* out, const bf16* e_row, const bf16* h_row,
                                           const MtpEhArgs& eh, int H, float* sh) {
  const int c0 = threadIdx.x * 16;
  float e[16], h[16], w[16];
  load8(e_row + c0, e);
  load8(e_row + c0 + 8, e + 8);
  load8(h_row + c0, h);
  load8(h_row + c0 + 8, h + 8);
  float se = 0.f, shh = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    se += e[i] * e[i];
    shh += h[i] * h[i];
  }
  se = block_sum(se, sh);
  shh = block_sum(shh, sh);
  const float re = rsqrtf(se / H + eh.eps), rh = rsqrtf(shh / H + eh.eps);
  load8(eh.enorm + c0, w);
  load8(eh.enorm + c0 + 8, w + 8);
#pragma unroll
  for (int i = 0; i < 16; ++i) e[i] = e[i] * re * w[i];
  store8(out + c0, e);
  store8(out + c0 + 8, e + 8);
  load8(eh.hnorm + c0, w);
  load8(eh.hnorm + c0 + 8, w + 8);
#pragma unroll
  for (int i = 0; i < 16; ++i) h[i] = h[i] * rh * w[i];
  store8(out + H + c0, h);
  store8(out + H + c0 + 8, h + 8);
}

__device__ __forceinline__ void copy_row(bf16* dst, const bf16* src, int n) {
  for (int i = threadIdx.x * 8; i < n; i += blockDim.x * 8)
    *reinterpret_cast<uint4*>(dst + i) = *reinterpret_cast<const uint4*>(src + i);
}

__global__ void mtp_accept_kernel(int* __restrict__ accepted, int* __restrict__ drafts,
                                  int ndraft, float force_accept, int* __restrict__ next_ids,
                                  int* __restrict__ next_pos, int* __restrict__ next_slot,
                                  bf16* __restrict__ next_hidden,
                                  const unsigned long long* __restrict__ keys,
                                  const bf16* __restrict__ g, const int* __restrict__ row0,
                                  const int* __restrict__ nrows, const int* __restrict__ sampled,
                                  const int* __restrict__ ids, const int* __restrict__ sample_idx,
                                  const int* __restrict__ positions,
                                  const int* __restrict__ token_slot, int hidden,
                                  const int* __restrict__ sel_src, int* __restrict__ sel_dst,
                                  int* __restrict__ vis_dst, int sel_n, int kpool,
                                  const MtpEhArgs eh, int compact) {
  pdl_entry();
  __shared__ float sh_eh[32];
  const int s = blockIdx.x;
  const int r0 = row0[s];
  const int a = mtp_accept_count(force_accept, r0, nrows[s], sampled, ids, sample_idx, positions,
                                 token_slot);
  const int r = r0 + a;
  // (compact: pass 1 ran its head at each sequence's last accepted row only, row s of keys / g)
  const int kr = compact ? s : r;
  const int d = argmax_key_token(keys[kr]);
  if (threadIdx.x == 0) {
    const int t = sample_idx[r];
    accepted[s] = a;
    drafts[(size_t)s * ndraft] = d;
    next_ids[s] = d;
    next_pos[s] = positions[t] + 1;
    next_slot[s] = token_slot[t];
  }
  if (next_hidden) copy_row(next_hidden + (size_t)s * hidden, g + (size_t)kr * hidden, hidden);
  if (eh.out)  // (the next position is >= 1: mtp_eh_norm reads the embedding)
    mtp_eh_row(eh.out + (size_t)s * 2 * hidden, eh.embed + (size_t)d * hidden,
               g + (size_t)kr * hidden, eh, hidden, sh_eh);
  if (sel_dst) {  // the last accepted token's selection, for the chained passes
    const int t = sample_idx[r];
    for (int i = threadIdx.x; i < sel_n; i += blockDim.x)
      sel_dst[(size_t)s * sel_n + i] = sel_src[(size_t)t * sel_n + i];
    if (threadIdx.x == 0) vis_dst[s] = (positions[t] + 1) / kpool;
  }
}

void mtp_accept(int* accepted, int* drafts, int ndraft, int* next_ids, int* next_pos,
                int* next_slot, bf16* next_hidden, const unsigned long long* keys, const bf16* g,
                const int* row0, const int* nrows, const int* sampled, const int* ids,
                const int* sample_idx, const int* positions, const int* token_slot, int B,
                int hidden, cudaStream_t st, float force_accept, const int* sel_src,
                int* sel_dst, int* vis_dst, int sel_n, int kpool, const MtpEhArgs* eh,
                bool compact) {
  if (B == 0) return;
  const MtpEhArgs e = eh ? *eh : MtpEhArgs{};
  CHECK(!e.out || (hidden % 512 == 0 && hidden / 16 <= 1024), "mtp_accept: eh hidden size");
  launch_pdl(mtp_accept_kernel, dim3(B), dim3(e.out ? hidden / 16 : 128), 0, st, accepted, drafts,
             ndraft, force_accept, next_ids, next_pos, next_slot, next_hidden, keys, g, row0, nrows,
             sampled, ids, sample_idx, positions, token_slot, hidden, sel_src, sel_dst, vis_dst,
             sel_n, kpool, e, compact ? 1 : 0);
}

// One CTA per sequence s: its last accepted row t (mtp_accept_count) and rows t of x / h (and of the
// FP8 input xq / xs) copied to rows s of xp / hp (xqp / xsp). The rows are the previous kernel's
// outputs: L2 loads after the wait.
__global__ void mtp_pick_kernel(bf16* __restrict__ xp, bf16* __restrict__ hp, fp8* __restrict__ xqp,
                                float* __restrict__ xsp, int* __restrict__ slotp, const bf16* x,
                                const bf16* h, const fp8* xq, const float* xs, float force_accept,
                                const int* __restrict__ row0, const int* __restrict__ nrows,
                                const int* __restrict__ sampled, const int* __restrict__ ids,
                                const int* __restrict__ sample_idx,
                                const int* __restrict__ positions,
                                const int* __restrict__ token_slot, int hidden) {
  pdl_entry();
  const int s = blockIdx.x;
  const int r0 = row0[s];
  const int a = mtp_accept_count(force_accept, r0, nrows[s], sampled, ids, sample_idx, positions,
                                 token_slot);
  const size_t t = (size_t)sample_idx[r0 + a];
  if (slotp && threadIdx.x == 0) slotp[s] = token_slot[t];
  const int n16 = hidden / 8;  // 16-byte chunks of a bf16 row
  const uint4* xs4 = reinterpret_cast<const uint4*>(x + t * hidden);
  const uint4* hs4 = reinterpret_cast<const uint4*>(h + t * hidden);
  uint4* xd4 = reinterpret_cast<uint4*>(xp + (size_t)s * hidden);
  uint4* hd4 = reinterpret_cast<uint4*>(hp + (size_t)s * hidden);
  for (int i = threadIdx.x; i < n16; i += blockDim.x) {
    const uint4 vx = __ldcg(xs4 + i), vh = __ldcg(hs4 + i);
    xd4[i] = vx;
    hd4[i] = vh;
  }
  if (xqp) {
    const uint4* q4 = reinterpret_cast<const uint4*>(xq + t * hidden);
    uint4* qd4 = reinterpret_cast<uint4*>(xqp + (size_t)s * hidden);
    for (int i = threadIdx.x; i < hidden / 16; i += blockDim.x) qd4[i] = __ldcg(q4 + i);
    for (int i = threadIdx.x; i < hidden / 128; i += blockDim.x)
      xsp[(size_t)s * (hidden / 128) + i] = __ldcg(xs + t * (hidden / 128) + i);
  }
}

void mtp_pick(bf16* xp, bf16* hp, fp8* xqp, float* xsp, int* slotp, const bf16* x, const bf16* h,
              const fp8* xq, const float* xs, float force_accept, const int* row0,
              const int* nrows, const int* sampled, const int* ids, const int* sample_idx,
              const int* positions, const int* token_slot, int B, int hidden, cudaStream_t st) {
  if (B == 0) return;
  CHECK(hidden % 128 == 0, "mtp_pick: hidden size");
  launch_pdl(mtp_pick_kernel, dim3(B), dim3(256), 0, st, xp, hp, xqp, xsp, slotp, x, h, xq, xs,
             force_accept, row0, nrows, sampled, ids, sample_idx, positions, token_slot, hidden);
}

__global__ void mtp_next_kernel(int* __restrict__ drafts, int ndraft, int j,
                                int* __restrict__ next_ids, int* __restrict__ next_pos,
                                bf16* __restrict__ next_hidden,
                                const unsigned long long* __restrict__ keys,
                                const bf16* __restrict__ g, int hidden, const MtpEhArgs eh) {
  pdl_entry();
  __shared__ float sh_eh[32];
  const int s = blockIdx.x;
  const int d = argmax_key_token(keys[s]);
  if (threadIdx.x == 0) {
    drafts[(size_t)s * ndraft + j - 1] = d;
    next_ids[s] = d;
    next_pos[s] += 1;
  }
  if (next_hidden) copy_row(next_hidden + (size_t)s * hidden, g + (size_t)s * hidden, hidden);
  if (eh.out)  // (the next position is >= 1: mtp_eh_norm reads the embedding)
    mtp_eh_row(eh.out + (size_t)s * 2 * hidden, eh.embed + (size_t)d * hidden,
               g + (size_t)s * hidden, eh, hidden, sh_eh);
}

void mtp_next(int* drafts, int ndraft, int j, int* next_ids, int* next_pos, bf16* next_hidden,
              const unsigned long long* keys, const bf16* g, int B, int hidden, cudaStream_t st,
              const MtpEhArgs* eh) {
  if (B == 0) return;
  const MtpEhArgs e = eh ? *eh : MtpEhArgs{};
  CHECK(!e.out || (hidden % 512 == 0 && hidden / 16 <= 1024), "mtp_next: eh hidden size");
  launch_pdl(mtp_next_kernel, dim3(B), dim3(e.out ? hidden / 16 : 128), 0, st, drafts, ndraft, j,
             next_ids, next_pos, next_hidden, keys, g, hidden, e);
}

__global__ void keys_to_tokens_kernel(int* __restrict__ tokens,
                                      const unsigned long long* __restrict__ keys, int n) {
  pdl_entry();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    tokens[i] = argmax_key_token(keys[i]);
}

void keys_to_tokens(int* tokens, const unsigned long long* keys, int n, cudaStream_t st) {
  if (n == 0) return;
  launch_pdl(keys_to_tokens_kernel, dim3(cdiv(n, 256)), dim3(256), 0, st, tokens, keys, n);
}

template <int CL>
static void launch_ak(unsigned long long* keys, const float* logits, int rows, int shard_vocab, int vocab_offset,
                      cudaStream_t st, const SampleParams* params) {
  const dim3 grid(rows * CL);
  if (params)
    launch_pdl_cluster(argmax_key_kernel<true, CL>, grid, dim3(1024), 0, st, CL, keys, logits, shard_vocab,
                       vocab_offset, params);
  else
    launch_pdl_cluster(argmax_key_kernel<false, CL>, grid, dim3(1024), 0, st, CL, keys, logits, shard_vocab,
                       vocab_offset, params);
}

void argmax_keys(unsigned long long* keys, const float* logits, int rows, int shard_vocab,
                 int vocab_offset, cudaStream_t st, const SampleParams* params) {
  if (rows == 0) return;
  CHECK(shard_vocab % 4 == 0 && vocab_offset % 4 == 0, "argmax_keys: vocab shards of 4k tokens");
  // a cluster of CL CTAs per row: 8 for few rows (a row over 8 SMs), fewer from 16 rows (one wave of fatter CTAs:
  // 64 rows on 8-CTA clusters would take 512 CTAs of 1024 threads)
  if (rows >= 32)
    launch_ak<2>(keys, logits, rows, shard_vocab, vocab_offset, st, params);
  else if (rows >= 16)
    launch_ak<4>(keys, logits, rows, shard_vocab, vocab_offset, st, params);
  else
    launch_ak<8>(keys, logits, rows, shard_vocab, vocab_offset, st, params);
}

}  // namespace k
}  // namespace glm
