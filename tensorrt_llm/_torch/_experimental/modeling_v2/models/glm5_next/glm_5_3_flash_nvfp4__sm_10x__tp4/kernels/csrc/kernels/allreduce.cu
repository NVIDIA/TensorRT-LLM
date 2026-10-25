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
// One-shot all-reduce over NVLink peer memory for small/medium messages.
//
// Each block b owns a contiguous chunk. It pushes the local chunk into every rank's receive
// buffer (slot = source rank, half = epoch parity), raises a per-block epoch flag on every rank,
// waits for the flags of all ranks for its chunk, then sums the received chunks in rank order
// (identical results on every rank). Per-block epoch counters make the kernel stateless on the
// host, so it is CUDA-graph capturable. Two parities keep a fast rank from overwriting a buffer
// a slower rank is still reading.
#include "pdl.cuh"
#include "allreduce.h"
#include "lamport.cuh"
#include "reduce.cuh"
#include "sampling_dev.cuh"

namespace glm {
namespace k {

// Receive buffer per rank: [3 Lamport buffers][2 flag-protocol parities] x [tp][max_elems].
constexpr size_t kFlagBase = 3;

__device__ __forceinline__ void st_flag_release(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t ld_flag_acquire(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}

template <int TP>
__global__ void __launch_bounds__(512) allreduce_kernel(bf16* __restrict__ data, int n,
                                                        const ArPeers peers, uint32_t* epochs,
                                                        int rank, size_t max_elems,
                                                        int max_blocks) {
  pdl_entry();
  const int b = blockIdx.x, tid = threadIdx.x;
  const int chunk = (n / 8 + gridDim.x - 1) / gridDim.x * 8;  // elements, multiple of 8
  const int e0 = b * chunk, e1 = min(n, e0 + chunk);
  __shared__ uint32_t s_epoch;
  if (tid == 0) s_epoch = epochs[b] + 1;
  __syncthreads();
  const uint32_t epoch = s_epoch;
  const int par = epoch & 1;
  // 1) push the local chunk to every rank
  for (int i = e0 + tid * 8; i < e1; i += blockDim.x * 8) {
    const uint4 v = *reinterpret_cast<const uint4*>(data + i);
#pragma unroll
    for (int r = 0; r < TP; ++r)
      *reinterpret_cast<uint4*>(peers.buf[r] + kFlagBase * TP * max_elems +
                                ((size_t)par * TP + rank) * max_elems + i) = v;
  }
  __syncthreads();
  if (tid < TP) {
    __threadfence_system();
    st_flag_release(peers.flags[tid] + ((size_t)par * TP + rank) * max_blocks + b, epoch);
  }
  // 2) wait for every rank's chunk
  if (tid < TP) {
    const uint32_t* f = peers.flags[rank] + ((size_t)par * TP + tid) * max_blocks + b;
    while (ld_flag_acquire(f) != epoch) {
    }
  }
  __syncthreads();
  // 3) sum in rank order
  const bf16* mine = peers.buf[rank] + kFlagBase * TP * max_elems + (size_t)par * TP * max_elems;
  for (int i = e0 + tid * 8; i < e1; i += blockDim.x * 8) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int r = 0; r < TP; ++r) {
      const uint4 u = __ldcg(reinterpret_cast<const uint4*>(mine + (size_t)r * max_elems + i));
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 f = __bfloat1622float2(h[q]);
        acc[2 * q] += f.x;
        acc[2 * q + 1] += f.y;
      }
    }
    store8(data + i, acc);
  }
  if (tid == 0) epochs[b] = epoch;
}

template <int TP>
__global__ void __launch_bounds__(512) allreduce_lamport_kernel(bf16* __restrict__ data, int n,
                                                                const ArPeers peers,
                                                                uint32_t* state, int rank,
                                                                size_t max_elems,
                                                                const MoeCombineArgs cm) {
  pdl_entry();
  const int b = blockIdx.x, tid = threadIdx.x;
  const int chunk = (n / 8 + gridDim.x - 1) / gridDim.x * 8;
  const int e0 = b * chunk, e1 = min(n, e0 + chunk);
  const uint32_t epoch = next_epoch(state[0]);
  const int prev_n = (int)state[1];
  const int cur = epoch % 3, old = (epoch + 2) % 3;
  const size_t slot = (size_t)TP * max_elems;  // elements per buffer (TP source slots)
  // 1) push this rank's chunk to every rank (slot `rank` of buffer `cur`); with cm.y the chunk
  //    is the MoE combine of the expert outputs (as moe_combine)
  for (int i = e0 + tid * 8; i < e1; i += blockDim.x * 8) {
    uint4 v;
    if (cm.y) {
      // every pair's position and weight, then every local expert row, as independent loads;
      // summed in pair order (moe_combine's order)
      const int t = i / cm.hidden, d = i % cm.hidden;
      int pp[kArMaxK1];
      float ws[kArMaxK1];
#pragma unroll
      for (int s2 = 0; s2 < kArMaxK1; ++s2)
        if (s2 < cm.K1) {
          pp[s2] = cm.pair_pos[(size_t)t * cm.K1 + s2];
          ws[s2] = cm.w[(size_t)t * cm.K1 + s2];
        }
      uint4 yr[kArMaxK1];
#pragma unroll
      for (int s2 = 0; s2 < kArMaxK1; ++s2)
        if (s2 < cm.K1 && pp[s2] >= 0)  // (negative: expert not on this rank)
          yr[s2] = *reinterpret_cast<const uint4*>(cm.y + (size_t)pp[s2] * cm.hidden + d);
      float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int s2 = 0; s2 < kArMaxK1; ++s2)
        if (s2 < cm.K1 && pp[s2] >= 0) {
          const __nv_bfloat162* hy = reinterpret_cast<const __nv_bfloat162*>(&yr[s2]);
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float2 f = __bfloat1622float2(hy[q]);
            acc[2 * q] += ws[s2] * f.x;
            acc[2 * q + 1] += ws[s2] * f.y;
          }
        }
      __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
      for (int q = 0; q < 4; ++q) h[q] = __floats2bfloat162_rn(acc[2 * q], acc[2 * q + 1]);
    } else {
      v = *reinterpret_cast<const uint4*>(data + i);
    }
    v.x = clear_neg_zero(v.x);
    v.y = clear_neg_zero(v.y);
    v.z = clear_neg_zero(v.z);
    v.w = clear_neg_zero(v.w);
#pragma unroll
    for (int r = 0; r < TP; ++r)
      *reinterpret_cast<uint4*>(peers.buf[r] + cur * slot + (size_t)rank * max_elems + i) = v;
  }
  // 2) poll every source's chunk until it has arrived (all sources at once), sum in rank order
  const bf16* mine = peers.buf[rank] + cur * slot;
  for (int i = e0 + tid * 8; i < e1; i += blockDim.x * 8) {
    uint4 us[TP];
    unsigned need = (1u << TP) - 1u;
    while (need) {
#pragma unroll
      for (int r = 0; r < TP; ++r)
        if (need >> r & 1u) us[r] = ld_volatile16(mine + (size_t)r * max_elems + i);
#pragma unroll
      for (int r = 0; r < TP; ++r)
        if ((need >> r & 1u) && !(has_sentinel(us[r].x) || has_sentinel(us[r].y) ||
                                   has_sentinel(us[r].z) || has_sentinel(us[r].w)))
          need &= ~(1u << r);
    }
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int r = 0; r < TP; ++r) {
      const uint4 u = us[r];
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 f = __bfloat1622float2(h[q]);
        acc[2 * q] += f.x;
        acc[2 * q + 1] += f.y;
      }
    }
    store8(data + i, acc);
  }
  // 3) reset the previous call's region of buffer `old`; record this call
  lamport_clear<TP>(peers, rank, old, prev_n, state[3], max_elems, b, gridDim.x, tid, blockDim.x);
  __syncthreads();
  if (threadIdx.x == 0) lamport_finish_thread(state, epoch, n, 1, gridDim.x);
}

// The MTP layer's producer-pushed one-shot Lamport exchanges consumed by the add_rmsnorm after them. CTA t owns row
// t with add_rmsnorm's layout (hidden / 16 threads, thread i elements [16 i, 16 i + 16)); kMlnClr more CTAs only
// share the clearing of the previous call's region.
//  gather = 0 (sum): every rank's GEMV pushed its partial sums of the whole row (the attention's o_proj, GemvArPush):
//    the TP chunks are polled and summed in rank order (allreduce_lamport_kernel's sums, bf16), then add_rmsnorm's
//    arithmetic: v = bf16(resid + sum) (stored back with write_resid), out = bf16(rmsnorm(v) * w), the FP8 1x128
//    quantization (xq / xs, optional).
//  gather = 1: rank q's GEMV pushed the row's columns [q hidden / TP, (q + 1) hidden / TP) (the column-parallel input
//    projection): each element is polled in its owner's slot (an all-gather), the other ranks' columns are stored
//    into resid (the producer stored its own), then out = bf16(rmsnorm(v) * w) (add_rmsnorm without a delta).
// The producer lets this grid launch only after its own PDL wait (late trigger), so every earlier kernel has
// completed: the call state is read, the previous call's region cleared and the arrivals polled before this grid's
// wait, overlapping the producer's end; the outputs are none of the producer's inputs or outputs. The call is
// recorded after the wait (the producer reads the call state until its last push) by the last CTA, no fences.
// late: the next kernel launches once the row is done instead of at this grid's start.
constexpr int kMlnClr = 8;
// MtpPickArgs' bookkeeping of sequences s0, s0 + ds, ... (mtp_accept's, from the same inputs: every earlier kernel's
// outputs are complete when the consumer runs, whether before its wait or not: L2 loads); the clearing CTAs take one
// sequence each (a sequence's chain is ~5 dependent L2 loads)
__device__ __noinline__ void mtp_book(int s0, int ds, int B, float force_accept, const int* row0, const int* nrows,
                                     const int* sampled, const int* ids, const int* sample_idx, const int* positions,
                                     const int* token_slot, int* accepted, int* next_pos, int npos, int* next_slot,
                                     const int* sel_src, int* sel_dst, int* vis_dst, int sel_n, int kpool) {
  for (int s = s0; s < B; s += ds) {
    const int r0 = __ldcg(row0 + s);
    const int a = mtp_accept_count<true>(force_accept, r0, __ldcg(nrows + s), sampled, ids, sample_idx, positions,
                                         token_slot);
    const int t = __ldcg(sample_idx + r0 + a);
    const int pos = __ldcg(positions + t);
    if (threadIdx.x == 0) {
      accepted[s] = a;
      for (int j = 0; j < npos; ++j) next_pos[j * B + s] = pos + 1 + j;
      next_slot[s] = __ldcg(token_slot + t);
      if (vis_dst) vis_dst[s] = (pos + 1) / kpool;
    }
    if (sel_dst)
      for (int i = threadIdx.x; i < sel_n; i += blockDim.x)
        sel_dst[(size_t)s * sel_n + i] = __ldcg(sel_src + (size_t)t * sel_n + i);
  }
}

template <int TP>
__global__ void __launch_bounds__(256) mtp_lamport_norm_kernel(int T, int hidden, const ArPeers peers,
                                                               uint32_t* state, int rank, size_t max_elems,
                                                               int gather, bf16* __restrict__ resid,
                                                               int write_resid, const bf16* __restrict__ w,
                                                               bf16* __restrict__ out, fp8* __restrict__ xq,
                                                               float* __restrict__ xs, float eps, int late,
                                                               int hoist, const MoeCombineArgs cm,
                                                               const MtpPickArgs pk, int rec1, unsigned* ready,
                                                               int rowp) {
  if (!late) pdl_trigger();
  if (!hoist) pdl_wait();
  __shared__ float sh[32];
  const int t = blockIdx.x, tid = threadIdx.x, c0 = tid * 16;
  const uint32_t epoch = next_epoch(__ldcg(&state[0]));
  const int prev_n = (int)__ldcg(&state[1]);
  const uint32_t prev_kind = __ldcg(&state[3]);
  const int cur = epoch % 3, old = (epoch + 2) % 3;
  if (cm.y) {
    // (combine, after the wait: no producer push) the MoE combine of this rank's expert rows (allreduce_lamport_kernel's
    // arithmetic, pairs in order) as 16-byte chunks spread over the grid's clearing CTAs, pushed into every rank's slot
    // (CR: the clearing CTAs push every chunk, the row CTAs none: a row CTA's end -- the head's rows-ready count after
    // its fence (HF) and the grid's completion -- then waits for no NVLink store of its own; the same chunks, the same
    // sums. rowp: every CTA of the grid pushes chunks)
    const bool cr = !rowp && (int)gridDim.x > T;
    const int nb = cr ? (int)gridDim.x - T : (int)gridDim.x, tb = cr ? t - T : t;
    const int nch = T * hidden / 8, per = (nch + nb - 1) / nb;
    const int ch0 = tb >= 0 ? tb * per : nch, ch1 = min(nch, ch0 + per);
    for (int ch = ch0 + tid; ch < ch1; ch += blockDim.x) {
      const int e = ch * 8, tok = e / cm.hidden, d = e % cm.hidden;
      int pp[kArMaxK1];
      float ws[kArMaxK1];
#pragma unroll
      for (int s2 = 0; s2 < kArMaxK1; ++s2)
        if (s2 < cm.K1) {
          pp[s2] = __ldcg(cm.pair_pos + (size_t)tok * cm.K1 + s2);
          ws[s2] = __ldcg(cm.w + (size_t)tok * cm.K1 + s2);
        }
      uint4 yr[kArMaxK1];
#pragma unroll
      for (int s2 = 0; s2 < kArMaxK1; ++s2)
        if (s2 < cm.K1 && pp[s2] >= 0)
          yr[s2] = __ldcg(reinterpret_cast<const uint4*>(cm.y + (size_t)pp[s2] * cm.hidden + d));
      float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int s2 = 0; s2 < kArMaxK1; ++s2)
        if (s2 < cm.K1 && pp[s2] >= 0) {
          const __nv_bfloat162* hy = reinterpret_cast<const __nv_bfloat162*>(&yr[s2]);
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float2 f = __bfloat1622float2(hy[q]);
            acc[2 * q] += ws[s2] * f.x;
            acc[2 * q + 1] += ws[s2] * f.y;
          }
        }
      uint4 v;
      __nv_bfloat162* hv = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
      for (int q = 0; q < 4; ++q) hv[q] = __floats2bfloat162_rn(acc[2 * q], acc[2 * q + 1]);
      v.x = clear_neg_zero(v.x);
      v.y = clear_neg_zero(v.y);
      v.z = clear_neg_zero(v.z);
      v.w = clear_neg_zero(v.w);
#pragma unroll
      for (int r = 0; r < TP; ++r)
        *reinterpret_cast<uint4*>(peers.buf[r] + (size_t)cur * TP * max_elems + (size_t)rank * max_elems + e) = v;
    }
    if (late == 2 && t >= T) pdl_trigger();  // (late 2: the CTAs without a row once their chunks are pushed)
  }
  if (t < T) {
    const size_t e0 = (size_t)t * hidden + c0;
    const bf16* mine = peers.buf[rank] + (size_t)cur * TP * max_elems;
    float wv[16], v[16];
    // (pick) the sequence s whose last accepted row this row is, found before the rows are polled: its inputs are
    // final when this grid starts (the step's sampling, every earlier kernel complete: L2 loads), and its chain of
    // dependent loads overlaps the producer's end instead of following the arrivals
    int ps = -1;
    for (int s2 = 0; s2 < pk.B; ++s2) {
      const int r0 = __ldcg(pk.row0 + s2);
      const int a = mtp_accept_count<true>(pk.force_accept, r0, __ldcg(pk.nrows + s2), pk.sampled, pk.ids,
                                           pk.sample_idx, pk.positions, pk.token_slot);
      if (__ldcg(pk.sample_idx + r0 + a) == t) ps = s2;
    }
    load8(w + c0, wv);  // (weights)
    load8(w + c0 + 8, wv + 8);
    if (!gather) {  // (the residual: written by a kernel that has completed)
      const uint4 r0 = __ldcg(reinterpret_cast<const uint4*>(resid + e0));
      const uint4 r1 = __ldcg(reinterpret_cast<const uint4*>(resid + e0 + 8));
      const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&r0);
      const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&r1);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 a = __bfloat1622float2(h0[q]), b = __bfloat1622float2(h1[q]);
        v[2 * q] = a.x;
        v[2 * q + 1] = a.y;
        v[8 + 2 * q] = b.x;
        v[8 + 2 * q + 1] = b.y;
      }
    }
    // the previous call's region of buffer `old` back to the sentinel while this call's chunks are in flight
    lamport_clear<TP>(peers, rank, old, prev_n, prev_kind, max_elems, t, gridDim.x, tid, blockDim.x);
    if (gather) {
      const int o = c0 / (hidden / TP);  // the owner of the thread's columns
      const bf16* src = mine + (size_t)o * max_elems + e0;
      uint4 u[2];
      unsigned need = 3u;
      while (need) {
#pragma unroll
        for (int h = 0; h < 2; ++h)
          if (need >> h & 1u) u[h] = ld_volatile16(src + 8 * h);
#pragma unroll
        for (int h = 0; h < 2; ++h)
          if ((need >> h & 1u) &&
              !(has_sentinel(u[h].x) || has_sentinel(u[h].y) || has_sentinel(u[h].z) || has_sentinel(u[h].w)))
            need &= ~(1u << h);
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const __nv_bfloat162* hq = reinterpret_cast<const __nv_bfloat162*>(&u[h]);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const float2 f = __bfloat1622float2(hq[q]);
          v[8 * h + 2 * q] = f.x;
          v[8 * h + 2 * q + 1] = f.y;
        }
        if (o != rank) *reinterpret_cast<uint4*>(resid + e0 + 8 * h) = u[h];
      }
    } else {
      uint4 us[2][TP];
      unsigned need = (1u << (2 * TP)) - 1u;
      while (need) {
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int r = 0; r < TP; ++r)
            if (need >> (h * TP + r) & 1u) us[h][r] = ld_volatile16(mine + (size_t)r * max_elems + e0 + 8 * h);
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int r = 0; r < TP; ++r)
            if ((need >> (h * TP + r) & 1u) && !(has_sentinel(us[h][r].x) || has_sentinel(us[h][r].y) ||
                                                  has_sentinel(us[h][r].z) || has_sentinel(us[h][r].w)))
              need &= ~(1u << (h * TP + r));
      }
      if (late == 2) pdl_trigger();  // (the next kernel launches once the row has arrived, before its arithmetic)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int r = 0; r < TP; ++r) {
          const __nv_bfloat162* hq = reinterpret_cast<const __nv_bfloat162*>(&us[h][r]);
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float2 f = __bfloat1622float2(hq[q]);
            acc[2 * q] += f.x;
            acc[2 * q + 1] += f.y;
          }
        }
#pragma unroll
        for (int q = 0; q < 8; ++q) v[8 * h + q] = bf16r(v[8 * h + q] + bf16r(acc[q]));
      }
      if (write_resid) {
        store8(resid + e0, v);
        store8(resid + e0 + 8, v + 8);
      }
      if (ps >= 0) {
        store8(pk.hp + (size_t)ps * hidden + c0, v);
        store8(pk.hp + (size_t)ps * hidden + c0 + 8, v + 8);
      }
    }
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) ss += v[i] * v[i];
    ss = block_sum(ss, sh);
    const float rr = rsqrtf(ss / hidden + eps);
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      v[i] = bf16r(v[i] * rr * wv[i]);
      amax = fmaxf(amax, fabsf(v[i]));
    }
    store8(out + e0, v);
    store8(out + e0 + 8, v + 8);
    if (ps >= 0) {
      store8(pk.xp + (size_t)ps * hidden + c0, v);
      store8(pk.xp + (size_t)ps * hidden + c0 + 8, v + 8);
    }
    if (xq) {
      // 8 threads (aligned lanes) per 128-element block
#pragma unroll
      for (int o = 1; o < 8; o <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
      const float scale = fmaxf(amax, 1e-12f) * (1.f / 448.f);
      const float ys = __frcp_rn(scale);
      __nv_fp8x4_e4m3 qv[4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        qv[i] = __nv_fp8x4_e4m3(make_float4(div_rn(v[4 * i], scale, ys), div_rn(v[4 * i + 1], scale, ys),
                                            div_rn(v[4 * i + 2], scale, ys), div_rn(v[4 * i + 3], scale, ys)));
      __nv_fp8x4_e4m3* q = reinterpret_cast<__nv_fp8x4_e4m3*>(xq + e0);
#pragma unroll
      for (int i = 0; i < 4; ++i) q[i] = qv[i];
      if ((tid & 7) == 0) xs[(size_t)t * (hidden / 128) + c0 / 128] = scale;
      if (ps >= 0) {
        __nv_fp8x4_e4m3* qp = reinterpret_cast<__nv_fp8x4_e4m3*>(pk.xqp + (size_t)ps * hidden + c0);
#pragma unroll
        for (int i = 0; i < 4; ++i) qp[i] = qv[i];
        if ((tid & 7) == 0) pk.xsp[(size_t)ps * (hidden / 128) + c0 / 128] = scale;
      }
    }
    if (ready) {  // (the head's rows-ready count: the row stored, for its bulk copies too)
      // each thread's row stores ordered for the async proxy, then one acq-rel fence by a thread of the last warp,
      // cumulative over the CTA barrier: a per-thread __threadfence would also wait for the combine's NVLink pushes in
      // flight; the row CTAs push no chunk
      asm volatile("fence.proxy.async.global;\n" ::: "memory");
      __syncthreads();
      if (tid == (int)blockDim.x - 1) {
        asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
        atomicAdd(ready, 1u);
      }
    }
  } else {  // (a clearing CTA)
    lamport_clear<TP>(peers, rank, old, prev_n, prev_kind, max_elems, t, gridDim.x, tid, blockDim.x);
    if (pk.accepted && t - T < pk.B)
      mtp_book(t - T, (int)gridDim.x - T, pk.B, pk.force_accept, pk.row0, pk.nrows, pk.sampled, pk.ids, pk.sample_idx,
               pk.positions, pk.token_slot, pk.accepted, pk.next_pos, pk.npos, pk.next_slot, pk.sel_src, pk.sel_dst,
               pk.vis_dst, pk.sel_n, pk.kpool);
  }
  if (late == 1) pdl_trigger();
  if (hoist) pdl_wait();
  if (rec1) {
    // (T == 1) the row's CTA records the call without the arrival counter (one L2 round trip less before the grid
    // ends): every CTA has read the call state by then -- the gathering / summing forms run the row's CTA alone, and
    // in the combine form every CTA pushed chunks of the row before the row's CTA saw them arrive
    if (t == 0 && tid == 0) {
      state[0] = epoch;
      state[1] = (uint32_t)(T * hidden);
      state[3] = 1u;
    }
    return;
  }
  __syncthreads();
  if (tid == 0 && atomicAdd(&state[2], 1u) == gridDim.x - 1) {
    state[0] = epoch;
    state[1] = (uint32_t)(T * hidden);
    state[3] = 1u;
    state[2] = 0u;
  }
}

// Two-shot Lamport all-reduce (larger messages): reduce-scatter (chunk b of shard s is pushed
// to rank s, which polls the TP contributions and sums them in rank order) then all-gather
// (the reduced chunk is pushed into every rank's gather region and polled there). Per rank,
// after the one-shot buffers: 3 rotating buffers of [scatter: tp slots x max_elems / tp |
// gather: tp shards x max_elems / tp] = 2 * max_elems elements, sentinel-filled. Same call
// state protocol as the one-shot kernel (shared epoch).
template <int TP>
__global__ void __launch_bounds__(512) allreduce_lamport2_kernel(bf16* __restrict__ data, int n,
                                                                 const ArPeers peers,
                                                                 uint32_t* state, int rank,
                                                                 size_t max_elems) {
  pdl_entry();
  const int b = blockIdx.x, tid = threadIdx.x;
  const int S = n / TP;
  const size_t sl = max_elems / TP;                  // slot / shard capacity
  const size_t base2 = (size_t)5 * TP * max_elems;   // after one-shot + flag buffers
  const int chunk = (S / 8 + gridDim.x - 1) / gridDim.x * 8;
  const int c0 = b * chunk, c1 = min(S, c0 + chunk);
  const uint32_t epoch = next_epoch(state[0]);
  const int prev_n = (int)state[1];
  const int cur = epoch % 3, old = (epoch + 2) % 3;
  auto buf2 = [&](int r, int which) { return peers.buf[r] + base2 + (size_t)which * 2 * max_elems; };
  // 1) scatter: chunk of shard s -> rank s, slot `rank`
#pragma unroll
  for (int s2 = 0; s2 < TP; ++s2) {
    bf16* dst = buf2(s2, cur) + (size_t)rank * sl;
    for (int i = c0 + tid * 8; i < c1; i += blockDim.x * 8) {
      uint4 v = *reinterpret_cast<const uint4*>(data + (size_t)s2 * S + i);
      v.x = clear_neg_zero(v.x);
      v.y = clear_neg_zero(v.y);
      v.z = clear_neg_zero(v.z);
      v.w = clear_neg_zero(v.w);
      *reinterpret_cast<uint4*>(dst + i) = v;
    }
  }
  // 2) reduce this rank's shard, push it into every rank's gather region (shard `rank`)
  const bf16* rs = buf2(rank, cur);
  for (int i = c0 + tid * 8; i < c1; i += blockDim.x * 8) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int r = 0; r < TP; ++r) {
      uint4 u;
      do {
        u = ld_volatile16(rs + (size_t)r * sl + i);
      } while (has_sentinel(u.x) || has_sentinel(u.y) || has_sentinel(u.z) ||
               has_sentinel(u.w));
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 f = __bfloat1622float2(h[q]);
        acc[2 * q] += f.x;
        acc[2 * q + 1] += f.y;
      }
    }
    uint4 pk;
    __nv_bfloat162* ph = reinterpret_cast<__nv_bfloat162*>(&pk);
#pragma unroll
    for (int q = 0; q < 4; ++q) ph[q] = __floats2bfloat162_rn(acc[2 * q], acc[2 * q + 1]);
    pk.x = clear_neg_zero(pk.x);
    pk.y = clear_neg_zero(pk.y);
    pk.z = clear_neg_zero(pk.z);
    pk.w = clear_neg_zero(pk.w);
#pragma unroll
    for (int r = 0; r < TP; ++r)
      *reinterpret_cast<uint4*>(buf2(r, cur) + max_elems + (size_t)rank * sl + i) = pk;
  }
  // 3) gather every shard's chunk
  const bf16* gs = buf2(rank, cur) + max_elems;
#pragma unroll
  for (int s2 = 0; s2 < TP; ++s2)
    for (int i = c0 + tid * 8; i < c1; i += blockDim.x * 8) {
      uint4 u;
      do {
        u = ld_volatile16(gs + (size_t)s2 * sl + i);
      } while (has_sentinel(u.x) || has_sentinel(u.y) || has_sentinel(u.z) ||
               has_sentinel(u.w));
      *reinterpret_cast<uint4*>(data + (size_t)s2 * S + i) = u;
    }
  // 4) reset the previous call's region of buffer `old`; record this call
  lamport_clear<TP>(peers, rank, old, prev_n, state[3], max_elems, b, gridDim.x, tid, blockDim.x);
  __syncthreads();
  if (threadIdx.x == 0) lamport_finish_thread(state, epoch, n, 2, gridDim.x);
}

// Lamport max all-reduce of nonzero keys: slot buffer e % 3 receives every rank's keys, the
// receiver polls until all are nonzero, and the previous call's buffer is reset to zero.
template <int TP>
__global__ void __launch_bounds__(1024) max_u64_kernel(unsigned long long* __restrict__ keys,
                                                       int n, const ArPeers peers,
                                                       uint32_t* state, int rank, int* tokens) {
  pdl_entry();
  const int i = threadIdx.x;
  const uint32_t epoch = next_epoch(state[0]);
  const int prev_n = (int)state[1];
  const int cur = epoch % 3, old = (epoch + 2) % 3;
  const size_t buf = (size_t)TP * kMaxKeyRows;
  unsigned long long v = 0;
  if (i < n) {
    v = keys[i];
#pragma unroll
    for (int r = 0; r < TP; ++r)
      *(volatile unsigned long long*)(peers.keys[r] + cur * buf + (size_t)rank * kMaxKeyRows + i) = v;
    unsigned long long m = 0;
#pragma unroll
    for (int r = 0; r < TP; ++r) {
      unsigned long long x;
      do {
        x = *(volatile unsigned long long*)(peers.keys[rank] + cur * buf + (size_t)r * kMaxKeyRows + i);
      } while (x == 0ull);
      m = m > x ? m : x;
    }
    keys[i] = m;
    if (tokens) tokens[i] = (int)(0xffffffffu - (uint32_t)(m & 0xffffffffull));
  }
  if (i < prev_n)
#pragma unroll
    for (int r = 0; r < TP; ++r) peers.keys[rank][old * buf + (size_t)r * kMaxKeyRows + i] = 0ull;
  __syncthreads();
  if (i == 0) {
    state[0] = epoch;
    state[1] = (uint32_t)n;
  }
}

void custom_max_u64(unsigned long long* keys, int n, const ArPeers& peers, uint32_t* state,
                    int rank, int tp, cudaStream_t st, int* tokens) {
  if (n == 0) return;
  CHECK(n <= kMaxKeyRows, "custom_max_u64: too many rows");
  switch (tp) {
    case 2: launch_pdl(max_u64_kernel<2>, dim3(1), dim3(kMaxKeyRows), 0, st, keys, n, peers, state, rank, tokens); break;
    case 4: launch_pdl(max_u64_kernel<4>, dim3(1), dim3(kMaxKeyRows), 0, st, keys, n, peers, state, rank, tokens); break;
    case 8: launch_pdl(max_u64_kernel<8>, dim3(1), dim3(kMaxKeyRows), 0, st, keys, n, peers, state, rank, tokens); break;
    default: CHECK(false, "custom_max_u64: unsupported tp");
  }
}

__global__ void fill_sentinel_kernel(uint32_t* p, size_t n_words) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n_words;
       i += (size_t)gridDim.x * blockDim.x)
    p[i] = kSentinel2;
}

// [3 one-shot Lamport buffers][2 flag-protocol parities] x tp x max_elems, then 3 two-shot
// Lamport buffers of 2 x max_elems
size_t allreduce_buf_elems(int tp, size_t max_elems) {
  return (size_t)5 * tp * max_elems + (size_t)6 * max_elems;
}

void allreduce_lamport_init(bf16* buf, int tp, size_t max_elems, cudaStream_t st) {
  // one-shot Lamport buffers
  fill_sentinel_kernel<<<256, 256, 0, st>>>(reinterpret_cast<uint32_t*>(buf),
                                            (size_t)3 * tp * max_elems / 2);
  // two-shot Lamport buffers
  fill_sentinel_kernel<<<256, 256, 0, st>>>(
      reinterpret_cast<uint32_t*>(buf + (size_t)5 * tp * max_elems), (size_t)6 * max_elems / 2);
  CUDA_CHECK(cudaGetLastError());
}

// Lamport protocol: the one-shot kernel up to 64 tokens of 4096, the two-shot one (tried first) from there
constexpr size_t kLamportMaxElems = (size_t)64 * 4096;
constexpr size_t kLamport2MinElems = (size_t)64 * 4096;
// elements up to which a one-shot all-reduce is fused into its consumer (the mHC boundary) even where
// the standalone all-reduce takes the two-shot kernel (64 tokens of 4096: the fused exchange saves the
// standalone kernel and the sums' store and reload)
constexpr size_t kFuseOneshotMaxElems = (size_t)64 * 4096;

bool allreduce_fusable(size_t n, int tp, size_t max_elems) {
  // the conditions under which custom_allreduce / custom_allreduce_moe pick the one-shot kernel,
  // or the fused one-shot's own bound
  return tp == 4 && n > 0 && n <= max_elems && n % 8 == 0 && n <= kLamportMaxElems &&
         (n < kLamport2MinElems || n <= kFuseOneshotMaxElems);
}

bool custom_allreduce_moe(bf16* out, const MoeCombineArgs& cm, size_t n, const ArPeers& peers,
                          uint32_t* epochs, int rank, int tp, size_t max_elems, int max_blocks,
                          cudaStream_t st) {
  if (n == 0) return true;
  if (n > max_elems || n % 8 != 0 || n > kLamportMaxElems || n >= kLamport2MinElems ||
      cm.hidden % 8 != 0)
    return false;
  uint32_t* state = epochs + max_blocks;
  const int blocks = (int)std::min<size_t>(max_blocks, std::max<size_t>(1, cdiv(n, 1024)));
  const int threads = n / blocks >= 4096 ? 512 : 128;
  switch (tp) {
    case 2:
      launch_pdl(allreduce_lamport_kernel<2>, dim3(blocks), dim3(threads), 0, st, out, (int)n,
                 peers, state, rank, max_elems, cm);
      return true;
    case 4:
      launch_pdl(allreduce_lamport_kernel<4>, dim3(blocks), dim3(threads), 0, st, out, (int)n,
                 peers, state, rank, max_elems, cm);
      return true;
    case 8:
      launch_pdl(allreduce_lamport_kernel<8>, dim3(blocks), dim3(threads), 0, st, out, (int)n,
                 peers, state, rank, max_elems, cm);
      return true;
    default:
      return false;
  }
}

bool mtp_lamport_norm_ok(int T, int hidden, int tp, size_t max_elems) {
  const size_t n = (size_t)T * hidden;
  return T >= 1 && tp == 4 && n <= max_elems && n <= kLamportMaxElems && n < kLamport2MinElems &&
         hidden % 128 == 0 && hidden / 16 <= 256 && (hidden / tp) % 16 == 0;
}

void mtp_lamport_norm(int T, int hidden, bool gather, const ArPeers& peers, uint32_t* epochs, int rank, int tp,
                      size_t max_elems, int max_blocks, bf16* resid, bool write_resid, const bf16* w, bf16* out,
                      fp8* xq, float* xs, float eps, bool late, cudaStream_t st, const MtpPickArgs* pick) {
  CHECK(mtp_lamport_norm_ok(T, hidden, tp, max_elems), "mtp_lamport_norm: unsupported shape");
  CHECK(!pick || (!gather && pick->xp && pick->hp && (!xq || pick->xqp)), "mtp_lamport_norm: pick args");
  CHECK(!pick || !pick->accepted || (pick->next_pos && pick->next_slot && (!pick->sel_dst || (pick->sel_src &&
                                     pick->vis_dst && pick->kpool > 0))), "mtp_lamport_norm: bookkeeping args");
  uint32_t* state = epochs + max_blocks;
  // one row without the bookkeeping: the row's CTA alone (it clears the previous call's region itself) records the call
  const bool rec1 = T == 1 && !(pick && pick->accepted);
  // (hoist: both forms poll the arrivals before the PDL wait)
  launch_pdl(mtp_lamport_norm_kernel<4>, dim3(rec1 ? 1 : T + kMlnClr), dim3(hidden / 16), 0, st, T, hidden, peers,
             state, rank, max_elems, gather ? 1 : 0, resid, write_resid ? 1 : 0, w, out, xq, xs, eps, late ? 1 : 0,
             /*hoist=*/1, MoeCombineArgs{}, pick ? *pick : MtpPickArgs{}, rec1 ? 1 : 0, (unsigned*)nullptr, 0);
}

void mtp_lamport_combine_norm(int T, int hidden, const MoeCombineArgs& cm, const ArPeers& peers, uint32_t* epochs,
                              int rank, int tp, size_t max_elems, int max_blocks, bf16* resid, const bf16* w,
                              bf16* out, float eps, int late, cudaStream_t st, unsigned* ready) {
  CHECK(mtp_lamport_norm_ok(T, hidden, tp, max_elems) && cm.y && cm.hidden == hidden && cm.K1 <= kArMaxK1 &&
        late >= 0 && late <= 2, "mtp_lamport_combine_norm: unsupported shape");
  uint32_t* state = epochs + max_blocks;
  // (cr: at <= 4 rows -- one chunk per thread of the clearing CTAs -- those push the combine's chunks, the row CTAs
  // none; above, every CTA)
  const bool cr = T <= 4;
  // one row: the row's CTA records the call (it reads the call state itself; every other CTA pushes chunks of the row
  // -- after reading the state -- before the row's CTA sees them arrive: each has a chunk below)
  const int nch = hidden / 8, per = cr ? (nch + kMlnClr - 1) / kMlnClr : (nch + kMlnClr) / (1 + kMlnClr);
  const bool rec1 = T == 1 && (cr ? (kMlnClr - 1) * per < nch : per * kMlnClr < nch);
  launch_pdl(mtp_lamport_norm_kernel<4>, dim3(T + kMlnClr), dim3(hidden / 16), 0, st, T, hidden, peers, state,
             rank, max_elems, 0, resid, 0, w, out, nullptr, nullptr, eps, late, 0, cm, MtpPickArgs{}, rec1 ? 1 : 0,
             ready, cr ? 0 : 1);
}

bool custom_allreduce(bf16* data, size_t n, const ArPeers& peers, uint32_t* epochs, int rank,
                      int tp, size_t max_elems, int max_blocks, cudaStream_t st) {
  if (n == 0) return true;
  if (n > max_elems || n % 8 != 0) return false;
  if (n >= kLamport2MinElems && n % (8 * tp) == 0 && max_elems % tp == 0) {
    uint32_t* state = epochs + max_blocks;
    const int blocks =
        (int)std::min<size_t>(max_blocks, std::max<size_t>(1, cdiv(n / tp, 512)));
    switch (tp) {
      case 2:
        launch_pdl(allreduce_lamport2_kernel<2>, dim3(blocks), dim3(128), 0, st, data, (int)n,
                   peers, state, rank, max_elems);
        return true;
      case 4:
        launch_pdl(allreduce_lamport2_kernel<4>, dim3(blocks), dim3(128), 0, st, data, (int)n,
                   peers, state, rank, max_elems);
        return true;
      case 8:
        launch_pdl(allreduce_lamport2_kernel<8>, dim3(blocks), dim3(128), 0, st, data, (int)n,
                   peers, state, rank, max_elems);
        return true;
      default:
        break;
    }
  }
  if (n <= kLamportMaxElems) {
    // ~2 KB (1K elements) per block so the pushes spread over many SMs
    uint32_t* state = epochs + max_blocks;  // epoch / previous size / finished blocks
    const int blocks = (int)std::min<size_t>(max_blocks, std::max<size_t>(1, cdiv(n, 1024)));
    const int threads = n / blocks >= 4096 ? 512 : 128;
    switch (tp) {
      case 2:
        launch_pdl(allreduce_lamport_kernel<2>, dim3(blocks), dim3(threads), 0, st, data, (int)n,
                   peers, state, rank, max_elems, MoeCombineArgs{});
        return true;
      case 4:
        launch_pdl(allreduce_lamport_kernel<4>, dim3(blocks), dim3(threads), 0, st, data, (int)n,
                   peers, state, rank, max_elems, MoeCombineArgs{});
        return true;
      case 8:
        launch_pdl(allreduce_lamport_kernel<8>, dim3(blocks), dim3(threads), 0, st, data, (int)n,
                   peers, state, rank, max_elems, MoeCombineArgs{});
        return true;
      default:
        break;
    }
  }
  // ~8 KB (4K elements) per block, at most max_blocks blocks
  int blocks = (int)std::min<size_t>(max_blocks, std::max<size_t>(1, cdiv(n, 4096)));
  const int threads = n / blocks >= 4096 ? 512 : 256;
  switch (tp) {
    case 2:
      launch_pdl(allreduce_kernel<2>, dim3(blocks), dim3(threads), 0, st, data, (int)n, peers, epochs, rank,
                                                      max_elems, max_blocks);
      break;
    case 4:
      launch_pdl(allreduce_kernel<4>, dim3(blocks), dim3(threads), 0, st, data, (int)n, peers, epochs, rank,
                                                      max_elems, max_blocks);
      break;
    case 8:
      launch_pdl(allreduce_kernel<8>, dim3(blocks), dim3(threads), 0, st, data, (int)n, peers, epochs, rank,
                                                      max_elems, max_blocks);
      break;
    default:
      return false;
  }
  return true;
}

}  // namespace k
}  // namespace glm
