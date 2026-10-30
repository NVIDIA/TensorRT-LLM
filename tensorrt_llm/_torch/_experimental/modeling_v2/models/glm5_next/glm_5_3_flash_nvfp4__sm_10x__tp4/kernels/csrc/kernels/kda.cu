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
// Kimi Delta Attention (KDA) kernels: short causal conv, gated delta-rule recurrence and the
// gated output RMSNorm. Sequences are packed (varlen) with per-sequence state slots.
#include <algorithm>

#include "pdl.cuh"
#include "kda.h"
#include "reduce.cuh"

namespace glm {
namespace k {

// ------------------------------------------------------------------ causal conv1d + SiLU
// Parallel over (token, channel): tap j < W-1 of token t reads the input of token t-(W-1)+j if it
// belongs to the same sequence's new tokens, else the sequence's conv state (zeros when fresh).
// The conv state is updated by a second kernel once every token has read the old state.
__device__ __forceinline__ int seq_of(const int* cu, int B, int t) {
  int lo = 0, hi = B;  // find b with cu[b] <= t < cu[b+1]
  while (hi - lo > 1) {
    const int mid = (lo + hi) >> 1;
    if (cu[mid] <= t) lo = mid; else hi = mid;
  }
  return lo;
}

template <int W>
__global__ void kda_conv_kernel(bf16* __restrict__ out, const bf16* __restrict__ in, int ld_in,
                                const bf16* __restrict__ w, const bf16* __restrict__ conv_state,
                                const KdaSeqMeta meta, int B, int t_begin, int C) {
  pdl_entry();
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int t = t_begin + blockIdx.y;
  if (c >= C) return;
  const int b = seq_of(meta.cu_seqlens, B, t);
  const int t0 = meta.cu_seqlens[b];
  const int i = t - t0;
  const bool has_state = meta.has_state[b];
  const bf16* st = conv_state + (size_t)meta.slots[b] * meta.conv_ss + (size_t)c * (W - 1);
  float acc = __bfloat162float(w[(size_t)c * W + W - 1]) * __bfloat162float(in[(size_t)t * ld_in + c]);
#pragma unroll
  for (int j = 0; j < W - 1; ++j) {
    const int off = i - (W - 1) + j;  // offset within the new tokens of the tapped input
    float x;
    if (off >= 0)
      x = __bfloat162float(in[(size_t)(t0 + off) * ld_in + c]);
    else
      x = has_state ? __bfloat162float(st[W - 1 + off]) : 0.f;
    acc += __bfloat162float(w[(size_t)c * W + j]) * x;
  }
  out[(size_t)t * C + c] = __float2bfloat16(acc / (1.f + __expf(-acc)));
}

__device__ __forceinline__ void load8(float* d, const bf16* p) {
  const uint4 u = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(h[i]);
    d[2 * i] = f.x;
    d[2 * i + 1] = f.y;
  }
}

// The same over tiles of TT tokens x 1024 channels with 4 channels per thread (8-byte slices, 256
// threads): each thread cp.asyncs its slice of the tile's TT + W - 1 input rows (zeros outside the
// step's new tokens) into shared memory, then walks its tokens with the window in registers (a thread
// reads only the rows it staged: no block barrier). The conv is bound by the SiLU's latency chains:
// 8 channels per thread over a run of tokens take 120 registers (16 warps per SM), half the registers
// per thread give several times the warps. Same arithmetic (bitwise).
template <int W, int TT>
__global__ void __launch_bounds__(256) kda_conv4s_kernel(bf16* __restrict__ out, const bf16* __restrict__ in,
                                                         int ld_in, const bf16* __restrict__ w,
                                                         const bf16* __restrict__ conv_state,
                                                         const KdaSeqMeta meta, int B, int t_begin, int T,
                                                         int C) {
  pdl_entry();
  extern __shared__ __align__(16) uint8_t kda_conv_smem[];
  uint2* tile = reinterpret_cast<uint2*>(kda_conv_smem);  // [TT + W - 1][256]: row r = token ta - (W - 1) + r
  const int tx = threadIdx.x;
  const int c = (blockIdx.x * 256 + tx) * 4;
  const bool cok = c < C;
  const int ta = t_begin + blockIdx.y * TT, tend = t_begin + T;
#pragma unroll 4
  for (int r = 0; r < TT + W - 1; ++r) {
    const int tr = ta - (W - 1) + r;
    const bool ok = cok && tr >= t_begin && tr < tend;
    const bf16* src = in + (size_t)(ok ? tr : t_begin) * ld_in + (cok ? c : 0);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(
                     static_cast<uint32_t>(__cvta_generic_to_shared(&tile[r * 256 + tx]))),
                 "l"(src), "r"(ok ? 8 : 0));
  }
  asm volatile("cp.async.commit_group;\n" ::);
  if (!cok) return;
  float wt[4][W];
  {
    float tmp[8 * W / 2];
#pragma unroll
    for (int i = 0; i < W / 2; ++i) load8(tmp + 8 * i, w + (size_t)c * W + 8 * i);
#pragma unroll
    for (int e = 0; e < 4; ++e)
#pragma unroll
      for (int j = 0; j < W; ++j) wt[e][j] = tmp[e * W + j];
  }
  int b = seq_of(meta.cu_seqlens, B, ta);
  int s0 = meta.cu_seqlens[b], s1 = meta.cu_seqlens[b + 1];
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  auto row = [&](float* d, int r) {
    const uint2 u = tile[r * 256 + tx];
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const float2 f = __bfloat1622float2(h[i]);
      d[2 * i] = f.x;
      d[2 * i + 1] = f.y;
    }
  };
  float x[W - 1][4];
  auto fill = [&](int t) {
    const bool has_state = meta.has_state[b];
    const bf16* st = conv_state + (size_t)meta.slots[b] * meta.conv_ss + (size_t)c * (W - 1);
#pragma unroll
    for (int j = 0; j < W - 1; ++j) {
      const int tt = t - (W - 1) + j;
      if (tt >= s0) {
        row(x[j], tt - (ta - (W - 1)));
      } else {
        const int si = W - 1 + (tt - s0);
#pragma unroll
        for (int e = 0; e < 4; ++e) x[j][e] = has_state ? __bfloat162float(st[e * (W - 1) + si]) : 0.f;
      }
    }
  };
  fill(ta);
#pragma unroll 2
  for (int u = 0; u < TT; ++u) {
    const int t = ta + u;
    if (t >= tend) break;
    if (t >= s1) {
      while (t >= meta.cu_seqlens[b + 1]) ++b;
      s0 = meta.cu_seqlens[b];
      s1 = meta.cu_seqlens[b + 1];
      fill(t);
    }
    float cur[4];
    row(cur, u + W - 1);
    uint2 o;
    __nv_bfloat162* oh = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
    for (int e = 0; e < 4; e += 2) {
      float a0 = wt[e][W - 1] * cur[e], a1 = wt[e + 1][W - 1] * cur[e + 1];
#pragma unroll
      for (int j = 0; j < W - 1; ++j) {
        a0 += wt[e][j] * x[j][e];
        a1 += wt[e + 1][j] * x[j][e + 1];
      }
      oh[e / 2] = __floats2bfloat162_rn(a0 / (1.f + __expf(-a0)), a1 / (1.f + __expf(-a1)));
    }
    *reinterpret_cast<uint2*>(out + (size_t)t * C + c) = o;
#pragma unroll
    for (int j = 0; j < W - 2; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) x[j][e] = x[j + 1][e];
#pragma unroll
    for (int e = 0; e < 4; ++e) x[W - 2][e] = cur[e];
  }
}

// new state = last W-1 inputs of [old state | new tokens]
template <int W>
__global__ void kda_conv_state_kernel(const bf16* __restrict__ in, int ld_in,
                                      bf16* __restrict__ conv_state, const KdaSeqMeta meta,
                                      int C) {
  pdl_entry();
  const int b = blockIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const int t0 = meta.cu_seqlens[b], n = meta.cu_seqlens[b + 1] - t0;
  const bool has_state = meta.has_state[b];
  bf16* st = conv_state + (size_t)meta.slots[b] * meta.conv_ss + (size_t)c * (W - 1);
  float old[W - 1];
#pragma unroll
  for (int j = 0; j < W - 1; ++j) old[j] = has_state ? __bfloat162float(st[j]) : 0.f;
#pragma unroll
  for (int j = 0; j < W - 1; ++j) {
    const int off = n - (W - 1) + j;  // offset of the input that becomes state slot j
    const float v = off >= 0 ? __bfloat162float(in[(size_t)(t0 + off) * ld_in + c])
                             : old[W - 1 + off];
    st[j] = __float2bfloat16(v);
  }
}

void kda_conv(bf16* out, const bf16* in, int ld_in, const bf16* w, bf16* conv_state,
              const KdaSeqMeta& meta_in, int B, int C, int W, cudaStream_t st, int T, int t_begin) {
  if (B == 0 || T == 0) return;
  CHECK(W == 4, "kda conv specialized for kernel size 4");
  const KdaSeqMeta meta = kda_dense_strides(meta_in, 0, 0, C, W, 0);
  const bool aligned = C % 8 == 0 && ld_in % 8 == 0 && reinterpret_cast<uintptr_t>(in) % 16 == 0 &&
                       reinterpret_cast<uintptr_t>(out) % 16 == 0 &&
                       reinterpret_cast<uintptr_t>(w) % 16 == 0;
  // the 4-channel staged tiles (16-byte aligned rows), else a thread per (token, channel)
  constexpr int kTT4 = 16;
  if (aligned) {
    constexpr size_t smem4 = (size_t)(kTT4 + 4 - 1) * 256 * 8;
    static bool configured4[64] = {};
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (!configured4[dev]) {
      CUDA_CHECK(cudaFuncSetAttribute(kda_conv4s_kernel<4, kTT4>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem4));
      configured4[dev] = true;
    }
    launch_pdl(kda_conv4s_kernel<4, kTT4>, dim3(cdiv(C / 4, 256), cdiv(T, kTT4)), dim3(256), smem4, st, out,
               in, ld_in, w, conv_state, meta, B, t_begin, T, C);
  } else
    launch_pdl(kda_conv_kernel<4>, dim3(cdiv(C, 128), T), dim3(128), 0, st, out, in, ld_in, w,
               conv_state, meta, B, t_begin, C);
  launch_pdl(kda_conv_state_kernel<4>, dim3(cdiv(C, 128), B), dim3(128), 0, st, in, ld_in,
             conv_state, meta, C);
}

// ------------------------------------------------------------------ prefix-reuse snapshots
// grid (npairs * L, blocks): block-strided float4 copies of the state, then the conv window.
__global__ void kda_state_copy_kernel(float* __restrict__ state, size_t state_ls,
                                      bf16* __restrict__ conv, size_t conv_ls, int L,
                                      const int2* __restrict__ pairs, size_t sn, size_t cn) {
  pdl_entry();
  const int pi = blockIdx.x / L, l = blockIdx.x % L;
  const int2 pr = pairs[pi];
  const float4* src = reinterpret_cast<const float4*>(state + l * state_ls + (size_t)pr.x * sn);
  float4* dst = reinterpret_cast<float4*>(state + l * state_ls + (size_t)pr.y * sn);
  for (size_t i = blockIdx.y * (size_t)blockDim.x + threadIdx.x; i < sn / 4;
       i += (size_t)gridDim.y * blockDim.x)
    dst[i] = src[i];
  const bf16* cs = conv + l * conv_ls + (size_t)pr.x * cn;
  bf16* cd = conv + l * conv_ls + (size_t)pr.y * cn;
  for (size_t i = blockIdx.y * (size_t)blockDim.x + threadIdx.x; i < cn;
       i += (size_t)gridDim.y * blockDim.x)
    cd[i] = cs[i];
}

void kda_state_copy(float* state, size_t state_ls, bf16* conv, size_t conv_ls, int L,
                    const int2* pairs, int npairs, int H, int D, int C, cudaStream_t st) {
  if (npairs == 0) return;
  const size_t sn = (size_t)H * D * D, cn = (size_t)C * 3;
  launch_pdl(kda_state_copy_kernel, dim3(npairs * L, (unsigned)std::min<size_t>(64, cdiv(sn / 4, 256 * 4))),
             dim3(256), 0, st, state, state_ls, conv, conv_ls, L, pairs, sn, cn);
}

// grid (n, blocks)
__global__ void kda_snapshot_kernel(float* __restrict__ state, bf16* __restrict__ conv_state,
                                    const float* __restrict__ chunk_states,
                                    const bf16* __restrict__ in, int ld_in,
                                    const KdaSnap* __restrict__ e, size_t sn, int C, int Wm1) {
  pdl_entry();
  const KdaSnap s = e[blockIdx.x];
  const float4* src = reinterpret_cast<const float4*>(
      s.chunk >= 0 ? chunk_states + (size_t)s.chunk * sn : state + (size_t)s.slot * sn);
  float4* dst = reinterpret_cast<float4*>(state + (size_t)s.dst * sn);
  for (size_t i = blockIdx.y * (size_t)blockDim.x + threadIdx.x; i < sn / 4;
       i += (size_t)gridDim.y * blockDim.x)
    dst[i] = src[i];
  // conv window [C][W-1]: the W-1 raw inputs before the boundary, oldest first (conv_row < 0:
  // the boundary is the end of the sequence's tokens -> the slot's updated window)
  bf16* cd = conv_state + (size_t)s.dst * C * Wm1;
  const bf16* cs = conv_state + (size_t)s.slot * C * Wm1;
  for (size_t i = blockIdx.y * (size_t)blockDim.x + threadIdx.x; i < (size_t)C * Wm1;
       i += (size_t)gridDim.y * blockDim.x) {
    const int c = (int)(i / Wm1), j = (int)(i % Wm1);
    cd[i] = s.conv_row < 0 ? cs[i] : in[(size_t)(s.conv_row + j) * ld_in + c];
  }
}

void kda_snapshot(float* state, bf16* conv_state, const float* chunk_states, const bf16* in,
                  int ld_in, const KdaSnap* e, int n, int H, int D, int C, int W,
                  cudaStream_t st) {
  if (n == 0) return;
  const size_t sn = (size_t)H * D * D;
  launch_pdl(kda_snapshot_kernel, dim3(n, (unsigned)std::min<size_t>(64, cdiv(sn / 4, 256 * 4))),
             dim3(256), 0, st, state, conv_state, chunk_states, in, ld_in, e, sn, C, W - 1);
}

__global__ void kda_slice_xfer_kernel(KdaPoolPeers peers, size_t state_off, size_t conv_off,
                                      float* __restrict__ lstate, bf16* __restrict__ lconv,
                                      const KdaXfer* __restrict__ x, int dir, int h0, int Hf, int Hs,
                                      int D, int Wm1) {
  const KdaXfer e = x[blockIdx.x];
  const size_t sn = (size_t)Hs * D * D;
  float* ost = peers.state[e.owner] + state_off + ((size_t)e.oslot * Hf + h0) * D * D;
  float* lst = lstate + (size_t)e.lslot * sn;
  const float4* __restrict__ ss = reinterpret_cast<const float4*>(dir ? lst : ost);
  float4* __restrict__ sd = reinterpret_cast<float4*>(dir ? ost : lst);
  const size_t step = (size_t)gridDim.y * blockDim.x, n4 = sn / 4;
  // four 16-byte loads in flight per thread before their stores (one remote load at a time left the
  // copy bound by the NVLink round trip)
  size_t i = blockIdx.y * (size_t)blockDim.x + threadIdx.x;
  for (; i + 3 * step < n4; i += 4 * step) {
    const float4 a0 = ss[i], a1 = ss[i + step], a2 = ss[i + 2 * step], a3 = ss[i + 3 * step];
    sd[i] = a0;
    sd[i + step] = a1;
    sd[i + 2 * step] = a2;
    sd[i + 3 * step] = a3;
  }
  for (; i < n4; i += step) sd[i] = ss[i];
  // conv windows: the q, k and v channel ranges of the slice's heads (8 bf16 per vector)
  const size_t HDf = (size_t)Hf * D, HDs = (size_t)Hs * D, rl = HDs * Wm1 / 8;
  for (int r3 = 0; r3 < 3; ++r3) {
    bf16* oc = peers.conv[e.owner] + conv_off + ((size_t)e.oslot * 3 * HDf + r3 * HDf + (size_t)h0 * D) * Wm1;
    bf16* lc = lconv + ((size_t)e.lslot * 3 * HDs + r3 * HDs) * Wm1;
    const uint4* cs = reinterpret_cast<const uint4*>(dir ? lc : oc);
    uint4* cd = reinterpret_cast<uint4*>(dir ? oc : lc);
    for (size_t i = blockIdx.y * (size_t)blockDim.x + threadIdx.x; i < rl; i += step) cd[i] = cs[i];
  }
}

void kda_slice_xfer(const KdaPoolPeers& peers, size_t state_off, size_t conv_off, float* lstate,
                    bf16* lconv, const KdaXfer* x, int n, int dir, int h0, int Hf, int Hs, int D,
                    int Wm1, cudaStream_t st) {
  if (n == 0) return;
  CHECK(((size_t)Hs * D * Wm1) % 8 == 0 && ((size_t)h0 * D * Wm1) % 8 == 0 &&
            ((size_t)Hf * D * Wm1) % 8 == 0,
        "kda_slice_xfer: 16-byte conv ranges");
  // blocks per transfer: about one 16-byte vector per thread for the state slice
  const int by = (int)std::min<size_t>(128, std::max<size_t>(32, cdiv((size_t)Hs * D * D / 4, 256 * 2)));
  kda_slice_xfer_kernel<<<dim3(n, by), 256, 0, st>>>(peers, state_off, conv_off, lstate, lconv, x,
                                                     dir, h0, Hf, Hs, D, Wm1);
  CUDA_CHECK(cudaGetLastError());
}

// ------------------------------------------------------------------ recurrence
// grid (B, H); 256 threads. Thread (half, j) owns rows [half*64, half*64+64) of column v=j of
// the [K=128, V=128] fp32 state; the two halves combine their partial dot products via smem.
constexpr int kD = 128;
constexpr int kHalf = kD / 2;

__global__ void __launch_bounds__(2 * kD) kda_recurrent_kernel(
    bf16* __restrict__ o, const bf16* __restrict__ qkv, const bf16* __restrict__ beta_raw,
    int ld_beta, const bf16* __restrict__ g1, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ state, const KdaSeqMeta meta, int H,
    float lower_bound, float scale) {
  pdl_entry();
  const int b = blockIdx.x, h = blockIdx.y;
  const int j = threadIdx.x & (kD - 1), half = threadIdx.x >> 7;
  const int t0 = meta.cu_seqlens[b], t1 = meta.cu_seqlens[b + 1];
  const int slot = meta.slots[b];
  const int HD = H * kD;
  float* S = state + (size_t)slot * meta.state_ss + (size_t)h * kD * kD + (size_t)half * kHalf * kD;
  __shared__ float sq[kD], sk[kD], sdecay[kD];
  __shared__ float part[2][kD];
  __shared__ float red[2][4];
  float s[kHalf];
  const bool has_state = meta.has_state[b];
#pragma unroll
  for (int kk = 0; kk < kHalf; ++kk) s[kk] = has_state ? S[kk * kD + j] : 0.f;
  const float amp = expf(a_log[h]);
  const float bias = dt_bias[h * kD + j];
  const int lane = j & 31, wid = j >> 5;
  for (int t = t0; t < t1; ++t) {
    const size_t row = (size_t)t * 3 * HD;
    const float v = __bfloat162float(qkv[row + 2 * HD + h * kD + j]);
    const float beta = 1.f / (1.f + __expf(-__bfloat162float(beta_raw[(size_t)t * ld_beta + h])));
    // thread half 0 prepares q / decay, half 1 prepares k
    float x = __bfloat162float(qkv[row + (half ? HD : 0) + h * kD + j]);
    float ss = warp_sum(x * x);
    if (lane == 0) red[half][wid] = ss;
    if (half == 0) {
      const float g = __bfloat162float(g1[(size_t)t * HD + h * kD + j]);
      sdecay[j] = __expf(lower_bound / (1.f + __expf(-amp * (g + bias))));
    }
    __syncthreads();
    ss = red[half][0] + red[half][1] + red[half][2] + red[half][3];
    x = x / sqrtf(ss + 1e-6f);
    if (half == 0)
      sq[j] = x * scale;
    else
      sk[j] = x;
    __syncthreads();
    const float* dk = sdecay + half * kHalf;
    const float* kk_ = sk + half * kHalf;
    const float* qq_ = sq + half * kHalf;
    float acc = 0.f;
#pragma unroll
    for (int kk = 0; kk < kHalf; ++kk) {
      s[kk] *= dk[kk];
      acc += s[kk] * kk_[kk];
    }
    part[half][j] = acc;
    __syncthreads();
    const float u = beta * (v - (part[0][j] + part[1][j]));
    float out = 0.f;
#pragma unroll
    for (int kk = 0; kk < kHalf; ++kk) {
      s[kk] += kk_[kk] * u;
      out += s[kk] * qq_[kk];
    }
    __syncthreads();  // everyone has read part[] before it is reused
    part[half][j] = out;
    __syncthreads();
    if (half == 0) o[(size_t)t * HD + h * kD + j] = __float2bfloat16(part[0][j] + part[1][j]);
  }
  if (t1 > t0 || !has_state) {
#pragma unroll
    for (int kk = 0; kk < kHalf; ++kk) S[kk * kD + j] = s[kk];
  }
}

void kda_recurrent(bf16* o, const bf16* qkv, const bf16* beta_raw, int ld_beta, const bf16* g1,
                   const float* a_log, const float* dt_bias, float* state,
                   const KdaSeqMeta& meta_in, int B, int H, int D, float lower_bound,
                   cudaStream_t st) {
  if (B == 0) return;
  CHECK(D == kD, "kda recurrence specialized for head_dim 128");
  const KdaSeqMeta meta = kda_dense_strides(meta_in, H, D, 3 * H * D, 4, 0);
  dim3 grid(B, H);
  launch_pdl(kda_recurrent_kernel, dim3(grid), dim3(2 * kD), 0, st, o, qkv, beta_raw, ld_beta, g1, a_log, dt_bias,
                                                state, meta, H, lower_bound, rsqrtf((float)D));
}

// ------------------------------------------------------------------ fused decode step
__device__ __forceinline__ void kda_cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(
                   static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
               "l"(src));
}
__device__ __forceinline__ void kda_ldsm_x4(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(p))));
}
// byte offset of 16-byte chunk c (0..15) of row r in a [128, 128] bf16 tile (XOR swizzle by row)
__device__ __forceinline__ int kda_swz(int r, int c) { return (r * 16 + (c ^ (r & 7))) * 16; }
constexpr int kKdaGateSmem = 2 * kD * kD * 2;  // the head's f_b and g_b rows (64 KB)

__device__ __forceinline__ void kda_mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// The gate projection of a warp's 16 outputs x 8 tokens (padded): all A fragments (the weight tile, by
// ldmatrix) and B fragments (the inputs) first, then the 8 MMAs in k order -- the asm statements
// keep their program order, so loads interleaved with the MMAs serialized each step behind the
// previous MMA (same MMAs in the same order: bitwise)
__device__ __forceinline__ void kda_gate_mma(float* c, const uint8_t* wt, const bf16 (*gin)[kD + 8], int gj0,
                                             int lane) {
  const int gq = lane >> 2, gt = lane & 3;
  uint32_t af[kD / 16][4], bf[kD / 16][2];
#pragma unroll
  for (int ks = 0; ks < kD / 16; ++ks)
    kda_ldsm_x4(af[ks], wt + kda_swz(gj0 + (lane & 15), ks * 2 + (lane >> 4)));
#pragma unroll
  for (int ks = 0; ks < kD / 16; ++ks) {
    bf[ks][0] = *reinterpret_cast<const uint32_t*>(&gin[gq][ks * 16 + 2 * gt]);
    bf[ks][1] = *reinterpret_cast<const uint32_t*>(&gin[gq][ks * 16 + 2 * gt + 8]);
  }
#pragma unroll
  for (int ks = 0; ks < kD / 16; ++ks) kda_mma_bf16(c, af[ks], bf[ks][0], bf[ks][1]);
}
// (G8) kda_gate_mma with the B fragments in registers (loaded by the warp itself from the projections)
__device__ __forceinline__ void kda_gate_mma_r(float* c, const uint8_t* wt, const uint32_t (*bf)[2], int gj0,
                                               int lane) {
  uint32_t af[kD / 16][4];
#pragma unroll
  for (int ks = 0; ks < kD / 16; ++ks)
    kda_ldsm_x4(af[ks], wt + kda_swz(gj0 + (lane & 15), ks * 2 + (lane >> 4)));
#pragma unroll
  for (int ks = 0; ks < kD / 16; ++ks) kda_mma_bf16(c, af[ks], bf[ks][0], bf[ks][1]);
}

// grid (B, H), 512 threads: each head's 128 state rows split over 4 thread groups. Each sequence has NT
// consecutive new tokens (NT > 1: speculative verification of [last token, drafts...]). Only the
// state / conv window after the first token (always committed) is written back; for each following
// draft token the kernel stores a replay record (normalized k, u, decay) and its raw conv inputs. The
// next step first re-applies the `replay[b]` accepted drafts' records, so rejected drafts never reach
// the state. PRE: the gate projections f_b . f_a and g_b . g_a of every token were computed beforehand
// (g1 / g2 rows [T, H*D], as the prefill path's GEMMs) instead of in the kernel.
// Everything that does not come from this step's in_proj -- the state, the conv window, the replay
// records, the gate weights -- is loaded before the PDL wait (the previous step's writes are complete
// before this step's launch), so after the wait only the projection rows are read from memory. The
// replayed drafts are elementwise updates (their records carry u), the new tokens one pass (WY form).
// (the body of kda_decode_fused4_kernel and of its two-CTAs-per-SM form below)
template <int NT, bool PRE, bool ONE, bool PF>
__device__ __forceinline__ void kda_decode_fused4_body(
    bf16* __restrict__ out, const bf16* __restrict__ proj, int ld, const bf16* __restrict__ conv_w,
    bf16* __restrict__ conv_state, const bf16* __restrict__ f_b, const bf16* __restrict__ g_b,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias,
    const bf16* __restrict__ o_norm, float* __restrict__ state, const KdaSeqMeta meta, int H,
    float lower_bound, float scale, float eps, float* __restrict__ rec, bf16* __restrict__ crec,
    int ndr, const bf16* __restrict__ g1p, const bf16* __restrict__ g2p) {
  // (pdl_trigger() first thing in the kernels below; pdl_wait() further down)
  constexpr int W = 4;
  constexpr int KS = 4, kP = kD / KS;  // row groups, rows per group
  const int b = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
  const int j = tid & (kD - 1), part = tid >> 7, lane = tid & 31, wid = tid >> 5;
  const int HD = H * kD, C = 3 * HD;
  // metadata: written by the step's metadata kernel, ordered before every kernel of the step
  const int t0 = meta.cu_seqlens[b];
  const int slot = meta.slots[b];
  const bool has_state = meta.has_state[b];
  const int rc = (meta.replay && has_state) ? meta.replay[b] : 0;
  __shared__ __align__(16) float sq[NT][kD], sk[NT][kD], sv[NT][kD];
  __shared__ __align__(16) float sdecay[NT][kD], sg2[NT][kD];
  __shared__ __align__(16) bf16 sgin[2][8][kD + 8];  // gate inputs f_a / g_a, tokens padded to 8
  __shared__ float red[NT][4 * KS];
  __shared__ __align__(16) float srec[3][kKdaRec];
  __shared__ float sbeta[NT];
  // new tokens (WY form, below): the drafts' decayed keys / queries, the token-pair dots, k_0 . q_0,
  // the row groups' partial sums, u_0
  constexpr int ND = NT > 1 ? NT - 1 : 1;
  __shared__ __align__(16) float skq[ND][kD], sqq[ND][kD];
  __shared__ float sM[NT][NT], sN[NT][NT], sKQ;
  __shared__ float pw[KS - 1][2 * NT][kD];
  __shared__ float su0[kD];
  // ---- before the wait: this thread's state rows, the gate tiles and replay records (cp.async), the
  //      conv window, its draft inputs, the conv weights, a_log and o_norm. One CTA per SM (ONE): the
  //      small loads first, the state rows last (the first post-wait phase reads the small ones; the
  //      replays, later, the state); two CTAs per SM: the state rows first
  float* S = state + (size_t)slot * meta.state_ss + (size_t)h * kD * kD + (size_t)part * kP * kD;
  // PF (one CTA per SM, attention DP: the launcher's l2pf): the head's state (64 KB) and gate tiles
  // (2 x 32 KB) into L2 by bulk prefetches first thing; the per-thread requests below, a few hundred
  // lines in flight per SM, then turn over at L2 latency. There the in_proj GEMV (gemv_tma) holds
  // every SM until it ends, most CTAs start only then, and their loads before the wait are exposed;
  // under tensor parallelism the CTAs enter while the cuBLAS in_proj streams from HBM, where the
  // prefetches slow the step. A template parameter: the block costs time even when skipped
  if (PF && tid == 0) {
    const char* st0 = reinterpret_cast<const char*>(state + (size_t)slot * meta.state_ss + (size_t)h * kD * kD);
    if (has_state) {
#pragma unroll
      for (int q = 0; q < 4; ++q)
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(st0 + q * (kD * kD)),
                     "r"((uint32_t)(kD * kD))
                     : "memory");
    }
    if (!PRE) {
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(f_b + (size_t)h * kD * kD),
                   "r"((uint32_t)(kD * kD * 2))
                   : "memory");
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(g_b + (size_t)h * kD * kD),
                   "r"((uint32_t)(kD * kD * 2))
                   : "memory");
    }
  }
  float s[kP];
  auto ld_state = [&] {
#pragma unroll
    for (int kk = 0; kk < kP; ++kk) s[kk] = has_state ? S[kk * kD + j] : 0.f;
  };
  // gate projections on tensor cores: warp w owns gate w / 8 (f: decay, g: output gate), outputs
  // [16 (w % 8), +16) over the whole head dim; its A fragments (the weight tile) are loaded here
  const int gq = lane >> 2, gt = lane & 3;  // mma group / thread in group
  const int gsel = wid >> 3, gj0 = (wid & 7) * 16;
  // the head's two 128 x 128 gate weight tiles into shared memory (16-byte cp.async, ldmatrix
  // later; scattered 4-byte fragment loads arrive late)
  extern __shared__ __align__(128) uint8_t kda_dsm[];
  float gdt[2] = {0.f, 0.f};
  const float* Rp = rec + (size_t)slot * meta.rec_ss + (size_t)h * kKdaRec;  // record i at + i * H * kKdaRec
  auto ld_gates_recs = [&] {
    if (!PRE) {
      for (int i = tid; i < 2 * kD * 16; i += KS * kD) {
        const int g = i / (kD * 16), r = (i / 16) % kD, c = i % 16;
        kda_cp_async16(kda_dsm + g * kD * kD * 2 + kda_swz(r, c),
                       (g ? g_b : f_b) + ((size_t)h * kD + r) * kD + c * 8);
      }
      asm volatile("cp.async.commit_group;\n" ::);
      gdt[0] = dt_bias[h * kD + gj0 + gq];
      gdt[1] = dt_bias[h * kD + gj0 + gq + 8];
    }
    // the replay records by 16-byte cp.async (rows of 384 floats: 16-byte aligned), waited for with
    // the gate tiles: plain loads stored to shared memory held the threads before the PDL wait
    for (int q = tid; q < rc * (kKdaRec / 4); q += KS * kD) {
      const int r = q / (kKdaRec / 4), c4 = q % (kKdaRec / 4);
      kda_cp_async16(&srec[r][c4 * 4], Rp + (size_t)r * H * kKdaRec + c4 * 4);
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };
  // conv channel of this thread (threads < 3 * 128: q | k | v channel tid of this head)
  const bool conv_t = tid < 3 * kD;
  const int sec = tid / kD, cc = tid % kD;
  const int ch = sec * HD + h * kD + cc;
  // (the conv window, its draft inputs, the conv weights, a_log and o_norm are requested here and
  //  read after the wait: their consumers before it held every thread until they had come in from
  //  HBM behind the grid's state rows and gate tiles)
  float win[W - 1 + NT], cw[W];
  uint32_t wr[W - 1] = {}, crr[3] = {}, cwr[W] = {};  // raw bf16 bits
  float alog = 0.f;
  uint32_t onr = 0;
  auto ld_small = [&] {
    if (conv_t) {
      const unsigned short* stp =
          reinterpret_cast<const unsigned short*>(conv_state) + (size_t)slot * meta.conv_ss + (size_t)ch * (W - 1);
      if (has_state) {
#pragma unroll
        for (int q = 0; q < W - 1; ++q) wr[q] = stp[q];
      }
#pragma unroll
      for (int i = 0; i < 3; ++i)
        if (i < rc) crr[i] = reinterpret_cast<const unsigned short*>(crec)[kda_crec_off(meta, slot, i, sec, ch - sec * HD)];
#pragma unroll
      for (int q = 0; q < W; ++q) cwr[q] = reinterpret_cast<const unsigned short*>(conv_w)[(size_t)ch * W + q];
    }
    alog = a_log[h];
    onr = reinterpret_cast<const unsigned short*>(o_norm)[j];
  };
  if constexpr (ONE) {
    ld_small();
    ld_gates_recs();
    ld_state();
  } else {
    ld_state();
    ld_gates_recs();
    ld_small();
  }
  pdl_wait();
  // the consumers (after the wait: the empty asm keeps them from being scheduled above it)
  asm volatile("" : "+r"(wr[0]), "+r"(wr[1]), "+r"(wr[2]), "+r"(crr[0]), "+r"(crr[1]), "+r"(crr[2]),
               "+r"(cwr[0]), "+r"(cwr[1]), "+r"(cwr[2]), "+r"(cwr[3]), "+f"(alog), "+r"(onr));
  static_assert(W == 4, "kda fused4: conv width 4");
  if (conv_t) {
    float cr[3];
#pragma unroll
    for (int q = 0; q < W - 1; ++q) win[q] = __uint_as_float(wr[q] << 16);
#pragma unroll
    for (int i = 0; i < 3; ++i) cr[i] = __uint_as_float(crr[i] << 16);
#pragma unroll
    for (int i = 0; i < 3; ++i)  // window after the accepted drafts of the previous step
      if (i < rc) {
#pragma unroll
        for (int q = 0; q < W - 2; ++q) win[q] = win[q + 1];
        win[W - 2] = cr[i];
      }
#pragma unroll
    for (int q = 0; q < W; ++q) cw[q] = __uint_as_float(cwr[q] << 16);
  }
  const float amp = expf(alog);
  const float onw = __uint_as_float(onr << 16);
  // ---- this step's projections: betas, conv inputs, gate inputs
  if (tid < NT)
    sbeta[tid] = 1.f / (1.f + __expf(-__bfloat162float(proj[(size_t)(t0 + tid) * ld + 3 * HD + h])));
  if (conv_t) {
#pragma unroll
    for (int i = 0; i < NT; ++i) win[W - 1 + i] = __bfloat162float(proj[(size_t)(t0 + i) * ld + ch]);
  }
  if (PRE) {
    if (tid >= 3 * kD) {
      const int jj = tid - 3 * kD;
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        const size_t o = (size_t)(t0 + i) * HD + h * kD + jj;
        const float g1 = __bfloat162float(g1p[o]);
        sdecay[i][jj] = __expf(lower_bound / (1.f + __expf(-amp * (g1 + dt_bias[h * kD + jj]))));
        sg2[i][jj] = __bfloat162float(g2p[o]);
      }
    }
  } else if (tid >= 3 * kD) {
    const int jj = tid - 3 * kD;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      bf16 fv = __float2bfloat16(0.f), gv = fv;
      if (i < NT) {
        const bf16* row = proj + (size_t)(t0 + i) * ld;
        fv = row[3 * HD + H + jj];
        gv = row[3 * HD + H + kD + jj];
      }
      sgin[0][i][jj] = fv;
      sgin[1][i][jj] = gv;
    }
  }
  // conv outputs; the q / k threads (warps 0-3 / 4-7) also reduce the l2 norms' sums of squares per
  // warp (normalized after the gate MMA)
  float yqk[NT];
  if (conv_t) {
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      float acc = 0.f;
#pragma unroll
      for (int q = 0; q < W; ++q) acc += cw[q] * win[i + q];
      const float y = bf16r(acc / (1.f + __expf(-acc)));
      (sec == 0 ? sq : sec == 1 ? sk : sv)[i][cc] = y;
      yqk[i] = y;
      if (sec < 2) {
        const float ss = warp_sum(y * y);
        if (lane == 0) red[i][wid] = ss;
      }
    }
    bf16* dst = conv_state + (size_t)slot * meta.conv_ss + (size_t)ch * (W - 1);
#pragma unroll
    for (int q = 0; q < W - 1; ++q) dst[q] = __float2bfloat16(win[1 + q]);
#pragma unroll
    for (int i = 1; i < NT; ++i)
      crec[kda_crec_off(meta, slot, i - 1, sec, ch - sec * HD)] = __float2bfloat16(win[W - 1 + i]);
  }
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  __syncthreads();
  // ---- gate projections (tensor cores): C[j][token] = W[j][:] . in[token][:], fp32 accumulation
  if (!PRE) {
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    const uint8_t* wt = kda_dsm + gsel * kD * kD * 2;
    kda_gate_mma(c, wt, sgin[gsel], gj0, lane);
    // c0 / c1: output gj0 + gq, tokens 2 gt / 2 gt + 1; c2 / c3: output gj0 + gq + 8
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int n = 2 * gt + (e & 1), jo = gj0 + gq + (e >> 1) * 8;
      if (n < NT) {
        const float g = bf16r(c[e]);
        if (gsel == 0)
          sdecay[n][jo] = __expf(lower_bound / (1.f + __expf(-amp * (g + gdt[e >> 1]))));
        else
          sg2[n][jo] = g;
      }
    }
  }
  // ---- l2 norms of q (group 0) and k (group 1): each thread its own channel, in place, from the
  //      conv phase's warp sums (nothing reads sq / sk between the conv barrier and the next one)
  if (part < 2) {
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      const float n2 = red[i][part * 4] + red[i][part * 4 + 1] + red[i][part * 4 + 2] +
                       red[i][part * 4 + 3];
      const float nx = yqk[i] / sqrtf(n2 + 1e-6f);
      if (part)
        sk[i][j] = nx;
      else
        sq[i][j] = nx * scale;
    }
  }
  __syncthreads();
  // ---- the new tokens' terms that do not depend on the state (WY form, below), with
  //      Q_i = d_1 ... d_i and R_im = d_{m+1} ... d_i (per row): kQ_i = k_i Q_i, qQ_i = q_i Q_i
  //      (an element per thread), M_im = k_i . (R_im k_m) (m < i), N_im = q_i . (R_im k_m) (m <= i),
  //      one dot per warp; and k_0 . q_0 summed as the one-token form does (row groups in row
  //      order, then in order). The products are unrolled with predicates (no dependent loop).
  if constexpr (NT > 1) {
    for (int e = tid; e < 2 * (NT - 1) * kD; e += KS * kD) {
      const int v = e / kD, row = e % kD, i = 1 + (v >> 1);  // v even: kQ_i, odd: qQ_i
      float qr = 1.f;
#pragma unroll
      for (int l = 1; l < NT; ++l)
        if (l <= i) qr *= sdecay[l][row];
      if (v & 1)
        sqq[i - 1][row] = sq[i][row] * qr;
      else
        skq[i - 1][row] = sk[i][row] * qr;
    }
    constexpr int NM = NT * (NT - 1) / 2, NN = NM + NT - 1;
    static_assert(NM + NN < KS * kD / 32, "kda fused4: a warp per dot and one for k_0 . q_0");
    if (wid < NM + NN) {
      const bool isn = wid >= NM;
      int w = isn ? wid - NM : wid, i = 1;  // (i, m) of dot w: M by i then m < i, N by i then m <= i
      for (;;) {
        const int cnt = isn ? i + 1 : i;
        if (w < cnt) break;
        w -= cnt;
        ++i;
      }
      const int m = w;
      float acc = 0.f;
#pragma unroll
      for (int e = 0; e < kD / 32; ++e) {
        const int row = lane + 32 * e;
        float r = 1.f;
#pragma unroll
        for (int l = 1; l < NT; ++l)
          if (l > m && l <= i) r *= sdecay[l][row];
        acc += (isn ? sq[i][row] : sk[i][row]) * r * sk[m][row];
      }
      acc = warp_sum(acc);
      if (lane == 0) (isn ? sN : sM)[i][m] = acc;
    }
  }
  if (wid == KS * kD / 32 - 1) {
    float kq = 0.f;
    if (lane < KS) {
#pragma unroll
      for (int kk = 0; kk < kP; ++kk) kq += sk[0][lane * kP + kk] * sq[0][lane * kP + kk];
    }
    const float k1 = __shfl_sync(0xffffffffu, kq, 1), k2 = __shfl_sync(0xffffffffu, kq, 2),
                k3 = __shfl_sync(0xffffffffu, kq, 3);
    if (lane == 0) sKQ = ((kq + k1) + k2) + k3;
  }
  __syncthreads();
  // ---- delta-rule updates of this thread's state rows: the accepted drafts of the previous
  //      step (replay records), then the new tokens
  // (the records carry each draft's u: the replays are elementwise, no dot products, no barriers)
  for (int i = 0; i < rc; ++i) {
    // 128-bit reads of the broadcast vectors (kKdaRec = 384 floats: 16-byte aligned rows)
    const float4* dk4 = reinterpret_cast<const float4*>(srec[i] + 2 * kD + part * kP);  // decay
    const float4* kk4 = reinterpret_cast<const float4*>(srec[i] + part * kP);           // k
    const float u = srec[i][kD + j];
#pragma unroll
    for (int k4 = 0; k4 < kP / 4; ++k4) {
      const float4 d = dk4[k4], kv = kk4[k4];
      s[4 * k4] *= d.x;
      s[4 * k4] += kv.x * u;
      s[4 * k4 + 1] *= d.y;
      s[4 * k4 + 1] += kv.y * u;
      s[4 * k4 + 2] *= d.z;
      s[4 * k4 + 2] += kv.z * u;
      s[4 * k4 + 3] *= d.w;
      s[4 * k4 + 3] += kv.w * u;
    }
  }
  // replay records of the draft tokens
  if (part == 0) {
#pragma unroll
    for (int i = 1; i < NT; ++i) {
      float* R = rec + (size_t)slot * meta.rec_ss + ((size_t)(i - 1) * H + h) * kKdaRec;
      R[j] = sk[i][j];
      R[2 * kD + j] = sdecay[i][j];  // (u after the substitution below)
    }
  }
  // ---- the new tokens in one pass (WY form). With x = d_0 S (the state decayed for token 0),
  //      token i's decayed state is Q_i x + sum_{m<i} R_im k_m u_m, so
  //        u_i = beta_i (v_i - kQ_i . x - sum_{m<i} M_im u_m),  o_i = qQ_i . x + sum_{m<=i} N_im u_m
  //      (o = (S' + k u^T)^T q), and the committed state is x + k_0 u_0. Token 0's sums keep the
  //      one-token form's order (its output and the state are bitwise those of the serial loop);
  //      the drafts' are reassociated. One barrier for the row groups' partial sums, one for u_0.
  const float4* dk4 = reinterpret_cast<const float4*>(sdecay[0] + part * kP);
  const float4* kk4 = reinterpret_cast<const float4*>(sk[0] + part * kP);
  const float4* qq4 = reinterpret_cast<const float4*>(sq[0] + part * kP);
  float acc[2 * NT];  // kQ_i . x, qQ_i . x (i = 0: k_0, q_0) over this group's rows
#pragma unroll
  for (int v = 0; v < 2 * NT; ++v) acc[v] = 0.f;
#pragma unroll
  for (int k4 = 0; k4 < kP / 4; ++k4) {
    const float4 d = dk4[k4], kv = kk4[k4], qv = qq4[k4];
    const float dd[4] = {d.x, d.y, d.z, d.w}, kd[4] = {kv.x, kv.y, kv.z, kv.w},
                qd[4] = {qv.x, qv.y, qv.z, qv.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      s[4 * k4 + e] *= dd[e];
      acc[0] += s[4 * k4 + e] * kd[e];
      acc[1] += s[4 * k4 + e] * qd[e];
    }
#pragma unroll
    for (int i = 1; i < NT; ++i) {
      const float4 a = reinterpret_cast<const float4*>(skq[i - 1] + part * kP)[k4];
      const float4 c = reinterpret_cast<const float4*>(sqq[i - 1] + part * kP)[k4];
      const float ad[4] = {a.x, a.y, a.z, a.w}, cd[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        acc[2 * i] += s[4 * k4 + e] * ad[e];
        acc[2 * i + 1] += s[4 * k4 + e] * cd[e];
      }
    }
  }
  if (part > 0) {
#pragma unroll
    for (int v = 0; v < 2 * NT; ++v) pw[part - 1][v][j] = acc[v];
  }
  __syncthreads();
  float ov[NT];
#pragma unroll
  for (int i = 0; i < NT; ++i) ov[i] = 0.f;
  if (part == 0) {
    float tot[2 * NT], u[NT];
#pragma unroll
    for (int v = 0; v < 2 * NT; ++v) tot[v] = ((acc[v] + pw[0][v][j]) + pw[1][v][j]) + pw[2][v][j];
    u[0] = sbeta[0] * (sv[0][j] - tot[0]);
    ov[0] = bf16r(tot[1] + sKQ * u[0]);
#pragma unroll
    for (int i = 1; i < NT; ++i) {
      float a = tot[2 * i], o = tot[2 * i + 1];
#pragma unroll
      for (int m = 0; m < i; ++m) a += sM[i][m] * u[m];
      u[i] = sbeta[i] * (sv[i][j] - a);
#pragma unroll
      for (int m = 0; m <= i; ++m) o += sN[i][m] * u[m];
      ov[i] = bf16r(o);
      rec[(size_t)slot * meta.rec_ss + ((size_t)(i - 1) * H + h) * kKdaRec + kD + j] = u[i];  // (the draft's record)
    }
    su0[j] = u[0];
  }
  __syncthreads();
  // ---- gated RMSNorm of the head outputs (group 0 threads: warps 0-3): the sums of squares, then
  //      the committed state's stores (every thread), which drain while group 0 writes the outputs
  if (part == 0) {
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      const float ss = warp_sum(ov[i] * ov[i]);
      if (lane == 0) red[i][wid] = ss;
    }
  }
  {  // the state after the committed token
    const float u = su0[j];
#pragma unroll
    for (int k4 = 0; k4 < kP / 4; ++k4) {
      const float4 kv = kk4[k4];
      s[4 * k4] += kv.x * u;
      s[4 * k4 + 1] += kv.y * u;
      s[4 * k4 + 2] += kv.z * u;
      s[4 * k4 + 3] += kv.w * u;
    }
#pragma unroll
    for (int kk = 0; kk < kP; ++kk) S[kk * kD + j] = s[kk];
  }
  __syncthreads();
  if (part == 0) {
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      const float n2 = red[i][0] + red[i][1] + red[i][2] + red[i][3];
      const float r = rsqrtf(n2 / kD + eps);
      const float nv = bf16r(ov[i] * r * onw);
      out[(size_t)(t0 + i) * HD + h * kD + j] = __float2bfloat16(nv / (1.f + __expf(-sg2[i][j])));
    }
  }
}

template <int NT, bool PRE, bool PF>
__global__ void __launch_bounds__(4 * kD) kda_decode_fused4_kernel(
    bf16* __restrict__ out, const bf16* __restrict__ proj, int ld, const bf16* __restrict__ conv_w,
    bf16* __restrict__ conv_state, const bf16* __restrict__ f_b, const bf16* __restrict__ g_b,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias,
    const bf16* __restrict__ o_norm, float* __restrict__ state, const KdaSeqMeta meta, int H,
    float lower_bound, float scale, float eps, float* __restrict__ rec, bf16* __restrict__ crec,
    int ndr, const bf16* __restrict__ g1p, const bf16* __restrict__ g2p) {
  pdl_trigger();
  kda_decode_fused4_body<NT, PRE, true, PF>(out, proj, ld, conv_w, conv_state, f_b, g_b, a_log,
                                            dt_bias, o_norm, state, meta, H, lower_bound, scale, eps,
                                            rec, crec, ndr, g1p, g2p);  // (its pdl_wait(); follows the PDL-independent loads)
}

// Grids of more CTAs than SMs (attention DP: 64 local heads x several sequences): two CTAs per SM
// (64 registers per thread) instead of one, so 2 instead of 3 waves of CTAs at 5 sequences
template <int NT, bool PRE>
__global__ void __launch_bounds__(4 * kD, 2) kda_decode_fused4_mb2_kernel(
    bf16* __restrict__ out, const bf16* __restrict__ proj, int ld, const bf16* __restrict__ conv_w,
    bf16* __restrict__ conv_state, const bf16* __restrict__ f_b, const bf16* __restrict__ g_b,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias,
    const bf16* __restrict__ o_norm, float* __restrict__ state, const KdaSeqMeta meta, int H,
    float lower_bound, float scale, float eps, float* __restrict__ rec, bf16* __restrict__ crec,
    int ndr, const bf16* __restrict__ g1p, const bf16* __restrict__ g2p) {
  pdl_trigger();
  kda_decode_fused4_body<NT, PRE, false, false>(out, proj, ld, conv_w, conv_state, f_b, g_b, a_log,
                                                dt_bias, o_norm, state, meta, H, lower_bound, scale,
                                                eps, rec, crec, ndr, g1p, g2p);
  // (its pdl_wait(); follows the PDL-independent loads)
}


// ------------------------------------------------------------------ column-split fused decode
// For few sequences (B x H CTAs leave most SMs idle): the delta rule is independent per state
// column -- column j of S only meets this step's k, q, decays and betas (whole vectors) and its own
// v[j] -- so each head's 128 columns are split over a cluster of NS CTAs (VC = 128 / NS columns
// each). Thread (column jj, row group g) owns rows g + TPC r of column jj and the TPC threads of a
// column share a warp: every per-token dot product is a TPC-lane xor-shuffle reduction, and the
// replayed drafts and new tokens run without block barriers. Every CTA repeats the per-token
// prologue (conv of q / k and of its v columns, the decay gate of all rows, its columns' output
// gate, the l2 norms: the same arithmetic as kda_decode_fused4_kernel); writes of per-channel and
// per-row data (conv window, draft inputs, replay records) are split over the cluster. The output
// RMSNorm's sum of squares (per 32-column warp, as the 512-thread form) is exchanged through
// distributed shared memory. Not bitwise kda_decode_fused4_kernel: the state dot products are
// summed in another order.
// IEEE div.rn / sqrt.rn of N operands with no branch between the N chains: when every operand is in the range of
// the compiler's own fast path for '/' (magnitudes in [2^-60, 2^60]: reciprocal + Newton step + remainder correction,
// the instructions it emits ahead of its FCHK slow-path branch) or for sqrtf (positive normal >= 2^-101: rsqrt + one
// correction), that fast path's exact instruction sequence; otherwise the plain operators. Bitwise the plain operators
__device__ __forceinline__ bool kda_div_ok(float x) {
  const unsigned e = (__float_as_uint(x) >> 23) & 0xffu;
  return e >= 127u - 60u && e <= 127u + 60u;
}
__device__ __forceinline__ float kda_div_fast(float a, float b) {
  float q;
  asm("{\n.reg .f32 r0, e, r1, q0, rm, nb;\n"
      "neg.f32 nb, %2;\n"
      "rcp.approx.ftz.f32 r0, %2;\n"
      "fma.rn.f32 e, r0, nb, 0f3F800000;\n"
      "fma.rn.f32 r1, r0, e, r0;\n"
      "fma.rn.f32 q0, %1, r1, 0f00000000;\n"
      "fma.rn.f32 rm, q0, nb, %1;\n"
      "fma.rn.f32 %0, r1, rm, q0;\n}"
      : "=f"(q)
      : "f"(a), "f"(b));
  return q;
}
template <int N>
__device__ __forceinline__ void kda_div_n(float* q, const float* a, const float* b) {
  bool ok = true;
#pragma unroll
  for (int i = 0; i < N; ++i) ok = ok && kda_div_ok(a[i]) && kda_div_ok(b[i]);
  if (ok) {
#pragma unroll
    for (int i = 0; i < N; ++i) q[i] = kda_div_fast(a[i], b[i]);
  } else {
#pragma unroll
    for (int i = 0; i < N; ++i) q[i] = a[i] / b[i];
  }
}
__device__ __forceinline__ bool kda_sqrt_ok(float x) { return __float_as_uint(x) - 0x0d000000u <= 0x727fffffu; }
__device__ __forceinline__ float kda_sqrt_fast(float x) {
  float r;
  asm("{\n.reg .f32 y, h, hy, nh, e;\n"
      "rsqrt.approx.ftz.f32 y, %1;\n"
      "mul.ftz.f32 h, %1, y;\n"
      "mul.ftz.f32 hy, y, 0f3F000000;\n"
      "neg.f32 nh, h;\n"
      "fma.rn.f32 e, nh, h, %1;\n"
      "fma.rn.f32 %0, e, hy, h;\n}"
      : "=f"(r)
      : "f"(x));
  return r;
}
template <int N>
__device__ __forceinline__ void kda_sqrt_n(float* r, const float* x) {
  bool ok = true;
#pragma unroll
  for (int i = 0; i < N; ++i) ok = ok && kda_sqrt_ok(x[i]);
  if (ok) {
#pragma unroll
    for (int i = 0; i < N; ++i) r[i] = kda_sqrt_fast(x[i]);
  } else {
#pragma unroll
    for (int i = 0; i < N; ++i) r[i] = sqrtf(x[i]);
  }
}
// (G8) kda_div_n with the operands' range test as one running unsigned maximum (LOP3 + VIADDMNMX per operand, no
// branch between them): (bits & 0x7f800000) - (67 << 23) wraps above 120 << 23 for exponents below 67, so the maximum is
// <= 120 << 23 exactly when every exponent is in [67, 187] (= kda_div_ok); the same quotients
template <int N>
__device__ __forceinline__ void kda_div_nb(float* q, const float* a, const float* b) {
  unsigned m = 0u;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    m = max(m, (__float_as_uint(a[i]) & 0x7f800000u) - (67u << 23));
    m = max(m, (__float_as_uint(b[i]) & 0x7f800000u) - (67u << 23));
  }
  if (m <= (120u << 23)) {
#pragma unroll
    for (int i = 0; i < N; ++i) q[i] = kda_div_fast(a[i], b[i]);
  } else {
#pragma unroll
    for (int i = 0; i < N; ++i) q[i] = a[i] / b[i];
  }
}
// (G8) 1 + __expf(x) with ex2.approx.ftz: __expf's product x * log2(e), then ex2; the non-ftz form differs only where
// 2^t is subnormal (t < -126), and 1 + a subnormal or 1 + 0 both round to 1: bitwise 1.f + __expf(x)
__device__ __forceinline__ float kda_exp1p(float x) {
  float e;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(e) : "f"(x * 1.4426950216293334961f));
  return 1.f + e;
}
// (G8) bf16 round trips of pairs (cvt.rn.bf16x2.f32: an ALU op instead of two F2F conversions; the same rounding)
__device__ __forceinline__ void kda_bf16r2(float& a, float& b) {
  const __nv_bfloat162 p = __floats2bfloat162_rn(a, b);
  a = __low2float(p);
  b = __high2float(p);
}
// (G8) the NT <= 4 tokens' warp sums by one reduce-scatter along warp_sum's xor tree (each kept partial gets the same
// pairs as warp_sum: bitwise; 6 shuffles instead of 4 x 5): lanes 8 i .. 8 i + 7 end with token i's sum
__device__ __forceinline__ float kda_warp_sum4(const float* v, int lane) {
  const bool h16 = (lane & 16) != 0, h8 = (lane & 8) != 0;
  const float k0 = (h16 ? v[2] : v[0]) + __shfl_xor_sync(0xffffffffu, h16 ? v[0] : v[2], 16);
  const float k1 = (h16 ? v[3] : v[1]) + __shfl_xor_sync(0xffffffffu, h16 ? v[1] : v[3], 16);
  float s = (h8 ? k1 : k0) + __shfl_xor_sync(0xffffffffu, h8 ? k0 : k1, 8);
#pragma unroll
  for (int o = 4; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  return s;
}
__device__ __forceinline__ void kda_bar_sync(int id, int n) {  // named barrier of n threads (whole warps)
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void kda_bar_arrive(int id, int n) {  // (its arrival only: the warp does not wait)
  asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void kda_cl_arrive() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void kda_cl_wait() {
  asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void kda_dsm_st(float* local_addr, int rank, float v) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(local_addr)), ra;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(a), "r"(rank));
  asm volatile("st.shared::cluster.f32 [%0], %1;\n" ::"r"(ra), "f"(v) : "memory");
}
// (the arrive without release semantics: barrier.cluster.arrive's default .release is a GPU-scope
// MEMBAR.ALL.GPU in SASS)
__device__ __forceinline__ void kda_cl_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
}
// a float to CTA `rank`'s shared memory at local_addr's offset, completing 4 bytes on that CTA's
// mbarrier at local_mbar's offset (no fence: the receiver waits on its mbarrier)
__device__ __forceinline__ void kda_dsm_st_async(float* local_addr, uint64_t* local_mbar, int rank, float v) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(local_addr));
  const uint32_t m = static_cast<uint32_t>(__cvta_generic_to_shared(local_mbar));
  uint32_t ra, rm;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(a), "r"(rank));
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(rm) : "r"(m), "r"(rank));
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];\n" ::"r"(ra),
               "r"(__float_as_uint(v)), "r"(rm)
               : "memory");
}
__device__ __forceinline__ void kda_mbar_init_tx(uint64_t* b, uint32_t bytes) {
  const uint32_t m = static_cast<uint32_t>(__cvta_generic_to_shared(b));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(m));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(m), "r"(bytes) : "memory");
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void kda_mbar_wait0(uint64_t* b) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "KDA_MBW_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n"
      "@!p bra KDA_MBW_%=;\n"
      "}\n" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(b)))
      : "memory");
}

__device__ __forceinline__ bf16 kda_ldbf(const bf16* p) {  // (L2: data a PDL predecessor writes)
  return __ushort_as_bfloat16(__ldcg(reinterpret_cast<const unsigned short*>(p)));
}
// (LP) reads of the TP in_proj's Lamport copy of the projections: a word is final once it is no longer the sentinel 0xFFFF
// (a NaN the in_proj never writes: its cvt.rn.bf16 conversions give the canonical NaN 0x7FFF); every poll an L2 load
__device__ __forceinline__ unsigned short kda_lp_u16(const bf16* p) {
  unsigned short v;
  asm volatile("ld.volatile.global.u16 %0, [%1];\n" : "=h"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ bf16 kda_lp_sent() { return __ushort_as_bfloat16((unsigned short)0xFFFFu); }
__device__ __forceinline__ bf16 kda_lp_ld(const bf16* p) {
  unsigned short v;
  do {
    v = kda_lp_u16(p);
  } while (v == 0xFFFFu);
  return __ushort_as_bfloat16(v);
}
// (N words a row of ld apart, polled together: one L2 round trip per round)
template <int N>
__device__ __forceinline__ void kda_lp_ldn(bf16* x, const bf16* p, int ld) {
  unsigned short v[N];
  bool all;
  do {
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = kda_lp_u16(p + (size_t)i * ld);
    all = true;
#pragma unroll
    for (int i = 0; i < N; ++i) all = all && v[i] != 0xFFFFu;
  } while (!all);
#pragma unroll
  for (int i = 0; i < N; ++i) x[i] = __ushort_as_bfloat16(v[i]);
}
// (8 words, until none is the sentinel: a zero 16-bit half of ~v marks one -- exact for "any half")
__device__ __forceinline__ uint4 kda_lp_ld16(const bf16* p) {
  uint4 v;
  for (;;) {
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p)
                 : "memory");
    const uint32_t a = ~v.x, b = ~v.y, c = ~v.z, d = ~v.w;
    const uint32_t m = ((a - 0x00010001u) & ~a) | ((b - 0x00010001u) & ~b) | ((c - 0x00010001u) & ~c) |
                       ((d - 0x00010001u) & ~d);
    if ((m & 0x80008000u) == 0u) return v;
  }
}

constexpr int kSplitThreads = 512;
// (kda_split_body.inc DT / DW1, NT = 4) the 15 token-pair dots by warps 8-15: warp 8 + w, w < 6, the pair M_im / N_im of
// (i, m) = (1,0) (2,0) (2,1) (3,0) (3,1) (3,2), which share R_im and k_m; warp 14 N_11 / N_22, warp 15 N_33. Per dot the dot
// warps' per-row products and order (R_im from 1.f in l order, then fmaf((x R_im), k_m, acc) over the lane's 4 rows), then one
// 2-value reduce-scatter along warp_sum's xor tree: each sum bitwise warp_sum's
template <int NT>
__device__ __forceinline__ void kda_dt_dots(const float (*sdecay)[kD], const float (*sk)[kD], const float (*sq)[kD],
                                            float (*sM)[NT], float (*sN)[NT], int lane, int wid) {
    const int w = wid - 8;
    float a0 = 0.f, a1 = 0.f;
    int i0, m0, i1, m1;
    bool n0, has1;
    if (w < 6) {
      constexpr unsigned kPairIdx = 0x00b73621u;  // (i | m << 2) of pair w: (1,0) (2,0) (2,1) (3,0) (3,1) (3,2)
      const unsigned pr = (kPairIdx >> (4 * w)) & 15u;
      i0 = i1 = pr & 3;
      m0 = m1 = pr >> 2;
      n0 = false;
      has1 = true;
#pragma unroll
      for (int e = 0; e < kD / 32; ++e) {
        const int row = lane + 32 * e;
        float r = 1.f;
#pragma unroll
        for (int l = 1; l < NT; ++l)
          if (l > m0 && l <= i0) r *= sdecay[l][row];
        const float km = sk[m0][row];
        a0 = __fmaf_rn(sk[i0][row] * r, km, a0);
        a1 = __fmaf_rn(sq[i0][row] * r, km, a1);
      }
    } else {
      i0 = m0 = w == 6 ? 1 : 3;
      i1 = m1 = 2;
      n0 = true;
      has1 = w == 6;
#pragma unroll
      for (int e = 0; e < kD / 32; ++e) {
        const int row = lane + 32 * e;
        a0 = __fmaf_rn(sq[i0][row], sk[i0][row], a0);  // (R_ii = 1: x * 1 = x)
        if (has1) a1 = __fmaf_rn(sq[i1][row], sk[i1][row], a1);
      }
    }
    {  // (lanes 0-15 keep value 0, lanes 16-31 value 1)
      const bool hi = lane & 16;
      const float send = hi ? a0 : a1, keep = hi ? a1 : a0;
      a0 = keep + __shfl_xor_sync(0xffffffffu, send, 16);
    }
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) a0 += __shfl_xor_sync(0xffffffffu, a0, o);
    if (lane == 0) (n0 ? sN : sM)[i0][m0] = a0;
    if (lane == 16 && has1) sN[i1][m1] = a0;
}

// the WY pass's token-pair dots of NT tokens: (i, m) of dot warp w in 4 bits at 4 w (i | m << 2): the M dots
// (m < i) by i then m, then the N dots (m <= i) by i then m
constexpr unsigned long long kda_dot_pairs(int nt) {
  unsigned long long v = 0;
  int w = 0;
  for (int i = 1; i < nt; ++i)
    for (int m = 0; m < i; ++m) v |= (unsigned long long)(i | m << 2) << (4 * w++);
  for (int i = 1; i < nt; ++i)
    for (int m = 0; m <= i; ++m) v |= (unsigned long long)(i | m << 2) << (4 * w++);
  return v;
}
// NS CTAs per head in a cluster (NS = 4: the 4-CTA form's specialized warps, kda_split_body.inc)
template <int NT, int NS>
__global__ void __cluster_dims__(1, 1, NS) __launch_bounds__(kSplitThreads, 1)
    kda_decode_split_kernel(bf16* __restrict__ out, const bf16* __restrict__ proj, int ld,
                            const bf16* __restrict__ conv_w, bf16* __restrict__ conv_state,
                            const bf16* __restrict__ f_b, const bf16* __restrict__ g_b,
                            const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                            const bf16* __restrict__ o_norm, float* __restrict__ state,
                            const KdaSeqMeta meta, int H, float lower_bound, float scale, float eps,
                            float* __restrict__ rec, bf16* __restrict__ crec, int ndr) {
  pdl_trigger();  // (its pdl_wait(); below, after the PDL-independent loads)
  constexpr bool LATE = false, G8 = false, C2 = false, LP = false;
  bf16* const lpw = nullptr;
  bf16* const lpfg = nullptr;
#include "kda_split_body.inc"
}
// the 4-CTA form at one sequence under tensor parallelism at most 96 registers, with only the shared memory it uses (58
// KB) and the largest carveout: a CTA of the KDA o_proj (gemv_pre_lean: 8 warps x 48 registers, 146 KB) fits beside each
// of its CTAs (registers bind per SM sub-partition, 16,384 each, warps by id % 4: 4 x 32 x 96 + 2 x 32 x 48 = 15,360;
// kda_decode_split_kernel's 4-CTA instances, at 107 - 112 registers, leave no room), and its dependents launch once the
// projections are in
// (C2: kda_split_body.inc's 2-column token pass at one sequence, with the dots by the other 8 warps, SUB and MD)
template <int NT, bool G8 = false, bool EARLY = false, bool C2 = false>
__global__ void __cluster_dims__(1, 1, 4) __maxnreg__(96)
    kda_decode_split_co_kernel(bf16* __restrict__ out, const bf16* __restrict__ proj, int ld,
                               const bf16* __restrict__ conv_w, bf16* __restrict__ conv_state,
                               const bf16* __restrict__ f_b, const bf16* __restrict__ g_b,
                               const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                               const bf16* __restrict__ o_norm, float* __restrict__ state,
                               const KdaSeqMeta meta, int H, float lower_bound, float scale, float eps,
                               float* __restrict__ rec, bf16* __restrict__ crec, int ndr) {
  if (EARLY) pdl_trigger();  // (EARLY: two or more sequences; else the late trigger in the body)
  constexpr int NS = 4;
  constexpr bool LATE = !EARLY, LP = false;
  bf16* const lpw = nullptr;
  bf16* const lpfg = nullptr;
#include "kda_split_body.inc"
}
// (LP) kda_decode_split_co_kernel<NT, true (G8), false (late trigger), C2> with this step's projections read from the TP
// in_proj's Lamport copy (kda_split_body.inc LP): proj = lpw = the copy (written by the in_proj; the CTAs poll the words
// they read and reset them to the sentinel after cluster barrier 2, once every CTA of the cluster has read them), lpfg = the
// other copy's f_a / g_a columns (rows 0-3, read by every CTA of the previous KDA layer), reset by CTA (0, 0, 0). The grid
// waits for the in_proj's only before its first global store. The same arithmetic as kda_decode_split_co_kernel (bitwise)
template <int NT, bool C2 = false>
__global__ void __cluster_dims__(1, 1, 4) __maxnreg__(96)
    kda_decode_split_lp_kernel(bf16* __restrict__ out, const bf16* proj, int ld,
                               const bf16* __restrict__ conv_w, bf16* __restrict__ conv_state,
                               const bf16* __restrict__ f_b, const bf16* __restrict__ g_b,
                               const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                               const bf16* __restrict__ o_norm, float* __restrict__ state,
                               const KdaSeqMeta meta, int H, float lower_bound, float scale, float eps,
                               float* __restrict__ rec, bf16* __restrict__ crec, int ndr, bf16* lpw, bf16* lpfg) {
  // (the late trigger in the body once the polls are done; pdl_wait(); after cluster barrier 2)
  constexpr int NS = 4;
  constexpr bool LATE = true, G8 = true, LP = true;
#include "kda_split_body.inc"
}

// (LP) whether kda_decode_fused takes the late co instance's G8 form for these arguments -- the only form that reads the TP
// in_proj's Lamport copy (the same conditions as the launcher below)
bool kda_decode_lp_path(int B, int H, int D, int tokens_per_seq, int ld, const void* proj, bool l2pf, bool pre) {
  if (B != 1 || D != kD || tokens_per_seq < 1 || tokens_per_seq > 4 || l2pf || pre) return false;
  int dev = 0, sms = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  return B * H * 4 <= sms && ld % 8 == 0 && H % 8 == 0 && (reinterpret_cast<uintptr_t>(proj) & 15) == 0;
}
void kda_decode_fused(bf16* out, const bf16* proj, int ld, const bf16* conv_w, bf16* conv_state,
                      const bf16* f_b, const bf16* g_b, const float* a_log, const float* dt_bias,
                      const bf16* o_norm, float* state, const KdaSeqMeta& meta_in, int B, int H,
                      int D, float lower_bound, float eps, cudaStream_t st, int tokens_per_seq,
                      float* rec, bf16* crec, int ndr, const bf16* g1, const bf16* g2,
                      bool l2pf, bf16* lpfg) {
  if (B == 0) return;
  const KdaSeqMeta meta = kda_dense_strides(meta_in, H, D, 3 * H * D, 4, ndr);
  // (LP: proj is the TP in_proj's Lamport copy; only the late co instance's G8 form reads it -- kda_decode_lp_path)
  CHECK(!lpfg || kda_decode_lp_path(B, H, D, tokens_per_seq, ld, proj, l2pf, g1 != nullptr),
        "kda fused decode: the Lamport copy needs the late co instance's G8 form");
  CHECK(D == kD, "kda fused decode specialized for head_dim 128");
  CHECK(tokens_per_seq >= 1 && tokens_per_seq <= 4, "kda fused decode: 1-4 tokens per sequence");
  CHECK(tokens_per_seq == 1 || (rec && crec && ndr >= tokens_per_seq - 1),
        "kda fused decode: draft tokens need replay records");
  CHECK((g1 == nullptr) == (g2 == nullptr), "kda fused decode: precomputed gates come in pairs");
  dim3 grid(B, H);
  const float scale = rsqrtf((float)D);
  // grids that fit on the SMs (B x H CTAs) get one CTA per SM: an unused dynamic shared memory
  // reservation keeps a second CTA off the SM (two CTAs of a small grid on one SM serialize
  // their token loops while other SMs idle)
  static int sms[64] = {};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  if (!sms[dev]) CUDA_CHECK(cudaDeviceGetAttribute(&sms[dev], cudaDevAttrMultiProcessorCount, dev));
  constexpr int kSpreadSmem = 120 * 1024;
  const size_t dsmem = B * H <= sms[dev] ? kSpreadSmem : 0;
  const bool pre = g1 != nullptr;
  // column split: a cluster of 4 CTAs per head while B x H x 4 CTAs fit on the SMs, one CTA per SM
  // (the 120 KB reservation); then the 2-CTA form while B x H x 2 fit; else one CTA per head (the
  // 512-thread form below)
  const int ns = pre ? 1 : (B * H * 4 <= sms[dev] ? 4 : (B * H * 2 <= sms[dev] ? 2 : 1));
  if (ns > 1) {
#define KDA_SPLIT_LAUNCH(NT, NS)                                                                \
  do {                                                                                          \
    static bool configured_s[64] = {};                                                          \
    if (!configured_s[dev]) {                                                                   \
      CUDA_CHECK(cudaFuncSetAttribute(kda_decode_split_kernel<NT, NS>,                          \
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, kSpreadSmem)); \
      configured_s[dev] = true;                                                                 \
    }                                                                                           \
    launch_pdl(kda_decode_split_kernel<NT, NS>, dim3(B, H, NS), dim3(kSplitThreads),            \
               (size_t)kSpreadSmem, st, out, proj, ld, conv_w, conv_state, f_b, g_b, a_log,    \
               dt_bias, o_norm, state, meta, H, lower_bound, scale, eps, rec, crec, ndr);       \
  } while (0)
  // (the co instance's dynamic shared memory: the gate tiles + the state slice of its 32 columns)
  constexpr size_t kCoSmem = 2 * (kD / 4) * kD * 2 + kD * (kD / 4 + 4) * 4;
#define KDA_SPLIT_LAUNCH_CO(NT, G8, EARLY)                                                      \
  do {                                                                                          \
    static bool configured_c[64] = {};                                                          \
    if (!configured_c[dev]) {                                                                   \
      CUDA_CHECK(cudaFuncSetAttribute(kda_decode_split_co_kernel<NT, G8, EARLY>,                           \
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kCoSmem)); \
      CUDA_CHECK(cudaFuncSetAttribute(kda_decode_split_co_kernel<NT, G8, EARLY>,                           \
                                      cudaFuncAttributePreferredSharedMemoryCarveout, 100));    \
      configured_c[dev] = true;                                                                 \
    }                                                                                           \
    launch_pdl(kda_decode_split_co_kernel<NT, G8, EARLY>, dim3(B, H, 4), dim3(kSplitThreads), kCoSmem, st, out, proj, ld, \
               conv_w, conv_state, f_b, g_b, a_log, dt_bias, o_norm, state, meta, H, lower_bound, scale, eps, rec, \
               crec, ndr);                                                                      \
  } while (0)
  // (the late co instance at one sequence: both C2 forms configured -- loaded -- whichever kC2 selects, so that the
  //  constant does not move the code placement)
#define KDA_CO_CFG(NT, G8, C2V)                                                                  \
  CUDA_CHECK(cudaFuncSetAttribute(kda_decode_split_co_kernel<NT, G8, false, C2V>,                \
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kCoSmem));    \
  CUDA_CHECK(cudaFuncSetAttribute(kda_decode_split_co_kernel<NT, G8, false, C2V>,                \
                                  cudaFuncAttributePreferredSharedMemoryCarveout, 100))
#define KDA_CO_RUN(NT, G8, C2V)                                                                  \
  launch_pdl(kda_decode_split_co_kernel<NT, G8, false, C2V>, dim3(B, H, 4), dim3(kSplitThreads), kCoSmem, st, out, proj, \
             ld, conv_w, conv_state, f_b, g_b, a_log, dt_bias, o_norm, state, meta, H, lower_bound, scale, eps, rec, crec, \
             ndr)
#define KDA_LP_CFG(NT, C2V)                                                                     \
  CUDA_CHECK(cudaFuncSetAttribute(kda_decode_split_lp_kernel<NT, C2V>,                          \
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kCoSmem));    \
  CUDA_CHECK(cudaFuncSetAttribute(kda_decode_split_lp_kernel<NT, C2V>,                          \
                                  cudaFuncAttributePreferredSharedMemoryCarveout, 100))
#define KDA_LP_RUN(NT, C2V)                                                                     \
  launch_pdl(kda_decode_split_lp_kernel<NT, C2V>, dim3(B, H, 4), dim3(kSplitThreads), kCoSmem, st, out, proj, ld,  \
             conv_w, conv_state, f_b, g_b, a_log, dt_bias, o_norm, state, meta, H, lower_bound, scale, eps, rec, crec, \
             ndr, const_cast<bf16*>(proj), lpfg)
  // (the LP instances configured -- loaded -- with the co ones, whether or not the call reads the Lamport copy)
#define KDA_SPLIT_LAUNCH_COL(NT, G8)                                                            \
  do {                                                                                          \
    static bool configured_l[64] = {};                                                          \
    if (!configured_l[dev]) {                                                                   \
      KDA_CO_CFG(NT, G8, false);                                                                \
      KDA_CO_CFG(NT, G8, true);                                                                 \
      if (G8) {                                                                                 \
        KDA_LP_CFG(NT, false);                                                                  \
        KDA_LP_CFG(NT, true);                                                                   \
      }                                                                                         \
      configured_l[dev] = true;                                                                 \
    }                                                                                           \
    if (lpfg && G8) {                                                                           \
      if (kC2)                                                                                  \
        KDA_LP_RUN(NT, true);                                                                   \
      else                                                                                      \
        KDA_LP_RUN(NT, false);                                                                  \
    } else if (kC2)                                                                             \
      KDA_CO_RUN(NT, G8, true);                                                                 \
    else                                                                                        \
      KDA_CO_RUN(NT, G8, false);                                                                \
  } while (0)
  // the late co instance's 2-column token pass at one sequence (kda_split_body.inc C2: the replays and the WY products
  // with 2 columns per thread on 256 threads, half their shared-memory wavefronts; the token-pair dots by the other 8 warps
  // meanwhile; the substitution's reads ahead of its stores; the replay count with the other per-sequence values;
  // bitwise): not taken
  constexpr bool kC2 = false;
  // one sequence under tensor parallelism (B = 1, 4 CTAs per head): the late-trigger co instance (attention DP keeps
  // the early trigger). The KDA o_proj's CTAs then stage their weights while this kernel runs instead of during the
  // in_proj's last stream, which they would slow; with two or more sequences the o_proj (gemv_pipe) gains from the
  // early launch: the co instance with the early trigger (G8 form).
  // G8 (kda_split_body.inc): the 4-CTA form's conv / gate phase with the gate warps loading their own tile rows and
  // their MMA's B fragments (no gate-input staging / barrier), branch-free quotient tests, the warp sums as one
  // reduce-scatter and packed bf16 round trips (bitwise); its gate warps copy the gate inputs by 16-byte cp.async
  const bool g8 = ld % 8 == 0 && H % 8 == 0 && (reinterpret_cast<uintptr_t>(proj) & 15) == 0;
#define KDA_SPLIT_NS(NT)                                                                        \
  do {                                                                                          \
    if (ns == 4 && B == 1 && !l2pf) {                                                           \
      if (g8)                                                                                   \
        KDA_SPLIT_LAUNCH_COL(NT, true);                                                         \
      else                                                                                      \
        KDA_SPLIT_LAUNCH_COL(NT, false);                                                        \
    } else if (ns == 4) {                                                                       \
      if (g8)                                                                                   \
        KDA_SPLIT_LAUNCH_CO(NT, true, true);                                                    \
      else                                                                                      \
        KDA_SPLIT_LAUNCH(NT, 4);                                                                \
    } else {                                                                                    \
      KDA_SPLIT_LAUNCH(NT, 2);                                                                  \
    }                                                                                           \
  } while (0)
    switch (tokens_per_seq) {
      case 1: KDA_SPLIT_NS(1); break;
      case 2: KDA_SPLIT_NS(2); break;
      case 3: KDA_SPLIT_NS(3); break;
      default: KDA_SPLIT_NS(4); break;
    }
#undef KDA_SPLIT_NS
#undef KDA_SPLIT_LAUNCH
#undef KDA_SPLIT_LAUNCH_CO
#undef KDA_SPLIT_LAUNCH_COL
#undef KDA_CO_CFG
#undef KDA_CO_RUN
#undef KDA_LP_CFG
#undef KDA_LP_RUN
    return;
  }
  // the gate tiles (64 KB); spreading: the 120 KB that keep one CTA per SM
  const size_t dsmem4 = std::max<size_t>(dsmem, pre ? 0 : (size_t)kKdaGateSmem);
  // grids larger than the SMs: two CTAs per SM (kda_decode_fused4_mb2_kernel)
  const bool two = !dsmem && B * H > sms[dev];
#define KDA_DECODE4_LAUNCH(NT, PRE)                                                             \
  do {                                                                                          \
    static bool configured4[64] = {};                                                           \
    if (!configured4[dev]) {                                                                    \
      CUDA_CHECK(cudaFuncSetAttribute(kda_decode_fused4_kernel<NT, PRE, false>,                 \
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, kSpreadSmem)); \
      CUDA_CHECK(cudaFuncSetAttribute(kda_decode_fused4_kernel<NT, PRE, true>,                  \
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, kSpreadSmem)); \
      CUDA_CHECK(cudaFuncSetAttribute(kda_decode_fused4_mb2_kernel<NT, PRE>,                    \
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, kSpreadSmem)); \
      configured4[dev] = true;                                                                  \
    }                                                                                           \
    if (two)                                                                                    \
      launch_pdl(kda_decode_fused4_mb2_kernel<NT, PRE>, grid, dim3(4 * kD), dsmem4, st, out,    \
                 proj, ld, conv_w, conv_state, f_b, g_b, a_log, dt_bias, o_norm, state, meta, H, \
                 lower_bound, scale, eps, rec, crec, ndr, g1, g2);                              \
    else if (l2pf)                                                                              \
      launch_pdl(kda_decode_fused4_kernel<NT, PRE, true>, grid, dim3(4 * kD), dsmem4, st, out,  \
                 proj, ld, conv_w, conv_state, f_b, g_b, a_log, dt_bias, o_norm, state, meta, H, \
                 lower_bound, scale, eps, rec, crec, ndr, g1, g2);                              \
    else                                                                                        \
      launch_pdl(kda_decode_fused4_kernel<NT, PRE, false>, grid, dim3(4 * kD), dsmem4, st, out, \
                 proj, ld, conv_w, conv_state, f_b, g_b, a_log, dt_bias, o_norm, state, meta, H, \
                 lower_bound, scale, eps, rec, crec, ndr, g1, g2);                              \
  } while (0)
  switch (tokens_per_seq) {
    case 1: if (pre) KDA_DECODE4_LAUNCH(1, true); else KDA_DECODE4_LAUNCH(1, false); break;
    case 2: if (pre) KDA_DECODE4_LAUNCH(2, true); else KDA_DECODE4_LAUNCH(2, false); break;
    case 3: if (pre) KDA_DECODE4_LAUNCH(3, true); else KDA_DECODE4_LAUNCH(3, false); break;
    default: if (pre) KDA_DECODE4_LAUNCH(4, true); else KDA_DECODE4_LAUNCH(4, false); break;
  }
#undef KDA_DECODE4_LAUNCH
}

// ------------------------------------------------------------------ gated output norm
// out = bf16(bf16(rmsnorm(o_head) * w) * sigmoid(g2)); one warp per kNormRows (token, head) rows, a
// lane's 4 elements of each by 8-byte accesses, every row's loads issued before the first row's math
// (one row per warp with 2-byte accesses left the kernel issue- then latency-bound)
constexpr int kNormRows = 4;
__global__ void kda_out_norm_kernel(bf16* __restrict__ out, const bf16* __restrict__ o,
                                    const bf16* __restrict__ g2, const bf16* __restrict__ w,
                                    int rows, float eps) {
  pdl_entry();
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int r0 = warp * kNormRows;
  if (r0 >= rows) return;
  uint2 xr[kNormRows], gr[kNormRows];
#pragma unroll
  for (int k = 0; k < kNormRows; ++k) {
    const int row = min(r0 + k, rows - 1);
    xr[k] = *reinterpret_cast<const uint2*>(o + (size_t)row * kD + lane * 4);
    gr[k] = *reinterpret_cast<const uint2*>(g2 + (size_t)row * kD + lane * 4);
  }
  const uint2 wr = *reinterpret_cast<const uint2*>(w + lane * 4);
  const __nv_bfloat162* wh = reinterpret_cast<const __nv_bfloat162*>(&wr);
  float wv[4];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const float2 wf = __bfloat1622float2(wh[i]);
    wv[2 * i] = wf.x;
    wv[2 * i + 1] = wf.y;
  }
#pragma unroll
  for (int k = 0; k < kNormRows; ++k) {
    if (r0 + k >= rows) break;
    const __nv_bfloat162* xh = reinterpret_cast<const __nv_bfloat162*>(&xr[k]);
    const __nv_bfloat162* gh = reinterpret_cast<const __nv_bfloat162*>(&gr[k]);
    float x[4], gv[4], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const float2 xf = __bfloat1622float2(xh[i]), gf = __bfloat1622float2(gh[i]);
      x[2 * i] = xf.x;
      x[2 * i + 1] = xf.y;
      gv[2 * i] = gf.x;
      gv[2 * i + 1] = gf.y;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) ss += x[i] * x[i];
    ss = warp_sum(ss);
    const float r = rsqrtf(ss / kD + eps);
    uint2 orr;
    __nv_bfloat162* oh = reinterpret_cast<__nv_bfloat162*>(&orr);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const float n0 = bf16r(x[2 * i] * r * wv[2 * i]), n1 = bf16r(x[2 * i + 1] * r * wv[2 * i + 1]);
      oh[i] = __halves2bfloat162(__float2bfloat16(n0 / (1.f + __expf(-gv[2 * i]))),
                                 __float2bfloat16(n1 / (1.f + __expf(-gv[2 * i + 1]))));
    }
    *reinterpret_cast<uint2*>(out + (size_t)(r0 + k) * kD + lane * 4) = orr;
  }
}

void kda_out_norm(bf16* out, const bf16* o, const bf16* g2, const bf16* w, int T, int H, int D,
                  float eps, cudaStream_t st) {
  if (T == 0) return;
  CHECK(D == kD, "kda norm specialized for head_dim 128");
  const int rows = T * H;
  launch_pdl(kda_out_norm_kernel, dim3(cdiv(rows, 8 * kNormRows)), dim3(256), 0, st, out, o, g2, w, rows, eps);
}

}  // namespace k
}  // namespace glm
