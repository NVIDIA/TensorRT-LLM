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
// BF16 GEMV for few rows (decode): D[M, N] = A[M, K] x W[N, K]^T with M <= 16 (MT = 8 or 16
// token columns, one or two MMA n-tiles). Every CTA owns ROWS
// consecutive weight rows and copies all of them (ROWS x K) into shared memory before the PDL
// wait: weights never depend on earlier kernels, and a CTA that becomes resident while the
// previous kernel runs (every kernel triggers its dependents at its start) streams its slice
// during that kernel. After the wait it loads the activations, warp w multiplies k-slice w of
// all rows on the tensor cores ("swap-AB": weight rows are the MMA M dimension, tokens the N
// dimension), and the 8 warps' partial sums are added in warp order (deterministic).
#include <algorithm>

#include "pdl.cuh"
#include "gemv_pre.h"
#include "lamport.cuh"

namespace glm {
namespace k {

namespace {
constexpr int kThreads = 256;  // 8 warps, one k-slice each
constexpr int kWarps = kThreads / 32;
constexpr int kSmemMax = 220 * 1024;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)),
               "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// byte offset of 16-byte chunk c of row r in a [rows, K] bf16 tile of rc chunks per row, the
// chunk index XOR-swizzled by the row (conflict-free ldmatrix over 8 consecutive rows)
__device__ __forceinline__ int swz(int r, int c, int rc) { return (r * rc + (c ^ (r & 7))) * 16; }

// weights [rows][K] + activations [M][K]; the warps' partials ([8][rows][mt]) reuse the weights'
// space
__host__ __device__ inline size_t wspace(int rows, int K, int mt) {
  const size_t w = (size_t)rows * K * 2, r = (size_t)kWarps * rows * mt * sizeof(float);
  return w > r ? w : r;
}
size_t smem_bytes(int rows, int K, int M) {
  return wspace(rows, K, M <= 8 ? 8 : 16) + (size_t)M * K * 2;
}

// The epilogue: the warps' partials (red: [kWarps][R][mt] floats) summed in warp order into D.
// With p.push (the next boundary's fused all-reduce): the same sums, one per thread with the 8
// columns of a 16-byte chunk of one row in 8 consecutive lanes, packed by shuffles; the chunk is
// stored and pushed into every rank's buffer of the call that follows (its epoch: the last
// call's + 1; read after the PDL wait).
template <int R>
__device__ __forceinline__ void gemv_store(const GemvPreArgs& p, const float* red, int mt, int n0,
                                           int tid) {
  if (p.push.state) {
    const uint32_t ep = next_epoch(__ldcg(p.push.state));
    const size_t off =
        (size_t)(ep % 3) * 4 * p.push.max_elems + (size_t)p.push.rank * p.push.max_elems;
    const int items = R * p.M;
    for (int i0 = 0; i0 < items; i0 += kThreads) {  // (warp-uniform: every lane shuffles)
      const int i = i0 + tid, e = i & 7, tok = (i >> 3) % p.M, rb = (i >> 3) / p.M;
      const int c0 = n0 + rb * 8;
      const bool ok = i < items && c0 < p.N;
      float v = 0.f;
      if (ok)
#pragma unroll
        for (int w = 0; w < kWarps; ++w) v += red[((size_t)w * R + rb * 8 + e) * mt + tok];
      const uint32_t h = __bfloat16_as_ushort(__float2bfloat16(v));
      const uint32_t w0 = h | (__shfl_down_sync(0xffffffffu, h, 1) << 16);  // (even lanes)
      const uint32_t w1 = __shfl_down_sync(0xffffffffu, w0, 2);
      const uint32_t w2 = __shfl_down_sync(0xffffffffu, w0, 4);
      const uint32_t w3 = __shfl_down_sync(0xffffffffu, w0, 6);
      if (ok && e == 0) {
        uint4 u = make_uint4(w0, w1, w2, w3);
        const size_t ee = (size_t)tok * p.ldd + c0;
        *reinterpret_cast<uint4*>(static_cast<bf16*>(p.D) + ee) = u;
        u.x = clear_neg_zero(u.x);
        u.y = clear_neg_zero(u.y);
        u.z = clear_neg_zero(u.z);
        u.w = clear_neg_zero(u.w);
#pragma unroll
        for (int q = 0; q < 4; ++q) *reinterpret_cast<uint4*>(p.push.buf[q] + off + ee) = u;
      }
    }
    return;
  }
  for (int i = tid; i < R * p.M; i += kThreads) {
    const int row = i / p.M, tok = i % p.M;
    if (n0 + row >= p.N) continue;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) v += red[((size_t)w * R + row) * mt + tok];
    if (p.d_f32)
      static_cast<float*>(p.D)[(size_t)blockIdx.y * p.sD + (size_t)tok * p.ldd + n0 + row] = v;
    else
      static_cast<bf16*>(p.D)[(size_t)blockIdx.y * p.sD + (size_t)tok * p.ldd + n0 + row] =
          __float2bfloat16(v);
  }
}
// (gemv_store with the push epoch loaded before the PDL wait: EP instances)
template <int R>
__device__ __forceinline__ void gemv_store_e(const GemvPreArgs& p, const float* red, int mt, int n0,
                                             int tid, uint32_t ep0) {
  if (p.push.state) {
    const uint32_t ep = next_epoch(ep0);
    const size_t off =
        (size_t)(ep % 3) * 4 * p.push.max_elems + (size_t)p.push.rank * p.push.max_elems;
    const int items = R * p.M;
    for (int i0 = 0; i0 < items; i0 += kThreads) {  // (warp-uniform: every lane shuffles)
      const int i = i0 + tid, e = i & 7, tok = (i >> 3) % p.M, rb = (i >> 3) / p.M;
      const int c0 = n0 + rb * 8;
      const bool ok = i < items && c0 < p.N;
      float v = 0.f;
      if (ok)
#pragma unroll
        for (int w = 0; w < kWarps; ++w) v += red[((size_t)w * R + rb * 8 + e) * mt + tok];
      const uint32_t h = __bfloat16_as_ushort(__float2bfloat16(v));
      const uint32_t w0 = h | (__shfl_down_sync(0xffffffffu, h, 1) << 16);  // (even lanes)
      const uint32_t w1 = __shfl_down_sync(0xffffffffu, w0, 2);
      const uint32_t w2 = __shfl_down_sync(0xffffffffu, w0, 4);
      const uint32_t w3 = __shfl_down_sync(0xffffffffu, w0, 6);
      if (ok && e == 0) {
        uint4 u = make_uint4(w0, w1, w2, w3);
        const size_t ee = (size_t)tok * p.ldd + c0;
        *reinterpret_cast<uint4*>(static_cast<bf16*>(p.D) + ee) = u;
        u.x = clear_neg_zero(u.x);
        u.y = clear_neg_zero(u.y);
        u.z = clear_neg_zero(u.z);
        u.w = clear_neg_zero(u.w);
#pragma unroll
        for (int q = 0; q < 4; ++q) *reinterpret_cast<uint4*>(p.push.buf[q] + off + ee) = u;
      }
    }
    return;
  }
  for (int i = tid; i < R * p.M; i += kThreads) {
    const int row = i / p.M, tok = i % p.M;
    if (n0 + row >= p.N) continue;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) v += red[((size_t)w * R + row) * mt + tok];
    if (p.d_f32)
      static_cast<float*>(p.D)[(size_t)blockIdx.y * p.sD + (size_t)tok * p.ldd + n0 + row] = v;
    else
      static_cast<bf16*>(p.D)[(size_t)blockIdx.y * p.sD + (size_t)tok * p.ldd + n0 + row] =
          __float2bfloat16(v);
  }
}
}  // namespace

template <int ROWS, int MT>
__global__ void __launch_bounds__(kThreads, 1) gemv_pre_kernel(const GemvPreArgs p) {
  if (!p.late_trigger) pdl_trigger();
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rc = p.K / 8;  // 16-byte chunks per row
  const int n0 = blockIdx.x * ROWS;
  const bf16* const W = p.W + (size_t)blockIdx.y * p.sW;  // (batched: this CTA's matrix)
  const bf16* const A = p.A + (size_t)blockIdx.y * p.sA;
  uint8_t* sw = sm;                                              // [ROWS][K] weights
  uint8_t* sx = sm + wspace(ROWS, p.K, MT);   // [M][K]
  float* red = reinterpret_cast<float*>(sm);  // [8 warps][ROWS][MT] partials, after the products
  // the CTA's weight rows (rows past N zero-filled), before the wait
  for (int i = tid; i < ROWS * rc; i += kThreads) {
    const int r = i / rc, c = i % rc;
    const bool ok = n0 + r < p.N;
    cp_async16(sw + swz(r, c, rc), ok ? W + (size_t)(n0 + r) * p.ldw + c * 8 : W, ok);
  }
  asm volatile("cp.async.commit_group;\n" ::);
  pdl_wait();
  if (p.late_trigger) pdl_trigger();
  // activations (the MMA's columns past M read row (column mod M): never stored)
  for (int i = tid; i < p.M * rc; i += kThreads) {
    const int r = i / rc, c = i % rc;
    cp_async16(sx + swz(r, c, rc), A + (size_t)r * p.lda + c * 8, true);
  }
  asm volatile("cp.async.commit_group;\n" ::);
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  __syncthreads();
  constexpr int MI = ROWS / 16, NJ = MT / 8;
  float acc[MI][NJ][4];
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int nj = 0; nj < NJ; ++nj) acc[mi][nj][0] = acc[mi][nj][1] = acc[mi][nj][2] = acc[mi][nj][3] = 0.f;
  const int ks_per = p.K / (16 * kWarps);  // 16-element k-steps of this warp's slice
  const int ks0 = warp * ks_per;
  for (int ks = ks0; ks < ks0 + ks_per; ++ks) {
    uint32_t bfr[NJ][2];
#pragma unroll
    for (int nj = 0; nj < NJ; ++nj)
      ldsm_x2(bfr[nj], sx + swz((nj * 8 + (lane & 7)) % p.M, ks * 2 + ((lane >> 3) & 1), rc));
#pragma unroll
    for (int mi = 0; mi < MI; ++mi) {
      uint32_t af[4];
      ldsm_x4(af, sw + swz(mi * 16 + (lane & 15), ks * 2 + (lane >> 4), rc));
#pragma unroll
      for (int nj = 0; nj < NJ; ++nj) mma_bf16(acc[mi][nj], af, bfr[nj]);
    }
  }
  // fragment element e: weight row mi * 16 + lane / 4 (+ 8 for e >= 2), token 2 * (lane % 4)
  // (+ 1 for odd e); the 8 warps' partials are summed in warp order
  __syncthreads();  // every warp is done with the weights (their space holds the partials)
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int nj = 0; nj < NJ; ++nj)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int row = mi * 16 + (lane >> 2) + (e >> 1) * 8, tok = nj * 8 + (lane & 3) * 2 + (e & 1);
        red[((size_t)warp * ROWS + row) * MT + tok] = acc[mi][nj][e];
      }
  __syncthreads();
  gemv_store<ROWS>(p, red, MT, n0, tid);
}

// rows per CTA: the fewest (multiple of 16, at most 64) that keep the grid within max_ctas
static int pick_rows(int N, int max_ctas) {
  for (int rows = 16; rows <= 64; rows += 16)
    if ((int)cdiv(N, rows) <= max_ctas) return rows;
  return 0;
}

bool gemv_pre_fits(int M, int N, int K, int max_ctas, int batch) {
  if (M < 1 || M > kGemvPreMaxRows || K % (16 * kWarps) != 0 || N < 1 || batch < 1) return false;
  const int rows = pick_rows(N, max_ctas / batch);
  return rows > 0 && smem_bytes(rows, K, M) <= (size_t)kSmemMax;
}

template <int ROWS, int MT>
static void launch(const GemvPreArgs& a, cudaStream_t st, int carveout = -1) {
  // (the all-reduce push addresses one output matrix: never with a head-batched launch)
  CHECK(a.batch == 1 || !a.push.state, "gemv_pre: all-reduce push with a batched launch");
  const size_t smem = smem_bytes(ROWS, a.K, a.M);
  static bool configured[64] = {};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  if (!configured[dev]) {
    CUDA_CHECK(cudaFuncSetAttribute(gemv_pre_kernel<ROWS, MT>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemMax));
    configured[dev] = true;
  }
  if (carveout >= 0) {
    // (a launch attribute: this launch's SMs take the preferred shared-memory carveout, the instance's other launches
    // keep theirs)
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(cdiv(a.N, ROWS), a.batch);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = st;
    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    attr[1].id = cudaLaunchAttributePreferredSharedMemoryCarveout;
    attr[1].val.sharedMemCarveout = (unsigned)carveout;
    cfg.attrs = attr;
    cfg.numAttrs = 2;
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, gemv_pre_kernel<ROWS, MT>, a));
    return;
  }
  launch_pdl(gemv_pre_kernel<ROWS, MT>, dim3(cdiv(a.N, ROWS), a.batch), dim3(kThreads), smem, st, a);
}

// ------------------------------------------------------------------ 32-row CTAs, TMA-staged rows (KDA o_proj)
constexpr int kLRows = 32, kLK = 2048;
constexpr int kLRS = kLK * 2 + 16;  // padded row stride (weights and activations: conflict-free ldmatrix)
__device__ __forceinline__ void l_bulk(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
                   smem_u32(dst)),
               "l"(src), "r"(bytes), "r"(smem_u32(b))
               : "memory");
}
__host__ __device__ inline size_t lean_smem(int M) {
  return (size_t)kLRows * kLRS + (size_t)M * kLRS + 16;
}
// 8 warps, warp w: k-slice w of the 32 rows (gemv_pre's products and warp-order sums); the rows staged by TMA bulk
// copies (one per row, padded by 16 bytes) on one mbarrier before the PDL wait; after it the activation rows: AB, one
// TMA bulk copy per token row (4 KB) on a second mbarrier -- the 128 CTAs read the same 8-16 KB that the KDA has just
// written, and 16-byte cp.async requests (8 per 128-byte line and CTA) queue at those lines' L2 slices; else each warp's
// slice of the rows by its own cp.async. EP (the pushed form): the push epoch requested by thread 0 before the PDL wait -- it
// is final at this grid's start: the boundary that last advanced it completed before this kernel's PDL predecessor (the
// KDA) passed its own wait, and the next one is launched only after this kernel's trigger (as c16's kPre call state) --
// instead of after the products, where its L2 round trip would sit on the tail before the push stores
template <bool AB, bool EP = false>
__global__ void __launch_bounds__(kThreads, 1) gemv_pre_lean_kernel(const GemvPreArgs p) {
  if (!p.late_trigger) pdl_trigger();
  extern __shared__ __align__(128) uint8_t sm[];
  constexpr int R = kLRows, MI = R / 16, MT = 8, RB = kLK * 2, KSS = kLK / 16 / kWarps;  // k-steps per slice
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int n0 = blockIdx.x * R;
  uint8_t* sw = sm;                                      // [R][kLRS] weights
  uint8_t* sx = sm + (size_t)R * kLRS;                   // [M][kLRS] activations
  uint64_t* wbar = reinterpret_cast<uint64_t*>(sx + (size_t)p.M * kLRS);  // [0] weights, [1] activations (AB)
  float* red = reinterpret_cast<float*>(sm);             // [8 slices][R][MT] partials, after the products
  const int rows = min(R, p.N - n0);
  if (tid == 0) {
    asm volatile("mbarrier.init.shared.b64 [%0], 1;\n" ::"r"(smem_u32(wbar)));
    if (AB) asm volatile("mbarrier.init.shared.b64 [%0], 1;\n" ::"r"(smem_u32(wbar + 1)));
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(smem_u32(wbar)),
                 "r"((uint32_t)rows * RB)
                 : "memory");
    if (AB)
      asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(smem_u32(wbar + 1)),
                   "r"((uint32_t)p.M * RB)
                   : "memory");
  }
  __syncthreads();
  if (warp == 0)
    for (int r = lane; r < rows; r += 32) l_bulk(sw + (size_t)r * kLRS, p.W + (size_t)(n0 + r) * p.ldw, RB, wbar);
  // (EP: the push epoch requested here, into a register -- nothing waits for it before the epilogue, which publishes it)
  __shared__ uint32_t s_ep;
  uint32_t ep_r = 0;
  if constexpr (EP) {
    if (tid == 0 && p.push.state) ep_r = __ldcg(p.push.state);
  }
  pdl_wait();
  if (p.late_trigger) pdl_trigger();
  const uint8_t* ag = reinterpret_cast<const uint8_t*>(p.A);
  if constexpr (AB) {
    // the activation rows, one bulk copy each by lanes of warp 0 (the TMA reads through L2: the PDL predecessor's data)
    if (warp == 0 && lane < p.M) l_bulk(sx + (size_t)lane * kLRS, ag + (size_t)lane * p.lda * 2, RB, wbar + 1);
    uint32_t ok = 0;
    while (!ok)
      asm volatile("{ .reg .pred P; mbarrier.try_wait.parity.shared.b64 P, [%1], 0; selp.u32 %0, 1, 0, P; }\n"
                   : "=r"(ok)
                   : "r"(smem_u32(wbar + 1))
                   : "memory");
  } else {
    // the activations of this warp's k-slice (512 bytes of each token row), its own copies
    for (int i = lane; i < p.M * 32; i += 32) {
      const int m = i >> 5, off = warp * 512 + (i & 31) * 16;
      cp_async16(sx + (size_t)m * kLRS + off, ag + (size_t)m * p.lda * 2 + off, true);
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncwarp();
  }
  {
    uint32_t ok = 0;
    while (!ok)
      asm volatile("{ .reg .pred P; mbarrier.try_wait.parity.shared.b64 P, [%1], 0; selp.u32 %0, 1, 0, P; }\n"
                   : "=r"(ok)
                   : "r"(smem_u32(wbar))
                   : "memory");
  }
  float acc[MI][4];
#pragma unroll
  for (int mi = 0; mi < MI; ++mi) acc[mi][0] = acc[mi][1] = acc[mi][2] = acc[mi][3] = 0.f;
  const uint32_t xa = smem_u32(sx) + (uint32_t)((lane & 7) % p.M) * kLRS + ((lane >> 3) & 1) * 16;
  const uint32_t wa = smem_u32(sw) + (uint32_t)(lane & 15) * kLRS + (lane >> 4) * 16;
#pragma unroll 4
  for (int q = 0; q < KSS; ++q) {
    const int ks = warp * KSS + q;  // 16-element k-step (gemv_pre's order)
    uint32_t bfr[2];
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(bfr[0]), "=r"(bfr[1])
                 : "r"(xa + ks * 32));
#pragma unroll
    for (int mi = 0; mi < MI; ++mi) {
      uint32_t af[4];
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(af[0]), "=r"(af[1]), "=r"(af[2]), "=r"(af[3])
                   : "r"(wa + (uint32_t)mi * 16 * kLRS + ks * 32));
      mma_bf16(acc[mi], af, bfr);
    }
  }
  __syncthreads();  // every warp is done with the weights (their space holds the partials)
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int row = mi * 16 + (lane >> 2) + (e >> 1) * 8, tok = (lane & 3) * 2 + (e & 1);
      red[((size_t)warp * R + row) * MT + tok] = acc[mi][e];
    }
  if constexpr (EP) {
    if (tid == 0) s_ep = ep_r;
  }
  __syncthreads();
  // the warps' partials summed in warp order (gemv_store<R>'s code)
  if (p.push.state) {
    const uint32_t ep = next_epoch(EP ? s_ep : __ldcg(p.push.state));
    const size_t off = (size_t)(ep % 3) * 4 * p.push.max_elems + (size_t)p.push.rank * p.push.max_elems;
    const int items = R * p.M;
    for (int i0 = 0; i0 < items; i0 += kThreads) {  // (warp-uniform: every lane shuffles)
      const int i = i0 + tid, e = i & 7, tok = (i >> 3) % p.M, rb = (i >> 3) / p.M;
      const int c0 = n0 + rb * 8;
      const bool ok = i < items && c0 < p.N;
      float v = 0.f;
      if (ok)
#pragma unroll
        for (int w = 0; w < kWarps; ++w) v += red[((size_t)w * R + rb * 8 + e) * MT + tok];
      const uint32_t hh = __bfloat16_as_ushort(__float2bfloat16(v));
      const uint32_t w0 = hh | (__shfl_down_sync(0xffffffffu, hh, 1) << 16);
      const uint32_t w1 = __shfl_down_sync(0xffffffffu, w0, 2);
      const uint32_t w2 = __shfl_down_sync(0xffffffffu, w0, 4);
      const uint32_t w3 = __shfl_down_sync(0xffffffffu, w0, 6);
      if (ok && e == 0) {
        uint4 u = make_uint4(w0, w1, w2, w3);
        const size_t ee = (size_t)tok * p.ldd + c0;
        *reinterpret_cast<uint4*>(static_cast<bf16*>(p.D) + ee) = u;
        u.x = clear_neg_zero(u.x);
        u.y = clear_neg_zero(u.y);
        u.z = clear_neg_zero(u.z);
        u.w = clear_neg_zero(u.w);
#pragma unroll
        for (int q = 0; q < 4; ++q) *reinterpret_cast<uint4*>(p.push.buf[q] + off + ee) = u;
      }
    }
    return;
  }
  for (int i = tid; i < R * p.M; i += kThreads) {
    const int row = i / p.M, tok = i % p.M;
    if (n0 + row >= p.N) continue;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) v += red[((size_t)w * R + row) * MT + tok];
    static_cast<bf16*>(p.D)[(size_t)tok * p.ldd + n0 + row] = __float2bfloat16(v);
  }
}

bool gemv_pre_lean_fits(int M, int N, int K) { return M >= 1 && M <= 8 && N >= 1 && K == kLK; }

void gemv_pre_lean(const GemvPreArgs& a, cudaStream_t st) {
  CHECK(gemv_pre_lean_fits(a.M, a.N, a.K) && a.batch == 1 && !a.d_f32, "gemv_pre_lean: shapes not supported");
  CHECK(a.lda % 8 == 0 && a.ldw % 8 == 0, "gemv_pre_lean: 16-byte aligned rows needed");
  // the activation rows as TMA bulk copies (AB; else 16-byte cp.async by each warp)
  constexpr bool kBulkRows = true;
  static bool configured[64] = {};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  if (!configured[dev]) {
    // (the largest shared-memory carveout, so that a CTA of the 16-CTA cluster boundary that follows fits beside)
    for (auto* fn : {gemv_pre_lean_kernel<false>, gemv_pre_lean_kernel<true>, gemv_pre_lean_kernel<true, true>}) {
      CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)lean_smem(8)));
      CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    }
    configured[dev] = true;
  }
  // (the pushed form: its push epoch loaded before the PDL wait, EP)
  if (kBulkRows && a.push.state)
    launch_pdl(gemv_pre_lean_kernel<true, true>, dim3(cdiv(a.N, kLRows)), dim3(kThreads), lean_smem(a.M), st, a);
  else if (kBulkRows)
    launch_pdl(gemv_pre_lean_kernel<true>, dim3(cdiv(a.N, kLRows)), dim3(kThreads), lean_smem(a.M), st, a);
  else
    launch_pdl(gemv_pre_lean_kernel<false>, dim3(cdiv(a.N, kLRows)), dim3(kThreads), lean_smem(a.M), st, a);
}

// ------------------------------------------------------------------ partly preloaded GEMV
// For weights larger than the grid's shared memory (KDA in_proj: 6416 x 4096 bf16 per GPU): CTA b
// owns R = 16 MI rows, streamed in k-chunks of kKc columns through a ring of NB shared-memory
// buffers (weights + the activations' chunk). Before the PDL wait -- during the previous kernel,
// a latency-bound mHC boundary that leaves HBM idle -- the first NB chunks are copied into the
// ring; after the wait the GEMV streams the rest. Same products (warp w: k-slice w of every
// chunk) and warp-order sums as gemv_pre, but the k-slices interleave: not bitwise gemv_pre /
// cuBLASLt.
constexpr int kKc = 512;
constexpr int kStreamMaxCtas = 116;  // one wave beside the previous kernel's CTAs (and the next's)

__host__ __device__ inline size_t stream_buf_bytes(int R, int M) {
  return ((size_t)(R + M) * kKc * 2 + 127) / 128 * 128;
}

template <int MI, int MT, int NB>
__global__ void __launch_bounds__(kThreads, 1) gemv_stream_kernel(const GemvPreArgs p) {
  if (!p.late_trigger) pdl_trigger();
  extern __shared__ __align__(128) uint8_t sm[];
  constexpr int R = MI * 16, NJ = MT / 8;
  constexpr int rcw = kKc / 8;  // 16-byte chunks of a row in one k-chunk
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int n0 = blockIdx.x * R;
  const int nch = p.K / kKc;
  const size_t bb = stream_buf_bytes(R, p.M);
  auto wbuf = [&](int b) { return sm + (size_t)b * bb; };
  auto xbuf = [&](int b) { return sm + (size_t)b * bb + (size_t)R * kKc * 2; };
  auto load_w = [&](int c, int b) {
    uint8_t* d = wbuf(b);
    for (int i = tid; i < R * rcw; i += kThreads) {
      const int r = i / rcw, q = i % rcw;
      const bool ok = n0 + r < p.N;
      cp_async16(d + swz(r, q, rcw), ok ? p.W + (size_t)(n0 + r) * p.ldw + c * kKc + q * 8 : p.W, ok);
    }
  };
  auto load_x = [&](int c, int b) {
    uint8_t* d = xbuf(b);
    for (int i = tid; i < p.M * rcw; i += kThreads) {
      const int r = i / rcw, q = i % rcw;
      cp_async16(d + swz(r, q, rcw), p.A + (size_t)r * p.lda + c * kKc + q * 8, true);
    }
  };
  const int pre = min(NB, nch);
  for (int c = 0; c < pre; ++c) load_w(c, c);
  asm volatile("cp.async.commit_group;\n" ::);
  pdl_wait();
  if (p.late_trigger) pdl_trigger();
  for (int c = 0; c < pre; ++c) load_x(c, c);
  asm volatile("cp.async.commit_group;\n" ::);
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  __syncthreads();
  float acc[MI][NJ][4];
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int nj = 0; nj < NJ; ++nj) acc[mi][nj][0] = acc[mi][nj][1] = acc[mi][nj][2] = acc[mi][nj][3] = 0.f;
  constexpr int kSteps = kKc / 16 / kWarps;  // 16-element k-steps per warp and chunk
  for (int c = 0; c < nch; ++c) {
    const int b = c % NB;
    const uint8_t* sw = wbuf(b);
    const uint8_t* sx = xbuf(b);
#pragma unroll
    for (int s = 0; s < kSteps; ++s) {
      const int ks = warp * kSteps + s;
      uint32_t bfr[NJ][2];
#pragma unroll
      for (int nj = 0; nj < NJ; ++nj)
        ldsm_x2(bfr[nj], sx + swz((nj * 8 + (lane & 7)) % p.M, ks * 2 + ((lane >> 3) & 1), rcw));
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        uint32_t af[4];
        ldsm_x4(af, sw + swz(mi * 16 + (lane & 15), ks * 2 + (lane >> 4), rcw));
#pragma unroll
        for (int nj = 0; nj < NJ; ++nj) mma_bf16(acc[mi][nj], af, bfr[nj]);
      }
    }
    __syncthreads();  // every warp is done with buffer b
    if (c + NB < nch) {
      load_w(c + NB, b);
      load_x(c + NB, b);
    }
    asm volatile("cp.async.commit_group;\n" ::);
    if (c + 1 < nch) {
      // chunk c + 1: committed NB - 1 groups ago (or complete before the loop)
      asm volatile("cp.async.wait_group %0;\n" ::"n"(NB - 1) : "memory");
      __syncthreads();
    }
  }
  // warps' partials ([8][R][MT], in buffer 0), summed in warp order
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  __syncthreads();
  float* red = reinterpret_cast<float*>(sm);
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int nj = 0; nj < NJ; ++nj)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int row = mi * 16 + (lane >> 2) + (e >> 1) * 8, tok = nj * 8 + (lane & 3) * 2 + (e & 1);
        red[((size_t)warp * R + row) * MT + tok] = acc[mi][nj][e];
      }
  __syncthreads();
  gemv_store<R>(p, red, MT, n0, tid);
}

// rows per CTA of the streamed GEMV: the fewest (16 .. 64) that keep the grid within max_ctas
static int pick_stream_rows(int N, int max_ctas) {
  for (int rows = 16; rows <= 64; rows += 16)
    if ((int)cdiv(N, rows) <= max_ctas) return rows;
  return 0;
}

bool gemv_stream_fits(int M, int N, int K, int max_ctas, int cap) {
  if (M < 1 || M > kGemvPreMaxRows || K % kKc != 0 || N < 1) return false;
  const int rows = pick_stream_rows(N, std::min(max_ctas, cap > 0 ? cap : kStreamMaxCtas));
  if (rows == 0) return false;
  const int nb = M <= 8 ? 3 : 2;
  return (size_t)nb * stream_buf_bytes(rows, M) <= (size_t)kSmemMax;
}

template <int MI, int MT, int NB>
static void launch_stream(const GemvPreArgs& a, cudaStream_t st) {
  const size_t smem = (size_t)NB * stream_buf_bytes(MI * 16, a.M);
  static bool configured[64] = {};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  if (!configured[dev]) {
    CUDA_CHECK(cudaFuncSetAttribute(gemv_stream_kernel<MI, MT, NB>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemMax));
    configured[dev] = true;
  }
  launch_pdl(gemv_stream_kernel<MI, MT, NB>, dim3(cdiv(a.N, MI * 16)), dim3(kThreads), smem, st, a);
}

void gemv_stream(const GemvPreArgs& a, int max_ctas, cudaStream_t st, int cap) {
  CHECK(gemv_stream_fits(a.M, a.N, a.K, max_ctas, cap), "gemv_stream: shapes not supported");
  CHECK(a.lda % 8 == 0 && a.ldw % 8 == 0, "gemv_stream: 16-byte aligned rows needed");
  const int rows = pick_stream_rows(a.N, std::min(max_ctas, cap > 0 ? cap : kStreamMaxCtas));
#define GEMV_STREAM_ROWS(MI)                                      \
  if (a.M <= 8)                                                   \
    launch_stream<MI, 8, 3>(a, st);                               \
  else                                                            \
    launch_stream<MI, 16, 2>(a, st);
  switch (rows) {
    case 16: GEMV_STREAM_ROWS(1); break;
    case 32: GEMV_STREAM_ROWS(2); break;
    case 48: GEMV_STREAM_ROWS(3); break;
    default: GEMV_STREAM_ROWS(4); break;
  }
#undef GEMV_STREAM_ROWS
}

template <int MT>
static void launch_rows(const GemvPreArgs& a, int rows, cudaStream_t st, int carveout) {
  switch (rows) {
    case 16: launch<16, MT>(a, st, carveout); break;
    case 32: launch<32, MT>(a, st, carveout); break;
    case 48: launch<48, MT>(a, st, carveout); break;
    default: launch<64, MT>(a, st, carveout); break;
  }
}

void gemv_pre(const GemvPreArgs& a, int max_ctas, cudaStream_t st, int carveout) {
  CHECK(gemv_pre_fits(a.M, a.N, a.K, max_ctas, a.batch), "gemv_pre: shapes not supported");
  CHECK(a.lda % 8 == 0 && a.ldw % 8 == 0 && a.sA % 8 == 0 && a.sW % 8 == 0,
        "gemv_pre: 16-byte aligned rows needed");
  const int rows = pick_rows(a.N, max_ctas / a.batch);
  if (a.M <= 8)
    launch_rows<8>(a, rows, st, carveout);
  else
    launch_rows<16>(a, rows, st, carveout);
}


// ------------------------------------------------------------------ pipelined GEMV (<= 16 rows)
// For decode shapes whose weight slices do not fit gemv_pre (9-16 token rows, or larger matrices),
// where cuBLASLt's small-M kernels take split-K reductions: CTA b owns R = 16 MI
// consecutive weight rows (one wave), and each of its 8 warps a contiguous eighth of K, streamed
// through the warp's own ring of S k-chunks of 64 columns (the R weight-row segments and the
// activations' 16) with 16-byte cp.async; a warp waits only for its own copies (no block barrier
// before the epilogue). The weight segments of the first S - 1 chunks are requested before the PDL
// wait (weights never depend on earlier kernels), the activations after it. The warps' partial
// sums are added in warp order (deterministic; not bitwise gemv_pre / cuBLASLt).
constexpr int kPKc = 64;  // k-chunk columns (one 128-byte row segment)

// (XR: activation rows per stage -- 16, or 8 for at most 8 token rows (gemv_pipe8): the first MMA n-tile only, same
//  products and sums for those tokens, and the stage bytes it saves buy ring stages)
// (EP: the pushed form's epoch loaded before the PDL wait -- final at this grid's start, as gemv_pre_lean_kernel's EP)
template <int MI, int S, int XR = 16, bool EP = false>
__global__ void __launch_bounds__(kThreads, 1) gemv_pipe_kernel(const GemvPreArgs p) {
  if (!p.late_trigger) pdl_trigger();
  extern __shared__ __align__(128) uint8_t sm[];
  constexpr int R = MI * 16, NJ = XR / 8;
  constexpr int SB = (R + XR) * kPKc * 2;  // stage bytes: R weight rows + XR activation rows
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int n0 = blockIdx.x * R;
  const int kw = p.K / kWarps;  // this warp's k-slice
  const int k0 = warp * kw;
  const int nch = kw / kPKc;
  uint8_t* ring = sm + (size_t)warp * S * SB;
  auto load_w = [&](int c, int st) {
    uint8_t* d = ring + (size_t)st * SB;
#pragma unroll
    for (int i = lane; i < R * 8; i += 32) {
      const int r = i >> 3, q = i & 7;
      const bool ok = n0 + r < p.N;
      cp_async16(d + swz(r, q, 8), ok ? p.W + (size_t)(n0 + r) * p.ldw + k0 + c * kPKc + q * 8 : p.W,
                 ok);
    }
  };
  auto load_x = [&](int c, int st) {
    uint8_t* d = ring + (size_t)st * SB + R * kPKc * 2;
#pragma unroll
    for (int i = lane; i < XR * 8; i += 32) {
      const int r = i >> 3, q = i & 7;
      const bool ok = r < p.M;  // (rows past M: zero-filled, no read)
      cp_async16(d + swz(r, q, 8), ok ? p.A + (size_t)r * p.lda + k0 + c * kPKc + q * 8 : p.A, ok);
    }
  };
  const int pre = min(S - 1, nch);
  for (int c = 0; c < pre; ++c) load_w(c, c);
  // (EP: the push epoch requested here, into a register -- nothing waits for it before the epilogue, which publishes it)
  __shared__ uint32_t s_ep;
  uint32_t ep_r = 0;
  if constexpr (EP) {
    if (tid == 0 && p.push.state) ep_r = __ldcg(p.push.state);
  }
  pdl_wait();
  if (p.late_trigger) pdl_trigger();
  // groups: g_c holds chunk c's activations (g_0 also the weights of chunks 0 .. pre - 1; S - 1
  // groups even for fewer chunks); the loop's group of iteration c holds chunk c + S - 1, so chunk c
  // is complete with groups 0 .. c
  for (int c = 0; c < S - 1; ++c) {
    if (c < nch) load_x(c, c);
    asm volatile("cp.async.commit_group;\n" ::);
  }
  float acc[MI][NJ][4];
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int nj = 0; nj < NJ; ++nj) acc[mi][nj][0] = acc[mi][nj][1] = acc[mi][nj][2] = acc[mi][nj][3] = 0.f;
  for (int c = 0; c < nch; ++c) {
    const int st = c % S;
    if (c + S - 1 < nch) {
      load_w(c + S - 1, (c + S - 1) % S);
      load_x(c + S - 1, (c + S - 1) % S);
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group %0;\n" ::"n"(S - 1) : "memory");
    __syncwarp();
    const uint8_t* w = ring + (size_t)st * SB;
    const uint8_t* x = w + R * kPKc * 2;
#pragma unroll
    for (int ks = 0; ks < kPKc / 16; ++ks) {
      uint32_t bfr[NJ][2];
#pragma unroll
      for (int nj = 0; nj < NJ; ++nj)
        ldsm_x2(bfr[nj], x + swz(nj * 8 + (lane & 7), ks * 2 + ((lane >> 3) & 1), 8));
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        uint32_t af[4];
        ldsm_x4(af, w + swz(mi * 16 + (lane & 15), ks * 2 + (lane >> 4), 8));
        mma_bf16(acc[mi][0], af, bfr[0]);
        if constexpr (NJ > 1) mma_bf16(acc[mi][1], af, bfr[1]);
      }
    }
    __syncwarp();  // (the warp is done with this stage before it is refilled)
  }
  // warps' partials ([8][R][16] floats over the rings), summed in warp order
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  __syncthreads();
  float* red = reinterpret_cast<float*>(sm);
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int nj = 0; nj < NJ; ++nj)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int row = mi * 16 + (lane >> 2) + (e >> 1) * 8, tok = nj * 8 + (lane & 3) * 2 + (e & 1);
        red[((size_t)warp * R + row) * 16 + tok] = acc[mi][nj][e];
      }
  if constexpr (EP) {
    if (tid == 0) s_ep = ep_r;
  }
  __syncthreads();
  if constexpr (EP)
    gemv_store_e<R>(p, red, 16, n0, tid, s_ep);
  else
    gemv_store<R>(p, red, 16, n0, tid);
}

namespace {
// ring stages per R rows: 24 KB per warp (192 KB) at most
constexpr int pipe_stages(int mi) { return mi == 1 ? 6 : (mi == 2 ? 4 : (mi == 3 ? 3 : 2)); }
size_t pipe_smem(int mi) {
  const size_t ring = (size_t)kWarps * pipe_stages(mi) * (mi * 16 + 16) * kPKc * 2;
  const size_t red = (size_t)kWarps * mi * 16 * 16 * sizeof(float);
  return ring > red ? ring : red;
}
}  // namespace

bool gemv_pipe_fits(int M, int N, int K, int max_ctas) {
  if (M < 1 || M > kGemvPreMaxRows || N < 1 || K % (kWarps * kPKc) != 0) return false;
  return pick_rows(N, max_ctas) > 0;
}

template <int MI>
static void launch_pipe(const GemvPreArgs& a, cudaStream_t st) {
  constexpr int S = pipe_stages(MI);
  const size_t smem = pipe_smem(MI);
  static bool configured[64] = {};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  if (!configured[dev]) {
    CUDA_CHECK(cudaFuncSetAttribute(gemv_pipe_kernel<MI, S>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    configured[dev] = true;
  }
  launch_pdl(gemv_pipe_kernel<MI, S>, dim3(cdiv(a.N, MI * 16)), dim3(kThreads), smem, st, a);
}

void gemv_pipe(const GemvPreArgs& a, int max_ctas, cudaStream_t st) {
  CHECK(gemv_pipe_fits(a.M, a.N, a.K, max_ctas), "gemv_pipe: shapes not supported");
  CHECK(a.lda % 8 == 0 && a.ldw % 8 == 0, "gemv_pipe: 16-byte aligned rows needed");
  switch (pick_rows(a.N, max_ctas)) {
    case 16: launch_pipe<1>(a, st); break;
    case 32: launch_pipe<2>(a, st); break;
    case 48: launch_pipe<3>(a, st); break;
    default: launch_pipe<4>(a, st); break;
  }
}

// gemv_pipe's pushed form with its push epoch loaded before the PDL wait (the KDA o_proj at 5-8 rows): bitwise gemv_pipe
template <int MI>
static void launch_pipe_ep(const GemvPreArgs& a, cudaStream_t st) {
  constexpr int S = pipe_stages(MI);
  const size_t smem = pipe_smem(MI);
  static bool configured[64] = {};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  if (!configured[dev]) {
    CUDA_CHECK(cudaFuncSetAttribute(gemv_pipe_kernel<MI, S, 16, true>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    configured[dev] = true;
  }
  launch_pdl(gemv_pipe_kernel<MI, S, 16, true>, dim3(cdiv(a.N, MI * 16)), dim3(kThreads), smem, st, a);
}

void gemv_pipe_ep(const GemvPreArgs& a, int max_ctas, cudaStream_t st) {
  CHECK(gemv_pipe_fits(a.M, a.N, a.K, max_ctas) && a.push.state && a.late_trigger, "gemv_pipe_ep: shapes not supported");
  CHECK(a.lda % 8 == 0 && a.ldw % 8 == 0, "gemv_pipe_ep: 16-byte aligned rows needed");
  switch (pick_rows(a.N, max_ctas)) {
    case 16: launch_pipe_ep<1>(a, st); break;
    case 32: launch_pipe_ep<2>(a, st); break;
    case 48: launch_pipe_ep<3>(a, st); break;
    default: launch_pipe_ep<4>(a, st); break;
  }
}

// gemv_pipe for at most 8 token rows with 8-row activation stages and as many ring stages as fit in 227 KB (48 rows:
// 4 stages instead of 3, three k-chunks in flight per warp and before the PDL wait instead of two): bitwise gemv_pipe
// (the same k-slices, MMAs and warp-order sums; the second n-tile only ever held zero rows)
namespace {
constexpr size_t kPipe8Smem = 227 * 1024;
constexpr int pipe8_stages(int mi) { return (int)(kPipe8Smem / ((size_t)kWarps * (mi * 16 + 8) * kPKc * 2)); }
template <int MI>
void launch_pipe8(const GemvPreArgs& a, cudaStream_t st) {
  constexpr int S = pipe8_stages(MI) < 8 ? pipe8_stages(MI) : 8;
  static_assert(S >= 2, "gemv_pipe8: ring stages");
  const size_t ring = (size_t)kWarps * S * (MI * 16 + 8) * kPKc * 2, red = (size_t)kWarps * MI * 16 * 16 * sizeof(float);
  const size_t smem = ring > red ? ring : red;
  static bool configured[64] = {};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  if (!configured[dev]) {
    CUDA_CHECK(cudaFuncSetAttribute(gemv_pipe_kernel<MI, S, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    configured[dev] = true;
  }
  launch_pdl(gemv_pipe_kernel<MI, S, 8>, dim3(cdiv(a.N, MI * 16)), dim3(kThreads), smem, st, a);
}
}  // namespace

bool gemv_pipe8_fits(int M, int N, int K, int max_ctas) { return M <= 8 && gemv_pipe_fits(M, N, K, max_ctas); }

void gemv_pipe8(const GemvPreArgs& a, int max_ctas, cudaStream_t st) {
  CHECK(gemv_pipe8_fits(a.M, a.N, a.K, max_ctas), "gemv_pipe8: shapes not supported");
  CHECK(a.lda % 8 == 0 && a.ldw % 8 == 0, "gemv_pipe8: 16-byte aligned rows needed");
  switch (pick_rows(a.N, max_ctas)) {
    case 16: launch_pipe8<1>(a, st); break;
    case 32: launch_pipe8<2>(a, st); break;
    case 48: launch_pipe8<3>(a, st); break;
    default: launch_pipe8<4>(a, st); break;
  }
}

}  // namespace k
}  // namespace glm
