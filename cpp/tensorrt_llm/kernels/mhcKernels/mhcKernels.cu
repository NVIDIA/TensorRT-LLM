/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mhcKernels.h"

#include "tensorrt_llm/common/assert.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::mhc
{

// ===================================================================
// Kernel 1: big_fuse — one CTA per token
//
//  Template parameters:
//    NUM_SPLITS:  split-K reduction (1 = direct, >1 = sum across splits)
//    BLOCK_SIZE:  threads per CTA (multiple of 32, >= 64)
//
//  Phase 1a (warp 0, lanes 0-3): RMS norm + sigmoid → s_pre_mix, post_mix
//  ── __syncthreads ──
//  Phase 1b (warp 0, lanes 0-3) ‖ Phase 2 (remaining warps) — overlapped
//    1b: parallel Sinkhorn (4 lanes, __shfl_xor col normalize) → comb_mix
//     2: stream residual × pre_mix → layer_input
// ===================================================================

template <int NUM_SPLITS, int BLOCK_SIZE, bool kFuseNorm = false>
__launch_bounds__(BLOCK_SIZE) __global__ void mhcBigFuseKernel(float const* __restrict__ y_acc,
    float const* __restrict__ r_acc, __nv_bfloat16 const* __restrict__ residual, float const* __restrict__ hc_scale,
    float const* __restrict__ hc_base, float* __restrict__ post_mix, float* __restrict__ comb_mix,
    __nv_bfloat16* __restrict__ layer_input, int M, int K, int hidden_size, float rms_eps, float hc_pre_eps,
    float hc_sinkhorn_eps, float hc_post_mult_value, int sinkhorn_repeat,
    // kFuseNorm: when true, applies next-layer RMSNorm to layer_input output
    // inline: layer_input[t,h] = bf16(li * rsqrt(mean(li²)+norm_eps) * norm_weight[h]).
    // norm_weight is bf16 [hidden_size]; ignored when kFuseNorm=false.
    __nv_bfloat16 const* __restrict__ norm_weight = nullptr, float norm_eps = 0.f)
{
    constexpr int HC_MULT = 4;
    constexpr int HC_MULT2 = HC_MULT * HC_MULT;       // 16 comb_mix entries
    constexpr int HC_MULT3 = HC_MULT * (2 + HC_MULT); // 24 = pre(4) + post(4) + comb(16)
    constexpr int WARP_SIZE = 32;
    constexpr int BF16_VEC = 8;                       // bf16 elements per uint4 load

    int const token = blockIdx.x;
    int const tid = threadIdx.x;
    int const warp_id = tid / WARP_SIZE;
    int const lane = tid % WARP_SIZE;

    __shared__ float s_pre_mix[HC_MULT];

    float cm[HC_MULT];

    // ---- Phase 1a (warp 0, lanes 0..3): split-K reduce → RMS norm → sigmoid ----
    // Each lane handles one of the 4 hc_mult slots.
    // Produces: s_pre_mix[4] (shared), post_mix[4] (global), cm[4×4] (registers).
    if (warp_id == 0 && lane < HC_MULT)
    {
        float r_val;
        float y_local[HC_MULT3];

        if constexpr (NUM_SPLITS == 1)
        {
            r_val = r_acc[token];
            float const* y_row = y_acc + token * HC_MULT3;
#pragma unroll
            for (int c = 0; c < HC_MULT3; c++)
                y_local[c] = y_row[c];
        }
        else
        {
            // Reduce across split-K partials
            r_val = 0.0f;
#pragma unroll
            for (int c = 0; c < HC_MULT3; c++)
                y_local[c] = 0.0f;
            for (int s = 0; s < NUM_SPLITS; s++)
            {
                r_val += r_acc[s * M + token];
                float const* y_row = y_acc + (static_cast<long long>(s) * M + token) * HC_MULT3;
#pragma unroll
                for (int c = 0; c < HC_MULT3; c++)
                    y_local[c] += y_row[c];
            }
        }

        // RMS norm: rstd = 1/sqrt(mean(x²) + eps)
        float const rstd = rsqrtf(r_val / static_cast<float>(K) + rms_eps);
        float const s0 = hc_scale[0], s1 = hc_scale[1], s2 = hc_scale[2];

        // y_local layout: [pre_mix(4) | post_mix(4) | comb_mix(4×4)]
        // pre_mix: sigmoid(norm * scale0 + base) + eps → shared for Phase 2
        float v = y_local[lane] * rstd * s0 + hc_base[lane];
        s_pre_mix[lane] = 1.0f / (1.0f + expf(-v)) + hc_pre_eps;

        // post_mix: sigmoid(norm * scale1 + base) * mult → global
        v = y_local[HC_MULT + lane] * rstd * s1 + hc_base[HC_MULT + lane];
        post_mix[token * HC_MULT + lane] = 1.0f / (1.0f + expf(-v)) * hc_post_mult_value;

        // comb_mix init: norm * scale2 + base → cm[4] per lane (one row of 4×4 matrix)
#pragma unroll
        for (int k = 0; k < HC_MULT; k++)
            cm[k] = y_local[2 * HC_MULT + lane * HC_MULT + k] * rstd * s2 + hc_base[2 * HC_MULT + lane * HC_MULT + k];
    }

    __syncthreads();

    // ---- Phase 1b (warp 0, lanes 0..3): Sinkhorn normalization ----
    // Each lane holds one row of the 4×4 comb matrix.
    // Row normalize via local sum, column normalize via __shfl_xor across 4 lanes.
    if (warp_id == 0 && lane < HC_MULT)
    {
        constexpr unsigned LANE_MASK = (1u << HC_MULT) - 1; // 0xf for HC_MULT=4

        // Softmax rows: subtract the row max to avoid inf / inf when comb logits
        // are large.
        float const rowMax = fmaxf(fmaxf(cm[0], cm[1]), fmaxf(cm[2], cm[3]));
#pragma unroll
        for (int k = 0; k < HC_MULT; k++)
            cm[k] = expf(cm[k] - rowMax);
        // Replace per-element fdiv with one reciprocal + 4 fmul. fp32 fdiv on
        // B200 is multi-cycle while fmul retires at peak rate; sinkhorn's
        // O(HC_MULT * sinkhorn_repeat) divisions per token (160 at sinkhorn=20)
        // dominate the bigfuse epilogue cost on this 4-lane warp. Math is
        // identical modulo last-bit round-off, which sinkhorn iteration
        // absorbs.
        float inv_rs = 1.0f / (cm[0] + cm[1] + cm[2] + cm[3]);
#pragma unroll
        for (int k = 0; k < HC_MULT; k++)
            cm[k] = cm[k] * inv_rs + hc_sinkhorn_eps;

            // Column normalize: sum across lanes (rows) via butterfly shuffle
#pragma unroll
        for (int k = 0; k < HC_MULT; k++)
        {
            float cs = cm[k];
            cs += __shfl_xor_sync(LANE_MASK, cs, 1);
            cs += __shfl_xor_sync(LANE_MASK, cs, 2);
            cm[k] *= 1.0f / (cs + hc_sinkhorn_eps);
        }

        // Remaining Sinkhorn iterations: alternate row / column normalize
        for (int it = 1; it < sinkhorn_repeat; it++)
        {
            inv_rs = 1.0f / (cm[0] + cm[1] + cm[2] + cm[3] + hc_sinkhorn_eps);
#pragma unroll
            for (int k = 0; k < HC_MULT; k++)
                cm[k] *= inv_rs;

#pragma unroll
            for (int k = 0; k < HC_MULT; k++)
            {
                float cs = cm[k];
                cs += __shfl_xor_sync(LANE_MASK, cs, 1);
                cs += __shfl_xor_sync(LANE_MASK, cs, 2);
                cm[k] *= 1.0f / (cs + hc_sinkhorn_eps);
            }
        }

        // Write 4×4 comb_mix to global (lane = row index, k = col index)
        float* cm_out = comb_mix + token * HC_MULT2;
#pragma unroll
        for (int k = 0; k < HC_MULT; k++)
            cm_out[lane * HC_MULT + k] = cm[k];
    }

    // s_sumsq is shared between Phase 2 (writers: bigfuse warps) and the
    // rsqrt reduction (reader: warp 0). Single allocation across both blocks.
    constexpr int kBigFuseWarps = BLOCK_SIZE / WARP_SIZE - 1;
    __shared__ float s_sumsq[kBigFuseWarps];
    __shared__ float s_rsqrt;

    // ---- Phase 2 (warps 1..N, overlapped with Phase 1b): weighted residual sum ----
    // layer_input[h] = sum_j pre_mix[j] * residual[j][h]
    // Vectorized: 8 bf16 per thread per iteration via uint4 (LDG.128).
    //
    // When kFuseNorm is true, we additionally accumulate sum_sq across pass 1
    // and apply next-layer RMSNorm in a pass 2 that re-LDGs layer_input from
    // L2 (hot from pass 1's just-issued STGs), multiplies by rsqrt*norm_weight
    // and re-STGs the normalized bf16. Saves a separate flashinfer.rmsnorm
    // kernel launch + its HBM round-trip on layer_input.
    if (warp_id > 0)
    {
        float pm[HC_MULT];
#pragma unroll
        for (int j = 0; j < HC_MULT; j++)
            pm[j] = s_pre_mix[j];

        __nv_bfloat16 const* rbase = residual + static_cast<long long>(token) * HC_MULT * hidden_size;
        __nv_bfloat16* obase = layer_input + static_cast<long long>(token) * hidden_size;

        int const p2_tid = tid - WARP_SIZE;
        constexpr int p2_threads = BLOCK_SIZE - WARP_SIZE;

        float sum_sq_local = 0.f;
        for (int h = p2_tid * BF16_VEC; h < hidden_size; h += p2_threads * BF16_VEC)
        {
            float acc[BF16_VEC] = {};

#pragma unroll
            for (int j = 0; j < HC_MULT; j++)
            {
                uint4 raw = *reinterpret_cast<uint4 const*>(&rbase[j * hidden_size + h]);
                __nv_bfloat162 const* pairs = reinterpret_cast<__nv_bfloat162 const*>(&raw);
#pragma unroll
                for (int v = 0; v < BF16_VEC / 2; v++)
                {
                    float2 f = __bfloat1622float2(pairs[v]);
                    acc[2 * v + 0] += pm[j] * f.x;
                    acc[2 * v + 1] += pm[j] * f.y;
                }
            }

            uint4 out_raw;
            __nv_bfloat162* opairs = reinterpret_cast<__nv_bfloat162*>(&out_raw);
#pragma unroll
            for (int v = 0; v < BF16_VEC / 2; v++)
                opairs[v] = __float22bfloat162_rn(make_float2(acc[2 * v], acc[2 * v + 1]));
            *reinterpret_cast<uint4*>(&obase[h]) = out_raw;

            if constexpr (kFuseNorm)
            {
                // Square the bf16-rounded values so sum_sq matches a separate
                // RMSNorm kernel's behavior (which reads the bf16 layer_input).
#pragma unroll
                for (int v = 0; v < BF16_VEC / 2; v++)
                {
                    float2 b = __bfloat1622float2(opairs[v]);
                    sum_sq_local += b.x * b.x + b.y * b.y;
                }
            }
        }

        if constexpr (kFuseNorm)
        {
            // Reduce sum_sq across all (BLOCK_SIZE - WARP_SIZE) threads on the
            // bigfuse warps.  Warp-internal __shfl_xor first, then SMEM ladder
            // across warps 1..N.  Warp 0 was idle since Phase 1b — we can also
            // use it as a clean reducer to avoid an extra sync round-trip.
            sum_sq_local += __shfl_xor_sync(0xffffffff, sum_sq_local, 16);
            sum_sq_local += __shfl_xor_sync(0xffffffff, sum_sq_local, 8);
            sum_sq_local += __shfl_xor_sync(0xffffffff, sum_sq_local, 4);
            sum_sq_local += __shfl_xor_sync(0xffffffff, sum_sq_local, 2);
            sum_sq_local += __shfl_xor_sync(0xffffffff, sum_sq_local, 1);

            int const warp_in_bf = warp_id - 1; // bigfuse warps are 1..N
            if ((tid & 31) == 0)
                s_sumsq[warp_in_bf] = sum_sq_local;
        }
    }
    else if constexpr (kFuseNorm)
    {
        // Warp 0 idle threads: still need to participate in __syncthreads
        // alongside warps 1..N when fusing norm (the syncthreads below).
    }

    if constexpr (kFuseNorm)
    {
        __syncthreads();

        // All threads now collaborate on pass 2: re-LDG layer_input bf16 from
        // L2, multiply by rsqrt * norm_weight, STG normalized bf16.
        // Lane 0 of warp 0 reduces the partial sums; broadcasts via SMEM cell.
        if (warp_id == 0 && (tid & 31) == 0)
        {
            float total = 0.f;
#pragma unroll
            for (int w = 0; w < kBigFuseWarps; w++)
                total += s_sumsq[w];
            s_rsqrt = rsqrtf(total / static_cast<float>(hidden_size) + norm_eps);
        }
        __syncthreads();

        if (warp_id > 0)
        {
            __nv_bfloat16* obase = layer_input + static_cast<long long>(token) * hidden_size;
            int const p2_tid = tid - WARP_SIZE;
            constexpr int p2_threads = BLOCK_SIZE - WARP_SIZE;
            float const rsqrt_val = s_rsqrt;
            for (int h = p2_tid * BF16_VEC; h < hidden_size; h += p2_threads * BF16_VEC)
            {
                uint4 li_raw = *reinterpret_cast<uint4 const*>(&obase[h]);
                uint4 nw_raw = *reinterpret_cast<uint4 const*>(&norm_weight[h]);
                __nv_bfloat162 const* li_pairs = reinterpret_cast<__nv_bfloat162 const*>(&li_raw);
                __nv_bfloat162 const* nw_pairs = reinterpret_cast<__nv_bfloat162 const*>(&nw_raw);
                uint4 out_raw;
                __nv_bfloat162* opairs = reinterpret_cast<__nv_bfloat162*>(&out_raw);
#pragma unroll
                for (int v = 0; v < BF16_VEC / 2; v++)
                {
                    float2 lif = __bfloat1622float2(li_pairs[v]);
                    float2 nwf = __bfloat1622float2(nw_pairs[v]);
                    opairs[v]
                        = __float22bfloat162_rn(make_float2(lif.x * rsqrt_val * nwf.x, lif.y * rsqrt_val * nwf.y));
                }
                *reinterpret_cast<uint4*>(&obase[h]) = out_raw;
            }
        }
    }
}

#define INST_BIGFUSE(NS, BS)                                                                                           \
    template __global__ void mhcBigFuseKernel<NS, BS, /*kFuseNorm=*/false>(float const*, float const*,                 \
        __nv_bfloat16 const*, float const*, float const*, float*, float*, __nv_bfloat16*, int, int, int, float, float, \
        float, float, int, __nv_bfloat16 const*, float);                                                               \
    template __global__ void mhcBigFuseKernel<NS, BS, /*kFuseNorm=*/true>(float const*, float const*,                  \
        __nv_bfloat16 const*, float const*, float const*, float*, float*, __nv_bfloat16*, int, int, int, float, float, \
        float, float, int, __nv_bfloat16 const*, float);

INST_BIGFUSE(1, 128)
INST_BIGFUSE(1, 256)
INST_BIGFUSE(1, 512)
INST_BIGFUSE(2, 128)
INST_BIGFUSE(2, 256)
INST_BIGFUSE(2, 512)
INST_BIGFUSE(4, 128)
INST_BIGFUSE(4, 256)
INST_BIGFUSE(4, 512)
INST_BIGFUSE(8, 128)
INST_BIGFUSE(8, 256)
INST_BIGFUSE(8, 512)
INST_BIGFUSE(16, 128)
INST_BIGFUSE(16, 256)
INST_BIGFUSE(16, 512)
#undef INST_BIGFUSE

// ===================================================================
// Kernel 3: gemm_sqrsum_fma — split-N FP32 FMA GEMM with fused sqrsum
//
//  Y[m, n] = dot(X[m, :], W_T[n, :])     for each (m, n)
//  R[m]    = sum_k X[m, k]^2              (only computed once per row)
//
//  Grid:   (M, ceil(N / N_PER_BLOCK))
//  Block:  256 threads = 8 warps × 32 lanes
//
//  Each block handles 1 row × N_PER_BLOCK output columns × full K.
//  The sqrsum R[row] is fused into the first N-tile (n_start==0) to
//  avoid a separate reduction pass.
//
//  Data layout: X is row-major bf16, W_T is row-major fp32 (transposed).
//  Uses PTX prmt.b32 for bf16→fp32 zero-extension instead of cvt,
//  and explicit cache hints (ld.global.cs for X, L1::evict_last for W_T)
//  since X is reused across N-tiles but W_T is streamed once.
//
//  Warp-level reduction via __shfl_xor → shared memory → cross-warp sum.
//  N_PER_BLOCK columns are processed in groups of NUM_WARPS (8), mapping
//  each warp to one output column for the final shared-memory reduce.
// ===================================================================

template <int N_PER_BLOCK>
__launch_bounds__(256) __global__ void mhcGemmSqrsumFmaKernel(__nv_bfloat16 const* __restrict__ X,
    float const* __restrict__ W_T, float* __restrict__ Y, float* __restrict__ R, int M, int N, int K)
{
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    constexpr int K_VEC = 4;                   // bf16 elements loaded per thread per iteration
    constexpr int K_STEP = BLOCK_SIZE * K_VEC; // K elements consumed per block per iteration

    int const tid = threadIdx.x;
    int const warp_id = tid / WARP_SIZE;
    int const lane = tid % WARP_SIZE;
    int const row = blockIdx.x;
    int const n_start = blockIdx.y * N_PER_BLOCK;

    if (row >= M || n_start >= N)
        return;
    __nv_bfloat16 const* x_row = X + static_cast<long long>(row) * K;

    float acc[N_PER_BLOCK];
#pragma unroll
    for (int n = 0; n < N_PER_BLOCK; n++)
        acc[n] = 0.0f;
    float sqr = 0.0f;
    bool const do_sqr = (n_start == 0);

    // Main loop: process K in chunks of K_STEP (256 threads × 4 = 1024 elements)
    for (int k_base = 0; k_base + K_STEP <= K; k_base += K_STEP)
    {
        int const my_k = k_base + tid * K_VEC;

        // Load 4 bf16 from X as packed u32 pair (cache-streaming hint: X reused across N-tiles)
        unsigned xp0, xp1;
        asm volatile("ld.global.cs.v2.b32 {%0, %1}, [%2];" : "=r"(xp0), "=r"(xp1) : "l"(x_row + my_k));

        // bf16→fp32 via prmt.b32: zero-extend each 16-bit half into a 32-bit float
        // prmt selectors: 0x5410 extracts low halfword, 0x7610 extracts high halfword
        float xv0, xv1, xv2, xv3;
        asm volatile(
            "{ .reg .b32 t0, t1, t2, t3;\n\t"
            "  prmt.b32 t0, %4, %5, %6;\n\t"
            "  prmt.b32 t1, %4, %5, %7;\n\t"
            "  prmt.b32 t2, %4, %8, %6;\n\t"
            "  prmt.b32 t3, %4, %8, %7;\n\t"
            "  mov.b32 %0, t0;\n\t  mov.b32 %1, t1;\n\t"
            "  mov.b32 %2, t2;\n\t  mov.b32 %3, t3; }"
            : "=f"(xv0), "=f"(xv1), "=f"(xv2), "=f"(xv3)
            : "r"(0), "r"(xp0), "r"(0x5410), "r"(0x7610), "r"(xp1));

        if (do_sqr)
        {
            sqr = fmaf(xv0, xv0, sqr);
            sqr = fmaf(xv1, xv1, sqr);
            sqr = fmaf(xv2, xv2, sqr);
            sqr = fmaf(xv3, xv3, sqr);
        }

        // Dot product: acc[n] += X[k:k+4] · W_T[n][k:k+4]
        // W_T uses evict_last hint since each weight row is accessed only once
#pragma unroll
        for (int n = 0; n < N_PER_BLOCK; n++)
        {
            float w0, w1, w2, w3;
            asm volatile("ld.global.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"
                         : "=f"(w0), "=f"(w1), "=f"(w2), "=f"(w3)
                         : "l"(W_T + static_cast<long long>(n_start + n) * K + my_k));
            acc[n] = fmaf(xv0, w0, acc[n]);
            acc[n] = fmaf(xv1, w1, acc[n]);
            acc[n] = fmaf(xv2, w2, acc[n]);
            acc[n] = fmaf(xv3, w3, acc[n]);
        }
    }

    // Tail: handle remaining K elements (K % K_STEP), scalar loads
    {
        int const tail_start = K - (K % K_STEP);
        for (int kk = tail_start + tid; kk < K; kk += BLOCK_SIZE)
        {
            float xv;
            asm volatile(
                "{ .reg .b16 tmp;\n\t"
                "  ld.global.cs.b16 tmp, [%1];\n\t"
                "  cvt.f32.bf16 %0, tmp; }"
                : "=f"(xv)
                : "l"(x_row + kk));
            if (do_sqr)
                sqr = fmaf(xv, xv, sqr);
#pragma unroll
            for (int n = 0; n < N_PER_BLOCK; n++)
            {
                float wv = W_T[static_cast<long long>(n_start + n) * K + kk];
                acc[n] = fmaf(xv, wv, acc[n]);
            }
        }
    }

    // ---- Cross-warp reduction ----
    // N_PER_BLOCK columns are grouped into chunks of NUM_WARPS (8).
    // Within each group: intra-warp butterfly → smem → one warp reads the column sums.
    // s_warp[warp_id][col] holds per-warp partial; extra column SQRSUM_SLOT for sqrsum.
    constexpr int COLS_PER_GROUP = NUM_WARPS;
    constexpr int SQRSUM_SLOT = COLS_PER_GROUP;
    constexpr int N_GROUPS = (N_PER_BLOCK + COLS_PER_GROUP - 1) / COLS_PER_GROUP;
    constexpr unsigned FULL_WARP_MASK = 0xffffffff;

    __shared__ float s_warp[NUM_WARPS][COLS_PER_GROUP + 1];

#pragma unroll
    for (int g = 0; g < N_GROUPS; g++)
    {
        constexpr int LAST_COLS = N_PER_BLOCK - (N_GROUPS - 1) * COLS_PER_GROUP;
        int const n_cols = (g == N_GROUPS - 1) ? LAST_COLS : COLS_PER_GROUP;

        // Intra-warp butterfly reduction for each column in this group
#pragma unroll
        for (int n = 0; n < COLS_PER_GROUP; n++)
        {
            if (n < n_cols)
            {
#pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    acc[g * COLS_PER_GROUP + n] += __shfl_xor_sync(FULL_WARP_MASK, acc[g * COLS_PER_GROUP + n], off);
            }
        }
        if (g == 0 && do_sqr)
        {
#pragma unroll
            for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                sqr += __shfl_xor_sync(FULL_WARP_MASK, sqr, off);
        }

        // Lane 0 of each warp writes its partial sum to shared memory
        if (lane == 0)
        {
#pragma unroll
            for (int n = 0; n < COLS_PER_GROUP; n++)
                if (n < n_cols)
                    s_warp[warp_id][n] = acc[g * COLS_PER_GROUP + n];
            if (g == 0 && do_sqr)
                s_warp[warp_id][SQRSUM_SLOT] = sqr;
        }
        __syncthreads();

        // Final cross-warp sum: warp_id maps to output column
        if (lane == 0 && warp_id < n_cols)
        {
            float val = 0.0f;
#pragma unroll
            for (int w = 0; w < NUM_WARPS; w++)
                val += s_warp[w][warp_id];
            Y[static_cast<long long>(row) * N + n_start + g * COLS_PER_GROUP + warp_id] = val;
        }
        if (g == 0 && do_sqr && lane == 0 && warp_id == 0)
        {
            float sq = 0.0f;
#pragma unroll
            for (int w = 0; w < NUM_WARPS; w++)
                sq += s_warp[w][SQRSUM_SLOT];
            R[row] = sq;
        }
        __syncthreads();
    }
}

#define INST_FMA(NPB)                                                                                                  \
    template __global__ void mhcGemmSqrsumFmaKernel<NPB>(                                                              \
        __nv_bfloat16 const*, float const*, float*, float*, int, int, int);

INST_FMA(1)
INST_FMA(2)
INST_FMA(3)
INST_FMA(4)
INST_FMA(6)
INST_FMA(8)
INST_FMA(12)
INST_FMA(24)
#undef INST_FMA

// ===================================================================
// Kernel 4: post_mapping — one CTA per token, 256 threads
//
//  Computes:  out[token][j][h] = post[j] * x[h] + sum_k comb[k][j] * residual[k][h]
//
//  post_mix:  [B, HC_MULT]           — per-token gating weights
//  comb_mix:  [B, HC_MULT, HC_MULT]  — per-token combination matrix (from Sinkhorn)
//  residual:  [B, HC_MULT, hidden]   — bf16 multi-head residual
//  x:         [B, hidden]            — bf16 layer output
//  out:       [B, HC_MULT, hidden]   — bf16 result
//
//  Vectorized: 8 bf16 per thread per iteration via uint4 (LDG.128).
// ===================================================================

__launch_bounds__(256) __global__ void mhcPostMappingKernel(__nv_bfloat16 const* __restrict__ residual,
    __nv_bfloat16 const* __restrict__ x, float const* __restrict__ post_mix, float const* __restrict__ comb_mix,
    __nv_bfloat16* __restrict__ out, int hidden_size)
{
    constexpr int HC_MULT = 4;
    constexpr int BLOCK_SIZE = 256;
    constexpr int BF16_VEC = 8;

    int const token = blockIdx.x;
    int const tid = threadIdx.x;

    // Load post_mix and comb_mix into shared memory (only first 20 threads active)
    __shared__ float s_post[HC_MULT];
    __shared__ float s_comb[HC_MULT][HC_MULT];

    if (tid < HC_MULT)
        s_post[tid] = post_mix[token * HC_MULT + tid];
    if (tid < HC_MULT * HC_MULT)
    {
        int const r = tid / HC_MULT;
        int const c = tid % HC_MULT;
        s_comb[r][c] = comb_mix[token * HC_MULT * HC_MULT + r * HC_MULT + c];
    }
    __syncthreads();

    // Cache post_mix and comb_mix in registers
    float pm[HC_MULT];
    float comb[HC_MULT][HC_MULT];
#pragma unroll
    for (int j = 0; j < HC_MULT; j++)
        pm[j] = s_post[j];
#pragma unroll
    for (int k = 0; k < HC_MULT; k++)
#pragma unroll
        for (int j = 0; j < HC_MULT; j++)
            comb[k][j] = s_comb[k][j];

    long long const tok_res = static_cast<long long>(token) * HC_MULT * hidden_size;
    long long const tok_x = static_cast<long long>(token) * hidden_size;

    for (int h = tid * BF16_VEC; h < hidden_size; h += BLOCK_SIZE * BF16_VEC)
    {
        // Load x[h:h+8] as bf16 → float
        uint4 x_raw = *reinterpret_cast<uint4 const*>(&x[tok_x + h]);
        __nv_bfloat162 const* xp = reinterpret_cast<__nv_bfloat162 const*>(&x_raw);
        float xf[BF16_VEC];
#pragma unroll
        for (int v = 0; v < BF16_VEC / 2; v++)
        {
            float2 f = __bfloat1622float2(xp[v]);
            xf[2 * v + 0] = f.x;
            xf[2 * v + 1] = f.y;
        }

        // acc[j][v] = post[j] * x[v]   (initialize with the x contribution)
        float acc[HC_MULT][BF16_VEC];
#pragma unroll
        for (int j = 0; j < HC_MULT; j++)
#pragma unroll
            for (int v = 0; v < BF16_VEC; v++)
                acc[j][v] = pm[j] * xf[v];

                // acc[j][v] += sum_k comb[k][j] * residual[k][v]
#pragma unroll
        for (int k = 0; k < HC_MULT; k++)
        {
            uint4 r_raw = *reinterpret_cast<uint4 const*>(&residual[tok_res + k * hidden_size + h]);
            __nv_bfloat162 const* rp = reinterpret_cast<__nv_bfloat162 const*>(&r_raw);
            float rf[BF16_VEC];
#pragma unroll
            for (int v = 0; v < BF16_VEC / 2; v++)
            {
                float2 f = __bfloat1622float2(rp[v]);
                rf[2 * v + 0] = f.x;
                rf[2 * v + 1] = f.y;
            }

#pragma unroll
            for (int j = 0; j < HC_MULT; j++)
#pragma unroll
                for (int v = 0; v < BF16_VEC; v++)
                    acc[j][v] = fmaf(comb[k][j], rf[v], acc[j][v]);
        }

        // Store acc → out[j][h:h+8] as bf16
#pragma unroll
        for (int j = 0; j < HC_MULT; j++)
        {
            uint4 o_raw;
            __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&o_raw);
#pragma unroll
            for (int v = 0; v < BF16_VEC / 2; v++)
                op[v] = __float22bfloat162_rn(make_float2(acc[j][2 * v], acc[j][2 * v + 1]));
            *reinterpret_cast<uint4*>(&out[tok_res + j * hidden_size + h]) = o_raw;
        }
    }
}

// ===================================================================
// Kernel 5: hc_head_apply — RMS norm → sigmoid → weighted sum
//
//  Used for the HC head's final reduction: collapse `mult` input streams
//  into a single hidden-size output per token.
//
//  Per token:
//    1. Compute mix weights: sigmoid(GEMM_output * rstd * scale + base) + eps
//    2. Weighted sum:  out[h] = sum_j mix[j] * x[j][h]
//
//  mult is typically 4 (hc_mult) but passed as a runtime parameter for
//  flexibility with different HC configurations.
// ===================================================================

__launch_bounds__(256) __global__
    void mhcHcHeadApplyKernel(float const* __restrict__ mixes, float const* __restrict__ sqrsum,
        __nv_bfloat16 const* __restrict__ x, __nv_bfloat16* __restrict__ out, float const* __restrict__ scale,
        float const* __restrict__ base, int mult, int hidden_size, int K, float norm_eps, float eps)
{
    constexpr int BLOCK_SIZE = 256;
    constexpr int BF16_VEC = 8;
    constexpr int MAX_MULT = 8;

    int const token = blockIdx.x;
    int const tid = threadIdx.x;

    // Phase 1: first `mult` threads compute sigmoid gating weights
    __shared__ float s_pre[MAX_MULT];

    if (tid < mult)
    {
        float sq = sqrsum[token];
        float rstd = rsqrtf(sq / static_cast<float>(K) + norm_eps);
        float m = mixes[token * mult + tid];
        float val = m * rstd * scale[0] + base[tid];
        float sig = 1.0f / (1.0f + expf(-val));
        s_pre[tid] = sig + eps;
    }
    __syncthreads();

    // Phase 2: all threads compute weighted sum across `mult` input streams
    __nv_bfloat16 const* xbase = x + static_cast<long long>(token) * mult * hidden_size;
    __nv_bfloat16* obase = out + static_cast<long long>(token) * hidden_size;

    for (int h = tid * BF16_VEC; h < hidden_size; h += BLOCK_SIZE * BF16_VEC)
    {
        float acc[BF16_VEC] = {};

        for (int j = 0; j < mult; j++)
        {
            float pm = s_pre[j];
            uint4 raw = *reinterpret_cast<uint4 const*>(&xbase[j * hidden_size + h]);
            __nv_bfloat162 const* pairs = reinterpret_cast<__nv_bfloat162 const*>(&raw);
#pragma unroll
            for (int v = 0; v < BF16_VEC / 2; v++)
            {
                float2 f = __bfloat1622float2(pairs[v]);
                acc[2 * v + 0] += pm * f.x;
                acc[2 * v + 1] += pm * f.y;
            }
        }

        uint4 out_raw;
        __nv_bfloat162* opairs = reinterpret_cast<__nv_bfloat162*>(&out_raw);
#pragma unroll
        for (int v = 0; v < BF16_VEC / 2; v++)
            opairs[v] = __float22bfloat162_rn(make_float2(acc[2 * v], acc[2 * v + 1]));
        *reinterpret_cast<uint4*>(&obase[h]) = out_raw;
    }
}

// ===================================================================
// Offline-tuned config selection
//
// Thresholds determined by bench_mhc_gridsearch.cu on B200 (148 SMs).
// Fixed params:  K=16384, hidden_size=4096, N∈{4,24}.
// Run bench_mhc_gridsearch and update thresholds if GPU changes.
// ===================================================================

// FMA: select tileN given M and N.  (B200, 148 SMs, K=16384)
//   N=4  (HCHead):      valid tileN ∈ {1, 2, 4}
//   N=24 (pre_mapping):  valid tileN ∈ {1, 2, 3, 4, 6, 8, 12, 24}
static int selectFmaTileN(int M, int N)
{
    if (N <= 4)
    {
        if (M <= 128)
            return 1;
        return 2;
    }
    if (M <= 32)
        return 1;
    return 8;
}

// BigFuse: select BLOCK_SIZE given M. Thresholds match the production autotuner
// fallback; we use 128 threads for tiny-M waves (too few tokens to hide global
// scoreboard latency at 256) and 256 otherwise. 512 is not picked here because
// it only wins when M is very large, which path B/D do not route through this
// helper.
static int selectBigFuseBlockSize(int M)
{
    if (M <= 16)
        return 128;
    return 256;
}

// ===================================================================
// Launch wrappers
//
// Two-level dispatch for BigFuse (NUM_SPLITS × BLOCK_SIZE) and a
// switch-dispatch for the FMA GEMM tile size.  These are the public
// C++ entry points called by the PyTorch custom-op layer.
// ===================================================================

template <int NUM_SPLITS, bool kFuseNorm>
static void mhcBigFuseDispatch(float const* y_acc, float const* r_acc, __nv_bfloat16 const* residual,
    float const* hc_scale, float const* hc_base, float* post_mix, float* comb_mix, __nv_bfloat16* layer_input, int M,
    int K, int hidden_size, float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps, float hc_post_mult_value,
    int sinkhorn_repeat, int block_size, __nv_bfloat16 const* norm_weight, float norm_eps, cudaStream_t stream)
{
    dim3 grid(static_cast<unsigned int>(M));

#define LAUNCH_BF(BS)                                                                                                  \
    mhcBigFuseKernel<NUM_SPLITS, BS, kFuseNorm><<<grid, BS, 0, stream>>>(y_acc, r_acc, residual, hc_scale, hc_base,    \
        post_mix, comb_mix, layer_input, M, K, hidden_size, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,  \
        sinkhorn_repeat, norm_weight, norm_eps)

    if (block_size >= 512)
    {
        LAUNCH_BF(512);
    }
    else if (block_size >= 256)
    {
        LAUNCH_BF(256);
    }
    else
    {
        LAUNCH_BF(128);
    }
#undef LAUNCH_BF
}

void mhcBigFuseLaunch(float const* y_acc, float const* r_acc, __nv_bfloat16 const* residual, float const* hc_scale,
    float const* hc_base, float* post_mix, float* comb_mix, __nv_bfloat16* layer_input, int M, int K, int hidden_size,
    float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps, float hc_post_mult_value, int sinkhorn_repeat,
    int num_splits, int block_size, __nv_bfloat16 const* norm_weight, float norm_eps, cudaStream_t stream)
{
    if (M <= 0)
        return;

    TLLM_CHECK_WITH_INFO(num_splits == 1 || num_splits == 2 || num_splits == 4 || num_splits == 8 || num_splits == 16,
        "mhcBigFuseLaunch: only num_splits ∈ {1,2,4,8,16} supported, got %d", num_splits);

    int const bs = (block_size > 0) ? block_size : selectBigFuseBlockSize(M);
    bool const fuse_norm = (norm_weight != nullptr);

#define DISPATCH_BF_INNER(NS, FN)                                                                                      \
    mhcBigFuseDispatch<NS, FN>(y_acc, r_acc, residual, hc_scale, hc_base, post_mix, comb_mix, layer_input, M, K,       \
        hidden_size, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, bs, norm_weight,       \
        norm_eps, stream)
#define DISPATCH_BF(NS) (fuse_norm ? DISPATCH_BF_INNER(NS, true) : DISPATCH_BF_INNER(NS, false))

    switch (num_splits)
    {
    case 1: DISPATCH_BF(1); break;
    case 2: DISPATCH_BF(2); break;
    case 4: DISPATCH_BF(4); break;
    case 8: DISPATCH_BF(8); break;
    case 16: DISPATCH_BF(16); break;
    }
#undef DISPATCH_BF
#undef DISPATCH_BF_INNER
}

void mhcGemmSqrsumFmaLaunch(__nv_bfloat16 const* x, float const* w_t, float* y, float* r, int M, int N, int K,
    int tile_n, int tile_m, cudaStream_t stream)
{
    if (M <= 0)
        return;

    (void) tile_m; // reserved for future multi-row kernel variant

    int const tileN = (tile_n > 0) ? tile_n : selectFmaTileN(M, N);

    TLLM_CHECK_WITH_INFO(N % tileN == 0, "mhcGemmSqrsumFmaLaunch: N=%d not divisible by tile_n=%d", N, tileN);

#define LAUNCH_FMA(TN) mhcGemmSqrsumFmaKernel<TN><<<dim3(M, (N + TN - 1) / TN), 256, 0, stream>>>(x, w_t, y, r, M, N, K)

    switch (tileN)
    {
    case 1: LAUNCH_FMA(1); break;
    case 2: LAUNCH_FMA(2); break;
    case 3: LAUNCH_FMA(3); break;
    case 4: LAUNCH_FMA(4); break;
    case 6: LAUNCH_FMA(6); break;
    case 8: LAUNCH_FMA(8); break;
    case 12: LAUNCH_FMA(12); break;
    default: LAUNCH_FMA(24); break;
    }
#undef LAUNCH_FMA
}

void mhcHcHeadApplyLaunch(float const* mixes, float const* sqrsum, __nv_bfloat16 const* x, __nv_bfloat16* out,
    float const* scale, float const* base, int M, int mult, int hidden_size, int K, float norm_eps, float eps,
    cudaStream_t stream)
{
    dim3 grid(static_cast<unsigned int>(M));
    dim3 block(256);

    mhcHcHeadApplyKernel<<<grid, block, 0, stream>>>(
        mixes, sqrsum, x, out, scale, base, mult, hidden_size, K, norm_eps, eps);
}

void mhcPostMappingLaunch(__nv_bfloat16 const* residual, __nv_bfloat16 const* x, float const* post_mix,
    float const* comb_mix, __nv_bfloat16* out, int B, int hidden_size, cudaStream_t stream)
{
    dim3 grid(static_cast<unsigned int>(B));
    dim3 block(256);

    mhcPostMappingKernel<<<grid, block, 0, stream>>>(residual, x, post_mix, comb_mix, out, hidden_size);
}

} // namespace kernels::mhc

TRTLLM_NAMESPACE_END
