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

// CUDA-core FMA fused post_mapping + pre_mapping kernels for the mhc
// fused_hc pipeline. Provides the small-M variants:
//   - fused_pmap_gemm_fma_ksplit   (Path E, 2-kernel: this + mhcBigFuseKernel)
//   - fused_pmap_gemm_fma_allinone (Path F, 1-kernel: pmap + GEMM + bigFuse)
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace fused_fma_kernels
{

// ===================================================================
// Fused K-split pmap+GEMM FMA: new_r is NEVER materialized to HBM.  Each CTA
// scans one HIDDEN-slice inline: for each h in its slice it loads residual[j,h]
// for ALL HC j-indices plus x[h], computes FULL new_r[j,h] = pm[j]*x[h] + Σ_k
// comb[k,j]*r[k,h] in registers, then multiplies by W_T and accumulates partial
// acc+sqr.  Writes Yp[k_split, token, n] and Rp[k_split, token].  A reducer
// kernel sums over splits.  Eliminates the 32-KB/token new_r round-trip at tiny M.
//
// CORRECTNESS: We split the HIDDEN dimension (not HC).  For each fixed h,
// new_r[*, h] depends only on x[h] and residual[*, h]; the HC summation is
// fully local to h, so a HIDDEN-split is algebraically exact.  The final GEMM
// acc[n] = Σ_{j,h} new_r[j,h] * W_T[n, j*HIDDEN+h] is then partitioned into
// disjoint h-ranges across splits and reduced by mhcBigFuseKernel.
//
// Grid: (M, N/N_PER_BLOCK, NUM_K_SPLITS).  Block: 256 threads.
// NUM_K_SPLITS must evenly divide HIDDEN (typically ∈ {2, 4, 8}).
// BF16_VEC_OVERRIDE lets the caller force a smaller per-thread vector (e.g. 4)
// so that BLOCK_SIZE*BF16_VEC == HIDDEN_PER_SPLIT and every thread is active.
// Pass 0 to auto-pick: 8 for ks=2, 4 for ks=4, 2 for ks=8.
// ===================================================================
// Optional WRITE_RESIDUAL path: when true, the CTA with (blockIdx.y == 0) emits
// the post-mapped residual (bf16 [M, HC_MULT, hidden]) into residual_out, using
// the same disjoint h-slice it already owns.  When false the residual_out arg
// may be a nullptr; the parameter stays in the signature for simplicity.
template <int N_PER_BLOCK, int NUM_K_SPLITS, int BF16_VEC_OVERRIDE = 0, bool WRITE_RESIDUAL = false>
__launch_bounds__(256) __global__ void fused_pmap_gemm_fma_ksplit(__nv_bfloat16 const* __restrict__ residual_in,
    __nv_bfloat16 const* __restrict__ x_in, float const* __restrict__ post_mix, float const* __restrict__ comb_mix,
    float const* __restrict__ W_T, float* __restrict__ Yp, float* __restrict__ Rp, int hidden_size, int N, int K,
    __nv_bfloat16* __restrict__ residual_out = nullptr)
{
    static_assert(NUM_K_SPLITS == 1 || NUM_K_SPLITS == 2 || NUM_K_SPLITS == 4 || NUM_K_SPLITS == 8,
        "ksplit ∈ {1, 2, 4, 8} supported");
    constexpr int HC_MULT = 4;
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    // Default BF16_VEC by ks so that BLOCK_SIZE*BF16_VEC = HIDDEN/ks at HIDDEN=4096.
    // ks=1 takes the full HIDDEN in one slice; reuses the BF16_VEC=8 (uint4) load path.
    constexpr int BF16_VEC = (BF16_VEC_OVERRIDE != 0) ? BF16_VEC_OVERRIDE
                                                      : ((NUM_K_SPLITS == 1 || NUM_K_SPLITS == 2) ? 8
                                                              : (NUM_K_SPLITS == 4)               ? 4
                                                                                                  : 2);
    static_assert(BF16_VEC == 2 || BF16_VEC == 4 || BF16_VEC == 8, "BF16_VEC must be 2, 4, or 8");

    int const token = blockIdx.x;
    int const n_start = blockIdx.y * N_PER_BLOCK;
    int const k_split = blockIdx.z;
    int const tid = threadIdx.x;
    int const warp_id = tid / WARP_SIZE;
    int const lane = tid % WARP_SIZE;
    bool const do_sqr = (n_start == 0);

    int const hidden_per_split = hidden_size / NUM_K_SPLITS;
    int const h_start = k_split * hidden_per_split;
    int const h_end = h_start + hidden_per_split;

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

    float pm[HC_MULT];
#pragma unroll
    for (int j = 0; j < HC_MULT; j++)
        pm[j] = s_post[j];

    float acc[N_PER_BLOCK];
#pragma unroll
    for (int n = 0; n < N_PER_BLOCK; n++)
        acc[n] = 0.0f;
    float sqr = 0.0f;

    long long const tok_res = static_cast<long long>(token) * HC_MULT * hidden_size;
    long long const tok_x = static_cast<long long>(token) * hidden_size;

    // Iterate h over this CTA's HIDDEN slice.  Each thread owns BF16_VEC
    // contiguous positions per step; stride BLOCK_SIZE*BF16_VEC.
    for (int h = h_start + tid * BF16_VEC; h < h_end; h += BLOCK_SIZE * BF16_VEC)
    {
        // Load x[h..h+BF16_VEC).  Vector width depends on BF16_VEC.
        float xf[BF16_VEC];
        if constexpr (BF16_VEC == 8)
        {
            uint4 x_raw = *reinterpret_cast<uint4 const*>(&x_in[tok_x + h]);
            __nv_bfloat162 const* xp = reinterpret_cast<__nv_bfloat162 const*>(&x_raw);
#pragma unroll
            for (int v = 0; v < 4; v++)
            {
                float2 f = __bfloat1622float2(xp[v]);
                xf[2 * v + 0] = f.x;
                xf[2 * v + 1] = f.y;
            }
        }
        else if constexpr (BF16_VEC == 4)
        {
            uint2 x_raw = *reinterpret_cast<uint2 const*>(&x_in[tok_x + h]);
            __nv_bfloat162 const* xp = reinterpret_cast<__nv_bfloat162 const*>(&x_raw);
#pragma unroll
            for (int v = 0; v < 2; v++)
            {
                float2 f = __bfloat1622float2(xp[v]);
                xf[2 * v + 0] = f.x;
                xf[2 * v + 1] = f.y;
            }
        }
        else
        { // BF16_VEC == 2
            unsigned x_raw = *reinterpret_cast<unsigned const*>(&x_in[tok_x + h]);
            __nv_bfloat162 xp = *reinterpret_cast<__nv_bfloat162 const*>(&x_raw);
            float2 f = __bfloat1622float2(xp);
            xf[0] = f.x;
            xf[1] = f.y;
        }

        // Seed new_r[j, v] = pm[j] * x[v] for all j
        float new_r[HC_MULT][BF16_VEC];
#pragma unroll
        for (int j = 0; j < HC_MULT; j++)
#pragma unroll
            for (int v = 0; v < BF16_VEC; v++)
                new_r[j][v] = pm[j] * xf[v];

// Add Σ_k comb[k,j] * r[k, v] — FULL HC sum (all k ∈ [0, HC_MULT))
#pragma unroll
        for (int k = 0; k < HC_MULT; k++)
        {
            float rf_k[BF16_VEC];
            __nv_bfloat16 const* r_ptr = &residual_in[tok_res + k * hidden_size + h];
            if constexpr (BF16_VEC == 8)
            {
                uint4 r_raw = *reinterpret_cast<uint4 const*>(r_ptr);
                __nv_bfloat162 const* rp = reinterpret_cast<__nv_bfloat162 const*>(&r_raw);
#pragma unroll
                for (int v = 0; v < 4; v++)
                {
                    float2 f = __bfloat1622float2(rp[v]);
                    rf_k[2 * v + 0] = f.x;
                    rf_k[2 * v + 1] = f.y;
                }
            }
            else if constexpr (BF16_VEC == 4)
            {
                uint2 r_raw = *reinterpret_cast<uint2 const*>(r_ptr);
                __nv_bfloat162 const* rp = reinterpret_cast<__nv_bfloat162 const*>(&r_raw);
#pragma unroll
                for (int v = 0; v < 2; v++)
                {
                    float2 f = __bfloat1622float2(rp[v]);
                    rf_k[2 * v + 0] = f.x;
                    rf_k[2 * v + 1] = f.y;
                }
            }
            else
            { // BF16_VEC == 2
                unsigned r_raw = *reinterpret_cast<unsigned const*>(r_ptr);
                __nv_bfloat162 rp = *reinterpret_cast<__nv_bfloat162 const*>(&r_raw);
                float2 f = __bfloat1622float2(rp);
                rf_k[0] = f.x;
                rf_k[1] = f.y;
            }
#pragma unroll
            for (int j = 0; j < HC_MULT; j++)
#pragma unroll
                for (int v = 0; v < BF16_VEC; v++)
                    new_r[j][v] = fmaf(s_comb[k][j], rf_k[v], new_r[j][v]);
        }

        if (do_sqr)
        {
#pragma unroll
            for (int j = 0; j < HC_MULT; j++)
#pragma unroll
                for (int v = 0; v < BF16_VEC; v++)
                    sqr = fmaf(new_r[j][v], new_r[j][v], sqr);
        }

        // Optional: emit the post-mapped residual to HBM (bf16) for downstream
        // bigfuse.  Guard on WRITE_RESIDUAL (compile-time) AND do_sqr so only
        // one n-tile per (token, k_split) writes — other n-tiles compute the
        // same new_r but we don't want duplicate writes.
        if constexpr (WRITE_RESIDUAL)
        {
            if (do_sqr)
            {
#pragma unroll
                for (int j = 0; j < HC_MULT; j++)
                {
                    __nv_bfloat16* o_ptr = residual_out + tok_res + j * hidden_size + h;
                    if constexpr (BF16_VEC == 8)
                    {
                        uint4 packed;
                        __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
                        for (int v = 0; v < 4; v++)
                            pairs[v] = __float22bfloat162_rn(make_float2(new_r[j][2 * v], new_r[j][2 * v + 1]));
                        *reinterpret_cast<uint4*>(o_ptr) = packed;
                    }
                    else if constexpr (BF16_VEC == 4)
                    {
                        uint2 packed;
                        __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
                        for (int v = 0; v < 2; v++)
                            pairs[v] = __float22bfloat162_rn(make_float2(new_r[j][2 * v], new_r[j][2 * v + 1]));
                        *reinterpret_cast<uint2*>(o_ptr) = packed;
                    }
                    else
                    { // BF16_VEC == 2
                        __nv_bfloat162 packed = __float22bfloat162_rn(make_float2(new_r[j][0], new_r[j][1]));
                        *reinterpret_cast<unsigned*>(o_ptr) = *reinterpret_cast<unsigned*>(&packed);
                    }
                }
            }
        }

// GEMM partial: for each n, contract over ALL HC j and this h-slice.
// W_T layout: W_T[n, K=HC_MULT*hidden_size], inner dim packed as (j, h).
#pragma unroll
        for (int n = 0; n < N_PER_BLOCK; n++)
        {
#pragma unroll
            for (int j = 0; j < HC_MULT; j++)
            {
                long long const w_base = static_cast<long long>(n_start + n) * K + j * hidden_size + h;
                float w[BF16_VEC];
                if constexpr (BF16_VEC == 8)
                {
                    float w0, w1, w2, w3, w4, w5, w6, w7;
                    asm volatile("ld.global.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"
                                 : "=f"(w0), "=f"(w1), "=f"(w2), "=f"(w3)
                                 : "l"(W_T + w_base));
                    asm volatile("ld.global.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"
                                 : "=f"(w4), "=f"(w5), "=f"(w6), "=f"(w7)
                                 : "l"(W_T + w_base + 4));
                    w[0] = w0;
                    w[1] = w1;
                    w[2] = w2;
                    w[3] = w3;
                    w[4] = w4;
                    w[5] = w5;
                    w[6] = w6;
                    w[7] = w7;
                }
                else if constexpr (BF16_VEC == 4)
                {
                    float w0, w1, w2, w3;
                    asm volatile("ld.global.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"
                                 : "=f"(w0), "=f"(w1), "=f"(w2), "=f"(w3)
                                 : "l"(W_T + w_base));
                    w[0] = w0;
                    w[1] = w1;
                    w[2] = w2;
                    w[3] = w3;
                }
                else
                { // BF16_VEC == 2
                    float w0, w1;
                    asm volatile("ld.global.L1::evict_last.v2.f32 {%0, %1}, [%2];"
                                 : "=f"(w0), "=f"(w1)
                                 : "l"(W_T + w_base));
                    w[0] = w0;
                    w[1] = w1;
                }
#pragma unroll
                for (int v = 0; v < BF16_VEC; v++)
                    acc[n] = fmaf(new_r[j][v], w[v], acc[n]);
            }
        }
    }

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
        if (lane == 0 && warp_id < n_cols)
        {
            float val = 0.0f;
#pragma unroll
            for (int w = 0; w < NUM_WARPS; w++)
                val += s_warp[w][warp_id];
            // Yp layout: [NUM_K_SPLITS, M, N]  (M = gridDim.x)
            int const n_global = n_start + g * COLS_PER_GROUP + warp_id;
            Yp[(static_cast<long long>(k_split) * static_cast<long long>(gridDim.x) + token) * N + n_global] = val;
        }
        if (g == 0 && do_sqr && lane == 0 && warp_id == 0)
        {
            float sq = 0.0f;
#pragma unroll
            for (int w = 0; w < NUM_WARPS; w++)
                sq += s_warp[w][SQRSUM_SLOT];
            // Rp layout: [NUM_K_SPLITS, M]
            Rp[static_cast<long long>(k_split) * static_cast<long long>(gridDim.x) + token] = sq;
        }
        __syncthreads();
    }
}

// ===================================================================
// fused_pmap_gemm_fma_allinone<TN, KS, TM, FULL_N>
// -------------------------------------------------------------------
// 1-kernel FMA path: pmap + GEMM + bigFuse, all in a single launch.
// Parallelism: TM tokens per CTA × TN N-columns × KS K-slices.
// Grid:  (ceil(M / TM), FULL_N / TN, KS)
// Block: 256 threads.
//
// Uses atomicAdd into y_acc[M, FULL_N] and r_acc[M] for cross-CTA reduction
// (KS > 1 and/or TN < FULL_N).  Uses per-token-batch atomic counter to elect
// the last-home CTA, which __threadfences, reloads the now-complete y_acc /
// r_acc / residual_cur rows, and runs bigFuse inline to produce
// post_mix_out, comb_mix_out, layer_input_out.
//
// Caller MUST zero y_acc[M, FULL_N], r_acc[M], done_counter[ceil(M / TM)]
// before launch.  FULL_N must equal HC_MULT * (2 + HC_MULT) = 24.
// ===================================================================
template <int TN, int KS, int TM = 1, int FULL_N = 24, int BF16_VEC_OVERRIDE = 0>
__launch_bounds__(256) __global__ void fused_pmap_gemm_fma_allinone(__nv_bfloat16 const* __restrict__ residual_in,
    __nv_bfloat16 const* __restrict__ x_in, float const* __restrict__ post_mix_prev,
    float const* __restrict__ comb_mix_prev, float const* __restrict__ W_T, float const* __restrict__ hc_scale,
    float const* __restrict__ hc_base, __nv_bfloat16* __restrict__ residual_out, float* __restrict__ post_mix_out,
    float* __restrict__ comb_mix_out, __nv_bfloat16* __restrict__ layer_input_out, float* __restrict__ y_acc,
    float* __restrict__ r_acc, int* __restrict__ done_counter, int M, int K, int hidden_size, float rms_eps,
    float hc_pre_eps, float hc_sinkhorn_eps, float hc_post_mult_value, int sinkhorn_repeat)
{
    constexpr int HC_MULT = 4;
    constexpr int HC_MULT2 = HC_MULT * HC_MULT;
    constexpr int HC_MULT3 = HC_MULT * (2 + HC_MULT);
    static_assert(TM >= 1, "TM>=1");
    static_assert(FULL_N % TN == 0, "TN must divide FULL_N");
    static_assert(HC_MULT3 == FULL_N, "FULL_N must be HC_MULT*(2+HC_MULT)=24");
    static_assert(KS == 1 || KS == 2 || KS == 4 || KS == 8, "KS in {1,2,4,8}");
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    constexpr int BF16_VEC = (BF16_VEC_OVERRIDE != 0) ? BF16_VEC_OVERRIDE
                                                      : ((KS == 1 || KS == 2) ? 8
                                                              : (KS == 4)     ? 4
                                                                              : 2);
    static_assert(BF16_VEC == 2 || BF16_VEC == 4 || BF16_VEC == 8, "BF16_VEC in {2,4,8}");
    constexpr int CTAS_PER_BATCH = (FULL_N / TN) * KS;

    int const base_tok = blockIdx.x * TM;
    int const n_start = blockIdx.y * TN;
    int const k_split = blockIdx.z;
    int const tid = threadIdx.x;
    int const warp_id = tid / WARP_SIZE;
    int const lane = tid % WARP_SIZE;
    bool const is_n0 = (n_start == 0);

    if (base_tok >= M)
        return;

    int const hidden_per_split = hidden_size / KS;
    int const h_lo = k_split * hidden_per_split;
    int const h_hi = h_lo + hidden_per_split;

    bool valid[TM];
#pragma unroll
    for (int t = 0; t < TM; t++)
        valid[t] = (base_tok + t) < M;

    __shared__ float s_pm[TM][HC_MULT];
    __shared__ float s_cm[TM][HC_MULT][HC_MULT];

    {
        constexpr int N_PM = TM * HC_MULT;
        for (int i = tid; i < N_PM; i += BLOCK_SIZE)
        {
            int const t = i / HC_MULT;
            int const j = i % HC_MULT;
            float v = 0.0f;
            if (valid[t])
                v = post_mix_prev[(base_tok + t) * HC_MULT + j];
            s_pm[t][j] = v;
        }
    }
    {
        constexpr int N_CM = TM * HC_MULT * HC_MULT;
        for (int i = tid; i < N_CM; i += BLOCK_SIZE)
        {
            int const t = i / HC_MULT2;
            int const r = (i / HC_MULT) % HC_MULT;
            int const c = i % HC_MULT;
            float v = 0.0f;
            if (valid[t])
                v = comb_mix_prev[(base_tok + t) * HC_MULT2 + r * HC_MULT + c];
            s_cm[t][r][c] = v;
        }
    }
    __syncthreads();

    float acc[TM][TN];
    float sqr[TM];
#pragma unroll
    for (int t = 0; t < TM; t++)
    {
        sqr[t] = 0.0f;
#pragma unroll
        for (int n = 0; n < TN; n++)
            acc[t][n] = 0.0f;
    }

    // Phase 1: pmap + GEMM + residual_out write + sqrsum accumulate.
    // Iterate h over this CTA's HIDDEN slice; each thread owns BF16_VEC.
    for (int h = h_lo + tid * BF16_VEC; h < h_hi; h += BLOCK_SIZE * BF16_VEC)
    {
#pragma unroll
        for (int t = 0; t < TM; t++)
        {
            if (!valid[t])
                continue;
            long long const tok_x = static_cast<long long>(base_tok + t) * hidden_size;
            long long const tok_r = static_cast<long long>(base_tok + t) * HC_MULT * hidden_size;

            float xf[BF16_VEC];
            if constexpr (BF16_VEC == 8)
            {
                uint4 x_raw = *reinterpret_cast<uint4 const*>(&x_in[tok_x + h]);
                __nv_bfloat162 const* xp = reinterpret_cast<__nv_bfloat162 const*>(&x_raw);
#pragma unroll
                for (int v = 0; v < 4; v++)
                {
                    float2 f = __bfloat1622float2(xp[v]);
                    xf[2 * v + 0] = f.x;
                    xf[2 * v + 1] = f.y;
                }
            }
            else if constexpr (BF16_VEC == 4)
            {
                uint2 x_raw = *reinterpret_cast<uint2 const*>(&x_in[tok_x + h]);
                __nv_bfloat162 const* xp = reinterpret_cast<__nv_bfloat162 const*>(&x_raw);
#pragma unroll
                for (int v = 0; v < 2; v++)
                {
                    float2 f = __bfloat1622float2(xp[v]);
                    xf[2 * v + 0] = f.x;
                    xf[2 * v + 1] = f.y;
                }
            }
            else
            {
                unsigned x_raw = *reinterpret_cast<unsigned const*>(&x_in[tok_x + h]);
                __nv_bfloat162 xp = *reinterpret_cast<__nv_bfloat162 const*>(&x_raw);
                float2 f = __bfloat1622float2(xp);
                xf[0] = f.x;
                xf[1] = f.y;
            }

            float rf[HC_MULT][BF16_VEC];
#pragma unroll
            for (int k = 0; k < HC_MULT; k++)
            {
                __nv_bfloat16 const* r_ptr = &residual_in[tok_r + k * hidden_size + h];
                if constexpr (BF16_VEC == 8)
                {
                    uint4 r_raw = *reinterpret_cast<uint4 const*>(r_ptr);
                    __nv_bfloat162 const* rp = reinterpret_cast<__nv_bfloat162 const*>(&r_raw);
#pragma unroll
                    for (int v = 0; v < 4; v++)
                    {
                        float2 f = __bfloat1622float2(rp[v]);
                        rf[k][2 * v + 0] = f.x;
                        rf[k][2 * v + 1] = f.y;
                    }
                }
                else if constexpr (BF16_VEC == 4)
                {
                    uint2 r_raw = *reinterpret_cast<uint2 const*>(r_ptr);
                    __nv_bfloat162 const* rp = reinterpret_cast<__nv_bfloat162 const*>(&r_raw);
#pragma unroll
                    for (int v = 0; v < 2; v++)
                    {
                        float2 f = __bfloat1622float2(rp[v]);
                        rf[k][2 * v + 0] = f.x;
                        rf[k][2 * v + 1] = f.y;
                    }
                }
                else
                {
                    unsigned r_raw = *reinterpret_cast<unsigned const*>(r_ptr);
                    __nv_bfloat162 rp = *reinterpret_cast<__nv_bfloat162 const*>(&r_raw);
                    float2 f = __bfloat1622float2(rp);
                    rf[k][0] = f.x;
                    rf[k][1] = f.y;
                }
            }

            float pm[HC_MULT];
            float cm[HC_MULT][HC_MULT];
#pragma unroll
            for (int j = 0; j < HC_MULT; j++)
                pm[j] = s_pm[t][j];
#pragma unroll
            for (int k = 0; k < HC_MULT; k++)
#pragma unroll
                for (int j = 0; j < HC_MULT; j++)
                    cm[k][j] = s_cm[t][k][j];

            float new_r[HC_MULT][BF16_VEC];
#pragma unroll
            for (int j = 0; j < HC_MULT; j++)
#pragma unroll
                for (int v = 0; v < BF16_VEC; v++)
                    new_r[j][v] = pm[j] * xf[v];
#pragma unroll
            for (int k = 0; k < HC_MULT; k++)
#pragma unroll
                for (int j = 0; j < HC_MULT; j++)
#pragma unroll
                    for (int v = 0; v < BF16_VEC; v++)
                        new_r[j][v] = fmaf(cm[k][j], rf[k][v], new_r[j][v]);

            if (is_n0)
            {
#pragma unroll
                for (int j = 0; j < HC_MULT; j++)
                {
                    __nv_bfloat16* o_ptr = residual_out + tok_r + j * hidden_size + h;
                    if constexpr (BF16_VEC == 8)
                    {
                        uint4 packed;
                        __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
                        for (int v = 0; v < 4; v++)
                            pairs[v] = __float22bfloat162_rn(make_float2(new_r[j][2 * v], new_r[j][2 * v + 1]));
                        *reinterpret_cast<uint4*>(o_ptr) = packed;
                    }
                    else if constexpr (BF16_VEC == 4)
                    {
                        uint2 packed;
                        __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
                        for (int v = 0; v < 2; v++)
                            pairs[v] = __float22bfloat162_rn(make_float2(new_r[j][2 * v], new_r[j][2 * v + 1]));
                        *reinterpret_cast<uint2*>(o_ptr) = packed;
                    }
                    else
                    {
                        __nv_bfloat162 packed = __float22bfloat162_rn(make_float2(new_r[j][0], new_r[j][1]));
                        *reinterpret_cast<unsigned*>(o_ptr) = *reinterpret_cast<unsigned*>(&packed);
                    }
                }

#pragma unroll
                for (int j = 0; j < HC_MULT; j++)
#pragma unroll
                    for (int v = 0; v < BF16_VEC; v++)
                        sqr[t] = fmaf(new_r[j][v], new_r[j][v], sqr[t]);
            }

#pragma unroll
            for (int n = 0; n < TN; n++)
            {
#pragma unroll
                for (int j = 0; j < HC_MULT; j++)
                {
                    long long const w_base = static_cast<long long>(n_start + n) * K + j * hidden_size + h;
                    float w[BF16_VEC];
                    if constexpr (BF16_VEC == 8)
                    {
                        float w0, w1, w2, w3, w4, w5, w6, w7;
                        asm volatile("ld.global.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"
                                     : "=f"(w0), "=f"(w1), "=f"(w2), "=f"(w3)
                                     : "l"(W_T + w_base));
                        asm volatile("ld.global.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"
                                     : "=f"(w4), "=f"(w5), "=f"(w6), "=f"(w7)
                                     : "l"(W_T + w_base + 4));
                        w[0] = w0;
                        w[1] = w1;
                        w[2] = w2;
                        w[3] = w3;
                        w[4] = w4;
                        w[5] = w5;
                        w[6] = w6;
                        w[7] = w7;
                    }
                    else if constexpr (BF16_VEC == 4)
                    {
                        float w0, w1, w2, w3;
                        asm volatile("ld.global.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"
                                     : "=f"(w0), "=f"(w1), "=f"(w2), "=f"(w3)
                                     : "l"(W_T + w_base));
                        w[0] = w0;
                        w[1] = w1;
                        w[2] = w2;
                        w[3] = w3;
                    }
                    else
                    {
                        float w0, w1;
                        asm volatile("ld.global.L1::evict_last.v2.f32 {%0, %1}, [%2];"
                                     : "=f"(w0), "=f"(w1)
                                     : "l"(W_T + w_base));
                        w[0] = w0;
                        w[1] = w1;
                    }
#pragma unroll
                    for (int v = 0; v < BF16_VEC; v++)
                        acc[t][n] = fmaf(new_r[j][v], w[v], acc[t][n]);
                }
            }
        }
    }

    // Phase 2: warp-reduce + SMEM cross-warp + atomicAdd into y_acc / r_acc.
    constexpr int COLS_PER_GROUP = NUM_WARPS;
    constexpr int SQRSUM_SLOT = COLS_PER_GROUP;
    constexpr int N_GROUPS = (TN + COLS_PER_GROUP - 1) / COLS_PER_GROUP;
    constexpr unsigned FULL_WARP_MASK = 0xffffffff;

    __shared__ float s_warp[NUM_WARPS][COLS_PER_GROUP + 1];

#pragma unroll
    for (int t = 0; t < TM; t++)
    {
        if (!valid[t])
            continue;
#pragma unroll
        for (int g = 0; g < N_GROUPS; g++)
        {
            constexpr int LAST_COLS = TN - (N_GROUPS - 1) * COLS_PER_GROUP;
            int const n_cols = (g == N_GROUPS - 1) ? LAST_COLS : COLS_PER_GROUP;
#pragma unroll
            for (int n = 0; n < COLS_PER_GROUP; n++)
            {
                if (n < n_cols)
                {
#pragma unroll
                    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                        acc[t][g * COLS_PER_GROUP + n]
                            += __shfl_xor_sync(FULL_WARP_MASK, acc[t][g * COLS_PER_GROUP + n], off);
                }
            }
            if (g == 0 && is_n0)
            {
#pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
                    sqr[t] += __shfl_xor_sync(FULL_WARP_MASK, sqr[t], off);
            }
            if (lane == 0)
            {
#pragma unroll
                for (int n = 0; n < COLS_PER_GROUP; n++)
                    if (n < n_cols)
                        s_warp[warp_id][n] = acc[t][g * COLS_PER_GROUP + n];
                if (g == 0 && is_n0)
                    s_warp[warp_id][SQRSUM_SLOT] = sqr[t];
            }
            __syncthreads();
            if (lane == 0 && warp_id < n_cols)
            {
                float val = 0.0f;
#pragma unroll
                for (int ww = 0; ww < NUM_WARPS; ww++)
                    val += s_warp[ww][warp_id];
                long long y_idx
                    = static_cast<long long>(base_tok + t) * FULL_N + n_start + g * COLS_PER_GROUP + warp_id;
                if constexpr (KS == 1)
                {
                    y_acc[y_idx] = val;
                }
                else
                {
                    atomicAdd(y_acc + y_idx, val);
                }
            }
            if (g == 0 && is_n0 && lane == 0 && warp_id == 0)
            {
                float sq = 0.0f;
#pragma unroll
                for (int ww = 0; ww < NUM_WARPS; ww++)
                    sq += s_warp[ww][SQRSUM_SLOT];
                if constexpr (KS == 1)
                {
                    r_acc[base_tok + t] = sq;
                }
                else
                {
                    atomicAdd(r_acc + (base_tok + t), sq);
                }
            }
            __syncthreads();
        }
    }

    // Phase 3: elect last-home CTA per (ceil(M/TM)) token-batch via counter.
    __shared__ int s_is_last;
    if constexpr (CTAS_PER_BATCH == 1)
    {
        s_is_last = 1;
    }
    else
    {
        __threadfence();
        __syncthreads();
        if (tid == 0)
        {
            int const prev = atomicAdd(&done_counter[blockIdx.x], 1);
            s_is_last = (prev == CTAS_PER_BATCH - 1) ? 1 : 0;
        }
        __syncthreads();
    }
    if (!s_is_last)
        return;

    // Phase 4: inline bigFuse for the TM tokens in this batch.
    // Layout: FULL_N = HC_MULT*(2+HC_MULT) = 24
    //   y_local[0..HC_MULT)               → s_pre_mix (sigmoid gate)
    //   y_local[HC_MULT..2*HC_MULT)       → post_mix_out (sigmoid*hc_post_mult)
    //   y_local[2*HC_MULT..FULL_N)        → comb_mix_out (Sinkhorn)
    __shared__ float s_pre_mix[TM][HC_MULT];

#pragma unroll
    for (int t = 0; t < TM; t++)
    {
        if (!valid[t])
        {
            __syncthreads();
            continue;
        }
        int const tok = base_tok + t;
        float cm_vals[HC_MULT];

        if (warp_id == 0 && lane < HC_MULT)
        {
            float const r_val = r_acc[tok];
            float y_local[HC_MULT3];
            float const* y_row = y_acc + static_cast<long long>(tok) * FULL_N;
#pragma unroll
            for (int c = 0; c < HC_MULT3; c++)
                y_local[c] = y_row[c];

            float const rstd = rsqrtf(r_val / static_cast<float>(K) + rms_eps);
            float const s0 = hc_scale[0], s1 = hc_scale[1], s2 = hc_scale[2];

            float v = y_local[lane] * rstd * s0 + hc_base[lane];
            s_pre_mix[t][lane] = 1.0f / (1.0f + expf(-v)) + hc_pre_eps;

            v = y_local[HC_MULT + lane] * rstd * s1 + hc_base[HC_MULT + lane];
            post_mix_out[tok * HC_MULT + lane] = 1.0f / (1.0f + expf(-v)) * hc_post_mult_value;

#pragma unroll
            for (int k = 0; k < HC_MULT; k++)
                cm_vals[k]
                    = y_local[2 * HC_MULT + lane * HC_MULT + k] * rstd * s2 + hc_base[2 * HC_MULT + lane * HC_MULT + k];

            constexpr unsigned LANE_MASK = (1u << HC_MULT) - 1;
            float const rowMax
                = fmaxf(fmaxf(cm_vals[0], cm_vals[1]), fmaxf(cm_vals[2], cm_vals[3]));
#pragma unroll
            for (int k = 0; k < HC_MULT; k++)
                cm_vals[k] = expf(cm_vals[k] - rowMax);
            float rs = cm_vals[0] + cm_vals[1] + cm_vals[2] + cm_vals[3];
#pragma unroll
            for (int k = 0; k < HC_MULT; k++)
                cm_vals[k] = cm_vals[k] / rs + hc_sinkhorn_eps;
#pragma unroll
            for (int k = 0; k < HC_MULT; k++)
            {
                float cs = cm_vals[k];
                cs += __shfl_xor_sync(LANE_MASK, cs, 1);
                cs += __shfl_xor_sync(LANE_MASK, cs, 2);
                cm_vals[k] /= (cs + hc_sinkhorn_eps);
            }
            for (int it = 1; it < sinkhorn_repeat; it++)
            {
                rs = cm_vals[0] + cm_vals[1] + cm_vals[2] + cm_vals[3] + hc_sinkhorn_eps;
#pragma unroll
                for (int k = 0; k < HC_MULT; k++)
                    cm_vals[k] /= rs;
#pragma unroll
                for (int k = 0; k < HC_MULT; k++)
                {
                    float cs = cm_vals[k];
                    cs += __shfl_xor_sync(LANE_MASK, cs, 1);
                    cs += __shfl_xor_sync(LANE_MASK, cs, 2);
                    cm_vals[k] /= (cs + hc_sinkhorn_eps);
                }
            }
            float* cm_out_ptr = comb_mix_out + tok * HC_MULT2;
#pragma unroll
            for (int k = 0; k < HC_MULT; k++)
                cm_out_ptr[lane * HC_MULT + k] = cm_vals[k];
        }
        __syncthreads();

        // layer_input: warp>0 threads process hidden in parallel.
        if (warp_id > 0)
        {
            float pm[HC_MULT];
#pragma unroll
            for (int j = 0; j < HC_MULT; j++)
                pm[j] = s_pre_mix[t][j];

            __nv_bfloat16 const* rbase = residual_out + static_cast<long long>(tok) * HC_MULT * hidden_size;
            __nv_bfloat16* obase = layer_input_out + static_cast<long long>(tok) * hidden_size;
            int const p2_tid = tid - WARP_SIZE;
            constexpr int p2_threads = BLOCK_SIZE - WARP_SIZE;
            constexpr int BF16_VEC_LI = 8;

            for (int h = p2_tid * BF16_VEC_LI; h < hidden_size; h += p2_threads * BF16_VEC_LI)
            {
                float acc_li[BF16_VEC_LI] = {};
#pragma unroll
                for (int j = 0; j < HC_MULT; j++)
                {
                    uint4 raw = *reinterpret_cast<uint4 const*>(&rbase[j * hidden_size + h]);
                    __nv_bfloat162 const* pairs = reinterpret_cast<__nv_bfloat162 const*>(&raw);
#pragma unroll
                    for (int v = 0; v < BF16_VEC_LI / 2; v++)
                    {
                        float2 f = __bfloat1622float2(pairs[v]);
                        acc_li[2 * v + 0] += pm[j] * f.x;
                        acc_li[2 * v + 1] += pm[j] * f.y;
                    }
                }
                uint4 out_raw;
                __nv_bfloat162* opairs = reinterpret_cast<__nv_bfloat162*>(&out_raw);
#pragma unroll
                for (int v = 0; v < BF16_VEC_LI / 2; v++)
                    opairs[v] = __float22bfloat162_rn(make_float2(acc_li[2 * v], acc_li[2 * v + 1]));
                *reinterpret_cast<uint4*>(&obase[h]) = out_raw;
            }
        }
        __syncthreads();
    }
}

} // namespace fused_fma_kernels
