/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include "cascadeAttentionKernel.h"
#include "cascadeMma.cuh"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace mmha
{
namespace cascade
{

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Internal helpers
//
////////////////////////////////////////////////////////////////////////////////////////////////////

// Per-block configuration for the Tensor-Core Phase 1 kernel.
//
//   - THDS_PER_BLOCK = 128 = 4 warps
//   - BEAM_TILE      = 16 = MMA M dimension (1 m16n8 tile of beams)
//   - TOKEN_TILE     = 16 = MMA K dimension for P·V and N dimension for Q·K^T
//
// Phase 2 (suffix decode) still uses THDS_PER_BLOCK threads-per-block with the
// per-channel block_sum reduction.  Supported Dh values are 64 and 128.
template <int Dh_>
struct CascadeConfig
{
    static constexpr int THDS_PER_BLOCK = 128;
    static constexpr int BEAM_TILE = 16;
    static constexpr int TOKEN_TILE = 16;
    static_assert(Dh_ == 64 || Dh_ == 128, "cascade kernel only supports Dh \\in {64, 128}");
};

// Convert a value of type T to float on device.
template <typename T>
__device__ inline float to_float(T v);

template <>
__device__ inline float to_float<float>(float v)
{
    return v;
}

template <>
__device__ inline float to_float<half>(half v)
{
    return __half2float(v);
}

#ifdef ENABLE_BF16
template <>
__device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 v)
{
    return __bfloat162float(v);
}
#endif

// Convert a float back to type T on device.
template <typename T>
__device__ inline T from_float(float v);

template <>
__device__ inline float from_float<float>(float v)
{
    return v;
}

template <>
__device__ inline half from_float<half>(float v)
{
    return __float2half(v);
}

#ifdef ENABLE_BF16
template <>
__device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float v)
{
    return __float2bfloat16(v);
}
#endif

// Read one element of K[token, head, channel] from the cache.
template <typename T_cache, typename KVCacheBuffer>
__device__ inline T_cache load_k(KVCacheBuffer const& kv, int seqIdx, int tokenIdx, int headIdx, int dim, int channel)
{
    auto const localTokenIdx = kv.getKVTokenIdx(tokenIdx);
    auto* blockPtr = reinterpret_cast<T_cache*>(kv.getKBlockPtr(seqIdx, localTokenIdx));
    auto const localOffset = kv.getKVLocalIdx(localTokenIdx, headIdx, dim, channel);
    return blockPtr[localOffset];
}

// Read one element of V[token, head, channel] from the cache.
template <typename T_cache, typename KVCacheBuffer>
__device__ inline T_cache load_v(KVCacheBuffer const& kv, int seqIdx, int tokenIdx, int headIdx, int dim, int channel)
{
    auto const localTokenIdx = kv.getKVTokenIdx(tokenIdx);
    auto* blockPtr = reinterpret_cast<T_cache*>(kv.getVBlockPtr(seqIdx, localTokenIdx));
    auto const localOffset = kv.getKVLocalIdx(localTokenIdx, headIdx, dim, channel);
    return blockPtr[localOffset];
}

template <int THDS>
__device__ inline float block_sum(float v, float* scratch)
{
    // ---------- v1: warp-shuffle + 1 SMEM cross-warp reduce -----------------
    static_assert(THDS % 32 == 0, "block_sum requires THDS to be a multiple of warpSize");
    constexpr int N_WARPS = THDS / 32;
    int const tid = threadIdx.x;
    int const warp_id = tid >> 5;
    int const lane_id = tid & 31;

    // Step 1: intra-warp shuffle reduction (no __syncthreads needed).
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, offset);
    }

    // Step 2: each warp's lane 0 publishes the warp partial.
    if (lane_id == 0)
    {
        scratch[warp_id] = v;
    }
    __syncthreads();

    // Step 3: warp 0 reduces the N_WARPS partials and broadcasts the result
    //         back through scratch[0].
    if (warp_id == 0)
    {
        float block_val = (lane_id < N_WARPS) ? scratch[lane_id] : 0.f;
#pragma unroll
        for (int offset = N_WARPS / 2; offset > 0; offset >>= 1)
        {
            block_val += __shfl_xor_sync(0xFFFFFFFFu, block_val, offset);
        }
        if (lane_id == 0)
        {
            scratch[0] = block_val;
        }
    }
    __syncthreads();

    float const result = scratch[0];
    __syncthreads(); // keep scratch free for the next invocation
    return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// RoPE helpers (GPT-NeoX style, full-head rotation).
//
// Cached K in TRT-LLM is stored *post-RoPE* whenever POS_SHIFT == false
// (which we already enforce in cascade_eligible), so cascade only needs to
// apply RoPE to the live Q tensor.  The pair layout is (c, c + rotary_dim/2).
//
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ void cascade_rope_cs(
    int c, int rotary_dim, int pos, float base, float scale, float& cos_a, float& sin_a)
{
    int const half = rotary_dim / 2;
    int const freq_idx = (c < half) ? c : (c - half);
    float const inv_freq = __powf(base, -2.0f * static_cast<float>(freq_idx) / static_cast<float>(rotary_dim));
    float const angle = static_cast<float>(pos) * inv_freq * scale;
    __sincosf(angle, &sin_a, &cos_a);
}

// Rotate the Dh-channel vector `vec` (shared memory) in-place.  Each thread owns
// its own channel `c` and reads the partner channel from `vec`.  All threads in
// the block must reach every __syncthreads() below, hence the `active` flag.
__device__ __forceinline__ void cascade_rope_apply_neox(
    float* vec, int c, int rotary_dim, int pos, float base, float scale)
{
    int const half = rotary_dim / 2;
    bool const active = (c < rotary_dim);
    float cos_a = 1.f, sin_a = 0.f;
    float val = 0.f, partner = 0.f;
    if (active)
    {
        cascade_rope_cs(c, rotary_dim, pos, base, scale, cos_a, sin_a);
        val = vec[c];
        partner = (c < half) ? vec[c + half] : vec[c - half];
    }
    __syncthreads();
    if (active)
    {
        vec[c] = (c < half) ? (val * cos_a - partner * sin_a) : (val * cos_a + partner * sin_a);
    }
}

template <typename T>
__device__ constexpr int cascade_smem_row_stride(int dh)
{
    // Keep 16B padding (= 16 / sizeof(T) elements): 8 for bf16/half.
    return dh + (16 / static_cast<int>(sizeof(T)));
}

template <typename T, typename T_cache, typename KVCacheBuffer, int Dh, int TOKEN_TILE, int THDS>
__device__ __forceinline__ void cascade_async_load_k_tile(
    KVCacheBuffer const& kv, int owner_seq, int kv_head_idx, int dim, int t0, int tile_end, int tid, T* K_smem_buf)
{
    constexpr int BYTES_PER_CHUNK = 16;
    constexpr int ELEMS_PER_CHUNK = BYTES_PER_CHUNK / sizeof(T);
    constexpr int CHUNKS_PER_TOK = Dh / ELEMS_PER_CHUNK;
    constexpr int TOTAL_CHUNKS = TOKEN_TILE * CHUNKS_PER_TOK;
    constexpr int ROW_STRIDE = cascade_smem_row_stride<T>(Dh);
    static_assert(Dh % ELEMS_PER_CHUNK == 0, "Dh must be a multiple of 8 (16B / bf16)");

#pragma unroll
    for (int idx = tid; idx < TOTAL_CHUNKS; idx += THDS)
    {
        int const tok_rel = idx / CHUNKS_PER_TOK;
        int const chunk = idx - tok_rel * CHUNKS_PER_TOK;
        int const dh_base = chunk * ELEMS_PER_CHUNK;
        int const tok = t0 + tok_rel;
        T* s_ptr = K_smem_buf + tok_rel * ROW_STRIDE + dh_base;
        if (tok < tile_end)
        {
            auto const localTokenIdx = kv.getKVTokenIdx(tok);
            auto* base = reinterpret_cast<T_cache*>(kv.getKBlockPtr(owner_seq, localTokenIdx));
            auto const off = kv.getKVLocalIdx(localTokenIdx, kv_head_idx, dim, dh_base);
            mma::cp_async_16B(mma::smem_addr_of(s_ptr), base + off);
        }
        else
        {
            uint4 zero = {0u, 0u, 0u, 0u};
            *reinterpret_cast<uint4*>(s_ptr) = zero;
        }
    }
}

template <typename T, typename T_cache, typename KVCacheBuffer, int Dh, int TOKEN_TILE, int THDS>
__device__ __forceinline__ void cascade_async_load_v_tile(
    KVCacheBuffer const& kv, int owner_seq, int kv_head_idx, int dim, int t0, int tile_end, int tid, T* V_smem_buf)
{
    constexpr int BYTES_PER_CHUNK = 16;
    constexpr int ELEMS_PER_CHUNK = BYTES_PER_CHUNK / sizeof(T);
    constexpr int CHUNKS_PER_TOK = Dh / ELEMS_PER_CHUNK;
    constexpr int TOTAL_CHUNKS = TOKEN_TILE * CHUNKS_PER_TOK;
    constexpr int ROW_STRIDE = cascade_smem_row_stride<T>(Dh);

#pragma unroll
    for (int idx = tid; idx < TOTAL_CHUNKS; idx += THDS)
    {
        int const tok_rel = idx / CHUNKS_PER_TOK;
        int const chunk = idx - tok_rel * CHUNKS_PER_TOK;
        int const dh_base = chunk * ELEMS_PER_CHUNK;
        int const tok = t0 + tok_rel;
        T* s_ptr = V_smem_buf + tok_rel * ROW_STRIDE + dh_base;
        if (tok < tile_end)
        {
            auto const localTokenIdx = kv.getKVTokenIdx(tok);
            auto* base = reinterpret_cast<T_cache*>(kv.getVBlockPtr(owner_seq, localTokenIdx));
            auto const off = kv.getKVLocalIdx(localTokenIdx, kv_head_idx, dim, dh_base);
            mma::cp_async_16B(mma::smem_addr_of(s_ptr), base + off);
        }
        else
        {
            uint4 zero = {0u, 0u, 0u, 0u};
            *reinterpret_cast<uint4*>(s_ptr) = zero;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Phase 1: Shared-prefix multi-query attention (Tensor-Core accelerated).
//
// Grid:  ( num_heads, num_requests, ceil(beam_width / BEAM_TILE) )
// Block: 128 threads = 4 warps
//
// Each block processes BEAM_TILE=16 beams x TOKEN_TILE=16 prefix tokens per
// inner iteration, using m16n8k16 Tensor-Core MMAs for both Q·K^T and P·V.
// Online-softmax statistics (m, l) are maintained per beam in SMEM, and an
// FP32 output accumulator is kept in registers.
//
//   * Warp 0 owns the Q·K^T MMA + online softmax + produces P_smem (bf16/half)
//     and alpha[beam] (=exp(prev_m - new_m)) in SMEM.
//   * All four warps own slices of Dh for P·V: each warp covers DH/4 channels.
//   * At iteration end the four warps rescale their O_acc by alpha[beam] and
//     issue PV MMAs reading A=P_smem and B=V_smem.
//
// SMEM layout (bytes for Dh=128, bf16; adds 16B padding per Q/K/V row):
//   Q_smem : [16][136]  = 4352  (Dh + 8 bf16 padding)
//   K_smem : [16][136]  = 4352
//   V_smem : [16][136]  = 4352
//   P_smem : [16][16]   =  512  (not padded; only 32B per row)
//   stats  : [16][2]    =  128 (fp32)
//   alpha  : [16]       =   64 (fp32)
//   Total              ≈ 13696 B single-buffer / 21.9 KB with K/V double-buffer
//   (well under the 46 KB threshold that would trigger
//   cudaFuncAttributeMaxDynamicSharedMemorySize).
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename T_cache, typename KVCacheBuffer, int Dh>
__global__ void cascade_prefix_mqa_kernel(Multihead_attention_params<T, false> params, KVCacheBuffer kv_cache_buffer,
    int const* __restrict__ d_input_lengths, float* __restrict__ partial_out, float* __restrict__ partial_m,
    float* __restrict__ partial_l)
{
    using Cfg = CascadeConfig<Dh>;
    constexpr int THDS = Cfg::THDS_PER_BLOCK;    // 128
    constexpr int BEAM_TILE = Cfg::BEAM_TILE;    // 16
    constexpr int TOKEN_TILE = Cfg::TOKEN_TILE;  // 16
    constexpr int WARPS = THDS / 32;             // 4
    constexpr int DH_PER_WARP = Dh / WARPS;      // 32 (Dh=128) / 16 (Dh=64)
    constexpr int PV_N_BLOCKS = DH_PER_WARP / 8; // m16n8 output tiles per warp
    static_assert(BEAM_TILE == 16 && TOKEN_TILE == 16, "MMA tiles are fixed at 16");
    static_assert(Dh % 16 == 0, "Dh must be divisible by MMA k-dim 16");
    static_assert(DH_PER_WARP >= 8 && DH_PER_WARP % 8 == 0, "PV needs at least one n8 per warp");

    int const head_idx = blockIdx.x;
    int const req_idx = blockIdx.y;
    int const beam_chunk = blockIdx.z;
    int const beam_base = beam_chunk * BEAM_TILE;
    int const tid = threadIdx.x;
    int const warp_id = tid >> 5;
    int const lane_id = tid & 31;

    int const beam_width = params.beam_width;
    int const num_heads = params.num_heads;
    int const num_kv_heads = params.num_kv_heads;
    int const dim = params.hidden_size_per_head;
    float const inv_sqrt_dh = params.inv_sqrt_dh;

    int const prefix_len = __ldg(&d_input_lengths[req_idx * beam_width]);

    if (beam_base >= beam_width || dim != Dh)
    {
        return;
    }

    int const kv_head_idx = (num_kv_heads > 0) ? (head_idx * num_kv_heads / num_heads) : head_idx;
    // All beams share prompt KV via beam 0 (TRT-LLM context-phase convention).
    int const owner_seq = req_idx * beam_width;

    // ------------------------------ SMEM layout -------------------------
    // P0-1: each Q/K/V row carries 16B of padding (8 bf16 for Dh=128) to
    // break the 8-way bank conflict seen on Q·K^T LDS.  Total SMEM grows from
    // 21.2 KB -> 21.9 KB for Dh=128, still well under the 46 KB threshold.
    // Q_smem   [16][Dh+8]    T      (4352B for Dh=128 bf16)
    // K_smem[2][16][Dh+8]    T      (2 * 4352 = 8704B)
    // V_smem[2][16][Dh+8]    T      (2 * 4352 = 8704B)
    // P_smem   [16][16]      T      (512B, not padded: small row)
    // stats    [16][2]       float  (128B)
    // alpha_s  [16]          float  (64B)
    constexpr int SMEM_STRIDE = cascade_smem_row_stride<T>(Dh);
    extern __shared__ char smem_raw[];
    T* Q_smem = reinterpret_cast<T*>(smem_raw);
    T* K_smem[2] = {Q_smem + BEAM_TILE * SMEM_STRIDE, Q_smem + BEAM_TILE * SMEM_STRIDE + TOKEN_TILE * SMEM_STRIDE};
    T* V_smem[2]
        = {K_smem[1] + TOKEN_TILE * SMEM_STRIDE, K_smem[1] + TOKEN_TILE * SMEM_STRIDE + TOKEN_TILE * SMEM_STRIDE};
    T* P_smem = V_smem[1] + TOKEN_TILE * SMEM_STRIDE;
    float* stats = reinterpret_cast<float*>(P_smem + BEAM_TILE * TOKEN_TILE);
    float* alpha_s = stats + BEAM_TILE * 2;

    // ------------------------------ O accumulator ----------------------------
    // Per warp: one m16 row-tile times PV_N_BLOCKS n8 col-tiles.
    // Each MMA C fragment holds 4 f32 per lane.
    float O_acc[PV_N_BLOCKS][4];
#pragma unroll
    for (int n = 0; n < PV_N_BLOCKS; ++n)
    {
        O_acc[n][0] = 0.f;
        O_acc[n][1] = 0.f;
        O_acc[n][2] = 0.f;
        O_acc[n][3] = 0.f;
    }

    // ------------------------------ stats init -------------------------------
    if (tid < BEAM_TILE)
    {
        stats[tid * 2 + 0] = -FLT_MAX;
        stats[tid * 2 + 1] = 0.f;
    }

    // ------------------------------ Load Q to SMEM ---------------------------
    // Layout Q_smem[beam][dh] row-major.  Every thread contributes BEAM_TILE*Dh/THDS elements.
    uint32_t const q_stride = params.stride ? static_cast<uint32_t>(params.stride) : (num_heads * dim);
    {
        constexpr int N_Q_ELEMS = BEAM_TILE * Dh;
#pragma unroll
        for (int i = tid; i < N_Q_ELEMS; i += THDS)
        {
            int const b = i / Dh;
            int const c = i - b * Dh;
            int const beam_idx = beam_base + b;
            T q_val;
            if (beam_idx < beam_width)
            {
                int const q_offset = (req_idx * beam_width + beam_idx) * q_stride + head_idx * dim + c;
                q_val = params.q[q_offset];
            }
            else
            {
                q_val = from_float<T>(0.f);
            }
            Q_smem[b * SMEM_STRIDE + c] = q_val;
        }
    }
    __syncthreads();

    // ------------------------------ RoPE on Q (NeoX full-head) ---------------
    // cascade_eligible enforces rotary_embedding_dim == hidden_size_per_head for NeoX,
    // so every beam covers all Dh channels.  Each thread handles Dh/(2*THDS/BEAM_TILE)
    // pairs per beam across the entire block.
    bool const rope_on = (params.position_embedding_type == PositionEmbeddingType::kROPE_GPT_NEOX);
    if (rope_on)
    {
        int const rotary_dim = params.rotary_embedding_dim;
        int const half = rotary_dim >> 1;
        float const rope_base = params.rotary_embedding_base;
        float const rope_scale = params.rotary_embedding_scale;
        int const q_pos = (params.length_per_sample != nullptr) ? (params.length_per_sample[req_idx * beam_width] - 1)
                                                                : params.timestep;

        int const n_pairs = BEAM_TILE * half;
#pragma unroll
        for (int i = tid; i < n_pairs; i += THDS)
        {
            int const b = i / half;
            int const c = i - b * half; // c \in [0, half)
            int const beam_idx = beam_base + b;
            if (beam_idx >= beam_width)
                continue;
            float cos_a, sin_a;
            cascade_rope_cs(c, rotary_dim, q_pos, rope_base, rope_scale, cos_a, sin_a);
            float const v0 = to_float<T>(Q_smem[b * SMEM_STRIDE + c]);
            float const v1 = to_float<T>(Q_smem[b * SMEM_STRIDE + c + half]);
            Q_smem[b * SMEM_STRIDE + c] = from_float<T>(v0 * cos_a - v1 * sin_a);
            Q_smem[b * SMEM_STRIDE + c + half] = from_float<T>(v1 * cos_a + v0 * sin_a);
        }
        __syncthreads();
    }

    // =====================================================================
    // Main loop over prefix tokens in tiles of TOKEN_TILE.
    //
    // M3 pipeline: while block computes on K/V_smem[tile_idx & 1], tile t+1's
    // HBM->SMEM transfer is issued via cp.async into K/V_smem[(tile_idx+1)&1].
    // Per-iter timeline:
    //   1. (if next exists) issue async load for tile t+1, commit -> 2 groups
    //      in flight; wait<1> to block until tile t's group is done.
    //   2. __syncthreads(): ensure tile t's K/V and OOB zero-stores are visible.
    //   3. compute Q.K / softmax / P.V on K_cur/V_cur.
    //   4. __syncthreads(): tile t's reads complete before next iter overwrites
    //      that buffer as its `buf_nxt`.
    // =====================================================================
    int const n_tiles = (prefix_len + TOKEN_TILE - 1) / TOKEN_TILE;

    // ---- Prologue: kick off tile 0 into buffer 0 ----
    if (n_tiles > 0)
    {
        int const t0_0 = 0;
        int const tile_end_0 = min(TOKEN_TILE, prefix_len);
        cascade_async_load_k_tile<T, T_cache, KVCacheBuffer, Dh, TOKEN_TILE, THDS>(
            kv_cache_buffer, owner_seq, kv_head_idx, dim, t0_0, tile_end_0, tid, K_smem[0]);
        cascade_async_load_v_tile<T, T_cache, KVCacheBuffer, Dh, TOKEN_TILE, THDS>(
            kv_cache_buffer, owner_seq, kv_head_idx, dim, t0_0, tile_end_0, tid, V_smem[0]);
        mma::cp_async_commit();
    }

    for (int tile_idx = 0; tile_idx < n_tiles; ++tile_idx)
    {
        int const t0 = tile_idx * TOKEN_TILE;
        int const buf_cur = tile_idx & 1;
        int const buf_nxt = buf_cur ^ 1;

        // 1. Prefetch tile t+1 (if exists), then wait for tile t.
        if (tile_idx + 1 < n_tiles)
        {
            int const t0_n = (tile_idx + 1) * TOKEN_TILE;
            int const tile_end_n = min(t0_n + TOKEN_TILE, prefix_len);
            cascade_async_load_k_tile<T, T_cache, KVCacheBuffer, Dh, TOKEN_TILE, THDS>(
                kv_cache_buffer, owner_seq, kv_head_idx, dim, t0_n, tile_end_n, tid, K_smem[buf_nxt]);
            cascade_async_load_v_tile<T, T_cache, KVCacheBuffer, Dh, TOKEN_TILE, THDS>(
                kv_cache_buffer, owner_seq, kv_head_idx, dim, t0_n, tile_end_n, tid, V_smem[buf_nxt]);
            mma::cp_async_commit();
            mma::cp_async_wait<1>(); // keep the just-issued prefetch in flight
        }
        else
        {
            mma::cp_async_wait<0>(); // drain on the last tile
        }
        __syncthreads();

        T* const K_cur = K_smem[buf_cur];
        T* const V_cur = V_smem[buf_cur];

        // ============ Q · K^T  (warp 0) ============
        // C[M=beam, N=tok] = A[M, K=dh] * B[K=dh, N=tok] col-major.
        // B col-major source layout: B[k, n] == K_cur[n][k].
        // 2 n-blocks (N=0..7, 8..15), 8 k-slices for Dh=128 (or 4 for Dh=64).
        if (warp_id == 0)
        {
            float qk_acc[2][4];
            int const tm = lane_id >> 2;        // owned rows: tm, tm+8
            int const tk4 = (lane_id & 3) << 1; // owned col base (0,2,4,6)
#pragma unroll
            for (int n = 0; n < 2; ++n)
            {
                qk_acc[n][0] = 0.f;
                qk_acc[n][1] = 0.f;
                qk_acc[n][2] = 0.f;
                qk_acc[n][3] = 0.f;
            }
            // Fetch A fragments from Q_smem per k-iter.
#pragma unroll
            for (int kk = 0; kk < Dh; kk += 16)
            {
                unsigned a0 = mma::load_pack2(&Q_smem[tm * SMEM_STRIDE + kk + tk4]);
                unsigned a1 = mma::load_pack2(&Q_smem[(tm + 8) * SMEM_STRIDE + kk + tk4]);
                unsigned a2 = mma::load_pack2(&Q_smem[tm * SMEM_STRIDE + kk + tk4 + 8]);
                unsigned a3 = mma::load_pack2(&Q_smem[(tm + 8) * SMEM_STRIDE + kk + tk4 + 8]);

                int const tn = tm; // col-index within n8 block (same lane pattern)
#pragma unroll
                for (int n = 0; n < 2; ++n)
                {
                    int const actual_tok = n * 8 + tn;
                    // K_cur is row-major [tok][dh]; B[k, n] = K_cur[actual_tok][k + kk].
                    // b[0] covers k = tk4..tk4+1 (2 consecutive dh),  b[1] covers k+8..k+9.
                    unsigned b0 = mma::load_pack2(&K_cur[actual_tok * SMEM_STRIDE + kk + tk4]);
                    unsigned b1 = mma::load_pack2(&K_cur[actual_tok * SMEM_STRIDE + kk + tk4 + 8]);
                    mma::mma_m16n8k16<T>(
                        qk_acc[n][0], qk_acc[n][1], qk_acc[n][2], qk_acc[n][3], a0, a1, a2, a3, b0, b1);
                }
            }

            // ---------------- Online softmax (warp 0) ----------------
            // Owned cells per lane: rows (tm, tm+8), cols within n-block:
            //   c = 2*(lane%4)..2*(lane%4)+1 and c+8..c+9
            int const col0 = (lane_id & 3) << 1; // 0,2,4,6
                                                 // 1. Apply inv_sqrt_dh and mask invalid tokens to -FLT_MAX.
#pragma unroll
            for (int n = 0; n < 2; ++n)
            {
                int const col_lo = n * 8 + col0;
                bool const v0v = (t0 + col_lo) < prefix_len;
                bool const v1v = (t0 + col_lo + 1) < prefix_len;
                qk_acc[n][0] = v0v ? qk_acc[n][0] * inv_sqrt_dh : -FLT_MAX;
                qk_acc[n][1] = v1v ? qk_acc[n][1] * inv_sqrt_dh : -FLT_MAX;
                qk_acc[n][2] = v0v ? qk_acc[n][2] * inv_sqrt_dh : -FLT_MAX;
                qk_acc[n][3] = v1v ? qk_acc[n][3] * inv_sqrt_dh : -FLT_MAX;
            }
            // 2. Row-max: for each of the two rows (tm, tm+8), reduce 16 cols.
            float row_max_lo = fmaxf(fmaxf(qk_acc[0][0], qk_acc[0][1]), fmaxf(qk_acc[1][0], qk_acc[1][1]));
            float row_max_hi = fmaxf(fmaxf(qk_acc[0][2], qk_acc[0][3]), fmaxf(qk_acc[1][2], qk_acc[1][3]));
            // Reduce across the 4 lanes that share the same tm (lane_id/4).
            row_max_lo = fmaxf(row_max_lo, __shfl_xor_sync(0xFFFFFFFFu, row_max_lo, 1));
            row_max_lo = fmaxf(row_max_lo, __shfl_xor_sync(0xFFFFFFFFu, row_max_lo, 2));
            row_max_hi = fmaxf(row_max_hi, __shfl_xor_sync(0xFFFFFFFFu, row_max_hi, 1));
            row_max_hi = fmaxf(row_max_hi, __shfl_xor_sync(0xFFFFFFFFu, row_max_hi, 2));

            // 3. Combine with previous online stats.
            float const prev_m_lo = stats[tm * 2 + 0];
            float const prev_l_lo = stats[tm * 2 + 1];
            float const prev_m_hi = stats[(tm + 8) * 2 + 0];
            float const prev_l_hi = stats[(tm + 8) * 2 + 1];
            float const new_m_lo = fmaxf(prev_m_lo, row_max_lo);
            float const new_m_hi = fmaxf(prev_m_hi, row_max_hi);
            // expf(-FLT_MAX - finite) underflows cleanly to 0, so no extra guard needed.
            float const alpha_lo = (prev_m_lo <= -FLT_MAX * 0.5f) ? 0.f : __expf(prev_m_lo - new_m_lo);
            float const alpha_hi = (prev_m_hi <= -FLT_MAX * 0.5f) ? 0.f : __expf(prev_m_hi - new_m_hi);

            // 4. Compute p = exp(qk - new_m) per cell, track row-sum.
            float p[2][4];
#pragma unroll
            for (int n = 0; n < 2; ++n)
            {
                p[n][0] = (qk_acc[n][0] <= -FLT_MAX * 0.5f) ? 0.f : __expf(qk_acc[n][0] - new_m_lo);
                p[n][1] = (qk_acc[n][1] <= -FLT_MAX * 0.5f) ? 0.f : __expf(qk_acc[n][1] - new_m_lo);
                p[n][2] = (qk_acc[n][2] <= -FLT_MAX * 0.5f) ? 0.f : __expf(qk_acc[n][2] - new_m_hi);
                p[n][3] = (qk_acc[n][3] <= -FLT_MAX * 0.5f) ? 0.f : __expf(qk_acc[n][3] - new_m_hi);
            }
            float sum_lo = p[0][0] + p[0][1] + p[1][0] + p[1][1];
            float sum_hi = p[0][2] + p[0][3] + p[1][2] + p[1][3];
            sum_lo += __shfl_xor_sync(0xFFFFFFFFu, sum_lo, 1);
            sum_lo += __shfl_xor_sync(0xFFFFFFFFu, sum_lo, 2);
            sum_hi += __shfl_xor_sync(0xFFFFFFFFu, sum_hi, 1);
            sum_hi += __shfl_xor_sync(0xFFFFFFFFu, sum_hi, 2);
            float const new_l_lo = prev_l_lo * alpha_lo + sum_lo;
            float const new_l_hi = prev_l_hi * alpha_hi + sum_hi;

            // 5. Commit per-row outputs (one lane per row writes).
            if ((lane_id & 3) == 0)
            {
                stats[tm * 2 + 0] = new_m_lo;
                stats[tm * 2 + 1] = new_l_lo;
                stats[(tm + 8) * 2 + 0] = new_m_hi;
                stats[(tm + 8) * 2 + 1] = new_l_hi;
                alpha_s[tm] = alpha_lo;
                alpha_s[tm + 8] = alpha_hi;
            }
            // 6. Write P_smem [beam][tok].  All 32 lanes cover the 256 cells.
#pragma unroll
            for (int n = 0; n < 2; ++n)
            {
                int const col_lo = n * 8 + col0;
                P_smem[tm * TOKEN_TILE + col_lo] = from_float<T>(p[n][0]);
                P_smem[tm * TOKEN_TILE + col_lo + 1] = from_float<T>(p[n][1]);
                P_smem[(tm + 8) * TOKEN_TILE + col_lo] = from_float<T>(p[n][2]);
                P_smem[(tm + 8) * TOKEN_TILE + col_lo + 1] = from_float<T>(p[n][3]);
            }
        } // warp 0
        __syncthreads();

        // ============ Rescale O_acc + P · V  (all warps) ============
        int const dh_base = warp_id * DH_PER_WARP;
        {
            int const tm = lane_id >> 2;
            float const alpha_lo = alpha_s[tm];
            float const alpha_hi = alpha_s[tm + 8];
#pragma unroll
            for (int n = 0; n < PV_N_BLOCKS; ++n)
            {
                O_acc[n][0] *= alpha_lo;
                O_acc[n][1] *= alpha_lo;
                O_acc[n][2] *= alpha_hi;
                O_acc[n][3] *= alpha_hi;
            }
        }

        // Fetch A fragment from P_smem (shared by all warps).
        // P_smem shape [beam=16][tok=16] row-major => matches A row-major m16 k16.
        unsigned pa0, pa1, pa2, pa3;
        {
            int const tm = lane_id >> 2;
            int const tk4 = (lane_id & 3) << 1;
            pa0 = mma::load_pack2(&P_smem[tm * TOKEN_TILE + tk4]);
            pa1 = mma::load_pack2(&P_smem[(tm + 8) * TOKEN_TILE + tk4]);
            pa2 = mma::load_pack2(&P_smem[tm * TOKEN_TILE + tk4 + 8]);
            pa3 = mma::load_pack2(&P_smem[(tm + 8) * TOKEN_TILE + tk4 + 8]);
        }
        // PV MMA: C[M=beam, N=dh_slice] += A[M,K=tok] * B[K=tok, N=dh] col-major.
        // V_cur[tok][dh] row-major => B[k, n] = V_cur[k][dh_base + n*8 + tn].
        // These span 2 different tok rows (stride Dh) and must be manually packed.
        {
            int const tn = lane_id >> 2;        // 0..7  (N within n8 block)
            int const tk4 = (lane_id & 3) << 1; // K-row base (0,2,4,6)
#pragma unroll
            for (int n = 0; n < PV_N_BLOCKS; ++n)
            {
                int const actual_dh = dh_base + n * 8 + tn;
                T v00 = V_cur[tk4 * SMEM_STRIDE + actual_dh];
                T v01 = V_cur[(tk4 + 1) * SMEM_STRIDE + actual_dh];
                T v10 = V_cur[(tk4 + 8) * SMEM_STRIDE + actual_dh];
                T v11 = V_cur[(tk4 + 9) * SMEM_STRIDE + actual_dh];
                unsigned vb0 = mma::pack2<T>(v00, v01);
                unsigned vb1 = mma::pack2<T>(v10, v11);
                mma::mma_m16n8k16<T>(O_acc[n][0], O_acc[n][1], O_acc[n][2], O_acc[n][3], pa0, pa1, pa2, pa3, vb0, vb1);
            }
        }
        __syncthreads();
    } // end prefix tile loop

    // =====================================================================
    // Write out: partial_out[(req*beam + beam_idx) * num_heads + head][dh]
    //            partial_m/l[(req*beam + beam_idx) * num_heads + head]
    //
    // Thread ownership (per warp):
    //   tm = lane/4       -> two beams: beam_base + tm, beam_base + tm + 8
    //   col0 = 2*(lane%4) -> dh offsets within each n8 block: col0, col0+1
    //   across PV_N_BLOCKS n8 blocks within the warp's DH_PER_WARP range.
    // =====================================================================
    {
        int const tm = lane_id >> 2;
        int const col0 = (lane_id & 3) << 1;
        int const beam_lo = beam_base + tm;
        int const beam_hi = beam_base + tm + 8;
        int const dh_base = warp_id * DH_PER_WARP;

#pragma unroll
        for (int n = 0; n < PV_N_BLOCKS; ++n)
        {
            int const dh_lo = dh_base + n * 8 + col0;
            int const dh_hi = dh_lo + 1;
            if (beam_lo < beam_width)
            {
                int const row = (req_idx * beam_width + beam_lo) * num_heads + head_idx;
                partial_out[row * Dh + dh_lo] = O_acc[n][0];
                partial_out[row * Dh + dh_hi] = O_acc[n][1];
            }
            if (beam_hi < beam_width)
            {
                int const row = (req_idx * beam_width + beam_hi) * num_heads + head_idx;
                partial_out[row * Dh + dh_lo] = O_acc[n][2];
                partial_out[row * Dh + dh_hi] = O_acc[n][3];
            }
        }

        // Stats: only warp 0's (lane%4 == 0) threads (one per beam row).
        if (warp_id == 0 && (lane_id & 3) == 0)
        {
            if (beam_lo < beam_width)
            {
                int const row = (req_idx * beam_width + beam_lo) * num_heads + head_idx;
                partial_m[row] = stats[tm * 2 + 0];
                partial_l[row] = stats[tm * 2 + 1];
            }
            if (beam_hi < beam_width)
            {
                int const row = (req_idx * beam_width + beam_hi) * num_heads + head_idx;
                partial_m[row] = stats[(tm + 8) * 2 + 0];
                partial_l[row] = stats[(tm + 8) * 2 + 1];
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Phase 2: per-beam suffix decode.
//
// Grid:  ( num_heads, batch_size * beam_width )
// Block: ( Dh threads )
//
// Walks the suffix tokens [prefix_len, T) following `cache_indir` to resolve
// the source beam for each cached token.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename T_cache, typename KVCacheBuffer, int Dh>
// P1 fusion: suffix kernel now merges the prefix partial state in-register at
// the end and writes the final params.out directly.  This eliminates Phase 3
// (cascade_merge_kernel) and saves a full kernel launch + one DRAM write +
// three DRAM reads per (seq, head).  The suffix partial values (m_s, l_s,
// v_acc) never leave registers.
__global__ void cascade_suffix_decode_kernel(Multihead_attention_params<T, false> params, KVCacheBuffer kv_cache_buffer,
    int const* __restrict__ d_input_lengths, float const* __restrict__ partial_out_p,
    float const* __restrict__ partial_m_p, float const* __restrict__ partial_l_p)
{
    using Cfg = CascadeConfig<Dh>;
    constexpr int THDS = Cfg::THDS_PER_BLOCK;

    int const head_idx = blockIdx.x;
    int const seq = blockIdx.y; // = req_idx * beam_width + beam_idx
    int const tid = threadIdx.x;

    int const beam_width = params.beam_width;
    int const req_idx = seq / beam_width;
    int const beam_idx = seq % beam_width;
    int const num_heads = params.num_heads;
    int const num_kv_heads = params.num_kv_heads;
    int const dim = params.hidden_size_per_head;
    float const inv_sqrt_dh = params.inv_sqrt_dh;

    // Read prefix_len from device memory (Graph-safe: no host copy needed).
    int const prefix_len = __ldg(&d_input_lengths[req_idx * beam_width]);

    if (dim != Dh || tid >= Dh)
    {
        return;
    }

    // Standard MMHA convention: length_per_sample INCLUDES the current timestep,
    // but we only attend over PAST tokens.  Subtract 1 to get the KV cache length.
    int const tlength = (params.length_per_sample != nullptr) ? (params.length_per_sample[seq] - 1) : params.timestep;
    int const kv_head_idx = (num_kv_heads > 0) ? (head_idx * num_kv_heads / num_heads) : head_idx;

    extern __shared__ float smem[];
    float* reduce = smem;          // THDS floats
    float* q_smem = reduce + THDS; // Dh floats
    float* k_smem = q_smem + Dh;   // Dh floats (for K RoPE partner exchange)

    // Load Q for this (head, beam) into shared mem so threads can read each
    // other's channels during RoPE rotation.
    // NOTE: Must use params.stride (packed QKV per-sample stride) not num_heads*dim.
    uint32_t const q_stride = params.stride ? static_cast<uint32_t>(params.stride) : (num_heads * dim);
    int const q_offset = seq * q_stride + head_idx * dim + tid;
    if (tid < Dh)
    {
        q_smem[tid] = to_float<T>(params.q[q_offset]);
    }
    __syncthreads();

    // Apply RoPE to Q if enabled.  Cached K is already post-RoPE.
    if (params.position_embedding_type == PositionEmbeddingType::kROPE_GPT_NEOX)
    {
        int const rotary_dim = params.rotary_embedding_dim;
        float const rope_base = params.rotary_embedding_base;
        float const rope_scale = params.rotary_embedding_scale;
        int const q_pos = (params.length_per_sample != nullptr) ? (params.length_per_sample[seq] - 1) : params.timestep;
        cascade_rope_apply_neox(q_smem, tid, rotary_dim, q_pos, rope_base, rope_scale);
        __syncthreads();
    }

    float const q_val = (tid < Dh) ? q_smem[tid] : 0.f;

    // =====================================================================
    // Fused Phase 0: compute the current step's K/V (bias + RoPE) and have
    // the leader Q-head in each GQA group persist them to the KV cache so
    // the next decode step sees them.  The per-channel values are also
    // kept in registers (k_cur / v_cur) and reused for the tok==tlength
    // round of the attention loop, avoiding an HBM write+read round-trip.
    // =====================================================================
    float k_cur = 0.f;
    float v_cur = 0.f;
    {
        int const num_kv_heads_eff = (num_kv_heads > 0) ? num_kv_heads : num_heads;
        int const kv_group = num_heads / num_kv_heads_eff;
        bool const is_leader = ((head_idx % kv_group) == 0);

        // K/V share the same packed QKV per-sample stride as Q.
        uint32_t const kv_stride
            = params.stride ? static_cast<uint32_t>(params.stride) : static_cast<uint32_t>(num_kv_heads_eff * Dh);
        int const k_off = seq * kv_stride + kv_head_idx * Dh + tid;
        int const v_off = seq * kv_stride + kv_head_idx * Dh + tid;
        k_cur = to_float<T>(params.k[k_off]);
        v_cur = to_float<T>(params.v[v_off]);

        if (params.k_bias != nullptr)
        {
            k_cur += to_float<T>(params.k_bias[kv_head_idx * Dh + tid]);
        }
        if (params.v_bias != nullptr)
        {
            v_cur += to_float<T>(params.v_bias[kv_head_idx * Dh + tid]);
        }

        // Apply NeoX RoPE to K.  Matches baseline numerics: prefer the
        // framework-provided cos_sin_cache when available.
        if (params.position_embedding_type == PositionEmbeddingType::kROPE_GPT_NEOX)
        {
            k_smem[tid] = k_cur;
            __syncthreads();

            int const rotary_dim = params.rotary_embedding_dim;
            if (tid < rotary_dim)
            {
                int const half = rotary_dim / 2;
                int const freq_idx = (tid < half) ? tid : (tid - half);
                float cos_a, sin_a;
                if (params.rotary_embedding_cos_sin_cache != nullptr)
                {
                    float2 const* cache_base
                        = params.rotary_embedding_cos_sin_cache + static_cast<int64_t>(tlength) * half;
                    float2 cs = __ldg(&cache_base[freq_idx]);
                    cos_a = cs.x;
                    sin_a = cs.y;
                }
                else
                {
                    cascade_rope_cs(tid, rotary_dim, tlength, params.rotary_embedding_base,
                        params.rotary_embedding_scale, cos_a, sin_a);
                }
                float const val = k_smem[tid];
                float const partner = (tid < half) ? k_smem[tid + half] : k_smem[tid - half];
                k_cur = (tid < half) ? (val * cos_a - partner * sin_a) : (val * cos_a + partner * sin_a);
            }
            __syncthreads();
        }

        // Leader (one Q-head per GQA group) persists current K/V to cache.
        if (is_leader)
        {
            auto const localTokenIdx = kv_cache_buffer.getKVTokenIdx(tlength);
            auto* kPtr = reinterpret_cast<T_cache*>(kv_cache_buffer.getKBlockPtr(seq, localTokenIdx));
            auto* vPtr = reinterpret_cast<T_cache*>(kv_cache_buffer.getVBlockPtr(seq, localTokenIdx));
            auto const off = kv_cache_buffer.getKVLocalIdx(localTokenIdx, kv_head_idx, Dh, tid);
            kPtr[off] = from_float<T_cache>(k_cur);
            vPtr[off] = from_float<T_cache>(v_cur);
        }
        // No __threadfence needed: the next decode step is launched on the
        // same stream and stream ordering guarantees cross-kernel visibility.
    }

    float m = -FLT_MAX;
    float l = 0.f;
    float v_acc = 0.f;

    int const max_attention_window = params.max_attention_window_size;
    int const* cache_indir = params.cache_indir;

    // tlength = position of the current decode token in the cache.
    // Attention covers all suffix tokens INCLUDING the current one.  Past
    // tokens [prefix_len, tlength) are read from cache via cache_indir;
    // the current token (tok == tlength) is handled separately below using
    // k_cur / v_cur that we computed and (if leader) wrote to cache above.
    for (int tok = prefix_len; tok < tlength; ++tok)
    {
        // Resolve the physical beam this cached token came from via
        // cache_indir (beam search's indirection table).
        int src_beam = beam_idx;
        if (cache_indir != nullptr)
        {
            int const indir_offset
                = req_idx * beam_width * max_attention_window + beam_idx * max_attention_window + tok;
            src_beam = cache_indir[indir_offset];
        }
        int const src_seq = req_idx * beam_width + src_beam;

        float k_val = to_float<T_cache>(load_k<T_cache>(kv_cache_buffer, src_seq, tok, kv_head_idx, dim, tid));
        float v_val = to_float<T_cache>(load_v<T_cache>(kv_cache_buffer, src_seq, tok, kv_head_idx, dim, tid));

        float partial = q_val * k_val;
        float qk = block_sum<THDS>(partial, reduce) * inv_sqrt_dh;

        float new_m = fmaxf(m, qk);
        float scale = expf(m - new_m);
        float p = expf(qk - new_m);
        l = l * scale + p;
        v_acc = v_acc * scale + p * v_val;
        m = new_m;
    }

    // Current step (tok == tlength): use K/V from registers — cache_indir
    // has not yet been updated for this position (beam search runs AFTER
    // attention), so the current step is always self-pointing.
    {
        float partial = q_val * k_cur;
        float qk = block_sum<THDS>(partial, reduce) * inv_sqrt_dh;

        float new_m = fmaxf(m, qk);
        float scale = expf(m - new_m);
        float p = expf(qk - new_m);
        l = l * scale + p;
        v_acc = v_acc * scale + p * v_cur;
        m = new_m;
    }

    // -------------------------------------------------------------------
    // P1 fusion: merge with prefix partial (formerly cascade_merge_kernel)
    // -------------------------------------------------------------------
    // Suffix partial is in registers: m (suffix max), l (suffix denom), v_acc
    // (suffix weighted V sum).  Prefix partial comes from Phase 1's workspace.
    int const out_row = seq * num_heads + head_idx;
    float const m_s = m;
    float const l_s = l;
    float const v_s = v_acc;

    float const m_p = __ldg(&partial_m_p[out_row]);
    float const l_p = __ldg(&partial_l_p[out_row]);
    float const v_p = __ldg(&partial_out_p[out_row * Dh + tid]);

    // Identical logic to the former cascade_merge_kernel (kept below for
    // reference).  Empty-side guards preserve semantics when prefix or suffix
    // contributes zero tokens (e.g. first decode step, or prefix_len > tlength).
    bool const has_p = (l_p > 0.f) && (m_p > -FLT_MAX / 2.f);
    bool const has_s = (l_s > 0.f) && (m_s > -FLT_MAX / 2.f);

    float out_val;
    if (has_p && has_s)
    {
        float const new_m = fmaxf(m_p, m_s);
        float const ep = expf(m_p - new_m);
        float const es = expf(m_s - new_m);
        float const new_l = l_p * ep + l_s * es;
        out_val = (v_p * ep + v_s * es) / fmaxf(new_l, 1e-30f);
    }
    else if (has_p)
    {
        out_val = v_p / fmaxf(l_p, 1e-30f);
    }
    else if (has_s)
    {
        out_val = v_s / fmaxf(l_s, 1e-30f);
    }
    else
    {
        out_val = 0.f;
    }

    int const out_offset = seq * num_heads * Dh + head_idx * Dh + tid;
    reinterpret_cast<T*>(params.out)[out_offset] = from_float<T>(out_val);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Eligibility & launcher.
//
// Note: The former Phase 3 `cascade_merge_kernel` has been removed.  Its
// online-softmax merge logic is now fused into the tail of
// `cascade_suffix_decode_kernel` (see P1 fusion comment there), so the merge
// happens in registers without an extra kernel launch / DRAM round-trip.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename KernelParamsType>
bool cascade_eligible(KernelParamsType const& params)
{
    if constexpr (KernelParamsType::DO_CROSS_ATTENTION)
    {
        TLLM_LOG_WARNING("cascade_eligible REJECT: DO_CROSS_ATTENTION=true");
        return false;
    }
    if (!tensorrt_llm::common::getEnvEnableCascadeMmha())
    {
        // Only print once to avoid flooding.
        static std::atomic<bool> once{false};
        bool exp = false;
        if (once.compare_exchange_strong(exp, true))
        {
            TLLM_LOG_WARNING("cascade_eligible REJECT: TRTLLM_ENABLE_CASCADE_MMHA not set or =0");
        }
        return false;
    }
    if (params.beam_width < tensorrt_llm::common::getEnvCascadeMmhaMinBeam())
    {
        TLLM_LOG_WARNING("cascade_eligible REJECT: beam_width=%d < min_beam=%d", params.beam_width,
            tensorrt_llm::common::getEnvCascadeMmhaMinBeam());
        return false;
    }
    if (params.position_shift_enabled || params.block_sparse_attention)
    {
        TLLM_LOG_WARNING("cascade_eligible REJECT: position_shift=%d block_sparse=%d",
            (int) params.position_shift_enabled, (int) params.block_sparse_attention);
        return false;
    }
    // v0.1 supports LEARNED_ABSOLUTE and full-head GPT-NeoX RoPE (Qwen / Llama /
    // Mistral style). Other variants (GPTJ interleaved, YARN, LONG_ROPE,
    // M-RoPE, ALIBI, ...) fall back to the baseline MMHA path.
    if (params.position_embedding_type != PositionEmbeddingType::kLEARNED_ABSOLUTE
        && params.position_embedding_type != PositionEmbeddingType::kROPE_GPT_NEOX)
    {
        TLLM_LOG_WARNING(
            "cascade_eligible REJECT: position_embedding_type=%d (want 0 or 2)", (int) params.position_embedding_type);
        return false;
    }
    if (params.position_embedding_type == PositionEmbeddingType::kROPE_GPT_NEOX)
    {
        // Require full-head rotation so the pair (c, c + dim/2) is resolvable
        // in-place within shared memory.
        if (params.rotary_embedding_dim != params.hidden_size_per_head)
        {
            TLLM_LOG_WARNING("cascade_eligible REJECT: rotary_dim=%d != head_dim=%d (partial rotation)",
                params.rotary_embedding_dim, params.hidden_size_per_head);
            return false;
        }
        // Dynamic / long / yarn / m-scaling require cached cos-sin tables and
        // per-request base updates that v0.1 does not implement.
        if (params.rotary_embedding_scale_type != RotaryScalingType::kNONE)
        {
            TLLM_LOG_WARNING("cascade_eligible REJECT: rotary_scale_type=%d (want kNONE=0)",
                (int) params.rotary_embedding_scale_type);
            return false;
        }
        // When scale_type == kNONE, both cos_sin_cache and inv_freq_cache being
        // non-null simply means the framework pre-computed constant tables
        // (cos/sin values and 1/(base^(2i/d)) frequencies respectively).  Our
        // kernel computes RoPE on-the-fly using the same formula with __sincosf
        // and __powf, so these tables are safely ignored.
        if (params.mrope_position_deltas != nullptr)
        {
            TLLM_LOG_WARNING("cascade_eligible REJECT: mrope_position_deltas non-null");
            return false;
        }
    }
    if (params.attn_logit_softcapping_scale != 0.0f)
    {
        TLLM_LOG_WARNING(
            "cascade_eligible REJECT: attn_logit_softcapping_scale=%f", params.attn_logit_softcapping_scale);
        return false;
    }
    if (params.relative_attention_bias != nullptr || params.linear_bias_slopes != nullptr)
    {
        TLLM_LOG_WARNING("cascade_eligible REJECT: relative_attn_bias or linear_bias_slopes non-null");
        return false;
    }
    if (params.attention_mask != nullptr || params.attention_sinks != nullptr)
    {
        TLLM_LOG_WARNING("cascade_eligible REJECT: attention_mask or attention_sinks non-null");
        return false;
    }
    if (params.int8_kv_cache || params.fp8_kv_cache)
    {
        TLLM_LOG_WARNING(
            "cascade_eligible REJECT: int8_kv=%d fp8_kv=%d", (int) params.int8_kv_cache, (int) params.fp8_kv_cache);
        return false;
    }
    if (params.input_lengths == nullptr || params.length_per_sample == nullptr)
    {
        TLLM_LOG_WARNING("cascade_eligible REJECT: input_lengths=%p length_per_sample=%p", (void*) params.input_lengths,
            (void*) params.length_per_sample);
        return false;
    }
    int const dh = params.hidden_size_per_head;
    if (dh != 64 && dh != 128)
    {
        TLLM_LOG_WARNING("cascade_eligible REJECT: hidden_size_per_head=%d (want 64 or 128)", dh);
        return false;
    }

    return true;
}

namespace
{

// Persistent workspace that is allocated ONCE on first use (during TRT
// warmup, before CUDA Graph capture) and reused for all subsequent calls.
// This avoids any cudaMalloc/cudaFree/cudaMallocAsync during execution,
// making the launch path 100% graph-capture compatible (only kernel
// launches happen at runtime).
//
// Layout of the single allocation (post-P1-fusion: suffix-side partials are
// kept in registers and merged in-kernel, so only prefix-side buffers remain):
//   [out_p: batch*beam*heads*Dh floats]
//   [m_p:   batch*beam*heads   floats]
//   [l_p:   batch*beam*heads   floats]
struct PersistentWorkspace
{
    void* base = nullptr;
    size_t capacity = 0;

    // Ensure the workspace is at least `needed` bytes.
    // Returns true if workspace is ready, false on allocation failure.
    bool ensure(size_t needed)
    {
        if (capacity >= needed)
        {
            return true;
        }
        // First call or size growth: allocate (happens before graph capture
        // during TRT warmup phase).
        if (base)
        {
            cudaFree(base);
        }
        cudaError_t err = cudaMalloc(&base, needed);
        if (err != cudaSuccess)
        {
            base = nullptr;
            capacity = 0;
            return false;
        }
        capacity = needed;
        return true;
    }
};

// One global workspace instance. Thread-safety note: TRT-LLM's MMHA plugin
// enqueue is called from a single executor thread per device, so no locking
// is needed for single-GPU inference.
static PersistentWorkspace g_cascade_workspace;

} // namespace

template <typename T, typename T_cache, typename KVCacheBuffer, int Dh>
bool launch_cascade_attention(
    Multihead_attention_params<T, false> const& params, KVCacheBuffer const& kv_cache_buffer, cudaStream_t stream)
{
    // Hard guard: the kernel implementation supports T_cache == T only in v0.
    if constexpr (!std::is_same_v<T, T_cache>)
    {
        return false;
    }

    if (!cascade_eligible<T>(params))
    {
        return false;
    }

    // IMPORTANT: In TRT-LLM MMHA, params.batch_size = total_sequences =
    // num_requests × beam_width.  Our cascade design separates the "shared
    // prefix" (per-request) from "per-beam suffix", so we need num_requests.
    int const total_seqs = params.batch_size; // = num_requests × beam_width
    int const beam = params.beam_width;
    int const num_requests = total_seqs / beam;
    int const num_heads = params.num_heads;
    int const dh = params.hidden_size_per_head;

    // One-shot launch confirmation so operators can verify the cascade path is
    // actually engaged without paying per-call logging cost.
    {
        static std::atomic<bool> printed_once{false};
        bool expected = false;
        if (printed_once.compare_exchange_strong(expected, true))
        {
            TLLM_LOG_INFO("cascade_attention: ENGAGED (Dh=%d beam=%d num_requests=%d total_seqs=%d)", dh, beam,
                num_requests, total_seqs);
        }
    }

    // Compute workspace layout: 6 buffers packed contiguously.
    // Each buffer is indexed by [total_seqs × num_heads (× Dh for out)].
    size_t const out_elems = static_cast<size_t>(total_seqs) * num_heads * dh;
    size_t const stat_elems = static_cast<size_t>(total_seqs) * num_heads;
    size_t const out_bytes = out_elems * sizeof(float);
    size_t const stat_bytes = stat_elems * sizeof(float);
    // Post-P1-fusion: only prefix-side out/m/l are persisted; suffix-side
    // partials live in registers inside the fused suffix+merge kernel.
    size_t const total_bytes = out_bytes + 2 * stat_bytes;

    // Persistent workspace: allocated ONCE on first call (during TRT warmup,
    // before graph capture). All subsequent calls (including during graph
    // capture & replay) reuse the same pointer — zero allocation at runtime.
    if (!g_cascade_workspace.ensure(total_bytes))
    {
        TLLM_LOG_WARNING("cascade_attention: workspace allocation failed (need %zu bytes), falling back", total_bytes);
        return false;
    }

    // Carve pointers from the contiguous workspace block.
    char* ws = static_cast<char*>(g_cascade_workspace.base);
    float* ws_out_p = reinterpret_cast<float*>(ws);
    float* ws_m_p = reinterpret_cast<float*>(ws + out_bytes);
    float* ws_l_p = reinterpret_cast<float*>(ws + out_bytes + stat_bytes);

    // The prefix_len is NOT fetched to host. Instead, the device pointer
    // params.input_lengths is passed directly to the kernels, and each kernel
    // reads the shared prefix length via __ldg (a single cached global load).
    // This keeps the entire launch path free of D2H copies and stream-sync,
    // making it fully compatible with CUDA Graph capture.
    int const* d_input_lengths = params.input_lengths;

    using Cfg = CascadeConfig<Dh>;
    constexpr int THDS = Cfg::THDS_PER_BLOCK;
    constexpr int BEAM_TILE = Cfg::BEAM_TILE;

    // Host-side banner: print exactly once per process per template instance.
    // Uses fprintf(stderr) so it is visible regardless of TLLM_LOG_LEVEL.  If
    // this line does not appear when cascade is eligible, the loaded .so is
    // NOT the freshly rebuilt one.
    {
        static std::atomic<bool> banner_printed{false};
        if (!banner_printed.exchange(true))
        {
            fprintf(stderr, "[CASCADE BANNER] [CASCADE BANNER] build=%s %s Dh=%d\n", __DATE__, __TIME__, Dh);
            fflush(stderr);
        }
    }

    // Phase 1: shared-prefix attention (all beams share the same KV prefix).
    // Grid Y = num_requests (NOT total_seqs!): each Y-block handles one request's
    // shared prefix, processing BEAM_TILE beams per Z-block.
    {
        dim3 grid(num_heads, num_requests, (beam + BEAM_TILE - 1) / BEAM_TILE);
        dim3 block(THDS);
        // M3 Tensor-Core SMEM layout: Q + K[2] + V[2] + P (bf16/half), then
        // stats/alpha (fp32).  K/V double-buffered for cp.async pipelining.
        // P0-1: Q/K/V rows carry 16B of padding to break bank conflicts, so
        // account for it via cascade_smem_row_stride<T>(Dh) here as well.
        constexpr int SMEM_STRIDE = cascade_smem_row_stride<T>(Dh);
        size_t const t_bytes = (BEAM_TILE * SMEM_STRIDE                // Q_smem
                                   + 2 * Cfg::TOKEN_TILE * SMEM_STRIDE // K_smem[2]
                                   + 2 * Cfg::TOKEN_TILE * SMEM_STRIDE // V_smem[2]
                                   + BEAM_TILE * Cfg::TOKEN_TILE)      // P_smem (not padded)
            * sizeof(T);
        size_t const f_bytes = (BEAM_TILE * 2                          // stats(m, l)
                                   + BEAM_TILE)                        // alpha_s
            * sizeof(float);
        size_t const smem_bytes = t_bytes + f_bytes;
        // Set smem attribute once (idempotent, avoids repeated driver calls).
        static bool smem_attr_set = false;
        if (!smem_attr_set && smem_bytes >= 46 * 1024)
        {
            cudaFuncSetAttribute(cascade_prefix_mqa_kernel<T, T_cache, KVCacheBuffer, Dh>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
            smem_attr_set = true;
        }
        cascade_prefix_mqa_kernel<T, T_cache, KVCacheBuffer, Dh>
            <<<grid, block, smem_bytes, stream>>>(params, kv_cache_buffer, d_input_lengths, ws_out_p, ws_m_p, ws_l_p);
    }

    // Phase 2 (FUSED): Phase-0 write_kv + per-beam suffix decode + in-register
    // merge with prefix.  Grid Y = total_seqs: one block per (request, beam).
    // The leader Q-head in each GQA group writes the current step's K/V to
    // cache; the other Q-heads keep K/V only in registers for their own use.
    {
        dim3 grid(num_heads, total_seqs);
        dim3 block(THDS);
        // reduce(THDS) + q_smem(Dh) + k_smem(Dh, for K RoPE partner exchange)
        size_t const smem_bytes = (THDS + 2 * Dh) * sizeof(float);
        cascade_suffix_decode_kernel<T, T_cache, KVCacheBuffer, Dh>
            <<<grid, block, smem_bytes, stream>>>(params, kv_cache_buffer, d_input_lengths, ws_out_p, ws_m_p, ws_l_p);
    }

    // Phase 0 + Phase 3 removed: current-step K/V write is now done in the
    // suffix kernel's prologue (leader-elected), and the prefix/suffix merge
    // is fused into the suffix kernel's epilogue.  The workspace only carries
    // prefix-side buffers, and we never leave the stream for launches.

    // No synchronization or deallocation needed. The workspace persists
    // across calls, and stream ordering guarantees correctness.
    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Explicit instantiations.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#define INSTANTIATE_CASCADE(T, KVB, DH)                                                                                \
    template bool launch_cascade_attention<T, T, KVB, DH>(                                                             \
        Multihead_attention_params<T, false> const&, KVB const&, cudaStream_t);

INSTANTIATE_CASCADE(half, KVLinearBuffer, 64)
INSTANTIATE_CASCADE(half, KVLinearBuffer, 128)
INSTANTIATE_CASCADE(half, KVBlockArray, 64)
INSTANTIATE_CASCADE(half, KVBlockArray, 128)

#ifdef ENABLE_BF16
INSTANTIATE_CASCADE(__nv_bfloat16, KVLinearBuffer, 64)
INSTANTIATE_CASCADE(__nv_bfloat16, KVLinearBuffer, 128)
INSTANTIATE_CASCADE(__nv_bfloat16, KVBlockArray, 64)
INSTANTIATE_CASCADE(__nv_bfloat16, KVBlockArray, 128)
#endif

template bool cascade_eligible<half, Masked_multihead_attention_params<half>>(
    Masked_multihead_attention_params<half> const&);
#ifdef ENABLE_BF16
template bool cascade_eligible<__nv_bfloat16, Masked_multihead_attention_params<__nv_bfloat16>>(
    Masked_multihead_attention_params<__nv_bfloat16> const&);
#endif
template bool cascade_eligible<float, Masked_multihead_attention_params<float>>(
    Masked_multihead_attention_params<float> const&);

#undef INSTANTIATE_CASCADE

} // namespace cascade
} // namespace mmha
} // namespace kernels

TRTLLM_NAMESPACE_END
