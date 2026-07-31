/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/mathUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttentionUtils.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/mlaKernels.h"
#include <algorithm>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// A stateful callback functor that maintains the running sum between consecutive scans.
struct BlockPrefixCallbackOp
{
    // Running prefix
    int mRunningTotal;

    // Constructor
    __device__ BlockPrefixCallbackOp(int runningTotal)
        : mRunningTotal(runningTotal)
    {
    }

    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    __device__ int operator()(int blockAggregate)
    {
        int oldPrefix = mRunningTotal;
        mRunningTotal += blockAggregate;
        return oldPrefix;
    }
};

template <typename T>
struct VecType
{
    using Type = T;
    using GPTJEltType = T;
};

template <>
struct VecType<float>
{
    using Type = float4;
    using GPTJEltType = float2;
};

template <>
struct VecType<half>
{
    using Type = uint4;
    using GPTJEltType = uint32_t;
};

template <>
struct VecType<__nv_bfloat16>
{
    using Type = mmha::bf16_8_t;
    using GPTJEltType = __nv_bfloat162;
};

struct __align__(16) fp8_16_t
{
    __nv_fp8x4_e4m3 x;
    __nv_fp8x4_e4m3 y;
    __nv_fp8x4_e4m3 z;
    __nv_fp8x4_e4m3 w;
};

template <>
struct VecType<__nv_fp8_e4m3>
{
    using Type = fp8_16_t;
    using GPTJEltType = __nv_fp8x2_e4m3;
};

template <typename T>
struct loadPagedKVKernelTraits
{
    static constexpr int kLoraSize = 512;
    static constexpr int kRopeSize = 64;
    static constexpr int kHeadSize = kLoraSize + kRopeSize;
    using VecT = typename VecType<T>::Type;
    static constexpr int kBytesPerElem = sizeof(T);
    static constexpr int kBytesPerLoad = 16;
    static constexpr int kElemPerLoad = kBytesPerLoad / kBytesPerElem;
    static_assert((kHeadSize * kBytesPerElem) % kBytesPerLoad == 0,
        "kHeadSize * kBytesPerElem must be multiple of kBytesPerLoad (16Bytes)");
    static constexpr int kVecPerHead = (kHeadSize * kBytesPerElem) / kBytesPerLoad;
    static constexpr int kThreadPerHead = kVecPerHead; // for each head, we use kThreadPerHead threads to fetch all the
                                                       // kv cache data, each thread read kv cache only once.
    static constexpr int kTokenPerBlock
        = std::is_same_v<T, float> ? 4 : 8; // for each block, we fetch 4 tokens for fp32, 8 tokens for other types.
    static constexpr int kBlockSize = kThreadPerHead * kTokenPerBlock;
    static constexpr int kKVThreadPerHead = (kLoraSize * kBytesPerElem) / kBytesPerLoad;
};

template <typename SrcType, int NUM>
inline __device__ void quantCopy(
    __nv_fp8_e4m3* dst_global_ptr, SrcType const* src_fragment_ptr, float const scale_val = 1.f)
{
    using DstVecType = typename std::conditional<sizeof(SrcType) == 2, float2, float>::type;
    using SrcType2 =
        typename std::conditional<sizeof(SrcType) == 2, typename TypeConverter<SrcType>::Type, float2>::type;
    static constexpr int COPY_SIZE = sizeof(DstVecType);
    static constexpr int TOTAL_COPY_SIZE = NUM * sizeof(__nv_fp8_e4m3);
    static constexpr int LOOP_NUM = TOTAL_COPY_SIZE / COPY_SIZE;
    static_assert(TOTAL_COPY_SIZE % COPY_SIZE == 0);
    static constexpr int CVT_NUM = COPY_SIZE / sizeof(__nv_fp8_e4m3) / 2;
    static_assert(COPY_SIZE % (sizeof(__nv_fp8_e4m3) * 2) == 0);
    DstVecType fragment;
    int offset = 0;
#pragma unroll
    for (int i = 0; i < LOOP_NUM; ++i)
    {
#pragma unroll
        for (int j = 0; j < CVT_NUM; ++j)
        {
            float2 val2 = cuda_cast<float2>(reinterpret_cast<SrcType2 const*>(src_fragment_ptr)[j + offset]);
            val2.x *= scale_val;
            val2.y *= scale_val;
            reinterpret_cast<__nv_fp8x2_e4m3*>(&fragment)[j] = __nv_fp8x2_e4m3(val2);
        }
        reinterpret_cast<DstVecType*>(dst_global_ptr)[i] = fragment;
        offset += CVT_NUM;
    }
}

template <typename DstType, int NUM>
inline __device__ void dequantCopy(
    DstType* dst_global_ptr, __nv_fp8_e4m3 const* src_fragment_ptr, float const scale_val = 1.f)
{
    using DstVecType = typename VecType<DstType>::Type;
    using DstType2 =
        typename std::conditional<sizeof(DstType) == 2, typename TypeConverter<DstType>::Type, float2>::type;
    static constexpr int COPY_SIZE = sizeof(DstVecType);
    static constexpr int TOTAL_COPY_SIZE = NUM * sizeof(DstType);
    static constexpr int LOOP_NUM = TOTAL_COPY_SIZE / COPY_SIZE;
    static_assert(TOTAL_COPY_SIZE % COPY_SIZE == 0);
    static constexpr int CVT_NUM = COPY_SIZE / sizeof(DstType) / 2;
    static_assert(COPY_SIZE % (sizeof(DstType) * 2) == 0);
    DstVecType fragment;
    int offset = 0;
#pragma unroll
    for (int i = 0; i < LOOP_NUM; ++i)
    {
#pragma unroll
        for (int j = 0; j < CVT_NUM; ++j)
        {
            float2 val2 = cuda_cast<float2>(reinterpret_cast<__nv_fp8x2_e4m3 const*>(src_fragment_ptr)[j + offset]);
            val2.x *= scale_val;
            val2.y *= scale_val;
            reinterpret_cast<DstType2*>(&fragment)[j] = cuda_cast<DstType2>(val2);
        }
        reinterpret_cast<DstVecType*>(dst_global_ptr)[i] = fragment;
        offset += CVT_NUM;
    }
}

// `kOutputFp8Q`: when true, write the rotated Q rope segment directly to
// `quant_q_buf` as FP8 (scaled by `*quant_scale_qkv`) and skip the bf16 STG to
// `q_ptr`. Companion: `deepseek_v4_q_norm_fused_fp8` pre-fills the nope segment
// of `quant_q_buf`, so the standalone quantizeCopyInputToFp8Kernel can be
// dropped. `quant_q_buf`/`quant_scale_qkv`/bmm_scale outputs are unused when
// `kOutputFp8Q == false`.
// `kFuseKvNorm`: when true, `fuse_buf` holds the RAW kv_a_proj output instead of
// the normalized latent. The KV region below then owns a whole
// `K_DIM + ROPE_DIM` row per warp, so the RMSNorm reduction is warp-local, and
// it applies norm -> weight -> RoPE -> quant -> paged write in one pass. The Q
// region must not touch `fuse_buf` in that mode (its copy would be
// un-normalized), so the redundant per-head k load/rotate is compiled out
// entirely -- only `head_idx == 0` ever wrote it anyway.
template <typename T, int BLOCK_SIZE, int K_DIM, int ROPE_DIM, typename KVCacheBuffer, bool kOutputFp8Q = false,
    bool kFuseKvNorm = false>
__global__ void applyMLARopeAndAssignQKVKernelOptContext(T* q_ptr, T* q_pe, T* k_ptr, T const* fuse_buf,
    KVCacheBuffer kv_cache, int q_pe_ld, int q_pe_stride, float2 const* cos_sin_cache, size_t head_num, int head_size,
    int c_k, int* cu_q_seqlens, int32_t const* kv_cache_lengths, uint32_t max_input_seq_len, KvCacheDataType cache_type,
    float const* quant_scale_kv, int32_t const* helix_position_offsets, bool absorption_mode,
    __nv_fp8_e4m3* quant_q_buf = nullptr, float const* quant_scale_qkv = nullptr, float* bmm1_scale_out = nullptr,
    float* bmm2_scale_out = nullptr, float const* dequant_scale_q = nullptr, float const* dequant_scale_kv = nullptr,
    float const* quant_scale_o = nullptr, float host_bmm1_scale = 1.0f, T const* kv_norm_weight = nullptr,
    float kv_norm_eps = 1e-6f, int latent_row_stride = 0, bool q_rope_done = false)
{
    // bmm scales — single thread emits them when we skip quantizeCopyInputToFp8Kernel.
    if constexpr (kOutputFp8Q)
    {
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
        {
            float const dq_q = dequant_scale_q ? dequant_scale_q[0] : 1.f;
            float const dq_kv = dequant_scale_kv ? dequant_scale_kv[0] : 1.f;
            float const q_o = quant_scale_o ? quant_scale_o[0] : 1.f;
            if (bmm1_scale_out)
            {
                constexpr float kLog2e = 1.4426950408889634074f;
                float const bmm1 = dq_q * dq_kv * host_bmm1_scale;
                bmm1_scale_out[0] = bmm1;
                bmm1_scale_out[1] = bmm1 * kLog2e;
            }
            if (bmm2_scale_out)
            {
                bmm2_scale_out[0] = q_o * dq_kv;
            }
        }
    }

    // Constants.
    using VecT = typename VecType<T>::Type;
    using GPTJEltT = typename VecType<T>::GPTJEltType;
    constexpr auto HEAD_SIZE = ROPE_DIM;
    constexpr auto K_HEAD_SIZE = K_DIM;
    constexpr auto BYTES_PER_ELT = sizeof(T);
    constexpr auto BYTES_PER_LOAD = 16;
    constexpr auto ELTS_PER_VEC = BYTES_PER_LOAD / BYTES_PER_ELT;
    static_assert((HEAD_SIZE * BYTES_PER_ELT) % BYTES_PER_LOAD == 0, "Head size needs to be multiple of 16 bytes.");
    constexpr auto VECS_PER_HEAD = HEAD_SIZE * BYTES_PER_ELT / BYTES_PER_LOAD;
    constexpr auto K_VECS_PER_HEAD = K_HEAD_SIZE * BYTES_PER_ELT / BYTES_PER_LOAD;
    static_assert(BLOCK_SIZE % VECS_PER_HEAD == 0, "Kernel block should be able to handle entire heads.");
    constexpr auto TOKENS_PER_BLOCK = BLOCK_SIZE / VECS_PER_HEAD;
    constexpr auto K_TOKENS_PER_BLOCK = BLOCK_SIZE / K_VECS_PER_HEAD;
    constexpr auto TOTAL_VECS_PER_HEAD = VECS_PER_HEAD + K_VECS_PER_HEAD;

    // Block/Head idx.
    size_t const batch_idx = blockIdx.y;
    size_t const head_idx = blockIdx.z;

    // The nope head_size for q.
    // Use the latent_space head size in the absorption mode.
    int nope_head_size_q = absorption_mode ? c_k : head_size;

    if (head_idx < head_num)
    {
        // `deepseekV4QNormFusedKernel` already rotated the rope segment and wrote it
        // FP8 into `quant_q_buf`. The bf16 `q_pe` this region would rotate was left
        // stale by that kernel, so rotating it here would overwrite good data. The
        // bmm-scale prologue above still runs -- it is outside this branch.
        if (q_rope_done)
        {
            return;
        }
        size_t const head_dim_vec_idx = (threadIdx.x % VECS_PER_HEAD);
        size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;

        size_t const seq_len_loop_end
            = size_t((max_input_seq_len + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK;
        float quant_scale_kv_val = quant_scale_kv ? quant_scale_kv[0] : 1.f;
        float quant_scale_qkv_val = (kOutputFp8Q && quant_scale_qkv) ? quant_scale_qkv[0] : 1.f;

        // Mainloop.
        for (int local_token_idx = (threadIdx.x / VECS_PER_HEAD) + blockIdx.x * TOKENS_PER_BLOCK;
             local_token_idx < seq_len_loop_end; local_token_idx += TOKENS_PER_BLOCK * gridDim.x)
        {

            int const global_token_offset = cu_q_seqlens[batch_idx];
            int const cache_seq_len = kv_cache_lengths[batch_idx];

            // Derive cached offset and current input length
            int const current_seq_len = cu_q_seqlens[batch_idx + 1] - global_token_offset;
            int const cached_offset = cache_seq_len - current_seq_len;

            int token_idx_in_kv_cache = local_token_idx + cached_offset;
            // Check against BOTH total cache length (valid slot) AND input length (valid read)
            bool const valid_token = (token_idx_in_kv_cache < cache_seq_len) && (local_token_idx < current_seq_len);

            // Limit the token_idx to cache seq length (we need all threads in this block to be involved).
            token_idx_in_kv_cache = std::min(token_idx_in_kv_cache, cache_seq_len - 1);
            int const safe_local_token_idx = std::min(local_token_idx, current_seq_len - 1);
            int const global_token_idx = safe_local_token_idx + global_token_offset;

            auto const position_id
                = helix_position_offsets ? helix_position_offsets[global_token_idx] : token_idx_in_kv_cache;
            float2 const* rotary_coef_cache_buffer
                = cos_sin_cache + static_cast<size_t>(ROPE_DIM) * position_id + (head_dim_idx / 2);

            VecT q, k;
            auto src_q_global_offset = static_cast<size_t>(global_token_idx) * head_num * (head_size + ROPE_DIM)
                + (head_size + ROPE_DIM) * head_idx + head_size;
            // In the absorption mode, we load pe from q_pe instead of q_ptr.
            T* q_pe_input = q_ptr;
            if (absorption_mode)
            {
                q_pe_input = q_pe;
                src_q_global_offset = static_cast<size_t>(global_token_idx) * q_pe_stride + q_pe_ld * head_idx;
            }

            q = *reinterpret_cast<VecT const*>(&q_pe_input[src_q_global_offset + head_dim_idx]);

            if constexpr (kFuseKvNorm)
            {
                // `fuse_buf` is un-normalized here; the KV region owns k_pe entirely.
                // Rotating q alone is bit-identical to the two-operand helper, which
                // is just two independent rotary_embedding_transform calls.
#pragma unroll
                for (int elt_id = 0; elt_id < ELTS_PER_VEC / 2; elt_id++)
                {
                    GPTJEltT& q_ = reinterpret_cast<GPTJEltT*>(&q)[elt_id];
                    q_ = mmha::rotary_embedding_transform(q_, rotary_coef_cache_buffer[elt_id]);
                }
            }
            else
            {
                auto const src_k_global_offset = static_cast<size_t>(global_token_idx) * (c_k + ROPE_DIM) + c_k;
                k = *reinterpret_cast<VecT const*>(&fuse_buf[src_k_global_offset + head_dim_idx]);

                // Pack two elements into one for gptj rotary embedding.
#pragma unroll
                for (int elt_id = 0; elt_id < ELTS_PER_VEC / 2; elt_id++)
                {
                    GPTJEltT& q_ = reinterpret_cast<GPTJEltT*>(&q)[elt_id];
                    GPTJEltT& k_ = reinterpret_cast<GPTJEltT*>(&k)[elt_id];

                    float2 rotary_coef_cache = rotary_coef_cache_buffer[elt_id];
                    mmha::apply_rotary_embedding_gptj(q_, k_, rotary_coef_cache);
                }
            }
            // do sync
            __syncwarp();
            if (valid_token)
            {
                if constexpr (!kFuseKvNorm)
                {
                    if (head_idx == 0)
                    {
                        auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
                        auto inBlockIdx = kv_cache.getKVLocalIdx(
                            token_idx_in_kv_cache, 0, TOTAL_VECS_PER_HEAD, K_VECS_PER_HEAD + head_dim_vec_idx);
                        if (cache_type == KvCacheDataType::FP8)
                        {

                            quantCopy<T, ELTS_PER_VEC>(
                                reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * ELTS_PER_VEC,
                                reinterpret_cast<T const*>(&k), quant_scale_kv_val);
                        }
                        else
                            reinterpret_cast<VecT*>(kDst)[inBlockIdx] = k;
                    }
                }
                auto const dst_q_idx = static_cast<size_t>(global_token_idx) * head_num * (nope_head_size_q + ROPE_DIM)
                    + head_idx * (nope_head_size_q + ROPE_DIM) + nope_head_size_q + head_dim_idx;
                auto const dst_k_idx = static_cast<size_t>(global_token_idx) * head_num * (head_size + ROPE_DIM)
                    + head_idx * (head_size + ROPE_DIM) + head_size + head_dim_idx;
                if constexpr (kOutputFp8Q)
                {
                    quantCopy<T, ELTS_PER_VEC>(
                        quant_q_buf + dst_q_idx, reinterpret_cast<T const*>(&q), quant_scale_qkv_val);
                }
                else
                {
                    reinterpret_cast<VecT*>(q_ptr)[dst_q_idx / ELTS_PER_VEC] = q;
                }
                // Only write to k_pe to k_buf in the non-absorption mode.
                // kFuseKvNorm implies absorption mode, so `k` is never loaded there.
                if constexpr (!kFuseKvNorm)
                {
                    if (!absorption_mode)
                    {
                        reinterpret_cast<VecT*>(k_ptr)[dst_k_idx / ELTS_PER_VEC] = k;
                    }
                }
            }
        }
    }
    else if constexpr (kFuseKvNorm)
    {
        // Fused kv_a_layernorm + RoPE + quant + paged write.
        //
        // One WARP owns one whole `K_DIM + ROPE_DIM` latent row, so the
        // sum-of-squares is a plain warp shuffle -- the same shape that makes
        // `deepseekV4QNormFusedKernel` work on the Q side. The un-fused path below
        // instead splits a row across this region (dims 0..K_DIM) and the Q region
        // (dims K_DIM..K_DIM+ROPE_DIM), which is what makes the reduction
        // impossible there.
        constexpr int kWarpSize = 32;
        constexpr int kRowElts = K_HEAD_SIZE + HEAD_SIZE;
        constexpr int kRowVecs = TOTAL_VECS_PER_HEAD;
        constexpr int kVecsPerLane = kRowVecs / kWarpSize;
        constexpr int kTokensPerBlock = BLOCK_SIZE / kWarpSize;
        static_assert(kRowVecs % kWarpSize == 0,
            "Fused kv-norm needs the latent row to split evenly across a warp's 16B vectors.");
        static_assert(kVecsPerLane >= 1, "Latent row is too narrow for one warp.");

        int const lane = threadIdx.x % kWarpSize;
        int const warp_id = threadIdx.x / kWarpSize;
        int const block_dim = gridDim.z - head_num;
        int const block_id = head_idx - head_num;
        float const quant_scale_kv_val = quant_scale_kv ? quant_scale_kv[0] : 1.f;

        // Mainloop. Every lane of a warp shares `local_token_idx`, so the
        // early-continue below keeps the shuffles convergent.
        for (int local_token_idx = warp_id + gridDim.x * kTokensPerBlock * block_id + blockIdx.x * kTokensPerBlock;
             local_token_idx < static_cast<int>(max_input_seq_len);
             local_token_idx += block_dim * kTokensPerBlock * gridDim.x)
        {
            int const global_token_offset = cu_q_seqlens[batch_idx];
            int const cache_seq_len = kv_cache_lengths[batch_idx];
            int const current_seq_len = cu_q_seqlens[batch_idx + 1] - global_token_offset;
            int const cached_offset = cache_seq_len - current_seq_len;
            int const token_idx_in_kv_cache = local_token_idx + cached_offset;

            if (token_idx_in_kv_cache >= cache_seq_len || local_token_idx >= current_seq_len)
            {
                continue;
            }
            int const global_token_idx = local_token_idx + global_token_offset;
            auto const position_id
                = helix_position_offsets ? helix_position_offsets[global_token_idx] : token_idx_in_kv_cache;

            // `fuse_buf` is the caller's raw kv_a_proj slice, whose row stride is
            // NOT kRowElts -- it is a last-dim view of a wider [q_lora | kv] buffer.
            // Only the innermost dim is unit-stride, which is what the 16B vector
            // loads below require.
            size_t const row_stride = latent_row_stride > 0 ? static_cast<size_t>(latent_row_stride) : kRowElts;
            T const* row = fuse_buf + static_cast<size_t>(global_token_idx) * row_stride;

            // Pass 1: load the whole row into registers and reduce.
            VecT vals[kVecsPerLane];
            float sum_squares = 0.f;
#pragma unroll
            for (int i = 0; i < kVecsPerLane; ++i)
            {
                int const vec_idx = i * kWarpSize + lane;
                vals[i] = *reinterpret_cast<VecT const*>(row + vec_idx * ELTS_PER_VEC);
                auto const* elts = reinterpret_cast<T const*>(&vals[i]);
#pragma unroll
                for (int j = 0; j < ELTS_PER_VEC; ++j)
                {
                    float const v = static_cast<float>(elts[j]);
                    sum_squares += v * v;
                }
            }
#pragma unroll
            for (int mask = kWarpSize / 2; mask > 0; mask >>= 1)
            {
                sum_squares += __shfl_xor_sync(0xFFFFFFFFu, sum_squares, mask);
            }
            float const norm_scale = rsqrtf(sum_squares / static_cast<float>(kRowElts) + kv_norm_eps);

            // Pass 2: scale by weight, rotate the rope tail, quantize, scatter.
            // Values never leave registers between the two passes.
            auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
#pragma unroll
            for (int i = 0; i < kVecsPerLane; ++i)
            {
                int const vec_idx = i * kWarpSize + lane;
                int const dim_idx = vec_idx * ELTS_PER_VEC;

                auto* elts = reinterpret_cast<T*>(&vals[i]);
#pragma unroll
                for (int j = 0; j < ELTS_PER_VEC; ++j)
                {
                    elts[j] = static_cast<T>(
                        static_cast<float>(elts[j]) * norm_scale * static_cast<float>(kv_norm_weight[dim_idx + j]));
                }

                // The rope tail occupies dims [K_HEAD_SIZE, K_HEAD_SIZE + ROPE_DIM).
                // ELTS_PER_VEC divides both segments, so a vector never straddles them.
                if (vec_idx >= K_VECS_PER_HEAD)
                {
                    float2 const* rope_coef
                        = cos_sin_cache + static_cast<size_t>(ROPE_DIM) * position_id + ((dim_idx - K_HEAD_SIZE) / 2);
#pragma unroll
                    for (int elt_id = 0; elt_id < ELTS_PER_VEC / 2; ++elt_id)
                    {
                        GPTJEltT& d = reinterpret_cast<GPTJEltT*>(&vals[i])[elt_id];
                        d = mmha::rotary_embedding_transform(d, rope_coef[elt_id]);
                    }
                }

                auto const inBlockIdx = kv_cache.getKVLocalIdx(token_idx_in_kv_cache, 0, TOTAL_VECS_PER_HEAD, vec_idx);
                if (cache_type == KvCacheDataType::FP8)
                {
                    quantCopy<T, ELTS_PER_VEC>(reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * ELTS_PER_VEC,
                        reinterpret_cast<T const*>(&vals[i]), quant_scale_kv_val);
                }
                else
                {
                    reinterpret_cast<VecT*>(kDst)[inBlockIdx] = vals[i];
                }
            }
        }
    }
    else
    {
        int block_dim = gridDim.z - head_num;
        int block_id = head_idx - head_num;
        size_t const head_dim_vec_idx = (threadIdx.x % K_VECS_PER_HEAD);
        size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;

        size_t const seq_len_loop_end
            = size_t((max_input_seq_len + K_TOKENS_PER_BLOCK - 1) / K_TOKENS_PER_BLOCK) * K_TOKENS_PER_BLOCK;
        float quant_scale_kv_val = quant_scale_kv ? quant_scale_kv[0] : 1.f;

        // Mainloop.
        for (int local_token_idx = (threadIdx.x / K_VECS_PER_HEAD) + gridDim.x * K_TOKENS_PER_BLOCK * block_id
                 + blockIdx.x * K_TOKENS_PER_BLOCK;
             local_token_idx < seq_len_loop_end; local_token_idx += block_dim * K_TOKENS_PER_BLOCK * gridDim.x)
        {

            int const global_token_offset = cu_q_seqlens[batch_idx];
            int const cache_seq_len = kv_cache_lengths[batch_idx];

            // Derive cached offset and current input length (same as first loop)
            int const current_seq_len = cu_q_seqlens[batch_idx + 1] - global_token_offset;
            int const cached_offset = cache_seq_len - current_seq_len;

            int token_idx_in_kv_cache = local_token_idx + cached_offset;
            // Check against BOTH total cache length (valid slot) AND input length (valid read)
            bool const valid_token = (token_idx_in_kv_cache < cache_seq_len) && (local_token_idx < current_seq_len);

            // Limit the token_idx to cache seq length (we need all threads in this block to be involved).
            token_idx_in_kv_cache = std::min(token_idx_in_kv_cache, cache_seq_len - 1);
            int const safe_local_token_idx = std::min(local_token_idx, current_seq_len - 1);
            int const global_token_idx = safe_local_token_idx + global_token_offset;

            if (valid_token)
            {
                auto const src_k_global_offset = static_cast<size_t>(global_token_idx) * (c_k + ROPE_DIM);

                auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
                auto inBlockIdx
                    = kv_cache.getKVLocalIdx(token_idx_in_kv_cache, 0, TOTAL_VECS_PER_HEAD, head_dim_vec_idx);
                if (cache_type == KvCacheDataType::FP8)
                {

                    quantCopy<T, ELTS_PER_VEC>(reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * ELTS_PER_VEC,
                        fuse_buf + src_k_global_offset + head_dim_idx, quant_scale_kv_val);
                }
                else
                    reinterpret_cast<VecT*>(kDst)[inBlockIdx]
                        = *reinterpret_cast<VecT const*>(&fuse_buf[src_k_global_offset + head_dim_idx]);
            }
        }
    }
}

// `kSkipKv`: the two KV regions (`blockIdx.y == head_num` for the rope tail and
// `head_num+1 .. head_num+8` for the nope segment) are compiled out because
// `mlaKvNormRopeQuantGenerationKernel` did that work -- fused with kv_a_layernorm --
// in its own launch. The `seqQOffset` stamp normally emitted by the nope region then
// moves to block (0,0); everything else (Q rope, q_nope FP8 quant, the scheduler
// prologue) is unchanged. The grid keeps its shape so the q_nope region's
// `block_id` arithmetic stays as-is; the 9 skipped blocks exit immediately.
template <typename T, int BLOCK_SIZE, int K_DIM, int ROPE_DIM, typename KVCacheBuffer, bool kSkipKv = false>
__global__ void applyMLARopeAndAssignQKVKernelGeneration(T* qkv_output, T* q_pe, T const* fuse_buf, void* quant_q,
    KVCacheBuffer kv_cache, float2 const* cos_sin_cache, size_t head_num, int c_k, int total_s_len, int seq_len,
    int* seqQOffset, uint32_t* fmha_tile_counter, int32_t const* kv_cache_lengths, int* seqKVOffsets, int q_pe_ld,
    int q_pe_stride, KvCacheDataType cache_type, float* bmm1_scale, float* bmm2_scale, float const* quant_scale_o,
    float const* quant_scale_q, float const* quant_scale_kv, float const* dequant_scale_q,
    float const* dequant_scale_kv, float host_bmm1_scale, int32_t const* helix_position_offsets,
    bool const* helix_is_inactive_rank, bool precomputed_cu_seqlens = false, bool precomputed_fmha_scheduler = false)
{
    // Constants.
    using VecT = typename VecType<T>::Type;
    using GPTJEltT = typename VecType<T>::GPTJEltType;
    constexpr auto HEAD_SIZE = ROPE_DIM;
    constexpr auto K_HEAD_SIZE = K_DIM;
    constexpr auto BYTES_PER_ELT = sizeof(T);
    constexpr auto BYTES_PER_LOAD = 16;
    constexpr auto ELTS_PER_VEC = BYTES_PER_LOAD / BYTES_PER_ELT;
    static_assert((HEAD_SIZE * BYTES_PER_ELT) % BYTES_PER_LOAD == 0, "Head size needs to be multiple of 16 bytes.");
    constexpr auto VECS_PER_HEAD = HEAD_SIZE * BYTES_PER_ELT / BYTES_PER_LOAD;
    constexpr auto K_VECS_PER_HEAD = K_HEAD_SIZE * BYTES_PER_ELT / BYTES_PER_LOAD;
    static_assert(BLOCK_SIZE % VECS_PER_HEAD == 0, "Kernel block should be able to handle entire heads.");
    constexpr auto TOKENS_PER_BLOCK = BLOCK_SIZE / VECS_PER_HEAD;
    constexpr auto K_TOKENS_PER_BLOCK = BLOCK_SIZE / K_VECS_PER_HEAD;
    constexpr auto TOTAL_VEC_PER_HEAD = VECS_PER_HEAD + K_VECS_PER_HEAD;

    // Block/Head idx.
    size_t const head_idx = blockIdx.y;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    if constexpr (kSkipKv)
    {
        // The nope region used to stamp `seqQOffset[b+1]`; it no longer runs, so do it
        // here. Pure arithmetic on the batch index -- no KV data involved. Skipped
        // entirely when the metadata already filled the array once for the iteration.
        if (!precomputed_cu_seqlens && blockIdx.x == 0 && blockIdx.y == 0)
        {
            int const batch_size_bound = total_s_len / seq_len;
            for (int b = threadIdx.x; b < batch_size_bound; b += BLOCK_SIZE)
            {
                seqQOffset[b + 1] = static_cast<int>(head_num) * seq_len * (b + 1);
            }
        }
    }

    // Under `kSkipKv` the whole FMHA scheduler prologue -- the tile counter and the
    // bmm scales -- has moved to `mlaKvNormRopeQuantGenerationKernel`, which runs
    // first in the pair. Only the `seqQOffset[0]` seed can still be owed here, and
    // only when the metadata did not precompute the array.
    // Whatever is left of the scheduler prologue. Both halves can be owned
    // elsewhere: `cu_seqlens` by the attention metadata (once per iteration) and the
    // tile counter + bmm scales by the DSv4 sparse indices kernel (last launch before
    // FMHA). When both are, block (0,0) has nothing to do and this kernel is pure Q.
    bool const owes_cu_seqlens = !precomputed_cu_seqlens;
    bool const owes_fmha_scheduler = !precomputed_fmha_scheduler && !kSkipKv;
    if ((owes_cu_seqlens || owes_fmha_scheduler) && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
    {
        if (owes_fmha_scheduler)
        {
            fmha_tile_counter[0] = 0;
        }
        if (owes_cu_seqlens)
        {
            seqQOffset[0] = 0;
        }

        // Calculate bmm scale for FP8 MLA
        if (owes_fmha_scheduler && cache_type == KvCacheDataType::FP8)
        {
            float dequant_scale_q_val = dequant_scale_q ? dequant_scale_q[0] : 1.f;
            float dequant_scale_kv_val = dequant_scale_kv ? dequant_scale_kv[0] : 1.f;
            float quant_scale_o_val = quant_scale_o ? quant_scale_o[0] : 1.f;
            if (bmm1_scale)
            {
                // The scale prepared for log2 optimization.
                constexpr float kLog2e = 1.4426950408889634074f;
                // The scale after fmha bmm1.
                float bmm1_scale_val = dequant_scale_q_val * dequant_scale_kv_val * host_bmm1_scale;
                bmm1_scale[0] = bmm1_scale_val;
                bmm1_scale[1] = bmm1_scale_val * kLog2e;
            }
            if (bmm2_scale)
            {
                // The scale after fmha bmm2.
                bmm2_scale[0] = quant_scale_o_val * dequant_scale_kv_val;
            }
        }
    }

    if (head_idx <= head_num)
    {
        if constexpr (kSkipKv)
        {
            // `head_idx == head_num` is the k_pe rope + cache-write block: fused away.
            // Trigger the programmatic-launch completion first -- the tail of the
            // kernel does it for the blocks that run to the end.
            if (head_idx == head_num)
            {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
                cudaTriggerProgrammaticLaunchCompletion();
#endif
                return;
            }
        }
        size_t const head_dim_vec_idx = (threadIdx.x % VECS_PER_HEAD);
        size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;

        int const seq_len_loop_end = size_t((total_s_len + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK;
        float const quant_scale_q_val = quant_scale_q ? quant_scale_q[0] : 1.0f;
        float const quant_scale_kv_val = quant_scale_kv ? quant_scale_kv[0] : 1.0f;

        // Mainloop.
        for (int global_token_idx = (threadIdx.x / VECS_PER_HEAD) + blockIdx.x * TOKENS_PER_BLOCK;
             global_token_idx < seq_len_loop_end; global_token_idx += TOKENS_PER_BLOCK * gridDim.x)
        {
            auto batch_idx = global_token_idx / seq_len;
            auto local_token_idx = global_token_idx % seq_len;
            bool const valid_token = global_token_idx < total_s_len;
            VecT data;

            if (valid_token)
            {

                auto const position_id
                    = (helix_position_offsets != nullptr ? helix_position_offsets[global_token_idx]
                                                         : kv_cache_lengths[batch_idx] - seq_len + local_token_idx);
                float2 const* rotary_coef_cache_buffer
                    = cos_sin_cache + static_cast<size_t>(ROPE_DIM) * position_id + (head_dim_idx / 2);

                if (head_idx == head_num)
                {
                    auto const src_k_global_offset = static_cast<size_t>(global_token_idx) * (c_k + ROPE_DIM) + c_k;

                    data = *reinterpret_cast<VecT const*>(&fuse_buf[src_k_global_offset + head_dim_idx]);
                }
                else
                {
                    auto const src_q_global_offset
                        = static_cast<size_t>(global_token_idx) * q_pe_stride + q_pe_ld * head_idx;

                    data = *reinterpret_cast<VecT const*>(&q_pe[src_q_global_offset + head_dim_idx]);
                }

                // Pack two elements into one for gptj rotary embedding.
#pragma unroll
                for (int elt_id = 0; elt_id < ELTS_PER_VEC / 2; elt_id++)
                {
                    GPTJEltT& data_ = reinterpret_cast<GPTJEltT*>(&data)[elt_id];

                    float2 rotary_coef_cache = rotary_coef_cache_buffer[elt_id];
                    data_ = mmha::rotary_embedding_transform(data_, rotary_coef_cache);
                }
            }

            __syncwarp();

            if (valid_token)
            {
                if (head_idx == head_num)
                {
                    // If helix parallelism is being used, only write to KV cache if current rank is active.
                    if (helix_is_inactive_rank == nullptr || !helix_is_inactive_rank[batch_idx])
                    {
                        auto const token_kv_idx = kv_cache_lengths[batch_idx] - seq_len + local_token_idx;

                        {
                            auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_kv_idx));
                            auto inBlockIdx = kv_cache.getKVLocalIdx(
                                token_kv_idx, 0, TOTAL_VEC_PER_HEAD, K_VECS_PER_HEAD + head_dim_vec_idx);
                            if (cache_type == KvCacheDataType::FP8)
                            {

                                quantCopy<T, ELTS_PER_VEC>(
                                    reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * ELTS_PER_VEC,
                                    reinterpret_cast<T const*>(&data), quant_scale_kv_val);
                            }
                            else
                                reinterpret_cast<VecT*>(kDst)[inBlockIdx] = data;
                        }
                    }
                }
                else
                {
                    auto const dst_q_idx = static_cast<size_t>(global_token_idx) * head_num * (c_k + ROPE_DIM)
                        + head_idx * (c_k + ROPE_DIM) + c_k + head_dim_idx;
                    if (cache_type == KvCacheDataType::FP8)
                    {
                        quantCopy<T, ELTS_PER_VEC>(reinterpret_cast<__nv_fp8_e4m3*>(quant_q) + dst_q_idx,
                            reinterpret_cast<T const*>(&data), quant_scale_q_val);
                    }
                    else
                        reinterpret_cast<VecT*>(qkv_output)[dst_q_idx / ELTS_PER_VEC] = data;
                }
            }
        }
    }
    else if (head_idx <= head_num + 8)
    {
        // compressed_kv copy region: fused away together with kv_a_layernorm.
        if constexpr (kSkipKv)
        {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
            cudaTriggerProgrammaticLaunchCompletion();
#endif
            return;
        }
        int block_dim = gridDim.y - head_num - 1;
        int block_id = head_idx - head_num - 1;
        size_t const head_dim_vec_idx = (threadIdx.x % K_VECS_PER_HEAD);
        size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;

        size_t const seq_len_loop_end
            = size_t((total_s_len + K_TOKENS_PER_BLOCK - 1) / K_TOKENS_PER_BLOCK) * K_TOKENS_PER_BLOCK;
        float quant_scale_kv_val = quant_scale_kv ? quant_scale_kv[0] : 1.0f;

        // Mainloop.
        for (int global_token_idx = (threadIdx.x / K_VECS_PER_HEAD) + gridDim.x * K_TOKENS_PER_BLOCK * block_id
                 + blockIdx.x * K_TOKENS_PER_BLOCK;
             global_token_idx < seq_len_loop_end; global_token_idx += block_dim * K_TOKENS_PER_BLOCK * gridDim.x)
        {
            auto batch_idx = global_token_idx / seq_len;
            auto local_token_idx = global_token_idx % seq_len;
            bool valid_token = global_token_idx < total_s_len;

            if (valid_token)
            {
                if (head_dim_vec_idx == 0)
                {
                    seqQOffset[batch_idx + 1] = head_num * seq_len * (batch_idx + 1);
                }

                // If helix parallelism is being used, only write to KV cache if current rank is active.
                if (helix_is_inactive_rank == nullptr || !helix_is_inactive_rank[batch_idx])
                {
                    auto const token_kv_idx = kv_cache_lengths[batch_idx] - seq_len + local_token_idx;
                    auto const src_kv_global_offset = static_cast<size_t>(global_token_idx) * (c_k + ROPE_DIM);

                    {
                        auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_kv_idx));
                        auto inBlockIdx = kv_cache.getKVLocalIdx(token_kv_idx, 0, TOTAL_VEC_PER_HEAD, head_dim_vec_idx);

                        if (cache_type == KvCacheDataType::FP8)
                        {
                            quantCopy<T, ELTS_PER_VEC>(
                                reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * ELTS_PER_VEC,
                                fuse_buf + src_kv_global_offset + head_dim_idx, quant_scale_kv_val);
                        }
                        else
                            reinterpret_cast<VecT*>(kDst)[inBlockIdx]
                                = *reinterpret_cast<VecT const*>(&fuse_buf[src_kv_global_offset + head_dim_idx]);
                    }
                }
            }
        }
    }
    else
    {
        if (cache_type == KvCacheDataType::FP8)
        {
            int block_dim = gridDim.y - head_num - 1 - 8;
            int block_id = head_idx - head_num - 1 - 8;
            size_t const head_dim_vec_idx = (threadIdx.x % K_VECS_PER_HEAD);
            size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;
            size_t const head_num_idx = (block_id % head_num) * (K_HEAD_SIZE + HEAD_SIZE);

            size_t const seq_len_loop_end
                = size_t((total_s_len + K_TOKENS_PER_BLOCK - 1) / K_TOKENS_PER_BLOCK) * K_TOKENS_PER_BLOCK;
            float quant_scale_q_val = quant_scale_q ? quant_scale_q[0] : 1.0f;

            // Mainloop.
            for (int global_token_idx = (threadIdx.x / K_VECS_PER_HEAD)
                     + (block_id / head_num) * gridDim.x * K_TOKENS_PER_BLOCK + blockIdx.x * K_TOKENS_PER_BLOCK;
                 global_token_idx < seq_len_loop_end;
                 global_token_idx += (block_dim / head_num) * gridDim.x * K_TOKENS_PER_BLOCK)
            {
                if (global_token_idx < total_s_len)
                {
                    size_t const load_idx
                        = global_token_idx * head_num * (K_HEAD_SIZE + HEAD_SIZE) + head_num_idx + head_dim_idx;
                    quantCopy<T, ELTS_PER_VEC>(
                        reinterpret_cast<__nv_fp8_e4m3*>(quant_q) + load_idx, qkv_output + load_idx, quant_scale_q_val);
                }
            }
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif

    // The implementation of the parallel scan in the thread block (see CUB for details).
    using BlockScan = cub::BlockScan<int, BLOCK_SIZE>;

    // Allocate storage in shared memory to do the scan.
    __shared__ typename BlockScan::TempStorage tempKVStorage;
    BlockPrefixCallbackOp prefixKVOp(0);

    if (blockIdx.x == 0 && blockIdx.y == 0 && !precomputed_cu_seqlens)
    {
        int const batchSizeBound = total_s_len / seq_len;
        for (int batchOffset = 0; batchOffset <= batchSizeBound; batchOffset += BLOCK_SIZE)
        {
            // The index of the batch.
            int batchIdx = batchOffset + threadIdx.x;
            int seqKVLength = 0;
            if (batchIdx < batchSizeBound)
            {
                seqKVLength = kv_cache_lengths[batchIdx];
            }
            int seqKVOffset;
            BlockScan(tempKVStorage).ExclusiveSum(seqKVLength, seqKVOffset, prefixKVOp);
            if (batchIdx <= batchSizeBound)
            {
                seqKVOffsets[batchIdx] = seqKVOffset;
            }
        }
    }
}

// Generation-phase KV prologue: kv_a_layernorm + RoPE + FP8 quant + paged write, fused.
//
// This is the standalone counterpart of the `kFuseKvNorm` KV region in
// `applyMLARopeAndAssignQKVKernelOptContext`, and it replaces the two KV regions of
// `applyMLARopeAndAssignQKVKernelGeneration` (`blockIdx.y == head_num` for the rope
// tail, `head_num+1 .. head_num+8` for the nope segment). Those regions split one
// latent row across two disjoint block groups, so no block could form the RMS
// denominator; here one WARP owns a whole `K_DIM + ROPE_DIM` row, making the
// sum-of-squares a single warp shuffle.
//
// `fuse_buf` is the RAW `kv_a_proj_with_mqa` slice, so its row stride is
// `q_lora_rank + K_DIM + ROPE_DIM`, not `K_DIM + ROPE_DIM` -- it is a last-dim view,
// which is why the stride is a parameter. Only the innermost dim is unit-stride,
// which is what the 16B vector loads require.
//
// DSv4-only by construction: no helix parameters (DSv4 + CP Helix raises in
// mla.py) and no Q-side arguments.
//
// This kernel also owns what is left of the FMHA scheduler prologue -- zeroing
// `fmha_tile_counter` and deriving the bmm1/bmm2 scales. Both are single-thread
// writes that used to sit in block (0,0) of the generation RoPE kernel. They
// belong to whichever kernel runs first in the pair, and this one does; moving
// them here empties that kernel's prologue so it is pure Q work. Correctness is
// per-LAUNCH, not per-iteration: the counter has to be zero before every FMHA
// launch, and this kernel is launched exactly once per RoPE kernel launch.
template <typename T, int BLOCK_SIZE, int K_DIM, int ROPE_DIM, typename KVCacheBuffer>
__global__ void mlaKvNormRopeQuantGenerationKernel(T const* fuse_buf, KVCacheBuffer kv_cache,
    float2 const* cos_sin_cache, T const* kv_norm_weight, float kv_norm_eps, int latent_row_stride, int total_s_len,
    int seq_len, int32_t const* kv_cache_lengths, KvCacheDataType cache_type, float const* quant_scale_kv,
    uint32_t* fmha_tile_counter, float* bmm1_scale, float* bmm2_scale, float const* quant_scale_o,
    float const* dequant_scale_q, float const* dequant_scale_kv, float host_bmm1_scale)
{
    using VecT = typename VecType<T>::Type;
    using GPTJEltT = typename VecType<T>::GPTJEltType;

    constexpr auto BYTES_PER_ELT = sizeof(T);
    constexpr auto BYTES_PER_LOAD = 16;
    constexpr auto ELTS_PER_VEC = BYTES_PER_LOAD / BYTES_PER_ELT;
    constexpr int kWarpSize = 32;
    constexpr int kRowElts = K_DIM + ROPE_DIM;
    constexpr int kRowVecs = kRowElts * BYTES_PER_ELT / BYTES_PER_LOAD;
    constexpr int kRopeVecStart = K_DIM * BYTES_PER_ELT / BYTES_PER_LOAD;
    constexpr int kVecsPerLane = kRowVecs / kWarpSize;
    constexpr int kRowsPerBlock = BLOCK_SIZE / kWarpSize;

    static_assert(
        kRowVecs % kWarpSize == 0, "Fused kv-norm needs the latent row to split evenly across a warp's 16B vectors.");
    static_assert(kVecsPerLane >= 1, "Latent row is too narrow for one warp.");
    static_assert(
        (ROPE_DIM * BYTES_PER_ELT) % BYTES_PER_LOAD == 0, "A 16B vector must not straddle the nope/rope boundary.");

    int const lane = threadIdx.x % kWarpSize;
    int const warp_id = threadIdx.x / kWarpSize;
    float const quant_scale_kv_val = quant_scale_kv ? quant_scale_kv[0] : 1.f;

    // FMHA scheduler prologue, rehomed from the RoPE kernel. It reads only static
    // scale tensors, so it deliberately runs BEFORE the grid-dependency wait below:
    // useful work while the producing GEMM drains. The launcher passes null pointers
    // when the DSv4 sparse indices kernel already owns this.
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        if (fmha_tile_counter != nullptr)
        {
            fmha_tile_counter[0] = 0;
        }
        if (cache_type == KvCacheDataType::FP8)
        {
            float const dequant_scale_q_val = dequant_scale_q ? dequant_scale_q[0] : 1.f;
            float const dequant_scale_kv_val = dequant_scale_kv ? dequant_scale_kv[0] : 1.f;
            float const quant_scale_o_val = quant_scale_o ? quant_scale_o[0] : 1.f;
            if (bmm1_scale)
            {
                // The scale prepared for log2 optimization.
                constexpr float kLog2e = 1.4426950408889634074f;
                float const bmm1_scale_val = dequant_scale_q_val * dequant_scale_kv_val * host_bmm1_scale;
                bmm1_scale[0] = bmm1_scale_val;
                bmm1_scale[1] = bmm1_scale_val * kLog2e;
            }
            if (bmm2_scale)
            {
                bmm2_scale[0] = quant_scale_o_val * dequant_scale_kv_val;
            }
        }
    }

    // The launch sets `programmaticStreamSerializationAllowed`, so this kernel can
    // start before its predecessor -- the kv_a_proj GEMM that produces `fuse_buf` --
    // has finished. Wait for it before touching the latent.
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    // One warp per row. Every lane of a warp shares `global_token_idx`, so the
    // early-continue below keeps the shuffles convergent.
    for (int global_token_idx = warp_id + blockIdx.x * kRowsPerBlock; global_token_idx < total_s_len;
         global_token_idx += kRowsPerBlock * gridDim.x)
    {
        int const batch_idx = global_token_idx / seq_len;
        int const local_token_idx = global_token_idx % seq_len;
        int const token_kv_idx = kv_cache_lengths[batch_idx] - seq_len + local_token_idx;
        if (token_kv_idx < 0)
        {
            continue;
        }
        int const position_id = token_kv_idx;

        T const* row = fuse_buf + static_cast<size_t>(global_token_idx) * static_cast<size_t>(latent_row_stride);

        // Pass 1: load the whole row into registers and reduce.
        VecT vals[kVecsPerLane];
        float sum_squares = 0.f;
#pragma unroll
        for (int i = 0; i < kVecsPerLane; ++i)
        {
            int const vec_idx = i * kWarpSize + lane;
            vals[i] = *reinterpret_cast<VecT const*>(row + vec_idx * ELTS_PER_VEC);
            auto const* elts = reinterpret_cast<T const*>(&vals[i]);
#pragma unroll
            for (int j = 0; j < ELTS_PER_VEC; ++j)
            {
                float const v = static_cast<float>(elts[j]);
                sum_squares += v * v;
            }
        }
#pragma unroll
        for (int mask = kWarpSize / 2; mask > 0; mask >>= 1)
        {
            sum_squares += __shfl_xor_sync(0xFFFFFFFFu, sum_squares, mask);
        }
        float const norm_scale = rsqrtf(sum_squares / static_cast<float>(kRowElts) + kv_norm_eps);

        // Pass 2: scale by weight, rotate the rope tail, quantize, scatter.
        // Values never leave registers between the two passes.
        auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_kv_idx));
#pragma unroll
        for (int i = 0; i < kVecsPerLane; ++i)
        {
            int const vec_idx = i * kWarpSize + lane;
            int const dim_idx = vec_idx * ELTS_PER_VEC;

            auto* elts = reinterpret_cast<T*>(&vals[i]);
#pragma unroll
            for (int j = 0; j < ELTS_PER_VEC; ++j)
            {
                elts[j] = static_cast<T>(
                    static_cast<float>(elts[j]) * norm_scale * static_cast<float>(kv_norm_weight[dim_idx + j]));
            }

            if (vec_idx >= kRopeVecStart)
            {
                float2 const* rope_coef
                    = cos_sin_cache + static_cast<size_t>(ROPE_DIM) * position_id + ((dim_idx - K_DIM) / 2);
#pragma unroll
                for (int elt_id = 0; elt_id < ELTS_PER_VEC / 2; ++elt_id)
                {
                    GPTJEltT& d = reinterpret_cast<GPTJEltT*>(&vals[i])[elt_id];
                    d = mmha::rotary_embedding_transform(d, rope_coef[elt_id]);
                }
            }

            auto const inBlockIdx = kv_cache.getKVLocalIdx(token_kv_idx, 0, kRowVecs, vec_idx);
            if (cache_type == KvCacheDataType::FP8)
            {
                quantCopy<T, ELTS_PER_VEC>(reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * ELTS_PER_VEC,
                    reinterpret_cast<T const*>(&vals[i]), quant_scale_kv_val);
            }
            else
            {
                reinterpret_cast<VecT*>(kDst)[inBlockIdx] = vals[i];
            }
        }
    }

    // Release the RoPE kernel that follows: it only needs the Q side, so it can
    // overlap the tail of this one.
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename T, typename TCache>
__global__ void loadPagedKVCacheForMLAKernel(T* compressed_kv_ptr, T* k_pe_ptr,
    tensorrt_llm::kernels::KVBlockArray const kv_cache, int64_t const* cu_ctx_cached_kv_lens, int max_input_seq_len,
    float const* kv_scale_quant_orig_ptr)
{
    static_assert(std::is_same_v<T, TCache> || std::is_same_v<TCache, __nv_fp8_e4m3>,
        "TCache must be either the same type as T or __nv_fp8_e4m3");
    using KT = typename tensorrt_llm::kernels::loadPagedKVKernelTraits<TCache>;

    int const batch_idx = static_cast<int>(blockIdx.y);
    float const kv_scale_quant_orig = kv_scale_quant_orig_ptr ? kv_scale_quant_orig_ptr[0] : 1.0f;

    size_t const head_dim_vec_idx = (threadIdx.x % KT::kVecPerHead);
    size_t const head_dim_idx = head_dim_vec_idx * KT::kElemPerLoad;
    bool const is_valid_kv = head_dim_vec_idx < KT::kKVThreadPerHead;

    size_t const seq_len_loop_end
        = (max_input_seq_len + KT::kTokenPerBlock - 1) / KT::kTokenPerBlock * KT::kTokenPerBlock;

    int64_t const global_token_offset = cu_ctx_cached_kv_lens[batch_idx];
    int64_t const cache_kv_len = cu_ctx_cached_kv_lens[batch_idx + 1] - cu_ctx_cached_kv_lens[batch_idx];

    for (int local_token_idx = (threadIdx.x / KT::kThreadPerHead) + blockIdx.x * KT::kTokenPerBlock;
         local_token_idx < seq_len_loop_end; local_token_idx += KT::kTokenPerBlock * gridDim.x)
    {
        int token_idx_in_kv_cache = local_token_idx;
        bool const valid_token = token_idx_in_kv_cache < cache_kv_len;

        if (valid_token)
        {
            auto* kvSrc = reinterpret_cast<TCache*>(kv_cache.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
            // head_idx === 0
            auto kvBlockIdx
                = kv_cache.getKVLocalIdx(token_idx_in_kv_cache, 0, KT::kVecPerHead, static_cast<int>(head_dim_vec_idx));

            auto src_data = reinterpret_cast<typename KT::VecT*>(kvSrc)[kvBlockIdx];

            int const global_token_idx = local_token_idx + global_token_offset;

            if (is_valid_kv)
            {
                // compressed_kv {total_token, lora_size}
                int const dstIdx = global_token_idx * KT::kLoraSize + head_dim_idx;

                // copy back to compressed_kv
                if constexpr (std::is_same_v<TCache, T>)
                {
                    *reinterpret_cast<typename KT::VecT*>(compressed_kv_ptr + dstIdx) = src_data;
                }
                else if constexpr (std::is_same_v<TCache, __nv_fp8_e4m3>)
                {
                    dequantCopy<T, KT::kElemPerLoad>(compressed_kv_ptr + dstIdx,
                        reinterpret_cast<__nv_fp8_e4m3 const*>(&src_data), kv_scale_quant_orig);
                }
            }
            else
            {
                // k_pe {total_token, rope_size}
                int const dstIdx = global_token_idx * KT::kRopeSize + (head_dim_idx - KT::kLoraSize);

                // copy back to k_pe
                if constexpr (std::is_same_v<TCache, T>)
                {
                    *reinterpret_cast<typename KT::VecT*>(k_pe_ptr + dstIdx) = src_data;
                }
                else if constexpr (std::is_same_v<TCache, __nv_fp8_e4m3>)
                {
                    dequantCopy<T, KT::kElemPerLoad>(
                        k_pe_ptr + dstIdx, reinterpret_cast<__nv_fp8_e4m3 const*>(&src_data), kv_scale_quant_orig);
                }
            }
        }
    }
}

// q {total_uncached_tokens, h, d_nope + d_rope}
// latent_cache {total_uncached_tokens, d_k + d_rope}
template <typename T, typename TCache, int BLOCK_SIZE, int K_DIM, int ROPE_DIM>
__global__ void applyMLARopeAppendPagedKVAssignQKernel(KVBlockArray kv_cache, T* q_ptr, T* latent_cache_ptr,
    int64_t const* cu_ctx_cached_kv_lens, int64_t const* cu_seq_lens, int const max_input_uncached_seq_len,
    float2 const* cos_sin_cache, size_t head_num, int nope_size, float const* kv_scale_orig_quant_ptr)
{
    static_assert(std::is_same_v<T, TCache> || std::is_same_v<TCache, __nv_fp8_e4m3>,
        "TCache must be either the same type as T or __nv_fp8_e4m3");
    // Constants.
    using VecT = typename VecType<T>::Type;
    using GPTJEltT = typename VecType<T>::GPTJEltType;
    constexpr auto HEAD_SIZE = ROPE_DIM;
    constexpr auto K_HEAD_SIZE = K_DIM;
    constexpr auto BYTES_PER_ELT = sizeof(T);
    constexpr auto BYTES_PER_LOAD = 16;
    constexpr auto ELTS_PER_VEC = BYTES_PER_LOAD / BYTES_PER_ELT;
    static_assert((HEAD_SIZE * BYTES_PER_ELT) % BYTES_PER_LOAD == 0, "Head size needs to be multiple of 16 bytes.");
    constexpr auto VECS_PER_HEAD = HEAD_SIZE * BYTES_PER_ELT / BYTES_PER_LOAD;
    constexpr auto K_VECS_PER_HEAD = K_HEAD_SIZE * BYTES_PER_ELT / BYTES_PER_LOAD;
    static_assert(BLOCK_SIZE % VECS_PER_HEAD == 0, "Kernel block should be able to handle entire heads.");
    constexpr auto TOKENS_PER_BLOCK = BLOCK_SIZE / VECS_PER_HEAD;
    constexpr auto K_TOKENS_PER_BLOCK = BLOCK_SIZE / K_VECS_PER_HEAD;
    constexpr auto TOTAL_VECS_PER_HEAD = VECS_PER_HEAD + K_VECS_PER_HEAD;

    // Block/Head idx.
    size_t const batch_idx = blockIdx.y;
    size_t const head_idx = blockIdx.z;

    int64_t const global_token_offset = cu_seq_lens[batch_idx] - cu_ctx_cached_kv_lens[batch_idx];
    int64_t const cached_kv_len = cu_ctx_cached_kv_lens[batch_idx + 1] - cu_ctx_cached_kv_lens[batch_idx];
    int64_t const uncached_kv_len = cu_seq_lens[batch_idx + 1] - cu_seq_lens[batch_idx] - cached_kv_len;

    if (head_idx <= head_num)
    {
        size_t const head_dim_vec_idx = (threadIdx.x % VECS_PER_HEAD);
        size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;

        size_t const seq_len_loop_end
            = size_t((max_input_uncached_seq_len + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK;
        float quant_scale_kv_val = kv_scale_orig_quant_ptr ? kv_scale_orig_quant_ptr[0] : 1.f;

        // Mainloop.
        for (int local_token_idx = (threadIdx.x / VECS_PER_HEAD) + blockIdx.x * TOKENS_PER_BLOCK;
             local_token_idx < seq_len_loop_end; local_token_idx += TOKENS_PER_BLOCK * gridDim.x)
        {

            int token_idx_in_kv_cache = local_token_idx + cached_kv_len;
            bool valid_token = local_token_idx < uncached_kv_len;
            int const global_token_idx = local_token_idx + global_token_offset;
            VecT data;

            if (valid_token)
            {
                auto const position_id = token_idx_in_kv_cache;
                float2 const* rotary_coef_cache_buffer
                    = cos_sin_cache + static_cast<size_t>(ROPE_DIM) * position_id + (head_dim_idx / 2);

                if (head_idx == head_num)
                {
                    auto const src_k_global_offset = static_cast<size_t>(global_token_idx) * (K_DIM + ROPE_DIM) + K_DIM;
                    data = *reinterpret_cast<VecT const*>(&latent_cache_ptr[src_k_global_offset + head_dim_idx]);
                }
                else
                {
                    auto const src_q_global_offset
                        = static_cast<size_t>(global_token_idx) * head_num * (nope_size + ROPE_DIM)
                        + (nope_size + ROPE_DIM) * head_idx + nope_size;
                    data = *reinterpret_cast<VecT const*>(&q_ptr[src_q_global_offset + head_dim_idx]);
                }

                // Pack two elements into one for gptj rotary embedding.
#pragma unroll
                for (int elt_id = 0; elt_id < ELTS_PER_VEC / 2; elt_id++)
                {
                    GPTJEltT& data_ = reinterpret_cast<GPTJEltT*>(&data)[elt_id];

                    float2 rotary_coef_cache = rotary_coef_cache_buffer[elt_id];
                    data_ = mmha::rotary_embedding_transform(data_, rotary_coef_cache);
                }
            }
            // do sync
            __syncwarp();
            if (valid_token)
            {
                if (head_idx == head_num)
                {
                    auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
                    auto inBlockIdx = kv_cache.getKVLocalIdx(
                        token_idx_in_kv_cache, 0, TOTAL_VECS_PER_HEAD, K_VECS_PER_HEAD + head_dim_vec_idx);
                    if constexpr (std::is_same_v<TCache, T>)
                    {
                        reinterpret_cast<VecT*>(kDst)[inBlockIdx] = data;
                    }
                    else if constexpr (std::is_same_v<TCache, __nv_fp8_e4m3>)
                    {
                        quantCopy<T, ELTS_PER_VEC>(reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * ELTS_PER_VEC,
                            reinterpret_cast<T const*>(&data), quant_scale_kv_val);
                    }
                    // copy to latent_cache (for chunked prefill, it will not load kv cache for uncached k_pe)
                    // we only need to copy original value.
                    auto const src_k_global_offset = static_cast<size_t>(global_token_idx) * (K_DIM + ROPE_DIM) + K_DIM;
                    *reinterpret_cast<VecT*>(&latent_cache_ptr[src_k_global_offset + head_dim_idx]) = data;
                }
                else
                {
                    auto const dst_q_idx = static_cast<size_t>(global_token_idx) * head_num * (nope_size + ROPE_DIM)
                        + head_idx * (nope_size + ROPE_DIM) + nope_size + head_dim_idx;
                    reinterpret_cast<VecT*>(q_ptr)[dst_q_idx / ELTS_PER_VEC] = data;
                }
            }
        }
    }
    else
    {
        int block_dim = gridDim.z - head_num - 1;
        int block_id = head_idx - head_num - 1;
        size_t const head_dim_vec_idx = (threadIdx.x % K_VECS_PER_HEAD);
        size_t const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;

        size_t const seq_len_loop_end
            = size_t((max_input_uncached_seq_len + K_TOKENS_PER_BLOCK - 1) / K_TOKENS_PER_BLOCK) * K_TOKENS_PER_BLOCK;
        float quant_scale_kv_val = kv_scale_orig_quant_ptr ? kv_scale_orig_quant_ptr[0] : 1.f;

        // Mainloop.
        for (int local_token_idx = (threadIdx.x / K_VECS_PER_HEAD) + gridDim.x * K_TOKENS_PER_BLOCK * block_id
                 + blockIdx.x * K_TOKENS_PER_BLOCK;
             local_token_idx < seq_len_loop_end; local_token_idx += block_dim * K_TOKENS_PER_BLOCK * gridDim.x)
        {

            int token_idx_in_kv_cache = local_token_idx + cached_kv_len;
            bool valid_token = local_token_idx < uncached_kv_len;
            int const global_token_idx = local_token_idx + global_token_offset;

            if (valid_token)
            {
                auto const src_k_global_offset = static_cast<size_t>(global_token_idx) * (K_DIM + ROPE_DIM);

                auto kDst = reinterpret_cast<T*>(kv_cache.getKBlockPtr(batch_idx, token_idx_in_kv_cache));
                auto inBlockIdx
                    = kv_cache.getKVLocalIdx(token_idx_in_kv_cache, 0, TOTAL_VECS_PER_HEAD, head_dim_vec_idx);
                if constexpr (std::is_same_v<TCache, T>)
                {
                    reinterpret_cast<VecT*>(kDst)[inBlockIdx]
                        = *reinterpret_cast<VecT const*>(&latent_cache_ptr[src_k_global_offset + head_dim_idx]);
                }
                else if constexpr (std::is_same_v<TCache, __nv_fp8_e4m3>)
                {
                    quantCopy<T, ELTS_PER_VEC>(reinterpret_cast<__nv_fp8_e4m3*>(kDst) + inBlockIdx * ELTS_PER_VEC,
                        latent_cache_ptr + src_k_global_offset + head_dim_idx, quant_scale_kv_val);
                }
            }
        }
    }
}

template <typename T, int BLOCK_SIZE, int QK_NOPE_HEAD_DIM, int QK_ROPE_HEAD_DIM, int V_HEAD_DIM, bool ABSORPTION_MODE>
__global__ void quantizeCopyInputToFp8Kernel(T const* q_buf, __nv_fp8_e4m3* quant_q_buf, T const* k_buf,
    __nv_fp8_e4m3* quant_k_buf, T const* v_buf, __nv_fp8_e4m3* quant_v_buf, int total_q_len, int total_kv_len,
    float const* quant_scale_qkv_ptr, float* bmm1_scale, float* bmm2_scale, float const* quant_scale_o,
    float const* dequant_scale_q, float const* dequant_scale_kv, float host_bmm1_scale)
{
    // Constants.
    using VecT = typename VecType<T>::Type;
    constexpr auto BYTES_PER_ELT = sizeof(T);
    constexpr auto BYTES_PER_LOAD = 16;
    constexpr auto ELTS_PER_VEC = BYTES_PER_LOAD / BYTES_PER_ELT;
    constexpr auto QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM;
    static_assert(
        (QK_HEAD_DIM * BYTES_PER_ELT) % BYTES_PER_LOAD == 0, "QK head size needs to be multiple of 16 bytes.");
    static_assert((V_HEAD_DIM * BYTES_PER_ELT) % BYTES_PER_LOAD == 0, "V head size needs to be multiple of 16 bytes.");
    constexpr auto QK_VECS_PER_HEAD = QK_HEAD_DIM * BYTES_PER_ELT / BYTES_PER_LOAD;
    constexpr auto V_VECS_PER_HEAD = V_HEAD_DIM * BYTES_PER_ELT / BYTES_PER_LOAD;
    static_assert(BLOCK_SIZE % QK_VECS_PER_HEAD == 0, "Kernel block should be able to handle entire heads.");
    static_assert(ABSORPTION_MODE || (BLOCK_SIZE % V_VECS_PER_HEAD) == 0,
        "Kernel block should be able to handle entire heads in non-absorption mode.");
    constexpr auto QK_TOKENS_PER_BLOCK = BLOCK_SIZE / QK_VECS_PER_HEAD;
    constexpr auto V_TOKENS_PER_BLOCK = BLOCK_SIZE / V_VECS_PER_HEAD;

    size_t const head_idx = blockIdx.z;
    size_t const head_num = gridDim.z;

    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
    {
        // Calculate bmm scale for FP8 MLA
        float dequant_scale_q_val = dequant_scale_q ? dequant_scale_q[0] : 1.f;
        float dequant_scale_kv_val = dequant_scale_kv ? dequant_scale_kv[0] : 1.f;
        float quant_scale_o_val = quant_scale_o ? quant_scale_o[0] : 1.f;
        if (bmm1_scale)
        {
            // The scale prepared for log2 optimization.
            constexpr float kLog2e = 1.4426950408889634074f;
            // The scale after fmha bmm1.
            float bmm1_scale_val = dequant_scale_q_val * dequant_scale_kv_val * host_bmm1_scale;
            bmm1_scale[0] = bmm1_scale_val;
            bmm1_scale[1] = bmm1_scale_val * kLog2e;
        }
        if (bmm2_scale)
        {
            // The scale after fmha bmm2.
            bmm2_scale[0] = quant_scale_o_val * dequant_scale_kv_val;
        }
    }

    size_t const qk_head_dim_vec_idx = (threadIdx.x % QK_VECS_PER_HEAD);
    size_t const v_head_dim_vec_idx = (threadIdx.x % V_VECS_PER_HEAD);
    size_t const qk_head_dim_idx = qk_head_dim_vec_idx * ELTS_PER_VEC;
    size_t const v_head_dim_idx = v_head_dim_vec_idx * ELTS_PER_VEC;

    size_t const q_len_loop_end
        = size_t((total_q_len + QK_TOKENS_PER_BLOCK - 1) / QK_TOKENS_PER_BLOCK) * QK_TOKENS_PER_BLOCK;
    size_t const k_len_loop_end
        = size_t((total_kv_len + QK_TOKENS_PER_BLOCK - 1) / QK_TOKENS_PER_BLOCK) * QK_TOKENS_PER_BLOCK;
    size_t const v_len_loop_end
        = size_t((total_kv_len + V_TOKENS_PER_BLOCK - 1) / V_TOKENS_PER_BLOCK) * V_TOKENS_PER_BLOCK;
    float quant_scale_qkv_val = quant_scale_qkv_ptr ? quant_scale_qkv_ptr[0] : 1.f;

    // Quantize Q, both src and dst are contiguous
    for (int q_token_idx = (threadIdx.x / QK_VECS_PER_HEAD) + blockIdx.x * QK_TOKENS_PER_BLOCK;
         q_token_idx < q_len_loop_end; q_token_idx += QK_TOKENS_PER_BLOCK * gridDim.x)
    {
        if (q_token_idx < total_q_len)
        {
            auto const src_q_idx
                = static_cast<size_t>(q_token_idx) * QK_HEAD_DIM * head_num + head_idx * QK_HEAD_DIM + qk_head_dim_idx;
            auto const dst_q_idx = src_q_idx;
            quantCopy<T, ELTS_PER_VEC>(quant_q_buf + dst_q_idx, &q_buf[src_q_idx], quant_scale_qkv_val);
        }
    }

    // Only quantize K and V in non-absorption mode.
    if constexpr (!ABSORPTION_MODE)
    {
        // Quantize K, both src and dst are contiguous
        for (int k_token_idx = (threadIdx.x / QK_VECS_PER_HEAD) + blockIdx.x * QK_TOKENS_PER_BLOCK;
             k_token_idx < k_len_loop_end; k_token_idx += QK_TOKENS_PER_BLOCK * gridDim.x)
        {
            if (k_token_idx < total_kv_len)
            {
                auto const src_k_idx = static_cast<size_t>(k_token_idx) * QK_HEAD_DIM * head_num
                    + head_idx * QK_HEAD_DIM + qk_head_dim_idx;
                auto const dst_k_idx = src_k_idx;
                quantCopy<T, ELTS_PER_VEC>(quant_k_buf + dst_k_idx, &k_buf[src_k_idx], quant_scale_qkv_val);
            }
        }
        // Quantize V, dst V is contiguous, but src V is not contiguous, so we need to calculate the stride
        size_t const src_v_token_stride = (QK_NOPE_HEAD_DIM + V_HEAD_DIM) * head_num;
        for (int v_token_idx = (threadIdx.x / V_VECS_PER_HEAD) + blockIdx.x * V_TOKENS_PER_BLOCK;
             v_token_idx < v_len_loop_end; v_token_idx += V_TOKENS_PER_BLOCK * gridDim.x)
        {
            if (v_token_idx < total_kv_len)
            {
                auto const src_v_idx
                    = static_cast<size_t>(v_token_idx) * src_v_token_stride + head_idx * V_HEAD_DIM + v_head_dim_idx;
                auto const dst_v_idx
                    = static_cast<size_t>(v_token_idx) * V_HEAD_DIM * head_num + head_idx * V_HEAD_DIM + v_head_dim_idx;
                quantCopy<T, ELTS_PER_VEC>(quant_v_buf + dst_v_idx, &v_buf[src_v_idx], quant_scale_qkv_val);
            }
        }
    }
}

template <typename T, typename KVCacheBuffer>
void invokeMLARopeContext(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream)
{
    dim3 grid(int(tensorrt_llm::common::divUp(params.max_input_seq_len, 32)), params.batch_size, params.head_num + 8);
    auto head_size = params.meta.qk_nope_head_dim;
    // Fused FP8-Q path: write the rotated Q rope segment directly to quant_q_buf
    // as FP8 so the caller can drop the standalone quantizeCopyInputToFp8Kernel.
    bool const useFusedFp8Q = params.fuse_q_fp8_in_rope && params.absorption_mode
        && params.cache_type == KvCacheDataType::FP8 && params.quant_q_buf != nullptr
        && params.quant_scale_qkv != nullptr;

    auto* quant_q_fp8 = useFusedFp8Q ? static_cast<__nv_fp8_e4m3*>(params.quant_q_buf) : nullptr;
    if (params.meta.rope_append)
    {
        if (useFusedFp8Q)
        {
            applyMLARopeAndAssignQKVKernelOptContext<T, 256, 512, 64, KVCacheBuffer, true>
                <<<grid, 256, 0, stream>>>(params.q_buf, params.q_pe, params.k_buf, params.latent_cache,
                    kv_cache_buffer, params.q_pe_ld, params.q_pe_stride, params.cos_sin_cache, params.head_num,
                    head_size, params.meta.kv_lora_rank, params.cu_q_seqlens, params.cache_seq_lens,
                    params.max_input_seq_len, params.cache_type, params.quant_scale_kv, params.helix_position_offsets,
                    params.absorption_mode, quant_q_fp8, params.quant_scale_qkv, params.bmm1_scale, params.bmm2_scale,
                    params.dequant_scale_q, params.dequant_scale_kv, params.quant_scale_o, params.host_bmm1_scale);
        }
        else
        {
            applyMLARopeAndAssignQKVKernelOptContext<T, 256, 512, 64, KVCacheBuffer, false><<<grid, 256, 0, stream>>>(
                params.q_buf, params.q_pe, params.k_buf, params.latent_cache, kv_cache_buffer, params.q_pe_ld,
                params.q_pe_stride, params.cos_sin_cache, params.head_num, head_size, params.meta.kv_lora_rank,
                params.cu_q_seqlens, params.cache_seq_lens, params.max_input_seq_len, params.cache_type,
                params.quant_scale_kv, params.helix_position_offsets, params.absorption_mode);
        }
    }
    else
    {
        // DSv4 layout (rope_append == false). The kv_a_layernorm fusion is only
        // wired on this instantiation: the kernel describes the latent row with its
        // K_DIM/ROPE_DIM template constants, so kv_lora_rank must actually match.
        bool const useFusedKvNorm = params.fuse_kv_norm_in_rope && params.absorption_mode
            && params.kv_norm_weight != nullptr && params.meta.kv_lora_rank == 448;
        TLLM_CHECK_WITH_INFO(params.fuse_kv_norm_in_rope == useFusedKvNorm,
            "MLA Context: fused kv-norm requested but preconditions not met "
            "(absorption_mode=%d, kv_norm_weight=%p, kv_lora_rank=%d, expected 448)",
            static_cast<int>(params.absorption_mode), params.kv_norm_weight, params.meta.kv_lora_rank);
        auto const* kv_norm_w = static_cast<T const*>(params.kv_norm_weight);

        if (useFusedFp8Q && useFusedKvNorm)
        {
            applyMLARopeAndAssignQKVKernelOptContext<T, 256, 448, 64, KVCacheBuffer, true, true>
                <<<grid, 256, 0, stream>>>(params.q_buf, params.q_pe, params.k_buf, params.latent_cache,
                    kv_cache_buffer, params.q_pe_ld, params.q_pe_stride, params.cos_sin_cache, params.head_num,
                    head_size, params.meta.kv_lora_rank, params.cu_q_seqlens, params.cache_seq_lens,
                    params.max_input_seq_len, params.cache_type, params.quant_scale_kv, params.helix_position_offsets,
                    params.absorption_mode, quant_q_fp8, params.quant_scale_qkv, params.bmm1_scale, params.bmm2_scale,
                    params.dequant_scale_q, params.dequant_scale_kv, params.quant_scale_o, params.host_bmm1_scale,
                    kv_norm_w, params.kv_norm_eps, params.latent_row_stride, params.q_rope_done);
        }
        else if (useFusedKvNorm)
        {
            applyMLARopeAndAssignQKVKernelOptContext<T, 256, 448, 64, KVCacheBuffer, false, true>
                <<<grid, 256, 0, stream>>>(params.q_buf, params.q_pe, params.k_buf, params.latent_cache,
                    kv_cache_buffer, params.q_pe_ld, params.q_pe_stride, params.cos_sin_cache, params.head_num,
                    head_size, params.meta.kv_lora_rank, params.cu_q_seqlens, params.cache_seq_lens,
                    params.max_input_seq_len, params.cache_type, params.quant_scale_kv, params.helix_position_offsets,
                    params.absorption_mode, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1.0f,
                    kv_norm_w, params.kv_norm_eps, params.latent_row_stride, params.q_rope_done);
        }
        else if (useFusedFp8Q)
        {
            applyMLARopeAndAssignQKVKernelOptContext<T, 256, 448, 64, KVCacheBuffer, true>
                <<<grid, 256, 0, stream>>>(params.q_buf, params.q_pe, params.k_buf, params.latent_cache,
                    kv_cache_buffer, params.q_pe_ld, params.q_pe_stride, params.cos_sin_cache, params.head_num,
                    head_size, params.meta.kv_lora_rank, params.cu_q_seqlens, params.cache_seq_lens,
                    params.max_input_seq_len, params.cache_type, params.quant_scale_kv, params.helix_position_offsets,
                    params.absorption_mode, quant_q_fp8, params.quant_scale_qkv, params.bmm1_scale, params.bmm2_scale,
                    params.dequant_scale_q, params.dequant_scale_kv, params.quant_scale_o, params.host_bmm1_scale);
        }
        else
        {
            applyMLARopeAndAssignQKVKernelOptContext<T, 256, 448, 64, KVCacheBuffer, false><<<grid, 256, 0, stream>>>(
                params.q_buf, params.q_pe, params.k_buf, params.latent_cache, kv_cache_buffer, params.q_pe_ld,
                params.q_pe_stride, params.cos_sin_cache, params.head_num, head_size, params.meta.kv_lora_rank,
                params.cu_q_seqlens, params.cache_seq_lens, params.max_input_seq_len, params.cache_type,
                params.quant_scale_kv, params.helix_position_offsets, params.absorption_mode);
        }
    }
}

template <typename T>
void invokeMLAContextFp8Quantize(MlaParams<T>& params, int total_kv_len, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(params.cache_type == KvCacheDataType::FP8, "MLA Context: cache_type must be FP8");
    TLLM_CHECK_WITH_INFO(params.q_buf != nullptr, "MLA Context: q_buf must be non-null");
    TLLM_CHECK_WITH_INFO(params.absorption_mode || params.k_buf != nullptr,
        "MLA Context: k_buf must be non-null in non-absorption mode");
    TLLM_CHECK_WITH_INFO(params.absorption_mode || params.v_buf != nullptr,
        "MLA Context: v_buf must be non-null in non-absorption mode");
    TLLM_CHECK_WITH_INFO(params.quant_q_buf != nullptr, "MLA Context: quant_q_buf must be non-null");
    TLLM_CHECK_WITH_INFO(params.absorption_mode || params.quant_k_buf != nullptr,
        "MLA Context: quant_k_buf must be non-null in non-absorption mode");
    TLLM_CHECK_WITH_INFO(params.absorption_mode || params.quant_v_buf != nullptr,
        "MLA Context: quant_v_buf must be non-null in non-absorption mode");

    TLLM_LOG_DEBUG("MLA RoPE Context: Quantizing separate qkv to FP8");

    if (params.acc_q_len > 0)
    {
        // The Q tensor has layout of [num_tokens, head_num, 576] in the absorption mode.
        // Convert Q to FP8 in absorption mode.
        if (params.absorption_mode)
        {

            if (params.meta.rope_append)
            {
                constexpr int threads_per_block = 288;
                constexpr int num_tokens_per_block = threads_per_block * 16 / 576 * sizeof(T);
                dim3 grid(int(tensorrt_llm::common::divUp(total_kv_len, num_tokens_per_block)), 1, params.head_num);

                TLLM_LOG_DEBUG(
                    "Launching quantizeCopyInputToFp8Kernel with grid_size: (%d, %d, %d), threads_per_block: %d, "
                    "total_kv_len: %d, acc_q_len: %d, absorption_mode: %d",
                    grid.x, grid.y, grid.z, threads_per_block, total_kv_len, params.acc_q_len, params.absorption_mode);

                quantizeCopyInputToFp8Kernel<T, threads_per_block, 512, 64, 512, true>
                    <<<grid, threads_per_block, 0, stream>>>(params.q_buf,
                        static_cast<__nv_fp8_e4m3*>(params.quant_q_buf), params.k_buf,
                        static_cast<__nv_fp8_e4m3*>(params.quant_k_buf), params.v_buf,
                        static_cast<__nv_fp8_e4m3*>(params.quant_v_buf), params.acc_q_len, total_kv_len,
                        params.quant_scale_qkv, params.bmm1_scale, params.bmm2_scale, params.quant_scale_o,
                        params.dequant_scale_q, params.dequant_scale_kv, params.host_bmm1_scale);
            }
            else
            {
                constexpr int threads_per_block = 256;
                constexpr int num_tokens_per_block = threads_per_block * 16 / 512 * sizeof(T);
                dim3 grid(int(tensorrt_llm::common::divUp(total_kv_len, num_tokens_per_block)), 1, params.head_num);

                TLLM_LOG_DEBUG(
                    "Launching quantizeCopyInputToFp8Kernel with grid_size: (%d, %d, %d), threads_per_block: %d, "
                    "total_kv_len: %d, acc_q_len: %d, absorption_mode: %d",
                    grid.x, grid.y, grid.z, threads_per_block, total_kv_len, params.acc_q_len, params.absorption_mode);

                quantizeCopyInputToFp8Kernel<T, threads_per_block, 448, 64, 512, true>
                    <<<grid, threads_per_block, 0, stream>>>(params.q_buf,
                        static_cast<__nv_fp8_e4m3*>(params.quant_q_buf), params.k_buf,
                        static_cast<__nv_fp8_e4m3*>(params.quant_k_buf), params.v_buf,
                        static_cast<__nv_fp8_e4m3*>(params.quant_v_buf), params.acc_q_len, total_kv_len,
                        params.quant_scale_qkv, params.bmm1_scale, params.bmm2_scale, params.quant_scale_o,
                        params.dequant_scale_q, params.dequant_scale_kv, params.host_bmm1_scale);
            }
        }
        else
        {
            // The Q or K tensor has layout of [num_tokens, head_num, 192] in the non-absorption mode.
            // The V tensor has layout of [num_tokens, head_num, 128] in the non-absorption mode.
            // Convert Q, K, V to FP8 in non-absorption mode.

            constexpr int threads_per_block = 384;
            constexpr int num_tokens_per_block = threads_per_block * 16 / 192 * sizeof(T);
            dim3 grid(int(tensorrt_llm::common::divUp(total_kv_len, num_tokens_per_block)), 1, params.head_num);

            TLLM_LOG_DEBUG(
                "Launching quantizeCopyInputToFp8Kernel with grid_size: (%d, %d, %d), threads_per_block: %d, "
                "total_kv_len: %d, acc_q_len: %d, absorption_mode: %d",
                grid.x, grid.y, grid.z, threads_per_block, total_kv_len, params.acc_q_len, params.absorption_mode);

            quantizeCopyInputToFp8Kernel<T, threads_per_block, 128, 64, 128, false>
                <<<grid, threads_per_block, 0, stream>>>(params.q_buf, static_cast<__nv_fp8_e4m3*>(params.quant_q_buf),
                    params.k_buf, static_cast<__nv_fp8_e4m3*>(params.quant_k_buf), params.v_buf,
                    static_cast<__nv_fp8_e4m3*>(params.quant_v_buf), params.acc_q_len, total_kv_len,
                    params.quant_scale_qkv, params.bmm1_scale, params.bmm2_scale, params.quant_scale_o,
                    params.dequant_scale_q, params.dequant_scale_kv, params.host_bmm1_scale);
        }
    }
    else
    {
        TLLM_LOG_WARNING("MLA RoPE Context: acc_q_len is 0, skipping quantization.");
    }
}

template <typename T, typename KVCacheBuffer>
void invokeMLARopeGeneration(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream)
{
    // The trailing `head_num * 8` block rows are the q_nope FP8 quantize-copy region
    // (`head_idx > head_num + 8`). It dominates the launch -- at head_num 128 it is
    // 1024 of 1161 rows, 88% -- and it is pure requantization of a buffer another
    // kernel already produced. `deepseek_v4_q_norm_fused_fp8` writes that segment
    // straight out of q_b_layernorm, exactly as it already does for context, so when
    // the caller took that path the rows have nothing to do. Dropping them from the
    // grid makes the region unreachable; no separate template instance needed.
    bool const useFusedFp8Q = params.fuse_q_fp8_in_rope && params.cache_type == KvCacheDataType::FP8
        && params.quant_q_buf != nullptr && params.quant_scale_qkv != nullptr;

    dim3 grid(int(tensorrt_llm::common::divUp(params.acc_q_len, 32)), params.head_num + 1 + 8);
    if (params.cache_type == KvCacheDataType::FP8 && !useFusedFp8Q)
        grid.y += params.head_num * 8;
    TLLM_CHECK_WITH_INFO(params.acc_q_len % params.batch_size == 0,
        "MLA can only support input sequences with the same sequence length.");
    auto seq_len = params.acc_q_len / params.batch_size;

    // `fuse_kv_norm_in_rope` here means the KV work already happened in
    // `mlaKvNormRopeQuantGenerationKernel` (launched separately, fused with
    // kv_a_layernorm), so this kernel runs Q-only. DSv4 layout is a precondition of
    // that path, hence it only pairs with the 448 instantiation.
    bool const skip_kv = params.fuse_kv_norm_in_rope;
    TLLM_CHECK_WITH_INFO(!skip_kv || !params.meta.rope_append,
        "Fused generation kv-norm requires the DSv4 latent layout (rope_append=false).");

    auto* kernel_instance = &applyMLARopeAndAssignQKVKernelGeneration<T, 256, 512, 64, KVCacheBuffer, false>;
    if (!params.meta.rope_append)
    {
        kernel_instance = skip_kv ? &applyMLARopeAndAssignQKVKernelGeneration<T, 256, 448, 64, KVCacheBuffer, true>
                                  : &applyMLARopeAndAssignQKVKernelGeneration<T, 256, 448, 64, KVCacheBuffer, false>;
    }
    cudaLaunchConfig_t config;
    config.gridDim = grid;
    config.blockDim = 256;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    // Both halves of a Q row must land on one scale: on the fused path the nope
    // segment was quantized by `deepseek_v4_q_norm_fused_fp8` with `quant_scale_qkv`,
    // so the rope segment written here has to use the same tensor. They are both 1.0
    // today, which is why nothing broke, but that is a coincidence, not a contract.
    float const* quant_scale_q_eff = useFusedFp8Q ? params.quant_scale_qkv : params.quant_scale_q;

    cudaLaunchKernelEx(&config, kernel_instance, params.q_buf, params.q_pe, params.latent_cache, params.quant_q_buf,
        kv_cache_buffer, params.cos_sin_cache, params.head_num, params.meta.kv_lora_rank, params.acc_q_len, seq_len,
        params.seqQOffset, params.fmha_tile_counter, params.cache_seq_lens, params.cu_kv_seqlens, params.q_pe_ld,
        params.q_pe_stride, params.cache_type, params.bmm1_scale, params.bmm2_scale, params.quant_scale_o,
        quant_scale_q_eff, params.quant_scale_kv, params.dequant_scale_q, params.dequant_scale_kv,
        params.host_bmm1_scale, params.helix_position_offsets, params.helix_is_inactive_rank,
        params.precomputed_cu_seqlens, params.precomputed_fmha_scheduler);
}

template <typename T, typename KVCacheBuffer>
void invokeMLAKvNormRopeQuantGeneration(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream)
{
    // DSv4 layout only: the kernel describes the latent row with its K_DIM/ROPE_DIM
    // template constants, so kv_lora_rank must actually match.
    TLLM_CHECK_WITH_INFO(!params.meta.rope_append && params.meta.kv_lora_rank == 448,
        "Fused generation kv-norm requires the DSv4 latent layout (rope_append=false, kv_lora_rank=448), got "
        "rope_append=%d, kv_lora_rank=%d",
        static_cast<int>(params.meta.rope_append), params.meta.kv_lora_rank);
    TLLM_CHECK_WITH_INFO(params.kv_norm_weight != nullptr, "Fused generation kv-norm requires kv_norm_weight.");
    TLLM_CHECK_WITH_INFO(params.acc_q_len % params.batch_size == 0,
        "MLA can only support input sequences with the same sequence length.");

    auto const seq_len = params.acc_q_len / params.batch_size;
    auto const* kv_norm_w = static_cast<T const*>(params.kv_norm_weight);
    int const row_stride = params.latent_row_stride > 0 ? params.latent_row_stride : (params.meta.kv_lora_rank + 64);

    // The DSv4 sparse indices kernel owns the FMHA scheduler prologue when it runs;
    // null pointers turn the in-kernel writes off rather than duplicating them.
    auto* tile_counter = params.precomputed_fmha_scheduler ? nullptr : params.fmha_tile_counter;
    auto* bmm1_scale = params.precomputed_fmha_scheduler ? nullptr : params.bmm1_scale;
    auto* bmm2_scale = params.precomputed_fmha_scheduler ? nullptr : params.bmm2_scale;

    // One warp owns one latent row, so the block size decides how many rows a block
    // retires and therefore the grid size. 4 warps (128 threads) measured best or
    // tied-best with ncu at both ends of the decode range on GB200 -- 128 rows
    // (batch 32, MTP3) and 896 rows (batch 224, MTP3):
    //
    //   rows/block   block   b32 kernel   b224 kernel
    //     1            32      6688 ns      7168 ns
    //     4           128      6432 ns      6784 ns   <-- default
    //     8           256      7328 ns      6960 ns
    //
    // Sizing the grid to cover the SMs instead (divUp(rows, sm_count)) picks 1 at
    // batch 32 and 8 at batch 224, i.e. the slower option at both ends: the kernel
    // is latency-bound, not occupancy-bound, so SM coverage is the wrong knob.
    // `TRTLLM_MLA_KVNORM_GEN_ROWS_PER_BLOCK` pins it for re-tuning.
    int rows_per_block = 4;
    if (auto const env_rows = tensorrt_llm::common::getIntEnv("TRTLLM_MLA_KVNORM_GEN_ROWS_PER_BLOCK"))
    {
        rows_per_block = env_rows.value();
    }
    rows_per_block = std::min(std::max(rows_per_block, 1), 8);

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();

    auto launch = [&](auto block_size_tag)
    {
        constexpr int kBlockSize = decltype(block_size_tag)::value;
        constexpr int kRowsPerBlock = kBlockSize / 32;
        cudaLaunchConfig_t config;
        // Grid is sized purely by rows -- no coupling to head_num or to the FP8 grid
        // expansion the RoPE kernel needs.
        config.gridDim = dim3(static_cast<int>(tensorrt_llm::common::divUp(params.acc_q_len, kRowsPerBlock)));
        config.blockDim = kBlockSize;
        config.dynamicSmemBytes = 0;
        config.stream = stream;
        config.numAttrs = 1;
        config.attrs = attrs;
        cudaLaunchKernelEx(&config, &mlaKvNormRopeQuantGenerationKernel<T, kBlockSize, 448, 64, KVCacheBuffer>,
            params.latent_cache, kv_cache_buffer, params.cos_sin_cache, kv_norm_w, params.kv_norm_eps, row_stride,
            params.acc_q_len, seq_len, params.cache_seq_lens, params.cache_type, params.quant_scale_kv, tile_counter,
            bmm1_scale, bmm2_scale, params.quant_scale_o, params.dequant_scale_q, params.dequant_scale_kv,
            params.host_bmm1_scale);
    };

    if (rows_per_block <= 1)
    {
        launch(std::integral_constant<int, 32>{});
    }
    else if (rows_per_block <= 2)
    {
        launch(std::integral_constant<int, 64>{});
    }
    else if (rows_per_block <= 4)
    {
        launch(std::integral_constant<int, 128>{});
    }
    else
    {
        launch(std::integral_constant<int, 256>{});
    }
}

template <typename T, typename TCache>
void invokeMLALoadPagedKV(T* compressed_kv_ptr, T* k_pe_ptr, KVBlockArray& kv_cache, int const num_contexts,
    int64_t const* cu_ctx_cached_kv_lens, int const max_input_seq_len, int const lora_size, int const rope_size,
    float const* kv_scale_quant_orig_ptr, cudaStream_t stream)
{
    using KT = typename tensorrt_llm::kernels::loadPagedKVKernelTraits<TCache>;
    // {seq_len / token_per_block, batch_size, head_num}
    TLLM_CHECK_WITH_INFO(lora_size == KT::kLoraSize, "lora_size should be equal to %d", KT::kLoraSize);
    TLLM_CHECK_WITH_INFO(rope_size == KT::kRopeSize, "rope_size should be equal to %d", KT::kRopeSize);
    TLLM_CHECK_WITH_INFO(lora_size + rope_size == KT::kHeadSize, "head dim should be equal to %d", KT::kHeadSize);
    dim3 grid(static_cast<int>(tensorrt_llm::common::divUp(max_input_seq_len, KT::kTokenPerBlock)), num_contexts, 1);
    loadPagedKVCacheForMLAKernel<T, TCache><<<grid, KT::kBlockSize, 0, stream>>>(
        compressed_kv_ptr, k_pe_ptr, kv_cache, cu_ctx_cached_kv_lens, max_input_seq_len, kv_scale_quant_orig_ptr);
}

template <typename T, typename TCache>
void invokeMLARopeAppendPagedKVAssignQ(KVBlockArray& kv_cache, T* q_ptr, T* latent_cache_ptr, int const num_requests,
    int64_t const* cu_ctx_cached_kv_lens, int64_t const* cu_seq_lens, int const max_input_uncached_seq_len,
    float2 const* cos_sin_cache, size_t head_num, int nope_size, int rope_size, int lora_size,
    float const* kv_scale_orig_quant_ptr, cudaStream_t stream)
{
    dim3 grid(int(tensorrt_llm::common::divUp(max_input_uncached_seq_len, 32)), num_requests, head_num + 1 + 8);
    TLLM_CHECK_WITH_INFO(lora_size == 512 || lora_size == 448, "lora_size should be equal to %d or %d", 512, 448);
    TLLM_CHECK_WITH_INFO(rope_size == 64, "rope_size should be equal to %d", 64);
    if (lora_size == 512)
    {
        applyMLARopeAppendPagedKVAssignQKernel<T, TCache, 256, 512, 64><<<grid, 256, 0, stream>>>(kv_cache, q_ptr,
            latent_cache_ptr, cu_ctx_cached_kv_lens, cu_seq_lens, max_input_uncached_seq_len, cos_sin_cache, head_num,
            nope_size, kv_scale_orig_quant_ptr);
    }
    else
    {
        applyMLARopeAppendPagedKVAssignQKernel<T, TCache, 256, 448, 64><<<grid, 256, 0, stream>>>(kv_cache, q_ptr,
            latent_cache_ptr, cu_ctx_cached_kv_lens, cu_seq_lens, max_input_uncached_seq_len, cos_sin_cache, head_num,
            nope_size, kv_scale_orig_quant_ptr);
    }
}

#define INSTANTIATE_MLA_ROPE(T, KVCacheBuffer)                                                                         \
    template void invokeMLARopeContext(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream);      \
    template void invokeMLARopeGeneration(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream);   \
    template void invokeMLAKvNormRopeQuantGeneration(                                                                  \
        MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream);

INSTANTIATE_MLA_ROPE(float, KVBlockArray);
INSTANTIATE_MLA_ROPE(half, KVBlockArray);
INSTANTIATE_MLA_ROPE(float, KVLinearBuffer);
INSTANTIATE_MLA_ROPE(half, KVLinearBuffer);
INSTANTIATE_MLA_ROPE(__nv_bfloat16, KVBlockArray);
INSTANTIATE_MLA_ROPE(__nv_bfloat16, KVLinearBuffer);

#define INSTANTIATE_MLA_QUANTIZE(T)                                                                                    \
    template void invokeMLAContextFp8Quantize<T>(MlaParams<T> & params, int total_kv_len, cudaStream_t stream);

INSTANTIATE_MLA_QUANTIZE(float);
INSTANTIATE_MLA_QUANTIZE(half);
INSTANTIATE_MLA_QUANTIZE(__nv_bfloat16);

#define INSTANTIATE_RW_KVCACHE_MLA(T, TCache)                                                                          \
    template void invokeMLALoadPagedKV<T, TCache>(T * compressed_kv_ptr, T * k_pe_ptr, KVBlockArray & kv_cache,        \
        int const num_contexts, int64_t const* cu_ctx_cached_kv_lens, int const max_input_seq_len,                     \
        int const lora_size, int const rope_size, float const* kv_scale_quant_orig_ptr, cudaStream_t stream);          \
    template void invokeMLARopeAppendPagedKVAssignQ<T, TCache>(KVBlockArray & kv_cache, T * q_ptr,                     \
        T * latent_cache_ptr, int const num_requests, int64_t const* cu_ctx_cached_kv_lens,                            \
        int64_t const* cu_seq_lens, int const max_input_uncached_seq_len, float2 const* cos_sin_cache,                 \
        size_t head_num, int nope_size, int rope_size, int lora_size, float const* kv_scale_orig_quant_ptr,            \
        cudaStream_t stream);

INSTANTIATE_RW_KVCACHE_MLA(float, float);
INSTANTIATE_RW_KVCACHE_MLA(float, __nv_fp8_e4m3);
INSTANTIATE_RW_KVCACHE_MLA(half, half);
INSTANTIATE_RW_KVCACHE_MLA(half, __nv_fp8_e4m3);
INSTANTIATE_RW_KVCACHE_MLA(__nv_bfloat16, __nv_bfloat16);
INSTANTIATE_RW_KVCACHE_MLA(__nv_bfloat16, __nv_fp8_e4m3);

// In-place MLA RoPE: apply RoPE to the last rope_dim elements of each [nope_dim + rope_dim] head.
// Uses 16-byte vectorized load/store (VecType) and mmha::rotary_embedding_transform for the
// interleaved path. Each thread handles ELTS_PER_VEC elements (8 bf16 = 4 rotation pairs).
// Grid: (num_tokens, ceil(num_heads / HPB)), Block: (VECS_PER_ROPE, HPB)
// cos_sin_cache layout: [max_positions, 2, half_rope] float (cos block then sin block)
template <typename T, bool IS_INVERSE, bool IS_NEOX, int HEADS_PER_BLOCK>
__global__ void mlaRoPEInplaceKernel(T* __restrict__ data, int32_t const* __restrict__ position_ids,
    float const* __restrict__ cos_sin_cache, int num_heads, int nope_dim, int rope_dim)
{
    using VecT = typename VecType<T>::Type;
    using GPTJEltT = typename VecType<T>::GPTJEltType;
    constexpr int BYTES_PER_ELT = sizeof(T);
    constexpr int BYTES_PER_LOAD = 16;
    constexpr int ELTS_PER_VEC = BYTES_PER_LOAD / BYTES_PER_ELT;

    int const tid = threadIdx.x;
    int const half_rope = rope_dim / 2;
    // Neox: each thread handles one VecT from each half → half_rope elements per half
    // Interleaved: each thread handles one VecT of interleaved pairs → rope_dim elements
    int const vecs_per_rope
        = IS_NEOX ? (half_rope * BYTES_PER_ELT / BYTES_PER_LOAD) : (rope_dim * BYTES_PER_ELT / BYTES_PER_LOAD);
    int const head_idx = blockIdx.y * HEADS_PER_BLOCK + threadIdx.y;
    if (head_idx >= num_heads || tid >= vecs_per_rope)
        return;

    int const head_size = nope_dim + rope_dim;
    T* head_ptr = data + (static_cast<int64_t>(blockIdx.x) * num_heads + head_idx) * head_size;

    int const pos = position_ids[blockIdx.x];
    int const elem_offset = tid * ELTS_PER_VEC;
    // cos at [pos, 0, ...], sin at [pos, 1, ...]
    float const* cos_ptr = cos_sin_cache + pos * 2 * half_rope + elem_offset;
    float const* sin_ptr = cos_ptr + half_rope;

    if constexpr (IS_NEOX)
    {
        // Neox: first half = x1[0..half), second half = x2[0..half) — two separate 16-byte loads
        VecT v1 = *reinterpret_cast<VecT const*>(&head_ptr[nope_dim + elem_offset]);
        VecT v2 = *reinterpret_cast<VecT const*>(&head_ptr[nope_dim + half_rope + elem_offset]);

        // Each GPTJEltT holds 2 consecutive elements from the same half.
        // For neox, we rotate (v1[j], v2[j]) independently for each element j.
#pragma unroll
        for (int i = 0; i < ELTS_PER_VEC / 2; i++)
        {
            GPTJEltT& e1 = reinterpret_cast<GPTJEltT*>(&v1)[i];
            GPTJEltT& e2 = reinterpret_cast<GPTJEltT*>(&v2)[i];

            // Construct (x1, x2) pairs and rotate — 2 pairs per GPTJElt
            float2 coef0{cos_ptr[i * 2], IS_INVERSE ? -sin_ptr[i * 2] : sin_ptr[i * 2]};
            float2 coef1{cos_ptr[i * 2 + 1], IS_INVERSE ? -sin_ptr[i * 2 + 1] : sin_ptr[i * 2 + 1]};

            float2 p1 = mmha::rotary_embedding_transform(float2{static_cast<float>(reinterpret_cast<T*>(&e1)[0]),
                                                             static_cast<float>(reinterpret_cast<T*>(&e2)[0])},
                coef0);
            float2 p2 = mmha::rotary_embedding_transform(float2{static_cast<float>(reinterpret_cast<T*>(&e1)[1]),
                                                             static_cast<float>(reinterpret_cast<T*>(&e2)[1])},
                coef1);

            reinterpret_cast<T*>(&e1)[0] = static_cast<T>(p1.x);
            reinterpret_cast<T*>(&e1)[1] = static_cast<T>(p2.x);
            reinterpret_cast<T*>(&e2)[0] = static_cast<T>(p1.y);
            reinterpret_cast<T*>(&e2)[1] = static_cast<T>(p2.y);
        }

        *reinterpret_cast<VecT*>(&head_ptr[nope_dim + elem_offset]) = v1;
        *reinterpret_cast<VecT*>(&head_ptr[nope_dim + half_rope + elem_offset]) = v2;
    }
    else
    {
        // Interleaved: (x1, x2) adjacent pairs — matches GPTJ layout, single 16-byte load
        VecT v = *reinterpret_cast<VecT const*>(&head_ptr[nope_dim + elem_offset]);

        // For interleaved, cos_ptr/sin_ptr index by pair (half the element count)
        float const* cos_pair = cos_sin_cache + pos * 2 * half_rope + (elem_offset / 2);
        float const* sin_pair = cos_pair + half_rope;

#pragma unroll
        for (int i = 0; i < ELTS_PER_VEC / 2; i++)
        {
            GPTJEltT& elt = reinterpret_cast<GPTJEltT*>(&v)[i];
            float2 coef{cos_pair[i], IS_INVERSE ? -sin_pair[i] : sin_pair[i]};
            elt = mmha::rotary_embedding_transform(elt, coef);
        }

        *reinterpret_cast<VecT*>(&head_ptr[nope_dim + elem_offset]) = v;
    }
}

template <typename T>
void invokeMLARoPEInplace(T* data, int32_t const* position_ids, float const* cos_sin_cache, int num_tokens,
    int num_heads, int nope_dim, int rope_dim, bool inverse, bool is_neox, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(rope_dim % 4 == 0, "rope_dim must be divisible by 4");
    constexpr int BYTES_PER_LOAD = 16;
    int const elt_size = static_cast<int>(sizeof(T));

    auto launch = [&](auto inverse_tag, auto neox_tag)
    {
        constexpr bool INV = decltype(inverse_tag)::value;
        constexpr bool NEOX = decltype(neox_tag)::value;
        // Neox loads from two halves → threads = half_rope elements / ELTS_PER_VEC
        // Interleaved loads contiguous → threads = rope_dim elements / ELTS_PER_VEC
        int const active_elts = NEOX ? (rope_dim / 2) : rope_dim;
        int const vecs_per_rope = active_elts * elt_size / BYTES_PER_LOAD;

        constexpr int kMaxBlockSize = 256;
        constexpr int kMaxHeadsPerBlock = 16;
        int const hpb = std::max(1, std::min({kMaxBlockSize / vecs_per_rope, num_heads, kMaxHeadsPerBlock}));
        dim3 grid(num_tokens, (num_heads + hpb - 1) / hpb);

        if (hpb <= 4)
        {
            mlaRoPEInplaceKernel<T, INV, NEOX, 4><<<grid, dim3(vecs_per_rope, 4), 0, stream>>>(
                data, position_ids, cos_sin_cache, num_heads, nope_dim, rope_dim);
        }
        else if (hpb <= 8)
        {
            mlaRoPEInplaceKernel<T, INV, NEOX, 8><<<grid, dim3(vecs_per_rope, 8), 0, stream>>>(
                data, position_ids, cos_sin_cache, num_heads, nope_dim, rope_dim);
        }
        else
        {
            mlaRoPEInplaceKernel<T, INV, NEOX, 16><<<grid, dim3(vecs_per_rope, 16), 0, stream>>>(
                data, position_ids, cos_sin_cache, num_heads, nope_dim, rope_dim);
        }
    };

    if (inverse && is_neox)
        launch(std::true_type{}, std::true_type{});
    else if (inverse && !is_neox)
        launch(std::true_type{}, std::false_type{});
    else if (!inverse && is_neox)
        launch(std::false_type{}, std::true_type{});
    else
        launch(std::false_type{}, std::false_type{});
}

#define INSTANTIATE_MLA_ROPE_INPLACE(T)                                                                                \
    template void invokeMLARoPEInplace<T>(T * data, int32_t const* position_ids, float const* cos_sin_cache,           \
        int num_tokens, int num_heads, int nope_dim, int rope_dim, bool inverse, bool is_neox, cudaStream_t stream);

INSTANTIATE_MLA_ROPE_INPLACE(__nv_bfloat16);
INSTANTIATE_MLA_ROPE_INPLACE(half);

} // namespace kernels

TRTLLM_NAMESPACE_END
