/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include <assert.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

enum class KvCacheDataType;

struct MlaMetaParams
{
    int32_t q_lora_rank = 0;
    int32_t kv_lora_rank = 0;
    int32_t qk_nope_head_dim = 0;
    int32_t qk_rope_head_dim = 0;
    int32_t v_head_dim = 0;
    int32_t predicted_tokens_per_seq = 1;
    int32_t num_layers = 0;
    int32_t rope_append = 1;

    auto data() const
    {
        return std::make_tuple(q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
            predicted_tokens_per_seq, num_layers, rope_append);
    }
};

template <typename T>
struct MlaParams
{
    struct Dsv4EpilogueFusionParams
    {
        // Enable DSv4 inverse-RoPE + FP8 quant epilogue fusion.
        bool enabled = false;
        // The cos/sin cache used by the fused inverse-RoPE epilogue.
        float const* cos_sin_cache = nullptr;
        // The physical token stride of the FP32 output scale tensor.
        int32_t scale_buf_m = 0;
    };

    T const* latent_cache; // cKV + k_pe
    // Tensor Q for both context and generation MLA, contiguous. Pre-process kernel will apply RoPE and modify it
    // in-place. For context MLA, shape: [total_q_len, h * (d_nope + d_rope)], stride: [h * (d_nope + d_rope), 1]
    T* q_buf;
    // Separate tensor K for context MLA, contiguous. Pre-process kernel will apply RoPE and modify it in-place.
    // shape: [total_kv_len, h * (d_nope + d_rope)], stride: [h * (d_nope + d_rope), 1]
    T* k_buf = nullptr;
    // Separate tensor V for context MLA, NOT contiguous,
    // shape: [total_kv_len, h * d_v], stride: [h * (d_nope + d_v), 1]
    T const* v_buf = nullptr;
    // Tensor quantized Q for both context and generation MLA.
    // For context MLA, shape: [total_q_len, h * (d_nope + d_rope)], stride: [h * (d_nope + d_rope), 1]
    void* quant_q_buf = nullptr;
    // Tensor quantized K for context MLA, contiguous
    // shape: [total_kv_len, h * (d_nope + d_rope)], stride: [h * (d_nope + d_rope), 1]
    void* quant_k_buf = nullptr;
    // Tensor quantized V for context MLA, contiguous
    // shape: [total_kv_len, h * d_v], stride: [h * d_v, 1]
    void* quant_v_buf = nullptr;
    T* context_buf;
    T* q_pe;                     // [b, h, d_r], strided

    float2 const* cos_sin_cache; // [s, rope]
    int32_t batch_size;
    int32_t acc_q_len;
    int32_t head_num; // h
    void* workspace;
    int32_t const* cache_seq_lens;
    int* seqQOffset;
    uint32_t* fmha_tile_counter;
    int32_t max_input_seq_len;
    int* cu_q_seqlens;
    int* cu_kv_seqlens;
    int32_t q_pe_ld;
    int32_t q_pe_stride;
    MlaMetaParams meta;
    int const* block_ids_per_seq;
    // Pre-computed FlashMLA tile-scheduler metadata and num_splits from Python.
    // When non-null, mlaGeneration uses these directly and skips get_mla_metadata_func.
    int const* flash_mla_tile_scheduler_metadata = nullptr;
    int const* flash_mla_num_splits = nullptr;
    KvCacheDataType cache_type;
    // Separate E4M3 block-scale pool used by NVFP4 paged MLA cache.
    KVBlockArray kv_cache_block_scales_buffer{};
    // Scales for mla quantization
    float* bmm1_scale;
    float* bmm2_scale;
    float const* quant_scale_o;
    float const* quant_scale_q;
    float const* quant_scale_kv;
    float const* dequant_scale_q;
    float const* dequant_scale_kv;
    float host_bmm1_scale;

    // `seqQOffset` / `cu_kv_seqlens` already filled per iteration by the attention
    // metadata; layer-invariant, so do not recompute per layer.
    bool precomputed_cu_seqlens = false;

    // `fmha_tile_counter` and the bmm scales already written by the DSv4 sparse
    // indices kernel; skip them here.
    bool precomputed_fmha_scheduler = false;

    // Is it absorption mode?
    bool absorption_mode = false;

    // For FP8 context qkv quantization
    float const* quant_scale_qkv = nullptr;

    // Context RoPE kernel writes the rope segment straight to `quant_q_buf` as FP8,
    // dropping the standalone quantize pass. Nope segment must be pre-filled by
    // deepseek_v4_q_norm_fused_fp8.
    bool fuse_q_fp8_in_rope = false;

    // Fold kv_a_layernorm into the KV kernels: `latent_cache` is then the RAW
    // kv_a_proj output, RMS-normed over kv_lora_rank + qk_rope_head_dim before
    // RoPE + quant + paged write. Needs absorption mode and kv_lora_rank == K_DIM.
    bool fuse_kv_norm_in_rope = false;
    void const* kv_norm_weight = nullptr;
    float kv_norm_eps = 1e-6f;
    // `latent_cache` row stride in elements; the fused path passes a slice of
    // kv_a_proj, so rows are wider than packed. 0 means packed.
    int latent_row_stride = 0;

    // DSv4 fused inverse-RoPE + FP8 quant epilogue parameters.
    Dsv4EpilogueFusionParams dsv4_epilogue_fusion;

    // for Helix parallelism: the rotary position offsets [b]
    int32_t const* helix_position_offsets{nullptr};

    // for Helix parallelism: whether the current rank is inactive, shape [b]
    // (the current query tokens are not appended to this rank's KV cache)
    bool const* helix_is_inactive_rank{nullptr};
};

template <typename T, typename KVCacheBuffer>
void invokeMLARopeContext(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream);

template <typename T>
void invokeMLAContextFp8Quantize(MlaParams<T>& params, int total_kv_len, cudaStream_t stream);

template <typename T, typename KVCacheBuffer>
void invokeMLARopeGeneration(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream);

// Generation KV prologue in one warp-per-row pass: kv_a_layernorm + RoPE + FP8 quant
// + paged write. DSv4 layout only; `params.latent_cache` is the RAW kv_a_proj slice.
template <typename T, typename KVCacheBuffer>
void invokeMLAKvNormRopeQuantGeneration(MlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream);

template <typename T, typename TCache>
void invokeMLALoadPagedKV(T* compressed_kv_ptr, T* k_pe_ptr, KVBlockArray& kv_cache, int const num_contexts,
    int64_t const* cu_ctx_cached_kv_lens, int const max_input_seq_len, int const lora_size, int const rope_size,
    float const* kv_scale_quant_orig_ptr, cudaStream_t stream);

template <typename T, typename TCache>
void invokeMLARopeAppendPagedKVAssignQ(KVBlockArray& kv_cache, KVBlockArray& kv_scale_cache, T* q_ptr,
    T* latent_cache_ptr, int const num_requests, int64_t const* cu_ctx_cached_kv_lens, int64_t const* cu_seq_lens,
    int const max_input_uncached_seq_len, float2 const* cos_sin_cache, size_t head_num, int nope_size, int rope_size,
    int lora_size, KvCacheDataType cache_type, float const* kv_scale_orig_quant_ptr, cudaStream_t stream);

// Apply neox-style RoPE in-place to only the last rope_dim elements of each head,
// leaving the first nope_dim elements untouched.
// data shape: [num_tokens, num_heads, nope_dim + rope_dim]
// cos_sin_cache shape: [max_positions, 2, rope_dim/2] (float)
// position_ids shape: [num_tokens]
template <typename T>
void invokeMLARoPEInplace(T* data, int32_t const* position_ids, float const* cos_sin_cache, int num_tokens,
    int num_heads, int nope_dim, int rope_dim, bool inverse, bool is_neox, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
