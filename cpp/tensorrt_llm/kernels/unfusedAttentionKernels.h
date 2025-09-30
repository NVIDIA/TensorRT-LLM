/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/mlaKernels.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include <cuda_runtime_api.h>

#ifdef ENABLE_FP4
#include <cuda_fp4.h>
#endif

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
void invokeAddQKVBiasIA3Transpose(T* q_buf, T* k_buf, T* v_buf, T* Q, T const* bias_Q, T* K, T const* bias_K, T* V,
    T const* bias_V, int const batch_size, int const seq_len, int const head_num, int const size_per_head,
    int const* ia3_tasks, T const* ia3_key_weights, T const* ia3_value_weights, cudaStream_t stream);

template <typename T, typename T_IN>
struct MaskedSoftmaxParam
{
    // Common parameters.
    T* attention_score = nullptr;      // (batch_size, head_num, q_length, k_length)
    const T_IN* qk = nullptr;          // (batch_size, head_num, q_length, k_length)
    T const* attention_mask = nullptr; // (batch_size, q_length, k_length)
    int batch_size = 0;
    int q_length = 0;
    int k_length = 0;
    int num_heads = 0;
    T qk_scale = T(0.0f);
    // always float compute data type.
    float attn_logit_softcapping_scale = 0.f;
    float attn_logit_softcapping_inverse_scale = 0.f;
    bool block_sparse_attn = false;
    BlockSparseParams block_sparse_params;
    int const* q_seq_lengths = nullptr; // (batch_size)

    // Optional parameters that depend on the type of attention.
    // The slopes of the linear position bias of ALiBi.
    T const* linear_bias_slopes = nullptr; // (head_num,), optional
};

enum class KvCacheDataType
{
    BASE = 0,
    INT8,
    FP8,
    NVFP4
};

enum class RotaryPositionEmbeddingType
{
    NONE = 0,
    GPTJ,
    GPT_NEOX,
};

inline void calGridSizeWithBestEfficiency(
    dim3 const block, dim3& grid, int numNeededBlockX, int multiProcessorCount, int availableThreadsPerSm)
{
    int numThreadsPerBlock = block.x * block.y * block.z;
    int numAvailableBlocksPerWave = int(availableThreadsPerSm / numThreadsPerBlock) * multiProcessorCount;
    int numBatchBlocks = grid.y * grid.z;

    // The best wave efficiency it can achieve with different number of blocks.
    float bestEfficiency = 0.f;
    int selectedBlockX = 0;
    // Iterate over all possible number of blocks in the x dimension.
    for (int blockX = 1; blockX <= numNeededBlockX; ++blockX)
    {
        float numWaves = float(blockX * numBatchBlocks) / numAvailableBlocksPerWave;
        float efficiency = numWaves / std::ceil(numWaves);
        if (efficiency > bestEfficiency)
        {
            bestEfficiency = efficiency;
            selectedBlockX = blockX;
        }
    }
    // Update grid.x.
    grid.x = selectedBlockX;
}

template <typename T, typename KVCacheBuffer>
struct QKVPreprocessingParams
{
    // Buffers.
    // source buffer
    // also acts as a dst buffer based on if separate_q_kv_output
    T* qkv_input{nullptr};
    // The cross attention kv (= qkv_project(encoder_output), and only slice the kv part).
    T* cross_kv_input{nullptr};
    // Only used by fp8 quantized output currently.
    void* quantized_qkv_output{nullptr};
    // The separate q output.
    T* q_output{nullptr};
    // the classes used for this template are either KVLinearBuffer, KVBlockArray
    // for more details, refer to kvCacheUtils.h
    KVCacheBuffer kv_cache_buffer{};
    // Pool for FP4 KV cache block scales. Unused for other KV cache dtypes.
    KVCacheBuffer kv_cache_block_scales_buffer{};
    T const* qkv_bias{nullptr};

    // Fuse the computation of FMHA quantization scales into the preprocessing kernels.
    // This can also be done in gptKernels.h if there is no preprocessing kernels.
    // The scale to dequant Q/Kv input.
    float const* qkv_scale_quant_orig{nullptr};
    float const* qkv_scale_orig_quant{nullptr};
    // The scale to quant O output.
    float const* o_scale_orig_quant{nullptr};
    // The scale after fmha bmm1.
    float* fmha_bmm1_scale{nullptr};
    // The scale after fmha bmm2.
    float* fmha_bmm2_scale{nullptr};
    // The fmha tile counter (used by hopper fmha kernels).
    float* fmha_tile_counter{nullptr};

    // Logn scaling pointer, of shape {max_position_embedding_length}
    float const* logn_scaling{nullptr};
    // The (batch_idx, token_idx_in_seq) int2 buffer of shape {num_tokens}.
    int2 const* tokens_info{nullptr};
    // list of sequence lengths, of shape {batch_size + 1}
    int const* seq_lens{nullptr};
    // list sequence lengths for the cache, of shape {batch_size + 1}
    // this is normally used to indicate if chunked context is used (i.e. cache_seqlen > input_seqlen).
    int const* cache_seq_lens{nullptr};
    // list sequence lengths for the encoder, of shape {batch_size + 1}
    int const* encoder_seq_lens{nullptr};
    // list of cumulative sequence lengths, of shape {batch_size + 1}
    int const* cu_seq_lens{nullptr};
    // list of cumulative KV sequence lengths, of shape {batch_size + 1}, used by cross attention only.
    int const* cu_kv_seq_lens{nullptr};
    // list of cumulative length of sparse KV indices, of shape {batch_size + 1}
    int const* sparse_kv_offsets{nullptr};
    // list of sparse KV indices for writing to KV cache, of shape {num_kv_heads, num_sparse_kv_indices}
    int const* sparse_kv_indices{nullptr};
    // inverse frequencies (angle raised at various powers) from the RoPE formula
    // shape of {batch_size , rotaryEmbeddingDim / 2}
    float const* rotary_embedding_inv_freq{nullptr};
    // the pre-computed RoPE factors. computed at model build time, stored in the engine
    // shape is {rotary_embedding_max_positions, rotary_embedding_dim}. eg (2048, 128)
    float2 const* rotary_coef_cache_buffer{nullptr};
    int const* spec_decoding_position_offsets{nullptr};

    float2 const* mrope_rotary_cos_sin{nullptr};
    int32_t const* mrope_position_deltas{nullptr};

    // Scalars.
    int batch_size{0};
    int max_input_seq_len{0};
    int max_kv_seq_len{0};
    int cyclic_kv_cache_len{0};
    int sink_token_len{0};
    int token_num{0};
    bool remove_padding{true};
    bool is_last_chunk{true};
    bool cross_attention{false};
    int head_num{0};
    int kv_head_num{0};
    int qheads_per_kv_head{0};
    int size_per_head{0};
    // The fmha bmm1 host scale (1.0f / sqrt(headSize) by default).
    float fmha_host_bmm1_scale{1.f};
    int rotary_embedding_dim{0};
    float rotary_embedding_base{0.};
    RotaryScalingType rotary_scale_type{};
    // TODO(dblanaru) remove this, its unused. Inv freq are precomputed elsewhere
    float rotary_embedding_scale{0.};
    int rotary_embedding_max_positions{0};
    PositionEmbeddingType position_embedding_type{};
    bool position_shift_enabled{false};
    KvCacheDataType cache_type{};
    bool separate_q_kv_output{false};
    bool quantized_fp8_output{false};
    bool generation_phase{false};
    int multi_processor_count{0};
    int rotary_vision_start{0};
    int rotary_vision_length{0};
    // Pre-compute on host.
    int half_rotary_dim{0};
    int q_hidden_size{0};
    int kv_hidden_size{0};
    int hidden_size{0};

    void setCommonParameters()
    {
        half_rotary_dim = rotary_embedding_dim / 2;
        q_hidden_size = head_num * size_per_head;
        kv_hidden_size = kv_head_num * size_per_head;
        hidden_size = q_hidden_size + 2 * kv_hidden_size;
    }

    std::string toString() const
    {
        std::stringstream ss;

        ss << "QKVPreprocessingParams ====================" << std::endl;
        ss << "qkv_input: " << qkv_input << std::endl;
        ss << "cross_kv_input: " << cross_kv_input << std::endl;
        ss << "quantized_qkv_output: " << quantized_qkv_output << std::endl;
        ss << "q_output: " << q_output << std::endl;
        ss << "kv_cache_buffer: " << kv_cache_buffer.data << std::endl;
        ss << "kv_cache_block_scales_buffer: " << kv_cache_block_scales_buffer.data << std::endl;
        ss << "qkv_bias: " << qkv_bias << std::endl;
        ss << "tokens_info: " << tokens_info << std::endl;
        if (seq_lens && batch_size > 0)
        {
            ss << "seq_lens: "
               << *(runtime::ITensor::wrap(
                      (void*) seq_lens, nvinfer1::DataType::kINT32, runtime::ITensor::makeShape({batch_size})));
        }
        if (cache_seq_lens && batch_size > 0)
        {
            ss << "cache_seq_lens: "
               << *(runtime::ITensor::wrap(
                      (void*) cache_seq_lens, nvinfer1::DataType::kINT32, runtime::ITensor::makeShape({batch_size})));
        }
        if (encoder_seq_lens && batch_size > 0)
        {
            ss << "encoder_seq_lens: "
               << *(runtime::ITensor::wrap(
                      (void*) encoder_seq_lens, nvinfer1::DataType::kINT32, runtime::ITensor::makeShape({batch_size})));
        }
        if (cu_seq_lens && batch_size > 0)
        {
            ss << "cu_seq_lens: "
               << *(runtime::ITensor::wrap(
                      (void*) cu_seq_lens, nvinfer1::DataType::kINT32, runtime::ITensor::makeShape({batch_size})));
        }
        if (cu_kv_seq_lens && batch_size > 0)
        {
            ss << "cu_kv_seq_lens: "
               << *(runtime::ITensor::wrap(
                      (void*) cu_kv_seq_lens, nvinfer1::DataType::kINT32, runtime::ITensor::makeShape({batch_size})));
        }
        if (sparse_kv_offsets)
        {
            ss << "sparse_kv_offsets: "
               << *(runtime::ITensor::wrap((void*) sparse_kv_offsets, nvinfer1::DataType::kINT32,
                      runtime::ITensor::makeShape({batch_size + 1})));
        }
        if (rotary_embedding_inv_freq && batch_size > 0 && rotary_embedding_dim > 0)
        {
            ss << "rotary_embedding_inv_freq: "
               << *(runtime::ITensor::wrap((void*) rotary_embedding_inv_freq, nvinfer1::DataType::kFLOAT,
                      runtime::ITensor::makeShape({batch_size, rotary_embedding_dim / 2})));
        }
        ss << "rotary_coef_cache_buffer: " << rotary_coef_cache_buffer << std::endl;
        ss << "qkv_scale_orig_quant: " << qkv_scale_orig_quant << std::endl;
        ss << "spec_decoding_position_offsets: " << spec_decoding_position_offsets << std::endl;
        ss << "batch_size: " << batch_size << std::endl;
        ss << "max_input_seq_len: " << max_input_seq_len << std::endl;
        ss << "max_kv_seq_len: " << max_kv_seq_len << std::endl;
        ss << "cyclic_kv_cache_len: " << cyclic_kv_cache_len << std::endl;
        ss << "sink_token_len: " << sink_token_len << std::endl;
        ss << "token_num: " << token_num << std::endl;
        ss << "remove_padding: " << remove_padding << std::endl;
        ss << "is_last_chunk: " << is_last_chunk << std::endl;
        ss << "cross_attention: " << cross_attention << std::endl;
        ss << "head_num: " << head_num << std::endl;
        ss << "kv_head_num: " << kv_head_num << std::endl;
        ss << "qheads_per_kv_head: " << qheads_per_kv_head << std::endl;
        ss << "size_per_head: " << size_per_head << std::endl;
        ss << "rotary_embedding_dim: " << rotary_embedding_dim << std::endl;
        ss << "rotary_embedding_base: " << rotary_embedding_base << std::endl;
        ss << "rotary_scale_type: " << static_cast<int>(rotary_scale_type) << std::endl;
        ss << "rotary_embedding_scale: " << rotary_embedding_scale << std::endl;
        ss << "rotary_embedding_max_positions: " << rotary_embedding_max_positions << std::endl;
        ss << "position_embedding_type: " << static_cast<int>(position_embedding_type) << std::endl;
        ss << "position_shift_enabled: " << std::boolalpha << position_shift_enabled << std::endl;
        ss << "cache_type: " << static_cast<int>(cache_type) << std::endl;
        ss << "separate_q_kv_output: " << std::boolalpha << separate_q_kv_output << std::endl;
        ss << "quantized_fp8_output: " << quantized_fp8_output << std::endl;
        ss << "generation_phase: " << generation_phase << std::endl;
        ss << "multi_processor_count: " << multi_processor_count << std::endl;

        return ss.str();
    }
};

template <typename T, typename T_IN>
void invokeMaskedSoftmax(MaskedSoftmaxParam<T, T_IN>& param, cudaStream_t stream);

template <typename T>
void invokeTransposeQKV(T* dst, T* src, int const batch_size, int const seq_len, int const head_num,
    int const size_per_head, float const* scale, int const int8_mode, cudaStream_t stream);

template <typename T>
void invokeAddQKVBiasIA3RebuildPadding(T* Q, T const* bias_Q, T* K, T const* bias_K, T* V, T const* bias_V, T* q_buf,
    T* k_buf, T* v_buf, int const batch_size, int const seq_len, int const head_num, int const size_per_head,
    int const valid_word_num, int const* mask_offset, int const* ia3_tasks, T const* ia3_key_weights,
    T const* ia3_value_weights, cudaStream_t stream);

template <typename T>
void invokeTransposeAttentionOutRemovePadding(T* src, T* dst, int const valid_word_num, int const batch_size,
    int const seq_len, int const head_num, int const size_per_head, int const* mask_offset, float const* scale,
    int const int8_mode, cudaStream_t stream);

template <typename T>
void invokeAddFusedQKVBiasTranspose(T* q_buf, T* k_buf, T* v_buf, T* QKV, T const* qkv_bias, int const* seq_lens,
    int const* padding_offset, int const batch_size, int const seq_len, int const token_num, int const head_num,
    int const kv_head_num, int const size_per_head, int const rotary_embedding_dim, float rotary_embedding_base,
    const RotaryScalingType rotary_scale_type, float rotary_embedding_scale, int const rotary_embedding_max_positions,
    PositionEmbeddingType const position_embedding_type, float const* scale, int const int8_mode, cudaStream_t stream);

template <typename T>
void invokeAddFusedQKVBiasTranspose(T* q_buf, T* k_buf, T* v_buf, T* QKV, T const* qkv_bias, int const* seq_lens,
    int const* padding_offset, int const batch_size, int const seq_len, int const token_num, int const head_num,
    int const kv_head_num, int const size_per_head, cudaStream_t stream)
{
    invokeAddFusedQKVBiasTranspose(q_buf, k_buf, v_buf, QKV, qkv_bias, seq_lens, padding_offset, batch_size, seq_len,
        token_num, head_num, kv_head_num, size_per_head, 0, false, (float*) nullptr, 0, stream);
}

template <typename T>
void invokeAddFusedQKVBiasTranspose(T* q_buf, T* k_buf, T* v_buf, T* QKV, int const* seq_lens,
    int const* padding_offset, int const batch_size, int const seq_len, int const token_num, int const head_num,
    int const kv_head_num, int const size_per_head, int const rotary_embedding_dim, float rotary_embedding_base,
    const RotaryScalingType rotary_scale_type, float rotary_embedding_scale, int const rotary_embedding_max_positions,
    PositionEmbeddingType const position_embedding_type, float const* scale, int const int8_mode, cudaStream_t stream)
{
    invokeAddFusedQKVBiasTranspose(q_buf, k_buf, v_buf, QKV, (T const*) nullptr, seq_lens, padding_offset, batch_size,
        seq_len, token_num, head_num, kv_head_num, size_per_head, rotary_embedding_dim, rotary_embedding_base,
        rotary_scale_type, rotary_embedding_scale, rotary_embedding_max_positions, position_embedding_type, scale,
        int8_mode, stream);
}

template <typename T, typename KVCacheBuffer>
void invokeTranspose4dBatchMajor(T const* k_src, T const* v_src, KVCacheBuffer& kvTable, int const local_batch_size,
    int const seq_len, int const max_attention_window_size, int const size_per_head, int const local_head_num,
    const KvCacheDataType cache_type, float const* kvScaleOrigQuant, int const* sequence_lengths, cudaStream_t stream);

template <typename T, typename T_cache, typename KVCacheBuffer>
void invokeApplyBiasRopeUpdateKVCacheDispatch(QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream);

// NOTE: this kernel is in-place, QKV will be modified, if other kernels need that, may need copy or use before it.
template <typename T, typename KVCacheBuffer>
void invokeQKVPreprocessing(QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream)
{
    params.setCommonParameters();
    if (params.cache_type == KvCacheDataType::INT8)
    {
        invokeApplyBiasRopeUpdateKVCacheDispatch<T, int8_t, KVCacheBuffer>(params, stream);
    }
#ifdef ENABLE_FP8
    else if (params.cache_type == KvCacheDataType::FP8)
    {
        invokeApplyBiasRopeUpdateKVCacheDispatch<T, __nv_fp8_e4m3, KVCacheBuffer>(params, stream);
    }
#endif // ENABLE_FP8
#ifdef ENABLE_FP4
    else if (params.cache_type == KvCacheDataType::NVFP4)
    {
        TLLM_CHECK_WITH_INFO(params.kv_cache_block_scales_buffer.data != nullptr,
            "Cannot append to FP4 KV cache without block scales pool");
        if constexpr (std::is_same_v<T, float>)
        {
            // TODO: needs special quantization logic. The existing quantization functions
            // are specially designed for 16 bit types.
            TLLM_THROW("Cannot use FP4 KV cache with FP32 model.");
        }
        else
        {
            invokeApplyBiasRopeUpdateKVCacheDispatch<T, __nv_fp4_e2m1, KVCacheBuffer>(params, stream);
        }
    }
#endif
    else
    {
        invokeApplyBiasRopeUpdateKVCacheDispatch<T, T, KVCacheBuffer>(params, stream);
    }
}

template <typename T, typename T_cache, typename KVCacheBuffer>
void invokeUpdateCyclicKvCacheAfterFmha(QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream);

template <typename T, typename T_cache, typename KVCacheBuffer>
void invokeUpdateSparseKvCacheAfterFmha(QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream);

// Debug function to test basic parameter access
template <typename T, typename KVCacheBuffer>
void invokeDebugSparseKvCacheParams(
    QKVPreprocessingParams<T, KVCacheBuffer> params, int* debug_output, cudaStream_t stream);

template <typename T, typename KVCacheBuffer>
void invokeKvCachePostprocessing(QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream)
{
    params.setCommonParameters();

    // handle sparse KV cache update if needed
    if (params.sparse_kv_indices != nullptr && params.sparse_kv_offsets != nullptr && params.is_last_chunk)
    {
        if (params.cache_type == KvCacheDataType::INT8)
        {
            invokeUpdateSparseKvCacheAfterFmha<T, int8_t, KVCacheBuffer>(params, stream);
        }
#ifdef ENABLE_FP8
        else if (params.cache_type == KvCacheDataType::FP8)
        {
            invokeUpdateSparseKvCacheAfterFmha<T, __nv_fp8_e4m3, KVCacheBuffer>(params, stream);
        }
#endif // ENABLE_FP8
        else
        {
            invokeUpdateSparseKvCacheAfterFmha<T, T, KVCacheBuffer>(params, stream);
        }
    }

    // handle cyclic KV cache update if needed
    // now we don't update cyclic KV cache
    if (false)
    {
        if (params.cache_type == KvCacheDataType::INT8)
        {
            invokeUpdateCyclicKvCacheAfterFmha<T, int8_t, KVCacheBuffer>(params, stream);
        }
#ifdef ENABLE_FP8
        else if (params.cache_type == KvCacheDataType::FP8)
        {
            invokeUpdateCyclicKvCacheAfterFmha<T, __nv_fp8_e4m3, KVCacheBuffer>(params, stream);
        }
#endif // ENABLE_FP8
        else
        {
            invokeUpdateCyclicKvCacheAfterFmha<T, T, KVCacheBuffer>(params, stream);
        }
    }
}

template <typename T, typename BT>
void invokeAddRelativeAttentionBiasUnaligned(T* qk_buf, const BT* relative_attention_bias, int const batch_size,
    int const head_num, int const seq_len, int const max_seq_len, cudaStream_t stream, bool implicit = false,
    int num_buckets = 0, int max_distance = 0, bool bidirectional = true);

template <typename T, typename KVCacheBuffer>
void invokeShiftKCache(KVCacheBuffer const& kvCacheBuffer, KVLinearBuffer const& shiftKCacheBuffer,
    const KvCacheDataType cache_type, int const sizePerHead, int const timestep, int const batch_beam,
    int const kv_head_num, int const beam_width, int const maxKCacheLen, int const sinkTokenLen,
    float const* kScaleQuantOrig, int const* sequence_lengths, int const* input_lengths, int const rotary_embedding_dim,
    float rotary_embedding_base, RotaryScalingType const rotary_scale_type, float rotary_embedding_scale,
    int const rotary_embedding_max_positions, PositionEmbeddingType const position_embedding_type, cudaStream_t stream);

// compute src[x] * scale[0] and write into dst[x]
template <typename Dst, typename Src>
void invokeConversion(Dst* dst, Src const* src, int64_t size, float const* __restrict__ scale, cudaStream_t stream);

template <typename T>
void invokeCpTranspose(T* dst, T* dst2, T const* src, int64_t partialLength, int64_t cpSize, int64_t partialQHeads,
    int64_t partialKVHeads, int64_t mqaBroadcast, int64_t headSize, int64_t rank, cudaStream_t stream);

template <typename T>
void invokeCpTransposeToSeqMajor(T* dst, T const* srcMyRank, T const* srcOtherRank, int64_t partialLength,
    int64_t cpSize, int64_t newPartialHeads, int64_t headSize, int64_t rank, cudaStream_t stream);

template <typename T>
void invokeCpTranspose2(T* dst, T const* src, int32_t const* q_seq_lengths, int32_t const* cu_q_seqlens,
    int32_t const* cu_cp_partial_seqlens, int64_t cpSize, int64_t maxPartalLength, int64_t batchSize,
    int64_t partialHeads, int64_t headSize, cudaStream_t stream);

template <typename T>
void invokeCpTransposeToSeqMajor2(T* dst, T const* src, int32_t const* q_seq_lengths, int32_t const* cu_q_seqlens,
    int32_t const* cu_cp_partial_seqlens, int64_t cpSize, int64_t maxPartalLength, int64_t batchSize,
    int64_t partialHeads, int64_t headSize, cudaStream_t stream);

} // namespace kernels

} // namespace tensorrt_llm
