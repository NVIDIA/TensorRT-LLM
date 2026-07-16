/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "tensorrt_llm/common/attentionOp.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/attentionOp.h"
#include "tensorrt_llm/thop/trtllmGenFusedOps.h"
#include <ATen/cuda/CUDAContext.h>
#include <optional>
#include <torch/extension.h>
#include <tuple>
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

using tensorrt_llm::common::op::AttentionOp;
using tensorrt_llm::kernels::AttentionMaskType;
using tensorrt_llm::kernels::BlockSparseParams;
using tensorrt_llm::kernels::BuildDecoderInfoParams;
using tensorrt_llm::kernels::KVBlockArray;
using tensorrt_llm::kernels::KvCacheDataType;
using tensorrt_llm::kernels::PositionEmbeddingType;
using tensorrt_llm::kernels::QKVPreprocessingParams;
using tensorrt_llm::kernels::RotaryScalingType;
using tensorrt_llm::kernels::cacheTypeFromQuantMode;
using tensorrt_llm::runtime::TorchUtils;

namespace
{

int64_t computeWindowLeft(
    int64_t const cyclicAttentionWindowSize, int64_t const maxKvLength, int64_t const attentionChunkSize)
{
    TORCH_CHECK(!(attentionChunkSize != 0 && cyclicAttentionWindowSize < maxKvLength),
        "Chunked-attention and sliding-window-attention should not be enabled at the same time.");
    if (0 < cyclicAttentionWindowSize && cyclicAttentionWindowSize < maxKvLength)
    {
        return cyclicAttentionWindowSize - 1;
    }
    return -1;
}

template <typename T, typename OptTensorT>
T* optPtr(OptTensorT&& t, std::enable_if_t<!std::is_const_v<OptTensorT>>* = nullptr)
{
    return t.has_value() ? static_cast<T*>(t->data_ptr()) : nullptr;
}

template <typename T, typename OptTensorT>
T const* optPtr(OptTensorT&& t, std::enable_if_t<std::is_const_v<OptTensorT>>* = nullptr)
{
    return t.has_value() ? static_cast<T const*>(t->data_ptr()) : nullptr;
}

cudaStream_t currentStreamFor(at::Tensor const& tensor)
{
    return at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
}

struct WorkspaceAccessor
{
    uint8_t* base{};
    int64_t sizeBytes{};
    at::Device device;

    explicit WorkspaceAccessor(at::Tensor const& workspace)
        : base(static_cast<uint8_t*>(workspace.data_ptr()))
        , sizeBytes(static_cast<int64_t>(workspace.nbytes()))
        , device(workspace.device())
    {
    }

    void* ptr(int64_t const offset, int64_t const numBytes) const
    {
        if (numBytes == 0)
        {
            return nullptr;
        }

        TORCH_CHECK(offset >= 0, "Negative workspace offset is invalid.");
        TORCH_CHECK(numBytes >= 0, "Negative workspace slice size is invalid.");
        TORCH_CHECK(offset + numBytes <= sizeBytes, "Workspace view exceeds bounds.");
        return base + offset;
    }

    template <typename T>
    T* ptr(int64_t const offset, int64_t const numBytes) const
    {
        return static_cast<T*>(ptr(offset, numBytes));
    }

    at::TensorOptions options(at::ScalarType const scalarType) const
    {
        return at::TensorOptions().dtype(scalarType).device(device);
    }

    at::Tensor tensor(int64_t const offset, int64_t const numBytes, int64_t const itemSize,
        at::TensorOptions const& tensorOptions) const
    {
        TORCH_CHECK(numBytes > 0, "Cannot create a Tensor view for an empty workspace slice.");
        TORCH_CHECK(numBytes % itemSize == 0, "Workspace slice is not aligned to dtype size.");
        return torch::from_blob(ptr(offset, numBytes), {numBytes / itemSize}, tensorOptions);
    }

    at::Tensor tensor(int64_t const offset, int64_t const numBytes, at::ScalarType const scalarType) const
    {
        auto const itemSize = static_cast<int64_t>(c10::elementSize(scalarType));
        return tensor(offset, numBytes, itemSize, options(scalarType));
    }
};

struct ContextWorkspaceRawPointers
{
    int* cuQSeqlensPtr{};
    int* cuKvSeqlensPtr{};
    int* cuMaskRowsPtr{};
    float* rotaryInvFreqBufPtr{};
    void* qBufPtr{};
    int2* tokensInfoPtr{};
    uint32_t* fmhaTileCounterPtr{};
    float* fmhaBmm1ScalePtr{};
    float* fmhaBmm2ScalePtr{};
};

ContextWorkspaceRawPointers makeContextWorkspaceRawPointers(
    WorkspaceAccessor const& workspace, TrtllmGenContextWorkspaceLayout const& layout)
{
    return ContextWorkspaceRawPointers{
        .cuQSeqlensPtr = workspace.ptr<int>(layout.cuQSeqlensOffset, layout.cuSeqlensSize),
        .cuKvSeqlensPtr = workspace.ptr<int>(layout.cuKvSeqlensOffset, layout.cuSeqlensSize),
        .cuMaskRowsPtr = workspace.ptr<int>(layout.cuMaskRowsOffset, layout.cuSeqlensSize),
        .rotaryInvFreqBufPtr = workspace.ptr<float>(layout.rotaryInvFreqOffset, layout.rotaryInvFreqSize),
        .qBufPtr = workspace.ptr(layout.qBufOffset, layout.qBufSize),
        .tokensInfoPtr = workspace.ptr<int2>(layout.tokensInfoOffset, layout.tokensInfoSize),
        .fmhaTileCounterPtr = workspace.ptr<uint32_t>(layout.fmhaTileCounterOffset, layout.fmhaTileCounterSize),
        .fmhaBmm1ScalePtr = workspace.ptr<float>(layout.fmhaBmm1ScaleOffset, layout.fmhaBmm1ScaleSize),
        .fmhaBmm2ScalePtr = workspace.ptr<float>(layout.fmhaBmm2ScaleOffset, layout.fmhaBmm2ScaleSize),
    };
}

struct ContextWorkspaceRawViews
{
    at::Tensor trtllmGenWorkspace;
    at::Tensor cuQSeqlens;
    at::Tensor cuKvSeqlens;
    at::Tensor qBuf;
    std::optional<at::Tensor> fmhaBmm1Scale;
    std::optional<at::Tensor> fmhaBmm2Scale;
    ContextWorkspaceRawPointers ptrs;
};

ContextWorkspaceRawViews makeContextWorkspaceRawViews(
    at::Tensor const& workspace, TrtllmGenContextWorkspaceLayout const& layout, bool const materializeQBufTensor)
{
    WorkspaceAccessor const workspaceView{workspace};
    auto const byteOptions = workspaceView.options(at::kByte);
    auto const intOptions = workspaceView.options(at::kInt);
    at::Tensor qBuf;
    if (materializeQBufTensor)
    {
        qBuf = workspaceView.tensor(layout.qBufOffset, layout.qBufSize, layout.qBufScalarType);
    }
    std::optional<at::Tensor> fmhaBmm1Scale;
    std::optional<at::Tensor> fmhaBmm2Scale;
    if (layout.fmhaBmm1ScaleSize > 0)
    {
        fmhaBmm1Scale = workspaceView.tensor(layout.fmhaBmm1ScaleOffset, layout.fmhaBmm1ScaleSize, at::kFloat);
    }
    if (layout.fmhaBmm2ScaleSize > 0)
    {
        fmhaBmm2Scale = workspaceView.tensor(layout.fmhaBmm2ScaleOffset, layout.fmhaBmm2ScaleSize, at::kFloat);
    }

    return ContextWorkspaceRawViews{
        .trtllmGenWorkspace
        = workspaceView.tensor(layout.trtllmGenWorkspaceOffset, layout.trtllmGenWorkspaceSize, 1, byteOptions),
        .cuQSeqlens = workspaceView.tensor(layout.cuQSeqlensOffset, layout.cuSeqlensSize, sizeof(int32_t), intOptions),
        .cuKvSeqlens
        = workspaceView.tensor(layout.cuKvSeqlensOffset, layout.cuSeqlensSize, sizeof(int32_t), intOptions),
        .qBuf = qBuf,
        .fmhaBmm1Scale = fmhaBmm1Scale,
        .fmhaBmm2Scale = fmhaBmm2Scale,
        .ptrs = makeContextWorkspaceRawPointers(workspaceView, layout),
    };
}

struct GenerationWorkspaceRawViews
{
    at::Tensor trtllmGenWorkspace;
    at::Tensor qBuf;
    at::Tensor bmm1Scale;
    at::Tensor bmm2Scale;
    int* cuSeqlensPtr{};
    int* cuKvSeqlensPtr{};
    float* rotaryInvFreqBufPtr{};
    int2* tokensInfoPtr{};
    void* qBufPtr{};
    float* bmm1ScalePtr{};
    float* bmm2ScalePtr{};
    TrtllmGenGenerationWorkspaceLayout layout{};
};

GenerationWorkspaceRawViews makeGenerationWorkspaceRawViews(
    at::Tensor const& workspace, TrtllmGenGenerationWorkspaceLayout const& layout)
{
    WorkspaceAccessor const workspaceView{workspace};
    auto const byteOptions = workspaceView.options(at::kByte);
    auto const qBufOptions = workspaceView.options(layout.qBufScalarType);
    auto const qBufItemSize = static_cast<int64_t>(c10::elementSize(layout.qBufScalarType));

    return GenerationWorkspaceRawViews{
        .trtllmGenWorkspace
        = workspaceView.tensor(layout.trtllmGenWorkspaceOffset, layout.trtllmGenWorkspaceSize, 1, byteOptions),
        .qBuf = workspaceView.tensor(layout.qBufOffset, layout.qBufSize, qBufItemSize, qBufOptions),
        .bmm1Scale = workspaceView.tensor(layout.bmm1ScaleOffset, layout.bmm1ScaleSize, at::kFloat),
        .bmm2Scale = workspaceView.tensor(layout.bmm2ScaleOffset, layout.bmm2ScaleSize, at::kFloat),
        .cuSeqlensPtr = workspaceView.ptr<int>(layout.cuSeqlensOffset, layout.cuSeqlensSize),
        .cuKvSeqlensPtr = workspaceView.ptr<int>(layout.cuKvSeqlensOffset, layout.cuKvSeqlensSize),
        .rotaryInvFreqBufPtr = workspaceView.ptr<float>(layout.rotaryInvFreqOffset, layout.rotaryInvFreqSize),
        .tokensInfoPtr = workspaceView.ptr<int2>(layout.tokensInfoOffset, layout.tokensInfoSize),
        .qBufPtr = workspaceView.ptr(layout.qBufOffset, layout.qBufSize),
        .bmm1ScalePtr = workspaceView.ptr<float>(layout.bmm1ScaleOffset, layout.bmm1ScaleSize),
        .bmm2ScalePtr = workspaceView.ptr<float>(layout.bmm2ScaleOffset, layout.bmm2ScaleSize),
        .layout = layout,
    };
}

} // anonymous namespace

std::tuple<at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>, std::optional<at::Tensor>,
    std::optional<at::Tensor>, std::optional<at::Tensor>, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t>
trtllmGenContextPreprocess(torch::Tensor qkv_input, torch::Tensor workspace, torch::Tensor sequence_lengths,
    torch::Tensor context_lengths, std::optional<torch::Tensor> kv_cache_block_offsets,
    std::optional<torch::Tensor> host_kv_cache_pool_pointers, std::optional<torch::Tensor> host_kv_cache_pool_mapping,
    std::optional<torch::Tensor> kv_scale_orig_quant, std::optional<torch::Tensor> kv_scale_quant_orig,
    std::optional<torch::Tensor> attention_output_orig_quant, std::optional<torch::Tensor> rotary_inv_freq,
    std::optional<torch::Tensor> rotary_cos_sin, std::optional<torch::Tensor> mrope_rotary_cos_sin,
    int64_t const layer_idx, int64_t const num_heads, int64_t const num_kv_heads, int64_t const head_size,
    int64_t const tokens_per_block, int64_t const mask_type, int64_t const kv_cache_quant_mode,
    int64_t const max_attention_window_size, int64_t const cyclic_attention_window_size, int64_t const num_tokens,
    int64_t const batch_size, int64_t const input_seq_length, int64_t const max_past_kv_length,
    int64_t const rotary_embedding_dim, double const rotary_embedding_base, int64_t const rotary_embedding_scale_type,
    double const rotary_embedding_scale, int64_t const rotary_embedding_max_positions,
    int64_t const position_embedding_type, double const bmm1_scale, double const bmm2_scale,
    int64_t const attention_chunk_size, bool const fp8_context_fmha, bool const paged_context_fmha,
    bool const is_mla_enable, int64_t const multi_processor_count, int64_t const total_num_blocks,
    int64_t const kv_factor, bool const need_build_kv_cache_metadata, std::optional<torch::Tensor> cross_kv,
    bool const cross_attention)
{
    (void) bmm2_scale;
    TORCH_CHECK(host_kv_cache_pool_pointers.has_value(), "host_kv_cache_pool_pointers is required.");
    TORCH_CHECK(host_kv_cache_pool_mapping.has_value(), "host_kv_cache_pool_mapping is required.");
    TORCH_CHECK(kv_cache_block_offsets.has_value(), "kv_cache_block_offsets is required.");
    TORCH_CHECK(!cross_attention || !is_mla_enable, "trtllm-gen cross attention does not support MLA.");

    bool const separateQKvOutput = paged_context_fmha || fp8_context_fmha || cross_attention;
    auto const qkvScalarType = qkv_input.scalar_type();
    auto const qkvElementSize = static_cast<size_t>(qkv_input.element_size());
    auto const quantMode = tensorrt_llm::common::QuantMode(static_cast<uint32_t>(kv_cache_quant_mode));
    int64_t const effectiveMaxAttentionWindowSize = cross_attention ? max_past_kv_length : max_attention_window_size;
    int64_t const effectiveCyclicAttentionWindowSize
        = cross_attention ? max_past_kv_length : cyclic_attention_window_size;
    auto const views = [&]
    {
        auto const layout = TrtllmAttentionWorkspaceManager::buildContextLayout(
            qkvScalarType, batch_size, num_tokens, num_heads, head_size, rotary_embedding_dim, true, fp8_context_fmha);
        return makeContextWorkspaceRawViews(workspace, layout, separateQKvOutput);
    }();
    auto const& ptrs = views.ptrs;
    auto stream = currentStreamFor(qkv_input);

    BuildDecoderInfoParams<float> decoderInfoParams{};
    decoderInfoParams.seqQOffsets = ptrs.cuQSeqlensPtr;
    decoderInfoParams.seqKVOffsets = ptrs.cuKvSeqlensPtr;
    decoderInfoParams.paddingOffsets = nullptr;
    decoderInfoParams.tokensInfo = ptrs.tokensInfoPtr;
    decoderInfoParams.encoderPaddingOffsets = nullptr;
    decoderInfoParams.packedMaskRowOffsets = ptrs.cuMaskRowsPtr;
    decoderInfoParams.seqCpPartialOffsets = nullptr;
    decoderInfoParams.attentionMask = nullptr;
    decoderInfoParams.seqQLengths = static_cast<int*>(context_lengths.data_ptr());
    decoderInfoParams.seqKVLengths = static_cast<int*>(sequence_lengths.data_ptr());
    decoderInfoParams.cpSize = 1;
    decoderInfoParams.fmhaTileCounter = ptrs.fmhaTileCounterPtr;
    decoderInfoParams.dequantScaleQkv = optPtr<float>(kv_scale_quant_orig);
    decoderInfoParams.separateQkvScales = quantMode.hasFp4KvCache();
    decoderInfoParams.quantScaleO = optPtr<float>(attention_output_orig_quant);
    decoderInfoParams.fmhaHostBmm1Scale = static_cast<float>(bmm1_scale);
    decoderInfoParams.fmhaBmm1Scale = ptrs.fmhaBmm1ScalePtr;
    decoderInfoParams.fmhaBmm2Scale = ptrs.fmhaBmm2ScalePtr;
    decoderInfoParams.batchSize = static_cast<int>(batch_size);
    decoderInfoParams.maxQSeqLength = static_cast<int>(input_seq_length);
    decoderInfoParams.maxEncoderQSeqLength = cross_attention ? static_cast<int>(max_past_kv_length) : 0;
    decoderInfoParams.attentionWindowSize = static_cast<int>(effectiveCyclicAttentionWindowSize);
    decoderInfoParams.numTokens = static_cast<int>(num_tokens);
    decoderInfoParams.removePadding = true;
    decoderInfoParams.attentionMaskType = static_cast<AttentionMaskType>(mask_type);
    decoderInfoParams.blockSparseParams = BlockSparseParams{};
    decoderInfoParams.rotaryEmbeddingScale = static_cast<float>(rotary_embedding_scale);
    decoderInfoParams.rotaryEmbeddingBase = static_cast<float>(rotary_embedding_base);
    decoderInfoParams.rotaryEmbeddingDim = static_cast<int>(rotary_embedding_dim);
    decoderInfoParams.rotaryScalingType = static_cast<RotaryScalingType>(rotary_embedding_scale_type);
    decoderInfoParams.rotaryEmbeddingInvFreq = ptrs.rotaryInvFreqBufPtr;
    decoderInfoParams.rotaryEmbeddingInvFreqCache = optPtr<float>(rotary_inv_freq);
    decoderInfoParams.rotaryEmbeddingCoeffCache = nullptr;
    decoderInfoParams.rotaryEmbeddingMaxPositions = static_cast<int>(rotary_embedding_max_positions);
    if (decoderInfoParams.isBuildDecoderInfoKernelNeeded())
    {
        tensorrt_llm::kernels::invokeBuildDecoderInfo(decoderInfoParams, stream);
        sync_check_cuda_error(stream);
    }

    {
        auto const qkvDtype = TorchUtils::dataType(qkvScalarType);
        auto const kvArrays = [&]
        {
            return buildPagedKvCacheBuffers(kv_cache_block_offsets, host_kv_cache_pool_pointers,
                host_kv_cache_pool_mapping, quantMode, layer_idx, batch_size, tokens_per_block, num_kv_heads, head_size,
                effectiveCyclicAttentionWindowSize, effectiveMaxAttentionWindowSize, 0, 0, is_mla_enable,
                qkvElementSize);
        }();

        QKVPreprocessingParams<void, KVBlockArray> qkvParams{};
        qkvParams.qkv_input = qkv_input.data_ptr();
        qkvParams.cross_kv_input = optPtr<void>(cross_kv);
        qkvParams.quantized_qkv_output = nullptr;
        qkvParams.q_output = ptrs.qBufPtr;
        qkvParams.kv_cache_buffer = kvArrays.kvCacheBuffer;
        qkvParams.kv_cache_block_scales_buffer = kvArrays.kvScaleCacheBuffer;
        qkvParams.qkv_bias = nullptr;
        qkvParams.qkv_scale_quant_orig = optPtr<float>(kv_scale_quant_orig);
        qkvParams.qkv_scale_orig_quant = optPtr<float>(kv_scale_orig_quant);
        qkvParams.o_scale_orig_quant = optPtr<float>(attention_output_orig_quant);
        qkvParams.fmha_bmm1_scale = ptrs.fmhaBmm1ScalePtr;
        qkvParams.fmha_bmm2_scale = ptrs.fmhaBmm2ScalePtr;
        qkvParams.fmha_tile_counter = reinterpret_cast<float*>(ptrs.fmhaTileCounterPtr);
        qkvParams.logn_scaling = nullptr;
        qkvParams.tokens_info = ptrs.tokensInfoPtr;
        qkvParams.seq_lens = static_cast<int*>(context_lengths.data_ptr());
        qkvParams.cache_seq_lens = cross_attention ? static_cast<int*>(context_lengths.data_ptr())
                                                   : static_cast<int*>(sequence_lengths.data_ptr());
        qkvParams.encoder_seq_lens = cross_attention ? static_cast<int*>(sequence_lengths.data_ptr()) : nullptr;
        qkvParams.cu_seq_lens = ptrs.cuQSeqlensPtr;
        qkvParams.cu_kv_seq_lens = ptrs.cuKvSeqlensPtr;
        qkvParams.sparse_kv_offsets = nullptr;
        qkvParams.sparse_kv_indices = nullptr;
        qkvParams.rotary_embedding_inv_freq = ptrs.rotaryInvFreqBufPtr;
        qkvParams.rotary_coef_cache_buffer = optPtr<float2>(rotary_cos_sin);
        qkvParams.spec_decoding_position_offsets = nullptr;
        qkvParams.mrope_rotary_cos_sin = optPtr<float2>(mrope_rotary_cos_sin);
        qkvParams.mrope_position_deltas = nullptr;
        qkvParams.batch_size = static_cast<int>(batch_size);
        qkvParams.max_input_seq_len = static_cast<int>(input_seq_length);
        qkvParams.max_kv_seq_len = static_cast<int>(max_past_kv_length);
        qkvParams.cyclic_kv_cache_len = static_cast<int>(effectiveCyclicAttentionWindowSize);
        qkvParams.token_num = static_cast<int>(num_tokens);
        qkvParams.remove_padding = true;
        qkvParams.is_last_chunk = attention_chunk_size == 0 || input_seq_length == max_past_kv_length;
        qkvParams.cross_attention = cross_attention;
        qkvParams.head_num = static_cast<int>(num_heads);
        qkvParams.kv_head_num = static_cast<int>(num_kv_heads);
        qkvParams.qheads_per_kv_head = static_cast<int>(num_heads / num_kv_heads);
        qkvParams.size_per_head = static_cast<int>(head_size);
        qkvParams.fmha_host_bmm1_scale = static_cast<float>(bmm1_scale);
        qkvParams.rotary_embedding_dim = static_cast<int>(rotary_embedding_dim);
        qkvParams.rotary_embedding_base = static_cast<float>(rotary_embedding_base);
        qkvParams.rotary_scale_type = static_cast<RotaryScalingType>(rotary_embedding_scale_type);
        qkvParams.rotary_embedding_scale = static_cast<float>(rotary_embedding_scale);
        qkvParams.rotary_embedding_max_positions = static_cast<int>(rotary_embedding_max_positions);
        qkvParams.position_embedding_type = static_cast<PositionEmbeddingType>(position_embedding_type);
        qkvParams.position_shift_enabled = false;
        qkvParams.cache_type = cacheTypeFromQuantMode(quantMode);
        qkvParams.separate_q_kv_output = separateQKvOutput;
        qkvParams.quantized_fp8_output = fp8_context_fmha;
        qkvParams.generation_phase = false;
        qkvParams.multi_processor_count = static_cast<int>(multi_processor_count);
        qkvParams.rotary_vision_start = 0;
        qkvParams.rotary_vision_length = 0;

        switch (qkvDtype)
        {
        case nvinfer1::DataType::kFLOAT:
            tensorrt_llm::kernels::invokeQKVPreprocessing(
                reinterpret_cast<QKVPreprocessingParams<float, KVBlockArray>&>(qkvParams), stream);
            break;
        case nvinfer1::DataType::kHALF:
            tensorrt_llm::kernels::invokeQKVPreprocessing(
                reinterpret_cast<QKVPreprocessingParams<half, KVBlockArray>&>(qkvParams), stream);
            break;
#ifdef ENABLE_BF16
        case nvinfer1::DataType::kBF16:
            tensorrt_llm::kernels::invokeQKVPreprocessing(
                reinterpret_cast<QKVPreprocessingParams<__nv_bfloat16, KVBlockArray>&>(qkvParams), stream);
            break;
#endif
        default: TORCH_CHECK(false, "Unsupported data type for QKV preprocessing.");
        }
        sync_check_cuda_error(stream);
    }

    std::optional<at::Tensor> kvPool;
    std::optional<at::Tensor> kvScalePool;
    std::optional<at::Tensor> blockTables;
    if (need_build_kv_cache_metadata)
    {
        std::tie(kvPool, kvScalePool) = buildFlashinferTrtllmGenPagedKvCacheBuffers(host_kv_cache_pool_pointers.value(),
            host_kv_cache_pool_mapping.value(), layer_idx, num_kv_heads, tokens_per_block, head_size, kv_factor,
            total_num_blocks, kv_cache_quant_mode, qkvScalarType);

        int32_t const poolIndex = readKvCachePoolMapping(host_kv_cache_pool_mapping.value(), layer_idx).poolIndex;
        blockTables = kv_cache_block_offsets.value().select(0, poolIndex).narrow(0, 0, batch_size);
    }

    at::Tensor qProcessed;
    if (separateQKvOutput)
    {
        qProcessed = views.qBuf.view({num_tokens, num_heads, head_size});
    }
    else
    {
        qProcessed = qkv_input.slice(1, 0, num_heads * head_size).view({num_tokens, num_heads, head_size});
    }

    // FlashInfer paged context launches trtllm-gen with multi-CTA-KV mode disabled, so it does not
    // consume the counter slab reserved at the head of the workspace.
    auto const windowLeft = cross_attention
        ? int64_t{-1}
        : computeWindowLeft(cyclic_attention_window_size, max_past_kv_length, attention_chunk_size);
    return {qProcessed, kvPool, blockTables, kvScalePool, views.fmhaBmm1Scale, views.fmhaBmm2Scale,
        views.trtllmGenWorkspace, views.cuQSeqlens, views.cuKvSeqlens, input_seq_length, max_past_kv_length,
        windowLeft};
}

void trtllmGenContextPostprocess(torch::Tensor qkv_input, torch::Tensor workspace, torch::Tensor sequence_lengths,
    torch::Tensor context_lengths, std::optional<torch::Tensor> kv_cache_block_offsets,
    std::optional<torch::Tensor> host_kv_cache_pool_pointers, std::optional<torch::Tensor> host_kv_cache_pool_mapping,
    std::optional<torch::Tensor> kv_scale_orig_quant, std::optional<torch::Tensor> kv_scale_quant_orig,
    std::optional<torch::Tensor> attention_output_orig_quant, std::optional<torch::Tensor> rotary_cos_sin,
    std::optional<torch::Tensor> mrope_rotary_cos_sin, int64_t const layer_idx, int64_t const num_heads,
    int64_t const num_kv_heads, int64_t const head_size, int64_t const tokens_per_block, int64_t const mask_type,
    int64_t const kv_cache_quant_mode, int64_t const max_attention_window_size,
    int64_t const cyclic_attention_window_size, int64_t const num_tokens, int64_t const batch_size,
    int64_t const input_seq_length, int64_t const max_past_kv_length, int64_t const rotary_embedding_dim,
    double const rotary_embedding_base, int64_t const rotary_embedding_scale_type, double const rotary_embedding_scale,
    int64_t const rotary_embedding_max_positions, int64_t const position_embedding_type, double const bmm1_scale,
    bool const fp8_context_fmha, bool const paged_context_fmha, bool const is_mla_enable,
    int64_t const attention_chunk_size, int64_t const multi_processor_count)
{
    (void) mask_type;
    auto const qkvScalarType = qkv_input.scalar_type();
    auto const qkvElementSize = static_cast<size_t>(qkv_input.element_size());
    auto const quantMode = tensorrt_llm::common::QuantMode(static_cast<uint32_t>(kv_cache_quant_mode));
    bool const separateQKvOutput = paged_context_fmha || fp8_context_fmha;
    auto const ptrs = [&]
    {
        auto const layout = TrtllmAttentionWorkspaceManager::buildContextLayout(
            qkvScalarType, batch_size, num_tokens, num_heads, head_size, rotary_embedding_dim, true, fp8_context_fmha);
        WorkspaceAccessor const workspaceView{workspace};
        return makeContextWorkspaceRawPointers(workspaceView, layout);
    }();

    {
        auto stream = currentStreamFor(qkv_input);
        auto const qkvDtype = TorchUtils::dataType(qkvScalarType);
        auto const kvArrays = [&]
        {
            return buildPagedKvCacheBuffers(kv_cache_block_offsets, host_kv_cache_pool_pointers,
                host_kv_cache_pool_mapping, quantMode, layer_idx, batch_size, tokens_per_block, num_kv_heads, head_size,
                cyclic_attention_window_size, max_attention_window_size, 0, 0, is_mla_enable, qkvElementSize);
        }();

        QKVPreprocessingParams<void, KVBlockArray> qkvParams{};
        qkvParams.qkv_input = qkv_input.data_ptr();
        qkvParams.cross_kv_input = nullptr;
        qkvParams.quantized_qkv_output = nullptr;
        qkvParams.q_output = ptrs.qBufPtr;
        qkvParams.kv_cache_buffer = kvArrays.kvCacheBuffer;
        qkvParams.kv_cache_block_scales_buffer = kvArrays.kvScaleCacheBuffer;
        qkvParams.qkv_bias = nullptr;
        qkvParams.qkv_scale_quant_orig = optPtr<float>(kv_scale_quant_orig);
        qkvParams.qkv_scale_orig_quant = optPtr<float>(kv_scale_orig_quant);
        qkvParams.o_scale_orig_quant = optPtr<float>(attention_output_orig_quant);
        qkvParams.fmha_bmm1_scale = ptrs.fmhaBmm1ScalePtr;
        qkvParams.fmha_bmm2_scale = ptrs.fmhaBmm2ScalePtr;
        qkvParams.fmha_tile_counter = reinterpret_cast<float*>(ptrs.fmhaTileCounterPtr);
        qkvParams.logn_scaling = nullptr;
        qkvParams.tokens_info = ptrs.tokensInfoPtr;
        qkvParams.seq_lens = static_cast<int*>(context_lengths.data_ptr());
        qkvParams.cache_seq_lens = static_cast<int*>(sequence_lengths.data_ptr());
        qkvParams.encoder_seq_lens = nullptr;
        qkvParams.cu_seq_lens = ptrs.cuQSeqlensPtr;
        qkvParams.cu_kv_seq_lens = ptrs.cuKvSeqlensPtr;
        qkvParams.sparse_kv_offsets = nullptr;
        qkvParams.sparse_kv_indices = nullptr;
        qkvParams.rotary_embedding_inv_freq = ptrs.rotaryInvFreqBufPtr;
        qkvParams.rotary_coef_cache_buffer = optPtr<float2>(rotary_cos_sin);
        qkvParams.spec_decoding_position_offsets = nullptr;
        qkvParams.mrope_rotary_cos_sin = optPtr<float2>(mrope_rotary_cos_sin);
        qkvParams.mrope_position_deltas = nullptr;
        qkvParams.batch_size = static_cast<int>(batch_size);
        qkvParams.max_input_seq_len = static_cast<int>(input_seq_length);
        qkvParams.max_kv_seq_len = static_cast<int>(max_past_kv_length);
        qkvParams.cyclic_kv_cache_len = static_cast<int>(cyclic_attention_window_size);
        qkvParams.token_num = static_cast<int>(num_tokens);
        qkvParams.remove_padding = true;
        qkvParams.is_last_chunk = attention_chunk_size == 0 || input_seq_length == max_past_kv_length;
        qkvParams.cross_attention = false;
        qkvParams.head_num = static_cast<int>(num_heads);
        qkvParams.kv_head_num = static_cast<int>(num_kv_heads);
        qkvParams.qheads_per_kv_head = static_cast<int>(num_heads / num_kv_heads);
        qkvParams.size_per_head = static_cast<int>(head_size);
        qkvParams.fmha_host_bmm1_scale = static_cast<float>(bmm1_scale);
        qkvParams.rotary_embedding_dim = static_cast<int>(rotary_embedding_dim);
        qkvParams.rotary_embedding_base = static_cast<float>(rotary_embedding_base);
        qkvParams.rotary_scale_type = static_cast<RotaryScalingType>(rotary_embedding_scale_type);
        qkvParams.rotary_embedding_scale = static_cast<float>(rotary_embedding_scale);
        qkvParams.rotary_embedding_max_positions = static_cast<int>(rotary_embedding_max_positions);
        qkvParams.position_embedding_type = static_cast<PositionEmbeddingType>(position_embedding_type);
        qkvParams.position_shift_enabled = false;
        qkvParams.cache_type = cacheTypeFromQuantMode(quantMode);
        qkvParams.separate_q_kv_output = separateQKvOutput;
        qkvParams.quantized_fp8_output = fp8_context_fmha;
        qkvParams.generation_phase = false;
        qkvParams.multi_processor_count = static_cast<int>(multi_processor_count);
        qkvParams.rotary_vision_start = 0;
        qkvParams.rotary_vision_length = 0;

        switch (qkvDtype)
        {
        case nvinfer1::DataType::kFLOAT:
            tensorrt_llm::kernels::invokeKvCachePostprocessing(
                reinterpret_cast<QKVPreprocessingParams<float, KVBlockArray>&>(qkvParams), stream);
            break;
        case nvinfer1::DataType::kHALF:
            tensorrt_llm::kernels::invokeKvCachePostprocessing(
                reinterpret_cast<QKVPreprocessingParams<half, KVBlockArray>&>(qkvParams), stream);
            break;
#ifdef ENABLE_BF16
        case nvinfer1::DataType::kBF16:
            tensorrt_llm::kernels::invokeKvCachePostprocessing(
                reinterpret_cast<QKVPreprocessingParams<__nv_bfloat16, KVBlockArray>&>(qkvParams), stream);
            break;
#endif
        default: TORCH_CHECK(false, "Unsupported data type for KV cache postprocessing.");
        }
        sync_check_cuda_error(stream);
    }
}

std::tuple<at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>, std::optional<at::Tensor>, at::Tensor,
    at::Tensor, at::Tensor, std::optional<at::Tensor>, int64_t, int64_t, int64_t, bool>
trtllmGenGenerationPreprocess(torch::Tensor qkv_input, torch::Tensor workspace, torch::Tensor sequence_lengths,
    std::optional<torch::Tensor> spec_decoding_generation_lengths,
    std::optional<torch::Tensor> spec_decoding_position_offsets, std::optional<torch::Tensor> kv_cache_block_offsets,
    std::optional<torch::Tensor> host_kv_cache_pool_pointers, std::optional<torch::Tensor> host_kv_cache_pool_mapping,
    std::optional<torch::Tensor> kv_scale_orig_quant, std::optional<torch::Tensor> kv_scale_quant_orig,
    std::optional<torch::Tensor> attention_output_orig_quant, std::optional<torch::Tensor> rotary_inv_freq,
    std::optional<torch::Tensor> rotary_cos_sin, std::optional<torch::Tensor> mrope_position_deltas,
    int64_t const layer_idx, int64_t const seq_offset, int64_t const num_heads, int64_t const num_kv_heads,
    int64_t const head_size, int64_t const tokens_per_block, int64_t const kv_cache_quant_mode,
    int64_t const max_attention_window_size, int64_t const cyclic_attention_window_size, int64_t const num_tokens,
    int64_t const batch_beam, int64_t const input_seq_length, int64_t const max_past_kv_length,
    int64_t const rotary_embedding_dim, double const rotary_embedding_base, int64_t const rotary_embedding_scale_type,
    double const rotary_embedding_scale, int64_t const rotary_embedding_max_positions,
    int64_t const position_embedding_type, double const bmm1_scale, double const bmm2_scale,
    bool const fp8_context_fmha, int64_t const predicted_tokens_per_seq, int64_t const attention_chunk_size,
    int64_t const multi_processor_count, int64_t const total_num_blocks, int64_t const kv_factor,
    bool const need_build_kv_cache_metadata, bool const cross_attention)
{
    TORCH_CHECK(host_kv_cache_pool_pointers.has_value(), "host_kv_cache_pool_pointers is required.");
    TORCH_CHECK(host_kv_cache_pool_mapping.has_value(), "host_kv_cache_pool_mapping is required.");
    TORCH_CHECK(kv_cache_block_offsets.has_value(), "kv_cache_block_offsets is required.");
    (void) bmm2_scale;

    bool const isMultiTokenGen = spec_decoding_generation_lengths.has_value() && predicted_tokens_per_seq > 1;
    TORCH_CHECK(
        !cross_attention || !isMultiTokenGen, "trtllm-gen cross attention does not support multi-token generation.");
    auto const qkvScalarType = qkv_input.scalar_type();
    auto const qkvElementSize = static_cast<size_t>(qkv_input.element_size());
    auto const quantMode = tensorrt_llm::common::QuantMode(static_cast<uint32_t>(kv_cache_quant_mode));
    int64_t const effectiveMaxAttentionWindowSize = cross_attention ? max_past_kv_length : max_attention_window_size;
    int64_t const effectiveCyclicAttentionWindowSize
        = cross_attention ? max_past_kv_length : cyclic_attention_window_size;
    auto const views = [&]
    {
        auto const layout = TrtllmAttentionWorkspaceManager::buildGenerationLayout(
            qkvScalarType, batch_beam, num_tokens, num_heads, head_size, rotary_embedding_dim, num_kv_heads, 0, false);
        return makeGenerationWorkspaceRawViews(workspace, layout);
    }();

    auto stream = currentStreamFor(qkv_input);
    BuildDecoderInfoParams<float> decoderInfoParams{};
    decoderInfoParams.seqQOffsets = views.cuSeqlensPtr;
    decoderInfoParams.seqKVOffsets = views.cuKvSeqlensPtr;
    decoderInfoParams.paddingOffsets = nullptr;
    decoderInfoParams.tokensInfo = views.tokensInfoPtr;
    decoderInfoParams.encoderPaddingOffsets = nullptr;
    decoderInfoParams.packedMaskRowOffsets = nullptr;
    decoderInfoParams.seqCpPartialOffsets = nullptr;
    decoderInfoParams.attentionMask = nullptr;
    decoderInfoParams.seqQLengths = isMultiTokenGen ? optPtr<int>(spec_decoding_generation_lengths) : nullptr;
    decoderInfoParams.seqKVLengths = static_cast<int*>(sequence_lengths.data_ptr());
    decoderInfoParams.cpSize = 1;
    decoderInfoParams.fmhaTileCounter = nullptr;
    decoderInfoParams.dequantScaleQkv = nullptr;
    decoderInfoParams.separateQkvScales = quantMode.hasFp4KvCache();
    decoderInfoParams.quantScaleO = nullptr;
    decoderInfoParams.fmhaHostBmm1Scale = static_cast<float>(bmm1_scale);
    decoderInfoParams.fmhaBmm1Scale = nullptr;
    decoderInfoParams.fmhaBmm2Scale = nullptr;
    decoderInfoParams.batchSize = static_cast<int>(batch_beam);
    decoderInfoParams.maxQSeqLength = static_cast<int>(input_seq_length);
    decoderInfoParams.maxEncoderQSeqLength = cross_attention ? static_cast<int>(max_past_kv_length) : 0;
    decoderInfoParams.attentionWindowSize = 0;
    decoderInfoParams.sinkTokenLength = 0;
    decoderInfoParams.numTokens = static_cast<int>(num_tokens);
    decoderInfoParams.removePadding = true;
    decoderInfoParams.attentionMaskType = static_cast<AttentionMaskType>(0);
    decoderInfoParams.blockSparseParams = BlockSparseParams{};
    decoderInfoParams.rotaryEmbeddingScale = static_cast<float>(rotary_embedding_scale);
    decoderInfoParams.rotaryEmbeddingBase = static_cast<float>(rotary_embedding_base);
    decoderInfoParams.rotaryEmbeddingDim = static_cast<int>(rotary_embedding_dim);
    decoderInfoParams.rotaryScalingType = static_cast<RotaryScalingType>(rotary_embedding_scale_type);
    decoderInfoParams.rotaryEmbeddingInvFreq = views.rotaryInvFreqBufPtr;
    decoderInfoParams.rotaryEmbeddingInvFreqCache = optPtr<float>(rotary_inv_freq);
    decoderInfoParams.rotaryEmbeddingCoeffCache = nullptr;
    decoderInfoParams.rotaryEmbeddingMaxPositions = static_cast<int>(rotary_embedding_max_positions);
    bool const buildDecoderInfoNeeded = decoderInfoParams.isBuildDecoderInfoKernelNeeded();
    if (buildDecoderInfoNeeded)
    {
        tensorrt_llm::kernels::invokeBuildDecoderInfo(decoderInfoParams, stream);
        sync_check_cuda_error(stream);
    }

    auto* rotaryInvFreqBuf = buildDecoderInfoNeeded ? views.rotaryInvFreqBufPtr : optPtr<float>(rotary_inv_freq);
    std::optional<at::Tensor> cuSeqlens;
    if (buildDecoderInfoNeeded)
    {
        WorkspaceAccessor const workspaceView{workspace};
        auto const intOptions = workspaceView.options(at::kInt);
        cuSeqlens = workspaceView.tensor(
            views.layout.cuSeqlensOffset, views.layout.cuSeqlensSize, sizeof(int32_t), intOptions);
    }

    {
        auto const qkvDtype = TorchUtils::dataType(qkvScalarType);
        auto const kvArrays = [&]
        {
            return buildPagedKvCacheBuffers(kv_cache_block_offsets, host_kv_cache_pool_pointers,
                host_kv_cache_pool_mapping, quantMode, layer_idx, batch_beam, tokens_per_block, num_kv_heads, head_size,
                effectiveCyclicAttentionWindowSize, effectiveMaxAttentionWindowSize, 1, seq_offset, false,
                qkvElementSize);
        }();

        QKVPreprocessingParams<void, KVBlockArray> qkvParams{};
        qkvParams.qkv_input = qkv_input.data_ptr();
        qkvParams.cross_kv_input = nullptr;
        qkvParams.quantized_qkv_output = nullptr;
        qkvParams.q_output = views.qBufPtr;
        qkvParams.kv_cache_buffer = kvArrays.kvCacheBuffer;
        qkvParams.kv_cache_block_scales_buffer = kvArrays.kvScaleCacheBuffer;
        qkvParams.qkv_bias = nullptr;
        qkvParams.qkv_scale_quant_orig = optPtr<float>(kv_scale_quant_orig);
        qkvParams.qkv_scale_orig_quant = optPtr<float>(kv_scale_orig_quant);
        qkvParams.o_scale_orig_quant = optPtr<float>(attention_output_orig_quant);
        qkvParams.fmha_bmm1_scale = views.bmm1ScalePtr;
        qkvParams.fmha_bmm2_scale = views.bmm2ScalePtr;
        qkvParams.fmha_tile_counter = nullptr;
        qkvParams.logn_scaling = nullptr;
        qkvParams.tokens_info = isMultiTokenGen ? views.tokensInfoPtr : nullptr;
        qkvParams.seq_lens = isMultiTokenGen ? optPtr<int>(spec_decoding_generation_lengths) : nullptr;
        qkvParams.cache_seq_lens = static_cast<int*>(sequence_lengths.data_ptr());
        qkvParams.encoder_seq_lens = cross_attention ? static_cast<int*>(sequence_lengths.data_ptr()) : nullptr;
        qkvParams.cu_seq_lens = buildDecoderInfoNeeded ? views.cuSeqlensPtr : nullptr;
        qkvParams.cu_kv_seq_lens = buildDecoderInfoNeeded ? views.cuKvSeqlensPtr : nullptr;
        qkvParams.sparse_kv_offsets = nullptr;
        qkvParams.sparse_kv_indices = nullptr;
        qkvParams.rotary_embedding_inv_freq = rotaryInvFreqBuf;
        qkvParams.rotary_coef_cache_buffer = optPtr<float2>(rotary_cos_sin);
        qkvParams.spec_decoding_position_offsets
            = isMultiTokenGen ? optPtr<int>(spec_decoding_position_offsets) : nullptr;
        qkvParams.mrope_rotary_cos_sin = nullptr;
        qkvParams.mrope_position_deltas = optPtr<int>(mrope_position_deltas);
        qkvParams.batch_size = static_cast<int>(batch_beam);
        qkvParams.max_input_seq_len = static_cast<int>(input_seq_length);
        qkvParams.max_kv_seq_len = static_cast<int>(max_past_kv_length);
        qkvParams.cyclic_kv_cache_len = static_cast<int>(effectiveCyclicAttentionWindowSize);
        qkvParams.token_num = static_cast<int>(num_tokens);
        qkvParams.remove_padding = true;
        qkvParams.is_last_chunk = false;
        qkvParams.cross_attention = cross_attention;
        qkvParams.head_num = static_cast<int>(num_heads);
        qkvParams.kv_head_num = static_cast<int>(num_kv_heads);
        qkvParams.qheads_per_kv_head = static_cast<int>(num_heads / num_kv_heads);
        qkvParams.size_per_head = static_cast<int>(head_size);
        qkvParams.fmha_host_bmm1_scale = static_cast<float>(bmm1_scale);
        qkvParams.rotary_embedding_dim = static_cast<int>(rotary_embedding_dim);
        qkvParams.rotary_embedding_base = static_cast<float>(rotary_embedding_base);
        qkvParams.rotary_scale_type = static_cast<RotaryScalingType>(rotary_embedding_scale_type);
        qkvParams.rotary_embedding_scale = static_cast<float>(rotary_embedding_scale);
        qkvParams.rotary_embedding_max_positions = static_cast<int>(rotary_embedding_max_positions);
        qkvParams.position_embedding_type = static_cast<PositionEmbeddingType>(position_embedding_type);
        qkvParams.position_shift_enabled = false;
        qkvParams.cache_type = cacheTypeFromQuantMode(quantMode);
        qkvParams.separate_q_kv_output = true;
        qkvParams.quantized_fp8_output = fp8_context_fmha;
        qkvParams.generation_phase = true;
        qkvParams.multi_processor_count = static_cast<int>(multi_processor_count);
        qkvParams.rotary_vision_start = 0;
        qkvParams.rotary_vision_length = 0;

        switch (qkvDtype)
        {
        case nvinfer1::DataType::kFLOAT:
            tensorrt_llm::kernels::invokeQKVPreprocessing(
                reinterpret_cast<QKVPreprocessingParams<float, KVBlockArray>&>(qkvParams), stream);
            break;
        case nvinfer1::DataType::kHALF:
            tensorrt_llm::kernels::invokeQKVPreprocessing(
                reinterpret_cast<QKVPreprocessingParams<half, KVBlockArray>&>(qkvParams), stream);
            break;
#ifdef ENABLE_BF16
        case nvinfer1::DataType::kBF16:
            tensorrt_llm::kernels::invokeQKVPreprocessing(
                reinterpret_cast<QKVPreprocessingParams<__nv_bfloat16, KVBlockArray>&>(qkvParams), stream);
            break;
#endif
        default: TORCH_CHECK(false, "Unsupported data type for QKV preprocessing.");
        }
        sync_check_cuda_error(stream);
    }

    std::optional<at::Tensor> kvPool;
    std::optional<at::Tensor> kvScalePool;
    std::optional<at::Tensor> blockTables;
    if (need_build_kv_cache_metadata)
    {
        std::tie(kvPool, kvScalePool) = buildFlashinferTrtllmGenPagedKvCacheBuffers(host_kv_cache_pool_pointers.value(),
            host_kv_cache_pool_mapping.value(), layer_idx, num_kv_heads, tokens_per_block, head_size, kv_factor,
            total_num_blocks, kv_cache_quant_mode, qkvScalarType);

        int32_t const poolIndex = readKvCachePoolMapping(host_kv_cache_pool_mapping.value(), layer_idx).poolIndex;
        blockTables = kv_cache_block_offsets.value().select(0, poolIndex).narrow(0, seq_offset, batch_beam);
    }

    auto qProcessed = views.qBuf.view({num_tokens, num_heads, head_size});

    auto const windowLeft = cross_attention
        ? int64_t{-1}
        : computeWindowLeft(cyclic_attention_window_size, max_past_kv_length, attention_chunk_size);
    return {qProcessed, kvPool, blockTables, kvScalePool, views.bmm1Scale, views.bmm2Scale, views.trtllmGenWorkspace,
        cuSeqlens, input_seq_length, max_past_kv_length, windowLeft, isMultiTokenGen};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
