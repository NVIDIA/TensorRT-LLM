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

#include "tensorrt_llm/thop/attentionOp.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/attentionWorkspace.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/sageQuant.h"
#include "tensorrt_llm/common/tllmDataType.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/cascadeAttentionKernel.h"
#include "tensorrt_llm/kernels/flashMLA/flash_mla.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/mlaKernels.h"
#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"
#include "tensorrt_llm/kernels/sparseAttentionKernels.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/debugUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <torch/extension.h>
#include <type_traits>

using namespace tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
using tensorrt_llm::common::op::AttentionContextWorkspaceSizes;
using tensorrt_llm::common::op::AttentionFlashMlaWorkspaceSizes;
using tensorrt_llm::common::op::AttentionGenerationWorkspaceSizes;
using tensorrt_llm::common::op::AttentionWorkspaceManager;
using tensorrt_llm::common::op::AttentionXqaWorkspaceSizes;

namespace
{

int64_t exportOffset(tensorrt_llm::common::op::WorkspaceSlice const& slice)
{
    if (slice.size == 0)
    {
        return -1;
    }
    return static_cast<int64_t>(slice.offset);
}

template <typename T>
T readHostTensor2D(at::Tensor const& tensor, int64_t const row, int64_t const col, char const* tensorName)
{
    TORCH_CHECK(tensor.device().is_cpu(), tensorName, " must be a CPU tensor.");
    TORCH_CHECK(tensor.dim() == 2, tensorName, " must be a 2D tensor.");
    TORCH_CHECK(row >= 0 && row < tensor.size(0), tensorName, " row is out of bounds.");
    TORCH_CHECK(col >= 0 && col < tensor.size(1), tensorName, " column is out of bounds.");

    auto const* data = tensor.data_ptr<T>();
    return data[row * tensor.stride(0) + col * tensor.stride(1)];
}

template <typename T>
T readHostTensor3D(at::Tensor const& tensor, int64_t const i, int64_t const j, int64_t const k, char const* tensorName)
{
    TORCH_CHECK(tensor.device().is_cpu(), tensorName, " must be a CPU tensor.");
    TORCH_CHECK(tensor.dim() == 3, tensorName, " must be a 3D tensor.");
    TORCH_CHECK(i >= 0 && i < tensor.size(0), tensorName, " dim 0 index is out of bounds.");
    TORCH_CHECK(j >= 0 && j < tensor.size(1), tensorName, " dim 1 index is out of bounds.");
    TORCH_CHECK(k >= 0 && k < tensor.size(2), tensorName, " dim 2 index is out of bounds.");

    auto const* data = tensor.data_ptr<T>();
    return data[i * tensor.stride(0) + j * tensor.stride(1) + k * tensor.stride(2)];
}

template <typename T>
T* tensorPtr2D(at::Tensor const& tensor, int64_t const row, int64_t const col, char const* tensorName)
{
    TORCH_CHECK(tensor.dim() >= 2, tensorName, " must have at least 2 dimensions.");
    TORCH_CHECK(row >= 0 && row < tensor.size(0), tensorName, " row is out of bounds.");
    TORCH_CHECK(col >= 0 && col < tensor.size(1), tensorName, " column is out of bounds.");

    using ValueType = std::remove_const_t<T>;
    auto* data = static_cast<ValueType*>(tensor.data_ptr());
    return data + row * tensor.stride(0) + col * tensor.stride(1);
}

} // namespace

KvCachePoolMapping readKvCachePoolMapping(at::Tensor const& hostKvCachePoolMapping, int64_t const layerIdx)
{
    TORCH_CHECK(hostKvCachePoolMapping.device().is_cpu(), "host_kv_cache_pool_mapping must be a CPU tensor.");
    TORCH_CHECK(hostKvCachePoolMapping.dim() == 2, "host_kv_cache_pool_mapping must be a 2D tensor.");
    TORCH_CHECK(hostKvCachePoolMapping.size(1) >= 2, "host_kv_cache_pool_mapping must have at least two columns.");
    TORCH_CHECK(layerIdx >= 0 && layerIdx < hostKvCachePoolMapping.size(0),
        "host_kv_cache_pool_mapping layer index is out of bounds.");

    auto const* data = hostKvCachePoolMapping.data_ptr<int32_t>();
    auto const rowOffset = layerIdx * hostKvCachePoolMapping.stride(0);
    auto const colStride = hostKvCachePoolMapping.stride(1);
    KvCachePoolMapping mapping;
    mapping.poolIndex = data[rowOffset];
    mapping.layerIdxInCachePool = data[rowOffset + colStride];
    return mapping;
}

std::optional<at::Tensor> TrtllmAttentionWorkspaceManager::makeWorkspaceView(
    at::Tensor const& workspace, int64_t const offset, int64_t const sizeBytes, at::ScalarType const scalarType)
{
    if (sizeBytes == 0)
    {
        return std::nullopt;
    }

    auto const* workspaceBase = static_cast<uint8_t const*>(workspace.data_ptr());
    auto const workspaceSizeBytes = static_cast<int64_t>(workspace.nbytes());
    TORCH_CHECK(offset >= 0, "Negative workspace offset is invalid.");
    TORCH_CHECK(offset + sizeBytes <= workspaceSizeBytes, "Workspace view exceeds workspace bounds.");

    auto const itemSize = static_cast<int64_t>(c10::elementSize(scalarType));
    TORCH_CHECK(sizeBytes % itemSize == 0, "Workspace slice is not aligned to dtype size.");

    auto options = at::TensorOptions().dtype(scalarType).device(workspace.device());
    return torch::from_blob(const_cast<uint8_t*>(workspaceBase) + offset, {sizeBytes / itemSize}, options);
}

TrtllmGenContextWorkspaceLayout TrtllmAttentionWorkspaceManager::buildContextLayout(at::ScalarType const qDtype,
    int64_t const batchSize, int64_t const numTokens, int64_t const numHeads, int64_t const headSize,
    int64_t const rotaryEmbeddingDim, bool const separateQKvInput, bool const fp8ContextFmha)
{
    auto const dtypeSize = static_cast<int64_t>(c10::elementSize(qDtype));
    auto const localHiddenUnitsQo = numHeads * headSize;
    auto const cuSeqlensSize = static_cast<int64_t>(sizeof(int32_t)) * (batchSize + 1);
    auto const rotaryInvFreqSize
        = rotaryEmbeddingDim > 0 ? static_cast<int64_t>(sizeof(float)) * batchSize * rotaryEmbeddingDim / 2 : 0;
    auto const qBufSize = separateQKvInput ? (fp8ContextFmha ? 1 : dtypeSize) * numTokens * localHiddenUnitsQo : 0;
    auto const tokensInfoSize = static_cast<int64_t>(sizeof(int32_t) * 2) * numTokens;
    auto const fmhaTileCounterSize = static_cast<int64_t>(sizeof(uint32_t));
    auto const fmhaBmm1ScaleSize = fp8ContextFmha ? static_cast<int64_t>(sizeof(float) * 2) : 0;
    auto const fmhaBmm2ScaleSize = fp8ContextFmha ? static_cast<int64_t>(sizeof(float)) : 0;

    tensorrt_llm::common::op::AttentionContextWorkspaceSizes workspaceSizes{};
    workspaceSizes.cuQSeqlens = cuSeqlensSize;
    workspaceSizes.cuKvSeqlens = cuSeqlensSize;
    workspaceSizes.cuMaskRows = cuSeqlensSize;
    workspaceSizes.rotaryInvFreq = rotaryInvFreqSize;
    workspaceSizes.qBuf = qBufSize;
    workspaceSizes.tokensInfo = tokensInfoSize;
    workspaceSizes.fmhaTileCounter = fmhaTileCounterSize;
    workspaceSizes.fmhaBmm1Scale = fmhaBmm1ScaleSize;
    workspaceSizes.fmhaBmm2Scale = fmhaBmm2ScaleSize;
    auto const layout = AttentionWorkspaceManager::buildContextLayout(workspaceSizes, kWorkspaceAlignment);

    return TrtllmGenContextWorkspaceLayout{
        .trtllmGenWorkspaceOffset = exportOffset(layout.cublasWorkspace),
        .cuQSeqlensOffset = exportOffset(layout.cuQSeqlens),
        .cuKvSeqlensOffset = exportOffset(layout.cuKvSeqlens),
        .cuMaskRowsOffset = exportOffset(layout.cuMaskRows),
        .rotaryInvFreqOffset = exportOffset(layout.rotaryInvFreq),
        .qBufOffset = exportOffset(layout.qBuf),
        .tokensInfoOffset = exportOffset(layout.tokensInfo),
        .fmhaTileCounterOffset = exportOffset(layout.fmhaTileCounter),
        .fmhaBmm1ScaleOffset = exportOffset(layout.fmhaBmm1Scale),
        .fmhaBmm2ScaleOffset = exportOffset(layout.fmhaBmm2Scale),
        .trtllmGenWorkspaceSize = kTrtllmGenWorkspaceSize,
        .cuSeqlensSize = cuSeqlensSize,
        .rotaryInvFreqSize = rotaryInvFreqSize,
        .qBufSize = qBufSize,
        .tokensInfoSize = tokensInfoSize,
        .fmhaTileCounterSize = fmhaTileCounterSize,
        .fmhaBmm1ScaleSize = fmhaBmm1ScaleSize,
        .fmhaBmm2ScaleSize = fmhaBmm2ScaleSize,
        .totalSize = static_cast<int64_t>(layout.totalSize),
        .qBufScalarType = fp8ContextFmha ? at::kByte : qDtype,
    };
}

TrtllmGenGenerationWorkspaceLayout TrtllmAttentionWorkspaceManager::buildGenerationLayout(at::ScalarType const qDtype,
    int64_t const batchBeam, int64_t const numTokens, int64_t const numHeads, int64_t const headSize,
    int64_t const rotaryEmbeddingDim, int64_t const numKvHeads, int64_t const maxBlocksPerSequence,
    bool const useSparseAttention)
{
    auto const dtypeSize = static_cast<int64_t>(c10::elementSize(qDtype));
    auto const cuSeqlensSize = static_cast<int64_t>(sizeof(int32_t)) * (batchBeam + 1);
    auto const cuKvSeqlensSize = static_cast<int64_t>(sizeof(int32_t)) * (batchBeam + 1);
    auto const rotaryInvFreqSize
        = rotaryEmbeddingDim > 0 ? static_cast<int64_t>(sizeof(float)) * batchBeam * rotaryEmbeddingDim / 2 : 0;
    auto const tokensInfoSize = static_cast<int64_t>(sizeof(int32_t) * 2) * numTokens;
    auto const qBufSize = dtypeSize * numTokens * numHeads * headSize;
    auto const bmm1ScaleSize = static_cast<int64_t>(sizeof(float) * 2);
    auto const bmm2ScaleSize = static_cast<int64_t>(sizeof(float));
    auto const sparseAttnCacheSize = useSparseAttention
        ? static_cast<int64_t>(sizeof(int32_t)) * (batchBeam + batchBeam * 2 * maxBlocksPerSequence) * numKvHeads
        : 0;

    tensorrt_llm::common::op::AttentionXqaWorkspaceSizes workspaceSizes{};
    workspaceSizes.cuSeqlens = cuSeqlensSize;
    workspaceSizes.cuKvSeqlens = cuKvSeqlensSize;
    workspaceSizes.rotaryInvFreq = rotaryInvFreqSize;
    workspaceSizes.tokensInfo = tokensInfoSize;
    workspaceSizes.bmm1Scale = bmm1ScaleSize;
    workspaceSizes.bmm2Scale = bmm2ScaleSize;
    workspaceSizes.sparseAttnCache = sparseAttnCacheSize;
    workspaceSizes.kernelWorkspace = qBufSize;
    auto const xqaLayout = AttentionWorkspaceManager::buildXqaLayout(workspaceSizes, kWorkspaceAlignment);
    auto const trtllmGenWorkspaceOffset = static_cast<int64_t>(xqaLayout.totalSize);
    auto const totalSize = xqaLayout.totalSize
        + tensorrt_llm::common::alignSize(static_cast<size_t>(kTrtllmGenWorkspaceSize), kWorkspaceAlignment);

    return TrtllmGenGenerationWorkspaceLayout{
        .trtllmGenWorkspaceOffset = trtllmGenWorkspaceOffset,
        .cuSeqlensOffset = exportOffset(xqaLayout.cuSeqlens),
        .cuKvSeqlensOffset = exportOffset(xqaLayout.cuKvSeqlens),
        .rotaryInvFreqOffset = exportOffset(xqaLayout.rotaryInvFreq),
        .tokensInfoOffset = exportOffset(xqaLayout.tokensInfo),
        .qBufOffset = exportOffset(xqaLayout.kernelWorkspace),
        .bmm1ScaleOffset = exportOffset(xqaLayout.bmm1Scale),
        .bmm2ScaleOffset = exportOffset(xqaLayout.bmm2Scale),
        .sparseAttnCacheOffset = exportOffset(xqaLayout.sparseAttnCache),
        .trtllmGenWorkspaceSize = kTrtllmGenWorkspaceSize,
        .cuSeqlensSize = cuSeqlensSize,
        .cuKvSeqlensSize = cuKvSeqlensSize,
        .rotaryInvFreqSize = rotaryInvFreqSize,
        .tokensInfoSize = tokensInfoSize,
        .qBufSize = qBufSize,
        .bmm1ScaleSize = bmm1ScaleSize,
        .bmm2ScaleSize = bmm2ScaleSize,
        .sparseAttnCacheSize = sparseAttnCacheSize,
        .totalSize = static_cast<int64_t>(totalSize),
        .qBufScalarType = qDtype,
    };
}

int64_t TrtllmAttentionWorkspaceManager::getContextWorkspaceSize(at::ScalarType const qDtype, int64_t const batchSize,
    int64_t const numTokens, int64_t const numHeads, int64_t const headSize, int64_t const rotaryEmbeddingDim,
    bool const separateQKvInput, bool const fp8ContextFmha)
{
    return buildContextLayout(
        qDtype, batchSize, numTokens, numHeads, headSize, rotaryEmbeddingDim, separateQKvInput, fp8ContextFmha)
        .totalSize;
}

int64_t TrtllmAttentionWorkspaceManager::getGenerationWorkspaceSize(at::ScalarType const qDtype,
    int64_t const batchBeam, int64_t const numTokens, int64_t const numHeads, int64_t const headSize,
    int64_t const rotaryEmbeddingDim, int64_t const numKvHeads, int64_t const maxBlocksPerSequence,
    bool const useSparseAttention)
{
    return buildGenerationLayout(qDtype, batchBeam, numTokens, numHeads, headSize, rotaryEmbeddingDim, numKvHeads,
        maxBlocksPerSequence, useSparseAttention)
        .totalSize;
}

TrtllmGenContextWorkspaceViews TrtllmAttentionWorkspaceManager::materializeContextWorkspace(
    at::Tensor const& workspace, TrtllmGenContextWorkspaceLayout const& layout)
{
    return TrtllmGenContextWorkspaceViews{
        .trtllmGenWorkspace
        = *makeWorkspaceView(workspace, layout.trtllmGenWorkspaceOffset, layout.trtllmGenWorkspaceSize, at::kByte),
        .cuQSeqlens = *makeWorkspaceView(workspace, layout.cuQSeqlensOffset, layout.cuSeqlensSize, at::kInt),
        .cuKvSeqlens = *makeWorkspaceView(workspace, layout.cuKvSeqlensOffset, layout.cuSeqlensSize, at::kInt),
        .cuMaskRows = *makeWorkspaceView(workspace, layout.cuMaskRowsOffset, layout.cuSeqlensSize, at::kInt),
        .rotaryInvFreqBuf
        = makeWorkspaceView(workspace, layout.rotaryInvFreqOffset, layout.rotaryInvFreqSize, at::kFloat),
        .qBuf = makeWorkspaceView(workspace, layout.qBufOffset, layout.qBufSize, layout.qBufScalarType),
        .tokensInfo = *makeWorkspaceView(workspace, layout.tokensInfoOffset, layout.tokensInfoSize, at::kInt),
        .fmhaTileCounter
        = *makeWorkspaceView(workspace, layout.fmhaTileCounterOffset, layout.fmhaTileCounterSize, at::kUInt32),
        .fmhaBmm1Scale = makeWorkspaceView(workspace, layout.fmhaBmm1ScaleOffset, layout.fmhaBmm1ScaleSize, at::kFloat),
        .fmhaBmm2Scale = makeWorkspaceView(workspace, layout.fmhaBmm2ScaleOffset, layout.fmhaBmm2ScaleSize, at::kFloat),
    };
}

TrtllmGenContextWorkspaceViews TrtllmAttentionWorkspaceManager::materializeContextWorkspace(at::Tensor const& workspace,
    at::ScalarType const qDtype, int64_t const batchSize, int64_t const numTokens, int64_t const numHeads,
    int64_t const headSize, int64_t const rotaryEmbeddingDim, bool const fp8ContextFmha)
{
    auto const layout = buildContextLayout(
        qDtype, batchSize, numTokens, numHeads, headSize, rotaryEmbeddingDim, true, fp8ContextFmha);
    return materializeContextWorkspace(workspace, layout);
}

TrtllmGenGenerationWorkspaceViews TrtllmAttentionWorkspaceManager::materializeGenerationWorkspace(
    at::Tensor const& workspace, TrtllmGenGenerationWorkspaceLayout const& layout)
{
    return TrtllmGenGenerationWorkspaceViews{
        .trtllmGenWorkspace
        = *makeWorkspaceView(workspace, layout.trtllmGenWorkspaceOffset, layout.trtllmGenWorkspaceSize, at::kByte),
        .cuSeqlens = *makeWorkspaceView(workspace, layout.cuSeqlensOffset, layout.cuSeqlensSize, at::kInt),
        .cuKvSeqlens = *makeWorkspaceView(workspace, layout.cuKvSeqlensOffset, layout.cuKvSeqlensSize, at::kInt),
        .rotaryInvFreqBuf
        = makeWorkspaceView(workspace, layout.rotaryInvFreqOffset, layout.rotaryInvFreqSize, at::kFloat),
        .tokensInfo = *makeWorkspaceView(workspace, layout.tokensInfoOffset, layout.tokensInfoSize, at::kInt),
        .qBuf = *makeWorkspaceView(workspace, layout.qBufOffset, layout.qBufSize, layout.qBufScalarType),
        .bmm1Scale = *makeWorkspaceView(workspace, layout.bmm1ScaleOffset, layout.bmm1ScaleSize, at::kFloat),
        .bmm2Scale = *makeWorkspaceView(workspace, layout.bmm2ScaleOffset, layout.bmm2ScaleSize, at::kFloat),
        .sparseAttnCache
        = makeWorkspaceView(workspace, layout.sparseAttnCacheOffset, layout.sparseAttnCacheSize, at::kInt),
    };
}

TrtllmGenGenerationWorkspaceViews TrtllmAttentionWorkspaceManager::materializeGenerationWorkspace(
    at::Tensor const& workspace, at::ScalarType const qDtype, int64_t const batchBeam, int64_t const numTokens,
    int64_t const numHeads, int64_t const headSize, int64_t const rotaryEmbeddingDim, int64_t const numKvHeads)
{
    auto const layout = buildGenerationLayout(
        qDtype, batchBeam, numTokens, numHeads, headSize, rotaryEmbeddingDim, numKvHeads, 0, false);
    return materializeGenerationWorkspace(workspace, layout);
}

// ===========================================================================
// AttentionOp implementation.
// ===========================================================================

template <typename T>
struct SATypeConverter
{
    using Type = T;
};

template <>
struct SATypeConverter<half>
{
    using Type = uint16_t;
};

template <typename T, typename KVCacheBuffer>
struct FusedQKVMaskedAttentionDispatchParams
{
    T const* qkv_buf;
    T const* qkv_bias;
    T const* relative_attention_bias;
    bool const* attention_mask;
    float const* attention_sinks;
    float const* logn_scaling_ptr;
    int const* cache_indir;
    void* context_buf;
    bool const* finished;
    int const* sequence_lengths;
    int max_batch_size;
    int inference_batch_size;
    int beam_width;
    int head_num;
    int kv_head_num;
    int size_per_head;
    int rotary_embedding_dim;
    float rotary_embedding_base;
    RotaryScalingType rotary_embedding_scale_type;
    float rotary_embedding_scale;
    float const* rotary_embedding_inv_freq_cache;
    float2 const* rotary_embedding_cos_sin_cache;
    float rotary_embedding_short_m_scale;
    float rotary_embedding_long_m_scale;
    int rotary_embedding_max_positions;
    int rotary_embedding_original_max_positions;
    int rotary_cogvlm_vision_start;
    int rotary_cogvlm_vision_length;
    PositionEmbeddingType position_embedding_type;
    bool position_shift_enabled;

    int chunked_attention_size;
    int attention_mask_stride;
    int max_attention_window_size;
    int cyclic_attention_window_size;
    int sink_token_length;
    int const* input_lengths;
    int timestep;
    float q_scaling;
    float attn_logit_softcapping_scale;
    int relative_attention_bias_stride;
    T const* linear_bias_slopes;
    int const* ia3_tasks;
    T const* ia3_key_weights;
    T const* ia3_value_weights;
    float const* qkv_scale_out;
    bool fp8_context_fmha;
    float const* attention_out_scale;
    bool mUnfuseQkvGemm;
    tc::QuantMode quant_option;
    bool multi_block_mode;
    int max_seq_len_tile;
    int min_seq_len_tile;
    T* partial_out;
    float* partial_sum;
    float* partial_max;
    int* block_counter;
    // Cascade attention prefix-side workspace (fp32).  Sliced from the same
    // generation workspace and forwarded into Multihead_attention_params, so the
    // cascade fast-path allocates nothing of its own.
    float* cascade_partial_out{};
    float* cascade_partial_max{};
    float* cascade_partial_sum{};
    float const* kv_scale_orig_quant;
    float const* kv_scale_quant_orig;
    tc::QuantMode kv_cache_quant_mode;
    int multi_processor_count;
    KVCacheBuffer kv_block_array;
    KVLinearBuffer shift_k_cache_buffer;
    bool cross_attention = false;
    int const* memory_length_per_sample = nullptr;
    int max_distance = 0;
    bool block_sparse_attention = false;
    BlockSparseParams block_sparse_params;
    int32_t const* mrope_position_deltas;
};

template <typename T, typename KVCacheBuffer>
struct ConvertMMHAToXQAParamsHelper
{
    static constexpr Data_type data_type = DATA_TYPE_FP16;
    static constexpr bool supported = false;
};

template <>
struct ConvertMMHAToXQAParamsHelper<__half, KVLinearBuffer>
{
    static constexpr Data_type data_type = DATA_TYPE_FP16;
    static constexpr bool supported = true;
};

template <>
struct ConvertMMHAToXQAParamsHelper<__half, KVBlockArray>
{
    static constexpr Data_type data_type = DATA_TYPE_FP16;
    static constexpr bool supported = true;
};

#ifdef ENABLE_BF16
template <>
struct ConvertMMHAToXQAParamsHelper<__nv_bfloat16, KVLinearBuffer>
{
    static constexpr Data_type data_type = DATA_TYPE_BF16;
    static constexpr bool supported = true;
};

template <>
struct ConvertMMHAToXQAParamsHelper<__nv_bfloat16, KVBlockArray>
{
    static constexpr Data_type data_type = DATA_TYPE_BF16;
    static constexpr bool supported = true;
};
#endif

template <typename T, typename KVCacheBuffer>
bool AttentionOp::convertMMHAParamsToXQAParams(
    tensorrt_llm::kernels::XQAParams& xqaParams, FmhaParams const& p, bool forConfigurePlugin)
{
    bool retval = ConvertMMHAToXQAParamsHelper<T, KVCacheBuffer>::supported;
    if (!retval)
    {
        return false;
    }
    xqaParams = {};
    xqaParams.data_type = ConvertMMHAToXQAParamsHelper<T, KVCacheBuffer>::data_type;

    xqaParams.num_q_heads = mNumAttnHeads;
    xqaParams.num_kv_heads = mNumAttnKVHeads;
    xqaParams.head_size = mHeadSize;
    xqaParams.unidirectional = p.unidirectional;
    xqaParams.q_scaling = p.q_scaling;
    xqaParams.rotary_embedding_dim = p.rotary_embedding_dim;
    xqaParams.rotary_embedding_base = p.rotary_embedding_base;
    xqaParams.rotary_embedding_scale_type = p.rotary_embedding_scale_type;
    xqaParams.rotary_embedding_scale = p.rotary_embedding_scale;
    xqaParams.rotary_embedding_max_positions = p.rotary_embedding_max_positions;
    xqaParams.rotary_vision_start = p.vision_start;
    xqaParams.rotary_vision_length = p.vision_length;
    xqaParams.rotary_cos_sin = p.getRotaryCosSin();
    xqaParams.position_embedding_type = p.position_embedding_type;
    xqaParams.position_shift_enabled = p.pos_shift_enabled;
    xqaParams.remove_padding = p.remove_padding;
    xqaParams.mask_type = p.mask_type;
    xqaParams.paged_kv_cache = mPagedKVCache;
    xqaParams.tokens_per_block = p.tokens_per_block;
    xqaParams.kv_cache_quant_mode = p.quant_mode;
    xqaParams.tp_size = 1;
    xqaParams.tp_rank = 0;
    xqaParams.qkv_bias_enabled = p.qkv_bias_enabled;
    xqaParams.cross_attention = p.cross_attention;
    xqaParams.max_distance = static_cast<int>(p.fwd.relative_attention_max_distance);
    xqaParams.multi_block_mode = common::getEnvForceDeterministicAttention() ? false : mMultiBlockMode;
    // Medusa mode will have multiple query tokens.
    xqaParams.multi_query_tokens = p.is_spec_decoding_enabled && p.use_spec_decoding;
    xqaParams.is_spec_dec_tree = p.is_spec_dec_tree;
    xqaParams.force_prepare_spec_dec_tree_mask = p.force_prepare_spec_dec_tree_mask;
    xqaParams.layer_idx = static_cast<int>(p.local_layer_idx);

    if (p.quant_mode.hasInt8KvCache())
    {
        xqaParams.kv_cache_data_type = DATA_TYPE_INT8;
    }
    else if (p.quant_mode.hasFp8KvCache())
    {
        // Inputs to MLA is FP8 instead of BF16/FP16 when using FP8 KV cache.
        if (xqaParams.isMLA())
        {
            xqaParams.data_type = DATA_TYPE_E4M3;
        }
        xqaParams.kv_cache_data_type = DATA_TYPE_E4M3;
    }
    else if (p.quant_mode.hasFp4KvCache())
    {
        xqaParams.kv_cache_data_type = DATA_TYPE_E2M1;
    }
    else
    {
        xqaParams.kv_cache_data_type = xqaParams.data_type;
    }
    if (xqaParams.kv_cache_data_type == DATA_TYPE_INT8
        || (xqaParams.kv_cache_data_type == DATA_TYPE_E4M3 && (mSM < kSM_90 || mSM > kSM_120)))
    {
        xqaParams.multi_block_mode = false;
    }

    xqaParams.output = p.getOutput();
    xqaParams.qkv = p.getQkvOrQ<T>();
    xqaParams.cache_indir = p.getCacheIndirection();
    xqaParams.attention_sinks = p.getAttentionSinks();
    xqaParams.kv_scale_orig_quant = p.getKvScaleOrigQuant();
    xqaParams.kv_scale_quant_orig = p.getKvScaleQuantOrig();
    xqaParams.host_past_key_value_lengths = p.getHostPastKeyValueLengths();
    xqaParams.host_context_lengths = p.getHostContextLengths();
    xqaParams.semaphores = p.getSemaphores();
    xqaParams.workspaces = p.getWorkspace();
    xqaParams.batch_size = p.num_requests;
    xqaParams.beam_width = p.beam_width;
    // Speculative decoding mode has generation input_length > 1.
    xqaParams.generation_input_length = p.input_seq_length;
    xqaParams.chunked_attention_size
        = p.attention_chunk_size && !tc::getEnvDisableChunkedAttentionInGenPhase() ? *p.attention_chunk_size : INT_MAX;
    xqaParams.max_attention_window_size = p.max_attention_window_size;
    xqaParams.cyclic_attention_window_size = p.cyclic_attention_window_size;
    xqaParams.max_blocks_per_sequence = p.max_blocks_per_sequence;
    xqaParams.sink_token_length = p.sink_token_length;
    xqaParams.max_past_kv_length = p.max_past_kv_length;
    xqaParams.qkv_bias = p.getQkvBias<T>();
    xqaParams.sequence_lengths = p.getSequenceLength();
    xqaParams.context_lengths = p.getContextLengths();
    xqaParams.alibi_slopes = p.getAlibiSlopes<T>();
    // Pre-computed rotary inv freq when building the engines.
    xqaParams.rotary_embedding_inv_freq_cache = p.getRotaryInvFreq();
    if (!forConfigurePlugin)
    {
        // Speculative decoding (need to take new generated ids into consideration).
        TLLM_CHECK_WITH_INFO(
            !(p.is_spec_decoding_enabled && p.use_spec_decoding) || p.getSpecDecodingPackedMask() != nullptr,
            "Speculative decoding mode needs a valid packed_mask input tensor.");
    }
    xqaParams.spec_decoding_packed_mask = p.getSpecDecodingPackedMask();
    xqaParams.spec_decoding_position_offsets = p.getSpecDecodingPositionOffsetsForCpp();
    xqaParams.spec_decoding_generation_lengths = p.getSpecDecodingGenerationLengths();
    xqaParams.spec_decoding_is_generation_length_variable = p.spec_decoding_is_generation_length_variable;
    xqaParams.spec_decoding_max_generation_length = p.spec_decoding_max_generation_length;
    xqaParams.spec_decoding_bl_tree_mask_offset = p.getSpecDecodingBlTreeMaskOffset();
    xqaParams.spec_decoding_bl_tree_mask = p.getSpecDecodingBlTreeMask();
    xqaParams.spec_bl_tree_first_sparse_mask_offset_kv = p.getSpecBlTreeFirstSparseMaskOffsetKv();
    xqaParams.mrope_position_deltas = p.getMropePositionDeltas();
    xqaParams.helix_position_offsets = p.getHelixPositionOffsets();
    xqaParams.helix_is_inactive_rank = p.getHelixIsInactiveRank();
    xqaParams.softmax_stats = p.getSoftmaxStatsTensor();
    xqaParams.trtllm_gen_jit_warmup = p.trtllm_gen_jit_warmup;
    xqaParams.trtllm_gen_jit_warmup_max_num_requests = p.max_num_requests;
    xqaParams.trtllm_gen_jit_warmup_max_seq_len_q = p.max_context_length;
    xqaParams.trtllm_gen_jit_warmup_max_seq_len_kv = p.max_seq_len;

    xqaParams.logn_scaling_ptr = p.getLognScalingPtr();
    xqaParams.total_num_input_tokens = p.num_tokens;
    xqaParams.is_fp8_output = mFP8AttenOutput;
    xqaParams.fp8_out_scale = ((mFP8AttenOutput) ? p.getOutScale() : nullptr);
    // Parameters required for FP4 output.
    xqaParams.output_sf = p.getOutputSf();
    xqaParams.fp4_out_sf_scale = p.getOutSfScale();
    xqaParams.start_token_idx_sf = p.token_offset;
    // Parameters for sparse attention
    xqaParams.sparse_params = p.sparse_params;
    xqaParams.use_sparse_attention_gen_paged = useTllmGenSparseAttentionPaged(p);
    // Skip softmax threshold.
    xqaParams.skip_softmax_threshold_scale_factor
        = static_cast<float>(p.fwd.sparse_runtime_params.threshold_scale_factor_decode);
#ifdef SKIP_SOFTMAX_STAT
    // Statistics of skip-softmax, pointers of device memory for output
    xqaParams.skip_softmax_total_blocks = mSkipSoftmaxTotalBlocks;
    xqaParams.skip_softmax_skipped_blocks = mSkipSoftmaxSkippedBlocks;
#endif
    // Cross attention parameters.
    xqaParams.encoder_input_lengths = p.getEncoderInputLengths();

    return true;
}

template <typename T_MMHA, typename T, typename KVCacheBuffer, bool CROSS_ATTENTION>
void fusedQKV_masked_attention_dispatch(Multihead_attention_params<T_MMHA, CROSS_ATTENTION>& params,
    FusedQKVMaskedAttentionDispatchParams<T, KVCacheBuffer> const& input_params, cudaStream_t stream)
{
    using DataType = typename SATypeConverter<T>::Type;

    // Prepare the parameters.
    params = {};

    int hidden_units = input_params.head_num * input_params.size_per_head;
    int hidden_units_kv = input_params.kv_head_num * input_params.size_per_head;
    if (input_params.qkv_bias != nullptr)
    {
        params.q_bias = reinterpret_cast<DataType const*>(input_params.qkv_bias);
        params.k_bias = reinterpret_cast<DataType const*>(input_params.qkv_bias) + hidden_units;
        params.v_bias = reinterpret_cast<DataType const*>(input_params.qkv_bias) + hidden_units + hidden_units_kv;
    }
    else
    {
        params.q_bias = nullptr;
        params.k_bias = nullptr;
        params.v_bias = nullptr;
    }

    // Set the output buffer.
    params.out = input_params.context_buf;

    // Set the input buffers.
    params.q = reinterpret_cast<DataType const*>(input_params.qkv_buf);
    params.k = reinterpret_cast<DataType const*>(input_params.qkv_buf) + hidden_units;
    params.v = reinterpret_cast<DataType const*>(input_params.qkv_buf) + hidden_units + hidden_units_kv;

    params.int8_kv_cache = input_params.kv_cache_quant_mode.hasInt8KvCache();
    params.fp8_kv_cache = input_params.kv_cache_quant_mode.hasFp8KvCache();
    if (input_params.kv_cache_quant_mode.hasKvCacheQuant())
    {
        params.kv_scale_orig_quant = input_params.kv_scale_orig_quant;
        params.kv_scale_quant_orig = input_params.kv_scale_quant_orig;
    }

    params.stride = hidden_units + 2 * hidden_units_kv;
    params.finished = const_cast<bool*>(input_params.finished);

    params.cache_indir = input_params.cache_indir;
    params.batch_size = input_params.inference_batch_size;
    params.beam_width = input_params.beam_width;
    params.chunked_attention_size = input_params.chunked_attention_size;
    if (input_params.chunked_attention_size != INT_MAX && !tc::getEnvDisableChunkedAttentionInGenPhase())
    {
        TLLM_CHECK_WITH_INFO((input_params.chunked_attention_size & (input_params.chunked_attention_size - 1)) == 0,
            "Attention chunk size should be a power of 2.");
        params.chunked_attention_size_log2 = std::log2(input_params.chunked_attention_size);
    }
    else
    {
        params.chunked_attention_size_log2 = 0;
    }
    params.max_attention_window_size = input_params.max_attention_window_size;
    params.cyclic_attention_window_size = input_params.cyclic_attention_window_size;
    params.sink_token_length = input_params.sink_token_length;
    params.length_per_sample = input_params.sequence_lengths; // max_input_length + current output length
    // timestep for shared memory size calculation and rotary embedding computation
    params.timestep = input_params.timestep;
    params.num_heads = input_params.head_num;
    params.num_kv_heads = input_params.kv_head_num;
    params.hidden_size_per_head = input_params.size_per_head;
    params.rotary_embedding_dim = input_params.rotary_embedding_dim;
    params.rotary_embedding_base = input_params.rotary_embedding_base;
    params.rotary_embedding_scale_type = input_params.rotary_embedding_scale_type;
    params.rotary_embedding_scale = input_params.rotary_embedding_scale;
    params.rotary_embedding_inv_freq_cache = input_params.rotary_embedding_inv_freq_cache;
    params.rotary_embedding_cos_sin_cache = input_params.rotary_embedding_cos_sin_cache;
    params.rotary_embedding_short_m_scale = input_params.rotary_embedding_short_m_scale;
    params.rotary_embedding_long_m_scale = input_params.rotary_embedding_long_m_scale;
    params.rotary_embedding_max_positions = input_params.rotary_embedding_max_positions;
    params.rotary_embedding_original_max_positions = input_params.rotary_embedding_original_max_positions;
    params.rotary_cogvlm_vision_start = input_params.rotary_cogvlm_vision_start;
    params.rotary_cogvlm_vision_length = input_params.rotary_cogvlm_vision_length;
    params.position_embedding_type = input_params.position_embedding_type;
    params.position_shift_enabled = input_params.position_shift_enabled;
    // Note: keep norm factor (sqrt(K_dim)) when adopting megatron T5 structure (may adjust)
    params.inv_sqrt_dh = 1.F / (sqrtf((float) params.hidden_size_per_head) * input_params.q_scaling);
    params.attn_logit_softcapping_scale = input_params.attn_logit_softcapping_scale;
    params.attn_logit_softcapping_inverse_scale = 1.0f / input_params.attn_logit_softcapping_scale;

    params.logn_scaling_ptr = input_params.logn_scaling_ptr;
    params.relative_attention_bias = reinterpret_cast<DataType const*>(input_params.relative_attention_bias);
    params.relative_attention_bias_stride = input_params.relative_attention_bias_stride;
    params.max_distance = input_params.max_distance;
    params.block_sparse_attention = input_params.block_sparse_attention;
    params.block_sparse_params = input_params.block_sparse_params;

    // Attention mask input.
    params.attention_mask = input_params.attention_mask;
    params.attention_mask_stride = input_params.attention_mask_stride;

    // Attention sinks.
    params.attention_sinks = input_params.attention_sinks;

    // The slope of linear position bias per head, e.g., ALiBi.
    if (input_params.linear_bias_slopes != nullptr)
    {
        params.linear_bias_slopes = reinterpret_cast<DataType const*>(input_params.linear_bias_slopes);
    }
    params.input_lengths = input_params.input_lengths;

    params.ia3_tasks = input_params.ia3_tasks;
    params.ia3_key_weights = reinterpret_cast<DataType const*>(input_params.ia3_key_weights);
    params.ia3_value_weights = reinterpret_cast<DataType const*>(input_params.ia3_value_weights);

    if (input_params.quant_option.hasStaticActivationScaling() || input_params.fp8_context_fmha)
    {
        // qkv_scale_out is nullptr currently (no scale).
        params.qkv_scale_quant_orig = input_params.qkv_scale_out;
        TLLM_CHECK_WITH_INFO(!input_params.fp8_context_fmha || input_params.attention_out_scale != nullptr,
            "attention output scale should be provided.");
        params.attention_out_scale_orig_quant = input_params.attention_out_scale;
    }

    params.multi_block_mode = input_params.multi_block_mode;
    // Cascade-attention partials must be wired regardless of multi_block_mode.
    // Cascade decode runs with multi_block disabled (short-decode workloads have
    // max_num_seq_len_tiles == 1, so enable_multi_block is structurally false).
    // Gating these behind multi_block_mode leaves cascade_partial_* null and makes
    // launch_cascade_attention fall back with "cascade workspace not provisioned".
    params.cascade_partial_out = input_params.cascade_partial_out;
    params.cascade_partial_max = input_params.cascade_partial_max;
    params.cascade_partial_sum = input_params.cascade_partial_sum;
    if (input_params.multi_block_mode)
    {
        params.min_seq_len_tile = input_params.min_seq_len_tile;
        params.max_seq_len_tile = input_params.max_seq_len_tile;

        params.partial_out = reinterpret_cast<DataType*>(input_params.partial_out);
        params.partial_sum = input_params.partial_sum;
        params.partial_max = input_params.partial_max;

        params.block_counter = input_params.block_counter;
    }

    params.multi_processor_count = input_params.multi_processor_count;

    // cross attn
    params.memory_length_per_sample = input_params.memory_length_per_sample;

    params.mrope_position_deltas = input_params.mrope_position_deltas;
    sync_check_cuda_error(stream);

    masked_multihead_attention(params, input_params.kv_block_array, input_params.shift_k_cache_buffer, stream);
}

#define INSTANTIATE_MMHA_DISPATCH(T_MMHA, T)                                                                           \
    template void fusedQKV_masked_attention_dispatch(Multihead_attention_params<T_MMHA, false>&,                       \
        FusedQKVMaskedAttentionDispatchParams<T, KVLinearBuffer> const&, cudaStream_t stream);                         \
    template void fusedQKV_masked_attention_dispatch(Multihead_attention_params<T_MMHA, true>&,                        \
        FusedQKVMaskedAttentionDispatchParams<T, KVLinearBuffer> const&, cudaStream_t stream);                         \
    template void fusedQKV_masked_attention_dispatch(Multihead_attention_params<T_MMHA, false>&,                       \
        FusedQKVMaskedAttentionDispatchParams<T, KVBlockArray> const&, cudaStream_t stream);                           \
    template void fusedQKV_masked_attention_dispatch(Multihead_attention_params<T_MMHA, true>&,                        \
        FusedQKVMaskedAttentionDispatchParams<T, KVBlockArray> const&, cudaStream_t stream);
INSTANTIATE_MMHA_DISPATCH(float, float)
INSTANTIATE_MMHA_DISPATCH(uint16_t, half)
#ifdef ENABLE_BF16
INSTANTIATE_MMHA_DISPATCH(__nv_bfloat16, __nv_bfloat16)
#endif
#undef INSTANTIATE_MMHA_DISPATCH

int AttentionOp::getHeadSize(bool checkInit) const
{
    if (checkInit)
    {
        TLLM_CHECK_WITH_INFO(mHeadSize > 0, "Trying to read mHeadSize before it's been initialized");
    }
    return mHeadSize;
}

size_t AttentionOp::getFmhaMultiCtasKvScratchSize(FmhaParams const& p) const noexcept
{
    static constexpr size_t kMultiCtasKvRowsPerCta = 256;
    static constexpr size_t kMultiCtasKvStatsPerRow = 2;
    static constexpr size_t kMultiCtasKvPartialOElementSize = 2;

    size_t const headDimV
        = p.is_mla_enable ? static_cast<size_t>(p.mla_params.kv_lora_rank) : static_cast<size_t>(getHeadSize());
    size_t const maxRows = kMultiCtasKvRowsPerCta * static_cast<size_t>(mMultiProcessorCount);
    size_t const partialStatsSize = sizeof(float) * kMultiCtasKvStatsPerRow * maxRows;
    size_t const partialOSize = kMultiCtasKvPartialOElementSize * maxRows * headDimV;

    return partialStatsSize + partialOSize;
}

size_t AttentionOp::contextMlaWorkspaceBytesPerToken(int32_t numAttnHeads, int32_t qkRopeHeadDim, int32_t qkNopeHeadDim,
    int32_t vHeadDim, bool fp8ContextMla, bool separateQAndKvInput, bool sparseMla) noexcept
{
    // Only the fp8 context-MLA separate-Q/KV path stages total_kv_len-scaled K/V dequant buffers.
    // Sparse MLA reads K/V directly from the paged KV cache (no staging), so its per-token cost is 0.
    if (!fp8ContextMla || !separateQAndKvInput || sparseMla)
    {
        return 0;
    }
    // Mirror getWorkspaceSizeForContext's dim layout for the non-sparse fp8 branch:
    //   total_k_dim_all_heads = numAttnHeads * (qk_rope_head_dim + qk_nope_head_dim)
    //   total_v_dim_all_heads = numAttnHeads * v_head_dim
    // The buffers are fp8 (1 byte/element), so bytes/token == element count.
    int const dimKPerHead = qkRopeHeadDim + qkNopeHeadDim;
    int const dimVPerHead = vHeadDim;
    return static_cast<size_t>(numAttnHeads) * static_cast<size_t>(dimKPerHead + dimVPerHead);
}

size_t AttentionOp::getWorkspaceSizeForContext(FmhaParams const& p, int32_t max_num_seq, int32_t input_seq_length,
    int32_t cross_kv_length, int32_t max_num_tokens, int32_t total_kv_len) const noexcept
{
    if (max_num_tokens == 0)
    {
        return 0;
    }

    int const local_hidden_units_qo = mNumAttnHeads * getHeadSize();
    int const local_hidden_units_kv = mNumAttnKVHeads * getHeadSize();

    auto const size = tensorrt_llm::runtime::BufferDataType(p.type).getSize();

    size_t context_workspace_size = 0;

    auto const batch_size = static_cast<size_t>(max_num_seq);
    auto const kv_seq_length = (isCrossAttention(p) ? cross_kv_length : input_seq_length);
    // The unfused-MHA buffers below must upper-bound the enqueueContext carve, which sizes them by
    // batch_size * input_seq_length (not num_tokens): with padding removal the actual token count can be
    // smaller than batch_size * max(context q length), so sizing by max_num_tokens underestimates.
    size_t const attention_mask_size = mEnableContextFMHA ? 0 : size * batch_size * input_seq_length * kv_seq_length;
    size_t const cu_seqlens_size = sizeof(int) * (batch_size + 1);
    size_t const rotary_inv_freq_size = sizeof(float) * batch_size * p.rotary_embedding_dim / 2;

    size_t q_buf_2_size = 0;
    if (!mEnableContextFMHA)
    {
        // Unfused mha
        q_buf_2_size = size * batch_size * input_seq_length * local_hidden_units_qo;
    }
    else if (mFmhaDispatcher->isSeparateQAndKvInput())
    {
        // Paged context fmha
        q_buf_2_size = (mFP8ContextFMHA ? 1 : size) * max_num_tokens * local_hidden_units_qo;
    }

    size_t const k_buf_2_size = mEnableContextFMHA ? 0 : size * batch_size * kv_seq_length * local_hidden_units_kv;
    size_t const v_buf_2_size = mEnableContextFMHA ? 0 : size * batch_size * kv_seq_length * local_hidden_units_kv;
    size_t const qk_buf_size
        = mEnableContextFMHA ? 0 : size * batch_size * p.num_heads * input_seq_length * kv_seq_length;
    size_t const qkv_buf_2_size = mEnableContextFMHA ? 0 : size * batch_size * input_seq_length * local_hidden_units_qo;
    size_t const qk_buf_float_size
        = mEnableContextFMHA ? 0 : sizeof(float) * batch_size * p.num_heads * input_seq_length * kv_seq_length;
    int dim_q_per_head = (p.mla_params.qk_rope_head_dim + p.mla_params.qk_nope_head_dim);
    int dim_k_per_head = (p.mla_params.qk_rope_head_dim + p.mla_params.qk_nope_head_dim);
    int dim_v_per_head = (p.mla_params.v_head_dim);
    if (useSparseMLA(p))
    {
        dim_q_per_head = p.mla_params.kv_lora_rank + p.mla_params.qk_rope_head_dim;
        dim_k_per_head = p.mla_params.kv_lora_rank + p.mla_params.qk_rope_head_dim;
        dim_v_per_head = p.mla_params.rope_append ? p.mla_params.kv_lora_rank
                                                  : p.mla_params.kv_lora_rank + p.mla_params.qk_rope_head_dim;
    }

    // Total dimension per token across all heads for Q, K, and V components respectively
    int const total_q_dim_all_heads = mNumAttnHeads * dim_q_per_head;
    int const total_k_dim_all_heads
        = mNumAttnHeads * dim_k_per_head; // Assuming effective num_kv_heads = head_num for layout
    int const total_v_dim_all_heads
        = mNumAttnHeads * dim_v_per_head; // Assuming effective num_kv_heads = head_num for layout
    bool const useSageAttnSeparateQkv = mEnableContextFMHA && !p.is_mla_enable
        && mFmhaDispatcher->isSeparateQAndKvInput()
        && (p.fwd.sage_attn_num_elts_per_blk_q > 0 || p.fwd.sage_attn_num_elts_per_blk_k > 0
            || p.fwd.sage_attn_num_elts_per_blk_v > 0);

    // Packed fp8 qkv buffer size for normal fp8 context FMHA
    size_t fp8_qkv_buffer_size = mFP8ContextFMHA && mEnableContextFMHA && !mFmhaDispatcher->isSeparateQAndKvInput()
        ? max_num_tokens * (local_hidden_units_qo + (2ULL * local_hidden_units_kv))
        : 0;
    // Separate fp8 q/k/v buffer size for fp8 context MLA
    size_t fp8_q_buf_size = 0;
    size_t fp8_k_buf_size = 0;
    size_t fp8_v_buf_size = 0;
    if (mEnableContextFMHA && mFP8ContextMLA && mFmhaDispatcher->isSeparateQAndKvInput())
    {
        fp8_q_buf_size = max_num_tokens * static_cast<size_t>(total_q_dim_all_heads);

        if (useSparseMLA(p))
        {
            // Sparse MLA (absorption mode): K and V are stored directly in KV cache during MLA RoPE kernel.
            // No separate FP8 buffers needed for K/V since they're read from paged KV cache (Q_PAGED_KV layout).
            fp8_k_buf_size = 0;
            fp8_v_buf_size = 0;
        }
        else
        {
            // Use total_kv_len when available (KV cache reuse causes total_kv_len >> max_num_tokens).
            // enqueueContext sizes these buffers by total_kv_len, so workspace must match.
            // NOTE: the per-token cost of these two buffers (total_k_dim_all_heads + total_v_dim_all_heads) is
            // the single source of truth exposed via contextMlaWorkspaceBytesPerToken() for the KV-cache
            // estimator's workspace reserve. Keep the two in sync if this dim layout changes.
            size_t const kv_buf_tokens = std::max(static_cast<size_t>(total_kv_len),
                static_cast<size_t>(p.fwd.chunked_prefill_buffer_batch_size) * max_num_tokens);
            fp8_k_buf_size = kv_buf_tokens * static_cast<size_t>(total_k_dim_all_heads);
            fp8_v_buf_size = kv_buf_tokens * static_cast<size_t>(total_v_dim_all_heads);
            TLLM_CHECK(static_cast<size_t>(total_k_dim_all_heads + total_v_dim_all_heads)
                == contextMlaWorkspaceBytesPerToken(mNumAttnHeads, p.mla_params.qk_rope_head_dim,
                    p.mla_params.qk_nope_head_dim, p.mla_params.v_head_dim, mFP8ContextMLA,
                    /*separateQAndKvInput=*/true, useSparseMLA(p)));
        }
    }
    else if (useSageAttnSeparateQkv)
    {
        fp8_q_buf_size = max_num_tokens * static_cast<size_t>(local_hidden_units_qo);
        fp8_k_buf_size = total_kv_len * static_cast<size_t>(local_hidden_units_kv);
        fp8_v_buf_size = total_kv_len * static_cast<size_t>(local_hidden_units_kv);
    }

    int32_t const q_max_n_blk = p.fwd.sage_attn_num_elts_per_blk_q > 0
        ? tc::divUp(max_num_tokens, p.fwd.sage_attn_num_elts_per_blk_q) + batch_size - 1
        : 0;
    int32_t const k_max_n_blk = p.fwd.sage_attn_num_elts_per_blk_k > 0
        ? tc::divUp(total_kv_len, p.fwd.sage_attn_num_elts_per_blk_k) + batch_size - 1
        : 0;
    size_t const sage_q_sfs_buffer_size = sizeof(float) * mNumAttnHeads * static_cast<size_t>(q_max_n_blk);
    size_t const sage_k_sfs_buffer_size = sizeof(float) * mNumAttnKVHeads * static_cast<size_t>(k_max_n_blk);
    size_t const sage_v_sfs_buffer_size = p.fwd.sage_attn_num_elts_per_blk_v > 0
        ? sizeof(float) * tc::divUp(local_hidden_units_kv, std::max<int64_t>(1, p.fwd.sage_attn_num_elts_per_blk_v))
        : 0;

    size_t const padding_offset_size = mEnableContextFMHA ? 0 : sizeof(int) * batch_size * input_seq_length;
    size_t const encoder_padding_offset_size = mEnableContextFMHA ? 0 : sizeof(int) * batch_size * cross_kv_length;
    // Each token holds (batch_idx, token_idx_in_seq) int2.
    size_t const tokens_info_size = sizeof(int2) * max_num_tokens;
    size_t const fmha_scheduler_counter = mEnableContextFMHA ? sizeof(uint32_t) : 0;
    size_t const fmha_bmm1_scale_size = (mFP8ContextFMHA || mFP8ContextMLA) ? sizeof(float) * 2 : 0;
    size_t const fmha_bmm2_scale_size = (mFP8ContextFMHA || mFP8ContextMLA) ? sizeof(float) : 0;

    size_t const fmha_multi_ctas_kv_scratch_size = useTllmGenSparseAttention(p) ? getFmhaMultiCtasKvScratchSize(p) : 0;

    AttentionContextWorkspaceSizes workspaceSizes{};
    workspaceSizes.attentionMask = attention_mask_size;
    workspaceSizes.cuQSeqlens = cu_seqlens_size;
    workspaceSizes.cuKvSeqlens = cu_seqlens_size;
    workspaceSizes.cuMaskRows = cu_seqlens_size;
    workspaceSizes.rotaryInvFreq = rotary_inv_freq_size;
    workspaceSizes.qBuf = q_buf_2_size;
    workspaceSizes.kBuf = k_buf_2_size;
    workspaceSizes.vBuf = v_buf_2_size;
    workspaceSizes.qkBuf = qk_buf_size;
    workspaceSizes.qkvBuf = qkv_buf_2_size;
    workspaceSizes.qkFloatBuf = qk_buf_float_size;
    workspaceSizes.fp8QkvBuf = fp8_qkv_buffer_size;
    workspaceSizes.fp8QBuf = fp8_q_buf_size;
    workspaceSizes.fp8KBuf = fp8_k_buf_size;
    workspaceSizes.fp8VBuf = fp8_v_buf_size;
    workspaceSizes.paddingOffset = padding_offset_size;
    workspaceSizes.encoderPaddingOffset = encoder_padding_offset_size;
    workspaceSizes.tokensInfo = tokens_info_size;
    workspaceSizes.fmhaTileCounter = fmha_scheduler_counter;
    workspaceSizes.fmhaBmm1Scale = fmha_bmm1_scale_size;
    workspaceSizes.fmhaBmm2Scale = fmha_bmm2_scale_size;
    workspaceSizes.sageQScale = sage_q_sfs_buffer_size;
    workspaceSizes.sageKScale = sage_k_sfs_buffer_size;
    workspaceSizes.sageVScale = sage_v_sfs_buffer_size;
    workspaceSizes.fmhaMultiCtasKvScratch = fmha_multi_ctas_kv_scratch_size;
    context_workspace_size = AttentionWorkspaceManager::buildContextLayout(workspaceSizes).totalSize;

    return context_workspace_size;
}

size_t AttentionOp::getWorkspaceSizeForGeneration(FmhaParams const& p, int32_t max_num_seq,
    int32_t max_attention_window_size, int32_t max_num_tokens, int32_t max_blocks_per_sequence) const noexcept
{
    if (max_num_tokens == 0)
    {
        return 0;
    }

    auto const size = tensorrt_llm::runtime::BufferDataType(p.type).getSize();
    int const batch_beam = max_num_seq;

    // Compute the workspace size for MLA.
    size_t fmha_v2_mla_workspace_size = 0;
    if (p.is_mla_enable)
    {
        size_t flash_mla_workspace_size = 0;
        if (mUseGenFlashMLA)
        {
            static constexpr int TileSchedulerMetaDataSize = 8;

            int s_q = p.mla_params.predicted_tokens_per_seq;

            int num_q_heads = p.num_heads;
            int num_kv_heads = mNumKVHeads;
            int head_size_v = (p.use_sparse_attention && !p.mla_params.rope_append)
                ? p.mla_params.kv_lora_rank + p.mla_params.qk_rope_head_dim
                : p.mla_params.kv_lora_rank;

            int num_sm_parts = getFlashMlaNumSmParts(s_q, num_q_heads, num_kv_heads, head_size_v);

            AttentionFlashMlaWorkspaceSizes flashMlaWorkspaceSizes{};
            flashMlaWorkspaceSizes.tileSchedulerMetadata = sizeof(int) * (num_sm_parts * TileSchedulerMetaDataSize);
            flashMlaWorkspaceSizes.numSplits = sizeof(int) * (batch_beam + 1);
            flashMlaWorkspaceSizes.softmaxLse = sizeof(float) * (batch_beam * s_q * num_q_heads);
            flashMlaWorkspaceSizes.softmaxLseAccum = sizeof(float) * ((batch_beam + num_sm_parts) * num_q_heads * s_q);
            flashMlaWorkspaceSizes.outAccum
                = sizeof(float) * ((batch_beam + num_sm_parts) * num_q_heads * s_q * head_size_v);
            flash_mla_workspace_size = AttentionWorkspaceManager::buildFlashMlaLayout(flashMlaWorkspaceSizes).totalSize;
        }

        size_t const cu_seqlens_size = sizeof(int) * (max_num_seq + 1);
        size_t const fmha_scheduler_counter = sizeof(uint32_t);
        size_t const fmha_multi_ctas_kv_scratch_size = getFmhaMultiCtasKvScratchSize(p);

        int const NUM_BUFFERS = 5;
        size_t workspaces[NUM_BUFFERS];
        workspaces[0] = mIsGenerationMLA ? 0 : cu_seqlens_size; // cu_q_len
        workspaces[1] = mIsGenerationMLA ? 0 : cu_seqlens_size; // cu_kv_len
        workspaces[2] = mIsGenerationMLA ? 0 : fmha_scheduler_counter;
        workspaces[3] = fmha_multi_ctas_kv_scratch_size;
        workspaces[4] = flash_mla_workspace_size;

        fmha_v2_mla_workspace_size = tc::calculateTotalWorkspaceSize(workspaces, NUM_BUFFERS);
    }

    size_t generation_workspace_size = 0;
    // The minimum number of sequence length tiles (limited by the shared memory size).
    int minSeqLenTile
        = estimate_min_multi_block_count(max_attention_window_size, mMaxSharedMemoryPerBlockOptin - 2048, size);
    int32_t const maxSeqLenTile = std::max(
        {minSeqLenTile, getMaxNumSeqLenTile(p, batch_beam), (int) tc::divUp(mMultiProcessorCount, p.num_heads)});

    size_t const partial_out_size = size * batch_beam * p.num_heads * mHeadSize * maxSeqLenTile;
    size_t const partial_sum_size = sizeof(float) * batch_beam * p.num_heads * maxSeqLenTile;
    size_t const partial_max_size = sizeof(float) * batch_beam * p.num_heads * maxSeqLenTile;
    size_t const shift_k_cache_size = (!p.pos_shift_enabled || isCrossAttention(p))
        ? 0
        : size * batch_beam * p.num_heads * mHeadSize * max_attention_window_size;
    AttentionGenerationWorkspaceSizes generationWorkspaceSizes{};
    generationWorkspaceSizes.partialOut = partial_out_size;
    generationWorkspaceSizes.partialSum = partial_sum_size;
    generationWorkspaceSizes.partialMax = partial_max_size;
    generationWorkspaceSizes.shiftKCache = shift_k_cache_size;
    {
        auto const cascadeSizes
            = tensorrt_llm::kernels::mmha::cascade::getCascadeWorkspaceSizes(batch_beam, p.num_heads, mHeadSize);
        generationWorkspaceSizes.cascadeOut = cascadeSizes.out;
        generationWorkspaceSizes.cascadeMax = cascadeSizes.mMax;
        generationWorkspaceSizes.cascadeSum = cascadeSizes.lSum;
    }
    generation_workspace_size = AttentionWorkspaceManager::buildGenerationLayout(generationWorkspaceSizes).totalSize;

    size_t xqa_workspace_size = 0;
    if (mEnableXQA)
    {
        size_t const cu_seqlens_size = sizeof(int) * (batch_beam + 1);
        size_t const cu_kv_seqlens_size = sizeof(int) * (batch_beam + 1);
        size_t const rotary_inv_freq_size = sizeof(float) * batch_beam * p.rotary_embedding_dim / 2;
        // Two workspaces for sparse attention. One for the sequence lengths, and one for kv block offsets.
        size_t const sparse_attn_cache_size = useTllmGenSparseAttentionPaged(p)
            ? sizeof(int) * (batch_beam + batch_beam * 2 * max_blocks_per_sequence) * mNumKVHeads
            : 0;
        AttentionXqaWorkspaceSizes xqaWorkspaceSizes{};
        xqaWorkspaceSizes.cuSeqlens = cu_seqlens_size;
        xqaWorkspaceSizes.cuKvSeqlens = cu_kv_seqlens_size;
        xqaWorkspaceSizes.rotaryInvFreq = rotary_inv_freq_size;
        xqaWorkspaceSizes.tokensInfo = max_num_tokens * sizeof(int2);
        xqaWorkspaceSizes.bmm1Scale = sizeof(float) * 2;
        xqaWorkspaceSizes.bmm2Scale = sizeof(float);
        xqaWorkspaceSizes.sparseAttnCache = sparse_attn_cache_size;
        xqaWorkspaceSizes.kernelWorkspace = mXqaDispatcher->getWorkspaceSize(
            std::min<uint32_t>(p.spec_decoding_max_generation_length * max_num_seq, max_num_tokens));
        xqa_workspace_size
            = AttentionWorkspaceManager::buildXqaLayout(xqaWorkspaceSizes, mXqaDispatcher->getWorkspaceAlignment())
                  .totalSize;
    }

    return std::max(std::max(generation_workspace_size, xqa_workspace_size), fmha_v2_mla_workspace_size);
}

int AttentionOp::getMaxNumSeqLenTile(FmhaParams const& p, int batch_beam_size) const
{
    if (mMultiBlockMode)
    {
        // And we allocate the buffer based on the maximum number of blocks per sequence (batch_beam_size = 1).
        // Assume we can only have 1 block (large block size like 1024) in SM, and we only want one wave of blocks.
        return tc::getEnvMmhaMultiblockDebug() ? std::max(kReservedMaxSeqLenTilePerSeq, getEnvMmhaBlocksPerSequence())
                                               : tc::divUp(mMultiProcessorCount, batch_beam_size * p.num_heads);
    }
    return 0;
}

template <typename T>
int AttentionOp::mlaGeneration(MlaParams<T>& params, FmhaParams const& p, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(params.seqQOffset != nullptr, "seqQOffset is nullptr.");
    TLLM_CHECK_WITH_INFO(params.cache_seq_lens != nullptr, "cache_seq_lens is nullptr.");
    TLLM_CHECK_WITH_INFO(params.fmha_tile_counter != nullptr, "fmha_tile_counter is nullptr.");
    if (mFP8GenerationMLA)
    {
        TLLM_CHECK_WITH_INFO(params.quant_q_buf != nullptr, "quant_q_buf is nullptr.");
        TLLM_CHECK_WITH_INFO(params.bmm1_scale != nullptr, "bmm1_scale is nullptr.");
        TLLM_CHECK_WITH_INFO(params.bmm2_scale != nullptr, "bmm2_scale is nullptr.");
    }

    int const num_kv_heads = 1;
    int const head_size = p.mla_params.kv_lora_rank + p.mla_params.qk_rope_head_dim;
    int const head_size_v = (useSparseMLA(p) && !p.mla_params.rope_append) ? head_size : p.mla_params.kv_lora_rank;
    int32_t const batch_beam = p.beam_width * p.num_requests;

    // The element size of the KV cache.
    auto const elemSize = mFP8GenerationMLA ? sizeof(__nv_fp8_e4m3) : sizeof(T);
    auto const sizePerToken = num_kv_heads * head_size * elemSize;
    params.cache_type = (mFP8GenerationMLA ? KvCacheDataType::FP8 : KvCacheDataType::BASE);

    int32_t const kvCachePoolIndex = p.getKvCachePoolIndex(p.local_layer_idx);
    auto kv_cache_buffer = KVBlockArray(batch_beam, p.getMaxBlocksPerSequence(), p.tokens_per_block, sizePerToken,
        p.cyclic_attention_window_size, p.max_cyclic_attention_window_size, p.sink_token_length,
        p.can_use_one_more_block, p.getHostPrimaryPoolPtr(), p.getHostSecondaryPoolPtr(),
        p.getKvCacheBlockOffsets(kvCachePoolIndex));

    // Static sparse NVFP4 MLA reads a separately dequantized FP8 scratch pool,
    // so this paged-cache scale descriptor is not consumed by the attention kernel.
    auto kv_scale_cache_buffer = KVBlockArray();

    void* scratchPtr = params.workspace;

    params.quant_scale_o = p.getOutScale();
    params.quant_scale_q = p.getKvScaleOrigQuant();
    params.quant_scale_kv = p.getKvScaleOrigQuant();
    params.dequant_scale_q = p.getKvScaleQuantOrig();
    params.dequant_scale_kv = p.getKvScaleQuantOrig();
    params.host_bmm1_scale
        = 1 / (p.q_scaling * sqrt((float) (p.mla_params.qk_nope_head_dim + p.mla_params.qk_rope_head_dim)));

    if (p.runtime_perf_knobs.has_value())
    {
        int64_t const* const runtimePerfKnobs = p.getRuntimePerfKnobs();
        int64_t multi_block_mode_val = runtimePerfKnobs[0];
        mMultiBlockMode = multi_block_mode_val == 1;
        int64_t enable_context_fmha_fp32_acc_val = runtimePerfKnobs[1];
        mFMHAForceFP32Acc = mFMHAForceFP32Acc || enable_context_fmha_fp32_acc_val == 1;
    }

    if (common::getEnvForceDeterministicAttention())
    {
        mMultiBlockMode = false;
    }

    if (mUseTllmGen)
    {
        TLLM_CHECK_WITH_INFO(mTllmGenFMHARunner.get(), "mTllmGenFMHARunner not initialized.");
        TllmGenFmhaRunnerParams tllmRunnerParams{};

        // Parameters to select kernels.
        // MLA generation kernels use dense mask. For multi-token generation, TRTLLM-Gen applies causality by
        // shrinking each token's effective KV length.
        tllmRunnerParams.mMaskType = TrtllmGenAttentionMaskType::Dense;
        tllmRunnerParams.mKernelType = FmhaKernelType::Generation;
        tllmRunnerParams.mMultiCtasKvMode = mMultiBlockMode;
        // Note that the tileScheduler and multiCtasKvMode will be automatically tuned when using multi_block mode.
        // Otherwise, always enable the persistent scheduler for better performance.
        tllmRunnerParams.mTileScheduler = mMultiBlockMode ? TileScheduler::Static : TileScheduler::Persistent;

        // Q buffer.
        tllmRunnerParams.qPtr = mFP8GenerationMLA ? reinterpret_cast<void const*>(params.quant_q_buf)
                                                  : reinterpret_cast<void const*>(params.q_buf);

        // KV buffer
        // Paged KV
        tllmRunnerParams.mQkvLayout = QkvLayout::PagedKv;
        tllmRunnerParams.kvPtr = kv_cache_buffer.mPrimaryPoolPtr;
        tllmRunnerParams.kvPageIdxPtr = reinterpret_cast<KVCacheIndex::UnderlyingType const*>(kv_cache_buffer.data);
        tllmRunnerParams.mMaxNumPagesPerSeqKv = kv_cache_buffer.mMaxBlocksPerSeq;
        tllmRunnerParams.mNumTokensPerPage = kv_cache_buffer.mTokensPerBlock;

        // The partial buffers' pointers when the multiCtasKv mode is enabled.
        tllmRunnerParams.multiCtasKvCounterPtr = p.getSemaphores();
        tllmRunnerParams.multiCtasKvScratchPtr = scratchPtr;

        // The sequence lengths for K/V.
        tllmRunnerParams.seqLensKvPtr = params.cache_seq_lens;

        tllmRunnerParams.oPtr = reinterpret_cast<void*>(params.context_buf);
        tllmRunnerParams.oSfPtr = p.getOutputSf();
        if (params.dsv4_epilogue_fusion.enabled)
        {
            tllmRunnerParams.mDsv4EpilogueFusion.enabled = true;
            tllmRunnerParams.mDsv4EpilogueFusion.cosSinCache = params.dsv4_epilogue_fusion.cos_sin_cache;
            tllmRunnerParams.mDsv4EpilogueFusion.scaleBufM = params.dsv4_epilogue_fusion.scale_buf_m;
        }

        // softmax stats if needed
        tllmRunnerParams.softmaxStatsPtr = p.getSoftmaxStatsTensor();

        // Per-head attention sink added to the softmax denominator.
        tllmRunnerParams.attentionSinksPtr = p.getAttentionSinks();

        // MLA uses different head dimensions for Qk and V.
        tllmRunnerParams.mHeadDimQk = head_size;
        tllmRunnerParams.mHeadDimV = head_size_v;

        auto const num_q_heads = mNumAttnHeads;
        tllmRunnerParams.mNumHeadsQ = num_q_heads;
        tllmRunnerParams.mNumHeadsKv = num_kv_heads;
        tllmRunnerParams.mNumHeadsQPerKv = num_q_heads / num_kv_heads;

        tllmRunnerParams.mBatchSize = batch_beam;
        // It is used to construct contiguous kv cache TMA descriptors.
        tllmRunnerParams.mMaxSeqLenCacheKv = p.max_attention_window_size;
        // This should be set to numDraftTokens + 1.
        tllmRunnerParams.mMaxSeqLenQ = params.acc_q_len / batch_beam;
        tllmRunnerParams.mMaxSeqLenKv = p.max_past_kv_length;
        tllmRunnerParams.mJITWarmup = p.trtllm_gen_jit_warmup;
        tllmRunnerParams.mJITWarmupMaxNumRequests = p.max_num_requests;
        tllmRunnerParams.mJITWarmupMaxSeqLenQ = p.max_context_length;
        tllmRunnerParams.mJITWarmupMaxSeqLenKv = p.max_seq_len;
        tllmRunnerParams.mSumOfSeqLensQ = int(batch_beam * tllmRunnerParams.mMaxSeqLenQ);
        // Not used in the generation kernels as contiguous_kv or paged_kv layouts are used.
        tllmRunnerParams.mSumOfSeqLensKv = int(batch_beam * tllmRunnerParams.mMaxSeqLenKv);

        // The attention window size.
        tllmRunnerParams.mAttentionWindowSize = p.cyclic_attention_window_size;
        // The chunked attention size.
        tllmRunnerParams.mChunkedAttentionSize = INT_MAX;

        // The scaleQ that will be applied to the BMM1 output.
        tllmRunnerParams.mScaleQ = p.q_scaling
            * sqrt((float) (p.mla_params.qk_nope_head_dim + p.mla_params.qk_rope_head_dim))
            / sqrtf((float) (p.mla_params.kv_lora_rank + p.mla_params.qk_rope_head_dim));

        // Set it to INT_MAX as the kv cache pageOffsets will ensure that there is no out-of-bounds access.
        tllmRunnerParams.mNumPagesInMemPool = INT_MAX;
        tllmRunnerParams.mMultiProcessorCount = mMultiProcessorCount;
        tllmRunnerParams.stream = stream;
        tllmRunnerParams.mSfStartTokenIdx = p.token_offset;

        // Scales for quantization
        if (mFP8GenerationMLA)
        {
            static constexpr int bmm1_scale_offset = 1;
            tllmRunnerParams.outputScalePtr = reinterpret_cast<float const*>(params.bmm2_scale);
            tllmRunnerParams.scaleSoftmaxLog2Ptr
                = reinterpret_cast<float const*>(params.bmm1_scale) + bmm1_scale_offset;
        }

        // Set the following parameters if sparseAttention is used.
        if (useSparseMLA(p))
        {
            bool const useDynamicSparseMLA = p.sparse_params.sparse_attn_kv_lens != nullptr;
            tllmRunnerParams.mSparseAttention
                = useDynamicSparseMLA ? SparseType::DynamicTokenSparse : SparseType::StaticTokenSparse;
            tllmRunnerParams.mSkipCorrThreshold = mSkipCorrectionThreshold;
            tllmRunnerParams.mSparseTopK = p.sparse_params.num_sparse_topk;
            tllmRunnerParams.ptrSparseMlaTopKLens = p.sparse_params.sparse_attn_kv_lens;
            tllmRunnerParams.kvPageIdxPtr
                = reinterpret_cast<KVCacheIndex::UnderlyingType const*>(p.sparse_params.sparse_attn_indices);
            if (useDynamicSparseMLA)
            {
                TLLM_CHECK_WITH_INFO(p.sparse_params.sliding_window_kv_cache_pool != nullptr,
                    "SWA KV pool must be set for dynamic sparse MLA.");
                // Dynamic sparse MLA always has an SWA pool. The compressed pool is optional; when it
                // is absent (ratio == 1), use SWA as kvPtr only to keep TG's primary TMA descriptor valid.
                tllmRunnerParams.kvPtr = p.sparse_params.sparse_kv_cache_pool != nullptr
                    ? p.sparse_params.sparse_kv_cache_pool
                    : p.sparse_params.sliding_window_kv_cache_pool;
                tllmRunnerParams.slidingWindowKvPoolBasePtr = p.sparse_params.sliding_window_kv_cache_pool;
            }
            else
            {
                tllmRunnerParams.kvPtr = p.sparse_params.sparse_kv_cache_pool;
            }

            bool const usesAuxiliaryKvPool
                = tllmRunnerParams.kvPtr != nullptr && tllmRunnerParams.kvPtr != kv_cache_buffer.mPrimaryPoolPtr;
            if (usesAuxiliaryKvPool)
            {
                // Static sparse MLA indexes a compact KV pool containing at most
                // mSparseTopK rows per query. Do not let the original dense KV
                // length drive kernel selection or launch geometry: for long
                // sequences that can select a multi-CTA kernel which addresses
                // beyond the compact page table.
                TLLM_CHECK_WITH_INFO(tllmRunnerParams.mSparseTopK > 0,
                    "Static sparse MLA requires a positive TopK, got %d", tllmRunnerParams.mSparseTopK);
                int32_t const originalMaxSeqLenKv = tllmRunnerParams.mMaxSeqLenKv;
                int32_t const effectiveMaxSeqLenKv = std::min(originalMaxSeqLenKv, tllmRunnerParams.mSparseTopK);
                tllmRunnerParams.mMaxSeqLenKv = effectiveMaxSeqLenKv;
                tllmRunnerParams.mJITWarmupMaxSeqLenKv
                    = std::min(tllmRunnerParams.mJITWarmupMaxSeqLenKv, effectiveMaxSeqLenKv);
                int64_t const sumOfSeqLensKv = static_cast<int64_t>(tllmRunnerParams.mBatchSize) * effectiveMaxSeqLenKv;
                TLLM_CHECK_WITH_INFO(sumOfSeqLensKv <= std::numeric_limits<int32_t>::max(),
                    "Static sparse MLA cumulative KV length exceeds int32 capacity: %ld", sumOfSeqLensKv);
                tllmRunnerParams.mSumOfSeqLensKv = static_cast<int32_t>(sumOfSeqLensKv);
                TLLM_LOG_DEBUG("Clamp static sparse MLA max KV length from %d to %d (TopK=%d)", originalMaxSeqLenKv,
                    effectiveMaxSeqLenKv, tllmRunnerParams.mSparseTopK);
            }
        }

        mTllmGenFMHARunner->run(tllmRunnerParams);
        sync_check_cuda_error(stream);
    }
    else if (mUseGenFlashMLA)
    {
        static constexpr int TileSchedulerMetaDataSize = 8;

        int const num_q_heads = p.num_heads;
        int const ngroups = num_q_heads / num_kv_heads;

        int const s_q = params.acc_q_len / batch_beam;
        assert(s_q == p.mla_params.predicted_tokens_per_seq);
        int const head_size_v = p.mla_params.kv_lora_rank;
        int const num_sm_parts = getFlashMlaNumSmParts(s_q, num_q_heads, num_kv_heads, head_size_v);

        size_t const num_splits_size = sizeof(int) * (batch_beam + 1);
        size_t const tile_scheduler_metadata_size = sizeof(int) * (num_sm_parts * TileSchedulerMetaDataSize);
        size_t const softmax_lse_size = sizeof(float) * (batch_beam * s_q * num_q_heads * num_kv_heads); // softmax_lse
        size_t const softmax_lse_accum_size = sizeof(float) * ((batch_beam + num_sm_parts) * num_q_heads * s_q);
        size_t const out_accum_size = sizeof(float) * ((batch_beam + num_sm_parts) * num_q_heads * s_q * head_size_v);

        AttentionFlashMlaWorkspaceSizes flashMlaWorkspaceSizes{};
        flashMlaWorkspaceSizes.tileSchedulerMetadata = tile_scheduler_metadata_size;
        flashMlaWorkspaceSizes.numSplits = num_splits_size;
        flashMlaWorkspaceSizes.softmaxLse = softmax_lse_size;
        flashMlaWorkspaceSizes.softmaxLseAccum = softmax_lse_accum_size;
        flashMlaWorkspaceSizes.outAccum = out_accum_size;
        auto const flashMlaWorkspaceLayout = AttentionWorkspaceManager::buildFlashMlaLayout(flashMlaWorkspaceSizes);
        float* softmax_lse_ptr
            = AttentionWorkspaceManager::ptr<float>(params.workspace, flashMlaWorkspaceLayout.softmaxLse);
        float* softmax_lse_accum_ptr
            = AttentionWorkspaceManager::ptr<float>(params.workspace, flashMlaWorkspaceLayout.softmaxLseAccum);
        float* out_accum_ptr
            = AttentionWorkspaceManager::ptr<float>(params.workspace, flashMlaWorkspaceLayout.outAccum);

        // Metadata must always be pre-computed by Python (compute_flash_mla_metadata) and passed in.
        TLLM_CHECK_WITH_INFO(params.flash_mla_tile_scheduler_metadata != nullptr,
            "FlashMLA tile-scheduler metadata must be pre-computed by Python.");
        TLLM_CHECK_WITH_INFO(
            params.flash_mla_num_splits != nullptr, "FlashMLA num_splits must be pre-computed by Python.");
        int* tile_scheduler_metadata_ptr = const_cast<int*>(params.flash_mla_tile_scheduler_metadata);
        int* num_splits_ptr = const_cast<int*>(params.flash_mla_num_splits);

        Flash_fwd_mla_params flashMlaParams{};
        flashMlaParams.b = batch_beam;
        flashMlaParams.seqlen_q = ngroups * s_q;
        flashMlaParams.cu_seqlens_k = const_cast<int*>(params.cache_seq_lens);
        flashMlaParams.h = 1;
        flashMlaParams.h_h_k_ratio = 1;

        float softmax_scale
            = 1.0f / (p.q_scaling * sqrtf((p.mla_params.qk_nope_head_dim + p.mla_params.qk_rope_head_dim) * 1.0f));

        flashMlaParams.ngroups = ngroups;
        flashMlaParams.is_causal = !(s_q == 1);
        flashMlaParams.d = head_size;
        flashMlaParams.d_v = head_size_v;
        flashMlaParams.scale_softmax = softmax_scale;
        flashMlaParams.scale_softmax_log2 = float(softmax_scale * M_LOG2E);

        flashMlaParams.q_ptr = mFP8GenerationMLA ? const_cast<void*>(reinterpret_cast<void const*>(params.quant_q_buf))
                                                 : const_cast<void*>(reinterpret_cast<void const*>(params.q_buf));
        flashMlaParams.k_ptr = kv_cache_buffer.mPrimaryPoolPtr;
        flashMlaParams.v_ptr = flashMlaParams.k_ptr;
        flashMlaParams.o_ptr = reinterpret_cast<void*>(params.context_buf);
        flashMlaParams.softmax_lse_ptr = softmax_lse_ptr;

        // since head_num_kv = 1
        flashMlaParams.q_batch_stride = head_size * params.head_num * s_q;
        flashMlaParams.k_batch_stride = p.tokens_per_block * num_kv_heads * head_size * p.mla_params.num_layers;
        flashMlaParams.o_batch_stride = s_q * num_q_heads * head_size_v;
        flashMlaParams.q_row_stride = head_size;
        flashMlaParams.k_row_stride = head_size;
        flashMlaParams.o_row_stride = head_size_v;
        flashMlaParams.q_head_stride = head_size;
        flashMlaParams.k_head_stride = head_size;
        flashMlaParams.o_head_stride = head_size_v;

        flashMlaParams.v_batch_stride = flashMlaParams.k_batch_stride;
        flashMlaParams.v_row_stride = flashMlaParams.k_row_stride;
        flashMlaParams.v_head_stride = flashMlaParams.k_head_stride;

        flashMlaParams.block_table = const_cast<int*>(params.block_ids_per_seq);
        flashMlaParams.block_table_batch_stride = p.max_blocks_per_sequence;
        flashMlaParams.page_block_size = p.tokens_per_block;

        flashMlaParams.descale_q_ptr = const_cast<float*>(params.dequant_scale_q);
        flashMlaParams.descale_k_ptr = const_cast<float*>(params.dequant_scale_kv);

        flashMlaParams.tile_scheduler_metadata_ptr = tile_scheduler_metadata_ptr;
        flashMlaParams.num_sm_parts = num_sm_parts;
        flashMlaParams.num_splits_ptr = num_splits_ptr;

        flashMlaParams.softmax_lseaccum_ptr = softmax_lse_accum_ptr;
        flashMlaParams.oaccum_ptr = out_accum_ptr;

        if constexpr (std::is_same<T, half>::value)
        {
            if (mFP8GenerationMLA)
            {
                TLLM_THROW("FP8 KV cache MLA is only supported for bf16 output");
            }
            else
            {
                run_mha_fwd_splitkv_mla<cutlass::half_t, cutlass::half_t, 576>(flashMlaParams, stream);
            }
        }
        else if constexpr (std::is_same<T, __nv_bfloat16>::value)
        {
            if (mFP8GenerationMLA)
            {
                run_mha_fwd_splitkv_mla<cutlass::float_e4m3_t, cutlass::bfloat16_t, 576>(flashMlaParams, stream);
            }
            else
            {
                run_mha_fwd_splitkv_mla<cutlass::bfloat16_t, cutlass::bfloat16_t, 576>(flashMlaParams, stream);
            }
        }
        else
        {
            TLLM_THROW("Unsupported data type for FlashMLA");
        }
    }
    else
    {
        // Try XQA optimization first.
        // NOTE: input_seq_length = num_medusa_tokens + 1 (new generated one from the original LM head)
        // self attn
        XQAParams xqaParams{};
        this->template convertMMHAParamsToXQAParams<T, decltype(kv_cache_buffer)>(
            xqaParams, p, /*forConfigurePlugin=*/false);
        xqaParams.quant_q_buffer_ptr = params.quant_q_buf;
        xqaParams.q_scaling
            = 1 / (p.q_scaling * sqrtf((float) (p.mla_params.qk_nope_head_dim + p.mla_params.qk_rope_head_dim)));
        if (mEnableXQA && mXqaDispatcher->shouldUse(xqaParams))
        {
            TLLM_LOG_DEBUG("XQA kernels are selected in the generation phase.");
            xqaParams.stream = stream;
            mXqaDispatcher->run(xqaParams, kv_cache_buffer, kv_scale_cache_buffer);
            return 0;
        }

        // Use FMHA otherwise.
        MHARunnerParams fmhaParams{};
        fmhaParams.b = batch_beam;
        fmhaParams.numGroupedHeads = params.head_num;
        fmhaParams.qSeqLen = params.head_num * (params.acc_q_len / batch_beam);
        fmhaParams.kvSeqLen = p.max_past_kv_length;
        // Disable sliding window attention when it is not needed.
        fmhaParams.slidingWindowSize = p.cyclic_attention_window_size;
        fmhaParams.totalQSeqLen = batch_beam * fmhaParams.qSeqLen;
        // TODO: set it correctly for contiguous kv buffer (cross-attention).
        // fmhaParams.totalKvSeqLen = params.num_tokens;
        // Device buffer pointers.
        // fmhaParams.qkvPtr = reinterpret_cast<void const*>(params.attention_input);
        fmhaParams.qPtr = mFP8GenerationMLA ? reinterpret_cast<void const*>(params.quant_q_buf)
                                            : reinterpret_cast<void const*>(params.q_buf);
        // TODO: add contiguous kv buffer (cross-attention).
        fmhaParams.kvPtr = nullptr;

        fmhaParams.outputPtr = reinterpret_cast<void*>(params.context_buf);

        // fmhaParams.packedMaskPtr = params.fmha_custom_mask;
        fmhaParams.pagedKvCache = kv_cache_buffer;
        fmhaParams.cuQSeqLenPtr = params.seqQOffset;
        fmhaParams.kvSeqLenPtr = params.cache_seq_lens;
        fmhaParams.cuKvSeqLenPtr = params.cu_kv_seqlens;
        fmhaParams.cuMaskRowsPtr = nullptr; // mla not support custorm mask right now
        fmhaParams.tileCounterPtr = params.fmha_tile_counter;
        fmhaParams.scaleBmm1Ptr = reinterpret_cast<float const*>(params.bmm1_scale);
        fmhaParams.scaleBmm2Ptr = reinterpret_cast<float const*>(params.bmm2_scale);
        fmhaParams.stream = stream;
        fmhaParams.forceFp32Acc = mFMHAForceFP32Acc;

        // Sparse attention parameters
        if (useSparseMLA(p))
        {
            fmhaParams.sparse_params = p.sparse_params;
        }

        // MLA does not support skip-softmax attention right now

        // Run the fmha kernel
        mDecoderFMHARunner->run(fmhaParams);
    }

    sync_check_cuda_error(stream);
    return 0;
}

#define MLA_FUNC_DEFINE(T)                                                                                             \
    template int AttentionOp::mlaGeneration<T>(MlaParams<T> & params, FmhaParams const& p, cudaStream_t stream);

MLA_FUNC_DEFINE(float)
MLA_FUNC_DEFINE(half)
#ifdef ENABLE_BF16
MLA_FUNC_DEFINE(__nv_bfloat16)
#endif

template <typename T, typename KVCacheBuffer>
int AttentionOp::enqueueContext(FmhaParams const& p, MlaParams<T>* mlaParam, cudaStream_t stream)
{
    int const headSize = getHeadSize();

    int const local_hidden_units_qo = p.num_heads * headSize;
    int const local_hidden_units_kv = mNumAttnKVHeads * headSize;
    PositionEmbeddingType const position_embedding_type = p.position_embedding_type;
    float const q_scaling = p.q_scaling;

    KVCacheBuffer kv_cache_buffer;
    KVCacheBuffer kv_scale_cache_buffer;

    auto sizePerToken = mNumAttnKVHeads * headSize * getKvCacheElemSizeInBits<T>(p) / 8 /*bits*/;

    if (useKVCache(p))
    {
        auto buffers = buildKvCacheBuffers<KVCacheBuffer>(p.num_seqs, p.getMaxBlocksPerSequence(), p.tokens_per_block,
            sizePerToken, p.cyclic_attention_window_size, p.max_cyclic_attention_window_size, p.sink_token_length,
            p.can_use_one_more_block, p.getHostPrimaryPoolPtr(), p.getHostSecondaryPoolPtr(),
            p.getHostPrimaryBlockScalePoolPtr(), p.getHostSecondaryBlockScalePoolPtr(),
            p.getKvCacheBlockOffsets(p.getKvCachePoolIndex(p.local_layer_idx)), p.quant_mode.hasFp4KvCache(),
            isCrossAttention(p) ? p.cross_kv_length : p.max_attention_window_size, p.getKeyValueCache());
        kv_cache_buffer = buffers.kvCacheBuffer;
        kv_scale_cache_buffer = buffers.kvScaleCacheBuffer;
    }

    auto cublasHandle = mCublasWrapper->getCublasHandle();
    TLLM_CUDA_CHECK(cublasSetStream(cublasHandle, stream));
    mCublasWrapper->setStream(stream);
    mCublasWrapper->setWorkspace(p.getWorkspace());
    if constexpr (std::is_same_v<T, half>)
    {
        mCublasWrapper->setFP16GemmConfig();
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        mCublasWrapper->setFP32GemmConfig();
    }
#ifdef ENABLE_BF16
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        mCublasWrapper->setBF16GemmConfig();
    }
#endif

    size_t const kv_seq_length = (isCrossAttention(p) ? p.cross_kv_length : p.input_seq_length);
    size_t const attention_mask_size
        = mEnableContextFMHA ? 0 : sizeof(T) * p.num_seqs * p.input_seq_length * kv_seq_length;
    size_t const cu_seqlens_size = sizeof(int) * (p.num_seqs + 1);
    size_t const rotary_inv_freq_size = sizeof(float) * p.num_seqs * p.rotary_embedding_dim / 2;
    size_t q_buf_2_size = 0;
    if (!mEnableContextFMHA)
    {
        // Unfused mha
        q_buf_2_size = sizeof(T) * p.num_seqs * p.input_seq_length * local_hidden_units_qo;
    }
    else if (mFmhaDispatcher->isSeparateQAndKvInput())
    {
        // Paged context fmha
        q_buf_2_size = (mFP8ContextFMHA ? 1 : sizeof(T)) * p.num_tokens * local_hidden_units_qo;
    }

    size_t const k_buf_2_size = mEnableContextFMHA ? 0 : sizeof(T) * p.num_seqs * kv_seq_length * local_hidden_units_kv;
    size_t const v_buf_2_size = mEnableContextFMHA ? 0 : sizeof(T) * p.num_seqs * kv_seq_length * local_hidden_units_kv;
    size_t const qk_buf_size
        = mEnableContextFMHA ? 0 : sizeof(T) * p.num_seqs * p.num_heads * p.input_seq_length * kv_seq_length;
    size_t const qkv_buf_2_size
        = mEnableContextFMHA ? 0 : sizeof(T) * p.num_seqs * p.input_seq_length * local_hidden_units_qo;
    size_t const qk_buf_float_size
        = mEnableContextFMHA ? 0 : sizeof(float) * p.num_seqs * p.num_heads * p.input_seq_length * kv_seq_length;
    int dim_q_per_head = (p.mla_params.qk_rope_head_dim + p.mla_params.qk_nope_head_dim);
    int dim_k_per_head = (p.mla_params.qk_rope_head_dim + p.mla_params.qk_nope_head_dim);
    int dim_v_per_head = (p.mla_params.v_head_dim);
    if (useSparseMLA(p))
    {
        dim_q_per_head = p.mla_params.kv_lora_rank + p.mla_params.qk_rope_head_dim;
        dim_k_per_head = p.mla_params.kv_lora_rank + p.mla_params.qk_rope_head_dim;
        dim_v_per_head = p.mla_params.rope_append ? p.mla_params.kv_lora_rank
                                                  : p.mla_params.kv_lora_rank + p.mla_params.qk_rope_head_dim;
    }

    // Total dimension per token across all heads for Q, K, and V components respectively
    int const total_q_dim_all_heads = mNumAttnHeads * dim_q_per_head;
    int const total_k_dim_all_heads
        = mNumAttnHeads * dim_k_per_head; // Assuming effective num_kv_heads = head_num for layout
    int const total_v_dim_all_heads
        = mNumAttnHeads * dim_v_per_head; // Assuming effective num_kv_heads = head_num for layout
    // Packed fp8 qkv buffer size for normal fp8 context FMHA
    size_t fp8_qkv_buffer_size = mEnableContextFMHA && mFP8ContextFMHA && !mFmhaDispatcher->isSeparateQAndKvInput()
        ? p.num_tokens * (local_hidden_units_qo + 2 * local_hidden_units_kv)
        : 0;
    // Separate fp8 q/k/v buffer size for fp8 context MLA
    size_t fp8_q_buf_size = 0;
    size_t fp8_k_buf_size = 0;
    size_t fp8_v_buf_size = 0;
    bool const useSageAttnSeparateQkv = mEnableContextFMHA && !p.is_mla_enable
        && mFmhaDispatcher->isSeparateQAndKvInput()
        && (p.fwd.sage_attn_num_elts_per_blk_q > 0 || p.fwd.sage_attn_num_elts_per_blk_k > 0
            || p.fwd.sage_attn_num_elts_per_blk_v > 0);
    if (mEnableContextFMHA && mFP8ContextMLA && mFmhaDispatcher->isSeparateQAndKvInput())
    {
        fp8_q_buf_size = p.num_tokens * static_cast<size_t>(total_q_dim_all_heads);

        if (useSparseMLA(p))
        {
            // Sparse MLA (absorption mode): K and V are stored directly in KV cache during MLA RoPE kernel.
            // No separate FP8 buffers needed for K/V since they're read from paged KV cache (Q_PAGED_KV layout).
            fp8_k_buf_size = 0;
            fp8_v_buf_size = 0;
        }
        else
        {
            fp8_k_buf_size = p.total_kv_len * static_cast<size_t>(total_k_dim_all_heads);
            fp8_v_buf_size = p.total_kv_len * static_cast<size_t>(total_v_dim_all_heads);
        }
    }
    else if (useSageAttnSeparateQkv)
    {
        fp8_q_buf_size = p.num_tokens * static_cast<size_t>(local_hidden_units_qo);
        fp8_k_buf_size = p.total_kv_len * static_cast<size_t>(local_hidden_units_kv);
        fp8_v_buf_size = p.total_kv_len * static_cast<size_t>(local_hidden_units_kv);
    }

    int const q_max_n_blk = p.fwd.sage_attn_num_elts_per_blk_q > 0
        ? static_cast<int>(tc::divUp(p.num_tokens, p.fwd.sage_attn_num_elts_per_blk_q) + p.num_seqs - 1)
        : 0;
    int const k_max_n_blk = p.fwd.sage_attn_num_elts_per_blk_k > 0
        ? static_cast<int>(tc::divUp(p.total_kv_len, p.fwd.sage_attn_num_elts_per_blk_k) + p.num_seqs - 1)
        : 0;
    int const v_max_n_blk = p.fwd.sage_attn_num_elts_per_blk_v > 0
        ? static_cast<int>(tc::divUp(local_hidden_units_kv, p.fwd.sage_attn_num_elts_per_blk_v))
        : 0;
    size_t const sage_q_sfs_buffer_size = sizeof(float) * mNumAttnHeads * static_cast<size_t>(q_max_n_blk);
    size_t const sage_k_sfs_buffer_size = sizeof(float) * mNumAttnKVHeads * static_cast<size_t>(k_max_n_blk);
    size_t const sage_v_sfs_buffer_size = sizeof(float) * v_max_n_blk;

    size_t const padding_offset_size = mEnableContextFMHA ? 0 : sizeof(int) * p.num_seqs * p.input_seq_length;
    size_t const encoder_padding_offset_size = mEnableContextFMHA ? 0 : sizeof(int) * p.num_seqs * p.cross_kv_length;
    // Each token holds (batch_idx, token_idx_in_seq) int2.
    size_t const tokens_info_size = sizeof(int2) * p.num_tokens;
    size_t const fmha_scheduler_counter = mEnableContextFMHA ? sizeof(uint32_t) : 0;
    size_t const fmha_bmm1_scale_size = (mFP8ContextFMHA || mFP8ContextMLA) ? sizeof(float) * 2 : 0;
    size_t const fmha_bmm2_scale_size = (mFP8ContextFMHA || mFP8ContextMLA) ? sizeof(float) : 0;

    size_t const fmha_multi_ctas_kv_scratch_size = useTllmGenSparseAttention(p) ? getFmhaMultiCtasKvScratchSize(p) : 0;

    bool const is_qk_buf_float_ = true;

    AttentionContextWorkspaceSizes workspaceSizes{};
    workspaceSizes.attentionMask = attention_mask_size;
    workspaceSizes.cuQSeqlens = cu_seqlens_size;
    workspaceSizes.cuKvSeqlens = cu_seqlens_size;
    workspaceSizes.cuMaskRows = cu_seqlens_size;
    workspaceSizes.rotaryInvFreq = rotary_inv_freq_size;
    workspaceSizes.qBuf = q_buf_2_size;
    workspaceSizes.kBuf = k_buf_2_size;
    workspaceSizes.vBuf = v_buf_2_size;
    workspaceSizes.qkBuf = qk_buf_size;
    workspaceSizes.qkvBuf = qkv_buf_2_size;
    workspaceSizes.qkFloatBuf = qk_buf_float_size;
    workspaceSizes.fp8QkvBuf = fp8_qkv_buffer_size;
    workspaceSizes.fp8QBuf = fp8_q_buf_size;
    workspaceSizes.fp8KBuf = fp8_k_buf_size;
    workspaceSizes.fp8VBuf = fp8_v_buf_size;
    workspaceSizes.paddingOffset = padding_offset_size;
    workspaceSizes.encoderPaddingOffset = encoder_padding_offset_size;
    workspaceSizes.tokensInfo = tokens_info_size;
    workspaceSizes.fmhaTileCounter = fmha_scheduler_counter;
    workspaceSizes.fmhaBmm1Scale = fmha_bmm1_scale_size;
    workspaceSizes.fmhaBmm2Scale = fmha_bmm2_scale_size;
    workspaceSizes.sageQScale = sage_q_sfs_buffer_size;
    workspaceSizes.sageKScale = sage_k_sfs_buffer_size;
    workspaceSizes.sageVScale = sage_v_sfs_buffer_size;
    workspaceSizes.fmhaMultiCtasKvScratch = fmha_multi_ctas_kv_scratch_size;
    auto const workspaceLayout = AttentionWorkspaceManager::buildContextLayout(workspaceSizes);
    auto const workspaceViews = AttentionWorkspaceManager::materializeContext<T>(p.getWorkspace(), workspaceLayout);

    auto* fp8QBuf = workspaceViews.fp8QBuf;
    // Fused FP8-Q path: caller pre-fills the nope segment of `quant_q_buf`;
    // route the context-MLA Q pointer to it so the fused RoPE kernel appends
    // rope FP8 in place and the FMHA Q load reads the merged [nope|rope] buffer.
    if (p.is_mla_enable && p.fwd.quant_q_buffer.has_value() && p.fwd.quant_scale_qkv.has_value()
        && p.getQuantQBuffer() != nullptr)
    {
        fp8QBuf = reinterpret_cast<__nv_fp8_e4m3*>(p.getQuantQBuffer());
    }

    // build attention mask, cu_seqlens, and padding offset tensors
    // Note: self attn and cross attn should use different p
    // cross attn's seqlen info is from encoder input lengths, not decoder input lengths!
    // moreover, attn mask for cross attn should be set separately (see below)
    BuildDecoderInfoParams<T> decoder_params{};
    int32_t const* precomputedCuQSeqlens = p.getCuQSeqlens();
    int32_t const* precomputedCuKvSeqlens = p.getCuKvSeqlens() != nullptr ? p.getCuKvSeqlens()
        : p.getCuQSeqlens() != nullptr                                    ? p.getCuQSeqlens()
                                                                          : nullptr;
    decoder_params.seqQOffsets = workspaceViews.cuQSeqlens;
    decoder_params.seqKVOffsets = workspaceViews.cuKvSeqlens;
    decoder_params.precomputedSeqQOffsets = precomputedCuQSeqlens;
    decoder_params.precomputedSeqKVOffsets = precomputedCuKvSeqlens;
    decoder_params.seqCpPartialOffsets = nullptr;
    decoder_params.cpSize = 1;
    decoder_params.packedMaskRowOffsets = workspaceViews.cuMaskRows;
    decoder_params.paddingOffsets = workspaceViews.paddingOffset;
    decoder_params.tokensInfo = workspaceViews.tokensInfo;
    // Cross attention takes offsets from encoder inputs.
    decoder_params.encoderPaddingOffsets = isCrossAttention(p) ? workspaceViews.encoderPaddingOffset : nullptr;
    // Manually set attention mask for unfused cross attention.
    decoder_params.attentionMask = isCrossAttention(p) ? nullptr : workspaceViews.attentionMask;
    // Fixed sequence length offset if not removing the padding (seqQOffsets[i] = i * seq_length).
    decoder_params.seqQLengths = p.getContextLengths();
    decoder_params.seqKVLengths = isCrossAttention(p) ? p.getEncoderInputLengths() : p.getSequenceLength();
    decoder_params.batchSize = p.num_seqs;
    decoder_params.maxQSeqLength = p.input_seq_length;
    decoder_params.maxEncoderQSeqLength
        = isCrossAttention(p) ? p.cross_kv_length : 0; // cross attention uses encoder seq length
    decoder_params.attentionWindowSize = p.cyclic_attention_window_size;
    decoder_params.sinkTokenLength = p.sink_token_length;
    decoder_params.numTokens = p.num_tokens;
    decoder_params.removePadding = p.remove_padding;
    decoder_params.attentionMaskType = p.mask_type;
    decoder_params.blockSparseParams = p.block_sparse_params;
    decoder_params.fmhaTileCounter = workspaceViews.fmhaTileCounter;
    decoder_params.quantScaleO = p.getOutScale();
    decoder_params.dequantScaleQkv = p.getKvScaleQuantOrig();
    decoder_params.separateQkvScales = p.quant_mode.hasFp4KvCache();
    decoder_params.fmhaHostBmm1Scale = 1.0f / (sqrtf(getHeadSize() * 1.0f) * q_scaling);
    decoder_params.fmhaBmm1Scale = workspaceViews.fmhaBmm1Scale;
    decoder_params.fmhaBmm2Scale = workspaceViews.fmhaBmm2Scale;
    // Rotary embedding inv_freq buffer.
    decoder_params.rotaryEmbeddingScale = p.rotary_embedding_scale;
    decoder_params.rotaryEmbeddingBase = p.rotary_embedding_base;
    decoder_params.rotaryEmbeddingDim = p.rotary_embedding_dim;
    decoder_params.rotaryScalingType = p.rotary_embedding_scale_type;
    // The inv freq might be updated during runtime with dynamic scaling type.
    decoder_params.rotaryEmbeddingInvFreq = workspaceViews.rotaryInvFreq;
    // This is pre-computed when building the engines.
    decoder_params.rotaryEmbeddingInvFreqCache = p.getRotaryInvFreq();
    decoder_params.rotaryEmbeddingMaxPositions = p.rotary_embedding_max_positions;

    invokeBuildDecoderInfo(decoder_params, stream);
    sync_check_cuda_error(stream);

    int32_t const* contextCuQSeqlens
        = precomputedCuQSeqlens != nullptr ? precomputedCuQSeqlens : workspaceViews.cuQSeqlens;
    int32_t const* contextCuKvSeqlens
        = precomputedCuKvSeqlens != nullptr ? precomputedCuKvSeqlens : workspaceViews.cuKvSeqlens;

    // In cross attention context phase, the attention mask should be a matrix of all ones.
    // Override the attention mask produced by invokeBuildDecoderInfo().
    // also, invokeBuildDecoderInfo can only handle square mask, not cross B x q_len x kv_len mask
    // TODO: put this logic in the kernel above. currently not much concern because q_len is mostly = 1
    if (isUnfusedCrossAttention(p))
    {
        std::vector<T> h_attention_mask(p.num_seqs * p.input_seq_length * p.cross_kv_length, 1.);
        std::vector<int32_t> h_encoder_input_lengths(p.num_seqs);
        tensorrt_llm::common::cudaMemcpyAsyncSanitized(h_encoder_input_lengths.data(), p.getEncoderInputLengths(),
            sizeof(int32_t) * p.num_seqs, cudaMemcpyDeviceToHost, stream);
        sync_check_cuda_error(stream);

        for (int bi = 0; bi < p.num_seqs; bi++)
        {
            int b_offset = bi * p.input_seq_length * p.cross_kv_length;
            for (int qi = 0; qi < p.input_seq_length; qi++)
            {
                int q_offset = b_offset + qi * p.cross_kv_length;
                if (h_encoder_input_lengths[bi] < p.cross_kv_length)
                {
                    std::fill(h_attention_mask.begin() + q_offset + h_encoder_input_lengths[bi],
                        h_attention_mask.begin() + q_offset + p.cross_kv_length, 0.f);
                }
            }
        }
        cudaMemcpyAsync(workspaceViews.attentionMask, h_attention_mask.data(),
            sizeof(T) * p.num_seqs * p.cross_kv_length * p.input_seq_length, cudaMemcpyHostToDevice, stream);
        sync_check_cuda_error(stream);
    }

    // FIXME: a temporary solution to make sure the padding part is 0.
    if (!p.remove_padding)
    {
        cudaMemsetAsync(p.getOutput(), 0, p.num_tokens * local_hidden_units_qo * sizeof(T), stream);
        sync_check_cuda_error(stream);
    }

    KvCacheDataType cache_type = cacheTypeFromQuantMode(p.quant_mode);

    cudaDataType_t const gemm_data_type = tc::CudaDataType<T>::value;
    int const attention_seq_len_1 = p.input_seq_length;                                           // q length
    int const attention_seq_len_2 = isCrossAttention(p) ? p.cross_kv_length : p.input_seq_length; // kv length

    // If the model has relative attentiona bias, q scaling should be applied in QK gemm stage and use 1 in
    // softamax stage (because to get softmax[scale(Q*K) + rel pos bias] here, q_scaling can't be applied during
    // softmax phase by qk_scale); otherwise, use 1 in gemm stage and apply scaling in softmax stage
    float const qk_scale
        = 1.0f / (sqrtf(getHeadSize() * 1.0f) * q_scaling); // q_scaling in denominator. by default q_scaling =1.0f
    float const qk_scale_gemm = isRelativePosition(p) ? qk_scale : 1.0f;
    T const qk_scale_softmax = static_cast<T>(isRelativePosition(p) ? 1.0f : qk_scale);

    // in context phase, currently FMHA runner has two restrictions:
    // 1. only apply to self attention. If want fused multi-head cross attention, FMHCA kernels and runner is needed
    // 2. doesn't apply to MHA with relative attention bias, i.e. softmax(QK + bias) * V
    // We update mEnableContextFMHA in constructor to check these conditions
    if (mEnableContextFMHA)
    {
        T* attention_input = p.getQkvOrQ<T>();
        bool const enablePagedKVContextFMHA = mPagedKVCache && p.paged_context_fmha;
        TLLM_CHECK_WITH_INFO(!(p.quant_mode.hasInt8KvCache() && enablePagedKVContextFMHA),
            "Paged Context FMHA doesn't work with int8 kv cache currently.");
        TLLM_CHECK_WITH_INFO(!(p.sink_token_length > 0 && enablePagedKVContextFMHA),
            "Cannot support StreamingLLM now when enabling paged KV context FMHA.");

        // The max_kv_seq_len comes from the encoder seqlen when cross attention is used.
        int const max_kv_seq_len = isCrossAttention(p) ? p.cross_kv_length : p.max_past_kv_length;

        // Prepare QKV preprocessing parameters.
        QKVPreprocessingParams<T, KVCacheBuffer> preprocessingParams;

        // Buffers.
        preprocessingParams.qkv_input = const_cast<T*>(attention_input);
        preprocessingParams.cross_kv_input = p.getCrossKv<T>();
        preprocessingParams.quantized_qkv_output = workspaceViews.fp8QkvBuf;
        preprocessingParams.q_output = workspaceViews.qBuf;
        preprocessingParams.kv_cache_buffer = kv_cache_buffer;
        preprocessingParams.kv_cache_block_scales_buffer = kv_scale_cache_buffer;
        preprocessingParams.qkv_bias = p.getQkvBias<T>();
        preprocessingParams.tokens_info = decoder_params.tokensInfo;
        preprocessingParams.seq_lens = p.getContextLengths();
        // For self-attention, cache_seq_lens indicates whether chunked context is used
        // (i.e. cache_seq_len > seq_len).
        // For cross-attention, callers do not consistently use sequence_lengths as decoder length; use decoder
        // context lengths so the encoder KV-cache write gate opens.
        preprocessingParams.cache_seq_lens = isCrossAttention(p) ? p.getContextLengths() : p.getSequenceLength();

        preprocessingParams.encoder_seq_lens = p.getEncoderInputLengths();
        preprocessingParams.cu_seq_lens = contextCuQSeqlens;
        // Cross-attention only.
        preprocessingParams.cu_kv_seq_lens = contextCuKvSeqlens;
        preprocessingParams.rotary_embedding_inv_freq = workspaceViews.rotaryInvFreq;
        preprocessingParams.rotary_coef_cache_buffer = p.getRotaryCosSin();
        preprocessingParams.mrope_rotary_cos_sin = p.getMropeRotaryCosSin();
        preprocessingParams.qkv_scale_orig_quant = p.getKvScaleOrigQuant();
        preprocessingParams.spec_decoding_position_offsets = nullptr;
        preprocessingParams.helix_position_offsets = p.getHelixPositionOffsets();
        preprocessingParams.helix_is_inactive_rank = p.getHelixIsInactiveRank();
        preprocessingParams.logn_scaling = p.getLognScalingPtr();

        // Sparse KV write
        preprocessingParams.sparse_kv_indices = p.sparse_params.sparse_kv_indices;
        preprocessingParams.sparse_kv_offsets = p.sparse_params.sparse_kv_offsets;

        // Scalars
        preprocessingParams.batch_size = p.num_seqs;
        preprocessingParams.max_input_seq_len = p.input_seq_length;
        preprocessingParams.max_kv_seq_len = max_kv_seq_len;
        preprocessingParams.cyclic_kv_cache_len
            = isCrossAttention(p) ? p.cross_kv_length : p.cyclic_attention_window_size;
        preprocessingParams.sink_token_len = p.sink_token_length;
        preprocessingParams.token_num = p.num_tokens;
        preprocessingParams.remove_padding = p.remove_padding;
        preprocessingParams.cross_attention = isCrossAttention(p);
        preprocessingParams.head_num = mNumAttnHeads;
        preprocessingParams.kv_head_num = mNumAttnKVHeads;
        preprocessingParams.qheads_per_kv_head = mNumAttnHeads / mNumAttnKVHeads;
        preprocessingParams.size_per_head = getHeadSize();
        preprocessingParams.rotary_embedding_dim = p.rotary_embedding_dim;
        preprocessingParams.rotary_embedding_base = p.rotary_embedding_base;
        preprocessingParams.rotary_scale_type = p.rotary_embedding_scale_type;
        preprocessingParams.rotary_embedding_scale = p.rotary_embedding_scale;
        preprocessingParams.rotary_embedding_max_positions = p.rotary_embedding_max_positions;
        preprocessingParams.position_embedding_type = position_embedding_type;
        preprocessingParams.position_shift_enabled = p.pos_shift_enabled;
        preprocessingParams.cache_type = cache_type;
        preprocessingParams.separate_q_kv_output = enablePagedKVContextFMHA || isCrossAttention(p);
        preprocessingParams.quantized_fp8_output = mFP8ContextFMHA;
        preprocessingParams.generation_phase = false;
        preprocessingParams.multi_processor_count = mMultiProcessorCount;

        preprocessingParams.rotary_vision_start = p.vision_start;
        preprocessingParams.rotary_vision_length = p.vision_length;
        preprocessingParams.is_last_chunk
            = !p.attention_chunk_size.has_value() || (p.input_seq_length == p.max_past_kv_length);

        {
            std::string const beforeRopeStr = "ctx attention before RoPE at layer " + std::to_string(p.layer_idx);
            TLLM_CHECK_DEBUG_WITH_INFO(tensorrt_llm::runtime::utils::tensorHasInvalid(p.num_tokens,
                                           (local_hidden_units_qo + 2 * local_hidden_units_kv), p.type,
                                           const_cast<T*>(attention_input), stream, beforeRopeStr)
                    == false,
                "Found invalid number (NaN or Inf) in " + beforeRopeStr);
        }

        if (p.is_mla_enable)
        {
            TLLM_CHECK_WITH_INFO(mlaParam != nullptr, "MLA param is nullptr");
            mlaParam->cache_type = cache_type;
            mlaParam->cu_q_seqlens = const_cast<int*>(contextCuQSeqlens);
            mlaParam->cu_kv_seqlens = const_cast<int*>(contextCuKvSeqlens);
            mlaParam->quant_scale_kv = p.getKvScaleOrigQuant();
            // Set BMM scales for FP8 context computation
            mlaParam->bmm1_scale = workspaceViews.fmhaBmm1Scale;
            mlaParam->bmm2_scale = workspaceViews.fmhaBmm2Scale;
            mlaParam->quant_q_buf = mFP8ContextMLA ? fp8QBuf : nullptr;
            mlaParam->quant_k_buf = mFP8ContextMLA ? workspaceViews.fp8KBuf : nullptr;
            mlaParam->quant_v_buf = mFP8ContextMLA ? workspaceViews.fp8VBuf : nullptr;
            // Set additional scales for context phase
            mlaParam->quant_scale_o = p.getOutScale();
            mlaParam->quant_scale_q = p.getKvScaleOrigQuant();
            mlaParam->quant_scale_kv = p.getKvScaleOrigQuant();
            mlaParam->dequant_scale_q = p.getKvScaleQuantOrig();
            mlaParam->dequant_scale_kv = cache_type == KvCacheDataType::NVFP4 ? nullptr : p.getKvScaleQuantOrig();
            mlaParam->host_bmm1_scale
                = 1 / (p.q_scaling * sqrt((float) (p.mla_params.qk_nope_head_dim + p.mla_params.qk_rope_head_dim)));
            // The sparse MLA is in the absorption mode for the context phase.
            mlaParam->absorption_mode = useSparseMLA(p);
            // Fused FP8-Q-quant: RoPE kernel writes FP8 rope into `quant_q_buf`,
            // so we skip the standalone invokeMLAContextFp8Quantize call below.
            bool const useFusedQFp8 = mlaParam->fuse_q_fp8_in_rope && mFP8ContextMLA && mlaParam->absorption_mode
                && cache_type == KvCacheDataType::FP8 && mlaParam->quant_q_buf != nullptr
                && mlaParam->quant_scale_qkv != nullptr;
            TLLM_CHECK_WITH_INFO(cache_type != KvCacheDataType::NVFP4 || mlaParam->latent_cache == nullptr,
                "NVFP4 sparse MLA context must append its latent cache before launching attention");
            if (mlaParam->latent_cache != nullptr)
            {
                invokeMLARopeContext<T, KVCacheBuffer>(*mlaParam, kv_cache_buffer, stream);
            }
            if (mFP8ContextMLA && !useFusedQFp8)
            {
                invokeMLAContextFp8Quantize(*mlaParam, p.total_kv_len, stream);
            }
        }
        else if (useSageAttnSeparateQkv)
        {
            TLLM_CHECK_WITH_INFO(mFP8ContextFMHA, "SageAttention kernel runs under mFP8ContextFMHA option.");
            TLLM_CHECK_WITH_INFO(mFmhaDispatcher->isSupported(), "SageAttention has no unfused fallback implemented.");
            TLLM_CHECK_WITH_INFO(p.mask_type == AttentionMaskType::PADDING,
                "SageAttention only supports dense (padding) mask, got mask type %d.", static_cast<int>(p.mask_type));
            TLLM_CHECK_WITH_INFO(p.fwd.sage_attn_num_elts_per_blk_q > 0 && p.fwd.sage_attn_num_elts_per_blk_k > 0
                    && p.fwd.sage_attn_num_elts_per_blk_v == 1,
                "SageQuant requires positive block sizes for Q and K while the block size for V must be 1.");
            TLLM_CHECK_WITH_INFO(!p.fwd.kv_scale_quant_orig,
                "SageAttention disregards the configured p.fwd.kv_scale_quant_orig, invalidating the result.");
            check_cuda_error(cudaMemsetAsync(workspaceViews.sageVScale, 0, sage_v_sfs_buffer_size, stream));

            // Common p for sageQuant
            tc::SageQuantParams sageQuantParams{};
            sageQuantParams.headDim = getHeadSize();
            sageQuantParams.inputType = std::is_same_v<T, __nv_bfloat16> ? DATA_TYPE_BF16 : DATA_TYPE_FP16;
            sageQuantParams.quantType = p.fwd.sage_attn_qk_int8 ? DATA_TYPE_INT8 : DATA_TYPE_E4M3;
            sageQuantParams.vStage = 0;
            sageQuantParams.sumSeqLensV = p.total_kv_len;
            sageQuantParams.numHeadsV = mNumAttnKVHeads;
            sageQuantParams.ptrV = p.getV<T>();
            sageQuantParams.ptrVQuant = workspaceViews.fp8VBuf;
            sageQuantParams.ptrVScale = workspaceViews.sageVScale;
            sageQuantParams.smCount = mMultiProcessorCount;
            sageQuantParams.stream = stream;

            // Quantize into Fp8Q, SfsQ, SfsV
            sageQuantParams.sumSeqLensQk = p.num_tokens;
            sageQuantParams.batchSize = p.num_seqs;
            sageQuantParams.numHeads = mNumAttnHeads;
            sageQuantParams.tokenBlockSize = p.fwd.sage_attn_num_elts_per_blk_q;
            sageQuantParams.ptrCuSeqLensQk = contextCuQSeqlens;
            sageQuantParams.ptrQk = attention_input;
            sageQuantParams.ptrQkQuant = workspaceViews.fp8QBuf;
            sageQuantParams.ptrQkScale = workspaceViews.sageQScale;
            sageQuantParams.vStage = 1;
            tc::invokeSageQuant(sageQuantParams);

            // Quantize into Fp8K, SfsK, Fp8V
            sageQuantParams.sumSeqLensQk = p.total_kv_len;
            sageQuantParams.batchSize = p.num_seqs;
            sageQuantParams.numHeads = mNumAttnKVHeads;
            sageQuantParams.tokenBlockSize = p.fwd.sage_attn_num_elts_per_blk_k;
            sageQuantParams.ptrCuSeqLensQk = contextCuKvSeqlens;
            sageQuantParams.ptrQk = p.getK<T>();
            sageQuantParams.ptrQkQuant = workspaceViews.fp8KBuf;
            sageQuantParams.ptrQkScale = workspaceViews.sageKScale;
            sageQuantParams.vStage = 2;
            tc::invokeSageQuant(sageQuantParams);
        }
        else
        {
            invokeQKVPreprocessing(preprocessingParams, stream);
        }
        sync_check_cuda_error(stream);
        {
            std::string const afterRopeStr = "ctx attention after RoPE at layer " + std::to_string(p.layer_idx);
            TLLM_CHECK_DEBUG_WITH_INFO(tensorrt_llm::runtime::utils::tensorHasInvalid(p.num_tokens,
                                           (local_hidden_units_qo + 2 * local_hidden_units_kv), p.type,
                                           const_cast<T*>(attention_input), stream, afterRopeStr)
                    == false,
                "Found invalid number (NaN or Inf) in " + afterRopeStr);
            sync_check_cuda_error(stream);
        }

        if (p.runtime_perf_knobs.has_value())
        {
            int64_t const enable_context_fmha_fp32_acc_val = p.getRuntimePerfKnobs()[1];
            mFMHAForceFP32Acc = mFMHAForceFP32Acc || enable_context_fmha_fp32_acc_val == 1;
        }

        // Unified FMHA runner interface for both packed QKV FMHA, contiguous Q_KV, paged KV FMHA, and separate QKV
        // FMHA.
        // Page KV input layout:
        //    - q_ptr: [B, S, H, D], which supports variable sequence length
        //    - paged_kv_cache: paged kv buffer
        //    - cu_q_seqlens: the cumulative query sequence lengths, needed for variable sequence length.
        //    - cu_kv_seqlens: the cumulative kv sequence lengths, needed for variable sequence length.
        //
        // Contiguous KV input layout:
        //    - q_ptr: [B, S, H, D], which supports variable sequence length
        //    - kv_ptr: [B, S, 2, H, D], which supports variable sequence length
        //    - cu_q_seqlens: the cumulative query sequence lengths, needed for variable sequence length.
        //    - cu_kv_seqlens: the cumulative kv sequence lengths, needed for variable sequence length.
        //
        // Separate QKV input layout (only for context MLA now):
        //    - q_ptr: [B, S, H, D], which supports variable sequence length
        //    - k_ptr: [B, S, H_kv, D], which supports variable sequence length
        //    - v_ptr: [B, S, H_kv, D_v], which supports variable sequence length
        //    - cu_q_seqlens: the cumulative query sequence lengths, needed for variable sequence length.
        //    - cu_kv_seqlens: the cumulative kv sequence lengths, needed for variable sequence length.
        //    - total_kv_len: the total kv sequence length, needed for variable sequence length.

        // Construct the fmha p for running kernels.
        MHARunnerParams fmhaParams{};
        fmhaParams.b = p.num_seqs;
        fmhaParams.qSeqLen = p.input_seq_length;
        fmhaParams.kvSeqLen = max_kv_seq_len;
        // Disable sliding window attention when it is not needed.
        fmhaParams.slidingWindowSize
            = (p.dense_context_fmha || isCrossAttention(p)) ? max_kv_seq_len : p.cyclic_attention_window_size;
        fmhaParams.totalQSeqLen = p.num_tokens;
        // TODO: set it correctly for contiguous kv buffer (cross-attention).
        fmhaParams.totalKvSeqLen = isCrossAttention(p) ? p.num_encoder_tokens : p.total_kv_len;
        // Device buffer pointers.
        if (p.is_mla_enable)
        {
            // separate QKV input for context MLA
            if (mFP8ContextMLA)
            {
                TLLM_CHECK_WITH_INFO(
                    mFmhaDispatcher->isSeparateQAndKvInput(), "Separate QKV input is required for fp8 context MLA");
                TLLM_CHECK_WITH_INFO(fp8QBuf != nullptr, "FP8 q buffer is required for fp8 context MLA");
                // In sparse MLA (absorption mode), K and V are stored in KV cache, not as separate FP8 buffers
                TLLM_CHECK_WITH_INFO(useSparseMLA(p) || workspaceViews.fp8KBuf != nullptr,
                    "FP8 k buffer is required for fp8 context MLA in non-sparse mode");
                TLLM_CHECK_WITH_INFO(useSparseMLA(p) || workspaceViews.fp8VBuf != nullptr,
                    "FP8 v buffer is required for fp8 context MLA in non-sparse mode");

                fmhaParams.qPtr = reinterpret_cast<void const*>(fp8QBuf);
                fmhaParams.kPtr = useSparseMLA(p) ? nullptr : reinterpret_cast<void const*>(workspaceViews.fp8KBuf);
                fmhaParams.vPtr = useSparseMLA(p) ? nullptr : reinterpret_cast<void const*>(workspaceViews.fp8VBuf);
            }
            else
            {
                fmhaParams.qPtr = attention_input;
                fmhaParams.kPtr = p.getK<T>();
                fmhaParams.vPtr = p.getV<T>();
            }
        }
        else if (useSageAttnSeparateQkv)
        {
            // SageAttention: use quantized FP8/INT8 Q/K/V buffers as separate inputs.
            TLLM_CHECK_WITH_INFO(
                mFmhaDispatcher->isSeparateQAndKvInput(), "Separate QKV input is required for sage attention FMHA");
            fmhaParams.qkvPtr = nullptr;
            fmhaParams.qPtr = reinterpret_cast<void const*>(workspaceViews.fp8QBuf);
            fmhaParams.kPtr = reinterpret_cast<void const*>(workspaceViews.fp8KBuf);
            fmhaParams.vPtr = reinterpret_cast<void const*>(workspaceViews.fp8VBuf);
            // Set sage attention scaling factor pointers.
            fmhaParams.qScalePtr = workspaceViews.sageQScale;
            fmhaParams.kScalePtr = workspaceViews.sageKScale;
            fmhaParams.vScalePtr = workspaceViews.sageVScale;
        }
        else
        {
            fmhaParams.qkvPtr = mFP8ContextFMHA ? reinterpret_cast<void const*>(workspaceViews.fp8QkvBuf)
                                                : reinterpret_cast<void const*>(attention_input);
            fmhaParams.qPtr = reinterpret_cast<void const*>(workspaceViews.qBuf);
        }
        // TODO: add contiguous kv buffer (cross-attention).
        fmhaParams.kvPtr = nullptr;
        if (isCrossAttention(p) && !useKVCache(p))
        {
            fmhaParams.kvPtr = p.getCrossKv<T>();
        }
        // Only use [totalLength, h / cpSize, Dh].
        fmhaParams.outputPtr = p.getOutput();
        fmhaParams.outputSfPtr = p.getOutputSf();
        if (mlaParam != nullptr && mlaParam->dsv4_epilogue_fusion.enabled)
        {
            fmhaParams.dsv4EpilogueFusion.enabled = true;
            fmhaParams.dsv4EpilogueFusion.cosSinCache = mlaParam->dsv4_epilogue_fusion.cos_sin_cache;
            fmhaParams.dsv4EpilogueFusion.scaleBufM = mlaParam->dsv4_epilogue_fusion.scale_buf_m;
        }
        fmhaParams.attentionSinksPtr = p.getAttentionSinks();
        fmhaParams.packedMaskPtr = p.getAttentionPackedMask();
        if constexpr (std::is_same_v<KVCacheBuffer, KVBlockArray>)
        {
            fmhaParams.pagedKvCache = kv_cache_buffer;
            fmhaParams.pagedKvSfCache = kv_scale_cache_buffer;
        }
        fmhaParams.cuQSeqLenPtr = contextCuQSeqlens;
        fmhaParams.kvSeqLenPtr = decoder_params.seqKVLengths;
        fmhaParams.cuKvSeqLenPtr = contextCuKvSeqlens;
        fmhaParams.cuMaskRowsPtr = workspaceViews.cuMaskRows;
        fmhaParams.tileCounterPtr = workspaceViews.fmhaTileCounter;
        fmhaParams.scaleBmm1Ptr = workspaceViews.fmhaBmm1Scale;
        fmhaParams.scaleBmm2Ptr = workspaceViews.fmhaBmm2Scale;
        fmhaParams.oSfScalePtr = p.getOutSfScale();
        fmhaParams.stream = stream;
        fmhaParams.forceFp32Acc = mFMHAForceFP32Acc;
        fmhaParams.skipCorrectionThreshold = mSkipCorrectionThreshold;
        fmhaParams.softmaxStatsPtr = p.getSoftmaxStatsTensor();
        fmhaParams.trtllmGenJITWarmup = p.trtllm_gen_jit_warmup;
        fmhaParams.trtllmGenJITWarmupMaxNumRequests = p.max_num_requests;
        fmhaParams.trtllmGenJITWarmupMaxSeqLenQ = p.max_context_length;
        fmhaParams.trtllmGenJITWarmupMaxSeqLenKv = p.max_seq_len;

        // Sparse attention parameters
        if (useTllmGenSparseAttention(p))
        {
            fmhaParams.sparse_params = p.sparse_params;
            // Sparse context reuses generation-style trtllm-gen kernels; provide the scratch pool
            // and per-CTA counter so the autotuner can select MultiCtasKv variants.
            fmhaParams.multiCtasKvScratchPtr = workspaceViews.fmhaMultiCtasKvScratch;
            fmhaParams.multiCtasKvCounterPtr = p.getSemaphores();
        }

        // Skip-softmax attention parameters
        fmhaParams.skipSoftmaxThresholdScaleFactor = p.fwd.sparse_runtime_params.threshold_scale_factor_prefill;
#ifdef SKIP_SOFTMAX_STAT
        fmhaParams.skipSoftmaxTotalBlocks = mSkipSoftmaxTotalBlocks;
        fmhaParams.skipSoftmaxSkippedBlocks = mSkipSoftmaxSkippedBlocks;
#else
        if (tensorrt_llm::common::getEnvPrintSkipSoftmaxStat())
        {
            TLLM_THROW("To print skip softmax stat, please run build_wheel.py with -DSKIP_SOFTMAX_STAT");
        }
#endif

        if (p.attention_chunk_size)
        {
            fmhaParams.chunkedAttentionSize = *p.attention_chunk_size;
        }

        // Run the fmha kernel.
        mFmhaDispatcher->run(fmhaParams);
        sync_check_cuda_error(stream);

        if (!p.is_mla_enable) // Only for non-MLA attention
        {
            invokeKvCachePostprocessing(preprocessingParams, stream);
            sync_check_cuda_error(stream);
        }
    }
    else
    {
        TLLM_CHECK_DEBUG_WITH_INFO(p.getLognScalingPtr() == nullptr, "Unfused MHA does not support logn scaling");
        TLLM_CHECK_WITH_INFO(p.attention_chunk_size == std::nullopt, "Unfused MHA does not support chunked attention");
        // FIXME: a temporary solution to make sure the padding part of key/value buffer is 0
        // NOTE: pointer subtraction is used below since there could be some extra gap due to alignment.
        //  Otherwise, we could do cudaMemsetAsync(workspaceViews.kBuf, 0, k_buf_2_size + v_buf_2_size, stream).
        // cudaMemsetAsync(workspaceViews.kBuf, 0,
        //     reinterpret_cast<int8_t*>(workspaceViews.qkBuf) - reinterpret_cast<int8_t*>(workspaceViews.kBuf),
        //     stream);
        cudaMemsetAsync(workspaceViews.kBuf, 0,
            reinterpret_cast<int8_t*>(workspaceViews.vBuf) - reinterpret_cast<int8_t*>(workspaceViews.kBuf)
                + v_buf_2_size,
            stream);

        if (!isCrossAttention(p))
        {
            // self attention, write to from QKV to Q/K/V
            invokeAddFusedQKVBiasTranspose(workspaceViews.qBuf, workspaceViews.kBuf, workspaceViews.vBuf,
                p.getQkvOrQ<T>(), p.getQkvBias<T>(), p.getContextLengths(),
                p.remove_padding ? workspaceViews.paddingOffset : nullptr, p.num_seqs, p.input_seq_length, p.num_tokens,
                p.num_heads, mNumKVHeads, getHeadSize(), p.rotary_embedding_dim, p.rotary_embedding_base,
                p.rotary_embedding_scale_type, p.rotary_embedding_scale, p.rotary_embedding_max_positions,
                position_embedding_type, (float*) nullptr, 0, stream);
            sync_check_cuda_error(stream);
        }
        else
        {
            // cross attention, write from self QKV [*, head_num * head_size + 2 * kv_head_num * head_size]to Q, write
            // from cross KV [*, 2 * kv_head_num * head_size] to K/V kernel modified accordingly to handle nullptr
            // buffer
            invokeAddFusedQKVBiasTranspose(workspaceViews.qBuf, (T*) nullptr, (T*) nullptr, p.getQkvOrQ<T>(),
                p.getQkvBias<T>(), p.getContextLengths(), p.remove_padding ? workspaceViews.paddingOffset : nullptr,
                p.num_seqs, p.input_seq_length, p.num_tokens, p.num_heads, mNumKVHeads, getHeadSize(),
                p.rotary_embedding_dim, p.rotary_embedding_base, p.rotary_embedding_scale_type,
                p.rotary_embedding_scale, p.rotary_embedding_max_positions, position_embedding_type, (float*) nullptr,
                0, stream);
            sync_check_cuda_error(stream);

            invokeAddFusedQKVBiasTranspose((T*) nullptr, workspaceViews.kBuf, workspaceViews.vBuf, p.getCrossKv<T>(),
                p.getQkvBias<T>(), p.getEncoderInputLengths(),
                p.remove_padding ? workspaceViews.encoderPaddingOffset : nullptr, p.num_seqs, p.cross_kv_length,
                p.num_encoder_tokens, /*p.num_heads*/ 0, mNumKVHeads, getHeadSize(), p.rotary_embedding_dim,
                p.rotary_embedding_base, p.rotary_embedding_scale_type, p.rotary_embedding_scale,
                p.rotary_embedding_max_positions, position_embedding_type, (float*) nullptr, 0, stream);
            sync_check_cuda_error(stream);
        }

        // write KV to cache
        if (useKVCache(p))
        {
            invokeTranspose4dBatchMajor(workspaceViews.kBuf, workspaceViews.vBuf, kv_cache_buffer, p.num_seqs,
                isCrossAttention(p) ? p.cross_kv_length : p.input_seq_length,
                isCrossAttention(p) ? p.cross_kv_length : p.cyclic_attention_window_size, getHeadSize(), mNumKVHeads,
                cache_type, p.getKvScaleOrigQuant(),
                isCrossAttention(p) ? p.getEncoderInputLengths() : p.getContextLengths(), stream);
        }
        sync_check_cuda_error(stream);

        T const* linear_bias_slopes = isALiBi(p) ? p.getAlibiSlopes<T>() : nullptr;
        T const* relative_attention_bias = isRelativePosition(p) ? p.getRelativeAttentionBias<T>() : nullptr;
        int const relative_attention_bias_stride = isRelativePosition(p) ? p.relative_attention_bias_stride : 0;
        int const max_distance = p.fwd.relative_attention_max_distance;
        cudaDataType_t gemm_out_data_type = is_qk_buf_float_ ? CUDA_R_32F : gemm_data_type;
        void* gemm_out_buf_ = is_qk_buf_float_ ? static_cast<void*>(workspaceViews.qkFloatBuf)
                                               : static_cast<void*>(workspaceViews.qkBuf);
        if (mNumKVHeads == 1) // MQA
        {
            // Attn_weight[b, h*s_q, s_k] = Q[b, h*s_q, d] * K'[b, d, s_k]
            // Attn_weight'[b, s_k, h*s_q] = K[b, s_k, d] * Q'[b, d, h*s_q]
            mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_T, CUBLAS_OP_N,
                attention_seq_len_2,                                     // n
                attention_seq_len_1 * p.num_heads,                       // m
                getHeadSize(),                                           // k
                qk_scale_gemm, workspaceViews.kBuf, gemm_data_type,
                getHeadSize(),                                           // k
                attention_seq_len_2 * getHeadSize(),                     // n * k
                workspaceViews.qBuf, gemm_data_type,
                getHeadSize(),                                           // k
                attention_seq_len_1 * p.num_heads * getHeadSize(),       // m * k
                0.0f, gemm_out_buf_, gemm_out_data_type,
                attention_seq_len_2,                                     // n
                attention_seq_len_1 * p.num_heads * attention_seq_len_2, // m * n
                p.num_seqs,                                              // global batch size
                CUDA_R_32F);
        }
        else if (mNumKVHeads == p.num_heads) // MHA
        {
            // Attn_weight[b*h, s_q, s_k] = Q[b*h, s_q, d] * K'[b*h, d, s_k]
            // Attn_weight'[b*h, s_k, s_q] = K[b*h, s_k, d] * Q'[b*h, d, s_q]
            mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_T, CUBLAS_OP_N,
                attention_seq_len_2,                 // n
                attention_seq_len_1,                 // m
                getHeadSize(),                       // k
                qk_scale_gemm, workspaceViews.kBuf, gemm_data_type,
                getHeadSize(),                       // k
                attention_seq_len_2 * getHeadSize(), // n * k
                workspaceViews.qBuf, gemm_data_type,
                getHeadSize(),                       // k
                attention_seq_len_1 * getHeadSize(), // m * k
                0.0f, gemm_out_buf_, gemm_out_data_type,
                attention_seq_len_2,                 // n
                attention_seq_len_2 * attention_seq_len_1,
                p.num_seqs * p.num_heads,            // global batch size
                CUDA_R_32F);
        }
        else // GQA
        {
            // Some number of contiguous Q heads will share the same K/V head
            // Since the KV stride is NOT fixed for all Q, we have 2 options:
            //  1. Loop over stridedBatchedGemm for each KV head. (multiple API calls/cuda kernels)
            //  2. Calculate the pointers and use batchedGemm() (extra device memory) ::TODO::
            int const num_qheads_per_kv_head = p.num_heads / mNumKVHeads;
            for (int ki = 0; ki < mNumKVHeads; ++ki)
            {
                T* qptr = workspaceViews.qBuf + (ki * num_qheads_per_kv_head * attention_seq_len_1 * getHeadSize());
                T* kptr = workspaceViews.kBuf + (ki * attention_seq_len_2 * getHeadSize());
                int const qk_offset = ki * attention_seq_len_1 * num_qheads_per_kv_head * attention_seq_len_2;
                void* qkptr = is_qk_buf_float_ ? static_cast<void*>(workspaceViews.qkFloatBuf + qk_offset)
                                               : static_cast<void*>(workspaceViews.qkBuf + qk_offset);
                mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_T, CUBLAS_OP_N,
                    attention_seq_len_2,                                     // n
                    attention_seq_len_1 * num_qheads_per_kv_head,            // m
                    getHeadSize(),                                           // k
                    qk_scale_gemm, kptr, gemm_data_type,
                    getHeadSize(),                                           // k
                    mNumKVHeads * attention_seq_len_2 * getHeadSize(),       // n * k
                    qptr, gemm_data_type,
                    getHeadSize(),                                           // k
                    attention_seq_len_1 * p.num_heads * getHeadSize(),       // m * k
                    0.0f, qkptr, gemm_out_data_type,
                    attention_seq_len_2,                                     // n
                    attention_seq_len_1 * p.num_heads * attention_seq_len_2, // m * n
                    p.num_seqs,                                              // global batch size
                    CUDA_R_32F);
            }
        }

        if (is_qk_buf_float_ == true)
        {
            // add relative position bias
            if (isRelativePosition(p))
            {
                // Add relative_attention_bias
                // QK is (batch_size, local_head_num, q_length, k_length), relative_attention_bias is (1,
                // local_head_num, max_output_len + 1, max_output_len + 1). broadcast along 1st dim. max_seq_len is
                // already max_output_len + 1. In implicit mode, relative_attention_bias is relative_attention_table
                // [num_heads, num_buckets], with necessary p (max_distance, num_buckets) passed at the end
                invokeAddRelativeAttentionBiasUnaligned(workspaceViews.qkFloatBuf, relative_attention_bias, p.num_seqs,
                    p.num_heads, attention_seq_len_1,
                    isCrossAttention(p) ? p.cross_kv_length : p.cyclic_attention_window_size, stream, max_distance > 0,
                    relative_attention_bias_stride, max_distance, false /* bidirectional */);
            }

            MaskedSoftmaxParam<T, float> param;
            param.attention_score = workspaceViews.qkBuf;        // (batch_size, head_num, q_length, k_length)
            param.qk = workspaceViews.qkFloatBuf;                // (batch_size, head_num, q_length, k_length)
            param.attention_mask = workspaceViews.attentionMask; // (batch_size, q_length, k_length)
            param.batch_size = p.num_seqs;
            param.q_length = attention_seq_len_1;
            param.k_length = attention_seq_len_2;
            param.num_heads = p.num_heads;
            param.qk_scale = qk_scale_softmax;
            param.attn_logit_softcapping_scale = p.attn_logit_softcapping_scale;
            param.attn_logit_softcapping_inverse_scale = 1.0f / p.attn_logit_softcapping_scale;
            param.linear_bias_slopes = const_cast<T*>(linear_bias_slopes); // (head_num,), optional
            param.block_sparse_attn = p.mask_type == AttentionMaskType::BLOCKSPARSE;
            param.block_sparse_params = p.block_sparse_params;
            param.q_seq_lengths = p.getContextLengths();
            invokeMaskedSoftmax(param, stream);
        }
        else
        {
            // add relative position bias
            if (isRelativePosition(p))
            {
                // Add relative_attention_bias
                // QK is (batch_size, local_head_num, q_length, k_length), relative_attention_bias is (1,
                // local_head_num, max_output_len + 1, max_output_len + 1). broadcast along 1st dim. max_seq_len is
                // already max_output_len + 1. In implicit mode, relative_attention_bias is relative_attention_table
                // [num_heads, num_buckets], with necessary p (max_distance, num_buckets) passed at the end
                invokeAddRelativeAttentionBiasUnaligned(workspaceViews.qkBuf, relative_attention_bias, p.num_seqs,
                    p.num_heads, attention_seq_len_1,
                    isCrossAttention(p) ? p.cross_kv_length : p.cyclic_attention_window_size, stream, max_distance > 0,
                    relative_attention_bias_stride, max_distance, false /* bidirectional */);
            }

            MaskedSoftmaxParam<T, T> param;
            param.attention_score = workspaceViews.qkBuf;        // (batch_size, head_num, q_length, k_length)
            param.qk = workspaceViews.qkBuf;                     // (batch_size, head_num, q_length, k_length)
            param.attention_mask = workspaceViews.attentionMask; // (batch_size, q_length, k_length)
            param.batch_size = p.num_seqs;
            param.q_length = attention_seq_len_1;
            param.k_length = attention_seq_len_2;
            param.num_heads = p.num_heads;
            param.qk_scale = qk_scale_softmax;
            param.attn_logit_softcapping_scale = p.attn_logit_softcapping_scale;
            param.attn_logit_softcapping_inverse_scale = 1.0f / p.attn_logit_softcapping_scale;
            param.linear_bias_slopes = const_cast<T*>(linear_bias_slopes); // (head_num,), optional
            param.block_sparse_attn = p.mask_type == AttentionMaskType::BLOCKSPARSE;
            param.block_sparse_params = p.block_sparse_params;
            param.q_seq_lengths = p.getContextLengths();
            invokeMaskedSoftmax(param, stream);
        }

        if (mNumKVHeads == 1)
        {
            // Attn_weight[b, h*s_q, s_k]
            // O[b, h*s_q, d] = Attn_weight[b, h*s_q, s_k] * V[b, s_k, d]
            // O'[b, d, h*s_q] = V'[b, d, s_k] * Attn_weight'[b, s_k, h*s_q]
            mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_N, CUBLAS_OP_N,
                getHeadSize(),                                           // n
                p.num_heads * attention_seq_len_1,                       // m
                attention_seq_len_2,                                     // k
                workspaceViews.vBuf,
                getHeadSize(),                                           // n
                getHeadSize() * attention_seq_len_2,                     // n * k
                workspaceViews.qkBuf,
                attention_seq_len_2,                                     // k
                attention_seq_len_2 * p.num_heads * attention_seq_len_1, // m * k
                workspaceViews.qkvBuf,
                getHeadSize(),                                           // n
                getHeadSize() * p.num_heads * attention_seq_len_1,       // n * m
                p.num_seqs                                               // global batch size
            );
        }
        else if (mNumKVHeads == p.num_heads) // MHA
        {
            // O[b*h, s_q, d] = Attn_weight[b*h, s_q, s_k] * V[b*h, s_k, d]
            // O'[b*h, d, s_q] = V'[b*h, d, s_k] * Attn_weight'[b*h, s_k, s_q]
            mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_N, CUBLAS_OP_N, getHeadSize(), attention_seq_len_1,
                attention_seq_len_2, workspaceViews.vBuf, getHeadSize(), attention_seq_len_2 * getHeadSize(),
                workspaceViews.qkBuf, attention_seq_len_2, attention_seq_len_1 * attention_seq_len_2,
                workspaceViews.qkvBuf, getHeadSize(), attention_seq_len_1 * getHeadSize(), p.num_seqs * p.num_heads);
        }
        else // GQA
        {
            // Attn_weight[b, h*s_q, s_k]
            // O[b, h*s_q, d] = Attn_weight[b, h*s_q, s_k] * V[b, s_k, d]
            // O'[b, d, h*s_q] = V'[b, d, s_k] * Attn_weight'[b, s_k, h*s_q]
            int const num_qheads_per_kv_head = p.num_heads / mNumKVHeads;
            for (int ki = 0; ki < mNumKVHeads; ++ki)
            {
                T* qkptr
                    = workspaceViews.qkBuf + (ki * num_qheads_per_kv_head * attention_seq_len_1 * attention_seq_len_2);
                T* vptr = workspaceViews.vBuf + (ki * attention_seq_len_2 * getHeadSize());
                T* qkvptr = workspaceViews.qkvBuf + (ki * attention_seq_len_1 * num_qheads_per_kv_head * getHeadSize());
                mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_N, CUBLAS_OP_N,
                    getHeadSize(),                                           // n
                    num_qheads_per_kv_head * attention_seq_len_1,            // m
                    attention_seq_len_2,                                     // k
                    vptr,
                    getHeadSize(),                                           // n
                    mNumKVHeads * getHeadSize() * attention_seq_len_2,       // n * k
                    qkptr,
                    attention_seq_len_2,                                     // k
                    attention_seq_len_2 * p.num_heads * attention_seq_len_1, // m * k
                    qkvptr,
                    getHeadSize(),                                           // n
                    getHeadSize() * p.num_heads * attention_seq_len_1,       // n * m
                    p.num_seqs                                               // global batch size
                );
            }
        }

        if (!p.remove_padding)
        {
            invokeTransposeQKV(static_cast<T*>(p.getOutput()), workspaceViews.qkvBuf, p.num_seqs, attention_seq_len_1,
                p.num_heads, getHeadSize(), (float*) nullptr, 0, stream);
        }
        else
        {
            invokeTransposeAttentionOutRemovePadding(workspaceViews.qkvBuf, static_cast<T*>(p.getOutput()),
                p.num_tokens, p.num_seqs, attention_seq_len_1, p.num_heads, getHeadSize(), workspaceViews.paddingOffset,
                (float*) nullptr, 0, stream);
        }
    }
    return 0;
}

template int AttentionOp::enqueueContext<half, KVLinearBuffer>(
    FmhaParams const& p, MlaParams<half>* mlaParam, cudaStream_t stream);

template int AttentionOp::enqueueContext<float, KVLinearBuffer>(
    FmhaParams const& p, MlaParams<float>* mlaParam, cudaStream_t stream);

#ifdef ENABLE_BF16
template int AttentionOp::enqueueContext<__nv_bfloat16, KVLinearBuffer>(
    FmhaParams const& p, MlaParams<__nv_bfloat16>* mlaParam, cudaStream_t stream);
#endif

template int AttentionOp::enqueueContext<half, KVBlockArray>(
    FmhaParams const& p, MlaParams<half>* mlaParam, cudaStream_t stream);

template int AttentionOp::enqueueContext<float, KVBlockArray>(
    FmhaParams const& p, MlaParams<float>* mlaParam, cudaStream_t stream);

#ifdef ENABLE_BF16
template int AttentionOp::enqueueContext<__nv_bfloat16, KVBlockArray>(
    FmhaParams const& p, MlaParams<__nv_bfloat16>* mlaParam, cudaStream_t stream);
#endif

template <typename T, typename KVCacheBuffer>
int AttentionOp::enqueueGeneration(FmhaParams const& p, cudaStream_t stream)
{
    int const headSize = getHeadSize();
    float const q_scaling = p.q_scaling;
    float const* logn_scaling_ptr = isLognScaling(p) ? p.getLognScalingPtr() : nullptr;
    T const* relative_attention_bias = isRelativePosition(p) ? p.getRelativeAttentionBias<T>() : nullptr;
    int const relative_attention_bias_stride = isRelativePosition(p) ? p.relative_attention_bias_stride : 0;
    int const max_distance = p.fwd.relative_attention_max_distance;
    bool const* finished = nullptr;

    auto const quant_option = tc::QuantMode{};
    float const* qkv_scale_out = nullptr;

    int const* ia3_tasks = nullptr;
    T const* ia3_key_weights = nullptr;
    T const* ia3_value_weights = nullptr;

    int32_t const batch_beam = p.beam_width * p.num_requests;

    KVCacheBuffer kv_cache_buffer;
    KVCacheBuffer kv_scale_cache_buffer;

    auto const sizePerToken = mNumAttnKVHeads * headSize * getKvCacheElemSizeInBits<T>(p) / 8 /*bits*/;

    if (useKVCache(p))
    {
        auto buffers = buildKvCacheBuffers<KVCacheBuffer>(batch_beam, p.getMaxBlocksPerSequence(), p.tokens_per_block,
            sizePerToken, p.cyclic_attention_window_size, p.max_cyclic_attention_window_size, p.sink_token_length,
            p.can_use_one_more_block, p.getHostPrimaryPoolPtr(), p.getHostSecondaryPoolPtr(),
            p.getHostPrimaryBlockScalePoolPtr(), p.getHostSecondaryBlockScalePoolPtr(),
            p.getKvCacheBlockOffsets(p.getKvCachePoolIndex(p.local_layer_idx)), p.quant_mode.hasFp4KvCache(),
            p.max_attention_window_size, p.getKeyValueCache());
        kv_cache_buffer = buffers.kvCacheBuffer;
        kv_scale_cache_buffer = buffers.kvScaleCacheBuffer;
    }
    sync_check_cuda_error(stream);

    if (p.runtime_perf_knobs.has_value())
    {
        int64_t const multi_block_mode_val = p.getRuntimePerfKnobs()[0];
        mMultiBlockMode = multi_block_mode_val == 1;
        if (common::getEnvForceDeterministicAttention())
        {
            mMultiBlockMode = false;
        }
    }

    if (common::getEnvForceDeterministicAttention())
    {
        mMultiBlockMode = false;
    }

    // TODO only for debug usage
    if (!mMultiBlockMode)
    {
        char* isForceMultiBlockModeChar = std::getenv("FORCE_MULTI_BLOCK_MODE");
        bool isForceMultiBlockMode
            = (isForceMultiBlockModeChar != nullptr && std::string(isForceMultiBlockModeChar) == "ON");
        TLLM_CHECK_WITH_INFO(!(common::getEnvForceDeterministicAttention() && isForceMultiBlockMode),
            "FORCE_MULTI_BLOCK_MODE and FORCE_DETERMINISTIC/FORCE_ATTENTION_KERNEL_DETERMINISTIC can not be set at "
            "the same time.");
        mMultiBlockMode = isForceMultiBlockMode;
    }

    // Check that the chunked-attention and sliding-window-attention are not enabled at the same time.
    TLLM_CHECK_WITH_INFO(!p.attention_chunk_size.has_value() || p.cyclic_attention_window_size >= p.max_past_kv_length,
        "Chunked-attention and sliding-window-attention should not be enabled at the same time.");

    T* attention_input = p.getQkvOrQ<T>();
    // Try XQA optimization first.
    {
        // NOTE: input_seq_length = num_medusa_tokens + 1 (new generated one from the original LM head)
        // self attn
        XQAParams xqaParams{};
        this->template convertMMHAParamsToXQAParams<T, KVCacheBuffer>(xqaParams, p, /*forConfigurePlugin=*/false);

        if (mEnableXQA && mXqaDispatcher->shouldUse(xqaParams))
        {
            TLLM_LOG_DEBUG("XQA kernels are selected in the generation phase.");
            xqaParams.stream = stream;
            {
                mXqaDispatcher->run(xqaParams, kv_cache_buffer, kv_scale_cache_buffer);
            }
            return 0;
        }
        else if (p.is_spec_decoding_enabled && p.use_spec_decoding)
        {
            TLLM_CHECK_WITH_INFO(false, "No available XQA kernels are found for speculative decoding mode.");
        }
        else if (mFuseFp4Quant)
        {
            TLLM_CHECK_WITH_INFO(false, "No available kernels are found for FP4 output.");
        }
        else if (p.quant_mode.hasFp4KvCache())
        {
            TLLM_CHECK_WITH_INFO(false, "No available kernels are found for FP4 KV cache.");
        }
        else
        {
            TLLM_LOG_DEBUG("XQA kernels are not selected in the generation phase.");
        }
    }

    // This is the number of kv tokens that q needs to visit, but excluding one as it will be processed before the kv
    // loop.
    int timestep = p.max_past_kv_length;
    int const max_timesteps = std::min(timestep, static_cast<int>(p.cyclic_attention_window_size));
    int estimated_min_multi_block_count
        = estimate_min_multi_block_count(max_timesteps, mMaxSharedMemoryPerBlockOptin - 2048, sizeof(T));

    if (!mMultiBlockMode && !mForceMultiBlockWarned && estimated_min_multi_block_count > 1)
    {
        mForceMultiBlockWarned = true;
        TLLM_LOG_WARNING(
            "Force using MultiBlockMode in MMHA as shared memory is not enough, "
            "MultiBlockMode may have different accuracy compared to non-MultiBlockMode.");
    }

    // estimate min block count to satisfy shared memory requirement to run kernel.
    // Runtime check to see the actual number of blocks per sequence we need.
    int32_t const max_num_seq_len_tiles = std::max(getMaxNumSeqLenTile(p, batch_beam), estimated_min_multi_block_count);
    int32_t const min_num_seq_len_tiles = std::max(1, estimated_min_multi_block_count);
    bool const enable_multi_block
        = (mMultiBlockMode && max_num_seq_len_tiles > 1) || estimated_min_multi_block_count > 1;
    size_t const partial_out_size
        = enable_multi_block ? sizeof(T) * batch_beam * p.num_heads * mHeadSize * max_num_seq_len_tiles : 0;
    size_t const partial_sum_size
        = enable_multi_block ? sizeof(float) * batch_beam * p.num_heads * max_num_seq_len_tiles : 0;
    size_t const partial_max_size
        = enable_multi_block ? sizeof(float) * batch_beam * p.num_heads * max_num_seq_len_tiles : 0;
    size_t const shift_k_cache_size = (!p.pos_shift_enabled || isCrossAttention(p))
        ? 0
        : sizeof(T) * batch_beam * p.num_heads * mHeadSize * p.max_attention_window_size;

    AttentionGenerationWorkspaceSizes workspaceSizes{};
    workspaceSizes.partialOut = partial_out_size;
    workspaceSizes.partialSum = partial_sum_size;
    workspaceSizes.partialMax = partial_max_size;
    workspaceSizes.shiftKCache = shift_k_cache_size;
    {
        auto const cascadeSizes
            = tensorrt_llm::kernels::mmha::cascade::getCascadeWorkspaceSizes(batch_beam, p.num_heads, mHeadSize);
        workspaceSizes.cascadeOut = cascadeSizes.out;
        workspaceSizes.cascadeMax = cascadeSizes.mMax;
        workspaceSizes.cascadeSum = cascadeSizes.lSum;
    }
    auto const workspaceLayout = AttentionWorkspaceManager::buildGenerationLayout(workspaceSizes);
    auto const workspaceViews = AttentionWorkspaceManager::materializeGeneration<T>(p.getWorkspace(), workspaceLayout);

    // Apply position embedding to the keys in the K cache
    KVLinearBuffer shift_k_cache_buffer;
    if (useKVCache(p) && p.pos_shift_enabled && !isCrossAttention(p))
    {
        shift_k_cache_buffer
            = KVLinearBuffer(batch_beam, p.max_attention_window_size, sizePerToken, p.cyclic_attention_window_size,
                p.sink_token_length, true, reinterpret_cast<int8_t*>(workspaceViews.shiftKCache));
        sync_check_cuda_error(stream);
        // KV cache type
        KvCacheDataType const kv_cache_type = KvCacheDataType::BASE;
        using DataType = typename SATypeConverter<T>::Type;
        invokeShiftKCache<DataType, KVCacheBuffer>(kv_cache_buffer, shift_k_cache_buffer, kv_cache_type, getHeadSize(),
            timestep, batch_beam, mNumKVHeads, p.beam_width, p.cyclic_attention_window_size, p.sink_token_length,
            p.getKvScaleQuantOrig(), p.getSequenceLength(), p.getContextLengths(), p.rotary_embedding_dim,
            p.rotary_embedding_base, p.rotary_embedding_scale_type, p.rotary_embedding_scale,
            p.rotary_embedding_max_positions, p.position_embedding_type, stream);
    }

    FusedQKVMaskedAttentionDispatchParams<T, KVCacheBuffer> dispatch_params{};
    dispatch_params.mUnfuseQkvGemm = p.unfuse_qkv_gemm;
    dispatch_params.qkv_buf = attention_input;
    dispatch_params.qkv_bias = p.getQkvBias<T>();
    dispatch_params.logn_scaling_ptr = logn_scaling_ptr;
    dispatch_params.relative_attention_bias = relative_attention_bias;
    dispatch_params.relative_attention_bias_stride = relative_attention_bias_stride;
    dispatch_params.attention_mask = p.getAttentionMask();
    dispatch_params.attention_mask_stride = p.attention_mask_stride;
    dispatch_params.attention_sinks = p.getAttentionSinks();
    dispatch_params.max_distance = max_distance;
    dispatch_params.cache_indir = p.getCacheIndirection();
    dispatch_params.context_buf = p.getOutput(); //
    dispatch_params.finished = finished;
    dispatch_params.sequence_lengths
        = p.getSequenceLength(); // NOTE: current seq len including padding (fixed after meeting the finished id)
    dispatch_params.max_batch_size = batch_beam;
    dispatch_params.inference_batch_size = batch_beam;
    dispatch_params.beam_width = p.beam_width;
    dispatch_params.head_num = mNumAttnHeads;
    dispatch_params.kv_head_num = mNumAttnKVHeads;
    dispatch_params.size_per_head = getHeadSize();
    dispatch_params.rotary_embedding_dim = p.rotary_embedding_dim;
    dispatch_params.position_embedding_type = p.position_embedding_type;
    dispatch_params.chunked_attention_size = p.attention_chunk_size ? *p.attention_chunk_size : INT_MAX;
    dispatch_params.max_attention_window_size = p.max_attention_window_size;
    dispatch_params.cyclic_attention_window_size = p.cyclic_attention_window_size;
    dispatch_params.sink_token_length = isCrossAttention(p) ? 0 : p.sink_token_length;
    dispatch_params.input_lengths = p.getContextLengths();
    dispatch_params.timestep = timestep;
    dispatch_params.q_scaling = q_scaling;
    dispatch_params.attn_logit_softcapping_scale = p.attn_logit_softcapping_scale;
    dispatch_params.linear_bias_slopes = isALiBi(p) ? p.getAlibiSlopes<T>() : nullptr;
    dispatch_params.ia3_tasks = ia3_tasks;
    dispatch_params.ia3_key_weights = ia3_key_weights;
    dispatch_params.ia3_value_weights = ia3_value_weights;
    dispatch_params.qkv_scale_out = qkv_scale_out;
    dispatch_params.fp8_context_fmha = mFP8ContextFMHA;
    dispatch_params.attention_out_scale = p.getOutScale();
    dispatch_params.quant_option = quant_option;
    dispatch_params.multi_block_mode = enable_multi_block;
    dispatch_params.max_seq_len_tile = max_num_seq_len_tiles;
    dispatch_params.min_seq_len_tile = min_num_seq_len_tiles;
    dispatch_params.partial_out = workspaceViews.partialOut;
    dispatch_params.partial_sum = workspaceViews.partialSum;
    dispatch_params.partial_max = workspaceViews.partialMax;
    dispatch_params.cascade_partial_out = workspaceViews.cascadeOut;
    dispatch_params.cascade_partial_max = workspaceViews.cascadeMax;
    dispatch_params.cascade_partial_sum = workspaceViews.cascadeSum;
    dispatch_params.block_counter = p.getSemaphores();
    dispatch_params.kv_cache_quant_mode = p.quant_mode;
    dispatch_params.kv_scale_orig_quant = p.getKvScaleOrigQuant();
    dispatch_params.kv_scale_quant_orig = p.getKvScaleQuantOrig();
    dispatch_params.kv_block_array = kv_cache_buffer;
    dispatch_params.shift_k_cache_buffer = shift_k_cache_buffer;
    dispatch_params.multi_processor_count = mMultiProcessorCount;
    dispatch_params.rotary_embedding_base = p.rotary_embedding_base;
    dispatch_params.rotary_embedding_scale_type = p.rotary_embedding_scale_type;
    dispatch_params.rotary_embedding_scale = p.rotary_embedding_scale;
    dispatch_params.rotary_embedding_inv_freq_cache = p.getRotaryInvFreq();
    dispatch_params.rotary_embedding_cos_sin_cache = p.getRotaryCosSin();
    dispatch_params.rotary_embedding_short_m_scale = p.rotary_embedding_short_mscale;
    dispatch_params.rotary_embedding_long_m_scale = p.rotary_embedding_long_mscale;
    dispatch_params.rotary_embedding_max_positions = p.rotary_embedding_max_positions;
    dispatch_params.rotary_embedding_original_max_positions = p.rotary_embedding_original_max_positions;
    dispatch_params.position_shift_enabled = p.pos_shift_enabled;
    dispatch_params.rotary_cogvlm_vision_start = p.vision_start;
    dispatch_params.rotary_cogvlm_vision_length = p.vision_length;
    dispatch_params.cross_attention = isCrossAttention(p);
    dispatch_params.memory_length_per_sample = p.getEncoderInputLengths();
    dispatch_params.block_sparse_attention = p.mask_type == AttentionMaskType::BLOCKSPARSE;
    dispatch_params.block_sparse_params = p.block_sparse_params;
    dispatch_params.mrope_position_deltas = p.getMropePositionDeltas();

    using DataType = typename SATypeConverter<T>::Type;
    {
        if (!isCrossAttention(p))
        {
            // self attn
            Masked_multihead_attention_params<DataType> mmha_params;
            fusedQKV_masked_attention_dispatch(mmha_params, dispatch_params, stream);
        }
        else
        {
            // cross attn
            Cross_multihead_attention_params<DataType> mmhca_params;
            fusedQKV_masked_attention_dispatch(mmhca_params, dispatch_params, stream);
        }
        sync_check_cuda_error(stream);
    }

    return 0;
}

template int AttentionOp::enqueueGeneration<half, KVLinearBuffer>(FmhaParams const& p, cudaStream_t stream);

template int AttentionOp::enqueueGeneration<float, KVLinearBuffer>(FmhaParams const& p, cudaStream_t stream);

#ifdef ENABLE_BF16
template int AttentionOp::enqueueGeneration<__nv_bfloat16, KVLinearBuffer>(FmhaParams const& p, cudaStream_t stream);
#endif

template int AttentionOp::enqueueGeneration<half, KVBlockArray>(FmhaParams const& p, cudaStream_t stream);

template int AttentionOp::enqueueGeneration<float, KVBlockArray>(FmhaParams const& p, cudaStream_t stream);

#ifdef ENABLE_BF16
template int AttentionOp::enqueueGeneration<__nv_bfloat16, KVBlockArray>(FmhaParams const& p, cudaStream_t stream);
#endif

template <typename T, typename KVCacheBuffer>
void AttentionOp::prepareEnqueueGeneration(FmhaParams const& p)
{
    // self attn
    if (mXqaDispatcher.get() != nullptr)
    {
        TLLM_LOG_TRACE("Preparing XQA kernels in prepareEnqueueGeneration.");
        XQAParams xqaParams{};
        this->template convertMMHAParamsToXQAParams<T, KVCacheBuffer>(xqaParams, p, /*forConfigurePlugin=*/true);
        mXqaDispatcher->prepare(xqaParams);
    }
}

template void AttentionOp::prepareEnqueueGeneration<half, KVLinearBuffer>(FmhaParams const& p);

template void AttentionOp::prepareEnqueueGeneration<float, KVLinearBuffer>(FmhaParams const& p);

#ifdef ENABLE_BF16
template void AttentionOp::prepareEnqueueGeneration<__nv_bfloat16, KVLinearBuffer>(FmhaParams const& p);
#endif

template void AttentionOp::prepareEnqueueGeneration<half, KVBlockArray>(FmhaParams const& p);

template void AttentionOp::prepareEnqueueGeneration<float, KVBlockArray>(FmhaParams const& p);

#ifdef ENABLE_BF16
template void AttentionOp::prepareEnqueueGeneration<__nv_bfloat16, KVBlockArray>(FmhaParams const& p);
#endif

template <typename KVCacheBuffer>
KvCacheBuffers<KVCacheBuffer> buildKvCacheBuffers(int32_t batchSize, int32_t maxBlocksPerSeq, int32_t tokensPerBlock,
    int32_t sizePerToken, int32_t cyclicAttentionWindowSize, int32_t maxCyclicAttentionWindowSize, int32_t sinkTokenLen,
    bool canUseOneMoreBlock, void* primaryPoolPtr, void* secondaryPoolPtr, void* primaryBlockScalePoolPtr,
    void* secondaryBlockScalePoolPtr, KVBlockArray::DataType* blockOffsets, bool hasFp4KvCache,
    int32_t maxAttentionWindowSize, void* keyValueCache)
{
    KvCacheBuffers<KVCacheBuffer> result;
    if constexpr (std::is_same_v<KVCacheBuffer, KVBlockArray>)
    {
        result.kvCacheBuffer = KVBlockArray(batchSize, maxBlocksPerSeq, tokensPerBlock, sizePerToken,
            cyclicAttentionWindowSize, maxCyclicAttentionWindowSize, sinkTokenLen, canUseOneMoreBlock, primaryPoolPtr,
            secondaryPoolPtr, blockOffsets);
        if (hasFp4KvCache)
        {
            result.kvScaleCacheBuffer = KVBlockArray(batchSize, maxBlocksPerSeq, tokensPerBlock, sizePerToken / 8,
                cyclicAttentionWindowSize, maxCyclicAttentionWindowSize, sinkTokenLen, canUseOneMoreBlock,
                primaryBlockScalePoolPtr, secondaryBlockScalePoolPtr, blockOffsets);
        }
    }
    else if constexpr (std::is_same_v<KVCacheBuffer, KVLinearBuffer>)
    {
        TLLM_CHECK_WITH_INFO(!hasFp4KvCache, "FP4 KV cache only supports paged KV.");
        TLLM_CHECK_WITH_INFO(keyValueCache != nullptr, "keyValueCache must not be null for linear KV cache.");
        using BufferDataType = typename KVCacheBuffer::DataType;
        result.kvCacheBuffer = KVLinearBuffer(batchSize, maxAttentionWindowSize, sizePerToken,
            cyclicAttentionWindowSize, sinkTokenLen, false, reinterpret_cast<BufferDataType*>(keyValueCache));
    }
    return result;
}

template KvCacheBuffers<KVBlockArray> buildKvCacheBuffers<KVBlockArray>(int32_t, int32_t, int32_t, int32_t, int32_t,
    int32_t, int32_t, bool, void*, void*, void*, void*, KVBlockArray::DataType*, bool, int32_t, void*);

template KvCacheBuffers<KVLinearBuffer> buildKvCacheBuffers<KVLinearBuffer>(int32_t, int32_t, int32_t, int32_t, int32_t,
    int32_t, int32_t, bool, void*, void*, void*, void*, KVBlockArray::DataType*, bool, int32_t, void*);

std::string AttentionOp::toString() const
{
    // Only the op's own state. The per-call parameters are the caller's to log; they
    // change every call, whereas these are what the op derived once and dispatches on.
    std::stringstream ss;
    ss << std::boolalpha;
#define TRTLLM_DUMP_MEMBER(M) ss << #M ": " << M << "\n";
    TRTLLM_DUMP_MEMBER(mNumKVHeads)
    TRTLLM_DUMP_MEMBER(mNumAttnHeads)
    TRTLLM_DUMP_MEMBER(mNumAttnKVHeads)
    TRTLLM_DUMP_MEMBER(mHeadSize)
    TRTLLM_DUMP_MEMBER(mPagedKVCache)
    TRTLLM_DUMP_MEMBER(mEnableContextFMHA)
    TRTLLM_DUMP_MEMBER(mFMHAForceFP32Acc)
    TRTLLM_DUMP_MEMBER(mMultiBlockMode)
    TRTLLM_DUMP_MEMBER(mEnableXQA)
    TRTLLM_DUMP_MEMBER(mFP8ContextFMHA)
    TRTLLM_DUMP_MEMBER(mFP8AttenOutput)
    TRTLLM_DUMP_MEMBER(mFP8ContextMLA)
    TRTLLM_DUMP_MEMBER(mFP8GenerationMLA)
    TRTLLM_DUMP_MEMBER(mFuseFp4Quant)
    TRTLLM_DUMP_MEMBER(mIsGenerationMLA)
    TRTLLM_DUMP_MEMBER(mUseGenFlashMLA)
    TRTLLM_DUMP_MEMBER(mSM)
    TRTLLM_DUMP_MEMBER(mUseTllmGen)
    TRTLLM_DUMP_MEMBER(mMultiProcessorCount)
    TRTLLM_DUMP_MEMBER(mMaxSharedMemoryPerBlockOptin)
    TRTLLM_DUMP_MEMBER(mForceMultiBlockWarned)
#undef TRTLLM_DUMP_MEMBER
    return ss.str();
}

namespace trtllm::attention
{
using tensorrt_llm::kernels::KVBlockArray;
using tensorrt_llm::kernels::MlaParams;
using tensorrt_llm::torch_ext::AttentionOp;
using tensorrt_llm::torch_ext::KvCachePoolPointers;

#ifdef ENABLE_BF16
#define _DISPATCH_ON_DTYPE_BF16(FN, ...)                                                                               \
    case tensorrt_llm::DataType::kBF16: FN<__nv_bfloat16>(__VA_ARGS__); break;
#else
#define _DISPATCH_ON_DTYPE_BF16(FN, ...)
#endif
#ifdef ENABLE_BF16
#define _DISPATCH_ON_TORCH_DTYPE_BF16(FN, ...)                                                                         \
    case torch::kBFloat16: FN<__nv_bfloat16>(__VA_ARGS__); break;
#else
#define _DISPATCH_ON_TORCH_DTYPE_BF16(FN, ...)
#endif

// Dispatches straight off the tensor because the entry points run before prepare()
// derives FmhaParams::type.
#define DISPATCH_ON_TORCH_DTYPE(SCALAR_TYPE, FN, ...)                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        switch (SCALAR_TYPE)                                                                                           \
        {                                                                                                              \
        case torch::kFloat32: FN<float>(__VA_ARGS__); break;                                                           \
        case torch::kFloat16:                                                                                          \
            FN<half>(__VA_ARGS__);                                                                                     \
            break;                                                                                                     \
            _DISPATCH_ON_TORCH_DTYPE_BF16(FN, __VA_ARGS__)                                                             \
        default: TLLM_CHECK_WITH_INFO(false, "Unsupported attention dtype"); break;                                    \
        }                                                                                                              \
    } while (0)

#define DISPATCH_ON_DTYPE(DTYPE, FN, ...)                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        switch (DTYPE)                                                                                                 \
        {                                                                                                              \
        case tensorrt_llm::DataType::kFLOAT: FN<float>(__VA_ARGS__); break;                                            \
        case tensorrt_llm::DataType::kHALF:                                                                            \
            FN<half>(__VA_ARGS__);                                                                                     \
            break;                                                                                                     \
            _DISPATCH_ON_DTYPE_BF16(FN, __VA_ARGS__)                                                                   \
        default: TLLM_CHECK_WITH_INFO(false, "Unsupported attention dtype"); break;                                    \
        }                                                                                                              \
    } while (0)

} // namespace trtllm::attention

template <typename T>
void FmhaParams::finalizeMlaParams(MlaParams<T>& mla) const
{
    mla.q_buf = getQkvOrQ<T>();
    mla.context_buf = static_cast<T*>(getOutput());
    // MlaParams has no default member initializers for these two, and the MLA RoPE
    // kernels use max_input_seq_len as a grid dimension and dereference cache_seq_lens.
    // Both phases must set them.
    mla.cache_seq_lens = getSequenceLength();
    mla.max_input_seq_len = input_seq_length;

    mla.cos_sin_cache = AttentionOp::isRoPE(*this) ? getRotaryCosSin() : nullptr;
    if (fwd.enable_dsv4_epilogue_fusion)
    {
        TORCH_CHECK(
            fwd.dsv4_inv_rope_cos_sin_cache.has_value(), "DSv4 fused epilogue requires inverse-RoPE cos/sin cache.");
        auto const& cosSinCache = fwd.dsv4_inv_rope_cos_sin_cache.value();
        auto const& outputSfTensor = fwd.output_sf.value();
        TORCH_CHECK(cosSinCache.scalar_type() == torch::kFloat32, "DSv4 fused epilogue cos/sin cache must be float32.");
        TORCH_CHECK(output.scalar_type() == torch::kFloat8_e4m3fn, "DSv4 fused epilogue output must be float8_e4m3fn.");
        TORCH_CHECK(output.dim() == 3 && output.is_contiguous(),
            "DSv4 fused epilogue output must be contiguous [groups, tokens, K].");
        TORCH_CHECK(
            outputSfTensor.scalar_type() == torch::kFloat32, "DSv4 fused epilogue fwd.output_sf must be float32.");
        TORCH_CHECK(outputSfTensor.dim() == 3 && outputSfTensor.is_contiguous(),
            "DSv4 fused epilogue fwd.output_sf must be contiguous [groups, K/128, padded_tokens].");
        TORCH_CHECK(output.size(1) >= num_tokens, "DSv4 fused epilogue output token dimension is too small.");
        TORCH_CHECK(mla_params.v_head_dim > 0 && mla_params.v_head_dim % 128 == 0,
            "DSv4 fused epilogue requires v_head_dim to be a positive multiple of 128.");
        TORCH_CHECK(
            outputSfTensor.size(2) >= num_tokens, "DSv4 fused epilogue fwd.output_sf token dimension is too small.");

        mla.dsv4_epilogue_fusion.enabled = true;
        mla.dsv4_epilogue_fusion.cos_sin_cache = getDsv4InvRopeCosSinCache();
        mla.dsv4_epilogue_fusion.scale_buf_m = static_cast<int32_t>(outputSfTensor.size(2));
    }
    mla.batch_size = num_seqs;
    mla.acc_q_len = num_tokens;
    mla.head_num = num_heads;
    mla.meta = mla_params;

    mla.workspace = getWorkspace();
}

template <typename T>
MlaParams<T> FmhaParams::buildContextMlaParams() const
{
    MlaParams<T> mla{};
    if (use_sparse_attention)
    {
        mla.latent_cache = getLatentCache<T>();
        TORCH_CHECK(fwd.q_pe.has_value());
        TORCH_CHECK(fwd.q_pe->dim() == 3);
        TORCH_CHECK(fwd.q_pe->strides()[2] == 1);

        mla.q_pe = getQPe<T>();
        mla.q_pe_ld = fwd.q_pe->strides()[1];
        mla.q_pe_stride = fwd.q_pe->strides()[0];

        // Fused FP8-Q path: forward caller's fwd.quant_q_buffer / scale so
        // applyMLARopeAndAssignQKVKernelOptContext<kOutputFp8Q=true>
        // appends rope FP8 in place and the standalone quantize is
        // skipped. Without this wiring the sparse-MLA context branch
        // runs the legacy quantize over the bf16 placeholder q.
        mla.bmm1_scale = getMlaBmm1Scale();
        mla.bmm2_scale = getMlaBmm2Scale();
        mla.quant_q_buf = getQuantQBuffer();
        mla.quant_scale_qkv = getQuantScaleQkv();
        mla.fuse_q_fp8_in_rope = (fwd.quant_q_buffer.has_value() && fwd.quant_scale_qkv.has_value());

        // Fused kv_a_layernorm: the norm weight implies `latent_cache` is the
        // RAW kv_a_proj output, with the caller's RMSNorm and concat dropped.
        if (fwd.kv_norm_weight)
        {
            auto const& kvNormWeight = fwd.kv_norm_weight.value();
            TORCH_CHECK(kvNormWeight.is_cuda(), "kv_norm_weight must be a CUDA tensor");
            TORCH_CHECK(kvNormWeight.is_contiguous(), "kv_norm_weight must be contiguous");
            TORCH_CHECK(kvNormWeight.scalar_type() == qkv_or_q.scalar_type(),
                "kv_norm_weight dtype must match the activation dtype");
            TORCH_CHECK(fwd.latent_cache, "fused kv-norm needs latent_cache (the raw kv_a_proj output) to be provided");
            // The kernel norms the whole latent row, so a narrower weight would
            // read out of bounds. dsv3RopeOp.cpp checks the same on the
            // generation side.
            auto const kvNormWidth = mla_params.kv_lora_rank + mla_params.qk_rope_head_dim;
            TORCH_CHECK(kvNormWeight.numel() == kvNormWidth,
                "kv_norm_weight must span kv_lora_rank + qk_rope_head_dim (", kvNormWidth, "), got ",
                kvNormWeight.numel());
            // A last-dim slice, so rows are wider than the row itself. Forward
            // the real stride; only the innermost dim must be unit-stride.
            auto const& latentCache = fwd.latent_cache.value();
            TORCH_CHECK(
                latentCache.dim() == 2, "latent_cache must be 2D for fused kv-norm, got ", latentCache.dim(), "D");
            TORCH_CHECK(latentCache.stride(1) == 1, "latent_cache must be unit-stride in its last dim");
            // The kernel walks rows with 16-byte vector loads, so a row start
            // that is not 16B-aligned faults with a bare misaligned-address
            // error far from here.
            auto const kEltsPer16B = 16 / latentCache.element_size();
            TORCH_CHECK(latentCache.stride(0) % kEltsPer16B == 0, "latent_cache row stride (", latentCache.stride(0),
                ") must be a multiple of ", kEltsPer16B, " for the fused kv-norm 16B vector loads");
            TORCH_CHECK(reinterpret_cast<uintptr_t>(latentCache.data_ptr()) % 16 == 0,
                "latent_cache must be 16B-aligned for the fused kv-norm vector loads");
            mla.latent_row_stride = static_cast<int>(latentCache.stride(0));
            mla.kv_norm_weight = static_cast<void const*>(kvNormWeight.data_ptr());
            mla.kv_norm_eps = static_cast<float>(fwd.kv_norm_eps);
            mla.fuse_kv_norm_in_rope = true;
        }
    }
    else
    {
        mla.latent_cache = getLatentCache<T>();
        TORCH_CHECK(k.has_value());
        TORCH_CHECK(v.has_value());
        TORCH_CHECK(k->dim() == 2);
        TORCH_CHECK(v->dim() == 2);
        TORCH_CHECK(k->strides()[1] == 1);
        TORCH_CHECK(v->strides()[1] == 1);

        mla.k_buf = getK<T>();
        mla.v_buf = getV<T>();

        mla.helix_position_offsets = getHelixPositionOffsets();
        mla.helix_is_inactive_rank = getHelixIsInactiveRank();
    }
    finalizeMlaParams<T>(mla);
    return mla;
}

template <typename T>
MlaParams<T> FmhaParams::buildGenerationMlaParams() const
{
    MlaParams<T> mla{};
    TORCH_CHECK(fwd.latent_cache.has_value());
    mla.latent_cache = getLatentCache<T>();
    TORCH_CHECK(fwd.q_pe.has_value());
    TORCH_CHECK(fwd.q_pe->dim() == 3);
    TORCH_CHECK(fwd.q_pe->strides()[2] == 1);

    mla.q_pe = getQPe<T>();
    mla.q_pe_ld = fwd.q_pe->strides()[1];
    mla.q_pe_stride = fwd.q_pe->strides()[0];

    mla.seqQOffset = const_cast<int*>(getCuQSeqlens());
    mla.cu_kv_seqlens = const_cast<int*>(getCuKvSeqlens());
    mla.fmha_tile_counter = reinterpret_cast<uint32_t*>(getFmhaSchedulerCounter());
    mla.bmm1_scale = getMlaBmm1Scale();
    mla.bmm2_scale = getMlaBmm2Scale();
    mla.quant_q_buf = getQuantQBuffer();
    mla.quant_scale_qkv = getQuantScaleQkv();
    mla.fuse_q_fp8_in_rope = (fwd.quant_q_buffer.has_value() && fwd.quant_scale_qkv.has_value());
    finalizeMlaParams<T>(mla);
    return mla;
}

template <typename T>
void FmhaParams::addFlashMlaGenerationParams(MlaParams<T>& mla) const
{
    TORCH_CHECK(block_ids_per_seq.has_value());
    mla.block_ids_per_seq = getBlockIdsPerSeq();
    if (flash_mla_tile_scheduler_metadata.has_value())
    {
        TORCH_CHECK(flash_mla_num_splits.has_value(),
            "flash_mla_num_splits must be provided when flash_mla_tile_scheduler_metadata is set.");
        mla.flash_mla_tile_scheduler_metadata = getFlashMlaTileSchedulerMetadata();
        mla.flash_mla_num_splits = getFlashMlaNumSplits();
    }
}

template <typename T>
void AttentionOp::runContextImpl(FmhaParams& p)
{
    prepare(p, /*isGen=*/false);
    auto const stream = at::cuda::getCurrentCUDAStream(p.qkv_or_q.get_device());
    MlaParams<T> mla{};
    MlaParams<T>* mlaParam = nullptr;
    if (AttentionOp::isMLAEnabled(p))
    {
        mla = p.buildContextMlaParams<T>();
        mlaParam = &mla;
    }
    enqueueContext<T, KVBlockArray>(p, mlaParam, stream);
    sync_check_cuda_error(stream);
}

template <typename T>
void AttentionOp::runGenerationImpl(FmhaParams& p)
{
    prepare(p, /*isGen=*/true);
    auto const stream = at::cuda::getCurrentCUDAStream(p.qkv_or_q.get_device());
    enqueueGeneration<T, KVBlockArray>(p, stream);
    {
        std::string const afterGenStr = "gen attention at layer " + std::to_string(p.layer_idx);
        TLLM_CHECK_DEBUG_WITH_INFO(tensorrt_llm::runtime::utils::tensorHasInvalid(p.num_tokens,
                                       p.num_heads * p.head_size, p.type, p.getOutput(), stream, afterGenStr)
                == false,
            "Found invalid number (NaN or Inf) in " + afterGenStr);
    }
    sync_check_cuda_error(stream);
}

template <typename T>
void AttentionOp::runMlaGenerationImpl(FmhaParams& p)
{
    prepare(p, /*isGen=*/true);
    auto const stream = at::cuda::getCurrentCUDAStream(p.qkv_or_q.get_device());
    auto mla = p.buildGenerationMlaParams<T>();
    if (mUseGenFlashMLA)
    {
        p.addFlashMlaGenerationParams<T>(mla);
    }
    mlaGeneration<T>(mla, p, stream);
    {
        std::string const afterGenStr = "mla gen attention at layer " + std::to_string(p.layer_idx);
        TLLM_CHECK_DEBUG_WITH_INFO(tensorrt_llm::runtime::utils::tensorHasInvalid(p.num_tokens,
                                       p.num_heads * p.head_size, p.type, p.getOutput(), stream, afterGenStr)
                == false,
            "Found invalid number (NaN or Inf) in " + afterGenStr);
    }
    sync_check_cuda_error(stream);
}

AttentionOp::AttentionOp(StaticAttentionConfig const& cfg)
    : mDriver(CUDADriverWrapper::getInstance())
    , mCublasWrapper(new tc::CublasMMWrapper(getCublasHandle(), getCublasLtHandle(), nullptr, nullptr))
{
    // Reading these caches the parsed values, keeping getenv() off the per-call path.
    getEnvMmhaMultiblockDebug();
    getEnvMmhaBlocksPerSequence();

    mNumKVHeads = static_cast<int>(cfg.num_kv_heads);
    mHeadSize = static_cast<int>(cfg.head_size);
    if (cfg.is_mla_enable)
    {
        mIsGenerationMLA = cfg.head_size == cfg.kv_lora_rank + cfg.qk_rope_head_dim;
        mUseGenFlashMLA = mSM == 90 && cfg.tokens_per_block == 64 && cfg.head_size == 576;
        mNumKVHeads = 1;
        mHeadSize = cfg.kv_lora_rank + cfg.qk_rope_head_dim;
    }

    auto constexpr kMaxSkipCorrectionThreshold = 32.0;
    TLLM_CHECK_WITH_INFO(
        cfg.skip_correction_threshold >= 0.0 && cfg.skip_correction_threshold <= kMaxSkipCorrectionThreshold,
        "skip_correction_threshold must be in the range (0, 32] when enabled, or 0 when disabled.");
    bool const applySkipCorrection = cfg.is_mla_enable && (mSM == 100 || mSM == 103);
    mSkipCorrectionThreshold = applySkipCorrection ? static_cast<float>(cfg.skip_correction_threshold) : 0.0F;

    // One rank per attention op: neither tensor nor context parallelism applies here.
    mNumAttnHeads = static_cast<int>(cfg.num_heads);
    mNumAttnKVHeads = mNumKVHeads;
}

int AttentionOp::prepare(FmhaParams& p, bool isGen)
{
    p.type = tensorrt_llm::runtime::TorchUtils::dataType(p.qkv_or_q.scalar_type());
    p.is_fp8_out = p.output.scalar_type() == torch::kFloat8_e4m3fn;
    p.is_fp4_out = p.output.scalar_type() == torch::kUInt8;
    p.use_kv_cache = p.hasKvCache();
    p.fuses_dsv4_inv_rope_fp8_quant = p.fwd.enable_dsv4_epilogue_fusion;
    p.cross_attention = p.is_cross;

    if (p.spec_decoding_target_max_draft_tokens.has_value() && p.spec_decoding_target_max_gen_len == 0)
    {
        p.spec_decoding_target_max_gen_len = static_cast<int32_t>(p.spec_decoding_target_max_draft_tokens.value()) + 1;
    }

    bool const hasSparseAttnIndices = p.fwd.sparse_runtime_params.sparse_attn_indices.has_value()
        && p.fwd.sparse_runtime_params.sparse_attn_indices.value().numel() > 0;
    p.use_sparse_attention = (p.fwd.sparse_runtime_params.sparse_kv_indices.has_value()
                                 && p.fwd.sparse_runtime_params.sparse_kv_indices.value().numel() > 0)
        || hasSparseAttnIndices;
    p.use_tllm_gen_sparse_attention_paged = hasSparseAttnIndices
        && p.fwd.sparse_runtime_params.sparse_attn_offsets.has_value()
        && p.fwd.sparse_runtime_params.sparse_attn_offsets.value().numel() > 0;
    p.use_tllm_gen_sparse_attention = hasSparseAttnIndices && !p.use_tllm_gen_sparse_attention_paged;

    if (p.is_mla_enable)
    {
        if (p.num_sparse_topk > 0 && hasSparseAttnIndices)
        {
            p.use_sparse_attention = true;
        }
        TLLM_CHECK(!p.is_fp4_out);
        p.mla_params = {static_cast<int>(p.q_lora_rank.value()), static_cast<int>(p.kv_lora_rank),
            static_cast<int>(p.qk_nope_head_dim), static_cast<int>(p.qk_rope_head_dim),
            static_cast<int>(p.v_head_dim.value()), static_cast<int>(p.predicted_tokens_per_seq),
            static_cast<int>(p.getMlaLayerNum()), static_cast<int>(p.rope_append.value_or(true))};
    }

    // Commonly identical to cyclic_attention_window_size unless layers use different
    // attention window sizes; beam search may consume one extra block.
    p.max_cyclic_attention_window_size = p.cyclic_attention_window_size;
    p.can_use_one_more_block = p.beam_width > 1;

    // Block-table stride, i.e. the trailing dimension of kv_cache_block_offsets.
    p.max_blocks_per_sequence = p.getMaxBlocksPerSequence();

    // The multi-block / multi-CTA-KV counter. Python owns it: TrtllmAttention caches
    // and zeroes the buffer, and hands it over under its scheduler-counter name.
    p.semaphores = p.fwd.fmha_scheduler_counter;

    // Generation consumers dereference the counter without a null check -- MMHA at
    // `params.block_counter[bhi]`, XQA at `semaphores[idxSeq]` -- so a caller that
    // forgets it produces an illegal memory access with no host-side stack. The phased
    // run_* ops are a new caller-facing boundary, so name the missing buffer here.
    TLLM_CHECK_WITH_INFO(!isGen || p.semaphores.has_value(),
        "Generation attention requires fmha_scheduler_counter; the caller owns this buffer.");

    // `out_scale` is the output scale for FP8 output, but the global scale for the
    // scaling factors when the NVFP4 quant epilogue is fused; the two feed different
    // kernel arguments and must not be mixed up.
    if (p.is_fp4_out)
    {
        p.out_sf_scale = p.fwd.out_scale;
        p.fwd.out_scale = std::nullopt;
    }

    if (p.fwd.attention_sinks.has_value())
    {
        TORCH_CHECK(p.fwd.attention_sinks.value().scalar_type() == torch::kFloat32,
            "Expected attention_sinks to have float dtype");
    }
    if (p.quant_mode.hasFp4KvCache())
    {
        TORCH_CHECK(!p.fwd.kv_scale_orig_quant.has_value() || p.fwd.kv_scale_orig_quant.value().size(0) == 3,
            "FP4 KV cache expects kv_scale_orig_quant to have 3 elements.");
        TORCH_CHECK(!p.fwd.kv_scale_quant_orig.has_value() || p.fwd.kv_scale_quant_orig.value().size(0) == 3,
            "FP4 KV cache expects kv_scale_quant_orig to have 3 elements.");
    }

    // The KV-cache scales are meaningful only when the cache is actually quantized.
    // `TrtllmAttention` hands them over unconditionally so the XQA kernels always see a
    // valid pointer when `isKVCacheQuantized` is true; consumers in turn read a null
    // pointer as "no KV-cache dequant", and some branch on exactly that. Drop the pair
    // here when the quant mode says otherwise, so the whole op -- including the
    // `!kv_scale_quant_orig` precondition on the SageAttention path -- keeps seeing a
    // consistent pair. They travel together: a lone scale describes only half of the
    // conversion and is never usable.
    if (!p.quant_mode.hasKvCacheQuant() || !p.fwd.kv_scale_orig_quant.has_value()
        || !p.fwd.kv_scale_quant_orig.has_value())
    {
        p.fwd.kv_scale_orig_quant.reset();
        p.fwd.kv_scale_quant_orig.reset();
    }
    else if (p.quant_mode.hasFp4KvCache())
    {
        // An FP4 cache reads the scales as raw float pointers, so the shape and layout
        // are part of the ABI rather than something the kernels can check.
        auto const& origQuantScale = p.fwd.kv_scale_orig_quant.value();
        auto const& quantOrigScale = p.fwd.kv_scale_quant_orig.value();
        if (p.is_mla_enable)
        {
            TORCH_CHECK(origQuantScale.scalar_type() == torch::kFloat32,
                "kv_scale_orig_quant must have float32 dtype for MLA with FP4 KV cache");
            TORCH_CHECK(quantOrigScale.scalar_type() == torch::kFloat32,
                "kv_scale_quant_orig must have float32 dtype for MLA with FP4 KV cache");
            TORCH_CHECK(
                origQuantScale.is_contiguous(), "kv_scale_orig_quant must be contiguous for MLA with FP4 KV cache");
            TORCH_CHECK(
                quantOrigScale.is_contiguous(), "kv_scale_quant_orig must be contiguous for MLA with FP4 KV cache");
            TORCH_CHECK(origQuantScale.dim() == 1 && origQuantScale.size(0) == 1,
                "kv_scale_orig_quant must have shape [1] for MLA with FP4 KV cache");
            TORCH_CHECK(quantOrigScale.dim() == 1 && quantOrigScale.size(0) == 1,
                "kv_scale_quant_orig must have shape [1] for MLA with FP4 KV cache");
        }
        else
        {
            TORCH_CHECK(origQuantScale.size(0) == 3, "kv_scale_orig_quant must have 3 entries for FP4 KV cache");
            TORCH_CHECK(quantOrigScale.size(0) == 3, "kv_scale_quant_orig must have 3 entries for FP4 KV cache");
        }
    }
    auto checkCuSeqlens = [&p](std::optional<torch::Tensor> const& cuSeqlens, char const* name)
    {
        if (!cuSeqlens.has_value())
        {
            return;
        }
        auto const& tensor = cuSeqlens.value();
        TORCH_CHECK(tensor.dim() == 1, name, " must be a 1-D tensor.");
        TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
        TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Int, name, " must be int32.");
        TORCH_CHECK(tensor.size(0) >= p.num_seqs + 1, name, " must have at least num_seqs + 1 elements.");
    };
    if (!isGen)
    {
        checkCuSeqlens(p.fwd.cu_q_seqlens, "cu_q_seqlens");
        checkCuSeqlens(p.fwd.cu_kv_seqlens, "cu_kv_seqlens");
    }

    p.relative_attention_bias_stride = 0;
    if (p.fwd.relative_attention_bias.has_value())
    {
        auto const& bias = p.fwd.relative_attention_bias.value();
        TORCH_CHECK(bias.dim() == 2 || bias.dim() == 3,
            "relative_attention_bias must be [num_heads, num_buckets] for implicit mode or "
            "[num_heads, max_seq_len, max_seq_len] for explicit mode");
        TORCH_CHECK(bias.is_contiguous(), "relative_attention_bias must be contiguous");
        TORCH_CHECK(bias.scalar_type() == p.qkv_or_q.scalar_type(),
            "relative_attention_bias dtype must match attention input dtype");
        p.relative_attention_bias_stride = static_cast<int32_t>(bias.size(1));
    }

    // Cross attention addresses the encoder KV by its own extents. The encoder lengths are
    // the phase-local sequence lengths, so no separate tensor is needed.
    p.num_encoder_tokens = 0;
    p.cross_kv_length = 0;
    if (p.cross_attention)
    {
        p.encoder_input_lengths = p.sequence_length;
        if (p.fwd.cross_kv.has_value() && p.num_seqs > 0)
        {
            p.num_encoder_tokens = p.getCrossKvNumTokens();
            p.cross_kv_length = p.getMaxHostPastKeyValueLength(p.seq_offset, p.num_seqs);
        }
    }

    // Speculative decoding extents come from the mask / position-offset tensor shapes and
    // only apply to the generation phase.
    if (isGen && p.is_spec_decoding_enabled && p.use_spec_decoding)
    {
        TORCH_CHECK(p.spec_decoding_generation_lengths.has_value(),
            "Expecting spec_decoding_generation_lengths in spec-dec mode.");
        TORCH_CHECK(p.spec_decoding_position_offsets_for_cpp.has_value(),
            "Expecting spec_decoding_position_offsets_for_cpp in spec-dec mode.");
        TORCH_CHECK(p.spec_decoding_packed_mask.has_value(), "Expecting spec_decoding_packed_mask in spec-dec mode.");

        auto const& positionOffsets = p.spec_decoding_position_offsets_for_cpp.value();
        // [batch_size, max_draft_len + 1]
        TORCH_CHECK(positionOffsets.dim() == 2, "spec_decoding_position_offsets_for_cpp must be 2-D.");
        p.spec_decoding_is_generation_length_variable = true;

        if (tensorrt_llm::common::isSM100Family())
        {
            TORCH_CHECK(p.spec_decoding_bl_tree_mask_offset.has_value(),
                "Expecting spec_decoding_bl_tree_mask_offset in trtllm-gen spec-dec mode.");
            TORCH_CHECK(p.spec_decoding_bl_tree_mask.has_value(),
                "Expecting spec_decoding_bl_tree_mask in trtllm-gen spec-dec mode.");
            TORCH_CHECK(p.spec_bl_tree_first_sparse_mask_offset_kv.has_value(),
                "Expecting spec_bl_tree_first_sparse_mask_offset_kv in trtllm-gen spec-dec mode.");
            // Blackwell uses the padded packed-mask row dim as the mask stride.
            auto const& packedMask = p.spec_decoding_packed_mask.value();
            TORCH_CHECK(packedMask.dim() == 3, "spec_decoding_packed_mask must be 3-D in trtllm-gen spec-dec mode.");
            p.spec_decoding_max_generation_length = static_cast<int32_t>(packedMask.size(1));
        }
        else
        {
            p.spec_decoding_max_generation_length = static_cast<int32_t>(positionOffsets.size(1));
        }
    }

    // Derived op state.
    mFMHAForceFP32Acc = p.type == tensorrt_llm::DataType::kBF16;
    // Static sparse MLA feeds the kernels a separately dequantized FP8 scratch pool,
    // so an NVFP4 paged cache still takes the FP8 paths. Both inputs arrive per call.
    mUseNvfp4MlaKvCache = p.quant_mode.hasFp4KvCache() && p.use_tllm_gen_sparse_attention
        && !p.fwd.sparse_runtime_params.sparse_attn_kv_lens.has_value()
        && p.fwd.sparse_runtime_params.aux_kv_cache_pool_ptr.has_value();
    if (p.is_mla_enable)
    {
        mFP8ContextMLA = (mSM == 90 || tensorrt_llm::common::isSM100Family(mSM) || mSM == 107 || mSM == 120)
            && (p.quant_mode.hasFp8KvCache() || mUseNvfp4MlaKvCache);
        mFP8GenerationMLA = p.quant_mode.hasFp8KvCache() || mUseNvfp4MlaKvCache;
    }
    mPagedKVCache = mPagedKVCache && p.use_kv_cache;
    bool const use_sage_attn = p.fwd.sage_attn_num_elts_per_blk_q > 0 || p.fwd.sage_attn_num_elts_per_blk_k > 0
        || p.fwd.sage_attn_num_elts_per_blk_v > 0;
    mFP8ContextFMHA
        = p.is_fp8_out || p.is_fp4_out || (p.quant_mode.hasFp8KvCache() && p.paged_context_fmha) || use_sage_attn;
    mFP8AttenOutput = p.is_fp8_out;
    mFuseFp4Quant = p.is_fp4_out;
    // Pre-check whether FMHA is supported in order to save memory allocation.
    if (mEnableContextFMHA)
    {
        mEnableContextFMHA = false;
        if (!(p.type == tensorrt_llm::DataType::kHALF || p.type == tensorrt_llm::DataType::kBF16))
        {
            TLLM_LOG_WARNING("Fall back to unfused MHA because of unsupported data type.");
        }
        else if (p.position_embedding_type == tensorrt_llm::kernels::PositionEmbeddingType::kRELATIVE)
        {
            TLLM_LOG_WARNING("Fall back to unfused MHA because of relative position embedding.");
        }
        else if (isCrossAttention(p) && useKVCache(p) && !mPagedKVCache)
        {
            // TODO: add the support for cross attention + contiguous kv cache.
            TLLM_LOG_WARNING("Fall back to unfused MHA because of cross attention + contiguous kv cache.");
        }
        else
        {
            mEnableContextFMHA = true;
        }
    }

    // Pre-Check of FP8 Context FMHA.
    if (mFP8ContextFMHA)
    {
        TLLM_CHECK_WITH_INFO(mEnableContextFMHA, "FP8 FMHA cannot be enabled because Context FMHA is not supported.");
        TLLM_CHECK_WITH_INFO(
            mSM == 89 || mSM == 90 || tensorrt_llm::common::isSM100Family(mSM) || mSM == 120 || mSM == 121,
            "FP8 FMHA can only be enabled on sm_89, sm_90, sm_100f, sm_120 or sm_121.");
    }

    // Pre-Check of FP8 Generation MLA.
    if (mFP8GenerationMLA)
    {
        TLLM_CHECK_WITH_INFO(p.is_mla_enable, "FP8 Generation MLA cannot be enabled because MLA is not supported.");
        TLLM_CHECK_WITH_INFO(
            mSM == 89 || mSM == 90 || tensorrt_llm::common::isSM100Family(mSM) || mSM == 120 || mSM == 121,
            "FP8 Generation MLA is supported on Ada, Hopper or Blackwell architecture.");
    }

    // Check requirements for FP4 output.
    TLLM_CHECK_WITH_INFO(!mFuseFp4Quant || mEnableContextFMHA, "Context FMHA must enable if fuse_fp4_quant is enabled");
    TLLM_CHECK_WITH_INFO(!mFuseFp4Quant || tensorrt_llm::common::isSM100Family(mSM) || mSM == 120 || mSM == 121,
        "fuse_fp4_quant only supports SM100f or SM120 or SM121 devices.");

    // Check requirements for FP4 KV cache.
    TLLM_CHECK_WITH_INFO(!p.quant_mode.hasFp4KvCache() || mFP8ContextFMHA || mUseNvfp4MlaKvCache,
        "FP4 KV cache requires FP8 context FMHA or static sparse MLA with an FP8 scratch pool");

    TLLM_CHECK(isRoPE(p) == (p.rotary_embedding_dim != 0));
    TLLM_CHECK_WITH_INFO((mSM >= 80) || (p.type != tensorrt_llm::DataType::kBF16),
        "Unsupported data type, pre SM 80 GPUs do not support bfloat16");

    // Pre-check whether the head size is supported by MMHA.
    // Support head size == 72 only for fmha kernels, so skip pre-check here.
    if (getHeadSize() == 72)
    {
        ;
    }
    else if (!mmha_supported(getHeadSize()) && !p.is_mla_enable)
    {
        TLLM_CHECK_WITH_INFO(false, "Head size %d is not supported by MMHA.", getHeadSize());
    }

    if (p.is_mla_enable)
    {
        TLLM_CHECK_WITH_INFO(mEnableContextFMHA, "MLA(Deepseek v2) only support fmha");
        TLLM_CHECK_WITH_INFO(!p.dense_context_fmha, "MLA(Deepseek v2) currently not support dense fmha");
        TLLM_CHECK_WITH_INFO(
            mPagedKVCache && p.use_kv_cache && p.remove_padding, "MLA(Deepseek v2) only support paged kv cache");
        TLLM_CHECK_WITH_INFO(!p.cross_attention, "MLA(Deepseek v2) do not support cross attention right now");
        TLLM_CHECK_WITH_INFO(p.mask_type != tensorrt_llm::kernels::AttentionMaskType::CUSTOM_MASK,
            "MLA(Deepseek v2) do not support custom mask right now");
        bool const mla_dims_supported = p.mla_params.qk_rope_head_dim == 64
            && ((p.mla_params.rope_append && p.mla_params.kv_lora_rank == 512)
                || (!p.mla_params.rope_append && p.mla_params.kv_lora_rank == 448));
        TLLM_CHECK_WITH_INFO(mla_dims_supported,
            "MLA(Deepseek v2) only supports qk_rope_head_dim=64 with kv_lora_rank=512 "
            "(rope_append=true) or "
            "kv_lora_rank=448 (rope_append=false).");
    }
    if (mEnableContextFMHA)
    {
        // Construct the fmha runner.
        MHARunnerFixedParams fmhaParams{};

        bool const useSageAttn = mFP8ContextFMHA && !p.is_mla_enable
            && (p.fwd.sage_attn_num_elts_per_blk_q > 0 || p.fwd.sage_attn_num_elts_per_blk_k > 0
                || p.fwd.sage_attn_num_elts_per_blk_v > 0);

        // Pre-checked during constructing.
        Data_type data_type, data_type_kv;
        if (p.type == tensorrt_llm::DataType::kHALF)
        {
            data_type = DATA_TYPE_FP16;
        }
        else if (p.type == tensorrt_llm::DataType::kBF16)
        {
            data_type = DATA_TYPE_BF16;
        }
        else
        {
            TLLM_CHECK_WITH_INFO(false, "GPTAttentionPlugin received wrong data type.");
        }
        // The output dtype.
        fmhaParams.dataTypeOut = mFP8AttenOutput ? DATA_TYPE_E4M3 : data_type;
        data_type_kv = data_type;

        // FP8 FMHA should be used with fp8 workflow together.
        if (mFP8ContextFMHA || mFP8ContextMLA)
        {
            if (mFP8ContextFMHA && useSageAttn && p.fwd.sage_attn_qk_int8)
            {
                data_type = DATA_TYPE_INT8;
                data_type_kv = DATA_TYPE_KV_INT8_E4M3;
            }
            else
            {
                data_type = DATA_TYPE_E4M3;
                data_type_kv = DATA_TYPE_E4M3;
            }
        }

        // The input dtype.
        fmhaParams.dataType = data_type;
        // The KV input data type. The default is same as dataType.
        fmhaParams.dataTypeKv = data_type_kv;
        // If the kernel must read from KV cache, set the dtype correctly.
        if (mPagedKVCache && p.paged_context_fmha)
        {
            if (p.quant_mode.hasFp8KvCache())
            {
                fmhaParams.dataTypeKv = DATA_TYPE_E4M3;
            }
            else if (p.quant_mode.hasFp4KvCache())
            {
                fmhaParams.dataTypeKv = DATA_TYPE_E2M1;
            }
        }
        if (mFuseFp4Quant)
        {
            // If FP4 quantization workflow is enabled, set output type to FP4.
            fmhaParams.dataTypeOut = DATA_TYPE_E2M1;
        }
        if (p.is_mla_enable)
        {
            // For FP8 MLA, currently context attention is performed in BF16.
            fmhaParams.dataTypeOut = DATA_TYPE_BF16;
            fmhaParams.dataTypeKv = DATA_TYPE_BF16;
        }
        if (mFP8ContextMLA)
        {
            fmhaParams.dataTypeKv = DATA_TYPE_E4M3;
            fmhaParams.dataTypeOut = DATA_TYPE_BF16;
        }
        if (p.fuses_dsv4_inv_rope_fp8_quant)
        {
            fmhaParams.dataTypeOut = DATA_TYPE_E4M3;
        }
        // TODO: remove forceFp32Acc from MHARunnerFixedParams after adding host_runtime_perf_knobs to
        // bertAttentionPlugin input tensors, so that we can change mLaunchParams.force_fp32_acc value
        // in runtime.
        fmhaParams.forceFp32Acc = false;

        // setting attention mask type based on the mask type
        fmhaParams.setAttentionMaskType(static_cast<std::int8_t>(p.mask_type));

        if (isCrossAttention(p))
        {
            // always use paged-kv-fmha if paged_kv cache is used.
            fmhaParams.attentionInputLayout
                = mPagedKVCache ? AttentionInputLayout::Q_PAGED_KV : AttentionInputLayout::Q_CONTIGUOUS_KV;
        }
        else if (!useKVCache(p))
        {
            if (useSageAttn)
            {
                fmhaParams.attentionInputLayout = AttentionInputLayout::SEPARATE_Q_K_V;
            }
            else
            {
                fmhaParams.attentionInputLayout = AttentionInputLayout::PACKED_QKV;
            }
        }
        else
        {
            fmhaParams.attentionInputLayout = (mPagedKVCache && p.paged_context_fmha)
                ? AttentionInputLayout::Q_PAGED_KV
                : AttentionInputLayout::PACKED_QKV;
        }
        fmhaParams.isSPadded = !p.remove_padding;
        fmhaParams.numQHeads = mNumAttnHeads;
        fmhaParams.numKvHeads = mNumAttnKVHeads;
        fmhaParams.numTokensPerBlock = p.tokens_per_block;
        fmhaParams.headSize = mHeadSize;
        fmhaParams.headSizeV = mHeadSize;
        fmhaParams.qScaling = p.q_scaling;

        // mFmhaDispatcher is not used for generation MLA, but we still need to modify these values to
        // avoid selecting the wrong kernel, no matter mIsGenerationMLA is true or false
        if (p.is_mla_enable)
        {
            if (useSparseMLA(p))
            {
                fmhaParams.attentionInputLayout = AttentionInputLayout::Q_PAGED_KV;
                fmhaParams.numKvHeads = 1;
                fmhaParams.headSize = p.mla_params.kv_lora_rank + p.mla_params.qk_rope_head_dim;
                fmhaParams.headSizeV = p.mla_params.rope_append
                    ? p.mla_params.kv_lora_rank
                    : p.mla_params.kv_lora_rank + p.mla_params.qk_rope_head_dim;
                fmhaParams.headSizeQkNope = p.mla_params.qk_nope_head_dim;
                // Adjust the qScaling for the absorption mode.
                fmhaParams.qScaling = p.q_scaling
                    * sqrt((float) (p.mla_params.qk_nope_head_dim + p.mla_params.qk_rope_head_dim))
                    / sqrtf((float) (p.mla_params.kv_lora_rank + p.mla_params.qk_rope_head_dim));
            }
            else
            {
                // Context MLA always use separate_q_k_v layout
                fmhaParams.attentionInputLayout = AttentionInputLayout::SEPARATE_Q_K_V;
                // Context attention of MLA is different
                fmhaParams.numKvHeads = p.num_heads;
                fmhaParams.headSize = p.mla_params.qk_nope_head_dim + p.mla_params.qk_rope_head_dim;
                // Ideally this should be p.mla_params.v_head_dim, but because we initialize both MLA
                // context(v_head_dim=128) and gen(v_head_dim=512) runners in a single op, the headSizeV
                // will be set to 512 when we create the gen attention op and that could fail to create
                // the FmhaDispatcher for context phase. Luckily, for deepseek, qk_nope_head_dim is the
                // same as v_head_dim in context phase.
                fmhaParams.headSizeV = p.mla_params.qk_nope_head_dim;
                fmhaParams.headSizeQkNope = p.mla_params.qk_nope_head_dim;
            }
        }
        fmhaParams.attnLogitSoftcappingScale = p.attn_logit_softcapping_scale;
        fmhaParams.hasAlibi = isALiBi(p);
        fmhaParams.scaleAlibi = isAliBiWithScale(p);
        fmhaParams.useSparseMLA = useSparseMLA(p);
        fmhaParams.useTllmGenSparseAttention = useTllmGenSparseAttention(p);
        fmhaParams.fusesDsv4InvRopeFp8Quant = p.fuses_dsv4_inv_rope_fp8_quant;

        // SageAttention: set block sizes for sage quantization.
        if (useSageAttn)
        {
            fmhaParams.sageBlockSizeQ = p.fwd.sage_attn_num_elts_per_blk_q;
            fmhaParams.sageBlockSizeK = p.fwd.sage_attn_num_elts_per_blk_k;
            fmhaParams.sageBlockSizeV = p.fwd.sage_attn_num_elts_per_blk_v;
        }

        // Load kernels from the pre-compiled cubins.
        mFmhaDispatcher.reset(new FmhaDispatcher(fmhaParams));

        // Deepseek-V2 Generation needs a differ fmha with different argumments
        if (p.is_mla_enable)
        {
            mEnableXQA = (mSM == kSM_120) && mIsGenerationMLA;
            if (mUseTllmGen)
            {
                Data_type qDataType = DATA_TYPE_FP32;
                Data_type kvDataType = DATA_TYPE_FP32;
                Data_type outputDataType = DATA_TYPE_FP32;

                if (p.type == tensorrt_llm::DataType::kHALF)
                {
                    qDataType = DATA_TYPE_FP16;
                    kvDataType = DATA_TYPE_FP16;
                    outputDataType = DATA_TYPE_FP16;
                }
                else if (p.type == tensorrt_llm::DataType::kBF16)
                {
                    qDataType = DATA_TYPE_BF16;
                    kvDataType = DATA_TYPE_BF16;
                    outputDataType = DATA_TYPE_BF16;
                }
                else
                {
                    TLLM_CHECK_WITH_INFO(false, "The data type is not supported.");
                }

                if (mFP8GenerationMLA)
                {
                    qDataType = DATA_TYPE_E4M3;
                    kvDataType = DATA_TYPE_E4M3;
                }
                if (p.fuses_dsv4_inv_rope_fp8_quant)
                {
                    outputDataType = DATA_TYPE_E4M3;
                }

                // Instantiate the mTllmGenFMHARunner used for MLA
                mTllmGenFMHARunner.reset(new TllmGenFmhaRunner(
                    qDataType, kvDataType, kvDataType, outputDataType, 0, 0, 0, 0, p.fuses_dsv4_inv_rope_fp8_quant));
            }
            else if (mIsGenerationMLA && !mUseGenFlashMLA)
            {
                // Construct the fmha runner for generation.
                if (mFP8GenerationMLA)
                {
                    data_type = DATA_TYPE_E4M3;
                }
                MHARunnerFixedParams fmhaParams{};
                fmhaParams.dataType = data_type;
                fmhaParams.dataTypeKv = data_type;
                fmhaParams.dataTypeOut = data_type;
                // For FP8 MLA generation, the output type is BF16, and the quantization before o_proj
                // is performed separately.
                if (mFP8GenerationMLA)
                {
                    fmhaParams.dataTypeOut = DATA_TYPE_BF16;
                }
                // TODO: remove forceFp32Acc from MHARunnerFixedParams after adding
                // host_runtime_perf_knobs to bertAttentionPlugin input tensors, so that we can change
                // mLaunchParams.force_fp32_acc value in runtime.
                fmhaParams.forceFp32Acc = true;
                fmhaParams.attentionMaskType
                    = useCustomMask(p) ? ContextAttentionMaskType::CUSTOM_MASK : ContextAttentionMaskType::PADDING;
                // TODO: set it to Q_CONTIGUOUS_KV layout for cross-attention.
                fmhaParams.attentionInputLayout = AttentionInputLayout::Q_PAGED_KV;
                fmhaParams.isSPadded = !p.remove_padding;
                fmhaParams.numQHeads = 1;
                fmhaParams.numKvHeads = 1;
                fmhaParams.headSize = p.mla_params.kv_lora_rank + p.mla_params.qk_rope_head_dim;
                fmhaParams.headSizeV = p.mla_params.kv_lora_rank;
                fmhaParams.qScaling = p.q_scaling
                    * sqrt((float) (p.mla_params.qk_nope_head_dim + p.mla_params.qk_rope_head_dim))
                    / sqrtf((float) (p.mla_params.kv_lora_rank + p.mla_params.qk_rope_head_dim));
                fmhaParams.attnLogitSoftcappingScale = p.attn_logit_softcapping_scale;
                fmhaParams.hasAlibi = isALiBi(p);
                fmhaParams.scaleAlibi = isAliBiWithScale(p);
                fmhaParams.tpSize = 1;
                fmhaParams.tpRank = 0;
                mDecoderFMHARunner.reset(new FusedMHARunnerV2(fmhaParams));

                // Only deepseek must using fmha in the generation phase when flash mla is not enabled.
                if (!mUseGenFlashMLA)
                {
                    TLLM_CHECK_WITH_INFO(mDecoderFMHARunner->isFmhaSupported(),
                        "Deepseek should be supported by fmha in generation part.");
                }
            }
            if (!mIsGenerationMLA)
            {
                TLLM_CHECK_WITH_INFO(
                    mFmhaDispatcher->isSupported(), "Deepseek should be supported by fmha in context part.");
            }
        }

        // Fall back to unfused MHA kernels if not supported.
        // Generation MLA reuses the context FMHA code path so set mEnableContextFMHA to true.
        // However, do not check mFmhaDispatcher which is not used for generation MLA.
        mEnableContextFMHA = mIsGenerationMLA || mFmhaDispatcher->isSupported();

        // Only FMHA supports custom mask currently.
        TLLM_CHECK_WITH_INFO(
            !useCustomMask(p) || mEnableContextFMHA, "Only Context FMHA supports custom mask input currently.");
    }

    mEnableXQA = (mEnableXQA || p.is_spec_decoding_enabled)
        && (p.type == tensorrt_llm::DataType::kHALF || p.type == tensorrt_llm::DataType::kBF16) && p.use_kv_cache;

    if (mEnableXQA)
    {
        TLLM_LOG_DEBUG("Enabling XQA kernels for GPTAttention.");

        XqaFixedParams fixedParams{};
        fixedParams.isMLA = mIsGenerationMLA;
        // TODO: support more combinations.
        // Update Q and O dtype.
        if (p.type == tensorrt_llm::DataType::kHALF)
        {
            fixedParams.inputDataType = DATA_TYPE_FP16;
            fixedParams.outputDataType = DATA_TYPE_FP16;
        }
        else if (p.type == tensorrt_llm::DataType::kBF16)
        {
            fixedParams.inputDataType = DATA_TYPE_BF16;
            fixedParams.outputDataType = DATA_TYPE_BF16;
        }
        // Update KV cache and math dtype.
        if (p.quant_mode.hasInt8KvCache())
        {
            fixedParams.kvDataType = DATA_TYPE_INT8;
            fixedParams.mathDataType = fixedParams.inputDataType;
        }
        else if (p.quant_mode.hasFp8KvCache())
        {
            fixedParams.kvDataType = DATA_TYPE_E4M3;
            fixedParams.mathDataType = DATA_TYPE_E4M3;
        }
        else if (p.quant_mode.hasFp4KvCache())
        {
            fixedParams.kvDataType = DATA_TYPE_E2M1;
            fixedParams.mathDataType = DATA_TYPE_E4M3;
        }
        else
        {
            fixedParams.kvDataType = fixedParams.inputDataType;
            fixedParams.mathDataType = fixedParams.inputDataType;
        }
        // If fuse_fp4_quant is enabled, set output data type to FP4.
        if (mFuseFp4Quant)
        {
            fixedParams.outputDataType = DATA_TYPE_E2M1;
        }
        else if (mFP8AttenOutput)
        {
            fixedParams.outputDataType = DATA_TYPE_E4M3;
        }
        if (p.is_spec_decoding_enabled && !mUseTllmGen)
        {
            fixedParams.outputDataType = DATA_TYPE_E4M3;
            TLLM_CHECK_WITH_INFO(p.num_heads % mNumKVHeads == 0, "p.num_heads should be multiples of mNumKVHeads.");
        }

        fixedParams.numQHeads = mNumAttnHeads;
        fixedParams.numKvHeads = mNumAttnKVHeads;
        fixedParams.numTokensPerBlock = p.tokens_per_block;
        fixedParams.headSize = mHeadSize;
        fixedParams.qScaling = p.q_scaling;
        fixedParams.multiBlockMode = mMultiBlockMode;
        fixedParams.isPagedKv = mPagedKVCache;
        fixedParams.isSpecDecoding = p.is_spec_decoding_enabled;
        fixedParams.hasAlibi = isALiBi(p);
        fixedParams.useTllmGenSparseAttention = useTllmGenSparseAttention(p);
        fixedParams.specDecodingTargetMaxGenLen = p.spec_decoding_target_max_gen_len;

        mXqaDispatcher.reset(new XqaDispatcher(fixedParams));

        // Fall back to unfused MHA kernels if not supported.
        mEnableXQA = mXqaDispatcher->isSupported();
    }
    else if (p.is_spec_decoding_enabled)
    {
        TLLM_CHECK_WITH_INFO(false, "Speculative decoding mode doesn't support the data type or cross attention.");
    }

#if ENABLE_MULTI_DEVICE
#endif // ENABLE_MULTI_DEVICE
    DISPATCH_ON_DTYPE(p.type, prepareEnqueueGeneration, p);

    p.sparse_params = {};
    p.sparse_params.sparse_kv_indices = p.getSparseKvIndices();
    p.sparse_params.sparse_kv_offsets = p.getSparseKvOffsets();
    p.sparse_params.sparse_attn_indices = p.getSparseAttnIndices();
    p.sparse_params.sparse_attn_offsets = p.getSparseAttnOffsets();
    p.sparse_params.sparse_attn_indices_block_size = p.fwd.sparse_runtime_params.sparse_attn_indices_block_size;
    p.sparse_params.sparse_attn_indices_stride = p.getSparseAttnIndicesStride();
    p.sparse_params.num_sparse_topk = p.num_sparse_topk;
    p.sparse_params.sparse_attn_kv_lens = p.getSparseAttnKvLens();

    p.kv_cache_pool_pointers = {};
    if (!AttentionOp::useKVCache(p) || !p.hasKvCache())
    {
        return 0;
    }

    int32_t const poolIndex = p.getKvCachePoolIndex(p.local_layer_idx);
    int32_t const layerIdxInCachePool = p.getLayerIdxInCachePool(p.local_layer_idx);
    size_t const dTypeSize = p.type == tensorrt_llm::DataType::kFLOAT ? sizeof(float) : sizeof(half);
    int const cacheElemBits = AttentionOp::getKvCacheElemSizeInBits(p.quant_mode, dTypeSize);
    auto const blockSize = static_cast<int64_t>(p.tokens_per_block) * mNumKVHeads * mHeadSize;
    auto const bytesPerBlock = blockSize * cacheElemBits / CHAR_BIT;
    int32_t const kvFactor = AttentionOp::isMLAEnabled(p) ? 1 : 2;
    auto const intraPoolOffset = layerIdxInCachePool * kvFactor * bytesPerBlock;

    p.kv_cache_pool_pointers = buildKvCachePoolPointers(p.getHostKvCachePoolPointers(), poolIndex, intraPoolOffset,
        blockSize, layerIdxInCachePool, kvFactor, p.quant_mode.hasFp4KvCache());

    if (p.use_sparse_attention)
    {
        auto* kvCachePool = p.getSparseKvCachePool(poolIndex);
        if (kvCachePool != nullptr)
        {
            if (p.fwd.sparse_runtime_params.sparse_attn_kv_lens.has_value())
            {
                // Deepseek V4 dynamic sparse MLA always uses the SWA pool for now.
                p.sparse_params.sliding_window_kv_cache_pool = kvCachePool;
                if (p.fwd.sparse_runtime_params.aux_kv_cache_pool_ptr.has_value())
                {
                    p.sparse_params.sparse_kv_cache_pool
                        = (void*) (intptr_t) p.fwd.sparse_runtime_params.aux_kv_cache_pool_ptr.value();
                }
            }
            else
            {
                p.sparse_params.sparse_kv_cache_pool = kvCachePool;
            }
        }
    }
    return 0;
}

int64_t AttentionOp::getAttentionWorkspaceSize(FmhaParams const& p, int64_t num_tokens,
    int64_t max_attention_window_size, int64_t num_gen_tokens, int64_t max_blocks_per_sequence,
    int64_t ctx_total_kv_len)
{
    auto params = p;
    AttentionOp& op = *this;
    op.prepare(params, /*isGen=*/false);
    // For cross-attention, several unfused-path context buffers scale with the encoder KV length.
    // Mirror the context-stage enqueue, which uses the max past-KV length over the context sequences
    // as cross_kv_length; sizing with 0 here under-allocates the workspace and the carved views in
    // enqueueContext land past the end of the allocation. The enqueue also gates on
    // cross_kv.has_value(), so this can over-allocate relative to the carve; that is safe.
    int32_t maxCrossKvLength = 0;
    if (AttentionOp::isCrossAttention(params) && params.num_seqs > 0)
    {
        maxCrossKvLength = params.getMaxHostPastKeyValueLength(0, params.num_seqs);
    }
    size_t const contextWorkspaceSize = op.getWorkspaceSizeForContext(params, static_cast<int>(params.max_num_requests),
        static_cast<int>(params.max_context_length), maxCrossKvLength, static_cast<int>(num_tokens),
        static_cast<int>(ctx_total_kv_len));
    // The generation workspace is sized per sequence (max_num_requests * beam_width), not
    // per request; they only coincide when beam_width == 1.
    int64_t const maxNumSequences = params.max_num_sequences > 0 ? params.max_num_sequences : params.max_num_requests;
    size_t const generationWorkspaceSize = op.getWorkspaceSizeForGeneration(params, static_cast<int>(maxNumSequences),
        static_cast<int>(max_attention_window_size), static_cast<int>(num_gen_tokens),
        static_cast<int>(max_blocks_per_sequence));
    return static_cast<int64_t>(std::max(contextWorkspaceSize, generationWorkspaceSize));
}

void AttentionOp::runContext(FmhaParams& p)
{
    DISPATCH_ON_TORCH_DTYPE(p.qkv_or_q.scalar_type(), runContextImpl, p);
}

void AttentionOp::runGeneration(FmhaParams& p)
{
    DISPATCH_ON_TORCH_DTYPE(p.qkv_or_q.scalar_type(), runGenerationImpl, p);
}

void AttentionOp::runMlaGeneration(FmhaParams& p)
{
    DISPATCH_ON_TORCH_DTYPE(p.qkv_or_q.scalar_type(), runMlaGenerationImpl, p);
}

bool attention_supports_nvfp4_output(int64_t const num_heads, int64_t const num_kv_heads, int64_t const head_size,
    std::optional<int64_t> const tokens_per_block, int64_t const mask_type, int64_t const quant_mode,
    bool const use_paged_context_fmha, bool is_mla_enable)
{
    // Only Blackwell supports NVFP4 output.
    // SM 120 does not support NVFP4 output.
    if (tensorrt_llm::common::getSMVersion() < 100 || tensorrt_llm::common::getSMVersion() == 120)
    {
        return false;
    }

    // MLA is not supported.
    if (is_mla_enable)
    {
        return false;
    }

    // A capability query has no call to describe, but `prepare` derives the dtypes from
    // the input and output tensors. Hand it empty ones carrying just those dtypes: half
    // in, uint8 out, which is how an NVFP4 output is typed.
    FmhaParams p;
    p.qkv_or_q = torch::empty({0}, torch::dtype(torch::kHalf));
    p.output = torch::empty({0}, torch::dtype(torch::kUInt8));
    p.num_heads = num_heads;
    p.num_kv_heads = num_kv_heads;
    p.head_size = head_size;
    p.mask_type = static_cast<tensorrt_llm::kernels::AttentionMaskType>(int32_t(mask_type));
    p.quant_mode = tensorrt_llm::common::QuantMode(uint32_t(quant_mode));
    p.unidirectional = 1;
    p.remove_padding = true;
    p.use_kv_cache = true;
    p.tokens_per_block = tokens_per_block.value_or(0);
    p.paged_context_fmha = use_paged_context_fmha;

    StaticAttentionConfig cfg{};
    cfg.num_heads = p.num_heads;
    cfg.num_kv_heads = num_kv_heads;
    cfg.head_size = head_size;
    cfg.tokens_per_block = p.tokens_per_block;
    cfg.quant_mode = p.quant_mode;

    AttentionOp op{cfg};
    op.mPagedKVCache = true;
    op.prepare(p, /*isGen=*/false);

    return op.supportsNvFp4Output(p);
}

KvCachePoolPointers buildKvCachePoolPointers(at::Tensor const& hostKvCachePoolPointers, int32_t poolIndex,
    int64_t intraPoolOffset, int64_t blockSize, int32_t layerIdxInCachePool, int32_t kvFactor, bool isFp4KvCache)
{
    KvCachePoolPointers pointers;
    if (isFp4KvCache)
    {
        // For NVFP4 KV cache, extra block scales are stored in separate pools.
        // The layout of host_kv_cache_pool_pointers is [num_pools, 2 (primary and secondary), 2 (data
        // and scale)].
        TORCH_CHECK(hostKvCachePoolPointers.dim() == 3);
        pointers.primaryPoolPtr = reinterpret_cast<void*>(
            reinterpret_cast<char*>(
                readHostTensor3D<int64_t>(hostKvCachePoolPointers, poolIndex, 0, 0, "host_kv_cache_pool_pointers"))
            + intraPoolOffset);
        pointers.secondaryPoolPtr = reinterpret_cast<void*>(
            reinterpret_cast<char*>(
                readHostTensor3D<int64_t>(hostKvCachePoolPointers, poolIndex, 1, 0, "host_kv_cache_pool_pointers"))
            + intraPoolOffset);
        // NVFP4 block scaling uses a fixed vector size of 16.
        auto constexpr vectorSize = 16;
        auto const bytesPerBlockSf = blockSize / vectorSize * 1 /*bytes per E4M3 sf*/;
        auto const intraPoolOffsetSf = layerIdxInCachePool * kvFactor * bytesPerBlockSf;
        pointers.primaryBlockScalePoolPtr = reinterpret_cast<void*>(
            reinterpret_cast<char*>(
                readHostTensor3D<int64_t>(hostKvCachePoolPointers, poolIndex, 0, 1, "host_kv_cache_pool_pointers"))
            + intraPoolOffsetSf);
        pointers.secondaryBlockScalePoolPtr = reinterpret_cast<void*>(
            reinterpret_cast<char*>(
                readHostTensor3D<int64_t>(hostKvCachePoolPointers, poolIndex, 1, 1, "host_kv_cache_pool_pointers"))
            + intraPoolOffsetSf);
    }
    else
    {
        TORCH_CHECK(hostKvCachePoolPointers.dim() == 2);
        pointers.primaryPoolPtr = reinterpret_cast<void*>(
            reinterpret_cast<char*>(
                readHostTensor2D<int64_t>(hostKvCachePoolPointers, poolIndex, 0, "host_kv_cache_pool_pointers"))
            + intraPoolOffset);
        pointers.secondaryPoolPtr = reinterpret_cast<void*>(
            reinterpret_cast<char*>(
                readHostTensor2D<int64_t>(hostKvCachePoolPointers, poolIndex, 1, "host_kv_cache_pool_pointers"))
            + intraPoolOffset);
    }
    return pointers;
}

KvCacheBuffers<kernels::KVBlockArray> buildPagedKvCacheBuffers(
    std::optional<torch::Tensor> const& kv_cache_block_offsets,
    std::optional<torch::Tensor> const& host_kv_cache_pool_pointers,
    std::optional<torch::Tensor> const& host_kv_cache_pool_mapping, common::QuantMode quantMode, int64_t layer_idx,
    int64_t batch_size, int64_t tokens_per_block, int64_t kv_head_num, int64_t size_per_head,
    int64_t cyclic_attention_window_size, int64_t max_attention_window_size, int64_t beam_width, int64_t seq_offset,
    bool is_mla_enable, size_t elem_size)
{
    using kernels::KVBlockArray;

    bool const useKvCache = kv_cache_block_offsets.has_value() && host_kv_cache_pool_pointers.has_value()
        && host_kv_cache_pool_mapping.has_value();
    if (!useKvCache)
    {
        return {};
    }

    auto const mapping = readKvCachePoolMapping(host_kv_cache_pool_mapping.value(), layer_idx);
    int32_t const poolIndex = mapping.poolIndex;
    int32_t const layerIdxInCachePool = mapping.layerIdxInCachePool;
    auto* blockOffsets = tensorPtr2D<KVBlockArray::DataType>(
        kv_cache_block_offsets.value(), poolIndex, static_cast<int64_t>(seq_offset), "kv_cache_block_offsets");

    int cacheElemBits = AttentionOp::getKvCacheElemSizeInBits(quantMode, elem_size);

    auto const blockSize = tokens_per_block * kv_head_num * size_per_head;
    auto const bytesPerBlock = blockSize * cacheElemBits / CHAR_BIT;
    int32_t const kvFactor = is_mla_enable ? 1 : 2;
    auto const intraPoolOffset = layerIdxInCachePool * kvFactor * bytesPerBlock;
    auto const sizePerToken = static_cast<int32_t>(kv_head_num * size_per_head * cacheElemBits / 8);

    auto poolPointers = buildKvCachePoolPointers(host_kv_cache_pool_pointers.value(), poolIndex, intraPoolOffset,
        blockSize, layerIdxInCachePool, kvFactor, quantMode.hasFp4KvCache());

    int32_t const maxBlocksPerSequence = static_cast<int32_t>(kv_cache_block_offsets->size(-1));
    return buildKvCacheBuffers<kernels::KVBlockArray>(static_cast<int32_t>(batch_size), maxBlocksPerSequence,
        static_cast<int32_t>(tokens_per_block), sizePerToken, static_cast<int32_t>(cyclic_attention_window_size),
        static_cast<int32_t>(std::max(cyclic_attention_window_size, max_attention_window_size)),
        /*sink_token_length=*/0, beam_width > 1, poolPointers.primaryPoolPtr, poolPointers.secondaryPoolPtr,
        poolPointers.primaryBlockScalePoolPtr, poolPointers.secondaryBlockScalePoolPtr, blockOffsets,
        quantMode.hasFp4KvCache());
}

std::tuple<at::Tensor, std::optional<at::Tensor>> buildFlashinferTrtllmGenPagedKvCacheBuffers(
    at::Tensor host_kv_cache_pool_pointers, at::Tensor host_kv_cache_pool_mapping, int64_t layer_idx,
    int64_t num_kv_heads, int64_t tokens_per_block, int64_t head_dim, int64_t kv_factor, int64_t total_num_blocks,
    int64_t kv_cache_quant_mode, at::ScalarType dtype)
{
    auto const mapping = readKvCachePoolMapping(host_kv_cache_pool_mapping, layer_idx);
    int32_t const poolIndex = mapping.poolIndex;
    int32_t const layerIdxInCachePool = mapping.layerIdxInCachePool;

    auto quantMode = tensorrt_llm::common::QuantMode(static_cast<uint32_t>(kv_cache_quant_mode));
    bool const isFp4 = quantMode.hasFp4KvCache();

    size_t const inputElemSize = isFp4 ? 1 : (quantMode.hasFp8KvCache() || quantMode.hasInt8KvCache() ? 1 : 2);
    int const cacheElemBits = AttentionOp::getKvCacheElemSizeInBits(quantMode, inputElemSize);

    auto const blockSize = tokens_per_block * num_kv_heads * head_dim;
    auto const bytesPerBlock = blockSize * cacheElemBits / CHAR_BIT;
    auto const intraPoolOffset = layerIdxInCachePool * kv_factor * bytesPerBlock;

    auto poolPointers = buildKvCachePoolPointers(host_kv_cache_pool_pointers, poolIndex, intraPoolOffset, blockSize,
        layerIdxInCachePool, static_cast<int32_t>(kv_factor), isFp4);
    TORCH_CHECK(poolPointers.primaryPoolPtr != nullptr, "Primary KV cache pool pointer is null.");

    at::ScalarType storageDtype = dtype;
    if (quantMode.hasFp8KvCache())
        storageDtype = at::kFloat8_e4m3fn;
    else if (quantMode.hasInt8KvCache())
        storageDtype = at::kByte;
    else if (quantMode.hasFp4KvCache())
        storageDtype = at::kByte; // FP4 packed as bytes

    int64_t containerDim = isFp4 ? head_dim / 2 : head_dim;

    // Flat-block KV cache: [total_blocks, num_kv_heads, tokens_per_block, containerDim]
    auto options = at::TensorOptions()
                       .dtype(storageDtype)
                       .device(c10::Device(at::kCUDA, static_cast<c10::DeviceIndex>(at::cuda::current_device())));
    auto kv_pool = torch::from_blob(
        poolPointers.primaryPoolPtr, {total_num_blocks, num_kv_heads, tokens_per_block, containerDim}, options);

    std::optional<at::Tensor> kvScalePool = std::nullopt;
    if (isFp4 && poolPointers.primaryBlockScalePoolPtr != nullptr)
    {
        auto scaleOptions
            = at::TensorOptions()
                  .dtype(at::kFloat8_e4m3fn)
                  .device(c10::Device(at::kCUDA, static_cast<c10::DeviceIndex>(at::cuda::current_device())));
        kvScalePool = torch::from_blob(poolPointers.primaryBlockScalePoolPtr,
            {total_num_blocks, num_kv_heads, tokens_per_block, head_dim / 16}, scaleOptions);
    }

    return {kv_pool, kvScalePool};
}

} // namespace torch_ext

void computeFlashMlaMetadata(torch::Tensor seqlens_k, torch::Tensor tile_scheduler_metadata, torch::Tensor num_splits,
    int64_t batch_size, int64_t s_q, int64_t num_q_heads, int64_t num_kv_heads, int64_t head_size_v)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(seqlens_k.get_device());
    static constexpr int block_size_n = 64;
    static constexpr int fixed_overhead_num_blocks = 5;
    int const num_sm_parts = tensorrt_llm::torch_ext::AttentionOp::getFlashMlaNumSmPartsStatic(static_cast<int>(s_q),
        static_cast<int>(num_q_heads), static_cast<int>(num_kv_heads), static_cast<int>(head_size_v));
    Mla_metadata_params params = {};
    params.seqlens_k_ptr = seqlens_k.data_ptr<int>();
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
    params.num_splits_ptr = num_splits.data_ptr<int>();
    params.batch_size = static_cast<int>(batch_size);
    params.block_size_n = block_size_n;
    params.fixed_overhead_num_blocks = fixed_overhead_num_blocks;
    params.num_sm_parts = num_sm_parts;
    get_mla_metadata_func(params, stream);
}

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("attention_supports_nvfp4_output", &tensorrt_llm::torch_ext::attention_supports_nvfp4_output);
}
