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
#include "tensorrt_llm/common/attentionWorkspace.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/tllmDataType.h"
#include "tensorrt_llm/kernels/flashMLA/flash_mla.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/mlaKernels.h"
#include "tensorrt_llm/kernels/sparseAttentionKernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/debugUtils.h"
#include "tensorrt_llm/thop/attentionOp.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <cstdint>
#include <functional>
#include <torch/extension.h>
#include <tuple>
#include <type_traits>
#include <utility>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
using tensorrt_llm::common::op::AttentionOp;
using tensorrt_llm::common::op::AttentionStaticConfig;
using tensorrt_llm::common::op::AttentionWorkspaceManager;

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

namespace trtllm::attention
{
using tensorrt_llm::kernels::KVBlockArray;
using tensorrt_llm::kernels::MlaParams;
using tensorrt_llm::kernels::SparseAttentionParams;
using tensorrt_llm::torch_ext::KvCachePoolPointers;
using tensorrt_llm::torch_ext::buildKvCachePoolPointers;

#ifdef ENABLE_BF16
#define _DISPATCH_ON_DTYPE_BF16(FN, ...)                                                                               \
    case tensorrt_llm::DataType::kBF16: FN<__nv_bfloat16>(__VA_ARGS__); break;
#else
#define _DISPATCH_ON_DTYPE_BF16(FN, ...)
#endif
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

template <typename T>
void attnPrepare(AttentionOp& op)
{
    AttentionOp::EnqueueGenerationParams<T> enqueueParams;
    enqueueParams.max_attention_window_size = op.mConfig.attention_window_size;
    enqueueParams.cyclic_attention_window_size = op.mConfig.attention_window_size;
    enqueueParams.max_cyclic_attention_window_size = op.mConfig.attention_window_size;
    enqueueParams.beam_width = op.mConfig.beam_width;
    enqueueParams.num_requests = op.mConfig.max_num_requests;

    op.prepareEnqueueGeneration<T, KVBlockArray>(enqueueParams);
}

template <class ParamsT>
void extractHelixParams(FmhaParams const& p, ParamsT& params)
{
    params.helix_position_offsets = p.getHelixPositionOffsets();
    params.helix_is_inactive_rank = p.getHelixIsInactiveRank();
}

template <typename T>
class EnqueueParamsBuilder
{
public:
    EnqueueParamsBuilder(AttentionOp& op, FmhaParams const& p)
        : mOp(op)
        , mParams(p)
    {
        mStream = at::cuda::getCurrentCUDAStream(p.qkv_or_q.get_device());
        mAttentionInput = p.getQkvOrQ<T>(p.token_offset);
        mContextBuf = p.getOutput(p.token_offset);
        TORCH_CHECK(!op.mFuseFp4Quant || p.output_sf.has_value());
        TORCH_CHECK(!p.enable_dsv4_epilogue_fusion || p.output_sf.has_value());
        void* contextBufSf = (op.mFuseFp4Quant || p.enable_dsv4_epilogue_fusion) ? p.getOutputSf() : nullptr;

        float const* rotaryInvFreqPtr = nullptr;
        if (op.isRoPE())
        {
            if (p.rotary_inv_freq.has_value())
            {
                rotaryInvFreqPtr = p.getRotaryInvFreq();
            }
            if (p.rotary_cos_sin.has_value())
            {
                mRotaryCosSinPtr = p.getRotaryCosSin();
            }
        }

        int const* contextLengthsPtr = p.getContextLengths(p.seq_offset);
        mSequenceLengthsPtr = p.getSequenceLength(p.seq_offset);
        // Note we still need context length during generation for MMHA optimization.
        // For encoder CUDA graphs compatibility, allow the caller to override the
        // max context Q length so FMHA kernel launch params (mMaxSeqLenQ-driven grid
        // and cluster dims) are stable across graph replays even when actual per-batch
        // sequence lengths vary.
        int32_t const maxContextQLenComputed = p.getMaxHostContextLength(p.seq_offset, p.num_seqs);
        int32_t const maxPastKvLengthComputed = p.getMaxHostPastKeyValueLength(p.seq_offset, p.num_seqs);

        if (p.max_context_q_len_override.has_value())
        {
            int32_t const overrideValue = static_cast<int32_t>(p.max_context_q_len_override.value());
            TORCH_CHECK(overrideValue >= maxContextQLenComputed,
                "p.max_context_q_len_override (%d) must be >= computed max context q length (%d).", overrideValue,
                maxContextQLenComputed);
            TORCH_CHECK(overrideValue >= maxPastKvLengthComputed,
                "p.max_context_q_len_override (%d) must be >= computed max past kv length (%d).", overrideValue,
                maxPastKvLengthComputed);
        }

        mMaxContextQLen = p.max_context_q_len_override.has_value()
            ? static_cast<int32_t>(p.max_context_q_len_override.value())
            : maxContextQLenComputed;
        int32_t const maxPastKvLength = p.max_context_q_len_override.has_value()
            ? static_cast<int32_t>(p.max_context_q_len_override.value())
            : maxPastKvLengthComputed;

        int const maxAttentionWindowSize = op.mConfig.beam_width == 1
            ? op.mConfig.attention_window_size
            : p.getCacheIndirectionWindowSize(op.mConfig.attention_window_size);
        int const cyclicAttentionWindowSize = op.mConfig.attention_window_size;
        bool const canUseOneMoreBlock = op.mConfig.beam_width > 1;

        bool const useKvCache = op.useKVCache() && p.hasKvCache();
        int const maxBlocksPerSequence = useKvCache ? p.getMaxBlocksPerSequence() : 0;
        int32_t const poolIndex = useKvCache ? p.getKvCachePoolIndex(op.mConfig.layer_idx) : 0;
        int32_t const layerIdxInCachePool = useKvCache ? p.getLayerIdxInCachePool(op.mConfig.layer_idx) : 0;
        KVBlockArray::DataType* blockOffsets = useKvCache ? p.getKvCacheBlockOffsets(poolIndex, p.seq_offset) : nullptr;
        KvCachePoolPointers poolPointers;

        int cacheElemBits = op.getKvCacheElemSizeInBits<T>();
        auto const blockSize = op.mConfig.tokens_per_block * op.mNumKVHeads * op.mHeadSize;
        auto const bytesPerBlock = blockSize * cacheElemBits / 8 /*bits*/;
        int32_t const kvFactor = op.isMLAEnabled() ? 1 : 2;
        auto const intraPoolOffset = layerIdxInCachePool * kvFactor * bytesPerBlock;

        if (useKvCache)
        {
            poolPointers = buildKvCachePoolPointers(p.getHostKvCachePoolPointers(), poolIndex, intraPoolOffset,
                blockSize, layerIdxInCachePool, kvFactor, op.mConfig.quant_mode.hasFp4KvCache());
        }

        float const* kvScaleOrigQuantPtr = nullptr;
        float const* kvScaleQuantOrigPtr = nullptr;
        if (op.mConfig.quant_mode.hasKvCacheQuant() && p.kv_scale_orig_quant.has_value()
            && p.kv_scale_quant_orig.has_value())
        {
            kvScaleOrigQuantPtr = p.getKvScaleOrigQuant();
            kvScaleQuantOrigPtr = p.getKvScaleQuantOrig();
            if (op.mConfig.quant_mode.hasFp4KvCache())
            {
                TORCH_CHECK(p.kv_scale_orig_quant.value().size(0) == 3);
                TORCH_CHECK(p.kv_scale_quant_orig.value().size(0) == 3);
            }
        }
        // For FP8 p.output, p.out_scale represents the p.output scale.
        float const* outScalePtr
            = (op.mFP8ContextFMHA && !op.mFuseFp4Quant && p.out_scale.has_value()) ? p.getOutScale() : nullptr;
        // For NVFP4 p.output, p.out_scale holds the global scale for scaling factors.
        float const* outSfScalePtr = op.mFuseFp4Quant && p.out_scale.has_value() ? p.getOutScale() : nullptr;

        float const* attentionSinksPtr = nullptr;
        if (p.attention_sinks.has_value())
        {
            TORCH_CHECK(
                p.attention_sinks.value().dtype() == torch::kFloat32, "Expected p.attention_sinks to have float dtype");
            attentionSinksPtr = p.getAttentionSinks();
        }
        T const* relativeAttentionBiasPtr = nullptr;
        int relativeAttentionBiasStride = 0;
        if (p.relative_attention_bias.has_value())
        {
            auto const& relativeAttentionBiasTensor = p.relative_attention_bias.value();
            TORCH_CHECK(relativeAttentionBiasTensor.dim() == 2 || relativeAttentionBiasTensor.dim() == 3,
                "p.relative_attention_bias must be [num_heads, num_buckets] for implicit mode or "
                "[num_heads, max_seq_len, max_seq_len] for explicit mode");
            TORCH_CHECK(relativeAttentionBiasTensor.is_contiguous(), "p.relative_attention_bias must be contiguous");
            TORCH_CHECK(relativeAttentionBiasTensor.scalar_type() == p.qkv_or_q.scalar_type(),
                "p.relative_attention_bias dtype must match attention input dtype");
            relativeAttentionBiasPtr = p.getRelativeAttentionBias<T>();
            relativeAttentionBiasStride = static_cast<int>(relativeAttentionBiasTensor.size(1));
        }

        op.mRuntimeSparseAttentionParams.sparse_kv_indices = p.getSparseKvIndices();
        op.mRuntimeSparseAttentionParams.sparse_kv_offsets = p.getSparseKvOffsets();
        op.mRuntimeSparseAttentionParams.sparse_attn_indices = p.getSparseAttnIndices();
        op.mRuntimeSparseAttentionParams.sparse_attn_offsets = p.getSparseAttnOffsets();
        op.mRuntimeSparseAttentionParams.sparse_attn_indices_block_size = p.sparse_attn_indices_block_size;
        op.mRuntimeSparseAttentionParams.sparse_attn_indices_stride = p.getSparseAttnIndicesStride();
        op.mRuntimeSparseAttentionParams.num_sparse_topk = p.num_sparse_topk;
        op.mRuntimeSparseAttentionParams.sparse_attn_kv_lens = p.getSparseAttnKvLens();
        op.mRuntimeSparseAttentionParams.sparse_kv_cache_pool = nullptr;
        op.mRuntimeSparseAttentionParams.sliding_window_kv_cache_pool = nullptr;
        if (op.mConfig.use_sparse_attention && useKvCache)
        {
            auto* kvCachePool = p.getSparseKvCachePool(poolIndex);
            if (kvCachePool != nullptr)
            {
                if (p.sparse_attn_kv_lens.has_value())
                {
                    // Deepseek V4 dynamic sparse MLA always uses the SWA pool for now.
                    op.mRuntimeSparseAttentionParams.sliding_window_kv_cache_pool = kvCachePool;
                    if (p.aux_kv_cache_pool_ptr.has_value())
                    {
                        op.mRuntimeSparseAttentionParams.sparse_kv_cache_pool
                            = reinterpret_cast<char*>(p.aux_kv_cache_pool_ptr.value());
                    }
                }
                else
                {
                    op.mRuntimeSparseAttentionParams.sparse_kv_cache_pool = kvCachePool;
                }
            }
        }

        mCommon.qkv_or_q = mAttentionInput;
        mCommon.attention_sinks = attentionSinksPtr;
        mCommon.rotary_inv_freq = rotaryInvFreqPtr;
        mCommon.rotary_cos_sin = mRotaryCosSinPtr;
        mCommon.relative_attention_bias = relativeAttentionBiasPtr;
        mCommon.relative_attention_bias_stride = relativeAttentionBiasStride;
        mCommon.max_past_kv_length = maxPastKvLength;
        mCommon.max_attention_window_size = maxAttentionWindowSize;
        mCommon.cyclic_attention_window_size = cyclicAttentionWindowSize;
        mCommon.max_cyclic_attention_window_size = cyclicAttentionWindowSize;
        mCommon.can_use_one_more_block = canUseOneMoreBlock;
        mCommon.kv_scale_orig_quant = kvScaleOrigQuantPtr;
        mCommon.kv_scale_quant_orig = kvScaleQuantOrigPtr;
        mCommon.out_scale = outScalePtr;
        mCommon.out_sf_scale = outSfScalePtr;
        mCommon.output = mContextBuf;
        mCommon.output_sf = contextBufSf;
        mCommon.kv_cache_block_offsets = blockOffsets;
        mCommon.host_primary_pool_pointer = poolPointers.primaryPoolPtr;
        mCommon.host_secondary_pool_pointer = poolPointers.secondaryPoolPtr;
        mCommon.host_primary_block_scale_pool_pointer = poolPointers.primaryBlockScalePoolPtr;
        mCommon.host_secondary_block_scale_pool_pointer = poolPointers.secondaryBlockScalePoolPtr;
        mCommon.num_tokens = p.num_tokens;
        mCommon.total_kv_len = p.total_kv_len;
        mCommon.max_blocks_per_sequence = maxBlocksPerSequence;
        mCommon.sequence_length = mSequenceLengthsPtr;
        mCommon.context_lengths = contextLengthsPtr;
        mCommon.host_context_lengths = p.getHostContextLengths();
        mCommon.workspace = p.getWorkspace();
        mCommon.trtllm_gen_jit_warmup = p.trtllm_gen_jit_warmup;
        if (p.is_cross)
        {
            // For cross attention, the KV (encoder) sequence lengths are passed in via
            // `p.sequence_length`, so reuse it directly instead of a redundant
            // `encoder_input_lengths` tensor.
            mCommon.encoder_input_lengths = mSequenceLengthsPtr;
        }
        if (p.softmax_stats_tensor.has_value())
        {
            TLLM_CHECK_WITH_INFO(p.softmax_stats_tensor.value().scalar_type() == at::ScalarType::Float,
                "p.softmax_stats_tensor must have float type");
            TLLM_CHECK_WITH_INFO(p.softmax_stats_tensor.value().size(0) >= p.num_tokens,
                "p.softmax_stats_tensor must have first dimension >= p.num_tokens");
            TLLM_CHECK_WITH_INFO(p.softmax_stats_tensor.value().size(1) >= op.mConfig.num_heads,
                "p.softmax_stats_tensor must have second dimension >= num_heads");
            TLLM_CHECK_WITH_INFO(
                p.softmax_stats_tensor.value().size(2) == 2, "p.softmax_stats_tensor must have third dimension == 2");
            mCommon.softmax_stats_tensor = p.getSoftmaxStatsTensor();
        }
    }

    cudaStream_t getCUDAStream() const
    {
        return mStream;
    }

    void* getContextBuf() const
    {
        return mContextBuf;
    }

    MlaParams<T>& getMlaParams()
    {
        return mMla;
    }

    void buildContextMlaParams()
    {
        auto& op = mOp;
        auto const& p = mParams;
        if (op.mConfig.use_sparse_attention)
        {
            mMla.latent_cache = p.getLatentCache<T>();
            TORCH_CHECK(p.q_pe.has_value());
            TORCH_CHECK(p.q_pe->dim() == 3);
            TORCH_CHECK(p.q_pe->strides()[2] == 1);

            mMla.q_pe = p.getQPe<T>();
            mMla.q_pe_ld = p.q_pe->strides()[1];
            mMla.q_pe_stride = p.q_pe->strides()[0];

            // Fused FP8-Q path: forward caller's p.quant_q_buffer / scale so
            // applyMLARopeAndAssignQKVKernelOptContext<kOutputFp8Q=true>
            // appends rope FP8 in place and the standalone quantize is
            // skipped. Without this wiring the sparse-MLA context branch
            // runs the legacy quantize over the bf16 placeholder q.
            mMla.bmm1_scale = p.getMlaBmm1Scale();
            mMla.bmm2_scale = p.getMlaBmm2Scale();
            mMla.quant_q_buf = p.getQuantQBuffer();
            mMla.quant_scale_qkv = p.getQuantScaleQkv();
            mMla.fuse_q_fp8_in_rope = (p.quant_q_buffer.has_value() && p.quant_scale_qkv.has_value());
        }
        else
        {
            mMla.latent_cache = p.getLatentCache<T>();
            TORCH_CHECK(p.k.has_value());
            TORCH_CHECK(p.v.has_value());
            TORCH_CHECK(p.k->dim() == 2);
            TORCH_CHECK(p.v->dim() == 2);
            TORCH_CHECK(p.k->strides()[1] == 1);
            TORCH_CHECK(p.v->strides()[1] == 1);

            mKPtr = p.getK<T>(p.token_offset);
            mVPtr = p.getV<T>(p.token_offset);
            mMla.k_buf = mKPtr;
            mMla.v_buf = mVPtr;

            mMla.helix_position_offsets = p.getHelixPositionOffsets();
            mMla.helix_is_inactive_rank = p.getHelixIsInactiveRank();
        }
        finalizeMlaParams();
    }

    void buildGenerationMlaParams()
    {
        auto const& p = mParams;
        TORCH_CHECK(p.latent_cache.has_value());
        mMla.latent_cache = p.getLatentCache<T>();
        TORCH_CHECK(p.q_pe.has_value());
        TORCH_CHECK(p.q_pe->dim() == 3);
        TORCH_CHECK(p.q_pe->strides()[2] == 1);

        mMla.q_pe = p.getQPe<T>();
        mMla.q_pe_ld = p.q_pe->strides()[1];
        mMla.q_pe_stride = p.q_pe->strides()[0];

        mMla.seqQOffset = const_cast<int*>(p.getCuQSeqlens());
        mMla.cu_kv_seqlens = const_cast<int*>(p.getCuKvSeqlens());
        mMla.fmha_tile_counter = reinterpret_cast<uint32_t*>(p.getFmhaSchedulerCounter());
        mMla.bmm1_scale = p.getMlaBmm1Scale();
        mMla.bmm2_scale = p.getMlaBmm2Scale();
        mMla.quant_q_buf = p.getQuantQBuffer();
        mMla.quant_scale_qkv = p.getQuantScaleQkv();
        mMla.fuse_q_fp8_in_rope = (p.quant_q_buffer.has_value() && p.quant_scale_qkv.has_value());
        finalizeMlaParams();
    }

    void applySageKv()
    {
        auto const& p = mParams;
        TORCH_CHECK(p.k.has_value() && p.v.has_value(), "SageAttention demands separate K and V buffers");
        mKPtr = p.getK<T>(p.token_offset);
        mVPtr = p.getV<T>(p.token_offset);
    }

    AttentionOp::EnqueueContextParams<T> buildContextParams()
    {
        auto& op = mOp;
        auto const& p = mParams;
        mCommon.input_seq_length = mMaxContextQLen;
        AttentionOp::EnqueueContextParams<T> enqueueParams{mCommon};
        enqueueParams.num_seqs = p.num_seqs;
        enqueueParams.k = mKPtr;
        enqueueParams.v = mVPtr;
        if (p.cu_q_seqlens.has_value())
        {
            TORCH_CHECK(p.cu_q_seqlens->dim() == 1, "p.cu_q_seqlens must be a 1-D tensor.");
            TORCH_CHECK(p.cu_q_seqlens->is_cuda(), "p.cu_q_seqlens must be a CUDA tensor.");
            TORCH_CHECK(p.cu_q_seqlens->scalar_type() == at::ScalarType::Int, "p.cu_q_seqlens must be int32.");
            TORCH_CHECK(p.cu_q_seqlens->size(0) >= p.num_seqs + 1,
                "p.cu_q_seqlens must have at least p.num_seqs + 1 elements.");
            enqueueParams.cu_q_seqlens = p.getCuQSeqlens();
        }
        if (p.cu_kv_seqlens.has_value())
        {
            TORCH_CHECK(p.cu_kv_seqlens->dim() == 1, "p.cu_kv_seqlens must be a 1-D tensor.");
            TORCH_CHECK(p.cu_kv_seqlens->is_cuda(), "p.cu_kv_seqlens must be a CUDA tensor.");
            TORCH_CHECK(p.cu_kv_seqlens->scalar_type() == at::ScalarType::Int, "p.cu_kv_seqlens must be int32.");
            TORCH_CHECK(p.cu_kv_seqlens->size(0) >= p.num_seqs + 1,
                "p.cu_kv_seqlens must have at least p.num_seqs + 1 elements.");
            enqueueParams.cu_kv_seqlens = p.getCuKvSeqlens();
        }
        if (mVPtr != nullptr && p.v.has_value())
        {
            enqueueParams.v_stride_in_bytes = p.v->strides()[0] * p.v->element_size();
        }
        if (p.is_cross && p.cross_kv.has_value())
        {
            enqueueParams.cross_kv = p.getCrossKv<T>();
            enqueueParams.num_encoder_tokens = p.getCrossKvNumTokens();
            enqueueParams.cross_kv_length = p.getMaxHostPastKeyValueLength(p.seq_offset, p.num_seqs);
        }

        if (op.isMLAEnabled())
        {
            mMla.cache_seq_lens = mSequenceLengthsPtr;
            mMla.max_input_seq_len = mMaxContextQLen;
            enqueueParams.mla_param = &mMla;
        }
        if (op.isMRoPE() && p.mrope_rotary_cos_sin.has_value())
        {
            enqueueParams.mrope_rotary_cos_sin = p.getMropeRotaryCosSin();
        }
        if (op.useTllmGenSparseAttention())
        {
            enqueueParams.semaphores = p.getFmhaSchedulerCounter();
        }
        extractHelixParams(p, enqueueParams);
        return enqueueParams;
    }

    AttentionOp::EnqueueGenerationParams<T> buildGenerationParams()
    {
        auto& op = mOp;
        auto const& p = mParams;
        int32_t const batchBeam = p.num_seqs;
        TLLM_CHECK(batchBeam % op.mConfig.beam_width == 0);
        int32_t const numRequests = batchBeam / op.mConfig.beam_width;

        TLLM_CHECK_WITH_INFO(p.num_tokens % p.num_seqs == 0,
            "seq_len should be same for all generation requests, p.num_tokens=%d, p.num_seqs=%d", p.num_tokens,
            p.num_seqs);
        int32_t const inputSeqLength = p.num_tokens / p.num_seqs;

        mCommon.input_seq_length = inputSeqLength;
        AttentionOp::EnqueueGenerationParams<T> enqueueParams{mCommon};
        enqueueParams.layer_idx = op.mConfig.layer_idx;
        enqueueParams.beam_width = op.mConfig.beam_width;
        enqueueParams.num_requests = numRequests;
        enqueueParams.cache_indirection = op.mConfig.beam_width == 1 ? nullptr : p.getCacheIndirection();
        enqueueParams.semaphores = p.getFmhaSchedulerCounter();
        enqueueParams.host_past_key_value_lengths = p.getHostPastKeyValueLengths();
        enqueueParams.token_offset = p.token_offset;

        if (op.isMRoPE() && p.mrope_position_deltas.has_value())
        {
            enqueueParams.mrope_position_deltas = p.getMropePositionDeltas();
        }
        if (op.mConfig.is_spec_decoding_enabled && op.mConfig.use_spec_decoding)
        {
            bool useTllmGen = tensorrt_llm::common::isSM100Family();
            TORCH_CHECK(p.spec_decoding_generation_lengths.has_value(),
                "Expecting p.spec_decoding_generation_lengths in spec-dec mode.");
            TORCH_CHECK(p.spec_decoding_position_offsets_for_cpp.has_value(),
                "Expecting p.spec_decoding_position_offsets_for_cpp in spec-dec mode.");
            TORCH_CHECK(
                p.spec_decoding_packed_mask.has_value(), "Expecting p.spec_decoding_packed_mask in spec-dec mode.");
            if (useTllmGen)
            {
                TORCH_CHECK(p.spec_decoding_bl_tree_mask_offset.has_value(),
                    "Expecting p.spec_decoding_bl_tree_mask_offset in trtllm-gen spec-dec mode.");
                TORCH_CHECK(p.spec_decoding_bl_tree_mask.has_value(),
                    "Expecting p.spec_decoding_bl_tree_mask in trtllm-gen spec-dec mode.");
                TORCH_CHECK(p.spec_bl_tree_first_sparse_mask_offset_kv.has_value(),
                    "Expecting p.spec_bl_tree_first_sparse_mask_offset_kv in trtllm-gen spec-dec mode.");
                enqueueParams.spec_decoding_bl_tree_mask_offset = p.getSpecDecodingBlTreeMaskOffset();
                enqueueParams.spec_decoding_bl_tree_mask = p.getSpecDecodingBlTreeMask();
                enqueueParams.spec_bl_tree_first_sparse_mask_offset_kv = p.getSpecBlTreeFirstSparseMaskOffsetKv();
            }
            enqueueParams.spec_decoding_generation_lengths = p.getSpecDecodingGenerationLengths();
            enqueueParams.spec_decoding_position_offsets_for_cpp = p.getSpecDecodingPositionOffsetsForCpp();
            enqueueParams.spec_decoding_packed_mask = p.getSpecDecodingPackedMask();
            enqueueParams.spec_decoding_is_generation_length_variable = true;
            TLLM_CHECK(p.spec_decoding_position_offsets_for_cpp->dim() == 2); // [batch_size, max_draft_len + 1]
            if (useTllmGen)
            {
                // Blackwell uses the padded packed-mask row dim as the mask stride.
                TLLM_CHECK(p.spec_decoding_packed_mask->dim() == 3);
                enqueueParams.spec_decoding_max_generation_length = p.spec_decoding_packed_mask->sizes()[1];
            }
            else
            {
                enqueueParams.spec_decoding_max_generation_length
                    = p.spec_decoding_position_offsets_for_cpp->sizes()[1];
            }
        }
        extractHelixParams(p, enqueueParams);
        return enqueueParams;
    }

    void prepareFlashMlaGeneration()
    {
        auto& op = mOp;
        auto const& p = mParams;
        if (op.mUseGenFlashMLA == true)
        {
            TORCH_CHECK(p.block_ids_per_seq.has_value());
            mMla.block_ids_per_seq = p.getBlockIdsPerSeq();
            if (p.flash_mla_tile_scheduler_metadata.has_value())
            {
                TORCH_CHECK(p.flash_mla_num_splits.has_value(),
                    "p.flash_mla_num_splits must be provided when p.flash_mla_tile_scheduler_metadata is set.");
                mMla.flash_mla_tile_scheduler_metadata = p.getFlashMlaTileSchedulerMetadata();
                mMla.flash_mla_num_splits = p.getFlashMlaNumSplits();
            }
        }
        mMla.cache_seq_lens = mSequenceLengthsPtr;
    }

private:
    void finalizeMlaParams()
    {
        auto& op = mOp;
        auto const& p = mParams;
        mMla.q_buf = mAttentionInput;
        mMla.context_buf = static_cast<T*>(mContextBuf);

        mMla.cos_sin_cache = mRotaryCosSinPtr;
        if (p.enable_dsv4_epilogue_fusion)
        {
            TORCH_CHECK(
                p.dsv4_inv_rope_cos_sin_cache.has_value(), "DSv4 fused epilogue requires inverse-RoPE cos/sin cache.");
            auto const& cosSinCache = p.dsv4_inv_rope_cos_sin_cache.value();
            auto const& outputSfTensor = p.output_sf.value();
            TORCH_CHECK(
                cosSinCache.scalar_type() == torch::kFloat32, "DSv4 fused epilogue cos/sin cache must be float32.");
            TORCH_CHECK(
                p.output.scalar_type() == torch::kFloat8_e4m3fn, "DSv4 fused epilogue p.output must be float8_e4m3fn.");
            TORCH_CHECK(p.output.dim() == 3 && p.output.is_contiguous(),
                "DSv4 fused epilogue p.output must be contiguous [groups, tokens, K].");
            TORCH_CHECK(
                outputSfTensor.scalar_type() == torch::kFloat32, "DSv4 fused epilogue p.output_sf must be float32.");
            TORCH_CHECK(outputSfTensor.dim() == 3 && outputSfTensor.is_contiguous(),
                "DSv4 fused epilogue p.output_sf must be contiguous [groups, K/128, padded_tokens].");
            TORCH_CHECK(p.output.size(1) >= p.num_tokens, "DSv4 fused epilogue p.output token dimension is too small.");
            TORCH_CHECK(op.mConfig.mla_params.v_head_dim > 0 && op.mConfig.mla_params.v_head_dim % 128 == 0,
                "DSv4 fused epilogue requires v_head_dim to be a positive multiple of 128.");
            TORCH_CHECK(outputSfTensor.size(2) >= p.num_tokens,
                "DSv4 fused epilogue p.output_sf token dimension is too small.");

            mMla.dsv4_epilogue_fusion.enabled = true;
            mMla.dsv4_epilogue_fusion.cos_sin_cache = p.getDsv4InvRopeCosSinCache();
            mMla.dsv4_epilogue_fusion.scale_buf_m = static_cast<int32_t>(outputSfTensor.size(2));
        }
        mMla.batch_size = p.num_seqs;
        mMla.acc_q_len = p.num_tokens;
        mMla.head_num = op.mConfig.num_heads;
        mMla.meta = op.mConfig.mla_params;

        mMla.workspace = mCommon.workspace;
    }

    AttentionOp& mOp;
    FmhaParams const& mParams;
    AttentionOp::EnqueueParams<T> mCommon;
    MlaParams<T> mMla;
    T* mKPtr = nullptr;
    T* mVPtr = nullptr;
    T* mAttentionInput = nullptr;
    void* mContextBuf = nullptr;
    float2 const* mRotaryCosSinPtr = nullptr;
    cudaStream_t mStream{};
    int const* mSequenceLengthsPtr = nullptr;
    int32_t mMaxContextQLen = 0;
};

template <typename T>
void runContextImpl(AttentionOp& op, FmhaParams const& p)
{
    EnqueueParamsBuilder<T> builder{op, p};
    if (op.isMLAEnabled())
    {
        builder.buildContextMlaParams();
    }
    else if (op.mConfig.sage_attn_num_elts_per_blk_q > 0 || op.mConfig.sage_attn_num_elts_per_blk_k > 0
        || op.mConfig.sage_attn_num_elts_per_blk_v > 0)
    {
        builder.applySageKv();
    }
    auto enqueueParams = builder.buildContextParams();
    op.enqueueContext<T, KVBlockArray>(enqueueParams, builder.getCUDAStream());
    sync_check_cuda_error(builder.getCUDAStream());
}

template <typename T>
void runGenerationImpl(AttentionOp& op, FmhaParams const& p)
{
    EnqueueParamsBuilder<T> builder{op, p};
    auto enqueueParams = builder.buildGenerationParams();
    op.enqueueGeneration<T, KVBlockArray>(enqueueParams, builder.getCUDAStream());
    {
        std::string const afterGenStr = "gen attention at layer " + std::to_string(op.mConfig.layer_idx);
        {
            TLLM_CHECK_DEBUG_WITH_INFO(
                tensorrt_llm::runtime::utils::tensorHasInvalid(p.num_tokens, p.output.size(1), op.mConfig.type,
                    builder.getContextBuf(), builder.getCUDAStream(), afterGenStr)
                    == false,
                "Found invalid number (NaN or Inf) in " + afterGenStr);
        }
    }
    sync_check_cuda_error(builder.getCUDAStream());
}

template <typename T>
void runMlaGenerationImpl(AttentionOp& op, FmhaParams const& p)
{
    EnqueueParamsBuilder<T> builder{op, p};
    builder.buildGenerationMlaParams();
    auto enqueueParams = builder.buildGenerationParams();
    builder.prepareFlashMlaGeneration();
    op.mlaGeneration<T>(builder.getMlaParams(), enqueueParams, builder.getCUDAStream());
    {
        std::string const afterGenStr = "gen attention at layer " + std::to_string(op.mConfig.layer_idx);
        {
            TLLM_CHECK_DEBUG_WITH_INFO(
                tensorrt_llm::runtime::utils::tensorHasInvalid(p.num_tokens, p.output.size(1), op.mConfig.type,
                    builder.getContextBuf(), builder.getCUDAStream(), afterGenStr)
                    == false,
                "Found invalid number (NaN or Inf) in " + afterGenStr);
        }
    }
    sync_check_cuda_error(builder.getCUDAStream());
}

} // namespace trtllm::attention

using torch_ext::trtllm::attention::attnPrepare;
using torch_ext::trtllm::attention::runContextImpl;
using torch_ext::trtllm::attention::runGenerationImpl;
using torch_ext::trtllm::attention::runMlaGenerationImpl;

static bool hasNonEmptyTensor(std::optional<torch::Tensor> const& tensor)
{
    return tensor.has_value() && tensor.value().numel() > 0;
}

static AttentionStaticConfig buildAttentionConfig(FmhaParams const& p)
{
    AttentionStaticConfig cfg{};
    cfg.type = tensorrt_llm::runtime::TorchUtils::dataType(p.qkv_or_q.scalar_type());
    cfg.is_fp8_out = p.output.scalar_type() == torch::kFloat8_e4m3fn;
    cfg.is_fp4_out = p.output.scalar_type() == torch::kUInt8; // Torch has no native nvfp4 type.
    cfg.layer_idx = p.layer_idx;
    cfg.num_heads = p.num_heads;
    cfg.num_kv_heads = p.num_kv_heads;
    cfg.head_size = p.head_size;
    cfg.mask_type = p.mask_type;
    cfg.quant_mode = p.quant_mode;
    cfg.use_kv_cache = p.hasKvCache();
    cfg.tokens_per_block = p.tokens_per_block;
    cfg.fuses_dsv4_inv_rope_fp8_quant = p.enable_dsv4_epilogue_fusion;
    cfg.max_context_length = p.max_context_length;
    cfg.max_seq_len = p.max_seq_len;
    cfg.max_num_requests = p.max_num_requests;
    cfg.beam_width = p.beam_width;
    cfg.attention_window_size = p.attention_window_size;
    cfg.q_scaling = p.q_scaling;
    cfg.position_embedding_type = p.position_embedding_type;
    cfg.max_distance = p.max_distance;
    cfg.rotary_embedding_dim = p.rotary_embedding_dim;
    cfg.rotary_embedding_base = p.rotary_embedding_base;
    cfg.rotary_embedding_scale_type = p.rotary_embedding_scale_type;
    cfg.rotary_embedding_scale = p.rotary_embedding_scale;
    cfg.rotary_embedding_short_mscale = p.rotary_embedding_short_mscale;
    cfg.rotary_embedding_long_mscale = p.rotary_embedding_long_mscale;
    cfg.rotary_embedding_max_positions = p.rotary_embedding_max_positions;
    cfg.rotary_embedding_original_max_positions = p.rotary_embedding_original_max_positions;
    cfg.sage_attn_num_elts_per_blk_q = p.sage_attn_num_elts_per_blk_q;
    cfg.sage_attn_num_elts_per_blk_k = p.sage_attn_num_elts_per_blk_k;
    cfg.sage_attn_num_elts_per_blk_v = p.sage_attn_num_elts_per_blk_v;
    cfg.sage_attn_qk_int8 = p.sage_attn_qk_int8;
    cfg.paged_context_fmha = p.paged_context_fmha;
    cfg.cross_attention = p.is_cross;
    cfg.attention_chunk_size = p.attention_chunk_size;
    cfg.skip_softmax_threshold_scale_factor_prefill = p.skip_softmax_threshold_scale_factor_prefill;
    cfg.skip_softmax_threshold_scale_factor_decode = p.skip_softmax_threshold_scale_factor_decode;
    cfg.is_spec_decoding_enabled = p.is_spec_decoding_enabled;
    cfg.use_spec_decoding = p.use_spec_decoding;
    cfg.is_spec_dec_tree = p.is_spec_dec_tree;
    if (p.spec_decoding_target_max_draft_tokens.has_value() && cfg.spec_decoding_target_max_gen_len == 0)
    {
        cfg.spec_decoding_target_max_gen_len
            = static_cast<int32_t>(p.spec_decoding_target_max_draft_tokens.value()) + 1;
    }
    cfg.force_prepare_spec_dec_tree_mask = p.force_prepare_spec_dec_tree_mask;

    bool const has_sparse_attn_indices = hasNonEmptyTensor(p.sparse_attn_indices);
    cfg.use_sparse_attention = hasNonEmptyTensor(p.sparse_kv_indices) || has_sparse_attn_indices;
    if (has_sparse_attn_indices)
    {
        if (hasNonEmptyTensor(p.sparse_attn_offsets))
        {
            cfg.use_tllm_gen_sparse_attention_paged = true;
        }
        else
        {
            cfg.use_tllm_gen_sparse_attention = true;
        }
    }

    if (p.is_mla_enable)
    {
        if (p.num_sparse_topk > 0 && has_sparse_attn_indices)
        {
            cfg.use_sparse_attention = true;
        }
        TLLM_CHECK(!cfg.is_fp4_out); // MLA does not support NVFP4 output yet.
        cfg.is_mla_enable = true;
        auto const mla_layer_num = p.getMlaLayerNum();
        cfg.mla_params = {static_cast<int>(p.q_lora_rank.value()), static_cast<int>(p.kv_lora_rank.value()),
            static_cast<int>(p.qk_nope_head_dim.value()), static_cast<int>(p.qk_rope_head_dim.value()),
            static_cast<int>(p.v_head_dim.value()), static_cast<int>(p.predicted_tokens_per_seq),
            static_cast<int>(mla_layer_num), static_cast<int>(p.rope_append.value_or(true))};
        cfg.chunk_prefill_buffer_batch_size = p.chunk_prefill_buffer_batch_size;
    }
    return cfg;
}

static void buildAttentionOp(AttentionOp& op, AttentionStaticConfig cfg)
{
    TLLM_LOG_TRACE("Building attention op for layer %lld", cfg.layer_idx);
    op.mConfig = std::move(cfg);

    auto const& opConfig = op.mConfig;
    op.mFMHAForceFP32Acc = opConfig.type == tensorrt_llm::DataType::kBF16;
    op.mPagedKVCache = op.mPagedKVCache && opConfig.use_kv_cache;
    op.mNumKVHeads = opConfig.num_kv_heads;
    op.mHeadSize = opConfig.head_size;
    bool const use_sage_attn = opConfig.sage_attn_num_elts_per_blk_q > 0 || opConfig.sage_attn_num_elts_per_blk_k > 0
        || opConfig.sage_attn_num_elts_per_blk_v > 0;
    op.mFP8ContextFMHA = opConfig.is_fp8_out || opConfig.is_fp4_out
        || (opConfig.quant_mode.hasFp8KvCache() && opConfig.paged_context_fmha) || use_sage_attn;
    op.mFP8AttenOutput = opConfig.is_fp8_out;
    op.mFuseFp4Quant = opConfig.is_fp4_out;
    op.mFP8GenerationMLA = false;
    if (opConfig.is_mla_enable)
    {
        int const sm = tensorrt_llm::common::getSMVersion();
        op.mFP8ContextMLA = (sm == 90 || sm == 100 || sm == 103 || sm == 120) && opConfig.quant_mode.hasFp8KvCache();
        op.mIsGenerationMLA
            = opConfig.head_size == opConfig.mla_params.kv_lora_rank + opConfig.mla_params.qk_rope_head_dim;
        op.mFP8GenerationMLA = opConfig.quant_mode.hasFp8KvCache();
        op.mUseGenFlashMLA = sm == 90 && opConfig.tokens_per_block == 64 && opConfig.head_size == 576;
        op.mNumKVHeads = 1;
        op.mHeadSize = opConfig.mla_params.kv_lora_rank + opConfig.mla_params.qk_rope_head_dim;
    }

    op.initialize();
    DISPATCH_ON_DTYPE(op.mConfig.type, attnPrepare, op);
}

int64_t get_attention_workspace_size(FmhaParams const& p, int64_t num_tokens, int64_t max_attention_window_size,
    int64_t num_gen_tokens, int64_t max_blocks_per_sequence, int64_t ctx_total_kv_len)
{
    AttentionOp op;
    buildAttentionOp(op, buildAttentionConfig(p));
    size_t const contextWorkspaceSize = op.getWorkspaceSizeForContext(op.mConfig.type,
        static_cast<int>(op.mConfig.max_num_requests), static_cast<int>(op.mConfig.max_context_length), 0,
        static_cast<int>(num_tokens), static_cast<int>(ctx_total_kv_len));
    size_t const generationWorkspaceSize = op.getWorkspaceSizeForGeneration(op.mConfig.type,
        static_cast<int>(op.mConfig.max_num_requests), static_cast<int>(max_attention_window_size),
        static_cast<int>(num_gen_tokens), static_cast<int>(max_blocks_per_sequence));
    return static_cast<int64_t>(std::max(contextWorkspaceSize, generationWorkspaceSize));
}

void run_context(FmhaParams const& p)
{
    AttentionOp op;
    buildAttentionOp(op, buildAttentionConfig(p));
    DISPATCH_ON_DTYPE(op.mConfig.type, runContextImpl, op, p);
}

void run_generation(FmhaParams const& p)
{
    AttentionOp op;
    buildAttentionOp(op, buildAttentionConfig(p));
    DISPATCH_ON_DTYPE(op.mConfig.type, runGenerationImpl, op, p);
}

void run_mla_generation(FmhaParams const& p)
{
    AttentionOp op;
    buildAttentionOp(op, buildAttentionConfig(p));
    DISPATCH_ON_DTYPE(op.mConfig.type, runMlaGenerationImpl, op, p);
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

    AttentionOp op;
    op.mConfig.type = tensorrt_llm::DataType::kHALF;
    op.mConfig.num_heads = num_heads;
    op.mConfig.num_kv_heads = num_kv_heads;
    op.mConfig.head_size = head_size;
    op.mNumKVHeads = num_kv_heads;
    op.mHeadSize = head_size;
    op.mConfig.mask_type = static_cast<tensorrt_llm::kernels::AttentionMaskType>(int32_t(mask_type));
    op.mConfig.quant_mode = tensorrt_llm::common::QuantMode(uint32_t(quant_mode));
    op.mFP8ContextFMHA = op.mConfig.quant_mode.hasFp8KvCache() || op.mConfig.quant_mode.hasFp4KvCache();
    op.mConfig.use_kv_cache = true;
    op.mPagedKVCache = true;
    op.mConfig.tokens_per_block = tokens_per_block.value_or(0);
    op.mFuseFp4Quant = true;
    op.mConfig.paged_context_fmha = use_paged_context_fmha;

    op.initialize();

    return op.supportsNvFp4Output();
}

KvCachePoolPointers buildKvCachePoolPointers(at::Tensor const& hostKvCachePoolPointers, int32_t poolIndex,
    int64_t intraPoolOffset, int64_t blockSize, int32_t layerIdxInCachePool, int32_t kvFactor, bool isFp4KvCache)
{
    KvCachePoolPointers pointers;
    if (isFp4KvCache)
    {
        // For NVFP4 KV cache, extra block scales are stored in separate pools.
        // The layout of host_kv_cache_pool_pointers is [num_pools, 2 (primary and secondary), 2 (data and scale)].
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

common::op::KvCacheBuffers<kernels::KVBlockArray> buildPagedKvCacheBuffers(
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

    int cacheElemBits = common::op::AttentionOp::getKvCacheElemSizeInBits(quantMode, elem_size);

    auto const blockSize = tokens_per_block * kv_head_num * size_per_head;
    auto const bytesPerBlock = blockSize * cacheElemBits / CHAR_BIT;
    int32_t const kvFactor = is_mla_enable ? 1 : 2;
    auto const intraPoolOffset = layerIdxInCachePool * kvFactor * bytesPerBlock;
    auto const sizePerToken = static_cast<int32_t>(kv_head_num * size_per_head * cacheElemBits / 8);

    auto poolPointers = buildKvCachePoolPointers(host_kv_cache_pool_pointers.value(), poolIndex, intraPoolOffset,
        blockSize, layerIdxInCachePool, kvFactor, quantMode.hasFp4KvCache());

    int32_t const maxBlocksPerSequence = static_cast<int32_t>(kv_cache_block_offsets->size(-1));
    return common::op::buildKvCacheBuffers<kernels::KVBlockArray>(static_cast<int32_t>(batch_size),
        maxBlocksPerSequence, static_cast<int32_t>(tokens_per_block), sizePerToken,
        static_cast<int32_t>(cyclic_attention_window_size),
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
    int const cacheElemBits = common::op::AttentionOp::getKvCacheElemSizeInBits(quantMode, inputElemSize);

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
    int const num_sm_parts = tensorrt_llm::common::op::AttentionOp::getFlashMlaNumSmPartsStatic(static_cast<int>(s_q),
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
