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
#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/prepareCustomMask.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/debugUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/thop/attentionVariant.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <sstream>
#include <torch/extension.h>
#include <type_traits>
#include <unordered_map>

using namespace tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
using tensorrt_llm::common::op::AttentionContextWorkspaceSizes;
using tensorrt_llm::common::op::AttentionGenerationWorkspaceSizes;
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
    int64_t const rotaryEmbeddingDim, bool const separateQKvInput, bool const fp8ContextFmha,
    bool const skipFmhaWorkspace)
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
    workspaceSizes.cublasWorkspace = skipFmhaWorkspace ? 0 : kTrtllmGenWorkspaceSize;
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
        .trtllmGenWorkspaceSize = skipFmhaWorkspace ? 0 : kTrtllmGenWorkspaceSize,
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
    bool const useSparseAttention, bool const skipFmhaWorkspace)
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
    auto const trtllmGenWorkspaceSize = skipFmhaWorkspace ? 0 : kTrtllmGenWorkspaceSize;
    auto const totalSize = xqaLayout.totalSize
        + tensorrt_llm::common::alignSize(static_cast<size_t>(trtllmGenWorkspaceSize), kWorkspaceAlignment);

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
        .trtllmGenWorkspaceSize = trtllmGenWorkspaceSize,
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
    auto const trtllmGenWorkspace = layout.trtllmGenWorkspaceSize == 0
        ? torch::empty({0}, workspace.options().dtype(at::kByte))
        : *makeWorkspaceView(workspace, layout.trtllmGenWorkspaceOffset, layout.trtllmGenWorkspaceSize, at::kByte);
    return TrtllmGenContextWorkspaceViews{
        .trtllmGenWorkspace = trtllmGenWorkspace,
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
    auto const trtllmGenWorkspace = layout.trtllmGenWorkspaceSize == 0
        ? torch::empty({0}, workspace.options().dtype(at::kByte))
        : *makeWorkspaceView(workspace, layout.trtllmGenWorkspaceOffset, layout.trtllmGenWorkspaceSize, at::kByte);
    return TrtllmGenGenerationWorkspaceViews{
        .trtllmGenWorkspace = trtllmGenWorkspace,
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

size_t contextMlaWorkspaceBytesPerToken(int32_t numAttnHeads, int32_t qkRopeHeadDim, int32_t qkNopeHeadDim,
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

AttentionContextWorkspaceSizes MHA::contextWorkspaceSizes(FmhaParams const& p) const
{
    auto const max_num_seq = static_cast<int32_t>(p.batch_size);
    auto const max_num_tokens = static_cast<int32_t>(p.num_tokens);
    auto const total_kv_len = static_cast<int32_t>(p.total_kv_len);

    int const local_hidden_units_qo = mNumAttnHeads * getHeadSize();
    int const local_hidden_units_kv = mNumAttnKVHeads * getHeadSize();

    auto const size = tensorrt_llm::runtime::BufferDataType(p.getType()).getSize();

    auto const batch_size = static_cast<size_t>(max_num_seq);
    size_t const cu_seqlens_size = sizeof(int) * (batch_size + 1);
    size_t const rotary_inv_freq_size = sizeof(float) * batch_size * rotary_embedding_dim / 2;

    size_t q_buf_2_size = 0;
    if (isSeparateQAndKvInput())
    {
        // Paged context fmha
        q_buf_2_size = (mFP8ContextFMHA ? 1 : size) * max_num_tokens * local_hidden_units_qo;
    }

    bool const useSageAttnSeparateQkv = useSageAttn() && isSeparateQAndKvInput();

    // Packed fp8 qkv buffer size for normal fp8 context FMHA
    size_t fp8_qkv_buffer_size = mFP8ContextFMHA && !isSeparateQAndKvInput()
        ? max_num_tokens * (local_hidden_units_qo + (2ULL * local_hidden_units_kv))
        : 0;
    // Separate fp8 q/k/v buffers for SageAttention.
    size_t fp8_q_buf_size = 0;
    size_t fp8_k_buf_size = 0;
    size_t fp8_v_buf_size = 0;
    if (useSageAttnSeparateQkv)
    {
        fp8_q_buf_size = max_num_tokens * static_cast<size_t>(local_hidden_units_qo);
        fp8_k_buf_size = total_kv_len * static_cast<size_t>(local_hidden_units_kv);
        fp8_v_buf_size = total_kv_len * static_cast<size_t>(local_hidden_units_kv);
    }

    bool const hopperSage = useHopperSageAttn();
    int32_t const q_max_n_blk = tc::getSageScaleHeadStride(
        tc::getSageQPartition(hopperSage), sage_attn_num_elts_per_blk_q, max_num_tokens, batch_size);
    int32_t const k_max_n_blk = tc::getSageScaleHeadStride(
        tc::getSageKPartition(hopperSage), sage_attn_num_elts_per_blk_k, total_kv_len, batch_size);
    int32_t const v_max_n_blk
        = sage_attn_num_elts_per_blk_v > 0 ? tc::divUp(local_hidden_units_kv, sage_attn_num_elts_per_blk_v) : 0;
    size_t const sage_q_sfs_buffer_size = sizeof(float) * mNumAttnHeads * static_cast<size_t>(q_max_n_blk);
    size_t const sage_k_sfs_buffer_size = sizeof(float) * mNumAttnKVHeads * static_cast<size_t>(k_max_n_blk);
    size_t const sage_v_sfs_buffer_size = sizeof(float) * static_cast<size_t>(v_max_n_blk);
    size_t const sage_k_mean_buffer_size = sage_attn_smooth_k ? sizeof(float) * local_hidden_units_kv : 0;
    size_t const sage_k_mean_sum_buffer_size = sage_attn_smooth_k ? sizeof(int64_t) * local_hidden_units_kv : 0;

    // Each token holds (batch_idx, token_idx_in_seq) int2.
    size_t const tokens_info_size = sizeof(int2) * max_num_tokens;
    size_t const fmha_scheduler_counter = sizeof(uint32_t);
    size_t const fmha_bmm1_scale_size = mFP8ContextFMHA ? sizeof(float) * 2 : 0;
    size_t const fmha_bmm2_scale_size = mFP8ContextFMHA ? sizeof(float) : 0;

    size_t const fmha_multi_ctas_kv_scratch_size = useTllmGenSparseAttention()
        ? TrtllmGenFmha<MHA>::getMultiCtasKvScratchSize(getHeadSize(), mMultiProcessorCount)
        : 0;

    AttentionContextWorkspaceSizes workspaceSizes{};
    workspaceSizes.cuQSeqlens = cu_seqlens_size;
    workspaceSizes.cuKvSeqlens = cu_seqlens_size;
    workspaceSizes.cuMaskRows = cu_seqlens_size;
    workspaceSizes.rotaryInvFreq = rotary_inv_freq_size;
    workspaceSizes.qBuf = q_buf_2_size;
    workspaceSizes.fp8QkvBuf = fp8_qkv_buffer_size;
    workspaceSizes.fp8QBuf = fp8_q_buf_size;
    workspaceSizes.fp8KBuf = fp8_k_buf_size;
    workspaceSizes.fp8VBuf = fp8_v_buf_size;
    workspaceSizes.tokensInfo = tokens_info_size;
    workspaceSizes.fmhaTileCounter = fmha_scheduler_counter;
    workspaceSizes.fmhaBmm1Scale = fmha_bmm1_scale_size;
    workspaceSizes.fmhaBmm2Scale = fmha_bmm2_scale_size;
    workspaceSizes.sageQScale = sage_q_sfs_buffer_size;
    workspaceSizes.sageKScale = sage_k_sfs_buffer_size;
    workspaceSizes.sageVScale = sage_v_sfs_buffer_size;
    workspaceSizes.sageKMean = sage_k_mean_buffer_size;
    workspaceSizes.sageKMeanSum = sage_k_mean_sum_buffer_size;
    workspaceSizes.fmhaMultiCtasKvScratch = fmha_multi_ctas_kv_scratch_size;
    return workspaceSizes;
}

tensorrt_llm::common::op::AttentionContextWorkspaceLayout MHA::contextWorkspaceLayout(FmhaParams const& p) const
{
    return AttentionWorkspaceManager::buildContextLayout(contextWorkspaceSizes(p));
}

size_t MHA::getWorkspaceSizeForContext(FmhaParams const& p) const
{
    return p.num_tokens == 0 ? 0 : contextWorkspaceLayout(p).totalSize;
}

template <typename T>
KvCacheBuffers<KVBlockArray> MHA::getKvCacheBuffers(FmhaParams const& p, bool isGen) const
{
    if (!useKVCache(p))
    {
        return {};
    }
    auto const sizePerToken = mNumAttnKVHeads * getHeadSize() * getKvCacheElemSizeInBits<T>() / 8 /*bits*/;
    auto const& pagedKv = feature<torch_ext::PagedKVCacheFeature>();
    auto const& pools = pagedKv.poolPointers();
    auto const& cross = feature<torch_ext::CrossAttentionFeature>();
    int32_t const batchSize = isGen ? static_cast<int32_t>(p.meta.effective_beam_width * p.num_requests)
                                    : static_cast<int32_t>(p.batch_size);
    int32_t const maxAttentionWindowSize
        = !isGen && cross.active() ? cross.crossKvLength() : static_cast<int32_t>(p.max_attention_window_size);
    return buildKvCacheBuffers<KVBlockArray>(batchSize, pagedKv.maxBlocksPerSeq(), tokens_per_block, sizePerToken,
        p.cyclic_attention_window_size, p.cyclic_attention_window_size, 0, p.meta.effective_beam_width > 1,
        pools.primaryPoolPtr, pools.secondaryPoolPtr, pools.primaryBlockScalePoolPtr, pools.secondaryBlockScalePoolPtr,
        pagedKv.blockOffsets(), quant_mode.hasFp4KvCache(), maxAttentionWindowSize, nullptr);
}

template <typename T>
void MHA::buildContextDecoderInfo(FmhaParams const& p,
    tensorrt_llm::common::op::AttentionContextWorkspaceViews<T> const& workspaceViews, cudaStream_t stream) const
{
    int const local_hidden_units_qo = num_heads * getHeadSize();
    float const q_scaling = static_cast<float>(this->q_scaling);
    auto const& cross = feature<torch_ext::CrossAttentionFeature>();

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
    decoder_params.encoderPaddingOffsets = cross.active() ? workspaceViews.encoderPaddingOffset : nullptr;
    decoder_params.attentionMask = workspaceViews.attentionMask;
    decoder_params.seqQLengths = p.getContextLengths();
    decoder_params.seqKVLengths = p.getSequenceLengths();
    decoder_params.batchSize = p.batch_size;
    decoder_params.maxQSeqLength = p.input_seq_length;
    decoder_params.maxEncoderQSeqLength = 0;
    decoder_params.attentionWindowSize = p.cyclic_attention_window_size;
    decoder_params.numTokens = p.num_tokens;
    decoder_params.removePadding = remove_padding;
    decoder_params.attentionMaskType = mask_type;
    decoder_params.fmhaTileCounter = workspaceViews.fmhaTileCounter;
    decoder_params.quantScaleO = getOutQuantScale(p);
    decoder_params.dequantScaleQkv = p.getKvScaleQuantOrig();
    decoder_params.separateQkvScales = quant_mode.hasFp4KvCache();
    decoder_params.fmhaHostBmm1Scale = 1.0f / (sqrtf(getHeadSize() * 1.0f) * q_scaling);
    decoder_params.fmhaBmm1Scale = workspaceViews.fmhaBmm1Scale;
    decoder_params.fmhaBmm2Scale = workspaceViews.fmhaBmm2Scale;
    decoder_params.rotaryEmbeddingInvFreq = workspaceViews.rotaryInvFreq;
    applyFeatures(decoder_params);

    invokeBuildDecoderInfo(decoder_params, stream);
    sync_check_cuda_error(stream);

    // FIXME: a temporary solution to make sure the padding part is 0.
    if (!remove_padding)
    {
        cudaMemsetAsync(p.getOutput(), 0, p.num_tokens * local_hidden_units_qo * sizeof(T), stream);
        sync_check_cuda_error(stream);
    }
}

template <typename T>
void MHA::enqueueFusedContext(
    FmhaParams const& p, cudaStream_t stream, std::function<void(MHARunnerParams const&)> const& launch) const
{
    int const headSize = getHeadSize();
    int const local_hidden_units_qo = num_heads * headSize;
    int const local_hidden_units_kv = mNumAttnKVHeads * headSize;
    using KVCacheBuffer = KVBlockArray;

    auto const kvBuffers = getKvCacheBuffers<T>(p, /*isGen=*/false);
    auto const& kv_cache_buffer = kvBuffers.kvCacheBuffer;
    auto const& kv_scale_cache_buffer = kvBuffers.kvScaleCacheBuffer;

    auto const workspaceLayout = contextWorkspaceLayout(p);
    TLLM_CHECK_WITH_INFO(!sageAttnRequested() || useSageAttn(),
        "SageAttention requires the FP8 context FMHA path without KV caching and does not apply to MLA.");
    bool const separateQAndKvInput = isSeparateQAndKvInput();
    bool const useSageAttnSeparateQkv = useSageAttn() && separateQAndKvInput;
    bool const hopperSage = useHopperSageAttn();
    int32_t const q_max_n_blk = tc::getSageScaleHeadStride(
        tc::getSageQPartition(hopperSage), sage_attn_num_elts_per_blk_q, p.num_tokens, p.batch_size);
    int32_t const k_max_n_blk = tc::getSageScaleHeadStride(
        tc::getSageKPartition(hopperSage), sage_attn_num_elts_per_blk_k, p.total_kv_len, p.batch_size);
    int32_t const v_max_n_blk
        = sage_attn_num_elts_per_blk_v > 0 ? tc::divUp(local_hidden_units_kv, sage_attn_num_elts_per_blk_v) : 0;
    auto const workspaceViews = AttentionWorkspaceManager::materializeContext<T>(p.getWorkspace(), workspaceLayout);
    buildContextDecoderInfo<T>(p, workspaceViews, stream);

    int32_t const* precomputedCuQSeqlens = p.getCuQSeqlens();
    int32_t const* precomputedCuKvSeqlens = p.getCuKvSeqlens() != nullptr ? p.getCuKvSeqlens()
        : p.getCuQSeqlens() != nullptr                                    ? p.getCuQSeqlens()
                                                                          : nullptr;
    int32_t const* contextCuQSeqlens
        = precomputedCuQSeqlens != nullptr ? precomputedCuQSeqlens : workspaceViews.cuQSeqlens;
    int32_t const* contextCuKvSeqlens
        = precomputedCuKvSeqlens != nullptr ? precomputedCuKvSeqlens : workspaceViews.cuKvSeqlens;

    KvCacheDataType cache_type = cacheTypeFromQuantMode(quant_mode);

    T* attention_input = p.getQueryInput<T>();
    bool const enablePagedKVContextFMHA = mPagedKVCache && paged_context_fmha;
    TLLM_CHECK_WITH_INFO(!(quant_mode.hasInt8KvCache() && enablePagedKVContextFMHA),
        "Paged Context FMHA doesn't work with int8 kv cache currently.");

    // Prepare QKV preprocessing parameters.
    QKVPreprocessingParams<T, KVCacheBuffer> preprocessingParams;

    // Buffers.
    preprocessingParams.qkv_input = const_cast<T*>(attention_input);
    preprocessingParams.quantized_qkv_output = workspaceViews.fp8QkvBuf;
    preprocessingParams.q_output = workspaceViews.qBuf;
    preprocessingParams.kv_cache_buffer = kv_cache_buffer;
    preprocessingParams.kv_cache_block_scales_buffer = kv_scale_cache_buffer;
    preprocessingParams.tokens_info = workspaceViews.tokensInfo;
    preprocessingParams.seq_lens = p.getContextLengths();
    // For self-attention, cache_seq_lens indicates whether chunked context is used
    // (i.e. cache_seq_len > seq_len).
    preprocessingParams.cache_seq_lens = p.getSequenceLengths();
    preprocessingParams.cu_seq_lens = contextCuQSeqlens;
    preprocessingParams.cu_kv_seq_lens = contextCuKvSeqlens;
    preprocessingParams.rotary_embedding_inv_freq = workspaceViews.rotaryInvFreq;
    preprocessingParams.qkv_scale_orig_quant = p.getKvScaleOrigQuant();
    preprocessingParams.spec_decoding_position_offsets = nullptr;
    preprocessingParams.helix_position_offsets = p.getHelixPositionOffsets();
    preprocessingParams.helix_is_inactive_rank = p.getHelixIsInactiveRank();

    // Sparse KV write
    preprocessingParams.sparse_kv_indices = p.getSparseKvIndices();
    preprocessingParams.sparse_kv_offsets = p.getSparseKvOffsets();

    // Scalars
    preprocessingParams.batch_size = p.batch_size;
    preprocessingParams.max_input_seq_len = p.input_seq_length;
    preprocessingParams.max_kv_seq_len = p.max_past_kv_length;
    preprocessingParams.cyclic_kv_cache_len = p.cyclic_attention_window_size;
    preprocessingParams.token_num = p.num_tokens;
    preprocessingParams.remove_padding = remove_padding;
    preprocessingParams.head_num = mNumAttnHeads;
    preprocessingParams.kv_head_num = mNumAttnKVHeads;
    preprocessingParams.qheads_per_kv_head = mNumAttnHeads / mNumAttnKVHeads;
    preprocessingParams.size_per_head = getHeadSize();
    preprocessingParams.cache_type = cache_type;
    preprocessingParams.separate_q_kv_output = enablePagedKVContextFMHA;
    preprocessingParams.quantized_fp8_output = mFP8ContextFMHA;
    preprocessingParams.generation_phase = false;
    preprocessingParams.multi_processor_count = mMultiProcessorCount;

    preprocessingParams.is_last_chunk
        = !p.attn.attention_chunk_size.has_value() || (p.input_seq_length == p.max_past_kv_length);
    applyFeatures(preprocessingParams);

    {
        std::string const beforeRopeStr = "ctx attention before RoPE at layer " + std::to_string(p.attn.layer_idx);
        TLLM_CHECK_DEBUG_WITH_INFO(tensorrt_llm::runtime::utils::tensorHasInvalid(p.num_tokens,
                                       (local_hidden_units_qo + 2 * local_hidden_units_kv), p.getType(),
                                       const_cast<T*>(attention_input), stream, beforeRopeStr)
                == false,
            "Found invalid number (NaN or Inf) in " + beforeRopeStr);
    }

    if (useSageAttnSeparateQkv)
    {
        TLLM_CHECK_WITH_INFO(mask_type == AttentionMaskType::PADDING,
            "SageAttention only supports dense (padding) mask, got mask type %d.", static_cast<int>(mask_type));
        TLLM_CHECK_WITH_INFO(
            sage_attn_num_elts_per_blk_q > 0 && sage_attn_num_elts_per_blk_k > 0 && sage_attn_num_elts_per_blk_v == 1,
            "SageQuant requires positive block sizes for Q and K while the block size for V must be 1.");
        // A zero stride means no quantizer kernel supports this block size.
        TLLM_CHECK_WITH_INFO(q_max_n_blk > 0 && k_max_n_blk > 0,
            "No SageQuant kernel on sm_%d for block sizes (q, k, v) = (%d, %d, %d) with qk_int8=%s. SM90 "
            "requires (2, 16, 1) together with INT8 Q/K; other architectures require q/k block sizes of 1, 4, 16, "
            "or 32.",
            mSM, static_cast<int>(sage_attn_num_elts_per_blk_q), static_cast<int>(sage_attn_num_elts_per_blk_k),
            static_cast<int>(sage_attn_num_elts_per_blk_v), sage_attn_qk_int8 ? "true" : "false");
        TLLM_CHECK_WITH_INFO(!p.fwd.kv_scale_quant_orig,
            "SageAttention disregards the configured p.fwd.kv_scale_quant_orig, invalidating the result.");

        // Common p for sageQuant
        tc::SageQuantParams sageQuantParams{};
        sageQuantParams.headDim = getHeadSize();
        sageQuantParams.inputType = std::is_same_v<T, __nv_bfloat16> ? DATA_TYPE_BF16 : DATA_TYPE_FP16;
        sageQuantParams.quantType = sage_attn_qk_int8 ? DATA_TYPE_INT8 : DATA_TYPE_E4M3;
        sageQuantParams.vStage = 0;
        sageQuantParams.sumSeqLensV = p.total_kv_len;
        sageQuantParams.numHeadsV = mNumAttnKVHeads;
        sageQuantParams.kSmooth = sage_attn_smooth_k;
        sageQuantParams.ptrV = p.getV<T>();
        sageQuantParams.ptrVQuant = workspaceViews.fp8VBuf;
        sageQuantParams.ptrVScale = workspaceViews.sageVScale;
        sageQuantParams.ptrKForMean = p.getK<T>();
        sageQuantParams.ptrKMeanSum = workspaceViews.sageKMeanSum;
        sageQuantParams.ptrKMean = workspaceViews.sageKMean;
        sageQuantParams.smCount = mMultiProcessorCount;
        sageQuantParams.stream = stream;

        // Quantize into Q, SfsQ, SfsV
        sageQuantParams.sumSeqLensQk = p.num_tokens;
        sageQuantParams.batchSize = p.batch_size;
        sageQuantParams.numHeads = mNumAttnHeads;
        sageQuantParams.partition = tc::getSageQPartition(hopperSage);
        sageQuantParams.tokenBlockSize = sage_attn_num_elts_per_blk_q;
        sageQuantParams.ptrCuSeqLensQk = contextCuQSeqlens;
        sageQuantParams.ptrQk = attention_input;
        sageQuantParams.ptrQkQuant = workspaceViews.fp8QBuf;
        sageQuantParams.ptrQkScale = workspaceViews.sageQScale;
        sageQuantParams.vStage = 1;
        tc::invokeSageQuant(sageQuantParams);

        // Quantize into K, SfsK, V
        sageQuantParams.sumSeqLensQk = p.total_kv_len;
        sageQuantParams.batchSize = p.batch_size;
        sageQuantParams.numHeads = mNumAttnKVHeads;
        sageQuantParams.partition = tc::getSageKPartition(hopperSage);
        sageQuantParams.tokenBlockSize = sage_attn_num_elts_per_blk_k;
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
        std::string const afterRopeStr = "ctx attention after RoPE at layer " + std::to_string(p.attn.layer_idx);
        TLLM_CHECK_DEBUG_WITH_INFO(tensorrt_llm::runtime::utils::tensorHasInvalid(p.num_tokens,
                                       (local_hidden_units_qo + 2 * local_hidden_units_kv), p.getType(),
                                       const_cast<T*>(attention_input), stream, afterRopeStr)
                == false,
            "Found invalid number (NaN or Inf) in " + afterRopeStr);
        sync_check_cuda_error(stream);
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
    // Separate QKV input layout (SageAttention):
    //    - q_ptr: [B, S, H, D], which supports variable sequence length
    //    - k_ptr: [B, S, H_kv, D], which supports variable sequence length
    //    - v_ptr: [B, S, H_kv, D_v], which supports variable sequence length
    //    - cu_q_seqlens: the cumulative query sequence lengths, needed for variable sequence length.
    //    - cu_kv_seqlens: the cumulative kv sequence lengths, needed for variable sequence length.
    //    - total_kv_len: the total kv sequence length, needed for variable sequence length.

    // Construct the fmha p for running kernels.
    MHARunnerParams fmhaParams{};
    fmhaParams.b = p.batch_size;
    fmhaParams.qSeqLen = p.input_seq_length;
    fmhaParams.kvSeqLen = p.max_past_kv_length;
    // Disable sliding window attention when it is not needed.
    fmhaParams.slidingWindowSize = dense_context_fmha ? p.max_past_kv_length : p.cyclic_attention_window_size;
    fmhaParams.totalQSeqLen = p.num_tokens;
    fmhaParams.totalKvSeqLen = p.total_kv_len;
    // Device buffer pointers.
    if (useSageAttnSeparateQkv)
    {
        // SageAttention: use quantized FP8/INT8 Q/K/V buffers as separate inputs.
        TLLM_CHECK_WITH_INFO(separateQAndKvInput, "Separate QKV input is required for sage attention FMHA");
        fmhaParams.qkvPtr = nullptr;
        fmhaParams.qPtr = reinterpret_cast<void const*>(workspaceViews.fp8QBuf);
        fmhaParams.kPtr = reinterpret_cast<void const*>(workspaceViews.fp8KBuf);
        fmhaParams.vPtr = reinterpret_cast<void const*>(workspaceViews.fp8VBuf);
    }
    else
    {
        fmhaParams.qkvPtr = mFP8ContextFMHA ? reinterpret_cast<void const*>(workspaceViews.fp8QkvBuf)
                                            : reinterpret_cast<void const*>(attention_input);
        fmhaParams.qPtr = reinterpret_cast<void const*>(workspaceViews.qBuf);
    }
    if (useSageAttnSeparateQkv)
    {
        fmhaParams.qScalePtr = workspaceViews.sageQScale;
        fmhaParams.kScalePtr = workspaceViews.sageKScale;
        fmhaParams.vScalePtr = workspaceViews.sageVScale;
        fmhaParams.qMaxNBlock = q_max_n_blk;
        fmhaParams.kMaxNBlock = k_max_n_blk;
        fmhaParams.vMaxNBlock = v_max_n_blk;
    }
    fmhaParams.kvPtr = nullptr;
    // Only use [totalLength, h / cpSize, Dh].
    fmhaParams.outputPtr = p.getOutput();
    fmhaParams.attentionSinksPtr = p.getAttentionSinks();
    if constexpr (std::is_same_v<KVCacheBuffer, KVBlockArray>)
    {
        fmhaParams.pagedKvCache = kv_cache_buffer;
        fmhaParams.pagedKvSfCache = kv_scale_cache_buffer;
    }
    fmhaParams.cuQSeqLenPtr = contextCuQSeqlens;
    fmhaParams.kvSeqLenPtr = p.getSequenceLengths();
    fmhaParams.cuKvSeqLenPtr = contextCuKvSeqlens;
    fmhaParams.cuMaskRowsPtr = workspaceViews.cuMaskRows;
    fmhaParams.variableWindowTokenStartsPtr = p.getVariableWindowTokenStarts();
    fmhaParams.variableWindowTokenEndsPtr = p.getVariableWindowTokenEnds();
    fmhaParams.tileCounterPtr = workspaceViews.fmhaTileCounter;
    fmhaParams.scaleBmm1Ptr = workspaceViews.fmhaBmm1Scale;
    fmhaParams.scaleBmm2Ptr = workspaceViews.fmhaBmm2Scale;
    fmhaParams.stream = stream;
    fmhaParams.forceFp32Acc = mFMHAForceFP32Acc;
    fmhaParams.softmaxStatsPtr = p.getSoftmaxStatsTensor();
    fmhaParams.trtllmGenJITWarmup = p.meta.trtllm_gen_jit_warmup;
    fmhaParams.trtllmGenJITWarmupMaxNumRequests = p.meta.max_num_requests;
    fmhaParams.trtllmGenJITWarmupMaxSeqLenQ = p.meta.max_context_length;
    fmhaParams.trtllmGenJITWarmupMaxSeqLenKv = p.meta.max_seq_len;

    if (useTllmGenSparseAttention())
    {
        // Sparse context reuses generation-style trtllm-gen kernels; provide the scratch pool
        // and per-CTA counter so the autotuner can select MultiCtasKv variants.
        fmhaParams.multiCtasKvScratchPtr = workspaceViews.fmhaMultiCtasKvScratch;
        fmhaParams.multiCtasKvCounterPtr = getMultiCtasKvCounter(p.getMaxNumSequences(), p.getQueryInput().device());
    }
    applyFeatures(fmhaParams);

    // Skip-softmax attention parameters
    fmhaParams.skipSoftmaxThresholdScaleFactor = p.fwd.sparse_runtime_params.threshold_scale_factor_prefill;
#ifdef SKIP_SOFTMAX_STAT
    fmhaParams.skipSoftmaxTotalBlocks = p.getSkipSoftmaxStat();
    fmhaParams.skipSoftmaxSkippedBlocks = p.getSkipSoftmaxStat() + 1;
#else
    if (tensorrt_llm::common::getEnvPrintSkipSoftmaxStat())
    {
        TLLM_THROW("To print skip softmax stat, please run build_wheel.py with -DSKIP_SOFTMAX_STAT");
    }
#endif

    if (p.attn.attention_chunk_size)
    {
        fmhaParams.chunkedAttentionSize = *p.attn.attention_chunk_size;
    }

    // Run the fmha kernel.
    launch(fmhaParams);
    sync_check_cuda_error(stream);

    invokeKvCachePostprocessing(preprocessingParams, stream);
    sync_check_cuda_error(stream);
}

#define INSTANTIATE_MHA_CONTEXT_HELPERS(T)                                                                             \
    template KvCacheBuffers<KVBlockArray> MHA::getKvCacheBuffers<T>(FmhaParams const&, bool) const;                    \
    template void MHA::buildContextDecoderInfo<T>(                                                                     \
        FmhaParams const&, tensorrt_llm::common::op::AttentionContextWorkspaceViews<T> const&, cudaStream_t) const;    \
    template void MHA::enqueueFusedContext<T>(                                                                         \
        FmhaParams const&, cudaStream_t, std::function<void(MHARunnerParams const&)> const&) const;
INSTANTIATE_MHA_CONTEXT_HELPERS(float)
INSTANTIATE_MHA_CONTEXT_HELPERS(half)
#ifdef ENABLE_BF16
INSTANTIATE_MHA_CONTEXT_HELPERS(__nv_bfloat16)
#endif
#undef INSTANTIATE_MHA_CONTEXT_HELPERS

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

std::string MHA::toString() const
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
    TRTLLM_DUMP_MEMBER(mFMHAForceFP32Acc)
    TRTLLM_DUMP_MEMBER(mMultiBlockMode)
    TRTLLM_DUMP_MEMBER(mFP8ContextFMHA)
    TRTLLM_DUMP_MEMBER(mSM)
    TRTLLM_DUMP_MEMBER(mUseTllmGen)
    TRTLLM_DUMP_MEMBER(mMultiProcessorCount)
    TRTLLM_DUMP_MEMBER(mMaxSharedMemoryPerBlockOptin)
#undef TRTLLM_DUMP_MEMBER
    return ss.str();
}

namespace trtllm::attention
{
using tensorrt_llm::kernels::KVBlockArray;
using tensorrt_llm::kernels::MlaParams;
using tensorrt_llm::torch_ext::MHA;
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

bool AttentionBase::operator==(AttentionBase const& other) const
{
    return keyFields(*this) == keyFields(other);
}

int AttentionBase::getHeadSize() const
{
    TLLM_CHECK_WITH_INFO(mHeadSize > 0, "Trying to read mHeadSize before it's been initialized");
    return mHeadSize;
}

MHA::MHA(AttentionBase const& key)
    : TrtllmAttention(key)
{
    // Derived op state.
    mPagedKVCache = use_kv_cache;
    // Only the FP8 context FMHA kernels read an NVFP4 KV cache, whatever the output type; they do so on SM100-family
    // GPUs only, so elsewhere an NVFP4 KV cache still fails the FP4 KV check.
    mFP8ContextFMHA = is_fp8_out || is_fp4_out || (quant_mode.hasFp8KvCache() && paged_context_fmha)
        || (quant_mode.hasFp4KvCache() && tensorrt_llm::common::isSM100Family(mSM)) || sageAttnRequested();
    if (uses_spcompress)
    {
        TORCH_CHECK(mSM == 107, "uses_spcompress is only supported on SM107. Got SM version: ", mSM);
        TORCH_CHECK(mFP8ContextFMHA, "uses_spcompress requires FP8 context FMHA.");
    }
    // The fused context libraries serve FP16 and BF16 without relative position embedding.
    bool fusedContext = false;
    if (!(type == tensorrt_llm::DataType::kHALF || type == tensorrt_llm::DataType::kBF16))
    {
        TLLM_LOG_WARNING("Fall back to unfused MHA because of unsupported data type.");
    }
    else if (position_embedding_type == tensorrt_llm::kernels::PositionEmbeddingType::kRELATIVE)
    {
        TLLM_LOG_WARNING("Fall back to unfused MHA because of relative position embedding.");
    }
    else if (isQuantizedCrossCacheUnfused())
    {
        TLLM_LOG_WARNING("Fall back to unfused MHA because of cross attention + quantized kv cache.");
    }
    else
    {
        fusedContext = true;
    }

    // Pre-Check of FP8 Context FMHA.
    if (mFP8ContextFMHA)
    {
        TLLM_CHECK_WITH_INFO(fusedContext, "FP8 FMHA cannot be enabled because Context FMHA is not supported.");
        TLLM_CHECK_WITH_INFO(mSM == 89 || mSM == 90 || tensorrt_llm::common::isSM100Family(mSM)
                || tensorrt_llm::common::isSM120Family(mSM),
            "FP8 FMHA can only be enabled on sm_89, sm_90, sm_100f or sm_120f.");
    }

    // Check requirements for FP4 output.
    TLLM_CHECK_WITH_INFO(!is_fp4_out || fusedContext, "Context FMHA must enable if fuse_fp4_quant is enabled");
    TLLM_CHECK_WITH_INFO(
        !is_fp4_out || tensorrt_llm::common::isSM100Family(mSM) || tensorrt_llm::common::isSM120Family(mSM),
        "fuse_fp4_quant only supports SM100f or SM120f devices.");

    // Check requirements for FP4 KV cache.
    TLLM_CHECK_WITH_INFO(!quant_mode.hasFp4KvCache() || mFP8ContextFMHA, "FP4 KV cache requires FP8 context FMHA");

    TLLM_CHECK(isRoPE() == (rotary_embedding_dim != 0));
    TLLM_CHECK_WITH_INFO((mSM >= 80) || (type != tensorrt_llm::DataType::kBF16),
        "Unsupported data type, pre SM 80 GPUs do not support bfloat16");
    TLLM_CHECK_WITH_INFO(!is_spec_decoding_enabled
            || ((type == tensorrt_llm::DataType::kHALF || type == tensorrt_llm::DataType::kBF16) && use_kv_cache),
        "Speculative decoding mode doesn't support the data type or cross attention.");

    mTrtllmGen = makeFmha<TrtllmGenFmha<MHA>>();
    mFmhaV2 = makeFmha<FmhaV2Fmha<MHA>>();
    mXqa = makeFmha<XqaFmha<MHA>>();
    mMmha = makeFmha<MmhaFmha>();
    mUnfused = makeFmha<UnfusedFmha>();
}

void MHA::initContext()
{
#ifdef SKIP_SOFTMAX_STAT
    if (!mDefaultSkipSoftmaxStat.defined())
    {
        mDefaultSkipSoftmaxStat = at::zeros({2}, at::TensorOptions().dtype(at::kUInt32).device(at::kCUDA));
    }
#endif
}

void MHA::initGeneration()
{
#ifdef SKIP_SOFTMAX_STAT
    if (!mDefaultSkipSoftmaxStat.defined())
    {
        mDefaultSkipSoftmaxStat = at::zeros({2}, at::TensorOptions().dtype(at::kUInt32).device(at::kCUDA));
    }
#endif
    // The multi-block choice is configuration; resolve the environment overrides once.
    if (common::getEnvForceDeterministicAttention())
    {
        mMultiBlockMode = false;
    }
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
}

MHARunnerFixedParams MHA::contextFixedParams() const
{
    MHARunnerFixedParams fmhaParams{};

    // The fused libraries serve only FP16 and BF16.
    Data_type data_type;
    if (type == tensorrt_llm::DataType::kHALF)
    {
        data_type = DATA_TYPE_FP16;
    }
    else if (type == tensorrt_llm::DataType::kBF16)
    {
        data_type = DATA_TYPE_BF16;
    }
    else
    {
        data_type = DATA_TYPE_FP32;
    }
    // The output dtype; the output-quant feature narrows it.
    fmhaParams.dataTypeOut = data_type;
    // FP8 FMHA should be used with fp8 workflow together.
    if (mFP8ContextFMHA)
    {
        data_type = DATA_TYPE_E4M3;
    }
    // The input dtype, and the KV input dtype, which defaults to the same.
    fmhaParams.dataType = data_type;
    fmhaParams.dataTypeKv = data_type;
    // TODO: remove forceFp32Acc from MHARunnerFixedParams after adding host_runtime_perf_knobs to
    // bertAttentionPlugin input tensors, so that we can change mLaunchParams.force_fp32_acc value
    // in runtime.
    fmhaParams.forceFp32Acc = false;

    // setting attention mask type based on the mask type
    if (use_variable_window)
    {
        TLLM_CHECK_WITH_INFO(mSM == 100 || mSM == 103, "Variable-window attention requires SM100 or SM103.");
        fmhaParams.attentionMaskType = ContextAttentionMaskType::VARIABLE_WINDOW;
    }
    else
    {
        fmhaParams.setAttentionMaskType(static_cast<std::int8_t>(mask_type));
    }

    if (cross_attention)
    {
        // always use paged-kv-fmha if paged_kv cache is used.
        fmhaParams.attentionInputLayout
            = mPagedKVCache ? AttentionInputLayout::Q_PAGED_KV : AttentionInputLayout::Q_CONTIGUOUS_KV;
    }
    else if (!use_kv_cache)
    {
        fmhaParams.attentionInputLayout = AttentionInputLayout::PACKED_QKV;
    }
    else
    {
        fmhaParams.attentionInputLayout = (mPagedKVCache && paged_context_fmha) ? AttentionInputLayout::Q_PAGED_KV
                                                                                : AttentionInputLayout::PACKED_QKV;
    }
    fmhaParams.isSPadded = !remove_padding;
    fmhaParams.numQHeads = mNumAttnHeads;
    fmhaParams.numKvHeads = mNumAttnKVHeads;
    fmhaParams.headSize = mHeadSize;
    fmhaParams.headSizeV = mHeadSize;
    fmhaParams.qScaling = q_scaling;
    fmhaParams.attnLogitSoftcappingScale = attn_logit_softcapping_scale;
    fmhaParams.useSpcompress = uses_spcompress;
    fmhaParams.fusesDsv4InvRopeFp8Quant = fuses_dsv4_inv_rope_fp8_quant;

    applyFeatures(fmhaParams);

    if (fuses_dsv4_inv_rope_fp8_quant)
    {
        fmhaParams.dataTypeOut = DATA_TYPE_E4M3;
    }
    return fmhaParams;
}

bool MHA::isSeparateQAndKvInput() const
{
    return contextFixedParams().attentionInputLayout != AttentionInputLayout::PACKED_QKV;
}

bool MHA::supportsNvFp4Output()
{
    initPhase(/*isGen=*/false);
    initPhase(/*isGen=*/true);
    bool const fusedContextKernel = (mTrtllmGen != nullptr && mTrtllmGen->hasKernels(/*isGen=*/false))
        || (mFmhaV2 != nullptr && mFmhaV2->hasKernels(/*isGen=*/false));
    bool const fp4GenerationKernel = (mTrtllmGen != nullptr && mTrtllmGen->hasKernels(/*isGen=*/true))
        || (mXqa != nullptr && mXqa->hasKernels(/*isGen=*/true));
    return fusedContextKernel && fp4GenerationKernel;
}

bool MHA::supportsVariableWindow()
{
    initPhase(/*isGen=*/false);
    return use_variable_window
        && ((mTrtllmGen != nullptr && mTrtllmGen->hasKernels(/*isGen=*/false))
            || (mFmhaV2 != nullptr && mFmhaV2->hasKernels(/*isGen=*/false)));
}

void MHA::prepareInputs(FmhaParams& p, bool isGen)
{
    TORCH_CHECK(!isGen || !use_variable_window, "variable-window bounds are supported only for context attention");
    if (!isGen && use_variable_window)
    {
        TORCH_CHECK(!p.is_cross && !p.use_spec_decoding && !sageAttnRequested(),
            "Variable-window attention requires non-cross, non-speculative context attention without SageAttention.");
        for (auto const& tensor :
            {p.fwd.variable_window_token_starts.value(), p.fwd.variable_window_token_ends.value()})
        {
            TORCH_CHECK(tensor.is_cuda() && tensor.device() == p.getQueryInput().device(),
                "variable-window bounds must be on the same device as qkv_or_q");
            TORCH_CHECK(tensor.scalar_type() == at::kInt && tensor.dim() == 1 && tensor.is_contiguous(),
                "variable-window bounds must be contiguous 1-D int32 tensors");
            TORCH_CHECK(tensor.numel() == p.token_offset + p.num_tokens,
                "variable-window bounds must have one element per packed context query token");
        }
    }
    TLLM_CHECK_WITH_INFO(p.fwd.is_fused_qkv || sageAttnRequested() || p.is_cross,
        "For non-cross, non-SageAttention attention, only fused QKV is supported now.");

    if (isGen)
    {
        TORCH_CHECK(p.batch_size > 0 && p.meta.effective_beam_width > 0,
            "Generation attention requires positive num_seqs and beam_width.");
        TORCH_CHECK(p.batch_size % p.meta.effective_beam_width == 0, "num_seqs must be divisible by beam_width.");
        TORCH_CHECK(p.num_tokens % p.batch_size == 0,
            "seq_len should be same for all generation requests, num_tokens=", p.num_tokens,
            ", num_seqs=", p.batch_size);
    }

    if (p.fwd.softmax_stats_tensor.has_value())
    {
        auto const& stats = p.fwd.softmax_stats_tensor.value();
        TORCH_CHECK(stats.scalar_type() == at::ScalarType::Float, "softmax_stats_tensor must have float type");
        TORCH_CHECK(stats.dim() == 3, "softmax_stats_tensor must have shape [num_tokens, num_heads, 2]");
        TORCH_CHECK(stats.size(0) >= p.num_tokens, "softmax_stats_tensor must have first dimension >= num_tokens");
        TORCH_CHECK(stats.size(1) >= num_heads, "softmax_stats_tensor must have second dimension >= num_heads");
        TORCH_CHECK(stats.size(2) == 2, "softmax_stats_tensor must have third dimension == 2");
    }

#ifdef SKIP_SOFTMAX_STAT
    if (!p.attn.skip_softmax_stat.defined())
    {
        p.attn.skip_softmax_stat = mDefaultSkipSoftmaxStat;
    }
    auto const& skipStats = p.attn.skip_softmax_stat;
    TORCH_CHECK(skipStats.is_cuda() && skipStats.is_contiguous() && skipStats.scalar_type() == at::ScalarType::UInt32
            && skipStats.numel() >= 2,
        "skip_softmax_stat must contain at least two contiguous CUDA uint32 values.");
#endif

    if (!isGen && p.batch_size > 0)
    {
        // Unfused self-attention cannot read a cached prefix. Cross attention can run
        // unfused when encoder K/V is supplied; UnfusedFmha checks that per call.
        bool const needsFusedPagedContext = paged_context_fmha && use_kv_cache && !cross_attention;
        TLLM_CHECK_WITH_INFO(!needsFusedPagedContext || !isRelativePosition(),
            "Paged-context attention (chunked prefill, KV cache reuse or speculative draft tokens) is not supported "
            "with relative position embedding: that attention always runs unfused, and the unfused path cannot attend "
            "to "
            "cached KV. Disable chunked prefill, KV cache reuse and speculative decoding for this model.");
    }

    // Phase extents. Python sends the phase's token and row counts, which only it can
    // know before the phase split; the rest is a reduction over the host length arrays,
    // which the op can do itself.
    p.total_kv_len = p.getTotalKvLen(p.seq_offset, p.batch_size, isGen);
    // Encoder CUDA graphs capture a padded context extent; the override widens both.
    if (!isGen && p.meta.max_context_q_len_override.has_value())
    {
        p.input_seq_length = p.meta.max_context_q_len_override.value();
        p.max_past_kv_length = p.meta.max_context_q_len_override.value();
    }

    if (p.fwd.attention_sinks.has_value())
    {
        TORCH_CHECK(p.fwd.attention_sinks.value().scalar_type() == torch::kFloat32,
            "Expected attention_sinks to have float dtype");
    }

    // The KV-cache scales are meaningful only when the cache is actually quantized.
    // `TrtllmAttention` hands them over unconditionally so the XQA kernels always see a
    // valid pointer when `isKVCacheQuantized` is true; consumers in turn read a null
    // pointer as "no KV-cache dequant", and some branch on exactly that. Drop the pair
    // here when the quant mode says otherwise, so the whole op -- including the
    // `!kv_scale_quant_orig` precondition on the SageAttention path -- keeps seeing a
    // consistent pair. They travel together: a lone scale describes only half of the
    // conversion and is never usable.
    if (!quant_mode.hasKvCacheQuant() || !p.fwd.kv_scale_orig_quant.has_value()
        || !p.fwd.kv_scale_quant_orig.has_value())
    {
        p.fwd.kv_scale_orig_quant.reset();
        p.fwd.kv_scale_quant_orig.reset();
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
        TORCH_CHECK(tensor.size(0) >= p.batch_size + 1, name, " must have at least num_seqs + 1 elements.");
    };
    if (!isGen)
    {
        checkCuSeqlens(p.fwd.cu_q_seqlens, "cu_q_seqlens");
        checkCuSeqlens(p.fwd.cu_kv_seqlens, "cu_kv_seqlens");
    }
}

void MHA::enqueueContext(FmhaParams& p, PhasedFmha<MHA>& fmha)
{
    auto const stream = at::cuda::getCurrentCUDAStream(p.getQueryInput().get_device());
    fmha.runContext(*this, p, stream);
    sync_check_cuda_error(stream);
}

void MHA::enqueueGeneration(FmhaParams& p, PhasedFmha<MHA>& fmha)
{
    auto const stream = at::cuda::getCurrentCUDAStream(p.getQueryInput().get_device());
    TLLM_CHECK_WITH_INFO(
        !p.attn.attention_chunk_size.has_value() || p.cyclic_attention_window_size >= p.max_past_kv_length,
        "Chunked-attention and sliding-window-attention should not be enabled at the same time.");
    fmha.runGeneration(*this, p, stream);
    {
        std::string const afterGenStr = "gen attention at layer " + std::to_string(p.attn.layer_idx);
        TLLM_CHECK_DEBUG_WITH_INFO(tensorrt_llm::runtime::utils::tensorHasInvalid(p.num_tokens, num_heads * head_size,
                                       p.getType(), p.getOutput(), stream, afterGenStr)
                == false,
            "Found invalid number (NaN or Inf) in " + afterGenStr);
    }
    sync_check_cuda_error(stream);
}

// ---------------------------------------------------------------------------
// MLA
// ---------------------------------------------------------------------------

MLA::MLA(AttentionBase const& key)
    : TrtllmAttention(key)
{
    TLLM_CHECK_WITH_INFO(type == tensorrt_llm::DataType::kHALF || type == tensorrt_llm::DataType::kBF16,
        "MLA(Deepseek v2) only support fmha, which needs fp16 or bf16 activations.");
    TLLM_CHECK_WITH_INFO((mSM >= 80) || (type != tensorrt_llm::DataType::kBF16),
        "Unsupported data type, pre SM 80 GPUs do not support bfloat16");
    TLLM_CHECK_WITH_INFO(!isRelativePosition(), "MLA(Deepseek v2) only support fmha");
    TLLM_CHECK_WITH_INFO(!dense_context_fmha, "MLA(Deepseek v2) currently not support dense fmha");
    TLLM_CHECK_WITH_INFO(use_kv_cache && remove_padding, "MLA(Deepseek v2) only support paged kv cache");
    TLLM_CHECK_WITH_INFO(!cross_attention, "MLA(Deepseek v2) do not support cross attention right now");
    TLLM_CHECK_WITH_INFO(
        mask_type != AttentionMaskType::CUSTOM_MASK, "MLA(Deepseek v2) do not support custom mask right now");
    TLLM_CHECK(isRoPE() == (rotary_embedding_dim != 0));

    mMLAParams = {static_cast<int>(q_lora_rank), static_cast<int>(kv_lora_rank), static_cast<int>(qk_nope_head_dim),
        static_cast<int>(qk_rope_head_dim), static_cast<int>(v_head_dim), static_cast<int>(predicted_tokens_per_seq),
        static_cast<int>(mla_layer_num), static_cast<int>(rope_append)};
    bool const mla_dims_supported = mMLAParams.qk_rope_head_dim == 64
        && ((mMLAParams.rope_append && mMLAParams.kv_lora_rank == 512)
            || (!mMLAParams.rope_append && mMLAParams.kv_lora_rank == 448));
    TLLM_CHECK_WITH_INFO(mla_dims_supported,
        "MLA(Deepseek v2) only supports qk_rope_head_dim=64 with kv_lora_rank=512 "
        "(rope_append=true) or "
        "kv_lora_rank=448 (rope_append=false).");

    mIsGenerationMLA = head_size == kv_lora_rank + qk_rope_head_dim;
    // Latent KV layout: one KV head of kv_lora_rank + qk_rope_head_dim.
    mNumKVHeads = 1;
    mHeadSize = kv_lora_rank + qk_rope_head_dim;
    mNumAttnKVHeads = mNumKVHeads;

    auto constexpr kMaxSkipCorrectionThreshold = 32.0;
    TLLM_CHECK_WITH_INFO(skip_correction_threshold >= 0.0 && skip_correction_threshold <= kMaxSkipCorrectionThreshold,
        "skip_correction_threshold must be in the range (0, 32] when enabled, or 0 when disabled.");
    bool const applySkipCorrection = mSM == 100 || mSM == 103;
    mSkipCorrectionThreshold = applySkipCorrection ? static_cast<float>(skip_correction_threshold) : 0.0F;

    mUseNvfp4MlaKvCache = use_nvfp4_mla_kv_cache;
    mFP8ContextMLA = (mSM == 90 || tensorrt_llm::common::isSM100Family(mSM) || tensorrt_llm::common::isSM120Family(mSM))
        && (quant_mode.hasFp8KvCache() || mUseNvfp4MlaKvCache);
    mFP8GenerationMLA = quant_mode.hasFp8KvCache() || mUseNvfp4MlaKvCache;
    if (uses_spcompress)
    {
        TORCH_CHECK(mSM == 107, "uses_spcompress is only supported on SM107. Got SM version: ", mSM);
        TORCH_CHECK(mFP8ContextMLA, "uses_spcompress requires FP8 context MLA.");
    }
    if (mFP8GenerationMLA)
    {
        TLLM_CHECK_WITH_INFO(mSM == 89 || mSM == 90 || tensorrt_llm::common::isSM100Family(mSM)
                || tensorrt_llm::common::isSM120Family(mSM),
            "FP8 Generation MLA is supported on Ada, Hopper or Blackwell architecture.");
    }
    TLLM_CHECK_WITH_INFO(!quant_mode.hasFp4KvCache() || mUseNvfp4MlaKvCache,
        "FP4 KV cache requires static sparse MLA with an FP8 scratch pool");

    mTrtllmGen = makeFmha<TrtllmGenFmha<MLA>>();
    mFlashMla = makeFmha<FlashMlaFmha>();
    mXqa = makeFmha<XqaFmha<MLA>>();
    mFmhaV2 = makeFmha<FmhaV2Fmha<MLA>>();
}

void MLA::initContext() {}

MHARunnerFixedParams MLA::contextFixedParams() const
{
    MHARunnerFixedParams fmhaParams{};
    fmhaParams.forceFp32Acc = false;
    fmhaParams.setAttentionMaskType(static_cast<std::int8_t>(mask_type));
    fmhaParams.isSPadded = !remove_padding;
    fmhaParams.numQHeads = mNumAttnHeads;
    fmhaParams.attnLogitSoftcappingScale = attn_logit_softcapping_scale;
    fmhaParams.useSpcompress = uses_spcompress;
    fmhaParams.fusesDsv4InvRopeFp8Quant = fuses_dsv4_inv_rope_fp8_quant;
    fmhaParams.headSizeQkNope = mMLAParams.qk_nope_head_dim;
    if (useSparseMLA())
    {
        fmhaParams.attentionInputLayout = AttentionInputLayout::Q_PAGED_KV;
        fmhaParams.numKvHeads = 1;
        fmhaParams.headSize = mMLAParams.kv_lora_rank + mMLAParams.qk_rope_head_dim;
        fmhaParams.headSizeV
            = mMLAParams.rope_append ? mMLAParams.kv_lora_rank : mMLAParams.kv_lora_rank + mMLAParams.qk_rope_head_dim;
        // Adjust the qScaling for the absorption mode.
        fmhaParams.qScaling = q_scaling * sqrt((float) (mMLAParams.qk_nope_head_dim + mMLAParams.qk_rope_head_dim))
            / sqrtf((float) (mMLAParams.kv_lora_rank + mMLAParams.qk_rope_head_dim));
    }
    else
    {
        // Context MLA always use separate_q_k_v layout
        fmhaParams.attentionInputLayout = AttentionInputLayout::SEPARATE_Q_K_V;
        // Context attention of MLA is different
        fmhaParams.numKvHeads = num_heads;
        fmhaParams.headSize = mMLAParams.qk_nope_head_dim + mMLAParams.qk_rope_head_dim;
        // Ideally this should be mMLAParams.v_head_dim, but because we initialize both MLA
        // context(v_head_dim=128) and gen(v_head_dim=512) runners in a single op, the headSizeV
        // will be set to 512 when we create the gen attention op and that could fail to create
        // the context FMHA libraries. Luckily, for deepseek, qk_nope_head_dim is the
        // same as v_head_dim in context phase.
        fmhaParams.headSizeV = mMLAParams.qk_nope_head_dim;
        fmhaParams.qScaling = q_scaling;
    }
    applyFeatures(fmhaParams);

    // Context MLA computes in BF16 over FP8 Q/K/V when the cache is quantized; only the DSv4
    // fused epilogue narrows the output.
    Data_type const ioType = type == tensorrt_llm::DataType::kHALF ? DATA_TYPE_FP16 : DATA_TYPE_BF16;
    fmhaParams.dataType = mFP8ContextMLA ? DATA_TYPE_E4M3 : ioType;
    fmhaParams.dataTypeKv = mFP8ContextMLA ? DATA_TYPE_E4M3 : DATA_TYPE_BF16;
    fmhaParams.dataTypeOut = fuses_dsv4_inv_rope_fp8_quant ? DATA_TYPE_E4M3 : DATA_TYPE_BF16;

    return fmhaParams;
}

void MLA::initGeneration()
{
    // The multi-block choice is configuration; resolve the environment overrides once.
    if (common::getEnvForceDeterministicAttention())
    {
        mMultiBlockMode = false;
    }
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
}

void MLA::prepareInputs(FmhaParams& p, bool isGen)
{
    if (isGen)
    {
        TORCH_CHECK(p.batch_size > 0 && p.meta.effective_beam_width > 0,
            "Generation attention requires positive num_seqs and beam_width.");
        TORCH_CHECK(p.batch_size % p.meta.effective_beam_width == 0, "num_seqs must be divisible by beam_width.");
        TORCH_CHECK(p.num_tokens % p.batch_size == 0,
            "seq_len should be same for all generation requests, num_tokens=", p.num_tokens,
            ", num_seqs=", p.batch_size);
    }

    if (p.fwd.softmax_stats_tensor.has_value())
    {
        auto const& stats = p.fwd.softmax_stats_tensor.value();
        TORCH_CHECK(stats.scalar_type() == at::ScalarType::Float, "softmax_stats_tensor must have float type");
        TORCH_CHECK(stats.dim() == 3, "softmax_stats_tensor must have shape [num_tokens, num_heads, 2]");
        TORCH_CHECK(stats.size(0) >= p.num_tokens, "softmax_stats_tensor must have first dimension >= num_tokens");
        TORCH_CHECK(stats.size(1) >= num_heads, "softmax_stats_tensor must have second dimension >= num_heads");
        TORCH_CHECK(stats.size(2) == 2, "softmax_stats_tensor must have third dimension == 2");
    }

#ifdef SKIP_SOFTMAX_STAT
    if (!mDefaultSkipSoftmaxStat.defined())
    {
        checkNotCapturing("skip-softmax statistics allocation");
        mDefaultSkipSoftmaxStat = at::zeros({2}, at::TensorOptions().dtype(at::kUInt32).device(at::kCUDA));
    }
    if (!p.attn.skip_softmax_stat.defined())
    {
        p.attn.skip_softmax_stat = mDefaultSkipSoftmaxStat;
    }
    auto const& skipStats = p.attn.skip_softmax_stat;
    TORCH_CHECK(skipStats.is_cuda() && skipStats.is_contiguous() && skipStats.scalar_type() == at::ScalarType::UInt32
            && skipStats.numel() >= 2,
        "skip_softmax_stat must contain at least two contiguous CUDA uint32 values.");
#endif

    // Phase extents. Python sends the phase's token and row counts, which only it can
    // know before the phase split; the rest is a reduction over the host length arrays,
    // which the op can do itself.
    p.total_kv_len = p.getTotalKvLen(p.seq_offset, p.batch_size, isGen);
    if (!isGen && p.meta.max_context_q_len_override.has_value())
    {
        p.input_seq_length = p.meta.max_context_q_len_override.value();
        p.max_past_kv_length = p.meta.max_context_q_len_override.value();
    }

    TLLM_CHECK(!p.isFp4Out());

    if (!isGen)
    {
        if (use_sparse_attention)
        {
            // A missing q_pe means the upstream q_b fusion has already written the complete
            // rotated FP8 Q into quant_q_buffer.
            if (p.fwd.q_pe.has_value())
            {
                TORCH_CHECK(p.fwd.q_pe->dim() == 3);
                TORCH_CHECK(p.fwd.q_pe->strides()[2] == 1);
            }
            else
            {
                TORCH_CHECK(p.fwd.quant_q_buffer.has_value() && p.fwd.quant_scale_qkv.has_value(),
                    "Sparse MLA context requires q_pe unless quant_q_buffer already contains rotated Q.");
            }
            // Fused kv_a_layernorm: the norm weight implies `latent_cache` is the
            // RAW kv_a_proj output, with the caller's RMSNorm and concat dropped.
            if (p.fwd.kv_norm_weight)
            {
                auto const& kvNormWeight = p.fwd.kv_norm_weight.value();
                TORCH_CHECK(kvNormWeight.is_cuda(), "kv_norm_weight must be a CUDA tensor");
                TORCH_CHECK(kvNormWeight.is_contiguous(), "kv_norm_weight must be contiguous");
                TORCH_CHECK(kvNormWeight.scalar_type() == p.getQueryInput().scalar_type(),
                    "kv_norm_weight dtype must match the activation dtype");
                TORCH_CHECK(
                    p.fwd.latent_cache, "fused kv-norm needs latent_cache (the raw kv_a_proj output) to be provided");
                // The kernel norms the whole latent row, so a narrower weight would
                // read out of bounds. dsv3RopeOp.cpp checks the same on the
                // generation side.
                auto const kvNormWidth = mMLAParams.kv_lora_rank + mMLAParams.qk_rope_head_dim;
                TORCH_CHECK(kvNormWeight.numel() == kvNormWidth,
                    "kv_norm_weight must span kv_lora_rank + qk_rope_head_dim (", kvNormWidth, "), got ",
                    kvNormWeight.numel());
                // A last-dim slice, so rows are wider than the row itself. Forward
                // the real stride; only the innermost dim must be unit-stride.
                auto const& latentCache = p.fwd.latent_cache.value();
                TORCH_CHECK(
                    latentCache.dim() == 2, "latent_cache must be 2D for fused kv-norm, got ", latentCache.dim(), "D");
                TORCH_CHECK(latentCache.stride(1) == 1, "latent_cache must be unit-stride in its last dim");
                // The kernel walks rows with 16-byte vector loads, so a row start
                // that is not 16B-aligned faults with a bare misaligned-address
                // error far from here.
                auto const kEltsPer16B = 16 / latentCache.element_size();
                TORCH_CHECK(latentCache.stride(0) % kEltsPer16B == 0, "latent_cache row stride (",
                    latentCache.stride(0), ") must be a multiple of ", kEltsPer16B,
                    " for the fused kv-norm 16B vector loads");
                TORCH_CHECK(reinterpret_cast<uintptr_t>(latentCache.data_ptr()) % 16 == 0,
                    "latent_cache must be 16B-aligned for the fused kv-norm vector loads");
            }
        }
        else
        {
            TORCH_CHECK(p.key_input.has_value());
            TORCH_CHECK(p.value_input.has_value());
            TORCH_CHECK(p.key_input->dim() == 2);
            TORCH_CHECK(p.value_input->dim() == 2);
            TORCH_CHECK(p.key_input->strides()[1] == 1,
                "Separate-K context FMHA requires contiguous K features within each token");
            TORCH_CHECK(p.value_input->strides()[1] == 1,
                "Separate-V context FMHA requires contiguous V features within each token");
            auto const& k = p.key_input.value();
            auto const& v = p.value_input.value();
            TORCH_CHECK(k.scalar_type() == v.scalar_type(), "MLA context k and v must have the same dtype.");
            // FP8 k / v (and an FP8 Q in quant_q_buffer, if given with quant_scale_qkv): the caller already
            // quantized them for the FP8 context FMHA, which reads them instead of quantizing BF16 inputs.
            if (k.scalar_type() == torch::kFloat8_e4m3fn)
            {
                TORCH_CHECK(mFP8ContextMLA, "FP8 MLA context k / v require the FP8 context MLA FMHA.");
                TORCH_CHECK(!p.fwd.latent_cache.has_value(),
                    "FP8 MLA context k / v are for cached-KV context attention, without latent_cache.");
                TORCH_CHECK(k.is_contiguous() && v.is_contiguous(), "FP8 MLA context k and v must be contiguous.");
                TORCH_CHECK(k.size(1) == num_heads * (mMLAParams.qk_nope_head_dim + mMLAParams.qk_rope_head_dim)
                        && v.size(1) == num_heads * mMLAParams.v_head_dim,
                    "FP8 MLA context k / v must be [kv_tokens, num_heads * qk_head_dim] / [kv_tokens, num_heads * "
                    "v_head_dim].");
                // The FMHA reads p.total_kv_len rows straight out of the caller's k / v;
                // reject a short tensor instead of reading past its end.
                TORCH_CHECK(k.size(0) >= p.total_kv_len && v.size(0) >= p.total_kv_len,
                    "FP8 MLA context k / v must hold at least total_kv_len (", p.total_kv_len, ") tokens.");
                if (p.fwd.quant_q_buffer.has_value() && p.fwd.quant_scale_qkv.has_value())
                {
                    auto const& quantQ = p.fwd.quant_q_buffer.value();
                    TORCH_CHECK(quantQ.scalar_type() == torch::kFloat8_e4m3fn && quantQ.is_contiguous()
                            && quantQ.numel() >= p.getQueryInput().numel(),
                        "The FP8 MLA context quant_q_buffer must be a contiguous FP8 copy of q.");
                    // The value is not read as a scale here; the FMHA dequantizes
                    // Q / K / V with the kv cache scale, so it must equal
                    // kv_scale_orig_quant. The mixed path (BF16 Q, FP8 k / v)
                    // also quantizes Q with it.
                    auto const& quantScaleQkv = p.fwd.quant_scale_qkv.value();
                    TORCH_CHECK(quantScaleQkv.is_cuda() && quantScaleQkv.is_contiguous()
                            && quantScaleQkv.scalar_type() == at::ScalarType::Float && quantScaleQkv.numel() >= 1,
                        "The FP8 MLA context quant_scale_qkv must be a contiguous CUDA float scalar.");
                }
            }
            else
            {
                TORCH_CHECK(k.scalar_type() == p.getQueryInput().scalar_type(),
                    "MLA context k / v must be FP8 E4M3 or have the dtype of q.");
            }
        }
    }
    else
    {
        TORCH_CHECK(p.fwd.latent_cache.has_value());
        TORCH_CHECK(p.fwd.q_pe.has_value());
        TORCH_CHECK(p.fwd.q_pe->dim() == 3);
        TORCH_CHECK(p.fwd.q_pe->strides()[2] == 1);
        TLLM_CHECK_WITH_INFO(p.getCuQSeqlens() != nullptr, "seqQOffset is nullptr.");
        TLLM_CHECK_WITH_INFO(p.getFmhaSchedulerCounter() != nullptr, "fmha_tile_counter is nullptr.");
        if (mFP8GenerationMLA)
        {
            TLLM_CHECK_WITH_INFO(p.getQuantQBuffer() != nullptr, "quant_q_buf is nullptr.");
            TLLM_CHECK_WITH_INFO(p.getMlaBmm1Scale() != nullptr, "bmm1_scale is nullptr.");
            TLLM_CHECK_WITH_INFO(p.getMlaBmm2Scale() != nullptr, "bmm2_scale is nullptr.");
        }
    }
    if (p.fwd.enable_dsv4_epilogue_fusion)
    {
        TORCH_CHECK(
            p.fwd.dsv4_inv_rope_cos_sin_cache.has_value(), "DSv4 fused epilogue requires inverse-RoPE cos/sin cache.");
        TORCH_CHECK(p.fwd.output_sf.has_value(), "DSv4 fused epilogue requires fwd.output_sf.");
        auto const& cosSinCache = p.fwd.dsv4_inv_rope_cos_sin_cache.value();
        auto const& outputSfTensor = p.fwd.output_sf.value();
        TORCH_CHECK(cosSinCache.scalar_type() == torch::kFloat32, "DSv4 fused epilogue cos/sin cache must be float32.");
        TORCH_CHECK(
            p.output->scalar_type() == torch::kFloat8_e4m3fn, "DSv4 fused epilogue output must be float8_e4m3fn.");
        TORCH_CHECK(p.output->dim() == 3 && p.output->is_contiguous(),
            "DSv4 fused epilogue output must be contiguous [groups, tokens, K].");
        TORCH_CHECK(
            outputSfTensor.scalar_type() == torch::kFloat32, "DSv4 fused epilogue fwd.output_sf must be float32.");
        TORCH_CHECK(outputSfTensor.dim() == 3 && outputSfTensor.is_contiguous(),
            "DSv4 fused epilogue fwd.output_sf must be contiguous [groups, K/128, padded_tokens].");
        TORCH_CHECK(p.output->size(1) >= p.num_tokens, "DSv4 fused epilogue output token dimension is too small.");
        TORCH_CHECK(mMLAParams.v_head_dim > 0 && mMLAParams.v_head_dim % 128 == 0,
            "DSv4 fused epilogue requires v_head_dim to be a positive multiple of 128.");
        TORCH_CHECK(
            outputSfTensor.size(2) >= p.num_tokens, "DSv4 fused epilogue fwd.output_sf token dimension is too small.");
    }

    if (p.fwd.attention_sinks.has_value())
    {
        TORCH_CHECK(p.fwd.attention_sinks.value().scalar_type() == torch::kFloat32,
            "Expected attention_sinks to have float dtype");
    }

    // The KV-cache scales travel together and only mean something for a quantized cache.
    if (!quant_mode.hasKvCacheQuant() || !p.fwd.kv_scale_orig_quant.has_value()
        || !p.fwd.kv_scale_quant_orig.has_value())
    {
        p.fwd.kv_scale_orig_quant.reset();
        p.fwd.kv_scale_quant_orig.reset();
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
        TORCH_CHECK(tensor.size(0) >= p.batch_size + 1, name, " must have at least num_seqs + 1 elements.");
    };
    if (!isGen)
    {
        checkCuSeqlens(p.fwd.cu_q_seqlens, "cu_q_seqlens");
        checkCuSeqlens(p.fwd.cu_kv_seqlens, "cu_kv_seqlens");
    }
}

tensorrt_llm::common::op::AttentionContextWorkspaceLayout MLA::contextWorkspaceLayout(FmhaParams const& p) const
{
    auto const batch_size = static_cast<size_t>(p.batch_size);
    auto const max_num_tokens = static_cast<size_t>(p.num_tokens);
    auto const total_kv_len = static_cast<size_t>(p.total_kv_len);

    size_t const cu_seqlens_size = sizeof(int) * (batch_size + 1);
    size_t const rotary_inv_freq_size = sizeof(float) * batch_size * rotary_embedding_dim / 2;

    int dim_q_per_head = (mMLAParams.qk_rope_head_dim + mMLAParams.qk_nope_head_dim);
    int dim_k_per_head = (mMLAParams.qk_rope_head_dim + mMLAParams.qk_nope_head_dim);
    int dim_v_per_head = (mMLAParams.v_head_dim);
    if (useSparseMLA())
    {
        dim_q_per_head = mMLAParams.kv_lora_rank + mMLAParams.qk_rope_head_dim;
        dim_k_per_head = mMLAParams.kv_lora_rank + mMLAParams.qk_rope_head_dim;
        dim_v_per_head
            = mMLAParams.rope_append ? mMLAParams.kv_lora_rank : mMLAParams.kv_lora_rank + mMLAParams.qk_rope_head_dim;
    }

    // Total dimension per token across all heads for Q, K, and V components respectively
    int const total_q_dim_all_heads = mNumAttnHeads * dim_q_per_head;
    int const total_k_dim_all_heads
        = mNumAttnHeads * dim_k_per_head; // Assuming effective num_kv_heads = head_num for layout
    int const total_v_dim_all_heads
        = mNumAttnHeads * dim_v_per_head; // Assuming effective num_kv_heads = head_num for layout

    // Separate fp8 q/k/v buffer size for fp8 context MLA
    size_t fp8_q_buf_size = 0;
    size_t fp8_k_buf_size = 0;
    size_t fp8_v_buf_size = 0;
    if (mFP8ContextMLA)
    {
        fp8_q_buf_size = max_num_tokens * static_cast<size_t>(total_q_dim_all_heads);

        if (useSparseMLA())
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
            size_t const kv_buf_tokens
                = std::max(total_kv_len, static_cast<size_t>(p.fwd.chunked_prefill_buffer_batch_size) * max_num_tokens);
            fp8_k_buf_size = kv_buf_tokens * static_cast<size_t>(total_k_dim_all_heads);
            fp8_v_buf_size = kv_buf_tokens * static_cast<size_t>(total_v_dim_all_heads);
            TLLM_CHECK(static_cast<size_t>(total_k_dim_all_heads + total_v_dim_all_heads)
                == contextMlaWorkspaceBytesPerToken(mNumAttnHeads, mMLAParams.qk_rope_head_dim,
                    mMLAParams.qk_nope_head_dim, mMLAParams.v_head_dim, mFP8ContextMLA,
                    /*separateQAndKvInput=*/true, useSparseMLA()));
        }
    }

    // Each token holds (batch_idx, token_idx_in_seq) int2.
    size_t const tokens_info_size = sizeof(int2) * max_num_tokens;
    size_t const fmha_scheduler_counter = sizeof(uint32_t);
    size_t const fmha_bmm1_scale_size = mFP8ContextMLA ? sizeof(float) * 2 : 0;
    size_t const fmha_bmm2_scale_size = mFP8ContextMLA ? sizeof(float) : 0;
    size_t const fmha_multi_ctas_kv_scratch_size = useSparseMLA()
        ? TrtllmGenFmha<MLA>::getMultiCtasKvScratchSize(mMLAParams.kv_lora_rank, mMultiProcessorCount)
        : 0;

    AttentionContextWorkspaceSizes workspaceSizes{};
    workspaceSizes.cuQSeqlens = cu_seqlens_size;
    workspaceSizes.cuKvSeqlens = cu_seqlens_size;
    workspaceSizes.cuMaskRows = cu_seqlens_size;
    workspaceSizes.rotaryInvFreq = rotary_inv_freq_size;
    workspaceSizes.fp8QBuf = fp8_q_buf_size;
    workspaceSizes.fp8KBuf = fp8_k_buf_size;
    workspaceSizes.fp8VBuf = fp8_v_buf_size;
    workspaceSizes.tokensInfo = tokens_info_size;
    workspaceSizes.fmhaTileCounter = fmha_scheduler_counter;
    workspaceSizes.fmhaBmm1Scale = fmha_bmm1_scale_size;
    workspaceSizes.fmhaBmm2Scale = fmha_bmm2_scale_size;
    workspaceSizes.fmhaMultiCtasKvScratch = fmha_multi_ctas_kv_scratch_size;
    return AttentionWorkspaceManager::buildContextLayout(workspaceSizes);
}

size_t MLA::getWorkspaceSizeForContext(FmhaParams const& p) const
{
    return p.num_tokens == 0 ? 0 : contextWorkspaceLayout(p).totalSize;
}

size_t MLA::getGenerationWorkspaceSize(FmhaParams const& p, size_t kernelWorkspaceSize) const noexcept
{
    // Sized per sequence (max_num_requests * beam_width) from the batch bound, not from this
    // call's rows, so a context call sizing a mixed batch and the following generation call
    // compute the same number.
    auto const max_num_seq = static_cast<int32_t>(p.getMaxNumSequences());
    auto const max_num_tokens = static_cast<int32_t>(p.num_gen_tokens);
    if (max_num_tokens == 0)
    {
        return 0;
    }

    size_t const cu_seqlens_size = sizeof(int) * (max_num_seq + 1);
    size_t const fmha_scheduler_counter = sizeof(uint32_t);
    size_t const fmha_multi_ctas_kv_scratch_size
        = TrtllmGenFmha<MLA>::getMultiCtasKvScratchSize(mMLAParams.kv_lora_rank, mMultiProcessorCount);

    int const NUM_BUFFERS = 5;
    size_t workspaces[NUM_BUFFERS];
    workspaces[0] = mIsGenerationMLA ? 0 : cu_seqlens_size; // cu_q_len
    workspaces[1] = mIsGenerationMLA ? 0 : cu_seqlens_size; // cu_kv_len
    workspaces[2] = mIsGenerationMLA ? 0 : fmha_scheduler_counter;
    workspaces[3] = fmha_multi_ctas_kv_scratch_size;
    workspaces[4] = kernelWorkspaceSize;
    size_t const mla_workspace_size = tc::calculateTotalWorkspaceSize(workspaces, NUM_BUFFERS);
    return mla_workspace_size;
}

template <typename T>
void MLA::enqueueFusedContext(
    FmhaParams const& p, cudaStream_t stream, std::function<void(MHARunnerParams const&)> const& launch) const
{
    float const q_scaling = static_cast<float>(this->q_scaling);

    MlaParams<T> mla{};
    mla.latent_cache = p.getLatentCache<T>();
    if (use_sparse_attention)
    {
        // Fused FP8-Q path: the RoPE kernel appends rope FP8 in place into the caller's
        // fwd.quant_q_buffer and the standalone quantize is skipped. A missing q_pe on this
        // path means the upstream q_b fusion has already written the complete rotated FP8 Q.
        mla.quant_scale_qkv = p.getQuantScaleQkv();
        mla.fuse_q_fp8_in_rope = (p.fwd.quant_q_buffer.has_value() && p.fwd.quant_scale_qkv.has_value());
        mla.q_rope_applied = mla.fuse_q_fp8_in_rope && !p.fwd.q_pe.has_value();
        if (p.fwd.q_pe.has_value())
        {
            mla.q_pe = p.getQPe<T>();
            mla.q_pe_ld = p.fwd.q_pe->strides()[1];
            mla.q_pe_stride = p.fwd.q_pe->strides()[0];
        }
        if (p.fwd.kv_norm_weight)
        {
            mla.latent_row_stride = static_cast<int>(p.fwd.latent_cache->stride(0));
            mla.kv_norm_weight = static_cast<void const*>(p.fwd.kv_norm_weight->data_ptr());
            mla.kv_norm_eps = static_cast<float>(p.fwd.kv_norm_eps);
            mla.fuse_kv_norm_in_rope = true;
        }
    }
    else
    {
        mla.k_buf = p.getK<T>();
        mla.v_buf = p.getV<T>();
        mla.fp8_kv_input = p.key_input.has_value() && p.key_input->scalar_type() == torch::kFloat8_e4m3fn;
        // `quant_scale_qkv` is a presence flag for caller-quantized FP8 Q; its value
        // is not read on this branch. The FP8 context FMHA dequantizes Q with the
        // kv cache dequant scale (see the context scale comment below), so the
        // caller must have quantized `quant_q_buffer` with kv_scale_orig_quant.
        mla.fp8_q_input = mla.fp8_kv_input && p.fwd.quant_q_buffer.has_value() && p.fwd.quant_scale_qkv.has_value();
        mla.helix_position_offsets = p.getHelixPositionOffsets();
        mla.helix_is_inactive_rank = p.getHelixIsInactiveRank();
    }
    mla.q_buf = p.getQueryInput<T>();
    mla.context_buf = static_cast<T*>(p.getOutput());
    // MlaParams has no default member initializers for these two, and the MLA RoPE
    // kernels use max_input_seq_len as a grid dimension and dereference cache_seq_lens.
    mla.cache_seq_lens = p.getSequenceLengths();
    mla.max_input_seq_len = p.input_seq_length;
    mla.cos_sin_cache = isRoPE() ? p.getRotaryCosSin() : nullptr;
    if (p.fwd.enable_dsv4_epilogue_fusion)
    {
        mla.dsv4_epilogue_fusion.enabled = true;
        mla.dsv4_epilogue_fusion.cos_sin_cache = p.getDsv4InvRopeCosSinCache();
        mla.dsv4_epilogue_fusion.scale_buf_m = static_cast<int32_t>(p.fwd.output_sf->size(2));
    }
    mla.batch_size = p.batch_size;
    mla.acc_q_len = p.num_tokens;
    mla.head_num = num_heads;
    mla.meta = mMLAParams;
    mla.workspace = p.getWorkspace();

    auto const& pagedKv = feature<torch_ext::PagedKVCacheFeature>();
    auto const& pools = pagedKv.poolPointers();
    auto const sizePerToken = mNumAttnKVHeads * mHeadSize * kvCacheElemSizeInBits(quant_mode, sizeof(T)) / 8 /*bits*/;
    auto const buffers = buildKvCacheBuffers<KVBlockArray>(p.batch_size, pagedKv.maxBlocksPerSeq(), tokens_per_block,
        sizePerToken, p.cyclic_attention_window_size, p.cyclic_attention_window_size, 0,
        p.meta.effective_beam_width > 1, pools.primaryPoolPtr, pools.secondaryPoolPtr, pools.primaryBlockScalePoolPtr,
        pools.secondaryBlockScalePoolPtr, pagedKv.blockOffsets(), quant_mode.hasFp4KvCache(),
        p.max_attention_window_size, nullptr);
    auto const& kv_cache_buffer = buffers.kvCacheBuffer;
    auto const& kv_scale_cache_buffer = buffers.kvScaleCacheBuffer;

    auto const workspaceLayout = contextWorkspaceLayout(p);
    auto const workspaceViews = AttentionWorkspaceManager::materializeContext<T>(p.getWorkspace(), workspaceLayout);

    auto* fp8QBuf = workspaceViews.fp8QBuf;
    // Fused FP8-Q path: caller pre-fills the nope segment of `quant_q_buf`;
    // route the context-MLA Q pointer to it so the fused RoPE kernel appends
    // rope FP8 in place and the FMHA Q load reads the merged [nope|rope] buffer.
    if (p.fwd.quant_q_buffer.has_value() && p.fwd.quant_scale_qkv.has_value() && p.getQuantQBuffer() != nullptr)
    {
        fp8QBuf = reinterpret_cast<__nv_fp8_e4m3*>(p.getQuantQBuffer());
    }

    // build cu_seqlens, tokens info and the FP8 scales
    BuildDecoderInfoParams<T> decoder_params{};
    int32_t const* precomputedCuQSeqlens = p.getCuQSeqlens();
    int32_t const* precomputedCuKvSeqlens = p.getCuKvSeqlens() != nullptr ? p.getCuKvSeqlens() : p.getCuQSeqlens();
    decoder_params.seqQOffsets = workspaceViews.cuQSeqlens;
    decoder_params.seqKVOffsets = workspaceViews.cuKvSeqlens;
    decoder_params.precomputedSeqQOffsets = precomputedCuQSeqlens;
    decoder_params.precomputedSeqKVOffsets = precomputedCuKvSeqlens;
    decoder_params.seqCpPartialOffsets = nullptr;
    decoder_params.cpSize = 1;
    decoder_params.packedMaskRowOffsets = workspaceViews.cuMaskRows;
    decoder_params.tokensInfo = workspaceViews.tokensInfo;
    decoder_params.seqQLengths = p.getContextLengths();
    decoder_params.seqKVLengths = p.getSequenceLengths();
    decoder_params.batchSize = p.batch_size;
    decoder_params.maxQSeqLength = p.input_seq_length;
    decoder_params.attentionWindowSize = p.cyclic_attention_window_size;
    decoder_params.numTokens = p.num_tokens;
    decoder_params.removePadding = remove_padding;
    decoder_params.attentionMaskType = mask_type;
    decoder_params.fmhaTileCounter = workspaceViews.fmhaTileCounter;
    decoder_params.quantScaleO = getOutQuantScale(p);
    decoder_params.dequantScaleQkv = p.getKvScaleQuantOrig();
    decoder_params.separateQkvScales = quant_mode.hasFp4KvCache();
    decoder_params.fmhaHostBmm1Scale = 1.0f / (sqrtf(getHeadSize() * 1.0f) * q_scaling);
    decoder_params.fmhaBmm1Scale = workspaceViews.fmhaBmm1Scale;
    decoder_params.fmhaBmm2Scale = workspaceViews.fmhaBmm2Scale;
    decoder_params.rotaryEmbeddingInvFreq = workspaceViews.rotaryInvFreq;
    applyFeatures(decoder_params);

    invokeBuildDecoderInfo(decoder_params, stream);
    sync_check_cuda_error(stream);

    int32_t const* contextCuQSeqlens
        = precomputedCuQSeqlens != nullptr ? precomputedCuQSeqlens : workspaceViews.cuQSeqlens;
    int32_t const* contextCuKvSeqlens
        = precomputedCuKvSeqlens != nullptr ? precomputedCuKvSeqlens : workspaceViews.cuKvSeqlens;

    KvCacheDataType const cache_type = cacheTypeFromQuantMode(quant_mode);

    mla.cache_type = cache_type;
    mla.cu_q_seqlens = const_cast<int*>(contextCuQSeqlens);
    mla.cu_kv_seqlens = const_cast<int*>(contextCuKvSeqlens);
    // Set BMM scales for FP8 context computation
    mla.bmm1_scale = workspaceViews.fmhaBmm1Scale;
    mla.bmm2_scale = workspaceViews.fmhaBmm2Scale;
    mla.quant_q_buf = mFP8ContextMLA ? fp8QBuf : nullptr;
    mla.quant_k_buf = mFP8ContextMLA ? workspaceViews.fp8KBuf : nullptr;
    mla.quant_v_buf = mFP8ContextMLA ? workspaceViews.fp8VBuf : nullptr;
    // Separate V has shape [tokens, numKvHeads * headDimV] with unit feature stride; context FMHA
    // and its FP8/NVFP4 input quantizer read it with its physical token stride.
    int64_t vStrideInBytes = 0;
    if (!use_sparse_attention && p.value_input.has_value())
    {
        auto const& v = p.value_input.value();
        TORCH_CHECK(v.stride(0) > 0, "Separate-V context FMHA requires a positive V token stride");
        vStrideInBytes = v.stride(0) * v.element_size();
        constexpr int64_t kTmaAlignmentBytes = 16;
        TORCH_CHECK(vStrideInBytes % kTmaAlignmentBytes == 0,
            "Separate-V context FMHA requires a 16-byte aligned V token stride");
        TORCH_CHECK(reinterpret_cast<std::uintptr_t>(v.data_ptr()) % kTmaAlignmentBytes == 0,
            "Separate-V context FMHA requires a 16-byte aligned V pointer");
        TLLM_CHECK_WITH_INFO(vStrideInBytes % static_cast<int64_t>(sizeof(T)) == 0,
            "MLA Context: V token stride must be a positive multiple of the element size");
        mla.v_token_stride_elements = vStrideInBytes / static_cast<int64_t>(sizeof(T));
    }
    // Set additional scales for context phase
    mla.quant_scale_o = getOutQuantScale(p);
    mla.quant_scale_q = p.getKvScaleOrigQuant();
    mla.quant_scale_kv = p.getKvScaleOrigQuant();
    mla.dequant_scale_q = p.getKvScaleQuantOrig();
    mla.dequant_scale_kv = cache_type == KvCacheDataType::NVFP4 ? nullptr : p.getKvScaleQuantOrig();
    // The FP8 context FMHA dequantizes Q / K / V with the kv cache dequant
    // scale: quantizeCopyInputToFp8Kernel folds dequant_scale_q / dequant_scale_kv
    // into bmm1_scale / bmm2_scale, and the FMHA reads those. Everything that
    // reaches it must have been quantized with the cache quant scale -- the
    // copies the op makes of BF16 Q / K / V, and any caller-quantized FP8 k / v
    // or `quant_q_buffer`. On the non-sparse branch `quant_scale_qkv` only flags
    // caller-quantized FP8 Q (its value is not read), so the copies take
    // `quant_scale_kv` here. NVFP4 keeps a unit scale: its fp8 context buffers are
    // an uncalibrated scratch quantization (nullptr dequant above). Sparse keeps
    // the caller's scale set above, which its fused RoPE kernel quantizes Q with.
    if (!use_sparse_attention && cache_type == KvCacheDataType::FP8)
    {
        mla.quant_scale_qkv = mla.quant_scale_kv;
    }
    mla.host_bmm1_scale = 1 / (q_scaling * sqrt((float) (mMLAParams.qk_nope_head_dim + mMLAParams.qk_rope_head_dim)));
    // The sparse MLA is in the absorption mode for the context phase.
    mla.absorption_mode = useSparseMLA();
    // Fused FP8-Q-quant: RoPE kernel writes FP8 rope into `quant_q_buf`,
    // so we skip the standalone invokeMLAContextFp8Quantize call below.
    bool const useFusedQFp8 = mla.fuse_q_fp8_in_rope && mFP8ContextMLA && mla.absorption_mode
        && cache_type == KvCacheDataType::FP8 && mla.quant_q_buf != nullptr && mla.quant_scale_qkv != nullptr;
    TLLM_CHECK_WITH_INFO(cache_type != KvCacheDataType::NVFP4 || mla.latent_cache == nullptr,
        "NVFP4 sparse MLA context must append its latent cache before launching attention");
    if (mla.latent_cache != nullptr)
    {
        invokeMLARopeContext<T, KVBlockArray>(mla, kv_cache_buffer, stream);
    }
    if (mFP8ContextMLA && !useFusedQFp8)
    {
        invokeMLAContextFp8Quantize(mla, p.total_kv_len, stream);
    }
    sync_check_cuda_error(stream);

    // Separate QKV input layout:
    //    - q_ptr: [B, S, H, D], which supports variable sequence length
    //    - k_ptr: [B, S, H_kv, D], which supports variable sequence length
    //    - v_ptr: [B, S, H_kv, D_v], which supports variable sequence length
    //    - cu_q_seqlens: the cumulative query sequence lengths, needed for variable sequence length.
    //    - cu_kv_seqlens: the cumulative kv sequence lengths, needed for variable sequence length.
    //    - total_kv_len: the total kv sequence length, needed for variable sequence length.
    // Sparse MLA reads K/V from the paged latent cache instead (Q_PAGED_KV).
    MHARunnerParams fmhaParams{};
    fmhaParams.b = p.batch_size;
    fmhaParams.qSeqLen = p.input_seq_length;
    fmhaParams.kvSeqLen = p.max_past_kv_length;
    // Disable sliding window attention when it is not needed.
    fmhaParams.slidingWindowSize = p.cyclic_attention_window_size;
    fmhaParams.totalQSeqLen = p.num_tokens;
    fmhaParams.totalKvSeqLen = p.total_kv_len;
    if (mFP8ContextMLA)
    {
        TLLM_CHECK_WITH_INFO(fp8QBuf != nullptr, "FP8 q buffer is required for fp8 context MLA");
        // In sparse MLA (absorption mode), K and V are stored in KV cache, not as separate FP8 buffers
        TLLM_CHECK_WITH_INFO(useSparseMLA() || workspaceViews.fp8KBuf != nullptr,
            "FP8 k buffer is required for fp8 context MLA in non-sparse mode");
        TLLM_CHECK_WITH_INFO(useSparseMLA() || workspaceViews.fp8VBuf != nullptr,
            "FP8 v buffer is required for fp8 context MLA in non-sparse mode");

        fmhaParams.qPtr = reinterpret_cast<void const*>(fp8QBuf);
        if (useSparseMLA())
        {
            fmhaParams.kPtr = nullptr;
            fmhaParams.vPtr = nullptr;
        }
        else if (mla.fp8_kv_input)
        {
            fmhaParams.kPtr = reinterpret_cast<void const*>(mla.k_buf);
            fmhaParams.vPtr = reinterpret_cast<void const*>(mla.v_buf);
        }
        else
        {
            fmhaParams.kPtr = reinterpret_cast<void const*>(workspaceViews.fp8KBuf);
            fmhaParams.vPtr = reinterpret_cast<void const*>(workspaceViews.fp8VBuf);
        }
    }
    else
    {
        fmhaParams.qPtr = p.getQueryInput<T>();
        fmhaParams.kPtr = p.getK<T>();
        fmhaParams.vPtr = p.getV<T>();
        fmhaParams.vStrideInBytes = vStrideInBytes;
    }
    fmhaParams.kvPtr = nullptr;
    // Only use [totalLength, h / cpSize, Dh].
    fmhaParams.outputPtr = p.getOutput();
    if (mla.dsv4_epilogue_fusion.enabled)
    {
        fmhaParams.dsv4EpilogueFusion.enabled = true;
        fmhaParams.dsv4EpilogueFusion.cosSinCache = mla.dsv4_epilogue_fusion.cos_sin_cache;
        fmhaParams.dsv4EpilogueFusion.scaleBufM = mla.dsv4_epilogue_fusion.scale_buf_m;
    }
    fmhaParams.attentionSinksPtr = p.getAttentionSinks();
    fmhaParams.pagedKvCache = kv_cache_buffer;
    fmhaParams.pagedKvSfCache = kv_scale_cache_buffer;
    fmhaParams.cuQSeqLenPtr = contextCuQSeqlens;
    fmhaParams.kvSeqLenPtr = decoder_params.seqKVLengths;
    fmhaParams.cuKvSeqLenPtr = contextCuKvSeqlens;
    fmhaParams.cuMaskRowsPtr = workspaceViews.cuMaskRows;
    fmhaParams.tileCounterPtr = workspaceViews.fmhaTileCounter;
    fmhaParams.scaleBmm1Ptr = workspaceViews.fmhaBmm1Scale;
    fmhaParams.scaleBmm2Ptr = workspaceViews.fmhaBmm2Scale;
    fmhaParams.stream = stream;
    fmhaParams.forceFp32Acc = mFMHAForceFP32Acc;
    fmhaParams.skipCorrectionThreshold = mSkipCorrectionThreshold;
    fmhaParams.softmaxStatsPtr = p.getSoftmaxStatsTensor();
    fmhaParams.trtllmGenJITWarmup = p.meta.trtllm_gen_jit_warmup;
    fmhaParams.trtllmGenJITWarmupMaxNumRequests = p.meta.max_num_requests;
    fmhaParams.trtllmGenJITWarmupMaxSeqLenQ = p.meta.max_context_length;
    fmhaParams.trtllmGenJITWarmupMaxSeqLenKv = p.meta.max_seq_len;

    if (useSparseMLA())
    {
        // Sparse context reuses generation-style trtllm-gen kernels; provide the scratch pool
        // and per-CTA counter so the autotuner can select MultiCtasKv variants.
        fmhaParams.multiCtasKvScratchPtr = workspaceViews.fmhaMultiCtasKvScratch;
        fmhaParams.multiCtasKvCounterPtr = getMultiCtasKvCounter(p.getMaxNumSequences(), p.getQueryInput().device());
    }
    applyFeatures(fmhaParams);

    // Skip-softmax attention parameters
    fmhaParams.skipSoftmaxThresholdScaleFactor = p.fwd.sparse_runtime_params.threshold_scale_factor_prefill;
#ifdef SKIP_SOFTMAX_STAT
    fmhaParams.skipSoftmaxTotalBlocks = p.getSkipSoftmaxStat();
    fmhaParams.skipSoftmaxSkippedBlocks = p.getSkipSoftmaxStat() + 1;
#else
    if (tensorrt_llm::common::getEnvPrintSkipSoftmaxStat())
    {
        TLLM_THROW("To print skip softmax stat, please run build_wheel.py with -DSKIP_SOFTMAX_STAT");
    }
#endif

    if (p.attn.attention_chunk_size)
    {
        fmhaParams.chunkedAttentionSize = *p.attn.attention_chunk_size;
    }

    // Run the fmha kernel.
    launch(fmhaParams);
    sync_check_cuda_error(stream);
}

template void MLA::enqueueFusedContext<float>(
    FmhaParams const&, cudaStream_t, std::function<void(MHARunnerParams const&)> const&) const;
template void MLA::enqueueFusedContext<half>(
    FmhaParams const&, cudaStream_t, std::function<void(MHARunnerParams const&)> const&) const;
#ifdef ENABLE_BF16
template void MLA::enqueueFusedContext<__nv_bfloat16>(
    FmhaParams const&, cudaStream_t, std::function<void(MHARunnerParams const&)> const&) const;
#endif

template <typename T>
MlaParams<T> MLA::getGenerationParams(FmhaParams const& p) const
{
    MlaParams<T> params{};
    params.latent_cache = p.getLatentCache<T>();
    params.q_pe = p.getQPe<T>();
    params.q_pe_ld = p.fwd.q_pe->strides()[1];
    params.q_pe_stride = p.fwd.q_pe->strides()[0];
    params.seqQOffset = const_cast<int*>(p.getCuQSeqlens());
    params.cu_kv_seqlens = const_cast<int*>(p.getCuKvSeqlens());
    params.fmha_tile_counter = p.getFmhaSchedulerCounter();
    params.bmm1_scale = p.getMlaBmm1Scale();
    params.bmm2_scale = p.getMlaBmm2Scale();
    params.quant_q_buf = p.getQuantQBuffer();
    params.quant_scale_qkv = p.getQuantScaleQkv();
    params.fuse_q_fp8_in_rope = (p.fwd.quant_q_buffer.has_value() && p.fwd.quant_scale_qkv.has_value());
    params.q_buf = p.getQueryInput<T>();
    params.context_buf = static_cast<T*>(p.getOutput());
    // MlaParams has no default member initializers for these two, and the MLA RoPE
    // kernels use max_input_seq_len as a grid dimension and dereference cache_seq_lens.
    params.cache_seq_lens = p.getSequenceLengths();
    params.max_input_seq_len = p.input_seq_length;
    params.cos_sin_cache = isRoPE() ? p.getRotaryCosSin() : nullptr;
    if (p.fwd.enable_dsv4_epilogue_fusion)
    {
        params.dsv4_epilogue_fusion.enabled = true;
        params.dsv4_epilogue_fusion.cos_sin_cache = p.getDsv4InvRopeCosSinCache();
        params.dsv4_epilogue_fusion.scale_buf_m = static_cast<int32_t>(p.fwd.output_sf->size(2));
    }
    params.batch_size = p.batch_size;
    params.acc_q_len = p.num_tokens;
    params.head_num = num_heads;
    params.meta = mMLAParams;
    params.workspace = p.getWorkspace();

    params.cache_type = (mFP8GenerationMLA ? KvCacheDataType::FP8 : KvCacheDataType::BASE);
    params.quant_scale_o = getOutQuantScale(p);
    params.quant_scale_q = p.getKvScaleOrigQuant();
    params.quant_scale_kv = p.getKvScaleOrigQuant();
    params.dequant_scale_q = p.getKvScaleQuantOrig();
    params.dequant_scale_kv = p.getKvScaleQuantOrig();
    params.host_bmm1_scale
        = 1 / (q_scaling * sqrt((float) (mMLAParams.qk_nope_head_dim + mMLAParams.qk_rope_head_dim)));

    return params;
}

template <typename T>
KVBlockArray MLA::getGenerationKvCacheBuffer(FmhaParams const& p) const
{
    int const num_kv_heads = 1;
    int const head_size = mMLAParams.kv_lora_rank + mMLAParams.qk_rope_head_dim;
    int32_t const batch_beam = p.meta.effective_beam_width * p.num_requests;
    // The element size of the KV cache.
    auto const elemSize = mFP8GenerationMLA ? sizeof(__nv_fp8_e4m3) : sizeof(T);
    auto const sizePerToken = num_kv_heads * head_size * elemSize;
    auto const& pagedKv = feature<torch_ext::PagedKVCacheFeature>();
    return KVBlockArray(batch_beam, pagedKv.maxBlocksPerSeq(), tokens_per_block, sizePerToken,
        p.cyclic_attention_window_size, p.cyclic_attention_window_size, 0, p.meta.effective_beam_width > 1,
        pagedKv.poolPointers().primaryPoolPtr, pagedKv.poolPointers().secondaryPoolPtr, pagedKv.blockOffsets());
}

#define INSTANTIATE_MLA_GENERATION_HELPERS(T)                                                                          \
    template MlaParams<T> MLA::getGenerationParams<T>(FmhaParams const&) const;                                        \
    template KVBlockArray MLA::getGenerationKvCacheBuffer<T>(FmhaParams const&) const;
INSTANTIATE_MLA_GENERATION_HELPERS(float)
INSTANTIATE_MLA_GENERATION_HELPERS(half)
#ifdef ENABLE_BF16
INSTANTIATE_MLA_GENERATION_HELPERS(__nv_bfloat16)
#endif
#undef INSTANTIATE_MLA_GENERATION_HELPERS

void MLA::enqueueContext(FmhaParams& p, PhasedFmha<MLA>& fmha)
{
    auto const stream = at::cuda::getCurrentCUDAStream(p.getQueryInput().get_device());
    fmha.runContext(*this, p, stream);
    sync_check_cuda_error(stream);
}

void MLA::enqueueGeneration(FmhaParams& p, PhasedFmha<MLA>& fmha)
{
    auto const stream = at::cuda::getCurrentCUDAStream(p.getQueryInput().get_device());
    fmha.runGeneration(*this, p, stream);
    {
        std::string const afterGenStr = "mla gen attention at layer " + std::to_string(p.attn.layer_idx);
        TLLM_CHECK_DEBUG_WITH_INFO(tensorrt_llm::runtime::utils::tensorHasInvalid(p.num_tokens, num_heads * head_size,
                                       p.getType(), p.getOutput(), stream, afterGenStr)
                == false,
            "Found invalid number (NaN or Inf) in " + afterGenStr);
    }
    sync_check_cuda_error(stream);
}

std::string MLA::toString() const
{
    std::stringstream ss;
    ss << std::boolalpha;
#define TRTLLM_DUMP_MEMBER(M) ss << #M ": " << M << "\n";
    TRTLLM_DUMP_MEMBER(mNumKVHeads)
    TRTLLM_DUMP_MEMBER(mNumAttnHeads)
    TRTLLM_DUMP_MEMBER(mNumAttnKVHeads)
    TRTLLM_DUMP_MEMBER(mHeadSize)
    TRTLLM_DUMP_MEMBER(mFMHAForceFP32Acc)
    TRTLLM_DUMP_MEMBER(mMultiBlockMode)
    TRTLLM_DUMP_MEMBER(mFP8ContextMLA)
    TRTLLM_DUMP_MEMBER(mFP8GenerationMLA)
    TRTLLM_DUMP_MEMBER(mIsGenerationMLA)
    TRTLLM_DUMP_MEMBER(mUseNvfp4MlaKvCache)
    TRTLLM_DUMP_MEMBER(mSkipCorrectionThreshold)
    TRTLLM_DUMP_MEMBER(mSM)
    TRTLLM_DUMP_MEMBER(mUseTllmGen)
    TRTLLM_DUMP_MEMBER(mMultiProcessorCount)
    TRTLLM_DUMP_MEMBER(mMaxSharedMemoryPerBlockOptin)
#undef TRTLLM_DUMP_MEMBER
    return ss.str();
}

AttentionBase* AttentionBase::getCached(AttentionBase const& key)
{
    // Per thread, like the Python-side parameter cache: no lock, and each thread builds
    // its own runners. Entries live until the thread exits.
    static thread_local std::unordered_map<AttentionBase, std::unique_ptr<AttentionBase>> cache;

    if (auto const it = cache.find(key); it != cache.end())
    {
        return it->second.get();
    }

    TLLM_LOG_DEBUG("Attention op not cached, building one for key: %s", key.toString().c_str());
    std::unique_ptr<AttentionBase> op = key.is_mla_enable ? std::unique_ptr<AttentionBase>(std::make_unique<MLA>(key))
                                                          : std::unique_ptr<AttentionBase>(std::make_unique<MHA>(key));
    auto* const result = op.get();
    cache.emplace(key, std::move(op));
    return result;
}

AttentionBase AttentionBase::getCacheKey(FmhaParams const& p)
{
    AttentionBase c;
    auto const& attn = p.attn;
    auto const& meta = p.meta;
    auto const& fwd = p.fwd;
    auto const& sparse = fwd.sparse_runtime_params;
    auto const nonEmpty = [](std::optional<torch::Tensor> const& t) { return t.has_value() && t->numel() > 0; };

    TORCH_CHECK(p.qkv_input.has_value() || p.query_input.has_value(), "FmhaParams requires qkv_input or query_input.");
    TORCH_CHECK(p.output.has_value(), "FmhaParams requires output.");
    TORCH_CHECK(p.sequence_lengths.has_value(), "FmhaParams requires sequence_lengths.");

    c.device = p.getQueryInput().get_device();
    c.type = p.getType();
    c.is_fp8_out = p.isFp8Out();
    c.is_fp4_out = p.isFp4Out();
    c.use_kv_cache = p.hasKvCache();
    c.cross_attention = p.is_cross;
    c.tokens_per_block = p.tokens_per_block;
    c.mask_type = static_cast<AttentionMaskType>(fwd.mask_type);
    TORCH_CHECK(fwd.variable_window_token_starts.has_value() == fwd.variable_window_token_ends.has_value(),
        "variable_window_token_starts and variable_window_token_ends must be provided together");
    c.use_variable_window = fwd.variable_window_token_starts.has_value() && p.seq_offset == 0;
    c.position_embedding_type = attn.position_embedding_type;
    c.num_heads = attn.num_heads;
    c.num_kv_heads = attn.num_kv_heads;
    c.head_size = attn.head_dim;
    c.q_scaling = attn.q_scaling;
    c.rotary_embedding_dim = attn.rope_params.dim;
    c.quant_mode = attn.quant_mode;
    c.uses_spcompress = attn.uses_spcompress;
    c.paged_context_fmha = meta.use_paged_context_fmha;
    c.is_spec_decoding_enabled = meta.is_spec_decoding_enabled;
    c.spec_decoding_target_max_gen_len
        = meta.max_total_draft_tokens.has_value() ? meta.max_total_draft_tokens.value() + 1 : 0;
    c.fuses_dsv4_inv_rope_fp8_quant = fwd.enable_dsv4_epilogue_fusion;

    bool const hasAttnIndices = nonEmpty(sparse.sparse_attn_indices);
    bool const hasPagedIndices = hasAttnIndices && nonEmpty(sparse.sparse_attn_offsets);
    c.use_sparse_attention = hasAttnIndices || nonEmpty(sparse.sparse_kv_indices);
    c.use_tllm_gen_sparse_attention = hasAttnIndices && !hasPagedIndices;

    c.is_mla_enable = attn.is_mla_enable;
    TORCH_CHECK(!c.use_variable_window || !c.is_mla_enable, "Variable-window attention does not support MLA.");
    c.q_lora_rank = attn.q_lora_rank.value_or(0);
    c.kv_lora_rank = attn.kv_lora_rank.value_or(0);
    c.qk_nope_head_dim = attn.qk_nope_head_dim.value_or(0);
    c.qk_rope_head_dim = attn.qk_rope_head_dim.value_or(0);
    c.v_head_dim = attn.v_head_dim;
    c.predicted_tokens_per_seq = attn.predicted_tokens_per_seq;
    c.mla_layer_num = p.getMlaLayerNum();
    c.rope_append = attn.rope_append.value_or(true);
    c.use_nvfp4_mla_kv_cache = attn.quant_mode.hasFp4KvCache() && c.use_tllm_gen_sparse_attention
        && !sparse.sparse_attn_kv_lens.has_value() && sparse.aux_kv_cache_pool_ptr.has_value();

    c.sage_attn_num_elts_per_blk_q = fwd.sage_attn_num_elts_per_blk_q;
    c.sage_attn_num_elts_per_blk_k = fwd.sage_attn_num_elts_per_blk_k;
    c.sage_attn_num_elts_per_blk_v = fwd.sage_attn_num_elts_per_blk_v;
    c.sage_attn_qk_int8 = fwd.sage_attn_qk_int8;
    c.sage_attn_smooth_k = fwd.sage_attn_smooth_k;
    c.skip_correction_threshold = attn.skip_correction_threshold;
    return c;
}

std::string AttentionBase::toString() const
{
    std::stringstream ss;
    ss << std::boolalpha;
#define TRTLLM_DUMP_FIELD(F) ss << #F "=" << F << " ";
    TRTLLM_DUMP_FIELD(device)
    ss << "type=" << static_cast<int>(type) << " ";
    TRTLLM_DUMP_FIELD(is_fp8_out)
    TRTLLM_DUMP_FIELD(is_fp4_out)
    TRTLLM_DUMP_FIELD(use_kv_cache)
    TRTLLM_DUMP_FIELD(cross_attention)
    TRTLLM_DUMP_FIELD(tokens_per_block)
    TRTLLM_DUMP_FIELD(use_variable_window)
    ss << "mask_type=" << static_cast<int>(mask_type) << " ";
    ss << "position_embedding_type=" << static_cast<int>(position_embedding_type) << " ";
    TRTLLM_DUMP_FIELD(num_heads)
    TRTLLM_DUMP_FIELD(num_kv_heads)
    TRTLLM_DUMP_FIELD(head_size)
    TRTLLM_DUMP_FIELD(q_scaling)
    TRTLLM_DUMP_FIELD(rotary_embedding_dim)
    ss << "quant_mode=" << quant_mode.value() << " ";
    TRTLLM_DUMP_FIELD(uses_spcompress)
    TRTLLM_DUMP_FIELD(paged_context_fmha)
    TRTLLM_DUMP_FIELD(is_spec_decoding_enabled)
    TRTLLM_DUMP_FIELD(spec_decoding_target_max_gen_len)
    TRTLLM_DUMP_FIELD(fuses_dsv4_inv_rope_fp8_quant)
    TRTLLM_DUMP_FIELD(use_sparse_attention)
    TRTLLM_DUMP_FIELD(use_tllm_gen_sparse_attention)
    TRTLLM_DUMP_FIELD(use_nvfp4_mla_kv_cache)
    TRTLLM_DUMP_FIELD(is_mla_enable)
    TRTLLM_DUMP_FIELD(q_lora_rank)
    TRTLLM_DUMP_FIELD(kv_lora_rank)
    TRTLLM_DUMP_FIELD(qk_nope_head_dim)
    TRTLLM_DUMP_FIELD(qk_rope_head_dim)
    TRTLLM_DUMP_FIELD(v_head_dim)
    TRTLLM_DUMP_FIELD(predicted_tokens_per_seq)
    TRTLLM_DUMP_FIELD(mla_layer_num)
    TRTLLM_DUMP_FIELD(rope_append)
    TRTLLM_DUMP_FIELD(sage_attn_num_elts_per_blk_q)
    TRTLLM_DUMP_FIELD(sage_attn_num_elts_per_blk_k)
    TRTLLM_DUMP_FIELD(sage_attn_num_elts_per_blk_v)
    TRTLLM_DUMP_FIELD(sage_attn_qk_int8)
    TRTLLM_DUMP_FIELD(sage_attn_smooth_k)
    TRTLLM_DUMP_FIELD(skip_correction_threshold)
#undef TRTLLM_DUMP_FIELD
    return ss.str();
}

bool attention_supports_nvfp4_output(int64_t const num_heads, int64_t const num_kv_heads, int64_t const head_size,
    std::optional<int64_t> const tokens_per_block, int64_t const mask_type, int64_t const quant_mode,
    bool const use_paged_context_fmha, bool is_mla_enable)
{
    // Only SM100 family supports NVFP4 output.
    if (!tensorrt_llm::common::isSM100Family())
    {
        return false;
    }

    // MLA is not supported.
    if (is_mla_enable)
    {
        return false;
    }

    int device;
    check_cuda_error(cudaGetDevice(&device));
    auto const key = std::make_tuple(device, num_heads, num_kv_heads, head_size, tokens_per_block.value_or(0),
        mask_type, quant_mode, use_paged_context_fmha);
    static thread_local std::map<std::decay_t<decltype(key)>, bool> cache;
    if (auto const it = cache.find(key); it != cache.end())
    {
        return it->second;
    }

    AttentionBase cfg;
    cfg.device = device;
    cfg.num_heads = num_heads;
    cfg.num_kv_heads = num_kv_heads;
    cfg.head_size = head_size;
    cfg.tokens_per_block = tokens_per_block.value_or(0);
    cfg.type = tensorrt_llm::DataType::kHALF;
    cfg.is_fp4_out = true;
    cfg.use_kv_cache = true;
    cfg.paged_context_fmha = use_paged_context_fmha;
    cfg.mask_type = static_cast<tensorrt_llm::kernels::AttentionMaskType>(int32_t(mask_type));
    cfg.q_scaling = 1.0;
    cfg.is_mla_enable = is_mla_enable;
    cfg.quant_mode = tensorrt_llm::common::QuantMode(uint32_t(quant_mode));

    MHA op{cfg};
    bool const supported = op.supportsNvFp4Output();
    cache.emplace(key, supported);
    return supported;
}

bool attention_supports_variable_window(c10::ScalarType const dtype, c10::ScalarType const output_dtype,
    int64_t const num_heads, int64_t const num_kv_heads, int64_t const head_size,
    std::optional<int64_t> const tokens_per_block, int64_t const quant_mode, int64_t const /*max_num_requests*/,
    int64_t const /*max_context_length*/, int64_t const /*max_seq_len*/,
    std::optional<int64_t> const /*attention_chunk_size*/, bool const use_kv_cache, bool const use_paged_context_fmha)
{
    auto const sm = common::getSMVersion();
    if ((sm != 100 && sm != 103)
        || (output_dtype != dtype && output_dtype != at::kFloat8_e4m3fn && output_dtype != at::kByte))
    {
        return false;
    }
    AttentionBase cfg;
    cfg.device = at::cuda::current_device();
    cfg.type = runtime::TorchUtils::dataType(dtype);
    cfg.num_heads = num_heads;
    cfg.num_kv_heads = num_kv_heads;
    cfg.head_size = head_size;
    cfg.is_fp8_out = output_dtype == at::kFloat8_e4m3fn;
    cfg.is_fp4_out = output_dtype == at::kByte;
    cfg.use_kv_cache = use_kv_cache;
    cfg.tokens_per_block = tokens_per_block.value_or(64);
    cfg.quant_mode = common::QuantMode(static_cast<uint32_t>(quant_mode));
    cfg.paged_context_fmha = use_paged_context_fmha;
    cfg.use_variable_window = true;
    // The phased runner's fixed configuration determines kernel availability.
    static thread_local std::unordered_map<AttentionBase, bool> cache;
    if (auto it = cache.find(cfg); it != cache.end())
    {
        return it->second;
    }
    bool supported = false;
    try
    {
        MHA op{cfg};
        supported = op.supportsVariableWindow();
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_DEBUG("Variable-window FMHA probe rejected the configuration: %s", e.what());
    }
    return cache.emplace(cfg, supported).first->second;
}

int64_t spec_dec_tree_custom_mask_capacity(
    int64_t const max_num_requests, int64_t const max_seq_len_q, int64_t const num_heads_q_per_kv)
{
    return tensorrt_llm::kernels::getCustomMaskCapacityUpperBound(static_cast<int32_t>(max_num_requests),
        static_cast<int32_t>(max_seq_len_q), static_cast<int32_t>(num_heads_q_per_kv));
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
        TORCH_CHECK(hostKvCachePoolPointers.dim() == 2 || hostKvCachePoolPointers.dim() == 3);
        // Non-FP4 layers use only the data pointers in a mixed-format pool table.
        auto const dataPointers
            = hostKvCachePoolPointers.dim() == 3 ? hostKvCachePoolPointers.select(2, 0) : hostKvCachePoolPointers;
        pointers.primaryPoolPtr = reinterpret_cast<void*>(
            reinterpret_cast<char*>(
                readHostTensor2D<int64_t>(dataPointers, poolIndex, 0, "host_kv_cache_pool_pointers"))
            + intraPoolOffset);
        pointers.secondaryPoolPtr = reinterpret_cast<void*>(
            reinterpret_cast<char*>(
                readHostTensor2D<int64_t>(dataPointers, poolIndex, 1, "host_kv_cache_pool_pointers"))
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

    int cacheElemBits = kvCacheElemSizeInBits(quantMode, elem_size);

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
    int const cacheElemBits = kvCacheElemSizeInBits(quantMode, inputElemSize);

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
    int device;
    cudaGetDevice(&device);
    int sm_cnt;
    cudaDeviceGetAttribute(&sm_cnt, cudaDevAttrMultiProcessorCount, device);
    int const num_sm_parts = tensorrt_llm::torch_ext::FlashMlaFmha::getNumSmParts(
        static_cast<int>(s_q), static_cast<int>(num_q_heads), static_cast<int>(num_kv_heads), sm_cnt);
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

namespace
{
template <typename T>
std::size_t hashCombine(std::size_t seed, T const& value)
{
    return seed ^ (std::hash<T>{}(value) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}
} // namespace

std::size_t std::hash<tensorrt_llm::torch_ext::AttentionBase>::operator()(
    tensorrt_llm::torch_ext::AttentionBase const& c) const noexcept
{
    std::size_t h = 0;
    for (auto const v :
        {static_cast<std::int64_t>(c.device), static_cast<std::int64_t>(c.type), std::int64_t{c.is_fp8_out},
            std::int64_t{c.is_fp4_out}, std::int64_t{c.use_kv_cache}, std::int64_t{c.cross_attention},
            c.tokens_per_block, static_cast<std::int64_t>(c.mask_type), std::int64_t{c.use_variable_window},
            static_cast<std::int64_t>(c.position_embedding_type), c.num_heads, c.num_kv_heads, c.head_size,
            c.rotary_embedding_dim, static_cast<std::int64_t>(c.quant_mode.value()), std::int64_t{c.uses_spcompress},
            std::int64_t{c.paged_context_fmha}, std::int64_t{c.is_spec_decoding_enabled},
            c.spec_decoding_target_max_gen_len, std::int64_t{c.fuses_dsv4_inv_rope_fp8_quant},
            std::int64_t{c.use_sparse_attention}, std::int64_t{c.use_tllm_gen_sparse_attention},
            std::int64_t{c.use_nvfp4_mla_kv_cache}, std::int64_t{c.is_mla_enable}, c.q_lora_rank, c.kv_lora_rank,
            c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim, c.predicted_tokens_per_seq, c.mla_layer_num,
            std::int64_t{c.rope_append}, c.sage_attn_num_elts_per_blk_q, c.sage_attn_num_elts_per_blk_k,
            c.sage_attn_num_elts_per_blk_v, std::int64_t{c.sage_attn_qk_int8}, std::int64_t{c.sage_attn_smooth_k}})
    {
        h = hashCombine(h, v);
    }
    h = hashCombine(h, c.q_scaling);
    h = hashCombine(h, c.skip_correction_threshold);
    return h;
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("attention_supports_nvfp4_output", &tensorrt_llm::torch_ext::attention_supports_nvfp4_output);
    m.def("attention_supports_variable_window", &tensorrt_llm::torch_ext::attention_supports_variable_window);
    m.def("spec_dec_tree_custom_mask_capacity", &tensorrt_llm::torch_ext::spec_dec_tree_custom_mask_capacity);
}
