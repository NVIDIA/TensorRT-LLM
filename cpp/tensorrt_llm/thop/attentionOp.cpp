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
#include <type_traits>
#include <unordered_set>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
using tensorrt_llm::common::op::AttentionOp;
using tensorrt_llm::common::op::AttentionWorkspaceManager;
using tensorrt_llm::common::op::hash;
using tensorrt_llm::runtime::RequestType;

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

enum class AttentionInputType : int8_t
{
    Mixed,
    ContextOnly,
    GenerationOnly,
};

class RunnerBase
{
public:
    int32_t beam_width;
    int32_t max_num_requests;
    int32_t attention_window_size;

    auto data() const
    {
        return std::make_tuple(beam_width, max_num_requests, attention_window_size);
    };

    virtual ~RunnerBase() = default;
    virtual void prepare(AttentionOp& op) const = 0;
    virtual int64_t getWorkspaceSize(AttentionOp const& op, int const num_tokens, int const max_attention_window_size,
        int const num_gen_tokens, int const max_blocks_per_sequence, int const ctx_total_kv_len = 0) const
        = 0;
    // typically, we use single qkv input, but for context MLA, we use separate qkv inputs
    virtual void run(AttentionOp& op, bool const is_context, int32_t const seq_offset, int32_t const num_seqs,
        int32_t const token_offset, int32_t const num_tokens, int32_t const predicted_tokens_per_seq,
        torch::Tensor workspace, torch::Tensor output, torch::optional<torch::Tensor> output_sf, torch::Tensor qkv_or_q,
        torch::optional<torch::Tensor> k, torch::optional<torch::Tensor> v, torch::Tensor sequence_length,
        torch::Tensor host_past_key_value_lengths, int32_t const total_kv_len, torch::Tensor context_lengths,
        torch::Tensor host_context_lengths, std::optional<int64_t> max_context_q_len_override,
        torch::optional<torch::Tensor> kv_cache_block_offsets,
        torch::optional<torch::Tensor> host_kv_cache_pool_pointers,
        torch::optional<torch::Tensor> host_kv_cache_pool_mapping, torch::optional<torch::Tensor> cache_indirection,
        torch::optional<torch::Tensor> kv_scale_orig_quant, torch::optional<torch::Tensor> kv_scale_quant_orig,
        torch::optional<torch::Tensor> out_scale, torch::optional<torch::Tensor> rotary_inv_freq,
        torch::optional<torch::Tensor> rotary_cos_sin, torch::optional<torch::Tensor> latent_cache,
        torch::optional<torch::Tensor> q_pe, torch::optional<torch::Tensor> block_ids_per_seq,
        torch::optional<torch::Tensor> mrope_rotary_cos_sin, torch::optional<torch::Tensor> mrope_position_deltas,
        std::optional<torch::Tensor> helix_position_offsets, std::optional<torch::Tensor> helix_is_inactive_rank,
        torch::optional<torch::Tensor> softmax_stats_tensor,
        std::optional<torch::Tensor> spec_decoding_generation_lengths,
        std::optional<torch::Tensor> spec_decoding_position_offsets_for_cpp,
        std::optional<torch::Tensor> spec_decoding_packed_mask,
        std::optional<torch::Tensor> spec_decoding_bl_tree_mask_offset,
        std::optional<torch::Tensor> spec_decoding_bl_tree_mask,
        std::optional<torch::Tensor> spec_bl_tree_first_sparse_mask_offset_kv,
        torch::optional<torch::Tensor> attention_sinks, torch::optional<torch::Tensor> sparse_kv_indices,
        torch::optional<torch::Tensor> sparse_kv_offsets, torch::optional<torch::Tensor> sparse_attn_indices,
        torch::optional<torch::Tensor> sparse_attn_offsets, int64_t const sparse_attn_indices_block_size,
        int32_t const num_sparse_topk, std::optional<torch::Tensor> sparse_mla_topk_lens,
        std::optional<torch::Tensor> cu_q_seqlens, std::optional<torch::Tensor> cu_kv_seqlens,
        std::optional<torch::Tensor> fmha_scheduler_counter, std::optional<torch::Tensor> mla_bmm1_scale,
        std::optional<torch::Tensor> mla_bmm2_scale, std::optional<torch::Tensor> quant_q_buffer,
        std::optional<torch::Tensor> flash_mla_tile_scheduler_metadata,
        std::optional<torch::Tensor> flash_mla_num_splits, bool trtllm_gen_jit_warmup,
        std::optional<int64_t> compressed_kv_cache_pool_ptr, bool const is_cross, std::optional<torch::Tensor> cross_kv,
        std::optional<torch::Tensor> relative_attention_bias,
        std::optional<torch::Tensor> quant_scale_qkv = std::nullopt,
        std::optional<torch::Tensor> dsv4_inv_rope_cos_sin_cache = std::nullopt,
        bool enable_dsv4_epilogue_fusion = false) const
        = 0;
};

template <typename T, typename AttentionOutT = T>
class Runner : public RunnerBase
{
public:
    void prepare(AttentionOp& op) const override
    {
        AttentionOp::EnqueueGenerationParams<T> enqueueParams;
        enqueueParams.max_attention_window_size = attention_window_size;
        enqueueParams.cyclic_attention_window_size = attention_window_size;
        enqueueParams.max_cyclic_attention_window_size = attention_window_size;
        enqueueParams.beam_width = beam_width;
        enqueueParams.num_requests = max_num_requests;

        op.prepareEnqueueGeneration<T, KVBlockArray>(enqueueParams);

        // Always reserve SemaphoreArray (for multi-block mode) as MMHA may enable multi-block mode when shared memory
        // is not enough.
        // The attention kernel might split the heads into multiple blocks, so we might need to reserve more semaphores.
        // Use mMultiProcessorCount as the lower-bound to make sure we reserve enough semaphores.
        op.reserveSemaphoreArray(std::max(op.mNumHeads * max_num_requests, op.getMultiProcessorCount()));
    }

    int64_t getWorkspaceSize(AttentionOp const& op, int const num_tokens, int const max_attention_window_size,
        int const num_gen_tokens, int const max_blocks_per_sequence, int const ctx_total_kv_len = 0) const override
    {
        size_t const context_workspace_size = op.getWorkspaceSizeForContext(
            op.mType, max_num_requests, op.mMaxContextLength, 0, num_tokens, ctx_total_kv_len);
        size_t const generation_workspace_size = op.getWorkspaceSizeForGeneration(
            op.mType, max_num_requests, max_attention_window_size, num_gen_tokens, max_blocks_per_sequence);

        return std::max(context_workspace_size, generation_workspace_size);
    }

    void run(AttentionOp& op, bool const is_context, int32_t const seq_offset, int32_t const num_seqs,
        int32_t const token_offset, int32_t const num_tokens, int32_t const predicted_tokens_per_seq,
        torch::Tensor workspace, torch::Tensor output, torch::optional<torch::Tensor> output_sf, torch::Tensor qkv_or_q,
        torch::optional<torch::Tensor> k, torch::optional<torch::Tensor> v, torch::Tensor sequence_length,
        torch::Tensor host_past_key_value_lengths, int32_t const total_kv_len, torch::Tensor context_lengths,
        torch::Tensor host_context_lengths, std::optional<int64_t> max_context_q_len_override,
        torch::optional<torch::Tensor> kv_cache_block_offsets,
        torch::optional<torch::Tensor> host_kv_cache_pool_pointers,
        torch::optional<torch::Tensor> host_kv_cache_pool_mapping, torch::optional<torch::Tensor> cache_indirection,
        torch::optional<torch::Tensor> kv_scale_orig_quant, torch::optional<torch::Tensor> kv_scale_quant_orig,
        torch::optional<torch::Tensor> out_scale, torch::optional<torch::Tensor> rotary_inv_freq,
        torch::optional<torch::Tensor> rotary_cos_sin, torch::optional<torch::Tensor> latent_cache,
        torch::optional<torch::Tensor> q_pe, torch::optional<torch::Tensor> block_ids_per_seq,
        torch::optional<torch::Tensor> mrope_rotary_cos_sin, torch::optional<torch::Tensor> mrope_position_deltas,
        std::optional<torch::Tensor> helix_position_offsets, std::optional<torch::Tensor> helix_is_inactive_rank,
        torch::optional<torch::Tensor> softmax_stats_tensor,
        std::optional<torch::Tensor> spec_decoding_generation_lengths,
        std::optional<torch::Tensor> spec_decoding_position_offsets_for_cpp,
        std::optional<torch::Tensor> spec_decoding_packed_mask,
        std::optional<torch::Tensor> spec_decoding_bl_tree_mask_offset,
        std::optional<torch::Tensor> spec_decoding_bl_tree_mask,
        std::optional<torch::Tensor> spec_bl_tree_first_sparse_mask_offset_kv,
        torch::optional<torch::Tensor> attention_sinks, torch::optional<torch::Tensor> sparse_kv_indices,
        torch::optional<torch::Tensor> sparse_kv_offsets, torch::optional<torch::Tensor> sparse_attn_indices,
        torch::optional<torch::Tensor> sparse_attn_offsets, int64_t const sparse_attn_indices_block_size,
        int32_t const num_sparse_topk, std::optional<torch::Tensor> sparse_mla_topk_lens,
        std::optional<torch::Tensor> cu_q_seqlens, std::optional<torch::Tensor> cu_kv_seqlens,
        std::optional<torch::Tensor> fmha_scheduler_counter, std::optional<torch::Tensor> mla_bmm1_scale,
        std::optional<torch::Tensor> mla_bmm2_scale, std::optional<torch::Tensor> quant_q_buffer,
        std::optional<torch::Tensor> flash_mla_tile_scheduler_metadata,
        std::optional<torch::Tensor> flash_mla_num_splits, bool trtllm_gen_jit_warmup,
        std::optional<int64_t> compressed_kv_cache_pool_ptr, bool const is_cross, std::optional<torch::Tensor> cross_kv,
        std::optional<torch::Tensor> relative_attention_bias, std::optional<torch::Tensor> quant_scale_qkv,
        std::optional<torch::Tensor> dsv4_inv_rope_cos_sin_cache, bool enable_dsv4_epilogue_fusion) const override
    {
        auto stream = at::cuda::getCurrentCUDAStream(qkv_or_q.get_device());
        T* attention_input = static_cast<T*>(qkv_or_q.slice(0, token_offset).data_ptr());
        T* k_ptr = nullptr;
        T* v_ptr = nullptr;
        AttentionOutT* context_buf = static_cast<AttentionOutT*>(output.slice(0, token_offset).data_ptr());
        TORCH_CHECK(!op.mFuseFp4Quant || output_sf.has_value());
        TORCH_CHECK(!enable_dsv4_epilogue_fusion || output_sf.has_value());
        void* context_buf_sf = (op.mFuseFp4Quant || enable_dsv4_epilogue_fusion) ? output_sf->data_ptr() : nullptr;

        // Rotary inv_freq, cos_sin cache to avoid re-computing.
        float const* rotary_inv_freq_ptr = nullptr;
        float2 const* rotary_cos_sin_ptr = nullptr;

        if (op.isRoPE())
        {
            if (rotary_inv_freq.has_value())
            {
                rotary_inv_freq_ptr = rotary_inv_freq.value().data_ptr<float>();
            }
            if (rotary_cos_sin.has_value())
            {
                rotary_cos_sin_ptr = static_cast<float2 const*>(rotary_cos_sin.value().data_ptr());
            }
        }

        void* workspace_ptr = workspace.data_ptr();
        [[maybe_unused]] MlaParams<T> mla_params;
        if (op.isMLAEnabled())
        {
            if (is_context && op.mUseSparseAttention)
            {
                if (latent_cache.has_value())
                {
                    mla_params.latent_cache = static_cast<T const*>(latent_cache->data_ptr());
                }
                else
                {
                    // kv cache reuse / chunked context cases, latent_cache is not used
                    mla_params.latent_cache = nullptr;
                }
                TORCH_CHECK(q_pe.has_value());
                TORCH_CHECK(q_pe->dim() == 3);
                TORCH_CHECK(q_pe->strides()[2] == 1);

                mla_params.q_pe = static_cast<T*>(q_pe->data_ptr());
                mla_params.q_pe_ld = q_pe->strides()[1];
                mla_params.q_pe_stride = q_pe->strides()[0];

                // Fused FP8-Q path: forward caller's quant_q_buffer / scale so
                // applyMLARopeAndAssignQKVKernelOptContext<kOutputFp8Q=true>
                // appends rope FP8 in place and the standalone quantize is
                // skipped. Without this wiring the sparse-MLA context branch
                // runs the legacy quantize over the bf16 placeholder q.
                mla_params.bmm1_scale = mla_bmm1_scale.has_value()
                    ? reinterpret_cast<float*>(mla_bmm1_scale.value().data_ptr())
                    : nullptr;
                mla_params.bmm2_scale = mla_bmm2_scale.has_value()
                    ? reinterpret_cast<float*>(mla_bmm2_scale.value().data_ptr())
                    : nullptr;
                mla_params.quant_q_buf
                    = quant_q_buffer.has_value() ? reinterpret_cast<void*>(quant_q_buffer.value().data_ptr()) : nullptr;
                mla_params.quant_scale_qkv = quant_scale_qkv.has_value()
                    ? reinterpret_cast<float const*>(quant_scale_qkv.value().data_ptr())
                    : nullptr;
                mla_params.fuse_q_fp8_in_rope = (quant_q_buffer.has_value() && quant_scale_qkv.has_value());
            }
            else if (is_context)
            {
                if (latent_cache.has_value())
                {
                    mla_params.latent_cache = static_cast<T const*>(latent_cache->data_ptr());
                }
                else
                {
                    // kv cache reuse / chunked context cases, latent_cache is not used
                    mla_params.latent_cache = nullptr;
                }
                TORCH_CHECK(k.has_value());
                TORCH_CHECK(v.has_value());
                TORCH_CHECK(k->dim() == 2);
                TORCH_CHECK(v->dim() == 2);
                TORCH_CHECK(k->strides()[1] == 1);
                TORCH_CHECK(v->strides()[1] == 1);

                k_ptr = static_cast<T*>(k->slice(0, token_offset).data_ptr());
                v_ptr = static_cast<T*>(v->slice(0, token_offset).data_ptr());
                mla_params.k_buf = k_ptr;
                mla_params.v_buf = v_ptr;

                // For generation, helix position is in ropeOp
                if (helix_position_offsets.has_value())
                {
                    mla_params.helix_position_offsets = helix_position_offsets->data_ptr<int32_t>();
                }
                if (helix_is_inactive_rank.has_value())
                {
                    mla_params.helix_is_inactive_rank = helix_is_inactive_rank->data_ptr<bool>();
                }
            }
            else
            {
                TORCH_CHECK(latent_cache.has_value());
                mla_params.latent_cache = static_cast<T const*>(latent_cache->data_ptr());
                TORCH_CHECK(q_pe.has_value());
                TORCH_CHECK(q_pe->dim() == 3);
                TORCH_CHECK(q_pe->strides()[2] == 1);

                mla_params.q_pe = static_cast<T*>(q_pe->data_ptr());
                mla_params.q_pe_ld = q_pe->strides()[1];
                mla_params.q_pe_stride = q_pe->strides()[0];

                mla_params.seqQOffset
                    = cu_q_seqlens.has_value() ? reinterpret_cast<int*>(cu_q_seqlens.value().data_ptr()) : nullptr;
                mla_params.cu_kv_seqlens
                    = cu_kv_seqlens.has_value() ? reinterpret_cast<int*>(cu_kv_seqlens.value().data_ptr()) : nullptr;
                mla_params.fmha_tile_counter = fmha_scheduler_counter.has_value()
                    ? reinterpret_cast<uint32_t*>(fmha_scheduler_counter.value().data_ptr())
                    : nullptr;
                mla_params.bmm1_scale = mla_bmm1_scale.has_value()
                    ? reinterpret_cast<float*>(mla_bmm1_scale.value().data_ptr())
                    : nullptr;
                mla_params.bmm2_scale = mla_bmm2_scale.has_value()
                    ? reinterpret_cast<float*>(mla_bmm2_scale.value().data_ptr())
                    : nullptr;
                mla_params.quant_q_buf
                    = quant_q_buffer.has_value() ? reinterpret_cast<void*>(quant_q_buffer.value().data_ptr()) : nullptr;
                mla_params.quant_scale_qkv = quant_scale_qkv.has_value()
                    ? reinterpret_cast<float const*>(quant_scale_qkv.value().data_ptr())
                    : nullptr;
                // Request the fused FP8-Q path; common/attentionOp.cpp gates the
                // actual skip on FP8 KV cache + absorption mode.
                mla_params.fuse_q_fp8_in_rope = (quant_q_buffer.has_value() && quant_scale_qkv.has_value());
            }
            mla_params.q_buf = attention_input;
            mla_params.context_buf = reinterpret_cast<T*>(context_buf);

            mla_params.cos_sin_cache = rotary_cos_sin_ptr;
            if (enable_dsv4_epilogue_fusion)
            {
                TORCH_CHECK(dsv4_inv_rope_cos_sin_cache.has_value(),
                    "DSv4 fused epilogue requires inverse-RoPE cos/sin cache.");
                auto const& cos_sin_cache = dsv4_inv_rope_cos_sin_cache.value();
                auto const& output_sf_tensor = output_sf.value();
                TORCH_CHECK(cos_sin_cache.scalar_type() == torch::kFloat32,
                    "DSv4 fused epilogue cos/sin cache must be float32.");
                TORCH_CHECK(
                    output.scalar_type() == torch::kFloat8_e4m3fn, "DSv4 fused epilogue output must be float8_e4m3fn.");
                TORCH_CHECK(output.dim() == 3 && output.is_contiguous(),
                    "DSv4 fused epilogue output must be contiguous [groups, tokens, K].");
                TORCH_CHECK(output_sf_tensor.scalar_type() == torch::kFloat32,
                    "DSv4 fused epilogue output_sf must be float32.");
                TORCH_CHECK(output_sf_tensor.dim() == 3 && output_sf_tensor.is_contiguous(),
                    "DSv4 fused epilogue output_sf must be contiguous [groups, K/128, padded_tokens].");
                TORCH_CHECK(output.size(1) >= num_tokens, "DSv4 fused epilogue output token dimension is too small.");
                TORCH_CHECK(op.mMLAParams.v_head_dim > 0 && op.mMLAParams.v_head_dim % 128 == 0,
                    "DSv4 fused epilogue requires v_head_dim to be a positive multiple of 128.");
                TORCH_CHECK(output_sf_tensor.size(2) >= num_tokens,
                    "DSv4 fused epilogue output_sf token dimension is too small.");

                mla_params.dsv4_epilogue_fusion.enabled = true;
                mla_params.dsv4_epilogue_fusion.cos_sin_cache = static_cast<float const*>(cos_sin_cache.data_ptr());
                mla_params.dsv4_epilogue_fusion.scale_buf_m = static_cast<int32_t>(output_sf_tensor.size(2));
            }
            mla_params.batch_size = num_seqs;
            mla_params.acc_q_len = num_tokens;
            mla_params.head_num = op.mNumHeads;
            mla_params.meta = op.mMLAParams;

            mla_params.workspace = workspace_ptr;
        }
        // Extract K/V pointers for sage attention (separate Q/K/V inputs).
        else if (is_context
            && (op.mSageAttnNumEltsPerBlkQ > 0 || op.mSageAttnNumEltsPerBlkK > 0 || op.mSageAttnNumEltsPerBlkV > 0))
        {
            TORCH_CHECK(k.has_value() && v.has_value(), "SageAttention demands separate K and V buffers");
            k_ptr = static_cast<T*>(k->slice(0, token_offset).data_ptr());
            v_ptr = static_cast<T*>(v->slice(0, token_offset).data_ptr());
        }

        int const* context_lengths_ptr = context_lengths.slice(0, seq_offset).data_ptr<int>();
        int const* sequence_lengths_ptr = sequence_length.slice(0, seq_offset).data_ptr<int>();
        // Note we still need context length during generation for MMHA optimization.
        // For encoder CUDA graphs compatibility, allow the caller to override the
        // max context Q length so FMHA kernel launch params (mMaxSeqLenQ-driven grid
        // and cluster dims) are stable across graph replays even when actual per-batch
        // sequence lengths vary.
        int32_t const max_context_q_len_computed
            = host_context_lengths.slice(0, seq_offset, seq_offset + num_seqs).max().item<int32_t>();
        int32_t const max_past_kv_length_computed
            = host_past_key_value_lengths.slice(0, seq_offset, seq_offset + num_seqs).max().item<int32_t>();

        if (max_context_q_len_override.has_value())
        {
            int32_t const override_value = static_cast<int32_t>(max_context_q_len_override.value());
            TORCH_CHECK(override_value >= max_context_q_len_computed,
                "max_context_q_len_override (%d) must be >= computed max context q length (%d).", override_value,
                max_context_q_len_computed);
            TORCH_CHECK(override_value >= max_past_kv_length_computed,
                "max_context_q_len_override (%d) must be >= computed max past kv length (%d).", override_value,
                max_past_kv_length_computed);
        }

        int32_t const max_context_q_len = max_context_q_len_override.has_value()
            ? static_cast<int32_t>(max_context_q_len_override.value())
            : max_context_q_len_computed;
        // Override the max_past_kv_length as well for encoder CUDA graph compatibility
        int32_t const max_past_kv_length = max_context_q_len_override.has_value()
            ? static_cast<int32_t>(max_context_q_len_override.value())
            : max_past_kv_length_computed;

        // Commonly, cyclic_attention_window_size, and max_attention_window_size will be the same
        // unless each layer has different attention window sizes.
        int const max_attention_window_size = beam_width == 1 ? attention_window_size
            : cache_indirection.has_value()                   ? cache_indirection.value().size(2)
                                                              : attention_window_size;
        // The cyclic_attention_window_size will determine the cyclic kv cache position of new tokens.
        // Note that this cyclic_attention_window_size might be smaller than the actual kv cache capactity.
        int const cyclic_attention_window_size = attention_window_size;
        bool const can_use_one_more_block = beam_width > 1;

        int max_blocks_per_sequence = 0;
        int32_t pool_index = 0;
        int32_t layer_idx_in_cache_pool = 0;
        KVBlockArray::DataType* block_offsets = nullptr;
        bool use_kv_cache = false;
        KvCachePoolPointers pool_pointers;
        max_blocks_per_sequence
            = op.useKVCache() && kv_cache_block_offsets.has_value() ? kv_cache_block_offsets.value().size(-1) : 0;
        pool_index = op.useKVCache() && host_kv_cache_pool_mapping.has_value()
            ? host_kv_cache_pool_mapping.value().index({op.mLayerIdx, 0}).item<int32_t>()
            : 0;
        layer_idx_in_cache_pool = op.useKVCache() && host_kv_cache_pool_mapping.has_value()
            ? host_kv_cache_pool_mapping.value().index({op.mLayerIdx, 1}).item<int32_t>()
            : 0;
        block_offsets = static_cast<KVBlockArray::DataType*>(op.useKVCache() && kv_cache_block_offsets.has_value()
                ? kv_cache_block_offsets.value().index({pool_index, seq_offset}).data_ptr()
                : nullptr);

        // The cache element size in bits.
        int cache_elem_bits = op.getKvCacheElemSizeInBits<T>();
        auto const block_size = op.mTokensPerBlock * op.mNumKVHeads * op.mHeadSize;
        auto const bytes_per_block = block_size * cache_elem_bits / 8 /*bits*/;
        int32_t const kv_factor = op.isMLAEnabled() ? 1 : 2;
        auto const intra_pool_offset = layer_idx_in_cache_pool * kv_factor * bytes_per_block;

        // Build KV cache pool pointers from the host tensor.
        use_kv_cache = op.useKVCache() && host_kv_cache_pool_pointers.has_value();
        if (use_kv_cache)
        {
            pool_pointers = buildKvCachePoolPointers(host_kv_cache_pool_pointers.value(), pool_index, intra_pool_offset,
                block_size, layer_idx_in_cache_pool, kv_factor, op.mKVCacheQuantMode.hasFp4KvCache());
        }

        float const* kv_scale_orig_quant_ptr = nullptr;
        float const* kv_scale_quant_orig_ptr = nullptr;
        if (op.mKVCacheQuantMode.hasKvCacheQuant() && kv_scale_orig_quant.has_value()
            && kv_scale_quant_orig.has_value())
        {
            kv_scale_orig_quant_ptr = kv_scale_orig_quant.value().data_ptr<float>();
            kv_scale_quant_orig_ptr = kv_scale_quant_orig.value().data_ptr<float>();
            if (op.mKVCacheQuantMode.hasFp4KvCache())
            {
                TORCH_CHECK(kv_scale_orig_quant.value().size(0) == 3);
                TORCH_CHECK(kv_scale_quant_orig.value().size(0) == 3);
            }
        }
        // For FP8 output, out_scale represents the output scale.
        float const* out_scale_ptr = (op.mFP8ContextFMHA && !op.mFuseFp4Quant && out_scale.has_value())
            ? out_scale.value().data_ptr<float>()
            : nullptr;
        // For NVFP4 output, out_scale holds the global scale for scaling factors.
        float const* out_sf_scale_ptr
            = op.mFuseFp4Quant && out_scale.has_value() ? out_scale.value().data_ptr<float>() : nullptr;

        // The attention_sinks is a float tensor with shape [num_heads_q]
        float const* attention_sinks_ptr = nullptr;
        if (attention_sinks.has_value())
        {
            TORCH_CHECK(
                attention_sinks.value().dtype() == torch::kFloat32, "Expected attention_sinks to have float dtype");
            attention_sinks_ptr = attention_sinks.value().data_ptr<float>();
        }
        T const* relative_attention_bias_ptr = nullptr;
        int relative_attention_bias_stride = 0;
        if (relative_attention_bias.has_value())
        {
            auto const& relative_attention_bias_tensor = relative_attention_bias.value();
            TORCH_CHECK(relative_attention_bias_tensor.dim() == 2 || relative_attention_bias_tensor.dim() == 3,
                "relative_attention_bias must be [num_heads, num_buckets] for implicit mode or "
                "[num_heads, max_seq_len, max_seq_len] for explicit mode");
            TORCH_CHECK(relative_attention_bias_tensor.is_contiguous(), "relative_attention_bias must be contiguous");
            TORCH_CHECK(relative_attention_bias_tensor.scalar_type() == qkv_or_q.scalar_type(),
                "relative_attention_bias dtype must match attention input dtype");
            relative_attention_bias_ptr = static_cast<T const*>(relative_attention_bias_tensor.data_ptr());
            relative_attention_bias_stride = static_cast<int>(relative_attention_bias_tensor.size(1));
        }

        // Prepare sparse attention parameters
        op.mRuntimeSparseAttentionParams.sparse_kv_indices
            = sparse_kv_indices.has_value() ? sparse_kv_indices.value().data_ptr<int32_t>() : nullptr;
        op.mRuntimeSparseAttentionParams.sparse_kv_offsets
            = sparse_kv_offsets.has_value() ? sparse_kv_offsets.value().data_ptr<int32_t>() : nullptr;
        op.mRuntimeSparseAttentionParams.sparse_attn_indices
            = sparse_attn_indices.has_value() ? sparse_attn_indices.value().data_ptr<int32_t>() : nullptr;
        op.mRuntimeSparseAttentionParams.sparse_attn_offsets
            = sparse_attn_offsets.has_value() ? sparse_attn_offsets.value().data_ptr<int32_t>() : nullptr;
        op.mRuntimeSparseAttentionParams.sparse_attn_indices_block_size = sparse_attn_indices_block_size;
        op.mRuntimeSparseAttentionParams.sparse_attn_indices_stride
            = sparse_attn_indices.has_value() ? sparse_attn_indices.value().size(-1) : 0;
        op.mRuntimeSparseAttentionParams.num_sparse_topk = num_sparse_topk;
        op.mRuntimeSparseAttentionParams.sparse_mla_topk_lens
            = sparse_mla_topk_lens.has_value() ? sparse_mla_topk_lens.value().data_ptr<int32_t>() : nullptr;
        op.mRuntimeSparseAttentionParams.sparse_kv_cache_pool = nullptr;
        op.mRuntimeSparseAttentionParams.sliding_window_kv_cache_pool = nullptr;
        if (op.mUseSparseAttention && use_kv_cache)
        {
            if (host_kv_cache_pool_pointers.has_value())
            {
                auto* kvCachePool = reinterpret_cast<char*>(
                    host_kv_cache_pool_pointers.value().index({pool_index, 0}).item<int64_t>());
                if (sparse_mla_topk_lens.has_value())
                {
                    // Deepseek V4 dynamic sparse MLA always uses the SWA pool for now.
                    op.mRuntimeSparseAttentionParams.sliding_window_kv_cache_pool = kvCachePool;
                    if (compressed_kv_cache_pool_ptr.has_value())
                    {
                        op.mRuntimeSparseAttentionParams.sparse_kv_cache_pool
                            = reinterpret_cast<char*>(compressed_kv_cache_pool_ptr.value());
                    }
                }
                else
                {
                    op.mRuntimeSparseAttentionParams.sparse_kv_cache_pool = kvCachePool;
                }
            }
        }

        AttentionOp::EnqueueParams<T> common_enqueue_params;
        common_enqueue_params.attention_input = attention_input;
        common_enqueue_params.attention_sinks = attention_sinks_ptr;
        common_enqueue_params.rotary_inv_freq = rotary_inv_freq_ptr;
        common_enqueue_params.rotary_cos_sin = rotary_cos_sin_ptr;
        common_enqueue_params.relative_attention_bias = relative_attention_bias_ptr;
        common_enqueue_params.relative_attention_bias_stride = relative_attention_bias_stride;
        common_enqueue_params.max_past_kv_length = max_past_kv_length;
        common_enqueue_params.max_attention_window_size = max_attention_window_size;
        common_enqueue_params.cyclic_attention_window_size = cyclic_attention_window_size;
        common_enqueue_params.max_cyclic_attention_window_size = cyclic_attention_window_size;
        common_enqueue_params.can_use_one_more_block = can_use_one_more_block;
        common_enqueue_params.kv_scale_orig_quant = kv_scale_orig_quant_ptr;
        common_enqueue_params.kv_scale_quant_orig = kv_scale_quant_orig_ptr;
        common_enqueue_params.attention_output_orig_quant = out_scale_ptr;
        common_enqueue_params.attention_output_sf_scale = out_sf_scale_ptr;
        common_enqueue_params.context_buf = context_buf;
        common_enqueue_params.context_buf_sf = context_buf_sf;
        common_enqueue_params.block_offsets = block_offsets;
        common_enqueue_params.host_primary_pool_pointer = pool_pointers.primaryPoolPtr;
        common_enqueue_params.host_secondary_pool_pointer = pool_pointers.secondaryPoolPtr;
        common_enqueue_params.host_primary_block_scale_pool_pointer = pool_pointers.primaryBlockScalePoolPtr;
        common_enqueue_params.host_secondary_block_scale_pool_pointer = pool_pointers.secondaryBlockScalePoolPtr;
        common_enqueue_params.num_tokens = num_tokens;
        common_enqueue_params.total_kv_len = total_kv_len;
        common_enqueue_params.max_blocks_per_sequence = max_blocks_per_sequence;
        common_enqueue_params.sequence_lengths = sequence_lengths_ptr;
        common_enqueue_params.context_lengths = context_lengths_ptr;
        common_enqueue_params.host_context_lengths = host_context_lengths.data_ptr<int32_t>();
        common_enqueue_params.workspace = workspace_ptr;
        common_enqueue_params.trtllm_gen_jit_warmup = trtllm_gen_jit_warmup;
        if (is_cross)
        {
            // For cross attention, the KV (encoder) sequence lengths are passed in via
            // `sequence_length` (already sliced into `sequence_lengths_ptr`), so reuse
            // it directly instead of a redundant `encoder_input_lengths` tensor.
            common_enqueue_params.encoder_input_lengths = sequence_lengths_ptr;
        }
        if (softmax_stats_tensor.has_value())
        {
            TLLM_CHECK_WITH_INFO(softmax_stats_tensor.value().scalar_type() == at::ScalarType::Float,
                "softmax_stats_tensor must have float type");
            TLLM_CHECK_WITH_INFO(softmax_stats_tensor.value().size(0) >= num_tokens,
                "softmax_stats_tensor must have first dimension >= num_tokens");
            TLLM_CHECK_WITH_INFO(softmax_stats_tensor.value().size(1) >= op.mNumHeads,
                "softmax_stats_tensor must have second dimension >= num_heads");
            TLLM_CHECK_WITH_INFO(
                softmax_stats_tensor.value().size(2) == 2, "softmax_stats_tensor must have third dimension == 2");
            common_enqueue_params.softmax_stats = static_cast<float2*>(softmax_stats_tensor.value().data_ptr());
        }

        // Shared helper to wire helix params into the enqueue params.
        // Works for both EnqueueContextParams and EnqueueGenerationParams since both have
        // helix_position_offsets and helix_is_inactive_rank fields.
        auto const extractHelixParams = [&helix_position_offsets, &helix_is_inactive_rank](auto& params)
        {
            if (helix_position_offsets.has_value())
            {
                params.helix_position_offsets = helix_position_offsets->data_ptr<int32_t>();
            }
            if (helix_is_inactive_rank.has_value())
            {
                params.helix_is_inactive_rank = helix_is_inactive_rank->data_ptr<bool>();
            }
        };

        if (is_context) // context stage
        {
            common_enqueue_params.input_seq_length = max_context_q_len;
            AttentionOp::EnqueueContextParams<T> enqueue_params{common_enqueue_params};
            enqueue_params.batch_size = num_seqs;
            enqueue_params.k_ptr = k_ptr;
            enqueue_params.v_ptr = v_ptr;
            if (cu_q_seqlens.has_value())
            {
                TORCH_CHECK(cu_q_seqlens->dim() == 1, "cu_q_seqlens must be a 1-D tensor.");
                TORCH_CHECK(cu_q_seqlens->is_cuda(), "cu_q_seqlens must be a CUDA tensor.");
                TORCH_CHECK(cu_q_seqlens->scalar_type() == at::ScalarType::Int, "cu_q_seqlens must be int32.");
                TORCH_CHECK(
                    cu_q_seqlens->size(0) >= num_seqs + 1, "cu_q_seqlens must have at least num_seqs + 1 elements.");
                enqueue_params.cu_q_seqlens = cu_q_seqlens->data_ptr<int32_t>();
            }
            if (cu_kv_seqlens.has_value())
            {
                TORCH_CHECK(cu_kv_seqlens->dim() == 1, "cu_kv_seqlens must be a 1-D tensor.");
                TORCH_CHECK(cu_kv_seqlens->is_cuda(), "cu_kv_seqlens must be a CUDA tensor.");
                TORCH_CHECK(cu_kv_seqlens->scalar_type() == at::ScalarType::Int, "cu_kv_seqlens must be int32.");
                TORCH_CHECK(
                    cu_kv_seqlens->size(0) >= num_seqs + 1, "cu_kv_seqlens must have at least num_seqs + 1 elements.");
                enqueue_params.cu_kv_seqlens = cu_kv_seqlens->data_ptr<int32_t>();
            }
            // Pass V's actual token stride so the FMHA runner handles both
            // contiguous V (AutoDeploy) and non-contiguous V (PyTorch backend
            // kv.split() view) correctly.
            if (v_ptr != nullptr && v.has_value())
            {
                enqueue_params.v_stride_in_bytes = v->strides()[0] * v->element_size();
            }
            if (is_cross && cross_kv.has_value())
            {
                auto const& cross_kv_tensor = cross_kv.value();
                enqueue_params.cross_kv = static_cast<T const*>(cross_kv_tensor.data_ptr());
                enqueue_params.num_encoder_tokens = static_cast<int32_t>(cross_kv_tensor.size(0));
                enqueue_params.cross_kv_length
                    = host_past_key_value_lengths.slice(0, seq_offset, seq_offset + num_seqs).max().item<int32_t>();
            }

            if (op.isMLAEnabled())
            {
                mla_params.cache_seq_lens = sequence_lengths_ptr;
                mla_params.max_input_seq_len = max_context_q_len;
                enqueue_params.mla_param = &mla_params;
            }
            if (op.isMRoPE() && mrope_rotary_cos_sin.has_value())
            {
                enqueue_params.mrope_rotary_cos_sin
                    = static_cast<float2 const*>(mrope_rotary_cos_sin.value().data_ptr());
            }
            extractHelixParams(enqueue_params);
            op.enqueueContext<T, KVBlockArray>(enqueue_params, stream);
        }
        else // generation stage
        {
            int32_t const batch_beam = num_seqs;
            TLLM_CHECK(batch_beam % beam_width == 0);
            int32_t const num_requests = batch_beam / beam_width;

            TLLM_CHECK_WITH_INFO(num_tokens % num_seqs == 0,
                "seq_len should be same for all generation requests, num_tokens=%d, num_seqs=%d", num_tokens, num_seqs);
            int32_t const input_seq_length = num_tokens / num_seqs;

            common_enqueue_params.input_seq_length = input_seq_length;
            AttentionOp::EnqueueGenerationParams<T> enqueue_params{common_enqueue_params};
            enqueue_params.layer_idx = op.mLayerIdx;
            enqueue_params.beam_width = beam_width;
            enqueue_params.num_requests = num_requests;
            enqueue_params.cache_indir = beam_width == 1
                ? nullptr
                : (cache_indirection.has_value() ? cache_indirection.value().data_ptr<int32_t>() : nullptr);
            enqueue_params.semaphores = op.multiBlockSemaphores();
            enqueue_params.host_past_key_value_lengths = host_past_key_value_lengths.data_ptr<int32_t>();
            enqueue_params.start_token_idx_sf = token_offset;

            if (op.isMRoPE() && mrope_position_deltas.has_value())
            {
                enqueue_params.mrope_position_deltas = mrope_position_deltas.value().data_ptr<int32_t>();
            }
            if (op.mIsSpecDecodingEnabled && op.mUseSpecDecoding)
            {
                bool useTllmGen = tensorrt_llm::common::isSM100Family();
                TORCH_CHECK(spec_decoding_generation_lengths.has_value(),
                    "Expecting spec_decoding_generation_lengths in spec-dec mode.");
                TORCH_CHECK(spec_decoding_position_offsets_for_cpp.has_value(),
                    "Expecting spec_decoding_position_offsets_for_cpp in spec-dec mode.");
                TORCH_CHECK(
                    spec_decoding_packed_mask.has_value(), "Expecting spec_decoding_packed_mask in spec-dec mode.");
                if (useTllmGen)
                {
                    TORCH_CHECK(spec_decoding_bl_tree_mask_offset.has_value(),
                        "Expecting spec_decoding_bl_tree_mask_offset in trtllm-gen spec-dec mode.");
                    TORCH_CHECK(spec_decoding_bl_tree_mask.has_value(),
                        "Expecting spec_decoding_bl_tree_mask in trtllm-gen spec-dec mode.");
                    TORCH_CHECK(spec_bl_tree_first_sparse_mask_offset_kv.has_value(),
                        "Expecting spec_bl_tree_first_sparse_mask_offset_kv in trtllm-gen spec-dec mode.");
                    enqueue_params.spec_decoding_bl_tree_mask_offset
                        = spec_decoding_bl_tree_mask_offset->data_ptr<int64_t>();
                    enqueue_params.spec_decoding_bl_tree_mask = spec_decoding_bl_tree_mask->data_ptr<uint32_t>();
                    enqueue_params.spec_bl_tree_first_sparse_mask_offset_kv
                        = spec_bl_tree_first_sparse_mask_offset_kv->data_ptr<int32_t>();
                }
                enqueue_params.spec_decoding_generation_lengths = spec_decoding_generation_lengths->data_ptr<int32_t>();
                enqueue_params.spec_decoding_position_offsets
                    = spec_decoding_position_offsets_for_cpp->data_ptr<int32_t>();
                enqueue_params.spec_decoding_packed_mask = spec_decoding_packed_mask->data_ptr<int32_t>();
                enqueue_params.spec_decoding_is_generation_length_variable = true;
                TLLM_CHECK(spec_decoding_position_offsets_for_cpp->dim() == 2); // [batch_size, max_draft_len + 1]
                if (useTllmGen)
                {
                    // Blackwell uses the padded packed-mask row dim as the mask stride.
                    TLLM_CHECK(spec_decoding_packed_mask->dim() == 3);
                    enqueue_params.spec_decoding_max_generation_length = spec_decoding_packed_mask->sizes()[1];
                }
                else
                {
                    enqueue_params.spec_decoding_max_generation_length
                        = spec_decoding_position_offsets_for_cpp->sizes()[1];
                }
            }

            // Current mlaGeneration will using fmha to do attention, so we don't go into enqueueGeneration
            if (op.isMLAEnabled())
            {
                if (op.mUseGenFlashMLA == true)
                {
                    TORCH_CHECK(block_ids_per_seq.has_value());
                    int const* block_ids_per_seq_ptr = static_cast<int*>(block_ids_per_seq->data_ptr());
                    mla_params.block_ids_per_seq = block_ids_per_seq_ptr;
                    // Use pre-computed metadata if provided.
                    if (flash_mla_tile_scheduler_metadata.has_value())
                    {
                        TORCH_CHECK(flash_mla_num_splits.has_value(),
                            "flash_mla_num_splits must be provided when flash_mla_tile_scheduler_metadata is set.");
                        mla_params.flash_mla_tile_scheduler_metadata
                            = flash_mla_tile_scheduler_metadata->data_ptr<int>();
                        mla_params.flash_mla_num_splits = flash_mla_num_splits->data_ptr<int>();
                    }
                }
                mla_params.cache_seq_lens = sequence_lengths_ptr;
                {
                    op.mlaGeneration<T>(mla_params, enqueue_params, stream);
                }
            }
            else
            {
                extractHelixParams(enqueue_params);
                {
                    op.enqueueGeneration<T, KVBlockArray>(enqueue_params, stream);
                }
            }

            {
                std::string const afterGenStr = "gen attention at layer " + std::to_string(op.mLayerIdx);
                {
                    TLLM_CHECK_DEBUG_WITH_INFO(tensorrt_llm::runtime::utils::tensorHasInvalid(num_tokens,
                                                   output.size(1), op.mType, context_buf, stream, afterGenStr)
                            == false,
                        "Found invalid number (NaN or Inf) in " + afterGenStr);
                }
            }
        }
        sync_check_cuda_error(stream);
    }
};

template class Runner<float>;
template class Runner<half>;
template class Runner<half, __nv_fp8_e4m3>;
#ifdef ENABLE_BF16
template class Runner<__nv_bfloat16>;
template class Runner<__nv_bfloat16, __nv_fp8_e4m3>;
#endif

} // namespace trtllm::attention

using RunnerPtr = std::shared_ptr<torch_ext::trtllm::attention::RunnerBase>;
using torch_ext::trtllm::attention::Runner;
using torch_ext::trtllm::attention::AttentionInputType;

void attention(torch::Tensor q, std::optional<torch::Tensor> k, std::optional<torch::Tensor> v, torch::Tensor& output,
    std::optional<torch::Tensor> output_sf, std::optional<torch::Tensor> workspace_, torch::Tensor sequence_length,
    torch::Tensor host_past_key_value_lengths, torch::Tensor host_total_kv_lens, torch::Tensor context_lengths,
    torch::Tensor host_context_lengths, torch::Tensor host_request_types,
    std::optional<int64_t> max_context_q_len_override, std::optional<torch::Tensor> kv_cache_block_offsets,
    std::optional<torch::Tensor> host_kv_cache_pool_pointers, std::optional<torch::Tensor> host_kv_cache_pool_mapping,
    std::optional<torch::Tensor> cache_indirection, std::optional<torch::Tensor> kv_scale_orig_quant,
    std::optional<torch::Tensor> kv_scale_quant_orig, std::optional<torch::Tensor> out_scale,
    std::optional<torch::Tensor> rotary_inv_freq, std::optional<torch::Tensor> rotary_cos_sin,
    std::optional<torch::Tensor> latent_cache, std::optional<torch::Tensor> q_pe,
    std::optional<torch::Tensor> block_ids_per_seq, std::optional<torch::Tensor> attention_sinks,
    bool const is_fused_qkv, bool const update_kv_cache, int64_t const predicted_tokens_per_seq,
    int64_t const local_layer_idx, int64_t const num_heads, int64_t const num_kv_heads, int64_t const head_size,
    std::optional<int64_t> const tokens_per_block, int64_t const max_num_requests, int64_t const max_context_length,
    int64_t const max_seq_len, int64_t const attention_window_size, int64_t const beam_width, int64_t const mask_type,
    int64_t const quant_mode, double const q_scaling, int64_t const position_embedding_type, int64_t const rope_dim,
    double const rope_base, int64_t const rope_scale_type, double const rope_scale, double const rope_short_m_scale,
    double const rope_long_m_scale, int64_t const rope_max_positions, int64_t const rope_original_max_positions,
    bool const use_paged_context_fmha, std::optional<int64_t> attention_input_type, bool is_mla_enable,
    std::optional<int64_t> chunked_prefill_buffer_batch_size, std::optional<int64_t> q_lora_rank,
    std::optional<int64_t> kv_lora_rank, std::optional<int64_t> qk_nope_head_dim,
    std::optional<int64_t> qk_rope_head_dim, std::optional<int64_t> v_head_dim, std::optional<bool> rope_append,
    std::optional<torch::Tensor> mrope_rotary_cos_sin, std::optional<torch::Tensor> mrope_position_deltas,
    std::optional<torch::Tensor> helix_position_offsets, std::optional<torch::Tensor> helix_is_inactive_rank,
    std::optional<int64_t> attention_chunk_size, std::optional<torch::Tensor> softmax_stats_tensor,
    bool const is_spec_decoding_enabled, bool const use_spec_decoding, bool const is_spec_dec_tree,
    std::optional<torch::Tensor> spec_decoding_generation_lengths,
    std::optional<torch::Tensor> spec_decoding_position_offsets_for_cpp,
    std::optional<torch::Tensor> spec_decoding_packed_mask,
    std::optional<torch::Tensor> spec_decoding_bl_tree_mask_offset,
    std::optional<torch::Tensor> spec_decoding_bl_tree_mask,
    std::optional<torch::Tensor> spec_bl_tree_first_sparse_mask_offset_kv,
    std::optional<torch::Tensor> sparse_kv_indices, std::optional<torch::Tensor> sparse_kv_offsets,
    std::optional<torch::Tensor> sparse_attn_indices, std::optional<torch::Tensor> sparse_attn_offsets,
    int64_t const sparse_attn_indices_block_size, std::optional<int64_t> num_sparse_topk,
    std::optional<torch::Tensor> sparse_mla_topk_lens,
    std::optional<double> skip_softmax_threshold_scale_factor_prefill,
    std::optional<double> skip_softmax_threshold_scale_factor_decode, std::optional<torch::Tensor> skip_softmax_stat,
    std::optional<torch::Tensor> cu_q_seqlens, std::optional<torch::Tensor> cu_kv_seqlens,
    std::optional<torch::Tensor> fmha_scheduler_counter, std::optional<torch::Tensor> mla_bmm1_scale,
    std::optional<torch::Tensor> mla_bmm2_scale, std::optional<torch::Tensor> quant_q_buffer,
    std::optional<torch::Tensor> flash_mla_tile_scheduler_metadata, std::optional<torch::Tensor> flash_mla_num_splits,
    int64_t sage_attn_num_elts_per_blk_q, int64_t sage_attn_num_elts_per_blk_k, int64_t sage_attn_num_elts_per_blk_v,
    bool sage_attn_qk_int8, int64_t num_contexts, int64_t num_ctx_tokens, bool trtllm_gen_jit_warmup,
    std::optional<int64_t> compressed_kv_cache_pool_ptr, bool const is_cross, std::optional<torch::Tensor> cross_kv,
    std::optional<torch::Tensor> relative_attention_bias, int64_t relative_attention_max_distance,
    std::optional<int64_t> spec_decoding_target_max_draft_tokens, std::optional<torch::Tensor> quant_scale_qkv,
    std::optional<torch::Tensor> dsv4_inv_rope_cos_sin_cache, bool enable_dsv4_epilogue_fusion)
{
    TLLM_LOG_TRACE("Attention op starts at layer %d", local_layer_idx);
    // Use these tensors to infer if the attention is using KV cache
    bool const use_kv_cache = kv_cache_block_offsets.has_value() && host_kv_cache_pool_pointers.has_value()
        && host_kv_cache_pool_mapping.has_value();

    bool const use_sage_attn
        = sage_attn_num_elts_per_blk_q > 0 || sage_attn_num_elts_per_blk_k > 0 || sage_attn_num_elts_per_blk_v > 0;
    TLLM_CHECK_WITH_INFO(is_mla_enable || is_fused_qkv || use_sage_attn || is_cross,
        "For non-MLA, non-cross, non-SageAttention attention, only fused QKV is supported now.");
    TLLM_CHECK_WITH_INFO(
        update_kv_cache || is_cross, "KV cache update cannot be disabled now (except for cross attention).");
    auto qkv_or_q = q;
    if (is_fused_qkv)
    {
        TLLM_CHECK_WITH_INFO(!k.has_value(), "The k tensor should be null if using fused QKV");
        TLLM_CHECK_WITH_INFO(!v.has_value(), "The v tensor should be null if using fused QKV");
    }
    if (!is_fused_qkv && update_kv_cache && !is_cross)
    {
        TLLM_CHECK_WITH_INFO(k.has_value(), "The k tensor should be provided if updating KV cache with unfused K/V");
        TLLM_CHECK_WITH_INFO(v.has_value(), "The v tensor should be provided if updating KV cache with unfused K/V");
    }
    if (use_sage_attn)
    {
        TLLM_CHECK_WITH_INFO(
            !is_fused_qkv, "SageAttention requires separate q/k/v tensors (is_fused_qkv must be false).");
        TLLM_CHECK_WITH_INFO(k.has_value(), "SageAttention requires k tensor to be provided.");
        TLLM_CHECK_WITH_INFO(v.has_value(), "SageAttention requires v tensor to be provided.");
    }

    auto const dtype = tensorrt_llm::runtime::TorchUtils::dataType(qkv_or_q.scalar_type());
    auto const out_dtype = output.scalar_type();
    bool const is_fp8_out = out_dtype == torch::kFloat8_e4m3fn;
    // Torch does not support native nvfp4 type.
    bool const is_fp4_out = out_dtype == torch::kUInt8;

    RunnerPtr runner;
    if (dtype == nvinfer1::DataType::kHALF)
    {
        if (is_fp8_out)
        {
            runner = std::make_shared<Runner<half, __nv_fp8_e4m3>>();
        }
        else if (is_fp4_out)
        {
            runner = std::make_shared<Runner<half, __nv_fp4_e2m1>>();
        }
        else
        {
            TLLM_CHECK(out_dtype == torch::kFloat16);
            runner = std::make_shared<Runner<half>>();
        }
    }
    else if (dtype == nvinfer1::DataType::kFLOAT)
    {
        TLLM_CHECK(out_dtype == torch::kFloat32);
        runner = std::make_shared<Runner<float>>();
    }
#ifdef ENABLE_BF16
    else if (dtype == nvinfer1::DataType::kBF16)
    {
        if (is_fp8_out)
        {
            runner = std::make_shared<Runner<__nv_bfloat16, __nv_fp8_e4m3>>();
        }
        else if (is_fp4_out)
        {
            runner = std::make_shared<Runner<__nv_bfloat16, __nv_fp4_e2m1>>();
        }
        else
        {
            TLLM_CHECK(out_dtype == torch::kBFloat16);
            runner = std::make_shared<Runner<__nv_bfloat16>>();
        }
    }
#endif
    runner->beam_width = beam_width;
    runner->max_num_requests = max_num_requests;
    runner->attention_window_size = attention_window_size;

    auto op = std::make_shared<AttentionOp>();
    op->mType = dtype;
    op->mFMHAForceFP32Acc = dtype == nvinfer1::DataType::kBF16;
    op->mLayerIdx = local_layer_idx;
    op->mNumHeads = num_heads;
    op->mNumKVHeads = num_kv_heads;
    op->mHeadSize = head_size;
    op->mMaskType = static_cast<tensorrt_llm::kernels::AttentionMaskType>(int32_t(mask_type));
    op->mKVCacheQuantMode = tensorrt_llm::common::QuantMode(uint32_t(quant_mode));
    op->mUseKVCache = use_kv_cache;
    op->mPagedKVCache = op->mPagedKVCache && use_kv_cache; // update mPagedKVCache based on use_kv_cache
    op->mTokensPerBlock = tokens_per_block.value_or(0);
    op->mFP8GenerationMLA = false;
    op->mFuseFp4Quant = is_fp4_out;
    op->mFusesDsv4InvRopeFp8Quant = enable_dsv4_epilogue_fusion;
    op->mMaxContextLength = max_context_length;
    op->mMaxSeqLen = max_seq_len;
    op->mMaxNumRequests = max_num_requests;
    op->mQScaling = q_scaling;
    op->mPositionEmbeddingType
        = static_cast<tensorrt_llm::kernels::PositionEmbeddingType>(int8_t(position_embedding_type));
    if (relative_attention_bias.has_value())
    {
        auto const relative_attention_bias_dim = relative_attention_bias.value().dim();
        TORCH_CHECK(relative_attention_bias_dim == 2 || relative_attention_bias_dim == 3,
            "relative_attention_bias must be [num_heads, num_buckets] for implicit mode or "
            "[num_heads, max_seq_len, max_seq_len] for explicit mode");
        TORCH_CHECK(relative_attention_bias_dim != 2 || relative_attention_max_distance > 0,
            "relative_attention_max_distance must be positive when relative_attention_bias is a bucket table");
        TORCH_CHECK(relative_attention_bias_dim != 3 || relative_attention_max_distance == 0,
            "relative_attention_max_distance must be 0 when relative_attention_bias is precomputed");
        TLLM_CHECK_WITH_INFO(op->mPositionEmbeddingType == tensorrt_llm::kernels::PositionEmbeddingType::kRELATIVE,
            "relative_attention_bias requires position_embedding_type to be relative.");
        op->mMaxDistance = static_cast<int>(relative_attention_max_distance);
    }
    op->mRotaryEmbeddingDim = rope_dim;
    op->mRotaryEmbeddingBase = rope_base;
    op->mRotaryEmbeddingScaleType = static_cast<tensorrt_llm::kernels::RotaryScalingType>(int8_t(rope_scale_type));
    op->mRotaryEmbeddingScale = rope_scale;
    op->mRotaryEmbeddingShortMscale = rope_short_m_scale;
    op->mRotaryEmbeddingLongMscale = rope_long_m_scale;
    op->mRotaryEmbeddingMaxPositions = rope_max_positions;
    op->mRotaryEmbeddingOriginalMaxPositions = rope_original_max_positions;
    op->mFP8ContextFMHA = is_fp8_out || is_fp4_out || (op->mKVCacheQuantMode.hasFp8KvCache() && use_paged_context_fmha)
        || use_sage_attn;
    // SageAttention block sizes and quantization mode.
    op->mSageAttnNumEltsPerBlkQ = static_cast<int>(sage_attn_num_elts_per_blk_q);
    op->mSageAttnNumEltsPerBlkK = static_cast<int>(sage_attn_num_elts_per_blk_k);
    op->mSageAttnNumEltsPerBlkV = static_cast<int>(sage_attn_num_elts_per_blk_v);
    op->mSageAttnQkInt8 = sage_attn_qk_int8;
    op->mFP8AttenOutput = is_fp8_out;
    op->mPagedContextFMHA = use_paged_context_fmha;
    op->mCrossAttention = is_cross;

    op->mAttentionChunkSize = attention_chunk_size;
    op->mSkipSoftmaxThresholdScaleFactorPrefill
        = static_cast<float>(skip_softmax_threshold_scale_factor_prefill.value_or(0));
    op->mSkipSoftmaxThresholdScaleFactorDecode
        = static_cast<float>(skip_softmax_threshold_scale_factor_decode.value_or(0));
#ifdef SKIP_SOFTMAX_STAT
    op->mSkipSoftmaxTotalBlocks = reinterpret_cast<uint32_t*>(skip_softmax_stat.value().data_ptr());
    op->mSkipSoftmaxSkippedBlocks = op->mSkipSoftmaxTotalBlocks + 1;
#endif
    op->mIsSpecDecodingEnabled = is_spec_decoding_enabled;
    op->mUseSpecDecoding = use_spec_decoding;
    op->mIsSpecDecTree = is_spec_dec_tree;
    // Include static tree length in the AttentionOp cache key.
    if (spec_decoding_target_max_draft_tokens.has_value() && op->mSpecDecodingTargetMaxGenLen == 0)
    {
        op->mSpecDecodingTargetMaxGenLen = static_cast<int32_t>(spec_decoding_target_max_draft_tokens.value()) + 1;
    }

    op->mUseSparseAttention = false;
    op->mUseTllmGenSparseAttentionPaged = false;
    op->mUseTllmGenSparseAttention = false;
    if ((sparse_kv_indices.has_value() && sparse_kv_indices.value().numel() > 0)
        || (sparse_attn_indices.has_value() && sparse_attn_indices.value().numel() > 0))
    {
        op->mUseSparseAttention = true;
        if (sparse_attn_indices.has_value() && sparse_attn_indices.value().numel() > 0)
        {
            // Dispatch based on sparse_attn_offsets presence:
            // - sparse_attn_offsets provided → generation paged sparse attention
            // - sparse_attn_offsets absent → context sparse attention
            if (sparse_attn_offsets.has_value() && sparse_attn_offsets.value().numel() > 0)
            {
                op->mUseTllmGenSparseAttentionPaged = true;
            }
            else
            {
                op->mUseTllmGenSparseAttention = true;
            }
        }
    }
    int32_t const num_sparse_topk_value = num_sparse_topk.has_value() ? num_sparse_topk.value() : 0;

    if (is_mla_enable)
    {
        // MLA does not support NVFP4 output yet.
        TLLM_CHECK(!is_fp4_out);

        TLLM_CHECK(host_kv_cache_pool_mapping.has_value());
        int32_t const layer_num = host_kv_cache_pool_mapping.value().size(0);
        bool const rope_append_value = rope_append.value_or(true);

        if (num_sparse_topk_value > 0 && sparse_attn_indices.has_value() && sparse_attn_indices.value().numel() > 0)
        {
            op->mUseSparseAttention = true;
        }

        op->mIsMLAEnabled = true;
        op->mMLAParams = {static_cast<int>(q_lora_rank.value()), static_cast<int>(kv_lora_rank.value()),
            static_cast<int>(qk_nope_head_dim.value()), static_cast<int>(qk_rope_head_dim.value()),
            static_cast<int>(v_head_dim.value()), static_cast<int>(predicted_tokens_per_seq),
            static_cast<int>(layer_num), static_cast<int>(rope_append_value)};

        op->mFP8ContextMLA
            = (tensorrt_llm::common::getSMVersion() == 90 || tensorrt_llm::common::getSMVersion() == 100
                  || tensorrt_llm::common::getSMVersion() == 103 || tensorrt_llm::common::getSMVersion() == 120)
            && op->mKVCacheQuantMode.hasFp8KvCache();
        op->mIsGenerationMLA = head_size == op->mMLAParams.kv_lora_rank + op->mMLAParams.qk_rope_head_dim;
        op->mFP8GenerationMLA = op->mKVCacheQuantMode.hasFp8KvCache();
        // only enable flash mla on sm90 and head_size == 576 and tokens_per_block == 64
        op->mUseGenFlashMLA = tensorrt_llm::common::getSMVersion() == 90 && tokens_per_block == 64 && head_size == 576;

        // The following two parameters are used to compute kvcache related parameters such as kvcache block_size. So
        // they need to be set to 1 and 512 + 64 for both context and generation. For MLA attention kernel configs,
        // mNumKVHeads/mHeadSize are overwritten in common/attentionOp.cpp.
        op->mNumKVHeads = 1;
        op->mHeadSize = op->mMLAParams.kv_lora_rank + op->mMLAParams.qk_rope_head_dim;

        // For chunked prefill MLA, we need larger buffer size for k and v
        op->mChunkPrefillBufferBatchSize
            = chunked_prefill_buffer_batch_size.has_value() ? chunked_prefill_buffer_batch_size.value() : 1;
    }

    auto cache_key = std::make_tuple(op->data(), runner->data());
    using CacheKey = decltype(cache_key);
    static std::unordered_map<CacheKey, std::shared_ptr<AttentionOp>, hash<CacheKey>> op_cache;
    if (auto it = op_cache.find(cache_key); it != op_cache.end())
    {
        TLLM_LOG_TRACE("Attention op for layer %d is cached", local_layer_idx);
        op = it->second;
    }
    else
    {
        TLLM_LOG_TRACE("Preparing new attention op for layer %d with cache key: %s", local_layer_idx,
            to_string(cache_key).c_str());
        op->initialize();
        runner->prepare(*op);
        op_cache[cache_key] = op;
    }

    int32_t const num_seqs = host_context_lengths.size(0);
    RequestType const* request_types = static_cast<RequestType const*>(host_request_types.data_ptr());

    AttentionInputType attn_input_type = AttentionInputType::Mixed;
    if (attention_input_type.has_value())
    {
        attn_input_type = static_cast<AttentionInputType>(attention_input_type.value());
    }
    bool const is_gen_only = attn_input_type == AttentionInputType::GenerationOnly;

    int32_t const num_generations = num_seqs - static_cast<int32_t>(num_contexts);
    int32_t const num_tokens = qkv_or_q.size(0);
    int32_t const num_gen_tokens = is_gen_only ? num_tokens : num_tokens - static_cast<int32_t>(num_ctx_tokens);
    auto const ctx_total_kv_len = host_total_kv_lens.index({0}).item<int32_t>();
    auto const gen_total_kv_len = host_total_kv_lens.index({1}).item<int32_t>();

    for (int32_t idx = num_contexts; idx < num_seqs; idx++)
    {
        TLLM_CHECK(request_types[idx] == RequestType::kGENERATION);
    }

    int32_t const max_attention_window_size
        = beam_width == 1 ? attention_window_size : cache_indirection.value().size(2);
    int32_t const max_blocks_per_sequence
        = use_kv_cache && kv_cache_block_offsets.has_value() ? kv_cache_block_offsets.value().size(-1) : 0;
    int64_t const workspace_size = runner->getWorkspaceSize(
        *op, num_tokens, max_attention_window_size, num_gen_tokens, max_blocks_per_sequence, ctx_total_kv_len);
    TLLM_LOG_TRACE("Expected workspace size is %ld bytes", workspace_size);

    torch::Tensor workspace;
    if (workspace_.has_value())
    {
        if (workspace_.value().numel() < workspace_size)
        {
            TLLM_LOG_WARNING("Attention workspace size is not enough, increase the size from %ld bytes to %ld bytes",
                workspace_.value().numel(), workspace_size);
            workspace_.value().resize_({workspace_size});
        }
        workspace = workspace_.value();
    }
    else
    {
        TLLM_LOG_TRACE("Allocate new attention workspace with size %ld bytes", workspace_size);
        workspace = torch::empty({workspace_size}, torch::dtype(torch::kByte).device(qkv_or_q.device()));
    }

    if ((num_contexts > 0) && (attn_input_type != AttentionInputType::GenerationOnly))
    {
        auto seq_offset = 0;
        auto token_offset = 0;
        runner->run(*op,
            /*is_context=*/true, seq_offset,
            /*num_seqs=*/num_contexts, token_offset,
            /*num_tokens=*/num_ctx_tokens, predicted_tokens_per_seq, workspace, output, output_sf, qkv_or_q, k, v,
            sequence_length, host_past_key_value_lengths, ctx_total_kv_len, context_lengths, host_context_lengths,
            max_context_q_len_override, kv_cache_block_offsets, host_kv_cache_pool_pointers, host_kv_cache_pool_mapping,
            cache_indirection, kv_scale_orig_quant, kv_scale_quant_orig, out_scale, rotary_inv_freq, rotary_cos_sin,
            latent_cache, q_pe, block_ids_per_seq, mrope_rotary_cos_sin, mrope_position_deltas, helix_position_offsets,
            helix_is_inactive_rank, softmax_stats_tensor, spec_decoding_generation_lengths,
            spec_decoding_position_offsets_for_cpp, spec_decoding_packed_mask, spec_decoding_bl_tree_mask_offset,
            spec_decoding_bl_tree_mask, spec_bl_tree_first_sparse_mask_offset_kv, attention_sinks, sparse_kv_indices,
            sparse_kv_offsets, sparse_attn_indices, sparse_attn_offsets, sparse_attn_indices_block_size,
            num_sparse_topk_value, sparse_mla_topk_lens, cu_q_seqlens, cu_kv_seqlens, fmha_scheduler_counter,
            mla_bmm1_scale, mla_bmm2_scale, quant_q_buffer, flash_mla_tile_scheduler_metadata, flash_mla_num_splits,
            trtllm_gen_jit_warmup, compressed_kv_cache_pool_ptr, is_cross, cross_kv, relative_attention_bias,
            quant_scale_qkv, dsv4_inv_rope_cos_sin_cache, enable_dsv4_epilogue_fusion);
    }

    if ((num_generations > 0) && (attn_input_type != AttentionInputType::ContextOnly))
    {

        auto seq_offset = num_contexts;
        auto token_offset = is_gen_only ? 0 : num_ctx_tokens;
        runner->run(*op,
            /*is_context=*/false, seq_offset,
            /*num_seqs=*/num_generations, token_offset,
            /*num_tokens=*/num_gen_tokens, predicted_tokens_per_seq, workspace, output, output_sf, qkv_or_q, k, v,
            sequence_length, host_past_key_value_lengths, gen_total_kv_len, context_lengths, host_context_lengths,
            max_context_q_len_override, kv_cache_block_offsets, host_kv_cache_pool_pointers, host_kv_cache_pool_mapping,
            cache_indirection, kv_scale_orig_quant, kv_scale_quant_orig, out_scale, rotary_inv_freq, rotary_cos_sin,
            latent_cache, q_pe, block_ids_per_seq, mrope_rotary_cos_sin, mrope_position_deltas, helix_position_offsets,
            helix_is_inactive_rank, softmax_stats_tensor, spec_decoding_generation_lengths,
            spec_decoding_position_offsets_for_cpp, spec_decoding_packed_mask, spec_decoding_bl_tree_mask_offset,
            spec_decoding_bl_tree_mask, spec_bl_tree_first_sparse_mask_offset_kv, attention_sinks, sparse_kv_indices,
            sparse_kv_offsets, sparse_attn_indices, sparse_attn_offsets, sparse_attn_indices_block_size,
            num_sparse_topk_value, sparse_mla_topk_lens, cu_q_seqlens, cu_kv_seqlens, fmha_scheduler_counter,
            mla_bmm1_scale, mla_bmm2_scale, quant_q_buffer, flash_mla_tile_scheduler_metadata, flash_mla_num_splits,
            trtllm_gen_jit_warmup, compressed_kv_cache_pool_ptr, is_cross, cross_kv, relative_attention_bias,
            quant_scale_qkv, dsv4_inv_rope_cos_sin_cache, enable_dsv4_epilogue_fusion);
    }

    TLLM_LOG_TRACE("Attention op stops at layer %d", local_layer_idx);
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

    auto op = std::make_shared<AttentionOp>();
    op->mType = nvinfer1::DataType::kHALF;
    op->mNumHeads = num_heads;
    op->mNumKVHeads = num_kv_heads;
    op->mHeadSize = head_size;
    op->mMaskType = static_cast<tensorrt_llm::kernels::AttentionMaskType>(int32_t(mask_type));
    op->mKVCacheQuantMode = tensorrt_llm::common::QuantMode(uint32_t(quant_mode));
    op->mFP8ContextFMHA = op->mKVCacheQuantMode.hasFp8KvCache() || op->mKVCacheQuantMode.hasFp4KvCache();
    op->mUseKVCache = true;
    op->mPagedKVCache = true;
    op->mTokensPerBlock = tokens_per_block.value_or(0);
    op->mFuseFp4Quant = true;
    op->mPagedContextFMHA = use_paged_context_fmha;

    auto cache_key = op->data();
    using CacheKey = decltype(cache_key);
    static std::unordered_map<CacheKey, bool, hash<CacheKey>> op_cache;
    if (auto it = op_cache.find(cache_key); it != op_cache.end())
    {
        TLLM_LOG_TRACE("Attention op runtime check is cached");
        return it->second;
    }
    else
    {
        TLLM_LOG_TRACE("Caching attention op runtime check with cache key: %s", to_string(cache_key).c_str());
        op->initialize();
        op_cache[cache_key] = op->supportsNvFp4Output();
    }

    return op->supportsNvFp4Output();
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
