/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <climits>
#include <optional>
#include <torch/extension.h>
#include <tuple>

#include "tensorrt_llm/common/attentionOp.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

struct FmhaParams
{
#define TRTLLM_FMHA_PARAM_FIELD(name, cpp_type, py_type, cpp_default, py_default) cpp_type name cpp_default;
#include "tensorrt_llm/thop/fmha_params_fields.inc"
#undef TRTLLM_FMHA_PARAM_FIELD

#define TRTLLM_FMHA_PARAM_GETTERS
#include "tensorrt_llm/thop/fmha_params_fields.inc"
#undef TRTLLM_FMHA_PARAM_GETTERS
};

int64_t get_attention_workspace_size(FmhaParams const& params, int64_t num_tokens, int64_t max_attention_window_size,
    int64_t num_gen_tokens, int64_t max_blocks_per_sequence, int64_t ctx_total_kv_len);

void run_context(FmhaParams const& params);
void run_generation(FmhaParams const& params);
void run_mla_generation(FmhaParams const& params);

struct KvCachePoolPointers
{
    void* primaryPoolPtr{nullptr};
    void* secondaryPoolPtr{nullptr};
    void* primaryBlockScalePoolPtr{nullptr};
    void* secondaryBlockScalePoolPtr{nullptr};
};

struct KvCachePoolMapping
{
    int32_t poolIndex{0};
    int32_t layerIdxInCachePool{0};
};

KvCachePoolMapping readKvCachePoolMapping(at::Tensor const& hostKvCachePoolMapping, int64_t layerIdx);

KvCachePoolPointers buildKvCachePoolPointers(at::Tensor const& hostKvCachePoolPointers, int32_t poolIndex,
    int64_t intraPoolOffset, int64_t blockSize, int32_t layerIdxInCachePool, int32_t kvFactor, bool isFp4KvCache);

common::op::KvCacheBuffers<kernels::KVBlockArray> buildPagedKvCacheBuffers(
    std::optional<torch::Tensor> const& kv_cache_block_offsets,
    std::optional<torch::Tensor> const& host_kv_cache_pool_pointers,
    std::optional<torch::Tensor> const& host_kv_cache_pool_mapping, common::QuantMode quantMode, int64_t layer_idx,
    int64_t batch_size, int64_t tokens_per_block, int64_t kv_head_num, int64_t size_per_head,
    int64_t cyclic_attention_window_size, int64_t max_attention_window_size, int64_t beam_width, int64_t seq_offset,
    bool is_mla_enable, size_t elem_size);

std::tuple<at::Tensor, std::optional<at::Tensor>> buildFlashinferTrtllmGenPagedKvCacheBuffers(
    at::Tensor host_kv_cache_pool_pointers, at::Tensor host_kv_cache_pool_mapping, int64_t layer_idx,
    int64_t num_kv_heads, int64_t tokens_per_block, int64_t head_dim, int64_t kv_factor, int64_t total_num_blocks,
    int64_t kv_cache_quant_mode, at::ScalarType dtype);

// Layout manager for the thop attention workspace slices used by trtllm-gen.
// Context follows AttentionOp::getWorkspaceSizeForContext() ordering. Generation
// follows the XQA workspace ordering used by AttentionOp generation.
struct TrtllmGenContextWorkspaceLayout
{
    int64_t trtllmGenWorkspaceOffset{};
    int64_t cuQSeqlensOffset{};
    int64_t cuKvSeqlensOffset{};
    int64_t cuMaskRowsOffset{};
    int64_t rotaryInvFreqOffset{};
    int64_t qBufOffset{};
    int64_t tokensInfoOffset{};
    int64_t fmhaTileCounterOffset{};
    int64_t fmhaBmm1ScaleOffset{};
    int64_t fmhaBmm2ScaleOffset{};
    int64_t trtllmGenWorkspaceSize{};
    int64_t cuSeqlensSize{};
    int64_t rotaryInvFreqSize{};
    int64_t qBufSize{};
    int64_t tokensInfoSize{};
    int64_t fmhaTileCounterSize{};
    int64_t fmhaBmm1ScaleSize{};
    int64_t fmhaBmm2ScaleSize{};
    int64_t totalSize{};
    at::ScalarType qBufScalarType{};
};

struct TrtllmGenGenerationWorkspaceLayout
{
    int64_t trtllmGenWorkspaceOffset{};
    int64_t cuSeqlensOffset{};
    int64_t cuKvSeqlensOffset{};
    int64_t rotaryInvFreqOffset{};
    int64_t tokensInfoOffset{};
    int64_t qBufOffset{};
    int64_t bmm1ScaleOffset{};
    int64_t bmm2ScaleOffset{};
    int64_t sparseAttnCacheOffset{};
    int64_t trtllmGenWorkspaceSize{};
    int64_t cuSeqlensSize{};
    int64_t cuKvSeqlensSize{};
    int64_t rotaryInvFreqSize{};
    int64_t tokensInfoSize{};
    int64_t qBufSize{};
    int64_t bmm1ScaleSize{};
    int64_t bmm2ScaleSize{};
    int64_t sparseAttnCacheSize{};
    int64_t totalSize{};
    at::ScalarType qBufScalarType{};
};

struct TrtllmGenContextWorkspaceViews
{
    at::Tensor trtllmGenWorkspace;
    at::Tensor cuQSeqlens;
    at::Tensor cuKvSeqlens;
    at::Tensor cuMaskRows;
    std::optional<at::Tensor> rotaryInvFreqBuf;
    std::optional<at::Tensor> qBuf;
    at::Tensor tokensInfo;
    at::Tensor fmhaTileCounter;
    std::optional<at::Tensor> fmhaBmm1Scale;
    std::optional<at::Tensor> fmhaBmm2Scale;
};

struct TrtllmGenGenerationWorkspaceViews
{
    at::Tensor trtllmGenWorkspace;
    at::Tensor cuSeqlens;
    at::Tensor cuKvSeqlens;
    std::optional<at::Tensor> rotaryInvFreqBuf;
    at::Tensor tokensInfo;
    at::Tensor qBuf;
    at::Tensor bmm1Scale;
    at::Tensor bmm2Scale;
    std::optional<at::Tensor> sparseAttnCache;
};

class TrtllmAttentionWorkspaceManager
{
public:
    static constexpr int64_t kWorkspaceAlignment = 256;
    static constexpr int64_t kTrtllmGenWorkspaceSize = CUBLAS_WORKSPACE_SIZE;

    static TrtllmGenContextWorkspaceLayout buildContextLayout(at::ScalarType qDtype, int64_t batchSize,
        int64_t numTokens, int64_t numHeads, int64_t headSize, int64_t rotaryEmbeddingDim, bool separateQKvInput,
        bool fp8ContextFmha);

    static TrtllmGenGenerationWorkspaceLayout buildGenerationLayout(at::ScalarType qDtype, int64_t batchBeam,
        int64_t numTokens, int64_t numHeads, int64_t headSize, int64_t rotaryEmbeddingDim, int64_t numKvHeads,
        int64_t maxBlocksPerSequence, bool useSparseAttention);

    static int64_t getContextWorkspaceSize(at::ScalarType qDtype, int64_t batchSize, int64_t numTokens,
        int64_t numHeads, int64_t headSize, int64_t rotaryEmbeddingDim, bool separateQKvInput, bool fp8ContextFmha);

    //! numKvHeads and maxBlocksPerSequence affect the size only when sparse attention is enabled.
    static int64_t getGenerationWorkspaceSize(at::ScalarType qDtype, int64_t batchBeam, int64_t numTokens,
        int64_t numHeads, int64_t headSize, int64_t rotaryEmbeddingDim, int64_t numKvHeads,
        int64_t maxBlocksPerSequence, bool useSparseAttention);

    static TrtllmGenContextWorkspaceViews materializeContextWorkspace(
        at::Tensor const& workspace, TrtllmGenContextWorkspaceLayout const& layout);

    static TrtllmGenContextWorkspaceViews materializeContextWorkspace(at::Tensor const& workspace,
        at::ScalarType qDtype, int64_t batchSize, int64_t numTokens, int64_t numHeads, int64_t headSize,
        int64_t rotaryEmbeddingDim, bool fp8ContextFmha);

    static TrtllmGenGenerationWorkspaceViews materializeGenerationWorkspace(
        at::Tensor const& workspace, TrtllmGenGenerationWorkspaceLayout const& layout);

    static TrtllmGenGenerationWorkspaceViews materializeGenerationWorkspace(at::Tensor const& workspace,
        at::ScalarType qDtype, int64_t batchBeam, int64_t numTokens, int64_t numHeads, int64_t headSize,
        int64_t rotaryEmbeddingDim, int64_t numKvHeads);

private:
    static std::optional<at::Tensor> makeWorkspaceView(
        at::Tensor const& workspace, int64_t offset, int64_t sizeBytes, at::ScalarType scalarType);
};

} // namespace torch_ext

/**
 * @brief Compute FlashMLA tile-scheduler metadata in-place.
 *
 * Call once per forward pass before the attention layers to pre-compute
 * get_mla_metadata and store the results in the provided tensors. Pass
 * these tensors to the attention op so all layers reuse the same metadata.
 */
void computeFlashMlaMetadata(torch::Tensor seqlens_k, torch::Tensor tile_scheduler_metadata, torch::Tensor num_splits,
    int64_t batch_size, int64_t s_q, int64_t num_q_heads, int64_t num_kv_heads, int64_t head_size_v);

TRTLLM_NAMESPACE_END
