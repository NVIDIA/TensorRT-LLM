/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/csrc/ropeKvAppend.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <optional>

namespace tk = tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;
namespace rope_kv_append = TRTLLM_MV2_NS::rope_kv_append;

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

void checkCuda(torch::Tensor const& t, char const* name)
{
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
}

//! Paged MLA latent cache of one layer: the KVBlockArray that buildPagedKvCacheBuffers (thop/attentionOp.cpp)
//! builds for an FP8 MLA cache (kvFactor 1, one KV head, 8-bit elements, beam width 1, no FP4 scale pool).
tk::KVBlockArray buildFp8MlaKvBlockArray(torch::Tensor const& kvCacheBlockOffsets,
    torch::Tensor const& hostKvCachePoolPointers, torch::Tensor const& hostKvCachePoolMapping, int64_t layerIdx,
    int32_t batchSize, int64_t tokensPerBlock, int64_t attentionWindowSize, int64_t seqOffset)
{
    TORCH_CHECK(hostKvCachePoolMapping.device().is_cpu() && hostKvCachePoolMapping.dim() == 2
            && hostKvCachePoolMapping.size(1) >= 2 && hostKvCachePoolMapping.scalar_type() == torch::kInt32,
        "host_kv_cache_pool_mapping must be a CPU int32 [layers, >=2] tensor");
    TORCH_CHECK(layerIdx >= 0 && layerIdx < hostKvCachePoolMapping.size(0), "layer index out of the pool mapping");
    auto const* mapping = hostKvCachePoolMapping.data_ptr<int32_t>();
    int32_t const poolIndex = mapping[layerIdx * hostKvCachePoolMapping.stride(0)];
    int32_t const layerIdxInCachePool
        = mapping[layerIdx * hostKvCachePoolMapping.stride(0) + hostKvCachePoolMapping.stride(1)];

    TORCH_CHECK(hostKvCachePoolPointers.device().is_cpu() && hostKvCachePoolPointers.dim() == 2
            && hostKvCachePoolPointers.scalar_type() == torch::kInt64 && poolIndex >= 0
            && poolIndex < hostKvCachePoolPointers.size(0) && hostKvCachePoolPointers.size(1) >= 2,
        "host_kv_cache_pool_pointers must be a CPU int64 [pools, 2] tensor");
    TORCH_CHECK(kvCacheBlockOffsets.is_cuda() && kvCacheBlockOffsets.scalar_type() == torch::kInt32
            && kvCacheBlockOffsets.dim() >= 2 && poolIndex < kvCacheBlockOffsets.size(0)
            && seqOffset < kvCacheBlockOffsets.size(1),
        "kv_cache_block_offsets must be CUDA int32 [pools, seqs, 2, max_blocks]");

    int64_t constexpr kCacheElemBits = 8;
    int64_t const blockSize = tokensPerBlock * /*kv_head_num=*/1 * rope_kv_append::kLatentDim;
    int64_t const bytesPerBlock = blockSize * kCacheElemBits / CHAR_BIT;
    int64_t const intraPoolOffset = layerIdxInCachePool * /*kvFactor=*/1 * bytesPerBlock;
    auto const sizePerToken = static_cast<int32_t>(rope_kv_append::kLatentDim * kCacheElemBits / 8);
    auto const* pointers = hostKvCachePoolPointers.data_ptr<int64_t>();
    auto const rowOffset = poolIndex * hostKvCachePoolPointers.stride(0);
    void* primaryPoolPtr = reinterpret_cast<char*>(pointers[rowOffset]) + intraPoolOffset;
    void* secondaryPoolPtr
        = reinterpret_cast<char*>(pointers[rowOffset + hostKvCachePoolPointers.stride(1)]) + intraPoolOffset;
    auto* blockOffsets
        = reinterpret_cast<tk::KVBlockArray::DataType*>(static_cast<int32_t*>(kvCacheBlockOffsets.data_ptr())
            + poolIndex * kvCacheBlockOffsets.stride(0) + seqOffset * kvCacheBlockOffsets.stride(1));
    auto const maxBlocksPerSequence = static_cast<int32_t>(kvCacheBlockOffsets.size(-1));
    auto const window = static_cast<int32_t>(attentionWindowSize);
    return tk::KVBlockArray(batchSize, maxBlocksPerSequence, static_cast<int32_t>(tokensPerBlock), sizePerToken, window,
        window, /*sinkTokenLen=*/0, /*canUseOneMoreBlock=*/false, primaryPoolPtr, secondaryPoolPtr, blockOffsets);
}

} // namespace

//! Generation MLA RoPE + FP8 query quantization + FP8 latent-cache append (the FP8-cache, absorption-mode,
//! non-helix subset of trtllm::mla_rope_generation), optionally fused with the DSA indexer K-cache scatter of
//! trtllm::indexer_k_cache_scatter_op. Argument meanings follow those two native ops.
void ropeKvAppend(torch::Tensor const& fused_q, torch::Tensor const& q_pe, torch::Tensor const& latent_cache,
    torch::Tensor const& rotary_cos_sin, torch::Tensor const& cu_q_seqlens, torch::Tensor const& cu_kv_seqlens,
    torch::Tensor const& fmha_scheduler_counter, torch::Tensor const& mla_bmm1_scale,
    torch::Tensor const& mla_bmm2_scale, torch::Tensor const& quant_q_buffer, torch::Tensor const& sequence_length,
    torch::Tensor const& host_context_lengths, int64_t num_contexts, torch::Tensor const& kv_cache_block_offsets,
    torch::Tensor const& host_kv_cache_pool_pointers, torch::Tensor const& host_kv_cache_pool_mapping,
    std::optional<torch::Tensor> const& kv_cache_scale_orig_quant, int64_t layer_idx, int64_t num_heads,
    int64_t tokens_per_block, int64_t attention_window_size, int64_t quant_mode, double q_scaling,
    int64_t qk_nope_head_dim, int64_t qk_rope_head_dim, int64_t kv_lora_rank,
    std::optional<torch::Tensor> const& idx_k_fp8, std::optional<torch::Tensor> const& idx_k_scale,
    std::optional<torch::Tensor> const& idx_k_cache, std::optional<torch::Tensor> const& idx_slot_mapping_fp8,
    std::optional<torch::Tensor> const& idx_slot_mapping_scale, int64_t idx_num_tokens)
{
    TORCH_CHECK(kv_lora_rank == rope_kv_append::kKvLoraRank && qk_rope_head_dim == rope_kv_append::kRopeDim,
        "glm_moe_dsa_rope_kv_append: supports kv_lora_rank 512 and qk_rope_head_dim 64 only");
    auto const kvQuantMode = tc::QuantMode(static_cast<uint32_t>(quant_mode));
    TORCH_CHECK(kvQuantMode.hasFp8KvCache() && !kvQuantMode.hasFp4KvCache(),
        "glm_moe_dsa_rope_kv_append: needs an FP8 MLA cache");
    checkCuda(fused_q, "fused_q");
    checkCuda(q_pe, "q_pe");
    checkCuda(latent_cache, "latent_cache");
    TORCH_CHECK(fused_q.scalar_type() == torch::kBFloat16 && q_pe.scalar_type() == torch::kBFloat16
            && latent_cache.scalar_type() == torch::kBFloat16,
        "glm_moe_dsa_rope_kv_append: needs BF16 fused_q, q_pe and latent_cache");
    TORCH_CHECK(rotary_cos_sin.scalar_type() == torch::kFloat32 && rotary_cos_sin.is_contiguous(),
        "rotary_cos_sin must be packed float32");
    int64_t const cosSinRows = rotary_cos_sin.numel() / (2 * rope_kv_append::kRopeDim);
    TORCH_CHECK(cosSinRows > 0 && cosSinRows <= INT32_MAX, "rotary_cos_sin has no complete position row");
    TORCH_CHECK(q_pe.dim() == 3 && q_pe.stride(2) == 1, "q_pe must be [tokens, heads, rope] with unit inner stride");
    TORCH_CHECK(latent_cache.dim() == 2 && latent_cache.size(1) == rope_kv_append::kLatentDim
            && latent_cache.stride(1) == 1 && latent_cache.stride(0) == rope_kv_append::kLatentDim,
        "latent_cache must be packed [tokens, 576]");
    TORCH_CHECK(fused_q.is_contiguous() && quant_q_buffer.is_contiguous(), "fused_q and quant_q_buffer must be packed");
    TORCH_CHECK(host_kv_cache_pool_mapping.dim() >= 1, "host_kv_cache_pool_mapping is required");

    int32_t const numTokens = static_cast<int32_t>(latent_cache.size(0));
    int32_t const numSeqs = static_cast<int32_t>(host_context_lengths.size(0));
    int32_t const numGenerations = numSeqs - static_cast<int32_t>(num_contexts);
    TORCH_CHECK(numGenerations > 0, "glm_moe_dsa_rope_kv_append: needs at least one generation request");
    TORCH_CHECK(numTokens % numGenerations == 0, "MLA can only support input sequences with the same sequence length.");
    TORCH_CHECK(fused_q.numel() >= static_cast<int64_t>(numTokens) * num_heads * rope_kv_append::kLatentDim,
        "fused_q is smaller than [tokens, heads, 576]");
    TORCH_CHECK(quant_q_buffer.numel() * quant_q_buffer.element_size()
            >= static_cast<int64_t>(numTokens) * num_heads * rope_kv_append::kLatentDim,
        "quant_q_buffer is smaller than [tokens, heads, 576] bytes");
    TORCH_CHECK(mla_bmm1_scale.numel() >= 2 && mla_bmm2_scale.numel() >= 1, "bmm scale buffers too small");
    TORCH_CHECK(cu_q_seqlens.numel() >= numGenerations + 1 && cu_kv_seqlens.numel() >= numGenerations + 1,
        "cu_seqlens buffers too small");

    auto stream = at::cuda::getCurrentCUDAStream(latent_cache.get_device());
    rope_kv_append::RopeAppendParams params{};
    params.fusedQ = reinterpret_cast<__nv_bfloat16 const*>(fused_q.data_ptr());
    params.qPe = reinterpret_cast<__nv_bfloat16 const*>(q_pe.data_ptr());
    params.qPeLd = static_cast<int32_t>(q_pe.stride(1));
    params.qPeStride = static_cast<int32_t>(q_pe.stride(0));
    params.quantQ = reinterpret_cast<uint8_t*>(quant_q_buffer.data_ptr());
    params.quantScaleQ = nullptr;
    params.latent = reinterpret_cast<__nv_bfloat16 const*>(latent_cache.data_ptr());
    params.cosSin = reinterpret_cast<float2 const*>(rotary_cos_sin.data_ptr());
    params.cosSinRows = static_cast<int32_t>(cosSinRows);
    params.kvLens = sequence_length.slice(0, num_contexts).data_ptr<int32_t>();
    params.quantScaleKv
        = kv_cache_scale_orig_quant.has_value() ? kv_cache_scale_orig_quant->data_ptr<float>() : nullptr;
    params.kvCache = buildFp8MlaKvBlockArray(kv_cache_block_offsets, host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping, layer_idx, numGenerations, tokens_per_block, attention_window_size, num_contexts);
    params.cuQSeqlens = cu_q_seqlens.data_ptr<int32_t>();
    params.cuKvSeqlens = cu_kv_seqlens.data_ptr<int32_t>();
    params.fmhaTileCounter = reinterpret_cast<uint32_t*>(fmha_scheduler_counter.data_ptr());
    params.bmm1Scale = mla_bmm1_scale.data_ptr<float>();
    params.bmm2Scale = mla_bmm2_scale.data_ptr<float>();
    params.dequantScaleQ = nullptr;
    params.dequantScaleKv = nullptr;
    params.quantScaleO = nullptr;
    // Same expression as the native op so the folded FMHA scales match bit for bit.
    params.hostBmm1Scale = 1.F / (q_scaling * std::sqrt(static_cast<float>(qk_nope_head_dim + qk_rope_head_dim)));
    params.numTokens = numTokens;
    params.seqLen = numTokens / numGenerations;
    params.numHeads = static_cast<int32_t>(num_heads);

    if (idx_num_tokens > 0)
    {
        TORCH_CHECK(idx_k_fp8.has_value() && idx_k_scale.has_value() && idx_k_cache.has_value()
                && idx_slot_mapping_fp8.has_value() && idx_slot_mapping_scale.has_value(),
            "indexer scatter needs k_fp8, k_scale, k_cache and both slot mappings");
        auto const& kFp8 = idx_k_fp8.value();
        auto const& kScale = idx_k_scale.value();
        auto const& kCache = idx_k_cache.value();
        auto const& slotFp8 = idx_slot_mapping_fp8.value();
        auto const& slotScale = idx_slot_mapping_scale.value();
        checkCuda(kFp8, "k_fp8");
        checkCuda(kScale, "k_scale");
        checkCuda(kCache, "k_cache");
        TORCH_CHECK(kFp8.dim() == 2 && kFp8.element_size() == 1 && kFp8.is_contiguous(), "k_fp8 must be packed bytes");
        TORCH_CHECK(kScale.dim() == 2 && kScale.element_size() == 4 && kScale.is_contiguous()
                && kScale.size(1) * kScale.element_size() == 4,
            "k_scale must be one packed 4-byte word per token");
        TORCH_CHECK(kCache.dim() == 4 && kCache.size(2) == 1, "k_cache must be [blocks, block_size, 1, bytes]");
        TORCH_CHECK(slotFp8.scalar_type() == torch::kInt64 && slotScale.scalar_type() == torch::kInt64
                && slotFp8.is_contiguous() && slotScale.is_contiguous(),
            "slot mappings must be packed int64");
        int32_t const headBytes = static_cast<int32_t>(kFp8.size(1));
        TORCH_CHECK(headBytes == 128 || headBytes == 64, "indexer k head bytes must be 128 or 64");
        TORCH_CHECK(kFp8.size(0) >= idx_num_tokens && kScale.size(0) >= idx_num_tokens
                && slotFp8.numel() >= idx_num_tokens && slotScale.numel() >= idx_num_tokens,
            "indexer inputs are shorter than idx_num_tokens");
        params.idxK = reinterpret_cast<uint8_t const*>(kFp8.data_ptr());
        params.idxScale = reinterpret_cast<uint8_t const*>(kScale.data_ptr());
        params.idxCache = reinterpret_cast<uint8_t*>(kCache.data_ptr());
        params.idxSlotData = slotFp8.data_ptr<int64_t>();
        params.idxSlotScale = slotScale.data_ptr<int64_t>();
        params.idxNumTokens = static_cast<int32_t>(idx_num_tokens);
        params.idxHeadBytes = headBytes;
        for (int32_t d = 0; d < 4; ++d)
        {
            params.idxDims[d] = static_cast<int32_t>(kCache.size(d));
            params.idxStrides[d] = kCache.stride(d);
        }
    }

    rope_kv_append::invokeRopeAppendScatter(params, tc::getEnvEnablePDL(), stream);
}

//! Local top-k token ids -> global KV-pool indices (values of trtllm::convert_req_index_to_global and its grouped
//! variant), optionally fused with the native top-k reuse copy. Returns [group, rows, topk] when layer_ids is given,
//! otherwise [rows, topk].
torch::Tensor topkIndexBridge(torch::Tensor const& topk, int64_t num_rows, torch::Tensor const& req_idx,
    int64_t req_idx_bias, torch::Tensor const& block_table, int64_t block_size, int64_t stride_factor,
    std::optional<torch::Tensor> const& layer_ids, int64_t layer_id, std::optional<torch::Tensor> const& copy_dst,
    int64_t copy_row_start, int64_t copy_rows, int64_t copy_mode, int64_t next_n,
    std::optional<torch::Tensor> const& num_accepted)
{
    checkCuda(topk, "topk");
    checkCuda(req_idx, "req_idx");
    checkCuda(block_table, "block_table");
    TORCH_CHECK(topk.scalar_type() == torch::kInt32 && topk.dim() == 2 && topk.stride(1) == 1,
        "topk must be int32 [rows, topk] with unit column stride");
    TORCH_CHECK(req_idx.scalar_type() == torch::kInt32 && req_idx.dim() == 1 && req_idx.is_contiguous(),
        "req_idx must be packed int32");
    TORCH_CHECK(block_table.scalar_type() == torch::kInt32 && block_table.dim() == 2, "block_table must be int32 2-D");
    TORCH_CHECK(copy_mode == 0 || copy_mode == 1, "copy_mode must be 0 or 1");
    int64_t const numTopk = topk.size(1);
    TORCH_CHECK(req_idx.size(0) >= num_rows, "req_idx shorter than num_rows");
    TORCH_CHECK(copy_mode == 1 || topk.size(0) >= num_rows, "topk shorter than num_rows");
    TORCH_CHECK(copy_mode == 0 || topk.size(0) >= copy_rows * next_n, "topk shorter than copy_rows * next_n");
    TORCH_CHECK(copy_mode == 1 || copy_rows <= num_rows, "copy_rows exceeds num_rows");

    int32_t groupSize = 1;
    int32_t const* layerIdsPtr = nullptr;
    if (layer_ids.has_value())
    {
        TORCH_CHECK(layer_ids->scalar_type() == torch::kInt32 && layer_ids->dim() == 1 && layer_ids->is_contiguous(),
            "layer_ids must be packed int32");
        groupSize = static_cast<int32_t>(layer_ids->size(0));
        layerIdsPtr = layer_ids->data_ptr<int32_t>();
    }
    torch::Tensor out = layer_ids.has_value() ? torch::empty({groupSize, num_rows, numTopk}, topk.options())
                                              : torch::empty({num_rows, numTopk}, topk.options());

    rope_kv_append::TopkBridgeParams params{};
    params.topk = topk.data_ptr<int32_t>();
    params.topkStride0 = topk.stride(0);
    params.numRows = static_cast<int32_t>(num_rows);
    params.numTopk = static_cast<int32_t>(numTopk);
    params.reqIdx = req_idx.data_ptr<int32_t>();
    params.reqIdxBias = static_cast<int32_t>(req_idx_bias);
    params.blockTable = block_table.data_ptr<int32_t>();
    params.btStride0 = block_table.stride(0);
    params.btStride1 = block_table.stride(1);
    params.maxBlocksPerReq = static_cast<int32_t>(block_table.size(1));
    params.blockSize = static_cast<int32_t>(block_size);
    params.strideFactor = static_cast<int32_t>(stride_factor);
    params.layerIds = layerIdsPtr;
    params.layerId = static_cast<int32_t>(layer_id);
    params.groupSize = groupSize;
    params.out = out.data_ptr<int32_t>();
    params.outStrideG = layer_ids.has_value() ? out.stride(0) : 0;
    params.outStride0 = layer_ids.has_value() ? out.stride(1) : out.stride(0);
    if (copy_dst.has_value() && copy_rows > 0)
    {
        auto const& dst = copy_dst.value();
        checkCuda(dst, "copy_dst");
        TORCH_CHECK(dst.scalar_type() == torch::kInt32 && dst.dim() == 2 && dst.stride(1) == 1 && dst.size(1) >= numTopk
                && dst.size(0) >= copy_row_start + copy_rows,
            "copy_dst must be int32 [>= start + rows, >= topk] with unit column stride");
        params.copyDst = dst.data_ptr<int32_t>();
        params.copyStride0 = dst.stride(0);
        params.copyRows = static_cast<int32_t>(copy_rows);
        params.copyRowStart = static_cast<int32_t>(copy_row_start);
        params.copyMode = static_cast<int32_t>(copy_mode);
        params.nextN = static_cast<int32_t>(next_n);
        if (copy_mode == 1 && num_accepted.has_value())
        {
            auto const& acc = num_accepted.value();
            checkCuda(acc, "num_accepted");
            TORCH_CHECK((acc.scalar_type() == torch::kInt32 || acc.scalar_type() == torch::kInt64) && acc.dim() == 1
                    && acc.is_contiguous() && acc.size(0) >= copy_rows,
                "num_accepted must be packed int32/int64 with at least copy_rows entries");
            params.numAccepted = acc.data_ptr();
            params.numAcceptedIs64 = acc.scalar_type() == torch::kInt64;
        }
    }

    auto stream = at::cuda::getCurrentCUDAStream(topk.get_device());
    rope_kv_append::invokeTopkBridge(params, tc::getEnvEnablePDL(), stream);
    return out;
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "rope_kv_append(Tensor fused_q, Tensor q_pe, Tensor latent_cache, Tensor rotary_cos_sin, "
        "Tensor(a!) cu_q_seqlens, Tensor(b!) cu_kv_seqlens, Tensor(c!) fmha_scheduler_counter, "
        "Tensor(d!) mla_bmm1_scale, Tensor(e!) mla_bmm2_scale, Tensor(f!) quant_q_buffer, Tensor sequence_length, "
        "Tensor host_context_lengths, int num_contexts, Tensor kv_cache_block_offsets, "
        "Tensor host_kv_cache_pool_pointers, Tensor host_kv_cache_pool_mapping, Tensor? kv_cache_scale_orig_quant, "
        "int layer_idx, int num_heads, int tokens_per_block, int attention_window_size, int quant_mode, "
        "float q_scaling, int qk_nope_head_dim, int qk_rope_head_dim, int kv_lora_rank, Tensor? idx_k_fp8, "
        "Tensor? idx_k_scale, Tensor(g!)? idx_k_cache, Tensor? idx_slot_mapping_fp8, Tensor? idx_slot_mapping_scale, "
        "int idx_num_tokens) -> ()");
    m.def(
        "topk_index_bridge(Tensor topk, int num_rows, Tensor req_idx, int req_idx_bias, Tensor "
        "block_table, "
        "int block_size, int stride_factor, Tensor? layer_ids, int layer_id, Tensor(a!)? copy_dst, "
        "int copy_row_start, int copy_rows, int copy_mode, int next_n, Tensor? num_accepted) -> Tensor");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("rope_kv_append", &TRTLLM_MV2_NS::torch_ext::ropeKvAppend);
    m.impl("topk_index_bridge", &TRTLLM_MV2_NS::torch_ext::topkIndexBridge);
}
