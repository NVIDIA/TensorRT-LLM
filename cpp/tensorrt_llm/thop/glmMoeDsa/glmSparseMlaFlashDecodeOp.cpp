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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/glmMoeDsa/glmSparseMlaFlashDecode.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <optional>

namespace th = torch;
namespace tk = tensorrt_llm::kernels::glm_moe_dsa;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
// The grid spans the query tokens in its y dimension.
constexpr int64_t kMaxQueryTokens = 65535;
constexpr int64_t kBulkCopyAlignment = 16;

bool isE4m3Storage(th::Tensor const& t)
{
    return t.scalar_type() == th::kFloat8_e4m3fn || t.scalar_type() == th::kByte;
}

bool isBulkCopyAligned(th::Tensor const& t)
{
    return reinterpret_cast<uintptr_t>(t.data_ptr()) % kBulkCopyAlignment == 0;
}

} // namespace

// Generation sparse MLA of GLM-5.2 at 4 query heads per rank, with the v_b BMM folded in: writes
// output[t][h * 256 + v] = sum_d bf16(attention(q[t, h]))[d] * v_b_proj[h][v][d] for the first q.size(0) rows of
// output. With block_table, indices hold token positions that the kernel maps to pool rows as
// convert_req_index_to_global does (block_table[s][p / tokens_per_block] * page_index_scale * tokens_per_block
// + p % tokens_per_block + layer_offset * tokens_per_block); without it, indices hold pool rows.
void glmSparseMlaFlashDecode(th::Tensor const& q, th::Tensor const& kvPool, th::Tensor const& indices,
    th::Tensor const& seqLensKv, th::Tensor const& mlaBmm1Scale, th::Tensor const& mlaBmm2Scale,
    th::Tensor const& vBProj, th::Tensor& output, int64_t seqLenQ, int64_t topK,
    std::optional<th::Tensor> const& blockTable, int64_t tokensPerBlock, int64_t pageIndexScale, int64_t layerOffset,
    bool gatherBeforeWait)
{
    TORCH_CHECK(tensorrt_llm::common::isSM100Family(), "glm_sparse_mla_flash_decode needs an SM100-family GPU");
    TORCH_CHECK(q.is_cuda() && kvPool.is_cuda() && indices.is_cuda() && seqLensKv.is_cuda() && mlaBmm1Scale.is_cuda()
            && mlaBmm2Scale.is_cuda() && vBProj.is_cuda() && output.is_cuda(),
        "glm_sparse_mla_flash_decode takes CUDA tensors");
    auto const device = q.device();
    TORCH_CHECK(kvPool.device() == device && indices.device() == device && seqLensKv.device() == device
            && mlaBmm1Scale.device() == device && mlaBmm2Scale.device() == device && vBProj.device() == device
            && output.device() == device,
        "glm_sparse_mla_flash_decode takes tensors on one device");

    TORCH_CHECK(isE4m3Storage(q), "q must hold E4M3 values (float8_e4m3fn or uint8)");
    TORCH_CHECK(q.dim() == 3 && q.size(1) == tk::kGlmSparseMlaNumHeads && q.size(2) == tk::kGlmSparseMlaHeadDimQk
            && q.is_contiguous() && isBulkCopyAligned(q),
        "q must be a contiguous, 16-byte aligned [tokens, ", tk::kGlmSparseMlaNumHeads, ", ",
        tk::kGlmSparseMlaHeadDimQk, "] tensor");
    int64_t const numTokens = q.size(0);
    TORCH_CHECK(numTokens <= kMaxQueryTokens, "glm_sparse_mla_flash_decode takes at most ", kMaxQueryTokens,
        " query tokens, got ", numTokens);
    TORCH_CHECK(seqLenQ > 0 && seqLenQ <= kInt32Max && numTokens % seqLenQ == 0,
        "seq_len_q must be positive and divide the query tokens; got ", seqLenQ, " for ", numTokens, " tokens");
    int64_t const numSeqs = numTokens / seqLenQ;

    TORCH_CHECK(isE4m3Storage(kvPool), "kv_pool must hold E4M3 values (float8_e4m3fn or uint8)");
    TORCH_CHECK(kvPool.is_contiguous() && isBulkCopyAligned(kvPool) && kvPool.numel() % tk::kGlmSparseMlaHeadDimQk == 0,
        "kv_pool must be a contiguous, 16-byte aligned pool of ", tk::kGlmSparseMlaHeadDimQk, "-byte token rows");
    TORCH_CHECK(kvPool.numel() / tk::kGlmSparseMlaHeadDimQk <= kInt32Max, "kv_pool rows must be int32-addressable");

    TORCH_CHECK(topK > 0 && topK <= tk::kGlmSparseMlaMaxTopK, "top_k must be in [1, ", tk::kGlmSparseMlaMaxTopK,
        "], got ", topK);
    TORCH_CHECK(indices.scalar_type() == th::kInt32 && indices.dim() == 2 && indices.size(0) == numTokens
            && indices.size(1) >= topK && indices.stride(1) == 1 && indices.stride(0) >= topK
            && indices.stride(0) <= kInt32Max,
        "indices must be an int32 [tokens, >= top_k] tensor with unit column stride");
    TORCH_CHECK(seqLensKv.scalar_type() == th::kInt32 && seqLensKv.dim() == 1 && seqLensKv.size(0) >= numSeqs
            && seqLensKv.is_contiguous(),
        "seq_lens_kv must be a contiguous int32 tensor with one entry per sequence");
    TORCH_CHECK(mlaBmm1Scale.scalar_type() == th::kFloat32 && mlaBmm1Scale.is_contiguous() && mlaBmm1Scale.numel() >= 2,
        "mla_bmm1_scale must hold two contiguous float32 values");
    TORCH_CHECK(mlaBmm2Scale.scalar_type() == th::kFloat32 && mlaBmm2Scale.numel() >= 1,
        "mla_bmm2_scale must hold a float32 value");

    TORCH_CHECK(vBProj.scalar_type() == th::kBFloat16 && vBProj.dim() == 3
            && vBProj.size(0) == tk::kGlmSparseMlaNumHeads && vBProj.size(1) == tk::kGlmSparseMlaVHeadDim
            && vBProj.size(2) == tk::kGlmSparseMlaHeadDimV && vBProj.is_contiguous() && isBulkCopyAligned(vBProj),
        "v_b_proj must be a contiguous, 16-byte aligned BF16 [", tk::kGlmSparseMlaNumHeads, ", ",
        tk::kGlmSparseMlaVHeadDim, ", ", tk::kGlmSparseMlaHeadDimV, "] tensor");
    TORCH_CHECK(output.scalar_type() == th::kBFloat16 && output.dim() == 2 && output.size(0) >= numTokens
            && output.size(1) == tk::kGlmSparseMlaNumHeads * tk::kGlmSparseMlaVHeadDim && output.stride(1) == 1,
        "output must be a BF16 [>= tokens, ", tk::kGlmSparseMlaNumHeads * tk::kGlmSparseMlaVHeadDim,
        "] tensor with unit column stride");

    tk::GlmSparseMlaFlashDecodeParams params{};
    params.q = q.data_ptr();
    params.kvPool = kvPool.data_ptr();
    params.indices = indices.data_ptr<int32_t>();
    params.seqLensKv = seqLensKv.data_ptr<int32_t>();
    params.scaleSoftmaxLog2 = mlaBmm1Scale.data_ptr<float>() + 1;
    params.outputScale = mlaBmm2Scale.data_ptr<float>();
    params.vbProj = vBProj.data_ptr();
    params.out = output.data_ptr();
    params.outStride = output.stride(0);
    params.numTokensQ = static_cast<int32_t>(numTokens);
    params.seqLenQ = static_cast<int32_t>(seqLenQ);
    params.indicesStride = static_cast<int32_t>(indices.stride(0));
    params.topK = static_cast<int32_t>(topK);
    params.gatherBeforeWait = gatherBeforeWait;
    if (blockTable.has_value())
    {
        auto const& table = blockTable.value();
        TORCH_CHECK(table.is_cuda() && table.device() == device && table.scalar_type() == th::kInt32 && table.dim() == 2
                && table.stride(1) == 1 && table.size(0) >= numSeqs && table.size(1) > 0 && table.size(1) <= kInt32Max
                && table.stride(0) >= table.size(1),
            "block_table must be an int32 CUDA [sequences, max_blocks] tensor with unit column stride");
        TORCH_CHECK(tokensPerBlock > 0 && pageIndexScale > 0 && layerOffset >= 0,
            "invalid block_table geometry: ", "tokens_per_block ", tokensPerBlock, ", page_index_scale ",
            pageIndexScale, ", layer_offset ", layerOffset);
        int64_t const strideFactor = pageIndexScale * tokensPerBlock;
        int64_t const rowOffset = layerOffset * tokensPerBlock;
        TORCH_CHECK(strideFactor <= kInt32Max && rowOffset <= kInt32Max, "block_table rows must be int32-addressable");
        params.remapBlockTable = table.data_ptr<int32_t>();
        params.remapBlockTableStride = table.stride(0);
        params.remapMaxBlocks = static_cast<int32_t>(table.size(1));
        params.remapTokensPerBlock = static_cast<int32_t>(tokensPerBlock);
        params.remapStrideFactor = static_cast<int32_t>(strideFactor);
        params.remapLayerOffset = static_cast<int32_t>(rowOffset);
    }
    else
    {
        TORCH_CHECK(!gatherBeforeWait, "gather_before_wait needs block_table");
    }

    auto const stream = at::cuda::getCurrentCUDAStream(q.get_device());
    tk::invokeGlmSparseMlaFlashDecode(params, stream);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "glm_sparse_mla_flash_decode(Tensor q, Tensor kv_pool, Tensor indices, Tensor seq_lens_kv, "
        "Tensor mla_bmm1_scale, Tensor mla_bmm2_scale, Tensor v_b_proj, Tensor(a!) output, int seq_len_q, int top_k, "
        "Tensor? block_table=None, int tokens_per_block=0, int page_index_scale=0, int layer_offset=0, "
        "bool gather_before_wait=False) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("glm_sparse_mla_flash_decode", &tensorrt_llm::torch_ext::glmSparseMlaFlashDecode);
}
