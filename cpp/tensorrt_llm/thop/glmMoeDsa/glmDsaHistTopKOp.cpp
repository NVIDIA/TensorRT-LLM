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

#include "tensorrt_llm/kernels/glmMoeDsa/glmDsaHistTopK.h"
#include "tensorrt_llm/runtime/mcastGPUBuffer.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>

#include <cstdint>

namespace tkh = tensorrt_llm::kernels::glm_moe_dsa::hist_topk;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Exact top-2048 of one generation request's key-sharded GLM DSA indexer logits (glmDsaHistTopK.h): this rank's shard
// logits [num_rows, >= shard size] FP32 in, the request-local token indices [num_rows, 2048] int32 out, identical on
// every rank of the group. comm_buffer is this rank's unicast view of a McastGPUBuffer workspace of
// kWorkspaceWords int32 words, all -1 at creation; state holds kStateWords int32, zero at creation; both are private
// to this op.
void glmDsaHistTopK(at::Tensor const& logits, at::Tensor const& kvLens, at::Tensor& out, at::Tensor& commBuffer,
    at::Tensor& state, int64_t nextN, int64_t groupSize)
{
    auto mcastMem = tensorrt_llm::runtime::getMcastDevMemBuffer(commBuffer);
    TORCH_CHECK(mcastMem != nullptr, "glm_dsa_hist_topk: comm_buffer must come from a McastGPUBuffer");
    TORCH_CHECK(mcastMem->isMapped(), "glm_dsa_hist_topk: the MNNVL workspace handles are not attached");
    TORCH_CHECK(commBuffer.data_ptr() == mcastMem->getUnicastPtr(mcastMem->getRank()),
        "glm_dsa_hist_topk: comm_buffer must start at the local unicast allocation base");
    TORCH_CHECK(static_cast<int64_t>(mcastMem->getWorldSize()) == groupSize, "glm_dsa_hist_topk: group_size ",
        groupSize, " does not match the workspace group of ", mcastMem->getWorldSize(), " ranks");
    CHECK_TH_CUDA(logits);
    CHECK_TH_CUDA(kvLens);
    CHECK_TH_CUDA(out);
    CHECK_TH_CUDA(state);
    TORCH_CHECK(logits.scalar_type() == at::ScalarType::Float && logits.dim() == 2 && logits.stride(1) == 1,
        "glm_dsa_hist_topk: logits must be a row-major FP32 matrix");
    TORCH_CHECK(kvLens.scalar_type() == at::ScalarType::Int && kvLens.numel() >= 1, "glm_dsa_hist_topk: kv_lens int32");
    TORCH_CHECK(out.scalar_type() == at::ScalarType::Int && out.dim() == 2 && out.size(1) == tkh::kTopK
            && out.stride(1) == 1 && out.stride(0) == tkh::kTopK,
        "glm_dsa_hist_topk: out must be a contiguous int32 [num_rows, 2048] matrix");
    int64_t const numRows = out.size(0);
    TORCH_CHECK(numRows >= 1 && numRows <= tkh::kMaxRows && logits.size(0) >= numRows && nextN >= numRows,
        "glm_dsa_hist_topk: 1..8 rows of one request");
    TORCH_CHECK(logits.size(1) <= tkh::kMaxShard, "glm_dsa_hist_topk: at most ", tkh::kMaxShard, " shard columns");
    TORCH_CHECK(commBuffer.scalar_type() == at::ScalarType::Int && commBuffer.numel() >= tkh::kWorkspaceWords,
        "glm_dsa_hist_topk: comm_buffer must hold ", tkh::kWorkspaceWords, " int32 words");
    TORCH_CHECK(state.scalar_type() == at::ScalarType::Int && state.numel() >= tkh::kStateWords,
        "glm_dsa_hist_topk: state must hold ", tkh::kStateWords, " int32 words");

    tkh::HistTopKParams params;
    params.logits = logits.data_ptr<float>();
    params.logitsStride = logits.stride(0);
    params.localCapacity = static_cast<int32_t>(logits.size(1));
    params.kvLens = kvLens.data_ptr<int32_t>();
    params.out = out.data_ptr<int32_t>();
    params.numRows = static_cast<int32_t>(numRows);
    params.nextN = static_cast<int32_t>(nextN);
    params.rank = static_cast<int32_t>(mcastMem->getRank());
    params.worldSize = static_cast<int32_t>(groupSize);
    params.mcWorkspace = static_cast<uint32_t*>(mcastMem->getMulticastPtr());
    params.ucWorkspace = reinterpret_cast<uint32_t*>(commBuffer.data_ptr());
    params.state = state.data_ptr<int32_t>();
    tkh::invokeDsaHistTopK(params, at::cuda::getCurrentCUDAStream(logits.get_device()));
}

// Shard metadata of the generation request for this rank (glmDsaHistTopK.h): shard_ctx [1] and shard_block_table
// [1, width] int32, from the request's KV length and block table row.
void glmDsaHistShardMeta(at::Tensor const& kvLens, at::Tensor const& blockTable, at::Tensor& shardCtx,
    at::Tensor& shardBlockTable, int64_t groupSize, int64_t rank, int64_t tokensPerBlock)
{
    CHECK_TH_CUDA(kvLens);
    CHECK_TH_CUDA(blockTable);
    TORCH_CHECK(kvLens.scalar_type() == at::ScalarType::Int && kvLens.numel() >= 1,
        "glm_dsa_hist_shard_meta: kv_lens must be int32");
    TORCH_CHECK(blockTable.scalar_type() == at::ScalarType::Int && blockTable.dim() >= 1 && blockTable.stride(-1) == 1,
        "glm_dsa_hist_shard_meta: block_table must be int32 with contiguous rows");
    TORCH_CHECK(shardCtx.scalar_type() == at::ScalarType::Int && shardCtx.numel() >= 1,
        "glm_dsa_hist_shard_meta: shard_ctx must be int32");
    TORCH_CHECK(shardBlockTable.scalar_type() == at::ScalarType::Int && shardBlockTable.is_contiguous(),
        "glm_dsa_hist_shard_meta: shard_block_table must be contiguous int32");
    tkh::invokeDsaHistShardMeta(kvLens.data_ptr<int32_t>(), blockTable.data_ptr<int32_t>(),
        static_cast<int32_t>(blockTable.size(-1)), static_cast<int32_t>(groupSize), static_cast<int32_t>(rank),
        static_cast<int32_t>(tokensPerBlock), shardCtx.data_ptr<int32_t>(), shardBlockTable.data_ptr<int32_t>(),
        static_cast<int32_t>(shardBlockTable.numel()), at::cuda::getCurrentCUDAStream(kvLens.get_device()));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "glm_dsa_hist_topk(Tensor logits, Tensor kv_lens, Tensor(a!) out, Tensor(b!) comm_buffer, Tensor(c!) state, "
        "int next_n, int group_size) -> ()");
    m.def(
        "glm_dsa_hist_shard_meta(Tensor kv_lens, Tensor block_table, Tensor(a!) shard_ctx, "
        "Tensor(b!) shard_block_table, int group_size, int rank, int tokens_per_block) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("glm_dsa_hist_topk", &tensorrt_llm::torch_ext::glmDsaHistTopK);
    m.impl("glm_dsa_hist_shard_meta", &tensorrt_llm::torch_ext::glmDsaHistShardMeta);
}
