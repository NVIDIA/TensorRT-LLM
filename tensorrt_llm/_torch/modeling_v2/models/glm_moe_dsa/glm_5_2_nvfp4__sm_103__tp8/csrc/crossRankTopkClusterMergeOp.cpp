/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/csrc/crossRankTopkClusterMerge.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

#include <optional>
#include <vector>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{
namespace tk = TRTLLM_MV2_NS::cross_rank_topk_cluster_merge;

void crossRankTopkClusterMerge(at::Tensor const& logits, at::Tensor const& kvLens, at::Tensor& out, at::Tensor& state,
    int64_t mcWorkspace, int64_t ucWorkspace, int64_t rank, int64_t worldSize, int64_t nextN,
    std::optional<at::Tensor> const& debugTimes)
{
    TORCH_CHECK(logits.is_cuda() && kvLens.is_cuda() && out.is_cuda() && state.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(logits.scalar_type() == at::kFloat && logits.dim() == 2 && logits.stride(1) == 1,
        "logits must be a row-major [rows, local_capacity] float32 view");
    TORCH_CHECK(kvLens.scalar_type() == at::kInt && kvLens.numel() >= 1, "kv_lens must be int32 with >= 1 element");
    TORCH_CHECK(out.scalar_type() == at::kInt && out.dim() == 2 && out.is_contiguous() && out.size(1) == tk::kTopK,
        "out must be a contiguous int32 [rows, ", tk::kTopK, "] tensor");
    TORCH_CHECK(out.size(0) == logits.size(0), "out and logits must have the same rows");
    TORCH_CHECK(logits.size(0) <= tk::kMaxRows, "at most ", tk::kMaxRows, " rows");
    TORCH_CHECK(state.scalar_type() == at::kInt && state.numel() >= tk::kStateWords && state.is_contiguous(),
        "state must be a contiguous int32 tensor of >= ", tk::kStateWords, " words");
    TORCH_CHECK(worldSize >= 1 && worldSize <= tk::kMaxRanks && rank >= 0 && rank < worldSize, "bad rank/world");
    TORCH_CHECK(nextN >= 1 && nextN == logits.size(0), "next_n must equal the number of rows (one request)");
    TORCH_CHECK(logits.size(1) <= tk::cpTopkMaxLocalCapacity(static_cast<int32_t>(worldSize)), "local capacity ",
        logits.size(1), " exceeds the kernel limit ", tk::cpTopkMaxLocalCapacity(static_cast<int32_t>(worldSize)));
    TORCH_CHECK(mcWorkspace != 0 && ucWorkspace != 0 && (mcWorkspace % 16) == 0 && (ucWorkspace % 16) == 0,
        "workspace addresses must be non-null and 16-byte aligned");

    tk::CpTopkParams params{};
    params.logits = logits.data_ptr<float>();
    params.logitsStride = logits.stride(0);
    params.localCapacity = static_cast<int32_t>(logits.size(1));
    params.kvLens = kvLens.data_ptr<int32_t>();
    params.out = out.data_ptr<int32_t>();
    params.numRows = static_cast<int32_t>(logits.size(0));
    params.nextN = static_cast<int32_t>(nextN);
    params.rank = static_cast<int32_t>(rank);
    params.worldSize = static_cast<int32_t>(worldSize);
    params.mcWorkspace = reinterpret_cast<uint32_t*>(mcWorkspace);
    params.ucWorkspace = reinterpret_cast<uint32_t*>(ucWorkspace);
    params.state = state.data_ptr<int32_t>();
    if (debugTimes.has_value())
    {
        TORCH_CHECK(debugTimes->is_cuda() && debugTimes->scalar_type() == at::kLong && debugTimes->is_contiguous()
                && debugTimes->numel() >= logits.size(0) * tk::kSourcesPerRank * tk::kDebugSlots,
            "debug_times must be a contiguous int64 CUDA tensor of rows x ", tk::kSourcesPerRank, " x ",
            tk::kDebugSlots);
        params.debugTimes = debugTimes->data_ptr<int64_t>();
    }
    tk::invokeCpTopk(params, at::cuda::getCurrentCUDAStream(logits.get_device()));
}

//! [workspace words, top-k, max ranks, max rows, page tokens, sources per rank, first-round slot capacity, max local
//! capacity, state words, Lamport buffers, default timeout (ms), resident clusters of kMaxRows-row launches]
std::vector<int64_t> crossRankTopkClusterMergeWorkspaceInfo(int64_t worldSize)
{
    return {tk::kWorkspaceWords, tk::kTopK, tk::kMaxRanks, tk::kMaxRows, tk::kPageTokens, tk::kSourcesPerRank,
        tk::kSourceCapacity, tk::cpTopkMaxLocalCapacity(static_cast<int32_t>(worldSize)), tk::kStateWords,
        tk::kNumLamportBuffers, tk::cpTopkDefaultTimeoutMs(),
        tk::cpTopkMaxActiveClusters(static_cast<int32_t>(worldSize))};
}

} // namespace
} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "cross_rank_topk_cluster_merge(Tensor logits, Tensor kv_lens, Tensor(a!) out, Tensor(b!) state, "
        "int mc_workspace, "
        "int uc_workspace, int rank, int world_size, int next_n, Tensor(c!)? debug_times=None) -> ()");
    m.def("cross_rank_topk_cluster_merge_workspace_info(int world_size) -> int[]");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("cross_rank_topk_cluster_merge", &TRTLLM_MV2_NS::torch_ext::crossRankTopkClusterMerge);
}

TRTLLM_MV2_LIBRARY_IMPL(CompositeExplicitAutograd, m)
{
    m.impl("cross_rank_topk_cluster_merge_workspace_info",
        &TRTLLM_MV2_NS::torch_ext::crossRankTopkClusterMergeWorkspaceInfo);
}
