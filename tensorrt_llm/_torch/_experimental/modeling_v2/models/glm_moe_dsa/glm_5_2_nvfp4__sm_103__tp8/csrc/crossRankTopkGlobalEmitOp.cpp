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

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/csrc/crossRankTopkGlobalEmit.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

#include <optional>
#include <vector>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{
namespace tk = TRTLLM_MV2_NS::cross_rank_topk_global_emit;

//! The cluster-merge cross-rank top-k (same arguments as glm_moe_dsa_cross_rank_topk_cluster_merge) that also writes
//! the outputs of the top-k index bridge (trtllm::glm_moe_dsa_topk_index_bridge with the same bridge arguments, applied
//! to `out`): returns the pool rows, [G, rows, top-k] with layer_ids (G = its length) or [rows, top-k] without, and
//! writes the copy rows into copy_dst.
at::Tensor crossRankTopkGlobalEmit(at::Tensor const& logits, at::Tensor const& kvLens, at::Tensor& out,
    at::Tensor& state, int64_t mcWorkspace, int64_t ucWorkspace, int64_t rank, int64_t worldSize, int64_t nextN,
    at::Tensor const& reqIdx, int64_t reqIdxBias, at::Tensor const& blockTable, int64_t blockSize, int64_t strideFactor,
    std::optional<at::Tensor> const& layerIds, int64_t layerId, std::optional<at::Tensor> const& copyDst,
    int64_t copyRowStart, int64_t copyRows, int64_t copyMode, int64_t copyNextN,
    std::optional<at::Tensor> const& numAccepted, std::optional<at::Tensor> const& debugTimes)
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

    // The bridge inputs, with the rules of trtllm::glm_moe_dsa_topk_index_bridge (topk = out, num_rows = rows).
    int64_t const rows = out.size(0);
    TORCH_CHECK(reqIdx.is_cuda() && blockTable.is_cuda(), "req_idx and block_table must be CUDA tensors");
    TORCH_CHECK(reqIdx.scalar_type() == at::kInt && reqIdx.dim() == 1 && reqIdx.is_contiguous(),
        "req_idx must be packed int32");
    TORCH_CHECK(reqIdx.size(0) >= rows, "req_idx shorter than the rows");
    TORCH_CHECK(blockTable.scalar_type() == at::kInt && blockTable.dim() == 2, "block_table must be int32 2-D");
    TORCH_CHECK(blockSize == tk::kPageTokens, "block_size must be ", tk::kPageTokens, " (the CP page)");
    TORCH_CHECK(copyMode == 0 || copyMode == 1, "copy_mode must be 0 or 1");
    TORCH_CHECK(copyNextN >= 1 && copyRows >= 0 && copyRowStart >= 0, "bad copy rows");
    TORCH_CHECK(copyMode == 1 || copyRows <= rows, "copy_rows exceeds the rows");
    TORCH_CHECK(copyMode == 0 || copyRows * copyNextN <= rows, "copy_rows * copy_next_n exceeds the rows");

    int32_t groupSize = 1;
    int32_t const* layerIdsPtr = nullptr;
    if (layerIds.has_value())
    {
        TORCH_CHECK(layerIds->is_cuda() && layerIds->scalar_type() == at::kInt && layerIds->dim() == 1
                && layerIds->is_contiguous(),
            "layer_ids must be packed int32 on the GPU");
        groupSize = static_cast<int32_t>(layerIds->size(0));
        TORCH_CHECK(groupSize >= 1 && groupSize <= tk::kMaxGroupSize,
            "glm_moe_dsa_cross_rank_topk_global_emit: remap group sizes must be 1..", tk::kMaxGroupSize, ", got ",
            groupSize);
        layerIdsPtr = layerIds->data_ptr<int32_t>();
    }
    int64_t const topK = tk::kTopK;
    at::Tensor pool = layerIds.has_value() ? at::empty({static_cast<int64_t>(groupSize), rows, topK}, out.options())
                                           : at::empty({rows, topK}, out.options());

    tk::CpTopkParams params{};
    params.logits = logits.data_ptr<float>();
    params.logitsStride = logits.stride(0);
    params.localCapacity = static_cast<int32_t>(logits.size(1));
    params.kvLens = kvLens.data_ptr<int32_t>();
    params.out = out.data_ptr<int32_t>();
    params.numRows = static_cast<int32_t>(rows);
    params.nextN = static_cast<int32_t>(nextN);
    params.rank = static_cast<int32_t>(rank);
    params.worldSize = static_cast<int32_t>(worldSize);
    params.mcWorkspace = reinterpret_cast<uint32_t*>(mcWorkspace);
    params.ucWorkspace = reinterpret_cast<uint32_t*>(ucWorkspace);
    params.state = state.data_ptr<int32_t>();

    tk::CpTopkBridgeParams& b = params.bridge;
    b.out = pool.data_ptr<int32_t>();
    b.outStrideG = layerIds.has_value() ? pool.stride(0) : 0;
    b.outStride0 = layerIds.has_value() ? pool.stride(1) : pool.stride(0);
    b.groupSize = groupSize;
    b.layerIds = layerIdsPtr;
    b.layerId = static_cast<int32_t>(layerId);
    b.reqIdx = reqIdx.data_ptr<int32_t>();
    b.reqIdxBias = static_cast<int32_t>(reqIdxBias);
    b.blockTable = blockTable.data_ptr<int32_t>();
    b.btStride0 = blockTable.stride(0);
    b.btStride1 = blockTable.stride(1);
    b.maxBlocksPerReq = static_cast<int32_t>(blockTable.size(1));
    b.strideFactor = static_cast<int32_t>(strideFactor);
    if (copyDst.has_value() && copyRows > 0)
    {
        auto const& dst = copyDst.value();
        TORCH_CHECK(dst.is_cuda() && dst.scalar_type() == at::kInt && dst.dim() == 2 && dst.stride(1) == 1
                && dst.size(1) >= tk::kTopK && dst.size(0) >= copyRowStart + copyRows,
            "copy_dst must be int32 [>= start + rows, >= top-k] with unit column stride");
        b.copyDst = dst.data_ptr<int32_t>();
        b.copyStride0 = dst.stride(0);
        b.copyRows = static_cast<int32_t>(copyRows);
        b.copyRowStart = static_cast<int32_t>(copyRowStart);
        b.copyMode = static_cast<int32_t>(copyMode);
        b.copyNextN = static_cast<int32_t>(copyNextN);
        if (copyMode == 1 && numAccepted.has_value())
        {
            auto const& acc = numAccepted.value();
            TORCH_CHECK(acc.is_cuda() && (acc.scalar_type() == at::kInt || acc.scalar_type() == at::kLong)
                    && acc.dim() == 1 && acc.is_contiguous() && acc.size(0) >= copyRows,
                "num_accepted must be packed int32/int64 with at least copy_rows entries");
            b.numAccepted = acc.data_ptr();
            b.numAcceptedIs64 = acc.scalar_type() == at::kLong;
        }
    }
    if (debugTimes.has_value())
    {
        TORCH_CHECK(debugTimes->is_cuda() && debugTimes->scalar_type() == at::kLong && debugTimes->is_contiguous()
                && debugTimes->numel() >= logits.size(0) * tk::kSourcesPerRank * tk::kDebugSlots,
            "debug_times must be a contiguous int64 CUDA tensor of rows x ", tk::kSourcesPerRank, " x ",
            tk::kDebugSlots);
        params.debugTimes = debugTimes->data_ptr<int64_t>();
    }
    tk::invokeCpTopk(params, at::cuda::getCurrentCUDAStream(logits.get_device()));
    return pool;
}

//! [workspace words, top-k, max ranks, max rows, page tokens, sources per rank, first-round slot capacity, max local
//! capacity, state words, Lamport buffers, default timeout (ms), resident clusters of kMaxRows-row launches, max remap
//! group size]
std::vector<int64_t> crossRankTopkGlobalEmitWorkspaceInfo(int64_t worldSize)
{
    return {tk::kWorkspaceWords, tk::kTopK, tk::kMaxRanks, tk::kMaxRows, tk::kPageTokens, tk::kSourcesPerRank,
        tk::kSourceCapacity, tk::cpTopkMaxLocalCapacity(static_cast<int32_t>(worldSize)), tk::kStateWords,
        tk::kNumLamportBuffers, tk::cpTopkDefaultTimeoutMs(),
        tk::cpTopkMaxActiveClusters(static_cast<int32_t>(worldSize)), tk::kMaxGroupSize};
}

} // namespace
} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "cross_rank_topk_global_emit(Tensor logits, Tensor kv_lens, Tensor(a!) out, Tensor(b!) state, int "
        "mc_workspace, "
        "int uc_workspace, int rank, int world_size, int next_n, Tensor req_idx, int req_idx_bias, "
        "Tensor block_table, int block_size, int stride_factor, Tensor? layer_ids, int layer_id, "
        "Tensor(c!)? copy_dst, int copy_row_start, int copy_rows, int copy_mode, int copy_next_n, "
        "Tensor? num_accepted, Tensor(d!)? debug_times=None) -> Tensor");
    m.def("cross_rank_topk_global_emit_workspace_info(int world_size) -> int[]");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("cross_rank_topk_global_emit", &TRTLLM_MV2_NS::torch_ext::crossRankTopkGlobalEmit);
}

TRTLLM_MV2_LIBRARY_IMPL(CompositeExplicitAutograd, m)
{
    m.impl(
        "cross_rank_topk_global_emit_workspace_info", &TRTLLM_MV2_NS::torch_ext::crossRankTopkGlobalEmitWorkspaceInfo);
}
