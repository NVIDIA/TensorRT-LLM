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

#include "minimax_m3/csrc/m31/idxkSplit.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{
namespace ks = TRTLLM_MV2_NS::kernels::minimaxM3::idxkSplit;

void checkLaunch(cudaError_t err, char const* op)
{
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        TORCH_CHECK(false, op, " launch failed: ", cudaGetErrorString(err));
    }
}

void checkEpoch(torch::Tensor const& epoch, char const* op)
{
    TORCH_CHECK(epoch.is_cuda() && epoch.scalar_type() == torch::kInt32 && epoch.numel() >= 1, op,
        ": epoch must be an int32 CUDA tensor of at least one element");
}

void checkWorkspace(torch::Tensor const& ws, int64_t rowsMax, char const* op)
{
    TORCH_CHECK(ws.is_cuda() && ws.scalar_type() == torch::kInt64 && ws.is_contiguous()
            && ws.numel() >= ks::layerWords(static_cast<int32_t>(rowsMax)),
        op, ": workspace must be a contiguous int64 CUDA tensor of at least layer_words(rows_max) elements");
}

//! mcPtr != 0: one multicast store per word; else dstPtrs: [8] int64 (CPU) unicast bases of every rank's layer slice.
ks::Dest makeDest(int64_t mcPtr, torch::Tensor const& dstPtrs, char const* op)
{
    ks::Dest d{};
    d.mc = reinterpret_cast<unsigned long long*>(mcPtr);
    d.world = 0;
    if (mcPtr == 0)
    {
        TORCH_CHECK(!dstPtrs.is_cuda() && dstPtrs.scalar_type() == torch::kInt64 && dstPtrs.numel() == ks::kWorld, op,
            ": without a multicast address, dst_ptrs must be a CPU int64 tensor of 8 workspace addresses");
        auto const* p = dstPtrs.data_ptr<int64_t>();
        for (int r = 0; r < ks::kWorld; ++r)
        {
            TORCH_CHECK(p[r] != 0, op, ": dst_ptrs holds a null address");
            d.dst[r] = reinterpret_cast<unsigned long long*>(p[r]);
        }
        d.world = ks::kWorld;
    }
    return d;
}
} // namespace

//! Step setup of IDXK_SPLIT on rank `rank`: the compact block table, the scorer lengths and the step epoch (+1).
//! blockTable int32 [B, P] (unit last stride), seqLens int32 [B], cbt int32 [B, W] (unit last stride), scoreLens
//! int32 [B], epoch int32 [>= 1].
void minimaxM3Sm107Tp8IdxkCompact(torch::Tensor const& blockTable, torch::Tensor const& seqLens, torch::Tensor& cbt,
    torch::Tensor& scoreLens, torch::Tensor& epoch, int64_t rank, int64_t qlen)
{
    char const* op = "minimax_m3_sm107tp8_idxk_compact";
    TORCH_CHECK(blockTable.is_cuda() && blockTable.scalar_type() == torch::kInt32 && blockTable.dim() == 2
            && blockTable.stride(1) == 1,
        op, ": block_table must be int32 CUDA [B, P] with unit last stride");
    int64_t const batch = blockTable.size(0);
    TORCH_CHECK(seqLens.is_cuda() && seqLens.scalar_type() == torch::kInt32 && seqLens.is_contiguous()
            && seqLens.numel() >= batch,
        op, ": seq_lens must be contiguous int32 CUDA with at least B elements");
    TORCH_CHECK(cbt.is_cuda() && cbt.scalar_type() == torch::kInt32 && cbt.dim() == 2 && cbt.size(0) == batch
            && cbt.stride(1) == 1,
        op, ": cbt must be int32 CUDA [B, W] with unit last stride");
    TORCH_CHECK(scoreLens.is_cuda() && scoreLens.scalar_type() == torch::kInt32 && scoreLens.is_contiguous()
            && scoreLens.numel() >= batch,
        op, ": score_lens must be contiguous int32 CUDA with at least B elements");
    checkEpoch(epoch, op);
    c10::cuda::CUDAGuard const guard{blockTable.device()};
    auto const stream = at::cuda::getCurrentCUDAStream(blockTable.get_device());
    checkLaunch(
        ks::invokeCompact(blockTable.data_ptr<int32_t>(), blockTable.stride(0),
            static_cast<int32_t>(blockTable.size(1)), seqLens.data_ptr<int32_t>(), static_cast<int32_t>(batch),
            cbt.data_ptr<int32_t>(), cbt.stride(0), static_cast<int32_t>(cbt.size(1)), scoreLens.data_ptr<int32_t>(),
            epoch.data_ptr<int32_t>(), static_cast<int32_t>(rank), static_cast<int32_t>(qlen), stream),
        op);
}

//! Index-q exchange: idxQ FP8 (any 1-byte dtype) [rows, 1, 128] or [rows, 128] with unit last stride and a row stride
//! that is a multiple of 4 bytes; workspace: this layer's local slice; qTile [rows, 4, 128] 1-byte contiguous.
void minimaxM3Sm107Tp8IdxkQx(torch::Tensor const& idxQ, torch::Tensor const& workspace, int64_t mcPtr,
    torch::Tensor const& dstPtrs, torch::Tensor& qTile, torch::Tensor const& epoch, int64_t rank, int64_t rowsMax,
    bool publish, bool gather, bool usePdl)
{
    char const* op = "minimax_m3_sm107tp8_idxk_qx";
    TORCH_CHECK(idxQ.is_cuda() && idxQ.element_size() == 1 && (idxQ.dim() == 2 || idxQ.dim() == 3)
            && idxQ.size(-1) == ks::kHeadDim && idxQ.stride(-1) == 1 && (idxQ.dim() == 2 || idxQ.size(1) == 1),
        op, ": idx_q must be a 1-byte CUDA tensor [rows, (1,) 128] with unit last stride");
    int64_t const rows = idxQ.size(0);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(idxQ.data_ptr()) % 4 == 0, op, ": idx_q must be 4-byte aligned");
    TORCH_CHECK(qTile.is_cuda() && qTile.element_size() == 1 && qTile.is_contiguous() && qTile.dim() == 3
            && qTile.size(0) >= rows && qTile.size(1) == ks::kHeads && qTile.size(2) == ks::kHeadDim,
        op, ": q_tile must be a contiguous 1-byte CUDA tensor [>= rows, 4, 128]");
    checkWorkspace(workspace, rowsMax, op);
    checkEpoch(epoch, op);
    ks::Dest const dest = makeDest(mcPtr, dstPtrs, op);
    c10::cuda::CUDAGuard const guard{idxQ.device()};
    auto const stream = at::cuda::getCurrentCUDAStream(idxQ.get_device());
    checkLaunch(ks::invokeQx(static_cast<uint8_t const*>(idxQ.data_ptr()), idxQ.stride(0), static_cast<int32_t>(rows),
                    static_cast<int32_t>(rowsMax), dest,
                    reinterpret_cast<unsigned long long const*>(workspace.data_ptr<int64_t>()),
                    static_cast<uint8_t*>(qTile.data_ptr()), epoch.data_ptr<int32_t>(), static_cast<int32_t>(rank),
                    publish, gather, usePdl, stream),
        op);
}

//! This rank's top 16 per (query row, head) from scores [4, rows, W] (fp32, any strides: the scorer's transposed
//! view of the [4, W, rows] buffer), published to every rank.
void minimaxM3Sm107Tp8IdxkTopk(torch::Tensor const& scores, torch::Tensor const& nValidBlocks, int64_t mcPtr,
    torch::Tensor const& dstPtrs, torch::Tensor const& epoch, int64_t rank, int64_t rowsMax, int64_t numCols,
    int64_t initBlocks, int64_t localBlocks, bool usePdl)
{
    char const* op = "minimax_m3_sm107tp8_idxk_topk";
    TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == torch::kFloat32 && scores.dim() == 3
            && scores.size(0) == ks::kHeads,
        op, ": scores must be float32 CUDA [4, rows, W]");
    int64_t const rows = scores.size(1);
    TORCH_CHECK(nValidBlocks.is_cuda() && nValidBlocks.scalar_type() == torch::kInt32 && nValidBlocks.is_contiguous()
            && nValidBlocks.numel() >= rows,
        op, ": n_valid_blocks must be contiguous int32 CUDA with at least rows elements");
    // The scores of compact block j hold no causal mask; the cut is n_valid - local_blocks, so it must be >= 1.
    TORCH_CHECK(localBlocks >= 1 && initBlocks >= 0, op, ": local_blocks must be >= 1 and init_blocks >= 0");
    TORCH_CHECK(
        numCols >= 1 && scores.size(2) >= (std::min<int64_t>(numCols, ks::kMaxBlocks) + ks::kWorld - 1) / ks::kWorld,
        op, ": the score buffer is narrower than the rank's compact blocks at num_cols");
    checkEpoch(epoch, op);
    ks::Dest const dest = makeDest(mcPtr, dstPtrs, op);
    c10::cuda::CUDAGuard const guard{scores.device()};
    auto const stream = at::cuda::getCurrentCUDAStream(scores.get_device());
    checkLaunch(ks::invokeTopk(scores.data_ptr<float>(), scores.stride(0), scores.stride(1), scores.stride(2),
                    nValidBlocks.data_ptr<int32_t>(), static_cast<int32_t>(rows), static_cast<int32_t>(rowsMax),
                    static_cast<int32_t>(numCols), static_cast<int32_t>(initBlocks), static_cast<int32_t>(localBlocks),
                    dest, epoch.data_ptr<int32_t>(), static_cast<int32_t>(rank), usePdl, stream),
        op);
}

//! The FSEL table of index head `head` into out [rows, 1, 16] int32 (unit last stride; every entry written).
void minimaxM3Sm107Tp8IdxkMerge(torch::Tensor& workspace, torch::Tensor const& nValidBlocks, torch::Tensor& out,
    torch::Tensor const& epoch, int64_t head, int64_t rowsMax, int64_t numCols, int64_t initBlocks, int64_t localBlocks,
    bool usePdl, int64_t triggerPhase)
{
    char const* op = "minimax_m3_sm107tp8_idxk_merge";
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kInt32 && out.dim() == 3 && out.size(1) == 1
            && out.size(2) == ks::kTopk && out.stride(2) == 1,
        op, ": out must be int32 CUDA [rows, 1, 16] with unit last stride");
    int64_t const rows = out.size(0);
    TORCH_CHECK(nValidBlocks.is_cuda() && nValidBlocks.scalar_type() == torch::kInt32 && nValidBlocks.is_contiguous()
            && nValidBlocks.numel() >= rows,
        op, ": n_valid_blocks must be contiguous int32 CUDA with at least rows elements");
    TORCH_CHECK(localBlocks >= 1 && initBlocks >= 0, op, ": local_blocks must be >= 1 and init_blocks >= 0");
    checkWorkspace(workspace, rowsMax, op);
    checkEpoch(epoch, op);
    c10::cuda::CUDAGuard const guard{out.device()};
    auto const stream = at::cuda::getCurrentCUDAStream(out.get_device());
    checkLaunch(ks::invokeMerge(reinterpret_cast<unsigned long long*>(workspace.data_ptr<int64_t>()),
                    nValidBlocks.data_ptr<int32_t>(), static_cast<int32_t>(rows), static_cast<int32_t>(rowsMax),
                    static_cast<int32_t>(numCols), static_cast<int32_t>(initBlocks), static_cast<int32_t>(localBlocks),
                    static_cast<int32_t>(head), out.data_ptr<int32_t>(), out.stride(0), epoch.data_ptr<int32_t>(),
                    usePdl, static_cast<int32_t>(triggerPhase), stream),
        op);
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "minimax_m3_sm107tp8_idxk_compact(Tensor block_table, Tensor seq_lens, Tensor(a!) cbt, Tensor(b!) score_lens, "
        "Tensor(c!) epoch, int rank, int qlen) -> ()");
    m.def(
        "minimax_m3_sm107tp8_idxk_qx(Tensor idx_q, Tensor workspace, int mc_ptr, Tensor dst_ptrs, Tensor(a!) q_tile, "
        "Tensor epoch, int rank, int rows_max, bool publish=True, bool gather=True, bool use_pdl=False) -> ()");
    m.def(
        "minimax_m3_sm107tp8_idxk_topk(Tensor scores, Tensor n_valid_blocks, int mc_ptr, Tensor dst_ptrs, "
        "Tensor epoch, int rank, int rows_max, int num_cols, int init_blocks, int local_blocks, bool use_pdl=False) "
        "-> ()");
    m.def(
        "minimax_m3_sm107tp8_idxk_merge(Tensor(b!) workspace, Tensor n_valid_blocks, Tensor(a!) out, Tensor epoch, "
        "int head, int rows_max, int num_cols, int init_blocks, int local_blocks, bool use_pdl=False, "
        "int trigger_phase=2) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m3_sm107tp8_idxk_compact", &TRTLLM_MV2_NS::torch_ext::minimaxM3Sm107Tp8IdxkCompact);
    m.impl("minimax_m3_sm107tp8_idxk_qx", &TRTLLM_MV2_NS::torch_ext::minimaxM3Sm107Tp8IdxkQx);
    m.impl("minimax_m3_sm107tp8_idxk_topk", &TRTLLM_MV2_NS::torch_ext::minimaxM3Sm107Tp8IdxkTopk);
    m.impl("minimax_m3_sm107tp8_idxk_merge", &TRTLLM_MV2_NS::torch_ext::minimaxM3Sm107Tp8IdxkMerge);
}
