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

#include "tensorrt_llm/kernels/minimaxM31/dsparkGlueSm107.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

bool isIndex(torch::Tensor const& t)
{
    return t.scalar_type() == torch::kInt32 || t.scalar_type() == torch::kInt64;
}

void checkI64Vec(torch::Tensor const& t, char const* what)
{
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kInt64 && t.dim() == 1 && t.is_contiguous(),
        "minimax_m31_dspark_advance: ", what, " must be a contiguous int64 1-D CUDA tensor");
}

} // namespace

//! DSPARK_GLUE_CPP (kernels/minimaxM31/dsparkGlueSm107.h): the drafter's context advance + read view, in place.
//! slots / rows: int64 [n]; nacc: int32 / int64 [n]; ctx_len: int64 [slots] (updated); counts: int64 [rows];
//! base_pages: int64 [slots]; tables / read_view: int32 [rows, width], unit column stride (read_view written);
//! out: int64 [n] (written).
void minimaxM31DsparkAdvance(torch::Tensor const& slots, torch::Tensor const& rows, torch::Tensor const& nacc,
    torch::Tensor& ctxLen, torch::Tensor const& counts, torch::Tensor const& basePages, torch::Tensor const& tables,
    torch::Tensor& readView, torch::Tensor& out, int64_t maxCtx, int64_t dummySlot, int64_t page, int64_t blockSize)
{
    checkI64Vec(slots, "slots");
    checkI64Vec(rows, "rows");
    checkI64Vec(ctxLen, "ctx_len");
    checkI64Vec(counts, "counts");
    checkI64Vec(basePages, "base_pages");
    checkI64Vec(out, "out");
    int64_t const n = slots.size(0);
    TORCH_CHECK(rows.size(0) == n && out.size(0) == n, "minimax_m31_dspark_advance: rows / out must have n entries");
    TORCH_CHECK(nacc.is_cuda() && isIndex(nacc) && nacc.dim() == 1 && nacc.size(0) == n && nacc.is_contiguous(),
        "minimax_m31_dspark_advance: num_accepted must be a contiguous int32 / int64 [n] CUDA tensor");
    TORCH_CHECK(tables.is_cuda() && tables.scalar_type() == torch::kInt32 && tables.dim() == 2 && tables.stride(1) == 1
            && readView.is_cuda() && readView.scalar_type() == torch::kInt32 && readView.sizes() == tables.sizes()
            && readView.stride(1) == 1,
        "minimax_m31_dspark_advance: tables / read_view must be int32 [rows, width] with unit column stride");
    TORCH_CHECK(page >= 1 && n <= std::numeric_limits<int32_t>::max(), "minimax_m31_dspark_advance: page must be >= 1");
    at::cuda::CUDAGuard const guard(slots.device());
    tensorrt_llm::kernels::minimaxM31::invokeDsparkAdvance(slots.data_ptr<int64_t>(), rows.data_ptr<int64_t>(),
        nacc.data_ptr(), nacc.scalar_type() == torch::kInt64, ctxLen.data_ptr<int64_t>(), counts.data_ptr<int64_t>(),
        basePages.data_ptr<int64_t>(), tables.data_ptr<int32_t>(), readView.data_ptr<int32_t>(),
        out.data_ptr<int64_t>(), n, tables.size(1), tables.stride(0), readView.stride(0), maxCtx, dummySlot, page,
        blockSize, at::cuda::getCurrentCUDAStream(slots.get_device()));
}

//! DSPARK_GLUE_CPP: the paged dflash_forward's index prelude (seven int32 outputs written in place).
//! num_ctx / batch_idx: int32 / int64 [B]; qpos: int32 / int64 [B, block] (any strides).
void minimaxM31DsparkIndices(torch::Tensor const& numCtx, torch::Tensor const& batchIdx, torch::Tensor const& qpos,
    torch::Tensor& seqlens, torch::Tensor& batchOut, torch::Tensor& after, torch::Tensor& indptr, torch::Tensor& last,
    torch::Tensor& append, torch::Tensor& qflat, int64_t block, int64_t pagesPerSlot, int64_t pageSize)
{
    TORCH_CHECK(numCtx.is_cuda() && isIndex(numCtx) && numCtx.dim() == 1 && numCtx.is_contiguous(),
        "minimax_m31_dspark_indices: num_ctx must be a contiguous int32 / int64 [B] CUDA tensor");
    int64_t const B = numCtx.size(0);
    TORCH_CHECK(isIndex(batchIdx) && batchIdx.dim() == 1 && batchIdx.size(0) == B && batchIdx.is_contiguous(),
        "minimax_m31_dspark_indices: batch_idx must be a contiguous int32 / int64 [B] tensor");
    TORCH_CHECK(isIndex(qpos) && qpos.dim() == 2 && qpos.size(0) == B && qpos.size(1) == block,
        "minimax_m31_dspark_indices: query_positions must be int32 / int64 [B, block]");
    TORCH_CHECK(
        block >= 1 && pagesPerSlot >= 1 && pageSize >= 1 && B >= 1 && B * block <= std::numeric_limits<int32_t>::max(),
        "minimax_m31_dspark_indices: invalid sizes");
    for (torch::Tensor const* t : {&seqlens, &batchOut, &after, &indptr, &last, &append, &qflat})
    {
        TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kInt32 && t->is_contiguous(),
            "minimax_m31_dspark_indices: outputs must be contiguous int32 CUDA tensors");
    }
    TORCH_CHECK(seqlens.numel() >= B && batchOut.numel() >= B && after.numel() >= B && indptr.numel() >= B + 1
            && last.numel() >= B && append.numel() >= B * block && qflat.numel() >= B * block,
        "minimax_m31_dspark_indices: an output is too small");
    at::cuda::CUDAGuard const guard(numCtx.device());
    tensorrt_llm::kernels::minimaxM31::invokeDsparkIndices(numCtx.data_ptr(), numCtx.scalar_type() == torch::kInt64,
        batchIdx.data_ptr(), batchIdx.scalar_type() == torch::kInt64, qpos.data_ptr(),
        qpos.scalar_type() == torch::kInt64, qpos.stride(0), qpos.stride(1), seqlens.data_ptr<int32_t>(),
        batchOut.data_ptr<int32_t>(), after.data_ptr<int32_t>(), indptr.data_ptr<int32_t>(), last.data_ptr<int32_t>(),
        append.data_ptr<int32_t>(), qflat.data_ptr<int32_t>(), static_cast<int32_t>(B), static_cast<int32_t>(block),
        static_cast<int32_t>(pagesPerSlot), static_cast<int32_t>(pageSize),
        at::cuda::getCurrentCUDAStream(numCtx.get_device()));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m31_dspark_advance(Tensor slots, Tensor rows, Tensor num_accepted, Tensor(a!) ctx_len, "
        "Tensor counts, Tensor base_pages, Tensor tables, Tensor(b!) read_view, Tensor(c!) out, int max_ctx, "
        "int dummy_slot, int page, int block_size) -> ()");
    m.def(
        "minimax_m31_dspark_indices(Tensor num_ctx, Tensor batch_idx, Tensor query_positions, Tensor(a!) seqlens, "
        "Tensor(b!) batch_out, Tensor(c!) after, Tensor(d!) indptr, Tensor(e!) last, Tensor(f!) append, "
        "Tensor(g!) qflat, int block, int pages_per_slot, int page_size) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m31_dspark_advance", &tensorrt_llm::torch_ext::minimaxM31DsparkAdvance);
    m.impl("minimax_m31_dspark_indices", &tensorrt_llm::torch_ext::minimaxM31DsparkIndices);
}
