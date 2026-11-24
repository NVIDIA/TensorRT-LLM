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

#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/csrc/tailGlueSm107.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <utility>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

constexpr char const* kOp = "minimax_m31_tail_glue_a: ";

bool isIndex(torch::Tensor const& t)
{
    return t.scalar_type() == torch::kInt32 || t.scalar_type() == torch::kInt64;
}

void checkVec(torch::Tensor const& t, torch::Device dev, c10::ScalarType dtype, int64_t numel, char const* what)
{
    TORCH_CHECK(t.device() == dev && t.scalar_type() == dtype && t.dim() == 1 && t.is_contiguous()
            && (numel < 0 || t.numel() == numel),
        kOp, what, " must be a contiguous ", c10::toString(dtype), " 1-D tensor on kv's device",
        numel < 0 ? "" : " with the expected length");
}

//! True when the byte spans of two 2-D tables (unit column stride) intersect: offset views of one
//! allocation overlap even when their start addresses differ.
bool overlaps(torch::Tensor const& a, torch::Tensor const& b)
{
    if (a.numel() == 0 || b.numel() == 0)
    {
        return false;
    }
    auto span = [](torch::Tensor const& t)
    {
        auto const lo = reinterpret_cast<uintptr_t>(t.data_ptr());
        auto const last = (t.size(0) - 1) * t.stride(0) + t.size(1);
        return std::pair<uintptr_t, uintptr_t>{lo, lo + static_cast<uintptr_t>(last) * t.element_size()};
    };
    auto const [aBegin, aEnd] = span(a);
    auto const [bBegin, bEnd] = span(b);
    return aBegin < bEnd && bBegin < aEnd;
}

} // namespace

//! TAILGLUE_A (csrc/tailGlueSm107.h): the forked DSpark step's context store, context advance, read
//! view and forward indices in one launch, every output written in place.
//! Store (minimax_m31_dspark_ctx_store with col_from_pos and bound): kv bf16 [n * kp1, >= L * 2 * NKV * HD] (unit
//! column stride); k bf16 [n * kp1, L, NKV, HD] contiguous; w bf16 [L, HD] contiguous (fold_w); cos_sin fp32
//! [max_pos, rot] contiguous; positions int64 [n * kp1]; num_accepted int32 / int64 [n]; request_rows int64 [n];
//! counts int64 [table rows]; block_table int32 [table rows, width] (unit column stride); cache the stacked bf16
//! [L, pages, 2, NKV, page_size, HD] cache or (ptr_list) an int64 [L] address array.
//! Advance (minimax_m31_dspark_advance_par): slots int64 [n]; ctx_len int64 [slots] (updated); base_pages int64
//! [slots]; read_view int32 like block_table (written); out int64 [n] (written).
//! Indices (minimax_m31_dspark_indices of out, request_rows, query_positions, block, width, index_page_size):
//! query_positions int32 / int64 [n, block] (any strides); seven contiguous int32 outputs.
void minimaxM31TailGlueA(torch::Tensor const& kv, torch::Tensor const& k, torch::Tensor const& w,
    torch::Tensor const& cosSin, torch::Tensor const& positions, torch::Tensor const& numAccepted,
    torch::Tensor const& requestRows, torch::Tensor const& counts, torch::Tensor const& blockTable,
    torch::Tensor& cache, int64_t cacheStrideLayer, int64_t tokensPerRequest, int64_t numLayers, int64_t numKvHeads,
    int64_t headDim, int64_t pageSize, bool ptrList, bool foldW, torch::Tensor const& slots, torch::Tensor& ctxLen,
    torch::Tensor const& basePages, torch::Tensor& readView, torch::Tensor& out, int64_t maxCtx, int64_t dummySlot,
    int64_t advPage, int64_t blockSize, torch::Tensor const& queryPositions, torch::Tensor& seqlens,
    torch::Tensor& batchOut, torch::Tensor& after, torch::Tensor& indptr, torch::Tensor& last, torch::Tensor& append,
    torch::Tensor& qflat, int64_t block, int64_t indexPageSize, bool enablePdl)
{
    TORCH_CHECK(kv.is_cuda() && kv.scalar_type() == torch::kBFloat16 && kv.dim() == 2 && kv.stride(1) == 1, kOp,
        "kv must be a bf16 [rows, L * 2 * NKV * HD] CUDA tensor with unit column stride");
    auto const dev = kv.device();
    int64_t const rowsN = kv.size(0);
    TORCH_CHECK(tokensPerRequest >= 1 && rowsN % tokensPerRequest == 0, kOp, "kv rows must be n x tokens_per_request");
    int64_t const n = rowsN / tokensPerRequest;
    TORCH_CHECK(n >= 1 && rowsN <= std::numeric_limits<int32_t>::max(), kOp, "invalid request count");
    TORCH_CHECK(numLayers >= 1 && numLayers <= 65535 && numKvHeads >= 1 && numKvHeads <= 65535 && headDim >= 16
            && headDim <= 256 && (headDim & (headDim - 1)) == 0 && pageSize >= 1
            && pageSize <= std::numeric_limits<int32_t>::max()
            && rowsN * numLayers * numKvHeads <= std::numeric_limits<int32_t>::max(),
        kOp, "invalid store sizes");
    TORCH_CHECK(kv.size(1) >= numLayers * 2 * numKvHeads * headDim, kOp, "kv is narrower than L * 2 * NKV * HD");
    TORCH_CHECK(k.device() == dev && k.scalar_type() == torch::kBFloat16 && k.is_contiguous()
            && k.numel() == rowsN * numLayers * numKvHeads * headDim,
        kOp, "k must be a contiguous bf16 [rows, L, NKV, HD] tensor on kv's device");
    if (foldW)
    {
        TORCH_CHECK(w.device() == dev && w.scalar_type() == torch::kBFloat16 && w.is_contiguous()
                && w.numel() == numLayers * headDim,
            kOp, "w must be a contiguous bf16 [L, HD] tensor on kv's device");
    }
    TORCH_CHECK(cosSin.device() == dev && cosSin.scalar_type() == torch::kFloat32 && cosSin.dim() == 2
            && cosSin.is_contiguous(),
        kOp, "cos_sin must be a contiguous fp32 [max_pos, rot] tensor on kv's device");
    int64_t const rot = cosSin.size(1);
    TORCH_CHECK(rot >= 2 && rot % 2 == 0 && rot <= headDim, kOp, "unsupported rotary dim");
    checkVec(positions, dev, torch::kInt64, rowsN, "positions");
    TORCH_CHECK(numAccepted.device() == dev && isIndex(numAccepted) && numAccepted.dim() == 1
            && numAccepted.numel() == n && numAccepted.is_contiguous(),
        kOp, "num_accepted must be a contiguous int32 / int64 [n] tensor on kv's device");
    checkVec(requestRows, dev, torch::kInt64, n, "request_rows");
    checkVec(counts, dev, torch::kInt64, -1, "counts");
    TORCH_CHECK(blockTable.device() == dev && blockTable.scalar_type() == torch::kInt32 && blockTable.dim() == 2
            && blockTable.stride(1) == 1 && blockTable.size(1) >= 1,
        kOp, "block_table must be an int32 [rows, width] tensor with unit column stride on kv's device");
    TORCH_CHECK(readView.device() == dev && readView.scalar_type() == torch::kInt32
            && readView.sizes() == blockTable.sizes() && readView.stride(1) == 1,
        kOp, "read_view must be an int32 tensor shaped like block_table with unit column stride");
    TORCH_CHECK(counts.numel() >= blockTable.size(0), kOp, "counts must cover every block-table row");
    if (ptrList)
    {
        TORCH_CHECK(cache.device() == dev && cache.scalar_type() == torch::kInt64 && cache.dim() == 1
                && cache.numel() == numLayers && cache.is_contiguous(),
            kOp, "a pointer-list cache must be a contiguous int64 [L] address array");
    }
    else
    {
        TORCH_CHECK(cache.device() == dev && cache.scalar_type() == torch::kBFloat16 && cache.dim() == 6
                && cache.is_contiguous() && cache.size(0) == numLayers && cache.size(2) == 2
                && cache.size(3) == numKvHeads && cache.size(4) == pageSize && cache.size(5) == headDim
                && cacheStrideLayer == cache.stride(0),
            kOp, "the stacked cache must be a contiguous bf16 [L, pages, 2, NKV, page_size, HD] tensor");
    }
    checkVec(slots, dev, torch::kInt64, n, "slots");
    checkVec(ctxLen, dev, torch::kInt64, -1, "ctx_len");
    checkVec(basePages, dev, torch::kInt64, -1, "base_pages");
    checkVec(out, dev, torch::kInt64, n, "out");
    TORCH_CHECK(dummySlot >= 0 && dummySlot < ctxLen.numel() && dummySlot < basePages.numel(), kOp,
        "dummy_slot out of range of ctx_len / base_pages");
    TORCH_CHECK(!overlaps(readView, blockTable), kOp,
        "read_view must not overlap block_table (the read view reads the table while it writes)");
    TORCH_CHECK(advPage >= 1 && indexPageSize >= 1 && indexPageSize <= std::numeric_limits<int32_t>::max(), kOp,
        "page sizes must be >= 1");
    int64_t const width = blockTable.size(1);
    TORCH_CHECK(block >= 1 && n * block <= std::numeric_limits<int32_t>::max()
            && (n + 1) * width <= std::numeric_limits<int32_t>::max(),
        kOp, "invalid index sizes");
    TORCH_CHECK(queryPositions.device() == dev && isIndex(queryPositions) && queryPositions.dim() == 2
            && queryPositions.size(0) == n && queryPositions.size(1) == block,
        kOp, "query_positions must be an int32 / int64 [n, block] tensor on kv's device");
    for (torch::Tensor const* t : {&seqlens, &batchOut, &after, &last})
    {
        checkVec(*t, dev, torch::kInt32, n, "an [n] index output");
    }
    checkVec(indptr, dev, torch::kInt32, n + 1, "kv_indptr");
    checkVec(append, dev, torch::kInt32, n * block, "append_positions");
    checkVec(qflat, dev, torch::kInt32, n * block, "query_positions_flat");

    TRTLLM_MV2_NS::kernels::minimaxM31::TailGlueParams p{};
    p.kv = kv.data_ptr();
    p.kvStrideRow = kv.stride(0);
    p.k = k.data_ptr();
    p.w = w.data_ptr();
    p.cosSin = cosSin.data_ptr<float>();
    p.pos = positions.data_ptr<int64_t>();
    p.nacc = numAccepted.data_ptr();
    p.naccIs64 = numAccepted.scalar_type() == torch::kInt64;
    p.rows = requestRows.data_ptr<int64_t>();
    p.counts = counts.data_ptr<int64_t>();
    p.table = blockTable.data_ptr<int32_t>();
    p.tableStride = blockTable.stride(0);
    p.cache = cache.data_ptr();
    p.cacheStrideLayer = ptrList ? 0 : cacheStrideLayer;
    p.kp1 = static_cast<int32_t>(tokensPerRequest);
    p.numLayers = static_cast<int32_t>(numLayers);
    p.numKvHeads = static_cast<int32_t>(numKvHeads);
    p.headDim = static_cast<int32_t>(headDim);
    p.rot = static_cast<int32_t>(rot);
    p.storePage = static_cast<int32_t>(pageSize);
    p.ptrList = ptrList;
    p.foldW = foldW;
    p.slots = slots.data_ptr<int64_t>();
    p.ctxLen = ctxLen.data_ptr<int64_t>();
    p.basePages = basePages.data_ptr<int64_t>();
    p.out = out.data_ptr<int64_t>();
    p.maxCtx = maxCtx;
    p.dummySlot = dummySlot;
    p.page = advPage;
    p.blockSize = blockSize;
    p.readView = readView.data_ptr<int32_t>();
    p.readStride = readView.stride(0);
    p.width = width;
    p.qpos = queryPositions.data_ptr();
    p.qposIs64 = queryPositions.scalar_type() == torch::kInt64;
    p.q0 = queryPositions.stride(0);
    p.q1 = queryPositions.stride(1);
    p.seqlens = seqlens.data_ptr<int32_t>();
    p.batchOut = batchOut.data_ptr<int32_t>();
    p.after = after.data_ptr<int32_t>();
    p.indptr = indptr.data_ptr<int32_t>();
    p.last = last.data_ptr<int32_t>();
    p.append = append.data_ptr<int32_t>();
    p.qflat = qflat.data_ptr<int32_t>();
    p.iblock = static_cast<int32_t>(block);
    p.ipage = static_cast<int32_t>(indexPageSize);
    p.n = static_cast<int32_t>(n);
    at::cuda::CUDAGuard const guard(dev);
    cudaError_t const err
        = TRTLLM_MV2_NS::kernels::minimaxM31::invokeTailGlue(p, enablePdl, at::cuda::getCurrentCUDAStream(dev.index()));
    TORCH_CHECK(err == cudaSuccess, kOp, cudaGetErrorString(err));
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "minimax_m31_tail_glue_a(Tensor kv, Tensor k, Tensor w, Tensor cos_sin, Tensor positions, "
        "Tensor num_accepted, Tensor request_rows, Tensor counts, Tensor block_table, Tensor(a!) cache, "
        "int cache_stride_layer, int tokens_per_request, int num_layers, int num_kv_heads, int head_dim, "
        "int page_size, bool ptr_list, bool fold_w, Tensor slots, Tensor(b!) ctx_len, Tensor base_pages, "
        "Tensor(c!) read_view, Tensor(d!) out, int max_ctx, int dummy_slot, int adv_page, int block_size, "
        "Tensor query_positions, Tensor(e!) seqlens, Tensor(f!) batch_out, Tensor(g!) after, Tensor(h!) indptr, "
        "Tensor(i!) last, Tensor(j!) append, Tensor(k!) qflat, int block, int index_page_size, "
        "bool enable_pdl=True) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m31_tail_glue_a", &TRTLLM_MV2_NS::torch_ext::minimaxM31TailGlueA);
}
