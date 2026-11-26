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

#include "minimax_m3/csrc/m31/dsparkCtxStore.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

bool isIndex(torch::Tensor const& t)
{
    return t.scalar_type() == torch::kInt32 || t.scalar_type() == torch::kInt64;
}

void checkIndexVec(torch::Tensor const& t, torch::Device dev, int64_t minNumel, char const* what)
{
    TORCH_CHECK(t.device() == dev && isIndex(t) && t.dim() == 1 && t.is_contiguous() && t.numel() >= minNumel,
        "minimax_m31_dspark_ctx_store: ", what,
        " must be a contiguous int32 / int64 1-D tensor on kv's device with at "
        "least ",
        minNumel, " elements");
}

} // namespace

//! DSPARK_CTX_STORE_CPP (csrc/m31/dsparkCtxStore.h): the drafter's context K / V store, every layer in one
//! launch, written into cache in place. kv: bf16 [n, L * 2 * NKV * HD] (unit column stride); k: bf16 [n, L, NKV, HD]
//! contiguous; w: bf16 [L, HD] contiguous (fold_w); cos_sin: fp32 [max_pos, rot] contiguous; positions / cols: [n];
//! num_accepted / request_rows: [n / tokens_per_request]; counts: [table rows] (bound); block_table: [rows, width]
//! (unit column stride); cache: the stacked bf16 [L, pages, 2, NKV, page_size, HD] cache, or (ptr_list) an int64 [L]
//! array of per-layer cache addresses. Index tensors int32 / int64. cols / counts may alias positions when unused.
void minimaxM31DsparkCtxStore(torch::Tensor const& kv, torch::Tensor const& k, torch::Tensor const& w,
    torch::Tensor const& cosSin, torch::Tensor const& positions, torch::Tensor const& numAccepted,
    torch::Tensor const& requestRows, torch::Tensor const& cols, torch::Tensor const& counts, int64_t maxCol,
    torch::Tensor const& blockTable, torch::Tensor& cache, int64_t cacheStrideLayer, int64_t tokensPerRequest,
    int64_t numLayers, int64_t numKvHeads, int64_t headDim, int64_t pageSize, bool ptrList, bool colFromPos, bool bound,
    bool foldW)
{
    TORCH_CHECK(kv.is_cuda() && kv.scalar_type() == torch::kBFloat16 && kv.dim() == 2 && kv.stride(1) == 1,
        "minimax_m31_dspark_ctx_store: kv must be a bf16 [n, L * 2 * NKV * HD] CUDA tensor with unit column stride");
    auto const dev = kv.device();
    int64_t const n = kv.size(0);
    TORCH_CHECK(tokensPerRequest >= 1 && numLayers >= 1 && numLayers <= 65535 && numKvHeads >= 1 && numKvHeads <= 65535
            && headDim >= 16 && headDim <= 256 && (headDim & (headDim - 1)) == 0 && pageSize >= 1
            && pageSize <= std::numeric_limits<int32_t>::max() && n <= std::numeric_limits<int32_t>::max()
            && n % tokensPerRequest == 0,
        "minimax_m31_dspark_ctx_store: invalid sizes");
    TORCH_CHECK(kv.size(1) >= numLayers * 2 * numKvHeads * headDim,
        "minimax_m31_dspark_ctx_store: kv is narrower than L * 2 * NKV * HD");
    TORCH_CHECK(k.device() == dev && k.scalar_type() == torch::kBFloat16 && k.is_contiguous()
            && k.numel() == n * numLayers * numKvHeads * headDim,
        "minimax_m31_dspark_ctx_store: k must be a contiguous bf16 [n, L, NKV, HD] tensor on kv's device");
    if (foldW)
    {
        TORCH_CHECK(w.device() == dev && w.scalar_type() == torch::kBFloat16 && w.is_contiguous()
                && w.numel() == numLayers * headDim,
            "minimax_m31_dspark_ctx_store: w must be a contiguous bf16 [L, HD] tensor on kv's device");
    }
    TORCH_CHECK(cosSin.device() == dev && cosSin.scalar_type() == torch::kFloat32 && cosSin.dim() == 2
            && cosSin.is_contiguous(),
        "minimax_m31_dspark_ctx_store: cos_sin must be a contiguous fp32 [max_pos, rot] tensor on kv's device");
    int64_t const rot = cosSin.size(1);
    TORCH_CHECK(rot >= 2 && rot % 2 == 0 && rot <= headDim, "minimax_m31_dspark_ctx_store: unsupported rotary dim");
    int64_t const numReq = n / tokensPerRequest;
    checkIndexVec(positions, dev, n, "positions");
    checkIndexVec(numAccepted, dev, numReq, "num_accepted");
    checkIndexVec(requestRows, dev, numReq, "request_rows");
    checkIndexVec(cols, dev, colFromPos ? 0 : n, "cols");
    checkIndexVec(counts, dev, 0, "counts");
    TORCH_CHECK(blockTable.device() == dev && isIndex(blockTable) && blockTable.dim() == 2
            && (blockTable.stride(1) == 1 || blockTable.size(1) == 1),
        "minimax_m31_dspark_ctx_store: block_table must be an int32 / int64 [rows, width] tensor with unit column "
        "stride on kv's device");
    if (ptrList)
    {
        TORCH_CHECK(cache.device() == dev && cache.scalar_type() == torch::kInt64 && cache.dim() == 1
                && cache.numel() == numLayers && cache.is_contiguous(),
            "minimax_m31_dspark_ctx_store: a pointer-list cache must be a contiguous int64 [L] address array");
    }
    else
    {
        TORCH_CHECK(cache.device() == dev && cache.scalar_type() == torch::kBFloat16 && cache.dim() == 6
                && cache.is_contiguous() && cache.size(0) == numLayers && cache.size(2) == 2
                && cache.size(3) == numKvHeads && cache.size(4) == pageSize && cache.size(5) == headDim
                && cacheStrideLayer == cache.stride(0),
            "minimax_m31_dspark_ctx_store: the stacked cache must be a contiguous bf16 [L, pages, 2, NKV, page_size, "
            "HD] tensor on kv's device");
    }
    if (n == 0)
    {
        return;
    }
    auto is64 = [](torch::Tensor const& t) { return t.scalar_type() == torch::kInt64; };
    TRTLLM_MV2_NS::kernels::minimaxM31::DsparkCtxStoreParams p{};
    p.kv = kv.data_ptr();
    p.kvStrideRow = kv.stride(0);
    p.k = k.data_ptr();
    p.w = w.data_ptr();
    p.cosSin = cosSin.data_ptr<float>();
    p.pos = positions.data_ptr();
    p.numAccepted = numAccepted.data_ptr();
    p.requestRows = requestRows.data_ptr();
    p.cols = cols.data_ptr();
    p.counts = counts.data_ptr();
    p.maxCol = maxCol;
    p.table = blockTable.data_ptr();
    p.tableStride = blockTable.stride(0);
    p.tableWidth = blockTable.size(1);
    p.cache = cache.data_ptr();
    p.cacheStrideLayer = ptrList ? 0 : cacheStrideLayer;
    p.n = static_cast<int32_t>(n);
    p.kp1 = static_cast<int32_t>(tokensPerRequest);
    p.numLayers = static_cast<int32_t>(numLayers);
    p.numKvHeads = static_cast<int32_t>(numKvHeads);
    p.headDim = static_cast<int32_t>(headDim);
    p.rot = static_cast<int32_t>(rot);
    p.pageSize = static_cast<int32_t>(pageSize);
    p.idxBits = (is64(positions) ? 1 : 0) | (is64(numAccepted) ? 2 : 0) | (is64(requestRows) ? 4 : 0)
        | (is64(cols) ? 8 : 0) | (is64(counts) ? 16 : 0) | (is64(blockTable) ? 32 : 0);
    p.ptrList = ptrList;
    p.colFromPos = colFromPos;
    p.bound = bound && colFromPos;
    p.foldW = foldW;
    at::cuda::CUDAGuard const guard(dev);
    TRTLLM_MV2_NS::kernels::minimaxM31::invokeDsparkCtxStore(p, at::cuda::getCurrentCUDAStream(dev.index()));
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "minimax_m31_dspark_ctx_store(Tensor kv, Tensor k, Tensor w, Tensor cos_sin, Tensor positions, "
        "Tensor num_accepted, Tensor request_rows, Tensor cols, Tensor counts, int max_col, Tensor block_table, "
        "Tensor(a!) cache, int cache_stride_layer, int tokens_per_request, int num_layers, int num_kv_heads, "
        "int head_dim, int page_size, bool ptr_list, bool col_from_pos, bool bound, bool fold_w) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m31_dspark_ctx_store", &TRTLLM_MV2_NS::torch_ext::minimaxM31DsparkCtxStore);
}
