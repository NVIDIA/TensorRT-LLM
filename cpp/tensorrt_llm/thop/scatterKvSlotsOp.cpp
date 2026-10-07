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

#include "tensorrt_llm/kernels/scatterKvSlots.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace torch_ext
{

namespace
{

void checkSlotIndex(torch::Tensor const& t, char const* name, int64_t numEntries, torch::Device device)
{
    TORCH_CHECK(t.dim() == 1 && t.size(0) == numEntries, "scatter_kv_slots_: ", name, " must be 1-D with ", numEntries,
        " entries");
    TORCH_CHECK(t.scalar_type() == torch::kInt64, "scatter_kv_slots_: ", name, " must be int64");
    TORCH_CHECK(t.is_contiguous(), "scatter_kv_slots_: ", name, " must be contiguous");
    TORCH_CHECK(t.device() == device, "scatter_kv_slots_: ", name, " must be on the pool's device");
}

} // namespace

//! In place: for every entry ``e``, write K and V of source token ``src[e]`` (``e`` without
//! ``src``), all heads, to pool slot ``dst[e]`` and, given ``dst2``, also to ``dst2[e]``. A slot id
//! is ``page * tokens_per_page + slot``.
//! ``pool``: ``[pages, 2, heads, tokens_per_page, head_dim]``, any strides with a unit head_dim stride.
//! ``k``, ``v``: ``[tokens, heads, head_dim]``, same dtype as ``pool``, unit head_dim stride, any
//! token and head strides (e.g. slices of a fused QKV projection). Indices are not range-checked
//! on the device. No allocation, safe inside CUDA graph capture.
void scatter_kv_slots_(torch::Tensor pool, torch::Tensor k, torch::Tensor v, torch::Tensor dst,
    std::optional<torch::Tensor> dst2, std::optional<torch::Tensor> src)
{
    CHECK_TH_CUDA(pool);
    CHECK_TH_CUDA(k);
    CHECK_TH_CUDA(v);
    TORCH_CHECK(pool.dim() == 5 && pool.size(1) == 2,
        "scatter_kv_slots_: pool must be [pages, 2, heads, tokens_per_page, head_dim]");
    TORCH_CHECK(
        k.dim() == 3 && v.sizes() == k.sizes(), "scatter_kv_slots_: k and v must both be [tokens, heads, head_dim]");
    TORCH_CHECK(k.size(1) == pool.size(2) && k.size(2) == pool.size(4), "scatter_kv_slots_: k is [", k.size(0), ", ",
        k.size(1), ", ", k.size(2), "] but the pool holds ", pool.size(2), " heads of ", pool.size(4));
    TORCH_CHECK(k.scalar_type() == pool.scalar_type() && v.scalar_type() == pool.scalar_type(),
        "scatter_kv_slots_: k, v and pool must share a dtype");
    TORCH_CHECK(pool.stride(4) == 1 && k.stride(2) == 1 && v.stride(2) == 1,
        "scatter_kv_slots_: head_dim must be contiguous in pool, k and v");
    TORCH_CHECK(k.device() == pool.device() && v.device() == pool.device(), "scatter_kv_slots_: device mismatch");

    int64_t const numEntries = dst.numel();
    checkSlotIndex(dst, "dst", numEntries, pool.device());
    if (dst2)
    {
        checkSlotIndex(*dst2, "dst2", numEntries, pool.device());
    }
    if (src)
    {
        checkSlotIndex(*src, "src", numEntries, pool.device());
    }
    else
    {
        TORCH_CHECK(numEntries <= k.size(0), "scatter_kv_slots_: ", numEntries, " entries without src but only ",
            k.size(0), " source tokens");
    }
    if (numEntries == 0)
    {
        return;
    }

    tensorrt_llm::kernels::ScatterKvSlotsParams params{};
    params.pool = pool.data_ptr();
    params.poolPageStride = pool.stride(0);
    params.poolKvStride = pool.stride(1);
    params.poolHeadStride = pool.stride(2);
    params.poolSlotStride = pool.stride(3);
    params.tokensPerPage = pool.size(3);
    params.k = k.data_ptr();
    params.v = v.data_ptr();
    params.kTokenStride = k.stride(0);
    params.kHeadStride = k.stride(1);
    params.vTokenStride = v.stride(0);
    params.vHeadStride = v.stride(1);
    params.numHeads = k.size(1);
    params.headDim = k.size(2);
    params.dst = dst.data_ptr<int64_t>();
    params.dst2 = dst2 ? dst2->data_ptr<int64_t>() : nullptr;
    params.src = src ? src->data_ptr<int64_t>() : nullptr;
    params.numEntries = numEntries;

    at::cuda::CUDAGuard const guard(pool.device());
    auto stream = at::cuda::getCurrentCUDAStream(pool.get_device());
    tensorrt_llm::kernels::invokeScatterKvSlots(params, static_cast<int>(pool.element_size()), stream);
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "scatter_kv_slots_(Tensor(a!) pool, Tensor k, Tensor v, Tensor dst, Tensor? dst2=None, Tensor? src=None) "
        "-> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("scatter_kv_slots_", &torch_ext::scatter_kv_slots_);
}
