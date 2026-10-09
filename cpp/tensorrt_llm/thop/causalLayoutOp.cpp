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

#include "tensorrt_llm/kernels/causalLayout.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace torch_ext
{

namespace
{

void checkIndex(torch::Tensor const& t, char const* op, char const* name, c10::ScalarType dtype,
    std::initializer_list<int64_t> shape, torch::Device device)
{
    TORCH_CHECK(t.scalar_type() == dtype, op, ": ", name, " must be ", dtype);
    TORCH_CHECK(t.is_contiguous(), op, ": ", name, " must be contiguous");
    TORCH_CHECK(t.device() == device, op, ": ", name, " must be on the table's device");
    TORCH_CHECK(t.sizes() == c10::IntArrayRef(shape.begin(), shape.size()), op, ": ", name, " is ", t.sizes(),
        " but must be ", c10::IntArrayRef(shape.begin(), shape.size()));
}

} // namespace

//! In place: the per-block tensors of one causal block size, from the cache's page table.
//! ``table``: ``[>= num_pages]`` int32 view pages. ``regions``: ``[num_blocks, region_pages]`` int32
//! private view pages. Written: ``rows [num_blocks, row_len]`` and ``block_offsets [1, num_blocks, 2,
//! row_len]`` int32, ``seq_len_kv [num_blocks]`` int32, ``own_slots [num_blocks * block_size]``,
//! ``extra_src``/``extra_dst [num_blocks * 2 * (tokens_per_page - 1)]`` and
//! ``piece_src``/``piece_dst [num_blocks * 3 * (tokens_per_page - 1)]`` int64 (-1 past the used
//! pieces). Given ``staged_slots`` (int64, the staged token count long) and ``refill_src``/
//! ``refill_dst`` (int64, ``tokens_per_page`` long), also writes those. ``drop_pages`` is the page
//! count the rotation preceding this call moved to the ring's tail, 0 if none. No allocation,
//! safe inside CUDA graph capture.
void causal_layout_(torch::Tensor table, torch::Tensor regions, torch::Tensor rows, torch::Tensor block_offsets,
    torch::Tensor seq_len_kv, torch::Tensor own_slots, torch::Tensor extra_src, torch::Tensor extra_dst,
    torch::Tensor piece_src, torch::Tensor piece_dst, std::optional<torch::Tensor> staged_slots,
    std::optional<torch::Tensor> refill_src, std::optional<torch::Tensor> refill_dst, int64_t block_size,
    int64_t tokens_per_page, int64_t past, int64_t fixed_tokens, int64_t window_tokens, int64_t num_pages,
    int64_t drop_pages, int64_t kv_factor, int64_t kv_offset, int64_t rows_per_page)
{
    char const* op = "causal_layout_";
    CHECK_TH_CUDA(table);
    TORCH_CHECK(table.dim() == 1 && table.scalar_type() == torch::kInt32 && table.is_contiguous(), op,
        ": table must be a contiguous 1-D int32 tensor");
    TORCH_CHECK(regions.dim() == 2, op, ": regions must be [num_blocks, region_pages]");
    int64_t const n = regions.size(0);
    int64_t const region_pages = regions.size(1);
    TORCH_CHECK(block_size > 0 && tokens_per_page > 1, op, ": block_size must be positive and tokens_per_page > 1");
    TORCH_CHECK(0 <= fixed_tokens && fixed_tokens <= past && window_tokens > 0, op,
        ": need 0 <= fixed_tokens <= past and window_tokens > 0");
    TORCH_CHECK(0 <= drop_pages && drop_pages < num_pages && num_pages <= table.size(0), op,
        ": need 0 <= drop_pages < num_pages <= table length");
    TORCH_CHECK((past + n * block_size + tokens_per_page - 1) / tokens_per_page <= num_pages, op,
        ": the staged tokens end past page ", num_pages);
    TORCH_CHECK(region_pages * tokens_per_page >= 3 * (tokens_per_page - 1) + block_size, op, ": a region of ",
        region_pages, " pages cannot hold three partial pages plus a block of ", block_size);
    auto const device = table.device();
    checkIndex(regions, op, "regions", torch::kInt32, {n, region_pages}, device);
    TORCH_CHECK(rows.dim() == 2 && rows.size(0) == n, op, ": rows must be [num_blocks, row_len]");
    int64_t const row_len = rows.size(1);
    TORCH_CHECK(row_len >= 1, op, ": rows must have at least one column");
    checkIndex(rows, op, "rows", torch::kInt32, {n, row_len}, device);
    checkIndex(block_offsets, op, "block_offsets", torch::kInt32, {1, n, 2, row_len}, device);
    checkIndex(seq_len_kv, op, "seq_len_kv", torch::kInt32, {n}, device);
    checkIndex(own_slots, op, "own_slots", torch::kInt64, {n * block_size}, device);
    int64_t const extra = n * 2 * (tokens_per_page - 1);
    int64_t const pieces = n * 3 * (tokens_per_page - 1);
    checkIndex(extra_src, op, "extra_src", torch::kInt64, {extra}, device);
    checkIndex(extra_dst, op, "extra_dst", torch::kInt64, {extra}, device);
    checkIndex(piece_src, op, "piece_src", torch::kInt64, {pieces}, device);
    checkIndex(piece_dst, op, "piece_dst", torch::kInt64, {pieces}, device);
    TORCH_CHECK(refill_src.has_value() == refill_dst.has_value(), op, ": refill_src and refill_dst go together");
    if (staged_slots)
    {
        TORCH_CHECK(staged_slots->dim() == 1 && staged_slots->scalar_type() == torch::kInt64
                && staged_slots->is_contiguous() && staged_slots->device() == device,
            op, ": staged_slots must be a contiguous 1-D int64 tensor on the table's device");
        TORCH_CHECK((past + staged_slots->size(0) + tokens_per_page - 1) / tokens_per_page <= num_pages, op,
            ": staged_slots reach past page ", num_pages);
    }
    if (refill_src)
    {
        checkIndex(*refill_src, op, "refill_src", torch::kInt64, {tokens_per_page}, device);
        checkIndex(*refill_dst, op, "refill_dst", torch::kInt64, {tokens_per_page}, device);
    }

    tensorrt_llm::kernels::CausalLayoutParams p{};
    p.table = table.data_ptr<int32_t>();
    p.regions = regions.data_ptr<int32_t>();
    p.numPages = num_pages;
    p.regionPages = region_pages;
    p.numBlocks = n;
    p.blockSize = block_size;
    p.tokensPerPage = tokens_per_page;
    p.past = past;
    p.fixedTokens = fixed_tokens;
    p.windowTokens = window_tokens;
    p.dropPages = drop_pages;
    p.kvFactor = kv_factor;
    p.kvOffset = kv_offset;
    p.rowsPerPage = rows_per_page;
    p.rows = rows.data_ptr<int32_t>();
    p.rowLen = row_len;
    p.blockOffsets = block_offsets.data_ptr<int32_t>();
    p.seqLenKv = seq_len_kv.data_ptr<int32_t>();
    p.ownSlots = own_slots.data_ptr<int64_t>();
    p.extraSrc = extra_src.data_ptr<int64_t>();
    p.extraDst = extra_dst.data_ptr<int64_t>();
    p.pieceSrc = piece_src.data_ptr<int64_t>();
    p.pieceDst = piece_dst.data_ptr<int64_t>();
    p.stagedSlots = staged_slots ? staged_slots->data_ptr<int64_t>() : nullptr;
    p.numStaged = staged_slots ? staged_slots->size(0) : 0;
    p.refillSrc = refill_src ? refill_src->data_ptr<int64_t>() : nullptr;
    p.refillDst = refill_dst ? refill_dst->data_ptr<int64_t>() : nullptr;

    at::cuda::CUDAGuard const guard(device);
    tensorrt_llm::kernels::invokeCausalLayout(p, at::cuda::getCurrentCUDAStream(device.index()));
}

//! In place, within one pool: entry ``e`` copies the token at pool row ``src[e]`` to pool row
//! ``dst[e]`` in every layer, K and V, every head; entries with a negative ``src`` are skipped.
//! ``pool`` is the whole contiguous K/V pool, its last dimension head_dim; a pool row is a view
//! page times ``2 * num_heads * tokens_per_page`` plus a slot, and layer ``l`` adds ``l`` times
//! that. Rows are not range-checked on the device. No allocation, safe inside CUDA graph capture.
void copy_kv_slots_(torch::Tensor pool, torch::Tensor src, torch::Tensor dst, int64_t num_layers, int64_t num_heads,
    int64_t tokens_per_page)
{
    char const* op = "copy_kv_slots_";
    CHECK_TH_CUDA(pool);
    TORCH_CHECK(pool.is_contiguous() && pool.dim() >= 1, op, ": pool must be contiguous");
    TORCH_CHECK(num_layers > 0 && num_heads > 0 && tokens_per_page > 0, op,
        ": num_layers, num_heads and tokens_per_page must be positive");
    TORCH_CHECK(
        src.dim() == 1 && src.scalar_type() == torch::kInt64 && src.is_contiguous() && src.device() == pool.device(),
        op, ": src must be a contiguous 1-D int64 tensor on the pool's device");
    TORCH_CHECK(dst.sizes() == src.sizes() && dst.scalar_type() == torch::kInt64 && dst.is_contiguous()
            && dst.device() == pool.device(),
        op, ": dst must match src");
    if (src.numel() == 0)
    {
        return;
    }
    tensorrt_llm::kernels::CopyKvSlotsParams p{};
    p.pool = pool.data_ptr();
    p.src = src.data_ptr<int64_t>();
    p.dst = dst.data_ptr<int64_t>();
    p.numEntries = src.numel();
    p.numLayers = num_layers;
    p.numHeads = num_heads;
    p.headDim = pool.size(-1);
    p.tokensPerPage = tokens_per_page;
    p.rowsPerPage = 2 * num_heads * tokens_per_page;

    at::cuda::CUDAGuard const guard(pool.device());
    tensorrt_llm::kernels::invokeCopyKvSlots(
        p, static_cast<int>(pool.element_size()), at::cuda::getCurrentCUDAStream(pool.get_device()));
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "causal_layout_(Tensor table, Tensor regions, Tensor(a!) rows, Tensor(b!) block_offsets, "
        "Tensor(c!) seq_len_kv, Tensor(d!) own_slots, Tensor(e!) extra_src, Tensor(f!) extra_dst, "
        "Tensor(g!) piece_src, Tensor(h!) piece_dst, Tensor(i!)? staged_slots, Tensor(j!)? refill_src, "
        "Tensor(k!)? refill_dst, int block_size, int tokens_per_page, int past, int fixed_tokens, "
        "int window_tokens, int num_pages, int drop_pages, int kv_factor, int kv_offset, int rows_per_page) -> ()");
    m.def(
        "copy_kv_slots_(Tensor(a!) pool, Tensor src, Tensor dst, int num_layers, int num_heads, "
        "int tokens_per_page) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("causal_layout_", &torch_ext::causal_layout_);
    m.impl("copy_kv_slots_", &torch_ext::copy_kv_slots_);
}
