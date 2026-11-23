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

#include "minimax_m3/csrc/m31/dsparkGlueSm107.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

TRTLLM_MV2_NAMESPACE_BEGIN

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

//! DSPARK_GLUE_CPP (csrc/m31/dsparkGlueSm107.h): the drafter's context advance + read view, in place.
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
    TRTLLM_MV2_NS::kernels::minimaxM31::invokeDsparkAdvance(slots.data_ptr<int64_t>(), rows.data_ptr<int64_t>(),
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
    TRTLLM_MV2_NS::kernels::minimaxM31::invokeDsparkIndices(numCtx.data_ptr(), numCtx.scalar_type() == torch::kInt64,
        batchIdx.data_ptr(), batchIdx.scalar_type() == torch::kInt64, qpos.data_ptr(),
        qpos.scalar_type() == torch::kInt64, qpos.stride(0), qpos.stride(1), seqlens.data_ptr<int32_t>(),
        batchOut.data_ptr<int32_t>(), after.data_ptr<int32_t>(), indptr.data_ptr<int32_t>(), last.data_ptr<int32_t>(),
        append.data_ptr<int32_t>(), qflat.data_ptr<int32_t>(), static_cast<int32_t>(B), static_cast<int32_t>(block),
        static_cast<int32_t>(pagesPerSlot), static_cast<int32_t>(pageSize),
        at::cuda::getCurrentCUDAStream(numCtx.get_device()));
}

//! DSPARK_GLUE_CPP (ADV_PAR): minimax_m31_dspark_advance's outputs in two launches (advance-only, then a parallel
//! read-view copy); same operands and contracts as minimaxM31DsparkAdvance. The int64 [n] rebase scratch is
//! allocated here (graph-safe: the caching allocator's capture pool).
void minimaxM31DsparkAdvancePar(torch::Tensor const& slots, torch::Tensor const& rows, torch::Tensor const& nacc,
    torch::Tensor& ctxLen, torch::Tensor const& counts, torch::Tensor const& basePages, torch::Tensor const& tables,
    torch::Tensor& readView, torch::Tensor& out, int64_t maxCtx, int64_t dummySlot, int64_t page, int64_t blockSize,
    bool enablePdl)
{
    checkI64Vec(slots, "slots");
    checkI64Vec(rows, "rows");
    checkI64Vec(ctxLen, "ctx_len");
    checkI64Vec(counts, "counts");
    checkI64Vec(basePages, "base_pages");
    checkI64Vec(out, "out");
    int64_t const n = slots.size(0);
    TORCH_CHECK(
        rows.size(0) == n && out.size(0) == n, "minimax_m31_dspark_advance_par: rows / out must have n entries");
    TORCH_CHECK(nacc.is_cuda() && isIndex(nacc) && nacc.dim() == 1 && nacc.size(0) == n && nacc.is_contiguous(),
        "minimax_m31_dspark_advance_par: num_accepted must be a contiguous int32 / int64 [n] CUDA tensor");
    TORCH_CHECK(tables.is_cuda() && tables.scalar_type() == torch::kInt32 && tables.dim() == 2 && tables.stride(1) == 1
            && readView.is_cuda() && readView.scalar_type() == torch::kInt32 && readView.sizes() == tables.sizes()
            && readView.stride(1) == 1,
        "minimax_m31_dspark_advance_par: tables / read_view must be int32 [rows, width] with unit column stride");
    TORCH_CHECK(
        page >= 1 && n <= std::numeric_limits<int32_t>::max(), "minimax_m31_dspark_advance_par: page must be >= 1");
    if (n == 0)
    {
        return;
    }
    at::cuda::CUDAGuard const guard(slots.device());
    torch::Tensor rebase = torch::empty({n}, slots.options());
    cudaError_t const err = TRTLLM_MV2_NS::kernels::minimaxM31::invokeDsparkAdvancePar(slots.data_ptr<int64_t>(),
        rows.data_ptr<int64_t>(), nacc.data_ptr(), nacc.scalar_type() == torch::kInt64, ctxLen.data_ptr<int64_t>(),
        counts.data_ptr<int64_t>(), basePages.data_ptr<int64_t>(), tables.data_ptr<int32_t>(),
        readView.data_ptr<int32_t>(), out.data_ptr<int64_t>(), rebase.data_ptr<int64_t>(), n, tables.size(1),
        tables.stride(0), readView.stride(0), maxCtx, dummySlot, page, blockSize, enablePdl,
        at::cuda::getCurrentCUDAStream(slots.get_device()));
    TORCH_CHECK(err == cudaSuccess, "minimax_m31_dspark_advance_par: ", cudaGetErrorString(err));
}

namespace
{

void checkIdxVec(std::optional<torch::Tensor> const& t, int64_t n, char const* what)
{
    TORCH_CHECK(t.has_value() && t->is_cuda() && isIndex(*t) && t->dim() == 1 && t->is_contiguous()
            && (n < 0 || t->size(0) == n),
        "minimax_m31_dspark_qk_norm_rope: ", what, " must be a contiguous int32 / int64 1-D CUDA tensor");
}

} // namespace

//! DSPARK_GLUE_CPP: the drafter block's q / k RMSNorm + partial NeoX RoPE in one launch (head_dim 128), writing
//! q_out [T, HQ * 128] / k_out [T, HKV * 128] (contiguous bf16); with cache given, also the rows' K / V into the
//! layer's paged HND cache [pages, 2, HKV, page_size, 128] (contiguous bf16) at
//! page kv_indices[kv_indptr[batch_indices[t]] + append_positions[t] / page_size],
//! slot append_positions[t] % page_size.
//! qkv: bf16 [T, >= (HQ + 2 HKV) * 128] with unit column stride (V at column (HQ + HKV) * 128); positions: int32 [T];
//! cos_sin_cache: fp32 [*, 2 half] with unit column stride, half % 4 == 0; q_weight / k_weight: contiguous bf16 [128].
void minimaxM31DsparkQkNormRope(torch::Tensor const& qkv, torch::Tensor const& positions,
    torch::Tensor const& cosSinCache, torch::Tensor const& qWeight, torch::Tensor const& kWeight, torch::Tensor& qOut,
    torch::Tensor& kOut, double epsQ, double epsK, int64_t numQHeads, int64_t numKvHeads,
    std::optional<torch::Tensor> cache, std::optional<torch::Tensor> const& batchIndices,
    std::optional<torch::Tensor> const& appendPositions, std::optional<torch::Tensor> const& kvIndices,
    std::optional<torch::Tensor> const& kvIndptr, bool enablePdl)
{
    constexpr int64_t kHd = 128;
    TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == torch::kBFloat16 && qkv.dim() == 2 && qkv.stride(1) == 1,
        "minimax_m31_dspark_qk_norm_rope: qkv must be a bf16 [T, width] CUDA tensor with unit column stride");
    int64_t const T = qkv.size(0);
    bool const store = cache.has_value();
    // q / k heads always; the V heads (columns from (num_q_heads + num_kv_heads) * 128) only with the append.
    TORCH_CHECK(numQHeads >= 0 && numKvHeads >= 1 && qkv.size(1) >= (numQHeads + (store ? 2 : 1) * numKvHeads) * kHd,
        "minimax_m31_dspark_qk_norm_rope: qkv is narrower than its q / k (and, with cache, v) heads");
    TORCH_CHECK(T <= std::numeric_limits<int32_t>::max() / std::max<int64_t>(numQHeads + numKvHeads, 1),
        "minimax_m31_dspark_qk_norm_rope: too many rows");
    TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt32 && positions.dim() == 1
            && positions.size(0) == T && positions.is_contiguous(),
        "minimax_m31_dspark_qk_norm_rope: positions must be a contiguous int32 [T] CUDA tensor");
    TORCH_CHECK(cosSinCache.is_cuda() && cosSinCache.scalar_type() == torch::kFloat32 && cosSinCache.dim() == 2
            && cosSinCache.stride(1) == 1,
        "minimax_m31_dspark_qk_norm_rope: cos_sin_cache must be fp32 [max_pos, rot] with unit column stride");
    int64_t const rot = cosSinCache.size(1);
    TORCH_CHECK(rot % 8 == 0 && rot >= 8 && rot <= kHd,
        "minimax_m31_dspark_qk_norm_rope: the rotary width must be a multiple of 8 in [8, 128]");
    for (torch::Tensor const* wt : {&qWeight, &kWeight})
    {
        TORCH_CHECK(wt->is_cuda() && wt->scalar_type() == torch::kBFloat16 && wt->numel() == kHd && wt->is_contiguous(),
            "minimax_m31_dspark_qk_norm_rope: q_weight / k_weight must be contiguous bf16 [128] CUDA tensors");
    }
    TORCH_CHECK(qOut.is_cuda() && qOut.scalar_type() == torch::kBFloat16 && qOut.is_contiguous()
            && qOut.numel() == T * numQHeads * kHd && kOut.is_cuda() && kOut.scalar_type() == torch::kBFloat16
            && kOut.is_contiguous() && kOut.numel() == T * numKvHeads * kHd,
        "minimax_m31_dspark_qk_norm_rope: q_out / k_out must be contiguous bf16 [T, heads * 128] CUDA tensors");
    int32_t idx64 = 0;
    int64_t ps = 1;
    int64_t pageStride = 0;
    void const* idx[4] = {nullptr, nullptr, nullptr, nullptr};
    if (store)
    {
        torch::Tensor const& c = *cache;
        TORCH_CHECK(c.is_cuda() && c.scalar_type() == torch::kBFloat16 && c.dim() == 5 && c.is_contiguous()
                && c.size(1) == 2 && c.size(2) == numKvHeads && c.size(3) >= 1 && c.size(4) == kHd
                && c.get_device() == qkv.get_device(),
            "minimax_m31_dspark_qk_norm_rope: cache must be a contiguous bf16 HND [pages, 2, num_kv_heads, page_size, "
            "128] tensor on qkv's device");
        checkIdxVec(batchIndices, T, "batch_indices");
        checkIdxVec(appendPositions, T, "append_positions");
        checkIdxVec(kvIndices, -1, "kv_indices");
        checkIdxVec(kvIndptr, -1, "kv_indptr");
        std::optional<torch::Tensor> const* it[4] = {&batchIndices, &appendPositions, &kvIndices, &kvIndptr};
        for (int j = 0; j < 4; ++j)
        {
            TORCH_CHECK((*it[j])->get_device() == qkv.get_device(),
                "minimax_m31_dspark_qk_norm_rope: the index tensors must be on qkv's device");
            idx[j] = (*it[j])->data_ptr();
            idx64 |= (*it[j])->scalar_type() == torch::kInt64 ? (1 << j) : 0;
        }
        ps = c.size(3);
        pageStride = c.stride(0);
    }
    if (T == 0)
    {
        return;
    }
    at::cuda::CUDAGuard const guard(qkv.device());
    cudaError_t const err = TRTLLM_MV2_NS::kernels::minimaxM31::invokeDsparkQkNormRope(
        reinterpret_cast<__nv_bfloat16 const*>(qkv.data_ptr()), qkv.stride(0), positions.data_ptr<int32_t>(),
        cosSinCache.data_ptr<float>(), cosSinCache.stride(0),
        reinterpret_cast<__nv_bfloat16 const*>(qWeight.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(kWeight.data_ptr()), reinterpret_cast<__nv_bfloat16*>(qOut.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(kOut.data_ptr()), static_cast<float>(epsQ), static_cast<float>(epsK),
        store ? reinterpret_cast<__nv_bfloat16*>(cache->data_ptr()) : nullptr, pageStride, idx[0], idx[1], idx[2],
        idx[3], idx64, (numQHeads + numKvHeads) * kHd, static_cast<int32_t>(T), static_cast<int32_t>(numQHeads),
        static_cast<int32_t>(numKvHeads), static_cast<int32_t>(rot / 2), static_cast<int32_t>(ps), store, enablePdl,
        at::cuda::getCurrentCUDAStream(qkv.get_device()));
    TORCH_CHECK(err == cudaSuccess, "minimax_m31_dspark_qk_norm_rope: ", cudaGetErrorString(err));
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "minimax_m31_dspark_advance(Tensor slots, Tensor rows, Tensor num_accepted, Tensor(a!) ctx_len, "
        "Tensor counts, Tensor base_pages, Tensor tables, Tensor(b!) read_view, Tensor(c!) out, int max_ctx, "
        "int dummy_slot, int page, int block_size) -> ()");
    m.def(
        "minimax_m31_dspark_indices(Tensor num_ctx, Tensor batch_idx, Tensor query_positions, Tensor(a!) seqlens, "
        "Tensor(b!) batch_out, Tensor(c!) after, Tensor(d!) indptr, Tensor(e!) last, Tensor(f!) append, "
        "Tensor(g!) qflat, int block, int pages_per_slot, int page_size) -> ()");
    m.def(
        "minimax_m31_dspark_advance_par(Tensor slots, Tensor rows, Tensor num_accepted, Tensor(a!) ctx_len, "
        "Tensor counts, Tensor base_pages, Tensor tables, Tensor(b!) read_view, Tensor(c!) out, int max_ctx, "
        "int dummy_slot, int page, int block_size, bool enable_pdl=True) -> ()");
    m.def(
        "minimax_m31_dspark_qk_norm_rope(Tensor qkv, Tensor positions, Tensor cos_sin_cache, Tensor q_weight, "
        "Tensor k_weight, Tensor(a!) q_out, Tensor(b!) k_out, float eps_q, float eps_k, int num_q_heads, "
        "int num_kv_heads, Tensor(c!)? cache=None, Tensor? batch_indices=None, Tensor? append_positions=None, "
        "Tensor? kv_indices=None, Tensor? kv_indptr=None, bool enable_pdl=True) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m31_dspark_advance", &TRTLLM_MV2_NS::torch_ext::minimaxM31DsparkAdvance);
    m.impl("minimax_m31_dspark_indices", &TRTLLM_MV2_NS::torch_ext::minimaxM31DsparkIndices);
    m.impl("minimax_m31_dspark_advance_par", &TRTLLM_MV2_NS::torch_ext::minimaxM31DsparkAdvancePar);
    m.impl("minimax_m31_dspark_qk_norm_rope", &TRTLLM_MV2_NS::torch_ext::minimaxM31DsparkQkNormRope);
}
