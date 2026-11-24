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

#include "tensorrt_llm/kernels/minimaxM3/specTail.h"

#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <initializer_list>
#include <optional>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

namespace st = tensorrt_llm::kernels::minimaxM3::specTail;

void check(cudaError_t err)
{
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}

cudaStream_t currentStream(torch::Tensor const& t)
{
    return at::cuda::getCurrentCUDAStream(t.get_device());
}

__nv_bfloat16 const* bf16In(torch::Tensor const& t)
{
    return reinterpret_cast<__nv_bfloat16 const*>(t.data_ptr());
}

__nv_bfloat16* bf16Out(torch::Tensor const& t)
{
    return reinterpret_cast<__nv_bfloat16*>(t.data_ptr());
}

//! Register-tile count of the norm kernels for hidden size H (H / 8 elements over kThreads threads).
int normIters(int H)
{
    return (H / 8 + st::kThreads - 1) / st::kThreads;
}

st::IntView intView(torch::Tensor const& t)
{
    return st::IntView{t.data_ptr<int>(), t.stride(0), t.stride(1), static_cast<int>(t.size(1))};
}

} // namespace

//! out = [rmsnorm(embed(ids)) * wEmbed, rmsnorm(hidden[rows]) * wHidden] [M, 2H]; residual = hidden[rows].
void minimaxM3DraftInput(torch::Tensor const& ids, torch::Tensor const& table, torch::Tensor const& hidden,
    std::optional<torch::Tensor> const& rows, torch::Tensor const& wEmbed, torch::Tensor const& wHidden,
    double epsEmbed, double epsHidden, torch::Tensor& out, std::optional<torch::Tensor> const& residual, bool pdl)
{
    c10::cuda::CUDAGuard const deviceGuard{table.device()};
    int const M = ids.size(0);
    int const H = table.size(1);
    TORCH_CHECK(ids.scalar_type() == at::kInt && ids.is_contiguous() && table.scalar_type() == at::kBFloat16
        && table.is_contiguous() && H % 8 == 0);
    TORCH_CHECK(
        hidden.scalar_type() == at::kBFloat16 && hidden.dim() == 2 && hidden.is_contiguous() && hidden.size(1) == H);
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.size(0) == M && out.size(1) == 2 * H && out.is_contiguous());
    TORCH_CHECK(wEmbed.scalar_type() == at::kBFloat16 && wEmbed.numel() == H && wHidden.scalar_type() == at::kBFloat16
        && wHidden.numel() == H);
    TORCH_CHECK(!rows.has_value() || (rows->scalar_type() == at::kLong && rows->is_contiguous() && rows->size(0) == M));
    TORCH_CHECK(rows.has_value() || hidden.size(0) >= M, "draft_input: hidden must hold a row per id");
    TORCH_CHECK(!residual.has_value()
            || (residual->scalar_type() == at::kBFloat16 && residual->is_contiguous() && residual->size(0) == M
                && residual->size(1) == H),
        "draft_input: residual must be contiguous [M, H]");
    TORCH_CHECK(normIters(H) <= st::kMaxIters, "draft_input: hidden size too large");
    check(st::invokeDraftInput(ids.data_ptr<int>(), static_cast<int>(table.size(0)), bf16In(table), bf16In(hidden),
        rows.has_value() ? reinterpret_cast<long long const*>(rows->data_ptr()) : nullptr, bf16In(wEmbed),
        bf16In(wHidden), static_cast<float>(epsEmbed), static_cast<float>(epsHidden), H, M, bf16Out(out),
        residual.has_value() ? bf16Out(*residual) : nullptr, pdl, currentStream(table)));
}

//! Per-chunk RMSNorm of src [M, >= C*H] (row stride may exceed C*H) with weights w [C, H] into dst [M, C*H].
void minimaxM3ChunkNorm(torch::Tensor const& src, torch::Tensor const& w, double eps, torch::Tensor& dst, bool pdl)
{
    c10::cuda::CUDAGuard const deviceGuard{src.device()};
    int const M = src.size(0);
    int const C = w.size(0);
    int const H = w.size(1);
    TORCH_CHECK(src.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 && src.stride(1) == 1);
    TORCH_CHECK(src.size(1) >= C * H && dst.is_contiguous() && dst.numel() == static_cast<int64_t>(M) * C * H);
    TORCH_CHECK(H % 8 == 0 && src.stride(0) % 8 == 0);
    TORCH_CHECK(normIters(H) <= st::kMaxIters, "chunk_norm: hidden size too large");
    check(st::invokeChunkNorm(bf16In(src), src.stride(0), bf16In(w), static_cast<float>(eps), H, C, M, bf16Out(dst),
        pdl, currentStream(src)));
}

//! dst [M, H] (row stride may exceed H) = rmsnorm(bf16(h + r)) * w, r optional.
void minimaxM3CaptureNorm(torch::Tensor const& h, std::optional<torch::Tensor> const& r, torch::Tensor const& w,
    double eps, torch::Tensor& dst)
{
    c10::cuda::CUDAGuard const deviceGuard{h.device()};
    int const M = h.size(0);
    int const H = w.size(0);
    TORCH_CHECK(
        h.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 && dst.scalar_type() == at::kBFloat16);
    TORCH_CHECK(h.dim() == 2 && h.size(1) == H && h.stride(1) == 1 && dst.dim() == 2 && dst.size(0) == M
        && dst.size(1) == H && dst.stride(1) == 1 && H % 8 == 0 && h.stride(0) % 8 == 0 && dst.stride(0) % 8 == 0);
    TORCH_CHECK(!r.has_value()
        || (r->sizes() == h.sizes() && r->stride(1) == 1 && r->stride(0) % 8 == 0
            && r->scalar_type() == at::kBFloat16));
    TORCH_CHECK(normIters(H) <= st::kMaxIters, "capture_norm: hidden size too large");
    check(st::invokeCaptureNorm(bf16In(h), h.stride(0), r.has_value() ? bf16In(*r) : nullptr,
        r.has_value() ? r->stride(0) : 0, bf16In(w), static_cast<float>(eps), H, M, bf16Out(dst), dst.stride(0),
        currentStream(h)));
}

//! The overlap scheduler's step inputs from the previous step's device outputs (see step_gather_kernel).
void minimaxM3StepGather(torch::Tensor const& newTok, torch::Tensor const& nextDraft, torch::Tensor const& lens,
    torch::Tensor const& slots, torch::Tensor const& ppi, int64_t P, int64_t T, int64_t D, torch::Tensor& ids,
    int64_t ids0, torch::Tensor& draft, int64_t d0, torch::Tensor& posOff, torch::Tensor& kvOff, int64_t b0)
{
    c10::cuda::CUDAGuard const deviceGuard{newTok.device()};
    for (auto const* t : std::initializer_list<torch::Tensor const*>{
             &newTok, &nextDraft, &lens, &slots, &ppi, &ids, &draft, &posOff, &kvOff})
    {
        TORCH_CHECK(t->scalar_type() == at::kInt, "step_gather: int32 tensors expected");
    }
    TORCH_CHECK(newTok.dim() >= 2 && nextDraft.dim() == 2 && lens.is_contiguous() && posOff.is_contiguous()
        && kvOff.is_contiguous() && ids.is_contiguous() && draft.is_contiguous());
    int const work = std::max<int>(std::max<int>(P * T, P * D), std::max<int>(posOff.numel(), kvOff.numel()));
    check(st::invokeStepGather(newTok.data_ptr<int>(), newTok.stride(0), newTok.stride(1), nextDraft.data_ptr<int>(),
        nextDraft.stride(0), nextDraft.stride(1), lens.data_ptr<int>(), slots.data_ptr<int>(), ppi.data_ptr<int>(),
        static_cast<int>(P), static_cast<int>(T), static_cast<int>(D), ids.data_ptr<int>(), static_cast<int>(ids0),
        draft.data_ptr<int>(), static_cast<int>(d0), posOff.data_ptr<int>(), static_cast<int>(posOff.numel()),
        kvOff.data_ptr<int>(), static_cast<int>(kvOff.numel()), static_cast<int>(b0), work, currentStream(newTok)));
}

//! The spec sampler's slot stores from one step's outputs (see slot_scatter_kernel).
void minimaxM3SlotScatter(torch::Tensor const& slots, torch::Tensor const& sa, torch::Tensor& da,
    torch::Tensor const& sb, torch::Tensor& db, torch::Tensor const& sl, torch::Tensor& dl, torch::Tensor const& sd,
    torch::Tensor& dd)
{
    c10::cuda::CUDAGuard const deviceGuard{slots.device()};
    for (auto const* t : std::initializer_list<torch::Tensor const*>{&sa, &da, &sb, &db, &sl, &dl, &sd, &dd})
    {
        TORCH_CHECK(t->scalar_type() == at::kInt && t->dim() <= 2, "slot_scatter: int32 tensors expected");
    }
    TORCH_CHECK(slots.scalar_type() == at::kLong);
    int const n = slots.numel();
    if (n == 0)
    {
        return;
    }
    check(st::invokeSlotScatter(reinterpret_cast<long long const*>(slots.data_ptr()), n, intView(sa), intView(da),
        intView(sb), intView(db), sl.data_ptr<int>(), dl.data_ptr<int>(), intView(sd), intView(dd),
        currentStream(slots)));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m3_draft_input(Tensor ids, Tensor table, Tensor hidden, Tensor? rows, Tensor w_embed, "
        "Tensor w_hidden, float eps_embed, float eps_hidden, Tensor(a!) out, Tensor(b!)? residual, bool pdl) -> ()");
    m.def("minimax_m3_chunk_norm(Tensor src, Tensor w, float eps, Tensor(a!) dst, bool pdl) -> ()");
    m.def("minimax_m3_capture_norm(Tensor h, Tensor? r, Tensor w, float eps, Tensor(a!) dst) -> ()");
    m.def(
        "minimax_m3_step_gather(Tensor new_tok, Tensor next_draft, Tensor lens, Tensor slots, Tensor ppi, int P, "
        "int T, int D, Tensor(a!) ids, int ids0, Tensor(b!) draft, int d0, Tensor(c!) pos_off, Tensor(d!) kv_off, "
        "int b0) -> ()");
    m.def(
        "minimax_m3_slot_scatter(Tensor slots, Tensor sa, Tensor(a!) da, Tensor sb, Tensor(b!) db, Tensor sl, "
        "Tensor(c!) dl, Tensor sd, Tensor(d!) dd) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_draft_input", &tensorrt_llm::torch_ext::minimaxM3DraftInput);
    m.impl("minimax_m3_chunk_norm", &tensorrt_llm::torch_ext::minimaxM3ChunkNorm);
    m.impl("minimax_m3_capture_norm", &tensorrt_llm::torch_ext::minimaxM3CaptureNorm);
    m.impl("minimax_m3_step_gather", &tensorrt_llm::torch_ext::minimaxM3StepGather);
    m.impl("minimax_m3_slot_scatter", &tensorrt_llm::torch_ext::minimaxM3SlotScatter);
}
