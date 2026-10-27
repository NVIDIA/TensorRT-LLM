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

#include "tensorrt_llm/kernels/minimaxM31/markovEpilogueSm107.h"
#include "tensorrt_llm/kernels/minimaxM31/scatterRowsSm107.h"
#include "tensorrt_llm/kernels/minimaxM31/tailBk3Sm107.h"
#include "tensorrt_llm/kernels/minimaxM31/zeroPaddingRowsSm107.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

bool aligned(void const* p, uintptr_t bytes)
{
    return reinterpret_cast<uintptr_t>(p) % bytes == 0;
}

} // namespace

//! S4FE (see kernels/minimaxM31/markovEpilogueSm107.h): one position k of the in-tree Markov chain's tail.
//! base / out: fp32 [rows, K, d] (out may be base's same-shape output buffer); bias: bf16 [rows, d] (the in-tree
//! F.linear output); prev: int64 [rows] (written); w1: bf16 [vocab, rank]; emb_out: bf16 [rows, rank] or None;
//! partials: a CUDA buffer of >= rows * splits * 8 bytes; counters: int32 [>= rows], zero (left zero).
void minimaxM31MarkovEpilogue(torch::Tensor const& base, int64_t k, torch::Tensor const& bias, torch::Tensor& out,
    torch::Tensor& prev, torch::Tensor const& w1, std::optional<torch::Tensor> embOut, torch::Tensor& partials,
    torch::Tensor& counters, int64_t splits)
{
    TORCH_CHECK(base.is_cuda() && (base.scalar_type() == torch::kFloat32 || base.scalar_type() == torch::kBFloat16)
            && base.dim() == 3 && base.is_contiguous(),
        "minimax_m31_markov_epilogue: base must be contiguous float32 or bfloat16 [rows, K, d] CUDA");
    int64_t const rows = base.size(0);
    int64_t const block = base.size(1);
    int64_t const d = base.size(2);
    TORCH_CHECK(k >= 0 && k < block, "minimax_m31_markov_epilogue: position k out of range");
    TORCH_CHECK(d % 4 == 0 && rows >= 1 && rows <= 65535 && rows * block * d <= std::numeric_limits<int64_t>::max() / 4
            && d <= std::numeric_limits<int32_t>::max(),
        "minimax_m31_markov_epilogue: width must be divisible by 4 and rows in 1..65535");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat32 && out.sizes() == base.sizes()
            && out.is_contiguous() && out.get_device() == base.get_device(),
        "minimax_m31_markov_epilogue: out must be a contiguous float32 tensor shaped like base on its device");
    TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == torch::kBFloat16 && bias.dim() == 2 && bias.is_contiguous()
            && bias.size(0) == rows && bias.size(1) == d && bias.get_device() == base.get_device(),
        "minimax_m31_markov_epilogue: bias must be a contiguous bf16 [rows, d] tensor on base's device");
    TORCH_CHECK(prev.is_cuda() && prev.scalar_type() == torch::kInt64 && prev.dim() == 1 && prev.size(0) == rows
            && prev.is_contiguous(),
        "minimax_m31_markov_epilogue: prev must be a contiguous int64 [rows] tensor");
    TORCH_CHECK(w1.is_cuda() && w1.scalar_type() == torch::kBFloat16 && w1.dim() == 2 && w1.is_contiguous()
            && w1.size(0) >= d && w1.size(1) >= 1 && w1.size(1) <= 1024,
        "minimax_m31_markov_epilogue: w1 must be a contiguous bf16 [vocab >= d, rank <= 1024] tensor");
    __nv_bfloat16* emb = nullptr;
    if (embOut.has_value())
    {
        auto& e = embOut.value();
        TORCH_CHECK(e.is_cuda() && e.scalar_type() == torch::kBFloat16 && e.is_contiguous() && e.dim() == 2
                && e.size(0) == rows && e.size(1) == w1.size(1),
            "minimax_m31_markov_epilogue: emb_out must be a contiguous bf16 [rows, rank] tensor");
        emb = reinterpret_cast<__nv_bfloat16*>(e.data_ptr());
    }
    TORCH_CHECK(splits >= 1 && splits <= 65535, "minimax_m31_markov_epilogue: splits must be in 1..65535");
    TORCH_CHECK(partials.is_cuda() && partials.nbytes() >= static_cast<size_t>(rows * splits * 8),
        "minimax_m31_markov_epilogue: partials is too small");
    TORCH_CHECK(counters.is_cuda() && counters.scalar_type() == torch::kInt32 && counters.numel() >= rows,
        "minimax_m31_markov_epilogue: counters must be an int32 tensor of >= rows entries");
    auto const* basePtr = static_cast<char const*>(base.data_ptr()) + k * d * base.element_size();
    float* outPtr = out.data_ptr<float>() + k * d;
    TORCH_CHECK(aligned(basePtr, 16) && aligned(outPtr, 16) && aligned(bias.data_ptr(), 8),
        "minimax_m31_markov_epilogue: base / out rows must be 16-byte aligned and bias rows 8-byte aligned");

    at::cuda::CUDAGuard const guard(base.device());
    if (base.scalar_type() == torch::kFloat32)
    {
        tensorrt_llm::kernels::minimaxM31::invokeMarkovEpilogue(reinterpret_cast<float const*>(basePtr), block * d,
            reinterpret_cast<__nv_bfloat16 const*>(bias.data_ptr()), outPtr, block * d, prev.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16 const*>(w1.data_ptr()), static_cast<int32_t>(w1.size(1)), emb,
            reinterpret_cast<tensorrt_llm::kernels::minimaxM31::MarkovEpilogueBest*>(partials.data_ptr()),
            counters.data_ptr<int32_t>(), static_cast<int32_t>(rows), static_cast<int32_t>(d),
            static_cast<int32_t>(splits), at::cuda::getCurrentCUDAStream(base.get_device()));
    }
    else
    {
        tensorrt_llm::kernels::minimaxM31::invokeMarkovEpilogue(reinterpret_cast<__nv_bfloat16 const*>(basePtr),
            block * d, reinterpret_cast<__nv_bfloat16 const*>(bias.data_ptr()), outPtr, block * d,
            prev.data_ptr<int64_t>(), reinterpret_cast<__nv_bfloat16 const*>(w1.data_ptr()),
            static_cast<int32_t>(w1.size(1)), emb,
            reinterpret_cast<tensorrt_llm::kernels::minimaxM31::MarkovEpilogueBest*>(partials.data_ptr()),
            counters.data_ptr<int32_t>(), static_cast<int32_t>(rows), static_cast<int32_t>(d),
            static_cast<int32_t>(splits), at::cuda::getCurrentCUDAStream(base.get_device()));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

//! The kernel's default split count for rows x d on this device.
int64_t minimaxM31MarkovEpilogueSplits(int64_t rows, int64_t d)
{
    int const sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    return tensorrt_llm::kernels::minimaxM31::markovEpilogueSplits(
        static_cast<int32_t>(rows), static_cast<int32_t>(d), sms);
}

//! TAIL16-G2 (see kernels/minimaxM31/zeroPaddingRowsSm107.h): in place, logits[r] = 0 for every row r whose request
//! slotIds[r / rows_per_request] is dummy_slot_row. logits: fp32 [rows, d] with unit column stride; slot_ids: int64
//! [rows / rows_per_request] (the generation requests' slot ids).
void minimaxM31ZeroPaddingRows(
    torch::Tensor& logits, torch::Tensor const& slotIds, int64_t dummySlotRow, int64_t rowsPerRequest)
{
    TORCH_CHECK(
        logits.is_cuda() && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2 && logits.stride(1) == 1,
        "minimax_m31_zero_padding_rows: logits must be a 2-D float32 CUDA tensor with unit column stride");
    TORCH_CHECK(rowsPerRequest >= 1 && logits.size(0) % rowsPerRequest == 0 && logits.size(0) <= 65535,
        "minimax_m31_zero_padding_rows: rows must be a multiple of rows_per_request (<= 65535 rows)");
    TORCH_CHECK(slotIds.is_cuda() && slotIds.scalar_type() == torch::kInt64 && slotIds.dim() == 1
            && slotIds.is_contiguous() && slotIds.size(0) == logits.size(0) / rowsPerRequest
            && slotIds.get_device() == logits.get_device(),
        "minimax_m31_zero_padding_rows: slot_ids must be a contiguous int64 [rows / rows_per_request] tensor");
    TORCH_CHECK(logits.size(1) <= std::numeric_limits<int32_t>::max(), "minimax_m31_zero_padding_rows: too wide");
    if (logits.size(0) == 0)
    {
        return;
    }
    at::cuda::CUDAGuard const guard(logits.device());
    tensorrt_llm::kernels::minimaxM31::invokeZeroPaddingRows(logits.data_ptr<float>(), logits.stride(0),
        slotIds.data_ptr<int64_t>(), dummySlotRow, static_cast<int32_t>(logits.size(0)),
        static_cast<int32_t>(rowsPerRequest), static_cast<int32_t>(logits.size(1)),
        at::cuda::getCurrentCUDAStream(logits.get_device()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

//! TAIL16-G3 (see kernels/minimaxM31/scatterRowsSm107.h): dst[slot_ids[i], :K, :v] = src[i] for src fp32 [n, K, v]
//! contiguous; dst fp32 [slots, >= K, >= v] with a unit last-dim stride. slot_ids int64 [n] on dst's device.
void minimaxM31ScatterRows(torch::Tensor& dst, torch::Tensor const& slotIds, torch::Tensor const& src)
{
    TORCH_CHECK(src.is_cuda() && src.scalar_type() == torch::kFloat32 && src.dim() == 3 && src.is_contiguous(),
        "minimax_m31_scatter_rows: src must be a contiguous float32 [n, K, v] CUDA tensor");
    TORCH_CHECK(dst.is_cuda() && dst.scalar_type() == torch::kFloat32 && dst.dim() == 3 && dst.stride(2) == 1
            && dst.size(1) >= src.size(1) && dst.size(2) >= src.size(2) && dst.get_device() == src.get_device(),
        "minimax_m31_scatter_rows: dst must be a float32 [slots, >= K, >= v] tensor with unit last-dim stride");
    TORCH_CHECK(slotIds.is_cuda() && slotIds.scalar_type() == torch::kInt64 && slotIds.dim() == 1
            && slotIds.is_contiguous() && slotIds.size(0) == src.size(0) && slotIds.get_device() == src.get_device(),
        "minimax_m31_scatter_rows: slot_ids must be a contiguous int64 [n] tensor");
    TORCH_CHECK(src.size(0) * src.size(1) <= 65535 && src.size(2) <= std::numeric_limits<int32_t>::max(),
        "minimax_m31_scatter_rows: too many rows");
    if (src.numel() == 0)
    {
        return;
    }
    at::cuda::CUDAGuard const guard(src.device());
    tensorrt_llm::kernels::minimaxM31::invokeScatterRows(dst.data_ptr<float>(), dst.stride(0), dst.stride(1),
        src.data_ptr<float>(), slotIds.data_ptr<int64_t>(), static_cast<int32_t>(src.size(0)),
        static_cast<int32_t>(src.size(1)), static_cast<int32_t>(src.size(2)),
        at::cuda::getCurrentCUDAStream(src.get_device()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

//! TAIL_BK3 (B): emb = w1[prev] with out-of-range ids as +0 rows (bitwise the in-tree masked embedding).
//! prev: int64 [n]; w1: contiguous bf16 [vocab, rank]; returns bf16 [n, rank].
torch::Tensor minimaxM31MarkovPrevEmbed(torch::Tensor const& prev, torch::Tensor const& w1)
{
    TORCH_CHECK(prev.is_cuda() && prev.scalar_type() == torch::kInt64 && prev.dim() == 1 && prev.is_contiguous(),
        "minimax_m31_markov_prev_embed: prev must be a contiguous int64 [n] CUDA tensor");
    TORCH_CHECK(w1.is_cuda() && w1.scalar_type() == torch::kBFloat16 && w1.dim() == 2 && w1.is_contiguous()
            && w1.get_device() == prev.get_device() && w1.size(1) >= 1
            && w1.size(1) <= std::numeric_limits<int32_t>::max(),
        "minimax_m31_markov_prev_embed: w1 must be a contiguous bf16 [vocab, rank] tensor on prev's device");
    TORCH_CHECK(prev.size(0) <= std::numeric_limits<int32_t>::max(), "minimax_m31_markov_prev_embed: too many rows");
    auto out = torch::empty({prev.size(0), w1.size(1)}, w1.options());
    if (prev.size(0) == 0)
    {
        return out;
    }
    at::cuda::CUDAGuard const guard(prev.device());
    tensorrt_llm::kernels::minimaxM31::invokeMarkovPrevEmbed(prev.data_ptr<int64_t>(),
        static_cast<int32_t>(prev.size(0)), reinterpret_cast<__nv_bfloat16 const*>(w1.data_ptr()), w1.size(0),
        static_cast<int32_t>(w1.size(1)), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        at::cuda::getCurrentCUDAStream(prev.get_device()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

//! TAIL_BK3 (C): the block sampler's [n * K] parameters (bitwise repeat_interleave / _rng_state_per_request).
//! Inputs: the request slices temps f32, top_ks i32, top_ps f32, min_ps f32, seeds i64, offsets i64, each contiguous
//! [n]; slot_offset = slot * span. Returns (temps, top_ks, top_ps, min_ps, seeds, offsets), each [n * K].
std::vector<torch::Tensor> minimaxM31ExpandBlockSampling(torch::Tensor const& temps, torch::Tensor const& topKs,
    torch::Tensor const& topPs, torch::Tensor const& minPs, torch::Tensor const& seeds, torch::Tensor const& offsets,
    int64_t K, int64_t slotOffset, int64_t span)
{
    auto check = [&](torch::Tensor const& t, torch::ScalarType dt, char const* name)
    {
        TORCH_CHECK(t.is_cuda() && t.scalar_type() == dt && t.dim() == 1 && t.is_contiguous()
                && t.size(0) == temps.size(0) && t.get_device() == temps.get_device(),
            "minimax_m31_expand_block_sampling: ", name, " must be a contiguous 1-D CUDA tensor of n entries");
    };
    check(temps, torch::kFloat32, "temps");
    check(topKs, torch::kInt32, "top_ks");
    check(topPs, torch::kFloat32, "top_ps");
    check(minPs, torch::kFloat32, "min_ps");
    check(seeds, torch::kInt64, "seeds");
    check(offsets, torch::kInt64, "offsets");
    int64_t const n = temps.size(0);
    TORCH_CHECK(K >= 1 && n * K <= std::numeric_limits<int32_t>::max(), "minimax_m31_expand_block_sampling: bad K");
    int64_t const total = n * K;
    auto tOut = torch::empty({total}, temps.options());
    auto kOut = torch::empty({total}, topKs.options());
    auto pOut = torch::empty({total}, topPs.options());
    auto mOut = torch::empty({total}, minPs.options());
    auto sOut = torch::empty({total}, seeds.options());
    auto oOut = torch::empty({total}, offsets.options());
    if (total > 0)
    {
        at::cuda::CUDAGuard const guard(temps.device());
        tensorrt_llm::kernels::minimaxM31::invokeExpandBlockSampling(temps.data_ptr<float>(), topKs.data_ptr<int32_t>(),
            topPs.data_ptr<float>(), minPs.data_ptr<float>(), seeds.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
            static_cast<int32_t>(n), static_cast<int32_t>(K), slotOffset, span, tOut.data_ptr<float>(),
            kOut.data_ptr<int32_t>(), pOut.data_ptr<float>(), mOut.data_ptr<float>(), sOut.data_ptr<int64_t>(),
            oOut.data_ptr<int64_t>(), at::cuda::getCurrentCUDAStream(temps.get_device()));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {tOut, kOut, pOut, mOut, sOut, oOut};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m31_markov_epilogue(Tensor base, int k, Tensor bias, Tensor(a!) out, Tensor(b!) prev, Tensor w1, "
        "Tensor(c!)? emb_out, Tensor(d!) partials, Tensor(e!) counters, int splits) -> ()");
    m.def(
        "minimax_m31_markov_epilogue_bf16(Tensor base, int k, Tensor bias, Tensor(a!) out, Tensor(b!) prev, Tensor w1, "
        "Tensor(c!)? emb_out, Tensor(d!) partials, Tensor(e!) counters, int splits) -> ()");
    m.def(
        "minimax_m31_zero_padding_rows(Tensor(a!) logits, Tensor slot_ids, int dummy_slot_row, int rows_per_request) "
        "-> ()");
    m.def("minimax_m31_scatter_rows(Tensor(a!) dst, Tensor slot_ids, Tensor src) -> ()");
    m.def("minimax_m31_markov_prev_embed(Tensor prev, Tensor w1) -> Tensor");
    m.def(
        "minimax_m31_expand_block_sampling(Tensor temps, Tensor top_ks, Tensor top_ps, Tensor min_ps, Tensor seeds, "
        "Tensor offsets, int K, int slot_offset, int span) -> Tensor[]");
    m.def("minimax_m31_markov_epilogue_splits(int rows, int d) -> int",
        &tensorrt_llm::torch_ext::minimaxM31MarkovEpilogueSplits);
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m31_markov_epilogue", &tensorrt_llm::torch_ext::minimaxM31MarkovEpilogue);
    m.impl("minimax_m31_markov_epilogue_bf16", &tensorrt_llm::torch_ext::minimaxM31MarkovEpilogue);
    m.impl("minimax_m31_zero_padding_rows", &tensorrt_llm::torch_ext::minimaxM31ZeroPaddingRows);
    m.impl("minimax_m31_scatter_rows", &tensorrt_llm::torch_ext::minimaxM31ScatterRows);
    m.impl("minimax_m31_markov_prev_embed", &tensorrt_llm::torch_ext::minimaxM31MarkovPrevEmbed);
    m.impl("minimax_m31_expand_block_sampling", &tensorrt_llm::torch_ext::minimaxM31ExpandBlockSampling);
}
