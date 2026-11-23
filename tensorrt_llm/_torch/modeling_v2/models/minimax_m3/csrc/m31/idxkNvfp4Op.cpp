/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

// Torch bindings of the MiniMax-M3.1 NVFP4 index-K writer and prefill gather (idxkNvfp4.h).

#include "minimax_m3/csrc/m31/idxkNvfp4.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

constexpr int64_t kPage = 128;
constexpr int64_t kDim = 128;
constexpr int64_t kPageBytes = kPage * kDim / 2 + kPage * (kDim / 16); // 9216

void checkPages(at::Tensor const& pages)
{
    TORCH_CHECK(pages.is_cuda() && pages.scalar_type() == at::kByte, "NVFP4 index-K pages must be a CUDA uint8 tensor");
    TORCH_CHECK(pages.numel() > 0 && pages.dim() >= 2, "NVFP4 index-K pages must be [pages, ...]");
    int64_t perPage = 1;
    for (int d = 1; d < pages.dim(); ++d)
    {
        perPage *= pages.size(d);
    }
    TORCH_CHECK(perPage == kPageBytes, "an NVFP4 index-K page must hold 9216 bytes, got ", perPage);
    // Within a page the bytes must be contiguous; the page stride is free (16-byte aligned).
    int64_t expect = 1;
    for (int d = pages.dim() - 1; d >= 1; --d)
    {
        TORCH_CHECK(pages.size(d) == 1 || pages.stride(d) == expect, "an NVFP4 index-K page must be contiguous");
        expect *= pages.size(d);
    }
    TORCH_CHECK(pages.stride(0) % 16 == 0 && reinterpret_cast<uintptr_t>(pages.data_ptr()) % 16 == 0,
        "NVFP4 index-K pages must be 16-byte aligned");
}

void launchWrite(at::Tensor const& rows, int64_t rowStride, int64_t rowsPageStride, bool rowsBySlot,
    at::Tensor const& outCacheLoc, int numTokens, at::Tensor& pages)
{
    if (numTokens == 0)
    {
        return;
    }
    c10::cuda::CUDAGuard const guard(rows.device());
    int64_t const* loc64 = outCacheLoc.scalar_type() == at::kLong ? outCacheLoc.data_ptr<int64_t>() : nullptr;
    int32_t const* loc32 = outCacheLoc.scalar_type() == at::kInt ? outCacheLoc.data_ptr<int32_t>() : nullptr;
    cudaError_t const err = kernels::minimaxM31::invokeNvfp4IdxkWrite(rows.data_ptr(),
        rows.scalar_type() != at::kBFloat16, rowStride, rowsPageStride, rowsBySlot, loc64, loc32, numTokens,
        pages.data_ptr<uint8_t>(), pages.stride(0), at::cuda::getCurrentCUDAStream(rows.get_device()));
    TORCH_CHECK(err == cudaSuccess, "NVFP4 index-K write launch failed: ", cudaGetErrorString(err));
}

} // namespace

//! Quantize index-K rows [T, 128] (bf16 or float8_e4m3fn, unit inner stride) into the NVFP4 pages of their slots
//! (out_cache_loc, int32 or int64; slots < 0 skipped).
void nvfp4IdxkWrite(at::Tensor const& rows, at::Tensor const& outCacheLoc, at::Tensor pages)
{
    checkPages(pages);
    TORCH_CHECK(rows.is_cuda() && rows.dim() == 2 && rows.size(1) == kDim && rows.stride(1) == 1,
        "index-K rows must be [T, 128] with unit inner stride");
    TORCH_CHECK(rows.scalar_type() == at::kBFloat16 || rows.scalar_type() == at::kFloat8_e4m3fn,
        "index-K rows must be bf16 or float8_e4m3fn");
    TORCH_CHECK(rows.stride(0) % 8 == 0 || rows.size(0) <= 1, "index-K row stride must keep 8-byte alignment");
    TORCH_CHECK(outCacheLoc.is_cuda() && outCacheLoc.dim() == 1 && outCacheLoc.is_contiguous(),
        "out_cache_loc must be a contiguous CUDA vector");
    TORCH_CHECK(outCacheLoc.scalar_type() == at::kInt || outCacheLoc.scalar_type() == at::kLong,
        "out_cache_loc must be int32 or int64");
    int const numTokens = static_cast<int>(rows.size(0));
    TORCH_CHECK(outCacheLoc.size(0) >= numTokens, "out_cache_loc is shorter than the rows");
    launchWrite(rows, rows.stride(0), 0, false, outCacheLoc, numTokens, pages);
}

//! The producer's FP8 sink [pages, 1, 128, 128] holds each token's E4M3 index-K at its slot: quantize the first
//! num_tokens slots of out_cache_loc from there (slots < 0 skipped).
void nvfp4IdxkWriteFromSink(at::Tensor const& sink, at::Tensor const& outCacheLoc, int64_t numTokens, at::Tensor pages)
{
    checkPages(pages);
    TORCH_CHECK(sink.is_cuda() && sink.scalar_type() == at::kFloat8_e4m3fn && sink.dim() == 4 && sink.size(1) == 1
            && sink.size(2) == kPage && sink.size(3) == kDim && sink.stride(3) == 1 && sink.stride(2) == kDim,
        "the index-K sink must be FP8 [pages, 1, 128, 128] with contiguous rows");
    TORCH_CHECK(sink.size(0) <= pages.size(0), "the index-K sink holds more pages than the NVFP4 cache");
    TORCH_CHECK(outCacheLoc.is_cuda() && outCacheLoc.dim() == 1 && outCacheLoc.is_contiguous()
            && outCacheLoc.size(0) >= numTokens && numTokens >= 0,
        "out_cache_loc must be a contiguous CUDA vector of at least num_tokens slots");
    TORCH_CHECK(outCacheLoc.scalar_type() == at::kInt || outCacheLoc.scalar_type() == at::kLong,
        "out_cache_loc must be int32 or int64");
    launchWrite(sink, sink.stride(2), sink.stride(0), true, outCacheLoc, static_cast<int>(numTokens), pages);
}

//! Dequantize the NVFP4 pages page_ids (int32; < 0 gives zeros) into out [N, 128, 128] float8_e4m3fn.
void nvfp4IdxkGatherFp8(at::Tensor const& pages, at::Tensor const& pageIds, at::Tensor out)
{
    checkPages(pages);
    TORCH_CHECK(pageIds.is_cuda() && pageIds.scalar_type() == at::kInt && pageIds.dim() == 1 && pageIds.is_contiguous(),
        "page_ids must be a contiguous CUDA int32 vector");
    int const n = static_cast<int>(pageIds.size(0));
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kFloat8_e4m3fn && out.is_contiguous()
            && out.numel() >= static_cast<int64_t>(n) * kPage * kDim,
        "out must be a contiguous CUDA float8_e4m3fn tensor of at least [N, 128, 128]");
    if (n == 0)
    {
        return;
    }
    c10::cuda::CUDAGuard const guard(pages.device());
    cudaError_t const err = kernels::minimaxM31::invokeNvfp4IdxkGatherFp8(pages.data_ptr<uint8_t>(), pages.stride(0),
        pageIds.data_ptr<int32_t>(), n, reinterpret_cast<uint8_t*>(out.data_ptr()),
        at::cuda::getCurrentCUDAStream(pages.get_device()));
    TORCH_CHECK(err == cudaSuccess, "NVFP4 index-K gather launch failed: ", cudaGetErrorString(err));
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def("m31idxk_nvfp4_idxk_write(Tensor rows, Tensor out_cache_loc, Tensor(a!) pages) -> ()");
    m.def(
        "m31idxk_nvfp4_idxk_write_from_sink(Tensor sink, Tensor out_cache_loc, int num_tokens, Tensor(a!) pages) -> "
        "()");
    m.def("m31idxk_nvfp4_idxk_gather_fp8(Tensor pages, Tensor page_ids, Tensor(a!) out) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("m31idxk_nvfp4_idxk_write", &TRTLLM_MV2_NS::torch_ext::nvfp4IdxkWrite);
    m.impl("m31idxk_nvfp4_idxk_write_from_sink", &TRTLLM_MV2_NS::torch_ext::nvfp4IdxkWriteFromSink);
    m.impl("m31idxk_nvfp4_idxk_gather_fp8", &TRTLLM_MV2_NS::torch_ext::nvfp4IdxkGatherFp8);
}
