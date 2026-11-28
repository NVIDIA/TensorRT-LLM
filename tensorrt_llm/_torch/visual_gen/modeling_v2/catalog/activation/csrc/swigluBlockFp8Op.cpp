/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "swigluBlockFp8.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/visual_gen/common/torchLibrary.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <limits>
#include <tuple>

TRTLLM_VG_NAMESPACE_BEGIN

namespace swiglu_block_fp8
{
namespace
{

std::tuple<at::Tensor, at::Tensor> allocateOutputs(at::Tensor const& input)
{
    TORCH_CHECK(input.dim() == 2 && input.scalar_type() == at::kBFloat16 && input.is_contiguous(),
        "vg_catalog::swiglu_block_fp8 requires a contiguous BF16 matrix");
    auto const rows = input.sym_size(0);
    auto const cols = input.sym_size(1);
    TORCH_CHECK(cols % (2 * kElemsPerLane) == 0, "Packed gate/up width must be divisible by 32");
    auto const width = cols / 2;
    auto const paddedRows = ((rows + 3) / 4) * 4;
    auto const words = (width + kColsPerWarp - 1) / kColsPerWarp;
    auto values = input.new_empty_symint({rows, width}, input.options().dtype(at::kFloat8_e4m3fn));
    auto storage = input.new_empty_symint({words, paddedRows}, input.options().dtype(at::kInt));
    return {values, storage.as_strided_symint({rows, words}, {1, paddedRows})};
}

std::tuple<at::Tensor, at::Tensor> swigluBlockFp8(at::Tensor const& input)
{
    c10::cuda::CUDAGuard const guard(input.device());
    TORCH_CHECK(common::isSM100Family(), "vg_catalog::swiglu_block_fp8 requires an SM100-family GPU");
    auto [values, scales] = allocateOutputs(input);
    int64_t const rows = values.size(0);
    int64_t const width = values.size(1);
    TORCH_CHECK(rows <= std::numeric_limits<int>::max() && width <= std::numeric_limits<int>::max(),
        "SwiGLU output dimensions must fit int32");
    if (rows > 0 && width > 0)
    {
        // A contiguous slice can start at a BF16 offset incompatible with double4 loads.
        auto aligned = reinterpret_cast<uintptr_t>(input.data_ptr()) % alignof(double4) == 0 ? input : input.clone();
        auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());
        launchSwiGLUBlockFp8(aligned.data_ptr(), values.data_ptr(), scales.data_ptr<int32_t>(), static_cast<int>(rows),
            static_cast<int>(width), scales.stride(1), stream.stream());
    }
    return {values, scales};
}

} // namespace

TRTLLM_VG_LIBRARY_FRAGMENT(m)
{
    m.def("swiglu_block_fp8(Tensor input) -> (Tensor, Tensor)");
}

TRTLLM_VG_LIBRARY_IMPL(CUDA, m)
{
    m.impl("swiglu_block_fp8", &swigluBlockFp8);
}

TRTLLM_VG_LIBRARY_IMPL(Meta, m)
{
    m.impl("swiglu_block_fp8", &allocateOutputs);
}

} // namespace swiglu_block_fp8

TRTLLM_VG_NAMESPACE_END
