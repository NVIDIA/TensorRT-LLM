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

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/gemm/device/gemm.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/cutlassGemmW4A16NVFP4.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/nvfp4ScaleLayout.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstddef>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace cutlass_gemm_w4a16_nvfp4
{
namespace sm120
{
namespace
{

using SizeType32 = tensorrt_llm::runtime::SizeType32;

__device__ float loadNvfp4Weight(uint8_t const* weight, __nv_fp8_e4m3 const* weightScale, float const weightGlobalScale,
    SizeType32 nIdx, SizeType32 kIdx, SizeType32 k)
{
    size_t const packedOffset = (static_cast<size_t>(nIdx) * static_cast<size_t>(k) + static_cast<size_t>(kIdx)) / 2;
    uint8_t const packed = weight[packedOffset];
    __nv_fp4_storage_t const nibble = (kIdx % 2 == 0) ? (packed & 0x0FU) : (packed >> 4U);
    half const weightValue = static_cast<half>(__nv_cvt_fp4_to_halfraw(nibble, __NV_E2M1));

    SizeType32 const scaleIdx = w4a16_nvfp4::getScaleIndex(nIdx, kIdx / w4a16_nvfp4::kScaleGranularity, k);
    float const scale = static_cast<float>(weightScale[scaleIdx]);
    return __half2float(weightValue) * scale * weightGlobalScale;
}

__global__ void dequantizeWeightToBf16ColumnMajor(__nv_bfloat16* dequantizedWeight, uint8_t const* weight,
    __nv_fp8_e4m3 const* weightScale, float const* weightGlobalScale, SizeType32 n, SizeType32 k)
{
    size_t const total = static_cast<size_t>(n) * static_cast<size_t>(k);
    float const globalScale = weightGlobalScale[0];
    for (size_t idx
         = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
         idx < total; idx += static_cast<size_t>(gridDim.x) * static_cast<size_t>(blockDim.x))
    {
        SizeType32 const nIdx = static_cast<SizeType32>(idx / static_cast<size_t>(k));
        SizeType32 const kIdx = static_cast<SizeType32>(idx % static_cast<size_t>(k));
        size_t const dequantizedOffset = static_cast<size_t>(kIdx) + static_cast<size_t>(nIdx) * static_cast<size_t>(k);
        dequantizedWeight[dequantizedOffset]
            = __float2bfloat16(loadNvfp4Weight(weight, weightScale, globalScale, nIdx, kIdx, k));
    }
}

void dequantizeWeight(Params const& params, __nv_bfloat16* dequantizedWeight, cudaStream_t stream)
{
    constexpr SizeType32 kBlockSize = 256;
    constexpr SizeType32 kMaxGridSize = 65535;
    size_t const total = static_cast<size_t>(params.n) * static_cast<size_t>(params.k);
    auto const gridSize = static_cast<SizeType32>(
        std::min((total + static_cast<size_t>(kBlockSize) - 1) / static_cast<size_t>(kBlockSize),
            static_cast<size_t>(kMaxGridSize)));
    dequantizeWeightToBf16ColumnMajor<<<gridSize, kBlockSize, 0, stream>>>(dequantizedWeight,
        reinterpret_cast<uint8_t const*>(params.weight), reinterpret_cast<__nv_fp8_e4m3 const*>(params.weightScale),
        params.weightGlobalScale, params.n, params.k);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

bool runCutlassBf16Gemm(Params const& params, __nv_bfloat16 const* dequantizedWeight, cudaStream_t stream)
{
    using ElementA = cutlass::bfloat16_t;
    using ElementB = cutlass::bfloat16_t;
    using ElementOutput = cutlass::bfloat16_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using EpilogueOp
        = cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementCompute, cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;
    using Gemm = cutlass::gemm::device::Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutOutput,
        ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, ThreadblockShape, WarpShape,
        InstructionShape, EpilogueOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 4, 8, 8>;

    typename Gemm::Arguments arguments({params.m, params.n, params.k},
        {reinterpret_cast<ElementA const*>(params.act), params.k},
        {reinterpret_cast<ElementB const*>(dequantizedWeight), params.k},
        {reinterpret_cast<ElementOutput const*>(params.output), params.n},
        {reinterpret_cast<ElementOutput*>(params.output), params.n}, {ElementCompute(1), ElementCompute(0)}, 1);

    Gemm gemm;
    cutlass::Status status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess)
    {
        TLLM_LOG_WARNING("w4a16_nvfp4 transient BF16 CUTLASS GEMM cannot implement shape m=%d n=%d k=%d: %s", params.m,
            params.n, params.k, cutlass::cutlassGetStatusString(status));
        return false;
    }

    size_t const workspaceBytes = Gemm::get_workspace_size(arguments);
    void* workspace = nullptr;
    if (workspaceBytes > 0)
    {
        TLLM_CUDA_CHECK(cudaMallocAsync(&workspace, workspaceBytes, stream));
    }

    status = gemm.initialize(arguments, workspace, stream);
    if (status == cutlass::Status::kSuccess)
    {
        status = gemm.run(stream);
    }

    if (workspace != nullptr)
    {
        TLLM_CUDA_CHECK(cudaFreeAsync(workspace, stream));
    }

    if (status != cutlass::Status::kSuccess)
    {
        TLLM_LOG_WARNING("w4a16_nvfp4 transient BF16 CUTLASS GEMM failed for shape m=%d n=%d k=%d: %s", params.m,
            params.n, params.k, cutlass::cutlassGetStatusString(status));
        return false;
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
    return true;
}

bool runTransientDequantCutlassGemm(Params const& params, cudaStream_t stream)
{
    void* dequantizedWeight = nullptr;
    size_t const dequantizedBytes
        = static_cast<size_t>(params.n) * static_cast<size_t>(params.k) * sizeof(__nv_bfloat16);
    TLLM_CUDA_CHECK(cudaMallocAsync(&dequantizedWeight, dequantizedBytes, stream));
    dequantizeWeight(params, reinterpret_cast<__nv_bfloat16*>(dequantizedWeight), stream);
    bool const dispatched
        = runCutlassBf16Gemm(params, reinterpret_cast<__nv_bfloat16 const*>(dequantizedWeight), stream);
    TLLM_CUDA_CHECK(cudaFreeAsync(dequantizedWeight, stream));
    return dispatched;
}

} // namespace

inline bool dispatch(Params const& params, cudaStream_t stream)
{
    if (params.inputType == CUDA_R_16BF && params.outputType == CUDA_R_16BF)
    {
        return runTransientDequantCutlassGemm(params, stream);
    }
    return false;
}

} // namespace sm120
} // namespace cutlass_gemm_w4a16_nvfp4
} // namespace kernels

TRTLLM_NAMESPACE_END
