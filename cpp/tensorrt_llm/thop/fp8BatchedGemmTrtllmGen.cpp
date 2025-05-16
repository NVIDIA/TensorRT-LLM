/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/KernelRunner.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/EmptyTensor.h>
#include <ATen/native/cuda/Resize.h>

#include <cuda_fp16.h>

#include <cstdint>

namespace
{
namespace tg = trtllm::gen;

template <tg::Dtype outDtype>
void runBatchedGemm(at::Tensor& out, at::Tensor& outSfC, at::Tensor const& mat1, at::Tensor const& mat2,
    std::optional<at::Tensor> const& dDqSfsA, std::optional<at::Tensor> const& dDqSfsB,
    std::optional<at::Tensor> const& scaleC, int64_t m, int64_t n, int64_t k, int32_t tileSize, int32_t epilogueTileM,
    std::vector<int32_t> const& batchedTokens, bool useDeepSeekFp8, bool lowLatencyKernel)
{
    auto eltType = tg::Dtype::E4m3;

    tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions options = {.eltType = eltType,
        .outputType = outDtype,
        .deepSeekFp8 = useDeepSeekFp8,
        .fusedAct = false,
        .routeAct = false,
        .staticBatch = true,
        .transposeMmaOutput = lowLatencyKernel,
        .tileSize = tileSize,
        .epilogueTileM = epilogueTileM};

    tensorrt_llm::kernels::TrtllmGenBatchedGemmRunner runner(options);

    // numTokens and maxNumCtasInBatchDim are not used for static batching
    int32_t numTokens = 0;
    int32_t maxNumCtasInBatchDim = 0;
    int64_t const numBytesWorkspace
        = runner.getWorkspaceSizeInBytes(m, n, k, batchedTokens, numTokens, batchedTokens.size(), maxNumCtasInBatchDim);
    at::Tensor workspace
        = at::detail::empty_cuda({numBytesWorkspace}, at::ScalarType::Char, mat1.device(), std::nullopt);

    auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());

    if (useDeepSeekFp8)
    {
        float* outSfCPtr = outDtype == tg::Dtype::E4m3 ? outSfC.data_ptr<float>() : nullptr;
        runner.run(m, n, k, batchedTokens, mat1.const_data_ptr(), dDqSfsA.value().const_data_ptr(),
            mat2.const_data_ptr(), dDqSfsB.value().const_data_ptr(), out.data_ptr(), outSfCPtr, workspace.data_ptr(),
            stream.stream(), mat1.get_device());
    }
    else
    {
        runner.run(m, n, k, batchedTokens, mat1.const_data_ptr(), mat2.const_data_ptr(),
            reinterpret_cast<float const*>(scaleC.value().const_data_ptr()), nullptr, out.data_ptr(),
            workspace.data_ptr(), stream.stream(), mat1.get_device());
    }
}

std::tuple<at::Tensor, at::Tensor> fp8_batched_gemm_sm100(at::Tensor const& mat1, at::Tensor const& mat2,
    int32_t tileSize, bool useDeepSeekFp8, bool lowLatencyKernel, int64_t epilogueTileM,
    std::optional<at::Tensor> const& dDqSfsA, std::optional<at::Tensor> const& dDqSfsB,
    std::optional<at::Tensor> const& scaleC, std::optional<c10::ScalarType> outDtype)
{
    TORCH_CHECK(mat1.dim() == 3, "Matrix A must be of size [B, M, K]");
    TORCH_CHECK(mat2.dim() == 3, "Matrix B must be of size [B, N, K]");

    auto const dimsA = mat1.sizes();
    auto const dimsB = mat2.sizes();
    int64_t const b = dimsB[0];
    int64_t const m = dimsA[1];
    int64_t const n = dimsB[1];
    int64_t const k = dimsB[2];

    if (!outDtype)
    {
        outDtype = torch::kHalf;
    }

    TORCH_CHECK(outDtype == at::ScalarType::Float8_e4m3fn || outDtype == torch::kHalf || outDtype == torch::kBFloat16,
        "outDtype must be one of fp16/bf16/e4m3. It defaults to fp16.");

    TORCH_CHECK(mat1.scalar_type() == at::ScalarType::Float8_e4m3fn, "Matrix A dtype must be FP8.");
    TORCH_CHECK(mat2.scalar_type() == at::ScalarType::Float8_e4m3fn, "Matrix B dtype must be FP8.");

    TORCH_CHECK(mat1.sizes()[2] == mat2.sizes()[2], "A and B shapes cannot be multiplied (", mat1.sizes()[0], "x",
        mat1.sizes()[1], "x", mat1.sizes()[2], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], "x", mat2.sizes()[2],
        ")");

    TORCH_CHECK(m % tileSize == 0, "M must be a multiple of tileSize");
    TORCH_CHECK(tileSize <= std::numeric_limits<int32_t>::max(), "tileSize must be within int32");
    TORCH_CHECK(m <= std::numeric_limits<int32_t>::max(), "M must be within int32");
    TORCH_CHECK(n <= std::numeric_limits<int32_t>::max(), "N must be within int32");
    TORCH_CHECK(k <= std::numeric_limits<int32_t>::max(), "K must be within int32");

    int32_t constexpr dsFp8QuantBlockSize = 128;
    if (useDeepSeekFp8)
    {
        TORCH_CHECK(n % dsFp8QuantBlockSize == 0, "N must be a multiple of ", dsFp8QuantBlockSize, ", (N=", n, ")");
        TORCH_CHECK(k % dsFp8QuantBlockSize == 0, "K must be a multiple of ", dsFp8QuantBlockSize, ", (K=", k, ")");
        TORCH_CHECK(dDqSfsA.has_value(), "dDqSfsA must be provided for DeepSeek FP8.");
        TORCH_CHECK(dDqSfsB.has_value(), "dDqSfsB must be provided for DeepSeek FP8.");
        TORCH_CHECK(dDqSfsA.value().scalar_type() == at::ScalarType::Float, "Scale dtype must be FP32.");
        TORCH_CHECK(dDqSfsB.value().scalar_type() == at::ScalarType::Float, "Scale dtype must be FP32.");
        TORCH_CHECK(dDqSfsA.value().dim() == 2, "batching M: dDqSfsA must be a 2D matrix");
        TORCH_CHECK(dDqSfsA.value().sizes()[0] == k / dsFp8QuantBlockSize,
            "batching M: dDqSfsA must have size B x K/dsFp8QuantBlockSize x divUp(m, dsFp8QuantBlockSize) * 128 * b");
        TORCH_CHECK(
            dDqSfsA.value().sizes()[1] == static_cast<int64_t>(tensorrt_llm::common::divUp(m, tileSize) * tileSize * b),
            "batching M: dDqSfsA must have size B x K/dsFp8QuantBlockSize x divUp(m, tileSize) * tileSize * b");

        TORCH_CHECK(dDqSfsB.value().dim() == 3, "batching M: dDqSfsB must be a 3D matrix");
        TORCH_CHECK(dDqSfsB.value().sizes()[0] == b,
            "batching M: dDqSfsB must have size B x N/dsFp8QuantBlockSize x K/dsFp8QuantBlockSize");
        TORCH_CHECK(dDqSfsB.value().sizes()[1] == n / dsFp8QuantBlockSize,
            "batching M: dDqSfsB must have size B x N/dsFp8QuantBlockSize x K/dsFp8QuantBlockSize");
        TORCH_CHECK(dDqSfsB.value().sizes()[2] == k / dsFp8QuantBlockSize,
            "batching M: dDqSfsB must have size B x N/dsFp8QuantBlockSize x K/dsFp8QuantBlockSize");
    }
    else
    {
        TORCH_CHECK(scaleC.has_value(), "scaleC must be provided for non DeepSeek FP8.");
        TORCH_CHECK(scaleC.value().scalar_type() == at::ScalarType::Float, "Scale dtype must be FP32.");
        TORCH_CHECK(scaleC.value().dim() == 1, "outScalingFactor must be a 1D matrix of size B");
        TORCH_CHECK(scaleC.value().sizes()[0] == b, "outScalingFactor must be a 1D matrix of size B");
    }

    int64_t outputN = n;

    // Create output tensor.
    at::Tensor out = at::detail::empty_cuda({b, m, outputN}, outDtype.value(), mat1.device(), std::nullopt);
    at::Tensor outSfC;
    if (useDeepSeekFp8 && outDtype.value() == at::ScalarType::Float8_e4m3fn)
    {
        outSfC = at::detail::empty_cuda(
            {outputN / dsFp8QuantBlockSize, m * b}, at::ScalarType::Float, mat1.device(), std::nullopt);
    }

    std::vector<int32_t> batchedTokens(b, m);

    switch (outDtype.value())
    {
    case at::ScalarType::Half:
        runBatchedGemm<tg::Dtype::Fp16>(out, outSfC, mat1, mat2, dDqSfsA, dDqSfsB, scaleC, m, n, k, tileSize,
            epilogueTileM, batchedTokens, useDeepSeekFp8, lowLatencyKernel);
        break;
    case at::ScalarType::BFloat16:
        runBatchedGemm<tg::Dtype::Bfloat16>(out, outSfC, mat1, mat2, dDqSfsA, dDqSfsB, scaleC, m, n, k, tileSize,
            epilogueTileM, batchedTokens, useDeepSeekFp8, lowLatencyKernel);
        break;
    case at::ScalarType::Float8_e4m3fn:
        runBatchedGemm<tg::Dtype::E4m3>(out, outSfC, mat1, mat2, dDqSfsA, dDqSfsB, scaleC, m, n, k, tileSize,
            epilogueTileM, batchedTokens, useDeepSeekFp8, lowLatencyKernel);
        break;
    default: C10_THROW_ERROR(NotImplementedError, "outDtype must be one of fp16/bf16/e4m3.");
    }

    return {out, outSfC};
}
} // namespace

namespace torch_ext
{

extern std::tuple<at::Tensor, at::Tensor> fp8_batched_gemm_trtllmgen(at::Tensor const& mat1, at::Tensor const& mat2,
    int64_t tileSize, bool useDeepSeekFp8, bool lowLatency, int64_t epilogueTileM,
    std::optional<at::Tensor> const& dDqSfsA, std::optional<at::Tensor> const& dDqSfsB,
    std::optional<at::Tensor> const& scaleC, std::optional<c10::ScalarType> outDtype)
{
    auto const smVersion = tensorrt_llm::common::getSMVersion();
    switch (smVersion)
    {
    case tensorrt_llm::kernels::kSM_100:
    {
        return fp8_batched_gemm_sm100(
            mat1, mat2, tileSize, useDeepSeekFp8, lowLatency, epilogueTileM, dDqSfsA, dDqSfsB, scaleC, outDtype);
    }
    default: TLLM_THROW("Unsupported or unimplemented compute capability for fp8 batched gemm: %i", smVersion);
    }
}
} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fp8_batched_gemm_trtllmgen(Tensor a, Tensor b, int tile_size,"
        "bool use_deep_seek_fp8=False, bool low_latency=False, "
        "int epilogue_tile_m=0, Tensor? dq_sfs_a=None, Tensor? dq_sfs_b=None, "
        "Tensor? scale_c=None, "
        "ScalarType? out_dtype=None) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fp8_batched_gemm_trtllmgen", &torch_ext::fp8_batched_gemm_trtllmgen);
}
