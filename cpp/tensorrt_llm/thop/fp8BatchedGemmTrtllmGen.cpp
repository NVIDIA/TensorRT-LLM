/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/kernelParams.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/kernelRunner.h"
#include <ATen/ATen.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/ops/zeros.h>
#include <algorithm>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <torch/custom_class.h>

#include <cuda_fp16.h>
#include <vector>

namespace
{
at::Tensor fp8_batched_gemm_sm100(at::Tensor& inputBatchA, int64_t m, at::Tensor const& dsPerInputAScalingFactors,
    at::Tensor& inputBatchB, int64_t n, at::Tensor const& dsPerInputBScalingFactors,
    at::Tensor const& dsPerOutputScalingFactors, at::Tensor const& outScalingFactor, int64_t tileSize,
    bool quantizeOutput, bool useDeepSeekFp8, bool batchM)
{
    tensorrt_llm::kernels::Data_type dtypeC;
    at::ScalarType dtypeCTorch;
    if (quantizeOutput)
    {
        dtypeC = tensorrt_llm::kernels::Data_type::DATA_TYPE_E4M3;
        dtypeCTorch = at::ScalarType::Float8_e4m3fn;
    }
    else
    {
        dtypeC = tensorrt_llm::kernels::Data_type::DATA_TYPE_BF16;
        dtypeCTorch = at::ScalarType::BFloat16;
    }

    TORCH_CHECK(inputBatchA.scalar_type() == at::ScalarType::Float8_e4m3fn, "Matrix A dtype must be FP8.");
    TORCH_CHECK(inputBatchB.scalar_type() == at::ScalarType::Float8_e4m3fn, "Matrix B dtype must be FP8.");

    TORCH_CHECK(inputBatchA.dim() == 3, "Matrix A must be of size [B*M/ts*K]");
    TORCH_CHECK(inputBatchB.dim() == 3, "Matrix B must be of size [B*N/ts*K]");

    TORCH_CHECK(inputBatchA.sizes()[2] == inputBatchB.sizes()[2], "A and B shapes cannot be multiplied (",
        inputBatchA.sizes()[0], "x", inputBatchA.sizes()[1], "x", inputBatchA.sizes()[2], " and ",
        inputBatchB.sizes()[0], "x", inputBatchB.sizes()[1], "x", inputBatchB.sizes()[2], ")");

    auto const dimsA = inputBatchA.sizes();
    auto const dimsB = inputBatchB.sizes();
    int64_t const b = dimsB[0];
    int64_t const mPadded = dimsA[1];
    int64_t const nPadded = dimsB[1];
    int64_t const k = dimsB[2];

    TORCH_CHECK(b <= tensorrt_llm::kernels::TrtllmGenBatchedGemmKernelParams::MaxBatchSize, "BMM max batch size is ",
        tensorrt_llm::kernels::TrtllmGenBatchedGemmKernelParams::MaxBatchSize);
    TORCH_CHECK(mPadded <= std::numeric_limits<int32_t>::max(), "M must be within int32");
    TORCH_CHECK(nPadded <= std::numeric_limits<int32_t>::max(), "N must be within int32");
    TORCH_CHECK(k <= std::numeric_limits<int32_t>::max(), "K must be within int32");

    if (batchM)
    {
        TORCH_CHECK(n % tileSize == 0, "N must be a multiple of ", tileSize, ", (N=", n, ")");
    }
    else
    {
        TORCH_CHECK(m % tileSize == 0, "M must be a multiple of ", tileSize, ", (M=", m, ")");
    }

    TORCH_CHECK(k % tileSize == 0, "K must be a multiple of ", tileSize, ", (K=", k, ")");

    float* ptrScaleC = nullptr;
    float* dDqSfsA = nullptr;
    float* dDqSfsB = nullptr;
    float* dDqSfsC = nullptr;

    int64_t const outputM = batchM ? mPadded : nPadded;
    int64_t const outputN = batchM ? nPadded : mPadded;

    if (useDeepSeekFp8)
    {
        TORCH_CHECK(dsPerInputAScalingFactors.scalar_type() == at::ScalarType::Float, "Scale dtype must be FP32.");
        TORCH_CHECK(dsPerInputBScalingFactors.scalar_type() == at::ScalarType::Float, "Scale dtype must be FP32.");
        TORCH_CHECK(dsPerOutputScalingFactors.scalar_type() == at::ScalarType::Float, "Scale dtype must be FP32.");

        if (batchM)
        {
            TORCH_CHECK(
                dsPerInputAScalingFactors.dim() == 2, "batching M: dsPerInputAScalingFactors must be a 2D matrix");
            TORCH_CHECK(dsPerInputAScalingFactors.sizes()[0] == k / tileSize,
                "batching M: dsPerInputAScalingFactors must have size B x K/tileSize x divUp(m, tileSize) * 128 * b");
            TORCH_CHECK(dsPerInputAScalingFactors.sizes()[1]
                    == (int64_t) tensorrt_llm::common::divUp(m, tileSize) * tileSize * b,
                "batching M: dsPerInputAScalingFactors must have size B x K/tileSize x divUp(m, tileSize) * 128 * b");

            TORCH_CHECK(
                dsPerInputBScalingFactors.dim() == 3, "batching M: dsPerInputBScalingFactors must be a 3D matrix");
            TORCH_CHECK(dsPerInputBScalingFactors.sizes()[0] == b,
                "batching M: dsPerInputBScalingFactors must have size B x N/tileSize x K/tileSize");
            TORCH_CHECK(dsPerInputBScalingFactors.sizes()[1] == n / tileSize,
                "batching M: dsPerInputBScalingFactors must have size B x N/tileSize x K/tileSize");
            TORCH_CHECK(dsPerInputBScalingFactors.sizes()[2] == k / tileSize,
                "batching M: dsPerInputBScalingFactors must have size B x N/tileSize x K/tileSize");

            TORCH_CHECK(
                dsPerOutputScalingFactors.dim() == 3, "batching M: dsPerOutputScalingFactors must be a 3D matrix");
            TORCH_CHECK(dsPerOutputScalingFactors.sizes()[0] == b,
                "batching M: dsPerOutputScalingFactors must have size B x N/tileSize x divUp(m, tileSize) * 128 * b");
            TORCH_CHECK(dsPerOutputScalingFactors.sizes()[1] == n / tileSize,
                "batching M: dsPerOutputScalingFactors must have size B x N/tileSize x divUp(m, tileSize) * 128 * b");
            TORCH_CHECK(
                dsPerOutputScalingFactors.sizes()[2] == (int64_t) tensorrt_llm::common::divUp(m, tileSize) * tileSize,
                "batching M: dsPerOutputScalingFactors must have size B x N/tileSize x divUp(m, tileSize) * 128 * b");
        }
        else
        {
            TORCH_CHECK(
                dsPerInputAScalingFactors.dim() == 3, "batching N: dsPerInputAScalingFactors must be a 3D matrix");
            TORCH_CHECK(dsPerInputAScalingFactors.sizes()[0] == b,
                "batching N: dsPerInputAScalingFactors must have size B x M/tileSize x K/tileSize");
            TORCH_CHECK(dsPerInputAScalingFactors.sizes()[1] == m / tileSize,
                "batching N: dsPerInputAScalingFactors must have size B x M/tileSize x K/tileSize");
            TORCH_CHECK(dsPerInputAScalingFactors.sizes()[2] == k / tileSize,
                "batching N: dsPerInputAScalingFactors must have size B x M/tileSize x K/tileSize");

            TORCH_CHECK(
                dsPerInputBScalingFactors.dim() == 2, "batching N: dsPerInputBScalingFactors must be a 2D matrix");
            TORCH_CHECK(dsPerInputBScalingFactors.sizes()[0] == k / tileSize,
                "batching N: dsPerInputBScalingFactors must have size K/tileSize x divUp(n, tileSize) * 128 * b");
            TORCH_CHECK(dsPerInputBScalingFactors.sizes()[1]
                    == (int64_t) tensorrt_llm::common::divUp(n, tileSize) * tileSize * b,
                "batching N: dsPerInputBScalingFactors must have size K/tileSize x divUp(n, tileSize) * 128 * b");

            TORCH_CHECK(
                dsPerOutputScalingFactors.dim() == 3, "batching N: dsPerOutputScalingFactors must be a 3D matrix");
            TORCH_CHECK(dsPerOutputScalingFactors.sizes()[0] == b,
                "batching N: dsPerOutputScalingFactors must have size B x M/128 x N");
            TORCH_CHECK(dsPerOutputScalingFactors.sizes()[1] == m / tileSize,
                "batching N: dsPerOutputScalingFactors must have size B x M/128 x N");
            TORCH_CHECK(
                dsPerOutputScalingFactors.sizes()[2] == (int64_t) tensorrt_llm::common::divUp(n, tileSize) * tileSize,
                "batching N: dsPerOutputScalingFactors must have size B x M/128 x N");
        }

        dDqSfsA = reinterpret_cast<float*>(dsPerInputAScalingFactors.data_ptr());
        dDqSfsB = reinterpret_cast<float*>(dsPerInputBScalingFactors.data_ptr());
        dDqSfsC = reinterpret_cast<float*>(dsPerOutputScalingFactors.data_ptr());
    }
    else
    {
        TORCH_CHECK(outScalingFactor.scalar_type() == at::ScalarType::Float, "Scale dtype must be FP32.");
        TORCH_CHECK(outScalingFactor.dim() == 1, "outScalingFactor must be a 1D matrix of size B");
        TORCH_CHECK(outScalingFactor.sizes()[0] == b, "outScalingFactor must be a 1D matrix of size B");

        ptrScaleC = reinterpret_cast<float*>(outScalingFactor.data_ptr());
    }

    // Create output tensor.
    at::Tensor out = at::detail::empty_cuda({b, outputM, outputN}, dtypeCTorch, inputBatchA.device(), std::nullopt);

    // Create runner.
    auto runner = tensorrt_llm::kernels::TrtllmGenBatchedGemmRunner{dtypeC, b, tileSize, useDeepSeekFp8, batchM};

    // Create sizes for the batch elements. No dynamic batching support yet.
    auto const bMn = batchM ? mPadded : nPadded;
    auto batchedMn = std::vector<int32_t>(b);
    std::fill(batchedMn.begin(), batchedMn.end(), bMn);

    auto stream = at::cuda::getCurrentCUDAStream(inputBatchA.get_device());

    runner.run(static_cast<int32_t>(mPadded), static_cast<int32_t>(nPadded), static_cast<int32_t>(k),
        inputBatchA.data_ptr(), inputBatchB.data_ptr(), out.data_ptr(), ptrScaleC, dDqSfsA, dDqSfsB, dDqSfsC,
        batchM ? batchedMn : std::vector<int32_t>(), batchM ? std::vector<int32_t>() : batchedMn, stream);

    // Unpad output
    out = batchM ? at::narrow(out, 1, 0, m) : at::narrow(out, 1, 0, n);

    return out;
}
} // namespace

namespace torch_ext
{

extern at::Tensor fp8_batched_gemm(at::Tensor& inputBatchA, int64_t m, at::Tensor const& dsPerInputAScalingFactors,
    at::Tensor& inputBatchB, int64_t n, at::Tensor const& dsPerInputBScalingFactors,
    at::Tensor const& dsPerOutputScalingFactors, at::Tensor const& outScalingFactor, int64_t tileSize,
    bool quantizeOutput, bool useDeepSeekFp8, bool batchM)
{
    auto const smVersion = tensorrt_llm::common::getSMVersion();
    switch (smVersion)
    {
    case tensorrt_llm::kernels::kSM_100:
    {
        return fp8_batched_gemm_sm100(inputBatchA, m, dsPerInputAScalingFactors, inputBatchB, n,
            dsPerInputBScalingFactors, dsPerOutputScalingFactors, outScalingFactor, tileSize, quantizeOutput,
            useDeepSeekFp8, batchM);
    }
    default: TLLM_THROW("Unsupported or unimplemented compute capability for fp8 batched gemm: %i", smVersion);
    }
}
} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fp8_batched_gemm(Tensor self, int m, Tensor dsPerInputAScalingFactors, Tensor inputBatchB, int n, Tensor "
        "dsPerInputBScalingFactors, Tensor dsPerOutputScalingFactors, Tensor outScalingFactor, "
        "int tileSize, bool quantizeOutput, bool useDeepSeekFp8, bool batchM) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fp8_batched_gemm", &torch_ext::fp8_batched_gemm);
}
