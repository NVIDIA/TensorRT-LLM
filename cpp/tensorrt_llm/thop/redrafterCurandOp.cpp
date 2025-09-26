/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/speculativeDecoding/explicitDraftTokensKernels.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/thop/thUtils.h"

#if ENABLE_BF16
#include <cuda_bf16.h>
#endif // ENABLE_BF16

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>

#include <cstdint>

namespace th = torch;
namespace tr = tensorrt_llm::runtime;
namespace tk = tensorrt_llm::kernels;
namespace tksd = tensorrt_llm::kernels::speculative_decoding;

namespace torch_ext
{

namespace
{
// Must be similar to ExplicitDraftTokensLayer<T>::setup
void initializeDeviceCurandStates(
    int64_t batchSize, th::Tensor& curandState, th::optional<th::Tensor>& randomSeeds, cudaStream_t stream)
{
    auto* curandStatePtr = get_ptr<curandState_t>(curandState);
    tr::SizeType32* batchSlotsPtr = nullptr;

    if (randomSeeds.has_value())
    {
        if (batchSize > 1 && randomSeeds->size(0) == 1)
        {
            TLLM_CHECK_WITH_INFO(randomSeeds->device().is_cpu(), "Random seed tensor expected on host.");
            auto const randomSeed = get_val<uint64_t>(randomSeeds.value(), 0);
            tk::invokeCurandInitialize(curandStatePtr, batchSlotsPtr, batchSize, randomSeed, stream);
        }
        else
        {
            TLLM_CHECK_WITH_INFO(
                randomSeeds->dim() == 1 && randomSeeds->size(0) == batchSize, "Random seed tensor size mismatch.");
            TLLM_CHECK_WITH_INFO(randomSeeds->device().is_cuda(), "Random seed tensor expected on device.");

            auto* randomSeedsPtr = get_ptr<uint64_t>(randomSeeds.value());
            tk::invokeCurandBatchInitialize(curandStatePtr, batchSlotsPtr, batchSize, randomSeedsPtr, stream);
        }
    }
    else
    {
        // Initialize curand states using the default seed 0.
        tk::invokeCurandInitialize(
            curandStatePtr, batchSlotsPtr, batchSize, tensorrt_llm::layers::DefaultDecodingParams::getSeed(), stream);
    }
    sync_check_cuda_error(stream);
}
} // namespace

void prepareRandomTensors(th::Tensor& curandState, // [maxBatchSize, 48], uint8_t
    th::Tensor& randDataSample,                    // [maxBatchSize], dtype (float or half)
    th::Tensor& randDataValidation,       // [maxBatchSize, maxNumPaths, maxPathDraftLength], dtype (float or half)
    th::optional<th::Tensor> randomSeeds, // [1] or [maxBatchSize], uint64_t
    int64_t const batchSize,              //
    int64_t const numPaths,               //
    int64_t const draftLength,            //
    bool const initialize                 //
)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto const scalarType = randDataSample.scalar_type();
    CHECK_TYPE(randDataValidation, scalarType);

    TLLM_CHECK_WITH_INFO(
        randDataSample.dim() == 1 && randDataSample.size(0) == batchSize, "Random sample tensor size mismatch.");
    TLLM_CHECK_WITH_INFO(randDataValidation.dim() == 3 && randDataValidation.size(0) == batchSize
            && randDataValidation.size(1) == numPaths && randDataValidation.size(2) == draftLength,
        "Random validation tensor size mismatch.");

    TLLM_CHECK_WITH_INFO(
        curandState.dim() == 2 && curandState.size(0) == batchSize && curandState.size(1) == sizeof(curandState_t),
        "Curand state tensor shpe mismatch."
        "(got (%lu, %lu), need (%lu, %lu)).",
        curandState.size(0), curandState.size(1), batchSize, sizeof(curandState_t));

    if (initialize)
    {
        initializeDeviceCurandStates(batchSize, curandState, randomSeeds, stream);
    }

    switch (scalarType)
    {
    case at::ScalarType::Float:
    {
        tksd::FillRandDataExplicitDraftTokensParams<float> params;
        params.batchSize = static_cast<tr::SizeType32>(batchSize);
        params.numPaths = static_cast<tr::SizeType32>(numPaths);
        params.draftLength = static_cast<tr::SizeType32>(draftLength);
        params.randDataSample = get_ptr<float>(randDataSample);
        params.randDataVerification = get_ptr<float>(randDataValidation);
        params.curandState = get_ptr<curandState_t>(curandState);
        params.batchSlots = nullptr;
        params.skipVerification = initialize;

        tksd::invokeFillRandData(params, stream);
    }
    break;
    case at::ScalarType::Half:
    {
        tksd::FillRandDataExplicitDraftTokensParams<half> params;
        params.batchSize = static_cast<tr::SizeType32>(batchSize);
        params.numPaths = static_cast<tr::SizeType32>(numPaths);
        params.draftLength = static_cast<tr::SizeType32>(draftLength);
        params.randDataSample = get_ptr<half>(randDataSample);
        params.randDataVerification = get_ptr<half>(randDataValidation);
        params.curandState = get_ptr<curandState_t>(curandState);
        params.batchSlots = nullptr;
        params.skipVerification = initialize;

        tksd::invokeFillRandData(params, stream);
    }
    break;
#ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        tksd::FillRandDataExplicitDraftTokensParams<__nv_bfloat16> params;
        params.batchSize = static_cast<tr::SizeType32>(batchSize);
        params.numPaths = static_cast<tr::SizeType32>(numPaths);
        params.draftLength = static_cast<tr::SizeType32>(draftLength);
        params.randDataSample = get_ptr<__nv_bfloat16>(randDataSample);
        params.randDataVerification = get_ptr<__nv_bfloat16>(randDataValidation);
        params.curandState = get_ptr<curandState_t>(curandState);
        params.batchSlots = nullptr;
        params.skipVerification = initialize;

        tksd::invokeFillRandData(params, stream);
    }
    break;
#endif // ENABLE_BF16
    default: throw std::runtime_error("Unsupported tensor type.");
    }
    sync_check_cuda_error(stream);
}

} // namespace torch_ext

static auto redrafter_prepare_random_tensors
    = torch::RegisterOperators("tensorrt_llm::redrafter_prepare_random_tensors", &torch_ext::prepareRandomTensors);
