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

#include "tensorrt_llm/kernels/speculativeDecoding/explicitDraftTokensKernels.h"
#include "tensorrt_llm/thop/thUtils.h"

namespace th = torch;

namespace torch_ext
{
void convertSpecDecodingMaskToPackedMask(torch::Tensor specDecodingGenerationLengthsTensor,
    torch::Tensor specDecodingMaskTensor, int64_t maxSpecDecodingTokens, torch::Tensor specDecodingPackedMaskTensor,
    torch::optional<int64_t> stream_ptr = torch::nullopt)
{
    TLLM_CHECK_WITH_INFO(
        at::cuda::is_available(), "convert_spec_decoding_mask_to_packed_mask should be called with cuda enabled.");
    cudaStream_t stream;
    if (stream_ptr.has_value())
    {
        stream = reinterpret_cast<cudaStream_t>(stream_ptr.value());
    }
    else
    {
        stream = at::cuda::getCurrentCUDAStream();
    }
    TLLM_CHECK_WITH_INFO(specDecodingGenerationLengthsTensor.dim() == 1
            && specDecodingGenerationLengthsTensor.scalar_type() == torch::kInt,
        "spec_decoding_generation_lengths tensor should be 1D int tensor.");

    TLLM_CHECK_WITH_INFO(specDecodingMaskTensor.dim() == 3 && specDecodingMaskTensor.scalar_type() == torch::kBool,
        "spec_decoding_mask tensor should be 3D bool tensor.");

    TLLM_CHECK_WITH_INFO(
        specDecodingPackedMaskTensor.dim() == 2 && specDecodingPackedMaskTensor.scalar_type() == torch::kInt,
        "spec_decoding_packed_mask tensor should be 2D int tensor.");

    int batchSize = specDecodingGenerationLengthsTensor.size(0);

    int64_t scanReduceTempMemoryBytes = tensorrt_llm::kernels::speculative_decoding::invokeScanReduceGenerationLengths(
        batchSize, nullptr, nullptr, 0, nullptr, nullptr, stream);

    torch::Tensor scanReduceTempMemoryStorage = torch::empty(
        {
            scanReduceTempMemoryBytes,
        },
        torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));
    torch::Tensor scanedSpecDecodingGenerationLengths = torch::empty(
        {
            batchSize,
        },
        torch::dtype(torch::kInt).device(torch::kCUDA).requires_grad(false));
    torch::Tensor maxSpecDecodingGenerationLengths = torch::empty(
        {
            1,
        },
        torch::dtype(torch::kInt).device(torch::kCUDA).requires_grad(false));

    tensorrt_llm::kernels::speculative_decoding::invokeScanReduceGenerationLengths(batchSize,
        specDecodingGenerationLengthsTensor.data_ptr<int>(),
        reinterpret_cast<void*>(scanReduceTempMemoryStorage.data_ptr<int8_t>()), scanReduceTempMemoryBytes,
        scanedSpecDecodingGenerationLengths.data_ptr<int>(), maxSpecDecodingGenerationLengths.data_ptr<int>(), stream);

    tensorrt_llm::kernels::speculative_decoding::invokeConvertMaskToPackedMask(batchSize,
        scanedSpecDecodingGenerationLengths.data_ptr<int>(), maxSpecDecodingGenerationLengths.data_ptr<int>(),
        specDecodingMaskTensor.data_ptr<bool>(), nullptr, maxSpecDecodingTokens, maxSpecDecodingTokens + 1,
        specDecodingPackedMaskTensor.data_ptr<int>(), stream);
}

} // namespace torch_ext

static auto convert_spec_decoding_mask_to_packed_mask = torch::RegisterOperators(
    "tensorrt_llm::convert_spec_decoding_mask_to_packed_mask", &torch_ext::convertSpecDecodingMaskToPackedMask);
