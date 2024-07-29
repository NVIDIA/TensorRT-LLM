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

#include "tensorrt_llm/thop/thUtils.h"
#include <NvInferRuntime.h>
#include <array>

namespace torch_ext
{

tensorrt_llm::runtime::ITensor::Shape convert_shape(torch::Tensor tensor)
{
    constexpr auto trtMaxDims = nvinfer1::Dims::MAX_DIMS;
    auto const torchTensorNumDims = tensor.dim();
    TLLM_CHECK_WITH_INFO(torchTensorNumDims <= trtMaxDims,
        "TensorRT supports at most %i tensor dimensions. Found a Torch tensor with %li dimensions.", trtMaxDims,
        torchTensorNumDims);
    auto result = nvinfer1::Dims{};
    result.nbDims = static_cast<int32_t>(torchTensorNumDims);
    for (int i = 0; i < torchTensorNumDims; i++)
    {
        result.d[i] = static_cast<int64_t>(tensor.size(i));
    }
    return result;
}

template <typename T>
tensorrt_llm::runtime::ITensor::UniquePtr convert_tensor(torch::Tensor tensor)
{
    return tensorrt_llm::runtime::ITensor::wrap(
        get_ptr<T>(tensor), tensorrt_llm::runtime::TRTDataType<T>::value, convert_shape(tensor));
}

// Template instantiations
template tensorrt_llm::runtime::ITensor::UniquePtr convert_tensor<int32_t*>(torch::Tensor tensor);
template tensorrt_llm::runtime::ITensor::UniquePtr convert_tensor<int32_t>(torch::Tensor tensor);
template tensorrt_llm::runtime::ITensor::UniquePtr convert_tensor<uint8_t>(torch::Tensor tensor);
template tensorrt_llm::runtime::ITensor::UniquePtr convert_tensor<int8_t>(torch::Tensor tensor);
template tensorrt_llm::runtime::ITensor::UniquePtr convert_tensor<float>(torch::Tensor tensor);
template tensorrt_llm::runtime::ITensor::UniquePtr convert_tensor<half>(torch::Tensor tensor);
#ifdef ENABLE_BF16
template tensorrt_llm::runtime::ITensor::UniquePtr convert_tensor<__nv_bfloat16>(torch::Tensor tensor);
#endif
template tensorrt_llm::runtime::ITensor::UniquePtr convert_tensor<bool>(torch::Tensor tensor);

} // namespace torch_ext
