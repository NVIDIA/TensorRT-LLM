/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/iTensor.h"

namespace tensorrt_llm::batch_manager
{
struct NamedTensor
{
    using TensorPtr = tensorrt_llm::runtime::ITensor::SharedPtr;

    TensorPtr tensor;
    std::string name;

    NamedTensor() = default;
    ~NamedTensor() = default;

    // Host Tensor constructor
    NamedTensor(
        nvinfer1::DataType _type, std::vector<int64_t> const& _shape, std::string _name, const void* _data = nullptr);

    NamedTensor(TensorPtr _tensor, std::string _name)
        : tensor(std::move(_tensor))
        , name(std::move(_name))
    {
    }

    std::vector<int64_t> serialize();
    static NamedTensor deserialize(const int64_t* packed);
};
} // namespace tensorrt_llm::batch_manager
