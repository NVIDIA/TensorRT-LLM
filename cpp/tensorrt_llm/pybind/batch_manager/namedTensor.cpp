/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "namedTensor.h"

#include "tensorrt_llm/batch_manager/inferenceRequest.h"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/torchView.h"
#include <memory>

namespace tb = tensorrt_llm::batch_manager;

namespace tensorrt_llm::pybind::batch_manager
{

NamedTensor::NamedTensor(const tb::NamedTensor& cppNamedTensor)
    : Base(cppNamedTensor.name)
{
    auto cppTensor = cppNamedTensor.tensor;
    std::vector<at::IntArrayRef::value_type> shapeValues;
    for (int i = 0; i < cppTensor->getShape().nbDims; ++i)
    {
        shapeValues.push_back(cppTensor->getShape().d[i]);
    }

    tensor = at::from_blob(cppTensor->data(), shapeValues,
        at::TensorOptions()
            .device(runtime::TorchUtils::deviceType(cppTensor->getMemoryType()))
            .pinned_memory(cppTensor->getMemoryType() == runtime::MemoryType::kPINNED)
            .dtype(runtime::TorchUtils::dataType(cppTensor->getDataType())));
}

} // namespace tensorrt_llm::pybind::batch_manager
