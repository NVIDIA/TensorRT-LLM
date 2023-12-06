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
#include "inferenceRequest.h"

#include "tensorrt_llm/batch_manager/inferenceRequest.h"
#include "tensorrt_llm/runtime/torchView.h"
#include <memory>

namespace tb = tensorrt_llm::batch_manager;
namespace tr = tensorrt_llm::runtime;

using namespace tensorrt_llm::pybind::batch_manager;

namespace
{

void copy_tensor(NamedTensor const& src, tb::NamedTensor& dst)
{
    TLLM_CHECK_WITH_INFO(src.name == dst.name, "names do not match: %s != %s", src.name.c_str(), dst.name.c_str());
    if (src.tensor.has_value())
    {
        dst.tensor = tr::TorchView::of(src.tensor.value());
    }
}
} // namespace

std::shared_ptr<tb::InferenceRequest> InferenceRequest::toTrtLlm() const
{
    tb::InferenceRequest::TensorMap tensorMap;
    for (auto const& [name, tensor] : mInputTensors)
    {
        if (tensor.has_value())
        {
            tensorMap[name] = tr::TorchView::of(tensor.value());
        }
    }
    return std::make_shared<tb::InferenceRequest>(tensorMap, mRequestId);
}
