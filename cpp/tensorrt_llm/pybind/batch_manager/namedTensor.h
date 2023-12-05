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

#pragma once

#include "tensorrt_llm/batch_manager/namedTensor.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/runtime/iBuffer.h"

#include <ATen/ATen.h>

#include <ATen/core/ATen_fwd.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/tensor.h>
#include <c10/core/DeviceType.h>
#include <c10/util/ArrayRef.h>
#include <memory>
#include <optional>

namespace tb = tensorrt_llm::batch_manager;

namespace tensorrt_llm::pybind::batch_manager
{

class NamedTensor : public tb::GenericNamedTensor<std::optional<at::Tensor>>
{
public:
    using Base = tb::GenericNamedTensor<std::optional<at::Tensor>>;
    using TensorPtr = Base::TensorPtr;

    NamedTensor(TensorPtr _tensor, std::string _name)
        : Base(std::move(_tensor), std::move(_name)){};

    explicit NamedTensor(std::string _name)
        : Base(std::move(_name)){};

    explicit NamedTensor(const tb::NamedTensor& cppNamedTensor);
};

} // namespace tensorrt_llm::pybind::batch_manager
