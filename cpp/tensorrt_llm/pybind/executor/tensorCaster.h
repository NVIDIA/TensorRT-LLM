/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <pybind11/pybind11.h>

#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/runtime/torch.h"
#include "tensorrt_llm/runtime/torchView.h"
#include <torch/extension.h>

namespace PYBIND11_NAMESPACE
{

namespace detail
{
template <>
struct type_caster<tensorrt_llm::executor::Tensor>
{
public:
    PYBIND11_TYPE_CASTER(tensorrt_llm::executor::Tensor, _("torch.Tensor"));

    // Convert PyObject(torch.Tensor) -> tensorrt_llm::executor::Tensor
    bool load(handle src, bool)
    {
        PyObject* obj = src.ptr();
        if (THPVariable_Check(obj))
        {
            at::Tensor const& t = THPVariable_Unpack(obj);
            value = tensorrt_llm::executor::detail::ofITensor(tensorrt_llm::runtime::TorchView::of(t));
            return true;
        }
        return false;
    }

    // Convert tensorrt_llm::executor::Tensor -> PyObject(torch.Tensor)
    static handle cast(tensorrt_llm::executor::Tensor const& src, return_value_policy /* policy */, handle /* parent */)
    {
        return THPVariable_Wrap(tensorrt_llm::runtime::Torch::tensor(tensorrt_llm::executor::detail::toITensor(src)));
    }
};

} // namespace detail
} // namespace PYBIND11_NAMESPACE
