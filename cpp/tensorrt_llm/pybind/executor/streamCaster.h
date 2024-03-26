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

#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/cudaStream.h"

namespace PYBIND11_NAMESPACE
{

namespace detail
{
template <>
struct type_caster<tensorrt_llm::executor::StreamPtr>
{
public:
    PYBIND11_TYPE_CASTER(tensorrt_llm::executor::StreamPtr, _("int"));

    bool load([[maybe_unused]] handle src, bool)
    {
        // We don't need to convert in this direction.
        return false;
    }

    static handle cast(
        tensorrt_llm::executor::StreamPtr const& src, return_value_policy /* policy */, handle /* parent */)
    {
        // Return cudaStream_t as integer.
        return PyLong_FromVoidPtr(src->get());
    }
};

} // namespace detail
} // namespace PYBIND11_NAMESPACE
