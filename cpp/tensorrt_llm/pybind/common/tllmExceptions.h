/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/tllmException.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace tc = tensorrt_llm::common;

namespace tensorrt_llm::pybind::common
{

/// @brief Bind RequestSpecificException and related types to Python
/// @param m The pybind11 module to bind to
void initExceptionsBindings(py::module_& m);

} // namespace tensorrt_llm::pybind::common
