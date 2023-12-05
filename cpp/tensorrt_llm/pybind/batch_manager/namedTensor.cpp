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
#include "tensorrt_llm/runtime/torch.h"

namespace tb = tensorrt_llm::batch_manager;

namespace tensorrt_llm::pybind::batch_manager
{

NamedTensor::NamedTensor(const tb::NamedTensor& cppNamedTensor)
    : Base(runtime::Torch::tensor(cppNamedTensor.tensor), cppNamedTensor.name)
{
}

} // namespace tensorrt_llm::pybind::batch_manager
