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
#include "inferenceRequest.h"
#include "namedTensor.h"
#include "tensorrt_llm/batch_manager/GptManager.h"
#include "tensorrt_llm/batch_manager/callbacks.h"
#include <pybind11/functional.h>

#include <ATen/ops/tensor.h>
#include <functional>

namespace py = pybind11;
namespace tb = tensorrt_llm::batch_manager;

namespace tensorrt_llm::pybind::batch_manager
{

using GetInferenceRequestsCallback = std::function<std::list<InferenceRequest>(int32_t)>;
using SendResponseCallback = std::function<void(uint64_t, std::list<NamedTensor> const&, bool, const std::string&)>;

tb::GetInferenceRequestsCallback callbackAdapter(GetInferenceRequestsCallback callback);
tb::SendResponseCallback callbackAdapter(SendResponseCallback callback);

class GptManager : tb::GptManager
{
public:
    GptManager(std::filesystem::path const& trtEnginePath, tb::TrtGptModelType modelType, int32_t maxBeamWidth,
        tb::batch_scheduler::SchedulerPolicy schedulerPolicy, GetInferenceRequestsCallback getInferenceRequestsCb,
        SendResponseCallback sendResponseCb, tb::PollStopSignalCallback pollStopSignalCb = nullptr,
        tb::ReturnBatchManagerStatsCallback returnBatchManagerStatsCb = nullptr,
        const tb::TrtGptModelOptionalParams& optionalParams = tb::TrtGptModelOptionalParams(),
        std::optional<uint64_t> terminateReqId = std::nullopt);

    py::object enter();
    void exit(py::handle type, py::handle value, py::handle traceback);
};

} // namespace tensorrt_llm::pybind::batch_manager
