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
#include "inferenceRequest.h"
#include "namedTensor.h"
#include "tensorrt_llm/batch_manager/GptManager.h"
#include "tensorrt_llm/batch_manager/callbacks.h"
#include <pybind11/functional.h>

#include <ATen/ops/tensor.h>
#include <functional>

namespace tensorrt_llm::pybind::batch_manager
{

using GetInferenceRequestsCallback = std::function<std::list<InferenceRequest>(int32_t)>;
using SendResponseCallback = std::function<void(uint64_t, std::list<NamedTensor> const&, bool, const std::string&)>;

tensorrt_llm::batch_manager::GetInferenceRequestsCallback callbackAdapter(GetInferenceRequestsCallback callback);
tensorrt_llm::batch_manager::SendResponseCallback callbackAdapter(SendResponseCallback callback);

class GptManager : tensorrt_llm::batch_manager::GptManager
{
public:
    GptManager(std::filesystem::path const& trtEnginePath, tensorrt_llm::batch_manager::TrtGptModelType modelType,
        int32_t maxBeamWidth, tensorrt_llm::batch_manager::batch_scheduler::SchedulerPolicy schedulerPolicy,
        GetInferenceRequestsCallback getInferenceRequestsCb, SendResponseCallback sendResponseCb,
        tensorrt_llm::batch_manager::PollStopSignalCallback pollStopSignalCb = nullptr,
        tensorrt_llm::batch_manager::ReturnBatchManagerStatsCallback returnBatchManagerStatsCb = nullptr,
        const tensorrt_llm::batch_manager::TrtGptModelOptionalParams& optionalParams
        = tensorrt_llm::batch_manager::TrtGptModelOptionalParams(),
        std::optional<uint64_t> terminateReqId = std::nullopt);

    pybind11::object enter();
    void exit(pybind11::handle type, pybind11::handle value, pybind11::handle traceback);

    static void initBindings(pybind11::module_& m);
};

} // namespace tensorrt_llm::pybind::batch_manager
