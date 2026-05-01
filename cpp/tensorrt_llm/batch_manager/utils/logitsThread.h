/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <memory>
#include <mutex>

namespace tensorrt_llm::batch_manager
{
class GenerateRequestOptions;
} // namespace tensorrt_llm::batch_manager

namespace tensorrt_llm::executor
{
struct SpeculativeDecodingFastLogitsInfo;
} // namespace tensorrt_llm::executor

namespace tensorrt_llm::batch_manager::utils
{

/// Background thread that sends draft model logits to the target model via MPI.
/// Both \p draftRequestsWaitingToSendLogits (consumed here) and \p draftRequestsDoneSendingLogits
/// (produced here for the main thread to drain) are guarded by \p draftRequestsMtx.
void draftModelSendLogitsThread(int device, std::atomic<bool>* draftModelThreadShouldExit,
    RequestVector* draftRequestsWaitingToSendLogits, RequestVector* draftRequestsDoneSendingLogits,
    std::mutex* draftRequestsMtx);

void targetModelReceiveLogits(runtime::ITensor::SharedPtr& draftLogitsHost,
    executor::SpeculativeDecodingFastLogitsInfo const& fastLogitsInfo, nvinfer1::DataType logitsDtype);

} // namespace tensorrt_llm::batch_manager::utils
