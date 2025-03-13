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
#include <deque>

namespace tensorrt_llm::executor
{

/// @brief Inserts a request into a request list sorted by priority / arrival time
void insertRequestInOrder(tensorrt_llm::batch_manager::RequestList& reqList,
    std::shared_ptr<tensorrt_llm::batch_manager::LlmRequest> const& req);

/// @brief Inserts a requestWithId into a request deque sorted by priority / arrival time
void insertRequestInOrder(std::deque<RequestWithId>& reqWithIdDeque, RequestWithId&& reqWithId);

} // namespace tensorrt_llm::executor
