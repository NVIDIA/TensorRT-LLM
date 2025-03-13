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

#include "tensorrt_llm/executor/requestUtils.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/executor/requestWithId.h"
#include <algorithm>

using tensorrt_llm::executor::RequestWithId;
using tensorrt_llm::batch_manager::RequestList;
using tensorrt_llm::batch_manager::LlmRequest;

void tensorrt_llm::executor::insertRequestInOrder(RequestList& reqList, std::shared_ptr<LlmRequest> const& req)
{
    auto const it = std::upper_bound(std::begin(reqList), std::end(reqList), req,
        [](std::shared_ptr<LlmRequest> const& a, std::shared_ptr<LlmRequest> const& b)
        { return a->priority() > b->priority(); });
    reqList.insert(it, req);
}

void tensorrt_llm::executor::insertRequestInOrder(std::deque<RequestWithId>& reqWithIdDeque, RequestWithId&& reqWithId)
{
    auto const it = std::upper_bound(std::begin(reqWithIdDeque), std::end(reqWithIdDeque), reqWithId,
        [](RequestWithId const& a, RequestWithId const& b) { return a.req.getPriority() > b.req.getPriority(); });
    reqWithIdDeque.insert(it, std::move(reqWithId));
}
