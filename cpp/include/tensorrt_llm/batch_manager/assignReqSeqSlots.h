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

#include "common.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/sequenceSlotManager.h"
#include "tensorrt_llm/common/algorithm.h"
#include "tensorrt_llm/runtime/common.h"

namespace tensorrt_llm::batch_manager
{

namespace tle = tensorrt_llm::executor;

class AssignReqSeqSlots : Algorithm
{
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

public:
    constexpr static auto name{"AssignReqSeqSlots"};

    AssignReqSeqSlots() = default;

    void operator()(SequenceSlotManager& seqSlotManager, RequestVector const& contextRequests,
        RequestVector const& generationRequests) const;
};

} // namespace tensorrt_llm::batch_manager
