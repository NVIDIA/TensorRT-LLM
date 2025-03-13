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

#include <iostream>

#include "tensorrt_llm/executor/types.h"

namespace tensorrt_llm::executor
{
std::ostream& operator<<(std::ostream& os, CapacitySchedulerPolicy policy)
{
    switch (policy)
    {
    case CapacitySchedulerPolicy::kMAX_UTILIZATION: os << "MAX_UTILIZATION"; break;
    case CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT: os << "GUARANTEED_NO_EVICT"; break;
    case CapacitySchedulerPolicy::kSTATIC_BATCH: os << "STATIC_BATCH"; break;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, ContextChunkingPolicy policy)
{
    switch (policy)
    {
    case ContextChunkingPolicy::kEQUAL_PROGRESS: os << "EQUAL_PROGRESS"; break;
    case ContextChunkingPolicy::kFIRST_COME_FIRST_SERVED: os << "FIRST_COME_FIRST_SERVED"; break;
    }
    return os;
}
} // namespace tensorrt_llm::executor
