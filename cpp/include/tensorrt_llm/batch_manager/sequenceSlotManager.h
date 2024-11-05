/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <chrono>
#include <cstdint>
#include <optional>
#include <queue>
#include <unordered_map>
#include <vector>

namespace tensorrt_llm::batch_manager
{

/// SequenceSlotManager
///
/// Helper class to manage sequence slots
/// This class is not thread-safe

class SequenceSlotManager
{
public:
    using SlotIdType = int32_t;
    using SequenceIdType = std::uint64_t;

    SequenceSlotManager(SlotIdType maxNumSlots, uint64_t maxSequenceIdleMicroseconds);

    /// Function that returns a slot for the provided sequenceId
    /// For a new sequence id, a new slot will be allocated
    /// In case no slot could be allocated or matched, optional will be empty
    std::optional<SlotIdType> getSequenceSlot(bool const& startFlag, SequenceIdType const& sequenceId);

    /// Function that frees the slot associated with the given sequence id
    void freeSequenceSlot(SequenceIdType sequenceId);

    /// Function that frees slots that have been idle for more than
    /// mMaxSequenceIdleMicroseconds
    void freeIdleSequenceSlots();

private:
    SlotIdType mMaxNumSlots;
    std::chrono::microseconds mMaxSequenceIdleMicroseconds;

    std::unordered_map<SequenceIdType, SlotIdType> mSequenceIdToSlot;
    std::queue<SlotIdType> mAvailableSlots;
    std::vector<std::chrono::steady_clock::time_point> mLastTimepoint;
};

} // namespace tensorrt_llm::batch_manager
