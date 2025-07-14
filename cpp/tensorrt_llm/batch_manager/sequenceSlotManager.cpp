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

#include "tensorrt_llm/batch_manager/sequenceSlotManager.h"
#include "tensorrt_llm/common/logger.h"

namespace tensorrt_llm::batch_manager
{

SequenceSlotManager::SequenceSlotManager(SlotIdType maxNumSlots, uint64_t maxSequenceIdleMicroseconds)
    : mMaxNumSlots(maxNumSlots)
    , mMaxSequenceIdleMicroseconds{std::chrono::microseconds(maxSequenceIdleMicroseconds)}
{
    mSequenceIdToSlot.reserve(maxNumSlots);
    for (SlotIdType slot = 0; slot < mMaxNumSlots; ++slot)
    {
        mAvailableSlots.emplace(slot);
    }
    mLastTimepoint.resize(mMaxNumSlots);
}

std::optional<SequenceSlotManager::SlotIdType> SequenceSlotManager::getSequenceSlot(
    bool const& startFlag, SequenceIdType const& sequenceId)
{
    std::optional<SlotIdType> slot;
    if (startFlag)
    {
        // Check if correlation_id already exists
        if (mSequenceIdToSlot.find(sequenceId) != mSequenceIdToSlot.end())
        {
            TLLM_LOG_ERROR("Already specified start flag for sequence id: %lu", sequenceId);
        }

        if (!mAvailableSlots.empty())
        {
            slot = mAvailableSlots.front();
            mAvailableSlots.pop();
            mSequenceIdToSlot.emplace(sequenceId, slot.value());
        }
        else
        {
            TLLM_LOG_ERROR("All available sequence slots are used");
        }
    }
    else
    {
        auto const it = mSequenceIdToSlot.find(sequenceId);
        if (it == mSequenceIdToSlot.end())
        {
            TLLM_LOG_ERROR("Could not find sequence id %lu in allocated sequence slots", sequenceId);
        }
        else
        {
            slot = it->second;
        }
    }
    if (slot)
    {
        mLastTimepoint[slot.value()] = std::chrono::steady_clock::now();
    }
    return slot;
}

void SequenceSlotManager::freeSequenceSlot(SequenceIdType sequenceId)
{
    auto const it = mSequenceIdToSlot.find(sequenceId);
    if (it != mSequenceIdToSlot.end())
    {
        auto const slot = it->second;
        mSequenceIdToSlot.erase(it);
        mAvailableSlots.push(slot);
    }
}

void SequenceSlotManager::freeIdleSequenceSlots()
{
    auto const now = std::chrono::steady_clock::now();
    for (auto it = mSequenceIdToSlot.begin(); it != mSequenceIdToSlot.end();)
    {
        auto const& [sequenceId, slot] = *it;
        auto const idleMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(now - mLastTimepoint[slot]);
        if (idleMicroseconds > mMaxSequenceIdleMicroseconds)
        {
            TLLM_LOG_INFO("Releasing idle sequence with correlation id %lu idle time %li us", sequenceId,
                idleMicroseconds.count());
            it = mSequenceIdToSlot.erase(it);
            mAvailableSlots.push(slot);
        }
        else
        {
            ++it;
        }
    }
}

} // namespace tensorrt_llm::batch_manager
