/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceResult.h"
#include "tensorrt_llm/executor/transferAgent.h"

#include <chrono>
#include <cstdint>
#include <future>
#include <string>
#include <utility>

namespace tensorrt_llm::executor::kv_cache::bounce
{

/// TransferStatus over the bounce transport's completion future (the value Python's wait() reads).
/// Header-only so its contract can be unit-tested over hand-built futures.
class BounceTransferStatus final : public TransferStatus
{
public:
    explicit BounceTransferStatus(std::shared_future<BounceResult> fut)
        : mFut(std::move(fut))
    {
    }

    /// The TransferStatus contract; see FailedRequest for when a failure gets there.
    [[nodiscard]] bool isCompleted() const override
    {
        // Readiness first: the sender records every outstanding access before it sets the result.
        return isReady() && mFut.get().quiesced();
    }

    [[nodiscard]] TransferState wait(int64_t timeoutMs) const override
    {
        if (!mFut.valid())
        {
            return TransferState::kFAILURE;
        }
        if (timeoutMs < 0)
        {
            return mFut.get().state;
        }
        if (mFut.wait_for(std::chrono::milliseconds(timeoutMs)) == std::future_status::ready)
        {
            return mFut.get().state;
        }
        return TransferState::kIN_PROGRESS;
    }

    /// Failure cause recorded by the bounce transport (empty while in flight / on success). An
    /// in-doubt failure says so, since only then may the destination still be written.
    [[nodiscard]] std::string getLastStatusStr() const override
    {
        if (!isReady())
        {
            return {};
        }
        auto const& result = mFut.get();
        if (result.reason == BounceFailReason::kNone)
        {
            return {};
        }
        std::string detail = toString(result.reason);
        if (!result.destinationQuiesced())
        {
            detail += "; destination not proven quiescent";
        }
        return detail;
    }

private:
    [[nodiscard]] bool isReady() const
    {
        return mFut.valid() && mFut.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }

    std::shared_future<BounceResult> mFut;
};

} // namespace tensorrt_llm::executor::kv_cache::bounce
