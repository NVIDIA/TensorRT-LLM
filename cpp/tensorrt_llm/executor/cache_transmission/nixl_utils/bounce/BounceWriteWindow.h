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

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace tensorrt_llm::executor::kv_cache::bounce
{

// A receiver publishes one window per writer and receive slice. Ranges use
// receiver-local byte addresses and represent half-open intervals.
class BounceWriteWindow
{
public:
    using Range = std::pair<std::uint64_t, std::uint64_t>; // (address, length)

    BounceWriteWindow(std::string peer, std::string endpoint, std::string key, std::vector<Range> ranges);
    [[nodiscard]] bool contains(std::uint64_t address, std::uint64_t length) const;
    [[nodiscard]] bool intersects(std::uint64_t address, std::uint64_t length) const;

    [[nodiscard]] std::string const& peer() const noexcept
    {
        return mPeer;
    }

    [[nodiscard]] std::string const& endpoint() const noexcept
    {
        return mEndpoint;
    }

    [[nodiscard]] std::string const& id() const noexcept
    {
        return mId;
    }

    [[nodiscard]] std::string const& key() const noexcept
    {
        return mKey;
    }

    [[nodiscard]] std::shared_lock<std::shared_mutex> acquire() const;
    void close();

private:
    std::string mPeer;
    std::string mEndpoint;
    std::string mKey;
    std::string mId;
    std::vector<Range> mRanges;
    mutable std::shared_mutex mMutex;
    bool mOpen{true};
};

//! Receiver-local geometry of one pool view for one writer. The offsets are relative to slot zero.
struct BounceSlotLayout
{
    std::uint64_t base;
    std::uint64_t stride;
    std::uint64_t slotBytes;
    std::uint64_t numSlots;
    std::span<std::int64_t const> offsets;
    std::span<std::int64_t const> lengths;
};

class BounceWriteWindows
{
public:
    std::shared_ptr<BounceWriteWindow> add(
        std::string peer, std::string endpoint, std::string key, std::vector<BounceWriteWindow::Range> ranges);
    std::shared_ptr<BounceWriteWindow> addSlots(std::string peer, std::string endpoint, std::string key,
        std::span<std::int64_t const> slots, BounceSlotLayout const& layout);
    [[nodiscard]] std::shared_ptr<BounceWriteWindow> find(std::string const& id, std::string const& peer) const;
    //! Stop destination writes while the owner waits for the writer's cancellation acknowledgement.
    void seal(std::string const& id);
    void remove(std::string const& id);
    void removeIntersecting(std::uint64_t address, std::uint64_t length);

private:
    mutable std::mutex mMutex;
    std::map<std::string, std::shared_ptr<BounceWriteWindow>> mWindows;
};

// Domain-separated from KV result reporting. The key is never transmitted in
// the bounce channel. Both the operation and its complete payload are signed.
std::string bounceWindowId(std::string const& key);
std::string signBounceFrame(std::string frame, std::string const& key);
std::string signBounceFrame(std::string frame, std::string const& key, std::string const& id);
bool readBounceWindowId(std::string const& envelope, std::size_t payloadEnd, std::string& id);
bool verifyBounceFrame(std::string const& envelope, std::string const& key);

} // namespace tensorrt_llm::executor::kv_cache::bounce
