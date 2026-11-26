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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceWriteWindow.h"

#include <openssl/crypto.h>
#include <openssl/hmac.h>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace tensorrt_llm::executor::kv_cache::bounce
{
namespace
{
constexpr std::size_t kTagBytes = 16;
constexpr std::size_t kKeyBytes = 16;

std::string signature(std::string const& key, std::string_view bytes)
{
    if (key.size() != kKeyBytes)
    {
        throw std::invalid_argument("Invalid bounce window key size");
    }
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int length{};
    if (!HMAC(EVP_sha256(), key.data(), static_cast<int>(key.size()),
            reinterpret_cast<unsigned char const*>(bytes.data()), bytes.size(), digest.data(), &length)
        || length < kTagBytes)
    {
        throw std::runtime_error("Cannot sign bounce control frame");
    }
    return {reinterpret_cast<char const*>(digest.data()), kTagBytes};
}
} // namespace

std::string bounceWindowId(std::string const& key)
{
    return signature(key, "TRTLLM bounce window id v1");
}

std::string signBounceFrame(std::string const& frame, std::string const& key)
{
    auto envelope = frame + bounceWindowId(key);
    return envelope + signature(key, envelope);
}

bool readBounceWindowId(std::string const& envelope, std::size_t payloadEnd, std::string& id)
{
    if (envelope.size() < payloadEnd || envelope.size() - payloadEnd != 2 * kTagBytes)
    {
        return false;
    }
    id.assign(envelope, payloadEnd, kTagBytes);
    return true;
}

bool verifyBounceFrame(std::string const& envelope, std::string const& key)
{
    if (envelope.size() < 2 * kTagBytes)
    {
        return false;
    }
    auto const end = envelope.size() - kTagBytes;
    auto const tag = signature(key, std::string_view(envelope.data(), end));
    return CRYPTO_memcmp(envelope.data() + end, tag.data(), kTagBytes) == 0;
}

BounceWriteWindow::BounceWriteWindow(std::string peer, std::string endpoint, std::string key, std::vector<Range> ranges)
    : mPeer(std::move(peer))
    , mEndpoint(std::move(endpoint))
    , mKey(std::move(key))
{
    std::sort(ranges.begin(), ranges.end());
    for (auto const& [base, len] : ranges)
    {
        if (!len || base > std::numeric_limits<std::uint64_t>::max() - len)
        {
            throw std::invalid_argument("Invalid bounce receive window interval");
        }
        auto const end = base + len;
        if (!mRanges.empty() && base <= mRanges.back().second)
        {
            mRanges.back().second = std::max(mRanges.back().second, end);
        }
        else
        {
            mRanges.emplace_back(base, end);
        }
    }
}

bool BounceWriteWindow::contains(std::uint64_t address, std::uint64_t length) const
{
    auto it = std::upper_bound(mRanges.begin(), mRanges.end(), address,
        [](std::uint64_t value, Range const& range) { return value < range.first; });
    if (it == mRanges.begin())
    {
        return false;
    }
    auto const end = std::prev(it)->second;
    return length && address < end && length <= end - address;
}

bool BounceWriteWindow::intersects(std::uint64_t address, std::uint64_t length) const
{
    auto const it = std::upper_bound(mRanges.begin(), mRanges.end(), address,
        [](std::uint64_t start, Range const& interval) { return start < interval.second; });
    return length && it != mRanges.end() && (it->first <= address || it->first - address < length);
}

std::shared_lock<std::shared_mutex> BounceWriteWindow::acquire() const
{
    std::shared_lock lock(mMutex);
    if (!mOpen)
    {
        lock.unlock();
    }
    return lock;
}

void BounceWriteWindow::close()
{
    std::unique_lock lock(mMutex);
    mOpen = false;
}

std::shared_ptr<BounceWriteWindow> BounceWriteWindows::add(
    std::string peer, std::string endpoint, std::string key, std::vector<BounceWriteWindow::Range> ranges)
{
    auto id = bounceWindowId(key);
    auto window
        = std::make_shared<BounceWriteWindow>(std::move(peer), std::move(endpoint), std::move(key), std::move(ranges));
    std::lock_guard lock(mMutex);
    if (!mWindows.emplace(std::move(id), window).second)
    {
        throw std::invalid_argument("Bounce receive window already registered");
    }
    return window;
}

std::shared_ptr<BounceWriteWindow> BounceWriteWindows::find(std::string const& id, std::string const& peer) const
{
    std::lock_guard lock(mMutex);
    auto it = mWindows.find(id);
    return it != mWindows.end() && it->second->peer() == peer ? it->second : nullptr;
}

void BounceWriteWindows::remove(std::string const& id)
{
    std::shared_ptr<BounceWriteWindow> window;
    {
        std::lock_guard lock(mMutex);
        auto it = mWindows.find(id);
        if (it == mWindows.end())
        {
            return;
        }
        window = std::move(it->second);
        mWindows.erase(it);
    }
    window->close();
}

void BounceWriteWindows::removeIntersecting(std::uint64_t address, std::uint64_t length)
{
    std::vector<std::shared_ptr<BounceWriteWindow>> retired;
    {
        std::lock_guard lock(mMutex);
        for (auto it = mWindows.begin(); it != mWindows.end();)
        {
            if (it->second->intersects(address, length))
            {
                retired.push_back(std::move(it->second));
                it = mWindows.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }
    for (auto& window : retired)
    {
        window->close();
    }
}
} // namespace tensorrt_llm::executor::kv_cache::bounce
