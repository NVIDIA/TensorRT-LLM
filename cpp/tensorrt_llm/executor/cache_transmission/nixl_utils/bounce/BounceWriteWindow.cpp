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
#if OPENSSL_VERSION_NUMBER >= 0x30000000L
#include <openssl/core_names.h>
#include <openssl/params.h>
#endif

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

#if OPENSSL_VERSION_NUMBER >= 0x30000000L
auto makeMacContext()
{
    std::unique_ptr<EVP_MAC, decltype(&EVP_MAC_free)> mac(EVP_MAC_fetch(nullptr, "HMAC", nullptr), EVP_MAC_free);
    std::unique_ptr<EVP_MAC_CTX, decltype(&EVP_MAC_CTX_free)> ctx(
        mac ? EVP_MAC_CTX_new(mac.get()) : nullptr, EVP_MAC_CTX_free);
    char digest[] = "SHA256";
    OSSL_PARAM params[]
        = {OSSL_PARAM_construct_utf8_string(OSSL_MAC_PARAM_DIGEST, digest, 0), OSSL_PARAM_construct_end()};
    if (!ctx || !EVP_MAC_CTX_set_params(ctx.get(), params))
    {
        throw std::runtime_error("Cannot initialize bounce control authentication");
    }
    return ctx;
}
#endif

std::array<unsigned char, EVP_MAX_MD_SIZE> signature(std::string const& key, std::string_view bytes)
{
    if (key.size() != kKeyBytes)
    {
        throw std::invalid_argument("Invalid bounce window key size");
    }
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    auto const* data = reinterpret_cast<unsigned char const*>(bytes.data());
#if OPENSSL_VERSION_NUMBER >= 0x30000000L
    // Only the provider setup is shared between calls. Each message starts with its own window key.
    thread_local auto ctx = makeMacContext();
    std::size_t length{};
    if (!EVP_MAC_init(ctx.get(), reinterpret_cast<unsigned char const*>(key.data()), key.size(), nullptr)
        || !EVP_MAC_update(ctx.get(), data, bytes.size())
        || !EVP_MAC_final(ctx.get(), digest.data(), &length, digest.size()) || length < kTagBytes)
#else
    unsigned int length{};
    if (!HMAC(EVP_sha256(), key.data(), static_cast<int>(key.size()), data, bytes.size(), digest.data(), &length)
        || length < kTagBytes)
#endif
    {
        throw std::runtime_error("Cannot sign bounce control frame");
    }
    return digest;
}
} // namespace

std::string bounceWindowId(std::string const& key)
{
    auto const digest = signature(key, "TRTLLM bounce window id v1");
    return {reinterpret_cast<char const*>(digest.data()), kTagBytes};
}

std::string signBounceFrame(std::string frame, std::string const& key)
{
    return signBounceFrame(std::move(frame), key, bounceWindowId(key));
}

std::string signBounceFrame(std::string frame, std::string const& key, std::string const& id)
{
    frame.reserve(frame.size() + 2 * kTagBytes);
    frame.append(id);
    auto const digest = signature(key, frame);
    frame.append(reinterpret_cast<char const*>(digest.data()), kTagBytes);
    return frame;
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
    , mId(bounceWindowId(mKey))
{
    if (!std::is_sorted(ranges.begin(), ranges.end()))
    {
        std::sort(ranges.begin(), ranges.end());
    }
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
    auto window
        = std::make_shared<BounceWriteWindow>(std::move(peer), std::move(endpoint), std::move(key), std::move(ranges));
    std::lock_guard lock(mMutex);
    if (!mWindows.emplace(window->id(), window).second)
    {
        throw std::invalid_argument("Bounce receive window already registered");
    }
    return window;
}

std::shared_ptr<BounceWriteWindow> BounceWriteWindows::addSlots(std::string peer, std::string endpoint, std::string key,
    std::span<std::int64_t const> slots, BounceSlotLayout const& layout)
{
    if (layout.slotBytes == 0 || layout.slotBytes > layout.stride || layout.offsets.size() != layout.lengths.size())
    {
        throw std::invalid_argument("Invalid bounce receive pool geometry");
    }
    auto const available = std::numeric_limits<std::uint64_t>::max() - layout.base;
    if (layout.numSlots
        && (layout.slotBytes > available || layout.numSlots - 1 > (available - layout.slotBytes) / layout.stride))
    {
        throw std::invalid_argument("Bounce receive pool exceeds the address space");
    }

    std::vector<BounceWriteWindow::Range> pieces;
    pieces.reserve(layout.offsets.size());
    for (std::size_t i = 0; i < layout.offsets.size(); ++i)
    {
        auto const offset = layout.offsets[i];
        auto const length = layout.lengths[i];
        if (offset < 0 || length <= 0 || static_cast<std::uint64_t>(offset) > layout.slotBytes
            || static_cast<std::uint64_t>(length) > layout.slotBytes - static_cast<std::uint64_t>(offset))
        {
            throw std::invalid_argument("Bounce writer range exceeds the slot payload");
        }
        pieces.emplace_back(offset, length);
    }
    std::vector<BounceWriteWindow::Range> ranges;
    if (!pieces.empty() && slots.size() > ranges.max_size() / pieces.size())
    {
        throw std::invalid_argument("Too many bounce receive ranges");
    }
    ranges.reserve(slots.size() * pieces.size());
    for (auto slot : slots)
    {
        if (slot < 0)
        {
            continue;
        }
        if (static_cast<std::uint64_t>(slot) >= layout.numSlots)
        {
            throw std::invalid_argument("Bounce receive slot exceeds the local pool");
        }
        auto const base = layout.base + static_cast<std::uint64_t>(slot) * layout.stride;
        for (auto const& [offset, length] : pieces)
        {
            ranges.emplace_back(base + offset, length);
        }
    }
    return add(std::move(peer), std::move(endpoint), std::move(key), std::move(ranges));
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

void BounceWriteWindows::seal(std::string const& id)
{
    std::shared_ptr<BounceWriteWindow> window;
    {
        std::lock_guard lock(mMutex);
        auto it = mWindows.find(id);
        if (it == mWindows.end())
        {
            return;
        }
        window = it->second;
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
