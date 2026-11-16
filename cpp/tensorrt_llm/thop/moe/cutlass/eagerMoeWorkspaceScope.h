/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include "tensorrt_llm/common/config.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/util/ScopeExit.h>
#include <torch/custom_class.h>

#include <atomic>
#include <exception>
#include <map>
#include <tuple>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! Native owners serialize workspace access with their kernel launches.
class EagerMoeWorkspaceOwner : public torch::CustomClassHolder
{
public:
    virtual bool beginWorkspaceForward(int64_t owner, bool warmup, int64_t numTokens) = 0;
    virtual void finishWorkspaceForward(bool completed) = 0;
};

//! One synchronous model forward, discovered from native MoE calls on this thread.
class EagerMoeWorkspaceScope : public torch::CustomClassHolder
{
public:
    explicit EagerMoeWorkspaceScope(int64_t owner)
        : mOwner(owner)
    {
    }

    void begin(bool warmup, int64_t numTokens)
    {
        TORCH_CHECK(current() == nullptr, "Overlapping CUTLASS workspace scopes are unsupported");
        TORCH_CHECK(!mActive.exchange(true), "CUTLASS workspace scope is already active on another thread");
        current() = this;
        mWarmup = warmup;
        mNumTokens = numTokens;
        try
        {
            // Reserve known demand before unrelated model intermediates can
            // split a released burst block. New owners are discovered lazily.
            if (numTokens > 0 && !mEntries.empty())
            {
                auto const stream = c10::cuda::getCurrentCUDAStream();
                for (auto& [key, entry] : mEntries)
                {
                    if (sameStream(entry.stream, stream))
                    {
                        start(entry);
                    }
                }
            }
        }
        catch (...)
        {
            // Unwind every owner already started if a later reservation fails.
            finish(false);
            throw;
        }
    }

    bool finish(bool completed)
    {
        TORCH_CHECK(current() == this, "CUTLASS workspace scope must finish on its starting thread");
        current() = nullptr;
        auto reset = c10::make_scope_exit(
            [&]()
            {
                mVisited.clear();
                mLast = nullptr;
                mActive.store(false);
            });
        std::exception_ptr error;
        for (auto* entry : mVisited)
        {
            entry->seen = false;
            if (!entry->accepted)
            {
                continue;
            }
            entry->accepted = false;
            try
            {
                c10::cuda::CUDAStreamGuard guard(entry->stream);
                entry->owner->finishWorkspaceForward(completed);
            }
            catch (...)
            {
                // Cleanup must reach all owners even if one allocation fails.
                if (!error)
                {
                    error = std::current_exception();
                }
            }
        }
        if (error)
        {
            std::rethrow_exception(error);
        }
        return !mEntries.empty();
    }

    static void registerCurrent(EagerMoeWorkspaceOwner* owner, c10::DeviceIndex device)
    {
        if (auto* scope = current())
        {
            scope->registerOwner(owner, c10::cuda::getCurrentCUDAStream(device));
        }
    }

private:
    struct Entry
    {
        c10::intrusive_ptr<EagerMoeWorkspaceOwner> owner;
        c10::cuda::CUDAStream stream;
        bool seen = false;
        bool accepted = false;
    };

    static EagerMoeWorkspaceScope*& current()
    {
        static thread_local EagerMoeWorkspaceScope* scope = nullptr;
        return scope;
    }

    static bool sameStream(c10::cuda::CUDAStream lhs, c10::cuda::CUDAStream rhs)
    {
        // An external wrapper can alias a pooled stream with a different c10
        // stream ID. Match the runner's native device/handle workspace key.
        return lhs.device_index() == rhs.device_index() && lhs.stream() == rhs.stream();
    }

    void start(Entry& entry)
    {
        // Append before begin: vector growth must not strand an active owner.
        mVisited.push_back(&entry);
        entry.seen = true;
        entry.accepted = entry.owner->beginWorkspaceForward(mOwner, mWarmup, mNumTokens);
    }

    void registerOwner(EagerMoeWorkspaceOwner* owner, c10::cuda::CUDAStream stream)
    {
        if (mLast && mLast->owner.get() == owner && sameStream(mLast->stream, stream))
        {
            return;
        }
        auto const key = std::make_tuple(
            reinterpret_cast<uintptr_t>(owner), stream.device_index(), reinterpret_cast<uintptr_t>(stream.stream()));
        auto it = mEntries.find(key);
        if (it == mEntries.end())
        {
            it = mEntries
                     .emplace(key,
                         Entry{
                             c10::intrusive_ptr<EagerMoeWorkspaceOwner>::unsafe_reclaim_from_nonowning(owner), stream})
                     .first;
        }
        auto& entry = it->second;
        if (!entry.seen)
        {
            c10::cuda::CUDAStreamGuard guard(stream);
            start(entry);
        }
        mLast = &entry;
    }

    int64_t mOwner;
    int64_t mNumTokens = 0;
    bool mWarmup = false;
    std::atomic<bool> mActive{false};
    std::map<std::tuple<uintptr_t, c10::DeviceIndex, uintptr_t>, Entry> mEntries;
    std::vector<Entry*> mVisited;
    Entry* mLast = nullptr;
};

} // namespace torch_ext

TRTLLM_NAMESPACE_END
