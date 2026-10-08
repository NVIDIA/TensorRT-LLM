/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "tensorrt_llm/common/config.h"

#include <c10/util/Exception.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <utility>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! Scratch owned by one CUTLASS runner/stream. The runner serializes access.
class EagerMoeWorkspace
{
public:
    bool beginForward(int64_t owner, bool warmup, int64_t numTokens = 0)
    {
        // A cached runner can be shared by engines. Never reclaim another
        // engine's storage, or count overlapping forwards as separate windows.
        if (mActive || (mOwner != 0 && mOwner != owner))
        {
            mDisabled = true;
        }
        if (mDisabled || (!warmup && !mWarmedUp))
        {
            return false;
        }
        mCapacityBefore = capacity();
        if (!warmup && numTokens > 0 && !mRequirements.empty())
        {
            // Reserve a previously observed demand before embeddings/attention
            // can split a released large block. This is only a hint: exact
            // kernel sizing still controls allocation and reclamation.
            auto it = mRequirements.lower_bound(numTokens);
            if (it == mRequirements.end())
            {
                --it;
            }
            if (it->second > capacity())
            {
                mWorkspace = torch::Tensor{};
                try
                {
                    mWorkspace = torch::empty({it->second}, torch::dtype(torch::kInt8).device(torch::kCUDA));
                }
                catch (c10::OutOfMemoryError const&)
                {
                    // A conservative hint must not reject a request whose
                    // actual workspace demand may still fit.
                }
            }
        }
        mOwner = owner;
        mNumTokens = numTokens;
        mActive = true;
        mWarmup = warmup;
        mRequired = 0;
        return true;
    }

    void finishForward(bool completed)
    {
        TORCH_CHECK(mActive, "CUTLASS workspace forward was not started");
        mActive = false;
        if (!completed || mDisabled || mRequired == 0)
        {
            resetWindow();
            return;
        }
        if (mNumTokens > 0)
        {
            auto& required = mRequirements[mNumTokens];
            required = std::max(required, mRequired);
        }
        if (mWarmup)
        {
            mWarmedUp = true;
            resetWindow();
            return;
        }
        if (capacity() > mCapacityBefore || mRequired > capacity() / 2)
        {
            resetWindow();
            return;
        }
        mWindowMax = std::max(mWindowMax, mRequired);
        if (--mRemaining != 0)
        {
            return;
        }
        // Warmup exercises maximum shapes, not a minimum serving requirement.
        // Pure scratch can shrink below that high-water mark. Requiring three
        // forwards at no more than half capacity avoids reallocating for small
        // fluctuations; retain the largest actual demand in that window.
        auto const target = mWindowMax;
        resetWindow();
        auto const options = mWorkspace.options();
        // Allocate the small replacement before releasing the large block.
        // Otherwise the caching allocator can split that block for the small
        // tensor, pinning its segment and preventing reuse by the next burst.
        // Pure scratch needs no payload copy; stream ordering protects queued
        // consumers of the old allocation.
        torch::Tensor replacement;
        try
        {
            replacement = torch::empty({target}, options);
        }
        catch (c10::OutOfMemoryError const&)
        {
            // Under pressure, release scratch before retrying rather than
            // requiring enough headroom to hold both allocations temporarily.
            mWorkspace = torch::Tensor{};
            replacement = torch::empty({target}, options);
        }
        mWorkspace = std::move(replacement);
    }

    torch::Tensor const& get(int64_t required, bool capturing)
    {
        TORCH_CHECK(required >= 0, "Negative CUTLASS workspace requirement");
        if (capturing || (!mActive && mWarmedUp))
        {
            // Graph replay is invisible to the host tracker. A captured
            // allocation, or use outside its engine scope, permanently opts out.
            mDisabled = true;
        }
        if (mActive)
        {
            mRequired = std::max(mRequired, required);
        }
        if (capturing || !mWorkspace.defined() || capacity() < required)
        {
            mWorkspace = torch::Tensor{};
            mWorkspace = torch::empty({required}, torch::dtype(torch::kInt8).device(torch::kCUDA));
        }
        return mWorkspace;
    }

    torch::Tensor const& tensor() const
    {
        return mWorkspace;
    }

    int64_t capacity() const
    {
        return mWorkspace.defined() ? static_cast<int64_t>(mWorkspace.storage().nbytes()) : 0;
    }

private:
    void resetWindow()
    {
        mRemaining = 3;
        mWindowMax = 0;
    }

    torch::Tensor mWorkspace;
    std::map<int64_t, int64_t> mRequirements;
    int64_t mOwner = 0;
    int64_t mNumTokens = 0;
    int64_t mCapacityBefore = 0;
    int64_t mRequired = 0;
    int64_t mWindowMax = 0;
    int mRemaining = 3;
    bool mWarmedUp = false;
    bool mActive = false;
    bool mWarmup = false;
    bool mDisabled = false;
};

} // namespace torch_ext

TRTLLM_NAMESPACE_END
