/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "tensorrt_llm/common/config.h"

#include <torch/types.h>

#include <algorithm>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! Scratch owned by one CUTLASS runner/stream. The runner serializes access.
class EagerMoeWorkspace
{
public:
    bool beginForward(int64_t owner, bool warmup)
    {
        // A cached runner can be shared by engines. Never reclaim another
        // engine's storage, or count overlapping forwards as separate windows.
        if (mActive || (mOwner != 0 && mOwner != owner))
        {
            mDisabled = true;
        }
        if (mDisabled || (!warmup && !mHasFloor))
        {
            return false;
        }
        mOwner = owner;
        mActive = true;
        mWarmup = warmup;
        mRequired = 0;
        mCapacityBefore = capacity();
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
        if (mWarmup)
        {
            mFloor = std::max(mFloor, capacity());
            mHasFloor = true;
            resetWindow();
            return;
        }
        if (capacity() > mCapacityBefore || capacity() <= mFloor || mRequired == capacity())
        {
            resetWindow();
            return;
        }
        mWindowMax = std::max(mWindowMax, mRequired);
        if (--mRemaining != 0)
        {
            return;
        }
        auto const target = std::max(mFloor, mWindowMax);
        resetWindow();
        auto const options = mWorkspace.options();
        // Pure scratch: no payload to preserve. All consumers were submitted
        // on this allocation's stream. Release before allocating its replacement.
        mWorkspace = torch::Tensor{};
        mWorkspace = torch::empty({target}, options);
    }

    torch::Tensor const& get(int64_t required, bool capturing)
    {
        TORCH_CHECK(required >= 0, "Negative CUTLASS workspace requirement");
        if (capturing || (!mActive && mHasFloor))
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
    int64_t mOwner = 0;
    int64_t mFloor = 0;
    int64_t mCapacityBefore = 0;
    int64_t mRequired = 0;
    int64_t mWindowMax = 0;
    int mRemaining = 3;
    bool mHasFloor = false;
    bool mActive = false;
    bool mWarmup = false;
    bool mDisabled = false;
};

} // namespace torch_ext

TRTLLM_NAMESPACE_END
