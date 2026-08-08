/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "ub_allocator.h"
#include "tensorrt_llm/common/opUtils.h"
#include <algorithm>
#include <set>
#include <stdexcept>

namespace tensorrt_llm::runtime::ub
{
UserBufferAllocator& UserBufferAllocator::Instance()
{
    static UserBufferAllocator _;
    return _;
}

void UserBufferAllocator::initialize(::tensorrt_llm::runtime::WorldConfig const& worldConfig)
{
    if (!isInitialized())
    {
        mUbComm = nullptr;
        mWorldConfig = worldConfig;
        create_communicator_grouped2(&mUbComm, worldConfig);
        TLLM_CHECK(mUbComm != nullptr);
        mIsInitialized = true;
    }
}

bool UserBufferAllocator::isInitialized()
{
    return mIsInitialized;
}

UBBuffer UserBufferAllocator::registerUBBuffer(size_t bytes)
{
    TLLM_CHECK(isInitialized());
    void* addr = nullptr;
    int handle = -1;
    handle = register_user_buffer_collective((void**) &addr, bytes, mUbComm);
    return {addr, handle, bytes};
}

UBBuffer UserBufferAllocator::allocate(size_t bytes)
{
    TLLM_CHECK(isInitialized());
    auto ub_buffer = registerUBBuffer(bytes);
    TLLM_CHECK(!ub_buffer.invalid());
    mBuffers.push_back(ub_buffer);
    mReleased.push_back(false);
    return ub_buffer;
}

void UserBufferAllocator::deallocate(void* addr)
{
    // This allocator is intentionally not internally synchronized. Manager-owned buffers reach this method under the
    // UserBuffersManager mutex; direct callers must serialize access even though the Python binding releases the GIL.
    auto const bufferIter
        = std::find_if(mBuffers.begin(), mBuffers.end(), [addr](auto const& buffer) { return buffer.addr == addr; });
    TLLM_CHECK(bufferIter != mBuffers.end());
    auto const index = std::distance(mBuffers.begin(), bufferIter);
    TLLM_CHECK(!mReleased[index]);
    mReleased[index] = true;

    while (!mBuffers.empty() && mReleased.back())
    {
        unregister_user_buffer_collective(mBuffers.back().handle, mUbComm);
        mBuffers.pop_back();
        mReleased.pop_back();
    }
}

UBBuffer UserBufferAllocator::get(int idx)
{
    TLLM_CHECK(isInitialized() && idx >= 0);
    auto const index = static_cast<size_t>(idx);
    TLLM_CHECK(index < mBuffers.size() && !mReleased[index] && !mBuffers[index].invalid());
    return mBuffers[index];
}

communicator* UserBufferAllocator::comm()
{
    TLLM_CHECK(isInitialized());
    return mUbComm;
}

}; // namespace tensorrt_llm::runtime::ub
