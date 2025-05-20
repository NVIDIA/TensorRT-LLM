/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include <set>

namespace tensorrt_llm::runtime::ub
{
UserBufferAllocator& UserBufferAllocator::Instance()
{
    if (use_nccl_symmetric)
    {
        static NCCLUserBufferAllocator _;
        return _;
    }
    else
    {
        static UserBufferAllocator _;
        return _;
    }
}

void UserBufferAllocator::initialize(tensorrt_llm::runtime::WorldConfig const& worldConfig)
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
    return ub_buffer;
}

void UserBufferAllocator::deallocate(void* addr) {}

UBBuffer UserBufferAllocator::get(int idx)
{
    TLLM_CHECK(isInitialized() && idx < mBuffers.size() && !mBuffers[idx].invalid());
    return mBuffers[idx];
}

communicator* UserBufferAllocator::comm()
{
    TLLM_CHECK(isInitialized());
    return mUbComm;
}

void NCCLUserBufferAllocator::initialize(tensorrt_llm::runtime::WorldConfig const& worldConfig)
{
    if (!isInitialized())
    {
        TLLM_LOG_INFO("Initializing NCCLUserBufferAllocator");
        std::set<int> group;
        for (int i = 0; i < worldConfig.getSize(); i++)
        {
            group.insert(i);
        }
        mComm = getComm(group);
        mIsInitialized = true;
    }
}

UBBuffer NCCLUserBufferAllocator::registerUBBuffer(size_t bytes)
{
    TLLM_CHECK(isInitialized());
    UBBuffer ub_buffer;
    NCCLCHECK(ncclMemAlloc(&ub_buffer.addr, bytes));
    NCCLCHECK(ncclCommWindowRegister((*mComm), ub_buffer.addr, bytes, &ub_buffer.window, NCCL_WIN_COLL_SYMMETRIC));
    ub_buffer.handle = 5;
    ub_buffer.size = bytes;
    return ub_buffer;
}

bool UserBufferAllocator::use_nccl_symmetric = false;

}; // namespace tensorrt_llm::runtime::ub
