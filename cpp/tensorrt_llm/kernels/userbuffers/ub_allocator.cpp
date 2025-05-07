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

namespace tensorrt_llm::runtime::ub
{
UserBufferAllocator& UserBufferAllocator::Instance()
{
    static UserBufferAllocator _;
    return _;
}

void UserBufferAllocator::initialize(tensorrt_llm::runtime::WorldConfig const& world_config)
{
    if (!is_initialized())
    {
        ub_comm_ = nullptr;
        world_config_ = world_config;
        create_communicator_grouped2(&ub_comm_, world_config_);
        TLLM_CHECK(ub_comm_ != nullptr);
        is_initialized_ = true;
    }
}

bool UserBufferAllocator::is_initialized()
{
    return is_initialized_;
}

UBBuffer UserBufferAllocator::register_ub_buffer(size_t bytes)
{
    TLLM_CHECK(is_initialized());
    void* addr = nullptr;
    int handle = -1;
    handle = register_user_buffer_collective((void**) &addr, bytes, ub_comm_);
    return {addr, handle, bytes};
}

UBBuffer UserBufferAllocator::allocate(size_t bytes)
{
    TLLM_CHECK(is_initialized());
    auto ub_buffer = register_ub_buffer(bytes);
    TLLM_CHECK(!ub_buffer.invalid());
    buffers_.push_back(ub_buffer);
    return ub_buffer;
}

void UserBufferAllocator::deallocate(void* addr) {}

UBBuffer UserBufferAllocator::get(int idx)
{
    TLLM_CHECK(is_initialized() && idx < buffers_.size() && !buffers_[idx].invalid());
    return buffers_[idx];
}

communicator* UserBufferAllocator::comm()
{
    TLLM_CHECK(is_initialized());
    return ub_comm_;
}
}; // namespace tensorrt_llm::runtime::ub
