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
#pragma once
#include "tensorrt_llm/runtime/worldConfig.h"
#if ENABLE_MULTI_DEVICE
#include "userbuffers.h"
#endif

namespace tensorrt_llm::runtime::ub
{
static auto constexpr tensor_prefix = "allreduce_ub_";

struct UBBuffer
{
    void* addr;
    int handle;
    size_t size;

    UBBuffer(void* a = nullptr, int h = -1, size_t s = 0)
        : addr(a)
        , handle(h)
        , size(s)
    {
    }

    [[nodiscard]] bool invalid() const
    {
        return (addr == nullptr) || (handle == -1) || (size == 0);
    }
};
#if ENABLE_MULTI_DEVICE
class UserBufferAllocator
{
public:
    static UserBufferAllocator& Instance();

    UserBufferAllocator() = default;

    void initialize(tensorrt_llm::runtime::WorldConfig const& world_config);
    bool is_initialized();
    UBBuffer allocate(size_t bytes);
    void deallocate(void* addr);
    UBBuffer get(int idx);
    communicator* comm();

private:
    UBBuffer register_ub_buffer(size_t bytes);

    communicator* ub_comm_;
    std::vector<UBBuffer> buffers_;
    bool is_initialized_;
    tensorrt_llm::runtime::WorldConfig world_config_;
};
#else
using communicator = void;
#endif
}; // namespace tensorrt_llm::runtime::ub
