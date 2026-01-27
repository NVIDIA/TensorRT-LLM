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

#include "tensorrt_llm/batch_manager/kvCacheManagerV2Utils.h"
#include "tensorrt_llm/common/logger.h"
#include <cassert>
#include <cstdio>
#include <cuda.h>
#include <fcntl.h>
#include <memory>
#include <unistd.h>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

template <typename Func>
bool loopedReadWrite(Func&& func, ssize_t size) noexcept
{
    ssize_t count = 0;
    while (count < size)
    {
        ssize_t bytes = func(count);
        if (bytes <= 0)
        {
            if (errno == EINTR)
            {
                continue; // Retry on interrupt
            }
            TLLM_LOG_ERROR("Disk read/write failed: %s\n", strerror(errno));
            return false;
        }
        count += bytes;
    }
    assert(count == size);
    return true;
}

bool writeAll(int fd, ssize_t pos, void const* data, ssize_t size) noexcept
{
    return loopedReadWrite([=](ssize_t finished)
        { return pwrite(fd, static_cast<std::byte const*>(data) + finished, size - finished, pos + finished); },
        size);
}

bool readAll(int fd, ssize_t pos, void* data, ssize_t size) noexcept
{
    return loopedReadWrite([=](ssize_t finished)
        { return pread(fd, static_cast<std::byte*>(data) + finished, size - finished, pos + finished); },
        size);
}

template <typename DstAddr, typename SrcAddr>
struct UserData
{
    std::vector<Task<DstAddr, SrcAddr>> tasks;
    ssize_t numBytes;
};

CUDA_CB void hostFnDiskToDiskCopy(void* userData) noexcept
{
    // @TODO: enable multi-threading with a thread pool
    using Data = UserData<DiskAddress, DiskAddress>;
    auto const data = std::unique_ptr<Data>(static_cast<Data*>(userData));
    std::vector<std::byte> buffer(data->numBytes);
    bool success = true;
    for (auto const& t : data->tasks)
    {
        success = success && readAll(t.src.fd, t.src.pos, buffer.data(), data->numBytes);
        success = success && writeAll(t.dst.fd, t.dst.pos, buffer.data(), data->numBytes);
    }
    if (!success)
    {
        TLLM_LOG_ERROR("[kvCacheManagerV2Utils] hostFnDiskToDiskCopy failed.\n");
    }
}

CUDA_CB void hostFnDiskToHostCopy(void* userData) noexcept
{
    // @TODO: enable multi-threading with a thread pool
    using Data = UserData<MemAddress, DiskAddress>;
    auto const data = std::unique_ptr<Data>(static_cast<Data*>(userData));
    bool success = true;
    for (auto const& t : data->tasks)
    {
        success = success && readAll(t.src.fd, t.src.pos, reinterpret_cast<void*>(t.dst), data->numBytes);
    }
    if (!success)
    {
        TLLM_LOG_ERROR("[kvCacheManagerV2Utils] hostFnDiskToHostCopy failed.\n");
    }
}

CUDA_CB void hostFnHostToDiskCopy(void* userData) noexcept
{
    // @TODO: enable multi-threading with a thread pool
    using Data = UserData<DiskAddress, MemAddress>;
    auto const data = std::unique_ptr<Data>(static_cast<Data*>(userData));
    bool success = true;
    for (auto const& t : data->tasks)
    {
        success = success && writeAll(t.dst.fd, t.dst.pos, reinterpret_cast<void const*>(t.src), data->numBytes);
    }
    if (!success)
    {
        TLLM_LOG_ERROR("[kvCacheManagerV2Utils] hostFnHostToDiskCopy failed.\n");
    }
}

CUDA_CB void hostFnHostToHostCopy(void* userData) noexcept
{
    // @TODO: enable multi-threading with a thread pool
    using Data = UserData<MemAddress, MemAddress>;
    auto const data = std::unique_ptr<Data>(static_cast<Data*>(userData));
    for (auto const& t : data->tasks)
    {
        memcpy(reinterpret_cast<void*>(t.dst), reinterpret_cast<void const*>(t.src), data->numBytes);
    }
}

CUresult copyDiskToDisk(std::vector<Task<DiskAddress, DiskAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept
{
    using Data = UserData<DiskAddress, DiskAddress>;
    auto data = std::make_unique<Data>(Data{std::move(tasks), numBytes});
    return cuLaunchHostFunc(stream, hostFnDiskToDiskCopy, data.release());
}

CUresult copyDiskToHost(std::vector<Task<MemAddress, DiskAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept
{
    using Data = UserData<MemAddress, DiskAddress>;
    auto data = std::make_unique<Data>(Data{std::move(tasks), numBytes});
    return cuLaunchHostFunc(stream, hostFnDiskToHostCopy, data.release());
}

CUresult copyHostToDisk(std::vector<Task<DiskAddress, MemAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept
{
    using Data = UserData<DiskAddress, MemAddress>;
    auto data = std::make_unique<Data>(Data{std::move(tasks), numBytes});
    return cuLaunchHostFunc(stream, hostFnHostToDiskCopy, data.release());
}

CUresult copyHostToHost(std::vector<Task<MemAddress, MemAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept
{
    using Data = UserData<MemAddress, MemAddress>;
    auto data = std::make_unique<Data>(Data{std::move(tasks), numBytes});
    return cuLaunchHostFunc(stream, hostFnHostToHostCopy, data.release());
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
