/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <cstdint>
#include <cuda.h>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{
struct DiskAddress
{
    int fd;
    ssize_t pos;
};

using MemAddress = std::uintptr_t;

template <typename DstAddr, typename SrcAddr>
struct Task
{
    DstAddr dst;
    SrcAddr src;
};

CUresult copyDiskToDisk(std::vector<Task<DiskAddress, DiskAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept;
CUresult copyDiskToHost(std::vector<Task<MemAddress, DiskAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept;
CUresult copyHostToDisk(std::vector<Task<DiskAddress, MemAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept;
CUresult copyHostToHost(std::vector<Task<MemAddress, MemAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept;
CUresult copyHostToDevice(
    std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, CUstream stream) noexcept;
CUresult copyDeviceToHost(
    std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, CUstream stream) noexcept;
CUresult copyDeviceToDevice(
    std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, CUstream stream) noexcept;
} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
