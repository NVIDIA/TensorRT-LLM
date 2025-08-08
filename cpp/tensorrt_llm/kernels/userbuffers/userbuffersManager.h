/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/userbuffers/ub_interface.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include <memory>
#include <mutex>
#include <vector>

namespace tensorrt_llm::runtime::ub
{

class UserBufferDeleter
{
public:
    void operator()(void* ptr);
};

using UBBufferPtr = std::unique_ptr<void, UserBufferDeleter>;
using UBBufferInfo = std::pair<tensorrt_llm::runtime::ub::UBBuffer, bool>;

class UserBuffersManager
{
public:
    static UserBuffersManager& get_instance();
    UserBuffersManager() = default;

    //! @brief Initialize the userbuffers manager.
    //! @param tp_size Tensor parallel size.
    //! @param pp_size Pipeline parallel size.
    //! @param cp_size Compute parallel size.
    //! @param rank The rank of the current GPU.
    //! @param gpus_per_node The number of GPUs per node.
    //! @param buffer_size The size of the buffer to allocate. All buffers allocated by this manager will have this
    //! size.
    //! @param use_nccl_symmetric Whether to use NCCL symmetric communication.
    void initialize(int64_t tp_size, int64_t pp_size, int64_t cp_size, int64_t rank, int64_t gpus_per_node,
        int64_t buffer_size, bool use_nccl_symmetric);

    //! @brief Create a UB tensor from the given shape, strides and data type. The function will choose available UB
    //! buffer or create a new one if no available buffer is found.
    //! @param buffer_size The size of the buffer to allocate.
    //! @return A unique_ptr to the buffer and the UBBuffer object.
    //! @note Do not manually call release_buffer with the buffer address in tensorrt_llm::runtime::ub::UBBuffer
    std::pair<UBBufferPtr, tensorrt_llm::runtime::ub::UBBuffer> allocate_userbuffers(int64_t buffer_size);

    //! @brief Search the buffer from the list of buffers.
    //! @param addr The address of the buffer to search for.
    //! @return The buffer and whether it is assigned to a tensor. If not found, the UBBuffer is invalid.
    tensorrt_llm::runtime::ub::UBBuffer search_buffer(void* addr);

    //! @brief Get the communicator.
    //! @return The communicator.
    tensorrt_llm::runtime::ub::communicator* comm();

    //! @brief Release the buffer. It does not deallocate the buffer, but just release it to the pool.
    //! @param addr The address of the buffer to release.
    void release_buffer(void* addr);

private:
    std::mutex mutex_;
    std::vector<UBBufferInfo> buffers_;
    int64_t buffer_size_;
};

void initialize_userbuffers_manager(int64_t tp_size, int64_t pp_size, int64_t cp_size, int64_t rank,
    int64_t gpus_per_node, int64_t buffer_size, bool use_nccl_symmetric);

} // namespace tensorrt_llm::runtime::ub
