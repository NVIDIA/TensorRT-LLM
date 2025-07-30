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

#include "tensorrt_llm/runtime/virtualMemory.h"
#include <cuda_runtime_api.h>
#include <sys/types.h>

extern "C"
{

    void* tensorrt_llm_virtual_memory_alloc(ssize_t size, int device, cudaStream_t) noexcept
    {
        void* ptr{};
        try
        {
            tensorrt_llm::runtime::getVirtualMemoryAllocator().allocate(&ptr, size, device);
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_EXCEPTION(e);
            ptr = {};
        }
        catch (...)
        {
            TLLM_LOG_ERROR("Unknown exception thrown allocating virtual memory");
            ptr = {};
        }

        return ptr;
    }

    void tensorrt_llm_virtual_memory_free(void* ptr, ssize_t size, cudaStream_t) noexcept
    {
        try
        {
            tensorrt_llm::runtime::getVirtualMemoryAllocator().deallocate(ptr, size);
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_EXCEPTION(e);
        }
        catch (...)
        {
            TLLM_LOG_ERROR("Unknown exception thrown deallocating virtual memory");
        }
    }
}
