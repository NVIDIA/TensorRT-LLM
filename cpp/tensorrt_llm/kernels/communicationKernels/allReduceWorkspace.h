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
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.h"
#include "tensorrt_llm/runtime/ipcUtils.h"

namespace tensorrt_llm::kernels::ar_fusion
{

class Workspace
{
public:
    Workspace(int rank, int tp_size, int max_token_num, int hidden_dim,
        std::shared_ptr<tensorrt_llm::runtime::CudaStream> stream_ptr);
    ~Workspace();
    void** get_workspace();

private:
    tensorrt_llm::runtime::WorldConfig m_world_config;
    std::shared_ptr<tensorrt_llm::runtime::BufferManager> m_buffer_mgr;
    std::vector<tensorrt_llm::runtime::IpcMemory> m_ipc_mem_handles;
    void* m_workspace;
    std::shared_ptr<tensorrt_llm::runtime::CudaStream> m_cuda_stream;
    void* m_flag_d_ptr;
};

void lamport_initialize(void* ptr, int bytes, cudaStream_t stream);
} // namespace tensorrt_llm::kernels::ar_fusion
