/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/communicationKernels/allReduceWorkspace.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::ar_fusion
{

__global__ void lamport_initialize_kernel(float* ptr, size_t size)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;
    ptr[idx] = -0.f;
}

void lamport_initialize(void* ptr, size_t bytes, cudaStream_t stream)
{
    int grid_size = static_cast<int>((bytes + 1023) / 1024);
    lamport_initialize_kernel<<<grid_size, 1024, 0, stream>>>(reinterpret_cast<float*>(ptr), bytes / sizeof(float));
}

Workspace::Workspace(int rank, int tp_size, int max_token_num, int hidden_dim,
    std::shared_ptr<tensorrt_llm::runtime::CudaStream> stream_ptr)
    : m_world_config(tp_size, 1, 1, rank, tp_size)
    , m_cuda_stream(stream_ptr)
{
    bool p2p_supported = tensorrt_llm::runtime::canAccessPeer(m_world_config);
    TLLM_CHECK(p2p_supported);
    int device_id;
    TLLM_CUDA_CHECK(cudaGetDevice(&device_id));
    m_buffer_mgr = std::make_shared<tensorrt_llm::runtime::BufferManager>(m_cuda_stream);
    size_t buffer_size = tp_size * max_token_num * hidden_dim * sizeof(half);
    size_t flag_size = tp_size * kBarrierFlagCount * sizeof(int);
    size_t lamport_comm_size
        = static_cast<size_t>(tp_size) * std::max(kOneShotMaxToken, max_token_num) * hidden_dim * sizeof(half);
    size_t lamport_buffer_size = 3 * lamport_comm_size;
    for (auto size : {buffer_size, flag_size, lamport_buffer_size})
    {
        m_ipc_mem_handles.emplace_back(size, *m_buffer_mgr, m_world_config, p2p_supported);
    }
    std::vector<void*> workspace;
    for (auto& ipc_mem_handle : m_ipc_mem_handles)
    {
        for (int r = 0; r < tp_size; ++r)
        {
            workspace.push_back(ipc_mem_handle.getCommPtrs()[r]);
        }
    }
    // flag_buffer[0], atomic flag read counter
    // flag_buffer[1], non-lamport flag
    // flag_buffer[2], lamport flag
    TLLM_CUDA_CHECK(cudaMalloc(&m_flag_d_ptr, 3 * sizeof(int)));
    std::vector<int> h_flag_data{0, 0, 0};
    TLLM_CUDA_CHECK(cudaMemcpy(m_flag_d_ptr, h_flag_data.data(), 3 * sizeof(int), cudaMemcpyHostToDevice));
    workspace.push_back(m_flag_d_ptr);
    // layout_buffer[0], clear size for next lamport kernel
    // layout_buffer[1], triple buffer offset for lamport kernel
    TLLM_CUDA_CHECK(cudaMalloc(&m_layout_d_ptr, 2 * sizeof(int64_t)));
    std::vector<int64_t> h_layout_data{0, static_cast<int64_t>(lamport_comm_size)};
    TLLM_CUDA_CHECK(cudaMemcpy(m_layout_d_ptr, h_layout_data.data(), 2 * sizeof(int64_t), cudaMemcpyHostToDevice));
    workspace.push_back(m_layout_d_ptr);

    TLLM_CUDA_CHECK(cudaMalloc(&m_workspace, workspace.size() * sizeof(void*)));
    TLLM_CUDA_CHECK(
        cudaMemcpy(m_workspace, workspace.data(), workspace.size() * sizeof(void*), cudaMemcpyHostToDevice));
    lamport_initialize(m_ipc_mem_handles[2].getCommPtrs()[rank], lamport_buffer_size, 0);
}

Workspace::~Workspace()
{
    if (m_flag_d_ptr)
    {
        TLLM_CUDA_CHECK(cudaFree(m_flag_d_ptr));
    }
    if (m_layout_d_ptr)
    {
        TLLM_CUDA_CHECK(cudaFree(m_layout_d_ptr));
    }
    if (m_workspace)
    {
        TLLM_CUDA_CHECK(cudaFree(m_workspace));
    }
}

void** Workspace::get_workspace()
{
    return reinterpret_cast<void**>(m_workspace);
}
}; // namespace kernels::ar_fusion

TRTLLM_NAMESPACE_END
