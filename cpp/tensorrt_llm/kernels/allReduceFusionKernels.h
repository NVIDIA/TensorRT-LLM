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
#include <NvInferRuntime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/ipcUtils.h"

namespace tensorrt_llm::kernels::ar_fusion
{
static constexpr int kElemsPerAccess = 8;
static constexpr int kOneShotMaxToken = 128;
static constexpr int kBarrierFlagCount = 256;

// DS R1
// pattern1: AR+Add_RMS+Quant
// [m, 7168] bf16 allreduce_in, [m, 7168] bf16 residual_in
// [m, 7168] bf16 residual_out, [m, 7168] fp4 quant_out
// pattern2: AR+AddRMS
// [m, 7168] bf16 allreduce_in, [m, 7168] bf16 residual_in
// [m, 7168] bf16 norm_out
struct AllReduceFusionParams
{
    int nranks;
    int rank;
    nvinfer1::DataType dtype;
    // size = token_num * hidden_dim
    int size;
    int hidden_dim;
    void** workspace;
    void* allreduce_in;
    void* residual_in;
    void* residual_out;
    void* norm_out;
    void* quant_out;
    void* scale_out;
    void* rms_gamma;
    float rms_eps;
    float* scale_factor;
    cudaStream_t stream;
};

void allreduce_fusion_op(AllReduceFusionParams const& params);

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

/////////////////////////////////////////////////////////////////
//                  * MoE Reduction Fusion *                   //
/////////////////////////////////////////////////////////////////

// Fuse MoE Reduction before AR + RMS
// pattern1: MoE Reduction + Add Residual + AR + ADD_RMS + Quant
// pattern2: MoE Reduction + Add Residual + AR + ADD_RMS
// [device_num_experts, m, 7168] bf16 moe_reduction_active_experts_token_input
// [m, 7168] bf16 moe_reduction_token_input
// [device_num_experts, m] moe_reduction_scale_input
struct MoeReductionAllReduceFusionParams : public AllReduceFusionParams
{
    // * moe reduction specific params
    // Refer to kernel implementation on layout of those params
    // number of active experts on current device
    int* moe_reduction_device_num_experts = nullptr;
    // per token per expert fp32 scale
    float* moe_reduction_scale_input = nullptr;
    // per token per expert input
    void* moe_reduction_active_experts_token_input = nullptr;
    // per token input
    void* moe_reduction_token_input = nullptr;
};

void moereduction_allreduce_fusion_op(MoeReductionAllReduceFusionParams const& params);

} // namespace tensorrt_llm::kernels::ar_fusion
