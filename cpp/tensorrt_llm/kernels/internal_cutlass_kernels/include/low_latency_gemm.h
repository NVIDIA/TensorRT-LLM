/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cutlass_extensions/gemm_configs.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda_runtime_api.h>
#include <vector>

// namespace tk = tensorrt_llm::common;

namespace tkc = tensorrt_llm::cutlass_extensions;

namespace tensorrt_llm
{
namespace kernels
{
namespace internal_cutlass_kernels
{

enum class KernelScheduleType
{
    AUTO,
    WS_PREFETECH,       // KernelTmaWarpSpecializedFP8FastAccumWithPrefetch
    WS_SPLIT_PREFETECH, // KernelTmaWarpSpecializedFP8FastAccumWithPrefetchAndSplitDMA
    //
};

struct LowLatencyCutlassGemmConfig
{

    cutlass_extensions::CutlassGemmConfig cutlass_gemm_config;
    KernelScheduleType kernel_schedule = KernelScheduleType::AUTO;

    std::string toString() const
    {

        std::stringstream tactic;
        tactic << cutlass_gemm_config.toString();

        tactic << "\tkernel sched: " << static_cast<int>(kernel_schedule);
        tactic << "\n";
        return tactic.str();
    }
};

inline std::ostream& operator<<(std::ostream& out, LowLatencyCutlassGemmConfig const& config)
{
    // clang-format off
    if (config.cutlass_gemm_config.is_tma_warp_specialized)
    {
        out << "tile_config_sm90_enum: " << int(config.cutlass_gemm_config.tile_config_sm90)
            << ", mainloop_schedule_enum: " << int(config.cutlass_gemm_config.mainloop_schedule)
            << ", epilogue_schedule_enum: " << int(config.cutlass_gemm_config.epilogue_schedule)
            << ", cluster_shape_enum: " << int(config.cutlass_gemm_config.cluster_shape);
    }
    else
    {
        out << "tile_config_enum: " << int(config.cutlass_gemm_config.tile_config_sm80)
            << ", split_k_style_enum: " << int(config.cutlass_gemm_config.split_k_style)
            << ", split_k_factor: " << config.cutlass_gemm_config.split_k_factor
            << ", stages: " << config.cutlass_gemm_config.stages;
    }
    out<<config.cutlass_gemm_config<<" kernel_schedule_enum: "<<static_cast<int>(config.kernel_schedule);
    return out;
}



class CutlassLowLatencyFp8GemmRunnerInterface
{
public:

    using ConfigType = LowLatencyCutlassGemmConfig;

    CutlassLowLatencyFp8GemmRunnerInterface() {}

    virtual ~CutlassLowLatencyFp8GemmRunnerInterface() {}

    virtual void gemm(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, float alpha, float beta, void const* C, void* D, int m, int n,
        int k,float pdl_overlap_ratio,float prefetch_ratio, ConfigType gemmConfig, char* workspacePtr, size_t const workspaceBytes, cudaStream_t stream)
        = 0;

    virtual size_t getWorkspaceSize(int const m, int const n, int const k) = 0;

    virtual std::vector<ConfigType> getConfigs() const = 0;
};

template <typename T>

class CutlassLowLatencyFp8GemmRunner : public virtual CutlassLowLatencyFp8GemmRunnerInterface
{
public:

    CutlassLowLatencyFp8GemmRunner();
    ~CutlassLowLatencyFp8GemmRunner() = default;
    // gemm A rowMajor,  B colMajor, C and D rowMajor
    void gemm(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, float alpha, float beta, void const* C, void* D, int m, int n, int k, float pdl_overlap_ratio,float prefetech_ratio,
        ConfigType gemmConfig, char* workspacePtr, size_t const workspaceBytes,
        cudaStream_t stream) override;
    size_t getWorkspaceSize(int const m, int const n, int const k) override;
    std::vector<ConfigType> getConfigs() const override;

private:
    size_t dispatchToArch(__nv_fp8_e4m3 const* A, __nv_fp8_e4m3 const* B, float alpha, float beta, void const* C,
        void* D, int m, int n, int k,float pdl_overlap_ratio,float prefetech_ratio, ConfigType gemmConfig, char* workspacePtr,
        size_t const workspaceBytes, cudaStream_t stream);
        size_t getWorkspaceSizeImpl(int const m, int const n, int const k);
    int mSm;
};

}; // namespace internal_cutlass_kernels
}; // namespace kernels
}; // namespace tensorrt_llm
