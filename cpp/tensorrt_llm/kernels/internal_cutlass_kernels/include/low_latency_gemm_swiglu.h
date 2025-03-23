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
#include "low_latency_gemm.h"

// namespace tk = tensorrt_llm::common;

namespace tkc = tensorrt_llm::cutlass_extensions;

namespace tensorrt_llm
{
namespace kernels
{
namespace internal_cutlass_kernels
{

class CutlassLowLatencyFp8GemmSwigluRunnerInterface
{
public:
    using ConfigType = LowLatencyCutlassGemmConfig;

    CutlassLowLatencyFp8GemmSwigluRunnerInterface() {}

    virtual ~CutlassLowLatencyFp8GemmSwigluRunnerInterface() {}

    // TODO...
    virtual void gemm(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, float alpha, float beta, float scale_d0, float scale_d1,
        void const* C, void* D, int m, int n, int k, float pdl_overlap_ratio, float prefetch_ratio,
        ConfigType gemmConfig, char* workspacePtr, size_t const workspaceBytes, cudaStream_t stream)
        = 0;

    virtual size_t getWorkspaceSize(int const m, int const n, int const k) = 0;

    virtual std::vector<ConfigType> getConfigs() const = 0;
};

template <typename T>
class CutlassLowLatencyFp8GemmSwigluRunner : public virtual CutlassLowLatencyFp8GemmSwigluRunnerInterface
{
public:
    CutlassLowLatencyFp8GemmSwigluRunner();
    ~CutlassLowLatencyFp8GemmSwigluRunner() = default;
    // gemm A rowMajor,  B colMajor, C and D rowMajor
    void gemm(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, float alpha, float beta, float scale_d0, float scale_d1,
        void const* C, void* D, int m, int n, int k, float pdl_overlap_ratio, float prefetech_ratio,
        ConfigType gemmConfig, char* workspacePtr, size_t const workspaceBytes, cudaStream_t stream) override;
    size_t getWorkspaceSize(int const m, int const n, int const k) override;
    std::vector<ConfigType> getConfigs() const override;

private:
    size_t dispatchToArch(__nv_fp8_e4m3 const* A, __nv_fp8_e4m3 const* B, float alpha, float beta, float scale_d0,
        float scale_d1, void const* C, void* D, int m, int n, int k, float pdl_overlap_ratio, float prefetech_ratio,
        ConfigType gemmConfig, char* workspacePtr, size_t const workspaceBytes, cudaStream_t stream);

    size_t getWorkspaceSizeImpl(int const m, int const n, int const k);
    int mSm;
};

}; // namespace internal_cutlass_kernels
}; // namespace kernels
}; // namespace tensorrt_llm
