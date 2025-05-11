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

#include <cuda_runtime_api.h>
#include <vector>

#include "cutlass_extensions/gemm_configs.h"
#include "tensorrt_llm/common/quantization.h"

namespace tk = tensorrt_llm::common;
namespace tkc = tensorrt_llm::cutlass_extensions;

namespace tensorrt_llm
{
namespace kernels
{
namespace internal_cutlass_kernels
{

/*
  This runner supports:
  FP4 inputs (A and B)
  float blockwise scaling factor
  float alpha scalings
  T output (D) where T = {float, half, __nv_bfloat16}

  Activations, biases and outputs are all assumed to be row-major.
  Weights are assumed to be column-major.
  Block scaling factor are interleaved.
*/

class CutlassFp4GemmRunnerInterface
{
public:
    CutlassFp4GemmRunnerInterface() {}

    virtual ~CutlassFp4GemmRunnerInterface() {}

    virtual void gemm(void* D, void const* A, void const* B, void const* input_sf, void const* weight_sf,
        float const* global_sf, int m, int n, int k, int batch_count, tkc::CutlassGemmConfig gemmConfig,
        char* workspace, const size_t workspaceBytes, cudaStream_t stream)
        = 0;

    // Returns desired workspace size in bytes.
    virtual size_t getWorkspaceSize(int const m, int const n, int const k, int batch_count) = 0;

    virtual std::vector<tkc::CutlassGemmConfig> getConfigs() const = 0;
};

template <typename T>
class CutlassFp4GemmRunner : public virtual CutlassFp4GemmRunnerInterface
{
public:
    CutlassFp4GemmRunner();
    ~CutlassFp4GemmRunner();

    void gemm(void* D, void const* A, void const* B, void const* input_sf, void const* weight_sf,
        float const* global_sf, int m, int n, int k, int batch_count, tkc::CutlassGemmConfig gemmConfig,
        char* workspace, const size_t workspaceBytes, cudaStream_t stream) override;

    // Returns desired workspace size in bytes.
    size_t getWorkspaceSize(int const m, int const n, int const k, int const batch_count) override;

    std::vector<tkc::CutlassGemmConfig> getConfigs() const override;

private:
    size_t dispatchToArch(T* D, void const* A, void const* B, void const* input_sf, void const* weight_sf,
        float const* global_sf, int m, int n, int k, int batch_count, tkc::CutlassGemmConfig gemmConfig,
        char* workspace, const size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr);

    size_t getWorkspaceSizeImpl(int const m, int const n, int const k, int const batch_count);

    int mSm;
    int mMultiProcessorCount;
};

} // namespace internal_cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
