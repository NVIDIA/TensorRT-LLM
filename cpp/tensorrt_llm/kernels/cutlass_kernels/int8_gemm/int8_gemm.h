/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "cutlass_extensions/gemm_configs.h"
#include "tensorrt_llm/common/quantization.h"
#include <cuda_runtime_api.h>

namespace tk = tensorrt_llm::common;
namespace tkc = tensorrt_llm::cutlass_extensions;

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

/*
  This runner supports:
  int8_t inputs (A and B)
  float alpha scalings (either per-col, or per-col x per-row)
  T output (D) where T = {float, half, __nv_bfloat16} // TODO(mseznec)

  Activations, biases, scales and outputs are all assumed to be row-major.
  Weights are assumed to be column-major.
*/

using perfMapType = std::unordered_map<int, tkc::CutlassGemmConfig>;

class CutlassInt8GemmRunnerInterface
{
public:
    CutlassInt8GemmRunnerInterface() {}

    virtual ~CutlassInt8GemmRunnerInterface() {}

    virtual void gemm(const int8_t* A, const int8_t* B, tk::QuantMode quantOption, const float* alphaCol,
        const float* alphaRow, void* C, int m, int n, int k, char* workspacePtr, const size_t workspaceBytes,
        cudaStream_t stream)
        = 0;

    virtual void profileGemms(tk::QuantMode quantOption, int minM, int maxM, int n, int k, int8_t* A, int8_t* B,
        void* C, float* alphaCol, float* alphaRow, char* workspace)
        = 0;

    // Returns desired workspace size in bytes.
    virtual int getWorkspaceSize(const int m, const int n, const int k) = 0;

    // Returns True if tactics has already been selected
    bool hasSelectedTactics() const
    {
        return mTacticsMap.size() > 0;
    }

    void setSelectedTactics(const perfMapType& tacticsMap)
    {
        mTacticsMap = tacticsMap;
    }

    const perfMapType& getSelectedTactics() const
    {
        return mTacticsMap;
    }

    void setMaxM(int maxM)
    {
        mMaxM = maxM;
    }

    int getMaxM() const
    {
        return mMaxM;
    }

protected:
    static constexpr int SPLIT_K_LIMIT = 7;
    static constexpr int MAX_STEP_M = 32768;
    static constexpr int MIN_M_TILE = 32;
    static constexpr int MIN_N_TILE = 64;

    int mMaxM;

    perfMapType mTacticsMap;
};

template <typename T>
class CutlassInt8GemmRunner : public virtual CutlassInt8GemmRunnerInterface
{
public:
    CutlassInt8GemmRunner();
    ~CutlassInt8GemmRunner();

    void gemm(const int8_t* A, const int8_t* B, tk::QuantMode quantOption, const float* alphaCol, const float* alphaRow,
        void* C, int m, int n, int k, char* workspacePtr, const size_t workspaceBytes, cudaStream_t stream) override;

    void profileGemms(tk::QuantMode quantOption, int minM, int maxM, int n, int k, int8_t* A, int8_t* B, void* C,
        float* alphaCol, float* alphaRow, char* workspace) override;

    // Returns desired workspace size in bytes.
    int getWorkspaceSize(const int m, const int n, const int k) override;

private:
    void dispatchToArch(const int8_t* A, const int8_t* B, tk::QuantMode quantOption, const float* alphaCol,
        const float* alphaRow, T* C, int m, int n, int k, tkc::CutlassGemmConfig gemmConfig, char* workspacePtr,
        const size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr);

    tkc::CutlassGemmConfig profileGemm(tk::QuantMode quant_option, int m, int n, int k, int8_t* A, int8_t* B, void* C,
        float* alphaCol, float* alphaRow, char* workspace);

    float profileConfig(const tkc::CutlassGemmConfig& config, tk::QuantMode quantOption, int m, int n, int k, int8_t* A,
        int8_t* B, void* C, float* alphaCol, float* alphaRow, char* workspace);

    int mSm;
    int mMultiProcessorCount;
};

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
