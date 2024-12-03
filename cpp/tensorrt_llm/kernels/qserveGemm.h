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

namespace tensorrt_llm
{
namespace kernels
{
namespace qserve
{

struct ParamsPerGroup
{
    int8_t const* A;
    int8_t const* B;
    int8_t const* s2_zeros;
    int8_t const* s2_scales;
    half const* s1_scales;
    half const* act_scales;
    half* C;
    int m;
    int n;
    int k;
};

struct ParamsPerChannel
{
    int8_t const* A;
    int8_t const* B;
    half const* s1_scales;
    half const* s1_szeros;
    half const* act_sums;
    half const* act_scales;
    half* C;
    int m;
    int n;
    int k;
};

class QServeGemmRunner
{
public:
    void gemmPerGroup(ParamsPerGroup const& params, cudaStream_t stream);
    void gemmPerChannel(ParamsPerChannel const& params, cudaStream_t stream);

    // We do not use workspace for now.
    // char* workspacePtr, const size_t workspaceBytes, cudaStream_t stream);

    // Returns desired workspace size in bytes.
    size_t getWorkspaceSize(int const m, int const n, int const k);

    // virtual std::vector<tkc::CutlassGemmConfig> getConfigs() const = 0;
};

} // namespace qserve
} // namespace kernels
} // namespace tensorrt_llm
