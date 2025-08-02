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

#include "cutlass_extensions/gemm_configs.h"
#include "cutlass_extensions/weight_only_quant_op.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "tensorrt_llm/thop/thUtils.h"
#include "tensorrt_llm/thop/userbuffersTensor.h"

#include <torch/extension.h>

using namespace tensorrt_llm::kernels::cutlass_kernels;
using namespace tensorrt_llm::kernels;

namespace torch_ext
{
using WeightOnlyQuantGemmRunnerPtr = std::shared_ptr<CutlassFpAIntBGemmRunnerInterface>;

class WeightOnlyQuantGemmRunner : public torch::CustomClassHolder
{
public:
    explicit WeightOnlyQuantGemmRunner(at::ScalarType activation_dtype, at::ScalarType weight_dtype);

    at::Tensor runGemm(at::Tensor const& mat_a, at::Tensor const& mat_b, at::Tensor const& weight_scales,
        int64_t config_idx, bool to_userbuffers, std::optional<c10::ScalarType> out_dtype);

    int64_t getNumConfigs() const;

private:
    WeightOnlyQuantGemmRunnerPtr mGemmRunner;
    at::ScalarType mActivationDtype;
    at::ScalarType mWeightDtype;
    std::vector<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> mConfigs;
};

} // namespace torch_ext
