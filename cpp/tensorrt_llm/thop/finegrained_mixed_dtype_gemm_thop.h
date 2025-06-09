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
#include "cutlass_extensions/weight_only_quant_op.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include <torch/extension.h>

namespace torch_ext
{

class finegrainedMixedDtypeGemmRunner : public torch::CustomClassHolder
{
public:
    explicit finegrainedMixedDtypeGemmRunner(
        at::ScalarType activationDtype, at::ScalarType outputDtype, int64_t quant_mode = 0);

    at::Tensor runGemm(at::Tensor const& A, at::Tensor const& B_packed, at::Tensor const& scales,
        int64_t group_size_long, int64_t configIdx = -1, std::optional<at::Tensor> bias = std::nullopt,
        std::optional<at::Tensor> zeros = std::nullopt, double alpha = 1.0f) const;

    int64_t getNumConfigs() const;

private:
    std::shared_ptr<tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface> mGemmRunner;
    std::vector<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> mConfigs;
    at::ScalarType mActivationDtype;
    at::ScalarType mOutputDtype;
};

} // namespace torch_ext
