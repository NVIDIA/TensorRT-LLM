/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/config.h"

#include <torch/extension.h>

#include <string>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

void configureFp8BlockScalingGemmDispatch(std::string const& deepGemmVersion);

std::string fp8BlockScalingGemmRuntimeBuildId();

bool shouldUseDeepGemm(torch::Tensor const& mat1, torch::Tensor const& mat2, torch::Tensor const& mat1Scale,
    torch::Tensor const& mat2Scale);

} // namespace torch_ext

TRTLLM_NAMESPACE_END
