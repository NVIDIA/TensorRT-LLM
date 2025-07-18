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
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/xqaParams.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunner.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace jit
{

bool supportConfigQGMMA(XQAParams const& xqaParams, int SM, bool forConfigurePlugin);
bool supportConfigHMMA(XQAParams const& xqaParams, int SM, bool forConfigurePlugin);
bool supportConfigMLA(XQAParams const& xqaParams, int SM, bool forConfigurePlugin);
bool supportConfigTllmGen(
    XQAParams const& xqaParams, int SM, bool forConfigurePlugin, TllmGenFmhaRunner const* tllmRunner);

} // namespace jit
} // namespace kernels
} // namespace tensorrt_llm
