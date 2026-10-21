/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3Sm107Tp4
{

//! The overlap scheduler's speculative step correction in one launch: pos[i] += posOff[i] for i < nPos and
//! kv[j] += kvOff[j] for j < nKv (int32, wrapping as the ATen adds do). A plain launch: it is a step-input writer that
//! pre-wait readers rely on (rubin-20261015/PREWAIT_AUDIT.md item b).
cudaError_t invokeSpecOffsets(int32_t* pos, int32_t const* posOff, int64_t nPos, int32_t* kv, int32_t const* kvOff,
    int64_t nKv, cudaStream_t stream);

} // namespace minimaxM3Sm107Tp4
} // namespace kernels

TRTLLM_NAMESPACE_END
