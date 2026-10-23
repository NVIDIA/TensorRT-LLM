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

#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

enum class ScaleAddDtype
{
    kFloat32,
    kBfloat16,
};

//! out[i] = x[i] * alpha + beta over `numel` contiguous elements of `dtype`. The product and the sum are each
//! rounded to fp32 (no fused multiply-add), then the result is rounded once to `dtype`, which is what
//! `(x.float() * alpha + beta).to(dtype)` computes in torch.
cudaError_t invokeScaleAdd(
    void const* x, void* out, int64_t numel, ScaleAddDtype dtype, float alpha, float beta, cudaStream_t stream);

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
