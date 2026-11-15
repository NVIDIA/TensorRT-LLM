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

// Host interface of the sentinel-polled MoE all-to-all (moeAlltoAllLamport.cuh). Host translation units see the
// workspace layout and the parameter structs only; the kernels are instantiated in moeAlltoAllLamportKernels.cu.

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm
{

#include "tensorrt_llm/kernels/moe/communication/moeAlltoAllLamport.cuh"

// Launch the dispatch for top-k in {1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 22}; throws on an invalid configuration.
void moe_a2a_lamport_dispatch_launch(moe_a2a_lamport::DispatchParams const& params, int topK, cudaStream_t stream);

// Launch the combine for bf16 (isBf16) or fp16 payloads; throws on an invalid configuration.
void moe_a2a_lamport_combine_launch(
    moe_a2a_lamport::CombineParams const& params, int topK, bool isBf16, cudaStream_t stream);

} // namespace kernels::moe_comm

TRTLLM_NAMESPACE_END
