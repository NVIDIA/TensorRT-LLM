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

// The split-KV merge for the MiniMax-M3 NVFP4 TP2 (Rubin) full-context decode. The producer is the PrimTS split-KV
// decode with its separate-reduction scratch: per (request, KV head, split, row) one normalized 16-bit partial O and
// one fp32 log2-LSE. Active splits per row follow the producer's split ranges (the causal union of the row's grouped
// Q CTA, instruction-group aligned): splits past them hold nothing and are not read.
//
// One CTA per (request, KV head, kRowsPerCta rows); per row kGroups split groups x kThreadsPerRow threads x 8 dims.
// A thread issues the loads of every split it owns (strided over the groups) before using any, folds them under its
// group's max, and the groups merge through SMEM with one max rescale; the row is normalized once and rounded to
// BF16 once.

#include "tensorrt_llm/kernels/minimaxM3Sm107/fullCtxReduce.cuh"
#include "tensorrt_llm/kernels/minimaxM3Sm107Tp2/fullCtxReduce.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3Sm107Tp2
{
namespace fullCtxReduce
{
cudaError_t invokeFullCtxReduce(Params const& params, bool pdl, cudaStream_t stream)
{
    return minimaxM3Sm107::fullCtxReduce::detail::invoke<2, false>(params, pdl, stream);
}
} // namespace fullCtxReduce
} // namespace minimaxM3Sm107Tp2
} // namespace kernels

TRTLLM_NAMESPACE_END
