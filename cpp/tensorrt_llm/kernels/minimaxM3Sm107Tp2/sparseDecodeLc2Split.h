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
#include "tensorrt_llm/kernels/minimaxM3Sm107Tp2/sparseDecodeLc2.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

//! Workspace bytes of LC2-split for up to `maxTokens` query rows: the fp32 partial rows and LSEs of every (row, KV
//! head, head, slot).
size_t minimaxM3Sm107Tp2SparseDecodeLcSplitWorkspaceBytes(int32_t maxTokens, int32_t numKvHeads);

//! LC2-split: the LC2 sparse decode (same operands, bit-identical outputs) without the thread-block cluster. A compute
//! grid writes each (row, slot) CTA's normalized partial and LSE to `workspace`; a merge grid combines the 16 partials
//! of each head with LC2's weights in slot order and writes the BF16 row and, when mxfp8, the MXFP8 copy. prewait and
//! N1 behave as in LC2 (hints; they do not change a value).
cudaError_t invokeMinimaxM3Sm107Tp2SparseDecodeLcSplit(MinimaxM3Sm107Tp2SparseDecodeLcParams const& params,
    int32_t numTokens, bool linearSoftmax, bool mxfp8, bool prewait, bool usePdl, void* workspace,
    size_t workspaceBytes, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
