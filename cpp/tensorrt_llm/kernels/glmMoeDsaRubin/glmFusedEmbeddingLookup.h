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
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{

//! Embedding lookup of an unsharded BF16 table with the out-of-range guard of the Embedding module:
//!   out[t, :] = (0 <= ids[t] < vocabSize) ? table[ids[t], :] : 0
//!
//! One thread copies one 16-byte vector of a row; the grid covers numTokens x (hiddenSize / 8) vectors.
//! Under PDL the kernel waits for its producer before reading the ids and then lets its dependents launch.
//!
//! \param out         Output [numTokens, hiddenSize], row-major, 16-byte aligned.
//! \param ids         Token ids [numTokens], int64 when idsAreInt64, else int32.
//! \param table       Embedding table [vocabSize, hiddenSize], row-major, 16-byte aligned.
//! \param hiddenSize  Row length in elements; must be a multiple of 8.
void invokeFusedEmbeddingLookup(__nv_bfloat16* out, void const* ids, bool idsAreInt64, __nv_bfloat16 const* table,
    int64_t numTokens, int64_t vocabSize, int64_t hiddenSize, cudaStream_t stream);

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
