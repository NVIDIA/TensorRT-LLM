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
#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

//! TMA bulk-copy variant of `invokeDwdpFabricGather` (dwdpFabricGatherTma.cu): same grid, descriptor
//! contract and cluster launch.
//!
//! Grid is (blocksPerRow, rows); row `y` walks descriptors `[rowStarts[y], rowStarts[y + 1])`, and
//! descriptor `d` copies `lines[d]` 16-byte lines from `srcs[d]` to `dsts[d]`.  All arrays are device
//! resident.  One elected thread per block drives a shared-memory ring, so `threads` is clamped to 128.
void invokeDwdpFabricGatherTma(void* const* dsts, void const* const* srcs, int64_t const* lines,
    int64_t const* rowStarts, int64_t rows, int64_t blocksPerRow, int64_t threads, cudaStream_t stream);

//! Whether `device` can run the TMA variant: bulk copies and cluster launch need SM90+, and the stage
//! ring needs its dynamic shared memory under the per-block opt-in limit.  Callers fall back to the
//! register gather otherwise.
bool dwdpFabricGatherTmaSupported(int device);

} // namespace kernels

TRTLLM_NAMESPACE_END
