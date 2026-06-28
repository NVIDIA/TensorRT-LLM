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

#include <cstdint>
#include <cuda_runtime_api.h>

namespace tensorrt_llm::executor::kv_cache::bounce
{

/// Batched device-to-device copy: for each of `n` buffers, copy `sizes[i]` bytes from
/// `srcs[i]` to `dsts[i]`. `srcs`, `dsts`, `sizes` are DEVICE pointers to arrays of length `n`.
///
/// Direction-agnostic: used for gather (srcs=scattered, dsts=slot+offset) and scatter
/// (srcs=slot+offset, dsts=scattered) — only the pointer arrays differ. 16-byte-aligned
/// buffers take a vectorized uint4 path; otherwise a byte path.
[[nodiscard]] cudaError_t launchBatchedCopy(std::uint64_t const* srcs, std::uint64_t const* dsts,
    std::uint32_t const* sizes, std::uint32_t n, cudaStream_t stream);

} // namespace tensorrt_llm::executor::kv_cache::bounce
