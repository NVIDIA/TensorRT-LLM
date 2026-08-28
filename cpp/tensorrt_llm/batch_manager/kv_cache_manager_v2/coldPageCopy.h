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

#include "coldPageCodec.h"

#include <cuda.h>

#include <cstddef>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2::detail
{

//! Copies ephemeral host page indices into device memory while leaving the destination update asynchronous.
void copyPageIndicesToDevice(CUdeviceptr dst, PageIndexPair const* src, size_t numPageIndices, CUstream stream);

//! Enqueues independent data copies in stream order.
void copyColdPageDataBatch(CUdeviceptr* dsts, CUdeviceptr* srcs, size_t* sizes, size_t count, CUstream stream);

#if CUDA_VERSION < 12080
//! Kernel fallback used when cuMemcpyBatchAsync is unavailable. Exposed for focused testing.
void copyPageIndicesToDeviceWithKernel(
    CUdeviceptr dst, PageIndexPair const* src, size_t numPageIndices, CUstream stream);

//! Per-copy fallback used when cuMemcpyBatchAsync is unavailable. Exposed for focused testing.
void copyColdPageDataBatchWithMemcpyAsync(
    CUdeviceptr* dsts, CUdeviceptr* srcs, size_t* sizes, size_t count, CUstream stream);
#endif

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2::detail
