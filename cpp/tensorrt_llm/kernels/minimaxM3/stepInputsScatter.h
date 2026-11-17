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

//! Write a packed batch of byte copies described by a staging buffer.
//!
//! The staging buffer starts with an int64 header `[numHostCopies, numDeviceCopies]`, followed by one
//! `(dst, src, bytes)` int64 triple per copy. A host copy's `src` is a byte offset into the staging buffer (its
//! bytes were uploaded with the header); a device copy's `src` is a device address. Every host copy is written
//! before any device copy, so a device copy may read a destination a host copy of the same batch filled.
void invokeStepInputsScatter(uint8_t const* stage, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
