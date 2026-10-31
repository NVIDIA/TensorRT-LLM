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
namespace minimaxM3Sm107Tp8
{

//! The decode graph's step-input prologue in one launch (step_inputs.StepInputPack with early_copy): the work of the
//! two parallel scatter launches and the header clear that follow them, in that order.
//!
//! The window is the pack's staging layout (kernels/stepInputsScatter.h): an int64 header `[numHostCopies,
//! numDeviceCopies]`, then one `(dst, src, bytes)` int64 triple per copy, host copies first; a host copy's `src` is a
//! byte offset into the window, a device copy's a device address. Up to `maxDesc` copies of each kind are written (the
//! slots the parallel scatter's grid covers), every host copy before any device copy, so a device copy may read a
//! destination a host copy filled; the copies of one kind must not overlap (the pack issues the window only then).
//! Afterwards the header is zeroed, so a replay with no new upload writes nothing. With `epoch` non-null, it is
//! incremented once.
cudaError_t invokeStepPrologue(uint8_t* window, int64_t* epoch, int64_t maxDesc, cudaStream_t stream);

} // namespace minimaxM3Sm107Tp8
} // namespace kernels

TRTLLM_NAMESPACE_END
