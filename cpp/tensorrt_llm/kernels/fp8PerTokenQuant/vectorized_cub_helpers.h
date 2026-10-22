/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

// Adapted from vLLM (Apache-2.0):
// https://github.com/vllm-project/vllm/blob/v0.25.0/csrc/libtorch_stable/cub_helpers.h

#pragma once

#include <cub/cub.cuh>
#if CUB_VERSION >= 200800
#include <cuda/std/functional>
using CubAddOp = cuda::std::plus<>;
using CubMaxOp = cuda::maximum<>;
#else
using CubAddOp = cub::Sum;
using CubMaxOp = cub::Max;
#endif
