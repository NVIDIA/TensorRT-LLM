/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 *
 * Adapted from Baseten's sa_spec library (Apache-2.0)
 * https://github.com/basetenlabs/sa_spec
 */

#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define SA_CUDA_CALLABLE __host__ __device__ __forceinline__
#else
#define SA_CUDA_CALLABLE
// Provide a placeholder type for cudaStream_t when not compiling with CUDA.
// Only define if not already defined to avoid conflicts with cuda_runtime_api.h.
#if !defined(cudaStream_t)
#define cudaStream_t int
#endif
#endif // __CUDACC__
