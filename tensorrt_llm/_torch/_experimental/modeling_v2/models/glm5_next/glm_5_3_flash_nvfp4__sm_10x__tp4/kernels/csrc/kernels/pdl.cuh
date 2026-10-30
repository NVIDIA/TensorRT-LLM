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
// Programmatic dependent launch (PDL): a kernel launched with launch_pdl may start while the
// previous kernel in the stream is still running. Every such kernel must call pdl_entry()
// before touching global memory: it lets the next kernel launch early and then waits for the
// previous grid to complete (which, because every kernel waits, also orders all earlier ones).
#pragma once

#include <cuda_runtime.h>

#include <utility>

#include "../common.h"

namespace glm {
namespace k {

__device__ __forceinline__ void pdl_entry() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}

// Split form for kernels that first touch only read-only data (weights): pdl_trigger() first
// thing, then e.g. weight prefetches, then pdl_wait() before reading anything an earlier kernel
// writes.
__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void prefetch_l2(const void* p) {
  asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p));
}

template <typename... KArgs, typename... Args>
inline void launch_pdl(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem,
                       cudaStream_t st, Args&&... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...));
}

// launch_pdl for a kernel launched in clusters of cluster_x CTAs along x (no __cluster_dims__)
template <typename... KArgs, typename... Args>
inline void launch_pdl_cluster(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem,
                               cudaStream_t st, int cluster_x, Args&&... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  attr[1].id = cudaLaunchAttributeClusterDimension;
  attr[1].val.clusterDim.x = cluster_x;
  attr[1].val.clusterDim.y = 1;
  attr[1].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 2;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...));
}

}  // namespace k
}  // namespace glm
