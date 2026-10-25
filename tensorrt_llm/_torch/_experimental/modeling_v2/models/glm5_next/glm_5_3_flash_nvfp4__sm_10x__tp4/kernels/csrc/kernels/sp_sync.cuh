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
// System-scope flag primitives of the sequence-parallel peer-memory collectives (sp_comm.cu and
// the boundary kernels in elementwise.cu that all-gather as they go).
#pragma once

#include <cstdint>

namespace glm {
namespace k {

__device__ __forceinline__ void sp_st_release_sys(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t sp_ld_acquire_sys(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void sp_red_release_sys_add(uint32_t* p, uint32_t v) {
  asm volatile("red.release.sys.global.add.u32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}

}  // namespace k
}  // namespace glm
