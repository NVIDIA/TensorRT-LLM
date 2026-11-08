/*
 * Copyright (c) 2011-2026, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "cuda_runtime_api.h"
#include <cstdint>

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// Helper function to manipulate the Smem descriptors for MMAs.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

union SmemDesc {
  uint64_t u64;
  uint32_t u32[2];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __device__ uint64_t createSmemDesc(T* smemPtr, uint32_t lo, uint32_t hi) {
  // Convert the SMEM address to uint32_t.
  uint32_t mask = 0x3ffffu;
  uint32_t smemAddr = (static_cast<uint32_t>(__cvta_generic_to_shared(smemPtr)) & mask) >> 4;
  // Force the compiler to go down the URF path.
  // In some rare cases, the compiler does not think smemAddr is uniform, and generates lots of
  // conversion between URF and RF.
  smemAddr = __shfl_sync(0xffffffff, smemAddr, 0);
  // Pack the values into an uint64_t.
  SmemDesc tmp;
  tmp.u32[0] = smemAddr | lo;
  tmp.u32[1] = hi;

  // Return the uint64_t.
  return tmp.u64;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __device__ uint64_t
createSmemDesc(T* smemPtr, T* smemPtrNextBuffer, uint32_t lo, uint32_t hi) {
  uint32_t maskFull = 0x3ffffu;
  uint32_t maskNoLsb = 0x3fff0u;
  uint32_t smemAddr = (static_cast<uint32_t>(__cvta_generic_to_shared(smemPtr)) & maskFull) >> 4;
  uint32_t smemNextBufferAddr =
    (static_cast<uint32_t>(__cvta_generic_to_shared(smemPtrNextBuffer)) & maskNoLsb) << (16 - 4);

  // Pack the values into an uint64_t.
  SmemDesc tmp;
  tmp.u32[0] = smemAddr | smemNextBufferAddr;
  tmp.u32[1] = hi;

  // Return the uint64_t.
  return tmp.u64;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __device__ uint64_t
createSmemDesc(T* smemPtr, int32_t nextBufferOffsetInBytes, uint32_t lo, uint32_t hi) {
  // Get the pointer to the next buffer
  T* smemPtrNextBuffer =
    reinterpret_cast<T*>(reinterpret_cast<char*>(smemPtr) + nextBufferOffsetInBytes);
  // Get the descriptor
  return createSmemDesc(smemPtr, smemPtrNextBuffer, lo, hi);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void decrSmemAddr(uint64_t& smemDesc, uint32_t offset) {
  SmemDesc tmp;
  tmp.u64 = smemDesc;
  tmp.u32[0] -= offset;
  smemDesc = tmp.u64;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void incrSmemAddr(uint64_t& smemDesc, uint32_t offset) {
  SmemDesc tmp;
  tmp.u64 = smemDesc;
  tmp.u32[0] += offset;
  smemDesc = tmp.u64;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void decrSmemNextBufferAddr(uint64_t& smemDesc, uint32_t offset) {
  SmemDesc tmp;
  tmp.u64 = smemDesc;
  tmp.u32[0] -= (offset << 16);
  smemDesc = tmp.u64;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void incrSmemNextBufferAddr(uint64_t& smemDesc, uint32_t offset) {
  SmemDesc tmp;
  tmp.u64 = smemDesc;
  tmp.u32[0] += (offset << 16);
  smemDesc = tmp.u64;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace dev
} // namespace trtllm
