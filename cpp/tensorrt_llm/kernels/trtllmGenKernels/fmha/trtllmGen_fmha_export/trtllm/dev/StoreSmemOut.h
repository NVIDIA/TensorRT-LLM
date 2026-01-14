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

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NumRegs, typename T>
inline __device__ void storeToSmemOut(T* smemPtr,
                                      int warpGrpThreadIdx,
                                      uint32_t const (&regsOut)[NumRegs]) {

  // The number of regs cannot exceed 32 per thread (a thread writes to at most one 128B SMEM row).
  static_assert(NumRegs <= 32, "Too many registers per thread");
  // The number of registers must be a multiple of 4 as we want to use STS.128.
  static_assert(NumRegs % 4 == 0, "Must be a multiple of 4");
  // The number of registers must be a power of 2.
  static_assert((NumRegs & (NumRegs - 1)) == 0, "Must be a power of 2");

  // The number of threads per SMEM row (128B per row).
  int constexpr NumThreadsPerSmemRow = 32 / NumRegs;
  // The number of SMEM rows per swizzle block.
  int constexpr NumSmemRowsPerSwizzleBlk = 8 / NumThreadsPerSmemRow;

  // To match CuTe, we apply the XOR on the address (not just the row as in XMMA).

  // The pointer as an UINT32.
  uint32_t smemPtrU32 = static_cast<uint32_t>(__cvta_generic_to_shared(smemPtr));
  // The offset of the thread.
  uint32_t smemOffset = smemPtrU32 + warpGrpThreadIdx * NumRegs * 4 /*numBytesPerReg*/;
  // The row.
  uint32_t smemRow = smemOffset / 128 /*numBytesPerSmemRow*/;
  // The XOR mask.
  uint32_t smemXorMask = smemRow % NumSmemRowsPerSwizzleBlk * 16 /*numBytesPerXor*/;

  // Store the elements using STS.128.
#pragma unroll
  for (int ii = 0; ii < NumRegs; ii += 4) {
    // The destination pointer.
    uint32_t dstPtr = (smemOffset + ii * 4 /*numBytesPerReg*/) ^ smemXorMask;
    // Write to SMEM using STS.128.
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(dstPtr),
                 "r"(regsOut[ii + 0]),
                 "r"(regsOut[ii + 1]),
                 "r"(regsOut[ii + 2]),
                 "r"(regsOut[ii + 3]));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NumSmemRowsPerSwizzleBlk, typename T>
inline __device__ void loadFromSmemOut(uint32_t (&regsOut)[4], T* smemPtr, int warpGrpThreadIdx) {

  // To match CuTe, we apply the XOR on the address (not just the row as in XMMA).

  // The pointer as an UINT32.
  uint32_t smemPtrU32 = static_cast<uint32_t>(__cvta_generic_to_shared(smemPtr));
  // The position of the 1st element loaded using LDS.128 for each thread.
  uint32_t smemOffset = smemPtrU32 + warpGrpThreadIdx * 16 /*numBytesPerLds128*/;
  // The row.
  uint32_t smemRow = smemOffset / 128 /*numBytesPerSmemRow*/;
  // The XOR mask.
  uint32_t smemXorMask = smemRow % NumSmemRowsPerSwizzleBlk * 16 /*numBytesPerXor*/;

  // Load from SMEM using LDS.128.
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(regsOut[0]), "=r"(regsOut[1]), "=r"(regsOut[2]), "=r"(regsOut[3])
               : "r"(smemOffset ^ smemXorMask));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace dev
} // namespace trtllm
