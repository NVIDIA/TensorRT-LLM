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

#include <cutlass/cutlass.h>

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// CutlassNamedBarrier class.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

// This class is a modified copy of cutlass::arch::NamedBarrier in order to
// overwrite ReservedNamedBarrierCount to 0.
class CutlassNamedBarrier {
  // Data Members:

  // Range = [1 , NUM_THREADS_PER_CTA]
  // Range % warp-size (i.e 32) == 0
  uint32_t const num_threads_;

  // Range : [0, 15]
  // Note that should be set to the final barrier ID, including ReserveNamedBarrierCount should be
  // considered
  uint32_t const id_;

public:
  // Constructor for CUTLASS users:
  // effective barrier ID starts from ReservedNamedBarrierCount
  CUTLASS_DEVICE
  CutlassNamedBarrier(uint32_t num_threads, uint32_t id = 0)
    : num_threads_(num_threads)
    , id_(id + ReservedNamedBarrierCount) {
    CUTLASS_ASSERT(id + ReservedNamedBarrierCount <= HardwareMaxNumNamedBarriers &&
                   "Effective barrier_id should not exceed 16.");
  }

  CUTLASS_DEVICE
  void arrive_and_wait() const {
    // Note: The value of id_ is already the final barrier id (set correctly in the constructor).
    CutlassNamedBarrier::arrive_and_wait_internal(num_threads_, id_);
  }

  CUTLASS_DEVICE
  void arrive_and_wait_unaligned() const {
    // Note: The value of id_ is already the final barrier id (set correctly in the constructor).
    CutlassNamedBarrier::arrive_and_wait_internal_unaligned(num_threads_, id_);
  }

  CUTLASS_DEVICE
  void arrive() const {
    // Note: The value of id_ is already the final barrier id (set correctly in the constructor).
    CutlassNamedBarrier::arrive_internal(num_threads_, id_);
  }

  CUTLASS_DEVICE
  void arrive_unaligned() const {
    // Note: The value of id_ is already the final barrier id (set correctly in the constructor).
    CutlassNamedBarrier::arrive_internal_unaligned(num_threads_, id_);
  }

  CUTLASS_DEVICE
  void sync() const { CutlassNamedBarrier::arrive_and_wait(); }

  //  Static variants

  // Calling interface for CUTLASS users:
  // effective barrier ID starts from ReservedNamedBarrierCount
  CUTLASS_DEVICE
  static void arrive_and_wait(uint32_t num_threads, uint32_t barrier_id) {
    arrive_and_wait_internal(num_threads, barrier_id + ReservedNamedBarrierCount);
  }

  // Calling interface for CUTLASS users:
  // effective barrier ID starts from ReservedNamedBarrierCount
  CUTLASS_DEVICE
  static void arrive(uint32_t num_threads, uint32_t barrier_id) {
    arrive_internal(num_threads, barrier_id + ReservedNamedBarrierCount);
  }

  // Calling interface for CUTLASS users:
  // effective barrier ID starts from ReservedNamedBarrierCount
  CUTLASS_DEVICE
  static void sync(uint32_t num_threads, uint32_t barrier_id) {
    sync_internal(num_threads, barrier_id + ReservedNamedBarrierCount);
  }

private:
  CUTLASS_DEVICE
  static void arrive_and_wait_internal(uint32_t num_threads, uint32_t barrier_id) {
#if CUDA_BARRIER_ENABLED
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
    cutlass::arch::synclog_emit_named_barrier_arrive_and_wait(__LINE__, num_threads, barrier_id);
#elif defined(__CUDA_ARCH__)
    asm volatile("brkpt;\n" ::);
#endif
  }

  CUTLASS_DEVICE
  static void arrive_and_wait_internal_unaligned(uint32_t num_threads, uint32_t barrier_id) {
#if CUDA_BARRIER_ENABLED
    asm volatile("barrier.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
    cutlass::arch::synclog_emit_named_barrier_arrive_and_wait(__LINE__, num_threads, barrier_id);
#elif defined(__CUDA_ARCH__)
    asm volatile("brkpt;\n" ::);
#endif
  }

  CUTLASS_DEVICE
  static void arrive_internal(uint32_t num_threads, uint32_t barrier_id) {
#if CUDA_BARRIER_ENABLED
    cutlass::arch::synclog_emit_named_barrier_arrive(__LINE__, num_threads, barrier_id);
    asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
#elif defined(__CUDA_ARCH__)
    asm volatile("brkpt;\n" ::);
#endif
  }

  CUTLASS_DEVICE
  static void arrive_internal_unaligned(uint32_t num_threads, uint32_t barrier_id) {
#if CUDA_BARRIER_ENABLED
    cutlass::arch::synclog_emit_named_barrier_arrive(__LINE__, num_threads, barrier_id);
    asm volatile("barrier.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
#elif defined(__CUDA_ARCH__)
    asm volatile("brkpt;\n" ::);
#endif
  }

  CUTLASS_DEVICE
  static void sync_internal(uint32_t num_threads, uint32_t barrier_id) {
    CutlassNamedBarrier::arrive_and_wait_internal(num_threads, barrier_id);
  }

public:
  // We do not reserve any NamedBarriers for CUTLASS' own use cases.
  static const uint32_t ReservedNamedBarrierCount = 0;
  static const uint32_t HardwareMaxNumNamedBarriers = 16;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace dev
} // namespace trtllm
