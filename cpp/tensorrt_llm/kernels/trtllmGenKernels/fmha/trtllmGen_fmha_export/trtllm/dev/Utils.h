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

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector_types.h>
#include <device_types.h>

#include <cute/tensor.hpp>
#include <cutlass/gemm/kernel/tile_scheduler_params.h>

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> inline __device__ T clamp(T x, T lb, T ub) {
  return (x < lb) ? lb : (x > ub ? ub : x);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void completeTransaction(uint64_t* barrier,
                                           uint32_t transactionBytes,
                                           uint32_t pred = 1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "setp.eq.u32 p, %2, 1;\n\t"
               "@p mbarrier.complete_tx.shared::cta.relaxed.cta.b64   [%1], %0;"
               "}"
               :
               : "r"(transactionBytes),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(barrier))),
                 "r"(pred));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __device__ void cp(T* dst,
                          T const* src,
                          int32_t dstOffset = 0,
                          int64_t srcOffset = 0,
                          const int cpSize = sizeof(T)) {
  if (cpSize == 4) {
    uint32_t* dstUInt32 = reinterpret_cast<uint32_t*>(dst + dstOffset);
    uint32_t const* srcUInt32 = reinterpret_cast<uint32_t const*>(src + srcOffset);
    *dstUInt32 = *srcUInt32;
  } else if (cpSize == 8) {
    uint64_t* dstUInt64 = reinterpret_cast<uint64_t*>(dst + dstOffset);
    uint64_t const* srcUInt64 = reinterpret_cast<uint64_t const*>(src + srcOffset);
    *dstUInt64 = *srcUInt64;
  } else if (cpSize == 16) {
    uint4* dstUInt128 = reinterpret_cast<uint4*>(dst + dstOffset);
    uint4 const* srcUInt128 = reinterpret_cast<uint4 const*>(src + srcOffset);
    *dstUInt128 = *srcUInt128;
  } else {
    assert(0 && "cpSize is not supported"); // The compiler will eliminate that code.
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __device__ void cpAsync(T* dst,
                               T const* src,
                               int32_t dstOffset = 0,
                               int64_t srcOffset = 0,
                               const int cpSize = sizeof(T)) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  if (cpSize == 4) {
    uint32_t* dstUInt32 = reinterpret_cast<uint32_t*>(dst + dstOffset);
    uint32_t const* srcUInt32 = reinterpret_cast<uint32_t const*>(src + srcOffset);
    uint32_t dstU32 = static_cast<uint32_t>(__cvta_generic_to_shared(dstUInt32));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dstU32), "l"(srcUInt32));
  } else if (cpSize == 8) {
    uint64_t* dstUInt64 = reinterpret_cast<uint64_t*>(dst + dstOffset);
    uint64_t const* srcUInt64 = reinterpret_cast<uint64_t const*>(src + srcOffset);
    uint32_t dstU32 = static_cast<uint32_t>(__cvta_generic_to_shared(dstUInt64));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(dstU32), "l"(srcUInt64));
  } else if (cpSize == 16) {
    uint4* dstUInt128 = reinterpret_cast<uint4*>(dst + dstOffset);
    uint4 const* srcUInt128 = reinterpret_cast<uint4 const*>(src + srcOffset);
    uint32_t dstU32 = static_cast<uint32_t>(__cvta_generic_to_shared(dstUInt128));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(dstU32), "l"(srcUInt128));
  } else {
    assert(0 && "cpSize is not supported"); // The compiler will eliminate that code.
  }
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldgInt2(int regs[2], int8_t const* src, int64_t srcOffset) {
  int2 i2 = *reinterpret_cast<int2 const*>(src + srcOffset);
  regs[0] = i2.x;
  regs[1] = i2.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldgInt4(int regs[4], int8_t const* src, int64_t srcOffset) {
  int4 i4 = *reinterpret_cast<int4 const*>(src + srcOffset);
  regs[0] = i4.x;
  regs[1] = i4.y;
  regs[2] = i4.z;
  regs[3] = i4.w;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldgInt8(int regs[8], int8_t const* src, int64_t srcOffset) {
  asm("ld.ca.global.v8.s32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8+0];"
      : "=r"(regs[0]),
        "=r"(regs[1]),
        "=r"(regs[2]),
        "=r"(regs[3]),
        "=r"(regs[4]),
        "=r"(regs[5]),
        "=r"(regs[6]),
        "=r"(regs[7])
      : "l"(src + srcOffset));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void cpAsyncCommitGroup() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N> inline __device__ void cpAsyncWaitGroup() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////


inline __device__ uint32_t mulBf16x2(uint32_t a, uint32_t b) {
  uint32_t c;
  asm("mul.bf16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert packed half2 (represented as uint32_t) to 2xe4m3.
inline __device__ uint16_t convert_half2_to_e4m3(uint32_t packed_h2) {

  // The output bits.
  uint16_t output_bits;
  // The ptx instruction to convert packed half2 to 2xe4m3.
  asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1 ;" : "=h"(output_bits) : "r"(packed_h2));
  return output_bits;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert packed half4 (represented as 2xuint32_t) to 4xe4m3 (packed in
// uint32_t).
inline __device__ uint32_t convert_half4_to_e4m3(uint32_t a, uint32_t b) {

  // The output bits.
  union {
    uint32_t output_bits;
    uint16_t output_bits_16[2];
  } output;
  // The ptx instruction to convert packed half2 to 2xe4m3.
  asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %2 ;\n"
               "cvt.rn.satfinite.e4m3x2.f16x2 %1, %3 ;\n"
               : "=h"(output.output_bits_16[0]), "=h"(output.output_bits_16[1])
               : "r"(a), "r"(b));
  return output.output_bits;
}

////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert bfloat16x2 to float and add them.
inline __device__ float fadd2_bf16(uint32_t x) {
  // BF16 to FP32: shift left by 16 (zero-pad mantissa)
  float lo = __int_as_float((x & 0xFFFFu) << 16);
  float hi = __int_as_float(x & 0xFFFF0000u);
  return lo + hi;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert half2 to float and add them.
inline __device__ float fadd2_fp16(uint32_t x) {
  float2 result = __half22float2(reinterpret_cast<__half2 const&>(x));
  return result.x + result.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float phi(float x) {
  return 0.5f * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float gelu(float x) {
  return x * phi(x);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to add bfloat16x2 values (reinterpreted as uint32_t).
inline __device__ uint32_t hadd2_bf16(uint32_t inA, uint32_t inB) {
  uint32_t output;
  asm volatile("{\n"
               "add.rn.bf16x2 %0, %1, %2;\n"
               "}\n"
               : "=r"(output)
               : "r"(inA), "r"(inB));
  return output;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to add half2 values (reinterpreted as uint32_t).
inline __device__ uint32_t hadd2_fp16(uint32_t inA, uint32_t inB) {
  uint32_t output;
  asm volatile("{\n"
               "add.rn.f16x2 %0, %1, %2;\n"
               "}\n"
               : "=r"(output)
               : "r"(inA), "r"(inB));
  return output;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> inline __device__ void prmt(uint32_t& out, T hi, T lo, uint32_t selectCode) {
  out = __byte_perm(reinterpret_cast<uint32_t&>(lo), reinterpret_cast<uint32_t&>(hi), selectCode);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float relu(float x) {
  return fmaxf(0.0f, x);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sigmoid_base2(float x) {
  return 1.0f / (1.0f + exp2f(-x));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float silu(float x) {
  return x * sigmoid(x);
}

////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void prefetchTensorMap(void const* ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("prefetch.tensormap [%0];" : : "l"(reinterpret_cast<uint64_t>(ptr)) : "memory");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void prefetchTensor3dL2(void const* desc_ptr,
                                          int32_t crd0,
                                          int32_t crd1,
                                          int32_t crd2) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // The .tile prefetch variant performed better for BatchedGemm prefetchB tuning.
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];"
               :
               : "l"(gmem_int_desc), "r"(crd0), "r"(crd1), "r"(crd2)
               : "memory");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void prefetchTensor4dL2(void const* desc_ptr,
                                          int32_t crd0,
                                          int32_t crd1,
                                          int32_t crd2,
                                          int32_t crd3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // The .tile prefetch variant performed better for BatchedGemm prefetchB tuning.
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];"
               :
               : "l"(gmem_int_desc), "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3)
               : "memory");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Initialize PersistentTileSchedulerSm90::Params on device for StaticPersistent scheduler.
//
// Note: This function is called on device during resource initialization. Basically copied from
// PersistentTileSchedulerSm90Params::initialize() but modified to become a device function and
// pinned raster/swizzle configuration to AlongM/Disabled only.
//
// TODO: overload the function with gemm::KernelParams and batchedGemm::KernelParams.
template <bool BatchM, typename KernelParams, typename ClusterShape, typename TileShape>
inline __device__ cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params
initPersistentSchedulerSm90Params(KernelParams const& kernelParams,
                                  ClusterShape clusterShape,
                                  TileShape /* tileShape */) {
  //  BatchM and  RasterAlongNonBatchDim implies RasterAlongN
  //  BatchM and !RasterAlongNonBatchDim implies RasterAlongM
  // !BatchM and  RasterAlongNonBatchDim implies RasterAlongM
  // !BatchM and !RasterAlongNonBatchDim implies RasterAlongN
  bool constexpr RasterAlongM{true};
  using Params = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params;
  using RasterOrder = typename Params::RasterOrder;
  Params schedulerParams{};

  int32_t const clusterM = cute::get<0>(clusterShape);
  int32_t const clusterN = cute::get<1>(clusterShape);

  // Calculate number of tiles based on batch mode.
  // - nm is the non-batched dimension: N if batchM, M if batchN.
  // - tileStridePerBatch is: N/TileN if batchM, M/TileM if batchN.
  // - ptrNumNonExitingCtas[0] gives the max number of tiles in the batched dimension. The legacy
  //   CTA-based name is kept for interface compatibility even though routed MoE paths may provide
  //   a CGA-granular count here.
  //
  // Note: For StaticPersistent scheduler, ptrNumNonExitingCtas MUST be set because we cannot
  // compute the tile count for the batched dimension from nm alone (nm is the OTHER dimension).
  int32_t numTilesM;
  int32_t numTilesN;
  if constexpr (BatchM) {
    // Batching along M: nm = N (fixed), numTilesN is known, numTilesM from early exit info
    numTilesN = kernelParams.tileStridePerBatch; // N / TileN
    // ptrNumNonExitingCtas[0] gives the max number of M tiles across all batches
    numTilesM = kernelParams.ptrNumNonExitingCtas[int32_t{0}];
  } else {
    // Batching along N: nm = M (fixed), numTilesM is known, numTilesN from early exit info
    numTilesM = kernelParams.tileStridePerBatch; // M / TileM
    // ptrNumNonExitingCtas[0] gives the max number of N tiles across all batches
    numTilesN = kernelParams.ptrNumNonExitingCtas[int32_t{0}];
  }

  // We map the ragged batch dim to M if BatchM==true else N. Hence always 1 in classic BMM sense.
  int32_t const numBatches = 1;

  schedulerParams.problem_tiles_m_ = numTilesM;
  schedulerParams.problem_tiles_n_ = numTilesN;
  schedulerParams.problem_tiles_l_ = numBatches;
  schedulerParams.cluster_shape_m_ = clusterM;
  schedulerParams.cluster_shape_n_ = clusterN;
  schedulerParams.log_swizzle_size_ = 0;
  // Raster order: AlongM if transposeMmaOutput==true, else AlongN.
  // TODO: check if raster order is compatible with how TRTLLM-Gen sets grid shape in x or y.
  schedulerParams.raster_order_ = RasterAlongM ? RasterOrder::AlongM : RasterOrder::AlongN;

  uint64_t const blocksPerProblem =
    uint64_t(numTilesM) * uint64_t(numTilesN) * uint64_t(numBatches);
  schedulerParams.blocks_per_problem_ = blocksPerProblem;

  // Use power of 2 variant for cluster shapes since they are typically 1, 2, or 4
  if constexpr (RasterAlongM) {
    schedulerParams.divmod_cluster_shape_major_ = cutlass::FastDivmodU64Pow2(clusterM);
    schedulerParams.divmod_cluster_shape_minor_ = cutlass::FastDivmodU64Pow2(clusterN);
  } else {
    schedulerParams.divmod_cluster_shape_major_ = cutlass::FastDivmodU64Pow2(clusterN);
    schedulerParams.divmod_cluster_shape_minor_ = cutlass::FastDivmodU64Pow2(clusterM);
  }

  uint64_t const tilesPerBatch = uint64_t(numTilesM) * uint64_t(numTilesN);
  schedulerParams.divmod_batch_ = cutlass::FastDivmodU64(tilesPerBatch);

  if constexpr (RasterAlongM) {
    uint64_t const clustersAlongMajor = (numTilesM + clusterM - 1) / clusterM;
    schedulerParams.divmod_cluster_blk_major_ = cutlass::FastDivmodU64(clustersAlongMajor);
  } else {
    uint64_t const clustersAlongMajor = (numTilesN + clusterN - 1) / clusterN;
    schedulerParams.divmod_cluster_blk_major_ = cutlass::FastDivmodU64(clustersAlongMajor);
  }

  return schedulerParams;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace dev
} // namespace trtllm
