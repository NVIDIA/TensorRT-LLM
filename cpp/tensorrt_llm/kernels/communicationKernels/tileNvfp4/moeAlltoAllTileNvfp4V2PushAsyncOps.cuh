/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

/*
 * Push-combine-owned raw-TMA async primitives for the packed-NVFP4 tileNvfp4 V2
 * family. This header is NEW and owned by the push-combine path: it exists so the
 * K1 producer-push kernel (moeAlltoAllTileNvfp4V2PushK1.cu) can publish whole
 * 4480-B packed records with a strong system-scope bulk S2G WITHOUT extending the
 * shared kernels/cudaAsyncOps.cuh header (which stays untouched).
 *
 * It reuses the family address-conversion + commit/wait-group primitives from
 * cudaAsyncOps.cuh verbatim (__as_ptr_gmem / __as_ptr_smem /
 * cp_async_bulk_commit_group / cp_async_bulk_wait_group) and adds only the two
 * push-combine-specific device helpers that the strong-scope publication protocol
 * requires. The helpers live in tensorrt_llm::kernels so K1 (in
 * tensorrt_llm::kernels::moe_comm) resolves them by the same unqualified lookup as
 * the rest of the async-ops family.
 */
#pragma once

#include "tensorrt_llm/kernels/cudaAsyncOps.cuh"

namespace tensorrt_llm
{
namespace kernels
{

// ============================================================================
// push-combine (raw TMA): strong SYSTEM-scope whole-record S2G + async-proxy fence
//
// The weak cp_async_bulk_s2g in cudaAsyncOps.cuh emits `.global.shared::cta` with
// NO ordering scope. That weak helper
// CANNOT carry the push-combine protocol: it does not order the 4480B payload write
// against the downstream K2 producer-global release, so a remote consumer that
// observes the K2 `ready` flag is not guaranteed to see the full payload. K1 must
// therefore publish each whole 4480B record with a strong system-scope bulk S2G.
//
// TARGET PTX (sm_100+/sm_107a):
//   cp.async.bulk.relaxed.sys.global.shared::cta.bulk_group.b128 [dst],[src],size ;
//   cp.async.bulk.commit_group ;  cp.async.bulk.wait_group 0 ;
//
// Memory-order requirement: the sm_107a toolchain must ACCEPT and PRESERVE the
// strong system-scope .b128 down to SASS; a silently weakened scope would break the
// publication protocol.
// ============================================================================
#ifndef TLLM_PUSHCOMBINE_STRONG_S2G
#define TLLM_PUSHCOMBINE_STRONG_S2G 1
#endif

__device__ __forceinline__ void cp_async_bulk_s2g_sys(void* dstMem, void const* srcMem, int copySize)
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 1000 && TLLM_PUSHCOMBINE_STRONG_S2G
    // Strong system-scope, 16B(b128)-granular 1D bulk shared::cta -> global copy.
    // copySize must be a 16B multiple (push-combine record = 4480 = 280*16).
    asm("cp.async.bulk.relaxed.sys.global.shared::cta.bulk_group.b128 [%0], [%1], %2;"
        :
        : "l"(__as_ptr_gmem(dstMem)), "r"(__as_ptr_smem(srcMem)), "r"(copySize)
        : "memory");
#elif defined(__CUDACC__) && __CUDA_ARCH__ >= 900
    // Pre-sm_100 compile placeholder ONLY (weak scope is protocol-INSUFFICIENT).
    // Reaching this for a real push at runtime MUST be rejected (HOST_REJECT).
    asm("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
        :
        : "l"(__as_ptr_gmem(dstMem)), "r"(__as_ptr_smem(srcMem)), "r"(copySize)
        : "memory");
#endif
}

// Async-proxy fence: orders the generic-proxy shared writes (the K1 per-writer NVFP4
// quant stores) before the async-proxy bulk S2G reads them. EVERY writer
// issues this fence, THEN a CTA barrier, THEN a single elected issuer performs the S2G.
__device__ __forceinline__ void fence_proxy_async_shared_cta()
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
    asm volatile("fence.proxy.async.shared::cta;" : : : "memory");
#endif
}

} // namespace kernels
} // namespace tensorrt_llm
