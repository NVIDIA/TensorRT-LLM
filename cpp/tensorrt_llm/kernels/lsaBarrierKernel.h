/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>
#include <memory>

#include <nccl.h>

#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Opaque LSA barrier wrapper around NCCL 2.28+ ncclDevComm + ncclLsaBarrierSession.
//
// Encapsulates ncclDevCommCreate(comm, reqs{lsaBarrierCount=N}, &devComm) and
// launches a 1-block / 32-thread `__global__` kernel that performs a single
// release-ordered LSA barrier across all ranks of `comm`'s LSA team.
//
// Thread-safety: emit() is safe to call concurrently from multiple host
// threads — the slot rotation uses an atomic counter. Barrier slots cycle
// modulo `lsaBarrierCount` (16 by default).
class LsaBarrier
{
public:
    // Create + ncclDevCommCreate. Returns nullptr on failure (e.g. NCCL <2.28
    // or comm lacks deviceApiSupport). Caller should fall back to a different
    // barrier (e.g. PyTorch SymmetricMemory CUDA backend).
    static std::unique_ptr<LsaBarrier> create(ncclComm_t comm, int lsaBarrierCount = 16);

    ~LsaBarrier();

    // Launch the barrier kernel on `stream`. 1 block, 32 threads. Cycles
    // through `lsaBarrierCount` barrier slots via atomic counter.
    void emit(cudaStream_t stream);

    LsaBarrier(LsaBarrier const&) = delete;
    LsaBarrier& operator=(LsaBarrier const&) = delete;

private:
    LsaBarrier();
    void* mImpl{nullptr}; // points to LsaBarrierImpl, defined in .cu
};

} // namespace kernels

TRTLLM_NAMESPACE_END
