/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorrt_llm/kernels/lsaBarrierKernel.h"

#include <atomic>

#include <cuda_runtime.h>
#include <nccl.h>

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
#include <nccl_device.h>
#endif

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)

namespace
{

struct LsaBarrierImpl
{
    ncclDevComm devComm;
    int lsaBarrierCount;
    std::atomic<int> nextSlot{0};
};

// 1-block / 32-thread standalone barrier kernel for post-write cross-rank
// fence. Per NCCL device-API doc, sync() = arrive() + wait(), so a single
// call is a complete cross-rank barrier — wait() blocks until all team
// members have arrived.
//
// We use `memory_order_release` because this kernel is invoked AFTER the
// CE push (cudaMemcpyAsync) wrote our data into peers' recv_buf via
// symmetric memory. The release fence ensures those prior writes are
// committed (visible to peers) before our arrive signal is published.
// The cross-rank wait inside sync() inherently acquires peers' arrival
// signals, picking up the writes they published with their own release
// fences — so downstream SDPA on each rank can safely read its own
// recv_buf with all peers' contributions visible.
//
// Single CTA × 32 thread keeps SM contention minimal — the GEMM that
// runs in parallel on the default stream is SM-occupancy-bound (B200
// 148 SMs at ~100% with NVFP4 tensor-core kernels), so we leave nearly
// all SMs free for compute and only consume 1 warp briefly.
__global__ void lsaBarrierReleaseKernel(ncclDevComm devComm, int slot)
{
    ncclLsaBarrierSession<ncclCoopCta> bar{
        ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, static_cast<uint32_t>(slot)};
    bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

} // anonymous namespace

std::unique_ptr<LsaBarrier> LsaBarrier::create(ncclComm_t comm, int lsaBarrierCount)
{
    if (comm == nullptr || lsaBarrierCount <= 0)
    {
        return nullptr;
    }

    auto self = std::unique_ptr<LsaBarrier>(new LsaBarrier());
    auto* impl = new LsaBarrierImpl();
    self->mImpl = impl;

    // NCCL requires the requirements struct to be initialized via the macro
    // (sets internal version tag); plain memset returns ncclInvalidUsage.
    ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = lsaBarrierCount;

    ncclResult_t rc = ncclDevCommCreate(comm, &reqs, &impl->devComm);
    if (rc != ncclSuccess)
    {
        delete impl;
        self->mImpl = nullptr;
        return nullptr;
    }

    impl->lsaBarrierCount = lsaBarrierCount;
    return self;
}

LsaBarrier::LsaBarrier() = default;

LsaBarrier::~LsaBarrier()
{
    if (mImpl != nullptr)
    {
        delete static_cast<LsaBarrierImpl*>(mImpl);
        mImpl = nullptr;
    }
}

void LsaBarrier::emit(cudaStream_t stream)
{
    if (mImpl == nullptr)
    {
        return;
    }
    auto* impl = static_cast<LsaBarrierImpl*>(mImpl);
    int slot = impl->nextSlot.fetch_add(1, std::memory_order_relaxed) % impl->lsaBarrierCount;
    lsaBarrierReleaseKernel<<<1, 32, 0, stream>>>(impl->devComm, slot);
}

#else  // NCCL < 2.28: device API unavailable

std::unique_ptr<LsaBarrier> LsaBarrier::create(ncclComm_t /*comm*/, int /*lsaBarrierCount*/)
{
    return nullptr;
}

LsaBarrier::LsaBarrier() = default;
LsaBarrier::~LsaBarrier() = default;

void LsaBarrier::emit(cudaStream_t /*stream*/) {}

#endif // NCCL_VERSION_CODE

} // namespace kernels

TRTLLM_NAMESPACE_END
