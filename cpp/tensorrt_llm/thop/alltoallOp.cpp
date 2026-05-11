/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/ncclUtils.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/helixAllToAll.h"
#include "tensorrt_llm/kernels/lsaBarrierKernel.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
#include <nccl_device.h> // ncclGetPeerDevicePointer (host-side window peer-VA query)
#endif

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
#if ENABLE_MULTI_DEVICE

namespace
{

class AllToAllHelixOp
{
public:
    AllToAllHelixOp(std::set<int> group)
        : mGroup(std::move(group))
    {
    }

    ~AllToAllHelixOp() = default;

    int initialize()
    {
        TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
        mNcclComm = getComm(mGroup);
        TLLM_CHECK_WITH_INFO(mGroup.size() > 0, "group size should be greater than 0");
        TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
        return 0;
    }

    std::vector<torch::Tensor> run(torch::TensorList input_list, torch::optional<int64_t> num_lists)
    {
        TLLM_CHECK_WITH_INFO(mNcclComm.get() != nullptr, "mNcclComm should be initialized before used");
        auto num_lists_ = static_cast<int>(num_lists.value_or(1));
        auto num_ranks = static_cast<int>(mGroup.size());
        // note: ensures that input_list size > 0
        TLLM_CHECK_WITH_INFO(static_cast<int>(input_list.size()) == num_ranks * num_lists_,
            "input_list size should be equal to group size * num_lists");
        for (auto const& input : input_list)
        {
            TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
        }
        std::vector<torch::Tensor> output_list(static_cast<size_t>(num_lists_));
        auto stream = at::cuda::getCurrentCUDAStream(input_list[0].get_device());
        ncclGroupStart();
        for (int il = 0; il < num_lists_; ++il)
        {
            auto off = il * num_ranks;
            auto output_shape = input_list[off].sizes().vec();
            output_shape.insert(output_shape.begin(), num_ranks);
            auto output = torch::empty(output_shape, input_list[off].options());
            output_list[il] = output;
            auto type = tensorrt_llm::runtime::TorchUtils::dataType(input_list[off].scalar_type());
            auto nccl_type = (*getDtypeMap())[type];
            for (int r = 0; r < num_ranks; ++r)
            {
                auto const& input = input_list[off + r];
                ncclSend(input.data_ptr(), input.numel(), nccl_type, r, *mNcclComm, stream);
                ncclRecv(output[r].mutable_data_ptr(), output[r].numel(), nccl_type, r, *mNcclComm, stream);
            }
        }
        NCCLCHECK_THROW(ncclGroupEnd());
        return output_list;
    }

private:
    std::set<int> mGroup;
    std::shared_ptr<ncclComm_t> mNcclComm;
};

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)

// ---------------------------------------------------------------------------
// CE NCCL comm bootstrap: two paths — MPI (legacy) or torch.distributed PG
// (VisualGen style, TLLM_DISABLE_MPI=1).
//
// The MPI path delegates to the repo-wide `getComm(group)` in opUtils.cpp.
// The PG path bootstraps a fresh NCCL comm using:
//   - ncclGetUniqueId on rank 0 of the group
//   - broadcast of the 128-byte id via ProcessGroup
//   - ncclCommInitRankConfig with CTAPolicy = NCCL_CTA_POLICY_ZERO
//     (required to trigger the Copy Engine path in ncclAlltoAll)
//
// Comms are cached per group so repeat calls within a process reuse them.
// ---------------------------------------------------------------------------
std::shared_ptr<ncclComm_t> bootstrapCeCommFromPg(
    std::set<int> const& group, c10::intrusive_ptr<c10d::ProcessGroup> const& pg)
{
    TLLM_CHECK_WITH_INFO(pg, "ProcessGroup is null");
    TLLM_CHECK_WITH_INFO(!group.empty(), "group must be non-empty");
    TLLM_CHECK_WITH_INFO(static_cast<int>(group.size()) == pg->getSize(),
        "group size (%d) must match ProcessGroup size (%d). Pass a sub-group covering exactly the requested ranks.",
        static_cast<int>(group.size()), pg->getSize());

    // Use pg-local rank for the NCCL comm. group_ just carries world-rank
    // metadata for caching / API symmetry with other TRT-LLM ops.
    int const pgRank = pg->getRank();
    int const groupSize = pg->getSize();

    // Leader of the sub-group is pg rank 0. Obtain the unique id there, then
    // broadcast across the whole pg.
    ncclUniqueId uid;
    std::memset(&uid, 0, sizeof(uid));
    if (pgRank == 0)
    {
        NCCLCHECK_THROW(ncclGetUniqueId(&uid));
    }

    // Exchange the unique id via the ProcessGroup's TCP Store. This is the
    // same mechanism torch's ProcessGroupNCCL uses internally for its own
    // bootstrap — purely CPU/TCP, no GPU sync subtleties, and avoids the
    // NCCL-on-NCCL ordering issue of broadcasting via pg->broadcast (the
    // collective-on-same-PG would leave uid on GPU only, with no safe way
    // to synchronize a subsequent D2H for the host side).
    auto store = pg->getStore();
    TLLM_CHECK_WITH_INFO(store, "ProcessGroup has no Store; cannot bootstrap CE NCCL comm");

    // Unique key per group so sub-groups don't collide.
    std::ostringstream keyStream;
    keyStream << "trtllm_ce_nccl_uid";
    for (int r : group)
        keyStream << '_' << r;
    std::string const storeKey = keyStream.str();

    if (pgRank == 0)
    {
        uint8_t const* uidBytes = reinterpret_cast<uint8_t const*>(&uid);
        std::vector<uint8_t> bytes(uidBytes, uidBytes + sizeof(ncclUniqueId));
        store->set(storeKey, bytes);
    }
    else
    {
        // store->get blocks until key is set by rank 0 (TCP poll).
        std::vector<uint8_t> bytes = store->get(storeKey);
        TLLM_CHECK_WITH_INFO(bytes.size() == sizeof(ncclUniqueId),
            "store->get returned %zu bytes, expected %zu", bytes.size(), sizeof(ncclUniqueId));
        std::memcpy(&uid, bytes.data(), sizeof(ncclUniqueId));
    }

    // Note: do NOT setenv NCCL_RUNTIME_CONNECT=0 or NCCL_GRAPH_REGISTER=0.
    // Those make NCCL skip connection/graph setup, which also skips the LSA
    // (Local Shared Address) VA mapping that the CE path depends on for
    // computing peer pointers via ncclDevrGetLsaRankPtr. With those disabled,
    // ncclAlltoAll's internal cudaMemcpyBatchAsync receives garbage peer ptrs
    // and returns invalid argument. Let NCCL do full init.
#if !defined(_WIN32)
    setenv("NCCL_CTA_POLICY", "2", 0);
#endif

    std::shared_ptr<ncclComm_t> ncclComm(new ncclComm_t,
        [](ncclComm_t* c)
        {
            if (c && *c)
            {
                tensorrt_llm::common::nccl_util::NcclCommResourceManager::getInstance().cleanupResources(*c);
                (void) ncclCommDestroy(*c);
                *c = nullptr;
            }
            delete c;
        });
    // Use plain ncclCommInitRank (no config struct). The compile-time NCCL
    // header (2.28) has ncclConfig_t layout that doesn't match the 2.29
    // runtime ABI, so passing a config can return success but leave the comm
    // in a state where downstream CE ops fail with "invalid argument".
    NCCLCHECK_THROW(ncclCommInitRank(ncclComm.get(), groupSize, uid, pgRank));
    return ncclComm;
}

// Caches one ncclComm_t per group for the PG bootstrap path.
std::shared_ptr<ncclComm_t> getCommForCe(
    std::set<int> const& group, c10::intrusive_ptr<c10d::ProcessGroup> const& pg)
{
    TLLM_CHECK_WITH_INFO(
        pg, "alltoall_ce / get_ce_registered_buffer require a torch ProcessGroup — the MPI bootstrap path has been "
            "removed. Pass dist.distributed_c10d._get_default_group() or a sub-group.");

    static std::map<std::set<int>, std::shared_ptr<ncclComm_t>> sCommCache;
    static std::mutex sCommMutex;
    std::lock_guard<std::mutex> lock(sCommMutex);
    auto it = sCommCache.find(group);
    if (it != sCommCache.end())
    {
        return it->second;
    }
    auto comm = bootstrapCeCommFromPg(group, pg);
    sCommCache[group] = comm;
    return comm;
}


// ============================================================================
// v11 hybrid path: self-segment via cuMemcpyAsync (driver auto-routes to SM,
// ~1650 GB/s intra-dev on B200) + peer segments via cuMemcpyAsync (driver
// auto-routes to NVLink peer CE, ~600 GB/s) + ncclSignal / ncclWaitSignal
// cross-rank barrier (SM-free, stream-ordered, graph-capturable).
//
// Why not the pure-CE path (AllToAllCEOp above)?
//   `cudaMemcpyBatchAsync` intra-dev D2D hits a fixed ~107 GB/s LCE quota on
//   B200 that NVIDIA's RM driver imposes (see REPORT.md §5.1 — 4-hypothesis
//   investigation across numOps × locHint × flag ruled out any workaround).
//   For Ulysses QKV each call copies one self-segment and (P-1) peer segments;
//   with pure CE the self-segment is 15.9× slower than SM. This hybrid path
//   lets the driver select SM for intra-dev and NVLink CE for peer, so each
//   segment runs on its optimal fabric.
//
// Barrier: NCCL 2.29+ ncclSignal / ncclWaitSignal. Not in NCCL 2.28 headers,
// so we lazily resolve via dlsym at init time and fail clearly if the runtime
// is < 2.29.
// ============================================================================
namespace v11_hybrid
{

struct NcclWaitSignalDesc
{
    int opCnt;
    int peer;
    int sigIdx;
    int ctx;
};

using NcclSignalFn
    = ncclResult_t (*)(int peer, int sigIdx, int opCnt, unsigned ctx, ncclComm_t comm, cudaStream_t stream);
using NcclWaitSignalFn
    = ncclResult_t (*)(int nDescs, NcclWaitSignalDesc const* descs, ncclComm_t comm, cudaStream_t stream);

struct Syms
{
    NcclSignalFn signal = nullptr;
    NcclWaitSignalFn waitSignal = nullptr;
    bool ok = false;
    static Syms const& get()
    {
        static Syms s = [] {
            Syms r;
            r.signal = reinterpret_cast<NcclSignalFn>(dlsym(RTLD_DEFAULT, "ncclSignal"));
            r.waitSignal = reinterpret_cast<NcclWaitSignalFn>(dlsym(RTLD_DEFAULT, "ncclWaitSignal"));
            r.ok = (r.signal != nullptr) && (r.waitSignal != nullptr);
            return r;
        }();
        return s;
    }
};

} // namespace v11_hybrid

class AllToAllV11Op
{
public:
    AllToAllV11Op(std::set<int> group, c10::intrusive_ptr<c10d::ProcessGroup> pg)
        : mGroup(std::move(group))
        , mPg(std::move(pg))
    {
    }

    int initialize()
    {
        TLLM_CHECK_WITH_INFO(mPg, "alltoall_v11 requires a torch ProcessGroup");
        TLLM_CHECK_WITH_INFO(!mGroup.empty(), "group must be non-empty");
        TLLM_CHECK_WITH_INFO(static_cast<int>(mGroup.size()) == mPg->getSize(),
            "group size (%d) must match ProcessGroup size (%d). Pass a sub-group covering exactly the requested ranks.",
            static_cast<int>(mGroup.size()), mPg->getSize());

        auto const& syms = v11_hybrid::Syms::get();
        TLLM_CHECK_WITH_INFO(syms.ok,
            "alltoall_v11 requires NCCL 2.29+ runtime with ncclSignal / ncclWaitSignal. "
            "Load libnccl.so.2.29+ before starting the process (e.g. LD_PRELOAD=libnccl.so.2.29.2).");

        // Reuse the PG-based NCCL comm bootstrap from the pure-CE path. The
        // comm is only used by v11 for ncclSignal / ncclWaitSignal (barrier);
        // data transport goes through cuMemcpyAsync directly, bypassing NCCL.
        mNcclComm = getCommForCe(mGroup, mPg);

        int const pSize = mPg->getSize();
        int const pgRank = mPg->getRank();
        int myDev = -1;
        TLLM_CUDA_CHECK(cudaGetDevice(&myDev));
        mMyDev = myDev;

        // Exchange each rank's CUDA device id via Store — needed to call
        // cudaDeviceEnablePeerAccess on the correct peer devices for NVLink
        // P2P. Rank-to-device mapping is typically rank==device on single-node
        // but we don't assume it.
        auto store = mPg->getStore();
        TLLM_CHECK_WITH_INFO(store, "ProcessGroup has no Store");
        std::ostringstream keyPrefix;
        keyPrefix << "trtllm_v11_dev";
        for (int r : mGroup)
        {
            keyPrefix << '_' << r;
        }
        std::string const keyBase = keyPrefix.str();
        {
            std::string myKey = keyBase + "_rank" + std::to_string(pgRank);
            std::vector<uint8_t> devBytes(sizeof(int));
            std::memcpy(devBytes.data(), &myDev, sizeof(int));
            store->set(myKey, devBytes);
        }
        mPeerDevIds.resize(pSize);
        for (int r = 0; r < pSize; ++r)
        {
            std::string key = keyBase + "_rank" + std::to_string(r);
            auto bytes = store->get(key);
            TLLM_CHECK_WITH_INFO(bytes.size() == sizeof(int),
                "Store returned %zu bytes for device id, expected %zu", bytes.size(), sizeof(int));
            std::memcpy(&mPeerDevIds[r], bytes.data(), sizeof(int));
        }

        // Enable peer access on each peer device. "Already enabled" is fine;
        // other errors we surface.
        for (int r = 0; r < pSize; ++r)
        {
            if (r == pgRank)
                continue;
            cudaError_t rc = cudaDeviceEnablePeerAccess(mPeerDevIds[r], 0);
            if (rc != cudaSuccess && rc != cudaErrorPeerAccessAlreadyEnabled)
            {
                TLLM_CHECK_WITH_INFO(false, "cudaDeviceEnablePeerAccess(dev=%d) failed: %s", mPeerDevIds[r],
                    cudaGetErrorString(rc));
            }
            // Consume any sticky error from "already enabled" so later checks are clean.
            (void) cudaGetLastError();
        }

        return 0;
    }

    // Op-local non-default stream (same rationale as AllToAllCEOp: avoid the
    // legacy NULL stream that some CUDA 13 APIs reject, and isolate op ordering
    // from caller's stream).
    static c10::cuda::CUDAStream& opStream(int device)
    {
        return opStreamStatic(device);
    }

    // Public static wrapper so alltoall_v11_peer_only can share the same
    // thread-local stream pool without constructing a full op instance.
    static c10::cuda::CUDAStream& opStreamStatic(int device)
    {
        thread_local std::map<int, c10::cuda::CUDAStream> streams;
        auto it = streams.find(device);
        if (it == streams.end())
        {
            it = streams.emplace(device, c10::cuda::getStreamFromPool(/*isHighPriority=*/false, device)).first;
        }
        return it->second;
    }

    // Accessors used by the peer-only variant.
    int getMyDevice() const { return mMyDev; }
    std::shared_ptr<ncclComm_t> getNcclComm() const { return mNcclComm; }
    at::cuda::CUDAEvent& getStartEvent() { return mStartEvent; }
    at::cuda::CUDAEvent& getDoneEvent() { return mDoneEvent; }
    int getPgSize() const { return mPg->getSize(); }

    // Emit a single release-ordered LSA barrier on `stream`. Lazy-creates
    // the underlying ncclDevComm on first call. No-op if NCCL <2.28 or
    // comm lacks device API support (caller responsible for fallback to
    // PyTorch SymMem barrier).
    void emitLsaBarrier(cudaStream_t stream)
    {
        // Fast path: barrier already created.
        if (mLsaBarrier)
        {
            mLsaBarrier->emit(stream);
            return;
        }
        // Slow path: lazy init under mutex (collective — all ranks must reach).
        {
            std::lock_guard<std::mutex> lock(mLsaBarrierMutex);
            if (!mLsaBarrier)
            {
                mLsaBarrier = tensorrt_llm::kernels::LsaBarrier::create(*mNcclComm, /*lsaBarrierCount=*/16);
            }
        }
        TLLM_CHECK_WITH_INFO(mLsaBarrier != nullptr,
            "LsaBarrier::create failed — NCCL <2.28 or comm lacks deviceApiSupport.");
        mLsaBarrier->emit(stream);
    }
    int getPgRank() const { return mPg->getRank(); }


#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    // NCCL Window variant of run_gc_direct: structurally identical hybrid
    // SM(self) + CE(peer) + cross-stream join, but the slot ring is backed
    // by ncclMemAlloc + ncclCommWindowRegister(SYMMETRIC) + ncclGetPeerDevicePointer
    // instead of cudaMalloc + cudaIpcOpenMemHandle. Cross-rank fence is
    // expected to come from a separate ulysses_lsa_barrier call (Phase A);
    // this op never emits its own barrier (do_barrier is implicit false).
    torch::Tensor run_gc_direct_issue_window(torch::Tensor input, cudaStream_t selfCopyStreamExt)
    {
        TLLM_CHECK_WITH_INFO(mNcclComm.get() != nullptr, "alltoall_v11 op not initialized");
        TORCH_CHECK(input.is_contiguous(), "alltoall_v11 input must be contiguous");
        TORCH_CHECK(selfCopyStreamExt != nullptr,
            "run_gc_direct_issue_window requires non-null selfCopyStreamExt");

        int const pSize = mPg->getSize();
        int const pgRank = mPg->getRank();
        TORCH_CHECK(input.numel() % pSize == 0, "input.numel() must be divisible by group size");

        size_t const elemSize = static_cast<size_t>(input.element_size());
        size_t const chunkBytes = (static_cast<size_t>(input.numel()) / pSize) * elemSize;
        size_t const bufferBytes = chunkBytes * static_cast<size_t>(pSize);
        int const dev = input.get_device();

        int slotIdx = nextSlot(*mNcclComm);
        WindowSlot& slot = getWindowSlot(slotIdx, bufferBytes);

        auto torchStream = at::cuda::getCurrentCUDAStream(dev);
        cudaStream_t stream = torchStream.stream();
        char const* srcBase = static_cast<char const*>(input.data_ptr());

        cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
        TLLM_CUDA_CHECK(cudaStreamIsCapturing(stream, &captureStatus));
        bool const underCapture = (captureStatus != cudaStreamCaptureStatusNone);

        // Fork caller → selfCopyStream via per-call event from pool. Same
        // CG-safe pattern as run_gc_direct.
        cudaEvent_t startEv = allocCallEvent();
        TLLM_CUDA_CHECK(cudaEventRecord(startEv, stream));
        TLLM_CUDA_CHECK(cudaStreamWaitEvent(selfCopyStreamExt, startEv, 0));
        cudaEvent_t selfDoneEv = allocCallEvent();

        // (a) self-copy on selfCopyStreamExt — write into self-mapped peer VA.
        //     P4.2: cudaMemcpyAsync to ncclGetPeerDevicePointer(self) is bit-exact.
        {
            char* dst_self = static_cast<char*>(slot.peerPtrs[pgRank])
                             + static_cast<size_t>(pgRank) * chunkBytes;
            char const* src_self = srcBase + static_cast<size_t>(pgRank) * chunkBytes;
            TLLM_CUDA_CHECK(cudaMemcpyAsync(dst_self, src_self, chunkBytes,
                cudaMemcpyDeviceToDevice, selfCopyStreamExt));
            TLLM_CUDA_CHECK(cudaEventRecord(selfDoneEv, selfCopyStreamExt));
        }

        // (b) peer copies — out-of-capture uses cudaMemcpyBatchAsync (driver
        //     fans out to multiple CE engines); under capture serialize to a
        //     single cudaMemcpyAsync loop on `stream`. We tested a fan-out
        //     to P-1 helper streams under capture: it does shorten per-memcpy
        //     GPU time (~12% at P=8) but adds 4(P-1) cross-stream events whose
        //     graph-node overhead exceeds the gain at LTX2 chunk sizes
        //     (~850KB-4MB per peer, P=2/4/8 graph mode all neutral or +1-3%
        //     slower). Revisit if a workload with multi-MB per-peer chunks
        //     where memcpy time dominates over event scheduling lands.
        if (pSize > 1)
        {
            int const nPeers = pSize - 1;
            if (!underCapture)
            {
                std::vector<void*> dsts;
                dsts.reserve(nPeers);
                std::vector<void const*> srcs;
                srcs.reserve(nPeers);
                std::vector<size_t> sizes;
                sizes.reserve(nPeers);
                for (int peer = 0; peer < pSize; ++peer)
                {
                    if (peer == pgRank) continue;
                    dsts.push_back(static_cast<char*>(slot.peerPtrs[peer])
                                   + static_cast<size_t>(pgRank) * chunkBytes);
                    srcs.push_back(srcBase + static_cast<size_t>(peer) * chunkBytes);
                    sizes.push_back(chunkBytes);
                }
                cudaMemcpyAttributes attrs[1];
                std::memset(&attrs[0], 0, sizeof(attrs[0]));
                attrs[0].srcAccessOrder = cudaMemcpySrcAccessOrderStream;
                attrs[0].flags = 1u;
                size_t attrIdxs[1] = {0};
                TLLM_CUDA_CHECK(cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(),
                    static_cast<size_t>(nPeers), attrs, attrIdxs, 1, stream));
            }
            else
            {
                // Capture-safe serial loop. Mirrors run_gc_direct's fallback.
                for (int peer = 0; peer < pSize; ++peer)
                {
                    if (peer == pgRank) continue;
                    char* dst = static_cast<char*>(slot.peerPtrs[peer])
                                + static_cast<size_t>(pgRank) * chunkBytes;
                    char const* src = srcBase + static_cast<size_t>(peer) * chunkBytes;
                    TLLM_CUDA_CHECK(cudaMemcpyAsync(dst, src, chunkBytes,
                        cudaMemcpyDeviceToDevice, stream));
                }
            }
        }

        // (c) join selfCopyStream's completion back into caller stream.
        //     Cross-rank fence is the caller's responsibility (LSA barrier).
        TLLM_CUDA_CHECK(cudaStreamWaitEvent(stream, selfDoneEv, 0));

        // Output wraps the local ncclMemAlloc'd base pointer (peer writes
        // land here through their respective peerPtrs[mMyRank] mappings).
        auto output = torch::from_blob(
            slot.basePtr, input.sizes(), input.strides(), /*deleter=*/[](void*) {},
            torch::dtype(input.scalar_type()).device(torch::kCUDA));
        return output;
    }
#endif // NCCL_VERSION_CODE >= 2.28


private:
    static constexpr int kNumSlots = 4;

    // Ring counter for output slot selection. Shared with callers so that
    // pipelined invocations don't collide on the same slot before the previous
    // one has consumed its output.
    static int nextSlot(ncclComm_t comm)
    {
        static std::map<ncclComm_t, int> counters;
        static std::mutex m;
        std::lock_guard<std::mutex> lock(m);
        int& cnt = counters[comm];
        int slot = cnt;
        cnt = (cnt + 1) % kNumSlots;
        return slot;
    }

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    // NCCL Window-backed slot: same shape as Slot but allocated via
    // ncclMemAlloc + ncclCommWindowRegister(SYMMETRIC) instead of
    // cudaMalloc + cudaIpcOpenMemHandle. peerPtrs[r] is a NCCL remapped VA
    // (host API ncclGetPeerDevicePointer) that this rank can write into to
    // reach peer-r's symmetric mapping; peerPtrs[mPg->getRank()] aliases the
    // local basePtr's remapped self-VA.
    struct WindowSlot
    {
        void* basePtr = nullptr;     // ncclMemAlloc'd local buffer (output)
        ncclWindow_t win = nullptr;  // SYMMETRIC-registered window handle
        size_t size = 0;
        std::vector<void*> peerPtrs; // size == pSize; remapped peer-write VAs
    };

    // Process-lifetime pool keyed by (comm, slotIdx). Mirrors slotMap() for
    // the IPC path. Buffers are obtained from NCCLWindowAllocator (which
    // pools by comm and reuses on best-fit), and are released back to that
    // pool on size-up so the larger replacement can be requested fresh.
    static std::map<std::pair<ncclComm_t, int>, WindowSlot>& windowSlotMap()
    {
        static std::map<std::pair<ncclComm_t, int>, WindowSlot> s;
        return s;
    }

    static std::mutex& windowSlotMutex()
    {
        static std::mutex m;
        return m;
    }

    // Lazy collective allocator for the window slot ring. Mirrors getSlot()
    // but uses NCCL window APIs end-to-end:
    //   1. NCCLWindowAllocator::requestBuffer (collective: ncclMemAlloc +
    //      ncclCommWindowRegister + cross-rank OOM allreduce sync).
    //   2. ncclGetPeerDevicePointer × pSize (host API, no Store roundtrip).
    // Caller invariant: all ranks must reach getWindowSlot with the same
    // (slotIdx, requiredSize) before the data path can replay under CUDA
    // Graph capture, since NCCLWindowAllocator skips alloc under capture.
    WindowSlot& getWindowSlot(int slotIdx, size_t requiredSize)
    {
        int const pSize = mPg->getSize();

        std::lock_guard<std::mutex> lock(windowSlotMutex());
        auto& slot = windowSlotMap()[{*mNcclComm, slotIdx}];

        if (slot.basePtr != nullptr && slot.size >= requiredSize)
        {
            return slot;
        }

        auto& alloc = common::nccl_util::NCCLWindowAllocator::getInstance();

        // On size-up: release the prior window buffer back to the pool and
        // request a larger one (NCCLWindowAllocator best-fits and falls back
        // to a fresh allocateAndRegisterBuffer when nothing in the pool is
        // big enough). Both releaseBuffer and requestBuffer are collective-
        // safe in the sense that all ranks call them in the same order; the
        // underlying ncclCommWindowRegister/Deregister is collective at the
        // NCCL level.
        if (slot.basePtr != nullptr)
        {
            alloc.releaseBuffer(*mNcclComm, slot.basePtr);
            slot.basePtr = nullptr;
            slot.win = nullptr;
            slot.size = 0;
            slot.peerPtrs.clear();
        }

        auto buf = alloc.requestBuffer(*mNcclComm, requiredSize);
        TLLM_CHECK_WITH_INFO(buf.isValid(),
            "NCCLWindowAllocator::requestBuffer failed (size=%zu). NCCL <2.28 or registration error.",
            requiredSize);

        slot.basePtr = buf.ptr;
        slot.win = buf.window;
        slot.size = buf.size;

        // Resolve peer write addresses via the host-side window query API.
        // ncclGetPeerDevicePointer returns a remapped VA: distinct from the
        // raw ncclMemAlloc pointer, suitable for cudaMemcpyAsync writes that
        // land in peer's local memory (verified in P4.2 standalone test).
        slot.peerPtrs.assign(pSize, nullptr);
        for (int peer = 0; peer < pSize; ++peer)
        {
            void* peerPtr = nullptr;
            NCCLCHECK_THROW(ncclGetPeerDevicePointer(slot.win, /*offset=*/0, peer, &peerPtr));
            slot.peerPtrs[peer] = peerPtr;
        }

        return slot;
    }
#endif // NCCL_VERSION_CODE >= 2.28

    std::set<int> mGroup;
    c10::intrusive_ptr<c10d::ProcessGroup> mPg;
    std::shared_ptr<ncclComm_t> mNcclComm;
    int mMyDev = -1;
    std::vector<int> mPeerDevIds;
    // Persistent events reused across run() calls. at::cuda::CUDAEvent lazily
    // creates the underlying cudaEvent_t on first record(), so keeping members
    // avoids paying ~10-20us cudaEventCreate per call.
    at::cuda::CUDAEvent mStartEvent;
    at::cuda::CUDAEvent mDoneEvent;
    // Raw cudaEvent_t used only when caller supplies an external self-copy
    // stream (Green Context partition). Raw API needed because we need to
    // record / wait against an arbitrary cudaStream_t the caller owns.
    // Lazily created on first use, destroyed in destructor.
    cudaEvent_t mSelfDoneCudaEvent = nullptr;

    // Dummy 1-int device buffer used by run_barrier_allreduce. Lazily
    // allocated on first call, freed in destructor.
    int* mBarrierBuf = nullptr;

    // NCCL 2.28+ device-API LSA barrier (alternative to PyTorch SymMem barrier).
    // Lazily created on first emitLsaBarrier() call to avoid `ncclDevCommCreate`
    // cost when not used. Returns nullptr if NCCL <2.28 or comm lacks deviceApiSupport.
    std::unique_ptr<tensorrt_llm::kernels::LsaBarrier> mLsaBarrier;
    std::mutex mLsaBarrierMutex;

    // Pre-allocated per-call event pool. Required for CUDA Graph capture
    // correctness: under capture, multiple cudaEventRecord on the same handle
    // creates ambiguous dependency edges — replays may pair waits with the
    // wrong record. Cycling through a pool of distinct events makes each
    // record/wait pair unique within a single capture. Pool size must exceed
    // max events used per CG capture (production: 48 blocks × ~7 alltoalls ×
    // ~3 events × 2 (warmup+capture) ≈ 4032; 65536 leaves large safety margin
    // for multi-step / multi-shape scenarios).
    static constexpr size_t kEventPoolSize = 65536;
    std::vector<cudaEvent_t> mEventPool;
    size_t mEventPoolIdx = 0;

    cudaEvent_t allocCallEvent()
    {
        if (mEventPool.empty())
        {
            mEventPool.reserve(kEventPoolSize);
            for (size_t i = 0; i < kEventPoolSize; ++i)
            {
                cudaEvent_t ev;
                TLLM_CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
                mEventPool.push_back(ev);
            }
        }
        // [BISECT #4] TRTLLM_ALLTOALL_LEGACY=1 cycles events through tiny pool
        // (mod 4) — emulates pre-fix shared-event behavior to test if SymMem
        // barrier × 3 alone subsumes the per-call event pool.
        static bool const sLegacy = std::getenv("TRTLLM_ALLTOALL_LEGACY") != nullptr;
        size_t const modSize = sLegacy ? size_t{4} : kEventPoolSize;
        cudaEvent_t ev = mEventPool[mEventPoolIdx % modSize];
        ++mEventPoolIdx;
        return ev;
    }

public:
    ~AllToAllV11Op()
    {
        if (mSelfDoneCudaEvent)
            cudaEventDestroy(mSelfDoneCudaEvent);
        if (mBarrierBuf)
            cudaFree(mBarrierBuf);
        for (auto ev : mEventPool)
            cudaEventDestroy(ev);
    }
};

// ============================================================================
// v11 "peer-only" variant (P1 refactor, inspired by v10):
//
// Architecture: caller pre-allocates a recv buffer via get_v11_recv_buffer, then
// during torch.compile'd pre-a2a compute writes the self chunk directly to
// recv_buf[rank*chunk] (SM kernel path, fused into compute graph) and peer
// chunks to a separate send_buf. The peer-only op then only does
//   - P-1 peer cudaMemcpyBatchAsync (or loop under graph) — NVLink CE
//   - ncclSignal/WaitSignal barrier
// ...skipping the self-copy entirely (saved ~15us CPU + self-chunk transfer time).
//
// Compared to AllToAllV11Op (which internally does self + peer + barrier), this
// variant reduces per-call CPU dispatch by ~15us (1 fewer cudaMemcpyAsync setup).
// The "self goes SM" benefit is preserved because torch.compile fuses the
// recv_buf[rank].copy_(self_data) into the compute graph's GEMM+permute kernels,
// which run on SMs at full HBM bandwidth.
// ============================================================================

torch::Tensor ulysses_alltoall_hybrid_symm(torch::Tensor input, torch::List<int64_t> group_,
    c10::intrusive_ptr<c10d::ProcessGroup> const& pg, int64_t self_copy_stream_handle)
{
#if ENABLE_MULTI_DEVICE
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    std::set<int> group;
    for (int64_t rank : group_)
    {
        group.insert(static_cast<int>(rank));
    }
    // Reuse the same AllToAllV11Op cache as the IPC issue op so the underlying
    // ncclComm + peer device map are shared (no double bootstrap).
    static std::map<std::set<int>, std::shared_ptr<AllToAllV11Op>> sOpCache;
    static std::mutex sOpMutex;
    std::shared_ptr<AllToAllV11Op> op;
    {
        std::lock_guard<std::mutex> lock(sOpMutex);
        auto it = sOpCache.find(group);
        if (it == sOpCache.end())
        {
            op = std::make_shared<AllToAllV11Op>(group, pg);
            op->initialize();
            sOpCache[group] = op;
        }
        else
        {
            op = it->second;
        }
    }
    cudaStream_t selfStream = reinterpret_cast<cudaStream_t>(self_copy_stream_handle);
    return op->run_gc_direct_issue_window(input, selfStream);
#else
    TORCH_CHECK(false, "ulysses_alltoall_hybrid_symm requires NCCL >= 2.28");
    return torch::Tensor();
#endif
#else
    return input;
#endif
}

// NCCL device-API LSA barrier: launches a 1-block / 32-thread kernel that
// performs a single release-ordered ncclLsaBarrierSession::sync. Designed as
// an alternative to PyTorch SymmetricMemory.barrier(0,0) for fencing
// alltoall slot writes across ranks under CUDA Graph capture.
//
// Reuses the same AllToAllV11Op cache as the data-path ops so the underlying
// ncclComm + ncclDevComm are shared. The lsa barrier is lazy-init on first
// emit (collective — all ranks must reach the call together).
void ulysses_lsa_barrier(
    torch::List<int64_t> group_, c10::intrusive_ptr<c10d::ProcessGroup> const& pg)
{
#if ENABLE_MULTI_DEVICE
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    std::set<int> group;
    for (int64_t rank : group_) { group.insert(static_cast<int>(rank)); }
    static std::map<std::set<int>, std::shared_ptr<AllToAllV11Op>> sOpCache;
    static std::mutex sOpMutex;
    std::shared_ptr<AllToAllV11Op> op;
    {
        std::lock_guard<std::mutex> lock(sOpMutex);
        auto it = sOpCache.find(group);
        if (it == sOpCache.end()) {
            op = std::make_shared<AllToAllV11Op>(group, pg);
            op->initialize();
            sOpCache[group] = op;
        } else { op = it->second; }
    }
    op->emitLsaBarrier(at::cuda::getCurrentCUDAStream().stream());
#else
    TORCH_CHECK(false, "ulysses_lsa_barrier requires NCCL >= 2.28");
#endif
#endif
}

#endif // NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)

} // namespace

#endif // ENABLE_MULTI_DEVICE

std::vector<torch::Tensor> alltoall_helix(
    torch::TensorList input_list, torch::List<int64_t> group_, torch::optional<int64_t> num_lists)
{
#if ENABLE_MULTI_DEVICE
    std::set<int> group;
    for (int64_t rank : group_)
    {
        group.insert(static_cast<int>(rank));
    }
    AllToAllHelixOp op(group);
    op.initialize();
    return op.run(input_list, num_lists);
#else
    return {};
#endif // ENABLE_MULTI_DEVICE
}

/**
 * Helix All-to-All operation with two fields.
 *
 * Input tensors have shape [..., cp_size, kv_lora_rank] for partial_o and [...,
 * cp_size, 2] for softmax_stats. The operation exchanges data along the cp_size
 * dimension across all ranks.
 *
 * @param partial_o Field 0 tensor (half precision, shape [..., cp_size,
 * kv_lora_rank])
 * @param softmax_stats Field 1 tensor (float32, shape [..., cp_size, 2])
 * @param workspace Workspace tensor (uint64, strided across ranks)
 * @param cp_rank Current context parallel rank
 * @param cp_size Total number of context parallel ranks
 * @return tuple of (partial_o_out, softmax_stats_out) with same shapes as inputs
 */
std::tuple<torch::Tensor, torch::Tensor> alltoall_helix_native(
    torch::Tensor partial_o, torch::Tensor softmax_stats, torch::Tensor workspace, int64_t cp_rank, int64_t cp_size)
{

    // Input validation
    CHECK_TH_CUDA(partial_o);
    CHECK_TH_CUDA(softmax_stats);
    CHECK_TH_CUDA(workspace);
    CHECK_CONTIGUOUS(partial_o);
    CHECK_CONTIGUOUS(softmax_stats);

    // Type checks
    TORCH_CHECK(partial_o.scalar_type() == at::ScalarType::Half || partial_o.scalar_type() == at::ScalarType::BFloat16,
        "partial_o must be half or bfloat16");
    CHECK_TYPE(softmax_stats, at::ScalarType::Float);
    CHECK_TYPE(workspace, at::ScalarType::UInt64);

    // Shape validation
    TORCH_CHECK(partial_o.dim() >= 2, "partial_o must have at least 2 dimensions");
    TORCH_CHECK(softmax_stats.dim() >= 2, "softmax_stats must have at least 2 dimensions");
    TORCH_CHECK(
        partial_o.dim() == softmax_stats.dim(), "partial_o and softmax_stats must have same number of dimensions");

    // Get dimensions
    int kv_lora_rank = partial_o.size(-1);
    TORCH_CHECK(partial_o.size(-2) == cp_size && softmax_stats.size(-2) == cp_size,
        "partial_o/softmax_stats second-to-last dimension must equal cp_size");
    TORCH_CHECK(softmax_stats.size(-1) % 2 == 0 && softmax_stats.size(-1) >= 2,
        "softmax_stats last dimension must be divisible by 2 (float2)");
    bool allowVariableField1 = softmax_stats.size(-1) > 2;

    // Check that leading dimensions match
    for (int i = 0; i < partial_o.dim() - 2; i++)
    {
        TORCH_CHECK(partial_o.size(i) == softmax_stats.size(i),
            "partial_o and softmax_stats must have matching dimensions except last two");
    }
    TORCH_CHECK(partial_o.size(-1) * partial_o.element_size() % 16 == 0, "partial_o must be aligned to 16 bytes");

    TORCH_CHECK(workspace.dim() == 2, "workspace must be 2D (strided across ranks)");
    TORCH_CHECK(workspace.size(0) == cp_size, "workspace must have cp_size rows");

    // Calculate entry count (product of all dimensions before cp_size)
    // This is the number of entries to process per peer rank
    int entry_count = 1;
    for (int i = 0; i < partial_o.dim() - 2; i++)
    {
        entry_count *= partial_o.size(i);
    }

    // Reshape to 3D: [entry_count, cp_size, feature_dim]
    torch::Tensor partial_o_3d = partial_o.reshape({entry_count, cp_size, kv_lora_rank});
    torch::Tensor softmax_stats_3d = softmax_stats.reshape({entry_count, cp_size, softmax_stats.size(-1)});

    // Allocate output tensors (same shape as input)
    torch::Tensor partial_o_out = torch::empty_like(partial_o);
    torch::Tensor softmax_stats_out = torch::empty_like(softmax_stats);

    torch::Tensor partial_o_out_3d = partial_o_out.reshape({entry_count, cp_size, kv_lora_rank});
    torch::Tensor softmax_stats_out_3d = softmax_stats_out.reshape({entry_count, cp_size, softmax_stats.size(-1)});

    // Setup parameters
    tensorrt_llm::kernels::HelixAllToAllParams params;

    // Field 0 (variable size half)
    params.sendFields[0].dataPtr = reinterpret_cast<uint8_t*>(partial_o_3d.data_ptr());
    params.sendFields[0].elementCount = kv_lora_rank;
    params.sendFields[0].elementSize = partial_o.element_size();
    params.sendFields[0].stride = partial_o_3d.stride(1) * partial_o.element_size();

    params.recvFields[0].dataPtr = reinterpret_cast<uint8_t*>(partial_o_out_3d.data_ptr());
    params.recvFields[0].elementCount = kv_lora_rank;
    params.recvFields[0].elementSize = partial_o.element_size();
    params.recvFields[0].stride = partial_o_out_3d.stride(1) * partial_o.element_size();

    // Field 1 (single float2)
    params.sendFields[1].dataPtr = reinterpret_cast<uint8_t*>(softmax_stats_3d.data_ptr<float>());
    params.sendFields[1].elementCount = softmax_stats.size(-1);
    params.sendFields[1].elementSize = softmax_stats.element_size();
    params.sendFields[1].stride = softmax_stats_3d.stride(1) * softmax_stats.element_size();

    params.recvFields[1].dataPtr = reinterpret_cast<uint8_t*>(softmax_stats_out_3d.data_ptr<float>());
    params.recvFields[1].elementCount = softmax_stats.size(-1);
    params.recvFields[1].elementSize = softmax_stats.element_size();
    params.recvFields[1].stride = softmax_stats_out_3d.stride(1) * softmax_stats.element_size();

    // Entry count and workspace
    params.entryCount = entry_count;
    params.workspace = workspace.data_ptr<uint64_t>();
    params.workspaceStrideInU64 = workspace.stride(0);

    // CP info
    params.cpRank = cp_rank;
    params.cpSize = cp_size;
    params.channelCount = 0; // auto-compute
    params.maxChannelCount = tensorrt_llm::kernels::computeHelixMaxChannelCount(cp_size);

    // Launch kernel
    auto stream = at::cuda::getCurrentCUDAStream();
    tensorrt_llm::kernels::launchHelixAllToAll(params, allowVariableField1, stream);

    return std::make_tuple(partial_o_out, softmax_stats_out);
}
void initialize_helix_workspace(torch::Tensor workspace, int64_t cp_rank, int64_t cp_size)
{
    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, at::ScalarType::UInt64);
    TORCH_CHECK(workspace.dim() == 2, "workspace must be 2D");
    TORCH_CHECK(workspace.size(0) == cp_size, "workspace must have cp_size rows");
    TORCH_CHECK(cp_rank >= 0 && cp_rank < cp_size, "cp_rank must be in [0, cp_size)");

    auto stream = at::cuda::getCurrentCUDAStream();
    uint64_t* global_workspace_ptr = workspace.data_ptr<uint64_t>();
    uint64_t* local_workspace_ptr = workspace[cp_rank].data_ptr<uint64_t>();
    TORCH_CHECK(local_workspace_ptr == global_workspace_ptr + cp_rank * workspace.stride(0),
        "local_workspace_ptr must be at the correct offset in the global "
        "workspace");
    tensorrt_llm::kernels::initializeHelixWorkspace(local_workspace_ptr, cp_size, stream);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("alltoall_helix(Tensor[] input_list, int[] group, int? num_lists) -> Tensor[]");
    m.def(
        "alltoall_helix_native(Tensor partial_o, Tensor softmax_stats, Tensor(a!) workspace, int "
        "cp_rank, int cp_size) -> (Tensor, Tensor)");
    m.def(
        "initialize_helix_workspace(Tensor(a!) workspace, int cp_rank, int cp_size) "
        "-> ()");
    m.def(
        "ulysses_alltoall_hybrid_symm(Tensor input, int[] group, "
        "__torch__.torch.classes.c10d.ProcessGroup pg, int self_copy_stream) -> Tensor");
    m.def(
        "ulysses_lsa_barrier(int[] group, "
        "__torch__.torch.classes.c10d.ProcessGroup pg) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("alltoall_helix", &tensorrt_llm::torch_ext::alltoall_helix);
    m.impl("alltoall_helix_native", &tensorrt_llm::torch_ext::alltoall_helix_native);
    m.impl("initialize_helix_workspace", &tensorrt_llm::torch_ext::initialize_helix_workspace);
    m.impl("ulysses_alltoall_hybrid_symm",
        &tensorrt_llm::torch_ext::ulysses_alltoall_hybrid_symm);
}

// ulysses_lsa_barrier takes no tensor inputs, so the dispatcher can't pick a
// backend from input types. Register on CompositeExplicitAutograd so it works
// regardless of the caller's intended device (the op always operates on CUDA).
TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("ulysses_lsa_barrier", &tensorrt_llm::torch_ext::ulysses_lsa_barrier);
}
