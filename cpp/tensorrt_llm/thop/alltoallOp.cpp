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
#include "tensorrt_llm/kernels/ulyssesPermuteScatterKernel.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
#include <nccl_device.h> // ncclGetPeerDevicePointer (host-side window peer-VA query)
#endif

#include <array>
#include <cctype>
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

// Opaque handle returned by `ulysses_a2a_async_prepare` and consumed by
// `ulysses_a2a_async`. Hides the raw send_buf pointer + per-peer recv-buf
// pointers + per-peer slot byte size from the Python API — callers just plumb
// the handle between the two ops. The `send_t` field keeps the NCCL window
// slot pair view alive across the two op calls (the underlying storage is a
// ring slot in the AsyncUlyssesOp, but holding a Tensor wrapper makes its
// lifetime explicit to the Python GC).
struct SendHandle : torch::CustomClassHolder
{
    torch::Tensor send_t;                // Tensor view of slot.sendBuf (kept alive)
    std::vector<int64_t> peer_recv_ptrs; // [P] int64-cast device pointers
    int64_t slot_bytes;                  // per-peer chunk byte size
};

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
        TLLM_CHECK_WITH_INFO(bytes.size() == sizeof(ncclUniqueId), "store->get returned %zu bytes, expected %zu",
            bytes.size(), sizeof(ncclUniqueId));
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
std::shared_ptr<ncclComm_t> getCommForCe(std::set<int> const& group, c10::intrusive_ptr<c10d::ProcessGroup> const& pg)
{
    TLLM_CHECK_WITH_INFO(pg,
        "alltoall_ce / get_ce_registered_buffer require a torch ProcessGroup — the MPI bootstrap path has been "
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

// Symmetric-memory backend selector. Two backends are kept side-by-side so
// the runtime data path (CE push + barrier) can be A/B-tested:
//   - NCCL_WINDOW    : original path — ncclMemAlloc + ncclCommWindowRegister
//                      + ncclGetPeerDevicePointer + NCCL device-API LSA
//                      barrier. Needs NCCL 2.28+ runtime.
//   - PYTORCH_CUDA_IPC : PyTorch _SymmetricMemory CUDA-IPC backend —
//                      empty_strided_p2p + rendezvous + buffer_ptrs +
//                      handle->barrier(channel). No NCCL version dep.
// Env var TLLM_SYMM_MEM_BACKEND={nccl,pytorch} selects; default nccl.
enum class SymmMemBackend
{
    NCCL_WINDOW,
    PYTORCH_CUDA_IPC,
};

static SymmMemBackend getSymmMemBackendFromEnv()
{
    char const* v = std::getenv("TLLM_SYMM_MEM_BACKEND");
    if (v == nullptr || std::strlen(v) == 0)
    {
        return SymmMemBackend::NCCL_WINDOW;
    }
    std::string s(v);
    for (auto& c : s)
    {
        c = static_cast<char>(std::tolower(c));
    }
    if (s == "pytorch" || s == "pt" || s == "torch")
    {
        return SymmMemBackend::PYTORCH_CUDA_IPC;
    }
    return SymmMemBackend::NCCL_WINDOW;
}

// ============================================================================
// AsyncUlyssesOp: async Ulysses A2A using a symmetric-memory window +
// CUDA C permute+scatter + cudaMemcpyBatchAsync peer push + LSA-style
// release barrier. Each rank's self segment is written directly to its
// symm-memory recv slot by the scatter kernel; peer segments are pushed to
// each peer's slot via batched CE memcpy; cross-rank fence is the barrier.
// Symm-memory backend is runtime-selectable (see SymmMemBackend above).
// ============================================================================

class AsyncUlyssesOp
{
public:
    AsyncUlyssesOp(std::set<int> group, c10::intrusive_ptr<c10d::ProcessGroup> pg)
        : mGroup(std::move(group))
        , mPg(std::move(pg))
        , mBackend(getSymmMemBackendFromEnv())
    {
    }

    int initialize()
    {
        TLLM_CHECK_WITH_INFO(mPg, "AsyncUlyssesOp requires a torch ProcessGroup");
        TLLM_CHECK_WITH_INFO(!mGroup.empty(), "group must be non-empty");
        TLLM_CHECK_WITH_INFO(static_cast<int>(mGroup.size()) == mPg->getSize(),
            "group size (%d) must match ProcessGroup size (%d). Pass a sub-group covering exactly the requested ranks.",
            static_cast<int>(mGroup.size()), mPg->getSize());

        TLLM_LOG_INFO("AsyncUlyssesOp: symm-mem backend = %s",
            mBackend == SymmMemBackend::NCCL_WINDOW ? "NCCL_WINDOW" : "PYTORCH_CUDA_IPC");

        if (mBackend == SymmMemBackend::PYTORCH_CUDA_IPC)
        {
            // PyTorch _SymmetricMemory CUDA-IPC backend handles peer-access
            // setup internally during rendezvous(). No NCCL comm bootstrap,
            // no manual cudaDeviceEnablePeerAccess loop needed. Slot ring is
            // populated lazily in getWindowSlot().
            return 0;
        }

        // ---- NCCL_WINDOW backend ----
        // Reuse the PG-based NCCL comm bootstrap. The comm is only used for
        // the device-API LSA barrier (`ncclLsaBarrierSession`); data transport
        // goes through cudaMemcpyBatchAsync + CUDA C kernel, bypassing NCCL.
        mNcclComm = getCommForCe(mGroup, mPg);

        int const pSize = mPg->getSize();
        int const pgRank = mPg->getRank();
        int myDev = -1;
        TLLM_CUDA_CHECK(cudaGetDevice(&myDev));

        // Exchange each rank's CUDA device id via Store — needed to call
        // cudaDeviceEnablePeerAccess on the correct peer devices for NVLink
        // P2P. Rank-to-device mapping is typically rank==device on single-node
        // but we don't assume it.
        auto store = mPg->getStore();
        TLLM_CHECK_WITH_INFO(store, "ProcessGroup has no Store");
        std::ostringstream keyPrefix;
        keyPrefix << "trtllm_async_ulysses_dev";
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
            TLLM_CHECK_WITH_INFO(bytes.size() == sizeof(int), "Store returned %zu bytes for device id, expected %zu",
                bytes.size(), sizeof(int));
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
                TLLM_CHECK_WITH_INFO(
                    false, "cudaDeviceEnablePeerAccess(dev=%d) failed: %s", mPeerDevIds[r], cudaGetErrorString(rc));
            }
            // Consume any sticky error from "already enabled" so later checks are clean.
            (void) cudaGetLastError();
        }

        return 0;
    }

    int getPgSize() const
    {
        return mPg->getSize();
    }

    // Emit a single release-ordered LSA barrier on `stream`. Lazy-creates
    // the underlying ncclDevComm on first call. No-op if NCCL <2.28 or
    // comm lacks device API support (caller responsible for fallback to
    // PyTorch SymMem barrier).
    void emitLsaBarrier(cudaStream_t stream)
    {
        if (mBackend == SymmMemBackend::PYTORCH_CUDA_IPC)
        {
            // PT _SymmetricMemory barrier picks up at::cuda::getCurrentCUDAStream()
            // internally. Pick any allocated slot's handle (all slots in this op
            // share the same group / barrier channel space). Caller is responsible
            // for ensuring the current stream is the one we want the barrier on.
            c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> h;
            {
                std::lock_guard<std::mutex> lock(mSlotsMutex);
                for (auto const& s : mSlots)
                {
                    if (s.pt_handle)
                    {
                        h = s.pt_handle;
                        break;
                    }
                }
            }
            TLLM_CHECK_WITH_INFO(h,
                "emitLsaBarrier (PYTORCH_CUDA_IPC): no slot has been allocated yet — "
                "rendezvous must precede the first barrier.");
            (void) stream; // unused: PT barrier uses at::cuda::getCurrentCUDAStream()
            h->barrier(/*channel=*/0, /*timeout_ms=*/10000);
            return;
        }

        // ---- NCCL_WINDOW backend ----
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
        TLLM_CHECK_WITH_INFO(
            mLsaBarrier != nullptr, "LsaBarrier::create failed — NCCL <2.28 or comm lacks deviceApiSupport.");
        mLsaBarrier->emit(stream);
    }

    int getPgRank() const
    {
        return mPg->getRank();
    }

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    // Acquire one ring-slot's send + recv tensor views (typed + shaped, no
    // Python-side view(dtype)/view(shape) chain — those trigger inductor
    // materialize-clones under cuda_graph capture).
    //
    // Returns (send_t, recv_t, peer_recv_ptrs, slot_bytes):
    //   send_t          : Tensor view of slot.sendBuf (local push source)
    //   recv_t          : Tensor view of slot.basePtr (symm-mem recv slot)
    //   peer_recv_ptrs  : [P] vector of peer-i.basePtr pointers (CE push targets)
    //   slot_bytes      : per-peer chunk size in bytes (= numel*elem_size / P)
    std::tuple<torch::Tensor, torch::Tensor, std::vector<int64_t>, int64_t> acquire_slot_pair(
        at::IntArrayRef shape, c10::ScalarType dtype)
    {
        TLLM_CHECK_WITH_INFO(mPg, "AsyncUlyssesOp not initialized (initialize() must be called first)");

        int64_t const elem_size = static_cast<int64_t>(c10::elementSize(dtype));
        TORCH_CHECK(elem_size > 0, "dtype must have positive itemsize");
        int64_t numel = 1;
        for (auto d : shape)
        {
            TORCH_CHECK(d > 0, "shape dims must be positive");
            numel *= d;
        }
        int64_t const bufferBytes = numel * elem_size;
        TORCH_CHECK(bufferBytes > 0, "bufferBytes must be positive");

        int const pSize = mPg->getSize();
        TORCH_CHECK(bufferBytes % pSize == 0, "bufferBytes must be divisible by world_size");

        int slotIdx = nextSlot();
        WindowSlot& slot = getWindowSlot(slotIdx, static_cast<size_t>(bufferBytes));

        auto opts = torch::dtype(dtype).device(torch::kCUDA);
        auto send_t = torch::from_blob(
            slot.sendBuf, shape, /*deleter=*/[](void*) {}, opts);
        auto recv_t = torch::from_blob(
            slot.basePtr, shape, /*deleter=*/[](void*) {}, opts);

        std::vector<int64_t> peer_recv_ptrs(pSize);
        for (int p = 0; p < pSize; ++p)
        {
            peer_recv_ptrs[p] = reinterpret_cast<int64_t>(slot.peerPtrs[p]);
        }

        int64_t slot_bytes = bufferBytes / pSize;
        return std::make_tuple(send_t, recv_t, std::move(peer_recv_ptrs), slot_bytes);
    }

    // CE peer push: out-of-capture uses cudaMemcpyBatchAsync (multi-CE
    // engine fan-out); under stream capture we serialize via per-peer
    // cudaMemcpyAsync (cudaMemcpyBatchAsync is not graph-capture-safe).
    // Self chunk is NOT pushed (already written by the upstream
    // fused-permute kernel into recv_buf[my_rank]).
    void run_a2a_ce_push(torch::Tensor send_buf, std::vector<int64_t> const& peer_recv_ptrs, int64_t slot_bytes)
    {
        TLLM_CHECK_WITH_INFO(mPg, "AsyncUlyssesOp not initialized (initialize() must be called first)");
        int const pSize = mPg->getSize();
        int const pgRank = mPg->getRank();
        TORCH_CHECK(static_cast<int>(peer_recv_ptrs.size()) == pSize, "peer_recv_ptrs size must equal world_size");

        int const nPeers = pSize - 1;
        if (nPeers == 0)
        {
            // P=1: self-only, recv_buf already populated by the upstream
            // fused-permute kernel; nothing to push.
            return;
        }

        char const* sendBase = static_cast<char const*>(send_buf.data_ptr());
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        cudaStreamCaptureStatus captureStatus;
        TLLM_CUDA_CHECK(cudaStreamIsCapturing(stream, &captureStatus));
        bool const underCapture = (captureStatus != cudaStreamCaptureStatusNone);

        if (!underCapture)
        {
            std::vector<void*> dsts;
            dsts.reserve(nPeers);
            std::vector<void const*> srcs;
            srcs.reserve(nPeers);
            std::vector<size_t> sizes;
            sizes.reserve(nPeers);
            for (int p = 0; p < pSize; ++p)
            {
                if (p == pgRank)
                    continue;
                void* peerBase = reinterpret_cast<void*>(peer_recv_ptrs[p]);
                dsts.push_back(
                    static_cast<char*>(peerBase) + static_cast<size_t>(pgRank) * static_cast<size_t>(slot_bytes));
                srcs.push_back(sendBase + static_cast<size_t>(p) * static_cast<size_t>(slot_bytes));
                sizes.push_back(static_cast<size_t>(slot_bytes));
            }
            cudaMemcpyAttributes attrs[1];
            std::memset(&attrs[0], 0, sizeof(attrs[0]));
            attrs[0].srcAccessOrder = cudaMemcpySrcAccessOrderStream;
            attrs[0].flags = 1u;
            size_t attrIdxs[1] = {0};
            TLLM_CUDA_CHECK(cudaMemcpyBatchAsync(
                dsts.data(), srcs.data(), sizes.data(), static_cast<size_t>(nPeers), attrs, attrIdxs, 1, stream));
        }
        else
        {
            // Capture-safe serial loop.
            for (int p = 0; p < pSize; ++p)
            {
                if (p == pgRank)
                    continue;
                void* peerBase = reinterpret_cast<void*>(peer_recv_ptrs[p]);
                void* dst
                    = static_cast<char*>(peerBase) + static_cast<size_t>(pgRank) * static_cast<size_t>(slot_bytes);
                void const* src = sendBase + static_cast<size_t>(p) * static_cast<size_t>(slot_bytes);
                TLLM_CUDA_CHECK(
                    cudaMemcpyAsync(dst, src, static_cast<size_t>(slot_bytes), cudaMemcpyDeviceToDevice, stream));
            }
        }
    }
#endif // NCCL_VERSION_CODE >= 2.28

private:
    static constexpr int kNumSlots = 4;

    // Register the PG's (group_name, rank, world_size, store) with PyTorch's
    // symm-mem registry exactly once per process per group. Subsequent
    // empty_strided_p2p + rendezvous calls in this group reuse it.
    void ensureSymmMemGroupInfo()
    {
        static std::set<std::string> sRegistered;
        static std::mutex sMutex;
        std::string const& name = mPg->getGroupName();
        std::lock_guard<std::mutex> lock(sMutex);
        if (sRegistered.count(name))
        {
            return;
        }
        c10d::symmetric_memory::set_group_info(name, mPg->getRank(), mPg->getSize(), mPg->getStore());
        sRegistered.insert(name);
    }

    // Ring counter for output slot selection. Pipelined invocations must not
    // collide on the same slot before the previous one has consumed it.
    int nextSlot()
    {
        std::lock_guard<std::mutex> lock(mNextSlotMutex);
        int slot = mNextSlot;
        mNextSlot = (mNextSlot + 1) % kNumSlots;
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
        void* basePtr = nullptr;     // NCCL: ncclMemAlloc'd buf. PT: pt_tensor.data_ptr().
        ncclWindow_t win = nullptr;  // NCCL-only: SYMMETRIC-registered window handle.
        size_t size = 0;
        std::vector<void*> peerPtrs; // size == pSize; peer-write pointers (both backends).

        // Per-slot local send buffer. Same byte size as basePtr (the recv
        // buffer in our peer-WRITE scheme). Local-only (no symm-mem registration);
        // cudaMalloc'd eagerly in getWindowSlot to stay cuda_graph-capture-safe.
        void* sendBuf = nullptr;
        size_t sendBufBytes = 0;

        // PT-only: keep storage + handle alive across calls. basePtr aliases
        // pt_tensor.data_ptr() and peerPtrs[i] comes from pt_handle->buffer_ptrs.
        at::Tensor pt_tensor;
        c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> pt_handle;
    };

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

        std::lock_guard<std::mutex> lock(mSlotsMutex);
        auto& slot = mSlots[slotIdx];

        if (mBackend == SymmMemBackend::PYTORCH_CUDA_IPC)
        {
            // ---- PT _SymmetricMemory CUDA-IPC backend ----
            if (slot.basePtr != nullptr && slot.size >= requiredSize)
            {
                return slot;
            }

            // Size-up: release previous storage by dropping refs.
            if (slot.basePtr != nullptr)
            {
                slot.pt_handle.reset();
                slot.pt_tensor = at::Tensor();
                slot.basePtr = nullptr;
                slot.size = 0;
                slot.peerPtrs.clear();
            }

            ensureSymmMemGroupInfo();
            int64_t const nbytes = static_cast<int64_t>(requiredSize);
            int currentDev = -1;
            TLLM_CUDA_CHECK(cudaGetDevice(&currentDev));
            c10::Device device(c10::DeviceType::CUDA, currentDev);
            std::string const& groupName = mPg->getGroupName();

            slot.pt_tensor = c10d::symmetric_memory::empty_strided_p2p(
                /*size=*/{nbytes}, /*stride=*/{1}, /*dtype=*/at::kByte, device,
                /*group_name=*/std::make_optional(groupName), /*alloc_id=*/std::nullopt);

            // Collective; all ranks must reach in the same order.
            slot.pt_handle = c10d::symmetric_memory::rendezvous(slot.pt_tensor, groupName);
            TLLM_CHECK_WITH_INFO(slot.pt_handle, "rendezvous returned null handle");

            slot.basePtr = slot.pt_tensor.data_ptr();
            slot.size = requiredSize;

            auto ptrs = slot.pt_handle->get_buffer_ptrs();
            TLLM_CHECK_WITH_INFO(static_cast<int>(ptrs.size()) == pSize, "get_buffer_ptrs size %zu != world_size %d",
                ptrs.size(), pSize);
            slot.peerPtrs.assign(pSize, nullptr);
            for (int p = 0; p < pSize; ++p)
            {
                slot.peerPtrs[p] = ptrs[p];
            }

            // sendBuf: same eager cudaMalloc as NCCL path (capture-safe).
            if (slot.sendBuf != nullptr && slot.sendBufBytes < requiredSize)
            {
                (void) cudaFree(slot.sendBuf);
                slot.sendBuf = nullptr;
                slot.sendBufBytes = 0;
            }
            if (slot.sendBuf == nullptr)
            {
                TLLM_CUDA_CHECK(cudaMalloc(&slot.sendBuf, requiredSize));
                slot.sendBufBytes = requiredSize;
            }

            return slot;
        }

        // ---- NCCL_WINDOW backend ----

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
            "NCCLWindowAllocator::requestBuffer failed (size=%zu). NCCL <2.28 or registration error.", requiredSize);

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

        // Eagerly allocate the per-slot local sendBuf alongside basePtr so
        // acquire_slot_pair never has to lazy-cudaMalloc on first use.
        // cudaMalloc is not allowed under CUDA Graph capture (autotuner runs
        // forward under capture); doing it here, during getWindowSlot which
        // runs out-of-capture during init/warmup, keeps the data path
        // capture-safe.
        if (slot.sendBuf != nullptr && slot.sendBufBytes < requiredSize)
        {
            (void) cudaFree(slot.sendBuf);
            slot.sendBuf = nullptr;
            slot.sendBufBytes = 0;
        }
        if (slot.sendBuf == nullptr)
        {
            TLLM_CUDA_CHECK(cudaMalloc(&slot.sendBuf, requiredSize));
            slot.sendBufBytes = requiredSize;
        }

        return slot;
    }
#endif // NCCL_VERSION_CODE >= 2.28

    std::set<int> mGroup;
    c10::intrusive_ptr<c10d::ProcessGroup> mPg;
    SymmMemBackend mBackend;
    std::shared_ptr<ncclComm_t> mNcclComm;
    std::vector<int> mPeerDevIds;

    int mNextSlot{0};
    std::mutex mNextSlotMutex;

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    std::array<WindowSlot, kNumSlots> mSlots{};
    std::mutex mSlotsMutex;
#endif

    // NCCL 2.28+ device-API LSA barrier. Lazily created on first
    // emitLsaBarrier() call to avoid `ncclDevCommCreate` cost when not used.
    std::unique_ptr<tensorrt_llm::kernels::LsaBarrier> mLsaBarrier;
    std::mutex mLsaBarrierMutex;
};

// Process-lifetime cache of AsyncUlyssesOp instances keyed by rank-set.
// Both thop wrappers go through this; first call collectively initializes.
static std::shared_ptr<AsyncUlyssesOp> getOrCreateAsyncOp(
    std::set<int> const& group, c10::intrusive_ptr<c10d::ProcessGroup> const& pg)
{
    static std::map<std::set<int>, std::shared_ptr<AsyncUlyssesOp>> sOpCache;
    static std::mutex sOpMutex;
    std::lock_guard<std::mutex> lock(sOpMutex);
    auto it = sOpCache.find(group);
    if (it != sOpCache.end())
    {
        return it->second;
    }
    auto op = std::make_shared<AsyncUlyssesOp>(group, pg);
    op->initialize();
    sOpCache[group] = op;
    return op;
}

// ─────────────────────────────────────────────────────────────────────────
// Fused 2-op Ulysses async A2A pipeline.
// Pairs with `_pre_attn_alltoall_async` in
// tensorrt_llm/_torch/visual_gen/attention_backend/parallel.py:
//
//   recv_5d, send_h = ulysses_a2a_async_prepare(input_4d, group, pg)
//   ev.record()
//   with torch.cuda.stream(comm_stream):
//       ev.wait()
//       ulysses_a2a_async(send_h, group, pg)
//
// Replaces the 4-op chain (`acquire_buffers` + Triton scatter + `push` +
// `barrier`). `SendHandle` hides peer_recv_ptrs / slot_bytes from Python.
// ─────────────────────────────────────────────────────────────────────────

// Step 1 (caller's compute stream): acquire slot ring entry + CUDA C
// permute+scatter (writes peer chunks to send_buf, self chunk directly to
// recv_buf[my_rank]). Returns the 5D recv-buf view (for downstream SDPA)
// and an opaque `SendHandle` that the second op consumes.
std::tuple<torch::Tensor, c10::intrusive_ptr<SendHandle>> ulysses_a2a_async_prepare(
    torch::Tensor input_4d, torch::List<int64_t> group_, c10::intrusive_ptr<c10d::ProcessGroup> const& pg)
{
#if ENABLE_MULTI_DEVICE
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    TORCH_CHECK(input_4d.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input_4d.dim() == 4, "input must be [B, S_local, H, D]");
    TORCH_CHECK(input_4d.scalar_type() == at::ScalarType::BFloat16, "bf16 only");

    int const B = static_cast<int>(input_4d.size(0));
    int const S_local = static_cast<int>(input_4d.size(1));
    int const H = static_cast<int>(input_4d.size(2));
    int const D = static_cast<int>(input_4d.size(3));
    TORCH_CHECK(D % 8 == 0, "D must be divisible by 8 (int4 vec)");

    std::set<int> group;
    for (int64_t rank : group_)
    {
        group.insert(static_cast<int>(rank));
    }
    auto op = getOrCreateAsyncOp(group, pg);

    int const P = op->getPgSize();
    int const my_rank = op->getPgRank();
    TORCH_CHECK(H % P == 0, "H must be divisible by world_size");
    int const H_local = H / P;

    // Acquire 5D send + recv slot views (peer-WRITE layout matching CUDA C kernel).
    auto [send_t, recv_t, peer_recv_ptrs, slot_bytes] = op->acquire_slot_pair(
        {(int64_t) P, (int64_t) B, (int64_t) S_local, (int64_t) H_local, (int64_t) D}, input_4d.scalar_type());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(input_4d.get_device()).stream();
    tensorrt_llm::kernels::launchUlyssesPermuteScatter(
        input_4d.data_ptr(), send_t.data_ptr(), recv_t.data_ptr(), my_rank, B, S_local, H, D, P, stream);

    auto send_h = c10::make_intrusive<SendHandle>();
    send_h->send_t = std::move(send_t);
    send_h->peer_recv_ptrs = std::move(peer_recv_ptrs);
    send_h->slot_bytes = slot_bytes;

    return std::make_tuple(std::move(recv_t), send_h);
#else
    TORCH_CHECK(false, "ulysses_a2a_async_prepare requires NCCL >= 2.28");
    return std::make_tuple(torch::Tensor(), c10::intrusive_ptr<SendHandle>());
#endif
#else
    TORCH_CHECK(false, "ulysses_a2a_async_prepare requires ENABLE_MULTI_DEVICE");
    return std::make_tuple(torch::Tensor(), c10::intrusive_ptr<SendHandle>());
#endif
}

// Step 2 (caller's comm stream): fire P-1 cudaMemcpyBatchAsync peer pushes,
// then emit the LSA barrier — both on the current stream. Caller is expected
// to event-sync from the compute stream onto the comm stream before calling.
void ulysses_a2a_async(c10::intrusive_ptr<SendHandle> const& send_h, torch::List<int64_t> group_,
    c10::intrusive_ptr<c10d::ProcessGroup> const& pg)
{
#if ENABLE_MULTI_DEVICE
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    TORCH_CHECK(send_h.get() != nullptr, "send_h is null");
    TORCH_CHECK(send_h->send_t.defined(), "send_h.send_t is undefined");

    std::set<int> group;
    for (int64_t rank : group_)
    {
        group.insert(static_cast<int>(rank));
    }
    auto op = getOrCreateAsyncOp(group, pg);
    op->run_a2a_ce_push(send_h->send_t, send_h->peer_recv_ptrs, send_h->slot_bytes);
    op->emitLsaBarrier(at::cuda::getCurrentCUDAStream().stream());
#else
    TORCH_CHECK(false, "ulysses_a2a_async requires NCCL >= 2.28");
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

/**
 * Initialize workspace for helix all-to-all
 */
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

    // Opaque handle returned by prepare, consumed by async. Hides
    // peer_recv_ptrs / slot_bytes / send_buf raw pointer from Python.
    m.class_<tensorrt_llm::torch_ext::SendHandle>("SendHandle");

    // Fused 2-op Ulysses async A2A pipeline (CUDA C permute+scatter +
    // cudaMemcpyBatchAsync push + LSA barrier).
    m.def(
        "ulysses_a2a_async_prepare(Tensor input, int[] group, "
        "__torch__.torch.classes.c10d.ProcessGroup pg) "
        "-> (Tensor, __torch__.torch.classes.trtllm.SendHandle)");
    m.def(
        "ulysses_a2a_async(__torch__.torch.classes.trtllm.SendHandle send_h, "
        "int[] group, __torch__.torch.classes.c10d.ProcessGroup pg) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("alltoall_helix", &tensorrt_llm::torch_ext::alltoall_helix);
    m.impl("alltoall_helix_native", &tensorrt_llm::torch_ext::alltoall_helix_native);
    m.impl("initialize_helix_workspace", &tensorrt_llm::torch_ext::initialize_helix_workspace);
}

// Both ops take/return a custom-class handle, not tensors, so the dispatcher
// can't pick a backend from input types. Register on CompositeExplicitAutograd
// (the underlying CUDA work runs on the caller's current CUDA stream).
TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("ulysses_a2a_async_prepare", &tensorrt_llm::torch_ext::ulysses_a2a_async_prepare);
    m.impl("ulysses_a2a_async", &tensorrt_llm::torch_ext::ulysses_a2a_async);
}
