/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
//
// Async Ulysses A2A — PyTorch _SymmetricMemory CUDA-IPC backend.
//
// Pipeline (paired with UlyssesAttention.forward_async in
// tensorrt_llm/_torch/visual_gen/attention_backend/parallel.py):
//
//   recv, send_h = ulysses_a2a_async_prepare(input, pg)   # default stream
//   ev.record()
//   with torch.cuda.stream(comm_stream):
//       ev.wait()
//       ulysses_a2a_async(send_h, pg)                     # CE push + barrier
//
// Phase 1 (`_prepare`) on the caller's compute stream:
//   - lazily allocate one slot of a ring of P-symmetric-memory buffers
//     via empty_strided_p2p + rendezvous (PyTorch CUDA-IPC backend);
//   - launch the fused permute+scatter kernel into (slot.sendBuf for
//     peer chunks, slot.basePtr+my_rank for self chunk);
//   - return the 5D recv view and an opaque SendHandle.
//
// Phase 2 (`_async`) on the comm stream:
//   - cudaMemcpyBatchAsync (capture-safe per-peer loop fallback) pushes
//     each peer's slice of sendBuf into peer.basePtr[my_rank];
//   - PT symm-mem `barrier(channel, timeout_ms)` is the cross-rank fence.
//
// No NCCL device API; no LSA barrier kernel.
//

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/ulyssesPermuteScatterKernel.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <array>
#include <cstring>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Opaque handle returned by `_prepare`, consumed by `_async`. Hides raw
// pointer plumbing from Python; `send_t` keeps the slot's sendBuf tensor
// view alive across the two op calls.
//
// `group_name` binds the handle to the PG that produced it: peer_recv_ptrs
// are valid only in that PG's symm-mem registration. `_async` rejects any
// PG whose group name doesn't match — two distinct PGs of the same size
// would otherwise pass the peer-pointer-count check and silently push
// into the wrong group's buffers.
struct SendHandle : torch::CustomClassHolder
{
    torch::Tensor send_t;
    std::vector<int64_t> peer_recv_ptrs;
    int64_t slot_bytes;
    std::string group_name;
};

#if ENABLE_MULTI_DEVICE

namespace
{

class AsyncUlyssesOp
{
public:
    // Slot ring depth. Minimum 3 = one slot each for V/Q/K within a single
    // forward_async call. A slot is touched by 4 ops in sequence:
    //   (a) default-stream Phase-1 write  — permute+scatter into slot.sendBuf
    //   (b) side-stream Phase-2 CE push   — reads slot.sendBuf
    //   (c) side-stream Phase-2 barrier   — peer writes into slot.recv
    //   (d) default-stream SDPA read       — reads slot.recv
    // Intra-layer hazard: V/Q/K must use distinct slots, otherwise (a) on the
    // default stream races (b) on the side stream — they touch the same
    // sendBuf and there is no stream sync between them until _join_async.
    // Cross-layer hazard (Layer N+1 V reusing Layer N V's slot): safe because
    // _join_async at end of Layer N waits the side stream's K barrier event,
    // and SDPA on the default stream drains the recv read before Layer N+1
    // starts. So kNumSlots = 3 is the tight minimum.
    static constexpr int kNumSlots = 3;

    explicit AsyncUlyssesOp(c10::intrusive_ptr<c10d::ProcessGroup> pg)
        : mPg(std::move(pg))
    {
    }

    void initialize()
    {
        TLLM_CHECK_WITH_INFO(mPg, "AsyncUlyssesOp requires a torch ProcessGroup");
        TLLM_CHECK_WITH_INFO(mPg->getSize() >= 1, "ProcessGroup size must be >= 1");
        // Register the PG's group_info with PT symm-mem (one-shot per process per group).
        ensureGroupRegistered();
    }

    int getPgSize() const
    {
        return mPg->getSize();
    }

    int getPgRank() const
    {
        return mPg->getRank();
    }

    // Phase 1: lazy-alloc next ring slot via PT symm-mem; return tensor
    // views over send_buf (local push source) and recv_buf (peer-writable)
    // plus the host-side peer-pointer array.
    std::tuple<torch::Tensor, torch::Tensor, std::vector<int64_t>, int64_t> acquireSlotPair(
        at::IntArrayRef shape, c10::ScalarType dtype)
    {
        int64_t const elemSize = static_cast<int64_t>(c10::elementSize(dtype));
        TORCH_CHECK(elemSize > 0, "dtype must have positive itemsize");
        int64_t numel = 1;
        for (auto d : shape)
        {
            TORCH_CHECK(d > 0, "shape dims must be positive");
            numel *= d;
        }
        int64_t const bufferBytes = numel * elemSize;
        TORCH_CHECK(bufferBytes > 0, "bufferBytes must be positive");
        int const pSize = mPg->getSize();
        TORCH_CHECK(bufferBytes % pSize == 0, "bufferBytes must be divisible by world_size");

        int const slotIdx = nextSlotIdx();
        Slot& slot = getOrAllocSlot(slotIdx, static_cast<size_t>(bufferBytes));

        auto opts = torch::dtype(dtype).device(torch::kCUDA);
        auto sendT = torch::from_blob(
            slot.sendBuf, shape, /*deleter=*/[](void*) {}, opts);
        auto recvT = torch::from_blob(
            slot.basePtr, shape, /*deleter=*/[](void*) {}, opts);

        std::vector<int64_t> peerRecvPtrs(pSize);
        for (int p = 0; p < pSize; ++p)
        {
            peerRecvPtrs[p] = reinterpret_cast<int64_t>(slot.peerPtrs[p]);
        }

        int64_t const slotBytes = bufferBytes / pSize;
        return std::make_tuple(sendT, recvT, std::move(peerRecvPtrs), slotBytes);
    }

    // Phase 2 (data): out-of-capture uses cudaMemcpyBatchAsync (multi-CE
    // engine fan-out); under stream capture we serialize via per-peer
    // cudaMemcpyAsync (cudaMemcpyBatchAsync is not graph-capture-safe).
    // Self chunk is NOT pushed (already written by the upstream
    // fused-permute kernel into recv_buf[my_rank]).
    void runCePush(torch::Tensor send_buf, std::vector<int64_t> const& peer_recv_ptrs, int64_t slot_bytes)
    {
        int const pSize = mPg->getSize();
        int const pgRank = mPg->getRank();
        TORCH_CHECK(static_cast<int>(peer_recv_ptrs.size()) == pSize, "peer_recv_ptrs size must equal world_size");

        int const nPeers = pSize - 1;
        if (nPeers == 0)
        {
            // P=1: self-only; recv_buf already populated by the permute kernel.
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

    // Phase 2 (fence): PT symm-mem barrier on the current CUDA stream.
    // Any allocated slot's handle works — they all belong to the same group.
    void emitBarrier()
    {
        c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> h;
        {
            std::lock_guard<std::mutex> lock(mSlotsMutex);
            for (auto const& s : mSlots)
            {
                if (s.handle)
                {
                    h = s.handle;
                    break;
                }
            }
        }
        TLLM_CHECK_WITH_INFO(h, "emitBarrier: no slot allocated yet — _prepare must precede the first _async barrier.");
        // 10s timeout: on hang, the kernel traps with rank+channel diagnostic instead of spinning silently
        // until SLURM wall-clock kills. Generous enough to absorb first-touch IPC + first cuda_graph
        // capture jitter. channel=0: V/Q/K issues all run on the same per-device side stream so
        // they FIFO-serialize; channel multiplexing only matters across distinct streams.
        h->barrier(/*channel=*/0, /*timeout_ms=*/10000);
    }

private:
    struct Slot
    {
        // PT _SymmetricMemory-backed recv buffer (peer-writable).
        at::Tensor symm_tensor;
        c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> handle;
        void* basePtr = nullptr;     // aliases symm_tensor.data_ptr()
        size_t size = 0;
        std::vector<void*> peerPtrs; // from handle->get_buffer_ptrs()

        // Local-only push source (no symm-mem). cudaMalloc'd eagerly to
        // stay cuda_graph-capture-safe.
        void* sendBuf = nullptr;
        size_t sendBufBytes = 0;
    };

    // One-shot per process per group: register PG's (name, rank, size, store)
    // with PT symm-mem's group registry. Subsequent rendezvous() calls reuse it.
    void ensureGroupRegistered()
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

    int nextSlotIdx()
    {
        std::lock_guard<std::mutex> lock(mNextMutex);
        int idx = mNextIdx;
        mNextIdx = (mNextIdx + 1) % kNumSlots;
        return idx;
    }

    // Lazy collective allocator. Cached when size is sufficient; reallocates
    // (releasing the old handle) on size-up. All ranks must reach this in the
    // same order (collective rendezvous).
    //
    // Commit-on-success: every allocation step writes to local variables
    // first, and the cached `slot` is mutated only after all steps succeed.
    // If `empty_strided_p2p`, `rendezvous`, `get_buffer_ptrs`, or `cudaMalloc`
    // throws mid-way, the local at::Tensor / intrusive_ptr clean up via RAII
    // and the previously-cached slot remains untouched (so the next call
    // either retries or reuses the still-valid prior state).
    Slot& getOrAllocSlot(int slotIdx, size_t requiredSize)
    {
        std::lock_guard<std::mutex> lock(mSlotsMutex);
        Slot& slot = mSlots[slotIdx];

        if (slot.basePtr != nullptr && slot.size >= requiredSize)
        {
            return slot;
        }

        // First-time / size-up allocation is NOT capture-safe:
        // empty_strided_p2p + rendezvous + cudaMalloc all violate stream
        // capture invariants. Caller must warm up out-of-capture so the slot
        // is allocated and cached before any cuda_graph capture begins.
        cudaStream_t const stream = at::cuda::getCurrentCUDAStream().stream();
        cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
        TLLM_CUDA_CHECK(cudaStreamIsCapturing(stream, &captureStatus));
        TORCH_CHECK(captureStatus == cudaStreamCaptureStatusNone,
            "async-ulysses: slot allocation (empty_strided_p2p + rendezvous + cudaMalloc) "
            "is not graph-capture-safe. Warm up the model out-of-capture (run one forward "
            "pass before enabling cuda_graph capture) so slots are cached.");

        int currentDev = -1;
        TLLM_CUDA_CHECK(cudaGetDevice(&currentDev));
        c10::Device device(c10::DeviceType::CUDA, currentDev);
        std::string const& groupName = mPg->getGroupName();
        int const pSize = mPg->getSize();

        // Build new state in local variables — no mutation of `slot` yet.
        at::Tensor newSymmTensor = c10d::symmetric_memory::empty_strided_p2p(
            /*size=*/{static_cast<int64_t>(requiredSize)}, /*stride=*/{1},
            /*dtype=*/at::kByte, device,
            /*group_name=*/std::make_optional(groupName), /*alloc_id=*/std::nullopt);
        auto newHandle = c10d::symmetric_memory::rendezvous(newSymmTensor, groupName);
        TLLM_CHECK_WITH_INFO(newHandle, "rendezvous returned null handle");

        auto ptrs = newHandle->get_buffer_ptrs();
        TLLM_CHECK_WITH_INFO(
            static_cast<int>(ptrs.size()) == pSize, "get_buffer_ptrs size %zu != world_size %d", ptrs.size(), pSize);
        std::vector<void*> newPeerPtrs(ptrs.begin(), ptrs.end());

        // cudaMalloc last so any throw above is cleaned up by newSymmTensor /
        // newHandle RAII without leaking GPU memory.
        void* newSendBuf = nullptr;
        TLLM_CUDA_CHECK(cudaMalloc(&newSendBuf, requiredSize));

        // All allocations succeeded — commit. Free old sendBuf (the raw void*
        // isn't owned by any RAII type in Slot); the at::Tensor / intrusive_ptr
        // fields are released by move-assign.
        if (slot.sendBuf != nullptr)
        {
            (void) cudaFree(slot.sendBuf);
        }
        slot.symm_tensor = std::move(newSymmTensor);
        slot.handle = std::move(newHandle);
        slot.basePtr = slot.symm_tensor.data_ptr();
        slot.size = requiredSize;
        slot.peerPtrs = std::move(newPeerPtrs);
        slot.sendBuf = newSendBuf;
        slot.sendBufBytes = requiredSize;

        return slot;
    }

    c10::intrusive_ptr<c10d::ProcessGroup> mPg;

    int mNextIdx{0};
    std::mutex mNextMutex;

    std::array<Slot, kNumSlots> mSlots{};
    std::mutex mSlotsMutex;
};

// Process-lifetime cache of AsyncUlyssesOp instances keyed by group_name.
static std::shared_ptr<AsyncUlyssesOp> getOrCreateOp(c10::intrusive_ptr<c10d::ProcessGroup> const& pg)
{
    TLLM_CHECK_WITH_INFO(pg, "ProcessGroup is null");
    static std::map<std::string, std::shared_ptr<AsyncUlyssesOp>> sCache;
    static std::mutex sMutex;
    std::string const& groupName = pg->getGroupName();
    std::lock_guard<std::mutex> lock(sMutex);
    auto it = sCache.find(groupName);
    if (it != sCache.end())
    {
        return it->second;
    }
    auto op = std::make_shared<AsyncUlyssesOp>(pg);
    op->initialize();
    sCache[groupName] = op;
    return op;
}

// Step 1 (caller's compute stream): acquire slot ring entry + CUDA C
// permute+scatter (writes peer chunks to send_buf, self chunk directly to
// recv_buf[my_rank]). Returns the 5D recv-buf view (for downstream SDPA)
// and an opaque SendHandle that the second op consumes.
std::tuple<torch::Tensor, c10::intrusive_ptr<SendHandle>> ulysses_a2a_async_prepare(
    torch::Tensor input_4d, c10::intrusive_ptr<c10d::ProcessGroup> const& pg)
{
    TORCH_CHECK(input_4d.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(input_4d.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input_4d.dim() == 4, "input must be [B, S_local, H, D]");
    TORCH_CHECK(input_4d.scalar_type() == at::ScalarType::BFloat16, "bf16 only");

    // Bind current device + slot allocator + kernel launch to the input's
    // device. `getOrAllocSlot` reads `cudaGetDevice()`, and the kernel stream
    // is taken from `input_4d.get_device()`; without this guard the two can
    // diverge (e.g. caller forgot a torch.cuda.set_device) → slot allocated
    // on dev A, kernel launched on dev B → illegal memory access.
    c10::cuda::CUDAGuard device_guard(input_4d.device());

    int const B = static_cast<int>(input_4d.size(0));
    int const S_local = static_cast<int>(input_4d.size(1));
    int const H = static_cast<int>(input_4d.size(2));
    int const D = static_cast<int>(input_4d.size(3));
    TORCH_CHECK(D % 8 == 0, "D must be divisible by 8 (int4 vec)");

    auto op = getOrCreateOp(pg);

    int const P = op->getPgSize();
    int const my_rank = op->getPgRank();
    TORCH_CHECK(H % P == 0, "H must be divisible by world_size");
    int const H_local = H / P;

    auto [send_t, recv_t, peer_recv_ptrs, slot_bytes] = op->acquireSlotPair(
        {(int64_t) P, (int64_t) B, (int64_t) S_local, (int64_t) H_local, (int64_t) D}, input_4d.scalar_type());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(input_4d.get_device()).stream();
    tensorrt_llm::kernels::launchUlyssesPermuteScatter(
        input_4d.data_ptr(), send_t.data_ptr(), recv_t.data_ptr(), my_rank, B, S_local, H, D, P, stream);

    auto send_h = c10::make_intrusive<SendHandle>();
    send_h->send_t = std::move(send_t);
    send_h->peer_recv_ptrs = std::move(peer_recv_ptrs);
    send_h->slot_bytes = slot_bytes;
    send_h->group_name = pg->getGroupName();

    return std::make_tuple(std::move(recv_t), send_h);
}

// Step 2 (caller's comm stream): fire P-1 cudaMemcpyBatchAsync peer pushes,
// then emit the symm-mem barrier — both on the current stream. Caller
// event-syncs from the compute stream onto the comm stream before calling.
void ulysses_a2a_async(c10::intrusive_ptr<SendHandle> const& send_h, c10::intrusive_ptr<c10d::ProcessGroup> const& pg)
{
    TORCH_CHECK(send_h.get() != nullptr, "send_h is null");
    TORCH_CHECK(send_h->send_t.defined(), "send_h.send_t is undefined");

    // Reject cross-PG handle use: peer_recv_ptrs are valid only in the symm-mem
    // group registered for the PG that produced this handle. Two PGs of the
    // same size would otherwise pass the peer-count check inside runCePush and
    // silently push into the wrong group's buffers.
    TORCH_CHECK(send_h->group_name == pg->getGroupName(), "SendHandle was produced by ProcessGroup '",
        send_h->group_name, "' but ulysses_a2a_async was called with ProcessGroup '", pg->getGroupName(),
        "'. Handle and PG must match.");

    auto op = getOrCreateOp(pg);
    op->runCePush(send_h->send_t, send_h->peer_recv_ptrs, send_h->slot_bytes);
    op->emitBarrier();
}

} // namespace

#endif // ENABLE_MULTI_DEVICE

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.class_<tensorrt_llm::torch_ext::SendHandle>("SendHandle");

    m.def(
        "ulysses_a2a_async_prepare(Tensor input, "
        "__torch__.torch.classes.c10d.ProcessGroup pg) "
        "-> (Tensor, __torch__.torch.classes.trtllm.SendHandle)");
    m.def(
        "ulysses_a2a_async(__torch__.torch.classes.trtllm.SendHandle send_h, "
        "__torch__.torch.classes.c10d.ProcessGroup pg) -> ()");
}

// Both ops take/return a custom-class handle, not tensors, so the dispatcher
// can't pick a backend from input types. Register on CompositeExplicitAutograd
// (the underlying CUDA work runs on the caller's current CUDA stream).
TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
#if ENABLE_MULTI_DEVICE
    m.impl("ulysses_a2a_async_prepare", &tensorrt_llm::torch_ext::ulysses_a2a_async_prepare);
    m.impl("ulysses_a2a_async", &tensorrt_llm::torch_ext::ulysses_a2a_async);
#endif
}
