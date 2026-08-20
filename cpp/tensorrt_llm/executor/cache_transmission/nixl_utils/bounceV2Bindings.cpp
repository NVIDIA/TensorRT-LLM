/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "bounceV2Bindings.h"

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce_v2/BatchedCopyPool.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce_v2/BounceArena.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce_v2/CompletionPoller.h"
#include "tensorrt_llm/executor/transferAgent.h"

#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nb = nanobind;
namespace kvc = tensorrt_llm::executor::kv_cache;

namespace tensorrt_llm::executor::kv_cache::bounce_v2
{

namespace
{

using U64Array = nb::ndarray<std::uint64_t const, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using U32Array = nb::ndarray<std::uint32_t const, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

} // namespace

void initBounceV2Bindings(nb::module_& m, nb::class_<kvc::NixlTransferAgent, kvc::BaseTransferAgent>& agentCls)
{
    // ---- CompletionPoller --------------------------------------------------------------------
    nb::class_<CompletionPoller>(m, "CompletionPoller",
        "The ONE polling thread for all async bounce completions (CUDA events + RDMA transfers). "
        "drain() returns every pending completion as one (n, 3) int64 array of (id, kind, ok).")
        .def(nb::init<std::uint32_t>(), nb::arg("poll_interval_us") = 50, nb::call_guard<nb::gil_scoped_release>())
        .def_ro_static("KIND_EVENT", &CompletionPoller::kKindEvent)
        .def_ro_static("KIND_XFER", &CompletionPoller::kKindXfer)
        .def_ro_static("FULFILL_DECLINED", &CompletionPoller::kFulfillDeclined)
        .def_ro_static("FULFILL_ARMED", &CompletionPoller::kFulfillArmed)
        .def_ro_static("FULFILL_POSTED", &CompletionPoller::kFulfillPosted)
        .def_ro_static("CANCEL_TERMINAL", &CompletionPoller::kCancelTerminal)
        .def_ro_static("CANCEL_REVERTED", &CompletionPoller::kCancelReverted)
        .def("set_wakeup_fd", &CompletionPoller::setWakeupFd, nb::arg("fd"), nb::call_guard<nb::gil_scoped_release>(),
            "Completion wakeup fd: publishing (or retiring) completions writes one 8-byte token "
            "(uint64 1 — eventfd- and pipe-compatible) to fd, non-blocking, errors ignored. Pass "
            "-1 to clear; after that returns no thread writes the old fd, so it may be closed.")
        .def("reserve_chain", &CompletionPoller::reserveChain, nb::arg("copy_id"),
            nb::call_guard<nb::gil_scoped_release>(),
            "Two-phase chain, phase 1 (TRTLLM_BOUNCE_V2_EXP_CPP_CHAIN): reserve a chain id on a "
            "still-pending copy_id BEFORE the RDMA destination is known. The event's own "
            "completion is consumed in C++ from here on: gather-ok waits for fulfill_chain_1to1 "
            "(nothing published), gather-fail publishes (reserved, KIND_EVENT, 0), shutdown "
            "publishes (reserved, KIND_XFER, 0). Returns the reserved id, or -1 when the copy_id "
            "is no longer pending — keep the classic route.")
        .def("cancel_chain", &CompletionPoller::cancelChain, nb::arg("reserved_id"),
            nb::call_guard<nb::gil_scoped_release>(),
            "Abandon a reserved-but-unfulfilled chain. Returns CANCEL_REVERTED when the gather is "
            "still pending (its completion publishes under the ORIGINAL copy_id again — re-route "
            "it), or CANCEL_TERMINAL when the reservation is already terminal (gather done or "
            "failed / shutdown: the staging region is recyclable now; any already-published "
            "failure row should find its route removed).")
        .def(
            "drain",
            [](CompletionPoller& self, int timeoutMs)
            {
                std::vector<CompletionPoller::Completion> done;
                {
                    // Release the GIL while (possibly) blocking for the first completion — the
                    // Python reactor must not stall other Python threads during its wait.
                    nb::gil_scoped_release release;
                    done = self.drain(timeoutMs);
                }
                std::size_t const n = done.size();
                auto* buf = new std::int64_t[n * 3];
                for (std::size_t i = 0; i < n; ++i)
                {
                    buf[3 * i + 0] = static_cast<std::int64_t>(done[i].id);
                    buf[3 * i + 1] = done[i].kind;
                    buf[3 * i + 2] = done[i].ok;
                }
                nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<std::int64_t*>(p); });
                return nb::ndarray<nb::numpy, std::int64_t, nb::ndim<2>>(buf, {n, 3}, owner);
            },
            nb::arg("timeout_ms"),
            "Return (and clear) ALL pending completions as an (n, 3) int64 array of (id, kind, ok), "
            "blocking up to timeout_ms for the first one (0 = non-blocking).")
        .def("shutdown", &CompletionPoller::shutdown, nb::call_guard<nb::gil_scoped_release>());

    // ---- BatchedCopyPool ---------------------------------------------------------------------
    nb::class_<BatchedCopyPool>(m, "BatchedCopyPool",
        "Gather/scatter copy engine: N greatest-priority CUDA streams with pinned zero-copy plan "
        "buffers. submit_copy() launches one batched copy kernel (64 KiB run-splitting happens "
        "inside C++) and returns the CompletionPoller id of its completion event, or BUSY (-1) "
        "when every stream context is in flight (retry next tick).")
        .def(nb::init<std::uint32_t, std::size_t, int, CompletionPoller&>(), nb::arg("num_streams"),
            nb::arg("max_plan_entries"), nb::arg("device_id"), nb::arg("poller"),
            // The pool registers its events with the poller; the poller must outlive the pool.
            nb::keep_alive<1, 5>(), nb::call_guard<nb::gil_scoped_release>())
        .def_ro_static("BUSY", &BatchedCopyPool::kBusy)
        .def_prop_ro("max_plan_entries", &BatchedCopyPool::maxPlanEntries)
        .def_prop_ro("num_streams", &BatchedCopyPool::size)
        .def("free_count", &BatchedCopyPool::freeCount)
        .def(
            "submit_copy",
            [](BatchedCopyPool& self, U64Array srcs, U64Array dsts, U32Array sizes)
            {
                std::size_t const n = srcs.shape(0);
                if (dsts.shape(0) != n || sizes.shape(0) != n)
                {
                    throw std::invalid_argument("submit_copy: srcs/dsts/sizes must have the same length");
                }
                return self.submitCopy(srcs.data(), dsts.data(), sizes.data(), n);
            },
            nb::arg("srcs"), nb::arg("dsts"), nb::arg("sizes"), nb::call_guard<nb::gil_scoped_release>(),
            "Copy sizes[i] bytes from srcs[i] to dsts[i] (uint64/uint64/uint32 host arrays of FINAL "
            "device addresses) as one batched kernel launch. Returns the poller completion id, or "
            "BUSY when no stream context is free.")
        .def(
            "submit_copy_chained",
            [](BatchedCopyPool& self, U64Array srcs, U64Array dsts, U32Array sizes)
            {
                std::size_t const n = srcs.shape(0);
                if (dsts.shape(0) != n || sizes.shape(0) != n)
                {
                    throw std::invalid_argument("submit_copy_chained: srcs/dsts/sizes must have the same length");
                }
                std::uint64_t reserved = 0;
                std::int64_t const copyId = self.submitCopy(srcs.data(), dsts.data(), sizes.data(), n, &reserved);
                return std::make_pair(copyId, reserved == 0 ? std::int64_t{-1} : static_cast<std::int64_t>(reserved));
            },
            nb::arg("srcs"), nb::arg("dsts"), nb::arg("sizes"), nb::call_guard<nb::gil_scoped_release>(),
            "submit_copy + a two-phase chain reservation taken ATOMICALLY with the completion-event "
            "registration (never loses the reserve race to the poll thread, unlike a separate "
            "reserve_chain call). Returns (copy_id, reserved_id); (BUSY, -1) when no stream context "
            "is free, and reserved_id -1 when the poller is already shut down (the copy_id then "
            "resolves classically).");

    // ---- FabricArena (wraps BounceArena) -------------------------------------------------------
    nb::class_<BounceArena>(m, "FabricArena",
        "The ONE shared bounce data buffer: a single contiguous device allocation, fabric memory "
        "(MNNVL / GPUDirect-RDMA capable) where supported, cudaMalloc otherwise. Registered once "
        "with NIXL via register_region(); the Python scheduler carves regions out of it by offset.")
        .def(
            "__init__",
            [](BounceArena* self, std::size_t nbytes, int deviceId, bool requireFabric)
            {
                new (self) BounceArena(nbytes, deviceId, /*allowFabric=*/true);
                if (requireFabric && !self->isFabric())
                {
                    self->~BounceArena();
                    throw std::runtime_error("FabricArena: fabric memory is required but not supported on this device");
                }
            },
            nb::arg("nbytes"), nb::arg("device_id"), nb::arg("require_fabric") = true,
            nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("base_ptr", &BounceArena::baseAddr)
        .def_prop_ro("size", &BounceArena::bytes)
        .def_prop_ro("is_fabric", &BounceArena::isFabric);

    // ---- Below-the-splitter agent primitives ---------------------------------------------------
    agentCls
        .def(
            "register_region",
            [](kvc::NixlTransferAgent& self, std::uintptr_t base, std::size_t nbytes, int deviceId)
            { return self.registerRegionLocked(reinterpret_cast<void*>(base), nbytes, deviceId); },
            nb::arg("base"), nb::arg("nbytes"), nb::arg("device_id"), nb::call_guard<nb::gil_scoped_release>(),
            "Register one raw device range with NIXL, below the VMM splitter (no region-map "
            "bookkeeping): one registration, single-descriptor per-chunk writes. Returns False on "
            "failure (logged).")
        .def(
            "deregister_region",
            [](kvc::NixlTransferAgent& self, std::uintptr_t base, std::size_t nbytes, int deviceId)
            { self.deregisterRegionLocked(reinterpret_cast<void*>(base), nbytes, deviceId); },
            nb::arg("base"), nb::arg("nbytes"), nb::arg("device_id"), nb::call_guard<nb::gil_scoped_release>())
        .def(
            "post_transfer_1to1",
            [](kvc::NixlTransferAgent& self, std::uintptr_t srcPtr, std::uintptr_t dstPtr, std::size_t nbytes,
                std::uint32_t srcDev, std::uint32_t dstDev, std::string const& peer,
                CompletionPoller& poller) -> std::int64_t
            {
                kvc::TransferDescs srcDescs{kvc::MemoryType::kVRAM, {kvc::MemoryDesc{srcPtr, nbytes, srcDev}}};
                kvc::TransferDescs dstDescs{kvc::MemoryType::kVRAM, {kvc::MemoryDesc{dstPtr, nbytes, dstDev}}};
                auto status
                    = self.postXferRequestLocked(kvc::TransferOp::kWRITE, srcDescs, dstDescs, peer, /*syncMessage=*/{});
                if (status == nullptr)
                {
                    return -1; // post failed or agent shut down (warning logged); Python fails the chunk
                }
                return static_cast<std::int64_t>(poller.registerXfer(std::move(status)));
            },
            nb::arg("src_ptr"), nb::arg("dst_ptr"), nb::arg("nbytes"), nb::arg("src_dev"), nb::arg("dst_dev"),
            nb::arg("peer"), nb::arg("poller"), nb::call_guard<nb::gil_scoped_release>(),
            "Post ONE single-descriptor no-notif RDMA write of FINAL device addresses (below the "
            "VMM splitter) and hand its TransferStatus to the poller. Returns the poller completion "
            "id, or -1 when the post failed.")
        .def(
            "post_transfer_1to1_on_event",
            [](kvc::NixlTransferAgent& self, std::uint64_t copyId, std::uintptr_t srcPtr, std::uintptr_t dstPtr,
                std::size_t nbytes, std::uint32_t srcDev, std::uint32_t dstDev, std::string const& peer,
                CompletionPoller& poller) -> std::int64_t
            {
                auto poster = [agent = &self, srcPtr, dstPtr, nbytes, srcDev, dstDev,
                                  peer]() -> std::unique_ptr<kvc::TransferStatus>
                {
                    kvc::TransferDescs srcDescs{kvc::MemoryType::kVRAM, {kvc::MemoryDesc{srcPtr, nbytes, srcDev}}};
                    kvc::TransferDescs dstDescs{kvc::MemoryType::kVRAM, {kvc::MemoryDesc{dstPtr, nbytes, dstDev}}};
                    return agent->postXferRequestLocked(
                        kvc::TransferOp::kWRITE, srcDescs, dstDescs, peer, /*syncMessage=*/{});
                };
                return poller.armXferAfterEvent(copyId, std::move(poster));
            },
            nb::arg("copy_id"), nb::arg("src_ptr"), nb::arg("dst_ptr"), nb::arg("nbytes"), nb::arg("src_dev"),
            nb::arg("dst_dev"), nb::arg("peer"), nb::arg("poller"),
            // The armed poster captures the agent; keep the agent alive at least as long as the
            // poller that may still run it (the engine teardown order already guarantees this;
            // keep_alive is belt-and-braces for ad-hoc scripts).
            nb::keep_alive<9, 1>(), nb::call_guard<nb::gil_scoped_release>(),
            "EXPERIMENTAL (TRTLLM_BOUNCE_V2_EXP_CPP_CHAIN): arm a gather->RDMA chain — when the "
            "given copy_id (a submit_copy completion id still pending in the poller) completes "
            "successfully, the C++ poll thread itself posts this single-descriptor RDMA write and "
            "polls it under a RESERVED completion id, which is returned. The gather completion is "
            "consumed in C++ (never drained), so Python sees exactly ONE completion per chunk: "
            "(reserved, KIND_XFER, ok) after the write, (reserved, KIND_XFER, 0) when the post "
            "failed or shutdown intervened, (reserved, KIND_EVENT, 0) when the gather itself "
            "failed (write never posted). Returns -1 when the copy_id is no longer pending "
            "(already completed / unknown / double-arm) — fall back to the classic path.")
        .def(
            "fulfill_chain_1to1",
            [](kvc::NixlTransferAgent& self, std::int64_t reservedId, std::uintptr_t srcPtr, std::uintptr_t dstPtr,
                std::size_t nbytes, std::uint32_t srcDev, std::uint32_t dstDev, std::string const& peer,
                CompletionPoller& poller) -> std::int64_t
            {
                auto poster = [agent = &self, srcPtr, dstPtr, nbytes, srcDev, dstDev,
                                  peer]() -> std::unique_ptr<kvc::TransferStatus>
                {
                    kvc::TransferDescs srcDescs{kvc::MemoryType::kVRAM, {kvc::MemoryDesc{srcPtr, nbytes, srcDev}}};
                    kvc::TransferDescs dstDescs{kvc::MemoryType::kVRAM, {kvc::MemoryDesc{dstPtr, nbytes, dstDev}}};
                    return agent->postXferRequestLocked(
                        kvc::TransferOp::kWRITE, srcDescs, dstDescs, peer, /*syncMessage=*/{});
                };
                return poller.fulfillChain(static_cast<std::uint64_t>(reservedId), std::move(poster));
            },
            nb::arg("reserved_id"), nb::arg("src_ptr"), nb::arg("dst_ptr"), nb::arg("nbytes"), nb::arg("src_dev"),
            nb::arg("dst_dev"), nb::arg("peer"), nb::arg("poller"),
            // Same keep-alive rationale as post_transfer_1to1_on_event: the poster captures the
            // agent, which the poller may still run after this call returns.
            nb::keep_alive<9, 1>(), nb::call_guard<nb::gil_scoped_release>(),
            "Two-phase chain, phase 2 (TRTLLM_BOUNCE_V2_EXP_CPP_CHAIN): attach this "
            "single-descriptor RDMA write to a reserve_chain() reservation now that the "
            "destination is known. Returns FULFILL_ARMED (gather still pending; posts on the poll "
            "thread when it fires), FULFILL_POSTED (gather already done; the write was posted "
            "inline and is polled under the reserved id), or FULFILL_DECLINED (a terminal row for "
            "the reserved id is already published/pending — do not post; just wait for it). Every "
            "outcome keeps exactly ONE terminal row per reserved id.");
}

} // namespace tensorrt_llm::executor::kv_cache::bounce_v2
