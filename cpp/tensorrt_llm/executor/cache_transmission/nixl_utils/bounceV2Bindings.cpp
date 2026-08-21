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
using U8Array = nb::ndarray<std::uint8_t const, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

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
        .def_ro_static("SCATTER_REJECTED", &BatchedCopyPool::kScatterRejected)
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
            "register_plan",
            [](BatchedCopyPool& self, U64Array srcs, U64Array bounceOffsets, U32Array sizes, U64Array chunkStarts)
            {
                std::size_t const n = srcs.shape(0);
                if (bounceOffsets.shape(0) != n || sizes.shape(0) != n)
                {
                    throw std::invalid_argument("register_plan: srcs/bounce_offsets/sizes must have the same length");
                }
                if (chunkStarts.shape(0) < 2)
                {
                    throw std::invalid_argument("register_plan: chunk_starts must have at least 2 entries");
                }
                return self.registerPlan(
                    srcs.data(), bounceOffsets.data(), sizes.data(), chunkStarts.data(), n, chunkStarts.shape(0) - 1);
            },
            nb::arg("srcs"), nb::arg("bounce_offsets"), nb::arg("sizes"), nb::arg("chunk_starts"),
            nb::call_guard<nb::gil_scoped_release>(),
            "Register one request's ENTIRE gather plan in ONE call (per-request plan handle): flat "
            "per-desc source addresses, REGION-RELATIVE bounce offsets and sizes over all chunks, "
            "sliced by chunk_starts ([n_chunks + 1] desc-index boundaries). The arrays are copied "
            "into pool-owned memory. Returns the plan handle; free it with release_plan() on the "
            "request's terminal paths (remaining plans are dropped at pool destruction). The plan "
            "holds no arena-region addresses — the staging base is a launch_chunk argument.")
        .def("release_plan", &BatchedCopyPool::releasePlan, nb::arg("handle"), nb::call_guard<nb::gil_scoped_release>(),
            "Drop a registered plan (idempotent). A launch_chunk racing this call either completes "
            "on its pinned plan snapshot or raises ValueError (unknown handle) deterministically; "
            "a launch AFTER release always raises.")
        .def(
            "launch_chunk",
            [](BatchedCopyPool& self, std::uint64_t handle, std::size_t chunkIdx, std::uint64_t stagingBase)
            { return self.launchChunk(handle, chunkIdx, stagingBase); },
            nb::arg("handle"), nb::arg("chunk_idx"), nb::arg("staging_base"), nb::call_guard<nb::gil_scoped_release>(),
            "Launch ONE chunk of a registered plan: gather its descs to staging_base + offset as "
            "one batched kernel, entirely from the pre-marshalled plan (scalar args only — no "
            "per-call array marshalling). Returns the poller completion id, or BUSY when no stream "
            "context is free. Raises ValueError on an unknown/released handle or chunk index.")
        .def(
            "submit_scatter_runs",
            [](BatchedCopyPool& self, std::uint64_t regionBase, std::uint64_t regionBytes, U8Array runs)
            {
                std::size_t const nbytes = runs.shape(0);
                if (nbytes % sizeof(ScatterRunWire) != 0)
                {
                    throw std::invalid_argument(
                        "submit_scatter_runs: runs blob length must be a multiple of the 36-byte wire run");
                }
                return self.submitScatterRuns(regionBase, regionBytes,
                    reinterpret_cast<ScatterRunWire const*>(runs.data()), nbytes / sizeof(ScatterRunWire));
            },
            nb::arg("region_base"), nb::arg("region_bytes"), nb::arg("runs"), nb::call_guard<nb::gil_scoped_release>(),
            "Receiver-side scatter of one DATA chunk in ONE call: validate the RAW wire runs (`runs` "
            "is the DATA payload viewed as a uint8 array — n*36-byte packed SCATTER_RUN_DTYPE "
            "records) against the granted region [region_base, region_base + region_bytes), expand "
            "them to per-piece copies, and launch as one batched kernel. The caller must have "
            "verified the region lies inside the registered arena. Returns the poller completion id, "
            "BUSY (-1) when no stream context is free (retry later), or SCATTER_REJECTED (-2) when "
            "validation failed — the caller must NOT ack the chunk but should release the region.");

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
            "launch_chunk_chained",
            [](kvc::NixlTransferAgent& self, BatchedCopyPool& pool, std::uint64_t handle, std::size_t chunkIdx,
                std::uint64_t stagingBase, std::uintptr_t dstPtr, std::size_t nbytes, std::uint32_t srcDev,
                std::uint32_t dstDev, std::string const& peer,
                CompletionPoller& poller) -> std::pair<std::int64_t, std::int64_t>
            {
                std::uint64_t reserved = 0;
                std::int64_t const copyId = pool.launchChunk(handle, chunkIdx, stagingBase, &reserved);
                if (copyId == BatchedCopyPool::kBusy || reserved == 0)
                {
                    // BUSY, or no reservation possible (poller shut down): the caller keeps the
                    // classic route — the copy_id (if any) resolves under its own completion id.
                    return {copyId, std::int64_t{-1}};
                }
                auto poster = [agent = &self, srcPtr = static_cast<std::uintptr_t>(stagingBase), dstPtr, nbytes, srcDev,
                                  dstDev, peer]() -> std::unique_ptr<kvc::TransferStatus>
                {
                    kvc::TransferDescs srcDescs{kvc::MemoryType::kVRAM, {kvc::MemoryDesc{srcPtr, nbytes, srcDev}}};
                    kvc::TransferDescs dstDescs{kvc::MemoryType::kVRAM, {kvc::MemoryDesc{dstPtr, nbytes, dstDev}}};
                    return agent->postXferRequestLocked(
                        kvc::TransferOp::kWRITE, srcDescs, dstDescs, peer, /*syncMessage=*/{});
                };
                // The reservation was taken atomically with the event registration, so this
                // fulfill can only be DECLINED when the gather already FAILED (its terminal row
                // (reserved, KIND_EVENT, 0) is published/pending) — nothing to do either way:
                // exactly one terminal row per reserved id is guaranteed by the poller.
                static_cast<void>(poller.fulfillChain(reserved, std::move(poster)));
                return {copyId, static_cast<std::int64_t>(reserved)};
            },
            nb::arg("pool"), nb::arg("handle"), nb::arg("chunk_idx"), nb::arg("staging_base"), nb::arg("dst_ptr"),
            nb::arg("nbytes"), nb::arg("src_dev"), nb::arg("dst_dev"), nb::arg("peer"), nb::arg("poller"),
            // The chain poster captures the agent; keep the agent alive at least as long as the
            // poller that may still run it (the engine teardown order already guarantees this;
            // keep_alive is belt-and-braces for ad-hoc scripts). `poller` MUST be the pool's own
            // poller (the engine wires exactly one).
            nb::keep_alive<11, 1>(), nb::call_guard<nb::gil_scoped_release>(),
            "Gather->RDMA chain for a credited chunk in ONE call: launch chunk_idx's gather from "
            "the registered plan (see BatchedCopyPool.register_plan) into staging_base, and have "
            "the C++ poll thread post this single-descriptor RDMA write of the staged region the "
            "moment the gather completes — no Python hop between gather and post. Returns "
            "(copy_id, reserved_id). Python sees exactly ONE completion per chained chunk, under "
            "reserved_id: (reserved, KIND_XFER, 1) after the write, (reserved, KIND_XFER, 0) when "
            "the post failed or shutdown intervened, (reserved, KIND_EVENT, 0) when the gather "
            "itself failed (write never posted). (BUSY, -1) when no stream context is free; "
            "(copy_id, -1) when the poller is already shut down — the copy_id then resolves "
            "classically under its own id. Raises ValueError on an unknown/released plan handle.");
}

} // namespace tensorrt_llm::executor::kv_cache::bounce_v2
