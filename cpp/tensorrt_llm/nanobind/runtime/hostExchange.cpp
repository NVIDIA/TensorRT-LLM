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

#include "hostExchange.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <thread>
#include <tuple>
#include <vector>

// Native wait of the host collectives over MNNVL memory (tensorrt_llm/_torch/distributed/host_exchange.py).
//
// Each rank owns a row of MNNVL memory that every rank maps. A row holds two slot sets, and each set holds one slot per
// sender rank. Rounds alternate between the sets by the parity of their epoch, so a fast rank never overwrites a slot
// that a slow rank still reads. A slot is a run of 8-byte packets:
//   packet 0 (header):  [ epoch : 32 | payload length in bytes : 32 ]
//   packet i (payload): [ epoch : 32 | 4 payload bytes             ]
// The epoch counts rounds and skips 0, because the rows start zeroed. A packet belongs to the current round when it
// carries the round's epoch, so a reader needs no separate ready flag. This relies on the copy engines moving each
// aligned 8-byte packet in one transaction, so that a reader never sees a packet's epoch with the other half of an
// older packet. CUDA does not document this; on GB200 no packet was seen torn in 3.8e9 polled packets of 4 KiB and
// 64 KiB slots, while a writer splitting each packet into two 4-byte copies tore 5.1e8. A torn header would carry an
// older payload length, and the wait would return a payload of that length; a torn payload packet would carry older
// payload bytes.
//
// One round on rank r:
//   1. Copy r's header into slot r of every row, and r's payload into the rows of the ranks that read it. A payload
//      short enough to arrive with a header poll goes to every row.
//   2. Poll the active ranks' headers in r's own row.
//   3. Once every header arrived, poll only the slots of the ranks whose payloads r reads, as wide as the longest one.
//   4. Return true when all of them carry the epoch, or false when the wake word is set first.
// Packets that carry the epoch stay in r's row until r finishes the round, because their sender reuses the slot set
// only after the next round, which needs r's header. So after the first poll of steps 2 and 3, each poll copies only
// the slots that still lack packets.
// The wake word is an int32 that another thread sets to make the wait return, so that the caller can check for a
// membership change.

namespace tensorrt_llm::nanobind::runtime
{
namespace
{

using HostPackets = nb::ndarray<uint64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using ReceivedPackets = nb::ndarray<uint64_t const, nb::ndim<2>, nb::device::cpu>;
using PayloadLengths = nb::ndarray<uint32_t const, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using PayloadRows = nb::ndarray<uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using RankMask = nb::ndarray<bool const, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using RankFlags = nb::ndarray<bool, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using WakeWord = nb::ndarray<int32_t, nb::shape<1>, nb::c_contig, nb::device::cpu>;
//! Rows sent by one copy as (destination, destination pitch, source, source pitch, width in bytes, height); a source
//! pitch of 0 sends one source row to every destination row.
using SlotCopy = std::tuple<uintptr_t, size_t, uintptr_t, size_t, size_t, size_t>;
using Seconds = std::chrono::duration<double>;

// After its spin a wait sleeps between polls, so a wait on a failed rank neither spins a core nor keeps a copy engine
// busy. The sleep starts at kMinSleep, about the shortest that the OS delivers, and doubles up to kMaxSleep, which
// bounds how late a long wait sees the packets or the wake word.
constexpr Seconds kMinSleep{100e-6};
constexpr Seconds kMaxSleep{1e-3};
// Payload bytes in the lower half of a payload packet.
constexpr size_t kPayloadBytesPerPacket = 4;

void copy2D(SlotCopy const& copy, cudaMemcpyKind kind, cudaStream_t stream)
{
    auto const [dst, dstPitch, src, srcPitch, width, height] = copy;
    TLLM_CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void*>(dst), dstPitch, reinterpret_cast<void const*>(src),
        srcPitch, width, height, kind, stream));
}

//! Enqueue every row of every copy in ``sends`` as one batch. A source pitch of 0 copies the same source row to every
//! destination row, which cudaMemcpy2DAsync rejects.
void enqueueSends(std::vector<SlotCopy> const& sends, cudaStream_t stream)
{
    std::vector<void*> dsts;
    std::vector<void const*> srcs;
    std::vector<size_t> sizes;
    size_t rows = 0;
    for (auto const& send : sends)
    {
        rows += std::get<5>(send);
    }
    dsts.reserve(rows);
    srcs.reserve(rows);
    sizes.reserve(rows);
    for (auto const& [dst, dstPitch, src, srcPitch, width, height] : sends)
    {
        for (size_t row = 0; row < height; ++row)
        {
            dsts.push_back(reinterpret_cast<void*>(dst + row * dstPitch));
            srcs.push_back(reinterpret_cast<void const*>(src + row * srcPitch));
            sizes.push_back(width);
        }
    }
    if (dsts.empty())
    {
        return;
    }
    cudaMemcpyAttributes attributes{};
    attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    // Prefer copy engines over SMs (cudaMemcpyFlagPreferOverlapWithCompute).
    attributes.flags = 1u;
    size_t firstCopy = 0;
    TLLM_CUDA_CHECK(
        cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), dsts.size(), &attributes, &firstCopy, 1, stream));
}

struct SlotCheck
{
    bool arrived; //!< Whether every active rank's header and every reader's payload arrived.
    //! The packets to poll per reader slot: the header plus the longest payload a reader waits for, at least 1 (the
    //! header alone). 0 while some active rank's header is missing.
    size_t payloadPackets;
};

//! Check the first ``polled`` packets of each rank's slot. Set ``waitingFor[r]`` for every active rank r whose header
//! is not from this epoch, and for every reader r whose polled payload packets are not. A payload longer than
//! ``polled`` has not arrived, but it does not set ``waitingFor``, because the caller widens the poll before it counts
//! the rank as missing. With ``headersFirst``, stop after the headers while any is missing.
SlotCheck checkSlots(uint64_t const* slots, size_t ranks, size_t slotPackets, bool const* active, bool const* readers,
    uint32_t epoch, size_t polled, bool headersFirst, bool* waitingFor)
{
    auto const current = [epoch](uint64_t packet) { return static_cast<uint32_t>(packet >> 32) == epoch; };
    SlotCheck check{true, 1};
    for (size_t rank = 0; rank < ranks; ++rank)
    {
        waitingFor[rank] = active[rank] && !current(slots[rank * slotPackets]);
        check.arrived = check.arrived && !waitingFor[rank];
    }
    bool const headers = check.arrived;
    if (!headers && headersFirst)
    {
        check.payloadPackets = 0;
        return check;
    }
    for (size_t rank = 0; rank < ranks; ++rank)
    {
        uint64_t const* slot = slots + rank * slotPackets;
        if (active[rank] && readers[rank] && !waitingFor[rank])
        {
            size_t const payloadWords
                = (static_cast<size_t>(static_cast<uint32_t>(slot[0])) + kPayloadBytesPerPacket - 1)
                / kPayloadBytesPerPacket;
            size_t const packets = 1 + std::min(slotPackets - 1, payloadWords);
            check.payloadPackets = std::max(check.payloadPackets, packets);
            waitingFor[rank] = !std::all_of(slot + 1, slot + std::min(packets, polled), current);
            check.arrived = check.arrived && !waitingFor[rank] && packets <= polled;
        }
    }
    if (!headers)
    {
        check.payloadPackets = 0;
    }
    return check;
}

//! Enqueue ``sends`` on ``stream``, then poll the device ``slots`` into ``received`` until the round's packets arrive.
//! ``slots`` holds one slot per rank, with the pitch of ``received``'s rows. Return true once the packets arrive, or
//! false once ``wake`` is set while packets are missing. Without ``slots``, the round was already completed from the
//! peers' rows during a membership change, so only send, wait for the sends to finish, and check ``received`` once.
bool waitForSlots(uintptr_t streamPtr, std::vector<SlotCopy> const& sends, std::optional<uintptr_t> slots,
    HostPackets const& received, RankMask const& active, RankMask const& readers, uint32_t epoch, size_t headerPackets,
    RankFlags const& waitingFor, WakeWord const& wake, double spinSeconds)
{
    size_t const ranks = received.shape(0);
    size_t const slotPackets = received.shape(1);
    TLLM_CHECK(active.shape(0) == ranks && readers.shape(0) == ranks && waitingFor.shape(0) == ranks);
    TLLM_CHECK(headerPackets > 0);
    auto const stream = reinterpret_cast<cudaStream_t>(streamPtr);
    enqueueSends(sends, stream);
    auto const check = [&](size_t polled, bool headersFirst)
    {
        return checkSlots(received.data(), ranks, slotPackets, active.data(), readers.data(), epoch, polled,
            headersFirst, waitingFor.data());
    };
    if (!slots)
    {
        // The next round rewrites the send buffer, so the sends must finish first.
        TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));
        return check(slotPackets, false).arrived;
    }
    // One 2D copy is cheaper than one copy per rank, so copy the slots of the ranks from the first to the last one set
    // in ``flags``, the ranks between them included, ``packets`` wide. Copying an arrived slot again leaves it as is.
    auto const pollCopy = [&](bool const* flags, size_t packets)
    {
        auto const first = static_cast<size_t>(std::find(flags, flags + ranks, true) - flags);
        size_t last = ranks;
        while (last > first && !flags[last - 1])
        {
            --last;
        }
        size_t const slotBytes = slotPackets * sizeof(uint64_t);
        return SlotCopy{reinterpret_cast<uintptr_t>(received.data() + first * slotPackets), slotBytes,
            *slots + first * slotBytes, slotBytes, packets * sizeof(uint64_t), last - first};
    };
    std::atomic_ref<int32_t> const wakeWord(*wake.data());
    size_t polled = std::min(headerPackets, slotPackets);
    SlotCopy poll = pollCopy(active.data(), polled);
    bool pollingPayloads = false;
    auto spinEnd = std::chrono::steady_clock::now() + Seconds(spinSeconds);
    auto sleep = kMinSleep;
    while (true)
    {
        copy2D(poll, cudaMemcpyDeviceToHost, stream);
        TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));
        auto const [arrived, payloadPackets] = check(polled, true);
        if (arrived)
        {
            return true;
        }
        if (payloadPackets > 0 && !pollingPayloads)
        {
            // Every header arrived, so poll only the readers' slots, as wide as the longest payload.
            pollingPayloads = true;
            polled = payloadPackets;
            poll = pollCopy(readers.data(), polled);
            spinEnd = std::chrono::steady_clock::now() + Seconds(spinSeconds);
            sleep = kMinSleep;
            continue;
        }
        poll = pollCopy(waitingFor.data(), polled);
        if (wakeWord.load(std::memory_order_relaxed) != 0)
        {
            return false;
        }
        if (std::chrono::steady_clock::now() >= spinEnd)
        {
            std::this_thread::sleep_for(sleep);
            sleep = std::min(2 * sleep, kMaxSleep);
        }
    }
}

//! Return the payload size of ``rank`` in ``received``: its entry in ``lengths``, cut to the packets that
//! ``received`` holds.
size_t payloadSize(ReceivedPackets const& received, PayloadLengths const& lengths, size_t rank)
{
    TLLM_CHECK(rank < received.shape(0));
    return std::min<size_t>(lengths.data()[rank], (received.shape(1) - 1) * kPayloadBytesPerPacket);
}

//! Copy the first ``size`` payload bytes of ``rank``, the lower halves of its payload packets, to ``out``.
template <typename Packets>
void copyPayload(Packets const& packets, size_t rank, size_t size, uint8_t* out)
{
    for (size_t offset = 0; offset < size; offset += kPayloadBytesPerPacket)
    {
        auto const word = static_cast<uint32_t>(packets(rank, 1 + offset / kPayloadBytesPerPacket));
        std::memcpy(out + offset, &word, std::min(kPayloadBytesPerPacket, size - offset));
    }
}

//! Return the payload of each rank in ``ranks`` in ``received`` as bytes.
nb::list unpackPayloads(
    ReceivedPackets const& received, PayloadLengths const& lengths, std::vector<size_t> const& ranks)
{
    TLLM_CHECK(received.shape(1) > 0 && lengths.shape(0) == received.shape(0));
    auto const packets = received.view();
    nb::list payloads;
    for (size_t const rank : ranks)
    {
        size_t const size = payloadSize(received, lengths, rank);
        auto const payload = nb::steal<nb::bytes>(PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(size)));
        if (!payload.is_valid())
        {
            throw nb::python_error();
        }
        copyPayload(packets, rank, size, reinterpret_cast<uint8_t*>(PyBytes_AS_STRING(payload.ptr())));
        payloads.append(payload);
    }
    return payloads;
}

//! Copy the payload of each rank in ``ranks`` in ``received`` to its row of ``out``, from column ``offset`` and cut to
//! the end of the row.
void unpackPayloadsInto(ReceivedPackets const& received, PayloadLengths const& lengths,
    std::vector<size_t> const& ranks, PayloadRows const& out, size_t offset)
{
    TLLM_CHECK(received.shape(1) > 0 && lengths.shape(0) == received.shape(0));
    TLLM_CHECK(out.shape(0) == received.shape(0) && offset <= out.shape(1));
    auto const packets = received.view();
    for (size_t const rank : ranks)
    {
        size_t const size = std::min(payloadSize(received, lengths, rank), out.shape(1) - offset);
        copyPayload(packets, rank, size, out.data() + rank * out.shape(1) + offset);
    }
}

} // namespace

void initHostExchangeBindings(nb::module_& m)
{
    // The wait writes into received and waiting_for and reads wake while it runs, so nanobind must pass the caller's
    // arrays, not converted copies.
    m.def("wait_for_slots", &waitForSlots, nb::arg("stream"), nb::arg("sends"), nb::arg("slots").none(),
        nb::arg("received").noconvert(), nb::arg("active"), nb::arg("readers"), nb::arg("epoch"),
        nb::arg("header_packets"), nb::arg("waiting_for").noconvert(), nb::arg("wake").noconvert(), nb::arg("spin_s"),
        "Enqueue the sends of one host exchange round and poll until the packets this rank reads arrive. Runs without "
        "the GIL. Returns false if the wake word is set first.",
        nb::call_guard<nb::gil_scoped_release>());
    m.def("unpack_payloads", &unpackPayloads, nb::arg("received"), nb::arg("lengths"), nb::arg("ranks"),
        "Return the payload bytes of each rank in ranks from the packets of one host exchange round.");
    m.def("unpack_payloads_into", &unpackPayloadsInto, nb::arg("received"), nb::arg("lengths"), nb::arg("ranks"),
        nb::arg("out").noconvert(), nb::arg("offset"),
        "Copy the payload bytes of each rank in ranks from the packets of one host exchange round to its row of out, "
        "starting at column offset.");
}

} // namespace tensorrt_llm::nanobind::runtime
