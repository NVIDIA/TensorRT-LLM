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
/* SM-driven unicast -> multicast copy, PTX 9.1 / CUDA 13.1 or newer.
 * One aligned TMA load handles each eligible 8 KiB logical slice. Physical
 * scratch slots include 16 bytes of padding for aligned head/tail accesses.
 *
 * Each communication warp owns a near-equal share of all whole 8 KiB slots, split into two
 * banks. A bank is one bulk group containing all of its 8 KiB slice slots.
 * A producer warp issues UC loads; its consumer warp issues MC stores.
 * Per-bank alternating tickets protect slot metadata and shared-memory reuse.
 * This permits every allocated slot to participate even when one warp owns
 * many slots, without exceeding the ISA's outstanding bulk-group limit.
 */
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalancePlanPolicy.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceTma.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceTmaGpuPlanBuilder.cuh"

#include <cuda.h>
#include <cuda/ptx>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>

#if CUDART_VERSION < 13010
#error "TMA multicast copy requires CUDA >= 13.1 (PTX >= 9.1)"
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#error "TMA multicast copy requires SM90 or newer"
#endif

namespace
{
constexpr unsigned kSlice = MEGAMOE_TMA_COPY_SLICE_BYTES;
constexpr unsigned kSlotControlBytes = MEGAMOE_TMA_COPY_SLOT_CONTROL_BYTES;
constexpr unsigned kSlotDataBytes = MEGAMOE_TMA_COPY_SLOT_DATA_BYTES;
constexpr unsigned kCoalesced = 4U;
constexpr unsigned kFallbackBulk = 8U;

// Small, dynamic plans travel with the launch instead of a separate DMA.
// Typed arrays preserve object lifetime/alignment and fit below 4 KiB of args.
constexpr uint64_t kInlineSegments = 16;
constexpr uint64_t kInlineRanges = 4;

struct InlineDescriptors
{
    MegamoeTmaCopySegment segments[kInlineSegments];
    MegamoeTmaCopyRange ranges[kInlineRanges];
};

static_assert(sizeof(InlineDescriptors) == 800, "inline descriptor layout");

struct alignas(16) SlotControl
{
    unsigned long long barrier;
    unsigned phase;
    unsigned reserved; // head=1, tail=2, coalesced=4, fallback-has-bulk=8
    uint64_t fast_dst;
    unsigned fast_bytes;
    unsigned fast_padding; // first slot in each bank: even=free, odd=ready ticket
};

static_assert(kSlotControlBytes == 32, "cached consume requires 32-byte control");
static_assert(offsetof(SlotControl, fast_dst) == 16, "cached destination alignment");
static_assert(offsetof(SlotControl, fast_bytes) == 24, "cached byte-count offset");
static_assert(kSlotDataBytes == kSlice + 16, "one aligned edge of slot padding");
static_assert(sizeof(SlotControl) == kSlotControlBytes, "shared slot control ABI");
static_assert(kSlotDataBytes % 16 == 0, "TMA slot alignment");
static_assert(sizeof(MegamoeTmaCopySegment) == 4 * sizeof(uint64_t), "segment ABI");
static_assert(sizeof(MegamoeTmaCopyRange) == 9 * sizeof(uint64_t), "range ABI");
static_assert(sizeof(MegamoeTmaCopyConfig) == 88, "configuration ABI");

__device__ __forceinline__ unsigned shared_address(void const* p)
{
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void initialize_barrier(SlotControl* slot)
{
    unsigned bar = shared_address(&slot->barrier);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar) : "memory");
    slot->phase = 0;
    slot->reserved = 0;
    slot->fast_padding = 0;
}

__device__ __forceinline__ void expect_slice(SlotControl* slot, unsigned bytes)
{
    unsigned bar = shared_address(&slot->barrier);
    unsigned long long arrival;
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;"
                 : "=l"(arrival)
                 : "r"(bar), "r"(bytes)
                 : "memory");
    asm volatile("" ::"l"(arrival));
}

__device__ __forceinline__ void load_fragment(unsigned char* buffer, SlotControl* slot, uint64_t src, unsigned bytes)
{
    unsigned bar = shared_address(&slot->barrier);
    unsigned smem = shared_address(buffer);
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];" ::"r"(smem),
        "l"(src), "r"(bytes), "r"(bar)
        : "memory");
}

__device__ __forceinline__ void await_slice(SlotControl* slot)
{
    unsigned ready;
    unsigned bar = shared_address(&slot->barrier);
    unsigned phase = slot->phase;
    do
    {
        asm volatile(
            "{ .reg .pred p; "
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; "
            "selp.u32 %0, 1, 0, p; }"
            : "=r"(ready)
            : "r"(bar), "r"(phase)
            : "memory");
        if (!ready)
            __nanosleep(32);
    } while (!ready);
    // Only a slot that actually issued and completed TMA advances its phase.
    slot->phase = phase ^ 1U;
}

__device__ __forceinline__ void multicast_slice(uint64_t dst, unsigned char const* buffer, unsigned bytes)
{
    unsigned smem = shared_address(buffer);
    asm volatile(
        "multimem.cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst), "r"(smem), "r"(bytes)
        : "memory");
}

__device__ __forceinline__ void multicast_edge(uint64_t dst, unsigned char const* buffer, uint16_t mask)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    unsigned smem = shared_address(buffer);
    asm volatile(
        "multimem.cp.async.bulk.global.shared::cta.bulk_group.cp_mask "
        "[%0], [%1], 16, %2;" ::"l"(dst),
        "r"(smem), "h"(mask)
        : "memory");
#else
    // coalesced_copy_flags() is zero on these targets; never use .cp_mask on SM90.
    asm volatile("trap;" ::: "memory");
#endif
}

__device__ __forceinline__ void wait_older_reads()
{
    asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
}

__device__ __forceinline__ void wait_all_reads()
{
    // Buffer reuse only requires completion of the SMEM reads. Destination
    // writes are drained separately before publishing READY.
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

__device__ __forceinline__ void wait_all_writes()
{
    // Deliberately no .read: the multicast destination writes must be complete.
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ void system_publish_fence()
{
    asm volatile("fence.proxy.alias;" ::: "memory");
    asm volatile("fence.release.sys;" ::: "memory");
}

__device__ __forceinline__ void multicast_generation(uint64_t flag_mc, uint64_t generation)
{
    // The notification uses the same MC backing/UC acquire protocol as the
    // former copy-engine terminal. Publish only after all issuers' writes.
    system_publish_fence();
    asm volatile("multimem.st.release.sys.global.u64 [%0], %1;" ::"l"(flag_mc), "l"(generation) : "memory");
    asm volatile("fence.proxy.alias;" ::: "memory");
}

__device__ __forceinline__ void finish_cta_and_notify(unsigned* completed_ctas, uint64_t flag_mc, uint64_t generation)
{
    unsigned previous;
    // A release/acquire RMW chain combines the completed payload of all CTAs.
    // No CTA waits for another: oversubscribed grids cannot deadlock here.
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(previous) : "l"(completed_ctas) : "memory");
    if (previous == gridDim.x - 1)
    {
        // All RMWs are done. The next launch is ordered after this kernel, so
        // the last CTA can rearm the counter without a host memset each time.
        asm volatile("st.relaxed.gpu.global.u32 [%0], 0;" ::"l"(completed_ctas) : "memory");
        multicast_generation(flag_mc, generation);
    }
}

struct Slice
{
    uint64_t begin;
    uint64_t end;
    uint64_t segment_begin;
    uint64_t segment_end;
};

struct Fragment
{
    uint64_t src;
    uint64_t dst;
    unsigned bytes;
    unsigned scalar_prefix;
    unsigned bulk_bytes;
    unsigned smem_offset;
};

__device__ __forceinline__ Slice locate_slice(
    MegamoeTmaCopySegment const* segments, MegamoeTmaCopyRange const* ranges, uint64_t range_count, uint64_t ordinal)
{
    uint64_t lo = 0, hi = range_count;
    while (lo < hi)
    {
        uint64_t mid = lo + (hi - lo) / 2;
        if (ordinal < ranges[mid].prefix_end)
            hi = mid;
        else
            lo = mid + 1;
    }
    const MegamoeTmaCopyRange range = ranges[lo];
    const uint64_t local = ordinal - (range.prefix_end - range.slice_count);
    uint64_t slice = range.first_slice + local * range.slice_stride;
    if (range.slice_divisor != 1)
    {
        // floor(n/d) from a cold-computed reciprocal; the low estimate is at
        // most one short. No runtime integer division in the TMA issue loop.
        const uint64_t estimate = __umul64hi(slice, range.slice_reciprocal);
        slice = estimate + (slice - estimate * range.slice_divisor >= range.slice_divisor);
    }
    const uint64_t begin = slice * kSlice;
    const uint64_t remaining = range.bytes - begin;
    const uint64_t end = begin + (remaining < kSlice ? remaining : kSlice);
    const uint64_t segment_end = range.segment_begin + range.segment_count;
    lo = range.segment_begin;
    hi = segment_end;
    while (lo < hi)
    {
        const uint64_t mid = lo + (hi - lo) / 2;
        if (segments[mid].virtual_begin + segments[mid].bytes <= begin)
            lo = mid + 1;
        else
            hi = mid;
    }
    return Slice{begin, end, lo, segment_end};
}

__device__ __forceinline__ Fragment locate_fragment(Slice const& slice, MegamoeTmaCopySegment const& segment)
{
    const uint64_t begin = slice.begin > segment.virtual_begin ? slice.begin : segment.virtual_begin;
    const uint64_t segment_end = segment.virtual_begin + segment.bytes;
    const uint64_t end = slice.end < segment_end ? slice.end : segment_end;
    unsigned const bytes = static_cast<unsigned>(end - begin);
    const uint64_t offset = begin - segment.virtual_begin;
    const uint64_t src = segment.src + offset, dst = segment.dst + offset;
    // UC -> SMEM alignment depends only on the source. A differently aligned
    // MC destination can consume the staged data with exact 32-bit stores.
    unsigned prefix = static_cast<unsigned>((16U - (src & 15U)) & 15U);
    if (prefix > bytes)
        prefix = bytes;
    unsigned const bulk = (bytes - prefix) & ~15U;
    // The virtual concat is NOT a physical address space. Align each bulk
    // fragment's scratch area independently; a preceding 4/8/12-byte plane
    // must not force the next large weight onto the scalar path.
    // end_of_bulk <= align_up(fragment_virtual_end,16) <= 8192, and the
    // next fragment's aligned start is no smaller. Scratch regions are disjoint.
    unsigned const smem_offset = static_cast<unsigned>((begin - slice.begin + prefix + 15U) & ~uint64_t(15U));
    return Fragment{src, dst, bytes, prefix, bulk, smem_offset};
}

// Only a virtual slice entirely contained in one physical segment can use
// one padded, aligned TMA load. Other slices retain exact scalar edge copies.
// The NOMINAL aligned 16-byte accesses, not just their masks, must lie within
// the original source and destination intervals. No small-plane overread.
__device__ __forceinline__ unsigned coalesced_copy_flags(
    Slice const& slice, MegamoeTmaCopySegment const& segment, Fragment const& f)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if (slice.begin < segment.virtual_begin || slice.end > segment.virtual_begin + segment.bytes
        || ((f.src ^ f.dst) & 15U) || segment.bytes < 16)
        return 0;
    unsigned const tail = f.bytes - f.scalar_prefix - f.bulk_bytes;
    unsigned const flags = (f.scalar_prefix ? 1U : 0U) | (tail ? 2U : 0U);
    const uint64_t first_src = f.src & ~uint64_t(15);
    const uint64_t first_dst = f.dst & ~uint64_t(15);
    const uint64_t last_src = (f.src + f.bytes - 1) & ~uint64_t(15);
    const uint64_t last_dst = (f.dst + f.bytes - 1) & ~uint64_t(15);
    // Subtraction form also avoids overflow of a one-past/rounded-up pointer.
    if (first_src < segment.src || first_dst < segment.dst || last_src < segment.src || last_dst < segment.dst
        || last_src - segment.src > segment.bytes - 16 || last_dst - segment.dst > segment.bytes - 16)
        return 0;
    return flags | kCoalesced;
#else
    return 0;
#endif
}

__device__ __forceinline__ uint16_t edge_byte_mask(unsigned begin, unsigned bytes)
{
    // begin+bytes <= 16; the shift is evaluated in 32 bits.
    return static_cast<uint16_t>(((1U << bytes) - 1U) << begin);
}

__device__ __forceinline__ void prefetch_bank(MegamoeTmaCopySegment const* segments, MegamoeTmaCopyRange const* ranges,
    uint64_t range_count, uint64_t ordinal_base, uint64_t ordinal_stride, uint64_t sequence, unsigned char* warp_buffer,
    SlotControl* warp_control, int slot_start, int count)
{
    for (int i = 0; i < count; ++i)
    {
        const Slice slice = locate_slice(segments, ranges, range_count, ordinal_base + ordinal_stride * (sequence + i));
        SlotControl* slot = warp_control + slot_start + i;
        unsigned char* slot_buffer = warp_buffer + static_cast<size_t>(slot_start + i) * kSlotDataBytes;
        const MegamoeTmaCopySegment first_segment = segments[slice.segment_begin];
        const Fragment first = locate_fragment(slice, first_segment);
        unsigned const fast = coalesced_copy_flags(slice, first_segment, first);
        if (fast)
        {
            // ONE aligned TMA load supplies head, body and tail. The only extra
            // storage is the final 16B of this physical slot; logical RR remains 8KiB.
            slot->reserved = fast;
            slot->fast_dst = first.dst;
            slot->fast_bytes = first.bytes;
            unsigned const load_bytes = (static_cast<unsigned>(first.src & 15U) + first.bytes + 15U) & ~15U;
            expect_slice(slot, load_bytes);
            load_fragment(slot_buffer, slot, first.src & ~uint64_t(15), load_bytes);
            continue;
        }
        unsigned bulk_bytes = first.bulk_bytes;
        for (uint64_t index = slice.segment_begin + 1;
             index < slice.segment_end && segments[index].virtual_begin < slice.end; ++index)
            bulk_bytes += locate_fragment(slice, segments[index]).bulk_bytes;
        slot->reserved = bulk_bytes ? kFallbackBulk : 0U;
        if (!bulk_bytes)
            continue;
        expect_slice(slot, bulk_bytes);
        if (first.bulk_bytes)
            load_fragment(slot_buffer + first.smem_offset, slot, first.src + first.scalar_prefix, first.bulk_bytes);
        for (uint64_t index = slice.segment_begin + 1;
             index < slice.segment_end && segments[index].virtual_begin < slice.end; ++index)
        {
            const Fragment f = locate_fragment(slice, segments[index]);
            if (f.bulk_bytes)
                load_fragment(slot_buffer + f.smem_offset, slot, f.src + f.scalar_prefix, f.bulk_bytes);
        }
    }
}

__device__ __forceinline__ void scalar_words(Fragment const& f, unsigned begin, unsigned end)
{
    for (unsigned off = begin; off < end; off += 4)
    {
        unsigned value;
        // UC means unicast. .cg avoids a stale read-only (.nc) cache.
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(f.src + off) : "memory");
        asm volatile("multimem.st.relaxed.sys.global.u32 [%0], %1;" ::"l"(f.dst + off), "r"(value) : "memory");
    }
}

__device__ __forceinline__ void multicast_shared_words(Fragment const& f, unsigned char const* buffer)
{
    // The slot mbarrier has completed: TMA data is visible to generic shared
    // loads. Only the source-aligned body is staged; no rounded overread.
    for (unsigned off = 0; off < f.bulk_bytes; off += 4)
    {
        unsigned value;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(value) : "r"(shared_address(buffer + off)) : "memory");
        asm volatile("multimem.st.relaxed.sys.global.u32 [%0], %1;" ::"l"(f.dst + f.scalar_prefix + off), "r"(value)
                     : "memory");
    }
}

__device__ __forceinline__ void consume_bank(MegamoeTmaCopySegment const* segments, MegamoeTmaCopyRange const* ranges,
    uint64_t range_count, uint64_t ordinal_base, uint64_t ordinal_stride, uint64_t sequence, unsigned char* warp_buffer,
    SlotControl* warp_control, int slot_start, int count, int bank, bool* pending, int& outstanding)
{
    bool issued = false;
    for (int i = 0; i < count; ++i)
    {
        SlotControl* slot = warp_control + slot_start + i;
        unsigned const flags = slot->reserved;
        unsigned char const* slot_buffer = warp_buffer + static_cast<size_t>(slot_start + i) * kSlotDataBytes;
        if (flags & kCoalesced)
        {
            // Prefetch cached these values before issuing the TMA load.
            // No range search, segment search, or descriptor reload on consume.
            const uint64_t dst = slot->fast_dst;
            unsigned const bytes = slot->fast_bytes;
            unsigned const shift = static_cast<unsigned>(dst & 15U);
            unsigned prefix = (16U - shift) & 15U;
            if (prefix > bytes)
                prefix = bytes;
            unsigned const bulk = (bytes - prefix) & ~15U;
            unsigned const tail_offset = prefix + bulk;
            const uint16_t head_mask = edge_byte_mask(shift, prefix);
            const uint16_t tail_mask = edge_byte_mask(0, bytes - tail_offset);
            await_slice(slot);
            if (bulk)
                multicast_slice(dst + prefix, slot_buffer + shift + prefix, bulk);
            if (flags & 1U)
                multicast_edge(dst & ~uint64_t(15), slot_buffer, head_mask);
            if (flags & 2U)
                multicast_edge(dst + tail_offset, slot_buffer + shift + tail_offset, tail_mask);
            issued = true;
            continue;
        }
        const Slice slice = locate_slice(segments, ranges, range_count, ordinal_base + ordinal_stride * (sequence + i));
        // Prefetch already determined whether this slice has any async data:
        // no duplicate fragment scan or second range lookup for scalar edges.
        if (flags & kFallbackBulk)
            await_slice(slot);
        for (uint64_t index = slice.segment_begin;
             index < slice.segment_end && segments[index].virtual_begin < slice.end; ++index)
        {
            const Fragment f = locate_fragment(slice, segments[index]);
            if (f.bulk_bytes)
            {
                if (((f.dst + f.scalar_prefix) & 15U) == 0)
                {
                    multicast_slice(f.dst + f.scalar_prefix, slot_buffer + f.smem_offset, f.bulk_bytes);
                    issued = true;
                }
                else
                {
                    // Different UC/MC alignment cannot use a bulk MC store.
                    // Reuse the prefetched bytes instead of serial remote loads.
                    multicast_shared_words(f, slot_buffer + f.smem_offset);
                }
            }
            // These exact scalar intervals are disjoint from ALL TMA writes.
            // No intermediate group drain; final full-write wait/fence is unchanged.
            scalar_words(f, 0, f.scalar_prefix);
            scalar_words(f, f.scalar_prefix + f.bulk_bytes, f.bytes);
        }
    }
    if (issued)
    {
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        pending[bank] = true;
        ++outstanding;
    }
}

// The two peers are the only writers of a bank ticket. Unsigned wrap is
// safe: a peer cannot publish its next ticket until the other responds.
__device__ __forceinline__ void wait_bank_ticket(unsigned const* address, unsigned expected)
{
    unsigned observed;
    do
    {
        asm volatile("ld.acquire.cta.shared::cta.u32 %0, [%1];"
                     : "=r"(observed)
                     : "r"(shared_address(address))
                     : "memory");
        if (observed != expected)
            __nanosleep(32);
    } while (observed != expected);
}

__device__ __forceinline__ void publish_bank_ticket(unsigned* address, unsigned value)
{
    asm volatile("st.release.cta.shared::cta.u32 [%0], %1;" ::"r"(shared_address(address)), "r"(value) : "memory");
}

__device__ __forceinline__ void tma_copy_body(MegamoeTmaCopySegment const* segments, MegamoeTmaCopyRange const* ranges,
    uint64_t range_count, uint64_t total_slices, int warps, int total_slots, unsigned* completed_ctas, uint64_t flag_mc,
    uint64_t generation)
{
    if (!total_slices)
    {
        // Empty source ranks still publish READY. One tiny CTA, same kernel;
        // no descriptor upload or additional host-side notification launch.
        if (threadIdx.x == 0 && flag_mc)
            multicast_generation(flag_mc, generation);
        return;
    }
    extern __shared__ __align__(128) unsigned char smem[];
    int const physical_warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int const warp = physical_warp / 2; // logical RR worker, shared by its two warps
    bool const producer = (physical_warp & 1) == 0;
    int const slots_per_warp = megamoe_tma_warp_slot_count(total_slots, warps, warp);
    int const first_slot = megamoe_tma_warp_slot_begin(total_slots, warps, warp);
    SlotControl* control = reinterpret_cast<SlotControl*>(smem + static_cast<size_t>(total_slots) * kSlotDataBytes);
    SlotControl* warp_control = control + first_slot;
    unsigned char* warp_buffer = smem + static_cast<size_t>(first_slot) * kSlotDataBytes;
    if (producer && lane == 0)
    {
        for (int slot = 0; slot < slots_per_warp; ++slot)
            initialize_barrier(warp_control + slot);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    // Elect once for the entire pipeline so ptxas can prove one TMA issuer per warp.
    if (cuda::ptx::elect_sync(0xffffffffu))
    {
        // CTA/SM is the fastest-varying dimension, then warp, then sequence.
        const uint64_t ordinal_base = megamoe_tma_worker_first_slice(blockIdx.x, gridDim.x, warp);
        const uint64_t ordinal_stride = megamoe_tma_worker_slice_stride(gridDim.x, warps);
        const uint64_t tasks = ordinal_base < total_slices ? 1 + (total_slices - 1 - ordinal_base) / ordinal_stride : 0;
        int const capacity[2] = {(slots_per_warp + 1) / 2, slots_per_warp / 2};
        int const start[2] = {0, capacity[0]};
        unsigned ticket[2] = {producer ? 0U : 1U, producer ? 0U : 1U};
        uint64_t next_sequence = 0;
        if (producer)
        {
            // Each bank is independent: bank 1 can already be loading while
            // the consumer drains bank 0's shared-memory reads and releases it.
            for (int bank = 0; next_sequence < tasks; bank ^= 1)
            {
                const uint64_t left = tasks - next_sequence;
                int const count
                    = static_cast<int>(left < static_cast<uint64_t>(capacity[bank]) ? left : capacity[bank]);
                unsigned* handshake = &warp_control[start[bank]].fast_padding;
                wait_bank_ticket(handshake, ticket[bank]);
                prefetch_bank(segments, ranges, range_count, ordinal_base, ordinal_stride, next_sequence, warp_buffer,
                    warp_control, start[bank], count);
                // Publish generic slot metadata after all TMA loads have been
                // issued. The consumer still waits on each load's mbarrier.
                publish_bank_ticket(handshake, ticket[bank] + 1U);
                ticket[bank] += 2U;
                next_sequence += count;
            }
        }
        else
        {
            bool pending[2] = {false, false};
            int outstanding = 0;
            int held_bank = -1;
            for (int bank = 0; next_sequence < tasks; bank ^= 1)
            {
                const uint64_t left = tasks - next_sequence;
                int const count
                    = static_cast<int>(left < static_cast<uint64_t>(capacity[bank]) ? left : capacity[bank]);
                unsigned* handshake = &warp_control[start[bank]].fast_padding;
                wait_bank_ticket(handshake, ticket[bank]);
                consume_bank(segments, ranges, range_count, ordinal_base, ordinal_stride, next_sequence, warp_buffer,
                    warp_control, start[bank], count, bank, pending, outstanding);
                if (held_bank >= 0)
                {
                    // Consume both banks before releasing the older one. The
                    // producer refills it while this bank's MC reads remain
                    // outstanding; at most two read-pending groups exist.
                    if (pending[held_bank])
                    {
                        if (outstanding == 2)
                            wait_older_reads();
                        else
                            wait_all_reads(); // this bank was scalar-only
                        pending[held_bank] = false;
                        --outstanding;
                    }
                    publish_bank_ticket(&warp_control[start[held_bank]].fast_padding, ticket[held_bank] + 1U);
                    ticket[held_bank] += 2U;
                }
                held_bank = bank;
                next_sequence += count;
            }
            // The final bank has no following consume to release it. Flush
            // remaining reads and publish its final free ticket explicitly.
            if (held_bank >= 0)
            {
                if (pending[held_bank])
                    wait_all_reads();
                publish_bank_ticket(&warp_control[start[held_bank]].fast_padding, ticket[held_bank] + 1U);
            }
            // READY still covers destination completion, not merely SMEM reads.
            wait_all_writes();
            // The notifying path joins every issuer through the CTA barrier
            // and the acq_rel counter chain. Its last CTA executes the alias
            // and SYS fences in multicast_generation before publishing READY.
            // Raw submit has no final publisher and retains issuer fences.
            if (!flag_mc)
                system_publish_fence();
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && flag_mc)
        finish_cta_and_notify(completed_ctas, flag_mc, generation);
}

// Both entry points execute the same payload and READY protocol.
__global__ void tma_copy_kernel(MegamoeTmaCopySegment const* segments, MegamoeTmaCopyRange const* ranges,
    uint64_t range_count, uint64_t total_slices, int warps, int total_slots, unsigned* completed_ctas, uint64_t flag_mc,
    uint64_t generation)
{
    tma_copy_body(segments, ranges, range_count, total_slices, warps, total_slots, completed_ctas, flag_mc, generation);
}

__global__ void tma_copy_inline_kernel(const __grid_constant__ InlineDescriptors descriptors, uint64_t range_count,
    uint64_t total_slices, int warps, int total_slots, unsigned* completed_ctas, uint64_t flag_mc, uint64_t generation)
{
    // __grid_constant__ keeps these addresses in parameter storage; no
    // per-thread local copy and no shared-memory descriptor staging.
    tma_copy_body(descriptors.segments, descriptors.ranges, range_count, total_slices, warps, total_slots,
        completed_ctas, flag_mc, generation);
}

// Compile-time validation table: no host division or mutable cache per submit.
constexpr auto make_slice_reciprocals()
{
    std::array<uint64_t, 3201> values{};
    for (unsigned divisor = 2; divisor < values.size(); ++divisor)
        values[divisor] = UINT64_MAX / divisor;
    return values;
}

constexpr auto kSliceReciprocals = make_slice_reciprocals();

// Per-CTA scratch is allocated once. No plan tensor is copied between kernels.
struct GpuDirectStorage
{
    MegamoeTmaGpuPlanConfig config;
    int const *ids, *levels, *owners, *workspace;
    int capacity;
    uint64_t max_segments;
    int* scratch;
    MegamoeTmaCopySegment* segments;
    MegamoeTmaCopyRange* ranges;
    MegamoeTmaGpuPlanResult* results;
};

__global__ void tma_copy_gpu_direct_kernel(GpuDirectStorage direct, int warps, int total_slots,
    unsigned* completed_ctas, uint64_t flag_mc, uint64_t generation)
{
    auto const& c = direct.config;
    const uint64_t index = blockIdx.x;
    int* plan = direct.scratch + index * (c.plan_words + 4ULL * c.helper_count);
    auto* segments = direct.segments + index * direct.max_segments;
    auto* ranges = direct.ranges + index * direct.max_segments;
    auto* result = direct.results + index;
    // HALO-Q's existing workspace: epoch, count, six capacity-strided fields.
    int const* fields = direct.workspace + 2;
    int const cap = direct.capacity;
    tensorrt_llm::kernels::sami_copy::PlacementView placement{direct.workspace[1], cap, fields, fields + cap,
        fields + 2 * cap, fields + 3 * cap, fields + 4 * cap, fields + 5 * cap};
    // Private routing scratch, not a CPU channel: no SYS publication, extra
    // CUDA operation or scheduler-side copy of the placement outputs.
    tensorrt_llm::kernels::sami_copy::publish_plan_channel_warp<false>(plan, direct.ids, direct.levels, direct.owners,
        placement, c.helper_count, c.world, c.rank, c.global_experts, c.plan_abi_version, c.route_features);
    __syncthreads();
    if (threadIdx.x == 0)
    {
        megamoe_tma_build_gpu_plan(
            &c, plan, plan + c.plan_words, segments, direct.max_segments, ranges, direct.max_segments, result);
        // Malformed plans must never publish READY. The CUDA error surfaces at
        // the caller's normal completion check, without a host wait in submit.
        if (result->error)
        {
            printf("GPU-direct in-switch copy: invalid plan, error=%d; no READY published\n", result->error);
            asm volatile("trap;" ::: "memory");
        }
    }
    __syncthreads();
    if (!result->total_slices)
    {
        // Work size is device-derived, so the launch always has config.sms CTAs.
        // Empty ranks must still join the last-CTA chain before one READY store.
        if (threadIdx.x == 0)
            finish_cta_and_notify(completed_ctas, flag_mc, generation);
        return;
    }
    tma_copy_body(segments, ranges, result->range_count, result->total_slices, warps, total_slots, completed_ctas,
        flag_mc, generation);
}

cudaError_t validate_descriptors(MegamoeTmaCopySegment const* segments, uint64_t segment_count,
    MegamoeTmaCopyRange const* ranges, uint64_t range_count, uint64_t* total_slices)
{
    for (uint64_t i = 0; i < segment_count; ++i)
    {
        auto const& s = segments[i];
        if (!s.src || !s.dst || ((s.src | s.dst | s.bytes | s.virtual_begin) & 3U) || !s.bytes
            || s.bytes - 1 > UINT64_MAX - s.src || s.bytes - 1 > UINT64_MAX - s.dst
            || s.bytes > UINT64_MAX - s.virtual_begin)
            return cudaErrorInvalidValue;
    }
    uint64_t prefix = 0, virtual_end = 0;
    for (uint64_t i = 0; i < range_count; ++i)
    {
        auto const& r = ranges[i];
        if (!r.bytes || (r.bytes & 3U) || !r.slice_stride || !r.slice_count || !r.segment_count
            || r.segment_begin > segment_count || r.segment_count > segment_count - r.segment_begin)
            return cudaErrorInvalidValue;
        // Adjacent residue ranges may share one virtual segment stream. Its
        // continuity is invariant within this call; validate each range below.
        if (!i || r.segment_begin != ranges[i - 1].segment_begin || r.segment_count != ranges[i - 1].segment_count)
        {
            virtual_end = 0;
            for (uint64_t j = r.segment_begin; j < r.segment_begin + r.segment_count; ++j)
            {
                if (segments[j].virtual_begin != virtual_end || segments[j].bytes > UINT64_MAX - virtual_end)
                    return cudaErrorInvalidValue;
                virtual_end += segments[j].bytes;
            }
        }
        if (virtual_end != r.bytes)
            return cudaErrorInvalidValue;
        const uint64_t slices = 1 + (r.bytes - 1) / kSlice;
        if (!r.slice_divisor || r.slice_divisor > 3200 || r.slice_stride < r.slice_divisor || slices > UINT64_MAX / 3200
            || r.slice_reciprocal != kSliceReciprocals[r.slice_divisor])
            return cudaErrorInvalidValue;
        const uint64_t limit = slices * r.slice_divisor;
        if (r.first_slice >= limit || r.slice_count != 1 + (limit - 1 - r.first_slice) / r.slice_stride
            || r.slice_count > UINT64_MAX - prefix)
            return cudaErrorInvalidValue;
        prefix += r.slice_count;
        if (r.prefix_end != prefix)
            return cudaErrorInvalidValue;
    }
    *total_slices = prefix;
    return cudaSuccess;
}

} // namespace

struct MegamoeTmaCopyState
{
    MegamoeTmaCopyConfig config{};
    MegamoeTmaCopySegment* device_segments = nullptr;
    MegamoeTmaCopySegment* host_segments = nullptr;
    unsigned* completed_ctas = nullptr;
    cudaEvent_t descriptors_uploaded = nullptr;
    cudaEvent_t kernel_complete = nullptr;
    cudaStream_t last_stream = nullptr;
    bool inline_supported = false;
    bool upload_recorded = false;
    bool completion_recorded = false;
    bool work_queued = false;
    cudaError_t poison = cudaSuccess;
    GpuDirectStorage direct{};
    uint64_t* direct_tables = nullptr;
    bool direct_configured = false;
    bool direct_bound = false;
    std::mutex submit_mutex;
};

extern "C" int megamoe_tma_copy_destroy(MegamoeTmaCopyState** state_ptr)
{
    if (!state_ptr || !*state_ptr)
        return static_cast<int>(cudaSuccess);
    MegamoeTmaCopyState* state = *state_ptr;
    cudaError_t first = cudaSuccess;
    auto record = [&first](cudaError_t error)
    {
        if (first == cudaSuccess)
            first = error;
    };
    int previous_device = -1;
    cudaError_t error = cudaGetDevice(&previous_device);
    record(error);
    if (error == cudaSuccess && previous_device != state->config.device)
    {
        error = cudaSetDevice(state->config.device);
        record(error);
    }
    if (error != cudaSuccess)
        return static_cast<int>(first); // Preserve state for a later cleanup attempt.
    // last_stream also covers work whose event recording failed. New streams
    // wait on the previous completion event before reusing device descriptors.
    if (state->work_queued)
        record(cudaStreamSynchronize(state->last_stream));
    if (state->direct_tables)
        record(cudaFree(state->direct_tables));
    if (state->direct.scratch)
        record(cudaFree(state->direct.scratch));
    if (state->direct.segments)
        record(cudaFree(state->direct.segments));
    if (state->direct.ranges)
        record(cudaFree(state->direct.ranges));
    if (state->direct.results)
        record(cudaFree(state->direct.results));
    if (state->completed_ctas)
        record(cudaFree(state->completed_ctas));
    if (state->device_segments)
        record(cudaFree(state->device_segments));
    if (state->host_segments)
        record(cudaFreeHost(state->host_segments));
    if (state->descriptors_uploaded)
        record(cudaEventDestroy(state->descriptors_uploaded));
    if (state->kernel_complete)
        record(cudaEventDestroy(state->kernel_complete));
    if (previous_device != state->config.device)
        record(cudaSetDevice(previous_device));
    delete state;
    *state_ptr = nullptr;
    return static_cast<int>(first);
}

extern "C" int megamoe_tma_copy_create(uint64_t max_segments, int sms, int warps, MegamoeTmaCopyState** out)
{
    if (!out)
        return static_cast<int>(cudaErrorInvalidValue);
    *out = nullptr;
    if (!max_segments || max_segments > SIZE_MAX / (sizeof(MegamoeTmaCopySegment) + sizeof(MegamoeTmaCopyRange))
        || sms < 0 || sms > MEGAMOE_TMA_COPY_MAX_SMS || warps < 0 || warps > MEGAMOE_TMA_COPY_MAX_WARPS)
        return static_cast<int>(cudaErrorInvalidValue);
    auto* state = new (std::nothrow) MegamoeTmaCopyState;
    if (!state)
        return static_cast<int>(cudaErrorMemoryAllocation);
    auto fail = [&state](cudaError_t error)
    {
        megamoe_tma_copy_destroy(&state);
        return static_cast<int>(error);
    };
    cudaError_t error = cudaGetDevice(&state->config.device);
    if (error != cudaSuccess)
    {
        delete state;
        return static_cast<int>(error);
    }
    cudaDeviceProp prop{};
    error = cudaGetDeviceProperties(&prop, state->config.device);
    if (error != cudaSuccess)
        return fail(error);
    int runtime_version = 0, multicast_supported = 0;
    error = cudaRuntimeGetVersion(&runtime_version);
    if (error != cudaSuccess)
        return fail(error);
    if (runtime_version < 13010 || prop.major < 9)
        return fail(cudaErrorNotSupported);
    // Multicast support is exposed by the Driver API, not cudaDeviceAttr.
    CUdevice driver_device;
    CUresult driver_error = cuDeviceGet(&driver_device, state->config.device);
    if (driver_error != CUDA_SUCCESS)
        return fail(cudaErrorUnknown);
    driver_error = cuDeviceGetAttribute(&multicast_supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, driver_device);
    if (driver_error != CUDA_SUCCESS)
        return fail(driver_error == CUDA_ERROR_NOT_SUPPORTED ? cudaErrorNotSupported : cudaErrorUnknown);
    if (!multicast_supported)
        return fail(cudaErrorNotSupported);
    auto& config = state->config;
    config.abi_version = MEGAMOE_TMA_COPY_ABI_VERSION;
    config.device_sm_count = prop.multiProcessorCount;
    config.sms = megamoe_tma_copy_sm_count(sms, prop.multiProcessorCount);
    config.warps = warps ? warps : 7;
    config.threads_per_cta = config.warps * MEGAMOE_TMA_COPY_THREADS_PER_WORKER;
    config.slice_bytes = kSlice;
    config.max_segments = max_segments;
    config.compute_major = prop.major;
    config.compute_minor = prop.minor;
    if (!config.sms)
        return fail(cudaErrorInvalidValue);
    error = cudaDeviceGetAttribute(
        &config.device_optin_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, config.device);
    if (error != cudaSuccess)
        return fail(error);
    error = cudaDeviceGetAttribute(
        &config.device_shared_bytes_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, config.device);
    if (error != cudaSuccess)
        return fail(error);
    cudaFuncAttributes attributes{};
    error = cudaFuncGetAttributes(&attributes, tma_copy_kernel);
    if (error != cudaSuccess)
        return fail(error);
    const size_t available = config.device_optin_shared_bytes > attributes.sharedSizeBytes
        ? config.device_optin_shared_bytes - attributes.sharedSizeBytes
        : 0;
    config.total_slots = megamoe_tma_copy_total_slots(static_cast<int>(available));
    config.max_warps = megamoe_tma_copy_max_warps(
        config.total_slots, std::min(prop.maxThreadsPerBlock, attributes.maxThreadsPerBlock));
    if (config.warps > config.max_warps)
        return fail(cudaErrorInvalidConfiguration);
    const size_t raw = static_cast<size_t>(config.total_slots) * (kSlotDataBytes + kSlotControlBytes);
    const size_t dynamic_bytes = (raw + 127) & ~size_t(127);
    if (dynamic_bytes + attributes.sharedSizeBytes <= static_cast<size_t>(config.device_shared_bytes_per_sm) / 2)
        return fail(cudaErrorNotSupported);
    config.slots_per_warp = config.total_slots / config.warps;
    config.extra_slot_warps = config.total_slots % config.warps;
    config.max_slots_per_warp = config.slots_per_warp + (config.extra_slot_warps != 0);
    config.bank0_slots_per_warp = (config.slots_per_warp + 1) / 2;
    config.bank1_slots_per_warp = config.slots_per_warp / 2;
    config.dynamic_shared_bytes = static_cast<int>(dynamic_bytes);
    error = cudaFuncSetAttribute(
        tma_copy_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(available));
    if (error != cudaSuccess)
        return fail(error);
    error = cudaFuncSetAttribute(
        tma_copy_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    if (error != cudaSuccess)
        return fail(error);
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &config.max_active_ctas_per_sm, tma_copy_kernel, config.threads_per_cta, config.dynamic_shared_bytes);
    if (error != cudaSuccess)
        return fail(error);
    if (config.max_active_ctas_per_sm != 1)
        return fail(cudaErrorNotSupported);
    // Keep the original geometry valid even if a future compiler needs more
    // resources for the inline entry. In that case use the upload path.
    cudaFuncAttributes inline_attributes{};
    error = cudaFuncGetAttributes(&inline_attributes, tma_copy_inline_kernel);
    if (error != cudaSuccess)
        return fail(error);
    if (config.threads_per_cta <= inline_attributes.maxThreadsPerBlock
        && dynamic_bytes + inline_attributes.sharedSizeBytes <= static_cast<size_t>(config.device_optin_shared_bytes))
    {
        error = cudaFuncSetAttribute(
            tma_copy_inline_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamic_shared_bytes);
        if (error != cudaSuccess)
            return fail(error);
        error = cudaFuncSetAttribute(
            tma_copy_inline_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        if (error != cudaSuccess)
            return fail(error);
        int inline_active_ctas = 0;
        error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &inline_active_ctas, tma_copy_inline_kernel, config.threads_per_cta, config.dynamic_shared_bytes);
        if (error != cudaSuccess)
            return fail(error);
        state->inline_supported = inline_active_ctas == 1;
    }
    // Pack the used ranges immediately after the used segments on each submit.
    // One cold allocation per side supports a single exact-size HtoD upload.
    static_assert(sizeof(MegamoeTmaCopySegment) % alignof(MegamoeTmaCopyRange) == 0, "packed range alignment");
    const size_t descriptor_bytes
        = static_cast<size_t>(max_segments) * (sizeof(MegamoeTmaCopySegment) + sizeof(MegamoeTmaCopyRange));
    error = cudaMalloc(reinterpret_cast<void**>(&state->device_segments), descriptor_bytes);
    if (error != cudaSuccess)
        return fail(error);
    error = cudaHostAlloc(reinterpret_cast<void**>(&state->host_segments), descriptor_bytes, cudaHostAllocDefault);
    if (error != cudaSuccess)
        return fail(error);
    error = cudaMalloc(reinterpret_cast<void**>(&state->completed_ctas), sizeof(unsigned));
    if (error != cudaSuccess)
        return fail(error);
    error = cudaMemset(state->completed_ctas, 0, sizeof(unsigned));
    if (error != cudaSuccess)
        return fail(error);
    // Cold initialization must be visible even when the first launch uses a
    // non-default stream. Subsequent launches rearm entirely on the GPU.
    error = cudaStreamSynchronize(nullptr);
    if (error != cudaSuccess)
        return fail(error);
    error = cudaEventCreateWithFlags(&state->descriptors_uploaded, cudaEventDisableTiming);
    if (error != cudaSuccess)
        return fail(error);
    error = cudaEventCreateWithFlags(&state->kernel_complete, cudaEventDisableTiming);
    if (error != cudaSuccess)
        return fail(error);
    *out = state;
    return static_cast<int>(cudaSuccess);
}

static int submit_impl(MegamoeTmaCopyState* state, MegamoeTmaCopySegment const* segments, uint64_t segment_count,
    MegamoeTmaCopyRange const* ranges, uint64_t range_count, void* cuda_stream, uint64_t flag_mc, uint64_t generation)
{
    if ((flag_mc && ((flag_mc & 7U) || !generation)) || (!flag_mc && generation))
        return static_cast<int>(cudaErrorInvalidValue);
    if (!state || state->direct_configured || segment_count > state->config.max_segments
        || range_count > state->config.max_segments || (segment_count && !segments) || (range_count && !ranges))
        return static_cast<int>(cudaErrorInvalidValue);
    std::lock_guard<std::mutex> lock(state->submit_mutex);
    if (state->poison != cudaSuccess)
        return static_cast<int>(state->poison);
    int current_device = -1;
    cudaError_t error = cudaGetDevice(&current_device);
    if (error != cudaSuccess)
        return static_cast<int>(error);
    if (current_device != state->config.device)
        return static_cast<int>(cudaErrorInvalidDevice);
    auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    cudaStreamCaptureStatus capture;
    error = cudaStreamIsCapturing(stream, &capture);
    if (error != cudaSuccess)
        return static_cast<int>(error);
    if (capture != cudaStreamCaptureStatusNone)
        return static_cast<int>(cudaErrorStreamCaptureUnsupported);
    uint64_t total_slices = 0;
    error = validate_descriptors(segments, segment_count, ranges, range_count, &total_slices);
    if (error != cudaSuccess)
        return static_cast<int>(error);
    bool const inline_descriptors
        = state->inline_supported && range_count && segment_count <= kInlineSegments && range_count <= kInlineRanges;
    // Only the upload path touches the pinned tables. Inline launch arguments
    // have their own storage, including after an outstanding large-plan DMA.
    // Device tables and the READY counter retain cross-stream completion waits.
    if (range_count && !inline_descriptors && state->upload_recorded)
    {
        error = cudaEventSynchronize(state->descriptors_uploaded);
        if (error != cudaSuccess)
        {
            state->poison = error;
            return static_cast<int>(error);
        }
    }
    if (state->completion_recorded)
    {
        error = cudaStreamWaitEvent(stream, state->kernel_complete, 0);
        if (error != cudaSuccess)
        {
            state->poison = error;
            return static_cast<int>(error);
        }
    }
    if (!range_count && !flag_mc)
        return static_cast<int>(cudaSuccess);
    state->last_stream = stream;
    state->work_queued = true;
    auto fail = [state](cudaError_t e)
    {
        state->poison = e;
        return static_cast<int>(e);
    };
    if (range_count && !inline_descriptors)
    {
        const size_t segment_bytes = static_cast<size_t>(segment_count) * sizeof(MegamoeTmaCopySegment);
        const size_t range_bytes = static_cast<size_t>(range_count) * sizeof(MegamoeTmaCopyRange);
        auto* host_ranges = reinterpret_cast<unsigned char*>(state->host_segments) + segment_bytes;
        std::memcpy(state->host_segments, segments, segment_bytes);
        std::memcpy(host_ranges, ranges, range_bytes);
        error = cudaMemcpyAsync(
            state->device_segments, state->host_segments, segment_bytes + range_bytes, cudaMemcpyHostToDevice, stream);
        if (error != cudaSuccess)
            return fail(error);
        error = cudaEventRecord(state->descriptors_uploaded, stream);
        if (error != cudaSuccess)
            return fail(error);
        state->upload_recorded = true;
    }
    auto const& config = state->config;
    int const ctas = range_count ? config.sms : 1;
    int const threads = range_count ? config.threads_per_cta : 32;
    int const shared_bytes = range_count ? config.dynamic_shared_bytes : 0;
    if (inline_descriptors)
    {
        InlineDescriptors descriptors{};
        std::memcpy(descriptors.segments, segments, static_cast<size_t>(segment_count) * sizeof(MegamoeTmaCopySegment));
        std::memcpy(descriptors.ranges, ranges, static_cast<size_t>(range_count) * sizeof(MegamoeTmaCopyRange));
        tma_copy_inline_kernel<<<ctas, threads, shared_bytes, stream>>>(descriptors, range_count, total_slices,
            config.warps, config.total_slots, state->completed_ctas, flag_mc, generation);
    }
    else
    {
        tma_copy_kernel<<<ctas, threads, shared_bytes, stream>>>(state->device_segments,
            reinterpret_cast<MegamoeTmaCopyRange const*>(state->device_segments + segment_count), range_count,
            total_slices, config.warps, config.total_slots, state->completed_ctas, flag_mc, generation);
    }
    error = cudaGetLastError();
    if (error != cudaSuccess)
        return fail(error);
    error = cudaEventRecord(state->kernel_complete, stream);
    if (error != cudaSuccess)
        return fail(error);
    state->completion_recorded = true;
    return static_cast<int>(cudaSuccess);
}

extern "C" int megamoe_tma_copy_submit(MegamoeTmaCopyState* state, MegamoeTmaCopySegment const* segments,
    uint64_t segment_count, MegamoeTmaCopyRange const* ranges, uint64_t range_count, void* cuda_stream)
{
    return submit_impl(state, segments, segment_count, ranges, range_count, cuda_stream, 0, 0);
}

extern "C" int megamoe_tma_copy_submit_notify(MegamoeTmaCopyState* state, MegamoeTmaCopySegment const* segments,
    uint64_t segment_count, MegamoeTmaCopyRange const* ranges, uint64_t range_count, uint64_t flag_mc,
    uint64_t generation, void* cuda_stream)
{
    if (!flag_mc || !generation)
        return static_cast<int>(cudaErrorInvalidValue);
    return submit_impl(state, segments, segment_count, ranges, range_count, cuda_stream, flag_mc, generation);
}

extern "C" int megamoe_tma_copy_config_info(MegamoeTmaCopyState const* state, MegamoeTmaCopyConfig* out)
{
    if (!state || !out)
        return static_cast<int>(cudaErrorInvalidValue);
    *out = state->config;
    return static_cast<int>(cudaSuccess);
}

extern "C" char const* megamoe_tma_copy_error_string(int error)
{
    if (error == static_cast<int>(cudaErrorInvalidConfiguration))
        return "invalid TMA launch geometry: each communication warp requires at least "
               "two 8 KiB slices; shared-memory capacity and kernel/device thread limits apply";
    return cudaGetErrorString(static_cast<cudaError_t>(error));
}

extern "C" int megamoe_tma_copy_configure_gpu_plan(MegamoeTmaCopyState* state, MegamoeTmaGpuPlanConfig const* input)
{
    if (!state || !input)
        return cudaErrorInvalidValue;
    std::lock_guard<std::mutex> lock(state->submit_mutex);
    if (state->work_queued || state->direct_configured || state->direct_tables)
        return cudaErrorInvalidValue;
    auto& d = state->direct;
    auto const& c = *input;
    if (c.world < 2 || c.world > 32 || (c.world & 1) || c.rank < 0 || c.rank >= c.world || c.global_experts % c.world
        || c.home_count != c.global_experts / c.world || c.plan_abi_version != 6
        || (c.route_features != 0 && c.route_features != 1) || c.tma_route < 0 || c.tma_route > 2
        || c.tma_source_load_percent < 1 || c.tma_source_load_percent > 199 || c.helper_count <= 0
        || c.helper_count > (INT32_MAX - 25) / 7 || c.global_experts <= 0 || c.global_experts > 384 || c.planes <= 0
        || c.planes > 32 || c.level_count <= 0 || c.level_count > 4 || c.target_count < 0 || c.target_count > 32
        || !c.src_table || !c.dst_table || !c.plane_bytes || (c.route_features && !c.foreign_dst_table))
        return cudaErrorInvalidValue;
    if (c.owner_stride != ((c.helper_count + 3) & ~3) || c.plan_words != 4 + 7 * c.owner_stride)
        return cudaErrorInvalidValue;
    int expected_targets = 0;
    for (int level = 0; level < c.level_count; ++level)
    {
        int group = c.group_sizes[level];
        if (group < (level ? 4 : 2) || group > c.world || c.world % group || (!level && group != c.world)
            || (level && (group >= c.group_sizes[level - 1] || c.group_sizes[level - 1] % group)))
            return cudaErrorInvalidValue;
        if (level)
            expected_targets += c.world / group;
    }
    if (c.target_count != expected_targets || (c.route_features && !expected_targets)
        || (!c.route_features && c.foreign_dst_table))
        return cudaErrorInvalidValue;
    uint64_t total_bytes = 0;
    for (int plane = 0; plane < c.planes; ++plane)
    {
        const uint64_t bytes = c.plane_bytes[plane];
        if (!bytes || (bytes & 3) || bytes > UINT64_MAX / c.world - total_bytes)
            return cudaErrorInvalidValue;
        total_bytes += bytes;
    }
    auto valid_address = [&](uint64_t address, int plane)
    { return address && !(address & 3) && c.plane_bytes[plane] <= UINT64_MAX - address; };
    for (uint64_t i = 0; i < uint64_t(c.global_experts) * c.planes; ++i)
        if (!valid_address(c.src_table[i], i % c.planes))
            return cudaErrorInvalidValue;
    for (uint64_t i = 0; i < uint64_t(c.level_count) * c.helper_count * c.planes; ++i)
        if (!valid_address(c.dst_table[i], i % c.planes))
            return cudaErrorInvalidValue;
    if (c.route_features)
    {
        uint64_t index = 0;
        for (int level = 1; level < c.level_count; ++level)
        {
            int group = c.group_sizes[level];
            for (int begin = 0; begin < c.world; begin += group)
                for (int slot = 0; slot < c.helper_count; ++slot)
                    for (int plane = 0; plane < c.planes; ++plane, ++index)
                    {
                        const uint64_t address = c.foreign_dst_table[index];
                        bool const member = c.rank >= begin && c.rank < begin + group;
                        if (member ? address != 0 : !valid_address(address, plane))
                            return cudaErrorInvalidValue;
                    }
        }
    }
    int device = -1;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess)
        return error;
    if (device != state->config.device)
        return cudaErrorInvalidDevice;
    auto fail = [state](cudaError_t e)
    {
        state->poison = e;
        return static_cast<int>(e);
    };
    d.config = c;
    const uint64_t sources = uint64_t(c.global_experts) * c.planes;
    const uint64_t destinations = uint64_t(c.level_count) * c.helper_count * c.planes;
    const uint64_t foreign = c.route_features ? uint64_t(c.target_count) * c.helper_count * c.planes : 0;
    const uint64_t words = sources + destinations + foreign + c.planes;
    if (words > SIZE_MAX / sizeof(uint64_t))
        return cudaErrorInvalidValue;
    error = cudaMalloc(reinterpret_cast<void**>(&state->direct_tables), words * sizeof(uint64_t));
    if (error != cudaSuccess)
        return fail(error);
    uint64_t* cursor = state->direct_tables;
    auto upload = [&](uint64_t const* host, uint64_t count, uint64_t const** target)
    {
        *target = count ? cursor : nullptr;
        auto e = count ? cudaMemcpy(cursor, host, count * sizeof(uint64_t), cudaMemcpyHostToDevice) : cudaSuccess;
        cursor += count;
        return e;
    };
    if ((error = upload(c.src_table, sources, &d.config.src_table)) != cudaSuccess
        || (error = upload(c.dst_table, destinations, &d.config.dst_table)) != cudaSuccess
        || (error = upload(c.foreign_dst_table, foreign, &d.config.foreign_dst_table)) != cudaSuccess
        || (error = upload(c.plane_bytes, c.planes, &d.config.plane_bytes)) != cudaSuccess)
        return fail(error);
    auto& config = state->config;
    cudaFuncAttributes attrs{};
    error = cudaFuncGetAttributes(&attrs, tma_copy_gpu_direct_kernel);
    if (error != cudaSuccess)
        return fail(error);
    int const available = config.device_optin_shared_bytes - static_cast<int>(attrs.sharedSizeBytes);
    config.total_slots = megamoe_tma_copy_total_slots(available);
    config.max_warps = megamoe_tma_copy_max_warps(config.total_slots, attrs.maxThreadsPerBlock);
    if (config.warps > config.max_warps)
        return fail(cudaErrorInvalidConfiguration);
    config.slots_per_warp = config.total_slots / config.warps;
    config.extra_slot_warps = config.total_slots % config.warps;
    config.max_slots_per_warp = config.slots_per_warp + (config.extra_slot_warps != 0);
    config.bank0_slots_per_warp = (config.slots_per_warp + 1) / 2;
    config.bank1_slots_per_warp = config.slots_per_warp / 2;
    config.dynamic_shared_bytes = (config.total_slots * (kSlotDataBytes + kSlotControlBytes) + 127) & ~127;
    error = cudaFuncSetAttribute(
        tma_copy_gpu_direct_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamic_shared_bytes);
    if (error != cudaSuccess)
        return fail(error);
    error = cudaFuncSetAttribute(
        tma_copy_gpu_direct_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    if (error != cudaSuccess)
        return fail(error);
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&config.max_active_ctas_per_sm, tma_copy_gpu_direct_kernel,
        config.threads_per_cta, config.dynamic_shared_bytes);
    if (error != cudaSuccess)
        return fail(error);
    if (config.max_active_ctas_per_sm != 1)
        return fail(cudaErrorNotSupported);
    const uint64_t n = config.sms;
    d.max_segments = config.max_segments;
    if (d.max_segments > SIZE_MAX / n / sizeof(MegamoeTmaCopyRange))
        return fail(cudaErrorInvalidValue);
    error = cudaMalloc(reinterpret_cast<void**>(&d.scratch), n * (c.plan_words + 4ULL * c.helper_count) * sizeof(int));
    if (error != cudaSuccess)
        return fail(error);
    error = cudaMalloc(reinterpret_cast<void**>(&d.segments), n * d.max_segments * sizeof(MegamoeTmaCopySegment));
    if (error != cudaSuccess)
        return fail(error);
    error = cudaMalloc(reinterpret_cast<void**>(&d.ranges), n * d.max_segments * sizeof(MegamoeTmaCopyRange));
    if (error != cudaSuccess)
        return fail(error);
    error = cudaMalloc(reinterpret_cast<void**>(&d.results), n * sizeof(MegamoeTmaGpuPlanResult));
    if (error != cudaSuccess)
        return fail(error);
    error = cudaMemset(d.results, 0, n * sizeof(MegamoeTmaGpuPlanResult));
    if (error != cudaSuccess)
        return fail(error);
    error = cudaStreamSynchronize(nullptr); // Cold initialization only.
    if (error != cudaSuccess)
        return fail(error);
    state->direct_configured = true;
    return cudaSuccess;
}

extern "C" int megamoe_tma_copy_bind_gpu_direct(
    MegamoeTmaCopyState* state, uint64_t ids, uint64_t levels, uint64_t owners, uint64_t workspace, int capacity)
{
    if (!state || !ids || !levels || !owners || !workspace || (ids | levels | owners | workspace) % alignof(int)
        || capacity < 1 || capacity > 384)
        return cudaErrorInvalidValue;
    std::lock_guard<std::mutex> lock(state->submit_mutex);
    if (!state->direct_configured || state->direct_bound || state->work_queued)
        return cudaErrorInvalidValue;
    int device = -1;
    auto error = cudaGetDevice(&device);
    if (error != cudaSuccess)
        return error;
    if (device != state->config.device)
        return cudaErrorInvalidDevice;
    for (auto ptr : {ids, levels, owners, workspace})
    {
        cudaPointerAttributes attr{};
        error = cudaPointerGetAttributes(&attr, reinterpret_cast<void*>(ptr));
        if (error != cudaSuccess)
            return error;
        if (attr.type != cudaMemoryTypeDevice || attr.device != device)
            return cudaErrorInvalidValue;
    }
    auto& d = state->direct;
    d.ids = reinterpret_cast<int const*>(ids);
    d.levels = reinterpret_cast<int const*>(levels);
    d.owners = reinterpret_cast<int const*>(owners);
    d.workspace = reinterpret_cast<int const*>(workspace);
    d.capacity = capacity;
    state->direct_bound = true;
    return cudaSuccess;
}

extern "C" int megamoe_tma_copy_submit_gpu_direct(
    MegamoeTmaCopyState* state, uint64_t flag_mc, uint64_t generation, void* cuda_stream)
{
    if (!state || !flag_mc || (flag_mc & 7) || !generation)
        return cudaErrorInvalidValue;
    std::lock_guard<std::mutex> lock(state->submit_mutex);
    if (state->poison != cudaSuccess)
        return state->poison;
    if (!state->direct_bound)
        return cudaErrorInvalidValue;
    int device = -1;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess)
        return error;
    if (device != state->config.device)
        return cudaErrorInvalidDevice;
    auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    cudaStreamCaptureStatus capture;
    error = cudaStreamIsCapturing(stream, &capture);
    if (error != cudaSuccess)
        return error;
    if (capture != cudaStreamCaptureStatusNone)
        return cudaErrorStreamCaptureUnsupported;
    // Raw direct ABI also fixes one stream: protecting descriptor scratch on a
    // different stream would not protect HALO-Q's already-enqueued output writes.
    if (state->work_queued && stream != state->last_stream)
        return cudaErrorInvalidValue;
    state->last_stream = stream;
    state->work_queued = true;
    auto const& c = state->config;
    tma_copy_gpu_direct_kernel<<<c.sms, c.threads_per_cta, c.dynamic_shared_bytes, stream>>>(
        state->direct, c.warps, c.total_slots, state->completed_ctas, flag_mc, generation);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        state->poison = error;
        return error;
    }
    return cudaSuccess;
}

extern "C" int megamoe_tma_copy_gpu_plan_result(MegamoeTmaCopyState* state, MegamoeTmaGpuPlanResult* result)
{
    if (!state || !result)
        return cudaErrorInvalidValue;
    std::lock_guard<std::mutex> lock(state->submit_mutex);
    if (!state->direct_configured)
        return cudaErrorInvalidValue;
    if (state->poison != cudaSuccess)
        return state->poison;
    int device = -1;
    auto device_error = cudaGetDevice(&device);
    if (device_error != cudaSuccess)
        return device_error;
    if (device != state->config.device)
        return cudaErrorInvalidDevice;
    if (state->work_queued)
    {
        auto error = cudaStreamSynchronize(state->last_stream);
        if (error != cudaSuccess)
        {
            state->poison = error;
            return error;
        }
    }
    return cudaMemcpy(result, state->direct.results, sizeof(*result), cudaMemcpyDeviceToHost);
}
