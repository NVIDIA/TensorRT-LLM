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

#pragma once

#include "kv_cache_manager_v2/sparseCache.h"

#include <cuda.h>

#include <cstddef>
#include <cstdint>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! Written to an output slot that carries no selection, i.e. selection padding.
inline constexpr int32_t kNoSparseSlot = -1;

//! These kernels enforce their preconditions by trapping, not by reporting: a violation is
//! a KVCM2 logic error, and the trap destroys the context, so no status could be read back.
//! A debug build names the condition before halting.

//! A list of units for the copy to move, compacted across rows.
//!
//! Each producer CTA claims a contiguous run with one `atomicAdd` on its list's counter and
//! writes its entries there, so the consumer walks a dense array: its grid follows the
//! machine rather than the worst case, and no warp is tied to a row that had no work. The
//! cost is that position no longer says which row an entry belongs to, so the entry says it.
//!
//! A fetch resolves one selection per launch and uses a single list, which every layer of
//! that selection then copies from. A write-back spans every layer of the life cycle at
//! once, so it takes one list per layer, chosen by the copied buffer's layer.
//!
//! Entry order is whatever the producer's CTAs raced to, which nothing depends on: the
//! entries name disjoint cells, so they may be applied in any order.
struct SparseWorkList
{
    //! [numLists][capacity] the row, the unit and the entry holding it.
    int32_t* row = nullptr;
    int32_t* ordinal = nullptr;
    int32_t* slot = nullptr;
    //! [numLists] entries written so far. Zeroed before the producer runs; the consumer
    //! bounds its loop by it, which is why the count never crosses to the host.
    int32_t* total = nullptr;

    //! Entries one list holds, and the distance between consecutive lists. Zero for a single
    //! list.
    int32_t capacity = 0;
    //! Lists behind `total`, so a producer can bound the layer it indexes.
    int32_t numLists = 1;
};

//! Fetch statistics, as the device counter and as its host copy.
//!
//! The two figures the resolve produces, not the two a reader wants: `hits()` is their
//! difference and is never stored. `misses` is the row's claimed run in the compacted miss
//! list -- the selected units that were not resident and so had to be copied from their
//! host pages.
//!
//! `unsigned long long` rather than `uint64_t` because the device side accumulates with
//! `atomicAdd`, which has no overload for `unsigned long` on LP64.
//!
//! 64-bit because a step resolves millions of units per layer and a 32-bit pair would wrap.
struct SparseHitMiss
{
    //! Selection entries that named real history. Padding and in-flight units are neither
    //! hit nor miss, so neither reaches this.
    unsigned long long resolved;
    //! Of those, the ones the cache did not hold.
    unsigned long long misses;

    //! Clamped, because a host copy races a running fetch and the two are separate atomics:
    //! a resolve can land between them and show a miss its selection has not been counted
    //! for yet. A diagnostic must not be able to report a negative.
    [[nodiscard]] unsigned long long hits() const noexcept
    {
        return resolved >= misses ? resolved - misses : 0;
    }
};

//! One selection's resolve across a batch: turn each row's selection of unit ordinals into
//! cache entry indices, allocating entries for the units that are not resident.
//!
//! The scope is one selection, which is usually one layer: a layer's selection normally
//! depends on the previous layer's output, so they are not all known at the start of a
//! step and each gets its own call, letting the first layer's copies start as early as
//! possible. Where a model runs one indexer across several layers they are known together,
//! and those layers name one selection and resolve once between them.
//!
//! One result serves every sparse buffer of every layer in the selection, since they hold
//! the same units in mirrored cells and share one metadata row. Layouts are therefore
//! [row][...], with no buffer or layer dimension. Every array is device memory.
struct SparseResolveArgs
{
    //! [numRows] the arena slot each row holds, as Batch::sparseLeases() publishes it. A
    //! row holding kNoSparseLease has no reservation -- free, or a member that has
    //! suspended -- and is skipped.
    SparseLeaseIndex const* leases;
    //! The life cycle's arena, which every row's lease is addressed within.
    SparseArenaView arena;
    int32_t numRows;
    //! Which of the life cycle's selections this call resolves: the metadata row every
    //! sparse buffer of every layer naming that selection shares.
    int32_t selectionSlot;

    //! [numRows][selectionWidth] unit ordinals, negative where the selection is padded.
    //! Ordinals must be unique within a row; duplicates would be resolved as separate
    //! misses and waste an entry each.
    int32_t const* selected;
    //! Selected units per row, uniform across requests so shapes stay static.
    int32_t selectionWidth;

    //! [numRows] history length and capacity, in tokens, as Batch::watermarks() publishes
    //! them; `unitTokens` divides both to units. Taking the source and the divisor rather
    //! than the two derived counts keeps one truth, and spares the caller a conversion pass
    //! over arrays it would otherwise have to allocate.
    //!
    //! Device-resident because both move every step: a captured graph must read the current
    //! value rather than one baked in at capture.
    //!
    //! Ordinals below historyUnits are selectable history. Ordinals in
    //! [historyUnits, capacityUnits) are in flight: the entry holds the only copy of their
    //! data, so it is pinned against eviction and there is nothing to fetch for it. See
    //! "Unflushed entries carry no state" in SPARSE_ATTENTION_DESIGN.md.
    int32_t const* historyTokens;
    int32_t const* capacityTokens;
    //! tokens_per_block / unitsPerPage, a per-layer-group constant.
    int32_t unitTokens;

    //! [numRows][selectionWidth] entry index per selection, kNoSparseSlot where padded or
    //! where the row holds no reservation. One list serves every buffer of the layer.
    int32_t* outSlot;

    //! [numRows][maxInFlightUnits] entry index per in-flight unit, in ordinal order from
    //! historyUnits. Entries past the live count are left untouched. The model writes new
    //! KV through these, so they are resolved before the selection takes any victim.
    int32_t* inFlightSlot;
    //! Array stride, bounding capacityUnits - historyUnits. Follows from the cache width,
    //! so the shape stays static while the live count varies.
    int32_t maxInFlightUnits;

    //! The history units that were not resident, and the entries given to them. One list,
    //! since a call resolves one selection; every buffer of every layer naming that
    //! selection copies from it, each into its own mirror of the entry.
    //!
    //! An in-flight unit that misses is allocated an entry but never appears here: its host
    //! page is not written until the watermark passes it, so there is nothing to fetch.
    SparseWorkList work;
    //! [numRows] entries this row contributed. The copy does not read it -- it walks the
    //! compacted list -- but it is what makes a row's hit rate observable.
    int32_t* missCount;

    //! Cumulative fetch statistics, which every row adds its own share of. Null skips it.
    //!
    //! Counted in the resolve because only the resolve can count either half: a padded
    //! selection entry is negative and an in-flight unit is not a fetch, neither of which
    //! the host can see. Both halves come from one pass, so they agree on what they count.
    //!
    //! A row adds its own contribution rather than anything reading a total afterwards. The
    //! step's counters are cleared once per publish, not per fetch, so a reader downstream
    //! would count two fetches of one layer twice.
    SparseHitMiss* hitMiss = nullptr;
};

//! Shared memory one CTA needs for `entriesPerSelection` entries and `selectionWidth`
//! selections. The probe index grows as next_pow2(2 * (selectionWidth + maxInFlightUnits));
//! per-entry recency and eviction scratch grow linearly with entriesPerSelection.
[[nodiscard]] size_t sparseResolveSharedBytes(
    int32_t entriesPerSelection, int32_t selectionWidth, int32_t maxInFlightUnits);

//! Return the resolve's shared-memory requirement after checking it against the current
//! device's opt-in limit less the kernel's static shared memory. Used when constructing
//! a Batch and before launching the kernel; also opts the kernel in to that limit.
[[nodiscard]] size_t checkedSparseResolveSharedBytes(
    int32_t entriesPerSelection, int32_t selectionWidth, int32_t maxInFlightUnits);

//! Multiprocessors `device` has, which is the one part of the grid sizing below that costs
//! a driver call and so the one part worth holding on to.
[[nodiscard]] int32_t sparseCopyMultiprocessors(CUdevice device);

//! CTAs to give a copy: enough to fill a machine of `smCount` multiprocessors, capped by
//! the most work items the list could ever hold, since beyond that the extra CTAs would
//! only exit.
//!
//! Fixed before a graph captures the copy, so it follows the machine rather than the step.
[[nodiscard]] int32_t sparseCopyGridCtas(int32_t workCapacity, int32_t smCount) noexcept;

//! Which way a unit moves between its cold page and its cache entry.
enum class SparseCopyDirection : uint8_t
{
    //! Host page -> cache entry, filling an entry the resolve allocated for a miss.
    kFetch = 0,
    //! Cache entry -> host page, writing back a unit the watermark has passed.
    kWriteBack = 1,
};

//! Vector type one pool's copy moves at a time, chosen from the layout's sizes alone.
//!
//! Sizes rather than addresses, so the width is a property of the layout and can live in a
//! descriptor built once. The two base pointers are checked against it on the host, where a
//! misaligned pool is a configuration error worth naming rather than a case to handle.
enum class SparseCopyWidth : int32_t
{
    kByte = 0,
    kWord = 1,
    kWord2 = 2,
    kWord4 = 3,
};

//! A maximal set of consecutive buffers sharing a pool, as the buffer plan groups them.
//!
//! Host-side only: it is what the manager knows about a pool, before the Batch folds each
//! buffer's own slot and index-table entry into the per-buffer descriptors below. Buffers
//! are ordered (layer, pool), so a run never spans layers.
struct SparseCopyRun
{
    int32_t bufferBase;
    int32_t numBuffers;
    //! Base of this pool's region in the cell arena, before a buffer's slot is added.
    CUdeviceptr deviceBase;
    uint32_t bufferBytes;
    int32_t cellsPerBuffer;
    int32_t cellsPerLease;
    uint32_t hostPageStride;
    SelectionUnitLayout layout;
    SparseCopyWidth width;
};

//! One buffer's share of a copy: everything a block needs that does not vary per work item.
//!
//! Device-resident and constant for the Batch's lifetime. Every field derives from the
//! buffer plan and the index tables, both fixed at construction, so these are built and
//! uploaded once -- which is what lets a captured graph replay a copy without any host
//! write, and what lets every buffer of every pool share one launch.
//!
//! Per buffer rather than per pool so a block resolves itself in a single load, and so the
//! two indirections a pool-level descriptor needs -- the buffer's slot within its pool and
//! its entry within the page table -- can be folded in here instead of being paid per work
//! item.
//!
//! Sixteen-byte aligned and exactly four of them wide, so the load vectorises; the strides
//! are 32-bit because a page, a lease and a cell are all far smaller than that, and only
//! their products with a page or lease index need the wider type.
struct alignas(16) SparseCopyDesc
{
    //! Base this buffer's converted page indices are relative to. Shared by the buffers of
    //! one run: the converted index already carries each buffer's own offset within the
    //! coalesced page, which is why the pool's slot size plays no part and the stride below
    //! is one buffer.
    CUdeviceptr hostBase;
    //! Converted page indices, already at this buffer's entry, so a row's run of blocks is
    //! `pageTable[row * pageTableStride ...]`. Buffers that merged onto one index table
    //! share an entry and so share this pointer; buffers of one request do not.
    int32_t const* pageTable;
    //! This buffer's sub-region of its pool in the cell arena, which is the pool's base
    //! plus its slot there. A lease's cells follow at `leaseIndex * leaseStride`.
    CUdeviceptr deviceBase;

    uint32_t hostPageStride;
    uint32_t pageTableStride;
    //! Blocks a row's page table describes; an ordinal past them is outside this call.
    uint32_t numBlocks;
    //! Bytes one lease takes in this pool's region, which is its cells there times the
    //! cell size. Multiplied by a lease index, so widened at the multiply.
    uint32_t leaseStride;
    //! Bytes of one cell, which is one buffer's worth of pages.
    uint32_t cellBytes;

    SelectionUnitLayout layout;

    SparseCopyWidth width;
};

static_assert(sizeof(SparseCopyDesc) == 64, "A copy descriptor is meant to be four vector loads");

//! Move units between their cold pages and their cache entries.
//!
//! The default codec relocates buffer bytes verbatim, so a unit has the same internal
//! shape on both sides and one layout describes both; only the buffer's offset within a
//! page and the unit's index differ. See "Fast path: a layout value" in
//! SPARSE_ATTENTION_DESIGN.md.
//!
//! Both directions use the same addressing: the cold side is reached through the page
//! table by unit ordinal, the hot side through the cache page list by entry index. Only
//! which of the two is the source changes.
struct SparseUnitCopyArgs
{
    //! [numRows] the arena slot each row holds; a batched copy spans rows whose cells
    //! belong to different requests and so are not contiguous. kNoSparseLease means the
    //! row holds no reservation.
    SparseLeaseIndex const* leases;
    //! The life cycle's arena: where a lease's cells sit, and the per-buffer pool slots.
    SparseArenaView arena;
    int32_t numRows;

    //! One descriptor per BufferId of the life cycle, indexed by `blockIdx.y` directly.
    SparseCopyDesc const* copyDesc;

    //! The buffers this call covers: `numBuffers` of them from `bufferBase`, which for a
    //! fetch is one layer's and for a write-back is the life cycle's. Consecutive, because
    //! buffers are grouped by layer and, within a layer, by pool.
    int32_t bufferBase;
    int32_t numBuffers;

    SparseCopyDirection direction;

    //! Selection units one page holds. A life cycle's property, so every pool shares it.
    int32_t unitsPerPage;

    //! The work list: the resolve's misses when fetching, the flush list when writing back.
    //! Compacted, so a warp takes the next entry rather than the next slot of a row that may
    //! hold nothing. The pools of one layer share it, because they hold the same units.
    SparseWorkList work;

    //! CTAs along the work dimension. A work item is one warp's worth -- a selection unit is
    //! 288 bytes of NVFP4 data or 36 of scales, which is 18 or 36 lanes, so a block per item
    //! would leave most of it idle -- and the warps walk the list with a grid-stride loop.
    //!
    //! Sized from the machine, not the work: the fetch is captured in a graph, so the grid
    //! is fixed before anyone knows how many units a step will miss. The device-side total
    //! is what the loop actually follows.
    int32_t gridCtas;
};

//! Widest vector a pool's layout admits, from its sizes alone. A layout admitting none
//! falls to bytes, so there is never a tail to handle in the kernel.
[[nodiscard]] SparseCopyWidth sparseCopyWidth(
    SelectionUnitLayout const& layout, uint64_t hostPageStride, uint64_t bufferBytes);

//! Bytes one `width` moves, for the caller that has to check its bases against it.
[[nodiscard]] size_t sparseCopyWidthBytes(SparseCopyWidth width) noexcept;

//! Launch the copies for every pool at once: the grid is `(gridCtas, numBuffers)`, where the
//! buffers of every pool form one flat range.
//!
//! One launch rather than one per pool, because nothing orders the pools against each other
//! -- their cells are disjoint and the converted table they share is read-only -- and a
//! single grid both runs them concurrently and lets every one of them carry the programmatic
//! dependency on the kernel that produced the work list.
//!
//! Enrolled in PDL when the environment enables it. The kernel resolves its pool, buffer and
//! list before waiting, so a block is resident and addressed by the time its producer drains.
void launchSparseUnitCopy(SparseUnitCopyArgs const& args, CUstream stream);

//! Turn a resolve's entry indices into the index list an attention kernel can read.
//!
//! The entry index the resolve produced is private to the cache: it names a unit within a
//! layer's partition. What a consumer needs is a value in the expanded index space,
//! `(page * buffersPerSlot + cell) * unitsPerPage + unit`.
//!
//! The result is compact -- one entry per selection slot, in the selection's own order --
//! which is the shape every sparse attention consumer takes: a token-sparse kernel reads it
//! in place of its page table, a block-sparse kernel beside one.
//!
//! The buffer dimension cannot be collapsed. A layer's buffers share the resolve but not
//! their cells, because the cache packs one buffer's entries densely across every cell of
//! the pages it reserved rather than leaving each at its nominal cell of a coalesced slot.
//! So one selection resolves to a different index per buffer.
struct SparseOutIndexArgs
{
    //! [numRows] the arena slot each row holds; a row holding kNoSparseLease is filled
    //! with kBadPageIndex and nothing else, which is what a free row should look like.
    SparseLeaseIndex const* leases;
    //! The life cycle's arena, which turns a lease into the cells its indices name.
    SparseArenaView arena;
    int32_t numRows;

    //! [numRows][selectionWidth] the ordinals selected, and the entries the resolve gave
    //! them. Negative in either array means the slot carried no selection.
    int32_t const* selected;
    int32_t const* outSlot;
    int32_t selectionWidth;

    //! Row stride of `outSlot`, when it is wider than the part this call reads. Zero means
    //! `selectionWidth`. The write-index path needs it: its slots come from the resolve's
    //! in-flight array, whose row is the reservation's headroom, while only the units
    //! actually in flight -- at most one -- are read out.
    int32_t outSlotStride = 0;

    //! [numRows][outBuffers][selectionWidth] written in full: entry `k` is the index
    //! holding `selected[row][k]` for the buffer, and kBadPageIndex where the selection is
    //! padded or the row holds no reservation. Padding stays where the selection put it
    //! rather than being packed out, so slot `k` of the output and slot `k` of the
    //! selection describe the same unit.
    //!
    //! One call fills `numBuffers` of the row's `outBuffers` entries, starting at `outBase`,
    //! because a call serves one pool while the caller asks for a whole layer.
    int32_t* out;
    int32_t outBuffers;
    int32_t outBase;

    //! The buffers this call writes, as for the copy.
    int32_t bufferBase;
    int32_t numBuffers;

    int32_t cellsPerBuffer;
    int32_t cellsPerLease;
    int32_t unitsPerPage;
};

//! Launch the index scatter. One CTA per (row, buffer).
void launchSparseOutIndices(SparseOutIndexArgs const& args, CUstream stream);

//! Collect the entries holding units the watermark has just passed, so they can be written
//! back to their host pages.
//!
//! Found by scanning the reverse map rather than probing: the units wanted are a contiguous
//! ordinal range, and the map has to be read in full either way.
struct SparseFlushListArgs
{
    //! [numRows] the arena slot each row holds. kNoSparseLease means the row holds no
    //! reservation and is skipped, as in the fetch.
    SparseLeaseIndex const* leases;
    //! The life cycle's arena, which every row's lease is addressed within.
    SparseArenaView arena;
    int32_t numRows;

    //! [numRows] each: the unit ordinals that row's watermark moved across, as [from, to).
    //! Device resident for the same reason the resolve's watermarks are.
    int32_t const* fromUnits;
    int32_t const* toUnits;

    //! Units to write back and the entries holding them: one list per layer, because the
    //! copy that consumes it walks buffers and a buffer belongs to one layer. Batch-owned,
    //! so one launch covering every row hands the copy one pointer -- the reservation in a
    //! row's lease is its own, but this is the step's.
    SparseWorkList work;
    //! [numRows][numLayers] entries this (row, layer) contributed, for the same reason the
    //! resolve keeps its miss count: the copy does not need it, an observer does.
    int32_t* outCount;
};

//! Launch the flush-list scan. One CTA per (layer, row).
void launchSparseFlushList(SparseFlushListArgs const& args, CUstream stream);

//! Launch the resolve. One CTA per row; enqueues and returns.
//!
//! Rejects shapes that are inconsistent or need more shared memory than the device allows.
void launchSparseResolve(SparseResolveArgs const& args, CUstream stream);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
