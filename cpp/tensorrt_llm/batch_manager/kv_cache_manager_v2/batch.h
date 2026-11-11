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

#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/lifeCycleRegistry.h"
#include "kv_cache_manager_v2/sparseCacheKernels.h"
#include "kv_cache_manager_v2/stagingBuffer.h"
#include "kv_cache_manager_v2/utils/cudaEvent.h"
#include "kv_cache_manager_v2/utils/generator.h"
#include "kv_cache_manager_v2/utils/staticHostMem.h"

#include <cuda.h>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

class KvCache;
class KvCacheManager;

//! One layer group's page table as the device sees it. Always packed: the Batch allocates
//! it, so a row is exactly `blocksPerRow` entries.
struct GpuCoalescedPageTable
{
    CUdeviceptr data = 0;
    int32_t numRows = 0;
    int32_t blocksPerRow = 0;
};

//! One index table's converted page indices as the device sees them, laid out
//! [numRows][numBuffers][blocksPerRow]. A buffer reads the entry `index_table_slot` gives
//! it.
//!
//! Row-major in the entries so a request's buffers sit together: an attention kernel takes
//! a key and value page list as one pair -- XQA's is
//! [batchSize][beamWidth][2][maxNbPagesPerSeq] -- and this is that shape, so a table can be
//! handed over as it stands instead of being repacked every step.
struct ConvertedIndexTable
{
    CUdeviceptr data = 0;
    int32_t numRows = 0;
    //! One per buffer of the requests this table serves; they differ only by the offset
    //! baked into them.
    int32_t numBuffers = 0;
    int32_t blocksPerRow = 0;
};

//! What a converted index means: the address it is relative to, and what it counts.
//!
//! A property of the spec rather than of the table it landed on. Specs whose converted
//! entries coincide are merged onto one table, and reading that one set of entries through
//! different bases is exactly what keeps a quantised layer's data and its block scales
//! apart. So a table has as many of these as it serves specs, and only the spec picks one.
struct PoolSpec
{
    //! What a converted index is added to.
    MemAddress base = 0;
    //! Bytes a converted index counts, which is the pool's page stride.
    size_t pageStride = 0;
};

//! One lifecycle's page table as the host writes it, in the Batch's pinned staging.
struct CoalescedPageTable
{
    //! Points into the pinned staging block, which publish() uploads asynchronously. A host
    //! write here races that copy and is in any case overwritten by the next resize().
    int32_t const* data = nullptr;
    int32_t numRows = 0;
    int32_t blocksPerRow = 0;
};

//! Identifies one converted index table of a Batch.
using IndexTableId = StrongIndex<int, struct IndexTableIdTag>;

//! Device scratch one lifecycle's step works through: what a fetch resolves into before
//! the copy and the index scatter read it back, and what a write-back lists before its own
//! copy reads it.
//!
//! Batch-owned: both consumers are one launch spanning every row, and every extent is a
//! configuration constant, so one allocation serves every step and the step allocates
//! nothing -- which is what lets a CUDA graph capture it.
struct SparseStepScratch
{
    int32_t* outSlot = nullptr;      //!< [numRows][selectionWidth]
    int32_t* missCount = nullptr;    //!< [numRows]
    int32_t* inFlightSlot = nullptr; //!< [numRows][maxInFlightUnits]
    int32_t selectionWidth = 0;
    int32_t maxInFlightUnits = 0;

    //! The misses of one selection, compacted across rows. A fetch resolves one selection
    //! at a time, so a single list serves the whole batch -- and every layer sharing that
    //! selection, each copying the same units into its own cells.
    SparseWorkList fetchWork;

    //! The write-back's counterpart. A write-back crosses every layer of a row at once and
    //! the copy that consumes it walks buffers, so it takes one list per layer rather than
    //! the one the fetch folds away.
    SparseWorkList flushWork;
    int32_t* flushCount = nullptr; //!< [numRows][numLayers]
    int32_t numLayers = 0;

    //! The Batch's cumulative fetch statistics, the same pointer for every life cycle.
    //! Carried here so a fetch reaches it the way it reaches the rest of its arguments.
    SparseHitMiss* hitMiss = nullptr;

    //! What differs between the pools of a copy, on the device for the kernel to index by
    //! `blockIdx.z`. Built and uploaded once with the Batch: every field comes from the
    //! buffer plan and the index tables, both fixed at construction, so a captured graph
    //! replays a copy with no host write at all.
    //!
    //! [numBuffers] one copy descriptor per BufferId, serving both directions. BufferIds
    //! are ordered (layer, pool), so one already names its own layer's index-table entry
    //! and needs no layer dimension; the fetch and the write-back then want the same
    //! descriptors and differ only in which of them they launch over.
    SparseCopyDesc const* copyDesc = nullptr;
    int32_t numBuffers = 0;
};

//! Per-row watermarks, in tokens. Everything a step needs about how far a request has got
//! derives from these two, at whatever granularity the consumer works in:
//!
//!     num_blocks     = historyLength / tokens_per_block     (valid page table entries)
//!     history_units  = historyLength / unit_tokens          (selectable units)
//!     capacity_units = capacity / unit_tokens               (the in-flight span's end)
//!
//! Publishing the source rather than the three derived counts keeps one truth; the divisors
//! are per-layer-group constants a kernel already has.
//!
//! A free row reads zero, so it describes a request with nothing eligible.
struct WatermarkView
{
    CUdeviceptr historyLength = 0;
    CUdeviceptr capacity = 0;
    int32_t numRows = 0;
};

// ---------------------------------------------------------------------------
// Batch — the KvCaches stepped together, and their device-side mirror.
//
// Owns the row order and the per-layer-group page tables the sparse fetch reads. Rows are
// slots, not positions in a compacted list: removing a member leaves a hole rather than
// shifting later rows, so array shapes and addresses survive a CUDA graph capture.
//
// Membership is exclusive. A KvCache detaches itself on destruction, so a dropped request
// cannot leave a dangling row.
// ---------------------------------------------------------------------------
class Batch
{
public:
    //! @param indexTables what consumers need addressable together. Requests whose
    //! converted indices coincide are merged onto one table, which is how a quantised
    //! layer's data and block scales come to share one.
    //!
    //! @param maxBatchSize    rows, i.e. the most requests that can be stepped together.
    //! @param maxBlocksPerSeq columns, i.e. the longest sequence a row can describe.
    //!
    //! Both bound device arrays whose shape must not change between steps, so neither can
    //! be grown later. Left unset they derive from the manager's `constraints`, which
    //! throws when none were declared.
    //!
    //! @param mode page index mode, applied to every life cycle. Left unset, each life
    //! cycle takes the smaller table -- SHARED where the manager says it is always
    //! available, PER_LAYER otherwise.
    //!
    //! Set, it overrides that answer rather than being checked against it, and SHARED may
    //! be named where KvCacheManager::supportsIndexMode() returns nullopt. That answer
    //! covers every request the life cycle could carry, so a sliding window leaves it
    //! undecided; naming SHARED is the caller's narrower claim that no member of *this*
    //! batch will hold scratch slots there. A decode-only batch is the case it exists for:
    //! scratch covers the blocks a prefill chunk adds, and a one-token request adds none.
    //!
    //! The claim is validated, not trusted: every publish() checks each member against
    //! KvCache::supportsIndexMode(SHARED, lc) and throws if one holds scratch after all.
    //! Seat only members that keep it -- the throw arrives mid-step, with the batch already
    //! built around the smaller table.
    //!
    //! The mode is fixed here because a converted table's shape is baked into a captured
    //! graph. It cannot follow a member's scratch state, which varies per step.
    //!
    //! @param stream the stream every member runs on: a member's own work and the Batch's
    //! transfers have to be ordered against each other. setCudaStream() moves the Batch and
    //! its members together, which is the only way it changes. The caller keeps it -- a
    //! Batch neither creates nor destroys it, and it must outlive the Batch.
    //!
    //! @param maxTotalInputTokens tokens one step may ingest across every row, as the
    //! runtime's own batching bound gives it. It sizes the write-back's work list: a row
    //! advancing `t` tokens crosses at most `t / unitTokens + 1` units, so the batch
    //! crosses at most `maxTotalInputTokens / unitTokens + maxBatchSize` of them.
    //!
    //! Left unset it falls back to `maxBatchSize * maxInFlightUnits`, which is
    //! `maxBatchSize` -- a decode step advances capacity one token per row, so one row
    //! crosses at most one unit boundary. Pass it once a step can ingest more than that,
    //! which is what chunked prefill under in-flight batching does.
    explicit Batch(KvCacheManager& manager, CUstream stream, std::vector<IndexTableSpec> indexTables,
        std::optional<int> maxBatchSize = std::nullopt, std::optional<int> maxBlocksPerSeq = std::nullopt,
        std::optional<PageIndexMode> mode = std::nullopt, std::optional<int> maxTotalInputTokens = std::nullopt);

    //! The stream the Batch and every member run on.
    [[nodiscard]] CUstream cudaStream() const noexcept
    {
        return mCudaStream;
    }

    //! Moves the Batch and every member onto `stream`.
    //!
    //! The new stream is made to wait on what the old one already has queued, so the
    //! caller need not synchronize: the device arrays and the staging buffer are the same
    //! objects either side of the move, and work that names them stays ordered.
    //!
    //! A KvCache's own stream setter refuses while it is a member.
    //!
    //! Like the constructor, this takes no ownership: the caller's stream must outlive the
    //! Batch, or be replaced here before it is destroyed.
    void setCudaStream(CUstream stream);

    //! Page index mode of one life cycle, as chosen at construction.
    [[nodiscard]] PageIndexMode indexMode(LayerGroupId layerGroupId) const;

    // ---- Converted index tables --------------------------------------------

    [[nodiscard]] IndexTableId numIndexTables() const noexcept;

    //! Every request one table satisfies. A request's buffer order is its own mapping into
    //! the buffer dimension, so the first buffer of each sits at entry zero.
    [[nodiscard]] std::vector<IndexTableSpec> const& indexTable(IndexTableId tableId) const;

    //! Which table serves this spec. The spec must be one the Batch was built with.
    //!
    //! Equivalent specs resolve to the same id -- a layer's data and its block scales name
    //! one table between them. The spec's buffers occupy entries [0, buffers.size()) in its
    //! own order.
    [[nodiscard]] IndexTableId indexTableOf(IndexTableSpec const& spec) const;

    //! How to turn this spec's converted indices into addresses.
    //!
    //! Keyed by the spec because the base is: a merged table serves several specs and each
    //! reads its entries through its own pool. The spec must be one the Batch was built
    //! with, and its buffers must share a page stride -- they sit in one pool, so one base
    //! serves them all.
    //!
    //! The result is stable for the Batch's lifetime, so a captured graph may bake it.
    [[nodiscard]] PoolSpec poolSpecFor(IndexTableSpec const& spec) const;

    //! One table's converted indices: int32[max_batch][entries][blocks].
    //!
    //! A row carries every entry before the next row starts, which is what puts a spec's
    //! key and value planes side by side. So one entry's plane is strided, not contiguous:
    //! it begins at `entry * blocks` elements in and repeats every `entries * blocks`.
    //!
    //! The address is stable for the Batch's lifetime, so a captured graph may bake it.
    [[nodiscard]] ConvertedIndexTable convertedIndexTable(IndexTableId tableId) const;

    ~Batch();

    Batch(Batch const&) = delete;
    Batch& operator=(Batch const&) = delete;

    //! Seat a cache in `row`, or in the lowest free row when none is named.
    //!
    //! Seat a request only once it is ready to be stepped. Seating takes the sparse arena
    //! lease, so a request that holds pages without being stepped -- a disaggregated one
    //! still receiving its KV, or one sitting out a step -- costs the arena nothing while it
    //! stays out. Nothing here checks it: an early seat costs arena pressure, not a throw.
    //!
    //! Throws when the row is taken or out of range, the batch is full, the cache already
    //! belongs to a batch, or its beam width exceeds one -- only beam 0 is mirrored, so a
    //! wider request would have every beam read beam 0's pages.
    BatchRow addKvCache(KvCache& kvCache, std::optional<BatchRow> row = std::nullopt);

    //! Free the cache's row, leaving a hole, and hand back the reference the Batch held.
    //! Throws when it is not a member.
    //!
    //! Returns the arena lease, which discards whatever the GPU cache held: a lease comes back
    //! with every entry free. Call flushSparseUnits() first for anything worth keeping.
    std::shared_ptr<KvCache> removeKvCache(KvCache& kvCache);

    //! The same, by the row addKvCache returned. Null when nothing is seated there.
    std::shared_ptr<KvCache> removeKvCache(BatchRow row);

    //! Move the member at `srcRow` to the empty `dstRow`, leaving `srcRow` free.
    //!
    //! Membership does not lapse: the arena lease, the GPU cache contents behind it and the
    //! stream binding all survive, which is what separates this from removeKvCache() followed
    //! by addKvCache(). That pair would hand the lease back -- a lease returns with every
    //! entry free -- and so would discard residency the member is still entitled to.
    //!
    //! Throws when either row is out of range, when `srcRow` holds nothing, or when `dstRow`
    //! is occupied -- including `dstRow == srcRow`, which is both at once.
    //!
    //! Waits for the previous staging upload before touching either row. Device tables retain
    //! their addresses; the move reaches them on the next publish.
    void move(BatchRow dstRow, BatchRow srcRow);

    //! Resize every member, then publish once.
    //!
    //! Waits on the previous publish before touching any member. Per-request operations with
    //! no Batch equivalent -- commit(), a beam fork, an SWA recycle -- still need an explicit
    //! waitPublished().
    //!
    //! One result per row, in row order; false for a free row. A member that fails rolls
    //! itself back, so a batch can partially fail and the table published afterwards
    //! describes the post-rollback state.
    //!
    //! `capacities` and `historyLengths` are indexed by row, so both must be maxBatchSize
    //! long. A nullopt leaves that dimension alone, matching KvCache::resize.
    //! Rejects, before touching any member, a history length that falls or that outruns the
    //! capacity its row already had: capacity granted by this call makes room for tokens
    //! written after it, so history can only cover tokens whose space was already there.
    //!
    //! A row whose growth fails still takes its history advance, which is legal by that same
    //! check and needs no slots -- both the stale range and the scratch range are monotone
    //! in history, so raising it only releases. Its result is still 0: the row is not at the
    //! state that was asked for.
    std::vector<uint8_t> resize(
        std::vector<std::optional<int>> const& capacities, std::vector<std::optional<int>> const& historyLengths);

    //! Copy the staged rows to the device, enqueuing on the Batch's stream.
    //!
    //! The staging rows are written by the members themselves, so they are always current;
    //! what this adds is a stream-ordered snapshot. Without it the host would be mutating
    //! the table a previously enqueued fetch is still reading.
    //!
    //! Two stages. The host-to-device fill runs on an upload stream of this Batch's own, so
    //! it can overlap the previous step; the Batch's stream then carries one linear
    //! device-to-device copy into the tables the fetch reads. Everything expensive is in the
    //! first stage, and only the second has to wait for anything.
    //!
    //! **The second stage must be ordered after the kernels that read the previous
    //! publish.** Those live in a captured graph, so the tables cannot be double-buffered
    //! out of the hazard -- the capture baked their addresses, which is why the staging
    //! sits upstream of them instead. The Batch's stream being the one the graph replays on
    //! is what provides that ordering. A torn publish is not a stale answer: the cache
    //! views carry pointers, so it can pair one request's residency with another's lease.
    //!
    //! The copy is asynchronous and reads pinned memory, so it is still in flight when this
    //! returns. **Every later host write to a member's page indices must be separated from
    //! it by waitPublished()** -- a KvCache::resize(), a commit(), a beam fork, an SWA
    //! recycle, anything that locks or unlocks a page. Writing sooner races the copy and
    //! publishes a torn table. This Batch's own mutators are not in that list: resize(),
    //! move(), addKvCache() and removeKvCache() each wait for themselves.
    //!
    //! Throws when the previous publish has not been waited on, which is how that rule is
    //! enforced: this call stages the watermarks and views itself, so it would be writing
    //! into the same hazard. It refuses rather than waiting because a wait here covers only
    //! its own write -- a member mutated before the call has already raced by then, and
    //! waiting would hide it.
    void publish();

    //! Write back the units each row's watermark is about to pass, given the history the
    //! caller is about to set. One per row; a row may repeat its current history, which
    //! asks for nothing.
    //!
    //! **Call this before advancing history, not after.** A block must be in its host page
    //! before it can be selected, and selectable means below the watermark; resize() refuses
    //! a history this has not covered.
    //!
    //! **Flush only what is final.** A history given here is one the row is going to take up:
    //! the units below it are written back once and the row is then expected to reach that
    //! history, which is what lets removeKvCache() tell a row that kept everything it meant
    //! to from one that left units behind in its cells. A history past the row's capacity is
    //! refused -- those units were never written, so they have no cell to write back from.
    //!
    //! Bounded by what was in flight, not by how far history travels: only a unit that was
    //! in flight has a cell holding its one copy, so a resumed request moving history across
    //! a whole prompt writes back nothing. What is refused is a batch whose rows together
    //! cross more units than the write-back list can name -- the list is one per layer and
    //! compacted across rows, sized from `maxTotalInputTokens`, so one prefilling row may
    //! use the budget several decoding rows leave unspent.
    //!
    //! Reads the index table as of the last publish, which covers the blocks in range for
    //! the same reason -- they were allocated by an earlier step.
    void flushSparseUnits(std::vector<int> const& newHistoryLengths);

    //! Where each of `buffers` reads its indices in fetchSparsePages()' output.
    //!
    //! Buffers whose converted indices coincide share a slice. A layer's data and scale
    //! pools mirror -- equal buffer counts, so a lease spans the same cells in both -- and
    //! a selection then resolves to one index serving each, the caller applying the base
    //! of the pool it is reading. Buffers that do not coincide keep slices of their own: a
    //! single scale shared by key and value is its own index table, so it gets one.
    //!
    //! One entry per buffer, in the order given. A slice no buffer maps to is never
    //! written and holds whatever was there: read a buffer's indices at its own slot and
    //! nowhere else.
    [[nodiscard]] std::vector<int> sparseOutSlots(std::vector<BufferId> const& buffers) const;

    //! Resolve a batch's selections against each buffer's per-request history cache.
    //!
    //! Device-driven and stream-ordered throughout: the selection is read from device
    //! memory, so the host neither inspects it nor decides anything, and the call enqueues
    //! onto `stream` and returns. Every shape is static across steps, so a CUDA graph can
    //! capture it.
    //!
    //! `buffers` must be exactly one layer's sparse buffers. One layer, not one layer
    //! group, because a layer's selection is its own unless it names a selection_id: an
    //! indexer of its own reads the previous layer's output, so a step's selections are not
    //! all known at its start. Layers that do share one resolve it through
    //! resolveSparseSelection() instead, which this refuses. The whole layer, not
    //! a subset, because the resolve marks entries resident on behalf of every buffer at
    //! once: filling only some would leave the rest stale while the metadata claimed
    //! otherwise.
    //!
    //! Only for a layer whose selection is its own. Where several layers name one
    //! selection_id their selection is known for all of them at once, and resolving it once
    //! per layer would have the later resolves count the earlier ones' admissions as hits
    //! and leave their cells unfilled; those layers call resolveSparseSelection() once and
    //! fetchResolvedPages() per layer instead. This rejects them rather than quietly
    //! mis-serving them.
    //!
    //! `selected` is [maxBatchSize][selectionWidth] unit ordinals, negative where padded,
    //! and must name only units below the history watermark -- a unit at or above it has
    //! not been written back, so fetching it would copy stale bytes. That is a wrong
    //! answer rather than a memory error, and the call cannot detect it, so it is the
    //! selector's responsibility.
    //!
    //! `out` is [maxBatchSize][buffers.size()][selectionWidth], written in full, in the
    //! expanded index space getPageIndexConverter() produces: entry k is the index holding
    //! `selected[row][k]`, or kBadPageIndex where that slot carries no selection.
    //!
    //! Relative to getSparseCacheBaseAddress() for the buffer, not getMemPoolBaseAddress()
    //! and not a cache level: the pool names the buffer's pages, which are host memory for a
    //! sparse life cycle, while the cells a kernel reads live in the arena.
    //!
    //! Compact, in the selection's own order, rather than a table indexed by unit ordinal,
    //! because that is the shape sparse attention kernels take. See
    //! SPARSE_ATTENTION_DESIGN.md's "Fetch".
    //!
    //! `stream` defaults to the Batch's own, which is the only correct choice while the
    //! layers are fetched in order: a layer's selection comes from the one before it. Name
    //! a stream to run ahead of that -- a model that fetches several layers early can do so
    //! on its own -- or to be captured, since a captured fetch must name the capture stream.
    //!
    //! Naming one moves the ordering to the caller, once per step rather than once per call:
    //! `stream` must wait on the Batch's stream after the publish() that prepared the step,
    //! and the Batch's stream must wait on `stream` before any call that begins the next one.
    //! Fetches within a step need no join between them, which is what naming a stream buys.
    //! Capturing them and replaying the graph on the Batch's stream satisfies both.
    //!
    //! Both halves are load-bearing: the resolve reads Batch state the publish wrote, and
    //! writes a row's residency back into state the next step rewrites on the Batch's stream.
    void fetchSparsePages(std::vector<BufferId> const& buffers, int32_t const* selected, int selectionWidth,
        int32_t* out, std::optional<CUstream> stream = std::nullopt);

    //! Resolve one selection, admitting what it names and listing what has to be copied.
    //!
    //! The first half of fetchSparsePages, split out for the layers that share a selection:
    //! they resolve once between them and then fetch a layer at a time. `buffers` names any
    //! one member layer -- the residency and the miss list belong to the selection, not to
    //! the layer that happened to name it -- on the same terms as fetchSparsePages.
    //!
    //! It leaves two things in the life cycle's step scratch: the miss list the copies walk
    //! and the per-slot cell mapping the index writes read. Both belong to the life cycle,
    //! not to the selection, so the next resolve of any selection in it overwrites them and
    //! must be ordered after every fetchResolvedPages() that consumes the current one. One
    //! stream gives that ordering; across streams it takes an event. A model gets it by
    //! running a group's layers together, as the SWA scratch already assumes of the
    //! currently executing layer.
    void resolveSparseSelection(std::vector<BufferId> const& buffers, int32_t const* selected, int selectionWidth,
        std::optional<CUstream> stream = std::nullopt);

    //! Copy one layer's share of a resolved selection into its cells, and write its indices.
    //!
    //! The second half of fetchSparsePages. `selected` must be the array the matching
    //! resolveSparseSelection() read, since the indices written are those units'; `out` is
    //! as fetchSparsePages describes it, for this layer's buffers alone.
    void fetchResolvedPages(std::vector<BufferId> const& buffers, int32_t const* selected, int selectionWidth,
        int32_t* out, std::optional<CUstream> stream = std::nullopt);

    //! Give the in-flight unit of each request a cell, and report where to write it.
    //!
    //! The unit between the two watermarks is the one the model is about to produce. It has
    //! no host copy yet -- a unit reaches its page only when resize() carries the watermark
    //! past it -- so nothing is fetched here: the call assigns a cell and leaves its bytes
    //! to the model.
    //!
    //! Needs only the watermarks, not the selection, so it runs before fetchSparsePages.
    //!
    //! Idempotent while the watermarks and the reservation hold: the resolve probes
    //! residency before it takes a victim, so a later call -- including the fetch's own
    //! resolve -- finds the entry and reuses it rather than assigning a second. The entry
    //! is pinned for as long as its unit is in flight, so no fetch can evict it.
    //!
    //! `buffers` is one layer's sparse buffers, exactly as for fetchSparsePages.
    //!
    //! `out` is [maxBatchSize][buffers.size()], written in full, in the same expanded index
    //! space fetchSparsePages produces and relative to the same per-pool arena region:
    //! kBadPageIndex where the row holds no reservation or has nothing in flight.
    //! `stream` defaults to the Batch's own, on the same terms as fetchSparsePages.
    void prepareSparseWriteIndices(
        std::vector<BufferId> const& buffers, int32_t* out, std::optional<CUstream> stream = std::nullopt);

    //! The history each row has been written back to. A row that has never needed a flush
    //! reads its own history, so a caller may always compare against it.
    [[nodiscard]] int flushedHistory(BatchRow row) const;

    //! Block until the last publish() has finished reading the staging rows.
    //!
    //! A no-op when nothing is in flight, so a caller may use it freely at the top of a step.
    void waitPublished();

    //! One layer group's page table on the host, as a publish will upload it. Read-only:
    //! the entries are the members' to write, through their page locks.
    [[nodiscard]] CoalescedPageTable coalescedPageTable(LayerGroupId layerGroupId) const;

    //! Per-row history length and capacity, published by the same transfer as the page
    //! tables. They have to travel together: the lengths bound which entries of a row are
    //! valid, so a kernel pairing one step's table with another step's lengths would read
    //! past the end of a row that has since been reused.
    [[nodiscard]] WatermarkView watermarks() const;

    //! Row of a member; nullopt when this cache is not one.
    [[nodiscard]] std::optional<BatchRow> rowOf(KvCache const& kvCache) const;

    //! Members seated, which is the rows occupied.
    [[nodiscard]] int numKvCaches() const noexcept;

    //! Cumulative fetch statistics, in selection units, since the Batch was built or the
    //! counters were last reset.
    //!
    //! A hit is a selected unit the GPU cache already held; a miss is one the fetch had to
    //! copy from its host page. Padded selection entries and in-flight units are neither,
    //! so they reach neither figure -- which is why this comes from the resolve rather than
    //! from a count the caller could keep.
    //!
    //! Reads back on a temporary stream and syncs only that, so the counters may be a few
    //! launches stale. That does not move a ratio; a caller wanting the two to agree
    //! exactly synchronises the device first and calls this after.
    [[nodiscard]] SparseHitMiss peekHitMissCounter() const;

    //! Zero the cumulative pair. Enqueued on the Batch's stream, so it orders against the
    //! fetches around it rather than tearing one.
    void resetHitMissCounter();

    //! Device bytes every life cycle's sparse arena holds, zero without sparse offload.
    //!
    //! The arenas sit outside the cache tiers' quota, so a caller budgeting device memory
    //! must account for them separately.
    [[nodiscard]] size_t sparseArenaBytes() const;

    //! Whether the occupied rows are exactly [0, numKvCaches()).
    //!
    //! A Batch does not require it. Removing a member leaves its row free and the next join
    //! takes the lowest free one, so a hole is legal and transient.
    [[nodiscard]] bool isContiguous() const noexcept;

    [[nodiscard]] int maxBatchSize() const noexcept
    {
        return mMaxBatchSize;
    }

    [[nodiscard]] int maxBlocksPerSeq() const noexcept
    {
        return mMaxBlocksPerSeq;
    }

    //! Base of the sparse cache arena region backing this buffer, for a consumer reading
    //! units the fetch resolved. Throws when the buffer has no sparse fetch path.
    //!
    //! Role-dependent: the arena mirrors the hot pool group, so buffers of different sizes
    //! sit in regions of their own and an index the fetch emits is relative to the region of
    //! the buffer it was emitted for.
    //!
    //! Distinct from KvCacheManager::getMemPoolBaseAddress, which names the buffer's *pages*
    //! for every buffer alike -- that is what the transceiver and the KV transfer path
    //! address, and it keeps one meaning whether or not a buffer is sparse. The arena is a
    //! second allocation holding the only GPU-resident copy a kernel can read, and it is the
    //! Batch's because it is sized by the rows the Batch seats.
    [[nodiscard]] MemAddress getSparseCacheBaseAddress(LayerId layerId, DataRole const& role) const;

    //! The member seated in `row`, or null when that row is free.
    //!
    //! The inverse of rowOf(). Throws when `row` is out of range, like removeKvCache(int):
    //! a free row is an ordinary answer, so null means empty and never means "no such row".
    //!
    //! The reference stays valid for the Batch's lifetime -- the row vector is sized once at
    //! construction -- but what it holds changes as members are seated and removed, so it is
    //! a view of the slot rather than of a member.
    [[nodiscard]] std::shared_ptr<KvCache> const& operator[](BatchRow row) const;

private:
    //! Reads the private step state below in order to report it.
    friend class KvCacheIntrospection;

    //! Where one table's converted indices live.
    [[nodiscard]] CUdeviceptr indexTableAddress(IndexTableId tableId) const;

    //! How a sparse copy addresses the run of buffers beginning at (layerId, role).
    //!
    //! The run's buffers occupy consecutive entries of one request, which construction
    //! checked, so naming the first is enough: the rest follow by BufferId.
    struct SparseIndexRun
    {
        //! The run's first buffer, for row zero.
        CUdeviceptr table = 0;
        //! What a converted index is added to. Shared by the whole run.
        MemAddress base = 0;
        //! Elements between consecutive buffers, and between consecutive rows.
        int bufferStride = 0;
        int rowStride = 0;
    };

    [[nodiscard]] SparseIndexRun sparseIndexRun(LayerId layerId, DataRole const& role) const;

    //! Which table serves this buffer, and its entry in that table's buffer dimension.
    [[nodiscard]] std::pair<IndexTableId, int> indexTableSlot(LayerId layerId, DataRole const& role) const;

    //! What a converted index for this buffer is added to.
    [[nodiscard]] MemAddress indexTableBase(LayerId layerId, DataRole const& role) const;

    //! Write back every row's crossed units for one life cycle, from the ranges the Batch
    //! staged. One scan over (layer, row), then one copy per pool covering the whole batch.
    //!
    //! `stream` carries the same obligations as fetchSparsePages' when it is not the
    //! Batch's own; unlike there it has no default, so a caller naming the Batch's stream
    //! is opting into the simple case rather than out of it.
    void writeBackSparseUnits(LifeCycleId lifeCycle, CUstream stream);

    //! One layer group's page table on the device, which the index conversion reads.
    //!
    //! In the device staging rather than the mirror, because that is where the conversion
    //! reads it and nothing captured reads the raw form at all. So it holds the staged step,
    //! not the published one, and it is live: the next publish() refills it.
    //!
    //! Read it between waitPublished() and the next publish(). Before the first, the upload
    //! that fills it may not have landed; after the second is enqueued, the upload stream is
    //! free to overwrite it. The event that guards the staging against that overwrite covers
    //! the Batch's own readers, which are enqueued before it is recorded, and cannot cover a
    //! caller's.
    [[nodiscard]] GpuCoalescedPageTable gpuCoalescedPageTable(LayerGroupId layerGroupId) const;

    //! Per-row sparse cache state, as the fetch kernel sees it.
    //!
    //! A device int32 array indexed by row, holding the arena slot each row's member took.
    //! A free row, or a member whose cache is not reserved, reads as kNoSparseLease, which
    //! the kernel skips.
    //!
    //! One array rather than one per layer group: a member is seated in every sparse life
    //! cycle or none, and the arenas are sized alike, so a slot addresses the request in
    //! all of them. An index rather than the pointers it implies, because a lease's cells
    //! and its metadata are both arithmetic from the bases a launch carries once in
    //! SparseArenaView. The slot changes whenever a member resumes or suspends, which is
    //! why it is restaged on every publish alongside the watermarks.
    [[nodiscard]] CUdeviceptr sparseLeases() const;

    //! Scratch for one layer group's fetch. Null pointers when the group carries no sparse
    //! cache, which is what makes "nothing to fetch here" cheap to detect.
    [[nodiscard]] SparseStepScratch const& sparseStepScratch(LayerGroupId layerGroupId) const;

    //! The unit ordinals bounding what each row's history watermark crossed in the last
    //! flushSparseUnits(), for one life cycle: row `r` crossed [from[r], to[r]).
    //!
    //! Two parallel arrays of maxBatchSize rather than an array of pairs: the scan reads
    //! one row per thread and takes the bounds as separate pointers, so each stays
    //! coalesced. A struct per row would stride both reads by two.
    //!
    //! Ordinals are in this life cycle's units -- tokensPerBlock / unitsPerPage -- so one
    //! token count quantises differently per life cycle. That is why these are per life
    //! cycle where flushedHistory(), which is in tokens, is not.
    //!
    //! Device memory, written by flushSparseUnits() and read by the scan it launches.
    //! Stale unless that flush found work: a step where no row crossed a boundary skips
    //! the upload, which is safe only because nothing reads these then.
    [[nodiscard]] int32_t const* flushFromUnits(LifeCycleId lifeCycle) const;

    //! @see flushFromUnits
    [[nodiscard]] int32_t const* flushToUnits(LifeCycleId lifeCycle) const;

    friend class KvCache;

    //! Called by a member being destroyed. Unreachable while seated -- the Batch holds a
    //! reference, so the destructor cannot run -- and kept as the assertion of that.
    void _detach(KvCache& kvCache) noexcept;

    //! Whether the last publish() is still reading the staging rows.
    //!
    //! For a seated member to check before writing its page indices, which live in those rows.
    [[nodiscard]] bool _publishInFlight() const noexcept
    {
        return !mPublished.isClosed();
    }

    //! Host-side page-index row for one (layer group, row), inside the pinned staging block.
    [[nodiscard]] int32_t* _stagingRow(LifeCycleId lifeCycleId, BatchRow row) const;
    //! A (life cycle, row)'s slice of the scratch slot staging. Only valid once the
    //! staging exists, i.e. when mMaxScratchSlots is positive.
    [[nodiscard]] int32_t* _scratchSlotRow(LifeCycleId lifeCycleId, BatchRow row) const;

    //! Block until every transfer this Batch issued has retired, so its buffers can go.
    void _drain() noexcept;

    [[nodiscard]] size_t _rowElements() const noexcept;

    //! The buffers the staged block carries, in layout order. Each is placed for its own element, so
    //! the one holding leases follows that index rather than the int32_t beside it.
    enum class BufName : int
    {
        kPageTables = 0,
        kHistory,
        kCapacity,
        kLeases,
        kScratch,
        kCount
    };

    static constexpr std::size_t kBufCount = static_cast<std::size_t>(BufName::kCount);

    //! The least a buffer is aligned to; one whose element asks for more gets more. The
    //! floor is here so a kernel may read any of them with the widest load the hardware
    //! offers, without the layout being revisited to allow it.
    static constexpr std::size_t kBufMinAlign = 16;

    //! Place and measure every buffer once, each from its own element and alignment.
    void _layoutBufs(size_t scratchElements) noexcept;

    //! A buffer, as what it holds and how much of it. The extent travels with the pointer,
    //! so nothing reads or fills past the buffer's own bytes; the padding that aligns the
    //! next one is outside it.
    template <typename T>
    [[nodiscard]] std::span<T> _buf(BufName name) const noexcept
    {
        auto* const base = reinterpret_cast<T*>(reinterpret_cast<std::byte*>(mStaging.address()) + mBuf[name].beg);
        return std::span<T>{base, static_cast<size_t>(mBuf[name].length()) / sizeof(T)};
    }

    //! Where a buffer sits in the device staging, which carries the whole block as uploaded.
    //! The conversion reads the page tables and the scratch from there, not from the mirror.
    [[nodiscard]] CUdeviceptr _deviceBufAt(BufName name) const noexcept;

    //! Seed a row's staged watermarks, so a row reads as its occupant rather than as
    //! whatever the last one left.
    void _seedRowWatermarks(BatchRow row, int historyLength, int capacity);

    //! Refresh the staged watermarks and cache views from the members. KvCache reaches the
    //! staging only through setBasePageIndexBuf, for its page indices.
    void _stageRowState();

    //! A buffer's layout: it yields how many int32 elements it needs and resumes with
    //! the memory, so one allocation can serve several of them.
    using LayoutGenerator = Generator<size_t, int32_t*>;

    //! Yields the elements one life cycle's sparse scratch needs, then carves the slice
    //! it is handed. Knows nothing of the counters, which are the driver's.
    LayoutGenerator _layoutStepScratchForLifecycle(LifeCycleId lcId);

    //! Sums what the life cycles need, yields that plus the counter block, and hands each
    //! life cycle its slice. Drives the leaves above while being driven itself.
    LayoutGenerator _layoutStepScratch();

    //! Entries one layer's write-back list must hold, from the step's token budget.
    [[nodiscard]] int32_t _flushCapacity(LifeCycleId lifeCycle) const;

    //! Fill the per-buffer copy descriptors and upload them. Runs once, after the index
    //! tables exist, because a descriptor names where in them a buffer's indices sit.
    void _allocateCopyDesc();

    //! Device room for the per-row unit ranges a write-back reads:
    //! [life cycle][2][maxBatchSize], as [from][to]. Allocated with the fetch scratch,
    //! which has the same lifetime.
    //! Yields the elements the write-back ranges need, then clears the slice.
    LayoutGenerator _layoutFlushRanges();

    //! Points into mDeviceArena; null when no life cycle holds a sparse cache.
    int32_t* mFlushRanges = nullptr;
    //! Staged host-side, then handed to a copy that consumes it before returning.
    std::vector<int32_t> mFlushRangeStaging;
    //! Per row, the history its units have been written back to.
    TypedVec<BatchRow, int> mFlushedHistory;
    //! Recorded after the write-back, so the next call's staging can wait for its reads.
    CachedCudaEvent mFlushRead = CachedCudaEvent::makeNull();
    //! Whether the write-back counters hold a total some earlier launch left there. The
    //! list kernel claims its entries with an atomic add, so it has to start from zero.
    //! publish() clears every counter at the head of a step, which covers the one flush a
    //! step is expected to issue; a flush that follows another, or the first of all, clears
    //! its own. True before the first publish, since mDeviceArena is not zeroed.
    bool mFlushCountersDirty = true;

    //! The same array where a kernel reads it, which is what the launches are handed.
    [[nodiscard]] SparseLeaseIndex const* _leaseMirror() const noexcept;

    //! Bytes of the mirror: the staged block from the histories through the leases, which is
    //! everything a captured kernel reads. The page tables ahead of them are read from the
    //! staging, by the conversion, so carrying them across would serve nothing.
    [[nodiscard]] size_t _mirrorBytes() const noexcept;

    //! A staged element offset, as an address in the mirror.
    [[nodiscard]] CUdeviceptr _mirrorAt(BufName name) const noexcept;

    //! Size the arenas from mMaxBatchSize, so every row the Batch can seat has a lease.
    void _buildSparseArenas();

    //! Everything a sparse copy needs that follows from configuration alone: the two pools,
    //! the unit layout and the cell geometry. A caller fills in whose cache it is copying
    //! into and which work list to read.
    //!
    //! One entry per maximal run of consecutive buffers sharing a hot pool. Buffers are
    //! ordered (layer, pool), so a life cycle whose sparse buffers are alike yields a single
    //! run, and the quantised shape -- data and block scales in pools of their own -- yields
    //! one per layer and pool. A caller that wants only some buffers clips each run to them.
    [[nodiscard]] std::vector<SparseCopyRun> _sparseCopyTemplates(LifeCycleId lifeCycle) const;

    //! The fetch's single work list, with the counter belonging to `selectionSlot`.
    //!
    //! One list serves every selection of a step because each is consumed, by the copies of
    //! the layers resolving within it, before the next resolve overwrites it; the counters
    //! are per selection because they are cleared once per publish rather than between
    //! resolves, so a second resolve on a used counter would claim past the list.
    [[nodiscard]] static SparseWorkList _sparseFetchWork(SparseStepScratch const& scratch, int32_t selectionSlot)
    {
        SparseWorkList work = scratch.fetchWork;
        work.total += selectionSlot;
        return work;
    }

    //! This life cycle's arena of GPU cells, from which a member takes its lease when it
    //! joins. Empty when the life cycle has no sparse fetch path configured.
    //!
    //! Owned here rather than by StorageManager: the cells must stay GPU-resident while the
    //! life cycle's hot pool group becomes host memory, and a cell is not a page, so it has
    //! no answer to the question every storage slot must answer -- which block it holds.
    //! The arena slot this row's member holds. Serves every sparse life cycle at once: the
    //! arenas are sized alike, so a slot that exists in one exists in all.
    [[nodiscard]] SparseLeaseIndex leaseOf(BatchRow row) const noexcept
    {
        return mRowLease[row];
    }

    [[nodiscard]] SparseCellArena& _sparseArena(LifeCycleId lifeCycle)
    {
        return mSparseArenas[lifeCycle];
    }

    [[nodiscard]] SparseCellArena const& _sparseArena(LifeCycleId lifeCycle) const
    {
        return mSparseArenas[lifeCycle];
    }

    //! Outlives the Batch.
    KvCacheManager* const mManager;
    CUstream mCudaStream = nullptr;
    int mMaxBatchSize = 0;
    int mMaxBlocksPerSeq = 0;
    std::optional<int> mMaxTotalInputTokens;

    //! A seated member cannot be destroyed under the Batch, so a caller may keep only the
    //! Batch. KvCache points back with a raw `mBatch`, so there is no cycle.
    TypedVec<BatchRow, std::shared_ptr<KvCache>> mRows;

    //! The arena lease this row owns, for the Batch's lifetime. A permutation of
    //! [0, mMaxBatchSize): one lease per row, so a free row always has a free lease and
    //! nothing has to be allocated when a member joins. move() swaps two entries, which is
    //! what lets a member change rows without its cells being copied.
    TypedVec<BatchRow, SparseLeaseIndex> mRowLease;

    //! Indexed by life cycle; empty for one with no sparse fetch path.
    TypedVec<LifeCycleId, SparseCellArena> mSparseArenas;

    //! Cumulative fetch statistics, across every life cycle and step. Outside mStepScratch
    //! because the step's clear must not touch it.
    CudaUniqPtr<SparseHitMiss> mHitMiss;

    //! Device storage behind every life cycle's per-BufferId copy descriptors.
    CudaUniqPtr<SparseCopyDesc> mCopyDesc;

    //! Every life cycle's work-list counters. A step clears them once, in publish(); a
    //! producer then raises its own and the copy that consumes the list reads it.
    //!
    //! Not owning: they lead `mStepScratch`, which is what makes one clear reach all of
    //! them, so that allocation frees them.
    int32_t* mCounters = nullptr;
    size_t mCounterElements = 0;

    //! Per life cycle, fixed at construction. Only scratch-slot substitution can make one
    //! layer's indices differ from another's in the same life cycle, so a life cycle that
    //! cannot hold scratch takes SHARED and the single smaller table that implies.
    TypedVec<LifeCycleId, PageIndexMode> mIndexModes;

    //! Whether the caller named the mode rather than letting it be derived. publish()
    //! validates a named SHARED against each member, every step.
    bool mIndexModeNamed = false;

    //! One converted index table: the specs it serves, and what a consumer needs to read
    //! it. `concatOffsets` is in page-index units, parallel to a request's buffers.
    struct IndexTable
    {
        std::vector<IndexTableSpec> served;
        LifeCycleId lifeCycle{0};
        int scale = 1;
        std::vector<int> concatOffsets;
        //! Element offset into mIndexTableMemory.
        size_t elementOffset = 0;
    };

    std::vector<IndexTable> mIndexTables;

    //! Where one buffer's indices are.
    struct BufferSlot
    {
        IndexTableId table{0};
        int entry = 0;
    };

    std::map<BufferId, BufferSlot> mBufferSlots;

    //! The slot naming this buffer. Throws naming `api` when no index table spec has it.
    [[nodiscard]] BufferSlot const& _bufferSlot(LayerId layerId, DataRole const& role, char const* api) const;

    //! The spec that named this buffer. Exactly one does: specs may not overlap.
    [[nodiscard]] IndexTableSpec const& _specOf(LayerId layerId, DataRole const& role, char const* api) const;

    //! What a converted index of `spec` is added to: its first buffer's pool base, under
    //! the index mode this Batch chose. Every buffer of the spec shares it.
    [[nodiscard]] MemAddress _specBase(IndexTableSpec const& spec) const;

    //! Validate the requests, merge those whose converted indices coincide, and record
    //! where each buffer landed.
    void _buildIndexTables(std::vector<IndexTableSpec> const& specs);

    //! Refuse a sparse life cycle whose buffers no single launch can address.
    //!
    //! A sparse copy walks one (layer, pool) run of buffers per launch, from one base
    //! pointer, so that run has to sit at consecutive entries of one request. Splitting it
    //! across requests would need two bases in one launch; declaring only part of it would
    //! leave the rest with no converted indices at all. Checked here because a run that
    //! cannot be addressed produces wrong page addresses at step time rather than an error.
    void _checkSparseRunsAreAddressable() const;

    //! Sorts the tables, settles how many scratch slots a row may hold, and yields what
    //! that costs; resumes with the uploaded block's scratch tail and builds the tables
    //! and the device-side descriptors the conversion reads.
    LayoutGenerator _layoutIndexTables();

    //! Stage each member's scratch block range, for the life cycles that can hold scratch.
    void _stageScratch();

    //! Elements of the scratch staging.
    [[nodiscard]] size_t _scratchElements() const noexcept;

    //! Host and device sides of the scratch metadata, laid out [begs][ends][slots].
    //!
    //! Both point into the tail of the uploaded block rather than owning anything, which
    //! is what lets one transfer carry them along with the page tables. Addressable only
    //! when mMaxScratchSlots is positive; the tail is empty otherwise.
    //! Byte extent of each buffer, measured once at construction. A lookup where the value
    //! was arithmetic over members, which the compiler cannot fold.
    EnumMap<BufName, HalfOpenRange<ptrdiff_t>, kBufCount> mBuf{};

    int32_t* mScratchStaging = nullptr;
    int32_t* mScratchDevice = nullptr;
    //! Slot ids a request may hold for one life cycle, which bounds the staging.
    int mMaxScratchSlots = 0;
    //! Tables whose life cycle can hold scratch sort last, so one launch covers each kind.
    int mNumPlainTables = 0;

    //! Convert this step's raw slot ids into every table, on `stream`.
    void _convertIndices(CUstream stream);

    //! Reset a vacated row's converted entries, so nothing it held outlives its occupant.
    void _clearIndexRow(BatchRow row);

    //! All converted tables, carved by IndexTable::elementOffset.
    CudaUniqPtr<int32_t> mIndexTableMemory;
    //! One IndexTableDesc per table, and the offsets they point at.
    CudaUniqPtr<> mIndexTableDescs;
    int mMaxTableEntries = 0;

    //! Pinned so the members' index writes land here with no extra copy, and so the
    //! publish is a single asynchronous transfer. Holds the page tables, the row state and
    //! the views, then the scratch metadata.
    StaticHostMem mStaging;

    //! Bytes of that block, and so of each step's transfer.
    size_t mUploadBytes = 0;

    //! The row state and the leases, as a captured kernel reads them. Its address is baked
    //! into a captured graph, so it is written only on the caller's stream, after the replay
    //! that read the previous publish.
    //!
    //! The staged block's tail, not all of it: the page tables leading that block are read
    //! from mDeviceStaging by the conversion, and what a captured kernel reads of them is
    //! the converted table instead.
    CudaUniqPtr<> mMirror;

    //! A device-side copy of the whole staged block, filled from the host ahead of time.
    //!
    //! Nothing captured reads it, so its fill may overlap the previous step.
    CudaUniqPtr<> mDeviceStaging;

    //! Carries the host-to-device fill, off the caller's stream so it can run early.
    CachedCudaStream mUploadStream;

    //! One allocation holding every layer group's fetch scratch, carved per group.
    //! One device block split between the layout generators. The pointers they hand out
    //! point into it, so it outlives every one of them.
    CudaUniqPtr<int32_t> mDeviceArena;
    TypedVec<LifeCycleId, SparseStepScratch> mScratch;

    //! Recorded after the host-to-device fill, so the host can tell when the staged rows
    //! are free to write again. Null when no publish is in flight.
    CachedCudaEvent mPublished = CachedCudaEvent::makeNull();

    //! Recorded after the device staging has been read into the final tables. The next
    //! fill must wait on it, since that fill runs on another stream and would otherwise
    //! overwrite the staging while this step's copy is still reading it.
    CachedCudaEvent mDeviceStagingConsumed = CachedCudaEvent::makeNull();
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
