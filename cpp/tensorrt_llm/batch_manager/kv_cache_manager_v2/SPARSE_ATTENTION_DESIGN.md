# Sparse attention in KVCacheManagerV2

Target: the **decode instance of a disaggregated deployment**.

**Implemented**: the resolve, copy, index and flush-list kernels; `Batch`; the per-request
reservation and device state; the batch-wide write-back as `Batch::flushSparseUnits()`,
which `resize()` checks has run rather than performing itself; `fetch_sparse_pages` and
`prepare_sparse_write_indices`; sparse buffers across more than one pool; sparsity as a pool
attribute, so one layer may hold both sparse and dense buffers.

**Pending**: nothing. Compression is a separate feature, configured through
`KvCacheCompressionConfig` rather than through anything here, and needs nothing
sparse-specific -- see *Compression*.

Out of scope, by decision rather than by omission: prefill, which a separate instance serves,
and speculative decoding — see *Supported scope*; and per-KV-head selection, which no
per-`(request, layer)` resident set can express — see *Device-side cache*. Kernel fusion
remains future work; so do multi-beam requests. `ISparseUnitGather` is retired — see
*Extraction interface*.

Read `AGENTS.md` in this directory first for the storage, locking and concurrency model
this builds on.

## Problem

Sparse-attention decoding reads only a small, per-step, per-layer subset of a sequence's
history. Keeping all of that history in GPU memory wastes the resource the sparse
algorithm exists to conserve: a request's memory bill still grows with its full context,
so decoding hits a capacity wall long before it runs out of compute.

The goal is to hold history on the host tier and let GPU residency be bounded by the
selection width rather than by context length.

Two existing properties make this cheap:

- The host tier is registered `CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP`
  (`utils/hostMem.cpp:305`), so a kernel can read a host page directly.
  `isGpuAccessibleMemory` (`storageManager.cpp:138`) already relies on this for the
  cold-page codec, which encodes and decodes against host memory with no staging buffer.
- Selection output is a device array. If everything the fetch path needs is also on the
  device, the host never enters the loop, and the operation is stream-ordered and
  capturable in a CUDA graph.

## Vocabulary

| term | meaning |
| --- | --- |
| page | the KVCM2 allocation, locking and migration unit: `tokens_per_block` tokens |
| buffer | one layer's slice of a page; several coalesce into one page |
| selection unit | what the backend selects; a factor of `tokens_per_block`, never larger |
| in-flight | blocks above the watermark: written but not yet flushed |
| units per page | selection units a page holds, `tokens_per_block / sparse_selection_unit_tokens` |
| pure history | a block whose ordinal is below the history watermark set by `resize()` |

## Scope

Sparse-attention families differ in what they select, and the difference is load-bearing:

| family | selection unit | models |
| --- | --- | --- |
| MSA | block, 16 × 128 tokens (`minimax_m3/common.py:70`) | MiniMax-M3 |
| NSA, MoBA | block | DeepSeek-V4-Flash, Kimi |
| Quest | page | training-free, any model |
| DSA | **token**, k = 2048 | DeepSeek-V3.2, V4.1, GLM-5.x |

Block-level families work with the selection unit equal to one page and need nothing
special. DSA needs a sub-page unit, which this design supports through the selection-unit
mechanism below, at the cost of a finer cache entry. First implementation should target
the block-level families; DSA follows without an API change.

DeepSeek-V4's compressed pools are worth noting as a middle case: per-layer
`compress_ratios` with `compressed_block_sizes = tokens_per_block // ratio`
(`deepseek_v4/cache_manager.py:370`), so at ratio 128 a compressed page holds exactly one
entry and the selection unit is already a page.

## Shape of the design

1. **No new tier or medium.** What varies is the *memory kind of a pool group*: a sparse
   life cycle's level-0 pool group is backed by host memory, a dense one by GPU memory.
   Level 1 stays an ordinary codec target for both.
2. **`BufferConfig.is_sparse` is a per-buffer flag** and selects that backing. A sparse
   buffer requires a lifecycle of its own. Sparse buffers of the same size still coalesce
   with each other.
3. **Every page of a sparse buffer locks at level 0**, like every other page. Sparse pages
   are ordinary hot pages — coalesced, addressed by page index, reachable through
   `get_mem_pool_base_address`, and visible to the transceiver — that happen to live in
   host memory.
4. **All GPU residency is the per-request cache**, which is a dedicated arena rather than a
   storage tier. A sparse buffer has no separately managed GPU pages: both the block being
   written and the history a step selects occupy cache cells, drawn from the same lease and
   replaced by the same LRU.
5. **The host page is authoritative once the watermark passes the block.** A block reaches
   its host page through `Batch::flushSparseUnits()`, which the caller runs before the
   `resize()` that raises `history_length` past it; `resize()` checks that flush has
   happened rather than performing it. After it the cache entry is evictable. Before it the
   entry holds the only copy and is pinned.
6. **Extraction is pluggable**, because KVCM2 buffers are opaque bytes and a sub-page unit
   cannot be located without layout knowledge.

## Locking and flush

`acquirePageIndex` already writes the page's *current* slot id with no level assumption
(`page.cpp:401`):

```cpp
int old = kvc->updateBasePageIndex(
    mUser.beamIndex, mUser.ordinal, mUser.lifeCycle, slotIdToPageIndexValue(pg.slotId()));
```

so `getBasePageIndices()` reports hot slot ids with no change, and because every page of a
sparse buffer locks there, the table is **uniform**: every entry is a level-0 slot id, as
for a dense buffer. Nothing resolves against the cache arena, and no caller has to know
which rule applies to which ordinal.

A block's data reaches its host page by a **flush**: a copy on the Batch's stream that does
**not** evict the GPU entry, which stays cached and merely stops being pinned.

The governing rule is that **a block must be flushed before it can be selected**. Selectable
means below the watermark, so the watermark move is the deadline: everything it passes must
already be in its host page.

### Who flushes

`Batch::flushSparseUnits(newHistoryLengths)`, given the per-row history the caller is about
to set. The Batch records it; `resize()` then *checks* against that record rather than
flushing anything itself.

The flush used to live inside `resize()`, on the grounds that it was the single chokepoint
through which `setHistoryLength` and `setCapacity` both pass. That reasoning held, but it
bought the chokepoint at the price of hiding an ordering requirement: the write-back reads
converted page indices, which only a publish produces, so a flush buried in `resize()` was
silently depending on when the last publish happened. Naming the flush makes that a stated
precondition of a call the caller issues.

It is also what lets the write-back be **batch-wide**. Per request, it was one scan and one
copy launch per pool *per row*, plus two synchronous host-to-device copies to tell the
kernel which units moved. Driven from the Batch it is one scan over `(layers, rows)` and one
copy launch per pool for the whole step, with a single small upload of per-row ranges.

The check `resize()` performs is `newHistory <= recorded`, not equality: a step may resize a
row more than once, and a row whose growth failed rolls back to a history below the one that
was flushed for it. Flushing early is always safe — a flush is a copy, and a block flushed
before a failed resize stays correctly flushed. Only flushing late is not.

`commit()` needs no flush site of its own: the manager maintains
`num_committed_tokens <= history_length <= capacity`, so a committed block is already at or
below the watermark and was flushed before the watermark passed it.

`history_length` therefore keeps two jobs: it defines selection eligibility, and it is the
flush deadline.

### How much a step may cross

The write-back's work list is sized by `maxTotalInputTokens`, the tokens one step may ingest
across every row, which the runtime already bounds for its own batching. A row advancing `t`
tokens crosses at most `t / unitTokens + 1` units -- the `+1` for the partial unit its
watermark starts inside -- so one layer's list needs

```
maxTotalInputTokens / unitTokens + maxBatchSize
```

entries, and `flushSparseUnits` rejects a move whose units exceed that, summed over rows.

The bound is an **aggregate** one, and it is usable only because the list is compacted. Were
it indexed by row, each row's slice would have to allow for that row consuming the entire
token budget, and the product over rows would be hundreds of times larger. The same property
is what makes the guard right for in-flight batching: one prefilling row may cross the whole
step's worth of units while every other row crosses none, which a per-row limit would refuse
no matter how idle the rest of the batch was.

Left unset, the bound falls back to `maxBatchSize * maxInFlightUnits`. That is today's decode
behaviour -- a step advances capacity one token per row, so a row crosses at most one unit
boundary -- and it is what the sparse path currently supports; see "Supported scope".

### Ordering

`flushSparseUnits` stages its per-row ranges on a helper stream and launches the write-back
on the Batch's stream:

```
helper: waitEvent(flushRead)          // the previous step finished reading the ranges
        cuMemcpyBatchAsync(ranges -> device, CU_MEMCPY_SRC_ACCESS_ORDER_DURING_API_CALL)
        staged = record(helper)
main:   waitEvent(staged)
        scan the flush lists, then one write-back copy per pool
        flushRead = record(main)
```

The helper stream buys both things, dominated by the first. `DURING_API_CALL` consumes the
host array *before the call returns*, so issuing the copy on the Batch's stream would hold
the host until the previous step's graph replay reached it; on a helper stream the source is
captured at once and the call returns promptly. The device write then overlaps that replay
too, since the main stream reaches its wait on `staged` only after draining what it already
had queued — though with two ints per row that part is worth a microsecond, where the
host-side decoupling is worth whatever the replay has left.

The kernel stays on the Batch's stream because the cells it reads are written there. The
wait is a stream wait, not a host sync: the hazard is the previous step's kernel still
reading the same destination, which is ordinary stream ordering. (Do not set
`CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE` on that copy — it breaks `DURING_API_CALL`
semantics; driver bug 6718200, see `batchedPageCopy.cu`.)

The write-back reads the index table as of the **last publish**. That is current enough
because the move is clamped to the capacity the step started from, so the blocks it touches
were allocated by an earlier resize and published then.

### Unflushed entries carry no state

An entry holds either a fetched copy of a history page or the live, still-growing block.
The first is read-only by construction — fetch only ever targets ordinals below the
watermark — and the second is written by definition. So the two populations are
distinguished by the ordinal alone:

```
unflushed  <=>  resident_ordinal >= history_length / unit_tokens
```

Nothing records this. History is monotonic (`resize()` rejects a decrease), so an ordinal
that was history stays history and a resident ordinal at or above the watermark cannot be a
stale fetched copy. The flush recorded for a `resize()` covers exactly
`[old_history, new_history)`, so
the flushed prefix always equals the watermark and there is no second high-water mark to
keep. A per-entry dirty bitmap would be derived state that can only ever disagree with the
ordinal it was derived from.

Prefill and decode share one manager under in-flight batching, so the unflushed set is not
one entry: it is every in-flight block, which during prefill is the whole chunk. Two facts
keep this from being a general write-back cache:

- **Unflushed entries are private.** Only finalized pages enter the radix tree, so an
  unflushed block has exactly one owner: no aliasing, no cross-request coherence, no
  concurrent writers.
- **The unflushed set is at most one unit**, not discovered — see `max_input_length` below.
  Reservation covers it, so a fetch never meets a cache with nothing to evict.

Eviction eligibility is therefore one comparison against the watermark. There is no
evict-an-unflushed-entry path to implement: such an entry is simply not a candidate, and
the reservation guarantees enough candidates remain.

## Per-request reservation

Two separate reservations, easily confused.

**Hot pages.** Every block of the sequence has a locked hot page, so the count is
`divUp(capacity, tokens_per_block)` — the existing per-sequence locking logic, entirely
unchanged. Only the backing medium differs. This is the payoff of uniform locking: nothing
new to size, and no notion of "history pages" distinct from the rest.

**Cache cells.** `max_cached_units` is the whole cache, counting both populations:

```
cells = max_cached_units                          one buffer's cells
lease = num_buffers * divUp(cells, expansion)     contiguous cells a request takes
```

The two populations are disjoint — the selection is below the watermark, everything in
flight is above it — so the cache must hold a full selection *plus* every in-flight unit at
once. That is the only constraint, and it is what makes the second population derivable
rather than declared.

An in-flight unit's data lives only in its cache cell until the watermark passes it, so the
cache must hold every unit above the watermark. Within the supported scope that is **at most
one**: a step advances capacity one token past history, and one token straddles at most one
unit boundary.

*Sizing*: two to four times the selection width is the usual starting point. A cache only as
wide as the selection retains nothing between steps, so what it holds above the selection is
what survives to be hit next step — it is all reuse, not headroom.

### `max_input_length`

A bound on `capacity - history_length`, i.e. on how far capacity may outrun the watermark.
**One token**, and read-only.

It is a property of the supported shape rather than of how wide a cache is. A sparse request
contributes exactly one query token per step, because the fetch resolves one selection per
request while an attention kernel indexes its selection per query token; the two agree only
at one token each. This is the guard that keeps prefill and speculative decoding out — see
*Supported scope* below.

It is one token rather than anything derived from the reservation's width. What the cache
*could* pin says nothing about what a step produces: the fetch cannot serve a multi-token
step at any width.

The lease therefore depends on configuration alone. It is taken when the cache is seated in
a `Batch` and returned when it leaves, so a request holds cells exactly while it is being
stepped: one still receiving its disaggregated history, or sitting a step out, holds none.
Seating cannot fail for want of a cell, because the arena is sized from the same
`constraints` the Batch takes its rows from -- a free row therefore has a free lease, and
the fetch path never has to handle exhaustion. Because every request of a life cycle needs
exactly the same number of cells, the arena is a fixed-size-block free list rather than a
general allocator.

Leaving returns the lease with every entry free, so it discards whatever the GPU cache held.
`flushSparseUnits()` is what preserves anything worth keeping, and a Batch that churns its
membership every step cold-starts every request's cache.

`resize()` still checks the gap against this bound. It is the host-side counterpart of the
resolve kernel's in-flight overflow report: one catches a request outgrowing its cache
before the step, the other if it somehow reaches the device.

### Why the cache is a flat array of cells

A lease is one contiguous run of buffer-sized cells, and an entry resolves to one by
arithmetic alone:

```
cell    = buffer * divUp(max_cached_units, expansion) + entry / expansion
address = lease_base + cell * buffer_bytes
unit    = entry % expansion                       which unit within the cell
```

No page table, no slot indirection, no coalescing. Coalescing earns nothing inside the
cache because different layers select different units, so a slot's cells would never be
filled together; the cache took whole coalesced slots only while it drew them from the
storage allocator, which deals in slots. A dedicated arena has no such constraint, and
dropping it removes a page lookup from both the copy and the index kernel.

The consequence worth stating is that **a cache cell is not a page**. It holds one buffer's
unit, identified by `(buffer, entry index)` rather than by a block ordinal. This breaks an
assumption that runs through KVCM2 — that a page is both the unit of allocation *and* the
unit of identity — which is precisely why the cache is not a storage tier: it would have to
answer "which block does this page hold", and it has no answer.

## Device-side cache

Per `(request, layer)`, managed entirely by kernels: a resident set, an LRU queue and a
free list.

**Scope is per `(request, layer)`, never per KV head.** Every head of a layer shares one
resident set, because a selection unit is the tokens it spans across all of them. A layer
that names a `selection_id` widens the scope rather than narrowing it: the members of a
selection share one resident set between them, one per `(request, selection)` — see
*Selection groups*. Their cells stay per layer and mirror. An
algorithm that lets different heads select different units is not expressible here, and this
is a real exclusion rather than a hypothetical: Rocket's selection indices are
`[num_kv_heads, total_tokens]` and MiniMax-M3's are `[tokens, num_kv_heads, topk]` — both
choose per head. Serving them would need a resident set per `(request, layer, head)`, which
multiplies the metadata and the reservation by the head count.

**Scope is per request.** Two requests sharing a prefix do not share a cached copy. This
matches HiSparse and closes a soundness hole by construction: entries are reachable only
through their own request's page table, so a host slot can never be recycled underneath a
surviving entry. A global, provenance-keyed cache would give cross-request sharing for
free via the radix tree, but needs generation counters or an invalidation pass; deferred.

**Replacement is true LRU**, not "evict whatever was not selected this step". Entries that
hit are promoted above newly fetched misses, so a unit selected repeatedly outranks a
first-time selection and one-off selections leave first. Retention across steps is the
point: HiSparse reports a large miss-rate reduction from holding hot entries beyond the
current top-k, which is only possible when the cache is larger than the selection width.

**Residency is tracked by a reverse map** — a dense array indexed by slot `0..N-1` holding
the resident unit ordinal, N entries per `(request, layer)` — not a forward map from
ordinal to slot. The two scale with different things:

| | size | at 1 M context, 256 requests, 64 layers |
| --- | --- | --- |
| forward, ordinal → slot | batch × layers × **max_ordinals** | hundreds of MB, and unbounded in context |
| reverse, slot → ordinal | batch × layers × **N** | proportional to cache already bought |

The forward map is proportional to context length, the quantity this feature exists to
stop paying for. The dense reverse form is also what the LRU order and free list want.

### Probe index

Answering "is this ordinal resident, and where" against a dense reverse map would be O(N)
per selection. The fetch kernel therefore builds a hash in shared memory at the top of each
launch and discards it at the end. Two properties follow from it being ephemeral: there are
no deletions, so no tombstones and none of the maintenance that made a persistent device
hash map unattractive; and nothing has to stay consistent across steps or graph replays.

**The hash is over the required ordinals** -- the in-flight units followed by the selection
-- and the resident set is then streamed past it. So the table is sized by `k` rather than
by the cache's capacity `N`:

```cpp
// Index the required ordinals, then match the resident set against that index.
for (int32_t i = threadIdx.x; i < required; i += kThreads) {
    int32_t ord = i < inFlight ? historyUnits + i : selected[i - inFlight];
    if (ord < 0) continue;
    uint32_t pos = slotOf(ord, mask);
    int32_t key = atomicCAS(&keys[pos], kEmptyKey, ord);
    while (key != kEmptyKey && key != ord) { pos = (pos + 1U) & mask; key = atomicCAS(...); }
}
__syncthreads();

for (int32_t s = threadIdx.x; s < entries; s += kThreads) {   // the resident set
    int32_t ord = residentOrdinal[s];
    cls[s] = kClassUntouched | (ord >= historyUnits ? kPinned : 0);
    previousLru[s] = lruOrder[s];
    if (ord == kFreeOrdinal) continue;
    ... find ord in the table ...
    if (key == ord) vals[pos] = s;        // the resident entry claims its query
}
```

Open addressing with linear probing, capacity `next_pow2(2 * required)` so the load factor
stays at or below one half, which also guarantees an unsuccessful probe terminates on an
empty entry. Keys and values are separate arrays rather than packed 64-bit entries: a
`__syncthreads()` separates build from match, so no thread reads an entry mid-insert and the
value needs no atomicity, keeping the CAS 32-bit. Multiplicative hashing rather than
identity, because ordinals are small and dense, and dense runs make an unsuccessful probe
walk the whole run.

The resident sweep is also where per-entry recency and pin state are cached, so the pass
the match needs is the pass the eviction ranking needs; see "Capacity and eviction".

**This reverses an earlier decision, deliberately.** The index was first built over the
*resident* set, keyed by ordinal and valued by slot, sized `next_pow2(2N)`. That was argued
for on the grounds that hashing the selection would leave the miss list unknown until the
resident walk finished. The argument does not hold: the resident-keyed build had to walk
every slot and cross a `__syncthreads()` before its first probe, so the misses were never
known any earlier. What the choice really bought was one fewer pass -- the probe returned a
hit's slot directly, where the selection-keyed form needs a third pass of size `k` to read
the slots back out of the table.

What the reversal costs, and why it is still worth it:

- **Shared memory falls from O(N) to O(k).** At the DSA geometry the table drops from 8192
  buckets to 2048. This is the whole point: `N` now costs linear per-entry scratch rather
  than a `next_pow2`-rounded table, so the quantity bounding how large a cache one launch
  can serve is the selection width, not the capacity.
- **A third pass of size `k`** resolves hits from the table. Shared-memory only, and small
  against the resident sweep.
- **The index build no longer precedes the dependency wait.** The resident-keyed build read
  only the layer's own residency table, so it could run while the producing kernel was still
  draining; hashing the selection cannot, because the selection is what the producer
  produces. Only shared-memory initialisation now sits above the wait.

Measured on DSA at the geometry above, the resolve kernel fell from 1.468 to 0.833 ms per
decode step summed across 21 sparse layers.

A reservation-only call -- `prepare_sparse_write_indices`, which has no selection -- skips
the table entirely and runs a separate kernel that scans the recency order for its single
in-flight ordinal. See "Writing the in-flight unit".

## Extraction interface

KVCM2 buffers are raw bytes with no layout information, so locating a selection unit inside
one requires the owner's knowledge. Two mechanisms, chosen per buffer, serving two
different situations.

### Declaring where a unit sits

A unit is the tokens it spans across *every* KV head, so where its bytes sit follows from
two numbers on `BufferConfig`, both consumed once at configuration:

```cpp
std::optional<int> sparseSelectionUnitTokens;  // unset: a unit is a whole page
int sparseUnitChunks = 1;                      // pieces a unit is gathered from
```

`sparseUnitChunks` is the page's dimension order expressed as a count:

- **1** for a sequence-major page (`[token][head][dim]`), where a unit's tokens carry their
  heads with them and the whole unit is one contiguous range. Also 1 for any single-head
  page — a compressed or pooled latent — where the two orders are the same bytes.
- **numHeads** for a head-major page (`[head][token][dim]`), where each head contributes
  its own run and a unit is that many strided pieces.

Everything else follows, because a page is densely packed:

```
unitsPerPage = tokens_per_block / sparseSelectionUnitTokens
chunkStride  = size / sparseUnitChunks
chunkBytes   = size / (sparseUnitChunks * unitsPerPage)
unitStride   = chunkBytes
```

The four-field `SelectionUnitLayout` the copy kernel reads is derived from these at
configuration, so the kernel is unchanged and the config surface is two integers. The
derivation assumes dense packing, so a page with an alignment gap — between heads, or
between units — is not expressible; no model in this repository has one.

**Why this is not `tokens_per_block_override`.** That knob reinterprets a buffer's page for
*storage*, and stretches its public page index space by the same factor. Sparse selection
needs only the cache's own granularity, so it gets its own field and leaves the buffer's
pages, page indices and index space on the manager's `tokens_per_block`. Setting both on
one buffer is rejected. `units per page` and `PageIndexConverter`'s `expansion` are
distinct quantities: the sparse code says `unitsPerPage`, and `expansion` belongs to the
index converter alone.

**What `sparseUnitChunks > 1` is for.** Every sparse model in the repository today passes 1:
DeepSeek V4 asserts `num_kv_heads == 1`, and GLM's kpool, QSA's index-K and MiniMax's
index-K are all single-head and sequence-major. The field earns its place on QSA, whose
selection is shared across heads (its index-K has one head) while the buffer attention then
gathers from is head-major with ordinary GQA heads — the one combination that needs strided
units. Rocket and MiniMax-M3 under MSA are also head-major with several heads, but both
select *per head*, which this cache excludes for a different reason (see *Device-side
cache*); strided units would not rescue them.

### The gather interface, and why there isn't one

Earlier revisions planned an `ISparseUnitGather` — a per-buffer, stream-taking device call
that would extract a unit from a representation arithmetic could not address. It existed for
exactly one reason: the fetch read cold pages, so a codec that compressed or permuted them
put the unit somewhere a layout value could not describe.

Reading hot pages removes the reason. The hot representation is KVCM2's own, always in
ordinary slot layout, and a codec by construction only transforms data on its way *out* of
that tier. A compressed level 1 is therefore invisible to the fetch, and no plugin boundary
is needed on this path at all.

This also retires the LTO-IR/nvJitLink fusion work that existed only to inline such a
gatherer back into the resolve. The extension point is gone rather than deferred, which is
worth more than the code it would have taken to build.

### Kernel structure

The descriptor path crosses **no plugin boundary**. KVCM2 resolves selections, updates LRU,
allocates cells and performs the copies itself, because a layout is data it can interpret
rather than code it must call.

It is two launches — a resolve and a copy — but that split is internal and chosen, not
forced: the resolve writes a miss list the copy consumes, which is what lets the PCIe
copies be issued before the recency bookkeeping. Fusing them would be a local optimisation,
not an interface change.

There is no second path to fuse with, so the structure is final rather than a first stage.

## Compression

Compressing history is attractive — it is the bulk of the footprint, written once and read
rarely — and with sparse history on the hot tier it needs **nothing sparse-specific**. A
codec already transforms pages crossing the hot boundary; a sparse life cycle's pages now
cross it like any other, so level 1 can hold a compressed form and the existing
`IKvCacheColdPageCodec` does the work.

The fetch is unaffected: it reads hot pages, which are never the compressed form.

`SparseHistoryPassesThroughTheCodec` pins this: a sparse life cycle's page is encoded on
demotion and decoded on promotion, with the test first asserting that the life cycle's hot
pool group really is `HOST_MEM` so that it cannot pass by accidentally describing a dense
one. Nothing in the fetch path is involved, which is the point -- it reads hot pages, which
are never the compressed form.

**The two are decoupled deliberately.** Compression is configured by
`KvCacheCompressionConfig`, not by `SparseAttentionConfig`, on the stated grounds that
compression changes *which KV is stored* while sparsity changes *which KV is read*. A
storage-bound method provides a native cold-page codec that KVCM2 retains and invokes
(`_util.py`, `cold_page_codec_provider`); an iteration-driven one runs on the
resource-manager cycle instead. Neither knows anything about this package.

Two algorithms ship: `quantization_for_cold_page`, which stores host and disk pages as
NVFP4 while leaving the active GPU cache alone, and `triattention`, which evicts tokens
periodically during decode.

So "can sparse history be compressed" reduces to "does a sparse page reach the codec", and
it does. Worth stating precisely which pages, because two true sentences sound like one:
sparse history is *hot* -- host-backed, but hot -- and a cold-page codec compresses pages
on their way to a **cold** tier. Sparse history is therefore compressed once demoted, at
which point the codec applies with nothing sparse-specific, and not before.

What remains is the question the codec has always posed and this design does not answer:
whether a lossy codec is acceptable for KV history, and under what accuracy budget.

## Fetch

One call per **layer** per step, on the caller's stream — not per layer group. A layer's
selection is normally computed from the previous layer's output, so a step's selections are
not all known at its start; the call has to be the unit the caller can actually fill.

Within a layer, every sparse buffer shares the call. They select the same units, so they
share one metadata row, and one resolve yields the entry list that addresses each buffer's
own cell. Their agreement is structural rather than a consequence of running identical
logic on identical input.

### Selection groups

A model that runs one indexer across a run of layers hands them all the *same* selection
every step. Resolving it per layer computes the same answer as many times as there are
layers, each against its own resident set and recency order — and they cannot diverge,
because `initialiseLease` clears every row alike and identical input then evolves
identically. So the copies are of one answer, paid for in kernel launches and in
`entriesPerSelection` of metadata per layer.

Those layers instead name one `selection_id`. Members of a selection share **one** resident
set, one recency order, one in-flight reservation and one miss list. The cells stay per
layer and are *mirrored*, exactly as a layer's K and k-scale mirror today: the entry index
is shared, the bytes behind it are not. `entriesPerSelection` is the metadata's first
dimension, and `layerSelection[layer]` is where a layer reads it — the identity where every
layer selects for itself.

The id is opaque, not a `BufferId`. KVCM2 never uses *which* buffer produced a selection, so
naming one would state a claim it does not rely on, and that claim goes stale for a model
with several indexer buffers per layer or one that computes a selection without caching
anything. The validations worth having are expressible over an opaque id: members must
agree on `num_selected_blocks`, `max_cached_units` and `unit_tokens`, must share a life
cycle (the metadata they share is the life cycle's), and an explicit id may not name a layer
that set none of its own — silence is never shared.

The call then splits in two. `resolveSparseSelection()` runs **once** per selection, and
`fetchResolvedPages()` runs **per layer**, each copying the same misses into its own cells
and writing its own index list. `fetchSparsePages()` remains as the fused form and serves a
layer whose selection is its own; it rejects a shared one rather than mis-serving it, since
a second resolve would count the first's admissions as hits, issue no copies, and leave the
second layer's cells unfilled.

The copy kernel needed no change for any of this. It consumes a work list plus a buffer
range, and nothing in it depended on which layer produced the list. Neither did the
write-back: `sparseResolveKernel` is the only writer of `residentOrdinal`/`lruOrder`, and
`sparseFlushListKernel` only reads them — so collapsing the resolve is sufficient, and a
per-layer flush against shared metadata stays correct.

Layers that are *not* tied keep a call each, which is what starts the first layer's copies
as early as possible instead of waiting for the whole run.

The tied layers stay separate layers. The alternative — declaring their buffers as one
`AttentionLayerConfig` with distinct roles — also yields one metadata row and one call, but
it merges the copies too, so nothing can be fetched until every member's bytes are ready.
It also misdescribes the model: GLM-5.2's four indexer-sharing layers are four layers, and
only the full-indexer one declares an `INDEX_KEY` buffer, the other three being wholly
sparse. It is the **life cycle** that needs a dense buffer, not each layer. A selection
group keeps the layers apart for the copy and shares only what is actually shared.

So a call covers one layer's buffers and the copy needs no way to name a subset of them.
A caller wanting a layer's copies split further would be asking for the inverse of the
configuration choice above, having already declared those buffers as one layer.

Each call then:

1. read the selection: unit ordinals, uniform width, `-1` padded;
2. build the probe index over the required ordinals — in-flight units, then the selection;
3. stream the resident set past it: a match is a hit with its slot, the rest are misses;
4. allocate a slot per miss from the free list, evicting LRU when empty;
5. extract and copy each miss from the device-mapped host page;
6. promote hits above new misses in recency order;
7. write the output index list: one index per selection slot, in the selection's own
   order, `kBadPageIndex` where a slot is padded.

The output is compact rather than a table indexed by unit ordinal, because that is the
shape every sparse attention consumer in this repository takes. A token-sparse kernel is
handed the index list in place of its page table — `attentionOp.cpp` assigns
`kvPageIdxPtr = sparse_attn_indices`, and the trtllm-gen K/V TMA descriptors are rebuilt
as a flat token array so one "page" is one token — while a block-sparse kernel takes the
list beside a page table it still uses. None indexes a table by ordinal, so producing one
would only oblige the caller to compact it again.

One index per buffer, though. A layer's buffers share the resolve but not their cells, so
the list carries a buffer dimension: `[row][buffer][selection]`, which is also the single
tensor covering K and V that attention kernels tend to want.

Steps 5 and 6 are independent. Because the probe in step 3 resolves misses directly, the
copies — the only step that touches PCIe, and so the long pole — can be issued before the
recency bookkeeping runs rather than after it.

No host involvement, so the call enqueues and returns. Every shape is static across steps.

The selection is not the only thing that needs a cell. The block being written needs one
too, and needs it before the selection takes any victim. Both are resolved by the same
probe, over one virtual list: the in-flight ordinals `[history_units, capacity_units)`
first, then the selection. An in-flight unit differs in exactly one way — a miss allocates
its cell but never joins the copy list, because its host page holds nothing to fetch until
the watermark passes it.

### Writing the in-flight unit

The unit between the two watermarks is the one the model is about to produce. `fetch_sparse_pages`
makes it resident and pins it, but never reported *where*, so nothing could write it.
`prepare_sparse_write_indices` closes that: it runs the resolve with no selection and converts
the in-flight slot into the same index space the fetch produces.

It is a separate call because the orders differ. A fetch needs the selection, which the
indexer computes from the previous layer's output; a write needs only the watermarks, so it
can run first — which is the order the model requires. Idempotent while the watermarks hold:
the resolve probes residency before taking a victim, so the fetch's own resolve finds the
entry and reuses it.

Nothing is fetched for it. An in-flight unit's host page is not written until the watermark
passes it, so the resolve excludes in-flight ordinals from the copy list (`ordinal <
historyUnits`) — copying would read stale bytes over what the model just wrote.

`out` is `[batch][buffers]`, one entry per request: `resize` bounds `capacity - history_length`
to a token, so at most one unit is in flight. `BAD_PAGE_INDEX` is an ordinary outcome rather
than an error — with 4-token units a unit completes one step in four, which is also the only
step on which a ratio-4 compressor has produced anything to write.

### Supported scope: one query token per request

The fetch is a decode-phase call. It resolves **one selection per request**, while a sparse
attention kernel indexes its selection **per query token** — trtllm-gen's
`ptrSparseMlaTopKLens` is `[numTokensQ]`, "where each tokenQ has a corresponding topK
length". The two agree only when a request contributes one query token per step, so that is
the supported shape, enforced by `max_input_length` and by `fetch_sparse_pages` requiring one
selection row per batch row.

Excluded, deliberately:

- **Prefill.** Served by a separate prefill instance under disaggregated serving, which needs
  no sparsity here: it computes the prompt and hands over the KV cache.
- **Speculative decoding and MTP.** Generation, but several query tokens per request per
  step.

Both are excluded for the same reason, and it is not the row count. The resolve makes units
*resident* in a request's cache, which holds `max_cached_units`; residency is a property of
the request, not of a query token. N query tokens each choosing their own units have a union
of up to N times the selection width, so the reservation could no longer cover a call, and a
unit evicted to make room for a later query token would invalidate an index already returned
for an earlier one. Taking one row per token would make that reachable, not fix it.

**What else is sized by this, and would need revisiting.** The restriction reaches further
than the fetch. `maxInputLength()` returns one token, which makes
`_sparseCacheMaxInFlightUnits` one, which in turn sizes the write-back's per-row flush list
and bounds how far one watermark move may travel. Those adapt on their own if the constant
rises -- `flushSparseUnits()` refuses a span wider than the headroom, and the scan asserts
the list cannot overrun. What does *not* adapt is throughput: the write-back is tuned on the
assumption that a row has at most one unit to move, so it is given a single warp per
request. Correct for any count, but once a flush list is long that warp is the long pole and
wants the same treatment as the fetch.

This is a difference in kind from the sparse implementations in this repository, not a gap in
degree. Their sparse prefill is real — the context path reuses the generation kernel with
per-token indices — but their indices point straight at a GPU-resident KV pool, and the
in-chunk tokens are reached by writing K/V to the cache before attention and forcing a local
window in as ordinary indices. There is nothing to stage, so no per-request cache is needed.
Here history is on host, which is exactly what makes prefill hard.

### Arriving with a history: the bulk watermark jump

A resumed request is the ordinary case under disaggregated serving, and it moves history from
zero to the whole prompt in one `resize()`. Nothing is written back, because nothing was ever
in flight: those units reached their host pages directly rather than through a cell.

The write-back is therefore bounded by the capacity the resize started from, not by how far
history travelled:

```
flush range = [old_history, min(new_history, old_capacity))    in units
```

Only a unit below the old capacity can have been in the cache at all. Without that clamp the
range is the whole prompt, which no reservation could cover, and the resize fails on a
write-back that had nothing to do.

That jump has to happen **before the request joins a Batch**. `Batch::resize` refuses a
history beyond the capacity its row already had, because capacity granted by a call makes
room for tokens written after it — so history can only cover tokens whose space was already
there. A resumed request sets capacity and history together from nothing, which is the one
shape that rule excludes. Seeding it on the way in costs nothing: nothing was in flight, so
there is nothing to write back and no converted table is needed yet.

### Capacity and eviction

Capacity is guaranteed by reservation, not back-pressure: a fetch can always satisfy its
own layer, because the reservation covers a full selection width. Within a step a fetch may
evict entries of layers already processed but never one a later kernel still needs, which
LRU recency delivers — layer N's entries are most-recently-used when layer N+1 fetches.

An entry is evictable only when its unit ordinal is **below the watermark**. That is what
protects the block being written, and it is a correctness predicate rather than a policy
hint — but it is derived, not stored: see *Unflushed entries carry no state*.

### Preconditions the caller owns

Selections must name only pure-history units; `num_blocks` carries each request's eligible
count.

A step's order is the caller's too, and it is now explicit rather than implied:

1. `flushSparseUnits(newHistory)` — writes back what the watermark is about to pass. It
   reads the index table as of the last publish, which covers the blocks in range.
2. `resize(...)` per row, or `Batch::resize` — refused if it would take history past what
   was flushed.
3. `publish()` — converts this step's indices for the fetch to read.
4. the fetch, inside the captured graph.

Steps 1 and 3 both read or write Batch-owned device state, so neither may be skipped on a
step whose watermark moves. A history advance with no flush before it is refused; a flush
with no publish behind it would write through indices for blocks that did not exist yet,
which is why a history past the row's capacity is refused at step 1.

`publish()` uploads the staged rows asynchronously, so anything that writes what it staged
has to be separated from it by `waitPublished()` — a direct `resize()` on a member, a
`commit()`, a beam fork, an SWA recycle, anything that locks, unlocks or recycles a page.
The Batch's own mutators do that wait themselves: `Batch::resize`, `move`, `addKvCache` and
`removeKvCache` each call it before touching a row. `publish()` refuses rather than waits
when the previous one is still outstanding, because a wait there would cover only its own
staging and would quietly hide a member mutated before the call, which had already raced.

Naming an ineligible one is no longer a memory-safety problem. The table is uniform, so the
ordinal resolves to a real host page of the same request — but an unflushed block's host
page has no valid contents yet, so the result is stale data rather than unrelated memory.
A debug check against the flushed state can catch it; silence would be a wrong answer, not
corruption.

### Measuring the hit rate

A hit is a selected unit the cache already held; a miss is one the fetch copied from its
host page. The ratio is what a deployment tunes the cache size against, so it is collected
rather than inferred.

**Both figures come from the resolve, because only the resolve can produce either.** The
obvious denominator — rows times selection width, which the host knows for free — counts
things that are neither. A selection entry may be padded, and padding is negative and lives
on the device, chosen by whatever selected the units. An in-flight unit is resolved in the
same pass but is not a fetch: its host page is not written until the watermark passes it,
so there is nothing to copy and nothing to miss. Counting either as a hit inflates the rate,
and does so worst where histories are short — early decode, short prompts — which is
exactly the regime someone reads the rate to understand.

So the resolve counts the selection entries that named real history, in the pass that
already classifies them. The miss half is `claimed`, the run the row takes in the compacted
list, which it computes anyway. The two are produced together and agree by construction on
what they count; hits are their difference and are never stored.

**Each row adds its own contribution, rather than anything reading a total afterwards.**
This is the part that is easy to get wrong. The step's counters — `work.total` and anything
beside it — are cleared once per `publish()`, not once per fetch, which is what lets one
clear cover every layer without a non-kernel node between each resolve and the work before
it. A consumer downstream that read those counters and folded them into a cumulative pair
would therefore be correct only if every layer were fetched exactly once per publish. That
is what a step does, but the interface permits more, and a second fetch of a layer would
silently re-add the first one's totals. Accumulating per CTA from the values the row holds
in registers has no such precondition, costs two atomics per row, and needs no ordering
against anything.

It also keeps the whole mechanism inside one kernel: nothing in the copy, no programmatic
dependency to reason about, and no need to exclude the write-back — which shares the copy
kernel and whose moved units are due to be written rather than units the cache failed to
hold. The in-flight resolve contributes nothing on its own, since it has no selection and
its ordinals sit above the watermark.

64-bit counters: a step at B = 256 with a 2048-wide selection resolves millions of units per
layer, so a 32-bit pair would wrap within a few hundred steps — and a wrapped miss count
makes the hit rate *improve*, which is the failure a reader would trust.

**There is no enable switch.** The cost is two atomics per row per fetched layer. A switch
would cost nothing to run either, but it would bake into a captured graph, so turning
statistics on mid-run would keep reporting zeros until the graph was captured again — a
disabled feature indistinguishable from a broken one.

`peekHitMissCounter()` reads the pair on a temporary stream and synchronises only that, so
it never stalls the step. The pair may be a few launches stale, which does not move a ratio;
a caller wanting the two to agree exactly synchronises the device first. Because the read
races a running fetch and the counters are separate atomics, a copy can land between them —
the result is clamped rather than checked, since a diagnostic must not be able to fail the
cache. Per-row detail is already available separately in `missCount`.

## One arena, one index space

The cache arena holds nothing but cells — selected history and the unit being written alike.
That is what lets the fetch emit a **single** index list covering both, and therefore why
existing attention kernels need no change: a kernel still indexes one list against one base.

Indices are unit-granular, and over the buffer's **arena region** — one per hot pool, see
*Sparse buffers across more than one pool*:

```
index = cell * expansion + unit
      = (lease * cells_per_lease + slot_in_pool * cells_per_buffer + entry / expansion)
        * expansion + entry % expansion
```

where `slot_in_pool` is the buffer's index among those of the life cycle sharing its pool,
and `cells_per_lease` is that pool's buffers times `cells_per_buffer`.

A sparse buffer therefore has **two** base addresses, answering different questions:

| accessor | names | consumer |
| --- | --- | --- |
| `get_mem_pool_base_address` | the buffer's pages, hot tier | transceiver, KV transfer, anything addressing blocks |
| `get_sparse_cache_base_address` | the cache arena region of the buffer's pool | the attention kernel, against the fetch's index list |

`get_mem_pool_base_address` keeps meaning "this buffer's pages" for every buffer, sparse or
dense, which is what lets the transceiver work unchanged on hot, coalesced pages. The arena
is a second allocation holding the only GPU-resident copy a kernel can read, so it gets a
second accessor rather than overloading the first. The consumer's side is otherwise
unchanged: one base, one index list, the expanded index space `PageIndexConverter` already
produces.

`tokens_per_block_override` plays no part in this. It is the index converter's own
mechanism: it scales `BufferConfig.size` and stretches the buffer's public page index space
by `expansion`, while allocation, locking and migration stay on the global
`tokens_per_block`. It is rejected on a sparse buffer, which says what it means with
`sparse_selection_unit_tokens` and leaves its index space alone. `unitsPerPage` and
`expansion` are distinct quantities and the sparse code never uses the latter.

## Batch

Every device array here is indexed by batch row, and every call must agree on which row a
request occupies. `Batch` owns that: membership, row order, the page table, `num_blocks`,
and the staging mirror. It is global, not per layer group, and carries one page table per
layer group.

It also owns three things the sparse path depends on:

- **the stream.** Taken at construction; joining binds the member to it, and moving a member
  off it is refused. A member's own work and the Batch's transfers have to be ordered
  against each other, and a stream per call left two sources of truth for which one that is.
  It is also what makes the write-back's ordering free rather than checked, since a member
  records its events on the stream the flush is launched on.
- **the converted index tables** (`INDEX_CONVERSION_DESIGN.md`), which both the fetch and
  the write-back read.
- **the write-back**, as `flushSparseUnits()` — see "Locking and flush".

**The upload is whole-table by decision.** `publish()` stages every row and sends the table
in one `cuMemcpyHtoDAsync` on the upload stream, which overlaps compute. Sending only the
rows whose locks moved would replace that single driver call with one per row, and the cost
of those launches outweighs the bytes they save. The table is large -- four layer groups,
about 32 MiB a step at a 1M `max_seq_len` and B = 256 -- but size is not the figure that
decides it.

**Dirty rows — planned, not built.** What dirty tracking would still save is the host-side
restaging, not the transfer: a `_KVCache` knows its `Batch` and row (it does, for detach),
any lock-changing operation could mark that row dirty, and `Batch` could restage only those,
following the existing `mark_stats_dirty` / `clear_stats_dirty` /
`get_dirty_stats_kv_cache_ids` pattern. Until then a step restages the whole table, which is
a cost rather than a wrong answer.

Rejecting direct mutation instead was considered and rejected: `commit` changes locks and
runs every decode step, so strictness would require a batched twin for every mutator before
anything could ship. Dirty tracking degrades into a slower step rather than a wrong answer.
A debug-mode check can still flag direct mutation while batched.

**Batched resize** resizes every member and publishes once at the end. It may initially
forward to `_KVCache.resize()`; the committed contract is row order and
publish-on-completion, which survives the later coalescing work. The result is per-request,
not a single flag, since `_KVCache.resize()` already rolls back per cache; a table published
after a partial failure must reflect the post-rollback state.

**Rows are slots.** Removing a request leaves a hole to be reused rather than shifting later
rows, which would invalidate the whole table and destabilise the shapes graph capture needs.
The `_torch` layer already works this way with `index_mapper_capacity` and a `copy_idx`
gather. Membership is exclusive: a row already occupied is refused rather than rebound,
because nothing downstream would notice the substitution.

**A row owns its member.** `mRows` holds a `shared_ptr`, so a seated cache cannot be
destroyed under the Batch and a caller may keep only the Batch; `removeKvCache()` hands the
reference back, which makes "the Batch kept it alive" and "here it is back" one operation.
The binding pinned every cache ever added for the Batch's whole life and released none of
them, which was the wrong shape as well as a leak: what a row needs is a reference for
exactly as long as it is seated. `KvCache` points back with a raw `mBatch`, so there is no
cycle, and `_detach()` -- the path for a member destroyed while seated -- is now unreachable
by construction and says so with a fatal check rather than carrying teardown that could
never run.

## Sizing

Per sparse buffer: **`num_selected_blocks`** (the selection width `k`) and **N** (the
reservation upper bound, in the same units).

N should be `2k`–`4k`, not `k`. `k` is the correctness floor — it guarantees a fetch can
satisfy its layer — but it leaves no room for retention, and retention across steps is
where the hit rate comes from. HiSparse finds `B ∈ [2k, 4k]` the useful range.

There is no separate budget for in-flight blocks. The sparse pool's GPU tier holds nothing
but cache cells, and an in-flight block occupies one of the same cells the selection draws
on, so `max_cached_units` covers both:

```
cache reservation = max_batch_size * N             (N in [2k, 4k])
in-flight headroom = 1 unit                        one query token per step
```

with `max_batch_size` from `constraints`. The surplus above `k` is retention, not headroom,
so `N` is sized by one demand rather than two: retention wants
`2k`–`4k`, and a prefill chunk wants enough cells to pin the whole chunk at once.

## Python surface

Declared or planned in `tensorrt_llm/runtime/kv_cache_manager_v2/__init__.pyi`:

- `BufferConfig.is_sparse: bool`, `BufferConfig.num_selected_blocks: int`, the reservation
  bound, `BufferConfig.sparse_selection_unit_tokens: int | None` and
  `BufferConfig.sparse_unit_chunks: int` — construction-time, per buffer.
- `KVCacheManager.is_sparse(layer_id, data_role) -> bool`.
- `Batch` — membership, row order, device arrays, `resize()`, dirty-row publication.
- `KVCacheManager.fetch_sparse_pages(batch, buffers, selected, out, stream)`.
- `Batch.sparse_out_slots(buffers) -> list[int]` — the slice of `out` each buffer's
  indices land in, for both the fetch and the write.

Publishing the page table is `Batch`'s job. It must not reuse the dense path's block
offsets: those are scaled and fold `BAD_PAGE_INDEX` to slot zero, whereas a sparse table
numbers history and input-token entries in different levels' pools and needs the sentinel
to distinguish an absent block from slot zero.

Device arrays are typed by a `DeviceArray` protocol over DLPack, so callers pass torch
tensors without this package depending on torch.

## Implementation notes

Slot release is already level-aware: `Page::~Page()` releases with the page's own
`cacheLevel`, and the `kHotLevel` in `ScratchSlotLock::unlock()` is correct, since SWA
prefill scratch is GPU-only by definition.

- **`_lockHeldBlocks` carries no level hazard.** The OOM rollback for `resize` re-locks at
  the hot level, which is where every buffer locks,
  so a page table's entries and the level its pages lock at cannot disagree. Nothing carries
  a per-target lock level: there is no `queryLockLevel`, no `BatchedLockTarget::cacheLevel`,
  no `StaleBackup::cacheLevel` and no `residencyLevel`.
- **`_unlockStaleBlocks` is not the flush hook.** It exists for SWA blocks outside the
  window, whose content is *dead*; sparse history is live and will be read. Same trigger and
  range arithmetic, different disposition — routing sparse layers through it drops the data
  instead of flushing it. It also gates on `windowSize.has_value()`, so for a full-attention
  sparse layer it fires on nothing.
- **`_shortcutSetHistoryLength` skips the flush for exactly our configuration.** It reports
  "unchanged" when no SWA stale range moved, and for a full-attention layer
  `windowSize.has_value()` is false, so nothing is ever reported as moved: it assigns
  `mHistoryLength` and returns true, and `resize()` returns early before reaching its body.
  A flush placed in that body never runs. The shortcut must either carry the flush or be
  declined while a sparse flush is pending.
- **Hook `resize()`, not `setHistoryLength()`.** The latter is `resize(nullopt, hist)` and
  `setCapacity()` is `resize(cap, nullopt)`, so `resize()` is the only place the watermark
  moves — and a caller may move capacity and history in the same call.
- **`commit()` is not a flush site at all.** The manager maintains
  `num_committed_tokens <= history_length <= capacity`, so a committed block is already at
  or below the watermark and was flushed by the `resize()` that passed it. Where `commit()`
  advances the watermark it does so by calling `resize()`.

Also: no attention kernel addresses a host page. A sparse layer's kernel reads its units
from the arena, which is GPU memory bounded by the lease count and the cells a lease holds,
and a non-sparse layer reads the device pool through its block table as it always has. The
host pool is reached only by this manager's own copies, which compute their offsets in
`size_t`.

## Planned: a selection unit size per layer

A layer declares its fetch path with `sparse_cache`, and layers sharing a life cycle must
currently declare the same one. The arena is per life cycle and its metadata is a single
stride — a selection's rows live at `residentOrdinal + slot * entriesPerSelection` — so one
entry count serves every layer, and `cellsPerBuffer` follows from `max_cached_units` and the
unit size together.

That is an implementation limit, not a property of the design. Lifting it:

- **Size a cell from the largest unit of the life cycle**, or simply from a whole page.
- **A layer with a smaller unit subdivides that cell.** This is the mechanism a cell already
  uses: it holds `unitsPerPage` units, addressed as `cell * unitsPerPage + unit`. A layer
  whose unit is half the cell's just has twice the divisor.
- **Require unit sizes to be powers of two**, so every such division is exact and a buffer's
  bytes split without remainder at any level.
- The resident set and the recency order are **already per selection**, so they need no
  change; only the entry count and the divisor become per layer rather than per life cycle.

Until then a mismatch is rejected at construction. It was previously taken as a `max` across
the life cycle, which quietly served one layer's shape to another.

## The output list merges across mirrored pools

The arena mirrors the hot pool group, and a lease takes the same cell index in every pool
of it. So **mirrored pools** -- equal `buffersInPool`, hence equal `cellsPerLease`, with
`cellsPerBuffer` one value for the whole shape -- resolve a selection to the same index.
`out` carries one slice per distinct index table entry rather than one per buffer, with
the caller applying each pool's own base.

For a life cycle whose pool group holds a dense indexer and two sparse pools -- `{K,V}`
and `{kscale,vscale}` -- `sparseIndexTableSpecs()` emits one spec per `(pool, layerRow)`
and `Batch::_buildIndexTables` folds the two into **one** table, because their `scale` and
`concatOffsets` agree. The layer's four buffers therefore take the slices `[0, 1, 0, 1]`:
`K` with its scale, `V` with its scale. A scale shared by `K` and `V` together is a pool
of one buffer, whose `cellsPerLease` differs, so it keeps a slice of its own: `[0, 1, 2]`.

`Batch::sparseOutSlots` is the query, and `fetchSparsePages`,
`fetchResolvedPages` and `prepareSparseWriteIndices` all follow it. They have to agree, or a caller would need one
rule for reading a fetch and another for a write. A run whose slices an earlier one
already wrote is not launched at all -- the saved work and the saved writes are the point
-- and the entries of `out` no buffer maps to are left untouched rather than packed out,
so `out`'s shape stays `len(buffers)` however the buffers happen to merge.

The grouping follows the caller's `IndexTableSpec`, so two buffers merge only where one
table was asked for over both. The merge is not read off the dense tables alone, though:
those merge on the hot pool's `scale` and `concatOffsets`, which is not the same quantity
as the arena's geometry -- a pool carrying only some of the group's layers has the same
scale and a different `cellsPerLease`. `sparseOutSlots` checks the arena geometry agrees
before sharing a slice, and raises rather than aliasing two different cells onto one.

Rests on a property only a second lease can observe:
`TestSparseFetch::test_mirrored_pools_resolve_a_selection_to_one_index` uses one, because
at lease zero the `leaseIndex * cellsPerLease` term vanishes and pools of unequal width
are indistinguishable.

The rest of the arrangement is what makes one base per slice enough:

- **A base per pool, in both spaces.** `getSparseCacheBaseAddress` returns
  `arena.base(poolIndex)` -- "per pool, not per buffer" -- so `{K,V}` share one arena base
  and `{kscale,vscale}` another, exactly as the two pools share one hot-pool base each.
- **The within-pool offset rides in the index**, as `slotInPool * cellsPerBuffer`. That is
  the concat-offset arrangement, so one base does serve a pool's buffers.
- **The output order is the plan's**, because `_sparseLayerSpan` rejects any other:
  "buffers must be exactly this layer's sparse buffers, in the cache's order".

## Sparse buffers across more than one pool

A life cycle's sparse buffers need not agree on size. The **quantised shape** — block scales
as a separate, smaller `DataRole` — works, which matters because it is a real configuration
rather than an exotic one.

Pools partition a life cycle's buffers by size, and the copy addresses one pool with one set
of strides: `bufferBytes`, the hot slot stride and the unit layout are all per pool. The
design follows from making the arena **mirror the hot pool group** rather than inventing a
layout of its own:

- **One region per pool**, sized `maxLeases * buffersInPool * cellsPerBuffer * cellBytes`.
  A lease is an index rather than a cell offset, because a pool's cells-per-lease is its own.
  A pool index no sparse buffer of the life cycle uses keeps its place in the list and sizes
  to nothing: the arena is indexed by the storage pool index rather than by a dense numbering
  of its own, and an empty region costs only the entry that says where the next one starts.
- **Buffers ordered `(layer, pool)`.** A layer's buffers stay consecutive, which the fetch
  needs to name them as a range; and within a layer a pool's buffers are consecutive too,
  which the copy needs for the same reason. `bufferLayer` stays non-decreasing.
- **One launch per maximal run of consecutive buffers sharing a pool.** When the buffers are
  alike that is a single run spanning the whole life cycle, so the uniform case pays nothing
  for the general one; the quantised shape yields one run per layer and pool. The fetch
  clips each run to the layer it resolved; the write-back takes them whole, because its work
  list is per layer and a run already spans as many layers as share its pool.
- **`get_sparse_cache_base_address` is role-dependent**, naming the region of the buffer's
  pool. It already took a role.

Every per-pool quantity is a launch scalar — `deviceBase`, `bufferBytes`, `cellsPerLease`,
`layout` — so the kernels keep their shape. What they gained is `bufferSlotInPool`, a
per-buffer map alongside `bufferLayer`, replacing the assumption that a buffer's position
among *all* the life cycle's buffers is also its position in its pool. A per-buffer offset
within the cold page sat here too, until the converted index absorbed it; see
`INDEX_CONVERSION_DESIGN.md`.

Buffers sharing a pool must still share a selection unit layout. A pool fixes a cell's size
but not how its units are strided within it, and the copy walks a run with one set of
strides; the check is per run rather than per life cycle.

**The runs share one launch.** Nothing orders one pool's copy against another's — their
cells are disjoint and the converted page table they share is read-only — while each is
small and latency-bound, so issuing them back to back on one stream would expose each one's
latency in turn. So one grid runs them all, and what differs between them lives in a
**device-resident descriptor**.

`gridDim.y` walks the buffers of every pool as **one flat range**, with a constant map from
buffer to pool. The alternative — pools on one grid axis, buffers-per-pool on another —
has to size the buffer axis for the widest pool, so every buffer a narrower pool lacks
costs an idle block. Pools of one life cycle are often alike, and then the two are the same
shape; they stop being alike as soon as a pool holds more buffers per layer than another,
which a key-and-value pool beside a scale pool already does.

The descriptors are built once, with the Batch. Every field comes from the buffer plan and
the index tables, both fixed at construction, so nothing is written per step and a captured
graph replays a copy with no host involvement at all. The fetch needs one row per layer — a
layer has its own index-table entry and its own slice of each pool's buffers — while the
write-back spans every layer at once and needs only the one.

The vector width is part of the descriptor, chosen from the layout's **sizes alone**. The
two base pointers are checked against it where the descriptor is built, because a pool whose
base is misaligned is a configuration error worth naming rather than a case to carry into
the kernel. A block switches on the width once and calls a device function templated on it;
`blockIdx.z` is block-uniform, so the switch costs no divergence.

One grid rather than several is also what lets **every** pool carry the launch dependency
below. A per-pool fan-out across streams could not: a programmatic dependency follows a
stream, not an event.

### Launching before the producer finishes

All three kernels are enrolled in programmatic dependent launch when the environment enables
it, and in each the order of the prologue is the point: a block does everything that is not
its predecessor's output *above* `cudaGridDependencySynchronize()`, so that by the time the
predecessor drains the block is resident, addressed, and as far along as it can be.

How far along differs by kernel. The resolve gets least out of it: its probe index is keyed
by the required ordinals, and the selection is exactly what the producer produces, so only
shared-memory initialisation sits above the wait. That was not always so -- an index built
over the resident set read only the layer's own residency table, which nothing before a
resolve touches, so the whole build could run while the predecessor drained. Keying the
table by `k` rather than by the cache's capacity was worth giving that up; see
"Probe index". The copy resolves its pool, its
buffer and its work list from descriptors and block coordinates; the flush-list scan
addresses its row from the staged view. Each then waits for the one thing that is genuinely
produced upstream.

**The work-list counters are cleared once a step, in `publish()`**, not once per producer.
That is not a micro-optimisation: a clear is a memset, and a memset is not a kernel, so one
standing between every resolve and whatever ran before it would sever the launch dependency
at each of a model's layers -- for DeepSeek-V4, sixty-one times a step.

What makes one clear enough is that the counters are per layer and per direction, so no two
producers in a step share one. The fetch keeps a *single* list, reused from layer to layer,
because a layer's copy has consumed it before the next layer's resolve overwrites it; only
the counter has to read zero when a resolve starts. Per-layer lists would have cost the
layer count times the list, which is megabytes, against a counter's four bytes.

A step that somehow skipped `publish()` would have its second resolve add on top of the
first's count, and the claim's bound check would fail rather than quietly overrun the list.

Global **writes** stay below the wait in all three, even where the datum looks private. A
block that has not yet waited may still be running beside a grid that could read what it
would overwrite, and the cost of keeping the rule is a few stores.

The producers release as early as they correctly can. The resolve's work list is final at
the end of its eviction loop, and everything after that — the recency rebuild, three block
scans over the resident set — is its own bookkeeping that nothing downstream reads; it calls
`cudaTriggerProgrammaticLaunchCompletion()` there and hands the copy that whole tail as
overlap. The flush-list scan has no such tail, since the list is the last thing it writes,
so its release only brings the copy's launch forward.

The copy releases immediately after its own wait. That looks wrong and is not: the kernel
that follows it on the stream builds the attention index table from the resolve's slots and
never reads a cell, so it must not be ordered behind a copy it does not depend on.

## Where sparse data lives

A sparse buffer's pages are ordinary hot pages that happen to be host memory, device-mapped
so the fetch reads them without a host round trip. Cells come from a per-life-cycle arena,
leased per request; the fetch reads hot pages at `BufferAttr::offset`.

A sparse deployment places no requirement on its cold tiers. History is hot, so level 1 may
be host, disk, or absent.

Memory kind is a property of the **pool**, not of the level and not of the life cycle. The
GPU tier allocates a sparse pool from host memory and a dense one from device memory, in the
same group; `AttnLifeCycle` carries no `isSparse`.

A layer may therefore hold both a sparse buffer and a dense one, which is what a model
wants: an indexer and the data it selects over are one attention computation and belong in
one layer. Sparsity constrains pool formation and nothing above it, so two layers differing
only in sparsity share a life cycle — one layer group, one page table in `Batch`.

**The codec is the hot↔cold transform**, not the level 0 → level 1 one.
`StorageManager::migrate` computes `srcIsHot = srcLevel == kHotLevel` and performs a plain
byte copy when source and destination are on the same side, invoking the codec only across
that boundary. Sparse history is hot, so it crosses that boundary on its way to a cold tier
like any other page: compression needs nothing sparse-specific.

The arena is sized from `constraints`, which already declares the batch shape `Batch` derives
its rows from, so the cells are a static reservation rather than an on-demand draw on the
shared pool.

### Sizing a host-backed pool

Sparsity is a property of a **pool**, not of a life cycle or of a pool group. A pool is
uniformly sparse or uniformly dense, because buffer grouping keys on `(size, isSparse)`;
the GPU tier allocates the sparse ones from host memory, device-mapped, and reports
`GPU_MEM` for the group either way.

That it reports GPU is sound because nothing reads the hot side's tier. The codec path keeps
only `coldTier = srcIsHot ? dstTier : srcTier`, and the copy-engine path needs both ends
hot, which never happens; host memory is device-accessible, so a mixed group needs no second
tier.

**A sparse pool's bytes are excluded from the device quota.** A slot is one row spanning
every pool of its group, so once a group holds at least one device-backed pool, the quota
sizes that pool and the sparse pools inherit the same row count. The quota therefore charges
only the dense pools.

This is why a pool group must hold **at least one dense pool**: a wholly sparse group would
have no device bytes per slot and so no slot count to derive. That is a validation, not a
heuristic.

**But the row count is not free.** Inheriting it gave the sparse pools no bound of their own:
their host memory grew with whatever the device quota bought, and the only way to hold it
down was to shrink the device cache that everything else shares.
`KVCacheManagerConfig::hostSparsePoolQuota` is that bound, and the allocator takes it as a
second budget over a disjoint set of the same pools rather than as a clamp applied afterwards.

The two budgets **partition** a group's pools — dense charged to the device quota, sparse to
the host budget, each in its own allocation grain — so one slot count settles both at once.
`ratioToSlotCountList` therefore takes every pool's size once, plus an index list naming the
pools each budget answers for. Three properties follow, and the first is the reason this is
one pass rather than two:

- **A group bound by one budget releases what it does not use of the other.** Clamping the
  slot count afterwards could only lower it; the device bytes a host-bound group then stopped
  using went to nobody, and a group with no sparse pool at all -- which can never be
  host-bound -- was left short for no reason.
- **Both budgets and `min_slots` are kept exactly; the pool ratio is what gives way.** The
  floors come off the top before anything is shared, so no division of the rest can break one.
- **Their only conflict is a configuration error, and it is caught once.** The floors can
  exceed a budget, but that names neither the ratio nor the device quota -- the two things a
  rebalance changes. So it is rejected the first time the allocator runs and cannot begin to
  fail later.

The split belongs to the hot level alone and is not a property a level can report about
itself: a cold group is a single pool holding dense and sparse buffers together, and a GPU
tier may itself be a cold level. The pools a quota pays for are therefore derived as the
complement of the ones the host budget names -- never passed alongside them -- so the two
cannot overlap, and no pool can be left for neither to pay.

#### The quota excludes sparse pools; the ratio counts them

These are deliberately different currencies, and the asymmetry looks like a bug until the
third party is in view.

- **The quota** (`roundedNumBytes(cacheTier())`) counts only the pools on the level's own
  medium. Host-backed bytes consume none of the device memory it measures.
- **The ratio** (`ratioList()`, and the profiler's) weighs *every* pool, host-backed
  included. It has to: it is projected onto cold levels, where a single pool holds dense and
  sparse buffers together, and it is compared against the achieved layout, which is measured
  the same way.

The consequence is easy to miss. `KvCacheManager::_needAdjustment` compares the layout's
current ratio against the target `constrainPoolGroupRatio` produces, and rebalances when they
differ. `constrainPoolGroupRatio` derives that target by *running the allocator* and
normalising the bytes it would produce -- so it must run it with the **same budgets**, the
sparse host quota included. Omit the budget there and the target describes a layout the
allocator will not produce: every comparison mismatches, and a rebalance re-triggers on every
check even though the profiler's answer never moved. The ratio, the measurement and the
constraint all have to speak all-bytes for that loop to settle.

What the asymmetry *does* require is that the division be driven by all-bytes demand, not
that the ratio be re-expressed per budget. Dividing the device quota alone by an all-bytes
ratio does not realise that ratio: a sparse-heavy group takes a device share its device cost
per slot does not warrant, and converts it into more slots than asked for.

The release step does not rescue this. It returns the device grains a group cannot use
*because the host budget held it* -- so where the host budget is generous, nothing is held,
nothing is released, and the over-share stands. The fix belongs in how the budgets are
divided: demand is computed from each group's all-bytes cost per slot, so the achieved
all-bytes shares track the ratio, and the per-medium budgets act as caps on that demand.
Re-expressing the ratio itself in a single medium's currency would be wrong for the reasons
above -- it has to stay comparable with what `ratioList()` measures and with what a cold
level inherits.

#### How the division runs

The allocator is a search over one scalar, and all-bytes is what that scalar buys.

Write `A(pg, n)` for what `n` slots of group `pg` cost in all-bytes — every pool rounded up
to the grain of the medium backing it, which is exactly `CacheLevelStorage::allMediaBytes()`. For
a *level* of all-bytes spending `L`, each group's demand is

    demand(pg, L) = max( minSlots[pg], maxSlotsForAllBytes(L · r[pg]) )

i.e. the most slots whose all-bytes cost fits that group's share of `L`, never below its
floor. `demand` is monotonic in `L`, so each budget admits a largest feasible level, found by
bisection: `levelQuota` for the device quota, `levelSecond` for the host budget. Both must
hold, so the search is run against both and the binding one wins.

Three details make the result exact rather than approximately right.

- **Floors come off the top.** Each group starts at `minSlots`, snapped up by `freeRun` to
  the largest count costing no more grains in any medium — a count short of what it has
  already paid for is not the count a re-derivation from the same bytes returns, and that
  disagreement is the grains↔slots round-trip bug in its original form. Host-backed pools
  count here even when no host quota is set: a pool nothing is bounding is still real memory,
  and letting the snap ignore it grows the floor to whatever the device grain absorbs.
- **A group frozen by one budget releases the other.** When the host budget pins a group, its
  slot count stops rising, and the device grains its demand would have claimed are returned to
  the search rather than reserved. This is what makes one pass strictly better than a quota
  followed by a clamp.
- **A bounded greedy finish spends the remainder.** Bisection lands on a level, not on an
  integral allocation; leftover grains in either budget are then handed out one group at a
  time, each step choosing the group whose resulting `A(pg, n)/r[pg]` is lowest. That is a
  leximin step on achieved all-bytes shares: equal shares mean the achieved ratio *is* `r`,
  so raising the currently worst-served group is the move that gets closest to it.

The fixed-point property falls out of measuring demand and achievement in the same currency:
feed the allocator the normalised all-bytes ratio of its own output and it returns the same
slot counts, which is what stops `_needAdjustment` from re-firing on an unchanged profile.
`test_feeding_back_the_layouts_own_ratio_changes_nothing` pins it, including the case where a
host-backed pool is most of a group's bytes and none of its quota.

A level's quota measures one medium, which is why host-backed bytes cannot be drawn from it:
they consume none of the device bytes it counts, so charging them would cap sparse history at
a GPU-sized number — the opposite of the point — while leaving those device bytes
unallocated. Attributing them to a *host tier's* quota instead was considered and rejected:
`cacheTiers` is a positional list, so "the host tier" may be several or none, and a hot
pool's size would then change when someone added a cold tier that has nothing to do with it.

**Coverage of the data path.** `TestSparseDataPath` asserts the bytes themselves on both
sides of the copy: a pattern written to a hot page must appear in the cell the fetch names,
and a pattern written to a cell must appear in its hot page once the watermark passes it.

This was worth building rather than arguing about. Offsetting the page base by one slot and
re-running showed **57 other sparse tests still passing** -- residency, index shape and
counts are all insensitive to which allocation the copy reads, which is exactly the mistake
the move to the hot tier could have introduced.

**No cell is ever read from, or written to, an encoded page.** Moving sparse pages to the
hot level made them eligible for demotion for the first time, which raised the question of
whether a page could be encoded while the cache still held a cell copied from it. It cannot:

- An `ACTIVE` request locks the pages it requires (`AGENTS.md`, *Per-request cache*), and
  `LOCKED` forbids eviction outright (*Page status*).
- Eviction is the only route to migration, and migration across the hot boundary is the only
  thing that invokes the codec.
- The lease is held inside the locks' lifetime, not alongside it: seating requires a resumed
  cache, which has already locked the pages it requires, and leaving returns the lease. A
  cell therefore exists inside its page's lock, with room to spare at both ends, in every
  configuration but the sliding-window one below.

`commit()` does downgrade a lock to a holder, but only for a page it is discarding in favour
of an existing tree page -- not one the request goes on to read.

The one configuration where a live request holds unlocked pages is a **sparse layer with a
sliding window**, where `_unlockStaleBlocks` releases out-of-window blocks. It is safe, and
not by caller discipline: the indexer that produces the selection applies the same window, so
a block the window has made stale is never named. Nothing reads an unlocked page, so nothing
can read one that has been encoded or recycled.

A cell fetched before the window moved does outlive its page's lock there. It is harmless:
nothing names it again, so the LRU evicts it within a few steps. The write-back is
unaffected in either case — the units it crosses lie in `[history, capacity)`, above the
watermark `_unlockStaleBlocks` works from, so their pages are locked for as long as the
request is active.

That is worth stating because it is the *selector*, not this package, that makes it true.
`fetch_sparse_pages` cannot detect an ineligible ordinal — the page table is uniform, so a
stale ordinal still resolves to a real page of the same request. The combination does not
arise today in any case: a sliding window already bounds a step's reads to a narrow fixed
span, which is the span a selection would otherwise have to discover.

## Settled by construction

Recorded because each looks like an open question and is not:

- **`num_blocks` lives on the device, in `Batch`.** Deriving it per call from the watermark
  would produce a host value the kernel cannot use, costing an H2D per layer.
- **Eviction never copies.** An entry is evictable only once flushed, so evicting is a
  free-list push. Flushes are triggered by the watermark alone, never by eviction pressure.
- **Pinning is not a general scheme.** The pinned set is the in-flight unit, private
  because unflushed blocks are not in the radix tree, and bounded to one by
  `max_input_length`. Ordering between the rest still comes from LRU recency alone.
- **No stale-key invalidation.** Per-request cache scope makes it impossible.
- **`tokens_per_block_override` is not the sparse granularity lever**, and is rejected on a
  sparse buffer. `sparse_selection_unit_tokens` is — see *Extraction interface*.

## Resolved

- **This *is* decode-only, deliberately.** Sharing one manager between prefill and decode
  is wrong in kind, not in size: a sparse kernel indexes its selection per query token, and a
  multi-token step's selections cannot all be made resident in one per-request cache. The
  decode instance of a disaggregated deployment is the supported target; see *Supported
  scope*.
- **A selection unit never exceeds `tokens_per_block`.** Sub-page units are supported
  through `expansion`; a unit spanning several pages is rejected, which retires the
  questions of slot contiguity and k-at-a-time eviction.
- **`N` is a uniform upper bound**, not per-request. Ragged device arrays would cost the
  static shapes graph capture needs.
- **History never moves backwards.** `resize()` rejects any decrease outright
  (`kvCache.cpp`: "History length cannot be decreased"), and the only other assignment
  initialises a fresh request from a prefix match. A flushed block can therefore never be
  rewound into, with no rule of our own required.

## References

- HiSparse: [arXiv:2608.07009](https://arxiv.org/abs/2608.07009),
  [SGLang docs](https://docs.sglang.io/docs/advanced_features/hisparse_guide),
  [LMSYS blog](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/)
- MiniMax Sparse Attention: [arXiv:2606.13392](https://arxiv.org/abs/2606.13392)
- DeepSeek-V4: [arXiv:2606.19348](https://arxiv.org/pdf/2606.19348)
