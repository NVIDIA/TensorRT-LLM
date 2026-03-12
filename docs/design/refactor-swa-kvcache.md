# Refactor SWA KV Cache

## Problem

Sliding Window Attention (SWA) requires evicting KV-cache blocks that slide outside the
attention window ("out-of-window" / OOW blocks).  Before this refactor, three separate
mechanisms were used to manage OOW blocks, each with correctness concerns:

1. **`mRepurposed` flag on `KVCacheBlock`** — set in `getFreeBlock` when another sequence
   steals an OOW block before the original sequence has stored it.  The flag was transient:
   if the stealing sequence freed the block without storing it, the flag remained set but
   the block content had been overwritten.  Worse, fresh blocks (never filled) could
   spuriously avoid being marked, creating an asymmetric invariant that was hard to reason
   about.

2. **`numOowBlocks` parameter in `storeBlocks`** — passed to distinguish which prefix
   positions correspond to OOW blocks so the stolen-block check could be applied only to
   those positions.  This made `storeBlocks` aware of caller-specific SWA state.

3. **`mCachedBlocksRootMutex` (recursive)** — a recursive mutex in `WindowBlockManager`
   used to serialise radix-tree access.  Recursive mutexes hide re-entrancy bugs and are
   harder to audit than plain mutexes.

Additionally, `storeContextBlocks` **skipped SWA windows entirely**, meaning SWA blocks
were never stored in the reuse trie during the context phase.  This violated the invariant
required for correct placeholder handling and prevented prefix reuse for SWA prompts.

## Design

### Core invariant

> Every fully filled block must be present in the radix reuse trie **before** its reference
> count drops to zero.

When this invariant holds, no block can be stolen with stale KV content without first
appearing in the trie.

### Placeholder approach

When a block slides out of the attention window (`detachFrontBlock`):

1. The block is **already in the trie** (stored by `storeContextBlocks` for context blocks,
   or by `storeNewBlock` for generation blocks).
2. Its reference count is decremented to zero; it enters the eviction queue with its
   original user-assigned priority unchanged.
3. Its slot in `mAllocatedBlocksPerSeq` is **replaced with a placeholder**
   (`KVCacheBlock::createPlaceholder`) that carries the same block-ID but has
   `mIsPlaceholder = true` and no GPU memory.

`storeBlocks` now accepts `std::vector<BlockPtr>` (block pointers, not IDs).  When it
encounters a placeholder at position *i*:

- **OOW block still in trie** (not evicted or stolen yet): `findMatchingBlock` succeeds →
  advance `searchRoot` to the cached position → continue storing subsequent blocks.
- **OOW block evicted or stolen**: `findMatchingBlock` returns null → chain is broken →
  stop (subsequent blocks cannot be anchored).

This replaces all three of the mechanisms above:

| Old mechanism | Replaced by |
|---|---|
| `mRepurposed` flag + `setRepurposed` in `getFreeBlock` | Placeholder in `mAllocatedBlocksPerSeq` |
| `numOowBlocks` parameter in `storeBlocks` | `block->isPlaceholder()` check in `storeBlocks` |
| `recursive_mutex mCachedBlocksRootMutex` | `std::mutex mMutex` inside `UnifiedBlockTree` |

### Mutex ownership

The radix tree (`UnifiedBlockTree`) owns a `std::mutex mMutex` exposed via `getMutex()`.
All callers lock `mLookupTree->getMutex()` directly.  Because `getFreeBlock` also acquires
the same mutex (to call `detachFromLookupNode`), `loadOrAllocateBlocks` uses a
`unique_lock` and releases the mutex around each `getFreeBlock` call to avoid deadlock.

`std::mutex` is not movable, so `UnifiedBlockTree` defines explicit move constructor and
move-assignment that move the trie contents (parent class) and default-construct a fresh
mutex in the destination.  This allows `BlockManager::resetReuseState()` to assign a
newly constructed tree to `mLookupTree`.

### `storeContextBlocks` for SWA

The previous SWA skip in `storeContextBlocks` is removed.  SWA context blocks are now
stored in the reuse trie during context processing, satisfying the core invariant: by the
time a block slides out of the window (which occurs `windowSize + tokensPerBlock` tokens
after the block fills), it is already in the trie.

## Affected components

| File | Change |
|------|--------|
| `radixBlockTree.h` | Add explicit move ctor + move-assignment to `UnifiedBlockTree` |
| `kvCacheManager.h` | Remove `mRepurposed`/`setRepurposed`/`isRepurposed` from `KVCacheBlock`; update `storeBlocks` signature |
| `kvCacheManager.cpp` | Implement placeholder insertion in `detachFrontBlock`; rewrite `storeBlocks`; remove `numOowBlocks` from all callers; remove SWA skip from `storeContextBlocks`; add `isPlaceholder()` guard in `schedulingReleaseBlocks` / `releaseBlocks` |
| `kvCacheManagerTest.cpp` | Replace `isRepurposed`-specific test with placeholder test; update all VSWA test comments |

## Why not alternatives?

**Why not keep `mRepurposed`?**  The flag is not structural: it must be set in `getFreeBlock`
(a generic code path) based on whether the block has prior token content.  Fresh blocks
(never written) must be excluded, creating an asymmetric condition.  The flag is also
invisible at the `mAllocatedBlocksPerSeq` level — callers must reconstruct the OOW
position from `numFrontBlocksRemoved`.

**Why not store OOW blocks inside `detachFrontBlock` itself?**  `detachFrontBlock` does not
have access to the request's unique tokens (needed to build `BlockKey`s).  Storing in
`detachFrontBlock` would require threading `LlmRequest` through multiple call sites, or
reading token content from the block itself (which only works post-`storeBlocks` when
`setBlockKey` has been called).  Satisfying the invariant via `storeContextBlocks` (context
phase) and `storeNewBlock` (generation phase) is simpler and already follows the existing
code structure.

**Why not use a sentinel block-ID (−1) as placeholder?**  `mAllBlocksById.at(-1)` throws,
causing `storeBlocks` to break at the OOW position via the existing exception handler.
This works but loses the ability to advance `searchRoot` to B0's trie position, so
subsequent in-window blocks (B1, B2, …) cannot be stored even when B0 is still in the
trie.  The `findMatchingBlock` path in the placeholder approach preserves the chain.

## Testing

All existing VSWA and TruePriorityEviction tests verify the same observable invariants
(no trie corruption, correct block counts, correct prefix reuse lengths).  The previously
`isRepurposed`-specific test (`VSWAStolenOOWBlockRepurposedWithoutStoring`) is replaced by
`VSWAPlaceholderReplacesOOWBlock`, which directly inspects the placeholder in
`mAllocatedBlocksPerSeq` (white-box) and also verifies the negative-reuse invariant
(black-box).
