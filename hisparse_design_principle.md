<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# HiSparse Design Principle: Generalize, Don't Hardcode

## The ask

We'd prefer a generalized approach that supports this method (HiSparse: top-K
selection + LRU refetch + host offload), scales seamlessly to others, and
leaves room for experimental explorations in the sparse attention + KV cache
offloading area.

## Why this matters now, not later

HiSparse involves three choices. Keep them clear even when one kernel handles
more than one:

1. **Selection policy** — which KV blocks/tokens matter for the next step.
   Today: top-K. Tomorrow: other sparse-attention schemes (Quest, NSA,
   MoBA-style, or whatever DeepSeek/GLM/Minimax ship next). These differ in
   *how* they score relevance, not in how staging/refetch works.
2. **Eviction/refetch policy** — which GPU copies to replace and which host
   entries to fetch. Today: LRU. Future kernels could use CLOCK or another policy.
3. **Transfer mechanism** — the actual host↔device copy/staging kernel. This
   is mechanism-agnostic: the same warp-vectorized copy kernel shouldn't care
   whether the caller picked top-K or Quest for selection.

Model selection, KV layout, and KVCM ownership must stay separate from kernel
execution. This lets different models supply their own selections and layouts
without rewriting storage management. HiSparse can still fuse lookup, LRU, and
copying in one kernel for speed.

This is not hypothetical: we're already targeting 3 models with different
attention variants (MLA/DSA, DeepSeek-V4 C4), so this pressure hits almost
immediately.

## Prior art (and its lesson)

SGLang's own codebase shows both sides of this. Its `mem_cache/sparsity/`
package splits selection into a coordinator + pluggable `BaseSparseAlgorithm`
+ backend adaptor — clean, and worth emulating architecturally. Its
allocator/memory-pool/coordinator layer, by contrast, is fairly hardwired to
scheduler internals — not portable, and the part that would need a rewrite
rather than an adaptation if reused as-is.

## What this means for the design doc

Use these interfaces:

- **`SelectedEntries`** — borrows model positions and existing batch metadata.
- **`EntryFormat`** — describes model tensor shapes, dtypes, entry axes, and compression,
  keyed by `BufferId`/`DataRole`. KVCM and the codec supply physical layout.
- **`HostSourceView`** — borrows KVCM's host-source table. KVCM owns its locations
  and completed token counts; read scopes protect the table and host copies.
- **`ensure_resident()`** — takes logical selections, separate model formats, codec layouts, and `HostSourceView`, and
  mutable GPU-cache state directly. HiSparse finds hits, replaces GPU copies with
  LRU, fetches misses, and returns indices protected through attention.

HiSparse is the planned GPU-cache implementation. Future experiments can replace
its kernel adapter while keeping selection, layout, and ownership interfaces.
Separate policy or copy launches are not required.

## Trade-off to keep in mind

Fusing LRU and copying means changing either may require another kernel adapter.
Keep the model and storage inputs stable so that work stays outside KVCM core.
Measure performance changes with the same layout, cache capacity, and protection
rules.
