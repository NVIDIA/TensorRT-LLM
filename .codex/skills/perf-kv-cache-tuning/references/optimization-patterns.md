<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Optimization Patterns for KV-Cache Features

These patterns removed real overhead from KV-cache features. Each one has a
guardrail; an optimization that breaks its guardrail is a regression, whatever
the speedup.

## Attribution Before Optimization

- **Use one ruler for every overhead chart.** Put the baseline (feature off, or
  FP8 KV cache) and the feature on the same axis. Break the feature's cost
  into the same named components in every chart.
- **List every kernel.** Do not label kernels "other". Before calling a kernel
  generic or unrelated, open its source and confirm who launches it.
- **Time a transfer or migration in a controlled test.** Use a fixed batch
  and fixed page counts. An end-to-end serving window mixes too many effects
  to attribute one transfer.
- **Compare components at the pressure they see in production.** A kernel
  timed alone at a tiny batch can rank differently at the production batch
  and sequence length.

## Host-Side Patterns

| Pattern | Do | Guardrail |
|---|---|---|
| Allocate once | Size scratch and staging buffers at initialization from the configured maxima, and account for them in the profiling run that sizes the KV cache memory budget | Check every caller: an object rebuilt per step re-runs its constructor, which turns a "once" memset of a large staging buffer into a per-step memset |
| Zero only what is read uninitialized | Drop blanket per-step `zero_()` when every element read is written first | Prove the read set is covered, for example with a sentinel-fill test. Do not rely on luck with stale data |
| Precompute static metadata | Build page-index maps, layer tables, and per-bucket metadata once, then update in place | Tensors captured in a CUDA graph must keep their addresses; update contents, never rebind |
| Stage host metadata in pinned memory | Write per-step metadata into pinned host buffers and copy it with one non-blocking transfer | Allocate pinned memory through the repository helpers (`maybe_pin_memory`, `prefer_pinned` in `tensorrt_llm/_utils.py`); a pre-commit check rejects raw pinned-memory calls in runtime code |
| Batch across requests and layers | One launch over all requests or layers instead of a Python loop of small launches | Keep per-request semantics identical; compare the batched launch bit for bit with the loop of small launches it replaces. If the batched kernel rounds differently, check its numerics locally on real runs and write the commands and results in the PR description |
| Remove synchronization from the step loop | Replace `.item()`, `.tolist()`, and host-side checks with device-side logic or checks done once at setup | Validation that matters moves to initialization or to an explicit debug mode, not into the hot path |
| Audit the hot path | Remove defensive clones, redundant copies, asserts that synchronize, and unnecessary protections | Remove only what a test proves redundant, and never a supported mode or feature |

## Device-Side Patterns

| Pattern | Do | Guardrail |
|---|---|---|
| Size work by actual length | In eager steps, convert or stage up to the longest sequence in the batch, not `max_seq_len` | Inside CUDA graphs the shape is static. Quantify the full-span cost and consider bucketing before accepting it |
| Fuse score, select, and move | Fuse kernels that pass large intermediates through global memory | Fusion must not change which tokens are selected or how ties break; compare selections bit for bit |
| Prefer existing production primitives | Reuse a library or in-tree kernel whose numerical and layout contract matches | Verify the contract: scale layout, rounding, supported dtypes, alignment |
| Vendor small kernels | Copy a small kernel in-tree instead of depending on a fragile external version or an unpublished API | Keep the vendored code minimal and attributed, with an owner test |
| Overlap migration | Run encode, decode, and transfer on a side stream with events, overlapped with model execution | Publish completion before any consumer reads or any resize or release; failure must not leave state half-updated |
| Keep index math wide | Use 64-bit offsets for page, slot, and element indices in large pools | The int32 overflow appears only at large pools or long sequences; test at the largest supported pool |

## Configuration and Kernel Selection

- A new KV cache dtype or layout can route attention to another kernel library, or
  to a library without a kernel for the required output dtype. Read the
  selected kernels in the profile or the startup log; the configured backend
  name is not enough.
- Fixing a regression by changing the experiment configuration, such as page
  size, pool ratios, the reuse policy, or a per-request option, is a design
  change, not an optimization. Propose it, measure it as its own arm (one
  configuration being compared), and keep the standard configuration as the
  reported baseline.
- Make policy changes, such as retention priority or eviction order, through
  the existing Python callbacks and configuration before touching the native
  core of KV cache manager V2. Measure them as their own arm.

## Validation After Every Optimization

1. Outputs identical to before for lossless changes, compared bit for bit or
   token for token. For lossy paths, compare against the accepted accuracy
   contract.
2. A comparison at the same batch size and sequence lengths, in the
   production configuration, repeated, with variation reported.
3. Peak memory and KV cache memory budget unchanged, or the change reported.
4. The compatibility combinations the feature claims (reuse, CUDA graph,
   overlap scheduler, disaggregated serving, speculative decoding) still
   pass.
