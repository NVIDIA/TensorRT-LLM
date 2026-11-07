<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Adding Kernels for KV Cache Compression Methods and KV Cache Manager V2 Features

This guide covers kernels for compression methods, cold-page codecs,
mixed-precision KV cache conversion, and KV cache manager V2 data movement.
The rule underneath it: the calling manager or compression provider owns
policy, and the kernel owns one batched transform.

## 1. Decide Whether a New Kernel Is Needed

Go through these in order and stop at the first that works:

1. **An existing production primitive with the same numerical and layout
   contract.** Examples are a quantization routine, a top-k selector, or a
   batched page copy. Check its scale layout, rounding, dtypes, and
   alignment.
2. **An existing kernel extended along one dimension.** The compaction op was
   extended with a layer dimension instead of being rewritten. Its other
   callers must stay bitwise unchanged.
3. **A fusion of existing steps**, when large intermediates go through global
   memory.
4. **A new kernel.** Write one only when a profile of the production
   configuration shows the gap, and say so in the design gate.

Do not replace a kernel with a chain of PyTorch ops to avoid writing one.
Per-step chains of small ops cost more than the kernel did.

## 2. Choose the Implementation Path

| Path | Where it lives | Use it for | Example on main |
|---|---|---|---|
| Triton | The method's Python package | Gathers, packing, index math, simple elementwise transforms | `_pack_move_sources_kernel` in `kv_cache_compression/compaction.py`; `triattention/triattention_kernels.py` |
| CuTe DSL | The method's package, or `tensorrt_llm/_torch/attention/kernels/blackwell/` for attention-adjacent conversion | Blackwell-specific fused scoring, selection, and conversion | `triattention/triattention_cute_score_fused.py`; `attention/kernels/blackwell/mixed_precision_kv.py` |
| CUDA C++ | `cpp/tensorrt_llm/kernels/`, registered with `add_tllm_kernel_library(...)` in `kernelComponents.cmake` | Performance-critical transforms on raw pointers, such as cold-page codecs on migration streams | `nvfp4ColdPageKernels.cu` (`tllm_kernel_nvfp4_cold_page`) |

A C++ kernel is exposed to Python in one of two ways:

- **As a torch custom op** in `cpp/tensorrt_llm/thop/`. Declare it with
  `TORCH_LIBRARY_FRAGMENT(trtllm, m)` and `m.def(...)`, implement it under
  `TORCH_LIBRARY_IMPL(trtllm, CUDA, m)`, and call it as
  `torch.ops.trtllm.<name>`. Use this for tensor-in, tensor-out work.
  Example: `sparse_kv_cache_compact_layers` in `sparseKvCacheCompactOp.cpp`.
  If the op runs inside a `torch.compile` region, add its fake (meta)
  implementation in `tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py`.
- **As a nanobind function** in the feature's binding module. Use this when
  the caller passes raw pointers, a stream, and scalars, as a cold-page codec
  launcher does. Example: `nvfp4_cold_page_encode` in
  `cpp/tensorrt_llm/nanobind/kvCacheCompression/bindings.cpp`, imported
  locally by the compression provider as
  `from tensorrt_llm.bindings.internal import kv_cache_compression as native`.

Keep imports of method kernels local to the factory branch or launcher, so
that other features do not pay their import or just-in-time (JIT)
compilation cost.

## 3. Contract Checklist

- **Stream.** Enqueue on the supplied or current stream. No device or stream
  synchronization inside a launcher. A boolean return means "submitted".
- **Batching.** One launch per batch of requests, pages, layers, or heads, not
  a Python loop of launches. Handle non-contiguous and partial pages, and the
  largest batch the caller can send.
- **Index width.** Use 64-bit offsets for page, slot, and element addressing.
  Triton treats Python integers as 32-bit, so cast before multiplying strides.
  At initialization, reject configurations whose worst-case size would
  overflow a 32-bit path.
- **Invalid entries.** A `-1` page entry must never be used as an address.
  Mask it, or map it to a safe page, before the load.
- **Alignment and dtypes.** State the required alignment (for example
  16-byte rows, 16-element scale groups) and the supported dtypes. Reject
  everything else with an actionable error.
- **Architecture.** Gate device code and imports by architecture family,
  for example Ampere (SM80, SM86), Hopper (SM90: H100, H200), datacenter
  Blackwell (SM100: B200, GB200; SM103: B300, GB300), and consumer Blackwell
  (SM120, SM121), using the helpers
  in `tensorrt_llm/_utils.py` such as `is_blackwell_datacenter()`. Give an
  actionable error on unsupported GPUs before any kernel is compiled, and
  test on every family you claim.
- **Determinism.** Selection kernels define tie-breaking explicitly, so that
  fused and unfused versions, and every tensor-parallel rank, agree.
- **CUDA graphs.** No allocation inside a captured region, static shapes and
  addresses, lengths read from device memory. Compile every JIT variant during
  warm-up, before capture.
- **Workspaces.** Allocate them once, at bind or initialization time, from
  the configured maxima. Count them in the KV cache memory budget.
- **Shared memory and launch limits.** Large selection widths can exceed
  shared-memory limits. Bound the parameters at configuration time.

## 4. Wire It Into the Pipeline

- The calling manager or compression provider decides what to transform; the
  launcher turns that decision into one call; the kernel does the transform.
- Pass layouts and keep decisions to generic kernels, not algorithm scores or
  opaque dictionaries.
- Order the work on the stream that the next consumer uses. Record a
  completion event, and publish lengths or mappings only after it.
- Keep host work per call constant in the batch size. Write metadata into
  pinned buffers allocated with the repository helpers (`maybe_pin_memory`,
  `prefer_pinned`), and copy it in one transfer.

## 5. Test It

1. **Standalone, during development.** Compare exact operations byte for
   byte with a CPU reference implementation. Test lossy operations through
   invariants, not a numeric reference: spans, scale groups, byte-exact
   lossless parts, and identical re-encode where the format promises it.
   Cover non-contiguous, partial, and large batches, and non-default
   streams. Keep only small, fast checks of this kind in pytest. Check lossy
   kernel numerics, such as the encode/decode round-trip accuracy for every
   supported runtime dtype, locally on real runs, and write the commands and
   results in the PR description.
2. **C++ gtest.** For C++ kernels, register it with `add_tllm_gtest(<name>
   OWNER ... SOURCES ... LINK_LIBRARIES ... LABELS "unit;kernels;...;gpu")` in
   `cpp/tests/unit_tests/kernels/CMakeLists.txt`. Make sure a CI stage runs
   the binary: some integration fixtures build and run only named binaries,
   and must be extended. Limit the gtest to the checks item 1 keeps in
   pytest; never compare lossy outputs against a hand-written numeric
   reference with a tolerance.
3. **Python tests.** Add them under the feature's test directory, and register
   new files in the test-db. Keep them small, with few GPU tests.
4. **In the pipeline.** Run under CUDA graphs and the overlap scheduler, with
   the KV cache manager V2 debug fills on, and compare against the
   feature-off path.
5. **Performance.** Do a microbenchmark for kernel development, then the
   end-to-end comparison in `perf-kv-cache-tuning`
   (`../../perf-kv-cache-tuning/SKILL.md`). The microbenchmark alone proves
   nothing about the feature.

## 6. Pitfalls Seen Before

- A per-layer launch made a fused quantize-and-copy slower than the plain
  batched copy it replaced. Launch at the granularity of the existing batched
  path.
- A fallback kernel dereferenced `-1` entries for out-of-window pages before
  masking them.
- Index math wrapped past 2^31 in large pools, but only at the largest
  supported sizes. Keep a large-size regression test.
- New C++ tests existed but no CI stage ran them.
- A reference implementation in a test was wrong, while the kernel was
  bit-correct. Keep tolerance-based comparisons against hand-written
  references out of pytest and gtests: test invariants there, and check
  lossy kernel numerics locally on real runs.
