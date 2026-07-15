<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Layer-wise Hugging Face Safetensors Loading for DeepSeek V4

## Status

This document describes the experimental implementation on branch
`fix/dsv4-mixed-precision`. The implementation is opt-in, supports the PyTorch
backend, and currently enables semantic-layer loading only for DeepSeek V4.

The immediate goal is to avoid host OOM while loading the
DeepSeek-V4-Pro-NVFP4 checkpoint on a 952 GiB GB300 node. It is not yet intended
to be a generic streaming-loader API for every model or checkpoint format.

## Problem statement

The normal Hugging Face loader materializes the complete checkpoint as one
host-side weight dictionary before the model-specific loader remaps, splits,
concatenates, transforms, and copies weights to the device. For a large NVFP4
checkpoint, all four tensor-parallel ranks can therefore retain both raw
checkpoint tensors and transformation intermediates at the same time. The
observed aggregate demand exceeded node memory.

DeepSeek V4 transformations are not all tensor-local. Examples include fused
attention projections, compressor `wkv` plus `wgate`, fused MoE projections,
scale conversion, TP slicing, and MTP name remapping. Streaming one tensor at a
time would break these fusion contracts. The smallest practical unit is a
semantic model layer.

## Goals

- Bound live raw checkpoint storage to top-level tensors or one semantic layer.
- Preserve the existing DeepSeek V4 remap, fusion, quantization, TP/EP split,
  and module-copy logic.
- Keep all tensors needed by a same-layer fusion in the same bucket, even when
  the tensors reside in different safetensors shards.
- Release a bucket only after asynchronous device consumers have completed.
- Leave the default eager loading path unchanged unless explicitly enabled.
- Fail before silently producing a partially or incorrectly loaded model.

## Non-goals

- Streaming `.bin` or `.pth` checkpoints.
- Supporting a separate speculative draft checkpoint in the first version.
- Eliminating the Linux page cache populated by checkpoint reads.
- Eliminating every host-side transformation temporary inside one layer.
- Providing a public LLM configuration field. The current switch is an
  environment variable to minimize API surface while the design is reviewed.
- Making arbitrary models layer-wise loadable without model-specific support.

## User-facing activation and compatibility boundary

Set:

```text
TRTLLM_HF_LAYERWISE_SAFETENSORS=1
```

Accepted true values are `1`, `true`, `yes`, and `on`, case-insensitively. The
feature remains disabled by default.

When enabled, `ModelLoader` verifies that `model.load_weights` explicitly
accepts `initial_bucket_loading` before opening the first bucket. Consequently,
an unsupported model fails early instead of receiving partial weights. The
DeepSeek V4 implementation is currently the only intended consumer.

## Architecture

```text
HF safetensors index
        |
        v
HfWeightLoader: build metadata-only bucket plan
        |  top -> layer 0 -> ... -> layer N-1 -> MTP
        v
HfCheckpointLoader: expose capability and bucket iterator
        |
        v
ModelLoader: initialize mapper once, then iterate buckets
        |
        v
DeepseekV4WeightLoader: remap and load only the active semantic layer
        |
        v
CUDA synchronize -> drop bucket -> close safetensors handles
        |
        v
Run the existing global post-load path once
```

The implementation changes four runtime layers:

1. `HfWeightLoader` plans and materializes semantic buckets.
2. `HfCheckpointLoader` exposes the opt-in capability and iterator.
3. `ModelLoader` owns bucket lifetime and synchronization.
4. `DeepseekV4WeightLoader` scopes existing loading logic to the active bucket.

## Bucket planning and lifetime

### Metadata discovery

`HfWeightLoader.iter_layer_weight_buckets` first discovers `.safetensors`
files. For an ordinary sharded checkpoint, it reads the first
`*.safetensors.index.json` as metadata and builds `(tensor name, shard path)`
entries without loading tensor storage.

Before model mutation, it verifies that every shard referenced by the index is
present and that the index has a non-empty `weight_map`. This prevents an
incomplete checkpoint directory from being interpreted as a valid partial
model.

For checkpoints without an index, the loader enumerates keys from each
safetensors file and rejects duplicate keys across files. Consolidated files
are intentionally handled independently of an ordinary sharded index when
`use_consolidated=True`.

### Semantic classification

Names are assigned to buckets as follows:

| Checkpoint name | Bucket |
|---|---|
| `layers.<n>.*` | `("layer", n)` |
| `model.layers.<n>.*` | `("layer", n)` |
| `mtp.<n>.*` | `("mtp", n)` |
| Everything else | `("top", 0)` |

Buckets are yielded in deterministic order: top-level weights, numeric base
layers, then numeric MTP layers. Classification is independent of shard
boundaries, so all same-layer fusion inputs remain atomic.

### Storage lifetime

Only the current bucket is materialized. An `ExitStack` keeps every
`safe_open` handle used by that bucket alive across the `yield`. Tensor storage
therefore remains valid while the model-specific loader consumes it. Handles
close when the caller advances the iterator or unwinds it after an exception.

After `model.load_weights` returns, `ModelLoader` calls
`torch.cuda.synchronize()`. This is required because non-blocking copies or
backend transforms may still reference mmap-backed source storage. The bucket
reference is deleted only after synchronization.

The Linux page cache may remain large after handles close. It is reclaimable
file-backed memory and is distinct from Python-owned anonymous tensor storage.

## Model-loader orchestration

The eager path remains the default. In layer-wise mode, `ModelLoader`:

1. Resolves `model.llm_checkpoint_dir` when present, otherwise the ordinary
   checkpoint directory.
2. Rejects a separate speculative draft checkpoint.
3. Checks the model loading capability before consuming the iterator.
4. Initializes the existing checkpoint weight mapper once.
5. Calls `model.load_weights(bucket, initial_bucket_loading=True)` for each
   bucket.
6. Synchronizes CUDA and releases the bucket.
7. Continues through the existing post-load path after all buckets complete.

`post_load_weights` is deliberately not run per bucket. Global alias setup,
derived state, and other finalization still occur once after the complete model
has been populated.

An error after earlier buckets have loaded leaves the in-process model
partially mutated, but startup fails and the model is never published. The
implementation does not attempt transactional rollback.

## DeepSeek V4 model behavior

`DeepseekV4WeightLoader.load_weights` accepts
`initial_bucket_loading=False` by default. When true, it derives one
`active_layer` from checkpoint keys and validates that the bucket is either:

- top-level weights only, or
- exactly one base/MTP semantic layer with no top-level weights mixed in.

The existing raw-key remap still performs the same fusion and quantization
logic. Named-parameter default synthesis and named-module traversal are scoped
to the active layer:

- A top bucket skips every `model.layers.*` module.
- A base-layer bucket visits only the matching runtime layer.
- An MTP bucket visits runtime MTP replicas and reuses the existing modulo
  mapping back to checkpoint MTP layers.
- Synthesized indexer defaults are created only for the active base layer.

No known DeepSeek V4 checkpoint fusion crosses a semantic layer boundary. The
attention, compressor, MoE, scale, and TP transformations used by this loader
consume tensors from one layer or from the top-level bucket.

The same patch also fixes routed-expert scale-layout detection for an isolated
`mtp.0.*` bucket. Without recognizing the MTP prefix, MTP-only loading could
select the wrong MXFP4/NVFP4 interpretation. This is a correctness fix rather
than an intentional numerical change.

## Numerical-equivalence argument

Layer-wise loading changes scheduling and lifetime, not the mathematical
operations. It uses the same checkpoint tensors, mapper, remap functions,
fusion order, TP/EP slicing, scale transforms, destination dtypes, and module
copy routines as eager loading. CUDA synchronization adds an ordering barrier
but does not change values.

The equivalence argument depends on these invariants:

- Every fusion input is in the same bucket.
- The model-specific loader never reads a different base layer while loading
  the active layer.
- Global post-load operations run once after all buckets.
- Missing or malformed buckets fail loudly rather than being silently skipped.
- MTP runtime replicas map to the same checkpoint MTP layer as eager loading.

These invariants require unit tests and an end-to-end accuracy gate; successful
model startup alone is not sufficient proof of numerical equivalence.

## Unit-test design

### Tests already implemented

`tests/unittest/_torch/models/checkpoints/hf/test_weight_loader.py` contains:

- `test_layerwise_safetensors_keeps_cross_shard_layer_atomic`: verifies that
  same-layer tensors split across shards are yielded together and that order is
  top, base layers, MTP.
- `test_layerwise_safetensors_without_index_discovers_keys`: verifies fallback
  key discovery and support for `model.layers.<n>` names.
- `test_layerwise_safetensors_rejects_missing_index_shard`: verifies that all
  index-referenced shards are validated before the first bucket is yielded.

`tests/unittest/_torch/modeling/test_modeling_deepseekv4.py` contains:

- `test_deepseek_v4_weight_remap_for_mtp_mxfp4_routed_experts`: verifies MTP
  routed-expert scale-layout detection and remapped dtypes.

These tests are CPU-only in intent. Full test-file collection still requires a
Linux TensorRT-LLM development environment because module imports depend on the
TensorRT-LLM runtime stack.

### Required tests before upstreaming

#### HF bucket iterator

- Environment parsing: false by default and all documented true spellings.
- Natural numeric order: layer 2 precedes layer 10 regardless of index order.
- Empty index rejection.
- Duplicate-key rejection for no-index multi-file checkpoints.
- Consolidated selection does not accidentally consume an ordinary sharded
  index or non-consolidated files.
- Non-safetensors input fails with the documented error.
- Handle lifetime: handles remain open during bucket consumption and close when
  advancing, on normal completion, and on consumer exception.
- Tensor contents exactly match the source files for cross-shard buckets.

#### ModelLoader orchestration

Use fake checkpoint loaders and a small fake model to avoid GPU/model weights:

- Mapper initialization occurs once and before the first bucket load.
- One model load call occurs per bucket in iterator order.
- `initial_bucket_loading=True` is passed only on the layer-wise path.
- CUDA synchronization occurs once after every successful bucket load and
  before the iterator advances.
- An unsupported model fails before the iterator is consumed or model state is
  changed.
- A separate draft checkpoint fails before bucket consumption.
- `llm_checkpoint_dir` is honored.
- Consumer exceptions close current handles and prevent later buckets and
  global post-load publication.
- The ordinary eager, preloaded, MX, GMS, and draft-loading paths retain their
  existing call ordering when the environment variable is unset.

#### DeepSeek V4 bucket scoping

- Reject a bucket containing two base layers.
- Reject a bucket mixing top-level and layer weights.
- A top bucket never visits layer modules.
- A base-layer bucket visits only its matching runtime layer.
- Synthesized indexer defaults are limited to the active layer.
- An MTP bucket reaches every intended runtime MTP replica and preserves modulo
  checkpoint mapping.
- Missing members of each fused group fail with the expected key/error rather
  than silently loading a partial fused tensor.
- A small synthetic model loaded eagerly and bucket-by-bucket produces an
  identical `state_dict`. Device-only transforms may be mocked for a CPU test,
  followed by a real-GPU equivalence test.

### Accuracy and integration gates

The following tests complement, but do not replace, unit tests:

1. Load the real checkpoint with TEP4 in CTX-only and GEN-only configurations.
2. Confirm all ranks traverse top, layers 0 through 60, and MTP without OOM.
3. Compare eager and layer-wise loading on a smaller compatible fixture using
   exact state-dict equality after post-load.
4. With fixed prompt, seed, and greedy decoding, compare generated token IDs.
5. If logits are available, compare first-token logits or top-k logits under an
   agreed tolerance and check for NaN/Inf.
6. Record per-task `MaxRSS`, job cgroup `memory.peak`, current anonymous memory,
   and file cache separately.
7. Run CTX/GEN throughput smoke tests to catch runtime-only fusion, shape, or
   communication failures.

## Current validation evidence

As of 2026-07-14:

- Static checkpoint-index validation found 285,660 tensors in 63 semantic
  buckets: one top bucket, 61 base layers, and one MTP bucket.
- `py_compile` passed for the modified source and test files.
- `git diff --check` passed.
- Real TEP4 CTX-only and disaggregated CTX/GEN workers loaded every bucket and
  reached server/benchmark startup without host OOM.
- The observed node cgroup peak was approximately 874-878 GiB on a 952 GiB
  node. After load, anonymous memory dropped substantially; most remaining
  cgroup memory was reclaimable checkpoint page cache.
- The focused pytest cases have not yet run in a complete development image;
  login and benchmark images used during bring-up did not include pytest and
  compute nodes could not fetch it from PyPI.

## Known limitations and review focus

- All ranks still read the full checkpoint namespace; this is layer-bounded,
  not rank-presharded I/O.
- Peak host memory is now below node capacity but remains high. Per-rank
  historical `MaxRSS` was approximately 227-235 GiB in the real checkpoint
  test. Review should separate anonymous transformation storage from mmap/file
  cache accounting.
- Bucket classification is name-pattern based and currently understands top,
  base layers, and MTP. New DeepSeek V4 checkpoint namespaces must be audited.
- Capability discovery uses Python signature inspection. Decorated loaders or
  future keyword-only signatures should be considered during review.
- The iterator API currently lives on the HF concrete loader rather than a
  generic base interface.
- There is no fused-group preflight manifest. Completeness currently relies on
  index validation plus strict failures in existing model-specific transforms.
- The opt-in is process-global. It intentionally fails early for unsupported
  models, but a future public configuration may need per-model scope.
- Page cache remains charged to the job cgroup until reclaimed by the kernel.
  This is expected and must not be confused with live Python-owned weights.

## Reviewer checklist

- Verify eager behavior is byte-for-byte unchanged when the environment
  variable is unset.
- Audit every DeepSeek V4 fusion for cross-layer dependencies.
- Audit bucket release ordering around asynchronous CUDA work.
- Verify top, base-layer, and MTP name classification against all supported
  checkpoint naming variants.
- Check MTP replica routing for `num_nextn_predict_layers > 1`.
- Check exception behavior for missing shards, missing fused members, and
  mid-stream consumer failure.
- Confirm no post-load transform is incorrectly executed once per bucket.
- Require eager-versus-layer-wise state/output equivalence before generalizing
  the feature to additional models.
