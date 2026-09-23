<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AutoDeploy Pipeline Cache Design

## Summary

The AutoDeploy pipeline cache skips the expensive front half of the AutoDeploy transform pipeline
for repeated runs of the same model, checkpoint, distributed configuration, and transform prefix.

The cache point is represented as a normal AutoDeploy transform named `pipeline_cache`. On a cache
miss, the transform snapshots the incoming pre-weight-loading module. On a cache hit, the optimizer
restores that snapshot before running the transform prefix and resumes execution immediately after
the cache point.

The current implementation intentionally supports only the minimal surface needed by the validated
pipeline cache path:

- FX `GraphModule` snapshots, including wrappers that contain `GraphModule` children.
- Pre-weight-loading modules.
- Declarative load-state-dict pre-hooks required by export, sharding, deduplication, and aliasing.
- Per-rank cache entries for distributed runs.

Unsupported hooks and unsupported module shapes cause cache save/restore to be skipped instead of
adding broader serialization logic.

## Goals

- Reduce repeated AutoDeploy startup time by avoiding rebuild/export/pattern-match/sharding work
  before the configured cache point.
- Keep cache correctness tied to model identity, checkpoint identity, transform-prefix config, and
  relevant distributed config.
- Store artifacts that are durable across processes without depending on raw FX graph pickling.
- Keep the supported hook surface explicit and small.
- Fail open: if a cache entry is missing, invalid, or unsupported, run the normal pipeline.

## Non-Goals

- Caching post-weight-loading modules or GPU-resident parameters.
- Supporting arbitrary Python hooks.
- Supporting forward hooks.
- Making cache entries portable across arbitrary code changes.
- Replacing later runtime compilation, KV cache allocation, or CUDA graph capture.

## Pipeline Placement

`pipeline_cache` must be placed at or before the sharding stage and before `load_weights`.

```text
build_model -> export_to_gm -> pattern transforms -> sharding -> pipeline_cache
    -> load_weights -> post-load fusion -> cache init -> compile
```

On a miss, execution is unchanged except the cache transform writes an artifact at the cache point.
On a hit, the optimizer restores the cached module and resumes after `pipeline_cache`:

```text
restore cached module -> load_weights -> post-load fusion -> cache init -> compile
```

This placement keeps weights out of the artifact while still skipping the expensive model build,
export, graph cleanup, pattern matching, and sharding prefix.

## Optimizer Integration

`InferenceOptimizer` keeps two versions of the transform config:

- `self.config`: the normal config used to run transforms.
- `self._cache_key_config`: a deep copy captured before transforms mutate config objects.

Before running transforms, the optimizer asks cache-capable transforms whether they can restore:

```text
InferenceOptimizer.__call__
    -> _maybe_restore_from_cache()
        -> PipelineCache.maybe_restore()
```

If restore succeeds, the optimizer starts at the transform immediately after `pipeline_cache`.
If restore fails or misses, the optimizer starts from the beginning.

The cache transform receives the stable `self._cache_key_config` so cache keys are based on the
original user configuration, not on mutations performed by earlier transforms.

## Cache Key

The cache key is a hash of:

- Model identity from `ModelFactory.get_pipeline_cache_model_identifier()`.
- Checkpoint fingerprint from `ModelFactory.get_pipeline_cache_checkpoint_fingerprint()`.
- Hash of transform configs before the `pipeline_cache` transform.
- Distributed config when the cached prefix includes sharding-stage transforms.

The distributed config excludes rank, so all ranks in the same run agree on the same cache entry
directory but write separate rank subdirectories.

The cache entry path is:

```text
{root}/{cache_key}/rank_{rank}/
```

The default root is:

```text
~/.cache/tensorrt_llm/auto_deploy/pipeline_cache
```

## Cache Reuse Across Config Changes

The cache key covers only the model/checkpoint identity, the transform configs before
`pipeline_cache`, and the distributed topology needed by that prefix. On a hit, the optimizer resumes
at the transform immediately after `pipeline_cache`, so later transforms and executor/runtime setup
run with the current run's configuration.

For example YAML files such as `examples/auto_deploy/nano_v3.yaml`,
`examples/auto_deploy/nano_v3_multi_device.yaml`, and the files under
`examples/auto_deploy/model_registry/configs/`, use this rule:

- Fields that are only used after `pipeline_cache` can change and still reuse the same cache entry.
  This includes all `transforms.*` entries whose stage is after the `pipeline_cache` stage.
- Fields used at or before `pipeline_cache` must be kept fixed for cache reuse, because changing them
  changes the module snapshot that the cache is meant to represent.

Common top-level fields in those YAML configs:

- `max_batch_size`: cache-reusable. It sizes runtime buffers, schedulers, and CUDA graph capture after
  restore.
- `enable_chunked_prefill`: cache-reusable. It is consumed by runtime scheduling after model
  optimization.
- `attn_backend`: cache-reusable for the default graph pipeline configs. The shortcut updates cached
  attention insertion, which runs after `pipeline_cache`.
- `compile_backend` and `cuda_graph_config`: cache-reusable. They affect compile/CUDA graph work after
  restore.
- `kv_cache_config`: cache-reusable when it only changes cache allocation/runtime sizing.
- `max_seq_len`: not generally cache-reusable today. Although Nano-style configs mark it as tunable,
  pre-cache graph rewrites can consult the factory's `max_seq_len` before `pipeline_cache`. Clear the
  cache, move the cache point earlier, or extend the cache key before relying on cache reuse across
  `max_seq_len` changes.
- `max_num_tokens`: cache-reusable only when it is not embedded by pre-cache sharding. MoE all-to-all
  sharding paths can write this value into graph ops before the default cache point, so treat it as
  cache-affecting for those configs.

The important boundary is behavioral, not the field name. If a field changes graph construction,
export, pattern matching, quantization, sharding, hook generation, or distributed layout before the
cache point, it must either be included in the cache key or cause a cache miss. Examples include model
identity, checkpoint identity, tokenizer/model kwargs that affect model construction, transform
configs before `pipeline_cache`, and distributed topology when sharding is in the cached prefix.

## Artifacts

Each rank directory contains three files:

```text
manifest.json
module.pt
hooks.json
```

`module.pt` stores the structural module snapshot.

`hooks.json` stores load-state-dict pre-hook specs that are scrubbed before `module.pt` is written
and reattached after restore.

`manifest.json` stores:

- Cache key.
- Rank.
- SHA-256 checksums for `module.pt` and `hooks.json`.

Restore only proceeds when every rank directory for the world size has all required files and the
local manifest/checksums match.

## Save Flow

On a miss, `PipelineCache._save_module()` performs:

1. Synchronize ranks.
1. Create a per-rank temporary directory.
1. Validate the module is pre-weight-loading.
1. Collect supported load hook specs.
1. Reject forward hooks and unsupported load hooks.
1. Temporarily clear load hooks from the module.
1. Write `module.pt`.
1. Restore the in-memory load hooks.
1. Write `hooks.json`.
1. Write `manifest.json` with file checksums.
1. Atomically publish the temporary rank directory.
1. Synchronize ranks.

If any rank fails to save, all ranks skip the cache entry and remove partial output.

## Restore Flow

On a hit, `PipelineCache.maybe_restore()` performs:

1. Build the expected cache context and cache key.
1. Check that every rank directory has a complete snapshot.
1. Validate local manifest and file checksums.
1. Collectively agree that all ranks can restore.
1. Load `module.pt`.
1. Load `hooks.json`.
1. Rebuild and reattach supported load hooks.
1. Collectively agree restore succeeded.
1. Return the restored module to the optimizer.

If any step fails on any rank, restore returns `None` and the normal transform pipeline runs.

## Structural Module Snapshot

Raw `GraphModule` pickling is brittle because it can capture private FX state, live node objects,
runtime-only module fields, and direct self-references. The cache therefore stores a structural
payload instead of pickling the FX graph directly.

For a direct `GraphModule`, `module.pt` stores:

- GraphModule class name.
- Sanitized GraphModule body.
- Ordered graph node state.
- Importable or literal node targets.
- Structurally encoded node args/kwargs.
- Pickleable node metadata.
- Fake tensor specs for placeholder/get_attr `meta["val"]`.
- PyTree codegen state when present.
- Rebindable GraphModule bound-method specs.

For wrapper modules, the cache:

1. Finds root `GraphModule` children.
1. Replaces each child with a temporary placeholder during `torch.save`.
1. Saves the wrapper plus structural payloads for the child graph modules.
1. Restores the original children in memory after saving.

On load, structural graph state is reconstructed into new `GraphModule` objects and inserted back
into the wrapper.

After restore, cached shape metadata tied to weight nodes is marked invalid so later transforms do
not trust stale shape-prop history.

## Hook Contract

Load hooks are not serialized inside `module.pt`. Instead, the cache serializes a small declarative
hook spec surface in `hooks.json`.

Supported hook types:

- `importable_load_hook`: importable functions, partials without positional args, and bound methods
  whose owner can be represented as JSON.
- `shard_tp`: tensor-parallel sharding hooks used by the validated sharding path.
- `dedup`: parameter deduplication hooks from export.
- `alias`: aliasing hooks from export.

Unsupported:

- Load-state-dict post hooks.
- `with_module=True` load hooks.
- Unknown marked hook specs.
- Arbitrary closures.
- Forward pre-hooks and forward hooks.

Unsupported hooks cause cache save to be skipped. This keeps the restore implementation small and
prevents false confidence from serializing hooks that are not needed by the validated cache path.

## Distributed Behavior

Each rank writes and restores its own rank-local snapshot. A cache entry is considered valid only
when all rank directories exist for the expected `world_size`.

Collective boolean checks are used for save and restore agreement:

- If any rank cannot save, no rank publishes a usable cache hit.
- If any rank cannot restore, all ranks fall back to the normal pipeline.

The cache key excludes rank but includes distributed topology when the cached prefix reaches
sharding. This allows rank-local artifacts under a shared cache entry while avoiding cross-topology
reuse.

## Failure Behavior

The cache is best-effort. Failures do not fail the model build unless they happen after normal
pipeline execution resumes.

Examples that skip cache save/restore:

- Missing files.
- Manifest mismatch.
- Checksum mismatch.
- Unsupported module payload.
- Materialized parameters.
- Unsupported hooks.
- Any rank failing distributed agreement.

The expected fallback is to run the original AutoDeploy pipeline and optionally overwrite the cache
entry with a valid snapshot on the miss path.

## Validated Coverage

The current minimal hook surface was validated by deleting the cache root and running each model
twice:

- `Qwen/Qwen3.5-35B-A3B` with world size 2.
- `google/gemma-4-26B-A4B-it`.
- `zai-org/GLM-4.7-Flash`.

The first pass created fresh cache entries. The second pass restored from those entries.

Fresh cache hook specs were:

```text
Qwen rank 0/1: importable_load_hook, shard_tp
Gemma rank 0: alias, dedup, importable_load_hook
GLM rank 0: importable_load_hook
```

The focused unit test suite also passed:

```bash
TLLM_DISABLE_MPI=1 pytest -vv tests/unittest/auto_deploy/singlegpu/transformations/test_pipeline_cache.py
```

Result:

```text
42 passed
```

## Open Items

- Decide whether to expose cache diagnostics in a structured way instead of relying on log lines.
- Decide whether future quantized/sharded paths should opt into cache support by adding explicit
  hook specs, or remain cache-miss paths.
- Consider adding a small cache inspection tool for artifact summaries and hook-spec counts.
