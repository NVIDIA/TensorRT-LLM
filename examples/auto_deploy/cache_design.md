# AutoDeploy Pipeline Cache: Pre-Weight-Loading Artifact

## Goal

Skip repeated graph-building work by restoring a previously saved pipeline
snapshot when the model, transform prefix, and execution environment match.

The AutoDeploy transform pipeline has the following stages:

```
FACTORY → EXPORT → POST_EXPORT → PATTERN_MATCHER → SHARDING
    → [CACHE BOUNDARY] →
WEIGHT_LOAD → POST_LOAD_FUSION → CACHE_INIT → COMPILE
```

The cache boundary sits between SHARDING and WEIGHT_LOAD. The cached artifact
is a **pre-weight-loading snapshot**: a fully sharded FX graph with all
parameters on the `meta` device (no real weights). On restore, the pipeline
resumes from `load_weights`, skipping the expensive factory, export, pattern
matching, and sharding stages.

The artifact is portable across machines with the same software stack (same
TensorRT-LLM version, git SHA, PyTorch/CUDA versions, checkpoint, and logical
topology). Different GPU SKUs are allowed. Runtime dimensions
(`max_batch_size`, `max_seq_len`, etc.) are not part of the cache key — the
graph is dimension-independent at this stage and mismatches are handled
gracefully on restore.

## Current Status

Current on-disk format: AD IR. Cache compatibility is determined by the
artifact contents plus the manifest's identity fields; there is no separate
manifest version field.

Implemented:

- `PipelineSnapshotManager` integrated into `InferenceOptimizer`
- Graph-only boundaries at or before the sharding stage (pre-weight-loading)
- Default boundary: `sharding_transform_executor`
- Graph mode enables pipeline cache by default via `default.yaml`
- AD IR artifact format: JSON-serialized FX graph + declarative hook specs
- Cross-machine reuse within the portability envelope
- Runtime dimension tolerance (batch size, seq len changes produce warnings, not errors)

## Why AD IR Instead of `torch.save` / Pickle

Earlier implementations saved the `GraphModule` via `torch.save`. This was
abandoned because:

- **Pickle fragility**: `GraphModule` node targets are C++ extension types
  (`torch._ops.OpOverload`, `OpOverloadPacket`) that don't reliably pickle
  across PyTorch versions.  `node.meta["val"]` contains `FakeTensor` proxy
  objects that also don't survive round-trip pickling.
- **Security**: `torch.load` uses `pickle`, which can execute arbitrary code.
  Cache files on shared filesystems are a risk.
- **Version sensitivity**: pickled objects encode exact class paths; internal
  PyTorch reorganizations break deserialization silently.
- **Debuggability**: JSON is human-readable, enabling direct inspection of
  cached graphs when diagnosing issues.

The AD IR format separates topology (JSON) from weight data (`real_buffers.pt`),
giving stable serialization without pickle.

`torch.export.save`/`torch.export.load` target `ExportedProgram`, not raw
`GraphModule`. The cached graph is mid-pipeline (still has `call_module` nodes,
custom ops, partial transformations), so it cannot be wrapped in
`ExportedProgram` without loss.

## Portability Envelope

Exact-match requirements:

- `producer_hash` (source fingerprint of prefix transforms plus core producer helpers)
- `trtllm_version`
- `repo_git_sha` (code checkout)
- `torch_version`
- `cuda_version`
- `model_identifier` and `checkpoint_fingerprint`
- `transform_prefix_hash` (ordered transform config through boundary)
- `mapping` (logical TP/PP topology)
- `world_size` and logical rank

Deliberately excluded from matching:

- hostname, PCI bus id, GPU name/SKU
- runtime-tuned allocations, compiled artifacts, device pointers

## Config Surface

Pipeline cache is configured through `PipelineCacheConfig` on `LlmArgs`.

Fields:

- `enabled`: default `false` (graph mode opts in via `default.yaml`)
- `root`: defaults to `~/.cache/tensorrt_llm/auto_deploy_pipeline`
- `boundaries`: defaults to `["sharding_transform_executor"]` (pre-weight-loading boundary)

Validation:

- duplicate boundary names are rejected
- every boundary must exist in `transforms`
- every boundary must be at or before the sharding stage (pre-weight-loading)

## Artifact Layout

```text
<cache_root>/<cache_key>/<boundary_name>/
  rank_0/
    ad_ir.json              # FX graph topology, params, buffers, hook specs, structural state
    real_buffers.pt         # non-meta buffer tensors (if any), saved via torch.save
    manifest.json           # written last, gates restore validity
  rank_1/
    ...
```

There is no separate manifest version. The AD IR schema is carried by
`ad_ir.json` itself.

### ad_ir.json

The core artifact. A structured JSON file containing:

- **nodes**: every FX graph node with `op`, `target` (string key), `args`,
  `kwargs`, and placeholder shape/dtype metadata
- **params**: parameter specs (name, shape, dtype, requires_grad)
- **buffers**: buffer specs (name, shape, dtype, is_meta flag)
- **submodule_classes**: class paths for `call_module` targets
- **hook_specs**: declarative load-hook specifications (see Hook Spec Types)
- **orig_args / in_spec / out_spec**: pytree metadata for correct
  input/output handling
- **autodeploy_meta**: structural `CachedSequenceInterface` state
  (`active_args`, `active_host_prep_args`, `use_flattened_layout`) —
  the only runtime state the graph depends on. Transform history and
  memory history are NOT cached; post-boundary transforms start fresh.
- **source_model_hooks_required**: flag for replaying HF model hooks

Serialization handles:

- `OpOverload` targets: stored as `"aten::linear.default"` (schema + overload)
- `OpOverloadPacket` targets: stored with `__packet__` marker
  (e.g. `"aten::to.__packet__"`) so the unspecialized packet is restored,
  letting PyTorch dispatch to the correct overload at runtime
- Python builtins: `operator.getitem` → `"operator.getitem"`, etc.
- General callables: `"module.qualname"` resolved via `importlib`

On restore, `build_graph_module` reconstructs the `torch.fx.Graph` node-by-node
via the Graph construction API (no re-tracing), then `hydrate_shapes` runs
`FakeTensorProp` to populate `node.meta["val"]` shape metadata from the
stored placeholder shapes.

### real_buffers.pt

Non-meta buffer tensors saved via `torch.save`. Only written when the graph
has buffers with real (non-meta) data. Most cached graphs have all parameters
and buffers on the `meta` device, so this file is often absent.

### manifest.json

Written last. Contains cache key, portability envelope fields, format version,
hook counts, and the `has_unserializable_hooks` flag. Restore checks this file
first; if it's missing or invalid, the boundary is treated as a cache miss.

## HookSpec Types

Hook specs are embedded in `ad_ir.json` under the `hook_specs` key.

### shard_tp

TP sharding hook (`_load_hook` with `_split_tensor_for_tp`).

Fields: `param_key`, `param_shape`, `dim`, `rank`, `world_size`,
`min_local_shape`, `fused_weight_dims`.

### shard_slice

BMM batch-dim sharding hook (`_load_hook` with a slice function).

Fields: `param_key`, `param_shape`, `start_idx`, `end_idx`.

### shard_quant

Quantization-aware sharding hooks (FP8, FineGrainedFP8, FP4).

Fields: `quant_class`, `weight_name`, `weight_original_shape`, `dim`,
`rank`, `world_size`, `min_local_shape`, `fused_weight_dims`.

### dedup

Parameter deduplication hook (`_load_hook_for_deduplication`).

Fields: `param_key_remaining`, `param_key_removed`.

### alias

Aliased-parameter broadcast hook.

Fields: `aliased_groups` (list of lists of parameter names).

### remove

Bias removal hook (`_load_hook_remove`).

Fields: `param_key`.

### source_model hooks

Hooks from the original HF model (including `@staticmethod` hooks) are not
individually serialized. Instead the IR records
`"source_model_hooks_required": true` and restore replays
`_add_missing_load_hooks(gm, factory.build_model("meta"))`.

## Hook Identification

Hooks are identified at save time by pattern-matching on `hook.func.__name__`
and `hook.keywords` for `partial`-based hooks, and `hook.__qualname__` plus
closure cell inspection for closure-based hooks. Both bound methods and
`@staticmethod` hooks on non-root submodules are recognized as source-model
hooks. If a hook cannot be identified, it is logged as a warning and the
manifest is flagged with `"has_unserializable_hooks": true`, which prevents
restore.

## Save Flow

1. Guard: enabled, boundary match, stage \<= SHARDING, mod is GraphModule
1. `barrier()`
1. `mkdir` rank directory
1. `collect_hook_specs(mod)` — introspect all load hooks
1. Embed structural `CachedSequenceInterface` state in `autodeploy_meta`
1. `extract_ir(mod, hook_specs, autodeploy_meta)` — serialize graph to `IRGraph`
1. `save_ir(ir, real_buffers, rank_dir)` — write `ad_ir.json` + `real_buffers.pt`
1. Write `manifest.json` last (gates restore validity)
1. On any exception: `shutil.rmtree(rank_dir)` + warn

## Restore Flow

1. Scan boundaries high-to-low by transform_index
1. Check all ranks have `manifest.json` (collective check)
1. Validate manifest (strict matching of identity fields)
1. Reject if `has_unserializable_hooks` is true
1. `load_ir(rank_dir)` → `(IRGraph, real_buffers)`
1. `build_graph_module(ir, real_buffers)` → reconstruct `GraphModule`
1. `hydrate_shapes(gm, ir)` → run `FakeTensorProp` with stored placeholder shapes
1. `reattach_hooks(gm, ir.hook_specs)` — rebuild live hooks from specs
1. If `source_model_hooks_required`: replay via factory
1. Restore autodeploy_meta and structural `CachedSequenceInterface` state
1. Return `(mod, next_transform_index)`

## AD IR Serialization Details

### Target Serialization (`serialize_target`)

| Python type | Serialized form | Example |
|---|---|---|
| `OpOverload` | `schema_name.overload` | `"aten::linear.default"` |
| `OpOverloadPacket` | `qualified_op_name.__packet__` | `"aten::to.__packet__"` |
| `operator.getitem` | `"operator.getitem"` | |
| `builtins.getattr` | `"builtins.getattr"` | |
| Other callable | `"module.qualname"` | `"torch.relu"` |

### Target Deserialization (`resolve_target`)

- `__packet__` suffix → return the `OpOverloadPacket` directly (PyTorch
  dispatches to correct overload based on call-site arguments)
- `::` in key → resolve via `torch.ops.namespace.op_name.overload`; falls back
  to the packet if the overload doesn't exist
- Otherwise → `importlib` resolution

### Graph Reconstruction (`build_graph_module`)

Node names are reconstructed exactly:

1. Pre-register all IR node names in the graph namespace (prevents auto-generated
   names like `getitem_215` from colliding with IR names)
1. Before creating each node, temporarily remove its name from `_used_names`
   so that `create_name` returns the exact name (prevents `_1` suffixes)
1. Nodes are created via `graph.create_node()` with explicit `name=` arguments

## Cache Key and Invalidation

Cache key inputs:

- boundary name, model identifier, checkpoint fingerprint, factory type
- model_kwargs, transform-prefix hash, producer hash, mapping payload
- TensorRT-LLM version, repo git SHA, torch version, CUDA version, world size

Manifest strict matching checks:

- `boundary_name`, `boundary_class`
- `cache_key`, `transform_prefix_hash`, `producer_hash`
- `world_size`, `rank`, `trtllm_version`, `torch_version`, `cuda_version`,
  `repo_git_sha`
- `has_unserializable_hooks` must be false

Invalidation:

- missing rank manifest invalidates the boundary
- invalid JSON or IR load failure invalidates the boundary
- save failure removes the partial rank directory

## Testing

Unit coverage:

- AD IR round trip (serialize → deserialize → compare graph structure)
- hook round-trip (dedup, alias, remove, shard_tp)
- hook specs embedded in IR JSON
- `OpOverloadPacket` serialization with `__packet__` marker
- `OpOverload` serialization with schema + overload
- node name preservation through graph reconstruction
- `hydrate_shapes` with placeholder shape/dtype metadata
- full save/restore with hooks end-to-end
- manifest metadata fields
- extra manifest fields are ignored
- producer-hash invalidation when pre-boundary transform code changes
- config validation for supported boundaries
- cache-key canonicalization for Mapping/MpiTopology objects
- graceful fallback when save fails
- static method hooks on source models correctly identified

## Future: Post-Fusion Caching via Weight Transformation Recipes

Post-load fusions (e.g. `fuse_moe`) cannot be cached without weight
serialization because they restructure weight tensors. A future extension
will store a declarative weight transformation recipe — a log of what each
fusion did to the weights. On restore, `load_weights` runs normally, then
the recipe replays fusion-time weight manipulations:

```json
[
  {"op": "stack", "sources": ["experts.0.w1", "..."], "target": "batched.w1", "dim": 0},
  {"op": "fuse_gate_up", "gate": "gate_proj.weight", "up": "up_proj.weight", "target": "gate_up.weight"}
]
```

This is out of scope for the current version and will be designed separately.
