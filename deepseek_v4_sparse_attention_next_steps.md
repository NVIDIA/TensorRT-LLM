<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# DeepSeek V4 Sparse Attention Next Steps

This note captures the current end-to-end AutoDeploy flow for
`torch_deepseek_v4_sparse_attention`, the exact boundary between what works now
and what is still missing, and a concrete plan for making DeepSeek V4 sparse
attention serve correctly with chunked prefill, decode, paged caches, and CUDA
graphs.

The current branch state supports a full-context/prefill-style model forward
through a DeepSeek V4 source attention op. It does not yet implement the cached
decode attention path needed for normal token-by-token serving.

## Current End-To-End Flow

### 1. Model Registry And Compile Path

The model registry config selects:

```yaml
runtime: trtllm
model_factory: DeepseekV4AutoModelForCausalLM
compile_backend: torch-cudagraph
max_batch_size: 64
max_seq_len: 8192
max_num_tokens: 8192
enable_chunked_prefill: true
model_kwargs:
  skip_mtp: true
  ad_rope_cache_len: 8192
```

The same config enables quantization and MoE lowering, but it does not select a
DeepSeek V4 cached attention backend. Default AutoDeploy cache insertion still
targets standard `torch_attention` and standard `torch_mla` source ops.

### 2. Model Forward Contract

`DeepseekV4ForCausalLM.forward` requires both `input_ids` and `position_ids`.
It embeds the full current token tensor, runs every layer, and returns logits for
all positions.

The model file explicitly documents the current scope:

```text
* prefill-only forward with mandatory position_ids
* omits decode caches and MTP blocks from the exported path
```

This is the first important boundary: the model can export and run a
full-context forward graph, but it is not yet a cache-aware incremental decode
model.

### 3. DeepSeek V4 Attention Forward

`DeepseekV4Attention.forward` performs these steps:

1. Build query states:
   - `wq_a`
   - `q_norm`
   - `wq_b`
   - per-head RMS normalization
   - RoPE on the trailing `qk_rope_head_dim`

2. Build high-resolution KV rows for the current forward tensor:
   - `wkv`
   - `kv_norm`
   - RoPE on the trailing `qk_rope_head_dim`

3. Build local-window indices:
   - `_window_topk_idxs(window_size, batch_size, seq_len, device)`
   - negative entries are padding/masked slots

4. For compressed layers, build compressed KV rows from the current forward
   tensor:
   - `DeepseekV4Compressor.forward`
   - `wkv`, `wgate`, APE, optional ratio-4 overlap transform
   - softmax pooling
   - RMSNorm
   - compressed-position RoPE

5. For compressed layers, append compressed rows:
   - `_compress_topk_idxs(...)`
   - concatenate local-window and compressed indices
   - concatenate high-resolution KV and compressed KV rows

6. Call the source attention op:

```python
torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
    q, kv, attn_sink, topk_idxs, softmax_scale
)
```

7. Apply inverse RoPE to the attention output, then run the `wo_a`/`wo_b`
   output projection path.

### 4. Source Op Semantics

The source op is defined as:

```python
@torch.library.custom_op("auto_deploy::torch_deepseek_v4_sparse_attention", mutates_args=())
def torch_deepseek_v4_sparse_attention(
    q,
    kv,
    attn_sink,
    topk_idxs,
    softmax_scale,
    out=None,
    enable_sharding=False,
    layer_type="unknown",
):
    ...
```

Important properties:

- `q` shape: `[batch, seq_len, num_heads, head_dim]`
- `kv` shape: `[batch, kv_rows, head_dim]`
- `attn_sink` shape: `[num_heads]`
- `topk_idxs` shape: `[batch, seq_len, k_select]`
- duplicate indices receive independent probability mass
- negative indices are masked before softmax
- the sink participates in softmax normalization but contributes no value vector
- `mutates_args=()` means it does not write any cache
- it has an optional `out=` contract for piecewise CUDA graph output injection

The tests cover output shape/dtype, duplicate index semantics, negative masks,
sink behavior, dynamic export, and the `out` buffer contract.

### 5. Existing Kernel Building Blocks

`deepseek_v4_kernels.py` contains reference-first semantic ops that look like
building blocks for a production cached path:

- `torch_deepseek_v4_q_rmsnorm_rope`
- `torch_deepseek_v4_kv_rmsnorm_rope_cache_insert`
- `torch_deepseek_v4_compressor_pool_norm_rope`
- `torch_deepseek_v4_indexer_q_rope_quant`
- `torch_deepseek_v4_inverse_rope_fp8_output_quant`
- `torch_deepseek_v4_sparse_attention_microkernel`

The most relevant one for decode is
`torch_deepseek_v4_kv_rmsnorm_rope_cache_insert`, which mutates flat
`nope_cache`, `rope_cache`, and `scale_cache` tensors using caller-supplied
`cache_indices`. Its own docstring says the current flat index contract
deliberately does not encode V4 paged-cache metadata.

The microkernel op currently assembles:

```text
local window KV + compressed KV + local/compressed topk indices
```

and then calls the same source op. It is a useful correctness wrapper, not yet a
paged-cache decode kernel.

### 6. Current Sharding Support

`DeepSeekV4SparseAttentionShardableNode` registers the source op as shardable.
The current rule shards `attn_sink` across tensor-parallel heads when
`enable_sharding` is set and `tp_size > 1`.

This is useful for the full-context source op. A future cached op will need its
own sharding rules for:

- local head ranges
- local `attn_sink`
- local or replicated cache metadata
- local slices of cache rows
- any all-reduce or gather needed after `wo_b`

## Existing Cache Resource Plumbing

The branch adds named DeepSeek V4 resource names:

```text
swa
mhc
indexer
compressor_state
```

The intended meanings are:

| Resource | Intended role |
| --- | --- |
| `swa` | high-resolution sliding-window KV |
| `mhc` | compressed KV entries for ratio-4 and ratio-128 layers |
| `indexer` | compact indexer/top-k selection state |
| `compressor_state` | per-sequence state for incomplete compression windows |

`DeepSeekV4PagedResourceHandler` reports `is_paged=True`, registers named page
table metadata, and allocates a local paged pool with shape:

```text
[num_pages, tokens_per_block, *token_shape]
```

`SequenceInfo.register_paged_resource_metadata(name, ...)` creates:

```text
<name>_cache_loc
<name>_cu_num_pages
<name>_last_page_len
<name>_seq_len_with_cache
```

The tests verify:

- V4 resources are paged, not unpaged fallback resources.
- SWA and MHC can have different logical lengths and page counts.
- ratio-128 compressed metadata uses fewer logical entries than token cache.
- prefix/block reuse remains enabled.
- `copy_on_partial_reuse` is disabled when non-KV-manager-owned paged resources
  exist.

This is necessary infrastructure, but it is not enough for decode by itself.

## Why Decode Is Missing Today

### 1. The Current Source Op Is Stateless

The source op takes `q`, `kv`, `attn_sink`, and `topk_idxs`. It does not take any
of:

```text
cache_loc
cu_num_pages
last_page_len
seq_len_with_cache
swa_cache
mhc_cache
indexer_cache
compressor_state
```

It also declares `mutates_args=()`, so it cannot update cache state.

During token-by-token decode, AutoDeploy normally passes only the new token
through the model and relies on cached attention to read prior context. If the
DeepSeek V4 source op sees only `seq_len == 1`, it only has the current token's
KV rows and cannot attend to earlier tokens.

### 2. Cache Insertion Cannot Find A DeepSeek V4 Backend

The generic cache insertion transform resolves an `AttentionDescriptor` through
`AttentionRegistry.get(self.config.backend)`. Existing registered backends
target source ops such as:

```text
auto_deploy::torch_attention
auto_deploy::torch_mla
```

There is no registered DeepSeek V4 sparse attention descriptor in the current
branch. A repository search finds no:

```text
@AttentionRegistry.register("deepseek...")
DeepSeek...Attention(AttentionDescriptor)
```

As a result, default `insert_cached_attention` and `insert_cached_mla_attention`
do not rewrite `torch_deepseek_v4_sparse_attention` into a cached op.

### 3. The Cached Op Name Exists Only As A Placeholder String

`piecewise_utils.py` lists:

```text
auto_deploy::triton_deepseek_v4_sparse_attention_with_cache
```

as a dynamic DeepSeek V4 attention op, but there is no implementation of that
custom op in the branch. That string is useful as a future compile partitioning
hook, but it is not an executable decode path.

### 4. Named Page Metadata Is Staged, But Not Advanced For Decode

`SequenceInfo.nest_sequences(..., paged_resource_metadata=...)` can stage named
page tables for `swa`, `mhc`, `indexer`, and `compressor_state`.

However, the decode offset helpers currently update the standard page metadata
path. They do not yet loop over all registered named resource metadata to
advance:

```text
swa_cache_loc
swa_cu_num_pages
swa_last_page_len
swa_seq_len_with_cache
...
```

That means a production decode path needs either:

1. fresh named page metadata supplied by the scheduler for every step, or
2. extended `SequenceInfo` offset/adjust logic for named paged resources.

### 5. Resource Layout Is Still Synthetic

The cache resource tests intentionally use synthetic shapes such as:

```python
swa: token_shape=(2, 1, 8), dtype=torch.float16
mhc: token_shape=(1, 8), dtype=torch.bfloat16
indexer: token_shape=(1, 4), dtype=torch.float16
compressor_state: token_shape=(4, 8), dtype=torch.bfloat16
```

Production DeepSeek V4 attention likely needs split or composite layouts,
especially for FP8 cache insertion:

```text
NoPE cache: FP8 E4M3 rows
RoPE cache: BF16/FP16 rows
Scale cache: E8M0 or FP32 fallback rows
```

The existing `torch_deepseek_v4_kv_rmsnorm_rope_cache_insert` already uses this
three-cache contract. The named resource abstraction needs to converge with
that contract before the cached attention op can be production-real.

### 6. Compression And Indexing Are Not Incremental

Current full-context model code computes compressed KV from the full current
`x` tensor. During decode, the compressor needs to update state incrementally.

Missing semantics include:

- how ratio-128 compressed entries are emitted when a compression block closes
- how ratio-4 overlap state is preserved across step boundaries
- how `compressor_state` stores incomplete windows
- how `mhc_cache` appends new compressed rows
- how the `indexer_cache` participates in top-k compressed row selection
- how local-window and compressed indices are built from paged metadata

Current `_compress_topk_idxs` is a full-context construction over rows already
present in the current `kv` tensor. It is not a paged-cache lookup plan.

## Required Implementation Pieces

### Piece 1: DeepSeek V4 Cached Attention Descriptor

Add an `AttentionDescriptor` implementation, for example:

```python
@AttentionRegistry.register("deepseek_v4_sparse")
class DeepSeekV4SparseAttention(AttentionDescriptor):
    ...
```

It should define:

- `get_source_attention_op()` returning
  `torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention`
- `get_cached_attention_op()` returning the new cached op
- `get_num_qkv_args()` for the source op contract
- `get_standard_metadata_args()` for sequence metadata needed by the cached op
- `get_cache_initializers()` returning DeepSeek V4 resource handlers
- constants or node metadata for layer index, window size, compression ratio,
  head dimensions, RoPE dimensions, cache dtypes, and FP8 block sizes

The deployment YAML should explicitly select this backend for DeepSeek V4
rather than relying on the default `trtllm` or `flashinfer_mla` attention cache
backends.

### Piece 2: Better Source Op Metadata

The current source op signature does not carry enough information for a cache
descriptor to allocate and lower correctly. Add explicit metadata to the source
op or attach reliable node metadata during export.

Recommended explicit arguments:

```text
layer_idx
window_size
compress_ratio
max_compressed_len
head_dim
rope_dim
num_heads
fp8_block_size
cache_dtype policy
```

This avoids fragile inference from fake tensor shapes and lets
`get_cache_initializers()` make per-layer decisions.

### Piece 3: Cached Op Contract

Define a first reference op, then a Triton/CUDA implementation. A likely
progression is:

```text
torch_deepseek_v4_sparse_attention_with_cache
triton_deepseek_v4_sparse_attention_with_cache
```

The cached op should:

1. accept current-token or current-chunk query/KV inputs
2. accept standard sequence metadata
3. accept named V4 page metadata
4. accept V4 cache tensors
5. update the cache tensors for new tokens/compressed rows
6. gather the local-window and compressed rows needed for attention
7. apply the sink-softmax sparse attention
8. return attention output
9. support an optional `out=` contract for piecewise CUDA graph integration

Start with a pure Torch reference op even if it is slow. That gives
AutoDeploy graph replacement, cache metadata, and correctness tests something
solid to target before optimizing kernels.

### Piece 4: Production Cache Layout

Replace synthetic cache shapes with an explicit production layout.

Open decisions:

- Should `swa` be one composite resource or separate resources for NoPE, RoPE,
  and scale?
- Should `mhc` use the same FP8 split layout as `swa`?
- Does `indexer` store FP8-quantized indexer keys, scales, or both?
- Is `compressor_state` one per sequence or one per layer/sequence?
- Does each layer own separate resources, or can compatible layers share a
  manager/pool with per-layer offsets?

The existing `DeepSeekV4CacheResourceDescriptor` supports one tensor per
resource descriptor. If production needs multiple tensors per logical resource,
add either:

- a composite resource handler, or
- separate stable resource names/suffixes for subresources.

### Piece 5: Named Page Allocation And Reuse

Decide how page IDs are assigned and reused for V4 resources.

The tests manually build `PagedResourceSequenceMetadata`. Production needs a
real source of these page assignments:

- reuse KVCacheManager page assignments where possible, or
- add a small V4 page allocator, or
- derive deterministic per-slot pages for the first bring-up and disable
  prefix-sharing for V4 side caches until correctness is established.

The final design must define behavior for:

- request admission
- prefix reuse
- partial reuse
- page allocation
- page free
- cache compaction, if any
- cuda graph padding slots
- multi-rank consistency

### Piece 6: Named Metadata Advancement

Extend `SequenceInfo`/`CachedSequenceInterface` so registered named paged
resources can advance during decode.

Required behavior:

- update each named `seq_len_with_cache`
- update each named `last_page_len`
- add pages when a named resource crosses its `tokens_per_block`
- respect different logical-length divisors per resource
- handle resources that do not advance every token, such as ratio-128 MHC
- handle `compressor_state`, which is per-sequence state rather than ordinary
  append-only token KV

An alternative is requiring the scheduler to provide fresh named metadata every
step. That may be simpler initially, but it has to be explicit and tested.

### Piece 7: Incremental Compression Semantics

Implement and test the decode update rules independently from attention.

Suggested staged scope:

1. Ratio-0 layers:
   - update `swa`
   - attend over local sliding window only

2. Ratio-128 layers:
   - accumulate compressor state
   - emit one compressed row when a block closes
   - append to `mhc`
   - attend over SWA plus compressed rows

3. Ratio-4 layers:
   - implement overlap handling across step boundaries
   - maintain `compressor_state`
   - build/update indexer cache
   - implement top-k compressed selection

### Piece 8: CUDA Graph Integration

The source op already has an `out=` contract. The cached op needs the same
contract.

For CUDA graph support:

- decode-only should be capturable with static batch sizes
- prefill/mixed batches may need piecewise mode because sparse/cache ops are
  dynamic
- `piecewise_utils.py` already anticipates a DSV4 cached op name, but the op
  must actually exist
- the DeepSeek V4 YAML currently sets `compile_model.piecewise_enabled: false`,
  so mixed/prefill piecewise capture is not part of the current default path

### Piece 9: Sharding Rules For Cached Attention

Add shardable-node support for the cached op.

The rule should validate and/or rewrite:

- local head count
- local `attn_sink`
- local cache row layout
- page metadata visibility across TP/EP ranks
- output shape before grouped `wo_a`

The current source-op sharding rule only shards `attn_sink`.

## Suggested Milestones

### Milestone A: Make The Gap Impossible To Miss

- Add a test or config guard that fails clearly if DeepSeek V4 is used for
  decode without a cached sparse attention backend.
- Update the branch summary/config comments to say "prefill/full-forward only"
  until cached decode lands.

### Milestone B: Ratio-0 Cached Reference Path

- Add `DeepSeekV4SparseAttention` descriptor.
- Add `torch_deepseek_v4_sparse_attention_with_cache`.
- Implement only SWA/local-window cache.
- Prove:
  - full prefill logits equal prefix prefill plus decode logits
  - cache metadata is staged correctly
  - CUDA graph decode capture works for fixed batch sizes

### Milestone C: Ratio-128 Compressed Cache

- Add incremental compressor state for ratio-128.
- Append completed compressed rows to MHC.
- Prove chunked prefill equivalence against full prefill.
- Prove decode equivalence after a prefix.

### Milestone D: Ratio-4 And Indexer Path

- Add overlap compressor state.
- Add indexer cache update.
- Add top-k compressed-row selection.
- Prove correctness on synthetic cases with forced top-k selections, then on
  model-level tests.

### Milestone E: Optimized Triton/CUDA Kernels

- Replace the Torch reference cached op with Triton/CUDA kernels.
- Keep the same op contract and tests.
- Add performance counters for:
  - local-window gather
  - compressed gather
  - sink softmax
  - cache insert bandwidth
  - FP8 dequant/scale overhead

### Milestone F: Full Serving Validation

- Run full-checkpoint AutoDeploy load.
- Run prefill-only correctness first.
- Run chunked prefill.
- Run decode generation.
- Run CUDA graph decode.
- Run multi-rank sharding.
- Benchmark serving throughput/latency and memory.

## Test Plan

### Unit Tests

- Descriptor registration:
  - `AttentionRegistry.has("deepseek_v4_sparse")`
  - source op maps to cached op
  - cache initializers include expected V4 resources

- Cache metadata:
  - named page tables for all resources
  - different logical divisors
  - decode advancement
  - ratio-specific page creation
  - prefix reuse behavior

- Cached op reference:
  - ratio-0 local window
  - ratio-128 compressed rows
  - ratio-4 overlap
  - masked and duplicate indices
  - sink probability mass
  - `out=` buffer contract

### Model-Level Tests

- Full prefill vs chunked prefill logits.
- Full prefill vs prefix prefill plus decode logits.
- Layer subsets:
  - all ratio-0
  - ratio-128 only
  - ratio-4 only
  - mixed ratios matching the real config pattern

### Runtime Tests

- AutoDeploy transform replaces source op with cached op when backend is
  `deepseek_v4_sparse`.
- `initialize_cache` creates all expected resources.
- Decode CUDA graph capture works for configured batch sizes.
- Piecewise path handles DSV4 dynamic op if enabled.

### Integration Tests

- Run the downloaded checkpoint:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev/hf_home/manual/deepseek-ai__DeepSeek-V4-Flash
```

- Serve with the DeepSeek V4 registry config.
- Compare generated tokens/logits against a trusted reference for short
  prompts.
- Stress long prompts, chunked prefill, and decode continuation.

All commands in this workspace should be run as:

```bash
bash -ic "f9 && <command>"
```

## Bottom Line

The current branch has a real DeepSeek V4 sparse attention source op and can run
that op in the exported AutoDeploy forward graph. The missing work is the
cache-aware decode path: descriptor, cached op, production cache layouts,
metadata advancement, incremental compression/indexing, sharding for the cached
op, and CUDA graph validation.
