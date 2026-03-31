# Porting Models to Sharding-Aware Custom Ops

## Purpose

This document provides step-by-step instructions for porting an existing
AutoDeploy custom model (`models/custom/modeling_*.py`) to use explicit
sharding hint ops (`models/custom/new_sharding/modeling_*.py`).

The result is a model whose FX graph is a complete, self-contained specification
of "how this model should be sharded." The `apply_sharding_hints` transform
reads the hints together with a runtime `DistConfig` to apply deterministic,
node-local sharding -- no pattern matching, no heuristics.

## Reference Examples

Study these before porting a new model:

| Original | Sharded | Layer types |
|----------|---------|-------------|
| `modeling_nemotron_h.py` | `new_sharding/modeling_nemotron_h.py` | Mamba SSM, MHA, SwiGLU MLP, MoE |
| `modeling_qwen3_5_moe.py` | `new_sharding/modeling_qwen3_5_moe.py` | GatedDeltaNet, Gated MHA, SwiGLU MLP, MoE |
| `modeling_mistral.py` | `new_sharding/modeling_mistral.py` | MHA, SwiGLU MLP (simplest) |
| `modeling_deepseek_v2.py` | `new_sharding/modeling_deepseek_v2.py` | MLA, SwiGLU MLP, MoE |

______________________________________________________________________

## Available Sharding-Aware Custom Ops

Every sharding-aware op accepts a `layer_type: str = "unknown"` as its LAST
parameter. This tag classifies the op for selective layer sharding via
`config.shard_layers`.

| Op | Sharding hints | When to use |
|----|---------------|-------------|
| `torch.ops.auto_deploy.torch_linear_simple` | `tp_mode`, `output_sizes`, `tp_min_local_shape`, `layer_type` | Replace every `nn.Linear` / `self.proj(x)` call |
| `torch.ops.auto_deploy.view` | `tp_scaled_dim`, `layer_type` | Replace `.view()` / `.reshape()` where a dimension contains a concrete head count that scales with TP |
| `torch.ops.auto_deploy.split_with_sizes` | `shardable`, `layer_type` | Replace `torch.split` / `torch.split_with_sizes` after a colwise-sharded projection |
| `torch.ops.auto_deploy.all_reduce` | `layer_type` | Insert after every rowwise projection. Identity when `world_size=1`; real `dist.all_reduce` when sharded |
| `torch.ops.auto_deploy.torch_causal_conv1d` | `shardable`, `output_sizes`, `layer_type` | Already used in model code; add sharding hints |
| `torch.ops.auto_deploy.torch_ssm` | `shardable`, `layer_type` | Already used in Mamba models; add sharding hint |
| `torch.ops.auto_deploy.torch_gated_delta_rule` | `shardable`, `layer_type` | Already used in GatedDeltaNet models; add sharding hint |
| `torch.ops.auto_deploy.torch_rmsnorm_gated` | `tp_mode`, `layer_type` | Gated RMSNorm whose weight scales with TP (e.g., Mamba norm) |
| `torch.ops.auto_deploy.torch_mla` | `shardable`, `layer_type` | MLA attention op; when shardable, `_apply_hint_mla` shards `kv_b_proj_weight` (arg\[4\]) colwise |
| `torch.ops.auto_deploy.torch_moe` | `layer_type` | Already used in MoE models; `apply_sharding_hints` handles EP/TP automatically |

### Hint parameter reference

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `tp_mode` | `str` | `"none"` | `"colwise"` = shard weight dim 0. `"rowwise"` = shard weight dim 1. `"none"` = no sharding. |
| `output_sizes` | `Optional[List[int]]` | `None` | Fused weight group sizes for proportional column sharding (e.g., `[key_dim, key_dim, value_dim]`). |
| `tp_min_local_shape` | `int` | `1` | Minimum output size per rank. Used for GQA where `num_kv_heads < tp_size` (set to `head_dim`). |
| `tp_scaled_dim` | `int` | `-1` | Index of the shape dimension that scales with TP. `-1` means no scaling. `apply_sharding_hints` replaces `shape[tp_scaled_dim]` with `-1` (inferred). |
| `shardable` | `bool` | `False` | When True, `apply_sharding_hints` shards the op's weights/parameters along the head dimension. |
| `layer_type` | `str` | `"unknown"` | Layer classification for selective sharding. Must match `LayerType` enum values: `"mha"`, `"mla"`, `"ssm"`, `"delta"`, `"mlp"`, `"moe"`, `"unknown"`. |

### Layer type values

| Value | When to use |
|-------|------------|
| `"mha"` | Standard multi-head attention: Q/K/V/O projections, head reshape views, all_reduce |
| `"mla"` | Multi-head latent attention (DeepSeek): q_a/q_b/kv_a projections, torch_mla, o_proj, all_reduce |
| `"mlp"` | SwiGLU MLP: gate/up/down projections, all_reduce |
| `"moe"` | MoE block: torch_moe, shared expert projections, merge all_reduce |
| `"ssm"` | Mamba SSM: in_proj, conv1d, torch_ssm, norm, out_proj, views, splits, all_reduce |
| `"delta"` | GatedDeltaNet: in_proj_qkv/z/b/a, conv1d, torch_gated_delta_rule, views, splits, out_proj, all_reduce |
| `"unknown"` | Default. Nodes with this value are skipped when `shard_layers` is set. |

______________________________________________________________________

## Step-by-Step Porting Procedure

### Step 1: Copy the source file

```
cp models/custom/modeling_foo.py models/custom/new_sharding/modeling_foo.py
```

### Step 2: Update the module docstring and add imports

Add at the top of the file:

```python
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 -- register all ops
```

Do NOT add global `SHARD_*` flags. Layer-level sharding control is handled
by the `layer_type` hint on each op and the `shard_layers` config in YAML.

### Step 3: Replace linear projections

For every `self.proj(x)` or `nn.Linear` call in the forward methods:

```python
# BEFORE
output = self.proj(hidden_states)

# AFTER
output = torch.ops.auto_deploy.torch_linear_simple(
    hidden_states,
    self.proj.weight,
    self.proj.bias,              # None if no bias
    tp_mode="colwise",           # or "rowwise"
    layer_type="mha",            # or "mlp", "moe", "ssm", "delta", "mla"
)
```

Always set `tp_mode` unconditionally (no `if _s else "none"` conditionals).
The `layer_type` tag enables selective sharding at the transform level via
`config.shard_layers`.

**Rules for choosing `tp_mode`:**

- **Opening projections** (Q, K, V, gate, up, in_proj) -> `"colwise"`
- **Closing projections** (O, down, out_proj) -> `"rowwise"`
- **Tiny projections** (shared_expert_gate with output_dim=1) -> `"none"` (cannot shard)
- **Latent projections** (MLA q_a_proj, kv_a_proj) -> `"none"` (replicated)

**Fused weights** -- when a single linear produces concatenated outputs that are
later split (e.g., QKV fused, or Mamba in_proj = \[gate | conv_input | dt\]):

```python
output = torch.ops.auto_deploy.torch_linear_simple(
    x, self.in_proj.weight, self.in_proj.bias,
    tp_mode="colwise",
    output_sizes=[gate_dim, conv_dim, dt_dim],
    layer_type="ssm",
)
```

**GQA (num_kv_heads \< num_q_heads)** -- K/V projections need:

```python
key = torch.ops.auto_deploy.torch_linear_simple(
    x, self.k_proj.weight, self.k_proj.bias,
    tp_mode="colwise",
    tp_min_local_shape=self.head_dim,
    layer_type="mha",
)
```

### Step 4: Replace split / chunk operations

After a colwise-sharded fused projection, replace `torch.split` with:

```python
gate, up = torch.ops.auto_deploy.split_with_sizes(
    projected, [gate_dim, up_dim], dim=-1,
    shardable=True,
    layer_type="ssm",
)
```

### Step 5: Replace view / reshape with concrete head counts

**Critical rule**: during `torch.export`, every `-1` in `.view()` / `.reshape()`
gets concretized to a concrete integer. After TP sharding changes tensor sizes,
these concrete values become wrong. Any reshape dimension that scales with TP
**must** use `auto_deploy::view`.

```python
key = torch.ops.auto_deploy.view(
    key_proj_output,
    [bsz, seq_len, self.num_kv_heads, self.head_dim],
    tp_scaled_dim=2,
    layer_type="mha",
)
```

**When NOT to use `auto_deploy::view`:**

- Flattening to 2D: `x.reshape(-1, x.shape[-1])` -- safe, no head count
- Flattening heads back: `x.reshape(bsz, seq_len, -1)` -- safe IF the input
  tensor already has the correct sharded shape

**When you MUST use `auto_deploy::view`:**

- Any reshape with a concrete `num_heads`, `num_kv_heads`, `num_v_heads` at
  position 2 (or any other position that scales with TP)
- Reshapes after norm that restore a 4D `[B, S, H, D]` shape

### Step 6: Insert all_reduce

After every rowwise projection, add unconditionally:

```python
output = torch.ops.auto_deploy.torch_linear_simple(
    x, self.out_proj.weight, self.out_proj.bias,
    tp_mode="rowwise",
    layer_type="mha",
)
output = torch.ops.auto_deploy.all_reduce(output, layer_type="mha")
```

**Parallel branch rule**: when two or more sharded branches merge by addition,
place a **single** `all_reduce` after the merge point instead of one per
branch. This is valid because all_reduce (sum across ranks) distributes over
addition: `all_reduce(A) + all_reduce(B) == all_reduce(A + B)`.

How to apply:

1. Identify parallel branches that each end with a rowwise projection
1. Suppress `all_reduce` inside each individual branch
1. Sum the branch outputs, then insert one `all_reduce` on the sum

**Example 1 — MoE with shared expert** (shared + routed branches):

```python
class SharedExpertMLP(nn.Module):
    def __init__(self, ..., add_all_reduce=True):
        self.add_all_reduce = add_all_reduce

    def forward(self, x):
        ...  # gate/up colwise, down rowwise
        if self.add_all_reduce:
            down = torch.ops.auto_deploy.all_reduce(down, layer_type="mlp")
        return down

class MoEBlock(nn.Module):
    def __init__(self, ...):
        self.shared_expert = SharedExpertMLP(..., add_all_reduce=False)

    def forward(self, x):
        routed_out = torch.ops.auto_deploy.torch_moe(..., layer_type="moe")
        shared_out = self.shared_expert(x)
        out = routed_out + shared_out
        out = torch.ops.auto_deploy.all_reduce(out, layer_type="moe")
        return out
```

**Example 2 — Parallel attention + MLP** (e.g., Cohere):

```python
class ParallelDecoderLayer(nn.Module):
    def forward(self, hidden_states, ...):
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)

        attn_out = self.self_attn(normed, ...)  # o_proj rowwise, NO all_reduce
        mlp_out = self.mlp(normed)              # down_proj rowwise, NO all_reduce

        out = attn_out + mlp_out
        out = torch.ops.auto_deploy.all_reduce(out, layer_type="mha")
        hidden_states = residual + out
        return hidden_states
```

### Step 7: Handle special layer-specific ops

**Conv1d** (Mamba, GatedDeltaNet):

```python
conv_out = torch.ops.auto_deploy.torch_causal_conv1d(
    x, self.conv1d.weight, self.conv1d.bias,
    self.conv1d.stride[0], self.conv1d.padding[0],
    self.conv1d.dilation[0], self.conv1d.groups,
    self.conv1d.padding_mode,
    shardable=True,
    output_sizes=[key_dim, key_dim, value_dim],
    layer_type="delta",
)
```

**SSM** (Mamba):

```python
y = torch.ops.auto_deploy.torch_ssm(..., shardable=True, layer_type="ssm")
```

**GatedDeltaNet**:

```python
out = torch.ops.auto_deploy.torch_gated_delta_rule(
    query, key, value, a, b, self.A_log, self.dt_bias,
    shardable=True, layer_type="delta",
)
```

**Gated RMSNorm** (Mamba):

```python
out = torch.ops.auto_deploy.torch_rmsnorm_gated(
    x, self.norm.weight, gate, self.norm.eps, group_size,
    tp_mode="colwise",
    layer_type="ssm",
)
```

Note: if the norm weight has constant size (e.g., `head_v_dim` that does not
scale with TP), keep it as a plain PyTorch module -- no custom op needed.

### Step 8: Handle MoE

`torch_moe` is sharded automatically by `apply_sharding_hints`:

```python
expert_output = torch.ops.auto_deploy.torch_moe(
    ..., layer_type="moe",
)
```

### Step 9: Register the new model

**CRITICAL**: Without registration, executor worker processes will load the
legacy model instead of yours, and `apply_sharding_hints` will report
**0 nodes processed**. You MUST add a registration import.

Add a side-effect import line to `models/custom/new_sharding/register_sharded_models.py`
in the `_IMPORT_LINES` list. Follow the exact pattern of existing entries:

```python
f"import tensorrt_llm._torch.auto_deploy.models.custom.new_sharding.modeling_foo  {_MARKER}",
```

Then run `python register_sharded_models.py 1` to activate the registration
before testing. This ensures ALL processes (including executor workers) see
your sharded model class.

### Step 10: Create YAML config

Create a YAML config file at `examples/auto_deploy/new_sharding/<model_family>/<model>_sharding_poc.yaml`.
This is REQUIRED -- without it the model cannot be tested.

```yaml
model: org/Model-Name
args:
  world_size: 8                    # MUST be 8 to catch head divisibility issues
  runtime: trtllm
  compile_backend: torch-cudagraph
  max_seq_len: 512
  max_num_tokens: 512
  max_batch_size: 8
  enable_chunked_prefill: true
  model_factory: AutoModelForCausalLM
  kv_cache_config:
    enable_block_reuse: false
    free_gpu_memory_fraction: 0.95
    tokens_per_block: 128
  skip_loading_weights: false
  model_kwargs:
    torch_dtype: bfloat16
  transforms:
    export_to_gm:
      num_moe_experts_for_export: 2    # REQUIRED for models with many experts (>64)
    detect_sharding:
      stage: sharding
      enabled: false                   # MUST disable legacy sharding
    sharding_transform_executor:
      stage: sharding
      enabled: false                   # MUST disable legacy executor
    apply_sharding_hints:
      stage: sharding
      enabled: true                    # Enable new hint-driven sharding
      run_shape_prop: true
      allreduce_strategy: NCCL
      # shard_layers: ['mha', 'mlp']  # Optional: selective layer sharding
    gather_logits_before_lm_head:
      enabled: true
```

The optional `shard_layers` config enables selective layer sharding:

- Not set or `null`: shard ALL shardable nodes (default)
- `['moe']`: shard only MoE layers
- `['moe', 'mha']`: shard MoE and attention layers
- Only nodes with a matching `layer_type` hint are processed

______________________________________________________________________

## Layer-Specific Sharding Patterns

### MHA (standard or gated)

All ops use `layer_type="mha"`:

```
q_proj  -> colwise (+ tp_min_local_shape for GQA)
k_proj  -> colwise (+ tp_min_local_shape for GQA)
v_proj  -> colwise (+ tp_min_local_shape for GQA)
view    -> tp_scaled_dim=2 (head count dimension)
o_proj  -> rowwise
          + all_reduce
```

If Q projection is fused with a gate (e.g., Qwen3.5 `q_proj` outputs `2 * num_heads * head_dim`):
the weight is interleaved per-head `[q_h0, g_h0, q_h1, g_h1, ...]`, so plain
colwise sharding is correct -- no `output_sizes` needed.

### SwiGLU MLP

All ops use `layer_type="mlp"`:

```
gate_proj -> colwise
up_proj   -> colwise
down_proj -> rowwise
              + all_reduce
```

### Mamba / SSM

All ops use `layer_type="ssm"`:

```
in_proj   -> colwise + output_sizes=[gate, conv_input, dt]
split     -> shardable (sizes scale with TP)
conv1d    -> shardable + output_sizes=[hidden, B, C]
split     -> shardable
view      -> tp_scaled_dim=2 (head count)
torch_ssm -> shardable (A, D, dt_bias sharded by handler)
norm      -> tp_mode="colwise" (if weight scales with TP)
out_proj  -> rowwise + all_reduce
```

### GatedDeltaNet

All ops use `layer_type="delta"`:

```
in_proj_qkv -> colwise + output_sizes=[key_dim, key_dim, value_dim]
in_proj_z   -> colwise
in_proj_b   -> colwise
in_proj_a   -> colwise
conv1d      -> shardable + output_sizes=[key_dim, key_dim, value_dim]
split       -> shardable
view        -> tp_scaled_dim=2 (Q, K, V, Z head reshapes)
torch_gated_delta_rule -> shardable (A_log, dt_bias sharded by handler)
norm        -> replicated (constant head_v_dim, plain PyTorch module)
view        -> tp_scaled_dim=2 (post-norm reshape back to [B,S,H,D])
out_proj    -> rowwise + all_reduce
```

### MoE + Shared Expert

All ops use `layer_type="moe"`:

```
router      -> replicated (not sharded, tp_mode="none")
torch_moe   -> automatic (EP/TP by apply_sharding_hints)
shared_expert:
  gate_proj -> colwise
  up_proj   -> colwise
  down_proj -> rowwise (NO all_reduce here)
shared_expert_gate -> replicated (output dim=1, tp_mode="none")
merge: routed + shared -> all_reduce
```

### MLA (DeepSeek)

All ops use `layer_type="mla"`:

MLA uses `torch_mla` which absorbs `kv_b_proj_weight` internally. Do NOT
decompose `torch_mla` into explicit `kv_b_proj` + `torch_attention` -- the
decomposition introduces `expand` ops with concrete `num_heads` that break
after TP sharding.

Instead, keep `torch_mla` and pass `shardable=True`. The `_apply_hint_mla`
handler shards `kv_b_proj_weight` (arg\[4\]) colwise along the head dimension.

```
q_a_proj    -> tp_mode="none" (replicated latent projection)
q_a_layernorm -> unchanged
q_b_proj    -> tp_mode="colwise" (shard by num_heads)
kv_a_proj   -> tp_mode="none" (replicated latent projection)
kv_a_layernorm -> unchanged
torch_mla   -> shardable=True (kv_b_proj_weight sharded by _apply_hint_mla)
view        -> tp_scaled_dim=2 for num_heads (Q reshape only)
o_proj      -> tp_mode="rowwise" + all_reduce
```

The Q split into `q_nope` and `q_pe` is on the LAST dim (head_dim), not the
head count dim, so it does NOT need `auto_deploy::split_with_sizes`.

______________________________________________________________________

## Common Pitfalls

1. **Forgetting `auto_deploy::view` for reshapes.** During `torch.export`, every
   `-1` in `.view()` / `.reshape()` is resolved to a concrete number. After
   sharding changes tensor sizes, these concrete values cause shape mismatches.
   Replace every reshape that has a head count at any position with `auto_deploy::view`.

1. **Sharding tiny projections.** Projections with output_dim=1 (like
   `shared_expert_gate`) cannot be sharded. Use `tp_mode="none"`.

1. **Wrong all_reduce count in MoE.** Use exactly ONE `all_reduce` at the merge
   point of routed + shared expert outputs. Do NOT add separate all_reduces for
   the shared expert and the routed expert path.

1. **Cross-layer parameter contamination.** When implementing `_apply_hint_*`
   handlers that use `get_source_nodes()` to find parameter ancestors, use the
   `allowed_ops` parameter to restrict traversal to elementwise ops only.
   Without this, the traversal crosses layer boundaries through residual
   connections and shards parameters from earlier layers.

1. **Missing `num_moe_experts_for_export: 2` in YAML.** Models with many experts
   (e.g., 128 or 256) hang during `torch.export` without this setting.

1. **Do NOT decompose custom ops that absorb weights.** Some custom ops like
   `torch_mla` take weight tensors as arguments and perform the linear projection
   internally. Do NOT decompose these into explicit `torch_linear_simple` +
   downstream ops -- the decomposition introduces `expand`/`view` operations
   with concrete `num_heads` that get baked into the FX graph and break after
   TP sharding. Instead, add `shardable=True` to the op and let the
   corresponding `_apply_hint_*` handler shard the weight.

1. **Interleaved vs contiguous fused weights.** If a fused weight is interleaved
   per-head-group (e.g., Qwen3Next `in_proj_qkvz`), plain colwise sharding
   works -- no `output_sizes` needed. If the weight is contiguously concatenated
   (e.g., Qwen3.5 `in_proj_qkv = [all_Q | all_K | all_V]`), you MUST provide
   `output_sizes` for proportional splitting.

1. **Missing `layer_type` when `shard_layers` is set.** When the YAML config
   specifies `shard_layers`, only nodes whose `layer_type` matches the list
   are sharded. Nodes with `layer_type="unknown"` (the default) are skipped.
   Always set `layer_type` explicitly on every sharding-aware op call.

1. **Adding `layer_type` to non-shardable ops.** Do NOT add `layer_type` to
   ops that are not in the "Available Sharding-Aware Custom Ops" table.
   Ops like `torch_attention`, `torch_l2norm`, `torch_rope_*` are
   sharding-invariant -- they operate correctly on whatever tensor shapes
   they receive. Adding `layer_type` to these ops will cause the string to
   be interpreted as another positional argument, breaking the op call.

1. **Using conditional `if _s else "none"` patterns.** Do NOT use global
   `SHARD_*` flags or conditional hint values. Always set sharding hints
   unconditionally (e.g., `tp_mode="colwise"`, not `tp_mode="colwise" if _s else "none"`). Layer-level sharding control is handled by `layer_type` +
   `shard_layers` at the transform level.

______________________________________________________________________

## Step 11: Validate

After creating the model file and YAML config, you MUST run the model and
verify it works. Do NOT report success until the test passes.

**IMPORTANT**: The YAML config MUST use `world_size: 8`. This is required to
catch head divisibility issues -- some models have `num_heads` or
`num_kv_heads` values that divide by 4 but not by 8, and the sharding
infrastructure will assert if the head count is not divisible by `world_size`.
If the test fails with an assert like `"Number of units (N) must be divisible by world_size (8)"`, this is an infrastructure limitation, not a bug in your
porting. Report it as a failure with the assert message and the compatible
world_sizes suggested in the error.

### 11a. Check GPU access and run

Before running, check if you have GPU access by running `nvidia-smi` or
`python -c "import torch; print(torch.cuda.device_count())"`. If GPU access
is available, you MUST complete the entire validation loop (run, parse, fix,
re-run) independently. If GPU access is NOT available (e.g. CUDA init blocked
by sandbox), log `ported_untested` in the porting log and report what test
command should be run.

```bash
export HF_HOME=/path/to/hf/cache
cd examples/auto_deploy
python build_and_run_ad.py --yaml-extra new_sharding/<family>/<model>_sharding_poc.yaml
```

### 11b. Update the porting log

After the test completes (success or failure), append a row to
`models/custom/new_sharding/porting_log.csv`:

```
model_name,date,porting_mode,status,notes
```

- `model_name`: the model filename without `modeling_` prefix (e.g., `llama3`)
- `date`: today's date in YYYY-MM-DD format
- `porting_mode`: `agent` (if ported by a subagent) or `manual`
- `status`: `success` or `failure`
- `notes`: if success, list validated scenarios (e.g., "ws=4 BF16, N nodes processed").
  If failure, include the error message and whether files were renamed to `broken_*`.

### 11c. Check for success

The run is successful if:

- Exit code is 0
- `apply_sharding_hints` log shows `N nodes processed` (N > 0)
- The model produces output (even garbage with reduced layers is acceptable)

### 11d. Handle errors

**If the error is in your generated files** (model `.py` or YAML config):

- Fix the error in your generated files
- Rerun and repeat until the test passes
- Common fixable errors: wrong argument order, missing `layer_type`, incorrect
  `tp_mode`, wrong `output_sizes`, missing `auto_deploy::view` for a reshape

**If the error requires modifying core infrastructure files** (custom op
definitions, `sharding.py`, `node_utils.py`, `quantization.py`, etc.):

- Do NOT attempt to fix infrastructure files
- Rename your generated files to indicate they are broken:
  - `modeling_<model>.py` -> `broken_modeling_<model>.py`
  - `<model>_sharding_poc.yaml` -> `broken_<model>_sharding_poc.yaml`
- Create an error report file at `new_sharding/broken_<model>_error.md` with:
  - **Status**: BLOCKED - requires infrastructure changes
  - **Steps to reproduce**: The exact command that was run
  - **Error message**: The full error output
  - **Root cause analysis**: Which infrastructure file needs changes and why
  - **Suggested fix**: What changes would resolve the issue
- Terminate. A human developer will review the error report and make the
  infrastructure changes.

Examples of infrastructure errors (do NOT fix yourself):

- `"layer_type" not found in op schema` -- a custom op is missing the hint
- `"Cannot find value for 'tp_mode'"` -- a quantized op variant lacks hints
- `_apply_hint_X not implemented` -- a new handler is needed in `sharding.py`
- `ShardableOp.X not recognized` -- a new op type needs enum + classifier

______________________________________________________________________

## Validation Checklist (for human review)

After the agent completes porting, verify these manually:

1. **world_size=1**: Run unsharded. `apply_sharding_hints` skips when `world_size < 2`.
   Verify output matches the original model.
1. **world_size=2**: Basic TP sharding. Check for shape mismatches, assertion
   errors, or garbage output.
1. **world_size=8**: Full TP. Verify coherent output and no OOM.
1. **Compare with legacy**: Run the same model on `main` branch with
   `detect_sharding` + manual TP plan. Compare output quality.
1. **Check node count**: The `apply_sharding_hints` log prints
   `N nodes processed`. Verify this matches your expectation (count all
   shardable ops in the model).
1. **Selective sharding**: Test with `shard_layers: ['moe']` in the YAML to
   verify only MoE layers are sharded while others remain replicated.
