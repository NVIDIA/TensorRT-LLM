---
name: ad-sharding-ir-port
description: >
  Ports an existing AutoDeploy custom model (modeling_*.py) to a sharding-aware
  IR variant (modeling_*_ir.py) by mechanically adding sharding hints, op
  substitutions, and all_reduce insertions. Validates with apply_sharding_hints
  and end-to-end multi-GPU runs.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Sharding-aware IR Model Porting (`modeling_*_ir.py`)

**Input:** An existing AutoDeploy custom model at `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_*.py`.
**Output:** A sharding-aware IR variant at `modeling_*_ir.py` in the same directory, plus YAML config and validation.

**Prerequisites:** Familiarity with AD canonical ops (see `ad-model-onboard` skill, Phase 3) and op registration patterns (Phase 4). Refer to the custom op docstrings in `tensorrt_llm/_torch/auto_deploy/custom_ops/` for the complete argument reference (including sharding hints, `tp_mode`, `layer_type`, and which ops accept hints).

The exported FX graph must fully specify how the model should be sharded: the `apply_sharding_hints` transform combines hints with a runtime `DistConfig` for deterministic, node-local sharding.

## Step 0 — IR delta contract (READ FIRST)

An IR port is a **mechanical, structural transform** of `modeling_<name>.py`, NOT a rewrite. The non-IR file at the target branch HEAD is the AUTHORITATIVE source of model logic. Any existing `modeling_<name>_ir.py` is a reference for SHARDING-HINT PATTERNS ONLY (and may be stale wrt logic — older IR files were correct when written, but the non-IR sources have evolved since).

You MAY introduce ONLY the following changes:

**ALLOWED:**

- **A1. Op substitutions:**
  - `nn.Linear(...)` / `F.linear(...)` → `torch.ops.auto_deploy.torch_linear_simple(...)`
  - `tensor.view(...)` / `tensor.reshape(...)` → `torch.ops.auto_deploy.view(...)` (only when the shape contains a TP-scaled dim)
  - `torch.split(...)` / `torch.split_with_sizes(...)` → `torch.ops.auto_deploy.split_with_sizes(...)`
- **A2. Sharding-hint kwargs added** to call sites of: `torch_moe`, `torch_ssm`, `torch_gated_delta_rule`, `torch_causal_conv1d`, `torch_rmsnorm_gated`, `torch_mla`, `torch_linear_simple`, `auto_deploy.split_with_sizes`, `auto_deploy.view`. Allowed kwargs: `tp_mode`, `layer_type`, `output_sizes`, `tp_min_local_shape`, `tp_scaled_dim`, `shardable`, `enable_sharding`.
- **A3. Inserting `torch.ops.auto_deploy.all_reduce(..., layer_type=...)`** after rowwise projections / at MoE merge points (single all_reduce after routed + shared sums).
- **A4. The registration block** at the bottom (`AutoModelForCausalLMFactory.register_custom_model_cls`) plus an `import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401` side-effect at the top.
- **A5. The module docstring update** describing the sharding strategy.

**FORBIDDEN (everything else, including but not limited to):**

- **F1. Replacing ANY `torch.ops.trtllm.*` op with vanilla PyTorch** (e.g. `noaux_tc_op`, `dsv3_router_gemm_op`, fused norm/MLP kernels). The router gate is TP-replicated; there is nothing to shard. AD has no fusion pass that recovers these kernels from a vanilla rewrite — keep the call site verbatim.
- **F2. Changing the input contract** of `forward()` — adding/removing/changing `assert` or `if` statements that change what the caller must pass (e.g. asserting `position_ids is not None` if the non-IR base silently fabricated it from `arange`). The IR port preserves the base's contract.
- **F3. Adding/removing/renaming `nn.Module` subclasses, parameters, buffers**, or `register_load_state_dict_pre_hook` registrations. Module hierarchy and state_dict keys must remain identical to the base.
- **F4. Changing dtype handling, scaling factors, normalization order, mask fill values** (e.g. `0.0` vs `-inf` in `masked_fill`), or any other numerical-semantics detail.
- **F5. Renaming methods, changing return types, changing forward signatures**, or reordering operations.
- **F6. "Cleanup" of allegedly unused code paths** from the non-IR base. If it is in the source file, it stays.
- **F7. Adding code that does not appear in the non-IR base** "because the IR reference has it" — the IR reference may be stale or wrong.

If a change is required that falls outside the allowlist, **STOP and report it to the parent** for explicit human approval BEFORE writing it. Never silently rewrite logic.

## Reference examples (study before porting)

The non-IR `modeling_<name>.py` at the current branch HEAD is the AUTHORITATIVE SOURCE for model logic. The `modeling_<name>_ir.py` files below are REFERENCES for SHARDING-HINT PATTERNS ONLY — they may be stale wrt logic because the corresponding non-IR file has evolved since. **Never copy logic from the IR reference; copy logic only from the non-IR source.** Use the IR reference solely to see how `tp_mode`, `layer_type`, `output_sizes`, `tp_scaled_dim`, `shardable`, `all_reduce`, etc. were placed for similar layer types.

| Original | IR / sharding-aware | Layer types |
|----------|---------------------|-------------|
| `modeling_nemotron_h.py` | `modeling_nemotron_h_ir.py` | Mamba SSM, MHA, SwiGLU MLP, MoE |
| `modeling_qwen3_5_moe.py` | `modeling_qwen3_5_moe_ir.py` | GatedDeltaNet, Gated MHA, SwiGLU MLP, MoE |
| `modeling_mistral.py` | `modeling_mistral_ir.py` | MHA, SwiGLU MLP (simplest) |
| `modeling_deepseek_v2.py` | `modeling_deepseek_v2_ir.py` | MLA, SwiGLU MLP, MoE |

## Step-by-step porting procedure

### Step 1: Copy the source file

```bash
cp tensorrt_llm/_torch/auto_deploy/models/custom/modeling_foo.py \
   tensorrt_llm/_torch/auto_deploy/models/custom/modeling_foo_ir.py
```

### Step 2: Update the module docstring and add imports

At the top of the IR file:

```python
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 -- register all ops
```

Do **not** add global `SHARD_*` flags. Layer-level control uses the `layer_type` hint on each op and `shard_layers` in YAML.

### Step 3: Replace linear projections

For every `self.proj(x)` or `nn.Linear` call, use `torch.ops.auto_deploy.torch_linear_simple` with explicit `tp_mode` and `layer_type`. Always set `tp_mode` unconditionally (no `if _s else "none"`). **Rules:** opening projections (Q/K/V/gate/up/in_proj) → `"colwise"`; closing (O/down/out_proj) → `"rowwise"`; tiny outputs (e.g. `shared_expert_gate` dim 1) → `"none"`; MLA latent projections (q_a, kv_a) → `"none"`. For fused weights split later, pass `output_sizes=[...]`. For GQA, use `tp_min_local_shape=self.head_dim` on K/V colwise lines.

### Step 4: Replace split / chunk after fused colwise projections

Use `torch.ops.auto_deploy.split_with_sizes` with `shardable` / `layer_type` where sizes scale with TP.

### Step 5: Replace view / reshape with concrete head counts

During `torch.export`, `-1` becomes concrete; after TP, wrong values break. Any reshape whose dimension is a head count that scales with TP must use `torch.ops.auto_deploy.view` with `tp_scaled_dim` set appropriately. Safe cases: flat-to-2D, or `[B,S,-1]` when the input is already correctly sharded.

### Step 6: Insert `all_reduce`

After every rowwise projection, add `torch.ops.auto_deploy.all_reduce(..., layer_type=...)`. **Parallel branch rule:** when branches merge by addition, use a **single** `all_reduce` after the sum (e.g. MoE routed + shared expert; parallel attention + MLP residual branches).

### Step 7: Special ops (Conv1d, SSM, GatedDeltaNet, gated RMSNorm)

Add sharding hints on `torch_causal_conv1d`, `torch_ssm`, `torch_gated_delta_rule`, `torch_rmsnorm_gated` per docstrings—typically `shardable` / `output_sizes` / `tp_mode` as required.

### Step 8: MoE

Pass `layer_type="moe"` into `torch_moe`; `apply_sharding_hints` handles EP/TP.

### Step 9: Register the IR model

1. Bottom of the IR file: `AutoModelForCausalLMFactory.register_custom_model_cls("ConfigClassName", ForCausalLM)` (same pattern as `ad-model-onboard` Phase 4).
2. Add a **side-effect import** in `tensorrt_llm/_torch/auto_deploy/models/custom/__init__.py` (e.g. `from . import modeling_foo_ir  # noqa: F401`) and extend `__all__` if you export symbols. Without this import, worker processes may not load your class and `apply_sharding_hints` can report **0 nodes processed**. Do **not** use a separate `register_sharded_models.py` indirection.

### Step 10: YAML — composable registry pattern

Prefer the model registry (`examples/auto_deploy/model_registry/models.yaml`) and **compose** shared fragments under `examples/auto_deploy/model_registry/configs/`, same as other models: list `dashboard_default.yaml`, the right `world_size_N.yaml`, then a dedicated fragment (e.g. `enable_sharder_ir.yaml`) that holds IR sharding transforms. That fragment should disable legacy sharding passes and enable hint-driven sharding. Registry fragments are deep-merged in `yaml_extra` order (see `DynamicYamlMixInForSettings` in `tensorrt_llm/_torch/auto_deploy/utils/_config.py`); place transform keys under `transforms:` so they merge with `dashboard_default.yaml`. Standalone experiment YAMLs for `build_and_run_ad` may wrap the same fields under a top-level `args:` block matching `LlmArgs`.

Example transform block:

```yaml
# Typical contents for enable_sharder_ir.yaml (registry composable fragment)
transforms:
  export_to_gm:
    num_moe_experts_for_export: 2   # often required when expert count is large (>64)
  detect_sharding:
    stage: sharding
    enabled: false
  sharding_transform_executor:
    stage: sharding
    enabled: false
  apply_sharding_hints:
    stage: sharding
    enabled: true
    run_shape_prop: true
    allreduce_strategy: NCCL
    # shard_layers: ['mha', 'mlp']   # optional selective sharding
  gather_logits_before_lm_head:
    enabled: true
```

Use `world_size: 8` when validating TP head-divisibility. Optional `shard_layers` limits which `layer_type` hints are processed; unset means shard all shardable nodes.

### Step 11: Validate

Do not report success until a run completes successfully.

1. Prefer `python examples/auto_deploy/build_and_run_ad.py --model <MODEL-ID> --use-registry` after adding/updating the registry entry and composable YAMLs.
2. `apply_sharding_hints` logs should show **`N nodes processed` with N > 0**.
3. If validation fails with infrastructure limits (e.g. head count not divisible by `world_size`), document the assert and compatible sizes; do not "fix" core `sharding.py` / custom op schemas without owner review.
4. If blocked by missing infrastructure support, rename artifacts to `broken_modeling_*_ir.py` / broken YAML and file a short error report for humans (do not silently patch core transforms).

**Layer type strings** (for `layer_type` / `shard_layers`): use `"mha"`, `"mla"`, `"mlp"`, `"moe"`, `"ssm"`, `"delta"`, or `"unknown"` (default; skipped when `shard_layers` is set). Match the conventions used in `apply_sharding_hints` and project enums.

### Step 12 — Pre-finalization self-audit (MANDATORY)

Before reporting the IR file as done, you MUST run:

```bash
diff -u tensorrt_llm/_torch/auto_deploy/models/custom/modeling_<name>.py \
        tensorrt_llm/_torch/auto_deploy/models/custom/modeling_<name>_ir.py
```

Then classify every hunk into one of the following categories (defined in Step 0 — IR delta contract):

| Category | Allowed? | Description |
|---|---|---|
| **A1** | yes | Op substitution (`linear` / `view` / `split`) |
| **A2** | yes | Sharding-hint kwarg added (`tp_mode`, `layer_type`, `output_sizes`, `tp_min_local_shape`, `tp_scaled_dim`, `shardable`, `enable_sharding`) |
| **A3** | yes | `auto_deploy.all_reduce` insertion |
| **A4** | yes | Registration / `custom_ops` side-effect import |
| **A5** | yes | Module docstring update describing sharding strategy |
| **F1** | NO | `torch.ops.trtllm.*` replaced with vanilla PyTorch |
| **F2** | NO | Input contract change (asserts, fallbacks added/removed) |
| **F3** | NO | Module hierarchy / parameter / buffer / load-hook change |
| **F4** | NO | Numerical-semantics change (dtype, scale, mask fill, order) |
| **F5** | NO | Method rename / signature change / op reorder |
| **F6** | NO | Removal of allegedly unused base code |
| **F7** | NO | Code added because the IR reference had it (and the non-IR base did not) |

**If you find any F# hunk, REVERT it to the non-IR source verbatim before reporting done.** Report the full diff classification table back to the parent agent in your final message, with one row per hunk:

```
| Hunk lines (in IR file) | Summary of change | Category | Verdict |
|---|---|---|---|
| 234-240                 | F.linear → torch_linear_simple, tp_mode="colwise" | A1 + A2 | OK |
| 264-340                 | noaux_tc_op replaced with vanilla PyTorch         | F1      | REVERTED to base |
| ...                     | ...                                               | ...     | ... |
```

You are NOT done until every row in the table is a yes-allowed category.

## Layer-specific sharding patterns

**MHA (standard or gated):** `layer_type="mha"`: q/k/v colwise (GQA: `tp_min_local_shape`), `view` with `tp_scaled_dim` for head dim, o rowwise + `all_reduce`. Fused Q+gate interleaved per head: colwise without `output_sizes`; contiguous Q|K|V fused blocks need `output_sizes`.

**SwiGLU MLP:** `layer_type="mlp"`: gate/up colwise, down rowwise + `all_reduce`.

**Mamba / SSM:** `layer_type="ssm"`: in_proj colwise + `output_sizes`, splits shardable, conv1d shardable + `output_sizes`, views, `torch_ssm` shardable, norm gated colwise if weight scales, out rowwise + `all_reduce`.

**GatedDeltaNet:** `layer_type="delta"`: in_proj_qkv with `output_sizes`, other in_projs colwise, conv1d/splits/views as above, `torch_gated_delta_rule` shardable, out rowwise + `all_reduce`.

**MoE + shared expert:** `layer_type="moe"`: router replicated; one `all_reduce` after `routed + shared`, not two.

**MLA (DeepSeek):** `layer_type="mla"`: keep `torch_mla` intact with `shardable=True`—do **not** decompose into separate linears + `torch_attention` (introduces bad `expand`/`view` with concrete head counts). q_a/kv_a latent: `tp_mode="none"`; q_b colwise; `o_proj` rowwise + `all_reduce`.

## Common pitfalls

1. **Missing `auto_deploy::view` for head reshapes** — concrete shapes from export break after sharding.
2. **Sharding tiny projections** — dim-1 gates: `tp_mode="none"`.
3. **Double `all_reduce` in MoE** — one merge-point reduction for routed + shared.
4. **Cross-layer parameter contamination** — in `_apply_hint_*` handlers using `get_source_nodes()`, restrict with `allowed_ops` so residual links do not pull weights from other layers.
5. **Missing `num_moe_experts_for_export`** for very large expert counts — export can hang.
6. **Decomposing ops that absorb weights** (e.g. `torch_mla`) — use `shardable` + handler instead of splitting into plain linears.
7. **Interleaved vs contiguous fused weights** — interleaved per-head groups: colwise only; contiguous Q|K|V blocks: require `output_sizes`.
8. **Omitting `layer_type` when using `shard_layers`** — `"unknown"` nodes are skipped; set hints explicitly on sharding-aware ops.
9. **`layer_type` on non-hint ops** — do **not** pass `layer_type` to ops that are not designed for sharding hints (e.g. `torch_attention`, `torch_l2norm`, `torch_rope_*`); extra positional args break calls. Confirm in `custom_ops/` docstrings which ops accept hints.
10. **Conditional hint values** — no `if _s else "none"`; use unconditional hints and rely on `shard_layers` / transform config.
11. **Replacing `torch.ops.trtllm.*` ops** — `noaux_tc_op`, `dsv3_router_gemm_op`, fused norm/MLP kernels are TP-replicated and must be kept verbatim (rule F1). AD has no fusion pass to recover them from vanilla PyTorch.

## Validation checklist (human review)

- `world_size=1`: unsharded path; hints should not break correctness.
- `world_size=2` and `8`: shape checks and coherent output.
- `apply_sharding_hints` node count vs expectation.
- Optional: `shard_layers: ['moe']` to verify selective sharding.
