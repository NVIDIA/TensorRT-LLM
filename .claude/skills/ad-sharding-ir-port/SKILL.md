---
name: ad-sharding-ir-port
description: >
  Adds sharding-aware IR hints (op substitutions, sharding kwargs, all_reduce
  insertions) directly into an existing AutoDeploy custom model
  (modeling_*.py). Edits the file in place — no separate _ir.py copy.
  Validates with apply_sharding_hints and end-to-end multi-GPU runs.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Adding Sharding IR Hints to an AutoDeploy Custom Model

**Input:** An existing AutoDeploy custom model at `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_*.py`.
**Output:** The same file, updated in place with sharding hints, plus YAML config and validation.

**No separate `_ir.py` file.** Sharding IR is the default path — hints are added directly to the canonical `modeling_*.py`. The legacy pattern of maintaining parallel `modeling_*_ir.py` files is deprecated.

**Prerequisites:** Familiarity with AD canonical ops (see `ad-model-onboard` skill, Phase 3) and op registration patterns (Phase 4). Refer to the custom op docstrings in `tensorrt_llm/_torch/auto_deploy/custom_ops/` for the complete argument reference (including sharding hints, `tp_mode`, `layer_type`, and which ops accept hints).

The exported FX graph must fully specify how the model should be sharded: the `apply_sharding_hints` transform combines hints with a runtime `DistConfig` for deterministic, node-local sharding.

## Step 0 — Sharding-hint delta contract (READ FIRST)

Adding sharding hints is a **mechanical, structural transform** of the existing `modeling_<name>.py`, NOT a rewrite. The file at the target branch HEAD before your changes is the AUTHORITATIVE source of model logic.

You MAY introduce ONLY the following changes:

**ALLOWED:**

- **A1. Op substitutions:**
  - `nn.Linear(...)` / `F.linear(...)` → `torch.ops.auto_deploy.torch_linear_simple(...)`
  - `tensor.view(...)` / `tensor.reshape(...)` → `torch.ops.auto_deploy.view(...)` (only when the shape contains a TP-scaled dim)
  - `torch.split(...)` / `torch.split_with_sizes(...)` → `torch.ops.auto_deploy.split_with_sizes(...)`
- **A2. Sharding-hint kwargs added** to call sites of: `torch_moe`, `torch_ssm`, `torch_gated_delta_rule`, `torch_causal_conv1d`, `torch_rmsnorm_gated`, `torch_mla`, `torch_linear_simple`, `auto_deploy.split_with_sizes`, `auto_deploy.view`. Allowed kwargs: `tp_mode`, `layer_type`, `output_sizes`, `tp_min_local_shape`, `tp_scaled_dim`, `shardable`, `enable_sharding`.
- **A3. Inserting `torch.ops.auto_deploy.all_reduce(..., layer_type=...)`** after rowwise projections / at MoE merge points (single all_reduce after routed + shared sums).
- **A4. Adding `import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401`** side-effect import at the top if not already present.
- **A5. The module docstring update** describing the sharding strategy.

**FORBIDDEN (everything else, including but not limited to):**

- **F1. Replacing ANY `torch.ops.trtllm.*` op with vanilla PyTorch** (e.g. `noaux_tc_op`, `dsv3_router_gemm_op`, fused norm/MLP kernels). The router gate is TP-replicated; there is nothing to shard. AD has no fusion pass that recovers these kernels from a vanilla rewrite — keep the call site verbatim.
- **F2. Changing the input contract** of `forward()` — adding/removing/changing `assert` or `if` statements that change what the caller must pass.
- **F3. Adding/removing/renaming `nn.Module` subclasses, parameters, buffers**, or `register_load_state_dict_pre_hook` registrations. Module hierarchy and state_dict keys must remain identical.
- **F4. Changing dtype handling, scaling factors, normalization order, mask fill values** (e.g. `0.0` vs `-inf` in `masked_fill`), or any other numerical-semantics detail.
- **F5. Renaming methods, changing return types, changing forward signatures**, or reordering operations.
- **F6. "Cleanup" of allegedly unused code paths.** If it is in the file, it stays.
- **F7. Adding code that does not appear in the original** "because a legacy `_ir.py` reference had it" — legacy IR files may be stale or wrong.

If a change is required that falls outside the allowlist, **STOP and report it to the parent** for explicit human approval BEFORE writing it. Never silently rewrite logic.

## Reference examples (study before porting)

The models below already have sharding hints integrated directly into their `modeling_*.py` files. Study them to see how `tp_mode`, `layer_type`, `output_sizes`, `tp_scaled_dim`, `shardable`, `all_reduce`, etc. are placed for different layer types.

| Model file | Layer types |
|----------|-------------|
| `modeling_nemotron_h.py` | Mamba SSM, MHA, SwiGLU MLP, MoE |
| `modeling_qwen3_5_moe.py` | GatedDeltaNet, Gated MHA, SwiGLU MLP, MoE |
| `modeling_deepseek.py` | MLA, SwiGLU MLP, MoE |
| `modeling_qwen3.py` | MHA, SwiGLU MLP (simplest MHA example) |

## Step-by-step procedure

### Step 1: Create a git checkpoint

Before editing, ensure the file is committed so you can diff against the original:

```bash
git stash  # or commit — ensure a clean baseline to diff against
```

### Step 2: Add the custom_ops side-effect import

If not already present at the top of the file:

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

### Step 9: Verify registration

The model's existing registration (`AutoModelForCausalLMFactory.register_custom_model_cls` at the bottom of the file and its import in `__init__.py`) stays unchanged. No new registration is needed — sharding hints do not change the model identity.

### Step 10: YAML — enable hint-driven sharding

Add `enable_sharder_ir.yaml` to the model's `yaml_extra` list in `examples/auto_deploy/model_registry/models.yaml` (if not already present). This composable fragment disables legacy sharding passes and enables `apply_sharding_hints`. Registry fragments are deep-merged in `yaml_extra` order (see `DynamicYamlMixInForSettings` in `tensorrt_llm/_torch/auto_deploy/utils/_config.py`).

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

### Step 11a — End-to-end run

Do not report success until a run completes successfully.

1. Prefer `python examples/auto_deploy/build_and_run_ad.py --model <MODEL-ID> --use-registry` after updating the registry entry.
2. `apply_sharding_hints` logs should show **`N nodes processed` with N > 0**.
3. If validation fails with infrastructure limits (e.g. head count not divisible by `world_size`), document the assert and compatible sizes; do not "fix" core `sharding.py` / custom op schemas without owner review.
4. If blocked by missing infrastructure support, revert the sharding-hint changes and file a short error report for humans (do not silently patch core transforms).

**Layer type strings** (for `layer_type` / `shard_layers`): use `"mha"`, `"mla"`, `"mlp"`, `"moe"`, `"ssm"`, `"delta"`, or `"unknown"` (default; skipped when `shard_layers` is set). Match the conventions used in `apply_sharding_hints` and project enums.

### Step 11b — Sharding equivalence test (MANDATORY)

Run the offline sharding-IR equivalence test ([`tests/unittest/auto_deploy/multigpu/transformations/library/test_sharding_ir_equivalence.py`](tests/unittest/auto_deploy/multigpu/transformations/library/test_sharding_ir_equivalence.py)) against the modeling file you just edited, under **every** parallelism configuration the test exposes. The port is **not** complete until every configuration passes. Skipping this step or treating a partial pass (e.g. only `tep`) as success is not allowed.

The test compares a sharded prefill against the unsharded eager reference on a tiny (4-layer, hidden_size=64) instance of the model and asserts `rel_rmse < tol`, where `tol` is the test-defined relative-RMSE tolerance (`REL_RMSE_TOL` constant in [`test_sharding_ir_equivalence.py`](tests/unittest/auto_deploy/multigpu/transformations/library/test_sharding_ir_equivalence.py); overridable per invocation via the `SHARDING_IR_REL_RMSE_TOL` env var). It uses no PyExecutor / no compile / no checkpoint download, so each cell runs in ~30s on 4xGPU.

**Run the matrix:**

```bash
MODEL=tensorrt_llm/_torch/auto_deploy/models/custom/modeling_<name>.py
TEST=tests/unittest/auto_deploy/multigpu/transformations/library/test_sharding_ir_equivalence.py

for CFG in tp-only ep-only tep attn-dp; do
  pytest "$TEST" --sharding-ir-modeling-file "$MODEL" --sharding-ir-dist-config "$CFG" -s -v \
    2>&1 | tee /tmp/sharding_ir_${CFG}.log
done
```

**Parse the output for each cell. A cell PASSES iff ALL of these are true:**

1. pytest exit code is `0`.
2. The log contains the line `1 passed` in the pytest summary block.
3. The log contains the rank-0 metrics line `[sharding-ir-eq] |y_s - y_u|: max=... mean=... rel_rmse=<X.XXXXXX> (tol=<Y.YYYYYY>)` and the parsed `rel_rmse` is **strictly less than the parsed `tol`** from the same line. Do not hardcode a tolerance value in the parser — read both `rel_rmse=` and `tol=` from the test's own log and compare them. This stays correct if the test's `REL_RMSE_TOL` is later changed or a per-invocation `SHARDING_IR_REL_RMSE_TOL` is supplied.

Quick one-liner that prints PASS/FAIL plus the parsed `rel_rmse` and `tol` per cell:

```bash
for CFG in tp-only ep-only tep attn-dp; do
  log=/tmp/sharding_ir_${CFG}.log
  if grep -q "1 passed" "$log"; then status=PASS; else status=FAIL; fi
  line=$(grep "sharding-ir-eq" "$log" | grep "rel_rmse=" | head -1)
  rmse=$(echo "$line" | sed -E 's/.*rel_rmse=([0-9.]+).*/\1/')
  tol=$(echo  "$line" | sed -E 's/.*\(tol=([0-9.]+)\).*/\1/')
  echo "${CFG}: ${status} rel_rmse=${rmse:-NA} tol=${tol:-NA}"
done
```

**Failure handling:**

- A cell failing with `KeyError`, `AttributeError`, `ValueError: You must specify exactly one of input_ids or inputs_embeds`, or any exception *before* `[sharding-ir-eq]` prints means the **modeling code itself** does not yet build / export on a tiny config — fix the modeling code (within the Step 0 allowlist) before proceeding. Do not silently skip the cell.
- A cell where `[sharding-ir-eq]` prints `rel_rmse >= tol` (from the same log line) means a **sharding-hint bug**: a missing `all_reduce`, a wrong `tp_mode`, a `view` without `tp_scaled_dim`, a `split_with_sizes` whose sizes do not scale, etc. Re-read Step 6 (all_reduce), Step 3 (tp_mode), Step 5 (view), Step 4 (split_with_sizes) and the layer-specific patterns. Iterate on the hints until clean. If the failure is small (rel_rmse just slightly above tol) and you have reason to believe it is real numerical noise from the specific layer mix of this model rather than a sharding-hint bug, raise it with the parent agent rather than silently bumping `SHARDING_IR_REL_RMSE_TOL`.
- A cell that the modeling file legitimately does not support (e.g. `ep-only` on a dense model with no MoE) is acceptable only if the failure is a documented `pytest.skip(...)` from the test infrastructure. A silent `FAIL` is **not** acceptable.

### Step 12 — Pre-finalization self-audit (MANDATORY)

Before reporting the file as done, you MUST diff your changes against the git baseline:

```bash
git diff tensorrt_llm/_torch/auto_deploy/models/custom/modeling_<name>.py
```

Then classify every hunk into one of the following categories (defined in Step 0):

| Category | Allowed? | Description |
|---|---|---|
| **A1** | yes | Op substitution (`linear` / `view` / `split`) |
| **A2** | yes | Sharding-hint kwarg added (`tp_mode`, `layer_type`, `output_sizes`, `tp_min_local_shape`, `tp_scaled_dim`, `shardable`, `enable_sharding`) |
| **A3** | yes | `auto_deploy.all_reduce` insertion |
| **A4** | yes | `custom_ops` side-effect import added |
| **A5** | yes | Module docstring update describing sharding strategy |
| **F1** | NO | `torch.ops.trtllm.*` replaced with vanilla PyTorch |
| **F2** | NO | Input contract change (asserts, fallbacks added/removed) |
| **F3** | NO | Module hierarchy / parameter / buffer / load-hook change |
| **F4** | NO | Numerical-semantics change (dtype, scale, mask fill, order) |
| **F5** | NO | Method rename / signature change / op reorder |
| **F6** | NO | Removal of allegedly unused base code |
| **F7** | NO | Code added because a legacy `_ir.py` reference had it (and the base did not) |

**If you find any F# hunk, REVERT it before reporting done.** Report the full diff classification table back to the parent agent in your final message, with one row per hunk:

```
| Hunk lines | Summary of change | Category | Verdict |
|---|---|---|---|
| 234-240    | F.linear → torch_linear_simple, tp_mode="colwise" | A1 + A2 | OK |
| 264-340    | noaux_tc_op replaced with vanilla PyTorch         | F1      | REVERTED |
| ...        | ...                                               | ...     | ... |
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

- All four configurations of the **sharding equivalence test** (Step 11b) pass with the parsed `rel_rmse` strictly below the parsed `tol` from the same rank-0 log line. Report the per-cell `rel_rmse` and `tol` pair.
- `world_size=1`: unsharded path; hints should not break correctness.
- `world_size=2` and `8`: shape checks and coherent output.
- `apply_sharding_hints` node count vs expectation.
- Optional: `shard_layers: ['moe']` to verify selective sharding.
