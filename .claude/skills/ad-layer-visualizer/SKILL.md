---
name: ad-layer-visualizer
description: >
  Visualize a specific transformer decoder layer from an AutoDeploy FX graph text dump
  as a hierarchical DOT/PNG diagram. Optionally annotate nodes with actual GPU kernel
  names and durations from an nsys trace. Use when the user wants to visualize, inspect,
  or debug a layer in an AutoDeploy model graph dump. Triggers on: "visualize layer",
  "show layer", "graph of layer", "layer visualization", "dump graph layer".
  Assumes graph dumps already exist in a directory (produced by AD_DUMP_GRAPHS_DIR).
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# AutoDeploy Layer Visualizer

Visualize a single transformer decoder layer from an AutoDeploy SSA graph dump.
Optionally overlay actual GPU kernel names and durations from an nsys trace.

> **Prerequisite knowledge**: This skill assumes familiarity with the `ad-graph-dump` skill, which covers how to enable dumps via `AD_DUMP_GRAPHS_DIR`, file naming conventions, SSA format basics, and GraphModule section structure. Refer to `ad-graph-dump` for that context.

## Inputs

1. **Dump directory** — path to a directory containing graph dump `.txt` files
2. **Layer number** — which decoder layer to visualize (e.g. 5)
3. **(Optional) Dump file** — specific `.txt` file. If not given, pick the file with the highest numeric prefix (final transform stage).
4. **(Optional) Trace file** — path to `.nsys-rep` or `.sqlite` trace file. When provided, GPU kernel names and durations are extracted and annotated onto each node in the visualization.

## Workflow

### Phase 0: Ask the user

Before starting, ask the user two questions (skip any already answered in their request):

1. **Do you have an nsys trace file?** (`.nsys-rep` or `.sqlite`) — if yes, kernel names and durations from the trace will be annotated directly on each node in the diagram.
2. **If yes — prefill or decode?** The CUDA graph structure differs between prefill and decode phases. Knowing which phase the trace captures helps correctly segment layers and map kernels.

Proceed once both are answered (or the user says no trace).

### Phase 1: Select the dump file

If the user didn't specify a file, pick the file with the highest numeric prefix (the final transform stage). See `ad-graph-dump` for the file naming convention and how lexicographic sort matches pipeline order.

### Phase 2: Read the graph dump

Read the selected `.txt` file. If it contains multiple `GraphModule` sections (delimited by `========` headers), pick the one labeled `monolithic.model` or the first/largest one with real operation nodes.

### Phase 3: Extract the layer subgraph

This is the core step — you do this yourself by understanding the graph structure. Do NOT delegate to a script.

The dump is an SSA-form graph where each line is one of:
- **Placeholder**: `%name : shape : dtype` (model inputs like input_ids, kv_cache, etc.)
- **Operation**: `%name = namespace.op_name(%input1, %input2, ..., const_args...) : shape : dtype`
- **Output**: `output(%name1, %name2, ...)`

**How to identify which nodes belong to layer N:**

A transformer decoder layer typically contains these blocks in order:
1. **Input LayerNorm** — consumes `model_layers_N_input_layernorm_weight`
2. **Self-Attention** (MLA/MHA/GQA) — consumes `model_layers_N_self_attn_*` weights (q_proj, k_proj, v_proj, o_proj, kv_a_proj, kv_b_proj, q_a_proj, q_b_proj, etc.)
3. **Post-Attention LayerNorm + Residual** — consumes `model_layers_N_post_attention_layernorm_weight`
4. **FFN / MoE** — consumes `model_layers_N_mlp_*` weights (gate_weight, shared_experts, etc.) and `fused_*_N_*` fused weight references

**Extraction rules:**

1. **Seed nodes**: Any operation whose arguments include a weight reference matching `layers_N_` or `layers.N.` or a fused weight pattern like `fused_*_N_*` (where these are non-`%` references, i.e. weight parameters, not activation outputs from other ops).

2. **Forward follow**: From seeds, follow the dataflow forward — if a node consumes the output of a seed/layer node, it belongs to this layer too. Stop when you hit a node that consumes weights from a *different* layer (e.g. `layers_M_` where M ≠ N).

3. **Backward follow**: From seeds, follow the dataflow backward to pick up intermediate ops (getitem, sub, floordiv, eq, mul, view, reshape, etc.) that feed into seed nodes. Stop when you hit:
   - A placeholder node (model input)
   - A node that **belongs to a different layer** (i.e., it was already identified as part of layer M ≠ N by the seed rule, or its own backward chain reaches a different layer's weights)
   - The output of a residual-add from the previous layer (this is the boundary between layers)

4. **Critical**: Generic arithmetic ops like `sub_1`, `floordiv_1`, `mul_1`, `eq_1` that sit between two layers' MoE routing logic can be tricky. Trace their inputs backward — if they ultimately derive from layer M's weights/operations (not layer N's), they belong to layer M, not layer N. The suffix number on these generic ops does NOT indicate which layer they belong to; you must trace the dataflow.

5. **External inputs**: Nodes from outside the layer that feed into layer nodes are "external inputs" (show them as input nodes in the diagram). These include:
   - Previous layer's residual output
   - Global model buffers (RoPE caches like `_ad_rotary_cos_sin_N`, batch metadata)
   - KV cache placeholders

### Phase 4: Produce the JSON

After extraction, output a JSON file at `<dump_dir>/<dump_stem>_layer<N>.json` with this structure:

```json
{
  "layer": 5,
  "source_file": "085_compile_compile_model.txt",
  "nodes": [
    {
      "id": "noaux_tc_op_default_2",
      "op": "trtllm.noaux_tc_op.default",
      "shape": "(8x8, 8x8)",
      "dtype": "(torch.bfloat16, torch.int32)",
      "group": "moe",
      "sub_group": "moe_router",
      "inputs": ["dsv3_router_gemm_op_default_2"],
      "weight_inputs": [
        {"name": "model_layers_5_mlp_gate_e_score_correction_bias", "shape": "256", "dtype": "torch.bfloat16"}
      ]
    }
  ],
  "edges": [
    {"from": "dsv3_router_gemm_op_default_2", "to": "noaux_tc_op_default_2"}
  ],
  "external_inputs": [
    {
      "id": "trtllm_fused_allreduce_residual_rmsnorm_default_4",
      "label": "Layer 4 residual output",
      "shape": "(2x4x7168, 2x4x7168)"
    }
  ]
}
```

**Node fields:**
- `id`: the SSA name (without `%` prefix)
- `op`: the full operation target string
- `shape`, `dtype`: output shape and dtype
- `group`: one of `norm`, `attention`/`mla`, `moe`, `mlp`, `mamba`, `gdn`, `other`
- `sub_group` (optional): finer classification like `q_branch`, `kv_branch`, `rope`, `moe_router`, `moe_experts`, `shared_experts`
- `inputs`: list of node IDs that this node consumes (only nodes within the layer or external inputs)
- `weight_inputs`: list of weight parameters consumed (name, shape, dtype)

**Edge fields:**
- `from`, `to`: node IDs (both must be in `nodes` or `external_inputs`)

**Group assignment heuristic:**
- `norm`: ops consuming `input_layernorm` or `post_attention_layernorm` weights
- `mla`/`attention`: ops consuming `self_attn_*` weights or named `*mla*`, `*rope*`, `*sdpa*`
- `moe`: ops consuming MoE weights (`*moe*`, `*experts*`), router ops (`noaux_tc_op`, topk), and the arithmetic ops that process router outputs (sub, floordiv, eq, mul between router and MoE fused op)
- `mlp`: ops consuming `mlp_*` weights that aren't MoE (e.g., `shared_experts`, `gate_proj`, `up_proj`, `down_proj`)
- `other`: everything else (getitem, view, reshape, etc.) — assign to the same group as neighbors

### Phase 5: (Optional) Extract trace kernels

If the user provided a trace file, extract per-layer kernel sequences:

```bash
python <skill_dir>/scripts/extract_trace_kernels.py <trace_file> --layer <N> --output <dump_dir>/<dump_stem>_layer<N>_kernels.json
```

This script uses `graphNodeId` to extract a single CUDA graph replay, groups kernels by stream, and identifies which streams belong to which layer. It outputs a JSON with per-layer kernel sequences including short names, full names, durations, and stream IDs.

### Phase 6: (Optional) Annotate each node with its trace kernels

When trace kernel data is available, you must map GPU kernels to individual FX graph nodes. This is the key step — the render script will display kernel names and durations directly on each node in the diagram.

**How to map kernels to nodes:**

Read the trace kernel JSON and the layer JSON side by side. For each FX graph node, identify which GPU kernel(s) it corresponds to based on the op type and the kernel execution order. Common mappings:

| FX graph op | Trace kernel(s) |
|---|---|
| `flashinfer_mla_with_cache` | `fmhaSm100...` |
| `finegrained_fp8_linear` | `fp8_blockscale` + `pack_fp32_to_ue8m0` + `deep_gemm_fp8` |
| `torch_linear_simple` | `nvjet_gemm` + `splitKreduce` |
| `flashinfer_rms_norm` | `rms_norm_reduce_fusion` |
| `flashinfer_fused_add_rms_norm` | `fused_kernel_a5fe...` |
| `mlir_fused` (input_layernorm) | `fused_kernel_1984...` |
| `mlir_fused` (post_attn) | `fused_kernel_2e06...` |
| `trtllm_moe_fused` | `bmm_E4m3...` + `moe_activation_deepseek` + `bmm_Bfloat16...` + `moe_finalize` |
| `noaux_tc_op` (top-k routing) | `deepseek_v3_topk` |
| `fused_swiglu_mlp` / `fused_finegrained_fp8_swiglu_mlp` | `deep_gemm_fp8` (×2) + `act_and_mul` |
| `trtllm_dist_all_reduce` | `nccl_allreduce_symk` or `symm_mem_allreduce` |
| `symm_mem_all_gather` | `symm_mem_allgather` or `nccl_allgather` |
| `triton_rope_on_interleaved_qk_inputs` | `mla_rope_assign_qkv` |

Add a `"trace_kernels"` list to each node that has corresponding GPU kernels:

```json
{
  "id": "flashinfer_mla_with_cache_default_5",
  "op": "auto_deploy.flashinfer_mla_with_cache.default",
  "group": "mla",
  "trace_kernels": [
    {"kernel": "fmhaSm100...", "duration_us": 50.1}
  ],
  ...
}
```

Also add top-level `trace_summary` for the whole layer:

```json
{
  "layer": 5,
  "trace_summary": {
    "total_duration_us": 650.9,
    "kernel_count": 66,
    "num_streams": 3
  },
  "nodes": [...]
}
```

The render script will display each node's kernels as `⚡ kernel_name (Xµs)` lines below the node label. Nodes without trace_kernels are left unchanged.

### Phase 7: Render

Run the bundled visualization script:

```bash
python <skill_dir>/scripts/render_layer.py <dump_dir>/<dump_stem>_layer<N>.json --output <dump_dir>/<dump_stem>_layer<N>
```

This reads the JSON and produces `.dot` and `.png` files. If nodes have `trace_kernels` fields, the script renders kernel names and durations directly on each node's label (prefixed with ⚡).

Present the output paths to the user.

## Notes

- The script requires `graphviz` (`dot` command) to be installed for PNG rendering
- If the dump has multiple GraphModules, prefer the `monolithic.model` module or auto-select the largest
- When in doubt about a node's layer membership, trace its weight dependencies all the way back — weight parameter names are the ground truth for layer assignment
- The trace extraction script auto-converts `.nsys-rep` to `.sqlite` if needed (requires `nsys` on PATH)
- CUDA graph kernels run across multiple streams per layer; the script handles multi-stream grouping automatically
