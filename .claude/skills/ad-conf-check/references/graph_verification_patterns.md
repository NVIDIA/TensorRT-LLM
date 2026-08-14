<!--
SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Graph Dump Verification Patterns

This documents how graph dumps (from `AD_DUMP_GRAPHS_DIR`) are used to verify config application.

## Graph File Format

- Files: `NNN_stage_transform.txt` — numbered sequentially, showing graph AFTER each transform
- Header: `# Transform: <name>`, `# Stage: <stage>`, `# GraphModules found: N`
- Body: FX graph nodes in format `%name = op(args) : shape : dtype`
- Custom ops use `auto_deploy.<op_name>.default(...)` prefix

## Verification Methods

### Sharding (`detect_sharding`)

| Config | What to Check | Graph Pattern |
|--------|--------------|---------------|
| `allreduce_strategy` | Collective ops use the configured strategy | `trtllm_dist_all_reduce.default(..., SYMM_MEM)` |
| `simple_shard_filter` | Target has collective op + sharded weight | Compare weight shape before/after sharding (e.g., 248320x4096 -> 31040x4096) and `all_gather` downstream |
| `dist_mapping` | Collective op counts present | Count `trtllm_dist_all_reduce` + `trtllm_dist_all_gather` in final graph |
| `shard_all_unprocessed` | Collective ops exist in final graph | Any `trtllm_dist_(all_reduce\|all_gather\|reduce_scatter)` present |

### Attention (`attn_backend`)

| Backend | Graph Pattern |
|---------|---------------|
| `trtllm` | `auto_deploy.torch_attention.default(...)` ops |
| `flashinfer` | `auto_deploy.flashinfer_attention.default(...)` ops |

### Fusion Transforms

| Transform | Graph Pattern |
|-----------|---------------|
| `fuse_nvfp4_moe` | `auto_deploy.torch_quant_nvfp4_moe` ops in final graph |
| `fuse_gemms_mixed_children` | Reduced count of `auto_deploy.torch_linear_simple` ops (compare before/after) |
| `match_rmsnorm_pattern` | `auto_deploy.torch_rmsnorm` ops |
| `match_swiglu_pattern` | `auto_deploy.torch_swiglu_mlp` or `auto_deploy.torch_nvfp4_swiglu` ops |
| `optimize_rope` | `auto_deploy.flashinfer_rope` or `auto_deploy.fused_rope` ops |

### Export (`export_to_gm`)

- Verify the `export_to_gm` graph dump file exists and contains ops (count `%` lines)

### Generic Transform Verification

- Compare op count (`%` lines) between consecutive graph files
- If count changed, transform modified the graph -> APPLIED
- If count unchanged, transform may have been a no-op -> check SUMMARY log

## How Graph Evidence Interacts with Log Evidence

1. **UNKNOWN + graph evidence** -> upgrades to the graph result (typically APPLIED)
2. **APPLIED + graph evidence** -> appends graph info as supplementary proof (separated by `|`)
3. **SKIPPED + graph=APPLIED** -> marked as **CONFLICT** (log says no-op but graph shows changes — requires manual investigation)
4. **FAILED + graph=APPLIED** -> marked as **CONFLICT** (log says failure but graph shows the transform took effect — requires manual investigation)
5. **APPLIED + graph=FAILED** -> marked as **CONFLICT** (log says success but graph shows issues)
6. **FAILED/SKIPPED + non-contradicting graph** -> log status takes precedence, graph info appended as supplementary evidence
7. **No graph dir** -> script works as before, log-only analysis
