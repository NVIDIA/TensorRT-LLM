<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Generate DSpark whole-step costs

Use this recipe when you need a new cost table. Reusing a compatible supplied
table does not require profiling again. Installation, artifact reuse and
ordinary execution are covered by the [reproduction guide](CONFIDENCE_REPRODUCTION.md).

The workflow is **profile → reduce → export**. Whole-step collection uses the
executor's complete-step `hostStepTimeMS` timer rather than only graph replay.
It uses the normal model, KV, sampler and
shutdown paths. No forced acceptance is permitted.

## 1. Configure the runtime and grid

K is the physical draft block, G is the rank-local request count, and V is the
total verifier-row budget of a captured compact layout. V0 is the sentinel for
native full-width `G*(K+1)` verification, not zero work. Every G needs a native
V0 control. Compact cells require K > 1 and `G <= V < G*(K+1)`; choose the compact
cells you want to price rather than sweeping every possible V.

Prepare a complete runtime YAML for the intended model, precision and backend.
It must have:

- `tensor_parallel_size` equal to the MPI world size, PP=1 and CP=1;
- the intended expert parallelism and attention DP, plus sufficient request,
  token, sequence-length and KV capacity;
- `enable_iter_perf_stats: true` for the complete-loop timer;
- padded CUDA graphs with a resolved rank-local ladder exactly matching all G
  values in the grid;
- embedded DSpark at the chosen physical K, with confidence scheduling and the
  fused scheduler OFF, no mixed evidence or draft-length schedule, and no
  SPS, STS, fingerprint or admission fields.

Do not use an enabled `dynamic.yaml` unchanged for profiling. The grid's
`record_confidence_head: true` records the loaded head inside the graphs without
enabling serving confidence policy.

The current whole-step exporter supports attention-DP fanout or a single rank;
multi-rank attention-DP-off export is refused. The offline driver supports
aggregate greedy generation and rejects guided decoding, LoRA, custom logits
processors, disaggregated KV transfer/connectors, in-graph sampling, encode-only
and resource-governor control. Multi-rank operation needs MPI `THREAD_MULTIPLE`.

For example, use this `profile-grid.json` for K3/G64:

```json
{
  "physical_k": 3,
  "record_confidence_head": true,
  "max_samples": 4096,
  "cells": [
    {"graph_batch_size": 64, "verifier_budget": 0},
    {"graph_batch_size": 64, "verifier_budget": 96},
    {"graph_batch_size": 64, "verifier_budget": 128},
    {"graph_batch_size": 64, "verifier_budget": 192}
  ]
}
```

This declares shapes, not measured prices. Merge the following fragment into
the complete working runtime YAML for this example; it is not a standalone
configuration:

```yaml
tensor_parallel_size: 8
pipeline_parallel_size: 1
context_parallel_size: 1
moe_expert_parallel_size: 8
enable_attention_dp: true
enable_iter_perf_stats: true
cuda_graph_config:
  enable_padding: true
  batch_sizes: [64]
speculative_config:
  decoding_type: DSpark
  max_draft_len: 3
  enable_confidence_scheduling: false
```

Change K and the grid/graph ladder together for another workload. Each additional
G needs its own native control and desired compact cells. Request/token limits
must retain the requested ladder: a silently clamped graph is not its requested
shape. `max_samples` must accommodate the run's raw replay records.

The JSONL workload must contain at least the largest global cohort, with a
nonempty integer `prompt_token_ids` list per line. Use the model's tokenizer and
ensure prompt plus output fits `max_seq_len`. Each cell repeats the same real
prompt cohort through prefill, generation and drain. Only decode steps with
exactly G requests on every rank and the declared graph key are samples;
bootstrap and shorter drain steps retain native execution.

## 2. Profile complete steps on GPUs

Run from the source checkout using its compatible installed runtime and the
normal allocation/container/MPI procedure. Set these Bash variables to your
paths and selected inputs; for the example, `K=3`, `RANKS=8` and `GRAPH_G=(64)`.
`SOURCE_ROOT` is the checkout used for execution; `PROFILE_YAML` is the complete
confidence-OFF configuration above. Output parents must exist and output paths
must be new.

```bash
mpirun -n "$RANKS" python -m tensorrt_llm.tools.verify_costs profile \
  --model "$MODEL_DIR" --config "$PROFILE_YAML" --grid "$GRID_JSON" \
  --physical-k "$K" --cost-scope whole-step --source-root "$SOURCE_ROOT" \
  --inputs "$INPUTS_JSONL" --output "$PROFILE_OUT" \
  --generations 3 --output-tokens 768 --timeout "$GENERATION_BUDGET_SECONDS"
```

`--timeout` is the shared generation-phase budget after construction and identity
checks, not the whole allocation budget. Reserve startup, generation, export and
natural shutdown in the outer launcher. Keep the full chosen output workload;
a short run may not meet the sample floors below.

The output contains `rank-R-record-N.json` for each cell/generation/rank, with
actual graph keys, per-step samples, request/sampler/KV joins, responses and
source/runtime/model/topology identity. Only completed runs with successful
shutdown can be exported as complete measurements.

## 3. Reduce all-rank observations on CPU

This reduction uses the standard library and does not import native TensorRT LLM:

```bash
python tensorrt_llm/tools/verify_costs.py export \
  --raw "$PROFILE_OUT"/rank-*-record-*.json --output "$SUMMARY_JSON" \
  --exclude-generations 1 --warmup-samples 2 \
  --minimum-generation-samples 12 --minimum-samples 24
```

The shell expands the raw-file glob. Generation 0 is excluded, as are the first
two eligible samples of each retained generation. Every retained generation
needs at least 12 exact samples, and every cell needs at least 24 pooled samples.
More samples in one generation do not waive another generation's floor.

The reducer joins replay ordinals and executor iterations across all ranks,
takes each joined rank maximum, then reports the retained median. It does not
average ranks or sum overlapping spans. It rejects missing/duplicate ranks or
cells, changed layouts, unequal identities, nonfinite/nonpositive times and
incomplete closure. Each cell must retain one consistent observed graph key;
different sequence-length/attention graph regimes need separate declared runs.

## 4. Export the runtime table on CPU

```bash
python tensorrt_llm/tools/dspark_cost_export.py \
  --input "$SUMMARY_JSON" --physical-k "$K" \
  --graph-batch-sizes "${GRAPH_G[@]}" --output "$COST_TABLE_JSON"
```

`GRAPH_G` is a Bash array of the resolved ladder, for example `(16 64)` for two
sizes. The exporter maps directly measured whole-step medians to schema-v2
`step_time_ms`, retaining their original source/fingerprint and summary hash.
It does not invent missing prices or refresh a historical identity. Use the
new table as `confidence_sps_table_path` in the normal runtime configuration.

See [Cost format, export and reuse](COST_EXPORT.md) for accepted formats and
structural checks. Costs do not calibrate the head, establish task accuracy or
grant economic admission.

## Advanced: graph-replay diagnostics

The same `profile` CLI defaults to `--cost-scope graph-replay`. That mode measures
graph replay including static input copies. Use it for
graph-level investigation, omitting the whole-step-only `--source-root` option.
Its replay medians are not serving prices and are rejected by the schema-v2 cost
exporter. GPU-forward timing and overlap `iterLatencyMS` are not substitutes for
the complete-loop boundary, and overlapping host/GPU spans must not be summed.

Supported/tested scope and optional developer controls are in the
[validation reference](CONFIDENCE_VALIDATION.md).
