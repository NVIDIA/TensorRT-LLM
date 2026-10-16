<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DSpark confidence scheduling and reproduction

Use this guide to run confidence scheduling with existing artifacts or generate
your own. Ordinary inference and benchmarking use the normal TensorRT LLM APIs,
runtime YAML and CLI; no reproduction wrapper or particular cluster is required.

Confidence scheduling is experimental and **off by default**. When enabled, it
still proposes the full draft block and uses normal target verification and
sampling; it may choose a smaller measured verification layout.

## 1. Choose your inputs and runtime

Start from the [DeepSeek-V4 Quick Start](README.md#quick-start) and the ordinary
[installation/build instructions](../../../../docs/source/installation/build-from-source.md).
Use a compatible installed runtime containing these tools; an unrelated Python
change does not by itself require rebuilding native TensorRT LLM.

You need a supported embedded DSpark checkpoint with **trained confidence-head
weights**, enough GPU/KV capacity, and a workload prepared with its tokenizer.
Temperature calibration does not train or replace those weights.

Choose these inputs once and use them consistently:

| Input | Meaning |
| --- | --- |
| Physical `K` | `speculative_config.max_draft_len` in YAML; `--physical-k` in offline/helper commands. It must match the checkpoint/runtime. Dynamic compact scheduling needs K > 1; K1 resolves to static. |
| Global `BS` (offline/helper) | `--batch-size` selects the helper's fixed request cohort. For that cohort with attention DP, rank-local `G = BS / attention-DP size`; otherwise G is the global batch. |
| Ordinary benchmark load | `--max_batch_size` is rank-local with attention DP; `--concurrency` is global concurrent load and `--num_requests` is total submitted work. These are separate controls, not aliases for helper BS. |
| Graph ladder | `cuda_graph_config.batch_sizes` contains rank-local G values covered by the physical-K-specific cost table. Runtime request/token/KV limits must accommodate them. |
| Model and data | Your checkpoint and workload paths; keep precision, backend, sampling and output work fixed for comparisons. |
| Artifacts and outputs | A supplied or self-generated cost table; optional temperature JSON; fresh writable output paths. Inputs can remain read-only. |

The interfaces accept the previously used BS128/256/512/1024/1536/2048 grid and
smaller points when supported by the actual graph/K/cost/capacity configuration.
This is not a validation claim for every point. The example below uses K3 and
G64 on eight attention-DP ranks; adjust the declared inputs for your workload.
Other route restrictions are in the [reference](CONFIDENCE_VALIDATION.md#supported-routes-and-configuration).

## 2. Reuse artifacts and enable confidence in normal YAML

Start with your working DeepSeek-V4 runtime YAML. Keep its model, topology,
precision, KV and backend settings. For this example, use padded graphs and
replace its speculative configuration with:

```yaml
cuda_graph_config:
  enable_padding: true
  batch_sizes: [64]
speculative_config:
  decoding_type: DSpark
  max_draft_len: 3
  enable_confidence_scheduling: true
  confidence_sps_table_path: /path/to/artifacts/costs.json
```

Save this as `dynamic.yaml`. For `static.yaml`, keep the same numerical/runtime
settings but set `enable_confidence_scheduling: false` and **omit all
confidence-only artifact, freshness, mixed-evidence and fused-scheduler fields**.
Do not replace the checkpoint's full model configuration with this YAML fragment.

The cost table is required for dynamic scheduling. It must contain finite
positive native/compact measurements compatible with physical K, captured graph
sizes and layouts. Default reuse allows a configured subset of a measured graph
ladder; any priced drain tail must still be measured and captured. Missing
coverage is an error, not a silently timed static result.

Temperature calibration is optional. Omit `confidence_sts_path` to use raw
confidence probabilities, equivalent to unit temperatures. To reuse a fitted
vector, add `confidence_sts_path: /path/to/artifacts/calibration.json` under
`speculative_config`.

By default, structurally valid artifacts can be reused across source revisions
and nodes with warnings that predictions may be stale. Their original provenance
is retained; different kernels or hardware can change prediction quality and
performance. New profiling is your choice, not a prerequisite to every run.
Strict freshness/workload-admission validation is an
[optional advanced mode](CONFIDENCE_VALIDATION.md#optional-strict-freshness-and-admission).

## 3. Run ordinary inference or benchmarks

Pass `dynamic.yaml` through the normal `LLM` configuration or
`trtllm-serve --config` interface described in [Serving](README.md#serving).
No calibration-collection/comparison helper is required for ordinary use.

For benchmarking, follow [DeepSeek-V4 Benchmarking](README.md#benchmarking) and
the [benchmarking guide](../../../../docs/source/developer-guide/perf-benchmarking.md#preparing-a-dataset)
to prepare a standard benchmark dataset. It uses `task_id`, `input_ids` or
`prompt`, and `output_tokens` on each JSONL line.

For an explicit greedy, fixed-output comparison, save this as `sampler.yaml`:

```yaml
temperature: 0.0
top_k: 1
ignore_eos: true
```

Run each arm in its own process/model lifetime, changing only the configuration
and report paths:

```bash
trtllm-bench --model <MODEL_ID> --model_path /path/to/checkpoint throughput \
  --tp 8 --ep 8 --dataset /path/to/benchmark.jsonl \
  --max_batch_size 64 --max_num_tokens 8192 --max_seq_len 4096 \
  --concurrency 512 --num_requests 512 --warmup 512 \
  --sampler_options /path/to/sampler.yaml --config /path/to/static.yaml \
  --report_json /path/to/results/static.json

trtllm-bench --model <MODEL_ID> --model_path /path/to/checkpoint throughput \
  --tp 8 --ep 8 --dataset /path/to/benchmark.jsonl \
  --max_batch_size 64 --max_num_tokens 8192 --max_seq_len 4096 \
  --concurrency 512 --num_requests 512 --warmup 512 \
  --sampler_options /path/to/sampler.yaml --config /path/to/dynamic.yaml \
  --report_json /path/to/results/dynamic.json
```

Both YAML files must include your intended `enable_attention_dp: true` and
matching topology for this example. With attention DP, `--max_batch_size` is
rank-local whereas `--concurrency` is global. Match the dataset output lengths
and ensure prompt plus output fits the sequence limit. Explicit CLI settings
override YAML values. Use your site's normal allocation/container procedure;
do not launch multiple competing copies on the same GPUs.

This ordinary confidence benchmark recipe has been checked against the public
source interfaces, **not GPU-validated as a new recipe**. Standard benchmark
reports use their own load/timing protocol and do not establish compact routing
by themselves. For a controlled routing-checked comparison, use the optional
helper below. Neither protocol is interchangeable with historical plateau TPS.

## 4. Generate new artifacts when useful

You can regenerate either artifact independently. Confidence scheduling does not
require temperature fitting before cost profiling or benchmarking.

### Whole-step costs

Follow [Offline cost profiling](COST_PROFILING.md) and
[Cost export](COST_EXPORT.md). Declare native V0 and legal compact cells at the
chosen K and rank-local graph ladder, enable `enable_iter_perf_stats: true` in
the profiling YAML, and collect with `--cost-scope whole-step`. Reduce the
completed all-rank observations, then export the schema-v2 runtime table with
`dspark_cost_export.py`.

Use directly measured complete-loop costs; graph-replay medians are diagnostics,
not serving prices. The current whole-step exporter supports attention-DP fanout
or a single rank, not multi-rank attention-DP-off export. The linked guides give
the grid, sample floors, output files and exact commands.

### Optional temperature calibration

The paired collector uses a separate tokenized input format:
each JSONL line has a nonempty integer `prompt_token_ids` list. Set the shell
variables below to the inputs chosen in step 1; `RUNTIME_YAML` is the matching
runtime YAML with overlap scheduling enabled. `SOURCE_ROOT` is your checkout,
`RUN_ROOT` is private writable storage, and the timeout reserves the full
startup/model/shutdown budget.

```bash
mpirun -n "$RANKS" python tensorrt_llm/tools/dspark_reproduction.py run \
  --mode collect --model "$MODEL_DIR" --config "$RUNTIME_YAML" \
  --inputs "$INPUTS_JSONL" --source-root "$SOURCE_ROOT" \
  --physical-k "$K" --batch-size "$BS" --output-tokens "$OUTPUT_TOKENS" \
  --timeout "$FULL_BUDGET_SECONDS" --output "$RUN_ROOT/calibration-observations"

python tensorrt_llm/tools/dspark_calibration.py \
  --shards "$RUN_ROOT"/calibration-observations/rank-*-pairs.pt \
  --physical-k "$K" --output "$RUN_ROOT/calibration.json"
```

The collector runs native full-K verification and joins real confidence logits
to the later accepted-prefix labels for the same proposal and request lifetime.
The fitter requires at least 1000 genuine paired rows by default and produces
per-position temperatures. Collection/fitting is not head training or task
accuracy evaluation. See the [pairing and output contract](CONFIDENCE_VALIDATION.md#paired-collection-and-developer-checks).

## 5. Optional controlled comparison and routing checks

`dspark_reproduction.py` is for paired collection and controlled fixed-cohort
experiments. It is not a required serving launcher. Its `compare` command
consumes its **own** `rank-N-run.json` records, not `trtllm-bench` reports.
To use it, run its own independent static and dynamic lifetimes with the same
inputs, configuration, output work, warm-up count and budget:

```bash
mpirun -n "$RANKS" python tensorrt_llm/tools/dspark_reproduction.py run \
  --mode static --model "$MODEL_DIR" --config "$RUNTIME_YAML" \
  --inputs "$INPUTS_JSONL" --source-root "$SOURCE_ROOT" \
  --physical-k "$K" --batch-size "$BS" --output-tokens "$OUTPUT_TOKENS" \
  --timeout "$FULL_BUDGET_SECONDS" --output "$RUN_ROOT/static"

mpirun -n "$RANKS" python tensorrt_llm/tools/dspark_reproduction.py run \
  --mode dynamic --model "$MODEL_DIR" --config "$RUNTIME_YAML" \
  --inputs "$INPUTS_JSONL" --source-root "$SOURCE_ROOT" \
  --physical-k "$K" --batch-size "$BS" --output-tokens "$OUTPUT_TOKENS" \
  --cost-table "$COST_TABLE_JSON" \
  --timeout "$FULL_BUDGET_SECONDS" --output "$RUN_ROOT/dynamic"

python tensorrt_llm/tools/dspark_reproduction.py compare \
  --runs "$RUN_ROOT/static/rank-0-run.json" "$RUN_ROOT/dynamic/rank-0-run.json" \
  --output "$RUN_ROOT/comparison.json"
```

Calibration is deliberately omitted above; optionally add
`--calibration "$CALIBRATION_JSON"` to the dynamic command. The helper records
resolved/runtime enablement, native/compact replay, source/artifact identity,
outputs and shutdown. An enabled policy may legally choose native throughout;
`compare` refuses to call that a reproduced compact gain.

Its TPS covers fixed-cohort enqueue through final response, including prefill
and drain. Use balanced fresh-lifetime repetitions for a stable estimate and
equal static/dynamic record counts for `compare`. Token equality is reported
diagnostically, not substituted for task-accuracy tests.

See [Validation and developer reference](CONFIDENCE_VALIDATION.md) for the
completed evidence scopes, advanced mixed/freshness contracts and optional tests.
