<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DSpark confidence scheduling and reproduction

Use this guide to run with supplied artifacts or generate your own. Ordinary
inference and benchmarking use the normal TensorRT LLM runtime YAML, APIs and
CLI; the optional reproduction helper is not a serving prerequisite.

Confidence scheduling is experimental and **off by default**. Explicitly enabling
it retains full physical draft proposals and normal target verification and
sampling, while allowing a smaller measured verification layout.

## 1. Prepare the runtime and inputs

Follow the [DeepSeek-V4 Quick Start](README.md#quick-start) and the ordinary
[installation/build instructions](../../../../docs/source/installation/build-from-source.md).
Use a compatible runtime and a supported embedded DSpark checkpoint containing
trained confidence-head weights. Temperature calibration does not train those
weights. No particular cluster is required.

Choose the following inputs for your workload:

| Input | Meaning |
| --- | --- |
| Physical `K` | YAML `speculative_config.max_draft_len`, and offline/helper `--physical-k`. It must match the loaded checkpoint/runtime. Compact scheduling needs K > 1; K1 resolves to feature-off static execution. |
| Graph size `G` | Rank-local request count in `cuda_graph_config.batch_sizes`. Captured graph layouts and the cost table must cover it; request, token and KV limits must accommodate it. |
| Verifier budget `V` | Total verifier rows in a captured rank-local compact layout: `G <= V < G*(K+1)`. V0 is a sentinel for native full-width `G*(K+1)` verification, not zero tokens. |
| Helper global `BS` | `--batch-size` is a fixed global request cohort, not a graph setting. With attention DP, G = BS / attention-DP size; otherwise G = BS. |
| Ordinary benchmark load | With attention DP, `--max_batch_size` is rank-local. `--concurrency` limits global simultaneous requests; `--num_requests` caps requests read from the dataset, with 0 meaning the dataset length. `--warmup` counts warm-up requests, not iterations or helper cohorts. |
| Model and data | Checkpoint and workload paths, using the checkpoint's tokenizer. Keep topology, precision, backend, sampling and output work fixed when comparing runs. |
| Artifacts | A compatible cost table is required for dynamic scheduling. Temperature calibration is optional. |

The examples use K3, G64 and eight attention-DP ranks. These are example inputs,
not universal requirements. See [supported routes and configuration](CONFIDENCE_VALIDATION.md#supported-routes-and-configuration)
for runtime and helper limits.

Observations can contain prompt and output tokens. Use a new private writable
output directory; keep inputs read-only if desired. Offline/helper commands
require fresh output paths with existing parent directories and do not overwrite
them.

## 2. Reuse artifacts in normal runtime YAML

Start with a complete working DeepSeek-V4 **runtime YAML**. Keep its topology,
precision, KV, backend and capacity settings, and merge this example into it,
replacing its existing speculative configuration:

```yaml
enable_attention_dp: true
cuda_graph_config:
  enable_padding: true
  batch_sizes: [64]
speculative_config:
  decoding_type: DSpark
  max_draft_len: 3
  enable_confidence_scheduling: true
  confidence_sps_table_path: /path/to/artifacts/costs.json
```

Save the merged runtime configuration as `dynamic.yaml`. This fragment is not a
complete configuration and does not replace the checkpoint's `config.json`.

Create `static.yaml` from the same runtime configuration, set
`enable_confidence_scheduling: false`, and omit all confidence-only artifact,
freshness, mixed-evidence and fused-scheduler fields. Keep the physical K and
numerical/runtime settings unchanged.

The cost table must contain finite positive native V0 measurements and the
compact cells it offers, compatible with physical K and graph/layout coverage.
Default reuse permits a configured subset of a measured graph ladder; any priced
drain tail must still be measured and captured. An enabled policy may legally
choose native throughout, but missing/incompatible coverage is not a compact
execution or a reproduced gain.

Structurally valid artifacts can be reused across source revisions and nodes
with warnings that freshness and performance economics are unchecked. Original
artifact provenance is retained; a different destination can change prediction
quality and performance. New profiling is optional when reusing compatible
artifacts. Strict freshness and economic admission are an
[optional advanced mode](CONFIDENCE_VALIDATION.md#optional-strict-freshness-and-admission).

## 3. Run ordinary inference or benchmarks

Use `dynamic.yaml` through the normal `LLM` configuration or
`trtllm-serve --config` interface in [Serving](README.md#serving).

For benchmarking, follow [DeepSeek-V4 Benchmarking](README.md#benchmarking) and
the [dataset preparation guide](../../../../docs/source/developer-guide/perf-benchmarking.md#preparing-a-dataset).
The standard benchmark JSONL format uses `task_id`, `input_ids` or `prompt`, and
`output_tokens`. Prepare at least as many requests as you intend to measure;
`--num_requests` does not repeat a shorter dataset.

For a greedy, fixed-output comparison, save this as `sampler.yaml`:

```yaml
temperature: 0.0
top_k: 1
ignore_eos: true
```

Run each arm in a separate process/model lifetime, changing only the config and
report paths:

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

Both runtime files use attention DP for this example. Match dataset output
lengths and ensure each prompt plus output fits the sequence limit. Explicit CLI
settings override conflicting YAML values. Use the normal allocation/container
procedure; `trtllm-bench` handles process launch, so do not add an MPI wrapper.

Standard benchmark reports use the benchmark's load and timing protocol; they do
not by themselves establish compact routing. The optional helper below runs its
own routing-checked lifetimes with a different timing boundary. See the
[validation reference](CONFIDENCE_VALIDATION.md#validation-scope) for the tested
scope and interpretation of results.

## 4. Generate artifacts when needed

You can generate costs, temperature calibration, or both. Neither requires
generating the other first.

### Whole-step cost table

Follow the complete [cost-generation recipe](COST_PROFILING.md): configure a
physical-K-specific native/compact grid, profile complete executor steps, reduce
all-rank observations, and export a schema-v2 runtime table. Graph-replay timing
is a separate diagnostic, not a serving price. The
[cost format/export reference](COST_EXPORT.md) describes structural reuse checks.

### Optional temperature calibration

Without calibration, the head uses raw sigmoid confidence probabilities
(unit temperatures). Omit both `confidence_sts_path` in runtime YAML and
`--calibration` in helper commands to select this behavior. Omitting only the
CLI option does not remove a calibration path already present in YAML.

To generate temperatures, use the paired collector followed by the CPU fitter.
The offline/helper input format is separate from the benchmark format: each
JSONL line contains a nonempty integer `prompt_token_ids` list.

Set `MODEL_DIR`, `SOURCE_ROOT`, `K`, `RANKS`, `BS` and output work to your chosen
inputs. `BASELINE_YAML` must be a complete matching, confidence-OFF runtime YAML
with no confidence artifact/freshness/mixed/fused fields, explicit topology and
capacity, padded graphs, and overlap scheduling enabled. `RUN_ROOT` is a new
writable directory; `FULL_BUDGET_SECONDS` covers startup, model work and shutdown.

```bash
mpirun -n "$RANKS" python tensorrt_llm/tools/dspark_reproduction.py run \
  --mode collect --model "$MODEL_DIR" --config "$BASELINE_YAML" \
  --inputs "$INPUTS_JSONL" --source-root "$SOURCE_ROOT" \
  --physical-k "$K" --batch-size "$BS" --output-tokens "$OUTPUT_TOKENS" \
  --timeout "$FULL_BUDGET_SECONDS" --output "$RUN_ROOT/calibration-observations"

python tensorrt_llm/tools/dspark_calibration.py \
  --shards "$RUN_ROOT"/calibration-observations/rank-*-pairs.pt \
  --physical-k "$K" --output "$RUN_ROOT/calibration.json"
```

Collection runs full-K verification and pairs confidence logits with the later
accepted-prefix labels for the same proposal/request lifetime. The fitter needs
at least 1,000 genuine paired rows by default and produces per-position
temperatures, input hashes and before/after ECE. It is not head training or
task-accuracy evaluation. The [pairing contract](CONFIDENCE_VALIDATION.md#paired-collection-and-developer-checks)
describes the observations.

To use the result, add `confidence_sts_path: /path/to/calibration.json` under
`speculative_config` in the dynamic runtime YAML, or add
`--calibration "$RUN_ROOT/calibration.json"` to a dynamic helper command.

## 5. Optional fixed-cohort comparison and routing checks

`dspark_reproduction.py` is for paired collection and controlled experiments,
not a required inference launcher. Its `compare` command consumes its own
`rank-N-run.json` records, not `trtllm-bench` reports.

Use the clean `BASELINE_YAML` and offline input format described above. The helper
requires TP equal to the MPI world size, PP=CP=1 and enough rank-local capacity
for the global cohort. With attention DP, BS must divide evenly across ranks.
Run independent static and dynamic lifetimes with the same inputs, output work,
warm-up count and full budget:

```bash
mpirun -n "$RANKS" python tensorrt_llm/tools/dspark_reproduction.py run \
  --mode static --model "$MODEL_DIR" --config "$BASELINE_YAML" \
  --inputs "$INPUTS_JSONL" --source-root "$SOURCE_ROOT" \
  --physical-k "$K" --batch-size "$BS" --output-tokens "$OUTPUT_TOKENS" \
  --warmup-cohorts 1 --timeout "$FULL_BUDGET_SECONDS" --output "$RUN_ROOT/static"

mpirun -n "$RANKS" python tensorrt_llm/tools/dspark_reproduction.py run \
  --mode dynamic --model "$MODEL_DIR" --config "$BASELINE_YAML" \
  --inputs "$INPUTS_JSONL" --source-root "$SOURCE_ROOT" \
  --physical-k "$K" --batch-size "$BS" --output-tokens "$OUTPUT_TOKENS" \
  --cost-table "$COST_TABLE_JSON" \
  --warmup-cohorts 1 --timeout "$FULL_BUDGET_SECONDS" --output "$RUN_ROOT/dynamic"

python tensorrt_llm/tools/dspark_reproduction.py compare \
  --runs "$RUN_ROOT/static/rank-0-run.json" "$RUN_ROOT/dynamic/rank-0-run.json" \
  --output "$RUN_ROOT/comparison.json"
```

The helper records resolved/runtime enablement, native/compact replay, outputs,
source/artifact identity and shutdown. `compare` requires actual compact routing
for a compact-gain claim; an enabled all-native run is not reported as one.
Its TPS covers fixed-cohort enqueue through final response, including prefill
and drain. Use balanced fresh-lifetime repetitions and equal static/dynamic
record counts. Do not pool these percentages with standard benchmark or plateau
measurements. Token equality is diagnostic, not a task-accuracy test.

Advanced configuration and optional developer checks are in the
[validation reference](CONFIDENCE_VALIDATION.md).
