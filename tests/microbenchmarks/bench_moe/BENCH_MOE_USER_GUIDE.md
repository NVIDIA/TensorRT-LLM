<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# bench_moe User Guide

`bench_moe` is the TensorRT-LLM MoE microbenchmark under
`tests/microbenchmarks/bench_moe`. It times `ConfigurableMoE.forward` directly
with synthetic inputs, so you can search a user-defined configuration space and
find the best MoE setup for a fixed model shape and token workload. The search
space can cover backends, communication methods, parallel layouts, routing
shapes, and CUDA Graph settings without loading a HuggingFace checkpoint.
Advanced users can also control source-to-target communication volume and hot
workloads on each local expert. Current kernel analysis records CUDA kernel
statistics and raw samples when CUPTI is available; a planned per-forward
breakdown mode will make the time spent by each kernel in every MoE forward pass
directly visible.

The preferred invocation style is:

```bash
PYTHONPATH=tests/microbenchmarks:${PYTHONPATH:-} python3 -m bench_moe ...
```

When running from a script that already exports `PYTHONPATH` with
`$PWD/tests/microbenchmarks`, use `python3 -m bench_moe ...` directly.

## Quick Model

1. Choose the workload.

   Pick a built-in `--model` or pass explicit shape fields, then set the token
   workload with `--balanced_total_num_tokens`. For source-rank skew, use
   `--per_rank_num_tokens`.

2. Choose the search space.

   Use `--search backend`, `--search comm`, `--search parallel`, or
   combinations such as `--search backend comm`. Multi-value `--backend`,
   `--comm_method`, and `--parallel_mode` flags implicitly enable the matching
   search axis. Use `--max_configs` or `--time_budget_minutes` when the
   Cartesian product is too large.

3. Inspect winners and skips.

   The JSON report contains `rankings` grouped by `(num_tokens, parallel_mode)`.
   Unsupported or pruned candidates are emitted as `status="skipped"` with a
   `skip_reason`.

## Launch Rules

For single-rank sanity checks:

```bash
PYTHONPATH=tests/microbenchmarks:${PYTHONPATH:-} python3 -m bench_moe \
  --world_size 1 \
  --model mixtral_8x7b \
  --backend CUTLASS \
  --balanced_total_num_tokens 8 \
  --no_cuda_graph \
  --analysis none
```

For multi-rank runs, prefer an external launcher:

```bash
PYTHONPATH=tests/microbenchmarks:${PYTHONPATH:-} \
mpirun --allow-run-as-root --oversubscribe --bind-to none --map-by slot -np 4 \
  python3 -m bench_moe \
  --world_size 4 \
  ...
```

`--world_size` must match the number of ranks started by external `mpirun` or
`srun`. Avoid bare `python3 -m bench_moe --world_size N` on OCI / Slurm / Pyxis
systems because that path uses `MPI.COMM_SELF.Spawn`, which is commonly disabled
there. `--word_size` is a typo; the valid option is `--world_size`.

Do not wrap `bench_moe` with `trtllm-llmapi-launch`. That launcher runs the user
command on rank 0 and starts MGMN workers on the other ranks, which is correct
for LLM API / serving workloads but not for `bench_moe`. `bench_moe` requires
every MPI rank to execute the benchmark worker.

## Common Cases

### Case A: single-GPU sanity check

```bash
PYTHONPATH=tests/microbenchmarks:${PYTHONPATH:-} python3 -m bench_moe \
  --world_size 1 \
  --model mixtral_8x7b \
  --backend CUTLASS \
  --balanced_total_num_tokens 8 \
  --no_cuda_graph \
  --analysis none
```

Use this to validate imports, CUDA availability, backend construction, and basic
timing.

### Case B: 4-GPU DEP, search all backends

```bash
PYTHONPATH=tests/microbenchmarks:${PYTHONPATH:-} \
mpirun --allow-run-as-root --oversubscribe --bind-to none --map-by slot -np 4 \
  python3 -m bench_moe \
  --world_size 4 \
  --parallel_mode DEP \
  --model deepseek_v3 \
  --search backend \
  --balanced_total_num_tokens 64 128 256 512 \
  --output_file out/deepseek_v3_backend_search.json
```

`--search backend` without `--backend` expands to all backends. `--backend ALL`
is equivalent.

### Case C: fixed backend, search forced communication methods

```bash
PYTHONPATH=tests/microbenchmarks:${PYTHONPATH:-} \
mpirun --allow-run-as-root --oversubscribe --bind-to none --map-by slot -np 4 \
  python3 -m bench_moe \
  --world_size 4 \
  --model deepseek_v3 \
  --parallel_mode DEP \
  --backend TRTLLM \
  --search comm \
  --balanced_total_num_tokens 256 \
  --output_file out/deepseek_v3_comm_search.json
```

`--search comm` compares concrete forced communication strategies. It
intentionally excludes `AUTO`, because `AUTO` is an alias resolved by
TensorRT-LLM at runtime. To measure `AUTO`, run a separate single-candidate case
with `--comm_method AUTO`.

### Case D: search backend and forced communication together

```bash
PYTHONPATH=tests/microbenchmarks:${PYTHONPATH:-} \
mpirun --allow-run-as-root --oversubscribe --bind-to none --map-by slot -np 4 \
  python3 -m bench_moe \
  --world_size 4 \
  --model deepseek_v3 \
  --parallel_mode DEP \
  --search backend comm \
  --balanced_total_num_tokens 64 128 256 \
  --output_file out/deepseek_v3_backend_comm_full.json
```

To limit the Cartesian product, pass subsets such as:

```bash
--backend CUTLASS DEEPGEMM --comm_method NVLINK_ONE_SIDED DEEPEP
```

### Case E: receiver hotspot

```bash
PYTHONPATH=tests/microbenchmarks:${PYTHONPATH:-} \
mpirun --allow-run-as-root --oversubscribe --bind-to none --map-by slot -np 4 \
  python3 -m bench_moe \
  --world_size 4 \
  --parallel_mode DEP \
  --model deepseek_v3 \
  --backend TRTLLM \
  --balanced_total_num_tokens 256 \
  --comm_pattern receiver_hotspot,hotness=0.75,rank=0 \
  --output_file out/recv_hotspot.json
```

This sends 75% of selected slots to rank 0 and studies receiver-side pressure.

### Case F: source-rank token skew

```bash
PYTHONPATH=tests/microbenchmarks:${PYTHONPATH:-} \
mpirun --allow-run-as-root --oversubscribe --bind-to none --map-by slot -np 4 \
  python3 -m bench_moe \
  --world_size 4 \
  --model deepseek_v3 \
  --parallel_mode DEP \
  --backend TRTLLM \
  --balanced_total_num_tokens 896 \
  --per_rank_num_tokens 128 128 512 128 \
  --routing_dump_matrix \
  --output_file out/per_rank_skew.json
```

Use this to study max tokens per rank, padding, workspace use, and chunking when
one source rank has many more tokens.

### Case G: local-only communication baseline

```bash
PYTHONPATH=tests/microbenchmarks:${PYTHONPATH:-} \
mpirun --allow-run-as-root --oversubscribe --bind-to none --map-by slot -np 4 \
  python3 -m bench_moe \
  --world_size 4 \
  --model deepseek_v3 \
  --parallel_mode DEP \
  --backend TRTLLM \
  --balanced_total_num_tokens 256 \
  --routing_mode forced \
  --comm_pattern local_only \
  --routing_dump_matrix \
  --output_file out/local_only_forced.json
```

This forces every selected slot to stay on its source rank. It estimates the
near-zero cross-rank traffic lower bound, but it is a supplied-topk path and is
not directly equivalent to native fused routing.

### Case H: routing pattern file for arbitrary trace shapes

Use `--routing_pattern_file` when a trace shape cannot be represented by the
built-in `--comm_pattern` / `--expert_pattern` templates. Production routing is
often irregular: different source ranks may prefer different target ranks, and
each target rank may have a different local-expert hotspot. The CLI templates
are intentionally compact and mostly symmetric; a pattern file is the escape
hatch for exact trace replay.

```json
{
  "ep_size": 4,
  "experts_per_rank": 4,
  "slot_dispatch_matrix": [
    [5, 0, 1, 2],
    [1, 5, 1, 1],
    [0, 2, 4, 2],
    [2, 1, 1, 4]
  ],
  "expert_histogram": [
    [5, 1, 1, 1],
    [0, 6, 1, 1],
    [0, 0, 7, 0],
    [2, 2, 1, 4]
  ]
}
```

`slot_dispatch_matrix[src][dst]` is the number of selected slots sent from
source rank `src` to target rank `dst`. Each row sum must equal
`per_rank_num_tokens[src] * top_k`. `expert_histogram[dst][local_expert]`
describes how the slots received by target rank `dst` are distributed across
its local experts.

Key invariants:

- `slot_dispatch_matrix` has shape `[ep_size][ep_size]`. Each cell counts slots,
  where one `(token, selected_expert)` pair is one slot.
- `sum(slot_dispatch_matrix[src]) == per_rank_num_tokens[src] * top_k` for every
  source rank. The benchmark validates this during parsing.
- `sum(expert_histogram) == sum(per_rank_num_tokens) * top_k`. The benchmark
  also validates this global total.
- For an exact realization, each `expert_histogram[dst]` row sum should match
  the corresponding dispatch column sum. If the row sum does not match, the
  materializer treats the row as weights and `routing_control.actual` may report
  `max_abs_slot_error > 0`.

In the example above, `slot_dispatch_matrix[0] = [5, 0, 1, 2]` means source rank
0 sends five slots to itself, none to rank 1, one to rank 2, and two to rank 3.
`expert_histogram[2] = [0, 0, 7, 0]` is an extreme compute hotspot: all slots
received by target rank 2 land on local expert 2.

```bash
PYTHONPATH=tests/microbenchmarks:${PYTHONPATH:-} \
mpirun --allow-run-as-root --oversubscribe --bind-to none --map-by slot -np 4 \
  python3 -m bench_moe \
  --world_size 4 \
  --parallel_mode DEP \
  --num_experts 16 \
  --top_k 2 \
  --hidden_size 1024 \
  --intermediate_size 4096 \
  --quant FP8 \
  --routing_method RENORMALIZE \
  --backend CUTLASS \
  --balanced_total_num_tokens 16 \
  --routing_mode forced \
  --routing_pattern_file advanced_routing_pattern.json \
  --routing_dump_matrix \
  --output_file out/advanced_routing_pattern.json
```

To verify the run, compare `routing_control.actual.observed_slot_dispatch_matrix`
and `routing_control.actual.observed_expert_histogram` in the output JSON with
the matrices in the input file. In `--routing_mode forced`, the supplied-topk
path should reproduce them exactly. In `--routing_mode native`, check
`routing_realization.status`, `max_abs_slot_error`, and `max_relative_slot_error`
to see how closely the model-native routing path matched the requested shape.

### Case I: JSON config for a dashboard sweep

```json
{
  "model": "deepseek_v3",
  "workload": {
    "balanced_total_num_tokens": [64, 128, 256, 512],
    "routing_control": {
      "comm_pattern": "receiver_hotspot,hotness=0.75,rank=0",
      "expert_pattern": "balanced",
      "routing_dump_matrix": false,
      "seed": 42
    }
  },
  "search": {
    "backend": ["CUTLASS", "TRTLLM", "DEEPGEMM"],
    "parallel_mode": ["DEP", "TEP"],
    "comm_method": ["NVLINK_ONE_SIDED", "DEEPEP"]
  },
  "analysis": ["kernels"],
  "output_file": "out/deepseek_v3_dashboard.json"
}
```

```bash
PYTHONPATH=tests/microbenchmarks:${PYTHONPATH:-} \
mpirun --allow-run-as-root --oversubscribe --bind-to none --map-by slot -np 4 \
  python3 -m bench_moe \
  --world_size 4 \
  --config_file configs/deepseek_v3_dashboard.json
```

The config file is useful when sharing a fixed customer or dashboard sweep. Its
`search` block accepts `backend`, `parallel_mode`, and `comm_method`; multi-value
lists enable the matching search axis. Other runtime knobs belong at top level
or under `workload` / `routing_control`. If the same field is set in the config
and on the CLI, the explicit CLI value wins.

### Case J: 2-node Slurm, world_size 8

This form starts eight ranks across two nodes, four GPUs per node. Use it as the
benchmark command inside your own Slurm allocation or batch script. The
environment should already have TensorRT-LLM installed and should run from the
repository root. Do not use internal spawn, and do not wrap the benchmark with
`trtllm-llmapi-launch`.

```bash
PYTHONPATH=tests/microbenchmarks:${PYTHONPATH:-} \
srun --mpi=pmix --nodes=2 --ntasks=8 --ntasks-per-node=4 --gres=gpu:4 \
  python3 -m bench_moe \
  --world_size 8 \
  --model deepseek_v4_flash \
  --quant W4A8_MXFP4_MXFP8 \
  --balanced_total_num_tokens 1024 \
  --backend MEGAMOE_DEEPGEMM \
  --parallel_mode DEP \
  --comm_method AUTO \
  --warmup 1 \
  --iters 2 \
  --analysis none \
  --comm_pattern balanced_alltoall \
  --expert_pattern balanced \
  --per_candidate_timeout_s 120 \
  --checkpoint_every 0 \
  --output_file out/deepseek_v4_flash_w8.json
```

## Search Spaces

`--search` controls which runtime axes are expanded. The benchmark first builds
a Cartesian product, then filters candidates through backend capability checks,
parallel mapping checks, and forced-communication validity checks.

| `--search` | Expanded axes | Common use |
|---|---|---|
| `none` | No expansion; run only the base config. | Single-candidate sanity or rerun. |
| `backend` | Backends. If `--backend` is omitted, all backends are searched. | Find the fastest backend. |
| `comm` | Forced communication methods: `NVLINK_ONE_SIDED`, `NVLINK_TWO_SIDED`, `DEEPEP`, `DEEPEPLOWLATENCY`, and `ALLGATHER`. `AUTO` is filtered out. | Compare concrete communication strategies. |
| `backend comm` | Backend x forced-communication product; other axes use the base config. | Most common combined search. |
| `parallel` | Parallel layouts: `DEP`, `TEP`, `DTP`, `TTP`, or a subset. | Compare EP/TP layout effects. |
| `full` | Backend, parallel layout, communication, and CUDA Graph on/off. | Full runtime sweep; can be large. |

Passing more than one value to `--backend`, `--comm_method`, or
`--parallel_mode` automatically enables that search axis. `--comm_method AUTO`
is valid for a single candidate, but it is not part of a communication sweep.

Use `--max_configs N` to keep only the first `N` valid candidates after pruning.
Use `--time_budget_minutes M` to stop launching new candidates after a wall-clock
deadline. The currently running candidate is not interrupted.

Recommended search patterns:

| Question | Command pattern |
|---|---|
| Which backend is fastest? | `--search backend` |
| Which forced communication strategy is fastest under one backend? | `--backend TRTLLM --search comm` |
| Which backend x communication pair wins? | `--search backend comm` |
| Compare only a backend subset. | `--backend CUTLASS DEEPGEMM` |
| Compare only a communication subset. | `--comm_method NVLINK_ONE_SIDED DEEPEP` |
| Check whether parallel layout dominates. | `--search parallel --backend TRTLLM` or `--parallel_mode DEP TEP` |
| Keep a customer sweep small. | List allowed values explicitly in CLI or in the config-file `search` block. |

Prefer space-separated search axes, for example `--search backend comm parallel`.
Comma-separated spelling such as `--search backend,comm` also parses, but the
space-separated form is consistent with other multi-value options such as
`--balanced_total_num_tokens 64 128 256`.

## Workload and Routing Shapes

Workload has two layers: token count controls how many tokens enter MoE, while
routing control shapes how selected experts are distributed across ranks and
local experts. Without routing-control flags, the default is
`native + balanced_alltoall + balanced`.

Routing control is useful when you need to reproduce a trace shape, stress a
specific communication path, or isolate expert-hotspot behavior without changing
the model definition. It can either keep model-native routing (`native`) or
materialize supplied top-k inputs (`forced`) for exact control experiments.

| Option | Meaning |
|---|---|
| `--balanced_total_num_tokens` | Global token counts to sweep. Tokens are distributed as evenly as possible across ranks. |
| `--per_rank_num_tokens` | Explicit token count per rank. Length must equal `world_size`; mutually exclusive with `--balanced_total_num_tokens`. |
| `--comm_pattern` | Source-to-target slot traffic shape. Examples: `balanced_alltoall`, `receiver_hotspot`, `pair_hotspot`, `local_only`, `ring`, `random`. |
| `--expert_pattern` | Local expert distribution on each target rank. Examples: `balanced`, `hotspot,hotness=0.5`, `hotspot,active_experts=2`, `random`. |
| `--routing_pattern_file` | JSON file that fixes both dispatch and expert matrices. |
| `--routing_mode native` | Keep model-native logits/fused routing while projecting toward the requested shape. |
| `--routing_mode forced` | Supply top-k ids/scales directly for exact control experiments. |
| `--enable_perfect_router` | Opt into the lower-level MoE perfect-router override. Normal routing-control cases leave it off and use `bench_moe`-projected logits. |

### Comm Patterns

`--comm_pattern` controls source-rank to target-rank slot traffic. One slot is
one `(token, selected_expert)` pair.

| Pattern | Meaning | Example |
|---|---|---|
| `balanced_alltoall` | Each source rank sends selected slots as evenly as possible to all target ranks. This is the default fair baseline. | `--comm_pattern balanced_alltoall` |
| `receiver_hotspot` | Each source row sends a chosen fraction of slots to one target rank. Useful for receiver-side bandwidth and queueing stress. | `--comm_pattern receiver_hotspot,hotness=0.75,rank=0` |
| `pair_hotspot` | Only one source-to-target pair becomes hot. Useful for peer-link hotspots. | `--comm_pattern pair_hotspot,hotness=0.5,src=0,dst=1` |
| `local_only` | All slots stay on their source rank. Useful as a near-zero cross-rank communication baseline. | `--routing_mode forced --comm_pattern local_only` |
| `ring` | Source rank `i` sends to `(i + 1) % ep_size`. Useful for structured peer traffic. | `--routing_mode forced --comm_pattern ring` |
| `random` | Generate deterministic pseudo-random dispatch with `--routing_seed`. | `--comm_pattern random --routing_seed 42` |

### Expert Patterns

`--expert_pattern` controls how received slots are distributed over the local
experts of each target rank.

| Pattern | Meaning | Example |
|---|---|---|
| `balanced` | Slots are approximately balanced across local experts. | `--expert_pattern balanced` |
| `hotspot,hotness=...` | A chosen fraction of slots goes to one local expert on each target rank. | `--expert_pattern hotspot,hotness=0.5` |
| `hotspot,active_experts=...` | Only a chosen number of local experts receive all slots. | `--expert_pattern hotspot,active_experts=2` |
| `random` | Generate deterministic pseudo-random local expert histograms with `--routing_seed`. | `--expert_pattern random --routing_seed 42` |

Common routing-control combinations:

| Goal | Flags |
|---|---|
| Fair baseline | `--comm_pattern balanced_alltoall --expert_pattern balanced` |
| Receiver hotspot | `--comm_pattern receiver_hotspot,hotness=0.75,rank=0` |
| Pair hotspot | `--comm_pattern pair_hotspot,hotness=0.5,src=0,dst=1` |
| Local-only baseline | `--routing_mode forced --comm_pattern local_only` |
| Reject inexact native projection | `--routing_mode native --projection_policy reject --comm_pattern local_only` |

### Native, Forced, and Projection Policy

| Mode | Best for | Caveat |
|---|---|---|
| `--routing_mode native` | Keep model-native logits and fused routing while pushing routing toward the requested shape. | Some routing methods cannot exactly express a requested shape. The result may be `projected` or `rejected`. |
| `--routing_mode forced` | Exact top-k id/scale control for dispatch matrices, expert hotspots, and trace replay. | It bypasses fused scoring. Use it for controlled experiments, not as a direct equivalent of native fused routing. |

`--projection_policy` only applies to `native` mode. With `project`, the
benchmark runs the closest valid projection and records the deviation in
`routing_realization`. With `reject`, the candidate is skipped if the requested
shape cannot be represented exactly.

Routing-method projection capability:

| Routing method | Capability | Notes |
|---|---|---|
| Default | `exact_ids` | Expert ids can be exact; scales come from softmax. |
| Renormalize / RenormalizeNaive | `exact` | Expert ids and scales can be controlled exactly. |
| SigmoidRenorm | `exact_ids` | Expert ids can be exact; scales are limited by sigmoid behavior. |
| Llama4 | `top1_exact` | Only top-1 is exact; `top_k > 1` may project. |
| MiniMax2 | `exact_with_zero_bias` | Exact ids when score-correction bias is zero. |
| DeepSeekV3 | `projected_or_exact` | `n_group` / `topk_group` constraints can force projection. |
| SparseMixer | `unsupported` | Current routing-control projection treats it as projected. |

`--enable_perfect_router` is normally not needed for routing-control experiments:

| Routing mode | Pattern | Perfect router |
|---|---|---|
| `native` | `balanced_alltoall + balanced` | May be enabled for the lower-level perfect-router baseline. |
| `native` | Any custom comm or expert pattern | Keep off to avoid conflicting with projection. |
| `forced` | Any pattern | Keep off; the supplied-topk path already owns the routing inputs. |

In `forced` mode, `routing_control.actual.observation_source` is `plan_exact`
because the benchmark patches the kernel inputs to consume the materialized
plan. In `native` mode, it is `plan_simulation`: observed matrix fields are
deterministic re-materializations of the requested plan, not a runtime capture
of the kernel's final selected experts. Use `routing_realization.status` and
`max_abs_slot_error` to decide whether the projection is close enough.

## Reading Outputs

Rank 0 prints the benchmark header and result rows to stdout. With
`--output_file`, the full payload is written to JSON. By default the benchmark
also writes an Excel workbook next to the JSON.

The top-level JSON has this shape:

```json
{
  "benchmark": "bench_moe",
  "environment": {},
  "model": {},
  "search": {},
  "base_config": {},
  "results": [],
  "rankings": []
}
```

Start with `rankings` when you only need the winner:

```json
{
  "num_tokens": 256,
  "parallel_mode": "DEP",
  "best": {
    "backend": "TRTLLM",
    "requested_backend": "TRTLLM",
    "comm_method": "NVLinkOneSided",
    "cuda_graph": true,
    "score_ms": 0.8502,
    "status": "success"
  },
  "ranking": []
}
```

`best` is the lowest-scoring successful candidate in the same
`(num_tokens, parallel_mode)` group. Skipped and failed candidates remain in
`ranking`, so a sweep can explain both "what won" and "why other candidates did
not run."

| Field or sheet | Meaning |
|---|---|
| `rankings` | Best candidates grouped by `(num_tokens, parallel_mode)`. |
| `workload` | Token settings, per-rank token list, and routing-control request. |
| `requested_config` | User-requested or search-expanded candidate configuration. |
| `actual_config` | Backend, communication method, scheduler, EP/TP layout, and chunk count that actually ran. |
| `status` / `skip_reason` | Candidate outcome and the reason for pruning, skipping, timeout, or failure. |
| `latency_ms.score` | Ranking score based on slowest-rank latency with robust outlier handling. |
| `latency_ms.raw_score` | Unfiltered slowest-rank mean. |
| `latency_ms.iter_max_stats` | Mean, median, p90, min, max, and stdev over per-iteration slowest-rank latency. |
| `latency_ms.iter_max_outliers` | Outliers detected by robust scoring, including index, value, center, and modified z-score. |
| `kernel_breakdown` | CUDA kernel-name statistics when `--analysis kernels` succeeds. |
| `raw_data.forward_times_ms` | Per-rank forward latency samples for every timed iteration. |
| `raw_data.kernel_times_ms` | Per-rank per-kernel samples when CUPTI collection succeeds. |
| `routing_control` | Requested routing shape and observed/planned routing shape. |
| `all_results` | Excel sheet with one row per candidate. |
| `best_by_workload` | Excel sheet with winners per workload. |
| `status_summary` | Excel sheet with skip/failure aggregation. |

Excel workbook sheets:

| Sheet | Contents | Typical use |
|---|---|---|
| `all_results` | One row per candidate, including requested/actual config, score, raw score, outlier count, kernel count, and routing summary. | Sort, filter, and compare candidates quickly. |
| `best_by_workload` | One winner per `(num_tokens, parallel_mode)` group. | Find winners without reading nested JSON. |
| `status_summary` | Aggregated counts by token, backend, communication method, status, and skip reason. | Check whether a sweep was mostly pruned or failed. |
| `kernel_breakdown` | One row per `(candidate, category, kernel_name, rank)` with mean, median, p90, min, max, and stdev. | Inspect real CUDA kernel-name statistics per rank. |
| `raw_data` | One row per raw sample. `record_type=forward` stores per-forward samples; `record_type=kernel` stores per-kernel samples. | Investigate outliers and build custom plots. |
| `workload_<num_tokens>` | Candidate rows split by workload. | Local analysis for a single token count. |

Raw data boundary: `raw_data.forward_times_ms` is always recorded for successful
timed candidates. `raw_data.kernel_times_ms` is present only when
`--analysis kernels` is enabled and CUPTI/profiler collection succeeds. Older
JSON files that did not save raw data cannot be retroactively expanded into raw
per-iteration samples.

`routing_control.actual` is the main place to inspect routing-control quality:

```json
{
  "routing_path": "logits_projected",
  "routing_realization": {
    "status": "projected",
    "reason": "DeepSeekV3 grouped routing: ...",
    "max_abs_slot_error": 3,
    "max_relative_slot_error": 0.004
  },
  "enable_perfect_router": false,
  "max_num_tokens_per_rank": 256,
  "num_chunks_observed": 1,
  "observed_dispatch_matrix_summary": {
    "row_sums": [2048, 2048, 2048, 2048],
    "col_sums": [6144, 682, 683, 683],
    "off_diagonal_ratio": 0.75,
    "max_abs_slot_error": 3
  }
}
```

`routing_path` can be `logits_native`, `logits_projected`,
`supplied_topk_apply`, or `supplied_topk_run_moe`. `routing_realization.status`
can be `exact`, `projected`, `rejected`, or `forced_exact`. With
`--routing_dump_matrix`, the row also includes full requested and observed
dispatch/expert matrices.

## Pitfalls and Limitations

| Symptom | Cause / fix |
|---|---|
| `Custom shapes ... AUTO has no safe default` | Custom shapes must explicitly set `--routing_method`. |
| Forced communication candidate is skipped | Some communication methods do not make sense for non-DP layouts, MoE TP, or `world_size=1`. |
| `routing_realization.status="projected"` | The routing method cannot exactly express the requested shape. Inspect `max_abs_slot_error`. |
| `routing_realization.status="rejected"` | `--projection_policy reject` was used with an inexact native-routing request. The skipped row still keeps routing-control context. |
| Forced mode under TEP is not directly comparable with native | `forced` supplies top-k inputs and skips fused scoring; use it for controlled execution-path experiments. |
| Routing pattern file row-sum error | Each dispatch row must equal `per_rank_num_tokens[src] * top_k`, and `ep_size` must match runtime EP size. |
| Routing pattern file conflicts with comm/expert pattern | `--routing_pattern_file` already fixes both dispatch and expert matrices; do not combine it with non-default `--comm_pattern` or `--expert_pattern`. |
| `per_rank_num_tokens` length error | The list length must equal `world_size`; it is mutually exclusive with `--balanced_total_num_tokens`. |
| `phase_times_ms` is empty | Scheduler-side phase markers are not implemented yet; use latency and kernel breakdown. |
| Backend appears as skipped | Backend `can_implement()`, mapping, or communication capability gates rejected the candidate. Read `skip_reason`. |
| CUDA Graph timing has no kernel breakdown | CUPTI must initialize before CUDA context creation. If that fails, breakdown is unavailable. |
| Excel `raw_data` sheet only has headers | The JSON may come from an older run without raw data, or the run had no successful candidate. Rerun with current code to capture raw samples. |
| `--search comm` does not include `AUTO` | This is intentional: `AUTO` is a runtime alias, not a concrete forced strategy. Run a separate base case with `--comm_method AUTO` when you need that number. |
| Bare `python3 -m bench_moe --world_size 4` hangs or reports `MPI_ERR_SPAWN` | This path calls `MPI.COMM_SELF.Spawn`. Many Slurm/Pyxis systems disable MPI dynamic spawn. Use external `mpirun` or `srun --mpi=pmix`. |
| `trtllm-llmapi-launch` times out with `bench_moe` | That launcher runs the user process on rank 0 and MGMN workers on other ranks. `bench_moe` requires every rank to run the benchmark worker. |
| `--word_size` does not work | The correct option is `--world_size`. |

## Built-in Model Presets

| Model | Experts | `top_k` | Hidden | Intermediate | Default quant | Default routing |
|---|---:|---:|---:|---:|---|---|
| `qwen1.5_moe` | 60 | 4 | 2048 | 1408 | `FP8` | `RENORMALIZE` |
| `deepseek_v2_lite` | 64 | 6 | 2048 | 1408 | `FP8_BLOCK_SCALES` | `DEEPSEEK_V3` |
| `deepseek_v3` | 256 | 8 | 7168 | 2048 | `FP8_BLOCK_SCALES` | `DEEPSEEK_V3` |
| `kimi_k2` | 384 | 8 | 7168 | 2048 | `FP8_BLOCK_SCALES` | `DEEPSEEK_V3` |
| `deepseek_v4_pro` | 384 | 6 | 7168 | 3072 | pass `--quant` | `RENORMALIZE` |
| `deepseek_v4_flash` | 256 | 6 | 4096 | 2048 | pass `--quant` | `RENORMALIZE` |
| `mixtral_8x7b` | 8 | 2 | 4096 | 14336 | `FP8` | `RENORMALIZE` |
| `gpt_oss_120b` | 128 | 4 | 2880 | 2880 | `W4A8_MXFP4_MXFP8` | `RENORMALIZE` |

Custom shapes can be used instead of `--model`:

```bash
PYTHONPATH=tests/microbenchmarks:${PYTHONPATH:-} \
mpirun --allow-run-as-root --oversubscribe --bind-to none --map-by slot -np 4 \
  python3 -m bench_moe \
  --world_size 4 \
  --num_experts 384 \
  --top_k 6 \
  --hidden_size 7168 \
  --intermediate_size 3072 \
  --quant FP8_BLOCK_SCALES \
  --routing_method RENORMALIZE \
  --backend TRTLLM \
  --balanced_total_num_tokens 64 128
```

When `--model` is omitted, do not rely on `--routing_method AUTO`; custom shapes
have no model preset from which a safe routing default can be inferred.

## Backend, Parallel, and Communication Options

`--backend`, `--comm_method`, and `--parallel_mode` all accept multiple values.
Passing more than one value automatically enables the matching search axis.
`--backend ALL` is a compatibility spelling for "search all backends".

Parallel modes:

| Mode | `moe_ep_size` | `moe_tp_size` | `enable_attention_dp` | Meaning |
|---|---:|---:|---|---|
| `DEP` | `world_size` | 1 | true | Data-parallel attention with expert-parallel MoE. |
| `TEP` | `world_size` | 1 | false | Tensor-parallel attention with expert-parallel MoE. |
| `DTP` | 1 | `world_size` | true | Data-parallel attention with MoE tensor parallelism. |
| `TTP` | 1 | `world_size` | false | Tensor-parallel attention with MoE tensor parallelism. |
| `CUSTOM` | user-specified | user-specified | user-specified | Advanced layout through explicit mapping flags. |

`CUSTOM` is not part of the default `--search parallel` expansion. A bare
`--search parallel` expands only `DEP`, `TEP`, `DTP`, and `TTP`. Use `CUSTOM`
when you need an EP/TP split that is not covered by the four presets.

Users provide a custom layout through these flags:

- `--parallel_mode CUSTOM`
- `--moe_ep_size <N>`: MoE expert-parallel size.
- `--moe_tp_size <M>`: MoE tensor-parallel size.
- `--enable_attention_dp`: optional; if omitted, attention DP is disabled for
  `CUSTOM`.

The invariant is `moe_ep_size * moe_tp_size == world_size`. The benchmark
validates this before running a candidate. As a convenience, passing
`--moe_ep_size` or `--moe_tp_size` while the base `--parallel_mode` is still one
of `DEP` / `TEP` / `DTP` / `TTP` is treated as opting into `CUSTOM`, so output
metadata reports the real layout.

`CUSTOM` must be a single parallel mode. Do not combine it with preset modes in
one `--parallel_mode` list, because one scalar pair of `--moe_ep_size` /
`--moe_tp_size` cannot describe multiple layouts. For JSON configs, keep the
search space in the file if desired, but still provide the custom EP/TP sizes on
the CLI:

```json
{
  "model": "deepseek_v3",
  "workload": {
    "balanced_total_num_tokens": [256]
  },
  "search": {
    "parallel_mode": ["CUSTOM"]
  },
  "output_file": "out/custom_layout.json"
}
```

```bash
PYTHONPATH=tests/microbenchmarks:${PYTHONPATH:-} \
mpirun --allow-run-as-root --oversubscribe --bind-to none --map-by slot -np 4 \
  python3 -m bench_moe \
  --world_size 4 \
  --config_file configs/custom_layout.json \
  --moe_ep_size 2 \
  --moe_tp_size 2 \
  --enable_attention_dp
```

Equivalent CLI-only custom layout:

```bash
PYTHONPATH=tests/microbenchmarks:${PYTHONPATH:-} \
mpirun --allow-run-as-root --oversubscribe --bind-to none --map-by slot -np 4 \
  python3 -m bench_moe \
  --world_size 4 \
  --model deepseek_v3 \
  --parallel_mode CUSTOM \
  --moe_ep_size 2 \
  --moe_tp_size 2 \
  --enable_attention_dp \
  --backend TRTLLM \
  --balanced_total_num_tokens 256
```

Communication methods are `AUTO`, `NVLINK_ONE_SIDED`, `NVLINK_TWO_SIDED`,
`DEEPEP`, `DEEPEPLOWLATENCY`, and `ALLGATHER`. For a single candidate, `AUTO`
lets TensorRT-LLM choose the concrete method. In a communication sweep,
`bench_moe` uses forced concrete methods and filters `AUTO` out to avoid
duplicating an alias.

## Timing, Output, and Limits

| Option | Default | Meaning |
|---|---|---|
| `--no_cuda_graph` | not set | Use eager timing instead of CUDA Graph timing. |
| `--warmup` | `1` | Warmup iterations before timed iterations. |
| `--iters` | `12` | Timed iterations. At least 8 samples enables MAD-based outlier filtering. |
| `--fast_autotune` | `false` | Shorten autotune repeat/warmup for quick debugging. |
| `--per_candidate_timeout_s` | `0.0` | Hard wall-clock timeout per candidate; useful for NCCL/CUDA hangs. |
| `--analysis` | `kernels` | Use `kernels` for CUPTI kernel breakdown or `none` for latency only. |
| `--dtype` | `bfloat16` | Activation dtype. Common values are `bfloat16` and `float16`. |
| `--output_file` | unset | Write the final JSON report and determine the default Excel path. |
| `--analysis_workbook_file` | `<output_file>.analysis.xlsx` | Write the Excel workbook. |
| `--resume_from` | unset | Read an existing JSON report and skip completed terminal candidates. |
| `--checkpoint_every` | `1` | Write a JSON checkpoint after every N newly completed candidates. Use `0` for final-only output. |
| `--max_configs N` | disabled | After pruning, keep only the first N valid candidates. |
| `--time_budget_minutes M` | disabled | Stop launching new candidates after the deadline. |

Autotune is an untimed pre-pass. The benchmark runs autotune before formal
timing to populate kernel caches; reported latency does not include the autotune
pass.

`--per_candidate_timeout_s` is a hard guard around one candidate. If a candidate
hangs in NCCL or CUDA, the watchdog terminates the process; use `--resume_from`
to rerun missing candidates from the last checkpoint. `--max_configs` and
`--time_budget_minutes` are sweep-level limiters and do not interrupt a candidate
that is already running.

Example bounded sweep:

```bash
PYTHONPATH=tests/microbenchmarks:${PYTHONPATH:-} \
mpirun --allow-run-as-root --oversubscribe --bind-to none --map-by slot -np 4 \
  python3 -m bench_moe \
  --world_size 4 \
  --model deepseek_v3 \
  --parallel_mode DEP \
  --search backend comm \
  --balanced_total_num_tokens 128 256 \
  --max_configs 32 \
  --time_budget_minutes 30 \
  --output_file out/bounded_sweep.json
```
