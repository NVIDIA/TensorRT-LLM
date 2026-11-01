<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DSpark confidence scheduling: reproduce the DeepSeek-V4-Pro results

This guide reproduces the confidence-scheduling throughput gain on DeepSeek-V4-Pro and shows how
to reuse or generate the cost table it needs. The feature itself is documented in the
[DeepSeek-V4 README](README.md#dspark-confidence-scheduling-prototype).

## Results

8x B300, DEP8, FP8 KV cache, refill load (constant concurrency), 768 output tokens per request,
request-level output throughput (tokens/s). Static and dynamic arms alternate on the same node;
values are means over independent runs (n).

Short chat prompts (7-203 tokens, mean 48), `max_seq_len: 1280`:

| Batch size (global) | Best static DSpark | Confidence scheduling (K5) | Gain |
| --- | --- | --- | --- |
| 128 | K5: 12,096 (n=2) | 12,082 (n=2) | -0.1% |
| 256 | K5: 20,470 (n=4) | 20,430 (n=3) | -0.2% |
| 512 | K5: 33,936 (n=4) | 34,201 (n=4) | +0.8% |
| 1024 | K3: 46,212 (n=4) | 49,980 (n=4) | +8.2% |
| 2048 | K3: 61,814 (n=6) | 65,192 (n=6) | +5.5% |
| 4096 | K2: 67,603 (n=3) | 67,069 (n=2), deeper budgets | -0.8% |

Long prompts (896-1152 tokens, built by concatenating the same chat prompts), `max_seq_len: 2048`,
with a cost table profiled on these prompts:

| Batch size (global) | Best static DSpark | Confidence scheduling (K5) | Gain |
| --- | --- | --- | --- |
| 512 | K4: 23,576 (n=3; K5 22,856, n=5) | 25,666 (n=5) | +8.9% |
| 1024 | K3: 23,764 (n=5) | 25,546 (n=4) | +7.5% |

"Best static" is the fastest fixed draft length on the same configuration. The gain is largest
when the best static K is 3 (BS1024 and BS2048), so a K5 draft can be verified at roughly K3 cost
when confidence is low, and on workloads with high acceptance. At BS256 and below compact
verification saves little time, and the policy matches static K5. Reusing the short-prompt table
on the long-prompt workload gave the same result (25,674 at BS512, 25,553 at BS1024).

At BS4096 the best static K is 2. The default budgets verify at least half of the drafted tokens,
which costs more than static K2, and the policy lost 3.8% (65,049, n=3). A G512 row profiled with
`--tier-fractions 0.2 0.3 0.4 0.5` (one to 2.5 draft tokens per request,
[`dspark_artifacts/costs-b300-dep8-bs4096.json`](dspark_artifacts/costs-b300-dep8-bs4096.json))
brings it within 1% of static K2. At BS2048 the same deeper budgets compact too eagerly (63,879
vs 64,789 with the default budgets on the same node, n=2 and n=1), so keep the defaults there.

Per-iteration logs (`print_iter_log`) explain the BS512 rows. With short prompts, compact
verification makes decode-only steps 14% shorter (47.3 to 40.6 ms), but each step also yields 13%
fewer tokens (4.12 to 3.59 per request), so the two cancel. With long prompts the same steps are
17% shorter for 7.5% fewer tokens. Steps with prefill cost the same in both arms. A head that
ranks draft positions better would raise the short-prompt BS512 gain.

## 1. Requirements

- A TensorRT LLM build that includes DSpark confidence scheduling (PR #20078).
- A DeepSeek-V4-Pro checkpoint with the embedded DSpark drafter and trained confidence-head weights
  (block size 5).
- One node with 8 GPUs. The numbers above are from B300; other GPUs need their own cost table.

## 2. Runtime configuration

Static baseline, `static.yaml` (BS1024 shown). For another global batch size use
`max_batch_size: BS/8` and `batch_sizes` from 16 up to `BS/8` (BS512: `[16, 32, 64]`), and the best
static `max_draft_len` from the results table:

```yaml
tensor_parallel_size: 8
moe_expert_parallel_size: 8
enable_attention_dp: true
attention_dp_config:
  enable_balance: true
  batching_wait_iters: 10
  timeout_iters: 50
max_batch_size: 128            # rank-local
max_num_tokens: 4096
max_seq_len: 1280
enable_chunked_prefill: false
custom_tokenizer: deepseek_v4
kv_cache_config:
  dtype: fp8
  free_gpu_memory_fraction: 0.5
  enable_block_reuse: false
  tokens_per_block: 128
moe_config:
  backend: MEGAMOE_DEEPGEMM
cuda_graph_config:
  enable_padding: true
  batch_sizes: [16, 32, 64, 128]
prefill_cuda_graph_backend: breakable   # default token buckets
speculative_config:
  decoding_type: DSpark
  max_draft_len: 3
  block_size: 3
```

Dynamic, `dynamic.yaml`: the same file with this `speculative_config`:

```yaml
speculative_config:
  decoding_type: DSpark
  max_draft_len: 5
  block_size: 5
  enable_confidence_scheduling: true
  confidence_cost_table_path: /path/to/dspark-costs.json
  confidence_calibration_path: /path/to/calibration.json
  confidence_min_predicted_gain: 0.04
```

Why these settings (each was measured against the alternatives, static and dynamic alike):

| Setting | Alternative | Static throughput change |
| --- | --- | --- |
| `prefill_cuda_graph_backend: breakable`, default buckets | disabled (eager) | eager is about 5% slower |
| | piecewise | fails to compile for DeepSeek-V4 |
| | buckets capped at 1024 or 2048 | about 3.6% slower at BS1024 |
| `attention_dp_config` balance, wait 10 / timeout 50 | no balance | about 2x slower at BS512 |
| | wait 0 / timeout 60 | 5-9% slower |
| `MEGAMOE_DEEPGEMM` | `TRTLLM` + low-precision combine | about 11% slower |
| static `max_draft_len` 3 (BS1024) / 5 (BS512) | other K | K3: 48.1k vs K4 46.9k, K5 46.5k at BS1024 |

`enable_lm_head_tp_in_adp` and `sparse_attention_config.enable_heuristic_topk` were neutral.

## 3. Dataset

The benchmark uses 1,024 short chat prompts (7-203 input tokens, mean 48) repeated ten times with
fixed 768-token outputs, in the `trtllm-bench` JSONL format:

```json
{"task_id": 0, "input_ids": [14885, 21141, ...], "output_tokens": 768}
```

TODO: publish the prompt file (or its generation script) with the artifacts.

## 4. Cost table

The cost table holds measured decode step times per draft length `K`, rank-local graph batch size
`G` and number of verified rows `V`. Step costs depend on the GPU model, parallel layout and
kernels, so a table is tied to its setup.

**Reuse** a table generated on the same GPU model and parallel layout: point
`confidence_cost_table_path` at it. [`dspark_artifacts/costs-b300-dep8.json`](dspark_artifacts/costs-b300-dep8.json)
holds K3, K4 and K5 for 8x B300, DEP8 and graph batch sizes 16/32/64/128 (global batch size up to
1024), generated with the command below;
[`dspark_artifacts/costs-b300-dep8-bs2048.json`](dspark_artifacts/costs-b300-dep8-bs2048.json) holds
K4 and K5 for graph batch sizes 16 to 256 (BS2048), and
[`dspark_artifacts/costs-b300-dep8-bs4096.json`](dspark_artifacts/costs-b300-dep8-bs4096.json) adds
G512 to K5 with deeper budgets (BS4096). The matching calibration is
[`dspark_artifacts/calibration.json`](dspark_artifacts/calibration.json). At startup the runtime
verifies the table and stops with the command to regenerate if the file is missing, has no entry for `max_draft_len`, lacks a row for a
captured `cuda_graph_config.batch_sizes` entry, or was measured on another GPU model, world size,
TP/EP size or attention-DP setting. A different TensorRT LLM version only warns.

**Generate** one with a single command, using the runtime YAML you will serve with:

```bash
python -m tensorrt_llm.tools.dspark_profile_costs \
  --model /path/to/checkpoint --config dynamic.yaml --k 3 4 5 \
  --dataset /path/to/benchmark.jsonl --output dspark-costs.json
```

For each `K` the tool loads the model once and, for every captured `G`, runs a wave of
`G x 8` requests (another one while a tier has too few samples). During the wave's full decode
steps it cycles the number of verified rows through native and four compact tiers, holds each for
16 steps, and records the median whole-step time (drafting included). Attention-DP balancing is
turned off while profiling; it only delays prefill and does not change decode-step cost. It takes
about 10 minutes per `K` on 8x B300 for the BS1024 ladder. Run it again with another `--k`, or
with `--graph-batch-sizes` for a larger graph, to add those rows to the same file. A file that
covers several `K` serves any of them; the runtime picks the entry for `max_draft_len`. The BS4096
table is the BS2048 table's K5 rows plus a G512 row, added to a copy of that file with:

```bash
python -m tensorrt_llm.tools.dspark_profile_costs \
  --model /path/to/checkpoint --config static-bs4096.yaml --k 5 --graph-batch-sizes 512 \
  --tier-fractions 0.2 0.3 0.4 0.5 --min-samples 12 \
  --dataset /path/to/benchmark.jsonl --output costs-b300-dep8-bs4096.json
```

A G512 wave admits its requests slowly, so three waves gave 13 samples per budget; hence
`--min-samples 12`.

Example output for K5 on 8x B300 (V counts verified rows; native is `G x 6`):

```text
K=5 G=64:  native=50.50ms V224=41.75ms (-17.3%), V256=43.05ms (-14.7%), V288=45.60ms (-9.7%), V320=47.88ms (-5.2%)
K=5 G=128: native=73.48ms V448=57.14ms (-22.2%), V512=58.09ms (-20.9%), V576=63.65ms (-13.4%), V640=67.03ms (-8.8%)
```

Options: `--graph-batch-sizes` (default: the YAML's CUDA-graph ladder), `--tier-fractions`
(default `0.5 0.6 0.7 0.8`: on average 50-80% of the K drafted tokens verified per request; for K5
this gives the tiers used in the results; each must be at least 1/K, since every request verifies
at least one draft token), `--rounds` and `--hold-steps` (samples per tier),
`--max-waves` (default 3) and `--min-samples` (default 20). Use `--dataset` with prompts like the
ones you serve.

## 5. Calibration (optional)

`confidence_calibration_path` holds per-position temperatures, `{"sts_temperatures": [...]}`. A
vector fitted at the checkpoint's block size serves every smaller `K` (the runtime uses the first
`K` entries). Without it the raw sigmoid confidence is used.

## 6. Run the benchmark

Run every arm in its own process. Save as `sampler.yaml`:

```yaml
temperature: 0.0
top_k: 1
ignore_eos: true
detokenize: false
```

```bash
trtllm-bench --model deepseek-ai/DeepSeek-V4-Pro --model_path /path/to/checkpoint \
  throughput --backend pytorch --config static.yaml --sampler_options sampler.yaml \
  --dataset benchmark.jsonl --num_requests 10240 --concurrency 1024 --warmup 1024 \
  --tp 8 --ep 8 --max_batch_size 128 --max_num_tokens 4096 --max_seq_len 1280 \
  --kv_cache_free_gpu_mem_fraction 0.5 --disable_chunked_context --custom_tokenizer deepseek_v4 \
  --eos_id -1 --report_json static.json --request_json static-requests.json
```

Repeat with `--config dynamic.yaml`. For another batch size set `--concurrency BS` and
`--max_batch_size BS/8`; the results used 2,560 requests at BS128, 5,120 at BS256, 10,240 at BS512
and BS1024, 20,480 at BS2048 and 40,960 at BS4096.

Throughput is total output tokens over the span from the first request's start to the last
request's end:

```python
import json
reqs = json.load(open("static-requests.json"))
span_s = (max(r["end_timestamp"] for r in reqs) - min(r["start_timestamp"] for r in reqs)) / 1e9
print(sum(r["num_total_output_tokens"] for r in reqs) / span_s)
```

Run-to-run variation is about 1%, so compare means of at least three runs per arm, alternating
static and dynamic on the same node.

## 7. Troubleshooting

| Message | Fix |
| --- | --- |
| `requires confidence_cost_table_path` | Set the path or generate a table (section 4). |
| `has no costs for max_draft_len=K` | Add that K: rerun the tool with `--k K` and the same `--output`. |
| `no measurements for captured graph batch sizes [...]` | Use covered `batch_sizes`, or regenerate with your ladder. |
| `measured on a different setup (...)` | The table came from another GPU or parallel layout; regenerate on this one. |
| `measured with TensorRT-LLM X; running Y` (warning) | Usually fine; regenerate if gains shrink. |
| `has N temperatures but the confidence head scores K` | Use a calibration fitted at a K at least as large. |
