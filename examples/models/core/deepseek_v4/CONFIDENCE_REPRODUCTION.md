<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DSpark confidence scheduling: reproduce the DeepSeek-V4-Pro results

This guide reproduces the confidence-scheduling throughput gain on DeepSeek-V4-Pro with prompts
from four public datasets, and shows how to reuse or generate the cost table the feature needs. The
feature itself is documented in the [DeepSeek-V4 README](README.md#dspark-confidence-scheduling-prototype).
The scripts are in [`dspark_benchmark/`](dspark_benchmark) and the B300 cost table and calibration
in [`dspark_artifacts/`](dspark_artifacts).

## Results

One node of 8x B300, attention data parallel with 8-way expert parallel, FP8 KV cache, TensorRT LLM
main (2d1a7d1) with PR #20078. Each dataset was sent at constant concurrency equal to the batch size,
from 64 to 2048 requests. Decoding is greedy, and requests stop at their end-of-sequence token or
after 1,024 output tokens.

Confidence scheduling (maximum draft length K3 to K5) is compared with fixed DSpark draft lengths
(K1 to K5) at equal per-user speed. For each method, the best points over all batch sizes and draft
lengths form a line, and the table shows where the confidence-scheduling line is furthest above the
fixed one (section 6):

| Dataset | Largest gain at equal per-user speed | At tok/s/user | Fixed → confidence, tok/s/GPU |
| --- | --- | --- | --- |
| SPEED-Bench throughput_1k | **+21.9%** | 48.9 | 1,469 → 1,791 |
| ShareGPT | **+19.9%** | 63.8 | 1,948 → 2,335 |
| Arena-Hard v0.1 | **+12.3%** | 73.2 | 2,892 → 3,248 |
| GSM8K 5-shot | +1.5% | 38.9 | 595 → 604 |

The best fixed draft length against the best confidence-scheduling setting at the same batch size:

| Batch size | 64 | 128 | 256 | 384 | 512 | 768 | 1024 | 2048 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPEED-Bench | +1.7% | -0.6% | +6.1% | +4.7% | +2.7% | +3.4% | +2.8% | -3.7% |
| ShareGPT | +0.6% | -0.3% | +2.4% | +3.1% | +4.2% | +3.1% | +3.8% | -1.0% |
| Arena-Hard | +0.7% | +0.2% | -0.2% | +1.3% | +1.5% | +1.2% | +4.3% | +1.5% |
| GSM8K | -0.2% | +0.8% | -0.8% | | -0.3% | | -0.3% | -0.3% |

The equal-speed gains are larger than the same-batch-size gains. Where confidence scheduling gains,
it also runs faster per user at the same batch size (by up to 5.6 tok/s/user), and between batch
sizes 256 and 512 throughput per GPU changes steeply with per-user speed. All three large peaks are
at batch sizes 256 to 384. GSM8K gains little: its 5-shot prompts average 880 tokens and its
answers are short, so prefill dominates and throughput per GPU is flat at about 640 tok/s from
batch size 256 up.

Points that set a peak are the mean of three runs on different nodes; the others are single runs.
Run-to-run variation in throughput per GPU is mostly within 2%.

## Quick start

From `examples/models/core/deepseek_v4`, with the checkpoint in `$CKPT`:

```bash
B=dspark_benchmark
# 1. Datasets (section 2)
python $B/build_datasets.py --model $CKPT --output-dir data --specdec-bench /path/to/Model-Optimizer/examples/specdec_bench
# 2. Runtime YAMLs (section 3)
python $B/make_configs.py --output-dir configs \
  --cost-table dspark_artifacts/costs-b300-dep8.json --calibration dspark_artifacts/calibration.json
# 3. One run per setting (section 5); here SPEED-Bench around its peak
for bs in 256 384 512; do
  for arm in fixed-k3 fixed-k4 fixed-k5 confidence-k5; do
    bash $B/run_benchmark.sh $CKPT configs/bs$bs-$arm.yaml data/speed.jsonl $bs runs/speed-bs$bs-$arm-1
  done
done
# 4. Compare at equal per-user speed (section 6)
python $B/analyze.py runs/* --plot speed.png
```

## 1. Requirements

- TensorRT LLM with DSpark confidence scheduling (PR #20078).
- A DeepSeek-V4-Pro checkpoint with the embedded DSpark drafter and trained confidence-head weights
  (block size 5). Its `tokenizer.json` and `encoding/encoding_dsv4.py` render the prompts.
- One node with 8 GPUs. The shipped cost table is for B300; other GPUs need their own (section 4).
- For the scripts: network access to Hugging Face, and `huggingface_hub`, `pyarrow`, `transformers`
  and `pyyaml` (all in the TensorRT LLM container). SPEED-Bench also needs Model-Optimizer's
  `specdec_bench` (its `requirements.txt`) and access to the gated HLE dataset.

## 2. Datasets

| Dataset | Source, pinned revision | Unique prompts | Input tokens (min / median / max) | Prompt |
| --- | --- | --- | --- | --- |
| SPEED-Bench | `nvidia/SPEED-Bench` throughput_1k @`487aa71` (before v1.0) | 1,480 | 794 / 1,047 / 1,280 | first turn |
| Arena-Hard v0.1 | `lmarena-ai/arena-hard-auto-v0.1` @`2a69efe` | 500 | 11 / 35 / 207 | first turn, stripped, first 512 characters |
| ShareGPT | `anon8231489123/ShareGPT_Vicuna_unfiltered` V3 cleaned split @`192ab21` | 3,712 | 8 / 134 / 1,024 | first turn |
| GSM8K | `openai/gsm8k` main @`740312a` | 1,319 (test set) | 553 / 873 / 1,505 | five train question/answer pairs, then the test question |

Every prompt is one user turn rendered with the checkpoint's DeepSeek-V4 chat template without
thinking, so the token counts include the template. Each dataset is a `trtllm-bench` JSONL of 20,480
requests: the unique prompts in a seeded shuffle, repeated. A run with N requests uses the first N.

```json
{"task_id": 0, "input_ids": [0, 128803, 201, 3476, ...], "output_tokens": 1024}
```

[`build_datasets.py`](dspark_benchmark/build_datasets.py) downloads the pinned files, selects and
orders the prompts by [`datasets_manifest.json`](dspark_benchmark/datasets_manifest.json) (record IDs
and text hashes only), renders and writes `data/{speed,arena,sharegpt,gsm8k}.jsonl`. It checks each
file against the SHA-256 of the one used for the results and prints `matches the published dataset`.

SPEED-Bench publishes 639 of its 1,536 throughput_1k rows as placeholders. The text of those rows
is fetched from the original sources (HLE, AdaLEval, BAMBOO) by Model-Optimizer's `specdec_bench`,
which the builder runs on the pinned file. Use Model-Optimizer at commit `5db26825`, which
reproduces the published prompts, and accept the terms of the gated `cais/hle` dataset on Hugging
Face first:

```bash
git clone https://github.com/NVIDIA/Model-Optimizer && git -C Model-Optimizer checkout 5db26825
pip install -r Model-Optimizer/examples/specdec_bench/requirements.txt
hf auth login
```

The other three datasets need neither; build them alone with `--datasets arena sharegpt gsm8k`.

The datasets keep their own terms: SPEED-Bench is under the NVIDIA Evaluation Dataset License,
and its placeholder rows come from third-party hosts under their own terms. Arena-Hard v0.1 and
ShareGPT (per its dataset card) are under Apache-2.0, and GSM8K and HLE under MIT. This repository
distributes none of their content: the manifest holds record identifiers and hashes, and you
download the data yourself. Check that each license fits your use.

## 3. Runtime configuration

[`make_configs.py`](dspark_benchmark/make_configs.py) writes one YAML per batch size and setting:
`bs<BS>-fixed-k<K>.yaml` for a fixed draft length and `bs<BS>-confidence-k<K>.yaml` for confidence
scheduling with maximum draft length K. BS512 with confidence scheduling, for example:

```yaml
gpus_per_node: 8
tensor_parallel_size: 8
moe_expert_parallel_size: 8
moe_tensor_parallel_size: 1
enable_attention_dp: true
attention_dp_config:
  enable_balance: true
  batching_wait_iters: 10
  timeout_iters: 50
max_batch_size: 64                 # rank-local: BS / 8
max_num_tokens: 4096
max_input_len: 1536
max_seq_len: 2560
enable_chunked_prefill: false
custom_tokenizer: deepseek_v4
kv_cache_config:
  dtype: fp8
  free_gpu_memory_fraction: 0.5
  enable_block_reuse: false
  tokens_per_block: 128
  host_cache_size: 0
moe_config:
  backend: MEGAMOE_DEEPGEMM
cuda_graph_config:
  enable_padding: true
  max_batch_size: 64
  batch_sizes: [16, 32, 64]        # 16, 32, 64, 128, 256 below BS / 8, then BS / 8
prefill_cuda_graph_backend: breakable   # default token buckets
speculative_config:
  decoding_type: DSpark
  max_draft_len: 5
  block_size: 5
  enable_confidence_scheduling: true
  confidence_cost_table_path: /abs/path/dspark_artifacts/costs-b300-dep8.json
  confidence_calibration_path: /abs/path/dspark_artifacts/calibration.json
```

A fixed draft length uses the same file with only `decoding_type`, `max_draft_len` and
`block_size` in `speculative_config`. `confidence_min_predicted_gain` keeps its default, 0.04
(section 7).

Why these settings, measured on short chat prompts at BS512 and BS1024 (fixed and confidence
scheduling alike):

| Setting | Alternative | Throughput change of the alternative |
| --- | --- | --- |
| `prefill_cuda_graph_backend: breakable`, default buckets | disabled (eager) | about 5% slower |
| | piecewise | fails to compile for DeepSeek-V4 |
| | buckets capped at 1024 or 2048 | about 3.6% slower at BS1024 |
| `attention_dp_config` balance, wait 10 / timeout 50 | no balance | about 2x slower at BS512 |
| | wait 0 / timeout 60 | 5-9% slower |
| `MEGAMOE_DEEPGEMM` | `TRTLLM` + low-precision combine | about 11% slower |

`enable_lm_head_tp_in_adp` and `sparse_attention_config.enable_heuristic_topk` were neutral.

## 4. Cost table and calibration

The cost table holds measured decode step times per draft length `K`, rank-local graph batch size
`G` and number of verified rows `V`. Step costs depend on the GPU model, parallel layout and
kernels, so a table is tied to its setup.

**Reuse** [`dspark_artifacts/costs-b300-dep8.json`](dspark_artifacts/costs-b300-dep8.json) on 8x B300
with attention DP and 8-way EP. It holds K3 and K4 for `G` 16 to 256 and K5 for `G` 8 to 256
(including 48 and 96), which covers every configuration above. All results used it, for every
dataset; it was profiled with short Arena-Hard prompts. At startup the runtime checks the table and
stops with the command to regenerate it if the file is missing, has no entry for `max_draft_len`,
lacks a row for a captured `cuda_graph_config.batch_sizes` entry, or was measured on another GPU
model, world size, TP/EP size or attention-DP setting. A different TensorRT LLM version only warns.

**Generate** one with the runtime YAML you will serve with:

```bash
python -m tensorrt_llm.tools.dspark_profile_costs \
  --model $CKPT --config configs/bs1024-confidence-k5.yaml --k 3 4 5 \
  --dataset data/arena.jsonl --output dspark-costs.json
```

For each `K` the tool loads the model once and, for every captured `G`, runs a wave of `G x 8`
requests (another one while a tier has too few samples). During the wave's full decode steps it
cycles the number of verified rows through native and four compact tiers, holds each for 16
steps, and records the median whole-step time (drafting included). Attention-DP balancing is
turned off while profiling; it only delays prefill and does not change decode-step cost. It takes
about 10 minutes per `K` on 8x B300 for the BS1024 ladder. Running it again with another `--k`, or
with `--graph-batch-sizes` (for example `48 96`, or `256` for BS2048), adds those rows to the same
file. Example output (V counts verified rows; native is `G x (K + 1)`):

```text
K=5 G=64:  native=50.75ms V224=41.92ms (-17.4%), V256=43.15ms (-15.0%), V288=45.85ms (-9.6%), V320=48.15ms (-5.1%)
K=5 G=128: native=73.93ms V448=57.38ms (-22.4%), V512=58.31ms (-21.1%), V576=63.65ms (-13.9%), V640=67.23ms (-9.1%)
```

Options: `--graph-batch-sizes` (default: the YAML's CUDA-graph ladder), `--tier-fractions` (default
`0.5 0.6 0.7 0.8`: on average 50-80% of the K drafted tokens verified per request; each must be at
least 1/K, since every request verifies at least one draft token), `--rounds` and `--hold-steps`
(samples per tier), `--max-waves` (default 3) and `--min-samples` (default 20).

**Calibration.** `confidence_calibration_path` holds per-position temperatures,
`{"sts_temperatures": [...]}`. A vector fitted at the checkpoint's block size serves every smaller
`K`; the runtime uses the first `K` entries. Without it the raw sigmoid confidence is used.
[`dspark_artifacts/calibration.json`](dspark_artifacts/calibration.json) is the one used for the
results.

## 5. Run the benchmark

[`run_benchmark.sh`](dspark_benchmark/run_benchmark.sh) runs one setting in its own process and
writes `report.json`, `requests.json` and `run.json` to its output directory:

```bash
bash dspark_benchmark/run_benchmark.sh $CKPT configs/bs512-confidence-k5.yaml data/sharegpt.jsonl 512 runs/sharegpt-bs512-confidence-k5-1
```

It runs `trtllm-bench throughput` at `--concurrency` equal to the batch size with the request and
warmup counts below, greedy sampling and stop at EOS ([`sampler.yaml`](dspark_benchmark/sampler.yaml),
`--eos_id 1`):

| Batch size | 64 | 128 | 256 | 384 | 512 | 768 | 1024 | 2048 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Requests | 1,280 | 2,560 | 5,120 | 7,680 | 10,240 | 10,240 | 10,240 | 20,480 |
| Warmup requests | 256 | 512 | 512 | 768 | 1,024 | 768 | 1,024 | 2,048 |

Settings run per batch size:

| Batch size | 64, 128 | 256 | 384 | 512 | 768 | 1024, 2048 |
| --- | --- | --- | --- | --- | --- | --- |
| Fixed | K3-K5 | K2-K5 | K3-K5 | K1-K5 | K2-K5 | K1-K5 |
| Confidence scheduling | K5 | K5 | K5 | K4, K5 | K5 | K3-K5 |

BS384 and BS768 were run for SPEED-Bench, ShareGPT and Arena-Hard only. Run every setting of one
dataset and batch size on the same node, one after another. To confirm a peak, repeat its points on
other nodes in a different order and let the analysis average them.

Throughput per GPU is `trtllm-bench`'s system output throughput divided by 8. Per-user speed is its
per-user output throughput: a request's output tokens over its latency, prefill included, averaged
over requests.

## 6. Compare at equal per-user speed

```bash
python dspark_benchmark/analyze.py runs/* --plot results.png
```

[`analyze.py`](dspark_benchmark/analyze.py) averages runs of the same dataset, batch size, method
and draft length, then prints for each dataset the best fixed and confidence-scheduling settings at
every batch size and the largest gain at equal per-user speed. For each method, the points that no
other point beats on both per-user speed and throughput per GPU form a line; neighbouring points
are joined linearly and the two lines are compared only where both exist, without extrapolation.
The ratio of two linear pieces peaks at an end, so the largest gain is found at a measured point of
either line. Run on the published runs, it prints the numbers in the results table.

## 7. Minimum predicted gain

In a decode-only step, confidence scheduling picks a compact budget only if its predicted accepted
tokens per millisecond beat full verification by at least `confidence_min_predicted_gain`
(default 0.04). Otherwise it verifies every row.

The margin absorbs prediction error. The savings are small per step: the smallest profitable cut
saves about 5% of the step time and loses about 3.6% of the accepted tokens. Both the head's
predictions and the measured step costs carry errors of that size, and choosing the best of four
noisy candidates favours the one whose gain was overestimated. In earlier tests on short prompts,
a gate of 0 lost 1-2% against 0.04 at BS1024 and tied at BS512, even after recalibrating the head.

On the real prompts the choice barely matters. Confidence scheduling at K5 was run with gates of 0,
2, 4, 6 and 8%, all five on the same node, for SPEED-Bench and ShareGPT at batch sizes 256, 384,
512 and 1024 (8 cells, one run per gate). Throughput per GPU relative to the default:

| Gate | 0 | 0.02 | 0.04 (default) | 0.06 | 0.08 |
| --- | --- | --- | --- | --- | --- |
| Mean over the 8 cells | -0.2% | +0.3% | 0 | -0.5% | -0.4% |
| Range | -0.9% to +0.2% | -1.0% to +1.4% | | -2.3% to +1.4% | -2.3% to +1.7% |

No gate is distinguishable from the default within run-to-run variation, and per-user speed moved
by less than 1.5 tok/s. Keep the default and leave `confidence_min_predicted_gain` out of the
configuration; there is no need to tune it per workload or batch size. Check it again only after
retraining the confidence head or changing its calibration.

## 8. Troubleshooting

| Message | Fix |
| --- | --- |
| `requires confidence_cost_table_path` | Set the path or generate a table (section 4). |
| `has no costs for max_draft_len=K` | Add that K: rerun the tool with `--k K` and the same `--output`. |
| `no measurements for captured graph batch sizes [...]` | Use covered `batch_sizes`, or add the missing `G` with `--graph-batch-sizes`. |
| `measured on a different setup (...)` | The table came from another GPU or parallel layout; regenerate on this one. |
| `measured with TensorRT-LLM X; running Y` (warning) | Usually fine; regenerate if gains shrink. |
| `has N temperatures but the confidence head scores K` | Use a calibration fitted at a K at least as large. |
| `N of 1480 SPEED-Bench prompts are missing` | Pass `--specdec-bench` at Model-Optimizer `5db26825` (section 2). |
| `DIFFERS from the published dataset` | A source file or the checkpoint's chat template differs from the one used for the results. |
