<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DSpark confidence scheduling: reproduce the DeepSeek-V4-Pro results

This guide reproduces the confidence-scheduling throughput gain on DeepSeek-V4-Pro with prompts
from four public datasets, and shows how to reuse or generate the cost table the feature needs. The
feature itself is documented in the [DeepSeek-V4 README](README.md#dspark-confidence-scheduling-prototype).
The scripts are in [`dspark_benchmark/`](dspark_benchmark) and the B300 cost tables and calibration
in [`dspark_artifacts/`](dspark_artifacts).

## Results

One node of 8x B300, attention data parallel with 8-way expert parallel, FP8 KV cache. The runs used
TensorRT LLM main (2d1a7d1) with confidence scheduling added, which main now includes; section 5
repeats some of them on a later main. Each dataset was sent at constant concurrency equal to the
batch size, from 64 to 2048 requests. Decoding is greedy, and requests stop at their end-of-sequence
token or after 1,024 output tokens.

Confidence scheduling (maximum draft length K2 to K5) is compared with fixed DSpark draft lengths
(K1 to K5) at equal per-user speed. For each method, the best points over all batch sizes and draft
lengths form a line, and the table shows where the confidence-scheduling line is furthest above the
fixed one (section 6). Confidence scheduling uses device re-ranking, the default (section 8), at
the 15 batch sizes and draft lengths where it was measured, each taking the faster of its runs with
and without it. Elsewhere its runs ranked on the host only; re-ranking beat those in 13 of the 15
settings where both were run, so these points slightly understate the default.

| Dataset | Largest gain at equal per-user speed | At tok/s/user | Fixed → confidence, tok/s/GPU |
| --- | --- | --- | --- |
| SPEED-Bench throughput_1k | **+23.5%** | 48.9 | 1,469 → 1,815 |
| ShareGPT | **+26.2%** | 63.8 | 1,948 → 2,457 |
| Arena-Hard v0.1 | **+23.2%** | 45.0 | 5,228 → 6,442 |
| GSM8K 5-shot | +1.5% | 38.9 | 595 → 604 |

Ranking on the host only, as TensorRT LLM did before device re-ranking, the largest gains were
+22.5%, +19.8%, +14.7% and +1.5% at the same per-user speeds (`analyze.py --no-rerank`).

The best fixed draft length against the best confidence-scheduling setting at the same batch size:

| Batch size | 64 | 128 | 256 | 384 | 512 | 768 | 1024 | 2048 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPEED-Bench | +1.7% | -0.6% | +6.7% | +5.5% | +4.6% | +3.4% | +1.4% | +0.1% |
| ShareGPT | +0.6% | -0.3% | +5.7% | +3.1% | +7.1% | +3.1% | +8.9% | +2.5% |
| Arena-Hard | +0.7% | +0.1% | +3.2% | +1.3% | +5.7% | +1.2% | +9.1% | +6.1% |
| GSM8K | -0.2% | +0.8% | +0.3% | | -0.3% | | +0.2% | -0.6% |

The equal-speed gains are larger than the same-batch-size gains. Where confidence scheduling gains,
it also runs faster per user at the same batch size (by up to 6.3 tok/s/user), and throughput per
GPU changes steeply with per-user speed between neighbouring batch sizes. SPEED-Bench and ShareGPT
peak at batch size 256. Arena-Hard peaks at the fixed line's batch-size-1024 point, which lies below
the confidence line between batch sizes 1024 and 2048. At batch size 2048 the best maximum draft
length is K2 for SPEED-Bench and ShareGPT; with K3 to K5 only, ranking on the host, confidence
scheduling was 2.7% and 1.1% slower there than the best fixed draft length. GSM8K gains little: its
5-shot prompts average 880 tokens and its answers are short, so prefill dominates and throughput per
GPU is flat at about 640 tok/s from batch size 256 up.

Points that set a peak are the mean of three or more runs on different nodes, and re-ranking points
the mean of two; the others are single runs. Run-to-run variation in throughput per GPU is mostly
within 2%.

These results use attention data parallelism. With tensor-parallel attention instead (no attention
DP) at batch sizes 8 to 64, confidence scheduling gains less: from 3.1% slower to 5.2% faster than
the best fixed draft length at the same batch size, with prefill batching (section 9).

## Quick start

From `examples/models/core/deepseek_v4`, with the checkpoint in `$CKPT`:

```bash
B=dspark_benchmark
# 1. Datasets (section 2)
python $B/build_datasets.py --model $CKPT --output-dir data --specdec-bench /path/to/Model-Optimizer/examples/specdec_bench
# 2. Runtime YAMLs (section 3); --host-ranking adds configs without device re-ranking (section 8)
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

- TensorRT LLM main at e3ba623 or later, which has DSpark confidence scheduling and device
  re-ranking.
- A DeepSeek-V4-Pro checkpoint with the embedded DSpark drafter and trained confidence-head weights
  (block size 5). Its `tokenizer.json` and `encoding/encoding_dsv4.py` render the prompts.
- One node with 8 GPUs. The shipped cost tables are for B300, one with attention DP and one with
  tensor-parallel attention (section 9); other GPUs need their own (section 4).
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
scheduling with maximum draft length K and device re-ranking, the default (section 8). With
`--host-ranking` it also writes `bs<BS>-confidence-host-k<K>.yaml`, the same with re-ranking off.
`--layout tep8` writes the tensor-parallel-attention configs of section 9 instead. BS512 with
confidence scheduling, for example:

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
  enable_fused_confidence_scheduler: true   # device re-ranking, the default (section 8)
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
kernels, so a table is tied to its setup. The [README](README.md#get-a-cost-table) describes the
profiler and the checks the runtime makes at startup.

**Reuse** [`dspark_artifacts/costs-b300-dep8.json`](dspark_artifacts/costs-b300-dep8.json) on 8x B300
with attention DP and 8-way EP (for tensor-parallel attention, see section 9). It holds K2 for `G`
16 to 256 (including 96), K3 and K4 for `G` 16 to 256, and K5 for `G` 8 to 256 (including 48 and
96). That covers every setting of section 5; `make_configs.py` skips the confidence configs it does
not cover, such as K2 to K4 at batch size 64 (`G` 8). All results used it, for every dataset; it was
profiled with short Arena-Hard prompts. Its GPUs reported the name `NVIDIA B300 SXM6 AC`, and the
runtime requires the same one: check `nvidia-smi --query-gpu=name --format=csv,noheader` first.

**Generate** one with the runtime YAML you will serve with, which also fixes the parallel layout (a
table measured with attention DP is rejected without it, and the reverse):

```bash
python -m tensorrt_llm.tools.dspark_profile_costs \
  --model $CKPT --config configs/bs1024-confidence-k5.yaml --k 2 3 4 5 \
  --dataset data/arena.jsonl --output dspark-costs.json
```

It takes about 25 minutes per `K` on 8x B300 for a ladder of five to eight graph batch sizes, most
of it loading the model and capturing graphs; a single `G` takes 10 to 20 minutes. Running it again
with another `--k`, or with `--graph-batch-sizes` (for example `48 96`, or `256` for BS2048), adds
those rows to the same file. Example output (native verifies all `G x (K + 1)` rows; the four
compact tiers verify 50% to 80% of the drafted tokens):

```text
K=5 G=64:  native=50.75ms V224=41.92ms (-17.4%), V256=43.15ms (-15.0%), V288=45.85ms (-9.6%), V320=48.15ms (-5.1%)
K=5 G=128: native=73.93ms V448=57.38ms (-22.4%), V512=58.31ms (-21.1%), V576=63.65ms (-13.9%), V640=67.23ms (-9.1%)
```

**Calibration.** [`dspark_artifacts/calibration.json`](dspark_artifacts/calibration.json) holds the
per-position temperatures used for the results, fitted offline on held-out prompts (format in the
[README](README.md#calibration)). It belongs to the checkpoint's confidence head, not to the GPU or
parallel layout, so it serves every setup with this checkpoint, TEP8 included. A retrained head
needs its own fit, for which TensorRT LLM has no tool yet.

## 5. Run the benchmark

[`run_benchmark.sh`](dspark_benchmark/run_benchmark.sh) runs one setting in its own process, copies
the config to its output directory as `config.yaml` and writes `report.json`, `requests.json` and
`run.json` there:

```bash
bash dspark_benchmark/run_benchmark.sh $CKPT configs/bs512-confidence-k5.yaml data/sharegpt.jsonl 512 runs/sharegpt-bs512-confidence-k5-1
```

It runs `trtllm-bench throughput` at `--concurrency` equal to the batch size with the request and
warmup counts below, greedy sampling and stop at EOS ([`sampler.yaml`](dspark_benchmark/sampler.yaml),
`--eos_id 1`), with `PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.99999` as for the published
results; set it too when running `trtllm-bench` by hand.

| Batch size | 64 | 128 | 256 | 384 | 512 | 768 | 1024 | 2048 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Requests | 1,280 | 2,560 | 5,120 | 7,680 | 10,240 | 10,240 | 10,240 | 20,480 |
| Warmup requests | 256 | 512 | 512 | 768 | 1,024 | 768 | 1,024 | 2,048 |

Settings run per batch size:

| Batch size | 64, 128 | 256 | 384 | 512 | 768 | 1024, 2048 |
| --- | --- | --- | --- | --- | --- | --- |
| Fixed | K3-K5 | K2-K5 | K3-K5 | K1-K5 | K2-K5 | K1-K5 |
| Confidence scheduling | K5 | K5 | K5 | K4, K5 | K5 | K2-K5 |

The published confidence runs ranked on the host. To compare the two (section 8), run the
`confidence-host-k<K>` config next to each `confidence-k<K>` one; the analysis counts both as
confidence scheduling and keeps the faster at each batch size. BS384 and BS768 were run for
SPEED-Bench, ShareGPT and Arena-Hard only. Run every setting of one dataset and batch size on the
same node, one after another. To confirm a peak, repeat its points on other nodes in a different
order and let the analysis average them.

Throughput per GPU is `trtllm-bench`'s system output throughput divided by 8. Per-user speed is its
per-user output throughput: a request's output tokens over its latency, prefill included, averaged
over requests.

**Check on a later main.** Three cells were rerun with these scripts on TensorRT LLM main 47cd51b
with device re-ranking added, one run per setting, every setting of a cell on one node (throughput
per GPU; confidence settings rank on the host unless marked re-ranking):

| Dataset | Batch size | Setting | Published | Later main | Change |
| --- | --- | --- | --- | --- | --- |
| SPEED-Bench | 256 | fixed K3 | 1,469 | 1,491 | +1.5% |
| | | fixed K5 | 1,466 | 1,475 | +0.6% |
| | | confidence K5 | 1,546 | 1,558 | +0.8% |
| | | confidence K5, re-ranking | 1,567 | 1,588 | +1.4% |
| ShareGPT | 1024 | fixed K3 | 4,376 | 4,472 | +2.2% |
| | | confidence K5 | 4,601 | 4,541 | -1.3% |
| | | confidence K5, re-ranking | 4,765 | 4,822 | +1.2% |
| Arena-Hard | 2048 | fixed K2 | 7,174 | 7,180 | +0.1% |
| | | confidence K5 | 7,257 | 7,325 | +0.9% |
| | | confidence K5, re-ranking | 7,611 | 7,648 | +0.5% |
| | | confidence K2 | 7,210 | 7,193 | -0.2% |

Every setting is within 2.2% of its published mean, about the run-to-run variation, and confidence
scheduling still beats the best fixed draft length in all three cells.

## 6. Compare at equal per-user speed

```bash
python dspark_benchmark/analyze.py runs/* --plot results.png
```

[`analyze.py`](dspark_benchmark/analyze.py) averages runs of the same dataset, batch size, method,
draft length and re-ranking setting, then prints for each dataset the best fixed and
confidence-scheduling settings at every batch size and the largest gain at equal per-user speed.
For each method, the points that no other point beats on both per-user speed and throughput per GPU
form a line; neighbouring points are joined linearly and the two lines are compared only where both
exist, without extrapolation. Confidence runs with and without re-ranking both count as confidence
scheduling; `--no-rerank` leaves out the re-ranking runs, which gives the host-ranking figures in
Results.

## 7. Minimum predicted gain

In a decode-only step, confidence scheduling picks a compact budget only if its predicted accepted
tokens per millisecond beat full verification by at least `confidence_min_predicted_gain`
(default 0.04). Otherwise it verifies every row.

The margin absorbs prediction error: the smallest profitable cut saves about 5% of the step time,
about the size of the errors in the head's predictions and in the measured step costs. On the real
prompts the choice barely matters. Confidence scheduling at K5 was run with gates of 0,
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

## 8. Device re-ranking (default)

Confidence scheduling chooses the verifier budget on the host, from a copy of the previous step's
confidence, so the host never waits for the GPU. Device re-ranking keeps that budget but re-ranks
the draft positions to verify on the GPU, using the confidence of the block being verified. It is on
by default. To rank on the host only, as the published confidence runs did:

```yaml
speculative_config:
  enable_confidence_scheduling: true
  enable_fused_confidence_scheduler: false   # host ranking only
```

`make_configs.py --host-ranking` writes these as `bs<BS>-confidence-host-k<K>.yaml` (section 3).

Re-ranking makes compact steps rebuild their inputs in full and run a small extra GPU pass. It
raised throughput in 13 of the 15 settings below, by up to 5.5%, and lowered it in two. Confidence
scheduling with and without it, alternating on one node, two runs each (throughput per GPU):

| Dataset | Batch size | Max K | Without | With | Change | Acceptance length |
| --- | --- | --- | --- | --- | --- | --- |
| SPEED-Bench | 256 | 5 | 1,528 | 1,567 | +2.5% | 2.78 → 2.86 |
| SPEED-Bench | 384 | 5 | 1,985 | 1,941 | -2.2% | 2.88 → 2.92 |
| SPEED-Bench | 512 | 5 | 2,189 | 2,228 | +1.8% | 2.72 → 2.83 |
| SPEED-Bench | 1024 | 5 | 2,163 | 2,202 | +1.8% | 2.74 → 2.83 |
| SPEED-Bench | 2048 | 2 | 2,270 | 2,276 | +0.3% | 2.19 → 2.21 |
| ShareGPT | 256 | 5 | 1,993 | 2,059 | +3.3% | 2.89 → 3.03 |
| ShareGPT | 512 | 5 | 3,255 | 3,362 | +3.3% | 2.88 → 3.02 |
| ShareGPT | 1024 | 5 | 4,612 | 4,765 | +3.3% | 2.88 → 3.02 |
| ShareGPT | 2048 | 2 | 5,994 | 6,049 | +0.9% | 2.32 → 2.33 |
| Arena-Hard | 256 | 5 | 2,295 | 2,374 | +3.4% | 3.25 → 3.41 |
| Arena-Hard | 512 | 5 | 3,890 | 4,044 | +3.9% | 3.20 → 3.37 |
| Arena-Hard | 1024 | 5 | 5,629 | 5,854 | +4.0% | 3.20 → 3.37 |
| Arena-Hard | 2048 | 5 | 7,216 | 7,611 | +5.5% | 3.02 → 3.19 |
| GSM8K | 256 | 5 | 633 | 630 | -0.4% | 4.02 → 3.99 |
| GSM8K | 1024 | 3 | 639 | 639 | 0.0% | 3.19 → 3.19 |

Repeated on TensorRT LLM main 47cd51b with device re-ranking, again two runs each on one node, the
-2.2% setting (SPEED-Bench, batch size 384) measured +0.4% (1,981 → 1,989) and ShareGPT at batch
size 512 +3.3% (3,303 → 3,412).

Re-ranking does not change what the model outputs beyond ordinary run-to-run variation: GSM8K
accuracy (1,319 questions, the 5-shot prompts above, greedy) was 96.7%, 96.6% and 96.8% in three
runs with it, 96.8% without it and 96.5% with fixed K5, and any two of these runs gave the same
answer to about 99% of the questions.

Without attention DP (tensor-parallel attention, 8-way EP), every TP rank ranks TP rank 0's
confidence, so the ranks verify the same positions. There, with prefill batching (section 9),
re-ranking raised throughput over host ranking in 14 of 16 settings at batch sizes 8 to 64 on the
four datasets, by up to 6.4%.

## 9. Tensor-parallel attention (TEP8)

Without attention DP, every GPU runs attention for every request, with 8-way expert parallel as
before (TEP8). Confidence scheduling works here too, with its own configs and cost table, and gains
less. The best confidence-scheduling setting against the best fixed draft length at the same batch
size, throughput per GPU, with device re-ranking and prefill batching (draft length K5 for both
unless shown):

| Dataset | Batch size 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- |
| SPEED-Bench | +0.9% (K4) | +1.6% | +5.2% | +2.3% |
| ShareGPT | +2.5% | +0.9% | +4.3% | +3.8% |
| Arena-Hard | -0.2% | -2.2% | +1.0% | +0.7% |
| GSM8K | | -3.1% | +0.4% | -0.9% |

Each cell ran fixed K3 to K5 and confidence scheduling with maximum K3 to K5 on one node, in one job
on TensorRT LLM main 47cd51b with re-ranking: fixed K5, confidence K5, fixed K4, confidence K4, fixed
K3, confidence K3, confidence K5, fixed K5. K5 was the best draft length for both methods in 14 of
16 cells; at SPEED-Bench batch size 8 it was K4 for both, within 2% of K5. GSM8K at batch size 8 is
left out: its settings ranged from 70 to 95 tok/s/GPU in no order of draft length. Where confidence
scheduling gains, it also runs faster per user, by up to 5.4 tok/s/user. `analyze.py` therefore
reports larger gains at equal per-user speed (+14.8% SPEED-Bench, +18.6% ShareGPT, +7.7%
Arena-Hard), but with batch sizes a factor of two apart its straight lines between points overstate
those; the same-batch-size gains above are the safer figures. At batch size 128 (Arena-Hard and
ShareGPT, runs pooled over several nodes) confidence scheduling was within 1% of fixed K5. Without
prefill batching and ranking on the host, it had ranged from 5.8% slower to 2.9% faster at batch
sizes 8 to 128.

**Configs.** `make_configs.py --layout tep8` writes the same file names for batch sizes 8, 16, 32
and 64:

```bash
python dspark_benchmark/make_configs.py --layout tep8 --output-dir configs-tep \
  --cost-table dspark_artifacts/costs-b300-tep8.json --calibration dspark_artifacts/calibration.json
```

They differ from the attention-DP configs of section 3 in:

| Setting | TEP8 |
| --- | --- |
| `enable_attention_dp` | `false`, without `attention_dp_config` |
| `max_batch_size`, `cuda_graph_config` | BS, since every GPU runs all requests; graphs at powers of two below BS, plus BS |
| `max_num_tokens` | 8192 |
| `moe_config` | `TRTLLM` with `use_low_precision_moe_combine: true`, as in [`deepseek-v4-pro-latency.yaml`](../../../configs/curated/deepseek-v4-pro-latency.yaml) |
| `batch_wait_timeout_iters`, `batch_wait_max_tokens_ratio` | 10 and 0.2: prefill batching |

**Prefill batching.** These two are existing TensorRT LLM scheduler settings, off by default.
Without attention DP, a new request's prefill runs on every GPU in the same step as the whole decode
batch, so at constant concurrency most steps are mixed steps, which confidence scheduling does not
trim. With these settings the scheduler holds new requests until 10 iterations have passed or the
step's tokens (decode plus waiting prompts) reach 20% of `max_num_tokens`, then admits them
together. With fixed K5 this raised throughput per GPU by 2% to 43% on the four datasets at batch
sizes 8 to 128, most at large batch sizes and on GSM8K's long prompts, and per-user speed rose too
in all but one setting. At Arena-Hard batch size 128, mixed steps fell from 60% to 16% of GPU time.
Use it for fixed draft lengths too; the results above do.

**Cost table.** Reuse [`dspark_artifacts/costs-b300-tep8.json`](dspark_artifacts/costs-b300-tep8.json)
on 8x B300 with tensor-parallel attention and 8-way EP. It holds K2 to K5 for `G` 1 to 128 (here `G`
is the whole batch, since every GPU runs every request) and was profiled like the attention-DP
table, with TRTLLM MoE. To generate one, pass a TEP8 config to the tool of section 4:

```bash
python -m tensorrt_llm.tools.dspark_profile_costs \
  --model $CKPT --config configs-tep/bs64-fixed-k5.yaml --k 2 3 4 5 \
  --dataset data/arena.jsonl --output dspark-costs-tep8.json
```

**Run and compare.** `run_benchmark.sh` takes the layout from the config. For TEP8 it sends 20
times the batch size in requests (at least 320) after twice the batch size in warmup requests (at
least 64), as for the results above. `analyze.py` reports TEP8 runs as `<dataset> TP attention`,
apart from attention-DP runs of the same dataset. The first run of a job can be slow: in GSM8K at
batch size 8 it ran at about half speed, whichever setting came first. Repeat the first setting at
the end of the job, and leave out the first run if it is far below the repeat.

## 10. Troubleshooting

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
