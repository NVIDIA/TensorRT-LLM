(perf-overview)=

# Overview

This document summarizes performance measurements of TensorRT-LLM on a number of GPUs across a set of key models.

The data in the following tables is provided as a reference point to help users validate observed performance.
It should *not* be considered as the peak performance that can be delivered by TensorRT-LLM.

Not all configurations were tested for all GPUs.

Commands are kept as simple as possible to ease reproducibility, with many options left at their default settings.
Tune batch sizes, parallelism configurations, and other options to improve performance for your specific situation.


For DeepSeek R1 performance, see the [performance guide](../blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md).

For more information on benchmarking with `trtllm-bench`, see this NVIDIA [blog post](https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/).

For NUMA systems, consult the ["CPU Affinity configuration in TensorRT LLM"](../deployment-guide/configuring-cpu-affinity.md) guide to achieve best performance. These options were enabled for relevant tests.

## Throughput Measurements

The tables below show performance data where a local inference client sends requests at a high rate with no delay between requests,
representing the throughput scenario under maximum load. The reported metric is `Output Throughput per GPU (tokens/sec/GPU)`.

The performance numbers below were collected using the steps described in [Reproducing Benchmarked Results](#reproducing-benchmarked-results).

All tested models use weights quantized with [ModelOpt](https://nvidia.github.io/Model-Optimizer/) and published by NVIDIA on the [Model Optimizer HuggingFace Collection](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4).

### Hardware
The following GPU variants were used for testing:
- H100 SXM 80GB (DGX H100)
- H200 SXM 141GB (DGX H200)
- B200 180GB (DGX B200)
- GB200 192GB (GB200 NVL72)
- RTX 6000 Pro Blackwell Server Edition

Note: As of `release/1.2`, support for B300 and GB300 is in beta, and performance data should not be considered finalized. 
- B300 288GB (DGX B300) (beta)
- GB300 288 GB (GB300 NVL72) (beta)

Other hardware variants may have different TDP, memory bandwidth, core count, or other features leading to performance differences on these workloads.

### FP4 Models

```text
nvidia/DeepSeek-R1-0528-NVFP4-v2
nvidia/Qwen3-235B-A22B-FP4
nvidia/Qwen3-30B-A3B-FP4
nvidia/Llama-3.3-70B-Instruct-FP4
```

### FP8 Models

```text
deepseek-ai/DeepSeek-R1-0528
nvidia/Qwen3-235B-A22B-FP8
nvidia/Llama-3.3-70B-Instruct-FP8
```

# Performance Summary - All Networks

All performance values are measured in `output tokens per second per GPU`, where `output tokens` refers to all generated tokens (excluding input/prompt tokens).

Data in these tables is taken from the `Per GPU Output Throughput (tps/gpu)` metric reported by `trtllm-bench`.
The metric calculations are defined in [reporting.py](../../../tensorrt_llm/bench/dataclasses/reporting.py#L570) and [statistics.py](../../../tensorrt_llm/bench/dataclasses/statistics.py#L188).

RTX 6000 systems can benefit from enabling pipeline parallelism (PP) in LLM workloads. Results for various TP x PP combinations on this GPU are presented in a separate table for each network.

## Units

All performance values are measured in **output tokens per second per GPU**.

## Table of Contents

- [Deepseek R1 0528](#deepseek-r1-0528)
  - [Deepseek R1 0528 - RTX 6000 Pro Blackwell Server Edition](#deepseek-r1-0528-rtx-configurations)
- [GPT-OSS 120B](#gpt-oss-120b)
- [GPT-OSS 20B](#gpt-oss-20b)
- [LLaMA v3.3 70B](#llama-v33-70b)
  - [LLaMA v3.3 70B - RTX 6000 Pro Blackwell Server Edition](#llama-v33-70b-rtx-configurations)
- [Qwen3 235B A22B](#qwen3-235b-a22b)
  - [Qwen3 235B A22B - RTX 6000 Pro Blackwell Server Edition](#qwen3-235b-a22b-rtx-configurations)
- [Qwen3 30B A3B](#qwen3-30b-a3b)
  - [Qwen3 30B A3B - RTX 6000 Pro Blackwell Server Edition](#qwen3-30b-a3b-rtx-configurations)

---

<a id="deepseek-r1-0528"></a>

# Deepseek R1 0528

| Sequence Length (ISL/OSL) | B200<br/>DEP4 (FP4) | B300* (beta)<br/>DEP4 (FP4) | GB200<br/>DEP4 (FP4) | GB300* (beta)<br/>DEP4 (FP4) | H200<br/>DEP8 (FP8) |
|---|---|---|---|---|---|
| 1024/1024 | 5,757 | 6,046 | 4,989 | 4,981 | 1,724 |
| 1024/8192 | 4,228 | 6,235 | 4,474 | 6,842 | 1,335 |
| 1024/32768 | 1,472 | 2,177 | 1,582 | 2,904 | |
| 8192/1024 | 1,176 | 1,489 | 1,276 | 1,599 | |
| 32768/1024 | | | | 335 | |

*Unit: output tokens per second per GPU*

---

<a id="gpt-oss-120b"></a>

# GPT-OSS 120B

| Sequence Length (ISL/OSL) | B200<br/>DEP2 (FP4) | B300* (beta)<br/>TP1 (FP4) | GB200<br/>TP1 (FP4) | GB300* (beta)<br/>TP1 (FP4) | H200<br/>DEP2 (FP8) |
|---|---|---|---|---|---|
| 1024/1024 | 29,061 | 34,737 | 29,056 | 38,702 | 6,391 |
| 1024/8192 | 16,434 | 18,338 | 17,079 | 19,866 | 4,260 |
| 1024/32768 | 7,020 | 7,284 | 6,313 | 7,958 | |
| 8192/1024 | 6,938 | 8,361 | 7,874 | 9,407 | 1,881 |
| 32768/1024 | 1,623 | 2,052 | 1,929 | 2,307 | 500 |

*Unit: output tokens per second per GPU*

---

<a id="gpt-oss-20b"></a>

# GPT-OSS 20B

| Sequence Length (ISL/OSL) | B200<br/>TP1 (FP4) | B300* (beta)<br/>TP1 (FP4) | GB200<br/>TP1 (FP4) | GB300* (beta)<br/>TP1 (FP4) | H200<br/>TP1 (FP8) |
|---|---|---|---|---|---|
| 1024/1024 | 59,084 | 60,925 | 51,870 | 53,604 | 14,719 |
| 1024/8192 | 37,824 | 42,986 | 35,280 | 44,998 | 12,441 |
| 1024/32768 | 15,799 | 17,461 | 17,414 | 19,147 | 4,240 |
| 8192/1024 | 13,439 | 14,955 | 15,466 | 16,880 | 4,196 |
| 32768/1024 | 2,970 | 3,408 | 3,331 | 3,833 | 965 |

*Unit: output tokens per second per GPU*

---

<a id="llama-v33-70b"></a>

# LLaMA v3.3 70B

| Sequence Length (ISL/OSL) | B200<br/>TP1 (FP4) | B300* (beta)<br/>TP1 (FP4) | GB200<br/>TP1 (FP4) | GB300* (beta)<br/>TP1 (FP4) | H200<br/>TP2 (FP8) |
|---|---|---|---|---|---|
| 1024/1024 | 6,943 | 8,196 | 7,576 | 8,910 | 2,637 |
| 1024/8192 | 3,270 | 3,926 | 3,856 | 4,739 | 1,995 |
| 8192/1024 | 1,347 | 1,597 | 1,593 | 1,900 | 544 |
| 32768/1024 | 279 | 351 | 315 | 418 | 120 |

*Unit: output tokens per second per GPU*

---

<a id="llama-v33-70b-rtx-6000-pro-blackwell-server-edition"></a>

# LLaMA v3.3 70B - RTX 6000 Pro Blackwell Server Edition

*Shows Tensor Parallel (TP) and Pipeline Parallel (PP) configurations*

| Sequence Length (ISL/OSL) | **1 GPUs**<br/>TP1,PP1 (FP4) | **2 GPUs**<br/>TP1,PP2 (FP4) | **4 GPUs**<br/>TP1,PP4 (FP4) | **8 GPUs**<br/>TP1,PP8 (FP4) |
|---|---|---|---|---|
| 1024/1024 | 1,724 | 1,881 | 1,798 | 1,545 |
| 1024/8192 | | | 675 | 630 |
| 8192/1024 | 306 | 329 | 323 | 307 |
| 32768/1024 | | 66 | 66 | 64 |

*Unit: output tokens per second per GPU*

---

<a id="qwen3-235b-a22b"></a>

# Qwen3 235B A22B

| Sequence Length (ISL/OSL) | B200<br/>DEP4 (FP4) | B300* (beta)<br/>DEP4 (FP4) | GB200<br/>DEP4 (FP4) | GB300* (beta)<br/>DEP4 (FP4) | H200<br/>DEP4 (FP8) |
|---|---|---|---|---|---|
| 1024/1024 | 6,423 | 8,143 | 6,777 | 8,430 | 3,494 |
| 1024/8192 | 3,881 | 4,938 | 3,955 | 5,110 | 1,677 |
| 1024/32768 | 1,216 | 1,922 | 1,240 | 1,841 | |
| 8192/1024 | 1,518 | 1,730 | 1,575 | 1,791 | 679 |
| 32768/1024 | 326 | 383 | 342 | 398 | 143 |

*Unit: output tokens per second per GPU*

---

<a id="qwen3-235b-a22b-rtx-6000-pro-blackwell-server-edition"></a>

# Qwen3 235B A22B - RTX 6000 Pro Blackwell Server Edition

*Shows Tensor Parallel (TP) and Pipeline Parallel (PP) configurations*

| Sequence Length (ISL/OSL) | **4 GPUs**<br/>TP1,PP4 (FP4) | **8 GPUs**<br/>TP1,PP8 (FP4) |
|---|---|---|
| 1024/1024 | 1,529 | 1,185 |
| 1024/8192 | 495 | 495 |
| 8192/1024 | 343 | 337 |
| 32768/1024 | 81 | 83 |

*Unit: output tokens per second per GPU*

---

<a id="qwen3-30b-a3b"></a>

# Qwen3 30B A3B

| Sequence Length (ISL/OSL) | B200<br/>TP1 (FP4) | B300* (beta)<br/>TP1 (FP4) | GB200<br/>TP1 (FP4) | GB300* (beta)<br/>TP1 (FP4) |
|---|---|---|---|---|
| 1024/1024 | 26,431 | 30,714 | 19,682 | 22,536 |
| 1024/8192 | 13,940 | 15,182 | 14,916 | 16,920 |
| 1024/32768 | 4,570 | 5,209 | | |
| 8192/1024 | 5,945 | 6,374 | 6,450 | 6,947 |
| 32768/1024 | 1,284 | 1,452 | 1,398 | 1,575 |

*Unit: output tokens per second per GPU*

---

<a id="qwen3-30b-a3b-rtx-6000-pro-blackwell-server-edition"></a>

# Qwen3 30B A3B - RTX 6000 Pro Blackwell Server Edition

*Shows Tensor Parallel (TP) and Pipeline Parallel (PP) configurations*

| Sequence Length (ISL/OSL) | **1 GPUs**<br/>TP1,PP1 (FP4) | **2 GPUs**<br/>TP1,PP2 (FP4) | **4 GPUs**<br/>TP1,PP4 (FP4) | **8 GPUs**<br/>TP1,PP8 (FP4) |
|---|---|---|---|---|
| 1024/1024 | 9,908 | 8,187 | 6,931 | 3,065 |
| 1024/8192 | 3,628 | 3,281 | 3,024 | 1,968 |
| 1024/32768 | | | | 918 |
| 8192/1024 | 1,925 | 1,805 | 1,656 | 1,193 |
| 32768/1024 | 372 | 356 | 349 | 318 |

*Unit: output tokens per second per GPU*

---

## Reproducing Benchmarked Results

The following tables provide reference commands used in the benchmarking process. For a more detailed description, see the [benchmarking suite documentation](./perf-benchmarking.md).

### Command Overview

This workflow uses the PyTorch backend and does not require building an engine.

| Stage | Description | Command |
| :- | - | - |
| [Dataset](#preparing-a-dataset) | Create a synthetic dataset | `python benchmarks/cpp/prepare_dataset.py --tokenizer=$model_name --stdout token-norm-dist --num-requests=$num_requests --input-mean=$isl --output-mean=$osl --input-stdev=0 --output-stdev=0 > $dataset_file` |
| [Run](#running-the-benchmark) | Run a benchmark with a dataset | `trtllm-bench --model $model_name throughput --dataset $dataset_file --backend pytorch --config $llm_options` |

### Variables

| Name | Description |
| :- | - |
| `$isl` | Benchmark input sequence length. |
| `$osl` | Benchmark output sequence length. |
| `$tp_size` | Tensor parallel mapping degree to run the benchmark with |
| `$pp_size` | Pipeline parallel mapping degree to run the benchmark with |
| `$ep_size` | Expert parallel mapping degree to run the benchmark with |
| `$model_name` | HuggingFace model name eg. meta-llama/Llama-2-7b-hf or use the path to a local weights directory |
| `$dataset_file` | Location of the dataset file generated by `prepare_dataset.py` |
| `$num_requests` | The number of requests to generate for dataset generation |
| `$seq_len` | A sequence length of ISL + OSL |
| `$llm_options` | (optional) A yaml file containing additional options for the LLM API |

### Preparing a Dataset

Use the provided [script](source:benchmarks/cpp/prepare_dataset.py) to generate a synthetic dataset.
Run the following command:

```shell
python benchmarks/cpp/prepare_dataset.py --tokenizer=$model_name --stdout token-norm-dist --num-requests=$num_requests --input-mean=$isl --output-mean=$osl --input-stdev=0 --output-stdev=0 > $dataset_file
```

This generates a text file at the path specified by `$dataset_file` where all requests share the same
input/output sequence length combination. The script uses the tokenizer to retrieve the vocabulary size and
randomly samples token IDs to create entirely random sequences. All requests are uniform
because the standard deviations for both input and output sequences are set to 0.


The table below lists the `$num_requests` used for each input/output sequence length combination. Shorter sequences cycle through faster and need more requests to reach steady state; longer sequences need fewer.


| Input Length | Output Length | Number of Requests |
|--------------|---------------|---------------------|
| 1024         | 1024          | 3000                |
| 8192         | 1024          | 1500                |
| 1024         | 8192          | 1500                |
| 32768        | 1024          | 1000                |
| 1024         | 32768         | 1000                |

### Running the Benchmark

Run the benchmark using the `trtllm-bench throughput` subcommand. This runs an offline maximum throughput scenario where all requests are queued in rapid succession. Provide a model name (HuggingFace reference or path to a local model), a [generated dataset](#preparing-a-dataset), and a YAML config file with additional LLM options (see [tensorrt_llm/llmapi/llm_args.py:LlmArgs](source:tensorrt_llm/llmapi/llm_args.py)).

For dense / non-MoE models:
```shell
trtllm-bench --tp $tp_size --pp $pp_size --model $model_name throughput --dataset $dataset_file --backend pytorch --config $llm_options  --concurrency -1
```
Llama 3.3

`llm_options.yml`
```yaml
cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024, 2048, 4096, 8192]
```

For MoE models:

```shell
trtllm-bench --tp $tp_size --pp $pp_size --ep $ep_size --model $model_name throughput --dataset $dataset_file --backend pytorch --config $llm_options
```

GPT-OSS:

`llm_options.yml`
```yaml
cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024, 2048, 4096, 8192]
enable_attention_dp: true
kv_cache_config:
  dtype: fp8
  # Hopper: use auto
moe_config:
  backend: TRTLLM
  # Hopper: use TRITON
```

DeepSeek R1:

`llm_options.yml`
```yaml
attention_dp_config:
  batching_wait_iters: 0
  enable_balance: true
  timeout_iters: 60
enable_attention_dp: true
cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024, 2048, 4096, 8192]
moe_config:
  backend: CUTLASS
kv_cache_config:
  dtype: fp8
```

Kimi K2:

`llm_options.yml`
```yaml
enable_attention_dp: true
cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256, 384]
moe_config:
  backend: CUTLASS
kv_cache_config:
  dtype: auto
```

Qwen3 MoE, Llama4 Maverick:

`llm_options.yml`
```yaml
enable_attention_dp: true
cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024, 2048, 4096, 8192]
```

Results are printed to the terminal upon benchmark completion. For example:

```shell
===========================================================
= PERFORMANCE OVERVIEW
===========================================================
Request Throughput (req/sec):                     43.2089
Total Output Throughput (tokens/sec):             5530.7382
Per User Output Throughput (tokens/sec/user):     2.0563
Per GPU Output Throughput (tokens/sec/gpu):       5530.7382
Total Token Throughput (tokens/sec):              94022.5497
Total Latency (ms):                               115716.9214
Average request latency (ms):                     75903.4456
Per User Output Speed [1/TPOT] (tokens/sec/user): 5.4656
Average time-to-first-token [TTFT] (ms):          52667.0339
Average time-per-output-token [TPOT] (ms):        182.9639

-- Per-Request Time-per-Output-Token [TPOT] Breakdown (ms)

[TPOT] MINIMUM: 32.8005
[TPOT] MAXIMUM: 208.4667
[TPOT] AVERAGE: 182.9639
[TPOT] P50    : 204.0463
[TPOT] P90    : 206.3863
[TPOT] P95    : 206.5064
[TPOT] P99    : 206.5821

-- Per-Request Time-to-First-Token [TTFT] Breakdown (ms)

[TTFT] MINIMUM: 3914.7621
[TTFT] MAXIMUM: 107501.2487
[TTFT] AVERAGE: 52667.0339
[TTFT] P50    : 52269.7072
[TTFT] P90    : 96583.7187
[TTFT] P95    : 101978.4566
[TTFT] P99    : 106563.4497

-- Request Latency Breakdown (ms) -----------------------

[Latency] P50    : 78509.2102
[Latency] P90    : 110804.0017
[Latency] P95    : 111302.9101
[Latency] P99    : 111618.2158
[Latency] MINIMUM: 24189.0838
[Latency] MAXIMUM: 111668.0964
[Latency] AVERAGE: 75903.4456
```

> [!WARNING] If the benchmarker does not print any output, this typically indicates an out-of-memory issue. Reduce the KV cache percentage
using the `--kv_cache_free_gpu_mem_fraction` option to lower memory usage.
