(perf-overview)=

# Overview

This document summarizes performance measurements of TensorRT-LLM on a number of GPUs across a set of key models.

The data in the following tables is provided as a reference point to help users validate observed performance.
It should *not* be considered as the peak performance that can be delivered by TensorRT-LLM.

Not all configurations were tested for all GPUs.

We attempted to keep commands as simple as possible to ease reproducibility and left many options at their default settings.
Tuning batch sizes, parallelism configurations, and other options may lead to improved performance depending on your situation.


For DeepSeek R1 performance, please check out our [performance guide](../blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md)

For more information on benchmarking with `trtllm-bench` see this NVIDIA [blog post](https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/).

## Throughput Measurements

The below table shows performance data where a local inference client is fed requests at an high rate / no delay between messages,
and shows the throughput scenario under maximum load. The reported metric is `Output Throughput per GPU (tokens/sec/GPU)`.

The performance numbers below were collected using the steps described in this document.

Testing was performed on models with weights quantized using [ModelOpt](https://nvidia.github.io/Model-Optimizer/) and published by NVIDIA on the [Model Optimizer HuggingFace Collection](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4).

RTX 6000 Pro Blackwell Server Edition data is now included in the perf overview. RTX 6000 systems can benefit from enabling pipeline parallelism (PP) in LLM workloads, so we included several new benchmarks for this GPU at various TP x PP combinations. That data is presented in a separate table for each network.


### Hardware
The following GPU variants were used for testing:
- H100 SXM 80GB (DGX H100)
- H200 SXM 141GB (DGX H200)
- B200 180GB (DGX B200)
- GB200 192GB (GB200 NVL72)
- RTX 6000 Pro Blackwell Server Edition

Other hardware variants may have different TDP, memory bandwidth, core count, or other features leading to performance differences on these workloads.

### FP4 Models

```text
nvidia/DeepSeek-R1-0528-NVFP4-v2
nvidia/Qwen3-235B-A22B-FP4
nvidia/Qwen3-30B-A3B-FP4
nvidia/Llama-3.3-70B-Instruct-FP4
nvidia/Llama-4-Maverick-17B-128E-Instruct-NVFP4
```

### FP8 Models

```text
deepseek-ai/DeepSeek-R1-0528
nvidia/Qwen3-235B-A22B-FP8
nvidia/Llama-3.3-70B-Instruct-FP8
nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8
```

# Performance Summary - All Networks

## Units

All performance values are measured in `output tokens per second per GPU`, where `output tokens` includes the first and all subsequent generated tokens (input tokens are not included).

Data in these tables is taken from the `Per GPU Output Throughput (tps/gpu)` metric reported by `trtllm-bench`.
The calculations for metrics reported by trtllm-bench can be found in the dataclasses [reporting.py](../../../tensorrt_llm/bench/dataclasses/reporting.py#L570) and [statistics.py](../../../tensorrt_llm/bench/dataclasses/statistics.py#L188)


## Table of Contents

- [Deepseek R1 0528](#deepseek-r1-0528)
- [GPT-OSS 120B](#gpt-oss-120b)
- [GPT-OSS 20B](#gpt-oss-20b)
- [LLaMA v3.3 70B](#llama-v33-70b)
  - [LLaMA v3.3 70B - RTX 6000 Pro Blackwell Server Edition](#llama-v33-70b-rtx-configurations)
- [LLaMA v4 Maverick](#llama-v4-maverick)
- [Qwen3 235B A22B](#qwen3-235b-a22b)
  - [Qwen3 235B A22B - RTX 6000 Pro Blackwell Server Edition](#qwen3-235b-a22b-rtx-configurations)
- [Qwen3 30B A3B](#qwen3-30b-a3b)
  - [Qwen3 30B A3B - RTX 6000 Pro Blackwell Server Edition](#qwen3-30b-a3b-rtx-configurations)

---

<a id="deepseek-r1-0528"></a>

# Deepseek R1 0528

| Sequence Length (ISL/OSL) | B200<br/>DEP4 (FP4) | GB200<br/>DEP4 (FP4) | H200<br/>DEP8 (FP8) |
|---|---|---|---|
| 1000/1000 | 6,463 | 6,939 | 1,627 |
| 1024/1024 | 6,430 | 6,924 | 1,620 |
| 1024/8192 | 3,862 | 4,379 | 1,218 |
| 1024/32768 | 1,451 | 1,465 | 438 |
| 8192/1024 | 1,168 | 1,192 | |

unit: `output tokens per second per GPU`

---

<a id="gpt-oss-120b"></a>

# GPT-OSS 120B

| Sequence Length (ISL/OSL) | B200<br/>DEP2 (FP4) | GB200<br/>TP1 (FP4) | H200<br/>TP1 (FP8) | H100<br/>DEP4 (FP8) |
|---|---|---|---|---|
| 1000/1000 | 25,943 | 27,198 | 6,868 | 4,685 |
| 1024/1024 | 25,870 | 26,609 | 6,798 | 4,715 |
| 1024/8192 | 17,289 | 14,800 | 3,543 | |
| 1024/32768 | 6,279 | 5,556 | | 1,177 |
| 8192/1024 | 6,111 | 6,835 | 1,828 | 1,169 |
| 32768/1024 | 1,392 | 1,645 | 519 | 333 |

unit: `output tokens per second per GPU`

---

<a id="gpt-oss-20b"></a>

# GPT-OSS 20B

| Sequence Length (ISL/OSL) | B200<br/>TP1 (FP4) | GB200<br/>TP1 (FP4) | H200<br/>TP1 (FP8) | H100<br/>TP1 (FP8) |
|---|---|---|---|---|
| 1000/1000 | 53,812 | 55,823 | 13,858 | 11,557 |
| 1024/1024 | 53,491 | 56,528 | 13,890 | 11,403 |
| 1024/8192 | 34,702 | 38,100 | 12,743 | 8,617 |
| 1024/32768 | 14,589 | 16,463 | | |
| 8192/1024 | 11,904 | 12,941 | 4,015 | 3,366 |
| 32768/1024 | 2,645 | 2,905 | 915 | 785 |

unit: `output tokens per second per GPU`

---

<a id="llama-v33-70b"></a>

# LLaMA v3.3 70B

| Sequence Length (ISL/OSL) | B200<br/>TP1 (FP4) | GB200<br/>TP1 (FP4) | H200<br/>TP2 (FP8) | H100<br/>TP2 (FP8) |
|---|---|---|---|---|
| 1000/1000 | 6,920 | 7,769 | 2,587 | 2,209 |
| 1024/1024 | 6,842 | 7,751 | 2,582 | |
| 1024/8192 | 3,242 | 3,805 | 2,009 | |
| 8192/1024 | 1,362 | 1,491 | 537 | 398 |
| 32768/1024 | 274 | 302 | 120 | |

unit: `output tokens per second per GPU`

---

<a id="llama-v33-70b-rtx-configurations"></a>

# LLaMA v3.3 70B - RTX 6000 Pro Blackwell Server Edition

*Shows Tensor Parallel (TP) and Pipeline Parallel (PP) configurations*

| Sequence Length (ISL/OSL) | **1 GPUs**<br/>TP1,PP1 (FP4) | **2 GPUs**<br/>TP1,PP2 (FP4) |
|---|---|---|
| 1000/1000 | 1,724 | 1,901 |
| 1024/1024 | 1,708 | 1,887 |
| 8192/1024 | 296 | 327 |
| 32768/1024 | | 67 |

unit: `output tokens per second per GPU`

---

<a id="llama-v4-maverick"></a>

# LLaMA v4 Maverick

| Sequence Length (ISL/OSL) | B200<br/>DEP4 (FP4) | GB200<br/>DEP4 (FP4) | H200<br/>DEP8 (FP8) |
|---|---|---|---|
| 1000/1000 | 11,337 | 11,828 | 4,146 |
| 1024/1024 | 11,227 | 11,905 | 4,180 |
| 1024/8192 | 5,174 | 5,508 | 1,157 |
| 1024/32768 | 2,204 | 2,300 | 679 |
| 8192/1024 | 3,279 | 3,444 | 1,276 |
| 32768/1024 | 859 | 963 | |

unit: `output tokens per second per GPU`

---

<a id="qwen3-235b-a22b"></a>

# Qwen3 235B A22B

| Sequence Length (ISL/OSL) | B200<br/>DEP4 (FP4) | GB200<br/>DEP4 (FP4) | H200<br/>DEP4 (FP8) | H100<br/>DEP8 (FP8) |
|---|---|---|---|---|
| 1000/1000 | 5,764 | 6,172 | 3,288 | 1,932 |
| 1024/1024 | 5,756 | 5,862 | 3,268 | 1,935 |
| 1024/8192 | 3,389 | 3,423 | 1,417 | 873 |
| 1024/32768 | 1,255 | | | |
| 8192/1024 | 1,410 | 1,464 | 627 | |
| 32768/1024 | 319 | 333 | 134 | |

unit: `output tokens per second per GPU`

---

<a id="qwen3-235b-a22b-rtx-configurations"></a>

# Qwen3 235B A22B - RTX 6000 Pro Blackwell Server Edition

*Shows Tensor Parallel (TP) and Pipeline Parallel (PP) configurations*

| Sequence Length (ISL/OSL) | **4 GPUs**<br/>DEP2,PP2 (FP4) | **8 GPUs**<br/>DEP8,PP1 (FP4) |
|---|---|---|
| 1000/1000 | 1,731 | 969 |
| 1024/1024 | 1,732 | 963 |
| 1024/8192 | 644 | 711 |
| 32768/1024 | 70 | |

unit: `output tokens per second per GPU`

---

<a id="qwen3-30b-a3b"></a>

# Qwen3 30B A3B

| Sequence Length (ISL/OSL) | B200<br/>TP1 (FP4) | GB200<br/>TP1 (FP4) |
|---|---|---|
| 1000/1000 | 26,971 | 22,856 |
| 1024/1024 | 26,611 | 22,201 |
| 1024/8192 | 13,497 | 14,272 |
| 1024/32768 | 4,494 | 4,925 |
| 8192/1024 | 5,735 | 6,201 |
| 32768/1024 | 1,265 | 1,380 |

unit: `output tokens per second per GPU`

---

<a id="qwen3-30b-a3b-rtx-configurations"></a>

# Qwen3 30B A3B - RTX 6000 Pro Blackwell Server Edition

*Shows Tensor Parallel (TP) and Pipeline Parallel (PP) configurations*

| Sequence Length (ISL/OSL) | **2 GPUs**<br/>DEP2,PP1 (FP4) | **4 GPUs**<br/>DEP2,PP2 (FP4) | **8 GPUs**<br/>DEP8,PP1 (FP4) | **1 GPUs**<br/>TP1,PP1 (FP4) |
|---|---|---|---|---|
| 1000/1000 | 8,409 | 7,059 | 3,985 | 9,938 |
| 1024/1024 | | 7,019 | | 9,755 |
| 1024/8192 | 3,577 | | 2,406 | 3,621 |
| 8192/1024 | | 1,416 | | 1,914 |
| 32768/1024 | | | 180 | 374 |

unit: `output tokens per second per GPU`

---



## Reproducing Benchmarked Results

```{note}
Only the models shown in the table above are supported by this workflow.
```

The following tables are references for commands that are used as part of the benchmarking process. For a more detailed description of this benchmarking workflow, see the [benchmarking suite documentation](./perf-benchmarking.md).

### Command Overview

Testing was performed using the PyTorch backend - this workflow does not require an engine to be built.

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

In order to prepare a dataset, you can use the provided [script](source:benchmarks/cpp/prepare_dataset.py).
To generate a synthetic dataset, run the following command:

```shell
python benchmarks/cpp/prepare_dataset.py --tokenizer=$model_name --stdout token-norm-dist --num-requests=$num_requests --input-mean=$isl --output-mean=$osl --input-stdev=0 --output-stdev=0 > $dataset_file
```

The command will generate a text file located at the path specified `$dataset_file` where all requests are of the same
input/output sequence length combinations. The script works by using the tokenizer to retrieve the vocabulary size and
randomly sample token IDs from it to create entirely random sequences. In the command above, all requests will be uniform
because the standard deviations for both input and output sequences are set to 0.


For each input and output sequence length combination, the table below details the `$num_requests` that were used. For
shorter input and output lengths, a larger number of messages were used to guarantee that the system hit a steady state
because requests enter and exit the system at a much faster rate. For longer input/output sequence lengths, requests
remain in the system longer and therefore require less requests to achieve steady state.


| Input Length | Output Length | Number of Requests |
|--------------|---------------|---------------------|
| 1024         | 1024          | 3000                |
| 8192         | 1024          | 1500                |
| 1024         | 8192          | 1500                |
| 32768        | 1024          | 1000                |
| 1024         | 32768         | 1000                |

### Running the Benchmark

To run the benchmark with the generated data set, simply use the `trtllm-bench throughput` subcommand. The benchmarker will
run an offline maximum throughput scenario such that all requests are queued in rapid succession. You simply need to provide
a model name (HuggingFace reference or path to a local model), a [generated dataset](#preparing-a-dataset), and a file containing any desired extra options to the LLM APIs (details in [tensorrt_llm/llmapi/llm_args.py:LlmArgs](source:tensorrt_llm/llmapi/llm_args.py)).

For dense / non-MoE models:
```shell
trtllm-bench --tp $tp_size --pp $pp_size --model $model_name throughput --dataset $dataset_file --backend pytorch --config $llm_options
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
  backend: CUTLASS
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

Qwen3 MoE, Llama4 Maverick:

`llm_options.yml`
```yaml
enable_attention_dp: true
cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024, 2048, 4096, 8192]
```

In many cases, we also use a higher KV cache percentage by setting `--kv_cache_free_gpu_mem_fraction 0.95` in the benchmark command. This allows us to obtain better performance than the default setting of `0.90`. We fall back to `0.90` or lower if out-of-memory errors are encountered.

The results will be printed to the terminal upon benchmark completion. For example,

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

> [!WARNING] In some cases, the benchmarker may not print anything at all. This behavior usually
means that the benchmark has hit an out of memory issue. Try reducing the KV cache percentage
using the `--kv_cache_free_gpu_mem_fraction` option to lower the percentage of used memory.
