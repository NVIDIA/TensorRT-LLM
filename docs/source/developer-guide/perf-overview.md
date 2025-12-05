(perf-overview)=

# Overview

This document summarizes performance measurements of TensorRT-LLM on a number of GPUs across a set of key models.

The data in the following tables is provided as a reference point to help users validate observed performance.
It should *not* be considered as the peak performance that can be delivered by TensorRT-LLM.

Not all configurations were tested for all GPUs. 

We attempted to keep commands as simple as possible to ease reproducibility and left many options at their default settings.
Tuning batch sizes, parallelism configurations, and other options may lead to improved performance depending on your situaiton.

For DeepSeek R1 performance, please check out our [performance guide](../blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md)

For more information on benchmarking with `trtllm-bench` see this NVIDIA [blog post](https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/).

## Throughput Measurements

The below table shows performance data where a local inference client is fed requests at an infinite rate (no delay between messages),
and shows the throughput scenario under maximum load. The reported metric is `Total Output Throughput (tokens/sec)`.

The performance numbers below were collected using the steps described in this document.

Testing was performed on models with weights quantized using [ModelOpt](https://nvidia.github.io/TensorRT-Model-Optimizer/#) and published by NVIDIA on the [Model Optimizer HuggingFace Collection](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4).

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

*This document contains performance data for all tested networks, organized alphabetically.*
*Total networks: 8*

## Units

All performance values are measured in **output tokens per second per GPU**.

## Table of Contents

- [Deepseek R1 0528](#Deepseek_R1_0528)
- [GPT-OSS 120B](#GPT-OSS_120B)
- [GPT-OSS 20B](#GPT-OSS_20B)
- [LLaMA v3.3 70B](#LLaMA_v3.3_70B)
  - [LLaMA v3.3 70B - RTX Configurations](#LLaMA_v3.3_70B_rtx)
- [LLaMA v4 Maverick](#LLaMA_v4_Maverick)
  - [LLaMA v4 Maverick - RTX Configurations](#LLaMA_v4_Maverick_rtx)
- [Qwen3 235B A22B](#Qwen3_235B_A22B)
  - [Qwen3 235B A22B - RTX Configurations](#Qwen3_235B_A22B_rtx)
- [Qwen3 30B A3B](#Qwen3_30B_A3B)
  - [Qwen3 30B A3B - RTX Configurations](#Qwen3_30B_A3B_rtx)

---

<a id="Deepseek_R1_0528"></a>

# Deepseek R1 0528

| Sequence Length (ISL/OSL) | B200<br/>DEP4 (FP4) | GB200<br/>DEP4 (FP4) | H200<br/>DEP8 (FP8) |
|---|---|---|---|
| 1000/1000 | 5,715 | 5,884 | 1,627 |
| 1024/1024 | 5,610 | 6,169 | 1,620 |
| 1024/8192 | 3,841 | 4,201 | 1,218 |
| 1024/32768 | 1,409 | 1,465 | 438 |
| 8192/1024 | 1,100 | 1,182 | |

---

<a id="GPT-OSS_120B"></a>

# GPT-OSS 120B

| Sequence Length (ISL/OSL) | B200<br/>DEP2 (FP4) | GB200<br/>TP1 (FP4) | H200<br/>TP1 (FP8) | H100<br/>DEP4 (FP8) |
|---|---|---|---|---|
| 1000/1000 | 25,926 | 27,198 | 6,602 | 4,685 |
| 1024/1024 | 25,833 | 26,609 | 6,798 | 4,715 |
| 1024/8192 | 17,277 | 14,800 | 3,543 | |
| 1024/32768 | 6,272 | 5,556 | | 1,177 |
| 8192/1024 | 6,094 | 6,835 | 1,828 | 1,169 |
| 32768/1024 | 1,388 | 1,645 | 519 | 333 |

---

<a id="GPT-OSS_20B"></a>

# GPT-OSS 20B

| Sequence Length (ISL/OSL) | B200<br/>TP1 (FP4) | GB200<br/>TP1 (FP4) | H200<br/>TP1 (FP8) | H100<br/>TP1 (FP8) |
|---|---|---|---|---|
| 1000/1000 | 53,761 | 55,823 | 13,858 | 11,557 |
| 1024/1024 | 53,112 | 56,528 | 13,248 | 11,403 |
| 1024/8192 | 34,665 | 38,100 | 12,743 | 8,617 |
| 1024/32768 | 14,560 | 16,463 | | |
| 8192/1024 | 11,898 | 12,941 | 3,848 | 3,366 |
| 32768/1024 | 2,641 | 2,905 | 875 | 785 |

---


<a id="LLaMA_v3.3_70B"></a>

# LLaMA v3.3 70B

| Sequence Length (ISL/OSL) | B200<br/>TP1 (FP4) | GB200<br/>TP1 (FP4) | H200<br/>TP1 (FP8) | H100<br/>TP2 (FP8) |
|---|---|---|---|---|
| 1000/1000 | 6,897 | 7,769 | 2,646 | 2,209 |
| 1024/1024 | 6,841 | 7,751 | 2,785 | |
| 1024/8192 | 3,240 | 3,805 | | |
| 8192/1024 | 1,362 | 1,491 | 521 | 398 |
| 32768/1024 | 274 | 302 | | |

---

<a id="LLaMA_v3.3_70B_rtx"></a>

# LLaMA v3.3 70B - RTX Configurations (TP/PP)

*Shows Tensor Parallel (TP) and Pipeline Parallel (PP) configurations*
*Shows only the best configuration per GPU count based on throughput per GPU*

| Sequence Length (ISL/OSL) | **1 GPUs**<br/>TP1,PP1 (FP4) | **2 GPUs**<br/>TP1,PP2 (FP4) |
|---|---|---|
| 1000/1000 | 1,454 | 1,524 |
| 1024/1024 | 1,708 | 1,887 |
| 8192/1024 | 296 | 327 |
| 32768/1024 | | 67 |

---

<a id="LLaMA_v4_Maverick"></a>

# LLaMA v4 Maverick

| Sequence Length (ISL/OSL) | B200<br/>DEP4 (FP4) | GB200<br/>DEP4 (FP4) | H200<br/>DEP8 (FP8) |
|---|---|---|---|
| 1000/1000 | 11,328 | 11,828 | 4,146 |
| 1024/1024 | 11,227 | 11,905 | 4,180 |
| 1024/8192 | 5,164 | 5,508 | 1,157 |
| 1024/32768 | 2,187 | 2,300 | 679 |
| 8192/1024 | 3,277 | 3,444 | 1,276 |
| 32768/1024 | 858 | 963 | |

---

<a id="LLaMA_v4_Maverick_rtx"></a>

# LLaMA v4 Maverick - RTX Configurations (TP/PP)

*Shows Tensor Parallel (TP) and Pipeline Parallel (PP) configurations*
*Shows only the best configuration per GPU count based on throughput per GPU*

| Sequence Length (ISL/OSL) | **4 GPUs**<br/>DEP4,PP1 (FP4) | **8 GPUs**<br/>DEP8,PP1 (FP4) |
|---|---|---|
| 1000/1000 | | 1,582 |
| 1024/1024 | 1,734 | |
| 1024/8192 | 620 | 638 |

---

<a id="Qwen3_235B_A22B"></a>

# Qwen3 235B A22B

| Sequence Length (ISL/OSL) | B200<br/>DEP4 (FP4) | GB200<br/>DEP4 (FP4) | H200<br/>DEP4 (FP8) | H100<br/>DEP8 (FP8) |
|---|---|---|---|---|
| 1000/1000 | 5,758 | 6,172 | 3,165 | 1,932 |
| 1024/1024 | 5,752 | 5,862 | 3,268 | 1,935 |
| 1024/8192 | 3,388 | 3,423 | 1,417 | 873 |
| 1024/32768 | 1,255 | | | |
| 8192/1024 | 1,409 | 1,464 | 627 | |
| 32768/1024 | 318 | 333 | 134 | |

---

<a id="Qwen3_235B_A22B_rtx"></a>

# Qwen3 235B A22B - RTX Configurations (TP/PP)

*Shows Tensor Parallel (TP) and Pipeline Parallel (PP) configurations*
*Shows only the best configuration per GPU count based on throughput per GPU*

| Sequence Length (ISL/OSL) | **4 GPUs**<br/>DEP4,PP1 (FP4) | **8 GPUs**<br/>DEP8,PP1 (FP4) |
|---|---|---|
| 1000/1000 | 1,378 | 946 |
| 1024/1024 | 1,377 | 941 |
| 1024/8192 | 656 | 669 |
| 8192/1024 | 246 | 177 |
| 32768/1024 | 57 | |

---

<a id="Qwen3_30B_A3B"></a>

# Qwen3 30B A3B

| Sequence Length (ISL/OSL) | B200<br/>TP1 (FP4) | GB200<br/>TP1 (FP4) |
|---|---|---|
| 1000/1000 | 26,941 | 22,856 |
| 1024/1024 | 26,587 | 22,201 |
| 1024/8192 | 13,492 | 14,272 |
| 1024/32768 | 4,494 | 4,925 |
| 8192/1024 | 5,730 | 6,201 |
| 32768/1024 | 1,263 | 1,380 |

---

<a id="Qwen3_30B_A3B_rtx"></a>

# Qwen3 30B A3B - RTX Configurations (TP/PP)

*Shows Tensor Parallel (TP) and Pipeline Parallel (PP) configurations*
*Shows only the best configuration per GPU count based on throughput per GPU*

| Sequence Length (ISL/OSL) | **2 GPUs**<br/>DEP2,PP1 (FP4) | **4 GPUs**<br/>DEP2,PP2 (FP4) | **8 GPUs**<br/>DEP4,PP2 (FP4) | **1 GPUs**<br/>TP1,PP1 (FP4) |
|---|---|---|---|---|
| 1000/1000 | 7,844 | 6,423 | 5,034 | 9,142 |
| 1024/1024 | 7,203 | 6,350 | 4,856 | 8,999 |
| 1024/8192 | 3,289 | 2,708 | 2,300 | 3,335 |
| 1024/32768 | | | 1,083 | |
| 8192/1024 | 1,447 | 1,384 | 973 | 1,758 |
| 32768/1024 | 251 | 242 | 220 | 324 |

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
| [Run](#running-the-benchmark) | Run a benchmark with a dataset | `trtllm-bench --model $model_name throughput --dataset $dataset_file --backend pytorch --extra_llm_api_options $llm_options` |

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
| 100000       | 1024          | 300                 |

### Running the Benchmark

To run the benchmark with the generated data set, simply use the `trtllm-bench throughput` subcommand. The benchmarker will
run an offline maximum throughput scenario such that all requests are queued in rapid succession. You simply need to provide
a model name (HuggingFace reference or path to a local model), a [generated dataset](#preparing-a-dataset), and a file containing any desired extra options to the LLM APIs (details in [tensorrt_llm/llmapi/llm_args.py:LlmArgs](source:tensorrt_llm/llmapi/llm_args.py)).

For dense / non-MoE models:

```shell
trtllm-bench --tp $tp_size --pp $pp_size --model $model_name throughput --dataset $dataset_file --backend pytorch --extra_llm_api_options $llm_options
```

`llm_options.yml`
```yaml
cuda_graph_config:
  enable_padding: true
  batch_sizes:
    - 1
    - 2
    - 4
    - 8
    - 16
    - 32
    - 64
    - 128
    - 256
    - 384
    - 512
    - 1024
    - 2048
    - 4096
    - 8192
```

For MoE models:

```shell
trtllm-bench --tp $tp_size --pp $pp_size --ep $ep_size --model $model_name throughput --dataset $dataset_file --backend pytorch --extra_llm_api_options $llm_options
```

`llm_options.yml`
```yaml
enable_attention_dp: true
cuda_graph_config:
  enable_padding: true
  batch_sizes:
    - 1
    - 2
    - 4
    - 8
    - 16
    - 32
    - 64
    - 128
    - 256
    - 384
    - 512
    - 1024
    - 2048
    - 4096
    - 8192
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
