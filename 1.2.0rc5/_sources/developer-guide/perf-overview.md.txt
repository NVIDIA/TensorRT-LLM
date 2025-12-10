(perf-overview)=

# Overview

This document summarizes performance measurements of TensorRT-LLM on a number of GPUs across a set of key models.

The data in the following tables is provided as a reference point to help users validate observed performance.
It should *not* be considered as the peak performance that can be delivered by TensorRT-LLM.

We attempted to keep commands as simple as possible to ease reproducibility and left many options at their default settings.
Tuning batch sizes, parallelism configurations, and other options may lead to improved performance depending on your situaiton.

For DeepSeek R1 performance, please check out our [performance guide](../blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md)

For more information on benchmarking with `trtllm-bench` see this NVIDIA [blog post](https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/).

## Throughput Measurements

The below table shows performance data where a local inference client is fed requests at an infinite rate (no delay between messages),
and shows the throughput scenario under maximum load. The reported metric is `Total Output Throughput (tokens/sec)`.

The performance numbers below were collected using the steps described in this document.

Testing was performed on models with weights quantized using [ModelOpt](https://nvidia.github.io/TensorRT-Model-Optimizer/#) and published by NVIDIA on the [Model Optimizer HuggingFace Collection](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4).

*(NEW for v1.0) RTX 6000 Pro Blackwell Server Edition Benchmarks:*

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
nvidia/Llama-3.3-70B-Instruct-FP4
nvidia/Llama-3.1-405B-Instruct-FP4
nvidia/Qwen3-235B-A22B-FP4
nvidia/Qwen3-30B-A3B-FP4
nvidia/DeepSeek-R1-0528-FP4
```

### FP8 Models

```text
nvidia/Llama-3.1-8B-Instruct-FP8
nvidia/Llama-3.3-70B-Instruct-FP8
nvidia/Llama-3.1-405B-Instruct-FP8
nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8
nvidia/Qwen3-235B-A22B-FP8
```

#### Llama 4 Scout

| Sequence Length (ISL/OSL) | B200<br/>TP1 (FP4) | GB200<br/>TP1 (FP4) | H200<br/>TP4 (FP8) | H100<br/>TP4 (FP8) |
|---------------------------|---------------------|---------------------|-------------------|-------------------|
| 128/2048                 | 14,699              | 15,238             | 34,316           | 15,130           |
| 128/4096                 | 8,932               | 9,556              | 21,332           | 8,603            |
| 500/2000                 | 11,977              | 11,795             | 24,630           | 12,399           |
| 1000/1000                | 10,591              | 7,738              | 21,636           | 12,129           |
| 1000/2000                | 9,356               | 8,581              | 18,499           | 9,838            |
| 2048/128                 | 3,137               | 3,295              | 3,699            | 3,253            |
| 2048/2048                | 7,152               | 7,464              | 14,949           | 7,972            |
| 5000/500                 | 2,937               | 3,107              | 4,605            | 3,342            |
| 20000/2000               | 1,644               | 1,767              | 2,105            |                  |

RTX 6000 Pro Blackwell Server Edition
| Sequence Length (ISL/OSL) | **4 GPUs**<br/>TP2,PP2 (FP4) | **8 GPUs**<br/>TP4,PP2 (FP4) |
|---|---|---|
| 128/2048 | 12,321 | 21,035 |
| 128/4096 | 7,643 | 13,421 |
| 1000/1000 | 9,476 | 15,781 |
| 1000/2000 | 8,919 | 16,434 |
| 2048/128 | 2,615 | 2,941 |
| 2048/2048 | 6,208 | 10,410 |
| 5000/500 | 2,662 | |

#### Llama 3.3 70B

| Sequence Length (ISL/OSL) | B200<br/>TP1 (FP4) | GB200<br/>TP1 (FP4) | H200<br/>TP1 (FP8) | H100<br/>TP2 (FP8) |
|---|---|---|---|---|
| 128/2048 | 9,922 | 11,309 | 4,336 | 6,651 |
| 128/4096 | 6,831 | 7,849 | 2,872 | 4,199 |
| 500/2000 | 7,762 | 9,028 | 3,666 | 5,222 |
| 1000/1000 | 7,007 | 7,326 | 2,909 | 4,205 |
| 1000/2000 | 6,271 | 6,513 | 2,994 | 4,146 |
| 2048/128 | 1,339 | 1,450 | 442 | 762 |
| 2048/2048 | 4,783 | 5,646 | 2,003 | 3,082 |
| 5000/500 | 1,459 | 1,602 | 566 | 898 |
| 20000/2000 | 665 | 755 | 283 | 437 |

RTX 6000 Pro Blackwell Server Edition
| Sequence Length (ISL/OSL) | **1 GPUs**<br/>TP1,PP1 (FP4) | **2 GPUs**<br/>TP1,PP2 (FP4) | **4 GPUs**<br/>TP1,PP4 (FP4) | **8 GPUs**<br/>TP1,PP8 (FP4) |
|---|---|---|---|---|
| 128/2048 | 2,422 | 4,993 | 7,922 | 9,833 |
| 128/4096 | 1,349 | 2,893 | 4,978 | 7,352 |
| 500/2000 | 1,856 | 4,114 | 6,939 | 9,435 |
| 1000/1000 | 1,787 | 3,707 | 5,961 | 8,166 |
| 1000/2000 | 1,594 | 2,993 | 5,274 | 6,943 |
| 2048/128 | 393 | 813 | 1,511 | 2,495 |
| 2048/2048 | 1,074 | 2,336 | 3,870 | 6,078 |
| 5000/500 | 401 | 812 | 1,511 | 2,491 |
| 20000/2000 | 142 | 319 | 630 | 1,148 |

#### Qwen3-235B-A22B

| Sequence Length (ISL/OSL) | B200<br/>TP8 (FP4) | H200<br/>TP8 (FP8) | H100<br/>TP8 (FP8) |
|---|---|---|---|
| 128/2048 | 66,057 | 42,821 | 19,658 |
| 128/4096 | 39,496 | 26,852 | 12,447 |
| 500/2000 | 57,117 | 28,026 | 18,351 |
| 1000/1000 | 42,391 | 23,789 | 14,898 |
| 1000/2000 | 34,105 | 22,061 | 15,136 |
| 2048/128 | 7,329 | 3,331 | |
| 2048/2048 | 26,854 | 16,672 | 9,924 |
| 5000/500 | 8,190 | 3,623 | 3,225 |
| 20000/2000 | 4,453 | 1,876 | |

RTX 6000 Pro Blackwell Server Edition
| Sequence Length (ISL/OSL) | **8 GPUs**<br/>TP2,PP4 (FP4) |
|---|---|
| 128/2048 | 12,494 |
| 128/4096 | 7,715 |
| 500/2000 | 11,157 |
| 1000/1000 | 10,697 |
| 1000/2000 | 10,109 |
| 2048/128 | 3,181 |
| 2048/2048 | 6,712 |
| 5000/500 | 3,173 |

#### Qwen3-30B-A3B

| Sequence Length (ISL/OSL) | B200<br/>TP1 (FP4) |
|---|---|
| 128/2048 | 37,844 |
| 128/4096 | 24,953 |
| 500/2000 | 27,817 |
| 1000/1000 | 25,828 |
| 1000/2000 | 22,051 |
| 2048/128 | 6,251 |
| 2048/2048 | 17,554 |
| 5000/500 | 6,142 |
| 20000/2000 | 2,944 |

RTX 6000 Pro Blackwell Server Edition
| Sequence Length (ISL/OSL) | **1 GPUs**<br/>TP1,PP1 (FP4) | **2 GPUs**<br/>TP2,PP1 (FP4) | **4 GPUs**<br/>TP4,PP1 (FP4) | **8 GPUs**<br/>TP8,PP1 (FP4) |
|---|---|---|---|---|
| 128/2048 | 12,540 | 22,744 | 35,715 | 52,676 |
| 128/4096 | 7,491 | 15,049 | 28,139 | 33,895 |
| 500/2000 | 10,695 | 17,266 | 26,175 | 44,088 |
| 1000/1000 | 9,910 | 16,431 | 24,046 | 31,785 |
| 1000/2000 | 8,378 | 13,323 | 25,131 | 28,881 |
| 2048/128 | 3,257 | 3,785 | 4,311 | 4,798 |
| 2048/2048 | 5,908 | 10,679 | 18,134 | 22,391 |
| 5000/500 | 2,530 | 3,799 | 5,212 | 5,965 |
| 20000/2000 | 871 | 1,558 | 2,551 | |

#### DeepSeek R1

| Sequence Length (ISL/OSL) | B200<br/>TP8 (FP4) |
|---|---|
| 128/2048 | 62,599 |
| 128/4096 | 44,046 |
| 1000/1000 | 37,634 |
| 1000/2000 | 40,538 |
| 2048/128 | 5,026 |
| 2048/2048 | 28,852 |

#### Llama 4 Maverick

| Sequence Length (ISL/OSL) | B200<br/>TP8 (FP4) | H200<br/>TP8 (FP8) | H100<br/>TP8 (FP8) |
|---|---|---|---|
| 128/2048 | 112,676 | 40,572 | 10,829 |
| 128/4096 | 68,170 | 24,616 | 6,744 |
| 500/2000 | | 37,835 | 10,108 |
| 1000/1000 | 79,617 | 31,782 | 9,677 |
| 1000/2000 | 63,766 | 34,734 | 9,151 |
| 2048/128 | 18,088 | 7,307 | |
| 2048/2048 | 52,195 | 20,957 | 6,916 |
| 5000/500 | | 8,456 | 3,457 |
| 20000/2000 | 12,678 | 4,106 | |

RTX 6000 Pro Blackwell Server Edition
| Sequence Length (ISL/OSL) | **8 GPUs**<br/>TP4,PP2 (FP4) |
|---|---|
| 128/2048 | 19,146 |
| 128/4096 | 12,165 |
| 500/2000 | 17,870 |
| 1000/1000 | 15,954 |
| 1000/2000 | 12,456 |
| 2048/128 | 4,463 |
| 2048/2048 | 10,727 |
| 5000/500 | 4,613 |

#### Llama 3.1 405B

| Sequence Length (ISL/OSL) | B200<br/>TP4 (FP4) | GB200<br/>TP4 (FP4) | H200<br/>TP8 (FP8) | H100<br/>TP8 (FP8) |
|---|---|---|---|---|
| 128/2048 | 8,020 | 8,151 | 5,348 | 4,340 |
| 128/4096 | 6,345 | 6,608 | 4,741 | 3,116 |
| 500/2000 | 6,244 | 6,540 | 4,724 | 3,994 |
| 1000/1000 | 5,209 | 5,389 | 3,330 | 2,919 |
| 1000/2000 | 4,933 | 5,135 | 3,722 | 2,895 |
| 2048/128 | 749 | 797 | 456 | 453 |
| 2048/2048 | 4,212 | 4,407 | 2,948 | 2,296 |
| 5000/500 | 1,048 | 1,112 | 650 | 610 |
| 20000/2000 | 672 | 739 | 505 | 345 |

RTX 6000 Pro Blackwell Server Edition
| Sequence Length (ISL/OSL) | **8 GPUs**<br/>TP1,PP8 (FP4) |
|---|---|
| 128/2048 | 2,981 |
| 1000/1000 | 2,369 |
| 1000/2000 | 1,931 |
| 2048/128 | 579 |
| 2048/2048 | 1,442 |

#### Llama 3.1 8B

| Sequence Length (ISL/OSL) | H200<br/>TP1 (FP8) | H100<br/>TP1 (FP8) |
|---|---|---|
| 128/2048 | 26,221 | 22,714 |
| 128/4096 | 18,027 | 14,325 |
| 500/2000 | 20,770 | 17,660 |
| 1000/1000 | 17,744 | 15,220 |
| 1000/2000 | 16,828 | 13,899 |
| 2048/128 | 3,538 | 3,450 |
| 2048/2048 | 12,194 | 9,305 |
| 5000/500 | 3,902 | 3,459 |
| 20000/2000 | 1,804 | 1,351 |


## Reproducing Benchmarked Results

```{note}
Only the models shown in the table above are supported by this workflow.
```

The following tables are references for commands that are used as part of the benchmarking process. For a more detailed description of this benchmarking workflow, see the [benchmarking suite documentation](./perf-benchmarking.md).

### Command Overview

Starting with v0.19, testing was performed using the PyTorch backend - this workflow does not require an engine to be built.

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


| Input Length | Output Length |  $seq_len  | $num_requests      |
| ------------ | ------------- | ---------- | ------------------ |
| 128          | 128           | 256        | 30000              |
| 128          | 2048          | 2176       | 3000               |
| 128          | 4096          | 4224       | 1500               |
| 1000         | 2000          | 3000       | 1500               |
| 2048         | 128           | 2176       | 3000               |
| 2048         | 2048          | 4096       | 1500               |
| 5000         | 500           | 5500       | 1500               |
| 1000         | 1000          | 2000       | 3000               |
| 500          | 2000          | 2500       | 3000               |
| 20000        | 2000          | 22000      | 1000               |

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
