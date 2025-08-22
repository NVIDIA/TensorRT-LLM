(perf-overview)=

# Overview

This document summarizes performance measurements of TensorRT-LLM on a number of GPUs across a set of key models.

The data in the following tables is provided as a reference point to help users validate observed performance.
It should *not* be considered as the peak performance that can be delivered by TensorRT-LLM.

We attempted to keep commands as simple as possible to ease reproducibility and left many options at their default settings.
Tuning batch sizes, parallelism configurations, and other options may lead to improved performance depending on your situaiton.

For DeepSeek R1 performance, please check out our [performance guide](../blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md)

## Throughput Measurements

The below table shows performance data where a local inference client is fed requests at an infinite rate (no delay between messages),
and shows the throughput scenario under maximum load. The reported metric is `Total Output Throughput (tokens/sec)`.

The performance numbers below were collected using the steps described in this document.

Testing was performed on models with weights quantized using [ModelOpt](https://nvidia.github.io/TensorRT-Model-Optimizer/#) and published by NVIDIA on the [Model Optimizer HuggingFace Collection](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4).

### FP4 Models:
```
nvidia/Llama-3.3-70B-Instruct-FP4
nvidia/Llama-3.1-405B-Instruct-FP4
```

#### Llama 3.3 70B FP4
|                         | GPU     | B200      |           |           |           |
|:-----------------------------|:---|:----------|:----------|:----------|:----------|
|         | TP Size    | 1         | 2         | 4         | 8         |
| ISL, OSL|    |           |           |           |           |
|                              |    |           |           |           |           |
| 128, 128                     |    | 11,253.28 | 17,867.66 | 24,944.50 | 27,471.49 |
| 128, 2048                    |    | 9,925.00  | 15,459.71 | 23,608.58 | 30,742.86 |
| 128, 4096                    |    | 6,318.92  | 8,711.88  | 17,659.74 | 24,947.05 |
| 500, 2000                    |    | 7,559.88  | 10,602.27 | 20,910.23 | 28,182.34 |
| 1000, 1000                   |    | 6,866.96  | 10,838.01 | 16,567.86 | 19,991.64 |
| 1000, 2000                   |    | 6,736.88  | 9,132.08  | 15,737.02 | 20,518.04 |
| 1024, 2048                   |    | 6,580.56  | 8,767.45  | 15,722.55 | 20,437.96 |
| 2048, 128                    |    | 1,375.49  | 1,610.69  | 2,707.58  | 3,717.82  |
| 2048, 2048                   |    | 4,544.73  | 6,956.14  | 12,292.23 | 15,661.22 |
| 5000, 500                    |    | 1,488.19  | 2,379.73  | 3,588.45  | 4,810.21  |
| 20000, 2000                  |    | 580.96    | 1,043.58  | 1,957.84  | 3,167.30  |

#### Llama 3.1 405B FP4
|                          | GPU    | B200      |
|:-----------------------------|:---|:----------|
|          | TP Size   | 8         |
| ISL, OSL|    |           |
|                              |    |           |
| 128, 128                     |    | 9,184.83  |
| 128, 2048                    |    | 10,387.23 |
| 128, 4096                    |    | 8,741.80  |
| 500, 2000                    |    | 9,242.34  |
| 1000, 1000                   |    | 7,565.50  |
| 1000, 2000                   |    | 7,696.76  |
| 1024, 2048                   |    | 7,568.93  |
| 2048, 128                    |    | 953.57    |
| 2048, 2048                   |    | 6,092.32  |
| 5000, 500                    |    | 1,332.22  |
| 20000, 2000                  |    | 961.58    |

### FP8 Models:
```
nvidia/Llama-3.1-8B-Instruct-FP8
nvidia/Llama-3.1-70B-Instruct-FP8
nvidia/Llama-3.1-405B-Instruct-FP8
```

#### Llama 3.1 8B FP8
|                          | GPU    | H200 141GB HBM3   | H100 80GB HBM3   |
|:-----------------------------|:---|:------------------|:-----------------|
|          | TP Size   | 1                 | 1                |
| ISL, OSL |    |                   |                  |
|                              |    |                   |                  |
| 128, 128                     |    | 28,447.38         | 27,568.68        |
| 128, 2048                    |    | 23,294.74         | 22,003.62        |
| 128, 4096                    |    | 17,481.48         | 13,640.35        |
| 500, 2000                    |    | 21,462.57         | 17,794.39        |
| 1000, 1000                   |    | 17,590.60         | 15,270.02        |
| 1000, 2000                   |    | 17,139.51         | 13,850.22        |
| 1024, 2048                   |    | 16,970.63         | 13,374.15        |
| 2048, 128                    |    | 3,531.33          | 3,495.05         |
| 2048, 2048                   |    | 12,022.38         | 9,653.67         |
| 5000, 500                    |    | 3,851.65          | 3,371.16         |
| 20000, 2000                  |    | 1,706.06          | 1,340.92         |

#### Llama 3.1 70B FP8
|                          | GPU   | H200 141GB HBM3   |          |           |           | H100 80GB HBM3   |          |           |           |
|:-----------------------------|:---|:------------------|:---------|:----------|:----------|:-----------------|:---------|:----------|:----------|
|    | TP Size   | 1                 | 2        | 4         | 8         | 1                | 2        | 4         | 8         |
| ISL, OSL|    |                   |          |           |           |                  |          |           |           |
|                              |    |                   |          |           |           |                  |          |           |           |
| 128, 128                     |    | 3,657.58          | 6,477.50 | 10,466.04 | 15,554.57 | 3,191.27         | 6,183.41 | 10,260.68 | 14,686.01 |
| 128, 2048                    |    | 4,351.07          | 8,450.31 | 13,438.71 | 20,750.58 | 745.19           | 5,822.02 | 11,442.01 | 17,463.99 |
| 128, 4096                    |    | 2,696.61          | 5,598.92 | 11,524.93 | 16,634.90 |                  | 3,714.87 | 8,209.91  | 12,598.55 |
| 500, 2000                    |    | 3,475.58          | 6,712.35 | 12,332.32 | 17,311.28 |                  | 4,704.31 | 10,278.02 | 14,630.41 |
| 1000, 1000                   |    | 2,727.42          | 5,097.36 | 8,698.15  | 12,794.92 | 734.67           | 4,191.26 | 7,427.35  | 11,082.48 |
| 1000, 2000                   |    | 2,913.54          | 5,841.15 | 9,016.49  | 13,174.68 | 526.31           | 3,920.44 | 7,590.35  | 11,108.11 |
| 1024, 2048                   |    | 2,893.02          | 5,565.28 | 9,017.72  | 13,117.34 | 525.43           | 3,896.14 | 7,557.32  | 11,028.32 |
| 2048, 128                    |    | 433.30            | 772.97   | 1,278.26  | 1,947.33  | 315.90           | 747.51   | 1,240.12  | 1,840.12  |
| 2048, 2048                   |    | 1,990.25          | 3,822.83 | 7,068.68  | 10,529.06 | 357.98           | 2,732.86 | 5,640.31  | 8,772.88  |
| 5000, 500                    |    | 543.88            | 1,005.81 | 1,714.77  | 2,683.22  | 203.27           | 866.77   | 1,571.92  | 2,399.78  |
| 20000, 2000                  |    | 276.99            | 618.01   | 1,175.35  | 2,021.08  |                  | 408.43   | 910.77    | 1,568.84  |

#### Llama 3.1 405B FP8
|                          | GPU   | H200 141GB HBM3   | H100 80GB HBM3   |
|:-----------------------------|:---|:------------------|:-----------------|
|          | TP Size   | 8                 | 8                |
| ISL, OSL |    |                   |                  |
|                              |    |                   |                  |
| 128, 128                     |    | 3,800.11          | 3,732.40         |
| 128, 2048                    |    | 5,661.13          | 4,572.23         |
| 128, 4096                    |    | 5,167.18          | 2,911.42         |
| 500, 2000                    |    | 4,854.29          | 3,661.85         |
| 1000, 1000                   |    | 3,332.15          | 2,963.36         |
| 1000, 2000                   |    | 3,682.15          | 3,253.17         |
| 1024, 2048                   |    | 3,685.56          | 3,089.16         |
| 2048, 128                    |    | 453.42            | 448.89           |
| 2048, 2048                   |    | 3,055.73          | 2,139.94         |
| 5000, 500                    |    | 656.11            | 579.14           |
| 20000, 2000                  |    | 514.02            | 370.26           |

## Reproducing Benchmarked Results

> [!NOTE] The only models supported in this workflow are those listed in the table above.

The following tables are references for commands that are used as part of the benchmarking process. For a more detailed
description of this benchmarking workflow, see the [benchmarking suite documentation](https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html).

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
| `$model_name` | HuggingFace model name eg. meta-llama/Llama-2-7b-hf or use the path to a local weights directory |
| `$dataset_file` | Location of the dataset file generated by `prepare_dataset.py` |
| `$num_requests` | The number of requests to generate for dataset generation |
| `$seq_len` | A sequence length of ISL + OSL |
| `$llm_options` | (optional) A yaml file containing additional options for the LLM API |

### Preparing a Dataset

In order to prepare a dataset, you can use the provided [script](../../../benchmarks/cpp/prepare_dataset.py).
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
a model name (HuggingFace reference or path to a local model), a [generated dataset](#preparing-a-dataset), and a file containing any desired extra options to the LLMApi (details in [tensorrt_llm/llmapi/llm_args.py:LlmArgs](../../../tensorrt_llm/llmapi/llm_args.py)).

```shell
trtllm-bench --model $model_name throughput --dataset $dataset_file --backend pytorch --extra_llm_api_options $llm_options
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

In majority of cases, we also use a higher KV cache percentage by setting `--kv_cache_free_gpu_mem_fraction 0.95` in the benchmark command. This allows us to obtain better performance than the default setting of `0.90`. We fall back to `0.90` if we hit an out of memory issue.

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
