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
|:------------------------|:--------|:----------|:----------|:----------|:----------|
|                         | TP Size | 1         | 2         | 4         | 8         |
| ISL, OSL                |         |           |           |           |           |
|                         |         |           |           |           |           |
| 128, 128                |         | 10,994.48 | 17,542.11 | 24,667.31 | 27,272.27 |
| 128, 2048               |         | 9,580.46  | 15,432.35 | 23,568.12 | 31,174.31 |
| 128, 4096               |         | 6,418.39  | 9,841.53  | 17,808.76 | 25,229.25 |
| 500, 2000               |         | 7,343.32  | 11,850.57 | 20,709.67 | 28,038.78 |
| 1000, 1000              |         | 6,752.53  | 10,815.88 | 16,413.04 | 20,060.66 |
| 1000, 2000              |         | 6,670.07  | 9,830.73  | 15,597.49 | 20,672.37 |
| 1024, 2048              |         | 6,636.75  | 9,807.13  | 15,519.23 | 20,617.28 |
| 2048, 128               |         | 1,342.17  | 1,989.41  | 3,033.14  | 4,035.64  |
| 5000, 500               |         | 1,429.67  | 2,419.67  | 3,686.84  | 5,182.96  |
| 20000, 2000             |         | 629.77    | 1,177.01  | 2,120.66  | 3,429.03  |

#### Llama 3.1 405B FP4

|                         | GPU     | B200     |           |
|:------------------------|:------- |:---------|:----------|
|                         | TP Size | 4        | 8         |
| ISL, OSL                |         |          |           |
|                         |         |          |           |
| 128, 128                |         | 6,163.81 | 9,002.90  |
| 128, 2048               |         | 7,081.21 | 10,288.28 |
| 128, 4096               |         | 6,028.37 | 8,713.77  |
| 500, 2000               |         | 5,858.75 | 9,125.86  |
| 1000, 1000              |         | 4,848.00 | 7,582.97  |
| 1000, 2000              |         | 5,375.25 | 7,626.28  |
| 1024, 2048              |         | 5,345.70 | 7,464.03  |
| 2048, 128               |         | 693.55   | 1,086.56  |
| 5000, 500               |         | 947.49   | 1,532.45  |
| 20000, 2000             |         | 641.11   | 1,097.84  |

### FP8 Models:
```
nvidia/Llama-3.1-8B-Instruct-FP8
nvidia/Llama-3.3-70B-Instruct-FP8
nvidia/Llama-3.1-405B-Instruct-FP8
nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8
```

#### Llama 3.1 8B FP8

|                         | GPU     | H200 141GB HBM3   | H100 80GB HBM3   |
|:-----------------------------|:---|:------------------|:-----------------|
|    | TP Size   | 1              | 1             |
| ISL, OSL |    |                   |                  |
|                              |    |                   |                  |
| 128, 128                     |    | 27,970.14         | 27,688.36        |
| 128, 2048                    |    | 23,326.38         | 21,841.15        |
| 128, 4096                    |    | 17,508.51         | 13,730.89        |
| 500, 2000                    |    | 21,390.41         | 17,833.34        |
| 1000, 1000                   |    | 17,366.89         | 15,270.62        |
| 1000, 2000                   |    | 16,831.31         | 13,798.08        |
| 1024, 2048                   |    | 16,737.03         | 13,385.50        |
| 2048, 128                    |    | 3,488.03          | 3,414.67         |
| 5000, 500                    |    | 3,813.69          | 3,394.54         |
| 20000, 2000                  |    | 1,696.66          | 1,345.42         |

#### Llama 3.3 70B FP8

|                          | GPU    | H200 141GB HBM3   |          |           |           | H100 80GB HBM3   |          |           |           |
|:-----------------------------|:---|:------------------|:---------|:----------|:----------|:-----------------|:---------|:----------|:----------|
|    | TP Size   | 1              | 2     | 4      | 8      | 1            | 2     | 4      | 8      |
| ISL, OSL |    |                   |          |           |           |                  |          |           |           |
|                              |    |                   |          |           |           |                  |          |           |           |
| 128, 128                     |    | 3,605.47          | 6,427.69 | 10,407.42 | 15,434.37 | 3,128.33         | 6,216.91 |           |           |
| 128, 2048                    |    | 4,315.80          | 8,464.03 | 13,508.59 | 20,759.72 | 756.42           | 5,782.57 | 11,464.94 | 17,424.32 |
| 128, 4096                    |    | 2,701.17          | 5,573.55 | 11,458.56 | 16,668.75 |                  | 3,868.37 | 8,206.39  | 12,624.61 |
| 500, 2000                    |    | 3,478.76          | 6,740.06 | 12,200.18 |           |                  | 4,684.06 | 9,903.53  | 14,553.93 |
| 1000, 1000                   |    | 2,744.32          | 5,119.72 | 8,685.44  | 12,744.51 | 742.14           | 4,247.19 | 7,435.65  | 11,018.81 |
| 1000, 2000                   |    | 2,896.44          | 5,847.26 | 9,031.21  | 13,141.17 | 533.74           | 3,866.53 | 7,611.12  | 11,139.22 |
| 1024, 2048                   |    | 2,874.18          | 5,568.61 | 8,946.71  | 13,082.62 | 530.16           | 3,796.68 | 7,575.24  | 11,004.31 |
| 2048, 128                    |    | 435.90            | 772.67   | 1,264.76  |           |                  | 736.89   | 1,213.33  | 1,839.22  |
| 2048, 2048                   |    |                   |          |           | 10,412.85 |                  |          |           |           |
| 5000, 500                    |    | 545.96            | 997.15   | 1,698.22  | 2,655.28  | 204.94           | 862.91   | 1,552.68  | 2,369.84  |
| 20000, 2000                  |    | 276.66            | 620.33   | 1,161.29  | 1,985.85  |                  | 416.13   | 903.66    | 1,554.10  |

#### Llama 3.1 405B FP8

|                          | GPU    | H200 141GB HBM3   | H100 80GB HBM3   |
|:-----------------------------|:---|:------------------|:-----------------|
|   | TP Size   | 8              | 8             |
| ISL, OSL |    |                   |                  |
|                              |    |                   |                  |
| 128, 2048                    |    | 5,567.87          |                  |
| 128, 4096                    |    | 5,136.85          |                  |
| 500, 2000                    |    | 4,787.61          | 3,673.91         |
| 1000, 1000                   |    | 3,286.30          | 3,012.22         |
| 1000, 2000                   |    | 3,636.76          | 3,262.20         |
| 1024, 2048                   |    | 3,618.66          | 3,109.70         |
| 2048, 128                    |    | 443.10            | 449.02           |
| 5000, 500                    |    | 645.46            |                  |
| 20000, 2000                  |    |                   | 372.12           |

#### Llama 4 Maverick FP8

|                          | GPU    | H200 141GB HBM3   | H100 80GB HBM3   |
|:-----------------------------|:---|:------------------|:-----------------|
|    | TP Size    | 8              | 8             |
| ISL, OSL |    |                   |                  |
|                              |    |                   |                  |
| 128, 2048                    |    | 27,543.87         |                  |
| 128, 4096                    |    | 18,541.01         | 11,163.12        |
| 500, 2000                    |    | 21,117.34         |                  |
| 1000, 2000                   |    |                   | 10,556.00        |
| 1024, 2048                   |    | 16,859.45         | 11,584.33        |
| 2048, 128                    |    | 4,364.06          | 3,832.38         |
| 2048, 2048                   |    | 12,800.89         |                  |
| 5000, 500                    |    | 5,128.60          |                  |
| 20000, 2000                  |    | 1,764.27          | 1,400.79         |

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

The data collected for the v0.20 benchmarks was run with the following file:

`llm_options.yml`
```yaml
use_cuda_graph: true
cuda_graph_padding_enabled: true
cuda_graph_batch_sizes:
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

In a majority of cases, we also use a higher KV cache percentage by setting `--kv_cache_free_gpu_mem_fraction 0.95` in the benchmark command. This allows us to obtain better performance than the default setting of `0.90`. We fall back to `0.90` if we hit an out of memory issue.

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
