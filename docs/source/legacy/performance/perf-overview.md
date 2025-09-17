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

### Hardware
The following GPU variants were used for testing:
- H100 SXM 80GB (DGX H100)
- H200 SXM 141GB (DGX H200)
- GH200 96GB HBM3 (480GB LPDDR5X)
- B200 180GB (DGX B200)
- GB200 192GB (GB200 NVL72)

Other hardware variants may have different TDP, memory bandwidth, core count, or other features leading to performance differences on these workloads.

### FP4 Models

```text
nvidia/Llama-3.3-70B-Instruct-FP4
nvidia/Llama-3.1-405B-Instruct-FP4
```

#### Llama 3.3 70B FP4

|                          | GPU:   | B200     | GB200  |
|:-----------------------------|:---|:----------|:--------------|
|    | TP Size   | 1      | 1          |
| ISL, OSL |    |           |               |
|                              |    |           |               |
| 128, 128                     |    | 10,613.84 | 11,100.97     |
| 128, 2048                    |    | 9,445.51  | 10,276.05     |
| 128, 4096                    |    | 6,276.85  | 7,351.12      |
| 500, 2000                    |    | 6,983.27  | 8,194.30      |
| 1000, 1000                   |    | 6,434.29  | 7,401.80      |
| 1000, 2000                   |    | 6,725.03  | 6,478.72      |
| 1024, 2048                   |    | 6,546.61  | 7,922.88      |
| 2048, 128                    |    | 1,330.35  | 1,418.47      |
| 2048, 2048                   |    | 4,528.48  | 5,326.77      |
| 5000, 500                    |    | 1,427.44  | 1,502.44      |
| 20000, 2000                  |    | 636.36    | 732.43        |

#### Llama 3.1 405B FP4

|                         | GPU:    | B200    | GB200  |
|:-----------------------------|:---|:---------|:--------------|
|   | TP Size   | 4     | 4          |
| ISL, OSL |    |          |               |
|                              |    |          |               |
| 128, 128                     |    | 6,218.89 | 6,598.97      |
| 128, 2048                    |    | 7,178.10 | 7,497.40      |
| 128, 4096                    |    | 5,890.89 | 5,898.19      |
| 500, 2000                    |    | 5,844.37 | 6,198.33      |
| 1000, 1000                   |    | 4,958.53 | 5,243.35      |
| 1000, 2000                   |    | 4,874.16 | 4,905.51      |
| 1024, 2048                   |    | 4,833.19 | 4,686.38      |
| 2048, 128                    |    | 737.95   | 761.58        |
| 2048, 2048                   |    | 4,024.02 | 4,326.56      |
| 5000, 500                    |    | 1,032.40 | 1,078.87      |
| 20000, 2000                  |    | 667.39   | 649.95        |

### FP8 Models

```text
nvidia/Llama-3.1-8B-Instruct-FP8
nvidia/Llama-3.3-70B-Instruct-FP8
nvidia/Llama-3.1-405B-Instruct-FP8
nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8
```

#### Llama 3.1 8B FP8

|                          | GPU:   | GH200  | H100   | H200   |
|:-----------------------------|:---|:--------------|:-----------------|:------------------|
|    | TP Size   | 1          | 1             | 1              |
| ISL, OSL |    |               |                  |                   |
|                              |    |               |                  |                   |
| 128, 128                     |    | 27,304.25     | 26,401.48        | 27,027.80         |
| 128, 2048                    |    | 24,045.60     | 21,413.21        | 23,102.25         |
| 128, 4096                    |    | 15,409.85     | 13,541.54        | 17,396.83         |
| 500, 2000                    |    | 20,123.88     | 17,571.01        | 19,759.16         |
| 1000, 1000                   |    | 16,352.99     | 14,991.62        | 17,162.49         |
| 1000, 2000                   |    | 15,705.82     | 13,505.23        | 16,227.11         |
| 1024, 2048                   |    | 16,102.52     | 13,165.91        | 16,057.66         |
| 2048, 128                    |    | 3,573.85      | 3,275.55         | 3,390.69          |
| 2048, 2048                   |    | 10,767.05     | 9,462.43         | 11,822.14         |
| 5000, 500                    |    | 3,584.74      | 3,276.47         | 3,758.08          |
| 20000, 2000                  |    | 1,393.31      | 1,340.69         | 1,705.68          |

#### Llama 3.3 70B FP8

|                        | GPU:     | H100   | H200   |
|:-----------------------------|:---|:-----------------|:------------------|
|    | TP Size   | 2             | 2              |
| ISL, OSL |    |                  |                   |
|                              |    |                  |                   |
| 128, 128                     |    | 6,092.28         | 6,327.98          |
| 128, 2048                    |    | 5,892.94         | 7,467.36          |
| 128, 4096                    |    | 3,828.46         | 5,526.42          |
| 500, 2000                    |    | 4,654.74         | 6,639.15          |
| 1000, 1000                   |    | 4,181.06         | 4,773.33          |
| 1000, 2000                   |    | 3,708.93         | 5,790.36          |
| 1024, 2048                   |    | 3,785.04         | 5,480.44          |
| 2048, 128                    |    | 723.40           | 747.55            |
| 2048, 2048                   |    | 2,785.53         | 3,775.80          |
| 5000, 500                    |    | 865.55           | 978.28            |
| 20000, 2000                  |    | 411.85           | 609.42            |

#### Llama 3.1 405B FP8
|                         | GPU:    | H100   | H200   |
|:-----------------------------|:---|:-----------------|:------------------|
|    | TP Size   | 8             | 8              |
| Runtime Input/Output Lengths |    |                  |                   |
|                              |    |                  |                   |
| 128, 128                     |    |                  | 3,705.18          |
| 128, 2048                    |    | 4,517.39         | 4,715.13          |
| 128, 4096                    |    | 2,910.31         | 4,475.91          |
| 500, 2000                    |    | 3,664.62         | 4,804.10          |
| 1000, 1000                   |    | 2,955.50         | 3,208.25          |
| 1000, 2000                   |    | 2,884.69         | 3,630.29          |
| 1024, 2048                   |    | 3,237.41         | 3,609.50          |
| 2048, 128                    |    | 433.47           | 441.35            |
| 2048, 2048                   |    | 2,216.55         | 2,840.86          |
| 5000, 500                    |    | 579.05           | 645.26            |
| 20000, 2000                  |    | 363.27           | 509.87            |

#### Llama 4 Maverick FP8

Note: Performance for Llama 4 on sequence lengths less than 8,192 tokens is affected by an issue introduced in v0.21. To reproduce the Llama 4 performance noted here, please use v0.20

|                          | GPU    | H200   | H100   |
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

The data collected for the v0.21 benchmarks was run with the following file:

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
