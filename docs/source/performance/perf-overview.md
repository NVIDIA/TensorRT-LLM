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
|                         | GPU:    | B200      |           |           |           |
|:-----------------------------|:---|:----------|:----------|:----------|:----------|
|    | TP Size   | 1         | 2         | 4         | 8         |
| ISL, OSL |    |           |           |           |           |
|                              |    |           |           |           |           |
| 128, 128                     |    | 11,859.97 | 17,755.36 | 24,415.80 | 27,306.19 |
| 128, 2048                    |    | 8,992.04  | 14,124.04 | 25,348.30 | 36,868.70 |
| 128, 4096                    |    | 5,080.64  | 9,255.85  | 16,714.98 | 27,576.13 |
| 500, 2000                    |    | 7,086.62  | 11,643.52 | 18,353.38 | 23,769.29 |
| 1000, 1000                   |    | 6,417.87  | 10,376.28 | 16,795.59 | 21,271.51 |
| 1000, 2000                   |    | 5,773.36  | 9,594.89  | 13,888.65 | 21,741.45 |
| 1024, 2048                   |    | 5,747.71  | 9,381.21  | 13,689.68 | 21,770.04 |
| 2048, 128                    |    | 1,343.79  | 1,992.55  | 3,014.17  | 3,992.34  |
| 2048, 2048                   |    |           |           | 5,702.33  | 15,339.80 |
| 5000, 500                    |    | 1,441.81  | 2,400.09  | 3,550.76  | 5,055.64  |
| 20000, 2000                  |    | 610.64    | 1,183.48  | 2,029.52  | 3,396.72  |

#### Llama 3.1 405B FP4

| GPU:                         |    | B200     |           |
|:-----------------------------|:---|:---------|:----------|
| TP Size   |    | 4        | 8         |
| ISL, OSL |    |          |           |
|                              |    |          |           |
| 128, 128                     |    | 5,925.36 | 8,977.74  |
| 128, 2048                    |    | 8,140.69 | 11,927.37 |
| 128, 4096                    |    | 5,888.69 | 9,470.74  |
| 500, 2000                    |    | 6,328.81 | 10,156.47 |
| 1000, 1000                   |    | 4,909.09 | 7,593.56  |
| 1000, 2000                   |    | 5,191.04 | 7,011.15  |
| 1024, 2048                   |    | 5,111.88 | 7,030.86  |
| 2048, 128                    |    | 701.27   | 1,089.03  |
| 2048, 2048                   |    |          | 5,176.81  |
| 5000, 500                    |    | 813.32   | 1,511.81  |
| 20000, 2000                  |    | 569.28   | 1,059.59  |

### FP8 Models:
```
nvidia/Llama-3.1-8B-Instruct-FP8
nvidia/Llama-3.1-70B-Instruct-FP8
nvidia/Llama-3.1-405B-Instruct-FP8
```

#### Llama 3.1 8B FP8
|                         | GPU:    | H200 141GB HBM3   | H100 80GB HBM3   |
|:-----------------------------|:---|:------------------|:-----------------|
|    | TP Size   | 1              | 1             |
| ISL, OSL |    |                   |                  |
|                              |    |                   |                  |
| 128, 128                     |    | 28,549.50         | 28,035.10        |
| 128, 2048                    |    | 26,812.33         | 21,050.90        |
| 128, 4096                    |    | 17,576.14         | 12,764.22        |
| 500, 2000                    |    | 21,243.42         | 16,880.05        |
| 1000, 1000                   |    | 17,619.36         | 15,045.78        |
| 1000, 2000                   |    | 16,961.83         | 13,832.71        |
| 1024, 2048                   |    | 16,793.56         | 13,570.46        |
| 2048, 128                    |    | 3,524.08          | 3,382.15         |
| 5000, 500                    |    | 3,845.42          | 3,353.48         |
| 20000, 2000                  |    | 1,716.87          | 1,320.63         |

#### Llama 3.1 70B FP8
|                         | GPU:    | H200 141GB HBM3   |          |           |           | H100 80GB HBM3   |          |           |           |
|:-----------------------------|:---|:------------------|:---------|:----------|:----------|:-----------------|:---------|:----------|:----------|
|   | TP Size    | 1              | 2     | 4      | 8      | 1             | 2     | 4      | 8      |
| ISL, OSL |    |                   |          |           |           |                  |          |           |           |
|                              |    |                   |          |           |           |                  |          |           |           |
| 128, 128                     |    | 3,853.83          | 6,489.29 | 10,567.69 | 15,591.42 | 3,253.48         | 6,376.85 | 10,237.10 | 14,656.69 |
| 128, 2048                    |    | 4,350.88          | 8,447.23 | 14,763.31 | 19,956.95 | 758.55           | 5,602.62 | 10,827.41 | 16,591.87 |
| 128, 4096                    |    | 2,730.79          | 6,414.68 | 10,803.74 | 14,467.15 |                  | 3,364.30 | 8,038.83  | 11,249.15 |
| 500, 2000                    |    | 3,488.64          | 7,114.59 | 12,430.60 | 18,069.33 |                  | 4,768.42 | 10,015.22 | 15,130.12 |
| 1000, 1000                   |    | 2,875.06          | 5,121.08 | 8,601.38  | 12,538.65 | 714.92           | 4,223.74 | 7,395.74  | 10,693.59 |
| 1000, 2000                   |    | 2,863.28          | 5,831.12 | 9,765.75  | 14,378.24 | 510.67           | 3,836.14 | 7,634.91  | 11,576.51 |
| 1024, 2048                   |    | 2,823.44          | 5,299.83 | 9,703.30  | 14,308.41 | 507.86           | 3,809.33 | 7,620.84  | 11,538.86 |
| 2048, 128                    |    | 436.39            | 771.09   | 1,276.25  | 1,943.06  |                  | 734.28   | 1,233.71  | 1,831.99  |
| 2048, 2048                   |    |                   |          |           | 9,970.15  |                  |          |           |           |
| 5000, 500                    |    | 547.20            | 1,001.16 | 1,694.54  | 2,656.12  | 200.16           | 878.63   | 1,551.48  | 2,365.88  |
| 20000, 2000                  |    | 270.45            | 631.72   | 1,163.86  | 1,987.52  |                  | 404.21   | 899.92    | 1,547.42  |

#### Llama 3.1 405B FP8
|                         | GPU:    | H200 141GB HBM3   | H100 80GB HBM3   |
|:-----------------------------|:---|:------------------|:-----------------|
|    | TP Size   | 8              | 8             |
| ISL, OSL |    |                   |                  |
|                              |    |                   |                  |
| 128, 128                     |    | 3,784.96          | 3,711.06         |
| 128, 2048                    |    | 6,139.81          | 4,571.68         |
| 128, 4096                    |    | 5,235.58          | 3,030.35         |
| 500, 2000                    |    | 4,902.22          | 3,839.54         |
| 1000, 1000                   |    | 3,400.47          | 2,949.12         |
| 1000, 2000                   |    | 4,115.62          | 3,192.05         |
| 1024, 2048                   |    | 4,097.04          | 3,195.78         |
| 2048, 128                    |    | 455.08            | 437.67           |
| 5000, 500                    |    | 659.79            | 577.93           |
| 20000, 2000                  |    | 519.68            | 316.81           |

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

*v0.20 PLEASE READ*:
The data collected for the v0.20 benchmarks was run with the following file:

`llm_options.yml`
```yaml

 pytorch_backend_config:
  cuda_graph_padding_enabled: true
```

However, this resulted in the unintended consequence of disabling cuda graphs, due to how the pytorch backend config options are handled. We would recommend users to run with the following file to explicitly turn off cuda graphs:

`llm_options.yml`
```yaml
 pytorch_backend_config:
  use_cuda_graph: false
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
[Latency] P90    : 11080417
[Latency] P95    : 111302.9101
[Latency] P99    : 111618.2158
[Latency] MINIMUM: 24189.0838
[Latency] MAXIMUM: 111668.0964
[Latency] AVERAGE: 75903.4456
```

> [!WARNING] In some cases, the benchmarker may not print anything at all. This behavior usually
means that the benchmark has hit an out of memory issue. Try reducing the KV cache percentage
using the `--kv_cache_free_gpu_mem_fraction` option to lower the percentage of used memory.
