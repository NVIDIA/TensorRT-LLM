(perf-overview)=

> [!IMPORTANT]
> As of TensorRT-LLM v0.10, these performance benchmarks have changed methodology to utilize in-flight batching and
no longer utilize static benchmarking. These numbers are initial measurements and are expected to improve in future
releases.

# Overview

This document summarizes performance measurements of TensorRT-LLM on H100
(Hopper), L40S (Ada) and A100 (Ampere) GPUs for a few key models.

The data in the following tables is provided as a reference point to help users
validate observed performance. It should not be considered as the peak
performance that can be delivered by TensorRT-LLM.

## Known Issues

The following issues are being addressed to improve the efficiency of TensorRT-LLM.

### Unexpected extra GPU memory allocation when enabling `--multiple_profiles`

We observed that enabling multiple profiles can lead to extra
unexpected GPU memory usage on some cases starting from v0.11.
The issue will be addressed in future releases.

### Fused Matmul + Gated-SiLU (LLaMA)

The current implementation combines two Matmul operations into one Matmul followed by
a separate SwiGLU kernel (when `--use_fused_mlp=enable` is enabled). There is also a more
efficient implementation that runs single Matmul + SwiGLU fused kernel for FP8 on Hopper
(when `--use_fused_mlp=enable --gemm_swiglu_plugin fp8` is enabled). The gemm_swiglu_plugin
will support more data types and GPU architectures in the future release.

## Throughput Measurements

The below table shows performance data where a local inference client is fed requests at an infinite rate (no delay between messages),
and shows the throughput client-server scenario under maximum load.


The performance numbers below were collected using the steps described in this document.

**All data in the table below was generated using version 0.12.0 and presents token throughput in tokens/second.**

|              |                          |               |                 |             |                |                |                |         |
| ------------ | ------------------------ | ------------- | --------------- | ----------- | -------------- | -------------- | -------------- | ------- |
|              |                          | **GPU**       | H200 141GB HBM3 | GH200 120GB | H100 80GB HBM3 | H100 80GB HBM3 | A100-SXM4-80GB | L40S    |
|              |                          | **Precision** | FP8             | FP8         | FP8            | Mixed          | Mixed          | FP8     |
| **Model**    | **Input/Output Lengths** | **TP**        |                 |             |                |                |                |         |
| GPTJ 6B      | 128/128                  | 1             | 24834.76        | 22454.79    | 24429.55       | 13085.91       | 5864.81        | 7647.24 |
|              | 128/2048                 | 1             | 8348.93         | 6656.25     | 7831.38        | 3882.21        | 2194.57        | 1843.91 |
|              | 128/4096                 | 1             | 5062.80         | 3678.91     | 3968.98        | 2046.53        | 1118.22        | 980.67  |
|              | 2048/128                 | 1             | 2776.53         | 2491.03     | 2724.38        | 1488.56        | 657.01         | 741.06  |
|              | 2048/2048                | 1             | 3631.54         | 2994.81     | 3004.17        | 1280.54        | 854.37         | 754.16  |
| LLaMA v2 7B  | 128/128                  | 1             | 19706.35        | 17803.58    | 19068.99       | 11393.48       | 5272.39        | 6345.72 |
|              | 128/2048                 | 1             | 7651.12         | 5472.34     | 6610.03        | 2964.65        | 1785.79        | 1551.37 |
|              | 128/4096                 | 1             | 4424.90         | 3271.61     | 3649.38        | 1596.87        | 957.12         | 817.24  |
|              | 2048/128                 | 1             | 2385.54         | 2035.42     | 2271.63        | 1189.06        | 564.77         | 625.09  |
|              | 2048/2048                | 1             | 3191.34         | 2726.29     | 2802.41        | 1243.96        | 735.19         | 641.56  |
| LLaMA v3 8B  | 128/128                  | 1             | 28288.75        | 25420.52    | 27399.75       | 15567.44       | 6586.88        | 8745.80 |
|              | 128/2048                 | 1             | 23230.62        | 16426.68    | 19198.73       | 8817.39        | 4882.13        | 5084.49 |
|              | 128/4096                 | 1             | 16144.44        | 9832.66     | 12084.97       | 5352.37        | 3079.90        | 2755.13 |
|              | 2048/128                 | 1             | 3623.79         | 3290.22     | 3463.26        | 1852.48        | 781.63         | 980.86  |
|              | 2048/2048                | 1             | 11093.62        | 7573.35     | 8894.11        | 3986.83        | 2268.13        | 2051.79 |
| Mistral 7B   | 128/128                  | 1             | 30223.01        | 27696.90    | 29788.46       | 16319.25       | 6807.02        | 9612.58 |
|              | 128/2048                 | 1             | 24989.54        | 17942.29    | 20509.72       | 9982.01        | 5296.02        | 5444.89 |
|              | 128/4096                 | 1             | 17036.14        | 10846.03    | 12807.80       | 5718.89        | 3241.33        | 2931.17 |
|              | 2048/128                 | 1             | 3678.80         | 3294.02     | 3521.71        | 1887.75        | 786.43         | 1002.49 |
|              | 2048/2048                | 1             | 11510.54        | 8357.75     | 9214.61        | 4284.82        | 2363.25        | 2154.26 |
| Mixtral 8x7B | 128/128                  | 2             | 24895.03        | 8785.80     | 24394.71       | 15529.86       | 5921.41        |         |
|              |                          | 4             | 42014.24        | 38828.53    | 40197.42       | 28132.17       | 11414.95       | 6820.26 |
|              | 128/2048                 | 2             | 29389.21        | 5474.69     | 20873.02       | 7066.02        | 4306.98        |         |
|              |                          | 4             | 52348.10        | 41573.66    | 40588.05       | 21285.72       | 10974.83       | 7467.15 |
|              | 128/4096                 | 2             | 21480.27        | 2277.66     | 12838.28       | 3986.01        | 2400.11        |         |
|              |                          | 4             | 39182.04        | 28626.55    | 28337.31       | 12447.13       | 7278.89        | 5233.43 |
|              | 2048/128                 | 2             | 2934.44         | 1003.51     | 2898.27        | 1834.77        | 693.51         |         |
|              |                          | 4             | 5152.40         | 4724.01     | 5028.61        | 3393.18        | 1362.93        | 805.49  |
|              | 2048/2048                | 2             | 14029.17        | 2671.88     | 10479.45       | 3531.31        | 1945.88        |         |
|              |                          | 4             | 25436.05        | 20302.56    | 19971.72       | 9622.66        | 5221.74        | 3616.30 |
| LLaMA v3 70B | 128/128                  | 2             | 5386.88         |             |                | 2959.22        | 1301.14        |         |
|              |                          | 4             | 8944.26         | 8587.01     | 8642.05        | 5966.47        | 2413.95        |         |
|              |                          | 8             | 16125.20        |             | 15397.47       | 10406.55       | 4548.32        | 1364.08 |
|              | 128/2048                 | 2             | 7007.27         |             |                | 720.73         | 500.83         |         |
|              |                          | 4             | 12906.75        | 10761.53    | 8978.95        | 4736.61        | 2380.02        |         |
|              |                          | 8             | 19417.37        |             | 14822.93       | 6672.14        | 3815.08        | 1809.40 |
|              | 128/4096                 | 2             | 6183.85         |             |                | 369.29         | 251.24         |         |
|              |                          | 4             | 8859.54         | 7270.77     | 6073.48        | 2969.99        | 1634.82        |         |
|              |                          | 8             | 13969.95        |             | 10094.57       | 4358.77        | 2847.54        | 1313.78 |
|              | 2048/128                 | 2             | 696.59          |             |                | 301.46         | 140.88         |         |
|              |                          | 4             | 1044.35         | 1000.55     | 1022.06        | 681.72         | 278.76         |         |
|              |                          | 8             | 2018.47         |             | 1933.15        | 1279.46        | 543.73         | 163.36  |
|              | 2048/2048                | 2             | 3525.18         |             |                |                | 87.54          |         |
|              |                          | 4             | 6550.76         | 4859.38     | 4870.26        | 2379.66        | 1209.69        |         |
|              |                          | 8             | 9706.95         |             | 7670.04        | 3692.41        | 2192.28        | 895.23  |
| LLaMA v2 70B | 128/128                  | 2             | 6355.16         |             |                | 2927.71        | 1374.05        |         |
|              |                          | 4             | 10818.97        | 10819.19    | 10754.99       | 6603.10        | 2765.94        |         |
|              |                          | 8             | 16667.25        |             | 16074.84       | 11369.11       | 4796.89        | 1402.92 |
|              | 128/2048                 | 2             | 6185.77         |             |                | 668.52         | 445.04         |         |
|              |                          | 4             | 12884.76        | 11356.48    | 8870.71        | 5067.06        | 2710.53        |         |
|              |                          | 8             | 19053.13        |             | 17534.62       | 8805.16        | 5665.93        | 2203.33 |
|              | 128/4096                 | 2             | 4873.24         |             |                | 334.10         | 215.70         |         |
|              |                          | 4             | 8664.90         | 6311.85     | 7564.99        | 3354.02        | 1884.46        |         |
|              |                          | 8             | 15110.32        |             | 10584.03       | 5373.10        | 3672.80        | 1787.76 |
|              | 2048/128                 | 2             | 732.09          |             |                | 302.49         | 141.70         |         |
|              |                          | 4             | 1272.90         | 1269.58     | 1265.80        | 774.93         | 320.79         |         |
|              |                          | 8             | 2015.77         |             | 1943.96        | 1355.78        | 569.48         | 165.52  |
|              | 2048/2048                | 2             | 3508.50         |             |                | 321.95         | 212.97         |         |
|              |                          | 4             | 6642.69         | 5545.83     | 4889.26        | 2439.10        | 1276.58        |         |
|              |                          | 8             | 10178.71        |             | 8071.77        | 4275.74        | 2589.60        | 1083.45 |
| Falcon 180B  | 128/128                  | 4             | 5129.55         |             |                |                |                |         |
|              |                          | 8             | 8370.98         |             | 8268.72        |                |                |         |
|              | 128/2048                 | 4             | 7823.79         |             |                |                |                |         |
|              |                          | 8             | 13278.59        |             | 13107.48       |                |                |         |
|              | 128/4096                 | 4             | 6374.10         |             |                |                |                |         |
|              |                          | 8             | 12660.89        |             | 10493.79       |                |                |         |
|              | 2048/128                 | 4             | 601.67          |             |                |                |                |         |
|              |                          | 8             | 1002.57         |             | 991.22         |                |                |         |
|              | 2048/2048                | 4             | 3869.76         |             |                |                |                |         |
|              |                          | 8             | 7134.33         |             | 6386.83        |                |                |         |
*TP stands for Tensor Parallelism*

## Reproducing Benchmarked Results

> [!NOTE] The only models supported in this workflow are those listed in the table above.

The following tables are references for commands that are used as part of the benchmarking process. For a more detailed
description of this benchmarking workflow, see the [Benchmarking Suite README](../../../benchmarks/Suite.md).

### Commands

| Stage | Description | Command |
| :- | - | - |
| [Dataset](#preparing-a-dataset) | Create a synthetic dataset | `python benchmarks/cpp/prepare_dataset.py --tokenizer=$model_name --stdout token-norm-dist --num-requests=$num_requests --input-mean=$isl --output-mean=$osl --input-stdev=0 --output-stdev=0 > $dataset_file` |
| [Build](#engine-building) | Build a TensorRT-LLM engine | `trtllm-bench --model $model_name build --tp_size $tp_size --quantization FP8 --dataset $dataset_file` |
| [Run](#running-the-benchmark) | Run a benchmark with a dataset | `trtllm-bench --model $model_name throughput --dataset $dataset_file --engine_dir $engine_dir` |

### Variables

| Name | Description |
| :- | - |
| `$isl` | Benchmark input sequence length. |
| `$osl` | Benchmark output sequence length. |
| `$tp_size` | Number of GPUs to run the benchmark with |
| `$engine_dir` | Location to store built engine file (can be deleted after running benchmarks). |
| `$model_name` | HuggingFace model name eg. meta-llama/Llama-2-7b-hf or use the path to a local weights directory |
| `$dataset_file` | Location of the dataset file generated by `prepare_dataset.py` |
| `$num_requests` | The number of requests to generate for dataset generation |
| `$seq_len` | A sequence length of ISL + OSL |

## Preparing a Dataset

In order to prepare a dataset, you can use the provided [script](../../../benchmarks/cpp/prepare_dataset.py).
To generate a synthetic dataset, run the following command:

```shell
python benchmarks/cpp/prepare_dataset.py --output=$dataset_file --tokenizer=$model_name token-norm-dist --num-requests=$num_requests --input-mean=$isl --output-mean=$osl --input-stdev=0 --output-stdev=0 > $dataset_file
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
| 2048         | 128           | 2176       | 3000               |
| 2048         | 2048          | 4096       | 1500               |


## Engine Building

All engines are built using the `trtllm-bench build` sub-command. The basic command for FP8 quantized engines is as follows:

```
trtllm-bench --model $model_name build --tp_size $tp_size --quantization FP8 --dataset $dataset_file
```

or if you would like to build for a specific sequence length:

```
trtllm-bench --model $model_name build --tp_size $tp_size --quantization FP8 --max_seq_length $seq_len
```

If you would like to build an FP16 engine without any quantization, simply remove the `--quantization FP8` option.

> [!NOTE] If you specify FP8 quantization, the KV cache will automatically be set to FP8 as well!

The `trtllm-bench build` sub-command will output the path where the engine is located upon a successful build. For example,

```shell
===========================================================
ENGINE SAVED: /tmp/meta-llama/Llama-2-7b-hf/tp_1_pp_1
===========================================================
```

## Running the Benchmark

To run the benchmark with the generated data set, simply use the `trtllm-bench throughput` sub-command. The benchmarker will
run an offline maximum throughput scenario such that all requests are queued in rapid succession. You simply need to provide
the patch to the engine from the [build](#engine-building) phase and a [generated dataset](#preparing-a-dataset).

```shell
trtllm-bench --model $model_name throughput --dataset $dataset_file --engine_dir $engine_dir
```

The results will be printed to the terminal upon benchmark completion. For example,

```shell
===========================================================
= ENGINE DETAILS
===========================================================
Model:                  meta-llama/Llama-2-7b-hf
Engine Directory:       /tmp/meta-llama/Llama-2-7b-hf/tp_1_pp_1
TensorRT-LLM Version:   0.12.0
Dtype:                  float16
KV Cache Dtype:         FP8
Quantization:           FP8
Max Input Length:       2048
Max Sequence Length:    4098

===========================================================
= WORLD + RUNTIME INFORMATION
===========================================================
TP Size:                1
PP Size:                1
Max Runtime Batch Size: 4096
Max Runtime Tokens:     8192
Scheduling Policy:      Guaranteed No Evict
KV Memory Percentage:   99.0%
Issue Rate (req/sec):   3.680275266452667e+18
===========================================================
= STATISTICS
===========================================================
Number of requests:             3000
Average Input Length (tokens):  128.0
Average Output Length (tokens): 128.0
Token Throughput (tokens/sec):  23405.927228471104
Request Throughput (req/sec):   182.8588064724305
Total Latency (seconds):        16.406100739
===========================================================
```

> [!WARNING] In some cases, the benchmarker may not print anything at all. This behavior usually
means that the benchmark has hit an out of memory issue. Try reducing the KV cache percentage
using the `--kv_cache_free_gpu_mem_fraction` option to lower the percentage of used memory.
