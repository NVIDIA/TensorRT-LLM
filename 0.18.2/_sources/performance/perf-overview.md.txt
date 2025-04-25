(perf-overview)=

> [!IMPORTANT]
> As of TensorRT-LLM v0.10, these performance benchmarks have changed methodology to utilize in-flight batching and
no longer utilize static benchmarking. These numbers are initial measurements and are expected to improve in future
releases.

# Overview

This document summarizes performance measurements of TensorRT-LLM on a number of GPUs across a set of key models.

The data in the following tables is provided as a reference point to help users
validate observed performance. It should not be considered as the peak
performance that can be delivered by TensorRT-LLM.

## Known Issues

The following issues are being addressed to improve the efficiency of TensorRT-LLM.

### Known AllReduce performance issue on AMD-based CPU platforms

We observed a performance issue on NCCL 2.23.4, which can be workarounded by setting `NCCL_P2P_LEVEL` to `SYS`:
```
export NCCL_P2P_LEVEL=SYS
```
Multi-GPU cases could be affected due to the issue, which is being addressed.

### Fused Matmul + Gated-SiLU (LLaMA)

The current implementation combines two Matmul operations into one Matmul followed by
a separate SwiGLU kernel (when `--use_fused_mlp=enable` is enabled). There is also a more
efficient implementation that runs single Matmul + SwiGLU fused kernel for FP8 on Hopper
(when `--use_fused_mlp=enable --gemm_swiglu_plugin fp8` is enabled). The gemm_swiglu_plugin
will support more data types and GPU architectures in the future release.

### Use *gptManagerBenchmark* for GH200

For release v0.17, on GH200 systems, we recommend using the legacy flow based on *gptManagerBenchmark* to measure performance.

## Throughput Measurements

The below table shows performance data where a local inference client is fed requests at an infinite rate (no delay between messages),
and shows the throughput client-server scenario under maximum load.

The performance numbers below were collected using the steps described in this document.

**All data in the table below was generated using version 0.17 and presents token throughput in tokens/second.**

| Throughput (tokens / sec)   |                            | GPU                          | H200 141GB HBM3   | H100 80GB HBM3   | GH200 480GB   | L40S    | A100-SXM4-80GB   |
|:----------------------------|:---------------------------|:-----------------------------|:------------------|:-----------------|:--------------|:--------|:-----------------|
|                             | Precision                  |                              | FP8               | FP8              | FP8           | FP8     | FP16             |
| Model                       | Tensor Model Parallel Size | Runtime Input/Output Lengths |                   |                  |               |         |                  |
| LLaMA v3.1 8B               | 1                          | 128, 128                     | 29526.04          | 28836.77         | 29852.96      | 9104.61 | 6627.27          |
|                             |                            | 128, 2048                    | 25398.86          | 21109.38         | 21769.55      | 5365.81 | 5255.99          |
|                             |                            | 128, 4096                    | 17370.8           | 13593.65         | 14189.89      | 3025.92 | 3453.79          |
|                             |                            | 500, 2000                    | 21020.81          | 16500.69         | 17137.29      | 4273.75 | 4276.58          |
|                             |                            | 1000, 1000                   | 17537.96          | 15244.78         | 16482.77      | 4054.71 | 3786.83          |
|                             |                            | 2048, 128                    | 3794.14           | 3556.73          | 3843.95       | 1066.52 | 799.61           |
|                             |                            | 2048, 2048                   | 11968.5           | 9488.42          | 10265.9       | 2225.27 | 2424.16          |
|                             |                            | 5000, 500                    | 3987.79           | 3559.36          | 3932.58       | 981.2   | 825.13           |
|                             |                            | 20000, 2000                  | 1804.1            | 1401.31          | 1560.2        | 327.97  | 330.04           |
| LLaMA v3.1 70B              | 1                          | 128, 128                     | 4020.75           | 3378.03          | 3636.91       |         |                  |
|                             |                            | 128, 2048                    | 4165.68           | 911.62           | 2082.74       |         |                  |
|                             |                            | 128, 4096                    | 2651.75           | 426.32           | 1263.98       |         |                  |
|                             |                            | 500, 2000                    | 3018.39           | 775.57           | 1973.86       |         |                  |
|                             |                            | 1000, 1000                   | 2823.45           | 839.97           | 1746.12       |         |                  |
|                             |                            | 2048, 128                    | 465.99            | 343.29           | 424.96        |         |                  |
|                             |                            | 2048, 2048                   | 1913.8            |                  | 1086.93       |         |                  |
|                             |                            | 5000, 500                    | 560.16            | 245.34           | 422.36        |         |                  |
|                             |                            | 20000, 2000                  | 279.52            |                  |               |         |                  |
|                             | 2                          | 128, 128                     | 6823.01           | 6645.12          |               |         | 1313.96          |
|                             |                            | 128, 2048                    | 8290.35           | 6169.58          |               |         | 531.26           |
|                             |                            | 128, 4096                    | 6526.67           | 3897.06          |               |         |                  |
|                             |                            | 500, 2000                    | 6848.02           | 4972.57          |               |         | 439.41           |
|                             |                            | 1000, 1000                   | 5164.76           | 4390.53          |               |         | 472.94           |
|                             |                            | 2048, 128                    | 809               | 772.66           |               |         | 148.96           |
|                             |                            | 2048, 2048                   | 4183.88           | 2898.16          |               |         | 261.1            |
|                             |                            | 5000, 500                    | 1025.38           | 919.73           |               |         | 121.47           |
|                             |                            | 20000, 2000                  | 640.62            | 443.01           |               |         |                  |
|                             | 4                          | 128, 128                     | 11098.63          | 11127.53         |               | 1523.52 | 2733.48          |
|                             |                            | 128, 2048                    | 14156             | 11511.93         |               | 1942.66 | 2811.27          |
|                             |                            | 128, 4096                    | 10574.06          | 7439.41          |               | 1440.23 | 1976.49          |
|                             |                            | 500, 2000                    | 12452.79          | 9836.7           |               | 1634.72 | 2275.79          |
|                             |                            | 1000, 1000                   | 8911.29           | 7430.99          |               | 1209.25 | 1921.77          |
|                             |                            | 2048, 128                    | 1358.06           | 1302.6           |               | 177.72  | 325.15           |
|                             |                            | 2048, 2048                   | 7130.44           | 5480.03          |               | 969.68  | 1393.64          |
|                             |                            | 5000, 500                    | 1811.55           | 1602.78          |               | 249.52  | 392.62           |
|                             |                            | 20000, 2000                  | 1199.68           | 920.19           |               | 162.25  | 212.08           |
|                             | 8                          | 128, 128                     | 15355.84          | 14730.69         |               | 1464.03 | 4717.62          |
|                             |                            | 128, 2048                    | 21195.88          | 17061.82         |               | 2303.31 | 5241.5           |
|                             |                            | 128, 4096                    | 16941.52          | 14171.43         |               | 2018.22 | 3724.67          |
|                             |                            | 500, 2000                    | 17278.4           | 14679.33         |               | 1971.96 | 4445.37          |
|                             |                            | 1000, 1000                   | 13181.24          | 11451.16         |               | 1333.62 | 3320.41          |
|                             |                            | 2048, 128                    | 1983.03           | 1923.41          |               | 176.16  | 542.38           |
|                             |                            | 2048, 2048                   | 11142.47          | 8801.95          |               | 1200.16 | 2553.71          |
|                             |                            | 5000, 500                    | 2717.83           | 2457.42          |               | 259.71  | 696.34           |
|                             |                            | 20000, 2000                  | 1920.45           | 1512.6           |               | 209.87  | 413.38           |
| LLaMA v3.1 405B             | 8                          | 128, 128                     | 3874.19           |                  |               |         |                  |
|                             |                            | 128, 2048                    | 5938.09           |                  |               |         |                  |
|                             |                            | 128, 4096                    | 5168.37           |                  |               |         |                  |
|                             |                            | 500, 2000                    | 5084.29           |                  |               |         |                  |
|                             |                            | 1000, 1000                   | 3399.69           |                  |               |         |                  |
|                             |                            | 2048, 128                    | 463.42            |                  |               |         |                  |
|                             |                            | 2048, 2048                   | 2940.62           |                  |               |         |                  |
|                             |                            | 5000, 500                    | 669.13            |                  |               |         |                  |
|                             |                            | 20000, 2000                  | 535.31            |                  |               |         |                  |
| Mistral 7B                  | 1                          | 128, 128                     | 31938.12          | 31674.49         | 32498.47      | 9664.13 | 6982.53          |
|                             |                            | 128, 2048                    | 27409.3           | 23496.42         | 23337.29      | 5720.65 | 5630.62          |
|                             |                            | 128, 4096                    | 18505.03          | 14350.99         | 15017.88      | 3136.33 | 3591.22          |
|                             |                            | 500, 2000                    | 22354.67          | 18026.27         | 18556         | 4521.77 | 4400.48          |
|                             |                            | 1000, 1000                   | 18426.16          | 16035.66         | 17252.11      | 4177.76 | 3896.58          |
|                             |                            | 2048, 128                    | 3834.43           | 3642.48          | 3813.13       | 1076.74 | 808.58           |
|                             |                            | 2048, 2048                   | 12347.37          | 9958.17          | 10755.94      | 2286.71 | 2489.77          |
|                             |                            | 5000, 500                    | 4041.59           | 3591.33          | 3949.66       | 1001.02 | 844.64           |
|                             |                            | 20000, 2000                  | 1822.69           | 1373.24          | 1601.28       | 337.83  | 332.3            |
| Mixtral 8x7B                | 1                          | 128, 128                     | 17157.72          | 15962.49         | 16859.18      |         |                  |
|                             |                            | 128, 2048                    | 15095.21          | 8290.13          | 11120.16      |         |                  |
|                             |                            | 128, 4096                    | 9534.62           | 4784.86          | 6610.47       |         |                  |
|                             |                            | 500, 2000                    | 12105.27          | 6800.6           | 9192.86       |         |                  |
|                             |                            | 1000, 1000                   | 10371.36          | 6868.52          | 8849.18       |         |                  |
|                             |                            | 2048, 128                    | 2009.67           | 1892.81          | 1994.31       |         |                  |
|                             |                            | 2048, 2048                   | 6940.32           | 3983.1           | 5545.46       |         |                  |
|                             |                            | 5000, 500                    | 2309.1            | 1764.7           | 2078.27       |         |                  |
|                             |                            | 20000, 2000                  | 1151.78           | 673.7            | 860.68        |         |                  |
|                             | 2                          | 128, 128                     | 27825.34          | 27451.13         |               |         | 5541.47          |
|                             |                            | 128, 2048                    | 29584.05          | 22830.08         |               |         | 4169.78          |
|                             |                            | 128, 4096                    | 21564.68          | 14237.01         |               |         | 2608.05          |
|                             |                            | 500, 2000                    | 23410.63          | 17036.04         |               |         | 3446.37          |
|                             |                            | 1000, 1000                   | 19151.19          | 15770.89         |               |         | 3154.52          |
|                             |                            | 2048, 128                    | 3383.16           | 3333.68          |               |         | 649              |
|                             |                            | 2048, 2048                   | 14007.29          | 10685.85         |               |         | 2056.58          |
|                             |                            | 5000, 500                    | 4223.68           | 3646.09          |               |         | 724.44           |
|                             |                            | 20000, 2000                  | 2299.21           | 1757.45          |               |         | 337.51           |
|                             | 4                          | 128, 128                     | 42551.59          | 41068.23         |               | 6921.87 | 10324.28         |
|                             |                            | 128, 2048                    | 52291.78          | 41164.73         |               | 7996.93 | 10911.86         |
|                             |                            | 128, 4096                    | 39513.73          | 27912.48         |               | 5736.09 | 7666.51          |
|                             |                            | 500, 2000                    | 43818.99          | 34489.34         |               | 6914.68 | 8456.21          |
|                             |                            | 1000, 1000                   | 33580.9           | 27784.74         |               | 5251.49 | 7122.84          |
|                             |                            | 2048, 128                    | 5467.62           | 5234.98          |               | 827.62  | 1237.62          |
|                             |                            | 2048, 2048                   | 24980.93          | 19432.08         |               | 3935.32 | 5222.98          |
|                             |                            | 5000, 500                    | 7084.94           | 6401.56          |               | 1092.88 | 1500.55          |
|                             |                            | 20000, 2000                  | 4236.84           | 3303.83          |               | 682.48  | 829.59           |
|                             | 8                          | 128, 128                     | 53212.55          | 50849.55         |               | 6740.84 | 17043.54         |
|                             |                            | 128, 2048                    | 68608.45          | 61607.7          |               | 10393.3 | 20277.88         |
|                             |                            | 128, 4096                    | 54827.78          | 48280.37         |               | 8472.35 | 15282.89         |
|                             |                            | 500, 2000                    | 58706.39          | 52583.65         |               | 8660.71 | 17184.24         |
|                             |                            | 1000, 1000                   | 44705.48          | 40631.71         |               | 5947.72 | 12851.44         |
|                             |                            | 2048, 128                    | 7554.38           | 6988.18          |               | 811.96  | 2165.52          |
|                             |                            | 2048, 2048                   | 36193.64          | 30983.35         |               | 5136.98 | 9809.76          |
|                             |                            | 5000, 500                    | 10271.8           | 9210.11          |               | 1153.76 | 2761.28          |
|                             |                            | 20000, 2000                  | 6835.53           | 5602.43          |               | 918.95  | 1592.53          |
| Mixtral 8x22B               | 8                          | 128, 128                     | 22948.57          | 21876.08         |               |         | 6381.95          |
|                             |                            | 128, 2048                    | 32415.81          | 25150.03         |               |         | 6685.99          |
|                             |                            | 128, 4096                    | 25753.14          | 18387.4          |               |         | 4789.13          |
|                             |                            | 500, 2000                    | 27429.6           | 21421.86         |               |         | 5648.46          |
|                             |                            | 1000, 1000                   | 19712.35          | 16573.24         |               |         | 4549.46          |
|                             |                            | 2048, 128                    | 2899.84           | 2794.97          |               |         | 761.56           |
|                             |                            | 2048, 2048                   | 15798.59          | 12244.93         |               |         | 3521.98          |
|                             |                            | 5000, 500                    | 4031.79           | 3645.27          |               |         | 959.14           |
|                             |                            | 20000, 2000                  | 2815.76           | 2227.63          |               |         | 575.02           |

*TP stands for Tensor Parallelism*

## Reproducing Benchmarked Results

> [!NOTE] The only models supported in this workflow are those listed in the table above.

The following tables are references for commands that are used as part of the benchmarking process. For a more detailed
description of this benchmarking workflow, see the [benchmarking suite documentation](https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html).

### Commands

#### For systems other than GH200
| Stage | Description | Command |
| :- | - | - |
| [Dataset](#preparing-a-dataset) | Create a synthetic dataset | `python benchmarks/cpp/prepare_dataset.py --tokenizer=$model_name --stdout token-norm-dist --num-requests=$num_requests --input-mean=$isl --output-mean=$osl --input-stdev=0 --output-stdev=0 > $dataset_file` |
| [Build](#engine-building) | Build a TensorRT-LLM engine | `trtllm-bench --model $model_name build --tp_size $tp_size --pp_size $pp_size --quantization FP8 --dataset $dataset_file` |
| [Run](#running-the-benchmark) | Run a benchmark with a dataset | `trtllm-bench --model $model_name throughput --dataset $dataset_file --engine_dir $engine_dir` |

#### For GH200 systems only
For release v0.17, on GH200 systems, the recommendation is to use the legacy flow based on *gptManagerBenchmark* to measure performance.

| Stage | Description | Command |
| :- | - | - |
| [Dataset](#preparing-a-dataset) | Create a synthetic dataset for engine building | `python benchmarks/cpp/prepare_dataset.py --tokenizer=$model_name --stdout token-norm-dist --num-requests=$num_requests --input-mean=$isl --output-mean=$osl --input-stdev=0 --output-stdev=0 > $dataset_file` |
| [Build](#engine-building) | Build a TensorRT-LLM engine | `trtllm-bench --model $model_name build --tp_size $tp_size --quantization FP8 --dataset $dataset_file` |
| [Dataset](#preparing-a-dataset) | Create a synthetic dataset for benchmarking in json format | `python benchmarks/cpp/prepare_dataset.py --output=$dataset_file_json --tokenizer=$model_name token-norm-dist --num-requests=$num_requests --input-mean=$isl --output-mean=$osl --input-stdev=0 --output-stdev=0` |
| [Run](#running-the-benchmark) | Run a benchmark with a dataset in json format | `/app/tensorrt_llm/benchmarks/cpp/gptManagerBenchmark --engine_dir $engine_dir --type IFB --api executor --dataset $dataset_file_json --eos_id -1 --log_iteration_data --scheduler_policy guaranteed_no_evict --kv_cache_free_gpu_mem_fraction 0.95 --output_csv result.csv --request_rate -1.0 --enable_chunked_context --warm_up 0` |


### Variables

| Name | Description |
| :- | - |
| `$isl` | Benchmark input sequence length. |
| `$osl` | Benchmark output sequence length. |
| `$tp_size` | Tensor parallel mapping degree to run the benchmark with |
| `$pp_size` | Pipeline parallel mapping degree to run the benchmark with |
| `$engine_dir` | Location to store built engine file (can be deleted after running benchmarks). |
| `$model_name` | HuggingFace model name eg. meta-llama/Llama-2-7b-hf or use the path to a local weights directory |
| `$dataset_file` | Location of the dataset file generated by `prepare_dataset.py` |
| `$num_requests` | The number of requests to generate for dataset generation |
| `$seq_len` | A sequence length of ISL + OSL |

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
| 2048         | 128           | 2176       | 3000               |
| 2048         | 2048          | 4096       | 1500               |
| 5000         | 500           | 5500       | 1500               |
| 1000         | 1000          | 2000       | 3000               |
| 500          | 2000          | 2500       | 3000               |
| 20000        | 2000          | 22000      | 1000               |

### Engine Building

All engines are built using the `trtllm-bench build` subcommand.
The basic command for FP8 quantized engines is as follows:

```
trtllm-bench --model $model_name build --tp_size $tp_size --pp_size $pp_size --quantization FP8 --dataset $dataset_file
```
When providing `--dataset` in the build subcommand, `trtllm-bench build` uses high-level statistics of the dataset (average ISL/OSL, max sequence length) and tuning heuristics to optimize engine build settings.

Alternatively, if you would like to build the engine with specific settings, you can do so by specifying the values for `max_batch_size` and `max_num_tokens`:

```
trtllm-bench --model $model_name build --tp_size $tp_size --pp_size $pp_size --quantization FP8 --max_seq_len $seq_len --max_batch_size $max_bs --max_num_tokens $max_token
```

If you would like to build an FP16 engine without any quantization, simply remove the `--quantization FP8` option. If using pre-quantized weights (e.g. `nvidia/Llama-3.1-70B-Instruct-FP8` from HuggingFace), please set the `--quantization` argument to the model dtype to ensure the KV Cache is set to the appropriate dtype.

> [!NOTE] If you specify FP8 quantization, the KV cache will automatically be set to FP8 as well!

The `trtllm-bench build` subcommand will output the path where the engine is located upon a successful build. For example,

```shell
===========================================================
ENGINE SAVED: /tmp/meta-llama/Llama-2-7b-hf/tp_1_pp_1
===========================================================
```

### Running the Benchmark

### For non GH200 systems
To run the benchmark with the generated data set, simply use the `trtllm-bench throughput` subcommand. The benchmarker will
run an offline maximum throughput scenario such that all requests are queued in rapid succession. You simply need to provide
the patch to the engine from the [build](#engine-building) phase and a [generated dataset](#preparing-a-dataset).

```shell
trtllm-bench --model $model_name throughput --dataset $dataset_file --engine_dir $engine_dir
```

In majority of cases, we also use a higher KV cache percentage by setting `--kv_cache_free_gpu_mem_fraction 0.95` in the benchmark command. This allows us to obtain better performance than the default setting of `0.90`. We fall back to `0.90` if we hit an out of memory issue.

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

## Online Serving Measurements

The [TensorRT-LLM backend](https://github.com/triton-inference-server/tensorrtllm_backend) is used to measure the performance of TensorRT-LLM for online serving.

The below table shows the throughput and latency under a serving scenario.

**All data in the table below was generated using version 0.14.0, with 500 requests and BF16 precision.**

|                 |                    |         |         |         |         |                  |                    |                    |                               |                         |
| --------------- | -------------------| --------| --------| --------| --------|------------------| ------------------ | ------------------ | ----------------------------- |------------------------ |
| **Model**       | **GPU**            | **TP**  | **Input Length** | **Output Length** | **QPS** | **Tput(req/s)**  | **Mean TTFT(ms)**  | **Mean ITL(ms)**   | **Total Token Tput (tok/s)**  | **Output Tput (tok/s)** |
|LLaMA 3.1 70B|H100 80GB HBM3|4|467|256|2|2|62|21|1406|498||
||||||4|4|68|24|2750|973|
||||||8|7|92|32|5256|1860|
||||||16|12|175|66|8941|3164|
||||||32|16|1229|86|11537|4083|
||||||INF|16|9123|85|11593|4103|
||||467|16|2|2|53|18|844|28|
||||||4|4|58|20|1908|63|
||||||8|8|71|24|3795|126|
||||||16|16|109|38|7492|248|
||||||32|28|1197|482|13655|452|
||||||INF|28|9126|548|13719|454|
||||202|214|2|2|48|20|780|401|
||||||4|4|51|22|1499|771|
||||||8|7|57|25|2702|1390|
||||||16|11|74|32|4364|2245|
||||||32|14|116|42|5837|3003|
||||||INF|16|4482|50|6725|3459|
|LLaMA 3.1 8B||1|467|256|2|2|23|8|1423|504|
||||||4|4|24|9|2624|929|
||||||8|8|26|9|5535|1959|
||||||16|15|30|11|10636|3765|
||||||32|26|50|19|19138|6774|
||||||INF|37|3335|39|26614|9420|
||||467|16|2|2|19|7|956|32|
||||||4|4|20|7|1910|63|
||||||8|8|22|7|3808|126|
||||||16|16|24|8|7567|251|
||||||32|31|29|10|14894|493|
||||||INF|79|3280|193|38319|1269|
||||202|214|2|2|19|7|809|416|
||||||4|4|20|8|1586|816|
||||||8|7|21|9|3047|1568|
||||||16|13|23|10|5597|2879|
||||||32|23|27|11|9381|4825|
||||||INF|39|1657|21|16117|8291|
|LLaMA 3.1 70B|H200 131GB HBM3|4|467|256|2|2|58|18|1411|499|
||||||4|4|63|20|2770|980|
||||||8|7|84|27|5328|1886|
||||||16|13|165|60|9224|3264|
||||||32|16|1279|83|11800|4176|
||||||INF|16|9222|83|11826|4185|
||||467|16|2|2|50|15|956|32|
||||||4|4|55|16|1909|63|
||||||8|8|67|20|3799|126|
||||||16|16|103|33|7499|248|
||||||32|28|1259|485|13586|450|
||||||INF|29|9074|546|13792|457|
||||202|214|2|2|43|17|793|408|
||||||4|4|46|18|1524|784|
||||||8|7|51|21|2796|1438|
||||||16|11|67|28|4639|2386|
||||||32|15|112|39|6288|3235|
||||||INF|17|4480|48|7230|3719|
|LLaMA 3.1 8B|H200 131GB HBM3|1|467|256|2|2|21|6|1425|504|
||||||4|4|23|7|2828|1001|
||||||8|8|24|7|5567|1971|
||||||16|15|27|9|10761|3809|
||||||32|27|44|16|19848|7025|
||||||INF|40|3237|36|28596|10121|
||||467|16|2|2|18|5|956|32|
||||||4|4|19|6|1910|63|
||||||8|8|20|6|3810|126|
||||||16|16|22|7|7567|250|
||||||32|31|27|9|14927|494|
||||||INF|81|3227|190|39007|1291|
||||202|214|2|2|17|6|812|418|
||||||4|4|18|6|1597|822|
||||||8|7|19|7|3088|1589|
||||||16|14|20|8|5771|2969|
||||||32|24|24|9|9931|5109|
||||||INF|43|1665|19|17861|9189|

*TP stands for Tensor Parallelism*

*TTFT stands for Time To First Token*

*ITL stands for Inter Token Latency*



### For GH200 systems only
For release v0.17, on GH200 systems, the recommendation is to use *gptManagerBenchmark* to measure performance. Throughput measurements are reported based on the below commands.
```shell
 /app/tensorrt_llm/benchmarks/cpp/gptManagerBenchmark  --engine_dir $engine_dir --type IFB --dataset $dataset_file_json --eos_id -1 --scheduler_policy guaranteed_no_evict --kv_cache_free_gpu_mem_fraction 0.95 --output_csv result.csv --request_rate -1.0 --enable_chunked_context --warm_up 0
```

The command will run the `gptManagerBenchmark` binary that will report the throughput and other metrics as part of its output
that can be compared with the table in the [Throughput Measurements](#throughput-measurements) of this README.
