(perf-overview)=

> [!IMPORTANT]
> As of TensorRT-LLM v0.10, these performance benchmarks have changed methodology to utilize in-flight batching and
no longer utilize static benchmarking. These numbers are initial measurements and are expected to improve in future
releases.

# Overview

This document summarizes performance measurements of TensorRT-LLM on H100
(Hopper), GH200 (Grace + Hopper), L40S (Ada) and A100 (Ampere) GPUs for a few key models.

The data in the following tables is provided as a reference point to help users
validate observed performance. It should not be considered as the peak
performance that can be delivered by TensorRT-LLM.

## Known Issues

The following issues are being addressed to improve the efficiency of TensorRT-LLM.

### Fused Matmul + Gated-SiLU (LLaMA)

The current implementation combines two Matmul operations into one Matmul followed by
a separate SwiGLU kernel (when `--use_fused_mlp=enable` is enabled). There is also a more
efficient implementation that runs single Matmul + SwiGLU fused kernel for FP8 on Hopper
(when `--use_fused_mlp=enable --gemm_swiglu_plugin fp8` is enabled). The gemm_swiglu_plugin
will support more data types and GPU architectures in the future release.

### Trtllm-bench has known issues on GH200

For release v0.15, on GH200 systems, we recommend using the legacy flow based on *gptManagerBenchmark* to measure performance.

## Throughput Measurements

The below table shows performance data where a local inference client is fed requests at an infinite rate (no delay between messages),
and shows the throughput client-server scenario under maximum load.

The performance numbers below were collected using the steps described in this document.

Note that for GH200 tests, TRT-LLM engines were built using *trtllm-bench build* but benchmarked with *gptManagerBenchmark*.

**All data in the table below was generated using version 0.15.0 and presents token throughput in tokens/second.**

|                 |                          |               |                     |                    |                    |                    |                    |           |
| --------------- | ------------------------ | ------------- | ------------------- | ------------------ | ------------------ | ------------------ | ------------------ | --------- |
| | GPU| | H100 80GB HBM3| | A100-SXM4-80GB| A100-PCIE-80GB| L40S| GH200 96GB HBM3 CG1 |
| | Precision| | FP8| Mixed| Mixed| Mixed| FP8| FP8 |
| Model| TP Size| Runtime Input/Output Lengths| | | | | |
| LLaMA v3 70B| 1| 128, 128| 3197.73| | | | | 4023.31
| | | 128, 2048| 826.72| | | | | 1855.98
| | | 128, 4096| | | | | | 915.15
| | | 500, 2000| 658.87| | | | | 1483.67
| | | 1000, 1000| 772.64| | | | | 1587.16
| | | 2048, 128| 331.26| | | | | 425.89
| | | 2048, 2048| 383.46| | | | | 823.43
| | | 5000, 500| 217.12| | | | | 391.38
| | 2| 128, 128| 6529.47| 3137.86| 1316.68| 792.95| |
| | | 128, 2048| 6008.16| 783.76| 532.07| | |
| | | 128, 4096| 3561.24| 404.23| 285.37| | |
| | | 500, 2000| 4792.7| 658.7| 436.46| | |
| | | 1000, 1000| 4221.4| 759.56| 484.59| 268.09| |
| | | 2048, 128| 773.11| 318.58| 147.22| 96.65| |
| | | 2048, 2048| 2648.62| 373.71| 255.21| | |
| | | 5000, 500| 905.34| 224.99| 123.5| 75.54| |
| | 4| 128, 128| 10848.71| 6387.29| 2713.51| 1347.36| 1474|
| | | 128, 2048| 10973.67| 5767.81| 2684.63| 1414.31| 1912.29|
| | | 128, 4096| 7426.74| 3421.36| 1914.57| 1140.75| 1357.84|
| | | 500, 2000| 9575.94| 4311.78| 2181.56| 1276.59| 1602.99|
| | | 1000, 1000| 7234.67| 4027.52| 1876.99| 927.93| 1193.23|
| | | 2048, 128| 1318.11| 781.29| 319.91| 161.66| 174.02|
| | | 2048, 2048| 5185.7| 2584.66| 1339.76| 872.31| 910.92|
| | | 5000, 500| 1568.88| 855.16| 388.86| 216.5| 242.62|
| | 8| 128, 128| 15440.55| 10966.81| 4647.93| 962.8| 1381.32|
| | | 128, 2048| 16416.2| 10270.37| 5046.42| 1487.53| 2120.54|
| | | 128, 4096| 12247.71| 6932.27| 3672.17| 1391.51| 1855.21|
| | | 500, 2000| 14561.62| 8967.15| 4379.68| 1205.63| 1879.86|
| | | 1000, 1000| 11226.01| 6973.77| 3236.83| 883.65| 1244.32|
| | | 2048, 128| 2057.59| 1341.65| 558.45| 141.12| 164.34|
| | | 2048, 2048| 7813.57| 4518.75| 2395.15| 769.53| 1091.57|
| | | 5000, 500| 2564.74| 1612.14| 706.33| 217.62| 243.14|
| LLaMA v3.1 8B| 1| 128, 128| 27792.16| 16116.63| 6552.62| 5158.57| 8982.97| 30803.29
| | | 128, 2048| 19965.18| 9894.49| 5220.03| 4640.02| 5297.21| 20770.93
| | | 128, 4096| 13222.06| 5758.98| 3326.45| 2906.77| 2989.17| 12487.35
| | | 500, 2000| 15782.2| 7953.1| 4191.62| 3736.1| 4263.97| 19175.02
| | | 1000, 1000| 14797.28| 7721.07| 3753.46| 3328.02| 4013.95| 15955.43
| | | 2048, 128| 3496.41| 1972.07| 789.56| 630.86| 1055.55| 4011.99
| | | 2048, 2048| 8980.42| 4370.61| 2366.86| 2125.4| 2162.8| 9072.93
| | | 5000, 500| 3477.61| 1802.2| 816.09| 693.38| 972.2| 3957.15
| | | 20000, 2000| 1378.69| 621.58| 330.47| 298.79| 326.02| 1459.86
| LLaMA v3.1 70B| 1| 128, 128| 3173.65| | | | | 4108.23
| | | 128, 2048| 804.73| | | | | 1940.33
| | | 128, 4096| | | | | | 981.15
| | | 500, 2000| 652.24| | | | | 1526.49
| | | 1000, 1000| 775.07| | | | | 1575.4
| | | 2048, 128| 328.44| | | | | 453.06
| | | 2048, 2048| 388.02| | | | | 838.55
| | | 5000, 500| 217.98| | | | | 383.32
| | | 20000, 2000| | | | | | 124.38
| | 2| 128, 128| 6399.24| 3143.32| 1330.41| 790.66| |
| | | 128, 2048| 5920.14| 784.73| 532.31| | |
| | | 128, 4096| 3580.79| 418.75| 285.01| | |
| | | 500, 2000| 4775.52| 660.68| 437.64| | |
| | | 1000, 1000| 4247.38| 785.36| 483.87| 267.63| |
| | | 2048, 128| 774.11| 315.43| 144.88| 94.83| |
| | | 2048, 2048| 2667.23| 384.36| 259.65| 137.09| |
| | | 5000, 500| 901.84| 210.7| 124.33| 76.77| |
| | | 20000, 2000| 410.93| | | | |
| | 4| 128, 128| 10589.19| 6392.74| 2716.71| 1192.33| 1469.28|
| | | 128, 2048| 11063.97| 5742.27| 2663.76| 1385.61| 1911.43|
| | | 128, 4096| 7428.89| 3457.03| 1913.13| 1206.15| 1357.83|
| | | 500, 2000| 9504.33| 4375.09| 2193.81| 1248.45| 1599.38|
| | | 1000, 1000| 7306.35| 4075.52| 1889.72| 999.4| 1187.23|
| | | 2048, 128| 1316.33| 779.81| 320.96| 162.09| 176.41|
| | | 2048, 2048| 5166.41| 2609.39| 1341.99| 874.11| 909.3|
| | | 5000, 500| 1566.63| 874.96| 389.99| 218.29| 242.95|
| | | 20000, 2000| 915.06| 406.36| 209.39| 141.13| 158.35|
| | 8| 128, 128| 15427.05| 10959.63| 4595.66| 943.87| 1381.25|
| | | 128, 2048| 16533.07| 10252.11| 4967.17| 1605.66| 2157.58|
| | | 128, 4096| 12008.26| 6915.81| 3594.1| 1449.32| 1895.68|
| | | 500, 2000| 14508.43| 8942.09| 4349.21| 1238.68| 1877.86|
| | | 1000, 1000| 11086.68| 6983.63| 3285.33| 907.21| 1242.34|
| | | 2048, 128| 2064.53| 1351.25| 556.48| 140.49| 163.53|
| | | 2048, 2048| 7768.15| 4515.31| 2464.13| 811.88| 1092.72|
| | | 5000, 500| 2533.55| 1589.18| 700.7| 212.07| 242.61|
| | | 20000, 2000| 1447.5| 847.42| 399.8| 140.86| 198.77|
| Mistral 7B| 1| 128, 128| 30177.4| 17025.15| 6968.4| 5444.55| 9526.7| 33795.78
| | | 128, 2048| 22060.45| 10324.05| 5556.98| 4960.48| 5669.19| 22724.8
| | | 128, 4096| 13773.03| 6205.41| 3430.11| 3077.47| 3091.88| 13916.10
| | | 500, 2000| 17229.29| 8294.02| 4339.77| 3883.38| 4498.74| 20702.51
| | | 1000, 1000| 15428.87| 7894.2| 3874.65| 3433.27| 4118.6| 17061.12
| | | 2048, 128| 3546.44| 2001.13| 793.57| 635.46| 1067.47| 4039.02
| | | 2048, 2048| 9118.64| 4520.74| 2440.45| 2187.82| 2231.66| 9998.65
| | | 5000, 500| 3493.52| 1838.75| 828.17| 702.36| 999.35| 4042.82
| | | 20000, 2000| 1267.96| 641| 334.06| 296.1| 336.18| 1521.67
| Mixtral 8x7B| 1| 128, 128| 15882.61| | | | | 16515.3
| | | 128, 2048| 8214.24| | | | | 10956.79
| | | 128, 4096| 4671.49| | | | | 6489.02
| | | 500, 2000| 6739.79| | | | | 8809.27
| | | 1000, 1000| 6787.62| | | | | 8402.89
| | | 2048, 128| 1885.43| | | | | 1932.28
| | | 2048, 2048| 3725.12| | | | | 5248.95
| | | 5000, 500| 1762.25| | | | | 2098.53
| | | 20000, 2000| 670.61| | | | | 870.76
| | 2| 128, 128| 27155.63| 15904.17| 5758.21| 3788.61| |
| | | 128, 2048| 23009.9| 7660.05| 4365.92| 2219.51| |
| | | 128, 4096| 14095.62| 4287.96| 2502.13| 1272.21| |
| | | 500, 2000| 16785.63| 6454.11| 3618.34| 1633.61| |
| | | 1000, 1000| 15867.12| 6492.47| 3316.43| 1734.39| |
| | | 2048, 128| 3367.65| 1895.85| 691.68| 465.45| |
| | | 2048, 2048| 10464.57| 3642.6| 1990.95| 1038.11| |
| | | 5000, 500| 3591.62| 1722.61| 755.64| 468.26| |
| | | 20000, 2000| 1739.08| 655.5| 334.67| 187.43| |
| | 4| 128, 128| 40731.73| 28272.32| 11612.27| 6075.21| 6756.75|
| | | 128, 2048| 41117.27| 23327.39| 11755.57| 7851.32| 7989.81|
| | | 128, 4096| 28143.35| 13906.89| 8052.85| 5920.37| 5655.07|
| | | 500, 2000| 34507.24| 16964.37| 9185.2| 6243.72| 6605.53|
| | | 1000, 1000| 27614.12| 16217.64| 7640.13| 4818.03| 5132.48|
| | | 2048, 128| 5275.25| 3416.82| 1383.85| 740| 811.01|
| | | 2048, 2048| 18441.12| 10381.54| 5403.69| 3842.39| 3837.68|
| | | 5000, 500| 6340.27| 3689.37| 1632.92| 966.38| 1072.16|
| | | 20000, 2000| 3231.36| 1717.02| 856.62| 619.01| 655.74|
| | 8| 128, 128| 51899.21| 40517.74| 18434.51| 5573.24| 6349.85|
| | | 128, 2048| 63701.21| 40322.45| 22120.7| 8657.63| 9696.71|
| | | 128, 4096| 47833.64| 27121.19| 16280.11| 7747.32| 8038.78|
| | | 500, 2000| 53260.36| 32190.46| 18439.46| 7393.45| 8319.84|
| | | 1000, 1000| 40321.28| 27487.98| 13842.01| 5041.55| 5593.52|
| | | 2048, 128| 7609.41| 5396.72| 2295.12| 670.71| 765.2|
| | | 2048, 2048| 25624.61| 17823.29| 10114.34| 4509.4| 4791.64|
| | | 5000, 500| 9527.29| 6475.64| 3009.15| 973.63| 1094.62|
| | | 20000, 2000| 5507.84| 3156.06| 1673.29| 770.41| 872.96|
| Mixtral 8x22B| 8| 128, 128| 22834.12| 16565.76| 6914.09| | 2470.15|
| | | 128, 2048| 24975.75| 11676.16| 7170.04| | 3629.98|
| | | 128, 4096| 17564.49| 7020.49| 5052.47| | 2933.79|
| | | 500, 2000| 21498.7| 10606.93| 6151.81| | 2959.66|
| | | 1000, 1000| 16383.52| 9803.47| 4790.88| | 2146.74|
| | | 2048, 128| 2945.44| 2028.84| 827.34| | 291.53|
| | | 2048, 2048| 11238.84| 5804.75| 3395| | 1830.44|
| | | 5000, 500| 3755.98| 2281.8| 1032.41| | 417.12|
| | | 20000, 2000| 2151.07| 1186.32| 597.81| | 323.37|

*TP stands for Tensor Parallelism*

## Reproducing Benchmarked Results

> [!NOTE] The only models supported in this workflow are those listed in the table above.

The following tables are references for commands that are used as part of the benchmarking process. For a more detailed
description of this benchmarking workflow, see the [benchmarking suite documentation](https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html).

### Commands

#### For non GH200 systems
| Stage | Description | Command |
| :- | - | - |
| [Dataset](#preparing-a-dataset) | Create a synthetic dataset | `python benchmarks/cpp/prepare_dataset.py --tokenizer=$model_name --stdout token-norm-dist --num-requests=$num_requests --input-mean=$isl --output-mean=$osl --input-stdev=0 --output-stdev=0 > $dataset_file` |
| [Build](#engine-building) | Build a TensorRT-LLM engine | `trtllm-bench --model $model_name build --tp_size $tp_size --pp_size $pp_size --quantization FP8 --dataset $dataset_file` |
| [Run](#running-the-benchmark) | Run a benchmark with a dataset | `trtllm-bench --model $model_name throughput --dataset $dataset_file --engine_dir $engine_dir` |

#### For GH200 systems only
For release v0.15, on GH200 systems, the recommendation is to use the legacy flow based on *gptManagerBenchmark* to measure performance.

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

If you would like to build an FP16 engine without any quantization, simply remove the `--quantization FP8` option.

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
For release v0.15, on GH200 systems, the recommendation is to use *gptManagerBenchmark* to measure performance. Throughput measurements are reported based on the below commands.
```shell
 /app/tensorrt_llm/benchmarks/cpp/gptManagerBenchmark  --engine_dir $engine_dir --type IFB --dataset $dataset_file_json --eos_id -1 --scheduler_policy guaranteed_no_evict --kv_cache_free_gpu_mem_fraction 0.95 --output_csv result.csv --request_rate -1.0 --enable_chunked_context --warm_up 0
```

> [!Warning] CUDA error: out of memory \
> For benchmarks with large models causing OOM error, the command above must be modified to use `--kv_cache_free_gpu_mem_fraction 0.90` to avoid the scenario.

The command will run the `gptManagerBenchmark` binary that will report the throughput and other metrics as part of its output
that can be compared with the table in the [Throughput Measurements](#throughput-measurements) of this README.
