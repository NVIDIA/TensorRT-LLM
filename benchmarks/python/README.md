# Benchmark Python Runtime

> [!WARNING] Python benchmark is not recommended to be used for benchmarking, please use C++ benchmark instead
> The Python benchmarking scripts can only benchmark the Python runtime, which do not support the latest features, such as in-flight batching.

This document explains how to benchmark the models supported by TensorRT-LLM on a single GPU, a single node with
multiple GPUs or multiple nodes with multiple GPUs using the Python runtime.

## Overview

The benchmark implementation and entrypoint can be found in [`benchmarks/python/benchmark.py`](./benchmark.py). There are some other scripts in the directory:

* [`benchmarks/python/allowed_configs.py`](./allowed_configs.py) to define configuration for each supported model.
* [`benchmarks/python/build.py`](./build.py) to build supported models for benchmarking.
* [`benchmarks/python/base_benchmark.py`](./base_benchmark.py) to implement the base class for benchmark.
* [`benchmarks/python/gpt_benchmark.py`](./gpt_benchmark.py) to implement benchmark scripts for GPT and GPT-like(LLaMA/OPT/GPT-J/SmoothQuant-GPT) models.
* [`benchmarks/python/bert_benchmark.py`](./bert_benchmark.py) to implement benchmark scripts for BERT models.
* [`benchmarks/python/enc_dec_benchmark.py`](./enc_dec_benchmark.py) to implement benchmark scripts for Encoder-Decoder models.

## Usage

Please use `help` option for detailed usages.
```
python benchmark.py -h
```

### 1. Single GPU benchmark
Take GPT-350M as an example:
```
python benchmark.py \
    -m gpt_350m \
    --mode plugin \
    --batch_size "1;8;64" \
    --input_output_len "60,20;128,20"
```
Expected outputs:
```
[BENCHMARK] model_name gpt_350m world_size 1 num_heads 16 num_kv_heads 16 num_layers 24 hidden_size 1024 vocab_size 51200 precision float16 batch_size 1 input_length 60 output_length 20 gpu_peak_mem(gb) 4.2 build_time(s) 25.67 tokens_per_sec 483.54 percentile95(ms) 41.537 percentile99(ms) 42.102 latency(ms) 41.362 compute_cap sm80
[BENCHMARK] model_name gpt_350m world_size 1 num_heads 16 num_kv_heads 16 num_layers 24 hidden_size 1024 vocab_size 51200 precision float16 batch_size 8 input_length 60 output_length 20 gpu_peak_mem(gb) 4.28 build_time(s) 25.67 tokens_per_sec 3477.28 percentile95(ms) 46.129 percentile99(ms) 46.276 latency(ms) 46.013 compute_cap sm80
[BENCHMARK] model_name gpt_350m world_size 1 num_heads 16 num_kv_heads 16 num_layers 24 hidden_size 1024 vocab_size 51200 precision float16 batch_size 64 input_length 60 output_length 20 gpu_peak_mem(gb) 4.8 build_time(s) 25.67 tokens_per_sec 19698.07 percentile95(ms) 65.739 percentile99(ms) 65.906 latency(ms) 64.981 compute_cap sm80
...
```
*Please note that the expected outputs is only for reference, specific performance numbers depend on the GPU you're using.*

### 2. Multi-GPU benchmark
Take GPT-175B as an example:
```
mpirun -n 8 python benchmark.py \
    -m gpt_175b \
    --mode plugin \
    --batch_size "1;8;64" \
    --input_output_len "60,20;128,20"
```

Note: Building multi-GPU engines in parallel could be a heavy workload for the CPU system. Tuning `mpirun --map-by <XXX>` option on your system may achieve significant boost in build time, for example:
```
mpirun --map-by socket -n 8 python build.py \
    --model gpt_175b \
    --mode ootb \
    --quantization fp8
```
