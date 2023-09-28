# Benchmark for Python Runtime

This document explains how to benchmark the models supported by TensorRT-LLM on a single GPU, a single node with
multiple GPUs or multiple nodes with multiple GPUs.

## Overview

The benchmark implementation and entrypoint can be found in [`benchmarks/python/benchmark.py`](./benchmark.py). There are some other scripts in the directory:

* [`benchmarks/python/allowed_configs.py`](./allowed_configs.py) to define configuration for each supported model.
* [`benchmarks/python/base_benchmark.py`](./base_benchmark.py) to implement the base class for benchmark.
* [`benchmarks/python/gpt_benchmark.py`](./gpt_benchmark.py) to implement benchmark scripts for GPT and GPT-like(LLaMA/OPT/GPT-J/SmoothQuant-GPT) models.
* [`benchmarks/python/bert_benchmark.py`](./bert_benchmark.py) to implement benchmark scripts for BERT models.

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
[BENCHMARK] model_name gpt_350m world_size 1 num_heads 16 num_layers 24 hidden_size 1024 vocab_size 51200 precision float16 batch_size 1 input_length 60 output_length 20 build_time(s) 89.8 tokens_per_sec 378.12 percentile95(ms) 53.284 percentile99(ms) 53.284 latency(ms) 52.893
[BENCHMARK] model_name gpt_350m world_size 1 num_heads 16 num_layers 24 hidden_size 1024 vocab_size 51200 precision float16 batch_size 8 input_length 60 output_length 20 build_time(s) 89.8 tokens_per_sec 361.06 percentile95(ms) 55.739 percentile99(ms) 55.739 latency(ms) 55.392
[BENCHMARK] model_name gpt_350m world_size 1 num_heads 16 num_layers 24 hidden_size 1024 vocab_size 51200 precision float16 batch_size 64 input_length 60 output_length 20 build_time(s) 89.8 tokens_per_sec 246.03 percentile95(ms) 81.533 percentile99(ms) 81.533 latency(ms) 81.29
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
