# Benchmark Python Runtime

> [!WARNING] Python benchmark is not recommended to be used for benchmarking, please use C++ benchmark instead
> The Python benchmarking scripts can only benchmark the Python runtime, which do not support the latest features, such as in-flight batching.

This document explains how to benchmark the models supported by TensorRT-LLM on a single GPU, a single node with
multiple GPUs or multiple nodes with multiple GPUs using the Python runtime.

## Overview

The benchmark implementation and entrypoint can be found in [`benchmarks/python/benchmark.py`](./benchmark.py). There are some other scripts in the directory:

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
Take LLaMA 7B as an example:
```
python benchmark.py \
    -m dec \
    --engine_dir llama_7b \
    --batch_size "1;8;64" \
    --input_output_len "60,20;128,20"
```
Expected outputs:
```
[BENCHMARK] model_name dec world_size 2 num_heads 32 num_kv_heads 32 num_layers 32 hidden_size 4096 vocab_size 32000 precision float16 batch_size 1 gpu_weights_percent 1.0 input_length 60 output_length 20 gpu_peak_mem(gb) 0.0 build_time(s) None tokens_per_sec 170.77 percentile95(ms) 117.591 percentile99(ms) 124.262 latency(ms) 117.115 compute_cap sm90 quantization QuantMode.FP8_QDQ|FP8_KV_CACHE generation_time(ms) 110.189 total_generated_tokens 19.0 generation_tokens_per_second 172.43
[BENCHMARK] model_name dec world_size 2 num_heads 32 num_kv_heads 32 num_layers 32 hidden_size 4096 vocab_size 32000 precision float16 batch_size 8 gpu_weights_percent 1.0 input_length 60 output_length 20 gpu_peak_mem(gb) 0.0 build_time(s) None tokens_per_sec 1478.55 percentile95(ms) 108.641 percentile99(ms) 109.546 latency(ms) 108.214 compute_cap sm90 quantization QuantMode.FP8_QDQ|FP8_KV_CACHE generation_time(ms) 98.194 total_generated_tokens 152.0 generation_tokens_per_second 1547.951
[BENCHMARK] model_name dec world_size 2 num_heads 32 num_kv_heads 32 num_layers 32 hidden_size 4096 vocab_size 32000 precision float16 batch_size 64 gpu_weights_percent 1.0 input_length 60 output_length 20 gpu_peak_mem(gb) 0.0 build_time(s) None tokens_per_sec 8214.87 percentile95(ms) 156.748 percentile99(ms) 160.203 latency(ms) 155.815 compute_cap sm90 quantization QuantMode.FP8_QDQ|FP8_KV_CACHE generation_time(ms) 111.078 total_generated_tokens 1216.0 generation_tokens_per_second 10947.303
...
```
*Please note that the expected outputs is only for reference, specific performance numbers depend on the GPU you're using.*

### 2. Multi-GPU benchmark
Take LLaMA 7B as an example:
```
mpirun -n 2 python benchmark.py \
    -m dec \
    --engine_dir llama_7b \
    --batch_size "1;8;64" \
    --input_output_len "60,20;128,20"
```
