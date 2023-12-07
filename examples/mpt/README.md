# MPT

This document explains how to build the [MPT](https://huggingface.co/mosaicml/mpt-7b) model using TensorRT-LLM and run on a single GPU and a single node with multiple GPUs.

## Overview
Currently we use `tensorrt_llm.models.GPTLMHeadModel` to build TRT engine for MPT models.
Support for float16, float32 and bfloat16 conversion. Just change `dtype` flags to any.

## Support Matrix
  * FP16
  * Tensor Parallel
  * MHA, MQA & GQA

#### MPT 7B

### 1. Build TensorRT engine(s) from HF

Examples of build invocations:

```bash
# Build a single-GPU float16 engine using HF weights.
# If you have your own model code and not the code from transformers library, then set trust_remote_code flag
# set the data type for conversion using dtype flag
python3 build.py --model_dir=/hf-model/mpt-7b \
                 --max_batch_size 16 \
                 --use_gpt_attention_plugin \
                 --use_gemm_plugin \
                 --dtype float16 \
                 --output_dir ./trt_engines/mpt-7b/fp16/1-gpu \
                 --trust_remote_code

# If you want to use model code from transformers library, then remove trust_remote_code flag and just send mosaicml/mpt-<> in the model_dir as shown below
python3 build.py --model_dir=mosaicml/mpt-7b \
                 --max_batch_size 16 \
                 --use_gpt_attention_plugin \
                 --use_gemm_plugin \
                 --output_dir ./trt_engines/mpt-7b/fp16/1-gpu

# Build 4-GPU MPT-7B float32 engines
# Enable several TensorRT-LLM plugins (gpt attention and gemm) to increase runtime performance. It also helps with build time.
python3 build.py --world_size=4 \
                 --dtype float32 \
                 --parallel_build \
                 --max_batch_size 64 \
                 --max_input_len 512 \
                 --max_output_len 64 \
                 --use_gpt_attention_plugin \
                 --use_gemm_plugin \
                 --model_dir /hf-model/mpt-7b \
                 --output_dir=./trt_engines/mpt-7b/fp32/4-gpu
```

### 2. Run TRT engine to check if the build was correct

```bash
python ../run.py --max_output_len 10 \
                 --engine_dir ./trt_engines/mpt-7b/fp16/1-gpu/ \
                 --tokenizer_dir mosaicml/mpt-7b

# Run 4-GPU MPT7B TRT engine on a sample input prompt
mpirun -n 4 --allow-run-as-root \
    python ../run.py --max_output_len 10 \
                     --engine_dir ./trt_engines/mpt-7b/fp32/4-gpu/ \
                     --tokenizer_dir mosaicml/mpt-7b
```

#### MPT 30B

Same commands can be changed to convert MPT 30B to TRT LLM format. Below is an example to build MPT30B fp16 4-way tensor parallelized TRT engine

### 1. Build TensorRT engine(s)

Examples of build invocations:

```bash
# Build 4-GPU MPT-30B float16 engines
python3 build.py --world_size=4 \
                 --dtype bfloat16 \
                 --parallel_build \
                 --max_batch_size 64 \
                 --max_input_len 512 \
                 --max_output_len 64 \
                 --use_gpt_attention_plugin \
                 --use_gemm_plugin \
                 --model_dir mosaicml/mpt-30b \
                 --output_dir=./trt_engines/mpt-30b/fp16/4-gpu
```

### 2. Run TRT engine to check if the build was correct

```bash
# Run 4-GPU MPT7B TRT engine on a sample input prompt
mpirun -n 4 --allow-run-as-root \
    python ../run.py --max_output_len 10 \
                     --engine_dir ./trt_engines/mpt-30b/fp16/4-gpu/  \
                     --tokenizer_dir mosaicml/mpt-30b
```

#### Replit Code V-1.5 3B
Same commands can be changed to convert [Replit Code V-1.5 3B](https://huggingface.co/replit/replit-code-v1_5-3b) to TRT LLM format. Below is an example to build Replit Code V-1.5 3B fp16 2-way tensor parallelized TRT engine.

### 1. Build TensorRT engine(s)

Examples of build invocations:

```bash
# Build 2-GPU Replit Code V-1.5 3B bfloat16 engines
python3 build.py --world_size=2 \
                 --parallel_build \
                 --max_batch_size 16 \
                 --max_input_len 512 \
                 --max_output_len 64 \
                 --use_gpt_attention_plugin \
                 --use_gemm_plugin \
                 --model_dir ./replit-code-v1_5-3b \
                 --output_dir=./trt_engines/replit-code-v1_5-3b/bf16/2-gpu
```
Here is the partial output of above command.

```bash
[11/15/2023-02:47:50] [TRT] [I] Total Activation Memory: 738233344
[11/15/2023-02:47:51] [TRT] [I] Total Weights Memory: 3523622456
[11/15/2023-02:47:51] [TRT] [I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +64, now: CPU 8316, GPU 5721 (MiB)
[11/15/2023-02:47:51] [TRT] [I] [MemUsageChange] Init cuDNN: CPU +0, GPU +64, now: CPU 8316, GPU 5785 (MiB)
[11/15/2023-02:47:51] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 192 MiB, GPU 3361 MiB
[11/15/2023-02:47:51] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in building engine: CPU +0, GPU +3361, now: CPU 0, GPU 3361 (MiB)
[11/15/2023-02:47:51] [TRT] [I] [MemUsageStats] Peak memory usage during Engine building and serialization: CPU: 12851 MiB
[11/15/2023-02:47:51] [TRT-LLM] [I] Total time of building gpt_bfloat16_tp2_rank1.engine: 00:00:04
[11/15/2023-02:47:51] [TRT-LLM] [I] Serializing engine to trt_engines/replit-code-v1_5-3b/bf16/2-gpu/gpt_bfloat16_tp2_rank1.engine...
[11/15/2023-02:48:02] [TRT-LLM] [I] Engine serialized. Total time: 00:00:10
[11/15/2023-02:48:02] [TRT-LLM] [I] Timing cache serialized to model.cache
[11/15/2023-02:48:02] [TRT-LLM] [I] Total time of building all 2 engines: 00:01:21
```

### 3. Run TRT engine to check if the build was correct

```bash
# Run 2-GPU Replit Code V-1.5 3B TRT engine on a sample input prompt
mpirun -n 2 --allow-run-as-root \
    python ../run.py --max_output_len 64 \
                     --input_text "def fibonacci" \
                     --engine_dir ./trt_engines/replit-code-v1_5-3b/bf16/2-gpu/ \
                     --tokenizer_dir ./replit-code-v1_5-3b/
```

Here is the output of above command.
```bash
Input: "def fibonacci"
Output: "(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))"
```
