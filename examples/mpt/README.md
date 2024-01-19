# MPT

This document explains how to build the [MPT](https://huggingface.co/mosaicml/mpt-7b) model using TensorRT-LLM and run on a single GPU and a single node with multiple GPUs.

## Overview
Currently we use `tensorrt_llm.models.GPTLMHeadModel` to build TRT engine for MPT models.
Support for float16, float32 and bfloat16 conversion. Just change `data_type` flags to any.

## Support Matrix
  * FP16
  * FP8
  * INT8 & INT4 Weight-Only
  * INT4 AWQ
  * FP8 KV CACHE
  * Tensor Parallel
  * MHA, MQA & GQA
  * STRONGLY TYPED

#### MPT 7B

### 1. Convert weights from HF Transformers to FT format

The [`convert_hf_mpt_to_ft.py`](./convert_hf_mpt_to_ft.py) script allows you to convert weights from HF Transformers format to FT format.

```bash
python convert_hf_mpt_to_ft.py -i mosaicml/mpt-7b -o ./ft_ckpts/mpt-7b/fp16/ -t float16

python convert_hf_mpt_to_ft.py -i mosaicml/mpt-7b -o ./ft_ckpts/mpt-7b/fp32/ --tensor_parallelism 4 -t float32
```

`--infer_gpu_num 4` is used to convert to FT format with 4-way tensor parallelism


### 2.1 Build TensorRT engine(s)

Examples of build invocations:

```bash
# Build a single-GPU float16 engine using FT weights.
python3 build.py --model_dir=./ft_ckpts/mpt-7b/fp16/1-gpu \
                 --max_batch_size 64 \
                 --use_gpt_attention_plugin \
                 --use_gemm_plugin \
                 --output_dir ./trt_engines/mpt-7b/fp16/1-gpu

# Build 4-GPU MPT-7B float32 engines
# Enable several TensorRT-LLM plugins to increase runtime performance. It also helps with build time.
python3 build.py --world_size=4 \
                 --parallel_build \
                 --max_batch_size 64 \
                 --max_input_len 512 \
                 --max_output_len 64 \
                 --use_gpt_attention_plugin \
                 --use_gemm_plugin \
                 --model_dir ./ft_ckpts/mpt-7b/fp32/4-gpu \
                 --output_dir=./trt_engines/mpt-7b/fp32/4-gpu
```

### 2.2 Build Smoothquant engine(s)
```bash
# Generate Smoothquantied weights and scaling factors.
python convert_hf_mpt_to_ft.py -i mosaicml/mpt-7b -o ./ft_ckpts/mpt-7b/sq/ -sq 0.5

# Build smoothquant engine.
python3 build.py --max_batch_size 64 \
                 --max_input_len 512 \
                 --max_output_len 64 \
                 --remove_input_padding \
                 --enable_context_fmha \
                 --use_gpt_attention_plugin \
                 --use_gemm_plugin \
                 --model_dir ./ft_ckpts/mpt-7b/sq/1-gpu \
                 --output_dir=./trt_engines/mpt-7b/sq/1-gpu-engine \
                 --use_smooth_quant \
                 --per_channel \
                 --per_token
```

### 2.3 Use Weight-only per-channel quantization
Examples of build invocations:

```bash
# Build INT8 weight-only engine.
python3 build.py --model_dir=./ft_ckpts/mpt-7b/fp16/1-gpu \
                 --max_batch_size 64 \
                 --use_weight_only \
                 --use_gpt_attention_plugin \
                 --use_gemm_plugin \
                 --output_dir ./trt_engines/mpt-7b/int8_weight_only/1-gpu

# Build INT4 weight-only engine.
python3 build.py --model_dir=./ft_ckpts/mpt-7b/fp16/1-gpu \
                 --max_batch_size 64 \
                 --use_weight_only \
                 --weight_only_precision int4 \
                 --use_gpt_attention_plugin \
                 --use_gemm_plugin \
                 --output_dir ./trt_engines/mpt-7b/int4_weight_only/1-gpu
```

### 2.4 Use AMMO quantization and AWQ-INT4

First make sure AMMO toolkit is installed (see [examples/quantization/README.md](/examples/quantization/README.md#preparation))


```bash
# RUN ammo
python3 examples/quantization/quantize.py --model_dir .mosaicml/mpt-7b \
                                          --qformat int4_awq \
                                          --calib_size 32 \
                                          --export_path ./
```

After quantization, `mpt_tp1_rank0.npz` file will be generated under export path, then We use `--quant_ckpt_path` pass it to build stage.

```bash
# Build INT4 AWQ engine
python3 build.py --model_dir=./ft_ckpts/mpt-7b/fp16/1-gpu \
                 --max_batch_size 64 \
                 --remove_input_padding \
                 --enable_context_fmha \
                 --use_gpt_attention_plugin \
                 --use_gemm_plugin \
                 --output_dir ./trt_engines/mpt-7b/int4_awq/1-gpu \
                 --use_weight_only \
                 --weight_only_precision int4_awq \
                 --per_group \
                 --quant_ckpt_path ./mpt_tp1_rank0.npz
```

### 3. Run TRT engine to check if the build was correct

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

### 1. Convert weights from HF Transformers to FT format

The [`convert_hf_mpt_to_ft.py`](./convert_hf_mpt_to_ft.py) script allows you to convert weights from HF Transformers format to FT format.


```bash
python convert_hf_mpt_to_ft.py -i mosaicml/mpt-30b -o ./ft_ckpts/mpt-7b/fp16/ --tensor_parallelism 4 -t float16
```

`--infer_gpu_num 4` is used to convert to FT format with 4-way tensor parallelism


### 2. Build TensorRT engine(s)

Examples of build invocations:

```bash
# Build 4-GPU MPT-30B float16 engines
python3 build.py --world_size=4 \
                 --parallel_build \
                 --max_batch_size 64 \
                 --max_input_len 512 \
                 --max_output_len 64 \
                 --use_gpt_attention_plugin \
                 --use_gemm_plugin \
                 --model_dir ./ft_ckpts/mpt-30b/fp16/4-gpu \
                 --output_dir=./trt_engines/mpt-30b/fp16/4-gpu
```

### 3. Run TRT engine to check if the build was correct

```bash
# Run 4-GPU MPT7B TRT engine on a sample input prompt
mpirun -n 4 --allow-run-as-root \
    python ../run.py --max_output_len 10 \
                     --engine_dir ./trt_engines/mpt-30b/fp16/4-gpu/  \
                     --tokenizer_dir mosaicml/mpt-30b
```

#### Replit Code V-1.5 3B
Same commands can be changed to convert [Replit Code V-1.5 3B](https://huggingface.co/replit/replit-code-v1_5-3b) to TRT LLM format. Below is an example to build Replit Code V-1.5 3B fp16 2-way tensor parallelized TRT engine.

### 1. Convert weights from HF Transformers to FT format

The [`convert_hf_mpt_to_ft.py`](./convert_hf_mpt_to_ft.py) script allows you to convert weights from HF Transformers format to FT format.


```bash
python convert_hf_mpt_to_ft.py -i ./replit-code-v1_5-3b -o ./ft_ckpts/replit-code-v1_5-3b/bf16/ --tensor_parallelism 2 -t bfloat16
```

`--infer_gpu_num 2` is used to convert to FT format with 2-way tensor parallelism


### 2. Build TensorRT engine(s)

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
                 --model_dir ./ft_ckpts/replit-code-v1_5-3b/bf16/2-gpu \
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
#### FP8 Post-Training Quantization

The example below uses the NVIDIA AMMO (AlgorithMic Model Optimization) toolkit for the model quantization process.

First make sure AMMO toolkit is installed (see [examples/quantization/README.md](/examples/quantization/README.md#preparation))

After successfully running the script, the output should be in .npz format, e.g. `quantized_fp8/llama_tp_1_rank0.npz`,
where FP8 scaling factors are stored.

```bash
# Quantize MPT 7B into FP8 and export a single-rank checkpoint
python examples/quantization/quantize.py --model_dir .mosaicml/mpt-7b \
                                         --dtype float16 \
                                         --qformat fp8 \
                                         --export_path ./quantized_fp8

# Build MPT 7B TP using binary checkpoint + PTQ scaling factors from the single-rank checkpoint
python build.py --model_dir ft_ckpts/mpt-7b/fp16 \
                --quantized_fp8_model_path ./quantized_fp8/mpt_tp1_rank0.npz \
                --use_gpt_attention_plugin \
                --use_gemm_plugin \
                --output_dir trt_engines/mpt-7b/fp8/1-gpu/ \
                --remove_input_padding \
                --enable_fp8 \
                --fp8_kv_cache
```
