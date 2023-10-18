# MPT

This document explains how to build the [MPT](https://huggingface.co/mosaicml/mpt-7b) model using TensorRT-LLM and run on a single GPU and  a single node with multiple GPUs

## Overview
Currently we use `tensorrt_llm.models.GPTLMHeadModel` to build TRT engine for MPT models.
Support for float16, float32 and bfloat16 conversion. Just change `data_type` flags to any.

## Support Matrix
  * FP16
  * FP8
  * INT8 & INT4 Weight-Only
  * FP8 KV CACHE
  * Tensor Parallel
  * STRONGLY TYPED

#### MPT 7B

### 1. Convert weights from HF Tranformers to FT format

The [`hf_gpt_convert.py`](./convert_hf_mpt_to_ft.py) script allows you to convert weights from HF Tranformers format to FT format.

```bash
python convert_hf_mpt_to_ft.py -i mosaicml/mpt-7b -o ./ft_ckpts/mpt-7b/fp16/ -t float16

python convert_hf_mpt_to_ft.py -i mosaicml/mpt-7b -o ./ft_ckpts/mpt-7b/fp32/ --tensor_parallelism 4 -t float32
```

`--infer_gpu_num 4` is used to convert to FT format with 4-way tensor parallelism


### 2. Build TensorRT engine(s)

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

### 3. Run TRT engine to check if the build was correct

```bash
python run.py --engine_dir ./trt_engines/mpt-7b/fp16/1-gpu/ --max_output_len 10

# Run 4-GPU MPT7B TRT engine on a sample input prompt
mpirun -n 4 --allow-run-as-root python run.py --engine_dir ./trt_engines/mpt-7b/fp32/4-gpu/ --max_output_len 10
```

#### MPT 30B

Same commands can be changed to convert MPT 30B to TRT LLM format. Below is an example to build MPT30B fp16 4-way tensor parallelized TRT engine

### 1. Convert weights from HF Tranformers to FT format

The [`convert_hf_mpt_to_ft.py`](./convert_hf_mpt_to_ft.py) script allows you to convert weights from HF Tranformers format to FT format.


```bash
python convert_hf_mpt_to_ft.py -i mosaicml/mpt-30b -o ./ft_ckpts/mpt-7b/fp16/ --tensor_parallelism 4 -t float16
```

`--infer_gpu_num 4` is used to convert to FT format with 4-way tensor parallelism


### 2. Build TensorRT engine(s)

Examples of build invocations:

```bash
# Build 4-GPU MPT-30B float16 engines
# ALiBi is not supported with GPT attention plugin so we can't use that plugin to increase runtime performance
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
mpirun -n 4 --allow-run-as-root python run.py --engine_dir ./trt_engines/mpt-30b/fp16/4-gpu/ --max_output_len 10
```
