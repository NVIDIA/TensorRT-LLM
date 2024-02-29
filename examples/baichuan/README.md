# Baichuan

This document shows how to build and run a Baichuan models (including `v1_7b`/`v1_13b`/`v2_7b`/`v2_13b`) in TensorRT-LLM on both single GPU and single node multi-GPU.

## Overview

The TensorRT-LLM Baichuan implementation can be found in [tensorrt_llm/models/baichuan/model.py](../../tensorrt_llm/models/baichuan/model.py). The TensorRT-LLM Baichuan example code is located in [`examples/baichuan`](./). There is one main file:

* [`copnvert_checkpoint.py`](./copnvert_checkpoint.py) to convert supported checkpoints into TensorRT-LLM format.

The script accepts an argument named model_version, whose value should be `v1_7b`/`v1_13b`/`v2_7b`/`v2_13b` and the default value is `v1_13b`.

In addition, there are two shared files in the parent folder [`examples`](../) for inference and evaluation:

* [`../run.py`](../run.py) to run the inference on an input text;
* [`../summarize.py`](../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset.

## Support Matrix
  * FP16
  * FP8
  * BF16
  * INT4 & INT8 Weight-Only
  * INT8 KV CACHE (+ AWQ/per-channel weight-only)
  * INT8 Smooth Quant
  * Groupwise quantization (AWQ/GPTQ)

## Usage

The TensorRT-LLM Baichuan example code locates at [examples/baichuan](./). It takes HF weights as input, and builds the corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### Build TensorRT engine(s)

Need to specify the HF Baichuan checkpoint path. For `v1_13b`, you should use whether [baichuan-inc/Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat) or [baichuan-inc/Baichuan-13B-Base](https://huggingface.co/baichuan-inc/Baichuan-13B-Base). For `v2_13b`, you should use whether [baichuan-inc/Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat) or [baichuan-inc/Baichuan2-13B-Base](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base). More Baichuan models could be found on [baichuan-inc](https://huggingface.co/baichuan-inc).

TensorRT-LLM Baichuan builds TensorRT engine(s) from HF checkpoint. If no checkpoint directory is specified, TensorRT-LLM will build engine(s) with dummy weights.

***For all kinds of checkpoints, they share the same trtllm-build command like:***

```bash
# Enable several TensorRT-LLM plugins to increase runtime performance. It also helps with build time.

# The TensorRT-LLM GPT Attention plugin (--gpt_attention_plugin) is
# enabled by default to increase runtime performance.
# 7B models should always enable `gpt_attention_plugin`` since RoPE is only
# supported with GPTAttention plugin now.
# Try gemm_plugin to prevent accuracy issue.
trtllm-build --checkpoint_dir ./trt_ckpt/baichuan_v1_13b/ \
             --output_dir ./trt_engines/baichuan_v1_13b/ \
             --gemm_plugin float16 \
             --max_batch_size=32 \
             --max_input_len=1024 \
             --max_output_len=512
```


Here're some examples for checkpoint conversion that take `v1_13b` as example:

```bash
# Build a single-GPU float16 engine from HF weights.
# Build the Baichuan V1 13B model using a single GPU and FP16.
python convert_checkpoint.py --model_version v1_13b \
                             --model_dir baichuan-inc/Baichuan-13B-Chat \
                             --dtype float16 \
                             --output_dir ./tmp/baichuan_v1_13b/trt_engines/fp16/1-gpu/

# Build the Baichuan V1 13B model using a single GPU and BF16.
python convert_checkpoint.py --model_version v1_13b \
                             --model_dir baichuan-inc/Baichuan-13B-Chat \
                             --dtype bfloat16 \
                             --output_dir ./tmp/baichuan_v1_13b/trt_engines/bf16/1-gpu/

# Build the Baichuan V1 13B model using a single GPU and apply INT8 weight-only quantization.
python convert_checkpoint.py --model_version v1_13b \
                             --model_dir baichuan-inc/Baichuan-13B-Chat \
                             --dtype float16 \
                             --use_weight_only \
                             --output_dir ./tmp/baichuan_v1_13b/trt_engines/int8_weight_only/1-gpu/

# Build the Baichuan V1 13B model using a single GPU and apply INT4 weight-only quantization.
python convert_checkpoint.py --model_version v1_13b \
                             --model_dir baichuan-inc/Baichuan-13B-Chat \
                             --dtype float16 \
                             --use_weight_only \
                             --weight_only_precision int4 \
                             --output_dir ./tmp/baichuan_v1_13b/trt_engines/int4_weight_only/1-gpu/

# Build Baichuan V1 13B using 2-way tensor parallelism.
python convert_checkpoint.py --model_version v1_13b \
                             --model_dir baichuan-inc/Baichuan-13B-Chat \
                             --dtype float16 \
                             --output_dir ./tmp/baichuan_v1_13b/trt_engines/fp16/1-gpu/ \
                             --world_size 2 \
                             --tp_size 2
```

#### INT8 KV cache
INT8 KV cache could be enabled to reduce memory footprint. It will bring more performance gains when batch size gets larger.

You can get the INT8 scale of KV cache through NVIDIA AMMO (AlgorithMic Model Optimization) toolkit, which features a
`--kv_cache_dtype` option.

Example:

```bash
python ../quantization/quantize.py --model_dir baichuan-inc/Baichuan-13B-Chat \
                                   --dtype float16 \
                                   --kv_cache_dtype int8 \
                                   --output_dir ./trt_ckpt/baichuan_int8kv_tp1 \
                                   --calib_size 512
```

**INT8 KV cache + per-channel weight-only quantization**

INT8 KV cache could be combined with per-channel weight-only quantization, as follows:
```bash
python ../quantization/quantize.py --model_dir baichuan-inc/Baichuan-13B-Chat \
                                   --dtype float16 \
                                   --qformat int4_wo \
                                   --kv_cache_dtype int8 \
                                   --output_dir ./trt_ckpt/baichuan_int4wo_int8kv_tp1 \
                                   --calib_size 512
```

**INT8 KV cache + AWQ**

In addition, you can enable INT8 KV cache together with AWQ (per-group INT4 weight-only quantization), as follows:

```bash
python ../quantization/quantize.py --model_dir baichuan-inc/Baichuan-13B-Chat \
                                   --dtype float16 \
                                   --qformat int4_awq \
                                   --kv_cache_dtype int8 \
                                   --output_dir ./trt_ckpt/baichuan_int4awq_int8kv_tp1 \
                                   --calib_size 512
```

#### SmoothQuant

The SmoothQuant supports all Baichuan model variants. Unlike the FP16 build where the HF weights are processed and loaded into the TensorRT-LLM directly, the SmoothQuant needs to load INT8 weights which should be pre-processed before building an engine.

`--smoothquant` is the starting point of INT8 inference. By default, it
will run the model in the _per-tensor_ mode.

Then, you can add any combination of `--per-token` and `--per-channel` to get the corresponding behaviors.

Examples of build invocations:
```bash
python convert_checkpoint.py --model_version v1_13b \
                --model_dir baichuan-inc/Baichuan-13B-Chat \
                --dtype float16 \
                --smoothquant 0.8 \
                --per_channel \
                --per_token \
                --output_dir ./tmp/baichuan_v1_13b/sq0.8/1-gpu/
```

#### FP8 Post-Training Quantization

The examples below uses the NVIDIA AMMO (AlgorithMic Model Optimization) toolkit for the model quantization process.

First make sure AMMO(version>=0.7.0) toolkit is installed (see [examples/quantization/README.md](/examples/quantization/README.md#preparation))

```bash
# Quantize HF Baichuan v2 13B into FP8 and export a single-rank checkpoint
python ../quantization/quantize.py --model_dir /code/model/Baichuan2-13B-Chat/ \
                                   --dtype float16 \
                                   --qformat fp8 \
                                   --output_dir ./quantized_fp8 \
                                   --calib_size 256
```

The quantized model checkpoint is saved to `./quantized_fp8/` for future TensorRT-LLM engine build directly with the `trtllm-build` command mentioned above.

#### Groupwise quantization (AWQ/GPTQ)
##### AWQ
NVIDIA AMMO toolkit is used for AWQ weight quantization. Please see [examples/quantization/README.md](/examples/quantization/README.md#preparation) for AMMO installation instructions.
```bash
# Quantize HF Baichuan 13B checkpoint into INT4 AWQ format
python ../quantization/quantize.py --model_dir /code/model/Baichuan2-13B-Chat/ \
                                   --dtype float16 \
                                   --qformat int4_awq \
                                   --output_dir ./quantized_int4-awq_gs128 \
                                   --calib_size 32
```
The quantized model checkpoint is saved to `./quantized_int4-awq_gs128/` for future TensorRT-LLM engine build directly with the `trtllm-build` command mentioned above.

##### GPTQ
To run the GPTQ Baichuan example, the following steps are required:

1. Weight quantization:

    Quantized weights for GPTQ can be generated using an open source project such as [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa.git).

    Let us build the TensorRT-LLM engine with the saved `./baichuan-2-13b-4bit-gs64.safetensors`.

2. Checkpoint conversion:

    ```bash
    # Build the Baichuan2 13B model using 2-way tensor parallelism and apply INT4 GPTQ quantization.
    # Compressed checkpoint safetensors are generated separately from GPTQ.
    python convert_checkpoint.py --model_version v2_13b \
                                 --quant_ckpt_path ./baichuan-2-13b-4bit-gs64.safetensors \
                                 --dtype float16 \
                                 --use_weight_only \
                                 --weight_only_precision int4_gptq \
                                 --group_size 64 \
                                 --world_size 2 \
                                 --tp_size 2 \
                                 --output_dir ./tmp/baichuan_v2_13b/trt_engines/int4_gptq_gs64/2-gpu/
    ```
    The quantized model checkpoint is saved for future TensorRT-LLM engine build directly with the `trtllm-build` command mentioned above.

### Run

To run a TensorRT-LLM Baichuan model using the engines generated by `trtllm-build`

```bash
# With fp16 inference
python ../run.py --input_text "世界上第二高的山峰是哪座？" \
                 --max_output_len=50 \
                 --tokenizer_dir baichuan-inc/Baichuan-13B-Chat \
                 --engine_dir=./tmp/baichuan_v1_13b/trt_engines/fp16/1-gpu/

# With bf16 inference
python ../run.py --input_text "世界上第二高的山峰是哪座？" \
                 --max_output_len=50 \
                 --tokenizer_dir baichuan-inc/Baichuan-13B-Chat \
                 --engine_dir=./tmp/baichuan_v1_13b/trt_engines/bf16/1-gpu/

# With INT8 weight-only quantization inference
python ../run.py --input_text "世界上第二高的山峰是哪座？" \
                 --max_output_len=50 \
                 --tokenizer_dir=baichuan-inc/Baichuan-13B-Chat \
                 --engine_dir=./tmp/baichuan_v1_13b/trt_engines/int8_weight_only/1-gpu/

# With INT4 weight-only quantization inference
python ../run.py --input_text "世界上第二高的山峰是哪座？" \
                 --max_output_len=50 \
                 --tokenizer_dir=baichuan-inc/Baichuan-13B-Chat \
                 --engine_dir=./tmp/baichuan_v1_13b/trt_engines/int8_weight_only/1-gpu/

# With 2-way tensor parallelism inference
mpirun -n 2 --allow-run-as-root \
    python ../run.py --input_text "世界上第二高的山峰是哪座？" \
                     --max_output_len=50 \
                     --tokenizer_dir=baichuan-inc/Baichuan-13B-Chat \
                     --engine_dir=./tmp/baichuan_v1_13b/trt_engines/fp16/2-gpu/
```

### Summarization using the Baichuan model

```bash
# Run summarization using the Baichuan V1 13B model in FP16.
python ../summarize.py --test_trt_llm \
                       --hf_model_dir baichuan-inc/Baichuan-13B-Chat \
                       --data_type fp16 \
                       --engine_dir ./tmp/baichuan_v1_13b/trt_engines/fp16/1-gpu/

# Run summarization using the Baichuan V1 13B model quantized to INT8.
python ../summarize.py --test_trt_llm \
                       --hf_model_dir baichuan-inc/Baichuan-13B-Chat \
                       --data_type fp16 \
                       --engine_dir ./tmp/baichuan_v1_13b/trt_engines/int8_weight_only/1-gpu/

# Run summarization using the Baichuan V1 13B model in FP16 using two GPUs.
mpirun -n 2 --allow-run-as-root \
    python ../summarize.py --test_trt_llm \
                           --hf_model_dir baichuan-inc/Baichuan-13B-Chat \
                           --data_type fp16 \
                           --engine_dir ./tmp/baichuan_v1_13b/trt_engines/fp16/2-gpu/
```
