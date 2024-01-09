# Baichuan

This document shows how to build and run a Baichuan models (including `v1_7b`/`v1_13b`/`v2_7b`/`v2_13b`) in TensorRT-LLM on both single GPU and single node multi-GPU.

## Overview

The TensorRT-LLM Baichuan implementation can be found in [tensorrt_llm/models/baichuan/model.py](../../tensorrt_llm/models/baichuan/model.py). The TensorRT-LLM Baichuan example code is located in [`examples/baichuan`](./). There is one main file:

* [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the Baichuan model.

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

Normally `build.py` only requires single GPU, but if you've already got all the GPUs needed while inferencing, you could enable parallelly building to make the engine building process faster by adding `--parallel_build` argument. Please note that currently `parallel_build` feature only supports single node.

Here're some examples that take `v1_13b` as example:

```bash
# Build a single-GPU float16 engine from HF weights.
# Enable the special TensorRT-LLM GPT Attention plugin (--use_gpt_attention_plugin) to increase runtime performance.
# 7B models should always add --use_gpt_attention_plugin since RoPE is only supported with GPTAttention plugin now.
# Try use_gemm_plugin to prevent accuracy issue.

# Build the Baichuan V1 13B model using a single GPU and FP16.
python build.py --model_version v1_13b \
                --model_dir baichuan-inc/Baichuan-13B-Chat \
                --dtype float16 \
                --use_gemm_plugin float16 \
                --use_gpt_attention_plugin float16 \
                --output_dir ./tmp/baichuan_v1_13b/trt_engines/fp16/1-gpu/

# Build the Baichuan V1 13B model using a single GPU and BF16.
python build.py --model_version v1_13b \
                --model_dir baichuan-inc/Baichuan-13B-Chat \
                --dtype bfloat16 \
                --use_gemm_plugin bfloat16 \
                --use_gpt_attention_plugin bfloat16 \
                --output_dir ./tmp/baichuan_v1_13b/trt_engines/bf16/1-gpu/

# Build the Baichuan V1 13B model using a single GPU and apply INT8 weight-only quantization.
python build.py --model_version v1_13b \
                --model_dir baichuan-inc/Baichuan-13B-Chat \
                --dtype float16 \
                --use_gemm_plugin float16 \
                --use_gpt_attention_plugin float16 \
                --use_weight_only \
                --output_dir ./tmp/baichuan_v1_13b/trt_engines/int8_weight_only/1-gpu/

# Build the Baichuan V1 13B model using a single GPU and apply INT4 weight-only quantization.
python build.py --model_version v1_13b \
                --model_dir baichuan-inc/Baichuan-13B-Chat \
                --dtype float16 \
                --use_gemm_plugin float16 \
                --use_gpt_attention_plugin float16 \
                --use_weight_only \
                --weight_only_precision int4 \
                --output_dir ./tmp/baichuan_v1_13b/trt_engines/int4_weight_only/1-gpu/

# Build Baichuan V1 13B using 2-way tensor parallelism.
python build.py --model_version v1_13b \
                --model_dir baichuan-inc/Baichuan-13B-Chat \
                --dtype float16 \
                --use_gemm_plugin float16 \
                --use_gpt_attention_plugin float16 \
                --output_dir ./tmp/baichuan_v1_13b/trt_engines/fp16/2-gpu/ \
                --world_size 2
```

#### INT8 KV cache
INT8 KV cache could be enabled to reduce memory footprint. It will bring more performance gains when batch size gets larger.

You can get the INT8 scale of KV cache through [`hf_baichuan_convert.py`](./hf_baichuan_convert.py), which features a
`--calibrate-kv-cache, -kv` option. Setting `-kv` will calibrate the model,
and then export the scaling factors needed for INT8 KV cache inference.


Example:

```bash
python3 hf_baichuan_convert.py -i baichuan-inc/Baichuan-13B-Chat -o ./tmp/baichuan_v1_13b/int8_kv_cache/ --calibrate-kv-cache -t fp16
```

[`build.py`](./build.py) add new options for the support of INT8 KV cache.

`--int8_kv_cache` is the command-line option to enable INT8 KV cache, and `--bin_model_dir` is the directory where the INT8 KV cache scales are located.

**INT8 KV cache + per-channel weight-only quantization**

INT8 KV cache could be combined with per-channel weight-only quantization, as follows:

Examples of INT8 weight-only quantization + INT8 KV cache

```bash
# Build model with both INT8 weight-only and INT8 KV cache enabled
python build.py --model_version v1_13b \
                --bin_model_dir ./tmp/baichuan_v1_13b/int8_kv_cache/1-gpu/ \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/baichuan_v1_13b/trt_engines/int8_kv_cache_weight_only/1-gpu \
                --int8_kv_cache \
                --use_weight_only
```

**INT8 KV cache + AWQ**

In addition, you can enable INT8 KV cache together with AWQ (per-group INT4 weight-only quantization) like the following command.

**NOTE**: AWQ checkpoint is passed through `--quant_ckpt_path`, and the INT8 scales for the KV cache are expected to be in the directory pointed by `--bin_model_dir`.

```bash
python build.py --model_version v1_13b \
                --quant_ckpt_path ./baichuan-v1-13b-4bit-gs128-awq.pt \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --use_weight_only \
                --weight_only_precision int4_awq \
                --per_group \
                --output_dir ./tmp/baichuan_v1_13b/trt_engines/int8_kv_cache_int4_awq/1-gpu \
                --int8_kv_cache \ # Turn on INT8 KV cache
                --bin_model_dir=./tmp/baichuan_v1_13b/int8_kv_cache/1-gpu/ # Directory to look for INT8 scale of KV cache
```

#### SmoothQuant

The SmoothQuant supports all Baichuan model variants. Unlike the FP16 build where the HF weights are processed and loaded into the TensorRT-LLM directly, the SmoothQuant needs to load INT8 weights which should be pre-processed before building an engine.

Example:
```bash
python3 hf_baichuan_convert.py -i baichuan-inc/Baichuan-13B-Chat -o ./tmp/baichuan_v1_13b/sq0.8/ -sq 0.8 --tensor-parallelism 1 --storage-type fp16
```

[`build.py`](./build.py) add new options for the support of INT8 inference of SmoothQuant models.

`--use_smooth_quant` is the starting point of INT8 inference. By default, it
will run the model in the _per-tensor_ mode.

Then, you can add any combination of `--per-token` and `--per-channel` to get the corresponding behaviors.

Examples of build invocations:

```bash
# Build model for SmoothQuant in the _per_token_ + _per_channel_ mode
python3 build.py --model_version v1_13b \
                 --bin_model_dir=./tmp/baichuan_v1_13b/sq0.8/1-gpu/ \
                 --use_gpt_attention_plugin float16 \
                 --remove_input_padding \
                 --enable_context_fmha \
                 --use_smooth_quant \
                 --per_token \
                 --per_channel
```

Note we use `--bin_model_dir` instead of `--model_dir` and `--meta_ckpt_dir` since SmoothQuant model needs INT8 weights and various scales from the binary files.

#### FP8 Post-Training Quantization

The examples below uses the NVIDIA AMMO (AlgorithMic Model Optimization) toolkit for the model quantization process.

First make sure AMMO(version>=0.4.0) toolkit is installed (see [examples/quantization/README.md](/examples/quantization/README.md#preparation))

After successfully running the script, the output should be in .npz format, e.g. `quantized_fp8/baichuan_tp_1_rank0.npz`,
where FP8 scaling factors are stored.

```bash
# Quantize HF Baichuan v2 13B into FP8 and export a single-rank checkpoint
python examples/quantization/quantize.py --model_dir /code/model/Baichuan2-13B-Chat/ \
                                         --dtype float16 \
                                         --qformat fp8 \
                                         --export_path ./quantized_fp8 \
                                         --calib_size 256 \

# Build Baichuan v2 13B TP=1 using original HF checkpoint + PTQ scaling factors from the single-rank checkpoint
python build.py --model_version v2_13b \
                --model_dir /code/model/Baichuan2-13B-Chat/ \
                --quantized_fp8_model_path ./quantized_fp8/baichuan_tp1_rank0.npz \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --output_dir ./tmp/baichuan_v2_13b/trt_engines/fp8/1-gpu/ \
                --remove_input_padding \
                --enable_context_fmha \
                --enable_fp8 \
                --fp8_kv_cache \
                --strongly_typed \
                --world_size 1
```

#### Groupwise quantization (AWQ/GPTQ)
One can enable AWQ/GPTQ INT4 weight-only quantization with these options when building engine with `build.py`:

- `--use_weight_only` enables weight-only GEMMs in the network.
- `--per_group` enable groupwise weight-only quantization.
- `--group_size` can support 64 and 128 now. Default value is 128. For Baichuan 13B models and TP=2, we should use 64 group size for kernel compatibility.
- `--weight_only_precision` should specify the weight-only quantization format. Supported formats are `int4_awq` or `int4_gptq`.
- `--quant_ckpt_path` passes the quantized checkpoint to build the engine.
- `--quantize_lm_head` add this flag to quantize lm_head layer for quantize.py and build.py when using AWQ. Do NOT quantize LM head by default.

AWQ/GPTQ examples below involves 2 steps:
1. Weight quantization
2. Build TRT-LLM engine

##### AWQ
1. Weight quantization:

    NVIDIA AMMO toolkit is used for AWQ weight quantization. Please see [examples/quantization/README.md](/examples/quantization/README.md#preparation) for AMMO installation instructions.

    ```bash
    # Quantize HF Baichuan 13B checkpoint into INT4 AWQ format
    python examples/quantization/quantize.py --model_dir baichuan-inc/Baichuan-13B-Chat \
                                             --dtype float16 \
                                             --qformat int4_awq \
                                             --group_size 128 \
                                             --export_path ./quantized_int4-awq_gs128 \
                                             --calib_size 32
    ```
    The quantized model checkpoint is saved to `./quantized_int4-awq_gs128/baichuan_tp1_rank0.npz` for future TensorRT-LLM engine build.

2. Build TRT-LLM engine:

    ```bash
    python build.py --model_version v1_13b \
                    --quant_ckpt_path ./quantized_int4-awq_gs128/baichuan_tp1_rank0.npz \
                    --dtype float16 \
                    --remove_input_padding \
                    --use_gpt_attention_plugin float16 \
                    --enable_context_fmha \
                    --use_gemm_plugin float16 \
                    --use_weight_only \
                    --weight_only_precision int4_awq \
                    --per_group \
                    --group_size 128 \
                    --output_dir ./tmp/baichuan_v1_13b/trt_engines/int4_awq_gs128/1-gpu/
    ```

##### GPTQ
To run the GPTQ Baichuan example, the following steps are required:

1. Weight quantization:

    Quantized weights for GPTQ can be generated using an open source project such as [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa.git).

    Let us build the TensorRT-LLM engine with the saved `./baichuan-2-13b-4bit-gs64.safetensors`.

2. Build TensorRT-LLM engine:

    ```bash
    # Build the Baichuan2 13B model using 2-way tensor parallelism and apply INT4 GPTQ quantization.
    # Compressed checkpoint safetensors are generated separately from GPTQ.
    python build.py --model_version v2_13b \
                    --quant_ckpt_path ./baichuan-2-13b-4bit-gs64.safetensors \
                    --dtype float16 \
                    --remove_input_padding \
                    --use_gpt_attention_plugin float16 \
                    --enable_context_fmha \
                    --use_gemm_plugin float16 \
                    --use_weight_only \
                    --weight_only_precision int4_gptq \
                    --per_group \
                    --group_size 64 \
                    --world_size 2 \
                    --tp_size 2 \
                    --output_dir ./tmp/baichuan_v2_13b/trt_engines/int4_gptq_gs64/2-gpu/
    ```

### Run

To run a TensorRT-LLM Baichuan model using the engines generated by build.py

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
