# Qwen

This document shows how to build and run a [Qwen](https://huggingface.co/Qwen) model in TensorRT LLM on both single GPU, single node multi-GPU.

- [Qwen](#qwen)
  - [Overview](#overview)
  - [Support Matrix](#support-matrix)
  - [Usage](#usage)
    - [Download model weights](#download-model-weights)
    - [Build TensorRT engine(s)](#build-tensorrt-engines)
      - [INT8 KV cache](#int8-kv-cache)
      - [SmoothQuant](#smoothquant)
      - [FP8 Post-Training Quantization](#fp8-post-training-quantization)
      - [INT4-GPTQ](#int4-gptq)
      - [INT4-AWQ](#int4-awq)
    - [Run](#run)
    - [Run models with LoRA](#run-models-with-lora)
    - [Summarization using the Qwen model](#summarization-using-the-qwen-model)
  - [Qwen3](#qwen3)
    - [Downloading the Model Weights](#downloading-the-model-weights)
    - [Quick start](#quick-start)
      - [Run a single inference](#run-a-single-inference)
    - [Evaluation](#evaluation)
    - [Model Quantization](#model-quantization)
    - [Benchmark](#benchmark)
    - [Serving](#serving)
      - [trtllm-serve](#trtllm-serve)
      - [Disaggregated Serving](#disaggregated-serving)
    - [Eagle3](#eagle3)
    - [Dynamo](#dynamo)
  - [Qwen3-Next](#qwen3-next)
  - [Notes and Troubleshooting](#notes-and-troubleshooting)
  - [Credits](#credits)

## Overview

The TensorRT LLM Qwen implementation can be found in [models/qwen](../../../../tensorrt_llm/models/qwen/). The TensorRT LLM Qwen example code is located in [`examples/models/core/qwen`](./). There is one main file:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the Qwen model.

In addition, there are two shared files in the parent folder [`examples`](../../../) for inference and evaluation:

* [`run.py`](../../../run.py) to run the inference on an input text;
* [`summarize.py`](../../../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset.

## Support Matrix
|   Model Name       | FP16/BF16  |  FP8  |  nvfp4  |  WO   |  AWQ  | GPTQ  |  SQ   |  TP   |  PP   |  EP   |  Arch   |
| :-------------:    |   :---:    | :---: | :-----: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :-----: |
| Qwen-1_8B(-Chat)   |     Y      |   Y   |    -    |   Y   |   Y*  |   Y   |   Y   |   Y   |   Y   |   -   | Ampere+ |
| Qwen-7B(-Chat)     |     Y      |   Y   |    -    |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   -   | Ampere+ |
| Qwen-14B(-Chat)    |     Y      |   Y   |    -    |   Y   |   Y*  |   Y   |   Y   |   Y   |   Y   |   -   | Ampere+ |
| Qwen-72B(-Chat)    |     Y      |   Y   |    -    |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   -   | Ampere+ |
| Qwen1.5-0.5B(-Chat)|     Y      |   Y   |    -    |   Y   |   Y*  |   Y   |   Y   |   Y   |   Y   |   -   | Ampere+ |
| Qwen1.5-1.8B(-Chat)|     Y      |   Y   |    -    |   Y   |   Y*  |   Y   |   Y   |   Y   |   Y   |   -   | Ampere+ |
| Qwen1.5-4B(-Chat)  |     Y      |   Y   |    -    |   Y   |   Y*  |   Y   |   Y   |   Y   |   Y   |   -   | Ampere+ |
| Qwen1.5-7B(-Chat)  |     Y      |   Y   |    -    |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   -   | Ampere+ |
| Qwen1.5-14B(-Chat) |     Y      |   Y   |    -    |   Y   |   Y*  |   Y   |   Y   |   Y   |   Y   |   -   | Ampere+ |
| Qwen1.5-32B(-Chat) |     Y      |   Y   |    -    |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   -   | Ampere+ |
| Qwen1.5-72B(-Chat) |     Y      |   Y   |    -    |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   -   | Ampere+ |
| Qwen1.5-110B(-Chat)|     Y      |   Y   |    -    |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   -   | Ampere+ |
| Qwen1.5-MoE-A2.7B(-Chat)|   Y   |   -   |    -    |   Y   |   -   |   -   |   -   |   Y   |   Y   |   -   | Ampere+ |
| Qwen2-0.5B(-Instruct)|     Y    |   Y   |    -    |   Y   |   Y*  |   Y   |   Y   |   Y   |   Y   |   -   | Ampere+ |
| Qwen2-1.5B(-Instruct)|     Y    |   Y   |    -    |   Y   |   Y*  |   Y   |   Y   |   Y   |   Y   |   -   | Ampere+ |
| Qwen2-7B(-Instruct)|     Y      |   Y   |    -    |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   -   | Ampere+ |
| Qwen2-57B-A14B(-Instruct)|  Y   |   -   |    -    |   Y   |   -   |   -   |   -   |   Y   |   Y   |   -   | Ampere+ |
| Qwen2-72B(-Instruct)|     Y     |   Y   |    -    |   Y   |   Y*  |   Y   |   Y   |   Y   |   Y   |   -   | Ampere+ |
| Qwen2.5-0.5B(-Instruct)|     Y  |   Y   |    -    |   Y   |   Y*  |   Y   |   Y   |   Y   |   Y   |   -   | Ampere+ |
| Qwen2.5-3B(-Instruct)|     Y    |   Y   |    -    |   Y   |   Y*  |   Y   |   Y   |   Y   |   Y   |   -   | Ampere+ |
| Qwen2.5-1.5B(-Instruct)|     Y  |   Y   |    -    |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   -   | Ampere+ |
| Qwen2.5-7B(-Instruct)|     Y    |   Y   |    -    |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   -   | Ampere+ |
| Qwen2.5-32B(-Instruct)|     Y   |   Y   |    -    |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   -   | Ampere+ |
| Qwen2.5-72B(-Instruct)|     Y   |   Y   |    -    |   Y   |   Y*  |   Y   |   Y   |   Y   |   Y   |   -   | Ampere+ |
| QwQ-32B            |     Y      |   Y   |    -    |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   -   | Ampere+ |
| Qwen3-32B          |     Y      |   Y   |    Y    |   -   |   -   |   -   |   -   |   Y   |   -   |   Y   | Hopper+ |
| Qwen3-235B-A22B    |     Y      |   Y   |    Y    |   -   |   -   |   -   |   -   |   Y   |   -   |   Y   | Hopper+ |

Please note that Y* sign means that the model does not support all the AWQ + TP combination.

* Model Name: the name of the model, the same as the name on HuggingFace
* WO: Weight Only Quantization (int8 / int4)
* AWQ: Activation Aware Weight Quantization (int4)
* GPTQ: Generative Pretrained Transformer Quantization (int4)
* SQ: Smooth Quantization (int8)
* TP: Tensor Parallel
* PP: Pipeline Parallel

Currently Qwen1 models does not support dynamic NTK and logn attention. Therefore, accuracy on long sequence input for the Qwen-7B and Qwen-14B model is not promised.

For Qwen3 models, we only list the largest models for dense and MoE architectures, but models of other sizes follow similar patterns.

## Usage

The TensorRT LLM Qwen example code locates at [examples/models/core/qwen](./). It takes HF weights as input, and builds the corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### Download model weights

Install the dependency packages and setup `git-lfs`.

```bash
# Install dependencies
pip install -r requirements.txt

# Setup git-lfs
git lfs install
```

Download one or more Qwen models that you would like to build to TensorRT LLM engines. You may download from the [HuggingFace](https://huggingface.co) hub:

```bash
git clone https://huggingface.co/Qwen/Qwen-7B-Chat   ./tmp/Qwen/7B
git clone https://huggingface.co/Qwen/Qwen-14B-Chat  ./tmp/Qwen/14B
git clone https://huggingface.co/Qwen/Qwen-72B-Chat  ./tmp/Qwen/72B
```

### Build TensorRT engine(s)

The [`convert_checkpoint.py`](./convert_checkpoint.py) script converts HF weights to TensorRT LLM checkpoints.

The `trtllm-build` command builds TensorRT LLM engines from TensorRT LLM checkpoints. The number of engine files is also same to the number of GPUs used to run inference.

Normally `trtllm-build` only requires single GPU, but if you've already got all the GPUs needed while inferencing, you could enable parallelly building to make the engine building process faster by adding `--workers` argument. Please note that currently `workers` feature only supports single node.

Here're some examples:

```bash
# Build a single-GPU float16 engine from HF weights.
# Try --gemm_plugin to prevent accuracy issue.

# Build the Qwen-7B-Chat model using a single GPU and FP16.
python convert_checkpoint.py --model_dir ./tmp/Qwen/7B/ \
                              --output_dir ./tllm_checkpoint_1gpu_fp16 \
                              --dtype float16

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp16 \
            --output_dir ./tmp/qwen/7B/trt_engines/fp16/1-gpu \
            --gemm_plugin float16

# Build the Qwen-7B-Chat model using a single GPU and BF16.
python convert_checkpoint.py --model_dir ./tmp/Qwen/7B/ \
                              --output_dir ./tllm_checkpoint_1gpu_bf16 \
                              --dtype bfloat16

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_bf16 \
            --output_dir ./tmp/qwen/7B/trt_engines/bf16/1-gpu \
            --gpt_attention_plugin bfloat16 \
            --gemm_plugin bfloat16

# Build the Qwen-7B-Chat model using a single GPU and apply INT8 weight-only quantization.
python convert_checkpoint.py --model_dir ./tmp/Qwen/7B/ \
                              --output_dir ./tllm_checkpoint_1gpu_fp16_wq \
                              --dtype float16 \
                              --use_weight_only \
                              --weight_only_precision int8

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp16_wq \
            --output_dir ./tmp/qwen/7B/trt_engines/weight_only/1-gpu/ \
            --gemm_plugin float16

# Build the Qwen-7B-Chat model using a single GPU and apply INT4 weight-only quantization.
python convert_checkpoint.py --model_dir ./tmp/Qwen/7B/ \
                              --output_dir ./tllm_checkpoint_1gpu_fp16_wq \
                              --dtype float16 \
                              --use_weight_only \
                              --weight_only_precision int4

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp16_wq \
            --output_dir ./tmp/qwen/7B/trt_engines/weight_only/1-gpu/ \
            --gemm_plugin float16

# Build Qwen-7B-Chat using 2-way tensor parallelism.
python convert_checkpoint.py --model_dir ./tmp/Qwen/7B/ \
                            --output_dir ./tllm_checkpoint_2gpu_tp2 \
                            --dtype float16 \
                            --tp_size 2

trtllm-build --checkpoint_dir ./tllm_checkpoint_2gpu_tp2 \
            --output_dir ./tmp/qwen/7B/trt_engines/fp16/2-gpu/ \
            --gemm_plugin float16

# Build Qwen-7B-Chat using 2-way tensor parallelism and 2-way pipeline parallelism.
python convert_checkpoint.py --model_dir ./tmp/Qwen/7B/ \
                            --output_dir ./tllm_checkpoint_4gpu_tp2_pp2 \
                            --dtype float16 \
                            --tp_size 2 \
                            --pp_size 2
trtllm-build --checkpoint_dir ./tllm_checkpoint_4gpu_tp2_pp2 \
            --output_dir ./tmp/qwen/7B/trt_engines/fp16/4-gpu/ \
            --gemm_plugin float16

# Build Qwen-14B-Chat using 2-way tensor parallelism.
python convert_checkpoint.py --model_dir ./tmp/Qwen/14B/ \
                            --output_dir ./tllm_checkpoint_2gpu_tp2 \
                            --dtype float16 \
                            --tp_size 2

trtllm-build --checkpoint_dir ./tllm_checkpoint_2gpu_tp2 \
            --output_dir ./tmp/qwen/14B/trt_engines/fp16/2-gpu/ \
            --gemm_plugin float16

# Build Qwen-72B-Chat using 8-way tensor parallelism.
python convert_checkpoint.py --model_dir ./tmp/Qwen/72B/ \
                            --output_dir ./tllm_checkpoint_8gpu_tp8 \
                            --dtype float16 \
                            --tp_size 8

trtllm-build --checkpoint_dir ./tllm_checkpoint_8gpu_tp8 \
            --output_dir ./tmp/qwen/72B/trt_engines/fp16/8-gpu/ \
            --gemm_plugin float16
```

#### INT8 KV cache
INT8 KV cache could be enabled to reduce memory footprint. It will bring more performance gains when batch size gets larger.

For INT8 KV cache, [`convert_checkpoint.py`](./convert_checkpoint.py) features a
`--int8_kv_cache` option. Setting `--int8_kv_cache` will calibrate the model,
and then export the scaling factors needed for INT8 KV cache inference.

Example:

```bash
python convert_checkpoint.py --model_dir ./tmp/Qwen/7B/   \
                             --output_dir ./tllm_checkpoint_1gpu_fp16_int8kv
                             --dtype float16  \
                             --int8_kv_cache

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_sq \
             --output_dir ./engine_outputs \
             --gemm_plugin float16
```

[`convert_checkpoint.py`](./convert_checkpoint.py) add new options for the support of INT8 KV cache.


#### SmoothQuant

The smoothquant supports Qwen models. Unlike the FP16 build where the HF weights are processed and loaded into the TensorRT LLM directly, the SmoothQuant needs to load INT8 weights which should be pre-processed before building an engine.

Example:
```bash
python3 convert_checkpoint.py --model_dir ./tmp/Qwen/7B/ --output_dir ./tllm_checkpoint_1gpu_sq --dtype float16 --smoothquant 0.5
trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_sq \
             --output_dir ./engine_outputs \
             --gemm_plugin float16
```

[`convert_checkpoint.py`](./convert_checkpoint.py) add new options for the support of INT8 inference of SmoothQuant models.

`--smoothquant` is the starting point of INT8 inference. By default, it
will run the model in the _per-tensor_ mode.

Then, you can add any combination of `--per-token` and `--per-channel` to get the corresponding behaviors.

Examples of build invocations:

```bash
# Build model for SmoothQuant in the _per_token_ + _per_channel_ mode
python3 convert_checkpoint.py --model_dir ./tmp/Qwen/7B/ \
                              --output_dir ./tllm_checkpoint_1gpu_sq \
                              --dtype float16 \
                              --smoothquant 0.5 \
                              --per_token \
                              --per_channel

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_sq \
             --output_dir ./engine_outputs \
             --gemm_plugin float16
```

#### FP8 Post-Training Quantization

The examples below uses the NVIDIA Modelopt (AlgorithMic Model Optimization) toolkit for the model quantization process.

First make sure Modelopt toolkit is installed (see [examples/quantization/README.md](/examples/quantization/README.md#preparation))


```bash
# Quantize model into FP8 and export trtllm checkpoint
python ../../../quantization/quantize.py --model_dir ./tmp/Qwen/7B/ \
                                   --dtype float16 \
                                   --qformat fp8 \
                                   --kv_cache_dtype fp8 \
                                   --output_dir ./tllm_checkpoint_1gpu_fp8 \
                                   --calib_size 512

# Build trtllm engines from the trtllm checkpoint
# Enable fp8 context fmha to get further acceleration by setting `--use_fp8_context_fmha enable`
trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp8 \
             --output_dir ./engine_outputs \
             --gemm_plugin float16 \
```

#### INT4-GPTQ
You may find the official GPTQ quantized INT4 weights of Qwen-7B-Chat here: [Qwen-7B-Chat-Int4](https://huggingface.co/Qwen/Qwen-7B-Chat-Int4).

Example of building engine for INT4 GPTQ quantized Qwen model:
```bash
python3 convert_checkpoint.py --model_dir ./tmp/Qwen-7B-Chat-Int4 \
                              --output_dir ./tllm_checkpoint_1gpu_gptq \
                              --dtype float16 \
                              --use_weight_only \
                              --weight_only_precision int4_gptq \
                              --per_group

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_gptq \
                --output_dir ./tmp/Qwen/7B/trt_engines/int4_GPTQ/1-gpu/ \
                --gemm_plugin float16
```

#### INT4-AWQ
To run the AWQ Qwen example, the following steps are required:
1. Weight quantization

    NVIDIA Modelopt toolkit is used for AWQ weight quantization. Please see [examples/quantization/README.md](/examples/quantization/README.md#preparation) for Modelopt installation instructions.

    ```bash
    # Quantize Qwen-7B-Chat checkpoint into INT4 AWQ format
    python ../../../quantization/quantize.py --model_dir ./tmp/Qwen/7B/ \
                                       --dtype float16 \
                                       --qformat int4_awq \
                                       --awq_block_size 128 \
                                       --output_dir ./quantized_int4-awq \
                                       --calib_size 32
    ```
    HF checkpoints generated with [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) are also supported through the following conversion script:

    ```bash
    # Convert AutoAWQ HF checkpoints into TRT-LLM checkpoint
    python convert_checkpoint.py --model_dir ./tmp/Qwen2-7B-Instruct-AWQ \
                                 --output_dir ./quantized_int4-awq
    ```

2. Build TRT-LLM engine:

    ```bash
    trtllm-build --checkpoint_dir ./quantized_int4-awq \
                 --output_dir ./tmp/qwen/7B/trt_engines/int4_AWQ/1-gpu/ \
                 --gemm_plugin float16
    ```

### Run

To run a TensorRT LLM Qwen model using the engines generated by `trtllm-build`

```bash
# With fp16 inference
python3 ../../../run.py --input_text "你好，请问你叫什么？" \
                  --max_output_len=50 \
                  --tokenizer_dir ./tmp/Qwen/7B/ \
                  --engine_dir=./tmp/Qwen/7B/trt_engines/fp16/1-gpu/

# With bf16 inference
python3 ../../../run.py --input_text "你好，请问你叫什么？" \
                  --max_output_len=50 \
                  --tokenizer_dir ./tmp/Qwen/7B/ \
                  --engine_dir=./tmp/Qwen/7B/trt_engines/bf16/1-gpu

# With int8 weight only inference
python3 ../../../run.py --input_text "你好，请问你叫什么？" \
                  --max_output_len=50 \
                  --tokenizer_dir ./tmp/Qwen/7B/ \
                  --engine_dir=./tmp/Qwen/7B/trt_engines/int8_weight_only/1-gpu/
```
```
Input [Text 0]: "<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好，请问你叫什么？<|im_end|>
<|im_start|>assistant
"
Output [Text 0 Beam 0]: "你好，我是来自阿里云的大规模语言模型，我叫通义千问。<|im_end|>
<|im_start|>
<|im_start|>

"
```

```bash
# With int4 weight only inference
python3 ../../../run.py --input_text "你好，请问你叫什么？" \
                  --max_output_len=50 \
                  --tokenizer_dir ./tmp/Qwen/7B/ \
                  --engine_dir=./tmp/Qwen/7B/trt_engines/int4_weight_only/1-gpu/
```
```
Input [Text 0]: "<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好，请问你叫什么？<|im_end|>
<|im_start|>assistant
"
Output [Text 0 Beam 0]: "我叫通义千问，是由阿里云开发的预训练语言模型。<|im_end|>
"
```

```bash
# With INT4 GPTQ quantization
python3 ../../../run.py --input_text "你好，请问你叫什么？" \
                  --max_output_len=50 \
                  --tokenizer_dir ./tmp/Qwen-7B-Chat-Int4 \
                  --engine_dir=./tmp/Qwen/7B/trt_engines/int4_GPTQ/1-gpu/
```
```
Input [Text 0]: "<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好，请问你叫什么？<|im_end|>
<|im_start|>assistant
"
Output [Text 0 Beam 0]: "你好，我是通义千问，由阿里云开发。<|im_end|>
"
```

```bash
# With INT4 AWQ quantization
python3 ../../../run.py --input_text "你好，请问你叫什么？" \
                  --max_output_len=50 \
                  --tokenizer_dir ./tmp/Qwen/7B/ \
                  --engine_dir=./tmp/Qwen/7B/trt_engines/int4_AWQ/1-gpu/
```
```
Input [Text 0]: "<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好，请问你叫什么？<|im_end|>
<|im_start|>assistant
"
Output [Text 0 Beam 0]: "你好，我是通义千问，由阿里云开发。<|im_end|>
"
```

```bash
# Run 72B model with 8-gpu
mpirun -n 8 --allow-run-as-root \
    python ../../../run.py --input_text "What is your name?" \
                     --max_output_len=50 \
                     --tokenizer_dir ./tmp/Qwen/72B/ \
                     --engine_dir=./tmp/Qwen/72B/trt_engines/fp16/8-gpu/
```
```
Input [Text 0]: "<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is your name?<|im_end|>
<|im_start|>assistant
"
Output [Text 0 Beam 0]: "I am QianWen, a large language model created by Alibaba Cloud."
```

### Run models with LoRA

Download the lora model from HF:

```bash
git clone https://huggingface.co/Jungwonchang/Ko-QWEN-7B-Chat-LoRA ./tmp/Ko-QWEN-7B-Chat-LoRA
```

Build engine, setting `--lora_plugin` and `--lora_dir`. If lora has separate lm_head and embedding, they will replace lm_head and embedding of base model.

```bash
python convert_checkpoint.py --model_dir ./tmp/Qwen/7B/ \
                              --output_dir ./tllm_checkpoint_1gpu_fp16 \
                              --dtype float16

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp16 \
            --output_dir ./tmp/qwen/7B_lora/trt_engines/fp16/1-gpu \
            --gemm_plugin auto \
            --lora_plugin auto \
            --lora_dir ./tmp/Ko-QWEN-7B-Chat-LoRA
```

Run inference:

```bash
python ../../../run.py --engine_dir ./tmp/qwen/7B_lora/trt_engines/fp16/1-gpu \
              --max_output_len 50 \
              --tokenizer_dir ./tmp/Qwen/7B/ \
              --input_text "안녕하세요, 혹시 이름이 뭐에요?" \
              --lora_task_uids 0 \
              --use_py_session

Input [Text 0]: "<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
안녕하세요, 혹시 이름이 뭐에요?<|im_end|>
<|im_start|>assistant
"
Output [Text 0 Beam 0]: "안녕하세요! 저는 인공지능 어시스턴트로, 여러분의 질문에 답하고 도움을 드리기 위해 여기 있습니다. 제가 무엇을 도와드릴까요?<|im_end|>
<|im_start|>0
<|im_start|><|im_end|>
<|im_start|>"
```

Users who want to skip LoRA module may pass uid -1 with `--lora_task_uids -1`.
In that case, the model will not run the LoRA module and the results will be
different.

```bash
python ../../../run.py --engine_dir ./tmp/qwen/7B_lora/trt_engines/fp16/1-gpu \
              --max_output_len 50 \
              --tokenizer_dir ./tmp/Qwen/7B/ \
              --input_text "안녕하세요, 혹시 이름이 뭐에요?" \
              --lora_task_uids -1 \
              --use_py_session

Input [Text 0]: "<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
안녕하세요, 혹시 이름이 뭐에요?<|im_end|>
<|im_start|>assistant
"
Output [Text 0 Beam 0]: "안녕하세요! 저는 "QianWen"입니다.<|im_end|>
"
```

### Summarization using the Qwen model

```bash
# Run summarization using the Qwen 7B model in FP16.
python ../../../summarize.py --test_trt_llm \
                       --hf_model_dir ./tmp/Qwen/7B/ \
                       --data_type fp16 \
                       --engine_dir ./tmp/Qwen/7B/trt_engines/fp16/1-gpu/ \
                       --max_input_length 2048 \
                       --output_len 2048

# Run summarization using the Qwen 7B model in BF16.
python ../../../summarize.py --test_trt_llm \
                       --hf_model_dir ./tmp/Qwen/7B/ \
                       --data_type fp16 \
                       --engine_dir ./tmp/Qwen/7B/trt_engines/bf16/1-gpu/ \
                       --max_input_length 2048 \
                       --output_len 2048

# Run summarization using the Qwen 7B model quantized to INT8.
python ../../../summarize.py --test_trt_llm \
                       --hf_model_dir  ./tmp/Qwen/7B/ \
                       --data_type fp16 \
                       --engine_dir ./tmp/Qwen/7B/trt_engines/int8_weight_only/1-gpu/ \
                       --max_input_length 2048 \
                       --output_len 2048

# Run summarization using the Qwen 7B model quantized to INT4.
python ../../../summarize.py --test_trt_llm \
                       --hf_model_dir  ./tmp/Qwen/7B/ \
                       --data_type fp16 \
                       --engine_dir ./tmp/Qwen/7B/trt_engines/int4_weight_only/1-gpu/ \
                       --max_input_length 2048 \
                       --output_len 2048

# Run summarization using the Qwen 7B model in FP16 using two GPUs.
mpirun -n 2 --allow-run-as-root \
    python ../../../summarize.py --test_trt_llm \
                           --hf_model_dir  ./tmp/Qwen/7B/ \
                           --data_type fp16 \
                           --engine_dir ./tmp/Qwen/7B/trt_engines/fp16/2-gpu/ \
                           --max_input_length 2048 \
                           --output_len 2048

# Run summarization using the Qwen 14B model in FP16 using two GPUs.
mpirun -n 2 --allow-run-as-root \
    python ../../../summarize.py --test_trt_llm \
                           --hf_model_dir  ./tmp/Qwen/14B/ \
                           --data_type fp16 \
                           --engine_dir ./tmp/Qwen/14B/trt_engines/fp16/2-gpu/ \
                           --max_input_length 2048 \
                           --output_len 2048
```
**Demo output of summarize.py:**
```bash
python ../../../summarize.py --test_trt_llm \
                       --hf_model_dir ./tmp/Qwen/7B/ \
                       --data_type fp16 \
                       --engine_dir ./tmp/Qwen/7B/trt_engines/fp16/1-gpu/ \
                       --max_input_length 2048 \
                       --output_len 2048
```
```
[11/09/2023-02:21:10] [TRT-LLM] [I] Load tokenizer takes: 0.4043385982513428 sec
Downloading builder script: 100%|███████████████████████████████████████████| 9.27k/9.27k [00:00<00:00, 35.4MB/s]
Downloading and preparing dataset cnn_dailymail/3.0.0 to /root/.cache/huggingface/datasets/ccdv___cnn_dailymail/3
......
[11/09/2023-02:23:33] [TRT-LLM] [I]
 Highlights : ['James Best, who played the sheriff on "The Dukes of Hazzard," died Monday at 88 .\n"Hazzard" ran from 1979 to 1985 and was among the most popular shows on TV .']
[11/09/2023-02:23:33] [TRT-LLM] [I]
 Summary : [['Actor James Best, known for his portrayal of bumbling sheriff Rosco P. Coltrane on TV\'s "The Dukes of Hazzard," has died at 88 after a brief illness. Best\'s career spanned decades in theater and Hollywood, but it was his role in "The Dukes of Hazzard" that made him a household name. The show ran for seven seasons from 1979 to 1985 and became a hit on TV, spawning TV movies, an animated series and video games. Best\'s portrayal of Rosco was beloved by fans for his childlike enthusiasm and goofy catchphrases. He is survived by friends and colleagues who paid tribute to him on social media.']]
[11/09/2023-02:23:33] [TRT-LLM] [I] ---------------------------------------------------------
load rouge ...
Downloading builder script: 5.60kB [00:00, 18.9MB/s]
load rouge done
[11/09/2023-02:24:06] [TRT-LLM] [I] TensorRT LLM (total latency: 30.13867211341858 sec)
[11/09/2023-02:24:06] [TRT-LLM] [I] TensorRT LLM beam 0 result
[11/09/2023-02:24:06] [TRT-LLM] [I]   rouge1 : 26.35215119137573
[11/09/2023-02:24:06] [TRT-LLM] [I]   rouge2 : 9.507814774384485
[11/09/2023-02:24:06] [TRT-LLM] [I]   rougeL : 18.171982659482865
[11/09/2023-02:24:06] [TRT-LLM] [I]   rougeLsum : 21.10413175647868
```

## Qwen3

TensorRT LLM now supports Qwen3, the latest version of the Qwen model series. This guide walks you through the examples to run the Qwen3 models using NVIDIA's TensorRT LLM framework with the PyTorch backend. According to the support matrix, TensorRT LLM provides comprehensive support for various Qwen3 model variants including:

- Qwen3-0.6B
- Qwen3-1.7B
- Qwen3-4B
- Qwen3-8B
- Qwen3-14B
- Qwen3-32B
- Qwen3-30B-A3B
- Qwen3-235B-A22B

Please refer to [this guide](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html) for how to build TensorRT LLM from source and start a TRT-LLM docker container if needed.

> [!NOTE]
> This guide assumes that you replace placeholder values (e.g. `<YOUR_MODEL_DIR>`) with the appropriate paths.

### Downloading the Model Weights

Qwen3 model weights are available on [Hugging Face](https://huggingface.co/Qwen/Qwen3-30B-A3B). To download the weights, execute the following commands (replace `<YOUR_MODEL_DIR>` with the target directory where you want the weights stored):

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-30B-A3B <YOUR_MODEL_DIR>
```

### Quick start

#### Run a single inference

To quickly run Qwen3, [examples/llm-api/quickstart_advanced.py](../../../llm-api/quickstart_advanced.py):

```bash
python3 examples/llm-api/quickstart_advanced.py --model_dir Qwen3-30B-A3B/ --kv_cache_fraction 0.6
```

### Evaluation

1. Evaluate accuracy on the MMLU dataset:

```bash
trtllm-eval --model=Qwen3-32B/ --tokenizer=Qwen3-32B/ --backend=pytorch mmlu --dataset_path=./datasets/mmlu/
[05/01/2025-13:56:15] [TRT-LLM] [I] MMLU weighted average accuracy: 79.09 (14042)
```

```bash
trtllm-eval --model=Qwen3-30B-A3B/ --tokenizer=Qwen3-30B-A3B/ --backend=pytorch mmlu --dataset_path=./datasets/mmlu/
[05/05/2025-11:33:02] [TRT-LLM] [I] MMLU weighted average accuracy: 79.44 (14042)
```

2. Evaluate accuracy on GSM8K dataset:

```bash
trtllm-eval --model=Qwen3-30B-A3B/ --tokenizer=Qwen3-30B-A3B/ --backend=pytorch gsm8k --dataset_path=./datasets/openai/gsm8k/
[05/05/2025-12:05:40] [TRT-LLM] [I] lm-eval gsm8k results (scores normalized to range 0~100):
|Tasks|Version|     Filter     |n-shot|  Metric   |   | Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|------:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |84.3063|±  |1.0019|
|     |       |strict-match    |     5|exact_match|↑  |88.6277|±  |0.8745|

```

### Model Quantization

To quantize the Qwen3 model for use with the PyTorch backend, we'll use NVIDIA's Model Optimizer (ModelOpt) tool. Follow these steps:

```bash
# Clone the TensorRT Model Optimizer (ModelOpt)
git clone https://github.com/NVIDIA/TensorRT-Model-Optimizer.git
pushd TensorRT-Model-Optimizer

# install the ModelOpt
pip install -e .

# Quantize the Qwen3-235B-A22B model by nvfp4
# By default, the checkpoint would be stored in `TensorRT-Model-Optimizer/examples/llm_ptq/saved_models_Qwen3-235B-A22B_nvfp4_hf/`.
./examples/llm_ptq/scripts/huggingface_example.sh --model Qwen3-235B-A22B/ --quant nvfp4 --export_fmt hf

# Quantize the Qwen3-32B model by fp8_pc_pt
# By default, the checkpoint would be stored in `TensorRT-Model-Optimizer/examples/llm_ptq/saved_models_Qwen3-32B_fp8_pc_pt_hf/`.
./examples/llm_ptq/scripts/huggingface_example.sh --model Qwen3-32B/ --quant fp8_pc_pt --export_fmt hf
popd
```

### Benchmark

To run the benchmark, we suggest using the `trtllm-bench` tool. Please refer to the following script on B200:

```bash
#!/bin/bash

folder_model=TensorRT-Model-Optimizer/examples/llm_ptq/saved_models_Qwen3-235B-A22B_nvfp4_hf/
path_config=extra-llm-api-config.yml
num_gpus=8
ep_size=8
max_input_len=1024
max_batch_size=512
# We want to limit the number of prefill requests to 1 with in-flight batching.
max_num_tokens=$(( max_input_len + max_batch_size - 1 ))
kv_cache_free_gpu_mem_fraction=0.9
concurrency=128

path_data=./aa_prompt_isl_1k_osl_2k_qwen3_10000samples.txt

# Setup the extra configuration for llm-api
echo -e "disable_overlap_scheduler: false
print_iter_log: true
cuda_graph_config:
  batch_sizes: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128]
enable_attention_dp: true " > ${path_config}

# Run trtllm-bench with pytorch backend
mpirun --allow-run-as-root --oversubscribe -n 1 \
trtllm-bench --model ${folder_model} --model_path ${folder_model} throughput \
  --backend pytorch \
  --max_batch_size ${max_batch_size} \
  --max_num_tokens ${max_num_tokens} \
  --dataset ${path_data} \
  --tp ${num_gpus}\
  --ep ${ep_size} \
  --kv_cache_free_gpu_mem_fraction ${kv_cache_free_gpu_mem_fraction} \
  --extra_llm_api_options ${path_config} \
  --concurrency ${concurrency} \
  --num_requests $(( concurrency * 5 )) \
  --warmup 0 \
  --streaming
```

We suggest benchmarking with a real dataset. It will prevent from having improperly distributed tokens in the MoE. Here, we use the `aa_prompt_isl_1k_osl_2k_qwen3_10000samples.txt` dataset. It has 10000 samples with an average input length of 1024 and an average output length of 2048. If you don't have a dataset (this or an other) and you want to run the benchmark, you can use the following command to generate a random dataset:

```bash
folder_model=TensorRT-Model-Optimizer/examples/llm_ptq/saved_models_Qwen3-235B-A22B_nvfp4_hf/
min_input_len=1024
min_output_len=2048
concurrency=128
path_data=random_data.txt

python3 benchmarks/cpp/prepare_dataset.py \
    --tokenizer=${folder_model} \
    --stdout token-norm-dist --num-requests=$(( concurrency * 5 )) \
    --input-mean=${min_input_len} --output-mean=${min_output_len} --input-stdev=0 --output-stdev=0 > ${path_data}
```

### Serving
#### trtllm-serve

To serve the model using `trtllm-serve`:

```bash
cat >./extra-llm-api-config.yml <<EOF
cuda_graph_config:
  enable_padding: true
  batch_sizes:
  - 1
  - 2
  - 4
  - 8
  - 16
  - 32
  - 64
  - 128
  - 256
  - 384
print_iter_log: true
enable_attention_dp: true
EOF

trtllm-serve \
  Qwen3-30B-A3B/ \
  --host localhost \
  --port 8000 \
  --max_batch_size 161 \
  --max_num_tokens 1160 \
  --tp_size 1 \
  --ep_size 1 \
  --pp_size 1 \
  --kv_cache_free_gpu_memory_fraction 0.8 \
  --extra_llm_api_options ./extra-llm-api-config.yml
```

To query the server, you can start with a `curl` command:
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "Qwen3-30B-A3B/",
      "prompt": "Please describe what is Qwen.",
      "max_tokens": 12,
      "temperature": 0
  }'
```
#### Disaggregated Serving

To serve the model in disaggregated mode, you should launch context and generation servers using `trtllm-serve`.

For example, you can launch a single context server on port 8001 with:

```bash
export TRTLLM_USE_UCX_KVCACHE=1

cat >./ctx-extra-llm-api-config.yml <<EOF
print_iter_log: true
enable_attention_dp: true
EOF

trtllm-serve \
  Qwen3-30B-A3B/ \
  --host localhost \
  --port 8001 \
  --max_batch_size 161 \
  --max_num_tokens 1160 \
  --tp_size 1 \
  --ep_size 1 \
  --pp_size 1 \
  --kv_cache_free_gpu_memory_fraction 0.8 \
  --extra_llm_api_options ./ctx-extra-llm-api-config.yml &> output_ctx &
```

And you can launch two generation servers on port 8002 and 8003 with:

```bash
export TRTLLM_USE_UCX_KVCACHE=1

cat >./gen-extra-llm-api-config.yml <<EOF
cuda_graph_config:
  enable_padding: true
  batch_sizes:
    - 1
    - 2
    - 4
    - 8
    - 16
    - 32
    - 64
    - 128
    - 256
    - 384
print_iter_log: true
enable_attention_dp: true
EOF

for port in {8002..8003}; do \
trtllm-serve \
  Qwen3-30B-A3B/ \
  --host localhost \
  --port ${port} \
  --max_batch_size 161 \
  --max_num_tokens 1160 \
  --tp_size 1 \
  --ep_size 1 \
  --pp_size 1 \
  --kv_cache_free_gpu_memory_fraction 0.8 \
  --extra_llm_api_options ./gen-extra-llm-api-config.yml \
  &> output_gen_${port} & \
done
```

Finally, you can launch the disaggregated server which will accept requests from the client and do
the orchestration between the context and generation servers with:

```bash
cat >./disagg-config.yml <<EOF
hostname: localhost
port: 8000
backend: pytorch
context_servers:
  num_instances: 1
  urls:
      - "localhost:8001"
generation_servers:
  num_instances: 1
  urls:
      - "localhost:8002"
EOF

trtllm-serve disaggregated -c disagg-config.yaml
```

To query the server, you can start with a `curl` command:
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "Qwen3-30B-A3B/",
      "prompt": "Please describe what is Qwen.",
      "max_tokens": 12,
      "temperature": 0
  }'
```

Note that the optimal disaggregated serving configuration (i.e. tp/pp/ep mappings, number of ctx/gen instances, etc.) will depend
on the request parameters, the number of concurrent requests and the GPU type. It is recommended to experiment to identify optimal
settings for your specific use case.

#### Eagle3

Qwen3 now supports Eagle3 (Speculative Decoding with Eagle3). To enable Eagle3 on Qwen3, you need to set the following arguments when running `trtllm-bench` or `trtllm-serve`:

- `speculative_config.decoding_type: Eagle`
  Set the decoding type to "Eagle" to enable Eagle3 speculative decoding.
- `speculative_config.max_draft_len: 3`
  Set the maximum number of draft tokens generated per step (this value can be adjusted as needed).
- `speculative_config.speculative_model_dir: <EAGLE3_DRAFT_MODEL_PATH>`
  Specify the path to the Eagle3 draft model (ensure the corresponding draft model weights are prepared).

Currently, there are some limitations when enabling Eagle3:

1. `attention_dp` is not supported. Please disable it or do not set the related flag (it is disabled by default).
2. If you want to use `enable_block_reuse`, the kv cache type of the target model and the draft model must be the same. Since the draft model only supports fp16/bf16, you need to disable `enable_block_reuse` when using fp8 kv cache.

Example `extra-llm-api-config.yml` snippet for Eagle3:

```bash
echo "
enable_attention_dp: false
speculative_config:
    decoding_type: Eagle
    max_draft_len: 3
    speculative_model_dir: <EAGLE3_DRAFT_MODEL_PATH>
kv_cache_config:
    enable_block_reuse: false
" >> ${path_config}
```

For further details, please refer to [speculative-decoding.md](../../../../docs/source/advanced/speculative-decoding.md)

### Dynamo

NVIDIA Dynamo is a high-throughput low-latency inference framework designed for serving generative AI and reasoning models in multi-node distributed environments.
Dynamo supports TensorRT LLM as one of its inference engine. For details on how to use TensorRT LLM with Dynamo please refer to [LLM Deployment Examples using TensorRT-LLM](https://github.com/ai-dynamo/dynamo/blob/main/examples/tensorrt_llm/README.md)

## Qwen3-Next

Below is the command to run the Qwen3-Next model.

```bash
mpirun -n 1 --allow-run-as-root --oversubscribe python3 examples/llm-api/quickstart_advanced.py --model_dir /Qwen3-Next-80B-A3B-Thinking --kv_cache_fraction 0.6 --disable_kv_cache_reuse --max_batch_size 1 --tp_size 4

```

## Notes and Troubleshooting

- **Model Directory:** Update `<YOUR_MODEL_DIR>` with the actual path where the model weights reside.
- **GPU Memory:** Adjust `--max_batch_size` and `--max_num_tokens` if you encounter out-of-memory errors.
- **Configuration Files:** Verify that the configuration files are correctly formatted to avoid runtime issues.

## Credits
This Qwen model example exists thanks to Tlntin (TlntinDeng01@gmail.com) and zhaohb (zhaohbcloud@126.com).
