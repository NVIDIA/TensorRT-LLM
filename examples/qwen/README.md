# Qwen

This document shows how to build and run a [Qwen](https://huggingface.co/Qwen) model in TensorRT-LLM on both single GPU, single node multi-GPU.

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
  - [Credits](#credits)

## Overview

The TensorRT-LLM Qwen implementation can be found in [models/qwen](../../tensorrt_llm/models/qwen/). The TensorRT-LLM Qwen example code is located in [`examples/qwen`](./). There is one main file:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the Qwen model.

In addition, there are two shared files in the parent folder [`examples`](../) for inference and evaluation:

* [`../run.py`](../run.py) to run the inference on an input text;
* [`../summarize.py`](../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset.

## Support Matrix
|   Model Name       | FP16/BF16  |  FP8  |  WO   |  AWQ  | GPTQ  |  SQ   |  TP   |  PP   |  Arch   |
| :-------------:    |   :---:    | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :-----: |
| Qwen-1_8B(-Chat)   |     Y      |   Y   |   Y   |   Y*  |   Y   |   Y   |   Y   |   Y   | Ampere+ |
| Qwen-7B(-Chat)     |     Y      |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   | Ampere+ |
| Qwen-14B(-Chat)    |     Y      |   Y   |   Y   |   Y*  |   Y   |   Y   |   Y   |   Y   | Ampere+ |
| Qwen-72B(-Chat)    |     Y      |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   | Ampere+ |
| Qwen1.5-0.5B(-Chat)|     Y      |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   | Ampere+ |
| Qwen1.5-1.8B(-Chat)|     Y      |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   | Ampere+ |
| Qwen1.5-4B(-Chat)  |     Y      |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   | Ampere+ |
| Qwen1.5-7B(-Chat)  |     Y      |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   | Ampere+ |
| Qwen1.5-14B(-Chat) |     Y      |   Y   |   Y   |   Y*  |   Y   |   Y   |   Y   |   Y   | Ampere+ |
| Qwen1.5-32B(-Chat) |     Y      |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   | Ampere+ |
| Qwen1.5-72B(-Chat) |     Y      |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   | Ampere+ |
| Qwen1.5-110B(-Chat)|     Y      |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   | Ampere+ |
| Qwen1.5-MoE-A2.7B(-Chat)|   Y   |   -   |   Y   |   -   |   -   |   -   |   Y   |   Y   | Ampere+ |
| Qwen2-0.5B(-Instruct)|     Y    |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   | Ampere+ |
| Qwen2-1.5B(-Instruct)|     Y    |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   | Ampere+ |
| Qwen2-7B(-Instruct)|     Y      |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   | Ampere+ |
| Qwen2-57B-A14B(-Instruct)|  Y   |   -   |   Y   |   -   |   -   |   -   |   Y   |   Y   | Ampere+ |
| Qwen2-72B(-Instruct)|     Y     |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   | Ampere+ |

*Please note that these models supports AWQ only with single GPU.

* Model Name: the name of the model, the same as the name on HuggingFace
* WO: Weight Only Quantization (int8 / int4)
* AWQ: Activation Aware Weight Quantization (int4)
* GPTQ: Generative Pretrained Transformer Quantization (int4)
* SQ: Smooth Quantization (int8)
* TP: Tensor Parallel
* PP: Pipeline Parallel

*Currently Qwen models does not support dynamic NTK and logn attention. Therefore, accuracy on long sequence input for the 7B and 14B model is not promised.

## Usage

The TensorRT-LLM Qwen example code locates at [examples/qwen](./). It takes HF weights as input, and builds the corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### Download model weights

Install the dependency packages and setup `git-lfs`.

```bash
# Install dependencies
pip install -r requirements.txt

# Setup git-lfs
git lfs install
```

Download one or more Qwen models that you would like to build to TensorRT-LLM engines. You may download from the [HuggingFace](https://huggingface.co) hub:

```bash
git clone https://huggingface.co/Qwen/Qwen-7B-Chat   ./tmp/Qwen/7B
git clone https://huggingface.co/Qwen/Qwen-14B-Chat  ./tmp/Qwen/14B
git clone https://huggingface.co/Qwen/Qwen-72B-Chat  ./tmp/Qwen/72B
```

### Build TensorRT engine(s)

The [`convert_checkpoint.py`](./convert_checkpoint.py) script converts HF weights to TensorRT-LLM checkpoints.

The `trtllm-build` command builds TensorRT-LLM engines from TensorRT-LLM checkpoints. The number of engine files is also same to the number of GPUs used to run inference.

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

The smoothquant supports Qwen models. Unlike the FP16 build where the HF weights are processed and loaded into the TensorRT-LLM directly, the SmoothQuant needs to load INT8 weights which should be pre-processed before building an engine.

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
python ../quantization/quantize.py --model_dir ./tmp/Qwen/7B/ \
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
You may find the official GPTQ quantized INT4 weights of Qwen-7B-Chat here: [Qwen-7B-Chat-Int4](https://huggingface.co/Qwen/Qwen-7B-Chat-Int4). And you need to first install auto-gptq:
```bash
pip install auto-gptq
```

Example of building engine for INT4 GPTQ quantized Qwen model:
```bash
python3 convert_checkpoint.py --model_dir ./tmp/Qwen-7B-Chat-Int4 \
                              --output_dir ./tllm_checkpoint_1gpu_gptq \
                              --dtype float16 \
                              --use_weight_only \
                              --weight_only_precision int4_gptq \
                              --per_group \

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
    python ../quantization/quantize.py --model_dir ./tmp/Qwen/7B/ \
                                       --dtype float16 \
                                       --qformat int4_awq \
                                       --awq_block_size 128 \
                                       --output_dir ./quantized_int4-awq \
                                       --calib_size 32
    ```

2. Build TRT-LLM engine:

    ```bash
    trtllm-build --checkpoint_dir ./quantized_int4-awq \
                 --output_dir ./tmp/qwen/7B/trt_engines/int4_AWQ/1-gpu/ \
                 --gemm_plugin float16
    ```

### Run

To run a TensorRT-LLM Qwen model using the engines generated by `trtllm-build`

```bash
# With fp16 inference
python3 ../run.py --input_text "你好，请问你叫什么？" \
                  --max_output_len=50 \
                  --tokenizer_dir ./tmp/Qwen/7B/ \
                  --engine_dir=./tmp/Qwen/7B/trt_engines/fp16/1-gpu/

# With bf16 inference
python3 ../run.py --input_text "你好，请问你叫什么？" \
                  --max_output_len=50 \
                  --tokenizer_dir ./tmp/Qwen/7B/ \
                  --engine_dir=./tmp/Qwen/7B/trt_engines/bf16/1-gpu

# With int8 weight only inference
python3 ../run.py --input_text "你好，请问你叫什么？" \
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
python3 ../run.py --input_text "你好，请问你叫什么？" \
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
python3 ../run.py --input_text "你好，请问你叫什么？" \
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
python3 ../run.py --input_text "你好，请问你叫什么？" \
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
    python ../run.py --input_text "What is your name?" \
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
python ../run.py --engine_dir ./tmp/qwen/7B_lora/trt_engines/fp16/1-gpu \
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
python ../run.py --engine_dir ./tmp/qwen/7B_lora/trt_engines/fp16/1-gpu \
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
python ../summarize.py --test_trt_llm \
                       --hf_model_dir ./tmp/Qwen/7B/ \
                       --data_type fp16 \
                       --engine_dir ./tmp/Qwen/7B/trt_engines/fp16/1-gpu/ \
                       --max_input_length 2048 \
                       --output_len 2048

# Run summarization using the Qwen 7B model in BF16.
python ../summarize.py --test_trt_llm \
                       --hf_model_dir ./tmp/Qwen/7B/ \
                       --data_type fp16 \
                       --engine_dir ./tmp/Qwen/7B/trt_engines/bf16/1-gpu/ \
                       --max_input_length 2048 \
                       --output_len 2048

# Run summarization using the Qwen 7B model quantized to INT8.
python ../summarize.py --test_trt_llm \
                       --hf_model_dir  ./tmp/Qwen/7B/ \
                       --data_type fp16 \
                       --engine_dir ./tmp/Qwen/7B/trt_engines/int8_weight_only/1-gpu/ \
                       --max_input_length 2048 \
                       --output_len 2048

# Run summarization using the Qwen 7B model quantized to INT4.
python ../summarize.py --test_trt_llm \
                       --hf_model_dir  ./tmp/Qwen/7B/ \
                       --data_type fp16 \
                       --engine_dir ./tmp/Qwen/7B/trt_engines/int4_weight_only/1-gpu/ \
                       --max_input_length 2048 \
                       --output_len 2048

# Run summarization using the Qwen 7B model in FP16 using two GPUs.
mpirun -n 2 --allow-run-as-root \
    python ../summarize.py --test_trt_llm \
                           --hf_model_dir  ./tmp/Qwen/7B/ \
                           --data_type fp16 \
                           --engine_dir ./tmp/Qwen/7B/trt_engines/fp16/2-gpu/ \
                           --max_input_length 2048 \
                           --output_len 2048

# Run summarization using the Qwen 14B model in FP16 using two GPUs.
mpirun -n 2 --allow-run-as-root \
    python ../summarize.py --test_trt_llm \
                           --hf_model_dir  ./tmp/Qwen/14B/ \
                           --data_type fp16 \
                           --engine_dir ./tmp/Qwen/14B/trt_engines/fp16/2-gpu/ \
                           --max_input_length 2048 \
                           --output_len 2048
```
**Demo output of summarize.py:**
```bash
python ../summarize.py --test_trt_llm \
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
[11/09/2023-02:24:06] [TRT-LLM] [I] TensorRT-LLM (total latency: 30.13867211341858 sec)
[11/09/2023-02:24:06] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[11/09/2023-02:24:06] [TRT-LLM] [I]   rouge1 : 26.35215119137573
[11/09/2023-02:24:06] [TRT-LLM] [I]   rouge2 : 9.507814774384485
[11/09/2023-02:24:06] [TRT-LLM] [I]   rougeL : 18.171982659482865
[11/09/2023-02:24:06] [TRT-LLM] [I]   rougeLsum : 21.10413175647868
```

## Credits
This Qwen model example exists thanks to Tlntin (TlntinDeng01@gmail.com) and zhaohb (zhaohbcloud@126.com).
