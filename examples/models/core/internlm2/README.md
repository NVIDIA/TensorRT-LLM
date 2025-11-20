# InternLM2

This document shows how to build and run InternLM2 7B / 20B models in TensorRT LLM on both single GPU, single node multi-GPU and multi-node multi-GPU.

## Overview

The TensorRT LLM InternLM2 implementation is based on the LLaMA model. The implementation can
be found in [model.py](../../../../tensorrt_llm/models/llama/model.py).
The TensorRT LLM InternLM2 example code lies in [`examples/models/core/internlm2`](./):

* [`convert_checkpoint.py`](./convert_checkpoint.py) converts the Huggingface Model of InternLM2 into TensorRT LLM checkpoint.


In addition, there are two shared files in the parent folder [`examples`](../../../) for inference and evaluation:

* [`run.py`](../../../run.py) to run the inference on an input text;
* [`summarize.py`](../../../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset.

## Support Matrix
  * FP16 / BF16
  * INT8 & INT4 Weight-Only
  * Tensor Parallel

## Usage

The TensorRT LLM InternLM2 example code locates at [examples/models/core/internlm2](./). It takes HF weights as input, and builds the corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### Build TensorRT engine(s)

Please install required packages first to make sure the example uses matched `tensorrt_llm` version:

```bash
pip install -r requirements.txt
```

TensorRT LLM InternLM2 builds TensorRT engine(s) from HF checkpoint. If no checkpoint directory is specified, TensorRT LLM will build engine(s) with dummy weights.

InternLM2 has released several checkpoints of different size or capabilities under https://huggingface.co/internlm. Users can pick any one repository and follow instructions to prepare the checkpoint.

Below examples use [internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b) and [internlm2-chat-20b](https://huggingface.co/internlm/internlm2-chat-20b) and assume these repositories are cloned or linked under this directory, for example `./internlm2-chat-7b`.

Normally `trtllm-build` only requires single GPU, but if you've already got all the GPUs needed for inference, you could enable parallel building to make the engine building process faster by adding `--workers` argument. Please note that currently `--workers` feature only supports single node.

Here're some examples:

```bash
# Build a single-GPU float16 engine from HF weights.
# gpt_attention_plugin is necessary in InternLM2.
# Try use_gemm_plugin to prevent accuracy issue.
cd examples/models/core/internlm2

# Convert the InternLM2 7B model using a single GPU and FP16.
python convert_checkpoint.py --model_dir ./internlm2-chat-7b/ \
                --dtype float16 \
                --output_dir ./internlm2-chat-7b/trt_engines/fp16/1-gpu/
# Note: setting `--dtype bfloat16` to use bfloat16 precision.

# BUild the InternLM2 7B model using a single GPU
trtllm-build --checkpoint_dir ./internlm2-chat-7b/trt_engines/fp16/1-gpu/ \
             --output_dir ./engine_outputs \
             --gemm_plugin float16

# Convert the InternLM2 7B model using a single GPU and apply INT8 weight-only quantization..
python convert_checkpoint.py --model_dir ./internlm2-chat-7b/ \
                --dtype float16 \
                --output_dir ./internlm2-chat-7b/trt_engines/int8/1-gpu/ \
                --use_weight_only \
                --weight_only_precision int8

trtllm-build --checkpoint_dir ./internlm2-chat-7b/trt_engines/int8/1-gpu/ \
             --output_dir ./engine_outputs \
             --gemm_plugin float16

# Note: setting `--weight_only_precision int4` to use INT4 weight-only quantization

# Build InternLM2 7B using 2-way tensor parallelism.
python convert_checkpoint.py --model_dir ./internlm2-chat-7b/ \
                --dtype float16 \
                --output_dir ./internlm2-chat-7b/trt_engines/fp16/2-gpu/ \
                --tp_size 2

trtllm-build --checkpoint_dir ./internlm2-chat-7b/trt_engines/fp16/2-gpu/ \
             --output_dir ./engine_outputs \
             --gemm_plugin float16

# Build InternLM2 20B using 2-way tensor parallelism.
python convert_checkpoint.py --model_dir ./internlm2-chat-20b/ \
                --dtype bfloat16 \
                --output_dir ./internlm2-chat-20b/trt_engines/bf16/2-gpu/ \
                --tp_size 2 --workers 2

trtllm-build --checkpoint_dir ./internlm2-chat-7b/trt_engines/bf16/2-gpu/ \
             --output_dir ./engine_outputs \
             --gpt_attention_plugin bfloat16  \
             --gemm_plugin bfloat16
```

#### INT8 weight only

Examples:

```bash
cd examples/models/core/internlm2

# For 7B models
python convert_checkpoint.py --model_dir ./internlm2-chat-7b  \
                             --output_dir ./internlm2-chat-7b/w8a16/ \
                             --dtype float16  \
                             --use_weight_only \
                             --weight_only_precision int8

# Build 7B model with both INT8 weight-only
trtllm-build --checkpoint_dir ./internlm2-chat-7b/w8a16 \
             --output_dir ./engine_outputs \
             --gemm_plugin float16
```


```bash
cd examples/models/core/internlm2

# For 20B models
python convert_checkpoint.py --model_dir ./internlm2-chat-20b  \
                            --output_dir ./internlm2-chat-20b/w8a16 \
                             --dtype float16  \
                             --use_weight_only \
                             --weight_only_precision int8

# Build 20B model with both INT8 weight-only
trtllm-build --checkpoint_dir ./internlm2-chat-20b/w8a16 \
              --output_dir ./engine_outputs \
              --gemm_plugin float16 \
```

### Run

To run a TensorRT LLM InternLM2 model using the engines generated by `trtllm-build`

```bash
# InternLM2 7B with fp16
python ../../../run.py --max_output_len=120 \
                 --input_text 'Tell me about yourself.' \
                 --tokenizer_dir ./internlm2-chat-7b/ \
                 --engine_dir=./internlm2-chat-7b/trt_engines/fp16/1-gpu/

# InternLM2 7B with bf16
python ../../../run.py --max_output_len=120 \
                 --input_text 'Tell me about yourself.' \
                 --tokenizer_dir ./internlm2-chat-7b/ \
                 --engine_dir=./internlm2-chat-7b/trt_engines/bf16/1-gpu/

# InternLM2 7B with int8 weight only quantization
python ../../../run.py --max_output_len=120 \
                 --input_text 'Tell me about yourself.' \
                 --tokenizer_dir ./internlm2-chat-7b/ \
                 --engine_dir=./internlm2-chat-7b/trt_engines/weight_only/1-gpu/

# InternLM2 7B with fp16 and tensor parallelism
mpirun -n 2 --allow-run-as-root \
    python ../../../run.py --max_output_len=120 \
                     --input_text 'Tell me about yourself.' \
                     --tokenizer_dir ./internlm2-chat-7b/ \
                     --engine_dir=./internlm2-chat-7b/trt_engines/fp16/2-gpu/

# InternLM2 20B with fp16 and tensor parallelism and pipeline parallelism
mpirun -n 4 --allow-run-as-root \
    python ../../../run.py --max_output_len=120 \
                     --input_text 'Tell me about yourself.' \
                     --tokenizer_dir ./internlm2-chat-7b/ \
                     --engine_dir=./internlm2-chat-7b/trt_engines/bf16/4-gpu/
```

### Summarization using the InternLM2 model

```bash
# Run summarization using the InternLM2 7B model in FP16.
python ../../../summarize.py --test_trt_llm --test_hf \
                       --hf_model_dir ./internlm2-chat-7b/ \
                       --data_type fp16 \
                       --engine_dir ./engine_outputs

# Run summarization using the InternLM2 7B model quantized to w8a16.
python ../../../summarize.py --test_trt_llm --test_hf \
                       --hf_model_dir ./internlm2-chat-7b/ \
                       --data_type fp16 \
                       --engine_dir ./engine_outputs

# Run summarization using the InternLM2 7B model in FP16 using two GPUs.
mpirun -n 2 --allow-run-as-root \
    python ../../../summarize.py --test_trt_llm --test_hf \
                           --hf_model_dir ./internlm2-chat-7b/ \
                           --data_type fp16 \
                           --engine_dir ./internlm2-chat-7b/trt_engines/fp16/2-gpu/

# Run summarization using the InternLM2 20B model in BF16 using 4 GPUs.
mpirun -n 4 --allow-run-as-root \
    python ../../../summarize.py --test_trt_llm --test_hf \
                           --hf_model_dir ./internlm2-chat-20b/ \
                           --data_type bf16 \
                           --engine_dir ./internlm2-chat-20b/trt_engines/bf16/4-gpu/
```
