# Jais

This document elaborates how to build Jais model to runnable engines on multi-GPU node and perform a summarization task using these engines.

Currently it has been tested on
- [Jais-13b-chat](https://huggingface.co/core42/jais-13b-chat)
- [Jais-30b-chat-v3](https://huggingface.co/core42/jais-30b-chat-v3)


- [Jais](#jais)
  - [Overview](#overview)
  - [Support Matrix](#support-matrix)
  - [Usage](#usage)
    - [Build TensorRT engine(s)](#build-tensorrt-engines)
    - [Run inference](#run)

## Overview

The TensorRT LLM support for Jais is based on the GPT model, the implementation can be found in [tensorrt_llm/models/gpt/model.py](../../../../tensorrt_llm/models/gpt/model.py). Jais model resembles GPT very much except it uses alibi embedding, embedding scale, swiglu, and logits scale, we therefore reuse the [GPT example code](../../../gpt) for Jais,

* [`convert_checkpoint.py`](../../../gpt/convert_checkpoint.py) to convert the Jais model into TensorRT LLM checkpoint format.

In addition, there are two shared files in the parent folder [`examples`](../) for inference and evaluation:

* [`../../../run.py`](../../../run.py) to run the inference on an input text;
* [`../../../summarize.py`](../../../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset.

## Support Matrix
The tested configurations are:
  * FP16
  * FP8
  * Inflight Batching
  * Tensor Parallel

## Usage

This section gives a whole process where we convert HF models, build TensorRT LLM engines and ultimately perform summarization.

### Build TensorRT engine(s)

Run the following commands and TRT-LLM will first transforms a HF model into its own checkpoint format, then builds a TRT engine based on the checkpoint

```bash
# single gpu, dtype float16 for jais-13b-chat
python3 ../../../gpt/convert_checkpoint.py --model_dir core42/jais-13b-chat \
        --dtype float16 \
        --output_dir jais-13b-chat/trt_ckpt/fp16/1-gpu

# 2-way tensor parallelism for jais-30b-chat-v3
python3 ../../../gpt/convert_checkpoint.py --model_dir core42/jais-30b-chat-v3 \
        --dtype float16 \
        --tp_size 2 \
        --output_dir jais-30b-chat-v3/trt_ckpt/fp16/2-gpu
```

```bash
# Build a single-GPU float16 engine from TensorRT LLM checkpoint for jais-13b-chat
# Enable the special TensorRT LLM GPT Attention plugin (--gpt_attention_plugin) to increase runtime performance.
# It is recommend to use --remove_input_padding along with --gpt_attention_plugin for better performance
trtllm-build --checkpoint_dir jais-13b-chat/trt_ckpt/fp16/1-gpu \
        --gpt_attention_plugin float16 \
        --remove_input_padding enable \
        --output_dir jais-13b-chat/trt_engines/fp16/1-gpu

# Build 2-way tensor parallelism engines from TensorRT LLM checkpoint for jais-30b-chat-v3
trtllm-build --checkpoint_dir jais-30b-chat-v3/trt_ckpt/fp16/2-gpu \
        --gpt_attention_plugin float16 \
        --remove_input_padding enable \
        --output_dir jais-30b-chat-v3/trt_engines/fp16/2-gpu
```


### Run

The [`../../../run.py`](../../../run.py) script can be used to run inference with the built engine(s).

```bash
python3 ../../../run.py --engine_dir jais-13b-chat/trt_engines/fp16/1-gpu \
        --tokenizer_dir core42/jais-13b-chat \
        --max_output_len 10
```

If the engines are run successfully, you will see output like:
```
......
Input [Text 0]: "Born in north-east France, Soyer trained as a"
Output [Text 0 Beam 0]: " chef in Paris before moving to England in 1816"
```

```bash
python3 ../../../run.py --engine_dir jais-13b-chat/trt_engines/fp16/1-gpu \
        --tokenizer_dir core42/jais-13b-chat \
        --max_output_len 8 \
        --input_text "ولد في 1304 ميلادياً ابن بطوطه, لقد ذهب"
```

If the engines are run successfully, you will see output like:
```
.....
Input [Text 0]: "ولد في 1304 ميلادياً ابن بطوطه, لقد ذهب"
Output [Text 0 Beam 0]: " في جميع أنحاء العالم المعروف في ذلك الوقت"
```


To run a 2 TP model you can do the following
```bash
mpirun -np 2 \
    python3 ../../../run.py --engine_dir jais-30b-chat-v3/trt_engines/fp16/2-gpu \
        --tokenizer_dir core42/jais-30b-chat-v3 \
        --max_output_len 30
```

If the engines are run successfully, you will see output like:
```
Input [Text 0]: "Born in north-east France, Soyer trained as a"
Output [Text 0 Beam 0]: " chef, working in a series of high-end establishments.

Soyer's career took him to work in a number of establishments across Europe,"
```
