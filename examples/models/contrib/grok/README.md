# Grok-1

This document shows how to build and run grok-1 model in TensorRT LLM on both single GPU, single node multi-GPU and multi-node multi-GPU.

- [Grok1](#Grok-1)
  - [Prerequisite](#prerequisite)
  - [Hardware](#hardware)
  - [Overview](#overview)
  - [Support Matrix](#support-matrix)
  - [Usage](#usage)
    - [Build TensorRT engine(s)](#build-tensorrt-engines)

## Prerequisite
First of all, please clone the official grok-1 code repo with below commands and install the dependencies.
```bash
git clone https://github.com/xai-org/grok-1.git /path/to/folder
```
And then downloading the weights per [instructions](https://github.com/xai-org/grok-1?tab=readme-ov-file#downloading-the-weights).

## Hardware
The grok-1 model requires a node with 8x80GB GPU memory(at least).

## Overview

The TensorRT LLM Grok-1 implementation can be found in [tensorrt_llm/models/grok/model.py](../../../../tensorrt_llm/models/grok/model.py). The TensorRT LLM Grok-1 example code is located in [`examples/models/contrib/grok`](./). There is one main file:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert the Grok-1 model into TensorRT LLM checkpoint format.

In addition, there are two shared files in the parent folder [`examples`](../../../) for inference and evaluation:

* [`../../../run.py`](../../../run.py) to run the inference on an input text;
* [`../../../summarize.py`](../../../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset.

## Support Matrix
  * INT8 Weight-Only
  * Tensor Parallel
  * STRONGLY TYPED

## Usage

The TensorRT LLM Grok-1 example code locates at [examples/models/contrib/grok](./). It takes xai weights as input, and builds the corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### Build TensorRT engine(s)

Please install required packages first to make sure the example uses matched `tensorrt_llm` version:

```bash
pip install -r requirements.txt
```

Need to prepare the Grok-1 checkpoint by following the guides here https://github.com/xai-org/grok-1.

TensorRT LLM Grok-1 builds TensorRT engine(s) from Xai's checkpoints.

Normally `trtllm-build` only requires single GPU, but if you've already got all the GPUs needed for inference, you could enable parallel building to make the engine building process faster by adding `--workers` argument. Please note that currently `workers` feature only supports single node.


Below is the step-by-step to run Grok-1 with TensorRT LLM.

```bash
# Build the bfloat16 engine from xai official weights.
python convert_checkpoint.py --model_dir ./tmp/grok-1/ \
                              --output_dir ./tllm_checkpoint_8gpus_bf16 \
                              --dtype bfloat16 \
                              --use_weight_only \
                              --tp_size 8 \
                              --workers 8

trtllm-build --checkpoint_dir ./tllm_checkpoint_8gpus_bf16 \
            --output_dir ./tmp/grok-1/trt_engines/bf16/8-gpus \
            --gpt_attention_plugin bfloat16 \
            --gemm_plugin bfloat16 \
            --moe_plugin bfloat16 \
            --paged_kv_cache enable \
            --remove_input_padding enable \
            --workers 8


# Run Grok-1 with 8 GPUs
mpirun -n 8 --allow-run-as-root \
    python ../../../run.py \
    --input_text "The answer to life the universe and everything is of course" \
    --engine_dir ./tmp/grok-1/trt_engines/bf16/8-gpus \
    --max_output_len 50 --top_p 1 --top_k 8 --temperature 0.3 \
    --vocab_file  ./tmp/grok-1/tokenizer.model
```
