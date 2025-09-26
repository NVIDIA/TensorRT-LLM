# Deepseek-v1

This document shows how to build and run [deepseek-v1](https://arxiv.org/pdf/2401.06066) model in TensorRT-LLM.

- [Deepseek-v1](#deepseek-v1)
    - [Prerequisite](#prerequistie)
    - [Hardware](#hardware)
    - [Overview](#overview)
    - [Support Matrix](#support-matrix)
    - [Usage](#usage)
        - [Build TensorRT engine(s)](#build-tensorrt-engines)

## Prerequisite

First, please download Deepseek-v1 weights from HF https://huggingface.co/deepseek-ai/deepseek-moe-16b-base.

```bash
git lfs install
git clone https://huggingface.co/deepseek-ai/deepseek-moe-16b-base
```

## Hardware

The Deepseek-v1 model requires 1x80G GPU memory.

## Overview

The TensorRT LLM Deepseek-v1 implementation can be found in [tensorrt_llm/models/deepseek_v1/model.py](../../tensorrt_llm/models/deepseek_v1/model.py). The TensorRT LLM Deepseek-v1 example code is located in [`examples/models/contrib/deepseek_v1`](./). There is one main file:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert the Deepseek-v1 model into TensorRT LLM checkpoint format.

In addition, there are three shared files in the parent folder [`examples`](../../../) can be used for inference and evaluation:

* [`../../../run.py`](../../../run.py) to run the model inference output by given an input text.
* [`../../../summarize.py`](../../../summarize.py) to summarize the article from [cnn_dailmail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset, it can running the summarize from HF model and TensorRT LLM model.
* [`../../../mmlu.py`](../../../mmlu.py) to running score script from https://github.com/declare-lab/instruct-eval to compare HF model and TensorRT LLM model on the MMLU dataset.

## Support Matrix

- [x] FP16
- [x] TENSOR PARALLEL
- [x] FP8

## Usage

The TensorRT LLM Deepseek-v1 example code locates at [examples/models/contrib/deepseek_v1](./). It takes PyTorch weights as input, and builds corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### Build TensorRT engine(s)

Below is the step-by-step to run Deepseek-v1 with TensorRT-LLM.

First the checkpoint will be converted to the TensorRT LLM checkpoint format by apply [`convert_checkpoint.py`](./convert_checkpoint.py). After that, the TensorRT engine(s) can be build with TensorRT LLM checkpoint.

```bash
# Build the bfloat16 engine from Deepseek-v1 HF weights.
python convert_checkpoint.py --model_dir ./deepseek_moe_16b/ \
                            --output_dir ./trtllm_checkpoint_deepseek_v1_1gpu_bf16 \
                            --dtype bfloat16 \
                            --tp_size 1
trtllm-build --checkpoint_dir ./trtllm_checkpoint_deepseek_v1_1gpu_bf16 \
            --output_dir ./trtllm_engines/deepseek_v1/bf16/tp1 \
            --gpt_attention_plugin bfloat16 \
            --gemm_plugin bfloat16 \
            --moe_plugin bfloat16 \
```

Then, test the engine with [run.py](../../../run.py) script:

```bash
python ../../../run.py --engine_dir ./trtllm_engines/deepseek_v1/bf16/tp1 \
                --tokenizer_dir ./deepseek_moe_16b/ \
                --max_output_len 32 \
                --top_p 0 \
                --input_text "The president of the United States is person who"
```

### FP8 Quantization

The [`../../../quantization/quantize.py`](../../../quantization/quantize.py) script can be used to quantize the models and export TensorRT LLM checkpoints.

```bash
# Deepseek-v1: single gpu, fp8 quantization
python ../../../quantization/quantize.py --model_dir deepseek_moe_16b \
        --dtype float16 \
        --qformat fp8 \
        --kv_cache_dtype fp8 \
        --output_dir trt_ckpt/deepseek_moe_16b/fp8/1-gpu \
        --calib_size 512

# Deepseek-v1: single-gpu engine with fp8 quantization, GPT Attention plugin, Gemm plugin
trtllm-build --checkpoint_dir ./trt_ckpt/deepseek_moe_16b/fp8/1-gpu \
             --gemm_plugin float16 \
             --gpt_attention_plugin bfloat16 \
             --output_dir ./trt_engines/fp8/1-gpu/
```
## Credits
This Deepseek-v1 model example exists thanks to @akhoroshev(https://github.com/akhoroshev) community contribution!
