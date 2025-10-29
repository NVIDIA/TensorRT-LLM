# Skywork

This document elaborates how to build the [Skywork](https://huggingface.co/Skywork/) model to runnable engines on single GPU node and perform a summarization task using these engines.

## Overview
The TensorRT LLM Skywork implementation is based on the LLaMA model. The implementation can
be found in [tensorrt_llm/models/llama/model.py](../../../../tensorrt_llm/models/llama/model.py).
The TensorRT LLM Skywork example code lies in [`examples/models/contrib/skywork`](./):

* [`convert_checkpoint.py`](../llama/convert_checkpoint.py) converts the Huggingface Model of Skywork into TensorRT LLM checkpoint.

In addition, there are two shared files in the parent folder [`examples`](../../../) for inference and evaluation:

* [`../../../run.py`](../../../run.py) to run the inference on an input text;
* [`../../../summarize.py`](../../../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset.

## Support Matrix
    * FP16 & BF16

## Usage

This section gives a whole process where we convert HF models, build TensorRT LLM engines and ultimately perform summarization.

### 1. Clone Code and Weights from Huggingface

To download checkpoints from HF, you need to have `git-lfs` installed in your machine:

```bash
pip install -r requirements.txt && sudo apt-get install git-lfs
```

Then clone the HF repository with:

```bash
# Skywork 13B Base Model
git clone https://huggingface.co/Skywork/Skywork-13B-base
```

### 2. Convert HF Model to TRT Checkpoint

```bash
cd examples/models/core/llama

# fp16 model
python3 convert_checkpoint.py --model_dir ./Skywork-13B-base \
                --dtype float16 \
                --output_dir ./skywork-13b-base/trt_ckpt/fp16

# bf16 model
python3 convert_checkpoint.py --model_dir ./Skywork-13B-base \
                --dtype bfloat16 \
                --output_dir ./skywork-13b-base/trt_ckpt/bf16
```

### 3. Build TensorRT Engine(s)

```bash
# fp16
trtllm-build --checkpoint_dir ./skywork-13b-base/trt_ckpt/fp16 \
                --gemm_plugin float16 \
                --gpt_attention_plugin float16 \
                --context_fmha enable \
                --max_batch_size 32 \
                --max_input_len 512 \
                --max_seq_len 1024 \
                --output_dir ./skywork-13b-base/trt_engine/fp16

# bf16
trtllm-build --checkpoint_dir ./skywork-13b-base/trt_ckpt/bf16 \
                --gemm_plugin bfloat16 \
                --gpt_attention_plugin bfloat16 \
                --context_fmha enable \
                --max_batch_size 32 \
                --max_input_len 512 \
                --max_seq_len 1024 \
                --output_dir ./skywork-13b-base/trt_engine/bf16
```

### 4. Summarization using the Engines

After building TRT engines, we can use them to perform various tasks. TensorRT LLM provides handy code to run summarization on [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset and get [ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)) scores. The `ROUGE-1` score can be used to validate model implementations.

```bash
# fp16
python ../../../summarize.py --hf_model_dir ./Skywork-13B-base \
                       --test_hf \
                       --batch_size 32 \
                       --max_input_length 512 \
                       --output_len 512 \
                       --test_trt_llm \
                       --engine_dir ./skywork-13b-base/trt_engine/fp16 \
                       --data_type fp16 \
                       --check_accuracy \
                       --tensorrt_llm_rouge1_threshold=14

# bf16
python ../../../summarize.py --hf_model_dir ./Skywork-13B-base \
                       --test_hf \
                       --batch_size 32 \
                       --max_input_length 512 \
                       --output_len 512 \
                       --test_trt_llm \
                       --engine_dir ./skywork-13b-base/trt_engine/bf16 \
                       --data_type bf16 \
                       --check_accuracy \
                       --tensorrt_llm_rouge1_threshold=14
```
