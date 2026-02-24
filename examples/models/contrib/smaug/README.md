# Smaug

This document elaborates how to build the [Smaug-72B-v0.1](https://huggingface.co/abacusai/Smaug-72B-v0.1) model to runnable engines on multi-GPU node and perform a summarization task using these engines.

## Overview

The TensorRT LLM support for Smaug-72B-v0.1 is based on the LLaMA model, the implementation can be found in [tensorrt_llm/models/llama/model.py](../../../../tensorrt_llm/models/llama/model.py). Smaug model resembles LLaMA very much except it uses bias term in its attention module, we therefore reuse the [LLaMA example code](../../../llama) for Smaug,

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert the LLaMA model into TensorRT LLM checkpoint format.

In addition, there are two shared files in the parent folder [`examples`](../../../) for inference and evaluation:

* [`../../../run.py`](../../../run.py) to run the inference on an input text;
* [`../../../summarize.py`](../../../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset.

## Support Matrix

* FP16

## Usage

This section gives a whole process where we convert HF models, build TensorRT LLM engines and ultimately perform summarization.

### Build TensorRT engine(s)

Run the following commands and TRT-LLM will first transforms a HF model into its own checkpoint format, then builds a TRT engine based on the checkpoint

```bash
python ../../../llama/convert_checkpoint.py \
    --model_dir ./Smaug-72B-v0.1 \
    --output_dir ./tllm_checkpoint_8gpu_tp8 \
    --dtype float16 \
    --tp_size 8

trtllm-build --checkpoint_dir ./tllm_checkpoint_8gpu_tp8 \
    --output_dir ./Smaug_72B_tp8 \
    --gemm_plugin float16 \
    --gpt_attention_plugin float16 \
    --context_fmha=enable \
    --max_batch_size 64 \
    --remove_input_padding=enable
```

### Run Summarization

After building TRT engine, we can use it to perform various tasks. TensorRT LLM provides handy code to run summarization on [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset and get [ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)) scores. The `ROUGE-1` score can be used to validate model implementations.

```bash
mpirun -n 8 -allow-run-as-root python ../../../summarize.py \
    --hf_model_dir ../Smaug-72B-v0.1 \
    --engine_dir ./Smaug_72B_tp8 \
    --data_type fp16 \
    --test_hf \
    --hf_device_map_auto \
    --test_trt_llm
```
