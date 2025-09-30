# GPT-J

This document explains how to build the [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6b) model using TensorRT LLM and run on a single GPU.

- [GPT-J](#gpt-j)
  - [Overview](#overview)
  - [Support Matrix](#support-matrix)
  - [Usage](#usage)
    - [1. Download weights from HuggingFace (HF) Transformers](#1-download-weights-from-huggingface-hf-transformers)
    - [2. Build TensorRT engine(s)](#2-build-tensorrt-engines)
      - [FP8 Post-Training Quantization](#fp8-post-training-quantization)
      - [AWQ INT4 weight only quantization](#awq-int4-weight-only-quantization)
      - [SmoothQuant (W8A8) quantization](#smoothquant-w8a8-quantization)
      - [Fused MultiHead Attention (FMHA)](#fused-multihead-attention-fmha)
      - [INT8 KV cache](#int8-kv-cache)
    - [3. Run](#3-run)
  - [Summarization using the GPT-J model](#summarization-using-the-gpt-j-model)

## Overview

The TensorRT LLM GPT-J implementation can be found in [`tensorrt_llm/models/gptj/model.py`](../../tensorrt_llm/models/gptj/model.py). The TensorRT LLM GPT-J example
code is located in [`examples/models/contrib/gptj`](./). There is one main file:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert a checkpoint from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers) format to the TensorRT LLM format.

In addition, there are two shared files in the parent folder [`examples`](../../../) for inference and evaluation:

* [`../../../run.py`](../../../run.py) to run the inference on an input text;
* [`../../../summarize.py`](../../../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset.

## Support Matrix
  * FP16
  * INT8 & INT4 per-channel weight-only
  * FP8 (with FP8 kv cache)
  * Groupwise quantization (AWQ)
  * INT8 KV CACHE (+ AWQ/per-channel weight-only)
  * INT8 SmoothQuant

## Usage

### 1. Download weights from HuggingFace (HF) Transformers

Please install required packages first:

```bash
pip install -r requirements.txt
```

```bash
# 1. Weights & config
git clone https://huggingface.co/EleutherAI/gpt-j-6b gptj_model
pushd gptj_model && \
  rm -f pytorch_model.bin && \
  wget https://huggingface.co/EleutherAI/gpt-j-6b/resolve/main/pytorch_model.bin && \
popd

# 2. Vocab and merge table
wget https://huggingface.co/EleutherAI/gpt-j-6b/resolve/main/vocab.json
wget https://huggingface.co/EleutherAI/gpt-j-6b/resolve/main/merges.txt
```

### 2. Build TensorRT engine(s)

TensorRT LLM builds TensorRT engine(s) using a HF checkpoint. If no checkpoint directory is specified, TensorRT LLM will build engine(s) using
dummy weights.

Examples of build invocations:

```bash
# Build a float16 engine using HF weights.
python convert_checkpoint.py --model_dir ./gpt-j-6b \
                             --dtype float16 \
                             --output_dir ./trt_ckpt/gptj_fp16_tp1/
```

***For all kinds of checkpoints, they share the same trtllm-build command like:***

```bash
# Enable several TensorRT LLM plugins to increase runtime performance. It also helps with build time.
trtllm-build --checkpoint_dir ./trt_ckpt/gptj_fp16_tp1/ \
             --output_dir ./trt_engines/gptj_fp16_tp1/ \
             --gemm_plugin float16 \
             --max_batch_size=32 \
             --max_input_len=1919 \
             --max_seq_len=2047
```

INT8 weight-only

```bash
# Build an int8 weight-only engine using HF weights with TP=2
python convert_checkpoint.py --model_dir ./gpt-j-6b \
                             --dtype float16 \
                             --use_weight_only \
                             --weight_only_precision int8 \
                             --output_dir ./trt_ckpt/gptj_int8_tp2/ \
                             --tp_size 2
```
Building command is identical to the common one above.

INT4 weight-only

```bash
# Build an int4 weight only quantization engine using int4 weight only quantized weights.
python convert_checkpoint.py --model_dir ./gpt-j-6b \
                             --dtype float16 \
                             --use_weight_only \
                             --weight_only_precision int4 \
                             --output_dir ./trt_ckpt/gptj_int4wo_tp1/
```
Building command is identical to the common one above.

#### FP8 Post-Training Quantization

The examples below uses the NVIDIA Modelopt (AlgorithMic Model Optimization) toolkit for the model quantization process.

First make sure Modelopt toolkit is installed (see [examples/quantization/README.md](/examples/quantization/README.md#preparation))

One can quantize HF GPT-J weights in FP8 as follows.

```bash
# Quantize HF GPT-J 6B checkpoint into FP8 format
python ../../../quantization/quantize.py --model_dir ./gpt-j-6b \
                                   --dtype float16 \
                                   --qformat fp8 \
                                   --kv_cache_dtype fp8 \
                                   --output_dir ./trt_ckpt/gptj_fp8_tp1 \
                                   --calib_size 512
```
Building command is identical to the common one above.
Note that you can enable fp8 context fmha to get further acceleration by setting `--use_fp8_context_fmha enable` when building the engines.

#### AWQ INT4 weight only quantization

One can enable AWQ INT4 weight only quantization like the following command.

```bash
# Enable AWQ int4 group-wise weight-only quantization.
python ../../../quantization/quantize.py --model_dir ./gpt-j-6b \
                                   --dtype float16 \
                                   --qformat int4_awq \
                                   --output_dir ./trt_ckpt/gptj_int4_awq_tp1 \
                                   --calib_size 512
```

```bash
# Enable AWQ int4 group-wise weight-only quantization with tp2.
python ../../../quantization/quantize.py --model_dir ./gpt-j-6b \
                                   --dtype float16 \
                                   --qformat int4_awq \
                                   --tp_size 2 \
                                   --output_dir ./trt_ckpt/gptj_int4_awq_tp2 \
                                   --calib_size 512
```

And build command is identical to the common one above.

#### SmoothQuant (W8A8) quantization

One can enable smoothquant W8A8 (weight per-channel, activation per-tensor) quantization like the following command.

```bash
# Enable smoothquant (W8A8 kernel).
python ../../../quantization/quantize.py --model_dir ./gpt-j-6b \
                                   --dtype float16 \
                                   --qformat int8_sq \
                                   --output_dir ./trt_ckpt/gptj_sq_tp1 \
                                   --calib_size 512
```
Building command is identical to the common one above.

#### Fused MultiHead Attention (FMHA)

You can enable the FMHA kernels for GPT by adding `--context_fmha` to the invocation of `trtllm-build`. Note that it is enabled by default.

If you find that the default fp16 accumulation (`--context_fmha`) cannot meet the requirement, you can try to enable fp32 accumulation by adding `--enable_context_fmha_fp32_acc` to the inference command (`run.py` or `summarize.py`). However, it is expected to see performance drop.

Note `--context_fmha` has to be used together with `--gpt_attention_plugin float16`.

#### INT8 KV cache
INT8 KV cache could be enabled to reduce memory footprint. It will bring more performance gains when batch size gets larger.

You can get the INT8 scale of KV cache through Modelopt:

```bash
# INT8 calibration
python ../../../quantization/quantize.py --model_dir ./gpt-j-6b \
                                   --dtype float16 \
                                   --kv_cache_dtype int8 \
                                   --output_dir ./trt_ckpt/gptj_fp16_int8kv_tp1 \
                                   --calib_size 512
```

And build command is identical to the common one above.

**INT8 KV cache + per-channel weight-only quantization**

For example, you can enable INT8 KV cache together with per-channel INT8/INT4 weight-only quantization like the following command.

```bash
# Enable INT8 KV cache together with per-channel INT8 weight-only quantization
python ../../../quantization/quantize.py --model_dir ./gpt-j-6b \
                                   --dtype float16 \
                                   --qformat int4_wo \
                                   --kv_cache_dtype int8 \
                                   --output_dir ./trt_ckpt/gptj_int4wo_int8kv_tp1 \
                                   --calib_size 512
```

Building command is identical to the common one above.

**INT8 KV cache + AWQ**

In addition, you can enable INT8 KV cache together with AWQ (per-group INT4 weight-only quantization)like the following command.

```bash
# Enable INT8 KV cache together with group-wise 4bit AWQ quantization
python ../../../quantization/quantize.py --model_dir ./gpt-j-6b \
                                   --dtype float16 \
                                   --qformat int4_awq \
                                   --kv_cache_dtype int8 \
                                   --output_dir ./trt_ckpt/gptj_int4awq_int8kv_tp1 \
                                   --calib_size 512
```

Building command is identical to the common one above.


### 3. Run


To run a TensorRT LLM GPT-J model:

```bash
python3 ../../../run.py --max_output_len=50 --engine_dir=gptj_engine --tokenizer_dir=gptj_model
```

## Summarization using the GPT-J model

The following section describes how to run a TensorRT LLM GPT-J model to summarize the articles from the
[cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset. For each summary, the script can compute the
[ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)) scores and use the `ROUGE-1` score to validate the implementation.
The script can also perform the same summarization using the HF GPT-J model.

As previously explained, the first step is to build the TensorRT engine as described above using HF weights. You also have to install the requirements:

```bash
pip install -r requirements.txt
```

The summarization can be done using the [`../../../summarize.py`](../../../summarize.py) script as follows:

```bash
# Run the summarization task.
python3 ../../../summarize.py --engine_dir ./trt_engines/gptj_fp16_tp1 \
                        --hf_model_dir ./gpt-j-6b \
                        --test_hf \
                        --batch_size 1 \
                        --test_trt_llm \
                        --tensorrt_llm_rouge1_threshold 14 \
                        --data_type fp16 \
                        --check_accuracy
```
