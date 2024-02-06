# Phi

This document explains how to build the [Phi](https://huggingface.co/microsoft/phi-2) model using TensorRT-LLM and run on a single GPU.

## Overview

The TensorRT-LLM Phi implementation can be found in [`tensorrt_llm/models/phi/model.py`](../../tensorrt_llm/models/phi/model.py). The TensorRT-LLM Phi example code is located in [`examples/phi`](./). There is one file:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert a checkpoint from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers) format to the TensorRT-LLM format

In addition, there are two shared files in the parent folder [`examples`](../) for inference and evaluation:

* [`../run.py`](../run.py) to run the inference on an input text;
* [`../summarize.py`](../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset.

## Support Matrix
  * FP16
  * BF16
  * Tensor Parallel

## Usage

### 1. Convert weights from HF Transformers to TensorRT-LLM format

```bash
python ./convert_checkpoint.py --model_dir "microsoft/phi-2" --output_dir ./phi-2-checkpoint --dtype float16
```

### 2. Build TensorRT engine(s)

TensorRT-LLM builds TensorRT engine(s) using a HF checkpoint. If no checkpoint directory is specified, TensorRT-LLM will build engine(s) using dummy weights.

Examples of build invocations:

```bash
# Build a float16 engine using a single GPU and HF weights.
# Enable several TensorRT-LLM plugins to increase runtime performance. It also helps with build time.
# workers == tp_size
trtllm-build \
    --checkpoint_dir ./phi-2-checkpoint \
    --output_dir ./phi-2-engine \
    --gemm_plugin float16 \
    --max_batch_size 8 \
    --max_input_len 1024 \
    --max_output_len 1024 \
    --workers 1
```

#### Fused MultiHead Attention (FMHA)

You can enable the FMHA kernels for phi by adding `--context_fmha enable` to the invocation of `trtllm-build`. Note that it is disabled by default because of possible accuracy issues due to the use of Flash Attention.

If you find that the default fp16 accumulation (`--context_fmha enable`) cannot meet the requirement, you can try to enable fp32 accumulation by adding `--context_fmha_fp32_acc enable`. However, it is expected to see performance drop.

Note `--context_fmha enable` / `--context_fmha_fp32_acc enable` has to be used together with `--gpt_attention_plugin float16`.

### 3. Summarization using the Phi model

The following section describes how to run a TensorRT-LLM Phi model to summarize the articles from the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset. For each summary, the script can compute the [ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)) scores and use the `ROUGE-1` score to validate the implementation.
The script can also perform the same summarization using the HF Phi model.

As previously explained, the first step is to build the TensorRT engine as described above using HF weights. You also have to install the requirements:

```bash
pip install -r requirements.txt
```

The summarization can be done using the [`../summarize.py`](../summarize.py) script as follows:

```bash
# Run the summarization task using a TensorRT-LLM model and a single GPU.
python3 ../summarize.py --engine_dir ./phi-2-engine \
                        --hf_model_dir "microsoft/phi-2" \
                        --batch_size 1 \
                        --test_trt_llm \
                        --test_hf \
                        --data_type fp16 \
                        --check_accuracy \
                        --tensorrt_llm_rouge1_threshold=20

# Run the summarization task using a TensorRT-LLM model and 2-way tensor parallelism.
mpirun -n 2 --allow-run-as-root                             \
python3 ../summarize.py --engine_dir ./phi-2-engine-tp2  \
                        --hf_model_dir "microsoft/phi-2"    \
                        --batch_size 1                      \
                        --test_hf                           \
                        --test_trt_llm                      \
                        --data_type fp16                    \
                        --check_accuracy                    \
                        --tensorrt_llm_rouge1_threshold 20
```
