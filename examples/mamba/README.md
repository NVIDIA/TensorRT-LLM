# Mamba

This document shows how to build and run a [Mamba](https://github.com/state-spaces/mamba) model in TensorRT-LLM on a single GPU.

## Overview

The TensorRT-LLM Mamba implementation can be found in [`tensorrt_llm/models/mamba/model.py`](../../tensorrt_llm/models/mamba/model.py). The TensorRT-LLM Mamba example code is located in [`examples/mamba`](./). There is one main file:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert a checkpoint from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers) format to the TensorRT-LLM format.

In addition, there are two shared files in the parent folder [`examples`](../) for inference and evaluation:

* [`../run.py`](../run.py) to run the inference on an input text;
* [`../summarize.py`](../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset.


## Support Matrix
  * FP16
  * BF16

## Usage

The next two sections describe how to convert the weights from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers)
format to the TensorRT-LLM format.

### 1. Download weights from HuggingFace Transformers

Install the dependency packages and setup `git-lfs`.

```bash
# Install dependencies
pip install -r requirements.txt

# Setup git-lfs
git lfs install
```

There are six HF checkpoints available. Use one of the following commands to fetch the checkpoint you are interested in.

```bash
# mamba-2.8b-slimpj
git clone https://huggingface.co/state-spaces/mamba-2.8b-slimpj

# mamba-2.8b
git clone https://huggingface.co/state-spaces/mamba-2.8b

# mamba-1.4b
git clone https://huggingface.co/state-spaces/mamba-1.4b

# mamba-790m
git clone https://huggingface.co/state-spaces/mamba-790m

# mamba-370m
git clone https://huggingface.co/state-spaces/mamba-370m

# mamba-130m
git clone https://huggingface.co/state-spaces/mamba-130m
```

Since mamba models use tokenizer from gpt-neox-20b model, use the following command to fetch the checkpoint of gpt-neox-20b.

```bash
# gpt-neox-20b
git clone https://huggingface.co/EleutherAI/gpt-neox-20b
```

### 2. Convert weights from HF Transformers to TensorRT-LLM format
The [`convert_checkpoint.py`](./convert_checkpoint.py) script converts HF weights to TensorRT-LLM checkpoints.

```bash
# mamba-2.8b-slimpj
python convert_checkpoint.py --model_dir ./mamba-2.8b-slimpj/ \
                             --dtype bfloat16 \
                             --output_dir ./mamba/mamba-2.8b-slimpj/trt_ckpt/bf16/1-gpu/

# mamba-2.8b
python convert_checkpoint.py --model_dir ./mamba-2.8b/ \
                             --dtype bfloat16 \
                             --output_dir ./mamba/mamba-2.8b/trt_ckpt/bf16/1-gpu/

# mamba-1.4b
python convert_checkpoint.py --model_dir ./mamba-1.4b/ \
                             --dtype float16 \
                             --output_dir ./mamba/mamba-1.4b/trt_ckpt/fp16/1-gpu/

# mamba-790m
python convert_checkpoint.py --model_dir ./mamba-790m/ \
                             --dtype float16 \
                             --output_dir ./mamba/mamba-790m/trt_ckpt/fp16/1-gpu/

# mamba-370m
python convert_checkpoint.py --model_dir ./mamba-370m/ \
                             --dtype float16 \
                             --output_dir ./mamba/mamba-370m/trt_ckpt/fp16/1-gpu/

# mamba-130m
python convert_checkpoint.py --model_dir ./mamba-130m/ \
                             --dtype float16 \
                             --output_dir ./mamba/mamba-130m/trt_ckpt/fp16/1-gpu/
```

### 3. Build TensorRT engine(s)
The `trtllm-build` command builds TensorRT-LLM engines from TensorRT-LLM checkpoints.

```bash
# mamba-2.8b-slimpj
trtllm-build --checkpoint_dir ./mamba/mamba-2.8b-slimpj/trt_ckpt/bf16/1-gpu/ \
             --use_gemm_plugin bfloat16 \
             --use_selective_scan_plugin bfloat16 \
             --max_batch_size 8 \
             --max_input_len 924 \
             --max_output_len 100 \
             --output_dir ./mamba/mamba-2.8b-slimpj/trt_engines/bf16/1-gpu/

# mamba-2.8b
trtllm-build --checkpoint_dir ./mamba/mamba-2.8b/trt_ckpt/bf16/1-gpu/ \
             --use_gemm_plugin bfloat16 \
             --use_selective_scan_plugin bfloat16 \
             --max_batch_size 8 \
             --max_input_len 924 \
             --max_output_len 100 \
             --output_dir ./mamba/mamba-2.8b/trt_engines/bf16/1-gpu/

# mamba-1.4b
trtllm-build --checkpoint_dir ./mamba/mamba-1.4b/trt_ckpt/fp16/1-gpu/ \
             --use_gemm_plugin float16 \
             --use_selective_scan_plugin float16 \
             --max_batch_size 8 \
             --max_input_len 924 \
             --max_output_len 100 \
             --output_dir ./mamba/mamba-1.4b/trt_engines/fp16/1-gpu/

# mamba-790m
trtllm-build --checkpoint_dir ./mamba/mamba-790m/trt_ckpt/fp16/1-gpu/ \
             --use_gemm_plugin float16 \
             --use_selective_scan_plugin float16 \
             --max_batch_size 8 \
             --max_input_len 924 \
             --max_output_len 100 \
             --output_dir ./mamba/mamba-790m/trt_engines/fp16/1-gpu/

# mamba-370m
trtllm-build --checkpoint_dir ./mamba/mamba-370m/trt_ckpt/fp16/1-gpu/ \
             --use_gemm_plugin float16 \
             --use_selective_scan_plugin float16 \
             --max_batch_size 8 \
             --max_input_len 924 \
             --max_output_len 100 \
             --output_dir ./mamba/mamba-370m/trt_engines/fp16/1-gpu/

# mamba-130m
trtllm-build --checkpoint_dir ./mamba/mamba-130m/trt_ckpt/fp16/1-gpu/ \
             --use_gemm_plugin float16 \
             --use_selective_scan_plugin float16 \
             --max_batch_size 8 \
             --max_input_len 924 \
             --max_output_len 100 \
             --output_dir ./mamba/mamba-130m/trt_engines/fp16/1-gpu/
```

### 4. Run summarization task with the TensorRT engine(s)

The following section describes how to run a TensorRT-LLM Mamba model to summarize the articles from the
[cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset. For each summary, the script can compute the
[ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)) scores and use the `ROUGE-1` score to validate the implementation.

### Run
```bash
# mamba-2.8b-slimpj
python ../summarize.py --test_trt_llm \
                       --use_py_session \
                       --hf_model_dir ./mamba-2.8b-slimpj/ \
                       --tokenizer_dir ./gpt-neox-20b/ \
                       --data_type bf16 \
                       --engine_dir ./mamba/mamba-2.8b-slimpj/trt_engines/bf16/1-gpu/

# mamba-2.8b
python ../summarize.py --test_trt_llm \
                       --use_py_session \
                       --hf_model_dir ./mamba-2.8b/ \
                       --tokenizer_dir ./gpt-neox-20b/ \
                       --data_type bf16 \
                       --engine_dir ./mamba/mamba-2.8b/trt_engines/bf16/1-gpu/

# mamba-1.4b
python ../summarize.py --test_trt_llm \
                       --use_py_session \
                       --hf_model_dir ./mamba-1.4b/ \
                       --tokenizer_dir ./gpt-neox-20b/ \
                       --data_type fp16 \
                       --engine_dir ./mamba/mamba-1.4b/trt_engines/fp16/1-gpu/

# mamba-790m
python ../summarize.py --test_trt_llm \
                       --use_py_session \
                       --hf_model_dir ./mamba-790m/ \
                       --tokenizer_dir ./gpt-neox-20b/ \
                       --data_type fp16 \
                       --engine_dir ./mamba/mamba-790m/trt_engines/fp16/1-gpu/

# mamba-370m
python ../summarize.py --test_trt_llm \
                       --use_py_session \
                       --hf_model_dir ./mamba-370m/ \
                       --tokenizer_dir ./gpt-neox-20b/ \
                       --data_type fp16 \
                       --engine_dir ./mamba/mamba-370m/trt_engines/fp16/1-gpu/

# mamba-130m
python ../summarize.py --test_trt_llm \
                       --use_py_session \
                       --hf_model_dir ./mamba-130m/ \
                       --tokenizer_dir ./gpt-neox-20b/ \
                       --data_type fp16 \
                       --engine_dir ./mamba/mamba-130m/trt_engines/fp16/1-gpu/
```
