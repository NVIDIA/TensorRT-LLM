# ChatGLM-6B

This document explains how to build the
[ChatGLM-6B](https://huggingface.co/THUDM/chatglm-6b) model using TensorRT-LLM
and run on a single GPU, a single node with multiple GPUs or multiple nodes
with multiple GPUs.

## Overview

The TensorRT-LLM ChatGLM-6B implementation can be found in
[`tensorrt_llm/models/chatglm6b/model.py`](../../tensorrt_llm/models/chatglm6b/model.py).
The TensorRT-LLM ChatGLM-6B example code is located in
[`examples/chatglm6b`](./). There are four main files in that folder:

* [`hf_chatglm6b_convert.py`](./hf_chatglm6b_convert.py) to convert a
  checkpoint from the [HuggingFace (HF)
  Transformers](https://github.com/huggingface/transformers) format to the
  [FasterTransformer (FT)](https://github.com/NVIDIA/FasterTransformer) format,
* [`build.py`](./build.py) to build the
  [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the
  ChatGLM-6B model,
* [`run.py`](./run.py) to run the inference on an input text,
* [`summarize.py`](./summarize.py) to summarize the articles in the
  [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset using
  the model.

## Usage

The next two sections describe how to convert the weights from the [HuggingFace
(HF) Transformers](https://github.com/huggingface/transformers) format to the
FT format. You can skip those two sections if you already have weights in the
FT format.

Note, also, that if your weights are neither in HF Transformers nor in FT
formats, you will need to convert to the FT format. The script like
[`hf_chatglm6b_convert.py`](./hf_chatglm6b_convert.py) can serve as a starting
point.

### 1. Prepare environment and download weights from HuggingFace Transformers

```bash
pip install -r requirements.txt
apt-get update
apt-get install git-lfs
git clone https://huggingface.co/THUDM/chatglm-6b pyTorchModel
```

### 2. Convert weights from HF Tranformers to FT format

TensorRT-LLM can directly load weights from FT. The
[`hf_chatglm6b_convert.py`](./hf_chatglm6b_convert.py) script allows you to
convert weights from HF Tranformers format to FT format.

```bash
# beckup the original file
cp pyTorchModel/modeling_chatglm.py pyTorchModel/modeling_chatglm.py-backup
# replace the file with our edited version for exporting the weight of LM
cp modeling_chatglm.py pyTorchModel
# export weight of LM
python3 exportLM.py
# restore the original file for the later use (for example, summarize.py)
cp pyTorchModel/modeling_chatglm.py-backup pyTorchModel/modeling_chatglm.py

python3 hf_chatglm6b_convert.py -i pyTorchModel -o ftModel --tensor-parallelism 1 --storage-type fp16
```

### 3. Build TensorRT engine(s)

TensorRT-LLM builds TensorRT engine(s) using a checkpoint in FT format. If no
checkpoint directory is specified, TensorRT-LLM will build engine(s) using
dummy weights. Note that the number of TensorRT engines depends on the number
of GPUs that will be used to run inference.

The [`build.py`](./build.py) script requires a single GPU to build the TensorRT
engine(s). However, if you have more than one GPU in your system (of the same
model), you can enable parallel builds to accelerate the engine building
process. For that, add the `--parallel_build` argument to the build command.
Please note that for the moment, the `parallel_build` feature cannot take
advantage of more than a single node.

Examples of build invocations:

```bash
# Build a single-GPU float16 engine using FT weights.
# Enable the special TensorRT-LLM ChatGLM-6B Attention plugin (--use_gpt_attention_plugin) to increase runtime performance.
python3 build.py --model_dir=./ftModel/1-gpu \
                 --dtype float16 \
                 --use_gpt_attention_plugin float16
```

#### Fused MultiHead Attention (FMHA)

You can enable the FMHA kernels for ChatGLM-6B by adding
`--enable_context_fmha` to the invocation of `build.py`. Note that it is
disabled by default because of possible accuracy issues due to the use of Flash
Attention.

### 4. Run

#### Single node, single GPU

To run a TensorRT-LLM ChatGLM-6B model on a single GPU, you can use `python3`:

```bash
# Run the ChatGLM-6B model on a single GPU.
python3 run.py
```

The summarization can be done using the [`summarize.py`](./summarize.py) script as follows:

```bash
# Run the summarization task.
python3 summarize.py --engine_dir trtModel \
                     --test_hf \
                     --batch_size 1 \
                     --test_trt_llm \
                     --hf_model_location=pyTorchModel \
                     --check_accuracy \
                     --tensorrt_llm_rouge1_threshold=14
```

## Benchmark

The TensorRT-LLM ChatGLM-6B benchmark is located in [benchmarks/](../../benchmarks/README.md)
