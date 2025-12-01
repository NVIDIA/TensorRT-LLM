# Phi

This document explains how to build Phi-2, Phi-3 and Phi-3.5 family of models using TensorRT LLM and run on a single or multiple GPUs.
For multimodal models (Phi-3-vision-128k-instruct and Phi-3.5-vision-instruct), see `../multimodal/README.md`.

- [Overview](#overview)
- [Support Matrix](#support-matrix)
- [Usage](#usage)
  - [1. Convert weights from HF Transformers to TensorRT LLM format](#1-convert-weights-from-hf-transformers-to-tensorrt-llm-format)
  - [2. Build TensorRT engine(s)](#2-build-tensorrt-engines)
  - [3. Summarization using the Phi model](#3-summarization-using-the-phi-model)
  - [4. Quantization](#4-quantization)
  - [5. Run Phi-3 with LoRA](#5-run-phi-3-with-lora)

## Overview

The TensorRT LLM Phi implementation can be found in [`tensorrt_llm/models/phi/model.py`](../../../../tensorrt_llm/models/phi/model.py) and [`tensorrt_llm/models/phi3/model.py`](../../../../tensorrt_llm/models/phi3/model.py). The TensorRT LLM Phi example code is located in [`examples/models/core/phi`](./) with a single file:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert a checkpoint from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers) format to the TensorRT LLM format

In addition, there are two shared files in the parent folder [`examples`](../../../) for inference and evaluation:

* [`run.py`](../../../run.py) to run the inference on an input text;
* [`summarize.py`](../../../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset.

## Support Matrix

|    Model Name    | FP16  | BF16  | FP8   | INT8  | TP   |
| :--------------: | :---: | :---: | :---: | :---: | :---: |
|    Phi-2    |   Y   |   Y    |   |    | Y |
| Phi-3-mini-4k-instruct    |   Y   |   Y   | Y   | Y  |
| Phi-3-mini-128k-instruct  |   Y   |   Y   | Y   | Y  |
| Phi-3-small-8k-instruct   |   Y   |   Y   | Y   | Y  | Y |
| Phi-3-small-128k-instruct |   Y   |   Y   | Y   | Y  | Y |
| Phi-3-medium-8k-instruct  |   Y   |   Y   | Y   | Y  |
| Phi-3-medium-128k-instruct |  Y   |   Y   | Y   | Y  |
| Phi-3.5-mini-instruct     |   Y   |   Y   | Y   | Y  |
| Phi-3.5-MoE-instruct      |   Y   |   Y   | Y   | Y  | Y |
| Phi-4                     |   Y   |   Y   | Y   | Y  |

* Model Name: the name of the model, the same as the name on HuggingFace
* TP: Tensor Parallel

## Usage

### 1. Convert weights from HF Transformers to TensorRT LLM format

Please install required packages first:

```bash
pip install -r requirements.txt
```

```bash
python ./convert_checkpoint.py \
                    --model_dir /path/to/phi-model \
                    --output_dir ./phi-checkpoint \
                    --dtype float16
```

If a model supports tensor-parallelism, number of tensor parallel ranks to split the model into can be specified as `--tp_size` argument to `convert_checkpoint.py`.

For Phi-3.5-MoE-instruct model, expert parallelism can be enabled using `--moe_tp_size` and `--moe_ep_size` arguments.
The section on Parallelism Modes in `../mixtral/README.md` discusses tensor and expert parallelism for Mixture of Experts models in detail.

### 2. Build TensorRT engine(s)

TensorRT LLM builds TensorRT engine(s) using a HF checkpoint. If no checkpoint directory is specified, TensorRT LLM will build engine(s) using dummy weights.

Examples of build invocations:

```bash
# Build a float16 engine using a single GPU and HF weights.
# Enable several TensorRT LLM plugins to increase runtime performance. It also helps with build time.
trtllm-build \
    --checkpoint_dir ./phi-checkpoint \
    --output_dir ./phi-engine \
    --gemm_plugin auto \
    --max_batch_size 8 \
    --max_input_len 1024 \
    --max_seq_len 2048
```

### 3. Summarization using the Phi model

The following section describes how to run a TensorRT LLM Phi model to summarize the articles from the [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset. For each summary, the script can compute the [ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)) scores and use the `ROUGE-1` score to validate the implementation.
The script can also perform the same summarization using the HF Phi model.

As previously explained, the first step is to build the TensorRT engine as described above using HF weights. You also have to install the requirements:

```bash
pip install -r requirements.txt
```

The summarization can be done using the [`summarize.py`](../../../summarize.py) script as follows:

```bash
# Run the summarization task using a TensorRT LLM model and a single GPU.
python3 ../../../summarize.py --engine_dir ./phi-engine \
                        --hf_model_dir /path/to/phi-model \
                        --batch_size 1 \
                        --test_trt_llm \
                        --test_hf \
                        --data_type fp16 \
                        --check_accuracy \
                        --tensorrt_llm_rouge1_threshold=20

# Run the summarization task using a TensorRT LLM model and 2-way tensor parallelism.
mpirun -n 2 --allow-run-as-root                             \
python3 ../../../summarize.py --engine_dir ./phi-engine-tp2  \
                        --hf_model_dir /path/to/phi-model    \
                        --batch_size 1                      \
                        --test_hf                           \
                        --test_trt_llm                      \
                        --data_type fp16                    \
                        --check_accuracy                    \
                        --tensorrt_llm_rouge1_threshold 20
```


### 4. Quantization

All Phi-3 variants support post-training quantization to FP8 and INT8 SmoothQuant formats.

FP8 checkpoints can be built as follows:

```bash
DTYPE=bfloat16
python3 ../../../quantization/quantize.py \
       --model_dir phi3-model \
       --output_dir ./phi3-checkpoint \
       --dtype $DTYPE \
       --qformat fp8 --kv_cache_dtype fp8
```

INT8 checkpoints can be built as follows:

```bash
DTYPE=bfloat16
python3 ../../../quantization/quantize.py \
       --model_dir phi3-model \
       --output_dir ./phi3-checkpoint \
       --dtype $DTYPE \
       --qformat int8_sq --kv_cache_dtype int8
```

The commands to [build TensorRT engines](#2-build-tensorrt-engines) from quantized checkpoints
and to run [summarization test](#3-summarization-using-the-phi-model) are same as those for unquantized checkpoints.

### 5. Run Phi-3 with LoRA

TensorRT LLM supports running Phi-3-mini/small models with FP16/BF16/FP32 LoRA. In this section, we use Phi-3-mini as an example to show how to run an FP8 base model with FP16 LoRA module.

* download the base model and lora model from HF

```bash
git-lfs clone https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
git-lfs clone https://huggingface.co/sikoraaxd/Phi-3-mini-4k-instruct-ru-lora
```

* Quantize the Phi-3-mini model to fp8 from HF
```bash
BASE_PHI_3_MINI_MODEL=./Phi-3-mini-4k-instruct
python ../../../quantization/quantize.py --model_dir ${BASE_PHI_3_MINI_MODEL} \
                                   --dtype float16 \
                                   --qformat fp8 \
                                   --kv_cache_dtype fp8 \
                                   --output_dir phi3_mini_4k_instruct/trt_ckpt/fp8/1-gpu \
                                   --calib_size 512
```

* Build engine and run inference.
```bash
trtllm-build --checkpoint_dir phi3_mini_4k_instruct/trt_ckpt/fp8/1-gpu \
             --output_dir phi3_mini_4k_instruct/trt_engines/fp8_lora/1-gpu \
             --gemm_plugin auto \
             --max_batch_size 8 \
             --max_input_len 1024 \
             --max_seq_len 2048 \
             --lora_plugin auto \
             --lora_dir ./Phi-3-mini-4k-instruct-ru-lora

python ../../../run.py --engine_dir phi3_mini_4k_instruct/trt_engines/fp8_lora/1-gpu \
                 --max_output_len 500 \
                 --tokenizer_dir ./Phi-3-mini-4k-instruct-ru-lora \
                 --input_text "<|user|>\nCan you provide ways to eat combinations of bananas and dragonfruits?<|end|>\n<|assistant|>" \
                 --lora_task_uids 0 \
                 --use_py_session
```
