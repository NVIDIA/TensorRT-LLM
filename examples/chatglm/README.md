# ChatGLM

This document explains how to build the [ChatGLM-6B](https://huggingface.co/THUDM/chatglm-6b), [ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b), [ChatGLM2-6B-32k](https://huggingface.co/THUDM/chatglm2-6b-32k), [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b), [ChatGLM3-6B-Base](https://huggingface.co/THUDM/chatglm3-6b-base), [ChatGLM3-6B-32k](https://huggingface.co/THUDM/chatglm3-6b-32k) models using TensorRT-LLM and run on a single GPU, a single node with multiple GPUs or multiple nodes with multiple GPUs.

## Overview

The TensorRT-LLM ChatGLM implementation can be found in [`tensorrt_llm/models/chatglm/model.py`](../../tensorrt_llm/models/chatglm/model.py).
The TensorRT-LLM ChatGLM example code is located in [`examples/chatglm`](./). There is one main file:

* [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the ChatGLM model.

In addition, there are two shared files in the parent folder [`examples`](../) for inference and evaluation:

* [`../run.py`](../run.py) to run the inference on an input text;
* [`../summarize.py`](../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset.

## Support Matrix

|    Model Name    | FP16  | FMHA  |  WO   |  AWQ  |  SQ   |  TP   |  PP   |  ST   | C++ Runtime | benchmark |  IFB  |
| :--------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---------: | :-------: | :---: |
|    chatglm_6b    |   Y   |   Y   |   Y   |       |       |   Y   |       |   Y   |      Y      |     Y     |   Y   |
|   chatglm2_6b    |   Y   |   Y   |   Y   |       |       |   Y   |       |   Y   |      Y      |     Y     |   Y   |
| chatglm2-6b_32k  |   Y   |   Y   |   Y   |       |       |   Y   |       |   Y   |      Y      |     Y     |   Y   |
|   chatglm3_6b    |   Y   |   Y   |   Y   |       |       |   Y   |       |   Y   |      Y      |     Y     |   Y   |
| chatglm3_6b_base |   Y   |   Y   |   Y   |       |       |   Y   |       |   Y   |      Y      |     Y     |   Y   |
| chatglm3_6b_32k  |   Y   |   Y   |   Y   |       |       |   Y   |       |   Y   |      Y      |     Y     |   Y   |
|     glm_10b      |   Y   |   Y   |   Y   |       |       |   Y   |       |   Y   |             |           |       |

* Model Name: the name of the model, the same as the name on HuggingFace
* FMHA: Fused MultiHead Attention (see introduction below)
* WO: Weight Only Quantization (int8 / int4)
* AWQ: Activation Aware Weight Quantization
* SQ:Smooth Quantization
* TP: Tensor Parallel
* PP: Pipeline Parallel
* ST: Strongly Typed
* IFB: In-flight Batching (see introduction below)

## Usage

The next section describe how to build the engine and run the inference demo.

### 1. Download repo and weights from HuggingFace Transformers

```bash
pip install -r requirements.txt
apt-get update
apt-get install git-lfs
rm -rf chatglm*

# clone one or more models we want to build
git clone https://huggingface.co/THUDM/chatglm-6b       chatglm_6b
git clone https://huggingface.co/THUDM/chatglm2-6b      chatglm2_6b
git clone https://huggingface.co/THUDM/chatglm2-6b-32k  chatglm2_6b_32k
git clone https://huggingface.co/THUDM/chatglm3-6b      chatglm3_6b
git clone https://huggingface.co/THUDM/chatglm3-6b-base chatglm3_6b_base
git clone https://huggingface.co/THUDM/chatglm3-6b-32k  chatglm3_6b_32k
git clone https://huggingface.co/THUDM/glm-10b          glm_10b
```

### 2. Build TensorRT engine(s)

* This ChatGLM example in TensorRT-LLM builds TensorRT engine(s) using HF checkpoint directly (rather than using FT checkpoints such as GPT example).
* If no checkpoint directory is specified, TensorRT-LLM will build engine(s) using dummy weights.
* The [`build.py`](./build.py) script requires a single GPU to build the TensorRT engine(s).
* You can enable parallel builds to accelerate the engine building process if you have more than one GPU in your system (of the same model).
* For parallel building, add the `--parallel_build` argument to the build command (this feature cannot take advantage of more than a single node).
* The number of TensorRT engines depends on the number of GPUs that will be used to run inference.
* argument [--model_name/-m] is required, which can be one of "chatglm_6b", "chatglm2_6b", "chatglm2_6b_32k", "chatglm3_6b", "chatglm3_6b_base", "chatglm3_6b_32k" or "glm-10b" (use "_" rather than "-") for ChatGLM-6B, ChatGLM2-6B, ChatGLM2-6B-32K ChatGLM3-6B, ChatGLM3-6B-Base, ChatGLM3-6B-32K or GLM-10B model respectively.

#### Examples of build invocations

```bash
# Build a default engine of ChatGLM3-6B on single GPU with FP16, GPT Attention plugin, Gemm plugin, RMS Normolization plugin
python3 build.py -m chatglm3_6b --output_dir trt_engines/chatglm3_6b/fp16/1-gpu

# Build a engine on single GPU with FMHA kernels (see introduction below), other configurations are the same as default example
python3 build.py -m chatglm3_6b --enable_context_fmha --output_dir trt_engines/chatglm3_6b/fp16/1-gpu  # or --enable_context_fmha_fp32_acc

# Build a engine on single GPU with int8/int4 Weight-Only quantization, other configurations are the same as default example
python3 build.py -m chatglm3_6b --use_weight_only --output_dir trt_engines/chatglm3_6b/weight_only/1-gpu  # or --use_weight_only --weight_only_precision int4

# Build a engine on single GPU with int8_kv_cache and remove_input_padding, other configurations are the same as default example
python3 build.py -m chatglm3_6b --paged_kv_cache --remove_input_padding --output_dir trt_engines/chatglm3_6b/fp16/1-gpu

# Build a engine on two GPU, other configurations are the same as default example
python3 build.py -m chatglm3_6b --world_size 2 --output_dir trt_engines/chatglm3_6b/fp16/2-gpu

# Build a engine of Chatglm-6B on single GPU, other configurations are the same as default example
python3 build.py -m chatglm_6b --output_dir trt_engines/chatglm_6b/fp16/1-gpu

# Build a engine of Chatglm2-6B on single GPU, other configurations are the same as default example
python3 build.py -m chatglm2_6b --output_dir trt_engines/chatglm2_6b/fp16/1-gpu

# Build a engine of ChatGLM2-6B-32k on single GPU, other configurations are the same as default example
python3 build.py -m chatglm2_6b-32k --output_dir trt_engines/chatglm2_6b-32k/fp16/1-gpu

# Build a engine of ChatGLM3-6B-Base on single GPU, other configurations are the same as default example
python3 build.py -m chatglm3_6b_base --output_dir trt_engines/chatglm3_6b_base/fp16/1-gpu

# Build a engine of ChatGLM3-6B-32k on single GPU, other configurations are the same as default example
python3 build.py -m chatglm3_6b-32k --output_dir trt_engines/chatglm3_6b-32k/fp16/1-gpu

# Build a engine of GLM-10B on single GPU, other configurations are the same as default example
python3 build.py -m glm_10b --output_dir trt_engines/glm_10b/fp16/1-gpu
```

#### Enabled plugins

* Use `--use_gpt_attention_plugin <DataType>` to configure GPT Attention plugin (default as float16)
* Use `--use_gemm_plugin <DataType>` to configure GEMM plugin (default as float16)

#### Fused MultiHead Attention (FMHA)

* Use `--enable_context_fmha` or `--enable_context_fmha_fp32_acc` to enable FMHA kernels, which can provide better performance and low GPU memory occupation.

* Switch `--use_gpt_attention_plugin float16` must be used when using FMHA.

* `--enable_context_fmha` uses FP16 accumulator, which might cause low accuracy. In this case, `--enable_context_fmha_fp32_acc` should be used to protect accuracy at a cost of small performance drop.

#### Weight Only quantization

* Use `--use_weight_only` to enable INT8-Weight-Only quantization, this will siginficantly lower the latency and memory footprint.

* Furthermore, use `--weight_only_precision int8` or `--weight_only_precision int4` to configure the data type of the weights.

#### In-flight batching

* The engine must be built accordingly if [in-flight batching in C++ runtime](../../docs/in_flight_batching.md) will be used.

* Use `--use_inflight_batching` to enable In-flight Batching.

* Switch `--use_gpt_attention_plugin=float16`, `--paged_kv_cache`, `--remove_input_padding` will be set when using In-flight Batching.

* It is possible to use `--use_gpt_attention_plugin float32` In-flight Batching.

* The size of the block in paged KV cache can be conteoled additionally by using `--tokens_per_block=N`.

### 3. Run

#### Single node, single GPU

```bash
# Run the default engine of ChatGLM3-6B on single GPU, other model name is available if built.
python3 ../run.py --input_text "What's new between ChatGLM3-6B and ChatGLM2-6B?" \
                  --max_output_len 50 \
                  --tokenizer_dir chatglm3_6b \
                  --engine_dir trt_engines/chatglm3_6b/fp16/1-gpu

# Run the default engine of ChatGLM3-6B on single GPU, using streaming output, other model name is available if built.
# In this case only the first sample in the first batch is shown,
# But actually all output of all batches are available.
python3 ../run.py --input_text "What's new between ChatGLM3-6B and ChatGLM2-6B?" \
                  --max_output_len 50 \
                  --tokenizer_dir chatglm3_6b \
                  --engine_dir trt_engines/chatglm3_6b/fp16/1-gpu \
                  --streaming

# Run the default engine of GLM3-10B on single GPU, other model name is available if built.
# Token "[MASK]" or "[sMASK]" or "[gMASK]" must be included inside the prompt as the original model commanded.
python3 ../run.py --input_text "Peking University is [MASK] than Tsinghua Univercity." \
                  --max_output_len 50 \
                  --tokenizer_dir glm_10b \
                  --engine_dir trt_engines/glm_10b/fp16/1-gpu
```

#### Single node, multi GPU

```bash
# Run the Tensor Parallel 2 engine of ChatGLM3-6B on two GPU, other model name is available if built.
mpirun -n 2 \
    python ../run.py --input_text "What's new between ChatGLM3-6B and ChatGLM2-6B?" \
                     --max_output_len 50 \
                     --tokenizer_dir chatglm3_6b \
                     --engine_dir trt_engines/chatglm3_6b/fp16/1-gpu
```

* `--allow-run-as-root` might be needed if using `mpirun` as root.

#### Run comparison of performance and accuracy

```bash
# Run the summarization of ChatGLM3-6B task, other model name is available if built.
python3 ../summarize.py --test_trt_llm \
                        --hf_model_dir chatglm3_6b \
                        --engine_dir trt_engines/chatglm3_6b/fp16/1-gpu
```

## Benchmark

* The TensorRT-LLM ChatGLM benchmark is located in [benchmarks/](../../benchmarks/README.md)
