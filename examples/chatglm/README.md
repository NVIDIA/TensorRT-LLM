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

|    Model Name    | FP16  | FMHA  |  WO   |  AWQ  |  SQ   |  TP   |  PP   |  ST   |  C++  | benchmark |  IFB  |
| :--------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :-------: | :---: |
|    chatglm_6b    |   Y   |       |   Y   |       |       |   Y   |       |   Y   |   Y   |     Y     |   Y   |
|   chatglm2_6b    |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |       |   Y   |   Y   |     Y     |   Y   |
| chatglm2_6b_32k  |   Y   |   Y   |   Y   |   Y   |       |   Y   |       |   Y   |   Y   |     Y     |   Y   |
|   chatglm3_6b    |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |       |   Y   |   Y   |     Y     |   Y   |
| chatglm3_6b_base |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |       |   Y   |   Y   |     Y     |   Y   |
| chatglm3_6b_32k  |   Y   |   Y   |   Y   |   Y   |   Y   |   Y   |       |   Y   |   Y   |     Y     |   Y   |
|     glm_10b      |   Y   |   Y   |   Y   |       |       |   Y   |       |   Y   |       |           |       |

* Model Name: the name of the model, the same as the name on HuggingFace
* FMHA: Fused MultiHead Attention (see introduction below)
* WO: Weight Only Quantization (int8 / int4)
* AWQ: Activation Aware Weight Quantization (int4)
* SQ: Smooth Quantization (int8)
* TP: Tensor Parallel
* PP: Pipeline Parallel
* ST: Strongly Typed
* C++: C++ Runtime
* benchmark: benchmark by python / C++ Runtime
* IFB: In-flight Batching (see introduction below)

## Model comparison

|       Name       |  nL   |  nAH  |  nKH  |  nHW  |  nH   |  nF   | nMSL  |   nV   | bP2D  | bBQKV | bBDense | Comments                                                           |
| :--------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :----: | :---: | :---: | :-----: | :----------------------------------------------------------------- |
|    chatglm_6b    |  28   |  32   |  32   |  128  | 4096  | 16384 | 2048  | 130528 |   Y   |   Y   |    Y    |                                                                    |
|   chatglm2_6b    |  28   |  32   |   2   |  128  | 4096  | 13696 | 32768 | 65024  |   N   |   Y   |    N    | Multi_query_attention, RMSNorm rather than LayerNorm in chatglm_6b |
| chatglm2_6b_32k  |  28   |  32   |   2   |  128  | 4096  | 13696 | 32768 | 65024  |   N   |   Y   |    N    | RoPE base=160000 rather than 10000 in chatglm2_6b                  |
|   chatglm3_6b    |  28   |  32   |   2   |  128  | 4096  | 13696 | 8192  | 65024  |   N   |   Y   |    N    | Different in preprocess and postprocess than chatglm2_6b           |
| chatglm3_6b_base |  28   |  32   |   2   |  128  | 4096  | 13696 | 32768 | 65024  |   N   |   Y   |    N    |                                                                    |
| chatglm3_6b_32k  |  28   |  32   |   2   |  128  | 4096  | 13696 | 32768 | 65024  |   N   |   Y   |    N    | RoPE base=500000 rather than 10000 in chatglm3_6b                  |
|     glm_10b      |  48   |  64   |  32   |  64   | 4096  | 16384 | 1024  | 50304  |   Y   |   Y   |    Y    |                                                                    |

* nL: number of layers
* nAH: number of attention heads
* nKH: number of kv heads (less than nAH if multi_query_attention is used)
* nHW: head width
* nH: hidden size
* nF: FFN hidden size
* nMSL: max sequence length (input + output)
* nV: vocabulary size
* bP2D: use position_encoding_2d (Y: Yes, N: No)
* bBQKV: use bias for QKV multiplication in self-attention
* bBDense: use bias for Dense multiplication in self-attention

## Tokenizer and special tokens comparison

|       Name       |    Tokenizer     |  bos   |  eos   |  pad  |  cls  | startofpiece | endofpiece |  mask  | smask | gmask  |
| :--------------: | :--------------: | :----: | :----: | :---: | :---: | :----------: | :--------: | :----: | :---: | :----: |
|    chatglm_6b    | ChatGLMTokenizer | 130004 | 130005 |   3   |       |    130004    |   130005   | 130000 |       | 130001 |
|   chatglm2_6b    | ChatGLMTokenizer |   1    |   2    |   0   |       |              |            |        |       |        |
| chatglm2_6b_32k  | ChatGLMTokenizer |   1    |   2    |   0   |       |              |            |        |       |        |
|   chatglm3_6b    | ChatGLMTokenizer |   1    |   2    |   0   |       |              |            | 130000 |       |        |
| chatglm2_6b_base | ChatGLMTokenizer |   1    |   2    |   0   |       |              |            | 130000 |       |        |
| chatglm2_6b_32k  | ChatGLMTokenizer |   1    |   2    |   0   |       |              |            | 130000 |       |        |
|     glm_10b      | GLMGPT2Tokenizer | 50257  | 50256  | 50256 | 50259 |    50257     |   50258    | 50260  | 50264 | 50263  |

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

# replace tokenizationfile if using transformers-4.36.1 for model ChatGLM-6B (this might be needless in the future)
cp chatglm_6b/tokenization_chatglm.py chatglm_6b/tokenization_chatglm.py-backup
cp tokenization_chatglm.py chatglm_6b
```

### 2. Build TensorRT engine(s)

* This ChatGLM example in TensorRT-LLM builds TensorRT engine(s) using HF checkpoint directly (rather than using FT checkpoints such as GPT example).
* If no checkpoint directory is specified, TensorRT-LLM will build engine(s) using dummy weights.
* The [`build.py`](./build.py) script requires a single GPU to build the TensorRT engine(s).
* You can enable parallel builds to accelerate the engine building process if you have more than one GPU in your system (of the same model).
* For parallel building, add the `--parallel_build` argument to the build command (this feature cannot take advantage of more than a single node).
* The number of TensorRT engines depends on the number of GPUs that will be used to run inference.
* model name can be one of "chatglm_6b", "chatglm2_6b", "chatglm2_6b_32k", "chatglm3_6b", "chatglm3_6b_base", "chatglm3_6b_32k" or "glm-10b" (use "_" rather than "-") for ChatGLM-6B, ChatGLM2-6B, ChatGLM2-6B-32K ChatGLM3-6B, ChatGLM3-6B-Base, ChatGLM3-6B-32K or GLM-10B model respectively.

* Using ChatGLM2-6B-32K / ChatGLM3-6B-32K models, we need to guarantee `max_batch_size * max_beam_width * (max_input_len + max_output_len) <= 78398 = 2^31 / (13696 * 2)` due to constrain of TensorRT. For example, we will fail to build engine while using default max_batch_size (8) and adding arguments `--max_beam_width=4 --max_input_len=20000 --max_output_len=100`.

#### Examples of build invocations

```bash
# Build a default engine of ChatGLM3-6B on single GPU with FP16, GPT Attention plugin, Gemm plugin, RMS Normolization plugin
python3 build.py --model_dir chatglm3_6b --output_dir trt_engines/chatglm3_6b/fp16/1-gpu

# Build a engine on single GPU with FMHA kernels (see introduction below), other configurations are the same as default example
python3 build.py --model_dir chatglm3_6b --enable_context_fmha --output_dir trt_engines/chatglm3_6b/fp16/1-gpu

# Build a engine on single GPU with int8/int4 Weight-Only quantization, other configurations are the same as default example
python3 build.py --model_dir chatglm3_6b --use_weight_only --output_dir trt_engines/chatglm3_6b/weight_only/1-gpu

# Build a engine on single GPU with In-flight-Batching supported, other configurations are the same as default example
python3 build.py --model_dir chatglm3_6b --paged_kv_cache --remove_input_padding --use_inflight_batching --output_dir trt_engines/chatglm3_6b/fp16/1-gpu

# Build a engine on two GPU, other configurations are the same as default example
python3 build.py --model_dir chatglm3_6b --world_size 2 --output_dir trt_engines/chatglm3_6b/fp16/2-gpu

# Build a engine of Chatglm-6B on single GPU, other configurations are the same as default example
python3 build.py --model_dir chatglm_6b --output_dir trt_engines/chatglm_6b/fp16/1-gpu

# Build a engine of Chatglm2-6B on single GPU, other configurations are the same as default example
python3 build.py --model_dir chatglm2_6b --output_dir trt_engines/chatglm2_6b/fp16/1-gpu

# Build a engine of ChatGLM2-6B-32k on single GPU, other configurations are the same as default example
python3 build.py --model_dir chatglm2_6b_32k --output_dir trt_engines/chatglm2_6b_32k/fp16/1-gpu

# Build a engine of ChatGLM3-6B-Base on single GPU, other configurations are the same as default example
python3 build.py --model_dir chatglm3_6b_base --output_dir trt_engines/chatglm3_6b_base/fp16/1-gpu

# Build a engine of ChatGLM3-6B-32k on single GPU, other configurations are the same as default example
python3 build.py --model_dir chatglm3_6b_32k --output_dir trt_engines/chatglm3_6b_32k/fp16/1-gpu

# Build a engine of GLM-10B on single GPU, other configurations are the same as default example
python3 build.py --model_dir glm_10b --max_input_len=512 --output_dir trt_engines/glm_10b/fp16/1-gpu
```

#### example of output from build.py with "--log_level=info"

```txt
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I] ========================================= Build Arguments ==========================================
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - model_name..............................: chatglm2_6b
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - gpus_per_node...........................: 8
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - world_size..............................: 1
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - tp_size.................................: 1
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - pp_size.................................: 1
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - model_dir...............................: chatglm2_6b
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - quant_ckpt_path.........................: awq/gpt2_tp1_rank0.npz
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - quantized_fp8_model_path................: None
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - output_dir..............................: engine_outputs
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - dtype...................................: float16
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - logits_dtype............................: float32
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - strongly_typed..........................: False
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - timing_cache............................: model.cache
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - log_level...............................: info
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - builder_opt.............................: None
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - parallel_build..........................: False
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - enable_debug_output.....................: False
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - visualize...............................: False
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - random_seed.............................: None
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - max_batch_size..........................: 8
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - max_input_len...........................: 1024
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - max_output_len..........................: 1024
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - max_beam_width..........................: 1
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - max_num_tokens..........................: 4294967296
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - use_gpt_attention_plugin................: float16
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - use_gemm_plugin.........................: float16
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - use_layernorm_plugin....................: float16
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - use_rmsnorm_plugin......................: float16
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - enable_context_fmha.....................: False
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - enable_context_fmha_fp32_acc............: False
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - multi_block_mode........................: False
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - gather_all_token_logits.................: False
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - use_custom_all_reduce...................: False
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - remove_input_padding....................: False
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - paged_kv_cache..........................: False
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - tokens_per_block........................: 128
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - use_inflight_batching...................: False
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - use_weight_only.........................: False
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - weight_only_precision...................: int8
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - disable_weight_only_quant_plugin........: False
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - use_smooth_quant........................: False
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - per_token...............................: False
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - per_channel.............................: False
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - per_group...............................: False
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - group_size..............................: 128
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - int8_kv_cache...........................: False
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - enable_fp8..............................: False
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - fp8_kv_cache............................: False
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - apply_query_key_layer_scaling...........: False
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - apply_residual_connection_post_layernorm: False
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - ffn_hidden_size.........................: 13696
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - hidden_act..............................: swiglu
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - hidden_size.............................: 4096
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - linear_bias.............................: False
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - max_seq_length..........................: 2048
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - multi_query_mode........................: False
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - norm_epsilon............................: 1e-05
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - num_heads...............................: 32
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - num_kv_heads............................: 2
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - num_layers..............................: 28
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - qkv_bias................................: True
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - rmsnorm.................................: True
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - rotary_embedding_scaling................: 1.0
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - use_cache...............................: True
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - vocab_size..............................: 65024
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I]  - quant_mode..............................: 0
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I] ====================================================================================================

... # more TensorRT building log

[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I] Activation memory size: 2062.03 MiB
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I] Weights memory size: 11909.60 MiB
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I] Max KV Cache memory size: 448.00 MiB
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I] Estimated max memory usage on runtime: 14419.63 MiB
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I] Serializing engine to engine_outputs/chatglm2_6b_float16_tp1_rank0.engine...
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I] Engine serialized. Total time: 00:00:05
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I] [MemUsage] Rank 0 Engine serialized - Allocated Memory: Host 0.3352 (GiB) Device 3.5739 (GiB)
[XX/XX/XXXX-XX:XX;XX] [TRT] [I] Serialized 59 bytes of code generator cache.
[XX/XX/XXXX-XX:XX;XX] [TRT] [I] Serialized 86850 bytes of compilation cache.
[XX/XX/XXXX-XX:XX;XX] [TRT] [I] Serialized 1531 timing cache entries
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I] Timing cache serialized to model.cache
[XX/XX/XXXX-XX:XX;XX] [TRT-LLM] [I] Total time of building all 1 engines: 00:01:14
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

#### Activation-aware Weight Quantization (AWQ)

* Use quantize.py to enable INT4-AWQ quantization

```bash
# Build INT4-AWQ weights file
python examples/quantization/quantize.py --model_dir=chatglm3_6b \
                                         --dtype=float16 \
                                         --qformat=int4_awq \
                                         --export_path=awq.pt \
                                         --calib_size=32 \
                                         --cache_dir=./dataset

# Build INT4-AWQ TRT-LLM engine
python build.py \
    --model_dir=chatglm3_6b \
    --quant_ckpt_path=awq.pt \
    --use_weight_only \
    --weight_only_precision=int4_awq \
    --per_group
```

#### Smooth Quantization (SQ)

* Use hf_chatglm_convert.py to enable smooth quantization

```bash
# Get  smooth quantization weights file
python3 convert_chatglm.py \
    -i chatglm3_6b/ \
    -o sq \
    -sq=0.5 \
    --cache_dir=dataset/ \
    --calib_size=64

# Build smooth quantization TRT-LLM engine
python build.py \
    -m chatglm3_6b \
    --model_dir=sq/1-gpu/ \
    --use_smooth_quant \
    --per_token \
    --per_channel
```

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
python3 ../run.py --input_text "What's new between ChatGLM3-6B and ChatGLM2-6B?" \
                  --max_output_len 50 \
                  --tokenizer_dir chatglm3_6b \
                  --engine_dir trt_engines/chatglm3_6b/fp16/1-gpu \
                  --streaming

# Run the default engine of GLM3-10B on single GPU, other model name is available if built.
# Token "[MASK]" or "[sMASK]" or "[gMASK]" must be included in the prompt as the original model commanded.
python3 ../run.py --input_text "Peking University is [MASK] than Tsinghua University." \
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

#### Example of output from run.py (might not be token-wise same on various environments)

```txt
Input [Text 0]: "Born in north-east France, Soyer trained as a"
Output [Text 0 Beam 0]: "lawyer and later became a prominent figure in the French Resistance during World War II. He was captured by the Gestapo in 1942 and sentenced to death, but was later reprieved and released. Soyer later became a member of the French National Assembly and served as a member of the European Parliament. He was also a prominent figure in the anti-Vichy movement and played a key role in the Resistance during the German occupation of France."
```

#### Run comparison of performance and accuracy

```bash
# Run the summarization of ChatGLM3-6B task, other model name is available if built.
python3 ../summarize.py \
    --test_trt_llm \
    --check_accuracy \
    --hf_model_dir chatglm3_6b \
    --engine_dir engine_outputs/chatglm3_6b/fp10/1-gpu \
```

#### Example of output from summarize.py (might not be actually same on various environments)

```txt
[12/26/2023-20:05:51] [TRT-LLM] [I] Load engine takes: 4.738909959793091 sec
[12/26/2023-20:06:11] [TRT-LLM] [I] ---------------------------------------------------------
[12/26/2023-20:06:11] [TRT-LLM] [I] TensorRT-LLM Generated :
[12/26/2023-20:06:11] [TRT-LLM] [I]  Input : ['(CNN)The Palestinian Authority officially became the ... .']
[12/26/2023-20:06:11] [TRT-LLM] [I]
 Reference : ['Membership gives the ICC jurisdiction over alleged crimes committed in Palestinian ... .']
[12/26/2023-20:06:11] [TRT-LLM] [I]
 Output : [['The Palestinian Authority has officially joined the International Criminal Court... .']]
[12/26/2023-20:06:11] [TRT-LLM] [I] ---------------------------------------------------------
[12/26/2023-20:06:53] [TRT-LLM] [I] TensorRT-LLM (total latency: 37.75311541557312 sec)
[12/26/2023-20:06:53] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[12/26/2023-20:06:53] [TRT-LLM] [I]   rouge1 : 29.333249265005644
[12/26/2023-20:06:53] [TRT-LLM] [I]   rouge2 : 13.082063357001006
[12/26/2023-20:06:53] [TRT-LLM] [I]   rougeL : 21.888991441896124
[12/26/2023-20:06:53] [TRT-LLM] [I]   rougeLsum : 25.018667613572227
```

## Benchmark

* The TensorRT-LLM ChatGLM benchmark is located in [benchmarks/](../../benchmarks/README.md)
