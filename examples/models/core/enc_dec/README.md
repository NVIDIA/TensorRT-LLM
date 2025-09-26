# Encoder-Decoder

This document shows how to build and run an Encoder-Decoder (Enc-Dec) model in TensorRT LLM on NVIDIA GPUs.

## Table of Contents

- [Encoder-Decoder](#encoder-decoder)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Usage](#usage)
  - [Encoder-Decoder Model Support](#encoder-decoder-model-support)
    - [Download weights from HuggingFace Transformers](#download-weights-from-huggingface-transformers)
    - [Convert and Split Weights](#convert-and-split-weights)
    - [Build TensorRT engine(s)](#build-tensorrt-engines)
    - [Run](#run)
      - [Run C++ runtime](#run-c-runtime)
      - [Run with Triton Backend](#run-with-triton-backend)
      - [Run Python runtime](#run-python-runtime)
    - [Benchmark](#benchmark)
      - [Benchmark C++ runtime](#benchmark-c-runtime)
    - [Run BART with LoRA](#run-bart-with-lora)
    - [Reminders](#reminders)
    - [Attention Scaling Factors](#attention-scaling-factors)
    - [Run FairSeq NMT (Neural Machine Translation) models](#run-fairseq-nmt-neural-machine-translation-models)
    - [FP8 Post-Training Quantization](#fp8-post-training-quantization)
      - [Get quantized checkpoint with ModelOpt](#get-quantized-checkpoint-with-modelopt)

## Overview

The TensorRT LLM Enc-Dec implementation can be found in [tensorrt_llm/models/enc_dec/model.py](../../../../tensorrt_llm/models/enc_dec/model.py). The TensorRT LLM Enc-Dec example code is located in [`examples/models/core/enc_dec`](./):

 * `trtllm-build` to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the Enc-Dec model,
 * [`run.py`](./run.py) to run the inference on an example input text.
 * Enc-Dec models can have specific implementations, such as the popular T5 family (T5, mT5, Flan-T5, ByT5), BART family (BART, mBART), and FairSeq family (WMTs). They are now merged into a single convert script:
   * [`convert_checkpoint.py`](./convert_checkpoint.py) to convert weights from HuggingFace or FairSeq format to TRT-LLM format, and split weights for multi-GPU inference,
## Usage

The TensorRT LLM Enc-Dec example code locates at [examples/models/core/enc_dec](./). It takes HuggingFace or FairSeq model name as input, and builds the corresponding TensorRT engines. On each GPU, there will be two TensorRT engines, one for Encoder and one for Decoder.

## Encoder-Decoder Model Support

The implementation is designed to support generic encoder-decoder models by abstracting the common and derivative components of different model architectures, such as:
- [T5](https://huggingface.co/docs/transformers/main/en/model_doc/t5)
- [T5v1.1](https://huggingface.co/docs/transformers/model_doc/t5v1.1) and [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5)
- [mT5](https://huggingface.co/docs/transformers/model_doc/mt5)
- [BART](https://huggingface.co/docs/transformers/model_doc/bart)
- [mBART](https://huggingface.co/docs/transformers/model_doc/mbart)
- [FairSeq NMT](https://pytorch.org/hub/pytorch_fairseq_translation/)
- [ByT5](https://huggingface.co/docs/transformers/main/en/model_doc/byt5)
- [UL2 (coming)](https://huggingface.co/docs/transformers/model_doc/ul2) and [Flan-UL2 (coming)](https://huggingface.co/docs/transformers/model_doc/flan-ul2)

It also supports full Tensor Parallelism (TP), Pipeline Parallelism (PP), and a hybrid of the two. Currently, Fused Multi-Head Attention (FMHA) is not yet enabled for T5 family due to its relative attention design.

In this example, we use T5 (`t5-small`) and Flan-T5 (`google/flan-t5-small`) to showcase TRT-LLM support on Enc-Dec models. BART models and FairSeq models can follow very similar steps by just replacing the model name.

### Download weights from HuggingFace Transformers

```bash
git clone https://huggingface.co/t5-small tmp/hf_models/t5-small
git clone https://huggingface.co/google/flan-t5-small tmp/hf_models/flan-t5-small
git clone https://huggingface.co/facebook/bart-large-cnn tmp/hf_models/bart-large-cnn
git clone https://huggingface.co/facebook/mbart-large-50-many-to-one-mmt tmp/hf_models/mbart-large-50-many-to-one-mmt
git clone https://huggingface.co/google/byt5-small tmp/hf_models/byt5-small
```

### Convert and Split Weights
The `convert_checkpoint.py` script converts weights from HuggingFace or FairSeq format to TRT-LLM format, and splits weights for multi-GPU inference. `--tp_size` specifies the number of GPUs for tensor parallelism during inference. Pipeline Parallelism size can be set with `--pp_size` for distributed inference.

The HuggingFace or Fairseq checkpoints of the enc-dec models mentioned in this Readme are all float32 precision. Use `--dtype` to set the target inference precision during the weight conversion.

After weight conversion, TensorRT LLM converted weights and model configuration will be saved under `<out_dir>/<tpX>` directory, which is the `--checkpoint_dir` input path you should give to the **next** engine building phase.

Take T5 for example:

```bash
# Example: build t5-small using 4-way tensor parallelism on a node with 8 GPUs (but only use 4 of them, for demonstration purpose), BF16, enabling beam search up to width=1.
export MODEL_NAME="t5-small" # or "flan-t5-small"
export MODEL_TYPE="t5"
export INFERENCE_PRECISION="bfloat16"
export TP_SIZE=4
export PP_SIZE=1
export WORLD_SIZE=4
export MAX_BEAM_WIDTH=1
python convert_checkpoint.py --model_type ${MODEL_TYPE} \
                --model_dir tmp/hf_models/${MODEL_NAME} \
                --output_dir tmp/trt_models/${MODEL_NAME}/${INFERENCE_PRECISION} \
                --tp_size ${TP_SIZE} \
                --pp_size ${PP_SIZE} \
                --dtype ${INFERENCE_PRECISION}
```

### Build TensorRT engine(s)

TensorRT LLM builds TensorRT engine(s) with flexible controls on different types of optimizations. Note that these are just examples to demonstrate multi-GPU inference. For small models like T5-small, single GPU is usually sufficient.

After engine building, TensorRT engines will be saved under `<out_dir>/<tpX>` directory, which is the `--engine_dir` path you should give to the next engine running phase. It is recommended to have `/<Y-gpu>` in the output path where `Y` is number of total GPU ranks in a multi-node, multi-GPU setup, because the same `Y` number GPUs could be executed with different TP (Tensor Parallelism) and PP (Pipeline Parallelism) combinations.

We should distinguish between `X` - TP size and `Y` - total number of GPU ranks:
* When `X = Y`, only TP is enabled
* When `X < Y`, both TP and PP are enabled. In such case, please make sure you have completed weight conversion step for `TP=X`.

The default value of `--max_input_len` is 1024. When building DecoderModel, specify decoder input length with `--max_input_len=1` for encoder-decoder model to start generation from decoder_start_token_id of length 1. If the start token is a single token (the default behavior of T5/BART/etc.), you should set `--max_input_len` as 1; if you want the decoder-only type of generation, set `--max_input_len` above 1 to get similar behavior as HF's `decoder_forced_input_ids`.

EncoderModel does not generate prompt. `--max_seq_len` should be the same as `--max_input_len`. `--max_seq_len` would be set as `--max_input_len` if not specified.

DecoderModel takes `--max_encoder_input_len` and `--max_input_len` as model inputs, `--max_encoder_input_len` is set to 1024 as default since `--max_input_len` is 1024 for EncoderModel.

To be noted:
1. For T5, add `--context_fmha disable`. FMHA with T5's relative attention bias is not implemented. Add `--use_implicit_relative_attention` when `--max_seq_len` is extremely large, causing decoder engine size to be too large to fit in memory. Compute relative attention on-the-fly (implicitly, without pre-computation) instead.
2. `--bert_attention_plugin`, `--gpt_attention_plugin`, `--remove_input_padding`, `--gemm_plugin` require explicit disabling and setting, or else they'll be set to default value in `trtllm-build`.

```bash
# --gpt_attention_plugin is necessary in Enc-Dec.
# Try --gemm_plugin to prevent accuracy issue.
# It is recommended to use --remove_input_padding along with --gpt_attention_plugin for better performance
trtllm-build --checkpoint_dir tmp/trt_models/${MODEL_NAME}/${INFERENCE_PRECISION}/encoder \
                --output_dir tmp/trt_engines/${MODEL_NAME}/${INFERENCE_PRECISION}/encoder \
                --paged_kv_cache disable \
                --moe_plugin disable \
                --max_beam_width ${MAX_BEAM_WIDTH} \
                --max_batch_size 8 \
                --max_input_len 1024 \
                --gemm_plugin ${INFERENCE_PRECISION} \
                --bert_attention_plugin ${INFERENCE_PRECISION} \
                --gpt_attention_plugin ${INFERENCE_PRECISION} \
                --remove_input_padding enable \
                --context_fmha disable

# For decoder, refer to the above content and set --max_input_len correctly
trtllm-build --checkpoint_dir tmp/trt_models/${MODEL_NAME}/${INFERENCE_PRECISION}/decoder \
                --output_dir tmp/trt_engines/${MODEL_NAME}/${INFERENCE_PRECISION}/decoder \
                --moe_plugin disable \
                --max_beam_width ${MAX_BEAM_WIDTH} \
                --max_batch_size 8 \
                --max_input_len 1 \
                --max_seq_len 201 \
                --max_encoder_input_len 1024 \
                --gemm_plugin ${INFERENCE_PRECISION} \
                --bert_attention_plugin ${INFERENCE_PRECISION} \
                --gpt_attention_plugin ${INFERENCE_PRECISION} \
                --remove_input_padding enable \
                --context_fmha disable

```

For BART, `--context_fmha` can be enabled. `trtllm-build` has the default setting to enable it.

```bash
# Example: build bart-large-cnn using a single GPU, FP32, running greedy search
export MODEL_NAME="bart-large-cnn" # or "mbart-large-50-many-to-one-mmt"
export MODEL_TYPE="bart"
export INFERENCE_PRECISION="float32"
export TP_SIZE=1
export PP_SIZE=1
export WORLD_SIZE=1
export MAX_BEAM_WIDTH=1
python convert_checkpoint.py --model_type ${MODEL_TYPE} \
                --model_dir tmp/hf_models/${MODEL_NAME} \
                --output_dir tmp/trt_models/${MODEL_NAME}/${INFERENCE_PRECISION} \
                --tp_size ${TP_SIZE} \
                --pp_size ${PP_SIZE} \
                --dtype ${INFERENCE_PRECISION}

# Note: non-T5 models can enable FMHA for the encoder part, for FP16/BF16, the default is enabled
trtllm-build --checkpoint_dir tmp/trt_models/${MODEL_NAME}/${INFERENCE_PRECISION}/encoder \
                --output_dir tmp/trt_engines/${MODEL_NAME}/${INFERENCE_PRECISION}/encoder \
                --paged_kv_cache disable \
                --moe_plugin disable \
                --max_beam_width ${MAX_BEAM_WIDTH} \
                --max_batch_size 8 \
                --max_input_len 1024 \
                --gemm_plugin ${INFERENCE_PRECISION} \
                --bert_attention_plugin ${INFERENCE_PRECISION} \
                --gpt_attention_plugin ${INFERENCE_PRECISION} \
                --remove_input_padding enable
                # --context_fmha disable should be removed

# Use the same command for decoder engine
trtllm-build --checkpoint_dir tmp/trt_models/${MODEL_NAME}/${INFERENCE_PRECISION}/decoder \
                --output_dir tmp/trt_engines/${MODEL_NAME}/${INFERENCE_PRECISION}/decoder \
                --moe_plugin disable \
                --max_beam_width ${MAX_BEAM_WIDTH} \
                --max_batch_size 8 \
                --max_input_len 1 \
                --max_seq_len 201 \
                --max_encoder_input_len 1024 \
                --gemm_plugin ${INFERENCE_PRECISION} \
                --bert_attention_plugin ${INFERENCE_PRECISION} \
                --gpt_attention_plugin ${INFERENCE_PRECISION} \
                --remove_input_padding enable
                # --context_fmha disable should be removed

```

### Run

Run a TensorRT LLM Enc-Dec model using the engines generated by build.py.
Note that during model deployment, only the TensorRT engine files are needed. Previously downloaded model checkpoints and converted weights can be removed.

Different types of runtime are provided for encoder-decoder models. Following an order of serving performance and good usability, we recommend:
- (NEW) Python binding of C++ runtime w/ Paged KV Cache and Inflight Batching (IFB)
- Python runtime w/ Static Batching
- (NEW) C++ runtime w/ Paged KV Cache and Inflight Batching

Please refer to the documentation for the details of [paged kv cache](../../../../docs/source/advanced/gpt-attention.md#paged-kv-cache) and [inflight batching](../../../../docs/source/advanced/gpt-attention.md#inflight-batching).

#### Run C++ runtime
**Note: to use inflight batching and paged kv cache features in C++ runtime, please make sure you have set `--paged_kv_cache enable` (which is by default enabled) in the `trtllm-build` command of the decoder. Meanwhile, if using Python runtime, it is recommended to disable this flag by `--paged_kv_cache disable` to avoid any unnecessary overhead.**

Note that for C++ runtime and Triton backend, Pipeline Parallelism (PP) is not supported yet, because PP usage is relatively rare for encoder-decoder models. If PP is really needed, it is recommended to use the Python runtime instead.

For good usability, Python binding of the C++ runtime is provided. You can use the high-level C++ `ModelRunner` under the `examples/` root folder.

```python
# Inferencing via python binding of C++ runtime with inflight batching (IFB)
python3 ../../../run.py --engine_dir tmp/trt_engines/${MODEL_NAME}/${INFERENCE_PRECISION} --tokenizer_dir tmp/hf_models/${MODEL_NAME} --max_output_len 64 --num_beams=1 --input_text "translate English to German: The house is wonderful."
```

You can specify `--kv_cache_free_gpu_memory_fraction` to control the percentage of free GPU memory to be used by KV cache (by default 0.9), and `--cross_kv_cache_fraction` to control the percentage of KV cache to be used by cross attention (by default 0.5, and rest of the KV cache will be used by self attention).

For pure C++ runtime, there is no example given yet. Please check the [`Executor`](../../../../cpp/include/tensorrt_llm/executor/executor.h) API to implement your own end-to-end workflow. It is highly recommended to leverage more encapsulated solutions such as the above C++ Python binding or [Triton backend](https://github.com/triton-inference-server/tensorrtllm_backend).

#### Run with Triton Backend
[Triton backend](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/encoder_decoder.md) contains the tutorial on how to run encoder-decoder engines with Tritonserver.

#### Run Python runtime

For pure Python runtime, you can still use the encoder-decoder specific script under `examples/models/core/enc_dec/`.

```bash
# Inferencing w/ single GPU greedy search, compare results with HuggingFace FP32
python3 run.py --engine_dir tmp/trt_engines/${MODEL_NAME}/${INFERENCE_PRECISION} --engine_name ${MODEL_NAME} --model_name tmp/hf_models/${MODEL_NAME} --max_new_token=64 --num_beams=1 --compare_hf_fp32

# Inferencing w/ 4 GPUs (4-way TP, as configured during the engine building step), greedy search, compare results with HuggingFace FP32
mpirun --allow-run-as-root -np ${WORLD_SIZE} python3 run.py --engine_dir tmp/trt_engines/${MODEL_NAME}/${INFERENCE_PRECISION} --engine_name ${MODEL_NAME} --model_name tmp/hf_models/${MODEL_NAME} --max_new_token=64 --num_beams=1 --compare_hf_fp32
```

### Benchmark

#### Benchmark C++ runtime

The tutorial for encoder-decoder C++ runtime benchmark can be found in [`benchmarks/cpp`](../../benchmarks/cpp/README.md#2-launch-c-benchmarking-inflightv1-batching)


### Run BART with LoRA

* Download the base model and lora model from HF:

```bash
git clone https://huggingface.co/facebook/bart-large-cnn tmp/hf_models/bart-large-cnn
git clone https://huggingface.co/sooolee/bart-large-cnn-samsum-lora tmp/hf_models/bart-large-cnn-samsum-lora
```

If using customize models, just put both the base model and lora model dirs into `tmp/hf_models`.

* Convert and Split Weights, setting `--hf_lora_dir`.

```bash
export INFERENCE_PRECISION="float16"
python convert_checkpoint.py --model_type bart \
                --model_dir tmp/hf_models/bart-large-cnn \
                --output_dir tmp/trt_models/bart-large-cnn/${INFERENCE_PRECISION} \
                --tp_size 1 \
                --pp_size 1 \
                --dtype ${INFERENCE_PRECISION}
```

* Build engine, setting `--use_lora_plugin`.

```bash

trtllm-build --checkpoint_dir tmp/trt_models/bart-large-cnn/${INFERENCE_PRECISION}/encoder \
                --output_dir tmp/trt_engines/bart-large-cnn/${INFERENCE_PRECISION}/encoder \
                --paged_kv_cache disable \
                --moe_plugin disable \
                --max_beam_width 1 \
                --max_batch_size 8 \
                --max_input_len 1024 \
                --gemm_plugin ${INFERENCE_PRECISION} \
                --bert_attention_plugin ${INFERENCE_PRECISION} \
                --gpt_attention_plugin ${INFERENCE_PRECISION} \
                --remove_input_padding disable \
                --lora_plugin ${INFERENCE_PRECISION} \
                --lora_dir tmp/hf_models/bart-large-cnn-samsum-lora/ \
                --lora_target_modules attn_q attn_v

trtllm-build --checkpoint_dir tmp/trt_models/bart-large-cnn/${INFERENCE_PRECISION}/decoder \
                --output_dir tmp/trt_engines/bart-large-cnn/${INFERENCE_PRECISION}/decoder \
                --moe_plugin disable \
                --max_beam_width 1 \
                --max_batch_size 8 \
                --max_input_len 1 \
                --max_seq_len 201 \
                --max_encoder_input_len 1024 \
                --gemm_plugin ${INFERENCE_PRECISION} \
                --bert_attention_plugin ${INFERENCE_PRECISION} \
                --gpt_attention_plugin ${INFERENCE_PRECISION} \
                --remove_input_padding disable \
                --lora_plugin ${INFERENCE_PRECISION} \
                --lora_dir tmp/hf_models/bart-large-cnn-samsum-lora/ \
                --lora_target_modules attn_q cross_attn_q attn_v cross_attn_v
```

* Run the engine, setting `--lora_dir` and `--lora_task_uids`. `--lora_task_uids` should be set as a list of uids which length equals to batch size. The following example is for batch size = 3:

```bash
python run.py \
        --engine_dir tmp/trt_engines/bart-large-cnn/${INFERENCE_PRECISION}/ \
        --engine_name bart-large-cnn \
        --model_name tmp/hf_models/bart-large-cnn \
        --max_new_token=64 \
        --num_beams=1 \
        --lora_dir tmp/hf_models/bart-large-cnn-samsum-lora/ \
        --lora_task_uids 0 0 0
```

* Run with multi-loRA, append `--lora_dir` with other lora directories and set `--lora_task_uids` according to the index of the lora directories. Set to "-1" to run with the base model:

```bash
python run.py \
        --engine_dir tmp/trt_engines/bart-large-cnn/${INFERENCE_PRECISION}/ \
        --engine_name bart-large-cnn \
        --model_name tmp/hf_models/bart-large-cnn \
        --max_new_token=64 \
        --num_beams=1 \
        --lora_dir tmp/hf_models/bart-large-cnn-samsum-lora/ ... \
        --lora_task_uids 0 -1 -1 0 0 -1
```

### Reminders

- Flan-T5 models have known issues regarding FP16 precision and using BF16 precision is recommended, regardless of TRT-LLM. Please stay with FP32 or BF16 precision for Flan-T5 family.
- For T5 and Flan-T5 family that have relative attention bias design, the relative attention table is split along `num_heads` dimension in Tensor Parallelism mode. Therefore, `num_heads` must be divisible by `tp_size`. Please be aware of this when setting the TP parameter.
- For mBART, models that can control output languages (e.g. [`mbart-large-50-many-to-many-mmt`](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt)) are not currently supported, as the script does not support `ForcedBOSTokenLogitsProcessor` to control output languages.

### Attention Scaling Factors

The `q_scaling` convention in the TRT-LLM plugin is defined as follows:
```
norm_factor = 1.f / (q_scaling * sqrt(head_size))
```
In the Multi-Head Attention (MHA) mechanism, the output of the `Q*K^T` product is scaled by this constant value `norm_factor` as `norm_factor * (Q*K^T)` for `softmax`. This scaling factor can be adjusted or neutralized based on the model's requirements.

Handling in Different Models:
- BART/FairSeq NMT: For the BART models, `q_scaling` is set to `1.f`. Therefore, the `norm_factor` for BART becomes `1.f / sqrt(head_size)`. TRT-LLM uses the default value `q_scaling = 1.f`. Similar to FairSeq NMT models.
- T5: For the T5 models, `q_scaling` is `1.f/sqrt(head_size)`, leading to a `norm_factor` of `1.f`. This is handled in T5 by the TRT-LLM's `get_offset_q_scaling()` function, which reads `head_size` from the T5 model configuration and sets `q_scaling = 1.f/sqrt(head_size)` to effectively offset the `norm_factor` to `1.f`.

### Run FairSeq NMT (Neural Machine Translation) models

FairSeq model download and library dependency are different from HuggingFace ones. Especially if you are following the recommended docker container setup in [README](../../README.md), it has a custom PyTorch build but FairSeq installation will force upgrade the PyTorch version. As a workaround, we skip the `torch` and `torchaudio` dependencies in FairSeq to make everything work nicely inside the TRT-LLM container.

```bash
# Download weights from HuggingFace Transformers
# Instructions from: https://github.com/facebookresearch/fairseq/blob/main/examples/translation/README.md#example-usage-cli-tools. Public model checkpoints are also listed there. Here we use WMT'14 Transformer model as an example.
mkdir -p tmp/fairseq_models && curl https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2 | tar xvjf - -C tmp/fairseq_models  --one-top-level=wmt14 --strip-components 1 --no-same-owner

# Install FairSeq dependency
# avoid base torch to be upgraded by fairseq
pushd tmp && (git clone https://github.com/facebookresearch/fairseq.git || true) && pushd fairseq && sed -i '/torch>=/d;/torchaudio>=/d' setup.py && pip install -e . && pip install sacremoses subword_nmt && popd && popd

# Convert and Split Weights, single GPU example
export TP_SIZE=1
export PP_SIZE=1
export WORLD_SIZE=1
export INFERENCE_PRECISION="float32"
python convert_checkpoint.py --model_type nmt \
                --model_dir tmp/fairseq_models/wmt14 \
                --output_dir tmp/trt_models/wmt14/${INFERENCE_PRECISION} \
                --tp_size ${TP_SIZE} \
                --pp_size ${PP_SIZE} \
                --dtype ${INFERENCE_PRECISION}

# Build TensorRT engine(s)
# Note: non-T5 models can enable FMHA for the encoder part, although only FP16/BF16 precisions are valid
trtllm-build --checkpoint_dir tmp/trt_models/wmt14/${INFERENCE_PRECISION}/encoder \
                --output_dir tmp/trt_engines/wmt14/${INFERENCE_PRECISION}/encoder \
                --paged_kv_cache disable \
                --moe_plugin disable \
                --max_beam_width 1 \
                --max_batch_size 8 \
                --max_input_len 1024 \
                --bert_attention_plugin ${INFERENCE_PRECISION} \
                --gpt_attention_plugin ${INFERENCE_PRECISION} \
                --remove_input_padding disable

trtllm-build --checkpoint_dir tmp/trt_models/wmt14/${INFERENCE_PRECISION}/decoder \
                --output_dir tmp/trt_engines/wmt14/${INFERENCE_PRECISION}/decoder \
                --moe_plugin disable \
                --max_beam_width 1 \
                --max_batch_size 8 \
                --max_input_len 1 \
                --max_seq_len 201 \
                --max_encoder_input_len 1024 \
                --bert_attention_plugin ${INFERENCE_PRECISION} \
                --gpt_attention_plugin ${INFERENCE_PRECISION} \
                --remove_input_padding disable
# Run
mpirun --allow-run-as-root -np ${WORLD_SIZE} python3 run.py --engine_dir tmp/trt_engines/wmt14/${INFERENCE_PRECISION} --engine_name wmt14 --model_name tmp/fairseq_models/wmt14/${INFERENCE_PRECISION} --max_new_token=24 --num_beams=1
```

### FP8 Post-Training Quantization

The examples below uses the NVIDIA Modelopt (AlgorithMic Model Optimization) toolkit for the model quantization process.

First make sure Modelopt toolkit `nvidia-modelopt>=0.22.1` is installed (see [examples/quantization/README.md](/examples/quantization/README.md#preparation)).

> [!NOTE]
> Modelopt 0.22.1 is not yet released.

#### Get quantized checkpoint with ModelOpt
Currently supported conversion are `bart-large-cnn` and `T5` family. For `bart`, please set `--dtype float16`; for `T5` family, please set `--dtype float32` due to known bug with apex+HF mentioned in [transformer:issue/34264](https://github.com/huggingface/transformers/issues/34264).

```bash
# Example: quantize bart-large-cnn using 4-way tensor parallelism on a node with 8 GPUs (but only use 4 of them, for demonstration purpose) into FP8 weight and convert to TRTLLM checkpoint.
export MODEL_NAME="bart-large-cnn"
export MODEL_TYPE="bart"
export INFERENCE_PRECISION="float16"
export TP_SIZE=4
export PP_SIZE=1
export WORLD_SIZE=4
export MAX_BEAM_WIDTH=1
python ../quantization/quantize.py \
                --model_dir tmp/hf_models/${MODEL_NAME} \
                --dtype ${INFERENCE_PRECISION} \
                --qformat fp8 \
                --kv_cache_dtype fp8 \
                --output_dir tmp/trt_models/${MODEL_NAME}/fp8 \
                --calib_size 512 \
                --batch_size 16 \
                --tp_size ${TP_SIZE} \
                --pp_size ${PP_SIZE}
```

The rest may follow the same command in [Build TensorRT engine(s)](#build-tensorrt-engines), with some notes:
* For `bart`, please add `--use_fp8_context_fmha enable` for fp8 context fmha support. For `t5`, context fmha is not supported due to relative attention bias.
* Please ensure `--paged_kv_cache enable` for decoder for fp8 paged kv cache.
* Please use `--gemm_plugin auto`, `--bert_attention_plugin auto`, `--gpt_attention_plugin auto` instead of setting precision to these plugins.
* Please use CPP runtime for better performance.
