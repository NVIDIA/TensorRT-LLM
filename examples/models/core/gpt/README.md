# GPT

This document explains how to build the [GPT](https://huggingface.co/gpt2) model using TensorRT LLM and run on a single GPU, a single node with multiple GPUs or multiple nodes with multiple GPUs.

- [GPT](#gpt)
  - [Overview](#overview)
  - [Support Matrix](#support-matrix)
  - [Usage](#usage)
    - [1. Download weights from HuggingFace Transformers](#1-download-weights-from-huggingface-transformers)
    - [2. Convert weights from HF Transformers to TensorRT LLM format](#2-convert-weights-from-hf-transformers-to-tensorrt-llm-format)
    - [3. Build TensorRT engine(s)](#3-build-tensorrt-engines)
      - [Fused MultiHead Attention (FMHA)](#fused-multihead-attention-fmha)
      - [In-flight batching and paged KV cache](#in-flight-batching-and-paged-kv-cache)
    - [4. Build TensorRT engine(s) with Random Weights](#4-build-tensorrt-engines-with-random-weights)
    - [5. Run inference](#5-run-inference)
      - [Single node, single GPU](#single-node-single-gpu)
      - [Single node, multiple GPUs](#single-node-multiple-gpus)
      - [Multiple nodes, multiple GPUs using Slurm](#multiple-nodes-multiple-gpus-using-slurm)
  - [Quantization](#quantization)
    - [SmoothQuant](#smoothquant)
      - [Model Transformation](#model-transformation)
      - [INT8 inference](#int8-inference)
      - [Usage](#usage-1)
    - [INT8 KV Cache](#int8-kv-cache)
    - [Weight Only Quantization](#weight-only-quantization)
    - [FP8 Quantization](#fp8-quantization)
  - [Embedding Parallelism](#embedding-parallelism)
    - [1. Embedding parallelism](#1-embedding-parallelism)
    - [2. The sharding dimension for embedding parallelism](#2-the-sharding-dimension-for-embedding-parallelism)
  - [GPT Variant - Granite(20B and 34B)](#gpt-variant---granite)
  - [GPT Variant - SantaCoder](#gpt-variant---santacoder)
  - [GPT Variant - StarCoder (v1 and v2)](#gpt-variant---starcoder-v1-and-v2)
    - [Run StarCoder2 with LoRA](#run-starcoder2-with-lora)
  - [GPT-Next](#gpt-next)
    - [Prompt-tuning](#prompt-tuning)
    - [MultiLoRA with the Nemo checkpoint](#multilora-with-the-nemo-checkpoint)

## Overview

The TensorRT LLM GPT implementation can be found in [`tensorrt_llm/models/gpt/model.py`](../../../../tensorrt_llm/models/gpt/model.py). The TensorRT LLM GPT example code is located in [`examples/models/core/gpt`](./). There is one main file:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert a checkpoint from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers) format to the TensorRT LLM format.

In addition, there are two shared files in the parent folder [`examples`](../../../) for inference and evaluation:

* [`run.py`](../../../run.py) to run the inference on an input text;
* [`summarize.py`](../../../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset.

## Support Matrix
  * FP16
  * FP8
  * Inflight Batching
  * PAGED_KV_CACHE
  * FP8 KV CACHE
  * Tensor Parallel
  * Pipeline Parallel
  * STRONGLY TYPED
  * INT8 SmoothQuant
  * INT8 weight only
  * INT4 weight only

## Usage

The next two sections describe how to convert the weights from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers)
format to the TensorRT LLM format.

### 1. Download weights from HuggingFace Transformers

Please install required packages first:

```bash
pip install -r requirements.txt
```

```bash
# Download hf gpt2 model
rm -rf gpt2 && git clone https://huggingface.co/gpt2-medium gpt2
pushd gpt2 && rm pytorch_model.bin model.safetensors && wget -q https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin && popd
```

### 2. Convert weights from HF Transformers to TensorRT LLM format
The [`convert_checkpoint.py`](./convert_checkpoint.py) script converts HF weights to TensorRT LLM checkpoints. The number of checkpoint files (in .safetensors format) is same to the number of GPUs used to run inference.

```bash
# single gpu, dtype float16
python3 convert_checkpoint.py --model_dir gpt2 \
        --dtype float16 \
        --output_dir gpt2/trt_ckpt/fp16/1-gpu

# 2-way tensor parallelism
python3 convert_checkpoint.py --model_dir gpt2 \
        --dtype float16 \
        --tp_size 2 \
        --output_dir gpt2/trt_ckpt/fp16/2-gpu

# 2-way tensor parallelism and 2-way pipeline parallelism
python3 convert_checkpoint.py --model_dir gpt2 \
        --dtype float16 \
        --tp_size 2 \
        --pp_size 2 \
        --output_dir gpt2/trt_ckpt/fp16/4-gpu
```

### 3. Build TensorRT engine(s)
The `trtllm-build` command builds TensorRT LLM engines from TensorRT LLM checkpoints. The checkpoint directory provides the model's weights and architecture configuration. The number of engine files is also same to the number of GPUs used to run inference.

`trtllm-build` command has a variety of options. In particular, the plugin-related options have two categories:
* Plugin options that requires a data type (e.g., `gpt_attention_plugin`), you can
    * explicitly specify `float16`/`bfloat16`/`float32`, so that the plugins are enabled with the specified precision;
    * implicitly specify `auto`, so that the plugins are enabled with the precision automatically inferred from model dtype (i.e., the dtype specified in weight conversion); or
    * disable the plugin by `disable`.
* Other features that requires a boolean (e.g., `context_fmha`, `paged_kv_cache`, `remove_input_padding`), you can
    * enable/disable the feature by specifying `enable`/`disable`.

The defaults have been carefully tuned for better performance. For example, `gpt_attention_plugin`, `context_fmha`, `paged_kv_cache` and `remove_input_padding` are enabled by default. See more details by `trtllm-build --help`.

Normally, the `trtllm-build` command only requires a single GPU, but you can enable parallel building by passing the number of GPUs to the `--workers` argument.

```bash
# Build a single-GPU float16 engine from TensorRT LLM checkpoint.
# gpt_attention_plugin (the special TensorRT LLM GPT Attention plugin) and remove_input_padding are enabled by default for runtime performance.
trtllm-build --checkpoint_dir gpt2/trt_ckpt/fp16/1-gpu \
        --output_dir gpt2/trt_engines/fp16/1-gpu

# Build 2-way tensor parallelism engines from TensorRT LLM checkpoint.
trtllm-build --checkpoint_dir gpt2/trt_ckpt/fp16/2-gpu \
        --output_dir gpt2/trt_engines/fp16/2-gpu

# Build 2-way tensor parallelism and 2-way pipeline parallelism engines from TensorRT LLM checkpoint.
trtllm-build --checkpoint_dir gpt2/trt_ckpt/fp16/4-gpu \
        --output_dir gpt2/trt_engines/fp16/4-gpu
```

If the engines are built successfully, you will see output like:
```
......
[03/12/2024-10:21:08] [TRT] [I] Engine generation completed in 35.9738 seconds.
[03/12/2024-10:21:08] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 212 MiB, GPU 775 MiB
[03/12/2024-10:21:08] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in building engine: CPU +0, GPU +775, now: CPU 0, GPU 775 (MiB)
[03/12/2024-10:21:09] [TRT] [I] [MemUsageStats] Peak memory usage during Engine building and serialization: CPU: 6600 MiB
[03/12/2024-10:21:09] [TRT-LLM] [I] Total time of building Unnamed Network 0: 00:00:36
[03/12/2024-10:21:09] [TRT-LLM] [I] Serializing engine to gpt2/trt_engines/fp16/1-gpu/rank0.engine...
[03/12/2024-10:21:11] [TRT-LLM] [I] Engine serialized. Total time: 00:00:02
[03/12/2024-10:21:11] [TRT-LLM] [I] Total time of building all engines: 00:00:41
```

#### Fused MultiHead Attention (FMHA)

`trtllm-build` enables FMHA kernels by default. You may disable it by adding `--context_fmha disable`.

If you find that the default fp16 accumulation cannot meet the requirement, you can try to enable fp32 accumulation by adding `--enable_context_fmha_fp32_acc` to the inference command (`run.py` or `summarize.py`). However, it is expected to see performance drop.

Note that the FMHA kernels have to be used together with `gpt_attention_plugin` enabled.

#### In-flight batching and paged KV cache

If one wants to use [in-flight batching in C++ runtime](../../docs/in_flight_batching.md), the engine(s) must be built accordingly. In-flight batching in C++ runtime works only with attention plugin, paged KV cache and with packed data. Currently, the `trtllm-build` by default enables `gpt_attention_plugin`, `paged_kv_cache` and `remove_input_padding`, so the built engine(s) can support in-flight batching (unless you explicitly disable one of these options). One can additionally control the size of the block in paged KV cache using `--tokens_per_block=N`.

### 4. Build TensorRT engine(s) with Random Weights
You can build engine(s) using random weights, which is useful for benchmarking. First, the [`../generate_checkpoint_config.py`](../generate_checkpoint_config.py) script can be used to generate a TensorRT LLM checkpoint config file:

```bash
# Generate an 8-GPU GPT-175B float16 checkpoint config file.
python3 ../../../generate_checkpoint_config.py --architecture GPTForCausalLM \
        --vocab_size 51200 \
        --hidden_size 12288 \
        --num_hidden_layers 96 \
        --num_attention_heads 96 \
        --dtype float16 \
        --tp_size 8 \
        --output_path gpt_175b/trt_ckpt/fp16/8-gpu/config.json


# Generate a 16-GPU GPT-530B float16 checkpoint config file.
python3 ../../../generate_checkpoint_config.py --architecture GPTForCausalLM \
        --vocab_size 51200 \
        --hidden_size 20480 \
        --num_hidden_layers 105 \
        --num_attention_heads 128 \
        --dtype float16 \
        --tp_size 16 \
        --output_path gpt_530b/trt_ckpt/fp16/16-gpu/config.json
```

Then, use `trtllm-build` command to build engine(s) with random weights and the model architecture specified by the generated config file.

```bash
# Build 8-GPU GPT-175B float16 engines using dummy weights, useful for performance tests.
# Enable several TensorRT LLM plugins to increase runtime performance. It also helps with build time.
trtllm-build --model_config gpt_175b/trt_ckpt/fp16/8-gpu/config.json \
        --gemm_plugin auto \
        --max_batch_size 256 \
        --output_dir gpt_175b/trt_engines/fp16/8-gpu \
        --workers 8

# Build 16-GPU GPT-530B float16 engines using dummy weights, useful for performance tests.
# Enable several TensorRT LLM plugins to increase runtime performance. It also helps with build time.
trtllm-build --model_config gpt_530b/trt_ckpt/fp16/16-gpu/config.json \
        --gemm_plugin auto \
        --max_batch_size 128 \
        --max_input_len 128 \
        --max_seq_len 148 \
        --output_dir gpt_530b/trt_engines/fp16/16-gpu \
        --workers 8
```

### 5. Run inference
#### Single node, single GPU

The [`run.py`](../../../run.py) script can be used to run inference with the built engine(s).

```bash
python3 ../../../run.py --engine_dir gpt2/trt_engines/fp16/1-gpu \
        --tokenizer_dir gpt2 \
        --max_output_len 8
```

If the engines are run successfully, you will see output like:
```
......
Input [Text 0]: "Born in north-east France, Soyer trained as a"
Output [Text 0 Beam 0]: " chef before moving to London in the early"
```

The [`summarize.py`](../../../summarize.py) script can run the built engines to summarize the articles from the [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset.
For each summary, the script can compute the
[ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)) scores and use the `ROUGE-1` score to validate the implementation.
By passing `--test_trt_llm` flag, the script will evaluate TensorRT LLM engines. You may also pass `--test_hf` flag to evaluate the HF model.

```bash
python3 ../../../summarize.py --engine_dir gpt2/trt_engines/fp16/1-gpu \
        --hf_model_dir gpt2 \
        --test_trt_llm \
        --test_hf
```
If the engines are run successfully, you will see output like:
```
......
[03/13/2024-05:43:18] [TRT-LLM] [I] TensorRT LLM (total latency: 1.520904541015625 sec)
[03/13/2024-05:43:18] [TRT-LLM] [I] TensorRT LLM (total output tokens: 0)
[03/13/2024-05:43:18] [TRT-LLM] [I] TensorRT LLM (tokens per second: 0.0)
[03/13/2024-05:43:18] [TRT-LLM] [I] TensorRT LLM beam 0 result
[03/13/2024-05:43:18] [TRT-LLM] [I]   rouge1 : 21.13474087351942
[03/13/2024-05:43:18] [TRT-LLM] [I]   rouge2 : 6.2641616526063775
[03/13/2024-05:43:18] [TRT-LLM] [I]   rougeL : 16.693574311238077
[03/13/2024-05:43:18] [TRT-LLM] [I]   rougeLsum : 18.477384201634088
[03/13/2024-05:43:18] [TRT-LLM] [I] Hugging Face (total latency: 8.76440143585205 sec)
[03/13/2024-05:43:18] [TRT-LLM] [I] HF beam 0 result
[03/13/2024-05:43:18] [TRT-LLM] [I]   rouge1 : 20.834898522466
[03/13/2024-05:43:18] [TRT-LLM] [I]   rouge2 : 5.6914719275508805
[03/13/2024-05:43:18] [TRT-LLM] [I]   rougeL : 16.297064309934132
[03/13/2024-05:43:18] [TRT-LLM] [I]   rougeLsum : 18.018627021792142
```

#### Single node, multiple GPUs

To run engines using multiple GPUs on a single node, you can use `mpirun` as:

```bash
mpirun -np 2 \
    python3 ../../../run.py --engine_dir gpt2/trt_engines/fp16/2-gpu \
        --tokenizer_dir gpt2 \
        --max_output_len 8

# Note that GPT-175B is built with random weights, so the output will also be random
mpirun -np 8 \
    python3 ../../../run.py --engine_dir gpt_175b/trt_engines/fp16/8-gpu \
        --max_output_len 8
```

#### Multiple nodes, multiple GPUs using [Slurm](https://slurm.schedmd.com)

To run engines using multiple nodes, you should use a cluster manager like `Slurm`. The following section shows how to configure TensorRT LLM to execute on two nodes using Slurm.

We start by preparing an `sbatch` script called `tensorrt_llm_run.sub`. That script contains the following code (you must replace the `<REPLACE ...>` strings with your own values):

```bash
#!/bin/bash
#SBATCH -o logs/tensorrt_llm.out
#SBATCH -e logs/tensorrt_llm.error
#SBATCH -J <REPLACE WITH YOUR JOB's NAME>
#SBATCH -A <REPLACE WITH YOUR ACCOUNT's NAME>
#SBATCH -p <REPLACE WITH YOUR PARTITION's NAME>
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:30:00

sudo nvidia-smi -lgc 1410,1410

srun --mpi=pmix \
    --container-image <image> \
    --container-mounts <path>:<path> \
    --container-workdir <path> \
    --output logs/tensorrt_llm_%t.out \
    --error logs/tensorrt_llm_%t.error \
        python3 -u ../../../run.py --engine_dir <engine_dir> --max_output_len 8
```

Then, submit the job using:

```bash
sbatch tensorrt_llm_run.sub
```

You might have to contact your cluster's administrator to help you customize the above script.


## Quantization

### SmoothQuant

This section explains how to use SmoothQuant on GPT models with TensorRT-LLM.

[SmoothQuant](https://arxiv.org/abs/2211.10438) is a post-training quantization
(PTQ) method to quantize LLM models to INT8 for faster inference. As explained
in the article, SmoothQuant modifies a model to enable INT8 quantization
without significantly altering the accuracy.

#### Model Transformation

A LLM model is made of multiple matrix-multiplication operations (or GEMMs): `Y
= XW` where `X` of shape `[n, k]`, holds the activation (produced at run-time)
and `W`, of shape `[k, m]` are the learned weights. `Y`, of shape `[n, m]`, is
the matrix product of `X` and `W`.

SmoothQuant introduces scaling along the `k` dimension by defining a vector of
strictly positive coefficients `s`. `Y = X diag(s)^{-1} diag(s) W`. We now have
`Y = X'W'` where `X' = X diag(s)^{-1}` and `W' = diag(s) W`. This
transformation is introduced so the quantization behaves better. In *normal*
models, `X` tends to be ill-conditioned: it has mostly small-magnitude
coefficients, but also some outliers that makes quantization difficult.
Conversely, the re-scaled `X'` is better suited for INT8 conversion.

In this example, we only replace Attention's QKV and MLP's FC1 GEMMs to their
Smoothquant'd version since it is sufficient to maintain the accuracy for the
GPT model. During inference, `X'` is computed by fusing the channel-wise
multiplication by `diag(s)^{-1}` with the preceding layernorm's lambda and beta
parameters. `W'` is pre-computed and doesn't need additional modification
during inference.

#### INT8 inference

The INT8 quantization scheme used in TensorRT LLM theoretically works on any
GPT model. However, Smoothquant'd models tend to produce more accurate results
with reduced precision.

INT8 inference modifies GEMMs `Y = XW` so that both `X` and `W` use INT8. The
matrix-multiplication is sped-up because of smaller weight size and fast matrix
products computation thanks to NVIDIA *Tensor Cores* operating on INT8 inputs.

During inference, X is transformed from its standard floating point (fp)
values: `X_{i8} <- X_{fp} * s_x`. This scaling puts `X` values in the INT8
range: `[-128, 127]`. Similarly, W is scaled, `W_{i8} <- W_{fp} * s_w` but that
operation is done at model export time, no need for subsequent operations at
run-time.

The optimized TensorRT LLM GEMM implementation for SmoothQuant does the integer
matrix-multiplication `Y_{i32} <- X_{i8} W_{i8}` and rescales the result to its
original range `Y_{fp} <- Y_{i32} * (s_x)^{-1} * (s_w)^{-1}`. Note that
`Y_{i32}` isn't stored in memory, the re-scaling happens in the GEMM's epilogue
and only `Y_{fp}` gets saved.

By default `s_x` and `s_w` are single-value coefficients. This is the
*per-tensor* mode. Values for `s_x` and `s_w` are static, estimated at model
export time.

TensorRT LLM also supports more elaborate modes:
 - per-channel: `s_w` is a fixed vector of size `[1, m]`. For that,
   TensorRT LLM loads the adequately scaled version of of `W_{i8}` at model
   construction time.
 - per-token: `s_x` is a vector of size `[n, 1]` determined at run-time, based
   on the per-token (a.k.a. per-row) absolute maximum of `X`.
Users can mix-and-match per-channel and per-token options. Both tend to
increase the accuracy of the model at the cost of a slightly increased latency.

#### Usage
[`convert_checkpoint.py`](./convert_checkpoint.py) features a
`--smoothquant` option. It must be set to a decimal value in `[0, 1]` and
corresponds to the `alpha` parameter in the [SmoothQuant
paper](https://arxiv.org/abs/2211.10438). Setting `--smoothquant` will smooth the model
as explained in [model transformation](#model-transformation) and export the
scaling factors needed for INT8 inference.

By default, it will run the model in the per-tensor mode, as explained in [INT8
inference](#int8-inference). You can add any combination of `--per_token` and `--per_channel` to get the corresponding behaviors.

```bash
# Per-tensor SmoothQuant
python3 convert_checkpoint.py --model_dir gpt2 \
        --dtype float16 \
        --smoothquant 0.5 \
        --output_dir gpt2/trt_ckpt/int8-sq/1-gpu

# Per-token per-channel SmoothQuant
python3 convert_checkpoint.py --model_dir gpt2 \
        --dtype float16 \
        --smoothquant 0.5 \
        --per_token \
        --per_channel \
        --output_dir gpt2/trt_ckpt/int8-sq-ptpc/1-gpu
```

Then, use `trtllm-build` to build engine(s).

```bash
# Per-tensor SmoothQuant
trtllm-build --checkpoint_dir gpt2/trt_ckpt/int8-sq/1-gpu \
        --output_dir gpt2/trt_engines/int8-sq/1-gpu

# Per-token per-channel SmoothQuant
trtllm-build --checkpoint_dir gpt2/trt_ckpt/int8-sq-ptpc/1-gpu \
        --output_dir gpt2/trt_engines/int8-sq-ptpc/1-gpu
```

Note that GPT attention plugin is required to be enabled for SmoothQuant for now.

User can also use `ModelOpt` to do INT8 quantization. Especially for gpt variant Starcoder2.
```bash
python3 example/quantization/quantize.py --model_dir starcoder2 \
        --dtype float16 \
        --qformat int8_sq \
        --output_dir starcoder2/trt_ckpt/int8-sq/
```
Then, use `trtllm-build` to build engine(s).

```bash
trtllm-build --checkpoint_dir starcoder2/trt_ckpt/int8-sq/ \
             --output_dir starcoder2/trt_engine/int8-sq/
```


### INT8 KV Cache

[`convert_checkpoint.py`](./convert_checkpoint.py) features a
`--int8_kv_cache` option. Setting `--int8_kv_cache` will calibrate the model and export the
scaling factors needed for INT8 KV cache inference.

```bash
# Int8 KV cache
python3 convert_checkpoint.py --model_dir gpt2 \
        --dtype float16 \
        --int8_kv_cache \
        --output_dir gpt2/trt_ckpt/int8kv/1-gpu

trtllm-build --checkpoint_dir gpt2/trt_ckpt/int8kv/1-gpu \
        --output_dir gpt2/trt_engines/int8kv/1-gpu
```

INT8 KV cache can be used with or without gpt attention plugin.

### Weight Only Quantization

[`convert_checkpoint.py`](./convert_checkpoint.py) features a `--use_weight_only` option that can enable weight-only quantization. You can further set the weight-only precision by passing `int8` or `int4` to the `--weight_only_precision` flag.

```bash
# Int8 weight-only quantization
python3 convert_checkpoint.py --model_dir gpt2 \
        --dtype float16 \
        --use_weight_only \
        --weight_only_precision int8 \
        --output_dir gpt2/trt_ckpt/int8-wo/1-gpu

# Int4 weight-only quantization
python3 convert_checkpoint.py --model_dir gpt2 \
        --dtype float16 \
        --use_weight_only \
        --weight_only_precision int4 \
        --output_dir gpt2/trt_ckpt/int4-wo/1-gpu
```

Then, use `trtllm-build` to build engine(s).

```bash
# Int8 weight-only quantization
trtllm-build --checkpoint_dir gpt2/trt_ckpt/int8-wo/1-gpu \
        --output_dir gpt2/trt_engines/int8-wo/1-gpu

# Int4 weight-only quantization
trtllm-build --checkpoint_dir gpt2/trt_ckpt/int4-wo/1-gpu \
        --output_dir gpt2/trt_engines/int4-wo/1-gpu
```

### FP8 Quantization

[`quantize.py`](../../../quantization/quantize.py) can do FP8 quantization and/or FP8 kv cache quantization, and export TensorRT LLM checkpoint.

```bash
# FP8 quantization with FP8 kv cache
python3 ../../../quantization/quantize.py --model_dir gpt2 \
        --dtype float16 \
        --qformat fp8 \
        --kv_cache_dtype fp8 \
        --output_dir gpt2/trt_ckpt/fp8/1-gpu

trtllm-build --checkpoint_dir gpt2/trt_ckpt/fp8/1-gpu \
        --output_dir gpt2/trt_engines/fp8/1-gpu
```

## Embedding Parallelism
Since the embedding lookup table can be several gigabytes in size. We can distribute this weight across multiple GPUs in order to reduce the memory consumption per GPU.

### 1. Embedding parallelism
To enable this feature, add the flag `--use_parallel_embedding` to `convert_checkpoint.py`.

### 2. The sharding dimension for embedding parallelism

Assume the size of embedding lookup table is (vocab\_size \* hidden\_size), we can shard it along the vocab\_size (`--embedding_sharding_dim 0`) or hidden\_size (`--embedding_sharding_dim 1`) dimension.

2.1 To shard the embedding lookup table along the hidden\_size dimension, set the flag `--use_parallel_embedding --embedding_sharding_dim 1`. Here is an example:

```bash
# 2-way tensor parallelism with embedding parallelism along hidden dimension
python3 convert_checkpoint.py --model_dir gpt2 \
        --dtype float16 \
        --tp_size 2 \
        --use_parallel_embedding \
        --embedding_sharding_dim 1 \
        --output_dir gpt2/trt_ckpt/fp16/2-gpu

trtllm-build --checkpoint_dir gpt2/trt_ckpt/fp16/2-gpu \
        --output_dir gpt2/trt_engines/fp16/2-gpu
```

2.2 To shard the embedding lookup table along the vocab\_size dimension, set the flag `--use_parallel_embedding --embedding_sharding_dim 0`. In this case, you can optionally enable the lookup plugin when building the engines.

```bash
# 2-way tensor parallelism with embedding parallelism along vocab dimension
python3 convert_checkpoint.py --model_dir gpt2 \
        --dtype float16 \
        --tp_size 2 \
        --use_parallel_embedding \
        --embedding_sharding_dim 0 \
        --output_dir gpt2/trt_ckpt/fp16/2-gpu

trtllm-build --checkpoint_dir gpt2/trt_ckpt/fp16/2-gpu \
        --output_dir gpt2/trt_engines/fp16/2-gpu
```

## GPT Variant - Granite

For Granite, the steps are similar to StarCoder.

```bash
# Download hf granite model
git clone https://huggingface.co/ibm-granite/granite-34b-code-instruct granite

# Convert to TensorRT LLM checkpoint
python3 convert_checkpoint.py --model_dir granite \
        --dtype float16 \
        --gpt_variant starcoder \
        --tp_size 4 \
        --output_dir granite/trt_ckpt/fp16/4-gpu

# Build TensorRT LLM engines
trtllm-build --checkpoint_dir granite/trt_ckpt/fp16/4-gpu \
        --gemm_plugin auto \
        --output_dir granite/trt_engines/fp16/4-gpu

# Run inference
mpirun -np 4 \
    python3 ../../../run.py --engine_dir granite/trt_engines/fp16/4-gpu \
        --tokenizer_dir granite \
        --input_text "def print_hello_world():" \
        --max_output_len 20
```

## GPT Variant - SantaCoder

The SantaCoder extends the existing GPT model with multi-query attention mechanism. The following example shows building a 4-GPU engine and running simple prompt to generate the implementation of `print_hello_world()`.

```bash
# Download hf santacoder model
git clone https://huggingface.co/bigcode/santacoder

# Convert to TensorRT LLM checkpoint
python3 convert_checkpoint.py --model_dir santacoder \
        --dtype float16 \
        --tp_size 4 \
        --output_dir santacoder/trt_ckpt/fp16/4-gpu

# Build TensorRT LLM engines
trtllm-build --checkpoint_dir santacoder/trt_ckpt/fp16/4-gpu \
        --gemm_plugin auto \
        --output_dir santacoder/trt_engines/fp16/4-gpu

# Run inference
mpirun -np 4 \
    python3 ../../../run.py --engine_dir santacoder/trt_engines/fp16/4-gpu \
        --tokenizer_dir santacoder \
        --input_text "def print_hello_world():" \
        --max_output_len 20
```


## GPT Variant - StarCoder (v1 and v2)

For StarCoder, the steps are similar to SantaCoder.

```bash
# Download hf starcoder model
git clone https://huggingface.co/bigcode/starcoder

# Convert to TensorRT LLM checkpoint
python3 convert_checkpoint.py --model_dir starcoder \
        --dtype float16 \
        --tp_size 4 \
        --output_dir starcoder/trt_ckpt/fp16/4-gpu

# Build TensorRT LLM engines
trtllm-build --checkpoint_dir starcoder/trt_ckpt/fp16/4-gpu \
        --gemm_plugin auto \
        --output_dir starcoder/trt_engines/fp16/4-gpu

# Run inference
mpirun -np 4 \
    python3 ../../../run.py --engine_dir starcoder/trt_engines/fp16/4-gpu \
        --tokenizer_dir starcoder \
        --input_text "def print_hello_world():" \
        --max_output_len 20
```

For StarCoder2, you can use almost the same steps as shown above.
 - Note that StarCoder2 hasn't been merged to the official releases of transformers package yet, so remember using the [main branch of transformers repo](https://github.com/huggingface/transformers).
 - Add `--max_attention_window_size 4096` when running with run.py or summarization, which enables the sliding window attention.
   - the sliding window size comes from the hf model [config.json](https://huggingface.co/bigcode/starcoder2-15b/blob/main/config.json#L23).

### Run StarCoder2 with LoRA

TensorRT LLM supports running StarCoder2 models with FP16/BF16/FP32 LoRA. In this section, we use starcoder2-15b as an example to show how to run an FP8 base model with FP16 LoRA module.

* download the base model and lora model from HF

```bash
git-lfs clone https://huggingface.co/bigcode/starcoder2-15b
git-lfs clone https://huggingface.co/KaQyn/peft-lora-starcoder2-15b-unity-copilot
```

* Quantize the StarCoder2 model to fp8 from HF
```bash
BASE_STARCODER2_MODEL=./starcoder2-15b
python ../../../quantization/quantize.py --model_dir ${BASE_STARCODER2_MODEL} \
                                   --dtype float16 \
                                   --qformat fp8 \
                                   --kv_cache_dtype fp8 \
                                   --output_dir starcoder2-15b/trt_ckpt/fp8/1-gpu \
                                   --calib_size 512
```

* Build engine and run inference.
```bash
trtllm-build --checkpoint_dir starcoder2-15b/trt_ckpt/fp8/1-gpu \
             --output_dir starcoder2-15b/trt_engines/fp8_lora/1-gpu \
             --gemm_plugin auto \
             --lora_plugin auto \
             --lora_dir ./peft-lora-starcoder2-15b-unity-copilot

python ../../../run.py --engine_dir starcoder2-15b/trt_engines/fp8_lora/1-gpu \
                 --max_output_len 20 \
                 --tokenizer_dir ${BASE_STARCODER2_MODEL} \
                 --input_text "def print_hello_world():" \
                 --lora_task_uids 0 \
                 --no_add_special_tokens \
                 --use_py_session
```

## GPT-Next

NVIDIA has released a GPT-like model with some architectural improvements, that you can find here: [https://huggingface.co/nvidia/GPT-2B-001](https://huggingface.co/nvidia/GPT-2B-001). This architecture is also supported by TensorRT-LLM.

Different from Huggingface's checkpoint, you should specify the NeMo checkpoint path using `--nemo_ckpt_path` for `convert_checkpoint.py`. The script also extracts the tokenizer file from the NeMo checkpoint and saves it to the TensorRT LLM checkpoint folder, which can be used in the inference scripts.

```bash
# Download NeMo checkpoint
wget https://huggingface.co/nvidia/GPT-2B-001/resolve/main/GPT-2B-001_bf16_tp1.nemo

# Convert to TensorRT LLM checkpoint
# It also extracts the tokenizer file and saves to the TensorRT LLM checkpoint folder
python3 convert_checkpoint.py --nemo_ckpt_path GPT-2B-001_bf16_tp1.nemo \
        --dtype bfloat16 \
        --output_dir gpt-next-2B/trt_ckpt/bf16/1-gpu

# Build TensorRT LLM engines
# --gpt_attention_plugin must be set for GPT-Next since Rotary positional embeddings (RoPE) is only supported by the gpt attention plugin at this time.
trtllm-build --checkpoint_dir gpt-next-2B/trt_ckpt/bf16/1-gpu \
        --output_dir gpt-next-2B/trt_engines/bf16/1-gpu

# Run inference
python3 ../../../run.py --engine_dir gpt-next-2B/trt_engines/bf16/1-gpu \
        --vocab_file gpt-next-2B/trt_ckpt/bf16/1-gpu/tokenizer.model \
        --no_add_special_tokens \
        --max_output_len 8
```

### Prompt-tuning

For efficient fine-tuning, the NeMo framework allows you to learn virtual tokens to accomplish a downstream task. For more details, please read the
NeMo documentation [here](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html).

TensorRT LLM supports inference with those virtual tokens. To enable it, pass the prompt embedding table's maximum size at build time with `--max_prompt_embedding_table_size N`. For example:

```bash
# Convert to TensorRT LLM checkpoint
python3 convert_checkpoint.py --nemo_ckpt_path megatron_converted_8b_tp4_pp1.nemo \
        --dtype float16 \
        --output_dir gpt-next-8B/trt_ckpt/fp16/1-gpu

# Build TensorRT LLM engines with prompt-tuning enabled
trtllm-build --checkpoint_dir gpt-next-8B/trt_ckpt/fp16/1-gpu \
        --max_prompt_embedding_table_size 100 \
        --output_dir gpt-next-8B/trt_engines/fp16/1-gpu
```

You can now export the learned embedding table with:
```bash
python3 nemo_prompt_convert.py -i email_composition.nemo -o email_composition.npy
```
It'll give you a summary of the different tasks in the table, that you can specify at runtime.

Finally, you can run inference on pre-defined tokens:
```bash
python3 ../../../run.py --engine_dir gpt-next-8B/trt_engines/fp16/1-gpu \
        --vocab_file gpt-next-8B/trt_ckpt/fp16/1-gpu/tokenizer.model \
        --no_add_special_tokens \
        --prompt_table_path email_composition.npy \
        --prompt_tasks 0 \
        --max_output_len 8
```


### MultiLoRA with the Nemo checkpoint

```bash
# Download NeMo checkpoint
wget https://huggingface.co/nvidia/GPT-2B-001/resolve/main/GPT-2B-001_bf16_tp1.nemo

# Convert to TensorRT LLM checkpoint
python3 convert_checkpoint.py --nemo_ckpt_path GPT-2B-001_bf16_tp1.nemo \
        --dtype float16 \
        --output_dir gpt-next-2B/trt_ckpt/fp16/1-gpu

# Build TensorRT LLM engines
trtllm-build --checkpoint_dir gpt-next-2B/trt_ckpt/fp16/1-gpu \
        --lora_plugin auto \
        --lora_dir gpt2b_lora-900.nemo gpt2b_lora-stories.nemo \
        --lora_ckpt_source "nemo" \
        --lora_target_modules attn_qkv \
        --max_batch_size 4 \
        --max_beam_width 2 \
        --max_input_len 512 \
        --max_seq_len 562 \
        --output_dir gpt-next-2B/trt_engines/fp16/1-gpu

# Run inference directly from NeMo LoRA checkpoint
# --lora_task_ids correspond to the index of the models given with --lora_dir. -1 means no LoRA
python3 ../../../run.py --engine_dir gpt-next-2B/trt_engines/fp16/1-gpu \
        --vocab_file gpt-next-2B/trt_ckpt/fp16/1-gpu/tokenizer.model \
        --no_add_special_tokens \
        --max_output_len 20 \
        --use_py_session \
        --lora_task_uids 0 -1 1 \
        --input_text "After Washington had returned to Williamsburg, Dinwiddie ordered him to lead a larger force to assist Trent in his work. While en route, Washington learned of Trent's retreat. Since Tanaghrisson had promised support to the British, Washington continued toward Fort Duquesne and met with the Mingo leader. Learning of a French scouting party in the area, Washington, with Tanaghrisson and his party, surprised the Canadians on May 28 in what became known as the Battle of Jumonville Glen. They killed many of the Canadians, including their commanding officer, Joseph Coulon de Jumonville, whose head was reportedly split open by Tanaghrisson with a tomahawk. The historian Fred Anderson suggests that Tanaghrisson was acting to gain the support of the British and regain authority over his own people. They had been inclined to support the French, with whom they had long trading relationships. One of Tanaghrisson's men told Contrecoeur that Jumonville had been killed by British musket fire. Question: Upon learning of a French scounting party in the area, what did Washington do? Answer:" "You hold the job title in the Wizarding World of Harry Potter where you say random words looking for spells" "You hold the job title in the Wizarding World of Harry Potter where you say random words looking for spells"
```

The output would look like (Note that in this case the adapters have only been trained for a few epochs, so the result quality is poor):

```
......
Input [Text 0]: "After Washington had returned to Williamsburg, Dinwiddie ordered him to lead a larger force to assist Trent in his work. While en route, Washington learned of Trent's retreat. Since Tanaghrisson had promised support to the British, Washington continued toward Fort Duquesne and met with the Mingo leader. Learning of a French scouting party in the area, Washington, with Tanaghrisson and his party, surprised the Canadians on May 28 in what became known as the Battle of Jumonville Glen. They killed many of the Canadians, including their commanding officer, Joseph Coulon de Jumonville, whose head was reportedly split open by Tanaghrisson with a tomahawk. The historian Fred Anderson suggests that Tanaghrisson was acting to gain the support of the British and regain authority over his own people. They had been inclined to support the French, with whom they had long trading relationships. One of Tanaghrisson's men told Contrecoeur that Jumonville had been killed by British musket fire. Question: Upon learning of a French scounting party in the area, what did Washington do? Answer:"
Output [Text 0 Beam 0]: "He surprised the Canadians on May 28 in what became known as the Battle of Jumonville"
Input [Text 1]: "You hold the job title in the Wizarding World of Harry Potter where you say random words looking for spells"
Output [Text 1 Beam 0]: ".

The game is played with a deck of cards, and the player who has the most"
Input [Text 2]: "You hold the job title in the Wizarding World of Harry Potter where you say random words looking for spells"
Output [Text 2 Beam 0]: ".

You are a wizard who is a wizard.

You are a wizard who is"
```
