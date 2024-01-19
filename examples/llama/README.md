# LLaMA

This document shows how to build and run a LLaMA model in TensorRT-LLM on both single GPU, single node multi-GPU and multi-node multi-GPU.

## Overview

The TensorRT-LLM LLaMA implementation can be found in [tensorrt_llm/models/llama/model.py](../../tensorrt_llm/models/llama/model.py). The TensorRT-LLM LLaMA example code is located in [`examples/llama`](./). There is one main file:

* [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the LLaMA model.

In addition, there are two shared files in the parent folder [`examples`](../) for inference and evaluation:

* [`../run.py`](../run.py) to run the inference on an input text;
* [`../summarize.py`](../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset.

## Support Matrix
  * FP16
  * FP8
  * INT8 & INT4 Weight-Only
  * SmoothQuant
  * Groupwise quantization (AWQ/GPTQ)
  * FP8 KV CACHE
  * INT8 KV CACHE (+ AWQ/per-channel weight-only)
  * Tensor Parallel
  * STRONGLY TYPED

## Usage

The TensorRT-LLM LLaMA example code locates at [examples/llama](./). It takes HF weights as input, and builds the corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### Build TensorRT engine(s)

Need to prepare the HF LLaMA checkpoint first by following the guides here https://huggingface.co/docs/transformers/main/en/model_doc/llama.

TensorRT-LLM LLaMA builds TensorRT engine(s) from HF checkpoint. If no checkpoint directory is specified, TensorRT-LLM will build engine(s) with dummy weights.

Normally `build.py` only requires single GPU, but if you've already got all the GPUs needed while inferencing, you could enable parallelly building to make the engine building process faster by adding `--parallel_build` argument. Please note that currently `parallel_build` feature only supports single node.

`--use_fused_mlp` enables GEMM horizontal fusion in gated MLP layer, which reduces input traffic and potentially improves performance. For FP8 PTQ, the downside is slight reduction of accuracy because one of the quantization scaling factors are discarded (accuracy 0.45734 vs 0.45755 for LLaMA-v2 7B using ammo/examples/hf/instruct_eval/mmlu.py).

Here're some examples:

```bash
# Build a single-GPU float16 engine from HF weights.
# use_gpt_attention_plugin is necessary in LLaMA.
# Try use_gemm_plugin to prevent accuracy issue.
# It is recommend to use --remove_input_padding along with --use_gpt_attention_plugin for better performance

# Build the LLaMA 7B model using a single GPU and FP16.
python build.py --model_dir ./tmp/llama/7B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/llama/7B/trt_engines/fp16/1-gpu/

# Build the LLaMA 7B model using a single GPU and BF16.
python build.py --model_dir ./tmp/llama/7B/ \
                --dtype bfloat16 \
                --remove_input_padding \
                --use_gpt_attention_plugin bfloat16 \
                --enable_context_fmha \
                --use_gemm_plugin bfloat16 \
                --output_dir ./tmp/llama/7B/trt_engines/bf16/1-gpu/

# Build the LLaMA 7B model using a single GPU and apply INT8 weight-only quantization.
python build.py --model_dir ./tmp/llama/7B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --use_weight_only \
                --output_dir ./tmp/llama/7B/trt_engines/weight_only/1-gpu/

# Build LLaMA 7B using 2-way tensor parallelism.
python build.py --model_dir ./tmp/llama/7B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/llama/7B/trt_engines/fp16/2-gpu/ \
                --world_size 2 \
                --tp_size 2

# Build LLaMA 7B using 2-way tensor parallelism and 2-way pipeline parallelism.
python build.py --model_dir ./tmp/llama/7B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/llama/7B/trt_engines/fp16/2-gpu/ \
                --world_size 4 \
                --tp_size 2 \
                --pp_size 2

# Build LLaMA 30B using 2-way tensor parallelism.
python build.py --model_dir ./tmp/llama/30B/hf/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/llama/30B/trt_engines/fp16/2-gpu/ \
                --world_size 2 \
                --tp_size 2
```

#### LLaMA v2 Updates
The LLaMA v2 models with 7B and 13B are compatible with the LLaMA v1 implementation. The above
commands still work.


For LLaMA v2 70B, there is a restriction on tensor parallelism that the number of KV heads
must be **divisible by the number of GPUs**. For example, since the 70B model has 8 KV heads, you can run it with
2, 4 or 8 GPUs (1 GPU as well for FP8).


```bash
# Build LLaMA 70B using 8-way tensor parallelism.
python build.py --model_dir ./tmp/llama/70B/hf/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/llama/70B/trt_engines/fp16/8-gpu/ \
                --world_size 8 \
                --tp_size 8

# Build LLaMA 70B using 4-way tensor parallelism and 2-way pipeline parallelism.
python build.py --model_dir ./tmp/llama/70B/hf/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/llama/70B/trt_engines/fp16/8-gpu/ \
                --world_size 8 \
                --tp_size 4 \
                --pp_size 2


# Build LLaMA 70B TP=8 using Meta checkpoints directly.
python build.py --meta_ckpt_dir ./tmp/llama/70B \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/llama/70B/trt_engines/fp16/8-gpu/ \
                --world_size 8 \
                --tp_size 8
```

Same instructions can be applied to fine-tuned versions of the LLaMA v2 models (e.g. 7Bf or llama-2-7b-chat).

### Using RoPE Scaling
RoPE scaling is supported through GPT Attention Plugin. You can add `--rotary_scaling <type> <factor>` during the build command to enable it.
- The value of `type` can be either `linear` and `dynamic`.
- The value of `factor` can be any value larger than `1.0`.

The implementation is identical to Huggingface's.
Please refer to https://huggingface.co/docs/transformers/model_doc/llama2#transformers.LlamaConfig.rope_scaling for more details.

### Long context length
To use the model with Long context lengths, it is necessary to add `--multi_block_mode` in the build command to enable faster decoding in multihead attention.


A few LLaMA models are fine-tuned for long context length that TRT-LLM can support today. For example https://huggingface.co/Yukang/LongAlpaca-70B employs rotary scaling plus fine-tuning to support up to 32K context length. The following show the steps for running LongAlpaca-70B in TRT-LLM:


```bash
# Build 8-GPU engine with long context LLaMA model
python build.py --model_dir ./tmp/LongAlpaca-70B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/llama/70B/trt_engines/fp16/8-gpu/ \
                --world_size 8 \
                --tp_size 8 \
                --pp_size 1 \
                --multi_block_mode \
                --max_input_len 32768 \
                --max_output_len 16384 \
                --vocab_size=32001 \
                --rotary_scaling linear 8.0

# Get the long text data from Gutenberg Project
wget https://www.gutenberg.org/cache/epub/64317/pg64317.txt

# Run with 8 GPUs
# Notice, `--max_input_length <n>` is a convenience option to limit the input length for the data.
# It should be set to the maximum context length the model supports. Here the limit is set to 32K.
mpirun -n 8 --allow-run-as-root \
    python ../run.py \
    --max_output_len 128 \
    --max_input_length 32768 \
    --input_file pg64317.txt \
    --engine_dir ./tmp/llama/70B/trt_engines/fp16/8-gpu/ \
    --tokenizer_dir ./tmp/LongAlpaca-70B/
```

Note that if engine is built with contiguous KV cache (i.e., without the flag `--paged_kv_cache`), you may need to reduce the max batch size (`--max_batch_size`) to fit the whole model and the KV cache in the GPU memory. The ballpark estimate for runtime memory consumption is given by

```
Total memory = (Model size + KV cache size + Activation memory) / Parallelism
```

where
- The model size is `the number of parameters * the size of data type`.
- The KV cache size is `the total number of tokens * the size of KV cache data type * the number of layers * the KV hidden dimension`
- The activation memory is determined by TRT engine, which can be a few GBs regardless of the degree of parallelism used

For LLaMA v2 70B FP16 weights + FP8 KV cache, the model size is 70B parameters * 2 bytes = 140GB. The KV cache size is 32K tokens * 1 bytes * 80 layers * 2048 KV hidden dimension = 5GB per 32K tokens. We have 145GB spread across 8 GPUs. The end result is ~18GB per GPU plus some GBs of flat scratch/activation memory allocated by TRT engine and the TRT-LLM runtime.

Note that the KV hidden dimension is derived by the number of KV heads times hidden dimension of each head. LLaMA v2 70B has hidden dimension of 8192, and uses grouped-query attention where 8 key heads and 8 value heads are associated with 64 query heads. Each head has hidden dimension of 8192/64 = 128. So the hidden dimension for KV in total is 128 * 8 * 2 = 2048.

The total number of tokens is determined by beam width, batch size, and maximum sequence length.

#### INT8 KV cache
INT8 KV cache could be enabled to reduce memory footprint. It will bring more performance gains when batch size gets larger.

You can get the INT8 scale of KV cache through [`hf_llama_convert.py`](./hf_llama_convert.py), which features a
`--calibrate-kv-cache, -kv` option. Setting `-kv` will calibrate the model,
and then export the scaling factors needed for INT8 KV cache inference.


Example:

```bash
python3 hf_llama_convert.py -i /llama-models/llama-7b-hf -o /llama/smooth_llama_7B/int8_kv_cache/ --calibrate-kv-cache -t fp16
```

[`build.py`](./build.py) add new options for the support of INT8 KV cache.

`--int8_kv_cache` is the command-line option to enable INT8 KV cache, and `--bin_model_dir` is the directory where the INT8 KV cache scales are located.

**INT8 KV cache + per-channel weight-only quantization**

INT8 KV cache could be combined with per-channel weight-only quantization, as follows:

Examples of INT8 weight-only quantization + INT8 KV cache

```bash
# Build model with both INT8 weight-only and INT8 KV cache enabled
python build.py --bin_model_dir=/llama/smooth_llama_7B/int8_kv_cache/1-gpu/ \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/llama/7B/trt_engines/int8_kv_cache_weight_only/1-gpu \
                --int8_kv_cache \
                --use_weight_only
```

Test with `../summarize.py`:

```bash
python ../summarize.py --test_trt_llm \
                       --hf_model_dir /llama-models/llama-7b-hf \
                       --data_type fp16 \
                       --engine_dir ./tmp/llama/7B/trt_engines/int8_kv_cache_weight_only/1-gpu \
                       --test_hf
```

**INT8 KV cache + AWQ**

In addition, you can enable INT8 KV cache together with AWQ (per-group INT4 weight-only quantization)like the following command.

**NOTE**: AWQ checkpoint is passed through `--quant_ckpt_path`, and the INT8 scales for the KV cache are expected to be in the directory pointed by `--bin_model_dir`.

```bash
python build.py --model_dir ./tmp/llama/7B/ \
                --quant_ckpt_path ./llama-7b-4bit-gs128-awq.pt \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --use_weight_only \
                --weight_only_precision int4_awq \
                --per_group \
                --output_dir ./tmp/llama/7B/trt_engines/int8_kv_cache_int4_AWQ/1-gpu/
                --int8_kv_cache \ # Turn on INT8 KV cache
                --bin_model_dir /llama/smooth_llama_7B/int8_kv_cache/1-gpu/ # Directory to look for INT8 scale of KV cache
```

Test with `../summarize.py`:

```bash
python ../summarize.py --test_trt_llm \
                       --hf_model_dir /llama-models/llama-7b-hf \
                       --data_type fp16 \
                       --engine_dir ./tmp/llama/7B/trt_engines/int8_kv_cache_int4_AWQ/1-gpu \
                       --test_hf
```

#### SmoothQuant

The smoothquant supports both LLaMA v1 and LLaMA v2. Unlike the FP16 build where the HF weights are processed and loaded into the TensorRT-LLM directly, the SmoothQuant needs to load INT8 weights which should be pre-processed before building an engine.

Example:
```bash
python3 hf_llama_convert.py -i /llama-models/llama-7b-hf -o /llama/smooth_llama_7B/sq0.8/ -sq 0.8 --tensor-parallelism 1 --storage-type fp16
```

[`build.py`](./build.py) add new options for the support of INT8 inference of SmoothQuant models.

`--use_smooth_quant` is the starting point of INT8 inference. By default, it
will run the model in the _per-tensor_ mode.

Then, you can add any combination of `--per-token` and `--per-channel` to get the corresponding behaviors.

Examples of build invocations:

```bash
# Build model for SmoothQuant in the _per_token_ + _per_channel_ mode
python3 build.py --bin_model_dir=/llama/smooth_llama_7B/sq0.8/1-gpu/ \
                 --use_gpt_attention_plugin float16 \
                 --remove_input_padding \
                 --enable_context_fmha \
                 --use_smooth_quant \
                 --per_token \
                 --per_channel
```

Note we use `--bin_model_dir` instead of `--model_dir` and `--meta_ckpt_dir` since SmoothQuant model needs INT8 weights and various scales from the binary files.

#### FP8 Post-Training Quantization

The examples below uses the NVIDIA AMMO (AlgorithMic Model Optimization) toolkit for the model quantization process.

First make sure AMMO toolkit is installed (see [examples/quantization/README.md](/examples/quantization/README.md#preparation))

After successfully running the script, the output should be in .npz format, e.g. `quantized_fp8/llama_tp_1_rank0.npz`,
where FP8 scaling factors are stored.

```bash
# Quantize HF LLaMA 70B into FP8 and export a single-rank checkpoint
python examples/quantization/quantize.py --model_dir ./tmp/llama/70B \
                                         --dtype float16 \
                                         --qformat fp8 \
                                         --export_path ./quantized_fp8 \
                                         --calib_size 512 \

# Build LLaMA 70B TP=2 using original HF checkpoint + PTQ scaling factors from the single-rank checkpoint
python build.py --model_dir ./tmp/llama/70B \
                --quantized_fp8_model_path ./quantized_fp8/llama_tp1_rank0.npz \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --output_dir ./tmp/llama/70B/trt_engines/fp8/2-gpu/ \
                --remove_input_padding \
                --enable_context_fmha \
                --enable_fp8 \
                --fp8_kv_cache \
                --strongly_typed \
                --world_size 2 \
                --tp_size 2 \
                --parallel_build
```

#### Groupwise quantization (AWQ/GPTQ)
One can enable AWQ/GPTQ INT4 weight only quantization with these options when building engine with `build.py`:

- `--use_weight_only` enables weight only GEMMs in the network.
- `--per_group` enable groupwise weight only quantization, for GPT-J example, we support AWQ with the group size default as 128.
- `--weight_only_precision` should specify the weight only quantization format. Supported formats are `int4_awq` or `int4_gptq`.
- `--quant_ckpt_path` passes the quantized checkpoint to build the engine.

AWQ/GPTQ examples below involves 2 steps:
1. Weight quantization
2. Build TRT-LLM engine

##### AWQ
1. Weight quantization:

    NVIDIA AMMO toolkit is used for AWQ weight quantization. Please see [examples/quantization/README.md](/examples/quantization/README.md#preparation) for AMMO installation instructions.

    ```bash
    # Quantize HF LLaMA 7B checkpoint into INT4 AWQ format
    python examples/quantization/quantize.py --model_dir ./tmp/llama/7B \
                                             --dtype float16 \
                                             --qformat int4_awq \
                                             --export_path ./quantized_int4-awq \
                                             --calib_size 32
    ```
    The quantized model checkpoint is saved to `./quantized_int4-awq/llama_tp1_rank0.npz` for future TRT-LLM engine build.

2. Build TRT-LLM engine:

    ```bash
    python build.py --model_dir ./tmp/llama/7B/ \
                    --quant_ckpt_path ./quantized_int4-awq/llama_tp1_rank0.npz \
                    --dtype float16 \
                    --remove_input_padding \
                    --use_gpt_attention_plugin float16 \
                    --enable_context_fmha \
                    --use_gemm_plugin float16 \
                    --use_weight_only \
                    --weight_only_precision int4_awq \
                    --per_group \
                    --output_dir ./tmp/llama/7B/trt_engines/int4_AWQ/1-gpu/
    ```

##### GPTQ
To run the GPTQ LLaMa example, the following steps are required:

1. Weight quantization:

    Quantized weights for GPTQ are generated using [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa.git) as follow:

    ```bash
    git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa.git
    cd GPTQ-for-LLaMa
    pip install -r requirements.txt

    # Quantize weights into INT4 and save as safetensors
    # Quantized weight with parameter "--act-order" is not supported in TRT-LLM
    python llama.py ./tmp/llama/7B/ c4 --wbits 4 --true-sequential --groupsize 128 --save_safetensors ./llama-7b-4bit-gs128.safetensors
    ```

    Let us build the TRT-LLM engine with the saved `./llama-7b-4bit-gs128.safetensors`.

2. Build TRT-LLM engine:

    ```bash
    # Build the LLaMA 7B model using 2-way tensor parallelism and apply INT4 GPTQ quantization.
    # Compressed checkpoint safetensors are generated separately from GPTQ.
    python build.py --model_dir ./tmp/llama/7B/ \
                    --quant_ckpt_path ./llama-7b-4bit-gs128.safetensors \
                    --dtype float16 \
                    --remove_input_padding \
                    --use_gpt_attention_plugin float16 \
                    --enable_context_fmha \
                    --use_gemm_plugin float16 \
                    --use_weight_only \
                    --weight_only_precision int4_gptq \
                    --per_group \
                    --world_size 2 \
                    --tp_size 2 \
                    --output_dir ./tmp/llama/7B/trt_engines/int4_GPTQ/2-gpu/
    ```

### Run

To run a TensorRT-LLM LLaMA model using the engines generated by build.py

```bash
# With fp16 inference
python3 ../run.py --max_output_len=50 \
                  --tokenizer_dir ./tmp/llama/7B/ \
                  --engine_dir=./tmp/llama/7B/trt_engines/fp16/1-gpu/

# With bf16 inference
python3 ../run.py --max_output_len=50 \
                  --tokenizer_dir ./tmp/llama/7B/ \
                  --engine_dir=./tmp/llama/7B/trt_engines/bf16/1-gpu/
```

### Summarization using the LLaMA model

```bash
# Run summarization using the LLaMA 7B model in FP16.
python ../summarize.py --test_trt_llm \
                       --hf_model_dir ./tmp/llama/7B/ \
                       --data_type fp16 \
                       --engine_dir ./tmp/llama/7B/trt_engines/fp16/1-gpu/

# Run summarization using the LLaMA 7B model quantized to INT8.
python ../summarize.py --test_trt_llm \
                       --hf_model_dir ./tmp/llama/7B/ \
                       --data_type fp16 \
                       --engine_dir ./tmp/llama/7B/trt_engines/weight_only/1-gpu/

# Run summarization using the LLaMA 7B model in FP16 using two GPUs.
mpirun -n 2 --allow-run-as-root \
    python ../summarize.py --test_trt_llm \
                           --hf_model_dir ./tmp/llama/7B/ \
                           --data_type fp16 \
                           --engine_dir ./tmp/llama/7B/trt_engines/fp16/2-gpu/

# Run summarization using the LLaMA 30B model in FP16 using two GPUs.
mpirun -n 2 --allow-run-as-root \
    python ../summarize.py --test_trt_llm \
                           --hf_model_dir ./tmp/llama/30B/ \
                           --data_type fp16 \
                           --engine_dir ./tmp/llama/30B/trt_engines/fp16/2-gpu/
```

#### Mistral v0.1
Mistral v0.1 is compatible with LLaMA interface and can be built and run using the same instructions.
Setting `--max_input_len`, corresponding to the `max_position_embeddings` in the original Mistral config explicitly regulates context size.
The `--max_attention_window_size` parameter is set to the `sliding_window` value in the config and regulates both sliding window attention in the context phase and rolling buffer cache in the generation phase.

```bash
# Build Mistral 7B with max input length 32256
python build.py --model_dir ./tmp/mistral/7B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/mistral/7B/trt_engines/fp16/1-gpu/ \
                --max_input_len 32256

# Run Mistral 7B fp16 inference with sliding window/cache size 4096
python3 run.py --max_output_len=50 \
               --tokenizer_dir ./tmp/llama/7B/ \
               --engine_dir=./tmp/llama/7B/trt_engines/fp16/1-gpu/ \
               --max_attention_window_size=4096
```

Note that if you are comparing TRT-LLM with Huggingface,
you should install `transformers` with version >= 4.34.1 in order to have Mistral model supported.
And upgrade `flash-attn` package by `pip install --upgrade flash-attn` or you may see wrong results generated by the huggingface implementation.

## Running CodeLlama
Those examples can be used to build and run the CodeLlama models. All 7b, 13b, and 34b sizes and variants are supported.

There are a couple of differences in CodeLlama in comparison to LLaMA v1/v2 models: rotary_base (`theta=1000000.0f`) and vocabulary size (`32016` (1)).

_(1): Only applicable to 7b and 13b model sizes_. 34b model variants use `32000`.

### Build
Use the following command to build `CodeLlama-7b-Instruct`:
```
python build.py --meta_ckpt_dir ./CodeLlama-7b-Instruct/ --dtype float16 \
    --remove_input_padding --use_gpt_attention_plugin float16 --use_gemm_plugin float16 \
    --enable_context_fmha --output_dir codellama_7b --rotary_base 1000000 --vocab_size 32016
```
Use the following command to build `CodeLlama-34b-Instruct` for 4 GPUs (TP=4):
```
python build.py --meta_ckpt_dir ./CodeLlama-34b-Instruct/ --dtype float16 \
    --remove_input_padding --use_gpt_attention_plugin float16 --use_gemm_plugin float16 \
    --enable_context_fmha --output_dir codellama_34b --rotary_base 1000000 --vocab_size 32000 --world_size 4 --tp_size 4
```

NOTE: CodeLlama uses the `max_position_embeddings` of 16K.
To build the engine for running similarly long input/output, you need to specify that during build.

Use `--max_input_len` and `--max_output_len` (which defaults to `2048` and `512`, respectively) according to your use case, e.g.:
```
python build.py --meta_ckpt_dir ./CodeLlama-34b-Instruct/ --dtype float16 \
    --remove_input_padding --use_gpt_attention_plugin float16 --use_gemm_plugin float16 \
    --output_dir codellama_34b --rotary_base 1000000 --vocab_size 32000 --world_size 8 --tp_size 8 --parallel_build \
    --enable_context_fmha --use_parallel_embedding --max_input_len 15360 --max_output_len 1024 --max_batch_size 4
```

### Run
Use the following command to run the 7b engine from above:
```
python ../run.py --max_output_len=40 --tokenizer_dir . --engine_dir codellama_7b --input_text "In Bash, how do I list all text files?"
```
Use the following command to run the 34b engine with long input/output from above:
```
mpirun -n 8 --allow-run-as-root \
    python ../run.py --max_output_len=160 --tokenizer_dir ./CodeLlama-34b-Instruct \
    --engine_dir codellama_34b --input_text "In python, write a function for binary searching an element in an integer array."
```

## Run LLaMa with LoRA

* download the base model and lora model from HF

```bash
git-lfs clone https://huggingface.co/meta-llama/Llama-2-13b-hf
git-lfs clone https://huggingface.co/hfl/chinese-llama-2-lora-13b
```

* Build engine, setting `--use_lora_plugin` and `--hf_lora_dir`. If lora has separate lm_head and embedding, they will replace lm_head and embedding of base model.

```bash
python build.py --model_dir llama-v2-13b-hf/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir "/tmp/new_lora_13b/trt_engines/fp16/2-gpu/" \
                --max_batch_size 1 \
                --max_input_len 512 \
                --max_output_len 50 \
                --use_lora_plugin float16 \
                --hf_lora_dir "chinese-llama-2-lora-13b/" \
                --world_size 2 --tp_size 2 --use_fused_mlp

```

* Run inference. Need to setup the `lora_dir`. Remember to use lora tokenizer because lora model has larger vocab size.

```bash
mpirun -n 2 python ../run.py --engine_dir "/tmp/new_lora_13b/trt_engines/fp16/2-gpu/" \
              --max_output_len 50 \
              --tokenizer_dir "chinese-llama-2-lora-13b/" \
              --input_text "今天天气很好，我到公园的时后，" \
              --lora_dir "chinese-llama-2-lora-13b/" \
              --lora_task_uids 0 \
              --no_add_special_tokens \
              --use_py_session

 Input: "今天天气很好，我到公园的时后，"
Output: "发现公园里人很多，有的在打羽毛球，有的在打乒乓球，有的在跳绳，还有的在跑步。我和妈妈来到一个空地上，我和妈妈一起跳绳，我跳了1"
```
Users who want to skip LoRA module may pass uid -1 with `--lora_task_uids -1`.
In that case, the model will not run the LoRA module and the results will be
different.

```bash
mpirun -n 2 python ../run.py --engine_dir "/tmp/new_lora_13b/trt_engines/fp16/2-gpu/" \
              --max_output_len 50 \
              --tokenizer_dir "chinese-llama-2-lora-13b/" \
              --input_text "今天天气很好，我到公园的时后，" \
              --lora_dir "chinese-llama-2-lora-13b/" \
              --lora_task_uids -1 \
              --no_add_special_tokens \
              --use_py_session

 Input: "今天天气很好，我到公园的时后，"
Output: "我看见一个人坐在那边边看书书，我看起来还挺像你，可是我走过过去问了一下他说你是你吗，他说没有，然后我就说你看我看看你像你，他说说你看我像你，我说你是你，他说你是你，"
```

### Run LLaMa with several lora checkpoints

In this section, we show how to run a model with multiple LoRA modules at the same time. Note that if one of the LoRA module has a
fine-tuned embedding table or logit GEMM, users should guarantee that all the instances of the model can use the same finetuned
embedding table or logit GEMM.
Here, we use two LoRA checkpoints as examples. These two LoRA checkponits add LoRA modules to `q_proj` and `v_proj`. Because we only
support adding lora modules on `q`, `k` and `v` at the same time, we need to add `--lora_target_modules "attn_q" "attn_k" "attn_v"`.
In this case, we assign null pointers for the `k` LoRA module in TensorRT-LLM and skip the computation at runtime.

As the rank of the LoRA modules of both checkpoints is 8, we can set `--max_lora_rank 8` to reduce the memory requirement for the LoRA plugin.

In this example, we use a LoRA checkpoint finetuned on the Chinese dataset `luotuo-lora-7b-0.1` and a LoRA checkpoint finetuned on
the Japanese dataset `Japanese-Alpaca-LoRA-7b-v0`. For the `lora_manager` to load several checkpoints, we pass several directories
of LoRA checkpoints at the same time: `--lora_dir  "luotuo-lora-7b-0.1/" "Japanese-Alpaca-LoRA-7b-v0/"`.
Then, `lora_manager` will assign `lora_task_uids` to these checkpoints. `lora_task_uids -1` is a predefined value, which corresponds to
the base model. If we pass `lora_task_uids 0 1`, this means we want to use the first LoRA checkpoint on first sentence and use the second LoRA checkpoint on the second sentence.

To verify the correctness, we pass the same Chinese input `美国的首都在哪里? \n答案:` three times as well as the same Japanese input `アメリカ合衆国の首都はどこですか? \n答え:` three times (both inputs mean `Where is the capital of America? \nAnswer`). We run on base model, `luotuo-lora-7b-0.1` and `Japanese-Alpaca-LoRA-7b-v0/`.

```bash
git-lfs clone https://huggingface.co/qychen/luotuo-lora-7b-0.1
git-lfs clone https://huggingface.co/kunishou/Japanese-Alpaca-LoRA-7b-v0
BASE_LLAMA_MODEL=llama-7b-hf/

python build.py --model_dir ${BASE_LLAMA_MODEL} \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir "/tmp/llama_7b_with_lora_qkv/trt_engines/fp16/1-gpu/" \
                --max_batch_size 128 \
                --max_input_len 512 \
                --max_output_len 50 \
                --use_lora_plugin float16 \
                --hf_lora_dir "Japanese-Alpaca-LoRA-7b-v0/" \
                --lora_target_modules "attn_q" "attn_k" "attn_v" \
                --max_lora_rank 8 \
                --world_size 1 --tp_size 1

python ../run.py --engine_dir "/tmp/llama_7b_with_lora_qkv/trt_engines/fp16/1-gpu/" \
              --max_output_len 10 \
              --tokenizer_dir ${BASE_LLAMA_MODEL} \
              --input_text "美国的首都在哪里? \n答案:" "美国的首都在哪里? \n答案:" "美国的首都在哪里? \n答案:" "アメリカ合衆国の首都はどこですか? \n答え:" "アメリカ合衆国の首都はどこですか? \n答え:" "アメリカ合衆国の首都はどこですか? \n答え:" \
              --lora_dir  "luotuo-lora-7b-0.1/" "Japanese-Alpaca-LoRA-7b-v0/" \
              --lora_task_uids -1 0 1 -1 0 1 \
              --use_py_session --top_p 0.5 --top_k 0
```

The results would be like

```bash
Input [Text 0]: "<s> 美国的首都在哪里? \n答案:"
Output [Text 0 Beam 0]: "Washington, D.C.
What is the"

Input [Text 1]: "<s> 美国的首都在哪里? \n答案:"
Output [Text 1 Beam 0]: "华盛顿。
"

Input [Text 2]: "<s> 美国的首都在哪里? \n答案:"
Output [Text 2 Beam 0]: "Washington D.C.�����"

Input [Text 3]: "<s> アメリカ合衆国の首都はどこですか? \n答え:"
Output [Text 3 Beam 0]: "Washington, D.C.
Which of"

Input [Text 4]: "<s> アメリカ合衆国の首都はどこですか? \n答え:"
Output [Text 4 Beam 0]: "华盛顿。
"

Input [Text 5]: "<s> アメリカ合衆国の首都はどこですか? \n答え:"
Output [Text 5 Beam 0]: "ワシントン D.C."
```

We can observe that `luotuo-lora-7b-0.1` produces correct answers on the first sentence and the fifth sentence (in Chinese), `Japanese-Alpaca-LoRA-7b-v0` produces correct answers on the sixth sentence (in Japanese).

## Run LLaMa with StreamingLLM

* Build engine. Set `--enable_pos_shift` to use positions in KV cache for RoPE, and set `--dense_context_fmha` to use dense context fmha in context phase.

```bash
# Build the LLaMA 7B model with StreamingLLM feature using a single GPU and FP16.
python build.py --model_dir ./tmp/llama/7B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --enable_pos_shift \
                --dense_context_fmha \
                --output_dir ./tmp/llama/7B/trt_engines/fp16_StreamingLLM/1-gpu/
```

* Run inference. Use `--sink_token_length` to set the number of sink tokens, and use `--max_attention_window_size` to set the `sliding_window` value.

```bash
# Run LLaMA 7B fp16 inference with sliding window/cache size 2048 and sink token length 4.
python3 run.py --max_output_len=50 \
               --tokenizer_dir ./tmp/llama/7B/ \
               --engine_dir=./tmp/llama/7B/trt_engines/fp16_StreamingLLM/1-gpu/ \
               --max_attention_window_size=2048 \
               --sink_token_length=4
```

Note that the sink tokens is included in the sliding attention tokens, and there are at most `max_attention_window_size` tokens are stored in the KV cache.
