# LLaMA

This document shows how to build and run a LLaMA model in TensorRT-LLM on both single GPU, single node multi-GPU and multi-node multi-GPU.

## Overview

The TensorRT-LLM LLaMA implementation can be found in [tensorrt_llm/models/llama/model.py](../../tensorrt_llm/models/llama/model.py). The TensorRT-LLM LLaMA example code is located in [`examples/llama`](./). There are three main files in that folder::

 * [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the LLaMA model,
 * [`run.py`](./run.py) to run the inference on an input text,
 * [`summarize.py`](./summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset using the model.

## Support Matrix
  * FP16
  * FP8
  * INT8 & INT4 Weight-Only
  * FP8 KV CACHE
  * Tensor Parallel
  * STRONGLY TYPED

## Usage

The TensorRT-LLM LLaMA example code locates at [examples/llama](./). It takes HF weights as input, and builds the corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### Build TensorRT engine(s)

Need to prepare the HF LLaMA checkpoint first by following the guides here https://huggingface.co/docs/transformers/main/en/model_doc/llama.

TensorRT-LLM LLaMA builds TensorRT engine(s) from HF checkpoint. If no checkpoint directory is specified, TensorRT-LLM will build engine(s) with dummy weights.

Normally `build.py` only requires single GPU, but if you've already got all the GPUs needed while inferencing, you could enable parallelly building to make the engine building process faster by adding `--parallel_build` argument. Please note that currently `parallel_build` feature only supports single node.

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

#### INT8 weight only + INT8 KV cache
For INT8 KV cache, [`hf_llama_convert.py`](./hf_llama_convert.py) features a
`--calibrate-kv-cache, -kv` option. Setting `-kv` will calibrate the model,
and then export the scaling factors needed for INT8 KV cache inference.


Example:

```bash
python3 hf_llama_convert.py -i /llama-models/llama-7b-hf -o /llama/smooth_llama_7B/int8_kv_cache/ --calibrate-kv-cache -t fp16
```

[`build.py`](./build.py) add new options for the support of INT8 KV cache.

`--int8_kv_cache` is the command-line option to enable INT8 KV cache.

In addition, it could be combined with INT8 weight-only quantization, as follows:

Examples of INT8 weight-only quantization + INT8 KV cache

```bash
# Build model with both INT8 weight-only and INT8 KV cache enabled
python build.py --ft_model_dir=/llama/smooth_llama_7B/int8_kv_cache/1-gpu/ \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/llama/7B/trt_engines/int8_kv_cache_weight_only/1-gpu \
                --int8_kv_cache \
                --use_weight_only
```

Test with `summarize.py`:

```bash
python summarize.py --test_trt_llm \
                    --hf_model_location /llama-models/llama-7b-hf \
                    --data_type fp16 \
                    --engine_dir ./tmp/llama/7B/trt_engines/int8_kv_cache_weight_only/1-gpu \
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
# Build model for SmoothQuant in the _per_tensor_ mode.
python3 build.py --ft_model_dir=/llama/smooth_llama_7B/sq0.8/1-gpu/ \
                 --use_smooth_quant

# Build model for SmoothQuant in the _per_token_ + _per_channel_ mode
python3 build.py --ft_model_dir=/llama/smooth_llama_7B/sq0.8/1-gpu/ \
                 --use_smooth_quant \
                 --per_token \
                 --per_channel
```

Note we use `--ft_model_dir` instead of `--model_dir` and `--meta_ckpt_dir` since SmoothQuant model needs INT8 weights and various scales from the binary files.

#### FP8 Post-Training Quantization

The examples below uses the NVIDIA AMMO (AlgorithMic Model Optimization) toolkit for the model quantization process.

First make sure AMMO toolkit is installed (see [examples/quantization/README.md](/examples/quantization/README.md#preparation))

After successfully running the script, the output should be in .npz format, e.g. `quantized_fp8/llama_tp_1_rank0.npz`,
where FP8 scaling factors are stored.

```bash
# Quantize HF LLaMA 70B into FP8 and export a single-rank checkpoint
python quantize.py --model_dir ./tmp/llama/70B \
                   --dtype float16 \
                   --qformat fp8 \
                   --export_path ./quantized_fp8 \
                   --calib_size 512 \

# Build LLaMA 70B TP=2 using original HF checkpoint + PTQ scaling factors from the single-rank checkpoint
python build.py --model_dir ./tmp/llama/70B \
                --quantized_fp8_model_path ./quantized_fp8/llama_tp1_rank0.npz \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/llama/70B/trt_engines/fp8/2-gpu/ \
                --remove_input_padding \
                --enable_fp8 \
                --fp8_kv_cache \
                --world_size 2 \
                --tp_size 2
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
    python quantize.py --model_dir ./tmp/llama/7B \
                    --dtype float16 \
                    --qformat int4_awq \
                    --export_path ./llama-7b-4bit-gs128-awq.pt \
                    --calib_size 32
    ```
    The quantized model checkpoint is saved to path `./llama-7b-4bit-gs128-awq.pt` for future TRT-LLM engine build.

2. Build TRT-LLM engine:

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
    # Compressed checkpoint safetensors are generated seperately from GPTQ.
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
python3 run.py --max_output_len=50 \
               --tokenizer_dir ./tmp/llama/7B/ \
               --engine_dir=./tmp/llama/7B/trt_engines/fp16/1-gpu/

# With bf16 inference
python3 run.py --max_output_len=50 \
               --tokenizer_dir ./tmp/llama/7B/ \
               --engine_dir=./tmp/llama/7B/trt_engines/bf16/1-gpu/
```

### Summarization using the LLaMA model

```bash
# Run summarization using the LLaMA 7B model in FP16.
python summarize.py --test_trt_llm \
                    --hf_model_location ./tmp/llama/7B/ \
                    --data_type fp16 \
                    --engine_dir ./tmp/llama/7B/trt_engines/fp16/1-gpu/

# Run summarization using the LLaMA 7B model quantized to INT8.
python summarize.py --test_trt_llm \
                    --hf_model_location ./tmp/llama/7B/ \
                    --data_type fp16 \
                    --engine_dir ./tmp/llama/7B/trt_engines/weight_only/1-gpu/

# Run summarization using the LLaMA 7B model in FP16 using two GPUs.
mpirun -n 2 --allow-run-as-root \
    python summarize.py --test_trt_llm \
                        --hf_model_location ./tmp/llama/7B/ \
                        --data_type fp16 \
                        --engine_dir ./tmp/llama/7B/trt_engines/fp16/2-gpu/

# Run summarization using the LLaMA 30B model in FP16 using two GPUs.
mpirun -n 2 --allow-run-as-root \
    python summarize.py --test_trt_llm \
                        --hf_model_location ./tmp/llama/30B/ \
                        --data_type fp16 \
                        --engine_dir ./tmp/llama/30B/trt_engines/fp16/2-gpu/
```

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
    --remove_input_padding --use_gpt_attention_plugin float16 --use_gemm_plugin float16 --use_rmsnorm_plugin float16 \
    --enable_context_fmha --output_dir codellama_34b --rotary_base 1000000 --vocab_size 32000 --world_size 4 --tp_size 4
```

NOTE: CodeLlama uses the `max_position_embeddings` of 16K.
To build the engine for running similarly long input/output, you need to specify that during build.

Use `--max_input_len` and `--max_output_len` (which defaults to `2048` and `512`, respectively) according to your use case, e.g.:
```
python build.py --meta_ckpt_dir ./CodeLlama-34b-Instruct/ --dtype float16 \
    --remove_input_padding --use_gpt_attention_plugin float16 --use_gemm_plugin float16 --use_rmsnorm_plugin float16 \
    --output_dir codellama_34b --rotary_base 1000000 --vocab_size 32000 --world_size 8 --tp_size 8 --parallel_build \
    --enable_context_fmha --use_parallel_embedding --max_input_len 15360 --max_output_len 1024 --max_batch_size 4
```

### Run
Use the following command to run the 7b engine from above:
```
python run.py --max_output_len=40 --tokenizer_dir . --engine_dir codellama_7b --input_text "In Bash, how do I list all text files?"
```
Use the following command to run the 34b engine with long input/output from above:
```
mpirun -n 8 --allow-run-as-root \
    python run.py --max_output_len=160 --tokenizer_dir ./CodeLlama-34b-Instruct \
    --engine_dir codellama_34b --input_text "In python, write a function for binary searching an element in an integer array."
```
