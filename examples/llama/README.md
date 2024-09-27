# LLaMA

This document shows how to build and run a LLaMA model in TensorRT-LLM on both single GPU, single node multi-GPU and multi-node multi-GPU.

- [LLaMA](#llama)
  - [Overview](#overview)
  - [Support Matrix](#support-matrix)
  - [Usage](#usage)
    - [Build TensorRT engine(s)](#build-tensorrt-engines)
      - [LLaMA v2 Updates](#llama-v2-updates)
      - [LLaMA v3 Updates](#llama-v3-updates)
    - [Long context length](#long-context-length)
      - [Long context evaluation](#long-context-evaluation)
      - [1M long context test case](#1m-long-context-test-case)
    - [INT8 KV cache](#int8-kv-cache)
    - [SmoothQuant](#smoothquant)
    - [FP8 Post-Training Quantization](#fp8-post-training-quantization)
    - [Groupwise quantization (AWQ/GPTQ)](#groupwise-quantization-awqgptq)
      - [AWQ](#awq)
      - [GPTQ](#gptq)
    - [Run](#run)
    - [Multi-GPU multi-node (MGMN) support](#multi-gpu-multi-node-mgmn-support)
    - [Summarization using the LLaMA model](#summarization-using-the-llama-model)
      - [Mistral v0.1](#mistral-v01)
      - [Mistral Nemo](#mistral-nemo)
  - [Running CodeLlama](#running-codellama)
    - [Build](#build)
    - [Run](#run-1)
  - [Run models with LoRA](#run-models-with-lora)
    - [Run LLaMa with several lora checkpoints](#run-llama-with-several-lora-checkpoints)
    - [Run FP8 Mistral v0.1 with FP16 lora checkpoint](#run-fp8-mistral-v01-with-fp16-lora-checkpoint)
    - [Run INT4-AWQ LLaMa with several FP16 lora checkpoints](#run-int4-awq-llama-with-several-fp16-lora-checkpoints)
  - [Run LLaMa with StreamingLLM](#run-llama-with-streamingllm)
  - [Run LLaMA-3.1 405B Model](#run-llama-31-405b-model)
    - [Convert Checkpoint to TensorRT-LLM Unified Checkpoint](#convert-checkpoint-to-tensorrt-llm-unified-checkpoint)
    - [Build Engine](#build-engine)
    - [Run Inference](#run-inference)

## Overview

The TensorRT-LLM LLaMA implementation can be found in [tensorrt_llm/models/llama/model.py](../../tensorrt_llm/models/llama/model.py). The TensorRT-LLM LLaMA example code is located in [`examples/llama`](./). There is one main file:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert the LLaMA model into tensorrt-llm checkpoint format.

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

Please install required packages first to make sure the example uses matched `tensorrt_llm` version:

```bash
pip install -r requirements.txt
```

Need to prepare the HF LLaMA checkpoint by following the guides here https://huggingface.co/docs/transformers/main/en/model_doc/llama.

The `trtllm-build` command builds TensorRT engine(s) from HF checkpoint. If no checkpoint directory is specified, TensorRT-LLM will build engine(s) with dummy weights.

`trtllm-build` command has a variety of options. In particular, the plugin-related options have two categories:
* Plugin options that requires a data type (e.g., `gpt_attention_plugin`), you can
    * explicitly specify `float16`/`bfloat16`/`float32`, so that the plugins are enabled with the specified precision;
    * implicitly specify `auto`, so that the plugins are enabled with the precision automatically inferred from model dtype (i.e., the dtype specified in weight conversion); or
    * disable the plugin by `disable`.
* Other features that requires a boolean (e.g., `context_fmha`, `paged_kv_cache`, `remove_input_padding`), you can
    * enable/disable the feature by specifying `enable`/`disable`.

The defaults have been carefully tuned for better performance. For example, `gpt_attention_plugin`, `context_fmha`, `paged_kv_cache` and `remove_input_padding` are enabled by default. See more details by `trtllm-build --help`.

Normally `trtllm-build` only requires single GPU, but if you've already got all the GPUs needed for inference, you could enable parallel building to make the engine building process faster by adding `--workers` argument. Please note that currently `workers` feature only supports single node.

`--use_fused_mlp=enable` enables GEMM horizontal fusion in gated MLP layer, which reduces input traffic and potentially improves performance. For FP8 PTQ, the downside is slight reduction of accuracy because one of the quantization scaling factors are discarded (accuracy 0.45734 vs 0.45755 for LLaMA-v2 7B using modelopt/examples/hf/instruct_eval/mmlu.py).

`--use_fused_mlp=enable --gemm_swiglu_plugin <dtype>` fuses 2 GEMMs without biases and SwiGLU into one kernel. This is a preview feature and is only supported for dtype `fp8`. The supported architecture is SM90.

Here're some examples:

```bash
# Build a single-GPU float16 engine from HF weights.
# Try use_gemm_plugin to prevent accuracy issue.

# Build the LLaMA 7B model using a single GPU and FP16.
python convert_checkpoint.py --model_dir ./tmp/llama/7B/ \
                              --output_dir ./tllm_checkpoint_1gpu_fp16 \
                              --dtype float16

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp16 \
            --output_dir ./tmp/llama/7B/trt_engines/fp16/1-gpu \
            --gemm_plugin auto

# Build the LLaMA 7B model using a single GPU and BF16.
python convert_checkpoint.py --model_dir ./tmp/llama/7B/ \
                              --output_dir ./tllm_checkpoint_1gpu_bf16 \
                              --dtype bfloat16

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_bf16 \
            --output_dir ./tmp/llama/7B/trt_engines/bf16/1-gpu \
            --gemm_plugin auto

# Build the LLaMA 7B model using a single GPU and apply INT8 weight-only quantization.
python convert_checkpoint.py --model_dir ./tmp/llama/7B/ \
                              --output_dir ./tllm_checkpoint_1gpu_fp16_wq \
                              --dtype float16 \
                              --use_weight_only \
                              --weight_only_precision int8

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp16_wq \
            --output_dir ./tmp/llama/7B/trt_engines/weight_only/1-gpu/ \
            --gemm_plugin auto

# Build LLaMA 7B using 2-way auto parallelism.
python convert_checkpoint.py --model_dir ./tmp/llama/7B/ \
                            --output_dir ./tllm_checkpoint_1gpu_fp16 \
                            --dtype float16

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp16 \
            --output_dir ./tmp/llama/7B/trt_engines/fp16/2-gpu/ \
            --gemm_plugin auto \
            --auto_parallel 2

# Build LLaMA 7B using 2-way tensor parallelism.
python convert_checkpoint.py --model_dir ./tmp/llama/7B/ \
                            --output_dir ./tllm_checkpoint_2gpu_tp2 \
                            --dtype float16 \
                            --tp_size 2

trtllm-build --checkpoint_dir ./tllm_checkpoint_2gpu_tp2 \
            --output_dir ./tmp/llama/7B/trt_engines/fp16/2-gpu/ \
            --gemm_plugin auto

# Build LLaMA 7B using 2-way tensor parallelism and 2-way pipeline parallelism.
python convert_checkpoint.py --model_dir ./tmp/llama/7B/ \
                            --output_dir ./tllm_checkpoint_4gpu_tp2_pp2 \
                            --dtype float16 \
                            --tp_size 2 \
                            --pp_size 2
trtllm-build --checkpoint_dir ./tllm_checkpoint_4gpu_tp2_pp2 \
            --output_dir ./tmp/llama/7B/trt_engines/fp16/4-gpu/ \
            --gemm_plugin auto

# Build LLaMA 30B using 2-way tensor parallelism.
python convert_checkpoint.py --model_dir ./tmp/llama/30B/hf/ \
                            --output_dir ./tllm_checkpoint_2gpu_tp2 \
                            --dtype float16 \
                            --tp_size 2

trtllm-build --checkpoint_dir ./tllm_checkpoint_2gpu_tp2 \
            --output_dir ./tmp/llama/30B/trt_engines/fp16/2-gpu/ \
            --gemm_plugin auto
```

#### LLaMA v2 Updates
The LLaMA v2 models with 7B and 13B are compatible with the LLaMA v1 implementation. The above
commands still work.


For LLaMA v2 70B, there is a restriction on tensor parallelism that the number of KV heads
must be **divisible by the number of GPUs**. For example, since the 70B model has 8 KV heads, you can run it with
2, 4 or 8 GPUs (1 GPU as well for FP8).


```bash
# Build LLaMA 70B using 8-way tensor parallelism.
python convert_checkpoint.py --model_dir ./tmp/llama/70B/hf/ \
                            --output_dir ./tllm_checkpoint_8gpu_tp8 \
                            --dtype float16 \
                            --tp_size 8

trtllm-build --checkpoint_dir ./tllm_checkpoint_8gpu_tp8 \
            --output_dir ./tmp/llama/70B/trt_engines/fp16/8-gpu/ \
            --gemm_plugin auto

# Build LLaMA 70B using 4-way tensor parallelism and 2-way pipeline parallelism.
python convert_checkpoint.py --model_dir ./tmp/llama/70B/hf/ \
                            --output_dir ./tllm_checkpoint_8gpu_tp4_pp2 \
                            --dtype float16 \
                            --tp_size 4 \
                            --pp_size 2

trtllm-build --checkpoint_dir ./tllm_checkpoint_8gpu_tp4_pp2 \
            --output_dir ./tmp/llama/70B/trt_engines/fp16/8-gpu/ \
            --gemm_plugin auto

# Build LLaMA 70B TP=8 using Meta checkpoints directly.
python convert_checkpoint.py --meta_ckpt_dir ./tmp/llama/70B/ \
                            --output_dir ./tllm_checkpoint_8gpu_tp8 \
                            --dtype float16 \
                            --tp_size 8

trtllm-build --checkpoint_dir ./tllm_checkpoint_8gpu_tp8 \
            --output_dir ./tmp/llama/70B/trt_engines/fp16/8-gpu/ \
            --gemm_plugin auto
```

Same instructions can be applied to fine-tuned versions of the LLaMA v2 models (e.g. 7Bf or llama-2-7b-chat).

#### LLaMA v3 Updates
The LLaMA v3 models with 8B and 70b are compatible with the LLaMA v2 implementation. The above
commands still work.

Note that the `rope_theta` and `vocab_size` are larger in LLaMA v3 models and these values are now inferred
or pickup up from the `params.json` when using the `meta_ckpt_dir`.

```bash
# Build LLaMA v3 8B TP=1 using HF checkpoints directly.
python convert_checkpoint.py --model_dir ./tmp/llama/8B/hf/ \
                            --output_dir ./tllm_checkpoint_1gpu_tp1 \
                            --dtype float16 \
                            --tp_size 1

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_tp1 \
            --output_dir ./tmp/llama/8B/trt_engines/fp16/1-gpu/ \
            --gemm_plugin auto

# Build LLaMA v3 8B TP=1 using Meta checkpoints directly.
python convert_checkpoint.py --meta_ckpt_dir ./tmp/llama/8B/ \
                            --output_dir ./tllm_checkpoint_1gpu_tp1 \
                            --dtype float16 \
                            --tp_size 1

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_tp1 \
            --output_dir ./tmp/llama/8B/trt_engines/fp16/1-gpu/ \
            --gemm_plugin auto

# Build LLaMA v3 70B using 8-way tensor parallelism.
python convert_checkpoint.py --model_dir ./tmp/llama/70B/hf/ \
                            --output_dir ./tllm_checkpoint_8gpu_tp8 \
                            --dtype float16 \
                            --tp_size 8

trtllm-build --checkpoint_dir ./tllm_checkpoint_8gpu_tp8 \
            --output_dir ./tmp/llama/70B/trt_engines/fp16/8-gpu/ \
            --gemm_plugin auto

# Build LLaMA v3 70B using 4-way tensor parallelism and 2-way pipeline parallelism.
python convert_checkpoint.py --model_dir ./tmp/llama/70B/hf/ \
                            --output_dir ./tllm_checkpoint_8gpu_tp4_pp2 \
                            --dtype float16 \
                            --tp_size 4 \
                            --pp_size 2

trtllm-build --checkpoint_dir ./tllm_checkpoint_8gpu_tp4_pp2 \
            --output_dir ./tmp/llama/70B/trt_engines/fp16/8-gpu/ \
            --gemm_plugin auto

# Build LLaMA v3 70B TP=8 using Meta checkpoints directly.
python convert_checkpoint.py --meta_ckpt_dir ./tmp/llama/70B/ \
                            --output_dir ./tllm_checkpoint_8gpu_tp8 \
                            --dtype float16 \
                            --tp_size 8

trtllm-build --checkpoint_dir ./tllm_checkpoint_8gpu_tp8 \
            --output_dir ./tmp/llama/70B/trt_engines/fp16/8-gpu/ \
            --gemm_plugin auto
```

Same instructions can be applied to fine-tuned versions of the LLaMA v2 models (e.g. 7Bf or llama-2-7b-chat).

### Long context length
With long context lengths, multi_block_mode is turned on by default to enable faster decoding in multi-head attention. To disable this feature, add `--multi_block_mode=False` to the runtime command.


A few LLaMA models are fine-tuned for long context length that TRT-LLM can support today. For example https://huggingface.co/Yukang/LongAlpaca-70B employs rotary scaling plus fine-tuning to support up to 32K context length. The following show the steps for running LongAlpaca-70B in TRT-LLM:


```bash
# Build 8-GPU engine with long context LLaMA model
python convert_checkpoint.py --model_dir ./tmp/LongAlpaca-70B/ \
                            --output_dir ./tllm_checkpoint_8gpu_tp8 \
                            --dtype float16 \
                            --tp_size 8 \

trtllm-build --checkpoint_dir ./tllm_checkpoint_8gpu_tp8 \
            --output_dir ./tmp/llama/70B/trt_engines/fp16/8-gpu/ \
            --gemm_plugin auto

# Get the long text data from Gutenberg Project
wget https://www.gutenberg.org/cache/epub/64317/pg64317.txt

# Replace the line breaks with special character '\n' and append "Summarize this story:" at end of text
awk '{printf "%s\\n", $0} END {printf "\\nSummarize this story:"}' pg64317.txt > pg64317_sanitized.txt

# Run with 8 GPUs
# Notice, `--max_input_length <n>` is a convenience option to limit the input length for the data.
# It should be set to the maximum context length the model supports. Here the limit is set to 32K.
mpirun -n 8 --allow-run-as-root \
    python ../run.py \
    --max_output_len 128 \
    --max_input_length 32768 \
    --input_file pg64317_sanitized.txt \
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

#### Long context evaluation

* Download dataset and model

```bash
git-lfs clone https://huggingface.co/datasets/DKYoon/SlimPajama-6B
git-lfs clone https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k/
```

* Run examples with max_input_len 16384

To evaluate the PPL of very long context, we need to enable `use_paged_context_fmha` and setup `max_num_tokens` to enable the chunked context inference, reducing the activation memory requirement. Also, we need to enable `gather_all_token_logits` to return the logits to compute the PPL.

```bash
python examples/llama/convert_checkpoint.py --model_dir ./Llama-3-8B-Instruct-Gradient-1048k/ \
                              --output_dir /tmp/llama-3-8B-1048k/trt_ckpts \
                              --dtype float16

python -m tensorrt_llm.commands.build --checkpoint_dir /tmp/llama-3-8B-1048k/trt_ckpts \
            --output_dir /tmp/llama-3-8B-1048k/trt_engines \
            --gemm_plugin float16 \
            --gather_all_token_logits \
            --max_num_tokens 4096 \
            --max_input_len 16384 \
            --max_seq_len 16394 \
            --use_paged_context_fmha enable

python ./examples/summarize.py --test_trt_llm \
                       --tokenizer_dir ./Llama-3-8B-Instruct-Gradient-1048k/ \
                       --data_type fp16 \
                       --engine_dir /tmp/llama-3-8B-1048k/trt_engines \
                       --eval_task eval_context_ppl \
                       --max_input_len 16384 \
                       --use_py_session \
                       --dataset_dir ./SlimPajama-6B/
```

* Run evaluation on passkey task

To evaluate the accuracy of very long context on `needle in haystack`, we need to enable `use_paged_context_fmha` and setup `max_num_tokens` to enable the chunked context inference, reducing the activation memory requirement. To save memory, we don't enable the `gather_all_token_logits` here because we don't need logits.

```bash
python3 examples/infinitebench/construct_synthetic_dataset.py --test_case build_passkey --test_level 4

python -m tensorrt_llm.commands.build --checkpoint_dir /tmp/llama-3-8B-1048k/trt_ckpts \
            --output_dir /tmp/llama-3-8B-1048k/trt_engines \
            --gemm_plugin float16 \
            --max_num_tokens 4096 \
            --max_input_len 131072 \
            --max_seq_len 131082 \
            --use_paged_context_fmha enable

python examples/eval_long_context.py  --task passkey \
                                      --engine_dir /tmp/llama-3-8B-1048k/trt_engines \
                                      --tokenizer_dir ./Llama-3-8B-Instruct-Gradient-1048k/ \
                                      --stop_idx 10 \
                                      --max_input_length 131072 \
                                      --enable_chunked_context \
                                      --max_tokens_in_paged_kv_cache 131136
```

* Run evaluation on kv_retrieval

`kv_retrieval` is harder than `passkey` and is helpful to distinguish the model capability.

To run the kv_retrieval, we need a third-party repo to prepare the keys.

```bash
git clone git@github.com:nelson-liu/lost-in-the-middle.git
pip install -r lost-in-the-middle/requirements.txt
python -u lost-in-the-middle/scripts/make_kv_retrieval_data.py --num-keys 3000 --num-examples 500 --output-path kv-retrieval-3000_keys.jsonl.gz
gzip -d kv-retrieval-3000_keys.jsonl.gz
```

Prepare input data and run evaluation.

```bash
python examples/infinitebench/construct_synthetic_dataset.py --test_case build_kv_retrieval --test_level 0

python examples/llama/convert_checkpoint.py --model_dir ./Llama-3-8B-Instruct-Gradient-1048k/ \
                              --output_dir /tmp/llama-3-8B-1048k/trt_ckpts \
                              --dtype float16 \
                              --tp_size 1

python -m tensorrt_llm.commands.build --checkpoint_dir /tmp/llama-3-8B-1048k/trt_ckpts \
            --output_dir /tmp/llama-3-8B-1048k/trt_engines \
            --gemm_plugin float16 \
            --max_num_tokens 4096 \
            --max_input_len 131072 \
            --max_seq_len 131082 \
            --use_paged_context_fmha enable

python examples/eval_long_context.py  --task kv_retrieval \
                                      --engine_dir /tmp/llama-3-8B-1048k/trt_engines \
                                      --tokenizer_dir ./Llama-3-8B-Instruct-Gradient-1048k/ \
                                      --stop_idx 10 \
                                      --max_input_length 131072 \
                                      --enable_chunked_context \
                                      --max_tokens_in_paged_kv_cache 131136 \
                                      --tensorrt_llm_accuracy_threshold 0.6
```

expected results:

```bash
[05/28/2024-03:31:43] [TRT-LLM] [I] ==== Evaluation ====
[05/28/2024-03:31:43] [TRT-LLM] [I] # examples: 500
[05/28/2024-03:31:43] [TRT-LLM] [I] Start index: 0
[05/28/2024-03:31:43] [TRT-LLM] [I] Stop index: 10
[05/28/2024-03:31:43] [TRT-LLM] [I] Max tokens: 50
[05/28/2024-03:34:50] [TRT-LLM] [I] Compute the score
10it [00:00, 131072.00it/s]
[05/28/2024-03:34:51] [TRT-LLM] [I] Evaluation takes: 187.19733428955078 sec.
[05/28/2024-03:34:51] [TRT-LLM] [I] accuracy of 10 examples: 0.6
```

#### 1M long context test case

- Prepare 1M needle-in-a-haystack datasets

```bash
python examples/infinitebench/construct_synthetic_dataset.py --test_case build_passkey --test_level 7
```

- Llama-3-8B example

```bash
git-lfs clone https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k/

python examples/llama/convert_checkpoint.py --model_dir ./Llama-3-8B-Instruct-Gradient-1048k/ \
                              --output_dir /tmp/llama-3-8B-1048k/trt_ckpts \
                              --dtype float16 \
                              --tp_size 4

python -m tensorrt_llm.commands.build --checkpoint_dir /tmp/llama-3-8B-1048k/trt_ckpts \
            --output_dir /tmp/llama-3-8B-1048k/trt_engines \
            --gemm_plugin float16 \
            --max_num_tokens 4096 \
            --max_batch_size 1 \
            --max_seq_len 1048576 \
            --use_paged_context_fmha enable \
            --workers 4

mpirun -n 4 --allow-run-as-root python examples/eval_long_context.py  --task passkey \
                                      --engine_dir /tmp/llama-3-8B-1048k/trt_engines \
                                      --tokenizer_dir ./Llama-3-8B-Instruct-Gradient-1048k/ \
                                      --stop_idx 1 \
                                      --max_input_length 1048566 \
                                      --enable_chunked_context \
                                      --max_tokens_in_paged_kv_cache 1100000
```

- Llama-3-70B example

For the 70B model, at least 8 A100 80GB GPUs are required.

```bash
git-lfs clone https://huggingface.co/gradientai/Llama-3-70B-Instruct-Gradient-1048k/

python examples/llama/convert_checkpoint.py --model_dir ./Llama-3-70B-Instruct-Gradient-1048k/ \
                              --output_dir /tmp/llama-3-70B-1048k/trt_ckpts \
                              --dtype float16 \
                              --tp_size 8

python -m tensorrt_llm.commands.build --checkpoint_dir /tmp/llama-3-70B-1048k/trt_ckpts \
            --output_dir /tmp/llama-3-70B-1048k/trt_engines \
            --gemm_plugin float16 \
            --max_num_tokens 4096 \
            --max_batch_size 1 \
            --max_seq_len 1048576 \
            --use_paged_context_fmha enable \
            --workers 8

mpirun -n 8 --allow-run-as-root python examples/eval_long_context.py  --task passkey \
                                      --engine_dir /tmp/llama-3-70B-1048k/trt_engines \
                                      --tokenizer_dir ./Llama-3-70B-Instruct-Gradient-1048k/ \
                                      --stop_idx 1 \
                                      --max_input_length 1048566 \
                                      --enable_chunked_context \
                                      --max_tokens_in_paged_kv_cache 1100000
```

expected result:

```bash
[05/27/2024-10:30:45] [TRT-LLM] [I] Compute the score
1it [00:00, 4215.38it/s]
[05/27/2024-10:30:45] [TRT-LLM] [I] accuracy of 1 examples: 1.0
```

### INT8 KV cache
INT8 KV cache could be enabled to reduce memory footprint. It will bring more performance gains when batch size gets larger.

For INT8 KV cache, [`convert_checkpoint.py`](./convert_checkpoint.py) features a
`--int8_kv_cache` option. Setting `--int8_kv_cache` will calibrate the model,
and then export the scaling factors needed for INT8 KV cache inference.

Example:

```bash
python convert_checkpoint.py --model_dir ./llama-models/llama-7b-hf   \
                             --output_dir ./llama-models/llama-7b-hf/int8_kv_cache/ \
                             --dtype float16  \
                             --int8_kv_cache
```

[`convert_checkpoint.py`](./convert_checkpoint.py) add new options for the support of INT8 KV cache.


**INT8 KV cache + per-channel weight-only quantization**

INT8 KV cache could be combined with per-channel weight-only quantization, as follows:

Examples of INT8 weight-only quantization + INT8 KV cache

```bash
# Build model with both INT8 weight-only and INT8 KV cache enabled
python convert_checkpoint.py --model_dir ./llama-models/llama-7b-hf   \
                             --output_dir ./tllm_checkpoint_1gpu_int8_kv_wq \
                             --dtype float16  \
                             --int8_kv_cache \
                             --use_weight_only \
                             --weight_only_precision int8

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_int8_kv_wq \
            --output_dir ./tmp/llama/7B/trt_engines/int8_kv_cache_weight_only/1-gpu \
            --gemm_plugin auto
```

Test with `../summarize.py`:

```bash
python ../summarize.py --test_trt_llm \
                       --hf_model_dir ./llama-models/llama-7b-hf \
                       --data_type fp16 \
                       --engine_dir ./tmp/llama/7B/trt_engines/int8_kv_cache_weight_only/1-gpu \
                       --test_hf
```

**INT8 KV cache + AWQ**

In addition, you can enable INT8 KV cache together with AWQ (per-group INT4 weight-only quantization)like the following command.

```bash
python ../quantization/quantize.py --model_dir /tmp/llama-7b-hf \
                                   --output_dir ./tllm_checkpoint_1gpu_awq_int8_kv_cache \
                                   --dtype float16 \
                                   --qformat int4_awq \
                                   --awq_block_size 128 \
                                   --kv_cache_dtype int8 \
                                   --calib_size 32

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_awq_int8_kv_cache \
            --output_dir ./tmp/llama/7B/trt_engines/int8_kv_cache_int4_AWQ/1-gpu/ \
            --gemm_plugin auto \
```

Test with `../summarize.py`:

```bash
python ../summarize.py --test_trt_llm \
                       --hf_model_dir /tmp/llama-7b-hf \
                       --data_type fp16 \
                       --engine_dir ./tmp/llama/7B/trt_engines/int8_kv_cache_int4_AWQ/1-gpu \
                       --test_hf
```

### SmoothQuant

The smoothquant supports both LLaMA v1 and LLaMA v2. Unlike the FP16 build where the HF weights are processed and loaded into the TensorRT-LLM directly, the SmoothQuant needs to load INT8 weights which should be pre-processed before building an engine.

Example:
```bash
python3 convert_checkpoint.py --model_dir /llama-models/llama-7b-hf  --output_dir /tmp/tllm_checkpoint_1gpu_sq --dtype float16 --smoothquant 0.5
trtllm-build --checkpoint_dir /tmp/tllm_checkpoint_1gpu_sq \
             --output_dir ./engine_outputs \
             --gemm_plugin auto
```

[`convert_checkpoint.py`](./convert_checkpoint.py) add new options for the support of INT8 inference of SmoothQuant models.

`--smoothquant` is the starting point of INT8 inference. By default, it
will run the model in the _per-tensor_ mode.

Then, you can add any combination of `--per-token` and `--per-channel` to get the corresponding behaviors.

Examples of build invocations:

```bash
# Build model for SmoothQuant in the _per_token_ + _per_channel_ mode
python3 convert_checkpoint.py --model_dir /llama-models/llama-7b-hf \
                            --output_dir /tmp/tllm_checkpoint_1gpu_sq \
                            --dtype float16 \
                            --smoothquant 0.5 \
                            --per_token \
                            --per_channel

trtllm-build --checkpoint_dir /tmp/tllm_checkpoint_1gpu_sq \
             --output_dir ./engine_outputs \
             --gemm_plugin auto
```

### FP8 Post-Training Quantization

The examples below uses the NVIDIA Modelopt (AlgorithMic Model Optimization) toolkit for the model quantization process.

First make sure Modelopt toolkit is installed (see [examples/quantization/README.md](/examples/quantization/README.md#preparation))


```bash
# Quantize HF LLaMA 70B into FP8 and export trtllm checkpoint
python ../quantization/quantize.py --model_dir ./tmp/llama/70B \
                                   --dtype float16 \
                                   --qformat fp8 \
                                   --kv_cache_dtype fp8 \
                                   --output_dir ./tllm_checkpoint_2gpu_fp8 \
                                   --calib_size 512 \
                                   --tp_size 2

# Build trtllm engines from the trtllm checkpoint
# Enable fp8 context fmha to get further acceleration by setting `--use_fp8_context_fmha enable`
trtllm-build --checkpoint_dir ./tllm_checkpoint_2gpu_fp8 \
             --output_dir ./engine_outputs \
             --gemm_plugin auto \
             --workers 2
```

**Note**: A LLaMA 70B model with BF16 is about 140GB, a LLaMA 70B model with FP8 is about 70GB.
The peak GPU memory consumption when doing FP8 quantizaton is more than 210GB (there is also some activation memory occupation when doing calibration).
So you need a node with at least 4 H100(A100) to run the quantization command. After quantization, 2 GPUs are okay to for building and run.

Experimental: use FP8 GEMV to optimize performance in FP8 small-batch-size cases.

```bash
# Quantize HF LLaMA 7B into FP8 and export trtllm checkpoint
python ../quantization/quantize.py --model_dir /tmp/llama-7b-hf \
                                   --dtype float16 \
                                   --qformat fp8 \
                                   --kv_cache_dtype fp8 \
                                   --output_dir ./tllm_checkpoint_1gpu_fp8 \
                                   --calib_size 512

# Build trtllm engines from the trtllm checkpoint
# Enable fp8 gemm plugin to get acceleration in small-batch-size cases
trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp8 \
             --output_dir ./engine_outputs \
             --gemm_plugin fp8
```

**Note**: FP8 gemm plugin is an experimental feature aimed to improve performance in small-batch-size cases(e.g. BS<=4). Although inputs with batch size larger than 4 can be correctly inferenced, the performance may decrease as batch size grows.

### Groupwise quantization (AWQ/GPTQ)
One can enable AWQ/GPTQ INT4 weight only quantization with these options when building engine with `trtllm-build`:

- `--use_weight_only` enables weight only GEMMs in the network.
- `--per_group` enable groupwise weight only quantization, for GPT-J example, we support AWQ with the group size default as 128.
- `--weight_only_precision` should specify the weight only quantization format. Supported formats are `int4_awq` or `int4_gptq`.
- `--quant_ckpt_path` passes the quantized checkpoint to build the engine.

AWQ/GPTQ examples below involves 2 steps:
1. Weight quantization
2. Build TRT-LLM engine

#### AWQ
1. Weight quantization:

    NVIDIA Modelopt toolkit is used for AWQ weight quantization. Please see [examples/quantization/README.md](/examples/quantization/README.md#preparation) for Modelopt installation instructions.

    ```bash
    # Quantize HF LLaMA 7B checkpoint into INT4 AWQ format
    python ../quantization/quantize.py --model_dir ./tmp/llama-7b-hf \
                                       --dtype float16 \
                                       --qformat int4_awq \
                                       --awq_block_size 128 \
                                       --output_dir ./quantized_int4-awq \
                                       --calib_size 32
    ```

2. Build TRT-LLM engine:

    ```bash
    trtllm-build --checkpoint_dir ./quantized_int4-awq \
                 --output_dir ./tmp/llama/7B/trt_engines/int4_AWQ/1-gpu/ \
                 --gemm_plugin auto
    ```

#### GPTQ
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
    python convert_checkpoint.py --model_dir /tmp/llama-7b-hf \
                                 --output_dir ./tllm_checkpoint_2gpu_gptq \
                                 --dtype float16 \
                                 --quant_ckpt_path ./llama-7b-4bit-gs128.safetensors  \
                                 --use_weight_only \
                                 --weight_only_precision int4_gptq \
                                 --per_group \
                                 --tp_size 2

    trtllm-build --checkpoint_dir ./tllm_checkpoint_2gpu_gptq \
                --output_dir ./tmp/llama/7B/trt_engines/int4_GPTQ/2-gpu/ \
                --gemm_plugin auto
    ```

### Run

To run a TensorRT-LLM LLaMA model using the engines generated by `trtllm-build`

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

### Multi-GPU multi-node (MGMN) support

In MGMN case, you can still convert and build engines on a single node and then run the model on a multi-node environment, such as [Slurm](https://slurm.schedmd.com/documentation.html).

For example, to build LLaMA 70B for 2 nodes with 8 GPUs per node, we can use 8-way tensor parallelism and 2-way pipeline parallelism:

```bash
python convert_checkpoint.py --model_dir ./tmp/llama/70B/hf/ \
                            --output_dir ./tllm_checkpoint_16gpu_tp8_pp2 \
                            --dtype float16 \
                            --tp_size 8 \
                            --pp_size 2

trtllm-build --checkpoint_dir ./tllm_checkpoint_16gpu_tp8_pp2 \
            --output_dir ./tmp/llama/70B/trt_engines/fp16/16-gpu/ \
            --workers 8 \
            --gemm_plugin auto
```

Note that `–-workers` is still set to 8 to build all engines within a single node.

To run the LLaMA 70B model on 2 nodes via Slurm, you need to prepare a Slurm script to submit the task, the script contains the following lines:

```bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH -p <partition>
# more sbatch options here...

srun --container-image=<docker-image> \
     --mpi=pmix \
     ... \ # more srun options here
     python3 ../run.py --max_output_len=50 \
                       --tokenizer_dir ./tmp/llama/70B/hf/ \
                       --engine_dir=./tmp/llama/70B/trt_engines/fp16/16-gpu/
```

Finally, you can submit the task with `sbatch <your-slurm-script>.sh`.

Considering the Slurm or other cluster management systems may be highly customized and the task-submit command may be variant, the forementioned example is for reference only. The key point is to submit the Python script with the MPI runtime, and TensorRT-LLM will take care of the rest.

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
python convert_checkpoint.py --model_dir ./mistral-7b-v0.1 \
                             --output_dir ./tllm_checkpoint_1gpu_mistral \
                             --dtype float16
trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_mistral \
             --output_dir ./tmp/mistral/7B/trt_engines/fp16/1-gpu/ \
             --gemm_plugin auto \
             --max_input_len 32256

# Run Mistral 7B fp16 inference with sliding window/cache size 4096
python ../run.py --max_output_len=50 \
                 --tokenizer_dir ./mistral-7b-v0.1 \
                 --engine_dir=./tmp/mistral/7B/trt_engines/fp16/1-gpu/ \
                 --max_attention_window_size=4096
```

Note that if you are comparing TRT-LLM with Huggingface,
you should install `transformers` with version >= 4.34.1 in order to have Mistral model supported.
And upgrade `flash-attn` package by `pip install --upgrade flash-attn` or you may see wrong results generated by the huggingface implementation.

#### Mistral Nemo
[Mistral Nemo](https://mistral.ai/news/mistral-nemo/) is compatible with LLaMA interface and can be built and run using the same instructions.
Please upgrade the [transformers](https://pypi.org/project/transformers/) to 4.43.0.dev0 or higher release for running this model.

```bash
# Build Mistral Nemo with max input length 10240
python convert_checkpoint.py --model_dir ./Mistral-Nemo-Instruct-2407 \
                             --output_dir ./tllm_checkpoint_1gpu_mistral_nemo \
                             --dtype bfloat16 \
                             --smoothquant 0.5 \
                             --per_channel \
                             --per_token

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_mistral_nemo \
             --output_dir ./tmp/mistral_nemo/trt_engines/bf16/1-gpu/ \
             --gemm_plugin bfloat16 \
             --max_input_len 10240

# Run summarization using the Mistral Nemo model quantized to INT8.
python ../summarize.py --test_trt_llm \
                       --hf_model_dir ./Mistral-Nemo-Instruct-2407 \
                       --data_type bf16 \
                       --engine_dir ./tmp/mistral_nemo/trt_engines/bf16/1-gpu//
```

## Running CodeLlama
Those examples can be used to build and run the CodeLlama models. All 7b, 13b, and 34b sizes and variants are supported.

There are a couple of differences in CodeLlama in comparison to LLaMA v1/v2 models: rotary_base (`theta=1000000.0f`) and vocabulary size (`32016` (1)).

_(1): Only applicable to 7b and 13b model sizes_. 34b model variants use `32000`.

### Build
Use the following command to build `CodeLlama-7b-Instruct`:
```bash
python convert_checkpoint.py --model_dir /tmp/CodeLlama-7b-Instruct-hf  \
                             --output_dir ./tllm_checkpoint_1gpu_codellama \
                             --dtype float16


trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_codellama \
            --output_dir ./tmp/codellama/trt_engines/fp16/1-gpu/ \
            --gemm_plugin auto
```
The example below uses the NVIDIA ModelOpt (AlgorithMic Model Optimization) toolkit for the model quantization process.
First make sure Modelopt toolkit is installed (see [examples/quantization/README.md](/examples/quantization/README.md#preparation))

```bash
# Quantize HF CodeLlama 7B into FP8 and export trtllm checkpoint
python ../quantization/quantize.py --model_dir /tmp/CodeLlama-7b-Instruct-hf \
                                   --dtype float16 \
                                   --qformat fp8 \
                                   --kv_cache_dtype fp8 \
                                   --output_dir ./tllm_checkpoint_1gpu_fp8 \
                                   --calib_size 512

# Build trtllm engines from the trtllm checkpoint
trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp8 \
             --output_dir ./engine_outputs \
             --gemm_plugin auto
```

Use the following command to build `CodeLlama-34b-Instruct` for 4 GPUs (TP=4):
```bash
python convert_checkpoint.py --model_dir /tmp/CodeLlama-34b-Instruct-hf  \
                             --output_dir ./tllm_checkpoint_4gpu_codellama \
                             --dtype float16 \
                             --tp_size 4

trtllm-build --checkpoint_dir ./tllm_checkpoint_4gpu_codellama \
            --output_dir ./tmp/codellama/trt_engines/fp16/4-gpu/ \
            --gemm_plugin auto
```

NOTE: CodeLlama uses the `max_position_embeddings` of 16K.
To build the engine for running similarly long input/output, you need to specify that during build.

Use `--max_input_len` (which defaults to `1024`) and `--max_seq_len` (which by default is deduced from `max_position_embeddings`) according to your use case, e.g.:
```bash
python convert_checkpoint.py --model_dir /tmp/CodeLlama-34b-Instruct-hf  \
                             --output_dir ./tllm_checkpoint_4gpu_codellama \
                             --dtype float16 \
                             --tp_size 8 \
                             --use_parallel_embedding

trtllm-build --checkpoint_dir ./tllm_checkpoint_4gpu_codellama \
            --output_dir ./tmp/codellama/trt_engines/fp16/4-gpu/ \
            --gemm_plugin auto \
            --max_input_len 15360 \
            --max_seq_len 16384 \
            --max_batch_size 4
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

## Run models with LoRA

Download the base model and lora model from HF:

```bash
git-lfs clone https://huggingface.co/meta-llama/Llama-2-13b-hf
git-lfs clone https://huggingface.co/hfl/chinese-llama-2-lora-13b
```

Build engine, setting `--lora_plugin` and `--lora_dir`. If lora has separate lm_head and embedding, they will replace lm_head and embedding of base model.

```bash
python convert_checkpoint.py --model_dir Llama-2-13b-hf \
                         --output_dir ./tllm_checkpoint_2gpu \
                         --dtype float16 \
                         --tp_size 2

trtllm-build --checkpoint_dir ./tllm_checkpoint_2gpu \
            --output_dir /tmp/new_lora_13b/trt_engines/fp16/2-gpu/ \
            --gemm_plugin auto \
            --lora_plugin auto \
            --max_batch_size 1 \
            --max_input_len 512 \
            --max_seq_len 562 \
            --lora_dir chinese-llama-2-lora-13b
```

Run inference. Remember to use lora tokenizer because lora model has larger vocab size.

```bash
mpirun -n 2 python ../run.py --engine_dir "/tmp/new_lora_13b/trt_engines/fp16/2-gpu/" \
              --max_output_len 50 \
              --tokenizer_dir "chinese-llama-2-lora-13b/" \
              --input_text "今天天气很好，我到公园的时候，" \
              --lora_task_uids 0 \
              --no_add_special_tokens \
              --use_py_session

 Input: "今天天气很好，我到公园的时候，"
Output: "发现公园里到处都是人，有的在跑步，有的在打羽毛球，还有的在跳绳，我和妈妈一起在公园里散步，我和妈妈在公园里散步的时候，看见了一位老爷爷在打羽毛球"
```

Users who want to skip LoRA module may pass uid -1 with `--lora_task_uids -1`.
In that case, the model will not run the LoRA module and the results will be
different. Since the LoRA tokenizer, embedding and LM head are still used,
the results will also be different with vanilla LLaMA and significantly degrade compared with `--lora_task_uids 0`.

```bash
mpirun -n 2 python ../run.py --engine_dir "/tmp/new_lora_13b/trt_engines/fp16/2-gpu/" \
              --max_output_len 50 \
              --tokenizer_dir "chinese-llama-2-lora-13b/" \
              --input_text "今天天气很好，我到公园的时候，" \
              --lora_task_uids -1 \
              --no_add_special_tokens \
              --use_py_session

 Input: "今天天气很好，我到公园的时候，"
Output: "看见好多人们都看书，看书书看书书，看书书看书书书书书书书书书书书书书书书书书书书书书书书书书书书书书书书书书书书书"
```

### Run LLaMa with several lora checkpoints

In this section, we show how to run a model with multiple LoRA modules at the same time. Note that if one of the LoRA module has a
fine-tuned embedding table or logit GEMM, users should guarantee that all the instances of the model can use the same fine-tuned
embedding table or logit GEMM.
Here, we use two LoRA checkpoints as examples. These two LoRA checkponits add LoRA modules to `q_proj` and `v_proj`. Because we only
support adding lora modules on `q`, `k` and `v` at the same time, we need to add `--lora_target_modules "attn_q" "attn_k" "attn_v"`.
In this case, we assign null pointers for the `k` LoRA module in TensorRT-LLM and skip the computation at runtime.

As the rank of the LoRA modules of both checkpoints is 8, we can set `--max_lora_rank 8` to reduce the memory requirement for the LoRA plugin.

In this example, we use a LoRA checkpoint fine-tuned on the Chinese dataset `luotuo-lora-7b-0.1` and a LoRA checkpoint fine-tuned on
the Japanese dataset `Japanese-Alpaca-LoRA-7b-v0`. For the `lora_manager` to load several checkpoints, we pass several directories
of LoRA checkpoints at the same time: `--lora_dir  "luotuo-lora-7b-0.1/" "Japanese-Alpaca-LoRA-7b-v0/"`.
Then, `lora_manager` will assign `lora_task_uids` to these checkpoints. `lora_task_uids -1` is a predefined value, which corresponds to
the base model. If we pass `lora_task_uids 0 1`, this means we want to use the first LoRA checkpoint on first sentence and use the second LoRA checkpoint on the second sentence.

To verify the correctness, we pass the same Chinese input `美国的首都在哪里? \n答案:` three times as well as the same Japanese input `アメリカ合衆国の首都はどこですか? \n答え:` three times (both inputs mean `Where is the capital of America? \nAnswer`). We run on base model, `luotuo-lora-7b-0.1` and `Japanese-Alpaca-LoRA-7b-v0/`.

```bash
git-lfs clone https://huggingface.co/qychen/luotuo-lora-7b-0.1
git-lfs clone https://huggingface.co/kunishou/Japanese-Alpaca-LoRA-7b-v0
BASE_LLAMA_MODEL=llama-7b-hf/

python convert_checkpoint.py --model_dir ${BASE_LLAMA_MODEL} \
                            --output_dir ./tllm_checkpoint_1gpu \
                            --dtype float16
trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu \
            --output_dir /tmp/llama_7b_with_lora_qkv/trt_engines/fp16/1-gpu/ \
            --gemm_plugin auto \
            --lora_plugin auto \
            --max_batch_size 8 \
            --max_input_len 512 \
            --max_seq_len 562 \
            --lora_dir  "luotuo-lora-7b-0.1/" "Japanese-Alpaca-LoRA-7b-v0/" \
            --max_lora_rank 8 \
            --lora_target_modules attn_q attn_k attn_v

python ../run.py --engine_dir "/tmp/llama_7b_with_lora_qkv/trt_engines/fp16/1-gpu/" \
              --max_output_len 10 \
              --tokenizer_dir ${BASE_LLAMA_MODEL} \
              --input_text "美国的首都在哪里? \n答案:" "美国的首都在哪里? \n答案:" "美国的首都在哪里? \n答案:" "アメリカ合衆国の首都はどこですか? \n答え:" "アメリカ合衆国の首都はどこですか? \n答え:" "アメリカ合衆国の首都はどこですか? \n答え:" \
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

### Run FP8 Mistral v0.1 with FP16 lora checkpoint

In this section, we use Mistral v0.1 as an example show how to run an FP8 base model with FP16 LoRA module.

* download the base model and lora model from HF

```bash
git-lfs clone https://huggingface.co/davidkim205/komt-mistral-7b-v1
git-lfs clone https://huggingface.co/davidkim205/komt-mistral-7b-v1-lora
```

* Quantize the Mistral v0.1 model to fp8 from HF
```bash
BASE_MISTRAL_MODEL=komt-mistral-7b-v1/
python ../quantization/quantize.py --model_dir ${BASE_MISTRAL_MODEL} \
                                   --dtype float16 \
                                   --qformat fp8 \
                                   --kv_cache_dtype fp8 \
                                   --output_dir ./tllm_checkpoint_1gpu_fp8 \
                                   --calib_size 512
```

* Build engine and run inference with sliding window/cache size 4096.
```bash
trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp8 \
            --output_dir /tmp/mistral_komt_lora/7B/trt_engines/fp8/1-gpu/ \
            --gemm_plugin auto \
            --lora_plugin auto \
            --max_batch_size 8 \
            --max_input_len 32256 \
            --max_seq_len 33280 \
            --lora_dir ./komt-mistral-7b-v1-lora

python ../run.py --max_output_len=1024 \
                 --tokenizer_dir ./komt-mistral-7b-v1 \
                 --engine_dir=/tmp/mistral_komt_lora/7B/trt_engines/fp8/1-gpu/ \
                 --input_text "[INST]오늘은 날씨가 아주 좋다 내가 공원에 갔을 때 [/INST]" \
                 --max_attention_window_size=4096 \
                 --lora_task_uids 0 \
                 --use_py_session \
                 --temperature 0.8 \
                 --top_p 0.8 \
                 --top_k 100
```

The results would be like

```bash
Input [Text 0]: "<s> [INST]오늘은 날씨가 아주 좋다 내가 공원에 갔을 때 [/INST]"
Output [Text 0 Beam 0]: "날씨가 아주 좋은 날에 공원에 갔을 때는 산책이나 운동을 즐기는 것이 좋습니다. 공원에서 걷거나 조깅을 하면서 신선한 공기를 마시고 자연 속에서 휴식을 취할 수 있습니다. 또한, 공원에서 가족이나 친구와 함께 피크닉을 즐기거나 야외 스포츠를 즐길 수도 있습니다. 날씨가 좋을 때 공원에 가는 것은 건강과 웰빙에 좋은 방법입니다."
```


### Run INT4-AWQ LLaMa with several FP16 lora checkpoints

TensorRT-LLM can also support Quantized base model + FP16/BF16 LoRA. We can first quantize the base model and build engine with the quantized checkpoint and different LoRA adapters. In this section, we show how to run an INT4-AWQ llama model with multiple FP16 LoRA modules.

* Quantize the llama model to INT4-AWQ from HF
```bash
BASE_LLAMA_MODEL=llama-7b-hf/
python ../quantization/quantize.py --model_dir ${BASE_LLAMA_MODEL} \
                                   --output_dir ./tllm_checkpoint_1gpu_awq \
                                   --dtype float16 \
                                   --qformat int4_awq \
                                   --awq_block_size 128 \
                                   --calib_size 32
```

* Download the lora model, build engine, and run inference.
```bash
git-lfs clone https://huggingface.co/qychen/luotuo-lora-7b-0.1
git-lfs clone https://huggingface.co/kunishou/Japanese-Alpaca-LoRA-7b-v0

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_awq \
            --output_dir /tmp/llama_7b_with_lora_qkv/trt_engines/int4_AWQ/1-gpu/ \
            --gemm_plugin auto \
            --lora_plugin auto \
            --max_batch_size 8 \
            --max_input_len 512 \
            --max_seq_len 562 \
            --lora_dir  "luotuo-lora-7b-0.1/" "Japanese-Alpaca-LoRA-7b-v0/" \
            --max_lora_rank 8 \
            --lora_target_modules attn_q attn_k attn_v

python ../run.py --engine_dir "/tmp/llama_7b_with_lora_qkv/trt_engines/int4_AWQ/1-gpu/" \
              --max_output_len 10 \
              --tokenizer_dir ${BASE_LLAMA_MODEL} \
              --input_text "美国的首都在哪里? \n答案:" "美国的首都在哪里? \n答案:" "美国的首都在哪里? \n答案:" "アメリカ合衆国の首都はどこですか? \n答え:" "アメリカ合衆国の首都はどこですか? \n答え:" "アメリカ合衆国の首都はどこですか? \n答え:" \
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
Output [Text 2 Beam 0]: "洛爱睿"

Input [Text 3]: "<s> アメリカ合衆国の首都はどこですか? \n答え:"
Output [Text 3 Beam 0]: "Washington, D.C.
Copyright "

Input [Text 4]: "<s> アメリカ合衆国の首都はどこですか? \n答え:"
Output [Text 4 Beam 0]: "华盛顿。
"

Input [Text 5]: "<s> アメリカ合衆国の首都はどこですか? \n答え:"
Output [Text 5 Beam 0]: "ワシントン、D.C"
```

## Run LLaMa with StreamingLLM

* Build engine. Set `--streamingllm enable` to enable StreamingLLM.

```bash
# Build the LLaMA 7B model with StreamingLLM feature using a single GPU and FP16.
python convert_checkpoint.py --model_dir ./tmp/llama/7B/ \
                         --output_dir ./tllm_checkpoint_1gpu_streamlingllm \
                         --dtype float16

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_streamlingllm \
            --output_dir ./tmp/llama/7B/trt_engines/fp16_StreamingLLM/1-gpu/ \
            --gemm_plugin auto \
            --streamingllm enable

```

* Run inference. Use `--sink_token_length` to set the number of sink tokens, and use `--max_attention_window_size` to set the `sliding_window` value.

```bash
# Run LLaMA 7B fp16 inference with sliding window/cache size 2048 and sink token length 4.
python3 ../run.py --max_output_len=50 \
                  --tokenizer_dir ./tmp/llama/7B/ \
                  --engine_dir=./tmp/llama/7B/trt_engines/fp16_StreamingLLM/1-gpu/ \
                  --max_attention_window_size=2048 \
                  --sink_token_length=4
```

Note that the sink tokens is included in the sliding attention tokens, and there are at most `max_attention_window_size` tokens are stored in the KV cache.


## Run LLaMA-3.1 405B Model

Currently, TensorRT-LLM supports Meta checkpoint and Huggingface checkpoint for LLaMA-3.1. In this section, we demonstrate how to run the LLaMA-3.1 405B model via TensorRT-LLM. Here, we assume users have downloaded the checkpoints and placed them at `llama_3.1_405B_meta_model/` (Meta BF16 checkpoint), `llama_3.1_405B_HF_model/` (HF BF16 checkpoint) and `llama_3.1_405B_HF_FP8_model/` (HF FP8 checkpoint). Before converting the checkpoints to TensorRT-LLM unified checkpoints, **please check that `{"rope_scaling": {"rope_type": "llama3"}}` is set in the configuration file**. With this flag, TensorRT-LLM will enable the rope scaling of LLaMA-3.1. If not, please add it to the config file.

Users can run the LLaMA-3.1 model with higher precision (bf16/fp16) or fp8. Here, to prevent accuracy drop, we perform per-channel per-token fp8 quantization (leveraged from https://github.com/pytorch/FBGEMM) on MLP layers, keeping other layers at higher precision. Note that per-channel per-token fp8 quantization is only supported on Huggingface checkpoint now. We will support it on Meta checkpoint soon. Note that this feature only supports SM90.

### Convert Checkpoint to TensorRT-LLM Unified Checkpoint

To use the fp8 quantization, please add the `--use_fp8_rowwise` flag during the checkpoint conversion. In this demonstration, we convert the Meta checkpoint to bfloat16 with TP8-PP2 and the HF checkpoint to FP8 with TP8.

Note: You may need to update your transformers installation via `pip install --upgrade transformers`.
Note: For 405B HF model cloned before 09 Aug 2024, there are duplicated kv head weights (The `num_key_value_heads` in `config.json` is 16). Users could use `--remove_duplicated_kv_heads` to remove them. The new checkpoint without duplicated kv heads is uploaded on 09 Aug 2024 and the `num_key_value_heads` is 8 now. For the new checkpoint, adding the flag `--remove_duplicated_kv_heads` would lead to error.

```bash
# Run BF16 model by BF16
python examples/llama/convert_checkpoint.py --meta_ckpt_dir llama_3.1_405B_meta_model/ \
                            --output_dir llama_3.1_405B_meta_model/trt_ckpts/tp8-pp2/ \
                            --dtype bfloat16 \
                            --tp_size 8 \
                            --pp_size 2 \
                            --load_by_shard \
                            --workers 2

# Run BF16 model by FP8
python examples/llama/convert_checkpoint.py --model_dir llama_3.1_405B_HF_model/ \
                            --output_dir llama_3.1_405B_HF_model/trt_ckpts/tp8-pp1/ \
                            --dtype bfloat16 \
                            --use_fp8_rowwise \
                            --tp_size 8 \
                            --pp_size 1 \
                            --load_by_shard \
                            --workers 8 \

# Run FP8 model by FP8
python examples/llama/convert_checkpoint.py --model_dir llama_3.1_405B_HF_FP8_model/ \
                            --output_dir llama_3.1_405B_HF_FP8_model/trt_ckpts/tp8-pp1/ \
                            --dtype bfloat16 \
                            --tp_size 8 \
                            --pp_size 1 \
                            --load_by_shard \
                            --workers 8
```

### Build Engine

```bash
trtllm-build --checkpoint_dir llama_3.1_405B_meta_model/trt_ckpts/tp8-pp2/ \
             --output_dir llama_3.1_405B_meta_model/trt_engines/tp8-pp2/ \
             --max_num_tokens 4096 \
             --max_input_len 255000 \
             --max_seq_len 256000 \
             --use_paged_context_fmha enable \
             --workers 8

trtllm-build --checkpoint_dir llama_3.1_405B_HF_model/trt_ckpts/tp8-pp1/ \
             --output_dir llama_3.1_405B_HF_model/trt_engines/tp8-pp1/ \
             --max_num_tokens 4096 \
             --max_input_len 64000 \
             --max_seq_len 65000 \
             --use_paged_context_fmha enable \
             --workers 8

trtllm-build --checkpoint_dir llama_3.1_405B_HF_FP8_model/trt_ckpts/tp8-pp1/ \
             --output_dir llama_3.1_405B_HF_FP8_model/trt_engines/tp8-pp1/ \
             --max_num_tokens 4096 \
             --max_input_len 64000 \
             --max_seq_len 65000 \
             --use_paged_context_fmha enable \
             --workers 8
```

### Run Inference

To run inference on the 405B model, we often need to use multi-node to accommodate the entire model. Here, we use slurm to launch the job on multiple nodes.

Notes:
* For the FP8 model, we can fit it on a single 8xH100 node, but we cannot support 128k context due to memory limitations. So, we test with 64k context in this demonstration.
* For convenience, we use the Huggingface tokenizer for tokenization.

The following script shows how to run evaluation on long context:

```bash
# Long context test for 128k
python ./examples/infinitebench/construct_synthetic_dataset.py --test_case build_passkey --test_level 4; mkdir -p 128k_context; mv passkey.jsonl 128k_context;

srun --mpi pmi2 -N 2 -n 16 --ntasks-per-node 8 --container-image <your container>  \
--container-mounts <your container mount> \
--container-name llama-3.1-405b \
--container-workdir <your container work directory> \
bash -c 'python ./examples/eval_long_context.py --task passkey \
                                                --engine_dir llama_3.1_405B_meta_model/trt_engines/tp8-pp2/ \
                                                --tokenizer_dir llama_3.1_405B_HF_model/ \
                                                --stop_idx 6 \
                                                --max_input_length 255000 \
                                                --enable_chunked_context \
                                                --kv_cache_free_gpu_memory_fraction 0.999 \
                                                --max_tokens_in_paged_kv_cache 256064 \
                                                --data_dir 128k_context \
                                                --output_dir 128k_context_tp8'

# output would be like
# 6it [00:00, 15354.38it/s]
# [07/22/2024-19:09:54] [TRT-LLM] [I] Evaluation takes: 372.16685914993286 sec.
# [07/22/2024-19:09:54] [TRT-LLM] [I] accuracy of 6 examples: 1.0

# Long context test for 64k
python ./examples/infinitebench/construct_synthetic_dataset.py --test_case build_passkey --test_level 3; mkdir -p 64k_context; mv passkey.jsonl 64k_context;

srun --mpi pmi2 -N 1 -n 8 --ntasks-per-node 8 --container-image <your container>  \
--container-mounts <your container mount> \
--container-name llama-3.1-405b \
--container-workdir <your container work directory> \
bash -c 'python ./examples/eval_long_context.py --task passkey \
                                                --engine_dir llama_3.1_405B_HF_model/trt_engines/tp8-pp1/ \
                                                --tokenizer_dir llama_3.1_405B_HF_model/ \
                                                --stop_idx 6 \
                                                --max_input_length 64000 \
                                                --enable_chunked_context \
                                                --kv_cache_free_gpu_memory_fraction 0.999 \
                                                --max_tokens_in_paged_kv_cache 65064 \
                                                --data_dir 64k_context \
                                                --output_dir 64k_context_tp8'

# Long context test for 64k
srun --mpi pmi2 -N 1 -n 8 --ntasks-per-node 8 --container-image <your container>  \
--container-mounts <your container mount> \
--container-name llama-3.1-405b \
--container-workdir <your container work directory> \
bash -c 'python ./examples/eval_long_context.py --task passkey \
                                                --engine_dir llama_3.1_405B_HF_FP8_model/trt_engines/tp8-pp1/ \
                                                --tokenizer_dir llama_3.1_405B_HF_FP8_model/ \
                                                --stop_idx 6 \
                                                --max_input_length 64000 \
                                                --enable_chunked_context \
                                                --kv_cache_free_gpu_memory_fraction 0.999 \
                                                --max_tokens_in_paged_kv_cache 65064 \
                                                --data_dir 64k_context \
                                                --output_dir 64k_context_tp8'
```

The following script shows how to run evaluation on MMLU tasks:

```bash
srun --mpi pmi2 -N 2 -n 16 --ntasks-per-node 8 --container-image <your container>  \
--container-mounts <your container mount> \
--container-name llama-3.1-405b \
--container-workdir <your container work directory> \
bash -c 'python ./examples/mmlu.py --test_trt_llm \
                                   --engine_dir llama_3.1_405B_meta_model/trt_engines/tp8-pp2/ \
                                   --tokenizer_dir llama_3.1_405B_HF_model/ \
                                   --enable_chunked_context \
                                   --kv_cache_free_gpu_memory_fraction 0.999 \
                                   --max_tokens_in_paged_kv_cache 256064'

srun --mpi pmi2 -N 1 -n 8 --ntasks-per-node 8 --container-image <your container>  \
--container-mounts <your container mount> \
--container-name llama-3.1-405b \
--container-workdir <your container work directory> \
bash -c 'python ./examples/mmlu.py --test_trt_llm \
                                   --engine_dir llama_3.1_405B_HF_model/trt_engines/tp8-pp1/ \
                                   --tokenizer_dir llama_3.1_405B_HF_model/ \
                                   --enable_chunked_context \
                                   --kv_cache_free_gpu_memory_fraction 0.999 \
                                   --max_tokens_in_paged_kv_cache 65064'

srun --mpi pmi2 -N 1 -n 8 --ntasks-per-node 8 --container-image <your container>  \
--container-mounts <your container mount> \
--container-name llama-3.1-405b \
--container-workdir <your container work directory> \
bash -c 'python ./examples/mmlu.py --test_trt_llm \
                                   --engine_dir llama_3.1_405B_HF_FP8_model/trt_engines/tp8-pp1/ \
                                   --tokenizer_dir llama_3.1_405B_HF_FP8_model/ \
                                   --enable_chunked_context \
                                   --kv_cache_free_gpu_memory_fraction 0.999 \
                                   --max_tokens_in_paged_kv_cache 65064'
```
