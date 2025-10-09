# Sample Weight-Stripping

## Table Of Contents

- [Overview](#overview)
   * [Build Weights Stripped Engine](#build-weights-stripped-engine)
   * [Engine Refitter](#engine-refitter)
- [Prerequisites](#prerequisites)
- [Weight-Stripping Workflow Example](#weight-stripping-workflow-example)
   * [GPT-J](#gpt-j)
   * [Llama-7b INT4](#llama-7b-int4)
   * [Llama-7b FP16 + WoQ INT8](#llama-7b-fp16-woq-int8)
   * [Llama2-70b FP8 with TP=2](#llama2-70b-fp8-with-tp2)
- [Engine Plan File Size Results](#engine-plan-file-size-results)
- [Prototype](#prototype)
   * [Checkpoint Pruner](#checkpoint-pruner)
      * [Pruning a TensorRT LLM Checkpoint](#pruning-a-tensorrt-llm-checkpoint)

## Overview

This workflow introduces a new script `trtllm-refit`. `trtllm-refit` allows you to refit the generated engine with weights from any TensorRT LLM checkpoint matching the same architecture, so long as you build the engine as refittable or stripped.

### Build Weights Stripped Engine
TensorRT can generate refittable engines with the same performance as the non-refittable ones when TensorRT builder optimize under the assumption that the engine will be refitted with weights identical to those provide at build time. Those refittable weights can be stripped to reduce the engine plan file size, with the option to subsequently supply them via the refit interface.

New option `--strip_plan` is introduced in `trtllm-build`

```bash
trtllm-build --strip_plan --checkpoint_dir ${CHECKPOINT_DIR} --output_dir ${ENGINE_DIR} ...
```

### Engine Refitter
The refitter allows you to refit an engine with weights in a TensorRT LLM checkpoint. It does this by doing a textual match between engine and checkpoint weight names. In order for the refitter to work, the engine must be built with refitting enabled. This can be accomplished by passing `--strip_plan` to `trtllm-build`.

After building a stripped engine via `trtllm-build`, run

```bash
trtllm-refit --checkpoint_dir ${CHECKPOINT_DIR} --engine_dir ${ENGINE_DIR}
```


## Prerequisites

Install [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/README.md) either through [pip](https://github.com/NVIDIA/TensorRT-LLM/blob/main/README.md#installation) or [from the source](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/installation/build-from-source-linux.md).

## Weight-Stripping Workflow Example

### GPT-J

1. Download the weights.
```bash
# 1. Weights & config
git clone https://huggingface.co/EleutherAI/gpt-j-6b
pushd gpt-j-6b && \
  rm -f pytorch_model.bin && \
  wget https://huggingface.co/EleutherAI/gpt-j-6b/resolve/main/pytorch_model.bin && \
popd

# 2. Vocab and merge table
wget https://huggingface.co/EleutherAI/gpt-j-6b/resolve/main/vocab.json
wget https://huggingface.co/EleutherAI/gpt-j-6b/resolve/main/merges.txt
```

2. Convert the Hugging Face checkpoint into TensorRT LLM format.
Run below command lines in [`examples/models/contrib/gpt`](../gptj) directory.
```bash
# Build a float16 checkpoint using HF weights.
python convert_checkpoint.py --model_dir ./gpt-j-6b \
                                     --dtype float16 \
                                     --output_dir ./trt_ckpt/gptj_fp16_tp1/

# Build an int8 weight-only checkpoint using HF weights.
python convert_checkpoint.py --model_dir ./gpt-j-6b \
                                     --dtype float16 \
                                     --use_weight_only \
                                     --weight_only_precision int8 \
                                     --output_dir ./trt_ckpt/gptj_int8_tp1/

```

3. Build the weights stripped engine.
```bash
# Build with --strip_plan. Requires TRT>=10.0.0
trtllm-build --checkpoint_dir ./trt_ckpt/gptj_fp16_tp1/ \
             --output_dir ./trt_engines/gptj_fp16_tp1/ \
             --gemm_plugin float16 \
             --max_batch_size=32 \
             --max_input_len=1919 \
             --max_seq_len=2047 \
             --strip_plan
```

4. Refit the engine. The refit engine lives at `${ENGINE_DIR}.refit`.
```bash
# --checkpoint_dir points to the path of the weights you want refit, in this case the original weights.
trtllm-refit --checkpoint_dir ./trt_ckpt/gptj_fp16_tp1/ --engine_dir ./trt_engines/gptj_fp16_tp1/ --output_dir ./trt_engines/gptj_fp16_tp1.refit/
```

5. Verify the engine.
```bash
# Run the summarization task.
python3 ../summarize.py --engine_dir ./trt_engines/gptj_fp16_tp1.refit \
                        --hf_model_dir ./gpt-j-6b \
                        --batch_size 1 \
                        --test_trt_llm \
                        --tensorrt_llm_rouge1_threshold 14 \
                        --data_type fp16 \
                        --check_accuracy
```

### Llama-7b INT4

1. Download the llama-7b-hf checkpoint and saved in /llm-models/llama-models/llama-7b-hf/.

2. Calibrate the checkpoint and convert into TensorRT LLM format.
Run below command lines in [`examples/models/core/llama`](../llama) directory.
```bash
# Calibrate INT4 using AMMO.
python ../quantization/quantize.py --model_dir  /llm-models/llama-models/llama-7b-hf/ \
               --dtype float16 \
               --qformat int4_awq \
               --awq_block_size 128 \
               --output_dir ./quantized_int4-awq \
               --calib_size 32
```

3. Build the weights stripped engine.
```bash
# Build with --strip_plan. Requires TRT>=10.0.0
trtllm-build --checkpoint_dir ./quantized_int4-awq \
                --strip_plan \
                --gemm_plugin float16 \
                --output_dir trt_int4_AWQ
```

4. Refit the engine.
```bash
trtllm-refit --checkpoint_dir ./quantized_int4-awq \
                --engine_dir trt_int4_AWQ \
                --output_dir trt_int4_AWQ_full_from_wtless
```

5. Verify the engine.
```bash
python3 ../summarize.py --engine_dir trt_int4_AWQ_full_from_wtless \
                --hf_model_dir /llm-models/llama-models/llama-7b-hf/ \
                --batch_size 1 \
                --test_trt_llm \
                --check_accuracy
```

### Llama-7b FP16 + WoQ INT8

1. Download the llama-7b-hf checkpoint and saved in /llm-models/llama-models/llama-7b-hf/.

2. Convert the checkpoint into TensorRT LLM format.
Run below command lines in [`examples/models/core/llama`](../llama) directory.
```bash
python3 convert_checkpoint.py --model_dir /llm-models/llama-models/llama-7b-hf/ \
                --output_dir ./llama-7b-hf-fp16-woq \
                --dtype float16 \
                --use_weight_only \
                --weight_only_precision int8
```

3. Build the weights stripped engine.
```bash
# Build with --strip_plan. Requires TRT>=10.0.0
trtllm-build --checkpoint_dir ./llama-7b-hf-fp16-woq \
                --output_dir ./engines/llama-7b-hf-fp16-woq-1gpu-wtless \
                --strip_plan \
                --gemm_plugin float16
```

4. Refit the engine.
```bash
trtllm-refit --checkpoint_dir ./llama-7b-hf-fp16-woq \
                --engine_dir ./engines/llama-7b-hf-fp16-woq-1gpu-wtless \
                --output_dir ./engines/llama-7b-hf-fp16-woq-1gpu-wtless-to-full
```

5. Verify the engine.
```bash
python3 ../summarize.py --engine_dir ./engines/llama-7b-hf-fp16-woq-1gpu-wtless-to-full \
                --hf_model_dir /llm-models/llama-models/llama-7b-hf/ \
                --batch_size 1 \
                --test_trt_llm \
                --check_accuracy
```


### Llama2-70b FP8 with TP=2

1. Download the llama-v2-70b-hf checkpoint and saved in /llm-models/llama-models-v2/llama-v2-70b-hf/.

2. Calibrate the checkpoint and convert into TensorRT LLM format.
Run below command lines in [`examples/models/core/llama`](../llama) directory.
```bash
# Calibrate FP8 using AMMO.
python ../quantization/quantize.py --model_dir /llm-models/llama-models-v2/llama-v2-70b-hf/ \
               --dtype float16 \
               --qformat fp8 \
               --kv_cache_dtype fp8 \
               --output_dir ./llama2-70b-hf-fp8-tp2 \
               --calib_size 512 \
               --tp_size 2
```

3. Build the weights stripped engine.
```bash
trtllm-build --checkpoint_dir ./llama2-70b-hf-fp8-tp2 \
                --output_dir engines/llama2-70b-hf-fp8-tp2 \
                --gemm_plugin float16 \
                --workers 2
```

4. Refit the engine.
```bash
trtllm-refit --checkpoint_dir ./llama2-70b-hf-fp8-tp2 \
                --engine_dir engines/llama2-70b-hf-fp8-tp2 \
                --output_dir engines/llama2-70b-hf-fp8-tp2.refit
```

5. Verify the engine.
```bash
python3 ../summarize.py --engine_dir engines/llama2-70b-hf-fp8-tp2.refit \
                --hf_model_dir /llm-models/llama-models-v2/llama-v2-70b-hf/ \
                --batch_size 1 \
                --test_trt_llm \
                --check_accuracy
```


## Engine Plan File Size Results

| **Model** | **Full Engine Plan Size** | **Weight-Stripped Engine Plan Size** |
|:---------:|:----------:|:----:|
|llama-7b INT4 | 3.7GB | 5.3MB |
|llama-7b FP16 + WoQ INT8 | 6.54GB | 28.69MB |
|llama2-70b FP8 + TP=2 | 64.78GB | 60.61MB |

## Prototype
### Checkpoint Pruner
The checkpoint pruner allows you to strip `Conv` and `Gemm` weights out of a TensorRT LLM [checkpoint](https://nvidia.github.io/TensorRT-LLM/latest/architecture/checkpoint.html). Since these make up the vast majority of weights, the pruner will decrease the size of your checkpoint up to 99%.

When building an engine with a pruned checkpoint, TensorRT LLM fills in the missing weights with random ones. These weights should later be [refit](#engine-refitter) with the original weights to preserve the intended behavior.

Building an engine from a pruned checkpoint will also allow the engine to be [refit](#engine-refitter).

#### Pruning a TensorRT LLM Checkpoint

1. Install [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/README.md) either through [pip](https://github.com/NVIDIA/TensorRT-LLM/blob/main/README.md#installation) or [from the source](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/installation/build-from-source-linux.md).
2. Download a model of your choice and convert it to a TensorRT LLM checkpoint ([llama instructions](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/llama/README.md#usage)).
3. (Optional) Run the `trtllm-prune` command.
```bash
# Prunes the TRT-LLM checkpoint at ${CHECKPOINT_DIR}, and stores it in the directory ${CHECKPOINT_DIR}.pruned
trtllm-prune --checkpoint_dir ${CHECKPOINT_DIR}
```

The pruned checkpoint lives at `${CHECKPOINT_DIR}.pruned` by default, however, this can be overridden by issuing the `--out_dir` flag.

4. Build the stripped engine.

```bash
# From pruned checkpoint.
trtllm-build --checkpoint_dir ${CHECKPOINT_DIR}.pruned \
             --output_dir ${ENGINE_OUT_DIR} \
             ${EXTRA_ARGS}
```
