# Weightless Engine (Experimental)
## Overview

This workflow contains two scripts: `prune.py` and `refit.py`. `prune.py` allows you to prune the majority of weights in a TensorRT-LLM checkpoint. `refit.py` allows you to refit the generated engine with weights from any TensorRT-LLM checkpoint matching the same architecture.

## Prerequisites

Install [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/README.md) either through [pip](https://github.com/NVIDIA/TensorRT-LLM/blob/main/README.md#installation) or [from the source](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/build_from_source.md).

## Weightless Engine Workflow Example

We will use GPT-J to illustrate the end-to-end workflow.

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

2. Convert the Hugging Face checkpoint into TensorRT-LLM format.
```bash
# Build a float16 engine using HF weights.
python ../gptj/convert_checkpoint.py --model_dir ./gpt-j-6b \
                                     --dtype float16 \
                                     --output_dir ./trt_ckpt/gptj_fp16_tp1/

# Build an int8 weight-only engine using HF weights.
python ../gptj/convert_checkpoint.py --model_dir ./gpt-j-6b \
                                     --dtype float16 \
                                     --use_weight_only \
                                     --weight_only_precision int8 \
                                     --output_dir ./trt_ckpt/gptj_int8_tp1/

# FP8, Tensor Parallelism, and Pipeline Parallelism are not currently supported.
```

3. Prune the engine. The pruned checkpoint lives at `${CHECKPOINT_DIR}.pruned`.
```bash
python prune.py --checkpoint_dir ./trt_ckpt/gptj_fp16_tp1/
```

4. Build the engine.
```bash
# Enable several TensorRT-LLM plugins to increase runtime performance. It also helps with build time.
trtllm-build --checkpoint_dir ./trt_ckpt/gptj_fp16_tp1.pruned/ \
             --output_dir ./trt_engines/gptj_fp16_tp1/ \
             --gemm_plugin float16 \
             --max_batch_size=32 \
             --max_input_len=1919 \
             --max_output_len=128
```

5. Refit the engine. The refit engine lives at `${ENGINE_DIR}.refit`.
```bash
# --checkpoint_dir points to the path of the weights you want refit, in this case the original weights.
python refit.py --checkpoint_dir ./trt_ckpt/gptj_fp16_tp1/ --engine_dir ./trt_engines/gptj_fp16_tp1/
```

6. Verify the engine.
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

### Known Issues

The pruner and refitter do not work when the tensor or pipeline parallelism is enabled. FP8 is also not supported at the moment.

## Checkpoint Pruner
The checkpoint pruner allows you to strip `Conv` and `Gemm` weights out of a TensorRT-LLM [checkpoint](./new_workflow.md). Since these make up the vast majority of weights, the pruner will decrease the size of your checkpoint up to 99%.

The checkpoint pruner is mainly meant to support [refitting](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Refitter.html); the underlying TensorRT engine.

When building an engine with a pruned checkpoint, TensorRT-LLM fills in the missing weights with random ones. These weights should later be [refit](#engine-refitter) with the original weights to preserve the intended behavior.

### Pruning a TensorRT-LLM Checkpoint

1. Install [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/README.md) either through [pip](https://github.com/NVIDIA/TensorRT-LLM/blob/main/README.md#installation) or [from the source](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/build_from_source.md).
2. Download a model of your choice and convert it to a TensorRT-LLM checkpoint ([llama instructions](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/llama/README.md#usage)).
3. Run the `prune.py` command.
```bash
# Prunes the TRT-LLM checkpoint at ${CHECKPOINT_DIR}, and stores it in the directory ${CHECKPOINT_DIR}.pruned
python prune.py --checkpoint_dir ${CHECKPOINT_DIR}
```

The pruned checkpoint lives at `${CHECKPOINT_DIR}.pruned` by default, however, this can be overridden by issuing the `--out_dir` flag.

4. Build using the pruned engine.

```
trtllm-build --checkpoint_dir ${CHECKPOINT_DIR}.pruned \
             --output_dir ${ENGINE_OUT_DIR} \
             ${EXTRA_ARGS}
```

This command is the same as the typical build command.

## Engine Refitter
The refitter allows you to refit an engine with weights in a TensorRT-LLM checkpoint. It does this by doing a textual match between engine and checkpoint weight names. In order for the refitter to work, the engine must be built with refitting enabled. This is currently only enabled if you build an engine from a pruned checkpoint.

## Refitting a TensorRT-LLM Engine
After [pruning](#usage) a TensorRT-LLM checkpoint and building an engine via `trtllm-build`, run

```bash
python refit.py --checkpoint_dir ${CHECKPOINT_DIR} --engine_dir ${ENGINE_DIR}
```

This command will _only_ work if the checkpoint used to build the engine with was pruned.
