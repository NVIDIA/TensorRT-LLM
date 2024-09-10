# EXAONE

This document shows how to build and run a [EXAONE](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct) model in TensorRT-LLM.

The TensorRT-LLM EXAONE implementation is based on the LLaMA model. The implementation can be found in [llama/model.py](../../tensorrt_llm/models/llama/model.py).
See the LLaMA example [`examples/llama`](../llama) for details.

- [EXAONE](#exaone)
  - [Support Matrix](#support-matrix)
  - [Download model checkpoints](#download-model-checkpoints)
  - [TensorRT-LLM workflow](#tensorrt-llm-workflow)
    - [Convert checkpoint and build TensorRT engine(s)](#convert-checkpoint-and-build-tensorrt-engines)
    - [Run Engine](#run-engine)

## Support Matrix
  * FP16
  * BF16
  * INT8 & INT4 Weight-Only

## Download model checkpoints

First, download the HuggingFace FP32 checkpoints of EXAONE model.

```bash
git clone https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct hf_models/exaone
```

## TensorRT-LLM workflow
Next, we build the model with `trtllm-build`.

### Convert checkpoint and build TensorRT engine(s)

As written above, we will use llama's [convert_checkpoint.py](../llama/convert_checkpoint.py) for EXAONE model.
```bash
# Build a single-GPU float16 engine from HF weights.

# Build the EXAONE model using a single GPU and FP16.
python ../llama/convert_checkpoint.py \
    --model_dir hf_models/exaone \
    --output_dir trt_models/exaone/fp16/1-gpu \
    --dtype float16

trtllm-build \
    --checkpoint_dir trt_models/exaone/fp16/1-gpu \
    --output_dir trt_engines/exaone/fp16/1-gpu \
    --gemm_plugin auto

# Build the EXAONE model using a single GPU and and apply INT8 weight-only quantization.
python ../llama/convert_checkpoint.py \
    --model_dir hf_models/exaone \
    --output_dir trt_models/exaone/fp16_wq_8/1-gpu \
    --use_weight_only \
    --weight_only_precision int8 \
    --dtype float16

trtllm-build \
    --checkpoint_dir trt_models/exaone/fp16_wq_8/1-gpu \
    --output_dir trt_engines/exaone/fp16_wq_8/1-gpu \
    --gemm_plugin auto
```
> **NOTE**: EXAONE model is currently not supported with `--load_by_shard`.


### Run Engine
Test your engine with the [run.py](../run.py) script:

```bash
python3 ../run.py \
    --input_text "When did the first world war end?" \
    --max_output_len=100 \
    --tokenizer_dir hf_models/exaone \
    --engine_dir trt_engines/exaone/fp16/1-gpu

python ../summarize.py \
    --test_trt_llm \
    --data_type fp16 \
    --hf_model_dir hf_models/exaone \
    --engine_dir trt_engines/exaone/fp16/1-gpu
```

For more examples see [`examples/llama/README.md`](../llama/README.md)
