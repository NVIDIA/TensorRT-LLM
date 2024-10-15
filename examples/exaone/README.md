# EXAONE

This document shows how to build and run a [EXAONE](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct) model in TensorRT-LLM.

The TensorRT-LLM EXAONE implementation is based on the LLaMA model. The implementation can be found in [llama/model.py](../../tensorrt_llm/models/llama/model.py).
See the LLaMA example [`examples/llama`](../llama) for details.

- [EXAONE](#exaone)
  - [Support Matrix](#support-matrix)
  - [Download model checkpoints](#download-model-checkpoints)
  - [Usage](#usage)
    - [Convert checkpoint and build TensorRT engine(s)](#convert-checkpoint-and-build-tensorrt-engines)
    - [FP8 Post-Training Quantization](#fp8-post-training-quantization)
    - [SmoothQuant](#smoothquant)
    - [Groupwise quantization (AWQ)](#groupwise-quantization-awq)
        - [W4A16 AWQ with FP8 GEMM (W4A8 AWQ)](#w4a16-awq-with-fp8-gemm-w4a8-awq)
    - [Run Engine](#run-engine)

## Support Matrix
  * FP16
  * BF16
  * Tensor Parallel
  * FP8
  * INT8 & INT4 Weight-Only
  * INT8 SmoothQuant
  * INT4 AWQ & W4A8 AWQ

## Download model checkpoints

First, download the HuggingFace FP32 checkpoints of EXAONE model.

```bash
git clone https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct hf_models/exaone
```

## Usage
The next section describe how to convert the weights from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers) format to the TensorRT-LLM format. We will use llama's [convert_checkpoint.py](../llama/convert_checkpoint.py) for EXAONE model and then we build the model with `trtllm-build`.

### Convert checkpoint and build TensorRT engine(s)

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
    --output_dir trt_models/exaone/int8_wq/1-gpu \
    --use_weight_only \
    --weight_only_precision int8 \
    --dtype float16

trtllm-build \
    --checkpoint_dir trt_models/exaone/int8_wq/1-gpu \
    --output_dir trt_engines/exaone/int8_wq/1-gpu \
    --gemm_plugin auto

# Build the EXAONE model using a single GPU and and apply INT4 weight-only quantization.
python ../llama/convert_checkpoint.py \
    --model_dir hf_models/exaone \
    --output_dir trt_models/exaone/int4_wq/1-gpu \
    --use_weight_only \
    --weight_only_precision int4 \
    --dtype float16

trtllm-build \
    --checkpoint_dir trt_models/exaone/int4_wq/1-gpu \
    --output_dir trt_engines/exaone/int4_wq/1-gpu \
    --gemm_plugin auto

# Build the EXAONE model using using 2-way tensor parallelism and FP16.
python ../llama/convert_checkpoint.py \
    --model_dir hf_models/exaone \
    --output_dir trt_models/exaone/fp16/2-gpu \
    --tp_size 2 \
    --dtype float16

trtllm-build \
    --checkpoint_dir trt_models/exaone/fp16/2-gpu \
    --output_dir trt_engines/exaone/fp16/2-gpu \
    --gemm_plugin auto
```
> **NOTE**: EXAONE model is not supported with `--load_by_shard`.

### FP8 Post-Training Quantization

The examples below uses the NVIDIA Modelopt (AlgorithMic Model Optimization) toolkit for the model quantization process.

First make sure Modelopt toolkit is installed (see [examples/quantization/README.md](/examples/quantization/README.md#preparation))

```bash
# Build the EXAONE model using a single GPU and and apply FP8 quantization.
python ../quantization/quantize.py \
    --model_dir hf_models/exaone \
    --dtype float16 \
    --qformat fp8 \
    --kv_cache_dtype fp8 \
    --output_dir trt_models/exaone/fp8/1-gpu \

trtllm-build \
    --checkpoint_dir trt_models/exaone/fp8/1-gpu \
    --output_dir trt_engines/exaone/fp8/1-gpu \
    --gemm_plugin auto
```

### SmoothQuant

The examples below uses the NVIDIA Modelopt (AlgorithMic Model Optimization) toolkit for the model quantization process.

First make sure Modelopt toolkit is installed (see [examples/quantization/README.md](/examples/quantization/README.md#preparation))

```bash
# Build the EXAONE model using a single GPU and and apply INT8 SmoothQuant.
python ../quantization/quantize.py \
    --model_dir hf_models/exaone \
    --dtype float16 \
    --qformat int8_sq \
    --output_dir trt_models/exaone/int8_sq/1-gpu

trtllm-build \
    --checkpoint_dir trt_models/exaone/int8_sq/1-gpu \
    --output_dir trt_engines/exaone/int8_sq/1-gpu \
    --gemm_plugin auto
```

### Groupwise quantization (AWQ)

The examples below uses the NVIDIA Modelopt (AlgorithMic Model Optimization) toolkit for the model quantization process.

First make sure Modelopt toolkit is installed (see [examples/quantization/README.md](/examples/quantization/README.md#preparation))

```bash
# Build the EXAONE model using a single GPU and and apply INT4 AWQ.
python ../quantization/quantize.py \
    --model_dir hf_models/exaone \
    --dtype float16 \
    --qformat int4_awq \
    --output_dir trt_models/exaone/int4_awq/1-gpu

trtllm-build \
    --checkpoint_dir trt_models/exaone/int4_awq/1-gpu \
    --output_dir trt_engines/exaone/int4_awq/1-gpu \
    --gemm_plugin auto
```

#### W4A16 AWQ with FP8 GEMM (W4A8 AWQ)
For Hopper GPUs, TRT-LLM also supports employing FP8 GEMM for accelerating linear layers. This mode is noted with `w4a8_awq` for Modelopt and TRT-LLM, in which both weights and activations are converted from W4A16 to FP8 for GEMM calculation.

Please make sure your system contains a Hopper GPU before trying the commands below.

```bash
# Build the EXAONE model using a single GPU and and apply W4A8 AWQ.
python ../quantization/quantize.py \
    --model_dir hf_models/exaone \
    --dtype float16 \
    --qformat w4a8_awq \
    --output_dir trt_models/exaone/w4a8_awq/1-gpu

trtllm-build \
    --checkpoint_dir trt_models/exaone/w4a8_awq/1-gpu \
    --output_dir trt_engines/exaone/w4a8_awq/1-gpu \
    --gemm_plugin auto
```


### Run Engine
Test your engine with the [run.py](../run.py) script:

```bash
python3 ../run.py \
    --input_text "When did the first world war end?" \
    --max_output_len=100 \
    --tokenizer_dir hf_models/exaone \
    --engine_dir trt_engines/exaone/fp16/1-gpu

# Run with 2 GPUs
mpirun -n 2 --allow-run-as-root \
    python3 ../run.py \
    --input_text "When did the first world war end?" \
    --max_output_len=100 \
    --tokenizer_dir hf_models/exaone \
    --engine_dir trt_engines/exaone/fp16/2-gpu

python ../summarize.py \
    --test_trt_llm \
    --data_type fp16 \
    --hf_model_dir hf_models/exaone \
    --engine_dir trt_engines/exaone/fp16/1-gpu
```

For more examples see [`examples/llama/README.md`](../llama/README.md)
