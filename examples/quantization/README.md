# TensorRT-LLM Quantization Toolkit Installation Guide

## Introduction

This document introduces:

- The steps to install the TensorRT-LLM quantization toolkit.
- The Python APIs to quantize the models.

The detailed LLM quantization recipe is distributed to the README.md of the corresponding model examples.

## Installation

The NVIDIA TensorRT Model Optimizer quantization toolkit is installed automatically as a dependency of TensorRT-LLM.

```bash
# Install the additional requirements
cd examples/quantization
pip install -r requirements.txt
```

## Usage

```bash
# FP8 quantization.
python quantize.py --model_dir $MODEL_PATH --qformat fp8 --kv_cache_dtype fp8 --output_dir $OUTPUT_PATH

# INT4_AWQ tp4 quantization.
python quantize.py --model_dir $MODEL_PATH --qformat int4_awq --awq_block_size 64 --tp_size 4 --output_dir $OUTPUT_PATH

# INT8 SQ with INT8 kv cache.
python quantize.py --model_dir $MODEL_PATH --qformat int8_sq --kv_cache_dtype int8 --output_dir $OUTPUT_PATH

# FP8 quantization for NeMo model.
python quantize.py --nemo_ckpt_path nemotron-3-8b-base-4k/Nemotron-3-8B-Base-4k.nemo \
                   --dtype bfloat16 \
                   --batch_size 64 \
                   --qformat fp8 \
                   --output_dir nemotron-3-8b/trt_ckpt/fp8/1-gpu

# FP8 quantization for Medusa model.
python quantize.py --model_dir $MODEL_PATH\
                   --dtype float16 \
                   --qformat fp8 \
                   --kv_cache_dtype fp8 \
                   --output_dir $OUTPUT_PATH \
                   --calib_size 512 \
                   --tp_size 1 \
                   --medusa_model_dir /path/to/medusa_head/ \
                   --num_medusa_heads 4
```
Checkpoint saved in `output_dir` can be directly passed to `trtllm-build`.

### Quantization Arguments:

- model_dir: Hugging Face model path.
- qformat: Specify the quantization algorithm applied to the checkpoint.
    - fp8: Weights are quantized to FP8 tensor wise. Activation ranges are calibrated tensor wise.
    - int8_sq: Weights are smoothed and quantized to INT8 channel wise. Activation ranges are calibrated tensor wise.
    - int4_awq: Weights are re-scaled and block-wise quantized to INT4. Block size is specified by `awq_block_size`.
    - w4a8_awq: Weights are re-scaled and block-wise quantized to INT4. Block size is specified by `awq_block_size`. Activation ranges are calibrated tensor wise.
    - int8_wo: Actually nothing is applied to weights. Weights are quantized to INT8 channel wise when TRTLLM building the engine.
    - int4_wo: Same as int8_wo but in INT4.
    - full_prec: No quantization.
- output_dir Path to save the quantized checkpoint.
- dtype: Specify data type of model when loading from Hugging Face.
- kv_cache_dtype: Specify kv cache data type.
    - int8: Use int8 kv cache.
    - fp8: Use FP8 kv cache.
    - None (default): Use kv cache as model dtype.
- batch_size: Batch size for calibration. Default is 1.
- calib_size: Number of samples. Default is 512.
- calib_max_seq_length: Max sequence length of calibration samples. Default is 512.
- tp_size: Checkpoint is tensor paralleled by tp_size. Default is 1.
- pp_size: Checkpoint is pipeline paralleled by pp_size. Default is 1.
- awq_block_size: AWQ algorithm specific parameter. Indicate the block size when quantizing weights. 64 and 128 are supported by TRTLLM.

#### NeMo model specific arguments:

- nemo_ckpt_path: NeMo checkpoint path.
- calib_tp_size: TP size for NeMo checkpoint calibration.
- calib_pp_size: PP size for NeMo checkpoint calibration.

#### Medusa specific arguments:

- medusa_model_dir: Model path of medusa.
- quant_medusa_head: Whether to quantize the weights of medusa heads.
- num_medusa_heads: Number of medusa heads.
- num_medusa_layers: Number of medusa layers.
- max_draft_len: Max length of draft.
- medusa_hidden_act: Activation function of medusa.

### Building Arguments:

There are several arguments for building stage which related to quantizaion.
- use_fp8_context_fmha: This is Hopper-only feature. Use FP8 Gemm to calculate the attention operation.

```python
qkv scale = 1.0
FP_O = quantize(softmax(FP8_Q * FP8_K), scale=1.0) * FP8_V
FP_O * output_scale = FP8_O
```

### Checkpoint Conversion Arguments (not supported by all models)

- FP8
    - use_fp8_rowwise: Enable FP8 per-token per-channel quantization for linear layer. (FP8 from `quantize.py` is per-tensor).
- INT8
    - smoothquant: Enable INT8 quantization for linear layer. Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf) to Smoothquant the model, and output int8 weights. A good first try is 0.5. Must be in [0, 1].
    - per_channel: Using per-channel quantization for weight when `smoothquant` is enabled.
    - per_token: Using per-token quantization for activation when `smoothquant` is enabled.
- Weight-Only
    - use_weight_only: Weights are quantized to INT4 or INT8 channel wise.
    - weight_only_precision: Indicate `int4` or `int8` when `use_weight_only` is enabled. Or `int4_gptq` when `quant_ckpt_path` is provided which means checkpoint is for GPTQ.
    - quant_ckpt_path: Path of a GPTQ quantized model checkpoint in `.safetensors` format.
    - group_size: Group size used in GPTQ quantization.
    - per_group: Should be enabled when load from GPTQ.
- KV Cache
    - int8_kv_cache: By default, we use dtype for KV cache. int8_kv_cache chooses int8 quantization for KV cache.
    - fp8_kv_cache: By default, we use dtype for KV cache. fp8_kv_cache chooses fp8 quantization for KV cache.

## APIs

[`quantize.py`](./quantize.py) uses the quantization toolkit to calibrate the PyTorch models and export TensorRT-LLM checkpoints. Each TensorRT-LLM checkpoint contains a config file (in .json format) and one or several rank weight files (in .safetensors format). The checkpoints can be directly used by `trtllm-build` command to build TensorRT-LLM engines. See this [`doc`](../../docs/source/architecture/checkpoint.md) for more details on the TensorRT-LLM checkpoint format.

> *This quantization step may take a long time to finish and requires large GPU memory. Please use a server grade GPU if a GPU out-of-memory error occurs*

> *If the model is trained with multi-GPU with tensor parallelism, the PTQ calibration process requires the same amount of GPUs as the training time too.*


### PTQ (Post Training Quantization)

PTQ can be achieved with simple calibration on a small set of training or evaluation data (typically 128-512 samples) after converting a regular PyTorch model to a quantized model.

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
import modelopt.torch.quantization as atq

model = AutoModelForCausalLM.from_pretrained(...)

# Select the quantization config, for example, FP8
config = atq.FP8_DEFAULT_CFG


# Prepare the calibration set and define a forward loop
calib_dataloader = DataLoader(...)
def calibrate_loop():
    for data in calib_dataloader:
        model(data)


# PTQ with in-place replacement to quantized modules
with torch.no_grad():
    atq.quantize(model, config, forward_loop=calibrate_loop)
```

### Export Quantized Model

After the model is quantized, it can be exported to a TensorRT-LLM checkpoint, which includes

- One json file recording the model structure and metadata, and
- One or several rank weight files storing quantized model weights and scaling factors.

The export API is

```python
from modelopt.torch.export import export_tensorrt_llm_checkpoint

with torch.inference_mode():
    export_tensorrt_llm_checkpoint(
        model,  # The quantized model.
        decoder_type,  # The type of the model as str, e.g gptj, llama or gptnext.
        dtype,  # The exported weights data type as torch.dtype.
        export_dir,  # The directory where the exported files will be stored.
        inference_tensor_parallel=tp_size,  # The tensor parallelism size for inference.
        inference_pipeline_parallel=pp_size,  # The pipeline parallelism size for inference.
    )
```
