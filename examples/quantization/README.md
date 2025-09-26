# TensorRT LLM Quantization Toolkit Installation Guide

## Introduction

This document introduces:

- The steps to install the TensorRT LLM quantization toolkit.
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

# Auto quantization(e.g. fp8 + int4_awq + w4a8_awq) using average weights bits 5
python quantize.py --model_dir $MODEL_PATH  --autoq_format fp8,int4_awq,w4a8_awq  --output_dir $OUTPUT_PATH --auto_quantize_bits 5 --tp_size 2

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
    - nvfp4: Weights are quantized to NVFP4 block-wise with size 16. Activation global scale are calibrated.
    - fp8: Weights are quantized to FP8 tensor wise. Activation ranges are calibrated tensor wise.
    - fp8_pc_pt: Weights are quantized to FP8 per-channel. Activation ranges are calibrated and quantized per-token.
    - int8_sq: Weights are smoothed and quantized to INT8 channel wise. Activation ranges are calibrated tensor wise.
    - int4_awq: Weights are re-scaled and block-wise quantized to INT4. Block size is specified by `awq_block_size`.
    - w4a8_awq: Weights are re-scaled and block-wise quantized to INT4. Block size is specified by `awq_block_size`. Activation ranges are calibrated tensor wise.
    - int8_wo: Actually nothing is applied to weights. Weights are quantized to INT8 channel wise when TRTLLM building the engine.
    - int4_wo: Same as int8_wo but in INT4.
    - full_prec: No quantization.
- autoq_format: Specific quantization algorithms are searched in auto quantization. The algorithm must in ['fp8', 'int4_awq', 'w4a8_awq', 'int8_sq'] and you can use ',' to separate more than one quantization algorithms, such as `--autoq_format fp8,int4_awq,w4a8_awq`. Please attention that using int8_sq and fp8 together is not supported.
- auto_quantize_bits: Effective bits constraint for auto quantization. If not set, regular quantization without auto quantization search is applied. Note: it must be set within correct range otherwise it will be set by lowest value if possible. For example, the weights of LLMs have 16 bits defaultly and it results in a weight compression rate of 40% if we set `auto_quantize_bits` to 9.6 (9.6 / 16 = 0.6), which means the average bits of the weights are 9.6 but not 16. However, which format to choose is determined by solving an optimization problem, so you need to generate the according checkpoint manually if you want to customize your checkpoint formats. The format of mixed precision checkpoint is described in detail below.
- output_dir: Path to save the quantized checkpoint.
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
- quantize_lm_head: Enable quantization of lm_head layer. This is only supported for FP8 quantization. Default is false.

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

There are several arguments for the building stage which relate to quantization.
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

### Format of Mixed Precision Checkpoints

ModelOpt can produce a mixed precision TensorRT LLM checkpoint. After producing the quantized checkpoint, you can build engine directly by `trtllm-build` command:
```bash
trtllm-build --checkpoint_dir <mixed-precision-checkpoint> --output_dir $OUTPUT_PATH
```
If you have some special needs about the model weights, such as int4 for MLP and int8 for the rest, you need to generate the checkpoint and config files by yourself.

The `trtllm-build` command consumes the same format of weights, which is presented in [TensorRT LLM checkpoint formats](https://nvidia.github.io/TensorRT-LLM/architecture/checkpoint.html), but has different quantization method for every linear. Therefore, each layer, such as layer30.mlp.fc, layer30.attention.dense, and so on, keeps the same model weights according to the quantization formats in TensorRT LLM checkpoint. What's more, the `quantization` field in `config.json` will be like this:
```
    "quantization": {
        "quant_algo": "MIXED_PRECISION",
        "kv_cache_quant_algo": "FP8" // The quant_algo of KV cache may change
    },
```
There will be another file about per-layer quantization information named `quant_cfg.json` in the same directory, the format of it is like:
```
{
    "quant_algo": "MIXED_PRECISION",
    "kv_cache_quant_algo": "FP8",
    "quantized_layers": { // one more filed presents per-layer's information
        "transformer.layers.0.attention.qkv": {
            "quant_algo": "FP8" // specific algorithm for each linear
        },
        "transformer.layers.0.attention.dense": {
            "quant_algo": "FP8"
        },
        "transformer.layers.0.mlp.fc": {
            "quant_algo": "W4A16_AWQ",
            "group_size": 128,
            "has_zero_point": false,
            "pre_quant_scale": true
        },
        "transformer.layers.0.mlp.proj": {
            "quant_algo": "W8A8_SQ_PER_CHANNEL"
        },
        ...
        "transformer.layers.31.mlp.proj": {
            "quant_algo": "FP8"
        }
    }
}
```

TensorRT LLM will automatically read `quant_cfg.json` after recogniziong the `MIXED_PRECISION` quantization method in `config.json`. All the specific algorithm keeps the same as what in `quantization` field before. If some layers are not listed, they'll be treated as no quantization.

## APIs

[`quantize.py`](./quantize.py) uses the quantization toolkit to calibrate the PyTorch models and export TensorRT LLM checkpoints. Each TensorRT LLM checkpoint contains a config file (in .json format) and one or several rank weight files (in .safetensors format). It will produce one another quantization config for per-layer's information when setting auto quantization. The checkpoints can be directly used by `trtllm-build` command to build TensorRT LLM engines. See this [`doc`](../../docs/source/architecture/checkpoint.md) for more details on the TensorRT LLM checkpoint format.

> *This quantization step may take a long time to finish and requires large GPU memory. Please use a server grade GPU if a GPU out-of-memory error occurs*

> *If the model is trained with multi-GPU with tensor parallelism, the PTQ calibration process requires the same amount of GPUs as the training time too.*


### PTQ (Post Training Quantization)

PTQ can be achieved with simple calibration on a small set of training or evaluation data (typically 128-512 samples) after converting a regular PyTorch model to a quantized model.

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
import modelopt.torch.quantization as mtq
import modelopt.torch.utils.dataset_utils as dataset_utils

model = AutoModelForCausalLM.from_pretrained(...)

# Select the quantization config, for example, FP8
config = mtq.FP8_DEFAULT_CFG

# Prepare the calibration set and define a forward loop
calib_dataloader = DataLoader(...)
calibrate_loop = dataset_utils.create_forward_loop(
    calib_dataloader, dataloader=calib_dataloader
)

# PTQ with in-place replacement to quantized modules
with torch.no_grad():
    mtq.quantize(model, config, forward_loop=calibrate_loop)

# or PTQ with auto quantization
with torch.no_grad():
    model, search_history = mtq.auto_quantize(
        model,
        data_loader=calib_dataloader,
        loss_func=lambda output, batch: output.loss,
        constraints={"effective_bits": auto_quantize_bits}, # The average bits of quantized weights
        forward_step=lambda model, batch: model(**batch),
        quantization_formats=[quant_algo1, quant_algo2,...] + [None],
        num_score_steps=min(
        num_calib_steps=len(calib_dataloader),
            len(calib_dataloader), 128 // batch_size
        ),  # Limit the number of score steps to avoid long calibration time
        verbose=True,
    )
```

### Export Quantized Model

After the model is quantized, it can be exported to a TensorRT LLM checkpoint, which includes

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
