# TensorRT-LLM Quantization Toolkit Installation Guide

## Introduction

This document introduces:

- The steps to install the TensorRT-LLM quantization toolkit.
- The Python APIs to quantize the models.

The detailed LLM quantization recipe is distributed to the README.md of the corresponding model examples.

## Installation

1. If the dev environment is a docker container, please launch the docker with the following flags

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --shm-size=20g -it <the docker image with TensorRT-LLM installed> bash
```

2. Install the quantization toolkit `modelopt` and the related dependencies on top of the TensorRT-LLM installation or docker file.

```bash
# Install Modelopt
pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com nvidia-modelopt==0.9.3
# Install the additional requirements
cd <this example folder>
pip install -r requirements.txt
```

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
