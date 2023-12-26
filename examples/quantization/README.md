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

2. Install the quantization toolkit `ammo` and the related dependencies on top of the TensorRT-LLM installation or docker file.

```bash
# Obtain the python version from the system.
pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com nvidia-ammo~=0.5.0
# Install the additional requirements
cd <this example folder>
pip install -r requirements.txt
```

## APIs

[`ammo.py`](../../tensorrt_llm/models/quantized/ammo.py) uses the quantization toolkit to calibrate the PyTorch models, and generate a model config, saved as a json (for the model structure) and npz files (for the model weights) that TensorRT-LLM could parse. The model config includes everything needed by TensorRT-LLM to build the TensorRT inference engine, as explained below.

> *This quantization step may take a long time to finish and requires large GPU memory. Please use a server grade GPU if a GPU out-of-memory error occurs*

> *If the model is trained with multi-GPU with tensor parallelism, the PTQ calibration process requires the same amount of GPUs as the training time too.*


### PTQ (Post Training Quantization)

PTQ can be achieved with simple calibration on a small set of training or evaluation data (typically 128-512 samples) after converting a regular PyTorch model to a quantized model.

```python
import ammo.torch.quantization as atq

model = AutoModelForCausalLM.from_pretrained("...")

# Select the quantization config, for example, FP8
config = atq.FP8_DEFAULT_CFG


# Prepare the calibration set and define a forward loop
def forward_loop():
    for data in calib_set:
        model(data)


# PTQ with in-place replacement to quantized modules
with torch.no_grad():
    atq.quantize(model, config, forward_loop)
```

### Export Quantized Model

After the model is quantized, the model config can be stored. The model config files include all the information needed by TensorRT-LLM to generate the deployable engine, including the quantized scaling factors.

The exported model config are stored as

- A single JSON file recording the model structure and metadata and
- A group of npz files each recording the model on a single tensor parallel rank (model weights, scaling factors per GPU).

The export API is

```python
from ammo.torch.export import export_model_config

with torch.inference_mode():
    export_model_config(
        model,  # The quantized model.
        decoder_type,  # The type of the model as str, e.g gptj, llama or gptnext.
        dtype,  # The exported weights data type as torch.dtype.
        export_dir,  # The directory where the exported files will be stored.
        inference_gpus,  # The number of GPUs used in the inference time for tensor parallelism.
    )
```
