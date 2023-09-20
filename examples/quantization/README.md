# Huggingface/NeMo Model Quantization Examples

## What's This Example Folder About?

This folder demonstrates how TensorRT-LLM quantizes an LLM and deployes the quantized LLM.

The examples below uses the NVIDIA AMMO (AlgorithMic Model Optimization) toolkit for the model quantization process.

This document introduces:

- The scripts to quantize, convert and evaluate LLMs,
- The Python code and APIs to quantize and deploy the models.

## Model Quantization and TRT LLM Conversion

### Preparation

1. If the dev environment is a docker container, please launch the docker with the following flags

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --shm-size=20g -it <the docker image with TensorRT-LLM installed> bash
```

2. Install the quantization library `ammo` and the related dependencies on top of the TensorRT-LLM installation or docker file.

```bash
# Obtain the cuda version from the system. Assuming nvcc is available in path.
cuda_version=$(nvcc --version | grep 'release' | awk '{print $6}' | awk -F'[V.]' '{print $2$3}')
# Obtain the python version from the system.
python_version=$(python3 --version 2>&1 | awk '{print $2}' | awk -F. '{print $1$2}')
# Download and install the AMMO package from the DevZone.
wget https://developer.nvidia.com/downloads/assets/cuda/files/nvidia-ammo/nvidia_ammo-0.1.0.tar.gz
tar -xzf nvidia_ammo-0.1.0.tar.gz
pip install nvidia_ammo-0.1.0/nvidia_ammo-0.1.0+cu$cuda_version-cp$python_version-cp$python_version-linux_x86_64.whl
# Install the additional requirements
cd <this example folder>
pip install -r requirements.txt
```

3. (Optional) Download the model checkpoint

   For Llama-2, check: [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)

   For NeMo GPTNext, check: [GPT-2B-001](https://huggingface.co/nvidia/GPT-2B-001)

### All-in-one Scripts for Quantization and Building

There are two quantization schemes supported in the example scripts:

1. The [FP8 format](https://developer.nvidia.com/blog/nvidia-arm-and-intel-publish-fp8-specification-for-standardization-as-an-interchange-format-for-ai/) is available on the Hopper and Ada GPUs with [CUDA compute capability](https://developer.nvidia.com/cuda-gpus) greater than or equal to 8.9.

1. The [INT8 SmoothQuant](https://arxiv.org/abs/2211.10438), developed by MIT HAN Lab and NVIDIA, is designed to reduce both the GPU memory footprint and inference latency of LLM inference.

The following scripts provide an all-in-one and step-by-step model quantization example for GPT-J, LlAMA-2 and NeMo GPTNext

```bash
cd <this example folder>
```

For [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6b):

```bash
scripts/gptj_example.sh  [fp8|int8_sq]
```

For [Llama-2](https://huggingface.co/meta-llama):

```bash
export LLAMA_PATH=<the downloaded LLaMA checkpoint from the Hugging Face hub>
scripts/llama2_example.sh  [fp8|int8_sq]
```

For NeMo [GPTNext](https://huggingface.co/nvidia/GPT-2B-001):

```bash
scripts/gptnext_example.sh  [fp8|int8_sq]
```

> *If GPU out-of-memory error is reported running the scripts, please try editing the scripts and reducing the max batch size of the TensorRT-LLM engine to save GPU memory.*

Please refer to the following section about the stage executed inside the script and the outputs per stage.

## Technical Details

### Quantization

[`hf_ptq.py`](./hf_ptq.py) and [`nemo_ptq.py`](nemo_ptq.py) will use AMMO to calibrate the PyTorch models, and generate a model config, saved as a json (for the model structure) and npz files (for the model weights) that TensorRT-LLM could parse. The model config includes everything needed by TensorRT-LLM to build the TensorRT inference engine, as explained below.

> *This quantization step may take a long time to finish and requires large GPU memory. Please use a server grade GPU if a GPU out-of-memory error occurs*

> *If the model is trained with multi-GPU with tensor parallelism, the PTQ calibration process requires the same amount of GPUs as the training time too.*

### TensorRT-LLM Engine Build

The script [`ammo_to_tensorrt_llm.py`](ammo_to_tensorrt_llm.py) constructs the TensorRT-LLM network and builds the TensorRT-LLM engine for deployment using the quantization outputs model config files from the previous step. The generated engine(s) will be saved as .engine file(s), ready for deployment.

### TensorRT-LLM Engine Validation

The [`summarize.py`](summarize.py) script can be used to test the accuracy and latency on [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset. For each summary, the script can compute the [ROUGE](<https://en.wikipedia.org/wiki/ROUGE_(metric)>) scores. The `ROUGE-1` score is used for implementation validation.

When the script finishes, it will report the latency and the ROUGE score.

The TensorRT-LLM engine and Hugging Face model evaluations are reported in separate stages.

> *By default, the evaluation only runs on 20 data samples. For accurate accuracy evaluation, it is recommended to use higher data sample counts, e.g. 2000 and above. Please modify the script and increase max_ite accordingly to customize the evaluation sample count.*

## APIs

### PTQ (Post Training Quantization)

PTQ can be achieved with simple calibration on a small set of training or evaluation data (typically 128-512 samples) after converting a regular PyTorch model to a quantized model.

```python
import ammo.torch.quantization as atq

model = AutoModelForCausalLM.from_pretrained("...")

# Select the quantization config, for example, INT8 Smooth Quant
config = atq.INT8_SMOOTHQUANT_CFG


# Prepare the calibration set and define a forward loop
def forward_loop():
    for data in calib_set:
        model(data)


# PTQ with in-place replacement to quantized modules
with torch.no_grad():
    atq.quantize(model, config, forward_loop)
```

### Export Quantized Model

After the model is quantized, the model config can be stored. The model config files include all the information needed by TensorRT-LLM to generate the deployable engine.

The exported model config are stored as

- A single JSON file recording the model structure and metadata and
- A group of npz files each recording the local calibrated model on a single GPU rank (model weights, scaling factors per GPU).

The export API is

```python
from ammo.torch.export.model_config import export_model_config

with torch.inference_mode():
    export_model_config(
        model,  # The quantized model.
        decoder_type,  # The type of the model, e.g gptj, llama or gptnext.
        dtype,  # The exported weights data type.
        quantization,  # The quantization algorithm applied, e.g. fp8 or int8_sq.
        export_dir,  # The directory where the exported files will be stored.
    )
```

### Convert to TensorRT-LLM

AMMO offers a single API to build the exported model from the quantization stage.

```python
from ammo.deploy.llm import load_model_configs, model_config_to_tensorrt_llm

# First, we load the exported model config (JSON) file back
# to the model_config(s) in memory representation.
# User can also define a smaller tensor parallel world size for inference
# other than the training or calibration time tensor parallel world size.
# If not specificed, we merge all tensor parallel to a single GPU.
model_configs = load_model_configs(model_config_json, inference_tensor_parallel)

# Then, we call the following API to convert the model_configs to TensorRT-LLM engine(s).
# User can specify:
# 1) The target number of GPUs to use during deployment and inference (gpus <= len(model_configs)).
# 2) the maximum input and output length the engine needs to support as well as the max batch size.
# max_beam_width for beam search is hardcoded as 1 in this example.
model_config_to_tensorrt_llm(
    model_configs,
    engine_dir,
    gpus=gpus,
    max_input_len=max_input_len,
    max_output_len=max_output_len,
    max_batch_size=max_batch_size,
)
```
