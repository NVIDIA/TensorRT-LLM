# Canary
This document shows how to build and run a NeMO [canary model](https://huggingface.co/nvidia/canary-1b) in TensorRT-LLM on a single GPU.

- [Canary](#canary)
  - [Overview](#overview)
  - [Support Matrix](#support-matrix)
  - [Usage](#usage)
    - [Build TensorRT engine(s)](#build-tensorrt-engines)
    - [Run](#run)
    - [Acknowledgment](#acknowledgment)
  
## Overview

The TensorRT-LLM Canary example code is located in [`examples/canary`](./).

 * [`convert_checkpoint.py`](./convert_checkpoint.py) does the following
   * Export the Fastconformer encoder model to onnx
   * Export the feat basis functions and features (preprocessor) configuration
   * Saves the weights of the Transformer decoder (and softmax) to TRT-LLM format
 * [`conformer_onnx_trt.py`](./conformer_onnx_trt.py) builds the Fast conformer encoder [TensorRT](https://developer.nvidia.com/tensorrt) engine from onnx 
 * `trtllm-build` to build the [TensorRT](https://developer.nvidia.com/tensorrt) Transformer decoder engine needed to run the Canary model.
 * [`run.py`](./run.py) to run the inference on a single wav file, or [a HuggingFace dataset](https://huggingface.co/datasets/librispeech_asr) [\(Librispeech test clean\)](https://www.openslr.org/12).

## Usage
### Build TensorRT engine(s)

```bash
# install requirements first
pip install -r requirements.txt

INFERENCE_PRECISION=bfloat16
WEIGHT_ONLY_PRECISION=int8
MAX_BEAM_WIDTH=4
MAX_BATCH_SIZE=8
engine_dir="engine"
checkpoint_dir=tllm_checkpoint


# Export the canary model TensorRT-LLM format.
python3 convert_checkpoint.py \
                --dtype=${INFERENCE_PRECISION} \
                --model_path <path_to_nemo_checkpoint> \
                ${engine_dir}

# Build the canary encoder model using conformer_onnx_trt.py
python3 conformer_onnx_trt.py \
        --max_BS <max_batch_size> \
        tllm_checkpoint/encoder/encoder.onnx \
        ${engine_dir}


# Build the canary decoder  using trtllm-build
trtllm-build  --checkpoint_dir ${checkpoint_dir}/decoder \
              --output_dir ${engine_dir}/decoder \
              --moe_plugin disable \
              --enable_xqa disable \
              --max_beam_width ${MAX_BEAM_WIDTH} \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --max_seq_len 114 \ #Max number of generated tokens
              --max_input_len 14 \ # Max number of prompt tokens
              --max_encoder_input_len 3000 \
              --gemm_plugin ${INFERENCE_PRECISION} \
              --bert_attention_plugin ${INFERENCE_PRECISION} \
              --gpt_attention_plugin ${INFERENCE_PRECISION}
```

### Run

```bash
# decode a single wav file
python3 run.py --engine_dir ${engine_dir}--name single_wav_test --input_file assets/1221-135766-0002.wav

# decode a whole dataset
python3 run.py --engine_dir ${engine_dir} --dataset hf-internal-testing/librispeech_asr_dummy --enable_warmup --name librispeech_dummy_large_v3

# decode with a manifest file and save to manifest. 
python3 run.py --engine_dir ${engine_dir} --enable_warmup --batch_size=<batch_size> --name <test_name> --manifest_file <path_to_manifest_file>

```
