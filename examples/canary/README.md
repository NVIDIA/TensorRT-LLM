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

INFERENCE_PRECISION=bfloat16 # precision float16 or bfloat16
MAX_BEAM_WIDTH=4 # max beam width of decoder
MAX_BATCH_SIZE=8 # max batch size
MAX_FEAT_LEN=3001 #Max audio duration(ms)/10ms (window shift). Assuming 30s audio
MAX_ENCODER_OUTPUT_LEN=376 #MAX_ENCODER_OUTPUT_LEN = 1 + (MAX_FEAT_LEN / 8), 8 is subsampling factor for canary conformer
MAX_TOKENS=196 # Max number of tokens to generate
MAX_PROMPT_TOKENS=10 # Max number of tokens to be passed


engine_dir="engine"_${INFERENCE_PRECISION}
checkpoint_dir="tllm_checkpoint"_${INFERENCE_PRECISION}
NEMO_MODEL="nvidia/canary-1b-flash"



# Export the canary model TensorRT-LLM format.
python3 convert_checkpoint.py \
                --dtype=${INFERENCE_PRECISION} \
                --model_name ${NEMO_MODEL} \
                --output_dir ${checkpoint_dir} \
                ${engine_dir}


# Build the canary encoder model using conformer_onnx_trt.py
python3 conformer_onnx_trt.py \
        --max_BS ${MAX_BATCH_SIZE} \
        --max_feat_len ${MAX_FEAT_LEN} \
        ${checkpoint_dir} \
        ${engine_dir}


# Build the canary decoder  using trtllm-build
trtllm-build  --checkpoint_dir ${checkpoint_dir}/decoder \
              --output_dir ${engine_dir}/decoder \
              --moe_plugin disable \
              --max_beam_width ${MAX_BEAM_WIDTH} \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --max_seq_len ${MAX_TOKENS} \
              --max_input_len ${MAX_PROMPT_TOKENS} \
              --max_encoder_input_len ${MAX_ENCODER_OUTPUT_LEN}  \
              --gemm_plugin ${INFERENCE_PRECISION} \
              --bert_attention_plugin disable \
              --gpt_attention_plugin ${INFERENCE_PRECISION} \
              --remove_input_padding enable
```

### Run

```bash
# decode a single wav file
python3 run.py --engine_dir ${engine_dir} --name single_wav_test --batch_size=1 --num_beam=<beam_len> --enable_warmup --input_file assets/1221-135766-0002.wav

# decode a whole dataset
python3 run.py --engine_dir ${engine_dir} --dataset hf-internal-testing/librispeech_asr_dummy --enable_warmup  --batch_size=<batch_size> --num_beam=<beam_len>  --name librispeech_dummy_large_v3

# decode with a manifest file and save to manifest.
python3 run.py --engine_dir ${engine_dir} --enable_warmup --batch_size=<batch_size> --num_beam=<beam_len> --name <test_name> --manifest_file <path_to_manifest_file>

```
