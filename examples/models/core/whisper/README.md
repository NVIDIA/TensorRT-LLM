# Whisper

This document shows how to build and run a [whisper model](https://github.com/openai/whisper/tree/main) in TensorRT-LLM on a single GPU.

- [Whisper](#whisper)
  - [Overview](#overview)
  - [Support Matrix](#support-matrix)
  - [Usage](#usage)
    - [Build TensorRT engine(s)](#build-tensorrt-engines)
    - [Run](#run)
      - [Run C++ runtime](#run-c-runtime)
      - [Run Python runtime](#run-python-runtime)
      - [Advanced Usage](#advanced-usage)
    - [Distil-Whisper](#distil-whisper)
    - [Acknowledgment](#acknowledgment)

## Overview

The TensorRT-LLM Whisper example code is located in [`examples/models/core/whisper`](./).

 * `trtllm-build` to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the Whisper model.
 * [`run.py`](./run.py) to run the inference on a single wav file, or [a HuggingFace dataset](https://huggingface.co/datasets/openslr/librispeech_asr) [\(Librispeech test clean\)](https://www.openslr.org/12).

## Support Matrix
  * FP16
  * FP8
  * INT8 (Weight Only Quant)
  * INT4 (Weight Only Quant)

## Usage

The TensorRT-LLM Whisper example code locates at [examples/models/core/whisper](./). It takes whisper pytorch weights as input, and builds the corresponding TensorRT engines.

### Build TensorRT engine(s)


```bash
wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
wget --directory-prefix=assets assets/mel_filters.npz https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz
wget --directory-prefix=assets https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0002.wav
```

TensorRT-LLM Whisper builds TensorRT engine(s) from the huggingface pytorch checkpoint.

```bash
# install requirements first
pip install -r requirements.txt

INFERENCE_PRECISION=float16
QUANT_FORMAT=fp8
MAX_BEAM_WIDTH=4
MAX_BATCH_SIZE=8
checkpoint_dir=whisper_large_v3_weights_${QUANT_FORMAT}
output_dir=whisper_large_v3_${QUANT_FORMAT}

# Convert the large-v3 model weights into TensorRT-LLM format in FP8 precision.
# Make sure that you have enough gpu memory for calibration, Weight-Only quantizations
# and full precision do not require calibration
python3 ../quantization/quantize.py \
        --model_dir openai/whisper-large-v3 \
        --qformat fp8 \
        --output_dir ${checkpoint_dir} \
        --calib_dataset peoples_speech \
        --calib_size 100

# Check examples/quantization/README.md for more options and quantization formats

# Build the large-v3 model using trtllm-build
trtllm-build  --checkpoint_dir ${checkpoint_dir}/encoder \
              --output_dir ${output_dir}/encoder \
              --moe_plugin disable \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --gemm_plugin disable \
              --bert_attention_plugin ${INFERENCE_PRECISION} \
              --max_input_len 3000 --max_seq_len=3000

trtllm-build  --checkpoint_dir ${checkpoint_dir}/decoder \
              --output_dir ${output_dir}/decoder \
              --moe_plugin disable \
              --max_beam_width ${MAX_BEAM_WIDTH} \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --max_seq_len 114 \
              --max_input_len 14 \
              --max_encoder_input_len 3000 \
              --gemm_plugin ${INFERENCE_PRECISION} \
              --bert_attention_plugin ${INFERENCE_PRECISION} \
              --gpt_attention_plugin ${INFERENCE_PRECISION}
```

### Run
Different types of runtime are provided for whisper models. Following an order of serving performance and good usability, we recommend:
- (NEW) Python binding of C++ runtime w/ Paged KV Cache and Inflight Batching (IFB)
- Python runtime w/ Static Batching

Please refer to the documentation for the details of [paged kv cache](../../../../docs/source/advanced/gpt-attention.md#paged-kv-cache) and [inflight batching](../../../../docs/source/advanced/gpt-attention.md#inflight-batching).

#### Run C++ runtime
**Note: to use inflight batching and paged kv cache features in C++ runtime, please make sure you have set `--paged_kv_cache enable` and `--remove_input_padding enable` (which is by default enabled) in the `trtllm-build` command. Meanwhile, if using Python runtime, it is recommended to disable these flag by `--paged_kv_cache disable` and `--remove_input_padding disable` to avoid any unnecessary overhead.**

```bash
# choose the engine you build [./whisper_large_v3, ./whisper_large_v3_int8]
output_dir=./whisper_large_v3
# decode a single audio file
# If the input file does not have a .wav extension, ffmpeg needs to be installed with the following command:
# apt-get update && apt-get install -y ffmpeg
# Inferencing via python binding of C++ runtime with inflight batching (IFB)
python3 run.py --name single_wav_test --engine_dir $output_dir --input_file assets/1221-135766-0002.wav
# decode a whole dataset
python3 run.py --engine_dir $output_dir --dataset hf-internal-testing/librispeech_asr_dummy --enable_warmup --name librispeech_dummy_large_v3
```


For pure C++ runtime, there is no example given yet. Please check the [`Executor`](../../../../cpp/include/tensorrt_llm/executor/executor.h) API to implement your own end-to-end workflow. It is highly recommended to leverage more encapsulated solutions such as the above C++ Python binding or [Triton backend](https://github.com/triton-inference-server/tensorrtllm_backend).

<!-- #### Run with Triton Backend
[Triton backend](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/whisper.md) contains the tutorial on how to run whisper engines with Tritonserver. -->

#### Run Python runtime

For pure Python runtime, you can simply add the `--use_py_session` option.

#### Advanced Usage

`--padding_strategy`
OpenAI's official Whisper models accept WAV files of up to 30 seconds in length. For files shorter than 30 seconds, padding is required to reach the 30-second mark, which may not be efficient. Currently, three padding strategies are supported:

1. **max (default)**: Pads to 30 seconds.
2. **longest**: Pads according to the longest duration in the current batch.
3. **nopad**: No padding is applied. You will need to fine-tune the Whisper model to maintain accuracy. See [examples](https://github.com/k2-fsa/icefall/blob/master/egs/aishell/ASR/whisper/whisper_encoder_forward_monkey_patch.py#L15).

`--text_prefix`
You can modify the input prompt for the Whisper decoder. For example, use `<|startoftranscript|><|en|><|zh|><|transcribe|><|notimestamps|>` to perform code-switching ASR between Chinese and English.

`--compute_cer`
Calculates the character error rate (CER) instead of the word error rate (WER) for languages such as Chinese and Japanese.

`--dataset`, `--dataset_name`, and `--dataset_split`
These options allow you to select different decoding audio datasets from Hugging Face.


### Acknowledgment

This implementation of TensorRT-LLM for Whisper has been adapted from the [NVIDIA TensorRT-LLM Hackathon 2023](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/Hackathon2023) submission of Jinheng Wang, which can be found in the repository [Eddie-Wang-Hackathon2023](https://github.com/Eddie-Wang1120/Eddie-Wang-Hackathon2023) on GitHub. We extend our gratitude to Jinheng for providing a foundation for the implementation.
