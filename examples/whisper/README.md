# Whisper

This document shows how to build and run a [whisper model](https://github.com/openai/whisper/tree/main) in TensorRT-LLM on a single GPU.
## Overview

The TensorRT-LLM Whisper example code is located in [`examples/whisper`](./). There are three main files in that folder:

 * [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the Whisper model.
 * [`run.py`](./run.py) to run the inference on a single wav file, or [a HuggingFace dataset](https://huggingface.co/datasets/librispeech_asr) [\(Librispeech test clean\)](https://www.openslr.org/12).
 * [`run_faster_whisper.py`](./run_faster_whisper.py) to do benchmark comparison with [Faster Whisper](https://github.com/SYSTRAN/faster-whisper/tree/master).

## Support Matrix
  * FP16
  * INT8

## Usage

The TensorRT-LLM Whisper example code locates at [examples/whisper](./). It takes whisper pytorch weights as input, and builds the corresponding TensorRT engines.

### Build TensorRT engine(s)

Need to prepare the whisper checkpoint first by downloading models from [here](https://github.com/openai/whisper/blob/main/whisper/__init__.py#L27-L28).


```bash
wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
wget --directory-prefix=assets assets/mel_filters.npz https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz
wget --directory-prefix=assets https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0002.wav
# large-v3 model
wget --directory-prefix=assets https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt
```

TensorRT-LLM Whisper builds TensorRT engine(s) from the pytorch checkpoint.

```bash
# install requirements first
pip install -r requirements.txt

# Build the large-v3 model using a single GPU with plugins.
python3 build.py --output_dir whisper_large_v3 --use_gpt_attention_plugin --use_gemm_plugin --use_layernorm_plugin  --use_bert_attention_plugin

# Build the large-v3 model using a single GPU with plugins and weight-only quantization.
python3 build.py --output_dir whisper_large_weight_only --use_gpt_attention_plugin --use_gemm_plugin --use_layernorm_plugin  --use_bert_attention_plugin --use_weight_only
```

### Run

```bash
# choose the engine you build [./whisper_large_v3, ./whisper_large_weight_only]
output_dir=./whisper_large_v3
# decode a single audio file
# If the input file does not have a .wav extension, ffmpeg needs to be installed with the following command:
# apt-get update && apt-get install -y ffmpeg
python3 run.py --name single_wav_test --engine_dir $output_dir --input_file assets/1221-135766-0002.wav
# decode a whole dataset
python3 run.py --engine_dir $output_dir --dataset hf-internal-testing/librispeech_asr_dummy --enable_warmup --name librispeech_dummy_large_v3_plugin
```

### Acknowledgment

This implementation of TensorRT-LLM for Whisper has been adapted from the [NVIDIA TensorRT-LLM Hackathon 2023](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/Hackathon2023) submission of Jinheng Wang, which can be found in the repository [Eddie-Wang-Hackathon2023](https://github.com/Eddie-Wang1120/Eddie-Wang-Hackathon2023) on GitHub. We extend our gratitude to Jinheng for providing a foundation for the implementation.
