# Deepseek-v3

This document shows how to build and run the `DeepSeek-v3` model in TensorRT-LLM.


***The current branch is a preview version only for supporting DeepSeek-V3, which will be part of the TRT-LLM official release in 0.18+ version.***


- [Deepseek-V3](#deepseek-v3)
    - [Support Matrix](#support-matrix)
    - [Prerequisite](#prerequisite)
    - [Hardware](#hardware)
    - [Overview](#overview)
    - [Usage](#usage)
        - [Build TensorRT engine(s)](#build-tensorrt-engines)

## Support Matrix

| Model          | FP16  | BF16  |  FP8  | TP  | EP | IB |
| :------------- | :---: | :---: | :---: | :-----: | :-----: | :-----: |
| DeepSeek-V3    |   Y   |   Y   |   Y   |    Y    |    Y    |    Y    |


- TP: Tensor Parallel
- EP: Expert Parallel
- IB: Inflight Batching
- FP8: Support for FP8 is currently in progress and will be released soon

***Please Note:***
- Prefer using BF16 over FP16 for DeepSeek-V3 since model original training precision is FP8 and we found direct convert FP8 -> FP16 may cause unknown accuracy issues.


## Prerequisite

First, please download DeepSeek-V3 weights from HF https://huggingface.co/deepseek-ai/DeepSeek-V3-Base.

```bash
git lfs install
git clone https://huggingface.co/deepseek-ai/DeepSeek-V3-Base
```
**Optional**: Convert the FP8 checkpoint to BF16. This is not necessary unless you want to run the model E2E in BF16 precision.
```bash
git clone https://github.com/deepseek-ai/DeepSeek-V3.git
cd DeepSeek-V3/inference/
python fp8_cast_bf16.py --input-fp8-hf-path /path/to/DeepSeek-V3 --output-bf16-hf-path /path/to/deepseek-v3-bf16
cp /path/to/DeepSeek-V3/config.json /path/to/DeepSeek-V3/configuration_deepseek.py /path/to/deepseek-v3-bf16/
```

## Hardware

The DeepSeek-V3 model requires at least 32x80G GPU memory, model contains 660B parameters, roughly 1.3TB memory (with BF16 precision).

***Caution: Current TRT-LLM MLA kernel only supports Hopper architecture (SM90). Ampere architecture (SM80 & SM86) will be supported in the future release.***

## Overview

The TensorRT-LLM DeepSeek-V3 implementation can be found in [tensorrt_llm/models/deepseek_v2/model.py](../../tensorrt_llm/models/deepseek_v2/model.py). The TensorRT-LLM Deepseek-V3 example code is located in [`example/deepseek_v3`](./). There is one main file:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert the DeepSeek-V3 model into tensorrt-llm checkpoint format.

In addition, there are three shared files in the parent folder [`examples`](../) can be used for inference and evaluation:

* [`../run.py`](../run.py) to run the model inference output by giving an input text.


## Usage

The TensorRT-LLM DeepSeek-V3 example code is located at [examples/deepseek_v3](./). It takes PyTorch weights as input, and builds corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### Build TensorRT engine(s)

Below is the step-by-step to run DeepSeek-V3 with TensorRT-LLM.


Firstly, convert the checkpoint to the TensorRT-LLM checkpoint format by running [`convert_checkpoint.py`](./convert_checkpoint.py). After that, the TensorRT engine(s) can be built with the TensorRT-LLM checkpoint.

To convert FP8 checkpoint:
```bash
# Convert Deepseek-v3 HF Native FP8 weights to TensorRT-LLM checkpoint.
python convert_checkpoint.py --model_dir ./DeepSeek-V3 \
                            --output_dir ./trtllm_checkpoint_deepseek_v3_16gpu_fp8 \
                            --dtype bfloat16 \
                            --use_fp8_weights \
                            --tp_size 16 \
                            --workers 8 # using multiple workers can accelerate the conversion process
```

To convert BF16 checkpoint:
```bash
# Convert Deepseek-v3 HF weights to TensorRT-LLM checkpoint in BF16.
python convert_checkpoint.py --model_dir ./DeepSeek-V3 \
                            --output_dir ./trtllm_checkpoint_deepseek_v3_32gpu_bf16 \
                            --dtype bfloat16 \
                            --tp_size 32 \
                            --workers 8 # using multiple workers can accelerate the conversion process
```
We observed the checkpoint conversion time took hours, while using a significant amount of CPU memory, please adjust the `--workers` parameter to balance your time and memory consumption.


After the checkpoint conversion, the TensorRT engine(s) can be built with the TensorRT-LLM checkpoint.

For FP8:
```bash
# Build FP8 engine
trtllm-build --checkpoint_dir ./trtllm_checkpoint_deepseek_v3_16gpu_fp8 \
            --output_dir ./trtllm_engines/deepseek_v3/fp8/tp16-sel4096-isl2048-bs4 \
            --max_batch_size 4 \
            --max_seq_len 4096 \
            --max_input_len 2048 \
            --use_paged_context_fmha enable \
            --workers 8
```

For BF16:
```bash
# Build BF16 engine
trtllm-build --checkpoint_dir ./trtllm_checkpoint_deepseek_v3_32gpu_bf16 \
            --output_dir ./trtllm_engines/deepseek_v3/bf16/tp32-sel4096-isl2048-bs4 \
            --gpt_attention_plugin bfloat16 \
            --gemm_plugin bfloat16 \
            --max_batch_size 4 \
            --max_seq_len 4096 \
            --max_input_len 2048 \
            --use_paged_context_fmha enable \
            --workers 8
```

***Caution: `--max_batch_size` and `--max_seq_len` are the main factors to determine how many GPU memory will be used during runtime, so later when try to run e.g., `summarize.py` or `mmlu.py` or `gptManagerBenchmark.cpp`may need adjust `--max_batch_size` and `--max_seq_len` accordingly to avoid OOM.(meaning rebuild TensorRT engine with smaller `--max_batch_size` and `--max_seq_len` if needed based on GPU memory size), there is beautiful technical log perf-best-practices.md (https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-best-practices.md) explained the mechanism.***

Test the FP8 engines with [run.py](../run.py) script:
```
# run.sh
python3 ../run.py --input_text "Today is a nice day." \
        --max_output_len 30 \
        --tokenizer_dir ./DeepSeek-V3 \
        --engine_dir ./trtllm_engines/deepseek_v3/fp8/tp16-sel4096-isl2048-bs4 \
        --top_p 0.95 \
        --temperature 0.3


```
For multi-nodes inference, let's take Slurm as an example using above command (run.sh):

```bash
srun -N 2 -w node-[1-2] --gres=gpu:8 --ntasks-per-node 8 \
    --container-image tensorrt_llm/release:latest \
    --container-mounts ${PWD}:/workspace \
    sh /workspace/command/run.sh
```

and the output will be like:

```
...
[TensorRT-LLM][INFO] Refreshed the MPI local session
[TensorRT-LLM][INFO] Refreshed the MPI local session
[TensorRT-LLM][INFO] Refreshed the MPI local session
[TensorRT-LLM][INFO] Refreshed the MPI local session
[TensorRT-LLM][INFO] Refreshed the MPI local session
[TensorRT-LLM][INFO] Refreshed the MPI local session
[TensorRT-LLM][INFO] Refreshed the MPI local session
[TensorRT-LLM][INFO] Refreshed the MPI local session
[TensorRT-LLM][INFO] Refreshed the MPI local session
[TensorRT-LLM][INFO] Refreshed the MPI local session
[TensorRT-LLM][INFO] Refreshed the MPI local session
[TensorRT-LLM][INFO] Refreshed the MPI local session
[TensorRT-LLM][INFO] Refreshed the MPI local session
[TensorRT-LLM][INFO] Refreshed the MPI local session
[TensorRT-LLM][INFO] Refreshed the MPI local session
[TensorRT-LLM][INFO] Refreshed the MPI local session
Input [Text 0]: "Today is a nice day."
Output [Text 0 Beam 0]: " I am going to the park with my friends. We are going to play soccer. We are going"
```
