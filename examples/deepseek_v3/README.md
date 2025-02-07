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

***Please Note:***
- Prefer using FP8 for DeepSeek-V3 since model original training precision is FP8.  

## Prerequisite

First, please download DeepSeek-V3 weights from HF https://huggingface.co/deepseek-ai/DeepSeek-V3-Base.

```bash
git lfs install
git clone https://huggingface.co/deepseek-ai/DeepSeek-V3-Base
```
**Optional**: Convert the FP8 checkpoint to BF16. 

**This is not necessary unless you want to run the model E2E in BF16 precision.**
```bash
git clone https://github.com/deepseek-ai/DeepSeek-V3.git
cd DeepSeek-V3/inference/
python fp8_cast_bf16.py --input-fp8-hf-path /path/to/DeepSeek-V3 --output-bf16-hf-path /path/to/deepseek-v3-bf16
cp /path/to/DeepSeek-V3/config.json /path/to/DeepSeek-V3/configuration_deepseek.py /path/to/deepseek-v3-bf16/
```

## Hardware

The DeepSeek-V3 model requires at least 8x141G GPU memory, model contains 660B parameters, roughly 660GB memory (with FP8 precision).

***Caution: Current TRT-LLM MLA kernel only supports Hopper architecture (SM90). Ampere architecture (SM80 & SM86) will be supported in the future release.***

Please follow the instructions [here](https://github.com/NVIDIA/TensorRT-LLM/blob/deepseek/docs/source/installation/build-from-source-linux.md#building-a-tensorrt-llm-docker-image
) to achieve a correct docker image.

## Overview

The TensorRT-LLM DeepSeek-V3 implementation can be found in [tensorrt_llm/models/deepseek_v2/model.py](../../tensorrt_llm/models/deepseek_v2/model.py). The TensorRT-LLM Deepseek-V3 example code is located in [`example/deepseek_v3`](./). There is one main file:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert the DeepSeek-V3 model into tensorrt-llm checkpoint format.

In addition, there are three shared files in the parent folder [`examples`](../) can be used for inference and evaluation:

* [`../run.py`](../run.py) to run the model inference output by giving an input text.

* [`../mmlu.py`](../mmlu.py) to running score script from https://github.com/declare-lab/instruct-eval to compare HF model and TensorRT-LLM model on the MMLU dataset.


## Usage

The TensorRT-LLM DeepSeek-V3 example code is located at [examples/deepseek_v3](./). It takes PyTorch weights as input, and builds corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### Build TensorRT engine(s)

Below is the step-by-step to run DeepSeek-V3 with TensorRT-LLM.


Firstly, convert the checkpoint to the TensorRT-LLM checkpoint format by running [`convert_checkpoint.py`](./convert_checkpoint.py). After that, the TensorRT engine(s) can be built with the TensorRT-LLM checkpoint.

To convert FP8 checkpoint:
```bash
# Convert Deepseek-v3 HF Native FP8 weights to TensorRT-LLM checkpoint.
python convert_checkpoint.py --model_dir ./DeepSeek-V3 \
                            --output_dir ./trtllm_checkpoint_deepseek_v3_8gpu_fp8 \
                            --dtype bfloat16 \
                            --use_fp8_weights \
                            --tp_size 8 \
                            --workers 8 # using multiple workers can accelerate the conversion process
```

**Optional**: To convert BF16 checkpoint:
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
trtllm-build --checkpoint_dir ./trtllm_checkpoint_deepseek_v3_8gpu_fp8 \
            --output_dir ./trtllm_engines/deepseek_v3/fp8/tp8-sel4096-isl2048-bs4 \
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
        --engine_dir ./trtllm_engines/deepseek_v3/fp8/tp8-sel4096-isl2048-bs4 \
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

At last, we can evaluate the model with [mmlu.py](../mmlu.py) script:


```bash
# Download MMLU dataset
mkdir mmlu_data && cd mmlu_data
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar && tar -xf data.tar
# Run MMLU evaluation
python3 mmlu.py \
        --hf_model_dir ${MODEL_DIR} \
        --engine_dir ./trtllm_engines/deepseek_v3/fp8/tp8-sel4096-isl2048-bs4 \
        --data_dir mmlu_data \
        --test_trt_llm 2>&1 | tee ${ENGINE_DIR}/test_with_mmlu.log
```

and the output will be like:

```
Average accuracy 0.926 - high_school_macroeconomics
Average accuracy 0.752 - high_school_mathematics
Average accuracy 0.954 - high_school_microeconomics
Average accuracy 0.848 - high_school_physics
Average accuracy 0.967 - high_school_psychology
Average accuracy 0.861 - high_school_statistics
Average accuracy 0.956 - high_school_us_history
Average accuracy 0.954 - high_school_world_history
Average accuracy 0.861 - human_aging
Average accuracy 0.931 - human_sexuality
Average accuracy 0.975 - international_law
Average accuracy 0.907 - jurisprudence
Average accuracy 0.920 - logical_fallacies
Average accuracy 0.848 - machine_learning
Average accuracy 0.951 - management
Average accuracy 0.957 - marketing
Average accuracy 0.950 - medical_genetics
Average accuracy 0.957 - miscellaneous
Average accuracy 0.870 - moral_disputes
Average accuracy 0.798 - moral_scenarios
Average accuracy 0.918 - nutrition
Average accuracy 0.916 - philosophy
Average accuracy 0.932 - prehistory
Average accuracy 0.869 - professional_accounting
Average accuracy 0.714 - professional_law
Average accuracy 0.956 - professional_medicine
Average accuracy 0.908 - professional_psychology
Average accuracy 0.800 - public_relations
Average accuracy 0.869 - security_studies
Average accuracy 0.960 - sociology
Average accuracy 0.950 - us_foreign_policy
Average accuracy 0.578 - virology
Average accuracy 0.930 - world_religions
Average accuracy 0.852 - math
Average accuracy 0.874 - health
Average accuracy 0.905 - physics
Average accuracy 0.936 - business
Average accuracy 0.958 - biology
Average accuracy 0.825 - chemistry
Average accuracy 0.888 - computer science
Average accuracy 0.912 - economics
Average accuracy 0.890 - engineering
Average accuracy 0.851 - philosophy
Average accuracy 0.917 - other
Average accuracy 0.932 - history
Average accuracy 0.944 - geography
Average accuracy 0.904 - politics
Average accuracy 0.936 - psychology
Average accuracy 0.949 - culture
Average accuracy 0.744 - law
Average accuracy 0.883 - STEM
Average accuracy 0.827 - humanities
Average accuracy 0.926 - social sciences
Average accuracy 0.898 - other (business, health, misc.)
Average accuracy: 0.877
```

**Known Issue**

1. The memory allocation for MoE is too large.

This issue prevents running larger batch sizes and long sequence inputs. We will optimize and fix this issue soon.

