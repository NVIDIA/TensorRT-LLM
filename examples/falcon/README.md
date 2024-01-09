# Falcon

This document shows how to build and run a Falcon model in TensorRT-LLM on single GPU, single node multi-GPU, and multi-node multi-GPU.

## Overview

The TensorRT-LLM Falcon implementation can be found in [tensorrt_llm/models/falcon/model.py](../../tensorrt_llm/models/falcon/model.py). The TensorRT-LLM Falcon example code is located in [`examples/falcon`](./). There is one main file:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert a checkpoint from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers) format to the TensorRT-LLM format.

In addition, there are two shared files in the parent folder [`examples`](../) for inference and evaluation:

* [`../run.py`](../run.py) to run the inference on an input text;
* [`../summarize.py`](../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset.

## Support Matrix
  * FP16
  * BF16
  * FP8
  * FP8 KV CACHE
  * Groupwise quantization (AWQ)
  * Tensor Parallel
  * STRONGLY TYPED

## Usage

The next two sections describe how to convert the weights from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers)
format to the TensorRT-LLM format.

### 1. Download weights from HuggingFace Transformers

Install the dependency packages and setup `git-lfs`.

```bash
# Install dependencies
pip install -r requirements.txt

# Setup git-lfs
git lfs install
```

There are four HF checkpoints available. Use one of the following commands to fetch the checkpoint you are interested in. Follow the guides here https://huggingface.co/docs/transformers/main/en/model_doc/falcon.

```bash
# falcon-rw-1b
git clone https://huggingface.co/tiiuae/falcon-rw-1b falcon/rw-1b

# falcon-7b-instruct
git clone https://huggingface.co/tiiuae/falcon-7b-instruct falcon/7b-instruct

# falcon-40b-instruct
git clone https://huggingface.co/tiiuae/falcon-40b-instruct falcon/40b-instruct

# falcon-180b
git clone https://huggingface.co/tiiuae/falcon-180B falcon/180b
```

### 2. Convert weights from HF Transformers to TensorRT-LLM format
The [`convert_checkpoint.py`](./convert_checkpoint.py) script converts HF weights to TensorRT-LLM checkpoints. The number of checkpoint files (in .safetensors format) is same to the number of GPUs used to run inference.

```bash
# falcon-rw-1b: single gpu, dtype float16
python3 convert_checkpoint.py --model_dir ./falcon/rw-1b \
                --dtype float16 \
                --output_dir ./falcon/rw-1b/trt_ckpt/fp16/1-gpu/

# falcon-7b-instruct: single gpu, dtype bfloat16
python3 convert_checkpoint.py --model_dir ./falcon/7b-instruct \
                --dtype bfloat16 \
                --output_dir ./falcon/7b-instruct/trt_ckpt/bf16/1-gpu/

# falcon-40b-instruct: 2-way tensor parallelism
python3 convert_checkpoint.py --model_dir ./falcon/40b-instruct \
                --dtype bfloat16 \
                --output_dir ./falcon/40b-instruct/trt_ckpt/bf16/tp2-pp1/ \
                --world_size 2 \
                --tp_size 2

# falcon-40b-instruct: 2-way tensor parallelism and 2-way pipeline parallelism
python3 convert_checkpoint.py --model_dir ./falcon/40b-instruct \
                --dtype bfloat16 \
                --output_dir ./falcon/40b-instruct/trt_ckpt/bf16/tp2-pp2/ \
                --world_size 4 \
                --tp_size 2 \
                --pp_size 2

# falcon-180b: 8-way tensor parallelism, loading weights shard-by-shard
python3 convert_checkpoint.py --model_dir ./falcon/180b \
                --dtype bfloat16 \
                --output_dir ./falcon/180b/trt_ckpt/bf16/tp8-pp1/ \
                --world_size 8 \
                --tp_size 8 \
                --load_by_shard \
                --workers 8

# falcon-180b: 4-way tensor parallelism and 2-way pipeline parallelism, loading weights shard-by-shard
python3 convert_checkpoint.py --model_dir ./falcon/180b \
                --dtype bfloat16 \
                --output_dir ./falcon/180b/trt_ckpt/bf16/tp4-pp2/ \
                --world_size 8 \
                --tp_size 4 \
                --pp_size 2 \
                --load_by_shard \
                --workers 8
```

Note that in order to use N-way tensor parallelism, the number of attention heads must be a multiple of N.
For example, you can't configure 2-way tensor parallelism for [falcon-7b](https://huggingface.co/tiiuae/falcon-7b) or [falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct), because the number of attention heads is 71 (not divisible by 2).


### 3. Build TensorRT engine(s)
The `trtllm-build` command builds TensorRT-LLM engines from TensorRT-LLM checkpoints. The number of engine files is also same to the number of GPUs used to run inference.

Normally, the `trtllm-build` command only requires a single GPU, but you can enable parallel building by passing the number of GPUs to the `--workers` argument.

```bash
# falcon-rw-1b
# It is recommend to use --remove_input_padding along with --use_gpt_attention_plugin for better performance
trtllm-build --checkpoint_dir ./falcon/rw-1b/trt_ckpt/fp16/1-gpu/ \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --output_dir ./falcon/rw-1b/trt_engines/fp16/1-gpu/

# falcon-7b-instruct
# Enabling --use_gpt_attention_plugin is necessary for rotary positional embedding (RoPE)
trtllm-build --checkpoint_dir ./falcon/7b-instruct/trt_ckpt/bf16/1-gpu/ \
                --use_gemm_plugin bfloat16 \
                --remove_input_padding \
                --use_gpt_attention_plugin bfloat16 \
                --enable_context_fmha \
                --output_dir ./falcon/7b-instruct/trt_engines/bf16/1-gpu/

# falcon-40b-instruct: 2-way tensor parallelism
trtllm-build --checkpoint_dir ./falcon/40b-instruct/trt_ckpt/bf16/tp2-pp1/ \
                --use_gemm_plugin bfloat16 \
                --remove_input_padding \
                --use_gpt_attention_plugin bfloat16 \
                --enable_context_fmha \
                --output_dir ./falcon/40b-instruct/trt_engines/bf16/tp2-pp1/

# falcon-40b-instruct: 2-way tensor parallelism and 2-way pipeline parallelism
trtllm-build --checkpoint_dir ./falcon/40b-instruct/trt_ckpt/bf16/tp2-pp2/ \
                --use_gemm_plugin bfloat16 \
                --remove_input_padding \
                --use_gpt_attention_plugin bfloat16 \
                --enable_context_fmha \
                --output_dir ./falcon/40b-instruct/trt_engines/bf16/tp2-pp2/

# falcon-180b: 8-way tensor parallelism
trtllm-build --checkpoint_dir ./falcon/180b/trt_ckpt/bf16/tp8-pp1/ \
                --use_gemm_plugin bfloat16 \
                --remove_input_padding \
                --use_gpt_attention_plugin bfloat16 \
                --enable_context_fmha \
                --output_dir ./falcon/180b/trt_engines/bf16/tp8-pp1/ \
                --workers 8

# falcon-180b: 4-way tensor parallelism and 2-way pipeline parallelism
trtllm-build --checkpoint_dir ./falcon/180b/trt_ckpt/bf16/tp4-pp2/ \
                --use_gemm_plugin bfloat16 \
                --remove_input_padding \
                --use_gpt_attention_plugin bfloat16 \
                --enable_context_fmha \
                --output_dir ./falcon/180b/trt_engines/bf16/tp4-pp2/ \
                --workers 8
```

If the engines are built successfully, you will see output like (falcon-rw-1b as the example):
```
......
[12/27/2023-03:46:29] [TRT] [I] Engine generation completed in 35.0677 seconds.
[12/27/2023-03:46:29] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 393 MiB, GPU 2699 MiB
[12/27/2023-03:46:29] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in building engine: CPU +0, GPU +2699, now: CPU 0, GPU 2699 (MiB)
[12/27/2023-03:46:29] [TRT] [I] [MemUsageStats] Peak memory usage during Engine building and serialization: CPU: 10624 MiB
[12/27/2023-03:46:29] [TRT-LLM] [I] Total time of building Unnamed Network 0: 00:00:36
[12/27/2023-03:46:31] [TRT-LLM] [I] Serializing engine to ./falcon/rw-1b/trt_engines/fp16/1-gpu/rank0.engine...
[12/27/2023-03:46:59] [TRT-LLM] [I] Engine serialized. Total time: 00:00:28
[12/27/2023-03:46:59] [TRT-LLM] [I] Total time of building all engines: 00:01:59
```

### 4. Run summarization task with the TensorRT engine(s)
The `../summarize.py` script can run the built engines to summarize the articles from the
[cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset.

```bash
# falcon-rw-1b
python ../summarize.py --test_trt_llm \
                       --hf_model_dir ./falcon/rw-1b \
                       --engine_dir ./falcon/rw-1b/trt_engines/fp16/1-gpu/

# falcon-7b-instruct
python ../summarize.py --test_trt_llm \
                       --hf_model_dir ./falcon/7b-instruct \
                       --engine_dir ./falcon/7b-instruct/trt_engines/bf16/1-gpu/

# falcon-40b-instruct: 2-way tensor parallelism
mpirun -n 2 --allow-run-as-root --oversubscribe \
    python ../summarize.py --test_trt_llm \
                           --hf_model_dir ./falcon/40b-instruct \
                           --engine_dir ./falcon/40b-instruct/trt_engines/bf16/tp2-pp1/

# falcon-40b-instruct: 2-way tensor parallelism and 2-way pipeline parallelism
mpirun -n 4 --allow-run-as-root --oversubscribe \
    python ../summarize.py --test_trt_llm \
                           --hf_model_dir ./falcon/40b-instruct \
                           --engine_dir ./falcon/40b-instruct/trt_engines/bf16/tp2-pp2/

# falcon-180b: 8-way tensor parallelism
mpirun -n 8 --allow-run-as-root --oversubscribe \
    python ../summarize.py --test_trt_llm \
                           --hf_model_dir ./falcon/180b \
                           --engine_dir ./falcon/180b/trt_engines/bf16/tp8-pp1/

# falcon-180b: 4-way tensor parallelism and 2-way pipeline parallelism
mpirun -n 8 --allow-run-as-root --oversubscribe \
    python ../summarize.py --test_trt_llm \
                           --hf_model_dir ./falcon/180b \
                           --engine_dir ./falcon/180b/trt_engines/bf16/tp4-pp2/
```

If the engines are run successfully, you will see output like (falcon-rw-1b as the example):
```
......
[12/27/2023-03:57:02] [TRT-LLM] [I] TensorRT-LLM (total latency: 5.816917419433594 sec)
[12/27/2023-03:57:02] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[12/27/2023-03:57:02] [TRT-LLM] [I]   rouge1 : 15.061493342516243
[12/27/2023-03:57:02] [TRT-LLM] [I]   rouge2 : 4.495335888974063
[12/27/2023-03:57:02] [TRT-LLM] [I]   rougeL : 11.800002670828547
[12/27/2023-03:57:02] [TRT-LLM] [I]   rougeLsum : 13.458777656925877
```

### FP8 Post-Training Quantization

The examples below use the NVIDIA AMMO (AlgorithMic Model Optimization) toolkit for the model quantization process.

First make sure AMMO toolkit is installed (see [examples/quantization/README.md](/examples/quantization/README.md#preparation))

Now quantize HF Falcon weights as follows.
After successfully running the script, the output should be in .npz format, e.g. `quantized_fp8/falcon_tp_1_rank0.npz`,
where FP8 scaling factors are stored.

```bash
# Quantize HF Falcon 180B checkpoint into FP8 and export a single-rank checkpoint
python examples/quantization/quantize.py --model_dir falcon/180b \
                                         --dtype float16 \
                                         --qformat fp8 \
                                         --export_path quantized_fp8 \
                                         --calib_size 16

# Convert the HF weights and AMMO quantization scales to trtllm checkpoint
python3 convert_checkpoint.py --model_dir ./falcon/180b \
                --dtype float16 \
                --ammo_quant_ckpt_path ./quantized_fp8/falcon_tp1_rank0.npz \
                --enable_fp8 \
                --fp8_kv_cache \
                --output_dir ./falcon/180b/trt_ckpt/fp8/tp8-pp1/ \
                --world_size 8 \
                --tp_size 4 \
                --pp_size 2 \
                --load_by_shard \
                --workers 8

# Build trtllm engines from the trtllm checkpoint
trtllm-build --checkpoint_dir ./falcon/180b/trt_ckpt/fp8/tp8-pp1/ \
                --use_gemm_plugin bfloat16 \
                --remove_input_padding \
                --use_gpt_attention_plugin bfloat16 \
                --enable_context_fmha \
                --strongly_typed \
                --output_dir ./falcon/180b/trt_engines/bf16/tp4-pp2/ \
                --workers 8

# Run the summarization task
mpirun -n 8 --allow-run-as-root --oversubscribe \
    python ../summarize.py --test_trt_llm \
                           --hf_model_dir ./falcon/180b \
                           --engine_dir ./falcon/180b/trt_engines/bf16/tp4-pp2/
```

### Groupwise quantization (AWQ)

The examples below use the NVIDIA AMMO (AlgorithMic Model Optimization) toolkit for the model quantization process.

First make sure AMMO toolkit is installed (see [examples/quantization/README.md](/examples/quantization/README.md#preparation))

Now quantize HF Falcon weights as follows.
After successfully running the script, the output should be in .npz format, e.g. `quantized_int4_awq/falcon_tp1_rank0.npz`,
where INT4 AWQ scaling factors are stored.

```bash
# Quantize HF Falcon 180B checkpoint into INT4-AWQ and export a single-rank checkpoint
python quantize.py --model_dir ./falcon/180b \
                   --dtype float16 \
                   --qformat int4_awq \
                   --export_path quantized_int4_awq \
                   --calib_size 16

# Convert the HF weights and AMMO quantization scales to trtllm checkpoint
python3 convert_checkpoint.py --model_dir ./falcon/180b \
                              --dtype float16 \
                              --ammo_quant_ckpt_path ./quantized_int4_awq/falcon_tp1_rank0.npz \
                              --use_weight_only \
                              --weight_only_precision int4_awq \
                              --per_group \
                              --output_dir ./falcon/180b/trt_ckpt/int4_awq/tp2/ \
                              --world_size 2 \
                              --tp_size 2 \
                              --pp_size 1 \
                              --load_by_shard \
                              --workers 2

# Build trtllm engines from the trtllm checkpoint
trtllm-build --checkpoint_dir ./falcon/180b/trt_ckpt/int4_awq/tp2/ \
             --use_gemm_plugin float16 \
             --remove_input_padding \
             --use_gpt_attention_plugin float16 \
             --enable_context_fmha \
             --output_dir ./falcon/180b/trt_engines/int4_awq/tp2/ \
             --workers 2

# Run the summarization task
mpirun -n 2 --allow-run-as-root --oversubscribe \
    python ../summarize.py --test_trt_llm \
                           --hf_model_dir ./falcon/180b \
                           --engine_dir ./falcon/180b/trt_engines/int4_awq/tp2/
```

## Troubleshooting

### 1. The HuggingFace Falcon may raise an error when using  the `accelerate` package.

One may find the following message.
```
Traceback (most recent call last):
  File "build.py", line 10, in <module>
    from transformers import FalconConfig, FalconForCausalLM
  File "<frozen importlib._bootstrap>", line 1039, in _handle_fromlist
  File "/usr/local/lib/python3.8/dist-packages/transformers/utils/import_utils.py", line 1090, in __getattr__
    value = getattr(module, name)
  File "/usr/local/lib/python3.8/dist-packages/transformers/utils/import_utils.py", line 1089, in __getattr__
    module = self._get_module(self._class_to_module[name])
  File "/usr/local/lib/python3.8/dist-packages/transformers/utils/import_utils.py", line 1101, in _get_module
    raise RuntimeError(
RuntimeError: Failed to import transformers.models.falcon.modeling_falcon because of the following error (look up to see its traceback):
```
It may be resolved by pinning the version of `typing-extensions` package by `4.5.0`.
```bash
pip install typing-extensions==4.5.0
```
