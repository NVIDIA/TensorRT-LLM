# BERT

This document explains how to build the [BERT](https://huggingface.co/docs/transformers/model_doc/bert) model using TensorRT-LLM. It also describes how to run on a single GPU and two GPUs.

## Overview

The TensorRT-LLM BERT implementation can be found in [`tensorrt_llm/models/bert/model.py`](../../tensorrt_llm/models/bert/model.py). The TensorRT-LLM BERT example
code is located in [`examples/bert`](./). There are four main files in that folder:

 * [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the BERT model,
 * [`run.py`](./run.py) to run the inference on an input text,

## Build and run BERT on a single GPU

In this example, TensorRT-LLM builds TensorRT engine(s) from the [HuggingFace BERT](https://huggingface.co/docs/transformers/model_doc/bert) model.
Use the following command to build the TensorRT engine:

```bash
python3 build.py --dtype=float16 --log_level=verbose

# Enable the special TensorRT-LLM BERT Attention plugin (--use_bert_attention_plugin) to increase runtime performance.
python3 build.py --dtype=float16 --log_level=verbose --use_bert_attention_plugin float16
# Enable half accumulation for attention BMM1 (applied to unfused MHA plugins)
python3 build.py --dtype=float16 --log_level=verbose --use_bert_attention_plugin float16 --enable_qk_half_accum
```

The following command can be used to run the BERT model on a single GPU:

```bash
python3 run.py
```

#### Fused MultiHead Attention (FMHA)

You can enable the FMHA kernels for BERT by adding `--enable_context_fmha` to the invocation of `build.py`. Note that it is disabled by default because of possible accuracy issues due to the use of Flash Attention.

If you find that the default fp16 accumulation (`--enable_context_fmha`) cannot meet the requirement, you can try to enable fp32 accumulation by adding `--enable_context_fmha_fp32_acc`. However, it is expected to see performance drop.

Note `--enable_context_fmha` / `--enable_context_fmha_fp32_acc` has to be used together with `--use_bert_attention_plugin float16`.

## Build and run BERT on two GPUs

The following two commands can be used to build TensorRT engines to run BERT on two GPUs. The first command builds one engine for the first GPU. The second command builds another engine for the second GPU.

```bash
python3 build.py --world_size=2 --rank=0
python3 build.py --world_size=2 --rank=1
```

The following command can be used to run the inference on 2 GPUs. It uses MPI with `mpirun`.

```bash
mpirun -n 2 python3 run.py
```
