# BERT and BERT Variants

This document explains how to build the BERT family, specifically [BERT](https://huggingface.co/docs/transformers/model_doc/bert) and [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta) model using TensorRT-LLM. It also describes how to run on a single GPU and two GPUs.

## Overview

The TensorRT-LLM BERT family implementation can be found in [`tensorrt_llm/models/bert/model.py`](../../tensorrt_llm/models/bert/model.py). The TensorRT-LLM BERT family example
code is located in [`examples/bert`](./). There are two main files in that folder:

 * [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the model,
 * [`run.py`](./run.py) to run the inference on an input text,

## Build and run on a single GPU

TensorRT-LLM converts HuggingFace BERT family models into TensorRT engine(s).
To build the TensorRT engine, use:

```bash
python3 build.py [--model <model_name> --dtype <data_type> ...]
```

Supported `model_name` options include: BertModel, BertForQuestionAnswering, BertForSequenceClassification, RobertaModel, RobertaForQuestionAnswering, and RobertaForSequenceClassification, with `BertModel` as the default.

Some examples are as follows:

```bash
# Build BertModel
python3 build.py --model BertModel --dtype=float16 --log_level=verbose

# Build RobertaModel
python3 build.py --model RobertaModel --dtype=float16 --log_level=verbose

# Build BertModel with TensorRT-LLM BERT Attention plugin for enhanced runtime performance
python3 build.py --dtype=float16 --log_level=verbose --use_bert_attention_plugin float16

# Build BertForSequenceClassification with TensorRT-LLM remove input padding knob for enhanced runtime performance
python3 build.py --model BertForSequenceClassification --remove_input_padding --use_bert_attention_plugin float16
```

The following command can be used to run the model on a single GPU:

```bash
python3 run.py

```
If the model built with **--remove_input_padding** knob, please run the model with below command
```bash
python3 run_remove_input_padding.py
```

#### Fused MultiHead Attention (FMHA)

You can enable the FMHA kernels for BERT by adding `--enable_context_fmha` to the invocation of `build.py`. Note that it is disabled by default because of possible accuracy issues due to the use of Flash Attention.

If you find that the default fp16 accumulation (`--enable_context_fmha`) cannot meet the requirement, you can try to enable fp32 accumulation by adding `--enable_context_fmha_fp32_acc`. However, it is expected to see performance drop.

Note `--enable_context_fmha` / `--enable_context_fmha_fp32_acc` has to be used together with `--use_bert_attention_plugin float16`.


#### Remove input padding
The remove input padding feature is enabled by adding `--remove_input_padding` into build command.
When input padding is removed, the different tokens are packed together. It reduces both the amount of computations and memory consumption. For more details, see this [Document](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.md#padded-and-packed-tensors).

Currently, this feature only enables for BertForSequenceClassification model.

## Build and run on two GPUs

The following two commands can be used to build TensorRT engines to run BERT on two GPUs. The first command builds one engine for the first GPU. The second command builds another engine for the second GPU. For example, to build `BertForQuestionAnswering` with two GPUs, run:

```bash
python3 build.py --model BertForQuestionAnswering --world_size=2 --rank=0
python3 build.py --model BertForQuestionAnswering --world_size=2 --rank=1
```

The following command can be used to run the inference on 2 GPUs. It uses MPI with `mpirun`.

```bash
mpirun -n 2 python3 run.py
```
