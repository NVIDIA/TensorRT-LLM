# Encoder-Decoder

This document shows how to build and run an Encoder-Decoder (Enc-Dec) model in TensorRT-LLM on NVIDIA GPUs.

## Overview

The TensorRT-LLM Enc-Dec implementation can be found in [tensorrt_llm/models/enc_dec/model.py](../../tensorrt_llm/models/enc_dec/model.py). The TensorRT-LLM Enc-Dec example code is located in [`examples/enc_dec`](./):

 * [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the Enc-Dec model,
 * [`run.py`](./run.py) to run the inference on an example input text.
 * Enc-Dec models can have specific implementations, such as the popular T5 family (T5, mT5, Flan-T5) and BART family (BART, mBART). They are located under subfolders `/t5` and `/bart`, each containing:
   * [`<model_type>/hf_convert.py`](./t5/hf_convert.py) to convert weights from HuggingFace PyTorch format to TRT-LLM format, and split weights for multi-GPU inference,
   * [`<model_type>/weight.py`](./t5/weight.py) to map the converted & split weights to TRT-LLM model.

## Usage

The TensorRT-LLM Enc-Dec example code locates at [examples/enc_dec](./). It takes HuggingFace model name as input, and builds the corresponding TensorRT engines. On each GPU, there will be two TensorRT engines, one for Encoder and one for Decoder.

## Encoder-Decoder Model Support

The implementation is designed to support generic encoder-decoder models by abstracting the common and derivative components of different model architectures, such as:
- [T5](https://huggingface.co/docs/transformers/main/en/model_doc/t5)
- [T5v1.1](https://huggingface.co/docs/transformers/model_doc/t5v1.1) and [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5)
- [mT5 (coming)](https://huggingface.co/docs/transformers/model_doc/mt5)
- [UL2 (coming)](https://huggingface.co/docs/transformers/model_doc/ul2) and [Flan-UL2 (coming)](https://huggingface.co/docs/transformers/model_doc/flan-ul2)
- [BART (coming)](https://huggingface.co/docs/transformers/model_doc/bart)
- [mBART (coming)](https://huggingface.co/docs/transformers/model_doc/mbart)

It also supports full Tensor Parallelism (TP), Pipeline Parallelism (PP), and a hybrid of the two. Currently, Fused Multi-Head Attention (FMHA) is not yet enabled for T5 family due to its relative attention design.

In this example, we use T5 (`t5-small`) and Flan-T5 (`google/flan-t5-small`) to showcase TRT-LLM support on Enc-Dec models.

### Download weights from HuggingFace Transformers
```bash
git clone https://huggingface.co/t5-small tmp/hf_models/t5-small
git clone https://huggingface.co/google/flan-t5-small tmp/hf_models/flan-t5-small
```

### Convert and Split Weights
The `<model_type>/hf_convert.py` script converts weights from HuggingFace format to TRT-LLM format, and splits weights for multi-GPU inference. `--inference_tensor_para_size` specifies the number of GPUs for tensor parallelism during inference.

It is fine to save one copy of converted weights at high precision, e.g. float32, if disk space allows. During the following engine building phase, engines of any inference precision can be built by weight dtype casting on the fly. Therefore, you can just keep one set of saved weights and build engines freely at different precisions, instead of saving weights for each inference precision.

After weight conversion, TensorRT-LLM converted weights and model configuration will be saved under `<out_dir>/<tpX>` directory, which is the `--weight_dir` input path you should give to the **next** engine building phase. `X` is Tensor Parallelim size for distributed inference.

```bash
python t5/hf_convert.py -i tmp/hf_models/t5-small -o tmp/trt_models/t5-small --weight_data_type float32 --inference_tensor_para_size <X>
```

### Build TensorRT engine(s)

TensorRT-LLM builds TensorRT engine(s) with flexible controls on different types of optimizations. Note that these are just examples to demonstrate multi-GPU inference. For small models like T5-small, single GPU is usually sufficient.

After engine building, TensorRT engines will be saved under `<out_dir>/<dtype>/<tpX>` directory, which is the `--engine_dir` path you should give to the next engine running phase. It is recommended to have `/<Y-gpu>` in the output path where `Y` is number of total GPU ranks in a multi-node, multi-GPU setup, because the same `Y` number GPUs could be executed with different TP (Tensor Parallelism) and PP (Pipeline Parallelism) combinations.

We should distinguish between `X` - TP size and `Y` - total number of GPU ranks:
* When `X = Y`, only TP is enabled
* When `X < Y`, both TP and PP are enabled. In such case, please make sure you have completed weight conversion step for `TP=X`.

```bash
# Example 1: build t5-small using a single GPU, FP32, running gready search
# use_gpt_attention_plugin is necessary in Enc-Dec.
# Try use_gemm_plugin to prevent accuracy issue.
# It is recommend to use --remove_input_padding along with --use_gpt_attention_plugin for better performance
python build.py --model_type t5 \
                --weight_dir tmp/trt_models/t5-small/tp1 \
                -o tmp/trt_engines/t5-small/1-gpu \
                --engine_name t5-small \
                --remove_input_padding \
                --use_bert_attention_plugin \
                --use_gpt_attention_plugin \
                --use_gemm_plugin \
                --use_rmsnorm_plugin \
                --dtype float32 \
                --max_beam_width 1

# Example 2: build t5-small using 4-way tensor parallelism on a node with 8 GPUs (but only use 4 of them, for demonstration purpose), BF16, enabling beam search up to width=3
python build.py --model_type t5 \
                --world_size 4 \
                --tp_size 4 \
                --gpus_per_node 4 \
                --weight_dir tmp/trt_models/t5-small/tp4 \
                -o tmp/trt_engines/t5-small/4-gpu \
                --engine_name t5-small \
                --remove_input_padding \
                --use_bert_attention_plugin \
                --use_gpt_attention_plugin \
                --use_gemm_plugin \
                --use_rmsnorm_plugin \
                --dtype bfloat16 \
                --max_beam_width 3

# Example 3: build flan-t5-small using 2-way tensor parallelism and 2-way pipeline parallelism on a node with 8 GPUs, BF16, enabling beam search up to width=3
python build.py --model_type t5 \
                --world_size 4 \
                --tp_size 2 \
                --pp_size 2 \
                --gpus_per_node 8 \
                --weight_dir tmp/trt_models/flan-t5-small/tp2 \
                -o tmp/trt_engines/flan-t5-small/4-gpu \
                --engine_name flan-t5-small \
                --remove_input_padding \
                --use_bert_attention_plugin \
                --use_gpt_attention_plugin \
                --use_gemm_plugin \
                --use_rmsnorm_plugin \
                --dtype bfloat16 \
                --max_beam_width 3
```

### Run

Run a TensorRT-LLM Enc-Dec model using the engines generated by build.py.
Note that during model deployment, only the TensorRT engine files are needed. Previously downloaded model checkpoints and converted weights can be removed.

```bash
# Example 1: inference w/ single GPU, FP32, greedy search, compare results with HuggingFace FP32
python3 run.py --engine_dir tmp/trt_engines/t5-small/1-gpu/float32/tp1 --engine_name t5-small --model_name t5-small --max_new_token=64 --num_beams=1 --compare_hf_fp32

# Example 2: inference w/ 4 GPUs (4-way TP, as configured during the engine building step), BF16, greedy search, compare results with HuggingFace FP32
mpirun --allow-run-as-root -np 4 python3 run.py --engine_dir tmp/trt_engines/t5-small/4-gpu/bfloat16/tp4 --engine_name t5-small --model_name t5-small --max_new_token=64 --num_beams=1 --compare_hf_fp32

# Example 3: inference w/ 4 GPUs (2-way TP and 2-way PP, as configured during the engine building step), BF16, greedy search
mpirun --allow-run-as-root -np 4 python3 run.py --engine_dir tmp/trt_engines/flan-t5-small/4-gpu/bfloat16/tp2 --engine_name flan-t5-small --model_name google/flan-t5-small --max_new_token=64 --num_beams=1
```

### Reminders

- Flan-T5 models have known issues regarding FP16 precision and using BF16 precision is recommended, regardless of TRT-LLM. While we are working on improving FP16 results, please stay with FP32 or BF16 precision for Flan-T5 family.
- Batched/Ragged input with beam search is having subtle issues with some sequence results being truncated. For the time being, please follow (1) if batch size = 1, no problem (2) if batched input is padded (i.e., not using `--remove_input_padding` flag), no problem (3) if batched input is ragged (i.e., using `--remove_input_padding`), only use greedy search for now.
- For T5 and Flan-T5 family that have relative attention bias design, the relative attention table is split along `num_heads` dimension in Tensor Parallelism mode. Therefore, `num_heads` must be divisible by `tp_size`. Please be aware of this when setting the TP parameter.
