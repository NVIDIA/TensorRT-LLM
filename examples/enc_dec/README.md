# Encoder-Decoder

This document shows how to build and run an Encoder-Decoder (Enc-Dec) model in TensorRT-LLM on NVIDIA GPUs.

## Overview

The TensorRT-LLM Enc-Dec implementation can be found in [tensorrt_llm/models/enc_dec/model.py](../../tensorrt_llm/models/enc_dec/model.py). The TensorRT-LLM Enc-Dec example code is located in [`examples/enc_dec`](./):

 * [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the Enc-Dec model,
 * [`run.py`](./run.py) to run the inference on an example input text.
 * Enc-Dec models can have specific implementations, such as the popular T5 family (T5, mT5, Flan-T5), BART family (BART, mBART), and FairSeq family (WMTs). They are located under subfolders `/t5`, `/bart`, and `/nmt`, each containing:
   * [`<model_type>/convert.py`](./t5/convert.py) to convert weights from HuggingFace or FairSeq format to TRT-LLM format, and split weights for multi-GPU inference,
   * [`<model_type>/weight.py`](./t5/weight.py) to map the converted & split weights to TRT-LLM model.

## Usage

The TensorRT-LLM Enc-Dec example code locates at [examples/enc_dec](./). It takes HuggingFace or FairSeq model name as input, and builds the corresponding TensorRT engines. On each GPU, there will be two TensorRT engines, one for Encoder and one for Decoder.

## Encoder-Decoder Model Support

The implementation is designed to support generic encoder-decoder models by abstracting the common and derivative components of different model architectures, such as:
- [T5](https://huggingface.co/docs/transformers/main/en/model_doc/t5)
- [T5v1.1](https://huggingface.co/docs/transformers/model_doc/t5v1.1) and [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5)
- [mT5](https://huggingface.co/docs/transformers/model_doc/mt5)
- [BART](https://huggingface.co/docs/transformers/model_doc/bart)
- [mBART](https://huggingface.co/docs/transformers/model_doc/mbart)
- [FairSeq NMT](https://pytorch.org/hub/pytorch_fairseq_translation/)
- [UL2 (coming)](https://huggingface.co/docs/transformers/model_doc/ul2) and [Flan-UL2 (coming)](https://huggingface.co/docs/transformers/model_doc/flan-ul2)

It also supports full Tensor Parallelism (TP), Pipeline Parallelism (PP), and a hybrid of the two. Currently, Fused Multi-Head Attention (FMHA) is not yet enabled for T5 family due to its relative attention design.

In this example, we use T5 (`t5-small`) and Flan-T5 (`google/flan-t5-small`) to showcase TRT-LLM support on Enc-Dec models. BART models and FairSeq models can follow very similar steps by just replacing the model name.

### Download weights from HuggingFace Transformers

```bash
git clone https://huggingface.co/t5-small tmp/hf_models/t5-small
git clone https://huggingface.co/google/flan-t5-small tmp/hf_models/flan-t5-small
git clone https://huggingface.co/facebook/bart-large-cnn tmp/hf_models/bart-large-cnn
git clone https://huggingface.co/facebook/mbart-large-50-many-to-one-mmt tmp/hf_models/mbart-large-50-many-to-one-mmt
```

### Convert and Split Weights
The `<model_type>/convert.py` script converts weights from HuggingFace or FairSeq format to TRT-LLM format, and splits weights for multi-GPU inference. `--inference_tensor_para_size` specifies the number of GPUs for tensor parallelism during inference.

It is fine to save one copy of converted weights at high precision, e.g. float32, if disk space allows. During the following engine building phase, engines of any inference precision can be built by weight dtype casting on the fly. Therefore, you can just keep one set of saved weights and build engines freely at different precisions, instead of saving weights for each inference precision.

After weight conversion, TensorRT-LLM converted weights and model configuration will be saved under `<out_dir>/<tpX>` directory, which is the `--weight_dir` input path you should give to the **next** engine building phase. `X` is Tensor Parallelim size for distributed inference.

```bash
# For T5
python t5/convert.py -i tmp/hf_models/t5-small -o tmp/trt_models/t5-small --weight_data_type float32 --inference_tensor_para_size <X>

# For BART or mBART
python bart/convert.py -i tmp/hf_models/bart-large-cnn -o tmp/trt_models/bart-large-cnn --weight_data_type float32 --inference_tensor_para_size <X>
```

### Build TensorRT engine(s)

TensorRT-LLM builds TensorRT engine(s) with flexible controls on different types of optimizations. Note that these are just examples to demonstrate multi-GPU inference. For small models like T5-small, single GPU is usually sufficient.

After engine building, TensorRT engines will be saved under `<out_dir>/<dtype>/<tpX>` directory, which is the `--engine_dir` path you should give to the next engine running phase. It is recommended to have `/<Y-gpu>` in the output path where `Y` is number of total GPU ranks in a multi-node, multi-GPU setup, because the same `Y` number GPUs could be executed with different TP (Tensor Parallelism) and PP (Pipeline Parallelism) combinations.

We should distinguish between `X` - TP size and `Y` - total number of GPU ranks:
* When `X = Y`, only TP is enabled
* When `X < Y`, both TP and PP are enabled. In such case, please make sure you have completed weight conversion step for `TP=X`.

```bash
# Example 1: build t5-small using a single GPU, FP32, running greedy search
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
                --dtype bfloat16 \
                --max_beam_width 3

# Example 4: build bart-large-cnn using a single GPU, FP32, running greedy search
python build.py --model_type bart \
                --weight_dir tmp/trt_models/bart-large-cnn/tp1 \
                -o tmp/trt_engines/bart-large-cnn/1-gpu \
                --engine_name bart-large-cnn \
                --remove_input_padding \
                --use_bert_attention_plugin \
                --use_gpt_attention_plugin \
                --use_gemm_plugin \
                --dtype float32 \
                --max_beam_width 1
```

### Run

Run a TensorRT-LLM Enc-Dec model using the engines generated by build.py.
Note that during model deployment, only the TensorRT engine files are needed. Previously downloaded model checkpoints and converted weights can be removed.

```bash
# Example 1: For T5, inference w/ single GPU, FP32, greedy search, compare results with HuggingFace FP32
python3 run.py --engine_dir tmp/trt_engines/t5-small/1-gpu/float32/tp1 --engine_name t5-small --model_name t5-small --max_new_token=64 --num_beams=1 --compare_hf_fp32

# Example 2: For T5, inference w/ 4 GPUs (4-way TP, as configured during the engine building step), BF16, greedy search, compare results with HuggingFace FP32
mpirun --allow-run-as-root -np 4 python3 run.py --engine_dir tmp/trt_engines/t5-small/4-gpu/bfloat16/tp4 --engine_name t5-small --model_name t5-small --max_new_token=64 --num_beams=1 --compare_hf_fp32

# Example 3: For T5, inference w/ 4 GPUs (2-way TP and 2-way PP, as configured during the engine building step), BF16, greedy search
mpirun --allow-run-as-root -np 4 python3 run.py --engine_dir tmp/trt_engines/flan-t5-small/4-gpu/bfloat16/tp2 --engine_name flan-t5-small --model_name google/flan-t5-small --max_new_token=64 --num_beams=1

# Example 4: For BART, inference w/ single GPU, FP32, greedy search, compare results with HuggingFace FP32
python3 run.py --engine_dir tmp/trt_engines/bart-large-cnn/1-gpu/float32/tp1 --engine_name bart-large-cnn --model_name tmp/hf_models/bart-large-cnn --max_new_token=64 --num_beams=1 --compare_hf_fp32
```

### Benchmark

The benchmark implementation and entrypoint can be found in [`benchmarks/python/benchmark.py`](../../benchmarks/python/benchmark.py). Specifically, [`benchmarks/python/enc_dec_benchmark.py`](../../benchmarks/python/enc_dec_benchmark.py) is the benchmark script for Encoder-Decoder models.

Step 1: In `examples/enc_dec/`:

After downloading the models and converting/splitting the weights, build the engine **without** the `--remove_input_padding` flag and **without** pipeline parallelism.

```bash
# Example 1: build t5-small using a single GPU, FP32, running greedy search
python build.py --model_type t5 \
                --weight_dir tmp/trt_models/t5-small/tp1 \
                -o tmp/trt_engines/t5-small/1-gpu \
                --engine_name t5-small \
                --use_bert_attention_plugin \
                --use_gpt_attention_plugin \
                --use_gemm_plugin \
                --dtype float32 \
                --max_beam_width 1

# Example 2: build t5-small using 4-way tensor parallelism on a node with 8 GPUs (but only use 4 of them for demonstration purpose), BF16, enabling beam search up to width=3
python build.py --model_type t5 \
                --world_size 4 \
                --tp_size 4 \
                --gpus_per_node 4 \
                --weight_dir tmp/trt_models/t5-small/tp4 \
                -o tmp/trt_engines/t5-small/4-gpu \
                --engine_name t5-small \
                --use_bert_attention_plugin \
                --use_gpt_attention_plugin \
                --use_gemm_plugin \
                --dtype bfloat16 \
                --max_beam_width 3
```

Step 2: In `benchmarks/python/`:

```bash
# Example 1: Single-GPU benchmark
python benchmark.py \
    -m t5_small \
    --batch_size "1;8" \
    --input_output_len "60,20;128,20" \
    --dtype float32 \
    --engine_dir ../../examples/enc_dec/tmp/trt_engines/t5-small/1-gpu/float32/tp1 \
    --csv # optional

# Example 2: Multi-GPU benchmark
mpirun --allow-run-as-root -np 4 python benchmark.py \
    -m t5_small \
    --batch_size "1;8" \
    --input_output_len "60,20;128,20" \
    --dtype bfloat16 \
    --engine_dir ../../examples/enc_dec/tmp/trt_engines/t5-small/4-gpu/bfloat16/tp4 \
    --csv # optional
```

### Reminders

- Flan-T5 models have known issues regarding FP16 precision and using BF16 precision is recommended, regardless of TRT-LLM. While we are working on improving FP16 results, please stay with FP32 or BF16 precision for Flan-T5 family.
- Batched/Ragged input with beam search is having subtle issues with some sequence results being truncated. For the time being, please follow (1) if batch size = 1, no problem (2) if batched input is padded (i.e., not using `--remove_input_padding` flag), no problem (3) if batched input is ragged (i.e., using `--remove_input_padding`), only use greedy search for now.
- For T5 and Flan-T5 family that have relative attention bias design, the relative attention table is split along `num_heads` dimension in Tensor Parallelism mode. Therefore, `num_heads` must be divisible by `tp_size`. Please be aware of this when setting the TP parameter.
- For mBART, models that can control output languages (e.g. [`mbart-large-50-many-to-many-mmt`](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt)) are not currently supported, as the script does not support `ForcedBOSTokenLogitsProcessor` to control output languages.

### Attention Scaling Factors

The `q_scaling` convention in the TRT-LLM plugin is defined as follows:
```
norm_factor = 1.f / (q_scaling * sqrt(head_size))
```
In the Multi-Head Attention (MHA) mechanism, the output of the `Q*K^T` product is scaled by this constant value `norm_factor` as `norm_factor * (Q*K^T)` for `softmax`. This scaling factor can be adjusted or neutralized based on the model's requirements.

Handling in Different Models:
- BART/FairSeq NMT: For the BART model, `q_scaling` is set to `1.f`. Therefore, the `norm_factor` for BART becomes `1.f / sqrt(head_size)`. TRT-LLM uses the default value `q_scaling = 1.f` as seen in [`bart/convert.py`](./bart/convert.py). Similar to FairSeq NMT models.
- T5: For the T5 model, `q_scaling` is `1.f/sqrt(head_size)`, leading to a `norm_factor` of `1.f`. This is handled in T5 by the TRT-LLM's `get_offset_q_scaling()` function in [`t5/convert.py`](./t5/convert.py#L158-163), which reads `head_size` from the T5 model configuration and sets `q_scaling = 1.f/sqrt(head_size)` to effectively offset the `norm_factor` to `1.f`.

### Run FairSeq NMT (Neural Machine Translation) models

FairSeq model download and library dependency are different from HuggingFace ones. Especially if you are following the recommended docker container setup in [README](../../README.md), it has a custom PyTorch build but FairSeq installation will force upgrade the PyTorch version. As a workaround, we skip the `torch` and `torchaudio` dependencies in FairSeq to make everything work nicely inside the TRT-LLM container.

```bash
# Download weights from HuggingFace Transformers
# Instructions from: https://github.com/facebookresearch/fairseq/blob/main/examples/translation/README.md#example-usage-cli-tools. Public model checkpoints are also listed there. Here we use WMT'14 Transformer model as an example.
mkdir -p tmp/fairseq_models && curl https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2 | tar xvjf - -C tmp/fairseq_models  --one-top-level=wmt14 --strip-components 1 --no-same-owner

# Install FairSeq dependency
# avoid base torch to be upgraded by fairseq
pushd tmp && (git clone https://github.com/facebookresearch/fairseq.git || true) && pushd fairseq && sed -i '/torch>=/d;/torchaudio>=/d' setup.py && pip install -e . && pip install sacremoses subword_nmt && popd && popd

# Convert and Split Weights, single GPU example
python nmt/convert.py -i tmp/fairseq_models/wmt14 -o tmp/trt_models/wmt14 --weight_data_type float32 --inference_tensor_para_size 1

# Build TensorRT engine(s)
python build.py --model_type nmt \
        --weight_dir tmp/trt_models/wmt14/tp1/  \
        -o  tmp/trt_engines/wmt14/1-gpu  \
        --engine_name wmt14 \
        --use_bert_attention_plugin \
        --use_gpt_attention_plugin \
        --dtype float32 \
        --max_beam_width 1

# Run
python3 run.py --engine_dir tmp/trt_engines/wmt14/1-gpu/float32/tp1 --engine_name wmt14 --model_name tmp/fairseq_models/wmt14 --max_new_token=24 --num_beams=1
```
