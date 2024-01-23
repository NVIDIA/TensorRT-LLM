# GPT-J

This document explains how to build the [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6b) model using TensorRT-LLM and run on a single GPU.

## Overview

The TensorRT-LLM GPT-J implementation can be found in [`tensorrt_llm/models/gptj/model.py`](../../tensorrt_llm/models/gptj/model.py). The TensorRT-LLM GPT-J example
code is located in [`examples/gptj`](./). There is one main file:

* [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the GPT-J model.

In addition, there are two shared files in the parent folder [`examples`](../) for inference and evaluation:

* [`../run.py`](../run.py) to run the inference on an input text;
* [`../summarize.py`](../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset.

## Support Matrix
  * FP16
  * FP8
  * INT8 & INT4 per-channel weight-only
  * Groupwise quantization (AWQ)
  * INT8 KV CACHE (+ AWQ/per-channel weight-only)
  * FP8 KV CACHE

## Usage

### 1. Download weights from HuggingFace (HF) Transformers

```bash
# 1. Weights & config
git clone https://huggingface.co/EleutherAI/gpt-j-6b gptj_model
pushd gptj_model && \
  rm -f pytorch_model.bin && \
  wget https://huggingface.co/EleutherAI/gpt-j-6b/resolve/main/pytorch_model.bin && \
popd

# 2. Vocab and merge table
wget https://huggingface.co/EleutherAI/gpt-j-6b/resolve/main/vocab.json
wget https://huggingface.co/EleutherAI/gpt-j-6b/resolve/main/merges.txt
```

### 2. Build TensorRT engine(s)

TensorRT-LLM builds TensorRT engine(s) using a HF checkpoint. If no checkpoint directory is specified, TensorRT-LLM will build engine(s) using
dummy weights.

Examples of build invocations:

```bash
# Build a float16 engine using HF weights.
# Enable several TensorRT-LLM plugins to increase runtime performance. It also helps with build time.

python3 build.py --dtype=float16 \
                 --log_level=verbose \
                 --enable_context_fmha \
                 --use_gpt_attention_plugin float16 \
                 --use_gemm_plugin float16 \
                 --max_batch_size=32 \
                 --max_input_len=1919 \
                 --max_output_len=128 \
                 --remove_input_padding \
                 --output_dir=gptj_engine \
                 --model_dir=gptj_model 2>&1 | tee build.log

# Build a float16 engine using dummy weights, useful for performance tests.
# Enable several TensorRT-LLM plugins to increase runtime performance. It also helps with build time.

python3 build.py --dtype=float16 \
                 --log_level=verbose \
                 --enable_context_fmha \
                 --use_gpt_attention_plugin float16 \
                 --use_gemm_plugin float16 \
                 --max_batch_size=32 \
                 --max_input_len=1919 \
                 --max_output_len=128 \
                 --remove_input_padding \
                 --output_dir=gptj_engine_dummy_weights 2>&1 | tee build.log

# Build an int4 weight only quantization engine using awq int4 weight only quantized weights.
# Enable several TensorRT-LLM plugins to increase runtime performance. It also helps with build time.

python3 build.py --dtype=float16 \
                 --log_level=verbose \
                 --enable_context_fmha \
                 --use_gpt_attention_plugin float16 \
                 --use_gemm_plugin float16 \
                 --max_batch_size=32 \
                 --max_input_len=1919 \
                 --max_output_len=128 \
                 --remove_input_padding \
                 --output_dir=gptj_engine \
                 --use_weight_only \
                 --per_group \
                 --weight_only_precision=int4 \
                 --model_dir=awq_int4_weight_only_quantized_models 2>&1 | tee build.log

```

#### FP8 Post-Training Quantization

The examples below uses the NVIDIA AMMO (AlgorithMic Model Optimization) toolkit for the model quantization process.

First make sure AMMO toolkit is installed (see [examples/quantization/README.md](/examples/quantization/README.md#preparation))

Now quantize HF GPT-J weights as follows.
After successfully running the script, the output should be in .npz format, e.g. `quantized_fp8/gptj_tp1_rank0.npz`,
where FP8 scaling factors are stored.

```bash
# Quantize HF GPT-J 6B checkpoint into FP8 format
python examples/quantization/quantize.py --model_dir gptj_model \
                                         --dtype float16 \
                                         --qformat fp8 \
                                         --export_path ./quantized_fp8 \
                                         --calib_size 512

# Build GPT-J 6B using original HF checkpoint + PTQ scaling factors
python build.py --model_dir gptj_model \
                --quantized_fp8_model_path ./quantized_fp8/gptj_tp1_rank0.npz \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --output_dir gptj_engine_fp8_quantized \
                --enable_fp8 \
                --fp8_kv_cache \
                --strongly_typed
```

#### AWQ INT4 weight only quantization

One can enable AWQ INT4 weight only quantization with these 3 options when building engine with `build.py`:

- `--use_weight_only` enables weight only GEMMs in the network.
- `--per_group` enable groupwise weight only quantization, for GPT-J example, we support AWQ with the group size default as 128.
- `--weight_only_precision=int4` the precision of weight only quantization. Only int4 is supported for groupwise weight only quantization.

The linear layer in the AWQ int4 weight only quantized weights should have 3 parameters:
1. FP16 smoothed_weights (=weights/pre_quant_scale) with shape [n, k] ;
2. FP16 amax (the max abs values of the smoothed_weights) with shape [n, k/group_size];
3. FP16 pre_quant_scale (the smooth scales used to multiply by activation) with shape [k];

```bash
# Using AMMO to generate INT4_AWQ .npz file.
python3 ../quantization/quantize.py --model_dir ./gpt-j-6b/ \
                                    --export_path ./awq/ \
                                    --qformat int4_awq \
                                    --dtype float16 \
                                    --quantize_lm_head
```
```bash
# Build AWQ engine.
python3 build.py --dtype float16 \
                 --enable_context_fmha \
                 --remove_input_padding \
                 --use_gpt_attention_plugin float16 \
                 --use_gemm_plugin float16 \
                 --max_batch_size 32 \
                 --max_input_len 2048 \
                 --max_output_len 128 \
                 --output_dir int4_awq \
                 --use_weight_only \
                 --weight_only_precision int4_awq \
                 --per_group \
                 --model_dir ./gpt-j-6b/ \
                 --quant_ckpt_path awq/gptj_tp1_rank0.npz \
                 --quantize_lm_head
```

#### Fused MultiHead Attention (FMHA)

You can enable the FMHA kernels for GPT by adding `--enable_context_fmha` to the invocation of `build.py`. Note that it is disabled by default because of possible accuracy issues due to the use of Flash Attention.

If you find that the default fp16 accumulation (`--enable_context_fmha`) cannot meet the requirement, you can try to enable fp32 accumulation by adding `--enable_context_fmha_fp32_acc`. However, it is expected to see performance drop.

Note `--enable_context_fmha` / `--enable_context_fmha_fp32_acc` has to be used together with `--use_gpt_attention_plugin float16`.

#### INT8 KV cache
INT8 KV cache could be enabled to reduce memory footprint. It will bring more performance gains when batch size gets larger.

You can get the INT8 scale of KV cache through `hf_gptj_convert.py`:
```bash
# Enable INT8 calibration, and save scales
python hf_gptj_convert.py -i gptj_model -o gptj_int8_kv --calibrate-kv-cache -t float16
```
Now the FT-format checkpoint with INT8 KV cache scales is saved to `gptj_int8_kv/1-gpu`.
You can pass this `gptj_int8_kv/1-gpu` directory to `build.py` through the argument called `--ft_model_dir`.

INT8 KV cache could be combined with either per-channel INT8/INT4 weight-only quantization or per-group INT4 quantization (which is AWQ, actually).

**INT8 KV cache + per-channel weight-only quantization**

For example, you can enable INT8 KV cache together with per-channel INT8/INT4 weight-only quantization like the following command.

**NOTE**: The whole checkpoint together with INT8 KV scales are passed to `--ft_model_dir`.
```bash
# Enable INT8 KV cache together with per-channel INT8 weight-only quantization
python3 build.py --dtype=float16 \
                 --log_level=verbose \
                 --enable_context_fmha \
                 --use_gpt_attention_plugin float16 \
                 --use_gemm_plugin float16 \
                 --max_batch_size=32 \
                 --max_input_len=1919 \
                 --max_output_len=128 \
                 --remove_input_padding \
                 --output_dir=gptj_engine_wo_int8_kv_cache \
                 --use_weight_only \
                 --weight_only_precision=int8 \
                 --int8_kv_cache \
                 --ft_model_dir=gptj_int8_kv/1-gpu/
```

**INT8 KV cache + AWQ**

In addition, you can enable INT8 KV cache together with AWQ (per-group INT4 weight-only quantization)like the following command.

**NOTE**: AWQ checkpoint is passed through `--quant_ckpt_path`, and the INT8 scales of KV cache is through `--ft_model_dir`. Both files are generated with the same command as sections above.

```bash
# Enable INT8 KV cache together with AWQ
python3 build.py --dtype=float16 \
                 --log_level=verbose \
                 --enable_context_fmha \
                 --use_gpt_attention_plugin float16 \
                 --use_gemm_plugin float16 \
                 --max_batch_size=32 \
                 --max_input_len=1919 \
                 --max_output_len=128 \
                 --remove_input_padding \
                 --output_dir=gptj_engine_awq_int8_kv_cache/ \
                 --use_weight_only \
                 --per_group \
                 --weight_only_precision int4_awq \
                 --quant_ckpt_path awq/gptj_tp1_rank0.npz \
                 --quantize_lm_head \
                 --int8_kv_cache \
                 --ft_model_dir gptj_int8_kv/1-gpu/ \
                 --model_dir gptj_model
```

#### FP8 KV cache

One can enable FP8 for KV cache to reduce memory footprint used by KV cache and improve the accuracy over INT8 KV cache. There are 3 options need to be added to the invocation of `build.py` for that:

- `--enable_fp8` enables FP8 GEMMs in the network.
- `--fp8_kv_cache` to enable FP8 accuracy for KV cache.
- `--quantized_fp8_model_path` to provide path to the quantized model calibrated for FP8. For more details see [quantization docs](../quantization/README.md).


### 3. Run


To run a TensorRT-LLM GPT-J model:

```bash
python3 ../run.py --max_output_len=50 --engine_dir=gptj_engine --tokenizer_dir=gptj_model
```

## Summarization using the GPT-J model

The following section describes how to run a TensorRT-LLM GPT-J model to summarize the articles from the
[cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset. For each summary, the script can compute the
[ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)) scores and use the `ROUGE-1` score to validate the implementation.
The script can also perform the same summarization using the HF GPT-J model.

As previously explained, the first step is to build the TensorRT engine as described above using HF weights. You also have to install the requirements:

```bash
pip install -r requirements.txt
```

The summarization can be done using the [`../summarize.py`](../summarize.py) script as follows:

```bash
# Run the summarization task.
python3 ../summarize.py --engine_dir gptj_engine \
                        --hf_model_dir gptj_model \
                        --test_hf \
                        --batch_size 1 \
                        --test_trt_llm \
                        --tensorrt_llm_rouge1_threshold 14 \
                        --data_type fp16 \
                        --check_accuracy
```
