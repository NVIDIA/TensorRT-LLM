# DBRX

This document shows how to build and run a DBRX model in TensorRT-LLM. DBRX is a leading large language model trained by Databricks. Read more details about the model here: [DBRX Technical Blog](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)

## Overview

The TensorRT-LLM DBRX implementation can be found in [tensorrt_llm/models/dbrx/model.py](../../tensorrt_llm/models/dbrx/model.py).

## Support Matrix
  * BF16
  * FP16
  * INT8 Weight-Only
  * INT8 KV CACHE
  * Tensor Parallel
  * Pipeline Parallel
  * Expert Parallel

### Build TensorRT engine(s)

Get the weights by downloading from HF https://huggingface.co/databricks/dbrx-base
To use instruct model, download https://huggingface.co/databricks/dbrx-instruct

```bash
git lfs install
git clone https://huggingface.co/databricks/dbrx-base
```

We use the DBRX `convert_checkpoint.py` script to convert and build the model. TensorRT-LLM DBRX builds TensorRT engine(s) from HF checkpoint provided by `--model_dir`.

`trtllm-build` uses one GPU by default, but if you have already more GPUs available at build time,
you may enable parallel builds to make the engine building process faster by adding the `--workers` argument.
Make sure to use atleast 4 GPUs when working with 16-bit precision engines as model weights would not fit on 2 x 80GB-DRAM GPUs. Use int8 weights if you want to run on 2 GPUs.

Here are some examples:

```bash
# Build DBRX with tensor parallelism
python convert_checkpoint.py --model_dir ./dbrx \
                             --output_dir ./tllm_checkpoint_dbrx_8gpu \
                             --dtype bfloat16 \
                             --tp_size 8
trtllm-build --checkpoint_dir ./tllm_checkpoint_dbrx_8gpu \
                 --output_dir ./trt_engines/dbrx/8-gpu \
                 --gemm_plugin bfloat16 \
                 --gpt_attention_plugin bfloat16 \
                 --moe_plugin bfloat16
```

```bash
# Build DBRX with pipeline and tensor parallelism
python convert_checkpoint.py --model_dir ./dbrx \
                             --output_dir ./tllm_checkpoint_dbrx_8gpu \
                             --dtype bfloat16 \
                             --pp_size 4 \
                             --tp_size 4
trtllm-build --checkpoint_dir ./tllm_checkpoint_dbrx_8gpu \
                 --output_dir ./trt_engines/dbrx/8-gpu \
                 --gemm_plugin bfloat16 \
                 --gpt_attention_plugin bfloat16 \
                 --moe_plugin bfloat16

```

```bash
# Build DBRX with expert parallelism for DbrxExperts layer and tensor parallelism for rest
# `moe_tp_mode` decides sharding for expert weights
# 1 is for expert parallel, 2 for tensor parallel
python convert_checkpoint.py --model_dir ./dbrx \
                             --output_dir ./tllm_checkpoint_dbrx_8gpu \
                             --dtype bfloat16 \
                             --tp_size 8 \
                             --moe_tp_mode 1 
trtllm-build --checkpoint_dir ./tllm_checkpoint_dbrx_8gpu \
                 --output_dir ./trt_engines/dbrx/8-gpu \
                 --gpt_attention_plugin bfloat16 \
                 --gemm_plugin bfloat16 \
                 --moe_plugin bfloat16

```

```bash
# Build the DBRX model with INT8 weight-only quantization.
python convert_checkpoint.py --model_dir ./dbrx \
                              --output_dir ./tllm_checkpoint_dbrx_4gpu \
                              --dtype float16 \
                              --use_weight_only \
                              --weight_only_precision int8 \
                              --tp_size 4

trtllm-build --checkpoint_dir ./tllm_checkpoint_dbrx_4gpu \
            --output_dir ./trt_engines/dbrx/weight_only/4-gpu/ \
            --gemm_plugin float16 \
            --gpt_attention_plugin float16
```

#### INT8 KV cache
INT8 KV cache can be enabled to reduce the memory footprint. It will bring performance gains at large batch sizes and long sequence lengths.

For INT8 KV cache, [`convert_checkpoint.py`](./convert_checkpoint.py) features a
`--int8_kv_cache` option. Setting `--int8_kv_cache` will calibrate the model,
and then export the scaling factors needed for INT8 KV cache inference.

Example:

```bash
python convert_checkpoint.py --model_dir ./dbrx   \
                             --output_dir ./tllm_checkpoint_4gpu_int8_kv \
                             --dtype float16  \
                             --int8_kv_cache
```


**INT8 KV cache + per-channel weight-only quantization**

INT8 KV cache could be combined with per-channel weight-only quantization, as follows:

```bash
# Build model with both INT8 weight-only and INT8 KV cache enabled
python convert_checkpoint.py --model_dir ./dbrx   \
                             --output_dir ./tllm_checkpoint_4gpu_int8_kv_wq \
                             --dtype float16  \
                             --int8_kv_cache \
                             --use_weight_only \
                             --weight_only_precision int8 \
                             --tp_size 4

trtllm-build --checkpoint_dir ./tllm_checkpoint_4gpu_int8_kv_wq \
            --output_dir ./trt_engines/dbrx/int8_kv_cache_weight_only/4-gpu \
            --gemm_plugin float16
```

### Run the engine

DBRX uses tiktoken as the tokenizer. Make sure it is installed:

```bash
pip install -r requirements.txt
```

Then, you can test your engine with the [run.py](../run.py) script:

```bash
mpirun -n 8 python3 ../run.py --engine_dir ./trt_engines/dbrx/tp8 --tokenizer_dir ./dbrx --max_output_len 10 --input_text "What is AGI?"
```
