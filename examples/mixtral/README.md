# Mixtral

This document shows how to build and run a Mixtral model in TensorRT-LLM on both single GPU, single node multi-GPU and
multi-node multi-GPU.

## Overview

The TensorRT-LLM Mixtral implementation is based on the LLaMA model, with Mixture of Experts enabled. The implementation can
be found in [tensorrt_llm/models/llama/model.py](../../tensorrt_llm/models/llama/model.py).
See the LLaMA example [`examples/llama`](../llama) for details.

### Build TensorRT engine(s)

Get the weights by downloading from HF https://huggingface.co/mistralai/Mixtral-8x7B-v0.1.
See also https://huggingface.co/docs/transformers/main/en/model_doc/mixtral

```bash
git lfs install
git clone https://huggingface.co/mistralai/Mixtral-8x7B-v0.1
```

We use the LLaMA `convert_checkpoint.py` script to convert and build the model. TensorRT-LLM LLaMA builds TensorRT engine(s) from HF checkpoint provided by `--model_dir`.
If no checkpoint directory is specified, TensorRT-LLM will build engine(s) with dummy weights.

`trtllm-build` uses one GPU by default, but if you have already more GPUs available at build time,
you may enable parallel builds to make the engine building process faster by adding the `--workers` argument.

Here are some examples:

```bash
# Build Mixtral8x7B with pipeline parallelism
python ../llama/convert_checkpoint.py --model_dir ./Mixtral-8x7B-v0.1 \
                             --output_dir ./tllm_checkpoint_mixtral_2gpu \
                             --dtype float16 \
                             --pp_size 2
trtllm-build --checkpoint_dir ./tllm_checkpoint_mixtral_2gpu \
                 --output_dir ./trt_engines/mixtral/pp2 \
                 --gemm_plugin float16

```

```bash
# Build Mixtral8x7B with tensor parallelism
python ../llama/convert_checkpoint.py --model_dir ./Mixtral-8x7B-v0.1 \
                             --output_dir ./tllm_checkpoint_mixtral_2gpu \
                             --dtype float16 \
                             --tp_size 2
trtllm-build --checkpoint_dir ./tllm_checkpoint_mixtral_2gpu \
                 --output_dir ./trt_engines/mixtral/tp2 \
                 --gemm_plugin float16
```

Then, you can test your engine with the [run.py](../run.py) script:

```bash
mpirun -n 2 python3 ../run.py --engine_dir ./trt_engines/mixtral/tp2 --tokenizer_dir ./Mixtral-8x7B-v0.1 --max_output_len 8 --input_text "I love french quiche"
```

For more examples see [`examples/llama/README.md`](../llama/README.md)

### Parallelism Modes

Mixture of Experts supports two parallelism modes, these are Expert Parallelism (EP) and Tensor Parallelism (TP).

In TP mode (default) expert weight matrices are sliced evenly between all GPUs, so that all GPUs work together to
calculate the result for each expert.

In EP mode each GPU is assigned a subset of the expert weights matrices, so each GPU works independently to calculate
the result for its assigned experts. This may cause load balancing issues where some GPUs have more work than others,
thus increasing latency.

Enable expert parallelism by providing `--moe_tp_mode 1` to `convert_checkpoint.py`, see [tensorrt_llm/layers/moe.py](../../tensorrt_llm/layers/moe.py#L51) for available values

```bash
# Build Mixtral8x7B with Expert Parallelism Mode
python ../llama/convert_checkpoint.py --model_dir ./Mixtral-8x7B-v0.1 \
                             --output_dir ./tllm_checkpoint_mixtral_2gpu \
                             --dtype float16 \
                             --tp_size 2 \
                             --moe_tp_mode 1 # 1 is expert parallel, 2 is tensor parallel (default 2)
trtllm-build --checkpoint_dir ./tllm_checkpoint_mixtral_2gpu \
                 --output_dir ./trt_engines/mixtral/tp2 \
                 --gemm_plugin float16
```

### Normalization Modes

MOE Supports different normalization modes which influence how the scales are calculated for the final weighted sum in
of the different top-k values.

- 0 (NONE) corresponds to: `scales = topk(softmax(routing values))`
- 1 (RENORM) corresponds to: `scales = softmax(topk(routing values))`

Mixtral uses `RENORM` mode, this is set as the default. To use a different mode use the `--moe_normalization_mode` flag.
See [tensorrt_llm/layers/moe.py](../../tensorrt_llm/layers/moe.py#L56) for available values


## Quantization

### Weight-only Quantization

Mixtral supports weight only quantization

```bash
# Build Mixtral8x7B with weight only
python ../llama/convert_checkpoint.py --model_dir ./Mixtral-8x7B-v0.1 \
                             --output_dir ./tllm_checkpoint_mixtral_2gpu \
                             --dtype float16 \
                             --tp_size 2 \
                             --use_weight_only \
                             --weight_only_precision int8
trtllm-build --checkpoint_dir ./tllm_checkpoint_mixtral_2gpu \
                 --output_dir ./trt_engines/mixtral/tp2 \
                 --gemm_plugin float16
```

### FP8 Post-Training Quantization

Mixtral supports FP8 quantization, using Modelopt. See [`examples/llama/README.md`](../llama/README.md#fp8-post-training-quantization) for full details on installing Modelopt

```bash
# Quantize HF Mixtral into FP8 and export trtllm checkpoint
python ../quantization/quantize.py --model_dir ./Mixtral-8x7B-v0.1 \
                                   --dtype float16 \
                                   --qformat fp8 \
                                   --kv_cache_dtype fp8 \
                                   --output_dir ./tllm_checkpoint_mixtral_2gpu \
                                   --calib_size 512 \
                                   --tp_size 2

# Build trtllm engines from the trtllm checkpoint
# Enable fp8 context fmha to get further acceleration by setting `--use_fp8_context_fmha enable`
trtllm-build --checkpoint_dir ./tllm_checkpoint_mixtral_2gpu \
             --output_dir ./engine_outputs \
             --gemm_plugin float16 \
             --strongly_typed \
             --workers 2
```

## OOTB

Mixtral supports OOTB operation without the plugin, however this comes at a significant performance cost. Users should prefer using the plugin path whenever possible
