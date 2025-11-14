# Mixtral

This document shows how to build and run a Mixtral model in TensorRT LLM on both single GPU, single node multi-GPU and
multi-node multi-GPU.  Mixtral 8x22B is also supported and can be replace Mixtral 8x7B below as long as GPU memory is
sufficient.

## Overview

The TensorRT LLM Mixtral implementation is based on the LLaMA model, with Mixture of Experts enabled. The implementation can
be found in [tensorrt_llm/models/llama/model.py](../../../../tensorrt_llm/models/llama/model.py).
See the LLaMA example [`examples/models/core/llama`](../llama) for details.

### Build TensorRT engine(s)

#### Download Mixtral 8x7b weights
Get the weights by downloading from HF https://huggingface.co/mistralai/Mixtral-8x7B-v0.1.
See also https://huggingface.co/docs/transformers/main/en/model_doc/mixtral

```bash
git lfs install
git clone https://huggingface.co/mistralai/Mixtral-8x7B-v0.1
```

#### Download Mixtral 8x22b weights
Get the weights by downloading from HF https://huggingface.co/mistralai/Mixtral-8x22B-v0.1.
See also https://huggingface.co/docs/transformers/main/en/model_doc/mixtral

```bash
git lfs install
git clone https://huggingface.co/mistralai/Mixtral-8x22B-v0.1
```

We use the LLaMA `convert_checkpoint.py` script to convert and build the model. TensorRT LLM LLaMA builds TensorRT engine(s) from HF checkpoint provided by `--model_dir`.
If no checkpoint directory is specified, TensorRT LLM will build engine(s) with dummy weights.

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
                             --tp_size 2 \
                             --moe_tp_size 2
trtllm-build --checkpoint_dir ./tllm_checkpoint_mixtral_2gpu \
                 --output_dir ./trt_engines/mixtral/tp2 \
                 --gemm_plugin float16


# Build Mixtral8x22B with tensor parallelism and expert parallelism
python ../llama/convert_checkpoint.py --model_dir ./Mixtral-8x22B-v0.1 \
                             --output_dir ./tllm_checkpoint_mixtral_8gpu \
                             --dtype float16 \
                             --tp_size 8 \
                             --moe_tp_size 2 \
                             --moe_ep_size 4
trtllm-build --checkpoint_dir ./tllm_checkpoint_mixtral_8gpu \
                 --output_dir ./trt_engines/mixtral/tp2ep4 \
                 --gemm_plugin float16
```

Then, you can test your engine with the [run.py](../../../run.py) script:

```bash
mpirun -n 2 python3 ../../../run.py --engine_dir ./trt_engines/mixtral/tp2 --tokenizer_dir ./Mixtral-8x7B-v0.1 --max_output_len 8 --input_text "I love french quiche"
```

For more examples see [`examples/models/core/llama/README.md`](../llama/README.md)

### Parallelism Modes

Mixture of Experts supports 3 parallelism modes, these are Expert Parallelism (EP), Tensor Parallelism (TP), and the hybrid of the two (TP+EP).

In TP mode (default) expert weight matrices are sliced evenly between all GPUs, so that all GPUs work together to calculate the result for each expert.

In EP mode each GPU is assigned a subset of the expert weights matrices, so each GPU works independently to calculate the result for its assigned experts. This may cause load balancing issues where some GPUs have more work than others, thus increasing latency.

In TP+EP mode, both strategies are used simultaneously. This means each GPU handles a portion of the expert weights matrices (as in EP mode) and these weights are further sliced across multiple GPUs (as in TP mode). This hybrid approach aims to balance the workload more evenly across GPUs, enhancing efficiency and reducing the likelihood of bottlenecks associated with EP mode alone.

You can enable Expert Parallel or hybrid parallel by setting `--moe_tp_size` and `--moe_ep_size` when calling `convert_coneckpoint.py`. If only `--moe_tp_size` is provided, TRT-LLM will use Tensor Parallel for the MoE model; if only `--moe_ep_size` is provided, TRT-LLM will use Expert Parallel; if both are provided, the hybrid parallel will be used.

Be sure that the product of `moe_tp_size` and `moe_ep_size` should equal to `tp_size`, since the total number of MoE parallelism across all GPUs must match the total number of parallelism in other parts of the model.

```bash
# Build Mixtral8x7B with Expert Parallelism
python ../llama/convert_checkpoint.py --model_dir ./Mixtral-8x7B-v0.1 \
                             --output_dir ./tllm_checkpoint_mixtral_2gpu \
                             --dtype float16 \
                             --tp_size 2 \
                             --moe_ep_size 2
trtllm-build --checkpoint_dir ./tllm_checkpoint_mixtral_2gpu \
                 --output_dir ./trt_engines/mixtral/ep2 \
                 --gemm_plugin float16

# Build Mixtral8x7B with Expert Parallelism and Tensor Parallelism
python ../llama/convert_checkpoint.py --model_dir ./Mixtral-8x7B-v0.1 \
                             --output_dir ./tllm_checkpoint_mixtral_4gpu \
                             --dtype float16 \
                             --tp_size 4 \
                             --moe_tp_size 2 \
                             --moe_ep_size 2
trtllm-build --checkpoint_dir ./tllm_checkpoint_mixtral_4gpu \
                 --output_dir ./trt_engines/mixtral/tp2ep2 \
                 --gemm_plugin float16
```

### Normalization Modes

MOE Supports different normalization modes which influence how the scales are calculated for the final weighted sum in
of the different top-k values.

- 0 (NONE) corresponds to: `scales = topk(softmax(routing values))`
- 1 (RENORM) corresponds to: `scales = softmax(topk(routing values))`
- 2 (SPARSE_MIXER) corresponds to: `scales = sparsemixer(routing values)`

Mixtral uses `RENORM` mode, this is set as the default. To use a different mode use the `--moe_normalization_mode` flag.
See [tensorrt_llm/layers/moe.py](../../../../tensorrt_llm/layers/moe.py#L56) for available values


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

Mixtral supports FP8 quantization, using Modelopt. See [`examples/models/core/llama/README.md`](../llama/README.md#fp8-post-training-quantization) for full details on installing Modelopt

```bash
# Quantize HF Mixtral into FP8 and export trtllm checkpoint
python ../../../quantization/quantize.py --model_dir ./Mixtral-8x7B-v0.1 \
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
             --workers 2
```

### AWQ Quantization

Mixtral supports AWQ quantization using [AutoAWQ](https://github.com/casper-hansen/AutoAWQ).

```bash
# Convert AutoAWQ HF checkpoints into TRT-LLM checkpoint
python ../llama/convert_checkpoint.py --model_dir ./tmp/mixtral-8x7b-v0.1-AWQ/ \
                                      --output_dir ./tllm_checkpoint_mixtral_awq_1gpu

# Build trtllm engines from the trtllm checkpoint
trtllm-build --checkpoint_dir ./tllm_checkpoint_mixtral_awq_1gpu \
             --output_dir ./engine_outputs
```

You may found `quant_algo = W4A16_GPTQ` in the configuration file of the converted checkpoints, and that's because AutoAWQ is using exactly the same components as GPTQ.

### NVFP4 Post-Training Quantization

Mixtral supports NVFP4 quantization.

```bash
# Quantize HF Mixtral into FP8 and export trtllm checkpoint
python ../../../quantization/quantize.py --model_dir ./Mixtral-8x7B-v0.1 \
                                   --dtype float16 \
                                   --qformat nvfp4 \
                                   --kv_cache_dtype fp8 \
                                   --output_dir ./tllm_checkpoint_mixtral_nvfp4_1gpu \
                                   --calib_size 512 \
                                   --tp_size 1

# Build trtllm engines from the trtllm checkpoint
# Enable fp8 context fmha to get further acceleration by setting `--use_fp8_context_fmha enable`
trtllm-build --checkpoint_dir ./tllm_checkpoint_mixtral_nvfp4_1gpu \
             --output_dir ./engine_outputs
```

## OOTB

Mixtral supports OOTB operation without the plugin, however this comes at a significant performance cost. Users should prefer using the plugin path whenever possible
