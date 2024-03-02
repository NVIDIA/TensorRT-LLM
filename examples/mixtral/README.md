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

```
pip install -r requirements.txt # install latest version of transformers, needed for Mixtral

git lfs install
git clone https://huggingface.co/mistralai/Mixtral-8x7B-v0.1
```

We use the LLaMA `convert_checkpoint.py` script to convert and build the model. TensorRT-LLM LLaMA builds TensorRT engine(s) from HF checkpoint provided by `--model_dir`.
If no checkpoint directory is specified, TensorRT-LLM will build engine(s) with dummy weights.

`trtllm-build` uses one GPU by default, but if you have already more GPUs available at build time,
you may enable parallel builds to make the engine building process faster by adding the `--workers` argument.

Here are some examples:

```
# Build Mixtral8x7B with pipeline parallelism
python convert_checkpoint.py --model_dir ./Mixtral-8x7B-v0.1 \
                             --output_dir ./tllm_checkpoint_mixtral_2gpu \
                             --dtype float16 \
                             --pp_size 2
trtllm-build --checkpoint_dir ./tllm_checkpoint_mixtral_2gpu \
                 --output_dir ./trt_engines/mixtral/pp2 \
                 --gemm_plugin float16

```

```
# Build Mixtral8x7B with tensor parallelism
python convert_checkpoint.py --model_dir ./Mixtral-8x7B-v0.1 \
                             --output_dir ./tllm_checkpoint_mixtral_2gpu \
                             --dtype float16 \
                             --tp_size 2
trtllm-build --checkpoint_dir ./tllm_checkpoint_mixtral_2gpu \
                 --output_dir ./trt_engines/mixtral/tp2 \
                 --gemm_plugin float16
```

Then, you can test your engine with the [run.py](./examples/run.py) script:

```
mpirun -n 2 python3 ../run.py --engine_dir ./trt_engines/mixtral/tp2 --tokenizer_dir ./Mixtral-8x7B-v0.1 --max_output_len 8 --input_text "I love french quiche"
```


For more examples see [`examples/llama/README.md`](../llama/README.md)


### OOTB

Mixtral supports OOTB operation without the plugin, however this comes at a significant performance cost. Users should prefer using the plugin path whenever possible
