# Flux
This document shows how to build and run a [Flux](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main) with TensorRT-LLM.

## Overview

The TensorRT-LLM Flux implementation can be found in [tensorrt_llm/models/flux/model.py](../../tensorrt_llm/models/flux/model.py). The TensorRT-LLM Flux example code is located in [`examples/flux`](./). There are main files to build and run Flux with TensorRT-LLM:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert the Flux model into tensorrt-llm checkpoint format.
* [`run.py`](./run.py) to run the [diffusers](https://huggingface.co/docs/diffusers/index) pipeline with TensorRT engine(s) to generate images.

## Support Matrix

- [x] TP
- [x] CP
- [ ] ControlNet
- [ ] FP8

## Usage

The TensorRT-LLM Flux example code locates at [examples/flux](./). It takes HuggingFace checkpiont as input, and builds the corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### Build TensorRT engine(s)

First, download the pretrained Flux checkpoint from [HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main)

This checkpoint will be converted to the TensorRT-LLM checkpoint format by [`convert_checkpoint.py`](./convert_checkpoint.py). After that, we can build TensorRT engine(s) with the TensorRT-LLM checkpoint.

```
# Convert to TRT-LLM
python convert_checkpoint.py --model_dir ./FLUX.1-dev
trtllm-build --checkpoint_dir ./tllm_checkpoint/ \
                --max_batch_size 1 \
                --remove_input_padding disable
```

Set `--max_batch_size` to tell how many images at most you would like to generate. We disable `--remove_input_padding` since we don't need to padding Flux's patches.

After build, we can find a `./engine_output` directory, it is ready for running Flux with TensorRT-LLM now.

### Generate images

A [`run.py`](./run.py) is provided to generated images with the optimized TensorRT engines.

Just run `python run.py` and we can see an image named `flux-dev.png` will be generated:
![flux-dev.png](./flux-dev.png).

### Tensor Parallel

We can levaerage tensor parallel to further reduce latency and memory consumption on each GPU. We take 4 GPUs parallelism as an example:

```
# build engine
python convert_checkpoint.py --model_dir ./FLUX.1-dev --tp_size 4
trtllm-build --checkpoint_dir ./tllm_checkpoint/ \
                --max_batch_size 1 \
                --remove_input_padding disable
# run
mpirun -n 4 --allow-run-as-root python run.py
```

### Context Parallel

Context parallel can be used to reduce latency since it has lower communication cost.

```
# build engine
python convert_checkpoint.py --model_dir ./FLUX.1-dev --cp_size 4
trtllm-build --checkpoint_dir ./tllm_checkpoint/ \
                --max_batch_size 1 \
                --bert_attention_plugin disable \
                --remove_input_padding disable
# run
mpirun -n 4 --allow-run-as-root python run.py
```

### Combine Tensor Parallel and Context Parallel

Tensor Parallel and Context Parallel can be used together to better balance latency and memory consumption.

```
# build engine
python convert_checkpoint.py --model_dir ./FLUX.1-dev --cp_size 2 --tp_size 2
trtllm-build --checkpoint_dir ./tllm_checkpoint/ \
                --max_batch_size 1 \
                --bert_attention_plugin disable \
                --remove_input_padding disable
# run
mpirun -n 4 --allow-run-as-root python run.py
```
