<!--
# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
!-->

# Running Disaggregated Serving with Triton TensorRT LLM Backend

## Overview

Disaggregated serving refers to a technique that uses separate GPUs for
running the context and generation phases of LLM inference.

For Triton integration, a BLS model named
[_disaggregated\_serving\_bls_](./disaggregated_serving_bls/1/model.py)
has been created that orchestrates the disaggregated serving pipeline. This
BLS model requires the TRT-LLM model names that are going to be used for
context and generation phases.

This example assumes access to a two GPU device systems with CUDA_VISIBLE_DEVICES
set to `0,1`.

## Model Repository Setup and Start Server

1. Setup the model repository as instructed in the [LLaMa](../docs/llama.md)
guide.

2. Create context and generation models with the desired tensor-parallel
configuration. We will be using `context` and `generation` model names for
context and generation models respectively. The context and generation models
should be copying the config
[tensorrt_llm](../inflight_batcher_llm/tensorrt_llm/) model.

3. Set the `participant_ids` for context and generation models to `1` and `2` respectively.

4. Set the `gpu_device_ids` for context and generation models to `0` and `1` respectively.

5. Set the `context_model_name` and `generation_model_name` to `context` and `generation` in the
[disaggregated_serving_bls](./disaggregated_serving_bls/config.pbtxt) model configuration.

Your model repository should look like below:

```
disaggreagted_serving/
|-- context
|   |-- 1
|   `-- config.pbtxt
|-- disaggregated_serving_bls
|   |-- 1
|   |   `-- model.py
|   `-- config.pbtxt
|-- ensemble
|   |-- 1
|   `-- config.pbtxt
|-- generation
|   |-- 1
|   `-- config.pbtxt
|-- postprocessing
|   |-- 1
|   |   `-- model.py
|   `-- config.pbtxt
`-- preprocessing
    |-- 1
    |   `-- model.py
    `-- config.pbtxt
```

6. Rename the `tensorrt_llm` model in the `ensemble` config.pbtxt file to `disaggregated_serving_bls`.

7. Launch the Triton Server:

```
python3 scripts/launch_triton_server.py --world_size 3 --tensorrt_llm_model_name context,generation --multi-model --disable-spawn-processes
```

> ![NOTE]
>
> The world size should be equal to `tp*pp` of context model + `tp*pp` of generation model + 1.
> The additional process is required for the orchestrator.

6. Send a request to the server.

```
python3 inflight_batcher_llm/client/end_to_end_grpc_client.py -S -p "Machine learning is"
```

## Creating Multiple Copies of the Context and Generation Models (Data Parallelism)

You can also create multiple copies of the context and generation models. This can be
achieved by setting the `participant_ids` and `gpu_device_ids` for each instance.

For example, if you have a context model with `tp=2` and you want to create 2
copies of it, you can set the `participant_ids` to `1,2;3,4`,
`gpu_device_ids` to `0,1;2,3` (assuming a 4-GPU system), and set the `count`
in `instance_groups` section of the model configuration to 2. This will create 2
copies of the context model where the first copy will be on GPU 0 and 1, and the
second copy will be on GPU 2 and 3.

## Known Issues

1. Only C++ version of the backend is supported right now.
