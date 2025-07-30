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
-->

# Testing TensorRT LLM backend

Tests in this CI directory can be run manually to provide extensive testing.

## Run QA Tests

Run the testing within the Triton container.

```bash
docker run --rm -it --net host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -v /path/to/tensorrtllm_backend:/opt/tritonserver/tensorrtllm_backend nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3 bash

# Change directory to the test and run the test.sh script
cd /opt/tritonserver/tensorrtllm_backend/ci/L0_backend_trtllm
bash -x ./test.sh
```

## Run the e2e/benchmark_core_model to benchmark

These two tests are ran in the [L0_backend_trtllm](./L0_backend_trtllm/)
test. Below are the instructions to run the tests manually.

### Generate the model repository

Follow the instructions in the
[Create the model repository](../README.md#prepare-the-model-repository)
section to prepare the model repository.

### Modify the model configuration

Follow the instructions in the
[Modify the model configuration](../README.md#modify-the-model-configuration)
section to modify the model configuration based on the needs.

### End to end test

[End to end test script](../tools/inflight_batcher_llm/end_to_end_test.py) sends
requests to the deployed `ensemble` model.

Ensemble model is ensembled by three models: `preprocessing`, `tensorrt_llm` and `postprocessing`:
- "preprocessing": This model is used for tokenizing, meaning the conversion from prompts(string) to input_ids(list of ints).
- "tensorrt_llm": This model is a wrapper of your TensorRT LLM model and is used for inferencing
- "postprocessing": This model is used for de-tokenizing, meaning the conversion from output_ids(list of ints) to outputs(string).

The end to end latency includes the total latency of the three parts of an ensemble model.

```bash
cd tools/inflight_batcher_llm
python3 end_to_end_test.py --dataset <dataset path>
```

Expected outputs
```
[INFO] Functionality test succeed.
[INFO] Warm up for benchmarking.
[INFO] Start benchmarking on 125 prompts.
[INFO] Total Latency: 11099.243 ms
```

### benchmark_core_model

[benchmark_core_model script](../tools/inflight_batcher_llm/benchmark_core_model.py)
sends requests directly to the deployed `tensorrt_llm` model, the benchmark_core_model
latency indicates the inference latency of TensorRT-LLM, not including the
pre/post-processing latency which is usually handled by a third-party library
such as HuggingFace.

```bash
cd tools/inflight_batcher_llm
python3 benchmark_core_model.py dataset --dataset <dataset path>
```

Expected outputs

```
[INFO] Warm up for benchmarking.
[INFO] Start benchmarking on 125 prompts.
[INFO] Total Latency: 10213.462 ms
```
*Please note that the expected outputs in that document are only for reference, specific performance numbers depend on the GPU you're using.*
