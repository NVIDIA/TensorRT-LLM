# Deployment Guide for Qwen3 on TensorRT LLM - Blackwell & Hopper Hardware

## Introduction

This is a functional quick-start guide for running the Qwen3 model on TensorRT LLM. It focuses on a working setup with recommended defaults. Additional performance optimizations and support will be rolled out in future updates.

## Prerequisites

* GPU: NVIDIA Blackwell or Hopper Architecture
* OS: Linux
* Drivers: CUDA Driver 575 or Later
* Docker with NVIDIA Container Toolkit installed
* Python3 and python3-pip (Optional, for accuracy evaluation only)

## Models

* [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)
* [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B)
* [Qwen3-235B-A22B-FP8](https://huggingface.co/Qwen/Qwen3-235B-A22B-FP8)
* [Qwen3-30B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3-30B-A3B-NVFP4)
* [Qwen3-235B-A22B-NVFP4](https://huggingface.co/nvidia/Qwen3-235B-A22B-NVFP4)

## Deployment Steps

### Run Docker Container

Build and run the docker container. See the [Docker guide](../../../docker/README.md) for details.

```shell
cd TensorRT-LLM

make -C docker release_build IMAGE_TAG=qwen3-local

make -C docker release_run IMAGE_NAME=tensorrt_llm IMAGE_TAG=qwen3-local LOCAL_USER=1
```

### Recommended Performance Settings

We maintain YAML configuration files with recommended performance settings in the [`examples/configs`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/configs) directory. These config files are present in the TensorRT LLM container at the path `/app/tensorrt_llm/examples/configs`. You can use these out-of-the-box, or adjust them to your specific use case.

```shell
TRTLLM_DIR=/app/tensorrt_llm # change as needed to match your environment
EXTRA_LLM_API_FILE=${TRTLLM_DIR}/examples/configs/curated/qwen3.yaml
```

Note: if you don't have access to the source code locally, you can manually create the YAML config file using the code in the dropdown below.

````{admonition} Show code
:class: dropdown

```{literalinclude} ../../../examples/configs/curated/qwen3.yaml
---
language: shell
prepend: |
  EXTRA_LLM_API_FILE=/tmp/config.yml

  cat << EOF > ${EXTRA_LLM_API_FILE}
append: EOF
---
```
````


### Launch the TensorRT LLM Server

Below is an example command to launch the TensorRT LLM server with the Qwen3 model from within the container.

```shell
trtllm-serve Qwen/Qwen3-30B-A3B --host 0.0.0.0 --port 8000 --config ${EXTRA_LLM_API_FILE}
```

After the server is set up, the client can now send prompt requests to the server and receive results.

### LLM API Options (YAML Configuration)

<!-- TODO: this section is duplicated across the deployment guides; they should be consolidated to a central file and imported as needed, or we can remove this and link to LLM API reference -->

These options provide control over TensorRT LLM's behavior and are set within the YAML file passed to the `trtllm-serve` command via the `--config` argument.

#### `tensor_parallel_size`

* **Description:** Sets the **tensor-parallel size**. This should typically match the number of GPUs you intend to use for a single model instance.

#### `moe_expert_parallel_size`

* **Description:** Sets the **expert-parallel size** for Mixture-of-Experts (MoE) models. Like `tensor_parallel_size`, this should generally match the number of GPUs you're using. This setting has no effect on non-MoE models.

#### `kv_cache_free_gpu_memory_fraction`

* **Description:** A value between `0.0` and `1.0` that specifies the fraction of free GPU memory to reserve for the KV cache after the model is loaded. Since memory usage can fluctuate, this buffer helps prevent out-of-memory (OOM) errors.
* **Recommendation:** If you experience OOM errors, try reducing this value to `0.7` or lower.


#### `max_batch_size`

* **Description:** The maximum number of user requests that can be grouped into a single batch for processing. The actual max batch size that can be achieved depends on total sequence length (input + output).

#### `max_num_tokens`

* **Description:** The maximum total number of tokens (across all requests) allowed inside a single scheduled batch.

#### `max_seq_len`

* **Description:** The maximum possible sequence length for a single request, including both input and generated output tokens. We won't specifically set it. It will be inferred from model config.

#### `trust_remote_code`
* **Description:** Allows TensorRT LLM to download models and tokenizers from Hugging Face. This flag is passed directly to the Hugging Face API.

#### `cuda_graph_config`

* **Description**: A section for configuring CUDA graphs to optimize performance.

* **Options**:

  * `enable_padding`: If `true`, input batches are padded to the nearest `cuda_graph_batch_size`. This can significantly improve performance.

    **Default**: `false`

  * `batch_sizes`: List of batch sizes for which CUDA graphs will be pre-captured.

    **Recommendation**: Set this to cover the range of batch sizes you expect in production.

#### `moe_config`

* **Description**: Configuration for Mixture-of-Experts (MoE) models.

* **Options**:

  * `backend`: The backend to use for MoE operations.

    **Default**: `CUTLASS`

See the [`TorchLlmArgs` class](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.TorchLlmArgs) for the full list of options which can be used in the YAML configuration file.

## Testing API Endpoint

### Basic Test

Start a new terminal on the host to test the TensorRT LLM server you just launched.

You can query the health/readiness of the server using:

```shell
curl -s -o /dev/null -w "Status: %{http_code}\n" "http://localhost:8000/health"
```

When the `Status: 200` code is returned, the server is ready for queries. Note that the very first query may take longer due to initialization and compilation.

After the TensorRT LLM server is set up and shows Application startup complete, you can send requests to the server.

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json"  -d '{
    "model": "Qwen/Qwen3-30B-A3B",
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.95
}' -w "\n"
```

Here is an example response:

```json
{
  "id": "chatcmpl-abc123def456",
  "object": "chat.completion",
  "created": 1759022940,
  "model": "Qwen/Qwen3-30B-A3B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris. Paris is not only the capital but also the largest city in France, known for its rich history, culture, art, and iconic landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral."
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 58,
    "total_tokens": 73
  }
}
```

### Troubleshooting Tips

* If you encounter CUDA out-of-memory errors, try reducing `max_batch_size`, `max_num_tokens`, or `kv_cache_free_gpu_memory_fraction`.
* Ensure your model checkpoints are compatible with the expected format.
* For performance issues, check GPU utilization with `nvidia-smi` while the server is running.
* If the container fails to start, verify that the NVIDIA Container Toolkit is properly installed.
* For connection issues, make sure the server port (`8000` in this guide) is not being used by another application.
* For MoE models (Qwen3-30B-A3B, Qwen3-235B-A22B), ensure `moe_expert_parallel_size` is properly configured.

## Benchmarking Performance

To benchmark the performance of your TensorRT LLM server you can leverage the built-in `benchmark_serving.py` script. To do this first create a wrapper `bench.sh` script.

```shell
cat <<'EOF' > bench.sh
#!/usr/bin/env bash
set -euo pipefail

# Adjust the model name based on which Qwen3 model you're benchmarking
MODEL_NAME="Qwen/Qwen3-30B-A3B"

concurrency_list="1 2 4 8 16 32 64 128"
multi_round=5
isl=1024
osl=1024
result_dir=/tmp/qwen3_output

for concurrency in ${concurrency_list}; do
    num_prompts=$((concurrency * multi_round))
    python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --model ${MODEL_NAME} \
        --backend openai \
        --dataset-name "random" \
        --random-input-len ${isl} \
        --random-output-len ${osl} \
        --random-prefix-len 0 \
        --random-ids \
        --num-prompts ${num_prompts} \
        --max-concurrency ${concurrency} \
        --ignore-eos \
        --tokenize-on-client \
        --percentile-metrics "ttft,tpot,itl,e2el"
done
EOF
chmod +x bench.sh
```

To achieve max through-put, with attention DP on, one needs to sweep up to `concurrency = max_batch_size * num_gpus`.

If you want to save the results to a file add the following options.

```shell
--save-result \
--result-dir "${result_dir}" \
--result-filename "concurrency_${concurrency}.json"
```

For more benchmarking options see [benchmark_serving.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/serve/scripts/benchmark_serving.py)

Run `bench.sh` to begin a serving benchmark. This will take a long time if you run all the concurrencies mentioned in the above `bench.sh` script.

```shell
./bench.sh
```
