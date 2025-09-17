# Run benchmarking with `trtllm-serve`

TensorRT LLM provides the OpenAI-compatiable API via `trtllm-serve` command.
A complete reference for the API is available in the [OpenAI API Reference](https://platform.openai.com/docs/api-reference).

This step-by-step tutorial covers the following topics for running online serving benchmarking with Llama 3.1 70B:
 * Methodology Introduction
 * Launch the OpenAI-Compatibale Server with NGC container
 * Run the performance benchmark
 * Using `extra_llm_api_options`


## Methodology Introduction

The overall performance benchmarking involves:
   1. Launch the OpenAI-compatible service with `trtllm-serve`
   2. Run the benchmark with [benchmark_serving.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/serve/scripts/benchmark_serving.py)


## Launch the NGC container

TensorRT LLM distributes the pre-built container on [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags).

You can launch the container using the following command:

```bash
docker run --rm -it --ipc host -p 8000:8000 --gpus all --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/tensorrt-llm/release:x.y.z
```


## Start the trtllm-serve service
> [!WARNING]
> The commands and configurations presented in this document are for illustrative purposes only.
> They serve as examples and may not deliver the optimal performance for your specific use case.
> Users are encouraged to tune the parameters based on their hardware and workload.
For benchmarking purposes, first create a bash script using the following code and name it start.sh.
```bash
#! /bin/bash
model_path=/path/to/llama3.1_70B
extra_llm_api_file=/tmp/extra-llm-api-config.yml

cat << EOF > ${extra_llm_api_file}
enable_attention_dp: false
print_iter_log: true
cuda_graph_config:
  enable_padding: true
  max_batch_size: 1024
kv_cache_config:
  dtype: fp8
EOF

trtllm-serve ${model_path} \
    --max_batch_size 1024 \
    --max_num_tokens 2048 \
    --max_seq_len 1024 \
    --kv_cache_free_gpu_memory_fraction 0.9 \
    --tp_size 1 \
    --ep_size 1 \
    --trust_remote_code \
    --extra_llm_api_options ${extra_llm_api_file}
```
> [!NOTE]
> The trtllm-llmapi-launch is a script that launches the LLM-API code on
> Slurm-like systems, and can support multi-node and multi-GPU setups.
> e.g, trtllm-llmapi-launch trtllm-serve .....

Run the start.sh script in the **background** with the following command:

```bash
bash -x start.sh &
```

Once the serving is set up, it will generate the output log as shown below.
```bash
INFO:     Started server process [80833]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:8000 (Press CTRL+C to quit)
```

## Run the benchmark

Similar to starting trtllm-serve, create a script to execute the benchmark using the following code and name it bench.sh.

```bash
concurrency_list="1 2 4 8 16 32 64 128 256"
multi_round=5
isl=1024
osl=1024
result_dir=/tmp/llama3.1_output
model_path=/path/to/llama3.1_70B

for concurrency in ${concurrency_list}; do
    num_prompts=$((concurrency * multi_round))
    python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --model ${model_path} \
        --backend openai \
        --dataset-name "random" \
        --random-input-len ${isl} \
        --random-output-len ${osl} \
        --random-prefix-len 0 \
        --num-prompts ${num_prompts} \
        --max-concurrency ${concurrency} \
        --ignore-eos \
        --save-result \
        --result-dir "${result_dir}" \
        --result-filename "concurrency_${concurrency}.json" \
        --percentile-metrics "ttft,tpot,itl,e2el"
done
```

Then we can run the benchmark using the command below.

```bash
bash -x bench.sh &> output_bench.log
```

Below is some example TensorRT LLM serving benchmark output. Your actual results may vary.

```
============ Serving Benchmark Result ============
Successful requests:                     1
Benchmark duration (s):                  1.64
Total input tokens:                      1024
Total generated tokens:                  1024
Request throughput (req/s):              0.61
Output token throughput (tok/s):         622.56
Total Token throughput (tok/s):          1245.12
User throughput (tok/s):                 623.08
Mean Request AR:                         0.9980
Median Request AR:                       0.9980
---------------Time to First Token----------------
Mean TTFT (ms):                          12.83
Median TTFT (ms):                        12.83
P99 TTFT (ms):                           12.83
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          1.59
Median TPOT (ms):                        1.59
P99 TPOT (ms):                           1.59
---------------Inter-token Latency----------------
Mean ITL (ms):                           1.59
Median ITL (ms):                         1.59
P99 ITL (ms):                            1.77
----------------End-to-end Latency----------------
Mean E2EL (ms):                          1643.44
Median E2EL (ms):                        1643.44
P99 E2EL (ms):                           1643.44
==================================================
```

### Key Metrics

* Median Time to First Token (TTFT)
  * The typical time elapsed from when a request is sent until the first output token is generated.
* Median Time Per Output Token (TPOT)
  * The typical time required to generate each token *after* the first one.
* Median Inter-Token Latency (ITL)
  * The typical time delay between the completion of one token and the completion of the next.
* Median End-to-End Latency (E2EL)
  * The typical total time from when a request is submitted until the final token of the response is received.
* Total Token Throughput
  * The combined rate at which the system processes both input (prompt) tokens and output (generated) tokens.

## About `extra_llm_api_options`
   trtllm-serve provides `extra_llm_api_options` knob to **overwrite** the parameters specified by trtllm-serve.
   Generally, We create a YAML file that contains various performance switches.
   e.g
   ```yaml
     cuda_graph_config:
      padding_enabled: true
     print_iter_log: true
     kv_cache_dtype: fp8
     enable_attention_dp: true
   ```

The following is a list of common performance switches.
#### `kv_cache_config`

&emsp;**Description**: A section for configuring the Key-Value (KV) cache.

&emsp;**Options**:

&emsp;&emsp;dtype: Sets the data type for the KV cache.

&emsp;&emsp;**Default**: auto (uses the data type specified in the model checkpoint).

#### `cuda_graph_config`

&emsp;**Description**: A section for configuring CUDA graphs to optimize performance.

&emsp;**Options**:

&emsp;&emsp;enable\_padding: If true, input batches are padded to the nearest cuda\_graph\_batch\_size. This can significantly improve performance.

&emsp;&emsp;**Default**: false

&emsp;&emsp;max\_batch\_size: Sets the maximum batch size for which a CUDA graph will be created.

&emsp;&emsp;**Default**: 0

&emsp;&emsp;**Recommendation**: Set this to the same value as the \--max\_batch\_size command-line option.

&emsp;&emsp;batch\_sizes: A specific list of batch sizes to create CUDA graphs for.

&emsp;&emsp;**Default**: None

#### `moe_config`

&emsp;**Description**: Configuration for Mixture-of-Experts (MoE) models.

&emsp;**Options**:

&emsp;&emsp;backend: The backend to use for MoE operations.

&emsp;&emsp;**Default**: CUTLASS

#### `attention_backend`

&emsp;**Description**: The backend to use for attention calculations.

&emsp;**Default**: TRTLLM

See the [TorchLlmArgs class](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.TorchLlmArgs) for the full list of options which can be used in the extra\_llm\_api\_options`.`
