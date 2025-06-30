# Run benchmarking for Llama3.1 70B with `trtllm-serve`

TRT-LLM provides the openai-compatiable API via `trtllm-serve` command.
A complete reference for the API is available in the [OpenAI API Reference](https://platform.openai.com/docs/api-reference).

This step-by-step tutorial covers below topics for running online serving benchmarking with Llama 3.1 70B
 * Methodology Introduction
 * Launch the OpenAI-Compatibale Server with NGC container
 * Run the Performance benchmark
 * Usage of `extra_llm_api_options` knob


## Methodology Introduction

The overall performance benchmarking involves
   1. Launch the openai-compatiable service with trtllm-serve
   2. Run the benchmark with [benchmark_serving.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/serve/scripts/benchmark_serving.py)


## Launch the NGC container

TRT-LLM deploys the pre-built container on [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags).

Typically, we can launch the container using the command below.
```bash
docker run --rm --ipc host -p 6666 --gpus all -it nvcr.io/nvidia/tensorrt-llm/release:0.21.0rc1
```

## Start the trtllm-serve service
For benchmarking purposes, we first need to create a bash file (e.g., `start.sh`).
```bash
#! /bin/bash
model_path=/path/to/llama3.1_70B
extra_llm_api_file=/tmp/extra-llm-api-config.yml

cat << EOF > ${extra_llm_api_file}
enable_attention_dp: false
print_iter_log: true
use_cuda_graph: true
cuda_graph_padding_enabled: true
cuda_graph_max_batch_size: 1024
kv_cache_dtype: fp8
EOF

trtllm-llmapi-launch trtllm-serve serve  ${model_path} \
    --backend pytorch \
    --max_batch_size 1024 \
    --max_num_tokens max_num_tokens \
    --max_seq_len 2k \
    --kv_cache_free_gpu_memory_fraction 0.9 \
    --tp_size 1 \
    --ep_size 1 \
    --trust_remote_code \
    --extra_llm_api_options ${extra_llm_api_file}
```
> [!NOTE]
> The trtllm-llmapi-launch is a script that launches the LLM-API code on
> Slurm-like systems, and can support multi-node and multi-GPU setups.

Make sure that `start.sh` is run in the **background**.
```bash
bash -x start.sh &
```

## Run the benchmark

Similarly to starting `trtllm-service`, we will also create a script to execute the benchmarking.

We name this script as `bench.sh` and below is the content of bench.sh

```bash
concurrency_list="1 2 4 8 16 32 64 128 256"
multi_round=5
isl=1024
osl=1024
result_dir=/tmp/llama3.1_output
model_path=/path/to/llama3.1_70B

for concurrency in ${concurrency_list}; do
    num_prompts=$((concurrency * multi_round))
    python ${trtllm_home}/tensorrt_llm/serve/scripts/benchmark_serving.py \
        --model ${model_path} \
        --backend openai \
        --dataset-name "random" \
        --random-input-len ${isl} \
        --random-output-len ${osl} \
        --random-prefix-len 0 \
        --random-ids \
        --num-prompts ${num_prompts} \
        --max-concurrency ${concurrency} \
        --ignore-eos \
        --save-result \
        --tokenize-on-client \
        --result-dir "${result_dir}" \
        --result-filename "concurrency_${concurrency}.json" \
        --percentile-metrics "ttft,tpot,itl,e2el"
done
```

Then we can run the benchmark using the command below.

```bash
bash -x bench.sh &< output_bench.log
```


## About `extra_llm_api_options` knob
   trtllm-serve provides `extra_llm_api_options` knob to overwtie the parameters specified by trtllm-serve.
   Generally, We create a YAML file that contains various performance switches.
   e.g
   ```yaml
     use_cuda_graph: true
     cuda_graph_padding_enabled: true
     print_iter_log: true
     kv_cache_dtype: fp8
     enable_attention_dp: true
   ```

   Here is a list of common performance switches.

   * print_iter_log: Print iteration logs. Default value is `False`.
   * kv_cache_dtype: Data type for KV cache. Default value is `auto`.
   * use_cuda_graph: Use CUDA graphs for decoding. Default value is `False`.
   * cuda_graph_batch_sizes: List of batch sizes to create CUDA graphs for. Default value is `None`.
   * cuda_graph_max_batch_size: Maximum batch size for CUDA graphs.Default value is `0`.
   * cuda_graph_padding_enabled: Batches are rounded up to the nearest cuda_graph_batch_size. This is usually a net win for performance. Default value is `False`.
   * autotuner_enabled: Enable autotuner only when torch compile is enabled.Default Value is `True`.
   * moe_backend: MoE backend to use. Default value is `cutlass`.
   * attention_backend:Attention backend to use. Default value is `TRTLLM`.
