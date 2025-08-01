# Run benchmarking with `trtllm-serve`

TensorRT-LLM provides the openai-compatiable API via `trtllm-serve` command.
A complete reference for the API is available in the [OpenAI API Reference](https://platform.openai.com/docs/api-reference).

This step-by-step tutorial covers below topics for running online serving benchmarking with Llama 3.1 70B.
 * Methodology Introduction
 * Launch the OpenAI-Compatibale Server with NGC container
 * Run the Performance benchmark
 * Usage of `extra_llm_api_options` knob


## Methodology Introduction

The overall performance benchmarking involves
   1. Launch the OpenAI-compatible service with `trtllm-serve`
   2. Run the benchmark with [benchmark_serving.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/serve/scripts/benchmark_serving.py)


## Launch the NGC container

TensorRT-LLM deploys the pre-built container on [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags).

Typically, we can launch the container using the command below.
```bash
docker run --rm --ipc host -p 8000:8000 --gpus all -it nvcr.io/nvidia/tensorrt-llm/release
```

## Start the trtllm-serve service
> [!WARNING]
> The commands and configurations presented in this document are for illustrative purposes only.
> They serve as examples and may not deliver the optimal performance for your specific use case.
> Users are encouraged to tune the parameters based on their hardware and workload.
For benchmarking purposes, we first need to create a bash file (e.g., `start.sh`).
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

Make sure that `start.sh` is run in the **background**.
```bash
bash -x start.sh &
```

## Run the benchmark

Similarly to starting `trtllm-serve`, we will also create a script to execute the benchmarking.

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
bash -x bench.sh &< output_bench.log
```


## About `extra_llm_api_options` knob
   trtllm-serve provides `extra_llm_api_options` knob to overwtie the parameters specified by trtllm-serve.
   Generally, We create a YAML file that contains various performance switches.
   e.g
   ```yaml
     cuda_graph_config:
      padding_enabled: true
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
   * moe_backend: MoE backend to use. Default value is `CUTLASS`.
   * attention_backend:Attention backend to use. Default value is `TRTLLM`.
