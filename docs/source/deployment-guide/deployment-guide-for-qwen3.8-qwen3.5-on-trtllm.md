# Deployment Guide for Qwen3.8 MoE and Qwen3.5 MoE on TensorRT LLM - Blackwell Hardware

## Introduction

This guide describes how to serve the Qwen3.8 MoE and Qwen3.5 MoE hybrid models with the TensorRT LLM PyTorch backend. It covers the following checkpoints:

* **Qwen3.8-2.4T-A95B MoE FP8** — 2.4 trillion total parameters and 95 billion active parameters per token.
* **Qwen3.5-397B-A17B MoE NVFP4** — 397 billion total parameters and 17 billion active parameters per token.

The models share the `qwen3_5_moe_text` decoder architecture and use the same TensorRT LLM implementation, registered as `Qwen3_5MoeForCausalLM`. Both interleave three gated-delta-network (GDN) linear-attention layers with one grouped-query-attention (GQA) layer, and both use 512 routed experts with top-10 routing.

Sharing an implementation does not make their deployment configurations interchangeable. Qwen3.8 MoE is substantially larger, uses an FP8 checkpoint, and is text-only. The NVIDIA Qwen3.5 MoE NVFP4 checkpoint uses a multimodal wrapper around the shared text decoder. Parallelism, quantization backends, cache sizing, and expert placement must be selected for the exact checkpoint.

## Architecture and Checkpoint Differences

| Property | Qwen3.8-2.4T-A95B MoE FP8 | Qwen3.5-397B-A17B MoE NVFP4 |
|---|---:|---:|
| Hugging Face text architecture | `Qwen3_5MoeForCausalLM` | `Qwen3_5MoeForCausalLM` inside `Qwen3_5MoeForConditionalGeneration` |
| Decoder layers | 92 | 60 |
| Hidden size | 8192 | 4096 |
| GDN / GQA layers | 69 / 23 | 45 / 15 |
| Routed experts / experts per token | 512 / 10 | 512 / 10 |
| Built-in MTP modules | 1 recurrent module | 1 recurrent module |
| Recommended checkpoint precision in this guide | Block-scaled FP8 routed experts | NVFP4 with FP8 KV cache |
| Modality | Text | Text, image, and video |

The Qwen3.8 MoE FP8 checkpoint quantizes the routed expert projections to FP8 (E4M3) with 128x128 weight block scales and dynamic activation scaling. Attention, GDN projections, shared experts, routers, embeddings, and the LM head are excluded from quantization and retain their checkpoint dtypes.

The serving configuration controls the attention KV-cache dtype and GDN recurrent-state dtype independently through `kv_cache_config.dtype` and `kv_cache_config.mamba_ssm_cache_dtype`. The GDN state supports `auto`, `float16`, `bfloat16`, and `float32`.

## Support Status

| Capability | Qwen3.8 MoE FP8 | Qwen3.5 MoE NVFP4 |
|---|---|---|
| Aggregated serving | Supported | Supported |
| Disaggregated serving | Supported | Supported |
| MTP speculative decoding | Supported | Supported |
| Static EPLB | Supported | Supported |
| Online EPLB | Supported | Supported |
| Attention data parallelism and expert parallelism | Supported | Supported |
| GDN state replay | Supported | Supported |
| KV cache manager V2 | Supported and default | Supported and default |

## Performance

The performance reference uses GB300, aggregated serving, and an exact 8192-input/1024-output-token workload. Low-latency profiles use TP16/EP1 on 16 GPUs at concurrency 1. High-throughput profiles use attention DP with TP32/EP32 on 32 GPUs and a 544-slot static EPLB map. Low-latency throughput is output tokens per second per user (`1000 / median TPOT`); high-throughput is total input-plus-output tokens per second per GPU.

The following are the best audited Qwen3.8 MoE FP8 checkpoint points across the validated implementation variants:

| Objective | Speculative decoding | Topology | Concurrency | Best audited value |
|---|---|---|---:|---:|
| Low latency | Disabled | TP16/EP1 | 1 | **133.502 output tok/s/user** (7.491 ms median TPOT) |
| Low latency | MTP3 | TP16/EP1 | 1 | **383.051 output tok/s/user** (2.611 ms median TPOT) |
| High throughput | Disabled | Attention DP32/EP32, Static544 | 3264 | **4059.200 total tok/s/GPU** |
| High throughput | MTP3 | Attention DP32/EP32, Static544 | 2304 | **4118.164 total tok/s/GPU** |

MTP3 performance results use a controlled accepted-draft count of 2.3.

## Prerequisites

* GPU: NVIDIA Blackwell or Hopper Architecture
* OS: Linux
* Drivers: CUDA Driver 575 or Later
* Docker with NVIDIA Container Toolkit installed
* Python3 and python3-pip (Optional, for accuracy evaluation only)

## Models

* [Qwen/Qwen3.8-2.4T-A95B-FP8](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B-FP8) (FP8; the checkpoint the Qwen3.8 profiles in this guide target)
* [Qwen/Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) (base, BF16)
* [nvidia/Qwen3.5-397B-A17B-NVFP4](https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4)
* [Qwen/Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) (base, BF16)

## GPU Requirements

The following table lists the minimum validated GPU counts for the checkpoints and platforms covered by this guide:

| Checkpoint | Platform | Minimum GPUs |
|---|---|---|
| Qwen3.8-2.4T-A95B MoE FP8 | B200 | 16x B200 |
| Qwen3.8-2.4T-A95B MoE FP8 | B300 | 16x B300 |
| Qwen3.8-2.4T-A95B MoE FP8 | GB200 | 16x GB200 |
| Qwen3.8-2.4T-A95B MoE FP8 | GB300 | 16x GB300 |
| Qwen3.5-397B-A17B MoE NVFP4 | B200 | 4x B200 |
| Qwen3.5-397B-A17B MoE NVFP4 | B300 | 4x B300 |
| Qwen3.5-397B-A17B MoE NVFP4 | GB200 | 4x GB200 |
| Qwen3.5-397B-A17B MoE NVFP4 | GB300 | 4x GB300 |

The Qwen3.8 MoE FP8 checkpoint uses `tensor_parallel_size: 16` for the minimum-GPU configurations on B200, B300, GB200, and GB300. The Qwen3.5 MoE NVFP4 checkpoint has been validated with `tensor_parallel_size: 4`.

## Deployment Steps

### Run Docker Container

Run the docker container using the TensorRT LLM NVIDIA NGC image.

```shell
docker run --rm -it \
--ipc=host \
--gpus all \
-p 8000:8000 \
-v ~/.cache:/root/.cache:rw \
--name tensorrt_llm \
nvcr.io/nvidia/tensorrt-llm/release:x.y.z \
/bin/bash
```

Note:

* The command mounts your user `.cache` directory to save the downloaded model checkpoints which are saved to `~/.cache/huggingface/hub/` by default. This prevents having to redownload the weights each time you rerun the container. If the `~/.cache` directory doesn't exist please create it using `$ mkdir ~/.cache`.
* You can mount additional directories and paths using the `-v <host_path>:<container_path>` flag if needed, such as mounting the downloaded weight paths.
* The command also maps port `8000` from the container to your host so you can access the LLM API endpoint from your host.
* See the <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags> for all the available containers. The containers published in the main branch weekly have `rcN` suffix, while the monthly release with QA tests has no `rcN` suffix. Use the `rc` release to get the latest model and feature support.

If you want to use latest main branch, you can choose to build from source to install TensorRT LLM, the steps refer to [https://nvidia.github.io/TensorRT-LLM/latest/installation/build-from-source.html](https://nvidia.github.io/TensorRT-LLM/latest/installation/build-from-source.html).

### Recommended Performance Settings

We maintain YAML configuration files with recommended performance settings in the [`examples/configs`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/configs) directory. These config files are present in the TensorRT LLM container at the path `/app/tensorrt_llm/examples/configs`. You can use these out-of-the-box, or adjust them to your specific use case.

Set the TensorRT LLM directory and select one of the configuration files listed below:

```shell
TRTLLM_DIR=/app/tensorrt_llm # change as needed to match your environment
cd "${TRTLLM_DIR}"
EXTRA_LLM_API_FILE=${TRTLLM_DIR}/examples/configs/curated/qwen3.8-low-latency.yaml
```

| Checkpoint | Profile | Configuration file |
|---|---|---|
| Qwen3.8-2.4T-A95B MoE FP8 | Low latency | [`qwen3.8-low-latency.yaml`](../../../examples/configs/curated/qwen3.8-low-latency.yaml) |
| Qwen3.8-2.4T-A95B MoE FP8 | Low latency with MTP3 | [`qwen3.8-low-latency-mtp3.yaml`](../../../examples/configs/curated/qwen3.8-low-latency-mtp3.yaml) |
| Qwen3.8-2.4T-A95B MoE FP8 | High throughput | [`qwen3.8-high-throughput.yaml`](../../../examples/configs/curated/qwen3.8-high-throughput.yaml) |
| Qwen3.8-2.4T-A95B MoE FP8 | High throughput with MTP3 | [`qwen3.8-high-throughput-mtp3.yaml`](../../../examples/configs/curated/qwen3.8-high-throughput-mtp3.yaml) |
| Qwen3.5-397B-A17B MoE NVFP4 | General deployment | [`qwen3.5.yaml`](../../../examples/configs/curated/qwen3.5.yaml) |

The MTP draft length is configurable through `speculative_config.max_draft_len`; the MTP3 profiles set it to 3. The high-throughput profiles use the following static expert placements:

| Profile | Static EPLB configuration |
|---|---|
| High throughput without MTP | [`qwen3.8-ep32-static544.yaml`](../../../examples/configs/curated/eplb/qwen3.8-ep32-static544.yaml) |
| High throughput with MTP3 | [`qwen3.8-ep32-static544-mtp3.yaml`](../../../examples/configs/curated/eplb/qwen3.8-ep32-static544-mtp3.yaml) |

Each file assigns expert IDs to 544 global slots for every MoE layer. Repeated expert IDs create replicas of heavily routed experts. `layer_updates_per_iter: 0` keeps the assignment static. The no-MTP map covers the 92 target-model layers, while the MTP3 map also includes the MTP layer. These placements are specific to Qwen3.8-2.4T-A95B, EP32, the selected MTP mode, and the benchmark traffic distribution. Generate a new placement when any of those inputs changes. See the [EPLB example](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/wide_ep/ep_load_balancer) for collecting routing statistics and generating a placement file.

### Launch the TensorRT LLM Server

Below is an example command to launch the TensorRT LLM server from within the container with the checkpoint that matches the configuration file you selected above.

```shell
trtllm-serve <model_path_or_hf_id> --host 0.0.0.0 --port 8000 --reasoning_parser qwen3_5 --tool_parser qwen3 --config ${EXTRA_LLM_API_FILE}
```

Qwen3.8 and Qwen3.5 both use the `qwen3_5` reasoning parser (their chat template pre-injects a `<think>` block, so reasoning starts at the beginning of the response). The `qwen3` tool parser handles the Qwen3 function-call format.

After the server is set up, the client can now send prompt requests to the server and receive results.

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
    "model": "<model_path_or_hf_id>",
    "messages": [
        {
            "role": "user",
            "content": "Where is New York?"
        }
    ],
    "max_tokens": 1024,
    "top_p": 1.0
}' -w "\n"
```

## Benchmarking Performance

To benchmark the performance of your TensorRT LLM server you can leverage the built-in `benchmark_serving.py` script. To do this, first create a wrapper `bench.sh` script.

```shell
cat <<'EOF' > bench.sh
#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="<model_path_or_hf_id>"

concurrency_list="1 2 4 8 16 32 64 128 256"
multi_round=5
isl=8192
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

To achieve max throughput, with attention DP on, one needs to sweep up to `concurrency = max_batch_size * num_gpus`.

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

## Key Configuration Options

These options control TensorRT LLM behavior and are set in the YAML file passed to `trtllm-serve` with the `--config` argument.

### `tensor_parallel_size`

Sets the tensor-parallel size. This should typically match the number of GPUs used by one model instance.

### `moe_expert_parallel_size`

Sets the expert-parallel size for MoE layers. It can differ from the attention tensor-parallel behavior when attention data parallelism is enabled.

### `enable_attention_dp`

Enables data parallelism for attention and linear-attention layers while keeping the MoE layers expert-parallel. This is primarily a throughput-oriented configuration.

### `kv_cache_config.free_gpu_memory_fraction`

Specifies the fraction of free GPU memory reserved for the KV cache after the model is loaded. Reduce this value if initialization reports an out-of-memory error.

### `kv_cache_config.use_kv_cache_manager_v2`

Selects KV cache manager V2 when set to `true`. Preserve the value in the curated profile unless the alternative cache-management path has been qualified with the selected topology and MTP mode.

### `kv_cache_config.avg_seq_len`

Helps KV cache manager V2 divide memory between attention KV blocks and recurrent GDN state. Set it to the expected average total sequence length for throughput deployments.

### `kv_cache_config.mamba_ssm_cache_dtype`

Selects the GDN recurrent-state dtype independently of the attention KV-cache dtype. Supported values are `auto`, `float16`, `bfloat16`, and `float32`.

### `max_batch_size`

Sets the maximum number of requests that can be grouped into one scheduled batch. The achievable batch size also depends on the total input and output sequence lengths.

### `max_num_tokens`

Sets the maximum total number of tokens across all requests in one scheduled batch.

### `max_seq_len`

Sets the maximum sequence length for one request, including input and generated tokens.

### `trust_remote_code`

Allows Hugging Face to load custom model and tokenizer code from the model repository. Enable it only for trusted model sources.

### `cuda_graph_config`

Controls CUDA graph capture and padding:

* `enable_padding`: Pads input batches to a captured CUDA graph batch size.
* `max_batch_size`: Sets the largest batch size for which CUDA graphs are captured.

### `moe_config`

Controls MoE execution:

* `backend`: Selects the MoE backend.
* `max_num_tokens`: Limits the number of tokens processed by one fused MoE invocation before chunking.
* `load_balancer`: Accepts a static EPLB configuration file or an inline Online EPLB configuration.

### `speculative_config.max_draft_len`

Sets the MTP draft length. Tune it for the deployment objective; the MTP3 profiles set it to 3.

See the [`TorchLlmArgs` API reference](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.TorchLlmArgs) for the complete configuration schema, [KV cache documentation](../features/kvcache.md) for hybrid-state cache settings, and [speculative decoding documentation](../features/speculative-decoding.md) for MTP details.

## Troubleshooting Tips

* If you encounter CUDA out-of-memory errors, try reducing `max_batch_size`, `max_num_tokens`, or `kv_cache_config.free_gpu_memory_fraction`. If the error occurs during CUDA graph capture, also reduce `cuda_graph_config.max_batch_size`.
* Ensure your model checkpoints are compatible with the expected format. If a Qwen3.8 MoE checkpoint is routed to an unsupported architecture, verify that its `config.json` advertises `Qwen3_5MoeForCausalLM` and `qwen3_5_moe_text`.
* If MTP output is empty or incorrect, confirm that the checkpoint contains its MTP weights and that the selected `max_draft_len` is configured consistently on every rank.
* For performance issues, check GPU utilization with `nvidia-smi` while the server is running and complete initialization and warmup before measuring performance.
* If static EPLB regresses throughput, verify the placement-file hash and `num_slots`, and recollect statistics using representative traffic. Do not reuse a map from another checkpoint or MTP mode.
* If online EPLB is enabled, wait for the configured observation and migration period before measuring steady state, and retain logs that show completed layer updates.
* If the container fails to start, verify that the NVIDIA Container Toolkit is properly installed.
* For connection issues, make sure the server port (`8000` in this guide) is not being used by another application.
* Reasoning is controlled with `--reasoning_parser qwen3_5`. To toggle thinking per request, pass `enable_thinking` through `chat_template_kwargs` in the request body, for example `{"chat_template_kwargs": {"enable_thinking": true}}` (set it to `false` to disable reasoning).
