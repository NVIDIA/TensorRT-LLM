# Deployment Guide for Qwen3.8-Flash-Next on TensorRT LLM - Blackwell Hardware

## Introduction

This guide describes how to serve Qwen3.8-Flash-Next with the TensorRT LLM PyTorch backend. It covers the BF16, block-scaled FP8, and NVFP4 checkpoints, aggregated and disaggregated serving, MTP speculative decoding, and image-language input.

The Hugging Face checkpoint is a composite vision-language model registered as `Qwen4ExpForConditionalGeneration`. Language-only serving flattens it to the `qwen4_exp_text` decoder, which TensorRT LLM registers as `Qwen4ExpForCausalLM`. The same decoder serves both modes; the vision tower is loaded only when the server accepts image input.

The decoder is a hybrid model rather than a plain MoE transformer. It interleaves Gated DeltaNet (GDN) linear-attention layers with QSA sparse full-attention layers, routes tokens through 512 experts with top-10 routing plus a shared expert, mixes several residual streams with Hyper-Connections, and adds a PLE n-gram side path. One recurrent MTP module ships with the checkpoint.

## Support Status

The BF16, block-FP8, and NVFP4 checkpoints support the same feature set:

| Capability | Status |
|---|---|
| Aggregated and disaggregated serving | Supported |
| MTP speculative decoding with draft length 3 | Supported |
| Image-language input | Supported |
| Tensor, expert, and attention data parallelism (TP, EP, ADP) | Supported |
| Chunked prefill | Supported |
| FP8 KV cache | Supported |
| Prefix caching (block reuse) | Supported |
| KV cache manager V2 | Required, and selected by default |

The KV-cache dtype and the GDN recurrent-state dtype are selected independently of the model weight precision, through `kv_cache_config.dtype` and `kv_cache_config.mamba_ssm_cache_dtype`.

## Prerequisites

* GPU: NVIDIA Blackwell architecture. The BF16 checkpoint also runs on Hopper.
* OS: Linux
* Drivers: CUDA Driver 575 or later
* Docker with NVIDIA Container Toolkit installed
* Python3 and python3-pip (optional, for accuracy evaluation only)

## Models

* [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (base, BF16)
* [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) (block-scaled FP8 routed experts)
* [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) (mixed-precision NVFP4)

The NVFP4 checkpoint is a mixed-precision export: its routed experts are NVFP4, while the PLE n-gram table and the MTP experts are FP8. TensorRT LLM therefore reports `MIXED_PRECISION` rather than `NVFP4` for it. The per-layer quantization metadata is part of the checkpoint contract, so do not override it with a single global quantization algorithm.

The FP8 checkpoint is pre-quantized. It is not produced by enabling an FP8 option while loading the BF16 checkpoint. Its routed-expert projections carry FP8 weights with 128x128 block scales, while attention, GDN projections, the shared expert, routers, embeddings, and the LM head keep their checkpoint dtypes.

## GPU Requirements

The following table lists the minimum resources for each checkpoint. All three configurations enable PLE host offload, which is what makes the host-memory floor part of the requirement.

| Checkpoint | Platform | Minimum GPUs | GPU memory per GPU | Host memory |
|---|---|---:|---:|---:|
| BF16 | Hopper or Blackwell | 2 | 142 GB | 128 GiB |
| Block-FP8 | Blackwell | 1 | 145 GB | 96 GiB |
| NVFP4 | Blackwell | 1 | 100 GB | 96 GiB |

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

### PLE Table Host Offload

The PLE n-gram table is a large weight that is read once per token. Offloading it to pinned host memory frees device memory for the KV cache at the cost of host memory and a host-to-device read. Enable it in the environment of every worker before starting the server:

```shell
export TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD=1
```

This is an environment-only option; there is no YAML field for it. It moves the table only. The per-request PLE short-convolution state and n-gram context stay in the cache manager on the device, and remain part of every disaggregated handoff.

### Recommended Performance Settings

We maintain YAML configuration files with recommended performance settings in the [`examples/configs`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/configs) directory. These config files are present in the TensorRT LLM container at the path `/app/tensorrt_llm/examples/configs`. You can use these out-of-the-box, or adjust them to your specific use case.

Set the TensorRT LLM directory and select one of the configuration files listed below:

```shell
TRTLLM_DIR=/app/tensorrt_llm # change as needed to match your environment
EXTRA_LLM_API_FILE=${TRTLLM_DIR}/examples/configs/curated/qwen3.8-flash-next.yaml
```

| Deployment | Configuration file |
|---|---|
| Aggregated serving | [`qwen3.8-flash-next.yaml`](../../../examples/configs/curated/qwen3.8-flash-next.yaml) |
| Disaggregated context worker | [`qwen3.8-flash-next-disagg-ctx.yaml`](../../../examples/configs/curated/qwen3.8-flash-next-disagg-ctx.yaml) |
| Disaggregated generation worker | [`qwen3.8-flash-next-disagg-gen.yaml`](../../../examples/configs/curated/qwen3.8-flash-next-disagg-gen.yaml) |

`Qwen4ExpForCausalLM` already selects QSA sparse attention and KV cache manager V2, and already disables block reuse, so these files set only what differs from those defaults. They target the block-FP8 and NVFP4 checkpoints on one GPU; size `max_batch_size`, `max_num_tokens`, and `max_seq_len` for your workload before treating them as a performance baseline.

To adapt them:

* **BF16**: set `tensor_parallel_size: 2` and `moe_config.backend: CUTLASS`.
* **Four GPUs**: set `tensor_parallel_size: 4` and `moe_expert_parallel_size: 4`, and leave `moe_tensor_parallel_size` at `1` so each expert keeps its complete intermediate dimension. Add `enable_attention_dp: true` with `enable_lm_head_tp_in_adp: true` for the throughput-oriented layout.
* **No speculative decoding**: remove the `speculative_config` block.

#### Non-greedy MTP

The `speculative_config` block in those files is correct for greedy decoding. For requests with a nonzero temperature or top-p, add rejection sampling so the sampled distribution stays correct:

```yaml
speculative_config:
  decoding_type: MTP
  max_draft_len: 3
  use_rejection_sampling: true
  advanced_sampling_mode: full
```

`advanced_sampling_mode: full` keeps per-request top-k and top-p filtering. The runtime detects non-greedy requests automatically. Acceptance depends on the prompt distribution, sampling parameters, checkpoint, and batch shape; a rate measured on one workload is not a model constant.

#### Prefix caching

Attention KV blocks alone cannot restore GDN and PLE state, so the model disables block reuse by default. Enable it together with a snapshot policy:

```yaml
kv_cache_config:
  enable_block_reuse: true
  enable_partial_reuse: true
  copy_on_partial_reuse: true
  mamba_state_config:
    periodic_snapshot_interval: 256
```

Block reuse stays on only if `mamba_state_config` sets at least one snapshot placement: `periodic_snapshot_interval`, `additional_snapshot_offsets_from_start`, or `additional_snapshot_offsets_from_end`. `enable_branch_snapshot` refines where snapshots land but does not by itself keep reuse enabled. The same settings apply in disaggregated serving; configure them on both workers. A snapshot restores the PLE short-convolution state and n-gram context along with the GDN state, because they share one recurrent page. Check the effective LLM arguments and the per-request `num_reused_blocks` metric rather than assuming the submitted setting took effect.

### Launch the TensorRT LLM Server

```shell
trtllm-serve <model_path_or_hf_id> \
  --host 0.0.0.0 --port 8000 \
  --reasoning_parser qwen3_5 \
  --tool_parser qwen3 \
  --config ${EXTRA_LLM_API_FILE}
```

The chat template pre-injects a `<think>` block, so reasoning starts at the beginning of the response and the `qwen3_5` reasoning parser applies. This architecture is not in the parser auto-detection table, so pass both parsers explicitly. Thinking is controlled per request through `chat_template_kwargs`, for example `{"chat_template_kwargs": {"enable_thinking": false}}` to answer directly, or `{"chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "xhigh"}}` for a longer trace.

### Disaggregated Serving

Disaggregated serving separates prefill (context) and decode (generation) onto different workers. For this model the transferred request state is larger than an attention KV cache: it also carries the QSA index state, the GDN recurrent state, and the PLE n-gram context and short-convolution state. Only the Python NIXL transceiver can move that state, and this model does not declare a preferred runtime, so both workers must set `cache_transceiver_config.transceiver_runtime` to `PYTHON` explicitly. Leaving it at its `auto` default selects the C++ transceiver, and KV cache manager V2 then rejects the configuration rather than serving wrong results.

Use [`qwen3.8-flash-next-disagg-ctx.yaml`](../../../examples/configs/curated/qwen3.8-flash-next-disagg-ctx.yaml) and [`qwen3.8-flash-next-disagg-gen.yaml`](../../../examples/configs/curated/qwen3.8-flash-next-disagg-gen.yaml) for the two workers. Both must load the same checkpoint revision, use the same parallel layout, and use the same PLE host-offload setting. Add `speculative_config` to the generation worker only. For long generations, raise `cache_transceiver_config.kv_transfer_timeout_ms` above the expected request duration.

The orchestrator is launched with a disaggregated config that lists the worker URLs:

```yaml
hostname: localhost
port: 8000
backend: pytorch
context_servers:
  num_instances: 1
  urls:
    - "localhost:8001"
generation_servers:
  num_instances: 1
  urls:
    - "localhost:8002"
```

```bash
# Start each worker first, and wait for both /health endpoints to return 200.
trtllm-serve <model_path_or_hf_id> --host 0.0.0.0 --port 8001 --config ${TRTLLM_DIR}/examples/configs/curated/qwen3.8-flash-next-disagg-ctx.yaml
trtllm-serve <model_path_or_hf_id> --host 0.0.0.0 --port 8002 --config ${TRTLLM_DIR}/examples/configs/curated/qwen3.8-flash-next-disagg-gen.yaml

trtllm-serve disaggregated -c disagg_config.yaml
```

Clients then send OpenAI-compatible requests to the orchestrator on port `8000`. To restrict the worker endpoints to requests coming from the orchestrator, set the same `internal_request_auth_key` value in both worker configs and in the orchestrator config. For the full walkthrough, per-worker GPU placement, and multi-node or SLURM launch, see the [Disaggregated Serving guide](../features/disagg-serving.md).

### Image-Language Serving

The same configurations serve image input; the vision tower is loaded unless `disable_mm_encoder: true` is set. For image workloads, size the encoder separately from the decoder:

```yaml
encoder_max_batch_size: 8
encoder_max_num_tokens: 65536
```

Image input is supported in aggregated serving with a local encoder. A separate encoder-to-prefill handoff is not part of this guide.

### Key Configuration Options

These options control TensorRT LLM behavior and are set in the YAML file passed to `trtllm-serve` with the `--config` argument.

#### `tensor_parallel_size`

Sets the tensor-parallel size for attention, GDN, and PLE layers. This should typically match the number of GPUs used by one model instance.

#### `moe_tensor_parallel_size`

Sets the tensor-parallel size for routed experts, independently of `tensor_parallel_size`. Keep it at `1` and use expert parallelism instead: the routed-expert intermediate dimension is five 128-element blocks, so a pure MoE tensor-parallel split of 2 or 4 would cut FP8 scale blocks.

#### `moe_expert_parallel_size`

Sets the expert-parallel size for MoE layers. Use it together with `moe_tensor_parallel_size: 1` for multi-GPU deployments.

#### `enable_attention_dp`

Runs attention and linear-attention layers data-parallel while the MoE layers stay expert-parallel. This is the throughput-oriented layout. Pair it with `enable_lm_head_tp_in_adp: true` to keep the LM head tensor-parallel.

#### `kv_cache_config.free_gpu_memory_fraction`

Fraction of free GPU memory reserved for the KV cache and recurrent state after the model is loaded. Reduce it if initialization reports an out-of-memory error.

#### `kv_cache_config.mamba_ssm_cache_dtype`

Selects the GDN recurrent-state dtype independently of the attention KV-cache dtype. Supported values are `auto`, `float16`, `bfloat16`, and `float32`.

#### `kv_cache_config.mamba_state_config.periodic_snapshot_interval`

Number of tokens between recurrent-state snapshots in the prefix cache. Snapshots at a fixed interval suit a workload whose shared prefixes vary in length; see [Prefix caching](#prefix-caching).

#### `kv_cache_config.mamba_state_config.additional_snapshot_offsets_from_start`, `..._from_end`

Snapshot the recurrent state at fixed token offsets measured from the start or the end of each prompt, instead of, or in addition to, a periodic interval. An offset of `0` from the end selects the prompt end, which suits multi-turn workloads that reuse a whole previous turn. Offsets that fall outside a prompt are ignored.

#### `max_batch_size`, `max_num_tokens`, `max_seq_len`

Set the maximum number of requests per scheduled batch, the maximum total tokens per scheduled batch, and the maximum length of a single request including generated tokens.

#### `cuda_graph_config`

Controls CUDA graph capture and padding. The model defaults to eager execution, so set this explicitly:

* `enable_padding`: Pads input batches to a captured CUDA graph batch size.
* `max_batch_size`: Largest batch size for which graphs are captured. Set it to match `max_batch_size`.

#### `moe_config`

Controls MoE execution:

* `backend`: Selects the MoE backend. Use `CUTLASS` for BF16 and `TRTLLM` for the FP8 and NVFP4 checkpoints; `DEEPGEMM` is an alternative for block-scaled FP8 on SM100 or later.
* `max_num_tokens`: Limits the tokens processed by one fused MoE invocation before chunking.
* `disable_finalize_fusion`: Uses the separate deterministic finalization path on the CUTLASS BF16 backend. The fused path is numerically valid but not bitwise reproducible, so set this only when exact run-to-run replay is required.

#### `speculative_config`

Configures MTP speculative decoding:

* `decoding_type`: Set to `MTP`.
* `max_draft_len`: Draft length; the checkpoint's recurrent MTP module supports up to `3`.
* `use_rejection_sampling` and `advanced_sampling_mode`: See [Non-greedy MTP](#non-greedy-mtp).

#### `cache_transceiver_config`

Configures the disaggregated state transfer. This model requires `backend: NIXL` with `transceiver_runtime: PYTHON`; see [Disaggregated Serving](#disaggregated-serving).

#### `trust_remote_code`

Allows Hugging Face to load custom model and tokenizer code from the model repository. Enable it only for trusted model sources.

See the [`TorchLlmArgs` API reference](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.TorchLlmArgs) for the complete configuration schema, [KV cache documentation](../features/kvcache.md) for hybrid-state cache settings, and [speculative decoding documentation](../features/speculative-decoding.md) for MTP details.

## Testing API Endpoint

### Health Check

Start a new terminal on the host to test the TensorRT LLM server you just launched.

```shell
curl -s -o /dev/null -w "Status: %{http_code}\n" "http://localhost:8000/health"
```

When the `Status: 200` code is returned, the server is ready for queries. The very first query may take longer due to initialization and compilation.

### Basic Test

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
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

### Image Request

An OpenAI-compatible image request uses an `image_url` content part followed by the text instruction. Add more `image_url` parts for a multi-image request and preserve their order:

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "<model_path_or_hf_id>",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
            {"type": "text", "text": "Describe the image briefly."}
        ]
    }],
    "max_tokens": 128
}' -w "\n"
```

The image URL must be reachable from the serving frontend; a supported data URL may be used instead.

## Running Evaluations to Verify Accuracy (Optional)

`trtllm-eval` runs the same tasks as the in-tree accuracy suite:

```shell
export TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD=1
trtllm-eval --model <model_path_or_hf_id> \
  --config config.yaml \
  gsm8k
```

The reference scores committed with the model are:

| Checkpoint | GSM8K | MMLU | MMMU |
|---|--:|--:|--:|
| BF16 | 95.72 | 87.82 | — |
| Block-FP8 with MTP3 | 95.11 | 87.91 | — |
| NVFP4 with MTP3 | 95.57 | 87.57 | 59.22 |

The GSM8K and MMLU runs use the chat template with `enable_thinking` on and off respectively; see `TestQwen3_8_Flash_Next` in `tests/integration/defs/accuracy/test_llm_api_pytorch.py` for the exact evaluator settings. Small differences of roughly half a point are expected across checkpoint and dependency revisions.

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
result_dir=/tmp/qwen3.8_flash_next_output

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

Complete initialization, CUDA graph capture, and warmup before measuring; the first requests after startup are not representative of steady state.

## Troubleshooting Tips

* If you encounter CUDA out-of-memory errors, try reducing `max_batch_size`, `max_num_tokens`, or `kv_cache_config.free_gpu_memory_fraction`. If the error occurs during CUDA graph capture, also reduce `cuda_graph_config.max_batch_size`. Enabling `TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD=1` frees additional device memory at the cost of host memory.
* If weight loading fails on the FP8 or NVFP4 checkpoint after a multi-GPU change, check that `moe_tensor_parallel_size` is `1`. A pure MoE tensor-parallel split of 2 or 4 cuts the routed experts' quantization blocks.
* If block reuse appears to have no effect, check the effective LLM arguments for `enable_block_reuse`. The runtime turns it off when `kv_cache_config.mamba_state_config` configures no snapshot placement, because the recurrent state cannot be restored from attention blocks alone.
* If a disaggregated worker fails to start with a KV cache manager V2 error, check that both workers set `cache_transceiver_config.backend: NIXL` and `transceiver_runtime: PYTHON`.
* If MTP output is empty or incorrect, confirm that the checkpoint contains its MTP weights and that `max_draft_len` is configured identically on every rank.
* If reasoning content is not separated from the answer, confirm that the server was started with `--reasoning_parser qwen3_5`; this architecture is not auto-detected.
* If the container fails to start, verify that the NVIDIA Container Toolkit is properly installed.
* For connection issues, make sure the server port (`8000` in this guide) is not being used by another application.
