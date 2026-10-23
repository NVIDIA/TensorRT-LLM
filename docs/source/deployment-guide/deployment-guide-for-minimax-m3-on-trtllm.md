# Deployment Guide for MiniMax-M3 on TensorRT LLM

## Introduction

This deployment guide provides step-by-step instructions for running the MiniMax-M3 model using TensorRT LLM. It covers the complete setup required; from accessing model weights and preparing the software environment to configuring TensorRT LLM parameters, launching the server, and validating inference output.

MiniMax-M3 is a Mixture-of-Experts (MoE) model that uses MiniMax block-sparse attention. The first few layers use dense attention with a dense MLP, while the remaining layers combine a sparse attention path (an index-K block selector followed by sparse grouped-query attention) with MoE (top-4 of 128 routed experts plus one shared expert). In TensorRT LLM it is served through the `MiniMaxM3SparseForConditionalGeneration` architecture (text, image, and video) and the text-only `MiniMaxM3SparseForCausalLM` architecture.

TensorRT LLM supports three checkpoint weight formats for MiniMax-M3:

* **BF16** — the official upstream checkpoint from MiniMaxAI.
* **MXFP8** — weights are stored in MXFP8. The curated configuration uses BF16 KV cache and is the default for this guide.
* **Mixed MXFP8/NVFP4** — MXFP8 base layers with NVFP4 routed experts. FP8 or NVFP4 KV cache is selected separately at runtime.

The MSA backend supports KV cache block reuse, linear Eagle3 with one shared GQA draft layer, and a standalone DSpark drafter; see [Feature Support Notes](#feature-support-notes) for the constraints and [Serving MiniMax-M3.1-Flash](#serving-minimax-m31-flash) for the MiniMax-M3.1-Flash checkpoint.

This guide deploys MiniMax-M3 on **8x NVIDIA GB200 GPUs across 2 nodes** (4 GPUs per node) using Slurm and the `trtllm-llmapi-launch` multi-node launcher, with the MoE experts distributed via expert parallelism. The attention layers can run with either Tensor-Expert Parallelism (TEP) or Data-Expert Parallelism (DEP); see [Recommended Performance Settings](#recommended-performance-settings).

The guide is intended for developers and practitioners seeking high-throughput or low-latency inference using NVIDIA's accelerated stack.

## Prerequisites

* GPU: 8x NVIDIA GB200 GPUs across 2 nodes (4 GPUs per node). Tensor/expert parallelism of 8 (`tensor_parallel_size: 8`, `moe_expert_parallel_size: 8` in the curated YAML) spans all 8 GPUs. Plan for the corresponding memory footprint — MXFP8 weights occupy roughly half of BF16.
* Multi-node launcher: Slurm with the pyxis/enroot container plugin (or an equivalent MPI launcher) to start one rank per GPU across both nodes.
* High-speed inter-node interconnect (e.g., InfiniBand) for tensor/expert-parallel traffic.
* Shared filesystem visible to both nodes for the model weights and the configuration file.
* OS: Linux
* Drivers: CUDA Driver 575 or later
* Container runtime with NVIDIA GPU support on each node

## Models

MiniMax-M3 checkpoints use the `MiniMaxM3SparseForConditionalGeneration` / `MiniMaxM3SparseForCausalLM` architectures. Checkpoint weight precision and runtime KV-cache precision are separate choices.

### MXFP8 (recommended for throughput)

* [MiniMaxAI/MiniMax-M3-MXFP8](https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8) — MiniMaxAI-published MXFP8-quantized checkpoint. Weights are stored in MXFP8 (block size 1×32). The curated configuration uses BF16 KV cache; quantized KV cache is selected separately.

```bash
git lfs install
git clone https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8 /models/MiniMax-M3-MXFP8
```

### BF16 (upstream)

* [MiniMaxAI/MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) — Official BF16 checkpoint from MiniMaxAI.

```bash
git lfs install
git clone https://huggingface.co/MiniMaxAI/MiniMax-M3 /models/MiniMax-M3
```

### Mixed MXFP8/NVFP4

The `MiniMax-M3-NVFP4` checkpoint uses MXFP8 base layers and NVFP4 routed experts
(`MIXED_PRECISION`). Its name describes weight precision; set
`kv_cache_config.dtype` explicitly to select FP8 or NVFP4 KV cache.

The compressed-tensors NVFP4 MoE packaging of the MiniMax M3 architecture is
also loaded as mixed MXFP8/NVFP4. Such a checkpoint declares
`quant_method: mxfp8` together with `moe_quant_algo: NVFP4`,
`moe_quant_format: nvfp4-pack-quantized` and `group_size: 16`, and stores each
routed expert projection as `weight_packed`, `weight_scale`,
`weight_global_scale` and `input_global_scale`. TensorRT LLM keeps the packed
FP4 bytes and E4M3 block scales as stored and converts the global scales at
load time. A global scale that is zero, negative or not finite fails the load
with an error naming the checkpoint tensor.
Both `moe_config.backend: TRTLLM` (TRTLLM-Gen, the backend the serving configs
use on SM100 and SM103) and `moe_config.backend: CUTLASS` (the portable
grouped-GEMM path, which `AUTO` also resolves to) load this packaging and apply
the same static recipe: the stored `input_global_scale` becomes the layer's
per-tensor FC1 input scale (1.0 in the shipped checkpoints, so each 16-element
block of the FC1 input is quantized with only its own E4M3 block scale),
`weight_global_scale` is inverted into the per-expert FC1 and FC2 dequantization
alphas, and the gated intermediate is requantized to NVFP4 with the static FC2
input scale before FC2, with the routing weights applied in the fused finalize.
Neither backend quantizes the FC1 input per token; they differ only in where the
FC1 dequantization and FC2 requantization happen (TRTLLM-Gen folds them into the
GEMM1 epilogue, CUTLASS applies them through its quantization scales), so the
two are numerically equivalent up to kernel rounding.

Use the checkpoint's chat template (`chat_template.jinja`) when launching the
server (see [Launch the TensorRT LLM Server](#launch-the-tensorrt-llm-server)).
Evaluate a checkpoint in this packaging through the chat template as well, for
example `trtllm-eval ... gsm8k --apply_chat_template`; the raw eight-shot prompts
of the `gsm8k` default bypass the template the model was trained with.

### MiniMax-M3.1-Flash

* [MiniMaxAI/MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) — MiniMaxAI-published checkpoint in this mixed MXFP8/NVFP4 packaging, with its DSpark drafter bundled in the `dspark/` subdirectory. See [Serving MiniMax-M3.1-Flash](#serving-minimax-m31-flash).

```bash
git lfs install
git clone https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash /models/MiniMax-M3.1-Flash
```

## Feature Support Notes

* **Block-sparse attention is required.** MiniMax-M3 runs on the block-sparse attention backend, which must be selected via `sparse_attention_config.algorithm: minimax_m3` in the YAML configuration. There is no dense fallback for the sparse layers.
* **Weight and KV precision.** BF16, MXFP8, and mixed MXFP8/NVFP4 checkpoints are supported. On SM100/SM103/SM107, the MSA backend supports FP8 and NVFP4 KV cache. NVFP4 storage is used in sparse target layers; dense target layers and the shared Eagle3 draft layer use FP8. The curated BF16-KV YAML remains a separate starting configuration.
* **KV cache reuse.** KV cache block reuse is supported on the sparse-attention path for text and multimodal requests; enable it with `kv_cache_config.enable_block_reuse: true`. The curated throughput YAML leaves it disabled. See [KV cache reuse](#kv-cache-reuse).
* **Speculative decoding.** The MSA path supports one shared GQA draft layer with linear Eagle3 drafting and the standalone DSpark drafter bundled with MiniMax-M3.1-Flash (see [Speculative decoding with the DSpark drafter](#speculative-decoding-with-the-dspark-drafter)). MTP and tree-drafting configurations are unsupported. Keep NVFP4 KV and piecewise CUDA graph qualification separate: the existing PCG accuracy cases use FP8 KV.
* **`max_seq_len` must be capped for CUDA graphs.** The dense GQA expansion in the first few attention layers and the per-Q FP32 expansion in the sparse decode kernel allocate temporary tensors whose size grows linearly with the warmup decode's `max_k`. If `max_seq_len` is left at the checkpoint default, that `max_k` follows `max_position_embeddings` (1M for MXFP8, 512K for BF16) and the resulting gigabyte-scale single-allocation request exceeds the caching allocator's CUDA-graph-safe path, so capture fails with `cudaErrorStreamCaptureUnsupported` / OOM. The curated YAML therefore sets `max_seq_len` to a small value just above ISL+OSL (`2068` for the 1k/1k benchmark) so CUDA graphs can capture cleanly. Raise it for longer-context workloads, but expect a corresponding cut in `max_batch_size`.
* **Parallelism.** MoE experts run with expert parallelism. The attention layers support both Tensor-Expert Parallelism (TEP) and Data-Expert Parallelism (DEP, via `enable_attention_dp: true`). The overlap scheduler is enabled by default.
* **Multimodal.** `MiniMaxM3SparseForConditionalGeneration` supports text, image, and video inputs. The text decoder is also usable standalone (text-only) via the `MiniMaxM3SparseForCausalLM` architecture. Text requests need only the tokenizer. The checkpoint's Hugging Face processor is loaded on the first image or video request. If `AutoProcessor` cannot build a processor with an image processor for the checkpoint directory (for example, it returns only a tokenizer because no loadable processor code is shipped), that request fails with a `ValueError` saying that MiniMax-M3 image and video inputs need an HF processor with an image processor. Text requests keep working. To serve images, point the model at a directory whose processor loads with `AutoProcessor`.

## Serving MiniMax-M3.1-Flash

MiniMax-M3.1-Flash is served with the same `MiniMaxM3SparseForConditionalGeneration` architecture, the same CLI flags, and the same YAML options as MiniMax-M3; everything above applies to it, except that all 60 of its attention layers are sparse (its `sparse_attention_freq` has no dense entries), so there are no dense target layers. The examples use a local copy of the checkpoint at `/models/MiniMax-M3.1-Flash`.

### Checkpoint

The checkpoint ships MXFP8 base layers with NVFP4 routed experts in the compressed-tensors packaging described in [Mixed MXFP8/NVFP4](#mixed-mxfp8nvfp4); the target loads natively in that mixed precision. `moe_config.backend: TRTLLM` is the serving choice on SM100/SM103/SM107 and `CUTLASS` the portable reference. The DSpark drafter is bundled in the `dspark/` subdirectory of the checkpoint, is loaded only when `speculative_config` selects it, and runs in BF16 (its MXFP8 linears are dequantized at load). The checkpoint ships its own Hugging Face processor, which the server loads on the first image or video request (see **Multimodal** under [Feature Support Notes](#feature-support-notes)).

### KV cache reuse

KV cache block reuse is supported on the sparse-attention path for text and multimodal requests, with or without speculative decoding. Enable it with `kv_cache_config.enable_block_reuse: true`; the MSA backend matches reused prefixes at token granularity on its 128-token pages, and a reuse hit or a chunk boundary inside an image or video placeholder span is handled. With the DSpark drafter enabled, the target's cache is reused as usual while the drafter keeps its own per-request context, so after a partial-prefix hit the drafter can run without context for the rest of its current 64-token page; this lowers the acceptance length for a few steps and does not change outputs.

### NVFP4 KV cache on SM107

The [NVFP4 KV-cache configuration](#nvfp4-kv-cache-configuration) above now also runs on SM107: the MSA backend and the Triton NVFP4 sparse decode kernels accept SM100, SM103, and SM107, and the generic SM107 fallback from NVFP4 to FP8 KV cache does not apply to MiniMax-M3. Dense target layers and the shared Eagle3 draft layer keep FP8 caches as before.

### Speculative decoding with the DSpark drafter

DSpark is a target-dependent block drafter: it reads hidden states from five target layers and proposes a block of up to 7 draft tokens per step, which the target verifies in one forward. Select it with `decoding_type: DSpark` and point `speculative_model` at the bundled drafter. The target attention backend must be `TRTLLM` (`FLASHINFER` is rejected for one-engine speculation), and the drafter's KV cache is a separate pool sized from `max_batch_size` and the drafter's sliding attention window rather than from `max_seq_len`. The following YAML runs on 4 SM100/SM103/SM107 GPUs with NVFP4 KV cache and block reuse enabled at the checkpoint's 1M context; set `max_seq_len` to the longest prompt plus output you serve:

```yaml
tensor_parallel_size: 4
moe_expert_parallel_size: 4
trust_remote_code: true
attn_backend: TRTLLM
moe_config:
  backend: TRTLLM
max_batch_size: 8
max_num_tokens: 16384
max_seq_len: 1048576
enable_chunked_prefill: true
cuda_graph_config:
  enable_padding: true
  max_batch_size: 8
kv_cache_config:
  dtype: nvfp4
  enable_block_reuse: true
  free_gpu_memory_fraction: 0.9
sparse_attention_config:
  algorithm: minimax_m3
  implementation: msa
  indexer_kv_dtype: fp8
  fuse_qkv_index_projection: true
speculative_config:
  decoding_type: DSpark
  max_draft_len: 7
  speculative_model: /models/MiniMax-M3.1-Flash/dspark
```

`max_draft_len` is the number of draft tokens per step, up to the drafter's native block of 7; smaller values trade acceptance length for cheaper drafting at high concurrency. `block_size` defaults to the drafter's value and must equal `max_draft_len`, so to draft 3 tokens set both `max_draft_len: 3` and `block_size: 3`. Launch the server with the same command as in [Launch the TensorRT LLM Server](#launch-the-tensorrt-llm-server), pointing `$MODEL` at `/models/MiniMax-M3.1-Flash` and `--extra_llm_api_options` at this file, and adding `--served_model_name MiniMax-M3.1-Flash` so requests can name the model as in the examples below; a single-node deployment uses `-N 1 --ntasks 4 --ntasks-per-node 4 --gres=gpu:4`. The startup log reports the drafter's cache as `Separate draft KV cache of the standalone drafter is funded for ...`.

In disaggregated serving the drafter's cache is not transferred with the prompt, so a generation worker starts drafting without prompt context and draft acceptance is reduced until its attention window fills; compare against generation workers without speculative decoding before enabling it there.

### Thinking mode and reasoning parsing

The MiniMax-M3.1-Flash chat template has no thinking-mode switch. It always renders the adaptive instructions (`Current thinking mode: adaptive`) and ends the prompt at the assistant header without prefilling `<mm:think>` or `</mm:think>`, so the model decides per turn whether to emit a reasoning block or a bare `</mm:think>` before its answer. A `thinking_mode` key in `chat_template_kwargs` is ignored by this template; the `enabled` / `disabled` / `adaptive` switch belongs to the upstream MiniMax-M3 checkpoints' template. The only key this template reads is `reasoning_effort` (alias `effort`), one of `low`, `medium`, `high`, `xhigh`, or `max`, rendered into an `<effort>` tag of the system prompt. MiniMaxAI recommends sampling with `temperature: 1.0` and `top_p: 0.95`. A plain request needs no template keys; this one also asks for high reasoning effort:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "MiniMax-M3.1-Flash",
      "messages": [{"role": "user", "content": "What is the capital of France?"}],
      "chat_template_kwargs": {"reasoning_effort": "high"},
      "temperature": 1.0,
      "top_p": 0.95,
      "max_tokens": 1024
  }'
```

With `--reasoning_parser minimax_m3` the server resolves the reasoning state from the rendered prompt: nothing is prefilled, so parsing starts outside reasoning and the model's own `<mm:think>...</mm:think>` block is returned in `reasoning_content` with the answer in `content`; the tags themselves are never returned, and a turn without a reasoning block has an empty `reasoning_content`. Reasoning cut off by `max_tokens` is returned as `reasoning_content` with empty `content`. Streaming responses split the deltas the same way (`reasoning_content` deltas, then `content` deltas, with no tag text leaking into `content`), so a client sees the same fields whether or not it streams. `trtllm-eval` applies the template with `--apply_chat_template` and needs no `--chat_template_kwargs`. The `/v1/completions` endpoint bypasses the chat template and returns the raw tags.

## Deployment Steps

### NVFP4 KV-cache configuration

For an SM100/SM103/SM107 deployment with a compatible checkpoint, use these overrides:

```yaml
kv_cache_config:
  dtype: nvfp4
sparse_attention_config:
  algorithm: minimax_m3
  implementation: msa
  indexer_kv_dtype: fp8
  fuse_qkv_index_projection: true
```

NVFP4 prefill uses MSA kernels, while decode uses TensorRT LLM's Triton sparse
kernels inside the MSA backend. The legacy `implementation: triton` backend
rejects NVFP4 KV and recommends MSA for its supported configurations too.
SM100, SM103, and SM107 support NVFP4; GB300-measured decode tuning applies
only to SM103. MSA requires 128-token sparse pages and the packaged `fmha_sm100`
dependency. The indexer stays FP8 in this configuration.

For mixed MXFP8/NVFP4 weights, `moe_config.backend: TRTLLM` is the serving
choice on SM100 and SM103 and `CUTLASS` the portable reference; both apply the
same static expert recipe. KV-cache quantization does not choose the MoE weight
backend.
Full weight reload refreshes the quantization and aligned prefill scale buffers
in place so CUDA-graph replay retains valid addresses.

MiniMax-M3 is deployed across 2 nodes (8x GB200 total) using Slurm with the pyxis/enroot container plugin. The model weights and the configuration file must live on a **shared filesystem** visible to both nodes.

### Container Image

The TensorRT LLM NVIDIA NGC image is used as the Slurm container:

```text
nvcr.io/nvidia/tensorrt-llm/release:x.y.z
```

Note:

* See <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags> for all available containers. Containers published in the main branch weekly have an `rcN` suffix, while the monthly release with QA tests has no `rcN` suffix. Use the `rc` release to get the latest model and feature support.
* If you want to use the latest main branch, you can build from source: [https://nvidia.github.io/TensorRT-LLM/latest/installation/build-from-source.html](https://nvidia.github.io/TensorRT-LLM/latest/installation/build-from-source.html)

### Recommended Performance Settings

**Treat these as a starting point and tune the parameters for your workload.**

We provide a curated YAML configuration in the TensorRT LLM repository that bundles the MiniMax-M3 sparse-attention backend, the KV-cache settings required by the sparse path, and a tuned set of throughput knobs:

```shell
TRTLLM_DIR=/app/tensorrt_llm # change as needed to match your environment
EXTRA_LLM_API_FILE=${TRTLLM_DIR}/examples/configs/curated/minimax-m3-throughput.yaml
```

If you don't have access to the source code locally, you can manually create the YAML config file using the code in the dropdown below.

````{admonition} Show MiniMax-M3 throughput config
:class: dropdown

```{literalinclude} ../../../examples/configs/curated/minimax-m3-throughput.yaml
:language: yaml
```
````

The configuration uses Data-Expert Parallelism (DEP): `enable_attention_dp: true` runs the attention layers data-parallel across ranks while the MoE experts run expert-parallel, which favors high-throughput / large-batch serving on MiniMax-M3.

For MXFP8 checkpoints, TensorRT LLM selects the GEMM backend automatically.
`TRTLLM_MXFP8_GEMM_BACKEND` is an advanced override for debugging and
performance experiments:

* `trtllm` uses the native TensorRT LLM GEMM for eager execution and CUDA graphs.
* `flashinfer` forces FlashInfer for eligible captured decode CUDA graphs; eager,
  context/prefill, and piecewise CUDA-graph execution remain on the native GEMM.
  It requires the pinned `flashinfer-python` package and Blackwell MXFP8 support.
* `auto` keeps eager execution on the native GEMM and uses FlashInfer in
  captured decode CUDA graphs after startup tuning.

Leave the variable unset for normal deployments. An explicit value disables
MiniMax-M3's automatic backend selection and uses the requested policy.

### Launch the TensorRT LLM Server

MiniMax-M3 is launched through the `trtllm-llmapi-launch` wrapper, which sets up the multi-rank (MPI/Slurm) environment that the parallel server requires. The wrapper is run once per rank by Slurm (`srun`), with one task (rank) per GPU. The example below launches the server across 2 nodes (`-N 2`), 4 GPUs per node (`--ntasks-per-node 4`, 8 ranks total), using the curated YAML to drive parallelism, batching, and the MiniMax-M3 sparse-attention backend:

```bash
export MODEL=/models/MiniMax-M3-MXFP8   # MXFP8 (recommended); use /models/MiniMax-M3 for BF16

srun -N 2 \
    --ntasks 8 --ntasks-per-node 4 \
    --mpi=pmix --gres=gpu:4 \
    --container-image=nvcr.io/nvidia/tensorrt-llm/release:x.y.z \
    --container-mounts=/models:/models,/workspace:/workspace,${TRTLLM_DIR}:${TRTLLM_DIR} \
    --container-workdir /workspace \
    bash -c "trtllm-llmapi-launch \
        python3 -m tensorrt_llm.commands.serve $MODEL \
          --trust_remote_code \
          --reasoning_parser minimax_m3 \
          --tool_parser minimax_m3 \
          --chat_template $MODEL/chat_template.jinja \
          --host 0.0.0.0 \
          --extra_llm_api_options $EXTRA_LLM_API_FILE"
```

The parallelism, batch, KV-cache, sparse-attention, and CUDA-graph settings all live in the YAML; no CLI flags need to change to tune them. The same `$EXTRA_LLM_API_FILE` is used for both the MXFP8 and BF16 checkpoints — point `$MODEL` at the directory of whichever checkpoint you want to serve.

> [!NOTE]
> Adjust `-N`, `--ntasks`, `--ntasks-per-node`, and `--gres=gpu:` to match your cluster's GPUs-per-node. The total number of tasks (ranks) must equal `tensor_parallel_size` (`8`). Add the partition / account / node-list flags (`-p`, `-A`, `-w`) required by your Slurm setup, and ensure `/models`, `/workspace`, and the TensorRT LLM repository resolve to the same shared paths on both nodes.

### Command-Line and YAML Options

#### Command-line options

* `--trust_remote_code`: Required to load the MiniMax-M3 configuration and custom code from the checkpoint.
* `--reasoning_parser minimax_m3`: Parses the MiniMax-M3 reasoning trace, which is wrapped in `<mm:think>...</mm:think>` tags, into the structured `reasoning_content` field of chat responses.
* `--tool_parser minimax_m3`: Parses MiniMax-M3 tool/function calls from the model output.
* `--chat_template`: Path to the chat template shipped with the checkpoint (`chat_template.jinja`).
* `--extra_llm_api_options`: Path to the YAML configuration file with the LLM API options described below.

#### `--extra_llm_api_options` YAML options

The curated config sets all the YAML knobs; this section only documents the MiniMax-M3-specific settings. Every other field in the YAML is a standard TensorRT LLM throughput knob — see the [`TorchLlmArgs` class](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.TorchLlmArgs) for the full reference.

##### `sparse_attention_config.algorithm`

Must be set to `minimax_m3`. There is no dense fallback for the MiniMax-M3 sparse-attention layers.

##### `kv_cache_config.enable_block_reuse`

Optional. KV cache block reuse is supported on the sparse-attention path; the curated throughput YAML leaves it disabled. Set it to `true` for workloads with shared prefixes (multi-turn chat, agent loops). See [KV cache reuse](#kv-cache-reuse).

## Testing API Endpoint

The server (the OpenAI-compatible REST endpoint) runs on the rank-0 node, listening on port `8000`. Send requests to that node's hostname or IP; `localhost` only works from the rank-0 node itself. The examples below use `localhost:8000` — replace it with the rank-0 node address when querying from elsewhere.

### Health Check

You can query the health/readiness of the server using:

```bash
curl -s -o /dev/null -w "Status: %{http_code}\n" "http://localhost:8000/health"
```

When `Status: 200` is returned, the server is ready for queries. Note that the very first query may take longer due to initialization and compilation.

### Basic Test

After the TensorRT LLM server is set up and shows *Application startup complete*, you can send requests to the server. MiniMax-M3 is a reasoning model, so use the `/v1/chat/completions` endpoint — that path applies the chat template and, together with the configured reasoning parser, returns the reasoning trace separately in `reasoning_content`. (The `/v1/completions` endpoint does not apply the chat template and is not recommended for this model.)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "MiniMaxAI/MiniMax-M3-MXFP8",
      "messages": [
          {"role": "user", "content": "What is the capital of France?"}
      ],
      "max_tokens": 256,
      "temperature": 0
  }'
```

Example response:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "MiniMaxAI/MiniMax-M3-MXFP8",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is **Paris**.",
        "reasoning_content": "The user is asking a simple factual question..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 134,
    "total_tokens": 181,
    "completion_tokens": 47
  }
}
```

The `message` object contains the visible answer in `content` and, when the model emits a `<mm:think>...</mm:think>` block, the parsed `reasoning_content`.

## Troubleshooting Tips

* **Sparse attention not applied:** Ensure `sparse_attention_config.algorithm` is set to `minimax_m3` in the YAML file passed to `--extra_llm_api_options`.
* **Model fails to load:** Make sure `--trust_remote_code` is set and that the checkpoint path is correct and fully downloaded (Git LFS).
* **Multi-node startup hangs or ranks can't find each other:** Verify that the model weights, the curated YAML, and all mounted paths resolve identically on both nodes (shared filesystem), that the total Slurm task count equals `tensor_parallel_size` (`8`), and that the inter-node interconnect (e.g., InfiniBand) is healthy.
* **Reasoning/tool output not parsed:** Verify `--reasoning_parser minimax_m3`, `--tool_parser minimax_m3`, and `--chat_template $MODEL/chat_template.jinja` are all passed.
* **GPU utilization:** For performance issues, check GPU utilization with `nvidia-smi` while the server is running.
* **Container startup:** If the container fails to start, verify that the NVIDIA Container Toolkit is properly installed.
* **Port conflicts:** Make sure the server port (`8000` in this guide) is not being used by another application.
* **Configuration files:** Ensure that YAML config files are correctly formatted to avoid runtime errors.

## Benchmarking Performance

To benchmark the performance of your TensorRT LLM server, you can use the built-in `benchmark_serving.py` script. First, create a wrapper `bench.sh` script:

```bash
cat << 'EOF' > bench.sh
concurrency_list="32 64 128 256 512 1024"
multi_round=5
isl=1024
osl=1024
result_dir=/tmp/minimax_m3_output

for concurrency in ${concurrency_list}; do
    num_prompts=$((concurrency * multi_round))
    python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --model MiniMaxAI/MiniMax-M3-MXFP8 \
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

To save results to files, add these options to each benchmark command:

```bash
--save-result \
--result-dir "${result_dir}" \
--result-filename "concurrency_${concurrency}.json"
```

For more benchmarking options see [benchmark_serving.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/serve/scripts/benchmark_serving.py).

Run `bench.sh` to begin a serving benchmark. This will take a long time if you run all the concurrencies.

```bash
./bench.sh
```

Sample TensorRT LLM serving benchmark output. Your results may vary due to ongoing software optimizations.

```
============ Serving Benchmark Result ============
Successful requests:                      [result]
Benchmark duration (s):                   [result]
Total input tokens:                       [result]
Total generated tokens:                   [result]
Request throughput (req/s):               [result]
Output token throughput (tok/s):          [result]
Total Token throughput (tok/s):           [result]
User throughput (tok/s):                  [result]
---------------Time to First Token----------------
Mean TTFT (ms):                           [result]
Median TTFT (ms):                         [result]
P99 TTFT (ms):                            [result]
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                           [result]
Median TPOT (ms):                         [result]
P99 TPOT (ms):                            [result]
---------------Inter-token Latency----------------
Mean ITL (ms):                            [result]
Median ITL (ms):                          [result]
P99 ITL (ms):                             [result]
----------------End-to-end Latency----------------
Mean E2EL (ms):                           [result]
Median E2EL (ms):                         [result]
P99 E2EL (ms):                            [result]
==================================================
```

### Key Metrics

#### Time to First Token (TTFT)
The typical time elapsed from when a request is sent until the first output token is generated.

#### Time Per Output Token (TPOT) and Inter-Token Latency (ITL)
* TPOT is the typical time required to generate each token *after* the first one.
* ITL is the typical time delay between the completion of one token and the completion of the next.
* Both TPOT and ITL ignore TTFT.

For a single request, ITLs are the time intervals between tokens, while TPOT is the average of those intervals:

$$
\text{TPOT (1 request)} = \text{Avg(ITL)} = \frac{\text{E2E latency} - \text{TTFT}}{\text{Num Output Tokens} - 1}
$$

Across different requests, **average TPOT** is the mean of each request's TPOT (all requests weighted equally), while **average ITL** is token-weighted (all tokens weighted equally):

$$
\text{Avg TPOT (N requests)} = \frac{\text{TPOT}_1 + \text{TPOT}_2 + \cdots + \text{TPOT}_N}{N}
$$

$$
\text{Avg ITL (N requests)} = \frac{\text{Sum of all ITLs across requests}}{\text{Num Output Tokens across requests}}
$$

#### End-to-End (E2E) Latency
The typical total time from when a request is submitted until the final token of the response is received.

#### Total Token Throughput
The combined rate at which the system processes both input (prompt) tokens and output (generated) tokens.

$$
\text{Total TPS} = \frac{\text{Num Input Tokens}+\text{Num Output Tokens}}{T_{last} - T_{first}}
$$

#### Tokens Per Second (TPS) or Output Token Throughput
How many output tokens the system generates each second.

$$
\text{TPS} = \frac{\text{Num Output Tokens}}{T_{last} - T_{first}}
$$
