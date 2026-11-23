<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-->

# Multimodal Support in TensorRT LLM

TensorRT LLM supports a variety of multimodal models, enabling efficient inference with inputs beyond just text.

---

## Background

Multimodal LLMs typically handle non-text inputs by combining a multimodal encoder with an LLM decoder. The encoder first transforms non-text modality input into embeddings, which are then fused with text embeddings and fed into the LLM decoder for downstream inference. Compared to standard LLM inference, multimodal LLM inference involves three additional stages to support non-text modalities.

* **Multimodal Input Processor**: Preprocess raw multimodal input into a format suitable for the multimodal encoder, such as pixel values for vision models.
* **Multimodal Encoder**: Encodes the processed input into embeddings that are aligned with the LLM’s embedding space.
* **Integration with LLM Decoder**: Fuses multimodal embeddings with text embeddings as the input to the LLM decoder.

## Optimizations

TensorRT LLM incorporates some key optimizations to enhance the performance of multimodal inference:

* **In-Flight Batching**: Batches multimodal requests within the GPU executor to improve GPU utilization and throughput.
* **CPU/GPU Concurrency**: Asynchronously overlaps data preprocessing on the CPU with image encoding on the GPU.
* **Raw data hashing**: Leverages image hashes and token chunk information to improve KV cache reuse and minimize collisions.

Further optimizations are under development and will be updated as they become available.

## Encoder attention backends

`multimodal_config.encoder_attention_backend` selects a TensorRT-LLM attention backend for each logical encoder. The public values are `VANILLA` (PyTorch scaled dot product attention), `TRTLLM`, and `FLASHINFER`; values are case-insensitive. Image and video inputs share the `vision` encoder.

```yaml
multimodal_config:
  encoder_attention_backend:
    vision: FLASHINFER
```

The same setting is available in the Python API:

```python
from tensorrt_llm import LLM

llm = LLM(
    model="Qwen/Qwen3-VL-8B-Instruct",
    multimodal_config={"encoder_attention_backend": {"vision": "VANILLA"}},
)
```

Omitted entries preserve the model's existing encoder defaults, including inheritance from `attn_backend` where applicable. An encoder override leaves the language model's backend unchanged. Backend availability and supported tensor shapes still depend on the installed kernels and hardware; floating-point differences between backends are expected. `VANILLA` does not support `multimodal_config.encoder_cuda_graph`, so selecting it for an encoder with CUDA graphs enabled fails during configuration.

The table covers every registered multimodal model implementation; model aliases share the same declaration. Image/video share `vision`, and `audio` names a separate encoder. "Not supported" means this option cannot change that encoder's attention backend; it does not mean the model or modality is unsupported. Gemma 4 Unified uses encoder-free projections, and Mllama's vision input is not implemented. The CPU registry test checks this table against each model's explicit `supported_encoder_attention_backends` declaration.

<!-- encoder-attention-backends:start -->
| Model | Logical encoder | Supported backends |
| --- | --- | --- |
| `Cosmos3EdgeModel` | `vision` | Not supported |
| `Cosmos3Model` | `vision` | `VANILLA`, `TRTLLM`, `FLASHINFER` |
| `DeepseekV41ForConditionalGeneration` | `vision` | Not supported |
| `Exaone4_5_ForConditionalGeneration` | `vision` | `VANILLA`, `TRTLLM`, `FLASHINFER` |
| `Gemma3VLM` | `vision` | Not supported |
| `Gemma4ForConditionalGeneration` | `audio` | Not supported |
| `Gemma4ForConditionalGeneration` | `vision` | `VANILLA`, `TRTLLM`, `FLASHINFER` |
| `Gemma4UnifiedForConditionalGeneration` | None (encoder-free) | Not supported |
| `Glm5NextVLM` | `vision` | `TRTLLM` |
| `HCXVisionForCausalLM` | `vision` | Not supported |
| `KimiK25ForConditionalGeneration` | `vision` | `VANILLA`, `TRTLLM`, `FLASHINFER` |
| `KimiK3ForConditionalGeneration` | `vision` | `VANILLA`, `TRTLLM`, `FLASHINFER` |
| `Llama4ForConditionalGeneration` | `vision` | Not supported |
| `LlavaNextModel` | `vision` | Not supported |
| `MiniCPMV4_6Model` | `vision` | `VANILLA`, `TRTLLM`, `FLASHINFER` |
| `MiniMaxM3VLForConditionalGeneration` | `vision` | Not supported |
| `Mistral3VLM` | `vision` | `VANILLA`, `TRTLLM`, `FLASHINFER` |
| `MllamaForConditionalGeneration` | `vision` | Not supported |
| `MuseSparkForConditionalGeneration` | `audio` | Not supported |
| `MuseSparkForConditionalGeneration` | `vision` | Not supported |
| `NemotronHMultimodalModel` | `audio` | Not supported |
| `NemotronHMultimodalModel` | `vision` | `VANILLA`, `TRTLLM`, `FLASHINFER` |
| `Phi4MMForCausalLM` | `audio` | Not supported |
| `Phi4MMForCausalLM` | `vision` | Not supported |
| `Qwen2VLModel` | `vision` | Not supported |
| `Qwen2_5_VLModel` | `vision` | `VANILLA`, `TRTLLM`, `FLASHINFER` |
| `Qwen3MoeVLModel` | `vision` | `VANILLA`, `TRTLLM`, `FLASHINFER` |
| `Qwen3VLModel` | `vision` | `VANILLA`, `TRTLLM`, `FLASHINFER` |
| `Qwen3_5MoeVLModel` | `vision` | `VANILLA`, `TRTLLM`, `FLASHINFER` |
| `Qwen3_5VLModel` | `vision` | `VANILLA`, `TRTLLM`, `FLASHINFER` |
| `Qwen4ExpForConditionalGeneration` | `vision` | `VANILLA`, `TRTLLM`, `FLASHINFER` |
| `Step3p7VLForConditionalGeneration` | `vision` | `VANILLA`, `TRTLLM`, `FLASHINFER` |
| `VilaModel` | `vision` | Not supported |
| `WhisperForConditionalGeneration` | `audio` | Not supported |
<!-- encoder-attention-backends:end -->

For MiniCPM-V 4.6, the selection applies to its TensorRT-LLM attention layers; its fixed window-merger attention is unchanged. The Nemotron-H RADIO vision encoder keeps its `FLASHINFER` default, and `VANILLA` and `TRTLLM` remain selectable.

Supported entries use TensorRT-LLM's `Attention` and apply encoder overrides explicitly. Other encoders keep their fixed implementation, including Hugging Face attention, direct SDPA, and custom attention paths. Overrides for entries marked "Not supported" fail during initialization, even if the requested value is `VANILLA`. Unknown encoder names and unsupported backends also fail during initialization; Hugging Face `attn_implementation` names are not accepted. With `disable_mm_encoder=True`, encoders that the model skips do not use the setting, so text-only runs can reuse a serving configuration.

## Model Support Matrix

Please refer to the latest multimodal [support matrix](../models/supported-models.md#multimodal-feature-support-matrix-pytorch-backend).

## Adding a Model

See the [multimodal onboarding guide](../developer-guide/multimodal.md#onboarding-a-model)
for processor metadata, payload selection, model hooks, and encoder/prefill handoff.

## Optional dependencies

Video input decoding uses OpenCV (`cv2`) and is not installed by default. Install it only for models with video inputs:

```bash
pip install opencv-python-headless
```

Image-only and text-only workflows do not require this package.

## Examples

The following examples demonstrate how to use TensorRT LLM's multimodal support in various scenarios, including quick run examples, serving endpoints, and performance benchmarking.

### Quick start

Quickly try out TensorRT LLM's multimodal support using our `LLM-API` and a ready-to-run [example](source:examples/llm-api/quickstart_multimodal.py). The script imports helpers from its sibling file [`quickstart_advanced.py`](source:examples/llm-api/quickstart_advanced.py), so run it from a checkout of the `examples/llm-api/` directory rather than as a single downloaded file:

```bash
# Get the examples directory first; quickstart_multimodal.py imports from
# quickstart_advanced.py next to it.
git clone --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM/examples/llm-api
python3 quickstart_multimodal.py --model_dir Efficient-Large-Model/NVILA-8B --modality image --disable_kv_cache_reuse
```

### OpenAI-Compatible Server via [`trtllm-serve`](../../source/commands/trtllm-serve/trtllm-serve.rst)

Launch an OpenAI-compatible server with multimodal support using the `trtllm-serve` command, for example:

```bash
trtllm-serve Qwen/Qwen2-VL-7B-Instruct  --backend pytorch
```

You can then send OpenAI-compatible requests, such as via curl or API clients, to the server endpoint. See [curl chat client for multimodal script](source:examples/serve/curl_chat_client_for_multimodal.sh) as an example.

`trtllm-serve` renders the chat template itself and marks the resulting `TextPrompt` with `chat_template_applied`, as do the multimodal prompt builders in `tensorrt_llm.inputs.utils`. An input processor that would otherwise apply the chat template to a raw `prompt` should use such a prompt verbatim; see `TextPrompt` in `tensorrt_llm/inputs/data.py`.

### Input processing and encoder metrics

Enable the existing metrics options in the serving YAML:

```yaml
return_perf_metrics: true
enable_iter_perf_stats: true
```

`/prometheus/metrics` exposes these histograms; `/metrics` serves iteration statistics as JSON.

| Metric | Measurement |
| --- | --- |
| `trtllm_input_processing_seconds{stage="total"}` | Wall time from entry into a chat, completion, encoder, Harmony or Responses handler through synchronous input preparation and request submission, before waiting for generation. Includes message parsing, rendering, media resolution, preprocessing and their waits. |
| `trtllm_input_processing_seconds{stage="media_load"}` | Time awaiting the common media resolver, including concurrent downloads, decodes and worker waits. |
| `trtllm_input_processing_seconds{stage="chat_template"}` | Time awaiting chat template rendering, when rendering is needed. |
| `trtllm_input_processing_seconds{stage="processor"}` | Input processor wall time inside its worker, including tokenization, model preprocessing and hashing. |
| `trtllm_input_processing_queue_seconds` | Wait for an input processor worker, separate from processor execution. |
| `trtllm_media_processing_seconds{stage="download", modality="image"}` | Per-item HTTP(S) download wall time, excluding shared HTTP session initialization. Also supports `audio` and `video`. Local and inline data inputs do not create download observations. |
| `trtllm_media_processing_seconds{stage="decode", modality="image"}` | Per-item decode execution in the common media worker, separate from queue time. Local file decoding includes file IO; inline decoding includes base64 conversion. |
| `trtllm_media_processing_seconds{stage="decode_queue", modality="image"}` | Per-item wait for a media decode worker. |
| `trtllm_multimodal_encoder_gpu_time_seconds` | Rank-0 CUDA-event elapsed time per encoder call through `MultimodalModelMixin`, including item scheduling, inline encoding and side-stream prefetch. |

Input timings are captured in a request context independently of Prometheus. When a route emits a completed executor metrics record, it also carries them in `perf_metrics.input_processing_metrics` in the existing JSONL output (`perf_metrics_output_dir`). Values are seconds: request-level stages use keys such as `total`, `media_load`, `chat_template`, `processor` and `processor_queue`. Per-item times are summed by stage and modality within each request, for example `media_download_image` and `media_decode_image`; these sums can exceed elapsed request time when items overlap. Batched completion inputs share the HTTP request's frontend timings. Anthropic Message Batch items have separate request contexts and JSONL records.

`/health_generate` probes do not contribute to the input-processing histograms; their request-local timings and existing executor metrics remain available. For Anthropic Messages, `total` starts in the delegated chat handler and excludes Anthropic request conversion and validation. Shared HTTP session initialization remains included in `media_load` and `total`.

Encoder-only servers expose the same Prometheus endpoint and completed-request input timings when metrics are enabled. Applications that configure a `ProcessPoolExecutor` for media loading retain the direct worker submission path; per-worker `decode` and `decode_queue` timings are available only for thread executors.

For selected requests, send `X-TRTLLM-return-metrics: 1` to receive the same values in `Server-Timing`, for example `server_input_total;dur=150.000000` (milliseconds). Streaming responses use the existing `trtllm.perf_metrics` SSE event. These fields survive context/generation worker handoff and remain separate from executor timing and batch-level encoder GPU measurements. Stage histograms include failed attempts; request JSONL records follow the existing completed-request reporting path.

Processor metrics cover the chat and text completion paths that offload preprocessing to the input processor pool. Encoder metrics exclude warmup, cache-only lookups and H2D transfers performed before the encoder call. Calls using precomputed embeddings or legacy paths outside `MultimodalModelMixin` do not produce encoder observations. The count is encoder calls, not requests or media items. The timings describe rank 0; they are not a sum or maximum across GPUs. Completed encoder observations may appear in a later iteration's JSON field, `multimodalEncoderStats.gpuTimesMS`, because collection does not wait for the GPU.

Media loading and chat rendering can overlap. Encoder work can overlap LLM execution. Do not add these histogram averages to reconstruct TTFT or divide a shared encoder call across requests. Use client TTFT and throughput alongside the stage distributions to identify bottlenecks.

[AIPerf server metrics](https://docs.nvidia.com/aiperf/server-metrics/server-metrics-collection) can scrape these metrics without a TRT-LLM-specific adapter. Add the explicit endpoint to your profile command:

```bash
aiperf profile --model MODEL --url http://localhost:8000 \
  --endpoint-type chat --endpoint /v1/chat/completions \
  --server-metrics http://localhost:8000/prometheus/metrics \
  --server-metrics-formats json csv jsonl
```

The histogram sum and count provide total measured time and mean; percentile values are bucket estimates. Grafana can query the same Prometheus series.

### Run with [`trtllm-bench`](../../source/commands/trtllm-bench.rst)

Evaluate offline inference performance with multimodal inputs using the `trtllm-bench` tool. For detailed instructions, see the [benchmarking guide](../developer-guide/perf-benchmarking.md).
