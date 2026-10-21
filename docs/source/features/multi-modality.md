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
| `trtllm_input_processing_seconds{stage="media_load"}` | Time awaiting media resolution, including fetch, decode and worker waits, per chat request with media. |
| `trtllm_input_processing_seconds{stage="chat_template"}` | Time awaiting chat template rendering, when rendering is needed. |
| `trtllm_input_processing_seconds{stage="processor"}` | Input processor wall time inside its worker, including tokenization, model preprocessing and hashing. |
| `trtllm_input_processing_queue_seconds` | Wait for an input processor worker, separate from processor execution. |
| `trtllm_multimodal_encoder_gpu_time_seconds` | Rank-0 CUDA-event elapsed time per encoder call through `MultimodalModelMixin`, including item scheduling, inline encoding and side-stream prefetch. |

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
