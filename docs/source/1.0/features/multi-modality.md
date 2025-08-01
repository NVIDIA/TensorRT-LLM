# Multimodal Support in TensorRT-LLM

TensorRT-LLM supports a variety of multimodal models, enabling efficient inference beyond just text.

---

## Background

Multimodal LLMs typically handle non-text inputs by combining a multimodal encoder with an LLM decoder. The encoder first transforms non-text modality input into embeddings, which are then fused with text embeddings and fed into the LLM decoder for downstream inference. Compared to standard LLM inference, multimodal LLM inference involves three additional stages to support non-text modalities.

* **Multimodal Input Processor**: Preprocess raw multimodal input into a format suitable for the multimodal encoder, such as pixel values for vision models.
* **Multimodal Encoder**: Encodes the processed input into embeddings that are aligned with the LLM’s embedding space.
* **Integration with LLM Decoder**: Fuses multimodal embeddings with text embeddings as the input to the LLM decoder.

## Optimizations

TensorRT-LLM incorporates some key optimizations to enhance the performance of multimodal inference:

* **In-Flight Batching**: Batches multimodal requests within the GPU executor to improve GPU utilization and throughput.
* **CPU/GPU Concurrency**: Asynchronously overlaps data preprocessing on the CPU with image encoding on the GPU.
* **Raw data hashing**: Leverages image hashes and token chunk info to improve KV cache reuse and minimize collisions.

Further optimizations are under development and will be updated as they become available.

## Model Support Matrix

Please refer to the latest multimodal [support matrix](/docs/source/reference/support-matrix.md).

## Examples

The following examples demonstrate how to use TensorRT-LLM's multimodal support in various scenarios, including quick run examples, serving endpoints, and performance benchmarking.

### Quick start

Quickly try out TensorRT-LLM's multimodal support using our `LLM-API` and a ready-to-run [example](/examples/llm-api/quickstart_multimodal.py):

```bash
python3 quickstart_multimodal.py --model_dir Efficient-Large-Model/NVILA-8B --modality image
```

### OpenAI-Compatible Server via `trtllm-serve`

Launch an OpenAI-compatible server with multimodal support using the `trtllm-serve` command, e.g.,

```bash
trtllm-serve Qwen/Qwen2-VL-7B-Instruct  --backend pytorch
```

You can then send OpenAI-compatible requests (e.g., via `curl` or API clients) to the server endpoint — for example, see [this script](/examples/serve/curl_chat_client_for_multimodal.sh).

### Run with `trtllm-bench`

Evaluate offline inference performance with multimodal inputs using the `trtllm-bench` tool. For detailed instructions, see the [benchmarking guide](/docs/source/performance/perf-benchmarking.md).