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

### Encoder attention backend

Use `multimodal_config.encoder_attention_backend` to select attention for a logical
encoder without changing the language model's `attn_backend`. Image and video share
the `vision` encoder; an audio tower uses `audio`. Omit a key to preserve the
model's current default, including decoder-backend inheritance where applicable.

For example, select FlashInfer for Qwen vision through the serving config:

```yaml
multimodal_config:
  encoder_attention_backend:
    vision: FLASHINFER
```

```bash
trtllm-serve Qwen/Qwen3.8-27B --config config.yaml
```

The Python API accepts the same mapping through `MultimodalConfig`:

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import MultimodalConfig

llm = LLM(
    model="Qwen/Qwen3.8-27B",
    multimodal_config=MultimodalConfig(
        encoder_attention_backend={"vision": "FLASHINFER"},
    ),
)
```

The following encoder implementations consume this option. Backend names are
case-insensitive. Explicit unknown encoder names or unsupported backends raise
an initialization error; an explicit FlashInfer request requires FlashInfer to
be installed.

| Encoder family | Vision backends | Audio backends |
| --- | --- | --- |
| Native Qwen2.5-VL/EXAONE4.5, Qwen3-VL, Qwen3.5/3.8, Qwen4Exp, Cosmos3 | `TRTLLM`, `FLASHINFER`, `VANILLA` | No audio tower |
| Nemotron H VL/Omni | `TRTLLM`, `FLASHINFER`, `VANILLA` | `SDPA` when the checkpoint has a sound encoder |
| Gemma4 | `TRTLLM`, `FLASHINFER`, `VANILLA` | `VANILLA` |
| Mistral3/Pixtral | `TRTLLM`, `FLASHINFER`, `VANILLA` | No audio tower |
| Kimi K2.5/K3, Step3.7 | `TRTLLM`, `FLASHINFER`, `VANILLA` | No audio tower |
| MiniCPM-V4.6 full attention | `TRTLLM`, `FLASHINFER`, `VANILLA` | No audio tower |
| GLM5 Next | `TRTLLM` | No audio tower |
| DeepSeek-V4.1, MiniMax M3 | `SDPA` | No audio tower |
| Hugging Face Qwen2/2.5-VL vision | `FLASH_ATTENTION_2`, `SDPA`, `VANILLA` | No audio tower |
| Gemma4 Unified | No encoder attention | No encoder attention |

For a Nemotron checkpoint with both towers, use
`{"vision": "FLASHINFER", "audio": "SDPA"}` to configure them separately.
Gemma4 audio retains its custom relative-position and softcap attention, exposed
as `VANILLA`; it does not implement the vision backends. Gemma4 Unified uses
encoder-free projections and rejects encoder-attention overrides.
MiniCPM-V4.6's fixed-size window merger retains its batched SDPA implementation;
the selection applies to its variable-length full attention.

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

### Run with [`trtllm-bench`](../../source/commands/trtllm-bench.rst)

Evaluate offline inference performance with multimodal inputs using the `trtllm-bench` tool. For detailed instructions, see the [benchmarking guide](../developer-guide/perf-benchmarking.md).
