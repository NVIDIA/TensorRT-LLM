```{warning}
**Legacy Workflow:** This page documents the legacy TensorRT engine-build
workflow. For new projects, use `trtllm-serve` or the LLM API with the
PyTorch backend. See the [Quick Start Guide](../quick-start-guide.md).
```

# Key Features

This document lists key features supported in TensorRT-LLM.

- [Quantization](reference/precision.md)
- [Inflight Batching](advanced/gpt-attention.md#in-flight-batching)
- [Chunked Context](advanced/gpt-attention.md#chunked-context)
- [LoRA](advanced/lora.md)
- [KV Cache Reuse](advanced/kv-cache-reuse.md)
- [Speculative Sampling](advanced/speculative-decoding.md)
