```{warning}
**Legacy Workflow:** This page documents the legacy TensorRT engine-build
workflow. For new projects, use `trtllm-serve` or the LLM API with the
PyTorch backend. See the [Quick Start Guide](../quick-start-guide.md).
```

# LLM API with TensorRT Engine
A simple inference example with TinyLlama using the LLM API:

```{literalinclude} ../../../examples/llm-api/_tensorrt_engine/quickstart_example.py
    :language: python
    :linenos:
```

For more advanced usage including distributed inference, multimodal, and speculative decoding, please refer to this [README](../../../examples/llm-api/README.md).
