# Nemotron-NAS

Nemotron-NAS models (for example `Llama-3_1-Nemotron-51B-Instruct`) run on the
TensorRT LLM **PyTorch backend** — HuggingFace checkpoints are loaded directly,
with no checkpoint-conversion or engine-build step.

The recommended flow is the TensorRT LLM PyTorch workflow:

- LLM Python API quickstart: [examples/llm-api/quickstart_advanced.py](../../../llm-api/quickstart_advanced.py)
- Serving: [`trtllm-serve`](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html)

## Support Matrix

* FP16
* BF16
* FP8
* Tensor parallelism
* Pipeline parallelism
