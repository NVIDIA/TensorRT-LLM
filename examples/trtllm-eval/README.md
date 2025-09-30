# Accuracy Evaluation Tool `trtllm-eval`

We provide a CLI tool `trtllm-eval` for evaluating model accuracy. It shares the core evaluation logics with the [accuracy test suite](../../tests/integration/defs/accuracy) of TensorRT-LLM.

`trtllm-eval` is built on the offline API -- [LLM API](https://nvidia.github.io/TensorRT-LLM/llm-api/index.html). It provides developers a unified entrypoint for accuracy evaluation. Compared with the online API [`trtllm-serve`](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html), offline API provides clearer error messages and simplifies the debugging workflow.

`trtllm-eval` follows the CLI interface of [`trtllm-serve`](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html).

```bash
pip install -r requirements.txt

# Evaluate Llama-3.1-8B-Instruct on MMLU
trtllm-eval --model meta-llama/Llama-3.1-8B-Instruct mmlu

# Evaluate Llama-3.1-8B-Instruct on GSM8K
trtllm-eval --model meta-llama/Llama-3.1-8B-Instruct gsm8k

# Evaluate Llama-3.3-70B-Instruct on GPQA Diamond
trtllm-eval --model meta-llama/Llama-3.3-70B-Instruct gpqa_diamond
```

The `--model` argument accepts either a Hugging Face model ID or a local checkpoint path. By default, `trtllm-eval` runs the model with the PyTorch backend; pass `--backend tensorrt` to switch to the TensorRT backend. Alternatively, the `--model` argument also accepts a local path to pre-built TensorRT engines; in that case, please pass the Hugging Face tokenizer path to the `--tokenizer` argument.

See more details by `trtllm-eval --help`.
