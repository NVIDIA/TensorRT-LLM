# Online Serving Examples with `trtllm-serve`

We provide a CLI command, `trtllm-serve`, to launch a FastAPI server compatible with OpenAI APIs, here are some client examples to query the server, you can check the source code here or refer to the [command documentation](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html) and [examples](https://nvidia.github.io/TensorRT-LLM/examples/trtllm_serve_examples.html) for detailed information and usage guidelines.

## LLM Serving Example

To serve a language model (e.g., TinyLlama), use the following command:

```bash
trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

This will start a FastAPI server on `localhost:8000` that's compatible with OpenAI's API format.

### Querying the Server

You can query the server using either Python or curl:

#### Python Example
```bash
# Chat completion example
python examples/serve/openai_chat_client.py

# Completion example
python examples/serve/openai_completion_client.py
```

#### Curl Example
```bash
# Chat completion
bash examples/serve/curl_chat_client.sh

# Completion
bash examples/serve/curl_completion_client.sh
```

## Multimodal Serving Example

For multimodal models (e.g., Qwen2-VL), you'll need to create a configuration file and start the server with additional options:

1. First, create a configuration file:
```bash
cat >./extra-llm-api-config.yml<<EOF
kv_cache_config:
    enable_block_reuse: false
EOF
```

2. Start the server with the configuration:
```bash
trtllm-serve Qwen/Qwen2-VL-7B-Instruct \
    --backend pytorch \
    --extra_llm_api_options extra-llm-api-config.yml
```

Note: Multimodal models are only compatible with `--backend pytorch`.

### Querying the Server

#### Python Example
Run the Python client script for multimodal:
```bash
# Chat completion
python examples/serve/openai_chat_client_for_multimodal.py
```

#### Curl Example
Run the curl client script for multimodal:
```bash
# Chat completion
bash examples/serve/curl_chat_client_for_multimodal.sh
```
