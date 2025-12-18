# OpenAI API Compatibility Examples

This directory contains individual, self-contained examples demonstrating TensorRT-LLM's OpenAI API compatibility. Examples are organized by API endpoint.

## Prerequisites

1. **Start the trtllm-serve server:**
   ```bash
   trtllm-serve meta-llama/Llama-3.1-8B-Instruct
   ```

   for reasoning model or model with tool calling ability. Specify `--tool_parser` and `--reasoning_parser`, e.g.

   ```bash
   trtllm-serve Qwen/Qwen3-8B --reasoning_parser "qwen3" --tool_parser "qwen3"
   ```


## Running Examples

Each example is a standalone Python script. Run from the example's directory:

```bash
# From chat_completions directory
cd chat_completions
python example_01_basic_chat.py
```

Or run with full path from the repository root:

```bash
python examples/serve/compatibility/chat_completions/example_01_basic_chat.py
```

### 📋 Complete Example List

#### Chat Completions (`/v1/chat/completions`)

| Example | File | Description |
|---------|------|-------------|
| **01** | `chat_completions/example_01_basic_chat.py` | Basic non-streaming chat completion |
| **02** | `chat_completions/example_02_streaming_chat.py` | Streaming responses with real-time delivery |
| **03** | `chat_completions/example_03_multi_turn_conversation.py` | Multi-turn conversation with context |
| **04** | `chat_completions/example_04_streaming_with_usage.py` | Streaming with continuous token usage stats |
| **05** | `chat_completions/example_05_json_mode.py` | Structured output with JSON schema |
| **06** | `chat_completions/example_06_tool_calling.py` | Function/tool calling with tools |
| **07** | `chat_completions/example_07_advanced_sampling.py` | TensorRT-LLM extended sampling parameters |

#### Responses (`/v1/responses`)

| Example | File | Description |
|---------|------|-------------|
| **01** | `responses/example_01_basic_chat.py` | Basic non-streaming response |
| **02** | `responses/example_02_streaming_chat.py` | Streaming with event handling |
| **03** | `responses/example_03_multi_turn_conversation.py` | Multi-turn using `previous_response_id` |
| **04** | `responses/example_04_json_mode.py` | Structured output with JSON schema |
| **05** | `responses/example_05_tool_calling.py` | Function/tool calling with tools |

## Configuration

All examples use these default settings:

```python
base_url = "http://localhost:8000/v1"
api_key = "tensorrt_llm"  # Can be any string
```

To use a different server:

```python
client = OpenAI(
    base_url="http://YOUR_SERVER:PORT/v1",
    api_key="your_key",
)
```

## Model Requirements

Some examples require specific model capabilities:

| Feature | Model Requirement |
|---------|------------------|
| JSON Mode | xgrammar support |
| Tool Calling | Tool-capable model (Qwen3, GPT-OSS, Kimi K2) |
| Others | Any model |
