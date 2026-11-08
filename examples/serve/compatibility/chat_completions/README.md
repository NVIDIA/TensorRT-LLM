# Chat Completions API Examples

Examples for the `/v1/chat/completions` endpoint - the most versatile API for conversational AI.

## Quick Start

```bash
# Run the basic example
python example_01_basic_chat.py
```

## Examples Overview

### Basic Examples

1. **`example_01_basic_chat.py`** - Start here!
   - Simple request/response
   - Shows token usage
   - Non-streaming mode

2. **`example_02_streaming_chat.py`** - Real-time responses
   - Stream tokens as generated
   - Better UX for long responses
   - Server-Sent Events (SSE)

3. **`example_03_multi_turn_conversation.py`** - Context management
   - Multiple conversation turns
   - Conversation history
   - Follow-up questions

4. **`example_04_streaming_with_usage.py`** - Streaming + metrics
   - Continuous token counts
   - `stream_options` parameter
   - Monitor resource usage

### Advanced Examples

5. **`example_05_json_mode.py`** - Structured output
   - JSON schema validation
   - Structured data extraction
   - Requires xgrammar

6. **`example_06_tool_calling.py`** - Function calling
   - External tool integration
   - Function definitions
   - Requires compatible model (Qwen3, gpt_oss)

7. **`example_07_advanced_sampling.py`** - Fine-tuned control
   - `top_k`, `repetition_penalty`
   - Custom stop sequences
   - TensorRT-LLM extensions

## Key Concepts

### Non-Streaming vs Streaming

**Non-Streaming** (`stream=False`):
- Wait for complete response
- Single response object
- Simple to use

**Streaming** (`stream=True`):
- Tokens delivered as generated
- Better perceived latency
- Server-Sent Events (SSE)

### Conversation Context

Messages accumulate in the `messages` array:
```python
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"},  # Next turn
]
```

### Tool Calling

Define functions the model can call:
```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {...}
    }
}]
```

## Model Requirements

| Feature | Requirement |
|---------|-------------|
| Basic chat | Any model |
| Streaming | Any model |
| Multi-turn | Any model |
| JSON mode | xgrammar support |
| Tool calling | Compatible model (Qwen3 and gpt_oss.) |
