# Responses API Examples

Examples for the `/v1/responses` endpoint. All examples in this directory use the Responses API, demonstrating features such as streaming, tool/function calling, and multi-turn dialogue.

## Quick Start

```bash
# Run the basic example
python example_01_basic_chat.py
```

## Examples Overview

### Basic Examples

1. **`example_01_basic_chat.py`** - Start here!
   - Simple request/response
   - Non-streaming mode
   - Uses `input` parameter for user message

2. **`example_02_streaming_chat.py`** - Real-time responses
   - Stream tokens as generated
   - Handles various event types (`response.created`, `response.output_text.delta`, etc.)
   - Server-Sent Events (SSE)

3. **`example_03_multi_turn_conversation.py`** - Context management
   - Multiple conversation turns
   - Uses `previous_response_id` to maintain context
   - Follow-up questions without resending history

### Advanced Examples

4. **`example_04_json_mode.py`** - Structured output
   - JSON schema validation via `text.format`
   - Structured data extraction
   - Requires xgrammar support

5. **`example_05_tool_calling.py`** - Function calling
   - External tool integration
   - Function definitions with `tools` parameter
   - Tool result handling with `function_call_output`
   - Requires compatible model (Qwen3, GPT-OSS, Kimi K2)

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

### Multi-turn Context

Use `previous_response_id` to continue conversations:
```python
# First turn
response1 = client.responses.create(
    model=model,
    input="What is 15 multiplied by 23?",
)

# Second turn - references previous response
response2 = client.responses.create(
    model=model,
    input="Now divide that result by 5",
    previous_response_id=response1.id,
)
```

### Tool Calling

Define functions the model can call:
```python
tools = [{
    "name": "get_weather",
    "type": "function",
    "description": "Get the current weather in a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
        },
        "required": ["location"],
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
| Tool calling | Compatible model (Qwen3, GPT-OSS, Kimi K2) |
