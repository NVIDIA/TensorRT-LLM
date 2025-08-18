# TensorRT-LLM OpenAPI Client Examples

This directory contains simple client examples using the `requests` library to interact with TensorRT-LLM OpenAPI servers.

## Files

- **`disagg_serving_test.py`** - Comprehensive client with multiple test cases
- **`simple_client_example.py`** - Minimal example showing core usage patterns

## Prerequisites

1. Start a TensorRT-LLM server manually using one of these methods:

### Option A: Using trtllm-serve (Recommended)
```bash
trtllm-serve "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

### Option B: Using the FastAPI server example
```bash
cd ../apps
python fastapi_server.py TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### Option C: Using any OpenAI-compatible server
Make sure it has a `/v1/chat/completions` endpoint that accepts:
```json
{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "Your prompt here"}],
    "max_tokens": 100,
    "temperature": 0.8,
    "stream": false
}
```

## Usage

### Simple Client (Minimal Example)
```bash
python simple_client_example.py
```

### Comprehensive Client
```bash
# Default server URL (http://localhost:8000)
python disagg_serving_test.py

# Custom server URL
python disagg_serving_test.py --server-url http://localhost:8080
```

## Example Output

```
🤖 Testing TensorRT-LLM server at: http://localhost:8000
==================================================
1. Health check...
   ✅ Server healthy: {'status': 'healthy'}

2. Testing text generation...

🎯 Test 1: 'Hello, my name is'
   Generated: 'John and I am a software engineer...'

🎯 Test 2: 'The capital of France is'
   Generated: 'Paris, the city of lights...'

3. Testing streaming generation...
🎯 Streaming: 'Write a short story about a robot:'
📡 Response: Once upon a time, there was a robot named...

✅ All tests completed!
```

## API Endpoints Used

The clients expect these endpoints:

- `GET /health` - Health check (fallback: simple chat completion request)
- `POST /v1/chat/completions` - OpenAI-compatible chat completions
- Streaming support via Server-Sent Events (SSE)

## Key Features Demonstrated

1. **Basic text generation** with requests.post()
2. **Streaming response** handling with SSE
3. **Error handling** for connection issues
4. **Session management** for efficient connections
5. **OpenAI-compatible format** with messages array

## Example Code

The minimal example shows exactly what you need:

```python
import requests

# Basic generation (OpenAI format)
payload = {
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "Hello, my name is"}],
    "max_tokens": 50,
    "temperature": 0.8
}
response = requests.post("http://localhost:8000/v1/chat/completions", json=payload)
result = response.json()
text = result["choices"][0]["message"]["content"]

# Streaming generation  
payload = {
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "The future of AI is"}],
    "stream": True
}
response = requests.post("http://localhost:8000/v1/chat/completions", json=payload, stream=True)
```

## Customization

You can modify the scripts to:
- Change server URLs
- Adjust generation parameters (temperature, max_tokens, etc.)
- Add new test prompts
- Handle different response formats 
