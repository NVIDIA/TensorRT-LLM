# Multimodal Disaggregated Serving (Experimental)

This example demonstrates how to set up disaggregated multimodal serving with TensorRT-LLM, where the vision encoder and language model decoder run as separate services for improved scalability and resource utilization.

## ⚠️ Disclaimer

**This is a Proof-of-Concept (POC) and early demonstration with several limitations:**

1. **API Support**: Only OpenAI chat completion mode is supported
2. **Model Support**: Limited to LLaVA-Next models only
3. **Modality Support**: Image modality only (no video support yet)
4. **Server Configuration**: Only supports 1 encoder server and 1 LLM server (though the LLM server can have multiple workers via tensor parallelism)

## Overview

Disaggregated multimodal serving separates the multimodal pipeline into distinct components:

- **Encoder Server**: Handles vision processing (images) using the multimodal encoder
- **LLM Decoder Server**: Processes text generation using the language model
- **Disaggregated Server**: Orchestrates requests between encoder and decoder services

This architecture enables better resource utilization and scalability by allowing independent scaling of vision and language processing components.

## Setup Instructions

### Step 1: Prepare Configuration Files

Create the required configuration files in your working directory:

#### LLM API Configuration (`extra-llm-api-config.yml`)
```bash
# Note: Current multimodal implementation does not support KV cache reuse,
# so we disable it for all cases
cat > ./extra-llm-api-config.yml << EOF
kv_cache_config:
    enable_block_reuse: false
EOF
```

#### Disaggregated Server Configuration (`disagg_config.yaml`)
```bash
cat > ./disagg_config.yaml << EOF
hostname: localhost
port: 8000
backend: pytorch
multimodal_servers:
  num_instances: 1
  urls:
      - "localhost:8001"
generation_servers:
  num_instances: 1
  urls:
      - "localhost:8002"
EOF
```

### Step 2: Start the Encoder Server

Launch the multimodal encoder server on GPU 0:

```bash
mkdir -p Logs/
CUDA_VISIBLE_DEVICES=0 trtllm-serve encoder llava-hf/llava-v1.6-mistral-7b-hf \
    --host localhost \
    --port 8001 \
    --backend pytorch \
    &> Logs/log_encoder_0 &
```

### Step 3: Start the LLM Decoder Server

Launch the language model decoder server on GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 trtllm-serve llava-hf/llava-v1.6-mistral-7b-hf \
    --host localhost \
    --port 8002 \
    --backend pytorch \
    --extra_llm_api_options ./extra-llm-api-config.yml \
    &> Logs/log_pd_tp1 &
```

### Step 4: Start the Disaggregated Orchestrator

Launch the disaggregated server that coordinates between encoder and decoder:

```bash
trtllm-serve disaggregated_mm -c disagg_config.yaml &> Logs/log_disagg_server &
```

## Alternative Setup

Instead of running Steps 2-4 manually, you can start all services at once using the provided script:

```bash
./start_disagg_mm.sh
```

This script will start the encoder server, LLM decoder server, and disaggregated orchestrator automatically with the same configuration as the manual steps above.

## Multi-GPU Decoder Configuration

For larger models and higher throughput, you can run the decoder with tensor parallelism (TP>1) across multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=1,2 trtllm-serve llava-hf/llava-v1.6-mistral-7b-hf \
    --host localhost \
    --port 8002 \
    --backend pytorch \
    --tp_size 2 \
    --extra_llm_api_options ./extra-llm-api-config.yml \
    &> Logs/log_pd_tp2 &
```

## Testing the Setup

### Basic Functionality Test

Test the setup with a multimodal chat completion request:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json"  \
    -d '{
        "model": "llava-hf/llava-v1.6-mistral-7b-hf",
        "messages":[{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the natural environment in the image."
                },
                {
                    "type":"image_url",
                    "image_url": {
                        "url": "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png"
                    }
                }
            ]
        }],
        "max_tokens": 64,
        "temperature": 0
    }'
```

### Performance Testing

Use the provided performance testing script for load testing (assuming you've already set up the multimodal disaggregated server):

#### Prerequisites
```bash
pip install genai_perf
```

#### Concurrency Testing
```bash
./test_client_disag_mm.sh --concurrency 1 --port 8000
```

#### Request Rate Testing
```bash
./test_client_disag_mm.sh --request-rate 10 --port 8000
```


## Roadmap & Future Improvements

- **Model Support**: Add support for more multimodal models beyond LLaVA-Next
- **Communication**: NIXL integration for transferring multimodal embeddings between servers
- **Scalability**: Enable support for multiple LLM servers and multimodal servers with a routing manager
- **Parallelism**: Enable data parallelism (DP) in multimodal server
- **Configuration**: Test/verify/enable major parallel configurations in LLM decoder server
- **Optimization**: Performance optimization and tuning
