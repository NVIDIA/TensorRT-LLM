---
name: TensorRT-LLM - NVIDIA TensorRT for Large Language Models
description: TensorRT-LLM is an open-source library from NVIDIA that optimizes Large Language Model inference on NVIDIA GPUs. It provides an easy-to-use Python API with state-of-the-art performance optimizations including custom kernels, quantization, paged KV caching, and multi-GPU support.
---

## Quick Start

```bash
# Pull the official Docker container
docker pull nvcr.io/nvidia/tritonserver:24.09-trtllm-python-py3

# Run with GPU support
docker run --gpus all -it nvcr.io/nvidia/tritonserver:24.09-trtllm-python-py3

# Inside container, install TensorRT-LLM
pip install tensorrt-llm

# Run a simple example with Llama
python examples/llama/run.py
```

## When to Use This Skill

Use TensorRT-LLM when you need to:
- Deploy Large Language Models (LLMs) with maximum performance on NVIDIA GPUs
- Optimize inference latency and throughput for production LLM serving
- Implement multi-GPU or multi-node LLM inference
- Use quantization (FP8, INT4, INT8) to reduce memory footprint
- Serve LLMs at scale with high throughput requirements
- Build custom LLM architectures with PyTorch-based optimizations
- Integrate LLM inference with Triton Inference Server
- Achieve state-of-the-art tokens/second performance

## Prerequisites

**Platform**: Linux (x86_64, aarch64)

**Required Dependencies**:
- NVIDIA GPU with Compute Capability 8.0+ (Ampere, Ada, Hopper, Blackwell)
- CUDA 13.1.0 or later
- Python 3.10 or 3.12
- PyTorch 2.9.1+
- Docker (recommended for easiest setup)

**Optional Dependencies**:
- NVIDIA Triton Inference Server (for production deployment)
- Multi-GPU setup (for tensor/pipeline parallelism)
- InfiniBand/NVLink (for multi-node deployments)
- Grace Hopper architecture (for specialized deployments)

## Compatibility

| TensorRT-LLM Version | CUDA Version | Python Version | PyTorch Version | GPU Architecture |
|---------------------|--------------|----------------|-----------------|------------------|
| 1.3.0rc2           | 13.1.0+      | 3.10, 3.12     | 2.9.1+         | Ampere, Ada, Hopper, Blackwell |
| 1.0+               | 12.4+        | 3.10, 3.12     | 2.x            | Ampere, Ada, Hopper |

**Supported Models** (100+ models):
- Meta Llama (3, 3.1, 3.3, 4)
- GPT variants (GPT-2, GPT-J, GPT-NeoX, GPT-OSS)
- DeepSeek (R1, V2, V3)
- Mistral, Mixtral
- Falcon (7B, 40B, 180B)
- BLOOM, OPT
- LG EXAONE
- And many more...

## Installation

### Docker Installation (Recommended)

```bash
# Pull the latest TensorRT-LLM container
docker pull nvcr.io/nvidia/tritonserver:24.09-trtllm-python-py3

# Run with GPU access
docker run --gpus all \
           --ipc=host \
           --ulimit memlock=-1 \
           --ulimit stack=67108864 \
           -v /path/to/models:/models \
           -it nvcr.io/nvidia/tritonserver:24.09-trtllm-python-py3

# Verify installation
python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
```

### PyPI Installation

```bash
# Create virtual environment
python3.10 -m venv trtllm-env
source trtllm-env/bin/activate

# Install TensorRT-LLM
pip install tensorrt-llm

# Install with specific CUDA version
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
```

### From Source (Advanced)

```bash
# Clone repository
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM

# Install dependencies
pip install -r requirements.txt

# Build and install
python setup.py develop

# Verify
python -c "import tensorrt_llm; print('Build successful')"
```

### JetPack Installation (Jetson Devices)

```bash
# For Jetson AGX Orin with JetPack 6.1
sudo apt-get install tensorrt-llm

# Configure for edge deployment
trtllm-build --checkpoint_dir=/path/to/model --output_dir=/engines
```

## Configuration

### Environment Variables

```bash
# Core configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3      # Select GPUs
export TRTLLM_LOG_LEVEL=INFO             # DEBUG, INFO, WARN, ERROR

# Performance tuning
export TRTLLM_ENABLE_KVCACHE_REUSE=1     # Enable KV cache reuse
export TRTLLM_BATCH_SCHEDULER=IFB        # Inflight batching
export TRTLLM_MAX_BATCH_SIZE=256         # Maximum batch size
export TRTLLM_MAX_INPUT_LEN=2048         # Max input sequence length
export TRTLLM_MAX_OUTPUT_LEN=1024        # Max output tokens

# Memory optimization
export TRTLLM_PAGED_KV_CACHE=1           # Enable paged KV cache
export TRTLLM_KV_CACHE_FREE_GPU_MEM=0.9  # Use 90% GPU memory for KV cache

# Multi-GPU settings
export TRTLLM_WORLD_SIZE=4               # Number of GPUs
export TRTLLM_RANK=0                     # Current GPU rank
export NCCL_P2P_DISABLE=0                # Enable P2P for NVLink

# Quantization
export TRTLLM_ENABLE_FP8=1               # Enable FP8 quantization
export TRTLLM_QUANT_MODE=int4_awq        # Quantization mode
```

### Model Configuration File

```python
# config.json example for Llama-3.1-70B
{
    "architecture": "LlamaForCausalLM",
    "dtype": "float16",
    "num_hidden_layers": 80,
    "num_attention_heads": 64,
    "hidden_size": 8192,
    "vocab_size": 128256,
    "max_position_embeddings": 131072,
    "max_batch_size": 256,
    "max_input_len": 2048,
    "max_output_len": 1024,
    "builder_opt": {
        "enable_kv_cache_reuse": true,
        "paged_kv_cache": true,
        "remove_input_padding": true,
        "use_custom_all_reduce": true
    },
    "plugin_config": {
        "gpt_attention_plugin": "float16",
        "gemm_plugin": "float16",
        "paged_kv_cache": true,
        "inflight_batching": true
    },
    "quantization": {
        "quant_algo": "FP8",
        "kv_cache_quant_algo": "FP8"
    }
}
```

### Build Configuration

```bash
# Build optimized engine for Llama model
trtllm-build \
    --checkpoint_dir=/models/llama-3.1-70b \
    --output_dir=/engines/llama-3.1-70b \
    --gemm_plugin=float16 \
    --gpt_attention_plugin=float16 \
    --max_batch_size=256 \
    --max_input_len=2048 \
    --max_output_len=1024 \
    --use_paged_context_fmha=enable \
    --remove_input_padding=enable \
    --enable_context_fmha=enable \
    --use_custom_all_reduce=enable \
    --workers=4
```

## Usage Patterns

### Basic Inference with Python API

```python
import tensorrt_llm
from tensorrt_llm.hlapi import LLM, SamplingParams

# Initialize model
llm = LLM(
    model="/models/llama-3.1-70b",
    tensor_parallel_size=4,
    dtype="float16"
)

# Set sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512
)

# Run inference
prompts = [
    "Explain quantum computing in simple terms:",
    "Write a Python function to sort a list:"
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.text}")
    print(f"Tokens: {output.num_tokens}")
```

### Multi-GPU Inference

```python
from tensorrt_llm import LLM

# Initialize with tensor parallelism across 8 GPUs
llm = LLM(
    model="/models/llama-3.1-405b",
    tensor_parallel_size=8,           # Split across 8 GPUs
    pipeline_parallel_size=1,
    dtype="float16",
    kv_cache_config={
        "enable_block_reuse": True,
        "max_tokens": 131072
    }
)

# Inference automatically distributed across GPUs
outputs = llm.generate(["Your prompt here"])
```

### Quantization

```python
from tensorrt_llm import LLM
from tensorrt_llm.quantization import QuantAlgo

# Load model with INT4-AWQ quantization
llm = LLM(
    model="/models/llama-3.1-70b",
    tensor_parallel_size=4,
    quant_config={
        "quant_algo": QuantAlgo.INT4_AWQ,
        "group_size": 128,
        "has_zero_point": False
    }
)

# Model now uses ~4x less memory
outputs = llm.generate(["Test prompt"])
```

### Batch Inference with Inflight Batching

```python
from tensorrt_llm import LLM, BatchConfig

# Configure inflight batching for high throughput
llm = LLM(
    model="/models/llama-3.1-70b",
    tensor_parallel_size=4,
    batch_config=BatchConfig(
        max_batch_size=256,
        max_num_tokens=8192,
        scheduler="IFB"  # Inflight batching
    )
)

# Submit multiple requests - automatically batched
requests = [f"Prompt {i}" for i in range(100)]
outputs = llm.generate(requests)
```

### Streaming Generation

```python
from tensorrt_llm import LLM

llm = LLM(model="/models/llama-3.1-70b", tensor_parallel_size=4)

# Stream tokens as they're generated
for output in llm.generate_stream("Tell me a story:", max_tokens=1024):
    print(output.text, end="", flush=True)
```

### Custom Engine Building

```python
from tensorrt_llm import BuildConfig, build

# Build custom engine with specific optimizations
config = BuildConfig(
    max_batch_size=128,
    max_input_len=4096,
    max_output_len=2048,
    max_beam_width=4,
    builder_opt={
        "precision": "float16",
        "enable_fmha": True,
        "use_paged_kv_cache": True,
        "use_gpt_attention_plugin": True,
        "remove_input_padding": True,
        "enable_context_fmha": True
    }
)

# Build engine from checkpoint
engine = build(
    checkpoint_dir="/models/llama-3.1-70b/checkpoint",
    build_config=config,
    output_dir="/engines/llama-optimized",
    workers=8  # Parallel engine building
)
```

## Key Features

- **State-of-the-Art Performance**: Optimized kernels for maximum throughput
- **Quantization Support**: FP8, FP4, INT4 AWQ, INT8 SmoothQuant
- **Paged KV Cache**: Efficient memory management for long sequences
- **Inflight Batching**: Dynamic batching for high throughput
- **Multi-GPU Support**: Tensor and pipeline parallelism
- **Speculative Decoding**: Up to 3.6x throughput improvement
- **100+ Model Support**: Pre-optimized profiles for popular LLMs
- **PyTorch Integration**: Native PyTorch API for easy customization
- **Production Ready**: Triton Inference Server integration

## Performance Optimization

### Best Practices

1. **Enable All Optimizations**
   ```bash
   trtllm-build \
       --gemm_plugin=float16 \
       --gpt_attention_plugin=float16 \
       --use_paged_context_fmha=enable \
       --remove_input_padding=enable \
       --enable_context_fmha=enable \
       --use_custom_all_reduce=enable
   ```

2. **Optimize Batch Size**
   ```python
   # Find optimal batch size for your workload
   for batch_size in [32, 64, 128, 256]:
       llm = LLM(model="...", max_batch_size=batch_size)
       # Benchmark throughput
   ```

3. **Use Appropriate Quantization**
   ```python
   # For 80GB GPUs (A100/H100)
   quant_algo = "FP8"  # Best quality/speed tradeoff

   # For 40GB GPUs (A100)
   quant_algo = "INT4_AWQ"  # Good quality, 4x memory reduction

   # For 24GB GPUs (3090/4090)
   quant_algo = "INT4_AWQ"  # Necessary for large models
   ```

4. **Tune KV Cache**
   ```python
   kv_cache_config = {
       "enable_block_reuse": True,
       "max_tokens": 131072,  # Match max sequence length
       "free_gpu_mem_fraction": 0.9  # Use 90% GPU memory
   }
   ```

5. **Enable Speculative Decoding**
   ```python
   llm = LLM(
       model="/models/llama-3.1-70b",
       draft_model="/models/llama-3.1-8b",  # Small draft model
       speculative_decoding=True,
       num_speculative_tokens=5
   )
   # Can achieve 2-3x speedup
   ```

6. **Use Context FMHA for Long Contexts**
   ```python
   # For inputs > 2048 tokens
   llm = LLM(
       model="...",
       enable_context_fmha=True,
       max_input_len=8192
   )
   ```

### Expected Performance

| Model | GPUs | Quantization | Batch Size | Throughput | Notes |
|-------|------|--------------|------------|------------|-------|
| Llama-3-8B | 1x H100 | FP16 | 128 | 24,000 tok/s | Single GPU |
| Llama-3.1-70B | 4x H100 | FP8 | 256 | 12,000 tok/s | Tensor parallel |
| Llama-3.1-405B | 8x H100 | FP8 | 256 | 400 tok/s/node | Multi-node |
| Llama-4 | 8x B200 | FP8 | 512 | 40,000 tok/s | Blackwell |
| DeepSeek-R1 | 8x B200 | FP8 | 256 | 35,000 tok/s | Optimized |

**Per-User Latency** (streaming):
- Llama-3.1-405B: ~37 tokens/second per user
- Llama-3.1-70B: ~120 tokens/second per user
- Llama-3-8B: ~300 tokens/second per user

## Use Cases

1. **Production LLM Serving**: Deploy LLMs with maximum throughput and minimum latency
2. **Chatbot Applications**: Real-time conversational AI with streaming responses
3. **Code Generation**: Fast code completion and generation services
4. **Content Creation**: High-throughput text generation for articles, summaries, etc.
5. **Multi-Modal Applications**: Combine with vision models for VLM inference
6. **Edge Deployment**: Run LLMs on Jetson devices with optimized performance
7. **Research Experiments**: Fast iteration on custom LLM architectures
8. **Enterprise AI**: Integration with Triton for scalable microservices

## Examples

### Example 1: Complete Inference Pipeline

```python
#!/usr/bin/env python3
"""Complete TensorRT-LLM inference example with error handling"""

import tensorrt_llm
from tensorrt_llm import LLM, SamplingParams
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialize model
        logger.info("Loading model...")
        llm = LLM(
            model="/models/llama-3.1-70b",
            tensor_parallel_size=4,
            dtype="float16",
            kv_cache_config={
                "enable_block_reuse": True,
                "max_tokens": 8192
            },
            enable_streaming=False
        )
        logger.info(f"Model loaded: {llm.model_config}")

        # Configure sampling
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_tokens=512,
            repetition_penalty=1.1,
            stop_sequences=["</s>", "\n\n"]
        )

        # Prepare prompts
        prompts = [
            "Explain the theory of relativity:",
            "Write a Python function to compute Fibonacci numbers:",
            "What are the key differences between supervised and unsupervised learning?"
        ]

        # Run inference
        logger.info(f"Running inference on {len(prompts)} prompts...")
        outputs = llm.generate(prompts, sampling_params)

        # Process results
        for idx, output in enumerate(outputs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Prompt {idx + 1}: {output.prompt[:50]}...")
            logger.info(f"Generated text:\n{output.text}")
            logger.info(f"Tokens generated: {output.num_tokens}")
            logger.info(f"Generation time: {output.generation_time:.2f}s")
            logger.info(f"Throughput: {output.num_tokens / output.generation_time:.2f} tok/s")

        return 0

    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        return 1
    finally:
        logger.info("Cleaning up...")
        # Cleanup handled automatically

if __name__ == "__main__":
    sys.exit(main())
```

### Example 2: Multi-GPU with Pipeline Parallelism

```python
#!/usr/bin/env python3
"""Multi-GPU inference with pipeline parallelism"""

from tensorrt_llm import LLM, ParallelConfig
import torch.distributed as dist

def setup_distributed():
    """Initialize distributed environment"""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

def main():
    rank, world_size = setup_distributed()

    # Configure parallelism
    parallel_config = ParallelConfig(
        tensor_parallel_size=4,      # Split model across 4 GPUs per pipeline stage
        pipeline_parallel_size=2,    # 2 pipeline stages
        world_size=world_size
    )

    # Initialize model (each process gets its own shard)
    llm = LLM(
        model="/models/llama-3.1-405b",
        parallel_config=parallel_config,
        dtype="float16"
    )

    # Only rank 0 submits prompts
    if rank == 0:
        prompts = ["Your prompt here"] * 10
        outputs = llm.generate(prompts)

        for output in outputs:
            print(f"Generated: {output.text}")

    # All ranks participate in computation
    dist.barrier()

if __name__ == "__main__":
    main()
```

### Example 3: Benchmarking with trtllm-bench

```bash
#!/bin/bash
# benchmark_trtllm.sh - Comprehensive benchmarking

set -e

MODEL_PATH="/models/llama-3.1-70b"
ENGINE_PATH="/engines/llama-3.1-70b"

echo "=== TensorRT-LLM Benchmark ==="

# Build engine if not exists
if [ ! -d "$ENGINE_PATH" ]; then
    echo "Building engine..."
    trtllm-build \
        --checkpoint_dir=$MODEL_PATH \
        --output_dir=$ENGINE_PATH \
        --gemm_plugin=float16 \
        --gpt_attention_plugin=float16 \
        --max_batch_size=256 \
        --max_input_len=2048 \
        --max_output_len=512
fi

# Run throughput benchmark
echo "Running throughput benchmark..."
trtllm-bench \
    --engine_dir=$ENGINE_PATH \
    --dataset=/data/prompts.txt \
    --batch_size=128 \
    --input_len=1024 \
    --output_len=128 \
    --num_runs=100 \
    --warm_up=10 \
    --output_csv=results_throughput.csv

# Run latency benchmark
echo "Running latency benchmark..."
trtllm-bench \
    --engine_dir=$ENGINE_PATH \
    --dataset=/data/prompts.txt \
    --batch_size=1 \
    --input_len=512 \
    --output_len=512 \
    --num_runs=100 \
    --measure_latency \
    --output_csv=results_latency.csv

# Run concurrency benchmark
echo "Running concurrency benchmark..."
for concurrency in 1 4 8 16 32; do
    trtllm-bench \
        --engine_dir=$ENGINE_PATH \
        --dataset=/data/prompts.txt \
        --concurrency=$concurrency \
        --max_requests=1000 \
        --output_csv=results_concurrency_${concurrency}.csv
done

# Analyze results
python3 << 'EOF'
import pandas as pd
import glob

print("\n=== Benchmark Results Summary ===\n")

# Throughput results
df = pd.read_csv("results_throughput.csv")
print(f"Average Throughput: {df['tokens_per_sec'].mean():.2f} tokens/s")
print(f"Peak Throughput: {df['tokens_per_sec'].max():.2f} tokens/s")

# Latency results
df = pd.read_csv("results_latency.csv")
print(f"\nLatency Statistics:")
print(f"  P50: {df['latency_ms'].quantile(0.5):.2f} ms")
print(f"  P95: {df['latency_ms'].quantile(0.95):.2f} ms")
print(f"  P99: {df['latency_ms'].quantile(0.99):.2f} ms")

# Concurrency results
print(f"\nConcurrency vs Throughput:")
for file in sorted(glob.glob("results_concurrency_*.csv")):
    concurrency = file.split("_")[-1].replace(".csv", "")
    df = pd.read_csv(file)
    throughput = df['tokens_per_sec'].mean()
    print(f"  Concurrency {concurrency}: {throughput:.2f} tokens/s")
EOF

echo "Benchmark complete!"
```

### Example 4: Quantization Workflow

```python
#!/usr/bin/env python3
"""Quantize a model to INT4-AWQ"""

from tensorrt_llm.quantization import quantize_and_export

# Step 1: Quantize model
print("Quantizing model to INT4-AWQ...")
quantize_and_export(
    model_dir="/models/llama-3.1-70b",
    output_dir="/models/llama-3.1-70b-int4-awq",
    quant_config={
        "quant_algo": "INT4_AWQ",
        "group_size": 128,
        "calib_dataset": "/data/calibration.json",
        "calib_size": 512
    }
)

# Step 2: Build quantized engine
from tensorrt_llm import build, BuildConfig

config = BuildConfig(
    max_batch_size=256,
    max_input_len=2048,
    max_output_len=512,
)

engine = build(
    checkpoint_dir="/models/llama-3.1-70b-int4-awq",
    build_config=config,
    output_dir="/engines/llama-3.1-70b-int4-awq"
)

# Step 3: Run inference
from tensorrt_llm import LLM

llm = LLM(engine_dir="/engines/llama-3.1-70b-int4-awq")
outputs = llm.generate(["Test prompt"])

print(f"Quantized model output: {outputs[0].text}")
print(f"Memory usage: {llm.get_memory_usage():.2f} GB")
```

### Example 5: Triton Integration

```python
# model_repository/llama/1/model.py
"""Triton Python backend for TensorRT-LLM"""

import triton_python_backend_utils as pb_utils
import tensorrt_llm
from tensorrt_llm import LLM
import json

class TritonPythonModel:
    def initialize(self, args):
        """Initialize TensorRT-LLM model"""
        self.model_config = json.loads(args['model_config'])

        # Initialize LLM
        self.llm = LLM(
            engine_dir="/engines/llama-3.1-70b",
            tensor_parallel_size=4
        )

        self.logger = pb_utils.Logger
        self.logger.log_info("TensorRT-LLM model initialized")

    def execute(self, requests):
        """Execute inference requests"""
        responses = []

        for request in requests:
            # Get input text
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            prompt = input_tensor.as_numpy()[0].decode('utf-8')

            # Get parameters
            max_tokens = pb_utils.get_input_tensor_by_name(request, "MAX_TOKENS")
            max_tokens = int(max_tokens.as_numpy()[0]) if max_tokens else 512

            temperature = pb_utils.get_input_tensor_by_name(request, "TEMPERATURE")
            temperature = float(temperature.as_numpy()[0]) if temperature else 0.8

            # Run inference
            outputs = self.llm.generate(
                [prompt],
                sampling_params={
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            )

            # Create response
            output_text = outputs[0].text
            output_tensor = pb_utils.Tensor(
                "OUTPUT",
                output_text.encode('utf-8')
            )

            response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(response)

        return responses

    def finalize(self):
        """Cleanup"""
        self.logger.log_info("Cleaning up TensorRT-LLM model")
```

**Triton config.pbtxt:**
```protobuf
name: "llama"
backend: "python"
max_batch_size: 256

input [
  {
    name: "INPUT"
    data_type: TYPE_STRING
    dims: [ 1 ]
  },
  {
    name: "MAX_TOKENS"
    data_type: TYPE_INT32
    dims: [ 1 ]
    optional: true
  },
  {
    name: "TEMPERATURE"
    data_type: TYPE_FP32
    dims: [ 1 ]
    optional: true
  }
]

output [
  {
    name: "OUTPUT"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0, 1, 2, 3 ]
  }
]
```

### Example 6: Streaming with FastAPI

```python
#!/usr/bin/env python3
"""FastAPI server with streaming TensorRT-LLM inference"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from tensorrt_llm import LLM, SamplingParams
import asyncio
from typing import AsyncIterator

app = FastAPI()

# Initialize model at startup
llm = None

@app.on_event("startup")
async def startup_event():
    global llm
    llm = LLM(
        model="/models/llama-3.1-70b",
        tensor_parallel_size=4,
        dtype="float16"
    )

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    stream: bool = False

@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate text with optional streaming"""

    if not llm:
        raise HTTPException(status_code=503, detail="Model not loaded")

    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens
    )

    if request.stream:
        # Streaming response
        async def generate_stream() -> AsyncIterator[str]:
            for output in llm.generate_stream(request.prompt, sampling_params):
                yield f"data: {output.text}\n\n"
                await asyncio.sleep(0)  # Allow other tasks to run

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming response
        outputs = llm.generate([request.prompt], sampling_params)
        return {
            "text": outputs[0].text,
            "num_tokens": outputs[0].num_tokens,
            "finish_reason": outputs[0].finish_reason
        }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": llm is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Usage:**
```bash
# Start server
python fastapi_server.py

# Non-streaming request
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_tokens": 100}'

# Streaming request
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me a story", "max_tokens": 500, "stream": true}'
```

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

**Problem**: Model fails to load with "CUDA out of memory" error.

**Solution**:
```bash
# Check GPU memory
nvidia-smi

# Option 1: Use quantization
trtllm-build --quant_mode=int4_awq ...

# Option 2: Reduce max_batch_size
trtllm-build --max_batch_size=64 ...  # Instead of 256

# Option 3: Use tensor parallelism
python run.py --tensor_parallel_size=2  # Split across 2 GPUs

# Option 4: Reduce KV cache size
export TRTLLM_KV_CACHE_FREE_GPU_MEM=0.7  # Use only 70% for cache
```

#### 2. Build Failures

**Problem**: Engine build fails with "Plugin not found" or compilation errors.

**Solution**:
```bash
# Verify CUDA installation
nvcc --version  # Should match required version

# Check TensorRT-LLM installation
python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"

# Rebuild with verbose logging
trtllm-build --log_level=verbose ...

# Clean cache and rebuild
rm -rf ~/.cache/tensorrt_llm
trtllm-build ...
```

#### 3. Low Throughput

**Problem**: Achieving much lower tokens/second than expected.

**Solution**:
```bash
# Enable all optimizations
trtllm-build \
    --gemm_plugin=float16 \
    --gpt_attention_plugin=float16 \
    --use_paged_context_fmha=enable \
    --remove_input_padding=enable \
    --enable_context_fmha=enable

# Check if using correct GPU
nvidia-smi  # Verify GPU utilization is high

# Increase batch size
export TRTLLM_MAX_BATCH_SIZE=256

# Enable inflight batching
export TRTLLM_BATCH_SCHEDULER=IFB
```

#### 4. Multi-GPU Communication Issues

**Problem**: Multi-GPU inference fails with NCCL errors.

**Solution**:
```bash
# Enable NCCL debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# For NVLink systems
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

# For InfiniBand systems
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0

# Test NCCL connectivity
nccl-test --nthreads=4 --ngpus=4
```

#### 5. Incorrect/Garbled Output

**Problem**: Model generates incorrect or garbled text.

**Solution**:
```python
# Check tokenizer configuration
llm = LLM(model="...", tokenizer="/path/to/tokenizer")

# Verify model checkpoint integrity
# Re-download or re-convert model

# Try different precision
trtllm-build --dtype=float32 ...  # Instead of float16

# Check sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,  # Not too high (> 1.5)
    top_p=0.9,
    repetition_penalty=1.1  # Prevent repetition
)
```

#### 6. Engine Build Takes Too Long

**Problem**: Building engines is extremely slow.

**Solution**:
```bash
# Use multiple workers
trtllm-build --workers=8 ...

# Enable fast build (less optimization)
trtllm-build --builder_opt=0 ...

# Use pre-built engines if available
# Download from NGC catalog or model zoo

# Build on high-performance machine, deploy elsewhere
# Engines are portable across same GPU architecture
```

#### 7. Incompatible Model Format

**Problem**: Cannot load model checkpoint.

**Solution**:
```bash
# Convert from Hugging Face format
python convert_checkpoint.py \
    --model_dir=/models/huggingface/llama-3.1-70b \
    --output_dir=/models/trtllm/llama-3.1-70b \
    --dtype=float16 \
    --tp_size=4

# Verify checkpoint format
python -c "from tensorrt_llm.models import LLaMAForCausalLM; \
           LLaMAForCausalLM.from_checkpoint('/models/trtllm/llama-3.1-70b')"
```

### Getting Help

1. **Enable verbose logging**:
   ```bash
   export TRTLLM_LOG_LEVEL=DEBUG
   export CUDA_LAUNCH_BLOCKING=1
   ```

2. **Check GitHub issues**: https://github.com/NVIDIA/TensorRT-LLM/issues

3. **Report bugs with**:
   - TensorRT-LLM version
   - CUDA version (`nvcc --version`)
   - GPU model (`nvidia-smi`)
   - Full error logs and traceback
   - Model name and configuration

4. **Community resources**:
   - GitHub Discussions
   - NVIDIA Developer Forums
   - WeChat Discussion Group (for Chinese users)

## Advanced Topics

### Custom Model Architectures

```python
from tensorrt_llm import Module
from tensorrt_llm.layers import Attention, MLP

class CustomLLM(Module):
    def __init__(self, config):
        super().__init__()
        self.layers = [
            CustomLayer(config) for _ in range(config.num_layers)
        ]

    def forward(self, input_ids, **kwargs):
        hidden_states = self.embed(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.lm_head(hidden_states)

class CustomLayer(Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.mlp = MLP(config)

    def forward(self, hidden_states):
        # Custom attention + MLP logic
        attn_out = self.attention(hidden_states)
        mlp_out = self.mlp(attn_out)
        return mlp_out
```

### Expert Parallelism for MoE Models

```python
from tensorrt_llm import LLM, ParallelConfig

# Configure expert parallelism for Mixtral
llm = LLM(
    model="/models/mixtral-8x7b",
    parallel_config=ParallelConfig(
        tensor_parallel_size=2,
        expert_parallel_size=4,  # Distribute experts across 4 GPUs
    )
)
```

### Custom Sampling Strategies

```python
from tensorrt_llm import SamplingParams, LogitsProcessor

class CustomLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        # Custom logits manipulation
        # E.g., boost certain token probabilities
        scores[:, special_token_id] += 2.0
        return scores

sampling_params = SamplingParams(
    temperature=0.8,
    logits_processors=[CustomLogitsProcessor()]
)
```

### Profiling and Optimization

```python
from tensorrt_llm.profiler import profile

# Profile inference
with profile(output_dir="/tmp/trtllm_profile"):
    outputs = llm.generate(prompts)

# Analyze profile
# Use NVIDIA Nsight Systems to visualize
# nsys profile -o profile.qdrep python your_script.py
```

### Mixed Precision Training

```python
# Train custom adapters with mixed precision
from tensorrt_llm.trainer import Trainer

trainer = Trainer(
    model=llm,
    precision="fp16",
    gradient_checkpointing=True,
    optimizer="adamw"
)

trainer.train(dataset, epochs=3)
```

## Security Considerations

### Model Validation

```python
# Verify model checksum before loading
import hashlib

def verify_checkpoint(path, expected_hash):
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        sha256.update(f.read())
    assert sha256.hexdigest() == expected_hash, "Checksum mismatch!"

verify_checkpoint("/models/checkpoint.bin", "abc123...")
```

### Input Sanitization

```python
def sanitize_input(prompt: str, max_length: int = 8192) -> str:
    """Sanitize user input"""
    # Remove control characters
    prompt = ''.join(c for c in prompt if c.isprintable() or c.isspace())

    # Truncate to max length
    prompt = prompt[:max_length]

    # Check for injection attempts
    if any(pattern in prompt.lower() for pattern in ["ignore previous", "system:", "<|im_start|>"]):
        raise ValueError("Potentially unsafe input detected")

    return prompt

# Use in production
safe_prompt = sanitize_input(user_input)
outputs = llm.generate([safe_prompt])
```

### Rate Limiting

```python
from fastapi import FastAPI, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter

@app.post("/generate")
@limiter.limit("10/minute")  # 10 requests per minute per IP
async def generate(request: GenerateRequest):
    # ... inference logic
    pass
```

## Resources

- **Repository**: https://github.com/NVIDIA/TensorRT-LLM
- **Documentation**: https://nvidia.github.io/TensorRT-LLM/
- **Installation Guide**: https://nvidia.github.io/TensorRT-LLM/installation/
- **Model Zoo**: Pre-optimized checkpoints for popular models
- **NGC Catalog**: https://catalog.ngc.nvidia.com/
- **Issue Tracker**: https://github.com/NVIDIA/TensorRT-LLM/issues
- **Developer Forums**: https://forums.developer.nvidia.com/

## Notes

### Platform Support
- **Linux**: x86_64, aarch64 (full support)
- **Windows**: Limited support (Docker recommended)
- **Jetson**: AGX Orin with JetPack 6.1+

### GPU Requirements
- **Minimum**: Compute Capability 8.0 (Ampere: A100, A30, A10)
- **Recommended**: Compute Capability 9.0+ (Hopper: H100, H200)
- **Optimal**: Blackwell architecture (B200, B100)
- **Memory**: 24GB minimum, 80GB+ recommended for large models

### Performance Characteristics
- Llama-3-8B: Up to 24,000 tokens/second on H100
- Llama-3.1-70B: 12,000+ tokens/second with 4x H100
- Llama-4: 40,000+ tokens/second on 8x B200
- Latency: Sub-100ms TTFT (Time To First Token) for most models

### Production Readiness
- Battle-tested in NVIDIA's production services
- Supports 100+ model architectures
- Integrated with Triton for enterprise deployment
- Regular updates and 3-month deprecation policy
- Semantic versioning for stable APIs

### Known Limitations
- Windows support is experimental
- Some models require conversion from Hugging Face format
- Multi-node requires NCCL-compatible network (InfiniBand/RoCE)
- Very long contexts (> 128k) may require specialized tuning

### Version Compatibility
- Follows semantic versioning (MAJOR.MINOR.PATCH)
- Deprecated APIs have 3-month migration window
- Engines built with one version may not work with different versions
- Rebuild engines when upgrading TensorRT-LLM

## Related Technologies

- **TensorRT**: NVIDIA's inference optimization SDK
- **Triton Inference Server**: Production inference serving platform
- **NVIDIA NeMo**: Framework for training conversational AI models
- **cuBLAS/cuDNN**: CUDA-accelerated linear algebra libraries
- **NCCL**: Multi-GPU communication library
- **Flash Attention**: Memory-efficient attention implementation
