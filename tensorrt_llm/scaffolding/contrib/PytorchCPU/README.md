# PyTorch CPU Worker for TensorRT-LLM Scaffolding

## Overview

The PyTorch CPU Worker enables CPU-based inference for TensorRT-LLM scaffolding, addressing the need for development and testing environments without GPU requirements. This implementation allows community contributors to work on scaffolding-based applications using only CPU resources.

## Motivation

Many community contributors find it difficult to access GPU-based development environments and prefer CPU-based scaffolding workers to complete development tasks. This worker provides:

- **Accessibility**: Enables development without expensive GPU hardware
- **Testing**: Allows testing of scaffolding logic on local machines
- **Prototyping**: Facilitates rapid prototyping of scaffolding applications
- **CI/CD**: Enables automated testing in CPU-only environments

## Features

- **CPU Inference**: Runs entirely on CPU using PyTorch backend
- **Scaffolding Integration**: Seamlessly integrates with TensorRT-LLM scaffolding system
- **Error Handling**: Comprehensive error handling and logging
- **Flexible Configuration**: Configurable batch sizes, token limits, and sampling parameters
- **Async Support**: Supports both synchronous and asynchronous generation


### Command Line Example

Run the provided example script:

```bash
python examples/scaffolding/contrib/PytorchCPU/pytorch_worker_run.py \
    --model_dir microsoft/DialoGPT-medium \
    --temperature 0.7 \
    --max_batch_size 4 \
    --verbose
```

## Configuration Options

### PytorchWorker Parameters

- `model_path` (str): Path to model directory or Hugging Face model name
- `max_batch_size` (int): Maximum batch size for inference (default: 32)
- `max_num_tokens` (int): Maximum number of tokens to process (default: 4096)
- `trust_remote_code` (bool): Whether to trust remote code (default: False)
- `device` (str): Device to use for inference (default: "cpu")

### Sampling Parameters

The worker supports all standard sampling parameters:

- `temperature`: Controls randomness (0.0 = deterministic, 1.0 = random)
- `top_p`: Nucleus sampling threshold
- `top_k`: Top-k sampling limit
- `max_tokens`: Maximum tokens to generate
- `frequency_penalty`: Penalty for repeated tokens
- `presence_penalty`: Penalty for new topics
- `stop`: Stop sequences

## Architecture

The PyTorch CPU Worker is built on top of:

1. **TensorRT-LLM Scaffolding**: Provides the worker interface and task management
2. **PyTorch Backend**: Uses TensorRT-LLM's PyTorch backend for model execution
3. **CPU Optimization**: Configured specifically for CPU inference with:
   - Disabled CUDA graphs
   - Torch attention backend
   - Disabled torch.compile
   - Single-process execution

## Running the Example

To test that the PyTorch CPU Worker works, run the example script in Docker:

```bash

# Run the example with a small model
python examples/scaffolding/contrib/PytorchCPU/pytorch_worker_run.py \
    --model_dir microsoft/DialoGPT-small \
    --max_batch_size 2 \
    --max_num_tokens 512 \
    --temperature 0.7 \
    --verbose
```

### Alternative Models and Options

```bash
# Run with async mode
python examples/scaffolding/contrib/PytorchCPU/pytorch_worker_run.py \
    --model_dir microsoft/DialoGPT-small \
    --run_async \
    --verbose

# Run with different parameters
python examples/scaffolding/contrib/PytorchCPU/pytorch_worker_run.py \
    --model_dir microsoft/DialoGPT-medium \
    --max_batch_size 4 \
    --max_num_tokens 1024 \
    --temperature 0.9 \
    --top_p 0.95 \
    --verbose
```

## File Structure

```
tensorrt_llm/scaffolding/contrib/PytorchCPU/
├── __init__.py                 # Package initialization and exports
├── pytorch_worker.py           # Main worker implementation
└── README.md                   # This documentation

examples/scaffolding/contrib/PytorchCPU/
└── pytorch_worker_run.py       # Example usage script
```
