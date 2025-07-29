# GLM4 MoE with TensorRT-LLM

This document explains how to build and run GLM4 MoE (Mixture of Experts) models using TensorRT-LLM with the PyTorch backend.

## Overview

GLM4 MoE is a Mixture of Experts variant of the GLM4 model architecture. This implementation extends TensorRT-LLM's existing GLM support to include MoE layers, similar to how Llama4 MoE is implemented.

## Features

- **MoE Support**: Full support for Mixture of Experts layers with configurable number of experts and top-k selection
- **TensorRT-LLM Integration**: Seamless integration with TensorRT-LLM's PyTorch backend
- **Parallelism**: Support for tensor parallelism, pipeline parallelism, and expert parallelism
- **Quantization**: Support for various quantization methods (FP8, INT8, etc.)
- **Performance**: Optimized MoE kernels for high-performance inference

## Architecture

The GLM4 MoE implementation includes:

- **GLM4MoEDecoderLayer**: Decoder layer that alternates between regular MLP and MoE layers
- **GLM4MoEModel**: The main transformer model with MoE support
- **GLM4MoEForCausalLM**: Complete language model with MoE layers

### MoE Configuration

The MoE layers are configured with the following parameters:

- `num_experts`: Number of experts (default: 8)
- `top_k`: Number of experts to select per token (default: 2)
- `interleave_moe_layer_step`: How often to use MoE layers (default: 2, meaning every 2nd layer)
- `normalization_mode`: How to normalize expert outputs (default: RENORMALIZE)

## Usage

### 1. Convert HuggingFace Model

First, convert your GLM4 MoE model from HuggingFace format to TensorRT-LLM format:

```bash
python convert_checkpoint.py \
    --model_dir /path/to/glm4-moe-model \
    --output_dir /path/to/converted/model \
    --dtype float16 \
    --tensor_parallel_size 2 \
    --expert_parallel_size 2
```

### 2. Build TensorRT Engine

Build the TensorRT engine from the converted model:

```bash
python -m tensorrt_llm.models.glm4_moe_model \
    --model_dir /path/to/converted/model \
    --output_dir /path/to/engine \
    --dtype float16 \
    --tensor_parallel_size 2 \
    --expert_parallel_size 2
```

### 3. Run Inference

Use the TensorRT-LLM Python API for inference:

```python
import tensorrt_llm
from tensorrt_llm import LLM

# Initialize the model
llm = LLM(
    model="/path/to/engine",
    tokenizer="/path/to/tokenizer",
    backend="torch",  # Use PyTorch backend
    tensor_parallel_size=2,
    expert_parallel_size=2
)

# Generate text
output = llm.generate("Hello, how are you?", max_tokens=100)
print(output)
```

## Configuration

### Model Configuration

The GLM4 MoE model supports the following configuration options:

```python
from tensorrt_llm.models.chatglm.config import ChatGLMConfig
from tensorrt_llm.layers.moe import MoeConfig

# Basic configuration
config = ChatGLMConfig(
    architecture="GLM4MoEForCausalLM",
    dtype="float16",
    num_hidden_layers=32,
    num_attention_heads=32,
    hidden_size=4096,
    intermediate_size=14336,
    vocab_size=100008,
    max_position_embeddings=8192,
    chatglm_version="glm4_moe"
)

# MoE configuration
moe_config = MoeConfig(
    num_experts=8,
    top_k=2,
    normalization_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
    interleave_moe_layer_step=2
)
config.moe_config = moe_config
```

### Parallelism Configuration

```python
from tensorrt_llm.mapping import Mapping

# Tensor parallelism
mapping = Mapping(
    world_size=4,
    tp_size=2,  # Tensor parallel size
    pp_size=1,  # Pipeline parallel size
    ep_size=2   # Expert parallel size
)
```

## Performance Optimization

### MoE Backend Selection

TensorRT-LLM supports multiple MoE backends:

- **CUTLASS**: Default backend, good for most use cases
- **TRTLLM**: Optimized for specific quantization modes (FP8, NVFP4)
- **VANILLA**: Simple implementation for debugging
- **WIDEEP**: For wide expert parallelism

```python
# Configure MoE backend
llm = LLM(
    model="/path/to/engine",
    moe_backend="CUTLASS",  # or "TRTLLM", "VANILLA", "WIDEEP"
    backend="torch"
)
```

### Quantization

GLM4 MoE supports various quantization methods:

```python
from tensorrt_llm.models.modeling_utils import QuantConfig, QuantMode

# FP8 quantization
quant_config = QuantConfig(quant_mode=QuantMode.FP8)

# INT8 weight-only quantization
quant_config = QuantConfig(quant_mode=QuantMode.use_weight_only())

# NVFP4 quantization
quant_config = QuantConfig(quant_mode=QuantMode.NVFP4)
```

## Examples

### Basic Inference

```python
from tensorrt_llm import LLM

# Load model
llm = LLM(
    model="/path/to/glm4-moe-engine",
    tokenizer="/path/to/tokenizer",
    backend="torch"
)

# Generate text
prompt = "Explain quantum computing in simple terms:"
output = llm.generate(prompt, max_tokens=200, temperature=0.7)
print(output)
```

### Batch Processing

```python
# Batch generation
prompts = [
    "What is machine learning?",
    "Explain neural networks:",
    "How does deep learning work?"
]

outputs = llm.generate(prompts, max_tokens=100)
for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Output: {output}\n")
```

### Streaming Generation

```python
# Streaming generation
for output in llm.generate_streaming("Write a story about a robot:", max_tokens=100):
    print(output, end="", flush=True)
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: GLM4 MoE models are large. Use appropriate parallelism:
   ```bash
   --tensor_parallel_size 2 --expert_parallel_size 2
   ```

2. **Expert Parallelism**: Ensure the number of experts is divisible by the expert parallel size:
   ```python
   # If you have 8 experts and expert_parallel_size=2, this works
   # If you have 7 experts and expert_parallel_size=2, this fails
   ```

3. **MoE Backend Compatibility**: Some backends require specific quantization:
   ```python
   # TRTLLM backend requires FP8 or NVFP4 quantization
   if moe_backend == "TRTLLM":
       assert quant_config.quant_mode.has_fp8_block_scales() or quant_config.quant_mode.has_nvfp4()
   ```

### Performance Tips

1. **Use Expert Parallelism**: For large models, expert parallelism can significantly improve performance
2. **Choose Appropriate Backend**: CUTLASS is generally the best choice for most use cases
3. **Optimize Batch Size**: MoE models benefit from larger batch sizes due to expert utilization
4. **Monitor Expert Utilization**: Ensure experts are being used effectively

## Comparison with vLLM

This TensorRT-LLM implementation provides similar functionality to the vLLM GLM4 MoE integration:

| Feature | TensorRT-LLM | vLLM |
|---------|--------------|------|
| MoE Support | ✅ | ✅ |
| PyTorch Backend | ✅ | ✅ |
| Tensor Parallelism | ✅ | ✅ |
| Expert Parallelism | ✅ | ✅ |
| Quantization | ✅ | ✅ |
| Streaming | ✅ | ✅ |

## References

- [TensorRT-LLM Documentation](https://github.com/NVIDIA/TensorRT-LLM)
- [GLM4 Paper](https://arxiv.org/abs/2401.09625)
- [vLLM GLM4 MoE Implementation](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/glm4_moe_mtp.py)
- [Mixture of Experts](https://arxiv.org/abs/2101.03961) 