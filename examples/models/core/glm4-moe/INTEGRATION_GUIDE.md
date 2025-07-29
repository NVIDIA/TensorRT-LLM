# GLM4 MoE Integration Guide for TensorRT-LLM

This guide explains how to integrate GLM4 MoE (Mixture of Experts) models with TensorRT-LLM's PyTorch backend, similar to the vLLM implementation.

## Overview

This implementation provides a complete GLM4 MoE integration for TensorRT-LLM, extending the existing GLM support to include Mixture of Experts layers. The implementation is designed to be compatible with the vLLM GLM4 MoE integration while leveraging TensorRT-LLM's optimized MoE kernels and PyTorch backend.

## Key Features

- **Full MoE Support**: Complete Mixture of Experts implementation with configurable experts and routing
- **PyTorch Backend**: Seamless integration with TensorRT-LLM's PyTorch backend
- **Multiple MoE Backends**: Support for CUTLASS, TRTLLM, VANILLA, and WIDEEP backends
- **Parallelism**: Tensor, pipeline, and expert parallelism support
- **Quantization**: FP8, INT8, and NVFP4 quantization support
- **Performance**: Optimized MoE kernels for high-performance inference

## Architecture

### Model Components

1. **GLM4MoEDecoderLayer**: Decoder layer that alternates between regular MLP and MoE layers
2. **GLM4MoEModel**: Main transformer model with MoE support
3. **GLM4MoEForCausalLM**: Complete language model with MoE layers

### MoE Configuration

```python
from tensorrt_llm.layers.moe import MoeConfig

moe_config = MoeConfig(
    num_experts=8,           # Number of experts
    top_k=2,                 # Number of experts to select per token
    normalization_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
    interleave_moe_layer_step=2  # Use MoE every 2nd layer
)
```

## Installation and Setup

### 1. Prerequisites

- TensorRT-LLM installed with PyTorch backend support
- CUDA-compatible GPU(s)
- Python 3.8+

### 2. Model Conversion

Convert your GLM4 MoE model from HuggingFace format:

```bash
python convert_checkpoint.py \
    --model_dir /path/to/glm4-moe-model \
    --output_dir /path/to/converted/model \
    --dtype float16 \
    --tensor_parallel_size 2 \
    --expert_parallel_size 2
```

### 3. Basic Usage

```python
from tensorrt_llm import LLM

# Load the model with PyTorch backend
llm = LLM(
    model="/path/to/converted/model",
    tokenizer="/path/to/tokenizer",
    backend="torch",  # Use PyTorch backend
    tensor_parallel_size=2,
    expert_parallel_size=2,
    moe_backend="CUTLASS"
)

# Generate text
output = llm.generate("Hello, how are you?", max_tokens=100)
print(output)
```

## Advanced Configuration

### MoE Backend Selection

```python
# CUTLASS backend (default, recommended)
llm = LLM(model=model_path, moe_backend="CUTLASS")

# TRTLLM backend (for FP8/NVFP4 quantization)
llm = LLM(model=model_path, moe_backend="TRTLLM")

# VANILLA backend (for debugging)
llm = LLM(model=model_path, moe_backend="VANILLA")

# WIDEEP backend (for wide expert parallelism)
llm = LLM(model=model_path, moe_backend="WIDEEP")
```

### Parallelism Configuration

```python
# Single GPU
llm = LLM(model=model_path, tensor_parallel_size=1, expert_parallel_size=1)

# Tensor parallelism (2 GPUs)
llm = LLM(model=model_path, tensor_parallel_size=2, expert_parallel_size=1)

# Expert parallelism (2 GPUs)
llm = LLM(model=model_path, tensor_parallel_size=1, expert_parallel_size=2)

# Combined parallelism (4 GPUs)
llm = LLM(model=model_path, tensor_parallel_size=2, expert_parallel_size=2)
```

### Quantization

```python
from tensorrt_llm.models.modeling_utils import QuantConfig, QuantMode

# FP8 quantization
quant_config = QuantConfig(quant_mode=QuantMode.FP8)

# INT8 weight-only quantization
quant_config = QuantConfig(quant_mode=QuantMode.use_weight_only())

# NVFP4 quantization
quant_config = QuantConfig(quant_mode=QuantMode.NVFP4)
```

## Performance Optimization

### 1. MoE Backend Selection

- **CUTLASS**: Best for general use cases, good performance across quantization modes
- **TRTLLM**: Optimized for FP8 and NVFP4 quantization
- **VANILLA**: Simple implementation for debugging
- **WIDEEP**: For wide expert parallelism scenarios

### 2. Expert Parallelism

For large models, expert parallelism can significantly improve performance:

```python
# For 8 experts, use 2 or 4 GPUs for expert parallelism
llm = LLM(
    model=model_path,
    expert_parallel_size=2,  # Distribute experts across 2 GPUs
    moe_backend="CUTLASS"
)
```

### 3. Batch Size Optimization

MoE models benefit from larger batch sizes due to expert utilization:

```python
# Use larger batch sizes for better expert utilization
llm = LLM(
    model=model_path,
    max_batch_size=16,  # Increase batch size
    max_input_len=2048,
    max_output_len=512
)
```

## Comparison with vLLM

| Feature | TensorRT-LLM | vLLM |
|---------|--------------|------|
| MoE Support | ✅ | ✅ |
| PyTorch Backend | ✅ | ✅ |
| Tensor Parallelism | ✅ | ✅ |
| Expert Parallelism | ✅ | ✅ |
| Multiple MoE Backends | ✅ | ✅ |
| FP8 Quantization | ✅ | ✅ |
| INT8 Quantization | ✅ | ✅ |
| NVFP4 Quantization | ✅ | ✅ |
| Streaming Generation | ✅ | ✅ |
| Batch Processing | ✅ | ✅ |

## Examples

### Basic Inference

```python
from tensorrt_llm import LLM

llm = LLM(
    model="/path/to/glm4-moe-model",
    tokenizer="/path/to/tokenizer",
    backend="torch",
    tensor_parallel_size=2,
    expert_parallel_size=2
)

prompt = "Explain quantum computing in simple terms:"
output = llm.generate(prompt, max_tokens=200, temperature=0.7)
print(output)
```

### Batch Processing

```python
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
for token in llm.generate_streaming("Write a story about a robot:", max_tokens=100):
    print(token, end="", flush=True)
```

### Advanced Configuration

```python
from tensorrt_llm.models.chatglm.config import ChatGLMConfig
from tensorrt_llm.layers.moe import MoeConfig

# Create custom configuration
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

# Configure MoE
moe_config = MoeConfig(
    num_experts=8,
    top_k=2,
    normalization_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
    interleave_moe_layer_step=2
)
config.moe_config = moe_config
```

## Troubleshooting

### Common Issues

1. **Memory Issues**
   ```bash
   # Use appropriate parallelism
   --tensor_parallel_size 2 --expert_parallel_size 2
   ```

2. **Expert Parallelism Requirements**
   ```python
   # Ensure number of experts is divisible by expert_parallel_size
   # 8 experts with expert_parallel_size=2 works
   # 7 experts with expert_parallel_size=2 fails
   ```

3. **MoE Backend Compatibility**
   ```python
   # TRTLLM backend requires specific quantization
   if moe_backend == "TRTLLM":
       assert quant_config.quant_mode.has_fp8_block_scales() or quant_config.quant_mode.has_nvfp4()
   ```

### Performance Tips

1. **Monitor Expert Utilization**: Ensure experts are being used effectively
2. **Choose Appropriate Backend**: CUTLASS is generally the best choice
3. **Optimize Batch Size**: Larger batches improve expert utilization
4. **Use Expert Parallelism**: For large models, expert parallelism improves performance

## Testing

Run the test suite to verify the implementation:

```bash
python test_glm4_moe.py
```

This will test:
- Model creation and configuration
- Forward pass functionality
- MoE layer configuration
- Weight loading compatibility

## References

- [TensorRT-LLM Documentation](https://github.com/NVIDIA/TensorRT-LLM)
- [vLLM GLM4 MoE Implementation](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/glm4_moe_mtp.py)
- [GLM4 Paper](https://arxiv.org/abs/2401.09625)
- [Mixture of Experts](https://arxiv.org/abs/2101.03961)

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the test suite for common problems
3. Consult the TensorRT-LLM documentation
4. Check the vLLM GLM4 MoE implementation for reference 