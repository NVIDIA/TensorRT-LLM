# Quantization

## ModelOpt Integration

The PyTorch backend currently supports only models quantized using [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer). When quantizing with ModelOpt using `export_fmt=hf`, the quantized checkpoint contains `hf_quant_config` which enables TensorRT-LLM to automatically understand and load the quantization format.

## Supported Quantization Methods

### FP8 Quantization
Reduces model size by using 8-bit floating-point representation for weights and activations. Supports both static and dynamic quantization modes.

### NVFP4 Quantization
NVIDIA's 4-bit floating-point format with optimized CUDA kernels. Uses 4-bit weights with FP8 scaling factors for maximum compression.

### W4A16 AWQ
Activation-aware weight quantization that preserves important weight channels. Uses 4-bit weights with 16-bit activations and group-wise quantization.

### W4A8 AWQ
Similar to W4A16 AWQ but with 8-bit activations. Provides better compression than W4A16 while maintaining good accuracy.


## Usage Example

### Quantize using ModelOpt


```bash
git clone https://github.com/NVIDIA/TensorRT-Model-Optimizer.git
cd TensorRT-Model-Optimizer/examples/llm_ptq
scripts/huggingface_example.sh --model <huggingface_model_card> --quant <fp8/int4_awq/nvfp4/w4a8_awq> --export_fmt hf
```

### Model Inference
Then run this using TensorRT-LLM
```python
from tensorrt_llm import LLM
from pathlib import Path

quantized_model_path = Path(...)
llm = LLM(model=quantized_model_path)
llm.generate("Hello, my name is")
```
