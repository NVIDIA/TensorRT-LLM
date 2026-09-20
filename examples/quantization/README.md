# Model Quantization

To run quantized models with TensorRT LLM:

- Use a pre-quantized Hugging Face checkpoint (for example the FP8/NVFP4
  checkpoints published on the [NVIDIA Hugging Face hub](https://huggingface.co/nvidia)).
  Quantization settings are detected automatically when the model loads.
- To quantize your own model, use the
  [NVIDIA TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
  Hugging Face export flow (`examples/llm_ptq` in that repository).

See the [quantization feature documentation](https://nvidia.github.io/TensorRT-LLM/features/quantization.html)
for supported formats per GPU architecture.

## Mixed-precision MoE checkpoints

[`quantize_mixed_precision_moe.py`](quantize_mixed_precision_moe.py) builds a
mixed-precision MoE checkpoint from separately quantized checkpoints; see the
script's argparse help for usage.
