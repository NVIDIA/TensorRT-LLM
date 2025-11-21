# Quantization

## Quantization in TensorRT LLM

Quantization is a technique used to reduce memory footprint and computational cost by converting the model's weights and/or activations from high-precision floating-point numbers (like BF16) to lower-precision data types, such as INT8, FP8, or FP4.

TensorRT LLM offers a variety of quantization recipes to optimize LLM inference. These recipes can be broadly categorized as follows:

* FP4
* FP8 Per Tensor
* FP8 Block Scaling
* FP8 Rowwise
* FP8 KV Cache
* W4A16 GPTQ
* W4A8 GPTQ
* W4A16 AWQ
* W4A8 AWQ


## Usage

The default PyTorch backend supports FP4 and FP8 quantization on the latest Blackwell and Hopper GPUs.

### Running Pre-quantized Models

TensorRT LLM can directly run [pre-quantized models](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4) generated with the [NVIDIA TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer).

```python
from tensorrt_llm import LLM
llm = LLM(model='nvidia/Llama-3.1-8B-Instruct-FP8')
llm.generate("Hello, my name is")
```

#### FP8 KV Cache

```{note}
TensorRT LLM allows you to enable the FP8 KV cache manually, even for checkpoints that do not have it enabled by default.
```

Here is an example of how to set the FP8 KV Cache option:

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig
llm = LLM(model='/path/to/model',
          kv_cache_config=KvCacheConfig(dtype='fp8'))
llm.generate("Hello, my name is")
```

### Offline Quantization with ModelOpt

If a pre-quantized model is not available on the [Hugging Face Hub](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4), you can quantize it offline using ModelOpt.

Follow this step-by-step guide to quantize a model:

```bash
git clone https://github.com/NVIDIA/TensorRT-Model-Optimizer.git
cd TensorRT-Model-Optimizer/examples/llm_ptq
scripts/huggingface_example.sh --model <huggingface_model_card> --quant fp8 --export_fmt hf
```

## Model Supported Matrix

| Model          |  NVFP4  | MXFP4  | FP8(per tensor)| FP8(block scaling) | FP8(rowwise) | FP8 KV Cache |W4A8 AWQ  | W4A16 AWQ | W4A8 GPTQ  | W4A16 GPTQ |
| :------------- | :---:   | :---:  | :---: | :---: | :---: | :---: | :-------: | :-------: | :--------: | :--------: |
| BERT           |   .     |   .    |   .   |   .   |   .   |   Y   |     .     |     .     |     .      |     .      |
| DeepSeek-R1    |   Y     |   .    |   .   |   Y   |   .   |   Y   |     .     |     .     |     .      |     .      |
| EXAONE         |   .     |   .    |   Y   |   .   |   .   |   Y   |     Y     |     Y     |     .      |     .      |
| Gemma 3        |   .     |   .    |   Y   |   .   |   .   |   Y   |     Y     |     Y     |     .      |     .      |
| GPT-OSS        |   .     |   Y    |   .   |   .   |   .   |   Y   |     .     |     .     |     .      |     .      |
| LLaMA          |   Y     |   .    |   Y   |   .   |   .   |   Y   |     .     |     Y     |     .      |     Y      |
| LLaMA-v2       |   Y     |   .    |   Y   |   .   |   .   |   Y   |     Y     |     Y     |     .      |     Y      |
| LLaMA 3        |   .     |   .    |   .   |   .   |   Y   |   Y   |     Y     |     .     |     .      |     .      |
| LLaMA 4        |   Y     |   .    |   Y   |   .   |   .   |   Y   |     .     |     .     |     .      |     .      |
| Mistral        |   .     |   .    |   Y   |   .   |   .   |   Y   |     .     |     Y     |     .      |     .      |
| Mixtral        |   Y     |   .    |   Y   |   .   |   .   |   Y   |     .     |     .     |     .      |     .      |
| Phi            |   .     |   .    |   .   |   .   |   .   |   Y   |     Y     |     .     |     .      |     .      |
| Qwen           |   .     |   .    |   .   |   .   |   .   |   Y   |     Y     |     Y     |     .      |     Y      |
| Qwen-2/2.5     |   Y     |   .    |   Y   |   .   |   .   |   Y   |     Y     |     Y     |     .      |     Y      |
| Qwen-3         |   Y     |   .    |   Y   |   .   |   .   |   Y   |     .     |     Y     |     .      |     Y      |
| BLIP2-OPT      |   .     |   .    |   .   |   .   |   .   |   Y   |     .     |     .     |     .      |     .      |
| BLIP2-T5       |   .     |   .    |   .   |   .   |   .   |   Y   |     .     |     .     |     .      |     .      |
| LLaVA          |   .     |   .    |   Y   |   .   |   .   |   Y   |     .     |     Y     |     .      |     Y      |
| VILA           |   .     |   .    |   Y   |   .   |   .   |   Y   |     .     |     Y     |     .      |     Y      |
| Nougat         |   .     |   .    |   .   |   .   |   .   |   Y   |     .     |     .     |     .      |     .      |


```{note}
The vision component of multi-modal models(BLIP2-OPT/BLIP2-T5/LLaVA/VILA/Nougat) uses FP16 by default.
The language component decides which quantization methods are supported by a given multi-modal model.
```


## Hardware Support Matrix 

| Model          |  NVFP4  | MXFP4  | FP8(per tensor)| FP8(block scaling) | FP8(rowwise) | FP8 KV Cache |W4A8 AWQ  | W4A16 AWQ | W4A8 GPTQ  | W4A16 GPTQ |
| :------------- | :---:   | :---:  | :---: | :---: | :---: | :---: | :-------: | :-------: | :--------: | :--------: |
| Blackwell(sm120)       |   Y     |   Y    |   Y   |   .   |   .   |   Y   |     .     |     .     |     .      |     .      |
| Blackwell(sm100)       |   Y     |   Y    |   Y   |   Y   |   .   |   Y   |     .     |     .     |     .      |     .      |
| Hopper           |   .     |   .    |   Y   |   Y   |   Y   |   Y   |     Y     |     Y     |     Y      |     Y      |
| Ada Lovelace          |   .     |   .    |   Y   |   .   |   .   |   Y   |     Y     |     Y     |     Y      |     Y      |
| Ampere         |   .     |   .    |   .   |   .   |   .   |   Y   |     .     |     Y     |     .      |     Y      |
```{note}
FP8 block wise scaling GEMM kernels for sm100 are using MXFP8 recipe (E4M3 act/weight and UE8M0 act/weight scale), which is slightly different from SM90 FP8 recipe (E4M3 act/weight and FP32 act/weight scale).
```


## Quick Links

- [Pre-quantized Models by ModelOpt](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4)
- [ModelOpt Support Matrix](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/0_support_matrix.html)
