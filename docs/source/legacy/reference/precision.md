(precision)=

# Numerical Precision

```{caution}
The legacy TensorRT backend has been removed and is no longer supported. This page is retained for cross-reference only.
```

> [!WARNING]
> Per-model TensorRT example trees cited below
> (`examples/models/core/gpt`, `examples/models/core/llama`,
> `examples/models/contrib/gptj`, and the deleted
> `examples/quantization/quantize.py` / `convert_checkpoint.py` flows) are **gone**
> with the TensorRT backend. Do not follow those paths.
> For current quantization on the PyTorch backend, see
> [features/quantization.md](../../features/quantization.md),
> [torch/features/quantization.md](../../torch/features/quantization.md),
> and [examples/quantization/README.md](../../../examples/quantization/README.md)
> (Model Optimizer / pre-quantized HF checkpoints). Also see
> [TensorRT Backend Removed](../tensorrt-backend-removal.md).

This document describes the different quantization recipes that were implemented
for the **legacy** TensorRT backend and contains a historical support matrix
for the different models.

## FP32, FP16 and BF16

The different models implemented in TensorRT-LLM work with 32-bit IEEE
floating-point (FP32) numbers. When checkpoints are available, the models also
support 16-bit IEEE floating-point numbers (FP16) and 16-bit Bfloat16 (BF16) as
described [here](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format).

## Quantization and Dequantization (Q/DQ)

Given a floating-point number `x` and a floating-point scaling factor `s`,
TensorRT-LLM implements INT8 quantization as:

```
q = int8.satfinite(x * s)
```

Given an INT8 number `q` and a floating-point scaling factor `s`, TensorRT-LLM
implements INT8 dequantization to the floating-point (FP) type as:

```
x = static_cast<FP>(q) * s
```

Given a matrix (2D tensor) of shape `M x N` (`M` rows and `N` columns) where
`M` is the number of tokens and `N` is the number of channels. TensorRT-LLM has
the three following modes to quantize and dequantize the elements of the
tensor:

 * Per-tensor: It uses a single scaling factor for all the elements,
 * Per-token: It uses a different scaling factor for each token. There are `M`
   scaling factors in that case,
 * Per-channel: It uses a different scaling factor for each channel. There are
   `N` scaling factors in that case.

Note that per-token and per-channel scaling modes can be used together (i.e.
they are _not_ mutually exclusive).

In pseudo-code, the quantization can be implemented as follows for the three
different modes:

```python
# Per-tensor scaling.
for mi in range(M):
    for ni in range(N):
        q[mi][ni] = int8.satfinite(x[mi][ni] * s)

# Per-token scaling.
for mi in range(M):
    for ni in range(N):
        q[mi][ni] = int8.satfinite(x[mi][ni] * s[mi])

# Per-channel scaling.
for mi in range(M):
    for ni in range(N):
        q[mi][ni] = int8.satfinite(x[mi][ni] * s[ni])
```

## INT8 SmoothQuant (W8A8)

The SmoothQuant technique was introduced in
[https://arxiv.org/abs/2211.10438](https://arxiv.org/abs/2211.10438). It is a
method to run inference using INT8 for both activations and weights while
maintaining the accuracy of the network (on downstream tasks).

As explained in the research paper, preprocessing must be applied to the
weights of the model. TensorRT-LLM includes scripts to prepare the model to
run using the SmoothQuant method.

Historically, SmoothQuant enablement examples for GPT, GPT-J and LLaMA lived under
the removed TensorRT `examples/quantization/quantize.py` / per-model convert flows.
Those scripts are gone; use the current quantization guides linked in the warning above.

## INT4 and INT8 Weight-Only (W4A16 and W8A16)

The INT4 and INT8 Weight-Only techniques consist in quantizing the weights of
a model and dequantizing those weights on-the-fly in linear layers (Matmuls).
The activations are encoded using floating-point values (FP16 or BF16).

To use INT4/INT8 Weight-Only methods, the user must determine the scaling
factors to use to quantize and dequantize the weights of the model.

The legacy TensorRT release shipped Weight-Only examples under the deleted
`examples/models/core/gpt` and `examples/models/core/llama` trees. Those directories
are no longer in the repository.

## GPTQ and AWQ (W4A16)

The GPTQ and AWQ techniques are presented in
[https://arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)
and
[https://arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978),
respectively. The legacy TensorRT backend supported per-group scaling factors and
zero-offsetting in linear layers to implement GPTQ and AWQ methods via the
removed `WeightOnlyGroupwiseQuantMatmulPlugin`
(`cpp/tensorrt_llm/plugins/weightOnlyGroupwiseQuantMatmulPlugin`) and related
Python helpers. Those plugin sources are no longer in the tree.

Legacy GPTQ/AWQ examples for GPT-NeoX, LLaMA-v2, and GPT-J lived under the deleted
`examples/models/core/gpt`, `examples/models/core/llama`, and
`examples/models/contrib/gptj` directories.

## FP8 (Hopper)

The legacy TensorRT release contained FP8 implementations for GPT-NeMo,
GPT-J and LLaMA under the removed `examples/quantization/quantize.py` convert flow.
For FP8 today, load a pre-quantized Hugging Face checkpoint or use Model Optimizer
(see the warning above).

## NVFP4 (Blackwell)

Llama and Mixtral historically ran in NVFP4 via the deleted Llama TensorRT example tree.
Use a pre-quantized NVFP4 Hugging Face checkpoint with the PyTorch backend instead.

## Support matrix

The following matrix records which precision recipes the **legacy** TensorRT
backend historically advertised for each model (not a guarantee of current
PyTorch-backend support):

| Model          | FP32  | FP16  | BF16  |  FP8  | NVFP4 | W8A8 SQ | W8A16 | W4A16 | W4A16 AWQ | W4A16 GPTQ |
| :------------- | :---: | :---: | :---: | :---: | :---: | :-----: | :---: | :---: | :-------: | :--------: |
| Baichuan       |   Y   |   Y   |   Y   |   Y   |   .   |    Y    |   Y   |   Y   |     Y     |     Y      |
| BERT           |   Y   |   Y   |   Y   |   .   |   .   |    .    |   .   |   .   |     .     |     .      |
| BLIP-2         |   Y   |   Y   |   Y   |   .   |   .   |    .    |   .   |   .   |     .     |     .      |
| BLOOM          |   Y   |   Y   |   Y   |   Y   |   .   |    Y    |   Y   |   Y   |     .     |     .      |
| ChatGLM        |   Y   |   Y   |   Y   |   .   |   .   |    .    |   .   |   .   |     .     |     .      |
| ChatGLM-v2     |   Y   |   Y   |   Y   |   .   |   .   |    .    |   .   |   .   |     .     |     .      |
| ChatGLM-v3     |   Y   |   Y   |   Y   |   .   |   .   |    .    |   .   |   .   |     .     |     .      |
| DBRX           |   Y   |   Y   |   Y   |   .   |   .   |    .    |   Y   |   Y   |     .     |     .      |
| Falcon         |   Y   |   Y   |   Y   |   Y   |   .   |    .    |   Y   |   Y   |     Y     |     .      |
| Flan-T5        |   Y   |   Y   |   Y   |   .   |   .   |    .    |   .   |   .   |     .     |     .      |
| Gemma          |   Y   |   Y   |   Y   |   Y   |   .   |    Y    |   Y   |   Y   |     Y     |     .      |
| GPT            |   Y   |   Y   |   Y   |   Y   |   .   |    Y    |   Y   |   Y   |     .     |     .      |
| GPT-J          |   Y   |   Y   |   Y   |   Y   |   .   |    Y    |   Y   |   Y   |     Y     |     .      |
| GPT-NeMo       |   Y   |   Y   |   Y   |   .   |   .   |    .    |   .   |   .   |     .     |     .      |
| GPT-NeoX       |   Y   |   Y   |   Y   |   .   |   .   |    .    |   .   |   .   |     .     |     Y      |
| InternLM       |   Y   |   Y   |   Y   |   .   |   .   |    Y    |   Y   |   Y   |     .     |     .      |
| InternLM2      |   Y   |   Y   |   Y   |   .   |   .   |    .    |   .   |   .   |     .     |     .      |
| LLaMA          |   Y   |   Y   |   Y   |   Y   |   Y   |    Y    |   Y   |   Y   |     Y     |     Y      |
| LLaMA-v2       |   Y   |   Y   |   Y   |   Y   |   Y   |    Y    |   Y   |   Y   |     Y     |     Y      |
| Mamba          |   Y   |   Y   |   Y   |   .   |   .   |    .    |   .   |   .   |     .     |     .      |
| Mistral        |   Y   |   Y   |   Y   |   Y   |   .   |    Y    |   Y   |   Y   |     Y     |     .      |
| Mixtral        |   Y   |   Y   |   Y   |   Y   |   Y   |    .    |   Y   |   Y   |     .     |     .      |
| MPT            |   Y   |   Y   |   Y   |   Y   |   .   |    Y    |   Y   |   Y   |     Y     |     .      |
| OPT            |   Y   |   Y   |   Y   |   .   |   .   |    .    |   .   |   .   |     .     |     .      |
| Phi            |   Y   |   Y   |   Y   |   .   |   .   |    .    |   .   |   .   |     .     |     .      |
| Qwen           |   Y   |   Y   |   Y   |   .   |   .   |    Y    |   Y   |   Y   |     Y     |     Y      |
| RecurrentGemma |   Y   |   Y   |   Y   |   Y   |   .   |    Y    |   .   |   .   |     Y     |     .      |
| Replit Code    |   Y   |   Y   |   Y   |   .   |   .   |    .    |   .   |   .   |     .     |     .      |
| SantaCoder     |   Y   |   Y   |   Y   |   .   |   .   |    .    |   Y   |   Y   |     .     |     .      |
| Skywork        |   Y   |   Y   |   Y   |   .   |   .   |    .    |   .   |   .   |     .     |     .      |
| StarCoder1     |   Y   |   Y   |   Y   |   .   |   .   |    .    |   Y   |   Y   |     .     |     .      |
| StarCoder2     |   Y   |   Y   |   Y   |   Y   |   .   |    .    |   Y   |   Y   |     .     |     .      |
| T5             |   Y   |   Y   |   Y   |   .   |   .   |    .    |   .   |   .   |     .     |     .      |
| Whisper        |   Y   |   Y   |   Y   |   .   |   .   |    .    |   Y   |   Y   |     .     |     .      |
| BLIP2-OPT      |   Y   |   Y   |   Y   |   .   |   .   |    .    |   .   |   .   |     .     |     .      |
| BLIP2-T5       |   Y   |   Y   |   Y   |   .   |   .   |    .    |   .   |   .   |     .     |     .      |
| LLaVA          |   Y   |   Y   |   Y   |   Y   |   .   |    Y    |   Y   |   Y   |     Y     |     Y      |
| VILA           |   Y   |   Y   |   Y   |   Y   |   .   |    Y    |   Y   |   Y   |     Y     |     Y      |
| Nougat         |   Y   |   Y   |   Y   |   .   |   .   |    .    |   .   |   .   |     .     |     .      |

Note: The vision component of multi-modal models(BLIP2-OPT/BLIP2-T5/LLaVA/VILA/Nougat) uses FP16 by default.
The language component decides which quantization methods are supported by a given multi-modal model.

## Technical Detail: The `QuantMode` Flags (legacy)

The legacy TensorRT quantization method was controlled by the
[`QuantMode`](source:tensorrt_llm/quantization/mode.py) flags (still present in-tree
for compatibility). The different fields are:

 * `INT4_WEIGHTS`, the weights are quantized to 4 bits (W4A\*),
 * `INT8_WEIGHTS`, the weights are quantized to 8 bits (W8A\*),
 * `ACTIVATIONS`, the activations are quantized to 8 bits (W\*A8),
 * `PER_CHANNEL`, the scaling factors are defined per channel,
 * `PER_TOKEN`, the scaling factors are defined per token,
 * `PER_GROUP`, the scaling factors are defined per group.

There are three additional flags that historically controlled TensorRT-LLM engine builds:

 * `INT8_KV_CACHE`, the K/V cache stores K and V using 8-bit integers,
 * `FP8_KV_CACHE`, the K/V cache stores K and V using 8-bit floating-point numbers,
 * `FP8_QDQ`, TensorRT-LLM relies on automatic fusion of Q/DQ nodes in TensorRT.
