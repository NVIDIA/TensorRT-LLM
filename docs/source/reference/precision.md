(precision)=

# Numerical Precision

This document describes the different quantization recipes implemented in TensorRT-LLM and contains a support matrix
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

Examples of how to enable SmoothQuant for GPT, GPT-J and LLaMA can be found in
the [examples/quantization](source:examples/quantization) folder of that release.

## INT4 and INT8 Weight-Only (W4A16 and W8A16)

The INT4 and INT8 Weight-Only techniques consist in quantizing the weights of
a model and dequantizing those weights on-the-fly in linear layers (Matmuls).
The activations are encoded using floating-point values (FP16 or BF16).

To use INT4/INT8 Weight-Only methods, the user must determine the scaling
factors to use to quantize and dequantize the weights of the model.

This release includes examples for [GPT](source:examples/models/core/gpt) and
[LLaMA](source:examples/models/core/llama).

## GPTQ and AWQ (W4A16)

The GPTQ and AWQ techniques are presented in
[https://arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)
and
[https://arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978),
respectively. TensorRT-LLM supports per-group scaling factors and
zero-offsetting in linear layers to implement GPTQ and AWQ methods. See the
[WeightOnlyGroupwiseQuantMatmulPlugin](source:cpp/tensorrt_llm/plugins/weightOnlyGroupwiseQuantMatmulPlugin)
plugin and the corresponding
[`weight_only_groupwise_quant_matmul`](source:tensorrt_llm/quantization/functional.py)
Python function, for details.

This release includes examples of applying GPTQ to [GPT-NeoX](source:examples/models/core/gpt)
and [LLaMA-v2](source:examples/models/core/llama), as well as an example of using AWQ with
[GPT-J](source:examples/models/contrib/gptj).

## FP8 (Hopper)

This release of TensorRT-LLM contains implementations of FP8 for GPT-NeMo,
GPT-J and LLaMA. Those examples can be found in
[examples/quantization](source:examples/quantization).

## NVFP4 (Blackwell)

LLama and Mixtral can run in NVFP4 datatype. Those examples can be found in Llama examples.

## Support matrix

This release of TensorRT-LLM contains the following examples:

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

## Technical Detail: The `QuantMode` Flags

The quantization method is controlled by the
[`QuantMode`](source:tensorrt_llm/quantization/mode.py) flags. The different fields
are:

 * `INT4_WEIGHTS`, the weights are quantized to 4 bits (W4A\*),
 * `INT8_WEIGHTS`, the weights are quantized to 8 bits (W8A\*),
 * `ACTIVATIONS`, the activations are quantized to 8 bits (W\*A8),
 * `PER_CHANNEL`, the scaling factors are defined per channel,
 * `PER_TOKEN`, the scaling factors are defined per token,
 * `PER_GROUP`, the scaling factors are defined per group.

There are three additional flags to control TensorRT-LLM:

 * `INT8_KV_CACHE`, the K/V cache stores K and V using 8-bit integers,
 * `FP8_KV_CACHE`, the K/V cache stores K and V using 8-bit floating-point numbers,
 * `FP8_QDQ`, TensorRT-LLM relies on automatic fusion of Q/DQ nodes in TensorRT.
