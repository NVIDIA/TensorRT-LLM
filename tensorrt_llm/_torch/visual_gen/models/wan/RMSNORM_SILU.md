# Experimental BF16 decoder RMSNorm + SiLU

`TLLM_WAN_VAE_FUSED_RMSNORM_SILU=1` opts an ordinary native BF16 Wan VAE
checkpoint into the fused decoder path. The default is off. The loader reads
this internal experimental override once, after loading weights and placing the
VAE in eval mode on its final device and dtype. Packed NVFP4 checkpoints,
including their BF16 dequantization route, and requested NVFP4 execution keep
their existing behavior.

Only residual-block norm1/norm2 and decoder norm_out are prepared. The encoder
and attention norm remain native. Eligible calls are eager CUDA inference with
BF16 dense channels-last 5D activations and 256, 512 or 1024 channels. Training,
autograd, CUDA graph capture, inputs on a non-current device, other layouts/dtypes,
empty inputs, instrumentation and unsupported
module state use the original normalization followed by the original activation.
The override does not enable VAE compilation or change convolution/cache order.

Each selected norm owns a nonpersistent positive-zero BF16 bias buffer. It is
prepared once, follows module device/dtype moves, adds no checkpoint key, and
is reused without allocation or filling during forward. The zero-bias addition
is retained to preserve the tested arithmetic and rounding boundaries. Calls
still allocate a fresh output with the input strides.

## Source and numerical behavior

The JIT body and launch configuration in `rmsnorm_silu.py` are reused from
[SGLang at cc171fbad0266e8eaabb031f4d3858557a23d7e8](https://github.com/sgl-project/sglang/blob/cc171fbad0266e8eaabb031f4d3858557a23d7e8/python/sglang/kernels/ops/diffusion/norm/wan_rmsnorm_silu_triton.py)
under Apache-2.0. Generic fusion credit belongs to SGLang. TensorRT-LLM adapts
operator registration, decoder ownership, lifecycle and native fallback.

The kernel accumulates the norm in FP32 and preserves the BF16 normalization,
scale, affine and zero-bias rounding boundaries before SiLU. It uses Triton's
sqrt and sigmoid lowering; it is not claimed to be bit-identical to the eager
native implementation. No approximation flag or arithmetic transformation is
added by this port.

A prior exact-kernel, per-instance integration was measured on FastWan 5B with
three fixed prompts, 3 steps, 704x1280x121, BF16, on one B200. Moving preparation
to load time and adapting to the current native implementation are new changes.
The previous measurement is not production-port validation. Before enabling by
default, validate this port's numerical error, output quality against the
existing golden, representative end-to-end performance, and cold-load behavior.
Other models, shapes, hardware and concurrency are not covered by that prior
measurement. The approximate-math accuracy study and acceptance bar remain
subject to the VisualGen engineering review.

## Tests

`tests/unittest/_torch/visual_gen/test_wan_rmsnorm_silu.py` covers the override,
metadata dispatch/fallback, hooks and gradients, decoder-only buffer ownership,
checkpoint/device lifecycle, FP4 loader isolation, and CUDA numerical checks.
The numerical test uses a FP32 oracle and the bound
`candidate_max_error <= 2 * native_max_error + 1e-6` for the same complete input.
It also verifies actual fused dispatch, output layout, and input/weight immutability.
GPU tests require CUDA; missing TensorRT-LLM test dependencies are not skipped.
