# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fused Triton kernel for the per-tensor FP8 (e4m3) dynamic-quant SCALE.

Replaces the amax half of ``torch.ops.tensorrt_llm.quantize_e4m3_per_tensor``
(``computeFP8QuantizeScale`` in ``cudaFp8Utils.cu``), a shared-memory block
reduce that is SM-underutilized / sync-bound (~6us) at skinny decode shapes and
serializes ahead of the projection GEMM.

For bf16/fp16 inputs (the activation dtypes this path uses) the produced scale
is *bitwise-identical* to the CUDA op's stored scale:

    amax  = x.abs().amax().float()              # exact over |x|, PER_TENSOR
    scale = clamp(amax * (1/448), min=1/(448*512))
    scale = scale.to(x.dtype).to(f32).reshape(1) # dtype rounding matches the op

(fp32 inputs agree to within one fp32 ULP: the op's stored fp32 scale has no
dtype rounding to absorb the last-bit difference between this kernel's
multiply-by-reciprocal and the op's divide — see the module's unit test.)

The apply half is left to the existing
``torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, scale)`` op,
whose output is bitwise-identical to the fused op for the same scale.

NaN/Inf are not expected on this path (post-projection activations are
finite). The Triton ``tl.maximum`` uses NaN-skipping semantics, which can
differ from the CUDA reduction on NaN-containing inputs, so the
bitwise-identity guarantee is scoped to finite inputs.

Design: 2-stage reduction (two tiny @triton.jit launches, no host sync ->
CUDA-graph capturable). A single-CTA loop is capped by one SM's bandwidth
(~5.8us on B200 at [64,2048]); the parallel reduction lands ~3.8us, so
[Triton scale + static apply] ~= 5.0us vs the fused op's ~9.2us (1.84x) as
measured under CUDA-graph replay (the decode critical-path condition).
"""

import torch
import triton
import triton.language as tl

FP8_E4M3_MAX = 448.0
MIN_SCALING_FACTOR = 1.0 / (448.0 * 512.0)

# Number of stage-1 partials: enough CTAs to spread the (tiny, ~256KB) read
# across SMs; a power of two so stage-2 loads them in one masked block.
_NUM_PARTIALS = 128
_STAGE1_BLOCK = 1024


@triton.jit
def _amax_partial_kernel(x_ptr, part_ptr, n_elements, P, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)  # |x| >= 0, so 0-init is safe
    stride = P * BLOCK
    for base in range(pid * BLOCK, n_elements, stride):
        idx = base + offs
        mask = idx < n_elements
        x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        acc = tl.maximum(acc, tl.abs(x))
    tl.store(part_ptr + pid, tl.max(acc, axis=0))


@triton.jit
def _scale_finalize_kernel(
    part_ptr,
    out_ptr,
    P: tl.constexpr,
    INV_FP8_MAX: tl.constexpr,
    MIN_SF: tl.constexpr,
    ROUND_DTYPE: tl.constexpr,
):
    offs = tl.arange(0, P)
    amax = tl.max(tl.load(part_ptr + offs), axis=0)
    # Match the live op `quantize_e4m3_per_tensor` EXACTLY: it multiplies by the
    # reciprocal (amax * (1/448)), NOT divide — verified empirically (1188
    # adversarial cases): the multiply form reproduces the op's returned scale
    # bitwise, the divide form does not. The returned scale is used downstream as
    # cublas_scaled_mm scale_a, so it must be bitwise-identical to baseline.
    scale = tl.maximum(amax * INV_FP8_MAX, MIN_SF)
    # Round to the input dtype exactly like the op stores the scale, then widen
    # back to fp32 — folded into this kernel (not a separate torch cast) so the
    # host/launch-overhead saving that motivates this op is not eroded at the
    # skinny decode shapes it targets.
    scale = scale.to(ROUND_DTYPE).to(tl.float32)
    tl.store(out_ptr, scale)


# torch activation dtype -> the Triton dtype the scale is rounded to (the op
# stores the scale in the input dtype). fp32 rounds to itself (a no-op).
_ROUND_DTYPE = {
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
}


def fp8_dynamic_quant_scale(x: torch.Tensor) -> torch.Tensor:
    """Compute the per-tensor FP8 (e4m3) dynamic-quant scale via a fused Triton reduction.

    For bf16/fp16 inputs the result is bitwise-identical to the scale returned by
    ``torch.ops.tensorrt_llm.quantize_e4m3_per_tensor`` (see the module
    docstring for the exact ``amax -> clamp -> dtype-round`` formula; fp32 agrees
    to within one ULP). The op stores the scale in the input dtype, so the fp32
    reduction is rounded to ``x.dtype`` and widened back to fp32 to match it.
    CUDA-graph capturable: static shapes, device-only, no ``.item()`` / host sync.

    Args:
        x: Activation tensor to quantize. Any floating dtype (bf16/fp16/fp32).
            Non-contiguous inputs are made contiguous first so the linear
            reduction sees the logical elements, not the raw storage.

    Returns:
        An fp32 tensor of shape ``[1]`` holding the per-tensor scale.
    """
    x = x.contiguous()
    n_elements = x.numel()
    P = _NUM_PARTIALS
    partial = torch.empty(P, dtype=torch.float32, device=x.device)
    out = torch.empty(1, dtype=torch.float32, device=x.device)
    _amax_partial_kernel[(P,)](x, partial, n_elements, P, BLOCK=_STAGE1_BLOCK, num_warps=4)
    _scale_finalize_kernel[(1,)](
        partial,
        out,
        P=P,
        INV_FP8_MAX=1.0 / FP8_E4M3_MAX,
        MIN_SF=MIN_SCALING_FACTOR,
        ROUND_DTYPE=_ROUND_DTYPE[x.dtype],
        num_warps=1,
    )
    return out


def fp8_dynamic_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-tensor FP8 (e4m3) quantization.

    Computes the scale with the fused Triton kernel
    (:func:`fp8_dynamic_quant_scale`), then applies it with the existing
    apply-only CUDA op, whose output is bitwise-identical to
    ``torch.ops.tensorrt_llm.quantize_e4m3_per_tensor`` for the same scale.

    Args:
        x: Activation tensor to quantize.

    Returns:
        A ``(qfp8_e4m3, scale)`` tuple: the e4m3 quantized tensor and the fp32
        per-tensor scale of shape ``[1]``.
    """
    scale = fp8_dynamic_quant_scale(x)
    qinput, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, scale)
    return qinput, scale
