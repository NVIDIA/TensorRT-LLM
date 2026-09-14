# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from enum import IntEnum
import torch
from typing import Optional, Union

from flashinfer.api_logging import flashinfer_api


# The type of method in top-K routing, for use in torch custom op
# Please keep this in sync with the counterpart defined in include/flashinfer/trtllm/fused_moe/runner.h
class RoutingMethodType(IntEnum):
    # Default: Softmax -> TopK
    Default = (0,)
    # Renormalize: TopK -> Softmax
    Renormalize = (1,)
    # DeepSeekV3: Sigmoid -> RoutingBiasAdd -> Top2 in group -> Top4 groups -> Top8 experts from the Top4 groups
    DeepSeekV3 = (2,)
    # Llama4: Top1 -> Sigmoid
    Llama4 = (3,)
    # Qwen3: Softmax -> TopK -> Renormalize
    RenormalizeNaive = (4,)
    # TopK only (no softmax)
    TopK = (5,)
    # SigmoidRenorm: Sigmoid -> TopK -> Renormalize (divide by sum of top-K weights)
    SigmoidRenorm = (6,)
    # MiniMax2: Sigmoid + Bias -> TopK -> ScaledSumNormalize (routeScale=1.0, epsilon=1e-20)
    MiniMax2 = (7,)
    # Sigmoid: Sigmoid -> TopK (no renormalization)
    Sigmoid = (8,)
    # TopKSigmoid: TopK -> Sigmoid (no renormalization)
    TopKSigmoid = (9,)
    # Unspecified
    Unspecified = (10,)

    # Eval-safe repr (``RoutingMethodType.Default`` rather than IntEnum's default
    # ``<RoutingMethodType.Default: 0>``) so configs that embed this member
    # round-trip through ``eval(repr(cfg))`` — relied on by the unified MoE API.
    def __repr__(self) -> str:
        return f"{type(self).__name__}.{self.name}"


# Routing input modes for FusedMoE launcher
# Please keep this in sync with the counterpart defined in csrc/trtllm_fused_moe_kernel_launcher.cu
class RoutingInputMode(IntEnum):
    # Mode 1: Compute routing from logits
    # - Input: routing_logits tensor provided
    # - topk_ids: OUTPUT buffer for computed expert indices
    # - topk_weights: OUTPUT buffer for computed weights
    FromLogits = 0
    # Mode 2: Pre-computed routing with packed format
    # - Input: topk_ids contains packed ``(expert_id << 16) | weight`` (high
    #   16 bits = int16 expert id, low 16 bits = float16/bfloat16 weight, see
    #   PackedScoreIdx in include/flashinfer/trtllm/fused_moe/RoutingKernel.h)
    # - topk_ids: INPUT with packed values
    # - topk_weights: OUTPUT buffer for extracted weights
    PackedPrecomputed = 1
    # Mode 3: Pre-computed routing with separate tensors
    # - Input: separate topk_ids (expert indices) and topk_weights (routing weights)
    # - topk_ids: INPUT - pre-computed expert indices
    # - topk_weights: INPUT - pre-computed routing weights
    UnpackedPrecomputed = 2

    # Eval-safe repr — see ``RoutingMethodType.__repr__``.
    def __repr__(self) -> str:
        return f"{type(self).__name__}.{self.name}"


# Copied from csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/common.h
class ActivationType(IntEnum):
    Gelu = 0
    Relu = 1
    Silu = 2
    Swiglu = 3
    Geglu = 4
    SwigluBias = 5
    Relu2 = 6
    SwigluStep = 7
    GegluTanh = 8
    Identity = 9
    Situ = 10
    InvalidType = 11

    # Eval-safe repr — see ``RoutingMethodType.__repr__``.
    def __repr__(self) -> str:
        return f"{type(self).__name__}.{self.name}"

    @property
    def is_gated(self) -> bool:
        """True for activations that consume a gate branch (SwiGLU family)."""
        return self in _GATED_ACTIVATION_TYPES


_GATED_ACTIVATION_TYPES = (
    ActivationType.Swiglu,
    ActivationType.Geglu,
    ActivationType.SwigluBias,
    ActivationType.SwigluStep,
    ActivationType.GegluTanh,
    ActivationType.Situ,
)


DEFAULT_SWIGLU_ALPHA = 1.0
DEFAULT_SWIGLU_BETA = 0.0
DEFAULT_SWIGLU_LIMIT = torch.finfo(torch.float32).max

# SiTU-GLU tanh scales. Must match the SituAdaptor defaults in
# csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh.
DEFAULT_SITU_BETA = 4.0
DEFAULT_SITU_LINEAR_BETA = 25.0


def normalize_activation_type(
    activation_type: Union[int, ActivationType],
) -> ActivationType:
    try:
        return ActivationType(activation_type)
    except ValueError as err:
        raise ValueError(f"Unsupported activation_type {activation_type!r}") from err


@flashinfer_api
def is_gated_activation(activation_type: Union[int, ActivationType]) -> bool:
    """Return whether the given activation type is a gated activation (e.g. SwiGLU family).

    Gated activations split their input along the feature dimension into a *gate* branch
    and a *value* branch; the two are combined element-wise before being passed to the
    next layer.  This helper mirrors the C++ ``isGatedActivation()`` predicate defined in
    ``include/flashinfer/trtllm/fused_moe/runner.h``.

    Parameters
    ----------
    activation_type : Union[int, ActivationType]
        The activation type to query.  May be an :class:`ActivationType` member or its
        integer value.

    Returns
    -------
    bool
        ``True`` if ``activation_type`` belongs to the gated activation family
        (``Swiglu``, ``Geglu``, ``SwigluBias``, ``SwigluStep``, ``GegluTanh``, ``Situ``);
        ``False`` otherwise.

    Examples
    --------
    >>> from tensorrt_llm._torch.moe.flashinfer.tllm_enums import ActivationType, is_gated_activation
    >>> is_gated_activation(ActivationType.Swiglu)
    True
    >>> is_gated_activation(ActivationType.Relu)
    False
    """
    # Keep this in sync with isGatedActivation() in include/flashinfer/trtllm/fused_moe/runner.h.
    return normalize_activation_type(activation_type) in _GATED_ACTIVATION_TYPES


class DtypeTrtllmGen(IntEnum):
    def __new__(cls, block_format_bit, signed_bit, integer_bit, num_bits, uid):
        value = (
            (block_format_bit << 24)
            | (signed_bit << 20)
            | (integer_bit << 16)
            | (num_bits << 8)
            | uid
        )
        obj = int.__new__(cls, value)
        obj._value_ = value
        return obj

    # keep the values in sync with include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h
    Bfloat16 = (0, 1, 0, 16, 0)
    Bool = (0, 0, 1, 1, 1)
    E2m1 = (1, 1, 0, 4, 2)
    E2m3 = (1, 1, 0, 6, 3)
    E3m2 = (1, 1, 0, 6, 4)
    E4m3 = (0, 1, 0, 8, 5)
    E5m2 = (0, 1, 0, 8, 6)
    Fp16 = (0, 1, 0, 16, 7)
    Fp32 = (0, 1, 0, 32, 8)
    Int8 = (0, 1, 1, 8, 9)
    Int32 = (0, 1, 1, 32, 10)
    Int64 = (0, 1, 1, 64, 11)
    MxE2m1 = (1, 1, 0, 4, 12)
    MxE4m3 = (1, 1, 0, 8, 13)
    MxInt4 = (1, 1, 1, 4, 14)
    UE8m0 = (0, 0, 0, 8, 15)
    UInt8 = (0, 0, 1, 8, 16)
    UInt16 = (0, 0, 1, 16, 17)
    UInt32 = (0, 0, 1, 32, 18)
    UInt64 = (0, 0, 1, 64, 19)
    UInt128 = (0, 0, 1, 128, 20)
    Void = (0, 1, 0, 0, 21)


def trtllm_gen_dtype_has_scale(dtype: DtypeTrtllmGen) -> bool:
    if dtype in [
        DtypeTrtllmGen.E2m1,
        DtypeTrtllmGen.MxE2m1,
        DtypeTrtllmGen.MxE4m3,
        DtypeTrtllmGen.MxInt4,
    ]:
        return True
    else:
        return False


def deduce_trtllm_gen_tensor_dtype(
    x: torch.Tensor, scale: Optional[torch.Tensor]
) -> DtypeTrtllmGen:
    x_numel = x.numel()
    if x.dtype == torch.uint8:  # FIXME(siyuan): use torch.float4_e2m1x2 after torch 2.8
        x_numel *= 2
    if x.dtype == torch.bfloat16:
        dtype = DtypeTrtllmGen.Bfloat16
    elif x.dtype == torch.float8_e4m3fn:
        dtype = DtypeTrtllmGen.E4m3 if scale is None else DtypeTrtllmGen.MxE4m3
    elif (
        x.dtype == torch.uint8
    ):  # FIXME(siyuan): use torch.float4_e2m1x2 after torch 2.8
        assert scale is not None, "Scale tensor must be provided for float4x2 input"
        if scale.numel() == x_numel // 16:
            dtype = DtypeTrtllmGen.E2m1
        else:
            dtype = DtypeTrtllmGen.MxE2m1
    else:
        raise ValueError("Unsupported trtllm-gen input tensor.")
    return dtype


# Please keep the values in sync with include/flashinfer/fp4_layout.cuh
class SfLayout(IntEnum):
    """
    Layout of scale factors for quantization.
    """

    layout_128x4 = 0
    layout_8x4 = 1
    layout_linear = 2


# See MatrixLayout from include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/Enums.h
class WeightLayout(IntEnum):
    # K-major layout (default). [Mn, K]
    MajorK = 0
    # M-major for A and N-major for B. [K, Mn]
    MajorMn = 1
    # Layout is blocked along the K dimension. [K / blockK, Mn, blockK]
    # where blockK is fixed at 128B
    BlockMajorK = 2


# The type of gated activation function
# Please keep this in sync with the counterpart defined in include/flashinfer/trtllm/fused_moe/runner.h
class GatedActType(IntEnum):
    # SwiGlu
    SwiGlu = 0
    # GeGlu
    GeGlu = 1


# The type of FP8 quantization
# Please keep this in sync with the counterpart defined in trtllm_fused_moe_kernel_launcher.cu
class Fp8QuantizationType(IntEnum):
    # No FP8 quantization
    NoneFp8 = 0
    # DeepSeek FP8
    DeepSeekFp8 = 1
    # MxFp8 x MxFp8
    MxFp8 = 2
    # Per-tensor FP8
    PerTensorFp8 = 3
    # Per-channel FP8
    PerChannelFp8 = 4
