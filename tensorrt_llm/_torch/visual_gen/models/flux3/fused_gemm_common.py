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
"""Helpers shared by the FLUX 3 fused block GEMMs (Prims-TS on SM100/103/107, CuTe DSL on SM120).

Fused kernels are picked from the GPU and the checkpoint, not by a user option. Set
``FUSED_GEMMS = False`` before the model loads to keep every projection on TRT-LLM's
GEMMs (tests and A/B runs).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor, nn

from tensorrt_llm._torch.modules.linear import Linear, NVFP4LinearMethod
from tensorrt_llm._torch.utils import compute_swizzled_sf_shape
from tensorrt_llm._torch.visual_gen import prims_ts

FUSED_GEMMS = True

HEAD_DIM = 128  # head size of the fused QK-norm + RoPE epilogues
RMS_EPS = 1e-6  # their fixed RMSNorm epsilon

# Set on a gate/up Linear whose rows were repacked to [up, gate] pairs for a fused SwiGLU
# epilogue; the Linear then no longer computes gate/up correctly on its own.
SWIGLU_PAIR_LAYOUT = "swiglu_pair_layout"

# (cluster, tmem_overlap, ab_stages, epilogue_warps, use_tma_store) per M bucket at the
# FLUX 3 klein shapes (hidden 3072, MLP 9216).
prims_ts.register_tactics(
    "fp8_e4m3",
    {
        ("swiglu", 3072, 18432): {
            1024: (2, False, 6, 8, False),
            8192: (4, False, 6, 8, False),
            10240: (4, False, 5, 8, False),
            32768: (2, False, 6, 8, False),
            40960: (2, False, 5, 8, False),
        },
        ("qkv_qknorm_rope", 3072, 9216): {
            1024: (2, False, 4, 8, True),
            8192: (2, False, 4, 8, True),
            10240: (2, False, 4, 8, True),
            32768: (4, False, 4, 8, True),
            40960: (4, False, 4, 8, True),
        },
    },
)
prims_ts.register_tactics(
    "nvfp4_e2m1",
    {
        ("qkv_qknorm_rope", 3072, 9216): {
            1024: (2, True, 4, 8, False),
            8192: (2, True, 4, 8, False),
            10240: (2, True, 4, 8, False),
            32768: (4, True, 4, 8, False),
            40960: (4, True, 4, 8, False),
        },
    },
)


def nvfp4_scale_bytes(rows: int, k: int) -> int:
    """Bytes of the padded 128x4-swizzled E4M3 block scales of a [rows, k] NVFP4 matrix."""
    return math.prod(compute_swizzled_sf_shape(rows, k // 16))


def nvfp4_projection_eligible(layer: nn.Module) -> bool:
    """Static NVFP4 projection whose checkpoint scales a fused GEMM can consume."""
    return (
        isinstance(layer, Linear)
        and type(layer.quant_method) is NVFP4LinearMethod
        and layer.bias is None
        and layer.pre_quant_scale is None
        and not layer.force_dynamic_quantization
        and getattr(layer, "input_scale", None) is not None
        and layer.input_scale.numel() == 1
        and layer.weight_scale_2.numel() == 1
        and layer.weight.dtype == torch.uint8
        and layer.weight.shape[0] % 128 == 0
        and layer.weight.shape[1] % 128 == 0
        and layer.weight_scale.numel()
        == nvfp4_scale_bytes(layer.weight.shape[0], layer.weight.shape[1] * 2)
        and layer.dtype in (None, torch.bfloat16)
    )


def qk_norm_epilogue_eligible(attn: nn.Module) -> bool:
    """Whether ``attn``'s QK-norm matches the fused epilogue (head 128, eps 1e-6, BF16)."""
    norms = (attn.norm_q, attn.norm_k)
    return (
        attn.head_dim == HEAD_DIM
        and attn.local_num_attention_heads == attn.local_num_key_value_heads
        and all(
            math.isclose(norm.variance_epsilon, RMS_EPS)
            and norm.weight.dtype == torch.bfloat16
            and tuple(norm.weight.shape) == (HEAD_DIM,)
            for norm in norms
        )
    )


def rope_compatible(x: Tensor, cos: Tensor) -> bool:
    """Whether ``cos`` holds one 128-wide row per token of ``x``, or per position of a batch."""
    rows = x.numel() // x.shape[-1] if x.numel() else 0
    if rows == 0 or cos.shape[-1] != HEAD_DIM:
        return False
    table_rows = cos.numel() // HEAD_DIM
    return table_rows == rows or (
        x.dim() == 3 and cos.shape[0] == 1 and table_rows * x.shape[0] == rows
    )


def fused_input_ok(x: Tensor, cos: Tensor, width: int, device: torch.device) -> bool:
    """Whether a fused block GEMM can take ``x``: CUDA BF16 [..., width] on ``device``.

    Autograd must be off, and ``cos`` must be a RoPE table that matches the tokens of ``x``.
    """
    return (
        not torch.is_grad_enabled()
        and x.is_cuda
        and x.device == device
        and x.dtype == torch.bfloat16
        and x.shape[-1] == width
        and rope_compatible(x, cos)
    )


def rope_table(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """FP32 [rows, 128] table of half cos | half sin for the fused epilogues.

    ``cos``/``sin`` are the interleaved FLUX tables (``rope_compatible`` with ``x``); a table
    for one sequence is broadcast over the batch.
    """
    if cos.numel() // HEAD_DIM != x.numel() // x.shape[-1]:
        cos = cos.expand(x.shape[0], *cos.shape[1:])
        sin = sin.expand(x.shape[0], *sin.shape[1:])
    table = torch.cat(
        (cos.reshape(-1, HEAD_DIM)[:, ::2], sin.reshape(-1, HEAD_DIM)[:, ::2]), dim=-1
    )
    return table.float().contiguous()


def pack_nvfp4_qkv(projections: Tuple[Linear, Linear, Linear]) -> Tuple[Tensor, Tensor, Tensor]:
    """Stack static NVFP4 q, k, v weights and block scales into one q|k|v projection.

    Row counts are multiples of 128, so the 128x4-swizzled block scales of the stacked
    projections are the per-projection buffers back to back. The TRT-LLM projections are
    repointed to views of the packed buffers, so no weight is stored twice and their own
    forward still works. Also returns ``qkv_scale``, the FP32 [3] per-plane dequantization
    factors (each projection's ``alpha``).
    """
    weight = torch.cat([p.weight.detach() for p in projections], dim=0)
    weight_scale = torch.cat(
        [p.weight_scale.detach().view(torch.uint8).reshape(-1) for p in projections]
    )
    rows, scale_bytes = projections[0].weight.shape[0], projections[0].weight_scale.numel()
    for i, p in enumerate(projections):
        p.weight.data = weight[i * rows : (i + 1) * rows]
        p.weight_scale.data = weight_scale[i * scale_bytes : (i + 1) * scale_bytes].view(
            p.weight_scale.dtype
        )
    qkv_scale = torch.cat([p.alpha.detach().float().reshape(1) for p in projections])
    return weight, weight_scale, qkv_scale


def swiglu_pairs(t: Tensor) -> Tensor:
    """[gate; up] rows to adjacent [up_i, gate_i] pairs, the order fused SwiGLU epilogues read."""
    half = t.shape[0] // 2
    return torch.stack((t[half:], t[:half]), dim=1).reshape_as(t).contiguous()


def plain_swiglu(ff: nn.Module) -> bool:
    """Whether a GatedMLP computes plain silu(gate) * up from one fused gate/up projection."""
    return (
        ff.activation is torch.nn.functional.silu
        and ff._is_plain_swiglu()
        and ff.swiglu_limit is None
        and not ff.split_gate_up
    )
