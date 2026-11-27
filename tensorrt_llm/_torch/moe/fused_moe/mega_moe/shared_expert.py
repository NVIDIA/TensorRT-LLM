# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The FP8 shared expert fused into the MegaMoE CuteDSL kernel: weight layout, input buffers and the
standalone MXFP8 quantization of its input."""

from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm.math_utils import ceil_div, pad_up
from tensorrt_llm.quantization.utils.fp8_utils import unpack_col_major_tma_aligned_packed_tensor

# The routed FC1 interleaves gate and up rows in groups of this many.
_GATE_UP_INTERLEAVE = 16
# MXFP8 scale-factor block and the checkpoint's block-scale granularity.
_MXFP8_BLOCK = 32
_FP8_BLOCK = 128
# Gate/up and down-projection outputs tile clusters of up to four 128-row CTAs.
_CLUSTER_ROWS = 512

# Per-(device, bucket rows, hidden) MXFP8 activation buffers, shared by every MoE layer.
_ACTIVATION_BUFFERS: Dict[Tuple[str, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}


def shared_activation_buffers(
    max_tokens: int, hidden: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(E4M3 rows, UE8M0 scales in 128x4 atoms) holding one launch bucket's MXFP8 input."""
    key = (str(device), int(max_tokens), int(hidden))
    buffers = _ACTIVATION_BUFFERS.get(key)
    if buffers is None:
        buffers = (
            torch.empty((max_tokens, hidden), dtype=torch.float8_e4m3fn, device=device),
            torch.zeros(
                (pad_up(max_tokens, 128) * hidden // _MXFP8_BLOCK,),
                dtype=torch.uint8,
                device=device,
            ),
        )
        _ACTIVATION_BUFFERS[key] = buffers
    return buffers


@triton.jit
def _mxfp8_quant_kernel(
    x_ptr, xq8_ptr, xsf8_ptr, stride_x, stride_xq8, HIDDEN: tl.constexpr, CHUNK: tl.constexpr
):
    """MXFP8 of one chunk: E4M3 data and per-32 UE8M0 scales, rounded up like cvt.rp.ue8m0,
    stored in the 128x4 scale-atom layout of the MegaMoE MMA."""
    row = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * CHUNK
    xv = tl.load(x_ptr + row * stride_x + c + tl.arange(0, CHUNK)).to(tl.float32)
    blocks: tl.constexpr = CHUNK // 32
    xb = tl.reshape(xv, (blocks, 32))
    bits = (tl.max(tl.abs(xb), axis=1) * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    e8 = tl.minimum((bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.int32), 254)
    # 2**(127 - e8): the exact reciprocal of the decode scale.
    encode = ((254 - e8) << 23).to(tl.float32, bitcast=True)
    q = tl.reshape((xb * encode[:, None]).to(tl.float8e4nv), (CHUNK,))
    tl.store(xq8_ptr + row * stride_xq8 + c + tl.arange(0, CHUNK), q.to(tl.uint8, bitcast=True))
    col = c // 32 + tl.arange(0, blocks)
    k_atoms: tl.constexpr = HIDDEN // 128
    offset = (
        (row // 128) * (k_atoms * 512)
        + (col // 4) * 512
        + (row % 32) * 16
        + ((row // 32) % 4) * 4
        + col % 4
    )
    tl.store(xsf8_ptr + offset, e8.to(tl.uint8))


def mxfp8_quantize_megamoe(x: torch.Tensor, xq8: torch.Tensor, xsf8: torch.Tensor) -> None:
    """Write MXFP8 ``x`` (M > 0 contiguous rows, hidden a multiple of 512) into ``xq8[:M]`` (E4M3)
    and ``xsf8`` (UE8M0, 128x4 scale atoms), buffers of a launch bucket that holds the M rows."""
    m, hidden = x.shape
    _mxfp8_quant_kernel[(m, hidden // 512)](
        x,
        xq8.view(torch.uint8),
        xsf8,
        x.stride(0),
        xq8.stride(0),
        HIDDEN=hidden,
        CHUNK=512,
        num_warps=4,
    )


def _ue8m0_row_scales(scale: torch.Tensor, rows: int, cols: int) -> Optional[torch.Tensor]:
    """Per-row UE8M0 exponents of 128-wide K blocks, or None unless every scale is a power of two."""
    if scale.dtype == torch.int32:
        # DeepGEMM's packed per-row UE8M0 layout.
        scale = unpack_col_major_tma_aligned_packed_tensor(scale, rows, ceil_div(cols, _FP8_BLOCK))
    elif scale.dtype == torch.float32 and scale.shape == (
        ceil_div(rows, _FP8_BLOCK),
        ceil_div(cols, _FP8_BLOCK),
    ):
        scale = scale.repeat_interleave(_FP8_BLOCK, dim=0)[:rows]
    else:
        return None
    bits = scale.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    is_power_of_two = (bits >= 0) & ((bits & 0x7FFFFF) == 0) & (exponent > 0) & (exponent < 255)
    if not bool(is_power_of_two.all()):
        return None
    return exponent.to(torch.uint8)


def _blocked_scale_factors(row_scales: torch.Tensor) -> torch.Tensor:
    """Per-32 UE8M0 scales in the 128x4 atom layout, padded to whole atoms."""
    rows, cols = row_scales.shape
    padded = torch.nn.functional.pad(row_scales, (0, -cols % 4, 0, -rows % 128))
    blocked = padded.view((rows + 127) // 128, 4, 32, (cols + 3) // 4, 4)
    return blocked.permute(0, 3, 2, 1, 4).reshape(-1).contiguous()


def _interleave_gate_up(tensor: torch.Tensor, intermediate: int) -> torch.Tensor:
    """[gate | up] rows -> [gate 0:16, up 0:16, gate 16:32, ...], the routed FC1 order."""
    groups = intermediate // _GATE_UP_INTERLEAVE
    gate = tensor[:intermediate].reshape(groups, _GATE_UP_INTERLEAVE, *tensor.shape[1:])
    up = tensor[intermediate:].reshape(groups, _GATE_UP_INTERLEAVE, *tensor.shape[1:])
    return torch.stack((gate, up), dim=1).reshape(tensor.shape).contiguous()


def _mega_projection(linear, *, gate_up_intermediate: Optional[int]):
    """FP8 weight and blocked per-32 scales of one block-scaled FP8 Linear, or None."""
    weight = getattr(linear, "weight", None)
    scale = getattr(linear, "weight_scale", None)
    if weight is None or scale is None or weight.dtype != torch.float8_e4m3fn or weight.dim() != 2:
        return None
    if getattr(linear, "bias", None) is not None:
        return None
    rows, cols = weight.shape
    if cols % _FP8_BLOCK:
        return None
    row_scales = _ue8m0_row_scales(scale, rows, cols)
    if row_scales is None:
        return None
    row_scales = row_scales.repeat_interleave(_FP8_BLOCK // _MXFP8_BLOCK, dim=1)
    row_scales = row_scales[:, : cols // _MXFP8_BLOCK]
    if gate_up_intermediate is not None:
        weight = _interleave_gate_up(weight, gate_up_intermediate)
        row_scales = _interleave_gate_up(row_scales, gate_up_intermediate)
    return weight.detach().contiguous(), _blocked_scale_factors(row_scales)


def build_shared_expert_weights(shared_experts, hidden_size: int) -> Optional[dict]:
    """MegaMoE-format weights of a gated FP8 block-scale MLP, or None if it cannot be fused.

    The 128x128 block scales become per-row, per-32 UE8M0 scales; every dequantized weight is unchanged.
    """
    gate_up_proj = getattr(shared_experts, "gate_up_proj", None)
    down_proj = getattr(shared_experts, "down_proj", None)
    if gate_up_proj is None or down_proj is None:
        return None
    if getattr(shared_experts, "swiglu_alpha", None) is not None:
        return None
    if getattr(shared_experts, "swiglu_beta", None) is not None:
        return None
    if getattr(shared_experts, "activation", None) is not torch.nn.functional.silu:
        return None
    gate_up_weight = getattr(gate_up_proj, "weight", None)
    if gate_up_weight is None or gate_up_weight.dim() != 2:
        return None
    gate_up_rows, gate_up_cols = gate_up_weight.shape
    intermediate = gate_up_rows // 2
    if (
        gate_up_cols != hidden_size
        or gate_up_rows % _CLUSTER_ROWS
        or hidden_size % _CLUSTER_ROWS
        or tuple(down_proj.weight.shape) != (hidden_size, intermediate)
    ):
        return None
    fc1 = _mega_projection(gate_up_proj, gate_up_intermediate=intermediate)
    fc2 = _mega_projection(down_proj, gate_up_intermediate=None)
    if fc1 is None or fc2 is None:
        return None
    limit = getattr(shared_experts, "swiglu_limit", None)
    # A non-positive limit disables the clamp, as in the unfused SwiGLU kernel.
    if limit is not None and limit <= 0:
        limit = None
    return {
        "fc1_weight": fc1[0],
        "fc1_weight_sf": fc1[1],
        "fc2_weight": fc2[0],
        "fc2_weight_sf": fc2[1],
        "intermediate_size": intermediate,
        "gate_up_clamp": None if limit is None else float(limit),
    }


__all__ = ["build_shared_expert_weights", "mxfp8_quantize_megamoe", "shared_activation_buffers"]
