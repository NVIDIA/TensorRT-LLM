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
"""FLUX 3 static-NVFP4 fused block GEMMs on the SM120/SM121 CuTe DSL kernels.

FlashInfer Prims-TS and TRT-LLM's fused NVFP4 SwiGLU GEMM run only on datacenter
Blackwell. On SM120/SM121 these kernels fuse what TRT-LLM runs as separate kernels: Q/K/V
in one GEMM with per-plane scales, QK-norm and RoPE; and gate/up with SwiGLU and NVFP4
requantization for the down projection.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from tensorrt_llm._torch.modules.linear import NVFP4LinearMethod
from tensorrt_llm._torch.utils import Fp4QuantizedTensor, unswizzle_sf
from tensorrt_llm._torch.visual_gen import prims_ts
from tensorrt_llm._torch.visual_gen.models.flux3.fused_gemm_common import (
    SWIGLU_PAIR_LAYOUT,
    fused_input_ok,
    nvfp4_projection_eligible,
    nvfp4_scale_bytes,
    pack_nvfp4_qkv,
    plain_swiglu,
    qk_norm_epilogue_eligible,
    rope_table,
    swiglu_pairs,
)
from tensorrt_llm._torch.visual_gen.models.flux3.packed import Bounds
from tensorrt_llm.math_utils import pad_up

_SM120 = ((12, 0), (12, 1))


def sm120_device(device: torch.device) -> bool:
    """Whether ``device`` is an SM120/SM121 GPU (the CuTe DSL kernels' targets)."""
    return device.type == "cuda" and torch.cuda.get_device_capability(device) in _SM120


@lru_cache(maxsize=None)
def sm120_kernels_available() -> bool:
    try:
        from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import (  # noqa: F401
            flux3_sm120_nvfp4,
        )
    except (ImportError, OSError):  # CuTe DSL unavailable
        return False
    return True


def _run(*args, **kwargs):
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.sm120 import flux3_sm120_nvfp4

    return flux3_sm120_nvfp4.run(*args, **kwargs)


@torch.library.custom_op("trtllm::flux3_sm120_nvfp4_qkv", mutates_args=(), device_types="cuda")
def _qkv(
    data: Tensor,
    scales: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    qkv_scale: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    cos_sin: Tensor,
) -> Tensor:
    """Packed q|k|v GEMM with per-plane scales, per-head RMSNorm and RoPE: BF16 [M, N]."""
    return _run(
        "qkv",
        data,
        weight,
        scales,
        weight_scale,
        q_norm=q_norm,
        k_norm=k_norm,
        cos_sin=cos_sin,
        # One position per real row: run() zero-pads them for the padded tile rows, which the
        # kernel would otherwise look up past the end of cos_sin.
        positions=prims_ts.token_positions(data.shape[0], data.device),
        qkv_scale=qkv_scale,
    )


@_qkv.register_fake
def _qkv_fake(data, scales, weight, weight_scale, qkv_scale, q_norm, k_norm, cos_sin):
    return data.new_empty((data.shape[0], weight.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op(
    "trtllm::flux3_sm120_nvfp4_swiglu_fp4", mutates_args=(), device_types="cuda"
)
def _swiglu_fp4(
    data: Tensor,
    scales: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    alpha: Tensor,
    output_inverse_scale: Tensor,
) -> Tuple[Tensor, Tensor]:
    """[up, gate]-paired GEMM, SwiGLU, NVFP4 requantization: ([M, N/4] u8, 128x4 scales)."""
    return _run(
        "packed",
        data,
        weight,
        scales,
        weight_scale,
        alpha,
        output_inverse_scale=output_inverse_scale,
    )


@_swiglu_fp4.register_fake
def _swiglu_fp4_fake(data, scales, weight, weight_scale, alpha, output_inverse_scale):
    rows, columns = data.shape[0], weight.shape[0] // 2
    return (
        data.new_empty((rows, columns // 2), dtype=torch.uint8),
        data.new_empty((nvfp4_scale_bytes(pad_up(rows, 128), columns),), dtype=torch.uint8),
    )


def repack_swiglu_rows(gate_up: nn.Module) -> None:
    """Repack a static NVFP4 [gate; up] Linear in place to [up, gate] row pairs. Idempotent."""
    if getattr(gate_up, SWIGLU_PAIR_LAYOUT, False):
        return
    weight = gate_up.weight.detach()
    rows = weight.shape[0]
    scales = unswizzle_sf(gate_up.weight_scale.detach(), rows, weight.shape[1] * 2)[:rows]
    gate_up.weight.data = swiglu_pairs(weight)
    gate_up.weight_scale.data = (
        torch.ops.trtllm.block_scale_interleave(swiglu_pairs(scales))
        .view(gate_up.weight_scale.dtype)
        .reshape(gate_up.weight_scale.shape)
    )
    setattr(gate_up, SWIGLU_PAIR_LAYOUT, True)


class Sm120NVFP4Block(nn.Module):
    """Fused NVFP4 QKV and gate/up GEMMs of one FLUX 3 block on SM120 kernels.

    Q/K/V are packed into one buffer that the TRT-LLM projections view; gate/up is repacked
    in place.
    """

    # TODO: run the fused QKV and gate/up paths through Flux3Attention and GatedMLP
    # methods instead of calling the ops from the block, so Linear.forward wrappers see
    # every projection (only attention-out and down do today).

    def __init__(self, block: nn.Module):
        super().__init__()
        attn, ff = block.attn, block.ff
        # Plain lists: the transformer block owns these modules.
        self._attn = [attn]
        self.hidden = attn.hidden_size
        projections = (attn.to_q, attn.to_k, attn.to_v)
        weight, weight_scale, qkv_scale = pack_nvfp4_qkv(projections)
        self.register_buffer("qkv_weight", weight, persistent=False)
        self.register_buffer("qkv_weight_scale", weight_scale, persistent=False)
        # Per-plane dequantization factors stay on the device, so nothing is captured by value.
        self.register_buffer("qkv_scale", qkv_scale, persistent=False)
        gate_up = ff.gate_up_proj
        repack_swiglu_rows(gate_up)
        self._layers = [attn.to_q, attn.to_out[0], gate_up, ff.down_proj]
        self._share_quant = bool(torch.equal(attn.to_q.input_scale, gate_up.input_scale))

    def can_run(self, x: Tensor, cos: Tensor) -> bool:
        return fused_input_ok(x, cos, self.hidden, self.qkv_weight.device)

    def forward(
        self,
        x: Tensor,
        image_rotary_emb: Tuple[Tensor, Tensor],
        sample_bounds: Optional[Bounds],
        timestep: Optional[Tensor],
    ) -> Tensor:
        """Attention output projection + MLP of the modulated block input ``x`` [B, S, hidden]."""
        attn = self._attn[0]
        to_q, out_proj, gate_up, down_proj = self._layers
        cos, sin = image_rotary_emb
        rows = x.reshape(-1, x.shape[-1]).contiguous()
        data, scales, _ = to_q.quant_method._input_prepare(to_q, rows)
        qkv = torch.ops.trtllm.flux3_sm120_nvfp4_qkv(
            data,
            scales.reshape(-1),
            self.qkv_weight,
            self.qkv_weight_scale,
            self.qkv_scale,
            attn.norm_q.weight,
            attn.norm_k.weight,
            rope_table(x, cos, sin),
        )
        width = self.qkv_weight.shape[0] // 3
        q, k, v = (t.view(*x.shape[:-1], width) for t in qkv.split(width, dim=-1))
        output = out_proj(attn.attend_projected(q, k, v, sample_bounds, timestep))
        if not self._share_quant:
            data, scales, _ = gate_up.quant_method._input_prepare(gate_up, rows)
        hidden, hidden_scales = torch.ops.trtllm.flux3_sm120_nvfp4_swiglu_fp4(
            data,
            scales.reshape(-1),
            gate_up.weight,
            gate_up.weight_scale,
            gate_up.alpha,
            down_proj.input_scale,
        )
        down = down_proj(Fp4QuantizedTensor(hidden, hidden_scales, is_sf_swizzled=True))
        return output + down.view_as(output)


def prepare_sm120_nvfp4(block: nn.Module, enabled: bool) -> Optional[Sm120NVFP4Block]:
    """Attach the fused SM120 NVFP4 QKV and gate/up GEMMs to ``block`` when they qualify."""
    block.sm120_nvfp4 = None
    attn, ff = block.attn, block.ff
    gate_up = ff.gate_up_proj
    nvfp4 = type(getattr(gate_up, "quant_method", None)) is NVFP4LinearMethod
    if nvfp4 and getattr(gate_up, SWIGLU_PAIR_LAYOUT, False):
        # Already repacked: GatedMLP cannot run this block any more.
        block.sm120_nvfp4 = Sm120NVFP4Block(block)
        return block.sm120_nvfp4
    if not enabled or getattr(attn, "tp_size", 1) != 1:
        return None
    projections = tuple(getattr(attn, name, None) for name in ("to_q", "to_k", "to_v"))
    layers = (*projections, gate_up, ff.down_proj)
    if any(layer is None for layer in layers) or not all(map(nvfp4_projection_eligible, layers)):
        return None
    if not sm120_device(gate_up.weight.device) or not qk_norm_epilogue_eligible(attn):
        return None
    if any(layer.alpha.numel() != 1 for layer in (*projections, gate_up)):
        return None
    if len({float(p.input_scale.item()) for p in projections}) != 1:
        # One activation quantize feeds all three Q/K/V planes.
        return None
    if not plain_swiglu(ff) or not sm120_kernels_available():
        return None
    block.sm120_nvfp4 = Sm120NVFP4Block(block)
    return block.sm120_nvfp4
