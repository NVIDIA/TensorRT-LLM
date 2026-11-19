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
"""FLUX 3 FP8-rowwise QKV + QK-norm + RoPE and gate/up + SwiGLU on FlashInfer Prims-TS.

One per-token quantization of the modulated block input feeds both.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from tensorrt_llm._torch.modules.linear import FP8RowwiseLinearMethod
from tensorrt_llm._torch.visual_gen import prims_ts
from tensorrt_llm._torch.visual_gen.models.flux3.fused_gemm_common import (
    SWIGLU_PAIR_LAYOUT,
    fused_input_ok,
    plain_swiglu,
    qk_norm_epilogue_eligible,
    rope_table,
    swiglu_pairs,
)
from tensorrt_llm._torch.visual_gen.models.flux3.packed import Bounds


class PrimsTsFP8Block(nn.Module):
    """Fused FP8 QKV and gate/up GEMMs of one FLUX 3 block.

    The gate/up weight is repacked in place to the [up, gate] row pairs the SwiGLU epilogue
    reads.
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
        gate_up = ff.gate_up_proj
        if not getattr(gate_up, SWIGLU_PAIR_LAYOUT, False):
            gate_up.weight.data = swiglu_pairs(gate_up.weight.detach())
            gate_up.weight_scale.data = swiglu_pairs(gate_up.weight_scale.detach()).float()
            setattr(gate_up, SWIGLU_PAIR_LAYOUT, True)
        self._layers = [attn.qkv_proj, attn.to_out[0], gate_up, ff.down_proj]

    def can_run(self, x: Tensor, cos: Tensor) -> bool:
        return fused_input_ok(x, cos, self.hidden, self._layers[0].weight.device)

    def forward(
        self,
        x: Tensor,
        image_rotary_emb: Tuple[Tensor, Tensor],
        sample_bounds: Optional[Bounds],
        timestep: Optional[Tensor],
    ) -> Tensor:
        """Attention output projection + MLP of the modulated block input ``x`` [B, S, hidden]."""
        attn = self._attn[0]
        qkv_proj, out_proj, gate_up, down_proj = self._layers
        cos, sin = image_rotary_emb
        data, scales = prims_ts.quantize_per_token(x.reshape(-1, x.shape[-1]))
        planes = torch.ops.trtllm.prims_ts_fp8_qkv_qknorm_rope(
            data,
            qkv_proj.weight,
            scales,
            qkv_proj.weight_scale,
            attn.norm_q.weight,
            attn.norm_k.weight,
            rope_table(x, cos, sin),
        )
        width = qkv_proj.weight.shape[0] // 3
        q, k, v = (t.view(*x.shape[:-1], width) for t in planes.split(width, dim=-1))
        output = out_proj(attn.attend_projected(q, k, v, sample_bounds, timestep))
        hidden = torch.ops.trtllm.prims_ts_fp8_linear_swiglu(
            data, gate_up.weight, scales, gate_up.weight_scale
        )
        output = output + down_proj(hidden).view_as(output)
        return output


def prepare_prims_ts_fp8(block: nn.Module, enabled: bool) -> Optional[PrimsTsFP8Block]:
    """Attach the fused Prims-TS FP8 QKV and gate/up GEMMs to ``block`` when they qualify."""
    block.prims_ts_fp8 = None
    attn, ff = block.attn, block.ff
    if getattr(ff.gate_up_proj, SWIGLU_PAIR_LAYOUT, False) and isinstance(
        ff.gate_up_proj.quant_method, FP8RowwiseLinearMethod
    ):
        # Already repacked: GatedMLP cannot run this block any more.
        block.prims_ts_fp8 = PrimsTsFP8Block(block)
        return block.prims_ts_fp8
    if not enabled or getattr(attn, "tp_size", 1) != 1:
        return None
    fused = (getattr(attn, "qkv_proj", None), ff.gate_up_proj)
    if any(layer is None for layer in fused) or not all(map(prims_ts.fp8_gemm_eligible, fused)):
        return None
    if not qk_norm_epilogue_eligible(attn) or not plain_swiglu(ff):
        return None
    if not prims_ts.fp8_gemm_available():
        return None
    block.prims_ts_fp8 = PrimsTsFP8Block(block)
    return block.prims_ts_fp8
