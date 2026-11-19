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
"""FLUX 3 static-NVFP4 Q/K/V projection fused with QK-norm and RoPE on FlashInfer Prims-TS.

A static NVFP4 checkpoint keeps separate q/k/v weight scales, so TRT-LLM runs the
projection as three GEMMs followed by QK-norm and RoPE. The Prims-TS GEMM takes a scale
per Q/K/V plane, so one GEMM produces normalized, rotated Q, K and V.
"""

from __future__ import annotations

import math
import weakref
from typing import TYPE_CHECKING, Optional, Tuple

import torch
from torch import Tensor, nn

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm._torch.visual_gen import prims_ts
from tensorrt_llm._torch.visual_gen.models.flux3.fused_gemm_common import (
    HEAD_DIM,
    fused_input_ok,
    nvfp4_projection_eligible,
    pack_nvfp4_qkv,
    qk_norm_epilogue_eligible,
    rope_table,
)

if TYPE_CHECKING:
    from flashinfer.prims_ts.gemm import PreparedFp4Linear

# Prepared projections by packed-weight storage, so the custom op takes only tensors.
_INSTANCES: "weakref.WeakSet[PrimsTsQKV]" = weakref.WeakSet()
_BY_STORAGE: "weakref.WeakValueDictionary[int, PrimsTsQKV]" = weakref.WeakValueDictionary()


def _lookup(weight: Tensor) -> "PrimsTsQKV":
    prepared = _BY_STORAGE.get(weight.data_ptr())
    if prepared is None or prepared.weight.data_ptr() != weight.data_ptr():
        # A device move replaced the storage; re-key every live projection.
        _BY_STORAGE.clear()
        for candidate in list(_INSTANCES):
            _BY_STORAGE[candidate.weight.data_ptr()] = candidate
        prepared = _BY_STORAGE.get(weight.data_ptr())
    if prepared is None:
        raise RuntimeError("FLUX 3 Prims-TS QKV weight has no prepared projection")
    return prepared


class PrimsTsQKV(nn.Module):
    """Packed q|k|v NVFP4 weights and the Prims-TS launchers prepared from them."""

    def __init__(self, attn: nn.Module, projections: Tuple[Linear, Linear, Linear]):
        super().__init__()
        input_scale = float(projections[0].input_scale.item())
        weight_global_scales = [float(p.weight_scale_2.item()) for p in projections]
        if not all(math.isfinite(s) and s > 0 for s in (input_scale, *weight_global_scales)):
            raise ValueError("Prims-TS QKV requires finite positive NVFP4 global scales")
        # TRT-LLM's input_scale is the quantize multiplier; the GEMM takes its inverse.
        self.input_global_scale = 1.0 / input_scale
        self.weight_global_scale = weight_global_scales[0]
        self.num_heads = attn.local_num_attention_heads
        weight, weight_scale, qkv_scale = pack_nvfp4_qkv(projections)
        self.register_buffer("weight", weight, persistent=False)
        self.register_buffer("weight_scale", weight_scale, persistent=False)
        # Per-plane dequantization factors (TRT-LLM's Linear.alpha of q, k and v).
        self.register_buffer("qkv_scale", qkv_scale, persistent=False)
        # Plain lists: the attention module owns these.
        self._to_q = [projections[0]]
        self._norms = [attn.norm_q, attn.norm_k]
        self._launchers: dict[int, "PreparedFp4Linear"] = {}
        _INSTANCES.add(self)
        _BY_STORAGE[self.weight.data_ptr()] = self

    def _apply(self, fn, recurse=True):
        # Launchers hold the old storage; rebuild them after .to()/.cuda().
        self._launchers.clear()
        return super()._apply(fn, recurse=recurse)

    @property
    def in_features(self) -> int:
        return self.weight.shape[1] * 2

    @property
    def input_scale(self) -> Tensor:
        return self._to_q[0].input_scale

    def can_run(self, x: Tensor, cos: Tensor) -> bool:
        return fused_input_ok(x, cos, self.in_features, self.weight.device)

    def quantize(self, x: Tensor) -> Fp4QuantizedTensor:
        to_q = self._to_q[0]
        data, scales, _ = to_q.quant_method._input_prepare(
            to_q, x.reshape(-1, x.shape[-1]).contiguous()
        )
        return Fp4QuantizedTensor(data, scales, is_sf_swizzled=True)

    def launcher(self, m: int) -> "PreparedFp4Linear":
        bucket = prims_ts.m_bucket(m)
        launcher = self._launchers.get(bucket)
        if launcher is None:
            from flashinfer.prims_ts import gemm

            n, k = self.weight.shape[0], self.in_features
            launcher = gemm.prepare_fp4_qkv_qknorm_rope(
                self.weight,
                self.weight_scale,
                self.input_global_scale,
                self.weight_global_scale,
                self._norms[0].weight.detach(),
                self._norms[1].weight.detach(),
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_heads,
                head_dim=HEAD_DIM,
                is_neox=False,
                qkv_scale=self.qkv_scale,
                config=prims_ts.config_for(
                    self.weight.device,
                    "nvfp4_e2m1",
                    "qkv_qknorm_rope",
                    m,
                    k,
                    n,
                    HEAD_DIM,
                    qkv_scale=True,
                ),
            )
            self._launchers[bucket] = launcher
        return launcher

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        quantized: Optional[Fp4QuantizedTensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Q, K, V of ``x`` [..., hidden], each [..., heads * 128], normalized and rotated.

        ``cos``/``sin`` are the interleaved FLUX rotary tables. ``quantized`` is ``x`` already
        quantized with this projection's input scale, when the caller shares it.
        """
        if quantized is None:
            quantized = self.quantize(x)
        out = torch.ops.trtllm.flux3_prims_ts_nvfp4_qkv(
            quantized.fp4_tensor,
            quantized.scaling_factor.reshape(-1),
            self.weight,
            rope_table(x, cos, sin),
        )
        shape = (*x.shape[:-1], out.shape[-1])
        return tuple(plane.view(shape) for plane in out.unbind(0))


@torch.library.custom_op("trtllm::flux3_prims_ts_nvfp4_qkv", mutates_args=(), device_types="cuda")
def _nvfp4_qkv(data: Tensor, scales: Tensor, weight: Tensor, cos_sin: Tensor) -> Tensor:
    """Prepared NVFP4 QKV GEMM + QK-norm + RoPE: BF16 [3, M, N/3], the q, k and v planes.

    The prepared projection is looked up by ``weight`` so per-layer host scales stay out of
    the traced graph. The planes are declared as the kernel writes them: viewing an [M, N]
    result as [3, M, N/3] under symbolic M leaves the plane width symbolic for every
    downstream kernel.
    """
    m = data.shape[0]
    out = _lookup(weight).launcher(m)(
        data.view(torch.uint8).contiguous(),
        scales.view(torch.uint8).contiguous(),
        cos_sin=cos_sin,
        positions=prims_ts.token_positions(m, data.device),
    )
    return out.view(3, m, weight.shape[0] // 3)


@_nvfp4_qkv.register_fake
def _nvfp4_qkv_fake(data: Tensor, scales: Tensor, weight: Tensor, cos_sin: Tensor) -> Tensor:
    return data.new_empty((3, data.shape[0], weight.shape[0] // 3), dtype=torch.bfloat16)


def prepare_prims_ts_qkv(attn: nn.Module, enabled: bool) -> Optional[PrimsTsQKV]:
    """Attach a Prims-TS QKV projection to ``attn`` when its checkpoint and GPU allow it.

    Returns the projection, or None when ``attn`` keeps the TRT-LLM path.
    """
    attn.prims_ts_qkv = None
    if not enabled or getattr(attn, "tp_size", 1) != 1:
        return None
    projections = tuple(getattr(attn, name, None) for name in ("to_q", "to_k", "to_v"))
    if any(p is None for p in projections) or not all(map(nvfp4_projection_eligible, projections)):
        return None
    if not prims_ts.device_supported(projections[0].weight.device):
        return None
    if not qk_norm_epilogue_eligible(attn):
        return None
    if len({float(p.input_scale.item()) for p in projections}) != 1:
        # One activation quantize feeds all three planes.
        return None
    if not prims_ts.nvfp4_qkv_available():
        return None
    attn.prims_ts_qkv = PrimsTsQKV(attn, projections)
    return attn.prims_ts_qkv
