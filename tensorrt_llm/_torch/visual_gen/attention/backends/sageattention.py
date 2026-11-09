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
"""SageAttention backend for visual generation models (SM120).

Wraps https://github.com/thu-ml/SageAttention, which quantizes Q/K to INT8 and
P/V to FP8 inside the attention kernel. Only the SM120 (RTX PRO 6000 Blackwell
Server Edition / RTX 50xx) kernel is wired up here.

Expects NHD layout (``[B, S, H, D]``) and FP16/BF16 inputs. The package is an
optional dependency: ``pip install sageattention`` (build from source for SM120).
"""

import math
from typing import Optional

import torch

from ....attention.backends.interface import PredefinedAttentionMask
from .interface import AttentionBackend, AttentionTensorLayout

_sageattn_import_error = None
try:
    from sageattention import sageattn_qk_int8_pv_fp8_cuda as _sageattn_sm120
except (AttributeError, ImportError, OSError) as e:
    _sageattn_sm120 = None
    _sageattn_import_error = e

_SUPPORTED_HEAD_DIMS = (64, 96, 128)


class SageAttention(AttentionBackend):
    """SageAttention (INT8 QK / FP8 PV) for self- and cross-attention on SM120."""

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 64,
        num_kv_heads: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.dtype = dtype
        self.scale = 1.0 / math.sqrt(head_dim)
        self._preferred_layout = AttentionTensorLayout.NHD

        if head_dim not in _SUPPORTED_HEAD_DIMS:
            raise ValueError(
                f"SAGEATTN supports head_dim in {_SUPPORTED_HEAD_DIMS}, got {head_dim}."
            )
        if self.num_heads != self.num_kv_heads:
            raise ValueError("SAGEATTN does not support GQA/MQA (num_kv_heads != num_heads).")

    @torch.compiler.disable
    def _fwd(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
        return _sageattn_sm120(
            q,
            k,
            v,
            tensor_layout="NHD",
            is_causal=causal,
            qk_quant_gran="per_warp",
            sm_scale=self.scale,
            pv_accum_dtype="fp32+fp32",
            smooth_k=True,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Attention over NHD tensors.

        Args:
            q: ``[B, S, H, D]``
            k: ``[B, S_kv, H, D]``
            v: ``[B, S_kv, H, D]``
            attention_mask: CAUSAL or FULL.
            key_padding_mask: Not supported.

        Returns:
            ``[B, S, H, D]`` in the input dtype.
        """
        if _sageattn_sm120 is None:
            raise ImportError(
                "SageAttention is not available (pip install sageattention). "
                f"Import error: {_sageattn_import_error}"
            ) from _sageattn_import_error
        if key_padding_mask is not None:
            raise ValueError("SAGEATTN does not support key_padding_mask.")
        if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device)[0] != 12:
            raise RuntimeError("SAGEATTN backend is only wired up for SM120 GPUs.")
        if q.dim() != 4 or q.shape[3] != self.head_dim:
            raise ValueError(
                f"Invalid q shape: expected [B, S, H, D={self.head_dim}], got {tuple(q.shape)}"
            )

        is_causal = attention_mask == PredefinedAttentionMask.CAUSAL
        origin_dtype = q.dtype
        if origin_dtype not in (torch.float16, torch.bfloat16):
            q, k, v = (t.to(torch.bfloat16) for t in (q, k, v))
        out = self._fwd(q, k, v, is_causal)
        return out if out.dtype == origin_dtype else out.to(origin_dtype)

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return self._preferred_layout

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False
