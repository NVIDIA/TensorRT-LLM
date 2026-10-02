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
"""TransformerEngine FP8 attention backend for visual generation.

Runs TransformerEngine's ``DotProductAttention`` (cuDNN fused FP8 attention)
under ``fp8_autocast`` with a ``DelayedScaling(fp8_dpa=True, fp8_mha=True)``
recipe, in NHD layout (``qkv_format="bshd"``, no transpose).

Inference-correct FP8 scales: ``DelayedScaling`` only refreshes its scales from
the amax history when autograd is enabled (TE runs the scale update in the
autocast exit only under ``torch.is_grad_enabled()``). Diffusion inference runs
grad-free, so the scales would stay at their initial 1.0. E4M3's relative
precision does not depend on the scale inside its normal range, but values
below it lose precision. That hits the attention output when softmax is spread
over many keys: each output row averages many V rows, so its magnitude is about
std(v) / sqrt(seq_len), e.g. ~4e-3 at 75,600 tokens.

This backend therefore writes the forward scales before every call from the
current tensors: Q/K/V = 448 / amax(q, k, v); O = 448 / amax(v), since each
output row is a convex combination of V rows, so |O| <= amax(v); S = 448,
since softmax probabilities lie in [0, 1].
"""

import math
from typing import Optional

import torch

from ...attention.backends.interface import PredefinedAttentionMask
from .interface import AttentionBackend, AttentionTensorLayout

try:
    from transformer_engine.common.recipe import DelayedScaling
    from transformer_engine.pytorch import DotProductAttention, fp8_autocast
except ImportError:  # TE absent: fail at construction, not at import.
    DotProductAttention = None
    DelayedScaling = None
    fp8_autocast = None

_FP8_E4M3_MAX = 448.0
# Forward fp8_meta scale slots of DotProductAttention: Q/K/V input, output O,
# softmax S. Verified at runtime on each module's first call (see _verify_slots),
# so a TE layout change raises instead of silently mis-scaling.
_SLOT_QKV, _SLOT_O, _SLOT_S = 2, 3, 8


class TEAttention(AttentionBackend):
    """FP8 attention via TransformerEngine ``DotProductAttention``.

    No KV cache (diffusion recomputes every step). Always FP8; use another
    backend for BF16 attention. Configure with ``AttentionConfig(backend="TE")``
    and no ``quant_attention_config``: TE owns its FP8 recipe.
    """

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 64,
        num_kv_heads: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        if DotProductAttention is None:
            raise ImportError(
                "TransformerEngine is required for the TE attention backend "
                "(transformer_engine.pytorch.DotProductAttention is not importable)."
            )
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.dtype = dtype
        self.scale = 1.0 / math.sqrt(head_dim)
        self.recipe = DelayedScaling(fp8_dpa=True, fp8_mha=True)
        # DotProductAttention holds FP8 metadata; rebuild only when the
        # (GQA groups, mask) traits change.
        self._attn_op = None
        self._traits = None
        self._slots_verified = False

    def _lazy_init(self, num_gqa_groups: Optional[int], attn_mask_type: str) -> None:
        traits = (num_gqa_groups, attn_mask_type)
        if traits != self._traits:
            self._attn_op = DotProductAttention(
                self.num_heads,
                self.head_dim,
                num_gqa_groups=num_gqa_groups,
                attn_mask_type=attn_mask_type,
                softmax_scale=self.scale,
                qkv_format="bshd",
            )
            self._traits = traits
            self._slots_verified = False

    @torch.no_grad()
    def _verify_slots(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """Check the slot map against the amax TE recorded on its first call."""
        amax = self._attn_op.fp8_meta["scaling_fwd"].amax_history[0].float()
        want_qkv = max(q.abs().amax().item(), k.abs().amax().item(), v.abs().amax().item())
        got_qkv, got_s = amax[_SLOT_QKV].item(), amax[_SLOT_S].item()
        if abs(got_qkv - want_qkv) > 1e-3 * max(want_qkv, 1.0) or not 0.0 < got_s <= 1.0 + 1e-3:
            raise RuntimeError(
                f"TE fp8_meta slot layout mismatch (layer {self.layer_idx}): "
                f"slot {_SLOT_QKV}={got_qkv} vs amax(q, k, v)={want_qkv}; "
                f"slot {_SLOT_S}={got_s} (softmax, expected in (0, 1]). "
                f"Recorded amax: {amax.tolist()}"
            )
        self._slots_verified = True

    @torch.no_grad()
    def _set_forward_scales(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write this call's forward FP8 scales in place (on device, no host sync)."""
        meta = self._attn_op.fp8_meta["scaling_fwd"]
        amax_v = v.abs().amax().float()
        amax_qkv = torch.maximum(torch.maximum(q.abs().amax(), k.abs().amax()).float(), amax_v)
        scale = meta.scale
        scale[_SLOT_QKV] = _FP8_E4M3_MAX / amax_qkv.clamp_min(1e-12)
        scale[_SLOT_O] = _FP8_E4M3_MAX / amax_v.clamp_min(1e-12)
        scale[_SLOT_S] = _FP8_E4M3_MAX
        if getattr(meta, "scale_inv", None) is not None:  # older TE keeps it separately
            meta.scale_inv.copy_(1.0 / scale)

    @torch.compiler.disable
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        *,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """FP8 attention. q/k/v are NHD ([B, S, H, D]); returns [B, S, H, D]."""
        if k is None or v is None:
            raise ValueError("TEAttention does not support fused QKV; pass q, k and v.")
        if q.dim() != 4 or q.shape[-1] != self.head_dim:
            raise ValueError(
                f"Expected q of shape [B, S, H, D={self.head_dim}], got {tuple(q.shape)}"
            )
        if key_padding_mask is not None:
            raise NotImplementedError("TEAttention does not support key_padding_mask.")

        is_causal = attention_mask == PredefinedAttentionMask.CAUSAL
        num_gqa_groups = k.shape[-2] if self.num_heads != self.num_kv_heads else None
        self._lazy_init(num_gqa_groups, "causal" if is_causal else "no_mask")

        if "scaling_fwd" not in getattr(self._attn_op, "fp8_meta", {}):
            # TE builds fp8_meta inside the first FP8 forward. Run once to
            # create it (output discarded), then scale correctly below.
            with fp8_autocast(enabled=True, fp8_recipe=self.recipe):
                self._attn_op(q, k, v, attention_mask=None)
        if not self._slots_verified:
            self._verify_slots(q, k, v)
        self._set_forward_scales(q, k, v)

        with fp8_autocast(enabled=True, fp8_recipe=self.recipe):
            out = self._attn_op(q, k, v, attention_mask=None)
        # TE returns [B, S, H*D]; restore [B, S, H, D].
        return out.unflatten(-1, (self.num_heads, self.head_dim))

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return AttentionTensorLayout.NHD
