# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Diffusion TRTLLM Attention Backend

Wraps TrtllmAttention for visual generation (diffusion) models, handling the
specifics of no-KV-cache operation and fused QKV requirements.
"""

from typing import Optional

import torch

from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.visual_gen.args import QuantAttentionConfig

from ...attention_backend.interface import PredefinedAttentionMask
from ...attention_backend.sparse.skip_softmax import SkipSoftmaxParams
from ...attention_backend.trtllm import TrtllmAttention as BaseTrtllmAttention
from ...attention_backend.trtllm import TrtllmAttentionMetadata
from .interface import AttentionBackend, AttentionTensorLayout


def _check_metadata(
    attn_metadata: TrtllmAttentionMetadata,
    batch_size: int,
    q_seq_len: int,
    kv_seq_len: int,
) -> None:
    """Validate that the metadata describes the tensors it is used with."""
    if attn_metadata is None:
        raise ValueError(
            "TrtllmAttention.forward requires `attn_metadata`. Build it with "
            "visual_gen.attention_backend.metadata.create_diffusion_attn_metadata() "
            "and prepare it with prepare_diffusion_attn_metadata()."
        )

    seq_lens = attn_metadata.seq_lens
    if seq_lens is None:
        raise ValueError(
            "`attn_metadata` has no seq_lens; call prepare_diffusion_attn_metadata() "
            "before the forward pass."
        )
    if seq_lens.shape[0] != batch_size:
        raise ValueError(
            f"attn_metadata batch_size mismatch: cached {seq_lens.shape[0]} != {batch_size=}."
        )

    # `seq_lens` / `seq_lens_kv` are host tensors
    if bool((seq_lens != q_seq_len).any()):
        raise ValueError(
            f"attn_metadata q length mismatch: cached {seq_lens.tolist()} != {q_seq_len=}."
        )
    seq_lens_kv = attn_metadata.seq_lens_kv
    if bool((seq_lens_kv != kv_seq_len).any()):
        raise ValueError(
            f"attn_metadata kv length mismatch: cached {seq_lens_kv.tolist()} != {kv_seq_len=}."
        )


class TrtllmAttention(BaseTrtllmAttention, AttentionBackend):
    """
    TRTLLM Attention wrapper for diffusion models.

    Handles:
    - Fused QKV requirement for TRTLLM kernel (used when no quant_attention_config is provided)
    - No KV cache operation
    - SageAttention per-block QKV quantization (when a quant_attention_config is provided. requires unfused QKV)
    """

    Metadata = TrtllmAttentionMetadata

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 64,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        dtype: Optional[torch.dtype] = None,
        quant_attention_config: Optional[QuantAttentionConfig] = None,
        sparse_params: Optional[SkipSoftmaxParams] = None,
    ):
        num_kv_heads = num_kv_heads or num_heads

        super().__init__(
            layer_idx=layer_idx,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            quant_config=quant_config,
            sparse_params=sparse_params,
            dtype=dtype,
        )

        # TRTLLM expects flat [B*S, H*D] format
        self._preferred_layout = AttentionTensorLayout.NHD

        self.quant_attention_config = quant_attention_config

    @property
    def requires_metadata(self) -> bool:
        """TrtllmAttention always needs a metadata as its backends always enables varlen."""
        return True

    @torch.compile
    def _concat_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_size: int,
        seq_len: int,
        kv_seq_len: int,
    ):
        # Separate Q, K, V provided - fuse them
        q = q.view(batch_size * seq_len, -1)
        k = k.view(batch_size * kv_seq_len, -1)
        v = v.view(batch_size * kv_seq_len, -1)
        qkv = torch.cat([q, k, v], dim=-1)
        return qkv

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        *,
        attn_metadata: TrtllmAttentionMetadata,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass against caller-supplied attention metadata.

        Dimensions are derived from tensor shapes (NHD layout: ``[B, S, H, D]``).

        For diffusion models, expects:
        - Fused QKV: q contains [Q, K, V] concatenated, k and v are None
            - does not support SageAttention
        - OR separate Q, K, V which:
            - for regular TRTLLM attention, will be fused internally
            - for SageAttention, will be used directly

        Args:
            q: Query tensor [B, S, H, D] or fused QKV [B, S, H_qkv, D]
            k: Key tensor [B, S_kv, H_kv, D] or None if fused
            v: Value tensor [B, S_kv, H_kv, D] or None if fused
            attn_metadata: Prepared metadata for this attention site;
                must match the actual tensor dimensions.
            attention_mask: Attention mask type
            seq_len_kv: Sequence length for K/V (for cross-attention, defaults to seq_len)

        Returns:
            Output tensor [B, S, H*D]
        """
        batch_size, seq_len, _, _ = q.shape
        _, kv_seq_len, _, _ = k.shape
        _check_metadata(attn_metadata, batch_size, seq_len, kv_seq_len)
        timestep = kwargs.pop("timestep", None)

        if (
            self.quant_attention_config is not None
            and attention_mask == PredefinedAttentionMask.FULL
        ):
            assert k is not None and v is not None, (
                "SageAttention requires separate Q, K, V tensors"
            )
            quant_cfg = self.quant_attention_config
            q = q.reshape(batch_size * seq_len, -1).contiguous()
            k = k.reshape(batch_size * kv_seq_len, -1).contiguous()
            v = v.reshape(batch_size * kv_seq_len, -1).contiguous()
            output = super().forward(
                q=q,
                k=k,
                v=v,
                metadata=attn_metadata,
                attention_mask=attention_mask,
                timestep=timestep,
                sage_attn_num_elts_per_blk_q=quant_cfg.q_block_size,
                sage_attn_num_elts_per_blk_k=quant_cfg.k_block_size,
                sage_attn_num_elts_per_blk_v=quant_cfg.v_block_size,
                sage_attn_qk_int8=(quant_cfg.qk_dtype == "int8"),
            )
        else:
            if k is None and v is None:
                qkv = q.reshape(batch_size * seq_len, -1)
            else:
                qkv = self._concat_qkv(q, k, v, batch_size, seq_len, kv_seq_len)
            output = super().forward(
                q=qkv,
                k=None,
                v=None,
                metadata=attn_metadata,
                attention_mask=attention_mask,
                timestep=timestep,
            )
        output = output.view(batch_size, seq_len, -1)
        return output

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        """Return the preferred tensor layout for this backend."""
        return self._preferred_layout

    def support_fused_qkv(self) -> bool:
        """Standard path fuses QKV; SageAttention path does not."""
        return self.quant_attention_config is None
