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

Wraps TrtllmAttention with simplified metadata for visual generation (diffusion) models.
Handles the specifics of no-KV-cache operation and fused QKV requirements.
"""

from typing import Optional

import torch

from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.visual_gen.args import QuantAttentionConfig

from ...attention_backend.interface import (
    AttentionForwardArgs,
    AttentionRuntimeFeatures,
    PredefinedAttentionMask,
)
from ...attention_backend.sparse.block_sparse import BlockSparseForwardInputs, BlockSparseParams
from ...attention_backend.sparse.params import SparseParams
from ...attention_backend.trtllm import TrtllmAttention as BaseTrtllmAttention
from ...attention_backend.trtllm import TrtllmAttentionMetadata as BaseTrtllmAttentionMetadata
from .interface import AttentionBackend, AttentionTensorLayout


class TrtllmAttentionMetadata:
    """Shape-keyed TRTLLM metadata adapter for diffusion models.

    Args:
        attention_metadata_state: Mutable per-component state shared by all
            VisualGen TRTLLM attention layers. This adapter consumes its
            shape-keyed metadata cache; FMHA implementations may attach runtime
            resources with the same cross-layer/CUDA-Graph lifetime.
    """

    def __init__(
        self,
        attention_metadata_state: Optional[dict] = None,
    ):
        if attention_metadata_state is None:
            raise ValueError(
                "TRTLLM attention requires `attention_metadata_state` to be provided "
                "by visual-gen config for component-scoped metadata sharing."
            )
        # Lazily created BaseTrtllmAttentionMetadata objects. Diffusion blocks
        # can launch video and audio attention back-to-back with different
        # sequence lengths, so keep separate metadata buffers per shape instead
        # of mutating one shared object while kernels may still be in flight.
        self._metadata_cache: dict[tuple[int, tuple[int, ...]], BaseTrtllmAttentionMetadata] = (
            attention_metadata_state.setdefault("metadata_cache", {})
        )

    def prepare(
        self,
        batch_size: int,
        seq_lens: int | torch.Tensor,
    ) -> BaseTrtllmAttentionMetadata:
        """Return prepared metadata dedicated to the exact sequence-length key."""

        if isinstance(seq_lens, int):
            seq_lens_tensor = torch.full((batch_size,), seq_lens, dtype=torch.int32)
        else:
            seq_lens_tensor = seq_lens.to(dtype=torch.int32)

        # Keep CUDA graph-captured metadata buffers stable per batch/seq-lens shape.
        cache_key = (batch_size, tuple(int(x) for x in seq_lens_tensor.tolist()))
        metadata = self._metadata_cache.get(cache_key)
        if metadata is not None:
            return metadata

        max_seq_len = int(seq_lens_tensor.max().item())
        metadata = BaseTrtllmAttentionMetadata(
            max_num_requests=batch_size,
            max_num_tokens=batch_size * max_seq_len,
            max_num_sequences=batch_size,
            kv_cache_manager=None,  # No KV cache for diffusion
            mapping=Mapping(),
            runtime_features=AttentionRuntimeFeatures(),
        )
        metadata.seq_lens = seq_lens_tensor.clone()
        metadata.num_contexts = batch_size
        metadata.max_seq_len = max_seq_len
        metadata.request_ids = list(range(batch_size))
        metadata.prepare()
        self._metadata_cache[cache_key] = metadata
        return metadata


class TrtllmAttention(BaseTrtllmAttention, AttentionBackend):
    """
    TRTLLM Attention wrapper for diffusion models.

    Handles:
    - Fused QKV requirement for TRTLLM kernel (used when no quant_attention_config is provided)
    - Metadata creation and preparation
    - No KV cache operation
    - SageAttention per-block QKV quantization (when a quant_attention_config is provided. requires unfused QKV)
    """

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 64,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        dtype: Optional[torch.dtype] = None,
        max_batch_size: int = 16,
        max_seq_len: int = 4096,
        quant_attention_config: Optional[QuantAttentionConfig] = None,
        attention_metadata_state: Optional[dict] = None,
        sparse_params: Optional[SparseParams] = None,
    ):
        num_kv_heads = num_kv_heads or num_heads
        if isinstance(sparse_params, BlockSparseParams) and quant_attention_config is not None:
            raise ValueError(
                "Generic block-sparse attention does not support quant_attention_config."
            )
        super().__init__(
            layer_idx=layer_idx,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            quant_config=quant_config,
            sparse_params=sparse_params,
            attention_metadata_state=attention_metadata_state,
            dtype=dtype,
        )

        # TRTLLM expects flat [B*S, H*D] format
        self._preferred_layout = AttentionTensorLayout.NHD

        self.metadata = TrtllmAttentionMetadata(
            attention_metadata_state=attention_metadata_state,
        )

        self.quant_attention_config = quant_attention_config

    # Needed to work with torch compile cause of attention metadata
    # make attn metadata as input for it to work
    @torch.compiler.disable
    def _prepare_metadata(self, batch_size: int, seq_len: int):
        return self.metadata.prepare(batch_size, seq_len)

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
        batch_size: int,
        seq_len: int,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        seq_len_kv: Optional[int] = None,
        block_sparse_inputs: Optional[BlockSparseForwardInputs] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with automatic metadata handling.

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
            batch_size: Batch size
            seq_len: Sequence length for Q
            attention_mask: Attention mask type
            seq_len_kv: Sequence length for K/V (for cross-attention, defaults to seq_len)

        Returns:
            Output tensor [B, S, H*D]
        """
        kv_seq_len = seq_len_kv if seq_len_kv is not None else seq_len
        timestep = kwargs.pop("timestep", None)

        prepared_metadata = self._prepare_metadata(batch_size, seq_len)

        if block_sparse_inputs is not None:
            if k is None or v is None:
                raise ValueError("Generic block-sparse attention requires separate Q/K/V tensors.")
            output = super().forward(
                q=q.reshape(batch_size * seq_len, -1).contiguous(),
                k=k.reshape(batch_size * kv_seq_len, -1).contiguous(),
                v=v.reshape(batch_size * kv_seq_len, -1).contiguous(),
                metadata=prepared_metadata,
                forward_args=AttentionForwardArgs(
                    attention_mask=attention_mask,
                    block_sparse_inputs=block_sparse_inputs,
                ),
            )
        elif self.quant_attention_config is not None:
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
                metadata=prepared_metadata,
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
                metadata=prepared_metadata,
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
        """SageAttention and generic block-sparse attention need separate Q/K/V."""
        return self.quant_attention_config is None and not isinstance(
            self.sparse_params, BlockSparseParams
        )
