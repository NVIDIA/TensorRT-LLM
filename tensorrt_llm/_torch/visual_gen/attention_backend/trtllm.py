# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional, Union

import torch

from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

from ...attention_backend.interface import AttentionRuntimeFeatures, PredefinedAttentionMask
from ...attention_backend.trtllm import TrtllmAttention as BaseTrtllmAttention
from ...attention_backend.trtllm import TrtllmAttentionMetadata as BaseTrtllmAttentionMetadata
from ..config import SageAttentionConfig
from .interface import AttentionTensorLayout


class TrtllmAttentionMetadata:
    """
    Simplified metadata adapter for diffusion models using TRTLLM backend.

    Lazy initialization with auto-growing capacity:
    - Metadata created only when capacity needs increase
    - prepare() called only when seq_lens actually change
    - Automatically reallocates when batch_size or seq_len exceeds current capacity

    Args:
        max_batch_size: Initial batch size hint. Will grow automatically if exceeded.
        max_seq_len: Initial sequence length hint. Will grow automatically if exceeded.
        device: Target device for tensors.
    """

    def __init__(
        self,
        max_batch_size: int = 16,
        max_seq_len: int = 4096,
        device: Optional[torch.device] = None,
    ):
        # These are initial hints, not hard limits - capacity grows as needed
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device or torch.device("cuda")

        # Lazily created BaseTrtllmAttentionMetadata
        self._metadata: Optional[BaseTrtllmAttentionMetadata] = None

        # Track allocated capacity
        self._allocated_batch_size = 0
        self._allocated_max_seq_len = 0

        # Track prepared state
        self._cached_batch_size: int = 0
        self._cached_max_seq_len: int = 0
        self._prepared = False

        # Reusable seq_lens tensor — avoids allocating a new tensor every call
        # when the caller passes an int (the common diffusion case).
        self._seq_lens_buf: Optional[torch.Tensor] = None
        self._seq_lens_buf_batch: int = 0
        self._seq_lens_buf_val: int = 0

    def _needs_new_metadata(self, batch_size: int, max_seq_len: int) -> bool:
        """Check if we need to create new metadata (capacity change)."""
        return (
            self._metadata is None
            or batch_size > self._allocated_batch_size
            or max_seq_len > self._allocated_max_seq_len
        )

    def _needs_prepare(self, batch_size: int, max_seq_len: int) -> bool:
        """Check if we need to call prepare() (batch or seq_len changed).

        Assumes uniform sequence length per batch; if per-sample lengths vary,
        we may need to check seq_lens tensor instead.
        """
        if not self._prepared:
            return True
        return batch_size != self._cached_batch_size or max_seq_len != self._cached_max_seq_len

    def _create_metadata(self, batch_size: int, max_seq_len: int) -> None:
        """Create new metadata with given capacity."""
        # Allocate with some headroom to avoid frequent reallocation
        alloc_batch = max(batch_size, self._allocated_batch_size)
        alloc_seq_len = max(max_seq_len, self._allocated_max_seq_len)

        self._metadata = BaseTrtllmAttentionMetadata(
            max_num_requests=alloc_batch,
            max_num_tokens=alloc_batch * alloc_seq_len,
            max_num_sequences=alloc_batch,
            kv_cache_manager=None,  # No KV cache for diffusion
            mapping=Mapping(),
            runtime_features=AttentionRuntimeFeatures(),
        )

        self._allocated_batch_size = alloc_batch
        self._allocated_max_seq_len = alloc_seq_len
        self._prepared = False  # Reset prepare state on new metadata

    def _get_seq_lens_tensor(
        self, batch_size: int, seq_lens: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        """Return a seq_lens tensor, reusing an internal buffer when possible."""
        if isinstance(seq_lens, int):
            # Fast path: return cached buffer if (batch_size, value) unchanged
            if batch_size == self._seq_lens_buf_batch and seq_lens == self._seq_lens_buf_val:
                return self._seq_lens_buf
            # Reuse buffer allocation if only the value changed
            if self._seq_lens_buf is not None and batch_size == self._seq_lens_buf_batch:
                self._seq_lens_buf.fill_(seq_lens)
            else:
                self._seq_lens_buf = torch.full((batch_size,), seq_lens, dtype=torch.int32)
            self._seq_lens_buf_batch = batch_size
            self._seq_lens_buf_val = seq_lens
            return self._seq_lens_buf
        return seq_lens.to(dtype=torch.int32)

    def prepare(
        self,
        batch_size: int,
        seq_lens: Union[int, torch.Tensor],
    ) -> BaseTrtllmAttentionMetadata:
        """
        Prepare metadata for a forward pass.

        Lazy behavior:
        - Creates metadata only when capacity needs increase
        - Calls prepare() only when (batch_size, max_seq_len) actually change
        """
        seq_lens_tensor = self._get_seq_lens_tensor(batch_size, seq_lens)
        max_seq_len = int(seq_lens_tensor.max())

        if self._needs_new_metadata(batch_size, max_seq_len):
            self._create_metadata(batch_size, max_seq_len)

        if self._needs_prepare(batch_size, max_seq_len):
            self._metadata.seq_lens = seq_lens_tensor
            self._metadata.num_contexts = batch_size
            self._metadata.max_seq_len = max_seq_len
            self._metadata.request_ids = list(range(batch_size))
            self._metadata.prepare()

            self._cached_batch_size = batch_size
            self._cached_max_seq_len = max_seq_len
            self._prepared = True

        return self._metadata


class TrtllmAttention(BaseTrtllmAttention):
    """
    TRTLLM Attention wrapper for diffusion models.

    Handles:
    - Metadata creation and preparation
    - No KV cache operation

    Two dispatch paths controlled by ``sage_attention_config``:
    - Standard (None): fuses Q/K/V into a single QKV tensor before calling
      the base kernel (``is_fused_qkv=True``).
    - SageAttention (non-None): passes separate Q/K/V with per-block
      quantization parameters (``is_fused_qkv=False``).
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
        sage_attention_config: Optional[SageAttentionConfig] = None,
    ):
        num_kv_heads = num_kv_heads or num_heads

        super().__init__(
            layer_idx=layer_idx,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            quant_config=quant_config,
            dtype=dtype,
        )

        # TRTLLM expects flat [B*S, H*D] format
        self._preferred_layout = AttentionTensorLayout.NHD

        self.metadata = TrtllmAttentionMetadata(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

        # SageAttention: presence of config object implies enablement
        self.sage_attention_config = sage_attention_config

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_size: int,
        seq_len: int,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        seq_len_kv: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with automatic metadata handling.

        Always receives separate Q, K, V tensors from the caller.
        Internally dispatches to one of two paths:
        - Standard: fuses Q/K/V then calls base kernel with fused QKV.
        - SageAttention: forwards separate Q/K/V with quantization params.

        Args:
            q: Query tensor [B, S, H, D]
            k: Key tensor [B, S_kv, H_kv, D]
            v: Value tensor [B, S_kv, H_kv, D]
            batch_size: Batch size
            seq_len: Sequence length for Q
            attention_mask: Attention mask type
            seq_len_kv: Sequence length for K/V (for cross-attention, defaults to seq_len)

        Returns:
            Output tensor [B, S, H*D]
        """
        # Handle cross-attention where K/V have different sequence length than Q
        kv_seq_len = seq_len_kv if seq_len_kv is not None else seq_len

        if self.sage_attention_config is not None:
            # SageAttention kernel requires separate Q/K/V tensors.
            sage_cfg = self.sage_attention_config
            q = q.reshape(batch_size * seq_len, -1).contiguous()
            k = k.reshape(batch_size * kv_seq_len, -1).contiguous()
            v = v.reshape(batch_size * kv_seq_len, -1).contiguous()
            prepared_metadata = self.metadata.prepare(batch_size, seq_len)
            output = super().forward(
                q=q,
                k=k,
                v=v,
                metadata=prepared_metadata,
                attention_mask=attention_mask,
                sage_attn_num_elts_per_blk_q=sage_cfg.num_elts_per_blk_q,
                sage_attn_num_elts_per_blk_k=sage_cfg.num_elts_per_blk_k,
                sage_attn_num_elts_per_blk_v=sage_cfg.num_elts_per_blk_v,
                sage_attn_qk_int8=sage_cfg.qk_int8,
            )
            output = output.view(batch_size, seq_len, -1)
        else:
            # Standard path: fuse QKV.
            q = q.view(batch_size * seq_len, -1)
            k = k.view(batch_size * kv_seq_len, -1)
            v = v.view(batch_size * kv_seq_len, -1)
            qkv = torch.cat([q, k, v], dim=-1)
            prepared_metadata = self.metadata.prepare(batch_size, seq_len)
            output = super().forward(
                q=qkv,
                k=None,
                v=None,
                metadata=prepared_metadata,
                attention_mask=attention_mask,
            )
            output = output.view(batch_size, seq_len, -1)

        return output

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        """Return the preferred tensor layout for this backend."""
        return self._preferred_layout

    @property
    def support_fused_qkv(self) -> bool:
        """Standard path fuses QKV; SageAttention path does not."""
        return self.sage_attention_config is None
