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
        self._cached_seq_lens: Optional[torch.Tensor] = None
        self._prepared = False

    def _needs_new_metadata(self, batch_size: int, max_seq_len: int) -> bool:
        """Check if we need to create new metadata (capacity change)."""
        return (
            self._metadata is None
            or batch_size > self._allocated_batch_size
            or max_seq_len > self._allocated_max_seq_len
        )

    def _needs_prepare(self, batch_size: int, seq_lens: torch.Tensor) -> bool:
        """Check if we need to call prepare() (seq_lens changed)."""
        if not self._prepared:
            return True
        if self._cached_seq_lens is None:
            return True
        if self._cached_seq_lens.shape[0] != batch_size:
            return True
        return not torch.equal(self._cached_seq_lens[:batch_size], seq_lens)

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

    def prepare(
        self,
        batch_size: int,
        seq_lens: Union[int, torch.Tensor],
    ) -> BaseTrtllmAttentionMetadata:
        """
        Prepare metadata for a forward pass.

        Lazy behavior:
        - Creates metadata only when capacity needs increase
        - Calls prepare() only when seq_lens actually change
        """
        if isinstance(seq_lens, int):
            seq_lens_tensor = torch.full((batch_size,), seq_lens, dtype=torch.int32)
        else:
            seq_lens_tensor = seq_lens.to(dtype=torch.int32)

        max_seq_len = seq_lens_tensor.max().item()

        if self._needs_new_metadata(batch_size, max_seq_len):
            self._create_metadata(batch_size, max_seq_len)

        if self._needs_prepare(batch_size, seq_lens_tensor):
            self._metadata.seq_lens = seq_lens_tensor
            self._metadata.num_contexts = batch_size
            self._metadata.max_seq_len = max_seq_len
            self._metadata.request_ids = list(range(batch_size))
            self._metadata.prepare()

            # Cache for next comparison
            if self._cached_seq_lens is None or self._cached_seq_lens.shape[0] < batch_size:
                self._cached_seq_lens = seq_lens_tensor.clone()
            else:
                self._cached_seq_lens[:batch_size].copy_(seq_lens_tensor)
            self._prepared = True

        return self._metadata


class TrtllmAttention(BaseTrtllmAttention):
    """
    TRTLLM Attention wrapper for diffusion models.

    Handles:
    - Fused QKV requirement for TRTLLM kernel
    - Metadata creation and preparation
    - No KV cache operation
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

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_size: int,
        seq_len: Union[int, torch.Tensor],
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        seq_len_kv: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with automatic metadata handling.

        For diffusion models, expects:
        - Fused QKV: q contains [Q, K, V] concatenated, k and v are None
        - OR separate Q, K, V which will be fused internally

        Args:
            q: Query tensor [num_tokens, hidden] or fused QKV [num_tokens, qkv_hidden]
            k: Key tensor [num_tokens, kv_hidden] or None if fused
            v: Value tensor [num_tokens, kv_hidden] or None if fused
            batch_size: Batch size (required if not inferable)
            seq_len: Sequence length for Q (required if not inferable)
            attention_mask: Attention mask type
            seq_len_kv: Sequence length for K/V (for cross-attention, defaults to seq_len)

        Returns:
            Output tensor [num_tokens, q_hidden]
        """
        # Handle cross-attention where K/V have different sequence length than Q
        kv_seq_len = seq_len_kv if seq_len_kv is not None else seq_len

        # Separate Q, K, V provided - fuse them
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

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return True
