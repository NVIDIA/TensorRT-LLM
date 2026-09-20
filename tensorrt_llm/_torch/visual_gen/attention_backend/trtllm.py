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

from typing import Optional, Union

import torch

from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.visual_gen.args import QuantAttentionConfig

from ...attention.backends.interface import (
    AttentionForwardArgs,
    AttentionRuntimeFeatures,
    PredefinedAttentionMask,
)
from ...attention.backends.sparse.params import SparseBackendForwardArgs, SparseParams
from ...attention.backends.sparse.timestep_phase import graph_phase_for_timestep, timestep_to_float
from ...attention.backends.trtllm import TrtllmAttention as BaseTrtllmAttention
from ...attention.backends.trtllm import TrtllmAttentionMetadata as BaseTrtllmAttentionMetadata
from .interface import AttentionBackend, AttentionTensorLayout


class TrtllmAttentionMetadata:
    """
    Simplified metadata adapter for diffusion models using TRTLLM backend.

    Lazy initialization with auto-growing capacity:
    - Metadata created only when capacity needs increase
    - prepare() called only when seq_lens actually change
    - Automatically reallocates when batch_size or seq_len exceeds current capacity

    Args:
        device: Target device for tensors.
        attention_metadata_state: Mutable model-scoped state shared by all
            attention layers in one model instance.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        attention_metadata_state: Optional[dict] = None,
    ):
        self.device = device or torch.device("cuda")
        if attention_metadata_state is None:
            raise ValueError(
                "TRTLLM attention requires `attention_metadata_state` to be provided "
                "by visual-gen config for model-scoped metadata sharing."
            )
        self._metadata_state = attention_metadata_state

        # Lazily created BaseTrtllmAttentionMetadata objects. Diffusion blocks
        # can launch video and audio attention back-to-back with different
        # sequence lengths, so keep separate metadata buffers per shape instead
        # of mutating one shared object while kernels may still be in flight.
        self._metadata_cache = self._metadata_state.setdefault("metadata_cache", {})
        self._metadata: Optional[BaseTrtllmAttentionMetadata] = None

        # Track prepared state
        self._cached_seq_lens: Optional[torch.Tensor] = None
        self._prepared = False

    def _needs_prepare(self, batch_size: int, seq_lens: torch.Tensor) -> bool:
        """Check if we need to call prepare() (current request seq_lens or shared metadata object seq_lens changed).

        Assumes uniform sequence length per batch; if per-sample lengths vary,
        we may need to check seq_lens tensor instead.

        In addition, multiple visual gen attention modules share one metadata object.  A
        different module may have prepared it for another sequence length even
        when this wrapper's local cached seq_lens are unchanged.
        """
        if not self._prepared:
            return True
        if self._cached_seq_lens is None:
            return True
        if self._cached_seq_lens.shape[0] != batch_size:
            return True
        if not torch.equal(self._cached_seq_lens[:batch_size], seq_lens):
            return True

        metadata = self._metadata
        if metadata is None:
            return True
        if getattr(metadata, "num_contexts", None) != batch_size:
            return True

        max_seq_len = seq_lens.max().item()
        if getattr(metadata, "max_seq_len", None) != max_seq_len:
            return True

        metadata_seq_lens = getattr(metadata, "seq_lens", None)
        if metadata_seq_lens is None or metadata_seq_lens.shape[0] < batch_size:
            return True
        if not torch.equal(metadata_seq_lens[:batch_size].to(seq_lens.device), seq_lens):
            return True

        return False

    def _create_metadata(self, batch_size: int, max_seq_len: int) -> None:
        """Create new metadata with given capacity."""
        self._metadata = BaseTrtllmAttentionMetadata(
            max_num_requests=batch_size,
            max_num_tokens=batch_size * max_seq_len,
            max_num_sequences=batch_size,
            kv_cache_manager=None,  # No KV cache for diffusion
            mapping=Mapping(),
            runtime_features=AttentionRuntimeFeatures(),
        )
        self._prepared = False  # Reset prepare state on new metadata

    def _select_cached_metadata(self, cached) -> None:
        self._metadata = cached["metadata"]
        self._prepared = cached["prepared"]
        self._cached_seq_lens = cached["seq_lens"]

    def prepare_timestep(self, timestep: object) -> Optional[float]:
        """Reduce ``timestep`` to a host scalar and keep it for CUDA Graph capture.

        Timestep-scheduled sparse algorithms read the timestep on the host,
        which CUDA Graph capture cannot do for a device tensor. Eager calls,
        including the warmup that precedes capture, reduce the tensor and store
        the value in the component state; capture returns the stored value.
        """

        state = self._metadata_state
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            if "timestep" not in state:
                raise RuntimeError(
                    "sparse attention timestep must be prepared before CUDA Graph capture"
                )
            return state["timestep"]
        value = timestep_to_float(timestep)
        state["timestep"] = value
        return value

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
        if isinstance(seq_lens, int):
            seq_lens_tensor = torch.full((batch_size,), seq_lens, dtype=torch.int32)
        else:
            seq_lens_tensor = seq_lens.to(dtype=torch.int32)
        max_seq_len = seq_lens_tensor.max().item()
        # Keep CUDA graph-captured metadata buffers stable per batch/seq-lens shape.
        cache_key = (batch_size, tuple(int(x) for x in seq_lens_tensor.tolist()))

        cached = self._metadata_cache.get(cache_key)
        if cached is None:
            self._create_metadata(batch_size, max_seq_len)
            cached = {
                "metadata": self._metadata,
                "prepared": False,
                "seq_lens": None,
            }
            self._metadata_cache[cache_key] = cached

        self._select_cached_metadata(cached)

        if self._needs_prepare(batch_size, seq_lens_tensor):
            cached_seq_lens = seq_lens_tensor.clone()
            self._metadata.seq_lens = cached_seq_lens
            self._metadata.num_contexts = batch_size
            self._metadata.max_seq_len = max_seq_len
            self._metadata.request_ids = list(range(batch_size))
            self._metadata.prepare()

            # Cache per-shape state without sharing the tensor across entries.
            cached["prepared"] = True
            cached["seq_lens"] = cached_seq_lens

            self._select_cached_metadata(cached)

        return self._metadata


class TrtllmAttention(BaseTrtllmAttention, AttentionBackend):
    """
    TRTLLM Attention wrapper for diffusion models.

    Handles:
    - Fused QKV requirement for TRTLLM kernel (used when no quant_attention_config is provided)
    - Metadata creation and preparation
    - No KV cache operation
    - SageAttention per-block QKV quantization (when a quant_attention_config is provided. requires unfused QKV)
    - Separate-QKV forwarding for generic block-sparse attention and backends that reject fused QKV
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
        if attention_metadata_state is None:
            raise ValueError(
                "TRTLLM attention requires `attention_metadata_state` to be provided "
                "by visual-gen config for model-scoped metadata and plan sharing."
            )

        super().__init__(
            layer_idx=layer_idx,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            quant_config=quant_config,
            sparse_params=sparse_params,
            dtype=dtype,
            # Every layer of one model component shares its FMHA plan caches.
            fmha_state=attention_metadata_state.setdefault("fmha_caches", {}),
        )

        # TRTLLM expects flat [B*S, H*D] format
        self._preferred_layout = AttentionTensorLayout.NHD

        self.metadata = TrtllmAttentionMetadata(
            attention_metadata_state=attention_metadata_state,
        )

        self.quant_attention_config = quant_attention_config

    @property
    def timestep_cutoff(self) -> Optional[float]:
        """Normalized timestep below which the sparse algorithm is enabled, if any."""

        return getattr(self.sparse_params, "disabled_until_timestep", None)

    def resolve_timestep(self, timestep: object) -> object:
        """Return the host timestep the sparse schedule consumes.

        Layers with a timestep cutoff hand the tensor to the metadata adapter,
        which reduces it during eager calls and reuses the prepared value under
        CUDA Graph capture. Without a cutoff the timestep passes through
        untouched.
        """

        if self.timestep_cutoff is None:
            return timestep
        return self.metadata.prepare_timestep(timestep)

    @property
    def dense_layers(self) -> frozenset[int]:
        """Layer indices that always run dense attention."""

        return getattr(self.sparse_params, "dense_layers", frozenset())

    def should_use_sparse(self, timestep: object) -> bool:
        """Return whether this layer runs its sparse path for the prepared ``timestep``.

        Dense layers never do. Otherwise the layer is sparse unless the timestep
        schedule places the call in the dense prefix; without a cutoff or a
        timestep the call is sparse.
        """

        if self.layer_idx in self.dense_layers:
            return False
        graph_phase = graph_phase_for_timestep(
            timestep, disabled_until_timestep=self.timestep_cutoff
        )
        return graph_phase is None or graph_phase == 1

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

    @torch.compile
    def _compact_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_size: int,
        seq_len: int,
        kv_seq_len: int,
    ):
        # Separate Q, K, V stay separate - compact each into a contiguous token-major matrix.
        # Slices of a fused QKV projection are strided; the compiled copy keeps them on a
        # vectorized kernel, while already contiguous inputs pass through without a copy.
        q = q.reshape(batch_size * seq_len, -1).contiguous()
        k = k.reshape(batch_size * kv_seq_len, -1).contiguous()
        v = v.reshape(batch_size * kv_seq_len, -1).contiguous()
        return q, k, v

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        seq_len_kv: Optional[int] = None,
        sparse_backend_args: Optional[SparseBackendForwardArgs] = None,
        timestep: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with automatic metadata handling.

        Dimensions are derived from tensor shapes (NHD layout: ``[B, S, H, D]``).

        For diffusion models, expects:
        - Fused QKV: q contains [Q, K, V] concatenated, k and v are None
            - does not support SageAttention or block-sparse routes
        - OR separate Q, K, V which:
            - for regular TRTLLM attention, will be fused internally
            - for SageAttention, block-sparse routes, and backends that reject
              fused QKV, will be passed to the core as separate tensors

        Args:
            q: Query tensor [B, S, H, D] or fused QKV [B, S, H_qkv, D]
            k: Key tensor [B, S_kv, H_kv, D] or None if fused
            v: Value tensor [B, S_kv, H_kv, D] or None if fused
            batch_size: Batch size
            seq_len: Sequence length for Q
            attention_mask: Attention mask type
            seq_len_kv: Sequence length for K/V (for cross-attention, defaults to seq_len)
            sparse_backend_args: Module-predicted sparse inputs handed to the core
                prediction hooks. A ``block_sparse_inputs`` payload selects the
                generic block-sparse FMHA.
            timestep: Denoising timestep consumed by the sparse prediction hooks.
            **kwargs: Backend-specific keyword arguments forwarded by the attention
                module, such as ``key_padding_mask``; ignored here, as by the other
                backends. The TRTLLM kernels have no key-padding input, so callers
                that need padded keys must guard at the model level or select the
                ``VANILLA`` backend.

        Returns:
            Output tensor [B, S, H*D]
        """
        timestep = self.resolve_timestep(timestep)
        block_sparse_inputs = (
            sparse_backend_args.block_sparse_inputs if sparse_backend_args is not None else None
        )
        use_separate_qkv = (
            block_sparse_inputs is not None
            or self.quant_attention_config is not None
            or not self.support_fused_qkv()
        )
        if use_separate_qkv and (k is None or v is None):
            raise ValueError("This TRTLLM attention call requires separate q, k, and v tensors.")
        if block_sparse_inputs is not None and self.quant_attention_config is not None:
            raise ValueError(
                "Generic block-sparse attention does not support quant_attention_config."
            )

        kv_seq_len = seq_len_kv if seq_len_kv is not None else seq_len
        prepared_metadata = self._prepare_metadata(batch_size, seq_len)
        sage_kwargs = {}
        if use_separate_qkv:
            q, k, v = self._compact_qkv(q, k, v, batch_size, seq_len, kv_seq_len)
            quant_cfg = self.quant_attention_config
            if quant_cfg is not None:
                sage_kwargs = {
                    "sage_attn_num_elts_per_blk_q": quant_cfg.q_block_size,
                    "sage_attn_num_elts_per_blk_k": quant_cfg.k_block_size,
                    "sage_attn_num_elts_per_blk_v": quant_cfg.v_block_size,
                    "sage_attn_qk_int8": quant_cfg.qk_dtype == "int8",
                }
        else:
            if k is None and v is None:
                q = q.reshape(batch_size * seq_len, -1)
            else:
                q = self._concat_qkv(q, k, v, batch_size, seq_len, kv_seq_len)
            k = None
            v = None
        output = super().forward(
            q=q,
            k=k,
            v=v,
            metadata=prepared_metadata,
            forward_args=AttentionForwardArgs(
                attention_mask=attention_mask,
                timestep=timestep,
                sparse_backend_args=sparse_backend_args,
                **sage_kwargs,
            ),
        )
        return output.view(batch_size, seq_len, -1)

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        """Return the preferred tensor layout for this backend."""
        return self._preferred_layout

    def support_fused_qkv(self) -> bool:
        """Standard path fuses QKV; SageAttention path does not."""
        return self.quant_attention_config is None
