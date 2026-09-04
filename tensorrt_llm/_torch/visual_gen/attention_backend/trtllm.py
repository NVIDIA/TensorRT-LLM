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

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping as MappingType
from typing import Optional, Union

import torch

from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.visual_gen.args import QuantAttentionConfig

from ...attention_backend.interface import (
    AttentionForwardArgs,
    AttentionRuntimeFeatures,
    PredefinedAttentionMask,
)
from ...attention_backend.sparse.params import SparseParams, SparseRuntimeParams
from ...attention_backend.trtllm import TrtllmAttention as BaseTrtllmAttention
from ...attention_backend.trtllm import TrtllmAttentionMetadata as BaseTrtllmAttentionMetadata
from .interface import AttentionBackend, AttentionTensorLayout


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class SparseForwardInputs:
    """Per-call inputs predicted for the normal TRTLLM attention path.

    The envelope is structurally immutable. Tensor payloads remain live objects
    so CUDA Graph-compatible predictors can publish values into stable buffers.
    """

    q: torch.Tensor = field(repr=False)
    k: Optional[torch.Tensor] = field(repr=False)
    v: Optional[torch.Tensor] = field(repr=False)
    batch_size: int
    seq_len: int
    seq_len_kv: int
    attention_mask: PredefinedAttentionMask
    sparse_runtime_params: SparseRuntimeParams = field(
        repr=False,
    )
    forward_kwargs: MappingType[str, object] = field(
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "forward_kwargs",
            MappingProxyType(dict(self.forward_kwargs)),
        )


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

    def get_fmha_cache_state(self, name: str) -> dict[str, object]:
        """Return one model-scoped cache owned by this metadata adapter."""

        fmha_caches = self._metadata_state.setdefault("fmha_caches", {})
        return fmha_caches.setdefault(name, {})

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
    - Separate-QKV forwarding for generic block-sparse attention
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
        self.metadata = TrtllmAttentionMetadata(
            attention_metadata_state=attention_metadata_state,
        )

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

    def update_quant_config(self, new_quant_config: Optional[QuantConfig]) -> None:
        """Rebuild FMHA libraries and bind VisualGen-owned shared plan caches."""

        super().update_quant_config(new_quant_config)
        from ...attention_backend.fmha.prims_ts_block_sparse import PrimsTSBlockSparseFmha

        cache_state = self.metadata.get_fmha_cache_state("prims_ts_block_sparse")
        for fmha in self._fmha_manager.fmha_libs:
            if isinstance(fmha, PrimsTSBlockSparseFmha):
                fmha.bind_plan_cache(cache_state)

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

    def block_sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        *,
        batch_size: int,
        seq_len: int,
        seq_len_kv: int,
        attention_mask: PredefinedAttentionMask,
        forward_kwargs: dict[str, object],
    ) -> Optional[SparseForwardInputs]:
        """Optionally predict inputs for an algorithm-specific sparse forward."""

        return None

    def sparse_post_process(
        self,
        output: torch.Tensor,
        sparse_inputs: SparseForwardInputs,
    ) -> torch.Tensor:
        """Finalize algorithm-specific output after normal TRTLLM forward."""

        return output

    def _forward_impl(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        seq_len_kv: Optional[int] = None,
        sparse_runtime_params: Optional[SparseRuntimeParams] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Execute TRTLLM attention with automatic metadata handling."""

        timestep = kwargs.pop("timestep", None)
        if kwargs:
            unexpected_names = ", ".join(sorted(kwargs))
            raise TypeError(
                f"Unexpected TRTLLM attention forward keyword arguments: {unexpected_names}"
            )

        kv_seq_len = seq_len_kv if seq_len_kv is not None else seq_len
        prepared_metadata = self._prepare_metadata(batch_size, seq_len)
        predicted_block_sparse_inputs = (
            sparse_runtime_params.block_sparse_inputs if sparse_runtime_params is not None else None
        )
        core_forward_args = AttentionForwardArgs(
            attention_mask=attention_mask,
            timestep=timestep,
            sparse_runtime_params=sparse_runtime_params,
        )

        if predicted_block_sparse_inputs is not None:
            if self.quant_attention_config is not None:
                raise ValueError(
                    "Generic block-sparse attention does not support quant_attention_config."
                )
            if k is None or v is None:
                raise ValueError("Generic block-sparse attention requires separate Q/K/V tensors.")
            output = super().forward(
                q=q.reshape(batch_size * seq_len, -1).contiguous(),
                k=k.reshape(batch_size * kv_seq_len, -1).contiguous(),
                v=v.reshape(batch_size * kv_seq_len, -1).contiguous(),
                metadata=prepared_metadata,
                forward_args=core_forward_args,
            )
        elif self.quant_attention_config is not None:
            assert k is not None and v is not None, (
                "SageAttention requires separate Q, K, V tensors"
            )
            quant_cfg = self.quant_attention_config
            q = q.reshape(batch_size * seq_len, -1).contiguous()
            k = k.reshape(batch_size * kv_seq_len, -1).contiguous()
            v = v.reshape(batch_size * kv_seq_len, -1).contiguous()
            core_forward_args = replace(
                core_forward_args,
                sage_attn_num_elts_per_blk_q=quant_cfg.q_block_size,
                sage_attn_num_elts_per_blk_k=quant_cfg.k_block_size,
                sage_attn_num_elts_per_blk_v=quant_cfg.v_block_size,
                sage_attn_qk_int8=(quant_cfg.qk_dtype == "int8"),
            )
            output = super().forward(
                q=q,
                k=k,
                v=v,
                metadata=prepared_metadata,
                forward_args=core_forward_args,
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
                forward_args=core_forward_args,
            )
        output = output.view(batch_size, seq_len, -1)
        return output

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        seq_len_kv: Optional[int] = None,
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
        legacy_sparse_inputs = {
            name for name in ("block_sparse_inputs", "sparse_runtime_params") if name in kwargs
        }
        if legacy_sparse_inputs:
            names = ", ".join(sorted(legacy_sparse_inputs))
            raise TypeError(
                f"{names} cannot be passed directly; use block_sparse_attn_predict "
                "to return SparseRuntimeParams instead"
            )

        sparse_inputs = self.block_sparse_attn_predict(
            q,
            k,
            v,
            batch_size=batch_size,
            seq_len=seq_len,
            seq_len_kv=seq_len if seq_len_kv is None else seq_len_kv,
            attention_mask=attention_mask,
            forward_kwargs=kwargs,
        )
        if sparse_inputs is None:
            return self._forward_impl(
                q,
                k,
                v,
                batch_size,
                seq_len,
                attention_mask=attention_mask,
                seq_len_kv=seq_len_kv,
                **kwargs,
            )

        output = self._forward_impl(
            sparse_inputs.q,
            sparse_inputs.k,
            sparse_inputs.v,
            sparse_inputs.batch_size,
            sparse_inputs.seq_len,
            attention_mask=sparse_inputs.attention_mask,
            seq_len_kv=sparse_inputs.seq_len_kv,
            sparse_runtime_params=sparse_inputs.sparse_runtime_params,
            **sparse_inputs.forward_kwargs,
        )
        return self.sparse_post_process(output, sparse_inputs)

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        """Return the preferred tensor layout for this backend."""
        return self._preferred_layout

    def support_fused_qkv(self) -> bool:
        """Standard path fuses QKV; SageAttention path does not."""
        return self.quant_attention_config is None
