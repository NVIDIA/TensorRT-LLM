# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The static state of one Kimi K3 MLA attention role, held by the family instead of an attention backend.

A K3 MLA layer runs two attention roles over one latent KV cache:

* the context role (``K3MlaAttnState.context(...)``): ``H`` context heads of 192 (K-nope 128 + K-pe 64)
  query / key dims and 128 value dims, the context FMHA's view;
* the generation role (``K3MlaAttnState.generation(...)``): ``H`` absorbed heads of 576 (latent 512 + K-pe
  64) dims over one KV head, the decode's view.

This object carries what those calls read besides the batch: the head geometry, the softmax scaling, the KV
cache quantization of the layer (FP8, NVFP4 or none), the unit KV scales, the identity RoPE table (K3 is
NoPE: every MLA kernel that rotates reads an identity ``(cos, sin) = (1, 0)`` table) and this layer's index
in the KV cache manager. The engine owns the KV cache manager; the ops read its pools and block offsets
from the attention metadata.

Attribute names match those the K3 FP4 MLA decode and KV-only append read from the layer's generation role
(``num_heads``, ``kv_lora_rank``, ``qk_*_head_dim``, ``q_scaling``, ``layer_idx``, ``rotary_cos_sin``,
``uses_fp4_mla_attention``, ``get_fp4_mla_local_layer_idx``, ``_ensure_rope_table_size``), so one state
object serves all of them.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from tensorrt_llm._utils import get_sm_version, is_sm_100f, is_sm_120f
from tensorrt_llm.functional import PositionEmbeddingType, RopeEmbeddingUtils, RotaryScalingType
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantMode

# K3's MLA RoPE configuration: GPT-NeoX positions over the 64 K-pe dims, interleaved (cos, sin) pairs
# duplicated over the rope dim (the FP4 MLA cache update requires the duplicated layout).
_ROPE_THETA = 10000.0
_ROPE_POSITION_EMBEDDING_TYPE = PositionEmbeddingType.rope_gpt_neox
_ROPE_DUPLICATE_DATA = True
# SMs whose MLA kernels honour a skip-correction threshold.
_SKIP_CORRECTION_SMS = (100, 103)

# Identity (cos, sin) = (1, 0) tables by (device index, float count), shared by every layer.
_IDENTITY_ROPE_TABLES: dict[Tuple[int, int], torch.Tensor] = {}
# The sinusoid inverse frequencies by (rope dim, theta), shared by every layer.
_ROPE_INV_FREQ: dict[Tuple[int, float, int], torch.Tensor] = {}


def identity_rope_table(num_floats: int, device: torch.device) -> torch.Tensor:
    """A ``[1, num_floats]`` FP32 table of interleaved ``(cos, sin) = (1, 0)`` pairs: the MLA RoPE kernels'
    ``rotary_cos_sin`` layout, with every rotation the identity."""
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, num_floats)
    table = _IDENTITY_ROPE_TABLES.get(key)
    if table is None:
        table = torch.zeros(1, num_floats, dtype=torch.float32, device=device)
        table.view(-1)[0::2] = 1.0
        if not torch.cuda.is_current_stream_capturing():
            # Written before any kernel on another stream reads it.
            torch.cuda.synchronize(device)
        _IDENTITY_ROPE_TABLES[key] = table
    return table


def _rope_inv_freq(
    rope_dim: int, max_positions: int, device: torch.device
) -> Optional[torch.Tensor]:
    """The sinusoid inverse frequencies the attention op receives beside the (identity) table; the MLA
    paths read the table."""
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (rope_dim, _ROPE_THETA, index)
    inv_freq = _ROPE_INV_FREQ.get(key)
    if inv_freq is None:
        inv_freq_np, _ = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
            max_positions,
            rope_dim,
            _ROPE_THETA,
            1.0,
            RotaryScalingType.none,
            rope_scaling_config={
                "factor": 1.0,
                "alpha": 1.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": max_positions,
            },
            duplicate_data=_ROPE_DUPLICATE_DATA,
        )
        if inv_freq_np is None:
            return None
        inv_freq = torch.tensor(inv_freq_np, dtype=torch.float32, device=device)
        _ROPE_INV_FREQ[key] = inv_freq
    return inv_freq


def has_fp8_context_mla_fmha(sm_version: int) -> bool:
    """Whether MLA context attention over an FP8 or NVFP4 KV cache runs the attention op's FP8 context MLA
    FMHA on this SM (it then reads FP8 Q, K and V)."""
    return sm_version == 90 or is_sm_100f(sm_version) or is_sm_120f(sm_version)


class K3MlaAttnState:
    """One K3 MLA attention role (context or generation) of one layer: geometry, quantization, scales,
    identity RoPE table and KV cache layer index (see the module docstring)."""

    def __init__(
        self,
        *,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        max_position_embeddings: int,
        quant_config: Optional[QuantConfig],
        skip_correction_threshold: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> None:
        device = torch.device("cuda") if device is None else torch.device(device)
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.is_mla_enable = True
        self.rope_append = True
        self.predicted_tokens_per_seq = 1
        self.q_scaling = 1.0
        self.position_embedding_type = _ROPE_POSITION_EMBEDDING_TYPE
        self.rope_dim = qk_rope_head_dim
        self.rope_base = _ROPE_THETA
        self.rope_max_positions = max_position_embeddings
        self.rope_original_max_positions = max_position_embeddings
        self.rope_duplicate_data = _ROPE_DUPLICATE_DATA
        sm = get_sm_version()
        threshold = float(skip_correction_threshold)
        self.skip_correction_threshold = (
            threshold if threshold > 0.0 and sm in _SKIP_CORRECTION_SMS else 0.0
        )
        self.rotary_inv_freq = _rope_inv_freq(qk_rope_head_dim, max_position_embeddings, device)
        self.rotary_cos_sin = identity_rope_table(
            max_position_embeddings * self._floats_per_position, device
        )
        # Unit FP8 KV-cache scales (the PyTorch runtime never overrides them).
        self.kv_cache_scaling_factor = torch.ones(1, dtype=torch.float32, device=device)
        self.kv_scale_quant_orig = self.kv_cache_scaling_factor
        self.kv_scale_orig_quant = 1.0 / self.kv_cache_scaling_factor
        # Counters the attention op updates when skip-softmax statistics are on.
        self.skip_softmax_stat = torch.zeros(2, dtype=torch.uint32, device=device)
        self.local_layer_idx: Optional[int] = None
        self.update_quant_config(quant_config)

    @classmethod
    def context(
        cls,
        *,
        num_context_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        **kwargs,
    ) -> "K3MlaAttnState":
        """The context role: ``num_context_heads`` heads of 192 Q / K dims and ``v_head_dim`` V dims."""
        return cls(
            num_heads=num_context_heads,
            head_dim=qk_nope_head_dim + qk_rope_head_dim,
            num_kv_heads=num_context_heads,
            v_head_dim=v_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            **kwargs,
        )

    @classmethod
    def generation(
        cls, *, num_heads: int, kv_lora_rank: int, qk_rope_head_dim: int, **kwargs
    ) -> "K3MlaAttnState":
        """The generation role: ``num_heads`` absorbed heads of ``kv_lora_rank + qk_rope_head_dim`` dims
        over one KV head, writing ``kv_lora_rank`` dims per head."""
        return cls(
            num_heads=num_heads,
            head_dim=kv_lora_rank + qk_rope_head_dim,
            num_kv_heads=1,
            v_head_dim=kv_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            **kwargs,
        )

    # ------------------------------------------------------------------------------------------
    # Quantization.
    # ------------------------------------------------------------------------------------------

    def update_quant_config(self, quant_config: Optional[QuantConfig]) -> None:
        """The layer's KV cache quantization (``quant_config.layer_quant_mode``)."""
        self.quant_config = quant_config or QuantConfig()
        mode = self.quant_config.layer_quant_mode
        self.quant_mode: QuantMode = mode
        self.has_fp8_kv_cache = bool(mode.has_fp8_kv_cache())
        self.has_fp4_kv_cache = bool(mode.has_fp4_kv_cache())

    @property
    def uses_fp4_mla_attention(self) -> bool:
        """Whether the layer's cache is the dense FP4 MLA (NVFP4) cache."""
        return self.has_fp4_kv_cache

    @property
    def has_fp8_context_mla(self) -> bool:
        """Whether context attention over this layer's FP8 cache runs the op's FP8 context MLA FMHA."""
        return self.has_fp8_kv_cache and has_fp8_context_mla_fmha(get_sm_version())

    @staticmethod
    def support_fp4_kv_cache() -> bool:
        return True

    # ------------------------------------------------------------------------------------------
    # RoPE table.
    # ------------------------------------------------------------------------------------------

    @property
    def _floats_per_position(self) -> int:
        return self.rope_dim * (2 if self.rope_duplicate_data else 1)

    def _ensure_rope_table_size(self, required_max_positions: int) -> None:
        """Grow the identity table (and the RoPE position count the op receives) to
        ``required_max_positions``."""
        table_positions = self.rotary_cos_sin.numel() // self._floats_per_position
        self.rope_max_positions = max(
            self.rope_max_positions, table_positions, required_max_positions
        )
        if required_max_positions > table_positions:
            self.rotary_cos_sin = identity_rope_table(
                self.rope_max_positions * self._floats_per_position, self.rotary_cos_sin.device
            )

    # ------------------------------------------------------------------------------------------
    # KV cache layer index and cache-state predicates.
    # ------------------------------------------------------------------------------------------

    def get_local_layer_idx(self, metadata) -> int:
        """This layer's index in the metadata's KV cache manager (``layer_offsets``), cached."""
        if self.local_layer_idx is not None:
            return self.local_layer_idx
        if metadata.kv_cache_manager is None:
            return self.layer_idx
        self.local_layer_idx = metadata.kv_cache_manager.layer_offsets[self.layer_idx]
        return self.local_layer_idx

    def get_fp4_mla_local_layer_idx(self, metadata) -> int:
        """This layer's compact index in the FP4 MLA side pools (the manager's
        ``_fp4_mla_compact_layer_idx``)."""
        local_layer_idx = self.get_local_layer_idx(metadata)
        if metadata.kv_cache_manager is None:
            return local_layer_idx
        to_compact = getattr(metadata.kv_cache_manager, "_fp4_mla_compact_layer_idx", None)
        if not callable(to_compact):
            raise RuntimeError("FP4 MLA requires a cache manager with compact layer mapping.")
        return to_compact(local_layer_idx)

    @staticmethod
    def has_cached_kv_for_mla_context(metadata) -> bool:
        """Whether the batch's context requests attend a cached prefix."""
        return (
            metadata.kv_cache_manager is not None
            and metadata.enable_context_mla_with_cached_kv
            and metadata.num_ctx_cached_tokens > 0
        )

    @staticmethod
    def is_chunked_prefill_for_mla_context(metadata) -> bool:
        """Whether the cached prefix is attended in chunked-prefill partitions."""
        return (
            metadata.kv_cache_manager is not None
            and metadata.enable_context_mla_with_cached_kv
            and metadata.num_ctx_cached_tokens > 0
            and metadata.runtime_features.chunked_prefill
        )

    @staticmethod
    def has_cached_kv_for_mla_context_warmup(metadata) -> bool:
        """Cached-prefix context is configured (warmup: the cached token count is not checked)."""
        return metadata.kv_cache_manager is not None and metadata.enable_context_mla_with_cached_kv


__all__ = ["K3MlaAttnState", "has_fp8_context_mla_fmha", "identity_rope_table"]
