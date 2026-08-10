# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 specialization of the shared PyTorch-backend MLA module.

The base ``MLA`` class owns context attention, cached/chunked prefill,
absorbed generation, and paged-cache handling. This module only supplies the
K3 projection topology, NoPE identity table, KV-B checkpoint layout, and
gated output projection.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from torch import nn

from ....functional import PositionEmbeddingType
from ....logger import logger
from ....models.modeling_utils import QuantConfig
from ...attention_backend import AttentionMetadata, TrtllmAttention
from ...attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ...model_config import ModelConfig
from ...pyexecutor.breakable_cuda_graph import is_in_breakable_cuda_graph
from ..mla import MLA, maybe_bcg_mla_custom_op_inplace

_KIMI_K3_MLA_GEN_BACKEND_ENV = "TLLM_K3_MLA_GEN_BACKEND"


def _select_mla_generation_backend(quant_config: Optional[QuantConfig]) -> str:
    """Select K3's absorbed-generation MLA backend.

    K3 was tuned with the FlashInfer CuTe-DSL backend for BF16 KV cache.
    FP8 KV cache carries device scales that CuTe-DSL does not support, so
    retain the TRTLLM-Gen fallback used by the pre-refactor implementation.
    """
    backend = os.environ.get(_KIMI_K3_MLA_GEN_BACKEND_ENV, "cute-dsl")
    has_fp8_kv_cache = bool(
        quant_config is not None and quant_config.layer_quant_mode.has_fp8_kv_cache()
    )
    if has_fp8_kv_cache and backend != "trtllm-gen":
        logger.info(
            "Kimi K3 MLA: FP8 KV cache requires the trtllm-gen MLA "
            f"generation backend; overriding '{backend}' -> 'trtllm-gen'."
        )
        return "trtllm-gen"
    return backend


def _kimi_k3_mla_decode_backend_policy(
    requested_backend: str,
    metadata,
    num_tokens: int,
) -> str:
    """Per-batch MLA decode backend selection for Kimi K3.

    Installed as ``mla_backend_policy`` on K3's generation attention backend
    (see :class:`KimiK3MLAAttention`); the general attention code applies no
    such policy on its own.

    CuTe-DSL reuses one staged page table across MLA layers for a
    generation-only, one-token-per-request batch. A mixed context/generation
    batch cannot use that reuse key, so selecting CuTe-DSL would repeat the
    staging copies in every MLA layer and regress time to first token. K3's
    128 padded query heads also restrict the CuTe-DSL kernel to one query
    token per request. Keep the requested CuTe-DSL fast path for plain decode
    and use TRTLLM-Gen for mixed batches and speculative multi-token
    verification.
    """
    is_single_token_generation = num_tokens == metadata.num_generations
    if requested_backend == "cute-dsl" and (
        metadata.num_contexts > 0 or not is_single_token_generation
    ):
        return "trtllm-gen"
    return requested_backend


def _meta_safe_cast_dtype(module, dtype):
    """``module.to(dtype=dtype)`` that also works under ``MetaInitMode``.

    ``Module.to`` dispatches ``aten._to_copy``, which MetaInitMode rejects
    (it would silently fall back to full CPU construction of the model —
    ~70 GB of host RAM per rank for Kimi K3). Under meta init the values
    are garbage anyway, so a dtype-only re-allocation via ``empty_like``
    (an allowed init op) is equivalent; off meta this matches ``.to``.
    """
    import torch as _torch

    def _cast(t):
        if not t.is_floating_point():
            return t
        if t.is_meta:
            return _torch.empty_like(t, dtype=dtype)
        return t.to(dtype=dtype)

    module._apply(_cast)


def _make_pos_embd_params(
    *,
    qk_rope_head_dim: int,
    max_position_embeddings: int,
) -> PositionalEmbeddingParams:
    """Build a valid rope config so the backend allocates a real cache.

    We use rope_gpt_neox with default theta=10000 and ``duplicate_data
    =True`` (the same convention DeepSeek-V3-style MLA uses when
    ``qk_rope_head_dim`` is present). The resulting ``rotary_cos_sin``
    has the exact shape the C++ MLA rope kernel indexes. Immediately
    after backend construction we overwrite the tensor values with
    ``(cos=1, sin=0)`` — an identity rotation, matching K3's NoPE.
    """
    rope_params = RopeParams(
        dim=qk_rope_head_dim,
        theta=10000.0,
        max_positions=max_position_embeddings,
        original_max_positions=max_position_embeddings,
        duplicate_data=True,
    )
    return PositionalEmbeddingParams(
        type=PositionEmbeddingType.rope_gpt_neox,
        rope=rope_params,
        # Match the working DeepSeek-V3-style MLA reference test
        # (tests/unittest/_torch/attention/test_attention_mla.py) which
        # sets ``is_neox=False``. The MLA fused rope kernel is GPT-J
        # style regardless of this flag, but the C++ FMHA reads this bit
        # elsewhere and stability under identity-cos-sin depends on the
        # standard non-neox layout.
        is_neox=False,
    )


def _write_identity_rope_values(cos_sin: torch.Tensor) -> None:
    """Overwrite a rotary cos/sin table with identity values in place.

    Interleaved (cos, sin) pairs: index [::2] = cos, [1::2] = sin.
    Setting cos=1 and sin=0 per position makes the rotation the
    identity — a mathematical no-op — which preserves K3's NoPE
    semantics without patching the backend.
    """
    flat = cos_sin.reshape(-1)
    with torch.no_grad():
        flat[0::2] = 1.0
        flat[1::2] = 0.0
    # Ensure the identity write reaches CUDA memory before any kernel
    # launched from a different stream can read the table.
    if cos_sin.is_cuda:
        torch.cuda.synchronize(cos_sin.device)


def _install_identity_rope_table(backend: TrtllmAttention) -> None:
    """Install an identity rotary cos/sin table on ``backend``.

    The C++ MLA rope kernels (``mla_rope_generation`` and the context
    preprocess) read this table and apply the rotation; identity values
    make that a copy, preserving K3's NoPE.

    The tensor SHAPE produced by ``create_rope_const_params`` is kept
    intact so the C++ ``float2`` indexing stays valid. Only the values
    are overwritten in place. ``_ensure_rope_table_size`` is replaced
    with an identity-preserving resize: the table may GROW (so the
    fused rope-generation op can never index out of bounds for long
    sequences) but its values are always rewritten to identity right
    after a regeneration, so the real sinusoids never leak in.
    """
    cos_sin = backend.rotary_cos_sin
    if cos_sin is None:
        raise RuntimeError(
            "backend.rotary_cos_sin is None after construction; check "
            "pos_embd_params has a valid RopeParams with dim > 0."
        )
    _write_identity_rope_values(cos_sin)

    orig_resize = backend._ensure_rope_table_size  # bound method

    def _identity_preserving_resize(required_max_positions: int) -> None:
        if required_max_positions <= backend.rope_params.max_positions:
            return
        orig_resize(required_max_positions)
        _write_identity_rope_values(backend.rotary_cos_sin)

    backend._ensure_rope_table_size = _identity_preserving_resize


# ---------------------------------------------------------------------------
# KimiK3MLAAttention.
# ---------------------------------------------------------------------------


class KimiK3MLAAttention(MLA):
    """Kimi K3 MLA implemented as a thin specialization of :class:`MLA`.

    K3 keeps the standard dense MLA attention/cache flow and only changes the
    checkpoint projection topology, positional encoding, KV-B runtime layout,
    and gated output projection.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rms_norm_eps: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        layer_idx: int = 0,
        use_output_gate: bool = True,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantConfig] = None,
        runtime_model_config: Optional[ModelConfig] = None,
    ) -> None:
        pos_embd_params = _make_pos_embd_params(
            qk_rope_head_dim=qk_rope_head_dim,
            max_position_embeddings=max_position_embeddings,
        )
        mla_model_config = ModelConfig(
            quant_config=quant_config if quant_config is not None else QuantConfig()
        )
        # Share the runtime's ``extra_attrs`` so the base ``MLA`` registers
        # itself into ``mla_layers`` (enables the breakable-CUDA-graph
        # eager-on-graph attention op for K3 MLA layers).
        if runtime_model_config is not None:
            mla_model_config.extra_attrs = runtime_model_config.extra_attrs
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            predicted_tokens_per_seq=1,
            max_position_embeddings=max_position_embeddings,
            bias=False,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=dtype,
            dense_bias=False,
            config=mla_model_config,
            reduce_output=False,
            fuse_qkv_a_proj=False,
            rms_norm_eps=rms_norm_eps,
            flashinfer_mla_backend=_select_mla_generation_backend(quant_config),
        )
        # K3 calls forward_impl() directly to insert its output gate before
        # the base row-parallel o_proj. The original executor metadata remains
        # intact, so MLA performs its native mixed context/generation split.
        # ``register_to_config`` is left as set by the base ``MLA`` (True when
        # a shared ``extra_attrs`` was threaded above): it gates the breakable
        # -CUDA-graph eager-on-graph attention op.

        self.use_output_gate = use_output_gate

        if use_output_gate:
            self.g_proj = nn.Linear(
                hidden_size,
                num_heads * v_head_dim,
                bias=False,
            )

        # K3 is NoPE. The base MLA backends still require real RoPE tables, so
        # retain their expected shape and replace every rotation with identity.
        assert isinstance(self.mha, TrtllmAttention)
        assert isinstance(self.mqa, TrtllmAttention)
        _install_identity_rope_table(self.mha)
        _install_identity_rope_table(self.mqa)
        # Only the absorbed-generation backend (mqa) requests CuTe-DSL, so
        # only it needs K3's per-batch fallback policy; mha keeps the
        # default trtllm-gen selection.
        self.mqa.mla_backend_policy = _kimi_k3_mla_decode_backend_policy
        self.rotary_emb = None
        self.apply_rotary_emb = False

        if dtype is not None:
            _meta_safe_cast_dtype(self, dtype)

    def _apply_output_gate_and_o_proj(
        self,
        hidden_states: torch.Tensor,
        attn_out: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_output_gate:
            attn_out = attn_out * self.g_proj(hidden_states).sigmoid()
        return self.o_proj(attn_out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        attn_out = self.create_output(
            hidden_states,
            attn_metadata.num_contexts,
        )
        if self.register_to_config and is_in_breakable_cuda_graph():
            # Breakable prefill CUDA graph: run the metadata-dependent MLA
            # attention eagerly between captured segments, writing into the
            # graph-allocated ``attn_out``. The output gate + row-parallel
            # o_proj below stay on-graph.
            maybe_bcg_mla_custom_op_inplace(
                hidden_states,
                None,
                self.layer_idx_str,
                attn_out,
                None,
                None,
                None,
                False,
            )
        else:
            super().forward_impl(
                None,
                hidden_states,
                attn_metadata,
                output=attn_out,
            )
        return self._apply_output_gate_and_o_proj(hidden_states, attn_out)
