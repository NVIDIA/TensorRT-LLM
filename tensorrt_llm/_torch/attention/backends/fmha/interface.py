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

from __future__ import annotations

import dataclasses
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Optional, final

import torch

from tensorrt_llm._torch.attention.backends.cpp_schema import CPP_METADATA_KEY, cpp_metadata
from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )
    from tensorrt_llm.bindings import BlockSparseParams, DataType, MlaMetaParams
    from tensorrt_llm.functional import (
        AttentionMaskType,
        PositionEmbeddingType,
        RopeParams,
        RotaryScalingType,
    )
    from tensorrt_llm.quantization.mode import QuantMode


@dataclass(kw_only=True, slots=True)
class StaticAttentionConfig:
    """The attention-layer configuration used to build native kernel runners."""

    num_heads: int = cpp_metadata(default=0)
    num_kv_heads: int = cpp_metadata(default=0)
    head_size: int = cpp_metadata(default=0)
    tokens_per_block: int = cpp_metadata(default=0)
    type: DataType = cpp_metadata(default=None)
    is_fp8_out: bool = cpp_metadata(default=False)
    is_fp4_out: bool = cpp_metadata(default=False)
    use_kv_cache: bool = cpp_metadata(default=False)
    paged_context_fmha: bool = cpp_metadata(default=False)
    position_embedding_type: PositionEmbeddingType = cpp_metadata(default=0)
    mask_type: AttentionMaskType = cpp_metadata(default=1)
    q_scaling: float = cpp_metadata(default=1.0)
    rotary_embedding_dim: int = cpp_metadata(default=0)
    attn_logit_softcapping_scale: float = cpp_metadata(default=0.0)
    remove_padding: bool = cpp_metadata(default=True)
    cross_attention: bool = cpp_metadata(default=False)
    dense_context_fmha: bool = cpp_metadata(default=False)
    fuses_dsv4_inv_rope_fp8_quant: bool = cpp_metadata(default=False)
    use_sparse_attention: bool = cpp_metadata(default=False)
    use_tllm_gen_sparse_attention: bool = cpp_metadata(default=False)
    use_nvfp4_mla_kv_cache: bool = cpp_metadata(default=False)
    is_spec_decoding_enabled: bool = cpp_metadata(default=False)
    spec_decoding_target_max_gen_len: int = cpp_metadata(default=0)
    is_mla_enable: bool = cpp_metadata(default=False)
    q_lora_rank: int = cpp_metadata(default=0)
    kv_lora_rank: int = cpp_metadata(default=0)
    qk_nope_head_dim: int = cpp_metadata(default=0)
    qk_rope_head_dim: int = cpp_metadata(default=0)
    v_head_dim: int = cpp_metadata(default=0)
    predicted_tokens_per_seq: int = cpp_metadata(default=1)
    mla_layer_num: int = cpp_metadata(default=0)
    rope_append: bool = cpp_metadata(default=True)
    sage_attn_num_elts_per_blk_q: int = cpp_metadata(default=0)
    sage_attn_num_elts_per_blk_k: int = cpp_metadata(default=0)
    sage_attn_num_elts_per_blk_v: int = cpp_metadata(default=0)
    sage_attn_qk_int8: bool = cpp_metadata(default=False)
    quant_mode: QuantMode = cpp_metadata(default=0)
    skip_correction_threshold: float = cpp_metadata(default=0.0)

    @classmethod
    def from_params(
        cls,
        params: FmhaParams,
        *,
        skip_correction_threshold: float = 0.0,
    ) -> StaticAttentionConfig:
        """Capture the fixed runner-selection inputs from one attention call."""
        from tensorrt_llm._utils import torch_dtype_to_binding

        if params.qkv_or_q is None:
            raise RuntimeError("StaticAttentionConfig requires qkv_or_q.")
        if params.output is None:
            raise RuntimeError("StaticAttentionConfig requires output.")
        if params.fwd is None:
            raise RuntimeError("StaticAttentionConfig requires forward args.")

        from tensorrt_llm.quantization.mode import QuantMode

        quant_mode = QuantMode(params.quant_mode)
        sparse = params.fwd.sparse_runtime_params
        has_sparse_attn_indices = (
            sparse.sparse_attn_indices is not None and sparse.sparse_attn_indices.numel() > 0
        )
        has_sparse_attention = (
            sparse.sparse_kv_indices is not None and sparse.sparse_kv_indices.numel() > 0
        ) or has_sparse_attn_indices
        has_paged_sparse_attention = (
            has_sparse_attn_indices
            and sparse.sparse_attn_offsets is not None
            and sparse.sparse_attn_offsets.numel() > 0
        )
        use_tllm_gen_sparse_attention = has_sparse_attn_indices and not has_paged_sparse_attention
        use_kv_cache = (
            params.kv_cache_block_offsets is not None
            and params.host_kv_cache_pool_pointers is not None
            and params.host_kv_cache_pool_mapping is not None
        )
        rotary_embedding_dim = (
            params.rope_params.dim
            if params.rope_params is not None
            else params.rotary_embedding_dim
        )
        mla_layer_num = (
            params.host_kv_cache_pool_mapping.size(0)
            if params.host_kv_cache_pool_mapping is not None
            else 0
        )
        target_max_gen_len = params.spec_decoding_target_max_gen_len
        if params.spec_decoding_target_max_draft_tokens is not None and target_max_gen_len == 0:
            target_max_gen_len = params.spec_decoding_target_max_draft_tokens + 1

        return cls(
            num_heads=params.num_heads,
            num_kv_heads=params.num_kv_heads,
            head_size=params.head_size,
            tokens_per_block=params.tokens_per_block,
            type=torch_dtype_to_binding(params.qkv_or_q.dtype),
            is_fp8_out=params.output.dtype == torch.float8_e4m3fn,
            is_fp4_out=params.output.dtype == torch.uint8,
            use_kv_cache=use_kv_cache,
            paged_context_fmha=params.paged_context_fmha,
            position_embedding_type=params.position_embedding_type,
            mask_type=params.mask_type,
            q_scaling=params.q_scaling,
            rotary_embedding_dim=rotary_embedding_dim,
            attn_logit_softcapping_scale=params.attn_logit_softcapping_scale,
            remove_padding=params.remove_padding,
            cross_attention=params.is_cross,
            dense_context_fmha=params.dense_context_fmha,
            fuses_dsv4_inv_rope_fp8_quant=params.fwd.enable_dsv4_epilogue_fusion,
            use_sparse_attention=has_sparse_attention,
            use_tllm_gen_sparse_attention=use_tllm_gen_sparse_attention,
            use_nvfp4_mla_kv_cache=(
                quant_mode.has_fp4_kv_cache()
                and use_tllm_gen_sparse_attention
                and sparse.sparse_attn_kv_lens is None
                and sparse.aux_kv_cache_pool_ptr is not None
            ),
            is_spec_decoding_enabled=params.is_spec_decoding_enabled,
            spec_decoding_target_max_gen_len=target_max_gen_len,
            is_mla_enable=params.is_mla_enable,
            q_lora_rank=params.q_lora_rank or 0,
            kv_lora_rank=params.kv_lora_rank,
            qk_nope_head_dim=params.qk_nope_head_dim,
            qk_rope_head_dim=params.qk_rope_head_dim,
            v_head_dim=params.v_head_dim or 0,
            predicted_tokens_per_seq=params.predicted_tokens_per_seq,
            mla_layer_num=mla_layer_num,
            rope_append=params.rope_append is not False,
            sage_attn_num_elts_per_blk_q=params.fwd.sage_attn_num_elts_per_blk_q,
            sage_attn_num_elts_per_blk_k=params.fwd.sage_attn_num_elts_per_blk_k,
            sage_attn_num_elts_per_blk_v=params.fwd.sage_attn_num_elts_per_blk_v,
            sage_attn_qk_int8=params.fwd.sage_attn_qk_int8,
            quant_mode=quant_mode,
            skip_correction_threshold=skip_correction_threshold,
        )

    def to_thop_config(self) -> Any:
        """Build the native constructor configuration."""
        from tensorrt_llm.bindings.internal import thop

        target = thop.StaticAttentionConfig()
        _lower_struct(target, self)
        return target


@dataclass(slots=True)
class FmhaParams:
    """Attention parameters shared by Python, DSL, Triton, and native FMHA paths.

    Offset contract, relied on by the native side. It splits by memory space:

    * **Device** per-token and per-sequence tensors are phase-local views. Slice them
      here for the context or generation phase; C++ never re-slices them.
    * **Host** tensors, the KV-cache block offsets and the FP4 scaling factors stay
      whole-batch. C++ indexes them with ``seq_offset`` / ``token_offset``: pointer
      accessors apply the offset themselves, so call sites never pass it; only the
      explicit max-over-range queries take a range.

    Applying an offset to the first group double-counts it; omitting it for the second
    shifts every sequence by the number of context requests.
    """

    fwd: AttentionForwardArgs = cpp_metadata(default=None)
    # Python-only back-references, skipped by the native codegen (no cpp_metadata).
    # FMHA backends that need layer/metadata state the flat schema does not carry
    # (e.g. the Triton custom-mask backend) read them from here.
    attn: Any = None
    meta: Any = None
    local_layer_idx: int = cpp_metadata(default=-1)
    has_fp8_kv_cache: bool = cpp_metadata(default=False)
    rope_params: RopeParams = None
    kv_pool: Optional[torch.Tensor] = None
    use_paged_context_fmha: bool = cpp_metadata(default=False)
    fp8_context_fmha: bool = cpp_metadata(default=False)
    kv_factor: int = cpp_metadata(default=1)
    total_num_blocks: int = cpp_metadata(default=0)
    seq_offset: int = cpp_metadata(default=0)
    num_seqs: int = cpp_metadata(default=0)
    token_offset: int = cpp_metadata(default=0)
    num_tokens: int = cpp_metadata(default=0)
    predicted_tokens_per_seq: int = cpp_metadata(default=0)
    input_seq_length: int = cpp_metadata(default=0)
    max_past_kv_length: int = cpp_metadata(default=0)
    num_requests: int = cpp_metadata(default=0)

    layer_idx: int = cpp_metadata(default=-1)
    num_heads: int = cpp_metadata(default=-1)
    num_kv_heads: int = cpp_metadata(default=-1)
    head_size: int = cpp_metadata(default=-1)
    q_scaling: float = cpp_metadata(default=1.0)
    rotary_embedding_dim: int = cpp_metadata(default=0)
    rotary_embedding_base: float = cpp_metadata(default=10000.0)
    rotary_embedding_scale_type: RotaryScalingType = cpp_metadata(default=0)
    rotary_embedding_scale: float = cpp_metadata(default=1.0)
    rotary_embedding_short_mscale: float = cpp_metadata(default=1.0)
    rotary_embedding_long_mscale: float = cpp_metadata(default=1.0)
    rotary_embedding_max_positions: int = cpp_metadata(default=1024)
    rotary_embedding_original_max_positions: int = cpp_metadata(default=1024)
    position_embedding_type: PositionEmbeddingType = cpp_metadata(default=0)
    mask_type: AttentionMaskType = cpp_metadata(default=1)
    tokens_per_block: int = cpp_metadata(default=0)
    quant_mode: QuantMode = cpp_metadata(default=0)
    max_context_length: int = cpp_metadata(default=0)
    max_seq_len: int = cpp_metadata(default=0)
    max_num_requests: int = cpp_metadata(default=0)
    # Total number of sequences, i.e. max_num_requests * beam_width. The generation
    # workspace and the multi-block counter are sized per sequence, not per request.
    max_num_sequences: int = cpp_metadata(default=0)
    beam_width: int = cpp_metadata(default=1)
    paged_context_fmha: bool = cpp_metadata(default=False)
    is_spec_decoding_enabled: bool = cpp_metadata(default=False)
    use_spec_decoding: bool = cpp_metadata(default=False)
    is_spec_dec_tree: bool = cpp_metadata(default=True)
    force_prepare_spec_dec_tree_mask: bool = cpp_metadata(default=False)
    is_mla_enable: bool = cpp_metadata(default=False)
    attention_chunk_size: Optional[int] = cpp_metadata(default=None)

    q_lora_rank: Optional[int] = cpp_metadata(default=None)
    kv_lora_rank: int = cpp_metadata(default=0)
    qk_nope_head_dim: int = cpp_metadata(default=0)
    qk_rope_head_dim: int = cpp_metadata(default=0)
    v_head_dim: Optional[int] = cpp_metadata(default=None)
    rope_append: Optional[bool] = cpp_metadata(default=None)
    spec_decoding_target_max_draft_tokens: Optional[int] = cpp_metadata(default=None)

    workspace: torch.Tensor = cpp_metadata(ctype=None, default=None)
    output: torch.Tensor = cpp_metadata(ctype=None, default=None)
    qkv_or_q: torch.Tensor = cpp_metadata(default=None)
    k: Optional[torch.Tensor] = cpp_metadata(default=None)
    v: Optional[torch.Tensor] = cpp_metadata(default=None)

    sequence_length: torch.Tensor = cpp_metadata(ctype=torch.int32, default=None)
    host_past_key_value_lengths: torch.Tensor = cpp_metadata(ctype=None, default=None)
    total_kv_len: int = cpp_metadata(default=0)
    context_lengths: torch.Tensor = cpp_metadata(ctype=torch.int32, default=None)
    host_context_lengths: torch.Tensor = cpp_metadata(ctype=None, default=None)
    max_context_q_len_override: Optional[int] = cpp_metadata(default=None)
    kv_cache_block_offsets: Optional[torch.Tensor] = cpp_metadata(ctype=None, default=None)
    host_kv_cache_pool_pointers: Optional[torch.Tensor] = cpp_metadata(ctype=None, default=None)
    host_kv_cache_pool_mapping: Optional[torch.Tensor] = cpp_metadata(ctype=None, default=None)
    cache_indirection: Optional[torch.Tensor] = cpp_metadata(ctype=torch.int32, default=None)
    max_attention_window_size: int = cpp_metadata(default=0)
    cyclic_attention_window_size: int = cpp_metadata(default=0)

    rotary_inv_freq: Optional[torch.Tensor] = cpp_metadata(ctype=torch.float32, default=None)
    rotary_cos_sin: Optional[torch.Tensor] = cpp_metadata(ctype=None, default=None)

    block_ids_per_seq: Optional[torch.Tensor] = cpp_metadata(ctype=torch.int32, default=None)

    helix_position_offsets: Optional[torch.Tensor] = cpp_metadata(ctype=torch.int32, default=None)
    helix_is_inactive_rank: Optional[torch.Tensor] = cpp_metadata(ctype=torch.bool, default=None)

    spec_decoding_generation_lengths: Optional[torch.Tensor] = cpp_metadata(
        ctype=torch.int32, default=None
    )
    spec_decoding_position_offsets_for_cpp: Optional[torch.Tensor] = cpp_metadata(
        ctype=torch.int32, default=None
    )
    spec_decoding_packed_mask: Optional[torch.Tensor] = cpp_metadata(
        ctype=torch.int32, default=None
    )
    spec_decoding_bl_tree_mask_offset: Optional[torch.Tensor] = cpp_metadata(
        ctype=torch.int64, default=None
    )
    spec_decoding_bl_tree_mask: Optional[torch.Tensor] = cpp_metadata(
        ctype=torch.uint32, default=None
    )
    spec_bl_tree_first_sparse_mask_offset_kv: Optional[torch.Tensor] = cpp_metadata(
        ctype=torch.int32, default=None
    )

    num_sparse_topk: int = cpp_metadata(default=0)

    flash_mla_tile_scheduler_metadata: Optional[torch.Tensor] = cpp_metadata(
        ctype=torch.int32, default=None
    )
    flash_mla_num_splits: Optional[torch.Tensor] = cpp_metadata(ctype=torch.int32, default=None)

    trtllm_gen_jit_warmup: bool = cpp_metadata(default=False)

    is_cross: bool = cpp_metadata(default=False)

    # Fused kv_a_layernorm for the DSv4 sparse context path: when set, `latent_cache`
    # is the raw kv_a_proj output and the context RoPE kernel norms it in place.

    # Mechanical state consumed by handwritten C++ lowering hooks. Defaults
    # remain Python-owned; the generated C++ holder is only value-initialized.
    vision_start: int = cpp_metadata(default=-1)
    vision_length: int = cpp_metadata(default=-1)
    unidirectional: int = cpp_metadata(default=1)
    attn_logit_softcapping_scale: float = cpp_metadata(default=0.0)
    use_logn_scaling: bool = cpp_metadata(default=False)
    remove_padding: bool = cpp_metadata(default=True)
    block_sparse_params: BlockSparseParams = cpp_metadata(default=None)
    unfuse_qkv_gemm: bool = cpp_metadata(default=False)
    type: DataType = cpp_metadata(default=None)
    is_fp8_out: bool = cpp_metadata(default=False)
    is_fp4_out: bool = cpp_metadata(default=False)
    qkv_bias_enabled: bool = cpp_metadata(default=False)
    cross_attention: bool = cpp_metadata(default=False)
    pos_shift_enabled: bool = cpp_metadata(default=False)
    dense_context_fmha: bool = cpp_metadata(default=False)
    has_full_attention_mask: bool = cpp_metadata(default=False)
    spec_decoding_is_generation_length_variable: bool = cpp_metadata(default=False)
    spec_decoding_max_generation_length: int = cpp_metadata(default=1)
    spec_decoding_target_max_gen_len: int = cpp_metadata(default=0)
    use_sparse_attention: bool = cpp_metadata(default=False)
    use_tllm_gen_sparse_attention_paged: bool = cpp_metadata(default=False)
    use_tllm_gen_sparse_attention: bool = cpp_metadata(default=False)
    mla_params: MlaMetaParams = cpp_metadata(default=None)
    use_kv_cache: bool = cpp_metadata(default=True)
    skip_attn: bool = cpp_metadata(default=False)
    fuses_dsv4_inv_rope_fp8_quant: bool = cpp_metadata(default=False)
    v_stride_in_bytes: int = cpp_metadata(default=0)
    qkv_bias: Optional[torch.Tensor] = cpp_metadata(default=None)
    attention_mask: Optional[torch.Tensor] = cpp_metadata(ctype=torch.bool, default=None)
    attention_packed_mask: Optional[torch.Tensor] = cpp_metadata(ctype=torch.uint32, default=None)
    max_blocks_per_sequence: int = cpp_metadata(default=0)
    # NOTE: the KV-cache pool base pointers are deliberately absent. They are derived from
    # host_kv_cache_pool_pointers plus a per-layer intra-pool byte offset, which depends on
    # the resolved KV-cache element size, so they live in handwritten C++ lowering
    # (FmhaParams::kv_cache_pool_pointers) rather than in this schema.
    max_cyclic_attention_window_size: int = cpp_metadata(default=0)
    can_use_one_more_block: bool = cpp_metadata(default=False)
    sink_token_length: int = cpp_metadata(default=0)
    key_value_cache: Optional[torch.Tensor] = cpp_metadata(ctype=None, default=None)
    out_sf_scale: Optional[torch.Tensor] = cpp_metadata(ctype=torch.float32, default=None)
    alibi_slopes: Optional[torch.Tensor] = cpp_metadata(default=None)
    logn_scaling_ptr: Optional[torch.Tensor] = cpp_metadata(ctype=torch.float32, default=None)
    runtime_perf_knobs: Optional[torch.Tensor] = cpp_metadata(ctype=torch.int64, default=None)
    spec_decoding_mask: Optional[torch.Tensor] = cpp_metadata(ctype=torch.bool, default=None)
    sage_attn_sfs_q: Optional[torch.Tensor] = cpp_metadata(ctype=torch.float32, default=None)
    sage_attn_sfs_k: Optional[torch.Tensor] = cpp_metadata(ctype=torch.float32, default=None)
    sage_attn_sfs_v: Optional[torch.Tensor] = cpp_metadata(ctype=torch.float32, default=None)
    attention_mask_stride: int = cpp_metadata(default=0)
    multi_ctas_kv_counter: Optional[torch.Tensor] = cpp_metadata(ctype=None, default=None)
    cross_kv_length: int = cpp_metadata(default=0)
    num_encoder_tokens: int = cpp_metadata(default=0)
    relative_attention_bias_stride: int = cpp_metadata(default=0)
    encoder_input_lengths: Optional[torch.Tensor] = cpp_metadata(ctype=torch.int32, default=None)

    @classmethod
    def _from_arguments(cls, arguments: Mapping[str, object], /, **overrides: object) -> FmhaParams:
        """Build compatibility parameters from same-named legacy arguments."""
        params_fields = dataclasses.fields(cls)
        fields_by_name = {field.name: field for field in params_fields}
        unknown = overrides.keys() - fields_by_name.keys()
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"FmhaParams has no field(s): {names}")
        values = {
            field.name: arguments[field.name] for field in params_fields if field.name in arguments
        }
        values.update(overrides)
        return cls(**values)

    def to_thop_params(self, context: object = None) -> Any:
        """Build native parameters from this Python interface."""
        from tensorrt_llm.bindings.internal import thop

        target = thop.FmhaParams()
        _lower_struct(target, self)
        _populate_nested_thop_params(target, self)
        return target


def _lower_struct(target: Any, source: object) -> None:
    """Copy a schema dataclass onto its native counterpart, recursing into nested ones.

    Names match one for one: both sides are generated from the same declaration, so
    there is no filtering and nothing can be dropped silently.
    """
    for python_field in dataclasses.fields(source):
        if CPP_METADATA_KEY not in python_field.metadata:
            continue
        value = getattr(source, python_field.name)
        if value is None:
            continue
        if dataclasses.is_dataclass(value):
            _lower_struct(getattr(target, python_field.name), value)
        else:
            setattr(target, python_field.name, value)


def _populate_nested_thop_params(target: Any, source: FmhaParams) -> None:
    """Lower the Python-only parameter objects that have no native counterpart."""
    rope_params = source.rope_params
    if rope_params is not None:
        target.rotary_embedding_dim = rope_params.dim
        target.rotary_embedding_base = rope_params.theta
        target.rotary_embedding_scale_type = rope_params.scale_type
        target.rotary_embedding_scale = rope_params.scale
        target.rotary_embedding_short_mscale = rope_params.short_m_scale
        target.rotary_embedding_long_mscale = rope_params.long_m_scale
        target.rotary_embedding_max_positions = rope_params.max_positions
        target.rotary_embedding_original_max_positions = rope_params.original_max_positions

    forward_args = source.fwd
    if forward_args is None:
        return

    # Derived values: computed here rather than declared, so they have no field of
    # their own on either side.
    target.mask_type = forward_args.mask_type
    target.beam_width = 1 if source.is_cross else source.beam_width


class FmhaPhase(str, Enum):
    """Attention phase checked by a phased FMHA library."""

    CONTEXT = "context"
    GENERATION = "generation"


class Fmha(ABC):
    """Common runtime contract for TRT-LLM attention FMHA libraries."""

    supports_skip_correction: ClassVar[bool] = False

    def __init__(self, attn: "TrtllmAttention"):
        self._attn_ref: weakref.ReferenceType["TrtllmAttention"] = weakref.ref(attn)

    @property
    def attn(self) -> "TrtllmAttention":
        attn = self._attn_ref()
        if attn is None:
            raise RuntimeError("The owning TrtllmAttention instance has been garbage collected.")
        return attn

    @classmethod
    @final
    def is_available(cls, attn: "TrtllmAttention") -> bool:
        """Return whether this library can serve the given attention layer.

        Check shared capabilities before the implementation's
        ``_is_available`` hook. Libraries declare their capabilities as class
        attributes and override only the hook for additional static checks.

        Evaluated once per ``FmhaManager`` construction, currently at the end
        of ``TrtllmAttention.update_quant_config()``. Conditions must depend
        only on state finalized before manager construction and invariant for
        its lifetime. Reading state that a model rewrites later, such as a
        remapped ``layer_idx``, silently leaves the library list stale because
        it is not revalidated. Request-varying conditions belong in
        ``is_supported`` instead.
        """
        if attn.skip_correction_threshold > 0.0 and not cls.supports_skip_correction:
            logger.debug(
                f"{cls.__name__} is unavailable: skip-correction is enabled and unsupported."
            )
            return False
        return cls._is_available(attn)

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        """Check implementation-specific static restrictions after capability checks.

        Delegate to ``super()._is_available(attn)`` to reuse a parent hook;
        calling ``is_available`` here would re-enter the shared wrapper.
        """
        return True

    @final
    def is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        """Return whether this library supports the request or requested phase.

        Shared request capability checks belong here, before delegating to
        ``_is_supported``. Libraries override only that hook for their
        request-specific restrictions.

        Forward-varying selection conditions must be represented in
        ``FmhaManager._make_cache_key``. Conditions omitted from that key must
        remain invariant for the attention instance. Size-based conditions
        must also preserve the same result throughout each FMHA cache grid
        cell or add the relevant boundary to the grid's candidate list.
        """
        return self._is_supported(q, k, v, metadata, forward_args, phase=phase)

    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        """Check implementation-specific request restrictions after capability checks.

        Delegate to ``super()._is_supported(...)`` to reuse a parent hook;
        calling ``is_supported`` here would re-enter the shared wrapper.
        """
        return True

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> None:
        raise NotImplementedError
