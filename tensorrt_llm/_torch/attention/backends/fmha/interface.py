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
from functools import cache
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Optional, final

import torch

from tensorrt_llm._torch.attention.backends.cpp_schema import cpp_metadata
from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.functional import AttentionMaskType, PositionEmbeddingType, RotaryScalingType
    from tensorrt_llm.quantization.mode import QuantMode


_STATIC_CONFIG_DIRECT_ARGS = (
    "num_heads",
    "num_kv_heads",
    "head_size",
    "tokens_per_block",
    "position_embedding_type",
    "mask_type",
    "q_scaling",
    "is_spec_decoding_enabled",
    "is_mla_enable",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "predicted_tokens_per_seq",
    "rope_append",
    "sage_attn_num_elts_per_blk_q",
    "sage_attn_num_elts_per_blk_k",
    "sage_attn_num_elts_per_blk_v",
    "sage_attn_qk_int8",
)


@dataclass(kw_only=True, slots=True, frozen=True)
class StaticAttentionConfig:
    """The attention-layer configuration used to build native kernel runners."""

    num_heads: int = 0
    num_kv_heads: int = 0
    head_size: int = 0
    tokens_per_block: int = 0
    type: DataType = None
    is_fp8_out: bool = False
    is_fp4_out: bool = False
    use_kv_cache: bool = False
    paged_context_fmha: bool = False
    position_embedding_type: PositionEmbeddingType = 0
    mask_type: AttentionMaskType = 1
    q_scaling: float = 1.0
    rotary_embedding_dim: int = 0
    attn_logit_softcapping_scale: float = 0.0
    remove_padding: bool = True
    cross_attention: bool = False
    dense_context_fmha: bool = False
    fuses_dsv4_inv_rope_fp8_quant: bool = False
    use_sparse_attention: bool = False
    use_tllm_gen_sparse_attention: bool = False
    use_nvfp4_mla_kv_cache: bool = False
    is_spec_decoding_enabled: bool = False
    spec_decoding_target_max_gen_len: int = 0
    is_mla_enable: bool = False
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    predicted_tokens_per_seq: int = 1
    mla_layer_num: int = 0
    rope_append: bool = True
    sage_attn_num_elts_per_blk_q: int = 0
    sage_attn_num_elts_per_blk_k: int = 0
    sage_attn_num_elts_per_blk_v: int = 0
    sage_attn_qk_int8: bool = False
    quant_mode: QuantMode = 0
    skip_correction_threshold: float = 0.0

    @classmethod
    def from_params(
        cls,
        params: FmhaParams,
        *,
        skip_correction_threshold: float = 0.0,
    ) -> StaticAttentionConfig:
        """Capture the fixed runner-selection inputs from one attention call.

        Sourced from the layer and the batch metadata rather than from the phase
        carrier: everything here is fixed for the layer, so reading it off `attn` and
        `meta` keeps the phase carrier free of a second copy.
        """
        from tensorrt_llm._utils import torch_dtype_to_binding
        from tensorrt_llm.quantization.mode import QuantMode

        attn, meta, fwd = params.attn, params.meta, params.fwd
        if fwd is None:
            raise RuntimeError("StaticAttentionConfig requires forward args.")
        if params.output is None:
            raise RuntimeError("StaticAttentionConfig requires output.")
        query = params.qkv_input if params.qkv_input is not None else params.query_input
        if query is None:
            raise RuntimeError("StaticAttentionConfig requires qkv_input or query_input.")

        quant_mode = QuantMode(attn.quant_mode)
        sparse = fwd.sparse_runtime_params
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
        pool_mapping = meta.host_kv_cache_pool_mapping
        use_kv_cache = (
            meta.kv_cache_block_offsets is not None
            and meta.host_kv_cache_pool_pointers is not None
            and pool_mapping is not None
        )
        target_max_gen_len = 0
        if meta.max_total_draft_tokens is not None:
            target_max_gen_len = meta.max_total_draft_tokens + 1

        return cls(
            num_heads=attn.num_heads,
            num_kv_heads=attn.num_kv_heads,
            head_size=attn.head_dim,
            tokens_per_block=params.tokens_per_block,
            type=torch_dtype_to_binding(query.dtype),
            is_fp8_out=params.output.dtype == torch.float8_e4m3fn,
            is_fp4_out=params.output.dtype == torch.uint8,
            use_kv_cache=use_kv_cache,
            paged_context_fmha=meta.use_paged_context_fmha,
            position_embedding_type=attn.position_embedding_type,
            mask_type=fwd.mask_type,
            q_scaling=attn.q_scaling,
            rotary_embedding_dim=attn.rope_params.dim if attn.rope_params is not None else 0,
            cross_attention=params.is_cross,
            fuses_dsv4_inv_rope_fp8_quant=fwd.enable_dsv4_epilogue_fusion,
            use_sparse_attention=has_sparse_attention,
            use_tllm_gen_sparse_attention=use_tllm_gen_sparse_attention,
            use_nvfp4_mla_kv_cache=(
                quant_mode.has_fp4_kv_cache()
                and use_tllm_gen_sparse_attention
                and sparse.sparse_attn_kv_lens is None
                and sparse.aux_kv_cache_pool_ptr is not None
            ),
            is_spec_decoding_enabled=meta.is_spec_decoding_enabled,
            spec_decoding_target_max_gen_len=target_max_gen_len,
            is_mla_enable=attn.is_mla_enable,
            q_lora_rank=attn.q_lora_rank or 0,
            kv_lora_rank=attn.kv_lora_rank or 0,
            qk_nope_head_dim=attn.qk_nope_head_dim or 0,
            qk_rope_head_dim=attn.qk_rope_head_dim or 0,
            v_head_dim=attn.v_head_dim or 0,
            predicted_tokens_per_seq=attn.predicted_tokens_per_seq,
            mla_layer_num=pool_mapping.size(0) if pool_mapping is not None else 0,
            rope_append=attn.rope_append is not False,
            sage_attn_num_elts_per_blk_q=fwd.sage_attn_num_elts_per_blk_q,
            sage_attn_num_elts_per_blk_k=fwd.sage_attn_num_elts_per_blk_k,
            sage_attn_num_elts_per_blk_v=fwd.sage_attn_num_elts_per_blk_v,
            sage_attn_qk_int8=fwd.sage_attn_qk_int8,
            quant_mode=quant_mode,
            skip_correction_threshold=skip_correction_threshold,
        )

    @classmethod
    def from_legacy_arguments(cls, arguments: Mapping[str, Any]) -> StaticAttentionConfig:
        """Build the config for the flat compatibility entry point.

        That entry point has no layer object to read from, so the values come from its
        own arguments; `from_params` covers every other caller.
        """
        from tensorrt_llm._utils import torch_dtype_to_binding
        from tensorrt_llm.quantization.mode import QuantMode

        def arg(name: str, default: Any = None) -> Any:
            value = arguments.get(name, default)
            return default if value is None else value

        quant_mode = QuantMode(arg("quant_mode", 0))
        attn_indices = arg("sparse_attn_indices")
        has_attn_indices = attn_indices is not None and attn_indices.numel() > 0
        kv_indices = arg("sparse_kv_indices")
        attn_offsets = arg("sparse_attn_offsets")
        use_tllm_gen_sparse = has_attn_indices and not (
            attn_offsets is not None and attn_offsets.numel() > 0
        )
        pool_mapping = arg("host_kv_cache_pool_mapping")
        output = arguments["output"]
        direct = {
            name: arguments[name]
            for name in _STATIC_CONFIG_DIRECT_ARGS
            if arguments.get(name) is not None
        }
        return cls(
            **direct,
            type=torch_dtype_to_binding(arguments["q"].dtype),
            is_fp8_out=output.dtype == torch.float8_e4m3fn,
            is_fp4_out=output.dtype == torch.uint8,
            use_kv_cache=(
                arg("kv_cache_block_offsets") is not None
                and arg("host_kv_cache_pool_pointers") is not None
                and pool_mapping is not None
            ),
            paged_context_fmha=bool(arg("use_paged_context_fmha", False)),
            rotary_embedding_dim=int(arg("rope_dim", 0)),
            cross_attention=bool(arg("is_cross", False)),
            fuses_dsv4_inv_rope_fp8_quant=bool(arg("enable_dsv4_epilogue_fusion", False)),
            use_sparse_attention=(kv_indices is not None and kv_indices.numel() > 0)
            or has_attn_indices,
            use_tllm_gen_sparse_attention=use_tllm_gen_sparse,
            use_nvfp4_mla_kv_cache=(
                quant_mode.has_fp4_kv_cache()
                and use_tllm_gen_sparse
                and arg("sparse_attn_kv_lens") is None
                and arg("aux_kv_cache_pool_ptr") is not None
            ),
            mla_layer_num=pool_mapping.size(0) if pool_mapping is not None else 0,
            quant_mode=quant_mode,
        )

    def to_thop_config(self) -> Any:
        """Build the native constructor configuration."""
        from tensorrt_llm.bindings.internal import thop

        target = thop.StaticAttentionConfig()
        build_op_params(target, self)
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

    fwd: AttentionForwardArgs = None
    # Objects whose types have no native schema are Python-only.
    # FMHA backends that need layer/metadata state the flat schema does not carry
    # (e.g. the Triton custom-mask backend) read them from here.
    local_layer_idx: int = -1
    has_fp8_kv_cache: bool = False
    kv_pool: Optional[torch.Tensor] = None
    use_paged_context_fmha: bool = False
    kv_factor: int = 1
    seq_offset: int = 0
    num_seqs: int = 0
    num_tokens: int = 0
    # First query token of this phase on the axis of the q handed to the
    # library. Phase tensors are already sliced; separate per-token inputs
    # such as sparse block tables use this offset to select the same phase.
    token_offset: int = 0
    num_requests: int = 0

    layer_idx: int = -1
    rotary_embedding_base: float = 10000.0
    rotary_embedding_scale_type: RotaryScalingType = 0
    rotary_embedding_scale: float = 1.0
    rotary_embedding_short_mscale: float = 1.0
    rotary_embedding_long_mscale: float = 1.0
    rotary_embedding_max_positions: int = 1024
    rotary_embedding_original_max_positions: int = 1024
    max_context_length: int = 0
    max_seq_len: int = 0
    max_num_requests: int = 0
    # Total sequence rows (max_num_requests * beam_width). The generation workspace and
    # the multi-block counter are sized per sequence, not per request.
    max_num_sequences: int = 0
    # Total number of sequences, i.e. max_num_requests * beam_width. The generation
    # workspace and the multi-block counter are sized per sequence, not per request.
    beam_width: int = 1
    use_spec_decoding: bool = False
    is_spec_dec_tree: bool = True
    force_prepare_spec_dec_tree_mask: bool = False
    attention_chunk_size: Optional[int] = None

    spec_decoding_target_max_draft_tokens: Optional[int] = None

    workspace: torch.Tensor = None
    output: torch.Tensor = None
    qkv_or_q: torch.Tensor = None
    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None

    sequence_length: torch.Tensor = cpp_metadata(dtype=torch.int32)
    host_past_key_value_lengths: torch.Tensor = None
    # CPU totals [context, generation] for callers whose past lengths exclude new tokens.
    host_total_kv_lens: Optional[torch.Tensor] = None
    context_lengths: torch.Tensor = cpp_metadata(dtype=torch.int32)
    host_context_lengths: torch.Tensor = None
    max_context_q_len_override: Optional[int] = None
    kv_cache_block_offsets: Optional[torch.Tensor] = None
    host_kv_cache_pool_pointers: Optional[torch.Tensor] = None
    host_kv_cache_pool_mapping: Optional[torch.Tensor] = None
    cache_indirection: Optional[torch.Tensor] = cpp_metadata(dtype=torch.int32)
    max_attention_window_size: int = 0
    cyclic_attention_window_size: int = 0

    rotary_inv_freq: Optional[torch.Tensor] = cpp_metadata(dtype=torch.float32)
    rotary_cos_sin: Optional[torch.Tensor] = None

    block_ids_per_seq: Optional[torch.Tensor] = cpp_metadata(dtype=torch.int32)

    helix_position_offsets: Optional[torch.Tensor] = cpp_metadata(dtype=torch.int32)
    helix_is_inactive_rank: Optional[torch.Tensor] = cpp_metadata(dtype=torch.bool)

    spec_decoding_generation_lengths: Optional[torch.Tensor] = cpp_metadata(dtype=torch.int32)
    spec_decoding_position_offsets: Optional[torch.Tensor] = cpp_metadata(dtype=torch.int32)
    spec_decoding_packed_mask: Optional[torch.Tensor] = cpp_metadata(dtype=torch.int32)
    spec_decoding_bl_tree_mask_offset: Optional[torch.Tensor] = cpp_metadata(dtype=torch.int64)
    spec_decoding_bl_tree_mask: Optional[torch.Tensor] = cpp_metadata(dtype=torch.uint32)
    spec_bl_tree_first_sparse_mask_offset_kv: Optional[torch.Tensor] = cpp_metadata(
        dtype=torch.int32
    )

    num_sparse_topk: int = 0

    flash_mla_tile_scheduler_metadata: Optional[torch.Tensor] = cpp_metadata(dtype=torch.int32)
    flash_mla_num_splits: Optional[torch.Tensor] = cpp_metadata(dtype=torch.int32)

    trtllm_gen_jit_warmup: bool = False

    is_cross: bool = False

    # Fused kv_a_layernorm for the DSv4 sparse context path: when set, `latent_cache`
    # is the raw kv_a_proj output and the context RoPE kernel norms it in place.

    # Mechanical state consumed by handwritten C++ lowering hooks. Defaults
    # remain Python-owned; the generated C++ holder is only value-initialized.
    # NOTE: the KV-cache pool base pointers are deliberately absent. They are derived from
    # host_kv_cache_pool_pointers plus a per-layer intra-pool byte offset, which depends on
    # the resolved KV-cache element size, so they live in handwritten C++ lowering
    # (FmhaParams::kv_cache_pool_pointers) rather than in this schema.
    multi_ctas_kv_counter: Optional[torch.Tensor] = None

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

    def to_op_params(self, context: object = None) -> Any:
        """Build native parameters from this Python interface."""
        from tensorrt_llm.bindings.internal import thop

        target = thop.FmhaParams()
        build_op_params(target, self)
        return target


@cache
def _native_field_names(source_type: type, native_type: type) -> tuple[str, ...]:
    """Resolve the schema boundary once, independently of per-call field values."""
    return tuple(
        field.name for field in dataclasses.fields(source_type) if hasattr(native_type, field.name)
    )


def build_op_params(target: Any, *sources: object) -> None:
    """Build native parameters from schema dataclasses, with later sources taking priority.

    The generated native bindings define which fields cross the boundary;
    Python-only fields are left alone. A later None suppresses an earlier value
    and leaves the native default intact, including for value-initialized nested structs.
    """
    values = {
        name: getattr(source, name)
        for source in sources
        for name in _native_field_names(type(source), type(target))
    }
    for name, value in values.items():
        if value is None:
            continue
        if dataclasses.is_dataclass(value):
            build_op_params(getattr(target, name), value)
        else:
            setattr(target, name, value)


class FmhaPhase(str, Enum):
    """Attention phase checked by a phased FMHA library."""

    CONTEXT = "context"
    GENERATION = "generation"


class Fmha(ABC):
    """Common runtime contract for TRT-LLM attention FMHA libraries."""

    supports_skip_correction: ClassVar[bool] = False
    supports_block_sparse_inputs: ClassVar[bool] = False
    supports_workspace_reclamation: bool = False
    supports_fp4_mla: ClassVar[bool] = False

    def __init__(self, attn: "TrtllmAttention"):
        self._attn_ref: weakref.ReferenceType["TrtllmAttention"] = weakref.ref(attn)

    def release(self) -> None:
        """Release implementation-owned resources before CUDA teardown."""

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
        if getattr(attn, "uses_fp4_mla_attention", False) and not cls.supports_fp4_mla:
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
        if (
            forward_args.sparse_runtime_params.block_sparse_inputs is not None
            and not self.supports_block_sparse_inputs
        ):
            logger.debug(f"{type(self).__name__} does not support block-sparse inputs.")
            return False
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
