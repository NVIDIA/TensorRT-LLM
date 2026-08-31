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
from typing import TYPE_CHECKING, Any, Mapping, NamedTuple, Optional, Protocol

import torch

from tensorrt_llm._torch.attention.backends.cpp_schema import CPP_METADATA_KEY, cpp_metadata
from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs

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
    """The part of an attention layer's shape that never changes.

    Passed once when the native op is built, so the op can derive its head counts,
    parallel layout and MLA flags up front. Everything that varies between calls --
    the mask type, the output dtype, speculative decoding -- stays in FmhaParams.
    """

    num_heads: int = cpp_metadata(default=0)
    num_kv_heads: int = cpp_metadata(default=0)
    head_size: int = cpp_metadata(default=0)
    tokens_per_block: int = cpp_metadata(default=0)
    is_mla_enable: bool = cpp_metadata(default=False)
    # Only the two MLA dimensions the op's own shape depends on.
    kv_lora_rank: int = cpp_metadata(default=0)
    qk_rope_head_dim: int = cpp_metadata(default=0)
    quant_mode: QuantMode = cpp_metadata(default=0)
    # Base-2 row-max threshold below which the MLA kernels skip correction.
    skip_correction_threshold: float = cpp_metadata(default=0.0)


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
    semaphores: Optional[torch.Tensor] = cpp_metadata(ctype=torch.int32, default=None)
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


class _CuteDslMlaStagingKey(NamedTuple):
    """Identifies CuTe-DSL MLA inputs staged into a shared workspace.

    Attributes:
        is_capturing: Whether the staging occurred during CUDA graph capture.
        workspace_ptr: Address of the shared staging workspace.
        block_tables_ptr: Address of the source block tables.
        block_tables_shape: Shape of the source block tables.
        sequence_lengths_ptr: Address of the source sequence lengths.
        sequence_lengths_offset: Offset applied to the source sequence lengths.
        batch_beam: Number of generation sequences, including beam expansion.
        padded_num_pages: Page-table width after CuTe-DSL alignment padding.
    """

    is_capturing: bool
    workspace_ptr: int
    block_tables_ptr: int
    block_tables_shape: tuple[int, ...]
    sequence_lengths_ptr: int
    sequence_lengths_offset: int
    batch_beam: int
    padded_num_pages: int


class MlaBackendPolicy(Protocol):
    """Selects the MLA generation backend for one scheduler batch."""

    def __call__(
        self,
        requested_backend: str,
        metadata: "TrtllmAttentionMetadata",
        num_gen_tokens: int,
    ) -> str:
        """Return the backend to use for the supplied batch composition.

        Args:
            requested_backend: Backend selected by the attention instance.
            metadata: Runtime metadata for the current scheduler batch.
            num_gen_tokens: Number of generation tokens in the batch.

        Returns:
            Backend name to use for MLA generation in this batch.
        """
        ...


class FmhaPhase(str, Enum):
    """Attention phase checked by a phased FMHA library."""

    CONTEXT = "context"
    GENERATION = "generation"


def fmha_scheduler_counter_elements(
    device: torch.device, num_heads: int, max_num_sequences: int
) -> int:
    """Number of int32 slots the multi-block scheduler counter needs.

    Python owns this buffer, so the sizing rule lives here, once. MMHA indexes it per
    (sequence, head) pair; the SM count is the lower bound that covers a kernel
    splitting a single head across several blocks.
    """
    return max(
        num_heads * max_num_sequences,
        torch.cuda.get_device_properties(device).multi_processor_count,
    )


def ensure_fmha_scheduler_counter(
    counter: Optional[torch.Tensor],
    device: torch.device,
    num_heads: int,
    max_num_sequences: int,
) -> torch.Tensor:
    """Return a zeroed scheduler counter, reusing ``counter`` when it already fits.

    The generation kernels dereference this buffer without a null check -- MMHA at
    ``params.block_counter[bhi]``, XQA at ``semaphores[idxSeq]`` -- so every caller that
    dispatches a generation phase must supply one. int32 specifically, not any 4-byte
    dtype: the native side reads it through ``data_ptr<int32_t>()``, which type-checks.
    """
    required_elements = fmha_scheduler_counter_elements(device, num_heads, max_num_sequences)
    if (
        counter is None
        or counter.device != device
        or counter.dtype != torch.int32
        or counter.numel() < required_elements
    ):
        counter = torch.empty(required_elements, dtype=torch.int32, device=device)
    counter.zero_()
    return counter


class Fmha(ABC):
    """Common runtime contract for TRT-LLM attention FMHA libraries."""

    supports_skip_correction = False

    def __init__(self, attn: "TrtllmAttention"):
        self._attn_ref: weakref.ReferenceType["TrtllmAttention"] = weakref.ref(attn)

    @property
    def attn(self) -> "TrtllmAttention":
        attn = self._attn_ref()
        if attn is None:
            raise RuntimeError("The owning TrtllmAttention instance has been garbage collected.")
        return attn

    @classmethod
    def is_available(cls, attn: "TrtllmAttention") -> bool:
        """Return whether this library can serve the given attention layer.

        Evaluated once per ``FmhaManager`` construction, currently at the end
        of ``TrtllmAttention.update_quant_config()``. Conditions must depend
        only on state finalized before manager construction and invariant for
        its lifetime. Reading state that a model rewrites later, such as a
        remapped ``layer_idx``, silently leaves the library list stale because
        it is not revalidated. Request-varying conditions belong in
        ``is_supported`` instead.
        """
        return True

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

        Forward-varying selection conditions must be represented in
        ``FmhaManager._make_cache_key``. Conditions omitted from that key must
        remain invariant for the attention instance. Size-based conditions
        must also preserve the same result throughout each FMHA cache grid
        cell or add the relevant boundary to the grid's candidate list.
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
