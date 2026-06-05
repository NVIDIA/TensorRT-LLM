# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import weakref
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Optional

import torch

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType, PositionEmbeddingType
from tensorrt_llm.logger import logger

from .interface import AttentionForwardArgs, AttentionInputType

if TYPE_CHECKING:
    from .trtllm import TrtllmAttention, TrtllmAttentionMetadata


# ``AttentionForwardArgs`` fields that the fallback ``thop.attention`` call
# does not consume. Sync test requires every other field to map to a kwarg name,
# a @property on the dataclass, or a field that some @property transitively
# reads; entries here are exempt.
_THOP_EXCLUDED_FIELDS: frozenset = frozenset(
    {
        "topk_indices",  # DSA-only
        "attention_mask_data",  # custom-mask code path
        "out_scale_sf",  # promoted into ``out_scale`` for NVFP4 output
    }
)

# ``thop.attention`` kwargs hard-wired to a literal at the call site.
_THOP_LITERALS: dict = {
    "sparse_mla_topk_lens": None,
    "compressed_kv_cache_pool_ptr": None,
}


@dataclass(frozen=True, slots=True)
class DTypeCombination:
    q: torch.dtype
    kv_cache: DataType
    output: torch.dtype


class FmhaPhase(Enum):
    context = "context"
    generation = "generation"


class FmhaFeature(Enum):
    kv_cache_manager = "kv_cache_manager"
    paged_kv_cache = "paged_kv_cache"
    output_buffer = "output_buffer"
    cross_attention = "cross_attention"
    helix = "helix"
    sage_attention = "sage_attention"
    sparse_attention = "sparse_attention"
    skip_softmax_attention = "skip_softmax_attention"
    padded_input = "padded_input"
    position_shift = "position_shift"


@dataclass(frozen=True, slots=True)
class AcceptedIntegerValues:
    values: frozenset[int] = field(default_factory=frozenset)
    min_value: Optional[int] = None
    max_value: Optional[int] = None

    def accepts(self, value: int) -> bool:
        if self.values and value not in self.values:
            return False
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True

    def describe(self) -> str:
        accepted = []
        if self.values:
            accepted.append(f"values={sorted(self.values)}")
        if self.min_value is not None:
            accepted.append(f"min={self.min_value}")
        if self.max_value is not None:
            accepted.append(f"max={self.max_value}")
        return ", ".join(accepted) if accepted else "any"


@dataclass(frozen=True, slots=True)
class MlaGenerationCase:
    head_dim_qk: int
    head_dim_v: int
    tokens_per_block: int


def _all_attention_input_types() -> frozenset[AttentionInputType]:
    return frozenset(AttentionInputType)


def _all_attention_mask_types() -> frozenset[AttentionMaskType]:
    return frozenset(AttentionMaskType)


def _all_position_embedding_types() -> frozenset[PositionEmbeddingType]:
    return frozenset(PositionEmbeddingType)


def _all_fmha_phases() -> frozenset[FmhaPhase]:
    return frozenset(FmhaPhase)


def _all_fmha_features() -> frozenset[FmhaFeature]:
    return frozenset(FmhaFeature)


def _positive_integers() -> AcceptedIntegerValues:
    return AcceptedIntegerValues(min_value=1)


@dataclass(frozen=True, slots=True)
class PhaseCapabilities:
    dtype_combinations: frozenset[DTypeCombination] = field(default_factory=frozenset)
    head_sizes: AcceptedIntegerValues = field(default_factory=_positive_integers)
    mask_types: frozenset[AttentionMaskType] = field(default_factory=_all_attention_mask_types)
    position_embedding_types: frozenset[PositionEmbeddingType] = field(
        default_factory=_all_position_embedding_types
    )
    features: frozenset[FmhaFeature] = field(default_factory=_all_fmha_features)


@dataclass(frozen=True, slots=True)
class MlaCapabilities:
    phases: frozenset[FmhaPhase] = field(default_factory=_all_fmha_phases)
    generation_cases: frozenset[MlaGenerationCase] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class FmhaCapabilities:
    name: str
    sm_versions: AcceptedIntegerValues = field(default_factory=AcceptedIntegerValues)
    attention_input_types: frozenset[AttentionInputType] = field(
        default_factory=_all_attention_input_types
    )
    runtime_features: frozenset[FmhaFeature] = field(default_factory=_all_fmha_features)
    required_runtime_features: frozenset[FmhaFeature] = field(default_factory=frozenset)
    tokens_per_block: AcceptedIntegerValues = field(default_factory=_positive_integers)
    generation_beam_widths: AcceptedIntegerValues = field(default_factory=_positive_integers)
    generation_head_ratios: AcceptedIntegerValues = field(default_factory=_positive_integers)
    context: PhaseCapabilities = field(default_factory=PhaseCapabilities)
    generation: PhaseCapabilities = field(default_factory=PhaseCapabilities)
    mla: MlaCapabilities = field(default_factory=MlaCapabilities)


@dataclass(frozen=True, slots=True)
class FmhaSupportContext:
    sm: int
    q_dtype: torch.dtype
    kv_cache_dtype: Optional[DataType]
    output_dtype: Optional[torch.dtype]
    num_heads: int
    num_kv_heads: int
    head_size: int
    attention_input_type: AttentionInputType
    mask_type: AttentionMaskType
    position_embedding_type: PositionEmbeddingType
    beam_width: int
    tokens_per_block: Optional[int]
    has_kv_cache_manager: bool
    use_paged_kv_cache: bool
    is_mla_enable: bool
    kv_lora_rank: Optional[int]
    qk_rope_head_dim: Optional[int]
    has_context_phase: bool
    has_generation_phase: bool
    is_cross_attention: bool
    is_spec_decoding: bool
    is_padded: bool
    position_shift_enabled: bool
    active_helix: bool
    use_sage_attention: bool
    has_sparse_attention: bool
    has_skip_softmax_attention: bool

    @property
    def runtime_features(self) -> frozenset[FmhaFeature]:
        features: set[FmhaFeature] = set()
        if self.has_kv_cache_manager:
            features.add(FmhaFeature.kv_cache_manager)
        if self.use_paged_kv_cache:
            features.add(FmhaFeature.paged_kv_cache)
        if self.output_dtype is not None:
            features.add(FmhaFeature.output_buffer)
        if self.is_cross_attention:
            features.add(FmhaFeature.cross_attention)
        if self.active_helix:
            features.add(FmhaFeature.helix)
        if self.use_sage_attention:
            features.add(FmhaFeature.sage_attention)
        if self.has_sparse_attention:
            features.add(FmhaFeature.sparse_attention)
        if self.has_skip_softmax_attention:
            features.add(FmhaFeature.skip_softmax_attention)
        return frozenset(features)

    @property
    def phase_features(self) -> frozenset[FmhaFeature]:
        features: set[FmhaFeature] = set()
        if self.is_padded:
            features.add(FmhaFeature.padded_input)
        if self.position_shift_enabled:
            features.add(FmhaFeature.position_shift)
        return frozenset(features)

    @classmethod
    def build(
        cls,
        attn: "TrtllmAttention",
        q: torch.Tensor,
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> "FmhaSupportContext":
        kv_cache_manager = metadata.kv_cache_manager
        kv_cache_dtype = kv_cache_manager.dtype if kv_cache_manager is not None else None
        q_dtype = torch.float8_e4m3fn if kv_cache_dtype == DataType.NVFP4 else q.dtype
        output_dtype = forward_args.output.dtype if forward_args.output is not None else None

        attention_input_type = forward_args.attention_input_type
        has_context_phase = attention_input_type != AttentionInputType.generation_only
        has_generation_phase = attention_input_type != AttentionInputType.context_only

        sparse_attention_config = attn.sparse_attention_config
        has_skip_softmax_attention = (
            getattr(sparse_attention_config, "algorithm", None) == "skip_softmax"
        )
        has_sparse_attention = (
            sparse_attention_config is not None and not has_skip_softmax_attention
        )

        use_sage_attention = (
            forward_args.sage_attn_num_elts_per_blk_q > 0
            or forward_args.sage_attn_num_elts_per_blk_k > 0
            or forward_args.sage_attn_num_elts_per_blk_v > 0
        )

        return cls(
            sm=get_sm_version(),
            q_dtype=q_dtype,
            kv_cache_dtype=kv_cache_dtype,
            output_dtype=output_dtype,
            num_heads=attn.num_heads,
            num_kv_heads=attn.num_kv_heads,
            head_size=attn.head_dim,
            attention_input_type=attention_input_type,
            mask_type=AttentionMaskType(forward_args.mask_type),
            position_embedding_type=PositionEmbeddingType(attn.position_embedding_type),
            beam_width=metadata.beam_width,
            tokens_per_block=metadata.tokens_per_block,
            has_kv_cache_manager=kv_cache_manager is not None,
            use_paged_kv_cache=metadata.kv_cache_block_offsets is not None,
            is_mla_enable=attn.is_mla_enable,
            kv_lora_rank=attn.kv_lora_rank,
            qk_rope_head_dim=attn.qk_rope_head_dim,
            has_context_phase=has_context_phase,
            has_generation_phase=has_generation_phase,
            is_cross_attention=metadata.is_cross,
            is_spec_decoding=metadata.is_spec_decoding_enabled,
            is_padded=False,
            position_shift_enabled=False,
            active_helix=metadata.helix_position_offsets is not None,
            use_sage_attention=use_sage_attention,
            has_sparse_attention=has_sparse_attention,
            has_skip_softmax_attention=has_skip_softmax_attention,
        )


class BaseFmha:
    capabilities: ClassVar[FmhaCapabilities]

    def __init__(self, attention_layer: "TrtllmAttention") -> None:
        self._attention_layer_ref = weakref.ref(attention_layer)

    @property
    def name(self) -> str:
        return self.capabilities.name

    def _get_attention_layer(self) -> "TrtllmAttention":
        attention_layer = self._attention_layer_ref()
        if attention_layer is None:
            raise RuntimeError(f"{self.name} attention layer has been destroyed.")
        return attention_layer

    @classmethod
    def _not_supported(cls, reason: str) -> bool:
        logger.debug("FMHA %s unsupported: %s", cls.capabilities.name, reason)
        return False

    @classmethod
    def is_supported(cls, context: FmhaSupportContext) -> bool:
        return _fmha_supported_by_capabilities(cls, context)

    def forward(
        self,
        attn: "TrtllmAttention",
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> None:
        raise NotImplementedError


def _dtype_combo_supported(
    fmha_cls: type[BaseFmha],
    phase_name: str,
    phase: PhaseCapabilities,
    context: FmhaSupportContext,
) -> bool:
    if not phase.dtype_combinations:
        return True
    if context.kv_cache_dtype is None:
        return fmha_cls._not_supported(f"[{phase_name}] missing KV cache dtype.")
    output_dtype = context.output_dtype if context.output_dtype is not None else context.q_dtype
    combo = DTypeCombination(context.q_dtype, context.kv_cache_dtype, output_dtype)
    if combo not in phase.dtype_combinations:
        return fmha_cls._not_supported(
            f"[{phase_name}] unsupported dtype combination: "
            f"Q={context.q_dtype}, KV={context.kv_cache_dtype}, O={output_dtype}.",
        )
    return True


def _phase_supported(
    fmha_cls: type[BaseFmha],
    phase_name: str,
    phase: PhaseCapabilities,
    context: FmhaSupportContext,
) -> bool:
    if not phase.head_sizes.accepts(context.head_size):
        return fmha_cls._not_supported(
            f"[{phase_name}] head size {context.head_size} is not accepted. "
            f"Accepted: {phase.head_sizes.describe()}."
        )
    if context.mask_type not in phase.mask_types:
        return fmha_cls._not_supported(
            f"[{phase_name}] mask type {context.mask_type.name} is not accepted. "
            f"Accepted: {[mask.name for mask in sorted(phase.mask_types, key=int)]}."
        )
    if context.position_embedding_type not in phase.position_embedding_types:
        return fmha_cls._not_supported(
            f"[{phase_name}] position embedding type {context.position_embedding_type.name} is not accepted. "
            "Accepted: "
            f"{[position.name for position in sorted(phase.position_embedding_types, key=int)]}."
        )
    unavailable_features = context.phase_features - phase.features
    if unavailable_features:
        return fmha_cls._not_supported(
            f"[{phase_name}] feature(s) not accepted: "
            f"{', '.join(sorted(feature.value for feature in unavailable_features))}."
        )
    return _dtype_combo_supported(fmha_cls, phase_name, phase, context)


def _mla_generation_supported(fmha_cls: type[BaseFmha], context: FmhaSupportContext) -> bool:
    mla = fmha_cls.capabilities.mla
    missing_params = [
        name
        for name, value in (
            ("kv_lora_rank", context.kv_lora_rank),
            ("qk_rope_head_dim", context.qk_rope_head_dim),
        )
        if value is None or value <= 0
    ]
    if missing_params:
        return fmha_cls._not_supported(
            f"[Generation][MLA] missing required MLA parameter(s): {', '.join(missing_params)}.",
        )

    kv_rank = int(context.kv_lora_rank)
    qk_rope_dim = int(context.qk_rope_head_dim)
    head_dim_qk = kv_rank + qk_rope_dim
    head_dim_v = kv_rank
    if context.head_size != head_dim_qk:
        return fmha_cls._not_supported(
            f"[Generation][MLA] head_size ({context.head_size}) must match "
            f"kv_lora_rank + qk_rope_head_dim ({head_dim_qk}).",
        )

    tokens_per_block = context.tokens_per_block or 0
    generation_case = MlaGenerationCase(head_dim_qk, head_dim_v, tokens_per_block)
    if mla.generation_cases and generation_case not in mla.generation_cases:
        accepted_cases = sorted(
            (
                case.head_dim_qk,
                case.head_dim_v,
                case.tokens_per_block,
            )
            for case in mla.generation_cases
        )
        return fmha_cls._not_supported(
            f"[Generation][MLA] case {generation_case} is not accepted. "
            f"Accepted: {accepted_cases}.",
        )
    return True


def _fmha_supported_by_capabilities(fmha_cls: type[BaseFmha], context: FmhaSupportContext) -> bool:
    capabilities = fmha_cls.capabilities

    if not capabilities.sm_versions.accepts(context.sm):
        return fmha_cls._not_supported(
            f"SM{context.sm} is not accepted. Accepted: {capabilities.sm_versions.describe()}."
        )
    if context.attention_input_type not in capabilities.attention_input_types:
        return fmha_cls._not_supported(
            f"attention input type {context.attention_input_type.name} is not accepted. "
            "Accepted: "
            f"{[input_type.name for input_type in sorted(capabilities.attention_input_types, key=int)]}."
        )

    runtime_features = context.runtime_features
    missing_required_features = capabilities.required_runtime_features - runtime_features
    if missing_required_features:
        return fmha_cls._not_supported(
            "required runtime feature(s) are absent: "
            f"{', '.join(sorted(feature.value for feature in missing_required_features))}."
        )
    unavailable_features = runtime_features - capabilities.runtime_features
    if unavailable_features:
        return fmha_cls._not_supported(
            "runtime feature(s) are not accepted: "
            f"{', '.join(sorted(feature.value for feature in unavailable_features))}."
        )

    if context.num_heads <= 0:
        return fmha_cls._not_supported("num_heads must be positive.")
    if context.num_kv_heads <= 0:
        return fmha_cls._not_supported("num_kv_heads must be positive.")
    if context.num_heads % context.num_kv_heads != 0:
        return fmha_cls._not_supported(
            f"num_heads ({context.num_heads}) must be divisible by num_kv_heads ({context.num_kv_heads}).",
        )

    if context.has_context_phase:
        if context.is_mla_enable and FmhaPhase.context not in capabilities.mla.phases:
            return fmha_cls._not_supported(
                "MLA context and mixed phases are not supported.",
            )
        if not context.is_mla_enable and not _phase_supported(
            fmha_cls, "Context", capabilities.context, context
        ):
            return False

    if context.has_generation_phase:
        if not _phase_supported(fmha_cls, "Generation", capabilities.generation, context):
            return False
        if not capabilities.generation_beam_widths.accepts(context.beam_width):
            return fmha_cls._not_supported(
                f"[Generation] beam width {context.beam_width} is not accepted. "
                f"Accepted: {capabilities.generation_beam_widths.describe()}."
            )
        if not context.is_mla_enable:
            heads_ratio = context.num_heads // context.num_kv_heads
            if not capabilities.generation_head_ratios.accepts(heads_ratio):
                return fmha_cls._not_supported(
                    f"[Generation] heads ratio {heads_ratio} is not accepted. "
                    f"Accepted: {capabilities.generation_head_ratios.describe()}."
                )
        if context.is_mla_enable:
            if FmhaPhase.generation not in capabilities.mla.phases:
                return fmha_cls._not_supported("[Generation][MLA] MLA generation is not supported.")
            if not _mla_generation_supported(fmha_cls, context):
                return False

    if context.use_paged_kv_cache:
        tokens_per_block = context.tokens_per_block or 0
        if tokens_per_block <= 0:
            return fmha_cls._not_supported("tokens_per_block must be positive.")
        if not capabilities.tokens_per_block.accepts(tokens_per_block):
            return fmha_cls._not_supported(
                f"tokens_per_block {tokens_per_block} is not accepted. "
                f"Accepted: {capabilities.tokens_per_block.describe()}."
            )

    logger.debug("FMHA %s supported.", capabilities.name)
    return True


def call_thop_attention(
    attn: "TrtllmAttention",
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    metadata: "TrtllmAttentionMetadata",
    forward_args: AttentionForwardArgs,
) -> None:
    # Every kwarg sources from ``attn`` / ``metadata`` / ``forward_args``
    # (with ``forward_args.sparse`` for sparse-attn inputs) or a literal
    # allowlisted in ``_THOP_LITERALS``. ``test_attention_op_sync.py``
    # enforces this statically.
    thop.attention(
        q=q,
        k=k,
        v=v,
        output=forward_args.output,
        output_sf=forward_args.output_sf,
        workspace_=metadata.effective_workspace,
        # --- Per-step batch state (TrtllmAttentionMetadata) ---
        sequence_length=metadata.kv_lens_cuda_runtime,
        host_past_key_value_lengths=metadata.kv_lens_runtime,
        host_total_kv_lens=metadata.host_total_kv_lens,
        context_lengths=metadata.prompt_lens_cuda_runtime,
        host_context_lengths=metadata.prompt_lens_cpu_runtime,
        host_request_types=metadata.host_request_types_runtime,
        max_context_q_len_override=metadata.max_context_q_len_override,
        kv_cache_block_offsets=metadata.kv_cache_block_offsets,
        host_kv_cache_pool_pointers=metadata.host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping=metadata.host_kv_cache_pool_mapping,
        cache_indirection=metadata.cache_indirection,
        block_ids_per_seq=metadata.block_ids_per_seq,
        tokens_per_block=metadata.tokens_per_block,
        max_num_requests=metadata.max_num_requests,
        beam_width=metadata.beam_width,
        use_paged_context_fmha=metadata.use_paged_context_fmha,
        helix_position_offsets=metadata.helix_position_offsets,
        helix_is_inactive_rank=metadata.helix_is_inactive_rank,
        is_spec_decoding_enabled=metadata.is_spec_decoding_enabled,
        use_spec_decoding=metadata.use_spec_decoding,
        is_spec_dec_tree=metadata.is_spec_dec_tree,
        spec_decoding_generation_lengths=metadata.spec_decoding_generation_lengths,
        spec_decoding_position_offsets_for_cpp=metadata.spec_decoding_position_offsets_for_cpp,
        spec_decoding_packed_mask=metadata.spec_decoding_packed_mask,
        spec_decoding_bl_tree_mask_offset=metadata.spec_decoding_bl_tree_mask_offset,
        spec_decoding_bl_tree_mask=metadata.spec_decoding_bl_tree_mask,
        spec_bl_tree_first_sparse_mask_offset_kv=metadata.spec_bl_tree_first_sparse_mask_offset_kv,
        num_sparse_topk=metadata.num_sparse_topk,
        flash_mla_tile_scheduler_metadata=metadata.flash_mla_tile_scheduler_metadata,
        flash_mla_num_splits=metadata.flash_mla_num_splits,
        num_contexts=metadata.num_contexts,
        num_ctx_tokens=metadata.num_ctx_tokens,
        max_context_length=metadata.max_context_length,
        # --- Per-call (AttentionForwardArgs) ---
        out_scale=forward_args.out_scale,
        kv_scale_orig_quant=forward_args.kv_scale_orig_quant,
        kv_scale_quant_orig=forward_args.kv_scale_quant_orig,
        latent_cache=forward_args.latent_cache,
        q_pe=forward_args.q_pe,
        attention_sinks=forward_args.attention_sinks,
        mask_type=forward_args.mask_type,
        attention_input_type=int(forward_args.attention_input_type),
        attention_window_size=forward_args.attention_window_size,
        chunked_prefill_buffer_batch_size=forward_args.chunked_prefill_buffer_batch_size,
        mrope_rotary_cos_sin=forward_args.mrope_rotary_cos_sin,
        mrope_position_deltas=forward_args.mrope_position_deltas,
        softmax_stats_tensor=forward_args.softmax_stats_tensor,
        cu_q_seqlens=forward_args.cu_q_seqlens,
        cu_kv_seqlens=forward_args.cu_kv_seqlens,
        fmha_scheduler_counter=forward_args.fmha_scheduler_counter,
        mla_bmm1_scale=forward_args.mla_bmm1_scale,
        mla_bmm2_scale=forward_args.mla_bmm2_scale,
        quant_q_buffer=forward_args.quant_q_buffer,
        sage_attn_num_elts_per_blk_q=forward_args.sage_attn_num_elts_per_blk_q,
        sage_attn_num_elts_per_blk_k=forward_args.sage_attn_num_elts_per_blk_k,
        sage_attn_num_elts_per_blk_v=forward_args.sage_attn_num_elts_per_blk_v,
        sage_attn_qk_int8=forward_args.sage_attn_qk_int8,
        is_fused_qkv=forward_args.is_fused_qkv,
        update_kv_cache=forward_args.update_kv_cache,
        # --- Module config (TrtllmAttention) ---
        rotary_inv_freq=attn.rotary_inv_freq,
        rotary_cos_sin=attn.rotary_cos_sin,
        predicted_tokens_per_seq=attn.predicted_tokens_per_seq,
        local_layer_idx=attn.local_layer_idx,
        num_heads=attn.num_heads,
        num_kv_heads=attn.num_kv_heads,
        head_size=attn.head_dim,
        quant_mode=attn.quant_mode,
        q_scaling=attn.q_scaling,
        position_embedding_type=attn.position_embedding_type,
        rope_dim=attn.rope_dim,
        rope_base=attn.rope_base,
        rope_scale_type=attn.rope_scale_type,
        rope_scale=attn.rope_scale,
        rope_short_m_scale=attn.rope_short_m_scale,
        rope_long_m_scale=attn.rope_long_m_scale,
        rope_max_positions=attn.rope_max_positions,
        rope_original_max_positions=attn.rope_original_max_positions,
        is_mla_enable=attn.is_mla_enable,
        q_lora_rank=attn.q_lora_rank,
        kv_lora_rank=attn.kv_lora_rank,
        qk_nope_head_dim=attn.qk_nope_head_dim,
        qk_rope_head_dim=attn.qk_rope_head_dim,
        v_head_dim=attn.v_head_dim,
        rope_append=attn.rope_append,
        attention_chunk_size=attn.attention_chunk_size,
        skip_softmax_threshold_scale_factor_prefill=attn.skip_softmax_threshold_scale_factor_prefill,
        skip_softmax_threshold_scale_factor_decode=attn.skip_softmax_threshold_scale_factor_decode,
        skip_softmax_stat=attn.skip_softmax_stat,
        # --- Sparse-specific (AttentionForwardArgs.sparse) ---
        sparse_kv_indices=forward_args.sparse.sparse_kv_indices,
        sparse_kv_offsets=forward_args.sparse.sparse_kv_offsets,
        sparse_attn_indices=forward_args.sparse.sparse_attn_indices,
        sparse_attn_offsets=forward_args.sparse.sparse_attn_offsets,
        sparse_attn_indices_block_size=forward_args.sparse.sparse_attn_indices_block_size,
        # --- Literals intentionally None (see _THOP_LITERALS) ---
        sparse_mla_topk_lens=None,
        compressed_kv_cache_pool_ptr=None,
    )


class FallbackFmha(BaseFmha):
    capabilities: ClassVar[FmhaCapabilities] = FmhaCapabilities(
        name="fallback",
        sm_versions=AcceptedIntegerValues(min_value=0),
        attention_input_types=frozenset(AttentionInputType),
        runtime_features=frozenset(FmhaFeature),
        required_runtime_features=frozenset(),
        tokens_per_block=AcceptedIntegerValues(min_value=1),
        generation_beam_widths=AcceptedIntegerValues(min_value=1),
        generation_head_ratios=AcceptedIntegerValues(min_value=1),
        context=PhaseCapabilities(
            head_sizes=AcceptedIntegerValues(min_value=1),
            mask_types=frozenset(AttentionMaskType),
            position_embedding_types=frozenset(PositionEmbeddingType),
            features=frozenset(FmhaFeature),
        ),
        generation=PhaseCapabilities(
            head_sizes=AcceptedIntegerValues(min_value=1),
            mask_types=frozenset(AttentionMaskType),
            position_embedding_types=frozenset(PositionEmbeddingType),
            features=frozenset(FmhaFeature),
        ),
        mla=MlaCapabilities(
            phases=frozenset(FmhaPhase),
            generation_cases=frozenset(),
        ),
    )

    def forward(
        self,
        attn: "TrtllmAttention",
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> None:
        call_thop_attention(attn, q, k, v, metadata, forward_args)
