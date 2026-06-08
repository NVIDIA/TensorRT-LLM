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
from typing import TYPE_CHECKING, ClassVar, Generic, Optional, TypeVar

import torch

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType, PositionEmbeddingType
from tensorrt_llm.logger import logger

from .interface import AttentionForwardArgs, AttentionInputType

if TYPE_CHECKING:
    from .trtllm import TrtllmAttention, TrtllmAttentionMetadata

T = TypeVar("T")


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
    kv_cache: Optional[DataType]
    output: torch.dtype


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


class FmhaQkvMode(Enum):
    fused_qkv = "fused_qkv"
    separate_qkv = "separate_qkv"
    separate_qkv_sage = "separate_qkv_sage"


class FmhaKvCacheUpdateMode(Enum):
    update = "update"
    no_update = "no_update"


class MlaRopeLayout(Enum):
    appended = "appended"
    separate = "separate"


def _accepted_value_name(value: object) -> str:
    if isinstance(value, Enum):
        return value.name
    return str(value)


@dataclass(frozen=True, slots=True)
class AcceptedValues(Generic[T]):
    values: frozenset[T] = field(default_factory=frozenset)

    def accepts(self, value: T) -> bool:
        return value in self.values

    def describe(self) -> str:
        return f"values={sorted(_accepted_value_name(value) for value in self.values)}"


@dataclass(frozen=True, slots=True)
class AcceptedFeatureSet:
    values: frozenset[FmhaFeature] = field(default_factory=frozenset)

    def accepts(self, features: frozenset[FmhaFeature]) -> bool:
        return features <= self.values

    def rejected(self, features: frozenset[FmhaFeature]) -> frozenset[FmhaFeature]:
        return features - self.values

    def describe(self) -> str:
        return f"values={sorted(feature.value for feature in self.values)}"


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
    tokens_per_block: AcceptedIntegerValues

    def accepts(self, head_dim_qk: int, head_dim_v: int, tokens_per_block: int) -> bool:
        return (
            self.head_dim_qk == head_dim_qk
            and self.head_dim_v == head_dim_v
            and self.tokens_per_block.accepts(tokens_per_block)
        )

    def describe(self) -> tuple[int, int, str]:
        return (self.head_dim_qk, self.head_dim_v, self.tokens_per_block.describe())


@dataclass(frozen=True, slots=True)
class MlaContextCase:
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    mla_rope_layout: MlaRopeLayout
    qkv_mode: FmhaQkvMode
    runtime_features: AcceptedFeatureSet
    required_runtime_features: frozenset[FmhaFeature] = field(default_factory=frozenset)

    def accepts(self, context: "FmhaSupportContext") -> bool:
        return (
            self.kv_lora_rank == context.kv_lora_rank
            and self.qk_nope_head_dim == context.qk_nope_head_dim
            and self.qk_rope_head_dim == context.qk_rope_head_dim
            and self.v_head_dim == context.v_head_dim
            and self.mla_rope_layout == context.mla_rope_layout
            and self.qkv_mode == context.qkv_mode
            and self.runtime_features.accepts(context.runtime_features)
            and self.required_runtime_features <= context.runtime_features
        )

    def describe(self) -> tuple[int, int, int, int, str, str, str, list[str]]:
        return (
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.mla_rope_layout.value,
            self.qkv_mode.value,
            self.runtime_features.describe(),
            sorted(feature.value for feature in self.required_runtime_features),
        )


def _accepted_attention_input_type() -> AcceptedValues[AttentionInputType]:
    return AcceptedValues(values=frozenset(AttentionInputType))


def _accepted_mask_type() -> AcceptedValues[AttentionMaskType]:
    return AcceptedValues(values=frozenset(AttentionMaskType))


def _accepted_position_embedding_type() -> AcceptedValues[PositionEmbeddingType]:
    return AcceptedValues(values=frozenset(PositionEmbeddingType))


def _accepted_runtime_features() -> AcceptedFeatureSet:
    return AcceptedFeatureSet(values=frozenset(FmhaFeature))


def _positive_integers() -> AcceptedIntegerValues:
    return AcceptedIntegerValues(min_value=1)


def _accepted_qkv_mode() -> AcceptedValues[FmhaQkvMode]:
    return AcceptedValues(values=frozenset(FmhaQkvMode))


def _accepted_kv_cache_update_mode() -> AcceptedValues[FmhaKvCacheUpdateMode]:
    return AcceptedValues(values=frozenset(FmhaKvCacheUpdateMode))


@dataclass(frozen=True, slots=True)
class StandardPhaseCapabilities:
    dtype_combinations: frozenset[DTypeCombination] = field(default_factory=frozenset)
    head_size: AcceptedIntegerValues = field(default_factory=_positive_integers)
    qkv_mode: AcceptedValues[FmhaQkvMode] = field(default_factory=_accepted_qkv_mode)
    kv_cache_update_mode: AcceptedValues[FmhaKvCacheUpdateMode] = field(
        default_factory=_accepted_kv_cache_update_mode
    )
    mask_type: AcceptedValues[AttentionMaskType] = field(default_factory=_accepted_mask_type)
    position_embedding_type: AcceptedValues[PositionEmbeddingType] = field(
        default_factory=_accepted_position_embedding_type
    )
    runtime_features: AcceptedFeatureSet = field(default_factory=_accepted_runtime_features)
    phase_features: AcceptedFeatureSet = field(default_factory=_accepted_runtime_features)
    required_runtime_features: frozenset[FmhaFeature] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class MlaPhaseCapabilities:
    dtype_combinations: frozenset[DTypeCombination] = field(default_factory=frozenset)
    context_cases: frozenset[MlaContextCase] = field(default_factory=frozenset)
    generation_cases: frozenset[MlaGenerationCase] = field(default_factory=frozenset)
    qkv_mode: AcceptedValues[FmhaQkvMode] = field(default_factory=_accepted_qkv_mode)
    kv_cache_update_mode: AcceptedValues[FmhaKvCacheUpdateMode] = field(
        default_factory=_accepted_kv_cache_update_mode
    )
    mask_type: AcceptedValues[AttentionMaskType] = field(default_factory=_accepted_mask_type)
    position_embedding_type: AcceptedValues[PositionEmbeddingType] = field(
        default_factory=_accepted_position_embedding_type
    )
    runtime_features: AcceptedFeatureSet = field(default_factory=_accepted_runtime_features)
    phase_features: AcceptedFeatureSet = field(default_factory=_accepted_runtime_features)
    required_runtime_features: frozenset[FmhaFeature] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class PhaseCapabilities:
    standard: Optional[StandardPhaseCapabilities] = None
    mla: Optional[MlaPhaseCapabilities] = None


@dataclass(frozen=True, slots=True)
class FmhaCapabilities:
    sm: AcceptedIntegerValues = field(default_factory=AcceptedIntegerValues)
    attention_input_type: AcceptedValues[AttentionInputType] = field(
        default_factory=_accepted_attention_input_type
    )
    runtime_features: AcceptedFeatureSet = field(default_factory=_accepted_runtime_features)
    required_runtime_features: frozenset[FmhaFeature] = field(default_factory=frozenset)
    tokens_per_block: AcceptedIntegerValues = field(default_factory=_positive_integers)
    beam_width: AcceptedIntegerValues = field(default_factory=_positive_integers)
    num_heads_per_kv_head: AcceptedIntegerValues = field(default_factory=_positive_integers)
    context: PhaseCapabilities = field(default_factory=PhaseCapabilities)
    generation: PhaseCapabilities = field(default_factory=PhaseCapabilities)


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
    qk_nope_head_dim: Optional[int]
    qk_rope_head_dim: Optional[int]
    v_head_dim: Optional[int]
    mla_rope_layout: Optional[MlaRopeLayout]
    is_fused_qkv: bool
    update_kv_cache: bool
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
    def qkv_mode(self) -> FmhaQkvMode:
        if self.is_fused_qkv:
            return FmhaQkvMode.fused_qkv
        if self.use_sage_attention:
            return FmhaQkvMode.separate_qkv_sage
        return FmhaQkvMode.separate_qkv

    @property
    def kv_cache_update_mode(self) -> FmhaKvCacheUpdateMode:
        if self.update_kv_cache:
            return FmhaKvCacheUpdateMode.update
        return FmhaKvCacheUpdateMode.no_update

    @property
    def dtype_combination(self) -> DTypeCombination:
        output_dtype = self.output_dtype if self.output_dtype is not None else self.q_dtype
        return DTypeCombination(self.q_dtype, self.kv_cache_dtype, output_dtype)

    @property
    def num_heads_per_kv_head(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def head_dim_qk(self) -> Optional[int]:
        if self.kv_lora_rank is None or self.qk_rope_head_dim is None:
            return None
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def head_dim_v(self) -> Optional[int]:
        return self.kv_lora_rank

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
            q_dtype=q.dtype,
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
            qk_nope_head_dim=attn.qk_nope_head_dim,
            qk_rope_head_dim=attn.qk_rope_head_dim,
            v_head_dim=attn.v_head_dim,
            mla_rope_layout=(
                None
                if attn.rope_append is None
                else MlaRopeLayout.appended
                if attn.rope_append
                else MlaRopeLayout.separate
            ),
            is_fused_qkv=forward_args.is_fused_qkv,
            update_kv_cache=forward_args.update_kv_cache,
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
        return type(self).__name__

    def _get_attention_layer(self) -> "TrtllmAttention":
        attention_layer = self._attention_layer_ref()
        if attention_layer is None:
            raise RuntimeError(f"{self.name} attention layer has been destroyed.")
        return attention_layer

    @classmethod
    def _not_supported(cls, reason: str) -> bool:
        logger.debug("FMHA %s unsupported: %s", cls.__name__, reason)
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
    phase: StandardPhaseCapabilities | MlaPhaseCapabilities,
    context: FmhaSupportContext,
) -> bool:
    if not phase.dtype_combinations:
        return True
    combo = context.dtype_combination
    if combo not in phase.dtype_combinations:
        return fmha_cls._not_supported(
            f"[{phase_name}] unsupported dtype combination: "
            f"Q={combo.q}, KV={combo.kv_cache}, O={combo.output}.",
        )
    return True


def _common_phase_properties_supported(
    fmha_cls: type[BaseFmha],
    phase_name: str,
    phase: StandardPhaseCapabilities | MlaPhaseCapabilities,
    context: FmhaSupportContext,
) -> bool:
    if not phase.mask_type.accepts(context.mask_type):
        return fmha_cls._not_supported(
            f"[{phase_name}] mask type {context.mask_type.name} is not accepted. "
            f"Accepted: {phase.mask_type.describe()}."
        )
    if not phase.position_embedding_type.accepts(context.position_embedding_type):
        return fmha_cls._not_supported(
            f"[{phase_name}] position embedding type {context.position_embedding_type.name} is not accepted. "
            f"Accepted: {phase.position_embedding_type.describe()}."
        )
    unavailable_features = phase.phase_features.rejected(context.phase_features)
    if unavailable_features:
        return fmha_cls._not_supported(
            f"[{phase_name}] feature(s) not accepted: "
            f"{', '.join(sorted(feature.value for feature in unavailable_features))}."
        )
    unavailable_runtime_features = phase.runtime_features.rejected(context.runtime_features)
    if unavailable_runtime_features:
        return fmha_cls._not_supported(
            f"[{phase_name}] runtime feature(s) are not accepted: "
            f"{', '.join(sorted(feature.value for feature in unavailable_runtime_features))}."
        )
    missing_required_features = phase.required_runtime_features - context.runtime_features
    if missing_required_features:
        return fmha_cls._not_supported(
            f"[{phase_name}] required runtime feature(s) are absent: "
            f"{', '.join(sorted(feature.value for feature in missing_required_features))}."
        )
    return _dtype_combo_supported(fmha_cls, phase_name, phase, context)


def _standard_phase_supported(
    fmha_cls: type[BaseFmha],
    phase_name: str,
    phase: StandardPhaseCapabilities,
    context: FmhaSupportContext,
) -> bool:
    if not phase.head_size.accepts(context.head_size):
        return fmha_cls._not_supported(
            f"[{phase_name}] head size {context.head_size} is not accepted. "
            f"Accepted: {phase.head_size.describe()}."
        )
    if not phase.qkv_mode.accepts(context.qkv_mode):
        return fmha_cls._not_supported(
            f"[{phase_name}] QKV mode {context.qkv_mode.value} is not accepted. "
            f"Accepted: {phase.qkv_mode.describe()}."
        )
    if not phase.kv_cache_update_mode.accepts(context.kv_cache_update_mode):
        return fmha_cls._not_supported(
            f"[{phase_name}] KV cache update mode {context.kv_cache_update_mode.value} is not accepted. "
            f"Accepted: {phase.kv_cache_update_mode.describe()}."
        )
    return _common_phase_properties_supported(fmha_cls, phase_name, phase, context)


def _mla_context_supported(
    fmha_cls: type[BaseFmha],
    phase_name: str,
    phase: MlaPhaseCapabilities,
    context: FmhaSupportContext,
) -> bool:
    missing_params = [
        name
        for name, value in (
            ("kv_lora_rank", context.kv_lora_rank),
            ("qk_nope_head_dim", context.qk_nope_head_dim),
            ("qk_rope_head_dim", context.qk_rope_head_dim),
            ("v_head_dim", context.v_head_dim),
            ("mla_rope_layout", context.mla_rope_layout),
        )
        if value is None or (isinstance(value, int) and value <= 0)
    ]
    if missing_params:
        return fmha_cls._not_supported(
            f"[{phase_name}][MLA] missing required MLA parameter(s): {', '.join(missing_params)}.",
        )

    if not any(case.accepts(context) for case in phase.context_cases):
        requested_case = (
            context.kv_lora_rank,
            context.qk_nope_head_dim,
            context.qk_rope_head_dim,
            context.v_head_dim,
            context.mla_rope_layout.value,
            context.qkv_mode.value,
            sorted(feature.value for feature in context.runtime_features),
        )
        accepted_cases = sorted(case.describe() for case in phase.context_cases)
        return fmha_cls._not_supported(
            f"[{phase_name}][MLA] case {requested_case} is not accepted. Accepted: {accepted_cases}.",
        )
    return True


def _mla_generation_supported(
    fmha_cls: type[BaseFmha],
    phase_name: str,
    phase: MlaPhaseCapabilities,
    context: FmhaSupportContext,
) -> bool:
    missing_params = [
        name
        for name, value in (
            ("kv_lora_rank", context.kv_lora_rank),
            ("qk_nope_head_dim", context.qk_nope_head_dim),
            ("qk_rope_head_dim", context.qk_rope_head_dim),
            ("v_head_dim", context.v_head_dim),
        )
        if value is None or value <= 0
    ]
    if missing_params:
        return fmha_cls._not_supported(
            f"[{phase_name}][MLA] missing required MLA parameter(s): {', '.join(missing_params)}.",
        )
    head_dim_qk = int(context.head_dim_qk)
    head_dim_v = int(context.head_dim_v)
    if context.head_size != head_dim_qk:
        return fmha_cls._not_supported(
            f"[{phase_name}][MLA] head_size ({context.head_size}) must match "
            f"kv_lora_rank + qk_rope_head_dim ({head_dim_qk}).",
        )

    tokens_per_block = context.tokens_per_block or 0
    if not any(
        case.accepts(head_dim_qk, head_dim_v, tokens_per_block) for case in phase.generation_cases
    ):
        accepted_cases = sorted(case.describe() for case in phase.generation_cases)
        return fmha_cls._not_supported(
            f"[{phase_name}][MLA] case "
            f"(head_dim_qk={head_dim_qk}, head_dim_v={head_dim_v}, tokens_per_block={tokens_per_block}) "
            "is not accepted. "
            f"Accepted: {accepted_cases}.",
        )
    return True


def _mla_phase_supported(
    fmha_cls: type[BaseFmha],
    phase_name: str,
    phase: MlaPhaseCapabilities,
    context: FmhaSupportContext,
) -> bool:
    if not phase.qkv_mode.accepts(context.qkv_mode):
        return fmha_cls._not_supported(
            f"[{phase_name}][MLA] QKV mode {context.qkv_mode.value} "
            f"is not accepted. Accepted: {phase.qkv_mode.describe()}."
        )
    if not phase.kv_cache_update_mode.accepts(context.kv_cache_update_mode):
        return fmha_cls._not_supported(
            f"[{phase_name}][MLA] KV cache update mode {context.kv_cache_update_mode.value} "
            f"is not accepted. Accepted: {phase.kv_cache_update_mode.describe()}."
        )
    if not _common_phase_properties_supported(fmha_cls, f"{phase_name}][MLA", phase, context):
        return False
    if phase_name == "Context":
        return _mla_context_supported(fmha_cls, phase_name, phase, context)
    return _mla_generation_supported(fmha_cls, phase_name, phase, context)


def _phase_supported(
    fmha_cls: type[BaseFmha],
    phase_name: str,
    phase: PhaseCapabilities,
    context: FmhaSupportContext,
) -> bool:
    if context.is_mla_enable:
        if phase.mla is None:
            return fmha_cls._not_supported(f"[{phase_name}][MLA] MLA is not supported.")
        return _mla_phase_supported(fmha_cls, phase_name, phase.mla, context)
    if phase.standard is None:
        return fmha_cls._not_supported(f"[{phase_name}] standard attention is not supported.")
    return _standard_phase_supported(fmha_cls, phase_name, phase.standard, context)


def _fmha_supported_by_capabilities(fmha_cls: type[BaseFmha], context: FmhaSupportContext) -> bool:
    capabilities = fmha_cls.capabilities

    if not capabilities.sm.accepts(context.sm):
        return fmha_cls._not_supported(
            f"SM{context.sm} is not accepted. Accepted: {capabilities.sm.describe()}."
        )
    if not capabilities.attention_input_type.accepts(context.attention_input_type):
        return fmha_cls._not_supported(
            f"attention input type {context.attention_input_type.name} is not accepted. "
            f"Accepted: {capabilities.attention_input_type.describe()}."
        )

    runtime_features = context.runtime_features
    missing_required_features = capabilities.required_runtime_features - runtime_features
    if missing_required_features:
        return fmha_cls._not_supported(
            "required runtime feature(s) are absent: "
            f"{', '.join(sorted(feature.value for feature in missing_required_features))}."
        )
    unavailable_features = capabilities.runtime_features.rejected(runtime_features)
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
        if not _phase_supported(fmha_cls, "Context", capabilities.context, context):
            return False

    if context.has_generation_phase:
        if not _phase_supported(fmha_cls, "Generation", capabilities.generation, context):
            return False
        if not capabilities.beam_width.accepts(context.beam_width):
            return fmha_cls._not_supported(
                f"[Generation] beam width {context.beam_width} is not accepted. "
                f"Accepted: {capabilities.beam_width.describe()}."
            )
        if not context.is_mla_enable:
            if not capabilities.num_heads_per_kv_head.accepts(context.num_heads_per_kv_head):
                return fmha_cls._not_supported(
                    f"[Generation] num_heads_per_kv_head {context.num_heads_per_kv_head} is not accepted. "
                    f"Accepted: {capabilities.num_heads_per_kv_head.describe()}."
                )

    if context.use_paged_kv_cache:
        tokens_per_block = context.tokens_per_block or 0
        if tokens_per_block <= 0:
            return fmha_cls._not_supported("tokens_per_block must be positive.")
        if not capabilities.tokens_per_block.accepts(tokens_per_block):
            return fmha_cls._not_supported(
                f"tokens_per_block {tokens_per_block} is not accepted. "
                f"Accepted: {capabilities.tokens_per_block.describe()}."
            )

    logger.debug("FMHA %s supported.", fmha_cls.__name__)
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
    # Mirrors static thop / AttentionOp prechecks; tensor presence and env-sensitive
    # checks stay in the backend call path.
    capabilities: ClassVar[FmhaCapabilities] = FmhaCapabilities(
        sm=AcceptedIntegerValues(min_value=0),
        attention_input_type=AcceptedValues(
            values=frozenset(
                {
                    AttentionInputType.mixed,
                    AttentionInputType.context_only,
                    AttentionInputType.generation_only,
                }
            )
        ),
        runtime_features=AcceptedFeatureSet(
            values=frozenset(
                {
                    FmhaFeature.kv_cache_manager,
                    FmhaFeature.paged_kv_cache,
                    FmhaFeature.output_buffer,
                    FmhaFeature.helix,
                    FmhaFeature.sage_attention,
                    FmhaFeature.sparse_attention,
                    FmhaFeature.skip_softmax_attention,
                }
            )
        ),
        required_runtime_features=frozenset(),
        tokens_per_block=AcceptedIntegerValues(min_value=1),
        beam_width=AcceptedIntegerValues(min_value=1),
        num_heads_per_kv_head=AcceptedIntegerValues(min_value=1),
        context=PhaseCapabilities(
            standard=StandardPhaseCapabilities(
                dtype_combinations=frozenset(
                    {
                        DTypeCombination(q_dtype, kv_cache_dtype, output_dtype)
                        for kv_cache_dtype in (
                            None,
                            DataType.HALF,
                            DataType.BF16,
                            DataType.FLOAT,
                            DataType.FP8,
                            DataType.INT8,
                        )
                        for q_dtype, output_dtype in (
                            (torch.float16, torch.float16),
                            (torch.float16, torch.float8_e4m3fn),
                            (torch.float16, torch.uint8),
                            (torch.bfloat16, torch.bfloat16),
                            (torch.bfloat16, torch.float8_e4m3fn),
                            (torch.bfloat16, torch.uint8),
                            (torch.float32, torch.float32),
                        )
                    }
                    | {
                        DTypeCombination(q_dtype, DataType.NVFP4, output_dtype)
                        for q_dtype, output_dtype in (
                            (torch.float16, torch.float16),
                            (torch.float16, torch.float8_e4m3fn),
                            (torch.float16, torch.uint8),
                            (torch.bfloat16, torch.bfloat16),
                            (torch.bfloat16, torch.float8_e4m3fn),
                            (torch.bfloat16, torch.uint8),
                        )
                    }
                ),
                head_size=AcceptedIntegerValues(
                    values=frozenset(
                        {32, 48, 64, 72, 80, 96, 104, 112, 128, 144, 160, 192, 224, 256}
                    )
                ),
                qkv_mode=AcceptedValues(
                    values=frozenset({FmhaQkvMode.fused_qkv, FmhaQkvMode.separate_qkv_sage})
                ),
                kv_cache_update_mode=AcceptedValues(
                    values=frozenset({FmhaKvCacheUpdateMode.update})
                ),
                mask_type=AcceptedValues(
                    values=frozenset(
                        {
                            AttentionMaskType.padding,
                            AttentionMaskType.causal,
                            AttentionMaskType.sliding_window_causal,
                            AttentionMaskType.bidirectional,
                            AttentionMaskType.bidirectionalglm,
                            AttentionMaskType.blocksparse,
                            AttentionMaskType.custom_mask,
                        }
                    )
                ),
                position_embedding_type=AcceptedValues(
                    values=frozenset(
                        {
                            PositionEmbeddingType.learned_absolute,
                            PositionEmbeddingType.rope_gptj,
                            PositionEmbeddingType.rope_gpt_neox,
                            PositionEmbeddingType.long_rope,
                            PositionEmbeddingType.alibi,
                            PositionEmbeddingType.alibi_with_scale,
                            PositionEmbeddingType.relative,
                            PositionEmbeddingType.chatglm,
                            PositionEmbeddingType.yarn,
                            PositionEmbeddingType.mrope,
                        }
                    )
                ),
                runtime_features=AcceptedFeatureSet(
                    values=frozenset(
                        {
                            FmhaFeature.kv_cache_manager,
                            FmhaFeature.paged_kv_cache,
                            FmhaFeature.output_buffer,
                            FmhaFeature.helix,
                            FmhaFeature.sage_attention,
                            FmhaFeature.sparse_attention,
                            FmhaFeature.skip_softmax_attention,
                        }
                    )
                ),
                phase_features=AcceptedFeatureSet(
                    values=frozenset({FmhaFeature.padded_input, FmhaFeature.position_shift})
                ),
            ),
            mla=MlaPhaseCapabilities(
                dtype_combinations=frozenset(
                    DTypeCombination(q_dtype, kv_cache_dtype, output_dtype)
                    for kv_cache_dtype in (DataType.HALF, DataType.BF16, DataType.FP8)
                    for q_dtype, output_dtype in (
                        (torch.float16, torch.float16),
                        (torch.float16, torch.float8_e4m3fn),
                        (torch.bfloat16, torch.bfloat16),
                        (torch.bfloat16, torch.float8_e4m3fn),
                    )
                ),
                context_cases=frozenset(
                    {
                        MlaContextCase(
                            512,
                            128,
                            64,
                            128,
                            MlaRopeLayout.appended,
                            FmhaQkvMode.separate_qkv,
                            AcceptedFeatureSet(
                                values=frozenset(
                                    {
                                        FmhaFeature.kv_cache_manager,
                                        FmhaFeature.paged_kv_cache,
                                        FmhaFeature.output_buffer,
                                    }
                                )
                            ),
                        ),
                        MlaContextCase(
                            512,
                            128,
                            64,
                            128,
                            MlaRopeLayout.appended,
                            FmhaQkvMode.fused_qkv,
                            AcceptedFeatureSet(
                                values=frozenset(
                                    {
                                        FmhaFeature.kv_cache_manager,
                                        FmhaFeature.paged_kv_cache,
                                        FmhaFeature.output_buffer,
                                        FmhaFeature.sparse_attention,
                                    }
                                )
                            ),
                            frozenset({FmhaFeature.sparse_attention}),
                        ),
                        MlaContextCase(
                            448,
                            128,
                            64,
                            128,
                            MlaRopeLayout.separate,
                            FmhaQkvMode.separate_qkv,
                            AcceptedFeatureSet(
                                values=frozenset(
                                    {
                                        FmhaFeature.kv_cache_manager,
                                        FmhaFeature.paged_kv_cache,
                                        FmhaFeature.output_buffer,
                                    }
                                )
                            ),
                        ),
                        MlaContextCase(
                            448,
                            128,
                            64,
                            128,
                            MlaRopeLayout.separate,
                            FmhaQkvMode.fused_qkv,
                            AcceptedFeatureSet(
                                values=frozenset(
                                    {
                                        FmhaFeature.kv_cache_manager,
                                        FmhaFeature.paged_kv_cache,
                                        FmhaFeature.output_buffer,
                                        FmhaFeature.sparse_attention,
                                    }
                                )
                            ),
                            frozenset({FmhaFeature.sparse_attention}),
                        ),
                    }
                ),
                qkv_mode=AcceptedValues(
                    values=frozenset({FmhaQkvMode.fused_qkv, FmhaQkvMode.separate_qkv})
                ),
                kv_cache_update_mode=AcceptedValues(
                    values=frozenset({FmhaKvCacheUpdateMode.update})
                ),
                mask_type=AcceptedValues(
                    values=frozenset(
                        {
                            AttentionMaskType.padding,
                            AttentionMaskType.causal,
                            AttentionMaskType.sliding_window_causal,
                            AttentionMaskType.bidirectional,
                            AttentionMaskType.bidirectionalglm,
                            AttentionMaskType.blocksparse,
                        }
                    )
                ),
                position_embedding_type=AcceptedValues(
                    values=frozenset(
                        {
                            PositionEmbeddingType.learned_absolute,
                            PositionEmbeddingType.rope_gptj,
                            PositionEmbeddingType.rope_gpt_neox,
                            PositionEmbeddingType.long_rope,
                            PositionEmbeddingType.alibi,
                            PositionEmbeddingType.alibi_with_scale,
                            PositionEmbeddingType.relative,
                            PositionEmbeddingType.chatglm,
                            PositionEmbeddingType.yarn,
                            PositionEmbeddingType.mrope,
                        }
                    )
                ),
                runtime_features=AcceptedFeatureSet(
                    values=frozenset(
                        {
                            FmhaFeature.kv_cache_manager,
                            FmhaFeature.paged_kv_cache,
                            FmhaFeature.output_buffer,
                            FmhaFeature.sparse_attention,
                        }
                    )
                ),
                phase_features=AcceptedFeatureSet(values=frozenset()),
                required_runtime_features=frozenset(
                    {FmhaFeature.kv_cache_manager, FmhaFeature.paged_kv_cache}
                ),
            ),
        ),
        generation=PhaseCapabilities(
            standard=StandardPhaseCapabilities(
                dtype_combinations=frozenset(
                    {
                        DTypeCombination(q_dtype, kv_cache_dtype, output_dtype)
                        for kv_cache_dtype in (
                            None,
                            DataType.HALF,
                            DataType.BF16,
                            DataType.FLOAT,
                            DataType.FP8,
                            DataType.INT8,
                        )
                        for q_dtype, output_dtype in (
                            (torch.float16, torch.float16),
                            (torch.float16, torch.float8_e4m3fn),
                            (torch.float16, torch.uint8),
                            (torch.bfloat16, torch.bfloat16),
                            (torch.bfloat16, torch.float8_e4m3fn),
                            (torch.bfloat16, torch.uint8),
                            (torch.float32, torch.float32),
                        )
                    }
                    | {
                        DTypeCombination(q_dtype, DataType.NVFP4, output_dtype)
                        for q_dtype, output_dtype in (
                            (torch.float16, torch.float16),
                            (torch.float16, torch.float8_e4m3fn),
                            (torch.float16, torch.uint8),
                            (torch.bfloat16, torch.bfloat16),
                            (torch.bfloat16, torch.float8_e4m3fn),
                            (torch.bfloat16, torch.uint8),
                        )
                    }
                ),
                head_size=AcceptedIntegerValues(
                    values=frozenset(
                        {32, 48, 64, 72, 80, 96, 104, 112, 128, 144, 160, 192, 224, 256}
                    )
                ),
                qkv_mode=AcceptedValues(
                    values=frozenset({FmhaQkvMode.fused_qkv, FmhaQkvMode.separate_qkv_sage})
                ),
                kv_cache_update_mode=AcceptedValues(
                    values=frozenset({FmhaKvCacheUpdateMode.update})
                ),
                mask_type=AcceptedValues(
                    values=frozenset(
                        {
                            AttentionMaskType.padding,
                            AttentionMaskType.causal,
                            AttentionMaskType.sliding_window_causal,
                            AttentionMaskType.bidirectional,
                            AttentionMaskType.bidirectionalglm,
                            AttentionMaskType.blocksparse,
                            AttentionMaskType.custom_mask,
                        }
                    )
                ),
                position_embedding_type=AcceptedValues(
                    values=frozenset(
                        {
                            PositionEmbeddingType.learned_absolute,
                            PositionEmbeddingType.rope_gptj,
                            PositionEmbeddingType.rope_gpt_neox,
                            PositionEmbeddingType.long_rope,
                            PositionEmbeddingType.alibi,
                            PositionEmbeddingType.alibi_with_scale,
                            PositionEmbeddingType.relative,
                            PositionEmbeddingType.chatglm,
                            PositionEmbeddingType.yarn,
                            PositionEmbeddingType.mrope,
                        }
                    )
                ),
                runtime_features=AcceptedFeatureSet(
                    values=frozenset(
                        {
                            FmhaFeature.kv_cache_manager,
                            FmhaFeature.paged_kv_cache,
                            FmhaFeature.output_buffer,
                            FmhaFeature.helix,
                            FmhaFeature.sage_attention,
                            FmhaFeature.sparse_attention,
                            FmhaFeature.skip_softmax_attention,
                        }
                    )
                ),
                phase_features=AcceptedFeatureSet(
                    values=frozenset({FmhaFeature.padded_input, FmhaFeature.position_shift})
                ),
            ),
            mla=MlaPhaseCapabilities(
                dtype_combinations=frozenset(
                    DTypeCombination(q_dtype, kv_cache_dtype, output_dtype)
                    for kv_cache_dtype in (DataType.HALF, DataType.BF16, DataType.FP8)
                    for q_dtype, output_dtype in (
                        (torch.float16, torch.float16),
                        (torch.float16, torch.float8_e4m3fn),
                        (torch.bfloat16, torch.bfloat16),
                        (torch.bfloat16, torch.float8_e4m3fn),
                    )
                ),
                generation_cases=frozenset(
                    {
                        MlaGenerationCase(576, 512, AcceptedIntegerValues(min_value=1)),
                        MlaGenerationCase(512, 448, AcceptedIntegerValues(min_value=1)),
                    }
                ),
                qkv_mode=AcceptedValues(values=frozenset({FmhaQkvMode.fused_qkv})),
                kv_cache_update_mode=AcceptedValues(
                    values=frozenset({FmhaKvCacheUpdateMode.update})
                ),
                mask_type=AcceptedValues(
                    values=frozenset(
                        {
                            AttentionMaskType.padding,
                            AttentionMaskType.causal,
                            AttentionMaskType.sliding_window_causal,
                            AttentionMaskType.bidirectional,
                            AttentionMaskType.bidirectionalglm,
                            AttentionMaskType.blocksparse,
                        }
                    )
                ),
                position_embedding_type=AcceptedValues(
                    values=frozenset(
                        {
                            PositionEmbeddingType.learned_absolute,
                            PositionEmbeddingType.rope_gptj,
                            PositionEmbeddingType.rope_gpt_neox,
                            PositionEmbeddingType.long_rope,
                            PositionEmbeddingType.alibi,
                            PositionEmbeddingType.alibi_with_scale,
                            PositionEmbeddingType.relative,
                            PositionEmbeddingType.chatglm,
                            PositionEmbeddingType.yarn,
                            PositionEmbeddingType.mrope,
                        }
                    )
                ),
                runtime_features=AcceptedFeatureSet(
                    values=frozenset(
                        {
                            FmhaFeature.kv_cache_manager,
                            FmhaFeature.paged_kv_cache,
                            FmhaFeature.output_buffer,
                            FmhaFeature.sparse_attention,
                        }
                    )
                ),
                phase_features=AcceptedFeatureSet(values=frozenset()),
                required_runtime_features=frozenset(
                    {FmhaFeature.kv_cache_manager, FmhaFeature.paged_kv_cache}
                ),
            ),
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
