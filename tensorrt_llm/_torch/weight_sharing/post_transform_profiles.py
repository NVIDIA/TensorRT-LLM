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
"""Qualified model profiles for sharing post-transform weights.

Staged post-load hooks make it possible to receive weights that already use
their final runtime layout. They do not prove that every root model and feature
combination is safe to skip one-shot transforms. This module records the exact
profiles that have completed that qualification.

The registry deliberately matches root classes by identity rather than
``isinstance``. A subclass must have its own profile unless it was explicitly
qualified under the same architecture and lifecycle contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from torch import nn

# Stable identifier for the tensor names, layouts, aliases, and receiver-side
# finalization contract produced by the currently qualified Llama target path.
# Never change the meaning of an existing ABI ID. Introduce a new ID whenever
# a transform changes transferred tensor semantics or the receiver must
# interpret/finalize the transferred state differently.
LLAMA_POST_TRANSFORM_LAYOUT_ABI_V1 = "trtllm-llama-target-layout-v1"
# Stable contract for unquantized Qwen2 dense fused-QKV and fused-gate-up
# tensors plus the target-only receiver finalization used by its first profile.
QWEN2_DENSE_POST_TRANSFORM_LAYOUT_ABI_V1 = "trtllm-qwen2-dense-target-layout-v1"
# Stable contract for unquantized Qwen3 dense fused-QKV and fused-gate-up
# tensors, Q/K norm state, and target-only receiver finalization.
QWEN3_DENSE_POST_TRANSFORM_LAYOUT_ABI_V1 = "trtllm-qwen3-dense-target-layout-v1"
_MISSING = object()


def _canonical_string(value: object) -> str | None:
    if value is None:
        return None
    value = getattr(value, "value", value)
    if isinstance(value, (str, int, float, bool)):
        return str(value).removeprefix("torch.")
    rendered_value = str(value)
    return rendered_value.removeprefix("torch.") if rendered_value.startswith("torch.") else None


def _canonical_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _canonical_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _realized_rope_fusion(model: nn.Module) -> bool | None:
    realized_values: set[bool] = set()
    for module in model.modules():
        value = getattr(module, "rope_fusion", _MISSING)
        if value is _MISSING:
            continue
        canonical_value = _canonical_bool(value)
        if canonical_value is None:
            return None
        realized_values.add(canonical_value)
    return realized_values.pop() if len(realized_values) == 1 else None


def _canonical_optional_string(container: object, attribute: str) -> str | None:
    value = getattr(container, attribute, _MISSING)
    if value is _MISSING:
        return None
    return "none" if value is None else _canonical_string(value)


@dataclass(frozen=True)
class PostTransformConfigIdentity:
    """Canonical model identity captured before constructor normalization."""

    architecture: str | None
    model_type: str | None

    @classmethod
    def from_model_config(cls, model_config: object) -> "PostTransformConfigIdentity":
        """Capture registry dimensions from a resolved pre-construction config."""

        pretrained_config = getattr(model_config, "pretrained_config", None)
        architectures = getattr(pretrained_config, "architectures", None)
        architecture = (
            architectures[0]
            if isinstance(architectures, (list, tuple))
            and architectures
            and isinstance(architectures[0], str)
            else None
        )
        configured_model_type = getattr(pretrained_config, "model_type", None)
        model_type = configured_model_type if isinstance(configured_model_type, str) else None
        return cls(architecture=architecture, model_type=model_type)


@dataclass(frozen=True)
class PostTransformRuntimeConfig:
    """Resolved runtime dimensions used to bound qualification evidence."""

    dtype: str | None
    quant_algorithm: str | None
    kv_cache_quant_algorithm: str | None
    layerwise_quantization: bool | None
    force_dynamic_quantization: bool | None
    lora_enabled: bool | None
    sparse_attention_enabled: bool | None
    attention_backend: str | None
    moe_backend: str | None
    tp_size: int | None
    pp_size: int | None
    cp_size: int | None
    moe_tp_size: int | None
    moe_ep_size: int | None
    attention_tp_size: int | None
    attention_cp_size: int | None
    attention_dp: bool | None
    multi_node: bool | None
    tied_word_embeddings: bool | None
    rope_type: str | None
    rope_fusion: bool | None

    @classmethod
    def from_model_config(
        cls, model_config: object, *, model: nn.Module | None = None
    ) -> "PostTransformRuntimeConfig":
        """Capture the support-matrix dimensions from a final model config."""

        pretrained_config = getattr(model_config, "pretrained_config", None)
        quant_config = getattr(model_config, "quant_config", None)
        quant_config_dict = getattr(model_config, "quant_config_dict", _MISSING)
        lora_config = getattr(model_config, "lora_config", _MISSING)
        sparse_attention_config = getattr(model_config, "sparse_attention_config", _MISSING)
        mapping = getattr(model_config, "mapping", None)
        multi_node = _canonical_bool(mapping.is_multi_node()) if mapping is not None else None

        if quant_config_dict is _MISSING:
            layerwise_quantization = None
        elif quant_config_dict is None:
            layerwise_quantization = False
        elif isinstance(quant_config_dict, dict):
            layerwise_quantization = bool(quant_config_dict)
        else:
            layerwise_quantization = None

        rope_scaling = getattr(pretrained_config, "rope_scaling", None)
        if rope_scaling is None:
            rope_type = "default"
        elif isinstance(rope_scaling, dict):
            rope_type = _canonical_string(rope_scaling.get("rope_type", rope_scaling.get("type")))
        else:
            rope_type = None

        rope_fusion = _realized_rope_fusion(model) if model is not None else None

        return cls(
            dtype=_canonical_string(getattr(model_config, "torch_dtype", None)),
            quant_algorithm=_canonical_optional_string(quant_config, "quant_algo"),
            kv_cache_quant_algorithm=_canonical_optional_string(
                quant_config, "kv_cache_quant_algo"
            ),
            layerwise_quantization=layerwise_quantization,
            force_dynamic_quantization=_canonical_bool(
                getattr(model_config, "force_dynamic_quantization", None)
            ),
            lora_enabled=None if lora_config is _MISSING else lora_config is not None,
            sparse_attention_enabled=(
                None if sparse_attention_config is _MISSING else sparse_attention_config is not None
            ),
            attention_backend=_canonical_string(getattr(model_config, "attn_backend", None)),
            moe_backend=_canonical_string(getattr(model_config, "moe_backend", None)),
            tp_size=_canonical_int(getattr(mapping, "tp_size", None)),
            pp_size=_canonical_int(getattr(mapping, "pp_size", None)),
            cp_size=_canonical_int(getattr(mapping, "cp_size", None)),
            moe_tp_size=_canonical_int(getattr(mapping, "moe_tp_size", None)),
            moe_ep_size=_canonical_int(getattr(mapping, "moe_ep_size", None)),
            attention_tp_size=_canonical_int(getattr(mapping, "attn_tp_size", None)),
            attention_cp_size=_canonical_int(getattr(mapping, "attn_cp_size", None)),
            attention_dp=_canonical_bool(getattr(mapping, "enable_attention_dp", None)),
            multi_node=multi_node,
            tied_word_embeddings=_canonical_bool(
                getattr(pretrained_config, "tie_word_embeddings", None)
            ),
            rope_type=rope_type,
            rope_fusion=rope_fusion,
        )


@dataclass(frozen=True)
class PostTransformRuntimeConstraints:
    """Allowed values for one qualified runtime support-matrix row."""

    _DIMENSIONS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("dtypes", "dtype"),
        ("quant_algorithms", "quant_algorithm"),
        ("kv_cache_quant_algorithms", "kv_cache_quant_algorithm"),
        ("layerwise_quantization", "layerwise_quantization"),
        ("force_dynamic_quantization", "force_dynamic_quantization"),
        ("lora_enabled", "lora_enabled"),
        ("sparse_attention_enabled", "sparse_attention_enabled"),
        ("attention_backends", "attention_backend"),
        ("moe_backends", "moe_backend"),
        ("tp_sizes", "tp_size"),
        ("pp_sizes", "pp_size"),
        ("cp_sizes", "cp_size"),
        ("moe_tp_sizes", "moe_tp_size"),
        ("moe_ep_sizes", "moe_ep_size"),
        ("attention_tp_sizes", "attention_tp_size"),
        ("attention_cp_sizes", "attention_cp_size"),
        ("attention_dp", "attention_dp"),
        ("multi_node", "multi_node"),
        ("tied_word_embeddings", "tied_word_embeddings"),
        ("rope_types", "rope_type"),
        ("rope_fusion", "rope_fusion"),
    )

    dtypes: frozenset[str | None] | None = None
    quant_algorithms: frozenset[str | None] | None = None
    kv_cache_quant_algorithms: frozenset[str | None] | None = None
    layerwise_quantization: frozenset[bool | None] | None = None
    force_dynamic_quantization: frozenset[bool | None] | None = None
    lora_enabled: frozenset[bool | None] | None = None
    sparse_attention_enabled: frozenset[bool | None] | None = None
    attention_backends: frozenset[str | None] | None = None
    moe_backends: frozenset[str | None] | None = None
    tp_sizes: frozenset[int | None] | None = None
    pp_sizes: frozenset[int | None] | None = None
    cp_sizes: frozenset[int | None] | None = None
    moe_tp_sizes: frozenset[int | None] | None = None
    moe_ep_sizes: frozenset[int | None] | None = None
    attention_tp_sizes: frozenset[int | None] | None = None
    attention_cp_sizes: frozenset[int | None] | None = None
    attention_dp: frozenset[bool | None] | None = None
    multi_node: frozenset[bool | None] | None = None
    tied_word_embeddings: frozenset[bool | None] | None = None
    rope_types: frozenset[str | None] | None = None
    rope_fusion: frozenset[bool | None] | None = None

    def __post_init__(self) -> None:
        for constraint_name, _runtime_name in self._DIMENSIONS:
            allowed_values = getattr(self, constraint_name)
            if allowed_values is None:
                continue
            normalized_values = frozenset(allowed_values)
            if not normalized_values:
                raise ValueError(
                    f"Post-transform runtime constraint {constraint_name} must not be empty"
                )
            object.__setattr__(self, constraint_name, normalized_values)

    def unsupported_dimensions(
        self, runtime_config: PostTransformRuntimeConfig | None
    ) -> frozenset[str]:
        """Return constrained dimensions not satisfied by a concrete run."""

        unsupported = set()
        for constraint_name, runtime_name in self._DIMENSIONS:
            allowed_values = getattr(self, constraint_name)
            if allowed_values is None:
                continue
            runtime_value = (
                None if runtime_config is None else getattr(runtime_config, runtime_name)
            )
            if runtime_value not in allowed_values:
                unsupported.add(runtime_name)
        return frozenset(unsupported)

    def overlaps(self, other: "PostTransformRuntimeConstraints") -> bool:
        """Whether two rows admit at least one common runtime configuration."""

        for constraint_name, _runtime_name in self._DIMENSIONS:
            allowed_values = getattr(self, constraint_name)
            other_allowed_values = getattr(other, constraint_name)
            if (
                allowed_values is not None
                and other_allowed_values is not None
                and allowed_values.isdisjoint(other_allowed_values)
            ):
                return False
        return True


class PostTransformTransferScope(str, Enum):
    """The model component represented by a post-transform transfer."""

    TARGET_MODEL = "target_model"
    LANGUAGE_MODEL = "language_model"
    COMPLETE_MODEL = "complete_model"


class PostTransformFeature(str, Enum):
    """Optional lifecycle features that require explicit qualification."""

    SEPARATE_DRAFT_MODEL = "separate_draft_model"


class PostTransformQualificationReason(str, Enum):
    """Structured outcome of matching a request to a qualified profile."""

    QUALIFIED = "qualified"
    ROOT_MODEL_CLASS_NOT_REGISTERED = "root_model_class_not_registered"
    ARCHITECTURE_NOT_REGISTERED = "architecture_not_registered"
    MODEL_TYPE_NOT_REGISTERED = "model_type_not_registered"
    SPECULATIVE_MODE_NOT_REGISTERED = "speculative_mode_not_registered"
    PROTOCOL_NOT_REGISTERED = "protocol_not_registered"
    TRANSFER_SCOPE_NOT_REGISTERED = "transfer_scope_not_registered"
    RUNTIME_CONFIG_NOT_SUPPORTED = "runtime_config_not_supported"
    FEATURE_NOT_SUPPORTED = "feature_not_supported"


@dataclass(frozen=True)
class PostTransformProfile:
    """One exact root-model profile qualified for post-transform sharing."""

    profile_id: str
    root_model_class: type[nn.Module]
    architecture: str
    model_type: str
    speculative_mode: str | None
    protocol_version: int
    transfer_scope: PostTransformTransferScope
    transform_abi_id: str
    runtime_constraints: PostTransformRuntimeConstraints
    supported_features: frozenset[PostTransformFeature] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("Post-transform profile_id must not be empty")
        if not self.architecture:
            raise ValueError("Post-transform architecture must not be empty")
        if not self.model_type:
            raise ValueError("Post-transform model_type must not be empty")
        if self.speculative_mode == "":
            raise ValueError("Post-transform speculative_mode must not be empty")
        if self.protocol_version < 1:
            raise ValueError("Post-transform protocol_version must be positive")
        if not isinstance(self.transform_abi_id, str) or not self.transform_abi_id:
            raise ValueError("Post-transform transform_abi_id must be a non-empty string")
        object.__setattr__(self, "supported_features", frozenset(self.supported_features))
        if not isinstance(self.runtime_constraints, PostTransformRuntimeConstraints):
            raise TypeError(
                "Post-transform runtime_constraints must be PostTransformRuntimeConstraints"
            )


@dataclass(frozen=True)
class PostTransformQualificationDecision:
    """Result of looking up a requested post-transform receiver profile."""

    reason: PostTransformQualificationReason
    profile: PostTransformProfile | None = None
    unsupported_features: frozenset[PostTransformFeature] = field(default_factory=frozenset)
    unsupported_runtime_dimensions: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "unsupported_features", frozenset(self.unsupported_features))
        object.__setattr__(
            self,
            "unsupported_runtime_dimensions",
            frozenset(self.unsupported_runtime_dimensions),
        )
        if self.reason is PostTransformQualificationReason.QUALIFIED:
            if self.profile is None:
                raise ValueError("A qualified decision must include its profile")
            if self.unsupported_features or self.unsupported_runtime_dimensions:
                raise ValueError("A qualified decision cannot include unsupported dimensions")
        else:
            if self.unsupported_features and (
                self.reason is not PostTransformQualificationReason.FEATURE_NOT_SUPPORTED
            ):
                raise ValueError("Unsupported features require a feature_not_supported decision")
            if self.unsupported_runtime_dimensions and (
                self.reason is not PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED
            ):
                raise ValueError(
                    "Unsupported runtime dimensions require a runtime_config_not_supported decision"
                )
            if (
                self.reason is PostTransformQualificationReason.FEATURE_NOT_SUPPORTED
                and not self.unsupported_features
            ):
                raise ValueError("A feature_not_supported decision must identify features")
            if (
                self.reason is PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED
                and not self.unsupported_runtime_dimensions
            ):
                raise ValueError("A runtime_config_not_supported decision must identify dimensions")

    @property
    def qualified(self) -> bool:
        """Whether the request exactly matched a qualified profile."""

        return (
            self.reason is PostTransformQualificationReason.QUALIFIED and self.profile is not None
        )

    @property
    def transform_abi_id(self) -> str | None:
        """The qualified layout ABI, or `None` for a rejected profile."""

        return (
            self.profile.transform_abi_id if self.qualified and self.profile is not None else None
        )


@dataclass(frozen=True)
class PostTransformProfileRegistry:
    """Immutable collection of audited post-transform profiles."""

    profiles: tuple[PostTransformProfile, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "profiles", tuple(self.profiles))

        profile_ids: set[str] = set()
        profile_keys: dict[
            tuple[type[nn.Module], str, str, str | None, int, PostTransformTransferScope],
            list[PostTransformProfile],
        ] = {}
        for profile in self.profiles:
            if profile.profile_id in profile_ids:
                raise ValueError(f"Duplicate post-transform profile_id: {profile.profile_id!r}")
            profile_ids.add(profile.profile_id)

            key = (
                profile.root_model_class,
                profile.architecture,
                profile.model_type,
                profile.speculative_mode,
                profile.protocol_version,
                profile.transfer_scope,
            )
            for existing_profile in profile_keys.setdefault(key, []):
                if profile.runtime_constraints.overlaps(existing_profile.runtime_constraints):
                    raise ValueError(
                        "Duplicate post-transform profile for "
                        f"{profile.root_model_class.__name__}/{profile.architecture}/"
                        f"{profile.model_type}/{profile.speculative_mode or 'target-only'}/"
                        f"v{profile.protocol_version}/{profile.transfer_scope.value}: "
                        "runtime constraints overlap"
                    )
            profile_keys[key].append(profile)

    def qualify(
        self,
        *,
        root_model_class: type[nn.Module],
        architecture: str | None,
        model_type: str | None,
        speculative_mode: str | None,
        protocol_version: int,
        transfer_scope: PostTransformTransferScope,
        enabled_features: frozenset[PostTransformFeature] = frozenset(),
        runtime_config: PostTransformRuntimeConfig | None = None,
    ) -> PostTransformQualificationDecision:
        """Match a receiver request against the exact qualified profile set.

        Args:
            root_model_class: Exact constructed root model class.
            architecture: Canonical architecture from the resolved model config.
            model_type: Canonical Hugging Face model type.
            speculative_mode: Canonical speculative decoding mode, or `None`
                for target-only loading.
            protocol_version: Post-transform transfer protocol version.
            transfer_scope: Component represented by the transfer.
            enabled_features: Optional lifecycle features active for this load.
            runtime_config: Resolved dtype, quantization, backend, topology, and
                model-feature dimensions for the concrete load.

        Returns:
            A structured qualification decision and the matching profile, when
            one exists.
        """

        model_profiles = tuple(
            profile for profile in self.profiles if profile.root_model_class is root_model_class
        )
        if not model_profiles:
            return PostTransformQualificationDecision(
                PostTransformQualificationReason.ROOT_MODEL_CLASS_NOT_REGISTERED
            )

        architecture_profiles = tuple(
            profile for profile in model_profiles if profile.architecture == architecture
        )
        if not architecture_profiles:
            return PostTransformQualificationDecision(
                PostTransformQualificationReason.ARCHITECTURE_NOT_REGISTERED
            )

        model_type_profiles = tuple(
            profile for profile in architecture_profiles if profile.model_type == model_type
        )
        if not model_type_profiles:
            return PostTransformQualificationDecision(
                PostTransformQualificationReason.MODEL_TYPE_NOT_REGISTERED
            )

        speculative_mode_profiles = tuple(
            profile
            for profile in model_type_profiles
            if profile.speculative_mode == speculative_mode
        )
        if not speculative_mode_profiles:
            return PostTransformQualificationDecision(
                PostTransformQualificationReason.SPECULATIVE_MODE_NOT_REGISTERED
            )

        protocol_profiles = tuple(
            profile
            for profile in speculative_mode_profiles
            if profile.protocol_version == protocol_version
        )
        if not protocol_profiles:
            return PostTransformQualificationDecision(
                PostTransformQualificationReason.PROTOCOL_NOT_REGISTERED
            )

        scope_profiles = tuple(
            profile for profile in protocol_profiles if profile.transfer_scope is transfer_scope
        )
        if not scope_profiles:
            return PostTransformQualificationDecision(
                PostTransformQualificationReason.TRANSFER_SCOPE_NOT_REGISTERED
            )

        runtime_matches = []
        runtime_mismatches = []
        for profile in scope_profiles:
            unsupported_runtime_dimensions = profile.runtime_constraints.unsupported_dimensions(
                runtime_config
            )
            if unsupported_runtime_dimensions:
                runtime_mismatches.append((profile, unsupported_runtime_dimensions))
            else:
                runtime_matches.append(profile)
        if not runtime_matches:
            profile, unsupported_runtime_dimensions = min(
                runtime_mismatches, key=lambda item: len(item[1])
            )
            return PostTransformQualificationDecision(
                PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED,
                profile=profile,
                unsupported_runtime_dimensions=unsupported_runtime_dimensions,
            )

        profile = runtime_matches[0]
        unsupported_features = frozenset(enabled_features) - profile.supported_features
        if unsupported_features:
            return PostTransformQualificationDecision(
                PostTransformQualificationReason.FEATURE_NOT_SUPPORTED,
                profile=profile,
                unsupported_features=unsupported_features,
            )

        return PostTransformQualificationDecision(
            PostTransformQualificationReason.QUALIFIED, profile=profile
        )
