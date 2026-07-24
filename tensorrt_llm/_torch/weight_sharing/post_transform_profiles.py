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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn


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
        object.__setattr__(self, "supported_features", frozenset(self.supported_features))


@dataclass(frozen=True)
class PostTransformQualificationDecision:
    """Result of looking up a requested post-transform receiver profile."""

    reason: PostTransformQualificationReason
    profile: PostTransformProfile | None = None
    unsupported_features: frozenset[PostTransformFeature] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "unsupported_features", frozenset(self.unsupported_features))
        if self.reason is PostTransformQualificationReason.QUALIFIED:
            if self.profile is None:
                raise ValueError("A qualified decision must include its profile")
            if self.unsupported_features:
                raise ValueError("A qualified decision cannot include unsupported features")
        elif self.unsupported_features and (
            self.reason is not PostTransformQualificationReason.FEATURE_NOT_SUPPORTED
        ):
            raise ValueError("Unsupported features require a feature_not_supported decision")

    @property
    def qualified(self) -> bool:
        """Whether the request exactly matched a qualified profile."""

        return (
            self.reason is PostTransformQualificationReason.QUALIFIED and self.profile is not None
        )


@dataclass(frozen=True)
class PostTransformProfileRegistry:
    """Immutable collection of audited post-transform profiles."""

    profiles: tuple[PostTransformProfile, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "profiles", tuple(self.profiles))

        profile_ids: set[str] = set()
        profile_keys: set[
            tuple[type[nn.Module], str, str, str | None, int, PostTransformTransferScope]
        ] = set()
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
            if key in profile_keys:
                raise ValueError(
                    "Duplicate post-transform profile for "
                    f"{profile.root_model_class.__name__}/{profile.architecture}/"
                    f"{profile.model_type}/{profile.speculative_mode or 'target-only'}/"
                    f"v{profile.protocol_version}/{profile.transfer_scope.value}"
                )
            profile_keys.add(key)

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

        profile = scope_profiles[0]
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
