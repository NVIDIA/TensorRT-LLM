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

import pytest
from torch import nn

from tensorrt_llm._torch.weight_sharing import (
    PostTransformFeature,
    PostTransformProfile,
    PostTransformProfileRegistry,
    PostTransformQualificationReason,
    PostTransformTransferScope,
)


class _Model(nn.Module):
    pass


class _ModelSubclass(_Model):
    pass


def _profile(
    *,
    profile_id: str = "model-target-v1",
    root_model_class: type[nn.Module] = _Model,
    architecture: str = "ModelForCausalLM",
    model_type: str = "model",
    speculative_mode: str | None = None,
    protocol_version: int = 1,
    transfer_scope: PostTransformTransferScope = PostTransformTransferScope.TARGET_MODEL,
    supported_features: frozenset[PostTransformFeature] = frozenset(),
) -> PostTransformProfile:
    return PostTransformProfile(
        profile_id=profile_id,
        root_model_class=root_model_class,
        architecture=architecture,
        model_type=model_type,
        speculative_mode=speculative_mode,
        protocol_version=protocol_version,
        transfer_scope=transfer_scope,
        supported_features=supported_features,
    )


def test_exact_profile_is_qualified() -> None:
    profile = _profile()
    registry = PostTransformProfileRegistry((profile,))

    decision = registry.qualify(
        root_model_class=_Model,
        architecture="ModelForCausalLM",
        model_type="model",
        speculative_mode=None,
        protocol_version=1,
        transfer_scope=PostTransformTransferScope.TARGET_MODEL,
    )

    assert decision.qualified
    assert decision.reason is PostTransformQualificationReason.QUALIFIED
    assert decision.profile is profile
    assert decision.unsupported_features == frozenset()


def test_subclass_does_not_inherit_qualification() -> None:
    registry = PostTransformProfileRegistry((_profile(),))

    decision = registry.qualify(
        root_model_class=_ModelSubclass,
        architecture="ModelForCausalLM",
        model_type="model",
        speculative_mode=None,
        protocol_version=1,
        transfer_scope=PostTransformTransferScope.TARGET_MODEL,
    )

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.ROOT_MODEL_CLASS_NOT_REGISTERED
    assert decision.profile is None


@pytest.mark.parametrize(
    "overrides, expected_reason",
    [
        pytest.param(
            {"architecture": "OtherForCausalLM"},
            PostTransformQualificationReason.ARCHITECTURE_NOT_REGISTERED,
            id="architecture",
        ),
        pytest.param(
            {"model_type": "other"},
            PostTransformQualificationReason.MODEL_TYPE_NOT_REGISTERED,
            id="model-type",
        ),
        pytest.param(
            {"speculative_mode": "mtp"},
            PostTransformQualificationReason.SPECULATIVE_MODE_NOT_REGISTERED,
            id="speculative-mode",
        ),
        pytest.param(
            {"protocol_version": 2},
            PostTransformQualificationReason.PROTOCOL_NOT_REGISTERED,
            id="protocol",
        ),
        pytest.param(
            {"transfer_scope": PostTransformTransferScope.COMPLETE_MODEL},
            PostTransformQualificationReason.TRANSFER_SCOPE_NOT_REGISTERED,
            id="transfer-scope",
        ),
    ],
)
def test_profile_dimensions_must_match_exactly(
    overrides: dict[str, object],
    expected_reason: PostTransformQualificationReason,
) -> None:
    registry = PostTransformProfileRegistry((_profile(),))
    request = {
        "root_model_class": _Model,
        "architecture": "ModelForCausalLM",
        "model_type": "model",
        "speculative_mode": None,
        "protocol_version": 1,
        "transfer_scope": PostTransformTransferScope.TARGET_MODEL,
    }
    request.update(overrides)

    decision = registry.qualify(**request)

    assert not decision.qualified
    assert decision.reason is expected_reason


def test_registry_selects_exact_speculative_mode_profile() -> None:
    target_profile = _profile()
    mtp_profile = _profile(
        profile_id="model-mtp-v1",
        speculative_mode="mtp",
    )
    registry = PostTransformProfileRegistry((target_profile, mtp_profile))

    decision = registry.qualify(
        root_model_class=_Model,
        architecture="ModelForCausalLM",
        model_type="model",
        speculative_mode="mtp",
        protocol_version=1,
        transfer_scope=PostTransformTransferScope.TARGET_MODEL,
    )

    assert decision.qualified
    assert decision.profile is mtp_profile


def test_optional_feature_requires_explicit_profile_support() -> None:
    profile = _profile()
    registry = PostTransformProfileRegistry((profile,))

    decision = registry.qualify(
        root_model_class=_Model,
        architecture="ModelForCausalLM",
        model_type="model",
        speculative_mode=None,
        protocol_version=1,
        transfer_scope=PostTransformTransferScope.TARGET_MODEL,
        enabled_features=frozenset({PostTransformFeature.SEPARATE_DRAFT_MODEL}),
    )

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.FEATURE_NOT_SUPPORTED
    assert decision.profile is profile
    assert decision.unsupported_features == frozenset({PostTransformFeature.SEPARATE_DRAFT_MODEL})


def test_explicitly_supported_optional_feature_is_qualified() -> None:
    profile = _profile(supported_features=frozenset({PostTransformFeature.SEPARATE_DRAFT_MODEL}))
    registry = PostTransformProfileRegistry((profile,))

    decision = registry.qualify(
        root_model_class=_Model,
        architecture="ModelForCausalLM",
        model_type="model",
        speculative_mode=None,
        protocol_version=1,
        transfer_scope=PostTransformTransferScope.TARGET_MODEL,
        enabled_features=frozenset({PostTransformFeature.SEPARATE_DRAFT_MODEL}),
    )

    assert decision.qualified
    assert decision.profile is profile


def test_registry_rejects_duplicate_profile_id() -> None:
    with pytest.raises(ValueError, match="Duplicate post-transform profile_id"):
        PostTransformProfileRegistry(
            (
                _profile(),
                _profile(
                    root_model_class=_ModelSubclass,
                    architecture="ModelSubclassForCausalLM",
                ),
            )
        )


def test_registry_rejects_duplicate_match_key() -> None:
    with pytest.raises(ValueError, match="Duplicate post-transform profile for"):
        PostTransformProfileRegistry((_profile(), _profile(profile_id="other-profile-id")))


@pytest.mark.parametrize(
    "kwargs, expected_message",
    [
        pytest.param({"profile_id": ""}, "profile_id", id="profile-id"),
        pytest.param({"architecture": ""}, "architecture", id="architecture"),
        pytest.param({"model_type": ""}, "model_type", id="model-type"),
        pytest.param(
            {"speculative_mode": ""},
            "speculative_mode",
            id="speculative-mode",
        ),
        pytest.param({"protocol_version": 0}, "protocol_version", id="protocol"),
    ],
)
def test_profile_rejects_invalid_required_fields(
    kwargs: dict[str, object], expected_message: str
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        _profile(**kwargs)
