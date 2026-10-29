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

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.weight_sharing import (
    LLAMA_POST_TRANSFORM_LAYOUT_ABI_V1,
    PostTransformConfigIdentity,
    PostTransformFeature,
    PostTransformProfile,
    PostTransformProfileRegistry,
    PostTransformQualificationReason,
    PostTransformRuntimeConfig,
    PostTransformRuntimeConstraints,
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
    transform_abi_id: str = LLAMA_POST_TRANSFORM_LAYOUT_ABI_V1,
    transfer_scope: PostTransformTransferScope = PostTransformTransferScope.TARGET_MODEL,
    supported_features: frozenset[PostTransformFeature] = frozenset(),
    runtime_constraints: PostTransformRuntimeConstraints | None = None,
) -> PostTransformProfile:
    return PostTransformProfile(
        profile_id=profile_id,
        root_model_class=root_model_class,
        architecture=architecture,
        model_type=model_type,
        speculative_mode=speculative_mode,
        protocol_version=protocol_version,
        transform_abi_id=transform_abi_id,
        transfer_scope=transfer_scope,
        supported_features=supported_features,
        runtime_constraints=runtime_constraints or PostTransformRuntimeConstraints(),
    )


def _runtime_config(**overrides: object) -> PostTransformRuntimeConfig:
    values = {
        "dtype": "bfloat16",
        "quant_algorithm": "none",
        "kv_cache_quant_algorithm": "none",
        "layerwise_quantization": False,
        "force_dynamic_quantization": False,
        "lora_enabled": False,
        "sparse_attention_enabled": False,
        "attention_backend": "TRTLLM",
        "moe_backend": "CUTLASS",
        "tp_size": 1,
        "pp_size": 1,
        "cp_size": 1,
        "moe_tp_size": 1,
        "moe_ep_size": 1,
        "attention_tp_size": 1,
        "attention_cp_size": 1,
        "attention_dp": False,
        "multi_node": False,
        "tied_word_embeddings": False,
        "rope_type": "default",
        "rope_fusion": True,
        # Constrained rows in this file pin full attention, so the base config
        # realizes it; the loader tests default to `None` for real tiny models.
        "sliding_window": "none",
    }
    values.update(overrides)
    return PostTransformRuntimeConfig(**values)


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
    assert decision.transform_abi_id == LLAMA_POST_TRANSFORM_LAYOUT_ABI_V1
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


def test_runtime_constraints_qualify_only_declared_rows() -> None:
    constraints = PostTransformRuntimeConstraints(
        dtypes=frozenset({"bfloat16"}),
        tp_sizes=frozenset({1, 2}),
        rope_types=frozenset({"default"}),
    )
    profile = _profile(runtime_constraints=constraints)
    registry = PostTransformProfileRegistry((profile,))

    decision = registry.qualify(
        root_model_class=_Model,
        architecture="ModelForCausalLM",
        model_type="model",
        speculative_mode=None,
        protocol_version=1,
        transfer_scope=PostTransformTransferScope.TARGET_MODEL,
        runtime_config=_runtime_config(tp_size=2, moe_tp_size=2, attention_tp_size=2),
    )

    assert decision.qualified
    assert decision.profile is profile


@pytest.mark.parametrize(
    "overrides, expected_dimension",
    [
        pytest.param({"dtype": "float16"}, "dtype", id="dtype"),
        pytest.param({"tp_size": 4}, "tp_size", id="tp-size"),
        pytest.param({"rope_type": "yarn"}, "rope_type", id="rope-type"),
        pytest.param({"sliding_window": "uniform"}, "sliding_window", id="sliding-window"),
    ],
)
def test_runtime_constraints_report_unsupported_dimension(
    overrides: dict[str, object],
    expected_dimension: str,
) -> None:
    profile = _profile(
        runtime_constraints=PostTransformRuntimeConstraints(
            dtypes=frozenset({"bfloat16"}),
            tp_sizes=frozenset({1, 2}),
            rope_types=frozenset({"default"}),
            sliding_windows=frozenset({"none"}),
        )
    )
    registry = PostTransformProfileRegistry((profile,))

    decision = registry.qualify(
        root_model_class=_Model,
        architecture="ModelForCausalLM",
        model_type="model",
        speculative_mode=None,
        protocol_version=1,
        transfer_scope=PostTransformTransferScope.TARGET_MODEL,
        runtime_config=replace(_runtime_config(), **overrides),
    )

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED
    assert decision.profile is profile
    assert decision.unsupported_runtime_dimensions == frozenset({expected_dimension})


def test_constrained_profile_rejects_missing_runtime_config() -> None:
    profile = _profile(
        runtime_constraints=PostTransformRuntimeConstraints(
            dtypes=frozenset({"bfloat16"}),
            tp_sizes=frozenset({1, 2}),
        )
    )
    registry = PostTransformProfileRegistry((profile,))

    decision = registry.qualify(
        root_model_class=_Model,
        architecture="ModelForCausalLM",
        model_type="model",
        speculative_mode=None,
        protocol_version=1,
        transfer_scope=PostTransformTransferScope.TARGET_MODEL,
    )

    assert not decision.qualified
    assert decision.reason is PostTransformQualificationReason.RUNTIME_CONFIG_NOT_SUPPORTED
    assert decision.unsupported_runtime_dimensions == frozenset({"dtype", "tp_size"})


def test_registry_selects_disjoint_runtime_profile() -> None:
    bf16_profile = _profile(
        runtime_constraints=PostTransformRuntimeConstraints(dtypes=frozenset({"bfloat16"}))
    )
    fp16_profile = _profile(
        profile_id="model-fp16-target-v1",
        runtime_constraints=PostTransformRuntimeConstraints(dtypes=frozenset({"float16"})),
    )
    registry = PostTransformProfileRegistry((bf16_profile, fp16_profile))

    decision = registry.qualify(
        root_model_class=_Model,
        architecture="ModelForCausalLM",
        model_type="model",
        speculative_mode=None,
        protocol_version=1,
        transfer_scope=PostTransformTransferScope.TARGET_MODEL,
        runtime_config=_runtime_config(dtype="float16"),
    )

    assert decision.qualified
    assert decision.profile is fp16_profile


def test_registry_rejects_overlapping_runtime_profiles() -> None:
    with pytest.raises(
        ValueError,
        match=r"Duplicate post-transform profile for .*runtime constraints overlap",
    ):
        PostTransformProfileRegistry(
            (
                _profile(
                    runtime_constraints=PostTransformRuntimeConstraints(
                        dtypes=frozenset({"bfloat16"}),
                        tp_sizes=frozenset({1, 2}),
                    )
                ),
                _profile(
                    profile_id="overlapping-profile",
                    runtime_constraints=PostTransformRuntimeConstraints(
                        dtypes=frozenset({"bfloat16"}),
                        tp_sizes=frozenset({2, 4}),
                    ),
                ),
            )
        )


def test_runtime_config_is_captured_from_final_model_config() -> None:
    model_config = SimpleNamespace(
        torch_dtype=torch.bfloat16,
        pretrained_config=SimpleNamespace(
            torch_dtype=torch.bfloat16,
            tie_word_embeddings=False,
            rope_scaling={"rope_type": "default"},
            disable_fuse_rope=False,
        ),
        quant_config=SimpleNamespace(
            quant_algo=None,
            kv_cache_quant_algo=None,
        ),
        quant_config_dict=None,
        force_dynamic_quantization=False,
        lora_config=None,
        sparse_attention_config=None,
        attn_backend="TRTLLM",
        moe_backend="CUTLASS",
        mapping=SimpleNamespace(
            world_size=2,
            gpus_per_node=8,
            is_multi_node=lambda: False,
            tp_size=2,
            pp_size=1,
            cp_size=1,
            moe_tp_size=2,
            moe_ep_size=1,
            attn_tp_size=2,
            attn_cp_size=1,
            enable_attention_dp=False,
        ),
    )

    assert PostTransformRuntimeConfig.from_model_config(model_config) == _runtime_config(
        tp_size=2,
        moe_tp_size=2,
        attention_tp_size=2,
        rope_fusion=None,
        sliding_window=None,
    )


def test_runtime_config_uses_resolved_dtype_when_checkpoint_dtype_is_missing() -> None:
    model_config = ModelConfig(
        pretrained_config=SimpleNamespace(torch_dtype=None),
    )

    runtime_config = PostTransformRuntimeConfig.from_model_config(model_config)

    assert model_config.pretrained_config.torch_dtype is None
    assert model_config.torch_dtype == torch.bfloat16
    assert runtime_config.dtype == "bfloat16"


def test_runtime_config_distinguishes_missing_from_unquantized() -> None:
    unquantized = PostTransformRuntimeConfig.from_model_config(
        SimpleNamespace(quant_config=SimpleNamespace(quant_algo=None, kv_cache_quant_algo=None))
    )
    missing = PostTransformRuntimeConfig.from_model_config(
        SimpleNamespace(quant_config=SimpleNamespace())
    )

    assert unquantized.quant_algorithm == "none"
    assert unquantized.kv_cache_quant_algorithm == "none"
    assert missing.quant_algorithm is None
    assert missing.kv_cache_quant_algorithm is None


def test_runtime_config_uses_mapping_multi_node_contract() -> None:
    runtime_config = PostTransformRuntimeConfig.from_model_config(
        SimpleNamespace(
            mapping=SimpleNamespace(
                world_size=2,
                gpus_per_node=8,
                is_multi_node=lambda: True,
            )
        )
    )

    assert runtime_config.multi_node is True


class _WindowedAttention(nn.Module):
    def __init__(self, attention_window_size: object) -> None:
        super().__init__()
        self.attention_window_size = attention_window_size


def _model_with_windows(*windows: object) -> nn.Module:
    model = nn.Module()
    model.layers = nn.ModuleList(_WindowedAttention(window) for window in windows)
    return model


@pytest.mark.parametrize(
    "windows, expected",
    [
        pytest.param((None, None), "none", id="full-attention"),
        pytest.param((4096, 4096), "uniform", id="uniform-window"),
        pytest.param((4096, None), "mixed", id="windowed-and-full"),
        pytest.param((4096, 8192), "mixed", id="different-windows"),
        pytest.param((), None, id="no-window-state"),
        pytest.param(("4096",), None, id="string-window"),
        pytest.param((True,), None, id="bool-window"),
        pytest.param((4096.0,), None, id="float-window"),
        pytest.param((0,), None, id="zero-window"),
        pytest.param((-1,), None, id="negative-window"),
    ],
)
def test_runtime_config_realizes_sliding_window_from_model(
    windows: tuple[object, ...],
    expected: str | None,
) -> None:
    runtime_config = PostTransformRuntimeConfig.from_model_config(
        SimpleNamespace(),
        model=_model_with_windows(*windows),
    )

    assert runtime_config.sliding_window == expected


def test_runtime_config_leaves_sliding_window_unrealized_without_model() -> None:
    runtime_config = PostTransformRuntimeConfig.from_model_config(SimpleNamespace())

    assert runtime_config.sliding_window is None


@pytest.mark.parametrize("sliding_window", ["uniform", "mixed", None])
def test_full_attention_constraint_rejects_other_sliding_windows(
    sliding_window: str | None,
) -> None:
    constraints = PostTransformRuntimeConstraints(sliding_windows=frozenset({"none"}))

    assert constraints.unsupported_dimensions(_runtime_config()) == frozenset()
    assert constraints.unsupported_dimensions(
        _runtime_config(sliding_window=sliding_window)
    ) == frozenset({"sliding_window"})
    assert "sliding_window" in constraints.unsupported_dimensions(None)


@pytest.mark.parametrize(
    "sm_version, expected",
    [
        pytest.param(90, 90, id="sm90"),
        pytest.param(100, 100, id="sm100"),
        pytest.param(-1, -1, id="no-cuda-device"),
        pytest.param(None, None, id="not-supplied"),
        pytest.param(True, None, id="bool"),
        pytest.param("90", None, id="string"),
        pytest.param(9.0, None, id="float"),
    ],
)
def test_runtime_config_records_canonical_sm_version(
    sm_version: object,
    expected: int | None,
) -> None:
    runtime_config = PostTransformRuntimeConfig.from_model_config(
        SimpleNamespace(),
        sm_version=sm_version,
    )

    assert runtime_config.sm_version == expected


def test_runtime_config_leaves_sm_version_unknown_by_default() -> None:
    assert PostTransformRuntimeConfig.from_model_config(SimpleNamespace()).sm_version is None
    assert _runtime_config().sm_version is None


@pytest.mark.parametrize("sm_version", [None, -1, 89, 100, 103, 107, 120, 121])
def test_sm_version_constraint_rejects_other_architectures(sm_version: int | None) -> None:
    constraints = PostTransformRuntimeConstraints(sm_versions=frozenset({90}))

    assert constraints.unsupported_dimensions(_runtime_config(sm_version=90)) == frozenset()
    assert constraints.unsupported_dimensions(_runtime_config(sm_version=sm_version)) == frozenset(
        {"sm_version"}
    )


@pytest.mark.parametrize("sm_version", [None, -1, 90, 100, 120])
def test_unconstrained_sm_version_admits_every_architecture(sm_version: int | None) -> None:
    constraints = PostTransformRuntimeConstraints(dtypes=frozenset({"bfloat16"}))

    assert constraints.unsupported_dimensions(_runtime_config(sm_version=sm_version)) == frozenset()


@pytest.mark.parametrize(
    "quantization_config, expected",
    [
        pytest.param(
            {"quant_method": "fp8", "weight_block_size": [128, 128]}, (128, 128), id="hf-dict"
        ),
        pytest.param(SimpleNamespace(weight_block_size=(128, 128)), (128, 128), id="parsed-config"),
        pytest.param({"quant_method": "fp8", "weight_block_size": [1, 128]}, (1, 128), id="1x128"),
        pytest.param({"quant_method": "fp8", "activation_scheme": "static"}, None, id="undeclared"),
        pytest.param({"weight_block_size": []}, None, id="empty"),
        pytest.param({"weight_block_size": ["128", 128]}, None, id="string-entry"),
        pytest.param({"weight_block_size": [True, 128]}, None, id="bool-entry"),
        pytest.param({"weight_block_size": 128}, None, id="scalar"),
        pytest.param(None, None, id="unquantized"),
    ],
)
def test_runtime_config_records_declared_weight_block_size(
    quantization_config: object,
    expected: tuple[int, ...] | None,
) -> None:
    runtime_config = PostTransformRuntimeConfig.from_model_config(
        SimpleNamespace(pretrained_config=SimpleNamespace(quantization_config=quantization_config))
    )

    assert runtime_config.weight_block_size == expected


def test_runtime_config_leaves_weight_block_size_unknown_by_default() -> None:
    assert PostTransformRuntimeConfig.from_model_config(SimpleNamespace()).weight_block_size is None
    assert _runtime_config().weight_block_size is None


@pytest.mark.parametrize("weight_block_size", [None, (1, 128), (64, 64), (128,), (128, 128, 1)])
def test_weight_block_size_constraint_rejects_other_block_sizes(
    weight_block_size: tuple[int, ...] | None,
) -> None:
    constraints = PostTransformRuntimeConstraints(weight_block_sizes=frozenset({(128, 128)}))

    assert (
        constraints.unsupported_dimensions(_runtime_config(weight_block_size=(128, 128)))
        == frozenset()
    )
    assert constraints.unsupported_dimensions(
        _runtime_config(weight_block_size=weight_block_size)
    ) == frozenset({"weight_block_size"})


@pytest.mark.parametrize("weight_block_size", [None, (1, 128), (128, 128)])
def test_unconstrained_weight_block_size_admits_every_block_size(
    weight_block_size: tuple[int, ...] | None,
) -> None:
    constraints = PostTransformRuntimeConstraints(dtypes=frozenset({"bfloat16"}))

    assert (
        constraints.unsupported_dimensions(_runtime_config(weight_block_size=weight_block_size))
        == frozenset()
    )


def test_rows_pinned_to_different_sm_versions_do_not_overlap() -> None:
    sm90 = PostTransformRuntimeConstraints(sm_versions=frozenset({90}))
    sm100 = PostTransformRuntimeConstraints(sm_versions=frozenset({100, 103}))
    unconstrained = PostTransformRuntimeConstraints()

    assert not sm90.overlaps(sm100)
    assert sm90.overlaps(unconstrained)
    registry = PostTransformProfileRegistry(
        (
            _profile(runtime_constraints=sm90),
            _profile(profile_id="model-sm100-target-v1", runtime_constraints=sm100),
        )
    )
    decision = registry.qualify(
        root_model_class=_Model,
        architecture="ModelForCausalLM",
        model_type="model",
        speculative_mode=None,
        protocol_version=1,
        transfer_scope=PostTransformTransferScope.TARGET_MODEL,
        runtime_config=_runtime_config(sm_version=103),
    )
    assert decision.qualified
    assert decision.profile is not None
    assert decision.profile.profile_id == "model-sm100-target-v1"


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


def test_runtime_constraints_reject_empty_allowed_values() -> None:
    with pytest.raises(ValueError, match="runtime constraint dtypes"):
        PostTransformRuntimeConstraints(dtypes=frozenset())


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
        pytest.param({"transform_abi_id": ""}, "transform_abi_id", id="transform-abi"),
        pytest.param({"transform_abi_id": 1}, "transform_abi_id", id="transform-abi-type"),
    ],
)
def test_profile_rejects_invalid_required_fields(
    kwargs: dict[str, object], expected_message: str
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        _profile(**kwargs)


def test_config_identity_is_captured_before_later_normalization() -> None:
    pretrained_config = SimpleNamespace(
        architectures=["ModelForCausalLM"],
        model_type="model",
    )
    model_config = SimpleNamespace(pretrained_config=pretrained_config)

    identity = PostTransformConfigIdentity.from_model_config(model_config)
    pretrained_config.architectures[0] = "NormalizedForCausalLM"
    pretrained_config.model_type = "normalized"

    assert identity == PostTransformConfigIdentity(
        architecture="ModelForCausalLM",
        model_type="model",
    )


@pytest.mark.parametrize(
    "architectures, model_type, expected",
    [
        pytest.param([], "model", (None, "model"), id="missing-architecture"),
        pytest.param([1], "model", (None, "model"), id="non-string-architecture"),
        pytest.param(["ModelForCausalLM"], 1, ("ModelForCausalLM", None), id="model-type"),
    ],
)
def test_config_identity_fails_closed_for_noncanonical_dimensions(
    architectures: list[object],
    model_type: object,
    expected: tuple[str | None, str | None],
) -> None:
    identity = PostTransformConfigIdentity.from_model_config(
        SimpleNamespace(
            pretrained_config=SimpleNamespace(
                architectures=architectures,
                model_type=model_type,
            )
        )
    )

    assert (identity.architecture, identity.model_type) == expected
