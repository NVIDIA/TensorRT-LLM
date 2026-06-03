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
"""Unit tests for the backend-agnostic SourceIdentity fingerprint.

Mock-based: no real model or weights are touched. Fakes expose only the
attributes that the fingerprint projection reads.
"""

import copy

import pytest
from _source_identity_fakes import (
    FakeMapping,
    FakeModel,
    FakeModelConfig,
    FakePretrainedConfig,
    FakeQuantConfig,
    FakeQuantConfigWithPythonOnlyField,
    identity_from,
)

from tensorrt_llm._torch.weight_sharing import (
    IdentityCheckPolicy,
    SourceIdentity,
    SourceIdentityMismatchError,
    check_source_identity,
)


def test_identical_configs_match():
    a = identity_from(FakeModelConfig())
    b = identity_from(FakeModelConfig())
    result = a.matches(b)
    assert result.matched
    assert result.mismatched_fields == []
    assert bool(result) is True


def test_rank_defaults_from_mapping():
    cfg = FakeModelConfig(mapping=FakeMapping(rank=3, tp_rank=3))
    identity = identity_from(cfg)
    assert identity.rank == 3


def test_backend_mismatch_flags_global():
    a = identity_from(FakeModelConfig(attn_backend="TRTLLM"))
    b = identity_from(FakeModelConfig(attn_backend="FLASHINFER"))
    result = a.matches(b)
    assert not result.matched
    assert "backend_fingerprint" in result.mismatched_fields


def test_model_layout_field_mismatch_flags_global():
    a = identity_from(FakeModelConfig(pretrained_config=FakePretrainedConfig(hidden_size=4096)))
    b = identity_from(FakeModelConfig(pretrained_config=FakePretrainedConfig(hidden_size=8192)))
    result = a.matches(b)
    assert not result.matched
    assert "model_fingerprint" in result.mismatched_fields


def test_non_layout_model_metadata_does_not_affect_match():
    a = identity_from(FakeModelConfig(pretrained_config=FakePretrainedConfig(repository_url="old")))
    b = identity_from(FakeModelConfig(pretrained_config=FakePretrainedConfig(repository_url="new")))
    assert a.matches(b).matched


def test_param_dtype_override_flags_global():
    # Same config, but the constructed module realizes a different compute
    # dtype (a runtime override). A config-only projection would miss this;
    # the realized-layout fingerprint catches it.
    cfg = FakeModelConfig()
    a = SourceIdentity.from_model_config(
        cfg, FakeModel(cfg.pretrained_config, dtype="torch.bfloat16")
    )
    b = SourceIdentity.from_model_config(
        cfg, FakeModel(cfg.pretrained_config, dtype="torch.float16")
    )
    result = a.matches(b)
    assert not result.matched
    assert "model_fingerprint" in result.mismatched_fields


def test_cross_architecture_same_shapes_flags_global():
    # Two models with identical tensor shapes but different architectures must
    # not be considered layout-compatible (guards against shape collisions).
    a = identity_from(
        FakeModelConfig(pretrained_config=FakePretrainedConfig(architectures=("LlamaForCausalLM",)))
    )
    b = identity_from(
        FakeModelConfig(
            pretrained_config=FakePretrainedConfig(architectures=("MistralForCausalLM",))
        )
    )
    result = a.matches(b)
    assert not result.matched
    assert "model_fingerprint" in result.mismatched_fields


def test_no_model_degrades_to_architecture_only():
    # Without a module, the fingerprint still builds (architecture-only) and
    # two identical configs still match.
    a = SourceIdentity.from_model_config(FakeModelConfig(), None)
    b = SourceIdentity.from_model_config(FakeModelConfig(), None)
    assert a.matches(b).matched


def test_quant_mismatch_flags_global():
    a = identity_from(FakeModelConfig(quant_config=FakeQuantConfig(quant_algo="FP8")))
    b = identity_from(FakeModelConfig(quant_config=FakeQuantConfig(quant_algo="NVFP4")))
    result = a.matches(b)
    assert not result.matched
    assert "quant_fingerprint" in result.mismatched_fields


def test_quant_model_dump_uses_python_mode():
    identity = identity_from(FakeModelConfig(quant_config=FakeQuantConfigWithPythonOnlyField()))
    assert identity.quant_fingerprint


def test_parallel_size_mismatch_flags_global():
    a = identity_from(FakeModelConfig(mapping=FakeMapping(tp_size=8)))
    b = identity_from(FakeModelConfig(mapping=FakeMapping(tp_size=4)))
    result = a.matches(b)
    assert not result.matched
    assert "parallel_fingerprint" in result.mismatched_fields


def test_shard_mismatch_only():
    # Same global layout, different per-rank slice.
    a = identity_from(FakeModelConfig(mapping=FakeMapping(rank=0, tp_rank=0)))
    b = identity_from(FakeModelConfig(mapping=FakeMapping(rank=1, tp_rank=1)))
    result = a.matches(b)
    assert not result.matched
    assert result.mismatched_fields == ["shard_fingerprint"]
    # Global parts still agree.
    assert a.global_fingerprint == b.global_fingerprint


def test_compare_global_only_ignores_shard():
    a = identity_from(FakeModelConfig(mapping=FakeMapping(rank=0, tp_rank=0)))
    b = identity_from(FakeModelConfig(mapping=FakeMapping(rank=1, tp_rank=1)))
    assert a.matches(b, compare_shard=False).matched


def test_enforced_sharing_skips_global():
    # Divergent runs (different model + backend) but enforced sharing trusts
    # the source by skipping the global comparison.
    a = identity_from(FakeModelConfig(attn_backend="TRTLLM"))
    b = identity_from(FakeModelConfig(attn_backend="FLASHINFER"))
    assert a.matches(b, compare_global=False, compare_shard=False).matched


def test_serialization_roundtrip():
    a = identity_from(FakeModelConfig())
    restored = SourceIdentity.from_json(a.to_json())
    assert restored == a
    assert a.matches(restored).matched


def test_check_warn_fallback_on_mismatch():
    local = identity_from(FakeModelConfig(attn_backend="TRTLLM"))
    source = identity_from(FakeModelConfig(attn_backend="FLASHINFER"))
    decision = check_source_identity(local, source, IdentityCheckPolicy.WARN_FALLBACK)
    assert decision.should_share is False
    assert not decision.match_result.matched


def test_check_warn_fallback_on_match():
    local = identity_from(FakeModelConfig())
    source = identity_from(FakeModelConfig())
    decision = check_source_identity(local, source, IdentityCheckPolicy.WARN_FALLBACK)
    assert decision.should_share is True


def test_check_strict_raises_on_mismatch():
    local = identity_from(FakeModelConfig(attn_backend="TRTLLM"))
    source = identity_from(FakeModelConfig(attn_backend="FLASHINFER"))
    with pytest.raises(SourceIdentityMismatchError):
        check_source_identity(local, source, IdentityCheckPolicy.STRICT)


def test_check_enforce_shares_despite_mismatch():
    local = identity_from(FakeModelConfig(attn_backend="TRTLLM"))
    source = identity_from(FakeModelConfig(attn_backend="FLASHINFER"))
    decision = check_source_identity(local, source, IdentityCheckPolicy.ENFORCE)
    assert decision.should_share is True


def test_format_version_mismatch_never_matches():
    a = identity_from(FakeModelConfig())
    b = (
        copy.replace(a, format_version=a.format_version + 1)
        if hasattr(copy, "replace")
        else SourceIdentity(
            format_version=a.format_version + 1,
            model_fingerprint=a.model_fingerprint,
            quant_fingerprint=a.quant_fingerprint,
            backend_fingerprint=a.backend_fingerprint,
            parallel_fingerprint=a.parallel_fingerprint,
            rank=a.rank,
            shard_fingerprint=a.shard_fingerprint,
        )
    )
    result = a.matches(b)
    assert not result.matched
    assert "format_version" in result.mismatched_fields
