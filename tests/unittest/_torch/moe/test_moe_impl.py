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
"""MoE implementation-identity unit tests.

Separate from ``test_moe_backend.py`` because the two have opposite lifetimes.
A backend literal is the pre-identity way to ask for an implementation: it names
a class, degrades to another when the request cannot be served, and reports
whatever ``_legacy_backend_name`` derives from ``__name__`` -- machinery the MoE
developer guide records as known debt, due for removal once every
implementation owns an id. What lives here is the mechanism that replaces it:
one ``MoEImplId`` per class, the registry as the only map from id to class, and
a pinned id that fails instead of degrading.

So these tests are expected to grow one section per implementation as the
migration proceeds, while the backend-literal tests next door shrink and
eventually go. Tests that merely *use* a DeepGEMM class -- weight loading,
staged hooks, numerical parity across backends -- stay with the backend file;
only identity, registration, and resolution-by-id belong here.
"""

from unittest.mock import MagicMock

import pytest
import torch
from _torch.moe.moe_test_utils import MoeBackendType

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.fused_moe.create_moe import create_moe_backend
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.moe.fused_moe.fused_moe_deepgemm import (
    DeepgemmCudaCppFp8BlockScalesImpl,
    DeepGemmFusedMoE,
)
from tensorrt_llm._torch.moe.fused_moe.impl_contract import (
    MoEDeployment,
    MoEEnvironment,
    MoEProblem,
    MoERejectReason,
    canonical_quant,
)
from tensorrt_llm._torch.moe.fused_moe.impl_environment import MoEDep, override_moe_environment
from tensorrt_llm._torch.moe.fused_moe.impl_identity import (
    MOE_IMPL_REGISTRY,
    MoEImplId,
    MoEImplQuery,
)
from tensorrt_llm._torch.moe.fused_moe.interface import MoESchedulerKind
from tensorrt_llm._torch.moe.fused_moe.mega_moe import (
    DeepgemmCudaCppW4a8Mxfp4Mxfp8Impl,
    MegaMoEDeepGemm,
)
from tensorrt_llm._torch.moe.fused_moe.moe_resolution import impl_class_for, resolve_moe_impl
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

# =====================================================================
# Canonical implementation identity
# =====================================================================
# DeepGemm is the first leaf to carry a MoEImplId, so these tests pin down the
# mechanism the remaining backends reuse as they migrate: one identity per
# class, the registry as the only map from identity to class, and a pinned
# identity that fails hard exactly where a backend literal would have degraded.

_DEEPGEMM_IMPL_ID = "deepgemm.cuda_cpp.grouped_gemm.fp8_block_scales"


def _deepgemm_model_config(quant_algo=QuantAlgo.FP8_BLOCK_SCALES):
    cfg = ModelConfig()
    cfg.moe_backend = "DEEPGEMM"
    cfg.quant_config = QuantConfig(quant_algo=quant_algo) if quant_algo else None
    return cfg


def _deepgemm_environment(sm: int = 100) -> MoEEnvironment:
    """DeepGemm's own SM window, so quantization stays the only variable."""
    return MoEEnvironment(sm=sm)


def test_deepgemm_identity_round_trips_through_registry():
    identity = DeepgemmCudaCppFp8BlockScalesImpl.descriptor.identity
    assert identity.canonical() == _DEEPGEMM_IMPL_ID
    assert MoEImplId.parse(_DEEPGEMM_IMPL_ID) == identity
    assert MOE_IMPL_REGISTRY.lookup(identity) is DeepgemmCudaCppFp8BlockScalesImpl


def test_pinned_deepgemm_identity_resolves_to_the_leaf():
    with override_moe_environment(_deepgemm_environment()):
        report = resolve_moe_impl(_deepgemm_model_config(), impl_id=_DEEPGEMM_IMPL_ID)
    assert impl_class_for(report) is DeepgemmCudaCppFp8BlockScalesImpl
    assert report.selected_by == "pinned"
    assert report.requested == _DEEPGEMM_IMPL_ID
    assert not report.degraded


def test_pinned_identity_fails_hard_where_the_backend_literal_degrades():
    """The two request tracks are meant to differ exactly here."""
    config = _deepgemm_model_config(QuantAlgo.NVFP4)
    with override_moe_environment(_deepgemm_environment()):
        by_literal = resolve_moe_impl(config)
        by_identity = resolve_moe_impl(config, impl_id=_DEEPGEMM_IMPL_ID)

    assert impl_class_for(by_literal) is CutlassFusedMoE
    assert by_literal.degraded

    assert by_identity.winner is None
    assert by_identity.selected_by == "failed"
    assert [rejection.reason for rejection in by_identity.rejected] == [
        MoERejectReason.QUANT_UNSUPPORTED
    ]
    with pytest.raises(ValueError, match="no MoE implementation can serve"):
        impl_class_for(by_identity)


def test_unknown_identity_token_raises_before_any_candidate_is_asked():
    with override_moe_environment(_deepgemm_environment()):
        with pytest.raises(ValueError, match="unknown MoE impl token"):
            resolve_moe_impl(_deepgemm_model_config(), impl_id="nosuchprovider")


def test_identity_matching_nothing_registered_raises():
    """Built as a query rather than parsed, to get past the token vocabulary."""
    with override_moe_environment(_deepgemm_environment()):
        with pytest.raises(ValueError, match="matches no registered implementation"):
            resolve_moe_impl(_deepgemm_model_config(), impl_id=MoEImplQuery(quant="nvfp4"))


# =====================================================================
# MegaMoE DeepGEMM implementation identity
# =====================================================================
# The same mechanism as the section above, on the second leaf to declare an
# identity. One thing is new: ``deepgemm`` as a bare provider token now matches
# two leaves, so the disambiguation is exercised rather than assumed.
#
# Problem and deployment are passed explicitly instead of being derived from a
# ModelConfig: MegaMoE gates on SM, dtype, both TMA alignments, topology, and
# an optional DeepGEMM build, and stating them here keeps quantization the only
# variable across these tests.

_MEGAMOE_DEEPGEMM_IMPL_ID = "deepgemm.cuda_cpp.mega_moe.w4a8_mxfp4_mxfp8"


def _megamoe_problem(quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8) -> MoEProblem:
    """A problem MegaMoE's non-quant gates all accept."""
    return MoEProblem(
        quant=canonical_quant(quant_algo),
        dtype_act=torch.bfloat16,
        # Both widths are % 512 == 0, which is what the packed-UE8M0 SF rows
        # need to stay TMA-aligned.
        hidden_size=1024,
        intermediate_size=1024,
        num_experts=8,
        top_k=2,
        swiglu_gptoss_style=False,
    )


def _megamoe_deployment() -> MoEDeployment:
    """EP-only single rank on SM100, with the DG mega kernel present."""
    return MoEDeployment(
        ep_size=1,
        tp_size=1,
        parallel_size=1,
        use_dp=False,
        num_slots=8,
        env=MoEEnvironment(sm=100, available_deps=(MoEDep.DEEPGEMM_MEGAMOE.value,)),
    )


def _megamoe_model_config() -> ModelConfig:
    cfg = ModelConfig()
    cfg.moe_backend = MoeBackendType.MEGAMOE_DEEPGEMM.value
    return cfg


def test_megamoe_deepgemm_identity_round_trips_through_registry():
    identity = DeepgemmCudaCppW4a8Mxfp4Mxfp8Impl.descriptor.identity
    assert identity.canonical() == _MEGAMOE_DEEPGEMM_IMPL_ID
    assert MoEImplId.parse(_MEGAMOE_DEEPGEMM_IMPL_ID) == identity
    assert MOE_IMPL_REGISTRY.lookup(identity) is DeepgemmCudaCppW4a8Mxfp4Mxfp8Impl


def test_pinned_megamoe_deepgemm_identity_resolves_to_the_leaf():
    report = resolve_moe_impl(
        _megamoe_model_config(),
        problem=_megamoe_problem(),
        deployment=_megamoe_deployment(),
        impl_id=_MEGAMOE_DEEPGEMM_IMPL_ID,
    )
    assert impl_class_for(report) is DeepgemmCudaCppW4a8Mxfp4Mxfp8Impl
    assert report.selected_by == "pinned"
    assert report.requested == _MEGAMOE_DEEPGEMM_IMPL_ID
    assert not report.degraded


def test_pinned_kernel_token_alone_reaches_the_megamoe_leaf():
    """``mega_moe`` belongs to one leaf, so the kernel segment is enough."""
    report = resolve_moe_impl(
        _megamoe_model_config(),
        problem=_megamoe_problem(),
        deployment=_megamoe_deployment(),
        impl_id="mega_moe",
    )
    assert impl_class_for(report) is DeepgemmCudaCppW4a8Mxfp4Mxfp8Impl
    assert report.requested == "*.*.mega_moe.*"


def test_bare_deepgemm_provider_is_disambiguated_by_the_quant_gates():
    """One provider token, two leaves, and the gates are what choose.

    The mega leaf precedes the grouped-GEMM leaf in ``IMPL_PRIORITY``, so
    priority order alone would send both requests below to the mega one. Their
    ``quant`` segments are disjoint, which is why each lands on its own.
    """
    deployment = _megamoe_deployment()
    mega = resolve_moe_impl(
        _megamoe_model_config(),
        problem=_megamoe_problem(),
        deployment=deployment,
        impl_id="deepgemm",
    )
    grouped = resolve_moe_impl(
        _megamoe_model_config(),
        problem=_megamoe_problem(QuantAlgo.FP8_BLOCK_SCALES),
        deployment=deployment,
        impl_id="deepgemm",
    )

    assert impl_class_for(mega) is DeepgemmCudaCppW4a8Mxfp4Mxfp8Impl
    assert impl_class_for(grouped) is DeepgemmCudaCppFp8BlockScalesImpl
    assert mega.requested == grouped.requested == "deepgemm.*.*.*"


def test_pinned_megamoe_identity_fails_hard_where_the_backend_literal_degrades():
    """The two request tracks are meant to differ exactly here."""
    config = _megamoe_model_config()
    problem = _megamoe_problem(QuantAlgo.NVFP4)
    deployment = _megamoe_deployment()

    by_literal = resolve_moe_impl(config, problem=problem, deployment=deployment)
    by_identity = resolve_moe_impl(
        config,
        problem=problem,
        deployment=deployment,
        impl_id=_MEGAMOE_DEEPGEMM_IMPL_ID,
    )

    assert impl_class_for(by_literal) is CutlassFusedMoE
    assert by_literal.degraded

    assert by_identity.winner is None
    assert by_identity.selected_by == "failed"
    assert [rejection.reason for rejection in by_identity.rejected] == [
        MoERejectReason.QUANT_UNSUPPORTED
    ]
    with pytest.raises(ValueError, match="no MoE implementation can serve"):
        impl_class_for(by_identity)


def test_registering_megamoe_leaves_the_backend_literal_path_unchanged():
    """Registration must not move which kernel MEGAMOE_DEEPGEMM picks.

    As above, the literal now lands on the leaf -- the constructible half --
    running the same kernel it always did.
    """
    report = resolve_moe_impl(
        _megamoe_model_config(),
        problem=_megamoe_problem(),
        deployment=_megamoe_deployment(),
    )
    assert impl_class_for(report) is DeepgemmCudaCppW4a8Mxfp4Mxfp8Impl
    assert report.selected_by == "pinned"
    assert report.requested == MoeBackendType.MEGAMOE_DEEPGEMM.value


# =====================================================================
# One class per identity
# =====================================================================
# Each DeepGEMM identity is declared on the class that executes it: one class
# owns the descriptor and all four abstract methods. No abstract parent sits
# above it, because there is nothing to share yet -- a parent carrying no
# identity, implementing nothing, and having exactly one subclass would only
# add a hop. The pre-identity names survive as module-level aliases, so both
# request tracks and every legacy call site land on that one class and a run
# reports one name. These are the checks that make that true rather than
# intended.

# The legacy alias, the class that carries the identity, and the scheduler kind
# that class publishes.
_DEEPGEMM_IMPLS = (
    pytest.param(
        DeepGemmFusedMoE,
        DeepgemmCudaCppFp8BlockScalesImpl,
        MoESchedulerKind.EXTERNAL_COMM,
        id="grouped_gemm",
    ),
    pytest.param(
        MegaMoEDeepGemm,
        DeepgemmCudaCppW4a8Mxfp4Mxfp8Impl,
        MoESchedulerKind.FUSED_COMM,
        id="mega_moe",
    ),
)

_IMPLS_ONLY = tuple(pytest.param(case.values[1], id=case.id) for case in _DEEPGEMM_IMPLS)

_ABSTRACT_METHODS = ("can_implement", "_get_quant_method", "quantize_input", "run_moe")


@pytest.mark.parametrize("legacy, impl, scheduler_kind", _DEEPGEMM_IMPLS)
def test_the_identity_and_the_implementation_sit_on_one_class(legacy, impl, scheduler_kind):
    """What is published and what executes are declared together, so neither drifts."""
    assert "descriptor" in vars(impl)
    for name in _ABSTRACT_METHODS:
        assert name in vars(impl)
    # Nothing left abstract: the class the tables name is the class that runs.
    assert not impl.__abstractmethods__

    # An alias, not a base class. Reintroducing a parent under the old name is
    # the tempting way to keep the legacy call sites working; it also splits the
    # kernel into a name that resolves and a name that does not.
    assert legacy is impl

    # Taken off the class's own descriptor rather than restated beside it.
    assert impl.scheduler_kind is impl.descriptor.scheduler_kind
    assert impl.capabilities is impl.descriptor.capabilities
    assert impl.input_requirement is impl.descriptor.input_requirement
    # Load-bearing for the MegaMoE one: publishing EXTERNAL_COMM would have
    # ConfigurableMoE layer host-side comm on top of its SymmBuffer.
    assert impl.descriptor.scheduler_kind is scheduler_kind


def test_both_entrances_reach_the_same_class_under_one_name():
    """The coarse literal and the pin agree on the class and on the report.

    ``_legacy_backend_name`` still answers with ``__name__``, so this also
    pins down that the reported name is the identity-derived one on both
    tracks -- there is no second name a caller could see.
    """
    with override_moe_environment(_deepgemm_environment()):
        by_literal = resolve_moe_impl(_deepgemm_model_config())
        by_pin = resolve_moe_impl(_deepgemm_model_config(), impl_id=_DEEPGEMM_IMPL_ID)

    assert impl_class_for(by_literal) is DeepgemmCudaCppFp8BlockScalesImpl
    assert impl_class_for(by_pin) is impl_class_for(by_literal)
    assert by_literal.winner == by_pin.winner == "DeepgemmCudaCppFp8BlockScalesImpl"


@pytest.mark.parametrize("impl", _IMPLS_ONLY)
def test_create_moe_backend_dispatches_the_impl(monkeypatch, impl):
    """Resolution stops at the class; this is where that class gets built.

    The dispatch chain in ``create_moe.py`` matches by ``issubclass``, and a
    class that falls past every branch reaches the ``Unsupported moe backend``
    raise at the tail. Resolution tests cannot catch that.
    """
    constructed = {}

    def record_only(self, **kwargs):
        constructed["cls"] = type(self)
        torch.nn.Module.__init__(self)

    monkeypatch.setattr(impl, "__init__", record_only)

    backend = create_moe_backend(
        moe_cls=impl,
        routing_method=MagicMock(),
        num_experts=8,
        hidden_size=512,
        intermediate_size=512,
    )

    assert constructed["cls"] is impl
    assert isinstance(backend, impl)


# =====================================================================
# Identity query parsing
# =====================================================================
# Two leaves are registered by the time these run, so the vocabulary spans two
# kernel names and two quants -- enough for the parse to have something to get
# wrong. A query locates each token's field by value, never by position, which
# is what lets a user type the one token they remember instead of
# reconstructing a four-field ID. Registration keeps the four fields' value
# sets disjoint precisely so that lookup stays unambiguous.


def test_identity_segments_may_be_written_in_any_order():
    """A reversed ID is the same request, and still renders canonically."""
    reversed_text = ".".join(reversed(_MEGAMOE_DEEPGEMM_IMPL_ID.split(".")))
    assert reversed_text != _MEGAMOE_DEEPGEMM_IMPL_ID

    canonical = MOE_IMPL_REGISTRY.parse_query(_MEGAMOE_DEEPGEMM_IMPL_ID)
    assert MOE_IMPL_REGISTRY.parse_query(reversed_text) == canonical
    # Rendering is canonical however it was typed, so a resolution report
    # reads the same either way.
    assert canonical.describe() == _MEGAMOE_DEEPGEMM_IMPL_ID


def test_two_values_for_one_field_is_rejected():
    """Free order does not mean a field may be pinned twice.

    With position no longer constraining anything, this check is the only
    thing standing between a contradictory request and a silent win for
    whichever token happened to be read last.
    """
    with pytest.raises(ValueError, match="sets field 'kernel_name' twice"):
        MOE_IMPL_REGISTRY.parse_query("mega_moe.grouped_gemm")


def test_more_segments_than_fields_is_rejected():
    """Wildcards claim no field, so only the segment count bounds them."""
    with pytest.raises(ValueError, match="more than the 4 fields"):
        MOE_IMPL_REGISTRY.parse_query("deepgemm.*.*.*.*")


def test_a_mistyped_token_is_answered_with_near_misses():
    """The vocabulary grows one entry per leaf, so listing all of it would not scale."""
    with pytest.raises(ValueError, match=r"Closest known values: \['mega_moe'\]"):
        MOE_IMPL_REGISTRY.parse_query("mega_moo")
