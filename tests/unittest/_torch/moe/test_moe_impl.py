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

Covers the identity mechanism: the ``MoEImplId`` each implementation declares,
the registry as the only map from an id to its class, resolution by a pinned id
-- which fails rather than degrading to another implementation -- and the
parsing rules for partial id queries.

Tests that merely *use* an implementation, such as weight loading, staged hooks,
or numerical parity across backends, belong in ``test_moe_backend.py``.
"""

import dataclasses
import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from _torch.moe.moe_test_utils import MoeBackendType

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.fused_moe import moe_resolution
from tensorrt_llm._torch.moe.fused_moe.activation import (
    ActivationParamShape,
    MoEActivationSupport,
    SwigluActivation,
)
from tensorrt_llm._torch.moe.fused_moe.create_moe import create_moe_backend
from tensorrt_llm._torch.moe.fused_moe.cutedsl import (
    TrtllmCutedslDenseGemmNvfp4Impl,
    TrtllmCutedslFusedFc12Nvfp4Impl,
    TrtllmCutedslGroupedGemmBf16Impl,
    TrtllmCutedslGroupedGemmNvfp4Impl,
)
from tensorrt_llm._torch.moe.fused_moe.cutedsl.identity import GROUPED_GEMM_FAMILY
from tensorrt_llm._torch.moe.fused_moe.cutlass import (
    CUTLASS_LEAVES,
    CutlassMoELoraMixin,
    DeepgemmCudaHopperFp8BlockScalesImpl,
    TrtllmCutlassFp8Impl,
    TrtllmCutlassMxfp8Impl,
    TrtllmCutlassNvfp4Impl,
    TrtllmCutlassUnquantizedImpl,
    TrtllmCutlassW4a8AwqImpl,
    TrtllmCutlassW4a8Mxfp4Fp8Impl,
    TrtllmCutlassW4a8Mxfp4Mxfp8Impl,
    TrtllmCutlassW4a16Mxfp4Impl,
    TrtllmCutlassW4a16Nvfp4Impl,
    TrtllmCutlassW8a16Impl,
    TrtllmTritonFp8BlockScalesImpl,
    cutlass_leaf,
)
from tensorrt_llm._torch.moe.fused_moe.cutlass.base import (
    CutlassFusedMoEBase,
    FusedMoeOpFlags,
    quantize_input_mxfp8,
    quantize_input_static_e4m3,
)
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl import CuteDslFusedMoE
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl_b12x import (
    CuteDslB12xNvfp4FusedMoE,
    CuteDslB12xW4a16Nvfp4FusedMoE,
)
from tensorrt_llm._torch.moe.fused_moe.fused_moe_deepgemm import (
    DeepgemmCudaFp8BlockScalesImpl,
    DeepGemmFusedMoE,
)
from tensorrt_llm._torch.moe.fused_moe.fused_moe_densegemm import DenseGEMMFusedMoE
from tensorrt_llm._torch.moe.fused_moe.fused_moe_marlin import MarlinFusedMoE
from tensorrt_llm._torch.moe.fused_moe.fused_moe_trtllm_gen import TRTLLMGenFusedMoE
from tensorrt_llm._torch.moe.fused_moe.impl_contract import (
    MoEDeployment,
    MoEEnvironment,
    MoEProblem,
    MoERejection,
    MoERejectReason,
    MoEResolutionReport,
    MoEStaticCapability,
    canonical_quant,
    canonical_routing,
    identity_quant_of,
    normalize_quant,
)
from tensorrt_llm._torch.moe.fused_moe.impl_environment import (
    MoEDep,
    MoEEnvFlag,
    override_moe_environment,
)
from tensorrt_llm._torch.moe.fused_moe.impl_identity import (
    MOE_IMPL_REGISTRY,
    MoEImplDescriptor,
    MoEImplId,
    MoEImplQuery,
    MoEImplRegistry,
)
from tensorrt_llm._torch.moe.fused_moe.interface import MoESchedulerKind
from tensorrt_llm._torch.moe.fused_moe.marlin import MarlinCudaNvfp4Impl, MarlinCudaW4a16Nvfp4Impl
from tensorrt_llm._torch.moe.fused_moe.mega_moe import (
    DeepgemmCudaW4a8Mxfp4Mxfp8Impl,
    MegaMoECuteDsl,
    MegaMoEDeepGemm,
    TrtllmCutedslMegaMoeNvfp4Impl,
)
from tensorrt_llm._torch.moe.fused_moe.moe_resolution import (
    BACKEND_FAMILY,
    IMPL_PRIORITY,
    impl_class_for,
    resolve_moe_impl,
)
from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV4MoeRoutingMethod, RoutingMethodType
from tensorrt_llm._torch.moe.fused_moe.trtllm_gen import (
    FlashinferTrtllmGenBf16Impl,
    FlashinferTrtllmGenNvfp4Impl,
    TrtllmTrtllmGenNvfp4Impl,
    TrtllmTrtllmGenW4a8Nvfp4Fp8Impl,
    trtllm_gen_leaves_in_resolution_order,
)
from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.eligibility import check_flashinfer_provider
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

# Everything here is a declaration; the one test that reaches a constructor
# stubs ``__init__`` out. The marker is also what makes the file reachable: the
# CPU stage collects only files that carry it.
pytestmark = pytest.mark.cpu_only

# The four abstract ``MoEImplBase`` methods.
_ABSTRACT_METHODS = ("can_implement", "_get_quant_method", "quantize_input", "run_moe")

# =====================================================================
# Canonical implementation identity
# =====================================================================
# The DeepGEMM FP8-block-scales identity, the four ways a pinned request is
# refused rather than served by something else (every gate declines, an
# unknown token, an id that matches nothing registered, or one that constrains
# nothing and so would match everything), and the one property every family
# shares: a class its backend literal reaches is pinned by the identity it
# publishes.

_DEEPGEMM_IMPL_ID = "deepgemm.cuda.grouped_gemm.fp8_block_scales"


def _deepgemm_model_config(quant_algo=QuantAlgo.FP8_BLOCK_SCALES):
    cfg = ModelConfig()
    cfg.moe_backend = "DEEPGEMM"
    cfg.quant_config = QuantConfig(quant_algo=quant_algo) if quant_algo else None
    return cfg


def _deepgemm_environment(sm: int = 100) -> MoEEnvironment:
    """DeepGemm's own SM window, so quantization stays the only variable."""
    return MoEEnvironment(sm=sm)


def test_pinned_deepgemm_identity_resolves_to_the_leaf():
    with override_moe_environment(_deepgemm_environment()):
        report = resolve_moe_impl(_deepgemm_model_config(), impl_id=_DEEPGEMM_IMPL_ID)
    assert impl_class_for(report) is DeepgemmCudaFp8BlockScalesImpl
    assert report.selected_by == "pinned"
    assert report.requested == _DEEPGEMM_IMPL_ID
    assert not report.degraded


def test_pinned_deepgemm_rejects_post_silu_clamping():
    activation = SwigluActivation(clamp=5.0, clamp_after_silu=True)
    with override_moe_environment(_deepgemm_environment()):
        report = resolve_moe_impl(
            _deepgemm_model_config(),
            activation=activation,
            impl_id=_DEEPGEMM_IMPL_ID,
        )

    assert report.winner is None
    assert report.rejected[-1].reason is MoERejectReason.ACTIVATION_UNSUPPORTED
    assert "post-SiLU clamping" in report.rejected[-1].detail


def test_pinned_identity_fails_hard_where_the_backend_literal_degrades():
    """The two request tracks are meant to differ exactly here."""
    config = _deepgemm_model_config(QuantAlgo.NVFP4)
    with override_moe_environment(_deepgemm_environment()):
        by_literal = resolve_moe_impl(config)
        by_identity = resolve_moe_impl(config, impl_id=_DEEPGEMM_IMPL_ID)

    assert impl_class_for(by_literal) is TrtllmCutlassNvfp4Impl
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
    """Built as a query rather than parsed, to get past the token vocabulary.

    ``w8a8_sq_per_channel`` because SmoothQuant has no MoE implementation,
    which is what makes the query match nothing while staying well-formed.
    """
    with override_moe_environment(_deepgemm_environment()):
        with pytest.raises(ValueError, match="matches no registered implementation"):
            resolve_moe_impl(
                _deepgemm_model_config(), impl_id=MoEImplQuery(quant="w8a8_sq_per_channel")
            )


@pytest.mark.parametrize("spec", ["*", "*.*.*.*", MoEImplQuery()], ids=["star", "wide", "query"])
def test_identity_constraining_no_field_is_rejected(spec):
    """Every spelling of "no constraint at all" has to be refused.

    A wildcard claims no field, so both texts parse to the same all-open query
    the bare dataclass builds. Serving one would widen a pin into "whichever
    registered impl ranks first" while still discarding the ``moe_backend`` the
    caller set, which is the one outcome neither request track promises.
    """
    with override_moe_environment(_deepgemm_environment()):
        with pytest.raises(ValueError, match="constrains no field"):
            resolve_moe_impl(_deepgemm_model_config(), impl_id=spec)


def test_family_members_are_reachable_by_the_identity_they_publish():
    """A literal reaches its family without the registry, so nothing else
    notices a class a pinned ``impl_id`` cannot reach, or one that publishes
    the identity it inherited from its parent."""
    for family in BACKEND_FAMILY.values():
        for impl in family:
            descriptor = getattr(impl, "descriptor", None)
            if descriptor is not None:
                assert MOE_IMPL_REGISTRY.lookup(descriptor.identity) is impl, impl.__name__


# =====================================================================
# MegaMoE DeepGEMM implementation identity
# =====================================================================
# The MegaMoE identity, where ``deepgemm`` as a bare provider token matches
# several leaves, so the disambiguation is exercised rather than assumed.
#
# Problem and deployment are passed explicitly instead of being derived from a
# ModelConfig: MegaMoE gates on SM, dtype, both TMA alignments, topology, and
# an optional DeepGEMM build, and stating them here keeps quantization the only
# variable across these tests.

_MEGAMOE_DEEPGEMM_IMPL_ID = "deepgemm.cuda.mega_moe.w4a8_mxfp4_mxfp8"


def _megamoe_problem(quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8) -> MoEProblem:
    """A problem MegaMoE's non-quant gates all accept."""
    return MoEProblem(
        quant=canonical_quant(quant_algo),
        dtype_act=torch.bfloat16,
        # Both widths are % 512 == 0, which is what keeps the packed-UE8M0 SF
        # records in DeepGEMM's dispatch buffer TMA-aligned.
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


def test_pinned_kernel_token_reaches_the_megamoe_deepgemm_leaf():
    """``mega_moe`` now spans two leaves, so the quant gate finishes the job.

    The CuteDSL leaf publishes the same kernel segment and admits NVFP4 only,
    so a W4A8_MXFP4_MXFP8 problem still arrives here.
    """
    report = resolve_moe_impl(
        _megamoe_model_config(),
        problem=_megamoe_problem(),
        deployment=_megamoe_deployment(),
        impl_id="mega_moe",
    )
    assert impl_class_for(report) is DeepgemmCudaW4a8Mxfp4Mxfp8Impl
    assert report.selected_by == "pinned"
    assert report.requested == "*.*.mega_moe.*"
    assert not report.degraded


def test_bare_deepgemm_provider_is_disambiguated_by_the_quant_gates():
    """One provider token, several leaves, and the gates are what choose.

    The mega leaf precedes the grouped-GEMM leaf in ``IMPL_PRIORITY``, so
    priority order alone would send both requests below to the mega one. Their
    ``quant`` segments are disjoint, which is why each lands on its own; the
    SM90 ``deepgemm.cuda.hopper_grouped_gemm`` leaf shares the grouped-GEMM
    leaf's quant and declines here on its SM gate.
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

    assert impl_class_for(mega) is DeepgemmCudaW4a8Mxfp4Mxfp8Impl
    assert impl_class_for(grouped) is DeepgemmCudaFp8BlockScalesImpl
    assert mega.requested == grouped.requested == "deepgemm.*.*.*"


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
    assert impl_class_for(report) is DeepgemmCudaW4a8Mxfp4Mxfp8Impl
    assert report.selected_by == "pinned"
    assert report.requested == MoeBackendType.MEGAMOE_DEEPGEMM.value


# =====================================================================
# MegaMoE CuteDSL implementation identity
# =====================================================================
# The second ``mega_moe`` leaf. Its per-architecture kernel variants are
# deliberately not separate identities, so nothing here parametrizes over
# architecture; the SM only has to be one this backend accepts.

_MEGAMOE_CUTEDSL_IMPL_ID = "trtllm.cutedsl.mega_moe.nvfp4"


def _megamoe_cutedsl_problem(quant_algo=QuantAlgo.NVFP4) -> MoEProblem:
    """A problem the CuteDSL leaf's non-quant gates all accept."""
    return MoEProblem(
        quant=canonical_quant(quant_algo),
        dtype_act=torch.bfloat16,
        # hidden % 32 covers the NVFP4 SF leg alignment, intermediate % 16 the
        # Fc1GateUpInterleave the fused FC1 packs against.
        hidden_size=1024,
        intermediate_size=1024,
        num_experts=8,
        top_k=2,
        swiglu_gptoss_style=False,
    )


def _megamoe_cutedsl_deployment() -> MoEDeployment:
    """EP-only single rank on SM100, with both CuteDSL deps present."""
    return MoEDeployment(
        ep_size=1,
        tp_size=1,
        parallel_size=1,
        use_dp=False,
        num_slots=8,
        env=MoEEnvironment(
            sm=100,
            available_deps=(
                MoEDep.MEGAMOE_CUTEDSL_RUNTIME.value,
                MoEDep.MEGAMOE_CUTEDSL_OP.value,
            ),
        ),
    )


def _megamoe_cutedsl_model_config() -> ModelConfig:
    cfg = ModelConfig()
    cfg.moe_backend = MoeBackendType.MEGAMOE_CUTEDSL.value
    return cfg


def test_pinned_megamoe_cutedsl_identity_resolves_to_the_leaf():
    """The full id round-trips through the registry onto the class that runs it."""
    report = resolve_moe_impl(
        _megamoe_cutedsl_model_config(),
        problem=_megamoe_cutedsl_problem(),
        deployment=_megamoe_cutedsl_deployment(),
        impl_id=_MEGAMOE_CUTEDSL_IMPL_ID,
    )
    assert impl_class_for(report) is TrtllmCutedslMegaMoeNvfp4Impl
    assert report.selected_by == "pinned"
    assert report.requested == _MEGAMOE_CUTEDSL_IMPL_ID
    assert not report.degraded


def test_dynamic_eplb_is_a_declared_backend_capability():
    assert TrtllmCutedslMegaMoeNvfp4Impl.capabilities.supports_per_layer_dynamic_eplb
    assert not DeepgemmCudaW4a8Mxfp4Mxfp8Impl.capabilities.supports_per_layer_dynamic_eplb


def test_per_layer_dynamic_eplb_filters_backends_by_declared_capability():
    deployment = dataclasses.replace(
        _megamoe_cutedsl_deployment(),
        eplb_enabled=True,
        eplb_mode="per_layer_dynamic",
    )
    accepted = resolve_moe_impl(
        _megamoe_cutedsl_model_config(),
        problem=_megamoe_cutedsl_problem(),
        deployment=deployment,
        impl_id=_MEGAMOE_CUTEDSL_IMPL_ID,
    )
    assert impl_class_for(accepted) is TrtllmCutedslMegaMoeNvfp4Impl

    rejected = resolve_moe_impl(
        _deepgemm_model_config(QuantAlgo.W4A8_MXFP4_MXFP8),
        problem=_megamoe_cutedsl_problem(QuantAlgo.W4A8_MXFP4_MXFP8),
        deployment=deployment,
        impl_id="deepgemm.cuda.mega_moe.w4a8_mxfp4_mxfp8",
    )
    assert rejected.winner is None
    assert [item.reason for item in rejected.rejected] == [MoERejectReason.EPLB_UNSUPPORTED]


def test_pinned_megamoe_cutedsl_identity_fails_hard_on_another_format():
    """A pin that cannot run must fail, not widen to the sibling that can.

    The leaf admits NVFP4 only, and the sibling ``mega_moe`` leaf does run
    W4A8_MXFP4_MXFP8, so substitution is the tempting failure here.
    """
    report = resolve_moe_impl(
        _megamoe_cutedsl_model_config(),
        problem=_megamoe_cutedsl_problem(QuantAlgo.W4A8_MXFP4_MXFP8),
        deployment=_megamoe_cutedsl_deployment(),
        impl_id=_MEGAMOE_CUTEDSL_IMPL_ID,
    )

    assert report.winner is None
    assert report.selected_by == "failed"
    assert [rejection.reason for rejection in report.rejected] == [
        MoERejectReason.QUANT_UNSUPPORTED
    ]
    with pytest.raises(ValueError, match="no MoE implementation can serve"):
        impl_class_for(report)


def test_the_two_mega_moe_leaves_are_separated_by_their_quant_gates():
    """One kernel segment, two leaves, and the format is what chooses.

    Both match ``*.*.mega_moe.*``, so the pin cannot tell them apart; their
    ``quant`` segments are disjoint, which is why each lands on its own class.

    One deployment carries every MegaMoE dep, so the loser has to decline on
    ``quant`` and cannot be masked by a missing dep -- otherwise widening
    either gate would leave this passing while testing nothing.
    """
    deployment = MoEDeployment(
        ep_size=1,
        tp_size=1,
        parallel_size=1,
        use_dp=False,
        num_slots=8,
        env=MoEEnvironment(
            sm=100,
            available_deps=(
                MoEDep.DEEPGEMM_MEGAMOE.value,
                MoEDep.MEGAMOE_CUTEDSL_RUNTIME.value,
                MoEDep.MEGAMOE_CUTEDSL_OP.value,
            ),
        ),
    )
    cutedsl = resolve_moe_impl(
        _megamoe_cutedsl_model_config(),
        problem=_megamoe_cutedsl_problem(),
        deployment=deployment,
        impl_id="mega_moe",
    )
    deepgemm = resolve_moe_impl(
        _megamoe_model_config(),
        problem=_megamoe_problem(),
        deployment=deployment,
        impl_id="mega_moe",
    )

    assert impl_class_for(cutedsl) is TrtllmCutedslMegaMoeNvfp4Impl
    assert impl_class_for(deepgemm) is DeepgemmCudaW4a8Mxfp4Mxfp8Impl
    assert cutedsl.requested == deepgemm.requested == "*.*.mega_moe.*"

    assert [(r.legacy_backend, r.reason) for r in cutedsl.rejected] == [
        (DeepgemmCudaW4a8Mxfp4Mxfp8Impl.__name__, MoERejectReason.QUANT_UNSUPPORTED)
    ]
    assert [(r.legacy_backend, r.reason) for r in deepgemm.rejected] == [
        (TrtllmCutedslMegaMoeNvfp4Impl.__name__, MoERejectReason.QUANT_UNSUPPORTED)
    ]


def test_registering_megamoe_cutedsl_leaves_the_backend_literal_path_unchanged():
    """Registration must not move which kernel MEGAMOE_CUTEDSL picks."""
    report = resolve_moe_impl(
        _megamoe_cutedsl_model_config(),
        problem=_megamoe_cutedsl_problem(),
        deployment=_megamoe_cutedsl_deployment(),
    )
    assert impl_class_for(report) is TrtllmCutedslMegaMoeNvfp4Impl
    assert report.selected_by == "pinned"
    assert report.requested == MoeBackendType.MEGAMOE_CUTEDSL.value


# =====================================================================
# TRTLLM-Gen implementation identity
# =====================================================================
# Eleven identities over one abstract parent that holds the shared
# ``__init__``, ``create_weights`` and input preparation. Most formats have a
# native and a FlashInfer leaf that differ only in provider, so these tests
# cover which provider a request lands on, and which leaves decline a routing,
# bias, activation constant or epilogue their cubins do not run.

# The full grid, written out rather than generated, so a leaf that silently
# stops registering fails here instead of shrinking a computed expectation.
# Six native leaves; the FlashInfer wheel serves five of the seven formats --
# it has no fp8/fp4-activation runner, and it alone has the unquantized one.
_TRTLLM_GEN_IDS = (
    "trtllm.trtllm_gen.fused_moe.nvfp4",
    "trtllm.trtllm_gen.fused_moe.fp8_block_scales",
    "trtllm.trtllm_gen.fused_moe.w4a16_mxfp4",
    "trtllm.trtllm_gen.fused_moe.w4a8_mxfp4_mxfp8",
    "trtllm.trtllm_gen.fused_moe.w4a8_nvfp4_fp8",
    "trtllm.trtllm_gen.fused_moe.w4a8_mxfp4_fp8",
    "flashinfer.trtllm_gen.fused_moe.nvfp4",
    "flashinfer.trtllm_gen.fused_moe.fp8_block_scales",
    "flashinfer.trtllm_gen.fused_moe.w4a16_mxfp4",
    "flashinfer.trtllm_gen.fused_moe.w4a8_mxfp4_mxfp8",
    "flashinfer.trtllm_gen.fused_moe.none",
)


def _trtllm_gen_environment(
    *, sm: int = 100, flashinfer: bool = False, use_flashinfer_flag: bool = False
) -> MoEEnvironment:
    """An SM100 host, with the FlashInfer wheel and its opt-in flag optional."""
    deps = []
    if flashinfer:
        deps = [MoEDep.FLASHINFER.value, MoEDep.FLASHINFER_BF16_MOE.value]
    return MoEEnvironment(
        sm=sm,
        available_deps=tuple(sorted(deps)),
        env_flags=(
            (MoEEnvFlag.TRTLLM_GEN_USE_FLASHINFER.value, "1" if use_flashinfer_flag else "0"),
        ),
    )


def _single_rank_deployment(env: MoEEnvironment) -> MoEDeployment:
    """EP-only single rank, so the environment stays the only variable."""
    return MoEDeployment(
        ep_size=1,
        tp_size=1,
        parallel_size=1,
        use_dp=False,
        num_slots=8,
        env=env,
    )


def _trtllm_gen_model_config(quant_algo=QuantAlgo.NVFP4):
    cfg = ModelConfig()
    cfg.moe_backend = "TRTLLM"
    cfg.quant_config = QuantConfig(quant_algo=quant_algo) if quant_algo else None
    return cfg


def test_trtllm_gen_registers_exactly_eleven_identities():
    """The grid is the deliverable: eleven addressable ids, no more, no fewer."""
    registered = sorted(
        identity.canonical()
        for identity in MOE_IMPL_REGISTRY.identities()
        if identity.technique == "trtllm_gen"
    )
    assert registered == sorted(_TRTLLM_GEN_IDS)


@pytest.mark.parametrize("impl_id", _TRTLLM_GEN_IDS)
def test_trtllm_gen_identity_round_trips_through_registry(impl_id):
    impl = MOE_IMPL_REGISTRY.lookup(MoEImplId.parse(impl_id))
    assert impl is not None
    # Declared on the leaf itself, not inherited from the parent: an inherited
    # descriptor would give eleven classes one identity.
    assert "descriptor" in vars(impl)
    assert impl.descriptor.identity.canonical() == impl_id
    assert not impl.__abstractmethods__
    assert impl.capabilities is impl.descriptor.capabilities
    assert impl.input_requirement is impl.descriptor.input_requirement


@pytest.mark.parametrize("impl_id", _TRTLLM_GEN_IDS)
def test_trtllm_gen_leaf_provider_agrees_with_its_identity(impl_id):
    """``provider`` is the op-backend key and the id segment, so it is one value.

    Derived from the descriptor by ``__init_subclass__``, so this guards the
    derivation rather than two hand-written declarations: a leaf that lands
    outside the hook -- registered without a descriptor of its own, or built
    by something that bypasses class creation -- would read the wrong
    provider and pick the wrong op backend.
    """
    impl = MOE_IMPL_REGISTRY.lookup(MoEImplId.parse(impl_id))
    assert impl.provider == impl.descriptor.identity.provider
    assert impl.use_flashinfer == (impl.provider == "flashinfer")
    # Only the native runners fill the all-to-all workspace output buffer,
    # which is exactly the provider split. The class stands in for ``self``:
    # the body reads only ``use_flashinfer``, and constructing a leaf needs a
    # GPU.
    assert impl.supports_moe_output_in_alltoall_workspace(impl) == (impl.provider == "trtllm")


@pytest.mark.parametrize("impl_id", _TRTLLM_GEN_IDS)
def test_trtllm_gen_leaf_admits_only_its_own_format(impl_id):
    """Two leaves differing only in provider still admit one format each."""
    impl = MOE_IMPL_REGISTRY.lookup(MoEImplId.parse(impl_id))
    quant = impl.descriptor.identity.quant
    other = "nvfp4" if quant != "nvfp4" else "w4a16_mxfp4"
    verdict = impl.can_implement(
        MoEProblem(quant=other.upper(), dtype_act=torch.bfloat16),
        _single_rank_deployment(_trtllm_gen_environment(flashinfer=True)),
    )
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.QUANT_UNSUPPORTED


def test_trtllm_gen_parent_is_abstract_and_unaddressable():
    """The name survives for ``issubclass``; it is not an implementation.

    Both halves matter. No descriptor means the registry cannot hand the
    parent to anyone, and no method bodies mean nothing could run if it did.
    """
    assert "descriptor" not in vars(TRTLLMGenFusedMoE)
    assert TRTLLMGenFusedMoE not in {
        MOE_IMPL_REGISTRY.lookup(ident) for ident in MOE_IMPL_REGISTRY.identities()
    }
    for name in _ABSTRACT_METHODS:
        assert name not in vars(TRTLLMGenFusedMoE)


def test_every_trtllm_gen_leaf_descends_from_the_surviving_name():
    """What the ``issubclass`` dispatch in create_moe.py depends on."""
    for impl_id in _TRTLLM_GEN_IDS:
        impl = MOE_IMPL_REGISTRY.lookup(MoEImplId.parse(impl_id))
        assert issubclass(impl, TRTLLMGenFusedMoE)


def test_trtllm_literal_picks_the_native_leaf_without_the_opt_in_flag():
    """The default provider must not move just because FlashInfer is installed."""
    with override_moe_environment(_trtllm_gen_environment(flashinfer=True)):
        report = resolve_moe_impl(_trtllm_gen_model_config())
    assert impl_class_for(report) is TrtllmTrtllmGenNvfp4Impl


def test_trtllm_literal_moves_to_flashinfer_under_the_opt_in_flag():
    """The old ``_check_flashinfer_backend_support`` switch, now in selection."""
    with override_moe_environment(
        _trtllm_gen_environment(flashinfer=True, use_flashinfer_flag=True)
    ):
        report = resolve_moe_impl(_trtllm_gen_model_config())
    assert impl_class_for(report).provider == "flashinfer"


def test_deepseek_v4_routing_is_its_own_problem_but_keeps_v3s_kernel_encoding():
    """The two answers a routing method gives, and why they have to differ.

    ``routing_method_type`` is the value handed to the C++ kernels, which have
    no encoding for V4's sqrtsoftplus scoring, so it borrows V3's. Resolution
    is not the kernel: it has a value for this algorithm and must use it, or
    every gate that names V3 silently catches V4 as well.
    """
    routing = DeepSeekV4MoeRoutingMethod(
        top_k=8,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        callable_e_score_correction_bias=lambda: None,
        callable_tid2eid=lambda: None,
    )

    assert routing.routing_method_type is RoutingMethodType.DeepSeekV3
    assert canonical_routing(routing) == RoutingMethodType.DeepSeekV4.name


@pytest.mark.parametrize(
    "routing,fused_by_the_native_runner",
    [
        ("DeepSeekV3", True),
        ("Default", True),
        ("DeepSeekV4", False),
        ("Renormalize", False),
    ],
)
def test_flashinfer_provider_turns_down_only_the_routing_it_cannot_fuse(
    routing, fused_by_the_native_runner
):
    """The routing half of the old ``_check_flashinfer_backend_support``.

    It has to name the algorithm rather than the kernel encoding: V4 shares
    V3's encoding while sharing none of the fused routing this gate objects
    to, and the provider serves it.
    """
    verdict = check_flashinfer_provider(
        FlashinferTrtllmGenNvfp4Impl,
        MoEProblem(quant="NVFP4", dtype_act=torch.bfloat16, routing=routing),
        _single_rank_deployment(_trtllm_gen_environment(flashinfer=True, use_flashinfer_flag=True)),
    )

    if fused_by_the_native_runner:
        assert verdict is not None
        assert verdict.reject_reason is MoERejectReason.ROUTING_UNSUPPORTED
    else:
        assert verdict is None


def test_pinned_flashinfer_leaf_is_turned_down_when_not_opted_in():
    """A pin fails hard, and says it was a policy call rather than a capability."""
    with override_moe_environment(_trtllm_gen_environment(flashinfer=True)):
        report = resolve_moe_impl(
            _trtllm_gen_model_config(), impl_id="flashinfer.trtllm_gen.fused_moe.nvfp4"
        )
    assert report.winner is None
    assert [rejection.reason for rejection in report.rejected] == [MoERejectReason.PATH_NOT_ENABLED]


def test_flashinfer_leaves_report_dep_missing_without_the_wheel():
    """On a host with no FlashInfer, every FlashInfer leaf declines for that reason."""
    deployment = _single_rank_deployment(_trtllm_gen_environment(use_flashinfer_flag=True))
    for impl_id in _TRTLLM_GEN_IDS:
        impl = MOE_IMPL_REGISTRY.lookup(MoEImplId.parse(impl_id))
        if impl.provider != "flashinfer":
            continue
        quant = impl.descriptor.identity.quant
        verdict = impl.can_implement(
            MoEProblem(
                quant=None if quant == "none" else quant.upper(),
                dtype_act=torch.bfloat16,
            ),
            deployment,
        )
        assert not verdict.eligible
        assert verdict.reject_reason is MoERejectReason.DEP_MISSING


@pytest.mark.parametrize(
    "activation_constants",
    [frozenset(), frozenset({"alpha"})],
    ids=["unpadded_method", "padded_method"],
)
def test_flashinfer_nvfp4_declines_expert_bias_under_either_quant_method(activation_constants):
    """Bias is a provider capability, so the shape gate cannot be its only home.

    Which quant method this leaf picks depends on the activation constants,
    and only one of the two goes through the shape gate. A bias answered there
    alone is answered for half the problems, and the other half reaches an
    ``__init__`` assert after resolution already named a winner.

    Distinct from ``swiglu_gptoss_style``: that is the bias-plus-alpha-beta
    package, and neither case here declares it.
    """
    verdict = FlashinferTrtllmGenNvfp4Impl.can_implement(
        MoEProblem(
            quant="NVFP4",
            dtype_act=torch.bfloat16,
            bias=True,
            activation_constants=activation_constants,
        ),
        _single_rank_deployment(_trtllm_gen_environment(flashinfer=True, use_flashinfer_flag=True)),
    )

    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.ACTIVATION_UNSUPPORTED


# The leaves whose runner reads no expert bias. Three because the FlashInfer
# provider passes none for any quantized format, three because the fp8/fp4
# activation cubins take none on either provider, and the unquantized one
# because its FlashInfer runner is the same.
_NO_EXPERT_BIAS_IDS = (
    "flashinfer.trtllm_gen.fused_moe.nvfp4",
    "flashinfer.trtllm_gen.fused_moe.w4a16_mxfp4",
    "flashinfer.trtllm_gen.fused_moe.w4a8_mxfp4_mxfp8",
    "trtllm.trtllm_gen.fused_moe.fp8_block_scales",
    "flashinfer.trtllm_gen.fused_moe.fp8_block_scales",
    "trtllm.trtllm_gen.fused_moe.w4a8_nvfp4_fp8",
    "flashinfer.trtllm_gen.fused_moe.none",
)


@pytest.mark.parametrize("impl_id", _NO_EXPERT_BIAS_IDS)
def test_trtllm_gen_leaves_without_a_bias_runner_decline_during_selection(impl_id):
    """A plain expert bias has to be a verdict, not a construction assert.

    ``capabilities.supports_expert_bias`` is one declaration shared by the
    whole family, and ``create_moe`` reads it against the class resolution
    already picked -- so a leaf answering only there is chosen and then
    refuses, with no chance for a sibling that can serve to be tried instead.

    Distinct from ``swiglu_gptoss_style``, which is the bias-plus-alpha-beta
    package: none of these problems declares it, and a checkpoint can carry a
    plain bias without it.
    """
    impl = MOE_IMPL_REGISTRY.lookup(MoEImplId.parse(impl_id))
    quant = impl.descriptor.identity.quant
    verdict = impl.can_implement(
        MoEProblem(
            quant=None if quant == "none" else quant.upper(),
            dtype_act=torch.bfloat16,
            bias=True,
        ),
        _single_rank_deployment(_trtllm_gen_environment(flashinfer=True, use_flashinfer_flag=True)),
    )

    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.ACTIVATION_UNSUPPORTED


@pytest.mark.parametrize("sm, eligible", [(100, True), (103, True), (107, False)])
def test_w4a8_nvfp4_fp8_leaf_is_pinned_to_its_arch_specific_cubins(sm, eligible):
    """Its E2m1 x E4m3 drop is sm_100a/sm_103a only, so the family gate is too wide."""
    impl = TrtllmTrtllmGenW4a8Nvfp4Fp8Impl
    verdict = impl.can_implement(
        MoEProblem(quant=impl.descriptor.identity.quant.upper(), dtype_act=torch.bfloat16),
        _single_rank_deployment(_trtllm_gen_environment(sm=sm)),
    )

    assert verdict.eligible == eligible
    if not eligible:
        assert verdict.reject_reason is MoERejectReason.SM_UNSUPPORTED


# The leaves whose runner reads no activation constant: their cubins have no
# parameter for alpha, beta or the clamp and pin ``act_type``.
_NO_ACTIVATION_CONSTANT_LEAVES = (
    TrtllmTrtllmGenW4a8Nvfp4Fp8Impl,
    FlashinferTrtllmGenBf16Impl,
)


@pytest.mark.parametrize(
    "impl", _NO_ACTIVATION_CONSTANT_LEAVES, ids=lambda c: c.descriptor.identity.canonical()
)
@pytest.mark.parametrize("constant", ["clamp", "alpha"])
def test_leaves_without_constant_support_decline_during_selection(impl, constant):
    """A lone clamp is not the gpt-oss package, and has to be its own verdict.

    ``swiglu_gptoss_style`` sees bias plus alpha and beta as one package, and
    the clamp is not in it at all, so a checkpoint carrying only a clamp gets
    past that gate. Answering it in ``_check_configs`` alone means resolution
    names a winner and construction then asserts -- and for bf16 it forecloses
    the fall back to ``CutlassFusedMoE`` that would have served the case.
    """
    quant = impl.descriptor.identity.quant
    verdict = impl.can_implement(
        MoEProblem(
            quant=None if quant == "none" else quant.upper(),
            dtype_act=torch.bfloat16,
            activation_constants=frozenset({constant}),
        ),
        _single_rank_deployment(_trtllm_gen_environment(flashinfer=True, use_flashinfer_flag=True)),
    )

    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.ACTIVATION_UNSUPPORTED


@pytest.mark.parametrize("alias", [QuantAlgo.NVFP4_AWQ, QuantAlgo.NVFP4_ARC])
def test_nvfp4_leaves_admit_the_calibration_aliases(alias):
    """The identity gate has to fold the aliases its own lookup folds.

    ``find_trtllm_gen_leaf`` maps NVFP4_AWQ and NVFP4_ARC onto the ``nvfp4``
    leaf -- the recipes differ in calibration, the weights and the kernel do
    not. A gate comparing the raw string would have that leaf turn down the
    format it is the registered implementation of.
    """
    verdict = TrtllmTrtllmGenNvfp4Impl.can_implement(
        MoEProblem(quant=alias.value, dtype_act=torch.bfloat16),
        _single_rank_deployment(_trtllm_gen_environment()),
    )

    assert verdict.eligible, verdict.detail


def test_resolution_order_lists_flashinfer_before_its_native_sibling():
    """Filtering callers need resolution's order, not a lookup's.

    ``IMPL_PRIORITY`` ranks each FlashInfer leaf ahead of its native sibling
    and the opt-in flag is what makes the FlashInfer ones reject, so a caller
    asking "would a run serve this?" has to walk both and fall through on a
    rejection. Provider-exclusive formats yield the one leaf that exists.
    """
    assert trtllm_gen_leaves_in_resolution_order(QuantAlgo.NVFP4) == (
        FlashinferTrtllmGenNvfp4Impl,
        TrtllmTrtllmGenNvfp4Impl,
    )
    assert trtllm_gen_leaves_in_resolution_order(None) == (FlashinferTrtllmGenBf16Impl,)
    assert trtllm_gen_leaves_in_resolution_order(QuantAlgo.W4A8_NVFP4_FP8) == (
        TrtllmTrtllmGenW4a8Nvfp4Fp8Impl,
    )


# The two native leaves whose cubin family has a fused SiTu FC1 epilogue:
# NVFP4 feeds the group-16 ``Bmm_E2m1_E2m1E2m1_..._siTuGlu_*`` kernels,
# W4A8_MXFP4_MXFP8 the group-32 ``Bmm_MxE4m3_..._siTuGlu_*`` ones.
_SITU_CAPABLE_IDS = frozenset(
    {
        "trtllm.trtllm_gen.fused_moe.nvfp4",
        "trtllm.trtllm_gen.fused_moe.w4a8_mxfp4_mxfp8",
    }
)


@pytest.mark.parametrize("impl_id", _TRTLLM_GEN_IDS)
def test_trtllm_gen_situ_admitted_only_by_the_leaves_with_a_fused_cubin(impl_id):
    """SiTu has to be answered during selection, not after a leaf is chosen.

    There is no standalone SiTu activation kernel, so a leaf whose quantization
    format has no fused epilogue cannot run it at all. Answering that only in
    ``__init__`` means resolution hands back a winner and construction then
    raises, which reads as a crash rather than as a format that was never
    eligible.

    The FlashInfer leaves are turned down here even for the two capable
    formats: SiTu is a native TRTLLM-Gen cubin and is absent from FlashInfer's
    activation enum.
    """
    impl = MOE_IMPL_REGISTRY.lookup(MoEImplId.parse(impl_id))
    quant = impl.descriptor.identity.quant
    verdict = impl.can_implement(
        MoEProblem(
            quant=None if quant == "none" else quant.upper(),
            dtype_act=torch.bfloat16,
            activation="SiTu",
        ),
        _single_rank_deployment(_trtllm_gen_environment(flashinfer=True, use_flashinfer_flag=True)),
    )
    if impl_id in _SITU_CAPABLE_IDS:
        assert verdict.eligible, verdict.detail
    else:
        assert not verdict.eligible
        assert verdict.reject_reason is MoERejectReason.ACTIVATION_UNSUPPORTED


# =====================================================================
# TrtllmCutedslFusedFc12Nvfp4Impl implementation identity
# =====================================================================
# Problem and deployment are passed explicitly, as for MegaMoE: the FC12 gates
# read SM, the Rubin CuTe DSL dependency, quantization, dtype and finalize
# fusion, and stating them keeps one variable per test.

_FC12_IMPL_ID = "trtllm.cutedsl.fused_fc12.nvfp4"


def _fc12_problem() -> MoEProblem:
    """The DeepSeek-V4-Pro routed shape, which every FC12 gate admits."""
    return MoEProblem(
        quant=canonical_quant(QuantAlgo.NVFP4),
        dtype_act=torch.bfloat16,
        hidden_size=7168,
        intermediate_size=3072,
        num_experts=384,
        top_k=6,
        swiglu_gptoss_style=False,
    )


def _fc12_deployment(sm: int = 107) -> MoEDeployment:
    """Single rank with the CuTe DSL Rubin helpers present; ``sm`` is the only variable."""
    return MoEDeployment(
        ep_size=1,
        tp_size=1,
        parallel_size=1,
        use_dp=False,
        num_slots=384,
        env=MoEEnvironment(sm=sm, available_deps=(MoEDep.CUTEDSL_RUBIN.value,)),
    )


def test_fc12_identity_round_trips_through_registry():
    """One class owns the id, the four methods, and the published contract."""
    impl = TrtllmCutedslFusedFc12Nvfp4Impl
    identity = impl.descriptor.identity
    assert identity.canonical() == _FC12_IMPL_ID
    assert MoEImplId.parse(_FC12_IMPL_ID) == identity
    assert MOE_IMPL_REGISTRY.lookup(identity) is impl
    assert "descriptor" in vars(impl)
    for name in ("can_implement", "_get_quant_method", "quantize_input", "run_moe"):
        assert name in vars(impl)
    assert not impl.__abstractmethods__
    assert impl.scheduler_kind is impl.descriptor.scheduler_kind
    assert impl.capabilities is impl.descriptor.capabilities
    assert impl.input_requirement is impl.descriptor.input_requirement


def test_pinned_fc12_identity_fails_hard_where_the_backend_literal_degrades():
    """On Rubin both tracks land on FC12; off Rubin the literal degrades and the pin does not."""
    config = ModelConfig()
    config.moe_backend = MoeBackendType.CUTEDSL_FC12.value
    problem = _fc12_problem()

    on_rubin = resolve_moe_impl(
        config, problem=problem, deployment=_fc12_deployment(), impl_id=_FC12_IMPL_ID
    )
    assert impl_class_for(on_rubin) is TrtllmCutedslFusedFc12Nvfp4Impl
    assert on_rubin.selected_by == "pinned"

    off_rubin = _fc12_deployment(sm=100)
    by_literal = resolve_moe_impl(config, problem=problem, deployment=off_rubin)
    by_identity = resolve_moe_impl(
        config, problem=problem, deployment=off_rubin, impl_id=_FC12_IMPL_ID
    )
    assert impl_class_for(by_literal) is not TrtllmCutedslFusedFc12Nvfp4Impl
    assert by_literal.degraded
    assert by_identity.winner is None
    assert [rejection.reason for rejection in by_identity.rejected] == [
        MoERejectReason.SM_UNSUPPORTED
    ]
    with pytest.raises(ValueError, match="no MoE implementation can serve"):
        impl_class_for(by_identity)


# =====================================================================
# Marlin: two formats over one kernel
# =====================================================================
# Both leaves run the same ``marlin_nvfp4_moe_gemm`` launch sequence, and are
# still two identities because the ``quant`` a leaf publishes has to be the
# ``quant`` the request named.

_MARLIN_IDS = (
    "marlin.cuda.fused_moe.nvfp4",
    "marlin.cuda.fused_moe.w4a16_nvfp4",
)


def _marlin_environment(sm: int = 90) -> MoEEnvironment:
    """Marlin's own SM window, so quantization stays the only variable."""
    return MoEEnvironment(sm=sm)


def _marlin_model_config(quant_algo=QuantAlgo.NVFP4):
    cfg = ModelConfig()
    cfg.moe_backend = "MARLIN"
    cfg.quant_config = QuantConfig(quant_algo=quant_algo) if quant_algo else None
    return cfg


def test_marlin_registers_exactly_two_identities():
    """The grid is the deliverable: two addressable ids, no more, no fewer."""
    registered = sorted(
        identity.canonical()
        for identity in MOE_IMPL_REGISTRY.identities()
        if identity.provider == "marlin"
    )
    assert registered == sorted(_MARLIN_IDS)


@pytest.mark.parametrize("impl_id", _MARLIN_IDS)
def test_marlin_identity_round_trips_through_registry(impl_id):
    impl = MOE_IMPL_REGISTRY.lookup(MoEImplId.parse(impl_id))
    assert impl is not None
    # ``vars``, not attribute access: an inherited descriptor would give both
    # classes one identity.
    assert "descriptor" in vars(impl)
    assert impl.descriptor.identity.canonical() == impl_id
    assert not impl.__abstractmethods__


@pytest.mark.parametrize("impl_id", _MARLIN_IDS)
def test_marlin_leaf_admits_only_its_own_format(impl_id):
    """Two leaves over one kernel still admit one format each."""
    impl = MOE_IMPL_REGISTRY.lookup(MoEImplId.parse(impl_id))
    quant = impl.descriptor.identity.quant
    other = "w4a16_nvfp4" if quant == "nvfp4" else "nvfp4"
    verdict = impl.can_implement(
        MoEProblem(quant=other.upper(), dtype_act=torch.bfloat16),
        _single_rank_deployment(_marlin_environment()),
    )
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.QUANT_UNSUPPORTED


@pytest.mark.parametrize("impl_id", _MARLIN_IDS)
def test_marlin_leaf_admits_the_format_it_publishes(impl_id):
    impl = MOE_IMPL_REGISTRY.lookup(MoEImplId.parse(impl_id))
    verdict = impl.can_implement(
        MoEProblem(
            quant=impl.descriptor.identity.quant.upper(),
            dtype_act=torch.bfloat16,
        ),
        _single_rank_deployment(_marlin_environment()),
    )
    assert verdict.eligible, verdict.detail


def test_marlin_parent_is_abstract_and_unaddressable():
    """The name survives for ``issubclass``; it is not an implementation."""
    assert "descriptor" not in vars(MarlinFusedMoE)
    assert MarlinFusedMoE not in {
        MOE_IMPL_REGISTRY.lookup(ident) for ident in MOE_IMPL_REGISTRY.identities()
    }
    assert "can_implement" not in vars(MarlinFusedMoE)
    assert MarlinFusedMoE.__abstractmethods__


def test_every_marlin_leaf_descends_from_the_surviving_name():
    """What the ``issubclass`` dispatch in create_moe.py depends on."""
    for impl_id in _MARLIN_IDS:
        assert issubclass(MOE_IMPL_REGISTRY.lookup(MoEImplId.parse(impl_id)), MarlinFusedMoE)


@pytest.mark.parametrize(
    "quant_algo, expected",
    [
        pytest.param(QuantAlgo.NVFP4, MarlinCudaNvfp4Impl, id="nvfp4"),
        pytest.param(QuantAlgo.W4A16_NVFP4, MarlinCudaW4a16Nvfp4Impl, id="w4a16_nvfp4"),
    ],
)
def test_marlin_literal_picks_the_leaf_that_publishes_the_format(quant_algo, expected):
    """``moe_backend: MARLIN`` names the family; the quant picks which leaf."""
    with override_moe_environment(_marlin_environment()):
        report = resolve_moe_impl(_marlin_model_config(quant_algo))
    assert impl_class_for(report) is expected
    assert report.selected_by == "pinned"


@pytest.mark.parametrize(
    "quant_algo, impl_id",
    [
        pytest.param(QuantAlgo.NVFP4, _MARLIN_IDS[0], id="nvfp4"),
        pytest.param(QuantAlgo.W4A16_NVFP4, _MARLIN_IDS[1], id="w4a16_nvfp4"),
    ],
)
def test_pinned_marlin_identity_resolves_to_its_leaf(quant_algo, impl_id):
    with override_moe_environment(_marlin_environment()):
        report = resolve_moe_impl(_marlin_model_config(quant_algo), impl_id=impl_id)
    assert impl_class_for(report).descriptor.identity.canonical() == impl_id
    assert report.selected_by == "pinned"


# =====================================================================
# CUTLASS: one leaf per format
# =====================================================================
# The family literal picks a leaf by format and, for ``fp8_block_scales``, by
# SM; the leaf is then fixed for the layer's lifetime.


def _cutlass_model_config(quant_algo=QuantAlgo.NVFP4):
    cfg = ModelConfig()
    cfg.moe_backend = "CUTLASS"
    cfg.quant_config = QuantConfig(quant_algo=quant_algo) if quant_algo else None
    return cfg


_HP = frozenset({torch.float16, torch.bfloat16})
_HP_FP32 = _HP | {torch.float32}
_FROM_AMPERE = frozenset(range(80, 125))
_PROBED_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float8_e4m3fn)

# Every CUTLASS leaf's declarations, written out by hand rather than read off
# the leaves, so one that drifts fails here: the SMs it admits (probed over
# SM75-124), its activation dtypes, whether its loader takes a gpt-oss / 1-D
# expert bias, what needs the open-source CUTLASS build, the kernel flags it
# hands the op, and whether it fuses LoRA.
_CUTLASS_DECLARATIONS = (
    pytest.param(
        TrtllmCutlassUnquantizedImpl,
        _FROM_AMPERE,
        _HP,
        (True, False),
        (False, False),
        FusedMoeOpFlags(),
        True,
        id="none",
    ),
    pytest.param(
        TrtllmCutlassFp8Impl,
        frozenset(range(89, 125)),
        _HP_FP32,
        (False, False),
        (False, False),
        FusedMoeOpFlags(),
        True,
        id="fp8",
    ),
    pytest.param(
        DeepgemmCudaHopperFp8BlockScalesImpl,
        {90},
        {torch.bfloat16},
        (False, False),
        (False, False),
        FusedMoeOpFlags(use_deepseek_fp8_block_scale=True),
        False,
        id="fp8_block_scales-sm90",
    ),
    pytest.param(
        TrtllmTritonFp8BlockScalesImpl,
        {120, 121},
        {torch.bfloat16},
        (False, False),
        (False, False),
        FusedMoeOpFlags(),
        False,
        id="fp8_block_scales-sm120",
    ),
    pytest.param(
        TrtllmCutlassNvfp4Impl,
        {100, 103, 107, 120, 121},
        _HP | {torch.float8_e4m3fn},
        (True, True),
        (False, True),
        FusedMoeOpFlags(),
        False,
        id="nvfp4",
    ),
    pytest.param(
        TrtllmCutlassW4a16Nvfp4Impl,
        _FROM_AMPERE,
        _HP,
        (False, False),
        (False, False),
        FusedMoeOpFlags(),
        False,
        id="w4a16_nvfp4",
    ),
    pytest.param(
        TrtllmCutlassW4a8AwqImpl,
        {89, 90},
        _HP,
        (False, False),
        (True, False),
        FusedMoeOpFlags(weight_dtype=torch.quint4x2, use_w4_group_scaling=True),
        False,
        id="w4a8_awq",
    ),
    pytest.param(
        TrtllmCutlassW8a16Impl,
        _FROM_AMPERE,
        _HP,
        (False, False),
        (False, False),
        FusedMoeOpFlags(use_int8_woq_per_channel=True),
        False,
        id="w8a16",
    ),
    pytest.param(
        TrtllmCutlassW4a16Mxfp4Impl,
        {90},
        _HP,
        (True, False),
        (True, False),
        FusedMoeOpFlags(weight_dtype=torch.uint8, use_w4_group_scaling=True),
        False,
        id="w4a16_mxfp4",
    ),
    pytest.param(
        TrtllmCutlassW4a8Mxfp4Fp8Impl,
        {100, 103, 107},
        _HP_FP32,
        (True, False),
        (False, False),
        FusedMoeOpFlags(),
        False,
        id="w4a8_mxfp4_fp8",
    ),
    pytest.param(
        TrtllmCutlassW4a8Mxfp4Mxfp8Impl,
        {100, 103, 107, 120, 121},
        _HP,
        (True, False),
        (True, False),
        FusedMoeOpFlags(use_mxfp8_act_scaling=True),
        False,
        id="w4a8_mxfp4_mxfp8",
    ),
    pytest.param(
        TrtllmCutlassMxfp8Impl,
        {100, 103, 107},
        _HP,
        (True, False),
        (True, False),
        FusedMoeOpFlags(use_mxfp8_act_scaling=True, use_mxfp8_weight_scaling=True),
        False,
        id="mxfp8",
    ),
)


def test_cutlass_declarations_cover_every_leaf():
    assert {case.values[0] for case in _CUTLASS_DECLARATIONS} == set(CUTLASS_LEAVES)


@pytest.mark.parametrize(
    "leaf, sms, dtypes, gptoss, oss_build, flags, fuses_lora", _CUTLASS_DECLARATIONS
)
def test_cutlass_leaf_declarations(leaf, sms, dtypes, gptoss, oss_build, flags, fuses_lora):
    # Everything but the op flags is probed through can_implement, so a gate
    # that stops enforcing a declaration fails here, not only a declaration
    # change; test_cutlass_leaf_run_moe_hands_the_op_its_declarations covers
    # the flags.
    identity = identity_quant_of(leaf)
    quant = next(
        canonical_quant(algo)
        for algo in (None, *QuantAlgo)
        if normalize_quant(canonical_quant(algo)) == identity
    )

    def admits(
        sm: int = min(sms),
        dtype: torch.dtype = torch.bfloat16,
        *,
        oss: bool = True,
        dynamic: bool = False,
        lora: bool = False,
        **problem_fields: object,
    ) -> bool:
        problem = MoEProblem(quant=quant, dtype_act=dtype, **problem_fields)
        deployment = dataclasses.replace(
            _single_rank_deployment(MoEEnvironment(sm=sm, oss_cutlass_moe=oss)),
            force_dynamic_quantization=dynamic,
            moe_lora_enabled=lora,
        )
        return leaf.can_implement(problem, deployment).eligible

    assert {sm for sm in range(75, 125) if admits(sm)} == set(sms)
    assert {dtype for dtype in _PROBED_DTYPES if admits(dtype=dtype)} == set(dtypes)
    gptoss_style = admits(swiglu_gptoss_style=True)
    gptoss_bias = admits(swiglu_gptoss_style=True, bias=True)
    assert (gptoss_style, gptoss_style and not gptoss_bias) == gptoss
    internal = admits(oss=False)
    assert (not internal, internal and not admits(oss=False, dynamic=True)) == oss_build
    assert leaf.FUSED_MOE_OP_FLAGS == flags
    # The scheduler hands lora_params only to a leaf declaring LoRA support,
    # and only the mixin consumes them.
    assert admits(lora=True) is fuses_lora
    assert leaf.capabilities.supports_moe_lora is fuses_lora
    assert issubclass(leaf, CutlassMoELoraMixin) is fuses_lora

    # Taken off the leaf's own descriptor.
    assert "descriptor" in vars(leaf)
    assert leaf.scheduler_kind is leaf.descriptor.scheduler_kind
    assert leaf.capabilities is leaf.descriptor.capabilities
    assert leaf.input_requirement is leaf.descriptor.input_requirement


# The ``quantize_input`` each leaf binds where it is not the base pass-through,
# and the size a leaf pads hidden / intermediate to at construction.
_CUTLASS_QUANTIZE_INPUT = {
    TrtllmCutlassFp8Impl: quantize_input_static_e4m3,
    TrtllmCutlassW4a8Mxfp4Fp8Impl: quantize_input_static_e4m3,
    TrtllmCutlassMxfp8Impl: quantize_input_mxfp8,
    TrtllmCutlassW4a8Mxfp4Mxfp8Impl: quantize_input_mxfp8,
    TrtllmCutlassNvfp4Impl: vars(TrtllmCutlassNvfp4Impl).get("quantize_input"),
    TrtllmCutlassW4a16Nvfp4Impl: vars(TrtllmCutlassW4a16Nvfp4Impl).get("quantize_input"),
}
_CUTLASS_PAD_TO = {TrtllmCutlassW4a16Mxfp4Impl: 128}


@pytest.mark.parametrize("leaf", CUTLASS_LEAVES, ids=lambda leaf: leaf.__name__)
def test_cutlass_leaf_input_declarations(leaf):
    expected = _CUTLASS_QUANTIZE_INPUT.get(leaf, CutlassFusedMoEBase.quantize_input)
    assert expected is not None
    assert leaf.quantize_input is expected
    assert leaf.pad_hidden_and_intermediate_to == _CUTLASS_PAD_TO.get(leaf)
    # The FP8 leaf quantizes before the op, so its LoRA operands are e4m3 too.
    lora_input_dtype = torch.float8_e4m3fn if leaf is TrtllmCutlassFp8Impl else None
    assert getattr(leaf, "LORA_OP_INPUT_DTYPE", None) == lora_input_dtype


@pytest.mark.parametrize(
    "sm, impl_id",
    [
        pytest.param(90, "deepgemm.cuda.hopper_grouped_gemm.fp8_block_scales", id="sm90"),
        pytest.param(120, "trtllm.triton.grouped_gemm.fp8_block_scales", id="sm120"),
    ],
)
def test_cutlass_literal_picks_the_fp8_block_scales_leaf_by_sm(sm, impl_id):
    """``fp8_block_scales`` is published twice in the family, once per SM."""
    with override_moe_environment(MoEEnvironment(sm=sm)):
        report = resolve_moe_impl(
            _cutlass_model_config(QuantAlgo.FP8_BLOCK_SCALES), dtype=torch.bfloat16
        )
    assert impl_class_for(report).descriptor.identity.canonical() == impl_id
    assert not report.degraded


def test_cutlass_leaf_names_the_registered_formats_when_none_serves():
    assert cutlass_leaf(QuantAlgo.NVFP4) is TrtllmCutlassNvfp4Impl
    assert cutlass_leaf(QuantAlgo.FP8_BLOCK_SCALES, sm=120) is TrtllmTritonFp8BlockScalesImpl
    with pytest.raises(
        ValueError, match="on SM100; registered: .*trtllm.cutlass.grouped_gemm.nvfp4"
    ):
        cutlass_leaf(QuantAlgo.FP8_BLOCK_SCALES, sm=100)
    # Without an explicit sm, the message names the SM the lookup resolved.
    with override_moe_environment(MoEEnvironment(sm=100)):
        with pytest.raises(ValueError, match="quant=fp8_block_scales on SM100; registered: "):
            cutlass_leaf(QuantAlgo.FP8_BLOCK_SCALES)
    # A format no leaf publishes is not blamed on the SM.
    with pytest.raises(ValueError, match="quant=w4a8_nvfp4_fp8; registered: "):
        cutlass_leaf(QuantAlgo.W4A8_NVFP4_FP8, sm=100)


def test_degradation_names_the_requested_backends_rejection(monkeypatch):
    """The fallback leaves are asked too, after the requested family; their
    rejections must not stand in for why the request could not be served."""
    logger = MagicMock()
    monkeypatch.setattr(moe_resolution, "logger", logger)
    with override_moe_environment(_deepgemm_environment(sm=90)):
        report = resolve_moe_impl(_deepgemm_model_config(), dtype=torch.bfloat16, layer_idx=3)

    assert impl_class_for(report) is DeepgemmCudaHopperFp8BlockScalesImpl
    assert report.degraded
    cause = report.degraded_from
    assert cause.legacy_backend == DeepgemmCudaFp8BlockScalesImpl.__name__
    assert cause.reason is not MoERejectReason.QUANT_UNSUPPORTED

    # The warning gives that cause in full and names no fallback leaf.
    (warning,), _ = logger.warning.call_args
    assert "cannot serve this layer [layer_idx=3]" in warning
    assert cause.describe() in warning
    assert f"running {DeepgemmCudaHopperFp8BlockScalesImpl.__name__} instead" in warning
    fallback_rejections = [
        r.legacy_backend for r in report.rejected if r.legacy_backend not in report.requested_impls
    ]
    assert fallback_rejections
    assert not any(name in warning for name in fallback_rejections)


def test_degradation_names_the_first_requested_rejection_not_about_format_or_sm():
    """In a multi-leaf family most members decline on format or SM alone; the
    reason worth reporting comes from the highest-priority one that declined
    for something else."""
    report = MoEResolutionReport(
        problem=MagicMock(),
        deployment=MagicMock(),
        winner="Fallback",
        selected_by="heuristic",
        rejected=(
            MoERejection("OtherFormat", MoERejectReason.QUANT_UNSUPPORTED),
            MoERejection("OtherSm", MoERejectReason.SM_UNSUPPORTED),
            MoERejection("ServesFormat", MoERejectReason.TOPOLOGY_UNSUPPORTED),
            MoERejection("FallbackLeaf", MoERejectReason.SM_UNSUPPORTED),
        ),
        requested_impls=frozenset({"OtherFormat", "OtherSm", "ServesFormat"}),
    )
    assert report.degraded_from.legacy_backend == "ServesFormat"

    sm_and_format = dataclasses.replace(report, rejected=report.rejected[:2])
    assert sm_and_format.degraded_from.legacy_backend == "OtherSm"

    only_format = dataclasses.replace(report, rejected=report.rejected[:1])
    assert only_format.degraded_from.legacy_backend == "OtherFormat"

    # A fallback leaf's SM rejection outranks a format one, but is never the cause.
    with_fallback = dataclasses.replace(report, rejected=(report.rejected[0], report.rejected[3]))
    assert with_fallback.degraded_from.legacy_backend == "OtherFormat"


def test_cutedsl_degradation_is_not_blamed_on_the_sm120_leaf():
    """The b12x leaves lead the CUTEDSL family and have SM120/121 kernels only.

    On SM100 the NVFP4 one passes its format gate and declines on SM, ahead of
    the grouped-GEMM leaf's rejection that is the actual reason this layer fell
    back.
    """
    cfg = ModelConfig()
    cfg.moe_backend = MoeBackendType.CUTEDSL.value
    problem = MoEProblem(
        quant=canonical_quant(QuantAlgo.NVFP4),
        # The grouped GEMM's output is hard-wired to bfloat16; Cutlass takes fp16.
        dtype_act=torch.float16,
        hidden_size=1024,
        intermediate_size=1024,
        num_experts=8,
        top_k=2,
        swiglu_gptoss_style=False,
    )
    deployment = MoEDeployment(
        ep_size=1,
        tp_size=1,
        parallel_size=1,
        use_dp=False,
        num_slots=8,
        env=MoEEnvironment(sm=100),
    )

    report = resolve_moe_impl(cfg, problem=problem, deployment=deployment)

    assert impl_class_for(report) is TrtllmCutlassNvfp4Impl
    assert report.degraded
    assert (report.rejected[0].legacy_backend, report.rejected[0].reason) == (
        CuteDslB12xNvfp4FusedMoE.__name__,
        MoERejectReason.SM_UNSUPPORTED,
    )
    assert (report.degraded_from.legacy_backend, report.degraded_from.reason) == (
        TrtllmCutedslGroupedGemmNvfp4Impl.__name__,
        MoERejectReason.DTYPE_UNSUPPORTED,
    )


def test_cutedsl_float16_nvfp4_on_sm120_falls_back_to_cutlass():
    """The b12x kernel reads and writes bfloat16 only, so on its own SM a
    float16 NVFP4 layer goes to the CUTLASS leaf, which takes float16, and the
    b12x leaf's dtype is named as the reason."""
    cfg = ModelConfig()
    cfg.moe_backend = MoeBackendType.CUTEDSL.value
    problem = MoEProblem(
        quant=canonical_quant(QuantAlgo.NVFP4),
        dtype_act=torch.float16,
        hidden_size=1024,
        intermediate_size=1024,
        num_experts=8,
        top_k=2,
        swiglu_gptoss_style=False,
    )
    deployment = MoEDeployment(
        ep_size=1,
        tp_size=1,
        parallel_size=1,
        use_dp=False,
        num_slots=8,
        env=MoEEnvironment(sm=120, available_deps=(MoEDep.FLASHINFER.value,)),
    )

    report = resolve_moe_impl(cfg, problem=problem, deployment=deployment)

    assert impl_class_for(report) is TrtllmCutlassNvfp4Impl
    assert (report.degraded_from.legacy_backend, report.degraded_from.reason) == (
        CuteDslB12xNvfp4FusedMoE.__name__,
        MoERejectReason.DTYPE_UNSUPPORTED,
    )


def test_cutedsl_dynamic_quant_nvfp4_on_sm120_falls_back_to_cutlass():
    """The b12x decode kernel quantizes against load-time scales, so a layer
    that forces dynamic quantization goes to the CUTLASS leaf, which honours it
    on every chunk. The b12x leaf is named as the reason: its rejection must not
    read as a format mismatch, which ``degraded_from`` passes over."""
    cfg = ModelConfig()
    cfg.moe_backend = MoeBackendType.CUTEDSL.value
    problem = MoEProblem(
        quant=canonical_quant(QuantAlgo.NVFP4),
        dtype_act=torch.bfloat16,
        hidden_size=1024,
        intermediate_size=1024,
        num_experts=8,
        top_k=2,
        swiglu_gptoss_style=False,
    )
    deployment = MoEDeployment(
        ep_size=1,
        tp_size=1,
        parallel_size=1,
        use_dp=False,
        num_slots=8,
        force_dynamic_quantization=True,
        env=MoEEnvironment(sm=120, available_deps=(MoEDep.FLASHINFER.value,)),
    )

    report = resolve_moe_impl(cfg, problem=problem, deployment=deployment)

    assert impl_class_for(report) is TrtllmCutlassNvfp4Impl
    assert (report.degraded_from.legacy_backend, report.degraded_from.reason) == (
        CuteDslB12xNvfp4FusedMoE.__name__,
        MoERejectReason.DYNAMIC_QUANT_UNSUPPORTED,
    )


# =====================================================================
# A layer's format is fixed when it is built
# =====================================================================


@pytest.mark.parametrize(
    "leaf, own_formats, moved_format",
    [
        pytest.param(
            TrtllmTrtllmGenNvfp4Impl,
            (QuantAlgo.NVFP4, QuantAlgo.NVFP4_AWQ),
            None,
            id="trtllm_gen",
        ),
        pytest.param(
            TrtllmCutlassNvfp4Impl, (QuantAlgo.NVFP4, QuantAlgo.NVFP4_AWQ), None, id="cutlass"
        ),
        pytest.param(
            CuteDslB12xNvfp4FusedMoE,
            (QuantAlgo.NVFP4, QuantAlgo.NVFP4_AWQ),
            None,
            id="b12x_nvfp4",
        ),
        pytest.param(
            CuteDslB12xW4a16Nvfp4FusedMoE,
            (QuantAlgo.W4A16_NVFP4,),
            QuantAlgo.NVFP4,
            id="b12x_w4a16_nvfp4",
        ),
        pytest.param(
            TrtllmCutedslGroupedGemmNvfp4Impl,
            (QuantAlgo.NVFP4, QuantAlgo.NVFP4_AWQ),
            None,
            id="cutedsl_grouped_gemm_nvfp4",
        ),
        pytest.param(
            TrtllmCutedslGroupedGemmBf16Impl,
            (None,),
            QuantAlgo.NVFP4,
            id="cutedsl_grouped_gemm_none",
        ),
    ],
)
def test_a_leaf_refuses_a_layer_whose_format_moved_after_it_was_picked(
    leaf, own_formats, moved_format
):
    """``create_weights`` refuses a final ``quant_config`` of another format.

    The ``__post_init__`` passes that moved the layer have returned by then, so
    the message carries the diagnosis. ``__new__`` without ``__init__``: the
    check reads three attributes and no allocated state, and a real
    constructor would need a GPU.
    """
    moved = leaf.__new__(leaf)
    moved._weights_created = False
    moved.layer_idx = 3
    moved.quant_config = QuantConfig(quant_algo=moved_format)
    with pytest.raises(ValueError) as excinfo:
        moved.create_weights()
    assert "layer 3" in str(excinfo.value)
    assert "override_quant_config" in str(excinfo.value)

    for quant_algo in own_formats:
        kept = leaf.__new__(leaf)
        kept.layer_idx = 3
        kept.quant_config = QuantConfig(quant_algo=quant_algo)
        kept._check_quant_config_is_my_format()


def test_moe_experts_quant_config_reads_the_layer_entry_then_the_exclusion():
    """What a model passes to ``create_moe`` as ``override_quant_config``."""
    from tensorrt_llm._torch.models.modeling_utils import moe_experts_quant_config

    w4a8 = QuantConfig(quant_algo=QuantAlgo.W4A8_AWQ)
    nvfp4 = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    cfg = SimpleNamespace(
        quant_config=QuantConfig(
            quant_algo=QuantAlgo.MIXED_PRECISION,
            kv_cache_quant_algo=QuantAlgo.FP8,
            exclude_modules=["model.layers.2.mlp.experts"],
        ),
        quant_config_dict={
            "model.layers.0.mlp.experts.0.gate_proj": w4a8,
            "model.layers.2.mlp.experts.0.gate_proj": w4a8,
            "model.language_model.layers.3.mlp.experts.0.gate_proj": nvfp4,
        },
    )

    assert moe_experts_quant_config(cfg, "model.layers.0.mlp.experts") is w4a8
    assert moe_experts_quant_config(cfg, "model.layers.1.mlp.experts") is None
    # A text model built inside a multimodal one keys its weights under the
    # outer namespace; __post_init__ still finds that entry, so this must too.
    assert moe_experts_quant_config(cfg, "model.layers.3.mlp.experts") is nvfp4
    excluded = moe_experts_quant_config(cfg, "model.layers.2.mlp.experts")
    assert excluded.quant_algo is None
    assert excluded.kv_cache_quant_algo == QuantAlgo.FP8


def test_moe_experts_quant_config_prefers_the_entry_naming_the_module():
    """``ModelConfig`` appends a coarse per-layer expert entry for DeepSeek-V4
    and MiniMax-M3 because the checkpoint's per-linear expert keys do not
    describe the module; those keys come first in the dict and must not win."""
    from tensorrt_llm._torch.models.modeling_utils import moe_experts_quant_config

    per_linear = QuantConfig(quant_algo=QuantAlgo.W4A8_AWQ)
    whole_module = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    experts = "model.layers.3.block_sparse_moe.experts"
    cfg = SimpleNamespace(
        quant_config=QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION),
        quant_config_dict={
            f"{experts}.0.w1": per_linear,
            experts: whole_module,
        },
    )

    assert moe_experts_quant_config(cfg, experts) is whole_module


# Each MoE block, the constructor arguments besides ``model_config`` and
# ``layer_idx``, and the attribute its decoder layer holds it under.
_MOE_BLOCKS = (
    pytest.param("modeling_mixtral", "MixtralMoE", {"aux_stream": None}, "block_sparse_moe"),
    pytest.param("modeling_qwen_moe", "QwenMoE", {"aux_stream": None}, "mlp"),
    pytest.param("modeling_qwen3_moe", "Qwen3MoE", {"aux_stream_dict": {}}, "mlp"),
    pytest.param("modeling_qwen3_next", "Qwen3NextSparseMoeBlock", {"aux_stream": None}, "mlp"),
    pytest.param("modeling_gemma4", "Gemma4MoE", {}, "moe"),
    pytest.param("modeling_hunyuan_moe", "HunyuanMoE", {"aux_stream": None}, "mlp"),
    pytest.param("modeling_afmoe", "AfmoeMoE", {"aux_stream": None}, "mlp"),
    pytest.param("modeling_minimaxm2", "MiniMaxM2MoE", {"aux_stream": None}, "block_sparse_moe"),
    pytest.param("modeling_laguna", "LagunaMoE", {"aux_stream_dict": {}}, "mlp"),
)

# Replaced wherever a model module imports them: only the create_moe call is
# under test.
_MOE_BLOCK_STUBS = (
    "create_moe",
    "resolve_moe_cls",
    "Linear",
    "GatedMLP",
    "AllReduce",
    "Qwen3Gate",
    "Qwen3NextGate",
    "AfmoeGate",
    "Gemma4Router",
    "LagunaGate",
)


@pytest.mark.parametrize("module_name, class_name, kwargs, attr", _MOE_BLOCKS)
def test_moe_blocks_pass_their_experts_quant_config(
    module_name, class_name, kwargs, attr, monkeypatch
):
    """A MIXED_PRECISION entry keyed per linear reaches ``create_moe``."""
    module = importlib.import_module(f"tensorrt_llm._torch.models.{module_name}")
    for name in _MOE_BLOCK_STUBS:
        if hasattr(module, name):
            monkeypatch.setattr(module, name, MagicMock())
    nvfp4 = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    model_config = ModelConfig(
        pretrained_config=SimpleNamespace(
            model_type="test",
            hidden_size=8,
            intermediate_size=16,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            num_experts=4,
            num_local_experts=4,
            num_experts_per_tok=2,
            moe_topk=2,
            top_k_experts=2,
            n_group=1,
            topk_group=1,
            mlp_bias=False,
            torch_dtype=torch.float32,
        ),
        quant_config=QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION),
        quant_config_dict={f"model.layers.3.{attr}.experts.0.gate_proj": nvfp4},
    )

    getattr(module, class_name)(model_config, layer_idx=3, **kwargs)

    assert module.create_moe.call_args.kwargs["override_quant_config"] is nvfp4


def test_llama4_min_latency_experts_keep_the_layer_quant_config(monkeypatch):
    """The min-latency block replaces Llama4MoE's routed experts with its own
    layer, which has to be built for the same per-layer format. The generic
    experts the base class builds are dropped by that replacement, so they are
    built without this layer's identity."""
    import tensorrt_llm._torch.distributed as distributed
    from tensorrt_llm._torch.models import modeling_llama, modeling_llama_min_latency

    for module, names in (
        (modeling_llama, ("create_moe", "Linear", "GatedMLP")),
        (
            modeling_llama_min_latency,
            ("Llama4MinLatencyGatedMLP", "Llama4MinLatencyFusedMoE", "Llama4MinLatencyLinear"),
        ),
    ):
        for name in names:
            monkeypatch.setattr(module, name, MagicMock())
    monkeypatch.setattr(distributed, "AllReduce", MagicMock())
    monkeypatch.setattr(torch.cuda, "Event", MagicMock())
    nvfp4 = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    model_config = ModelConfig(
        pretrained_config=SimpleNamespace(torch_dtype=torch.float32),
        quant_config=QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION),
        quant_config_dict={"model.layers.3.feed_forward.experts.0.gate_proj": nvfp4},
    )

    modeling_llama_min_latency.Llama4MinLatencyMoE(
        num_experts=4,
        top_k=1,
        hidden_size=8,
        intermediate_size=16,
        shared_expert_intermediate_size=16,
        aux_stream=None,
        dtype=torch.float32,
        model_config=model_config,
        layer_idx=3,
    )

    base_experts = modeling_llama.create_moe.call_args.kwargs
    assert base_experts["layer_idx"] is None
    assert base_experts["override_quant_config"] is None
    fused_moe = modeling_llama_min_latency.Llama4MinLatencyFusedMoE
    assert fused_moe.call_args.kwargs["override_quant_config"] is nvfp4


def test_gpt_oss_translates_exclusions_before_building_its_layers(monkeypatch):
    """GPT-OSS checkpoints name modules ``block.N...``; the experts resolve
    their exclusion while they are built, so the patterns are translated first."""
    from tensorrt_llm._torch.models import modeling_gpt_oss

    class _Built(Exception):
        pass

    seen = []

    def transformer(model_config):
        seen.append(list(model_config.quant_config.exclude_modules))
        raise _Built

    monkeypatch.setattr(modeling_gpt_oss, "Transformer", transformer)
    model_config = ModelConfig(
        pretrained_config=SimpleNamespace(torch_dtype=torch.bfloat16),
        quant_config=QuantConfig(
            quant_algo=QuantAlgo.W4A8_MXFP4_FP8,
            exclude_modules=["block.0.mlp.experts", "block.*.attn.qkv"],
        ),
    )

    with pytest.raises(_Built):
        modeling_gpt_oss.GptOssForCausalLM(model_config)

    assert seen == [["model.block.0.mlp.experts", "model.block.*.attn.qkv_proj"]]


def test_gpt_oss_excluded_experts_are_built_unquantized(monkeypatch):
    """An excluded experts module resolves to the unquantized format at
    construction, matching what ``__post_init__`` applies afterwards."""
    from tensorrt_llm._torch.models import modeling_gpt_oss

    for name in ("create_moe", "Linear", "RMSNorm"):
        monkeypatch.setattr(modeling_gpt_oss, name, MagicMock())
    model_config = ModelConfig(
        pretrained_config=SimpleNamespace(
            hidden_size=8,
            intermediate_size=16,
            num_local_experts=4,
            num_experts_per_tok=2,
            rms_norm_eps=1e-5,
            torch_dtype=torch.bfloat16,
        ),
        quant_config=QuantConfig(
            quant_algo=QuantAlgo.W4A8_MXFP4_FP8,
            kv_cache_quant_algo=QuantAlgo.FP8,
            exclude_modules=["model.block.0.mlp.experts"],
        ),
    )

    modeling_gpt_oss.MLPBlock(model_config, layer_idx=0)
    excluded = modeling_gpt_oss.create_moe.call_args.kwargs["override_quant_config"]
    assert excluded.quant_algo is None
    assert excluded.kv_cache_quant_algo == QuantAlgo.FP8

    modeling_gpt_oss.MLPBlock(model_config, layer_idx=1)
    assert modeling_gpt_oss.create_moe.call_args.kwargs["override_quant_config"] is None


def test_gpt_oss_keeps_a_regex_experts_exclusion_a_regex(monkeypatch):
    """A ``re:`` exclusion is left alone: its body spells dots its own way and
    the marker has to stay in front, or ``QuantConfig`` reads the entry as a
    glob and the experts are no longer excluded."""
    from tensorrt_llm._torch.models import modeling_gpt_oss

    class _Built(Exception):
        pass

    def transformer(model_config):
        raise _Built

    monkeypatch.setattr(modeling_gpt_oss, "Transformer", transformer)
    for name in ("create_moe", "Linear", "RMSNorm"):
        monkeypatch.setattr(modeling_gpt_oss, name, MagicMock())
    experts_exclude = r"re:model[.]block[.]0[.]mlp[.]experts"
    model_config = ModelConfig(
        pretrained_config=SimpleNamespace(
            hidden_size=8,
            intermediate_size=16,
            num_local_experts=4,
            num_experts_per_tok=2,
            rms_norm_eps=1e-5,
            torch_dtype=torch.bfloat16,
        ),
        quant_config=QuantConfig(
            quant_algo=QuantAlgo.W4A8_MXFP4_FP8,
            kv_cache_quant_algo=QuantAlgo.FP8,
            exclude_modules=[experts_exclude, "block.*.attn.qkv"],
        ),
    )

    with pytest.raises(_Built):
        modeling_gpt_oss.GptOssForCausalLM(model_config)

    # The regex survives verbatim; the glob entry is still translated.
    assert model_config.quant_config.exclude_modules == [
        experts_exclude,
        "model.block.*.attn.qkv_proj",
    ]
    assert model_config.quant_config.is_module_excluded_from_quantization(
        "model.block.0.mlp.experts"
    )

    # ... so the excluded experts layer is built unquantized.
    modeling_gpt_oss.MLPBlock(model_config, layer_idx=0)
    excluded = modeling_gpt_oss.create_moe.call_args.kwargs["override_quant_config"]
    assert excluded.quant_algo is None
    assert excluded.kv_cache_quant_algo == QuantAlgo.FP8

    modeling_gpt_oss.MLPBlock(model_config, layer_idx=1)
    assert modeling_gpt_oss.create_moe.call_args.kwargs["override_quant_config"] is None


_HOPPER_FP8_BLOCK_SCALES_ID = "deepgemm.cuda.hopper_grouped_gemm.fp8_block_scales"
_TRITON_FP8_BLOCK_SCALES_ID = "trtllm.triton.grouped_gemm.fp8_block_scales"


@pytest.mark.parametrize(
    "sm, impl_id, activation, rejection",
    [
        pytest.param(
            90,
            _HOPPER_FP8_BLOCK_SCALES_ID,
            SwigluActivation(clamp=5.0, clamp_after_silu=True),
            None,
            id="hopper-post-silu-clamp",
        ),
        pytest.param(
            120, _TRITON_FP8_BLOCK_SCALES_ID, SwigluActivation(), None, id="triton-plain-swiglu"
        ),
        pytest.param(
            120, _TRITON_FP8_BLOCK_SCALES_ID, SwigluActivation(clamp=5.0), None, id="triton-clamp"
        ),
        pytest.param(
            120,
            _TRITON_FP8_BLOCK_SCALES_ID,
            SwigluActivation(clamp=5.0, clamp_after_silu=True),
            "post-SiLU clamping",
            id="triton-post-silu-clamp",
        ),
    ],
)
def test_fp8_block_scales_leaves_admit_only_what_their_kernel_runs(
    sm, impl_id, activation, rejection
):
    """The SM90 leaf issues the CUTLASS op, which takes every activation constant;
    the SM120 Triton kernel runs gated SiLU / GELU and clamps before the SiLU."""
    with override_moe_environment(MoEEnvironment(sm=sm)):
        report = resolve_moe_impl(
            _cutlass_model_config(QuantAlgo.FP8_BLOCK_SCALES),
            activation=activation,
            impl_id=impl_id,
            dtype=torch.bfloat16,
        )

    if rejection is None:
        assert impl_class_for(report).descriptor.identity.canonical() == impl_id
    else:
        assert report.winner is None
        assert report.rejected[-1].reason is MoERejectReason.ACTIVATION_UNSUPPORTED
        assert rejection in report.rejected[-1].detail


@pytest.mark.parametrize(
    "activation, clamp_after_silu, constants, detail",
    [
        pytest.param(
            "Swiglu", True, frozenset({"clamp"}), "post-SiLU clamping", id="post_silu_clamp"
        ),
        pytest.param(
            "Relu2",
            False,
            frozenset(),
            "does not execute Relu2 (executes: Geglu, Swiglu)",
            id="relu2",
        ),
    ],
)
def test_triton_fp8_block_scales_leaf_declines_what_its_kernel_lacks(
    activation, clamp_after_silu, constants, detail
):
    """Declined by the leaf itself, as resolution declines it, and not blamed on
    the build: the Triton kernel never enters the CUTLASS op."""
    problem = MoEProblem(
        quant=canonical_quant(QuantAlgo.FP8_BLOCK_SCALES),
        dtype_act=torch.bfloat16,
        activation=activation,
        activation_constants=constants,
        clamp_after_silu=clamp_after_silu,
    )
    verdict = TrtllmTritonFp8BlockScalesImpl.can_implement(
        problem, _single_rank_deployment(MoEEnvironment(sm=120, oss_cutlass_moe=False))
    )

    assert verdict.reject_reason is MoERejectReason.ACTIVATION_UNSUPPORTED
    assert detail in verdict.detail
    assert "USING_OSS_CUTLASS_MOE_GEMM" not in verdict.detail


def test_only_leaves_issuing_the_op_are_bound_by_the_oss_build():
    """The internal kernel archive limits ``torch.ops.trtllm.fused_moe``; the
    Triton leaf never calls it, so an internal build keeps its all-to-all."""
    with override_moe_environment(MoEEnvironment(sm=120, oss_cutlass_moe=False)):
        triton_leaf = TrtllmTritonFp8BlockScalesImpl.__new__(TrtllmTritonFp8BlockScalesImpl)
        nvfp4_leaf = TrtllmCutlassNvfp4Impl.__new__(TrtllmCutlassNvfp4Impl)
        assert triton_leaf.supports_alltoall
        assert not nvfp4_leaf.supports_alltoall


def test_cutlass_build_gate_does_not_answer_for_an_unsupported_sm():
    """An internal build has no MXFP8 kernel, but no build has one for SM90:
    the leaf names the SM, so nobody rebuilds for a GPU that cannot run it."""
    problem = MoEProblem(quant=canonical_quant(QuantAlgo.MXFP8), dtype_act=torch.bfloat16)
    verdict = TrtllmCutlassMxfp8Impl.can_implement(
        problem, _single_rank_deployment(MoEEnvironment(sm=90, oss_cutlass_moe=False))
    )

    assert verdict.reject_reason is MoERejectReason.SM_UNSUPPORTED
    assert "USING_OSS_CUTLASS_MOE_GEMM" not in verdict.detail


@pytest.mark.parametrize(
    "leaf, problem, deployment, reason",
    [
        pytest.param(
            TrtllmCutlassNvfp4Impl,
            # FC1 rows 2 * 1000 are not a multiple of 128.
            MoEProblem(
                quant=canonical_quant(QuantAlgo.NVFP4),
                dtype_act=torch.bfloat16,
                intermediate_size=1000,
                clamp_after_silu=True,
            ),
            _single_rank_deployment(MoEEnvironment(sm=100, oss_cutlass_moe=False)),
            MoERejectReason.SHAPE_UNALIGNED,
            id="cutlass-unaligned-fc1",
        ),
        pytest.param(
            CuteDslB12xNvfp4FusedMoE,
            MoEProblem(
                quant=canonical_quant(QuantAlgo.NVFP4), dtype_act=torch.bfloat16, activation="Relu2"
            ),
            MoEDeployment(
                ep_size=2,
                tp_size=1,
                parallel_size=2,
                use_dp=False,
                num_slots=8,
                env=MoEEnvironment(
                    sm=120, oss_cutlass_moe=False, available_deps=(MoEDep.FLASHINFER,)
                ),
            ),
            MoERejectReason.TOPOLOGY_UNSUPPORTED,
            id="b12x-ep2",
        ),
    ],
)
def test_build_gate_runs_after_the_leaf_gates(leaf, problem, deployment, reason):
    """A problem the leaf turns down on every build is not blamed on this one."""
    verdict = leaf.can_implement(problem, deployment)

    assert verdict.reject_reason is reason
    assert "USING_OSS_CUTLASS_MOE_GEMM" not in verdict.detail


# =====================================================================
# CuteDSL grouped GEMM: one leaf per format
# =====================================================================
# The CUTEDSL literal picks a grouped-GEMM leaf by format; the leaf is then
# fixed for the layer's lifetime.

_CUTEDSL_GROUPED_GEMM_IDS = (
    "trtllm.cutedsl.grouped_gemm.nvfp4",
    "trtllm.cutedsl.grouped_gemm.none",
)


def _cutedsl_environment(sm: int = 107) -> MoEEnvironment:
    """``sm``, with CuTe DSL's Rubin support present on SM107."""
    deps = (MoEDep.CUTEDSL_RUBIN.value,) if sm == 107 else ()
    return MoEEnvironment(sm=sm, available_deps=deps)


def _cutedsl_model_config(quant_algo: QuantAlgo | None = QuantAlgo.NVFP4) -> ModelConfig:
    cfg = ModelConfig()
    cfg.moe_backend = "CUTEDSL"
    cfg.quant_config = QuantConfig(quant_algo=quant_algo) if quant_algo else None
    return cfg


# Every CuteDSL grouped-GEMM leaf's declarations, written out by hand rather
# than read off the leaves, so one that drifts fails here: the format it
# serves, the SMs its gate admits (probed over SM75-124 with CuTe DSL's Rubin
# support present), its static capabilities, and the activations and constants
# it declares.
_CUTEDSL_DECLARATIONS = (
    pytest.param(
        TrtllmCutedslGroupedGemmNvfp4Impl,
        QuantAlgo.NVFP4,
        {100, 103, 107},
        MoEStaticCapability(
            supports_dwdp=True,
            supports_eplb=True,
            supports_apply_router_weight_on_input=True,
            supports_deep_ep_direct_metadata=True,
        ),
        MoEActivationSupport(
            kinds=frozenset({ActivationType.Swiglu, ActivationType.Relu2, ActivationType.SiTu}),
            alpha_beta=ActivationParamShape.UNIFORM_SCALAR,
            limit=ActivationParamShape.UNIFORM_SCALAR,
            limit_when_absent=float("inf"),
        ),
        id="nvfp4",
    ),
    pytest.param(
        TrtllmCutedslGroupedGemmBf16Impl,
        None,
        {107},
        MoEStaticCapability(supports_eplb=True, supports_apply_router_weight_on_input=True),
        MoEActivationSupport(kinds=frozenset({ActivationType.Swiglu})),
        id="none",
    ),
)


def test_cutedsl_declarations_cover_every_leaf():
    """Every registered grouped-GEMM leaf is in the table and ranked by resolution."""
    leaves = {case.values[0] for case in _CUTEDSL_DECLARATIONS}
    assert leaves == {impl for _, impl in MOE_IMPL_REGISTRY.find(GROUPED_GEMM_FAMILY)}
    assert leaves <= set(IMPL_PRIORITY)


@pytest.mark.parametrize(
    "leaf, quant_algo, sms, capabilities, activation_support", _CUTEDSL_DECLARATIONS
)
def test_cutedsl_leaf_declarations(leaf, quant_algo, sms, capabilities, activation_support):
    problem = MoEProblem(quant=canonical_quant(quant_algo), dtype_act=torch.bfloat16)
    rubin = (MoEDep.CUTEDSL_RUBIN.value,)
    admitted = {
        sm
        for sm in range(75, 125)
        if leaf.can_implement(
            problem, _single_rank_deployment(MoEEnvironment(sm=sm, available_deps=rubin))
        ).eligible
    }
    assert admitted == set(sms)
    assert leaf.capabilities == capabilities
    assert leaf.activation_support == activation_support

    # Taken off the leaf's own descriptor, which the registry returns it for.
    assert "descriptor" in vars(leaf)
    assert MOE_IMPL_REGISTRY.lookup(leaf.descriptor.identity) is leaf
    assert leaf.scheduler_kind is leaf.descriptor.scheduler_kind
    assert leaf.capabilities is leaf.descriptor.capabilities
    assert leaf.input_requirement is leaf.descriptor.input_requirement
    # The root implements none of the four, so the class that publishes the
    # identity is the one that executes it.
    for name in _ABSTRACT_METHODS:
        assert name in vars(leaf)
    # Concrete, and under the family name create_moe_backend dispatches on,
    # not under CUTLASS's, whose branch is tried first and passes ``bias``.
    assert not leaf.__abstractmethods__
    assert issubclass(leaf, CuteDslFusedMoE)
    assert not issubclass(leaf, CutlassFusedMoEBase)


@pytest.mark.parametrize(
    "quant_algo, sm, impl_id",
    [
        pytest.param(QuantAlgo.NVFP4, 100, _CUTEDSL_GROUPED_GEMM_IDS[0], id="nvfp4-sm100"),
        pytest.param(None, 107, _CUTEDSL_GROUPED_GEMM_IDS[1], id="none-sm107"),
    ],
)
def test_cutedsl_literal_picks_the_leaf_that_publishes_the_format(quant_algo, sm, impl_id):
    """``moe_backend: CUTEDSL`` names the family; the quant picks which leaf."""
    with override_moe_environment(_cutedsl_environment(sm)):
        report = resolve_moe_impl(_cutedsl_model_config(quant_algo))
    assert impl_class_for(report).descriptor.identity.canonical() == impl_id
    assert not report.degraded


@pytest.mark.parametrize(
    "quant_algo, impl_id",
    [
        pytest.param(QuantAlgo.NVFP4, _CUTEDSL_GROUPED_GEMM_IDS[0], id="nvfp4"),
        pytest.param(None, _CUTEDSL_GROUPED_GEMM_IDS[1], id="none"),
    ],
)
def test_pinned_cutedsl_grouped_gemm_identity_resolves_to_its_leaf(quant_algo, impl_id):
    with override_moe_environment(_cutedsl_environment()):
        report = resolve_moe_impl(_cutedsl_model_config(quant_algo), impl_id=impl_id)
    assert impl_class_for(report).descriptor.identity.canonical() == impl_id
    assert report.selected_by == "pinned"


def test_a_partial_grouped_gemm_pin_resolves_by_priority_across_techniques():
    """``grouped_gemm.nvfp4`` names the CUTLASS and the CuteDSL leaf; ``IMPL_PRIORITY`` picks."""
    with override_moe_environment(_cutedsl_environment(100)):
        report = resolve_moe_impl(_cutedsl_model_config(), impl_id="grouped_gemm.nvfp4")
    assert impl_class_for(report) is TrtllmCutedslGroupedGemmNvfp4Impl
    assert report.selected_by == "pinned"


@pytest.mark.parametrize(
    "problem",
    [
        MoEProblem(quant=None, dtype_act=torch.bfloat16, activation="Relu2"),
        MoEProblem(
            quant=None,
            dtype_act=torch.bfloat16,
            activation="Swiglu",
            activation_constants=frozenset({"clamp"}),
        ),
    ],
    ids=["relu2", "clamped_swiglu"],
)
def test_cutedsl_bf16_leaf_declines_another_activation_when_asked_directly(problem):
    verdict = TrtllmCutedslGroupedGemmBf16Impl.can_implement(
        problem, _single_rank_deployment(_cutedsl_environment())
    )
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.ACTIVATION_UNSUPPORTED


def test_a_clamped_bf16_swiglu_leaves_the_cutedsl_literal_for_cutlass_and_fails_a_pin():
    """The literal degrades to the CUTLASS leaf, whose epilogue applies the clamp."""
    config = _cutedsl_model_config(None)
    activation = SwigluActivation(clamp=7.0)
    with override_moe_environment(_cutedsl_environment()):
        by_literal = resolve_moe_impl(config, activation=activation)
        by_identity = resolve_moe_impl(
            config, activation=activation, impl_id=_CUTEDSL_GROUPED_GEMM_IDS[1]
        )

    assert impl_class_for(by_literal) is TrtllmCutlassUnquantizedImpl
    assert by_literal.degraded

    assert by_identity.winner is None
    assert [rejection.reason for rejection in by_identity.rejected] == [
        MoERejectReason.ACTIVATION_UNSUPPORTED
    ]
    with pytest.raises(ValueError, match="no MoE implementation can serve"):
        impl_class_for(by_identity)


# =====================================================================
# One class per identity
# =====================================================================
# Each single-format identity is declared on the class that executes it: one
# class owns the descriptor and all four abstract methods, and the pre-identity
# name survives only as a module-level alias. So both request tracks and every
# legacy call site land on that one class, and a run reports one name. The
# multi-format families split the alias off onto an abstract parent instead,
# and have their own sections above.

_DENSEGEMM_IMPL_ID = "trtllm.cutedsl.dense_gemm.nvfp4"

# The legacy alias, the class that carries the identity, the id it publishes,
# and the scheduler kind that class declares.
_SINGLE_FORMAT_IMPLS = (
    pytest.param(
        DeepGemmFusedMoE,
        DeepgemmCudaFp8BlockScalesImpl,
        _DEEPGEMM_IMPL_ID,
        MoESchedulerKind.EXTERNAL_COMM,
        id="grouped_gemm",
    ),
    pytest.param(
        MegaMoEDeepGemm,
        DeepgemmCudaW4a8Mxfp4Mxfp8Impl,
        _MEGAMOE_DEEPGEMM_IMPL_ID,
        MoESchedulerKind.FUSED_COMM,
        id="mega_moe",
    ),
    pytest.param(
        MegaMoECuteDsl,
        TrtllmCutedslMegaMoeNvfp4Impl,
        _MEGAMOE_CUTEDSL_IMPL_ID,
        MoESchedulerKind.FUSED_COMM,
        id="mega_moe_cutedsl",
    ),
    pytest.param(
        DenseGEMMFusedMoE,
        TrtllmCutedslDenseGemmNvfp4Impl,
        _DENSEGEMM_IMPL_ID,
        MoESchedulerKind.EXTERNAL_COMM,
        id="dense_gemm",
    ),
)

# Every class resolution can return is one create_moe_backend has to build.
_DISPATCHED_IMPLS = tuple(pytest.param(impl, id=impl.__name__) for impl in IMPL_PRIORITY)


@pytest.mark.parametrize("legacy, impl, impl_id, scheduler_kind", _SINGLE_FORMAT_IMPLS)
def test_the_identity_and_the_implementation_sit_on_one_class(
    legacy, impl, impl_id, scheduler_kind
):
    """What is published and what executes are declared together, so neither drifts."""
    assert "descriptor" in vars(impl)
    for name in _ABSTRACT_METHODS:
        assert name in vars(impl)
    # Nothing left abstract: the class the tables name is the class that runs.
    assert not impl.__abstractmethods__

    # The published id, and the round trip back to this same class.
    identity = impl.descriptor.identity
    assert identity.canonical() == impl_id
    assert MoEImplId.parse(impl_id) == identity
    assert MOE_IMPL_REGISTRY.lookup(identity) is impl

    # An alias, not a base class: reintroducing a parent under the old name
    # splits the kernel into a name that resolves and one that does not.
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

    ``_legacy_backend_name`` answers with ``__name__``, so this also pins that
    the reported name is the identity-derived one on both tracks.
    """
    with override_moe_environment(_deepgemm_environment()):
        by_literal = resolve_moe_impl(_deepgemm_model_config())
        by_pin = resolve_moe_impl(_deepgemm_model_config(), impl_id=_DEEPGEMM_IMPL_ID)

    assert impl_class_for(by_literal) is DeepgemmCudaFp8BlockScalesImpl
    assert impl_class_for(by_pin) is impl_class_for(by_literal)
    assert by_literal.winner == by_pin.winner == "DeepgemmCudaFp8BlockScalesImpl"


@pytest.mark.parametrize("impl", _DISPATCHED_IMPLS)
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
# wrong. A query locates each token's field by value, never by position, so a
# user can type the one token they remember; registration keeps the four
# fields' value sets disjoint so that stays unambiguous.


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


def test_an_identity_reusing_a_token_across_its_own_fields_is_rejected():
    """The four fields' value sets have to stay disjoint inside one identity, too.

    Without it such an id registers cleanly and is then unaddressable: the later
    field wins the token, so ``parse_query`` gives both segments to that one
    field and refuses the id's own canonical string -- including when the pin
    arrives as a ``MoEImplId`` rather than as text.

    Uses a private registry because the failure is a registration-time one,
    while the global registry is shared with the rest of this file.
    """
    registry = MoEImplRegistry()

    class _SelfCollidingImpl:
        descriptor = MoEImplDescriptor(
            identity=MoEImplId("torch", "torch", "vanilla", "none"),
            scheduler_kind=MoESchedulerKind.EXTERNAL_COMM,
        )

    with pytest.raises(ValueError, match=r"token 'torch' is already a value of field 'provider'"):
        registry.register(_SelfCollidingImpl)


def _only_format_mismatches(report: MoEResolutionReport, served_by: type) -> bool:
    """True when every leaf but ``served_by`` declined only on format."""
    return all(
        r.reason is MoERejectReason.QUANT_UNSUPPORTED
        for r in report.rejected
        if r.legacy_backend != served_by.__name__
    )


def _rejection_of(report: MoEResolutionReport, impl: type) -> MoERejection:
    return next(r for r in report.rejected if r.legacy_backend == impl.__name__)


@pytest.mark.parametrize("is_oss", [False, True], ids=["internal", "oss"])
@pytest.mark.parametrize(
    "activation,quant_algo,sm,leaf,reason",
    [
        pytest.param(
            "SiTu",
            None,
            90,
            TrtllmCutlassUnquantizedImpl,
            MoERejectReason.ACTIVATION_UNSUPPORTED,
            id="situ",
        ),
        pytest.param(
            "Relu2",
            None,
            90,
            TrtllmCutlassUnquantizedImpl,
            MoERejectReason.ACTIVATION_UNSUPPORTED,
            id="relu2",
        ),
        pytest.param(
            "Swiglu",
            QuantAlgo.W4A16_MXFP4,
            90,
            TrtllmCutlassW4a16Mxfp4Impl,
            MoERejectReason.QUANT_UNSUPPORTED,
            id="w4a16_mxfp4",
        ),
        pytest.param(
            "Swiglu",
            QuantAlgo.W4A8_AWQ,
            90,
            TrtllmCutlassW4a8AwqImpl,
            MoERejectReason.QUANT_UNSUPPORTED,
            id="w4a8_awq",
        ),
        pytest.param(
            "Swiglu",
            QuantAlgo.MXFP8,
            100,
            TrtllmCutlassMxfp8Impl,
            MoERejectReason.QUANT_UNSUPPORTED,
            id="mxfp8",
        ),
        pytest.param(
            "Swiglu",
            QuantAlgo.W4A8_MXFP4_MXFP8,
            100,
            TrtllmCutlassW4a8Mxfp4Mxfp8Impl,
            MoERejectReason.QUANT_UNSUPPORTED,
            id="w4a8_mxfp4_mxfp8",
        ),
    ],
)
def test_cutlass_build_mode_resolution(is_oss, activation, quant_algo, sm, leaf, reason):
    report = resolve_moe_impl(
        ModelConfig(moe_backend="CUTLASS"),
        problem=MoEProblem(
            quant=quant_algo.value if quant_algo is not None else None,
            dtype_act=torch.bfloat16,
            hidden_size=512,
            intermediate_size=512,
            num_experts=8,
            top_k=2,
            activation=activation,
        ),
        deployment=MoEDeployment(
            ep_size=1,
            tp_size=1,
            parallel_size=1,
            use_dp=False,
            num_slots=8,
            env=MoEEnvironment(sm=sm, oss_cutlass_moe=is_oss),
        ),
    )
    assert _only_format_mismatches(report, served_by=leaf)
    if is_oss:
        assert impl_class_for(report) is leaf
    else:
        assert report.winner is None
        rejection = _rejection_of(report, leaf)
        assert rejection.reason is reason
        assert "USING_OSS_CUTLASS_MOE_GEMM=ON" in rejection.detail
        with pytest.raises(ValueError, match="USING_OSS_CUTLASS_MOE_GEMM=ON"):
            impl_class_for(report)


@pytest.mark.parametrize("is_oss", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_cutlass_dynamic_scale_resolution(is_oss: bool, dynamic: bool) -> None:
    from tensorrt_llm._torch.moe.fused_moe.moe_resolution import build_moe_deployment

    deployment = build_moe_deployment(
        ModelConfig(force_dynamic_quantization=dynamic),
        num_experts=8,
        environment=MoEEnvironment(sm=100, oss_cutlass_moe=is_oss),
    )
    assert deployment.force_dynamic_quantization is dynamic
    report = resolve_moe_impl(
        ModelConfig(moe_backend="CUTLASS"),
        problem=MoEProblem(
            quant=QuantAlgo.NVFP4.value,
            dtype_act=torch.bfloat16,
            hidden_size=512,
            intermediate_size=512,
            num_experts=8,
            top_k=2,
        ),
        deployment=deployment,
    )
    assert _only_format_mismatches(report, served_by=TrtllmCutlassNvfp4Impl)
    if dynamic and not is_oss:
        assert report.winner is None
        rejection = _rejection_of(report, TrtllmCutlassNvfp4Impl)
        assert rejection.reason is MoERejectReason.QUANT_UNSUPPORTED
        assert "dynamic FC2 scaling" in rejection.detail
    else:
        assert impl_class_for(report) is TrtllmCutlassNvfp4Impl


@pytest.mark.parametrize("is_oss", [False, True])
@pytest.mark.parametrize("alltoall", [False, True])
def test_cutlass_communication_validation(is_oss: bool, alltoall: bool) -> None:
    from types import SimpleNamespace

    backend = TrtllmCutlassUnquantizedImpl.__new__(TrtllmCutlassUnquantizedImpl)
    moe = SimpleNamespace(enable_alltoall=alltoall)
    with override_moe_environment(MoEEnvironment(sm=90, oss_cutlass_moe=is_oss)):
        if alltoall and not is_oss:
            with pytest.raises(ValueError, match="all-to-all.*USING_OSS_CUTLASS_MOE_GEMM=ON"):
                backend.validate_configurable_moe(moe)
        else:
            backend.validate_configurable_moe(moe)


@pytest.mark.parametrize("mode", [None, False, True])
def test_cutlass_build_mode_probe(monkeypatch: pytest.MonkeyPatch, mode: bool | None) -> None:
    from types import SimpleNamespace

    from tensorrt_llm._torch.moe.fused_moe import impl_environment

    query = MagicMock(return_value=mode)
    debug = MagicMock()
    monkeypatch.setattr(impl_environment.logger, "debug", debug)
    namespace = SimpleNamespace() if mode is None else SimpleNamespace(is_oss_cutlass_moe=query)
    monkeypatch.setattr(torch.ops, "trtllm", namespace)
    impl_environment.reset_moe_environment_cache()
    try:
        expected = True if mode is None else mode
        assert impl_environment._is_oss_cutlass_moe() is expected
        assert impl_environment._is_oss_cutlass_moe() is expected
        assert query.call_count == (0 if mode is None else 1)
        if mode is None:
            debug.assert_called_once_with(
                "CUTLASS MoE build-mode query is unavailable; assuming OSS kernels"
            )
        else:
            debug.assert_not_called()
    finally:
        impl_environment.reset_moe_environment_cache()


def test_cutlass_build_mode_fingerprint() -> None:
    assert (
        MoEEnvironment(sm=90).fingerprint()
        != MoEEnvironment(sm=90, oss_cutlass_moe=False).fingerprint()
    )


@pytest.mark.parametrize("is_oss", [False, True])
@pytest.mark.parametrize("method", [None, "ALLGATHER", "NVLINK_ONE_SIDED"])
@pytest.mark.parametrize("force_source", ["model", "env"])
def test_cutlass_communication_selection(monkeypatch, is_oss, method, force_source):
    from types import SimpleNamespace

    from tensorrt_llm._torch.moe.fused_moe.communication import communication_factory
    from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE

    monkeypatch.delenv("TRTLLM_FORCE_COMM_METHOD", raising=False)
    if force_source == "env" and method is not None:
        monkeypatch.setenv("TRTLLM_FORCE_COMM_METHOD", method)
    mapping = SimpleNamespace(
        enable_attention_dp=True,
        dp_size=2,
        moe_tp_size=1,
        moe_ep_size=2,
        has_cp_helix=lambda: False,
    )
    config = SimpleNamespace(
        mapping=mapping,
        pretrained_config=SimpleNamespace(hidden_size=128),
        torch_dtype=torch.bfloat16,
        quant_config=None,
        max_num_tokens=8,
        moe_max_num_tokens=8,
        use_cuda_graph=False,
        use_low_precision_moe_combine=False,
        moe_load_balancer=None,
    )
    constructors = {}
    for name in (
        "AllGatherReduceScatter",
        "NVLinkOneSided",
        "NVLinkTwoSided",
        "NVLinkTwoSidedFlashinfer",
        "NcclEP",
        "DeepEP",
        "DeepEPLowLatency",
    ):
        constructors[name] = MagicMock(name=name)
        monkeypatch.setattr(communication_factory, name, constructors[name])
    monkeypatch.setattr(communication_factory, "get_wide_ep_ft_options", lambda _: (None, 60, 1))
    moe = SimpleNamespace(
        backend=TrtllmCutlassUnquantizedImpl.__new__(TrtllmCutlassUnquantizedImpl),
        enable_dwdp=False,
        model_config=config,
        num_experts=4,
        num_slots=4,
        routing_method=SimpleNamespace(experts_per_token=1),
        expert_size_per_partition=2,
        use_flashinfer=False,
        hidden_size=128,
        communication_method=method if force_source == "model" else None,
    )
    with override_moe_environment(MoEEnvironment(sm=90, oss_cutlass_moe=is_oss)):
        if not is_oss and method == "NVLINK_ONE_SIDED":
            with pytest.raises(ValueError, match="requires all-to-all.*use ALLGATHER"):
                ConfigurableMoE._create_comm_strategy_auto(moe)
            assert all(constructor.call_count == 0 for constructor in constructors.values())
        else:
            strategy = ConfigurableMoE._create_comm_strategy_auto(moe)
            selected = (
                "AllGatherReduceScatter"
                if not is_oss or method == "ALLGATHER"
                else "NVLinkOneSided"
            )
            assert strategy is constructors[selected].return_value
            constructors[selected].assert_called_once()
            assert all(
                constructor.call_count == 0
                for name, constructor in constructors.items()
                if name != selected
            )


@pytest.mark.parametrize("is_oss", [False, True])
@pytest.mark.parametrize("sm", [120, 121])
@pytest.mark.parametrize("quant_algo", [QuantAlgo.NVFP4, QuantAlgo.W4A16_NVFP4])
@pytest.mark.parametrize("activation", ["Relu2", "Swiglu"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_b12x_cutlass_build_mode_resolution(is_oss, sm, quant_algo, activation, dynamic):
    leaf = {
        QuantAlgo.NVFP4: CuteDslB12xNvfp4FusedMoE,
        QuantAlgo.W4A16_NVFP4: CuteDslB12xW4a16Nvfp4FusedMoE,
    }[quant_algo]
    report = resolve_moe_impl(
        ModelConfig(moe_backend="CUTEDSL"),
        problem=MoEProblem(
            quant=canonical_quant(quant_algo),
            dtype_act=torch.bfloat16,
            hidden_size=512,
            intermediate_size=512,
            num_experts=8,
            top_k=2,
            activation=activation,
        ),
        deployment=MoEDeployment(
            ep_size=1,
            tp_size=1,
            parallel_size=1,
            use_dp=False,
            num_slots=8,
            force_dynamic_quantization=dynamic,
            env=MoEEnvironment(sm=sm, oss_cutlass_moe=is_oss, available_deps=(MoEDep.FLASHINFER,)),
        ),
    )
    nvfp4 = quant_algo == QuantAlgo.NVFP4
    if nvfp4 and dynamic:
        # Declined ahead of the build gate, since decode cannot honour it on
        # any build; the CUTLASS fallback needs the open-source kernels for it.
        b12x_rejection = _rejection_of(report, leaf)
        assert b12x_rejection.reason is MoERejectReason.DYNAMIC_QUANT_UNSUPPORTED
        if is_oss:
            assert impl_class_for(report) is TrtllmCutlassNvfp4Impl
        else:
            assert report.winner is None
    elif nvfp4 and not is_oss and activation == "Relu2":
        assert report.winner is None
        b12x_rejection = _rejection_of(report, leaf)
        assert b12x_rejection.reason is MoERejectReason.ACTIVATION_UNSUPPORTED
        assert "USING_OSS_CUTLASS_MOE_GEMM=ON" in b12x_rejection.detail
        assert leaf.__name__ in b12x_rejection.detail
        assert b12x_rejection.detail.startswith("NVFP4 prefill routes through CUTLASS:")
    else:
        assert impl_class_for(report) is leaf


@pytest.mark.parametrize("is_oss", [False, True])
def test_b12x_lora_rejection_is_build_independent(is_oss):
    verdict = CuteDslB12xNvfp4FusedMoE.can_implement(
        MoEProblem(quant=canonical_quant(QuantAlgo.NVFP4), dtype_act=torch.bfloat16),
        MoEDeployment(
            ep_size=1,
            tp_size=1,
            parallel_size=1,
            use_dp=False,
            num_slots=8,
            moe_lora_enabled=True,
            env=MoEEnvironment(sm=120, oss_cutlass_moe=is_oss, available_deps=(MoEDep.FLASHINFER,)),
        ),
    )
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.LORA_UNSUPPORTED
    assert verdict.detail == "CuteDslB12xNvfp4FusedMoE does not fuse routed-expert LoRA"
