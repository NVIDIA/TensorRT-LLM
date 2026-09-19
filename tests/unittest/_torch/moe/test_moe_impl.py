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

from unittest.mock import MagicMock

import pytest
import torch
from _torch.moe.moe_test_utils import MoeBackendType

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.fused_moe.activation import SwigluActivation
from tensorrt_llm._torch.moe.fused_moe.create_moe import create_moe_backend
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl_fc12 import (
    TrtllmCutedslFusedFc12Nvfp4Impl,
)
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.moe.fused_moe.fused_moe_deepgemm import (
    DeepgemmCudaFp8BlockScalesImpl,
    DeepGemmFusedMoE,
)
from tensorrt_llm._torch.moe.fused_moe.fused_moe_trtllm_gen import TRTLLMGenFusedMoE
from tensorrt_llm._torch.moe.fused_moe.impl_contract import (
    MoEDeployment,
    MoEEnvironment,
    MoEProblem,
    MoERejectReason,
    canonical_quant,
    canonical_routing,
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
from tensorrt_llm._torch.moe.fused_moe.mega_moe import (
    DeepgemmCudaW4a8Mxfp4Mxfp8Impl,
    MegaMoEDeepGemm,
)
from tensorrt_llm._torch.moe.fused_moe.moe_resolution import impl_class_for, resolve_moe_impl
from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV4MoeRoutingMethod, RoutingMethodType
from tensorrt_llm._torch.moe.fused_moe.trtllm_gen import (
    FlashinferTrtllmGenBf16Impl,
    FlashinferTrtllmGenNvfp4Impl,
    TrtllmTrtllmGenNvfp4Impl,
    TrtllmTrtllmGenW4a8Nvfp4Fp8Impl,
    trtllm_gen_leaves_in_resolution_order,
)
from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.eligibility import check_flashinfer_provider
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

# Everything here is a declaration; the one test that reaches a constructor
# stubs ``__init__`` out. The marker is also what makes the file reachable: the
# CPU stage collects only files that carry it.
pytestmark = pytest.mark.cpu_only

# The four ``MoEImplBase`` methods a registered class has to define itself. A
# leaf defines all four, an abstract parent defines none.
_ABSTRACT_METHODS = ("can_implement", "_get_quant_method", "quantize_input", "run_moe")

# =====================================================================
# Canonical implementation identity
# =====================================================================
# The DeepGEMM FP8-block-scales identity, and the four ways a pinned request
# is refused rather than served by something else: every gate declines, an
# unknown token, an id that matches nothing registered, or one that constrains
# nothing and so would match everything.

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


@pytest.mark.parametrize("sm, eligible", [(90, True), (120, False)])
def test_cutlass_post_silu_clamp_rejects_triton_fallback(sm, eligible):
    problem = MoEProblem(
        quant=canonical_quant(QuantAlgo.FP8_BLOCK_SCALES),
        dtype_act=torch.bfloat16,
        activation_constants=frozenset({"clamp"}),
        clamp_after_silu=True,
    )
    deployment = MoEDeployment(
        ep_size=1,
        tp_size=1,
        use_dp=False,
        num_slots=8,
        env=MoEEnvironment(sm=sm),
        parallel_size=1,
    )

    verdict = CutlassFusedMoE.can_implement(problem, deployment)

    assert verdict.eligible is eligible
    if not eligible:
        assert verdict.reject_reason is MoERejectReason.ACTIVATION_UNSUPPORTED
        assert "Triton fallback" in verdict.detail


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
    """Built as a query rather than parsed, to get past the token vocabulary.

    ``fp8`` because plain per-tensor FP8 has no registered MoE implementation,
    which is what makes the query match nothing while staying well-formed
    (``nvfp4`` is registered by TrtllmCutedslFusedFc12Nvfp4Impl).
    """
    with override_moe_environment(_deepgemm_environment()):
        with pytest.raises(ValueError, match="matches no registered implementation"):
            resolve_moe_impl(_deepgemm_model_config(), impl_id=MoEImplQuery(quant="fp8"))


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


# =====================================================================
# MegaMoE DeepGEMM implementation identity
# =====================================================================
# The MegaMoE identity, where ``deepgemm`` as a bare provider token matches two
# leaves, so the disambiguation is exercised rather than assumed.
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


def test_pinned_kernel_token_alone_reaches_the_megamoe_leaf():
    """``mega_moe`` belongs to one leaf, so the kernel segment is enough."""
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
# TRTLLM-Gen implementation identity
# =====================================================================
# The one backend that does need the abstract parent the DeepGEMM section
# below does without: eleven identities share one ``__init__``, one
# ``create_weights`` and one input-preparation path. These tests pin the
# split -- each leaf declares its own descriptor and all four abstract
# methods, the parent declares neither, so no request can reach it.

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


def test_a_leaf_refuses_a_layer_whose_format_moved_after_it_was_picked():
    """Pinned because it is a behaviour change, not because it is desirable.

    Resolution keys on the model-level ``quant_algo``, and
    ``apply_layerwise_quant_config`` / ``apply_quant_config_exclude_modules``
    give a layer its own ``quant_config`` afterwards. A leaf *is* its format,
    so a checkpoint that used to re-derive its quant method and carry on now
    stops at load, and the message has to carry the diagnosis because the pass
    that moved the layer has already returned.

    ``__new__`` without ``__init__``: the check reads three attributes and no
    allocated state, and a real constructor would need a GPU.
    """
    moved = TrtllmTrtllmGenNvfp4Impl.__new__(TrtllmTrtllmGenNvfp4Impl)
    moved.layer_idx = 3
    moved.quant_config = QuantConfig(quant_algo=None)

    with pytest.raises(ValueError) as excinfo:
        moved._check_quant_config_is_my_format()
    message = str(excinfo.value)
    assert "layer 3" in message
    assert "module exclusion" in message

    kept = TrtllmTrtllmGenNvfp4Impl.__new__(TrtllmTrtllmGenNvfp4Impl)
    kept.layer_idx = 3
    kept.quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    kept._check_quant_config_is_my_format()


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
# One class per identity
# =====================================================================
# Each DeepGEMM identity is declared on the class that executes it: one class
# owns the descriptor and all four abstract methods, and the pre-identity name
# survives only as a module-level alias. So both request tracks and every
# legacy call site land on that one class, and a run reports one name.

# The legacy alias, the class that carries the identity, the id it publishes,
# and the scheduler kind that class declares.
_DEEPGEMM_IMPLS = (
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
)

_IMPLS_ONLY = tuple(pytest.param(case.values[1], id=case.id) for case in _DEEPGEMM_IMPLS)


@pytest.mark.parametrize("legacy, impl, impl_id, scheduler_kind", _DEEPGEMM_IMPLS)
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
