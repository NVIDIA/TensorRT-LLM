# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for selecting the MiniMax-M3.1 MegaMoE local leaf, registered from its model family
(``register_impl_ahead_of``, ``register_optional_env_flag``)."""

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.moe.megamoe_local import (
    local_impl as local_mod,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.moe.megamoe_local import (
    local_plan as plan,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.moe.megamoe_local.local_impl import (
    TrtllmCutedslMegaMoeLocalNvfp4Impl,
)
from tensorrt_llm._torch.moe.fused_moe.impl_contract import (
    MoEDeployment,
    MoEEnvironment,
    MoEProblem,
    MoERejectReason,
)
from tensorrt_llm._torch.moe.fused_moe.impl_environment import MoEDep
from tensorrt_llm._torch.moe.fused_moe.mega_moe import TrtllmCutedslMegaMoeNvfp4Impl
from tensorrt_llm._torch.moe.fused_moe.moe_resolution import (
    FALLBACK_IMPLS,
    impl_class_for,
    moe_impl_request,
    register_impl_ahead_of,
    resolve_moe_impl,
)
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm.models.modeling_utils import QuantAlgo

DISABLE = plan.DISABLE_ENV

pytestmark = pytest.mark.cpu_only


def _problem(num_experts: int = 128) -> MoEProblem:
    """A MiniMax-M3.1 layer that the stock CuteDSL leaf admits."""
    return MoEProblem(
        quant=QuantAlgo.NVFP4.value,
        dtype_act=torch.bfloat16,
        hidden_size=6144,
        intermediate_size=3072,
        num_experts=num_experts,
        top_k=4,
        swiglu_gptoss_style=True,
        bias=False,
        activation=ActivationType.SwigluBias.name,
        activation_constants=frozenset({"alpha", "beta", "clamp"}),
    )


def _deployment(
    *,
    ep_size: int = 8,
    num_slots: int = 128,
    eplb_enabled: bool = False,
    sm: int = 107,
    env_flags=(),
) -> MoEDeployment:
    return MoEDeployment(
        ep_size=ep_size,
        tp_size=1,
        parallel_size=ep_size,
        use_dp=True,
        num_slots=num_slots,
        eplb_enabled=eplb_enabled,
        env=MoEEnvironment(
            sm=sm,
            available_deps=(
                MoEDep.MEGAMOE_CUTEDSL_RUNTIME.value,
                MoEDep.MEGAMOE_CUTEDSL_OP.value,
            ),
            env_flags=tuple(sorted(env_flags)),
        ),
    )


def _model_config() -> ModelConfig:
    cfg = ModelConfig()
    cfg.moe_backend = "MEGAMOE_CUTEDSL"
    return cfg


def _resolve(deployment, problem=None, impl_id=None):
    return resolve_moe_impl(
        _model_config(),
        problem=_problem() if problem is None else problem,
        deployment=deployment,
        impl_id=impl_id,
    )


@pytest.fixture
def warnings(monkeypatch):
    seen = []
    monkeypatch.setattr(
        local_mod.logger,
        "warning_once",
        lambda *msg, key: seen.append((key, " ".join(str(m) for m in msg))),
    )
    return seen


_FAMILY_NAMES = {
    TrtllmCutedslMegaMoeLocalNvfp4Impl.__name__,
    TrtllmCutedslMegaMoeNvfp4Impl.__name__,
}


def _trail(report):
    """The MEGAMOE_CUTEDSL leaves' rejections; the fallback leaves' own are not this backend's."""
    return [
        (r.legacy_backend, r.reason) for r in report.rejected if r.legacy_backend in _FAMILY_NAMES
    ]


def _candidates_for(backend):
    request = moe_impl_request(backend)
    return list(request.impls) + list(request.fallback)


def test_importing_the_family_module_registers_the_leaf_once():
    assert local_mod.is_registered()
    assert local_mod.register()
    register_impl_ahead_of(
        TrtllmCutedslMegaMoeLocalNvfp4Impl,
        ahead_of=TrtllmCutedslMegaMoeNvfp4Impl,
        family="MEGAMOE_CUTEDSL",
    )
    candidates = _candidates_for("MEGAMOE_CUTEDSL")
    # Local, then stock, then the fallback leaves; registering again adds nothing.
    assert candidates[:2] == [TrtllmCutedslMegaMoeLocalNvfp4Impl, TrtllmCutedslMegaMoeNvfp4Impl]
    assert candidates[2:] == list(FALLBACK_IMPLS)


def test_a_second_class_with_a_registered_identity_is_rejected():
    from tensorrt_llm._torch.moe.fused_moe import moe_resolution

    class _Dup(TrtllmCutedslMegaMoeNvfp4Impl):  # inherits the stock leaf's descriptor
        pass

    with pytest.raises(ValueError, match="duplicate MoEImplId"):
        register_impl_ahead_of(
            _Dup, ahead_of=TrtllmCutedslMegaMoeNvfp4Impl, family="MEGAMOE_CUTEDSL"
        )
    assert _Dup not in moe_resolution.IMPL_PRIORITY
    assert _Dup not in moe_resolution.BACKEND_FAMILY["MEGAMOE_CUTEDSL"]


@pytest.mark.parametrize("ep_size", [8, 16], ids=["dep8", "dep16"])
def test_default_coarse_request_selects_the_local_leaf(warnings, ep_size):
    report = _resolve(_deployment(ep_size=ep_size))
    assert impl_class_for(report) is TrtllmCutedslMegaMoeLocalNvfp4Impl
    assert report.selected_by == "pinned"
    assert _trail(report) == []
    assert warnings == []


def test_the_kill_switch_resolves_the_coarse_request_to_stock(warnings):
    report = _resolve(_deployment(env_flags=((DISABLE, "1"),)))
    assert impl_class_for(report) is TrtllmCutedslMegaMoeNvfp4Impl
    assert report.selected_by == "pinned" and not report.degraded
    assert _trail(report) == [
        (TrtllmCutedslMegaMoeLocalNvfp4Impl.__name__, MoERejectReason.PATH_NOT_ENABLED)
    ]
    assert warnings == []


REJECTS = {
    # A registered load balancer is declined by the leaf's static capability before its own checks run, so only the
    # redundant-slot layout reaches the backend's warning; other SMs decline silently.
    "eplb": (dict(eplb_enabled=True), None, MoERejectReason.EPLB_UNSUPPORTED, []),
    "redundant_slots": (
        dict(num_slots=144),
        None,
        MoERejectReason.EPLB_UNSUPPORTED,
        ["megamoe_local_vd_eplb_unsupported"],
    ),
    "single_rank_ep": (
        dict(ep_size=1),
        None,
        MoERejectReason.TOPOLOGY_UNSUPPORTED,
        ["megamoe_local_vd_topology_unsupported"],
    ),
    "experts_not_split_by_2ep": (
        dict(num_slots=72),
        _problem(num_experts=72),
        MoERejectReason.TOPOLOGY_UNSUPPORTED,
        ["megamoe_local_vd_topology_unsupported"],
    ),
    "sm100": (dict(sm=100), None, MoERejectReason.SM_UNSUPPORTED, []),
}


@pytest.mark.parametrize("name", list(REJECTS))
def test_a_reject_resolves_the_coarse_request_to_stock(warnings, name):
    deployment_kwargs, problem, reason, expected_warnings = REJECTS[name]
    report = _resolve(_deployment(**deployment_kwargs), problem=problem)
    assert impl_class_for(report) is TrtllmCutedslMegaMoeNvfp4Impl
    assert _trail(report) == [(TrtllmCutedslMegaMoeLocalNvfp4Impl.__name__, reason)]
    assert [key for key, _ in warnings] == expected_warnings
