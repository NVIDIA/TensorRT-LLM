# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MiniMax-M3.1 DEP target's construction checks, on stub layers.

The target serves only DEP on MEGAMOE_CUTEDSL and refuses to build when any MoE layer's experts are not its arm's
exact MegaMoE CuteDSL identity: the resolver serves a layer every family member rejects with CutlassFusedMoE, and a
MegaMoE local layer that declines resolves to the stock member, so that check keeps each arm on its own kernel.
CPU only.
"""

import types

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import routing
from tensorrt_llm.mapping import Mapping

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        engagement,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

_FUSED = "fused_comm"
_DENSE = 3
_MOE_LAYERS = 57


class _Constants(types.SimpleNamespace):
    pass


class _SwigluBias:
    def constants(self):
        return _Constants(alpha=1.702, beta=1.0, limit=7.0)


class _Mega:
    descriptor = types.SimpleNamespace(identity="trtllm.cutedsl.mega_moe.nvfp4")
    scheduler_kind = _FUSED

    def __init__(self):
        self.bias = False
        self.activation = _SwigluBias()
        self.combine_format = "bf16"


class _Cutlass:
    descriptor = types.SimpleNamespace(identity="trtllm.cutlass.grouped_gemm.nvfp4")
    scheduler_kind = "separated_comm"

    def __init__(self):
        self.bias = False
        self.activation = _SwigluBias()


class _DeepgemmMega(_Mega):
    descriptor = types.SimpleNamespace(identity="deepgemm.cuda.mega_moe.w4a8_mxfp4_mxfp8")


class _Undeclared(_Mega):
    descriptor = None


class _MegaLocal(_Mega):
    descriptor = types.SimpleNamespace(identity="trtllm.cutedsl.mega_moe_local.nvfp4")


def _layers(backends):
    """Three dense layers, then one MoE layer per backend, each wrapping its backend like the configurable MoE."""
    dense = [types.SimpleNamespace(mlp=object()) for _ in range(_DENSE)]
    moe = [
        types.SimpleNamespace(
            block_sparse_moe=types.SimpleNamespace(experts=types.SimpleNamespace(backend=b))
        )
        for b in backends
    ]
    return dense + moe


def _moe_ids(n):
    return list(range(_DENSE, _DENSE + n))


def _engage(backends, moe_layer_ids=None, local=False):
    ids = _moe_ids(len(backends)) if moe_layer_ids is None else moe_layer_ids
    return engagement.moe_engagement(
        _layers(backends), ids, _FUSED, engagement.ARM_IDENTITIES[local]
    )


_DISABLE = "TRTLLM_M31_DISABLE_MEGAMOE_LOCAL"


@pytest.fixture
def megamoe_local(monkeypatch):
    """Request the MegaMoE local arm (disable switch cleared) or the stock arm (TRTLLM_M31_DISABLE_MEGAMOE_LOCAL=1)
    for one case; the stock arm is the starting state."""

    def set_(enabled):
        if enabled:
            monkeypatch.delenv(_DISABLE, raising=False)
        else:
            monkeypatch.setenv(_DISABLE, "1")

    set_(False)
    return set_


def _dep(ep=8, **extra):
    return Mapping(
        world_size=ep, tp_size=ep, moe_tp_size=1, moe_ep_size=ep, enable_attention_dp=True, **extra
    )


def test_every_layer_on_megamoe_engages():
    identities = _engage([_Mega() for _ in range(_MOE_LAYERS)])
    assert len(identities) == _MOE_LAYERS
    assert {i.impl for i in identities} == {"_Mega"}
    assert {i.bias for i in identities} == {False}


def test_a_layer_resolved_to_the_cutlass_fallback_raises():
    backends = [_Mega() for _ in range(_MOE_LAYERS)]
    backends[10] = _Cutlass()
    with pytest.raises(
        RuntimeError, match=r"layer 13: _Cutlass \(identity trtllm\.cutlass\.grouped_gemm\.nvfp4"
    ):
        _engage(backends)


def test_the_cutlass_identity_is_refused_even_with_fused_comm():
    backend = _Cutlass()
    backend.scheduler_kind = _FUSED
    with pytest.raises(RuntimeError, match="layer 3: _Cutlass"):
        _engage([backend])


def test_another_mega_moe_identity_is_refused():
    with pytest.raises(RuntimeError, match="layer 3: _DeepgemmMega"):
        _engage([_DeepgemmMega()])


def test_an_implementation_without_a_descriptor_is_refused():
    with pytest.raises(RuntimeError, match=r"layer 3: _Undeclared \(identity None"):
        _engage([_Undeclared()])


def test_the_pinned_identity_without_fused_comm_raises():
    backend = _Mega()
    backend.scheduler_kind = "separated_comm"
    with pytest.raises(RuntimeError, match="layer 3: _Mega"):
        _engage([backend])


def test_each_arm_pins_its_exact_identity():
    assert engagement.ARM_IDENTITIES == {
        False: "trtllm.cutedsl.mega_moe.nvfp4",
        True: "trtllm.cutedsl.mega_moe_local.nvfp4",
    }


def test_the_stock_arm_is_the_one_with_megamoe_local_disabled(megamoe_local):
    assert engagement.arm_identity() == "trtllm.cutedsl.mega_moe.nvfp4"
    megamoe_local(True)
    assert engagement.arm_identity() == "trtllm.cutedsl.mega_moe_local.nvfp4"


@pytest.mark.parametrize(
    "environ, local",
    [
        ({}, True),
        ({_DISABLE: "0"}, True),
        ({_DISABLE: "1"}, False),
        ({_DISABLE: "true"}, True),
        ({_DISABLE: "on"}, True),
        ({_DISABLE: " 1"}, True),
    ],
)
def test_the_arm_follows_the_megamoe_backend_rule(environ, local):
    """On unless DISABLE is exactly "1", as the MegaMoE backend reads it: only that value builds the stock kernel, so
    the target must require the stock identity for it and the local identity otherwise."""
    assert engagement.megamoe_local_requested(environ) is local
    assert engagement.arm_identity(environ) == engagement.ARM_IDENTITIES[local]


def test_disable_builds_the_stock_arm(monkeypatch):
    """The MegaMoE local kill: with the disable switch set the backend builds stock MegaMoE, which the target
    engages."""
    monkeypatch.setenv(_DISABLE, "1")
    assert engagement.arm_identity() == "trtllm.cutedsl.mega_moe.nvfp4"
    identities = engagement.moe_engagement(
        _layers([_Mega() for _ in range(_MOE_LAYERS)]),
        _moe_ids(_MOE_LAYERS),
        _FUSED,
        engagement.arm_identity(),
    )
    assert {i.identity for i in identities} == {"trtllm.cutedsl.mega_moe.nvfp4"}


def test_the_megamoe_local_switch_is_read_at_construction(monkeypatch):
    monkeypatch.delenv(_DISABLE, raising=False)
    assert engagement.arm_identity() == "trtllm.cutedsl.mega_moe_local.nvfp4"
    monkeypatch.setenv(_DISABLE, "1")
    assert engagement.arm_identity() == "trtllm.cutedsl.mega_moe.nvfp4"
    monkeypatch.delenv(_DISABLE)
    assert engagement.arm_identity() == "trtllm.cutedsl.mega_moe_local.nvfp4"


def test_the_stock_arm_refuses_the_local_identity():
    with pytest.raises(
        RuntimeError,
        match=r"MegaMoE local not requested.*layer 3: _MegaLocal "
        r"\(identity trtllm\.cutedsl\.mega_moe_local\.nvfp4",
    ):
        _engage([_MegaLocal()] + [_Mega() for _ in range(_MOE_LAYERS - 1)])


def test_the_local_arm_engages_the_local_identity():
    identities = _engage([_MegaLocal() for _ in range(_MOE_LAYERS)], local=True)
    assert {i.identity for i in identities} == {"trtllm.cutedsl.mega_moe_local.nvfp4"}


def test_the_local_arm_refuses_a_layer_that_fell_back_to_stock():
    backends = [_MegaLocal() for _ in range(_MOE_LAYERS)]
    backends[10] = _Mega()
    with pytest.raises(
        RuntimeError,
        match=r"MegaMoE local requested.*layer 13: _Mega "
        r"\(identity trtllm\.cutedsl\.mega_moe\.nvfp4",
    ):
        _engage(backends, local=True)


def test_a_config_without_moe_layers_raises():
    with pytest.raises(RuntimeError, match="no MoE layer in the config"):
        _engage([_Mega()], moe_layer_ids=[])


def test_a_config_moe_layer_without_a_moe_raises():
    with pytest.raises(RuntimeError, match="layer 2: no MoE on a layer the config makes MoE"):
        _engage([_Mega() for _ in range(_MOE_LAYERS)], moe_layer_ids=[2] + _moe_ids(_MOE_LAYERS))


def test_a_moe_on_a_config_dense_layer_raises():
    with pytest.raises(RuntimeError, match="layer 3: a MoE on a layer the config makes dense"):
        _engage([_Mega() for _ in range(_MOE_LAYERS)], moe_layer_ids=_moe_ids(_MOE_LAYERS)[1:])


def test_a_config_moe_layer_that_was_not_built_raises():
    ids = _moe_ids(_MOE_LAYERS) + [_DENSE + _MOE_LAYERS]
    with pytest.raises(RuntimeError, match=r"layer 60: not built \(60 layers\)"):
        _engage([_Mega() for _ in range(_MOE_LAYERS)], moe_layer_ids=ids)


def test_experts_without_a_backend_wrapper_are_checked_directly():
    layer = types.SimpleNamespace(block_sparse_moe=types.SimpleNamespace(experts=_Mega()))
    assert (
        len(engagement.moe_engagement([layer], [0], _FUSED, engagement.ARM_IDENTITIES[False])) == 1
    )


def test_identity_lines_group_uniform_layers_into_one_line():
    identities = _engage([_Mega() for _ in range(_MOE_LAYERS)])
    (line,) = engagement.identity_lines(3, 8, identities, "K")
    assert line.startswith(
        "[M3.1 DEP] rank 3/8 MoE trtllm.cutedsl.mega_moe.nvfp4 x57 impl=_Mega kernel=K "
    )
    assert "bias=False" in line
    assert "act=_SwigluBias(alpha=1.702, beta=1.0, limit=7.0)" in line
    assert line.endswith("combine=bf16")


@pytest.mark.parametrize("ep", [4, 8, 16])
def test_dep_on_megamoe_is_served(ep):
    assert engagement.deployment_mismatch(_dep(ep), "MEGAMOE_CUTEDSL") is None
    assert engagement.deployment_mismatch(_dep(ep), "megamoe_cutedsl") is None


def test_another_moe_backend_is_refused():
    assert "moe_backend TRTLLM" in engagement.deployment_mismatch(_dep(), "TRTLLM")


def test_a_vocab_sharded_head_is_refused():
    why = engagement.deployment_mismatch(_dep(enable_lm_head_tp_in_adp=True), "MEGAMOE_CUTEDSL")
    assert "enable_lm_head_tp_in_adp" in why


def test_a_topology_other_than_dep_is_refused():
    tp8 = Mapping(world_size=8, tp_size=8, moe_tp_size=8, moe_ep_size=1)
    assert "is not DEP" in engagement.deployment_mismatch(tp8, "MEGAMOE_CUTEDSL")


@pytest.mark.parametrize("ep", [4, 8, 16])
def test_the_route_and_the_target_share_the_topology_test(ep):
    assert routing._dep(_dep(ep)) is engagement.is_dep(_dep(ep)) is True
    tp8 = Mapping(world_size=8, tp_size=8, moe_tp_size=8, moe_ep_size=1)
    assert routing._dep(tp8) is engagement.is_dep(tp8) is False


def test_the_resolver_tables_pin_the_megamoe_identity():
    from tensorrt_llm._torch.moe.fused_moe import moe_resolution as resolution
    from tensorrt_llm._torch.moe.fused_moe.impl_identity import MOE_IMPL_REGISTRY, MoEImplId

    pinned = MOE_IMPL_REGISTRY.lookup(MoEImplId.parse(engagement.ARM_IDENTITIES[False]))
    assert pinned is resolution.TrtllmCutedslMegaMoeNvfp4Impl
    assert pinned in resolution.BACKEND_FAMILY[engagement.MOE_BACKEND]
    assert engagement.MOE_BACKEND not in resolution.NO_FALLBACK_BACKENDS


def test_the_resolver_tables_megamoe_engages_as_the_stock_arm(megamoe_local):
    from tensorrt_llm._torch.moe.fused_moe import moe_resolution as resolution
    from tensorrt_llm._torch.moe.fused_moe.interface import MoESchedulerKind

    impl = resolution.TrtllmCutedslMegaMoeNvfp4Impl
    backends = [object.__new__(impl) for _ in range(_MOE_LAYERS)]
    identities = engagement.moe_engagement(
        _layers(backends),
        _moe_ids(_MOE_LAYERS),
        MoESchedulerKind.FUSED_COMM,
        engagement.arm_identity(),
    )
    assert len(identities) == _MOE_LAYERS
    assert {(i.impl, i.identity) for i in identities} == {
        (impl.__name__, "trtllm.cutedsl.mega_moe.nvfp4")
    }


def test_the_builtin_config_reader_names_the_moe_layers():
    from tensorrt_llm._torch.models.modeling_minimaxm3 import get_moe_layer_ids

    config = types.SimpleNamespace(
        num_hidden_layers=_DENSE + _MOE_LAYERS, moe_layer_freq=[0] * _DENSE + [1] * _MOE_LAYERS
    )
    dense, moe = get_moe_layer_ids(config)
    assert dense == list(range(_DENSE))
    assert moe == _moe_ids(_MOE_LAYERS)
    assert len(_engage([_Mega() for _ in range(_MOE_LAYERS)], moe_layer_ids=moe)) == _MOE_LAYERS
