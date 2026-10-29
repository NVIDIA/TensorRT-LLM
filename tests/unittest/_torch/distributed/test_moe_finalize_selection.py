# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for the measured MoE finalize profiles and their fallback."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import tensorrt_llm._torch.distributed.ops as ops
from tensorrt_llm._torch.autotuner import PostAutotuneHook

pytestmark = pytest.mark.cpu_only

_MEASURED_IMPL_ID = "flashinfer.trtllm_gen.fused_moe.nvfp4"


def _impl(impl_id: str) -> object:
    return type("_Impl", (), {"descriptor": SimpleNamespace(impl_id=impl_id)})()


class _Experts(torch.nn.Module):
    """Measured MoE geometry; weights are not created yet."""

    def __init__(self, impl_id: str = _MEASURED_IMPL_ID) -> None:
        super().__init__()
        self.backend = _impl(impl_id)
        self.hidden_size = 4096
        self.num_experts = 128
        self.routing_method = SimpleNamespace(experts_per_token=8)
        self.intermediate_size = 1536
        self.dtype = torch.bfloat16

    @property
    def has_nvfp4(self) -> bool:
        raise AssertionError("Selection must not inspect weights before loading")


@pytest.fixture
def setup(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ops, "get_sm_version", lambda: 100)
    monkeypatch.setattr(ops, "get_allreduce_workspace", lambda mapping: None)
    monkeypatch.delenv("TLLM_DISABLE_ALLREDUCE_AUTOTUNE", raising=False)
    mapping = SimpleNamespace(
        tp_size=4, moe_ep_size=4, tp_group=[0, 1, 2, 3], is_multi_node=lambda: False
    )
    allreduce = SimpleNamespace(
        strategy=ops.AllReduceStrategy.ONESHOT,
        mapping=mapping,
        workspace=torch.empty(8),
        _disable_mpi=False,
        mnnvl_allreduce=None,
    )
    return ops.MoEAllReduce(mapping), _Experts(), allreduce


def test_default_keeps_capacity_for_every_caller(setup) -> None:
    moe_allreduce, _, _ = setup
    assert moe_allreduce.finalize_max_tokens == moe_allreduce.max_token == 128
    assert isinstance(moe_allreduce, PostAutotuneHook)
    moe_allreduce.resolve_post_autotune()
    assert moe_allreduce.finalize_max_tokens == 128


def test_measured_profile_lowers_cutoff(setup) -> None:
    moe_allreduce, experts, allreduce = setup
    assert moe_allreduce.select_finalize_max_tokens(experts, allreduce) == 47
    assert moe_allreduce.finalize_max_tokens == 47
    assert moe_allreduce.max_token == 128
    # Selection holds the modules without registering them as submodules.
    assert list(moe_allreduce.children()) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hidden_size", 2048),
        ("num_experts", 64),
        ("intermediate_size", 768),
        ("dtype", torch.float16),
    ],
)
def test_unmeasured_model_geometry_keeps_capacity(setup, field: str, value: object) -> None:
    moe_allreduce, experts, allreduce = setup
    setattr(experts, field, value)
    assert moe_allreduce.select_finalize_max_tokens(experts, allreduce) == 128


@pytest.mark.parametrize(
    "case",
    ["sm", "tp", "ep", "top_k", "ar", "nodes", "provider", "no_impl", "no_allreduce"],
)
def test_unmeasured_execution_geometry_keeps_capacity(setup, monkeypatch, case: str) -> None:
    moe_allreduce, experts, allreduce = setup
    if case == "sm":
        monkeypatch.setattr(ops, "get_sm_version", lambda: 103)
    elif case == "tp":
        moe_allreduce.mapping.tp_size = 2
    elif case == "ep":
        moe_allreduce.mapping.moe_ep_size = 1
    elif case == "top_k":
        experts.routing_method.experts_per_token = 4
    elif case == "ar":
        allreduce.strategy = ops.AllReduceStrategy.NCCL
    elif case == "nodes":
        moe_allreduce.mapping.is_multi_node = lambda: True
    elif case == "provider":
        experts.backend = _impl("trtllm.trtllm_gen.fused_moe.nvfp4")
    elif case == "no_impl":
        experts = torch.nn.Identity()
    elif case == "no_allreduce":
        allreduce = None
    assert moe_allreduce.select_finalize_max_tokens(experts, allreduce) == 128


def test_measured_impl_id_matches_the_registered_leaf() -> None:
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.flashinfer_nvfp4 import (
        FlashinferTrtllmGenNvfp4Impl,
    )

    assert FlashinferTrtllmGenNvfp4Impl.descriptor.impl_id == _MEASURED_IMPL_ID


@pytest.mark.parametrize("capacity", [0, 16, 47, 48, 128])
def test_profile_cannot_exceed_capacity(setup, capacity: int) -> None:
    moe_allreduce, experts, allreduce = setup
    moe_allreduce.max_token = capacity
    assert moe_allreduce.select_finalize_max_tokens(experts, allreduce) == min(capacity, 47)


def test_another_registered_profile_uses_its_own_cutoff(setup, monkeypatch) -> None:
    moe_allreduce, experts, allreduce = setup
    experts.hidden_size = 2048
    experts.intermediate_size = 768
    profile = ops.MoEFinalizeProfile(
        sm_version=100,
        tp_size=4,
        moe_ep_size=4,
        hidden_size=2048,
        num_experts=128,
        top_k=8,
        intermediate_size=768,
        activation_dtype=torch.bfloat16,
        allreduce_strategy=ops.AllReduceStrategy.ONESHOT,
        moe_impl_id=_MEASURED_IMPL_ID,
    )
    # A test-only entry checks that lookup is by profile; it is not measurement evidence.
    monkeypatch.setitem(ops._MOE_FINALIZE_MAX_TOKENS, profile, 63)
    assert moe_allreduce.select_finalize_max_tokens(experts, allreduce) == 63


def _mock_tuner(monkeypatch: pytest.MonkeyPatch, cache: object) -> None:
    from tensorrt_llm._torch.autotuner import AutoTuner

    monkeypatch.setattr(AutoTuner, "get", lambda: SimpleNamespace(profiling_cache=cache))


@pytest.mark.parametrize("tactic", [ops.AllReduceStrategy.ONESHOT, ops.AllReduceStrategy.TWOSHOT])
def test_auto_uses_tuned_strategy_without_changing_policy(
    setup, monkeypatch: pytest.MonkeyPatch, tactic: ops.AllReduceStrategy
) -> None:
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import AllReduceRunner

    moe_allreduce, experts, allreduce = setup
    allreduce.strategy = ops.AllReduceStrategy.AUTO
    cache = Mock()
    cache.search_cache.return_value = (True, 0, tactic, 0.0)
    _mock_tuner(monkeypatch, cache)
    monkeypatch.setattr(
        AllReduceRunner,
        "tuning_config",
        SimpleNamespace(
            dynamic_tensor_specs=[
                SimpleNamespace(gen_tuning_buckets=[1, 2, 4, 8, 16, 32, 64, 128, 256])
            ]
        ),
    )
    # Before tuning, AUTO is not a measured strategy.
    assert moe_allreduce.select_finalize_max_tokens(experts, allreduce) == 128
    moe_allreduce.resolve_post_autotune()
    expected = 47 if tactic == ops.AllReduceStrategy.ONESHOT else 128
    assert moe_allreduce.finalize_max_tokens == expected
    assert allreduce.strategy == ops.AllReduceStrategy.AUTO
    searched = [call.args[2][0][0] for call in cache.search_cache.call_args_list]
    assert searched == [1, 2, 4, 8, 16, 32, 64, 128]


@pytest.mark.parametrize("case", ["missing", "mixed", "disabled", "process_group", "mnnvl"])
def test_unresolved_auto_keeps_capacity(setup, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    moe_allreduce, experts, allreduce = setup
    allreduce.strategy = ops.AllReduceStrategy.AUTO
    allreduce._disable_mpi = case == "process_group"
    allreduce.mnnvl_allreduce = object() if case == "mnnvl" else None
    cache = Mock()
    cache.search_cache.side_effect = [
        (case != "missing", 0, ops.AllReduceStrategy.ONESHOT, 0.0),
        *[(True, 0, ops.AllReduceStrategy.TWOSHOT, 0.0)] * 32,
    ]
    _mock_tuner(monkeypatch, cache)
    monkeypatch.setenv("TLLM_DISABLE_ALLREDUCE_AUTOTUNE", "1" if case == "disabled" else "0")
    moe_allreduce.select_finalize_max_tokens(experts, allreduce)
    moe_allreduce.resolve_post_autotune()
    assert moe_allreduce.finalize_max_tokens == 128


@pytest.mark.parametrize("mismatch", [None, "group", "dtype", "fusion"])
def test_auto_reads_the_real_tactic_cache_key(
    setup, monkeypatch: pytest.MonkeyPatch, mismatch: str | None
) -> None:
    from tensorrt_llm._torch.autotuner import AutoTuner, AutoTunerProfilingCache
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import AllReduceRunner

    moe_allreduce, experts, allreduce = setup
    allreduce.strategy = ops.AllReduceStrategy.AUTO
    cache = AutoTunerProfilingCache.__new__(AutoTunerProfilingCache)
    cache.cache = {}
    tuner = SimpleNamespace(
        profiling_cache=cache, _find_nearest_profile=AutoTuner._find_nearest_profile
    )
    monkeypatch.setattr(AutoTuner, "get", lambda: tuner)
    runner = AllReduceRunner(
        4,
        [4, 5, 6, 7] if mismatch == "group" else allreduce.mapping.tp_group,
        torch.float16 if mismatch == "dtype" else torch.bfloat16,
        int(
            ops.AllReduceFusionOp.NONE
            if mismatch == "fusion"
            else ops.AllReduceFusionOp.RESIDUAL_RMS_NORM
        ),
        0.0,
        True,
    )
    for tokens in runner.tuning_config.dynamic_tensor_specs[0].gen_tuning_buckets:
        shape = torch.Size((tokens, 4096))
        shapes = (
            shape,
            shape,
            torch.Size((4096,)),
            torch.Size((0,)),
            torch.Size((0,)),
            torch.Size((8,)),
        )
        key = cache.get_cache_key(
            "trtllm::tunable_allreduce::allreduce", runner, shapes, runner.tuning_config
        )
        cache[key] = (0, ops.AllReduceStrategy.ONESHOT, 0.01)
    moe_allreduce.select_finalize_max_tokens(experts, allreduce)
    moe_allreduce.resolve_post_autotune()
    assert moe_allreduce.finalize_max_tokens == (47 if mismatch is None else 128)
