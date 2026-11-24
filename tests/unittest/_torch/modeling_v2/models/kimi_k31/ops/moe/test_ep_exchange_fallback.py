# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU test of the Kimi K3.1 MoE exchange choice (``create_ep_all_to_all``): where building the one-sided all-to-all
raises what an unsupported platform raises, the all-gather exchange serves the layer; any other error propagates."""

import pynvml
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import (
    ep_dispatch_combine as exchange,
)
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


def _dep4(rank: int = 1) -> Mapping:
    return Mapping(
        world_size=4, rank=rank, tp_size=4, moe_tp_size=1, moe_ep_size=4, enable_attention_dp=True
    )


@pytest.fixture
def stubs(monkeypatch):
    """The exchange classes replaced by stand-ins: the one-sided build raises ``state["error"]``."""
    state = {"error": None, "gathers": 0, "registered": 0}

    class _OneSided:
        def __init__(self, *args, **kwargs):
            raise state["error"]("one-sided all-to-all unavailable")

    class _AllGather:
        def __init__(self, *args):
            state["gathers"] += 1

    def register(proxy):
        state["registered"] += 1

    monkeypatch.setattr(exchange, "K31EpAllToAll", _OneSided)
    monkeypatch.setattr(exchange, "K31EpAllGather", _AllGather)
    monkeypatch.setattr(exchange, "register_moe_comm_timeout_proxy", register)
    monkeypatch.delenv("TRTLLM_FORCE_COMM_METHOD", raising=False)
    for name in (exchange.CFT_MAX_BATCH_FOR_DISPATCH_ENV, exchange.CFT_MAX_BATCH_FOR_COMBINE_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(exchange.FORCE_CFT_ENV, raising=False)
    state["all_gather"] = _AllGather
    return state


def _build():
    return exchange.create_ep_all_to_all(_dep4(), 896, 16, 256, 3584, torch.bfloat16)


class _NvmlError(pynvml.NVMLError):
    """An NVML query failure (the NVLink probe), without the NVML library behind it."""

    def __new__(cls, *args):
        return Exception.__new__(cls)

    def __init__(self, *args):
        Exception.__init__(self, *args)
        self.value = 999


@pytest.mark.parametrize(
    "error",
    [RuntimeError, ValueError, AssertionError, OSError, TimeoutError, _NvmlError],
    ids=lambda e: e.__name__,
)
def test_one_sided_build_failure_takes_the_all_gather(stubs, error) -> None:
    stubs["error"] = error
    assert isinstance(_build(), stubs["all_gather"])
    assert stubs["gathers"] == 1
    # The timeout budgets are read before any exchange is tried, as the engine factory reads them.
    assert stubs["registered"] == 1


@pytest.mark.parametrize("error", [TypeError, KeyError], ids=lambda e: e.__name__)
def test_other_errors_propagate(stubs, error) -> None:
    stubs["error"] = error
    with pytest.raises(error):
        _build()
    assert stubs["gathers"] == 0


def test_no_exchange_without_attention_dp(stubs) -> None:
    mapping = Mapping(world_size=4, rank=0, tp_size=4, moe_tp_size=1, moe_ep_size=4)
    assert exchange.create_ep_all_to_all(mapping, 896, 16, 256, 3584, torch.bfloat16) is None


@pytest.mark.parametrize("value", ["many", "-1"])
def test_a_bad_counted_writes_threshold_raises(stubs, monkeypatch, value) -> None:
    """A malformed counted-writes threshold is a configuration error, not a platform the all-to-all cannot serve: it
    raises before any exchange is built."""
    stubs["error"] = RuntimeError
    monkeypatch.setenv(exchange.CFT_MAX_BATCH_FOR_COMBINE_ENV, value)
    with pytest.raises(ValueError, match=exchange.CFT_MAX_BATCH_FOR_COMBINE_ENV):
        _build()
    assert stubs["gathers"] == 0
