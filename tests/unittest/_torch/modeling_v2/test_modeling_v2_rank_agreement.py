# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``modeling_v2_resolve_on_all_ranks``: every rank builds the same code, or every rank fails.

Each rank routes on its own facts -- its device, whether this installation built
the target's native code -- so ranks can disagree, and a disagreement surfaces
as a hang in the first collective unless the resolver catches it. These tests
stand in for the other ranks with a fake ``Distributed`` whose ``allgather``
returns their outcomes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch._experimental.modeling_v2 import _router_index
from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    modeling_v2_resolve_on_all_ranks,
)
from tensorrt_llm._torch.distributed import communicator

pytestmark = pytest.mark.cpu_only


class Target:
    """Stands in for a target class."""


_TARGET = f"{Target.__module__}.{Target.__qualname__}"


class _FakeDist:
    def __init__(self, others):
        self.others = others
        self.calls = 0

    def allgather(self, outcome):
        self.calls += 1
        return [outcome, *self.others]


def _config(rank=0, world_size=2):
    return SimpleNamespace(mapping=SimpleNamespace(rank=rank, world_size=world_size))


def _args(mode):
    return SimpleNamespace(modeling_v2=mode)


@pytest.fixture
def dist(monkeypatch):
    fake = _FakeDist(others=[])
    monkeypatch.setattr(communicator.Distributed, "get", staticmethod(lambda mapping: fake))
    return fake


def _route_to(monkeypatch, target=None, fallback=None, error=None):
    def route(config, llm_args, mode):
        if error is not None:
            raise error
        return target, fallback

    monkeypatch.setattr(_router_index, "_route", route)


def test_off_exchanges_nothing(monkeypatch, dist):
    _route_to(monkeypatch, error=AssertionError("routing ran under off"))
    assert modeling_v2_resolve_on_all_ranks(_config(), _args("off")) is None
    assert dist.calls == 0


def test_one_rank_exchanges_nothing(monkeypatch, dist):
    _route_to(monkeypatch, target=Target)
    assert modeling_v2_resolve_on_all_ranks(_config(world_size=1), _args("auto")) is Target
    assert dist.calls == 0


def test_agreeing_ranks_build_the_target(monkeypatch, dist):
    dist.others = [(_TARGET, None, None)]
    _route_to(monkeypatch, target=Target)
    assert modeling_v2_resolve_on_all_ranks(_config(), _args("require")) is Target
    assert dist.calls == 1


def test_agreeing_fallbacks_build_the_builtin_model(monkeypatch, dist):
    dist.others = [(None, "no target fits 'X' (needs sm_107)", None)]
    _route_to(monkeypatch, fallback="no target fits 'X' (needs sm_107)")
    assert modeling_v2_resolve_on_all_ranks(_config(), _args("auto")) is None


def test_disagreeing_ranks_all_fail(monkeypatch, dist):
    dist.others = [(None, "no target fits 'X' (native code not built)", None)]
    _route_to(monkeypatch, target=Target)
    with pytest.raises(RuntimeError, match="ranks disagree"):
        modeling_v2_resolve_on_all_ranks(_config(), _args("auto"))


def test_another_ranks_error_fails_this_rank(monkeypatch, dist):
    dist.others = [(None, None, "NativeLoadError: libtrtllm_mv2_x.so cannot be loaded")]
    _route_to(monkeypatch, target=Target)
    with pytest.raises(RuntimeError, match="rank 1: NativeLoadError"):
        modeling_v2_resolve_on_all_ranks(_config(), _args("auto"))


def test_this_ranks_error_is_raised_after_the_exchange(monkeypatch, dist):
    """Raising before the exchange would leave the other ranks waiting in it."""
    dist.others = [(_TARGET, None, None)]
    _route_to(monkeypatch, error=ValueError("modeling_v2='require', but no target fits"))
    with pytest.raises(ValueError, match="no target fits"):
        modeling_v2_resolve_on_all_ranks(_config(), _args("require"))
    assert dist.calls == 1
