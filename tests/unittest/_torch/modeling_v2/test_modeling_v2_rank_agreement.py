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


@pytest.fixture
def ranks(monkeypatch):
    """``ranks(mode, mine, others, world_size=2)``: resolve with this rank's own
    decision ``mine`` (a target, None or an exception) and the other ranks'
    outcomes ``others``; returns the result and the number of exchanges."""
    calls = []

    def resolve_on_all_ranks(mode, mine, others, world_size=2):
        def resolve(config, llm_args):
            if isinstance(mine, Exception):
                raise mine
            return mine

        def allgather(outcome):
            calls.append(outcome)
            return [outcome, *others]

        monkeypatch.setattr(_router_index, "modeling_v2_resolve", resolve)
        monkeypatch.setattr(
            communicator.Distributed,
            "get",
            staticmethod(lambda mapping: SimpleNamespace(allgather=allgather)),
        )
        calls.clear()
        config = SimpleNamespace(mapping=SimpleNamespace(rank=0, world_size=world_size))
        return modeling_v2_resolve_on_all_ranks(config, SimpleNamespace(modeling_v2=mode))

    resolve_on_all_ranks.calls = calls
    return resolve_on_all_ranks


def test_off_and_one_rank_exchange_nothing(ranks):
    assert ranks("off", AssertionError("routing ran under off"), []) is None
    assert ranks.calls == []
    assert ranks("auto", Target, [], world_size=1) is Target
    assert ranks.calls == []


def test_ranks_build_the_same_code_or_all_fail(ranks):
    assert ranks("require", Target, [(_TARGET, None)]) is Target
    assert ranks("auto", None, [(None, None)]) is None
    with pytest.raises(RuntimeError, match="ranks disagree.*rank 1: built-in"):
        ranks("auto", Target, [(None, None)])
    with pytest.raises(RuntimeError, match="rank 1: NativeLoadError"):
        ranks("auto", Target, [(None, "NativeLoadError: libth_modeling_v2_x.so cannot be loaded")])
    # This rank's own error is raised after the exchange, not before it, which
    # would leave the other ranks waiting in it.
    with pytest.raises(ValueError, match="no target fits"):
        ranks("require", ValueError("modeling_v2='require', but no target fits"), [(_TARGET, None)])
    assert len(ranks.calls) == 1
