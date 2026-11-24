# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The vocab-parallel pair exchange stays off the one-shot all-reduce workspace.

``VocabParallelHead`` selects target tokens with one float32 all-reduce of
``(value, index)`` pairs. With AUTO on a TP group inside one node, that call may
run the one-shot Lamport kernel on the IPC workspace that the model's bf16
all-reduces share. The kernel records the size to clear in elements of its
caller's dtype, and the next call clears that many of its own elements, so a
bf16 call after the float32 one clears half of its bytes. A later one-shot call
can then read the stale half as a peer's data. A one-node TP2 context server
with DFlash took wrong and NaN target tokens this way.

The exchange uses AUTO only where AUTO builds the MNNVL all-reduce, whose
kernel records bytes to clear, and NCCL elsewhere.
"""

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.distributed.ops as dist_ops
import tensorrt_llm._torch.speculative.vocab_parallel as vocab_parallel
from tensorrt_llm._torch.distributed import AllReduceStrategy
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_VOCAB = 163840


def _lm_head(mapping: Mapping) -> SimpleNamespace:
    return SimpleNamespace(
        mapping=mapping, num_embeddings=_VOCAB, out_features=_VOCAB // mapping.tp_size
    )


class _RecordedAllReduce:
    """Stands in for AllReduce and records how the head builds it. AUTO gets an
    MNNVL all-reduce when ``mnnvl_builds``."""

    built = []
    mnnvl_builds = True

    def __init__(self, mapping, strategy=AllReduceStrategy.AUTO, dtype=None):
        self.mapping = mapping
        self.strategy = strategy
        self.dtype = dtype
        mnnvl = strategy == AllReduceStrategy.AUTO and _RecordedAllReduce.mnnvl_builds
        self.mnnvl_allreduce = object() if mnnvl else None
        _RecordedAllReduce.built.append(self)


@pytest.mark.parametrize(
    "mnnvl, builds, expected",
    [
        (False, True, [AllReduceStrategy.NCCL]),
        (True, True, [AllReduceStrategy.AUTO]),
        (True, False, [AllReduceStrategy.AUTO, AllReduceStrategy.NCCL]),
    ],
    ids=["no-mnnvl-takes-nccl", "mnnvl-keeps-auto", "mnnvl-not-built-takes-nccl"],
)
@pytest.mark.parametrize("tp_size", [2, 8])
def test_exchange_strategy_follows_mnnvl(monkeypatch, tp_size, mnnvl, builds, expected):
    queries = []

    def _is_mnnvl(mapping, dtype, explicitly_requested=False):
        queries.append((mapping, dtype, explicitly_requested))
        return mnnvl

    monkeypatch.setattr(_RecordedAllReduce, "built", [])
    monkeypatch.setattr(_RecordedAllReduce, "mnnvl_builds", builds)
    monkeypatch.setattr(vocab_parallel, "AllReduce", _RecordedAllReduce)
    monkeypatch.setattr(vocab_parallel.MNNVLAllReduce, "is_mnnvl", staticmethod(_is_mnnvl))
    mapping = Mapping(world_size=tp_size, rank=1, gpus_per_node=4, tp_size=tp_size)

    head = vocab_parallel.VocabParallelHead(_lm_head(mapping))

    assert queries == [(mapping, torch.float32, False)]
    built = _RecordedAllReduce.built
    assert [all_reduce.strategy for all_reduce in built] == expected
    assert all(a.mapping is mapping and a.dtype == torch.float32 for a in built)
    assert head._exchange.all_reduce is built[-1]
    assert (head._exchange.tp_size, head._exchange.tp_rank) == (tp_size, 1)


def test_one_node_exchange_takes_no_lamport_workspace(monkeypatch):
    """The real AllReduce of a one-node TP2 exchange gets no IPC or MNNVL workspace."""

    def _no_workspace(*args, **kwargs):
        raise AssertionError("the pair exchange must not take the one-shot workspace")

    monkeypatch.setattr(dist_ops, "get_allreduce_workspace", _no_workspace)
    monkeypatch.setattr(
        dist_ops.MNNVLAllReduce, "is_mnnvl", staticmethod(lambda *args, **kwargs: False)
    )
    mapping = Mapping(world_size=2, rank=0, gpus_per_node=4, tp_size=2)

    head = vocab_parallel.VocabParallelHead(_lm_head(mapping))

    all_reduce = head._exchange.all_reduce
    assert isinstance(all_reduce, dist_ops.AllReduce)
    assert all_reduce.strategy == AllReduceStrategy.NCCL
    assert all_reduce.workspace is None
    assert all_reduce.mnnvl_allreduce is None
