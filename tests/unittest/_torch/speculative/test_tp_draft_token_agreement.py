# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ranks of a request-sharing TP group propose the same draft tokens.

Under TP without attention DP every rank runs the drafter on the same requests
and picks the drafts itself. The next step accepts a draft only where it equals
the target token, which every rank selects identically (the vocab-parallel pair
exchange). If one rank's draft differs in one position, only that rank's
accepted count changes: the ranks finish the request a step apart, one runs a
forward alone, and both wait in a TP collective forever. A worker that sets
``agree_draft_tokens_across_tp`` has ``SpecWorkerBase.forward`` replace the
drafts with those of the group's rank 0 (one all-gather), so every rank feeds
and accepts the same tokens.

The ranks run in threads here, and a fake all-gather concatenates their inputs
in rank order, as the TP all-gather does.
"""

import threading
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.distributed.ops as dist_ops
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm._torch.speculative.dspark import DSparkWorker
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm.mapping import CpType, Mapping

pytestmark = pytest.mark.cpu_only

_K = 7

# One verify step: the target tokens are the same on both ranks, and rank 1's
# first draft differs from rank 0's.
_TARGET = torch.tensor([[11, 40, 41, 42, 43, 44, 45, 46]], dtype=torch.int32)
_DRAFTS = {
    0: torch.tensor([[11, 90, 91, 92, 93, 94, 95]], dtype=torch.int32),
    1: torch.tensor([[12, 90, 91, 92, 93, 94, 95]], dtype=torch.int32),
}


class _Group:
    """An all-gather over the ranks of one group, each rank on its own thread."""

    def __init__(self, size: int):
        self.size = size
        self.calls = []
        self._inputs = [None] * size
        self._barrier = threading.Barrier(size, timeout=10)

    def allgather(self, input, mapping, dim=-1, sizes=None):
        assert sizes is None and dim == 0
        assert mapping.tp_size == self.size
        self.calls.append((mapping.tp_rank, tuple(input.shape)))
        self._inputs[mapping.tp_rank] = input.clone()
        self._barrier.wait()
        out = torch.cat(self._inputs, dim=0)
        self._barrier.wait()
        return out


def _no_collective(*args, **kwargs):
    raise AssertionError("no collective expected")


class _StubWorker(SpecWorkerBase):
    """A worker whose forward returns this rank's drafts; the base forward does the rest."""

    agree_draft_tokens_across_tp = True

    def __init__(self, mapping, drafts: torch.Tensor):
        super().__init__()
        self.mapping = mapping
        self._drafts = drafts

    @property
    def max_draft_len(self) -> int:
        return _K

    def _forward_impl(self, *args, **kwargs):
        rows = self._drafts.shape[0]
        last = torch.full((rows, 1), 7, dtype=torch.int32)
        return {
            "new_tokens": _TARGET.clone(),
            "new_tokens_lens": torch.ones(rows, dtype=torch.int32),
            "next_draft_tokens": self._drafts.clone(),
            "next_new_tokens": torch.cat([last, self._drafts], dim=1),
        }

    def _ensure_spec_dec_state_restored(self, attn_metadata, spec_metadata):
        pass


class _NoAgreementWorker(_StubWorker):
    agree_draft_tokens_across_tp = False


def _run_ranks(make_worker, size: int) -> list[dict]:
    """``forward`` of one worker per rank, the ranks concurrently."""
    outputs = [None] * size
    errors = []

    def run(rank):
        try:
            outputs[rank] = make_worker(rank).forward()
        except Exception as e:  # surfaced below
            errors.append(e)

    threads = [threading.Thread(target=run, args=(r,)) for r in range(size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    if errors:
        raise errors[0]
    return outputs


def test_tp_ranks_take_rank0_drafts(monkeypatch):
    group = _Group(2)
    monkeypatch.setattr(dist_ops, "allgather", group.allgather)

    outputs = _run_ranks(
        lambda r: _StubWorker(Mapping(world_size=2, rank=r, tp_size=2), _DRAFTS[r]), 2
    )

    assert sorted(group.calls) == [(0, (1, _K)), (1, (1, _K))]
    for out in outputs:
        assert torch.equal(out["next_draft_tokens"], _DRAFTS[0])
        assert torch.equal(out["next_new_tokens"][:, 1:], _DRAFTS[0])
        assert out["next_new_tokens"][:, 0].tolist() == [7]
        assert out["next_new_tokens"].dtype == torch.int32
        # The accepted tokens of this step are left as they are.
        assert torch.equal(out["new_tokens"], _TARGET)
        assert out["new_tokens_lens"].tolist() == [1]


def test_helix_cp_ranks_agree_over_the_folded_group(monkeypatch):
    group = _Group(2)
    monkeypatch.setattr(dist_ops, "allgather", group.allgather)

    def helix(rank):
        return Mapping(
            world_size=2, rank=rank, tp_size=1, cp_size=2, cp_config={"cp_type": CpType.HELIX}
        )

    outputs = _run_ranks(lambda r: _StubWorker(helix(r), _DRAFTS[r]), 2)

    assert sorted(group.calls) == [(0, (1, _K)), (1, (1, _K))]
    for out in outputs:
        assert torch.equal(out["next_draft_tokens"], _DRAFTS[0])


@pytest.mark.parametrize(
    "mapping",
    [
        Mapping(world_size=2, rank=1, tp_size=2, enable_attention_dp=True),
        Mapping(world_size=1, rank=0, tp_size=1),
        None,
    ],
    ids=["attention_dp", "one_rank", "no_mapping"],
)
def test_ranks_with_their_own_requests_keep_their_drafts(monkeypatch, mapping):
    monkeypatch.setattr(dist_ops, "allgather", _no_collective)

    out = _StubWorker(mapping, _DRAFTS[1]).forward()

    assert torch.equal(out["next_draft_tokens"], _DRAFTS[1])
    assert torch.equal(out["next_new_tokens"][:, 1:], _DRAFTS[1])


def test_a_worker_that_does_not_ask_keeps_its_drafts(monkeypatch):
    monkeypatch.setattr(dist_ops, "allgather", _no_collective)

    out = _NoAgreementWorker(Mapping(world_size=2, rank=1, tp_size=2), _DRAFTS[1]).forward()

    assert torch.equal(out["next_draft_tokens"], _DRAFTS[1])


def test_no_drafts_take_no_collective(monkeypatch):
    monkeypatch.setattr(dist_ops, "allgather", _no_collective)
    worker = _StubWorker(Mapping(world_size=2, rank=0, tp_size=2), _DRAFTS[0])
    outputs = {
        "next_draft_tokens": torch.zeros((3, 0), dtype=torch.int32),
        "next_new_tokens": torch.zeros((3, 1), dtype=torch.int32),
    }

    worker._agree_draft_tokens(outputs)

    assert outputs["next_draft_tokens"].shape == (3, 0)


def test_an_unknown_next_token_layout_raises_before_the_collective(monkeypatch):
    monkeypatch.setattr(dist_ops, "allgather", _no_collective)
    worker = _StubWorker(Mapping(world_size=2, rank=0, tp_size=2), _DRAFTS[0])
    outputs = {
        "next_draft_tokens": _DRAFTS[0].clone(),
        "next_new_tokens": torch.zeros((1, 2 * _K), dtype=torch.int32),
    }

    with pytest.raises(ValueError, match="next_new_tokens must be"):
        worker._agree_draft_tokens(outputs)


def test_dflash_workers_agree_and_the_base_does_not():
    assert SpecWorkerBase.agree_draft_tokens_across_tp is False
    assert DFlashWorker.agree_draft_tokens_across_tp is True
    assert DSparkWorker.agree_draft_tokens_across_tp is True
    # The family workers subclass the engine DFlashWorker and cannot replace forward.
    with pytest.raises(TypeError, match="must not override"):
        type("_Override", (DFlashWorker,), {"forward": lambda self: None})


def test_forward_still_restores_the_metadata_when_agreement_fails(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("collective failed")

    monkeypatch.setattr(dist_ops, "allgather", fail)
    restored = []
    worker = _StubWorker(Mapping(world_size=2, rank=0, tp_size=2), _DRAFTS[0])
    worker._ensure_spec_dec_state_restored = lambda a, s: restored.append((a, s))
    attn, spec = SimpleNamespace(), SimpleNamespace()

    with pytest.raises(RuntimeError, match="collective failed"):
        worker.forward(attn_metadata=attn, spec_metadata=spec)

    assert restored == [(attn, spec)]
